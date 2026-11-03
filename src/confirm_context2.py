"""Stability and Kaggle-budget checks for the accepted ``context2_v1`` model.

Retrains seeds 42, 7 and 123 on the same train matrix. Does not read the
test split and does not replace ``models/context2_v1.joblib``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from src.context import label_index_for_split, load_context_split
from src.data import file_sha256, read_labels
from src.expkit import (
    disk_free_gib,
    git_commit,
    per_class_text,
    prediction_sha,
    project_config,
    require_disk,
    restore_env,
    score_arrays,
    subject_map,
    thread_env_one,
    upsert_result,
    write_prediction_csv,
)
from src.features import (
    FEATURE_VERSION,
    _load_signal,
    _process_peaks_mb,
    acquire_lock,
    align_labels,
    features_from_signal,
    release_lock,
    worker_count,
)
from src.train import build_model

ROOT = Path(__file__).resolve().parents[1]
TEST_EPOCHS = 66222
VAL_EPOCHS = 43206


def _ordered_proba(model, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float32)
    classes = [int(value) for value in model.classes_]
    if classes == [0, 1, 2, 3, 4]:
        return raw
    ordered = np.zeros((raw.shape[0], 5), dtype=np.float32)
    for column, label in enumerate(classes):
        ordered[:, label] = raw[:, column]
    return ordered


def _fit_seed(features: np.ndarray, y: np.ndarray, seed: int, n_jobs: int):
    model, name = build_model(seed, n_jobs)
    started = time.perf_counter()
    model.fit(features, y)
    return model, name, time.perf_counter() - started


def _log_confirmation(model_id: str, model, scores: dict, fit_s: float, seed: int, digest: str) -> None:
    upsert_result(
        {
            "id": model_id,
            "hypothesis": (
                "Confirmation fit of context2_v1. Not a candidate for replacing "
                "the accepted model."
            ),
            "features": (
                "features_v1 cache; same-record epoch_index offsets -2,-1,+1,+2; "
                "n_neighbors 0..4"
            ),
            "model": (
                f"LGBMClassifier(n_estimators=200,learning_rate=0.1,num_leaves=31,"
                f"class_weight=balanced,deterministic=True,n_jobs={model.n_jobs},seed={seed})"
            ),
            "val_macro_f1": f"{scores['macro_f1']:.6f}",
            "per_class_f1": per_class_text(scores),
            "n_subjects_improved": "",
            "time_s": f"fit={fit_s:.3f}",
            "seed": str(seed),
            "decision": "confirmation",
            "model_sha256": digest,
            "est_kaggle_minutes": "",
            "ram_peak_gb": "",
            "deterministic": "",
            "commit": git_commit(),
        }
    )


def _ram_probe() -> int:
    """Peak memory of loading one val record and predicting. Test is not read."""
    config = project_config()
    model_path = ROOT / os.environ["CONFIRM_MODEL"]
    record_id = os.environ.get("CONFIRM_RECORD", "SC4041")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    labels = read_labels(config.tables_dir / "labels.csv")
    index = label_index_for_split(labels, "val")
    from src.expkit import load_record_base, apply_offsets

    record = load_record_base(config, "val", record_id, index)
    width = int(getattr(model, "n_features_in_", 0))
    if width == record.base.shape[1]:
        features = record.base
    else:
        features = apply_offsets(record.base, record.epoch_index, (-2, -1, 1, 2))
    if features.shape[1] != width:
        raise RuntimeError(f"probe width {features.shape[1]} != model {width}")
    _ = model.predict(features[:1])
    private_mb, working_mb = _process_peaks_mb()
    print(f"peak_private_mb {private_mb:.1f}")
    print(f"peak_working_mb {working_mb:.1f}")
    print(f"probe_rows {features.shape[0]} probe_cols {features.shape[1]}")
    return 0


def _run_probe(model_relative: str, record_id: str) -> tuple[float, float]:
    env = os.environ.copy()
    env["CONFIRM_RAM"] = "1"
    env["CONFIRM_MODEL"] = model_relative
    env["CONFIRM_RECORD"] = record_id
    env["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [sys.executable, "-m", "src.confirm_context2"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:] or completed.stdout[-2000:])
    private = working = float("nan")
    for line in completed.stdout.splitlines():
        if line.startswith("peak_private_mb "):
            private = float(line.split()[1])
        if line.startswith("peak_working_mb "):
            working = float(line.split()[1])
    print(completed.stdout, flush=True)
    return private, working


def main() -> int:
    """Retrain confirmation seeds and measure the CPU inference budget."""
    if os.environ.get("CONFIRM_RAM") == "1":
        return _ram_probe()
    config = project_config()
    n_jobs = worker_count()
    free = require_disk()
    print(f"disk_free_gib {free:.2f} n_jobs={n_jobs}", flush=True)
    try:
        import torch

        torch.set_num_threads(4)
        gpu = bool(torch.cuda.is_available())
        print(f"torch {torch.__version__} cuda {gpu}", flush=True)
    except ModuleNotFoundError:
        gpu = False
        print("torch unavailable; GPU inference skipped; CPU threads set via n_jobs=4", flush=True)
    lock = acquire_lock(config)
    try:
        labels = read_labels(config.tables_dir / "labels.csv")
        train_index = label_index_for_split(labels, "train")
        val_index = label_index_for_split(labels, "val")
        print("load context matrices", flush=True)
        train = load_context_split(config, "train", train_index, keep_base=False)
        val = load_context_split(config, "val", val_index, keep_base=True)
        y_train, _train_subjects = align_labels(train.ids, labels, "train")
        y_val, val_subjects = align_labels(val.ids, labels, "val")
        original = joblib.load(config.models_dir / "context2_v1.joblib")
        original_pred = np.asarray(original["model"].predict(val.features), dtype=np.int64)
        original_sha = prediction_sha(val.ids, original_pred)
        canonical = ROOT / "experiments" / "context2_v1_val_id_label.csv"
        file_sha = write_prediction_csv(canonical, val.ids, original_pred)
        if file_sha != original_sha:
            raise RuntimeError("prediction file hash disagrees with the in-memory hash")
        print(f"original_pred_sha {original_sha}", flush=True)
        names_path = ROOT / "experiments" / "feature_list_context2_v1.txt"
        names_path.write_text("\n".join(val.feature_names) + "\n", encoding="utf-8")
        np.save(config.models_dir / "context2_v1_val_proba.npy", _ordered_proba(original["model"], val.features))

        scores_by_seed: dict[int, float] = {42: float(score_arrays(y_val, original_pred)["macro_f1"])}
        print(f"seed 42 macro {scores_by_seed[42]:.6f}", flush=True)

        repeat, _name, repeat_s = _fit_seed(train.features, y_train, 42, n_jobs)
        repeat_pred = np.asarray(repeat.predict(val.features), dtype=np.int64)
        repeat_sha = prediction_sha(val.ids, repeat_pred)
        repeat_path = config.models_dir / "context2_seed42_repeat.joblib"
        joblib.dump(
            {"model": repeat, "feature_version": FEATURE_VERSION, "seed": 42, "fit_seconds": repeat_s},
            repeat_path,
            compress=3,
        )
        repeat_digest = file_sha256(repeat_path)
        repeat_scores = score_arrays(y_val, repeat_pred)
        _log_confirmation("context2_seed42_repeat", repeat, repeat_scores, repeat_s, 42, repeat_digest)
        train_match = repeat_sha == original_sha
        print(
            f"repeat_fit_s {repeat_s:.2f} repeat_macro {repeat_scores['macro_f1']:.6f} "
            f"pred_sha {repeat_sha} match {int(train_match)}",
            flush=True,
        )
        del repeat
        # Two consecutive inference passes on the saved model.
        infer_env = thread_env_one()
        os.environ["OMP_NUM_THREADS"] = "4"
        saved = original["model"]
        saved.set_params(n_jobs=4)
        first = np.asarray(saved.predict(val.features), dtype=np.int64)
        second = np.asarray(saved.predict(val.features), dtype=np.int64)
        infer_match = prediction_sha(val.ids, first) == prediction_sha(val.ids, second)
        restore_env(infer_env)
        print(f"infer_sha_match {int(infer_match)}", flush=True)

        started = time.perf_counter()
        reloaded = joblib.load(config.models_dir / "context2_v1.joblib")
        load_s = time.perf_counter() - started
        reloaded["model"].set_params(n_jobs=4)
        os.environ["OMP_NUM_THREADS"] = "4"
        predict_started = time.perf_counter()
        _ = reloaded["model"].predict(val.features)
        predict_s = time.perf_counter() - predict_started
        predict_test_s = predict_s * (TEST_EPOCHS / VAL_EPOCHS)
        print(
            f"load_s {load_s:.3f} val_predict_cpu4_s {predict_s:.3f} "
            f"scaled_66222_s {predict_test_s:.3f} gpu {gpu}",
            flush=True,
        )

        # One val record, one process, one thread. Not the test split.
        signal_path = config.signals_dir / "val" / "SC4041.npz"
        previous = thread_env_one()
        t0 = time.perf_counter()
        signal, ids = _load_signal(signal_path)
        features = features_from_signal(signal, 100.0)
        feature_s = time.perf_counter() - t0
        del signal, features, ids
        restore_env(previous)
        feature_24_s = feature_s * 24.0
        print(f"feature_one_record_s {feature_s:.3f} feature_24_s {feature_24_s:.1f}", flush=True)

        private_mb, _working = _run_probe("models/context2_v1.joblib", "SC4041")
        ram_gb = private_mb / 1024.0
        est_minutes = (load_s + feature_24_s + 3.0 * predict_test_s) / 60.0
        per_record_infer = predict_s / 16.0
        infer_24_min = (load_s + per_record_infer * 24.0) / 60.0
        deterministic = "yes" if train_match and infer_match else "no"
        print(
            "\n".join(
                [
                    f"ram_peak_gb {ram_gb:.3f}",
                    f"est_kaggle_minutes {est_minutes:.3f}",
                    f"infer_24_records_min {infer_24_min:.3f}",
                    f"deterministic {deterministic}",
                ]
            ),
            flush=True,
        )
        _update_budget("context2_v1", est_minutes, ram_gb, deterministic)

        seed_scores = [scores_by_seed[42]]
        for seed in (7, 123):
            model, _name, fit_s = _fit_seed(train.features, y_train, seed, n_jobs)
            predicted = np.asarray(model.predict(val.features), dtype=np.int64)
            scores = score_arrays(y_val, predicted)
            by_subject = subject_map(y_val, predicted, val_subjects)
            path = config.models_dir / f"context2_seed{seed}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "feature_version": FEATURE_VERSION,
                    "feature_names": val.feature_names,
                    "seed": seed,
                    "fit_seconds": fit_s,
                },
                path,
                compress=3,
            )
            digest = file_sha256(path)
            _log_confirmation(f"context2_seed{seed}", model, scores, fit_s, seed, digest)
            seed_scores.append(scores["macro_f1"])
            print(
                f"seed {seed} macro {scores['macro_f1']:.6f} fit_s {fit_s:.2f} "
                f"subjects {','.join(f'{key}:{value:.4f}' for key, value in sorted(by_subject.items()))}",
                flush=True,
            )
            del model
        seed_min = min(seed_scores)
        seed_max = max(seed_scores)
        seed_mean = float(np.mean(seed_scores))
        unstable = (seed_max - seed_min) > 0.01
        print(
            f"seeds 42,7,123 macro {seed_scores[0]:.6f} {seed_scores[1]:.6f} {seed_scores[2]:.6f} "
            f"mean {seed_mean:.6f} min {seed_min:.6f} max {seed_max:.6f} "
            f"unstable {int(unstable)}",
            flush=True,
        )
        summary = ROOT / "experiments" / "context2_confirmation.txt"
        summary.write_text(
            "\n".join(
                [
                    f"macro_42 {seed_scores[0]:.6f}",
                    f"macro_7 {seed_scores[1]:.6f}",
                    f"macro_123 {seed_scores[2]:.6f}",
                    f"mean {seed_mean:.6f}",
                    f"min {seed_min:.6f}",
                    f"max {seed_max:.6f}",
                    f"unstable {'yes' if unstable else 'no'}",
                    f"train_sha_original {original_sha}",
                    f"train_sha_repeat {repeat_sha}",
                    f"train_sha_match {'yes' if train_match else 'no'}",
                    f"infer_sha_match {'yes' if infer_match else 'no'}",
                    f"deterministic {deterministic}",
                    f"load_s {load_s:.3f}",
                    f"feature_one_record_s {feature_s:.3f}",
                    f"feature_24_s {feature_24_s:.1f}",
                    f"val_predict_cpu4_s {predict_s:.3f}",
                    f"scaled_66222_s {predict_test_s:.3f}",
                    f"ram_peak_gb {ram_gb:.3f}",
                    f"est_kaggle_minutes {est_minutes:.3f}",
                    f"infer_24_records_min {infer_24_min:.3f}",
                    "gpu no",
                    "record SC4041",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    finally:
        release_lock(lock)


def _update_budget(model_id: str, minutes: float, ram_gb: float, deterministic: str) -> None:
    import pandas as pd

    path = ROOT / "experiments" / "results.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    mask = frame["id"] == model_id
    if int(mask.sum()) != 1:
        raise RuntimeError(f"expected one row for {model_id}")
    frame.loc[mask, "est_kaggle_minutes"] = f"{minutes:.3f}"
    frame.loc[mask, "ram_peak_gb"] = f"{ram_gb:.3f}"
    frame.loc[mask, "deterministic"] = deterministic
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())

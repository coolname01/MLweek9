"""Experiment C: per-record standardization of ``features_v1``.

Two variants only. C1 is the ±2 context of the standardized columns (271).
C2 appends that context to the raw context (541, one ``n_neighbors``).
``cache/features_v1`` is read and hashed before and after. The test split
is not opened. The formula lives in :mod:`src.record_z` and is checked
against ``configs/config.yaml`` before any recording is loaded.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.context import OFFSETS, gap_rows
from src.data import EXPECTED_EPOCHS, file_sha256, read_labels
from src.expkit import (
    ROOT,
    TRAIN_TIME_CAP_S,
    fit_model,
    git_commit,
    kaggle_minutes,
    load_bases,
    load_champion,
    load_record_base,
    markdown_table,
    model_description,
    n_improved,
    per_class_text,
    prediction_sha,
    project_config,
    require_disk,
    restore_env,
    score_arrays,
    stack_offsets,
    subject_map,
    thread_env_one,
    tracked_subject_f1,
    upsert_result,
    write_prediction_csv,
)
from src.features import (
    N_FEATURES,
    _load_signal,
    _pid_is_running,
    _process_peaks_mb,
    align_labels,
    features_from_signal,
    release_lock,
)
from src.record_z import (
    C1_COLUMNS,
    C2_COLUMNS,
    CACHE_NAME,
    FORMULA_TEXT,
    SCOPE_TEXT,
    STD_DDOF,
    STD_EPS,
    assert_single_record_api,
    c1_cache_path,
    cache_matches,
    feature_names_for,
    population_mean_std,
    self_check,
    transform_record,
    variant_matrix,
    write_c1_cache,
    zscore_record_features,
)

GATE_MACRO = 0.769678
TIE_MACRO = 0.005
N_JOBS = 10
SEED = 42
CONTEXT2_SHA = "ef7125020f85b4fdbdedb5c28f705fe3695b8ca33d3ba688f56576df41b5e4bd"
SMOKE_RECORDS: tuple[tuple[str, str], ...] = (
    ("train", "SC4011"),
    ("train", "SC4012"),
    ("val", "SC4041"),
)
SUBJECTS: tuple[str, ...] = (
    "SC404",
    "SC407",
    "SC417",
    "SC424",
    "SC426",
    "SC427",
    "SC457",
    "SC459",
)
SUMMARY_PATH = ROOT / "experiments" / "record_z_c_summary.json"
HYPOTHESIS = {
    "c1": (
        "Per-record z-scoring of the 54 features_v1 columns, followed by the "
        "same ±2 context computed on those z-scored values, improves val macro-F1 "
        "over context2_v1."
    ),
    "c2": (
        "Concatenating the raw ±2 context with the same context of per-record "
        "z-scored features improves val macro-F1 over context2_v1."
    ),
}


def _features_text(variant: str) -> str:
    """Feature description stored in ``experiments/results.csv``."""
    common = (
        "features_v1 cache read only and not modified; "
        f"formula {FORMULA_TEXT}; population std ddof={STD_DDOF}; eps={STD_EPS}; "
        "mean and std over all epochs of that record only; "
        "neighbors taken from z at epoch_index offsets -2,-1,+1,+2; "
        "n_neighbors in 0..4; cache features_c1 stores the 54 z columns"
    )
    if variant == "c1":
        return common + "; C1 is the z context only, 271 columns"
    if variant == "c2":
        return (
            common
            + "; C2 is the raw context body (270), the z context body (270), "
            "and one n_neighbors column, 541 columns"
        )
    raise ValueError(variant)


def _assert_frozen_config() -> None:
    """Stop if ``config.yaml`` does not repeat the frozen formula."""
    raw = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    block = raw["record_z"]
    offsets = [int(value) for value in block["offsets"]]
    checks = (
        (block["formula"] != FORMULA_TEXT, "formula"),
        (float(block["eps"]) != STD_EPS, "eps"),
        (int(block["ddof"]) != STD_DDOF, "ddof"),
        (block["scope"] != SCOPE_TEXT, "scope"),
        (block["neighbor_source"] != "z", "neighbor_source"),
        (offsets != list(OFFSETS), "offsets"),
        (int(block["c1_columns"]) != C1_COLUMNS, "c1_columns"),
        (int(block["c2_columns"]) != C2_COLUMNS, "c2_columns"),
        (block["cache"] != "cache/features_c1", "cache"),
        (int(raw["seed"]) != SEED, "seed"),
    )
    drifted = [name for failed, name in checks if failed]
    if drifted:
        raise RuntimeError(f"config.yaml does not match the frozen formula: {drifted}")


def _tree_digest(folder: Path) -> tuple[str, int, int]:
    """SHA-256 of every file under ``folder``, plus file count and bytes."""
    digest = hashlib.sha256()
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    total = 0
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(b"\0")
        file_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                file_hash.update(chunk)
        total += path.stat().st_size
        digest.update(file_hash.digest())
    return digest.hexdigest(), len(files), total


def _acquire_lock(cache_dir: Path) -> Path:
    """Lock ``cache/features_c1`` without touching ``cache/features_v1``."""
    folder = cache_dir / CACHE_NAME
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ".experiment.lock"
    if path.is_file():
        raw = path.read_text(encoding="utf-8").strip()
        pid = int(raw) if raw.isdigit() else -1
        if _pid_is_running(pid):
            raise RuntimeError(f"another experiment is running (pid {pid}, lock {path})")
        path.unlink()
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return path


def _fraction(values: np.ndarray, kind: str) -> float:
    """Element fraction of NaN or non-finite values."""
    if kind == "nan":
        return float(np.isnan(values).mean())
    if kind == "inf":
        return float(np.isinf(values).mean())
    raise ValueError(kind)


def _n1_rates(scores: dict[str, Any]) -> tuple[float, float]:
    """Precision and recall of class N1 from the confusion matrix."""
    matrix = scores["confusion_matrix"]
    true_positive = int(matrix[1, 1])
    predicted = int(matrix[:, 1].sum())
    actual = int(matrix[1, :].sum())
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    return precision, recall


def _prepare(record: Any) -> dict[str, Any]:
    """Standardize one loaded recording and check the frozen formula."""
    mean, std = population_mean_std(record.base)
    standardized = zscore_record_features(record.base)
    values = np.asarray(record.base, dtype=np.float64)
    again = np.ascontiguousarray(
        (values - mean) / (std + np.float64(STD_EPS)),
        dtype=np.float32,
    )
    if not np.array_equal(again, standardized):
        raise RuntimeError(f"{record.record_id} z-score drifted from the frozen formula")
    if not np.isfinite(standardized).all():
        raise RuntimeError(f"{record.record_id} z-score is not finite")
    return {
        "split": record.split,
        "record_id": record.record_id,
        "ids": record.ids,
        "epoch_index": record.epoch_index,
        "raw": record.base,
        "z": standardized,
        "mean": mean,
        "std": std,
        "n_constant": int(np.count_nonzero(std == 0.0)),
    }


def _stack(records: list[dict[str, Any]], variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Stack one variant. Each recording is transformed on its own."""
    names = feature_names_for(variant)
    blocks: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for record in records:
        matrix = variant_matrix(record["raw"], record["z"], record["epoch_index"], variant)
        if matrix.shape != (record["raw"].shape[0], len(names)):
            raise RuntimeError(f"{record['record_id']} {variant} shape {matrix.shape}")
        blocks.append(matrix)
        ids.append(record["ids"])
    return np.concatenate(blocks, axis=0), np.concatenate(ids, axis=0)


def _smoke(config: Any, labels: pd.DataFrame) -> dict[str, Any]:
    """Shape, non-finite fraction, constant columns and time on three records."""
    started = time.perf_counter()
    loaded = []
    for split, record_id in SMOKE_RECORDS:
        from src.context import label_index_for_split

        index = label_index_for_split(labels, split)
        loaded.append(load_record_base(config, split, record_id, index))
    if [record.record_id for record in loaded] != [item[1] for item in SMOKE_RECORDS]:
        raise RuntimeError("smoke record order drifted")
    rows: list[dict[str, Any]] = []
    prepared = [_prepare(record) for record in loaded]
    pooled = zscore_record_features(np.vstack([prepared[0]["raw"], prepared[1]["raw"]]))
    separate = prepared[0]["z"]
    isolation = float(
        np.max(np.abs(pooled[: separate.shape[0]].astype(np.float64) - separate.astype(np.float64)))
    )
    if isolation <= 0.0:
        raise RuntimeError("SC4011 z-score did not change when pooled with SC4012")
    for record in prepared:
        record_started = time.perf_counter()
        first = variant_matrix(record["raw"], record["z"], record["epoch_index"], "c1")
        second = variant_matrix(record["raw"], record["z"], record["epoch_index"], "c2")
        elapsed = time.perf_counter() - record_started
        if first.shape != (record["raw"].shape[0], C1_COLUMNS):
            raise RuntimeError(f"{record['record_id']} c1 shape {first.shape}")
        if second.shape != (record["raw"].shape[0], C2_COLUMNS):
            raise RuntimeError(f"{record['record_id']} c2 shape {second.shape}")
        if not np.array_equal(first[:, :N_FEATURES], record["z"]):
            raise RuntimeError(f"{record['record_id']} context replaced the current z row")
        if not np.array_equal(second[:, -1], first[:, -1]):
            raise RuntimeError(f"{record['record_id']} neighbor counts differ")
        if np.isinf(first).any() or np.isinf(second).any():
            raise RuntimeError(f"{record['record_id']} context contains inf")
        row = {
            "split": record["split"],
            "record_id": record["record_id"],
            "n_epochs": int(record["raw"].shape[0]),
            "c1_shape": list(first.shape),
            "c2_shape": list(second.shape),
            "z_nan": _fraction(record["z"], "nan"),
            "z_inf": _fraction(record["z"], "inf"),
            "c1_nan": _fraction(first, "nan"),
            "c1_inf": _fraction(first, "inf"),
            "c2_nan": _fraction(second, "nan"),
            "c2_inf": _fraction(second, "inf"),
            "n_constant": record["n_constant"],
            "seconds": elapsed,
        }
        if record["record_id"] == "SC4041":
            described = gap_rows(record["z"], record["epoch_index"], first)
            anchor = next(
                item
                for item in described
                if item.next_epoch is not None and item.next_epoch > item.epoch_index + 1
            )
            row["gap_epoch"] = int(anchor.epoch_index)
            row["gap_sources"] = [None if source is None else int(source) for source in anchor.sources]
            row["gap_plus1_is_next_row"] = bool(anchor.plus1_is_next_row)
            plus1 = OFFSETS.index(1)
            if anchor.sources[plus1] is not None or anchor.plus1_is_next_row:
                raise RuntimeError("SC4041 offset +1 followed the next file row")
        rows.append(row)
        print(
            f"smoke {record['split']}/{record['record_id']} "
            f"epochs={row['n_epochs']} c1={tuple(first.shape)} c2={tuple(second.shape)} "
            f"z_nan={row['z_nan']:.6f} z_inf={row['z_inf']:.6f} "
            f"c1_nan={row['c1_nan']:.6f} c2_nan={row['c2_nan']:.6f} "
            f"constant={row['n_constant']} seconds={elapsed:.3f}",
            flush=True,
        )
    print(f"smoke_isolation_max_abs {isolation:.6f}", flush=True)
    print("SMOKE_PASS", flush=True)
    return {
        "records": rows,
        "isolation_max_abs": isolation,
        "seconds": time.perf_counter() - started,
    }


def _write_cache(config: Any, records: list[dict[str, Any]]) -> float:
    """Store the 54 standardized columns under ``cache/features_c1``."""
    started = time.perf_counter()
    written = 0
    reused = 0
    for record in records:
        path = c1_cache_path(config, record["split"], record["record_id"])
        if cache_matches(path, record["z"], record["ids"]):
            reused += 1
            fresh = False
        else:
            write_c1_cache(path, record["z"], record["ids"], record["mean"], record["std"])
            if not cache_matches(path, record["z"], record["ids"]):
                raise RuntimeError(f"cache round-trip failed for {path}")
            written += 1
            fresh = True
        print(
            f"cache {record['split']}/{record['record_id']} epochs={record['raw'].shape[0]} "
            f"constant={record['n_constant']} wrote={int(fresh)}",
            flush=True,
        )
    print(f"cache_written {written} cache_reused {reused}", flush=True)
    return time.perf_counter() - started


def _reproduce_context2(config: Any, val_records: list[Any]) -> None:
    """The raw ±2 matrix must reproduce the saved context2_v1 val labels."""
    stacked = stack_offsets(val_records, OFFSETS)
    frame = pd.read_csv(
        ROOT / "experiments" / "context2_v1_val_id_label.csv",
        dtype=str,
        keep_default_na=False,
    )
    if list(frame.columns) != ["id", "label"]:
        raise RuntimeError(f"context2 prediction columns {list(frame.columns)}")
    mapping = dict(zip(frame["id"].tolist(), frame["label"].tolist(), strict=True))
    if len(mapping) != len(frame):
        raise RuntimeError("duplicate ids in context2_v1 val predictions")
    expected = np.array(
        [int(mapping[str(epoch_id)]) for epoch_id in stacked.ids.tolist()],
        dtype=np.int64,
    )
    bundle = joblib.load(config.models_dir / "context2_v1.joblib")
    predicted = np.asarray(bundle["model"].predict(stacked.features), dtype=np.int64)
    mismatch = int(np.sum(predicted != expected))
    if mismatch:
        raise RuntimeError(f"raw ±2 context does not reproduce context2_v1 ({mismatch})")
    print(f"context2_reproduced rows={predicted.shape[0]}", flush=True)


def _subject_rows(
    by_subject: dict[str, float],
    reference: dict[str, float],
) -> dict[str, dict[str, float | bool]]:
    """Per-subject macro-F1, delta and a strict-improvement flag."""
    if tuple(sorted(by_subject)) != SUBJECTS:
        raise RuntimeError(f"subjects {sorted(by_subject)}")
    rows: dict[str, dict[str, float | bool]] = {}
    for subject in SUBJECTS:
        score = float(by_subject[subject])
        base = float(reference[subject])
        rows[subject] = {
            "f1": score,
            "reference": base,
            "delta": score - base,
            "improved": score > base,
        }
    return rows


def _fit_variant(
    config: Any,
    labels: pd.DataFrame,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    variant: str,
    reference_subjects: dict[str, float],
    smoke_s: float,
    cache_s: float,
) -> dict[str, Any]:
    """Fit one variant on train and score val."""
    columns = C1_COLUMNS if variant == "c1" else C2_COLUMNS
    model_id = f"record_z_{variant}"
    print(f"FIT {model_id} columns={columns}", flush=True)
    build_started = time.perf_counter()
    train_x, train_ids = _stack(train_records, variant)
    matrix_s = time.perf_counter() - build_started
    if train_x.shape != (EXPECTED_EPOCHS["train"], columns):
        raise RuntimeError(f"{variant} train shape {train_x.shape}")
    y_train, _train_subjects = align_labels(train_ids, labels, "train")
    model, fit_s = fit_model(train_x, y_train, SEED, N_JOBS, TRAIN_TIME_CAP_S)
    del train_x
    gc.collect()
    val_started = time.perf_counter()
    val_x, val_ids = _stack(val_records, variant)
    matrix_s += time.perf_counter() - val_started
    if val_x.shape != (EXPECTED_EPOCHS["val"], columns):
        raise RuntimeError(f"{variant} val shape {val_x.shape}")
    y_val, val_subjects = align_labels(val_ids, labels, "val")
    predict_started = time.perf_counter()
    predicted = np.asarray(model.predict(val_x), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(val_x), dtype=np.float32)
    predict_s = time.perf_counter() - predict_started
    scores = score_arrays(y_val, predicted)
    by_subject = subject_map(y_val, predicted, val_subjects)
    improved = n_improved(by_subject, reference_subjects)
    precision, recall = _n1_rates(scores)
    passed = bool(scores["macro_f1"] >= GATE_MACRO and improved >= 5)
    model_path = config.models_dir / f"{model_id}.joblib"
    config.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_version": model_id,
            "formula": FORMULA_TEXT,
            "eps": STD_EPS,
            "ddof": STD_DDOF,
            "scope": SCOPE_TEXT,
            "variant": variant,
            "feature_names": feature_names_for(variant),
            "context_offsets": list(OFFSETS),
            "neighbor_source": "z",
            "single_record_function": "src.record_z.transform_record",
            "seed": SEED,
            "model_name": model_description(model),
            "classes": list(config.classes),
            "parent": "context2_v1",
            "train_rows": EXPECTED_EPOCHS["train"],
            "val_rows": int(val_x.shape[0]),
            "n_features": columns,
            "fit_seconds": fit_s,
            "n_jobs": N_JOBS,
        },
        model_path,
        compress=3,
    )
    digest = file_sha256(model_path)
    np.save(config.models_dir / f"{model_id}_val_proba.npy", probabilities)
    prediction_path = ROOT / "experiments" / f"{model_id}_val_id_label.csv"
    pred_sha = write_prediction_csv(prediction_path, val_ids, predicted)
    if prediction_sha(val_ids, predicted) != pred_sha:
        raise RuntimeError("prediction SHA drifted from the written file")
    sc457, sc459 = tracked_subject_f1(by_subject)
    print(
        f"{model_id} macro_f1={scores['macro_f1']:.6f} improved={improved}/8 "
        f"gate={'pass' if passed else 'fail'} fit_s={fit_s:.1f}",
        flush=True,
    )
    payload = {
        "variant": variant,
        "model_id": model_id,
        "macro_f1": float(scores["macro_f1"]),
        "accuracy": float(scores["accuracy"]),
        "per_class_f1": [float(value) for value in scores["per_class_f1"]],
        "n1_precision": precision,
        "n1_recall": recall,
        "improved": improved,
        "passed": passed,
        "subjects": _subject_rows(by_subject, reference_subjects),
        "f1_sc457": sc457,
        "f1_sc459": sc459,
        "matrix_s": matrix_s,
        "fit_s": fit_s,
        "predict_s": predict_s,
        "smoke_s": smoke_s,
        "cache_s": cache_s,
        "sha256": digest,
        "pred_sha": pred_sha,
        "model_path": str(model_path),
        "model_name": model_description(model),
        "scores": scores,
        "val_ids": val_ids,
        "y_val": y_val,
        "y_train": y_train,
        "predicted": predicted,
        "val_x": val_x,
    }
    upsert_result(_result_row(payload, "scored"))
    return payload


def _choose(c1: dict[str, Any], c2: dict[str, Any]) -> str | None:
    """Return the winning variant, or None when neither passes the gate.

    When both pass and C2 leads by more than 0.005 macro-F1, C2 wins.
    A passing C1 wins in every other tie, including a gap of at most 0.005.
    """
    if c1["passed"] and c2["passed"]:
        if c2["macro_f1"] > c1["macro_f1"] + TIE_MACRO:
            return "c2"
        return "c1"
    if c1["passed"]:
        return "c1"
    if c2["passed"]:
        return "c2"
    return None


def _decision(variant: str, winner: str | None, passed: bool) -> str:
    """Log decision for one variant."""
    if winner == variant:
        return "accepted"
    if passed:
        return "not_selected"
    return "rejected"


def _result_row(result: dict[str, Any], decision: str, extra_time: str = "") -> dict[str, str]:
    """One ``results.csv`` row."""
    time_s = (
        f"smoke={result['smoke_s']:.3f};cache={result['cache_s']:.3f};"
        f"matrix={result['matrix_s']:.3f};fit={result['fit_s']:.3f};"
        f"predict={result['predict_s']:.3f};"
        f"n1_p={result['n1_precision']:.6f};n1_r={result['n1_recall']:.6f};"
        f"gate={'pass' if result['passed'] else 'fail'}"
    )
    if extra_time:
        time_s = f"{time_s};{extra_time}"
    return {
        "id": result["model_id"],
        "hypothesis": HYPOTHESIS[result["variant"]],
        "features": _features_text(result["variant"]),
        "model": result["model_name"],
        "val_macro_f1": f"{result['macro_f1']:.6f}",
        "per_class_f1": ",".join(f"{value:.6f}" for value in result["per_class_f1"]),
        "n_subjects_improved": str(result["improved"]),
        "time_s": time_s,
        "seed": str(SEED),
        "decision": decision,
        "model_sha256": result["sha256"],
        "est_kaggle_minutes": result.get("est_kaggle_minutes", ""),
        "ram_peak_gb": result.get("ram_peak_gb", ""),
        "deterministic": result.get("deterministic", ""),
        "commit": git_commit(),
        "f1_sc457": result["f1_sc457"],
        "f1_sc459": result["f1_sc459"],
    }


def _save_model(
    config: Any,
    model: Any,
    variant: str,
    model_id: str,
    columns: int,
    fit_s: float,
    seed: int,
) -> str:
    """Write a checkpoint and return its SHA-256."""
    path = config.models_dir / f"{model_id}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_version": model_id,
            "formula": FORMULA_TEXT,
            "eps": STD_EPS,
            "ddof": STD_DDOF,
            "scope": SCOPE_TEXT,
            "variant": variant,
            "feature_names": feature_names_for(variant),
            "context_offsets": list(OFFSETS),
            "neighbor_source": "z",
            "single_record_function": "src.record_z.transform_record",
            "seed": seed,
            "model_name": model_description(model),
            "classes": list(config.classes),
            "parent": "context2_v1",
            "n_features": columns,
            "fit_seconds": fit_s,
            "n_jobs": N_JOBS,
        },
        path,
        compress=3,
    )
    return file_sha256(path)


def _confirm(
    config: Any,
    labels: pd.DataFrame,
    train_records: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Seeds 7 and 123, a second seed-42 fit, and the Kaggle budget."""
    variant = result["variant"]
    columns = C1_COLUMNS if variant == "c1" else C2_COLUMNS
    val_x = result["val_x"]
    val_ids = result["val_ids"]
    y_val = result["y_val"]
    bundle = joblib.load(result["model_path"])
    reloaded = bundle["model"]
    pred_a = np.asarray(reloaded.predict(val_x), dtype=np.int64)
    pred_b = np.asarray(reloaded.predict(val_x), dtype=np.int64)
    sha_a = prediction_sha(val_ids, pred_a)
    sha_b = prediction_sha(val_ids, pred_b)
    if not np.array_equal(pred_a, result["predicted"]) or sha_a != result["pred_sha"]:
        raise RuntimeError("reloaded seed-42 predictions differ from the scoring pass")
    del reloaded
    gc.collect()
    train_x, train_ids = _stack(train_records, variant)
    y_train, _ignored = align_labels(train_ids, labels, "train")
    if not np.array_equal(y_train, result["y_train"]):
        raise RuntimeError("train labels drifted before the confirmation fits")
    repeat, repeat_s = fit_model(train_x, y_train, SEED, N_JOBS, TRAIN_TIME_CAP_S)
    pred_c = np.asarray(repeat.predict(val_x), dtype=np.int64)
    sha_c = prediction_sha(val_ids, pred_c)
    repeat_sha = _save_model(
        config,
        repeat,
        variant,
        f"{result['model_id']}_seed42_repeat",
        columns,
        repeat_s,
        SEED,
    )
    repeat_macro = float(score_arrays(y_val, pred_c)["macro_f1"])
    del repeat
    gc.collect()
    deterministic = "yes" if sha_a == sha_b == sha_c else "no"
    print(
        f"determinism {deterministic} sha_a={sha_a} sha_b={sha_b} sha_c={sha_c}",
        flush=True,
    )
    seed_scores = {SEED: result["macro_f1"]}
    seed_fits = {"42_repeat_s": repeat_s, "42_repeat_sha256": repeat_sha}
    for seed in (7, 123):
        model, fit_s = fit_model(train_x, y_train, seed, N_JOBS, TRAIN_TIME_CAP_S)
        predicted = np.asarray(model.predict(val_x), dtype=np.int64)
        scores = score_arrays(y_val, predicted)
        by_subject = subject_map(y_val, predicted, align_labels(val_ids, labels, "val")[1])
        digest = _save_model(
            config,
            model,
            variant,
            f"{result['model_id']}_seed{seed}",
            columns,
            fit_s,
            seed,
        )
        pred_sha = prediction_sha(val_ids, predicted)
        sc457, sc459 = tracked_subject_f1(by_subject)
        upsert_result(
            {
                "id": f"{result['model_id']}_seed{seed}",
                "hypothesis": (
                    f"Confirmation fit of {result['model_id']}. Not a candidate "
                    "for replacing the accepted model."
                ),
                "features": _features_text(variant),
                "model": (
                    f"{model_description(model)},seed={seed}"
                ),
                "val_macro_f1": f"{scores['macro_f1']:.6f}",
                "per_class_f1": per_class_text(scores),
                "n_subjects_improved": "",
                "time_s": f"fit={fit_s:.3f};pred_sha={pred_sha}",
                "seed": str(seed),
                "decision": "confirmation",
                "model_sha256": digest,
                "est_kaggle_minutes": "",
                "ram_peak_gb": "",
                "deterministic": "",
                "commit": git_commit(),
                "f1_sc457": sc457,
                "f1_sc459": sc459,
            }
        )
        seed_scores[seed] = float(scores["macro_f1"])
        seed_fits[str(seed)] = {"fit_s": fit_s, "sha256": digest, "macro_f1": float(scores["macro_f1"])}
        print(f"seed {seed} macro_f1={scores['macro_f1']:.6f} fit_s={fit_s:.1f}", flush=True)
        del model
        gc.collect()
    del train_x
    gc.collect()
    values = np.array([seed_scores[SEED], seed_scores[7], seed_scores[123]], dtype=np.float64)
    spread = float(values.max() - values.min())
    budget = _measure_budget(config, labels, Path(result["model_path"]), val_x, variant)
    upsert_result(
        {
            "id": f"{result['model_id']}_seed42_repeat",
            "hypothesis": (
                f"Second seed-42 fit of {result['model_id']}, used only to check "
                "that val id,label is unchanged."
            ),
            "features": _features_text(variant),
            "model": f"LGBMClassifier(same as {result['model_id']},seed=42)",
            "val_macro_f1": f"{repeat_macro:.6f}",
            "per_class_f1": "",
            "n_subjects_improved": "",
            "time_s": f"fit={repeat_s:.3f};pred_sha={sha_c}",
            "seed": str(SEED),
            "decision": "confirmation",
            "model_sha256": repeat_sha,
            "est_kaggle_minutes": "",
            "ram_peak_gb": "",
            "deterministic": deterministic,
            "commit": git_commit(),
            "f1_sc457": "",
            "f1_sc459": "",
        }
    )
    return {
        "deterministic": deterministic,
        "pred_sha": sha_a,
        "pred_sha_repeat": sha_c,
        "seeds": {
            "42": seed_scores[SEED],
            "7": seed_scores[7],
            "123": seed_scores[123],
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "range": spread,
            "stable": spread <= 0.01,
        },
        "seed_fits": seed_fits,
        "budget": budget,
    }


def _feature_one_record(config: Any, labels: pd.DataFrame, variant: str) -> float:
    """Welch, z-score and context for val record SC4041. Test is not read."""
    from src.context import label_index_for_split

    path = config.signals_dir / "val" / "SC4041.npz"
    if path.parent.name != "val":
        raise RuntimeError(f"feature timing refused {path}")
    index = label_index_for_split(labels, "val")
    previous = thread_env_one()
    try:
        started = time.perf_counter()
        signal, ids = _load_signal(path)
        raw = features_from_signal(signal, float(config.fs))
        located = index.reindex(list(ids))
        if bool(located["epoch_index"].isna().any()):
            raise RuntimeError("SC4041 ids are missing from labels")
        if not np.all(located["record_id"].astype(str).to_numpy() == "SC4041"):
            raise RuntimeError("SC4041 label record_id mismatch")
        epoch_index = located["epoch_index"].to_numpy(dtype=np.int64)
        matrix = transform_record(raw, epoch_index, variant)
        elapsed = time.perf_counter() - started
    finally:
        restore_env(previous)
    expected = C1_COLUMNS if variant == "c1" else C2_COLUMNS
    if matrix.shape != (signal.shape[0], expected):
        raise RuntimeError(f"SC4041 {variant} shape {matrix.shape}")
    return float(elapsed)


def _ram_probe() -> int:
    """Peak memory of one val record. Test is not read."""
    config = project_config()
    labels = read_labels(config.tables_dir / "labels.csv")
    variant = os.environ["E11_VARIANT"]
    from src.context import label_index_for_split

    index = label_index_for_split(labels, "val")
    record = load_record_base(config, "val", "SC4041", index)
    matrix = transform_record(record.base, record.epoch_index, variant)
    model = joblib.load(config.models_dir / os.environ["E11_MODEL"])["model"]
    model.set_params(n_jobs=4)
    if int(model.n_features_in_) != matrix.shape[1]:
        raise RuntimeError(f"probe width {matrix.shape[1]} != model {model.n_features_in_}")
    _ = model.predict(matrix)
    private_mb, working_mb = _process_peaks_mb()
    print(f"peak_private_mb {private_mb:.1f}")
    print(f"peak_working_mb {working_mb:.1f}")
    print(f"probe_rows {matrix.shape[0]} probe_cols {matrix.shape[1]}")
    return 0


def _run_ram_probe(model_name: str, variant: str) -> tuple[float, float]:
    """Run the memory probe in a fresh process."""
    last = ""
    for attempt in (1, 2):
        env = os.environ.copy()
        env["E11_RAM"] = "1"
        env["E11_VARIANT"] = variant
        env["E11_MODEL"] = model_name
        env["OMP_NUM_THREADS"] = "4"
        completed = subprocess.run(
            [sys.executable, "-m", "src.e11_record_z"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        text = completed.stdout
        if completed.returncode == 0:
            private = working = float("nan")
            for line in text.splitlines():
                if line.startswith("peak_private_mb "):
                    private = float(line.split()[1])
                elif line.startswith("peak_working_mb "):
                    working = float(line.split()[1])
            if np.isfinite(private) and np.isfinite(working):
                print(text, flush=True)
                return private, working
        last = (completed.stderr or text)[-2000:]
        print(f"ram probe attempt {attempt} failed", flush=True)
    raise RuntimeError(last)


def _gpu_note() -> str:
    """LightGBM inference stays on CPU. Torch is only a device note."""
    try:
        import torch
    except ModuleNotFoundError:
        return "torch_absent"
    if torch.cuda.is_available():
        return "lgbm_cpu_only"
    return "no_cuda"


def _measure_budget(
    config: Any,
    labels: pd.DataFrame,
    model_path: Path,
    features: np.ndarray,
    variant: str,
) -> dict[str, Any]:
    """Kaggle-style minutes and the private-RAM peak of one record."""
    gpu = _gpu_note()
    started = time.perf_counter()
    reloaded = joblib.load(model_path)
    load_s = time.perf_counter() - started
    reloaded["model"].set_params(n_jobs=4)
    _ = reloaded["model"].predict(features[:8])
    started = time.perf_counter()
    _ = reloaded["model"].predict(features)
    predict_s = time.perf_counter() - started
    feature_s = _feature_one_record(config, labels, variant)
    private_mb, working_mb = _run_ram_probe(model_path.name, variant)
    minutes = kaggle_minutes(load_s, feature_s, predict_s)
    print(
        f"budget load_s={load_s:.3f} feature_one_s={feature_s:.3f} "
        f"predict_s={predict_s:.3f} est_kaggle_minutes={minutes:.3f} "
        f"ram_private_mb={private_mb:.1f} ram_working_mb={working_mb:.1f} gpu={gpu}",
        flush=True,
    )
    return {
        "load_s": load_s,
        "feature_one_s": feature_s,
        "predict_s": predict_s,
        "est_kaggle_minutes": minutes,
        "ram_peak_gb": private_mb / 1024.0,
        "ram_working_gb": working_mb / 1024.0,
        "gpu": gpu,
    }


def _print_tables(
    reference_macro: float,
    reference_subjects: dict[str, float],
    results: dict[str, dict[str, Any]],
    winner: str | None,
) -> None:
    """Print the comparison used in the experiment log."""
    headers = ["model", "macro-F1", "W", "N1", "N2", "N3", "REM", "N1_P", "N1_R", "improved", "fit_s"]
    rows = [["context2_v1", f"{reference_macro:.6f}", "", "", "", "", "", "", "", "", ""]]
    for variant in ("c1", "c2"):
        item = results[variant]
        per_class = item["per_class_f1"]
        rows.append(
            [
                item["model_id"],
                f"{item['macro_f1']:.6f}",
                *[f"{value:.4f}" for value in per_class],
                f"{item['n1_precision']:.4f}",
                f"{item['n1_recall']:.4f}",
                f"{item['improved']}/8",
                f"{item['fit_s']:.1f}",
            ]
        )
    print(markdown_table(headers, rows), flush=True)
    subject_headers = ["subject", "context2_v1", "c1", "c1_delta", "c2", "c2_delta"]
    subject_rows = []
    for subject in SUBJECTS:
        c1 = results["c1"]["subjects"][subject]
        c2 = results["c2"]["subjects"][subject]
        subject_rows.append(
            [
                subject,
                f"{c1['reference']:.4f}",
                f"{c1['f1']:.4f}",
                f"{c1['delta']:+.4f}",
                f"{c2['f1']:.4f}",
                f"{c2['delta']:+.4f}",
            ]
        )
    print(markdown_table(subject_headers, subject_rows), flush=True)
    for variant in ("c1", "c2"):
        item = results[variant]
        print(
            f"{item['model_id']} SC457={item['f1_sc457']} SC459={item['f1_sc459']} "
            f"decision={_decision(variant, winner, item['passed'])}",
            flush=True,
        )
    if set(reference_subjects) != set(SUBJECTS):
        raise RuntimeError("subject table is missing a val subject")


def _json_ready(value: Any) -> Any:
    """Convert experiment objects to JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items() if key not in {"scores", "val_ids", "y_val", "y_train", "predicted", "val_x"}}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer, np.bool_)):
        return int(value) if isinstance(value, np.integer) else bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save_champion(result: dict[str, Any]) -> None:
    """Point ``champion.json`` at the accepted variant."""
    from src.expkit import save_champion

    save_champion(
        {
            "id": result["model_id"],
            "macro_f1": result["macro_f1"],
            "per_class_f1": result["per_class_f1"],
            "accuracy": result["accuracy"],
            "subject_f1": {
                subject: float(result["subjects"][subject]["f1"]) for subject in SUBJECTS
            },
            "offsets": list(OFFSETS),
            "base": "features_v1",
            "formula": FORMULA_TEXT,
            "eps": STD_EPS,
            "ddof": STD_DDOF,
            "variant": result["variant"],
            "model_path": f"models/{result['model_id']}.joblib",
            "feature_names": list(feature_names_for(result["variant"])),
            "n_features": C1_COLUMNS if result["variant"] == "c1" else C2_COLUMNS,
            "parent": "context2_v1",
        }
    )


def _log_stop(variant: str, reason: str) -> None:
    """Record a variant that did not produce a score."""
    upsert_result(
        {
            "id": f"record_z_{variant}",
            "hypothesis": HYPOTHESIS[variant],
            "features": _features_text(variant),
            "model": "",
            "val_macro_f1": "",
            "per_class_f1": "",
            "n_subjects_improved": "",
            "time_s": reason,
            "seed": str(SEED),
            "decision": "stopped_time",
            "model_sha256": "",
            "est_kaggle_minutes": "",
            "ram_peak_gb": "",
            "deterministic": "",
            "commit": git_commit(),
            "f1_sc457": "",
            "f1_sc459": "",
        }
    )


def main() -> int:
    """Smoke, then C1, then C2. Extra seeds run only for a passing winner."""
    if os.environ.get("E11_RAM") == "1":
        return _ram_probe()
    print(
        f"FORMULA {FORMULA_TEXT} scope={SCOPE_TEXT} eps={STD_EPS} ddof={STD_DDOF} "
        f"neighbors=z offsets={list(OFFSETS)} c1={C1_COLUMNS} c2={C2_COLUMNS} "
        f"gate={GATE_MACRO:.6f} tie={TIE_MACRO}",
        flush=True,
    )
    _assert_frozen_config()
    self_check()
    assert_single_record_api()
    print("single_record_api zscore_record_features(features); transform_record(features, epoch_index, variant)", flush=True)
    config = project_config()
    if config.seed != SEED:
        raise RuntimeError(f"seed {config.seed}")
    free = require_disk()
    print(f"disk_free_gib {free:.2f} n_jobs={N_JOBS}", flush=True)
    context2_path = config.models_dir / "context2_v1.joblib"
    context2_before = file_sha256(context2_path)
    if context2_before != CONTEXT2_SHA:
        raise RuntimeError(f"context2_v1 sha256 is {context2_before}")
    v1_dir = config.cache_dir / "features_v1"
    digest_before, n_files, n_bytes = _tree_digest(v1_dir)
    print(
        f"features_v1_before sha256={digest_before} files={n_files} bytes={n_bytes}",
        flush=True,
    )
    lock = _acquire_lock(config.cache_dir)
    try:
        labels = read_labels(config.tables_dir / "labels.csv")
        champion = load_champion()
        if champion["id"] != "context2_v1":
            raise RuntimeError(f"comparison base must stay context2_v1, got {champion['id']}")
        reference_macro = float(champion["macro_f1"])
        reference_subjects = {
            str(key): float(value) for key, value in champion["subject_f1"].items()
        }
        if f"{reference_macro:.6f}" != "0.759678":
            raise RuntimeError(f"reference macro {reference_macro}")
        if tuple(sorted(reference_subjects)) != SUBJECTS:
            raise RuntimeError("champion subjects drifted")
        smoke = _smoke(config, labels)
        train_raw = load_bases(config, "train", labels)
        val_raw = load_bases(config, "val", labels)
        if len(train_raw) != 112 or len(val_raw) != 16:
            raise RuntimeError("record counts drifted")
        train_records = [_prepare(record) for record in train_raw]
        val_records = [_prepare(record) for record in val_raw]
        if sum(record["raw"].shape[0] for record in train_records) != EXPECTED_EPOCHS["train"]:
            raise RuntimeError("train epoch count drifted")
        if sum(record["raw"].shape[0] for record in val_records) != EXPECTED_EPOCHS["val"]:
            raise RuntimeError("val epoch count drifted")
        cache_s = _write_cache(config, train_records + val_records)
        digest_after, n_files_after, n_bytes_after = _tree_digest(v1_dir)
        if (digest_after, n_files_after, n_bytes_after) != (digest_before, n_files, n_bytes):
            raise RuntimeError("cache/features_v1 changed during the experiment")
        print(f"features_v1_after sha256={digest_after} unchanged=1", flush=True)
        _reproduce_context2(config, val_raw)
        del train_raw, val_raw
        gc.collect()
        results: dict[str, dict[str, Any]] = {}
        for variant in ("c1", "c2"):
            try:
                results[variant] = _fit_variant(
                    config,
                    labels,
                    train_records,
                    val_records,
                    variant,
                    reference_subjects,
                    smoke["seconds"],
                    cache_s,
                )
            except TimeoutError as exc:
                print(f"STOP {variant} {exc}", flush=True)
                _log_stop(variant, f"fit=timeout;detail={exc}")
                return 3
        winner = _choose(results["c1"], results["c2"])
        _print_tables(reference_macro, reference_subjects, results, winner)
        for variant in ("c1", "c2"):
            upsert_result(
                _result_row(results[variant], _decision(variant, winner, results[variant]["passed"]))
            )
        SUMMARY_PATH.write_text(
            json.dumps(
                {
                    "formula": FORMULA_TEXT,
                    "eps": STD_EPS,
                    "ddof": STD_DDOF,
                    "gate_macro": GATE_MACRO,
                    "reference_macro": reference_macro,
                    "reference_subjects": reference_subjects,
                    "smoke": smoke,
                    "winner": winner,
                    "variants": _json_ready(results),
                    "confirmation": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        confirmation: dict[str, Any] | None = None
        if winner is not None:
            confirmation = _confirm(config, labels, train_records, results[winner])
            budget = confirmation["budget"]
            results[winner]["deterministic"] = confirmation["deterministic"]
            results[winner]["est_kaggle_minutes"] = f"{budget['est_kaggle_minutes']:.3f}"
            results[winner]["ram_peak_gb"] = f"{budget['ram_peak_gb']:.3f}"
            _save_champion(results[winner])
        for variant in ("c1", "c2"):
            extra = ""
            if winner == variant and confirmation is not None:
                seeds = confirmation["seeds"]
                budget = confirmation["budget"]
                extra = (
                    f"pred_sha={confirmation['pred_sha']};"
                    f"repeat_sha={confirmation['pred_sha_repeat']};"
                    f"seed7={seeds['7']:.6f};seed123={seeds['123']:.6f};"
                    f"seed_mean={seeds['mean']:.6f};seed_min={seeds['min']:.6f};"
                    f"seed_max={seeds['max']:.6f};seed_range={seeds['range']:.6f};"
                    f"stable={int(seeds['stable'])};"
                    f"load={budget['load_s']:.3f};feature_one={budget['feature_one_s']:.3f};"
                    f"cpu_predict={budget['predict_s']:.3f};gpu={budget['gpu']}"
                )
            upsert_result(
                _result_row(results[variant], _decision(variant, winner, results[variant]["passed"]), extra)
            )
        if file_sha256(context2_path) != context2_before:
            raise RuntimeError("context2_v1.joblib was modified")
        digest_final, _, _ = _tree_digest(v1_dir)
        if digest_final != digest_before:
            raise RuntimeError("cache/features_v1 changed before exit")
        summary = {
            "formula": FORMULA_TEXT,
            "eps": STD_EPS,
            "ddof": STD_DDOF,
            "scope": SCOPE_TEXT,
            "gate_macro": GATE_MACRO,
            "tie_macro": TIE_MACRO,
            "reference_macro": reference_macro,
            "reference_subjects": reference_subjects,
            "features_v1_sha256": digest_before,
            "features_v1_unchanged": True,
            "context2_sha256": context2_before,
            "single_record_api": [
                "zscore_record_features(features)",
                "transform_record(features, epoch_index, variant)",
            ],
            "smoke": smoke,
            "cache_s": cache_s,
            "winner": winner,
            "variants": _json_ready(results),
            "confirmation": _json_ready(confirmation),
        }
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if winner is None:
            print("DECISION rejected both; accepted best remains context2_v1", flush=True)
        else:
            item = results[winner]
            print(
                f"DECISION accepted {item['model_id']} macro_f1={item['macro_f1']:.6f}",
                flush=True,
            )
        return 0
    finally:
        release_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())

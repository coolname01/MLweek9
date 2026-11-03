"""E4: add fixed v2 features to the accepted context-±2 representation.

Val is not used for fitting. The test split is not read.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import joblib
import numpy as np

from src.context import cached_record_ids
from src.data import EXPECTED_EPOCHS, EXPECTED_RECORDS, file_sha256, read_labels
from src.expkit import (
    TRAIN_TIME_CAP_S,
    RecordBase,
    accepts,
    apply_offsets,
    disk_free_gib,
    fit_model,
    git_commit,
    load_bases,
    load_champion,
    markdown_table,
    matrix_nbytes,
    model_description,
    n_improved,
    per_class_text,
    project_config,
    require_disk,
    save_champion,
    score_arrays,
    stack_offsets,
    subject_map,
    upsert_result,
)
from src.features import _selection, acquire_lock, align_labels, release_lock, worker_count
from src.features_v2 import (
    N_V2,
    V2_NAMES,
    extract_records,
    load_v2_map,
    self_check,
)
from src.train import load_baseline

OFFSETS2: tuple[int, ...] = (-2, -1, 1, 2)
HYPOTHESIS = (
    "Spectral entropy, EEG delta/beta and theta/alpha, EEG sigma power, "
    "EOG log-power at 0.5-5 Hz, and 2 s subwindow variance, added to "
    "context ±2, improve val macro-F1."
)
FEATURES = (
    "context2 on features_v1 (offsets -2,-1,+1,+2, n_neighbors) plus current-epoch "
    "v2: normalized Welch entropy 0.5-50 Hz per channel; EEG delta/beta and "
    "theta/alpha; linear EEG sigma power 12-16 Hz; EOG log10 power 0.5-5 Hz; "
    "mean and max population variance of 15 non-overlapping 2 s blocks per channel; "
    "per-record z-score; cache features_v2"
)


def _smoke_selection(config: Any) -> list[tuple[str, str]]:
    selected = _selection(config, ("train", "val"))
    chosen = [selected[0], selected[1], selected[EXPECTED_RECORDS["train"]]]
    return [(split, path.stem) for split, path in chosen]


def _join(records: list[RecordBase], v2: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    blocks: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for record in records:
        context = apply_offsets(record.base, record.epoch_index, OFFSETS2)
        extra = v2[record.record_id]
        if extra.shape != (context.shape[0], N_V2):
            raise RuntimeError(f"{record.record_id} v2 shape {extra.shape}")
        blocks.append(np.concatenate([context, extra], axis=1))
        ids.append(record.ids)
        if names is None:
            from src.expkit import context_feature_names
            from src.features import feature_names

            names = context_feature_names(feature_names(), OFFSETS2) + V2_NAMES
    if names is None:
        raise RuntimeError("no records")
    return np.concatenate(blocks, axis=0), np.concatenate(ids, axis=0), names


def _labels_for_ids(ids: np.ndarray, labels: Any, split: str) -> np.ndarray:
    part = labels.loc[labels["split"].astype(str) == split, ["id", "label"]]
    mapping = dict(
        zip(part["id"].astype(str).tolist(), part["label"].astype(int).tolist(), strict=True)
    )
    y = np.empty(ids.shape[0], dtype=np.int64)
    for index, epoch_id in enumerate(ids.tolist()):
        y[index] = int(mapping[str(epoch_id)])
    return y


def main() -> int:
    """Extract v2 on a smoke set, then fit when the estimate is inside the cap."""
    config = project_config()
    n_jobs = worker_count()
    np.random.seed(config.seed)
    free = require_disk()
    print(f"disk_free_gib {free:.2f} n_jobs={n_jobs} seed={config.seed}", flush=True)
    print("v2 self-check: running", flush=True)
    self_check()
    print("v2 self-check: ok", flush=True)
    lock = acquire_lock(config)
    try:
        labels = read_labels(config.tables_dir / "labels.csv")
        champion = load_champion()
        if champion["id"] != "context2_v1":
            raise RuntimeError(f"E4 expected context2_v1 as the accepted model, found {champion['id']}")
        reference_macro = float(champion["macro_f1"])
        reference_subjects = {str(key): float(value) for key, value in champion["subject_f1"].items()}
        smoke = _smoke_selection(config)
        smoke_started = time.perf_counter()
        smoke_timings = []
        for split, record_id in smoke:
            smoke_timings.extend(extract_records(config, split, [record_id], n_jobs=1))
        smoke_extract_s = time.perf_counter() - smoke_started
        computed = [row for row in smoke_timings if not row.skipped]
        if len(computed) != 3:
            raise RuntimeError(
                "smoke v2 cache was already complete, so the extract time is unknown; "
                "refusing to guess"
            )
        mean_record_s = float(np.mean([row.seconds for row in computed]))
        extract_estimate_s = mean_record_s * (EXPECTED_RECORDS["train"] + EXPECTED_RECORDS["val"]) / n_jobs
        print(
            f"smoke_extract_wall_s {smoke_extract_s:.2f} mean_record_s {mean_record_s:.3f} "
            f"est_extract_min {extract_estimate_s / 60:.2f}",
            flush=True,
        )
        smoke_records: list[RecordBase] = []
        for split, record_id in smoke:
            smoke_records.extend(load_bases(config, split, labels, [record_id]))
        smoke_v2 = load_v2_map(config, smoke_records)
        train_smoke = [record for record in smoke_records if record.split == "train"]
        val_smoke = [record for record in smoke_records if record.split == "val"]
        smoke_x, smoke_ids, names = _join(train_smoke, smoke_v2)
        y_smoke = _labels_for_ids(smoke_ids, labels, "train")
        _model, smoke_fit_s = fit_model(
            smoke_x, y_smoke, config.seed, n_jobs, min(TRAIN_TIME_CAP_S, 20 * 60)
        )
        del _model
        n_cols = int(smoke_x.shape[1])
        full_gib = matrix_nbytes(EXPECTED_EPOCHS["train"], n_cols) / (1024**3)
        fit_estimate_s = smoke_fit_s * EXPECTED_EPOCHS["train"] / smoke_x.shape[0]
        print(
            f"smoke_shape {smoke_x.shape} nan {float(np.isnan(smoke_x).mean()):.6f} "
            f"smoke_fit_s {smoke_fit_s:.3f} est_fit_min {fit_estimate_s / 60:.2f} "
            f"matrix_gib {full_gib:.4f}",
            flush=True,
        )
        del smoke_x
        gc.collect()
        reasons: list[str] = []
        if full_gib > 8:
            reasons.append(f"matrix {full_gib:.3f} GiB exceeds 8 GiB")
        if fit_estimate_s > TRAIN_TIME_CAP_S:
            reasons.append(f"estimated fit {fit_estimate_s / 60:.2f} min exceeds 40 min")
        if extract_estimate_s > TRAIN_TIME_CAP_S:
            reasons.append(f"estimated extract {extract_estimate_s / 60:.2f} min exceeds 40 min")
        if reasons:
            print("SMOKE decision: STOP\nreason: " + "; ".join(reasons), flush=True)
            upsert_result(
                {
                    "id": "v2_context2_v1",
                    "hypothesis": HYPOTHESIS,
                    "features": FEATURES,
                    "model": "",
                    "val_macro_f1": "",
                    "per_class_f1": "",
                    "n_subjects_improved": "",
                    "time_s": f"smoke_fit={smoke_fit_s:.3f};stopped_before_fit",
                    "seed": str(config.seed),
                    "decision": "stopped_smoke",
                    "model_sha256": "",
                    "est_kaggle_minutes": "",
                    "ram_peak_gb": "",
                    "deterministic": "",
                    "commit": git_commit(),
                }
            )
            return 3
        print(
            "SMOKE decision: PROCEED\nreason: "
            f"extract {extract_estimate_s / 60:.2f} min, fit {fit_estimate_s / 60:.2f} min, "
            f"matrix {full_gib:.3f} GiB",
            flush=True,
        )
        print("v2 extract train+val", flush=True)
        extract_started = time.perf_counter()
        extract_records(config, "train", cached_record_ids(config, "train"), n_jobs)
        extract_records(config, "val", cached_record_ids(config, "val"), n_jobs)
        extract_s = time.perf_counter() - extract_started
        print(f"extract_wall_s {extract_s:.2f}", flush=True)
        print("matrix train", flush=True)
        build_started = time.perf_counter()
        train_records = load_bases(config, "train", labels)
        val_records = load_bases(config, "val", labels)
        train_x, train_ids, train_names = _join(train_records, load_v2_map(config, train_records))
        val_context = stack_offsets(val_records, OFFSETS2)
        val_v2 = load_v2_map(config, val_records)
        val_x, val_ids, val_names = _join(val_records, val_v2)
        build_s = time.perf_counter() - build_started
        if train_names != names or val_names != names:
            raise RuntimeError("feature names changed between smoke and full")
        if train_x.shape != (EXPECTED_EPOCHS["train"], 271 + N_V2):
            raise RuntimeError(f"train shape {train_x.shape}")
        if val_x.shape != (EXPECTED_EPOCHS["val"], 271 + N_V2):
            raise RuntimeError(f"val shape {val_x.shape}")
        if not np.array_equal(val_context.ids, val_ids):
            raise RuntimeError("val id order drifted")
        bundle = load_baseline(config.models_dir / "context2_v1.joblib")
        if tuple(bundle["feature_names"]) != val_context.feature_names:
            raise RuntimeError("context block is not the frozen context2 layout")
        y_train, _train_subjects = align_labels(train_ids, labels, "train")
        y_val, val_subjects = align_labels(val_ids, labels, "val")
        rebuilt = np.asarray(bundle["model"].predict(val_context.features), dtype=np.int64)
        rebuilt_scores = score_arrays(y_val, rebuilt)
        if abs(rebuilt_scores["macro_f1"] - reference_macro) > 1e-6:
            raise RuntimeError(
                f"context block macro-F1 {rebuilt_scores['macro_f1']:.6f} != {reference_macro:.6f}"
            )
        print(f"nested_context2_macro {rebuilt_scores['macro_f1']:.6f}", flush=True)
        limit_s = min(TRAIN_TIME_CAP_S, 2.0 * fit_estimate_s)
        print(f"fit_limit_s {limit_s:.1f}", flush=True)
        try:
            model, fit_s = fit_model(train_x, y_train, config.seed, n_jobs, limit_s)
        except TimeoutError as exc:
            print(f"STOP {exc}", flush=True)
            upsert_result(
                {
                    "id": "v2_context2_v1",
                    "hypothesis": HYPOTHESIS,
                    "features": FEATURES,
                    "model": "",
                    "val_macro_f1": "",
                    "per_class_f1": "",
                    "n_subjects_improved": "",
                    "time_s": (
                        f"mean_record_s={mean_record_s:.3f};extract={extract_s:.3f};fit=timeout"
                    ),
                    "seed": str(config.seed),
                    "decision": "stopped_time",
                    "model_sha256": "",
                    "est_kaggle_minutes": "",
                    "ram_peak_gb": "",
                    "deterministic": "",
                    "commit": git_commit(),
                }
            )
            return 3
        del train_x
        gc.collect()
        predict_started = time.perf_counter()
        predicted = np.asarray(model.predict(val_x), dtype=np.int64)
        proba = np.asarray(model.predict_proba(val_x), dtype=np.float32)
        predict_s = time.perf_counter() - predict_started
        scores = score_arrays(y_val, predicted)
        by_subject = subject_map(y_val, predicted, val_subjects)
        improved = n_improved(by_subject, reference_subjects)
        passed = accepts(scores["macro_f1"], improved, reference_macro)
        total_s = extract_s + fit_s
        estimate_total = extract_estimate_s + fit_estimate_s
        overtime = fit_s > TRAIN_TIME_CAP_S or total_s > 2.0 * estimate_total
        model_path = config.models_dir / "v2_context2_v1.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_version": "v1+v2",
                "feature_names": val_names,
                "context_offsets": list(OFFSETS2),
                "v2_names": list(V2_NAMES),
                "seed": config.seed,
                "model_name": model_description(model),
                "classes": list(config.classes),
                "train_rows": EXPECTED_EPOCHS["train"],
                "val_rows": int(val_x.shape[0]),
                "fit_seconds": fit_s,
                "n_jobs": n_jobs,
            },
            model_path,
            compress=3,
        )
        digest = file_sha256(model_path)
        np.save(config.models_dir / "v2_context2_v1_val_proba.npy", proba)
        np.save(config.models_dir / "v2_context2_v1_val_pred.npy", predicted)
        if overtime:
            decision = "stopped_time"
        elif passed:
            decision = "accepted"
        else:
            decision = "rejected"
        upsert_result(
            {
                "id": "v2_context2_v1",
                "hypothesis": HYPOTHESIS,
                "features": FEATURES,
                "model": model_description(model),
                "val_macro_f1": f"{scores['macro_f1']:.6f}",
                "per_class_f1": per_class_text(scores),
                "n_subjects_improved": str(improved),
                "time_s": (
                    f"mean_record_s={mean_record_s:.3f};smoke_fit={smoke_fit_s:.3f};"
                    f"extract={extract_s:.3f};matrix={build_s:.3f};fit={fit_s:.3f};"
                    f"predict={predict_s:.3f}"
                ),
                "seed": str(config.seed),
                "decision": decision,
                "model_sha256": digest,
                "est_kaggle_minutes": "",
                "ram_peak_gb": "",
                "deterministic": "",
                "commit": git_commit(),
            }
        )
        _print_report(
            config,
            scores,
            by_subject,
            reference_subjects,
            reference_macro,
            improved,
            passed,
            fit_s,
            predict_s,
            extract_s,
            digest,
        )
        if passed and not overtime:
            save_champion(
                {
                    "id": "v2_context2_v1",
                    "macro_f1": scores["macro_f1"],
                    "per_class_f1": [float(value) for value in scores["per_class_f1"]],
                    "accuracy": scores["accuracy"],
                    "subject_f1": by_subject,
                    "offsets": list(OFFSETS2),
                    "base": "features_v1",
                    "extras": ["v2_current"],
                    "model_path": "models/v2_context2_v1.joblib",
                    "feature_names": list(val_names),
                    "n_features": int(val_x.shape[1]),
                }
            )
        if overtime:
            print(
                f"STOP total {total_s:.1f}s vs estimate {estimate_total:.1f}s "
                f"or fit {fit_s:.1f}s",
                flush=True,
            )
            return 3
        _ = val_smoke
        return 0
    finally:
        release_lock(lock)


def _print_report(
    config: Any,
    scores: dict[str, Any],
    by_subject: dict[str, float],
    reference: dict[str, float],
    reference_macro: float,
    improved: int,
    passed: bool,
    fit_s: float,
    predict_s: float,
    extract_s: float,
    digest: str,
) -> None:
    classes = list(config.classes)
    per_class = scores["per_class_f1"]
    print(
        markdown_table(
            ["model", "macro-F1", *classes, "accuracy"],
            [
                ["context2_v1", f"{reference_macro:.4f}", "", "", "", "", "", ""],
                [
                    "v2_context2_v1",
                    f"{scores['macro_f1']:.4f}",
                    *[f"{float(per_class[index]):.4f}" for index in range(5)],
                    f"{scores['accuracy']:.4f}",
                ],
            ],
        ),
        flush=True,
    )
    matrix = scores["confusion_matrix"]
    print(
        markdown_table(
            ["true\\pred", *classes],
            [
                [classes[index], *[str(int(value)) for value in matrix[index]]]
                for index in range(5)
            ],
        ),
        flush=True,
    )
    rows = []
    for subject in sorted(by_subject):
        delta = by_subject[subject] - reference[subject]
        rows.append(
            [
                subject,
                f"{reference[subject]:.4f}",
                f"{by_subject[subject]:.4f}",
                f"{delta:+.4f}",
                "yes" if delta > 0 else "no",
            ]
        )
    print(markdown_table(["subject_id", "context2_v1", "v2_context2_v1", "delta", "improved"], rows), flush=True)
    print(
        "\n".join(
            [
                f"macro_f1 {scores['macro_f1']:.6f}",
                f"delta_vs_best {scores['macro_f1'] - reference_macro:+.6f}",
                f"subjects_improved {improved}/8",
                f"decision {'accepted' if passed else 'rejected'}",
                f"extract_s {extract_s:.2f}",
                f"fit_s {fit_s:.2f}",
                f"predict_s {predict_s:.2f}",
                f"sha256 {digest}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

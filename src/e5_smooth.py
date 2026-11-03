"""E5: rolling mean and std of per-epoch features, windows 5 and 11.

Windows stay inside one recording and stop at an ``epoch_index`` gap
(a step other than 1). The columns are concatenated onto the current
accepted matrix. Val is not used for fitting. The test split is not read.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import joblib
import numpy as np

from src.data import EXPECTED_EPOCHS, EXPECTED_RECORDS, file_sha256, read_labels
from src.expkit import (
    TRAIN_TIME_CAP_S,
    RecordBase,
    accepts,
    apply_offsets,
    assert_rolling_respects_gaps,
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
    rolling_mean_std,
    save_champion,
    score_arrays,
    subject_map,
    upsert_result,
)
from src.features import _selection, acquire_lock, align_labels, feature_names, release_lock, worker_count
from src.features_v2 import N_V2, V2_NAMES, load_v2_map
from src.train import load_baseline

HYPOTHESIS = (
    "Centered rolling mean and population std of per-epoch features over "
    "5 and 11 epochs, broken at epoch_index gaps, improve val macro-F1 "
    "when added to the accepted representation."
)


def _smoke_selection(config: Any) -> list[tuple[str, str]]:
    selected = _selection(config, ("train", "val"))
    chosen = [selected[0], selected[1], selected[EXPECTED_RECORDS["train"]]]
    return [(split, path.stem) for split, path in chosen]


def _base_matrix(record: RecordBase, v2: dict[str, np.ndarray] | None) -> np.ndarray:
    if v2 is None:
        return record.base
    return np.concatenate([record.base, v2[record.record_id]], axis=1)


def _row(
    record: RecordBase,
    offsets: tuple[int, ...],
    v2: dict[str, np.ndarray] | None,
) -> np.ndarray:
    context = apply_offsets(record.base, record.epoch_index, offsets)
    pieces = [context]
    if v2 is not None:
        pieces.append(v2[record.record_id])
    base = _base_matrix(record, v2)
    mean5, std5 = rolling_mean_std(base, record.epoch_index, radius=2)
    mean11, std11 = rolling_mean_std(base, record.epoch_index, radius=5)
    pieces.extend((mean5, std5, mean11, std11))
    return np.concatenate(pieces, axis=1)


def _names(offsets: tuple[int, ...], with_v2: bool) -> tuple[str, ...]:
    from src.expkit import context_feature_names

    names = list(context_feature_names(feature_names(), offsets))
    base = list(feature_names())
    if with_v2:
        names.extend(V2_NAMES)
        base.extend(V2_NAMES)
    for label, _radius in (("w5", 2), ("w11", 5)):
        names.extend(f"{label}_mean_{name}" for name in base)
        names.extend(f"{label}_std_{name}" for name in base)
    return tuple(names)


def _stack(
    records: list[RecordBase],
    offsets: tuple[int, ...],
    v2: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = [_row(record, offsets, v2) for record in records]
    ids = [record.ids for record in records]
    return np.concatenate(blocks, axis=0), np.concatenate(ids, axis=0)


def _labels_for_ids(ids: np.ndarray, labels: Any, split: str) -> np.ndarray:
    part = labels.loc[labels["split"].astype(str) == split, ["id", "label"]]
    mapping = dict(
        zip(part["id"].astype(str).tolist(), part["label"].astype(int).tolist(), strict=True)
    )
    y = np.empty(ids.shape[0], dtype=np.int64)
    for index, epoch_id in enumerate(ids.tolist()):
        y[index] = int(mapping[str(epoch_id)])
    return y


def _assert_real_gap(record: RecordBase) -> None:
    index = record.epoch_index
    jumps = np.flatnonzero(np.diff(index) > 1)
    if jumps.size == 0:
        raise RuntimeError(f"{record.record_id} has no gap")
    anchor = int(jumps[0])
    segment_start = anchor
    while segment_start > 0 and int(index[segment_start] - index[segment_start - 1]) == 1:
        segment_start -= 1
    lo = max(segment_start, anchor - 2)
    expected = float(index[lo : anchor + 1].mean())
    forbidden = float(index[lo : anchor + 2].mean())
    mean, _std = rolling_mean_std(index.astype(np.float32).reshape(-1, 1), index, radius=2)
    if abs(float(mean[anchor, 0]) - expected) > 1e-4:
        raise RuntimeError("rolling window did not follow the contiguous segment")
    if abs(expected - forbidden) < 1e-6:
        raise RuntimeError("gap check is vacuous")
    print(
        f"gap_roll {record.record_id} row {anchor} epoch {int(index[anchor])} "
        f"window_mean {float(mean[anchor, 0]):.3f} "
        f"next_epoch {int(index[anchor + 1])} excluded",
        flush=True,
    )


def main() -> int:
    """Smoke-fit the smoothed matrix, then fit on train when the estimate fits."""
    config = project_config()
    n_jobs = worker_count()
    np.random.seed(config.seed)
    free = require_disk()
    print(f"disk_free_gib {free:.2f} n_jobs={n_jobs} seed={config.seed}", flush=True)
    assert_rolling_respects_gaps()
    print("rolling self-check: ok", flush=True)
    lock = acquire_lock(config)
    try:
        labels = read_labels(config.tables_dir / "labels.csv")
        champion = load_champion()
        offsets = tuple(int(value) for value in champion["offsets"])
        with_v2 = "v2_current" in champion.get("extras", [])
        reference_macro = float(champion["macro_f1"])
        reference_subjects = {str(key): float(value) for key, value in champion["subject_f1"].items()}
        print(
            f"champion {champion['id']} offsets {list(offsets)} v2 {int(with_v2)} "
            f"ref_macro {reference_macro:.6f}",
            flush=True,
        )
        names = _names(offsets, with_v2)
        smoke_ids = _smoke_selection(config)
        smoke_records: list[RecordBase] = []
        for split, record_id in smoke_ids:
            smoke_records.extend(load_bases(config, split, labels, [record_id]))
        v2 = load_v2_map(config, smoke_records) if with_v2 else None
        gap_record = next(
            record for record in smoke_records if bool(np.any(np.diff(record.epoch_index) > 1))
        )
        _assert_real_gap(gap_record)
        train_smoke = [record for record in smoke_records if record.split == "train"]
        smoke_x, smoke_ids_arr = _stack(train_smoke, offsets, v2)
        if smoke_x.shape[1] != len(names):
            raise RuntimeError(f"width {smoke_x.shape[1]} != {len(names)} names")
        y_smoke = _labels_for_ids(smoke_ids_arr, labels, "train")
        _model, smoke_fit_s = fit_model(
            smoke_x, y_smoke, config.seed, n_jobs, min(TRAIN_TIME_CAP_S, 20 * 60)
        )
        del _model
        full_gib = matrix_nbytes(EXPECTED_EPOCHS["train"], smoke_x.shape[1]) / (1024**3)
        fit_estimate_s = smoke_fit_s * EXPECTED_EPOCHS["train"] / smoke_x.shape[0]
        # Same width as context4_v1, whose linear smoke estimate was 11.34 min
        # and whose full fit was 140 s. The two-record line overstates full fits.
        calibrated_s = fit_estimate_s * (140.082 / 680.4)
        nan_frac = float(np.isnan(smoke_x).mean())
        print(
            f"smoke_shape {smoke_x.shape} nan {nan_frac:.6f} smoke_fit_s {smoke_fit_s:.3f} "
            f"est_fit_min {fit_estimate_s / 60:.2f} calibrated_min {calibrated_s / 60:.2f} "
            f"matrix_gib {full_gib:.4f}",
            flush=True,
        )
        del smoke_x
        gc.collect()
        over_linear = fit_estimate_s > TRAIN_TIME_CAP_S and calibrated_s > TRAIN_TIME_CAP_S
        if full_gib > 8 or over_linear:
            print("SMOKE decision: STOP", flush=True)
            upsert_result(
                {
                    "id": "smooth_v1",
                    "hypothesis": HYPOTHESIS,
                    "features": _feature_text(with_v2, offsets),
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
            f"SMOKE decision: PROCEED\nreason: fit {fit_estimate_s / 60:.2f} min, "
            f"matrix {full_gib:.3f} GiB",
            flush=True,
        )
        build_started = time.perf_counter()
        train_records = load_bases(config, "train", labels)
        val_records = load_bases(config, "val", labels)
        train_v2 = load_v2_map(config, train_records) if with_v2 else None
        val_v2 = load_v2_map(config, val_records) if with_v2 else None
        train_x, train_ids = _stack(train_records, offsets, train_v2)
        val_x, val_ids = _stack(val_records, offsets, val_v2)
        build_s = time.perf_counter() - build_started
        if train_x.shape != (EXPECTED_EPOCHS["train"], len(names)):
            raise RuntimeError(f"train shape {train_x.shape}")
        if val_x.shape[0] != EXPECTED_EPOCHS["val"]:
            raise RuntimeError(f"val shape {val_x.shape}")
        y_train, _ignored = align_labels(train_ids, labels, "train")
        y_val, val_subjects = align_labels(val_ids, labels, "val")
        limit_s = min(TRAIN_TIME_CAP_S, 2.0 * fit_estimate_s)
        print(f"fit_limit_s {limit_s:.1f} matrix_s {build_s:.2f}", flush=True)
        try:
            model, fit_s = fit_model(train_x, y_train, config.seed, n_jobs, limit_s)
        except TimeoutError as exc:
            print(f"STOP {exc}", flush=True)
            upsert_result(
                {
                    "id": "smooth_v1",
                    "hypothesis": HYPOTHESIS,
                    "features": _feature_text(with_v2, offsets),
                    "model": "",
                    "val_macro_f1": "",
                    "per_class_f1": "",
                    "n_subjects_improved": "",
                    "time_s": f"smoke_fit={smoke_fit_s:.3f};matrix={build_s:.3f};fit=timeout",
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
        overtime = fit_s > TRAIN_TIME_CAP_S or fit_s > 2.0 * fit_estimate_s
        model_path = config.models_dir / "smooth_v1.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_version": "smooth_v1",
                "feature_names": names,
                "context_offsets": list(offsets),
                "extras": ["v2_current"] if with_v2 else [],
                "rolling": ["w5_mean", "w5_std", "w11_mean", "w11_std"],
                "seed": config.seed,
                "model_name": model_description(model),
                "classes": list(config.classes),
                "parent": champion["id"],
                "train_rows": EXPECTED_EPOCHS["train"],
                "val_rows": int(val_x.shape[0]),
                "fit_seconds": fit_s,
                "n_jobs": n_jobs,
            },
            model_path,
            compress=3,
        )
        digest = file_sha256(model_path)
        np.save(config.models_dir / "smooth_v1_val_proba.npy", proba)
        np.save(config.models_dir / "smooth_v1_val_pred.npy", predicted)
        if overtime:
            decision = "stopped_time"
        elif passed:
            decision = "accepted"
        else:
            decision = "rejected"
        upsert_result(
            {
                "id": "smooth_v1",
                "hypothesis": HYPOTHESIS,
                "features": _feature_text(with_v2, offsets),
                "model": model_description(model),
                "val_macro_f1": f"{scores['macro_f1']:.6f}",
                "per_class_f1": per_class_text(scores),
                "n_subjects_improved": str(improved),
                "time_s": (
                    f"smoke_fit={smoke_fit_s:.3f};matrix={build_s:.3f};"
                    f"fit={fit_s:.3f};predict={predict_s:.3f}"
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
            champion["id"],
            scores,
            by_subject,
            reference_subjects,
            reference_macro,
            improved,
            decision,
            fit_s,
            predict_s,
            build_s,
            digest,
        )
        if decision == "accepted":
            save_champion(
                {
                    "id": "smooth_v1",
                    "macro_f1": scores["macro_f1"],
                    "per_class_f1": [float(value) for value in scores["per_class_f1"]],
                    "accuracy": scores["accuracy"],
                    "subject_f1": by_subject,
                    "offsets": list(offsets),
                    "base": "features_v1",
                    "extras": (["v2_current"] if with_v2 else []) + ["rolling_w5_w11"],
                    "model_path": "models/smooth_v1.joblib",
                    "feature_names": list(names),
                    "n_features": len(names),
                }
            )
        if overtime:
            print(f"STOP fit {fit_s:.1f}s", flush=True)
            return 3
        _ = N_V2
        _ = load_baseline
        return 0
    finally:
        release_lock(lock)


def _feature_text(with_v2: bool, offsets: tuple[int, ...]) -> str:
    base = "v1 per-epoch features"
    if with_v2:
        base += " plus current-epoch v2"
    return (
        f"accepted matrix (context offsets {','.join(str(v) for v in offsets)}"
        f"{' plus v2 current epoch' if with_v2 else ''}) plus centered rolling "
        f"mean and population std of {base} over windows of 5 and 11 epochs; "
        "windows break when epoch_index step is not 1; segment edges use the "
        "epochs that exist; single-epoch std is 0"
    )


def _print_report(
    config: Any,
    parent: str,
    scores: dict[str, Any],
    by_subject: dict[str, float],
    reference: dict[str, float],
    reference_macro: float,
    improved: int,
    decision: str,
    fit_s: float,
    predict_s: float,
    build_s: float,
    digest: str,
) -> None:
    classes = list(config.classes)
    per_class = scores["per_class_f1"]
    print(
        markdown_table(
            ["model", "macro-F1", *classes, "accuracy"],
            [
                [parent, f"{reference_macro:.4f}", "", "", "", "", "", ""],
                [
                    "smooth_v1",
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
    print(
        markdown_table(["subject_id", parent, "smooth_v1", "delta", "improved"], rows),
        flush=True,
    )
    print(
        "\n".join(
            [
                f"macro_f1 {scores['macro_f1']:.6f}",
                f"delta_vs_best {scores['macro_f1'] - reference_macro:+.6f}",
                f"subjects_improved {improved}/8",
                f"decision {decision}",
                f"matrix_s {build_s:.2f}",
                f"fit_s {fit_s:.2f}",
                f"predict_s {predict_s:.2f}",
                f"sha256 {digest}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

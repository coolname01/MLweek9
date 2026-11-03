"""E3: same-record context at epoch_index offsets -4..-1 and +1..+4.

The v1 cache is not recomputed. Val is scored only. The test split is not
read. Acceptance is against the current best, ``context2_v1``.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import joblib
import numpy as np

from src.context import read_record_index
from src.data import EXPECTED_EPOCHS, EXPECTED_RECORDS, file_sha256, read_labels
from src.expkit import (
    ACCEPT_DELTA,
    TRAIN_TIME_CAP_S,
    RecordBase,
    SplitMatrix,
    accepts,
    apply_offsets,
    assert_matches_context2,
    default_model,
    disk_free_gib,
    fit_model,
    git_commit,
    load_bases,
    markdown_table,
    matrix_nbytes,
    model_description,
    n_improved,
    per_class_text,
    prediction_sha,
    project_config,
    require_disk,
    save_champion,
    score_arrays,
    stack_offsets,
    subject_map,
    upsert_result,
    worker_count,
)
from src.features import FEATURE_VERSION, _selection, acquire_lock, align_labels, release_lock
from src.train import load_baseline, score_predictions

OFFSETS4: tuple[int, ...] = (-4, -3, -2, -1, 1, 2, 3, 4)
REFERENCE_MACRO = 0.759678
REFERENCE_SUBJECTS = {
    "SC404": 0.8064,
    "SC407": 0.8699,
    "SC417": 0.6785,
    "SC424": 0.7773,
    "SC426": 0.7801,
    "SC427": 0.7674,
    "SC457": 0.5716,
    "SC459": 0.6109,
}
HYPOTHESIS = (
    "A wider same-record window, epoch_index offsets -4..-1 and +1..+4, "
    "improves val macro-F1 over context offsets limited to ±2."
)
FEATURES = (
    "features_v1 cache, not recomputed; same-record neighbor features at "
    "epoch_index offsets -4,-3,-2,-1,+1,+2,+3,+4 (NaN when that index is absent); "
    "n_neighbors in 0..8"
)


def _smoke_ids(config: Any) -> list[tuple[str, str, Any]]:
    selected = _selection(config, ("train", "val"))
    chosen = [selected[0], selected[1], selected[EXPECTED_RECORDS["train"]]]
    return [(split, path.stem, path) for split, path in chosen]


def _gap_table(record: RecordBase, matrix: np.ndarray) -> str:
    index = record.epoch_index
    jumps = np.flatnonzero(np.diff(index) > 1)
    if jumps.size == 0:
        raise RuntimeError(f"{record.record_id} has no epoch_index gap")
    anchor = int(jumps[0])
    start = max(0, min(anchor - 2, index.shape[0] - 5))
    rows = list(range(start, start + 5))
    if anchor not in rows or anchor + 1 not in rows:
        raise RuntimeError("gap window missed the jump")
    lookup = {int(epoch): row for row, epoch in enumerate(index.tolist())}
    width = record.base.shape[1]
    body: list[list[str]] = []
    for row in rows:
        sources: list[str] = []
        for slot, offset in enumerate(OFFSETS4):
            source = lookup.get(int(index[row]) + int(offset))
            block = matrix[row, width * (slot + 1) : width * (slot + 2)]
            if source is None:
                if not np.isnan(block).all():
                    raise RuntimeError(f"offset {offset} filled a missing epoch")
                sources.append("nan")
            else:
                if not np.array_equal(block, record.base[source]):
                    raise RuntimeError(f"offset {offset} copied the wrong epoch")
                sources.append(str(int(index[source])))
        body.append(
            [
                record.record_id,
                str(row),
                str(int(index[row])),
                *sources,
                str(int(matrix[row, -1])),
                str(int(index[row + 1])) if row + 1 < index.shape[0] else "nan",
            ]
        )
    step = int(index[anchor + 1] - index[anchor])
    anchor_sources = body[rows.index(anchor)]
    plus1 = anchor_sources[3 + OFFSETS4.index(1)]
    if plus1 != "nan":
        raise RuntimeError("offset +1 crossed an epoch_index gap")
    if step == 2:
        plus2 = anchor_sources[3 + OFFSETS4.index(2)]
        if plus2 != str(int(index[anchor + 1])):
            raise RuntimeError("offset +2 missed epoch_index+2 on a one-slot gap")
    headers = ["record", "row", "epoch_index", *[f"off{k:+d}" for k in OFFSETS4], "n", "next_epoch"]
    return markdown_table(headers, body)


def _labels_for_ids(ids: np.ndarray, labels: Any, split: str) -> np.ndarray:
    part = labels.loc[labels["split"].astype(str) == split, ["id", "label"]]
    mapping = dict(
        zip(part["id"].astype(str).tolist(), part["label"].astype(int).tolist(), strict=True)
    )
    y = np.empty(ids.shape[0], dtype=np.int64)
    for index, epoch_id in enumerate(ids.tolist()):
        value = int(mapping[str(epoch_id)])
        if value not in (0, 1, 2, 3, 4):
            raise ValueError(f"label {value} for {epoch_id}")
        y[index] = value
    return y


def run_smoke(config: Any, labels: Any, n_jobs: int) -> tuple[bool, str, float, int]:
    """Smoke on 2 train recordings and 1 val recording.

    Returns
    -------
    proceed, reason, smoke_fit_s, smoke_rows
    """
    print("e3 self-check: running", flush=True)
    toy_index = np.array([10, 11, 12, 14, 15], dtype=np.int64)
    assert_matches_context2(np.arange(5 * 54, dtype=np.float32).reshape(5, 54), toy_index)
    print("e3 self-check: ±2 matches src.context.apply_context", flush=True)
    chosen = _smoke_ids(config)
    loaded: list[RecordBase] = []
    for split, record_id, signal_path in chosen:
        group = [record for record in load_bases(config, split, labels, [record_id])]
        record = group[0]
        npz_ids, npz_epoch = read_record_index(signal_path)
        if not np.array_equal(record.ids, npz_ids) or not np.array_equal(record.epoch_index, npz_epoch):
            raise RuntimeError(f"{record_id} label epoch_index differs from the NPZ")
        assert_matches_context2(record.base, record.epoch_index)
        loaded.append(record)
        print(
            f"smoke {split}/{record_id} epochs={record.base.shape[0]} "
            f"epoch {int(record.epoch_index.min())}..{int(record.epoch_index.max())}",
            flush=True,
        )
    train_records = [record for record in loaded if record.split == "train"]
    val_records = [record for record in loaded if record.split == "val"]
    if len(train_records) != 2 or len(val_records) != 1:
        raise RuntimeError("smoke selection is not 2 train and 1 val")
    try:
        apply_offsets(
            np.concatenate([record.base for record in train_records], axis=0),
            np.concatenate([record.epoch_index for record in train_records], axis=0),
            OFFSETS4,
        )
    except ValueError as exc:
        print(f"glue_refused {[record.record_id for record in train_records]}: {exc}", flush=True)
    else:
        raise RuntimeError("stacking two train recordings was accepted")
    gap_record = next(
        record
        for record in loaded
        if bool(np.any(np.diff(record.epoch_index) > 1))
    )
    gap_matrix = apply_offsets(gap_record.base, gap_record.epoch_index, OFFSETS4)
    print(f"gap_record {gap_record.split}/{gap_record.record_id}", flush=True)
    print(_gap_table(gap_record, gap_matrix), flush=True)
    train_matrix = stack_offsets(train_records, OFFSETS4)
    val_matrix = stack_offsets(val_records, OFFSETS4)
    print(
        f"smoke_shape train {train_matrix.features.shape} val {val_matrix.features.shape} "
        f"dtype={train_matrix.features.dtype} "
        f"nan_train {float(np.isnan(train_matrix.features).mean()):.6f} "
        f"nan_val {float(np.isnan(val_matrix.features).mean()):.6f}",
        flush=True,
    )
    n_cols = int(train_matrix.features.shape[1])
    smoke_rows = int(train_matrix.features.shape[0])
    full_bytes = matrix_nbytes(EXPECTED_EPOCHS["train"], n_cols)
    full_gib = full_bytes / (1024**3)
    y_smoke = np.concatenate(
        [_labels_for_ids(record.ids, labels, "train") for record in train_records]
    )
    limit = min(TRAIN_TIME_CAP_S, 20.0 * 60.0)
    _model, smoke_fit_s = fit_model(train_matrix.features, y_smoke, config.seed, n_jobs, limit)
    estimate_s = smoke_fit_s * EXPECTED_EPOCHS["train"] / train_matrix.features.shape[0]
    print(
        "\n".join(
            [
                f"smoke_rows {train_matrix.features.shape[0]}",
                f"smoke_cols {n_cols}",
                f"smoke_fit_s {smoke_fit_s:.3f}",
                f"est_fit_s {estimate_s:.1f}",
                f"est_fit_min {estimate_s / 60:.2f}",
                f"full_train_matrix_gib {full_gib:.4f}",
                f"full_train_matrix_bytes {full_bytes}",
                f"disk_free_gib {disk_free_gib():.2f}",
            ]
        ),
        flush=True,
    )
    reasons: list[str] = []
    if full_gib > 8:
        reasons.append(f"full train matrix {full_gib:.3f} GiB exceeds 8 GiB")
    if estimate_s > TRAIN_TIME_CAP_S:
        reasons.append(f"estimated fit {estimate_s / 60:.2f} min exceeds 40 min")
    del train_matrix, val_matrix, gap_matrix
    gc.collect()
    if reasons:
        return False, "; ".join(reasons), smoke_fit_s, smoke_rows
    return (
        True,
        f"estimate {estimate_s / 60:.2f} min and matrix {full_gib:.3f} GiB are inside the limits",
        smoke_fit_s,
        smoke_rows,
    )


def _verify_reference(
    config: Any,
    labels: Any,
    val_records: list[RecordBase],
    y_val: np.ndarray,
    subjects: np.ndarray,
) -> dict[str, float]:
    """Rebuild ±2 features and require the saved context2 scores."""
    rebuilt = stack_offsets(val_records, (-2, -1, 1, 2))
    if rebuilt.features.shape != (EXPECTED_EPOCHS["val"], 271):
        raise RuntimeError(f"rebuilt context2 shape {rebuilt.features.shape}")
    bundle = load_baseline(config.models_dir / "context2_v1.joblib")
    # load_baseline checks feature_version == v1, which context2 stored.
    if tuple(bundle["feature_names"]) != rebuilt.feature_names:
        raise RuntimeError("rebuilt ±2 names differ from context2_v1")
    predicted = np.asarray(bundle["model"].predict(rebuilt.features), dtype=np.int64)
    scores = score_predictions(y_val, predicted)
    if abs(scores["macro_f1"] - REFERENCE_MACRO) > 1e-6:
        raise RuntimeError(
            f"rebuilt context2 macro-F1 {scores['macro_f1']:.6f} != {REFERENCE_MACRO:.6f}"
        )
    by_subject = subject_map(y_val, predicted, subjects)
    for subject, expected in REFERENCE_SUBJECTS.items():
        got = round(by_subject[subject], 4)
        if abs(got - expected) > 1e-9:
            raise RuntimeError(f"{subject} rebuilt {got:.4f} != logged {expected:.4f}")
    proba = np.asarray(bundle["model"].predict_proba(rebuilt.features), dtype=np.float32)
    out_dir = config.models_dir
    np.save(out_dir / "context2_v1_val_proba.npy", proba)
    np.save(out_dir / "context2_v1_val_pred.npy", predicted)
    print(
        f"reference_context2_macro {scores['macro_f1']:.6f} "
        f"pred_sha {prediction_sha(rebuilt.ids, predicted)}",
        flush=True,
    )
    save_champion(
        {
            "id": "context2_v1",
            "macro_f1": scores["macro_f1"],
            "per_class_f1": [float(value) for value in scores["per_class_f1"]],
            "accuracy": scores["accuracy"],
            "subject_f1": by_subject,
            "offsets": [-2, -1, 1, 2],
            "base": "features_v1",
            "model_path": "models/context2_v1.joblib",
            "feature_names": list(rebuilt.feature_names),
            "n_features": 271,
        }
    )
    return by_subject


def run_full(config: Any, labels: Any, n_jobs: int, smoke_fit_s: float, smoke_rows: int) -> int:
    """Fit context ±4 on train and score val."""
    print("context_build train", flush=True)
    started = time.perf_counter()
    train_records = load_bases(config, "train", labels)
    val_records = load_bases(config, "val", labels)
    train = stack_offsets(train_records, OFFSETS4)
    val = stack_offsets(val_records, OFFSETS4)
    build_s = time.perf_counter() - started
    if train.features.shape != (EXPECTED_EPOCHS["train"], 54 * 9 + 1):
        raise RuntimeError(f"train shape {train.features.shape}")
    if val.features.shape != (EXPECTED_EPOCHS["val"], train.features.shape[1]):
        raise RuntimeError(f"val shape {val.features.shape}")
    y_train, _train_subjects = align_labels(train.ids, labels, "train")
    y_val, val_subjects = align_labels(val.ids, labels, "val")
    reference = _verify_reference(config, labels, val_records, y_val, val_subjects)
    estimate_s = smoke_fit_s * EXPECTED_EPOCHS["train"] / smoke_rows
    limit_s = min(TRAIN_TIME_CAP_S, 2.0 * estimate_s)
    print(f"fit_limit_s {limit_s:.1f}", flush=True)
    _probe, probe_name = default_model(config.seed, n_jobs)
    del _probe
    print(f"classifier_template {probe_name}", flush=True)
    try:
        model, fit_s = fit_model(train.features, y_train, config.seed, n_jobs, limit_s)
    except TimeoutError as exc:
        print(f"STOP {exc}", flush=True)
        upsert_result(
            {
                "id": "context4_v1",
                "hypothesis": HYPOTHESIS,
                "features": FEATURES,
                "model": probe_name,
                "val_macro_f1": "",
                "per_class_f1": "",
                "n_subjects_improved": "",
                "time_s": f"smoke_fit={smoke_fit_s:.3f};context_build={build_s:.3f};fit=timeout",
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
    overtime = fit_s > 2.0 * estimate_s or fit_s > TRAIN_TIME_CAP_S
    predict_started = time.perf_counter()
    predicted = np.asarray(model.predict(val.features), dtype=np.int64)
    proba = np.asarray(model.predict_proba(val.features), dtype=np.float32)
    predict_s = time.perf_counter() - predict_started
    scores = score_arrays(y_val, predicted)
    by_subject = subject_map(y_val, predicted, val_subjects)
    improved = n_improved(by_subject, reference)
    passed = accepts(scores["macro_f1"], improved, REFERENCE_MACRO)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.models_dir / "context4_v1.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_version": FEATURE_VERSION,
            "feature_names": val.feature_names,
            "context_offsets": list(OFFSETS4),
            "seed": config.seed,
            "model_name": model_description(model),
            "classes": list(config.classes),
            "train_rows": int(train.features.shape[0]),
            "val_rows": int(val.features.shape[0]),
            "fit_seconds": fit_s,
            "n_jobs": n_jobs,
        },
        model_path,
        compress=3,
    )
    digest = file_sha256(model_path)
    np.save(config.models_dir / "context4_v1_val_proba.npy", proba)
    np.save(config.models_dir / "context4_v1_val_pred.npy", predicted)
    if overtime:
        decision = "stopped_time"
    elif passed:
        decision = "accepted"
    else:
        decision = "rejected"
    upsert_result(
        {
            "id": "context4_v1",
            "hypothesis": HYPOTHESIS,
            "features": FEATURES,
            "model": model_description(model),
            "val_macro_f1": f"{scores['macro_f1']:.6f}",
            "per_class_f1": per_class_text(scores),
            "n_subjects_improved": str(improved),
            "time_s": (
                f"smoke_fit={smoke_fit_s:.3f};context_build={build_s:.3f};"
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
    _print_report(config, scores, by_subject, reference, improved, passed, fit_s, predict_s, build_s, digest)
    if passed and not overtime:
        save_champion(
            {
                "id": "context4_v1",
                "macro_f1": scores["macro_f1"],
                "per_class_f1": [float(value) for value in scores["per_class_f1"]],
                "accuracy": scores["accuracy"],
                "subject_f1": by_subject,
                "offsets": list(OFFSETS4),
                "base": "features_v1",
                "model_path": "models/context4_v1.joblib",
                "feature_names": list(val.feature_names),
                "n_features": int(val.features.shape[1]),
            }
        )
    del train
    gc.collect()
    if overtime:
        print(f"STOP fit {fit_s:.1f}s exceeded the smoke gate or 40 min", flush=True)
        return 3
    return 0


def _print_report(
    config: Any,
    scores: dict[str, Any],
    by_subject: dict[str, float],
    reference: dict[str, float],
    improved: int,
    passed: bool,
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
                [
                    "context2_v1",
                    f"{REFERENCE_MACRO:.4f}",
                    "0.9692",
                    "0.4418",
                    "0.8151",
                    "0.7797",
                    "0.7926",
                    "",
                ],
                [
                    "context4_v1",
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
        markdown_table(["subject_id", "context2_v1", "context4_v1", "delta", "improved"], rows),
        flush=True,
    )
    print(
        "\n".join(
            [
                f"macro_f1 {scores['macro_f1']:.6f}",
                f"delta_vs_context2 {scores['macro_f1'] - REFERENCE_MACRO:+.6f}",
                f"subjects_improved {improved}/8",
                f"criterion_macro {scores['macro_f1']:.6f} >= {REFERENCE_MACRO + ACCEPT_DELTA:.6f}: "
                f"{'yes' if scores['macro_f1'] >= REFERENCE_MACRO + ACCEPT_DELTA else 'no'}",
                f"criterion_subjects {improved} >= 5: {'yes' if improved >= 5 else 'no'}",
                f"decision {'accepted' if passed else 'rejected'}",
                f"context_build_s {build_s:.2f}",
                f"fit_s {fit_s:.2f}",
                f"predict_s {predict_s:.2f}",
                f"sha256 {digest}",
            ]
        ),
        flush=True,
    )


def main() -> int:
    """Smoke, then fit when the estimate is inside the caps."""
    config = project_config()
    n_jobs = worker_count()
    np.random.seed(config.seed)
    free = require_disk()
    print(f"disk_free_gib {free:.2f} n_jobs={n_jobs} seed={config.seed}", flush=True)
    lock = acquire_lock(config)
    try:
        labels = read_labels(config.tables_dir / "labels.csv")
        proceed, reason, smoke_fit_s, smoke_rows = run_smoke(config, labels, n_jobs)
        print(f"SMOKE decision: {'PROCEED' if proceed else 'STOP'}\nreason: {reason}", flush=True)
        if not proceed:
            upsert_result(
                {
                    "id": "context4_v1",
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
        return run_full(config, labels, n_jobs, smoke_fit_s, smoke_rows)
    finally:
        release_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())

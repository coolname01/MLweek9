"""E6: second LightGBM on out-of-fold probabilities of epochs -3..+3.

Stage 1 is the accepted model, refit inside GroupKFold by ``subject_id``
(5 folds, deterministic given row order). Stage 2 sees only those
out-of-fold probabilities and the same offsets on val, where stage 1 is
the model already trained on all of train. Val labels are not used for
fitting. The test split is not read.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from src.data import EXPECTED_EPOCHS, EXPECTED_RECORDS, file_sha256, read_labels
from src.e4_v2 import _join
from src.e5_smooth import _stack
from src.expkit import (
    TRAIN_TIME_CAP_S,
    RecordBase,
    accepts,
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
from src.features_v2 import load_v2_map

PROB_OFFSETS: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
HYPOTHESIS = (
    "A second LightGBM on out-of-fold class probabilities of the same "
    "record at epoch_index offsets -3..+3 improves val macro-F1 over the "
    "accepted single model."
)
FEATURES = (
    "stage-1 probabilities, 5 classes, at epoch_index offsets -3,-2,-1,0,+1,+2,+3 "
    "inside the same record; missing indexes are NaN; train probabilities are "
    "GroupKFold(5) by subject_id; val probabilities come from the stage-1 model "
    "trained on all train"
)


def _prob_context(proba: np.ndarray, epoch_index: np.ndarray) -> np.ndarray:
    """Neighbor probability blocks for one recording. Width is 35."""
    values = np.asarray(proba, dtype=np.float32)
    index = np.asarray(epoch_index, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != index.shape[0] or values.shape[1] != 5:
        raise ValueError(f"proba shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    n_epochs = int(values.shape[0])
    order = np.argsort(index, kind="mergesort")
    ordered = index[order]
    if n_epochs > 1 and bool(np.any(np.diff(ordered) == 0)):
        raise ValueError("duplicate epoch_index")
    out = np.empty((n_epochs, 5 * len(PROB_OFFSETS)), dtype=np.float32)
    for slot, offset in enumerate(PROB_OFFSETS):
        target = index + np.int64(offset)
        pos = np.searchsorted(ordered, target)
        valid = pos < n_epochs
        clipped = np.where(valid, pos, 0)
        matched = valid & (ordered[clipped] == target)
        block = out[:, slot * 5 : (slot + 1) * 5]
        block.fill(np.nan)
        source = order[clipped]
        block[matched] = values[source[matched]]
    return out


def _prob_matrix(records: list[RecordBase], proba: np.ndarray) -> np.ndarray:
    blocks: list[np.ndarray] = []
    cursor = 0
    for record in records:
        n_epochs = int(record.ids.shape[0])
        blocks.append(_prob_context(proba[cursor : cursor + n_epochs], record.epoch_index))
        cursor += n_epochs
    if cursor != proba.shape[0]:
        raise RuntimeError("probability rows do not match records")
    return np.concatenate(blocks, axis=0)


def _ordered_proba(model: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float32)
    classes = [int(value) for value in model.classes_]
    if classes == [0, 1, 2, 3, 4]:
        return raw
    ordered = np.zeros((raw.shape[0], 5), dtype=np.float32)
    for column, label in enumerate(classes):
        ordered[:, label] = raw[:, column]
    return ordered


def _champion_matrix(
    records: list[RecordBase],
    champion: dict[str, Any],
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    extras = list(champion.get("extras", []))
    offsets = tuple(int(value) for value in champion["offsets"])
    use_v2 = "v2_current" in extras
    use_roll = "rolling_w5_w11" in extras
    v2 = load_v2_map(config, records) if use_v2 or use_roll and use_v2 else None
    if use_roll:
        return _stack(records, offsets, v2 if use_v2 else None)
    if use_v2:
        features, ids, _names = _join(records, load_v2_map(config, records))
        return features, ids
    matrix = stack_offsets(records, offsets)
    return matrix.features, matrix.ids


def _gap_report(record: RecordBase) -> None:
    index = record.epoch_index
    jumps = np.flatnonzero(np.diff(index) > 1)
    if jumps.size == 0:
        raise RuntimeError("smoke record has no gap")
    anchor = int(jumps[0])
    fake = np.tile(np.arange(5, dtype=np.float32), (index.shape[0], 1))
    # Make each row's probability identifiable: class 0 holds the epoch index.
    fake[:, 0] = index.astype(np.float32)
    context = _prob_context(fake, index)
    # Offset +1 is slot 4 (offsets -3,-2,-1,0,+1 are slots 0..4).
    plus1 = context[anchor, 4 * 5]
    plus_next = context[anchor, 5 * 5]
    if not np.isnan(plus1):
        raise RuntimeError("probability offset +1 crossed a gap")
    step = int(index[anchor + 1] - index[anchor])
    if step == 2 and int(plus_next) != int(index[anchor + 1]):
        raise RuntimeError("probability offset +2 missed epoch_index+2")
    print(
        f"gap_proba {record.record_id} row {anchor} epoch {int(index[anchor])} "
        f"off+1 nan off+2 {plus_next}",
        flush=True,
    )


def _smoke_records(config: Any, labels: Any) -> list[RecordBase]:
    selected = _selection(config, ("train", "val"))
    chosen = [selected[0], selected[1], selected[EXPECTED_RECORDS["train"]]]
    records: list[RecordBase] = []
    for split, path in chosen:
        records.extend(load_bases(config, split, labels, [path.stem]))
    return records


def main() -> int:
    """Fit the probability stack when the five-fold estimate is inside 40 min."""
    config = project_config()
    n_jobs = worker_count()
    np.random.seed(config.seed)
    free = require_disk()
    print(f"disk_free_gib {free:.2f} n_jobs={n_jobs} seed={config.seed}", flush=True)
    lock = acquire_lock(config)
    try:
        labels = read_labels(config.tables_dir / "labels.csv")
        champion = load_champion()
        reference_macro = float(champion["macro_f1"])
        reference_subjects = {str(key): float(value) for key, value in champion["subject_f1"].items()}
        bundle = joblib.load(config.root / champion["model_path"])
        parent_fit_s = float(bundle.get("fit_seconds", 120.0))
        fold_estimate_s = 5.0 * 0.8 * parent_fit_s
        stage2_width = 5 * len(PROB_OFFSETS)
        stage2_bytes = matrix_nbytes(EXPECTED_EPOCHS["train"], stage2_width)
        print(
            f"champion {champion['id']} parent_fit_s {parent_fit_s:.1f} "
            f"est_oof_min {fold_estimate_s / 60:.2f} "
            f"stage2_gib {stage2_bytes / 1024**3:.4f}",
            flush=True,
        )
        if fold_estimate_s > TRAIN_TIME_CAP_S:
            print("SMOKE decision: STOP\nreason: five-fold estimate exceeds 40 min", flush=True)
            upsert_result(
                {
                    "id": "stack_v1",
                    "hypothesis": HYPOTHESIS,
                    "features": FEATURES,
                    "model": "",
                    "val_macro_f1": "",
                    "per_class_f1": "",
                    "n_subjects_improved": "",
                    "time_s": f"est_oof={fold_estimate_s:.3f};stopped_before_fit",
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
        smoke = _smoke_records(config, labels)
        gap_record = next(
            record for record in smoke if bool(np.any(np.diff(record.epoch_index) > 1))
        )
        _gap_report(gap_record)
        try:
            apply_ids = [record.record_id for record in smoke if record.split == "train"]
            _prob_context(
                np.zeros((sum(r.ids.shape[0] for r in smoke if r.split == "train"), 5), np.float32),
                np.concatenate([r.epoch_index for r in smoke if r.split == "train"]),
            )
        except ValueError as exc:
            print(f"glue_refused {apply_ids}: {exc}", flush=True)
        else:
            raise RuntimeError("glued probability lookup was accepted")
        print(
            "SMOKE decision: PROCEED\nreason: "
            f"OOF estimate {fold_estimate_s / 60:.2f} min from the parent fit, "
            f"stage-2 matrix {stage2_bytes / 1024**3:.3f} GiB",
            flush=True,
        )
        print("stage1 matrix", flush=True)
        build_started = time.perf_counter()
        train_records = load_bases(config, "train", labels)
        val_records = load_bases(config, "val", labels)
        train_x, train_ids = _champion_matrix(train_records, champion, config)
        val_x, val_ids = _champion_matrix(val_records, champion, config)
        build_s = time.perf_counter() - build_started
        y_train, train_subjects = align_labels(train_ids, labels, "train")
        y_val, val_subjects = align_labels(val_ids, labels, "val")
        if train_x.shape[0] != EXPECTED_EPOCHS["train"] or val_x.shape[0] != EXPECTED_EPOCHS["val"]:
            raise RuntimeError(f"shapes train {train_x.shape} val {val_x.shape}")
        print(f"oof start cols {train_x.shape[1]}", flush=True)
        oof = np.zeros((train_x.shape[0], 5), dtype=np.float32)
        filled = np.zeros(train_x.shape[0], dtype=bool)
        oof_started = time.perf_counter()
        limit_s = min(TRAIN_TIME_CAP_S, 2.0 * fold_estimate_s)
        splitter = GroupKFold(n_splits=5)
        fold_times: list[float] = []
        try:
            for fold, (train_pos, hold_pos) in enumerate(
                splitter.split(train_x, y_train, train_subjects)
            ):
                fold_model, fold_s = fit_model(
                    train_x[train_pos],
                    y_train[train_pos],
                    config.seed,
                    n_jobs,
                    limit_s,
                )
                oof[hold_pos] = _ordered_proba(fold_model, train_x[hold_pos])
                filled[hold_pos] = True
                fold_times.append(fold_s)
                print(f"fold {fold} fit_s {fold_s:.1f} hold {hold_pos.shape[0]}", flush=True)
                del fold_model
                gc.collect()
                if sum(fold_times) > TRAIN_TIME_CAP_S or sum(fold_times) > 2.0 * fold_estimate_s:
                    raise TimeoutError(
                        f"OOF accumulated {sum(fold_times):.1f}s"
                    )
        except TimeoutError as exc:
            print(f"STOP {exc}", flush=True)
            upsert_result(
                {
                    "id": "stack_v1",
                    "hypothesis": HYPOTHESIS,
                    "features": FEATURES,
                    "model": "",
                    "val_macro_f1": "",
                    "per_class_f1": "",
                    "n_subjects_improved": "",
                    "time_s": f"matrix={build_s:.3f};oof_partial={sum(fold_times):.3f};fit=timeout",
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
        if not bool(filled.all()):
            raise RuntimeError("OOF left some train rows empty")
        oof_s = time.perf_counter() - oof_started
        stage2_x = _prob_matrix(train_records, oof)
        if stage2_x.shape != (EXPECTED_EPOCHS["train"], 35):
            raise RuntimeError(f"stage2 shape {stage2_x.shape}")
        print(
            f"oof_s {oof_s:.1f} stage2_nan {float(np.isnan(stage2_x).mean()):.6f}",
            flush=True,
        )
        stage2, stage2_s = fit_model(stage2_x, y_train, config.seed, n_jobs, TRAIN_TIME_CAP_S)
        del train_x, stage2_x
        gc.collect()
        val_parent = _ordered_proba(bundle["model"], val_x)
        val_stage2 = _prob_matrix(val_records, val_parent)
        predict_started = time.perf_counter()
        predicted = np.asarray(stage2.predict(val_stage2), dtype=np.int64)
        proba = _ordered_proba(stage2, val_stage2)
        predict_s = time.perf_counter() - predict_started
        scores = score_arrays(y_val, predicted)
        by_subject = subject_map(y_val, predicted, val_subjects)
        improved = n_improved(by_subject, reference_subjects)
        passed = accepts(scores["macro_f1"], improved, reference_macro)
        total_fit = oof_s + stage2_s
        overtime = total_fit > TRAIN_TIME_CAP_S or total_fit > 2.0 * fold_estimate_s
        model_path = config.models_dir / "stack_v1.joblib"
        joblib.dump(
            {
                "model": stage2,
                "stage1_path": champion["model_path"],
                "feature_version": "stack_v1",
                "prob_offsets": list(PROB_OFFSETS),
                "seed": config.seed,
                "model_name": model_description(stage2),
                "classes": [0, 1, 2, 3, 4],
                "parent": champion["id"],
                "train_rows": EXPECTED_EPOCHS["train"],
                "val_rows": int(val_x.shape[0]),
                "fit_seconds": total_fit,
                "n_jobs": n_jobs,
            },
            model_path,
            compress=3,
        )
        digest = file_sha256(model_path)
        np.save(config.models_dir / "stack_v1_val_proba.npy", proba)
        np.save(config.models_dir / "stack_v1_val_pred.npy", predicted)
        if overtime:
            decision = "stopped_time"
        elif passed:
            decision = "accepted"
        else:
            decision = "rejected"
        upsert_result(
            {
                "id": "stack_v1",
                "hypothesis": HYPOTHESIS,
                "features": FEATURES,
                "model": "stage2 " + model_description(stage2),
                "val_macro_f1": f"{scores['macro_f1']:.6f}",
                "per_class_f1": per_class_text(scores),
                "n_subjects_improved": str(improved),
                "time_s": (
                    f"matrix={build_s:.3f};oof={oof_s:.3f};stage2_fit={stage2_s:.3f};"
                    f"predict={predict_s:.3f};folds={','.join(f'{value:.1f}' for value in fold_times)}"
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
            oof_s,
            stage2_s,
            predict_s,
            digest,
        )
        if decision == "accepted":
            save_champion(
                {
                    "id": "stack_v1",
                    "macro_f1": scores["macro_f1"],
                    "per_class_f1": [float(value) for value in scores["per_class_f1"]],
                    "accuracy": scores["accuracy"],
                    "subject_f1": by_subject,
                    "offsets": list(champion["offsets"]),
                    "base": champion.get("base", "features_v1"),
                    "extras": list(champion.get("extras", [])) + ["prob_context_-3_3"],
                    "model_path": "models/stack_v1.joblib",
                    "stage1_path": champion["model_path"],
                    "feature_names": [
                        f"off{offset:+d}_p{label}"
                        for offset in PROB_OFFSETS
                        for label in range(5)
                    ],
                    "n_features": 35,
                    "parent": champion["id"],
                }
            )
        if overtime:
            print(f"STOP oof+stage2 {total_fit:.1f}s", flush=True)
            return 3
        return 0
    finally:
        release_lock(lock)


def _print_report(
    config: Any,
    parent: str,
    scores: dict[str, Any],
    by_subject: dict[str, float],
    reference: dict[str, float],
    reference_macro: float,
    improved: int,
    decision: str,
    oof_s: float,
    stage2_s: float,
    predict_s: float,
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
                    "stack_v1",
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
    print(markdown_table(["subject_id", parent, "stack_v1", "delta", "improved"], rows), flush=True)
    print(
        "\n".join(
            [
                f"macro_f1 {scores['macro_f1']:.6f}",
                f"delta_vs_best {scores['macro_f1'] - reference_macro:+.6f}",
                f"subjects_improved {improved}/8",
                f"decision {decision}",
                f"oof_s {oof_s:.2f}",
                f"stage2_fit_s {stage2_s:.2f}",
                f"predict_s {predict_s:.2f}",
                f"sha256 {digest}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

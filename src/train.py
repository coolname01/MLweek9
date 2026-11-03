"""Train the spectral baseline, or the same model with neighbor context.

``python -m src.train`` fits ``baseline_v1``. ``python -m src.train --context 2``
adds same-record neighbor features and fits ``context2_v1``. Val epochs are
not used to fit either model. The test split is not read. ``--context 2``
reads ``cache/features_v1`` and does not recompute it.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.context import (
    N_CONTEXT_FEATURES,
    OFFSETS,
    GapRow,
    RecordContext,
    apply_context,
    assert_labels_match_record,
    gap_rows,
    label_index_for_split,
    load_context_split,
    load_record_context,
    matrix_nbytes,
    read_record_index,
    self_check,
)
from src.data import (
    EXPECTED_EPOCHS,
    EXPECTED_RECORDS,
    file_sha256,
    load_config,
    read_labels,
)
from src.features import (
    FEATURE_VERSION,
    ExtractReport,
    FeatureTable,
    _self_check,
    N_FEATURES,
    acquire_lock,
    align_labels,
    extract_records,
    load_cached_split,
    release_lock,
    worker_count,
    _selection,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_COLUMNS = [
    "id",
    "hypothesis",
    "features",
    "model",
    "val_macro_f1",
    "per_class_f1",
    "time_s",
    "seed",
    "commit",
]
CLASS_INDEXES = [0, 1, 2, 3, 4]
CONTEXT_ACCEPT_MACRO_F1 = 0.723
CONTEXT_ACCEPT_SUBJECTS = 5
CONTEXT_MATRIX_LIMIT_BYTES = 8 * (1024**3)
CONTEXT_FIT_LIMIT_S = 20.0 * 60.0
CONTEXT_FEATURE_DESCRIPTION = (
    "features_v1 cache, not recomputed; same-record neighbor features at "
    "epoch_index offsets -2,-1,+1,+2 (NaN when that index is absent); "
    "n_neighbors in 0..4"
)
FEATURE_DESCRIPTION = (
    "welch fs=100 nperseg=256 noverlap=128 density; "
    "log10 and relative power delta 0.5-4 theta 4-8 alpha 8-12 sigma 12-16 beta 16-30; "
    "mean_abs std skew excess_kurtosis range Hjorth activity mobility complexity; "
    "per-record per-channel z-score"
)


def build_model(seed: int, n_jobs: int) -> tuple[Any, str]:
    """Build the classical classifier.

    LightGBM is used when its wheel imports. There is no source build.
    If the import fails, a histogram gradient booster with balanced class
    weights is returned instead.

    Parameters
    ----------
    seed : int
        ``random_state``. Taken from the project config.
    n_jobs : int
        Threads for LightGBM. Histogram boosting uses its own OpenMP pool.

    Returns
    -------
    model : estimator
        Unfitted classifier with ``class_weight='balanced'``.
    description : str
        Short model name for the experiment log.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print(
            "lightgbm is not importable; no source build was attempted. "
            "Using HistGradientBoostingClassifier.",
            flush=True,
        )
        model = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
            max_iter=200,
            learning_rate=0.1,
            early_stopping=False,
        )
        return model, (
            "HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,"
            "class_weight=balanced,early_stopping=False)"
        )
    model = LGBMClassifier(
        boosting_type="gbdt",
        objective="multiclass",
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        class_weight="balanced",
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    return model, (
        f"LGBMClassifier(n_estimators=200,learning_rate=0.1,num_leaves=31,"
        f"class_weight=balanced,deterministic=True,n_jobs={n_jobs})"
    )


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Score five-class predictions with the competition contract.

    Parameters
    ----------
    y_true, y_pred : ndarray of int
        Epoch labels in ``{0, 1, 2, 3, 4}``.

    Returns
    -------
    dict
        ``macro_f1``, ``per_class_f1`` (length 5), ``accuracy`` and
        ``confusion_matrix`` with rows as true labels.
    """
    per_class = f1_score(
        y_true,
        y_pred,
        labels=CLASS_INDEXES,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_INDEXES,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_f1": np.asarray(per_class, dtype=np.float64),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": np.asarray(
            confusion_matrix(y_true, y_pred, labels=CLASS_INDEXES),
            dtype=np.int64,
        ),
    }


def subject_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
) -> list[tuple[str, int, str, float]]:
    """Macro-F1 of each subject, with all five classes in the average.

    Parameters
    ----------
    y_true, y_pred : ndarray of int
        Val labels and predictions.
    subjects : ndarray of str
        ``subject_id`` aligned to those rows.

    Returns
    -------
    list of tuple
        ``(subject_id, n_epochs, classes_present, macro_f1)`` sorted by id.
        A class absent from the subject contributes F1 0.
    """
    rows: list[tuple[str, int, str, float]] = []
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        present = ",".join(str(int(value)) for value in np.unique(y_true[mask]))
        score = float(
            f1_score(
                y_true[mask],
                y_pred[mask],
                labels=CLASS_INDEXES,
                average="macro",
                zero_division=0,
            )
        )
        rows.append((str(subject), int(mask.sum()), present, score))
    return rows


def load_baseline(path: Path) -> dict[str, Any]:
    """Load ``models/baseline_v1.joblib`` and check the feature contract.

    Parameters
    ----------
    path : pathlib.Path
        Joblib bundle written by :func:`main`.

    Returns
    -------
    dict
        Keys include ``model``, ``feature_names`` and ``feature_version``.
    """
    bundle = joblib.load(path)
    required = {"model", "feature_names", "feature_version", "classes"}
    missing = required.difference(bundle)
    if missing:
        raise RuntimeError(f"model bundle missing {sorted(missing)}")
    if bundle["feature_version"] != FEATURE_VERSION:
        raise RuntimeError(
            f"model version {bundle['feature_version']} != cache {FEATURE_VERSION}"
        )
    return bundle


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _git_commit(root: Path) -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "none"
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
    )
    if status.strip():
        return f"{revision}-dirty"
    return revision


def _upsert_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        frame = pd.read_csv(path, dtype=str, encoding="utf-8")
        if list(frame.columns) != RESULT_COLUMNS:
            raise RuntimeError(f"unexpected results columns {list(frame.columns)}")
    else:
        frame = pd.DataFrame(columns=RESULT_COLUMNS)
    replace = {row["id"] for row in rows}
    if len(frame):
        frame = frame.loc[~frame["id"].isin(replace)].copy()
    updated = pd.concat([frame, pd.DataFrame(rows, columns=RESULT_COLUMNS)], ignore_index=True)
    updated.to_csv(path, index=False, encoding="utf-8")


def _metric_row(
    classes: tuple[str, ...],
    name: str,
    scores: dict[str, Any],
) -> list[str]:
    per_class = scores["per_class_f1"]
    return [
        name,
        f"{scores['macro_f1']:.4f}",
        *[f"{float(per_class[index]):.4f}" for index in range(len(classes))],
        f"{scores['accuracy']:.4f}",
    ]


def _prepare_split(
    cache_dir: Path,
    labels: pd.DataFrame,
    split: str,
) -> tuple[FeatureTable, np.ndarray, np.ndarray]:
    table = load_cached_split(cache_dir, split)
    if table.features.shape[0] != EXPECTED_EPOCHS[split]:
        raise RuntimeError(
            f"{split}: {table.features.shape[0]} epochs, expected {EXPECTED_EPOCHS[split]}"
        )
    y, subjects = align_labels(table.ids, labels, split)
    if len(y) != table.features.shape[0]:
        raise RuntimeError(f"{split} label alignment length mismatch")
    return table, y, subjects


def _print_extract(report: ExtractReport) -> None:
    print(
        "\n".join(
            [
                f"extract_wall_s {report.wall_s:.2f}",
                f"extract_computed {report.computed}",
                f"extract_skipped {report.skipped}",
                f"extract_mean_record_s {report.mean_record_s:.3f}",
                f"extract_epochs {report.epochs}",
                f"extract_cache_mb {report.cache_bytes / (1024 * 1024):.2f}",
                f"extract_peak_private_mb {report.max_private_mb:.1f}",
                f"extract_peak_ws_mb {report.max_working_set_mb:.1f}",
            ]
        ),
        flush=True,
    )


def main() -> int:
    """Extract train/val features, fit on train, and score val.

    Returns
    -------
    int
        Zero when the baseline and the always-W control were written.
    """
    config = load_config(ROOT / "configs" / "config.yaml")
    n_jobs = worker_count()
    np.random.seed(config.seed)
    lock = acquire_lock(config)
    try:
        print(
            f"python numpy={np.__version__} pandas={pd.__version__} "
            f"scipy={__import__('scipy').__version__} sklearn={sklearn.__version__} "
            f"joblib={joblib.__version__} pyyaml={yaml.__version__} seed={config.seed} "
            f"n_jobs={n_jobs}",
            flush=True,
        )
        _self_check()
        print("formula self-check: ok", flush=True)
        report = extract_records(config, _selection(config, ("train", "val")), n_jobs)
        _print_extract(report)
        labels = read_labels(config.tables_dir / "labels.csv")
        train, y_train, _train_subjects = _prepare_split(config.cache_dir, labels, "train")
        val, y_val, val_subjects = _prepare_split(config.cache_dir, labels, "val")
        if train.feature_names != val.feature_names:
            raise RuntimeError("train and val feature names differ")
        missing = sorted(set(CLASS_INDEXES).difference(int(value) for value in np.unique(y_train)))
        if missing:
            raise RuntimeError(f"train is missing classes {missing}")
        if len(set(val_subjects.tolist())) != 8:
            raise RuntimeError(
                f"expected 8 val subjects, found {len(set(val_subjects.tolist()))}"
            )
        model, model_name = build_model(config.seed, n_jobs)
        print(f"classifier {model_name}", flush=True)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train.features, y_train)
        fit_s = time.perf_counter() - fit_started
        for item in caught:
            print(f"warning: {item.category.__name__}: {item.message}", flush=True)
        predict_started = time.perf_counter()
        predicted = np.asarray(model.predict(val.features), dtype=np.int64)
        predict_s = time.perf_counter() - predict_started
        if not np.isin(predicted, CLASS_INDEXES).all():
            raise RuntimeError("prediction outside 0..4")
        scores = score_predictions(y_val, predicted)
        always = np.zeros_like(y_val)
        always_scores = score_predictions(y_val, always)
        by_subject = subject_macro_f1(y_val, predicted, val_subjects)

        config.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = config.models_dir / "baseline_v1.joblib"
        bundle = {
            "model": model,
            "feature_version": FEATURE_VERSION,
            "feature_names": train.feature_names,
            "seed": config.seed,
            "model_name": model_name,
            "classes": list(config.classes),
            "train_rows": int(train.features.shape[0]),
            "val_rows": int(val.features.shape[0]),
            "fit_seconds": fit_s,
            "n_jobs": n_jobs,
        }
        joblib.dump(bundle, model_path, compress=3)
        digest = file_sha256(model_path)
        commit = _git_commit(config.root)
        time_model = (
            f"extract_wall={report.wall_s:.3f};fit={fit_s:.3f};predict={predict_s:.3f}"
        )
        per_class_text = ",".join(f"{float(value):.6f}" for value in scores["per_class_f1"])
        always_text = ",".join(
            f"{float(value):.6f}" for value in always_scores["per_class_f1"]
        )
        _upsert_results(
            config.root / "experiments" / "results.csv",
            [
                {
                    "id": "baseline_v1",
                    "hypothesis": (
                        "Band power and amplitude statistics, without neighboring-epoch "
                        "context, separate the five stages under balanced class weights."
                    ),
                    "features": FEATURE_DESCRIPTION,
                    "model": model_name,
                    "val_macro_f1": f"{scores['macro_f1']:.6f}",
                    "per_class_f1": per_class_text,
                    "time_s": time_model,
                    "seed": str(config.seed),
                    "commit": commit,
                },
                {
                    "id": "always_W",
                    "hypothesis": "Control that predicts W for every val epoch.",
                    "features": "none",
                    "model": "always_W",
                    "val_macro_f1": f"{always_scores['macro_f1']:.6f}",
                    "per_class_f1": always_text,
                    "time_s": "extract_wall=0;fit=0;predict=0",
                    "seed": str(config.seed),
                    "commit": commit,
                },
            ],
        )
        classes = tuple(config.classes)
        print(
            _markdown_table(
                ["model", "macro-F1", *list(classes), "accuracy"],
                [
                    _metric_row(classes, "baseline_v1", scores),
                    _metric_row(classes, "always_W", always_scores),
                ],
            ),
            flush=True,
        )
        print("confusion matrix (rows true, columns predicted)", flush=True)
        matrix = scores["confusion_matrix"]
        matrix_rows = [
            [classes[index], *[str(int(value)) for value in matrix[index]]]
            for index in range(len(classes))
        ]
        print(_markdown_table(["true\\pred", *list(classes)], matrix_rows), flush=True)
        print(
            _markdown_table(
                ["subject_id", "epochs", "classes_present", "macro-F1"],
                [
                    [subject, str(n_epochs), present, f"{score:.4f}"]
                    for subject, n_epochs, present, score in by_subject
                ],
            ),
            flush=True,
        )
        print(
            f"fit_s {fit_s:.2f}\npredict_s {predict_s:.2f}\n"
            f"model {model_path}\nsha256 {digest}",
            flush=True,
        )
        return 0
    finally:
        release_lock(lock)


def _smoke_records(config: Any) -> list[tuple[str, str, Path]]:
    """First two train recordings and the first val recording.

    Paths are signal files. Only their ids and ``epoch_index`` are read,
    and only during the smoke check.
    """
    selected = _selection(config, ("train", "val"))
    chosen = [selected[0], selected[1], selected[EXPECTED_RECORDS["train"]]]
    return [(split, path.stem, path) for split, path in chosen]


def _format_source(value: int | None) -> str:
    return "nan" if value is None else str(value)


def _gap_table(record_id: str, rows: tuple[GapRow, ...]) -> str:
    body = [
        [
            record_id,
            str(row.row),
            str(row.epoch_index),
            *[_format_source(source) for source in row.sources],
            str(row.n_neighbors),
            _format_source(row.next_epoch),
            str(int(row.plus1_is_next_row)),
            str(int(row.copy_ok)),
        ]
        for row in rows
    ]
    return _markdown_table(
        [
            "record",
            "row",
            "epoch_index",
            "off-2",
            "off-1",
            "off+1",
            "off+2",
            "n_neighbors",
            "next_epoch",
            "+1_is_next_row",
            "copy_ok",
        ],
        body,
    )


def _labels_for_ids(ids: np.ndarray, labels: pd.DataFrame, split: str) -> np.ndarray:
    """Map a subset of epoch ids to labels. The full split is not required."""
    part = labels.loc[labels["split"].astype(str) == split, ["id", "label"]]
    mapping = dict(
        zip(part["id"].astype(str).tolist(), part["label"].astype(int).tolist(), strict=True)
    )
    y = np.empty(ids.shape[0], dtype=np.int64)
    for index, epoch_id in enumerate(ids.tolist()):
        value = int(mapping[str(epoch_id)])
        if value not in CLASS_INDEXES:
            raise ValueError(f"label {value} for {epoch_id}")
        y[index] = value
    return y


def _same_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.allclose(left, right, equal_nan=True)
    )


def _report_glue(left: RecordContext, right: RecordContext) -> None:
    """Show that neighbor lookup is not allowed to cross the two recordings."""
    names = [left.record_id, right.record_id]
    try:
        glued = apply_context(
            np.concatenate([left.base, right.base], axis=0),
            np.concatenate([left.epoch_index, right.epoch_index], axis=0),
        )
    except ValueError as exc:
        print(f"glue_refused {names}: {exc}", flush=True)
        return
    glued_right = glued[left.base.shape[0] :]
    if _same_with_nan(glued_right, right.features):
        print(
            f"glue_compare {names[0]}+{names[1]} stacked_equals_separate=1 "
            "(no epoch_index of one recording is a neighbor of the other)",
            flush=True,
        )
        return
    mismatch = np.flatnonzero(
        ~np.isclose(glued_right, right.features, equal_nan=True).all(axis=1)
    )
    row = int(mismatch[0])
    print(
        f"glue_compare {names[0]}+{names[1]} stacked_equals_separate=0 "
        f"first_changed_row={row} epoch={int(right.epoch_index[row])} "
        "(training keeps the separate lookup)",
        flush=True,
    )


def _nan_fraction(features: np.ndarray) -> tuple[float, float]:
    width = (features.shape[1] - 1) // (1 + len(OFFSETS))
    neighbor = features[:, width:-1]
    return float(np.isnan(features).mean()), float(np.isnan(neighbor).mean())


def _context_smoke(
    config: Any,
    n_jobs: int,
) -> tuple[bool, str, float]:
    """Build context on 2 train records and 1 val record, then estimate.

    Returns
    -------
    proceed : bool
        False when the full train matrix would exceed 8 GiB or the scaled
        fit would exceed 20 minutes.
    reason : str
        Gate explanation.
    smoke_fit_s : float
        Fit time on the two train recordings.
    """
    print("context self-check: running", flush=True)
    self_check()
    print("context self-check: ok", flush=True)
    labels = read_labels(config.tables_dir / "labels.csv")
    chosen = _smoke_records(config)
    loaded: list[RecordContext] = []
    for split, record_id, signal_path in chosen:
        index = label_index_for_split(labels, split)
        record = load_record_context(config, split, record_id, index)
        npz_ids, npz_epoch = read_record_index(signal_path)
        assert_labels_match_record(record, npz_ids, npz_epoch)
        loaded.append(record)
        print(
            f"smoke {split}/{record_id} epochs={record.features.shape[0]} "
            f"cols={record.features.shape[1]} "
            f"epoch {int(record.epoch_index.min())}..{int(record.epoch_index.max())} "
            f"row0_n={int(record.features[0, -1])} "
            f"row_last_n={int(record.features[-1, -1])}",
            flush=True,
        )
    train_records = [record for record in loaded if record.split == "train"]
    val_records = [record for record in loaded if record.split == "val"]
    if len(train_records) != 2 or len(val_records) != 1:
        raise RuntimeError("smoke selection is not 2 train and 1 val")
    _report_glue(train_records[0], train_records[1])
    gap_record = next(
        (
            record
            for record in loaded
            if record.epoch_index.size > 1 and bool(np.any(np.diff(record.epoch_index) > 1))
        ),
        None,
    )
    if gap_record is None:
        raise RuntimeError("smoke recordings have no epoch_index gap")
    described = gap_rows(gap_record.base, gap_record.epoch_index, gap_record.features)
    print(f"gap_record {gap_record.split}/{gap_record.record_id}", flush=True)
    print(_gap_table(gap_record.record_id, described), flush=True)
    train_matrix = np.concatenate([record.features for record in train_records], axis=0)
    val_matrix = val_records[0].features
    train_nan, train_neighbor_nan = _nan_fraction(train_matrix)
    val_nan, val_neighbor_nan = _nan_fraction(val_matrix)
    print(
        f"smoke_shape train {train_matrix.shape} val {val_matrix.shape} "
        f"dtype={train_matrix.dtype}",
        flush=True,
    )
    print(
        f"nan_fraction train {train_nan:.6f} neighbor_cols {train_neighbor_nan:.6f}",
        flush=True,
    )
    print(
        f"nan_fraction val {val_nan:.6f} neighbor_cols {val_neighbor_nan:.6f}",
        flush=True,
    )
    full_bytes = matrix_nbytes(EXPECTED_EPOCHS["train"])
    full_gib = full_bytes / (1024**3)
    y_smoke = np.concatenate(
        [_labels_for_ids(record.ids, labels, record.split) for record in train_records]
    )
    model, model_name = build_model(config.seed, n_jobs)
    print(f"smoke_classifier {model_name}", flush=True)
    started = time.perf_counter()
    model.fit(train_matrix, y_smoke)
    smoke_fit_s = time.perf_counter() - started
    estimate_s = smoke_fit_s * EXPECTED_EPOCHS["train"] / train_matrix.shape[0]
    print(
        "\n".join(
            [
                f"smoke_rows {train_matrix.shape[0]}",
                f"smoke_cols {train_matrix.shape[1]}",
                f"expected_cols {N_CONTEXT_FEATURES}",
                f"smoke_fit_s {smoke_fit_s:.3f}",
                f"est_fit_s {estimate_s:.1f}",
                f"est_fit_min {estimate_s / 60:.2f}",
                "est_formula smoke_fit_s * 302653 / smoke_rows",
                f"full_train_matrix_gib {full_gib:.4f}",
                f"full_train_matrix_bytes {full_bytes}",
                f"matrix_limit_gib {CONTEXT_MATRIX_LIMIT_BYTES / (1024**3):.0f}",
                f"fit_limit_min {CONTEXT_FIT_LIMIT_S / 60:.0f}",
            ]
        ),
        flush=True,
    )
    reasons: list[str] = []
    if full_bytes > CONTEXT_MATRIX_LIMIT_BYTES:
        reasons.append(
            f"full train matrix {full_gib:.3f} GiB exceeds 8 GiB"
        )
    if estimate_s > CONTEXT_FIT_LIMIT_S:
        reasons.append(
            f"estimated fit {estimate_s / 60:.2f} min exceeds 20 min"
        )
    del model, train_matrix, val_matrix
    gc.collect()
    if reasons:
        return False, "; ".join(reasons), smoke_fit_s
    return (
        True,
        f"estimate {estimate_s / 60:.2f} min and matrix {full_gib:.3f} GiB "
        "are inside the limits",
        smoke_fit_s,
    )


def _subject_table(
    classes_present: dict[str, str],
    epochs: dict[str, int],
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> tuple[str, int]:
    rows: list[list[str]] = []
    improved = 0
    for subject in sorted(candidate):
        delta = candidate[subject] - baseline[subject]
        if delta > 0:
            improved += 1
        rows.append(
            [
                subject,
                str(epochs[subject]),
                classes_present[subject],
                f"{baseline[subject]:.4f}",
                f"{candidate[subject]:.4f}",
                f"{delta:+.4f}",
                "yes" if delta > 0 else "no",
            ]
        )
    table = _markdown_table(
        ["subject_id", "epochs", "classes", "baseline_v1", "context2_v1", "delta", "improved"],
        rows,
    )
    return table, improved


def main_context(smoke_only: bool = False) -> int:
    """Score same-record context with the frozen baseline hyperparameters.

    A smoke fit on two train recordings runs first. The full fit starts only
    when the train matrix is at most 8 GiB and the scaled fit is at most
    20 minutes. The baseline model file is only loaded, not rewritten.

    Parameters
    ----------
    smoke_only : bool, optional
        Stop after the smoke report.

    Returns
    -------
    int
        0 when the requested stage finished, 2 when the smoke gate stops
        the full fit.
    """
    config = load_config(ROOT / "configs" / "config.yaml")
    n_jobs = worker_count()
    np.random.seed(config.seed)
    lock = acquire_lock(config)
    try:
        print(
            f"python numpy={np.__version__} pandas={pd.__version__} "
            f"sklearn={sklearn.__version__} joblib={joblib.__version__} "
            f"seed={config.seed} n_jobs={n_jobs} context_offsets={list(OFFSETS)}",
            flush=True,
        )
        proceed, reason, smoke_fit_s = _context_smoke(config, n_jobs)
        print(
            f"SMOKE decision: {'PROCEED' if proceed else 'STOP'}\nreason: {reason}",
            flush=True,
        )
        if not proceed:
            return 2
        if smoke_only:
            return 0
        labels = read_labels(config.tables_dir / "labels.csv")
        train_index = label_index_for_split(labels, "train")
        val_index = label_index_for_split(labels, "val")
        print("context_build train", flush=True)
        build_started = time.perf_counter()
        train = load_context_split(config, "train", train_index, keep_base=False)
        print("context_build val", flush=True)
        val = load_context_split(config, "val", val_index, keep_base=True)
        build_s = time.perf_counter() - build_started
        if train.features.shape != (EXPECTED_EPOCHS["train"], N_CONTEXT_FEATURES):
            raise RuntimeError(f"train context shape {train.features.shape}")
        if val.features.shape != (EXPECTED_EPOCHS["val"], N_CONTEXT_FEATURES):
            raise RuntimeError(f"val context shape {val.features.shape}")
        if val.base is None or val.base.shape != (EXPECTED_EPOCHS["val"], N_FEATURES):
            raise RuntimeError("val base matrix was not kept")
        if train.feature_names != val.feature_names:
            raise RuntimeError("train and val context names differ")
        y_train, _train_subjects = align_labels(train.ids, labels, "train")
        y_val, val_subjects = align_labels(val.ids, labels, "val")
        missing = sorted(set(CLASS_INDEXES).difference(int(value) for value in np.unique(y_train)))
        if missing:
            raise RuntimeError(f"train is missing classes {missing}")
        if len(set(val_subjects.tolist())) != 8:
            raise RuntimeError(
                f"expected 8 val subjects, found {len(set(val_subjects.tolist()))}"
            )
        baseline_path = config.models_dir / "baseline_v1.joblib"
        baseline = load_baseline(baseline_path)
        if tuple(baseline["feature_names"]) != val.feature_names[:N_FEATURES]:
            raise RuntimeError("baseline feature names differ from the cache")
        model, model_name = build_model(config.seed, n_jobs)
        print(f"classifier {model_name}", flush=True)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train.features, y_train)
        fit_s = time.perf_counter() - fit_started
        for item in caught:
            print(f"warning: {item.category.__name__}: {item.message}", flush=True)
        del train
        gc.collect()
        predict_started = time.perf_counter()
        predicted = np.asarray(model.predict(val.features), dtype=np.int64)
        predict_s = time.perf_counter() - predict_started
        if not np.isin(predicted, CLASS_INDEXES).all():
            raise RuntimeError("prediction outside 0..4")
        scores = score_predictions(y_val, predicted)
        base_predicted = np.asarray(baseline["model"].predict(val.base), dtype=np.int64)
        base_scores = score_predictions(y_val, base_predicted)
        logged = 0.712991
        if abs(base_scores["macro_f1"] - logged) > 1e-4:
            raise RuntimeError(
                f"reloaded baseline macro-F1 {base_scores['macro_f1']:.6f} "
                f"disagrees with the logged {logged:.6f}"
            )
        by_subject = subject_macro_f1(y_val, predicted, val_subjects)
        by_base = subject_macro_f1(y_val, base_predicted, val_subjects)
        candidate = {subject: score for subject, _n, _present, score in by_subject}
        reference = {subject: score for subject, _n, _present, score in by_base}
        epochs = {subject: n_epochs for subject, n_epochs, _present, _score in by_subject}
        present = {subject: classes for subject, _n, classes, _score in by_subject}
        subject_text, improved = _subject_table(present, epochs, reference, candidate)
        config.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = config.models_dir / "context2_v1.joblib"
        bundle = {
            "model": model,
            "feature_version": FEATURE_VERSION,
            "feature_names": val.feature_names,
            "context_offsets": list(OFFSETS),
            "seed": config.seed,
            "model_name": model_name,
            "classes": list(config.classes),
            "train_rows": EXPECTED_EPOCHS["train"],
            "val_rows": int(val.features.shape[0]),
            "fit_seconds": fit_s,
            "n_jobs": n_jobs,
        }
        joblib.dump(bundle, model_path, compress=3)
        digest = file_sha256(model_path)
        commit = _git_commit(config.root)
        per_class_text = ",".join(f"{float(value):.6f}" for value in scores["per_class_f1"])
        _upsert_results(
            config.root / "experiments" / "results.csv",
            [
                {
                    "id": "context2_v1",
                    "hypothesis": (
                        "Sleep stages follow a sequence, so features from the same "
                        "record at epoch_index offsets -2, -1, +1 and +2 raise val "
                        "macro-F1, especially for N1 and REM."
                    ),
                    "features": CONTEXT_FEATURE_DESCRIPTION,
                    "model": model_name,
                    "val_macro_f1": f"{scores['macro_f1']:.6f}",
                    "per_class_f1": per_class_text,
                    "time_s": (
                        f"smoke_fit={smoke_fit_s:.3f};context_build={build_s:.3f};"
                        f"fit={fit_s:.3f};predict={predict_s:.3f}"
                    ),
                    "seed": str(config.seed),
                    "commit": commit,
                }
            ],
        )
        classes = tuple(config.classes)
        print(
            _markdown_table(
                ["model", "macro-F1", *list(classes), "accuracy"],
                [
                    _metric_row(classes, "baseline_v1", base_scores),
                    _metric_row(classes, "context2_v1", scores),
                ],
            ),
            flush=True,
        )
        print("confusion matrix context2_v1 (rows true, columns predicted)", flush=True)
        matrix = scores["confusion_matrix"]
        matrix_rows = [
            [classes[index], *[str(int(value)) for value in matrix[index]]]
            for index in range(len(classes))
        ]
        print(_markdown_table(["true\\pred", *list(classes)], matrix_rows), flush=True)
        print(subject_text, flush=True)
        macro_ok = scores["macro_f1"] >= CONTEXT_ACCEPT_MACRO_F1
        subjects_ok = improved >= CONTEXT_ACCEPT_SUBJECTS
        print(
            "\n".join(
                [
                    f"subjects_improved {improved}/8",
                    f"criterion_macro {scores['macro_f1']:.6f} >= {CONTEXT_ACCEPT_MACRO_F1:.3f}: "
                    f"{'yes' if macro_ok else 'no'}",
                    f"criterion_subjects {improved} >= {CONTEXT_ACCEPT_SUBJECTS}: "
                    f"{'yes' if subjects_ok else 'no'}",
                    f"criterion {'pass' if macro_ok and subjects_ok else 'fail'}",
                    f"context_build_s {build_s:.2f}",
                    f"fit_s {fit_s:.2f}",
                    f"predict_s {predict_s:.2f}",
                    f"smoke_fit_s {smoke_fit_s:.2f}",
                    f"model {model_path}",
                    f"sha256 {digest}",
                    f"baseline_macro_f1 {base_scores['macro_f1']:.6f}",
                ]
            ),
            flush=True,
        )
        return 0
    finally:
        release_lock(lock)


def cli(argv: list[str] | None = None) -> int:
    """Dispatch baseline training or ``--context 2``.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments. An empty context flag keeps the baseline entry point.

    Returns
    -------
    int
        Exit code from the selected mode.
    """
    parser = argparse.ArgumentParser(description="Train baseline_v1 or context2_v1.")
    parser.add_argument(
        "--context",
        type=int,
        choices=(2,),
        help="Attach same-record neighbors at offsets -2,-1,+1,+2 and fit context2_v1.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="With --context 2, stop after the 3-record estimate.",
    )
    args = parser.parse_args(argv)
    if args.context is None:
        if args.smoke:
            parser.error("baseline mode has no --smoke; use python -m src.features --smoke")
        return main()
    return main_context(smoke_only=args.smoke)


if __name__ == "__main__":
    raise SystemExit(cli())

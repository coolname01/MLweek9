"""Train the spectral baseline on train and score it on val.

Val epochs are not used to fit the model. The test split is not read.
"""

from __future__ import annotations

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

from src.data import (
    EXPECTED_EPOCHS,
    file_sha256,
    load_config,
    read_labels,
)
from src.features import (
    FEATURE_VERSION,
    ExtractReport,
    FeatureTable,
    _self_check,
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


if __name__ == "__main__":
    raise SystemExit(main())

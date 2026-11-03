"""Shared helpers for the overnight queue.

Existing training code is imported, not modified. The test split is never
opened. New caches and models are new files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.context import apply_context, cached_record_ids, label_index_for_split
from src.data import (
    EXPECTED_EPOCHS,
    EXPECTED_RECORDS,
    Config,
    _as_text_list,
    file_sha256,
    load_config,
    read_labels,
)
from src.features import (
    N_FEATURES,
    _cache_ready,
    cache_file,
    feature_names,
    worker_count,
)
from src.train import CLASS_INDEXES, build_model, subject_macro_f1

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "results.csv"
CHAMPION_PATH = ROOT / "experiments" / "champion.json"
DISK_FLOOR_GIB = 5.0
TRAIN_TIME_CAP_S = 40.0 * 60.0
ACCEPT_DELTA = 0.01
ACCEPT_SUBJECTS = 5
RESULT_COLUMNS = [
    "id",
    "hypothesis",
    "features",
    "model",
    "val_macro_f1",
    "per_class_f1",
    "n_subjects_improved",
    "time_s",
    "seed",
    "decision",
    "model_sha256",
    "est_kaggle_minutes",
    "ram_peak_gb",
    "deterministic",
    "commit",
]


@dataclass(frozen=True)
class RecordBase:
    """One recording's cached v1 matrix and its own ``epoch_index``."""

    split: str
    record_id: str
    ids: np.ndarray
    epoch_index: np.ndarray
    base: np.ndarray


@dataclass(frozen=True)
class SplitMatrix:
    """Row-aligned features for one split. Records are never glued."""

    features: np.ndarray
    ids: np.ndarray
    epoch_index: np.ndarray
    record_id: np.ndarray
    feature_names: tuple[str, ...]


def project_config() -> Config:
    """Load ``configs/config.yaml``."""
    return load_config(ROOT / "configs" / "config.yaml")


def disk_free_gib(path: Path | None = None) -> float:
    """Return free space in GiB on the project drive."""
    usage = shutil.disk_usage(path or ROOT)
    return float(usage.free) / (1024.0**3)


def require_disk() -> float:
    """Stop the queue when less than 5 GiB is free.

    Returns
    -------
    float
        Free GiB.

    Raises
    ------
    RuntimeError
        When free space is below the floor. Callers record this and stop.
    """
    free = disk_free_gib()
    if free < DISK_FLOOR_GIB:
        raise RuntimeError(f"free disk {free:.2f} GiB is below {DISK_FLOOR_GIB:.0f} GiB")
    return free


def git_commit(root: Path | None = None) -> str:
    """Return HEAD and a dirty suffix. ``none`` when git is unavailable."""
    folder = root or ROOT
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=folder,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "none"
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=folder,
        text=True,
    )
    if status.strip():
        return f"{revision}-dirty"
    return revision


def context_feature_names(base: tuple[str, ...], offsets: tuple[int, ...]) -> tuple[str, ...]:
    """Return base names, one block per offset, then ``n_neighbors``."""
    names = list(base)
    for offset in offsets:
        prefix = f"off{offset:+d}_"
        names.extend(prefix + name for name in base)
    names.append("n_neighbors")
    return tuple(names)


def apply_offsets(
    features: np.ndarray,
    epoch_index: np.ndarray,
    offsets: tuple[int, ...],
) -> np.ndarray:
    """Attach neighbor columns for one recording.

    A neighbor is copied only when ``epoch_index + k`` exists in this
    recording. Gaps and edges are NaN. The last column is the number of
    found offsets.

    Parameters
    ----------
    features : ndarray, shape (n_epochs, n_features)
        Finite features of this recording only.
    epoch_index : ndarray of int, shape (n_epochs,)
        Index aligned with ``features``. Duplicates are refused.
    offsets : tuple of int
        Offsets such as ``(-4, -3, -2, -1, 1, 2, 3, 4)``. Zero is refused
        because the current epoch is already the first block.

    Returns
    -------
    ndarray, shape (n_epochs, n_features * (1 + n_offsets) + 1), float32
    """
    if any(int(offset) == 0 for offset in offsets):
        raise ValueError("offset 0 is the current epoch, not a neighbor")
    values = np.ascontiguousarray(features, dtype=np.float32)
    index = np.asarray(epoch_index, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != index.shape[0]:
        raise ValueError(f"features {values.shape} do not match epoch_index {index.shape}")
    n_epochs, width = values.shape
    if n_epochs == 0:
        raise ValueError("empty record")
    if not np.isfinite(values).all():
        raise ValueError("base features must be finite")
    order = np.argsort(index, kind="mergesort")
    ordered = index[order]
    if n_epochs > 1 and bool(np.any(np.diff(ordered) == 0)):
        raise ValueError("duplicate epoch_index in one record")
    out = np.empty((n_epochs, width * (1 + len(offsets)) + 1), dtype=np.float32)
    out[:, :width] = values
    counts = np.zeros(n_epochs, dtype=np.int16)
    for slot, offset in enumerate(offsets):
        target = index + np.int64(offset)
        pos = np.searchsorted(ordered, target)
        valid = pos < n_epochs
        clipped = np.where(valid, pos, 0)
        matched = valid & (ordered[clipped] == target)
        start = width * (slot + 1)
        block = out[:, start : start + width]
        block.fill(np.nan)
        source = order[clipped]
        block[matched] = values[source[matched]]
        counts += matched.astype(np.int16)
    out[:, -1] = counts.astype(np.float32)
    _assert_offset_blocks(out, width, len(offsets))
    return out


def _assert_offset_blocks(context: np.ndarray, width: int, n_offsets: int) -> None:
    if np.isnan(context[:, :width]).any() or np.isnan(context[:, -1]).any():
        raise RuntimeError("NaN in the current epoch or in n_neighbors")
    found = np.zeros(context.shape[0], dtype=np.float32)
    for slot in range(n_offsets):
        block = context[:, width * (slot + 1) : width * (slot + 2)]
        missing = np.isnan(block)
        partial = missing.any(axis=1) & ~missing.all(axis=1)
        if bool(partial.any()):
            raise RuntimeError("neighbor row is only partly NaN")
        found += (~missing.all(axis=1)).astype(np.float32)
    if not np.array_equal(found, context[:, -1]):
        raise RuntimeError("n_neighbors does not match the NaN pattern")
    if float(context[:, -1].min()) < 0 or float(context[:, -1].max()) > n_offsets:
        raise RuntimeError("n_neighbors outside 0..n_offsets")


def rolling_mean_std(
    features: np.ndarray,
    epoch_index: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered rolling mean and population std inside one recording.

    A window of width ``2 * radius + 1`` never crosses an ``epoch_index`` gap
    or a recording boundary. Edges and short segments use the epochs that
    exist in the contiguous run. The standard deviation of one epoch is 0.

    Parameters
    ----------
    features : ndarray, shape (n_epochs, n_features)
        Per-epoch features in file order.
    epoch_index : ndarray of int
        Strictly increasing index for this recording only.
    radius : int
        2 for a 5-epoch window, 5 for an 11-epoch window.

    Returns
    -------
    mean, std : ndarray, float32
        Same shape as ``features``.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    values = np.asarray(features, dtype=np.float64)
    index = np.asarray(epoch_index, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != index.shape[0]:
        raise ValueError("rolling features are not aligned with epoch_index")
    n_epochs, width = values.shape
    if n_epochs == 0:
        raise ValueError("empty record")
    if n_epochs > 1 and bool(np.any(np.diff(index) <= 0)):
        raise ValueError("epoch_index must be strictly increasing")
    mean = np.empty((n_epochs, width), dtype=np.float32)
    std = np.empty((n_epochs, width), dtype=np.float32)
    if n_epochs == 1:
        mean[0] = values[0]
        std[0] = 0.0
        return mean, std
    breaks = np.flatnonzero(np.diff(index) != 1)
    starts = np.concatenate((np.array([0], dtype=int), breaks + 1))
    ends = np.concatenate((breaks + 1, np.array([n_epochs], dtype=int)))
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        block = values[start:end]
        length = int(block.shape[0])
        cumulative = np.cumsum(block, axis=0)
        cumulative_sq = np.cumsum(block * block, axis=0)
        position = np.arange(length)
        lo = np.maximum(0, position - radius)
        hi = np.minimum(length - 1, position + radius)
        total = np.array(cumulative[hi], copy=True)
        total_sq = np.array(cumulative_sq[hi], copy=True)
        previous = lo - 1
        present = previous >= 0
        if bool(np.any(present)):
            total[present] -= cumulative[previous[present]]
            total_sq[present] -= cumulative_sq[previous[present]]
        count = (hi - lo + 1).astype(np.float64)[:, None]
        center = total / count
        variance = np.maximum(total_sq / count - center * center, 0.0)
        mean[start:end] = center
        std[start:end] = np.sqrt(variance)
    return mean, std


def assert_rolling_respects_gaps() -> None:
    """Epochs on the far side of a gap must not enter the window."""
    index = np.array([0, 1, 2, 5, 6], dtype=np.int64)
    features = index.astype(np.float32).reshape(-1, 1)
    mean, std = rolling_mean_std(features, index, radius=2)
    if abs(float(mean[2, 0]) - 1.0) > 1e-6:
        raise RuntimeError("rolling mixed a gap into the left segment")
    if abs(float(mean[3, 0]) - 5.5) > 1e-6:
        raise RuntimeError("rolling mixed a gap into the right segment")
    if float(std[3, 0]) <= 0:
        raise RuntimeError("two-epoch window should have a positive std")
    alone_mean, alone_std = rolling_mean_std(features[:1], index[:1], radius=5)
    if float(alone_mean[0, 0]) != 0.0 or float(alone_std[0, 0]) != 0.0:
        raise RuntimeError("single-epoch window is not centered on itself")


def same_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    """True when finite values match and NaN masks match."""
    return bool(
        left.shape == right.shape
        and np.array_equal(np.isnan(left), np.isnan(right))
        and np.allclose(left, right, equal_nan=True, rtol=0.0, atol=0.0)
    )


def assert_matches_context2(features: np.ndarray, epoch_index: np.ndarray) -> None:
    """The ±2 helper must match the frozen ``src.context.apply_context``."""
    offsets = (-2, -1, 1, 2)
    got = apply_offsets(features, epoch_index, offsets)
    expected = apply_context(features, epoch_index)
    if not same_with_nan(got, expected):
        raise RuntimeError("apply_offsets ±2 disagrees with src.context.apply_context")


def load_record_base(
    config: Config,
    split: str,
    record_id: str,
    label_index: pd.DataFrame,
) -> RecordBase:
    """Load one v1 cache file and join ``epoch_index`` by ``id``.

    The signal file is not opened. ``test`` is refused.
    """
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    path = cache_file(config, split, record_id)
    if not _cache_ready(path):
        raise RuntimeError(f"cache missing or stale: {path}")
    with np.load(path, allow_pickle=False) as archive:
        base = np.array(archive["features"], dtype=np.float32, copy=True)
        ids = np.asarray(_as_text_list(archive["ids"]), dtype=str)
        names = tuple(_as_text_list(archive["feature_names"]))
    if names != feature_names():
        raise RuntimeError(f"{split}/{record_id} feature names differ from v1")
    if base.shape != (ids.shape[0], N_FEATURES):
        raise RuntimeError(f"{split}/{record_id} feature shape {base.shape}")
    located = label_index.reindex(ids.tolist())
    if bool(located["epoch_index"].isna().any()) or bool(located["record_id"].isna().any()):
        raise ValueError(f"{split}/{record_id} ids are absent from labels")
    if not np.all(located["record_id"].astype(str).to_numpy() == record_id):
        raise ValueError(f"{split}/{record_id} label record_id mismatch")
    epoch_index = located["epoch_index"].to_numpy(dtype=np.int64)
    if epoch_index.size > 1 and bool(np.any(np.diff(epoch_index) <= 0)):
        raise ValueError(f"{split}/{record_id} epoch_index is not strictly increasing")
    return RecordBase(
        split=split,
        record_id=record_id,
        ids=ids,
        epoch_index=epoch_index,
        base=base,
    )


def load_bases(
    config: Config,
    split: str,
    labels: pd.DataFrame,
    record_ids: list[str] | None = None,
) -> list[RecordBase]:
    """Load cached v1 rows for a split, one recording at a time."""
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    index = label_index_for_split(labels, split)
    chosen = list(record_ids) if record_ids is not None else cached_record_ids(config, split)
    if record_ids is None and len(chosen) != EXPECTED_RECORDS[split]:
        raise RuntimeError(f"{split} cache count {len(chosen)}")
    return [load_record_base(config, split, record_id, index) for record_id in chosen]


def stack_offsets(records: list[RecordBase], offsets: tuple[int, ...]) -> SplitMatrix:
    """Context matrix. Each recording is transformed on its own."""
    if not records:
        raise ValueError("no records")
    names = context_feature_names(feature_names(), offsets)
    blocks: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    epochs: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for record in records:
        blocks.append(apply_offsets(record.base, record.epoch_index, offsets))
        ids.append(record.ids)
        epochs.append(record.epoch_index)
        owners.append(np.full(record.ids.shape[0], record.record_id, dtype=object))
    return SplitMatrix(
        features=np.concatenate(blocks, axis=0),
        ids=np.concatenate(ids, axis=0),
        epoch_index=np.concatenate(epochs, axis=0),
        record_id=np.concatenate(owners, axis=0).astype(str),
        feature_names=names,
    )


def matrix_nbytes(n_rows: int, n_cols: int) -> int:
    """Float32 bytes of a dense matrix."""
    return int(n_rows) * int(n_cols) * int(np.dtype(np.float32).itemsize)


def score_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Macro-F1, per-class F1, accuracy and the confusion matrix."""
    per_class = f1_score(
        y_true,
        y_pred,
        labels=CLASS_INDEXES,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_INDEXES)
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
        "confusion_matrix": np.asarray(matrix, dtype=np.int64),
    }


def subject_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
) -> dict[str, float]:
    """Macro-F1 by ``subject_id``. Missing classes contribute F1 0."""
    rows = subject_macro_f1(y_true, y_pred, subjects)
    return {subject: score for subject, _n, _present, score in rows}


def n_improved(candidate: dict[str, float], reference: dict[str, float]) -> int:
    """Count subjects whose macro-F1 is strictly higher than the reference."""
    if set(candidate) != set(reference):
        raise RuntimeError("subject sets differ")
    return sum(1 for subject in candidate if candidate[subject] > reference[subject])


def accepts(macro_f1: float, improved: int, reference_macro: float) -> bool:
    """True when both acceptance gates pass against the current best."""
    return bool(macro_f1 >= reference_macro + ACCEPT_DELTA and improved >= ACCEPT_SUBJECTS)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a pipe table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def per_class_text(scores: dict[str, Any]) -> str:
    """Comma-separated per-class F1 with 6 decimals."""
    return ",".join(f"{float(value):.6f}" for value in scores["per_class_f1"])


def model_description(model: Any) -> str:
    """Short LightGBM name for the experiment log."""
    params = model.get_params()
    weight = params.get("class_weight")
    weight_text = weight if isinstance(weight, str) else "custom"
    return (
        f"LGBMClassifier(n_estimators={params.get('n_estimators')},"
        f"learning_rate={params.get('learning_rate')},num_leaves={params.get('num_leaves')},"
        f"class_weight={weight_text},deterministic={params.get('deterministic')},"
        f"n_jobs={params.get('n_jobs')})"
    )


def fit_model(
    features: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_jobs: int,
    limit_s: float,
    overrides: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    """Fit LightGBM and abort when the wall clock passes ``limit_s``.

    The default hyperparameters match ``src.train.build_model``. Overrides
    replace individual constructor arguments. The test split is not read.
    """
    from lightgbm import LGBMClassifier

    params: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "class_weight": "balanced",
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    if overrides:
        params.update(overrides)
    model = LGBMClassifier(**params)
    started = time.perf_counter()

    def _time_limit(env: Any) -> None:
        del env
        if time.perf_counter() - started > limit_s:
            raise TimeoutError(f"fit exceeded {limit_s:.1f} s")

    model.fit(features, y, callbacks=[_time_limit])
    return model, time.perf_counter() - started


def default_model(seed: int, n_jobs: int) -> tuple[Any, str]:
    """The frozen baseline LightGBM, via ``build_model``."""
    return build_model(seed, n_jobs)


def upsert_result(row: dict[str, str]) -> None:
    """Insert or replace one experiment row. Existing ids keep their values."""
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULT_PATH.is_file() and RESULT_PATH.stat().st_size > 0:
        frame = pd.read_csv(RESULT_PATH, dtype=str, encoding="utf-8", keep_default_na=False)
    else:
        frame = pd.DataFrame(columns=RESULT_COLUMNS)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[RESULT_COLUMNS]
    payload = {column: str(row.get(column, "")) for column in RESULT_COLUMNS}
    if len(frame):
        frame = frame.loc[frame["id"] != payload["id"]].copy()
    updated = pd.concat([frame, pd.DataFrame([payload])], ignore_index=True)
    updated.to_csv(RESULT_PATH, index=False, encoding="utf-8", lineterminator="\n")


def load_champion() -> dict[str, Any]:
    """Return the current accepted model description."""
    if not CHAMPION_PATH.is_file():
        raise RuntimeError(f"missing {CHAMPION_PATH}")
    return json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))


def save_champion(payload: dict[str, Any]) -> None:
    """Write the accepted-model pointer. This file is new."""
    CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAMPION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_prediction_csv(path: Path, ids: np.ndarray, labels: np.ndarray) -> str:
    """Write ``id,label`` with ``\\n`` newlines and return the SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,label\n"]
    for epoch_id, label in zip(ids.tolist(), labels.tolist(), strict=True):
        lines.append(f"{epoch_id},{int(label)}\n")
    path.write_bytes("".join(lines).encode("utf-8"))
    return file_sha256(path)


def prediction_sha(ids: np.ndarray, labels: np.ndarray) -> str:
    """SHA-256 of an in-memory ``id,label`` file. Nothing is written."""
    lines = ["id,label\n"]
    for epoch_id, label in zip(ids.tolist(), labels.tolist(), strict=True):
        lines.append(f"{epoch_id},{int(label)}\n")
    import hashlib

    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def yaml_version() -> str:
    """pyyaml version string for the run log."""
    return str(yaml.__version__)


def thread_env_one() -> dict[str, str | None]:
    """Limit numeric libraries to one thread. Returns the previous values."""
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    """Restore environment variables captured by :func:`thread_env_one`."""
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

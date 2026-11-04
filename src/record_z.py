"""Per-record z-score of the 54 ``features_v1`` columns.

Frozen formula, one recording at a time::

    z = (x - mean_rec) / (population_std_rec + 1e-6)

``mean_rec`` and ``population_std_rec`` (ddof 0) are taken over every epoch
of this recording and over no other recording. Context offsets -2, -1, +1
and +2 are then applied to ``z``. ``cache/features_v1`` is only read.
Normalized columns are written to ``cache/features_c1``. The test split is
never opened.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.context import OFFSETS
from src.data import Config, _as_text_list
from src.expkit import apply_offsets, context_feature_names
from src.features import N_FEATURES, feature_names

FORMULA_TEXT = "(x - mean_rec) / (population_std_rec + 1e-6)"
SCOPE_TEXT = "one_record_all_epochs"
CACHE_NAME = "features_c1"
FORMULA_VERSION = "c1"
STD_EPS = 1e-6
STD_DDOF = 0
C1_COLUMNS = N_FEATURES * (1 + len(OFFSETS)) + 1
C2_COLUMNS = N_FEATURES * (1 + len(OFFSETS)) * 2 + 1
VARIANTS: tuple[str, ...] = ("c1", "c2")

if OFFSETS != (-2, -1, 1, 2):
    raise RuntimeError(f"frozen offsets drifted: {OFFSETS}")
if C1_COLUMNS != 271 or C2_COLUMNS != 541:
    raise RuntimeError(f"frozen widths drifted: {C1_COLUMNS}, {C2_COLUMNS}")
if STD_EPS != 1e-6 or STD_DDOF != 0:
    raise RuntimeError("frozen standardization constants drifted")


def _checked_float64(features: np.ndarray) -> np.ndarray:
    """Return one record as a finite float64 matrix of width 54."""
    if isinstance(features, (list, tuple)):
        raise TypeError("pass one record ndarray, not a sequence of records")
    if not isinstance(features, np.ndarray):
        raise TypeError(f"features must be one ndarray, got {type(features).__name__}")
    if features.ndim != 2 or features.shape[1] != N_FEATURES or features.shape[0] < 1:
        raise ValueError(
            f"expected one record of shape (n_epochs, {N_FEATURES}), got {features.shape}"
        )
    values = np.asarray(features, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("record features must be finite")
    return values


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population mean and standard deviation of each column.

    Parameters
    ----------
    values : ndarray, shape (n_epochs, 54)
        Finite float64 features of one recording.

    Returns
    -------
    mean, std : ndarray, shape (54,)
        ``std`` is ``sqrt(mean((x - mean) ** 2))`` with ddof 0.
    """
    mean = values.mean(axis=0)
    std = np.sqrt(np.mean((values - mean) ** 2, axis=0))
    return mean, std


def population_mean_std(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std of one record's 54 features.

    Parameters
    ----------
    features : ndarray, shape (n_epochs, 54)
        ``features_v1`` rows of a single recording.

    Returns
    -------
    mean, std : ndarray, shape (54,)
        Statistics over every epoch in ``features``.
    """
    return _mean_std(_checked_float64(features))


def zscore_record_features(features: np.ndarray) -> np.ndarray:
    """Standardize the 54 features of one recording.

    Parameters
    ----------
    features : ndarray, shape (n_epochs, 54)
        ``features_v1`` rows of a single recording. Epochs from any other
        recording must not be included. The argument is the only data input.

    Returns
    -------
    ndarray, shape (n_epochs, 54), float32
        ``(x - mean_rec) / (population_std_rec + 1e-6)``. A constant column
        has standard deviation 0 and becomes 0.
    """
    values = _checked_float64(features)
    mean, std = _mean_std(values)
    standardized = (values - mean) / (std + np.float64(STD_EPS))
    return np.ascontiguousarray(standardized, dtype=np.float32)


def feature_names_for(variant: str) -> tuple[str, ...]:
    """Column names for ``c1`` (271) or ``c2`` (541).

    Parameters
    ----------
    variant : str
        ``c1`` or ``c2``.

    Returns
    -------
    tuple of str
        ``c1`` is the z-scored context. ``c2`` is the raw context body, then
        the z-scored context body, then one ``n_neighbors`` column.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant}")
    raw_names = context_feature_names(feature_names(), OFFSETS)
    z_base = tuple(f"z_{name}" for name in feature_names())
    z_names = context_feature_names(z_base, OFFSETS)
    if variant == "c1":
        if len(z_names) != C1_COLUMNS or z_names[-1] != "n_neighbors":
            raise RuntimeError("c1 names drifted")
        return z_names
    names = raw_names[:-1] + z_names[:-1] + ("n_neighbors",)
    if len(names) != C2_COLUMNS or raw_names[-1] != "n_neighbors":
        raise RuntimeError("c2 names drifted")
    if len(set(names)) != len(names):
        raise RuntimeError("c2 feature names are not unique")
    return names


def variant_matrix(
    raw: np.ndarray,
    standardized: np.ndarray,
    epoch_index: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Attach ±2 context for one recording.

    Parameters
    ----------
    raw : ndarray, shape (n_epochs, 54)
        Finite ``features_v1`` rows of this recording only.
    standardized : ndarray, shape (n_epochs, 54)
        Output of :func:`zscore_record_features` for the same rows.
    epoch_index : ndarray of int, shape (n_epochs,)
        ``epoch_index`` of this recording. Neighbors exist only when that
        index is present here.
    variant : str
        ``c1`` returns the z-scored context (271). ``c2`` concatenates the
        raw context body, the z-scored context body and one neighbor count
        (541).

    Returns
    -------
    ndarray, float32
        Model matrix for this recording only.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant}")
    raw_values = np.ascontiguousarray(raw, dtype=np.float32)
    z_values = np.ascontiguousarray(standardized, dtype=np.float32)
    if raw_values.shape != z_values.shape or raw_values.shape[1] != N_FEATURES:
        raise ValueError(f"raw {raw_values.shape} and z {z_values.shape} differ")
    z_context = apply_offsets(z_values, epoch_index, OFFSETS)
    if variant == "c1":
        if z_context.shape[1] != C1_COLUMNS:
            raise RuntimeError(f"c1 width {z_context.shape[1]}")
        return z_context
    raw_context = apply_offsets(raw_values, epoch_index, OFFSETS)
    if not np.array_equal(raw_context[:, -1], z_context[:, -1]):
        raise RuntimeError("n_neighbors differs between raw and z context")
    matrix = np.concatenate(
        [raw_context[:, :-1], z_context[:, :-1], raw_context[:, -1:]],
        axis=1,
    )
    if matrix.shape[1] != C2_COLUMNS:
        raise RuntimeError(f"c2 width {matrix.shape[1]}")
    return matrix


def transform_record(
    features: np.ndarray,
    epoch_index: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Build the model matrix of one recording.

    Parameters
    ----------
    features : ndarray, shape (n_epochs, 54)
        Raw ``features_v1`` rows of a single recording.
    epoch_index : ndarray of int, shape (n_epochs,)
        ``epoch_index`` aligned with ``features``.
    variant : str
        ``c1`` or ``c2``.

    Returns
    -------
    ndarray, float32
        271 columns for ``c1``, or 541 columns for ``c2``. No other
        recording is an argument, and none is read.
    """
    standardized = zscore_record_features(features)
    return variant_matrix(features, standardized, epoch_index, variant)


def c1_cache_path(config: Config, split: str, record_id: str) -> Path:
    """Return ``cache/features_c1/{split}/{record_id}.npz``.

    Parameters
    ----------
    config : Config
        Project paths.
    split : str
        ``train`` or ``val``. ``test`` is refused.
    record_id : str
        Recording stem.

    Returns
    -------
    pathlib.Path
        Destination that is not under ``features_v1``.
    """
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    path = config.cache_dir / CACHE_NAME / split / f"{record_id}.npz"
    if CACHE_NAME == "features_v1" or "features_v1" in path.parts:
        raise RuntimeError(f"refusing to write {path}")
    return path


def _scalar_text(value: Any) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def write_c1_cache(
    path: Path,
    features: np.ndarray,
    ids: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """Write one record's 54 standardized columns.

    Parameters
    ----------
    path : pathlib.Path
        Destination from :func:`c1_cache_path`.
    features : ndarray, shape (n_epochs, 54), float32
        Standardized rows.
    ids : ndarray of str
        Epoch ids in the same order.
    mean, std : ndarray, shape (54,)
        Population statistics used for this recording.
    """
    if "features_v1" in path.parts or path.parent.parent.name != CACHE_NAME:
        raise RuntimeError(f"refusing to write {path}")
    if features.shape != (ids.shape[0], N_FEATURES):
        raise ValueError(f"cache shape {features.shape} ids {ids.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.npz")
    payload = {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "ids": np.asarray(ids),
        "feature_names": np.asarray(feature_names()),
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "eps": np.float64(STD_EPS),
        "ddof": np.int64(STD_DDOF),
        "version": np.asarray(FORMULA_VERSION),
        "formula": np.asarray(FORMULA_TEXT),
        "scope": np.asarray(SCOPE_TEXT),
        "source": np.asarray("features_v1"),
    }
    try:
        np.savez(temporary, **payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def cache_matches(path: Path, features: np.ndarray, ids: np.ndarray) -> bool:
    """Return whether ``path`` already stores this record's standardized rows."""
    if not path.is_file() or path.name.startswith("."):
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            stored = np.asarray(archive["features"])
            stored_ids = np.asarray(_as_text_list(archive["ids"]), dtype=str)
            ready = (
                _scalar_text(archive["version"]) == FORMULA_VERSION
                and _scalar_text(archive["formula"]) == FORMULA_TEXT
                and float(np.asarray(archive["eps"]).reshape(-1)[0]) == STD_EPS
                and int(np.asarray(archive["ddof"]).reshape(-1)[0]) == STD_DDOF
                and stored.shape == features.shape
                and stored.dtype == np.float32
                and np.array_equal(stored, features)
                and np.array_equal(stored_ids, np.asarray(ids, dtype=str))
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return bool(ready)


def assert_single_record_api() -> None:
    """Refuse a standardization entry point that accepts more than one record."""
    z_params = list(inspect.signature(zscore_record_features).parameters)
    transform_params = list(inspect.signature(transform_record).parameters)
    if z_params != ["features"]:
        raise RuntimeError(f"zscore parameters {z_params}")
    if transform_params != ["features", "epoch_index", "variant"]:
        raise RuntimeError(f"transform parameters {transform_params}")
    z_code = zscore_record_features.__code__
    if z_code.co_argcount != 1 or z_code.co_kwonlyargcount != 0:
        raise RuntimeError("zscore_record_features must take exactly one argument")


def self_check() -> None:
    """Check the frozen formula, a gap, and that two records are not pooled."""
    assert_single_record_api()
    generator = np.random.default_rng(0)
    left = generator.normal(size=(6, N_FEATURES))
    right = generator.normal(loc=3.0, scale=2.0, size=(5, N_FEATURES))
    left[:, 1] = 4.0
    right[:, 1] = -2.0
    z_left = zscore_record_features(left)
    z_right = zscore_record_features(right)
    pooled = zscore_record_features(np.vstack([left, right]))
    separate = np.vstack([z_left, z_right])
    if np.allclose(pooled, separate):
        raise RuntimeError("two records were standardized as one")
    if not np.array_equal(z_left, zscore_record_features(left)):
        raise RuntimeError("standardization changed between calls")
    if not np.allclose(z_left[:, 1], 0.0) or not np.allclose(z_right[:, 1], 0.0):
        raise RuntimeError("a constant column was not mapped to 0")
    column = left[:, 0].astype(np.float64)
    mean = float(column.mean())
    std = float(np.sqrt(np.mean((column - mean) ** 2)))
    std_np = float(np.std(column, ddof=STD_DDOF))
    if abs(std - std_np) > 1e-12:
        raise RuntimeError("population std drifted from ddof 0")
    expected = ((column - mean) / (std + STD_EPS)).astype(np.float32)
    if not np.array_equal(z_left[:, 0], expected):
        raise RuntimeError("z-score does not match the frozen formula")
    try:
        zscore_record_features([left, right])  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise RuntimeError("a list of records was accepted")
    epoch_index = np.array([10, 11, 12, 14, 15], dtype=np.int64)
    raw = generator.normal(size=(5, N_FEATURES)).astype(np.float32)
    first = transform_record(raw, epoch_index, "c1")
    second = transform_record(raw, epoch_index, "c2")
    if first.shape != (5, C1_COLUMNS) or second.shape != (5, C2_COLUMNS):
        raise RuntimeError(f"context shapes {first.shape} {second.shape}")
    standardized = zscore_record_features(raw)
    plus1 = first[2, N_FEATURES * 3 : N_FEATURES * 4]
    plus2 = first[2, N_FEATURES * 4 : N_FEATURES * 5]
    if not np.isnan(plus1).all():
        raise RuntimeError("offset +1 crossed a gap")
    if not np.array_equal(plus2, standardized[3]):
        raise RuntimeError("offset +2 was not taken from z")
    if np.isnan(first[:, :N_FEATURES]).any() or np.isnan(first[:, -1]).any():
        raise RuntimeError("NaN in the current z block or n_neighbors")
    if not np.array_equal(second[:, -1], first[:, -1]):
        raise RuntimeError("c2 dropped the shared neighbor count")
    if not np.array_equal(second[:, N_FEATURES * 5 : N_FEATURES * 6], first[:, :N_FEATURES]):
        raise RuntimeError("c2 z block does not follow the raw block")
    if len(feature_names_for("c1")) != C1_COLUMNS or len(feature_names_for("c2")) != C2_COLUMNS:
        raise RuntimeError("feature-name width drifted")


if __name__ == "__main__":
    self_check()
    print("record_z self_check ok", flush=True)
    print(f"formula {FORMULA_TEXT}", flush=True)
    print(f"scope {SCOPE_TEXT} eps {STD_EPS} ddof {STD_DDOF}", flush=True)
    print(f"c1_columns {C1_COLUMNS} c2_columns {C2_COLUMNS}", flush=True)

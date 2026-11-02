"""Per-record readers for ``full_psg_v1``.

Signals are opened one NPZ at a time and are not retained after the
format check. Tables are joined by ``id``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SPLITS: tuple[str, ...] = ("train", "val", "test")
LABEL_COLUMNS: tuple[str, ...] = (
    "id",
    "label",
    "split",
    "record_id",
    "epoch_index",
    "subject_id",
)
SUBMISSION_COLUMNS: tuple[str, ...] = ("id", "label")
# Official full_psg_v1 sizes. Do not replace these with measured counts.
EXPECTED_RECORDS: dict[str, int] = {"train": 112, "val": 16, "test": 24}
EXPECTED_EPOCHS: dict[str, int] = {"train": 302653, "val": 43206, "test": 66222}


@dataclass(frozen=True)
class Config:
    """Paths and fixed signal contract from ``configs/config.yaml``."""

    seed: int
    signals_dir: Path
    tables_dir: Path
    cache_dir: Path
    models_dir: Path
    classes: tuple[str, ...]
    channels: tuple[str, ...]
    fs: int
    epoch_length: int
    root: Path


@dataclass(frozen=True)
class RecordView:
    """Identifiers and format facts for one recording.

    The signal matrix is checked inside :func:`load_record` and then dropped.
    """

    split: str
    record_id: str
    path: Path
    n_epochs: int
    x_shape: tuple[int, ...]
    x_dtype: str
    finite: bool
    fs: float | None
    channel_names: tuple[str, ...]
    ids: tuple[str, ...]
    epoch_index: tuple[int, ...]
    npz_keys: tuple[str, ...]


def load_config(path: Path) -> Config:
    """Load project paths relative to the repository root.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``config.yaml``. The root is its parent directory's parent.

    Returns
    -------
    Config
        Resolved directories and the signal contract.
    """
    root = path.resolve().parents[1]
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config(
        seed=int(raw["seed"]),
        signals_dir=(root / raw["signals_dir"]).resolve(),
        tables_dir=(root / raw["tables_dir"]).resolve(),
        cache_dir=(root / raw["cache_dir"]).resolve(),
        models_dir=(root / raw["models_dir"]).resolve(),
        classes=tuple(str(name) for name in raw["classes"]),
        channels=tuple(str(name) for name in raw["channels"]),
        fs=int(raw["fs"]),
        epoch_length=int(raw["epoch_length"]),
        root=root,
    )


def record_paths(signals_dir: Path, split: str) -> list[Path]:
    """Return sorted NPZ paths for one split.

    Parameters
    ----------
    signals_dir : pathlib.Path
        Directory that contains ``train``, ``val`` and ``test``.
    split : str
        Split name.

    Returns
    -------
    list of pathlib.Path
        Record files, sorted by file name.
    """
    folder = signals_dir / split
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.glob("*.npz") if path.is_file())


def _as_text(value: Any) -> str:
    """Decode one NPZ string scalar."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise TypeError(f"expected a scalar string, got shape {value.shape}")
        return _as_text(value.item())
    return str(value)


def _as_text_list(values: np.ndarray) -> tuple[str, ...]:
    """Decode a 1-D NPZ string array."""
    array = np.asarray(values)
    if array.ndim == 0:
        return (_as_text(array),)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D string array, got shape {array.shape}")
    return tuple(_as_text(item) for item in array.tolist())


def _scalar_float(value: Any) -> float:
    """Convert an NPZ scalar to ``float``."""
    array = np.asarray(value)
    if array.shape not in ((), (1,)):
        raise ValueError(f"expected a scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _all_finite(values: np.ndarray) -> bool:
    """Return whether every element is finite, without a second full copy."""
    if values.size == 0:
        return True
    if not np.issubdtype(values.dtype, np.floating):
        return False
    flat = values.reshape(-1)
    step = 1_000_000
    for start in range(0, flat.size, step):
        if not bool(np.isfinite(flat[start : start + step]).all()):
            return False
    return True


def load_record(path: Path, split: str) -> RecordView:
    """Load one NPZ, check finiteness, and release the signal.

    Parameters
    ----------
    path : pathlib.Path
        NPZ of a single recording.
    split : str
        Folder name: ``train``, ``val`` or ``test``.

    Returns
    -------
    RecordView
        Identifiers, epoch indices and format flags. ``x`` is not returned.
    """
    with np.load(path, allow_pickle=False) as archive:
        keys = tuple(archive.files)
        x = np.asarray(archive["x"]) if "x" in keys else np.empty((0,))
        finite = _all_finite(x) if "x" in keys else False
        shape = tuple(int(dim) for dim in x.shape)
        dtype = str(x.dtype)
        n_epochs = int(shape[0]) if shape else 0
        ids = _as_text_list(archive["ids"]) if "ids" in keys else ()
        if "epoch_index" in keys:
            epoch_index = tuple(
                int(item) for item in np.asarray(archive["epoch_index"]).reshape(-1)
            )
        else:
            epoch_index = ()
        fs = _scalar_float(archive["fs"]) if "fs" in keys else None
        channels = (
            _as_text_list(archive["channel_names"]) if "channel_names" in keys else ()
        )
    return RecordView(
        split=split,
        record_id=path.stem,
        path=path,
        n_epochs=n_epochs,
        x_shape=shape,
        x_dtype=dtype,
        finite=finite,
        fs=fs,
        channel_names=channels,
        ids=ids,
        epoch_index=epoch_index,
        npz_keys=keys,
    )


def record_format_issues(record: RecordView, config: Config) -> list[str]:
    """Compare one record with the fixed signal contract.

    Parameters
    ----------
    record : RecordView
        Result of :func:`load_record`.
    config : Config
        Expected sampling rate, length and channel order.

    Returns
    -------
    list of str
        Empty when the record matches the contract.
    """
    issues: list[str] = []
    required = {"x", "ids", "epoch_index", "fs", "channel_names"}
    missing = sorted(required.difference(record.npz_keys))
    if missing:
        issues.append(f"missing keys {missing}")
    expected_shape = (record.n_epochs, config.epoch_length, len(config.channels))
    if record.x_shape != expected_shape:
        issues.append(f"x shape {record.x_shape}, expected {expected_shape}")
    if record.x_dtype != "float32":
        issues.append(f"x dtype {record.x_dtype}, expected float32")
    if not record.finite:
        issues.append("x has non-finite values or is not floating-point")
    if record.channel_names != config.channels:
        issues.append(
            f"channel_names {record.channel_names}, expected {config.channels}"
        )
    if record.fs != float(config.fs):
        issues.append(f"fs {record.fs}, expected {config.fs}")
    if not (len(record.ids) == len(record.epoch_index) == record.n_epochs):
        issues.append(
            "length mismatch "
            f"ids={len(record.ids)} epoch_index={len(record.epoch_index)} "
            f"N={record.n_epochs}"
        )
    if len(set(record.ids)) != len(record.ids):
        issues.append("duplicate ids inside the record")
    steps = np.diff(np.asarray(record.epoch_index, dtype=np.int64))
    nonpositive = int(np.sum(steps <= 0)) if steps.size else 0
    if nonpositive:
        issues.append(f"epoch_index is not strictly increasing ({nonpositive} steps)")
    return issues


def gap_lengths(epoch_index: tuple[int, ...]) -> list[int]:
    """Return the length of each ``epoch_index`` gap.

    A gap is a consecutive step greater than 1. Its length is the number of
    missing 30-second slots: ``epoch_index[i + 1] - epoch_index[i] - 1``.

    Parameters
    ----------
    epoch_index : tuple of int
        Epoch indices in file order.

    Returns
    -------
    list of int
        Gap lengths. Non-positive steps are not counted as gaps.
    """
    if len(epoch_index) < 2:
        return []
    steps = np.diff(np.asarray(epoch_index, dtype=np.int64))
    return [int(step - 1) for step in steps if int(step) > 1]


def read_labels(path: Path) -> pd.DataFrame:
    """Read ``labels.csv`` without reordering rows.

    Parameters
    ----------
    path : pathlib.Path
        Labels table for train and val.

    Returns
    -------
    pandas.DataFrame
        Columns kept as strings so missing cells stay visible.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    return frame


def read_submission(path: Path) -> pd.DataFrame:
    """Read ``sample_submission.csv`` without reordering rows.

    Parameters
    ----------
    path : pathlib.Path
        Submission template. ``label`` values are placeholders.

    Returns
    -------
    pandas.DataFrame
        String columns in file order.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def manifest_local_path(
    manifest_path: str,
    config: Config,
) -> Path | None:
    """Map a manifest path onto this checkout.

    Parameters
    ----------
    manifest_path : str
        Path stored in ``MANIFEST.json``, using forward slashes.
    config : Config
        Local signal and table directories.

    Returns
    -------
    pathlib.Path or None
        Local file, or ``None`` for ``team_info.csv``, which is not opened.
    """
    parts = tuple(Path(manifest_path).parts)
    if parts and parts[-1] == "team_info.csv":
        return None
    if len(parts) == 3 and parts[0] == "epochs" and parts[1] in SPLITS:
        return config.signals_dir / parts[1] / parts[2]
    if len(parts) == 1:
        return config.tables_dir / parts[0]
    return config.tables_dir.joinpath(*parts)


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Hash a file without reading it all at once.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.
    chunk_bytes : int, optional
        Read size in bytes.

    Returns
    -------
    str
        Lower-case SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

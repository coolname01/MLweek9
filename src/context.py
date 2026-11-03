"""Same-record neighbor features at ``epoch_index`` offsets -2, -1, +1, +2.

A neighbor is copied only when that index exists in the same record. A gap
or a record edge is NaN. Records are never stacked before the lookup.
Base columns come from ``cache/features_v1`` and are not recomputed.
``epoch_index`` is taken from ``labels.csv`` joined by ``id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import EXPECTED_RECORDS, Config, _as_text_list
from src.features import (
    CACHE_NAME,
    N_FEATURES,
    _cache_ready,
    cache_file,
    feature_names,
)

OFFSETS: tuple[int, ...] = (-2, -1, 1, 2)
N_CONTEXT_FEATURES = N_FEATURES * (1 + len(OFFSETS)) + 1


@dataclass(frozen=True)
class RecordContext:
    """One recording after neighbor columns have been attached."""

    split: str
    record_id: str
    ids: np.ndarray
    epoch_index: np.ndarray
    base: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ContextSplit:
    """Context matrix for a whole split, rows aligned with ``ids``."""

    features: np.ndarray
    ids: np.ndarray
    feature_names: tuple[str, ...]
    base: np.ndarray | None


@dataclass(frozen=True)
class GapRow:
    """One epoch in the window around an ``epoch_index`` gap."""

    row: int
    epoch_index: int
    sources: tuple[int | None, ...]
    n_neighbors: int
    next_epoch: int | None
    plus1_is_next_row: bool
    copy_ok: bool


def context_feature_names(base: tuple[str, ...]) -> tuple[str, ...]:
    """Return base names, then one block per offset, then ``n_neighbors``.

    Parameters
    ----------
    base : tuple of str
        Column names of the current epoch, in cache order.

    Returns
    -------
    tuple of str
        Length ``len(base) * 5 + 1``. Offset blocks follow :data:`OFFSETS`.
    """
    names = list(base)
    for offset in OFFSETS:
        prefix = f"off{offset:+d}_"
        names.extend(prefix + name for name in base)
    names.append("n_neighbors")
    if len(names) != len(base) * (1 + len(OFFSETS)) + 1:
        raise RuntimeError("context name width drifted")
    return tuple(names)


def apply_context(features: np.ndarray, epoch_index: np.ndarray) -> np.ndarray:
    """Attach neighbor features for one recording.

    Parameters
    ----------
    features : ndarray, shape (n_epochs, n_features)
        Finite features of this recording only, in file order.
    epoch_index : ndarray of int, shape (n_epochs,)
        ``epoch_index`` aligned with ``features``. Values may be unsorted.
        Duplicates are refused.

    Returns
    -------
    ndarray, shape (n_epochs, n_features * 5 + 1), float32
        Current features, then offsets -2, -1, +1, +2, then the number of
        offsets that existed in this recording (0..4). Missing neighbors
        are NaN. LightGBM treats those NaNs as missing values.
    """
    values = np.ascontiguousarray(features, dtype=np.float32)
    index = np.asarray(epoch_index, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != index.shape[0]:
        raise ValueError(
            f"features {values.shape} do not match epoch_index {index.shape}"
        )
    n_epochs, width = values.shape
    if n_epochs == 0:
        raise ValueError("empty record")
    if not np.isfinite(values).all():
        raise ValueError("base features must be finite")
    order = np.argsort(index, kind="mergesort")
    ordered = index[order]
    if n_epochs > 1 and bool(np.any(np.diff(ordered) == 0)):
        raise ValueError("duplicate epoch_index in one record")
    out = np.empty((n_epochs, width * (1 + len(OFFSETS)) + 1), dtype=np.float32)
    out[:, :width] = values
    counts = np.zeros(n_epochs, dtype=np.int8)
    for slot, offset in enumerate(OFFSETS):
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
        counts += matched.astype(np.int8)
    out[:, -1] = counts.astype(np.float32)
    _assert_blocks(out, width)
    return out


def _assert_blocks(context: np.ndarray, width: int) -> None:
    """Refuse partial neighbor rows and a count that disagrees with NaNs."""
    if np.isnan(context[:, :width]).any() or np.isnan(context[:, -1]).any():
        raise RuntimeError("NaN in the current epoch or in n_neighbors")
    found = np.zeros(context.shape[0], dtype=np.float32)
    for slot in range(len(OFFSETS)):
        block = context[:, width * (slot + 1) : width * (slot + 2)]
        missing = np.isnan(block)
        partial = missing.any(axis=1) & ~missing.all(axis=1)
        if bool(partial.any()):
            raise RuntimeError("neighbor row is only partly NaN")
        found += (~missing.all(axis=1)).astype(np.float32)
    if not np.array_equal(found, context[:, -1]):
        raise RuntimeError("n_neighbors does not match the NaN pattern")
    if float(context[:, -1].min()) < 0 or float(context[:, -1].max()) > len(OFFSETS):
        raise RuntimeError("n_neighbors outside 0..4")


def label_index_for_split(labels: pd.DataFrame, split: str) -> pd.DataFrame:
    """Index ``id`` to ``record_id`` and ``epoch_index`` for one split.

    Parameters
    ----------
    labels : pandas.DataFrame
        ``labels.csv``. Rows are not reordered; the index is ``id``.
    split : str
        ``train`` or ``val``. ``test`` is refused.

    Returns
    -------
    pandas.DataFrame
        Unique ``id`` index and columns ``record_id``, ``epoch_index``.
    """
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    columns = ["id", "record_id", "epoch_index", "split"]
    missing = [name for name in columns if name not in labels.columns]
    if missing:
        raise ValueError(f"labels missing {missing}")
    part = labels.loc[labels["split"].astype(str) == split, columns]
    if part.empty:
        raise ValueError(f"no labels for {split}")
    ids = part["id"].astype(str)
    if bool(ids.duplicated().any()):
        raise ValueError(f"duplicate ids in {split} labels")
    epoch = pd.to_numeric(part["epoch_index"], errors="raise")
    if bool(epoch.isna().any()) or bool((epoch != np.floor(epoch)).any()):
        raise ValueError(f"{split} epoch_index is not an integer")
    frame = pd.DataFrame(
        {
            "record_id": part["record_id"].astype(str).to_numpy(),
            "epoch_index": epoch.astype(np.int64).to_numpy(),
        },
        index=pd.Index(ids.to_numpy(), name="id"),
    )
    return frame


def cached_record_ids(config: Config, split: str) -> list[str]:
    """Return sorted cache stems for one split.

    Parameters
    ----------
    config : Config
        Project paths.
    split : str
        ``train`` or ``val``.

    Returns
    -------
    list of str
        Record ids. The test cache is not listed.
    """
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    folder = config.cache_dir / CACHE_NAME / split
    paths = sorted(
        path
        for path in folder.glob("*.npz")
        if path.is_file() and not path.name.startswith(".")
    )
    if len(paths) != EXPECTED_RECORDS[split]:
        raise RuntimeError(
            f"{split}: {len(paths)} cache files, expected {EXPECTED_RECORDS[split]}. "
            "Refusing to recompute features."
        )
    return [path.stem for path in paths]


def load_record_context(
    config: Config,
    split: str,
    record_id: str,
    label_index: pd.DataFrame,
) -> RecordContext:
    """Load one cached recording and attach its neighbors.

    Parameters
    ----------
    config : Config
        Project paths.
    split : str
        ``train`` or ``val``.
    record_id : str
        Cache file stem.
    label_index : pandas.DataFrame
        Frame from :func:`label_index_for_split`.

    Returns
    -------
    RecordContext
        Base matrix and context matrix for this recording only.
    """
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    path = cache_file(config, split, record_id)
    if not _cache_ready(path):
        raise RuntimeError(f"cache missing or stale: {path}. Refusing to recompute.")
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
        raise ValueError(f"{split}/{record_id} label record_id does not match the cache")
    epoch_index = located["epoch_index"].to_numpy(dtype=np.int64)
    features = apply_context(base, epoch_index)
    if features.shape != (base.shape[0], N_CONTEXT_FEATURES):
        raise RuntimeError(f"{split}/{record_id} context shape {features.shape}")
    return RecordContext(
        split=split,
        record_id=record_id,
        ids=ids,
        epoch_index=epoch_index,
        base=base,
        features=features,
        feature_names=context_feature_names(names),
    )


def load_context_split(
    config: Config,
    split: str,
    label_index: pd.DataFrame,
    record_ids: list[str] | None = None,
    keep_base: bool = False,
) -> ContextSplit:
    """Build a split matrix one recording at a time.

    Parameters
    ----------
    config : Config
        Project paths.
    split : str
        ``train`` or ``val``.
    label_index : pandas.DataFrame
        Frame from :func:`label_index_for_split`.
    record_ids : list of str or None
        Subset in this order. ``None`` loads every cached recording, sorted
        by file name, which matches :func:`src.features.load_cached_split`.
    keep_base : bool, optional
        When True, also return the 54 base columns. They stay row-aligned
        with the context matrix.

    Returns
    -------
    ContextSplit
        Float32 context features. The test split is refused.
    """
    ids_to_load = cached_record_ids(config, split) if record_ids is None else list(record_ids)
    if not ids_to_load:
        raise ValueError(f"no records selected for {split}")
    context_blocks: list[np.ndarray] = []
    base_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for record_id in ids_to_load:
        record = load_record_context(config, split, record_id, label_index)
        if names is None:
            names = record.feature_names
        elif names != record.feature_names:
            raise RuntimeError("context feature names differ between records")
        _assert_edges(record)
        context_blocks.append(record.features)
        id_blocks.append(record.ids)
        if keep_base:
            base_blocks.append(record.base)
    if names is None:
        raise RuntimeError(f"{split} produced no context columns")
    base = np.concatenate(base_blocks, axis=0) if keep_base else None
    return ContextSplit(
        features=np.concatenate(context_blocks, axis=0),
        ids=np.concatenate(id_blocks, axis=0),
        feature_names=names,
        base=base,
    )


def _assert_edges(record: RecordContext) -> None:
    """A missing index at either end of the recording stays NaN."""
    width = record.base.shape[1]
    present = set(int(value) for value in record.epoch_index.tolist())
    ends = (
        (0, int(record.epoch_index[0])),
        (-1, int(record.epoch_index[-1])),
    )
    for row, epoch in ends:
        for slot, offset in enumerate(OFFSETS):
            block = record.features[row, width * (slot + 1) : width * (slot + 2)]
            if (epoch + offset) in present:
                if bool(np.isnan(block).any()):
                    raise RuntimeError(
                        f"{record.record_id} dropped an existing neighbor at row {row}"
                    )
            elif not bool(np.isnan(block).all()):
                raise RuntimeError(
                    f"{record.record_id} filled a missing edge at row {row} offset {offset}"
                )


def read_record_index(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``ids`` and ``epoch_index`` from one NPZ, never the signal.

    Parameters
    ----------
    path : pathlib.Path
        Train or val recording. A path inside a ``test`` directory is refused.

    Returns
    -------
    ids : ndarray of str
        Epoch ids in file order.
    epoch_index : ndarray of int64
        ``epoch_index`` in the same order.
    """
    if "test" in {part.lower() for part in path.parts}:
        raise RuntimeError(f"refusing to read a test record: {path}")
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(_as_text_list(archive["ids"]), dtype=str)
        epoch_index = np.asarray(archive["epoch_index"], dtype=np.int64).reshape(-1)
    if ids.shape[0] != epoch_index.shape[0]:
        raise ValueError(f"{path.name} ids and epoch_index differ in length")
    return ids, epoch_index


def assert_labels_match_record(
    record: RecordContext,
    npz_ids: np.ndarray,
    npz_epoch_index: np.ndarray,
) -> None:
    """Check the label join against the recording's own ``epoch_index``.

    Parameters
    ----------
    record : RecordContext
        Cache row order and the label-joined index.
    npz_ids, npz_epoch_index : ndarray
        Arrays from :func:`read_record_index`. The signal is not involved.

    Raises
    ------
    RuntimeError
        If ids or epoch indexes differ.
    """
    if record.ids.shape != npz_ids.shape or not np.array_equal(record.ids, npz_ids):
        raise RuntimeError(f"{record.record_id} cache ids differ from the NPZ ids")
    if not np.array_equal(record.epoch_index, npz_epoch_index):
        raise RuntimeError(
            f"{record.record_id} label epoch_index differs from the NPZ epoch_index"
        )


def gap_rows(features: np.ndarray, epoch_index: np.ndarray, context: np.ndarray) -> tuple[GapRow, ...]:
    """Return five rows around the first ``epoch_index`` gap.

    The row just before the gap must not take offset +1 from the next file
    row. A one-slot gap (step 2) must take offset +2 from that next epoch.

    Parameters
    ----------
    features : ndarray
        Base features of one recording.
    epoch_index : ndarray of int64
        Index aligned with ``features``.
    context : ndarray
        Output of :func:`apply_context` for the same recording.

    Returns
    -------
    tuple of GapRow
        Five file rows. ``sources`` are epoch indexes, or None when absent.
    """
    index = np.asarray(epoch_index, dtype=np.int64).reshape(-1)
    n_epochs = int(index.shape[0])
    if n_epochs < 5:
        raise ValueError("need at least 5 epochs to show a gap window")
    jumps = np.flatnonzero(np.diff(index) > 1)
    if jumps.size == 0:
        raise ValueError("recording has no epoch_index gap")
    anchor = int(jumps[0])
    step = int(index[anchor + 1] - index[anchor])
    start = anchor - 2
    stop = start + 5
    if start < 0:
        start = 0
        stop = 5
    if stop > n_epochs:
        stop = n_epochs
        start = n_epochs - 5
    rows = list(range(start, stop))
    if anchor not in rows or (anchor + 1) not in rows:
        raise RuntimeError("gap window missed the jump")
    width = int(features.shape[1])
    lookup = {int(epoch): row for row, epoch in enumerate(index.tolist())}
    described: list[GapRow] = []
    for row in rows:
        sources: list[int | None] = []
        copy_ok = bool(np.array_equal(context[row, :width], features[row]))
        count = 0
        for slot, offset in enumerate(OFFSETS):
            source = lookup.get(int(index[row]) + int(offset))
            block = context[row, width * (slot + 1) : width * (slot + 2)]
            if source is None:
                sources.append(None)
                copy_ok = copy_ok and bool(np.isnan(block).all())
            else:
                sources.append(int(index[source]))
                count += 1
                copy_ok = copy_ok and bool(np.array_equal(block, features[source]))
        if int(context[row, -1]) != count:
            copy_ok = False
        next_epoch = int(index[row + 1]) if row + 1 < n_epochs else None
        plus1 = sources[OFFSETS.index(1)]
        plus1_is_next_row = plus1 is not None and next_epoch is not None and plus1 == next_epoch
        described.append(
            GapRow(
                row=row,
                epoch_index=int(index[row]),
                sources=tuple(sources),
                n_neighbors=count,
                next_epoch=next_epoch,
                plus1_is_next_row=plus1_is_next_row,
                copy_ok=copy_ok,
            )
        )
    anchor_row = described[rows.index(anchor)]
    if anchor_row.sources[OFFSETS.index(1)] is not None or anchor_row.plus1_is_next_row:
        raise RuntimeError("offset +1 followed the next file row across a gap")
    if step == 2 and anchor_row.sources[OFFSETS.index(2)] != int(index[anchor + 1]):
        raise RuntimeError("offset +2 missed epoch_index+2 on a one-slot gap")
    if not all(row.copy_ok for row in described):
        raise RuntimeError("neighbor values do not match the epoch_index lookup")
    return tuple(described)


def matrix_nbytes(n_rows: int) -> int:
    """Return the float32 size of a context matrix with ``n_rows`` epochs.

    Parameters
    ----------
    n_rows : int
        Epoch count.

    Returns
    -------
    int
        Bytes of an ``(n_rows, N_CONTEXT_FEATURES)`` float32 array.
    """
    return int(n_rows) * int(N_CONTEXT_FEATURES) * int(np.dtype(np.float32).itemsize)


def self_check() -> None:
    """Check gap lookup, record edges, and unsorted indexes on a toy matrix."""
    width = 2
    features = np.arange(5 * width, dtype=np.float32).reshape(5, width)
    epoch_index = np.array([10, 11, 12, 14, 15], dtype=np.int64)
    out = apply_context(features, epoch_index)
    if out.shape != (5, width * 5 + 1) or out.dtype != np.float32:
        raise RuntimeError(f"self-check shape {out.shape} {out.dtype}")
    if not np.array_equal(out[2, :width], features[2]):
        raise RuntimeError("current epoch was replaced")
    if not np.array_equal(out[2, width : 2 * width], features[0]):
        raise RuntimeError("offset -2")
    if not np.array_equal(out[2, 2 * width : 3 * width], features[1]):
        raise RuntimeError("offset -1")
    if not np.isnan(out[2, 3 * width : 4 * width]).all():
        raise RuntimeError("offset +1 crossed a gap")
    if not np.array_equal(out[2, 4 * width : 5 * width], features[3]):
        raise RuntimeError("offset +2")
    if float(out[2, -1]) != 3.0 or float(out[0, -1]) != 2.0:
        raise RuntimeError("n_neighbors")
    if not np.isnan(out[0, width : 3 * width]).all():
        raise RuntimeError("leading edge borrowed a neighbor")
    scrambled_epoch = np.array([12, 10, 15, 11, 14], dtype=np.int64)
    scrambled = features[np.array([2, 0, 4, 1, 3])]
    scrambled_out = apply_context(scrambled, scrambled_epoch)
    if not np.isnan(scrambled_out[0, 3 * width : 4 * width]).all():
        raise RuntimeError("unsorted gap")
    if not np.array_equal(scrambled_out[0, 4 * width : 5 * width], scrambled[4]):
        raise RuntimeError("unsorted offset +2")
    right = apply_context(features[3:], epoch_index[3:])
    if not np.isnan(right[0, width : 3 * width]).all() or float(right[0, -1]) != 1.0:
        raise RuntimeError("two slices were treated as one recording")
    try:
        apply_context(features[:2], np.array([1, 1], dtype=np.int64))
    except ValueError:
        pass
    else:
        raise RuntimeError("duplicate epoch_index was accepted")
    names = context_feature_names(("a", "b"))
    expected = (
        "a",
        "b",
        "off-2_a",
        "off-2_b",
        "off-1_a",
        "off-1_b",
        "off+1_a",
        "off+1_b",
        "off+2_a",
        "off+2_b",
        "n_neighbors",
    )
    if names != expected:
        raise RuntimeError(f"names {names}")
    described = gap_rows(features, epoch_index, out)
    anchor = next(row for row in described if row.epoch_index == 12)
    if anchor.sources != (10, 11, None, 14) or anchor.plus1_is_next_row:
        raise RuntimeError(f"gap sources {anchor.sources}")
    if matrix_nbytes(1) != N_CONTEXT_FEATURES * 4:
        raise RuntimeError("matrix size")

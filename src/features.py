"""Welch band powers and amplitude statistics, one record at a time.

Each cache file is ``cache/features_v1/{split}/{record_id}.npz``. Statistics
used for scaling are the mean and standard deviation of that record only.
The test split is never read.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.signal import welch

from src.data import (
    EXPECTED_RECORDS,
    Config,
    _as_text_list,
    _scalar_float,
    load_config,
    read_labels,
    record_paths,
)

FEATURE_VERSION = "v1"
FS = 100
EPOCH_LENGTH = 3000
NPERSEG = 256
NOVERLAP = 128
ZSCORE = "per_record_channel"
LOG_POWER_FLOOR = 1e-20
FULL_RECORD_COUNT = 152
TIME_LIMIT_S = 60.0 * 60.0
CACHE_NAME = "features_v1"

CHANNEL_SPECS: tuple[tuple[str, str], ...] = (
    ("EEG Fpz-Cz", "eeg_fpz_cz"),
    ("EEG Pz-Oz", "eeg_pz_oz"),
    ("EOG horizontal", "eog_horizontal"),
)
BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
)
AMPLITUDE_NAMES: tuple[str, ...] = (
    "mean_abs",
    "std",
    "skew",
    "kurtosis",
    "range",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
)
FEATURES_PER_CHANNEL = len(BANDS) * 2 + len(AMPLITUDE_NAMES)
N_FEATURES = len(CHANNEL_SPECS) * FEATURES_PER_CHANNEL
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FeatureTable:
    """Feature matrix aligned with epoch ids from one split."""

    features: np.ndarray
    ids: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class RecordTiming:
    """Timing and memory for one cache write or cache hit."""

    split: str
    record_id: str
    skipped: bool
    seconds: float
    n_epochs: int
    private_mb: float
    working_set_mb: float
    cache_bytes: int


@dataclass(frozen=True)
class ExtractReport:
    """Aggregate of one extraction call."""

    wall_s: float
    n_jobs: int
    computed: int
    skipped: int
    mean_record_s: float
    max_private_mb: float
    max_working_set_mb: float
    cache_bytes: int
    epochs: int
    records: tuple[RecordTiming, ...]


def feature_names() -> tuple[str, ...]:
    """Return the column names in the order stored in the cache.

    Returns
    -------
    tuple of str
        Length :data:`N_FEATURES`. Spectral pairs come before amplitude
        statistics, and channels follow :data:`CHANNEL_SPECS`.
    """
    names: list[str] = []
    for _channel_name, slug in CHANNEL_SPECS:
        for band_name, _low, _high in BANDS:
            names.append(f"{slug}_{band_name}_logpower")
            names.append(f"{slug}_{band_name}_relpower")
        names.extend(f"{slug}_{stat}" for stat in AMPLITUDE_NAMES)
    return tuple(names)


def worker_count() -> int:
    """Return the process count: logical CPUs minus two, at least one.

    Returns
    -------
    int
        ``n_jobs`` for :class:`joblib.Parallel`.
    """
    count = os.cpu_count() or 1
    return max(1, count - 2)


def cache_file(config: Config, split: str, record_id: str) -> Path:
    """Return the feature-cache path for one recording.

    Parameters
    ----------
    config : Config
        Project paths.
    split : str
        ``train`` or ``val``.
    record_id : str
        File stem of the recording.

    Returns
    -------
    pathlib.Path
        ``cache/features_v1/{split}/{record_id}.npz``.
    """
    return config.cache_dir / CACHE_NAME / split / f"{record_id}.npz"


def _config_path() -> Path:
    return ROOT / "configs" / "config.yaml"


def _push_one_thread() -> dict[str, str | None]:
    previous = {name: os.environ.get(name) for name in _THREAD_VARS}
    for name in _THREAD_VARS:
        os.environ[name] = "1"
    return previous


def _pop_threads(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _memory_status() -> tuple[float, float]:
    """Return total and available physical memory in mebibytes."""

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    scale = 1024.0 * 1024.0
    return status.ullTotalPhys / scale, status.ullAvailPhys / scale


def _process_peaks_mb() -> tuple[float, float]:
    """Return peak private bytes and peak working set of this process."""

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return float("nan"), float("nan")
    scale = 1024.0 * 1024.0
    return counters.PeakPagefileUsage / scale, counters.PeakWorkingSetSize / scale


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _lock_path(config: Config) -> Path:
    return config.cache_dir / CACHE_NAME / ".experiment.lock"


def acquire_lock(config: Config) -> Path:
    """Create the extraction lock, refusing a live holder.

    A lock whose process is gone is removed. Test files are not touched.

    Parameters
    ----------
    config : Config
        Project paths. The lock lives under the feature cache.

    Returns
    -------
    pathlib.Path
        Lock file. The caller must pass it to :func:`release_lock`.
    """
    path = _lock_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raw = path.read_text(encoding="utf-8").strip()
        pid = int(raw) if raw.isdigit() else -1
        if _pid_is_running(pid):
            raise RuntimeError(f"another experiment is running (pid {pid}, lock {path})")
        path.unlink()
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return path


def release_lock(path: Path) -> None:
    """Remove the lock when this process still owns it.

    Parameters
    ----------
    path : pathlib.Path
        File returned by :func:`acquire_lock`.
    """
    if not path.is_file():
        return
    owner = path.read_text(encoding="utf-8").strip()
    if owner == str(os.getpid()):
        path.unlink()


def _text(value: Any) -> str:
    array = np.asarray(value)
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _cache_ready(path: Path) -> bool:
    """Return whether ``path`` matches the current feature version."""
    if not path.is_file() or path.name.startswith("."):
        return False
    expected = feature_names()
    try:
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"])
            ids = _as_text_list(archive["ids"])
            names = _as_text_list(archive["feature_names"])
            ready = (
                _text(archive["version"]) == FEATURE_VERSION
                and int(np.asarray(archive["fs"]).reshape(-1)[0]) == FS
                and int(np.asarray(archive["nperseg"]).reshape(-1)[0]) == NPERSEG
                and int(np.asarray(archive["noverlap"]).reshape(-1)[0]) == NOVERLAP
                and _text(archive["zscore"]) == ZSCORE
                and names == expected
                and features.dtype == np.float32
                and features.ndim == 2
                and features.shape[1] == N_FEATURES
                and features.shape[0] == len(ids)
                and len(set(ids)) == len(ids)
                and bool(np.isfinite(features).all())
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return ready


def _band_mask(freqs: np.ndarray, band_index: int) -> np.ndarray:
    _name, low, high = BANDS[band_index]
    if band_index == len(BANDS) - 1:
        return (freqs >= low) & (freqs <= high)
    return (freqs >= low) & (freqs < high)


def _check_band_grid(fs: float) -> None:
    freqs = np.fft.rfftfreq(NPERSEG, d=1.0 / fs)
    for index, (name, _low, _high) in enumerate(BANDS):
        if int(_band_mask(freqs, index).sum()) < 2:
            raise RuntimeError(f"band {name} has fewer than 2 Welch bins at fs={fs}")


def _zscore_record(x: np.ndarray) -> None:
    """Z-score each channel with this record's own mean and std, in place.

    Parameters
    ----------
    x : ndarray, shape (n_epochs, n_samples, n_channels)
        Float32 signal owned by the caller. Channels are scaled independently.
        A channel with standard deviation below ``1e-8`` is set to zero.
    """
    for index in range(x.shape[-1]):
        channel = x[:, :, index]
        values = np.ascontiguousarray(channel, dtype=np.float64).reshape(-1)
        mean = float(values.mean())
        variance = float(np.mean((values - mean) ** 2))
        scale = math.sqrt(variance)
        if scale < 1e-8:
            channel.fill(0.0)
            continue
        np.subtract(channel, np.float32(mean), out=channel, casting="unsafe")
        np.divide(channel, np.float32(scale), out=channel, casting="unsafe")


def _spectral_block(channel: np.ndarray, fs: float) -> np.ndarray:
    """Log absolute power and relative power for every band.

    Parameters
    ----------
    channel : ndarray, shape (n_epochs, n_samples)
        One z-scored channel.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    ndarray, shape (n_epochs, 10)
        Pairs ``(log10 band power, relative band power)`` in band order.
        Relative power is the band integral divided by the sum of the five
        band integrals. The log uses a fixed floor of :data:`LOG_POWER_FLOOR`.
    """
    freqs, psd = welch(
        channel,
        fs=fs,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NPERSEG,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
    )
    spectrum = np.maximum(np.asarray(psd, dtype=np.float64), 0.0)
    if spectrum.shape[0] != channel.shape[0]:
        raise RuntimeError(f"welch returned shape {spectrum.shape}")
    powers = np.empty((channel.shape[0], len(BANDS)), dtype=np.float64)
    frequency = np.asarray(freqs, dtype=np.float64)
    for index in range(len(BANDS)):
        mask = _band_mask(frequency, index)
        powers[:, index] = np.trapezoid(spectrum[:, mask], frequency[mask], axis=1)
    powers = np.maximum(powers, 0.0)
    total = powers.sum(axis=1, keepdims=True)
    relative = np.zeros_like(powers)
    np.divide(powers, total, out=relative, where=total > 0)
    log_power = np.log10(np.maximum(powers, LOG_POWER_FLOOR))
    block = np.empty((channel.shape[0], len(BANDS) * 2), dtype=np.float64)
    block[:, 0::2] = log_power
    block[:, 1::2] = relative
    return block


def _amplitude_block(channel: np.ndarray) -> np.ndarray:
    """Amplitude moments and Hjorth parameters for one channel.

    Kurtosis is the excess (Fisher) moment. Hjorth activity is the population
    variance. Mobility and complexity use first and second differences.
    Constant epochs get skew, kurtosis, mobility and complexity equal to 0.

    Parameters
    ----------
    channel : ndarray, shape (n_epochs, n_samples)
        One z-scored channel.

    Returns
    -------
    ndarray, shape (n_epochs, 8)
        Columns follow :data:`AMPLITUDE_NAMES`.
    """
    values = np.asarray(channel, dtype=np.float64)
    n_epochs = values.shape[0]
    centered = values - values.mean(axis=1, keepdims=True)
    second = centered * centered
    m2 = np.mean(second, axis=1)
    m3 = np.mean(second * centered, axis=1)
    m4 = np.mean(second * second, axis=1)
    std = np.sqrt(m2)
    skew = np.zeros(n_epochs, dtype=np.float64)
    kurtosis = np.zeros(n_epochs, dtype=np.float64)
    usable = m2 > 1e-20
    skew[usable] = m3[usable] / np.power(m2[usable], 1.5)
    kurtosis[usable] = m4[usable] / np.square(m2[usable]) - 3.0
    mean_abs = np.mean(np.abs(values), axis=1)
    data_range = values.max(axis=1) - values.min(axis=1)

    first = np.diff(values, axis=1)
    first_centered = first - first.mean(axis=1, keepdims=True)
    v1 = np.mean(first_centered * first_centered, axis=1)
    second_diff = np.diff(first, axis=1)
    second_centered = second_diff - second_diff.mean(axis=1, keepdims=True)
    v2 = np.mean(second_centered * second_centered, axis=1)

    activity = m2
    mobility = np.zeros(n_epochs, dtype=np.float64)
    complexity = np.zeros(n_epochs, dtype=np.float64)
    moving = activity > 1e-20
    mobility[moving] = np.sqrt(v1[moving] / activity[moving])
    curved = moving & (v1 > 1e-20) & (mobility > 1e-20)
    complexity[curved] = np.sqrt(v2[curved] / v1[curved]) / mobility[curved]
    return np.column_stack(
        (mean_abs, std, skew, kurtosis, data_range, activity, mobility, complexity)
    )


def _check_relative(features: np.ndarray) -> None:
    width = FEATURES_PER_CHANNEL
    for channel in range(len(CHANNEL_SPECS)):
        columns = [channel * width + 1 + 2 * band for band in range(len(BANDS))]
        total = features[:, columns].sum(axis=1)
        bad = ~((np.abs(total - 1.0) < 1e-4) | (np.abs(total) < 1e-4))
        if bool(bad.any()):
            raise RuntimeError("relative band powers do not sum to 1")


def features_from_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """Compute per-epoch features for one recording.

    The array is z-scored in place. No statistic is taken from any other
    recording.

    Parameters
    ----------
    x : ndarray, shape (n_epochs, n_samples, n_channels)
        Float32 signal in channel order ``EEG Fpz-Cz``, ``EEG Pz-Oz``,
        ``EOG horizontal``.
    fs : float
        Sampling rate in Hz. Must be 100.

    Returns
    -------
    ndarray, shape (n_epochs, 54)
        Float64 features in :func:`feature_names` order.
    """
    if fs != FS:
        raise ValueError(f"fs {fs}, expected {FS}")
    if x.ndim != 3 or x.shape[1:] != (EPOCH_LENGTH, len(CHANNEL_SPECS)):
        raise ValueError(f"expected (N, {EPOCH_LENGTH}, 3), got {x.shape}")
    _check_band_grid(fs)
    _zscore_record(x)
    out = np.empty((x.shape[0], N_FEATURES), dtype=np.float64)
    spectral_width = len(BANDS) * 2
    for index in range(x.shape[-1]):
        channel = np.ascontiguousarray(x[:, :, index], dtype=np.float64)
        spectral = _spectral_block(channel, fs)
        amplitude = _amplitude_block(channel)
        start = index * FEATURES_PER_CHANNEL
        out[:, start : start + spectral_width] = spectral
        out[:, start + spectral_width : start + FEATURES_PER_CHANNEL] = amplitude
    _check_relative(out)
    if not np.isfinite(out).all():
        raise RuntimeError("features are not finite")
    return out


def _self_check() -> None:
    """Pure sines must land in alpha, delta and beta."""
    time_s = np.arange(EPOCH_LENGTH, dtype=np.float64) / FS
    signal = np.zeros((4, EPOCH_LENGTH, 3), dtype=np.float32)
    signal[:, :, 0] = np.sin(2.0 * np.pi * 10.0 * time_s)
    signal[:, :, 1] = np.sin(2.0 * np.pi * 2.0 * time_s)
    signal[:, :, 2] = np.sin(2.0 * np.pi * 20.0 * time_s)
    features = features_from_signal(signal, FS)
    expected = {0: 2, 1: 0, 2: 4}
    for channel, band in expected.items():
        columns = [
            channel * FEATURES_PER_CHANNEL + 1 + 2 * index for index in range(len(BANDS))
        ]
        relative = features[0, columns]
        winner = int(np.argmax(relative))
        if winner != band or float(relative[winner]) < 0.5:
            raise RuntimeError(
                f"self-check channel {channel}: relative {relative.tolist()}, "
                f"expected band {band}"
            )
    if features.shape != (4, N_FEATURES):
        raise RuntimeError(f"self-check shape {features.shape}")


def _load_signal(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    config = load_config(_config_path())
    with np.load(path, allow_pickle=False) as archive:
        signal = np.array(archive["x"], dtype=np.float32, copy=True)
        ids = _as_text_list(archive["ids"])
        channels = _as_text_list(archive["channel_names"])
        fs = _scalar_float(archive["fs"])
    if signal.shape[1:] != (EPOCH_LENGTH, len(CHANNEL_SPECS)):
        raise ValueError(f"x shape {signal.shape}")
    if signal.dtype != np.float32:
        raise ValueError(f"x dtype {signal.dtype}")
    if fs != float(FS) or config.fs != FS or config.epoch_length != EPOCH_LENGTH:
        raise ValueError(f"fs/length contract mismatch fs={fs}")
    if channels != config.channels or channels != tuple(name for name, _slug in CHANNEL_SPECS):
        raise ValueError(f"channel_names {channels}")
    if len(ids) != signal.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("id count does not match unique epochs")
    if signal.shape[0] == 0 or not np.isfinite(signal).all():
        raise ValueError("signal is empty or non-finite")
    return signal, ids


def _write_cache(path: Path, features: np.ndarray, ids: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.npz")
    payload = {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "ids": np.asarray(ids),
        "feature_names": np.asarray(feature_names()),
        "version": np.asarray(FEATURE_VERSION),
        "fs": np.int64(FS),
        "nperseg": np.int64(NPERSEG),
        "noverlap": np.int64(NOVERLAP),
        "zscore": np.asarray(ZSCORE),
    }
    try:
        np.savez(temporary, **payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not _cache_ready(path):
        raise RuntimeError(f"wrote an unreadable cache file {path}")


def extract_one(npz_path: str, split: str, cache_file_path: str) -> RecordTiming:
    """Extract or reuse features for a single recording.

    Parameters
    ----------
    npz_path : str
        Source NPZ. Opened once with ``np.load`` and then released.
    split : str
        ``train`` or ``val``. ``test`` is refused.
    cache_file_path : str
        Destination ``.npz``. A matching version is not recomputed.

    Returns
    -------
    RecordTiming
        Elapsed seconds, epoch count and peak memory of this process.
    """
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    started = time.perf_counter()
    source = Path(npz_path)
    destination = Path(cache_file_path)
    record_id = source.stem
    try:
        if _cache_ready(destination):
            with np.load(destination, allow_pickle=False) as archive:
                n_epochs = int(archive["features"].shape[0])
            skipped = True
        else:
            signal, ids = _load_signal(source)
            features = features_from_signal(signal, float(FS))
            del signal
            if features.shape != (len(ids), N_FEATURES):
                raise RuntimeError(f"feature shape {features.shape}")
            _write_cache(destination, features, ids)
            n_epochs = len(ids)
            skipped = False
        private_mb, working_set_mb = _process_peaks_mb()
        timing = RecordTiming(
            split=split,
            record_id=record_id,
            skipped=skipped,
            seconds=time.perf_counter() - started,
            n_epochs=n_epochs,
            private_mb=private_mb,
            working_set_mb=working_set_mb,
            cache_bytes=destination.stat().st_size,
        )
    except Exception as exc:
        raise RuntimeError(f"{split}/{record_id} failed: {exc}") from exc
    print(
        f"{timing.split}/{timing.record_id} epochs={timing.n_epochs} "
        f"seconds={timing.seconds:.2f} skipped={int(timing.skipped)} "
        f"cache_kb={timing.cache_bytes / 1024:.1f} "
        f"private_mb={timing.private_mb:.1f} ws_mb={timing.working_set_mb:.1f}",
        flush=True,
    )
    return timing


def _selection(config: Config, splits: tuple[str, ...]) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for split in splits:
        if split not in ("train", "val"):
            raise RuntimeError(f"refusing split {split}")
        paths = record_paths(config.signals_dir, split)
        if len(paths) != EXPECTED_RECORDS[split]:
            raise RuntimeError(
                f"{split}: found {len(paths)} files, expected {EXPECTED_RECORDS[split]}"
            )
        selected.extend((split, path) for path in paths)
    return selected


def extract_records(
    config: Config,
    records: list[tuple[str, Path]],
    n_jobs: int,
) -> ExtractReport:
    """Extract features for the given recordings.

    Parameters
    ----------
    config : Config
        Project paths. Cache directories are created here.
    records : list of (str, Path)
        Split name and NPZ path. Only ``train`` and ``val`` are accepted.
    n_jobs : int
        Worker processes. Native libraries inside a worker are limited to
        one thread.

    Returns
    -------
    ExtractReport
        Wall time and per-record timings. Cache hits are not included in
        ``mean_record_s``.
    """
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    for split, _path in records:
        if split not in ("train", "val"):
            raise RuntimeError(f"refusing split {split}")
        (config.cache_dir / CACHE_NAME / split).mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(path), split, str(cache_file(config, split, path.stem)))
        for split, path in records
    ]
    started = time.perf_counter()
    previous = _push_one_thread()
    try:
        with parallel_config(backend="loky", inner_max_num_threads=1, n_jobs=n_jobs):
            timings = Parallel(n_jobs=n_jobs)(
                delayed(extract_one)(npz_path, split, destination)
                for npz_path, split, destination in tasks
            )
    finally:
        _pop_threads(previous)
    rows = tuple(timings)
    computed = [row for row in rows if not row.skipped]
    mean_record_s = (
        float(np.mean([row.seconds for row in computed])) if computed else float("nan")
    )
    return ExtractReport(
        wall_s=time.perf_counter() - started,
        n_jobs=n_jobs,
        computed=len(computed),
        skipped=len(rows) - len(computed),
        mean_record_s=mean_record_s,
        max_private_mb=max((row.private_mb for row in computed), default=float("nan")),
        max_working_set_mb=max(
            (row.working_set_mb for row in computed), default=float("nan")
        ),
        cache_bytes=sum(row.cache_bytes for row in rows),
        epochs=sum(row.n_epochs for row in rows),
        records=rows,
    )


def load_cached_split(cache_dir: Path, split: str) -> FeatureTable:
    """Load one split's cache, joined later by id.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory that contains ``features_v1``.
    split : str
        ``train`` or ``val``.

    Returns
    -------
    FeatureTable
        Rows follow sorted record names and, inside a record, file order.

    Raises
    ------
    RuntimeError
        If the cache is incomplete, stale, or the test split is requested.
    """
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    folder = cache_dir / CACHE_NAME / split
    paths = sorted(
        path for path in folder.glob("*.npz") if path.is_file() and not path.name.startswith(".")
    )
    if len(paths) != EXPECTED_RECORDS[split]:
        raise RuntimeError(
            f"{split}: {len(paths)} cache files, expected {EXPECTED_RECORDS[split]}"
        )
    blocks: list[np.ndarray] = []
    ids: list[str] = []
    names = feature_names()
    for path in paths:
        if not _cache_ready(path):
            raise RuntimeError(f"stale or unreadable cache {path}")
        with np.load(path, allow_pickle=False) as archive:
            blocks.append(np.asarray(archive["features"], dtype=np.float32))
            ids.extend(_as_text_list(archive["ids"]))
    features = np.concatenate(blocks, axis=0)
    id_array = np.asarray(ids, dtype=str)
    if features.shape != (len(id_array), N_FEATURES):
        raise RuntimeError(f"{split} feature shape {features.shape}")
    if len(set(ids)) != len(ids) or not np.isfinite(features).all():
        raise RuntimeError(f"{split} features failed the id or finite check")
    return FeatureTable(features=features, ids=id_array, feature_names=names)


def align_labels(
    ids: np.ndarray,
    labels: Any,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align labels to feature rows by ``id``.

    Parameters
    ----------
    ids : ndarray of str
        Epoch ids in feature-row order.
    labels : pandas.DataFrame
        ``labels.csv`` with ``id``, ``label``, ``split`` and ``subject_id``.
    split : str
        Split to keep from the table.

    Returns
    -------
    y : ndarray of int64
        Labels in ``{0, 1, 2, 3, 4}``, one per feature row.
    subjects : ndarray of str
        ``subject_id`` in the same order.
    """
    id_list = [str(item) for item in ids.tolist()]
    if len(set(id_list)) != len(id_list):
        raise ValueError("duplicate feature ids")
    part = labels.loc[
        labels["split"].astype(str) == split, ["id", "label", "subject_id"]
    ]
    part_ids = part["id"].astype(str)
    if bool(part_ids.duplicated().any()):
        raise ValueError(f"duplicate ids in {split} labels")
    feature_ids = set(id_list)
    table_ids = set(part_ids.tolist())
    missing = len(feature_ids - table_ids)
    extra = len(table_ids - feature_ids)
    if missing or extra:
        raise ValueError(
            f"{split} id mismatch: {missing} features without labels, "
            f"{extra} labels without features"
        )
    label_map = dict(zip(part_ids.tolist(), part["label"].astype(str).tolist(), strict=True))
    subject_map = dict(
        zip(part_ids.tolist(), part["subject_id"].astype(str).tolist(), strict=True)
    )
    y = np.empty(len(id_list), dtype=np.int64)
    subjects = np.empty(len(id_list), dtype=object)
    for index, epoch_id in enumerate(id_list):
        value = int(label_map[epoch_id])
        if value not in (0, 1, 2, 3, 4):
            raise ValueError(f"label {value} for {epoch_id}")
        subject = subject_map[epoch_id].strip()
        if not subject or subject.lower() == "nan":
            raise ValueError(f"empty subject_id for {epoch_id}")
        y[index] = value
        subjects[index] = subject
    return y, subjects.astype(str)


def _assert_smoke_join(config: Config, rows: tuple[RecordTiming, ...]) -> None:
    labels = read_labels(config.tables_dir / "labels.csv")
    for row in rows:
        with np.load(cache_file(config, row.split, row.record_id), allow_pickle=False) as archive:
            ids = set(_as_text_list(archive["ids"]))
        part = labels.loc[
            (labels["split"].astype(str) == row.split)
            & (labels["record_id"].astype(str) == row.record_id),
            "id",
        ].astype(str)
        if set(part.tolist()) != ids:
            raise RuntimeError(f"label join failed for {row.split}/{row.record_id}")


def assess_smoke(report: ExtractReport, total_mb: float, avail_mb: float) -> tuple[bool, str]:
    """Apply the full-extraction time and memory gate.

    The time formula is mean seconds per extracted record times 152 divided
    by the worker count. Memory uses the max private peak of a worker times
    the worker count, compared with 85% of currently available RAM.

    Parameters
    ----------
    report : ExtractReport
        Smoke extraction. Cache hits do not provide a timing.
    total_mb, avail_mb : float
        Physical memory at the moment of the decision.

    Returns
    -------
    proceed : bool
        False when the estimate exceeds 60 minutes or available memory.
    reason : str
        Short explanation.
    """
    if report.computed == 0 or not math.isfinite(report.mean_record_s):
        return False, "no freshly extracted record; cannot time the full run"
    estimate_s = report.mean_record_s * FULL_RECORD_COUNT / report.n_jobs
    peak = report.max_private_mb
    if not math.isfinite(peak) or peak <= 0:
        peak = report.max_working_set_mb
    if not math.isfinite(peak) or peak <= 0:
        return False, "peak memory could not be measured"
    estimated_peak = peak * report.n_jobs
    limit = avail_mb * 0.85
    reasons: list[str] = []
    if estimate_s > TIME_LIMIT_S:
        reasons.append(
            f"estimated full extraction {estimate_s / 60:.1f} min exceeds 60 min"
        )
    if estimated_peak > limit:
        reasons.append(
            f"estimated peak {estimated_peak:.0f} MiB exceeds 85% of available RAM "
            f"({limit:.0f} MiB of {avail_mb:.0f} MiB; total {total_mb:.0f} MiB)"
        )
    if reasons:
        return False, "; ".join(reasons)
    return (
        True,
        f"estimate {estimate_s / 60:.2f} min and peak {estimated_peak:.0f} MiB "
        f"are inside the limits",
    )


def _print_smoke(report: ExtractReport, total_mb: float, avail_mb: float, decision: str, reason: str) -> None:
    estimate_s = report.mean_record_s * FULL_RECORD_COUNT / report.n_jobs
    mean_bytes = report.cache_bytes / max(len(report.records), 1)
    print(
        "\n".join(
            [
                f"n_jobs {report.n_jobs}",
                f"smoke_records {len(report.records)} computed {report.computed} "
                f"skipped {report.skipped}",
                f"smoke_wall_s {report.wall_s:.2f}",
                f"mean_record_s {report.mean_record_s:.3f}",
                f"estimate_152_s {estimate_s:.1f}",
                f"estimate_152_min {estimate_s / 60:.2f}",
                f"estimate_128_min {report.mean_record_s * 128 / report.n_jobs / 60:.2f}",
                f"cache_smoke_mb {report.cache_bytes / (1024 * 1024):.2f}",
                f"cache_152_mb {mean_bytes * FULL_RECORD_COUNT / (1024 * 1024):.2f}",
                f"peak_private_mb {report.max_private_mb:.1f}",
                f"peak_working_set_mb {report.max_working_set_mb:.1f}",
                f"est_private_x_jobs_mb {report.max_private_mb * report.n_jobs:.1f}",
                f"ram_total_mb {total_mb:.1f}",
                f"ram_avail_mb {avail_mb:.1f}",
                f"SMOKE decision: {decision}",
                f"reason: {reason}",
            ]
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the formula self-check and the 3-record smoke extraction.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments. Only ``--smoke`` starts extraction.

    Returns
    -------
    int
        0 when the full run is allowed, 2 when it must stop, 1 on usage.
    """
    parser = argparse.ArgumentParser(description="Smoke-test Welch feature extraction.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Extract 2 train records and 1 val record, then estimate the full run.",
    )
    args = parser.parse_args(argv)
    if not args.smoke:
        print("Run python -m src.features --smoke before python -m src.train.")
        return 1
    config = load_config(_config_path())
    lock = acquire_lock(config)
    try:
        print("formula self-check: running", flush=True)
        _self_check()
        print("formula self-check: ok", flush=True)
        selected = _selection(config, ("train", "val"))
        smoke_records = selected[:2] + [selected[EXPECTED_RECORDS["train"]]]
        if [path.stem for _split, path in smoke_records] != [
            selected[0][1].stem,
            selected[1][1].stem,
            selected[EXPECTED_RECORDS["train"]][1].stem,
        ]:
            raise RuntimeError("smoke record selection is not the first files")
        report = extract_records(config, smoke_records, worker_count())
        _assert_smoke_join(config, report.records)
        total_mb, avail_mb = _memory_status()
        proceed, reason = assess_smoke(report, total_mb, avail_mb)
        _print_smoke(report, total_mb, avail_mb, "PROCEED" if proceed else "STOP", reason)
        return 0 if proceed else 2
    finally:
        release_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())

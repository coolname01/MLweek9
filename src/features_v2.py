"""Extra per-epoch features. Formulas are fixed; nothing is fit across records.

The signal is z-scored with :func:`src.features._zscore_record` (that
recording's own mean and standard deviation) before every formula.
Welch matches v1: Hann, ``nperseg=256``, ``noverlap=128``, density,
constant detrend. Band edges match :func:`src.features._band_mask`.

Columns, in order
-----------------
1. Normalized Shannon entropy of the Welch PSD on ``0.5 <= f <= 50`` Hz,
   one value per channel. ``p = psd / sum(psd)``; ``0 ln 0`` is 0; a silent
   spectrum has entropy 0. Divide by ``ln(n_bins)``.
2. ``delta/beta`` and ``theta/alpha`` on EEG Fpz-Cz and EEG Pz-Oz.
   Denominator is floored at ``1e-20``. Powers are the trapezoidal integrals
   of the v1 bands.
3. Linear sigma power (12-16 Hz, v1 mask) on both EEG channels. This is a
   monotone transform of the v1 log-sigma column and is kept because the
   queue asks for the sigma band on each EEG.
4. ``log10`` EOG power on ``0.5 <= f < 5`` Hz, floor ``1e-20``. v1 delta
   stops at 4 Hz, so this band is new.
5. Population variance of non-overlapping 2 s blocks (200 samples, 15
   blocks) of the z-scored channel. Mean and max across the 15 blocks,
   per channel.

The test split is never opened. Cache files are new: ``cache/features_v2``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.signal import welch

from src.data import Config
from src.expkit import restore_env, thread_env_one
from src.features import (
    BANDS,
    EPOCH_LENGTH,
    FS,
    LOG_POWER_FLOOR,
    NPERSEG,
    NOVERLAP,
    _band_mask,
    _load_signal,
    _zscore_record,
    worker_count,
)

V2_VERSION = "v2"
N_V2 = 16
SUBWINDOW = 200
CACHE_NAME = "features_v2"
V2_NAMES: tuple[str, ...] = (
    "eeg_fpz_cz_spec_entropy",
    "eeg_pz_oz_spec_entropy",
    "eog_horizontal_spec_entropy",
    "eeg_fpz_cz_delta_over_beta",
    "eeg_fpz_cz_theta_over_alpha",
    "eeg_pz_oz_delta_over_beta",
    "eeg_pz_oz_theta_over_alpha",
    "eeg_fpz_cz_sigma_power",
    "eeg_pz_oz_sigma_power",
    "eog_horizontal_logpower_0p5_5",
    "eeg_fpz_cz_subvar_mean",
    "eeg_fpz_cz_subvar_max",
    "eeg_pz_oz_subvar_mean",
    "eeg_pz_oz_subvar_max",
    "eog_horizontal_subvar_mean",
    "eog_horizontal_subvar_max",
)
_BAND_INDEX = {name: index for index, (name, _low, _high) in enumerate(BANDS)}


@dataclass(frozen=True)
class V2Timing:
    """One cache write or cache hit."""

    split: str
    record_id: str
    skipped: bool
    seconds: float
    n_epochs: int


def cache_path(config: Config, split: str, record_id: str) -> Path:
    """Path of one v2 cache file. ``test`` is refused."""
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    return config.cache_dir / CACHE_NAME / split / f"{record_id}.npz"


def _welch(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = welch(
        channel,
        fs=FS,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NPERSEG,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
    )
    return np.asarray(freqs, dtype=np.float64), np.maximum(np.asarray(psd, dtype=np.float64), 0.0)


def _integrate(spectrum: np.ndarray, freqs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) < 2:
        raise RuntimeError("band mask has fewer than 2 bins")
    return np.trapezoid(spectrum[:, mask], freqs[mask], axis=1)


def _entropy(spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    mask = (freqs >= 0.5) & (freqs <= 50.0)
    power = spectrum[:, mask]
    total = power.sum(axis=1, keepdims=True)
    probability = np.zeros_like(power)
    np.divide(power, total, out=probability, where=total > 0)
    log_p = np.zeros_like(probability)
    np.log(probability, out=log_p, where=probability > 0)
    n_bins = int(mask.sum())
    if n_bins < 2:
        raise RuntimeError("entropy grid has fewer than 2 bins")
    entropy = -(probability * log_p).sum(axis=1) / np.log(n_bins)
    silent = total[:, 0] <= 0
    entropy[silent] = 0.0
    return np.maximum(entropy, 0.0)


def _subwindow_stats(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if channel.shape[1] != EPOCH_LENGTH or EPOCH_LENGTH % SUBWINDOW != 0:
        raise RuntimeError("epoch length is not a multiple of 2 seconds")
    blocks = channel.reshape(channel.shape[0], EPOCH_LENGTH // SUBWINDOW, SUBWINDOW)
    variance = np.var(blocks, axis=2)
    return variance.mean(axis=1), variance.max(axis=1)


def features_v2(signal: np.ndarray) -> np.ndarray:
    """Compute the 16 extra columns for one z-scored recording.

    Parameters
    ----------
    signal : ndarray, shape (n_epochs, 3000, 3), float32
        Raw microvolts. This function z-scores the array in place with the
        recording's own mean and standard deviation.

    Returns
    -------
    ndarray, shape (n_epochs, 16), float64
    """
    if signal.ndim != 3 or signal.shape[1:] != (EPOCH_LENGTH, 3):
        raise ValueError(f"expected (N, {EPOCH_LENGTH}, 3), got {signal.shape}")
    _zscore_record(signal)
    freqs_ref: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    for index in range(3):
        channel = np.ascontiguousarray(signal[:, :, index], dtype=np.float64)
        freqs, spectrum = _welch(channel)
        if freqs_ref is None:
            freqs_ref = freqs
        spectra.append(spectrum)
    if freqs_ref is None:
        raise RuntimeError("no channels")
    out = np.empty((signal.shape[0], N_V2), dtype=np.float64)
    for index, spectrum in enumerate(spectra):
        out[:, index] = _entropy(spectrum, freqs_ref)
    eeg_slots = (
        (0, 3),
        (1, 5),
    )
    for channel, column in eeg_slots:
        spectrum = spectra[channel]
        delta = _integrate(spectrum, freqs_ref, _band_mask(freqs_ref, _BAND_INDEX["delta"]))
        theta = _integrate(spectrum, freqs_ref, _band_mask(freqs_ref, _BAND_INDEX["theta"]))
        alpha = _integrate(spectrum, freqs_ref, _band_mask(freqs_ref, _BAND_INDEX["alpha"]))
        beta = _integrate(spectrum, freqs_ref, _band_mask(freqs_ref, _BAND_INDEX["beta"]))
        out[:, column] = delta / np.maximum(beta, LOG_POWER_FLOOR)
        out[:, column + 1] = theta / np.maximum(alpha, LOG_POWER_FLOOR)
    sigma_mask = _band_mask(freqs_ref, _BAND_INDEX["sigma"])
    out[:, 7] = _integrate(spectra[0], freqs_ref, sigma_mask)
    out[:, 8] = _integrate(spectra[1], freqs_ref, sigma_mask)
    eog_mask = (freqs_ref >= 0.5) & (freqs_ref < 5.0)
    eog_power = _integrate(spectra[2], freqs_ref, eog_mask)
    out[:, 9] = np.log10(np.maximum(eog_power, LOG_POWER_FLOOR))
    for index in range(3):
        channel = np.asarray(signal[:, :, index], dtype=np.float64)
        mean_var, max_var = _subwindow_stats(channel)
        out[:, 10 + 2 * index] = mean_var
        out[:, 11 + 2 * index] = max_var
    if not np.isfinite(out).all():
        raise RuntimeError("v2 features are not finite")
    return out


def self_check() -> None:
    """Sines must land in the expected bands; a gap-free shape is 16 columns."""
    time_s = np.arange(EPOCH_LENGTH, dtype=np.float64) / FS
    tone = np.zeros((2, EPOCH_LENGTH, 3), dtype=np.float32)
    tone[:, :, 0] = np.sin(2.0 * np.pi * 14.0 * time_s)
    tone[:, :, 1] = np.sin(2.0 * np.pi * 2.0 * time_s)
    tone[:, :, 2] = np.sin(2.0 * np.pi * 2.0 * time_s)
    features = features_v2(tone)
    if features.shape != (2, N_V2):
        raise RuntimeError(f"v2 shape {features.shape}")
    if not (0.0 <= float(features[0, 0]) <= 1.0):
        raise RuntimeError("entropy outside [0, 1]")
    # 14 Hz on Fpz-Cz is sigma, so sigma power exceeds the 2 Hz Pz-Oz sigma power.
    if float(features[0, 7]) <= float(features[0, 8]):
        raise RuntimeError("14 Hz sigma power did not exceed the 2 Hz channel")
    noise = np.zeros((2, EPOCH_LENGTH, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    noise[:, :, 0] = rng.normal(size=(2, EPOCH_LENGTH))
    noise[:, :, 1] = rng.normal(size=(2, EPOCH_LENGTH))
    noise[:, :, 2] = rng.normal(size=(2, EPOCH_LENGTH))
    broad = features_v2(noise)
    if float(broad[0, 0]) <= float(features[0, 0]):
        raise RuntimeError("broadband noise is not higher entropy than a 14 Hz sine")
    if len(V2_NAMES) != N_V2:
        raise RuntimeError("v2 name count")


def _ready(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"])
            ids = archive["ids"]
            names = tuple(str(item) for item in np.asarray(archive["feature_names"]).tolist())
            version = str(np.asarray(archive["version"]).reshape(-1)[0])
            ready = (
                version == V2_VERSION
                and names == V2_NAMES
                and features.dtype == np.float32
                and features.ndim == 2
                and features.shape[1] == N_V2
                and features.shape[0] == len(ids)
                and bool(np.isfinite(features).all())
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return ready


def _write(path: Path, features: np.ndarray, ids: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.npz")
    payload = {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "ids": np.asarray(ids),
        "feature_names": np.asarray(V2_NAMES),
        "version": np.asarray(V2_VERSION),
    }
    try:
        np.savez(temporary, **payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not _ready(path):
        raise RuntimeError(f"wrote an unreadable v2 cache {path}")


def extract_one(npz_path: str, split: str, destination: str) -> V2Timing:
    """Extract or reuse v2 features for one train or val recording."""
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    source = Path(npz_path)
    if "test" in {part.lower() for part in source.parts}:
        raise RuntimeError(f"refusing to read a test record: {source}")
    started = time.perf_counter()
    dest = Path(destination)
    record_id = source.stem
    if _ready(dest):
        with np.load(dest, allow_pickle=False) as archive:
            n_epochs = int(archive["features"].shape[0])
        skipped = True
    else:
        signal, ids = _load_signal(source)
        features = features_v2(signal)
        del signal
        if features.shape[0] != len(ids):
            raise RuntimeError(f"{record_id} v2 length mismatch")
        _write(dest, features, ids)
        n_epochs = len(ids)
        skipped = False
    return V2Timing(
        split=split,
        record_id=record_id,
        skipped=skipped,
        seconds=time.perf_counter() - started,
        n_epochs=n_epochs,
    )


def extract_records(
    config: Config,
    split: str,
    record_ids: list[str],
    n_jobs: int,
) -> tuple[V2Timing, ...]:
    """Extract v2 features. Only ``train`` and ``val`` paths are built."""
    if split not in ("train", "val"):
        raise RuntimeError(f"refusing split {split}")
    tasks = []
    for record_id in record_ids:
        source = config.signals_dir / split / f"{record_id}.npz"
        if "test" in {part.lower() for part in source.parts}:
            raise RuntimeError(f"refusing to read a test record: {source}")
        tasks.append((str(source), split, str(cache_path(config, split, record_id))))
    previous = thread_env_one()
    try:
        with parallel_config(backend="loky", inner_max_num_threads=1, n_jobs=n_jobs):
            timings = Parallel(n_jobs=n_jobs)(
                delayed(extract_one)(source, split_name, dest)
                for source, split_name, dest in tasks
            )
    finally:
        restore_env(previous)
    return tuple(timings)


def load_v2(config: Config, split: str, record_id: str, ids: np.ndarray) -> np.ndarray:
    """Load one v2 matrix aligned to ``ids``."""
    path = cache_path(config, split, record_id)
    if not _ready(path):
        raise RuntimeError(f"v2 cache missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        features = np.array(archive["features"], dtype=np.float32, copy=True)
        cached_ids = np.asarray(archive["ids"], dtype=str)
    if cached_ids.shape != ids.shape or not np.array_equal(cached_ids, ids):
        raise RuntimeError(f"{split}/{record_id} v2 ids differ from the v1 cache")
    return features


def load_v2_map(
    config: Config,
    records: list,
) -> dict[str, np.ndarray]:
    """Map ``record_id`` to a v2 matrix in v1 row order."""
    loaded: dict[str, np.ndarray] = {}
    for record in records:
        loaded[record.record_id] = load_v2(config, record.split, record.record_id, record.ids)
    return loaded


def default_jobs() -> int:
    """Worker count used by the v1 extractor."""
    return worker_count()

"""Cross-view offset from the audio track, by generalized cross-correlation.

Every canonical asset carries one mono AAC track, so this route covers the
whole corpus rather than a subset.  Two cameras started by hand record the same
room, and the phase transform reduces a shared acoustic event to a single
correlation peak whose lag is the offset between the two timelines.

Sign convention, published because a reversed offset is a silent defect: the
returned lag is ``t_B - t_A``, the local time of one shared event in B's
timeline minus its local time in A's.  Positive means B started recording
earlier, so the event sits further into B's clip.

Nothing here shares a line, a signal or a threshold with ``visual_offset``.
That independence is the whole instrument: closure around a three-camera
triangle is blind to acoustic path delay by construction, because the
propagation terms form an exact cocycle, so agreement between two estimators is
the only accuracy statistic this corpus yields.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
from fractions import Fraction

import av
import numpy as np
from av.audio.resampler import AudioResampler
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.signal import resample_poly
from scipy.stats import linregress

TARGET_RATE = 16_000
COARSE_RATE = 4_000
MIN_OVERLAP_S = 3.0
PEAK_GUARD_S = 0.25
MIN_PEAK_RMS = 8.0
MIN_PEAK_RATIO = 2.1
EDGE_FADE_S = 0.05
BAND_LOW_HZ = 80.0
BAND_HIGH_FRACTION = 0.475

DRIFT_WINDOW_S = 3.0
DRIFT_MIN_WINDOW_S = 1.5
DRIFT_SEARCH_RADIUS_S = 0.1
DRIFT_MAX_WINDOWS = 24
DRIFT_MIN_WINDOWS = 3
LOCAL_PEAK_GUARD_S = 0.005
LOCAL_MIN_PEAK_RMS = 5.0
LOCAL_MIN_PEAK_RATIO = 1.2

# Every constant above reaches the sidecar manifest.  A threshold that moves the
# numbers and is not recorded makes the published figures unattributable to the
# code that produced them, and none of these doubles as an instrument parameter:
# the accept thresholds scale a confidence that the correlation computes without
# them, so no statistic here is judged against a constant that also shapes it.
PROVENANCE: dict[str, float | int | str] = {
    "estimator": "gcc_phat",
    "target_rate_hz": TARGET_RATE,
    "coarse_rate_hz": COARSE_RATE,
    "min_overlap_s": MIN_OVERLAP_S,
    "peak_guard_s": PEAK_GUARD_S,
    "min_peak_rms": MIN_PEAK_RMS,
    "min_peak_ratio": MIN_PEAK_RATIO,
    "edge_fade_s": EDGE_FADE_S,
    "band_low_hz": BAND_LOW_HZ,
    "band_high_fraction": BAND_HIGH_FRACTION,
    "drift_window_s": DRIFT_WINDOW_S,
    "drift_min_window_s": DRIFT_MIN_WINDOW_S,
    "drift_search_radius_s": DRIFT_SEARCH_RADIUS_S,
    "drift_max_windows": DRIFT_MAX_WINDOWS,
    "drift_min_windows": DRIFT_MIN_WINDOWS,
    "local_peak_guard_s": LOCAL_PEAK_GUARD_S,
    "local_min_peak_rms": LOCAL_MIN_PEAK_RMS,
    "local_min_peak_ratio": LOCAL_MIN_PEAK_RATIO,
    "sign_convention": "t_b_minus_t_a",
}


@dataclasses.dataclass(frozen=True)
class Peak:
    """One correlation peak, with the statistics that decide whether to keep it."""

    lag_s: float
    confidence: float
    peak_rms: float
    peak_ratio: float
    overlap_s: float
    status: str


@dataclasses.dataclass(frozen=True)
class Drift:
    """A linear rate difference fitted across the pair's overlap."""

    ppm: float
    standard_error: float
    window_count: int
    span_s: float
    rmse_s: float
    status: str


def _seconds(value: int | None, time_base: Fraction | None) -> float:
    if value is None or time_base is None:
        return math.nan
    return float(value * time_base)


def _skip_samples(packet: av.packet.Packet) -> int:
    """Return the encoder priming the container asks the decoder to discard.

    AAC priming is rate-dependent — 2112 samples is 47.9 ms at 44.1 kHz and
    44.0 ms at 48 kHz — so an untrimmed mixed-rate pair would carry a fixed
    3.9 ms bias, and 55 of this corpus's multi-view families mix the two rates.
    PyAV honours the edit lists, so the bias never reaches the estimator; this
    value is read so the cancellation stays checkable rather than assumed.
    """
    for side_data in packet.iter_sidedata():
        if getattr(side_data, "data_type", None) == "skip_samples":
            raw = bytes(side_data)
            return int.from_bytes(raw[:4], "little") if len(raw) >= 4 else 0
    return 0


def decode_audio(path: pathlib.Path, target_rate: int = TARGET_RATE) -> tuple[np.ndarray, dict]:
    """Decode mono float PCM onto a PTS-derived timeline whose sample 0 is media t=0.

    Frames are placed by their own presentation time rather than concatenated.
    A gap is zero-filled and an overlap is trimmed, so a dropped packet shifts
    nothing downstream: concatenation would silently slide every later sample
    forward by the length of the hole and corrupt the very quantity being
    measured.
    """
    chunks: list[np.ndarray] = []
    cursor = 0
    inserted = 0
    removed = 0
    first_packet_s = math.nan
    first_decoded_s = math.nan
    skip_samples = 0

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        source_rate = int(stream.rate)
        resampler = AudioResampler(format="fltp", layout="mono", rate=target_rate)

        def append(frame: av.AudioFrame) -> None:
            nonlocal cursor, inserted, removed
            start_s = _seconds(frame.pts, frame.time_base)
            start = cursor if math.isnan(start_s) else round(start_s * target_rate)
            values = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
            if start < 0:
                trim = min(-start, values.size)
                values = values[trim:]
                start += trim
                removed += trim
            if start > cursor:
                chunks.append(np.zeros(start - cursor, dtype=np.float32))
                inserted += start - cursor
                cursor = start
            elif start < cursor:
                trim = min(cursor - start, values.size)
                values = values[trim:]
                removed += trim
            if values.size:
                chunks.append(values)
                cursor += values.size

        for packet in container.demux(stream):
            if math.isnan(first_packet_s) and packet.pts is not None:
                first_packet_s = _seconds(packet.pts, packet.time_base)
                skip_samples = _skip_samples(packet)
            for decoded in packet.decode():
                if math.isnan(first_decoded_s):
                    first_decoded_s = _seconds(decoded.pts, decoded.time_base)
                for frame in resampler.resample(decoded):
                    append(frame)
        for frame in resampler.resample(None):
            append(frame)

    samples = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    timing = {
        "source_rate": source_rate,
        "first_packet_s": None if math.isnan(first_packet_s) else first_packet_s,
        "first_decoded_s": None if math.isnan(first_decoded_s) else first_decoded_s,
        "skip_samples": skip_samples,
        "duration_s": samples.size / target_rate,
        "inserted_gap_samples": inserted,
        "removed_overlap_samples": removed,
    }
    return samples, timing


def _condition(samples: np.ndarray, rate: int) -> np.ndarray:
    """Remove the mean and taper both edges before correlating.

    A hard edge is a broadband impulse, and the phase transform whitens it into
    a correlation peak that competes with the acoustic one.
    """
    values = np.asarray(samples, dtype=np.float32).copy()
    if values.size == 0:
        return values
    values -= np.mean(values, dtype=np.float64)
    fade = min(round(EDGE_FADE_S * rate), values.size // 2)
    if fade:
        ramp = np.sin(np.linspace(0.0, np.pi / 2, fade, endpoint=False, dtype=np.float32)) ** 2
        values[:fade] *= ramp
        values[-fade:] *= ramp[::-1]
    return values


def _overlap_samples(lags: np.ndarray, size_a: int, size_b: int) -> np.ndarray:
    starts = np.maximum(0, -lags)
    stops = np.minimum(size_a, size_b - lags)
    return np.maximum(0, stops - starts)


def gcc_phat_peak(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    rate: int,
    *,
    min_overlap_s: float = MIN_OVERLAP_S,
    lag_center_s: float | None = None,
    lag_radius_s: float | None = None,
    peak_guard_s: float = PEAK_GUARD_S,
    min_peak_rms: float = MIN_PEAK_RMS,
    min_peak_ratio: float = MIN_PEAK_RATIO,
) -> Peak:
    """Return ``t_B - t_A`` over the full feasible lag range, using no metadata.

    The search spans every lag the two clips can physically share.  Seeding it
    from a wall-clock prior would make the estimate a refinement of a timestamp
    this corpus has already shown to be whole-second and multi-second biased,
    and the independence from metadata is what lets the result check it.
    """
    a = _condition(samples_a, rate)
    b = _condition(samples_b, rate)
    minimum = max(2, round(min_overlap_s * rate))
    if min(a.size, b.size) < minimum:
        return Peak(math.nan, math.nan, math.nan, math.nan, 0.0, "short_audio")
    if float(np.sqrt(np.mean(a * a))) < 1e-5 or float(np.sqrt(np.mean(b * b))) < 1e-5:
        return Peak(math.nan, math.nan, math.nan, math.nan, 0.0, "silent")

    fft_size = next_fast_len(a.size + b.size - 1)
    cross = rfft(b, fft_size) * np.conj(rfft(a, fft_size))
    magnitude = np.abs(cross)
    floor = np.finfo(magnitude.dtype).eps * max(1.0, float(np.max(magnitude)))
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > floor)
    frequencies = rfftfreq(fft_size, 1 / rate)
    cross[(frequencies < BAND_LOW_HZ) | (frequencies > rate * BAND_HIGH_FRACTION)] = 0
    circular = irfft(cross, fft_size)

    negative = a.size - 1
    correlation = np.concatenate((circular[fft_size - negative :], circular[: b.size]))
    lags = np.arange(-negative, b.size, dtype=np.int64)
    overlaps = _overlap_samples(lags, a.size, b.size)
    valid = overlaps >= minimum
    if lag_center_s is not None and lag_radius_s is not None:
        valid &= np.abs(lags / rate - lag_center_s) <= lag_radius_s
    if not np.any(valid):
        return Peak(math.nan, math.nan, math.nan, math.nan, 0.0, "no_feasible_lag")

    absolute = np.abs(correlation)
    index = int(np.argmax(np.where(valid, absolute, -np.inf)))
    lag = float(lags[index])
    peak = float(absolute[index])
    if 0 < index < absolute.size - 1 and valid[index - 1] and valid[index + 1]:
        left, middle, right = absolute[index - 1 : index + 2]
        curvature = float(left - 2 * middle + right)
        if curvature < 0:
            lag += float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))

    guard = max(1, round(peak_guard_s * rate))
    background = absolute[valid & (np.abs(lags - lags[index]) > guard)]
    if background.size == 0:
        return Peak(math.nan, math.nan, math.nan, math.nan, 0.0, "no_background")
    rms = float(np.sqrt(np.mean(background * background, dtype=np.float64)))
    second = float(np.max(background))
    peak_rms = peak / rms if rms > 0 else math.inf
    peak_ratio = peak / second if second > 0 else math.inf
    confidence = min(peak_rms / min_peak_rms, peak_ratio / min_peak_ratio)
    at_boundary = not valid[index - 1] if index > 0 else True
    at_boundary |= not valid[index + 1] if index + 1 < valid.size else True
    if at_boundary:
        status = "boundary_peak"
    elif confidence < 1.0:
        status = "low_confidence"
    else:
        status = "ok"
    return Peak(lag / rate, confidence, peak_rms, peak_ratio, float(overlaps[index] / rate), status)


def _empty_drift(status: str, window_count: int = 0) -> Drift:
    return Drift(math.nan, math.nan, window_count, math.nan, math.nan, status)


def estimate_drift(
    samples_a: np.ndarray, samples_b: np.ndarray, offset_s: float, rate: int = TARGET_RATE
) -> Drift:
    """Fit a rate difference by re-estimating the offset in sliding windows.

    A per-window local offset regressed against window centre gives a slope in
    parts per million.  Measured across this corpus, no qualified drift moves
    alignment by more than one frame over its own overlap, which is what lets
    the published schema carry a single constant offset per pair.
    """
    duration_a = samples_a.size / rate
    duration_b = samples_b.size / rate
    overlap_start = max(0.0, -offset_s)
    overlap_stop = min(duration_a, duration_b - offset_s)
    overlap_s = overlap_stop - overlap_start
    window_s = min(DRIFT_WINDOW_S, overlap_s / DRIFT_MIN_WINDOWS)
    if window_s < DRIFT_MIN_WINDOW_S:
        return _empty_drift("short_overlap")

    requested = min(DRIFT_MAX_WINDOWS, max(DRIFT_MIN_WINDOWS, int(overlap_s // window_s)))
    centers = np.linspace(overlap_start + window_s / 2, overlap_stop - window_s / 2, requested)
    window_samples = round(window_s * rate)
    local_centers: list[float] = []
    local_offsets: list[float] = []
    for center in centers:
        start_a = round((center - window_s / 2) * rate)
        start_b = round((center + offset_s - window_s / 2) * rate)
        if (
            start_a < 0
            or start_b < 0
            or start_a + window_samples > samples_a.size
            or start_b + window_samples > samples_b.size
        ):
            continue
        local = gcc_phat_peak(
            samples_a[start_a : start_a + window_samples],
            samples_b[start_b : start_b + window_samples],
            rate,
            min_overlap_s=window_s - 2 * DRIFT_SEARCH_RADIUS_S,
            lag_center_s=0.0,
            lag_radius_s=DRIFT_SEARCH_RADIUS_S,
            peak_guard_s=LOCAL_PEAK_GUARD_S,
            min_peak_rms=LOCAL_MIN_PEAK_RMS,
            min_peak_ratio=LOCAL_MIN_PEAK_RATIO,
        )
        if local.status != "ok":
            continue
        local_centers.append((start_a + window_samples / 2) / rate)
        local_offsets.append((start_b - start_a) / rate + local.lag_s)
    if len(local_offsets) < DRIFT_MIN_WINDOWS:
        return _empty_drift("insufficient_windows", len(local_offsets))

    x = np.asarray(local_centers, dtype=np.float64)
    y = np.asarray(local_offsets, dtype=np.float64)
    fit = linregress(x, y)
    residual = y - (fit.intercept + fit.slope * x)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    # One window that locked onto a different acoustic event drags a
    # three-point regression arbitrarily; the MAD limit removes it and the
    # refit is skipped when it would leave too few windows to fit.
    keep = np.abs(residual - median) <= max(2 / rate, 4 * 1.4826 * mad)
    if DRIFT_MIN_WINDOWS <= int(np.sum(keep)) < y.size:
        x, y = x[keep], y[keep]
        fit = linregress(x, y)
        residual = y - (fit.intercept + fit.slope * x)
    span_s = float(np.ptp(x))
    if span_s <= 0 or fit.stderr is None or not math.isfinite(fit.stderr):
        return _empty_drift("degenerate_regression", y.size)
    return Drift(
        ppm=float(fit.slope * 1_000_000),
        standard_error=float(fit.stderr * 1_000_000),
        window_count=int(y.size),
        span_s=span_s,
        rmse_s=float(np.sqrt(np.mean(residual * residual))),
        status="ok",
    )


def cache_paths(cache_dir: pathlib.Path, asset_id: str) -> tuple[pathlib.Path, ...]:
    return (
        cache_dir / f"{asset_id}.16k.npy",
        cache_dir / f"{asset_id}.4k.npy",
        cache_dir / f"{asset_id}.json",
    )


def _write_array(path: pathlib.Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def ensure_cached(path: pathlib.Path, cache_dir: pathlib.Path, asset_id: str) -> None:
    """Decode one asset's audio into the cache, at both analysis rates.

    The coarse rate carries the global search, which spans every feasible lag
    and would otherwise cost a full-rate correlation over the whole clip; the
    full rate carries the drift windows, which are short and need the
    resolution.  Both are cached because a rerun of the pair grain must not
    re-decode.
    """
    full, coarse, timing = cache_paths(cache_dir, asset_id)
    if full.exists() and coarse.exists() and timing.exists():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples, facts = decode_audio(path)
    _write_array(full, samples)
    _write_array(
        coarse, resample_poly(samples, COARSE_RATE, TARGET_RATE).astype(np.float32, copy=False)
    )
    temporary = timing.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(facts, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(timing)


def estimate(
    cache_dir: pathlib.Path, asset_a: str, asset_b: str
) -> tuple[Peak, Drift, float, float]:
    """Return the peak, the drift fit and both durations for one cached pair."""
    full_a, coarse_a, _ = cache_paths(cache_dir, asset_a)
    full_b, coarse_b, _ = cache_paths(cache_dir, asset_b)
    left = np.load(coarse_a, mmap_mode="r", allow_pickle=False)
    right = np.load(coarse_b, mmap_mode="r", allow_pickle=False)
    peak = gcc_phat_peak(left, right, COARSE_RATE)
    drift = _empty_drift("global_abstention")
    if peak.status == "ok":
        drift = estimate_drift(
            np.load(full_a, mmap_mode="r", allow_pickle=False),
            np.load(full_b, mmap_mode="r", allow_pickle=False),
            peak.lag_s,
        )
    return peak, drift, left.size / COARSE_RATE, right.size / COARSE_RATE


def source_rate(cache_dir: pathlib.Path, asset_id: str) -> int | None:
    """Return the asset's native audio sample rate from its cached timing."""
    _, _, timing = cache_paths(cache_dir, asset_id)
    try:
        return int(json.loads(timing.read_text(encoding="utf-8"))["source_rate"])
    except (OSError, ValueError, KeyError):
        return None

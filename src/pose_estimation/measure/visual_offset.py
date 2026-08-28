"""Cross-view offset from frame-to-frame motion energy, by normalized correlation.

The corroborator for ``audio_offset``, and deliberately disjoint from it: a
different physical signal, a different similarity measure, a different accept
rule, and no shared line of code.  Two cameras watching one room see the same
movements begin and end, so the scalar motion-energy trace of each view carries
the same coarse event structure at the same times.

Its value is precisely that it can disagree.  Three-camera closure is blind to
acoustic path delay because the propagation terms cancel exactly around a
triangle, so audio self-consistency cannot bound audio accuracy.  A second
estimator carrying none of those delays can.

It is a corroborator and never a fallback.  Its held-out control result is
0/200, yet among the pairs it accepts and audio rejects one disagrees by 87
seconds — a clean control rate bounds the false-accept rate against unrelated
clips and says nothing about gross error on related ones.  Agreement is the
statistic; either estimator alone is not.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib

import av
import cv2
import numpy as np
from scipy import ndimage, signal

SIGNAL_VERSION = 1
# The centre band, never the whole frame.  A hand-held camera writes its own
# motion into the border, so the whole-frame trace mixes camera movement with
# subject movement and the two views stop sharing a signal: measured over the
# corpus, whole-frame acceptance is 43 of 246 pairs against the centre
# trace's 74.  This field is the estimator, not a tuning knob.
SIGNAL_FIELD = "center_motion"
DISPLAY_WIDTH = 160
DISPLAY_HEIGHT = 90
BORDER_FRACTION = 0.15
GRID_HZ = 60.0
SMOOTH_S = 0.03
PEAK_EXCLUSION_S = 1.0
MIN_OVERLAP_S = 4.0
MIN_PEAK_CORRELATION = 0.72
MIN_CONFIDENCE = 4.0
MIN_PEAK_RATIO = 1.10
EDGE_GUARD_S = 0.25

# The accept gate is control-optimal: at these three thresholds the estimator
# takes 74 of 246 within-family pairs and 0 of 200 held-out cross-family
# controls.  No threshold here shapes the statistic it judges — the peak
# correlation, its prominence and its ratio are computed before any of them
# apply.
PROVENANCE: dict[str, float | int | str] = {
    "estimator": "motion_energy_normalized_correlation",
    "signal_version": SIGNAL_VERSION,
    "signal_field": SIGNAL_FIELD,
    "display_width": DISPLAY_WIDTH,
    "display_height": DISPLAY_HEIGHT,
    "border_fraction": BORDER_FRACTION,
    "grid_hz": GRID_HZ,
    "smooth_s": SMOOTH_S,
    "peak_exclusion_s": PEAK_EXCLUSION_S,
    "min_overlap_s": MIN_OVERLAP_S,
    "min_peak_correlation": MIN_PEAK_CORRELATION,
    "min_confidence": MIN_CONFIDENCE,
    "min_peak_ratio": MIN_PEAK_RATIO,
    "edge_guard_s": EDGE_GUARD_S,
    "sign_convention": "t_b_minus_t_a",
}


def _border_mask() -> np.ndarray:
    y = max(1, round(DISPLAY_HEIGHT * BORDER_FRACTION))
    x = max(1, round(DISPLAY_WIDTH * BORDER_FRACTION))
    mask = np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=bool)
    mask[y:-y, x:-x] = False
    return mask


BORDER_MASK = _border_mask()
CENTER_MASK = ~BORDER_MASK


@dataclasses.dataclass(frozen=True)
class Offset:
    """One motion-energy correlation peak and the statistics that judge it."""

    offset_s: float
    confidence: float
    peak_ratio: float
    peak_correlation: float
    overlap_s: float
    status: str


@dataclasses.dataclass(frozen=True)
class Signal:
    """A motion trace resampled onto a uniform grid, with its support mask."""

    values: np.ndarray
    valid: np.ndarray
    duration_s: float


def _display_gray(frame: av.VideoFrame, rotation_deg: int) -> np.ndarray:
    """Return one blurred, upright, thumbnail-sized luma frame.

    Rotation is applied here rather than left to the decoder because two views
    of one family are compared, and a sideways trace correlates against an
    upright one only by accident.  The blur suppresses sensor noise, which at
    this scale would otherwise dominate the frame difference.
    """
    if rotation_deg in {90, 270}:
        gray = frame.reformat(
            width=DISPLAY_HEIGHT, height=DISPLAY_WIDTH, format="gray"
        ).to_ndarray()
        gray = np.rot90(gray, -1 if rotation_deg == 90 else 1)
    else:
        gray = frame.reformat(
            width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, format="gray"
        ).to_ndarray()
        if rotation_deg == 180:
            gray = np.rot90(gray, 2)
    if gray.shape != (DISPLAY_HEIGHT, DISPLAY_WIDTH):
        raise ValueError(f"unexpected display-frame shape: {gray.shape}")
    return cv2.GaussianBlur(np.ascontiguousarray(gray), (3, 3), 0)


def _cache_valid(path: pathlib.Path, content_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cached:
            return bool(
                int(cached["version"]) == SIGNAL_VERSION
                and str(cached["content_sha256"]) == content_sha256
                and len(cached["time_s"]) == len(cached["motion"])
                and len(cached["time_s"]) > 1
            )
    except (KeyError, OSError, ValueError):
        return False


def ensure_cached(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    asset_id: str,
    content_sha256: str,
    rotation_deg: int,
) -> str:
    """Decode one asset into a cached motion trace, keyed to its content digest.

    The digest is the cache key, not the path or the modification time: this
    trace costs a full decode, and a stale trace silently answers for a file it
    never saw.  Border and centre traces are stored beside the whole-frame one
    because a camera that moves shows motion at the quiet edges of the scene,
    which is a rigidity signal rather than a subject signal.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{asset_id}.npz"
    if _cache_valid(cache_path, content_sha256):
        return "cached"

    times: list[float] = []
    motion: list[float] = []
    border: list[float] = []
    center: list[float] = []
    previous: np.ndarray | None = None
    previous_t = -math.inf

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("a video frame carries no presentation time")
            frame_t = float(frame.pts * frame.time_base)
            if frame_t <= previous_t:
                raise ValueError("decoded presentation times are not strictly increasing")
            gray = _display_gray(frame, rotation_deg)
            if previous is not None:
                delta = gray.astype(np.float32) - previous
                # The median removes a global exposure step, which every pixel
                # shares and which is not motion.
                magnitude = np.abs(delta - np.median(delta))
                times.append(frame_t)
                motion.append(float(np.mean(magnitude)))
                border.append(float(np.mean(magnitude[BORDER_MASK])))
                center.append(float(np.mean(magnitude[CENTER_MASK])))
            previous = gray
            previous_t = frame_t

    if len(times) < 2:
        raise ValueError("the asset yields fewer than two frame differences")
    temporary = cache_dir / f".{asset_id}.{os.getpid()}.npz"
    np.savez_compressed(
        temporary,
        version=np.int64(SIGNAL_VERSION),
        content_sha256=np.asarray(content_sha256),
        time_s=np.asarray(times, dtype=np.float64),
        motion=np.asarray(motion, dtype=np.float32),
        border_motion=np.asarray(border, dtype=np.float32),
        center_motion=np.asarray(center, dtype=np.float32),
        media_duration_s=np.float64(previous_t),
        rotation_deg=np.int64(rotation_deg),
    )
    temporary.replace(cache_path)
    return "decoded"


def load_signal(
    cache_dir: pathlib.Path, asset_id: str, field: str = SIGNAL_FIELD
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_dir / f"{asset_id}.npz", allow_pickle=False) as cached:
        return cached["time_s"].astype(np.float64), cached[field].astype(np.float64)


def resample(time_s: np.ndarray, values: np.ndarray, *, grid_hz: float = GRID_HZ) -> Signal:
    """Put one trace on a uniform grid, log-compressed and standardized.

    The grid exists because the two clips do not share a frame rate — every
    file in this corpus differs — so a lag in samples is not a lag in seconds
    unless both sides are resampled first.  ``log1p`` compresses the heavy tail
    that one large movement would otherwise impose on the correlation, and the
    validity mask keeps the normalization honest where the grid runs past the
    trace.
    """
    if len(time_s) != len(values) or len(time_s) < 3:
        raise ValueError("a signal needs at least three samples of equal length")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("signal times must be strictly increasing")
    grid = np.arange(math.floor(float(time_s[-1]) * grid_hz) + 1, dtype=np.float64) / grid_hz
    interpolated = np.interp(
        grid, time_s, np.log1p(np.maximum(values, 0)), left=np.nan, right=np.nan
    )
    valid = np.isfinite(interpolated)
    if np.count_nonzero(valid) < round(MIN_OVERLAP_S * grid_hz):
        raise ValueError("a signal carries less than the minimum support")
    smoothed = interpolated.copy()
    smoothed[valid] = ndimage.gaussian_filter1d(
        smoothed[valid], sigma=SMOOTH_S * grid_hz, mode="nearest"
    )
    mean = float(np.mean(smoothed[valid]))
    std = float(np.std(smoothed[valid]))
    if not math.isfinite(std) or std <= np.finfo(np.float64).eps:
        raise ValueError("a signal carries no variance")
    smoothed[valid] = (smoothed[valid] - mean) / std
    smoothed[~valid] = 0
    return Signal(smoothed, valid, float(time_s[-1]))


def _parabolic_delta(y_minus: float, y_zero: float, y_plus: float) -> float:
    denominator = y_minus - 2 * y_zero + y_plus
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y_minus - y_plus) / denominator, -0.5, 0.5))


def estimate(
    time_a: np.ndarray,
    values_a: np.ndarray,
    time_b: np.ndarray,
    values_b: np.ndarray,
    *,
    grid_hz: float = GRID_HZ,
) -> Offset:
    """Return ``t_B - t_A`` from the normalized cross-correlation peak.

    The correlation is normalized per lag by the energy actually overlapping at
    that lag, so a lag where the clips barely touch cannot win on sample count
    alone.  Every lag whose overlap falls under half the shorter clip is
    ineligible, which is what stops the estimator from reporting a confident
    alignment built on a second of shared trace.
    """
    a = resample(time_a, values_a, grid_hz=grid_hz)
    b = resample(time_b, values_b, grid_hz=grid_hz)
    required = math.ceil(max(MIN_OVERLAP_S, 0.5 * min(a.duration_s, b.duration_s)) * grid_hz)

    valid_a = a.valid.astype(np.float64)
    valid_b = b.valid.astype(np.float64)
    correlation = signal.correlate(b.values, a.values, mode="full", method="fft")
    overlap = signal.correlate(valid_b, valid_a, mode="full", method="fft")
    denominator = np.sqrt(
        np.maximum(signal.correlate(valid_b, a.values**2, mode="full", method="fft"), 0)
        * np.maximum(signal.correlate(b.values**2, valid_a, mode="full", method="fft"), 0)
    )
    overlap = np.rint(np.maximum(overlap, 0)).astype(np.int64)
    lags = signal.correlation_lags(len(b.values), len(a.values), mode="full")
    eligible = (overlap >= required) & (denominator > np.finfo(np.float64).eps)
    candidates = np.flatnonzero(eligible)
    if len(candidates) < 3:
        return Offset(math.nan, math.nan, math.nan, math.nan, math.nan, "insufficient_overlap")

    scores = np.full_like(correlation, np.nan, dtype=np.float64)
    scores[eligible] = correlation[eligible] / denominator[eligible]
    peak_index = int(candidates[np.nanargmax(scores[candidates])])
    peak_score = float(scores[peak_index])
    delta = 0.0
    if peak_index > 0 and peak_index + 1 < len(scores):
        neighbors = scores[peak_index - 1 : peak_index + 2]
        if np.all(np.isfinite(neighbors)):
            delta = _parabolic_delta(*neighbors)
    offset_s = (float(lags[peak_index]) + delta) / grid_hz

    exclusion = max(1, round(PEAK_EXCLUSION_S * grid_hz))
    background_mask = eligible.copy()
    background_mask[max(0, peak_index - exclusion) : peak_index + exclusion + 1] = False
    background = scores[background_mask]
    background = background[np.isfinite(background)]
    if len(background) < 10:
        confidence = peak_ratio = math.nan
    else:
        median = float(np.median(background))
        mad = float(np.median(np.abs(background - median)))
        confidence = (peak_score - median) / max(1.4826 * mad, np.finfo(np.float64).eps)
        peak_ratio = (peak_score - median) / max(
            float(np.max(background)) - median, np.finfo(np.float64).eps
        )

    search_min_s = float(lags[candidates[0]]) / grid_hz
    search_max_s = float(lags[candidates[-1]]) / grid_hz
    if min(offset_s - search_min_s, search_max_s - offset_s) < EDGE_GUARD_S:
        status = "edge_peak"
    elif not math.isfinite(confidence) or not math.isfinite(peak_ratio):
        status = "undefined_confidence"
    elif peak_score < MIN_PEAK_CORRELATION:
        status = "low_peak_correlation"
    elif confidence < MIN_CONFIDENCE:
        status = "low_prominence"
    elif peak_ratio < MIN_PEAK_RATIO:
        status = "ambiguous_peak"
    else:
        status = "ok"
    return Offset(
        offset_s,
        confidence,
        peak_ratio,
        peak_score,
        float(overlap[peak_index]) / grid_hz,
        status,
    )

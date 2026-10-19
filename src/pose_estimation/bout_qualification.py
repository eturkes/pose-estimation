"""M4.5a D04 + D06: synthetic wrist-motion banks with known bouts, matching, scoring, ranking.

Contract `.agent/archive/contract-m4u5a.md`. The detector and every feature run in the shipped
R producer (`analysis/clinical_features.R`, driven by `analysis/bout_qualification.R`); this
module owns the ground truth and the grading, so the cell a qualification freezes is decided by
shipped code end to end. Units = frame-normalised coordinates, seconds.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

RATES = (25.0, 30.0, 60.0)
NOISE_SCALES = (0.5, 1.0, 2.0)
KINDS = ("standard", "close", "unequal", "busy", "stationary")
#: D04 grid, enumerated in this order; ties in the ranking fall to the earlier cell.
GRID = {
    "threshold": (0.05, 0.1, 0.15, 0.2),
    "normaliser_q": (0.95, 0.99, 1.0),
    "min_sec": (0.2, 0.3, 0.4),
    "merge_sec": (0.1, 0.2, 0.3),
    "noise_k": (0.0, 4.0, 8.0),
    "bounds": ("level", "minimum"),
}
SUPPORTS = ("bounds", "extended")
PROMINENCES = (0.1, 0.2, 0.3)
PEAK_SPACING_SEC = 0.15
#: A true episode is in the supported domain when it stands this far from its neighbours, the
#: series edges and every artifact, and its peak speed is at least this share of the series' top.
SUPPORT_GAP_SEC = 0.3
SUPPORT_PEAK_SHARE = 0.3
SERIES_SEC = 16.0
PEAK_SPEED = (0.05, 0.5)  # frame-normalised units / s
#: Truth bouts = rest-to-rest runs of noise-free speed, bounded where speed crosses this share of
#: the run's own peak.
TRUTH_ONSET_SHARE = 0.05
#: Seven-slot quadratic Savitzky-Golay smoothing weights (D06).
SG7 = np.array([-2, 3, 6, 7, 6, 3, -2]) / 21.0
MAD_SCALE = 0.6744897501960817
SLOW_SHARE = 0.2
MIN_CENTRES = 30


def cells() -> list[dict[str, float | str]]:
    names = list(GRID)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*GRID.values())]


def _min_jerk(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


@dataclass
class Episode:
    start: float
    end: float
    parts: list[tuple[float, float, float, float]]  # (t0, duration, dx, dy)
    supported: bool = True

    def position(self, t: np.ndarray) -> np.ndarray:
        out = np.zeros((len(t), 2))
        for t0, dur, dx, dy in self.parts:
            s = _min_jerk((t - t0) / dur)
            out[:, 0] += dx * s
            out[:, 1] += dy * s
        return out


@dataclass
class TruthBout:
    """A kinematic bout of the noise-free speed: onset / offset where speed crosses
    `TRUTH_ONSET_SHARE` of the bout's own peak (the standard kinematic onset rule, fixed
    independently of every candidate); `clean` = the noise-free interval speed from rest to
    rest (the whole movement unit SPARC is defined on)."""

    start: float
    end: float
    clean: np.ndarray
    peak: float
    supported: bool = True


@dataclass
class Series:
    sid: str
    kind: str
    rate: float
    noise_scale: float
    t: np.ndarray
    x: np.ndarray  # NaN = missing slot (kept row)
    y: np.ndarray
    frame: np.ndarray
    truth: list[TruthBout]
    artifacts: list[tuple[float, float]] = field(default_factory=list)


def _episode(rng: np.random.Generator, t0: float, peak_speed: float) -> Episode:
    """1-3 overlapping minimum-jerk parts; the first part's peak speed = `peak_speed`
    (a minimum-jerk part of amplitude A over T peaks at 1.875 A / T)."""
    parts: list[tuple[float, float, float, float]] = []
    n_parts = int(rng.choice([1, 1, 2, 3]))
    start = t0
    angle = rng.uniform(0, 2 * np.pi)
    for i in range(n_parts):
        duration = float(rng.uniform(0.4, 1.4)) if n_parts > 1 else float(rng.uniform(0.4, 2.5))
        speed = peak_speed
        if i:
            prev_t0, prev_dur, *_ = parts[-1]
            start = prev_t0 + prev_dur * (1 - rng.uniform(0.2, 0.6))
            angle += rng.choice([rng.uniform(-1.0, 1.0), np.pi + rng.uniform(-0.5, 0.5)])
            speed = peak_speed * rng.uniform(0.5, 1.0)
        a = speed * duration / 1.875
        parts.append((start, duration, a * np.cos(angle), a * np.sin(angle)))
    end = max(p[0] + p[1] for p in parts)
    return Episode(t0, end, parts)


def _noise(rng: np.random.Generator, n: int, sd: float) -> np.ndarray:
    """White + AR(1)-correlated (rho 0.8) noise, equal variance shares; per-coordinate SD `sd`."""
    white = rng.normal(0, sd / math.sqrt(2), (n, 2))
    corr = np.zeros((n, 2))
    innov = rng.normal(0, sd / math.sqrt(2) * math.sqrt(1 - 0.8**2), (n, 2))
    for i in range(1, n):
        corr[i] = 0.8 * corr[i - 1] + innov[i]
    return white + corr


def make_series(
    rng: np.random.Generator,
    sid: str,
    kind: str,
    rate: float,
    noise_sd: float,
    scale: float,
    artifacts_on: bool = True,
) -> Series:
    """One synthetic wrist series of `kind` (KINDS) with known episodes and, unless
    `artifacts_on` is False (the noise-free admissibility bank), drift, glitches and holes."""
    pause = {
        "standard": (0.3, 2.0),
        "close": (0.15, 0.3),
        "unequal": (0.3, 2.0),
        "busy": (0.15, 0.5),
    }
    episodes: list[Episode] = []
    if kind != "stationary":
        t = float(rng.uniform(0.5, 1.5))
        while True:
            # Corpus wrist p95 smoothed speed spans 0.05-0.35 frame/s (asset-side p10-p90).
            peak = float(np.exp(rng.uniform(np.log(PEAK_SPEED[0]), np.log(PEAK_SPEED[1]))))
            if kind == "unequal" and rng.random() < 0.5:
                peak *= 0.2
            ep = _episode(rng, t, peak)
            if ep.end > SERIES_SEC - 0.5:
                break
            episodes.append(ep)
            t = ep.end + float(rng.uniform(*pause[kind]))
    n = int(SERIES_SEC * rate)
    t_grid = np.arange(n) / rate
    pos = np.tile([0.5, 0.5], (n, 1)).astype(float)
    for ep in episodes:
        pos += ep.position(t_grid)
    clean_speed = np.hypot(*np.diff(pos, axis=0).T) * rate
    if artifacts_on and rng.random() < 0.5:  # slow drift
        pos[:, 0] += 0.004 * np.sin(2 * np.pi * t_grid / float(rng.uniform(6, 12)))
    pos += _noise(rng, n, noise_sd * scale)
    artifacts: list[tuple[float, float]] = []
    if artifacts_on and kind != "stationary" and rng.random() < 0.5:  # glitch surviving median
        g0 = int(rng.integers(int(0.1 * n), int(0.9 * n)))
        length = int(rng.integers(5, 8))
        sign = np.where(np.arange(length) % 2 == 0, 1.0, -1.0)
        pos[g0 : g0 + length, 0] += sign * float(rng.uniform(0.02, 0.05))
        artifacts.append((g0 / rate, (g0 + length) / rate))
    observed = np.ones(n, bool)
    for _ in range(int(rng.integers(0, 3)) if artifacts_on else 0):  # 1-2 slot holes
        h0 = int(rng.integers(5, n - 5))
        observed[h0 : h0 + int(rng.integers(1, 3))] = False
    if artifacts_on and rng.random() < 0.4:  # one splitting hole
        h0 = int(rng.integers(int(0.1 * n), int(0.9 * n)))
        hl = int(rng.integers(int(0.35 * rate), int(1.0 * rate)))
        observed[h0 : h0 + hl] = False
        artifacts.append((h0 / rate, (h0 + hl) / rate))
    # Half the holes drop the row instead of blanking it: both forms must read alike.
    drop_rows = rng.random() < 0.5
    keep = observed | (not drop_rows)
    x = np.where(observed, pos[:, 0], np.nan)[keep]
    y = np.where(observed, pos[:, 1], np.nan)[keep]
    frames = np.arange(n)[keep]
    truth = truth_bouts(clean_speed, rate)
    _mark_supported(truth, artifacts)
    return Series(sid, kind, rate, scale, t_grid[keep], x, y, frames, truth, artifacts)


def truth_bouts(speed: np.ndarray, rate: float) -> list[TruthBout]:
    """Kinematic bouts of a noise-free interval speed series (interval i = slot i -> i + 1):
    each rest-to-rest run of motion, bounded where speed crosses `TRUTH_ONSET_SHARE` of the
    run's own peak; `clean` spans the whole run (speed zero at both ends)."""
    moving = speed > 1e-12
    edges = np.flatnonzero(np.diff(np.concatenate([[0], moving.astype(int), [0]])))
    out = []
    for lo, hi in zip(edges[::2], edges[1::2] - 1, strict=True):
        run = speed[lo : hi + 1]
        own = float(run.max())
        above = np.flatnonzero(run >= TRUTH_ONSET_SHARE * own)
        a, b = lo + int(above[0]), lo + int(above[-1])
        clean = speed[max(lo - 1, 0) : min(hi + 2, len(speed))].copy()
        out.append(TruthBout(a / rate, (b + 1) / rate, clean, own))
    return out


def _mark_supported(truth: list[TruthBout], artifacts: Sequence[tuple[float, float]]) -> None:
    top = max((b.peak for b in truth), default=0.0)
    for i, b in enumerate(truth):
        before = truth[i - 1].end if i else -math.inf
        after = truth[i + 1].start if i + 1 < len(truth) else math.inf
        b.supported = (
            b.start - before >= SUPPORT_GAP_SEC
            and after - b.end >= SUPPORT_GAP_SEC
            and b.start >= SUPPORT_GAP_SEC
            and b.end <= SERIES_SEC - SUPPORT_GAP_SEC
            and b.peak >= SUPPORT_PEAK_SHARE * top
            and all(
                a1 < b.start - SUPPORT_GAP_SEC or a0 > b.end + SUPPORT_GAP_SEC
                for a0, a1 in artifacts
            )
        )


def make_bank(
    seed: int,
    per_stratum: int,
    noise_sd: float,
    scales: Sequence[float] = NOISE_SCALES,
    artifacts_on: bool = True,
) -> list[Series]:
    """Every rate x noise scale x kind stratum, `per_stratum` series each, from one seed."""
    rng = np.random.default_rng(seed)
    bank = []
    for rate, scale, kind in itertools.product(RATES, scales, KINDS):
        for i in range(per_stratum):
            sid = f"{kind}-{rate:g}-{scale:g}-{i}"
            bank.append(make_series(rng, sid, kind, rate, noise_sd, scale, artifacts_on))
    return bank


def match(
    truth: Sequence[tuple[float, float]], detected: Sequence[tuple[float, float]]
) -> list[tuple[int, int]]:
    """One-to-one maximum-cardinality matching, then minimum total boundary error.

    A pair is admissible when both bounds fall within max(0.1 s, 0.1 x true duration).
    """
    if not truth or not detected:
        return []
    big = 1e6
    cost = np.full((len(truth), len(detected)), big)
    for i, (ts, te) in enumerate(truth):
        tol = max(0.1, 0.1 * (te - ts))
        for j, (ds, de) in enumerate(detected):
            if abs(ds - ts) <= tol and abs(de - te) <= tol:
                cost[i, j] = abs(ds - ts) + abs(de - te)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if cost[i, j] < big]


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def score(
    bank: Sequence[Series],
    detections: Mapping[str, Sequence[Mapping]],
    truth_features: Mapping[str, Sequence[Mapping]],
    support: str,
    prominence_index: int,
) -> dict[str, dict]:
    """Pooled detection + feature error per stratum (`rate/noise/kind`) and `all`.

    `detections[sid]` = R rows (start_sec, end_sec, censored, sparc_<support>, peaks[...]);
    `truth_features[sid]` = per episode `sparc`, `peaks[...]` from the shipped kernels.
    """
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    sparc_err: dict[str, list[float]] = defaultdict(list)
    for s in bank:
        det = list(detections.get(s.sid, []))
        truths = [(b.start, b.end) for b in s.truth]
        pairs = match(truths, [(d["start_sec"], d["end_sec"]) for d in det])
        for key in (f"{s.rate:g}/{s.noise_scale:g}/{s.kind}", "all"):
            a = acc[key]
            a["truth"] += len(truths)
            a["detected"] += len(det)
            a["tp"] += len(pairs)
            a["fn"] += len(truths) - len(pairs)
            a["fp"] += len(det) - len(pairs)
            a["fragmented"] += sum(
                sum(_overlap(t, (d["start_sec"], d["end_sec"])) for d in det) >= 2 for t in truths
            )
            a["merged"] += sum(
                sum(_overlap(t, (d["start_sec"], d["end_sec"])) for t in truths) >= 2 for d in det
            )
            a["retained"] += sum(not det[j]["censored"] for _, j in pairs)
            for i, j in pairs:
                d, tf = det[j], truth_features[s.sid][i]
                if d["censored"]:
                    continue
                value = d.get(f"sparc_{support}")
                if value is not None and tf["sparc"] is not None:
                    sparc_err[key].append(abs(value - tf["sparc"]))
                a["peaks_scored"] += 1
                a["peaks_exact"] += d["peaks"][prominence_index] == tf["peaks"][prominence_index]
                a["duration_err"] += abs(
                    (d["end_sec"] - d["start_sec"]) - (s.truth[i].end - s.truth[i].start)
                )
    out = {}
    for key, a in acc.items():
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        out[key] = {
            "truth": int(a["truth"]),
            "detected": int(a["detected"]),
            "precision": tp / (tp + fp) if tp + fp else math.nan,
            "recall": tp / (tp + fn) if tp + fn else math.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan,
            "fragmented": int(a["fragmented"]),
            "merged": int(a["merged"]),
            "retention": a["retained"] / tp if tp else math.nan,
            "sparc_abs_err_median": float(np.median(sparc_err[key]))
            if sparc_err[key]
            else math.nan,
            "peaks_exact": a["peaks_exact"] / a["peaks_scored"] if a["peaks_scored"] else math.nan,
            "duration_abs_err_mean": a["duration_err"] / a["peaks_scored"]
            if a["peaks_scored"]
            else math.nan,
        }
    return out


def exact_on_supported(bank: Sequence[Series], detections: Mapping[str, Sequence[Mapping]]) -> bool:
    """Admissibility: every supported episode matched, no detection outside unsupported truth."""
    for s in bank:
        det = list(detections.get(s.sid, []))
        truths = [(b.start, b.end) for b in s.truth]
        pairs = dict(match(truths, [(d["start_sec"], d["end_sec"]) for d in det]))
        for i, e in enumerate(s.truth):
            if e.supported and i not in pairs:
                return False
        used = set(pairs.values())
        for j, d in enumerate(det):
            span = (d["start_sec"], d["end_sec"])
            if j not in used and not any(
                _overlap(span, (e.start, e.end)) for e in s.truth if not e.supported
            ):
                return False
    return True


def rank(results: Sequence[Mapping]) -> list[Mapping]:
    """Admissible cells, development F1 desc, then SPARC error asc, then enumeration order."""
    ok = [r for r in results if r["admissible"]]
    return sorted(
        ok,
        key=lambda r: (
            -r["dev"]["all"]["f1"],
            r["dev"]["all"]["sparc_abs_err_median"],
            r["order"],
        ),
    )


def noise_proxy(t: np.ndarray, x: np.ndarray, y: np.ndarray, rate: float) -> float | None:
    """D06 residual proxy: per-coordinate MAD-scaled residual of a centred 7-slot quadratic SG
    over fully observed stencils whose SG speed is in the lowest 20 %, coordinates by RMS."""
    slots = np.rint((t - t[0]) * rate).astype(int)
    n = int(slots[-1]) + 1
    gx, gy = np.full(n, np.nan), np.full(n, np.nan)
    gx[slots], gy[slots] = x, y
    ok = np.isfinite(gx) & np.isfinite(gy)
    if n < 7:
        return None
    full = np.lib.stride_tricks.sliding_window_view(ok, 7).all(axis=1)
    if full.sum() < MIN_CENTRES:
        return None
    wx = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(gx), 7)[full]
    wy = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(gy), 7)[full]
    centre = np.flatnonzero(full) + 3
    sx, sy = wx @ SG7, wy @ SG7
    # SG first derivative (7-slot quadratic): weights -3..3 / 28.
    d = np.arange(-3, 4) / 28.0
    speed = np.hypot(wx @ d, wy @ d) * rate
    slow = speed <= np.quantile(speed, SLOW_SHARE)
    rx, ry = (gx[centre] - sx)[slow], (gy[centre] - sy)[slow]

    def mad(v: np.ndarray) -> float:
        return float(np.median(np.abs(v - np.median(v))) / MAD_SCALE)

    return math.sqrt((mad(rx) ** 2 + mad(ry) ** 2) / 2)


def proxy_response(rate: float, seed: int, sd: float = 1e-3, n: int = 2000) -> float:
    """Measured proxy / true per-coordinate noise SD on stationary white + AR(1) noise."""
    rng = np.random.default_rng(seed)
    noise = _noise(rng, n, sd)
    t = np.arange(n) / rate
    value = noise_proxy(t, 0.5 + noise[:, 0], 0.5 + noise[:, 1], rate)
    if value is None:
        raise ValueError("noise proxy undefined on its own calibration input")
    return value / sd


def summarise(values: Iterable[float]) -> dict[str, float]:
    v = np.asarray(list(values), float)
    return {"n": len(v), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9))}

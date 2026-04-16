"""Minimal benchmark harness (no external deps).

Each benchmark is a zero-argument callable.  ``run_bench`` calls it
*warmup* times to prime caches, then *iters* times back-to-back,
measures wall time with ``time.perf_counter_ns``, and returns summary
statistics in nanoseconds.  A pytest-benchmark-style dict layout makes
the JSON easy to aggregate.
"""

from __future__ import annotations

import gc
import json
import pathlib
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field


@dataclass
class BenchResult:
    name: str
    group: str
    params: dict
    n: int
    warmup: int
    min_ns: int
    median_ns: int
    mean_ns: float
    p95_ns: int
    max_ns: int
    stdev_ns: float
    total_ns: int
    throughput_per_s: float
    notes: str = ""
    samples_ns: list[int] = field(default_factory=list)


def _format_ns(ns: float) -> str:
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def run_bench(
    name: str,
    fn: Callable[[], object],
    *,
    group: str,
    params: dict | None = None,
    iters: int = 200,
    warmup: int = 10,
    notes: str = "",
    keep_samples: bool = False,
) -> BenchResult:
    """Run *fn* ``iters`` times, return a :class:`BenchResult`.

    GC is disabled during timing and re-enabled afterwards.  Warmup runs
    are always executed before timing, even when ``iters`` is small.
    """
    params = params or {}

    # Warmup
    for _ in range(max(0, warmup)):
        fn()

    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        samples: list[int] = []
        t_total = 0
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            dt = t1 - t0
            samples.append(dt)
            t_total += dt
    finally:
        if gc_enabled:
            gc.enable()

    samples.sort()
    n = len(samples)
    p95_idx = min(n - 1, max(0, round(0.95 * (n - 1))))
    mean = t_total / n
    return BenchResult(
        name=name,
        group=group,
        params=params,
        n=n,
        warmup=warmup,
        min_ns=samples[0],
        median_ns=samples[n // 2],
        mean_ns=mean,
        p95_ns=samples[p95_idx],
        max_ns=samples[-1],
        stdev_ns=statistics.pstdev(samples) if n > 1 else 0.0,
        total_ns=t_total,
        throughput_per_s=(1e9 / mean) if mean > 0 else 0.0,
        notes=notes,
        samples_ns=samples if keep_samples else [],
    )


def print_result(r: BenchResult) -> None:
    p = ", ".join(f"{k}={v}" for k, v in r.params.items()) or "-"
    print(
        f"  {r.name:<48} [{p:<38}] "
        f"median={_format_ns(r.median_ns):>10}  "
        f"p95={_format_ns(r.p95_ns):>10}  "
        f"{r.throughput_per_s:>10.0f} ops/s"
    )


def save_results(results: list[BenchResult], out_path: str | pathlib.Path) -> pathlib.Path:
    """Dump results to JSON (newest run overwrites)."""
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def run_group(
    group: str,
    cases: list[tuple[str, Callable[[], object], dict]],
    *,
    iters: int = 200,
    warmup: int = 10,
) -> list[BenchResult]:
    """Run every (name, fn, params) case in *cases* and print as they land."""
    print(f"\n=== {group} ===")
    out: list[BenchResult] = []
    for name, fn, params in cases:
        r = run_bench(name, fn, group=group, params=params, iters=iters, warmup=warmup)
        print_result(r)
        out.append(r)
    return out

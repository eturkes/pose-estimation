#!/usr/bin/env python3
"""Run the full benchmark suite and write JSON results.

Usage (from project root):
    uv run python scripts/benchmarks/run.py

    # Limit to one group:
    uv run python scripts/benchmarks/run.py smoothing

    # Quick pass with fewer iterations:
    uv run python scripts/benchmarks/run.py --quick
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import time

_GROUPS = [
    "smoothing",
    "constraints",
    "matching",
    "detection",
    "processing",
    "drawing",
    "metrics",
]


def _load_module(group: str):
    return importlib.import_module(f"scripts.benchmarks.bench_{group}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pose_estimation micro-benchmarks")
    parser.add_argument(
        "groups",
        nargs="*",
        choices=[*_GROUPS, []],
        default=[],
        help=f"Groups to run (default: all).  Options: {', '.join(_GROUPS)}",
    )
    parser.add_argument(
        "--output",
        default="output/benchmarks/results.json",
        help="Where to write the aggregated JSON results",
    )
    parser.add_argument("--quick", action="store_true", help="Fewer iterations per case")
    args = parser.parse_args()

    groups = args.groups or _GROUPS

    # Inject project root onto sys.path so ``scripts.benchmarks.*`` resolves
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scripts.benchmarks._harness import save_results

    all_results = []
    t0 = time.time()
    for g in groups:
        mod = _load_module(g)
        kwargs: dict = {}
        if args.quick:
            kwargs["iters"] = max(10, mod.run.__defaults__[0] // 4)
            kwargs["warmup"] = 2
        results = mod.run(**kwargs)
        all_results.extend(results)

    out_path = save_results(all_results, args.output)
    elapsed = time.time() - t0
    print(f"\nSaved {len(all_results)} results in {elapsed:.1f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

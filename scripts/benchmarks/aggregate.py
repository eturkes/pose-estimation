#!/usr/bin/env python3
"""Aggregate ``output/benchmarks/results.json`` into digestible tables.

Produces a Markdown file summarising per-group medians, top-N hotspots
overall, and throughput tables for repeated parametric sweeps.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict


def _format_ns(ns: float) -> str:
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def _params_str(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items()) if p else "-"


def _group_stats(results: list[dict]) -> list[dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)

    out = []
    for g, items in sorted(by_group.items()):
        medians = [r["median_ns"] for r in items]
        out.append(
            {
                "group": g,
                "n_cases": len(items),
                "min_median_ns": min(medians),
                "max_median_ns": max(medians),
                "slowest_case": max(items, key=lambda r: r["median_ns"]),
                "fastest_case": min(items, key=lambda r: r["median_ns"]),
            }
        )
    return out


def _top_hotspots(results: list[dict], n: int = 15) -> list[dict]:
    return sorted(results, key=lambda r: r["median_ns"], reverse=True)[:n]


def render_markdown(results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# pose_estimation micro-benchmark results")
    lines.append("")
    lines.append(f"Total cases measured: **{len(results)}**")
    lines.append("")

    # Group summary
    lines.append("## Per-group summary")
    lines.append("")
    lines.append("| Group | Cases | Fastest median | Slowest median | Slowest case |")
    lines.append("|-------|------:|---------------:|---------------:|--------------|")
    for g in _group_stats(results):
        slow = g["slowest_case"]
        lines.append(
            f"| {g['group']} | {g['n_cases']} | {_format_ns(g['min_median_ns'])} "
            f"| {_format_ns(g['max_median_ns'])} "
            f"| `{slow['name']}` [{_params_str(slow['params'])}] |"
        )
    lines.append("")

    # Top hotspots overall
    lines.append("## Top 20 slowest cases (by median)")
    lines.append("")
    lines.append("| # | Case | Params | Median | p95 | ops/s |")
    lines.append("|---|------|--------|-------:|----:|------:|")
    for i, r in enumerate(_top_hotspots(results, n=20), 1):
        lines.append(
            f"| {i} | `{r['name']}` ({r['group']}) | {_params_str(r['params'])} | "
            f"{_format_ns(r['median_ns'])} | {_format_ns(r['p95_ns'])} | "
            f"{r['throughput_per_s']:.0f} |"
        )
    lines.append("")

    # Per-group detail tables
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)

    for g in sorted(by_group):
        lines.append(f"## {g}")
        lines.append("")
        lines.append("| Case | Params | Median | p95 | ops/s |")
        lines.append("|------|--------|-------:|----:|------:|")
        rows = sorted(by_group[g], key=lambda r: r["median_ns"], reverse=True)
        for r in rows:
            lines.append(
                f"| `{r['name']}` | {_params_str(r['params'])} | "
                f"{_format_ns(r['median_ns'])} | {_format_ns(r['p95_ns'])} | "
                f"{r['throughput_per_s']:.0f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="output/benchmarks/results.json")
    p.add_argument("--output", default="output/benchmarks/results.md")
    args = p.parse_args()

    data = json.loads(pathlib.Path(args.input).read_text())
    md = render_markdown(data["results"])
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

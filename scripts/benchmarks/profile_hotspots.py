#!/usr/bin/env python3
"""cProfile the slowest N benchmark cases by median_ns and dump stats.

Reads ``output/benchmarks/results.json``, picks the top cases (excluding
aggregate-over-frames cases that dominate due to sheer iteration), and
runs them under cProfile.  Emits one ``.prof`` file per hotspot plus a
human-readable text summary.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pathlib
import pstats
import sys

# Known per-iteration hotspots to profile.  Each entry is a (label,
# importable factory) pair.  The factory must return a zero-arg callable
# that runs one representative iteration of the hot path.
_PROFILE_TARGETS: list[tuple[str, str]] = [
    # (label, "module:factory_name:arg_json")
    ("draw_body_landmarks_body33_4bodies", "drawing_factory:body4_33"),
    ("draw_hand_landmarks_8hands", "drawing_factory:hand8"),
    ("savgol_body33_1800rows", "savgol_factory:body33_1800"),
    ("match_hands_to_arms_4x8_body33", "matching_factory:b4h8_body33"),
    ("metrics_record_480frames_body33", "metrics_factory:f480_body33"),
    ("pose_smoother_bodies_4_body33_120f", "smoothing_factory:b4_body33_120f"),
    ("synthesise_hands_4bodies_6palm_body", "processing_factory:syn_b4p6_body"),
]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def drawing_factory(which: str):
    from scripts.benchmarks._fixtures import ARM_KP, BODY_KP
    from scripts.benchmarks.bench_drawing import _draw_body_case, _draw_hand_case

    if which == "body4_33":
        return _draw_body_case(n_bodies=4, n_kp=BODY_KP)
    if which == "body4_12":
        return _draw_body_case(n_bodies=4, n_kp=ARM_KP)
    if which == "hand8":
        return _draw_hand_case(n_hands=8)
    raise ValueError(which)


def savgol_factory(which: str):
    from scripts.benchmarks.bench_metrics import _savgol_csv_case

    if which == "body33_1800":
        return _savgol_csv_case(n_rows=1800, n_cols=99, window=11)
    if which == "arm12_1800":
        return _savgol_csv_case(n_rows=1800, n_cols=36, window=11)
    raise ValueError(which)


def matching_factory(which: str):
    from scripts.benchmarks._fixtures import BODY_KP
    from scripts.benchmarks.bench_matching import _match_case

    if which == "b4h8_body33":
        return _match_case(n_bodies=4, n_hands=8, n_kp=BODY_KP, close=True)
    raise ValueError(which)


def metrics_factory(which: str):
    from scripts.benchmarks._fixtures import BODY_KP
    from scripts.benchmarks.bench_metrics import _metrics_record_case

    if which == "f480_body33":
        return _metrics_record_case(n_frames=480, n_kp=BODY_KP, detail=False)
    raise ValueError(which)


def smoothing_factory(which: str):
    from scripts.benchmarks._fixtures import BODY_KP
    from scripts.benchmarks.bench_smoothing import _pose_smoother_bodies_case

    if which == "b4_body33_120f":
        return _pose_smoother_bodies_case(n_bodies=4, n_kp=BODY_KP, n_frames=120)
    raise ValueError(which)


def processing_factory(which: str):
    from scripts.benchmarks.bench_processing import _synthesise_hands_case

    if which == "syn_b4p6_body":
        return _synthesise_hands_case(n_bodies=4, n_palm_dets=6, mode="body")
    raise ValueError(which)


_FACTORIES = {
    "drawing_factory": drawing_factory,
    "savgol_factory": savgol_factory,
    "matching_factory": matching_factory,
    "metrics_factory": metrics_factory,
    "smoothing_factory": smoothing_factory,
    "processing_factory": processing_factory,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _profile_target(label: str, spec: str, runs: int, outdir: pathlib.Path) -> str:
    factory_name, arg = spec.split(":", 1)
    fn = _FACTORIES[factory_name](arg)

    # Warmup
    for _ in range(3):
        fn()

    prof_path = outdir / f"{label}.prof"
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(runs):
        fn()
    pr.disable()
    pr.dump_stats(str(prof_path))

    # Build the text summary
    lines = [f"# {label}  ({runs} runs under cProfile)"]
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    # Capture top 20 cumulative
    import io

    buf = io.StringIO()
    stats.stream = buf
    stats.print_stats(20)
    lines.append("## Top 20 cumulative:")
    lines.append(buf.getvalue())

    stats.sort_stats(pstats.SortKey.TIME)
    buf2 = io.StringIO()
    stats.stream = buf2
    stats.print_stats(20)
    lines.append("## Top 20 tottime:")
    lines.append(buf2.getvalue())

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="output/benchmarks/profiles")
    parser.add_argument("--runs", type=int, default=50, help="Iterations per target")
    parser.add_argument(
        "targets",
        nargs="*",
        help="Target labels (default: all)",
    )
    args = parser.parse_args()

    # Make project-root importable
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected = _PROFILE_TARGETS
    if args.targets:
        selected = [t for t in _PROFILE_TARGETS if t[0] in set(args.targets)]
        if not selected:
            print(f"No matching targets.  Known: {[t[0] for t in _PROFILE_TARGETS]}")
            return 2

    summaries: list[str] = []
    for label, spec in selected:
        print(f"Profiling {label} ...")
        summary = _profile_target(label, spec, runs=args.runs, outdir=outdir)
        summaries.append(summary)

    summary_path = outdir / "summary.txt"
    summary_path.write_text("\n\n".join(summaries))
    print(f"Wrote {summary_path} ({len(selected)} targets, {args.runs} runs each)")

    # Also write a quick JSON summary
    results = []
    for label, spec in selected:
        prof_path = outdir / f"{label}.prof"
        if not prof_path.exists():
            continue
        st = pstats.Stats(str(prof_path))
        # Unused, but loaded to confirm the file is valid
        total = sum(v[3] for v in st.stats.values())  # cumulative
        results.append({"label": label, "spec": spec, "total_cumulative_s": total})
    (outdir / "summary.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

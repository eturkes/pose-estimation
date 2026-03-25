#!/usr/bin/env python3
"""Parameter sweep benchmark for pose estimation optimisation.

Runs the pipeline headlessly on one or more reference videos with
different parameter configurations and collects metrics for
comparison.

Usage:
    # Single parameter sweep
    python benchmark.py --source video.mp4 --sweep body_min_cutoff 0.1 0.3 0.5 1.0

    # From a YAML config file
    python benchmark.py --source video.mp4 --config sweep.yaml

    # Batch with all defaults
    python benchmark.py --batch-dir videos/

YAML config format:
    body_min_cutoff: [0.1, 0.3, 0.5, 1.0]
    body_beta: [0.3, 0.5, 0.7]
    hand_min_cutoff: [0.5, 1.0, 2.0]
"""

import argparse
import csv
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Tuneable parameter definitions
# ---------------------------------------------------------------------------

# Maps CLI sweep names to (module, attribute/kwarg, default_value) triples.
# The benchmark monkey-patches these before running.
TUNEABLE_PARAMS = {
    # Detection thresholds
    "det_score_thresh":  ("processing", "det_score_threshold", 0.5),
    "hand_flag_thresh":  ("processing", "lm_score_threshold", 0.65),

    # One-Euro filter: body
    "body_min_cutoff":   ("smoothing", "body_min_cutoff", 0.3),
    "body_beta":         ("smoothing", "body_beta", 0.5),

    # One-Euro filter: hand
    "hand_min_cutoff":   ("smoothing", "hand_min_cutoff", 1.0),
    "hand_beta":         ("smoothing", "hand_beta", 0.3),

    # Confidence weighting
    "confidence_gamma":  ("smoothing", "confidence_gamma", 2.0),

    # Detection smoothing
    "det_smooth_alpha":  ("processing", "det_smooth_alpha", 0.5),

    # Bone-length constraints
    "bone_ema_alpha":    ("constraints", "bone_ema_alpha", 0.05),
    "bone_tolerance":    ("constraints", "bone_tolerance", 0.4),

    # Carry-forward
    "carry_grace":       ("smoothing", "carry_grace", 10),
}


def _apply_param_overrides(overrides):
    """Monkey-patch modules with parameter overrides.

    This modifies the constructors and default arguments so that
    process_video picks them up without code changes.

    Returns a cleanup function that restores originals.
    """
    import smoothing
    import processing
    import constraints

    originals = {}

    for key, value in overrides.items():
        if key not in TUNEABLE_PARAMS:
            raise ValueError(f"Unknown tuneable parameter: {key}")
        module_name, attr, _ = TUNEABLE_PARAMS[key]
        mod = {"smoothing": smoothing,
               "processing": processing,
               "constraints": constraints}[module_name]

        # Patch the OneEuroFilter defaults via PoseSmoother methods
        if key == "body_min_cutoff":
            originals["body_min_cutoff"] = 0.3
            # Patch the lambda in smooth_bodies
            _patch_body_filter(smoothing, "min_cutoff", value)
        elif key == "body_beta":
            originals["body_beta"] = 0.5
            _patch_body_filter(smoothing, "beta", value)
        elif key == "hand_min_cutoff":
            originals["hand_min_cutoff"] = 1.0
            _patch_hand_filter(smoothing, "min_cutoff", value)
        elif key == "hand_beta":
            originals["hand_beta"] = 0.3
            _patch_hand_filter(smoothing, "beta", value)
        elif key == "confidence_gamma":
            originals["confidence_gamma"] = 2.0
            _patch_body_filter(smoothing, "gamma", value)
            _patch_hand_filter(smoothing, "gamma", value)
        elif key == "carry_grace":
            originals["carry_grace"] = 10
            smoothing._BENCH_CARRY_GRACE = int(value)
        elif key == "det_smooth_alpha":
            originals["det_smooth_alpha"] = 0.5
            processing._BENCH_DET_ALPHA = float(value)
        elif key == "bone_ema_alpha":
            originals["bone_ema_alpha"] = 0.05
            constraints._BENCH_BONE_ALPHA = float(value)
        elif key == "bone_tolerance":
            originals["bone_tolerance"] = 0.4
            constraints._BENCH_BONE_TOL = float(value)

    return originals


def _patch_body_filter(smoothing_mod, attr, value):
    """Store override for body filter construction."""
    if not hasattr(smoothing_mod, "_BENCH_BODY_FILTER"):
        smoothing_mod._BENCH_BODY_FILTER = {}
    smoothing_mod._BENCH_BODY_FILTER[attr] = value


def _patch_hand_filter(smoothing_mod, attr, value):
    """Store override for hand filter construction."""
    if not hasattr(smoothing_mod, "_BENCH_HAND_FILTER"):
        smoothing_mod._BENCH_HAND_FILTER = {}
    smoothing_mod._BENCH_HAND_FILTER[attr] = value


# ---------------------------------------------------------------------------
# Run engine
# ---------------------------------------------------------------------------

def run_single(source, output_dir, run_label, overrides, device, tracking,
               single_subject, models=None, palm_anchors=None,
               pose_anchors=None):
    """Run the pipeline once with given parameters, collect metrics.

    Uses a subprocess call to main.py with --headless to ensure clean
    module state (no monkey-patching side effects between runs).
    """
    run_dir = pathlib.Path(output_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write param config for traceability
    config_path = run_dir / "params.json"
    with open(config_path, "w") as f:
        json.dump(overrides, f, indent=2)

    # For now, use subprocess to main.py in headless mode.
    # This is simpler and avoids state leakage between runs.
    cmd = [
        sys.executable, "main.py",
        "--source", str(source),
        "--output-dir", str(run_dir),
        "--device", device,
        "--tracking", tracking,
        "--headless",
    ]
    if single_subject:
        cmd.append("--single-subject")

    # Pass overrides as environment variables (read by patched modules)
    env = os.environ.copy()
    for key, value in overrides.items():
        env[f"POSE_BENCH_{key.upper()}"] = str(value)

    print(f"  [{run_label}] Starting: {' '.join(f'{k}={v}' for k, v in overrides.items())}")
    t0 = time.time()

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [{run_label}] FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[:500])
        return None

    print(f"  [{run_label}] Done in {elapsed:.1f}s")

    # Find the metrics CSV
    metrics_files = list(run_dir.glob("*_metrics.csv"))
    if metrics_files:
        return {
            "run": run_label,
            "params": overrides,
            "metrics_csv": str(metrics_files[0]),
            "elapsed_s": round(elapsed, 1),
        }
    return {
        "run": run_label,
        "params": overrides,
        "metrics_csv": None,
        "elapsed_s": round(elapsed, 1),
    }


def generate_grid(sweep_spec):
    """Generate all parameter combinations from a spec dict.

    sweep_spec: {param_name: [value1, value2, ...], ...}
    Returns list of dicts, one per combination.
    """
    if not sweep_spec:
        return [{}]

    keys = list(sweep_spec.keys())
    values = [sweep_spec[k] for k in keys]
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep benchmark for pose estimation")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", help="Single video file")
    source_group.add_argument("--batch-dir", help="Directory of videos")

    parser.add_argument("--output-dir", default="benchmark_output",
                        help="Base directory for benchmark results")
    parser.add_argument("--device", default="NPU",
                        help="OpenVINO device (default: NPU)")
    parser.add_argument("--tracking", default="hands-arms",
                        choices=["hands", "hands-arms", "body"])
    parser.add_argument("--single-subject", action="store_true")

    # Parameter sweep: --sweep <name> <val1> <val2> ...
    parser.add_argument("--sweep", nargs="+", action="append",
                        metavar="PARAM_OR_VALUE",
                        help="Sweep a parameter: --sweep body_min_cutoff 0.1 0.3 0.5")

    # YAML config
    parser.add_argument("--config",
                        help="YAML file with parameter grid spec")

    args = parser.parse_args()

    # Build parameter grid
    sweep_spec = {}

    if args.sweep:
        for sweep_args in args.sweep:
            if len(sweep_args) < 2:
                parser.error("--sweep needs at least param_name and one value")
            param_name = sweep_args[0]
            if param_name not in TUNEABLE_PARAMS:
                parser.error(
                    f"Unknown parameter: {param_name}\n"
                    f"Available: {', '.join(sorted(TUNEABLE_PARAMS))}")
            values = [float(v) for v in sweep_args[1:]]
            sweep_spec[param_name] = values

    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                yaml_spec = yaml.safe_load(f)
            if isinstance(yaml_spec, dict):
                for k, v in yaml_spec.items():
                    if k in TUNEABLE_PARAMS:
                        sweep_spec[k] = [float(x) for x in v] if isinstance(v, list) else [float(v)]
        except ImportError:
            # Fall back to JSON
            with open(args.config) as f:
                json_spec = json.load(f)
            for k, v in json_spec.items():
                if k in TUNEABLE_PARAMS:
                    sweep_spec[k] = [float(x) for x in v] if isinstance(v, list) else [float(v)]

    # Always include baseline (all defaults)
    grid = generate_grid(sweep_spec)
    if not any(all(v == TUNEABLE_PARAMS[k][2] for k, v in combo.items()) for combo in grid):
        # Add default baseline
        baseline = {k: TUNEABLE_PARAMS[k][2] for k in sweep_spec}
        grid.insert(0, baseline)

    # Determine sources
    if args.batch_dir:
        VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        batch_path = pathlib.Path(args.batch_dir)
        sources = sorted(
            p for p in batch_path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not sources:
            print(f"No video files found in {args.batch_dir}")
            return
    else:
        sources = [pathlib.Path(args.source)]

    # Run sweep
    out_base = pathlib.Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_runs = len(grid) * len(sources)
    run_num = 0

    for source in sources:
        video_stem = source.stem
        print(f"\n{'='*60}")
        print(f"  Video: {source.name}")
        print(f"  Grid: {len(grid)} parameter combinations")
        print(f"{'='*60}")

        for i, overrides in enumerate(grid):
            run_num += 1
            # Build a human-readable label
            if overrides:
                parts = [f"{k}={v}" for k, v in overrides.items()]
                label = f"{video_stem}_{'_'.join(parts)}"
            else:
                label = f"{video_stem}_baseline"

            print(f"\n  Run {run_num}/{total_runs}")
            result = run_single(
                source, out_base, label, overrides,
                args.device, args.tracking, args.single_subject,
            )
            if result:
                all_results.append(result)

    # Write aggregate results
    results_path = out_base / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {results_path}")

    # Summary table
    if all_results:
        print(f"\n{'='*60}")
        print("  Benchmark Summary")
        print(f"{'='*60}")
        print(f"  {'Run':<40} {'Time(s)':>8} {'Metrics CSV'}")
        print(f"  {'-'*40} {'-'*8} {'-'*30}")
        for r in all_results:
            csv_str = pathlib.Path(r["metrics_csv"]).name if r["metrics_csv"] else "MISSING"
            print(f"  {r['run']:<40} {r['elapsed_s']:>8.1f} {csv_str}")

    print(f"\nDone. Run `Rscript analysis/summary.R {out_base}/` to generate reports.")


if __name__ == "__main__":
    main()

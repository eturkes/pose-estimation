# Optimization & benchmarking

Two distinct surfaces: **pipeline parameter sweeps** (find best params for the live pipeline) and **micro-benchmarks** (track hot-path performance regressions).

## Pipeline parameter sweeps — `benchmark.py`

Spawns headless subprocesses with `POSE_BENCH_*` env-var overrides; each run emits `*_metrics.csv` for comparison via `analysis/summary.R` or `analysis/compare.R`.

```bash
python -m pose_estimation.benchmark --source v.mp4 --sweep body_min_cutoff 0.1 0.3 0.5 1.0
python -m pose_estimation.benchmark --source v.mp4 --config sweep_quick.yaml
python -m pose_estimation.benchmark --source v.mp4 --config sweep_default.yaml
```

### Sweep parameters (env-var override, name → meaning)

| Param | Notes |
|-------|-------|
| `det_score_thresh` | Detection score threshold. |
| `hand_flag_thresh` | Hand-presence flag threshold. |
| `body_min_cutoff` | One Euro min cutoff (body). |
| `body_beta` | One Euro beta (body). |
| `hand_min_cutoff` | One Euro min cutoff (hands). |
| `hand_beta` | One Euro beta (hands). |
| `confidence_gamma` | Confidence-weighting exponent. |
| `det_smooth_alpha` | Detection EMA smoothing. |
| `bone_ema_alpha` | Bone-length EMA smoothing. |
| `bone_tolerance` | Bone-length deviation tolerance. |
| `bone_distal_weight` | Fraction of bone-length correction applied to distal keypoint (default 0.8). |
| `carry_grace` | Frames to keep using a carried detection. |
| `carry_damping` | Velocity decay for carry-forward extrapolation (default 0.8). |
| `outlier_cap` | Max unexpected displacement (px) before clamping (default 30). 0 disables. |
| `det_carry_frames` | Detection-level carry-forward grace period (default 3). |

### Sweep configs

- `sweep_default.yaml` — full grid across 8 params. Run 1–2 at a time to keep combos tractable.
- `sweep_quick.yaml` — `body_min_cutoff × det_smooth_alpha` (6 combos), targeted first pass.

## Metrics CSVs (`metrics.MetricsCollector`)

Headless or post-run, writes:

- `*_metrics.csv` — per-frame: jitter, confidence, smoothing delta, constraint corrections, detection counts, carry-forward state.
- `*_kp_detail.csv` (with `--metrics-detail`) — per-keypoint detail. Large; only enable when needed.

Diagnostic carriers: `metrics.SmoothingDiagnostics`, `metrics.ConstraintDiagnostics`.

## Micro-benchmarks — `scripts/benchmarks/`

Separate from the parameter sweep. Tracks hot-path regressions in core modules.

```bash
uv run python scripts/benchmarks/run.py             # all groups
uv run python scripts/benchmarks/run.py smoothing   # single group
uv run python scripts/benchmarks/run.py --quick     # fewer iterations
```

Groups (one `bench_<group>.py` per module): `smoothing`, `constraints`, `matching`, `detection`, `processing`, `drawing`, `metrics`.

Support files:
- `_fixtures.py` — synthetic detection/landmark inputs.
- `_harness.py` — timing utilities, JSON output.
- `aggregate.py` — combine multiple runs.
- `profile_hotspots.py` — cProfile sampling for deeper dives.

## NPU / OpenVINO

- `scripts/npu_compat.py` — sniff which models compile and run on NPU (some ops fall back to CPU). Useful before adding a new model to `MODEL_REGISTRY` (`run.py`).

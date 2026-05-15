# Pose Estimation

Real-time pose estimation combining MediaPipe TFLite models with Intel OpenVINO inference and rtmlib-based models (RTMW, DWPose, RTMPose). Supports hands-only, arms+hands (default), or whole-body+hands tracking from a webcam or video, with temporal smoothing, biomechanical constraints, and skeleton visualization.

> **AI agents:** start at `/CLAUDE.md` and `.claude/INDEX.md`. The rest of this README targets humans.

## Requirements

- Python 3.10+
- An OpenVINO-compatible device (NPU, CPU, or GPU)
- R + `renv` (only for the analysis scripts in `analysis/`)

## Setup

```bash
uv sync
```

Models download automatically on first run and are cached in `model/`. The directories `videos/`, `output/`, and `model/` are git-ignored.

## Entry points

Four console scripts (defined in `pyproject.toml`):

| Script | Module | Purpose |
|--------|--------|---------|
| `pose-estimation` | `pose_estimation.main` | MediaPipe pipeline (default). |
| `pose-estimation-run` | `pose_estimation.run` | Multi-backend: rtmlib (RTMW, DWPose, RTMPose) + MediaPipe. |
| `pose-estimation-benchmark` | `pose_estimation.benchmark` | Parameter sweep harness. |
| `pose-estimation-postprocess` | `pose_estimation.postprocess` | Savitzky-Golay smoothing for existing CSVs. |

### MediaPipe pipeline

```bash
python -m pose_estimation.main                            # webcam 0, NPU, hands-arms
python -m pose_estimation.main --source video.mp4
python -m pose_estimation.main --source 1                 # camera index
python -m pose_estimation.main --batch-dir videos/        # batch
python -m pose_estimation.main --device CPU               # CPU instead of NPU
python -m pose_estimation.main --no-flip                  # rear camera
python -m pose_estimation.main --model-dir /path/to/cache
```

Close the window or press **ESC** to exit.

### Unified entry point (rtmlib + MediaPipe)

```bash
python -m pose_estimation.run                                          # webcam 0, default model (rtmw-l)
python -m pose_estimation.run --model dwpose-m                         # DWPose wholebody
python -m pose_estimation.run --model rtmpose-m                        # body-only (17 kps)
python -m pose_estimation.run --model mediapipe                        # delegates to main.py
python -m pose_estimation.run --source video.mp4 --backend openvino --device NPU
python -m pose_estimation.run --headless                               # no display
```

| Model | Keypoints | Notes |
|-------|-----------|-------|
| `rtmw-l` (default) | 133 | RTMW-L wholebody (body + hands + face + feet). |
| `dwpose-m` | 133 | DWPose-M wholebody. |
| `rtmpose-m` | 17 | RTMPose-M body-only. |
| `mediapipe` | — | Delegates to `pose_estimation.main`. |

rtmlib models use [rtmlib](https://github.com/Tau-J/rtmlib) for lightweight ONNX/OpenVINO inference without `mmcv`/`mmpose`.

## Tracking modes

`--tracking {hands|hands-arms|body}` controls which body parts are tracked.

| Mode | Body keypoints | Hand keypoints | Pose detection |
|------|----------------|----------------|----------------|
| `hands` | — | 2 × 21 | Skipped (no arm-guided ROI fallback). |
| `hands-arms` (default) | 12 (shoulders → finger bases) | 2 × 21 | Yes. |
| `body` | 33 (face, torso, arms, legs) | 2 × 21 | Yes. |

In `hands` mode, hands are assigned left/right by wrist x-coordinate. In `hands-arms`/`body`, hands are matched to arms via Hungarian (optimal) assignment with a distality reject.

## Batch processing and single-subject

```bash
python -m pose_estimation.main --batch-dir videos/
python -m pose_estimation.main --batch-dir videos/ --single-subject     # keep largest body only
python -m pose_estimation.main --batch-dir videos/ --postprocess        # +Savitzky-Golay smoothing
python -m pose_estimation.main --batch-dir videos/ --postprocess --savgol-window 15 --savgol-polyorder 3
```

Single-subject mode is resilient in three layers: (1) keep the largest detected body each frame; (2) carry forward the last body for up to ~0.5 s when detection drops; (3) hand-only fallback (left/right by x-coordinate) when carry-forward expires.

Standalone post-processing:

```bash
python -m pose_estimation.postprocess output/video1.csv --window 15 --polyorder 3
```

## CSV output

One row per person per frame; normalised (0–1) landmark coordinates.

| Mode | Body columns | Hand columns | Metadata | Total |
|------|--------------|--------------|----------|-------|
| `hands` | — | 2 × 21 × 3 = 126 | 4 | 130 |
| `hands-arms` | 12 × 4 = 48 | 126 | 4 | 178 |
| `body` | 33 × 4 = 132 | 126 | 4 | 262 |

Body columns use prefix `arm_` in hands-arms mode and `body_` in body mode. Each body keypoint exports `x, y, z, visibility`; hand keypoints export `x, y, z` only. Missing hand data is blank; under `--single-subject`, body columns may also be blank on hand-only fallback frames.

## Headless mode & metrics

```bash
python -m pose_estimation.main --source video.mp4 --headless
python -m pose_estimation.main --source video.mp4 --headless --metrics-detail
```

`--headless` skips pygame and emits a `*_metrics.csv` (jitter, confidence, smoothing deltas, constraint corrections, etc.). `--metrics-detail` additionally writes a per-keypoint `*_kp_detail.csv` (large).

## Parameter benchmarking

```bash
python -m pose_estimation.benchmark --source video.mp4 --sweep body_min_cutoff 0.1 0.3 0.5 1.0
python -m pose_estimation.benchmark --source video.mp4 --config sweep_quick.yaml
python -m pose_estimation.benchmark --source video.mp4 --config sweep_default.yaml
```

Spawns headless subprocesses with `POSE_BENCH_*` env-var overrides. Sweep param list and YAML configs documented in `.claude/tech/optimization.md`.

## Analysis (R)

R scripts in `analysis/` consume the CSVs (`*_metrics.csv`, landmark, `*_clinical.csv`, etc.). Quick reference:

```bash
Rscript analysis/summary.R output/                         # text report + JSON
Rscript analysis/timeseries.R output/                       # temporal diagnostic plots
Rscript analysis/clinical_features.R output/                # kinematic features
Rscript analysis/clinical_correlation.R output/ clinical_scores.csv
Rscript analysis/longitudinal.R output/ sessions.csv         # recovery tracking
Rscript analysis/explore_clinical.R output/                  # exploratory plots
Rscript analysis/temporal_clinical.R output/                 # per-video feature timelines
Rscript analysis/compare_clinical.R output/                  # between-video comparison
Rscript analysis/clinical_dimreduce.R output/                # PCA + UMAP
Rscript analysis/make_templates.R output/                    # metadata templates
Rscript analysis/validate_metadata.R clinical_scores.csv output/
```

Full script-by-script documentation: `.claude/tech/analysis.md`. The bundled R Markdown report is `analysis/analysis_summary.Rmd` (renders to `analysis/analysis_summary.html`).

## Architecture

| File | Role |
|------|------|
| `main.py` | MediaPipe entry point; CLI, capture loop, pygame display. |
| `run.py` | Unified entry point with rtmlib backends. |
| `models.py` | Downloads MediaPipe TFLite, converts to OpenVINO IR, compiles. |
| `detection.py` | SSD anchor generation, NMS, detection decoding. |
| `processing.py` | Preprocessing, crop, landmark inference, hand↔arm matching. |
| `drawing.py` | Catmull-Rom splines, skeleton rendering, overlay blending. |
| `smoothing.py` | One Euro Filter, confidence-weighted temporal smoothing. |
| `constraints.py` | Bone-length consistency and joint-angle limits. |
| `export.py` | CSV schema, per-frame landmark row conversion. |
| `postprocess.py` | Savitzky-Golay offline smoothing. |
| `metrics.py` | Per-frame quality metrics collection. |
| `benchmark.py` | Parameter sweep harness (headless). |
| `_types.py` | TypedDicts: `Detection`, `HandDetectionDiag`, `PipelineState`. |
| `analysis/` | R scripts for metrics summarisation, visualisation, clinical features. |
| `scripts/benchmarks/` | Hot-path micro-benchmarks. |

## Technical notes

- **Display:** `pygame-ce` (SDL2) because OpenCV's bundled Qt backend does not render on Wayland.
- **Image processing:** `opencv-python-headless`.
- **Inference:** OpenVINO (NPU, CPU, GPU). rtmlib path supports ONNX Runtime or OpenVINO via `--backend`.
- **Hand↔arm matching:** Hungarian assignment via `scipy.optimize.linear_sum_assignment`; rejects hands that are closer to the shoulder midpoint than the wrist.
- **Frame pipeline:** capture (BGR) → flip → resize → detect → arm-guided hand ROI fallback → landmark → smooth → bone-length → joint-angle → match → optional single-subject filter → draw → RGB → pygame surface.

## Dependencies

Defined in `pyproject.toml`. Core: `openvino`, `opencv-python-headless`, `numpy`, `scipy`, `pygame-ce`, `tqdm`, `requests`, `rtmlib`, `pandas`, `pyyaml`. Dev groups: `ruff`, `ty`, `pytest`, `pytest-cov`.

## License

Apache-2.0 WITH LLVM-exception. See `LICENSE`.

# Pose Estimation

Real-time human pose estimation for movement analysis. It tracks the hands, the arms,
or the whole body from a webcam or a video file, and it exports per-frame landmark CSV
files. With a multi-camera calibration, it fuses the synchronized views into metric 3D.
The output feeds a clinical/rehabilitation kinematics pipeline (the R scripts under
`analysis/`).

Two inference paths share one pipeline:

- **MediaPipe** hand/pose TFLite models, converted to OpenVINO IR at runtime.
- **rtmlib** models (RTMW, DWPose, RTMPose) via ONNX Runtime or OpenVINO.

Both apply temporal smoothing, biomechanical constraints, and a skeleton overlay.

> **Working on this with an AI coding agent?** Start at [`CLAUDE.md`](CLAUDE.md).
> Task-specific internals live under [`docs/technical/`](docs/technical/), and active
> long-horizon state lives in [`.agent/roadmap.md`](.agent/roadmap.md).

## Requirements

- **Python 3.10+.** The tests cover 3.10 to 3.13.
- **[`uv`](https://docs.astral.sh/uv/)** for dependency and virtualenv management.
- An **OpenVINO device.** The NPU is the default, and it is optional. CPU and GPU also
  work (`--det-device CPU --pose-device CPU`), so you can contribute without special
  hardware.
- **R + [`renv`](https://rstudio.github.io/renv/)** for the `analysis/` scripts only.

## Quick start

```bash
uv sync                                          # runtime + dev tooling (tests, lint, types)
uv run python -m pose_estimation.run                # live webcam overlay; ESC or close window to quit
```

The first run downloads the models to `model/`. Git ignores the `videos/`, `output/`
and `model/` directories. They hold input recordings and derived data, which stay out
of commits.

## Entry points

`pyproject.toml` defines seven console scripts (`[project.scripts]`). You can also run
each one as `python -m pose_estimation.<module>`.

| Script | Module | Purpose |
|--------|--------|---------|
| `pose-estimation` | `pose_estimation.main` | MediaPipe pipeline (default). |
| `pose-estimation-run` | `pose_estimation.run` | Multi-backend: rtmlib (RTMW, DWPose, RTMPose) + MediaPipe. |
| `pose-estimation-benchmark` | `pose_estimation.benchmark` | Parameter sweep harness. |
| `pose-estimation-postprocess` | `pose_estimation.postprocess` | Savitzky-Golay smoothing for existing CSV files. |
| `pose-estimation-calibrate` | `pose_estimation.calibration_cli` | Multi-camera calibration (ChArUco). |
| `pose-estimation-inventory` | `pose_estimation.inventory` | Task-side family registry and aggregate corpus census. |
| `pose-estimation-sessions` | `pose_estimation.sessions` | Recording-event tree that multi-camera discovery reads. |
| `pose-estimation-validate` | `pose_estimation.validation` | Capture QA + end-to-end validation reports. |

### MediaPipe pipeline (`main`)

```bash
python -m pose_estimation.main                            # webcam 0, NPU, hands-arms
python -m pose_estimation.main --source video.mp4
python -m pose_estimation.main --source 1                 # camera index
python -m pose_estimation.main --batch-dir videos/        # process every video in a dir
python -m pose_estimation.main --batch-dir videos/ --single-subject --postprocess
python -m pose_estimation.main --pose-device CPU --no-flip  # CPU landmarks; rear camera (no mirror)
python -m pose_estimation.main --headless                 # no window; writes *_metrics.csv
```

`--headless` skips the pygame window and emits a `*_metrics.csv` (jitter, confidence,
smoothing deltas, constraint corrections). Add `--metrics-detail` for a large per-keypoint
`*_kp_detail.csv`.

### Multi-backend (`run`)

```bash
python -m pose_estimation.run                                  # webcam 0, rtmw-l
python -m pose_estimation.run --model dwpose-m                 # DWPose wholebody
python -m pose_estimation.run --model rtmpose-m               # body-only (17 kp)
python -m pose_estimation.run --model mediapipe               # delegates to main
python -m pose_estimation.run --source video.mp4 --det-device CPU --pose-device NPU
python -m pose_estimation.run --headless
```

| Model | Keypoints | Notes |
|-------|-----------|-------|
| `rtmw-l` (default) | 133 | RTMW-L wholebody (body + hands + face + feet). |
| `dwpose-m` | 133 | DWPose-M wholebody. |
| `rtmpose-m` | 17 | RTMPose-M body-only. |
| `mediapipe` | n/a | Delegates to `pose_estimation.main`. |

rtmlib models use [rtmlib](https://github.com/Tau-J/rtmlib) for lightweight ONNX/OpenVINO
inference without `mmcv`/`mmpose`.

### Multi-camera & calibration

A *session* is one recording from N synchronized cameras
(`videos/<session_id>/cam*.mp4`). If a camera calibration exists, the pipeline
triangulates the per-camera 2D keypoints into one fused 3D track (`world3d.csv`, in
metres).

```bash
# Calibrate a rig (ChArUco): print board → capture → solve → verify
pose-estimation-calibrate board   --output board.png
pose-estimation-calibrate capture --session-dir videos/calib/ --devices 0,1,2
pose-estimation-calibrate solve   --session-dir videos/calib/ --output calib.json
pose-estimation-calibrate verify  --calibration calib.json

# Process a session (both backends accept the session flags)
python -m pose_estimation.main --session-dir  videos/session_a/ --calibration calib.json
python -m pose_estimation.run  --sessions-dir videos/           --calibration calib.json
```

`--session-dir`, `--sessions-dir`, and `--calibration` are mutually exclusive with
`--source`/`--batch-dir`. [`docs/technical/multicam.md`](docs/technical/multicam.md) and
[`docs/technical/calibration.md`](docs/technical/calibration.md) document the session
layout, the manifest schema and the 3D-fusion model.

### Benchmarking & post-processing

```bash
# Parameter sweep (spawns headless subprocesses with POSE_BENCH_* overrides)
python -m pose_estimation.benchmark --source video.mp4 --sweep body_min_cutoff 0.1 0.3 0.5
python -m pose_estimation.benchmark --source video.mp4 --config sweep_default.yaml

# Savitzky-Golay smoothing of an existing CSV (also available as main's --postprocess)
python -m pose_estimation.postprocess output/video1.csv --window 15 --polyorder 3
```

Sweep parameters and YAML config format: [`docs/technical/optimization.md`](docs/technical/optimization.md).

## Tracking modes

`--tracking {hands|hands-arms|body}` selects the tracked body parts.

| Mode | Body keypoints | Hand keypoints | Pose detection |
|------|----------------|----------------|----------------|
| `hands` | 0 | 2 × 21 | Skipped (no arm-guided ROI fallback). |
| `hands-arms` (default) | 12 (shoulders → finger bases) | 2 × 21 | Yes. |
| `body` | 33 (face, torso, arms, legs) | 2 × 21 | Yes. |

In `hands` mode, confident model handedness assigns the anatomical left and right sides.
The wrist x-coordinate is the ambiguity fallback. In `hands-arms` and `body` modes, a
Hungarian (optimal) assignment matches the hands to the arms, with a distality reject.

Single-subject mode (`--single-subject`) tracks one subject through three fallback
layers:

1. It keeps the largest detected body in each frame.
2. If detection drops, it carries the last body forward for approximately 0.5 s.
3. When the carry-forward expires, it falls back to the hands alone. Model handedness
   assigns the sides, with the x-coordinate as the ambiguity fallback.

## CSV output

One row per person per frame; normalised (0-1) landmark coordinates.

| Mode | Body columns | Hand columns | Metadata | Total |
|------|--------------|--------------|----------|-------|
| `hands` | 0 | 2 × 21 × 4 = 168 | 4 | 172 |
| `hands-arms` | 12 × 4 = 48 | 168 | 4 | 220 |
| `body` | 33 × 4 = 132 | 168 | 4 | 304 |

Body columns use the prefix `arm_` in hands-arms mode, and `body_` in body mode. Each
body keypoint exports `x, y, z, visibility`, and each hand keypoint exports `x, y, z,
confidence`. Missing hand coordinates stay blank, with confidence zero. Under
`--single-subject`, the body columns stay blank on hand-only fallback frames.
The multi-camera path also writes `world3d.csv` (metric 3D + per-keypoint
fusion diagnostics). Legacy three-column hand CSV files stay readable, with
coordinate-presence confidence.

## Analysis (R)

The R scripts in `analysis/` read the CSV files, and produce diagnostics and clinical
kinematic features. The most common entry points follow:

```bash
Rscript analysis/summary.R output/                          # text report + JSON
Rscript analysis/timeseries.R output/                       # temporal diagnostic plots
Rscript analysis/clinical_features.R output/                # kinematic feature extraction
Rscript analysis/longitudinal.R output/ sessions.csv        # recovery tracking
```

The full script-by-script reference is [`docs/technical/analysis.md`](docs/technical/analysis.md).
The bundled report `analysis/analysis_summary.Rmd` renders to `analysis_summary.html`.

## Project layout

Source lives in `src/pose_estimation/`. The single-camera 2D pipeline:

| Module | Role |
|--------|------|
| `main.py` | MediaPipe entry point; CLI, capture loop, pygame display. |
| `run.py` | Unified entry point with rtmlib backends. |
| `models.py` | Downloads MediaPipe TFLite, converts to OpenVINO IR, compiles. |
| `detection.py` | SSD anchor generation, NMS, detection decoding. |
| `processing.py` | Preprocessing, crop, landmark inference, hand↔arm matching. |
| `mapping.py` | COCO-WholeBody (rtmlib 133/17-kp) → MediaPipe keypoint translation. |
| `smoothing.py` | One Euro Filter, confidence-weighted temporal smoothing (MediaPipe). |
| `rtmlib_smoothing.py` | `KeypointSmoother`: smoothing + carry-forward + person matching (rtmlib). |
| `rtmlib_openvino.py` | Monkeypatch making rtmlib run on the OpenVINO backend. |
| `constraints.py` | Bone-length consistency and joint-angle limits. |
| `drawing.py` | Catmull-Rom splines, skeleton rendering, overlay blending. |
| `export.py` | CSV schema, per-frame landmark row conversion. |
| `metrics.py` | Per-frame quality metrics collection. |
| `postprocess.py` | Savitzky-Golay offline smoothing. |
| `benchmark.py` | Parameter sweep harness (headless). |
| `video_io.py` | Capture open, FPS clamp, frame→pygame surface, file discovery. |
| `_types.py` | TypedDicts documenting dict-passed pipeline state. |

The multi-camera 3D subsystem:

| Module | Role |
|--------|------|
| `multicam.py` | `Session` discovery/sync, `process_session` orchestration, output fusion. |
| `calibration.py` | Camera/session calibration IO + validation (cv2-free). |
| `charuco.py` | ChArUco board construction/rendering and the `solve_charuco` solver. |
| `triangulation.py` | DLT helpers and `fuse_session_frame` (weighted DLT + outlier rejection). |
| `calibration_cli.py` | `pose-estimation-calibrate` console script. |

Non-pipeline directories: `analysis/` (R scripts), `scripts/benchmarks/` (hot-path
micro-benchmarks), `tests/` (pytest suite).

## Development

`uv sync` installs the full dev toolchain (the default `dev` group bundles tests, lint,
and types) alongside the runtime dependencies.

| Task | Command |
|------|---------|
| Run tests | `uv run pytest` |
| Tests + coverage | `uv run pytest --cov=pose_estimation` |
| Lint (autofix) | `uv run ruff check --fix` |
| Format | `uv run ruff format` |
| Type-check | `uv run ty check` |

Learn these conventions before you open a pull request:

- **Strict tests.** Warnings are errors (`filterwarnings = ["error", …]`). For new
  behaviour, write the failing test first (red-green-refactor).
- **Public API guard.** The package surface is whatever `src/pose_estimation/__init__.py`
  re-exports. If that surface drifts, `tests/test_public_api.py` fails, so update both
  files together.
- **Commit style.** [Scoped Commits](https://scopedcommits.com/):
  `<scope>: <imperative subject>` (≤50 chars), where scope is a subsystem (`tracking`,
  `calibration`, `multicam`) or a cross-cutting label (`Tooling`, `Docs`, `Refactor`).

Deeper technical reference (architecture, tracking modes, multicam, calibration, analysis,
optimization, tests, environment) lives under [`docs/technical/`](docs/technical/).

## Technical notes

- **Display:** `pygame-ce` (SDL2), because the Qt backend bundled with OpenCV does not
  render on Wayland. Image processing uses `opencv-python-headless`.
- **Inference:** OpenVINO (NPU, CPU, GPU). `main` converts MediaPipe TFLite to IR. The
  rtmlib path (`run`) supports ONNX Runtime or OpenVINO through `--backend`.
- **Detector and pose model take separate devices** (`--det-device` / `--pose-device`),
  because their device compatibility differs. The YOLOX detector in rtmlib exports NMS
  into the graph, so its output shape is dynamic. A static-shape-only device such as the
  NPU returns a fixed-size buffer, and the unused rows hold uninitialised scores. Every
  frame then saturates at the padded row count, with scores outside `[0, 1]`. Therefore
  `run` defaults to `--det-device CPU --pose-device NPU`. The pose models are
  static-shaped. They agree with CPU to approximately 0.5 px median, and they run
  approximately 19× faster on the NPU. MediaPipe decodes the anchors and runs NMS in
  Python, so both of its roles default to the NPU.
- **Single cv2 wheel:** `[tool.uv] override-dependencies` excludes the
  `opencv-python`/`opencv-contrib-python` requirements of rtmlib, so the environment
  holds `opencv-python-headless` alone. All cv2 wheels unpack the same tree, and they
  would otherwise file-stomp.
- **Frame pipeline:** capture (BGR) → flip → resize → detect → arm-guided hand ROI
  fallback → landmark → smooth → bone-length → joint-angle → match → optional
  single-subject filter → draw → RGB → pygame surface.

`pyproject.toml` declares the core dependencies. `uv.lock` (Python) and `renv.lock` (R)
pin the exact versions.

## License

Apache-2.0 WITH LLVM-exception. See [`LICENSE`](LICENSE).

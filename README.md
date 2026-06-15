# Pose Estimation

Real-time human pose estimation for movement analysis. It tracks hands, arms, or the
whole body from a webcam or video, exports per-frame landmark CSVs, and — with
multi-camera calibration — fuses synchronized views into metric 3D. The output feeds a
clinical/rehabilitation kinematics pipeline (the R scripts under `analysis/`).

Two inference paths share one pipeline:

- **MediaPipe** hand/pose TFLite models, converted to OpenVINO IR at runtime.
- **rtmlib** models (RTMW, DWPose, RTMPose) via ONNX Runtime or OpenVINO.

Both apply temporal smoothing, biomechanical constraints, and a skeleton overlay.

> **Working on this with an AI coding agent?** Start at [`CLAUDE.md`](CLAUDE.md) and
> [`.claude/INDEX.md`](.claude/INDEX.md) instead — they index the agent-oriented technical
> notes under `.claude/tech/`. The rest of this README targets human contributors.

## Requirements

- **Python 3.10+** (3.10–3.13 are tested).
- **[`uv`](https://docs.astral.sh/uv/)** for dependency and virtualenv management.
- An **OpenVINO device.** NPU is the default but optional — CPU and GPU also work
  (`--device CPU`), so no special hardware is needed to contribute.
- **R + [`renv`](https://rstudio.github.io/renv/)** — only for the `analysis/` scripts.

## Quick start

```bash
uv sync                                          # runtime + dev tooling (tests, lint, types)
uv run python -m pose_estimation.run --device CPU   # live webcam overlay; ESC or close window to quit
```

Models download to `model/` on first run. The `videos/`, `output/`, and `model/`
directories are git-ignored — they hold input recordings and derived data, which stay out
of commits.

## Entry points

Five console scripts are defined in `pyproject.toml` (`[project.scripts]`); each is also
runnable as `python -m pose_estimation.<module>`.

| Script | Module | Purpose |
|--------|--------|---------|
| `pose-estimation` | `pose_estimation.main` | MediaPipe pipeline (default). |
| `pose-estimation-run` | `pose_estimation.run` | Multi-backend: rtmlib (RTMW, DWPose, RTMPose) + MediaPipe. |
| `pose-estimation-benchmark` | `pose_estimation.benchmark` | Parameter sweep harness. |
| `pose-estimation-postprocess` | `pose_estimation.postprocess` | Savitzky-Golay smoothing for existing CSVs. |
| `pose-estimation-calibrate` | `pose_estimation.calibration_cli` | Multi-camera calibration (ChArUco). |

### MediaPipe pipeline (`main`)

```bash
python -m pose_estimation.main                            # webcam 0, NPU, hands-arms
python -m pose_estimation.main --source video.mp4
python -m pose_estimation.main --source 1                 # camera index
python -m pose_estimation.main --batch-dir videos/        # process every video in a dir
python -m pose_estimation.main --batch-dir videos/ --single-subject --postprocess
python -m pose_estimation.main --device CPU --no-flip     # CPU; rear camera (no mirror)
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
python -m pose_estimation.run --source video.mp4 --backend openvino --device NPU
python -m pose_estimation.run --headless
```

| Model | Keypoints | Notes |
|-------|-----------|-------|
| `rtmw-l` (default) | 133 | RTMW-L wholebody (body + hands + face + feet). |
| `dwpose-m` | 133 | DWPose-M wholebody. |
| `rtmpose-m` | 17 | RTMPose-M body-only. |
| `mediapipe` | — | Delegates to `pose_estimation.main`. |

rtmlib models use [rtmlib](https://github.com/Tau-J/rtmlib) for lightweight ONNX/OpenVINO
inference without `mmcv`/`mmpose`.

### Multi-camera & calibration

A *session* is one recording from N synchronized cameras
(`videos/<session_id>/cam*.mp4`). With camera calibration present, per-camera 2D keypoints
are triangulated into a fused 3D track (`world3d.csv`, in metres).

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
`--source`/`--batch-dir`. Session layout, manifest schema, and the 3D-fusion model are
documented in [`.claude/tech/multicam.md`](.claude/tech/multicam.md) and
[`.claude/tech/calibration.md`](.claude/tech/calibration.md).

### Benchmarking & post-processing

```bash
# Parameter sweep (spawns headless subprocesses with POSE_BENCH_* overrides)
python -m pose_estimation.benchmark --source video.mp4 --sweep body_min_cutoff 0.1 0.3 0.5
python -m pose_estimation.benchmark --source video.mp4 --config sweep_default.yaml

# Savitzky-Golay smoothing of an existing CSV (also available as main's --postprocess)
python -m pose_estimation.postprocess output/video1.csv --window 15 --polyorder 3
```

Sweep parameters and YAML config format: [`.claude/tech/optimization.md`](.claude/tech/optimization.md).

## Tracking modes

`--tracking {hands|hands-arms|body}` selects which body parts are tracked.

| Mode | Body keypoints | Hand keypoints | Pose detection |
|------|----------------|----------------|----------------|
| `hands` | — | 2 × 21 | Skipped (no arm-guided ROI fallback). |
| `hands-arms` (default) | 12 (shoulders → finger bases) | 2 × 21 | Yes. |
| `body` | 33 (face, torso, arms, legs) | 2 × 21 | Yes. |

In `hands` mode, hands are assigned left/right by wrist x-coordinate. In
`hands-arms`/`body`, hands are matched to arms via Hungarian (optimal) assignment with a
distality reject.

Single-subject mode (`--single-subject`) is resilient in three layers: (1) keep the
largest detected body each frame; (2) carry forward the last body for up to ~0.5 s when
detection drops; (3) hand-only fallback (left/right by x-coordinate) once carry-forward
expires.

## CSV output

One row per person per frame; normalised (0–1) landmark coordinates.

| Mode | Body columns | Hand columns | Metadata | Total |
|------|--------------|--------------|----------|-------|
| `hands` | — | 2 × 21 × 3 = 126 | 4 | 130 |
| `hands-arms` | 12 × 4 = 48 | 126 | 4 | 178 |
| `body` | 33 × 4 = 132 | 126 | 4 | 262 |

Body columns use prefix `arm_` in hands-arms mode and `body_` in body mode. Each body
keypoint exports `x, y, z, visibility`; hand keypoints export `x, y, z` only. Missing hand
data is blank; under `--single-subject`, body columns may also be blank on hand-only
fallback frames. The multi-camera path additionally writes `world3d.csv` (metric 3D +
per-keypoint fusion diagnostics).

## Analysis (R)

The R scripts in `analysis/` consume the CSVs to produce diagnostics and clinical
kinematic features. A few common entry points:

```bash
Rscript analysis/summary.R output/                          # text report + JSON
Rscript analysis/timeseries.R output/                       # temporal diagnostic plots
Rscript analysis/clinical_features.R output/                # kinematic feature extraction
Rscript analysis/longitudinal.R output/ sessions.csv        # recovery tracking
```

The full script-by-script reference is [`.claude/tech/analysis.md`](.claude/tech/analysis.md).
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

A few conventions worth knowing before you open a PR:

- **Strict tests.** Warnings are errors (`filterwarnings = ["error", …]`). For new
  behaviour, prefer writing the failing test first (red-green-refactor).
- **Public API guard.** The package surface is whatever `src/pose_estimation/__init__.py`
  re-exports; `tests/test_public_api.py` fails if it drifts — update both together.
- **Repo-map guard.** `.claude/repomap.md` is an autogenerated symbol index;
  `tests/test_repomap.py` fails when it is stale. Regenerate with
  `python scripts/repomap.py` after adding, moving, or renaming a symbol.
- **Commit style.** [Scoped Commits](https://scopedcommits.com/):
  `<scope>: <imperative subject>` (≤50 chars), where scope is a subsystem (`tracking`,
  `calibration`, `multicam`) or a cross-cutting label (`Tooling`, `Docs`, `Refactor`).

Deeper technical reference (architecture, tracking modes, multicam, calibration, analysis,
optimization, tests, environment) lives under `.claude/tech/`, indexed by
[`.claude/INDEX.md`](.claude/INDEX.md). Those notes are written for AI agents but read fine
for humans.

## Technical notes

- **Display:** `pygame-ce` (SDL2), because OpenCV's bundled Qt backend does not render on
  Wayland. Image processing uses `opencv-python-headless`.
- **Inference:** OpenVINO (NPU, CPU, GPU). `main` converts MediaPipe TFLite to IR; the
  rtmlib path (`run`) supports ONNX Runtime or OpenVINO via `--backend`.
- **Single cv2 wheel:** `[tool.uv] override-dependencies` excludes rtmlib's
  `opencv-python`/`opencv-contrib-python` so only `opencv-python-headless` is installed
  (all cv2 wheels unpack the same tree and would otherwise file-stomp).
- **Frame pipeline:** capture (BGR) → flip → resize → detect → arm-guided hand ROI
  fallback → landmark → smooth → bone-length → joint-angle → match → optional
  single-subject filter → draw → RGB → pygame surface.

Core dependencies are declared in `pyproject.toml`; exact versions are pinned in `uv.lock`
(Python) and `renv.lock` (R).

## License

Apache-2.0 WITH LLVM-exception. See [`LICENSE`](LICENSE).

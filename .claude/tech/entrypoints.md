# Entry points

Six console scripts (see `pyproject.toml:[project.scripts]`):

| Script | Module | Purpose |
|--------|--------|---------|
| `pose-estimation` | `pose_estimation.main` | MediaPipe pipeline (default). |
| `pose-estimation-run` | `pose_estimation.run` | Multi-backend (rtmlib + MediaPipe). |
| `pose-estimation-benchmark` | `pose_estimation.benchmark` | Parameter sweep harness. |
| `pose-estimation-postprocess` | `pose_estimation.postprocess` | Savitzky-Golay smoothing on existing CSVs. |
| `pose-estimation-calibrate` | `pose_estimation.calibration_cli` | Multi-camera calibration management. |
| `pose-estimation-validate` | `pose_estimation.validation` | End-to-end pipeline validation report. |

## `main.py` — MediaPipe path

```bash
python -m pose_estimation.main                            # webcam 0, NPU, hands-arms
python -m pose_estimation.main --source video.mp4
python -m pose_estimation.main --batch-dir videos/        # batch over a directory
python -m pose_estimation.main --headless                 # no pygame, emit metrics CSV
python -m pose_estimation.main --metrics-detail           # adds *_kp_detail.csv
python -m pose_estimation.main --single-subject           # keep largest body only
python -m pose_estimation.main --postprocess              # +Savitzky-Golay smoothing pass
python -m pose_estimation.main --device CPU               # CPU instead of NPU
python -m pose_estimation.main --no-flip                  # disable mirror flip
python -m pose_estimation.main --tracking hands|hands-arms|body
```

Key flags: `--source`, `--batch-dir`, `--session-dir`, `--sessions-dir`, `--calibration`, `--output-dir`, `--device`, `--model-dir`, `--tracking`, `--single-subject`, `--headless`, `--metrics-detail`, `--postprocess`, `--savgol-window`, `--savgol-polyorder`, `--no-flip`.

Multi-camera flags (`--session-dir`, `--sessions-dir`, `--calibration`) are mutually exclusive with `--source`/`--batch-dir`. They resolve a `Session` (per `tech/multicam.md`) and call `process_session(...)` with a MediaPipe camera processor callback that wraps `process_video()`. Per-camera CSVs are written to `<output-dir>/<session_id>/camN.csv`.

## `run.py` — unified entry point (rtmlib + MediaPipe)

`MODEL_REGISTRY` (in `run.py`):

| Model key | Keypoints | Notes |
|-----------|-----------|-------|
| `rtmw-l` (default) | 133 | RTMW-L wholebody (body + hands + face + feet). |
| `dwpose-m` | 133 | DWPose-M wholebody. |
| `rtmpose-m` | 17 | RTMPose-M body-only. |
| `mediapipe` | — | Delegates to `main.py`. |

```bash
python -m pose_estimation.run                                          # webcam 0, rtmw-l
python -m pose_estimation.run --model dwpose-m
python -m pose_estimation.run --source video.mp4 --backend openvino --device NPU
python -m pose_estimation.run --batch-dir videos/ --single-subject
python -m pose_estimation.run --session-dir videos/session_a/           # multi-cam (stub)
python -m pose_estimation.run --sessions-dir videos/ --calibration calib.json
python -m pose_estimation.run --list-sessions                          # read-only discovery probe
python -m pose_estimation.run --headless                               # no display
```

All rtmlib models share the YOLOX-m detector (640×640). Detector + pose URLs are pinned in `MODEL_REGISTRY`. Models download on first run.

`--session-dir`/`--sessions-dir`/`--calibration` route through the same multi-camera dispatcher as `main.py`, using an rtmlib camera processor callback that wraps `process_source()`. Session dispatch occurs after model setup so the pose tracker, smoother, and bone smoother are available. With `--model mediapipe`, `_run_mediapipe` forwards these flags to `pose-estimation` via subprocess. `--list-sessions` short-circuits *before* model setup: it resolves `--session-dir`/`--sessions-dir`/`--calibration` (sessions root defaults to `videos/`) through `resolve_cli_sessions(..., summary_label="Discovered sessions", redact_identifiers=True)` — filesystem + `session.json`/`calibration.json` discovery, no frame decoding, no dispatch — prints `session #<i>: N cameras; calibration: present|absent` per session, then exits (`0` = ≥1 found, `1` = none/error). Read-only probe backing the roadmap M2 footage gate; `redact_identifiers` surfaces only an ordinal + camera count + calibration presence, keeping the deny-listed tree's session ids / camera names (and all frame + calibration values) out of context.

## `benchmark.py` — parameter sweep

```bash
python -m pose_estimation.benchmark --source video.mp4 --sweep body_min_cutoff 0.1 0.3 0.5 1.0
python -m pose_estimation.benchmark --source video.mp4 --config sweep_default.yaml
python -m pose_estimation.benchmark --source video.mp4 --config sweep_quick.yaml
```

Spawns headless subprocesses with `POSE_BENCH_*` env-var overrides. See `tech/optimization.md` for the parameter list.

## `calibration_cli.py` — multi-camera calibration

```bash
pose-estimation-calibrate verify  --calibration calib.json
pose-estimation-calibrate solve   --session-dir videos/calib_session/ --output calib.json
pose-estimation-calibrate board   --output board.png
pose-estimation-calibrate capture --session-dir videos/calib_session/ --devices 0,1,2
```

`verify` prints a summary; `solve` runs the ChArUco solver (`charuco.py`); `board` renders the printable pattern; `capture` records synchronized per-camera AVIs via a pygame grid (SPACE = save one frame per camera). Full flags + workflow: `tech/calibration.md`.

## `validation.py` — end-to-end validation report

```bash
pose-estimation-validate --session-dir videos/session_a --calibration calib.json \
    --out report.json --markdown report.md
pose-estimation-validate --session-dir videos/session_a --baseline ref.json
```

Runs the full 3D clinical chain (calibration → 2D tracking → `world3d.csv` fusion → R clinical metrics) on one session and emits a structured `ValidationReport` (JSON for CI, Markdown for humans). Flags: `--session-dir` (required), `--calibration` (file | dir | omit to reuse `<session>/calibration.json`), `--baseline`, `--device`, `--backend`, `--out`, `--markdown`. Exit `2` on `ValidationError` (e.g. no calibration). Raw numbers only — the PASS/WARN/FAIL verdict lands in roadmap Session 1B. Full schema + surrogates: `tech/validation.md`.

## `postprocess.py` — offline Savitzky-Golay

```bash
python -m pose_estimation.postprocess output/video1.csv --window 15 --polyorder 3
```

Writes `<stem>_smooth.csv` next to the input. Also exposed as the `--postprocess` flag on `main.py`.

## `scripts/benchmarks/run.py` — micro-benchmarks (separate from sweep)

```bash
uv run python scripts/benchmarks/run.py                # full suite
uv run python scripts/benchmarks/run.py smoothing      # single group
uv run python scripts/benchmarks/run.py --quick        # fewer iterations
```

Groups: `smoothing`, `constraints`, `matching`, `detection`, `processing`, `drawing`, `metrics`. See `tech/optimization.md`.

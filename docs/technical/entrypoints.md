# Command-line entry points

Ten console scripts (see `pyproject.toml:[project.scripts]`):

| Script | Module | Purpose |
|--------|--------|---------|
| `pose-estimation` | `pose_estimation.main` | MediaPipe pipeline (default). |
| `pose-estimation-run` | `pose_estimation.run` | Multi-backend (rtmlib + MediaPipe). |
| `pose-estimation-benchmark` | `pose_estimation.benchmark` | Parameter sweep harness. |
| `pose-estimation-postprocess` | `pose_estimation.postprocess` | Savitzky-Golay smoothing on existing CSVs. |
| `pose-estimation-calibrate` | `pose_estimation.calibration_cli` | Multi-camera calibration management. |
| `pose-estimation-inventory` | `pose_estimation.inventory` | Task-side family registry and aggregate container census. |
| `pose-estimation-sessions` | `pose_estimation.sessions` | Recording-event tree that multi-camera discovery reads. |
| `pose-estimation-qualify` | `pose_estimation.qualify` | Capture-qualification evidence publisher. |
| `pose-estimation-calibration-qc` | `pose_estimation.calibration_qc` | Corpus-level calibration ruling and evidence publisher. |
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
python -m pose_estimation.main --det-device CPU --pose-device CPU   # CPU instead of NPU
python -m pose_estimation.main --no-flip                  # disable mirror flip
python -m pose_estimation.main --tracking hands|hands-arms|body
```

Key flags: `--source`, `--batch-dir`, `--session-dir`, `--sessions-dir`, `--calibration`, `--output-dir`, `--det-device`, `--pose-device`, `--model-dir`, `--tracking`, `--single-subject`, `--headless`, `--metrics-detail`, `--postprocess`, `--savgol-window`, `--savgol-polyorder`, `--no-flip`.

Multi-camera flags (`--session-dir`, `--sessions-dir`, `--calibration`) are mutually exclusive with `--source`/`--batch-dir`. They resolve a `Session` (per `multicam.md`) and call `process_session(...)` with a MediaPipe camera processor callback that wraps `process_video()`. Per-camera CSVs are written to `<output-dir>/<session_id>/camN.csv`.

Both backends export zero-based decoded-source `frame_idx` values and source
presentation timestamps. A malformed decoded frame therefore leaves an index
gap instead of shifting every later row relative to the other cameras.

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
python -m pose_estimation.run --source video.mp4 --det-device CPU --pose-device NPU
python -m pose_estimation.run --batch-dir videos/ --single-subject
python -m pose_estimation.run --session-dir videos/session_a/           # multi-camera
python -m pose_estimation.run --sessions-dir videos/ --calibration calib.json
python -m pose_estimation.run --list-sessions                          # read-only discovery probe
python -m pose_estimation.run --headless                               # no display
```

All rtmlib models share the YOLOX-m detector (640×640). Detector + pose URLs are pinned in `MODEL_REGISTRY`. Models download on first run.

`--session-dir`/`--sessions-dir`/`--calibration` route through the same multi-camera dispatcher as `main.py`, using an rtmlib camera processor callback that wraps `process_source()`. Session dispatch occurs after model setup so the pose tracker, smoother, and bone smoother are available. With `--model mediapipe`, `_run_mediapipe` forwards these flags to `pose-estimation` via subprocess. `--list-sessions` short-circuits *before* model setup: it resolves `--session-dir`/`--sessions-dir`/`--calibration` (sessions root defaults to `sessions/`, the tree `pose-estimation-sessions` publishes — the old `videos/` default named a non-recursive raw-media root that never held a session directory) through `resolve_cli_sessions(..., summary_label="Discovered sessions", redact_identifiers=True)` — filesystem + `session.json`/`calibration.json` discovery, no frame decoding, no dispatch — prints `session #<i>: N cameras; calibration: present|absent` per session, then exits (`0` = ≥1 found, `1` = none/error). Read-only probe backing the roadmap M2 footage gate; `redact_identifiers` surfaces only an ordinal + camera count + calibration presence, keeping the deny-listed tree's session ids / camera names (and all frame + calibration values) out of context.

## `benchmark.py` — parameter sweep

```bash
python -m pose_estimation.benchmark --source video.mp4 --sweep body_min_cutoff 0.1 0.3 0.5 1.0
python -m pose_estimation.benchmark --source video.mp4 --config sweep_default.yaml
python -m pose_estimation.benchmark --source video.mp4 --config sweep_quick.yaml
```

Spawns headless subprocesses with `POSE_BENCH_*` env-var overrides. See `optimization.md` for the parameter list.

## `calibration_cli.py` — multi-camera calibration

```bash
pose-estimation-calibrate verify  --calibration calib.json
pose-estimation-calibrate solve   --session-dir videos/calib_session/ --output calib.json
pose-estimation-calibrate board   --output board.png
pose-estimation-calibrate capture --session-dir videos/calib_session/ --devices 0,1,2
```

`verify` prints a summary; `solve` runs the ChArUco solver (`charuco.py`); `board` renders the printable pattern; `capture` records synchronized per-camera AVIs via a pygame grid (SPACE = save one frame per camera). Full flags + workflow: `calibration.md`.

## `inventory.py` — task-side family registry and census

```bash
pose-estimation-inventory --corpus synthetic-corpus --out inventory
pose-estimation-inventory --corpus synthetic-corpus --out inventory --no-checksums
pose-estimation-inventory --corpus synthetic-corpus --out inventory --strict
python -m pose_estimation.inventory --corpus synthetic-corpus --out inventory
```

`--corpus` selects the required directory and searches every subdirectory.
`--out` selects the artifact directory and defaults to `inventory`.
The output directory must resolve outside the corpus.
`--no-checksums` skips each full-file SHA-256 scan, but it keeps header probing.
`--strict` returns status 1 when at least one asset is not canonical.

The default run probes headers and reads every source byte for fixity.
The tool never calls `VideoCapture.read` or `VideoCapture.grab`.
`--out` follows the operating umask because the tool sets no file mode.
`assets.csv` carries corpus-relative paths, so the output is as sensitive as the corpus.

The module defaults `OPENCV_FFMPEG_LOGLEVEL` to `-8` before the first probe.
If you need native FFmpeg diagnostics, set the variable before the command.
Native FFmpeg output bypasses Python and can contain the source URL.

Exit status 0 means that the run completed.
Exit status 1 means that `--strict` found at least one non-canonical asset.
Exit status 2 means that a usage, domain, or registry I/O error occurred.
Argparse usage errors keep argparse's text.

Handled domain and registry I/O errors start with `ERROR:` on stderr.
The common operator messages are path-free:

- `ERROR: The corpus path is not a directory.`
- `ERROR: The output directory must sit outside the corpus.`
- `ERROR: A directory under the corpus cannot be read. Correct its permissions.`
- `ERROR: The registry could not be written. Check the output directory.`

The console summarizes files, bytes, dispositions, reasons, family coverage, header totals, and probe metadata.
Its `Captures:` label refers to task-side families, not physical takes.
It ends with `Wrote: assets.csv, captures.csv, census.json`.
No success or handled-error line contains a filesystem path.

The tool writes `assets.csv`, `captures.csv`, and `census.json`.
Every consumer must call `validate_generation(out_dir)` before reading a row.
See `inventory.md` for identities, schemas, claim limits, and generation validation.

## `sessions.py` — recording-event tree

```bash
pose-estimation-sessions --inventory inventory --corpus videos/3-cam --out sessions
pose-estimation-sessions --strict
python -m pose_estimation.sessions --out sessions
```

`--inventory` selects the published registry and defaults to `inventory`.
`--corpus` selects the root that the registry's relative paths resolve against, and defaults to `videos/3-cam`.
`--out` selects the tree directory and defaults to `sessions`.
`--strict` returns status 1 when the tool holds any asset out.
Status 2 reports a usage or registry error.

The output directory must not contain, equal, or sit inside `--corpus` or `--inventory`.
The tool reads the registry alone and never walks the corpus. See `sessions.md`.

## `qualify.py` — capture-qualification evidence publisher

```bash
pose-estimation-qualify \
  --inventory inventory \
  --sessions sessions \
  --corpus videos/3-cam \
  --out qualification \
  --measurements measurements

python -m pose_estimation.qualify \
  --inventory inventory \
  --sessions sessions \
  --corpus videos/3-cam \
  --out qualification
```

`--inventory`, `--sessions`, `--corpus`, and `--out` are required.
`--measurements` is optional. Omit it to publish the expensive axes unmeasured.

The tool validates each supplied generation before it reads a row.
It publishes `assets_qc.csv`, `pairs_qc.csv`, `cameras_qc.csv`, `events_qc.csv`, and `qualification.json`.
The output must not equal, contain, or sit inside any input.

Help and successful publication exit 0.
Argparse usage errors exit 2 before dispatch.
A handled qualification, session, or inventory error prints one `Error:` message and exits 2.

Every consumer must call `qualify.validate_generation` before reading a row.
See [Capture qualification](qualification.md) for the schemas, measurement limits, and consumer contract.

## `calibration_qc.py` — corpus-level calibration ruling

```bash
pose-estimation-calibration-qc \
  --qualification qualification \
  --evidence evidence \
  --probes scripts \
  --out calibration_qc \
  --sessions sessions \
  --inventory inventory

python -m pose_estimation.calibration_qc \
  --qualification qualification \
  --evidence evidence \
  --probes scripts \
  --out calibration_qc
```

`--qualification`, `--evidence`, `--probes`, and `--out` are required.
`--sessions` and `--inventory` are optional upstream freshness checks.
Pass both when those trees remain available.

The tool validates captured probe stdout and publishes one fixed corpus ruling.
It runs no probe, computes no statistic, and never modifies `qualification/`.
It publishes `corpus_qc.csv`, `evidence_qc.csv`, and `calibration_qc.json`.

Help and successful publication exit 0.
Argparse usage errors exit 2 before dispatch.
A handled calibration-QC, qualification, session, or inventory error prints one `Error:` message and exits 2.
Other exceptions propagate.

Every consumer must call `calibration_qc.validate_generation` before reading a row.
Pass `probes_dir=` to catch a cited probe script that changed after publication.
See [Calibration ruling](calibration_qc.md) for the schemas, claim bound, and consumer contract.

## `validation.py` — end-to-end validation report

```bash
pose-estimation-validate --session-dir videos/session_a --calibration calib.json \
    --out report.json --markdown report.md
pose-estimation-validate --session-dir videos/session_a --baseline ref.json
```

Runs the full 3D clinical chain (calibration → 2D tracking → `world3d.csv` fusion → R clinical metrics) on one session and emits JSON + Markdown with a PASS/WARN/FAIL verdict. `--qa-only` runs pre-flight capture QA without fusion/clinical analysis; `--strict` makes WARN fail. Exit codes: 0 = accepted verdict, 1 = FAIL (or WARN under `--strict`), 2 = harness/input error. Full flags, schemas, thresholds, and surrogates: `validation.md`.

## `postprocess.py` — offline Savitzky-Golay

```bash
python -m pose_estimation.postprocess output/video1.csv --window 15 --polyorder 3
```

Writes `<stem>_smooth.csv` next to the input. Also exposed as the `--postprocess` flag on `main.py`. Before interpolation/filtering, current-schema coordinates are treated as missing wherever their matching `_vis` or `_conf` value is blank, nonfinite, or zero; this prevents held predictions from pulling adjacent observations. Evidence columns are copied unchanged, and legacy coordinates without a matching evidence column retain the original smoothing behavior.

## `scripts/benchmarks/run.py` — micro-benchmarks (separate from sweep)

```bash
uv run python scripts/benchmarks/run.py                # full suite
uv run python scripts/benchmarks/run.py smoothing      # single group
uv run python scripts/benchmarks/run.py --quick        # fewer iterations
```

Groups: `smoothing`, `constraints`, `matching`, `detection`, `processing`, `drawing`, `metrics`. See `optimization.md`.

# Multi-camera sessions

A *session* is a single recording with N synchronized cameras. The codebase treats N as variable; the deployed setup uses N=3.

## Directory layout

```
videos/<session_id>/
├── cam1.mp4            # discovered via glob: cam*.{mp4,avi,mov,mkv,webm}
├── cam2.mp4
├── cam3.mp4
├── session.json        # optional manifest (see schema below)
└── calibration.json    # optional; --calibration <path> overrides this
```

```
output/<session_id>/
├── cam1.csv            # per-camera keypoint CSV (existing schema)
├── cam1_diag.csv       # per-camera diagnostics
├── cam2.csv
├── cam3.csv
├── world3d.csv         # FUTURE: triangulated 3D keypoints (not yet wired)
└── world3d_diag.csv    # FUTURE
```

Per-camera CSV columns are unchanged from the single-source schema (`tech/tracking-modes.md`). The world3d schema lands when `triangulation.fuse_session_frame()` is wired.

## `session.json` manifest (optional)

```json
{
  "format_version": 1,
  "session_id": "session_2026-05-20_subject01",
  "cameras": [
    { "name": "cam1", "file": "cam1.mp4", "sync_offset": 0 },
    { "name": "cam2", "file": "cam2.mp4", "sync_offset": 2 },
    { "name": "cam3", "file": "cam3.mp4", "sync_offset": -1 }
  ],
  "calibration": "calibration.json"
}
```

Field semantics:
- `cameras[*].file` — relative to session directory; falls back to glob discovery if omitted.
- `cameras[*].sync_offset` — non-negative number of frames to discard from the start of this camera's video before alignment begins. Use to trim pre-roll: if this camera started recording N frames earlier than the latest-starting camera, set `sync_offset=N`. Default `0`.
- `calibration` — optional relative path; the `--calibration` CLI flag wins if both are present.

When the manifest is absent, `discover_session()` falls back to glob-discovered cameras sorted by name, zero sync offsets, and `calibration.json` if present in the directory.

## Synchronization model

Software sync only (no hardware genlock assumed). Three layers:

1. **Recorder-aligned (default).** Assume cameras share frame indices. `sync_offset=0` for all.
2. **Manifest-declared integer offsets.** `session.json:cameras[*].sync_offset` skips N frames on the late camera.
3. **Audio cross-correlation.** FUTURE — `--sync-strategy audio` will compute offsets from the audio tracks.

`iter_synchronized_frames()` yields a `SessionFrame` per *logical* frame index (post-offset). Cameras that exhaust early end the iteration when any one camera is done.

## Module split

| File | Role |
|------|------|
| `src/pose_estimation/multicam.py` | `Session` dataclass, `discover_session`, `iter_synchronized_frames`, `process_session` (callback-based orchestrator). |
| `src/pose_estimation/calibration.py` | `CameraCalibration` / `SessionCalibration` IO, validation, charuco solver (stub). See `tech/calibration.md`. |
| `src/pose_estimation/triangulation.py` | DLT helpers + `fuse_session_frame` (stub). |
| `src/pose_estimation/calibration_cli.py` | `pose-estimation-calibrate` console script. |

`_types.py` extensions: `CameraCalibration`, `SessionCalibration`, `SessionFrame`, `MultiCamPipelineState`.

## CLI surface

Both `pose-estimation` (`main.py`) and `pose-estimation-run` (`run.py`) accept:

| Flag | Effect |
|------|--------|
| `--session-dir <dir>` | Process one session (mutually exclusive with `--source`/`--batch-dir`/`--sessions-dir`). |
| `--sessions-dir <dir>` | Iterate over all session subdirectories. |
| `--calibration <file>` | Override calibration path. Otherwise the session's `calibration.json` (if present) is used. |

New console script:
- `pose-estimation-calibrate verify --calibration <file>` — load + print summary (works now).
- `pose-estimation-calibrate solve --session-dir <dir> --output <file>` — charuco solve (stub).
- `pose-estimation-calibrate capture --session-dir <dir>` — guided capture (stub).

## Processing flow

`process_session()` orchestrates per-camera video processing via a caller-supplied `camera_processor` callback:

1. `discover_session(<dir>)` → `Session` (cameras + calibration).
2. Create output directory: `<output_dir>/<session_id>/`.
3. For each camera, call `camera_processor(source=..., output_csv=..., output_diag=..., video_name=...)`.
4. Return `dict[str, Any]` mapping camera name → processor result.

The `camera_processor` callback encapsulates backend-specific logic:
- **MediaPipe path** (`main.py`): closure wraps `process_video()` with CSV writer, diag writer, metrics collector setup/teardown.
- **rtmlib path** (`run.py`): closure wraps `process_source()` with smoother reset; returns latency list.

Both `_dispatch_sessions()` functions construct the callback from pre-initialized model state (models/anchors/tracker/smoother) and pass it to `process_session()`.

Current state: per-camera 2D processing is fully wired. 3D fusion (`triangulation.fuse_session_frame`) is a follow-up — will be called inside `process_session()` after per-camera processing completes, when calibration is present.

## Cross-references

- Calibration file schema + workflow: `tech/calibration.md`
- Per-camera tracking modes: `tech/tracking-modes.md`
- CLI surface: `tech/entrypoints.md`
- Tests: `tech/tests.md` (`test_multicam.py`, `test_calibration.py`)

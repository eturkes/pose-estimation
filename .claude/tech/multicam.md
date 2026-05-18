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
| `src/pose_estimation/multicam.py` | `Session` dataclass, `discover_session`, `iter_synchronized_frames`, `process_session` (stub). |
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

## Processing flow (planned)

When wired (follow-up), per session:

1. `discover_session(<dir>)` → `Session` (cameras + calibration).
2. For each camera, open `cv2.VideoCapture` and skip `sync_offset` frames.
3. Per logical frame: read N frames, build `SessionFrame`.
4. Per camera, run existing single-source pipeline (`processing.process_frame` or rtmlib path) → per-camera keypoint set + `PipelineState`.
5. `triangulation.fuse_session_frame(session_frame, per_cam_keypoints, calibration)` → 3D world keypoints.
6. Write per-camera CSV rows (existing schema) + `world3d.csv` row.

Current state: steps 1–3 are implemented; step 4 happens only via the existing single-source CLI; steps 5–6 raise `NotImplementedError`.

## Cross-references

- Calibration file schema + workflow: `tech/calibration.md`
- Per-camera tracking modes: `tech/tracking-modes.md`
- CLI surface: `tech/entrypoints.md`
- Tests: `tech/tests.md` (`test_multicam.py`, `test_calibration.py`)

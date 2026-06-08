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
├── world3d.csv         # FUTURE (3D-export task): fusion runs in-memory; CSV export pending
└── world3d_diag.csv    # FUTURE
```

Per-camera CSV columns are unchanged from the single-source schema (`tech/tracking-modes.md`). Fusion (`fuse_session_outputs`) is wired; the world3d CSV schema is the remaining 3D-export task.

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
| `src/pose_estimation/multicam.py` | `Session` dataclass, `discover_session`, `iter_synchronized_frames`, `process_session` (callback-based orchestrator + post-hoc fusion hook), `fuse_session_outputs` → `SessionFusion`. |
| `src/pose_estimation/calibration.py` | `CameraCalibration` / `SessionCalibration` IO, validation, charuco solver (stub). See `tech/calibration.md`. |
| `src/pose_estimation/triangulation.py` | DLT helpers + `fuse_session_frame` policy layer (validity masking, weighted DLT, outlier-view rejection, cheirality flag, `FusionDiagnostics`). |
| `src/pose_estimation/calibration_cli.py` | `pose-estimation-calibrate` console script. |

`_types.py` extensions: `CameraCalibration`, `SessionCalibration`, `SessionFrame`, `MultiCamPipelineState`, `FusionDiagnostics`.

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

## 3D fusion (post-hoc, CSV read-back)

When `session.calibration` is present, `process_session()` ends by calling `fuse_session_outputs(session, output_dir)` (non-fatal: failures print a WARNING — the 2D CSVs are already on disk and fusion can be re-run standalone). Data flow:

1. `export.read_csv_keypoints(<cam>.csv)` per camera → `frame_idx → ((N,2) normalised kps, (N,) conf)`; `person_idx == 0` rows only (cross-camera person matching is out of scope); body/arm `_vis` is the confidence, hand keypoints get presence 1.0/0.0.
2. Normalised → pixels via the camera's **calibrated** `resolution` (normalised coords make CSV resolution-independent).
3. CSV rows hold *raw* per-camera frame indices → logical index = raw − `sync_offset` (negatives dropped).
4. Every logical frame observed by ≥ `min_views` (default 2) cameras → `triangulation.fuse_session_frame(per_camera_kps_px, calibration, confidences=...)` → `(N,3)` world metres + `FusionDiagnostics` (n_views, confidence, reprojection_error_px, cheirality_ok).
5. Result: `SessionFusion(keypoint_names, frames=[(frame_idx, world, diag), ...])` — in-memory only; `world3d.csv` export is the remaining 3D-export task.

`fuse_session_frame` policy: a view contributes when coords are finite and conf > `min_confidence` (0.0); greedy outlier rejection re-triangulates while any view reprojects worse than `max_view_reproj_px` (20.0) and > `min_views` views remain; cheirality violations are flagged, not dropped; at exactly `min_views` a residual outlier stays visible via the reprojection diagnostic.

## Cross-references

- Calibration file schema + workflow: `tech/calibration.md`
- Per-camera tracking modes: `tech/tracking-modes.md`
- CLI surface: `tech/entrypoints.md`
- Tests: `tech/tests.md` (`test_multicam.py`, `test_calibration.py`)

# Multi-camera sessions and fusion

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
└── world3d.csv         # fused 3D output (written when calibration present)
```

Per-camera CSV columns are unchanged from the single-source schema (`tracking-modes.md`).

A hand-authored directory is one way to reach this layout. The corpus reaches it through `pose-estimation-sessions`, which publishes `sessions/<event_id>/` with one `cam-<view>` symbolic link per camera and one generated manifest. Discovery treats both trees identically. See `sessions.md`.

### `world3d.csv` schema

Written by `export.write_world3d_csv` (header from
`export.make_world3d_header`). Metadata: `video` (= session_id), `frame_idx`
(logical), `timestamp_sec`, `person_idx` (always 0). Per keypoint (names match
the 2D schema): `{name}_x_m,_y_m,_z_m` (world metres, 6dp),
`{name}_confidence` (4dp), `{name}_reproj_err_px` (3dp),
`{name}_candidate_n_views` (eligible views before robust consensus),
`{name}_n_views` (final consensus contributors), `{name}_cheirality_ok` (0/1), and
`{name}_triangulation_angle_deg` (3dp). The angle is the maximum acute angle
between consensus viewing rays; it exposes weak-baseline/near-parallel geometry.
Unfused keypoints have blank coordinates/reprojection error, while a computed
angle can remain present when the angle gate rejects the point. World frame =
the `world_frame` camera's OpenCV frame (+x right, +y down, +z away from
camera); "up" = −y assumes that camera is level. Diagnostics are embedded
per keypoint so consumers can independently gate downstream analysis.

## `session.json` manifest (optional)

```json
{
  "format_version": 1,
  "session_id": "session_2026-05-20_subject01",
  "cameras": [
    { "name": "cam1", "file": "cam1.mp4", "sync_offset": 0 },
    { "name": "cam2", "file": "cam2.mp4", "sync_offset": 2 },
    { "name": "cam3", "file": "cam3.mp4", "sync_offset": 1 }
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
2. **Manifest-declared integer offsets.** `session.json:cameras[*].sync_offset` skips N pre-roll frames on an earlier-starting camera.
3. **Audio cross-correlation.** FUTURE — `--sync-strategy audio` will compute offsets from the audio tracks.

`iter_synchronized_frames()` yields a `SessionFrame` per *logical* frame index (post-offset). Cameras that exhaust early end the iteration when any one camera is done.

## Module split

| File | Role |
|------|------|
| `src/pose_estimation/multicam.py` | `Session` dataclass, `discover_session`, `iter_synchronized_frames`, `process_session` (callback-based orchestrator + post-hoc fusion hook), `fuse_session_outputs` → `SessionFusion` → `world3d.csv`. |
| `src/pose_estimation/calibration.py` | `CameraCalibration` / `SessionCalibration` IO + validation (cv2-free). See `calibration.md`. |
| `src/pose_estimation/charuco.py` | ChArUco board construction/rendering, corner detection, `solve_charuco` (intrinsics + pairwise extrinsics + global RMS). See `calibration.md`. |
| `src/pose_estimation/triangulation.py` | DLT helpers + `fuse_session_frame` policy layer (validity masking, minimal-set consensus, confidence-weighted refit, geometric refinement, residual/ray-angle gates, cheirality, `FusionDiagnostics`). |
| `src/pose_estimation/calibration_cli.py` | `pose-estimation-calibrate` console script (verify/solve/board/capture). |

`_types.py` extensions: `CameraCalibration`, `SessionCalibration`, `SessionFrame`, `MultiCamPipelineState`, `FusionDiagnostics`.

## CLI surface

Both `pose-estimation` (`main.py`) and `pose-estimation-run` (`run.py`) accept:

| Flag | Effect |
|------|--------|
| `--session-dir <dir>` | Process one session (mutually exclusive with `--source`/`--batch-dir`/`--sessions-dir`). |
| `--sessions-dir <dir>` | Iterate over all session subdirectories. |
| `--calibration <file>` | Override calibration path. Otherwise the session's `calibration.json` (if present) is used. |

Console script `pose-estimation-calibrate` (verify/solve/board/capture): see `calibration.md` § CLI surface.

## Processing flow

`process_session()` orchestrates per-camera video processing via a caller-supplied `camera_processor` callback:

1. `discover_session(<dir>)` → `Session` (cameras + calibration).
2. Create output directory: `<output_dir>/<session_id>/`.
3. For each camera, call `camera_processor(source=..., output_csv=..., output_diag=..., video_name=...)`.
4. Return `dict[str, Any]` mapping camera name → processor result.

The `camera_processor` callback encapsulates backend-specific logic:
- **MediaPipe path** (`main.py`): closure wraps `process_video()` with CSV writer, diag writer, metrics collector setup/teardown.
- **rtmlib path** (`run.py`): closure wraps `process_source()` with smoother reset; returns latency list.

Both `_dispatch_sessions()` functions resolve `--session-dir`/`--sessions-dir` via the shared `multicam.resolve_cli_sessions()` (mutual-exclusion guard, per-session `--calibration` override, dispatch summary print; raises `SessionError`), construct the callback from pre-initialized model state (models/anchors/tracker/smoother), and pass it to `process_session()`.

## 3D fusion (post-hoc, CSV read-back)

When `session.calibration` is present, `process_session()` ends by calling `fuse_session_outputs(session, output_dir)` (non-fatal: failures print a WARNING — the 2D CSVs are already on disk and fusion can be re-run standalone). Data flow:

1. `export.read_csv_keypoints(<cam>.csv)` per camera → `frame_idx → ((N,2) normalised kps, (N,) conf, timestamp_sec)`; `person_idx == 0` rows only (cross-camera person matching is out of scope); body/arm `_vis` and hand `_conf` carry observation confidence. Legacy hand CSVs without `_conf` use finite-coordinate presence 1.0/0.0.
2. Normalised → pixels via the camera's **calibrated** `resolution` (normalised coords make CSV resolution-independent).
3. New CSV rows hold zero-based decoded-source frame indices (legacy one-based
   files remain readable by validation). Logical index = raw − `sync_offset`
   (negatives dropped); malformed frames leave gaps rather than compressing the
   timeline.
4. Every logical frame observed by ≥ `min_views` (default 2) cameras → `triangulation.fuse_session_frame(per_camera_kps_px, calibration, confidences=...)` → `(N,3)` world metres + `FusionDiagnostics` (candidate_n_views, n_views, confidence, reprojection_error_px, cheirality_ok, triangulation_angle_deg).
5. Result: `SessionFusion(keypoint_names, frames=[(frame_idx, timestamp_sec, world, diag), ...])` — the exact row layout `export.write_world3d_csv` consumes; `_fuse_and_report` writes `world3d.csv` in the session output dir (same non-fatal try block). Timestamp per logical frame: taken from the `world_frame` camera when finite, else first finite among the others (session order) — per-camera timestamps are raw-index-based, so a non-world-frame fallback can carry a constant offset; downstream uses dt-median, which is shift-invariant.

`fuse_session_frame` applies the following per-keypoint policy:

1. A view is eligible when x/y are finite and confidence is greater than
   `min_confidence` (default 0.0). Supplied confidences are validated as finite
   and clipped to `[0, 1]`.
2. Every eligible two-view minimal set proposes a 3D point. Hypotheses are
   ranked deterministically by reprojection-inlier count, summed inlier
   confidence, cheirality count, then confidence-weighted truncated loss.
3. The winning consensus is refit with DLT. Each DLT row is scaled by
   `sqrt(confidence)`, so the least-squares objective receives the intended
   confidence weight rather than its square.
4. A bounded, backtracked robust refinement minimizes geometric reprojection
   error in the original distorted pixel frame. The seed is retained unless
   robust loss and mean error are both non-worsening.
5. The point is invalidated unless every final consensus-view residual is at
   most `max_view_reproj_px` (default 20.0) and the maximum acute pairwise ray
   angle is at least `min_triangulation_angle_deg` (default 1.0°). Cheirality
   violations remain explicitly flagged rather than silently discarded.

The 20 px residual gate, 1° ray-angle gate, and confidence/min-view settings
are provisional engineering defaults. The implementation and failure modes are
covered by synthetic geometry tests; no sensitive recordings or real clinical
footage were opened or used to establish accuracy.

## Cross-references

- Calibration file schema + workflow: `calibration.md`
- Capture/QA protocol + the frame-count-parity desync proxy that grades a raw session: `../capture_protocol.md`, `validation.md` (`qa_check`).
- R consumption of `world3d.csv` (gating, 3D features): `analysis.md`
- Per-camera tracking modes: `tracking-modes.md`
- CLI surface: `entrypoints.md`
- Generated session trees (event grain, take resolution, placement ledger): `sessions.md`
- Tests: `tests.md` (`test_multicam.py`, `test_calibration.py`, `test_charuco.py`, `test_calibration_cli.py`)

# Camera calibration

Multi-camera 3D triangulation needs per-camera intrinsics (`K`, distortion) and extrinsics (rotation + translation in a shared world frame). This document describes the file format, IO contract, and the ChArUco-based solve workflow.

Module split: `calibration.py` is cv2-free (IO + validation); the solver lives in `charuco.py` (imports calibration/multicam/_types, acyclic).

## File format

`calibration.json`:

```json
{
  "format_version": 1,
  "session_id": "calib_2026-05-20_lab",
  "world_frame": "cam1",
  "cameras": [
    {
      "name": "cam1",
      "resolution": [1920, 1080],
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "distortion": [k1, k2, p1, p2, k3],
      "rvec": [0.0, 0.0, 0.0],
      "tvec": [0.0, 0.0, 0.0]
    },
    { "name": "cam2", ... },
    { "name": "cam3", ... }
  ],
  "reprojection_error_px": 0.34,
  "solver": "opencv-charuco",
  "solved_at": "2026-05-20T14:30:00Z"
}
```

Field semantics:

| Field | Meaning |
|-------|---------|
| `format_version` | Schema version. Current: `1`. Loader rejects unknown versions. |
| `world_frame` | Name of the camera whose frame defines the world origin (its `rvec`/`tvec` MUST be zeros). |
| `cameras[*].K` | 3×3 intrinsic matrix in pixel units. |
| `cameras[*].distortion` | OpenCV-format distortion coefficients (length 4, 5, 8, 12, or 14). |
| `cameras[*].rvec` | Rodrigues rotation vector (3,) — camera-from-world. |
| `cameras[*].tvec` | Translation vector (3,) — camera-from-world, in metres (consistent with board square size). |
| `reprojection_error_px` | RMS reprojection error from the solve. Diagnostic, not load-critical. |
| `solver` | Provenance tag. |
| `solved_at` | ISO-8601 UTC timestamp. |

## Resolution rules

`load_calibration(path)` enforces:
- `format_version == 1`.
- `K` is 3×3, last row `[0, 0, 1]`.
- `distortion` length ∈ {4, 5, 8, 12, 14}.
- `rvec` and `tvec` are length-3.
- `resolution` is a positive `[W, H]` pair.
- The camera named in `world_frame` exists and has zero `rvec`/`tvec` (small numerical tolerance applied).
- Camera names are unique within the file.

Validation errors raise `CalibrationError` with a message naming the offending field.

## Resolution lookup

```python
load_calibration(explicit_path)            # explicit --calibration <path>
load_session_calibration(session_dir)      # auto-discovers <session>/calibration.json
```

The CLI passes `--calibration` if provided; otherwise it falls back to `load_session_calibration()`.

## Solve workflow (charuco.py)

`charuco.solve_charuco(session_dir, *, board=None, world_frame=None, max_frames=50, min_corners=6, min_shared_frames=5) -> SessionCalibration`:

1. **Session discovery.** Reuses `multicam.discover_session` — same `cam*.{avi,mp4,...}` glob, `session.json` manifest, per-camera `sync_offset`. Detection frame indices are logical (`raw - sync_offset`), matching the 2D fusion convention.
2. **Per-camera detection + intrinsics.** `cv2.aruco.CharucoDetector(board).detectBoard(img)` per frame → keep frames with ≥ `min_corners` corners → `_subsample` to ≤ `max_frames` (linspace) → `board.matchImagePoints` per frame → `cv2.calibrateCamera`. Needs ≥ `MIN_INTRINSIC_FRAMES` (8) usable frames.
   - **API constraint**: `calibrateCameraCharuco`/`Extended` are ABSENT from modern OpenCV wheels entirely (verified at 4.13: absent even with the contrib binary loaded — legacy aruco API dropped upstream). The modern detectBoard→matchImagePoints→calibrateCamera path is the only option and is the current upstream recommendation.
3. **Pairwise extrinsics.** `world_frame` defaults to the first camera name. For each other camera: intersect corner ids per logical frame (`np.intersect1d`), keep frames with ≥ `MIN_SHARED_CORNERS` (6) shared corners, need ≥ `min_shared_frames` frames; `cv2.stereoCalibrate(..., CALIB_FIX_INTRINSIC)`. Its `(R, T)` maps world-cam coords → other-cam coords = camera-from-world directly (world cam IS the world frame).
   - **Topology limit**: direct pairs only — every camera must share board views with the world-frame camera. Chained extrinsics (A↔B↔C) are unsupported; arrange capture so the board is visible to the world camera + at least one other simultaneously.
4. **Global reprojection RMS.** Per logical frame: anchor board pose via `solvePnP` in the world-frame camera (fallback: first detecting camera), lift board points to world (`x_w = R_aᵀ(x_a − t_a)`), reproject into every detecting camera. Stored as `reprojection_error_px`.
5. **Persist.** Caller (CLI) writes via `save_calibration`. Solver tag: `"opencv-charuco"`.

No bundle-refinement stage: stereoCalibrate residuals + the global RMS check were sufficient on synthetic data (global RMS ≈ 0.4 px, f within 2%, rotation < 1°, translation < 15 mm on a 0.84 m baseline). Revisit only if real captures show drift.

Board defaults (constants in `charuco.py`): 6×9 squares, `DICT_4X4_250`, 40 mm squares, 30 mm markers. `make_charuco_board()` validates marker < square; `render_charuco_board()` produces the printable image (texture mapping is identity: `texture_px = obj_m / square_size * px_per_square`, +y down).

**Capture accuracy lesson**: a narrow board-pose cloud weakly constrains oblique cameras' intrinsics, and fx error couples into stereo tvec (16 mm error on synthetic data). Move the board through the full working volume — translation AND tilt diversity — not just the centre.

## CLI surface

```bash
pose-estimation-calibrate verify  --calibration calib.json
pose-estimation-calibrate solve   --session-dir videos/calib_session/ --output calib.json [--world-frame cam1] [--max-frames 50] [board args]
pose-estimation-calibrate board   --output board.png [--px-per-square 240] [board args]
pose-estimation-calibrate capture --session-dir videos/calib_session/ --devices 0,1,2 [--names cam1,cam2,cam3] [--width W --height H]
```

Board args: `--squares COLSxROWS --square-size-m 0.04 --marker-size-m 0.03`. All errors → stderr `ERROR: ...`, exit 2.

- `board` renders a printable PNG and prints the physical pattern size; print at 100% scale and verify one square with a ruler.
- `capture` opens a pygame grid of live feeds with green ChArUco-corner overlay; SPACE appends one synchronized frame per camera to per-camera MJPG AVIs (frame index = press index, so the videos are inherently synchronized and feed straight into `solve`); Q/ESC quits. Exit 1 if nothing captured.

## Numerical conventions

- Coordinate frame: right-handed, +X right, +Y down, +Z forward (OpenCV convention).
- Units: metres for tvec / world keypoints, pixels for K / 2D keypoints.
- Distortion model: OpenCV (radial + tangential, with optional rational/thin-prism terms via 8/12/14 coefficients).
- `world_frame` camera's pose is identity by definition. Other cameras' poses are expressed *as* their `cv2.projectPoints` argument — i.e. rotation of world points into camera space.

## Cross-references

- Session abstraction: `tech/multicam.md`
- Capture procedure (board sweep, ≳25 px/square, world-camera co-visibility topology) + the automated QA gate that grades a capture: `docs/capture_protocol.md`, `tech/validation.md` (`qa_check`).
- Data sensitivity: calibration files capture lab geometry; treat as patient-adjacent. The `videos/` symlink is already git-ignored.
- Tests: `tech/tests.md` (`test_calibration.py` IO, `test_charuco.py` solver on synthetic renders, `test_calibration_cli.py` CLI wiring)

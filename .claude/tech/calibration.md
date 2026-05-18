# Camera calibration

Multi-camera 3D triangulation needs per-camera intrinsics (`K`, distortion) and extrinsics (rotation + translation in a shared world frame). This document describes the file format, IO contract, and the (planned) ChArUco-based solve workflow.

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

## Planned solve workflow (charuco)

`calibration.solve_charuco()` is a `NotImplementedError` stub. The intended pipeline:

1. **Capture.** Record N synchronized videos of an OpenCV ChArUco board. Move the board through the full working volume; cover all orientations.
2. **Per-camera intrinsics.** For each camera independently:
   - Detect ChArUco corners via `cv2.aruco.CharucoDetector`.
   - Pick frames with ≥ 6 corners; subsample to ~50 frames spread across the video.
   - Solve with `cv2.aruco.calibrateCameraCharucoExtended` → `K`, `distortion`, per-frame `(rvec, tvec)` board poses.
3. **Pairwise extrinsics.** Choose `cam1` as the world reference. For each other camera, find frames where both cameras see the board, compute relative pose via `cv2.solvePnP` on shared 3D-2D correspondences, average over frames (or run a bundle adjustment step via `cv2.sba`/g2o/pycolmap for a refined estimate).
4. **Bundle refinement (optional).** Joint refinement of all intrinsics + extrinsics minimising total reprojection error. Implementation TBD — Ceres via pycolmap is the most direct, scipy.optimize.least_squares is the dependency-free fallback.
5. **Persist.** Write `calibration.json` with `solved_at` and `reprojection_error_px`.

The ChArUco board geometry (square size, marker size, board layout, dictionary) lives in `calibration.CHARUCO_BOARD_DEFAULT` once implemented; capture script will print the recommended board for printing.

## CLI surface

```bash
pose-estimation-calibrate verify --calibration calib.json
pose-estimation-calibrate solve --session-dir videos/calib_session/ --output calib.json
pose-estimation-calibrate capture --session-dir videos/calib_session/
```

`verify` is implemented (load + summary table). `solve` and `capture` are stubs that print the planned design when invoked.

## Numerical conventions

- Coordinate frame: right-handed, +X right, +Y down, +Z forward (OpenCV convention).
- Units: metres for tvec / world keypoints, pixels for K / 2D keypoints.
- Distortion model: OpenCV (radial + tangential, with optional rational/thin-prism terms via 8/12/14 coefficients).
- `world_frame` camera's pose is identity by definition. Other cameras' poses are expressed *as* their `cv2.projectPoints` argument — i.e. rotation of world points into camera space.

## Cross-references

- Session abstraction: `tech/multicam.md`
- Data sensitivity: calibration files capture lab geometry; treat as patient-adjacent. The `videos/` symlink is already git-ignored.
- Tests: `tech/tests.md` (`test_calibration.py`)

# Session prompts

Paste the kickoff prompt from `prompts/kickoff.md`, then append one of the task blocks below as the `<TASK>` section. Sessions are designed to run autonomously with minimal user input.

## Completed roadmap (2026-05-24)

The Clinical Pipeline E2E roadmap (8 tasks) is fully complete:
1. COCO-WholeBody → MediaPipe keypoint mapping ✓
2. Wire CSV export into rtmlib process_source() ✓
3. Test rtmlib CSV schema compat with R pipeline ✓
4. Harden R scripts for edge cases ✓
5. E2E clinical pipeline smoke test ✓
6. Dependency update + security audit ✓
7. Refactor main.py/run.py (analysis: not worthwhile) ✓
8. Tech notes drift audit ✓

## Next phase: Multi-cam 3D pipeline

Blocked on incoming 3-camera footage and calibration board data. The scaffolding (Session, calibration JSON, triangulation math) is in place — only the algorithmic implementations remain as stubs.

---

## Session: Implement ChArUco calibration solver

```
Execute: Implement solve_charuco() in calibration.py.

Load tech notes: calibration.md, multicam.md.

Prerequisites: 3-camera calibration board footage must be available.

1. Implement ChArUco board detection using OpenCV's aruco module.
2. Compute per-camera intrinsics (K, distortion) from detected corners.
3. Compute pairwise extrinsics (rvec, tvec) relative to camera 0.
4. Validate with reprojection error (target: <1px RMS).
5. Save result via save_calibration() in the existing JSON schema.
6. Wire into calibration_cli.py's `solve` subcommand.
7. Update calibration.md with the implemented workflow.
```

---

## Session: Implement 3D triangulation fusion

```
Execute: Implement fuse_session_frame() in triangulation.py.

Load tech notes: architecture.md, multicam.md, calibration.md.

Prerequisites: solve_charuco must be implemented. Calibrated multi-cam footage available.

1. Implement fuse_session_frame(): accept per-camera 2D keypoints + calibration, produce 3D keypoints via weighted DLT.
2. Confidence weighting: use per-keypoint visibility scores to weight triangulation.
3. Handle missing keypoints (some cameras may not detect all joints).
4. Wire into process_session() — after per-camera 2D processing, call fuse_session_frame() when calibration is present.
5. Output 3D keypoints to a new CSV schema (x, y, z in world coordinates).
6. Add tests with synthetic multi-view data.
7. Update architecture.md and multicam.md.
```

---

## Session: Real clinical footage validation

```
Execute: Validate the full pipeline on real clinical footage.

Load tech notes: architecture.md, tracking-modes.md, analysis.md.

Prerequisites: Clinical footage available in videos/.

1. Run the rtmlib pipeline (RTMW-L, hands-arms mode) on the footage.
2. Verify CSV export works end-to-end.
3. Run clinical_features.R on the output.
4. Compare results qualitatively (are joint angles/reach distances plausible?).
5. If jitter/drops persist, tune parameters (outlier_cap, carry_frames, min_cutoff).
6. Document any issues found and fixes applied.
```

---

## Session: Making a plan for continued development with minimal input from me

```
Making a plan for continued development with minimal input from me
```

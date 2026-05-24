# Scratchpad

Transient working notes — anything from "current investigation" to "half-finished idea". Safe to prune at session boundaries. Never treat as a source of truth; promote stable findings to `tech/`, decisions to `decisions.md`, lessons to `lessons.md`.

## How to use

- Start each session by skimming the most-recent entries (top of file).
- Append `## YYYY-MM-DD HH:MM — <session topic>` and write freely.
- When closing a session, either prune the entry or summarise it into a longer-lived file.

---

## 2026-05-24 — E2E roadmap complete; all 8 tasks resolved

Session completed the full Clinical Pipeline E2E roadmap:

- **Tasks #1/#2** (prior session): keypoint mapping + CSV export wiring
- **Task #3**: R pipeline compat — renv updated for R 4.6.0 (Matrix 1.7-5, renv 1.2.3). All 13 pytest tests pass (9 schema + 4 R integration). rtmlib CSV schema is fully compatible with clinical_features.R in both hands-arms and body modes.
- **Task #4**: R hardening — Fixed 3 crash-on-edge-case bugs: compare_clinical.R (zero-variance stop→warn), clinical_dimreduce.R (same pattern), features.R (NaN from scale() with zero-variance columns), longitudinal.R (scalar/vector if_else mismatch). All 10 R scripts now exit 0 on edge-case data.
- **Task #5**: Dependencies upgraded (openvino 2026.1, onnxruntime 1.26, ruff 0.15.14, ty 0.0.39, etc.) + security audit. Fixed 2 path traversal vulnerabilities in multicam.py (session.json camera/calibration references). 2 new tests. 241 total tests passing.
- **Task #6**: Refactor analysis concluded dedup not worthwhile (~15 shared lines across 2020 total; fundamentally different pipelines).
- **Task #7**: Tech notes audit — fixed environment.md (Python version, kernel), added test_r_pipeline.py to tests.md, added edge-case resilience section to analysis.md.
- **Task #8**: Session prompts and kickoff updated.

**Next phase:** Multi-cam 3D pipeline (blocked on incoming footage/calibration data). Tasks: solve_charuco, fuse_session_frame, 3D visualization. Independent work: performance profiling on real clinical footage.

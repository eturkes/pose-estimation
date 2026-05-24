# Scratchpad

Transient working notes — anything from "current investigation" to "half-finished idea". Safe to prune at session boundaries. Never treat as a source of truth; promote stable findings to `tech/`, decisions to `decisions.md`, lessons to `lessons.md`.

## How to use

- Start each session by skimming the most-recent entries (top of file).
- Append `## YYYY-MM-DD HH:MM — <session topic>` and write freely.
- When closing a session, either prune the entry or summarise it into a longer-lived file.

---

## 2026-05-24 — Session 2D complete: Temporal movement segmentation

Added velocity-profile + aperture-derivative movement segmentation to `clinical_features.R`:

1. **running_median()**: sliding median filter for speed/aperture smoothing.
2. **classify_movement_phases()**: state machine classifies frames within a movement as REACH → GRASP → TRANSPORT → RELEASE using smoothed grasp-aperture derivative. Adaptive threshold (5% of aperture range). Requires `min_phase_frames` (default 3) consecutive frames for transition. Falls back to REACH-only when no hand data or insufficient aperture variation.
3. **segment_movements()**: main orchestrator — detects movements via speed threshold (5% of peak, RLE-based), merges close segments, rejects short ones, classifies phases, extracts per-phase features (peak/mean velocity, path length, NJ, SAL, bilateral reach symmetry) and per-movement summary (duration, n_phases, efficiency).

Output: `*_movement_phases.csv` — one row per phase per movement per side per person. 19 columns total.

File exclusion filter updated to skip `*_movement_phases.csv` when processing directories.

Tests: 251 passing (+2 new: smoke test for output schema + crafted reach-grasp trajectory validation). The crafted test writes a CSV directly in MediaPipe format with a known half-sine reach trajectory and open→close→closed→open aperture pattern, then verifies REACH is detected and precedes GRASP.

**Roadmap status:** Phase 1 ✓ (1A, 1B), Phase 2 ✓ (2A, 2B, 2C, 2D) → Phase 2 complete. Next: Phase 3 (3D pipeline, awaiting footage) or Phase 4 (maintenance).

---

## 2026-05-24 — Session 2C complete: Trunk/torso metrics (body mode)

Added 4 trunk/torso metrics to `clinical_features.R`, all gated behind `tracking == "body"`:

1. **trunk_lean_deg**: Reuses existing `trunk_lean_angle()` — unsigned total lean from vertical (0 = upright).
2. **trunk_lean_lateral_deg**: New `trunk_lean_lateral()` — signed lateral lean via `atan2(dx, -dy)`. Positive = leaning right.
3. **trunk_rotation_deg**: New `trunk_rotation()` — signed angle difference between shoulder line (L→R) and hip line (L→R). Positive = clockwise rotation (viewed from front).
4. **posture_symmetry**: New `posture_symmetry()` — (lsh_y − rsh_y) / shoulder_width_2d. Positive = right shoulder higher.

Per-frame: 4 new columns. Per-window: 9 new columns (mean/sd/range for lean, mean/sd for the other three).

Non-body modes: columns emitted with NA values for schema consistency across modes.

Tests: 249 passing (1 new: hands-arms trunk columns are NA). Ruff clean.

**Roadmap status:** 1A ✓, 1B ✓, 2A ✓, 2B ✓, 2C ✓ → Next: 2D (temporal segmentation, only remaining Phase 2 task).

---

## 2026-05-24 — Session 2B complete: Movement quality scores

Added 3 new movement quality metrics + 1 SAL improvement to `clinical_features.R`:

1. **normalized_jerk()**: Dimensionless jerk metric (Hogan & Sternad 2009). Applied to wrist and fingertip per side per window. NJ = sqrt(T^5 / (2*a^2) * integral(jerk^2 dt)). Lower = smoother (min-jerk ≈ 18.97).
2. **movement_efficiency()**: path_length / straight_line_distance for wrist per side per window. 1.0 = perfectly straight.
3. **compensatory_pattern_index**: Pearson cor(trunk_lean, max(L/R reach)) per window. Body mode only (requires hip keypoints). Uses new `trunk_lean_angle()` helper (reusable for session 2C).
4. **SAL fc parameter**: `spectral_arc_length(v, fs, fc)` — frequency cutoff now configurable (default 10 Hz unchanged, matches Balasubramanian et al. 2012).

New window columns (per side): `{side}_wrist_normalized_jerk`, `{side}_wrist_movement_efficiency`, `{side}_fingertip_normalized_jerk`. Plus `compensatory_pattern_index` (single column). Bilateral comparison extended from 3 to 6 metric pairs (9 new bilateral columns).

Total new columns in _clinical_windows.csv: 6 (per-side metrics) + 1 (CPI) + 9 (new bilateral × 3 each) = 16.

Tests: 248 passing (1 new: body-mode window quality metrics). Ruff clean.

**Roadmap status:** 1A ✓, 1B ✓, 2A ✓, 2B ✓ → Next: 2C (trunk/torso metrics) or 2D (temporal segmentation, blocked by 2B which is now done).

---

## 2026-05-24 — Session 2A complete: Bilateral comparison metrics

Added `compute_bilateral()` helper to `clinical_features.R` and wired it into both `compute_frame_features()` (9 metric pairs × 3 bilateral columns = 27 new columns) and `compute_window_features()` (3 metric pairs × 3 = 9 new columns). Total: 36 new bilateral columns across the two output CSVs.

Key design: uses abs() internally so the formulas work for both non-negative metrics (angles, distances) and negative metrics (SAL). Division-by-zero guard at denom > 1e-12. R's NA propagation handles missing-side gracefully.

Tests: updated test_r_pipeline.py with bilateral column assertions for both per-frame and per-window outputs. All 247 tests pass. Ruff clean.

**Roadmap status:** 1A ✓, 1B ✓, 2A ✓ → Next: 2B (movement quality scores), 2C (trunk/torso metrics), or both in parallel.

---

## 2026-05-24 — Session 1B complete: Adaptive smoothing

Implemented movement-phase-aware min_cutoff adaptation in both filter implementations (OneEuroFilter in smoothing.py, _OneEuro in run.py). Core mechanism: per-keypoint EMA of velocity magnitude classifies each keypoint as REST/SLOW/FAST, then interpolates effective min_cutoff between rest_cutoff (heavy smoothing) and min_cutoff (normal). Beta mechanism still handles fast movement via cutoff += beta*|speed|; adaptive mode only affects the floor.

Defaults: body rest_cutoff=0.05 (6x heavier than mc=0.3), hand rest_cutoff=0.15 (3.3x heavier than mc=0.5). Thresholds: rest_speed=2.0, fast_speed=10.0 px/frame. Speed EMA alpha=0.1 (~10 frame time constant).

Design choice: embedded adaptive logic directly in the filter __call__ rather than wrapper/mixin (KISS — no extra class, no indirection, backwards-compatible via rest_cutoff=None). PoseSmoother and KeypointSmoother pass env-var-driven rest_cutoff through to their filter constructors.

6 new tests, 247 total passing. sweep_default.yaml updated with 4 new params.

**Roadmap status:** 1A ✓, 1B ✓ → Phase 1 complete. Next: Phase 2 (2A-2D clinical metrics in R).

---

## 2026-05-24 — New roadmap: Stability + Clinical Metrics + 3D Pipeline

User confirmed: jitter/drops persist across backends/modes, and four categories of new clinical metrics needed (trunk/torso, movement quality, bilateral comparison, temporal segmentation). 3-cam footage is ~2-4 weeks away.

Plan rationale:
- **Phase 1 (Tracking stability)** is highest priority because noisy input corrupts ALL downstream clinical metrics. Fixing jitter first ensures the new metrics in Phase 2 are computed from clean data.
- **Phase 2 (Clinical metrics)** is the bulk of the work. Four sub-tasks ordered by complexity: bilateral comparison (easiest, derives from existing per-side metrics) -> movement quality scores (extends SAL infrastructure) -> trunk/torso (body-mode gated, needs careful keypoint availability checks) -> temporal segmentation (most complex, benefits from having other metrics available for per-phase extraction).
- **Phase 3 (3D pipeline)** can overlap with Phase 2 since they're independent codepaths. Start when footage timeline firms up. All stubs can be implemented and tested against synthetic data.
- **Phase 4 (Maintenance)** is periodic and interleaved freely.

Key design decisions for this plan:
- Trunk metrics are body-mode only (hands-arms has NO hip keypoints). Must gate behind detect_tracking().
- Temporal segmentation uses velocity-profile + grasp-aperture state machine (rule-based, interpretable for clinicians, avoids ML training data requirements).
- Bilateral comparison uses min(L,R)/max(L,R) symmetry ratio (1.0 = symmetric) and (R-L)/(R+L) dominance index (signed, handles division-by-zero).
- fuse_session_frame() can be implemented entirely with synthetic tests before real footage arrives.

Session prompts updated in `.claude/prompts/sessions.md`. 10 tasks tracked.

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

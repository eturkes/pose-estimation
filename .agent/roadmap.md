# Roadmap

Plan + status. `/session-prompt` with no argument takes the next open step here. Finished-roadmap detail and per-session execution prompts are in git history.

## Current: Real-World 3D Clinical Validation (adopted 2026-06-15)

**Goal.** Prove the 3D clinical pipeline on *real* 3-camera recordings — not new features. A reproducible harness (calibration → 2D tracking → `world3d.csv` fusion → clinical metrics) reporting reprojection error, dropped/low-confidence keypoints, timing, and clinical-metric agreement; plus a capture/QA protocol, anonymized fixtures, failure-mode tests, and thresholds + a clinical-validity gap register.

**Carried caveat.** The entire 3D path (calibration solve, fusion, `world3d.csv`, 3D clinical metrics) is validated by **synthetic unit tests only**; real 3-camera footage has never run end-to-end. Closing that gap is the point of this roadmap.

**Blockers (as of seed, 2026-06-15).**
- **No footage** under `rehab/data/videos`.
- **No baseline** (known-geometry object / goniometer / second reference system) → clinical-metric *agreement* is unmeasurable. Until one exists, substitute internal evidence: reprojection error, cross-camera consistency, temporal stability, inter-trial repeatability.

- **Phase 1 — validation scaffolding (footage-independent):** done ✓
  - 1A harness core (`validation.py`: `run_validation`, `ValidationReport`, `pose-estimation-validate`) + synthetic E2E fixture
  - 1B `THRESHOLDS` + `verdict()` PASS/WARN/FAIL (exit code for CI) + clinical-validity gap register
  - 1C capture/QA protocol (`docs/capture_protocol.md`) + `qa_check` / `--qa-only` + anonymization strategy
  - 1D failure-mode test suite (`tests/test_validation_failuremodes.py`)
  - reference: validation, multicam, calibration, analysis, tests
- **Phase 2 — real-data validation (gated on footage; agreement also on a baseline):** open
  - 2A first real-capture dry run; calibrate protocol + `THRESHOLDS` against reality; record the deltas
  - 2B quantitative validation over multiple sessions (reprojection, drops, timing, stability, inter-trial repeatability)
  - 2C clinical-metric agreement vs a baseline if one exists (Bland–Altman, ICC); else record the standing gap + cheapest-baseline plan; derive anonymized real fixtures; add a regression lock
  - reference: validation, calibration, multicam, analysis, environment
  - **Gate:** when a Phase 2 step is next and `rehab/data/videos` holds no real session, stop and report the block — never fabricate results.
- **Phase 3 — maintenance (periodic, interleave freely):** see Maintenance cycle.

## Backlog (unscheduled)
- 3D-aware downstream aggregation (`analysis/` is currently 2D-oriented).
- Multi-person cross-camera identity matching (fusion is single-person).
- Host-launch caveat: the container-native `.venv` won't resolve if the pipeline is launched from the host (e.g. NPU runs); would need a host-side `uv sync`.

## Maintenance cycle (reusable; roadmap-agnostic)
reference: environment, architecture, tests
1. Python deps: `uv lock --upgrade` → `uv sync`; full suite green.
2. R deps: `renv::update()` → `renv::snapshot()`; R scripts exit 0.
3. Security: CLI injection vectors; session.json / calibration.json path traversal (`_safe_resolve` coverage); new CVEs in openvino, onnxruntime, opencv, rtmlib (web sweep).
4. `uv run pytest` / `ruff check` / `ruff format` / `ty check`.
5. Tech-notes drift: `.claude/tech/*.md` vs current code (module map, CLI flags, test inventory, API surface).
6. Record outcome in `.agent/memory.md`; scoped commit.

## Completed roadmaps (status only; detail in git)
- **Clinical Pipeline E2E (2026-05-24) — 8/8:** COCO-WholeBody→MediaPipe mapping; rtmlib CSV export (the keystone — rtmlib had none, blocking R analysis); R-schema compat; R edge-case hardening; E2E smoke test; dep/security audit; main/run refactor (found not worthwhile); tech-notes drift audit.
- **Stability + Clinical Metrics + 3D Pipeline (2026-05-24 → 2026-06-08) — Phases 1–3, synthetic-validated:** P1 tracking stability (jitter/drops fix, movement-phase adaptive smoothing); P2 clinical metrics (bilateral, movement quality, trunk/torso, temporal segmentation); P3 3D pipeline (`fuse_session_frame`, ChArUco solve + calibration CLI, 3D CSV export + R analysis).

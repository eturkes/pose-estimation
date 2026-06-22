# Roadmap

Milestone ledger + active-milestone detail. Session MODE/flow lives in `/session-prompt`; finished detail + per-session prompts in git history.

Status: UNPLANNED → IN-PROGRESS → IMPLEMENTED → REVIEWED. Legacy pre-methodology milestones are DONE (never milestone-reviewed). Active = first milestone not DONE/REVIEWED. Milestones map to the current roadmap's Phases; future milestones draw from Backlog.

## Milestones
- **M1** — Phase 1 validation scaffolding — DONE (legacy; units 1A–1D in git, pre-methodology).
- **M2** — Phase 2 real-data 3D validation — ACTIVE, UNPLANNED, footage-gated. Plan when a real 3-camera session exists under `videos/`; until then a no-arg PLANNING session records the block and stops. Detail below.
- **M3+** — deferred. Sources in Backlog. Divide + plan after M2 is REVIEWED, using M2's commit range (esp. context-usage) to right-size units.

## Active: M2 — Real-World 3D Clinical Validation (outline, adopted 2026-06-15)

**Goal.** Prove the 3D clinical pipeline on *real* 3-camera recordings — not new features. A reproducible harness (calibration → 2D tracking → `world3d.csv` fusion → clinical metrics) reporting reprojection error, dropped/low-confidence keypoints, timing, and clinical-metric agreement; plus a capture/QA protocol, anonymized fixtures, failure-mode tests, and thresholds + a clinical-validity gap register.

**Carried caveat.** The entire 3D path (calibration solve, fusion, `world3d.csv`, 3D clinical metrics) is validated by **synthetic unit tests only**; real 3-camera footage has never run end-to-end. Closing that gap is the point of this milestone.

**Blockers.**
- **No real 3-camera session** under `videos/`. Single-camera clips are present (they drive the already-validated 2D pipeline); a synced 3-cam capture with `calibration.json` + `session.json` has never been recorded. Footage is expected, timing unknown.
- **No baseline** (known-geometry object / goniometer / second reference system) → clinical-metric *agreement* is unmeasurable. Until one exists, substitute internal evidence: reprojection error, cross-camera consistency, temporal stability, inter-trial repeatability.

**Outline to unit-plan (gated on footage; agreement also on a baseline).**
- 2A first real-capture dry run; calibrate protocol + `THRESHOLDS` against reality; record the deltas.
- 2B quantitative validation over multiple sessions (reprojection, drops, timing, stability, inter-trial repeatability).
- 2C clinical-metric agreement vs a baseline if one exists (Bland–Altman, ICC); else record the standing gap + cheapest-baseline plan; derive anonymized real fixtures; add a regression lock.
- reference: validation, calibration, multicam, analysis, environment
- **Gate:** a footage-gated unit with no real 3-cam session under `videos/` stops and reports the block — never fabricate results.

**Units.** UNPLANNED (see M2 ledger entry). When planned, one line per unit:
`M2.<u> <scope>: <desc> — <status> — ctx <pct> — <commit-tag>`

## Backlog (unscheduled — M3+ sources)
- 3D-aware downstream aggregation (`analysis/` is currently 2D-oriented).
- Multi-person cross-camera identity matching (fusion is single-person).
- Host-launch caveat: the container-native `.venv` won't resolve if the pipeline is launched from the host (e.g. NPU runs); would need a host-side `uv sync`.

## Maintenance cycle (recurring, roadmap-agnostic; run via explicit task)
reference: environment, architecture, tests
1. Python deps: `uv lock --upgrade` → `uv sync`; full suite green.
2. R deps: `renv::update()` → `renv::snapshot()`; R scripts exit 0.
3. Security: CLI injection vectors; session.json / calibration.json path traversal (`_safe_resolve` coverage); new CVEs in openvino, onnxruntime, opencv, rtmlib (web sweep).
4. `uv run pytest` / `ruff check` / `ruff format` / `ty check`.
5. Tech-notes drift: `.claude/tech/*.md` vs current code (module map, CLI flags, test inventory, API surface).
6. Record outcome in `.agent/memory.md`; scoped commit.

## Completed (status only; detail in git)
- **M1 / Phase 1 — validation scaffolding (2026-06-15→16):** 1A harness core (`validation.py`: `run_validation`, `ValidationReport`, `pose-estimation-validate`) + synthetic E2E fixture; 1B `THRESHOLDS` + `verdict()` PASS/WARN/FAIL (CI exit code) + clinical-validity gap register; 1C capture/QA protocol (`docs/capture_protocol.md`) + `qa_check` / `--qa-only` + anonymization; 1D failure-mode suite (`tests/test_validation_failuremodes.py`).
- **Clinical Pipeline E2E (2026-05-24) — 8/8:** COCO-WholeBody→MediaPipe mapping; rtmlib CSV export (keystone — rtmlib had none, blocking R analysis); R-schema compat; R edge-case hardening; E2E smoke test; dep/security audit; main/run refactor (found not worthwhile); tech-notes drift audit.
- **Stability + Clinical Metrics + 3D Pipeline (2026-05-24 → 2026-06-08) — Phases 1–3, synthetic-validated:** P1 tracking stability (jitter/drops fix, movement-phase adaptive smoothing); P2 clinical metrics (bilateral, movement quality, trunk/torso, temporal segmentation); P3 3D pipeline (`fuse_session_frame`, ChArUco solve + calibration CLI, 3D CSV export + R analysis).

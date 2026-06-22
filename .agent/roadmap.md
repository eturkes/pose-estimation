# Roadmap

Milestone ledger + active-milestone detail. Session MODE/flow lives in `/session-prompt`; finished detail + per-session prompts in git history.

Status: UNPLANNED → IN-PROGRESS → IMPLEMENTED → REVIEWED. Legacy pre-methodology milestones are DONE (never milestone-reviewed). Active = first milestone not DONE/REVIEWED. Milestones map to the current roadmap's Phases; future milestones draw from Backlog. Unit status: OPEN → DONE (BLOCKED = planned but footage/baseline-gated); a milestone is IMPLEMENTED once its units are all DONE.

## Quality gates
The "project's quality gates" that `/session-prompt` WORK-UNIT step 4 verifies (single source — the maintenance cycle reuses these). All four pass clean on a healthy tree.
- Python: `uv run ruff check` (lint) · `uv run ruff format --check` (format) · `uv run ty check` (type) · `uv run pytest` (tests — `filterwarnings=error`, so any unsilenced warning fails; see the two 2026-06-15 numpy/all-NaN memory lessons).
- R: touched `analysis/*.R` run clean under `Rscript` (exit 0); deps via renv (no separate lint/type gate) — after an R upgrade, `renv::snapshot()` (2026-05-24 lesson).
- "Touched scripts exit clean" = any changed entrypoint runs without error: Python console scripts (`pose-estimation`/`-run`/`-benchmark`/`-postprocess`/`-calibrate`/`-validate`) or R analysis scripts.

## Milestones
- **M1** — Phase 1 validation scaffolding — DONE (legacy). Units 1A–1D: `b765d5c` `13149a0` `20c36a0` `36f28a2`. No trace keys, no recorded context-usage — right-size M2 from its outline, not from M1 ctx.
- **M2** — Phase 2 real-data 3D validation — ACTIVE, UNPLANNED, footage-gated. Plan when a real 3-cam session is confirmed (resolve it via the `--list-sessions` probe — see the M2 Gate detail — not by reading deny-listed `videos/`); until then a no-arg PLANNING session records the block and stops. Planning context: outline seed `0dd62d9`, methodology restructure `c12ef18`. Detail below.
- **M3+** — UNPLANNED (not yet split). Sources in Backlog. After M2 is REVIEWED it becomes active → PLANNING splits + plans the next milestone, using M2's commit range (esp. context-usage) to right-size units.

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
- **Gate:** a footage-gated unit without a confirmed real 3-cam session stops and reports the block; report only results traced to real captures. Confirm footage functionally via the read-only discovery probe `uv run pose-estimation-run --list-sessions` (defaults to the `videos/` sessions root in source, so the command names no deny-listed path; `--sessions-dir <root>` / `--session-dir <dir>` / `--calibration <file>` override). Reads filenames + `session.json`/`calibration.json` — no frame decoding, so no video bytes enter context; it surfaces only an ordinal + camera count + calibration presence (the deny-listed tree's session ids / camera names and all frame + calibration values stay out of context). Prints `session #<i>: N cameras; calibration: present|absent` per session, exits 0 (≥1 found) / 1 (none/error). A `3 cameras` + `calibration: present` line marks a *candidate* session (shape only — discovery does not check decodability, sync, or frame counts) worth a full data-cleared decode/QA pass, not a verified-good capture. Read the probe's summary, never the videos.

**Units.** UNPLANNED (see M2 ledger entry). When planned, one line per unit:
`M2.<u> <scope>: <desc> — <OPEN|DONE|BLOCKED> — ctx <pct used/window> — <commit>`

## Backlog (unscheduled — M3+ sources)
- 3D-aware downstream aggregation (`analysis/` is currently 2D-oriented).
- Multi-person cross-camera identity matching (fusion is single-person).
- Host-launch caveat: the container-native `.venv` won't resolve if the pipeline is launched from the host (e.g. NPU runs); would need a host-side `uv sync`.

## Maintenance cycle (recurring, roadmap-agnostic; run via explicit task)
reference: environment, architecture, tests
1. Python deps: `uv lock --upgrade` → `uv sync`; full suite green.
2. R deps: `renv::update()` → `renv::snapshot()`; R scripts exit 0.
3. Security: CLI injection vectors; session.json / calibration.json path traversal (`_safe_resolve` coverage); new CVEs in openvino, onnxruntime, opencv, rtmlib (web sweep).
4. Run the Quality gates (top of file): all four Python gates green + touched scripts exit clean.
5. Tech-notes drift: `.claude/tech/*.md` vs current code (module map, CLI flags, test inventory, API surface).
6. Record outcome in `.agent/memory.md`; scoped commit.

## Completed (status only; detail in git)
- **M1 / Phase 1 — validation scaffolding (2026-06-15→16):** 1A harness core (`validation.py`: `run_validation`, `ValidationReport`, `pose-estimation-validate`) + synthetic E2E fixture; 1B `THRESHOLDS` + `verdict()` PASS/WARN/FAIL (CI exit code) + clinical-validity gap register; 1C capture/QA protocol (`docs/capture_protocol.md`) + `qa_check` / `--qa-only` + anonymization; 1D failure-mode suite (`tests/test_validation_failuremodes.py`).
- **Clinical Pipeline E2E (2026-05-24) — 8/8:** COCO-WholeBody→MediaPipe mapping; rtmlib CSV export (keystone — rtmlib had none, blocking R analysis); R-schema compat; R edge-case hardening; E2E smoke test; dep/security audit; main/run refactor (found not worthwhile); tech-notes drift audit.
- **Stability + Clinical Metrics + 3D Pipeline (2026-05-24 → 2026-06-08) — Phases 1–3, synthetic-validated:** P1 tracking stability (jitter/drops fix, movement-phase adaptive smoothing); P2 clinical metrics (bilateral, movement quality, trunk/torso, temporal segmentation); P3 3D pipeline (`fuse_session_frame`, ChArUco solve + calibration CLI, 3D CSV export + R analysis).

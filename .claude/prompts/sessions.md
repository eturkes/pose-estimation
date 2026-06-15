# Session prompts

Run the `/session-prompt` command (`.claude/commands/session-prompt.md`) to bootstrap a session. With no argument it continues the **Current roadmap** below, adopting the matching task block. Pass `/session-prompt <TASK>` to override with an ad-hoc task. Sessions are designed to run autonomously with minimal user input.

The **Session: Maintenance cycle** prompt at the bottom is roadmap-agnostic and reused by every roadmap's maintenance phase.

## Completed roadmaps

Per-session prompt blocks for finished roadmaps are pruned (recoverable from Git history); only the status checklist is kept.

### Clinical Pipeline E2E (2026-05-24) — all 8 tasks done

1. COCO-WholeBody -> MediaPipe keypoint mapping
2. Wire CSV export into rtmlib process_source()
3. Test rtmlib CSV schema compat with R pipeline
4. Harden R scripts for edge cases
5. E2E clinical pipeline smoke test
6. Dependency update + security audit
7. Refactor main.py/run.py (analysis: not worthwhile)
8. Tech notes drift audit

### Stability + Clinical Metrics + 3D Pipeline (2026-05-24 → 2026-06-08) — Phases 1–3 done, synthetic-validated

- Phase 1 (tracking stability): 1A jitter/drops fix ✓, 1B movement-phase adaptive smoothing ✓
- Phase 2 (clinical metrics): 2A bilateral ✓, 2B movement quality ✓, 2C trunk/torso ✓, 2D temporal segmentation ✓
- Phase 3 (3D pipeline): 3A fuse_session_frame ✓, 3B solve_charuco + calibration CLI ✓, 3C 3D CSV export + R analysis ✓
- Phase 4 (maintenance): periodic — folded forward into the current roadmap's maintenance phase.

**Caveat carried forward:** the entire 3D path (calibration solve, fusion, world3d.csv, 3D clinical metrics) is validated by **synthetic unit tests only**. Real 3-camera footage has never been run end-to-end. That gap is the subject of the current roadmap.

---

## Current roadmap: Real-World 3D Clinical Validation (2026-06-15)

**Goal.** Prove the 3D clinical pipeline works on *real* 3-camera recordings — not new features. Build a reproducible validation harness (calibration → 2D tracking → world3d.csv fusion, end-to-end) that reports reprojection error, dropped/low-confidence keypoints, timing, and clinical-metric agreement; add a capture/QA protocol, an anonymized fixture strategy, failure-mode tests, and tech notes carrying pass/fail thresholds plus the unresolved clinical-validity gaps.

**Status at seed time (2026-06-15).**
- **Footage: none yet.** No real 3-camera recordings exist under `rehab/data/videos`.
- **Baseline: none yet.** No external ground truth (no known-geometry object, manual goniometer measurement, or second reference system). Clinical-metric *agreement* is therefore currently **unmeasurable** — the headline clinical-validity gap. Until a baseline exists, "agreement" is replaced by internal evidence: reprojection error, cross-camera consistency, temporal stability, and inter-trial repeatability.

**Phase structure.**
- **Phase 1 (validation scaffolding) is footage-independent** — every task runs *now* against synthetic data (reuse `test_charuco`'s supersampled board renders + `test_multicam`'s synthetic projected CSVs). A `/session-prompt` with no args should pick up Phase 1 immediately.
- **Phase 2 (real-data validation) is gated on footage arriving** (and the agreement leg additionally on a baseline). When `/session-prompt` reaches a Phase 2 item and `rehab/data/videos` still holds no real session, **stop and report the block to the user** rather than fabricating results.
- **Phase 3 (maintenance)** is periodic; reuse the shared Maintenance cycle prompt.

### Phase 1: Validation scaffolding (footage-independent)
- 1A: Validation harness core + report schema + synthetic E2E fixture ✓
- 1B: Pass/fail thresholds + clinical-validity gap register ✓
- 1C: Capture/QA protocol + automated QA gate + anonymization strategy
- 1D: Failure-mode test suite

### Phase 2: Real-data validation (gated on footage; agreement also gated on a baseline)
- 2A: First real-capture dry run + protocol/threshold calibration
- 2B: Quantitative real-data validation (reprojection, drops, timing, stability, repeatability)
- 2C: Clinical-metric agreement + anonymized real fixtures + regression lock

### Phase 3: Maintenance (periodic, interleave freely)
- Dependency update + security audit + tech notes drift — see **Session: Maintenance cycle**.

---

## Session 1A: Validation harness core

```
Execute: Build the reproducible end-to-end validation harness that runs the full 3D clinical pipeline (calibration → 2D tracking → world3d.csv fusion → clinical metrics) on one session and emits a structured validation report. Validate it against synthetic data — no real footage required.

Load tech notes: architecture.md, entrypoints.md, multicam.md, calibration.md, analysis.md.

Context: The 3D pipeline (prior roadmap Phases 1–3) is synthetic-unit-tested only; no single command runs the whole chain on a session and reports quality. No real footage or external baseline exists yet (confirmed 2026-06-15), so the harness must (a) run fully on synthetic sessions now, and (b) make the clinical-metric "agreement" leg baseline-optional: when no reference is supplied, report internal self-consistency / temporal stability / cross-camera consistency instead of agreement error.

1. New module src/pose_estimation/validation.py exposing run_validation(session_dir, *, calibration=None, baseline=None, device=..., backend=...) -> ValidationReport. Reuse existing building blocks — do not reimplement: charuco.solve_charuco (when a calibration session is given and no calibration.json present), multicam.process_session + multicam.fuse_session_outputs, export.read_csv_keypoints, the world3d.csv they write, and the R clinical pipeline (analysis/clinical_features.R) for clinical metrics.
2. ValidationReport (dataclass + to_json + to_markdown) with sections:
   - calibration: reprojection_error_px (RMS from the solve), per-camera intrinsic summary, world_frame, camera count.
   - tracking_2d: per-camera frame count, detection rate, low-confidence keypoint fraction (below a configurable floor), dropped/empty-frame count.
   - fusion_3d (parsed from world3d.csv per-keypoint diagnostics): reproj_err_px distribution (median / p95 / max), n_views distribution, cheirality-violation rate, unfused-keypoint fraction.
   - timing: per-stage wall-clock (solve, 2D per camera, fusion, R metrics), throughput fps, device + backend.
   - agreement: baseline-optional. With a baseline → per-metric error vs reference. Without → self-consistency surrogates: bone-length coefficient-of-variation across frames, rest-period jitter, left/right symmetry on symmetric input. Also confirm the R pipeline consumes the fused world3d.csv and produces *_clinical_3d.csv.
3. Console script pose-estimation-validate (mirror the pyproject [project.scripts] pattern "pose_estimation.<module>:main", e.g. pose-estimation-calibrate / pose-estimation-benchmark). Flags: --session-dir, --calibration, --baseline, --device, --backend, --out report.json, --markdown.
4. Emit both report.json (CI-parseable) and a human-readable markdown summary. Report raw numbers only here — the PASS/FAIL verdict arrives in 1B.
5. Synthetic E2E fixture: reuse test_charuco's 3× supersampled warp render for a calibration session and test_multicam's synthetic projected CSVs (or render a synthetic moving skeleton) so the harness has deterministic input. Add tests/test_validation.py: harness runs end-to-end on the synthetic session; report has all sections; numbers are finite and within sane synthetic ranges.
6. Update tech notes: create .claude/tech/validation.md (harness overview, report schema, how to run), add its row to .claude/INDEX.md, cross-link from architecture.md + entrypoints.md. Regenerate the repo map (python scripts/repomap.py) and run uv run ruff check --fix / ruff format / ty check / pytest.

Deliverable: one command produces a full validation report on a synthetic session; suite green.
```

---

## Session 1B: Pass/fail thresholds + clinical-validity gap register

```
Execute: Define acceptance thresholds for every validation metric and an honest register of clinical-validity gaps; encode them so the harness renders PASS/WARN/FAIL.

Load tech notes: validation.md, multicam.md, calibration.md, analysis.md, optimization.md.

Prerequisite: 1A (harness + report schema).

Context: The harness reports raw numbers; clinical acceptance needs documented thresholds with rationale and an explicit register of what cannot yet be validated. No external baseline exists (confirmed 2026-06-15), so clinical-metric agreement is currently unmeasurable — that is the headline gap.

1. Define THRESHOLDS as a single-source-of-truth versioned dataclass/dict in validation.py for: calibration reprojection RMS (px), per-keypoint fusion reproj_err_px median + p95, min n_views, confidence floor + max low-confidence fraction, max unfused-keypoint fraction, max cheirality-violation rate, timing budget (min fps), and metric-agreement tolerance (used only when a baseline exists). Cite clinical / photogrammetry literature for each value where possible (web search); record the rationale inline and in validation.md.
2. Map each threshold to PASS / WARN / FAIL bands. Add ValidationReport.verdict() grading a report against THRESHOLDS, surfaced in to_json + to_markdown and as the pose-estimation-validate exit code (0 pass / nonzero fail) for CI.
3. Author the clinical-validity gap register in validation.md: per clinical metric, its current validation status. Mark explicitly: "clinical-metric agreement vs ground truth: UNVALIDATED (no baseline)"; 2D-foreshortening limits on bone-length constraints; world-frame-level assumption underlying trunk metrics; single-person-only fusion; sync model is software-only.
4. Tests: verdict() grades known-good and known-bad synthetic reports correctly; exit code matches verdict.
5. Update validation.md (thresholds table + gap register) and cross-link optimization.md (tunables ↔ thresholds). Regenerate repo map; run ruff / ty / pytest.
```

---

## Session 1C: Capture/QA protocol + automated QA gate + anonymization strategy

```
Execute: Write the capture/QA protocol for real 3-camera recording, an automated QA gate that grades a capture before clinical metrics are trusted, and the anonymized-fixture strategy.

Load tech notes: validation.md, calibration.md, multicam.md.

Prerequisite: 1B (thresholds).

Context: Real captures will be produced against this protocol. A capture that fails QA (poor calibration coverage, desync, low detection) must be caught early.

1. Author docs/capture_protocol.md: physical setup (3-camera placement and field-of-view overlap, working volume, frame rate, lighting, shutter/sync method), ChArUco capture procedure (move the board through the full working volume with translation AND tilt diversity; keep ≳ 25 px/square; the world-frame camera must co-see the board with every other camera simultaneously — see calibration.md topology limit), the subject task script for clinical trials, and a per-capture acceptance checklist.
2. Implement qa_check(session_dir) in validation.py grading a raw capture against QA thresholds: calibration RMS, board-pose coverage/spread, per-camera ChArUco detection rate, frame-count parity across cameras (desync proxy), per-camera 2D detection rate on the subject clip → PASS / WARN / FAIL with reasons. Wire pose-estimation-validate --qa-only.
3. Anonymization strategy section (in docs/capture_protocol.md; also record a decision entry): raw patient video is never committed (videos/ is already gitignored; calibration.json + calibration/ are gitignored by default, so a vetted de-identified calibration fixture needs a scoped `!` negation or `git add -f`); shareable fixtures are de-identified derived artifacts only — per-camera keypoint CSVs + calibration.json + world3d.csv (coordinates, no imagery); any retained imagery must be synthetic (ChArUco board / synthetic skeleton render), never patient frames; strip capture metadata; derive fixtures only from consented recordings.
4. Tests: qa_check flags a synthetic bad capture (sparse board poses, camera frame-count mismatch) and passes a good one.
5. Update tech notes + INDEX (note docs/ if newly added). Regenerate repo map; run ruff / ty / pytest.
```

---

## Session 1D: Failure-mode test suite

```
Execute: Add a failure-mode test suite proving the harness detects and reports degraded inputs rather than silently producing plausible-but-wrong output.

Load tech notes: validation.md, multicam.md, tests.md.

Prerequisite: 1A–1C.

Context: Clinical safety depends on the harness surfacing problems. Each test injects a known degradation into a synthetic session and asserts the corresponding report field crosses its threshold and the verdict goes WARN/FAIL. Phrase every assertion as the harness correctly identifying the fault.

Add tests/test_validation_failuremodes.py with one test per fault:
1. Camera dropout — one camera missing frames → unfused-keypoint fraction rises above threshold; report flags it; no crash.
2. Miscalibration — perturb one camera's extrinsics → fusion reproj_err_px p95 exceeds threshold → FAIL.
3. Desync — apply a wrong sync_offset → reprojection / temporal coherence degrades → flagged.
4. Low confidence — scale all confidences below the floor → low-confidence fraction exceeds threshold; gating to NaN works.
5. Occlusion — zero a body region in one camera → that region relies on remaining views or reports NaN, never garbage.
6. Degenerate calibration — tiny baseline / near-collinear cameras → ill-conditioned triangulation flagged via high reproj or instability.

Update tests.md inventory. Run ruff / ty / pytest.
```

---

## Session 2A: First real-capture dry run

```
Execute: Run the first real 3-camera capture through the protocol and harness end-to-end; calibrate protocol and thresholds against reality.

Load tech notes: validation.md, calibration.md, multicam.md, environment.md.

Prerequisite: real footage exists under rehab/data/videos; Phase 1 complete. If no real session exists yet, stop and report the block to the user.

Context: First contact with real data. Goal is a clean end-to-end pass and realistic thresholds, not clinical conclusions yet.

1. Ingest one real 3-camera subject session + one ChArUco calibration session captured per docs/capture_protocol.md.
2. Run pose-estimation-validate --qa-only; resolve any QA failures (recapture if needed).
3. Solve calibration (pose-estimation-calibrate solve); run full pose-estimation-validate; review report.json + markdown.
4. Compare real numbers to the Phase-1 synthetic thresholds; adjust validation.py THRESHOLDS to realistic-but-strict values with rationale; record the deltas and reasoning as a decision entry.
5. Fold any protocol gaps discovered (lighting, sync, board coverage) back into docs/capture_protocol.md.
6. Keep raw patient video out of Git; note only where it lives (rehab/data). Run ruff / ty / pytest.
```

---

## Session 2B: Quantitative real-data validation

```
Execute: Quantify pipeline quality across multiple real sessions — reprojection, drops/low-confidence, timing, and stability/repeatability.

Load tech notes: validation.md, multicam.md, analysis.md, optimization.md.

Prerequisite: 2A complete; multiple real sessions available.

Context: No external baseline yet (confirmed 2026-06-15), so agreement is established via internal evidence: reprojection error, cross-camera consistency, temporal stability, and inter-trial repeatability.

1. Run the harness over the available real sessions; aggregate report metrics.
2. Reprojection + drops: report the distribution of fusion reproj_err_px, n_views, and unfused/low-confidence rates per session and pooled; compare to thresholds.
3. Timing: real throughput per device (NPU/CPU/GPU) and backend (onnxruntime/openvino); confirm the fps budget.
4. Stability: temporal jitter during rest periods; cross-camera 3D consistency via bone-length coefficient-of-variation.
5. Repeatability: inter-trial ICC / CoV of key clinical metrics across repeated identical trials by the same subject — the strongest evidence available absent a baseline.
6. Write a real-data results section into validation.md; flag any metric failing its threshold for follow-up. Run ruff / ty / pytest.
```

---

## Session 2C: Clinical-metric agreement + anonymized fixtures + regression lock

```
Execute: Establish clinical-metric agreement against a baseline if one now exists; derive anonymized real fixtures; lock validated behavior with a regression test.

Load tech notes: validation.md, analysis.md, tests.md.

Prerequisite: 2B complete.

Context: As of 2026-06-15 no baseline exists. If a baseline (known-geometry object, manual goniometer, or second reference system) has since materialized, compute agreement; otherwise record the unresolved clinical-validity gap and proceed with fixtures + regression lock on self-consistency.

1. If a baseline is available: compute per-metric agreement (Bland–Altman bias + limits-of-agreement, ICC, %error) for joint angles / reach distance / etc. vs the reference; compare to the agreement-tolerance threshold. Else: record "clinical-metric agreement UNVALIDATED — no baseline" in the gap register and add an acquisition plan (cheapest viable baseline: a known-length rod for absolute scale + a goniometer for ≥ 1 joint).
2. Anonymized fixtures: from one QA-passing real session, derive de-identified fixtures per the 1C strategy (per-camera keypoint CSVs + calibration.json + world3d.csv, no imagery); store under tests/fixtures/ (committed); confirm no biometric imagery is present.
3. Regression lock: add a real-fixture regression test asserting the harness reproduces the validated report within tolerance, so future changes cannot silently regress validated behavior.
4. Update validation.md (agreement results or the standing gap) + tests.md. Run ruff / ty / pytest.
```

---

## Session: Maintenance cycle

```
Execute: Periodic maintenance — dependency updates, security audit, tech notes drift check.

Load tech notes: environment.md, architecture.md, tests.md.

1. Update Python dependencies: uv lock --upgrade, then uv sync. Verify all tests pass.
2. Update R packages: renv::update(), then renv::snapshot(). Verify R scripts exit 0.
3. Security audit:
   - Review CLI argument handling for injection vectors.
   - Review session.json / calibration.json parsing for path traversal (recheck _safe_resolve coverage).
   - Check for new CVEs in openvino, onnxruntime, opencv, rtmlib.
4. Run full test suite: uv run pytest. Run linter: uv run ruff check. Run type checker: uv run ty check.
5. Tech notes drift audit: verify all .claude/tech/*.md files match current code (module map, CLI flags, test inventory, API surface).
6. Update .claude/memory/scratchpad.md with maintenance log.
7. Commit all changes.
```

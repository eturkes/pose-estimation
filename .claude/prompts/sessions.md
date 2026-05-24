# Session prompts for Clinical Pipeline E2E roadmap

Paste the kickoff prompt from `prompts/kickoff.md`, then append one of the task blocks below as the `<TASK>` section. Sessions are designed to run autonomously with minimal user input.

## Dependency graph

```
#1 → #2 → #3 → #4  (can parallel with #5)
                └→ #5
      └→ #7
#6  (independent, anytime)
#8  (independent, anytime — best after other tasks land)
```

---

## Session 1: COCO-WholeBody → MediaPipe keypoint mapping

```
Execute Task #1: Add COCO-WholeBody → MediaPipe keypoint mapping layer.

Load tech notes: architecture.md, tracking-modes.md.

Create a mapping module (or extend export.py) that translates rtmlib's COCO-WholeBody 133-keypoint layout to the existing MediaPipe CSV schema. Key index mappings:

COCO-WholeBody 133 layout: body(0-16), feet(17-22), face(23-90), left_hand(91-111), right_hand(112-132).

For `hands-arms` mode (12 ARM_KEYPOINT_NAMES + 2×21 hands):
- Shoulders: COCO 5→left_shoulder, 6→right_shoulder
- Elbows: COCO 7→left_elbow, 8→right_elbow
- Wrists: COCO 9→left_wrist, 10→right_wrist
- Finger bases: left hand MCP joints at indices 91+5, 91+9, 91+17; right at 112+5, 112+9, 112+17
- Hands: COCO 91-111→left hand (21 kps), 112-132→right hand (21 kps)

For `body` mode (33 BODY_KEYPOINT_NAMES + 2×21 hands):
- Direct COCO→MP mappings exist for: nose(0→0), left_eye(1→2), right_eye(2→5), left_ear(3→7), right_ear(4→8), shoulders(5-6→11-12), elbows(7-8→13-14), wrists(9-10→15-16), hips(11-12→23-24), knees(13-14→25-26), ankles(15-16→27-28)
- Feet: COCO 17→left_foot_index(31), 20→right_foot_index(32), 19→left_heel(29), 22→right_heel(30)
- MediaPipe-only keypoints (eye_inner/outer, mouth_left/right, pinky/index/thumb at wrist): fill from nearest COCO face keypoints where feasible, or leave as NaN/zero with low visibility
- Hands: same as hands-arms

For 17-keypoint RTMPose-M: map to body mode with NaN-filled feet/face-derived keypoints and no hands.

Visibility: use rtmlib per-keypoint scores directly as the `vis` column.

Provide the mapping as a function: `coco_to_mediapipe(keypoints, scores, n_kps, tracking) → (body_landmarks, body_visibilities, hand_landmarks, matches)` matching frame_to_rows() signature.

Tests: verify mapping output shapes, verify round-trip through make_csv_header/frame_to_rows for each mode.
```

---

## Session 2: Wire CSV export into rtmlib

```
Execute Task #2: Wire CSV export into rtmlib process_source().

Load tech notes: architecture.md, entrypoints.md, tracking-modes.md.
Read scratchpad.md top for Task #1 completion notes.

Prerequisites: Task #1 (keypoint mapping) must be complete.

Wire CSV output into run.py:
1. Add `output_csv: str | None = None` parameter to process_source(). When provided, open_csv_writer() at start, frame_to_rows() each frame using the mapping from Task #1, close at end.
2. Note: rtmlib keypoints are pixel-space (x, y), not normalized. frame_to_rows() expects pixel-space landmarks and normalizes internally (divides by frame_w/frame_h). Verify this matches.
3. The rtmlib smoother outputs (keypoints, scores) after smoothing — use the smoothed values for export.
4. Wire into single-source mode: add --output-dir flag to run.py argparse, derive CSV path from video name.
5. Wire into batch-dir mode: CSV per video in output dir.
6. Wire into session mode: the _dispatch_sessions callback already receives output_csv kwarg from process_session() — pass it through to process_source().
7. Preserve backward compat: CSV export is opt-in (only when --output-dir is specified or in session mode).
8. Run full test suite. Add a test that invokes process_source with a synthetic video and verifies CSV output.
```

---

## Session 3: R pipeline compatibility testing

```
Execute Task #3: Test rtmlib CSV export schema compatibility with R analysis pipeline.

Load tech notes: analysis.md, tracking-modes.md.
Read scratchpad.md top for Task #1/#2 completion notes.

Prerequisites: Tasks #1 and #2 must be complete.

1. Generate synthetic CSVs using the rtmlib export path: create a short synthetic video (MJPG/AVI, ~30 frames), run it through process_source() with CSV export enabled, collect the output CSV.
2. Alternatively, generate CSVs programmatically using the mapping + export functions directly (faster, no model needed).
3. Run clinical_features.R on the synthetic CSV. It needs columns: arm_left_shoulder_x/y/z, arm_left_elbow_x/y/z, arm_left_wrist_x/y/z, left_hand_*_x/y/z (and right equivalents). Verify it produces *_clinical.csv and *_clinical_windows.csv.
4. Run summary.R on a synthetic *_metrics.csv (or skip if metrics export isn't wired for rtmlib — note this gap).
5. Document any schema mismatches found and fix them.
6. Add pytest tests verifying CSV header correctness for each tracking mode × backend combination.
```

---

## Session 4: R script hardening

```
Execute Task #4: Harden R analysis scripts for edge cases and both-backend input.

Load tech notes: analysis.md.

Prerequisites: Task #3 must be complete.

Systematically test every R script in analysis/ with synthetic data covering edge cases:
- Short videos (<10 frames)
- Missing hand data (all-blank hand columns)
- Single-person vs multi-person
- Single-video vs multi-video input directories
- Videos shorter than SAL window size (clinical_features.R)
- Metadata CSVs with subset of videos (correlation/longitudinal scripts)

For each script that fails, fix defensively (e.g., skip gracefully with a warning rather than crash).
Update analysis.md with any behavior changes.
```

---

## Session 5: E2E smoke test

```
Execute Task #5: Create end-to-end clinical pipeline smoke test.

Load tech notes: architecture.md, analysis.md, tests.md.

Prerequisites: Task #3 must be complete.

Build an automated test (pytest + standalone script) that chains:
1. Generate synthetic landmark CSV (using export functions directly — no model inference needed)
2. Run clinical_features.R → verify *_clinical.csv exists and has expected columns
3. Optionally run summary.R or validate_metadata.R

Use pytest markers to allow skipping when R is unavailable.
Guard against pipeline-level regressions (column renames, normalization changes).
```

---

## Session 6: Dependency update

```
Execute Task #6: Dependency update + security audit.

Load tech notes: environment.md.

Independent task — can run anytime.

1. `uv lock --upgrade` and review changes for breaking versions.
2. `uv run pytest` — full suite green.
3. `uv run ruff check --fix && uv run ruff format`.
4. Check model download URLs still valid (test_models_checksum.py).
5. R: `renv::update()` then `renv::snapshot()`.
6. Update environment.md with version changes.
```

---

## Session 7: Refactor main.py / run.py

```
Execute Task #7: Proactive codebase refactor — reduce main.py / run.py duplication.

Load tech notes: architecture.md, entrypoints.md.

Prerequisites: Task #2 must be complete (CSV export wired, so refactor doesn't need to touch moving parts).

Extract shared patterns: CLI arg groups, video capture loop, pygame display, batch-dir iteration, progress reporting. Keep entry points distinct (main.py = MediaPipe init; run.py = rtmlib init).

Verify both CLI tools work identically before and after. Run full test suite + smoke test.
Update architecture.md module map.
```

---

## Session 8: Tech notes audit

```
Execute Task #8: Sync tech notes and memory files with current codebase state.

Load all tech notes.

Independent task — best run after other tasks have landed so the audit catches everything.

Verify every file:line reference. Verify module map. Verify CLI flags. Verify column counts. Prune stale scratchpad entries. Update kickoff prompt if conventions changed.
```

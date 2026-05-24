# Scratchpad

Transient working notes — anything from "current investigation" to "half-finished idea". Safe to prune at session boundaries. Never treat as a source of truth; promote stable findings to `tech/`, decisions to `decisions.md`, lessons to `lessons.md`.

## How to use

- Start each session by skimming the most-recent entries (top of file).
- Append `## YYYY-MM-DD HH:MM — <session topic>` and write freely.
- When closing a session, either prune the entry or summarise it into a longer-lived file.

---

## 2026-05-24 — Tasks #1 and #2 complete; #3 next

Completed roadmap tasks #1 (keypoint mapping) and #2 (CSV export wiring) in this session.

**Task #1 result:** `mapping.py` with `coco_to_mediapipe()` — translates rtmlib (n_persons, n_kps, 2) + scores to frame_to_rows() interface. Supports 133-kp wholebody (RTMW-L, DWPose-M) and 17-kp body-only (RTMPose-M) across all tracking modes. 28 tests.

**Task #2 result:** `process_source()` now accepts `output_csv`/`video_name` params. Each frame: smoothed keypoints → coco_to_mediapipe → frame_to_rows → CSV. Wired into all dispatch modes (single-source, batch-dir, session). `--output-dir` flag added to run.py argparse. 9 tests.

**Next session (Task #3):** R pipeline compatibility. Need `renv::restore()` first (deps not installed on host). Then run `clinical_features.R` on a synthetic CSV from the rtmlib path. R 4.6.0 is available, renv bootstrapped, but tidyverse packages missing.

Key fact for Task #3: `clinical_features.R` accesses columns via `body_col(tracking, side, keypoint, coord)` which produces e.g. `arm_left_shoulder_x`. Our export uses `ARM_KEYPOINT_NAMES` which produces identical column names. Also accesses `left_hand_0_x` (hand wrist = landmark 0) — matches our `_fill_hand_side` output. Schema should be compatible; needs verification run.

---

## 2026-05-24 — Clinical Pipeline E2E roadmap (8 tasks)

User confirmed: 3-cam footage still incoming, calibration solver deferred. Priority is **Clinical Pipeline E2E**: video → CSV → clinical features → longitudinal analysis.

Critical blocker identified: rtmlib backend (RTMW-L 133-keypoint, the default and most capable model) has **zero CSV export**. The entire R analysis pipeline is unreachable from rtmlib-processed footage.

Roadmap (dependency order):
1. COCO-WholeBody → MediaPipe keypoint mapping layer (unblocked)
2. Wire CSV export into rtmlib process_source() (blocked by #1)
3. Test rtmlib CSV schema compat with R pipeline (blocked by #2)
4. Harden R scripts for edge cases + both backends (blocked by #3)
5. E2E clinical pipeline smoke test (blocked by #3)
6. Dependency update + security audit (independent)
7. Proactive refactor: main.py/run.py dedup (blocked by #2)
8. Tech notes drift audit (independent)

Tasks #4/#5 can run in parallel after #3. Tasks #6/#8 are independent and can run anytime.

Keypoint mapping notes (for Task #1):
- COCO-WholeBody 133: body(0-16) + feet(17-22) + face(23-90) + left_hand(91-111) + right_hand(112-132)
- hands-arms mode needs: shoulders(5,6), elbows(7,8), wrists(9,10) from COCO body, finger bases from hand (MCP joints at hand offsets 5,9,17), plus full 21-kp hand ranges
- body mode: COCO body 0-16 maps to ~20 of MediaPipe's 33; feet 17-22 partially cover the rest; some MP keypoints (eye_inner/outer, mouth_left/right) need face keypoint mapping or NaN fill
- RTMPose-M (17 kps): body-only, maps directly to COCO body 0-16 subset

---

## 2026-05-24 — process_session() wired; 7-task roadmap active

Implemented `process_session()` with callback-based design. Both `main.py` and `run.py` now construct backend-specific camera processor closures and pass them to `process_session()`. 4 new tests (replaced 1 stub test). All 189 tests pass.

Key design insight: `run.py` session dispatch had to move from before model setup (line 976) to after it (line 1044) so the PoseTracker and smoother are available for the callback.

Observation: the rtmlib path (`run.py`) has no CSV export for any mode (single-source or multi-cam). Adding CSV export to the rtmlib path would be a valuable enhancement but is separate from the multi-cam orchestration work.

Next session: Task #2 (solve_charuco) or Task #4 (fuse_session_frame) — both are partially unblocked. Task #2 is independent; Task #4 depends on #1 (now complete).

---

## 2026-05-24 — Jitter/drop fixes shipped; multi-cam roadmap created

Shipped jitter + detection drop fixes (commit `231b609`). Three changes:
1. Velocity-aware outlier rejection in OneEuroFilter (outlier_cap=30px default)
2. Hand min_cutoff 1.0→0.5, detection EMA alpha 0.5→0.35
3. Multi-frame detection carry 1→3 frames with velocity prediction

Created full development roadmap (7 tasks). Next session: pick up task #1 (process_session) or #2 (solve_charuco) — both are unblocked. User confirmed single-cam quality issues were jitter + detection drops (now addressed). 3-cam footage is still incoming; all dev/test can proceed on synthetic data.

Validation needed: these fixes should be tested on real clinical footage to confirm improvement. If jitter/drops persist, next levers are: rtmlib carry_frames 5→10 (match MediaPipe), recovery blending after detection gaps, ROI expansion during carry.

---

## 2026-05-16 — `.claude/` infrastructure seeded

Initial setup of `.claude/` (INDEX, tech/, memory/, prompts/) and migration off `AGENTS.md`. No code changes. Watch for future drift between `.claude/tech/*.md` and the code itself; the `test_public_api.py` test guards `__init__.py` re-exports but nothing else guards the other tech notes.

Open follow-ups (not yet decisions):
- Consider a lightweight CI/precommit doc-drift check (e.g. assert key file paths mentioned in `tech/*.md` exist).
- `analysis_summary.html` is committed and large (~5 MB); future audit can decide if it should be tracked via Git LFS or moved to a release artifact.

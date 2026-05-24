# Scratchpad

Transient working notes — anything from "current investigation" to "half-finished idea". Safe to prune at session boundaries. Never treat as a source of truth; promote stable findings to `tech/`, decisions to `decisions.md`, lessons to `lessons.md`.

## How to use

- Start each session by skimming the most-recent entries (top of file).
- Append `## YYYY-MM-DD HH:MM — <session topic>` and write freely.
- When closing a session, either prune the entry or summarise it into a longer-lived file.

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

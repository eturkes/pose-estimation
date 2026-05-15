# Lessons learned

Append-only. Each lesson should yield a positive, actionable rule (avoid "do not" phrasings — see CLAUDE.md on the pink-elephant problem).

## Entry schema

```
## YYYY-MM-DD — <short title>

**Symptom.** What went wrong or wasted effort.
**Root cause.** Why it happened.
**Rule (positive form).** What to do going forward; phrase as "always X" / "first X then Y".
**Where to check.** File paths or tests that encode the rule, if any.
```

---

## 2026-05-16 — Treat `AGENTS.md` as drifting unless tests guard it

**Symptom.** `AGENTS.md` listed `test_extrapolation.py` at repo root and omitted `run.py`, the rtmlib registry, the expanded test suite, `_types.py`, the `scripts/benchmarks/` micro-bench suite, `analysis_summary.Rmd`, and `analysis/utils.R`. Multiple commits had landed without updating it.
**Root cause.** No mechanical check links docs to code, so an out-of-band markdown file silently rots.
**Rule (positive form).** Always extract project-specific tech notes from code at audit time, not from prior `AGENTS.md` versions. Prefer file path + `:line` references that an agent will detect when stale. When adding a new module, public API export, or top-level script in the same commit also touch the matching `.claude/tech/*.md` file.
**Where to check.** `.claude/tech/architecture.md` (module map), `.claude/tech/tests.md` (test inventory), `tests/test_public_api.py` (asserts package surface).

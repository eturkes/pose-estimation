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

## 2026-05-24 — Always validate resolved paths stay within the expected directory

**Symptom.** Security audit found `(base / user_ref).resolve()` in multicam.py accepts `../` traversal in session.json manifest fields.
**Root cause.** pathlib's `/` operator followed by `.resolve()` happily escapes the base directory. Missing containment check.
**Rule (positive form).** Always validate that resolved relative paths stay within their base directory before using them. Use a helper: resolve, then check `str(resolved).startswith(str(base_resolved) + "/")`.
**Where to check.** `multicam.py:_safe_resolve`, `tests/test_multicam.py:test_*_path_traversal_*`.

---

## 2026-05-24 — R 4.6 broke the C API: always use renv::snapshot() after upgrading R

**Symptom.** `renv::restore()` failed because locked package versions (for R 4.5) used C functions removed in R 4.6.0.
**Root cause.** The lockfile was created under R 4.5; R 4.6 changed the C API (`Rf_findVar`, `Rf_allocSExp` removed). Locked versions of Matrix, magrittr, backports, rlang all failed to compile.
**Rule (positive form).** After upgrading R, always install packages at latest CRAN versions first, then `renv::snapshot()` to update the lockfile. Use `renv::record("pkg@version")` to fix individual lockfile entries when needed.
**Where to check.** `renv.lock` (R version field), `.claude/tech/environment.md`.

---

## 2026-05-24 — R `if_else()` requires vector-length arguments in dplyr 1.2+

**Symptom.** `longitudinal.R` crashed with `vctrs::stop_recycle_incompatible_size` when `if_else()` received a scalar condition and vector-length true/false branches.
**Root cause.** dplyr 1.2+ delegates to `vctrs::vec_if_else()` which enforces strict length matching. The old behavior (scalar recycling) no longer works.
**Rule (positive form).** Always use base R `if()`/`else` for scalar conditions inside `mutate()`, reserving `if_else()` for vectorized element-wise branching.
**Where to check.** `analysis/longitudinal.R:65-72`.

---

## 2026-05-16 — Treat `AGENTS.md` as drifting unless tests guard it

**Symptom.** `AGENTS.md` listed `test_extrapolation.py` at repo root and omitted `run.py`, the rtmlib registry, the expanded test suite, `_types.py`, the `scripts/benchmarks/` micro-bench suite, `analysis_summary.Rmd`, and `analysis/utils.R`. Multiple commits had landed without updating it.
**Root cause.** No mechanical check links docs to code, so an out-of-band markdown file silently rots.
**Rule (positive form).** Always extract project-specific tech notes from code at audit time, not from prior `AGENTS.md` versions. Prefer file path + `:line` references that an agent will detect when stale. When adding a new module, public API export, or top-level script in the same commit also touch the matching `.claude/tech/*.md` file.
**Where to check.** `.claude/tech/architecture.md` (module map), `.claude/tech/tests.md` (test inventory), `tests/test_public_api.py` (asserts package surface).

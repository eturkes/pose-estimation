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

## 2026-06-04 — When judging a split seam, audit every free name in the block, not a hand-picked list

**Symptom.** While splitting `run.py` I also scoped extracting the cropping helpers from `processing.py` into a new module. It failed late: `get_hand_crop` references `PALM_WRIST_KP_IDX`/`PALM_FINGER_KP_IDX`, which five other `processing.py` sites and a test also use, so extraction forces a circular import (new module ↔ `processing.py`). I abandoned that split after scoping it.
**Root cause.** I vetted a hand-picked list of constants the function "obviously" needed instead of enumerating every free name (globals, constants, helpers) referenced inside the candidate block. The coupling lived in names I had not listed.
**Rule (positive form).** Before extracting a block, first enumerate every free name it references (walk the AST or `rg` each global) and confirm each either moves with the block or stays importable one-directionally. Treat any name shared with the source module's other sites as coupling that blocks a clean extraction; split only when the seam is acyclic.
**Where to check.** `.claude/memory/decisions.md` (2026-06-04 token-efficiency program), `src/pose_estimation/processing.py` (`PALM_*_KP_IDX` shared sites).

---

## 2026-06-04 — Repair venv absolute paths after a project move; never byte-edit .pyc/.so

**Symptom.** After the project was relocated (`~/Documents/pro/pose-estimation` → `~/Projects/pose-estimation`), `import pose_estimation` raised `ModuleNotFoundError` and every `.venv/bin/*` console script (pytest, pose-estimation, coverage) had a dead shebang.
**Root cause.** A uv `.venv` hardcodes the project's absolute path in `bin/*` shebangs, `activate*` (`VIRTUAL_ENV`), and the editable `_editable_impl_*.pth` (→ old `src/`). Moving the directory invalidates all of them. Stale paths also linger harmlessly in regenerable caches (`*.pyc` `co_filename`, `.ruff_cache`) and as cosmetic build strings in renv `.so` — none of which break loading.
**Rule (positive form).** After a move, first rewrite old→new in `.venv` **text** files only (shebangs, `activate*`, editable `.pth`, `direct_url.json`) and clear `__pycache__`/`.ruff_cache`; always skip `*.pyc`/`*.so` (path lengths differ → in-place edit corrupts the binary). Enumerate with `find -exec grep` or Python, since the shell's `grep` is a profile **function** that prunes dot-dirs (so `grep -r .venv` silently finds nothing). Verify with `import pose_estimation`, a console script, `pytest`, `Rscript -e 'renv::project()'`. Canonical alternative: `uv sync` on the host.
**Where to check.** `.claude/tech/environment.md` (Relocation section).

---

## 2026-06-01 — Reinstall R graphics sysreqs after container recreation

**Symptom.** After a Distrobox container rebuild, `library(ragg)` failed (`libwebpmux.so.3: cannot open shared object file`) and `renv::restore()` warned that `libfontconfig1-dev`/`libfreetype6-dev` were missing — though the 2026-05-24 R migration had installed them.
**Root cause.** apt-installed system packages live in the container, not the project. Recreating the container drops them while the project-local renv library survives, leaving its pre-compiled `.so` files (e.g. `ragg.so`) unable to find their runtime libs.
**Rule (positive form).** After a container recreation, first reinstall the documented R-graphics sysreqs (the apt list in the 2026-05-24 "R environment migrated" decision), then verify with `Rscript -e 'library(ragg); library(ggplot2)'`. A durable fix is a project setup script encoding that list.
**Where to check.** `.claude/memory/decisions.md` (2026-05-24 R migration — canonical apt list), `.claude/tech/environment.md`.

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

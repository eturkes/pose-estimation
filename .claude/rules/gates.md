# Gate invocation

## The prefix — every gate command in this project

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <cmd>
```

Use both halves; `PYTHONPATH` is the half that does the work. Measured 4 ways, `pytest --collect-only -q tests/test_public_api.py` in the primary tree under an interactive shell's inherited environment:

| form | rc | outcome |
| --- | --- | --- |
| bare `uv run --no-sync` | 4 | `ImportError … GLIBC_2.43 not found` (host `libopenvino.so.<ver>`) |
| `env -u LD_LIBRARY_PATH` alone | 4 | `ImportError: libopenvino.so.<ver>: cannot open shared object file` |
| `PYTHONPATH="$PWD/src"` alone | 0 | 12 collected |
| both | 0 | 12 collected |

The cause is the **inherited `PYTHONPATH`**, which names the host OpenVINO python package built against glibc 2.43; pointing `PYTHONPATH` at `src` displaces it, so `tests/conftest.py` imports the `.venv` wheel instead and collection proceeds. Unsetting `LD_LIBRARY_PATH` keeps the matching host runtime out of the loader path and **cannot substitute** — alone it strips the libraries while leaving the host package importable, which is the second row. Keep one form and carry it verbatim in every teammate brief; a partially-remembered prefix costs a window (`scripts/gate.sh` port → `.agent/deferred.md`). The worktree recipe below therefore runs green without `env -u`: it exports `PYTHONPATH` itself.

- **Full suite ≈ 28 min (2129 tests).** Always `run_in_background: true` + redirect to a log, then work while it runs — a background command piped through `tail` buffers everything until exit.
- **Run the decisive gate alone.** `tests/test_r_timebase_truth.py::test_c8_08` runs the WHOLE suite in a subprocess (`pytest -q --maxfail=1 -k "not test_c8_08"`, cwd = project root), asserts rc=0 and reconciles the pass count against a separate collection. Five concurrent worktree gates plus MAIN's on 8 cores drove it past its timeout — that is the whole difference between `1701 passed / 1 failed` and green. Never run it beside a decode, an inference sweep or a mutation campaign; brief reviewers to targeted files and keep the full-suite slot for MAIN.
- **One red file anywhere under `tests/` takes `test_c8_08` down** and with it the decisive gate for the whole unit. Measured: red file present → 59 failed / 1643 passed; absent → 1642 passed. So a diff-blind or reviewer red suite lives on its `archive/*` tag (→ `retention.md`), and the implementing unit restores it with `git show <tag>:tests/<file> > tests/<file>`, drives it green, and commits it green. The generic "put the red suite in the primary tree" instruction does not hold here.
- **Global `CLAUDE.md` `Engineering`'s red witness takes a targeted pre-commit run here, never a red commit.** A red file cascades — one drives the suite to `59 failed`, `test_c8_08` among them (above) — and the gate has to close green, so the red and the green are separate runs by construction rather than one run read twice. Witness with `P pytest tests/<file>` against the unfixed tree, then fix, then commit green. Record **two** refs in the commit body, never one: the unfixed *implementation* revision the red fired against (primary-tree HEAD at witness time), and — for a restored `archive/*` suite — the tag that supplied the test bytes. They are different objects: `archive/m2u82-test` (`c92ffcc`) differs from `9ed947f` in 9 `src/` files, so a tag alone cannot separate a red against unfixed code from a red against code already repaired.
- **`green` is machine-checked here, and `skipif` is not a demotion.** A32 inside `test_c8_08` asserts zero across `failed`, `error`, `errors`, `skipped`, `xfailed`, `xpassed`, so a green decisive gate already carries global `CLAUDE.md`'s run-and-passed reading of `green` for every case the run **collects** and owes no separate skip census for them. Every `skipif` in this tree is an entry gate on environment completeness — R + packages, the MJPG/AVI codec, a present `inventory`/`videos`, a published v4 `cameras_qc.csv`, a permission-model probe — which is why A32 passes at zero on a complete environment and why the three read-only symlinks under *Worktree gate recipe* are load-bearing. An environment gate is not a demoted case and earns no `.agent/deferred.md` row. **No complete test-membership gate exists here**, so a case that stops being collected can pass unseen: A32 reconciles `passed` against its own `--collect-only` on the same tree, so a deletion, a rename out of `test_*.py` or a fresh import guard moves both figures together. Cover is partial and accidental — a determinism campaign's `SOURCE_FILES` names specific test files (`check_qualify_determinism.py` names `tests/test_measure.py`, `test_qualify.py`, `test_sessions.py`), and `source_digests()` reads each, so `test_m2u5_alignment.py::test_c56_determinism_reach` fails if one of those three is deleted. Everything outside those tuples is invisible, which is why the approval rule and not the gate is what catches demotion. Checks *outside* the suite are separately run or separately not-run and each reports by name or `none` — `ruff`, `ty`, the R stage, every auxiliary campaign (→ `docs/technical/conventions.md`).
- **A determinism campaign REFUSES to regenerate over a stale result — `rm` the JSON first.** Both `check_calibration_qc_determinism.py` and `check_qualify_determinism.py` compare their recorded source digests against current bytes and exit rather than overwrite — the field is `source_digests` in the calibration-QC campaign, so read the key before asserting on it — which is the stale-green barrier working. Editing any shared module (`multicam.py`, `sessions.py`, `video_io.py`) fires the matching `tests/test_*.py` source tripwire, so the sequence is: land every source and test edit, `rm tests/<name>_determinism_results.json`, rerun each campaign, then commit. Regeneration is a fixpoint.
- **Find every affected campaign before regenerating: `rg -ln '<file>' tests/*_results.json`.** One source file sits in several campaigns' declared sources — `qualify.py` is in the qualify, calibration-QC **and** cohort lists, `probe_sync_policy.py` in the qualify list alone. Regenerating the obvious one and running the suite is how you discover the rest, at a full suite each time: measured, a comment-only edit to `qualify.py` left 4 failures after the qualify campaign alone was regenerated — `test_calibration_qc`, `test_cohort_review` R35, `test_isotropic_coords` P14, and `test_c8_08` behind them, 9 min 26 s to learn it. The cohort campaign holds **one** `source_digest` over its whole tuple, so its assertion names no file; the sweep is the only thing that names one. Its tuple (`scripts/check_cohort_determinism.py` `SOURCES`) = `cohort.py`, `corpus_run.py`, `inventory.py`, `qualify.py`, `sessions.py`, `tests/test_cohort.py`, `tests/test_sessions.py` and the two `2d_idx` R goldens — M2.9 edited `corpus_run.py` twice and the decisive gate failed on R35 both times; read that tuple before every gate whose diff touches one of them.
- **A green suite measures nothing about pinning.** `30280c3` shipped 12 review fixes with 96 tests green and the first mutation campaign showed 18 of 72 mutants surviving. Fix-plus-test is not fix-plus-*pinning*-test.

## Hosted CI — replaced by the local gate (user ruling on `CLAUDE.md` `Session flow` IMPLEMENT)

The template puts security scanning + update automation "in gate + CI". **Here the full local gate
replaces hosted CI.** The decisive gate reads `videos/` + `inventory/` — patient data that never
leaves this machine — and `renv/library`; a runner without them fails or skips cases, and A32 fails
any skipped case (above), so no hosted runner can close it green. Scanners — dependency audit, secret
scan, static analysis — join the local gate command, and update automation joins the GitHub
remote, at the next IMPLEMENT or MAINTAIN session (→ `.agent/deferred.md`).

## The accelerator run recipe — mutually exclusive with the gate prefix

The gate prefix is exactly what strips the accelerator: it reports `['CPU']`, compiles, runs and produces correct output with no warning anywhere, because CPU is a supported device. **Only the wall clock tells.** Measured on the same two events, `--tracking body`: **1015.5 s / 927.4 s under the gate prefix vs 638.5 s / 497.2 s** under the run recipe = **1.6-1.9×**, which is 8 h against 15 h over the corpus.

```sh
source /var/home/eturkes/.local/app/intel-accel/env.sh
PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/<driver>.py
```

- Verify placement **before** funding hours: `openvino.Core().available_devices` must read `['CPU', 'GPU', 'NPU']`, and the event's own `run.log` must open `pose-device=NPU` + "loaded … with the openvino/NPU backend".
- The inherited `PYTHONPATH` selects the *host* OpenVINO build, which needs glibc 2.43 — a loud failure in the container, never a silent CPU fallback. Stripping the env entirely falls back to the `.venv` pip wheel at `['CPU']`.
- Enablement, device preference + self-test → `CLAUDE.local.md`.

## Worktree gate recipe (`.scratch/worktrees/<name>`)

Every teammate worktree runs the full Python gate concurrently off the one primary environment, read-only:

```sh
export UV_PROJECT_ENVIRONMENT=<primary-tree>/.venv PYTHONPATH="$PWD/src"
uv run --no-sync ruff check && uv run --no-sync ruff format --check \
  && uv run --no-sync ty check && uv run --no-sync pytest
```

- **Two resolution paths with opposite defaults — the most expensive trap in this project.** `pyproject.toml` sets `pythonpath = ["src", "tests"]`, which pytest resolves against **rootdir** and inserts at `sys.path[0]`: `pytest` launched from a worktree imports **that worktree's** `pose_estimation`, with or without `PYTHONPATH`. Everything else — `python -c`, `python -m pose_estimation.<mod>`, `ty check` — resolves through the hatchling editable install to the **primary** tree's `src/`. So a reviewer running its red suite from its own worktree tests its own copy and every primary-tree fix reads as still-broken. **Print `<module>.__file__` before believing a red**, and copy a test file into the primary `tests/` to exercise primary code.
- `UV_PROJECT_ENVIRONMENT` must be exported on **every** call: `uv run` inside a worktree otherwise creates and selects a worktree-local `.venv`, which `--no-sync` leaves empty → `No module named pytest`.
- `--no-sync` keeps the shared environment unmutated, which is what makes concurrent worktree gating safe.
- Tool caches (`.ruff_cache`, `.pytest_cache`, `.ty_cache`, `.coverage`) are cwd-relative → already private per worktree.
- **Four gitignored trees must be symlinked in read-only, or the gate is not equivalent**: `videos`, `inventory`, `qualification` → `<worktree>/<name>`, and `renv/library` → `<worktree>/renv/library` (one `ln -sfn` each; a brace list into `<worktree>/` lands `library` at the root). Without `qualification` the 3 `tests/test_m2u5_alignment.py` cases gated on a published v4 `cameras_qc.csv` SKIP and C8.08 fails on them. Without `videos` every real-data command fails at a missing path; without `renv/library` every R case SKIPs (with it: 469 passed / 0 skipped, `tests/test_r_pipeline.py` 25 passed / 0 skipped); without `inventory` `tests/test_sessions.py::test_p05_real_corpus_headline_counts` SKIPs and that single skip fails C8.08, whose A32 reconciliation demands zero skipped. Never write through any of the four — `inventory/` carries source paths and is as sensitive as the corpus, and R must never `renv::install`/`renv::snapshot` through the link.

## Check firing + scratch validators

Firing records + scratch-validator regeneration paths live in `paths:` rules on the files they grade: `checks.md` (`scripts/check_*.py`, mutation campaigns, report grader), `review-ui.md`, `pose-instruments.md`, `text-register.md`, `capture-docs.md` (`nc_m2u74.py`), `spec-lint.md`.

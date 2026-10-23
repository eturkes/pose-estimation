# Gate invocation

## The gate command — `scripts/gate.sh` (`CLAUDE.md` `Execution`)

- **Full gate = `scripts/gate.sh`**: static steps (`.agent/spec-lint`, `ruff check` incl. security
  rules `S`, `ruff format --check`, `ty check` over `src/` + `tests/`, `shellcheck`, Dependabot
  schema), scanners (`audit`, `secrets`, `scanner-seeds` → `scanners.md`; network + one
  `scripts/install-tools.sh`), `cocowb-eval` (`bench.md`), then `P pytest --durations=15`, in that
  order, each under the prefix below and each printing `gate: <step> rc=<rc> <wall> s`. A failed
  step leaves the later ones running → one run reports every failure; rc=1 + `gate: FAIL <steps>`,
  else `gate: PASS`.
- **Fast gate (iteration) = `scripts/gate.sh --fast <tests…>`**: static steps (~2 s, sequential,
  measured → parallelism buys nothing) + the named tests, A32 unarmed (a subset is not the
  decisive gate). Scanners run in the full gate alone.
- **`scripts/gate.sh <cmd…>` = `<cmd>` under the prefix**, from the repo root — the form a brief
  carries.
- **pytest dominates the full gate** → its equal-strength speedup is a `.agent/deferred.md` row.
- **Full suite: collected count + measured wall → `.agent/spec.md` `Artifacts`.** Always
  `run_in_background: true` + redirect to a log; a background command piped through `tail`
  buffers everything until exit.

## The prefix — every gate command in this project

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <cmd>
```

Carry it verbatim (`scripts/gate.sh <cmd…>` does). `PYTHONPATH` is the half that does the work:
the **inherited** `PYTHONPATH` names the host OpenVINO package (glibc 2.43) and pointing it at
`src` displaces it so `tests/conftest.py` imports the `.venv` wheel. Measured, `<form> pytest
--collect-only -q tests/test_public_api.py` under an interactive shell's environment:

| form | rc | outcome |
| --- | --- | --- |
| bare `uv run --no-sync` | 4 | `ImportError … GLIBC_2.43 not found` (host `libopenvino.so.<ver>`) |
| `env -u LD_LIBRARY_PATH` alone | 4 | `ImportError: libopenvino.so.<ver>: cannot open shared object file` |
| `PYTHONPATH="$PWD/src"` alone | 0 | 12 collected |
| both | 0 | 12 collected |

## The decisive gate

- **= the full gate's pytest step with A32 armed** (`POSE_GATE_FULL=1`, set by `scripts/gate.sh`;
  plugin `tests/a32_gate.py`, hooked in through `tests/conftest.py`): the whole collection, no
  deselection or narrowed path, every case passed with zero `failed`, `error`, `errors`,
  `skipped`, `xfailed`, `xpassed` — runtest and collection reports both (a module-level
  `pytest.skip` / `importorskip` never reaches runtest); a breach prints `A32 FAIL: …`.
  `test_c8_08` keeps its id (`scripts/check_m2u4_suite_seed.py` needs all 82) and drives the
  plugin over synthetic sessions: clean → rc 0; `-k` subset, narrowed path, skip, xfail → rc 1;
  env unset → inert.
- **Run the full gate alone**: R cases drive timed subprocesses, so a concurrent decode, inference
  sweep, probe collection or mutation campaign can time one out.
- **One red file anywhere under `tests/` fails the decisive gate** → a diff-blind or reviewer red
  suite lives on its `archive/*` tag (→ `retention.md`); the implementing unit restores it with
  `git show <tag>:tests/<file> > tests/<file>`, drives it green, commits it green.
- **`CLAUDE.md` `Engineering`'s red witness = a targeted pre-commit run, never a red commit.**
  Witness with `P pytest tests/<file>` against the unfixed tree, fix, commit green. The commit body
  records **two** refs: the unfixed implementation revision the red fired against (primary HEAD
  at witness time) and, for a restored suite, the `archive/*` tag that supplied the test bytes — a
  tag alone cannot separate a red against unfixed code from one against repaired code.
- **`green` is machine-checked, and `skipif` is not a demotion.** A green decisive gate carries
  `CLAUDE.md` `Engineering`'s run-and-passed `green` for every collected case. Every `skipif` here
  is an entry gate on environment completeness (R + packages, the MJPG/AVI codec, a present
  `inventory`/`videos`, a published v4 `cameras_qc.csv`, a permission-model probe) → earns no
  `.agent/deferred.md` row, and is why the worktree symlinks below are load-bearing.
- **No complete test-membership gate exists**: A32 reconciles `passed` against the same session's
  collection, so a deletion, a rename out of `test_*.py` or a new import guard moves both figures
  together. Cover is partial (a determinism campaign's `SOURCE_FILES` reads named test files) →
  the approval rule, not the gate, catches demotion. Checks outside the suite (`ruff`, `ty`, the R
  stage, auxiliary campaigns → `docs/technical/conventions.md`) report by name or `none`.
- **A green suite measures nothing about pinning**: 12 review fixes shipped with 96 tests green
  left 18 of 72 mutants surviving → fix-plus-test is not fix-plus-pinning-test.
- Determinism-campaign regeneration order → `determinism.md` (injected on touching a campaign's
  sources).

## Hosted CI — replaced by the local gate (user ruling on `CLAUDE.md` `Session flow` IMPLEMENT + `Execution` `CI gate`)

The decisive gate reads `videos/` + `inventory/` (patient data that never leaves this machine) and
`renv/library`; a hosted runner without them fails or skips cases, and A32 fails any skip → no
hosted runner can close it green. Scanners (dependency audit, secret scan, static analysis) run in
the local gate (→ `scanners.md`); update automation = `.github/dependabot.yml` (uv, all three
lockfiles). User ruling: no hosted scanner workflow either; the remote carries Dependabot config
alone. ⇒ `CI gate` + its `ci/*` pushes = inapplicable, though the repo is public: no workflow
exists to fan out to, and the full gate runs locally (`scripts/gate.sh`).

## The accelerator run recipe — mutually exclusive with the gate prefix

```sh
source <intel-accel>/env.sh
PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/<driver>.py
```

- **The gate prefix silently strips the accelerator**: it reports `['CPU']`, compiles, runs and
  produces correct output with no warning; only the wall clock tells (same two events: 1015.5 s /
  927.4 s under the prefix vs 638.5 s / 497.2 s under the recipe).
- Verify placement before funding hours: `openvino.Core().available_devices` reads
  `['CPU', 'GPU', 'NPU']`, and the event's `run.log` opens with the arm's registry pose device
  (`pose-device=NPU` for `rtmw-m-256`, `GPU` for `rtmw-l-384`) + "loaded … with the openvino/<device>
  backend".
- The inherited `PYTHONPATH` alone selects the host build → a loud glibc failure, never a silent
  CPU fallback; a stripped env falls back to the `.venv` wheel at `['CPU']`.
- **`<intel-accel>` = this machine's OpenVINO environment directory, resolved in `CLAUDE.local.md`**
  (gitignored; user ruling on `CLAUDE.md` `Authoring` portability) → every committed recipe writes
  `source <intel-accel>/env.sh`, never the literal path.
- **This machine's host/container layout stays named, by user ruling** (on `CLAUDE.md` `Authoring`
  portability): `docs/technical/environment.md` documents it and `.envrc` switches on it (keys on
  `/run/host/*`, else `.venv-host`).

## Worktree gate recipe (`.scratch/worktrees/<name>`)

Teammate worktrees share the primary environment read-only; run each full gate alone, never concurrently:

```sh
export UV_PROJECT_ENVIRONMENT=<primary-tree>/.venv
scripts/gate.sh
```

- **Two resolution paths with opposite defaults — the most expensive trap here.** `pyproject.toml`
  `pythonpath = ["src", "tests"]` resolves against pytest's **rootdir** → `pytest` from a worktree
  imports **that worktree's** `pose_estimation`. Everything else (`python -c`, `python -m`,
  `ty check`) resolves through the hatchling editable install to the **primary** `src/` — unless
  `PYTHONPATH` names a worktree's `src`, as `<worktree>/scripts/gate.sh <cmd…>` does. A reviewer's
  red suite run from its worktree tests its own copy → **print `<module>.__file__` before
  believing a red**; copy the test into the primary `tests/` to exercise primary code.
- **An exported `UV_PROJECT_ENVIRONMENT` + any `uv run --directory <subproject>` (`tools/cocowb-eval`,
  `tools/review-ui`) syncs that subproject INTO the named environment — the primary `.venv` becomes
  the subproject's, including its Python version (cocowb-eval: Python 3.12).** The gate's
  `cocowb-eval` step and `bench.py _evaluate` unset it; a hand-run subproject command must too
  (`env -u UV_PROJECT_ENVIRONMENT uv run --directory …`). Repair = `uv sync --locked` from the root.
- Export `UV_PROJECT_ENVIRONMENT` on **every** call: otherwise `uv run` creates a worktree-local
  `.venv` that `--no-sync` leaves empty → `No module named pytest`. `--no-sync` keeps the shared
  environment unmutated (what makes environment sharing safe); tool caches are cwd-relative →
  private per worktree.
- **Four gitignored trees symlinked in read-only, or the gate is not equivalent**: `videos`,
  `inventory`, `qualification` → `<worktree>/<name>`, `renv/library` → `<worktree>/renv/library`
  (one `ln -sfn` each; a brace list into `<worktree>/` lands `library` at the root). Missing
  `qualification` → 3 `tests/test_m2u5_alignment.py` cases SKIP; missing `renv/library` → every R
  case SKIPs; missing `inventory` → `test_p05_real_corpus_headline_counts` SKIPs; each skip fails
  A32. Never write through any of the four — `inventory/` carries source paths, and R must never
  `renv::install`/`renv::snapshot` through the link.
- **A full gate from a worktree adds two tool links**: `.tools` → `<primary>/.tools` (else `audit`
  + `secrets` fail "scanners missing") and `tools/cocowb-eval/.venv` → the primary's (else the
  step syncs a fresh Python 3.12 env). Measured at `d2eafbd` with all six links: `gate: PASS`, 3142
  passed, pytest 1358 s. A republish in the primary (`qualification/` is linked) waits for the
  worktree gate to finish.

## Check firing + scratch validators

Firing records + scratch-validator regeneration paths live in `paths:` rules on the files they
grade: `checks.md` (`scripts/check_*.py`, mutation campaigns, report grader), `determinism.md`,
`review-ui.md`, `pose-instruments.md`, `text-register.md`, `capture-docs.md` (`nc_m2u74.py`),
`spec-lint.md`, `scanners.md`, `bench.md`.

# Gate invocation

## The prefix — every gate command in this project

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <cmd>
```

Use both halves; `PYTHONPATH` is the half that does the work. Measured 4 ways, `pytest --collect-only -q tests/test_public_api.py` in the primary tree under an interactive shell's inherited environment:

| form | rc | outcome |
| --- | --- | --- |
| bare `uv run --no-sync` | 4 | `ImportError … GLIBC_2.43 not found` (host `libopenvino.so.2630`) |
| `env -u LD_LIBRARY_PATH` alone | 4 | `ImportError: libopenvino.so.2630: cannot open shared object file` |
| `PYTHONPATH="$PWD/src"` alone | 0 | 12 collected |
| both | 0 | 12 collected |

The cause is the **inherited `PYTHONPATH`**, which names the host OpenVINO python package built against glibc 2.43; pointing `PYTHONPATH` at `src` displaces it, so `tests/conftest.py` imports the `.venv` wheel instead and collection proceeds. Unsetting `LD_LIBRARY_PATH` keeps the matching host runtime out of the loader path and **cannot substitute** — alone it strips the libraries while leaving the host package importable, which is the second row. Keep one form and carry it verbatim in every teammate brief; a partially-remembered prefix costs a window (`scripts/gate.sh` port → `.agent/deferred.md`). The worktree recipe below therefore runs green without `env -u`: it exports `PYTHONPATH` itself.

- **Full suite ≈ 14-25 min (1750 tests).** Always `run_in_background: true` + redirect to a log, then work while it runs — a background command piped through `tail` buffers everything until exit.
- **Run the decisive gate alone.** `tests/test_r_timebase_truth.py::test_c8_08` runs the WHOLE suite in a subprocess (`pytest -q --maxfail=1 -k "not test_c8_08"`, cwd = project root), asserts rc=0 and reconciles the pass count against a separate collection. Five concurrent worktree gates plus MAIN's on 8 cores drove it past its 900 s timeout — that is the whole difference between `1701 passed / 1 failed` and green. Never run it beside a decode, an inference sweep or a mutation campaign; brief reviewers to targeted files and keep the full-suite slot for MAIN.
- **One red file anywhere under `tests/` takes `test_c8_08` down** and with it the decisive gate for the whole unit. Measured: red file present → 59 failed / 1643 passed; absent → 1642 passed. So a diff-blind or reviewer red suite lives on its `archive/*` tag (→ `retention.md`), and the implementing unit restores it with `git show <tag>:tests/<file> > tests/<file>`, drives it green, and commits it green. The generic "put the red suite in the primary tree" instruction does not hold here.
- **`CLAUDE.md` `Engineering`'s red witness takes a targeted pre-commit run here, never a red commit.** A red file cascades — one drives the suite to `59 failed`, `test_c8_08` among them (above) — and the gate has to close green, so the red and the green are separate runs by construction rather than one run read twice. Witness with `P pytest tests/<file>` against the unfixed tree, then fix, then commit green. Record **two** refs in the commit body, never one: the unfixed *implementation* revision the red fired against (primary-tree HEAD at witness time), and — for a restored `archive/*` suite — the tag that supplied the test bytes. They are different objects: `archive/m2u82-test` (`c92ffcc`) differs from `9ed947f` in 9 `src/` files, so a tag alone cannot separate a red against unfixed code from a red against code already repaired.
- **`green` is machine-checked here, and `skipif` is not a demotion.** A32 inside `test_c8_08` asserts zero across `failed`, `error`, `errors`, `skipped`, `xfailed`, `xpassed`, so a green decisive gate already carries `CLAUDE.md`'s run-and-passed reading of `green` for every case the run **collects** and owes no separate skip census for them. Every `skipif` in this tree is an entry gate on environment completeness — R + packages, the MJPG/AVI codec, a present `inventory`/`videos`, a published v4 `cameras_qc.csv`, a permission-model probe — which is why A32 passes at zero on a complete environment and why the three read-only symlinks under *Worktree gate recipe* are load-bearing. An environment gate is not a demoted case and earns no `.agent/deferred.md` row. **No complete test-membership gate exists here**, so a case that stops being collected can pass unseen: A32 reconciles `passed` against its own `--collect-only` on the same tree, so a deletion, a rename out of `test_*.py` or a fresh import guard moves both figures together. Cover is partial and accidental — a determinism campaign's `SOURCE_FILES` names specific test files (`check_qualify_determinism.py` names `tests/test_measure.py`, `test_qualify.py`, `test_sessions.py`), and `source_digests()` reads each, so `test_m2u5_alignment.py::test_c56_determinism_reach` fails if one of those three is deleted. Everything outside those tuples is invisible, which is why the approval rule and not the gate is what catches demotion. Checks *outside* the suite are separately run or separately not-run and each reports by name or `none` — `ruff`, `ty`, the R stage, every auxiliary campaign (→ `docs/technical/conventions.md`).
- **A determinism campaign REFUSES to regenerate over a stale result — `rm` the JSON first.** Both `check_calibration_qc_determinism.py` and `check_qualify_determinism.py` compare their recorded source digests against current bytes and exit rather than overwrite — the field is `source_digests` in the calibration-QC campaign, so read the key before asserting on it — which is the stale-green barrier working. Editing any shared module (`multicam.py`, `sessions.py`, `video_io.py`) fires the matching `tests/test_*.py` source tripwire, so the sequence is: land every source and test edit, `rm tests/<name>_determinism_results.json`, rerun each campaign, then commit. Regeneration is a fixpoint.
- **Find every affected campaign before regenerating: `rg -ln '<file>' tests/*_results.json`.** One source file sits in several campaigns' declared sources — `qualify.py` is in the qualify, calibration-QC **and** cohort lists, `probe_sync_policy.py` in the qualify list alone. Regenerating the obvious one and running the suite is how you discover the rest, at a full suite each time: measured, a comment-only edit to `qualify.py` left 4 failures after the qualify campaign alone was regenerated — `test_calibration_qc`, `test_cohort_review` R35, `test_isotropic_coords` P14, and `test_c8_08` behind them, 9 min 26 s to learn it. The cohort campaign holds **one** `source_digest` over its whole tuple, so its assertion names no file; the sweep is the only thing that names one.
- **A green suite measures nothing about pinning.** `30280c3` shipped 12 review fixes with 96 tests green and the first mutation campaign showed 18 of 72 mutants surviving. Fix-plus-test is not fix-plus-*pinning*-test.

## The accelerator run recipe — mutually exclusive with the gate prefix

The gate prefix is exactly what strips the accelerator: it reports `['CPU']`, compiles, runs and produces correct output with no warning anywhere, because CPU is a supported device. **Only the wall clock tells.** Measured on the same two events, `--tracking body`: **1015.5 s / 927.4 s under the gate prefix vs 638.5 s / 497.2 s** under the run recipe = **1.6-1.9×**, which is 8 h against 15 h over the corpus.

```sh
source /var/home/eturkes/.local/app/intel-accel/env.sh
PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/<driver>.py
```

- Verify placement **before** funding hours: `openvino.Core().available_devices` must read `['CPU', 'GPU', 'NPU']`, and the event's own `run.log` must open `pose-device=NPU` + "loaded … with the openvino/NPU backend".
- The inherited `PYTHONPATH` selects the *host* OpenVINO build, which needs glibc 2.43 — a loud failure in the container, never a silent CPU fallback. Stripping the env entirely falls back to the `.venv` pip wheel at `['CPU']`.
- Enablement, device preference + self-test → `CLAUDE.local.md` → `~/agents/docs/openvino.md`.

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
- **Three gitignored trees must be symlinked in read-only, or the gate is not equivalent**: `ln -sfn <primary>/{videos,renv/library,inventory} <worktree>/`. Without `videos` every real-data command fails at a missing path; without `renv/library` every R case SKIPs (with it: 469 passed / 0 skipped, `tests/test_r_pipeline.py` 25 passed / 0 skipped); without `inventory` `tests/test_sessions.py::test_p05_real_corpus_headline_counts` SKIPs and that single skip fails C8.08, whose A32 reconciliation demands zero skipped. Never write through any of the three — `inventory/` carries source paths and is as sensitive as the corpus, and R must never `renv::install`/`renv::snapshot` through the link.

## Check firing evidence

`CLAUDE.md` `Engineering`: a purpose-built check ships with the input that makes it fail, and that firing is recorded here. **A check that cannot fail and a clean tree emit the same green.** Census over the 16 committed checks, measured:

- **12/12 `scripts/check_*.py` ship green-only.** No committed input drives any of them to a nonzero rc or a named `FAIL`. Every suite that touches one either calls it on the conforming committed document (`test_claim_report.py`, `test_prospective_capture.py`, `test_calibration_qc_fixtures.py`) or grades a helper the check imports without reaching the check's own refusal (`test_inventory_review.py`, `test_m2u5_alignment.py` — both stop at `stale_source_mismatches` and never call `main`, so no case asserts rc=2 or `REFUSED`). `check_cameras_qc_census.py`, `check_isotropy_angle_fidelity.py`, `check_m2u5_determinism.py` and `check_review_report.py` have no committed caller at all. The gap rides `.agent/deferred.md`; per-check acceptance checks were written with the evidence fresh.
- **The four mutation campaigns fire only by accident.** A campaign's own failure path is a surviving mutant or a red baseline, and no committed input produces either — the kill counts in `docs/technical/conventions.md` § *Auxiliary campaigns* (51/51 calibration-QC, 71/72 inventory + `M028` ruled equivalent, 25/25 M2.5) record each campaign *passing*, which is not the same evidence. What fires today is a defect: `tests/inventory_mutation_results.json` (2/2 targets) and `tests/measure_mutation_results.json` (2/3) both hold `target_sha256` against source bytes that have since moved, so each campaign refuses at `result file targets a different source baseline`. **Repairing that staleness removes the only firing either campaign has**, so the repair and the seed land together. `run_measure_mutations.py` is unrecorded besides — no live surface names an invocation or a kill count. **`run_inventory_mutations.py` cannot fail on a survivor at all**: it records the `SURVIVED` verdict in its payload and then `return 0` unconditionally, and `M028`'s equivalence is prose-ruled rather than allowlisted in code, so a new unexpected survivor exits green.
- **`check_review_report.py` firing, measured**: an 18-row all-`unknown` seed grades `FAIL P03 — 18 rows still unknown`, rc=1; the same file with every verdict filled grades `PASS`, rc=0. That pair is what makes global `Subagents`' seed-both-ways rule a check here rather than a slogan, and it lives only in `.scratch/agents/` — gitignored, so it does not ship.
- Recorded firings that ship no input: `check_prospective_capture.py` at 9/9 through `.scratch/nc_m2u74.py` (below). A recorded count with a gitignored input satisfies the record half and fails the shipped half.

## Scratch validators pending port

A gate backing a durable claim must rerun from committed state, so a scratch-local validator is a temporary encoding: its regeneration path is recorded here and its port is a `.agent/deferred.md` row.

- `.scratch/nc_m2u74.py` — the nine M2.7.4 negative controls over `docs/prospective_capture.md`. Each control mutates the document in place, grades `scripts/check_prospective_capture.py` in a fresh `runpy` namespace, and restores the bytes under `try/finally`; the run reports `controls_firing N of 9` and proves the file byte-identical against its pre-run digest. **9/9 fire, restored `55eb769a1768`.** Seed rules learned here: P03 needs S20's single `MUST` (S14 carries three, so lowercasing one grades nothing), and NC4's needle must come from `calibration_qc.PROHIBITED_PARAPHRASES` rather than invented prose.
- `.scratch/theme_qa.mjs` — review-ui theme control, the paths a still capture cannot show.
  `node .scratch/theme_qa.mjs http://127.0.0.1:<port>` against a running `python -m review_ui`;
  resolves `chromiumfish` + `playwright-core` out of the pnpm global store exactly as `webcap` does,
  and drives a **persistent** context, which is what makes `localStorage` outlive the reload the
  rehydration check needs. 13 checks: auto is the default and resolves light under CDP (this build
  reports `prefers-color-scheme` light), the cycle auto→light→dark→auto with its label and its
  stored value per step, `data-theme` present at document *commit* after a reload, `palette()`
  returning a used `rgb()` rather than the `light-dark()` call text, and the `:has()` stage backdrop
  across show/dim/hide. **13/13 green; 2 of 2 seeds fire** — neutering the `.stage:has(...)`
  background reds exactly the 2 backdrop rows, misspelling the pre-paint `review-ui-theme` key reds
  the 3 rehydration rows and nothing else, both files restored byte-identical by digest.
- `.scratch/player_layout_qa.mjs` — review-ui clip player layout: the stage fits the room the UI
  leaves and the transport stays on screen. `node .scratch/player_layout_qa.mjs http://127.0.0.1:<after>
  [<before>]` against a running `python -m review_ui`; same chromiumfish + `playwright-core`
  resolution as `theme_qa.mjs`, plain context. 38 checks: pane shape (no clip title, banner hidden,
  no codec sentence, one selected row, overlay + strip backed at device resolution), the fit over
  five viewports + the stacked breakpoint (fits its room, fills an axis, ratio error < 1 %, video
  fills the stage, every control row on screen, page fits the viewport), a width-only resize
  rebacking the strip, and the census + cohort panel rectangles against the before state — the
  change touched shared CSS and those two views are the committed captures. **38/38 green; 4 of 4
  seeds fire their own rows and nothing else.** The second URL is a `git worktree` of the previous
  commit served with `--repo <primary>` and `UV_PROJECT_ENVIRONMENT=<primary>/prototype/review-ui/.venv
  uv run --no-sync`, which measured controls below the stage at 322 px → 156 px. Seed rules learned
  here: `.stage`'s own `max-width: 100%` absorbs a `width` seed, so a seed that must escape the room
  needs `min-width`; and a `goto` differing only in its fragment is a same-document navigation, so an
  injected style tag survives into the next seed — the script carries a per-load query parameter for
  that. `.scratch/player_shot.mjs` takes the layout-only capture beside it (→ `data-boundary.md`).
- `.scratch/default_view_qa.mjs` — review-ui landing view: the player is the first tab and the view a
  bare URL opens. `node .scratch/default_view_qa.mjs http://127.0.0.1:<port>` against a running
  `python -m review_ui`; same chromiumfish + `playwright-core` resolution as `theme_qa.mjs`, plain
  context, no capture of any kind — it prints tab order, the active view, the hash and booleans, which
  is what keeps a landing-view check inside `data-boundary.md`. 16 checks: tab order, section order
  and the first tab's label, the bare URL settling on `#player` with its stage drawn, a `#cohort` deep
  link, a tab click, an unknown fragment falling back to the player, and the same-document group — a
  `location.hash` write selecting its view, the Back button, an unknown same-document hash.
  **16/16 green; 4 of 4 seeds fire their own rows and nothing else** — `DEFAULT_VIEW` player→census
  reds 6 landing rows, the `index.html` nav order player→census reds the 2 tab rows, deleting the
  `hashchange` listener reds the 3 same-document rows, and moving `location.hash = name` back after
  the render `await` reds exactly the Back row; both files restored byte-identical by sha256.
  Rules learned here, all three measured against a tree that read green first:
  - `show()` toggles the tab and view classes *before* it awaits the render, so a class-only wait
    reads a half-switched page and reported the previous hash under a seed. Every wait is hash +
    active view + drawn node, through `waitForFunction`.
  - A `waitForSelector` a seed can never satisfy must be caught into a `fail` row, or the seed
    reports as a crashed script rather than as a failing check.
  - **A row timed against a warm render grades the race, not the code.** The Back row passed under
    its own ordering seed once the cohort render went warm — the stale trailing write needs the
    superseded render still in flight. The row holds the window open itself (`page.route` delaying
    `/api/cohort` 1500 ms) and reads the state *after* the delay elapses, since a trailing write
    lands after Back settles. It is also two-sample — it requires the view to have BEEN cohort —
    because a Back row asserting only the destination passes vacuously wherever the hash never
    moved the view at all, which is exactly the state its own precondition row reds.
  The same-document fragment trap above applies here too and for a second reason: a fragment-only
  `goto` is not a load, and view selection now runs through `hashchange` rather than boot alone.
- `.scratch/steq.py` — ASD-STE100 register scan over the human-facing surface (inventory: `docs/technical/conventions.md` → *Text register*). Drops fences/tables/headings/frontmatter, joins wrapped lines into blocks so a sentence is measured whole, splits on `.!?`, flags `LONG` (> `--max`; 20 for instructions, 25 for descriptions), `FILLER`, `CONTRACTION` (also fires on possessive `'s`), `PASSIVE` (be-verb + participle heuristic). Code-file mode samples quoted `help=`/`description=`/`title=` strings only. Measured at `--max 20`: `README.md` 14 → 2, `docs/capture_protocol.md` 20 → 8; residual flags are 21-25-word descriptions, which the rule allows. **The scanner cannot apply its own rule.** One `--max` covers every sentence, so the instruction-vs-description call that picks 20 or 25 is made by hand on each residual. Measured over the four shipped surfaces: 48 flags at `--max 20`, 24 at `--max 25`; **25 of the 26 `LONG` verdicts sit in the 21-25 band** and turn entirely on that call, 1 fails either way. Of the 24 residuals, both `CONTRACTION` hits are possessives (`instrument's`, `solve's`) and at least 4 of 21 `PASSIVE` hits are predicate adjectives (`is untested`, `is unmeasured`, `is unaffected`, `is closed`) — so **23 of 24 are heuristic output awaiting a human**, and several true passives are mandated by the claim boundary's own "may not be claimed" phrasing.
- `.scratch/fidelity.sh <base-ref> <file>…` — pairs with it: diffs the multiset of format specifiers, `--flags`, backticked spans, file names and numbers between a base ref and the working tree. A register-only edit must show no delta; every delta needs an explanation. Caught the p-value reformat (`p<.05` → `p < 0.05`) and confirmed 14 R files invariant.

## Committed report grader

`scripts/check_review_report.py` — 9 predicates over the two-tier report shape, so a review wave's report-shape claim reruns from committed state: three required sections, parseable `| <ID> |` rows, unique ids, a verdict cell per row, **zero `unknown`** (P03 — what makes an all-`unknown` seed grade nonzero), every `pass` row stating what was checked, a `### <id>` detail section per `fail` row carrying `file:line` + `predicate` + `impact` + `acceptance`, an acceptance check on every register entry. This checker is the command global `Subagents` deliverable-first says to name in the brief, and its measured both-ways firing is above.

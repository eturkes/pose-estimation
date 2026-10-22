---
paths:
  - "scripts/check_*.py"
  - "scripts/run_*_mutations.py"
  - "tests/*_results.json"
  - "scripts/gate.sh"
  - "tests/test_gate_prefix_recall.py"
---

# Committed checks — firing records

`CLAUDE.md` `Engineering`: a purpose-built check ships with the input that makes it fail, its firing recorded in a `paths:` rule on the check's files. **A check that cannot fail and a clean tree emit the same green.** Census over the 16 committed checks, measured:

- **12/12 `scripts/check_*.py` fire on a committed seed** — `P python scripts/check_seeds.py`: each seed builds its negative input in a temp tree and calls the checker's shipped `main`, requiring its own named failure + rc (re-derived from the checker's source: refusal literal, predicate id). Measured: `12/12 refused`, rc 0; `--inert check_prospective_capture:NC4` → `SEED-INERT` naming the checker, `11/12 refused`, rc 1. The four determinism seeds certify the stale-source barrier alone (the campaign body is stubbed past it); `check_m2u5_determinism`'s seed feeds synthetic publisher CSVs to D09 and certifies that refusal alone. Seeds needing another input path use the minimal flags the checkers gained (`--fixtures`, `--repo`, `--output`, `--suite`); default runs unchanged. `tests/test_check_seeds.py` drives both outcomes in the decisive gate. A new `check_*.py` joins the runner's census automatically and needs its seed, else rc 1. `check_isotropy_angle_fidelity.py` also fires through `tests/test_isotropy_angle_fidelity.py` (rc 0/2/1).
- **The inventory + measure mutation campaigns fire on a seed** (`--seed-survivor`: a comment-only mutant, no result file written) → `S001 SURVIVED`, rc 1 (inventory); `S01-seeded-survivor SURVIVED`, rc 1 (measure). Each catalogue names its ruled equivalent in code (`equivalent_reason`: inventory `M028`, measure `M02-reconcile-omitted-return`); `campaign_status` fails an unruled survivor or a killed equivalent. Measured from a clean worktree checkout: inventory 72 → 71 killed + `M028`, 1124 s; measure 18 → 17 killed + `M02`, 406 s — the recorded kill lists, verdict for verdict. **A determinism-evidence binding test kills every mutant by construction** (it fails on any target byte change; one per campaign, found by a comment-only probe per target) → `BYTE_BINDING_TESTS` deselects it, and the seed surviving is what proves no other oracle case reads bytes alone. `tests/test_mutation_campaigns.py` reds on a stale catalogue anchor, a result targeting other bytes, a result missing a mutant or carrying a failing verdict. The calibration-QC and M2.5 campaigns (51/51, 25/25) ship no seed yet.
- **`check_review_report.py` firing, measured**: an 18-row all-`unknown` seed grades `FAIL P03 — 18 rows still unknown`, rc=1; the same file with every verdict filled grades `PASS`, rc=0. That pair is what makes `CLAUDE.md` `Teammates`' "graded both ways at seed" a check here rather than a slogan; `scripts/check_seeds.py` ships the all-`unknown` half.
- **A template drop-in check stays declined** (`CLAUDE.md`, `.claude/agents/`, `.agent/spec-lint` against
  their template source): that source sits outside the repo, which `CLAUDE.md` `Authoring` portability
  keeps every committed check from naming, and each refresh, started by the user, copies the files in whole.
- A recorded count with a gitignored input satisfies the record half and fails the shipped half → every firing above ships its input.

## Committed report grader

`scripts/check_review_report.py` — 9 predicates over the two-tier report shape, so a review wave's report-shape claim reruns from committed state: three required sections, parseable `| <ID> |` rows, unique ids, a verdict cell per row, **zero `unknown`** (P03 — what makes an all-`unknown` seed grade nonzero), every `pass` row stating what was checked, a `### <id>` detail section per `fail` row carrying `file:line` + `predicate` + `impact` + `acceptance`, an acceptance check on every register entry. This checker is the validator a brief names beside its seeded deliverable (`CLAUDE.md` `Teammates`), and its measured both-ways firing is above.

## Gate runner

`scripts/gate.sh` (invocation law → `gates.md`) grades nothing itself; its strength = the steps it
runs + its rc. Firing, measured (`scripts/gate.sh --fast tests/test_spec_lint.py`): a planted
history line in `.agent/spec.md` + an unused import in a planted `scripts/*.py` → rc=1,
`gate: FAIL spec-lint ruff-check pytest`, every later step still run; both plants removed →
`gate: PASS`, spec digest restored.

## Gate-prefix recall scan

`tests/test_gate_prefix_recall.py` scans `.agent/spec.md`, `.claude/rules/`, `docs/` and `scripts/` for a root-project command (`uv run`, `python`, `pytest`, `pose-estimation-*`) without `P`/`$P`/`scripts/gate.sh`/the spelled prefix/the accelerator recipe; code spans (wrapped ones included), fenced lines and script comments, shell continuations joined. Firing, measured: a planted `` `pytest -q tests/test_x.py` `` line in `docs/technical/tests.md` → 1 failed naming `docs/technical/tests.md:103`; removed → 2 passed. At its landing the four surfaces held 90 bare forms (`d2eafbd`-era docs). The seed case pins each bare + accepted form.

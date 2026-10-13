---
paths:
  - "scripts/check_*.py"
  - "scripts/run_*_mutations.py"
  - "tests/*_results.json"
---

# Committed checks — firing records

Global `CLAUDE.md` `Engineering`: a purpose-built check ships with the input that makes it fail, its firing recorded in a `paths:` rule on the check's files. **A check that cannot fail and a clean tree emit the same green.** Census over the 16 committed checks, measured:

- **12/12 `scripts/check_*.py` ship green-only.** No committed input drives any of them to a nonzero rc or a named `FAIL`. Every suite that touches one either calls it on the conforming committed document (`test_claim_report.py`, `test_prospective_capture.py`, `test_calibration_qc_fixtures.py`) or grades a helper the check imports without reaching the check's own refusal (`test_inventory_review.py`, `test_m2u5_alignment.py` — both stop at `stale_source_mismatches` and never call `main`, so no case asserts rc=2 or `REFUSED`). `check_cameras_qc_census.py`, `check_isotropy_angle_fidelity.py`, `check_m2u5_determinism.py` and `check_review_report.py` have no committed caller at all. The gap rides `.agent/deferred.md`; per-check acceptance checks were written with the evidence fresh.
- **The four mutation campaigns fire only by accident.** A campaign's own failure path is a surviving mutant or a red baseline, and no committed input produces either — the kill counts in `docs/technical/conventions.md` § *Auxiliary campaigns* (51/51 calibration-QC, 71/72 inventory + `M028` ruled equivalent, 25/25 M2.5) record each campaign *passing*, which is not the same evidence. What fires today is a defect: `tests/inventory_mutation_results.json` (2/2 targets) and `tests/measure_mutation_results.json` (2/3) both hold `target_sha256` against source bytes that have since moved, so each campaign refuses at `result file targets a different source baseline`. **Repairing that staleness removes the only firing either campaign has**, so the repair and the seed land together. `run_measure_mutations.py` is unrecorded besides — no live surface names an invocation or a kill count. **`run_inventory_mutations.py` cannot fail on a survivor at all**: it records the `SURVIVED` verdict in its payload and then `return 0` unconditionally, and `M028`'s equivalence is prose-ruled rather than allowlisted in code, so a new unexpected survivor exits green.
- **`check_review_report.py` firing, measured**: an 18-row all-`unknown` seed grades `FAIL P03 — 18 rows still unknown`, rc=1; the same file with every verdict filled grades `PASS`, rc=0. That pair is what makes global `Subagents`' seed-both-ways rule a check here rather than a slogan, and it lives only in `.scratch/agents/` — gitignored, so it does not ship.
- Recorded firings that ship no input: `check_prospective_capture.py` at 8/8 through `.scratch/nc_m2u74.py` (→ `capture-docs.md`). A recorded count with a gitignored input satisfies the record half and fails the shipped half.

## Committed report grader

`scripts/check_review_report.py` — 9 predicates over the two-tier report shape, so a review wave's report-shape claim reruns from committed state: three required sections, parseable `| <ID> |` rows, unique ids, a verdict cell per row, **zero `unknown`** (P03 — what makes an all-`unknown` seed grade nonzero), every `pass` row stating what was checked, a `### <id>` detail section per `fail` row carrying `file:line` + `predicate` + `impact` + `acceptance`, an acceptance check on every register entry. This checker is the command global `Subagents` deliverable-first says to name in the brief, and its measured both-ways firing is above.

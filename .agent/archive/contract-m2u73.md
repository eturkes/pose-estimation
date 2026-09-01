# Contract — M2.7.3 claim-bounded negative report

Tier `docs` (assurance = `doc` + consistency pass). Base `ab3d53a`. Frozen at dispatch; amendments
append to §8.

## 1. Artifact

`docs/calibration_finding.md` — the human-facing statement of what the two calibration probes decide,
its evidence, and the wording boundary a person must hold when citing it.

Register = **human** (ASD-STE100), so the file joins `README.md` + `docs/capture_protocol.md` at
`docs/technical/conventions.md:45`. An artifact left off that inventory defaults to the agent
register, so the registration edit is what puts the file under the rule it is written to.

## 2. Design decisions

**D01 — the row set is the published claim set, never MAIN's summary of it.** Rows = the 15 entries of
`calibration_qc.CLAIMS` in module order, keyed by the `C01`-`C15` ids
`docs/technical/calibration_qc.md` already assigns. The report invents no conclusion, so it cannot
publish a sixteenth claim the publisher would refuse.

**D02 — permitted wording = the claim sentence verbatim.** Measured: all 15 claims are ≤25 words
(max 25, claim 6), so verbatim quotation satisfies the human register's description cap. A restated
claim is a new claim; the report quotes.

**D03 — the report never spells a prohibited paraphrase; it states the refused overreach by shape.**
Measured: no claim contains any of the 23 `PROHIBITED_PARAPHRASES` entries under `calibration_qc._fold`,
so full verbatim quotation and a **total** zero-occurrence prohibition scan are simultaneously
satisfiable. This is what keeps the scan exemption-free — the standing ruling that a scan quoting its
own needles buys itself a hole exactly where the hazard lives. The literal needles stay in `src/`,
where the machine reads them.

**D04 — this report is the sole published home of the `calibration_bias` numbers.** The publisher
cites and digests that probe and ingests nothing from it, so the claims resting on it publish with no
number attached. `evidence_qc.csv` carries `bias_transfer` rows alone. Every `calibration_bias`
figure a human may quote therefore has to appear here or nowhere.

**D05 — repair the drift the unit found rather than adding a third divergent copy.**
`docs/technical/calibration_qc.md` says "The marker publishes these supported statements" and then
prints text that is **not** what the marker publishes on 6 of 15 rows — C03, C05, C06, C09, C12, C15
(sentence splits, one Oxford comma, one dropped `here`). Semantically faithful, so no false claim
shipped; but a person copying C09 out of that document writes text `_assert_claim_conformance` would
refuse, which is the exact failure this unit exists to prevent. No test or script referenced the
document, so nothing pinned it. Both files are repaired to quote the constant and both are pinned, so
the constant is the single source of truth and neither copy can drift again.

**D06 — the consistency pass ships as a committed checker.** `scripts/check_claim_report.py`, driven
from `tests/` through one case, so a durable claim reruns from committed state instead of resting on
one session's reading. It imports the constants from `calibration_qc` — never a second copy — and
reuses that module's own `_fold`, so the report is normalised exactly as published bytes are.

**D07 — anti-duplication boundary.** `docs/technical/calibration_qc.md` owns the CLI, the schemas,
the refusal codes, the ownership and validation contract, and the publisher's own scope statement.
The report carries **evidence and wording alone** and links there for the tool. A schema table, a
refusal-code list or a CLI invocation appearing in the report is a defect.

**D08 — numbers carry their population.** Every count in the report names what it quantifies over in
the same sentence or the same cell. This project has published two closure statistics over different
event sets, two family-connectivity figures, and a 329-vs-355 census under one shape; a bare count is
the defect that survives review.

## 3. Predicates

Checker-owned, P01-P09:

- **P01** Each of the 15 `CLAIMS` appears verbatim in `docs/calibration_finding.md` exactly once.
- **P02** Each of the 15 `CLAIMS` appears verbatim in `docs/technical/calibration_qc.md` exactly once.
- **P03** Zero occurrences of any `PROHIBITED_PARAPHRASES` entry in either file, folded through
  `calibration_qc._fold`. No exemption, no excluded span.
- **P04** The report names both `PROBE_SCRIPTS` values verbatim, and each named script exists on disk.
- **P05** The report quotes `RULING["reason"]` and `RULING["unrun_arm"]` verbatim — the two
  identifiers a human must not respell.
- **P06** The report is registered in the `docs/technical/conventions.md` human-register inventory.
- **P07** Every repo-relative path the report names resolves on disk.
- **P08** The report carries no corpus identifier: no `s\d\d-<task>-<side>` capture id, no path
  segment below `videos/3-cam`, no media filename extension.
- **P09** The checker runs from `tests/` and shares one implementation with its CLI.

MAIN-owned, judgment:

- **P10** Human register spot-check: `.scratch/steq.py` over the report reports no `LONG` outside
  quoted claim text, tables, headings and fences. Quoted claims and identifiers are code surface and
  stay verbatim whatever the scan says. Recorded at close, not gated — the committed port is a
  standing polish row and stays off this unit's spine.

## 4. Invariant surfaces

1. `src/pose_estimation/calibration_qc.py` — **read-only this unit.** The report conforms to the
   constants; the constants never move to fit the report.
2. The published `calibration_qc/` tree and `qualification/` — untouched. No republication, no
   `GENERATOR_VERSION` bump.
3. `docs/technical/calibration_qc.md` — only the 6 divergent C-rows change. No schema, refusal-code,
   CLI, ownership or scope text moves.
4. Decisive-gate collection moves by exactly the new test file.

## 5. Gate identity

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <ruff check|ruff format --check|ty check|pytest>
```

Baseline `1450 passed / 0 skipped / rc=0` at `3c8a80d`. Close expects `1450 + N`, zero skipped.
Both halves of the prefix are load-bearing: `PYTHONPATH` selects the primary tree, and unsetting
`LD_LIBRARY_PATH` keeps the host OpenVINO build off the loader path.

## 6. Probe-corpus seed

Negative controls the checker must fail on, each seeded and reverted at implementation time:

1. A claim reworded by one word in the report → P01 fails, naming the claim index.
2. A claim reworded in the technical document → P02 fails.
3. `clinical validity` written into the report's prose → P03 fails.
4. A hyphenated respelling of the unrun arm followed by `ran` → P03 fails through `_fold`.
5. A renamed probe script in the report → P04 fails.
6. A dangling repo-relative path → P07 fails.
7. A synthetic capture id in the report → P08 fails.

## 7. Verdict table

Appended at close.

## 8. Amendments

Appended as ruled.

### Verdict table (appended at close)

| row | verdict | evidence |
| --- | ------- | -------- |
| D01 row set = 15 `CLAIMS`, module order, keyed C01-C15 | pass | `docs/calibration_finding.md` §3; P01 |
| D02 permitted wording verbatim | pass | P01 + P02 green; all 15 claims <=25 words |
| D03 no needle spelled, scan total | pass | P03 `0 of 23 prohibited paraphrases in 2 documents` |
| D04 sole home of `calibration_bias` numbers | pass | §3 C01-C04 carry closure/control/BA/subset figures |
| D05 technical-doc drift repaired | pass | P02 was `C03x0, C05x0, C06x0, C09x0, C12x0, C15x0` -> green |
| D06 checker committed + suite-driven | pass | `scripts/check_claim_report.py`, `tests/test_claim_report.py`, P09 |
| D07 anti-duplication | pass | no schema, CLI or refusal-code table in the report |
| D08 population beside every count | pass | §2 table + §4; two populations named apart |
| P01-P09 | pass | 9/9 green, rc=0 |
| P10 register spot-check | pass | `.scratch/steq.py --max 25`: 0 `LONG`, 0 `FILLER`; 12 advisory `PASSIVE`/`CONTRACTION` |
| NC1-NC7 negative controls | pass | 6 fire as specified; the unrun-arm control fires only on true adjacency |

### Amendments

**A01 — P07 was vacuous at first green and is now floored.** It reported
`pass ... 0 named repo paths all resolve`: the report carried no backticked repo path, so the
predicate quantified over the empty set. Widened to Markdown link targets, resolved against the
report's directory, and floored at >=1 of each kind. Reads `2 quoted paths + 3 link targets`.

**A02 — the contract's NC4 as written does not fire, and the checker is right.**
`... bias-and-pose arm ran` folds to `... bias and pose arm ran`, which is not the needle
`per_event_double_centered_bias_and_pose ran`. An intervening word defeats adjacency the constant
does not forbid. Re-driven without the intervening word: P03 fires. The seed's wording is corrected
here rather than the predicate being loosened.

---
paths:
  - "src/pose_estimation/calibration_qc.py"
  - "scripts/check_claim_report.py"
  - "scripts/make_calibration_qc_fixtures.py"
  - "scripts/check_calibration_qc_fixtures.py"
  - "scripts/check_calibration_qc_determinism.py"
  - "scripts/run_calibration_qc_mutations.py"
  - "docs/technical/calibration_qc.md"
  - "docs/calibration_finding.md"
  - "tests/test_calibration_qc*.py"
  - "tests/test_claim_report.py"
---

# `calibration_qc` — the claim set and its byte oracle

Publication contract → `publishers.md`.

## One constant, two pinned prose copies

- **`calibration_qc.CLAIMS` is the single source of truth for all 15 supported statements**, quoted verbatim by two documents: `docs/technical/calibration_qc.md` (agent register, C01-C15 under *Claim boundary*) and `docs/calibration_finding.md` (human register, the shipped report). `scripts/check_claim_report.py` P01/P02 pin both. **Never reword a claim in either document** — the publisher checks published bytes against the constant, so a reworded copy hands an editor text `_assert_claim_conformance` refuses. Independent prose copies can drift (measured without the document check: 6/15 claims).
- **The report never spells a prohibited paraphrase; it states each refused overreach by shape.** Measured: no claim contains any of the 23 `PROHIBITED_PARAPHRASES` entries under `_fold`, so quoting all 15 claims verbatim and carrying zero needles are simultaneously satisfiable — which is what keeps P03 total over both documents with no excluded span. `_fold` flattens `_` and `-` alike, so a hyphenated respelling of the unrun arm followed by an outcome word is caught; an intervening word is not, and that adjacency gap is the constant's, not the checker's.
- **A scan's own documentation must not spell the needles it forbids**, and `PROHIBITED_PARAPHRASES` stays module-side and never publishes. A document inside the scanned set carries every string it quotes. Keep the needle list in the checker, describe the rule without spelling it, and the scan stays total over every byte. An exclusion list is the wrong fix — it opens a hole exactly where data could hide.
- **A needle list is scoped to the surface it was written for.** Ranged over `docs/capture_protocol.md`, `PROHIBITED_PARAPHRASES` fires on *"This clinical-validity gap stays open"* — `_fold` flattens `-`, so the concession of a gap reads as the overreach the needle exists to catch. Scope the scan; never reword a DONE unit's shipped document to satisfy a rule it was not written for.
- **`docs/calibration_finding.md` is the sole published home of the `calibration_bias` numbers.** The publisher cites and digests that probe and ingests nothing from it, so `evidence_qc.csv` carries `bias_transfer` rows alone. Quote C01-C04's closure, control, bundle-adjustment and subset figures from the report, never from a published cell — there is none.
- `python scripts/check_claim_report.py` = 9 predicates in under a second, driven identically from `tests/test_claim_report.py` through `runpy`. The report re-derives nothing.

## The committed fixture set = this publisher's byte oracle

- `python scripts/make_calibration_qc_fixtures.py --force` then `python scripts/check_calibration_qc_fixtures.py` = **12/12 in 1.9 s**, both under the standard gate prefix.
- `inputs/upstream/` is one `qualify.run()` generation over `_canonical(90, "above")` — subject 90 keeps the capture inside the scan's synthetic namespace, and `calibration_qc.run` validates that tree standalone, so no registry, session tree or media is committed.
- Refusal matrix = **26 file-only reasons** (17 run, 9 validate) + **4 state-only**: `claim_missing`, `corpus_cardinality`, `tree_unreadable`, `output_overlap` (overlap needs a symlink the fixture set forbids). `claim_prohibited` IS file-only, reached through an arm label carrying a prohibited paraphrase.
- **The generator's `--force` deletes its destination, so it carries the publishers' ownership rule**: it refuses any non-empty destination whose `manifest.json` does not name it, and still accepts an empty directory, which is how the idempotence predicate drives it.
- **The fixture set binds to its own INPUT digests, never to `src/` source digests** — it refuses a stale golden by re-running the publisher, which has no fixpoint, rather than by comparing a recorded `source_sha256`.
- `manifest.json` carries the acceptance-contract path as a data field that `check_calibration_qc_fixtures.py` never resolves → a contract rename missing the generator leaves a dangling pointer no gate reports (→ `retention.md` contract-path bullet).

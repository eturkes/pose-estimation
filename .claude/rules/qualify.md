---
paths:
  - "src/pose_estimation/qualify.py"
  - "scripts/check_qualify_determinism.py"
  - "tests/test_qualify.py"
  - "docs/technical/qualification.md"
---

# Qualification publisher

Publication contract shared with five siblings → `publishers.md`.

- **`scripts/check_qualify_determinism.py` refuses to overwrite a result measured against different source digests** — that refusal is the stale-green barrier, so an intentional source change needs an explicit `rm -f tests/qualify_determinism_results.json` first. Its own `rm -f` barrier is separate from and additional to the publisher's version-ownership rule.
- Any schema rename must also reach the script's own fixture, which builds real sidecar rows and fails loudly with `ValueError: dict contains fields not in fieldnames` when it lags the schema.
- **`SOURCE_FILES` is the tripwire's whole reach**: a new module that shapes the published bytes is invisible to it until listed, so **every module added to `measure.AXES` joins that tuple in the same commit**.
- **Regeneration is a fixpoint** — `tests/test_qualify.py` sits in `SOURCE_FILES` *and* holds `test_m2u5_p20_committed_determinism_evidence_matches_its_sources`, so adding one test after regenerating re-breaks the evidence the regeneration just produced. Land every source and test edit first, regenerate once, commit. The 40-sweep run costs minutes, so a late red-suite harvest pays for it twice.
- **A published artifact must not contradict its own census.** Wiring a sidecar axis into one table does not populate the others (measured sync-only fill: all 193 event rows left `sync_unmeasured`). Check every governed table for that axis's unmeasured sentinel before the census claims it measured.
- **The axis census enumerates sidecar axes alone.** `unmeasured_axes = sorted(set(SIDECAR_AXES) - measured_axes)`, so a non-sidecar axis can never appear in either list — all 193 `events_qc` rows carry `geom_unmeasured` while the census is silent on it. Not a wrong number; widening it costs a `GENERATOR_VERSION` bump, a corpus republish and a determinism regeneration (`.agent/archive/polish.md`).
- **Event membership lives in `sessions/placements.csv`, never in the capture family.** A view-conflict family resolves to several single-camera events, so any family-wide derivation credits each of them with cameras it does not hold (measured overcount: `above|left` on 7 single-camera events). Where the session tree already publishes a per-event cell, copy it; re-deriving published text is how two spellings of one fact drift apart.
- Consumers re-derive rather than trust: a keyless digest detects and does not authenticate, and both `qualify.py` and `docs/technical/qualification.md` state that bound. Keep it stated.

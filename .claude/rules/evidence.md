# Evidence — how a claim earns its keep

Cross-cutting law. Every number, predicate and gate in this repo answers to it.

## Counts + populations

- **Name the population beside the count, always.** A bare count is the defect shape that survives
  review here: two closure statistics over different event sets, a contract's `329` vs the true
  `355`, a gauge sampled mid-window recorded as high-water.
- **A population count carries the instrument setting that produced it.** Rigidity's eligible 298
  moves to 311 when RANSAC sweeps 4→48 px at a fixed 20 px gate → quote a denominator with its
  threshold or not at all.
- **A gate constant never doubles as an instrument parameter.** Tell = a near-constant statistic
  inside its own gate; test = a threshold sweep before gating on any statistic, ratios included.
- **Re-derive every contract census at implementation time.** A frozen census can predate the
  predicate that changed which rows exist → grep the contract for every number quantifying over
  rows and recompute each.
- **Freeze the schema, measure the data.** Contract-owned = field-name sets, JSON sub-schemas,
  reason-code enumerations, unit vocabularies → freezing is legitimate (a membership predicate
  that never fixes its set cannot fire). Input-owned = row counts, column censuses, exclusion
  lists → the next run changes them; hard-coding one is `CLAUDE.md` `Engineering`'s
  expected-output defect.
- **Count the guards, not the columns.** Predicting a restored set from one lifted guard
  overcounts by what a second guard suppresses (predicted 17/92, measured 14/89); a partition
  ruled **measured** absorbs that with zero predicate edits.
- **Every rule over a partitioned artifact states its grain.** Run-level presence decides
  publication, cell-level emptiness decides statistics; two amendments right alone can be jointly
  inconsistent until the grain is named.

## Sizing

- **Record each number's provenance — measured / projected / assumed — and refuse to size on a
  projection whose measurement is affordable.** A 6.5 h corpus estimate from per-call
  micro-benchmarks measured 4-5× low end to end.
- **Size on the measured actual/estimate ratio, record both figures**; corrections go into the
  multiplier, never a re-argued bottom-up. A multiplier cannot absorb a structural miss: a unit
  whose halves share exactly one immutable value is two units → split there, found at planning.

## Instruments + mechanisms

- **Calibrate the instrument against known ground truth before interpreting a result.** A
  synthetic positive control driven through the real masks, sizes and population (known extrinsics
  → cycle closure 0.000 deg) is what made the real statistic readable; a null is uninterpretable
  until the response to a known input is known.
- **A discriminating statistic needs calibrated references, not a suggestive value.** Run it on
  synthetic data spanning every candidate mechanism across the magnitude range bracketing the real
  value (r = 0.703 against 0.010 noise / 0.997 bias references), and check the confound.
- **A null bounds the route it ran, never the route it skipped** → state the unrun arm beside
  every negative.
- **Test a repair's premise before building the repair**; a re-split of an existing statistic can
  refute it with no estimator written.
- **An in-family statistic certifies consistency, never correctness** (held-out reprojection on a
  solve's own keypoints → `geometry.md`).
- **A defect inferred from a derived artifact needs one measurement of its mechanism before it is
  funded**, and **re-read the logs the last measurement wrote before funding a new one**.
- **Measure a mechanism before editing anything derived from it** — this repo's law is a
  transcription too: `gates.md` once called both prefix halves load-bearing, and a defect report
  built on that wording targeted a recipe that measures green.

## Validators + witnesses

- **A validation function is never a byte witness.** Semantic validators canonicalize, so
  same-semantics rewrites pass. A read-only-over-upstream claim needs a recursive non-following
  pre/post snapshot (relative path, entry kind, symlink target text, regular-file digest) plus
  validity as a separate predicate; non-following matters because `videos`, `inventory` and
  `renv/library` are symlinks in teammate worktrees.
- **A keyless digest detects, it does not authenticate.** Anyone can edit a claim and recompute
  it → never let a passing check read as tamper-proofing; `qualify.py` +
  `docs/technical/qualification.md` state the bound.
- **Committed evidence binds to source digests, never a head SHA** (a regeneration always precedes
  its commit, so a recorded HEAD names the parent). **Regeneration is a fixpoint → the last
  action before the commit**: a test file inside `SOURCE_FILES` edited after regenerating
  re-breaks it.
- **A green predicate whose detail line reports zero items is failing.** Every quantified
  predicate carries a `nonvacuous` floor; read a validator's detail line, never its rc alone.
- **A case that grades a stand-in grades nothing.** Encode predicates over shipped behaviour
  against the shipped symbol — a re-implemented verdict passed while the shipped validator was
  broken, and a name match is a spelling test a comment passes.
- **Re-derive, never transcribe**: parse reason codes, schema headers, filters, markers, key
  patterns, strata, allowlist call sites and superseded figures from source at check time.
- **A negative-control seed is a claim; grade it against the predicate text when the contract is
  written.** Idempotence can never detect a stable wrong order.
- **A frozen predicate is the artifact most likely to be wrong**: a diff-blind grading pass found
  11 contract defects and 0 driver defects in one unit → budget one per `kernel` unit and expect
  its findings to land in the contract.

## Sweeps

- **A repo-root sweep skips `.claude/rules/` + `.agent/`** (`rg`/`ugrep` skip hidden dirs) → sweep
  law with `--hidden --glob '!.git/'` or name each dot-dir, positive control drawn from that same
  search space.

## Context records

- **Name the reading — last turn or high-water — and size in tokens, never percent**: recorded
  gauges here straddle 240K, 273K and 1M denominators. High-water = max assistant-turn
  `input + cache_creation + cache_read + output`, last executor pass where `usage.iterations`
  lists passes; an auto-compaction splits it into two regimes → report the pre-compaction peak.
- **The harness sheds MAIN's content silently** → re-derive from the file, gate or commit before
  acting on a claim whose supporting read happened earlier in the session.

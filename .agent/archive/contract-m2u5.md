# M2.5 acceptance contract — Cross-view alignment

Frozen at wave-1 close. Baseline `03a6e1c`: `ruff`, `ruff format`, `ty` clean and **1201 passed /
0 skipped** in 876.49 s, outcome characters counted from the run (1201 `.`, nothing else).

## 1 Problem

Every camera in the 193-event session tree declares `sync_offset: 0` — an integer frame count that
asserts nothing. The corpus has no rig and no genlock: three hand-held tablets were started and
stopped by hand, so real per-camera starts spread by a median 2.56 s and reach 72.8 s within one
event. A zero is not a neutral default there; it is the only alignment claim any consumer can read.

The offsets are already measured. M2.3 recovered them pairwise — 246 within-family pairs, 201
accepted under R6 — and `qualify` already solves one offset per event member by breadth-first
search over accepted edges. It publishes three derived statistics (`graph_connected`,
`offset_span_s`, `closure_residual_s`) and **discards the per-camera solution that produced them**.
M2.5 promotes that discarded solution into a published, validated, consumable artifact.

## 2 Scope

In: a per-camera offset table inside the existing qualification set; the solver, reference rule and
sign convention that produce it; the event-level QC that already exists, made consistent with it; a
real-corpus republish; the documentation obligations the unit invalidates.

Out, and deliberately:

- **A fourth publisher.** `map-m2u5` S14 priced it: `ffcfdfa`, the exact construction analog for a
  new publisher module plus console script plus doc plus suite, is **+1424/−0 over 5 files**, and
  that is the construction floor before trust-root, crash-state, mutation and determinism work. The
  schema-widening analog `5e40922` is **+601/−170 over 10 files** with every assurance gate already
  standing. M2.5's spine is alignment, not publication scaffolding.
- **Writing offsets back into `sessions/`.** S09 measured the cycle: `sessions` → `qualify` →
  alignment → `sessions`. Patching published manifests after qualification changes the session tree
  digest, which invalidates qualification's recorded upstream snapshot, which invalidates the
  alignment input. Materialized session manifests may never be the authoritative derived carrier.
- **The fusion frame reader.** `multicam`'s integer skip loop and `raw − sync_offset` logical index
  stay untouched. Nothing downstream of them is reachable until a calibration exists, which is M2.6.
- **A clock-rate or drift term.** M2.3 R5: 0/132 qualified drifts move alignment by more than one
  frame.
- **Per-camera uncertainty.** Ruled out by measurement, not by omission — see P11.
- **Take resolution for the 7 unresolved families**, and metric scale, which R3 closed negative.

## 3 The findings that shape the unit

**The solver choice is priced and small, and the weighting choice is not a preference.** Unweighted
least-squares over every accepted edge differs from the shipped breadth-first tree by a median of 0,
p95 3.708 ms and **max 10.095 ms = 0.303 nominal frame**, moving 60 of 329 solved cameras, every one
of them inside the 30 events that carry a redundant edge. Least-squares uses all the evidence and
distributes closure error instead of charging it to whichever edge the traversal happened to take.

Confidence weighting is **rejected on measurement**. A precision weight must rise as error falls;
Spearman correlation of published `peak_rms` against absolute audio-visual disagreement over the 65
pairs both estimators accepted is **+0.4141**, the wrong sign, and `peak_ratio` is +0.0659. Neither
is an inverse-variance estimate, so weighting would dress an uncalibrated number as precision.

**The sign is proved rather than assumed.** A deterministic oracle built two 20 s PCM clips from one
random signal with camera B leading by 375 ms, ran the shipped `audio_offset` estimator in both
orders, and composed the result through the solve: A→B `+0.375000058` s, B→A `−0.375000058` s,
antisymmetry error 0, recovered composed offset within a 0.5 ms tolerance of the constructed lead. A
sign flip would produce a fully connected, digest-valid artifact whose cameras move twice as far
apart in time, and no structural check could see it (S08).

**No per-camera uncertainty is defensible from this solve.** 74 connected two-camera events carry
1 edge for 1 free offset and 11 connected three-camera trees carry 2 for 2 — **0 residual degrees of
freedom on 85 of the 115 connected multi-camera events**. Only the 30 closed triangles reach 1 df,
and there the independent, homoscedastic, zero-mean model is violated by correlated acoustic-path
and rolling-shutter bias, which closure is structurally blind to because propagation delay cancels
around a cocycle. Publishing a standard error here would publish a number the data cannot support.

**A reference rule must be total over 193 events, and only two candidates are.** Lowest `asset_id`
is total but semantically arbitrary. Latest-start is defined on 173/193 — undefined for exactly the
20 events whose graph does not connect, which is where a reference is most needed. Highest degree is
unique on only 69/193. The view hierarchy `above` > `left` > `right` is **total at 193/193** (155
above, 24 left, 14 right) and names a reference a reader can interpret.

## 4 Predicates

Carrier and schema.

- **P01** Alignment publishes as `qualification/cameras_qc.csv`, inside the existing qualification
  set. No new publisher, no write-back into `sessions/`.
- **P02** `qualify.GENERATOR_VERSION` moves `v3` → `v4`; the new filename joins `CSV_FILENAMES` and
  `GENERATION_KEYS`, and the marker digests it as a payload like every other table.
- **P03** Columns are `event_id`, `asset_id`, `camera_name`, `view`, `offset_s`, `offset_status`,
  `is_reference`, `reference_camera`. Every cell alphabet is declared and matched with `fullmatch`.
- **P04** One row per placed asset of every event — 379 rows over 193 events — so a camera that has
  no offset is published as a row that says so, never as an absent row.

Solver.

- **P05** Offsets come from an unweighted least-squares solve of `x_b − x_a = offset_s` over accepted
  edges, gauge-fixed by pinning the reference at `0`. Each connected component's incidence system is
  rank-deficient by exactly 1, and the pin is what removes it.
- **P06** Accepted edges are exactly `QUALIFIED_PAIR_STATUSES` = `{ok_corroborated,
  ok_uncorroborated}`. R6 is unchanged: `visual_only` is never usable, estimators are never averaged,
  and the visual estimate is never a fallback value.
- **P07** The connected component holding the reference is solved and published. A camera outside it
  publishes an empty `offset_s` with `offset_status = unreachable`, and the event additionally
  publishes `sync_status = unconnected` beside `graph_connected = 0`, so no consumer can read a
  partially solved event as an aligned one. All-or-nothing event failure is retired: 10 of the 20
  unconnected events hold three cameras with one accepted edge, so a usable two-camera alignment
  currently reads as total failure.
  **Ruled against `spike-m2u5-solve` Q08**, which recommended nulling every member of a failed event
  including its nominated reference. Its stated hazard — a consumer treating a partial solve as event
  alignment — is what the event-level `sync_status` closes, and discarding 10 recoverable two-camera
  events to prevent a misreading that a published flag already prevents is the more expensive error.
  The ruling is conditional on one unmeasured quantity and reopens if it falls: **in how many of the
  10 three-camera failures does the view-hierarchy reference sit inside the two-camera component**.
  If that count is 0, P07 yields nothing and collapses into Q08's recommendation.
- **P08** Reference = view hierarchy `above` > `left` > `right`, tie-broken by lowest `asset_id`.
  Exactly one row per event carries `is_reference` true, its `offset_s` is exactly `0`, and every row
  of that event names it in `reference_camera`.
- **P09** `offset_s = t_camera − t_reference` for one shared instant; positive means that camera
  started earlier; the application transform `t_ref = t_camera − offset_s` is stated beside the field
  in code and in the schema doc.
- **P10** No confidence weighting, with the rejection recorded beside its measurement.
- **P11** No per-camera uncertainty column. Event-level triangle closure stays the published
  self-consistency statistic, and stays labelled as self-consistency rather than accuracy.
- **P12** No rate or drift term reaches the schema.

Consistency inside the published set.

- **P13** `events_qc.offset_span_s` is derived from the published per-camera offsets, so the two
  tables cannot disagree. `closure_residual_s` stays a pure function of the accepted edge set and is
  unaffected by the solver change.
- **P14** `graph_connected` keeps its P19 meaning — every camera of the event reachable — so its
  173/193 census is unchanged by P07's partial publication.
- **P15** Every derived cell recomputes from published columns alone: a test rebuilds `offset_span_s`,
  the reference-zero invariant and the one-reference-per-event invariant from `cameras_qc.csv`.

Consumers and the unmeasured zero.

- **P16** `sessions.render_manifest` keeps `sync_offset: 0`, and a test pins that the generator can
  never write a non-zero value there. The two quantities are named apart rather than merged:
  `sync_offset` is the legacy integer pre-roll trim in the fusion reader's frame domain, and
  `qualification/cameras_qc.csv` is the authoritative time-domain alignment. Removal was priced and
  refused — `multicam` reads an absent field as `0` (S04), so removal moves the zero from explicit to
  implicit while forcing a second generator bump and a 193-manifest republish.
- **P17** Documentation obligations the unit invalidates are all closed: `docs/technical/multicam.md`
  and `docs/technical/validation.md` stop calling audio cross-correlation FUTURE (O11, O25);
  `docs/technical/sessions.md:118` stops saying M2.5 will fill `sync_offset` (O09);
  `docs/technical/qualification.md` gains the `cameras_qc` schema, the sign convention and the
  application transform; and the capture guidance stops implying that integer trim or raw frame
  parity proves the delivered sub-frame alignment (O24).
- **P18** Every timing limit survives every edit: rolling shutter stays a 0–33.33 ms sweep and is
  never called negligible; AAC priming is quoted as the measured 0 ms and never as the predicted
  3.891 ms; closure certifies self-consistency and never accuracy (O19, O22).

Evidence and gate.

- **P19** Real-corpus republish: 379 rows over 193 events, 329 carrying an offset and 50 not; every
  event's reference row exactly `0`; the census reproduced by an independent recount.
- **P20** `scripts/check_qualify_determinism.py` regenerates green after its `rm -f` barrier, and any
  new module that shapes published bytes joins its `SOURCE_FILES` tuple in the same commit.
- **P21** Full suite green in the primary tree — 0 skipped, xfailed, xpassed, deselected or errored —
  at a count of at least 1201.

## 5 Invariant surfaces

1. **R6 pair policy.** Which pairs are accepted is not this unit's business and does not move.
2. **Sign antisymmetry and the reference zero.** Reversing a pair negates its edge; the reference is
   exactly `0`.
3. **Published-cell reproducibility.** Every derived cell recomputes from published columns alone.
4. **Determinism.** The qualification set is a function of its validated inputs, not of locale, hash
   seed, time zone or output name.
5. **Timing-limit language.** Rolling-shutter and closure caveats survive every documentation edit.

## 6 Gate identity

`env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync pytest`, primary tree. Both halves are
load-bearing: without them `conftest.py` dies at `ImportError … GLIBC_2.43 not found` before
collecting one test. Baseline `03a6e1c` = **1201 passed / 0 skipped** in 876.49 s.

## 7 Probe-corpus seed — 8 classes

1. Singleton event — the reference is itself, `offset_s` exactly `0`, no edges.
2. Two-camera connected event — one edge; the solve must reproduce the published pair offset exactly.
3. Three-camera closed triangle — a redundant edge; least-squares distributes the closure residual.
4. Three-camera open path — two edges; least-squares must reduce to the tree solution exactly.
5. Disconnected event — a camera outside the reference's component publishes `unreachable`.
6. Reference-rule exercise — an event without `above`, one without `above` or `left`, and a view tie.
7. Sign oracle — a synthetic pair with a constructed lead, estimated in both orders and composed.
8. Real corpus — 193 events, 379 placed cameras.

## 8 Amendments

Filled by MAIN's batch rulings on the `test-m2u5` phase-1 table.

Open at freeze, carried into wave 2:

- **A01 (open).** P07's conditional measurement above. `spike-m2u5-solve`'s retained worktree already
  enumerates the component structure of all 20 failures, so this is one query against
  `scripts/probe_alignment_solver.py` `q08_unconnected_events`, not a new investigation.

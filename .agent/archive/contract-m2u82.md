# Contract — M2.8.2, full corpus 2D run

Frozen at base `e97d417`. Tier: spine artifact = `data`; **D01's shipped-code fix carries `kernel`
assurance** (diff-blind red suite + MAIN rerun), because it is judgment-bearing code in a shipped
module. Planning sized this unit as delivery; the unit's own first question turned it into a defect
fix plus delivery, and §10 records that as the sizing datum.

## 1. Spine

379 canonical assets → per-asset 2D clinical features + a run manifest giving **every** asset an
explicit disposition. Resumable. Cohort aggregation is M2.8.3 and is out of scope here.

## 2. The unit's first question, closed: the 40× per-asset split is one defect, not a cost profile

M2.8.1 measured per-asset cost bimodal at 40× with equal yield — 6/16 assets at 7.2-12.2 ms/frame,
10/16 at 338.6-543.5 — and named per-detected-box cost as the leading candidate. **Both candidates
are refuted. Both bands are the same upstream state-machine defect in `rtmlib.PoseTracker`.**

`PoseTracker.__call__` reorders the current frame's keypoints by *persistent track id*:

```
self.track_ids_last_frame = track_ids_current_frame
try:
    keypoints = np.array([keypoints[i] for i in self.track_ids_last_frame])   # id used as index
    scores    = np.array([scores[i]    for i in self.track_ids_last_frame])
except:                       # noqa — upstream bare except
    return keypoints, scores,          # returns BEFORE the two state updates below
self.bboxes_last_frame = bboxes_current_frame
self.frame_cnt += 1
```

`track_by_iou` assigns `track_id = self.next_id; self.next_id += 1` on any unmatched box with
`area >= MIN_AREA`. One missed IoU match therefore mints an id ≥ 1, and `keypoints[1]` on a
one-person frame raises `IndexError` → the early return fires → **`frame_cnt` and
`bboxes_last_frame` freeze for the remainder of the source**. Every later frame retries the same bad
index, so the freeze is permanent, and the returned pre-reorder keypoints keep yield near 1.0, which
is why nothing downstream looked broken.

**The residue of the frozen counter selects the band**, `det_frequency = 7`:

| freeze residue | detector cadence after freeze | bbox list | measured band |
| -------------- | ----------------------------- | --------- | ------------- |
| `frame_cnt % 7 == 0` | **every frame** | fresh, correct | 338.6-543.5 ms/frame — correct output, ~6× the necessary cost |
| `frame_cnt % 7 != 0` | **never again** | drains to empty via `track_by_iou`'s pops | 7.2-12.2 ms/frame — **corrupted output** |

The low band is not fast, it is wrong. `RTMPose.__call__` opens with
`if len(bboxes) == 0: bboxes = [[0, 0, image.shape[1], image.shape[0]]]`, so a starved box list makes
a top-down pose model estimate from the **whole 1080p frame** instead of a person crop, at
confident-looking scores.

**Synthetic reproduction, stub det/pose models, 140 frames, `det_frequency = 7`**, one induced IoU
miss whose frame index selects the residue:

| scenario | `frame_cnt` | detector calls | whole-frame pose calls | modelled ms/frame |
| -------- | ----------- | -------------- | ---------------------- | ----------------- |
| no miss, `tracking=True` | 140 | 20 | 0 | 58.0 |
| miss → residue ≠ 0 | **3 (frozen)** | 1 | **135/140** | 10.5 |
| miss → residue 0 | **7 (frozen)** | 134 | 0 | 343.0 |
| miss, `tracking=False` | 140 | 20 | 0 | 58.0 |

Cost model = detector 350 ms on detector frames + pose 8 ms/frame. It reproduces both measured bands
from the same single defect and shows the fix immunising both.

## 3. Design decisions

- **D01 — rtmlib IoU tracking is disabled: `PoseTracker(..., tracking=False)`.** The stateless branch
  (`not self.tracking and self.det_frequency != 1`) skips the reorder entirely, so no id is ever used
  as an index and `frame_cnt` always advances. Redundancy is what makes this safe rather than a
  capability loss: `KeypointSmoother` already owns temporal association through Hungarian
  `gated_assignment` (`src/pose_estimation/smoothing.py`), and rtmlib's tracker contributed only the
  IoU drop of unmatched people, which `--single-subject` overrides by taking the confidence argmax.
  Not patching upstream's reorder: the project needs no track ids from rtmlib, so removing the
  dependency beats forking a routine whose output is discarded.
- **D01a — `--det-frequency 1` stays legal and needs no guard.** It re-enters the tracking branch, but
  a freeze there is harmless: `frame_cnt % 1 == 0` holds at every residue, so the detector runs every
  frame and the box list is never starved. The only lost effect is the reorder, which D01 discards
  anyway. P04 pins the claim rather than asserting it.
- **D02 — corpus-run sizing is MEASURED post-fix, never projected.** M2.8.1's 26-31 h and the plan's
  6.5 h both described a run that no longer exists: the first measured a frozen tracker, the second
  extrapolated per-call latency. The post-fix pilot re-run is the only admissible input to the corpus
  estimate, and the estimate is labelled `measured` with its sample.
- **D03 — the detector stays on CPU for the corpus run.** GPU detection is 22× faster on the synthetic
  probe (9.7 ms vs 213 ms median) but pads its dynamic in-graph-NMS output to a fixed 100-row buffer;
  CLAUDE.local.md requires a per-device shape-and-range qualification before any dynamic-output model
  is trusted, and GPU zero-fill has not been qualified against the live detector post-processing. A
  corpus run is the wrong place to qualify a device. Deferred to `.agent/polish.md` with its
  acceptance check.
- **D04 — resumability grain = the event (session), not the asset.** `process_session` runs every
  camera of a session in one call, so an asset cannot be run alone (M2.8.1's ruling, unchanged).
- **D05 — resume is keyed on a per-event completion marker, never on output presence.** A killed run
  leaves a partial landmark CSV that is indistinguishable from a complete one by existence or by row
  count, because the true row count is not known until the source is fully decoded. The marker is
  written after the event's outputs are complete.
- **D06 — the manifest is total over the registry's canonical assets and its disposition vocabulary is
  frozen.** Every asset reaches exactly one disposition; a new failure mode earns a code rather than
  an absent row. This is M2.8.1's partition discipline applied at the corpus grain: an asset silently
  missing from a denominator is the defect the artifact exists to prevent.
- **D07 — outputs are patient-adjacent.** The run tree and every landmark CSV live outside git, and
  every report obeys M2.8.1's allowlist redaction: emitted strings are published stratum labels, R
  reason codes, or frozen disposition codes; keys read as code-authored field names.

## 4. Predicates

| id | predicate |
| -- | --------- |
| P01 | `run.py` constructs `PoseTracker` with `tracking=False`. |
| P02 | Characterization of the upstream defect: with `tracking=True`, one induced IoU miss freezes `frame_cnt` and `bboxes_last_frame`; with `tracking=False` on the identical stimulus both advance for all frames. |
| P03 | Under the shipped construction no frame reaches the pose model with an empty bbox list, so no whole-frame pose call occurs on a source that detects a person. |
| P04 | Under the shipped construction the detector call count equals `ceil(frames / det_frequency)`; under `det_frequency = 1` it equals `frames` whether or not a miss occurs (D01a). |
| P05 | Resume is idempotent: a second driver run over a tree whose events all completed performs zero inference and leaves every output byte-identical. |
| P06 | Resume does not credit partial work: an event interrupted before its marker is re-run from scratch on the next pass. |
| P07 | The manifest carries exactly one row per canonical registry asset — 379 rows, no duplicate, no absent asset. |
| P08 | Manifest dispositions partition the asset set: every row's code is in the frozen set, and the per-code counts sum to the row count. |
| P09 | Every `ok` asset has a landmark CSV and exactly one source-diagnostics row; no non-`ok` asset claims either. |
| P10 | Every landmark CSV that reaches the R feature stage yields a group-disposition artifact, header-only or populated (M2.8.1 D05 at corpus grain). |
| P11 | The run writes nothing inside the published session tree and leaves `sessions` `tree_digest` unmoved end to end. |
| P12 | Per-asset CFR counters (`pts_accepted`, `index_fallback`, `monotonic_forced`) are recorded for every `ok` asset and the pooled fallback rate is published. |
| P13 | Report redaction is an allowlist: every emitted string is a published stratum label, an R reason code or a frozen disposition code, and every key matches `[a-z][a-z0-9_]*`. |
| P14 | The corpus-run cost is recorded as `measured` with its measuring sample named, and no projected figure is carried as a budget. |

## 5. Invariant surfaces

`src/pose_estimation/run.py` (tracker construction, frame loop), `src/pose_estimation/multicam.py`
(`process_session`, `process_source`), `src/pose_estimation/video_io.py` (`SourceTimestampClock`
counters), `analysis/clinical_features.R` (group disposition), the published `sessions/` tree
(read-only to this unit).

## 6. Gate identity

`env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync` × {`ruff check`,
`ruff format --check`, `ty check`, `pytest`}, primary tree, zero skipped.

## 7. Negative-control seed

| id | seed | must fire |
| -- | ---- | --------- |
| N1 | revert `tracking=False` | P01, P02, P03 |
| N2 | force `det_frequency = 1` with tracking restored | P04 |
| N3 | delete one event's completion marker after a full run | P05 |
| N4 | truncate a landmark CSV and leave the marker | P06 detects nothing — **ruled out of scope**: the marker is the contract, and byte-verification of a patient-adjacent CSV is M2.8.3's consumer problem |
| N5 | drop one asset row from the manifest | P07 |
| N6 | spell an unlisted disposition code | P08 |
| N7 | emit a capture id in the report | P13 |

## 8. Amendments

**A01 — D02's measured corpus cost is 7.07-7.61 h, from the post-fix pilot re-run.** Same instrument,
same 8 events / 16 assets / 8971 reported frames, same configuration (`rtmw-l`, `hands-arms`,
`--single-subject`, det CPU / pose NPU, `--det-frequency 7`), out tree `.scratch/pilot-m2u82/` kept
separate from M2.8.1's `.scratch/pilot-m2u81/` so both populations stay re-readable.

| | pre-fix | post-fix |
| - | ------- | -------- |
| per-asset ms/frame | 7.2-12.2 **and** 338.6-543.5 (bimodal, 40×) | **58.6-92.2** (unimodal, 1.57×) |
| run wall | 2959.56 s | **729.05 s** |
| fps incl. startup / steady | 3.031 / 3.598 | **12.305 / 13.253** |
| corpus hours incl. startup / steady | 30.89 / 26.03 | **7.61 / 7.07** |

Unchanged and therefore re-confirmed: CFR pooled fallback rate **0.000000** over 8971 decoded frames,
group partition total at 16 input = 16 windowed + 0 dropped, guard refusals 8/8 events. **Window rows
moved 572 → 540**, which is the expected tell rather than a regression: the pre-fix run computed part
of its features from whole-frame pose, so the keypoints feeding window enumeration genuinely changed.

**A02 — the pilot's per-frame latency log is a standing diagnostic, not just a progress trace.**
`run.py` prints `Frame … | <dt> ms | …` at `frame_idx <= 5 or frame_idx % 50 == 0`; grouping those
`dt` values by `(frame_idx - 1) % det_frequency` separates detector frames from pose-only frames and
makes the tracker's cadence directly observable. That is what closed §2 with no new decode. Any
future claim about run cost checks this split before it hypothesises a workload cause.

**A03 — P02's characterization pins upstream behaviour, so it may legitimately go green on an rtmlib
upgrade.** If a future rtmlib fixes the reorder, `tracking=True` stops freezing and P02's freeze
assertions fail. That is the predicate working: it is a tripwire on the dependency, and its failure
is a signal to re-rule D01, not a defect in the suite. The row is not to be weakened into a tautology.
`test-m2u82` reached the same ruling independently and marked P02 `encoded-green`, reproducing §2's
table exactly including the modelled ms/frame column.

*Amendments A04-A06 are rulings on `test-m2u82`'s diff-blind findings, harvested at window-1 close.*

**A04 — N2 is a mis-specified negative control and is REPLACED.** N2 said "force `det_frequency = 1`
with tracking restored ⇒ P04 fires". Measured: P04 is green at `det_frequency = 1` under **every**
induced miss, so N2 could never fire it. This is not a P04 defect — it is D01a's claim holding, and
the seed contradicted the design decision it was written beside. **N2 becomes: restore `tracking=True`
at `det_frequency` in {2, 5, 7, 13} ⇒ P04 fires on each.** The `det_frequency = 1` case stays in the
suite as a *positive* control for D01a rather than as a negative control.

**A05 — P07 needs a second clause, because row-set equality is not row-set identity.** As frozen, P07
reads "exactly one row per canonical asset — 379 rows, no duplicate, no absent asset". A manifest that
duplicates one asset and drops another has the same 379-row count **and the same row set**, so a set
comparison passes it. P07 is amended to require, as separate conjuncts: row count 379, key set equal
to the registry's canonical asset set, **and keys unique**. The uniqueness clause is the one that
carries the defect; the other two cannot see it.

**A06 — D06's frozen disposition vocabulary must ship as a machine-readable constant in `src/`, and
that is a window-2 implementation obligation.** The contract froze a vocabulary that exists nowhere in
`src/` or `scripts/`, so no gate can re-derive it and P08 is currently unsatisfiable by construction.
Same shape as M2.8.1's `GROUP_QC_REASONS`, which is re-derived from the R source at check time rather
than transcribed: **a frozen value set that lives only in a contract is a stale number waiting to
happen.** The manifest writer and its validator both read the constant; neither restates it.

*Amendments A07-A12 rule `test-m2u82-2`'s 11 diff-blind findings, harvested at window-2 close. Every
one of the 11 is a defect in this contract; none is a defect in the driver. F6 (pooled rate as an
unweighted mean of per-asset rates) and F2/F4 below were already closed by the driver's design and
ship encoded-green.*

**A07 — P11 names two witnesses and both are blind to the one file inside the tree (F3).** `tree_digest`
excludes `generation.json` because a document cannot digest itself, and `validate_generation` compares
that document's *fields* — so reindenting or reordering the marker rewrites bytes inside the published
tree while both witnesses stay green. Both exclusions are correct. **P11 gains a third witness:
`sessions.generation_digest`, the marker's own bytes**, and the run publishes
`generation_marker_unmoved` beside `generation_digest_unmoved`. F4 (the containment guard disarms on a
tree carrying no marker) needs no change: the driver calls `validate_generation` before it resolves any
output, so an unmarked tree is refused before the guard is reached.

**A08 — P12's `derived, never stored` is refused; the counters are authoritative instead (F5).** The
M2.8.1 diagnostics schema stores both quantities P12 calls derived: `cfr_fallback_rate` under its own
name and `n_timestamps` under the alias `n_frames_decoded`, which a reader will take for a decode
count. Dropping either is a schema change to a published artifact, and the alias is the true name of
what is counted. **P12 is amended: the three counters are the authoritative record, and every stored
derived value must equal its derivation** — written from the clock property of that quantity, never
recomputed at the call site. The run publishes `stored_rate_equals_its_derivation` per `ok` asset.

**A09 — P13's key clause is a shape test, so it is a denylist wearing an allowlist's name (F7, F8,
F9).** Three separate defects in one predicate. A key matching `[a-z][a-z0-9_]*` admits every
identifier of that shape, so N7's capture id is refused as a *value* and admitted as a *key* — one
string, two verdicts (F7). The analog's own `_coverage` block keys by integer stratum value, which no
such pattern can match, so P13 as frozen refuses a report shape M2.8.1 already publishes (F8). And the
three admissible value classes cannot name the generator, its version or the four device echoes, so a
conforming report could not say which program wrote it (F9). **P13 becomes one composite rule, both
placements decided by membership: a value is admissible iff it is a published label — stratum label, R
reason code, disposition code — or a code-authored constant the emitting program spells at its own
call site; a key is admissible iff it is a frozen field name of the report schema or a published
label.** The driver's call site already enumerates exactly this union. The analog's
`_assert_redacted` keeps its weaker allowlisted-OR-matching guard as a runtime backstop; the suite is
the membership oracle.

**A10 — two suite defects MAIN found while driving the reds, both of the same kind: a case that grades
a stand-in grades nothing.** P08's discovery pattern `^[A-Z][A-Z0-9_]*DISPOSITION[A-Z0-9_]*\s*=`
cannot match `ASSET_DISPOSITIONS: tuple[str, ...] = (`, which is the shape of shipped
`SOURCE_DIAGNOSTIC_FIELDS` — so it reported an absent constant that was present, and finding a *name*
is a spelling test a comment passes anyway; the case now reads the published set. P07/P08's stand-in
graded a local `_manifest_verdict` re-implementation, so a broken `validate_manifest` passed it; two
cases now drive the shipped validator over five mutations. **Rule for this contract: a predicate over
shipped behaviour is encoded against the shipped symbol, never against a local model of it.**

**A11 — P10's scope term is ruled: every landmark CSV R processes *past its input-type gates* (F1),
and the event is the isolation grain (F2).** R's main loop reaches the disposition write past two
exits, and neither judges a run: `next` fires when `detect_tracking` reports hands-only, which carries
no arm keypoints to dispose of, and `stop(` fires when a 3D input names no single capture identity,
which the 2D corpus route cannot reach. Both are decided from `names(df)` alone, so they classify the
input rather than report an outcome, and an input R never processed owns no disposition. An exit added
on a *processing* outcome would be a real hole, and that is what the encoded case fails on. F2 is
correct that `stop()` is process-scoped — at directory grain one rejected asset voids its event's
whole clinical pass — and needs no code change: `_attempt_event` reads R's exit code and
`asset_disposition` lands **every** asset of that event on `clinical_failed`, a published code, so the
loss is recorded rather than silent and the next event is untouched.

**A12 — judgment-bearing code moves out of the driver script into `src/`, because a gate backing a
durable claim must exercise the shipped decision.** A07 and A11 are both predicates over a decision
the driver script made privately, and `scripts/corpus_run_2d.py` is not importable by the suite. The
marker witness ships as `sessions.generation_digest` beside `tree_digest`, and the D06 partition rule
ships as `corpus_run.asset_disposition` beside the frozen vocabulary it reads, with `STAGE_RUN` /
`STAGE_CLINICAL`. The driver keeps orchestration alone.

**A13 — D02's corpus cost is 8.702 h MEASURED over the whole corpus, and the pilot projection was
14% low.** Run wall 30 666.53 s + clinical 660.90 s over 193/193 events, 337 090 decoded frames,
**10.99 fps including startup**. A01's post-fix pilot projected 7.07-7.61 h from 8 events / 16 assets
/ 8971 frames; the corpus ran 10.7% slower per frame than that sample. **Stratification covers axes,
not per-frame cost** — the pilot draw is built for codec / device / rotation coverage, and nothing in
it constrains the cost distribution. A01's band stands as what the pilot measured; 8.702 h is what
the corpus cost, and it is the figure any successor sizes against.

Found and fixed at close: the published throughput block divided **all-corpus** decoded frames by
**this invocation's** wall seconds, so the resumed final pass reported 13.88 fps / 6.903 h — a rate
the pipeline never reached. The wall now sums each event's marker (`run_s` / `clinical_s`), which is
correct across any number of passes, and `sample` reads `corpus` only when
`events_measured == events_total`. **Datum: a per-invocation accumulator paired with a whole-corpus
numerator becomes a false rate the moment the job is resumable.**

## 9. Verdict table

Run evidence = `output/corpus-2d/run_report.json` (gitignored, patient-adjacent), republished by
`scripts/corpus_run_2d.py --analyse-only`. Suite = `tests/test_corpus_run_2d.py`, **104 cases, 0 red**.

| id | verdict | evidence |
| -- | ------- | -------- |
| P01 | pass | `run.py` constructs `PoseTracker(..., tracking=False)`; encoded + probe-backed. |
| P02 | pass (encoded-green, A03) | `scripts/probe_tracker_freeze.py`, 5 scenarios × 7 verdicts, rc=0. |
| P03 | pass | probe: 0 whole-frame pose calls on both stimuli under the shipped construction. |
| P04 | pass | cadence swept over `det_frequency` ∈ {1,2,5,7,13}; N2 as amended by A04. |
| P05 | pass | resume idempotence; the whole-corpus pass skipped 25 already-complete events and re-ran none. |
| P06 | pass | marker-keyed resume; `_attempt_event` `rmtree`s before re-attempting. |
| P07 | pass (A05) | manifest 379 rows, key set = canonical set, keys unique; `validate_manifest` driven over 5 mutations. |
| P08 | pass | census `ok` 379, every other code 0, counts sum to 379; vocabulary read from `ASSET_DISPOSITIONS`. |
| P09 | pass | artifacts 0 missing CSV / 0 wrong diagnostics / 0 trespass over 379 assets. |
| P10 | pass (A11) | both pre-write R exits proved input-type gates; `asset_disposition` isolates a failed event at `clinical_failed`. |
| P11 | pass (A07) | `generation_digest_unmoved` + `generation_marker_unmoved` + `tree_digest` unmoved end to end. |
| P12 | pass (A08) | CFR pooled 0.0 over 337 090 frames; `stored_rate_equals_its_derivation` true per `ok` asset. |
| P13 | pass (A09) | composite membership rule, both placements; report emits published labels + call-site constants only. |
| P14 | pass (A13) | 8.702 h `measured`, `sample: corpus`, 193/193 events named; no projection carried as a budget. |

Negative-control seed: N1, N3, N5, N6, N7 fire as written; **N2 replaced by A04**; **N4 ruled out of
scope** at freeze. Run verdicts, all true: `manifest_total`, `every_event_complete`, `artifacts_owned`,
`group_disposition_published`, `partition_total`, `partition_disjoint`, `group_qc_header_frozen`,
`counters_classify_every_frame`, `stored_rate_equals_its_derivation`, `generation_digest_unmoved`,
`generation_marker_unmoved`.

**Branch tips — MILESTONE-REVIEW dispatch inputs.** `wt/test-m2u82` retained at **`c92ffcc`**
(worktree removed): `3997a69` P01-P03, `c6937c2` P04-P06, `ecf32fd` D06 validator (window-1 harvest
point), `fbb5c2c` P10, `78cdfc6` P11, `f0010b9` P12, `6521ee5` P13, `c92ffcc` P14. Reports at
`.scratch/agents/test-m2u82.md` and `.scratch/agents/test-m2u82-2.md` (gitignored; the findings
themselves are A04-A13 above). No `orc` or `diff` teammate ran: the unit's spine is a data artifact
behind a validator, and the contract's own predicates are the differential instrument.

## 10. Sizing

Planning sized M2.8.2 at 1-2 windows as delivery over a working pipeline. The pipeline was not
working. **The unit's named first question was the whole unit**, exactly as M2.6's was, and the
answer arrived from evidence M2.8.1 had already committed — the pilot's per-frame latency logs,
re-read at the detector cadence, separated detector frames from pose-only frames and made the
freeze visible without decoding one new frame. Datum: **before funding a measurement, re-read the
logs the last measurement already wrote.**

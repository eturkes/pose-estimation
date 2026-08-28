# M2.3 verdict table — capture qualification + 3D-route ruling

MAIN's rulings on the contract's open questions (`.agent/archive/contract-m2u3.md` §7). Each ruling
binds for the milestone; new evidence is what reopens it. Amend in place — a superseded ruling left
standing poisons every artifact that reads this table.

Evidence pointers: `main:F<n>` → `.scratch/agents/main-checkpoint-m2u3.md`; `geom:U1` →
`.scratch/agents/prod-m2u3-geom.md`; `detect` → `.scratch/agents/prod-m2u3-detect.md`;
`audio:U<n>` / `visual:U<n>` → spike reports; probes → `scripts/probe_*.py`.

## R1 — does `above`/`left` track a position or a device across the era boundary?

**Ruling: neither. The label is not a stable camera geometry, and no per-view prior may cross the
era boundary.**

Measured, rigidity measurability by `(device_config, view)` — the within-configuration contrast is
what breaks the era confound that a raw view split leaves standing:

| device_config | view | n | unmeasurable | frac | valid_fraction med | quiet-border energy |
| ------------- | ---- | - | ------------ | ---- | ------------------ | ------------------- |
| iPad (5th generation) / 16.6 | above | 89 | 76 | **0.85** | **0.212** | **14.52** |
| iPad (5th generation) / 16.6 | left | 36 | 2 | 0.06 | 1.000 | 10.43 |
| iPad (5th generation) / 16.7 | right | 131 | 2 | 0.02 | 1.000 | 9.54 |
| iPad Air 11-inch (M2) / 18.1.1 | left | 57 | 1 | 0.02 | 1.000 | 7.09 |
| iPad Air 11-inch (M2) / 26.5 | above | 66 | 2 | 0.03 | 1.000 | 6.47 |

Two contrasts, and they point in different directions on purpose:

- **Within one configuration**, view matters: iPad(5)/16.6 `above` 85% unmeasurable against the same
  tablet's `left` at 6%. The label carries geometric content that device identity does not explain.
- **Within one label**, configuration matters: `above` is 85% unmeasurable on iPad(5)/16.6 and 3% on
  Air-M2/26.5. So `above` does not name one geometry — it named two different setups in two eras.

`decode_status` is `ok` on 89/89 of the pathological cell and its sampled-frame count is normal
(median 30, against 33 for the healthy `above` cell), so this is scene and camera behaviour, not a
decode or sampling failure. That cell also carries the corpus's highest quiet-border motion energy,
and it is where 27 of the visual spike's 28 independent scalar flags concentrate (visual:U6). Read
together: **iPad(5)/16.6 `above` is an unstable camera** — 89 assets, 23% of the corpus.

Binding consequences: a per-view geometric prior, a per-view rotation constant, and any reuse of one
view's calibration across the era boundary are all incorrect. `(device_config, view)` is the coarsest
key that names a stable geometry, and even that is unverified for `left` versus `right` handedness,
which nothing in this corpus resolves.

## R2 — is background rigidity sufficient for per-event extrinsics, and on what fraction?

**Ruling: yes on the measurable population, and P21's gate is replaced. The 2 px / 4 px absolute
gate was unadjudicable, because the accept threshold and the instrument were the same constant.**

P21 as frozen accepted `drift_median ≤ 2 px` and `drift_p95 ≤ 4 px`, and flagged **210 of 286**
eligible assets. That verdict was an artefact of two compounding defects:

1. **The gate sat inside its own instrument's noise band.** `residual_p95` ran 2.10–3.64 px across
   all 286 eligible assets (median 3.14), so `drift_p95 > 2 px` held on 278/286 before any real
   camera motion was required. 175 of the 210 failures failed on p95 alone with `drift_median ≤ 2`.
2. **`DRIFT_P95_GATE_PX` doubled as the MAGSAC inlier threshold** (`geometry_qualification.py:326`,
   `threshold = DRIFT_P95_GATE_PX / mean_scale`). No residual above the gate could ever be reported,
   so the statistic the gate was judged against was pinned by the gate.

A threshold sweep over 20 eligible assets separates the robust statistics from the artefact
(`scripts/probe_ransac_threshold.py`):

| RANSAC thr (px) | residual_p95 med | drift_p95 med | drift_median med | inliers med |
| --------------- | ---------------- | ------------- | ---------------- | ----------- |
| 4 | 3.118 | 5.209 | 1.354 | 257.8 |
| 6 | 3.856 | 5.382 | 1.380 | 264.8 |
| 8 | 4.793 | 5.529 | 1.396 | 268.5 |
| 12 | 5.733 | 5.924 | 1.391 | 271.0 |
| 20 | 6.358 | 6.359 | 1.428 | 272.0 |
| 32 | 7.670 | 6.602 | 1.422 | 273.0 |

`residual_p95` **never plateaus** — it tracks the threshold monotonically over an 8× range while the
inlier count grows only 6%, so the same correspondence set is being re-scored and the p95 reads where
the cut falls, not how good the matches are. **`residual_p95` is not a measurement of the scene at
any threshold setting, and no gate may be built on it** — including a drift-to-residual ratio, which
was considered and is rejected for exactly this reason. `drift_median` moves 5.0% and `drift_p95`
26.7% across that same 8× range: **drift is the robust statistic.**

**Replacement gate.** `rigidity_flag = camera_motion` when `drift_p95_px > 20`, else `rigid`. Drift
is already expressed in native pixels (`_match_sample` scales by `scale_x`/`scale_y`), so it compares
directly against the **20 px reprojection tolerance the 3D pipeline already applies**
(`src/pose_estimation/triangulation.py:423-424`). The gate is anchored to a downstream tolerance that
exists independently of this measurement, which is what P21's constants lacked.

| gate | assets pass / 286 eligible | multi-asset families with every member rigid |
| ---- | -------------------------- | -------------------------------------------- |
| P21 as frozen (med ≤ 2 and p95 ≤ 4) | 76 (26.6%) | 3 / 137 |
| **drift_p95 ≤ 20 (ruled)** | **278 (97.2%)** | **71 / 137** |

104 of 137 multi-asset families keep at least two rigid cameras. The reading is direction-safe: taking
drift entirely at face value as real camera motion is the pessimistic case, and 97.2% still sit inside
the tolerance; if any part of drift is instrument noise the true motion is smaller and the conclusion
only strengthens.

**Shipped implementation changes.** `RANSAC_THRESHOLD_PX = 8.0` becomes its own constant, decoupled
from every accept gate — inliers have saturated by 8 (268.5, within 1.3% of the 32 px value) while
drift is not yet inflated. `residual_p95` leaves the published schema; the RANSAC threshold is
recorded in measurement provenance instead, because publishing it as a "residual" is what invited the
misreading in the first place.

**Coverage limit, stated as a fact about the work:** 83 assets are support-unmeasurable and 10 are
orientation-excluded, so **93 of 379 assets carry no rigidity verdict under any gate**. 80 of the 83
fail `MIN_VALID_FRACTION = 0.80`, and 76 of those sit in the single unstable cell R1 names. Raising
that coverage raises M2.6's yield; it does not change M2.6's existence → `.agent/polish.md`.

## R3 — does any metric scale reference exist in frame?

**Ruling: closed negative. This corpus carries no metric reference, and the standing fallback is
therefore permanent for it rather than provisional.** The survey ran and found no route.

Three findings carry the ruling. Apparatus class does not supply length: the battery is GRASSP v1
Prehension Performance, which publishes no dimension or tolerance for any of its six object classes,
and the plausible spread within a class runs from 6% (key blanks) to 82% (hex nut sizes) to 2.9×
(twist closures). Identity is unresolvable in the frames themselves: a stratified 52/379 asset sample
covering all 18 task×view cells decoded 364/364 frames, confirmed the task object present in every
sampled asset, and found at least one visually sharp crop in 50/52 — yet two adversarial metrology
prompts requiring a visible make, denomination, thread size or marking accepted **0/52**. Visibility
is not the blocker; dimensional identity is. Every fallback is absent rather than imprecise:
`inventory/assets.csv` carries no anthropometric column and 193 `session.json` manifests expose zero
anthropometric keys, no furniture survey or height setting exists, the cameras were hand-held with no
surveyed baseline, and 0/379 canonical paths contain a calibration-target token against a 379/379
positive control. Audio time-of-flight cannot supply it either — triangle closure cancels propagation
exactly, which is the same cocycle property that makes closure a consistency statistic and not an
accuracy one.

Even the best conditional route fails the requirement. An individual coin is the only object whose
resolved identity would carry a statutory diameter, but no crop resolved a denomination, and its
measured extent (p10/median/p90 = 38.5/48.7/86.0 px) puts an optimistic root-sum-square floor at
**±17.7%** for a reconstructed-scene scale and **±22.2%** per image, before lens distortion, which
has no corpus calibration at all. No finite end-to-end interval can be stated.

Pipeline behaviour is unchanged, because the fallback was specified before the survey ran:
`scale_ref_class` publishes empty with a `scale_unmeasured` flag, and every artifact states arbitrary
scale explicitly. Angles, angular velocities, timing, normalized trajectory shape and dimensionless
ratios survive intact; every metre-valued distance, velocity and jerk does not (P24, P26). M2.6 does
not rescue this: recovering extrinsics removes the depth-plane penalty and would let one metric
object be triangulated for a single global factor per event, but it supplies no physical length, and
scale stays per-event because no fixed rig or reusable extrinsic exists.

Reopening requires new provenance from outside this corpus — caliper measurement of each retained
apparatus instance, a filmed ruler or board, or a cardinality-checked participant-anthropometry
mapping with contemporaneous side-specific segment lengths. All three are acquisition actions, not
analysis, so none is schedulable as a unit here.

## R4 — does M2.6 exist?

**Ruling: M2.6 exists, and its route is re-specified. Scene-feature extrinsics is eliminated by
measurement. Subject-keypoint calibration is the route.**

The contract's condition was "feasible extrinsics on a usable fraction of events." Rigidity (R2) is
necessary but not sufficient — it says a camera held still, not that two cameras share enough scene
to calibrate against. That second question was never measured; `prod-m2u3-geom` delivered U1 and left
U2, U3 and U4 at `unknown`. It is now measured (`scripts/probe_crossview_pose.py`, all 246
within-family pairs, 161 s):

| statistic | min | p25 | med | p75 | p95 | max |
| --------- | --- | --- | --- | --- | --- | --- |
| cross-view mutual SIFT matches | 8 | 11 | **13.5** | 17 | 22 | 732.5 |
| F-inliers | 7 | 8 | **8.0** | 9 | 10 | 724.5 |

**0 of 246 pairs reach a recoverable pose**; 242 read `weak_correspondence`. Only 2 of 244 pairs
exceed 30 mutual matches. The F-inlier median of exactly 8.0 is the algebraic minimum for a
fundamental matrix — MAGSAC is fitting minimal samples, so those "inliers" carry no evidence.

The null is geometric, not procedural, and two independent controls establish that:

- **Baseline ladder** (`scripts/control_crossview.py`): same frame 2812 mutual matches → same asset,
  adjacent samples 1252 → same asset, far samples 962 → **cross-view 12–19**. A 65–100× collapse with
  1740–3355 SIFT keypoints present in every view, so the scene has texture and the matcher works.
- **Unplanned in-corpus control**: the only 2 pairs of 244 that match richly are the `above|above`
  (298.5) and `left|left` (732.5) pairs — the two `view_conflict` families, where both assets claim
  the *same* view. The probe recovers correspondence exactly where two cameras share a viewpoint and
  nowhere else.

By view pair: `above|left` 12.5 (n=73), `above|right` 12.0 (n=99), `left|right` 19.0 (n=72). No
pairing is viable, and no learned-matcher substitution was assumed — the elimination is of classical
scene-feature correspondence at these baselines, which is what was measured.

**The route that survives** is the one markerless mocap already uses when no checkerboard exists: the
subject is the calibration object. Correspondence is *assigned*, not matched — keypoint `k` in view A
is keypoint `k` in view B — so the wide-baseline collapse that kills SIFT does not apply. The corpus
supports it: `detect_rate` median **1.0** (mean 0.989886, min 0.333333, n=379, 0 inference-failure
frames), 133 keypoints per detection, and audio sync agreeing with an independent visual estimator to
median 12.89 ms, well inside one 33.4 ms frame. Route selection detail and its published accuracy
basis → `.scratch/agents/res-m2u3-3.md`.

**M2.6's shape changes accordingly**: it recovers extrinsics by bundle adjustment over
time-synchronized 2D keypoints under the per-model intrinsics prior (P25, every value labelled
`prior`), not by scene-feature SfM. It binds to the M2.2 instance grain, never to `capture_id`.
M2.6 is gated on M2.5 delivering sub-frame offsets, because the keypoint route consumes them.

**Standing risk to carry into M2.6**: subject-only calibration degrades when the subject's keypoints
are near-coplanar or the subject barely moves. Both are live here — the tasks are seated upper-limb
movements. M2.6 must measure pose variety per event before it claims an extrinsic.

## R5 — what replaces the integer `sync_offset`?

**Ruling: one float `offset_s` per camera, expressed in seconds against the event's reference
camera. No rate or drift term.**

Measured: 0/132 qualified audio drifts move alignment by more than one frame over the pair's overlap
(audio:U5); visual agrees, 0/15 at the 95% lower bound (visual:U5); an independent 16–31 ppm prior
bounds accumulation at 4.5–8.5 ms over the corpus's 274.8 s maximum clip (res2:Q2). A constant offset
is therefore sufficient **for this corpus**, and the artifact says so rather than asserting it
generally. The integer `sync_offset` cannot express a sub-frame value at all, and the corpus's
cameras do not share a frame rate (29.963–29.987 Hz, 7 of 137 families agreeing to 3 dp), so the
integer form is not merely imprecise — it is unrepresentable for this data.

## A1 — how do the expensive axes reach `assets_qc.csv`?

**Ruling: a validated sidecar generation, ingested by `qualify.py` as a third upstream.**

The axes split cleanly by cost and by determinism, and that split is the whole argument:

| axis | cost | deterministic? | placement |
| ---- | ---- | -------------- | --------- |
| timebase, orientation | 33 s, metadata only | yes | inline in `qualify.py` |
| rigidity | ~30 min | yes, CPU-only | sidecar |
| detectability | ~33 min | **no** — device-dependent | sidecar |
| visual sync corroborator | ~20 min decode cache | yes | sidecar |
| audio sync | ~8 s after cache | yes | sidecar (pair grain) |
| scale | unrun | unknown | sidecar |

P08 requires the published set to be a function of corpus bytes alone, and detectability cannot
satisfy it: GPU↔CPU box deltas run 0.18–3.21 px (median 0.58) on the 32-frame parity set. Inlining it
would force P08 to be weakened for every axis at once. A sidecar quarantines the non-determinism
behind its own generation, and `qualify.py` keeps its ~33 s run and P08 intact by validating that
generation's digest rather than reproducing its bytes.

**The sidecar is a measurement record, not a publication.** It carries a table, a `generation` block
digesting that table and itself, the upstream `inventory` digest, and provenance (devices, thresholds,
analysis resolution, sampling rule). It does **not** inherit the full M2.1/M2.2 crash-safety contract:
a torn sidecar is caught by digest mismatch and repaired by rerunning it, which is the whole
difference between a regenerable measurement and a published artifact.

Pair-grain sync results live in the sidecar too, because the expense is the decode; `qualify.py`
retains the cheap cross-asset reasoning — closure, connectivity, event qualification — since those
are graph operations over already-measured offsets.

**Deferred, with its acceptance check written now**: `inventory.py`, `sessions.py` and `qualify.py`
each carry their own copy of the publication contract. Extracting it is correct and is not this
unit's work → `.agent/polish.md`.

## R6 — P17b's accept policy, priced on measurement

**Question.** P17b as frozen reads "qualifies a pair on **agreement** between the two estimators,
never on either alone". Audio accepts 210/246 within-family pairs, the visual corroborator 74, both
65. What does each reading cost?

**Evidence.** `scripts/probe_sync_policy.py` over the published sidecar (`measurements/`, sync axis,
committed estimators). Every figure reruns from committed state.

| policy | pairs | families view-recoverable (of 137) | triangles closed | closure median | closure max |
| ------ | ----- | --------------------------------- | ---------------- | -------------- | ----------- |
| audio alone | 210 | 122 | 35 | 4.451 ms | 30.286 ms |
| **audio + corroborator veto** | **201** | **117** | **31** | **4.994 ms** | **30.286 ms** |
| strict agreement | 56 | 26 | 2 | 11.647 ms | 11.881 ms |
| visual alone | 74 | 34 | 9 | 8.080 ms | 34.121 ms |

**View-recoverable is P38's statistic and the only one quoted here**: a family counts when *some*
choice of one camera per view is spanned by cross-view accepted pairs. That is what M2.6 consumes —
a family holding two files of one view needs only one of them, and a same-view edge carries no
cross-view geometry. The probe also reports `families_all_assets_connected`, a strictly harder
question (every asset joined, same-view edges counted) that reads 121 and 116 on the top two rows.
The two must never be quoted as one number. Under the P38 rule the port returns **122/137** on audio
alone with closure median/max **4.451/30.286 ms**, reproducing the spike digit-for-digit, so P38 is
verified in full rather than on its acceptance count alone.

Cross-modality agreement on the 65 both accept: median **12.886 ms**, p75 23.10, p95 50.72, max
74.819; 56/65 within one 33.4 ms frame, 27/65 under 10 ms. Reproduces P17 exactly.

**Ruling.** The strict reading is refused. It leaves **111 of 137 families unrecoverable**, 96 more
than audio alone, and leaves 2 closing
triangles, so it would end M2.5 and M2.6 for most of the corpus, and it buys nothing the veto does
not: the gross-error evidence behind P17b bounds the **visual** estimator alone — of the 9 pairs it
accepts and audio rejects, one disagrees by **87.421 s** despite its 0/200 held-out control rate.
No comparable evidence exists against audio: 2/100 control false positives, 35/35 accepted triangles
closing under one frame, and 12.9 ms median agreement wherever the second instrument speaks.

**Audio estimates; the corroborator holds a veto where it spoke and no vote where it did not.**

| both statuses | n | verdict | published `status` |
| ------------- | - | ------- | ------------------ |
| audio ok, visual ok, agree ≤ 1 frame | 56 | qualified | `ok_corroborated` |
| audio ok, visual ok, disagree > 1 frame | 9 | **refused** | `contradicted` |
| audio ok, visual not ok | 145 | qualified, flagged | `ok_uncorroborated` |
| audio not ok, visual ok | 9 | **refused** | `visual_only` |
| neither ok | 27 | refused | `neither_accepted` |

The alphabet is closed at **five** tokens and the strata partition all 246 pairs exactly
(56+9+145+9+27), qualifying 201. The fifth token is not optional: publishing the audio abstention
token verbatim for the both-refused stratum would leave the output alphabet open to every estimator
status, which is what C19 refuses. `neither_accepted` says both instruments measured the pair and
neither accepted it — distinct from `contradicted`, where both accepted and disagreed.

A pair both instruments accept and contradict is two independent measurements disagreeing, and
neither is preferred, so it is refused rather than resolved — 9 pairs, costing 5 families. The
`visual_only` stratum is exactly where the 87 s gross error lives and is never qualified.

**Consequence.** M2.5 inherits 201 qualified pairs and 117/137 view-recoverable families — the veto
costs 5 families against audio alone. `sync_qualified`
is true for an event only when accepted pairs connect its cameras (P19), and the closure statistic
stays labelled self-consistency, never accuracy (P16).

## R7 — the port's signal field, found by reproduction

The visual estimator ported at its library default (`motion`, the whole frame) accepted **43/246**
against the spike's 74. The spike passed `center_motion`. A hand-held camera writes its own movement
into the frame border, so the whole-frame trace mixes camera motion with subject motion and the two
views stop sharing a signal. The field is the estimator, not a knob: it is now a module constant
recorded in the sidecar's provenance (`visual_offset.SIGNAL_FIELD`), and the port reproduces
210/74/65 exactly.

**Standing rule this earns:** a ported estimator reproduces its source's acceptance count before any
number it produces is quoted. A default argument is a silent parameter.

## R8 — sidecar ingestion contract (batch ruling over `test-m2u3-measure` PREP-1)

138 enumerated rows (A01-A36, S01-S28, K01-K20, C01-C19, V01-V14, D01-D21). Seven governing
rulings decide most of them; the rest are ruled individually below. Every row is adjudicated.

### Governing rulings

**G1 — the ingestion path validates exactly what the write path validates.** `load_axis` is not a
trusted reader. `write_axis` checks cells, keys and pair order; `load_axis` checked only the header,
so a coherently hand-edited or third-party table reached `qualify` with no cell, key, duplicate or
count check. The sidecar's whole premise is that it is *independently produced* and re-validated at
use, which makes the read path the one that must be strict. Ingestion re-runs the cell alphabets,
the key rules, the declared row count and canonical row order.
→ decides A11 A12 A13 A16 A24 A25, C01-C18, K05 K08 K09 K10 K18, S14 S21 S22.

**G2 — validation and reading share one byte buffer.** `validate` returns the bytes it digested and
`load_axis` parses that buffer, never reopening the file. A digest proves nothing about bytes read
through a second `open`. → A23 S25.

**G3 — the manifest is the trust root, so its own read is hardened** to the standard already applied
to tables: regular non-symlink file, and duplicate JSON keys rejected rather than resolved
last-key-wins. A document that makes two claims must not validate on one of them. → A07 A10 S11 S16.

**G4 — an axis entry asserts "this axis was produced".** Present with zero rows = produced and
empty; every canonical key publishes unmeasured and the axis counts as measured. Absent = not
produced; the `*_unmeasured` flag stays. Producers must therefore write the manifest entry only
after the axis completes. → A03 K03 K04 S03.

**G5 — digests prove internal consistency, never authorship or cache freshness.** A coherently
recomputed sidecar is accepted; that is a property of the design, not a hole in it. A stale cached
result inside a freshly digested table is unreachable by any digest and is answered by cache-key
provenance (V12), never by ingestion. → S06 S08 S26 S27 S28.

**G6 — P35 binds ingestion, P36 binds the generator.** An enumerated pair the sidecar omits ingests
as unmeasured (P35) and fails a generator conformance test (P36). Ingestion never hard-errors on
omission, and the sync generator is separately required to emit all 246. → A14 K14.

**G7 — fusion lives in `qualify`, never in the record.** `status_audio`/`status_visual` stay
estimator-local tokens. The measurement bytes must be byte-identical when only fusion policy
changes, which is what makes R6 re-rulable without re-decoding. → A35 V14 D21.

### Status alphabets and the required-cell matrix

Enums are ruled into code beside the branches that produce them, never transcribed into prose that
can drift. `status_audio` = the **Peak** statuses only: `ok`, `short_audio`, `silent`,
`no_feasible_lag`, `no_background`, `boundary_peak`, `low_confidence`. `short_overlap`,
`insufficient_windows`, `degenerate_regression` and `global_abstention` are **Drift** statuses and
reach no published column. `status_visual` = `ok`, `insufficient_overlap`, `edge_peak`,
`undefined_confidence`, `low_peak_correlation`, `low_prominence`, `ambiguous_peak`, `signal_absent`.

Measured on the shipped table (246 rows): `status_audio` = ok 210 / low_confidence 36;
`status_visual` = low_peak_correlation 152 / ok 74 / low_prominence 15 / edge_peak 5. Every row
populates offset, confidence and peak for **both** estimators regardless of status; only
`drift_ppm`/`drift_se` are empty, on 114 of 246 rows.

That measurement corrects the proposed C10. **A refused row still publishes its statistics** — the
gate rejects the estimate, it does not erase it, which is P39's separation seen from the data side.
Emptiness means "no peak was computed", not "the peak was rejected". Ruled matrix:

- `status == "ok"` ⇒ that estimator's offset, confidence and peak cells are populated. Empty = HARD.
  Provable from the code: `ok` is reachable only past the finiteness checks.
- `status != "ok"` ⇒ those cells may be empty; a populated cell still matches its alphabet.
- Both status columns are non-empty on every row (P36). Empty = HARD.
- `drift_ppm`/`drift_se` carry no status of their own and are legally empty. Their abstention reason
  is unpublished; nothing downstream consumes it (M2.5 takes no rate term, R5), so P36's frozen
  column list stands and the gap is recorded rather than fixed.

### Individually ruled rows

- **A01** axis names closed to sync/rigidity/detect/scale. **A02** `generation` and axis-entry keys
  closed; the **provenance payload stays open** — it is evidence, not schema, it is covered by the
  manifest digest, and a closed provenance schema would version-lock every instrument change.
- **A04** table basenames are fixed per axis in `measure.AXES`, never manifest-selected. This makes
  **S19** unreachable by construction: no traversal, no collision, no aliasing input exists.
- **A05** the self-digest is over the **canonically re-rendered** manifest minus `generation`, so
  whitespace and key order survive and only semantics move it. **A22** qualification records that
  self-digest, not a file SHA.
- **A06** the sidecar directory is closed: any entry that is not the manifest or a named table is a
  hard error, so writer debris cannot sit undigested beside a valid record. → **S18** HARD.
- **A09** concurrent axis writers are **unsupported and declared so**, not locked. The sidecar is
  produced by scheduled runs; a merge protocol's failure modes exceed the hazard it removes.
  → **D14** pins the refusal rather than a merge.
- **A20** each axis entry's `generator_version` is checked at ingestion; the supported set is exactly
  `{"v1"}` today. Mixed-age sidecars stay expressible without inventing migration machinery now.
- **A21** `--measurements` adds its upstream key to `qualification.json`'s `generation` block **only
  when the flag is given**. Flagless output stays byte-identical (P34, P08); schema closure is
  evaluated per mode. An always-present nullable key would change bytes for every existing consumer.
- **A27** the non-sync schemas are already pinned in `measure.AXES` — `rigidity_assets.csv`,
  `detect_assets.csv`, `scale_assets.csv` with the column tuples in `measure/__init__.py`. They are
  contract, not module-private.
- **A28** a `--measurements` directory that is missing, unreadable or manifest-less is a hard error.
  The flag asserts the upstream; degrading to all-unmeasured would turn an operator typo into a
  silent publication. **A29** every failure surfaces as `QualifyError`, exit 2; `MeasureError` is
  wrapped at the `qualify` boundary so callers face one error domain.
- **A30, K08-K13** malformed keys are refused, never normalized: a reversed pair, a self-pair, a
  cross-family pair, a wrong `capture_id`, or an id absent from the canonical registry is HARD.
  Normalizing would publish malformed provenance and can silently collapse duplicates. **K20** ids
  compare as exact code-point strings; no locale or Unicode normalization.
- **A31-A34, C19** R6's fusion tokens: `ok_corroborated`, `ok_uncorroborated`, `contradicted`,
  `visual_only`. "Spoke" = `status_visual == "ok"`, i.e. the corroborator cleared **its own** gate —
  a low-quality visual estimate holds no veto. "Veto" = both ok and disagreeing by more than one
  frame. "No vote" = any non-`ok` visual status, whatever diagnostics its cells carry. `visual_only`
  = visual ok while audio is **not** ok, covering both audio rejection and audio abstention. The
  alphabet closes at **five** tokens: `neither_accepted` carries the 27 pairs neither instrument
  accepted, which the four-token reading left unmapped. **A36** the mapping must reproduce 201/246
  qualified pairs and 117/137 view-recoverable families.
- **K01-K02, K06-K07, K15-K17** stand as the teammate proposed. **K19** registry validation precedes
  sidecar key checks; the sidecar never adjudicates a malformed registry.
- **S01, S02, S04, S05, S07, S09-S13, S15, S17, S20, S23, S24** stand as proposed: each is HARD on
  the named predicate. **S03** accepted per G4.
- **V01-V13** are accepted as the standard P31 already states. Auditing the shipped provenance block
  against this list is a `gate` deliverable, not a claim that it currently passes. **D01-D20** are
  accepted as the determinism battery and owned by the `gate` track; **D16** and **D17** are P39's
  mechanical test — sweeping a gate must move only statuses, sweeping an instrument must move
  provenance and miss the cache — and rank first.

### Consequences

`load_axis` gains cell, key, count and order validation; `validate` returns the bytes it digested;
the manifest read rejects symlinks and duplicate JSON keys; the status enums become code constants.

## R9 — `conf_audio` is gate-relative, not an instrument reading (P39 breach, CONFIRMED)

**Finding, from `test-m2u3-measure`'s independent D16 probe, verified in source.**
`audio_offset.py:286` computes `confidence = min(peak_rms / min_peak_rms, peak_ratio /
min_peak_ratio)` and `:291` gates on `confidence < 1.0`. `MIN_PEAK_RMS = 8.0` and
`MIN_PEAK_RATIO = 2.1` are accept thresholds, and they divide into a published statistic. Sweeping
them alone over 0.1→1e9 on one fixed shifted-noise signal holds lag, peak_rms, peak_ratio and
overlap identical while `conf_audio` moves 970.338606 → 9.703e-08.

**This is R2's defect, repeated.** P21 was unadjudicable because `DRIFT_P95_GATE_PX` was also the
MAGSAC inlier threshold, so no residual above the gate could be reported. Here the coupling runs the
other way — the gate does not pin the statistic, it *rescales* it — but the consequence P39 names is
the same: re-ruling either threshold silently rewrites every published confidence while the signal
is unchanged, so a confidence is comparable only within one threshold setting. The spike's recorded
ROC AUC 0.96083 and its 2/100 held-out control rate are both figures on the normalized quantity and
carry that dependency with them.

The coupling was deliberate and commented (`audio_offset.py:58`), which does not exempt it: P39 is
frozen and says a statistic gated by a constant that also shapes it is refused at review. **D16, the
probe I ranked first in R8, fails on shipped code.**

**Ruled fix — publish the raw statistics and let the gate be a comparison.** `conf_audio` leaves the
schema and `peak_rms_audio` replaces it, carrying the raw statistic; `peak_ratio_audio` already
carries its raw partner. The status becomes `peak_rms >= MIN_PEAK_RMS and peak_ratio >=
MIN_PEAK_RATIO`. `LOCAL_MIN_PEAK_RMS`/`LOCAL_MIN_PEAK_RATIO` on the drift path (`:344`) carry the
identical defect and are fixed the same way.

**The refactor is status-preserving by construction**: `min(a/r, b/t) >= 1` holds exactly when
`a >= r and b >= t`, so no acceptance changes. R7's bar therefore still applies and must be met —
the re-measured corpus must return **210/246 accepted, 122/137 view-recoverable, closure median/max
4.451/30.286 ms**. A moved count means the refactor changed the estimator and is a finding.

**Consequences.** P36's frozen column list is amended (`conf_audio` → `peak_rms_audio`), the corpus
is re-measured, and R6's numbers are re-derived from the new table. Confidence figures quoted
anywhere as absolute — the ROC AUC above included — are threshold-relative until re-derived. This is
**spine work**, not polish: a frozen contract predicate is red against shipped code.

## R10 — where P35's key rules live, and three R8 gaps closed

`test-m2u3-measure`'s independent suite against `d6d7516`: 63 pass, 16 red. Adjudication.

**K01/K11/K12 are correctly red and are not defects in shipped code — they were aimed at the wrong
entry point.** `load_axis` takes a `Sidecar` and no registry, so it cannot know which family an asset
belongs to, whether an id is canonical, or whether a `capture_id` matches its pair. Those are
registry-relative facts and no schema reader can decide them. `validate(inventory_dir=…)` proves the
sidecar was measured against the registry still on disk; it never loads the asset rows.

Ruled: `load_axis` **stays registry-blind** — a pure schema reader is what makes it testable without
a corpus — and P35's key rules move to a named reconciliation step that `qualify` calls with the
canonical key set it already enumerates. K01-K20 target that function, not `load_axis`. It is
unimplemented, so those reds stay red until the ingestion unit lands, which is their purpose.

**C12's semantic ranges are intended, and R8's G1 under-specified them.** G1 says ingestion re-runs
"the cell alphabets", and an alphabet is not a domain: `-3.000000000` matches `DECIMAL_CELL`, and a
visual correlation of `5.000000000` is not a correlation. Ruled in as spine, checked after parse —
`conf_*`, `peak_ratio_audio`, `drift_se`, `overlap_s`, `dur_a`, `dur_b` nonnegative;
`peak_corr_visual` within [-1, 1]. Nine reds, all correct.

**Top-level manifest keys close to exactly `{axes, generation}`.** A02's argument — an added or
renamed key means a different writer — applies at the top level and R8 applied it only one level
down. The self-digest stops a key being added after the fact; it does not stop a producer emitting
one. Red is correct.

**A09's declaration is a docstring obligation**, satisfied by stating one-writer-per-directory in
`measure/__init__.py`. Lowest priority of the set.

**Absent `rigidity`/`detect`/`scale` modules are expected, not defects.** Those axes are unbuilt;
their schemas are pinned in `measure.AXES` so a suite can target them ahead of the code.

Spine additions from this ruling, ordered after R9: semantic ranges, top-level key closure, the
reconciliation function carrying K01-K20.

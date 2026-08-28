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

**Ruling: open. The scale axis has not run.** No survey has been performed, so no ruling is
available, and the roadmap requires the survey to precede any request for participant
anthropometrics. Until it runs, `scale_ref_class` publishes empty with a `scale_unmeasured` flag, and
every artifact states arbitrary scale explicitly: angles and dimensionless ratios survive, every
metre-valued distance, velocity and jerk does not (P24, P26).

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

| policy | pairs | families connected (of 137) | triangles closed | closure median | closure max |
| ------ | ----- | --------------------------- | ---------------- | -------------- | ----------- |
| audio alone | 210 | 121 | 35 | 4.451 ms | 30.286 ms |
| **audio + corroborator veto** | **201** | **116** | **31** | **4.994 ms** | **30.286 ms** |
| strict agreement | 56 | 26 | 2 | 11.647 ms | 11.881 ms |
| visual alone | 74 | 34 | 9 | 8.080 ms | 34.121 ms |

Cross-modality agreement on the 65 both accept: median **12.886 ms**, p75 23.10, p95 50.72, max
74.819; 56/65 within one 33.4 ms frame, 27/65 under 10 ms. Reproduces P17 exactly.

**Ruling.** The strict reading is refused. It discards **95 of 137 families** and leaves 2 closing
triangles, so it would end M2.5 and M2.6 for most of the corpus, and it buys nothing the veto does
not: the gross-error evidence behind P17b bounds the **visual** estimator alone — of the 9 pairs it
accepts and audio rejects, one disagrees by **87.421 s** despite its 0/200 held-out control rate.
No comparable evidence exists against audio: 2/100 control false positives, 35/35 accepted triangles
closing under one frame, and 12.9 ms median agreement wherever the second instrument speaks.

**Audio estimates; the corroborator holds a veto where it spoke and no vote where it did not.**

| both statuses | verdict | published `status` |
| ------------- | ------- | ------------------ |
| audio ok, visual ok, agree ≤ 1 frame | qualified | `ok_corroborated` |
| audio ok, visual ok, disagree > 1 frame | **refused** | `contradicted` |
| audio ok, visual not ok | qualified, flagged | `ok_uncorroborated` |
| audio not ok, visual ok | **refused** | `visual_only` |
| neither ok | refused | the audio abstention token |

A pair both instruments accept and contradict is two independent measurements disagreeing, and
neither is preferred, so it is refused rather than resolved — 9 pairs, costing 5 families. The
`visual_only` stratum is exactly where the 87 s gross error lives and is never qualified.

**Consequence.** M2.5 inherits 201 qualified pairs and 116/137 connected families. `sync_qualified`
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

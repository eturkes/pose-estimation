# The calibration finding

Three-camera 3D reconstruction is not recoverable from this video corpus. Two committed probes
measured the cause and closed the route. This document states what those probes decide, shows the
evidence behind each statement, and fixes the wording you may use when you cite it.

Read this document before you write about the 3D result in a paper, a slide, or a report. The
project publishes the same statements as machine-readable cells; see
[Calibration ruling](technical/calibration_qc.md) for the tool, the schemas, and the consumer
contract. This document carries the evidence and the wording alone.

## 1. The ruling

The ruling is corpus-grained. It has one cause and one named unrun arm.

| cell | value |
| ---- | ----- |
| grain | corpus |
| recovery status | unachievable |
| cause | `cross_view_keypoint_bias` |
| transfer status | absent |
| keypoint source | RTMW-L |
| image height | 1080 px |
| intrinsics basis | per-model prior |
| unrun arm | `per_event_double_centered_bias_and_pose` |

Three qualifiers travel with the ruling, and they are load-bearing. The measurement covers RTMW-L
keypoints, this corpus at 1080p, and per-model intrinsic priors. Change any one of the three and the
result is unmeasured, not settled.

## 2. What the two probes are

| probe | question | population |
| ----- | -------- | ---------- |
| `scripts/probe_calibration_bias.py` | Why does extrinsic recovery fail? | 22-event sample, 32 frames per event |
| `scripts/probe_bias_transfer.py` | Is the failure a bias a model could remove? | all 115 eligible events, 178 camera pairs over 103 events |

Both probes replay a cached set of detected keypoints. Neither probe decodes video at analysis time.
Section 5 gives the commands that rebuild the caches and reproduce every number below.

Each probe also runs synthetic arms. A synthetic arm drives **known** geometry through the real
validity masks, image sizes, and device models of the corpus. Its value is the reference band that
makes a corpus number readable. It is never a corpus measurement.

## 3. The claim matrix

Each entry quotes the supported statement, gives the evidence that decides it, and names the
overreach it refuses. Quote the statement as written. Do not reword it, because the publisher checks
published bytes against the same text.

### C01 — extrinsic recovery

> Extrinsic recovery from RTMW-L keypoints on this corpus at 1080p under per-model intrinsic priors is measured unachievable.

**Evidence.** Three-camera rotation cycle closure, against a 10-degree bound declared before the
run. The corpus closes 2 of 10 three-camera events, at a median of 39.0 to 47.4 degrees. The
synthetic control recovers known extrinsics through the same masks at a median of 0.000 degrees, with
a maximum of 0.004 degrees. The control holds 10 of 10 events inside the bound out to 8 px of
correspondence error. It reaches 7 of 10 at 16 px and 1 of 10 at 32 px. The corpus therefore prices
its own correspondence error near 30 px at 1080p.

**Refused.** Do not extend this to another detector, another estimator, or any capture. The bound
covers one keypoint source on one corpus.

### C02 — the measured cause

> Within-event cross-view RTMW-L correspondence carries a measured 15-20 px systematic component at 1080p.

**Evidence.** One pooled pose per camera pair is fitted on a training block of frames. Per-keypoint
mean signed epipolar residuals are then correlated across two disjoint held-out blocks. Over 39
camera pairs the correlation reads a median of 0.703, and 26 of the 39 pairs exceed 0.5. The same
statistic on synthetic data separates the two mechanisms cleanly. Zero-mean noise returns 0.010,
0.007, and 0.120 at 2, 8, and 32 px. A fixed bias returns 0.993 to 0.998. Residual magnitude
reads a median of 20.8 px. Two decompositions put the systematic part at 15 to 20 px.

**Refused.** Do not read the 15-20 px figure as detector error inside one image. The number measures
disagreement between two views of one event.

### C03 — the estimator is exonerated

> The shipped estimator is exact on exact synthetic correspondence, and independent bundle adjustment worsens corpus closure.

**Evidence.** At zero correspondence error the shipped route returns a closure median of 0.000
degrees. Independent pairwise bundle adjustment then moves closure the wrong way. At 8 frames the
median rises from 37.17 to 40.53 degrees, and events inside the bound fall from 2 of 10 to 1 of 10.
At 32 frames the median rises from 39.00 to 78.89 degrees, with a median maximum pose move of 43.32
degrees. A better fit to biased correspondences travels further from the truth, and the damage grows
with data. That is the signature of a biased estimator, not of a weak one.

**Refused.** Do not report the solver as the cause, and do not treat a closer fit as a closer answer.

### C04 — no keypoint subset rescues it

> No disjointly selected RTMW-L subset beats all 65 keypoints on the measured corpus folds.

**Evidence.** Keypoints were ranked on one event fold and scored on the disjoint fold, in both
directions. On fold 1 the full 65 keypoints close at 47.42 degrees, the cleanest 40 at 41.87, and the
cleanest 24 at 108.96. On fold 2 the full 65 close at 21.04 degrees, the cleanest 40 at 64.35, and
the cleanest 24 at 49.72. The 40-keypoint subset wins one direction and loses the other by three
times, so no subset holds across the folds. The cleanest ten keypoints still carry 49.6 to 53.9 px of
mean absolute residual, so the bias is corpus-wide rather than concentrated.

**Refused.** Do not generalize this to a subset drawn from a different detector.

### C05 — the bias does not transfer

> Signed bias transfer is absent at the tested view-pair, device-model, task and subject groupings over the full eligible population.

**Evidence.** Measured on the full eligible population: all 115 eligible events yield 178 camera
pairs over 103 events. Per-keypoint mean signed residual vectors were correlated between **distinct**
events at four successively stricter groupings.

| grouping | pair comparisons | signed correlation |
| -------- | ---------------- | ------------------ |
| same view pair | 4341 | 0.0108 |
| + same device-model pair | 2738 | 0.0102 |
| + same task | 1071 | 0.0103 |
| + same subject | 275 | -0.0296 |
| keypoints permuted (null) | 4692 | 0.0051 |

Inside one event the same statistic reaches a median of 0.8138, with 129 of 178 pairs above 0.5. So
the bias is reproducible within an event and absent between events. Synthetic references separate
completely. Shared-bias arms hold a correlation of 0.180 or more, even under 1.2 m of rig jitter.
Non-shared arms span -0.011 to 0.030. The corpus spans -0.030 to 0.031 across the five groupings,
which places it inside the non-shared band.

**Refused.** Do not extend this to a bias model built on a grouping nobody tested here.

### C06 — shared difficulty is not a shared offset

> The same keypoints share difficulty across events while the signed offset direction is redrawn every event, so that magnitude is not a correctable coordinate offset.

**Evidence.** Correlating the residual **magnitude** instead of the signed residual returns 0.1499
pooled, against a permutation null of -0.0324 and a per-event-bias reference of -0.0033. It stays at
0.1462 within one subject and reaches 0.2191 on the left-right view pair. The same keypoints are hard
everywhere, at every grouping. A difficulty ranking carries no direction, so nothing can subtract it
from a coordinate.

**Refused.** Do not state that the detector behaves unpredictably from keypoint to keypoint. The
magnitude structure is real and repeatable; only its direction is not.

### C07 — what a reprojection number proves

> Held-out reprojection on the solve's own keypoint family is self-consistency.

**Evidence.** Published, not measured here. Pätzold and colleagues (GCPR 2022) report a
keypoint-recovered calibration beating the reference calibration on human reprojection, at 4.01
against 4.57 px. The same recovered calibration then loses to that reference on independent
AprilTags by 3.05 px, at 5.00 against 1.95 px. A statistic computed on the family that produced the
solve certifies consistency alone.

**Refused.** Do not quote a held-out reprojection figure as evidence of a correct calibration.

### C08 — the evidence class

> This evidence is internal geometric and QC evidence only.

**Evidence.** Scope statement. No clinical protocol ran, and no reference measurement system was
present at capture.

**Refused.** Do not carry any statement here into a clinical conclusion.

### C09 — pixels and degrees are not millimetres

> Every pixel and degree statistic here stays separate from absolute metric accuracy.

**Evidence.** Scope statement, resting on a measured absence. Image geometry has a gauge freedom, so
image-only routes recover shape up to one unknown factor. A stratified 52-asset survey resolved an
exact dimensional identity in 0 assets, and every fallback route was absent rather than imprecise.
The corpus therefore carries no scale reference.

**Refused.** Do not convert a pixel or degree figure here into millimetres or metres.

### C10 — no marker comparison

> No marker-based comparison was run.

**Evidence.** Scope statement. The corpus holds video alone.

**Refused.** Do not compare these figures against a marker-based system, in either direction.

### C11 — what a different detector leaves open

> A lower-bias keypoint source and a detector trained for multi-view consistency stay outside the measured bound.

**Evidence.** Bound statement. C02 measures the cause as a property of the keypoint source, so a
source with lower cross-view bias is untested rather than refused. The detector and the viewpoint
separation are what a future attempt must change. The solver and the sample size are not.

**Refused.** Do not present the result as a limit on detectors in general.

### C12 — the route that reopens 3D

> Prospective calibrated capture stays outside the measured bound and is the route that can reopen 3D.

**Evidence.** Bound statement. Every measurement here is retrospective, over video captured without
calibration, without synchronization hardware, and without a fixed rig.

**Refused.** Do not present the result as a limit on a future calibrated capture.

### C13 — the arm nobody ran

> The per-event double-centered bias-and-pose synthetic-control arm is unrun.

**Evidence.** The `per_event_double_centered_bias_and_pose` arm carries no result. A per-event bias
field holds 390 free parameters against 11 pose degrees of freedom, with no external anchor. The
parameter count therefore argues that the arm is degenerate. C05 and C07 also remove every acceptance
statistic that could credit such a solve on this corpus.

**Refused.** Do not give this arm any outcome, in either direction. It has none.

### C14 — the ruling's grain

> One corpus-level ruling holds while every per-event geometry cell stays unmeasured.

**Evidence.** The qualification tree reports the geometry axis as unmeasured on all 193 event rows,
and the scale axis as unmeasured on all 379 asset rows. The ruling adds one corpus row beside that
tree. It patches no cell.

**Refused.** Do not read a per-event verdict out of a corpus-level ruling.

### C15 — how to read a synthetic arm

> Each synthetic arm is instrument calibration whose meaning arises only in contrast with the corpus row.

**Evidence.** Every synthetic arm drives known geometry through the real masks, sizes, and device
models. Its number reports the instrument's response to a known input.

**Refused.** Do not quote a synthetic arm as a corpus measurement.

## 4. Name the population beside every count

Each number above quantifies over a stated population. Two populations appear, and they are not
interchangeable.

- The 22-event sample at 32 frames per event supports C01 to C04.
- All 115 eligible events, giving 178 camera pairs over 103 events, support C05 and C06.

Carry the population with the number whenever you quote one. A bare count is the error this project
has already made three times.

## 5. Reproduce the numbers

Both caches are large and are excluded from version control. Rebuild the one you need first, then run
the analysis probe. Collection decodes video, so run it on a machine that holds the corpus.

```bash
# 22-event sample at 32 frames -> C01 to C04
uv run python scripts/probe_calibration_observability.py \
  --cache .scratch/calib-obs-f32 --frames-per-event 32 collect
uv run python scripts/probe_calibration_bias.py --cache .scratch/calib-obs-f32 all

# all 115 eligible events at 32 frames -> C05 and C06
uv run python scripts/probe_calibration_observability.py \
  --cache .scratch/calib-obs-wide --frames-per-event 32 --stratum-events 25 collect
uv run python scripts/probe_bias_transfer.py --cache .scratch/calib-obs-wide
```

Give each population its own `--cache` directory. `--stratum-events` joins the cache fingerprint, so
a raised value re-collects the whole sample in place and you lose the narrow population.

`probe_bias_transfer.py` streams one JSON record per arm. The numbers are those records. Its final
line prints the sorted key list alone.

Set up the environment first; see [Environment](technical/environment.md).

## 6. What comes next

The 3D line is closed for this corpus. It reopens on prospective calibrated capture alone, which is a
separate project with its own ethics and clearance footprint. See
[3-camera capture and QA protocol](capture_protocol.md) for the capture design.

The two-dimensional analysis line is unaffected. Alignment, qualification, and the clinical feature
path all remain in use.

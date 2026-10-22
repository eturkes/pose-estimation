# 2D pose weak points — catalog + repair map

Population = the superseded rtmlib-f7 generation (rtmlib `PoseTracker`, `tracking=False`,
CPU detector every 7 frames, RTMW-m on NPU, `--single-subject`, `--tracking body`): 379 clips /
193 recording families. The M2.9 generation that replaced it → *M2.9 corpus, measured* below.
Two instruments:

- **Watch pass** — every clip through the review UI, overlay at keypoint threshold 0.3, 8 frames
  spread over each clip, 97 contact sheets. "Families noted" counts families where the class was
  seen at least once (approximate: one note line can name two families).
- **Triage** — per clip, person 0, image-isotropic coordinates (`.scratch/triage.py`, → `pose-instruments.md`):
  - `whole` = frames where >= 8 of the 10 feature keypoints jump > 0.10, per 1000 observed frames.
  - `iso` = frames where 1-2 of them jump > 0.10, per 1000.
  - `alt` = median |2nd difference| / mean |1st difference| over moving frames (0 smooth, 2 alternating).
  - `flick` = hand present↔absent transitions per 100 frames.
  - `oof` = share of feature-keypoint observations with visibility >= 0.3 outside the frame.
  - `gap` = decoded frames with no exported row.
  - `lowvis` = share of feature-keypoint observations with visibility < 0.3.

Views: above 155 clips, left 93, right 131.

## Classes

| id | symptom | mechanism | prevalence | repair |
| --- | --- | --- | --- | --- |
| C01 | overlay drawn early after the first frame with no row | player indexed the landmark series by playhead | every clip: first row >= frame 2 on 379/379 (exactly 2 on 352) | review UI `397cc8a` |
| C02 | skeleton jumps to a therapist, the person across the table or a passer-by | single-subject pick = argmax of mean score over 133 keypoints, 68 facial → a small fully visible bystander beats a large truncated patient | `whole > 10`: 144/379 (right 90/131, left 49/93, above 5/155); families noted 104/193 | M2.9.1 sticky largest-box subject |
| C03 | hand or arm untracked while visible, hands blinking | pose-derived crop between detector calls loses the limb; detector every 7th frame | `flick > 5`: 143/379 (above 117/155) | M2.9.1 detector every frame + box transport |
| C04 | limbs drawn along a therapist's arm or glove | two people inside one top-down crop | families noted 22/193 | M2.9.1 in part (crop = the subject's detector box); residual = model limit |
| C05 | hallucinated occluded parts: legs under the table, far arm + hand, vertical body lines from the frame edge (above), limbs stretched to frame corners, duplicate hand | a top-down model emits all 133 keypoints, many at moderate confidence | `iso > 10`: 106/379; families noted 107/193 | M2.9.2 drops the duplicate hand; M2.9.5 drops legs everywhere + hips under the overhead camera (user ruling); far arm/hand + stretched limbs remain a model limit |
| C06 | camera-setup footage tracked (floor, ceiling, operator) | no task segmentation | families noted 13/193 | deferred (`.agent/deferred.md`) |
| C07 | confident points outside the frame | extrapolation past the crop and frame edge at visibility >= 0.3 | `oof > 0.05`: 150/379 (above 106/155) | M2.9.2 hygiene |
| C08 | rows withheld at track birth + long gaps | smoother `min_track_age = 3` on every re-keyed track | first row >= frame 2 on 379/379; `gap > 0.05`: 28/379 | M2.9.1 identity-keyed smoothing |
| C09 | finger jitter and dropout | small hands inside a 256×192 whole-person crop | fingertip jitter + alternation (`.scratch/hand_metrics.py`) | M2.9.1 halves it; second-stage models evaluated, not adopted (below) |
| C10 | face landmarks on the wrong side | `_COCO_TO_BODY_FACE` reads the mirrored iBUG layout: face-derived eye points on the opposite side in 98-99 % of 11 315 frames, mouth 83-90 % | every clip | M2.9.2 mapping fix |
| C11 | arms-only subject untracked, background people tracked | same argmax as C02; the detector box on the arms is the largest (0.13-0.32 of the frame against 0.012-0.032, 3 probed clips) | families noted 19/193 | M2.9.1 sticky largest-box subject |

Shipped-corpus distributions over 379 clips: `whole` median 0 / mean 29.4 / p90 106.8; `iso` median
0 / mean 18.4 / p90 40.6; `alt` median 1.504; `flick` median 1.81 / mean 6.66 / p90 20.7; `oof`
median 0.025 / mean 0.077; `lowvis` median 0.072 / mean 0.144.

## M2.9.1 premise, measured

Watch set = 9 events / 22 clips / first 400 frames each, accelerator recipe. v0 = the shipped
chain; v1 = `SubjectTracker` + GPU f32 detector every frame + identity-keyed smoothing.

| metric (22 clips) | v0 | v1 |
| --- | --- | --- |
| `whole` mean | 46.1 | 0 |
| `iso` mean | 21.8 | 0 |
| `alt` median | 1.475 | 0.493 |
| `flick` mean | 8.95 | 0.17 |
| hand present, mean share | 0.94 | 0.997 |
| `gap` mean | 0.037 | 0.020 |
| `oof` mean | 0.024 | 0.026 |
| wall, 9 events | 733 s | 523 s |

`oof` and `lowvis` do not move: those classes belong to M2.9.2.

## M2.9.3 hand evaluation — not adopted

Same watch set. Finger metrics (`.scratch/hand_metrics.py`, median over 22 clips): `jitter` =
fingertip step / hand extent on quasi-static frames; `alt` = fingertip alternation on moving
frames; `collapse` = present hand frames with extent < 0.25 × forearm.

| arm | pose model | jitter | alt | hand conf | collapse (worst clip) | wall, 9 events |
| --- | --- | --- | --- | --- | --- | --- |
| v0 shipped chain | RTMW-m 256×192 NPU | 0.0198 | 1.376 | 0.57 | 3.3 % | 733 s |
| v1 `SubjectTracker` | RTMW-m 256×192 NPU | 0.0102 | 0.572 | 0.60 | 11.4 % | 523-666 s |
| v2x | RTMW-L 384×288 GPU f32 (labelled RTMW-X then) | 0.0177 | 1.186 | 1.00 | 22.6 % | 847 s |
| v3h | v1 + RTMPose-m hand 256×256 on hand crops | 0.0050 | 0.405 | 0.43 | 47.7 % | 737 s |

- v2x (checkpoint `rtmw-dw-x-l` 384×288 = RTMW-L, read as RTMW-X then): 92.9 % of body and 96.7 %
  of hand observations clip to 1.0, which disarms every confidence gate downstream (smoother,
  hygiene, R). Rejected then. The export emits unnormalised scores (600 COCO crops: median 4.2,
  max 8.4), so the rejection measured the export, not the model; true RTMW-X 384 scores on the
  normal scale (median 0.75, max 0.97).
- Hand-crop refinement: halves visible-hand jitter, but its confidence scale is lower (0.60 →
  0.43, moving every threshold calibrated on RTMW-m), and on occluded hands the crop holds no
  hand, so the output collapses (47.7 / 35.1 / 33.1 % of hand frames in the three worst clips).
  Not adopted this pass → `.agent/deferred.md`.
- Gated refinement (second attempt, user-requested): refine only hands the whole-body model
  scores >= 0.5, keep the whole-body confidences, fall back when the refined hand's extent leaves
  0.5-2× the whole-body hand's. Against the current pipeline (tracker + hygiene, same watch set):
  jitter 0.0098 → 0.0079 (above 0.0076 → 0.0051, right 0.0112 → 0.0122), alternation 0.600 →
  0.566, collapse rose on 0 of 22 clips, 59 % of hands refined / 36 % gated / 5 % fell back, wall
  523.8 → 741.1 s (+41 %). The user's adoption bar (jitter <= 0.006, wall <= +15 %) is missed on
  both → not adopted. Gating fixed the collapse; cost and side-view jitter remain.
- What shipped for C09 is M2.9.1's effect: jitter 0.0198 → 0.0102, alternation 1.376 → 0.572.
- `collapse` is not like-for-like across v0 and v1: v1 exports a hand on every frame (above-view
  presence 1.00 against 0.85), so its population includes the occluded frames v0 dropped. The
  median clip reads 0 on every arm; the worst clips are above views.

## M2.9 corpus, measured

Population = the M2.9 generation, now `output/corpus-2d-superseded-m29/` (`SubjectTracker`, GPU f32 detector every frame,
RTMW-m on NPU, hygiene, seated drops, SPARC v3), 379 clips; before = the rtmlib-f7 generation,
same 379 clips paired by label. Same triage instrument; it reproduces the catalog's own before
figures exactly except `flick > 5` (144 against the catalog's 143). `flick` = max of the two hands.

| metric | before | after |
| --- | --- | --- |
| `whole > 10` (C02, C11) | 144/379 | 0/379 — 144 fixed, 0 new; mean 29.4 → 0.018 |
| `iso > 10` (C05) | 106/379 | 1/379 — 106 fixed, 1 new; mean 18.4 → 0.097 |
| `alt` median | 1.504 | 0.525 (above 0.376, left 0.586, right 0.697) |
| `oof > 0.05` (C07) | 150/379 | 4/379; mean 0.077 → 0.003 |
| `gap > 0.05` (C08) | 28/379 | 11/379 (10 above, the C06 setup footage); mean 0.019 → 0.006 |
| `flick > 5` (C03) | 144/379 (above 117/155) | 167/379 (above 71, left 34, right 62) |
| `lowvis` median | 0.072 | 0.183 — above 0.509: hips score 0 there by ruling, so not like-for-like |

Corpus predicates of contract M2.9.2 (`.agent/archive/contract-m2u92.md`, frozen at `open`):

- **P08 pass** — face-derived eye points on the body eyes' side in 263 531/264 708 point-side
  comparisons, 4 per eligible frame over 66 177 frames (0.9956, bar >= 0.95; before
  33 754/732 788 over 183 197 frames = 0.0461). Mouth-corner comparisons 0.2850 → 0.7411, not
  graded by P08.
- **P09 fail** — 3648 of 2 555 251 feature-keypoint observations at visibility >= 0.3 lie outside
  the frame (0.143 %), on 84/379 clips (56 above). Overshoot median 36 px, p99 301 px, so not
  boundary rounding; elbows 1701, shoulders 1318, wrists 395; bottom edge 1709. Hygiene zeroes
  raw out-of-frame points before the smoother, so a later stage relocated these after it — the
  bone-length projection (`constraints.BoneLengthSmoother`, moves both endpoints) or the filter;
  which one is unmeasured. Seen on the overhead view: forearm lines run past the bottom edge.
- **P10 instrument vacuous** — its census counts only frames with both hands' coordinates
  present, and a hand whose mean score falls to <= 0.1 (`mapping._HAND_PRESENCE_THRESHOLD`)
  exports no coordinates, so a suppressed hand never enters it: 0 on both generations.

Hand blink-outs (present → absent on consecutive frames) per 100 rows, before → after: above
12.92 → 3.10, left 5.75 → 2.63, right 0.46 → 3.27; corpus 18 744 → 10 074. On the right view,
46 % of blink-outs follow a frame where the blinking hand sat on or near the other hand (median
point distance < 0.6 of hand extent); the attribution reads the frame before the blink, and no
counter records D02 firing (P10 above), so the cause is inferred, not measured. Watched on two
side-view clips: the blinker is the occluded far hand, hallucinated on or under the visible one,
present on some frames and absent on others; the superseded generation drew it on every frame,
beside stretched limbs and legs on the table. Inference: the right-view rise is residual C05
hallucination toggled by D02 rather than a tracking regression — a per-frame D02 counter decides
it (`.agent/deferred.md`). A far hand duplicating the near one can also survive the rule
outright (hand-to-wrist distance up to 9.8 forearm lengths on one clip).

## M2.10 messy footage, measured

The footage is hand-held and crowded. Over all 379 clips × 8 sampled positions (GPU detector,
`.scratch/messy_probe.py`), 186 clips show two or more people on at least half the positions
(above 21/155, left 58/93, right 107/131), and 142 clips have a second box at least half the size
of the largest. Labelled watch set = 81 clips / 36 families / 14 subjects, 12 frames per clip =
972 frames (MAIN + `general-purpose-1/-2/-3`); 20 clips tuned thresholds, 61 held them out.

| class | measured on the labels | M2.10 response |
| --- | --- | --- |
| hallucinated hand (C05) | 357 of 1 494 decided drawn hands sit where no hand is (24 %; 195 more marked unsure), mostly the occluded far hand under the overhead camera | M2.10.1 hand presence gate: held-out 191/212 removed, 28/754 real hands lost (3.7 %; per subject median 0, max 12.3 %) |
| setup / handling footage (C06) | 44 of 972 sampled frames; dense labels on 13 clips: 6 836 setup frames | M2.10.2 task span: setup excluded 0.967 (dense) / 41/44 (sparse); task lost 0.011 (dense) / 1/855 (sparse), + 4 frames the one-span rule drops by design |
| hand-held shake | camera motion at 0.5 s is 2.5 % of wrist motion at the median clip, ≥ 50 % on 10 overhead clips; per-frame it reaches wrist speed | M2.10.2 compensation: on known synthetic shake (14 clips) SPARC abs error median 0.3-1.9 uncompensated → 0.01-0.07 compensated |
| wrong person (C02, C11) | patient's box chosen on 459/473 visible labelled task frames (0.970); body on another person or mixed on 17 labelled task frames in 9 clips (3 more in setup footage); the two worst clips (4 and 3 frames) show limb lines drawn across to another arm and a therapist leaning over the patient | none this unit → `.agent/deferred.md` |
| other person's hand drawn as the patient's (C04) | 21 of 1 494 decided drawn hands | the gate removes 7/10 held-out; mask-conditioned pose → `.agent/deferred.md` |
| inconsistent angles | 2D image-plane elbow flexion disagrees across synced views by median 60° on 14 series, no better than time-shuffled pairing (ratio 1.00) | SAM 3D Body pilot, 60 events / 1940 synced instants: elbow 5.4° synced vs 8.8° shuffled (2D 26.7° vs 29.9°) — declined (user ruling, `.agent/spec.md` Decisions) |

The label vocabulary, graders and their regeneration paths → `.claude/rules/pose-instruments.md`.

## M2.10 corpus, measured

Population = `output/corpus-2d/` M2.10 generation (M2.9 + hand presence gate + camera survey: task
span, per-row camera similarity applied in R), 379 clips; before = the M2.9 generation at
`output/corpus-2d-superseded-m29/`, same 379 clips paired by label. Same triage instrument as above.

| metric | M2.9 | M2.10 |
| --- | --- | --- |
| `whole > 10` (C02, C11) | 0/379 | 0/379; mean 0.018 → 0.000 |
| `iso > 10` (C05) | 1/379 | 0/379 — 1 fixed, 0 new; mean 0.097 → 0.000 |
| `alt` median | 0.525 | 0.531 |
| `oof > 0.05` (C07) | 4/379 | 2/379 — 3 fixed, 1 new; 67 clips rise by a median 0.00014 share |
| `gap > 0.05` (C08) | 11/379 | 13/379 — the 11 clips whose `gap` rose = exactly the 11 trimmed clips (`gap` counts decoded frames with no row, so trimming raises it by design) |
| `flick > 5` (C03) | 167/379 | 0/379 — 167 fixed, 0 new; median 3.415 → 0.000, max 35.6 → 4.2; gate-conditioned: a hand held off cannot blink, so this is no real-hand tracking claim |
| hand blink-outs / 100 rows | above 3.10, left 2.63, right 3.27 (10 074) | above 0.23, left 0.33, right 0.14 (744) |

- **Hand gate, corpus.** 160 720 of 599 140 drawn hand-frames gated (0.268; above 0.342, left
  0.229, right 0.248) — compatible with the watch set's 357/1 494 decided hands hallucinated (24 %;
  195 more unsure) under held-out removal 0.90 / real loss 0.037. Compatibility only: no corpus
  labels grade which gated hands were real.
- **P08 at corpus scale: 366/379 clips within 0.01** of the offline replay over M2.9 (347 exact);
  13 exceed it (5 above 0.05, max 0.766) — the every-clip bar fails. On 12 of the 13 the hand
  scores the pipeline exports equal M2.9's (identity over those frames alone; gated frames carry
  no score to compare) and the pipeline gates only frames the replay keeps, never the reverse.
  Mechanism, measured on the 0.766 clip (in-process rerun, CSV byte-identical to the corpus, every
  gate call logged): the smoother emitted a transient second track for 9 frames that took the
  exported row with the hands, while the subject's own track carried a handless row that switched
  its right-hand state off; back on the exported row its levels sat in the 0.50-0.65 band, so
  hysteresis held the hand off for most of the remaining 241 frames. The gate is keyed by track,
  presence is judged on the exported subject. Partition of the 13: 1 measured, 11 sharing its
  signature (compatible with the mechanism, not diagnosed — other gate histories can share it), 1
  (positional 24) whose exported scores differ from M2.9 (identity 0.90/0.92), cause unmeasured.
  → `.agent/deferred.md`.
- **P11 at corpus scale reproduces the offline module**: published `task_start_frame` /
  `task_end_frame` equal `camera_survey.survey_source` over the detection cache on 83/83 labelled
  clips → setup excluded 0.967 (dense, 13 clips) / 41/44 (sparse), task lost 0.011 / 1/855.
- **Task span, corpus**: 6 864 of 337 090 decoded frames outside the task on 11 assets, all 11
  overhead and all inside the watch set (10 densely labelled); 9 of M2.9's 10 overhead
  `gap > 0.05` clips (a label-independent setup signature; one stays untrimmed) are among them.
  Observed: trimming and high gap concentrate on ~10 overhead clips. Setup prevalence outside the
  watch set is unmeasured — the watch set samples problem clips, and `gap` is no setup ground
  truth. `assets_without_reference` 0.
- **P09 still fails**: 2134 of 2 426 249 feature-keypoint observations at vis >= 0.3 out of frame
  (0.088 %, M2.9 0.143 %) on 83/379 clips; overshoot median 17 px (M2.9 36 px).
- **Cohort** (`cohort/`, 1068 rows unchanged in shape): frame rows 334 129 → 329 810, window rows
  21 507 → 21 168; 154/1068 medians identical. Median absolute relative shift by family: elbow
  angle (60 rows) 0.000, hand angles (120) 0.080 with 96/120 changing subject count, trunk 0.005, SPARC 0.011, velocity 0.027, jerk 0.047; relative tails (p90 to ~1.4) sit on
  near-zero medians (`compensatory_pattern_index`, `posture_symmetry`: <= 0.24 of the M2.9 cohort
  IQR). Against that IQR, 89/360 hand rows and 13/708 others move > 0.5 IQR (others max 0.70; 12
  motion or wrist-path features + `elbow_angle_deg_symmetry_ratio`; 5 in `glass-l`, home of two
  heavily trimmed clips). Hand features
  lose subjects on the non-task hand (→ `.claude/rules/cohort.md`). The M2.10 mechanisms
  (compensation, trimming, gate membership, reference vertical) can move these features; no
  factorization attributes each shift, and the one distinct-score P08 clip stays unexplained.

Instruments: `.scratch/triage.py` → `.scratch/watch/triage_m210.json`, `.scratch/after_table.py`,
`.scratch/oof_probe.py`, `.scratch/flick_cause.py`, `.scratch/messy/{p08_compare,p08_sign,p08_keys,
p11_corpus}.py` (→ `.claude/rules/pose-instruments.md`).

## M4.6 out-of-frame residue — stage attribution (P07)

Since M4.6 the export pass zeroes every scored point outside the frame and counts it
(`exported_out_of_frame_zeroed`), so no stage can export one. P07 names the stage that places them.
Hit set = the 6 clips with the most M2.10 P09 observations (83 hit clips, 2134 observations);
each replayed whole at HEAD (`bench.replay`, RTMW-m, CPU detector = GPU f32 to IoU 1.0000) under
three arms. Census = points the export pass zeroed, all 133 keypoints:

| clip (view) | all stages | bone-length off | smoother + bone-length off |
| --- | --- | --- | --- |
| 1 (above) | 272 | 32 | 0 |
| 2 (above) | 253 | 224 | 0 |
| 3 (right) | 270 | 285 | 0 |
| 4 (above) | 284 | 252 | 0 |
| 5 (above) | 234 | 234 | 0 |
| 6 (left) | 3877 | 3909 | 0 |

- **The temporal filter places the residue**: with bone-length projection off it persists on every
  clip (4936 of 5190 points); with the filter off too, 0 on all 6.
- **Bone-length projection adds to it on 3 clips** (clip 1: 240 of 272; clips 2 + 4: 29 + 32),
  pulls points back inside on 2 (clips 3 + 6) and moves none on clip 5.
- Unrun arm: filter off with bone-length on. `--no-smooth` also skips the bone-length stage,
  which needs the smoother's track keys, so the filter's sole share is bounded by the two arms
  above, not isolated.

Instrument: `.scratch/impl/p07_stage.py` (→ `.claude/rules/pose-instruments.md`).

## M4 corpus, measured

Population = the two M4 corpus runs, 379 clips each, one code state (landmarks `48078b4`, clinical
stage at HEAD R): RTMW-m 256 (`output/corpus-2d-m4-rtmw-m-256`, NPU pose, identity calibration)
and RTMW-L 384 (`output/corpus-2d-m4-rtmw-l-384`, GPU pose, reference-equivalent calibration);
before = M2.10 (`output/corpus-2d/`), paired by label. RTMW-m M4 vs M2.10 = the generation change
(M4.5 bouts, M4.6 fixes, OpenVINO 2026.4.1); RTMW-m → RTMW-L 384 = the deployment change
(checkpoint + input size + score map + device), not an isolated architecture effect. Own support
= each arm's points at vis >= 0.3.

| metric | M2.10 | RTMW-m 256 | RTMW-L 384 |
| --- | --- | --- | --- |
| `alt` mean, own support (above / left / right) | 0.569 (0.418 / 0.607 / 0.719) | 0.569 | 0.478 (0.387 / 0.472 / 0.590) |
| `alt` per-clip mean, matched moving support (above / left / right) | — | 0.356 / 0.477 / 0.550 | 0.367 / 0.428 / 0.505 |
| `flick` mean | 0.269 | 0.256 | 0.155 |
| hand blink-outs / 100 rows (above / left / right) | 0.23 / 0.33 / 0.14 | 0.22 / 0.31 / 0.13 | 0.13 / 0.19 / 0.08 |
| `lowvis` mean (above) | 0.291 (0.507) | 0.291 (0.509) | 0.201 (0.377) |
| `oof > 0.05` | 2/379 | 0/379 | 0/379 |
| transitions eligible for `whole` / `iso` (all ten used points) | — | 38 619 | 90 287 |
| hand-frames gated / drawn | 160 720 / 599 140 (0.268) | 159 842 / 599 140 (0.267) | 143 816 / 623 822 (0.231) |
| pose latency median (assets > 100 ms) | 51.4 ms (10) | 54.7 ms (1) | 130.2 ms (264) |

- **Generation change, small on shared cells.** On the 888 cohort cells both schemas share: 749
  medians identical, 0 move > 0.5 of the M2.10 IQR (max 0.20, `compensatory_pattern_index`), 0
  change subject count. The schema itself changes (180 window-smoothness cells leave as
  `construct_invalid`, 168 bout cells join), out-of-frame observations fall to 0, and one clip each
  gains a nonzero `whole` / `iso` rate below the 10 flag.
- **`whole` / `iso` are conditional rates.** They count transitions with all ten used points
  scored; 278/379 RTMW-m M4 clips — every overhead clip — have none, and `triage.py` prints those
  as 0 (→ `.agent/deferred.md` triage row).
- **Deployment change, large.** Over the 1056 v2 cells: 221 move > 0.5 IQR (median 0.225, max
  2.50), 7 identical, subject count changes on 265 (211 up, 54 down). Summed feature values rise
  25.5 % — repeated measurements of largely the same frames (frame rows 329 810 → 329 705, bout
  rows 6524 → 6185), not more data. On identical finite frame + window support 149/888 cells still
  move > 0.5 IQR (all support: 182/888), strongest in displacement + velocity → values move, not
  coverage alone. Which arm sits nearer the truth is unidentified: public localisation improves
  (§ Public screening in `benchmark.md`), M4.3 is a gross non-inferiority pass, and the cross-view
  residual rises (ratio 1.23).
- **Temporal proxies.** Alternation, flicker and blink-outs fall on each arm's own support; on
  matched moving support left + right fall and overhead rises; gross steps (> 0.10) on common point
  support 788 → 433 over 2 388 953 transitions. Lower irregularity of the exported trajectories —
  no dynamic GT separates denoising from attenuation, so no motion-fidelity gain is claimed.
- **Coverage.** At the 0.3 floor RTMW-L 384 admits 317 109 matched frame × point observations that
  RTMW-m holds below it (12 166 the reverse); the gap persists from floor 0.2 to 0.65, so no single
  cutoff explains it. In-domain correctness of the added observations is unmeasured
  (→ `.agent/deferred.md` calibration-transfer row).
- **Jump tails.** RTMW-L 384 gains `iso` flags on 11 clips and `whole` on 4; 16 of 16 and 5 of 6
  flagged transitions have no RTMW-m eligibility, so the flags follow the denominator. The steps
  are not all minor: max isolated step 0.876, whole-body 0.501 and 0.885 of the frame's max
  dimension.
- **Severe steps: 1.5× on RTMW-L 384, carried by the points it adds.** Severe point-step = one
  of the ten used points moving > 0.25 of the max frame dimension between adjacent frames, both
  ends at vis >= 0.3. Per million valid point-steps: M2.10 262 / 2 402 082 (109.1), RTMW-m
  256 / 2 400 939 (106.6), RTMW-L 384 442 / 2 717 441 (162.7). On points both arms score at both
  ends (2 388 953 point-steps): RTMW-L 249 (104.2), RTMW-m 235 (98.4); on RTMW-L's own extra
  points (328 470): 190 (578.4) → the excess sits in the low-score observations RTMW-L adds
  (→ calibration-transfer row), not in its localisation of shared points. Frames: 79 of RTMW-L's
  165 severe transitions coincide with an RTMW-m severe transition (mechanism unmeasured: shared
  image content, correlated model error, crop or identity); 1 per arm is camera-wide. Bouts with a
  severe step on their own wrist or index: RTMW-L 84/6185 (1.4 %), RTMW-m 66/6524 (1.0 %)
  (→ `.agent/deferred.md` severe-step row).
- **P09 + P10 close**: 0 out-of-frame observations on both trees; `duplicate_hand_fired` 34 477
  over 212 assets (RTMW-L 384), 60 077 over 269 (RTMW-m). **P08**: RTMW-m tree 3/379 clips beyond
  0.01, max 0.049 (M2.10: 13/379, max 0.766) — the measured transient-track mechanism is addressed,
  3 residuals unexplained.

Instruments: `.scratch/triage.py` → `.scratch/impl/triage_{m256,l384}.json`,
`.scratch/after_table.py`, `.scratch/impl/probes/{cohort_shift,iso_clip}.py`,
`.scratch/scientist-2/{aggregate_probe,common_support,landmark_support}.py` (appraisal
`.scratch/agents/scientist-2.md`), `.scratch/impl/jump_{count,trace,bouts}.py`
(→ `.claude/rules/pose-instruments.md`).

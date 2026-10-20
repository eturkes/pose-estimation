# Pose-pipeline benchmark — instruments

Contracts: `.agent/archive/contract-m4u1.md` (instrument qualification). Code:
`src/pose_estimation/bench/`, CLI `scripts/bench.py` (run under the gate prefix). Every
patient-adjacent output (samples, corpus sheets, labels, judge keys) lives under `.scratch/bench/`,
git-ignored; public-GT tables carry no corpus identifier.

## Codebook v2 — labels on raw frames

A labeller never sees model output. One record per frame (`bench/labels.py`, validator
`labels.validate`):

| field | values |
| --- | --- |
| `task` | `task` (the patient does the task or rests between attempts) · `setup` (camera handled, pointing away, operator footage) · `?` |
| `patient` | box `[x0, y0, x1, y1]` around the patient's visible body, or `null` |
| `others` | one box per other visible person |
| `sides.L/R.hand` | the patient's anatomical left / right hand: `vis` all of it visible · `part` some joints visible · `occ` in frame, hidden · `oof` out of frame · `?` |
| `sides.*.extent` | tight box over the hand's visible pixels (`vis`/`part` only) |
| `sides.*.joints` | `wrist`, `thumb_tip`, `index_mcp`, `index_tip`, `little_tip`: `state` ∈ `vis`/`occ`/`oof`/`?`, `xy` when `vis`, optional `own` ∈ `patient`/`other`/`ambig` |
| `sides.*.body` | `shoulder`, `elbow`, `wrist`, same joint form |

Laterality = the patient's own side, read from the body (an overhead view shows the patient from
above; a side view mirrors nothing). Unsure → `?`, never a guess.

## Sheets + answers

Labellers answer in grid units; `scripts/bench.py ingest` converts to source-frame pixels.

- **Frame sheet** (pass A): the whole frame, ticks every 5 % of width / height, labels 0-100.
  Answer: `{"sheet": "<token>.png", "task": "task", "patient": [u0, v0, u1, v1] | null, "others":
  [[…]], "L": {"hand": "vis", "extent": [u0, v0, u1, v1]}, "R": {"hand": "occ"}, "note": ""}`.
- **Body sheet** (pass B): a square crop around the patient box, a 16 × 16 grid (0-16). Answer:
  `{"sheet": …, "L": {"shoulder": {"state": "vis", "uv": [u, v]}, "elbow": …, "wrist": …}, "R": …}`.
- **Hand sheet** (pass B): a square crop at 2 × the hand extent, 16 × 16 grid; the caption names
  the side. Answer: `{"sheet": …, "joints": {"wrist": {"state": "vis", "uv": [u, v]},
  "thumb_tip": …, "index_mcp": …, "index_tip": …, "little_tip": …}}`.
- Joint definitions: `wrist` = centre of the wrist crease line; `index_mcp` = the index knuckle;
  tips = the fingertip's end (nail tip side), not the pad centre.

## Pairwise judgment sheets

Two identical markers, one yellow and one blue, on a crop around one joint; the job names the
joint. Answer one of `yellow` · `blue` (that marker sits closer to the true joint) · `both-ok`
(both on the joint) · `both-wrong` (neither) · `?`. Each pair appears twice with the colours
swapped; the key mapping colours to models stays sealed outside the sheet directory.

## Regeneration

```sh
P=scripts/gate.sh
$P python scripts/bench.py public-sheets --coco-wb .scratch/data/cocowb/coco_wholebody_val_v1.0.json \
  --images .scratch/data/coco/val2017 --seed 20261019 --out .scratch/bench/qual/public \
  --gt .scratch/bench/qual/sealed/public_gt.json
$P python scripts/bench.py sample --watch .scratch/messy/watch_order.json --seed 20261019 \
  --out .scratch/bench/pilot/sample.json
$P python scripts/bench.py frame-sheets --sample .scratch/bench/pilot/sample.json --seed 20261019 \
  --out .scratch/bench/pilot/sheets
# M4.2 — inference under the accelerator recipe (`.claude/rules/gates.md`), scoring under $P:
bench.py public-eval --coco-wb .scratch/data/cocowb/coco_wholebody_val_v1.0.json \
  --images .scratch/data/coco/val2017 --arm <arm> --out .scratch/bench/public
bench.py hand-eval --set dexycb --root .scratch/data/dexycb/20200709-subject-01 --arm <arm> \
  --out .scratch/bench/handsets   # likewise --set hint --root …/TEST_newdays_img
$P python scripts/bench.py screen --coco-wb .scratch/data/cocowb/coco_wholebody_val_v1.0.json \
  --public .scratch/bench/public --dexycb .scratch/data/dexycb/20200709-subject-01 \
  --handsets .scratch/bench/handsets --arms <arm…> --out .scratch/bench/screen.json
# Paired 500-person COCO subset (the Sapiens draw): per arm, then screen over the same draw.
bench.py public-eval … --arm <arm> --limit 500 --seed 20261019 --out .scratch/bench/public
$P python scripts/bench.py screen … --coco-limit 500 --coco-seed 20261019 --arms <arm…>
# Cross-view residual: collect under the accelerator recipe, compare under $P.
probe_crossview_arm.py collect --arm <arm> --device CPU --cache .scratch/calib-obs-arm-<arm> \
  --frames-per-event 32 --stratum-events 25
$P python scripts/probe_crossview_arm.py compare --cache rtmw-m-256=.scratch/calib-obs-wide \
  --cache <arm>=.scratch/calib-obs-arm-<arm> --baseline rtmw-m-256
```

COCO val2017 images + `person_keypoints_val2017.json` come from `images.cocodataset.org`; the
COCO-WholeBody v1.0 val JSON from the authors' Google Drive id `1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb`.

## Measured resolutions (M4.1)

Public-GT labeller error (60 COCO-WholeBody hands with all 5 feature joints visible, 40 COCO
persons; 2 labellers, `scripts/bench.py public-score`): error / GT box diagonal, p50 / p90.

| joints | general-purpose-4 | general-purpose-5 |
| --- | --- | --- |
| body shoulder / elbow / wrist (person box) | 0.023-0.035 / 0.048-0.083 | 0.020-0.027 / 0.045-0.071 |
| hand wrist / thumb tip / index knuckle (hand box) | 0.103-0.131 / 0.395-0.595 | 0.101-0.105 / 0.284-0.466 |
| hand index tip / little tip (hand box) | 0.043-0.060 / 0.120-0.297 | 0.044-0.073 / 0.202-0.436 |

Joint states vs COCO visibility: of 500 GT-visible joints the labellers called 380 / 374 visible,
98 / 101 unsure (`?`), 20 / 22 hidden, 2 / 3 out of frame. RTMW-m on the same COCO-WB hands (GT person boxes): hand PCK@0.1 =
0.551, i.e. its median hand error ≈ 0.09 of the hand diagonal — as coarse as the labellers.
**Verdict:** agent coordinates qualify as gross-error ground truth only (hand joints at >= 0.6 of
the hand diagonal, arm joints at >= 0.1 of the person diagonal); they cannot rank models on fine
hand accuracy.

Pairwise judge (general-purpose-6; 120 COCO-WB hand-joint pairs from real RTMW-m vs RTMW-X
predictions, each shown twice with colours swapped; `judge-score`): decisive accuracy by true
error margin |eA − eB| / hand diagonal.

| margin | shown | decisive | correct | Wilson 95 % | abstain |
| --- | --- | --- | --- | --- | --- |
| < 0.05 | 60 | 30 | 18 | 0.42-0.75 | 30 |
| 0.05-0.1 | 60 | 46 | 25 | 0.40-0.68 | 14 |
| 0.1-0.2 | 60 | 44 | 34 | 0.63-0.87 | 16 |
| >= 0.2 | 60 | 25 | 13 | 0.34-0.70 | 35 |

Colour-swap consistency 116/120 pairs: stable answers, mostly not the truth. **Verdict:** the
judge is not qualified in any bin (bar: Wilson lower bound >= 0.8, abstention <= 0.5); no
in-domain fine-accuracy ranking rests on agent judgment. In-domain fine accuracy comes from the
user's clicked confirmation frames (`/label`), public GT from `public-eval` + `hand-eval`.

Corpus pilot inter-labeller agreement (102 frames, 14 subjects; general-purpose-4 vs -5, each
labelling its own frame + crop sheets; `scripts/bench.py agree`; `?` on either side excluded):

| field | agreeing / compared | Wilson 95 % |
| --- | --- | --- |
| task state | 101 / 101 | 0.963-1.000 |
| hand state (per side) | 186 / 192 | 0.934-0.986 |
| hand joint state | 549 / 575 | 0.935-0.969 |
| arm joint state | 557 / 612 | 0.885-0.930 |
| patient box IoU >= 0.5 | 98 / 98 | — |

Hand states in the corpus are mostly partial (103 agreed `part`, 63 `occ`, 14 `vis`): the patient's
hands hold objects. Coordinate disagreement between the labellers, p50 / p90: hand joints
0.015-0.066 / 0.040-0.176 of the labelled hand extent diagonal (wrist 0.052 / 0.117), arm joints
0.018-0.031 / 0.039-0.089 of the patient-box diagonal. Disagreement measures shared precision,
not truth. **Verdict:** categorical states qualify (every Wilson lower bound >= 0.8); consensus
coordinates (agreed state, mean point) qualify for gross localisation only — hand joints at 0.3 of
the extent diagonal, arm joints at 0.1 of the patient diagonal (above both labellers' p90).

Pre-roll (20 untrimmed pilot frames, RTMW-m; `bench.py preroll`): a replay window ending at the
frame against the whole clip replayed under the same code, equivalent = every point the R stage
reads within 0.001 frame-normalised with no visibility flip.

| pre-roll | 0 s | 1 s | 2 s | 4 s | 8 s |
| --- | --- | --- | --- | --- | --- |
| equivalent frames | 0/20 | 6/20 | 8/20 | 11/20 | 14/20 |
| frames with a visibility flip | 6 | 4 | 4 | 2 | 0 |

Hands converge by 2-4 s; at 8 s, 4 frames still differ by 0.16-0.39 on face and shoulder points —
the subject tracker's sticky choice locked onto another person. **Verdict:** no finite pre-roll
stands in for the whole clip, so `presence`, `pilot-eval` and the M4.4 confirmation replay each
clip whole.

## Public screening (M4.2)

Pose-only protocol: GT person boxes on COCO-WholeBody v1.0 val (10 777 non-crowd persons,
82 749 annotated hand joints, 24 476 visible arm joints); the shipped detector's box on DexYCB
subject 01 (4 034 of 4 433 hands covered, 84 714 joints, every 12th frame) and HInt NewDays
(1 548 of 1 754 hands covered; 19 910 visible + 11 313 occluded joints). Intervals = paired
image-bootstrap percentiles, 1 000 resamples shared by every arm (`bench.py screen`, contract
`contract-m4u2.md` A02); GPU f32 for every arm.

| arm | whole-body AP | hand AP (mean L+R) | DexYCB PCK@0.1 | HInt PCK@0.1 vis / occ | occl. AUROC | arm PCK@0.05 | D03 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTMW-m 256 (shipped) | 0.634 | 0.534 [0.520, 0.548] | 0.691 [0.684, 0.699] | 0.782 / 0.524 | 0.646 | 0.909 | baseline |
| RTMW-L 256 | 0.669 | 0.577 [0.563, 0.593] | 0.723 [0.716, 0.730] | 0.824 / 0.587 | 0.668 | 0.922 | dominated (L 384, X 384) |
| RTMW-L 384 | 0.714 | 0.648 [0.635, 0.664] | 0.740 [0.733, 0.747] | 0.888 / 0.674 | 0.706 | 0.931 | **finalist** |
| RTMW-X 384 (cocktail13) | 0.708 | 0.640 [0.626, 0.655] | 0.733 [0.726, 0.740] | 0.862 / 0.634 | 0.674 | 0.934 | dominated (L 384) |
| DWPose-L 384 | 0.620 | 0.516 [0.501, 0.532] | 0.676 [0.668, 0.684] | 0.690 / 0.479 | 0.617 | 0.888 | dominated (all RTMW) |

Paired RTMW-L 384 − RTMW-X 384: hand AP +0.009 [0.003, 0.016], DexYCB +0.007 [0.004, 0.011] —
the larger checkpoint loses on hands at equal input size.

Sapiens2-0.4B (OpenVINO GPU f16, flip TTA; 500-person subset, seed 20261019, paired over its
436 images): hand AP mean 0.626 against RTMW-L 384's 0.650 on the same persons (paired −0.024
[−0.063, 0.018]), body AP 0.792, arm PCK@0.05 0.941, at ~1.37 s per person (~17× RTMW-L 384).
Dropped by user ruling: no hand gain for ~5 GPU-days per corpus run. ARCTIC, Ego-Exo4D and UBody
are unrun (no registration).

### Sapiens v1 (gated checkpoints)

COCO-WholeBody checkpoints from `noahcao/sapiens-pose-coco` (access granted), strict-loaded and
exported per size; OpenVINO GPU f16, the same 500-person subset
(`public-eval --limit 500 --seed 20261019`, 436 images, the Sapiens2 draw). Hand AP pairs = image
bootstrap vs RTMW-L 384 on that draw (1 000 resamples). s/person = measured `public-eval` wall over
the 500 persons (decode included; both 2B runs overlapped a CPU-bound INT8 calibration). Corpus
days = projection: 337 090 frames × (s/person + 0.1205 s), the non-pose share of RTMW-L 384's
measured corpus run (150.4 ms/frame − 29.9 ms/call).

| arm | whole AP | body AP | hand AP L / R (mean) | hand AP − RTMW-L 384 | arm PCK@0.05 | s/person | corpus days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTMW-m 256 (GPU f32) | 0.622 | 0.717 | 0.526 / 0.509 (0.517) | −0.133 [−0.174, −0.094] | 0.902 | — | 0.33 (measured, NPU) |
| RTMW-L 384 (GPU f32) | 0.710 | 0.773 | 0.655 / 0.646 (0.650) | — | 0.934 | — | 0.59 (measured) |
| Sapiens 0.3B flip | 0.612 | 0.665 | 0.580 / 0.552 (0.566) | −0.085 [−0.120, −0.049] | 0.877 | 0.72 | 3.3 |
| Sapiens 0.3B | 0.587 | 0.648 | 0.537 / 0.546 (0.541) | −0.109 [−0.145, −0.072] | 0.868 | 0.38 | 2.0 |
| Sapiens 0.6B flip | 0.672 | 0.738 | 0.642 / 0.622 (0.632) | −0.018 [−0.048, 0.015] | 0.909 | 1.53 | 6.4 |
| Sapiens 0.6B | 0.661 | 0.729 | 0.629 / 0.627 (0.628) | −0.022 [−0.050, 0.008] | 0.901 | 0.75 | 3.4 |
| Sapiens 1B flip | 0.719 | 0.785 | 0.699 / 0.654 (0.677) | +0.026 [−0.006, 0.057] | 0.934 | 2.09 | 8.6 |
| Sapiens 1B | 0.716 | 0.777 | 0.695 / 0.641 (0.668) | +0.018 [−0.010, 0.048] | 0.928 | 1.03 | 4.5 |
| Sapiens 2B flip | 0.735 | 0.804 | 0.693 / 0.692 (0.692) | +0.042 [0.007, 0.078] | 0.941 | 5.12 | 20.5 |
| Sapiens 2B | 0.724 | 0.795 | 0.667 / 0.692 (0.680) | +0.029 [−0.003, 0.062] | 0.929 | 2.54 | 10.4 |

Idle GPU f16 per call (median, batch 1): 0.3B 333 ms (flip pair batched 528 ms), 0.6B 694, 1B
909, 2B 2021; GPU f32 0.3B 3688 ms (no XMX path); NPU 0.3B 659 ms. Parity, 20 COCO crops × 133
joints vs PyTorch f32: 0.3B f32 IR on CPU max 0.31 px; f16 on GPU median 0.005 px, 23 of 2 660 points
over 1 px, every point over 5 px at score <= 0.22 (argmax flips between near-equal modes). Whole AP
sits 0.008-0.023 under the published table (0.620 / 0.695 / 0.727 / 0.745, detector boxes) while
RTMW-L 384 sits 0.013 over its own → these Sapiens figures are plausibly conservative (GT boxes,
f16, n = 500). 2.0 % of corpus frames lie outside the task span (6 864 / 337 090), so a span-only
second pass saves nothing. DexYCB is unrun for every Sapiens size. **Dropped** (user ruling,
`contract-m4u2.md` A04): one corpus run may take <= 1 day of iGPU, and every size projects over it.
The converter, heatmap arm and tests live at `f35f83e` (removed from the tree with the weights);
regenerate there: `scripts/convert/sapiens.py --stage export|parity`, then `public-eval --arm
sapiens-<size>[-noflip] --limit 500 --seed 20261019`.

SDPose-Wholebody (MIT; `teemosliang/SDPose-Wholebody` @ `0c8153e`, code `T-S-Liang/SDPose-OOD` @
`0ba1ced`): strict CPU port (1 316 tensors), OpenVINO f32 + f16 IR at 1024×768, two-crop CPU f32
parity max 0.0002 px. GPU f16, idle, batch 1: 481 ms/call (MAIN re-time; port author 489) →
2.35 days per corpus run without flip, 4.22 with → over the 1-day ceiling; accuracy unscreened
(A04 latency gate). Port removed with its weights.

## Cross-view residual per arm (3D-negative ruling)

The 3D ruling rests on RTMW-m keypoints: per-keypoint signed epipolar residuals at 15-20 px
systematic (`.claude/rules/geometry.md`). `scripts/probe_crossview_arm.py collect` recollects
M2.6's observability cache with the pose model swapped for an arm — same synchronised frames, CPU
detector, cache layout, the probe imported unchanged — and maps scores through the arm's
`score_calibrations/` file, so the probe's 0.5 validity cut keeps one meaning. `compare` runs
`probe_calibration_bias.pair_structure` per (event, camera pair).

Rule, committed before any arm result: over the units every arm scores, ratio = median
`median_abs_px` arm / RTMW-m; material drop = ratio <= 0.75 with an event-clustered bootstrap 95 %
upper bound < 1 → the 3D ruling returns to the user. 0.75 takes the ~17 px residual below its
15-20 px band.

Result (115 events, 32 synchronised frames each; L384 pose on CPU f32, the production arm's
GPU f32 numerics; baseline = the M2.6b RTMW-m cache, which the wrapper reproduces byte-equal in
points + boxes over the 22-event sample):

| arm | pairs scored | paired median residual | split-half r (paired) | r > 0.5 |
| --- | --- | --- | --- | --- |
| RTMW-m 256 | 178 | 17.0 px | 0.814 | 129/178 |
| RTMW-L 384 | 197 | 21.0 px | 0.817 | 143/178 |

Ratio 1.234 [1.023, 1.463] over 178 paired units / 103 events → **no material drop; the 3D
ruling stands.** The per-keypoint bias stays as reproducible as on RTMW-m. Confound checked: the
calibrated 0.5 cut admits more points per pair for L384 (median 34 vs 24), yet on the 39 pairs
where it admits no more, the ratio is 1.408 — the rise is not a population artifact.

## Public data provenance

COCO val2017 + annotations: `images.cocodataset.org` (HTTPS verified). COCO-WholeBody v1.0 val:
the authors' Google Drive id above. DexYCB subject 01 (`20200709-subject-01.tar.gz`, 12 GB):
the official Google Drive link on `dex-ycb.github.io`; joint order checked against the image (wrist,
then thumb → little finger, base → tip = COCO-WholeBody hand order). HInt partial annotations:
`fouheylab.eecs.umich.edu`, whose TLS certificate had **expired** at download — fetched with
certificate verification off (`curl -k`), so its bytes are unauthenticated; HInt feeds only the
occlusion-AUROC screening row, never a selection gate. Every set lives under `.scratch/data/`.

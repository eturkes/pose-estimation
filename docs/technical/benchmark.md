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
the larger checkpoint loses on hands at equal input size. Sapiens / Sapiens2 rows follow when
their arm runs (`contract-m4u2.md` D01).

## Public data provenance

COCO val2017 + annotations: `images.cocodataset.org` (HTTPS verified). COCO-WholeBody v1.0 val:
the authors' Google Drive id above. DexYCB subject 01 (`20200709-subject-01.tar.gz`, 12 GB):
the official Google Drive link on `dex-ycb.github.io`; joint order checked against the image (wrist,
then thumb → little finger, base → tip = COCO-WholeBody hand order). HInt partial annotations:
`fouheylab.eecs.umich.edu`, whose TLS certificate had **expired** at download — fetched with
certificate verification off (`curl -k`), so its bytes are unauthenticated; HInt feeds only the
occlusion-AUROC screening row, never a selection gate. Every set lives under `.scratch/data/`.

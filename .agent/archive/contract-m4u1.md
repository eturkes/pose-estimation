# Contract — M4.1 benchmark instrument qualification

Tier `kernel`: the instruments it qualifies select every model + parameter the corpus publishes,
and nothing downstream re-checks them. Base = the commit carrying this file. Frozen at dispatch;
amendments append to §8. Design review = `consultant-2` (`.scratch/agents/consultant-2.md`, C01-C09:
direction right, D10 rejected, qualify instruments before any selection protocol freezes).

User rulings (this session): accuracy above all, no cost bar; GT = public 2D GT through the
deployment + agent-placed in-domain labels + GT-free proxies; corpus frames viewable this session
by MAIN + labelling teammates; any model family + licence.

Scope: measure what each in-domain instrument can resolve. This unit ranks no arm and selects
nothing; its outputs are the resolutions the selection contract (M4.4) may rely on.

## 0. Exposure ledger

Seen before this unit: COCO val2017 body-17 OKS of 4 RTMW arms on 600 GT-box crops (`13fc4dd`
body); corpus watch-set metrics of the shipped chain, RTMW-L 384 (as "RTMW-X") and two hand-crop
refinements (`docs/technical/pose-weak-points.md` § M2.9.3); M2.10 hand-gate + span graders on the
972-frame watch set (81 clips / 36 families / 14 subjects). Agent-placement pilot (MAIN, 4 COCO-WB
hand crops): median error 0.16 of hand-box diagonal, max 0.53.

## 1. Artifact — API frozen here (A01)

Package `src/pose_estimation/bench/` (new); CLI `scripts/bench.py` (`sample`, `frame-sheets`,
`crop-sheets`, `ingest`, `public-sheets`, `public-score`, `judge-public`, `judge-score`); doc
`docs/technical/benchmark.md` (codebook, sheets, answers, regeneration, measured resolutions).

- `labels`: `TASK_STATES = ("task", "setup", "?")`, `HAND_STATES = ("vis", "part", "occ", "oof",
  "?")`, `JOINT_STATES = ("vis", "occ", "oof", "?")`, `OWNERS = ("patient", "other", "ambig")`,
  `SIDES = ("L", "R")`, `HAND_JOINTS = {"wrist": 0, "thumb_tip": 4, "index_mcp": 5, "index_tip": 8,
  "little_tip": 20}`, `BODY_JOINTS = ("shoulder", "elbow", "wrist")`; `validate(record) ->
  list[str]` (every defect, empty = valid); `dumps(record) -> str` (sorted keys, compact);
  `load(path) -> Iterator[dict]`; `write(path, records)` (one `dumps` line each).
- `render`: `PAD = 40`; `SheetMap(origin_x, origin_y, scale_x, scale_y, zoom)` with
  `to_frame(u, v)`, `to_units(x, y)`, `to_sheet(x, y)`; `frame_sheet(image, max_side=1600, step=5)
  -> (canvas, SheetMap)` (units = percent of width / height); `crop_sheet(image, centre, side,
  out=640, grid=16) -> (canvas, SheetMap)` (units = grid cells; outside-frame area grey).
- `sample`: `VIEWS = ("above", "left", "right")`; `Clip(event_id, camera_name, subject, task, side,
  view, n_frames)`; `SampleFrame(event_id, camera_name, frame, stratum, subject, view)`;
  `corpus_clips(sessions, inventory, run) -> list[Clip]`; `family_labels(clips) -> {(event_id,
  camera_name): "#NNN task/side/view"}`; `corpus_sample(clips, seed, watch, subjects=8, per_view=3,
  stress=30) -> list[SampleFrame]` (`watch` = set of `(event_id, camera_name)`); `read_frames(path,
  indices)` yields `(index, frame)` in sequential-decode numbering.
- `judge`: `COLOURS` (yellow, blue), `RESPONSES = ("yellow", "blue", "both-ok", "both-wrong",
  "?")`, `BINS = (0, .05, .1, .2, inf)`; `Pair(pair_id, image, joint, a, b, arm_a, arm_b, side)`;
  `make_sheets(pairs, load_image, sheet_dir, key_path, seed) -> jobs` (two swapped sheets per pair,
  key JSONL at `key_path`); `decode(key, answers) -> rows` (`answers` = sheet file name → response;
  outcome = the arm under the answered colour, else the response); `bin_of(margin) -> int`.
- `public`: `GtHand(image_id, ann_id, side, box, kpts)` (`diagonal` property); `GtPerson(image_id,
  ann_id, box, body, area, hands)`; `load_coco_wholebody(path) -> (files, persons)`.
- `qualify`: `NON_DECISIVE`; `wilson(k, n)`; `quantiles(values, qs=(.5, .9))`;
  `normalized_errors(pred, gt, norm)`; `agreement(a, b) -> (agreeing, compared)` (`?` excluded);
  `box_iou(a, b)`; `judge_bins(margins, truths, outcomes, bins) -> rows`; `paired_subject_sd(
  diffs_by_subject) -> (sd, n_subjects)`.
- `sheets`: `SheetEntry(token, kind, key, index, width, height, side, map)`; `token(seed, *key)`;
  `write_sheet(...)`, `hand_crop(image, extent)`, `body_crop(image, box)`, `save(out, entries,
  seed)` (`manifest.json` + `jobs.json`), `load_manifest(out)`, `frame_box(sm, box)`,
  `ingest(sheet_dir, answers, labeller) -> records`.
- `arms`: `PoseArm(name, url, input_size, note)`, `POSE_ARMS` (mmpose-table names), `load(arm,
  device="GPU")`, `predict(model, image, boxes_xyxy)`.
- Patient-adjacent outputs (sample manifests, sheets, labels, keys) → `.scratch/bench/`
  (git-ignored); public-GT-only tables may be committed under `docs/technical/benchmark/`.

## 2. Design decisions

**D01 — codebook v2 (raw frames, no model overlay).** Per frame: `task` ∈ {`task`, `setup`, `?`};
`patient` box or `-`; `others` = boxes of every other visible person (owner regions); per
anatomical side of the patient (`L`, `R`): `hand` ∈ {`vis`, `part`, `occ`, `oof`, `?`} (`part` =
some hand joints visible); `extent` = tight box over the hand's visible pixels (required for
`vis`/`part`); per joint in `HAND_JOINTS` = wrist(0), thumb tip(4), index MCP(5), index tip(8),
little tip(20) and `BODY_JOINTS` = shoulder, elbow, wrist: `state` ∈ {`vis`, `occ`, `oof`, `?`},
`xy` when `vis`, optional `own` ∈ {`patient`, `other`, `ambig`} where limbs overlap. Laterality =
the patient's anatomical side; `?` when the labeller cannot tell. Coordinates in source-frame
pixels (the sheet maps them).

**D02 — labelling passes.** Pass A on a frame sheet (whole frame ≤ 1600 px long side, grid every
5 % with labelled ticks): task, patient, others, hand states, extents, body joints. Pass B per
`vis`/`part` hand on a crop sheet (2 × extent, 640 px, 16 × 16 grid): hand joint states + points.
A labeller sees no model output at any pass.

**D03 — pairwise judgment sheets.** A crop sheet centred on one joint's two candidate points shows
point A and point B as identical markers in two colours drawn from a fixed pair, which colour is
`A` randomised per sheet and recorded only in a sealed key; arm ids, file names and crop
provenance never reach the sheet. Response ∈ {`A`, `B`, `both-ok`, `both-wrong`, `?`}. Each pair
is shown twice with colours swapped. Pairs come from real model outputs on public GT
(qualification) or corpus frames (later units).

**D04 — corpus pilot sample.** 8 subjects (seeded) × 3 views × 3 frames, frame positions seeded
uniform over the whole decoded clip — not the published span, which equals the whole clip on 368
of 379 assets — so missed setup footage and false trimming both stay sampleable (the `task` label
grades them); assets outside the M2.10 watch-set clips; plus 30 frames from watch-set clips
(disclosed stress subset). Two labellers label every pilot frame independently (agreement), and
both label the same 60 public-GT hands (COCO-WB val, `*_valid` hands whose 5 `HAND_JOINTS` carry
annotation value 2, seeded, stratified by hand-box size tertile) + 40 public-GT persons for body
joints (COCO val2017 `person_keypoints`, visibility 1 = labelled occluded, 2 = visible).

**D05 — qualification measurements.** (a) Labeller coordinate error vs public GT: quantiles of
error / GT hand-box diagonal per size tertile and joint (hands), / GT person-box diagonal (body);
body joint-state confusion vs COCO visibility; hand joint-state confusion vs HInt occlusion +
existence labels once HInt is fetched (else recorded unmeasured).
(b) Inter-labeller: per-field agreement (task, patient box IoU >= 0.5, hand state, joint state,
side), coordinate disagreement quantiles normalised by labelled extent diagonal. (c) Judge accuracy
on public GT pairs from real arms (RTMW m256 vs l384, m256 vs x384) binned by |eA − eB| / hand
diagonal ∈ [0, .05), [.05, .1), [.1, .2), [.2, ∞): share of correct decisive answers, abstention
share, colour-swap consistency, Wilson 95 % per bin; `both-wrong`/`both-ok` scored against GT at
0.2 diagonal. (d) Paired subject-level SD of hand-presence F1 difference between two arms on the
pilot (RTMW m256 vs x384, raw per-frame outputs at the gate's shipped thresholds) — the input to
M4.4's power statement. (e) Pre-roll: frames of production pipeline replay before a sampled frame
needed for its exported row to equal a whole-clip run (smoother + gate state), over 20 pilot
frames, pre-roll ∈ {0, 1, 2, 4, 8} s.

**D06 — resolution verdict.** An instrument is qualified at the finest bin / tolerance where its
Wilson lower bound clears: coordinates — error p90 below the tolerance claimed; categorical states —
per-class agreement >= 0.8 (lower bound) and confusion published; judge — decisive accuracy lower
bound >= 0.8 with abstention <= 0.5 in that bin. Unqualified resolutions are recorded as such and
may not be used by M4.4.

## 3. Predicates

Tester-owned (synthetic, no media, no model):
- **P01** `labels.validate`: accepts a complete v2 record; names each defect for missing extent on a
  `vis`/`part` hand, `xy` on a non-`vis` joint, a `vis` joint without `xy`, a point outside the
  frame, an unknown state / joint / owner / key, a box with non-positive area or leaving the frame,
  `sides` without exactly L + R, a `vis` hand with a hidden joint, a `part` hand with no visible
  joint, an `occ`/`oof` hand with a visible joint; `dumps` → `load` → `dumps` byte-identical.
- **P02** `render`: `to_frame(to_units(p)) == p` and `to_units(to_frame(u)) == u` (< 1e-9); a drawn
  pixel at a known frame point lands where `to_sheet` says (frame + crop sheets; landscape, portrait,
  crops clipped by the frame edge, grey outside); canvas = body + `PAD` margins.
- **P03** `sample.corpus_sample`: deterministic for a seed, different for another; `subjects` ×
  `VIEWS` × `per_view` main frames, each subject carrying every view outside `watch`; no main frame
  from a `watch` clip, every stress frame from one; `0 <= frame < n_frames`; raises when fewer
  subjects qualify or `watch` is empty with `stress > 0`. `family_labels` numbers events in
  `(task, side, event_id, camera_name)` order, shared across an event's views.
- **P04** `judge.make_sheets`: no arm name, image name or pair id in any sheet file name or job;
  each pair → exactly two sheets with opposite colour assignments; each colour = `arm_a` on half
  the sheets; key rows map tokens to arms; `decode` maps a colour answer to its arm, passes the
  three non-decisive classes through, raises on an unknown response, skips unanswered sheets;
  `bin_of` boundaries half-open as `BINS`.
- **P05** `public.load_coco_wholebody`: on a fixture JSON, crowd dropped, hands only where
  `*_valid`, `lefthand` → `L`, arrays reshaped `(21, 3)` / `(17, 3)` in native order, `diagonal`.
- **P06** `qualify`: Wilson equals the closed form (incl. k = 0, k = n); `quantiles`, `agreement`,
  `box_iou`, `judge_bins` (decisive / correct / abstain counts per bin) exact on fixtures;
  `paired_subject_sd` = SD of subject means; each raises on its empty population.
- **P06b** `sheets.ingest`: frame answer → record with boxes in frame pixels through the stored map;
  body + hand answers fill joints; a crop answer without its frame answer raises; the output of a
  complete synthetic answer set passes `labels.validate`.

MAIN-owned (real data, recorded in `docs/technical/benchmark.md`):
- **P07** D05(a)-(e) measured; populations named beside every count.
- **P08** D06 verdict per instrument, with the unqualified resolutions listed.

## 4. Invariant surfaces

No `src/` module outside `bench/` changes; no published tree is written; `tests/` gains
`test_m4u1_*.py` only.

## 6. Negative controls

- NC1 a label record with `xy` on an `occ` joint fails P01 by name.
- NC2 a judge sheet whose filename carries the arm id fails P04.
- NC3 a sample whose stratum quota exceeds its eligible frames raises in P03.
- NC4 a labeller answer file built from GT plus 0.3-diagonal noise is not qualified at 0.1 (P08 rule
  on a synthetic answer set).

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01-P06 | open | |
| P07-P08 | open | |

## 8. Amendments

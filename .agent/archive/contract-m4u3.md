# Contract — M4.3 complete-pipeline pilot evaluation (finalists → M4.4 candidates)

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
Purpose: a screening winner on public GT (pose-only, GT boxes) can still lose inside the shipped
pipeline — detector boxes, subject tracker, smoother, hand gate, out-of-frame zeroing,
calibrated scores. Replay each M4.2 finalist as a complete configuration over the pilot frames
(development data: agent labels, gross resolution, M4.1) and pass to M4.4 only configurations
that hold the baseline's gross behaviour. The user's confirmation frames stay untouched.

## 1. Inputs + artifact

- **Arms** = every M4.2 finalist (`bench.py screen` → `finalists`) as a registry configuration
  (M4.3a key + M4.3b calibration + transferred hand gate), baseline `rtmw-m-256` (shipped).
- **Frames** = the M4.1 pilot (`.scratch/bench/pilot/sample.json`, 102 frames, 8 subjects; main +
  disclosed stress), labels = the two labellers' consensus (`bench.py` `_consensus`).
- **Replay** = `bench.replay.replay_window` ending at each pilot frame with the pre-roll M4.1
  D05(e) measured (the smallest tested pre-roll whose exported rows equal the whole-clip run on
  every replayed frame; none → the largest tested, disclosed); the exported row at the frame is
  scored.
- `scripts/bench.py pilot-eval --arm <key> --preroll <s>` → `.scratch/bench/pilot_eval/<arm>/`
  (replay windows keyed by frame + pre-roll, as `presence`), one JSON of per-(frame, side) rows +
  summary.

## 2. Endpoints (per side of a pilot frame; agreed labels only — `?` excluded)

- **G1 presence F1** — exported hand present (`qualify.exported_hands`) vs agreed hand state
  (`vis`/`part` present, `occ`/`oof` absent).
- **G2 gross hand** — agreed `vis` hand joints of a present hand: share within 0.3 of the agreed
  hand extent diagonal (M4.1 qualified resolution); a joint the arm does not export = outside.
- **G3 gross arm** — agreed `vis` shoulder / elbow / wrist: share within 0.1 of the consensus
  patient-box diagonal; not exported = outside.
- **G4 side swaps** — frames with both hands present + exported: exported hand centroid nearer
  the other side's labelled hand; share.
- Wall: replay seconds per frame, reported (no bar — user ruling).

## 3. Decision

Per arm vs baseline, paired subject-cluster bootstrap (2000 resamples, seed 20261023), 95 %
percentile interval of (arm − baseline):

| state | rule |
| --- | --- |
| pass | lower(ΔG1) >= −0.05 AND lower(ΔG2) >= −0.05 AND lower(ΔG3) >= −0.05 AND upper(ΔG4) <= +0.05 |
| refit | lower(ΔG1) < −0.05 alone, or ΔG1 point < −0.02 (M4.3b D02) → grid refit of the hand gate on the pilot, then one re-evaluation; still failing → fail |
| fail | any other breach |

Margins are gross by construction: agent labels resolve 0.3 of hand extent and 0.1 of the patient
box (M4.1), and 8 subjects bound the interval width; finer accuracy is M4.4's job on the user's
frames. Every `pass` arm (plus the baseline) becomes an M4.4 candidate, named + digested in
`contract-m4u4.md` §8 A01.

## 4. Predicates

- **P01** G1-G4 exact on a synthetic replay table (labels + exported rows built in-test),
  incl. not-exported-as-outside, `?` exclusion, extent + patient-box normalisers.
- **P02** paired subject bootstrap: constant difference → zero-width interval at that value;
  seeded reproducibility; subjects (not frames) resampled.
- **P03** decision table over constructed intervals hits pass / refit / fail; refit runs once.
- **P04** `pilot-eval` refuses a pre-roll not in M4.1's tested set and an arm absent from the
  registry.
- MAIN-owned **P05**: every finalist evaluated, states + intervals + wall recorded in
  `docs/technical/benchmark.md`; candidates written into `contract-m4u4.md` §8 A01.

## 6. Negative controls

- NC1 dropping not-exported joints instead of scoring them outside reds P01.
- NC2 frame-level resampling reds P02.

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01 | pass | `tests/test_m4u3_pilot.py` P01 ×2 (`bench.pilot.side_rows` + `endpoints`): not-exported = miss, `?` out, both normalisers, swap, FP presence |
| P02 | pass | constant difference → `(0.5, 0.5, 0.5)`; subject-only loss reaches −1.0 and 0.0; seeded |
| P03 | pass | `pilot.decide` pass / refit (interval + M4.3b point rule) / fail rows |
| P04 | pass | `pilot.check_request` refuses an unknown key (A01); red 6/6 at `36e525b` |
| P05 | open | after the pre-roll (D05(e)) and the finalist set |

## 8. Amendments

- **A01 (replay).** `contract-m4u1.md` A05 measured no finite pre-roll standing in for the
  whole clip (8 s: 14/20 frames equivalent, 4 on a different subject choice), so
  §1's replay = each pilot clip replayed whole, frame 0 to its last decoded frame (the corpus run
  of that clip under the arm), rows read at the sampled frames; wall = clip seconds over its
  sampled frames. P04 = refusal of an arm outside `run.MODEL_REGISTRY` (the pre-roll argument is
  gone); `bench.pilot.check_request(arm)`.

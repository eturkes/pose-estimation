# Contract — M4.3b per-arm calibration onto the reference scale + gate refit rule

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
Purpose: give every screened arm the shipped arm's score semantics, so the pipeline's thresholds
(tracker transport floor 0.3, duplicate-hand 0.5 / 0.6, hand gate 0.65 / 0.5, mapping presence
0.1) carry their meaning across arms with zero fitted parameters — 102 pilot frames cannot be
overfit by thresholds that are never fitted on them.

## 1. Artifact

- `score_calibration.fit_reference(raw, correct, ref_raw, ref_correct) -> Curve`: per keypoint
  group, logistic (Platt) fits `P(correct | raw) = σ(a·raw + b)` for the arm and the reference
  (RTMW-m, raw); the map `raw → (a_arm·raw + b_arm − b_ref) / a_ref` = the reference raw score with
  the same probability of being correct. Affine in `raw` (σ cancels), clamped at 0, written as
  `Curve` knots spanning the arm's observed raw range (+ the zero crossing when it falls inside).
  Refuses a fit with a non-positive slope.
- `scripts/bench.py calibrate --arm <name>` writes `src/pose_estimation/calibration/<arm>.json`
  via `fit_reference` against `rtmw-m-256`'s cached COCO-WholeBody predictions (same persons,
  `CAL_TOLERANCE` correctness: body / foot 0.05, face 0.02 of the person-box diagonal, hand 0.1 of
  the hand-box diagonal); `run.MODEL_REGISTRY[<arm>]["calibration"]` names the file. RTMW-m keeps
  `None` (identity = the reference itself).
- `scripts/bench.py presence --arm <name>`: pipeline replay (`bench.replay`, M4.1 pre-roll) over
  the pilot frames; per side, exported hand present vs the labellers' agreed hand state
  (`vis`/`part` → present, `occ`/`oof` → absent, else excluded) → precision, recall, F1 with
  subject-cluster bootstrap intervals.

## 2. Design decisions

**D01 — reference-equivalent scores.** A threshold tuned on RTMW-m's raw scale means "this
probability of being correct" once each arm is mapped through its own correctness curve onto
the reference's; a stronger arm's confident keypoints map to higher reference scores, so it is
not forced into RTMW-m's confidence distribution (the quantile-matching alternative would be).
Logistic rather than isotonic: strictly monotone, so no threshold falls on a flat step where
ties would flip decisions.

**D02 — refit the hand gate only on evidence of loss.** An arm's transferred gate (0.65 / 0.5)
stays unless its pilot presence F1 falls below RTMW-m's by more than 0.02 (point estimate). Then
and only then a grid over on ∈ {0.5 … 0.8 step 0.05} × off ∈ {0.35 … on − 0.05 step 0.05} picks the
pair maximising pilot F1 (ties → the pair nearest 0.65 / 0.5), recorded in the registry entry as
`hand_gate`; the confirmation set never sees the grid.

## 3. Predicates

- **P01** `fit_reference` on synthetic logistic data recovers the affine map within 2 % of the
  generating parameters; strictly increasing; clamps at 0; refuses a non-positive slope.
- **P02** a calibrated arm's transferred thresholds: a score equal to the reference's threshold
  maps across iff the two arms' fitted P(correct) at those raw scores are equal (round trip).
- **P03** `presence` metrics exact on a synthetic replay table; sides with disagreeing or `?`
  states excluded; F1 bootstrap over subjects.
- **P04** the registry entry's `hand_gate` (when present) reaches `HandPresenceGate(on, off)` in
  `run.main`, and the pose config records it.

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01-P04 | open | |

## 8. Amendments

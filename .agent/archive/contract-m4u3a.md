# Contract — M4.3a pose-arm plumbing: registry, devices, score calibration

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
Purpose: any screened arm can run through the production pipeline (`run.py`, the corpus drivers,
`bench.replay`) with its scores on one calibrated scale, so M4.3b can refit thresholds per arm and
M4.4 can compare complete configurations. Behaviour of the shipped configuration (`--model
rtmw-l`, identity calibration) stays byte-identical.

## 1. Artifact

- `src/pose_estimation/run.py` `MODEL_REGISTRY`: entries keyed by mmpose-table names —
  `rtmw-m-256` (the checkpoint `rtmw-l` names; `rtmw-l` stays as an alias key so every recorded
  `pose_config.json` keeps meaning), `rtmw-l-256`, `rtmw-l-384`, `rtmw-x-384`, `dwpose-l-384`;
  each entry gains `pose_device` (default device: `NPU` for `rtmw-m-256`, `GPU` for the rest —
  the L/X checkpoints fail NPU compile) and `calibration` (file name under
  `src/pose_estimation/calibration/`, or `None` = identity). `--pose-device` default becomes
  `None` = the entry's device.
- `src/pose_estimation/score_calibration.py` (new): `ScoreCalibration` = per keypoint group
  (`body` 0-16, `foot` 17-22, `face` 23-90, `hand` 91-132) a monotone non-decreasing piecewise
  linear map `raw → [0, 1]` (`Curve(x, y)`, knots rising in x, not falling in y, y in [0, 1]);
  `ScoreCalibration(curves).apply(scores) -> scores` (shape-preserving, NaN stays NaN, a group
  without a curve passes raw), `load(name)` (`None` → identity), `identity()`; `fit(raw, correct)
  -> Curve` = isotonic regression (pool-adjacent-violators) of the 0/1 `correct` on `raw`,
  compressed to <= `MAX_KNOTS = 64` knots.
- `run.SplitDeviceSolution`: the pose model's scores pass through the entry's calibration before
  the tracker sees them (`CalibratedPose` wrapper), so hygiene, the smoother, the hand gate and the
  export read calibrated scores.
- `scripts/bench.py calibrate --arm <name>`: fits the arm's calibration on COCO-WholeBody val
  (GT boxes, the cached `public-eval` predictions) and writes
  `src/pose_estimation/calibration/<arm>.json` (public data, committed). Target `correct` =
  normalised error under the group's tolerance (body/foot 0.05 of person-box diagonal, face 0.02,
  hand 0.1 of the hand-box diagonal) on annotated joints; unannotated joints = incorrect.
- `pose_estimation.corpus_run.POSE_CONFIG_FIELDS` already carries `model`; the calibration file's
  sha256 joins it as `calibration` (a refit is a new generation).

## 2. Design decisions

**D01 — calibrate to correctness, per arm.** Raw SimCC/heatmap scores are checkpoint-specific
(RTMW-L 384's raw median 4.2 vs RTMW-m 0.74 on the same crops), and the smoother clips raw scores
to [0, 1] before anything downstream, so an uncalibrated arm saturates every gate. Isotonic
P(correct | raw) per group gives every arm one meaning for a score; thresholds downstream are then
refit per arm (M4.3b), not transferred.

**D02 — identity for the shipped arm until M4.3b.** `rtmw-m-256` / `rtmw-l` keep `calibration =
None`, so the shipped pipeline stays byte-identical and the M2.10 generation stays reproducible;
whether RTMW-m also runs calibrated is M4.3b's ruling, made with its thresholds.

**D03 — device from the entry.** A model that cannot compile on the requested device fails loudly
(existing behaviour keeps the NPU→CPU fallback warning); the default is the entry's device so a
corpus run cannot silently put an L/X arm on CPU.

## 3. Predicates

- **P01** Registry: every entry's URL names the checkpoint mmpose's table assigns to its key
  (`rtmw-dw-l-m` → `rtmw-m-256`, `rtmw-dw-x-l` 256/384 → `rtmw-l-*`, `rtmw-x_` → `rtmw-x-384`);
  `rtmw-l` resolves to the same entry as `rtmw-m-256`; input sizes match the URL's resolution.
- **P02** `ScoreCalibration.fit`: output monotone non-decreasing; equals the PAV solution on a
  fixture; <= 64 knots; `apply` piecewise-linear between knots, clamped outside, NaN preserved,
  per-group slices independent; `identity().apply(s) == clip(s, 0, 1)` is NOT required —
  identity returns `s` unchanged (the shipped path must stay byte-identical).
- **P03** `SplitDeviceSolution` with a calibration: the tracker receives calibrated scores
  (synthetic pose model returning known raw scores); without one it receives the raw scores
  unchanged.
- **P04** Shipped configuration unchanged: a synthetic `process_source` run with `--model rtmw-l`
  writes byte-identical CSV + diagnostics before and after this unit.
- **P05** `--pose-device` default resolves to the entry's device; an explicit value wins;
  `pose_config.json` records the resolved device and the calibration digest.

## 6. Negative controls

- NC1 a calibration with a decreasing knot pair fails `load` (P02).
- NC2 calibrating the shipped arm by default reds P04.

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01-P05 | pass | `tests/test_m4u3a_arm_plumbing.py` 36 cases (26 of the first 28 red on `487f790` in a detached worktree; the rest added for reviewer-10 R06-R08, each red on the pre-fix tree via `.scratch/agents/reviewer-10/`); P04 per A02 |

## 8. Amendments

- **A01 (before commit; M4.3b design).** `Curve` values need only be non-decreasing and >= 0: a
  calibration may map raw scores onto a reference arm's equivalent raw scale (values up to that
  arm's own range, e.g. RTMW-m's ~1.1), not only onto P(correct). `fit` (isotonic P(correct))
  stays as specified.
- **A02 (reviewer-10 register).** P04 = the shipped configuration's CSV bytes unchanged (pinned by
  `test_m2u101_hand_gate.py::test_p05_ungated_baseline_bytes_and_zero_diagnostics` + the gated
  baseline cases) and `SplitDeviceSolution` leaving an uncalibrated pose model unwrapped; the
  diagnostics row changes only by M4.6's two census fields, latency never repeating.
- **A03 (M4.3b).** Calibration files live in `src/pose_estimation/score_calibrations/` (a
  `calibration/` directory beside the camera-calibration module `calibration.py` read as the same
  import name); `run.hand_gate_for(entry, enabled=…)` builds the gate from the entry's optional
  `hand_gate` (on, off), and `calibration_digest` covers that pair too.

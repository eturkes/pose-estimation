# Contract — M2.9.5 seated-subject body-part drop: legs everywhere, hips under an overhead camera

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
User rulings (this session): zero knees, ankles and feet in the pipeline; zero the hips in above
views so trunk metrics come from the side views; hips stay elsewhere.

## 1. Artifact

- `src/pose_estimation/keypoint_hygiene.py` — `HIPS`, `LOWER_BODY`, `drop_body_parts`;
  `apply_hygiene(..., drop_lower_body=False, drop_hips=False)`.
- `src/pose_estimation/run.py` — `--drop-lower-body`, `--drop-hips-camera TOKEN`;
  `process_source(..., drop_lower_body=False, drop_hips_camera=None)`.
- `src/pose_estimation/corpus_run.py` — `POSE_CONFIG_FIELDS` gains both settings.
- `scripts/corpus_run_2d.py`, `scripts/pilot_corpus_run.py` — flags, defaults, report fields.
- `docs/technical/tracking-modes.md`, `docs/technical/entrypoints.md` — reference updates.

## 2. Design decisions

**D01 — what drops.** COCO-WholeBody indices 13-22 (knees, ankles, the six foot points) score 0
when `drop_lower_body`; indices 11-12 (hips) score 0 when `drop_hips`. Coordinates stay. Evidence:
no clinical feature in `analysis/clinical_features.R` reads a knee, ankle or foot; seated subjects'
legs sit under the table, and their scores overlap real points (right-view knee median 0.48 vs
wrist q25 0.49), so no confidence cut separates them. Above-view hips score median 0.37 at 0.000
out-of-frame — placed, not seen — and fed trunk lean, lateral lean, rotation and posture symmetry.

**D02 — where.** Inside `apply_hygiene`, after out-of-frame zeroing and before duplicate-hand
suppression (the drop touches no hand index, so the order cannot change the duplicate test), on
the tracker output before the smoother (M2.9.2 D04 unchanged). Rows of any keypoint count: a
17-keypoint row drops 13-16 and 11-12; indices past the row's end are skipped.

**D03 — which camera loses the hips.** `process_source` derives the camera label as the text after
the last `/` of `video_name` (session labels read `<session>/<camera>`; otherwise the file name);
`drop_hips = drop_hips_camera` is truthy and a substring of that label. Both settings are
`process_source` keyword parameters defaulting off; the CLI passes the parsed values, so callers
that pass neither keep every body part.

**D04 — generation identity.** `drop_lower_body` and `drop_hips_camera` join
`corpus_run.POSE_CONFIG_FIELDS`, the drivers' `REPORT_FIELDS` and `configuration`; the camera
token joins each driver's redaction allowlist. Drivers default `--drop-lower-body` on and
`--drop-hips-camera above`; `run.py` defaults both off. `GENERATOR_VERSION`: corpus v4, pilot v3.

## 3. Predicates

Tester-owned (synthetic, no media):

- **P01** `drop_body_parts`: `lower_body` zeroes exactly 13-22, `hips` exactly 11-12, together
  both sets, neither nothing; every other score unchanged; coordinates untouched; input not
  mutated; 17-keypoint rows handled; empty input keeps its shape.
- **P02** `apply_hygiene` order: out-of-frame, then the drop, then duplicate-hand — a drop never
  changes the duplicate verdict; with both flags off the output equals M2.9.2's.
- **P03** `process_source`: a session label `event/cam-above` with `drop_hips_camera="above"` drops
  the hips; `event/cam-left` keeps them; a file-name label obeys the same substring rule; no token
  → no hip drop; the exported CSV carries visibility 0 on the dropped body columns (MediaPipe
  23-32) and unchanged values elsewhere.
- **P04** CLI: `run.main` forwards `--drop-lower-body` and `--drop-hips-camera` through
  `_dispatch_sessions` to `process_source`; defaults off.
- **P05** Drivers: both pass the flags to `pose_estimation.run`, publish both in `configuration`,
  carry both in `REPORT_FIELDS` and `pose_config.json`, default on / `above`, and a complete event
  recorded under other drop settings is due on resume and refused by `--analyse-only` /
  `--reuse-run`.

MAIN-owned, real frames (watch set, 22 clips × 400 frames, against the same run without the flags):

- **P06** Knee, ankle, heel and foot-index visibility = 0 on every exported row; hip visibility = 0
  on every above-camera row and > 0 on side-camera rows where the run without the flags had it.
- **P07** Upper body moves only through the bone-length constraint: `BoneLengthSmoother` skips a
  segment with an invalid end, so dropping knees and above-view hips removes the hip-knee and
  shoulder-hip segments and can shift what they constrained. Reported per view against the run
  without the flags: median and p99 displacement of shoulders, elbows, wrists and hand roots
  (px), and the share of frames byte-identical; no other keypoint's visibility changes.

## 4. Invariant surfaces

1. Every M2.9.1 / M2.9.2 suite passes; `process_source` callers that pass neither setting keep
   M2.9.2 behaviour.
2. CSV schema (304 columns), R stage, trackers, smoother: unchanged.

## 5. Gate identity

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <ruff check|ruff format --check|ty check|pytest>
```

## 6. Negative controls

1. `LOWER_BODY = slice(13, 22)` → P01 fires.
2. Hips dropped on every camera → P03 fires.
3. The drop applied after the smoother → P03's CSV case fires (a held point under a nonzero score).
4. Drivers default the drop off → P05 fires.

## 7. Verdict table

Appended at close.

## 8. Amendments

Appended as ruled.

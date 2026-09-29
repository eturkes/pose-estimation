# 2D pose weak points — catalog + repair map

Population = the shipped `output/corpus-2d/` generation (rtmlib `PoseTracker`, `tracking=False`,
CPU detector every 7 frames, RTMW-L on NPU, `--single-subject`, `--tracking body`): 379 clips /
193 recording families. Two instruments:

- **Watch pass** — every clip through the review UI, overlay at keypoint threshold 0.3, 8 frames
  spread over each clip, 97 contact sheets. "Families noted" counts families where the class was
  seen at least once (approximate: one note line can name two families).
- **Triage** — per clip, person 0, image-isotropic coordinates (`.scratch/triage.py`, → `gates.md`):
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
| C05 | hallucinated occluded parts: legs under the table, far arm + hand, vertical body lines from the frame edge (above), limbs stretched to frame corners, duplicate hand | a top-down model emits all 133 keypoints, many at moderate confidence | `iso > 10`: 106/379; families noted 107/193 | M2.9.2 hygiene; residual = model limit |
| C06 | camera-setup footage tracked (floor, ceiling, operator) | no task segmentation | families noted 13/193 | deferred (`.agent/deferred.md`) |
| C07 | confident points outside the frame | extrapolation past the crop and frame edge at visibility >= 0.3 | `oof > 0.05`: 150/379 (above 106/155) | M2.9.2 hygiene |
| C08 | rows withheld at track birth + long gaps | smoother `min_track_age = 3` on every re-keyed track | first row >= frame 2 on 379/379; `gap > 0.05`: 28/379 | M2.9.1 identity-keyed smoothing |
| C09 | finger jitter and dropout | small hands inside a 256×192 whole-person crop | no triage metric; M2.9.3 defines one | M2.9.3 hand evaluation |
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

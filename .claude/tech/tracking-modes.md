# Tracking modes

Selected with `--tracking {hands|hands-arms|body}`. Mode constants live in `processing.tracking_pose_indices()`.

| Mode | Body keypoints | Hand keypoints | Pose detection | CSV body prefix |
|------|----------------|----------------|----------------|-----------------|
| `hands` | — | 2 × 21 | Skipped | — |
| `hands-arms` (default) | 12 (shoulders → finger bases) | 2 × 21 | Yes | `arm_` |
| `body` | 33 (all MediaPipe pose) | 2 × 21 | Yes | `body_` |

## Mode constants

- `TRACKING_HANDS`, `TRACKING_HANDS_ARMS`, `TRACKING_BODY` — string keys (also re-exported from package root).
- `BONE_SEGMENTS` vs `BONE_SEGMENTS_BODY` — arm-only vs full-body bone graph.
- `ANGLE_LIMITS` vs `ANGLE_LIMITS_BODY` — joint-angle clamps.
- Wrist/shoulder index pairs are mode-specific; consult `processing.tracking_pose_indices()` rather than hardcoding indices.

## Hand assignment

- `hands` mode: assigns left/right by wrist x-coordinate.
- `hands-arms` / `body`: Hungarian matching against arm wrists (`match_hands_to_arms`). Distality reject prevents matching a hand to an arm whose wrist is farther from the hand than the shoulder midpoint.

## Single-subject mode (`--single-subject`)

Three resilience layers for unreliable body detection (e.g. top-down views):

1. **Primary body selection** — keep the largest body. Hands that pass age/spatial filters are preserved; body-level matches re-indexed to primary.
2. **Body carry-forward** — when body detection drops, reuse the last known body for up to ~0.5 s so hands-arms matching continues. Tuned by `carry_grace` and `carry_damping`.
3. **Hand-only fallback** — when carry-forward expires (or no body was ever seen), export a row with blank arm columns and hand data assigned left/right by x-coordinate.

## CSV column counts

| Mode | Body columns | Hand columns | Metadata | Total |
|------|--------------|--------------|----------|-------|
| `hands` | — | 2 × 21 × 3 = 126 | 4 | 130 |
| `hands-arms` | 12 × 4 = 48 | 126 | 4 | 178 |
| `body` | 33 × 4 = 132 | 126 | 4 | 262 |

Body keypoints export `x, y, z, visibility`. Hand keypoints export `x, y, z` only. Missing hand data is left blank. With `--single-subject`, body columns may be blank on hand-only fallback frames.

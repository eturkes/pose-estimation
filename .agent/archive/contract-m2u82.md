# Contract — M2.8.2, full corpus 2D run

Frozen at base `e97d417`. Tier: spine artifact = `data`; **D01's shipped-code fix carries `kernel`
assurance** (diff-blind red suite + MAIN rerun), because it is judgment-bearing code in a shipped
module. Planning sized this unit as delivery; the unit's own first question turned it into a defect
fix plus delivery, and §10 records that as the sizing datum.

## 1. Spine

379 canonical assets → per-asset 2D clinical features + a run manifest giving **every** asset an
explicit disposition. Resumable. Cohort aggregation is M2.8.3 and is out of scope here.

## 2. The unit's first question, closed: the 40× per-asset split is one defect, not a cost profile

M2.8.1 measured per-asset cost bimodal at 40× with equal yield — 6/16 assets at 7.2-12.2 ms/frame,
10/16 at 338.6-543.5 — and named per-detected-box cost as the leading candidate. **Both candidates
are refuted. Both bands are the same upstream state-machine defect in `rtmlib.PoseTracker`.**

`PoseTracker.__call__` reorders the current frame's keypoints by *persistent track id*:

```
self.track_ids_last_frame = track_ids_current_frame
try:
    keypoints = np.array([keypoints[i] for i in self.track_ids_last_frame])   # id used as index
    scores    = np.array([scores[i]    for i in self.track_ids_last_frame])
except:                       # noqa — upstream bare except
    return keypoints, scores,          # returns BEFORE the two state updates below
self.bboxes_last_frame = bboxes_current_frame
self.frame_cnt += 1
```

`track_by_iou` assigns `track_id = self.next_id; self.next_id += 1` on any unmatched box with
`area >= MIN_AREA`. One missed IoU match therefore mints an id ≥ 1, and `keypoints[1]` on a
one-person frame raises `IndexError` → the early return fires → **`frame_cnt` and
`bboxes_last_frame` freeze for the remainder of the source**. Every later frame retries the same bad
index, so the freeze is permanent, and the returned pre-reorder keypoints keep yield near 1.0, which
is why nothing downstream looked broken.

**The residue of the frozen counter selects the band**, `det_frequency = 7`:

| freeze residue | detector cadence after freeze | bbox list | measured band |
| -------------- | ----------------------------- | --------- | ------------- |
| `frame_cnt % 7 == 0` | **every frame** | fresh, correct | 338.6-543.5 ms/frame — correct output, ~6× the necessary cost |
| `frame_cnt % 7 != 0` | **never again** | drains to empty via `track_by_iou`'s pops | 7.2-12.2 ms/frame — **corrupted output** |

The low band is not fast, it is wrong. `RTMPose.__call__` opens with
`if len(bboxes) == 0: bboxes = [[0, 0, image.shape[1], image.shape[0]]]`, so a starved box list makes
a top-down pose model estimate from the **whole 1080p frame** instead of a person crop, at
confident-looking scores.

**Synthetic reproduction, stub det/pose models, 140 frames, `det_frequency = 7`**, one induced IoU
miss whose frame index selects the residue:

| scenario | `frame_cnt` | detector calls | whole-frame pose calls | modelled ms/frame |
| -------- | ----------- | -------------- | ---------------------- | ----------------- |
| no miss, `tracking=True` | 140 | 20 | 0 | 58.0 |
| miss → residue ≠ 0 | **3 (frozen)** | 1 | **135/140** | 10.5 |
| miss → residue 0 | **7 (frozen)** | 134 | 0 | 343.0 |
| miss, `tracking=False` | 140 | 20 | 0 | 58.0 |

Cost model = detector 350 ms on detector frames + pose 8 ms/frame. It reproduces both measured bands
from the same single defect and shows the fix immunising both.

## 3. Design decisions

- **D01 — rtmlib IoU tracking is disabled: `PoseTracker(..., tracking=False)`.** The stateless branch
  (`not self.tracking and self.det_frequency != 1`) skips the reorder entirely, so no id is ever used
  as an index and `frame_cnt` always advances. Redundancy is what makes this safe rather than a
  capability loss: `KeypointSmoother` already owns temporal association through Hungarian
  `gated_assignment` (`src/pose_estimation/smoothing.py`), and rtmlib's tracker contributed only the
  IoU drop of unmatched people, which `--single-subject` overrides by taking the confidence argmax.
  Not patching upstream's reorder: the project needs no track ids from rtmlib, so removing the
  dependency beats forking a routine whose output is discarded.
- **D01a — `--det-frequency 1` stays legal and needs no guard.** It re-enters the tracking branch, but
  a freeze there is harmless: `frame_cnt % 1 == 0` holds at every residue, so the detector runs every
  frame and the box list is never starved. The only lost effect is the reorder, which D01 discards
  anyway. P04 pins the claim rather than asserting it.
- **D02 — corpus-run sizing is MEASURED post-fix, never projected.** M2.8.1's 26-31 h and the plan's
  6.5 h both described a run that no longer exists: the first measured a frozen tracker, the second
  extrapolated per-call latency. The post-fix pilot re-run is the only admissible input to the corpus
  estimate, and the estimate is labelled `measured` with its sample.
- **D03 — the detector stays on CPU for the corpus run.** GPU detection is 22× faster on the synthetic
  probe (9.7 ms vs 213 ms median) but pads its dynamic in-graph-NMS output to a fixed 100-row buffer;
  CLAUDE.local.md requires a per-device shape-and-range qualification before any dynamic-output model
  is trusted, and GPU zero-fill has not been qualified against the live detector post-processing. A
  corpus run is the wrong place to qualify a device. Deferred to `.agent/polish.md` with its
  acceptance check.
- **D04 — resumability grain = the event (session), not the asset.** `process_session` runs every
  camera of a session in one call, so an asset cannot be run alone (M2.8.1's ruling, unchanged).
- **D05 — resume is keyed on a per-event completion marker, never on output presence.** A killed run
  leaves a partial landmark CSV that is indistinguishable from a complete one by existence or by row
  count, because the true row count is not known until the source is fully decoded. The marker is
  written after the event's outputs are complete.
- **D06 — the manifest is total over the registry's canonical assets and its disposition vocabulary is
  frozen.** Every asset reaches exactly one disposition; a new failure mode earns a code rather than
  an absent row. This is M2.8.1's partition discipline applied at the corpus grain: an asset silently
  missing from a denominator is the defect the artifact exists to prevent.
- **D07 — outputs are patient-adjacent.** The run tree and every landmark CSV live outside git, and
  every report obeys M2.8.1's allowlist redaction: emitted strings are published stratum labels, R
  reason codes, or frozen disposition codes; keys read as code-authored field names.

## 4. Predicates

| id | predicate |
| -- | --------- |
| P01 | `run.py` constructs `PoseTracker` with `tracking=False`. |
| P02 | Characterization of the upstream defect: with `tracking=True`, one induced IoU miss freezes `frame_cnt` and `bboxes_last_frame`; with `tracking=False` on the identical stimulus both advance for all frames. |
| P03 | Under the shipped construction no frame reaches the pose model with an empty bbox list, so no whole-frame pose call occurs on a source that detects a person. |
| P04 | Under the shipped construction the detector call count equals `ceil(frames / det_frequency)`; under `det_frequency = 1` it equals `frames` whether or not a miss occurs (D01a). |
| P05 | Resume is idempotent: a second driver run over a tree whose events all completed performs zero inference and leaves every output byte-identical. |
| P06 | Resume does not credit partial work: an event interrupted before its marker is re-run from scratch on the next pass. |
| P07 | The manifest carries exactly one row per canonical registry asset — 379 rows, no duplicate, no absent asset. |
| P08 | Manifest dispositions partition the asset set: every row's code is in the frozen set, and the per-code counts sum to the row count. |
| P09 | Every `ok` asset has a landmark CSV and exactly one source-diagnostics row; no non-`ok` asset claims either. |
| P10 | Every landmark CSV that reaches the R feature stage yields a group-disposition artifact, header-only or populated (M2.8.1 D05 at corpus grain). |
| P11 | The run writes nothing inside the published session tree and leaves `sessions` `tree_digest` unmoved end to end. |
| P12 | Per-asset CFR counters (`pts_accepted`, `index_fallback`, `monotonic_forced`) are recorded for every `ok` asset and the pooled fallback rate is published. |
| P13 | Report redaction is an allowlist: every emitted string is a published stratum label, an R reason code or a frozen disposition code, and every key matches `[a-z][a-z0-9_]*`. |
| P14 | The corpus-run cost is recorded as `measured` with its measuring sample named, and no projected figure is carried as a budget. |

## 5. Invariant surfaces

`src/pose_estimation/run.py` (tracker construction, frame loop), `src/pose_estimation/multicam.py`
(`process_session`, `process_source`), `src/pose_estimation/video_io.py` (`SourceTimestampClock`
counters), `analysis/clinical_features.R` (group disposition), the published `sessions/` tree
(read-only to this unit).

## 6. Gate identity

`env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync` × {`ruff check`,
`ruff format --check`, `ty check`, `pytest`}, primary tree, zero skipped.

## 7. Negative-control seed

| id | seed | must fire |
| -- | ---- | --------- |
| N1 | revert `tracking=False` | P01, P02, P03 |
| N2 | force `det_frequency = 1` with tracking restored | P04 |
| N3 | delete one event's completion marker after a full run | P05 |
| N4 | truncate a landmark CSV and leave the marker | P06 detects nothing — **ruled out of scope**: the marker is the contract, and byte-verification of a patient-adjacent CSV is M2.8.3's consumer problem |
| N5 | drop one asset row from the manifest | P07 |
| N6 | spell an unlisted disposition code | P08 |
| N7 | emit a capture id in the report | P13 |

## 8. Amendments

*(none yet)*

## 9. Verdict table

*(filled at unit close)*

## 10. Sizing

Planning sized M2.8.2 at 1-2 windows as delivery over a working pipeline. The pipeline was not
working. **The unit's named first question was the whole unit**, exactly as M2.6's was, and the
answer arrived from evidence M2.8.1 had already committed — the pilot's per-frame latency logs,
re-read at the detector cadence, separated detector frames from pose-only frames and made the
freeze visible without decoding one new frame. Datum: **before funding a measurement, re-read the
logs the last measurement already wrote.**

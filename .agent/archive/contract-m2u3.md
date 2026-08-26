# M2.3 acceptance contract — capture qualification + 3D-route ruling

Tier `kernel`. MAIN-authored. Every downstream artifact decides against this file. Evidence pointers
are `main:F<n>` (`.scratch/agents/main-checkpoint-m2u3.md`), `audio:U<n>` / `visual:U<n>` (spike
reports), `res1:U<n>` / `res2:U<n>` (research), `map:U<n>` (surface map), `digest:U<n>`.

## 1. What the unit ships

1. `src/pose_estimation/qualify.py` + `pose-estimation-qualify` console script — a third
   artifact-publishing tool, inheriting the M2.1/M2.2 publication contract in full (map:U2).
2. The published evidence set over the real corpus, regenerable byte-identically from committed
   state.
3. **MAIN's ruling** on the 3D route, written to `.agent/archive/rulings-m2u3.md` and summarised in
   `.agent/roadmap.md`; it fixes the shape of M2.5, M2.6 and M2.7, including whether M2.6 exists.
4. `docs/technical/qualification.md` — schema owner for the published set.
5. Gates: a committed determinism sweep (`scripts/check_qualify_determinism.py`) and a red suite
   (`tests/test_qualify.py`). The mutation campaign is **deferred to `.agent/polish.md`**, on the
   precedent of M2.2's own deferral (`contract-m2u2.md` §10) — a `prod`-graded artifact behind a
   determinism sweep plus a red suite is the tier-appropriate bar, and the campaign is a standing
   `gate` track rather than unit work.

Out of scope, by name: production cross-view alignment (M2.5), extrinsics recovery (M2.6), fusion
and the corpus study (M2.7), and any repair of the session tree's `sync_offset` field.

## 2. Artifact schema

`qualification/` — gitignored, patient-adjacent (it carries `capture_id` pseudonyms and per-asset
rows). Publication is per-file atomic, whole-set digest-verified, exactly as `inventory/`.

- `assets_qc.csv` — one row per canonical asset (379):
  `asset_id, capture_id, view, task, side, subject_ordinal, device_config, codec,
   decode_status, frames_decoded, frames_reported, pts_dt_median_s, pts_dt_p95_s, pts_dt_max_s,
   pts_monotonic, orientation_values, orientation_changes, rigidity_stat, rigidity_flag,
   detect_rate, detect_conf_median, subject_px_height_median, scale_ref_class, scale_ref_conf,
   qc_flags`
- `pairs_qc.csv` — one row per unordered within-family asset pair (246):
  `capture_id, asset_a, asset_b, view_a, view_b, offset_s, confidence, peak_ratio, status,
   drift_ppm, drift_se, overlap_s, dur_a, dur_b, same_device_config, same_audio_rate`
- `events_qc.csv` — one row per session event from `sessions/` (193):
  `event_id, capture_id, n_cameras, views, graph_connected, closure_residual_s, offset_span_s,
   sync_qualified, geom_qualified, qualified, reason`
- `qualification.json` — redaction-safe aggregates only, plus a `generation` block digesting the
  three CSVs and itself. **This is the only artifact whose numbers may be quoted anywhere.**

## 3. Predicates

Each is testable and each earns at least one committed test. `P##` is the stable id.

### Publication and identity (LAW inherited from M2.1/M2.2, map:U2)

- **P01** Every consumer calls `qualify.validate_generation(out_dir, sessions_dir=…, inventory_dir=…)`
  before reading a row; the multi-argument form is the only check that catches a qualification set
  rebuilt against a different registry or tree.
- **P02** `generation` carries a digest of each CSV and of `qualification.json` minus its own key; a
  half-published set, an edited CSV and an edited census each fail.
- **P03** The tool reads `inventory/assets.csv` and `sessions/` and **never walks the corpus
  directory itself**; asset paths come from the registry's canonical column (main:F-handoff).
- **P04** Publication replaces a whole tree: `--out` must overlap neither `--corpus` nor
  `--inventory` nor `--sessions` in either direction, and a symlinked `--out` publishes to its
  resolved target.
- **P05** Crash-state ordering: the orphan sweep runs **after** the swap, never before.
- **P06** Every alphabet uses `fullmatch`; every published id matches its declared alphabet; every
  integer cell is ASCII `[0-9]+`.
- **P07** A zero-row CSV still validates its header, and a short header fails rather than publishing
  an empty artifact.
- **P08** The published set is a function of corpus bytes alone — identical under a changed locale,
  `PYTHONHASHSEED`, timezone, `umask`, `iterdir` order, `--out` name and `-O`.
- **P09** `qualification.json` holds no filename, no path, no subject-directory name and no GPS
  value.

### Timebase and decode (main:F1, F4)

- **P10** Every per-frame time comes from `PTS × time_base`, never from `frame_index / fps`. A test
  pins that a file whose PTS are non-uniform yields non-uniform `pts_dt`.
- **P11** `frames_decoded` is compared against `reported_frame_count`; a mismatch sets a `qc_flag`
  and never silently truncates.
- **P12** `orientation_values` records **every** distinct `com.apple.quicktime.video-orientation`
  value in the timed track, and `orientation_changes` counts transitions. The 7 assets with
  mid-clip changes (main:F4) carry a `qc_flag` and are excluded from any per-clip geometry claim,
  because the container's single display matrix — the value cv2 applies — cannot express them.
- **P13** Assets with no orientation track (3 of 379) are flagged, not assumed upright.

### Synchronization (audio:U1–U6, visual:U1–U6, main:F8)

- **P14** The offset estimator is **audio-first**: measured acceptance 210/246 pairs, confidence
  ROC AUC 0.96083, 2 false positives per 100 held-out controls, full-corpus cold run 8.256 s
  (audio:U6). The visual estimator is retained as a **corroborator**, not a fallback: 67/246
  acceptance, AUC 0.761 (visual:U6).
- **P15** `confidence` separates within-family pairs from cross-family controls; the committed
  threshold is the one whose held-out control false-positive rate is ≤ 2/100.
- **P16** `closure_residual_s` is published per three-camera event and is **labelled a
  self-consistency statistic, never an accuracy statistic**. Acoustic propagation delay is an exact
  cocycle around a triangle, so closure is blind to it by construction (main:F8). Any document that
  cites closure as evidence of sync accuracy is in breach of this predicate.
- **P17** The accuracy statement rests on the cross-modality agreement instead: two estimators
  sharing no code and no signal agree to **median 10.86 ms, 88.3% within one 33.4 ms frame** on the
  60 pairs both accept (main, `join_spikes.py`). The published claim must quote that number with its
  n and its subset definition.
- **P18** No rate/drift term is modelled. Measured: 0/132 qualified audio drifts move alignment by
  more than one frame over the pair's overlap (audio:U5); visual agrees, 0/15 at the 95% lower bound
  (visual:U5). Independent prior 16–31 ppm ⇒ 4.5–8.5 ms over the 274.8 s maximum clip (res2:Q2).
  A single constant offset per pair is sufficient **for this corpus** and the artifact says so.
- **P19** `sync_qualified` is true for an event only when its camera graph is connected by accepted
  pairs. Measured connectivity: 122/137 multi-view families (89.05%, audio:U6).
- **P20** The two `view_conflict` families stay `take_resolution = "unresolved"`. Neither estimator
  places their same-view pairs together (0.589 s and 5.602 s cross-modality disagreement, both with
  at least one estimator abstaining). Overturning this needs a wave-2 measurement, not an argument.

### Geometry, rigidity, detectability, scale — wave-2 evidence

- **P21** `rigidity_stat` is an image-space background-drift statistic in pixels with a stated
  sampling rule. The accept gate is `median drift ≤ 2 px` and `p95 ≤ 4 px` over the clip
  (res1:U6); an asset outside it is flagged, not dropped. The visual spike's provisional scalar
  flags 28/379 (visual:U6) and is a candidate implementation, not the definition.
- **P22** `view_label_agrees_with_geometry` is measured, not assumed. View tokens are lexical
  (roadmap standing constraint) and main:F3 shows the `above` and `left` labels were served by
  **different physical tablets in two eras**, so agreement must be established per era.
- **P23** `detect_rate` and `detect_conf_median` come from the repo's own pose pipeline on a stated
  frame sample, with the detector on a device whose output is not padded with uninitialised memory
  (`CLAUDE.local.md`; NPU-YOLOX is excluded by measurement).
- **P24** `scale_ref_class` records what metric reference, if any, is visible. Absent any reference,
  the artifact states arbitrary scale explicitly: angles and dimensionless ratios survive, every
  metre-valued distance, velocity and jerk does not (roadmap claim boundary).
- **P25** Intrinsics carry no metadata provenance anywhere in this corpus (main:F2). The only
  available priors are per device model — `iPad (5th generation)` fx ≈ 1873.3 px from a 54.267°
  horizontal field of view, `iPad Air 11-inch (M2)` fx ≈ 1553.2 px from a 3 mm/28 mm nominal lens,
  both with a 4:3→16:9 crop factor 1.08947× and an **unreported** readout/stabilisation factor
  (res1:U2). Every intrinsics value the artifact publishes is labelled `prior`, never `measured`.

### Claim boundary (roadmap, binding)

- **P26** No artifact and no document produced by this unit claims clinical validity, absolute
  metric accuracy or marker-based equivalence.
- **P27** The rolling-shutter contribution is stated wherever a timing claim is made. Neither iPad
  model's readout time is published, so the artifact carries a **sweep, not a value**: 0–33.33 ms,
  with Apple-mobile 1080p line-scan evidence of 12.4–30.9 ms (37–93% of one 30 Hz frame period)
  named explicitly as a proxy, not as a measurement of these devices (res2:Q7). Rolling shutter is
  not removed by synchronisation, and no document may call it negligible.
- **P28** AAC encoder priming is a fixed, rate-dependent bias and the artifact accounts for it:
  2112 samples = 47.891 ms at 44 100 Hz and 44.000 ms at 48 000 Hz, so a raw untrimmed mixed-rate
  pair carries a fixed **3.891 ms** bias and a one-sided trim carries 44.000 or 47.891 ms
  (res2:Q1). 55 of 137 multi-view families mix the two rates (main:F3). `pairs_qc.csv` records
  `same_audio_rate` so the bias stratum is visible, and the estimator's priming handling is stated
  in `docs/technical/qualification.md`.
- **P29** Sync QC is stratified by `(model, OS, sample_rate)`, because exact iPad input-to-timestamp
  latency is unbenchmarked and a per-configuration constant is the only way an unmodelled device
  latency shows up as structure rather than as noise (res2:Q1).

## 4. Invariant surfaces

- `inventory/` and `sessions/` are read-only inputs. This unit republishes neither.
- Legacy 2D and 3D producer schemas stay unwidened (`analysis/utils.R:59-87` treats every numeric
  non-metadata column as a feature).
- `capture_id` never names a recording event; `event_id` = `{capture_id}_run-{run_index:02d}`.
- No new runtime dependency beyond `av`, already added and gate-verified (844 passed, 0 skipped).

## 5. Gate identity

Decisive gate, primary tree, `PYTHONPATH="$PWD/src"` mandatory in both trees:

```sh
PYTHONPATH="$PWD/src" uv run --no-sync ruff check \
  && uv run --no-sync ruff format --check \
  && PYTHONPATH="$PWD/src" uv run --no-sync ty check \
  && PYTHONPATH="$PWD/src" uv run --no-sync pytest
```

Baseline to beat: 844 passed, 0 skipped. Plus `scripts/check_qualify_determinism.py` green over its
declared sweep set, streaming to `tests/qualify_determinism_results.json` and refusing to append to
a file measured against different source digests.

## 6. Probe-corpus seed

- The 7 mid-clip-orientation assets (main:F4) — geometry must refuse or flag them.
- The 3 assets with no orientation track.
- The 2 `view_conflict` families, one 4-asset family.
- The 51 single-view families — every event-level predicate must survive `n_cameras = 1`.
- The 1280×720 @ 119.971 fps outlier and the 28 portrait assets.
- The 3 quarantined-stem assets — held out, never silently admitted.
- The 55 multi-view families that mix 44 100 and 48 000 Hz audio.
- Both eras of the `above` and `left` view labels (main:F3).

## 7. Open rulings — MAIN decides on wave-2 evidence

- **R1** Does the `above`/`left` label track a position or a device across the era boundary?
- **R2** Is background rigidity sufficient for per-event extrinsics, and on what fraction of events?
- **R3** Does any metric scale reference exist in frame? If none, the user is asked for participant
  anthropometrics — the roadmap requires the survey to precede the request.
- **R4** Does M2.6 exist? Feasible extrinsics on a usable fraction of events is the condition.
- **R5** What sub-frame offset representation replaces the integer `sync_offset`, given that this
  corpus needs no drift term but M2.5 must still express a non-integer offset (res2:Q4)?

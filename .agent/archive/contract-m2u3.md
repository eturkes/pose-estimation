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
   decode_status, pts_source, frames_source, frames_decoded, frames_reported, pts_dt_median_s,
   pts_dt_p95_s, pts_dt_max_s, pts_monotonic, orientation_values, orientation_changes,
   rigidity_drift_median_px, rigidity_drift_p95_px, rigidity_valid_fraction, rigidity_flag,
   detect_rate, detect_conf_median, subject_px_height_median, scale_ref_class, scale_ref_conf,
   qc_flags`
  **Amended** — the frozen text read `rigidity_stat` and carried no measurement-provenance columns.
  R2 split the rigidity statistic into two quantiles plus a valid fraction, and `pts_source` /
  `frames_source` exist because `SourceTimestampClock` substitutes `frame_index / fps` with callers
  unable to tell that from a measurement.
- `pairs_qc.csv` — one row per unordered within-family asset pair (246):
  `capture_id, asset_a, asset_b, view_a, view_b, offset_s, peak_rms, peak_ratio, status_audio,
   offset_visual_s, status_visual, status, drift_ppm, drift_se, overlap_s, dur_a, dur_b,
   same_device_config, same_audio_rate`
  **Amended** — the frozen text read `confidence` and published no corroborator column. R9 removed
  the normalized confidence, so `peak_rms` carries the raw statistic under a name that does not
  assert a quantity the code no longer computes. `status_audio`, `offset_visual_s` and
  `status_visual` join it so that **`status` is a pure function of columns this table publishes**:
  a reader re-derives every fusion verdict from the artifact, and R6 stays re-rulable against the
  bytes it changed rather than against the sidecar alone.
- `events_qc.csv` — one row per session event from `sessions/` (193):
  `event_id, capture_id, n_cameras, views, graph_connected, closure_residual_s, offset_span_s,
   sync_qualified, geom_qualified, qualified, reason`
  Schema unchanged; the cells below the identity block are filled by the sync axis under
  `--measurements` (P19, P19b, P19c) and stay empty without it.
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
  (audio:U6). The visual estimator is retained as a **corroborator**, not a fallback: **74/246**
  acceptance at its corrected control-optimal gate `f82a9a9` (`corr ≥ 0.72`, `conf ≥ 4`,
  `ratio ≥ 1.10`), 0/200 held-out controls, 26/137 families, closure on 9/52 families with |r|
  median 8.08 ms and max 34.12 ms. **Amended** — the frozen text read 67/246 with AUC 0.761, which
  is the superseded pre-correction gate; that AUC was never re-derived against `f82a9a9` and must
  not be quoted for the corrected gate.
- **P15** `confidence` separates within-family pairs from cross-family controls; the committed
  threshold is the one whose held-out control false-positive rate is ≤ 2/100.
- **P16** `closure_residual_s` is published per three-camera event and is **labelled a
  self-consistency statistic, never an accuracy statistic**. Acoustic propagation delay is an exact
  cocycle around a triangle, so closure is blind to it by construction (main:F8). Any document that
  cites closure as evidence of sync accuracy is in breach of this predicate.
  **Amended** — the artifact's closure population is not the probe's. `events_qc.csv` groups by
  **event** and accepts on the **fused** verdict: 30 triangles, median 5.403 ms, max 30.286 ms.
  `scripts/probe_sync_policy.py` groups by **capture family** and accepts on **audio alone**: 35
  triangles, median 4.451 ms, max 30.286 ms — the P38 figure, and it still reruns exactly. Both are
  correct for their own population and neither may be quoted for the other. A three-camera event
  whose three pairs are not all accepted publishes an empty cell: a path joins three cameras without
  closing them, and a partial residual would be a number for a triangle that was never measured.
- **P17** The accuracy statement rests on the cross-modality agreement instead: two estimators
  sharing no code and no signal agree to **median 12.89 ms, 86.2% within one 33.4 ms frame, 41.5%
  under 10 ms** on the **65** pairs both accept (p75 23.10 ms, p95 50.72 ms, max 74.8 ms; main,
  `join_spikes.py` against the corrected visual gate `f82a9a9`). The published claim must quote that
  number with its n and its subset definition. **Amended** — the frozen text read 10.86 ms / 88.3% /
  n=60, computed against the visual spike's superseded 67/246 gate; that basis is labelled
  superseded, not deleted, and the conclusion is unchanged.
- **P17b** The corroborator's control result does not bound its gross-error rate. Of the 9 pairs the
  visual estimator accepts and audio rejects, one disagrees by **87.4 s**, despite the visual gate
  scoring 0/200 on held-out controls. ~~The artifact therefore qualifies a pair on **agreement
  between the two estimators**, never on either alone, and any pair accepted by exactly one carries a
  `qc_flag`.~~
  **Amended, and the struck sentence is superseded rather than merely rephrased.** R6 priced the
  strict reading on measurement and **refused** it: agreement-required leaves 111 of 137 families
  unrecoverable — 96 more than audio alone — and 2 closing triangles, ending M2.5 and M2.6 for most
  of the corpus, while buying nothing the veto does not. The gross-error evidence bounds the
  **visual** estimator alone; no comparable evidence exists against audio (2/100 control false
  positives, 35/35 accepted triangles closing under one frame, 12.9 ms median agreement wherever the
  second instrument speaks). The shipped policy is R6's: **audio estimates; the corroborator holds a
  veto where it cleared its own gate, and no vote where it did not.** Published: 201/246 qualified
  (56 `ok_corroborated`, 145 `ok_uncorroborated`, 9 `contradicted`, 9 `visual_only`, 27
  `neither_accepted`).
  The `qc_flag` clause is superseded too, on a second ground: `pairs_qc.csv` carries no flag column,
  and the six-token `status` alphabet already names "accepted by exactly one" more precisely than a
  flag could — `ok_uncorroborated` and `visual_only` say which one, and in which direction.
  The frozen sentence stays struck-through and visible: a contract that silently acquires the
  ruling's text loses the record of what was priced and refused.
- **P18** No rate/drift term is modelled. Measured: 0/132 qualified audio drifts move alignment by
  more than one frame over the pair's overlap (audio:U5); visual agrees, 0/15 at the 95% lower bound
  (visual:U5). Independent prior 16–31 ppm ⇒ 4.5–8.5 ms over the 274.8 s maximum clip (res2:Q2).
  A single constant offset per pair is sufficient **for this corpus** and the artifact says so.
- **P19** `sync_qualified` is true for an event only when its camera graph is connected by accepted
  pairs. Measured connectivity: 122/137 multi-view families (89.05%, audio:U6).
  **Amended** — the predicate quantifies over the **event's own cameras**, and event membership comes
  from the session tree's `placements.csv`, never from the capture family: a view-conflict family
  resolves to several single-camera events, so a family-wide member list credits each of them with
  cameras it does not hold. Published: **173/193 events sync-qualified** (58/58 one-camera, 74/84
  two-camera, 41/51 three-camera), under R6's fused acceptance. A one-camera event is connected —
  it carries no alignment that can fail, and geometry is the axis that refuses a single camera.
  The 122/137 figure stays the **family**-grain statistic under audio-alone acceptance; the same
  probe under the shipped fused policy reads 117/137.
- **P19b** Each camera's offset is solved breadth-first from the event's lowest asset id over
  accepted edges alone, so a camera one accepted pair reaches directly keeps that measured offset
  rather than an accumulated path. Where a triangle does not close, the route decides the answer, so
  the traversal is fixed rather than incidental and `closure_residual_s` publishes exactly the
  disagreement between the two routes. `offset_span_s` is the spread of that solution and is
  published only for a connected event of two or more cameras.
- **P19c** `views` is copied from the session tree's own per-event cell rather than re-derived from
  the capture family. Re-deriving it published `above|left` on seven single-camera events of the two
  view-conflict families — cameras those events do not hold.
- **P20** The two `view_conflict` families stay `take_resolution = "unresolved"`. Neither estimator
  places their same-view pairs together (0.589 s and 5.602 s cross-modality disagreement, both with
  at least one estimator abstaining). Overturning this needs a wave-2 measurement, not an argument.

### Geometry, rigidity, detectability, scale — wave-2 evidence

- **P21** `rigidity_stat` is an image-space background-drift statistic in **native pixels** with a
  stated sampling rule. The accept gate is `drift_p95_px ≤ 20`, the reprojection tolerance the 3D
  pipeline already applies (`src/pose_estimation/triangulation.py:423-424`); an asset outside it is
  flagged `camera_motion`, not dropped. Measured on the ruled implementation: **280/298 eligible
  assets pass (94.0%), 71/137 multi-asset families keep every member rigid**, 81/379 carrying no
  verdict. **Amended twice.** The second amendment retires the figures **278/286** and **93/379**:
  those were measured with the 4 px instrument this ruling replaced, so the eligible denominator was
  the old instrument's, and decoupling `RANSAC_THRESHOLD_PX = 8.0` recovers 12 support-unmeasurable
  assets. Whether that denominator saturates in the instrument is open, and the sweep answering it
  gates the final number. First: the frozen text read `median ≤ 2 px` and
  `p95 ≤ 4 px` (res1:U6), which flagged 210/286 and was **unadjudicable**: the 4 px accept threshold
  also served as the MAGSAC inlier threshold (`geometry_qualification.py:326`), so `residual_p95`
  could never exceed the gate it was judged against. A threshold sweep over 8× shows `residual_p95`
  tracking the threshold monotonically while inliers grow 6%, so **no gate may be built on
  `residual_p95`**, ratio gates included; `drift_median` moves 5.0% over that range and is the robust
  statistic. `RANSAC_THRESHOLD_PX = 8.0` is now an independent constant and `residual_p95` leaves the
  published schema. Ruling + evidence → `.agent/archive/rulings-m2u3.md` R2.
- **P22** `view_label_agrees_with_geometry` is measured, not assumed. **Measured, and the answer is
  no**: within one configuration the label carries geometry (iPad(5)/16.6 `above` 85% unmeasurable
  against the same tablet's `left` at 6%), but within one label the configuration changes it
  (`above` 85% unmeasurable on iPad(5)/16.6, 3% on Air-M2/26.5). The label named two different
  setups in two eras. `(device_config, view)` is the coarsest key naming a stable geometry, and
  `left`-versus-`right` handedness stays unresolved by anything in this corpus. No per-view prior,
  rotation constant or calibration may cross the era boundary. → rulings R1.
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
- **P28** AAC encoder priming reaches the estimator as a **measured 0 ms residual**, not as the
  predicted bias. The prediction was rate-dependent: 2112 samples = 47.891 ms at 44 100 Hz and
  44.000 ms at 48 000 Hz, so a raw untrimmed mixed-rate pair would carry a fixed **3.891 ms** bias
  (res2:Q1), and 55 of 137 multi-view families mix the two rates (main:F3). The decode path
  cancels it: skipped samples 2112 on 379/379 and first decoded PTS 0 on 379/379, because PyAV
  honours the edit lists (`049684a`). Two independent checks agree that no rate bias survives —
  decoder residual priming bias 0 ms, and a mixed-vs-same-rate residual difference of 54.1 ms
  whose capture-clustered 10k bootstrap 95% CI **[-119.4, +171.1] ms** contains 0 and is two orders
  wider than 3.891 ms. `pairs_qc.csv` still records `same_audio_rate` so the stratum stays visible
  and the cancellation stays falsifiable, and `docs/technical/qualification.md` states the measured
  residual rather than a correction the estimator does not apply.
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

## 6b. Measurement sidecar — the shape ruling A1 fixed

The four expensive axes cannot run inside `qualify.py`'s ~33 s publication: rigidity decodes, the
sync axis decodes audio corpus-wide, detectability runs the pose pipeline. They reach the published
tables through a **validated measurement sidecar** — a record, not a publication.

- **P30** The sidecar lives in `src/pose_estimation/measure/`, one module per axis, and writes one
  directory holding one table per axis plus a `measurements.json` manifest. It inherits **no**
  publication contract: no whole-tree swap, no retiring sibling, no orphan sweep. A torn sidecar is
  caught by digest and repaired by rerun, which is the tier-appropriate bar for an artifact whose
  only consumer re-validates every byte before reading it.
- **P31** The manifest carries, per axis: the table filename, its SHA-256, its row count, its
  generator version, and its **provenance** — every constant, device, sampling rule and analysis
  resolution the measurement depended on. A constant that moves the numbers and does not appear in
  provenance is a defect, because the sidecar's numbers are otherwise unattributable to the code
  that produced them.
- **P32** The manifest digests itself minus its own `generation` key, and records the upstream
  `inventory` generation block. A sidecar measured against a different registry fails ingestion
  rather than publishing rows keyed to assets the registry no longer carries.
- **P33** Axes are independently producible and independently absent. An axis absent from the
  manifest publishes as unmeasured, keeping its named `*_unmeasured` flag. An axis **present** in
  the manifest whose table is missing, unreadable or digest-mismatched is a hard error, and a table
  on disk that no manifest entry names is also a hard error — a stale table can never be read as
  current.
- **P34** `qualify.py --measurements DIR` validates the sidecar before reading a row and records its
  manifest digest in `qualification.json`'s `generation` block as a third upstream. Without the flag
  the tool behaves exactly as it does today, so P08 determinism holds in both modes.
- **P35** Per-asset rows key on the registry's `asset_id`; per-pair rows key on the ordered pair
  with `asset_a < asset_b`, the same order `qualify.py` enumerates. A sidecar key the registry does
  not carry is a hard error; a registry key the sidecar omits publishes unmeasured.

### Sync axis — the port P14–P19 bind

- **P36** `sync_pairs.csv` carries **both** estimators on every enumerated pair, each with its own
  status, unfused: `capture_id, asset_a, asset_b, offset_audio_s, peak_rms_audio, peak_ratio_audio,
  status_audio, drift_ppm, drift_se, offset_visual_s, conf_visual, peak_corr_visual, status_visual,
  overlap_s, dur_a, dur_b, same_audio_rate`. **Amended** — the frozen text read `conf_audio`; R9
  removed the normalized confidence, so the column carries the raw peak RMS under a name that
  asserts only what the estimator computes. Publishing them unfused is what keeps P17b a policy
  `qualify.py` applies rather than a fact baked into the measurement, and it is what lets the
  policy be re-ruled without re-decoding the corpus.
- **P37** The sign convention is published and tested: `offset_audio_s` is `t_B − t_A`, the local
  time of one shared event in B's timeline minus its local time in A's. Positive means B started
  recording earlier. The visual column carries the identical convention, so the two are directly
  differenced.
- **P38** The port reproduces the spike's measured acceptance on the real corpus — **210/246**
  audio-accepted at the committed thresholds, 122/137 families connected. A port that moves those
  counts has changed the estimator, and the change is a finding rather than a refinement.
- **P39** Every threshold the estimator applies is a module constant recorded in the manifest's
  provenance, and **no threshold doubles as an instrument parameter** (rulings R2: `P21`'s gate was
  unadjudicable for exactly that reason). A statistic gated by a constant that also shapes it is
  refused at review.

## 7. Rulings — verdict table at `.agent/archive/rulings-m2u3.md`

| id | question | ruling |
| -- | -------- | ------ |
| R1 | Does `above`/`left` track a position or a device across the era boundary? | **Neither.** The label named two setups in two eras; no per-view prior crosses that boundary. |
| R2 | Is background rigidity sufficient for per-event extrinsics, and on what fraction? | **Yes on the measurable population.** P21's gate replaced by `drift_p95 ≤ 20 px`; 280/298 assets, 71/137 families. 81/379 assets carry no verdict under any gate. The retired 278/286 and 93/379 were the 4 px instrument's population. |
| R3 | Does any metric scale reference exist in frame? | **Closed negative.** A stratified 52-asset survey found no exact dimensional identity in any cell; the axis stays unproduced and every asset keeps `scale_unmeasured`. |
| R4 | Does M2.6 exist? | **Yes, route re-specified.** Scene-feature extrinsics eliminated by measurement (0/246 pairs recoverable); subject-keypoint calibration is the route. |
| R5 | What replaces the integer `sync_offset`? | One float `offset_s` per camera against the event reference. No rate or drift term. |
| A1 | How do the expensive axes reach `assets_qc.csv`? | A validated **sidecar generation**, ingested by `qualify.py` as a third upstream. |

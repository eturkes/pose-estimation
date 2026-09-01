# Capture qualification

`pose-estimation-qualify` publishes the capture-qualification evidence set. The set records what the
recordings can support. It decides which recording events are eligible for 3D work.

The tool is the third artifact publisher, after `pose-estimation-inventory` and
`pose-estimation-sessions`. It follows the same publication rules.

## Run the tool

```sh
pose-estimation-qualify \
  --inventory inventory \
  --sessions sessions \
  --corpus videos/3-cam \
  --out qualification \
  --measurements measurements
```

The command exits 0 on success. It exits 2 when it refuses to publish. A refusal prints one message
and no path.

Omit `--measurements` to publish the expensive axes unmeasured. Both modes publish the same five
files with the same columns. The flag adds one key to the generation block, and it fills the cells
that the sidecar supplies.

The column set never depends on the flag. A schema change instead moves `generator_version`, and
`validate_generation` then refuses a set that an older generator published.

## Inputs

The tool reads four inputs. It writes none of them.

| Input | Purpose |
| ----- | ------- |
| `--inventory` | The registry. It supplies every asset row and every canonical source path. |
| `--sessions` | The session tree. It supplies the recording events. |
| `--corpus` | The recordings. The tool opens only files the registry names. |
| `--measurements` | The measurement sidecar. It supplies the expensive axes. Optional. |

The tool never walks the corpus directory. Every path comes from the registry `source_path` column.
An asset that the registry does not list cannot enter the evidence set.

The tool validates every upstream generation before it reads a row. A rebuilt registry, a rebuilt
session tree, or an altered sidecar stops the run.

## Measurement sidecar

The sidecar records the axes that cost a decode. See `src/pose_estimation/measure/`. The tool
ingests it; the tool never produces it.

If you give `--measurements`, the tool asserts that the sidecar is there. A directory that is
missing, unreadable, or without a manifest stops the run. The tool does not fall back to unmeasured
axes, because that turns an operator mistake into a silent publication.

The tool validates the sidecar before it decodes one frame. It then binds every sidecar key to the
registry:

- A key that the registry does not carry stops the run.
- A row that names a different capture family than its own key stops the run.
- A registry key that the sidecar omits publishes as unmeasured.

The two directions differ on purpose. A row keyed to an asset that does not exist is provenance for
nothing. An omitted row is the normal state of an axis that is still in production.

`qualification.json` records the sidecar manifest digest as a third upstream, under
`generation.measurements`. That key is present only in the runs that used the flag.

## Outputs

The tool publishes five files into `--out`. The directory is patient-adjacent. Never commit it.
The schema is `GENERATOR_VERSION` v4.

| File | Grain |
| ---- | ----- |
| `assets_qc.csv` | One row for each canonical asset. |
| `pairs_qc.csv` | One row for each asset pair inside a capture family. |
| `cameras_qc.csv` | One row for each placed camera inside a recording event. |
| `events_qc.csv` | One row for each recording event. |
| `qualification.json` | Aggregates, and the generation marker. |

**Related calibration ruling.** The separate `calibration_qc/` set publishes the corpus-level
geometry-recovery ruling. It is not an output or an upstream of `pose-estimation-qualify`. It leaves
every per-event geometry cell unmeasured. Read [Calibration ruling](calibration_qc.md) before you
consume that set.

`qualification.json` holds counts and distributions only. It holds no filename, no path, no
identifier and no location. It is the only artifact you may quote outside the published set.

## Measurement provenance

Each timing cell names the source that produced it. Read the source column before you trust a
number.

| Column | Value | Meaning |
| ------ | ----- | ------- |
| `pts_source` | `container_pts_x_time_base` | The time comes from the container timestamps. |
| `frames_source` | `demuxed_packet_count` | The count comes from the demuxed packets. |

This tool never computes a time from a frame index and a nominal frame rate. That substitution
produces a plausible number that no column can distinguish from a measurement.

The decode clock in `src/pose_estimation/video_io.py` does make that substitution. It is a monotonic
processing clock for the pose pipeline. Do not read its timestamps as container timestamps.

## Unmeasured axes

An axis that has not run publishes an empty cell. It also publishes a named flag in `qc_flags`.

An empty cell alone cannot separate two different states. The first state is a check that ran and
found nothing. The second state is a check that has not run. The flag separates them.

`qualification.json` lists both `measured_axes` and `unmeasured_axes`. Read those two lists first.

The tool measures the timebase and orientation axes itself, on every run. It ingests the rigidity,
detect, scale and sync axes from the sidecar. An axis that the sidecar manifest does not name stays
unmeasured.

The sync axis reaches three tables. `pairs_qc.csv` carries the per-pair measurement + fused
verdict; `cameras_qc.csv` carries per-camera alignment + reachability; `events_qc.csv` carries
per-event connectivity + verdict. Without `--measurements`, pair rows carry `status = unmeasured`,
camera rows carry `offset_status = unmeasured`, and event sync cells stay empty.

An axis that the manifest names with zero rows counts as measured. Its producer completed and found
nothing to record. Coverage of one asset is a different question, and `qc_flags` answers it.

Current state: the sync axis is produced, over all 246 within-family pairs. It qualifies 201 pairs
and 173 of the 193 events. The rigidity, detect and scale axes are not yet produced, so no event is
`qualified` yet.

## `assets_qc.csv`

| Column | Meaning |
| ------ | ------- |
| `asset_id`, `capture_id`, `view`, `task`, `side`, `subject_ordinal` | Registry identity. |
| `device_config` | The camera model and the operating system version, as `model/software`. |
| `audio_rate_hz` | The exact audio sample rate, in Hz, from the container header. |
| `codec` | The video codec name. |
| `decode_status` | `ok`, `open_failed`, `no_video_stream` or `no_pts`. |
| `pts_source`, `frames_source` | Measurement provenance. See above. |
| `frames_decoded` | The demuxed packet count. |
| `frames_reported` | The frame count the registry published. |
| `pts_dt_median_s`, `pts_dt_p95_s`, `pts_dt_max_s` | Presentation intervals, in seconds. |
| `pts_monotonic` | `1` when the packets demux in presentation order. |
| `orientation_values` | The distinct device-orientation codes, pipe-separated and ascending. |
| `orientation_changes` | The number of orientation transitions in the track. |
| `rigidity_drift_median_px`, `rigidity_drift_p95_px`, `rigidity_valid_fraction`, `rigidity_flag` | Background stability, as image-space drift from one reference frame. Two quantiles, never one summary statistic: a single residual figure tracks whichever threshold judges it. From the sidecar. |
| `detect_rate`, `detect_conf_median`, `subject_px_height_median` | Detection. From the sidecar. |
| `scale_ref_class`, `scale_ref_conf` | Metric scale reference. From the sidecar. |
| `qc_flags` | Pipe-separated flags. |

### Metric scale

This corpus carries no metric scale reference. A survey of a stratified 52-asset sample found no
exact dimensional identity in any cell. Every fallback is absent rather than imprecise.

That negative comes from a sample, so no asset publishes a measured `none`. Every asset keeps the
`scale_unmeasured` flag and two empty cells instead. Read every 3D output from this corpus as
arbitrary-scale. Angles, timing and dimensionless ratios survive. Distance, velocity and jerk in
metres do not.

The tool sorts the presentation timestamps before it measures an interval. An HEVC stream demuxes
out of presentation order. `pts_monotonic` records whether the sort changed the order.

The tool compares `frames_decoded` against `frames_reported`. A disagreement sets the
`frame_count_mismatch` flag. The tool never truncates either number to make them agree.

## Device orientation

The orientation an asset was shot at is a track, not a header constant. A tablet that turns during
a recording writes a new orientation sample. One rotation constant applied to the whole asset is
then wrong for part of it.

The tool reads the QuickTime timed-metadata track directly. It walks the `moov` atom for each
`mebx` track, collects the `keyd` key declarations, and matches each sample against the key that
ends in `video-orientation`. PyAV supplies the packets. PyAV does not expose the key declarations.

| Flag | Meaning |
| ---- | ------- |
| `orientation_absent` | The asset carries no orientation track. |
| `orientation_changed` | The orientation changes at least once during the asset. |

Read `orientation_changes` before you apply a rotation to a whole asset.

## `pairs_qc.csv`

One row covers one unordered pair of assets inside one capture family. `asset_a` always sorts before
`asset_b`.

A pair that is absent from this table is a pair that no estimator was asked about. That is a
different claim from an estimator that abstained. Enumeration is therefore part of the artifact.

| Column | Meaning |
| ------ | ------- |
| `capture_id`, `asset_a`, `asset_b`, `view_a`, `view_b` | Pair identity. |
| `offset_s` | The audio offset, in seconds. `t_B` minus `t_A` for one shared event. A positive value means that camera B started earlier. |
| `peak_rms`, `peak_ratio` | The audio estimator's raw peak statistics. |
| `status_audio` | The audio estimator's own verdict. |
| `offset_visual_s` | The corroborator's offset, in the same convention. |
| `status_visual` | The corroborator's own verdict. |
| `status` | The fused verdict. See the table below. |
| `drift_ppm`, `drift_se` | The rate fit. No consumer applies a rate term. |
| `overlap_s`, `dur_a`, `dur_b` | The analysed overlap, and each asset's duration. |
| `stratum_a`, `stratum_b` | Each side's `(model, OS, sample_rate)` stratum, as `model/software/rate_hz`. |
| `same_device_config`, `same_audio_rate` | Strata that keep an unmodelled device bias visible. |

Both statistics are raw instrument readings. Never divide a published statistic by its own accept
threshold. A re-ruled threshold then rewrites the statistic while the signal is unchanged.

### Stratification

Input-to-timestamp latency is unbenchmarked for both iPad models. Such a latency is a constant
inside one device configuration. It is noise across a corpus that mixes configurations. Sync
statistics are therefore grouped by the `(model, OS, sample_rate)` tuple.

Each pair carries the stratum of both sides. Both boolean columns beside them are pure functions of
those two cells. Read a stratum as unmeasured when either component is absent. A partial tuple is
not a wider population.

`qualification.json` groups the pairs under `pairs.sync_strata`. The key is the two strata sorted,
so one configuration pair is one population. A pair that is missing either stratum takes the key
`unmeasured`.

Each key holds a record, not a count: `pairs`, `audio_ok`, and `offset_s`. The `offset_s` field is
the same distribution shape as `pts_dt_median_s`, and it is `null` when the group accepted no pair.
The distribution is the part that shows structure. A count alone reports corpus composition and
reports nothing about synchronization.

A group holding one pair publishes that one offset in every distribution field. The census
suppresses no count, and an offset names no asset, no capture, no view, no subject and no time.

Current corpus: 4 configurations, and each one uses a single sample rate. The rate therefore adds
no split here. Publish it anyway. An assumed stratum cannot show when that stops being true.

Manual camera start times dominate the offsets inside a stratum. Do not read a stratum median as a
device latency.

If the run includes a sidecar, its decode rate must equal the header rate. The tool refuses to
publish on any disagreement. Both sides read the first audio stream, so a disagreement means that
the two ran against different bytes.

### Fusion

The audio estimator supplies the offset. The corroborator holds a veto where it cleared its own
gate, and no vote where it did not.

| `status` | Meaning |
| -------- | ------- |
| `ok_corroborated` | Both estimators accepted, and they agree inside one frame. Qualified. |
| `ok_uncorroborated` | Audio accepted, and the corroborator did not speak. Qualified. |
| `contradicted` | Both accepted, and they disagree by more than one frame. Refused. |
| `visual_only` | The corroborator accepted, and audio did not. Refused. |
| `neither_accepted` | Both estimators measured the pair, and neither accepted. Refused. |
| `unmeasured` | No sidecar carried this pair. |

`status` is a function of `status_audio`, `status_visual`, `offset_s` and `offset_visual_s`. All
four are in this table, so you can re-derive every verdict.

Two estimators that accept and disagree are two measurements in conflict. The tool prefers neither.
The `visual_only` stratum is where the corroborator's known gross errors live: one such pair
disagrees with audio by 87 seconds.

### Timing limits

Every offset in this artifact carries two limits. Read both with every timing number here,
including the closure figures below.

**Rolling shutter is not removed by synchronization.** A rolling-shutter camera exposes the top and
the bottom of one frame at different times. An offset aligns the frames. It does not align the
lines inside a frame. Neither iPad model has a published readout time, so this artifact gives a
sweep and not a value: **0 to 33.33 ms**. Apple-mobile 1080p line-scan evidence puts the readout at
**12.4 to 30.9 ms**, which is 37% to 93% of one 30 Hz frame period. That range is a proxy from
other devices. It is not a measurement of these two iPads. No document may call this contribution
negligible.

**AAC priming reaches the estimator as a measured 0 ms residual.** An AAC encoder puts priming
samples in front of the signal. The predicted bias is rate-dependent: 2112 samples is 47.891 ms at
44 100 Hz and 44.000 ms at 48 000 Hz. A raw untrimmed mixed-rate pair therefore carries a fixed
**3.891 ms** bias, and 55 of 137 multi-view families mix the two rates. The decode path cancels it.
PyAV trims the priming samples. The measured skip is 2112 samples on 379 of 379 assets, and the
first decoded presentation timestamp is 0 on 379 of 379 assets. Quote the measured 0 ms residual.
Never quote the 3.891 ms prediction as a bias in this artifact.

## `cameras_qc.csv`

Authoritative time-domain alignment, one row per placed asset of every recording event. Rows sort
by `(event_id, asset_id)`. This table publishes evidence only: the fusion frame reader does not
consume `offset_s`; its applied path still uses integer `session.json:cameras[*].sync_offset`.

| Column | Meaning |
| ------ | ------- |
| `event_id` | Recording-event identity from the session tree. |
| `asset_id` | Placed-asset identity; the row key inside its event. |
| `camera_name` | Camera slot from the session tree. |
| `view` | Semantic view from the registry. |
| `offset_s` | Seconds, fixed nine decimals. Empty when the row carries no offset. |
| `offset_status` | Total status over every published row; meanings below. |
| `is_reference` | `1` or `0` when measured; empty when unmeasured. Exactly one row per measured event carries `1`. |
| `reference_camera` | Camera name every row of a measured event points at; empty when unmeasured. |

| `offset_status` | Meaning |
| --------------- | ------- |
| `reference` | Gauge pin: a definitional zero, not an estimate. |
| `solved` | Unweighted least-squares estimate. |
| `unreachable` | Accepted edges do not join this camera to the event reference. |
| `unmeasured` | The sync axis did not run. |

### Sign + application

One indivisible convention for one shared instant:

- `offset_s = t_camera − t_reference`.
- Positive `offset_s` means that camera started earlier.
- Apply it as `t_ref = t_camera − offset_s`.

Example: `cam-above` is the reference at `0.000000000 s`; `cam-right` publishes
`0.375000000 s`. A shared instant at `t_camera = 5.375000000 s` maps to
`t_ref = 5.000000000 s`; `cam-right` started 375 ms earlier.

### Solver

Accepted edges = `pairs_qc.status` exactly in `{ok_corroborated, ok_uncorroborated}`.
`visual_only` is never usable; estimators are never averaged; the visual estimate is never a
fallback value. Restrict the system to the reference's connected component. Then solve unweighted
`x_b − x_a = offset_s` with `numpy.linalg.lstsq(..., rcond=None)`. Pinning the reference at `0`
fixes the gauge.

### Weighting + path comparison

Unweighted = measured choice, not a default. Spearman correlation of published `peak_rms` against
absolute audio-visual disagreement = **+0.4141**; `peak_ratio` = **+0.0659**. Both have the wrong
sign for a precision weight. Weighting would dress an uncalibrated number as an inverse variance.

Least squares vs a breadth-first tree solve over the 30 events carrying a redundant edge: median
difference = 0; max = **10.095 ms**. **60 of 355 cameras** differ at all; **0** differ by more than
one frame. Where they differ, least squares distributes the closure residual over the evidence
instead of charging it to whichever edge a traversal happened to take.

### Reference + census

Reference = view hierarchy `above` > `left` > `right`, tie-broken by lowest `asset_id`. Corpus total
by reference view = **155 / 24 / 14**. The rule is total over all 193 events.

Corpus = **379 rows over 193 events: 355 carrying an offset, 24 `unreachable`**. The 355 comprise
193 event references at exactly `0.000000000` + 162 solved non-reference cameras. The 24 comprise
10 two-camera unconnected + 6 three-camera with the reference inside the accepted pair + 8
three-camera with the reference isolated. `graph_connected` remains **173/193**.

## `events_qc.csv`

One row covers one recording event from the session tree. `qualified` states whether the event is
eligible for 3D work. `reason` names every axis that blocks it.

| Column | Meaning |
| ------ | ------- |
| `event_id`, `capture_id`, `n_cameras`, `views` | Event identity, copied from the session tree. |
| `graph_connected` | `1` when accepted pairs join every camera of the event. |
| `sync_status` | Immediately follows `graph_connected`: `connected`, `unconnected`, or empty when the sync axis did not run. |
| `closure_residual_s` | The three-camera closure residual, in seconds. |
| `offset_span_s` | The spread of the solved camera offsets, in seconds. |
| `sync_qualified` | `1` when the sync axis accepts the event. |
| `geom_qualified` | `1` when the geometry axis accepts the event. |
| `qualified` | `1` when every axis accepts the event. |
| `reason` | Pipe-separated blockers. |

The tool reads event membership from the session tree `placements.csv`. It never reads membership
from the capture family. A view-conflict family becomes several single-camera events, so a
family-wide member list gives each of those events cameras that it does not hold.

The tool derives `graph_connected`, `sync_status` and `offset_span_s` from the published
`cameras_qc.csv` rows. Any unreachable camera publishes `sync_status = unconnected` beside
`graph_connected = 0`; a partial solve can never read as an aligned event. On that event,
`offset_span_s` covers the reference's reachable component and `sync_status` qualifies its scope.

A one-camera event is connected. It carries no alignment that can fail. The geometry axis is what
refuses a single camera.

### Closure

`closure_residual_s` is a self-consistency statistic. Never read it as an accuracy statistic.
Acoustic propagation delay is an exact cocycle around a triangle, so closure cancels that delay
exactly. A perfect closure and a biased offset are the same number.

The tool publishes closure for a three-camera event whose three pairs are all accepted. It publishes
an empty cell for every other event. A path joins three cameras, but a path cannot close them.

Current corpus: 30 event triangles close, at 5.403 ms median and 30.286 ms maximum. Both figures sit
inside the 0–33.33 ms rolling-shutter sweep that *Timing limits* states. Closure at this scale
therefore does not show camera agreement below one readout.

The policy probe in `scripts/probe_sync_policy.py` reports a different closure population. It groups
by capture family instead of by event, and it accepts on audio alone instead of on the fused verdict.
It reports 35 triangles at 4.451 ms median. Both numbers are correct for their own population. Never
quote one for the other.

## Validate before you read

Call `validate_generation` before you read any row. Pass every upstream the set was published from.
An upstream you do not pass is an upstream this call does not check.

A set published without `--measurements` has two upstreams:

```python
from pose_estimation import qualify

qualify.validate_generation("qualification", sessions_dir="sessions", inventory_dir="inventory")
```

A set published with `--measurements` has three. Add the sidecar:

```python
qualify.validate_generation(
    "qualification",
    sessions_dir="sessions",
    inventory_dir="inventory",
    measurements_dir="measurements",
)
```

The check proves five things:

1. `qualification.json` is this generator's document. It is a regular file, and it holds one
   unambiguous JSON document.
2. Each CSV matches its published digest.
3. The census matches its own digest. The digest covers the generation block, so an edited
   provenance claim fails here.
4. No file was added, removed or changed.
5. Every upstream generation you passed still matches.

The census digest detects an edit. It does not authenticate the set. A set carries no key, so
anyone who rewrites a claim and recomputes the digest produces a document this check accepts. What
the check rules out is corruption, and every edit that stops at the claim.

## Determinism

The published set is a function of the corpus bytes and the upstream generations. The bytes do not
change under a different locale, hash seed, time zone, umask, directory order, output name or
optimized interpreter. `scripts/check_qualify_determinism.py` proves this in both modes.

## Crash safety

The tool builds into a staging sibling. It then renames the old set aside, and renames the staging
set into place.

The orphan sweep runs after the swap. A process that dies between the two renames leaves the only
complete generation under a dead process id. A sweep before the swap would delete it.

The tool refuses an output directory that overlaps any input, in either direction. Publication
replaces the whole output tree.

The tool refuses a non-empty output directory that carries no marker it wrote.

Ownership includes the generator version. A tree from an older version is therefore not this
version's to replace. Delete that tree yourself before the first run of a new version.

## Claim boundary

The evidence set supports no clinical claim. It supports no absolute metric claim. It supports no
claim of equivalence to a marker-based system.

Absent a metric scale reference, 3D output stays at an arbitrary scale. Angles and dimensionless
ratios survive. Distances, velocities and jerks in metres do not.

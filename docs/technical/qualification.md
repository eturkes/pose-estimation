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

Omit `--measurements` to publish the expensive axes unmeasured. Both modes publish the same four
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

The tool publishes four files into `--out`. The directory is patient-adjacent. Never commit it.

| File | Grain |
| ---- | ----- |
| `assets_qc.csv` | One row for each canonical asset. |
| `pairs_qc.csv` | One row for each asset pair inside a capture family. |
| `events_qc.csv` | One row for each recording event. |
| `qualification.json` | Aggregates, and the generation marker. |

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

The sync axis reaches two tables. `pairs_qc.csv` carries the per-pair verdict. `events_qc.csv`
carries the per-event verdict that the accepted pairs support. Without `--measurements`, every
sync cell in both tables stays empty.

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
arbitrary-scale: angles, timing and dimensionless ratios survive; distance, velocity and jerk in
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
| `same_device_config`, `same_audio_rate` | Strata that keep an unmodelled device bias visible. |

Both statistics are raw instrument readings. Never divide a published statistic by the threshold
that accepts it: a re-ruled threshold then rewrites the statistic while the signal is unchanged.

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

## `events_qc.csv`

One row covers one recording event from the session tree. `qualified` states whether the event is
eligible for 3D work. `reason` names every axis that blocks it.

| Column | Meaning |
| ------ | ------- |
| `event_id`, `capture_id`, `n_cameras`, `views` | Event identity, copied from the session tree. |
| `graph_connected` | `1` when accepted pairs join every camera of the event. |
| `closure_residual_s` | The three-camera closure residual, in seconds. |
| `offset_span_s` | The spread of the solved camera offsets, in seconds. |
| `sync_qualified` | `1` when the sync axis accepts the event. |
| `geom_qualified` | `1` when the geometry axis accepts the event. |
| `qualified` | `1` when every axis accepts the event. |
| `reason` | Pipe-separated blockers. |

The tool reads event membership from the session tree `placements.csv`. It never reads membership
from the capture family. A view-conflict family becomes several single-camera events, so a
family-wide member list gives each of those events cameras that it does not hold.

The tool solves one offset for each camera. It starts at the lowest asset id, and it walks accepted
pairs only. A camera that one accepted pair reaches directly keeps that measured offset. The tool
does not accumulate a longer path over a direct measurement.

A one-camera event is connected. It carries no alignment that can fail. The geometry axis is what
refuses a single camera.

### Closure

`closure_residual_s` is a self-consistency statistic. Never read it as an accuracy statistic.
Acoustic propagation delay is an exact cocycle around a triangle, so closure cancels that delay
exactly. A perfect closure and a biased offset are the same number.

The tool publishes closure for a three-camera event whose three pairs are all accepted. It publishes
an empty cell for every other event. A path joins three cameras, but a path cannot close them.

Current corpus: 30 event triangles close, at 5.403 ms median and 30.286 ms maximum.

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

## Claim boundary

The evidence set supports no clinical claim. It supports no absolute metric claim. It supports no
claim of equivalence to a marker-based system.

Absent a metric scale reference, 3D output stays at an arbitrary scale. Angles and dimensionless
ratios survive. Distances, velocities and jerks in metres do not.

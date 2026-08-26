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
  --out qualification
```

The command exits 0 on success. It exits 2 when it refuses to publish. A refusal prints one message
and no path.

## Inputs

The tool reads three inputs. It writes none of them.

| Input | Purpose |
| ----- | ------- |
| `--inventory` | The registry. It supplies every asset row and every canonical source path. |
| `--sessions` | The session tree. It supplies the recording events. |
| `--corpus` | The recordings. The tool opens only files the registry names. |

The tool never walks the corpus directory. Every path comes from the registry `source_path` column.
An asset that the registry does not list cannot enter the evidence set.

The tool validates both upstream generations before it reads a row. A rebuilt registry or a rebuilt
session tree stops the run.

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

Current state: the timebase axis is measured. The orientation, rigidity, detectability, scale and
sync axes are not.

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
| `orientation_values`, `orientation_changes` | The device orientation track. Not yet measured. |
| `rigidity_stat`, `rigidity_flag` | Background stability. Not yet measured. |
| `detect_rate`, `detect_conf_median`, `subject_px_height_median` | Detection. Not yet measured. |
| `scale_ref_class`, `scale_ref_conf` | Metric scale reference. Not yet measured. |
| `qc_flags` | Pipe-separated flags. |

The tool sorts the presentation timestamps before it measures an interval. An HEVC stream demuxes
out of presentation order. `pts_monotonic` records whether the sort changed the order.

The tool compares `frames_decoded` against `frames_reported`. A disagreement sets the
`frame_count_mismatch` flag. The tool never truncates either number to make them agree.

## `pairs_qc.csv`

One row covers one unordered pair of assets inside one capture family. `asset_a` always sorts before
`asset_b`.

A pair that is absent from this table is a pair that no estimator was asked about. That is a
different claim from an estimator that abstained. Enumeration is therefore part of the artifact.

The offset columns are not yet measured. `status` reads `unmeasured` until the sync axis runs.

## `events_qc.csv`

One row covers one recording event from the session tree. `qualified` states whether the event is
eligible for 3D work. `reason` names every axis that blocks it.

## Validate before you read

Call `validate_generation` before you read any row.

```python
from pose_estimation import qualify

qualify.validate_generation("qualification", sessions_dir="sessions", inventory_dir="inventory")
```

The three-argument form is the only form that catches an upstream rebuilt underneath a set that
still looks internally consistent. Always pass all three.

The check proves five things:

1. `qualification.json` is this generator's document.
2. Each CSV matches its published digest.
3. The census matches its own digest.
4. No file was added, removed or changed.
5. Both upstream generations still match.

## Determinism

The published set is a function of the corpus bytes and the two upstream generations. The bytes do
not change under a different locale, hash seed, time zone, umask, directory order, output name or
optimized interpreter.

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

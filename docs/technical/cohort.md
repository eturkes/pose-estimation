# Cohort aggregate

`pose-estimation-cohort` publishes one aggregate over the 2D corpus run.
It reports each clinical feature for each `(task, side)` cell of the task battery.

The tool publishes beside `run/`. It never modifies the run tree, the session tree, or the registry.
It is the sixth artifact publisher, after `inventory`, `sessions`, `qualify`, the `measure` sidecar, and `calibration-qc`.
It runs no pose model and re-derives no per-frame value.

## Run the tool

`$P` = the gate prefix, `$A` = the accelerator recipe — both defined in `entrypoints.md`.

```sh
$P pose-estimation-cohort \
  --inventory inventory \
  --sessions sessions \
  --run run \
  --out cohort \
  --baseline previous-cohort   # optional
```

The tool publishes four files: `cohort_cells.csv`, `cohort_features.csv`, `descriptors.yaml`, and `cohort.json`.
With `--baseline`, it also publishes `cohort_sensitivity.csv`.
`cohort.json` is the ownership marker. It carries the census, the estimand, the column partition, the measurement configuration, and the tree digest.
Consumers must call `cohort.validate_generation(out)` before they read a row.
That call recomputes the digest over the published bytes. It refuses a set that changed after publication.

## Measurement configuration

`cohort.json` `measurement` = the instrument; a model swap moves ~1/5 of cells > 0.5 IQR
(`pose-weak-points.md` § M4 corpus) → the model is part of the estimand.

- `pose_config` = the per-event `pose_config.json` (model, tracker, devices, hand gate, OpenVINO
  build, score calibration); `pose_config_sha256` = its canonical digest; `analysis_sha256` = the
  R producer the completion markers name.
- Read per contributing event, never from `run_report.json`; > 1 configuration or > 1 producer →
  refusal.
- Equal `measurement` blocks → comparable cohorts; unequal → the difference includes an instrument
  change.

## Descriptor collision

`cohort.json` `descriptor_collision` = the `descriptors.yaml` raws checked against `../rehab`
`schema/columns.yaml` (families expanded): `checked`, `source`, `source_sha256` (the bytes the
check read; `null` when the sibling does not resolve), `n_external`, `n_collisions`.
`cohort.recheck_descriptor_collision(out, schema)` reruns the check against a schema file and
refuses one whose bytes differ from `source_sha256` → a `../rehab` schema edit moves
`cohort.json` at the next republish.

## Generation sensitivity

`--baseline <dir>` = a previous generation of this tool (v1-v3): digest recomputed, feature table
re-validated before any read. `<dir>` = `--out` itself only when it holds v3; an older generation
moves to another path first.

`cohort_sensitivity.csv` = one row per `(task, side, level, feature)` in either generation:

- `status` = `common` | `new` (this generation only) | `retired` (baseline only).
- `n_subjects`, `n_values`, `median` + `baseline_` twins.
- `delta_median` = median − baseline median; `baseline_iqr` = baseline q75 − q25;
  `delta_iqr_scaled` = `delta_median` / `baseline_iqr`, blank at IQR <= 0. Exact decimal
  arithmetic on the serialized cells, rendered at 9 places.
- Reading = a change of the published statistic: never patient change, never accuracy; subjects,
  retained observations and feature definitions can differ.
- `cohort.json` `sensitivity.baseline` = version + tree digest + sha256 of the feature bytes read.

## The estimand

The tool aggregates in four stages. Each stage takes a median:

1. **Asset median.** The median of the finite rows of one camera artifact.
   For the bout level, the rows are the asset's uncensored wrist-motion bouts of that side.
2. **Event median.** The median of the asset medians of one event.
3. **Subject median.** The median of the event medians of one subject.
4. **Cohort statistic.** The median, quartiles, mean, and standard deviation over the subject values.

The median at each stage keeps one long recording, one extra camera, or one repeated event from dominating a cell.
Only finite values contribute. The tool discards `NA`, `NaN`, and both infinities before every stage.

A cell publishes its counts always. A cell publishes its distribution only at five subjects or more.
Below that floor, `median`, `q25`, `q75`, `mean`, and `sd` are empty.
A statistic over four subjects identifies its subjects. The counts do not.

## Cross-view agreement

Two or three cameras film each recording event. The agreement columns describe how much those views disagree on the same event.
The statistic is descriptive. It includes projection, observed time span, bout segmentation, and tracking.
It is not the reliability of a person's measurement, and it is not accuracy.

- An event qualifies when two or more of its assets carry a finite asset median. `n_events_multiview` counts these events, and `n_subjects_multiview` counts their subjects.
- `view_abs_diff` is the primary value. For each qualifying event, the tool takes the mean absolute difference over its view pairs. The cell value is the median over the events, in the feature's own unit.
- `view_icc` is a one-way intraclass correlation, ICC(1,1), over the same asset medians. Each event is a target, and each view is a rating. The tool uses the unbalanced view count `k0` and does not clip a negative value. A sample with no variance has no ICC, so the cell is empty.
- `view_icc_range_lo` and `view_icc_range_hi` are the 2.5th and 97.5th percentiles of the ICC over 1000 subject resamples. The range is descriptive. On simulated data with 6 to 16 subjects it held the true value 84 % to 92 % of the time, so it is not a 95 % confidence interval.

Each agreement column publishes only when its qualifying events come from five subjects or more.
`cohort.json` `agreement_reasons` counts the empty rows by reason: below the floor, no multi-view event, no variance, or too few finite resamples.
The tool assigns no category such as "good" or "poor".

## The published column set

The tool reads three levels per asset: frames (`_clinical.csv`), 1 s windows (`_clinical_windows.csv`), and wrist-motion bouts (`_clinical_bouts.csv`).
A column is published when the run carries at least one finite value for it.
Every excluded source column carries one reason:

- `construct_invalid`: the window SPARC and normalized jerk columns (`*_wrist_sal`, `*_wrist_normalized_jerk`, `*_fingertip_normalized_jerk`, with their bilateral forms; 15 columns). These measures are defined on one movement, and a fixed window mixes movement and rest. The bout level replaces them.
- `structurally_absent`: **3** columns, `trunk_lean_sagittal_deg`, `trunk_lean_sagittal_mean`, and `trunk_lean_sagittal_sd`. The upstream R code computes sagittal trunk lean from a depth difference. A single camera measures no depth, so these columns hold no finite value anywhere in the corpus.
- `no_eligible_bout_values`: a bout column with no finite value from an uncensored bout.

The `FEATURES` label table, the retired set, and the absent set name every source column.
An unknown source column is a refusal, and the tool publishes nothing.
A frame or window feature without a finite value is also a refusal.
This makes a new upstream column a loud failure instead of a silent omission.

## Smoothness features

Bout SPARC (`*_bout_sparc`) is the primary smoothness feature.
Bout normalized jerk (`*_bout_normalized_jerk`) is secondary.
Noise distorts log dimensionless jerk even at a high signal-to-noise ratio, and SPARC stays stable at a much lower one.
The descriptor schema has no rank field, so this section states the ranking.
A bout is a detected interval of wrist motion, not a recovered reach or grasp.
Definitions and the detector's qualification → `analysis.md` § *Wrist-motion bouts (M4.5a)*.

## Claim boundaries

Read every published number under these four boundaries.

**The angles are not anatomical angles.**
Each value is an image-plane angle from uncalibrated 2D keypoints.
The pipeline normalizes x and y by one scalar, the larger frame dimension.
That scalar is a similarity map, so the published value is the true image-plane angle.
No step corrects lens distortion, and no step recovers the anatomical angle.
The unit token `deg_image_plane` states this in the data itself.

**Projection geometry limits every value.**
One camera measures a 3D movement through a 2D projection.
A movement toward the camera shortens in the image. A movement out of plane loses magnitude.
Reported distances and velocities are image-plane magnitudes, not true magnitudes; projection can shrink or enlarge them.

**The estimand is directional per-limb, not clinical.**
Each cell reports one limb performing one task in one direction.
The tool makes no cross-cell comparison, no norm, and no clinical judgment.
A difference between two cells may come from the task, the view, or the projection.

**The sample is small.**
The corpus holds few subjects per cell. The tool publishes no minimum, no maximum, and no subject row.
An extreme value at this sample size identifies one person.

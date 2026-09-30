# Cohort aggregate

`pose-estimation-cohort` publishes one aggregate over the 2D corpus run.
It reports each clinical feature for each `(task, side)` cell of the task battery.

The tool publishes beside `run/`. It never modifies the run tree, the session tree, or the registry.
It is the sixth artifact publisher, after `inventory`, `sessions`, `qualify`, the `measure` sidecar, and `calibration-qc`.
It runs no pose model and re-derives no per-frame value.

## Run the tool

```sh
pose-estimation-cohort \
  --inventory inventory \
  --sessions sessions \
  --run run \
  --out cohort
```

The tool publishes four files: `cohort_cells.csv`, `cohort_features.csv`, `descriptors.yaml`, and `cohort.json`.
`cohort.json` is the ownership marker. It carries the census, the estimand, the column partition, and the tree digest.
Consumers must call `cohort.validate_generation(out)` before they read a row.
That call recomputes the digest over the published bytes. It refuses a set that changed after publication.

## The estimand

The tool aggregates in four stages. Each stage takes a median:

1. **Asset median.** The median of the finite rows of one camera artifact.
2. **Event median.** The median of the asset medians of one event.
3. **Subject median.** The median of the event medians of one subject.
4. **Cohort statistic.** The median, quartiles, mean, and standard deviation over the subject values.

The median at each stage keeps one long recording, one extra camera, or one repeated event from dominating a cell.
Only finite values contribute. The tool discards `NA`, `NaN`, and both infinities before every stage.

A cell publishes its counts always. A cell publishes its distribution only at five subjects or more.
Below that floor, `median`, `q25`, `q75`, `mean`, `sd`, and `view_dispersion` are empty.
A statistic over four subjects identifies its subjects. The counts do not.

`view_dispersion` measures how much the two camera views of one event disagree.
For each event with two or more finite asset values, the tool divides the population standard deviation by the absolute mean.
The cell value is the median of those per-event ratios. `n_events_multiview` counts the events that qualified.
An event with a zero mean has no scale, so it leaves both the statistic and the count.

## The published column set

The tool measures the published column set from the run itself.
A column is published when the run carries at least one finite value for it.
Every other source column is excluded with the reason `structurally_absent`.

The run excludes **3** columns: `trunk_lean_sagittal_deg`, `trunk_lean_sagittal_mean`, and `trunk_lean_sagittal_sd`.
The upstream R code computes sagittal trunk lean from a depth difference.
A single camera measures no depth, so these three columns hold no finite value anywhere in the corpus.

The measured set and the `FEATURES` label table must agree by name.
A disagreement in either direction is a refusal, and the tool publishes nothing.
This makes a new upstream column a loud failure instead of a silent omission.

## Smoothness features

SPARC (`*_wrist_sal`) is the primary smoothness feature.
The normalized jerk columns (`*_wrist_normalized_jerk`, `*_fingertip_normalized_jerk`) are secondary.
Noise distorts log dimensionless jerk even at a high signal-to-noise ratio, and SPARC stays stable at a much lower one.
The descriptor schema has no rank field, so this section states the ranking.
Definition and reference → `analysis.md` § *SPARC — the primary smoothness feature*.

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
Reported distances and velocities are therefore lower bounds, not true magnitudes.

**The estimand is directional per-limb, not clinical.**
Each cell reports one limb performing one task in one direction.
The tool makes no cross-cell comparison, no norm, and no clinical judgment.
A difference between two cells may come from the task, the view, or the projection.

**The sample is small.**
The corpus holds few subjects per cell. The tool publishes no minimum, no maximum, and no subject row.
An extreme value at this sample size identifies one person.

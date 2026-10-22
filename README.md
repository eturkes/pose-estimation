# Pose Estimation

This project measures movement in the hospital's three-camera recordings of spinal cord injury
(SCI) rehabilitation tasks. It tracks each patient's body and hands in 2D, computes kinematic
features for every clip, and publishes cohort statistics that the `../rehab` clinical dashboard
reads.

The input is `videos/3-cam/`: 382 hand-held clips of 16 subjects. Each subject does 6 tasks
(cap, coin, glass, key, nut, peg) with each hand. Two or three cameras film each recording, with
no fixed rig and no calibration.

The output contains no patient identifier. The cohort export holds aggregates per task and side,
never a subject row.

> **Working on this with an AI coding agent?** Start at [`CLAUDE.md`](CLAUDE.md). The live
> project state is in [`.agent/spec.md`](.agent/spec.md). Technical references are in
> [`docs/technical/`](docs/technical/).

## What the pipeline produces

| Output | Contents |
|--------|----------|
| `output/corpus-2d/` | One landmark CSV per clip (body, arms, both hands), its diagnostics, and the clinical features at three levels: per frame, per 1 s window, and per wrist-motion bout. `run_report.json` records the model and every setting the run used. |
| `cohort/` | The `../rehab` export: `cohort_cells.csv` (population per task and side), `cohort_features.csv` (median, quartiles, mean, SD and cross-view agreement per feature), `descriptors.yaml` (Japanese and English labels, units, ranges), `cohort.json` (census, estimand, the pose model and settings that produced the numbers, digests) and, when a previous generation is given, `cohort_sensitivity.csv` (how each feature moved against it). |

Each per-clip step runs in this order:

1. A YOLOX detector finds people on every frame. A subject tracker keeps the patient.
2. An RTMW whole-body model places 133 keypoints. `run_report.json` names the model.
3. Smoothing, a hand-presence gate, and out-of-frame zeroing clean the keypoints.
4. The R stage (`analysis/clinical_features.R`) computes angles, distances, speeds, trunk
   posture, and per-bout smoothness (SPARC) and timing.

## Measurement limits

Read every published number under these limits:

- **Angles are image-plane angles.** Each value comes from uncalibrated 2D keypoints. It is not
  an anatomical joint angle. In a sample of 60 recordings, two cameras that film the same moment
  disagree on the elbow angle by a median of 27 degrees.
- **No metric scale.** Distances and speeds use frame-normalized units, not metres. The tool
  publishes `scale_unmeasured` on every clip.
- **No 3D.** The tested calibration methods could not recover the camera positions after the
  recording, so the tool does not fuse the views. A calibrated capture can change this result.
- **A bout is a span of detected wrist motion.** It is not a recovered reach or grasp. A pause
  can divide one action into two bouts.
- **Cross-view agreement is descriptive.** It shows how much the camera views of one recording
  disagree. It is not a reliability coefficient or a confidence interval.
- **The sample is small.** A cell holds 6 to 16 subjects. The export holds no minimum, no
  maximum, and no subject row.

[`docs/calibration_finding.md`](docs/calibration_finding.md) states the 3D and scale findings.
[`docs/technical/cohort.md`](docs/technical/cohort.md) states the estimand.

## Requirements

- **Python 3.10 to 3.13** and **[`uv`](https://docs.astral.sh/uv/)**.
- **R** with **[`renv`](https://rstudio.github.io/renv/)** for the clinical features.
- An **OpenVINO** device for the corpus run. The GPU runs the detector and the large pose
  models, and the NPU runs the small ones. The CPU also works, but it is several times slower.

## Install

```bash
uv sync
Rscript -e 'renv::restore()'
```

## Run the corpus pipeline

Run each step from the repository root. Each step publishes one directory that the next step
reads. A step refuses to write over a directory that it did not publish.

1. Register the clips and their task, side and view:

   ```bash
   pose-estimation-inventory --corpus videos/3-cam --out inventory
   ```

2. Group the cameras of each recording into one event:

   ```bash
   pose-estimation-sessions --inventory inventory --corpus videos/3-cam --out sessions
   ```

3. Measure the capture properties (frame rate, rotation, synchronization):

   ```bash
   pose-estimation-qualify --inventory inventory --sessions sessions --corpus videos/3-cam --out qualification
   ```

4. Run pose estimation and the clinical features over every event. Load the accelerator
   environment first, or inference runs on the CPU:

   ```bash
   source /var/home/eturkes/.local/app/intel-accel/env.sh
   PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/corpus_run_2d.py --out output/corpus-2d
   ```

   The run resumes at the first unfinished event. To recompute only the clinical features
   after a change to the R stage, add `--analyse-only --refresh-clinical`.

5. Publish the cohort export:

   ```bash
   pose-estimation-cohort --inventory inventory --sessions sessions --run output/corpus-2d --out cohort
   ```

   To record how each feature moved against the previous export, add `--baseline <previous-cohort>`.
   Compare two exports only when their `cohort.json` names the same pose model, settings and clinical method.

[`docs/technical/entrypoints.md`](docs/technical/entrypoints.md) lists every flag.

## Review the results

The review tool shows the clip player, the corpus census and the cohort statistics in a local
web page, in Japanese or English:

```bash
uv run --directory tools/review-ui python -m review_ui
```

Then open `http://127.0.0.1:8791/`. [`tools/review-ui/README.md`](tools/review-ui/README.md)
describes each view.

## How the model was chosen

The benchmark compared five pose models on public ground truth (COCO-WholeBody, DexYCB, HInt),
and RTMW-L 384 led on hand accuracy. Two more steps decide the shipped model: a check of each
finalist inside the full pipeline on labelled corpus frames, and a confirmation on corpus frames
that a person labels by hand. [`docs/technical/benchmark.md`](docs/technical/benchmark.md) records
the results and the decision rules.

## Other uses

The same package also runs on a webcam or a single video file:

```bash
uv run python -m pose_estimation.run --source video.mp4
```

[`docs/technical/entrypoints.md`](docs/technical/entrypoints.md) and
[`docs/technical/tracking-modes.md`](docs/technical/tracking-modes.md) describe these modes.

## Development

The gate runs every check: lint, format, types, the dependency audit, the secret scan, and the
test suite.

```bash
scripts/gate.sh                       # full gate
scripts/gate.sh --fast tests/test_x.py   # static checks and the named tests
```

If a host OpenVINO build is on your `PYTHONPATH`, run commands through `scripts/gate.sh <cmd>`.
A bare `uv run pytest` then fails at import. [`docs/technical/conventions.md`](docs/technical/conventions.md)
explains the gate and the commit style.

## License

Apache-2.0 WITH LLVM-exception. See [`LICENSE`](LICENSE).

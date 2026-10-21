---
paths:
  - "src/pose_estimation/run.py"
  - "src/pose_estimation/rtmlib_*.py"
  - "src/pose_estimation/models.py"
  - "src/pose_estimation/detection.py"
  - "src/pose_estimation/subject_tracker.py"
  - "src/pose_estimation/keypoint_hygiene.py"
  - "src/pose_estimation/camera_survey.py"
  - "src/pose_estimation/corpus_run.py"
  - "scripts/corpus_run_2d.py"
  - "scripts/pilot_corpus_run.py"
  - "docs/technical/pose-weak-points.md"
---

# 2D pose instruments — scratch validators pending port

Scratch-local validators = temporary encodings (`CLAUDE.md` `Engineering`); each port = a `.agent/deferred.md` row.

- **M2.10 watch-set labels + graders** (`.scratch/messy/`; `P` = gate prefix; GPU steps under the
  accelerator recipe). Labels = `.scratch/agents/general-purpose-{1,2,3}-labels.md` (972 frames,
  81 clips; codebook `.scratch/messy/CODEBOOK.md`, validator `python3 .scratch/messy/label_check.py
  <file>` — fires `FAIL` on a seeded bad code) + `.scratch/messy/setup_spans.json` (MAIN's dense
  setup spans, 13 clips). Inputs: `detcache.py cache <labels.json>` (per-frame GPU-f32 boxes + GMC),
  `sheet.py` / `render800.sh` (contact sheets; patient frames → `data-boundary.md`, viewing
  authorized per session by the user only). Graders: `gate_flicker.py <labels.md>…` (hand gate
  replay: drop shares by label, per-subject loss), `select_eval.py <labels.md>… --algos a0`
  (subject-box choice vs labelled rank), `p11_shipped.py` + `p11_grade.py <labels.md>…` (shipped
  task span vs dense + sparse setup labels), `shake_control.py --variant shipped [--view above]`
  (known synthetic camera shake through the shipped compensation), `p08_compare.py <pilot-tree>`
  (in-pipeline gate vs offline replay); corpus-scale: `p11_corpus.py <tree> <labels.md>…` (published
  diag spans → `p11_grade.py` via `P11_SPANS`; 83/83 equal the offline module), `p08_sign.py <m210>
  <m29>` (signed P08 delta + exported-score identity per divergent clip), `p08_keys.py <index> <tree>
  <out>` under the accelerator recipe (one event rerun in-process with every `HandPresenceGate` call
  logged; its CSV must cmp equal the corpus's). Port → `.agent/deferred.md`.
- `.scratch/triage.py [<run-tree>] [--json <out>]` — per-clip 2D weak-point metrics behind
  `docs/technical/pose-weak-points.md` (`whole`, `iso`, `alt`, `flick`, `oof`, `gap`, `lowvis`,
  `swapH`; definitions in that file). `P python .scratch/triage.py output/corpus-2d --json <out>`;
  labels clips by the review UI's positional ordinals (imports `tools/review-ui`), prints
  numbers only. Baselines, 379 clips each: `.scratch/watch/triage_m210.json` (M2.10, shipped), `.scratch/watch/triage_m29.json` (M2.9, superseded) and
  `.scratch/watch/triage_f7.json` (superseded rtmlib-f7).
- `.scratch/impl/p07_stage.py [--tree <run>] [--k N] [--out DIR]` — M4.6 P07 stage attribution
  under the accelerator recipe: ranks clips by `oof_probe.py`'s predicate (reproduces M2.10's 83
  clips / 2134), replays the top N whole under all / `--no-constraints` / `--no-smooth`, reads
  `exported_out_of_frame_zeroed`. `run.main --source` writes no `_diag.csv`, so the script wraps
  `run.process_source` to add one; resumes per clip from `<out>/results.jsonl`. Port →
  `.agent/deferred.md`.
- Corpus after-table instruments, numbers only, review-UI ordinals for labels:
  `P python .scratch/oof_probe.py [<tree>]` (P09: out-of-frame observations at vis >= 0.3, overshoot
  px by edge + keypoint; M2.9 = 3648/2 555 251); `P python .scratch/flick_cause.py [<tree>]` (hand
  blink-outs per 100 rows by view, attributed by the hand's state one frame earlier);
  `python3 .scratch/after_table.py .scratch/watch/triage_f7.json <after.json>` (catalog prevalence
  before/after + paired fixed/new; reproduces the catalog's before figures, `flick` 144 vs 143);
  `P python .scratch/frames_of.py '<#NNN task/side/view>' <metric>` (frame indices behind one
  clip's worst metric, for `bgrab.py` sheets).
- M4 corpus section instruments (numbers only): `python3 .scratch/impl/probes/cohort_shift.py
  <before-cohort> <after-cohort>` (median shift per family, > 0.5 IQR counts, subject-count
  changes); `python3 .scratch/impl/probes/iso_clip.py '<#NNN task/side/view>' <tree>…` (one clip's
  `iso` anatomy); `scientist-2`'s `.scratch/scientist-2/{aggregate_probe,common_support,
  landmark_support}.py` under `.venv/bin/python` (matched-support cohort, alternation, floor sweep,
  flagged-step magnitudes; commands in `.scratch/agents/scientist-2.md` F0). Scratch cohorts per
  arm = `P pose-estimation-cohort … --run output/corpus-2d-m4-<arm> --out .scratch/impl/cohort-<arm>`.
- M2.9 instruments, all `P python .scratch/<name>.py [<tree>]`, review-UI ordinals for labels,
  numbers only: `hygiene_stats.py` (per view × keypoint group: visibility quantiles, out-of-frame
  share — the M2.9.2 premise); `dup_hand.py [--ratio --weak --rel --sample]` (duplicate-hand rule
  firing by view); `hand_metrics.py <tree>…` (fingertip jitter / alternation / hand confidence /
  collapse per view — the M2.9.3 table); `p08_p10.py` (face-eye side agreement at vis >= 0.5 +
  the one-hand-all-zero footprint; superseded tree = 4.6 % agreement, 0 footprint).
- `.scratch/ovupgrade/` — OpenVINO build-change qualification of the run side (container archive,
  accelerator recipe); the gate side (`.venv` wheel, CPU) is graded by the decisive gate alone.
  Run all three after any archive swap. `pilot.sh <label>` = the det_frequency
  sample under the corpus defaults → `.scratch/ovupgrade/<label>`; `P python
  .scratch/ovupgrade/compare.py <treeA> <treeB>` = per-file-kind byte-identity + numeric drift.
  Same build twice: every landmark, clinical, phases, windows + qc CSV byte-identical, only
  `diag.csv` latency + report throughput fields move → any other delta is the build's.
  `det_qual.py [40] [6]` under the accelerator recipe = zero-tensor padding per device + CPU vs
  GPU-f32 detector over 240 frames / 40 clips (→ `rtmlib-runtime.md`). An older archive build
  graded beside the new one = `env.sh` with its paths `sed`-rewritten to the kept directory.

- `.scratch/detfreq_pilot.sh` · `.scratch/detfreq_up.sh` · `.scratch/detfreq_table.sh` — the
  det_frequency sweep. Each script drives `scripts/pilot_corpus_run.py` under the accelerator recipe
  at `--seed 20260922 --min-assets 4 --max-frames 400`, which picks its events from seed plus
  min-assets alone, so **every arm decodes the same 5 events / 11 assets** and wall clocks compare
  directly. Downward arms 7/3/2/1 then upward 14/21/35; `detfreq_table.sh` joins the logged
  `outer_wall_s` to the per-arm defect statistics. All seven arms regenerate from these scripts;
  their figures live in `.agent/archive/instability-m2u9.md`.
- `.scratch/resweep.sh` (`ARMS="1 2 3 7 14 21 35"` default) — the same sample and instruments
  under `--tracker subject --det-device GPU`, arms to `.scratch/resweep/s<F>`; the M2.9.1
  acceptance table (contract P14-P17). `.scratch/cadence_at.py <tree>… [--hz F]` reads
  peak/background at a fixed frequency (default the old 4.29 Hz cadence) through
  `cadence_peak.py`'s own per-track extraction; positive control = old f7 reading ~1.34.
- Offline instruments over any run tree, all six run as
  `P python .scratch/<name>.py [<sample>]` and all six import shared constants from
  `.scratch/recover.py` (which parses `argv[1]` as a sample size **at import time** — a caller
  taking its own arguments must stash and restore `sys.argv` around the import):
  - `mechanism.py` — splits relocations into the whole-skeleton (>= 8 of 10 used body keypoints) and
    isolated (1-2) populations and prints the displacement distribution against the shipped 30 px
    `outlier_cap` / 150 px `match_thresh` constants. Established the bimodality: p90 18.4 px,
    p95 203.9 px, nothing between.
  - `phase_split.py` — the same two populations by detector phase, chi-square against a uniform null
    over the 7 phases. Whole-skeleton decays monotonically from the detector frame (chi2 = 57.66);
    the isolated mode is U-shaped (chi2 = 33.15), which is what proves the two causes distinct.
  - `rigid.py` — decomposes a relocation into centroid translation plus shape change with
    translation removed. Refuted the rigid-translation hypothesis and measured the 509/321 px split.
  - `arm_stats.py <tree> <det_frequency>` — per-arm relocation rate, both populations normalised by
    **observed frames**, alternation ratio and phase enrichment. The normalisation is load-bearing:
    raw counts across arms compare different denominators and reversed the sign of one reading.
  - `frozen_check.py` — the artifact control for any smoothness claim. Exact bit-identical
    consecutive keypoint repeats (0.000 % on all seven arms, so no frozen data) **and** the moving
    fraction, because an arm that is merely still would grade stillness as smoothness.
  - `cadence_peak.py` — power in a +-12 % band around `fps/det_frequency` over the background beside
    it, with a 6.7 Hz control column that must read ~1.0 or the statistic is broken. Needs no
    cross-arm alignment, so it carries ~75-82 tracks per arm against `band_arms.py`'s 10-14.
  - `band_arms.py` — fixed-band energy per arm against f7 on frame-aligned common tracks. Weakest of
    the six on sample size; quote it only beside `cadence_peak.py`.


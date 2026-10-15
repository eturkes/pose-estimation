# Spec

## Intent

Turn the hospital's uncontrolled three-camera recordings of spinal-cord-injury rehabilitation
tasks into measured movement statistics that feed the `../rehab` clinical dashboard over the
hospital SCI database.

- The corpus is `videos/3-cam/`: 382 hand-held clips of 16 subjects × 6 tasks × 2 sides. Each
  family has three views at best. The recordings have no fixed rig and no calibration.
- The pipeline tracks 2D pose per clip, extracts clinical kinematic features, and aggregates to
  cohort level — no per-subject rows, no patient identifier, no join column leaving this repo.
- The shareable review UI is itself an intended deliverable for collaborators, not only a
  prototype step.
- The eventual destination is a subject↔patient join into the hospital SCI database. That mapping
  does not exist yet.

## Artifacts

`P` = the gate prefix; `P` + the mutually-exclusive accelerator recipe → `.claude/rules/gates.md`.

- `prototype/review-ui/` — **the prototype, the inspectable artifact.** One local web UI,
  bilingual ja/en (`?lang=en`), themed auto|light|dark (`?theme=`, auto = `light-dark()` + the
  OS), three views, clip player (the landing view) · corpus census · cohort explorer.
  FastAPI + vanilla JS canvas + vendored Plotly/IBM Plex, own uv project, read-only over the
  published trees, degrading per absent tree. Commits no media at all → the player needs the
  published trees and lists nothing without them; `README.md` = view guide + regeneration + limits.
  `uv run --directory prototype/review-ui python -m review_ui` → `http://127.0.0.1:8791/`.
- `cohort/` — the `../rehab` export, from the M2.10 generation. 12 `(task, side)` cells · 89
  features · 1068 rows; trunk features 6-16 subjects (side views alone), non-task-hand + bilateral
  hand features 6-16 (hand gate) → `cohort.md`; `descriptors.yaml` = ja/en labels, units, ranges.
  `P pose-estimation-cohort --inventory inventory --sessions sessions --run output/corpus-2d --out cohort`
- `output/corpus-2d/` — per-asset 2D landmarks + clinical features, M2.10 generation (M2.9's subject
  tracker, GPU-f32 detector every frame, NPU pose, seated drops, SPARC v3 + hand presence gate +
  camera survey: task span, per-row camera similarity in R): 193/193 events, 379/379 `ok`,
  337 090 frames, 8.197 h measured, all 11 `run_report.json` verdicts true, `task_span` published.
  `.scratch/rerun_m210.sh` = `python scripts/corpus_run_2d.py` under the accelerator recipe;
  resumes per event marker. Before/after vs M2.9 → `docs/technical/pose-weak-points.md`
  § M2.10 corpus. Superseded generations kept for the user's before/after review (user: keep
  both; delete on the user's word): M2.9 at `output/corpus-2d-superseded-m29/` + cohort
  `.scratch/compare-m29/cohort/`; rtmlib-f7 at `output/corpus-2d-superseded-rtmlib-f7/` +
  `.scratch/compare-root/cohort/`. Comparison UI = `uv run --directory prototype/review-ui python
  -m review_ui --port 8792 --repo "$PWD/.scratch/compare-m29"` (`compare-root` for rtmlib-f7;
  symlinks onto the primary trees + the superseded run).
- `inventory/` `sessions/` `qualification/` `calibration_qc/` — four publishers upstream of the
  run; each `P pose-estimation-<name> … --out <dir>`; `--help` = args.
- Gate — `scripts/gate.sh` (full chain + per-step wall-clock; `--fast <tests>` iterates) →
  `.claude/rules/gates.md`. Its pytest step = the decisive gate: 2370 collected, 21:54 wall measured
  at `502affa` (host load moves it), run alone.

## Decisions

- **Repo scope = `videos/3-cam/`** — retired data + siblings → `.claude/rules/data-boundary.md`.
- **`src/` + `tests/` + `analysis/` + six publishers = production spine**, gates + verification
  integrity binding; the prototype (`prototype/review-ui/`, → `Artifacts`) sits outside under
  PROTOTYPE law, `testpaths = ["tests"]` keeping it out of collection. Spine rows in `Tasks` =
  MAINTAIN requests on it, under MAINTAIN law, never PROTOTYPE or ITERATE law (→
  `.claude/rules/delegation.md`).
- **Claim boundary.** Retrospective 3D feasibility may be claimed from internal geometric + QC
  evidence alone; clinical validity, absolute metric accuracy and marker-based equivalence may not.
  Crossing it needs the prospective calibrated capture in `docs/prospective_capture.md`.
- **3D is closed negative.** Extrinsic recovery is unachievable here — 15-20 px systematic
  cross-view keypoint bias, refused by the shipped estimator and by independent bundle adjustment,
  not transferring across events, so no repair route survives. Publishes through `calibration_qc/`;
  reopens on prospective calibrated capture.
- **Metric scale is unavailable.** Over a stratified 52/379 sample exact dimensional identity
  resolved 0/52, best conditional route floors at ±17.7 %. Angles, angular velocities, timing +
  dimensionless shape survive; metre-valued distance, velocity + jerk do not. Publishes
  `scale_unmeasured` on all 379 rows.
- **Calibration is per recording event at best** — no fixed rig existed; orientation, codec,
  parity + duration spread each measure it independently.
- **M3 (analysis-ready 3D aggregation) DESCOPED + terminal** — M3.1/M3.2/M3.3a ship + keep gates;
  M3.3b + M3.4-M3.6 cut. Revive = user ruling → `.agent/archive/m3.md`.
- **`publication.py` extraction stays DECLINED** — six publishers keep their own
  staging/swap/digest/ownership copies; the measured five-way drift is the evidence.
- **Detector on GPU at f32, pose on NPU** — NPU pads YOLOX's dynamic output with uninitialised
  memory that passes every validity filter as real rows; GPU f32 matches the CPU detector
  (240 corpus frames: IoU 1.0000, score Δ < 3.5e-6, 0 count mismatches) at 41.6 vs 318.7 ms/call,
  and GPU f16 flips a box at the 0.3 cut.
- **An out-of-frame keypoint is not an observation** (user ruling) — it scores 0, so above-view
  elbow and reach features lose the ~42 % of shoulder frames the camera cannot see rather than use
  the model's guess.
- **Seated subjects drop what the camera cannot see** (user ruling) — knees, ankles and feet score
  0 on every camera, hips score 0 under the overhead camera; trunk lean and rotation come from
  side views (posture symmetry reads the shoulders alone and keeps every view).
- **The hand model stays RTMW-L's own hands** — a second-stage hand model is adopted only at
  fingertip jitter <= 0.006 on the watch set, wall <= +15 %, collapse unchanged per clip (user
  bar); RTMW-X and both hand-crop refinements missed it.
- **Acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`** — 9 files under
  `scripts/ src/ tests/` break if it moves, one a generated data field no gate resolves;
  re-derive → `.claude/rules/retention.md`.
- **External AI-judgment services stay out of this repo** — surveyed and declined by the user.
  The three defects the survey measured live on as `.agent/deferred.md` rows with mechanism left
  open; the survey itself is not repeated.
- **Assurance tier = `kernel`** across pipeline, publishers + analysis.
- **2D instability repair = `SubjectTracker`** — every crop a detector box, never re-sized from
  the pose; detector every frame on GPU at f32 (41.6 ms/call, CPU-equal to IoU 1.0000 on 240
  corpus frames); box transported by pose motion across detector misses; sticky largest-box subject;
  identity-keyed smoothing. Supersedes the *reconcile* subclass (user ruling, `.agent/archive/instability-m2u9.md`)
  whose premise was the CPU detector's 8.23× cost for `det_frequency=1`; that ruling's acceptance
  (seven-arm re-sweep, alternation <= 0.55, isolated ~0, no cadence peak, wall near f7's 327 s)
  stands → `.agent/archive/contract-m2u91.md` P14-P17.
- **No low-pass filter stage** — `det_frequency=1` alone reaches alternation 0.495 against the
  post-hoc Hampel+6 Hz stage's 0.483 while keeping 0.697 of the 2-5 Hz band against its 0.464.
  Removing the noise at source is what makes filtering unnecessary; a 6 Hz cutoff on top would
  destroy tremor-band energy the source no longer carries as noise, and SPARC is cutoff-sensitive
  so it would move the smoothness feature too.
- **SPARC is the primary smoothness feature, `normalized_jerk` secondary** — `*_wrist_sal`
  computes SPARC (adaptive cutoff, method v3); it publishes with the M2.9 corpus rerun.
- **Hallucinated hands are gated by their exported score** (user ruling) — hysteresis on 0.65 /
  off 0.5 per track after the smoother; held out: 90 % of hallucinated hands removed, 3.7 % of real
  hands lost (one subject 12 %) → `docs/technical/pose-weak-points.md` § M2.10.
- **Clips are trimmed to their estimated task span and motion is read in the reference frame** (user
  ruling) — frames outside the span export no rows (setup frames excluded 0.967 on 13 densely
  labelled clips); each row's estimated camera similarity is applied in R before velocity, jerk,
  SPARC and reach (validated on synthetic shake); per-row inter-segment angles are unchanged.
- **Fix upstream only** — the pipeline changes and the corpus reruns, so `output/corpus-2d/` stays
  the single source of truth and no post-hoc filter bolts onto published tracks.

## Tasks

- [x] 397cc8a **Review UI overlay time map (C01)**
- [x] b8ccd71 **Review UI stacked layout drifted right + shrank at ratio 1.1 (user report)**
- [x] aed1e7d **Review UI player needed page scroll to show the stage (user request)** — side by
  side down to 761 px, fitted 2:5 stack below; user-approved ("much better and stable").
- [x] ecd0d86 **M2.10 messy-footage compensation, run prep (user request)** — M2.10.1 hand gate
  `4c471ab`, M2.10.2 camera survey `ecd0d86`; wrong-person residue + 3D angles → `.agent/deferred.md`.
- [x] 08d0594 **M2.10 corpus run** — 193/193 events, 379/379 `ok`, 11/11 verdicts, `task_span`
  published; cohort republished; after-table → `pose-weak-points.md` § M2.10 corpus.
- [ ] **User reviews the M2.10 videos** — player at `:8791` (M2.10) beside `:8792 --repo
  "$PWD/.scratch/compare-m29"` (M2.9); the user's word also decides the superseded trees' deletion.
- [x] 677e56b **Review UI JP subset builds from gitignored `cohort/descriptors.yaml`** — builder
  refuses absent; `tools/check_glyphs.py` 0 missing (→ `review-ui.md` firing).
- [x] 57a891d **Overlay landmark->pixel map proven** — `tools/overlay_map_qa.mjs`, max 0.089 px after
  the contain-matched transform (was 0.871 px).
- [x] 57a891d **HEVC decode banner exercised** — `tools/decode_banner_qa.mjs`, synthetic hvc1.

Evidence → `.agent/archive/{polish,review-m2}.md`; regen → each check's `paths:` rule in `.claude/rules/`; queue → `.agent/deferred.md`.

## Phase

ITERATE — review UI (`prototype/review-ui/`, the one prototype in `Artifacts`)

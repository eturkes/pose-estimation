# Spec

## Intent

Turn the hospital's uncontrolled three-camera recordings of spinal-cord-injury rehabilitation
tasks into measured movement statistics that feed the `../rehab` clinical dashboard over the
hospital SCI database.

- The corpus is `videos/3-cam/`: 382 hand-held clips of 16 subjects × 6 tasks × 2 sides. Each
  family has three views at best. The recordings have no fixed rig and no calibration.
- The pipeline tracks 2D pose per clip, extracts clinical kinematic features, and aggregates to
  cohort level — no per-subject rows, no patient identifier, no join column leaving this repo.
- The review UI confirmed the shape of the deliverable as the prototype and stays as the review
  instrument. The deliverable is the analysed videos — movement statistics ready for `../rehab`.
- The eventual destination is a subject↔patient join into the hospital SCI database. That mapping
  does not exist yet.

## Artifacts

`P` = the gate prefix; `P` + the mutually-exclusive accelerator recipe → `.claude/rules/gates.md`.

- `tools/review-ui/` — **the review instrument** (retired prototype, tag `prototype-review-ui`;
  user ruling): inspects each corpus generation, PROTOTYPE law, outside the gates. One local web UI,
  bilingual ja/en (`?lang=en`), themed auto|light|dark (`?theme=`, auto = `light-dark()` + the
  OS), three views, clip player (the landing view) · corpus census · cohort explorer.
  FastAPI + vanilla JS canvas + vendored Plotly/IBM Plex, own uv project, read-only over the
  published trees, degrading per absent tree. Commits no media at all → the player needs the
  published trees and lists nothing without them; `README.md` = view guide + regeneration + limits.
  `uv run --directory tools/review-ui python -m review_ui` → `http://127.0.0.1:8791/`.
- `cohort/` — the `../rehab` export, v3 from the adopted RTMW-L 384 generation. 12 `(task, side)`
  cells · 88 features · 1056 rows · 6185 bout rows; min 6 subjects (trunk features, side views
  alone), 0 rows under the 5-subject floor → `cohort.md`; `descriptors.yaml` = ja/en labels, units,
  ranges; `measurement` = pose config `45c77741…` + R producer `2e02883b…`;
  `cohort_sensitivity.csv` vs the M2.10 cohort (v1, `.scratch/compare-m210/cohort`): 888 common /
  168 new / 180 retired, 178/888 move > 0.5 baseline IQR.
  `P pose-estimation-cohort --inventory inventory --sessions sessions --run output/corpus-2d --out cohort
  [--baseline <previous cohort>]`
- `output/corpus-2d/` — per-asset 2D landmarks + clinical features, M4 generation adopted by user
  ruling: RTMW-L 384 on GPU f32 + its reference-equivalent score calibration, GPU-f32 detector every
  frame, subject tracker, seated drops, hand gate, camera survey, M4.6 fixes, OpenVINO 2026.4.1
  (landmarks `48078b4`), clinical stage = bouts + SPARC v3 (`2e02883b…`): 193/193 events, 379/379
  `ok`, 337 090 frames, 14.365 h measured, all 12 `run_report.json` verdicts true. The defaults
  reproduce it (`run.DEFAULT_MODEL` = `rtmw-l-384`; 193/193 recorded configurations equal):
  `A scripts/corpus_run_2d.py` (`A` = the accelerator recipe), resumes per event marker.
  Effect vs M2.10 + RTMW-m → `docs/technical/pose-weak-points.md` § M4 corpus. Kept beside it:
  `output/corpus-2d-m4-rtmw-m-256` (the M4 RTMW-m arm; comparison side of the F4/F5/F6
  `.agent/deferred.md` rows). Superseded generations kept by user ruling (delete on the user's
  word): M2.10 at `output/corpus-2d-superseded-m210/` + cohort `.scratch/compare-m210/cohort/`
  (republish only from its own code `487f790`: its `pose_config.json` predates `openvino_build` +
  `calibration`); M2.9 at `output/corpus-2d-superseded-m29/` + `.scratch/compare-m29/cohort/`;
  rtmlib-f7 at `output/corpus-2d-superseded-rtmlib-f7/` + `.scratch/compare-root/cohort/`.
  Comparison UI = `uv run --directory tools/review-ui python -m review_ui --port 8792 --repo
  "$PWD/.scratch/compare-m210"` (`compare-m29`, `compare-root` likewise; symlinks onto the primary
  trees + the superseded run).
- `inventory/` `sessions/` `qualification/` `calibration_qc/` — four publishers upstream of the
  run; each `P pose-estimation-<name> … --out <dir>`; `--help` = args.
- Benchmark — `scripts/bench.py <command>` (sample, sheets, ingest, agree, public-eval, hand-eval,
  corpus-gross, calibrate, judge-*, preroll, user-sample; inference + replay under the accelerator
  recipe, the rest under the gate prefix → `--help`) →
  `docs/technical/benchmark.md`; COCO-WholeBody AP via `tools/cocowb-eval/` (own Python 3.12 uv
  project); review UI `/label` = a labelling page with no live use (M4.4 waived).
- Gate — `scripts/gate.sh` (full chain + per-step wall-clock; `--fast <tests>` iterates) →
  `.claude/rules/gates.md`. Its pytest step = the decisive gate (A32 reconciled in-session): 3142 passed at
  IMPLEMENT close, 1251 s alone with another project's suite on the host (670 s for 2733 at M4.2b,
  quiet); run it alone.

## Decisions

- **Repo scope = `videos/3-cam/`** — retired data + siblings → `.claude/rules/data-boundary.md`.
- **`src/` + `tests/` + `analysis/` + six publishers = production spine** = the gated scope,
  Engineering law + verification integrity binding; the prototype (`tools/review-ui/`, → `Artifacts`)
  sits outside under PROTOTYPE law, `testpaths = ["tests"]` keeping it out of collection (→
  `.claude/rules/delegation.md`).
- **Model selection, M4** (user rulings) — accuracy within the corpus cost ceiling: one full
  corpus run <= 1 day of iGPU, an arm over it never a finalist (`contract-m4u2.md` A04). Public 2D
  GT through the deployment (COCO-WholeBody, DexYCB, HInt; ARCTIC, Ego-Exo4D, UBody unregistered)
  under a finalist rule frozen before results → RTMW-L 384 the one challenger (Sapiens v1/v2 +
  SDPose over the ceiling); M4.3 complete-pipeline gross check passed; M4.4 in-domain confirmation
  waived + the user's labelling dropped → RTMW-L 384 adopted on public evidence + M4.3. Fine
  accuracy on this footage is unmeasured. Any later candidate competes against the shipped model
  under the same ceiling; data-boundary, claim-boundary, 3D-negative + metric-scale rulings stay
  binding.
- **In-domain accuracy evidence** — public GT decides fine accuracy; agent labels grade gross
  errors alone, hand points at 0.6 of the hand extent — their consensus's own error vs public GT
  (p90 0.518 thumb tip), never finer (M4.3 G2, contract-m4u3 A02); no user-labelled confirmation
  set exists (M4.4 waived, user ruling).
- **Shipped pose model = RTMW-L 384x288** (`rtmw-l-384` = mmpose `rtmw-dw-x-l` cocktail14 at
  384x288; mmpose table hand AP 0.663 / whole 0.701; here, GT boxes: hand 0.648, whole 0.714 vs
  RTMW-m 0.534 / 0.634) on GPU f32, scores through its reference-equivalent Platt map
  (`score_calibrations/rtmw-l-384.json`; the export's raw scores are unnormalised). Every generation
  through M2.10 ran RTMW-m 256x192 (checkpoint `rtmw-dw-l-m`), recorded as `rtmw-l` → that key
  stays an alias of `rtmw-m-256`. True RTMW-L + RTMW-X fail NPU compile (CMX scheduler) and run on
  GPU f32. Registry keys = mmpose-table names (`rtmw-m-256`, `rtmw-l-256`, `rtmw-l-384`,
  `rtmw-x-384`, `dwpose-l-384`).
- **Review UI shape** (user rulings, ITERATE) — Japanese default, English by switch; theme
  auto|light|dark, auto = the OS; the clip player = first tab + the view a bare URL opens; the whole
  frame + transport visible with no page scroll (list beside stage >= 761 px, fitted stack below);
  no committed media, no demo clip, no stored proof.
- **Phase-close dispositions** (user rulings) — the review UI moves to `tools/review-ui/` (tag
  `prototype-review-ui`), still outside the gates; scanners run in the local gate alone, the remote
  gets Dependabot config only; every superseded corpus generation stays until the user's word.
- **Claim boundary.** Retrospective 3D feasibility may be claimed from internal geometric + QC
  evidence alone; clinical validity, absolute metric accuracy and marker-based equivalence may not.
  Crossing it needs the prospective calibrated capture in `docs/prospective_capture.md`.
- **3D is closed negative.** Extrinsic recovery is unachievable here — 15-20 px systematic
  cross-view keypoint bias, refused by the shipped estimator and by independent bundle adjustment,
  not transferring across events, so no repair route survives. Publishes through `calibration_qc/`;
  reopens on prospective calibrated capture. Each arm's cross-view residual is measured against
  RTMW-m's (a material drop returns the ruling to the user): RTMW-L 384 rises, ratio 1.23 [1.02,
  1.46] → the ruling stands (`docs/technical/benchmark.md`).
- **Monocular 3D angles declined** (user ruling) — SAM 3D Body over 60 events / 1940 synced
  instants: elbow flexion cross-view MAE 5.4° synced vs 8.8° time-shuffled (2D image-plane 26.7° vs
  29.9°, r 0.19), 3D closer on 65/66 series; shoulder 3D within-series SD half of 2D (prior); the
  trunk-vs-pelvis angle reads the prior (SD 0.5°). Agreement = consistency, no GT; cost 0.43 s/frame
  (0.56 s with seeks) ≈ 40-52 h per corpus pass at every frame. Pilot code + gated weights deleted.
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
- **Detector on GPU at f32; pose on the model's registry device** (RTMW-L 384: GPU f32) — NPU pads
  YOLOX's dynamic output with uninitialised memory that passes every validity filter as real rows;
  GPU f32 matches the CPU detector (240 corpus frames: IoU 1.0000, score Δ < 3.5e-6, 0 count
  mismatches) at 41.6 vs 318.7 ms/call, and GPU f16 flips a box at the 0.3 cut.
- **An out-of-frame keypoint is not an observation** (user ruling) — it scores 0, so above-view
  elbow and reach features lose the ~42 % of shoulder frames the camera cannot see rather than use
  the model's guess.
- **Seated subjects drop what the camera cannot see** (user ruling) — knees, ankles and feet score
  0 on every camera, hips score 0 under the overhead camera; trunk lean and rotation come from
  side views (posture symmetry reads the shoulders alone and keeps every view).
- **Hand keypoints = the shipped whole-body model's** (RTMW-L 384; user ruling: the benchmark
  selects within the corpus cost ceiling) — no second-stage hand model; the earlier bar (jitter
  <= 0.006, wall <= +15 %) retired; its "RTMW-X" arm was RTMW-L 384x288, whose ONNX export emits
  unnormalised scores (600 COCO crops: median 4.2, max 8.4) → that rejection measured the export.
- **Acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`** — the files a move
  breaks (one a generated data field no gate resolves) re-derive by `.claude/rules/retention.md`.
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
- **SPARC is the primary smoothness feature, `normalized_jerk` secondary — per wrist-motion
  bout** (M4.5a) — SPARC (adaptive cutoff, method v3) and NJ are defined on one movement, so the
  export carries `*_bout_sparc` / `*_bout_normalized_jerk` and lists the window forms
  `construct_invalid`; detector parameters = a synthetic qualification's cell
  (`contract-m4u5a.md` A01).
- **Cross-view agreement is descriptive** (M4.5b) — `view_abs_diff` primary, unclipped one-way
  `view_icc` + subject-resample range, never a reliability or a confidence interval.
- **Hallucinated hands are gated by their exported score** (user ruling) — hysteresis on 0.65 /
  off 0.5 per track after the smoother, on the model's calibrated score (RTMW-L 384: its Platt map,
  no refit at M4.3). Held out on RTMW-m (M2.10): 90 % of hallucinated hands removed, 3.7 % of real
  hands lost (one subject 12 %) → `docs/technical/pose-weak-points.md` § M2.10; RTMW-L 384 corpus:
  143 816 / 623 822 hand-frames gated (0.231), held-out loss unmeasured.
- **Clips are trimmed to their estimated task span and motion is read in the reference frame** (user
  ruling) — frames outside the span export no rows (setup frames excluded 0.967 on 13 densely
  labelled clips); each row's estimated camera similarity is applied in R before velocity, jerk,
  SPARC and reach (validated on synthetic shake); per-row inter-segment angles are unchanged.
- **Fix upstream only** — the pipeline changes and the corpus reruns, so `output/corpus-2d/` stays
  the single source of truth and no post-hoc filter bolts onto published tracks.

## Tasks

- [ ] RESUME: request = work every `.agent/deferred.md` row in rank order (user's MAINTAIN body,
  session 2026-10-22); `Met when` = every row closed by its acceptance check in its own commit,
  recorded blocked on the user with what it owes, waiting on its trigger, or routed (`route.md`),
  + full gate green on a clean tree at the closing commit. One AskUserQuestion collects user-owed
  parts after agent-side work; new finds fund after the session-start rows. Working state +
  per-row classification + owed list = `.scratch/maintain/plan.md` (read first).
  - Full gate PASS at 52d9427, ac21d21, 6d3ab19 (3153 passed, 1929 s under co-tenant load;
    worktree `.scratch/maintain/gate-wt`, logs `.scratch/maintain/gate-<sha7>.log`). NOT gated:
    55f3c31 c4f7231 ca6b013 57b69bc 317e91f 97bbeee 6615eb4 9650496 f98b850 (all in `queue.txt`)
    → rerun the full gate per commit (`nohup .scratch/maintain/gateq.sh &`; it skips SHAs in
    `done.txt`, gates `queue.txt` in order). The template-refresh commit after f98b850 = gated in
    its own body; R15's 3 files were parked out of its tree + restored uncommitted after it. Its
    full gate reds `test_p05_temporal_panel_names_sparc` since c4f7231 → `.agent/deferred.md`
    (fund first: it blocks this row's full-gate finish line).
  - Uncommitted (R15, rank 15): `src/pose_estimation/run.py` (--source writes `<stem>_diag.csv`),
    `tests/test_run_source_diag.py` (2 green; red at 9650496 = 1 failed), `scripts/bench.py` `p07`
    (port of `.scratch/impl/p07_stage.py`, UNRUN); snapshot = branch `wip/r15-source-diag` 579032f.
    Next: run `bench.py p07 --tree output/corpus-2d-superseded-m210 --k 6 --out .scratch/bench/p07
    --run-args --model=rtmw-m-256` under the accelerator recipe (add `--det-device=CPU` if counts
    miss the pose-weak-points P07 table), then commit R15 + prune its row.
  - Teammate branches (worktrees kept, unmerged, last states NOT rerun by MAIN): `wt/general-purpose-2`
    748aea0 (R10 register scanner; report `.scratch/agents/general-purpose-2.md`: S01-S08 pass, the
    `robust Sampson` exact-phrase ruling applied; its fast gate 42 passed; Register = rule +
    conventions text proposals); `wt/general-purpose-3` 5e7a855 (review-UI ports: R20 294ec7d, R21 a46ace9, R22
    251e4dc, R28 50b81c4, R29 5e7a855; report `.scratch/agents/general-purpose-3.md`: Q02 owes a
    10/10 rehydration trial under a live full gate, Q05 owes the clock ruling in
    `.claude/rules/review-ui.md`) → rerun each port in the primary, merge one commit per row.
  - Next rows after R15: R16 triage port onward (rank order, `.agent/deferred.md`); owed-to-user
    so far (plan.md): R07 ../rehab edit, R13 conjunct residues, R14 retired bar; N01 pytest
    speedup + N02 P12 spelling test are new rows (fund after the session-start rows).
Evidence → `.agent/archive/{polish,review-m2,phases}.md`; regen → each check's `paths:` rule in `.claude/rules/`; queue → `.agent/deferred.md`.

## Phase

MAINTAIN — video analysis (2D pose, features, cohort export to `../rehab`)

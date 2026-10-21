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
- `cohort/` — the `../rehab` export, from the M2.10 generation. 12 `(task, side)` cells · 89
  features · 1068 rows; trunk features 6-16 subjects (side views alone), non-task-hand + bilateral
  hand features 6-16 (hand gate) → `cohort.md`; `descriptors.yaml` = ja/en labels, units, ranges.
  `P pose-estimation-cohort --inventory inventory --sessions sessions --run output/corpus-2d --out cohort`
- `output/corpus-2d/` — per-asset 2D landmarks + clinical features, M2.10 generation (M2.9's subject
  tracker, GPU-f32 detector every frame, NPU pose, seated drops, SPARC v3 + hand presence gate +
  camera survey: task span, per-row camera similarity in R): 193/193 events, 379/379 `ok`,
  337 090 frames, 8.197 h measured, all 11 `run_report.json` verdicts true, `task_span` published.
  `.scratch/rerun_m210.sh` = `python scripts/corpus_run_2d.py` under the accelerator recipe;
  resumes per event marker. Its `pose_config.json` files predate `openvino_build` + `calibration`
  (M4.6, M4.3a), so today's drivers refuse `--analyse-only` / `--reuse-run` over it as another
  configuration — by design; republish it only from its own code (`487f790`). Before/after vs M2.9 → `docs/technical/pose-weak-points.md`
  § M2.10 corpus. Superseded generations kept by user ruling (M2.10 videos reviewed; delete on
  the user's word): M2.9 at `output/corpus-2d-superseded-m29/` + cohort
  `.scratch/compare-m29/cohort/`; rtmlib-f7 at `output/corpus-2d-superseded-rtmlib-f7/` +
  `.scratch/compare-root/cohort/`. Comparison UI = `uv run --directory tools/review-ui python
  -m review_ui --port 8792 --repo "$PWD/.scratch/compare-m29"` (`compare-root` for rtmlib-f7;
  symlinks onto the primary trees + the superseded run).
- `inventory/` `sessions/` `qualification/` `calibration_qc/` — four publishers upstream of the
  run; each `P pose-estimation-<name> … --out <dir>`; `--help` = args.
- Benchmark — `scripts/bench.py <command>` (sample, sheets, ingest, agree, public-eval, hand-eval,
  corpus-gross, calibrate, judge-*, preroll, user-sample; inference + replay under the accelerator
  recipe, the rest under the gate prefix → `--help`) →
  `docs/technical/benchmark.md`; COCO-WholeBody AP via `tools/cocowb-eval/` (own Python 3.12 uv
  project); user labelling page = review UI `/label` over `.scratch/bench/user/`.
- Gate — `scripts/gate.sh` (full chain + per-step wall-clock; `--fast <tests>` iterates) →
  `.claude/rules/gates.md`. Its pytest step = the decisive gate (A32 reconciled in-session): 2733 passed at
  the M4.2b close, 670 s measured alone (1339 s beside a corpus run); run it alone.

## Decisions

- **Repo scope = `videos/3-cam/`** — retired data + siblings → `.claude/rules/data-boundary.md`.
- **`src/` + `tests/` + `analysis/` + six publishers = production spine** = the IMPLEMENT scope,
  gates + verification integrity binding; the prototype (`tools/review-ui/`, → `Artifacts`)
  sits outside under PROTOTYPE law, `testpaths = ["tests"]` keeping it out of collection (→
  `.claude/rules/delegation.md`).
- **IMPLEMENT optimises the video analysis itself** (user rulings) — model selection + parameter
  optimisation across detector, pose + hand model, tracker, filters, hand gate, span trim and
  features, benchmarked; every model + parameter ruling below is open, a replacement lands as a new
  ruling. Accuracy wins within the corpus cost ceiling (user ruling): one full corpus run <= 1 day
  of iGPU, an arm over it never a finalist (`contract-m4u2.md` A04); any licence class; any
  candidate family; features may change definition (per-movement SPARC, timing features, a
  published per-feature reliability) or join + leave the export. Ground truth = public 2D GT run
  through the deployment (COCO-WholeBody, ARCTIC, Ego-Exo4D, UBody, DexYCB, HInt) + in-domain
  keypoints + per-side hand visibility placed by agents on corpus frames + GT-free proxies;
  selection = a rule frozen before candidate results. Any `.agent/deferred.md` row may join.
  Data-boundary, claim-boundary, 3D-negative + metric-scale rulings stay binding.
- **In-domain accuracy evidence** (user ruling, after M4.1 measured agent hand points at ~0.10 of
  hand size and the blinded pairwise judge near chance) — public GT decides fine accuracy now;
  agent labels grade gross errors only; the user clicks wrist + fingertip points on ~100 sampled
  corpus frames in a review-UI labelling page, and those frames are the confirmation set before
  the final corpus run.
- **Shipped pose model = RTMW-m 256x192** — registry key `rtmw-l` names checkpoint
  `rtmw-dw-l-m` = RTMW-m in mmpose's own table (COCO-WholeBody hand AP 0.491, whole 0.582; true
  RTMW-L `rtmw-dw-x-l` 0.598 / 0.660, at 384x288 0.663 / 0.701); every generation through M2.10 ran
  it. True RTMW-L + RTMW-X fail NPU compile (CMX scheduler) and run on GPU f32. The registry keys
  models by mmpose-table names (`rtmw-m-256`, `rtmw-l-256`, `rtmw-l-384`, `rtmw-x-384`,
  `dwpose-l-384`, M4.3a); `rtmw-l` stays an alias of `rtmw-m-256` for recorded configurations.
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
- **Hand model = whichever arm the benchmark selects** (user ruling, within the corpus cost ceiling) — the earlier
  second-stage bar (jitter <= 0.006, wall <= +15 %) retires with it; its "RTMW-X" arm was RTMW-L
  384x288 whose ONNX export emits unnormalised scores (600 COCO crops: median 4.2, max 8.4), so its
  rejection measured the export, not the model.
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
  off 0.5 per track after the smoother; held out: 90 % of hallucinated hands removed, 3.7 % of real
  hands lost (one subject 12 %) → `docs/technical/pose-weak-points.md` § M2.10.
- **Clips are trimmed to their estimated task span and motion is read in the reference frame** (user
  ruling) — frames outside the span export no rows (setup frames excluded 0.967 on 13 densely
  labelled clips); each row's estimated camera similarity is applied in R before velocity, jerk,
  SPARC and reach (validated on synthetic shake); per-row inter-segment angles are unchanged.
- **Fix upstream only** — the pipeline changes and the corpus reruns, so `output/corpus-2d/` stays
  the single source of truth and no post-hoc filter bolts onto published tracks.

## Tasks

- [x] 13fc4dd **RTMW-m label + phase rulings** — live surfaces name the shipped checkpoint; rulings above.
- [x] 622317e **Security scanners + update automation** — `uv audit` over both uv lockfiles, secret scan,
  security lint, each with a planted firing; Dependabot `uv` × 2; urllib3 2.7.0 → 2.8.0 (6 advisories).
- [ ] **M4.1 instrument qualification** (`2ef508e`; pre-roll measured `48078b4` — no finite pre-roll matches the whole clip; paired-subject SD open, from the M4 corpus runs) (`kernel`, `.agent/archive/contract-m4u1.md`; design review
  `consultant-2`: qualify before any selection protocol freezes) — codebook v2 on raw frames,
  labeller error vs public GT + inter-labeller agreement, pairwise judge on real model-error pairs,
  pre-roll for stateful replay, paired-subject variance → qualified resolutions. Corpus frame
  viewing authorized this session for MAIN + labelling teammates (user).
- [x] 36e525b 8bc1b81 **M4.2 public screening** (finalists RTMW-L 384 + baseline; Sapiens2 dropped + gated sets unrun by user ruling, contract A03) — COCO-WholeBody official evaluator (GT boxes; body + hand parts),
  one qualified manipulation set (ARCTIC / Ego-Exo4D / DexYCB), HInt visibility; every candidate
  (RTMW m/L/X, DWPose, Sapiens, SDPose, hand models, detectors) → a diverse finalist set.
- [x] 73927ef **On-path defects** (M4.6; P07: the temporal filter places the out-of-frame residue,
  bone-length adds on 2 of 6 clips → `docs/technical/pose-weak-points.md` § M4.6) — P08 gate keying,
  P09 out-of-frame residue, D02 counter, OpenVINO build in `pose_config.json`, raw-score decode.
- [x] 73927ef **M4.3a arm plumbing** — mmpose-named registry, per-model device, score-calibration hook.
- [x] bf7728a **M4.3b calibration + gate rule** — reference-equivalent Platt maps, per-model hand gate, presence instrument; refit rule applies in M4.3.
- [ ] **M4.3 complete-pipeline finalists** (instrument `b88b0e9`; scored from the M4 corpus runs) — pose × hand × detector/identity (incl. a wrong-person
  arm), each recalibrated on development data, dynamic + feature controls, production replay.
- [ ] **M4.4 locked confirmation** (table frozen `2ef508e`; user labels 0 of 96 saved, review UI stopped → restart for `/label`) — decision table (superiority / non-inferiority / inconclusive)
  frozen with margins before a one-use corpus subset is scored; gates vs the shipped baseline.
- [x] 2d0270f 8fe4e2b 48ef1fd **M4.5 features** — qualified wrist-motion bouts (SPARC, NJ, timing,
  peaks, path ratio, dwell), cohort v2 (bout level, construct-invalid exclusions, descriptive
  agreement), `--refresh-clinical` for runs made before the bout stage.
- [x] 3410a90 **`calibration_qc` claims name RTMW-m** — published claim text + fixtures + determinism campaigns.
- [ ] **Monocular 3D angles pilot** (SAM 3D Body, >= 40 events) → user adoption ruling.
- [x] fc58765 **Gate speed** — A32 reconciled in-session (`tests/a32_gate.py`), `test_c8_08` = plugin firing; pytest 2465 s vs 5634 s (44 %).
- [x] f35f83e **M4.2b heavy arms** (user request: HF-gated models first) — Sapiens v1 0.3B-2B
  (gated, access granted) + SDPose screened; every arm over the 1-day corpus ceiling → dropped
  (`contract-m4u2.md` A04, `docs/technical/benchmark.md` § Sapiens v1); finalists unchanged.
- [ ] **Corpus rerun** (RTMW-L 384 done → `output/corpus-2d-m4-rtmw-l-384`, 193/193, clinical stage
  pre-bouts; RTMW-m 256 not started → move `.scratch/impl/corpus-rtmw-l-384.log` aside, relaunch
  `.scratch/impl/gpu_queue3.sh` (~8-10 h); code pinned at worktree `.scratch/worktrees/corpus-m4`
  (`48078b4`) → `--refresh-clinical` after each; ordered next steps → `.scratch/impl/checkpoint.md`)
  — cohort republished; after-table vs M2.10; `scientist`.
- [ ] **Closing review** — one `reviewer` per lens + MAIN's security lens → `.agent/review.md`.
- [ ] **Close** — README, review UI → `tools/review-ui/`, `Phase: MAINTAIN`. Met when: full gate (scanners
  included) passes on a clean tree; every `Decisions` entry implemented or deferred with its reason;
  `.agent/review.md` holds no open row.

Evidence → `.agent/archive/{polish,review-m2}.md`; regen → each check's `paths:` rule in `.claude/rules/`; queue → `.agent/deferred.md`.

## Phase

IMPLEMENT — video analysis (2D pose, features, cohort export to `../rehab`)

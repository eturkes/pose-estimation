# Spec

## Intent

Turn the hospital's uncontrolled three-camera recordings of spinal-cord-injury rehabilitation
tasks into measured movement statistics that feed the `../rehab` clinical dashboard over the
hospital SCI database.

- Corpus = `videos/3-cam/`: 382 hand-held clips, 16 subjects × 6 tasks × 2 sides, three views per
  family at best, no fixed rig, no calibration.
- The pipeline tracks 2D pose per clip, extracts clinical kinematic features, and aggregates to
  cohort level — no per-subject rows, no patient identifier, no join column leaving this repo.
- The shareable review UI is itself an intended deliverable for collaborators, not only a
  prototype step.
- Eventual destination = a subject↔patient join into the hospital SCI database; that mapping does
  not exist yet.

## Artifacts

`P` = the gate prefix; `P` + the mutually-exclusive accelerator recipe → `.claude/rules/gates.md`.

- `prototype/review-ui/` — **the prototype, the inspectable artifact.** One local web UI,
  bilingual ja/en (`?lang=en`), themed auto|light|dark (`?theme=`, auto = `light-dark()` + the
  OS), three views, clip player (the landing view) · corpus census · cohort explorer.
  FastAPI + vanilla JS canvas + vendored Plotly/IBM Plex, own uv project, read-only over the
  published trees, degrading per absent tree. Commits no media at all → the player needs the
  published trees and lists nothing without them; `README.md` = view guide + regeneration + limits.
  `uv run --directory prototype/review-ui python -m review_ui` → `http://127.0.0.1:8791/`.
- `cohort/` — the `../rehab` export. 12 `(task, side)` cells · 89 features · 1068 rows;
  `descriptors.yaml` = ja/en labels, units, ranges.
  `P pose-estimation-cohort --inventory inventory --sessions sessions --run output/corpus-2d --out cohort`
- `output/corpus-2d/` — per-asset 2D landmarks + clinical features, 379 assets / 193 events /
  331 152 frame rows, 7.828 h under the accelerator recipe. `python scripts/corpus_run_2d.py`.
- `inventory/` `sessions/` `qualification/` `calibration_qc/` — four publishers upstream of the
  run; each `P pose-estimation-<name> … --out <dir>`; `--help` = args.
- Decisive gate — `P pytest`, 1750 tests, 14-25 min, alone.

## Decisions

- **Repo scope = `videos/3-cam/`** — retired data + siblings → `.claude/rules/data-boundary.md`.
- **`src/` + `tests/` + `analysis/` + six publishers = production spine**, gates + verification
  integrity binding; the prototype (`prototype/review-ui/`, → `Artifacts`) sits outside under
  PROTOTYPE law, `testpaths = ["tests"]` keeping it out of collection.
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
- **Detector on CPU, pose on NPU** — NPU pads YOLOX's dynamic output with uninitialised memory
  that passes every validity filter as real rows.
- **Acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`** — 7 files under
  `scripts/ src/ tests/` break if it moves, one a generated data field no gate resolves;
  re-derive → `upstream-sync.md`.
- **External AI-judgment services stay out of this repo** — surveyed and declined by the user.
  The three defects the survey measured live on as `.agent/deferred.md` rows with mechanism left
  open; the survey itself is not repeated.
- **Assurance tier = `kernel`** across pipeline, publishers + analysis.
- **2D instability repair = `SubjectTracker`** — every crop a detector box, never re-sized from
  the pose; detector every frame on GPU at f32 (41.6 ms/call, bit-identical to CPU on 240 corpus
  frames); box transported by pose motion across detector misses; sticky largest-box subject;
  identity-keyed smoothing. Supersedes the *reconcile* subclass (user ruling, `.agent/archive/instability-m2u9.md`)
  whose premise was the CPU detector's 8.23× cost for `det_frequency=1`; that ruling's acceptance
  (seven-arm re-sweep, alternation <= 0.55, isolated ~0, no cadence peak, wall near f7's 327 s)
  stands → `.agent/archive/contract-m2u91.md` P14-P17.
- **No low-pass filter stage** — `det_frequency=1` alone reaches alternation 0.495 against the
  post-hoc Hampel+6 Hz stage's 0.483 while keeping 0.697 of the 2-5 Hz band against its 0.464.
  Removing the noise at source is what makes filtering unnecessary; a 6 Hz cutoff on top would
  destroy tremor-band energy the source no longer carries as noise, and SPARC is cutoff-sensitive
  so it would move the smoothness feature too.
- **SPARC becomes the primary smoothness feature, `normalized_jerk` demoted to secondary** —
  shipped in the instability rerun, since `cohort/` republishes anyway.
- **Fix upstream only** — the pipeline changes and the corpus reruns, so `output/corpus-2d/` stays
  the single source of truth and no post-hoc filter bolts onto published tracks.

## Tasks

- **Resume note — pose-quality session (user body: watch the review UI → catalog every situation
  where pose estimation performs poorly → repair each → rerun all videos in one pass; no
  questions).** Finish line = catalog committed; each situation repaired under spine law
  (`kernel`: contract, red witness, `tester`, `reviewer`, gate green) or reported as a failed
  attempt with what it taught; corpus rerun 193 events / 379 assets in ONE pass →
  `output/corpus-2d/` + `cohort/` republished (SPARC primary); review UI serving the new tracks;
  closing commit on a clean tree. Catalog → `docs/technical/pose-weak-points.md`; instruments +
  roster → `.scratch/tasks.md`.
- [x] 397cc8a **Review UI overlay time map (C01)**
- [x] 481ba22 **M2.9.1 subject tracker (C02 C03 C08 C11)** — contract `.agent/archive/contract-m2u91.md`;
  diff-blind `tester` → own red witness → implementation → `reviewer` → gate → seven-arm re-sweep
  (P14-P17). Mechanism law → `.claude/rules/rtmlib-runtime.md`.
- [ ] **M2.9.2 output hygiene (C05 subset, C07, C10)** — contract
  `.agent/archive/contract-m2u92.md`: out-of-frame points score 0, duplicate hand suppressed,
  face landmark sides corrected. General occlusion hallucination (legs under the table, hips
  from above) is not separable by score — hallucinated knee median 0.48 vs real wrist q25 0.49
  (right views) — so it stays a reported model limit.
- [ ] **M2.9.3 hands (C09) — evaluated twice, not adopted.** RTMW-X scores saturate; ungated
  hand-crop refinement moves the confidence scale + collapses occluded hands; the gated version
  (user-requested) removes the collapse but misses jitter <= 0.006 (0.0079) at +41 % wall →
  catalog § M2.9.3, queued.
- [ ] **M2.9.5 seated body-part drop (user ruling)** — contract
  `.agent/archive/contract-m2u95.md`: knees, ankles, feet score 0 everywhere; hips score 0 under
  the overhead camera.
- [ ] **M2.9.4 SPARC** — contract `.agent/archive/contract-m2u94.md`: the R stage's fixed-cutoff
  SAL becomes SPARC (adaptive cutoff, Balasubramanian 2015), method version v3; runs while the
  corpus decodes, then an R-only pass over the new tree.
- [ ] **Corpus rerun** — one pass 193/379 under the repaired pipeline → R pass with SPARC →
  `cohort/` republished → affected determinism campaigns → decisive gate → review UI restart.
- [ ] **Review UI JP subset builds from gitignored `cohort/descriptors.yaml`**
  - Acceptance: `build_assets.py` refuses with a named cause when absent; a committed check reports
    0 missing code points.
- [ ] **Overlay landmark->pixel map proven by eye alone**
  - Acceptance: a headless check injects fabricated landmark coordinates into the served player and
    grades the drawn pixel positions against the canvas scale math, max deviation < 1 px, failing
    on a seeded off-by-one. Needs no committed media.
- [ ] **HEVC decode failure reported but never exercised** (123/379 hevc)
  - Acceptance: an hevc clip on a decoder-less build shows the `player.decode_failed` banner.
  - The rAF advance itself is proven by `.scratch/player_clock_qa.mjs` (→ `gates.md`); the banner
    half is what remains.

Evidence → `.agent/archive/{polish,review-m2}.md`; regen → `.claude/rules/gates.md`.
Queue → `.agent/deferred.md`.

## Phase

**ITERATE — review UI** (`prototype/review-ui/`, the one prototype in `Artifacts`). The production
spine ships beside it under its own gates; spine rows in `Tasks` (the instability repair) close
under those gates, never under PROTOTYPE law. IMPLEMENT of the review UI starts on the user's go.

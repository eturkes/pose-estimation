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

- `prototype/review-ui/` — **the inspectable artifact.** One local web UI, bilingual ja/en
  (`?lang=en`), themed auto|light|dark (`?theme=`, auto = `light-dark()` + the OS), three views,
  clip player (the landing view) · corpus census · cohort explorer.
  FastAPI + vanilla JS canvas + vendored Plotly/IBM Plex, own uv project, read-only over the
  published trees, degrading per absent tree. Commits no media at all → the player needs the
  published trees and lists nothing without them; `README.md` = view guide + regeneration + limits.
  `uv run --directory prototype/review-ui python -m review_ui` → `http://127.0.0.1:8791/`.
  Proof → `proof/`: 4 captures (3 light + 1 dark, each theme-pinned) + API transcript, by
  `tools/capture_proof.py`. The player view takes no capture — its stage is patient video.
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
  integrity binding; `prototype/` sits outside under PROTOTYPE law, `testpaths = ["tests"]`
  keeping it out of collection.
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

## Deferred

Queue → `.agent/deferred.md`; evidence → `.agent/archive/{polish,review-m2}.md`; regen →
`.claude/rules/gates.md`. The unfinished units = ITERATE's spine, each closing on its own check:

- **Review UI JP subset builds from gitignored `cohort/descriptors.yaml`** → `build_assets.py`
  refuses with a named cause when absent; a committed check reports 0 missing code points.
- **Overlay landmark->pixel map proven by eye alone** → headless check injects fabricated landmark
  coordinates into the served player and grades the drawn pixel positions against the canvas scale
  math, max deviation < 1 px, failing on a seeded off-by-one. Needs no committed media.
- **HEVC decode failure reported but never exercised** (123/379 hevc) → an hevc clip on a
  decoder-less build shows the `player.decode_failed` banner. The rAF advance itself is now proven
  by `.scratch/player_clock_qa.mjs` (→ `gates.md`); the banner half is what remains.
- **2D landmark instability — cause found, repair not yet built. THE NEXT UNIT.**
  The source is rtmlib's **pose→box→crop→pose feedback loop**: between detector calls the box is
  pose-derived (`bboxes_last_frame = pose_to_bbox(kpts)`), and every `det_frequency` frames the
  detector replaces it. Mixing the two box sources is the defect — when they disagree the crop jumps
  and the top-down model returns a *different skeleton* (509 px translation **plus** 321 px
  deformation, against 2.0/5.1 px for ordinary motion). It injects a periodic artifact at
  `fps/det_frequency`, and the shipped `det_frequency=7` lands it at **4.29 Hz — inside the clinical
  2-5 Hz band and inside the 1.9-5.8 Hz intention-tremor band**. Raising `det_frequency` only moves
  the artifact into 0-2 Hz gross transport. Seven-arm sweep, per-arm statistics, spectral evidence
  and the population split → `.agent/deferred.md`; mechanism law → `.claude/rules/rtmlib-runtime.md`.
  Three user rulings bind the repair:
  - **Fix the loop in code first, then re-sweep** — subclass `PoseTracker` so a detector frame
    *reconciles* its box against the pose-derived box instead of replacing it. Target = the quality
    `det_frequency=1` reaches (alternation 0.495, zero isolated relocations, no cadence) at
    something near `det_frequency=7` cost, rather than paying 8.23× = ~64 h for the corpus rerun.
    `kernel` tier: acceptance contract + diff-blind suite + gate-green before any corpus rerun.
  - **No low-pass filter stage** — `det_frequency=1` alone reaches alternation 0.495 against the
    post-hoc Hampel+6 Hz stage's 0.483 while keeping 0.697 of the 2-5 Hz band against its 0.464.
    Removing the noise at source is what makes filtering unnecessary; a 6 Hz cutoff on top would
    destroy tremor-band energy the source no longer carries as noise, and SPARC is cutoff-sensitive
    so it would move the smoothness feature too.
  - **SPARC becomes the primary smoothness feature, `normalized_jerk` demoted to secondary** —
    shipped in the same rerun, since `cohort/` republishes anyway.
  Standing ruling, unchanged: **fix upstream only** — the pipeline changes and the corpus reruns, so
  `output/corpus-2d/` stays the single source of truth and no post-hoc filter bolts onto published
  tracks. Closes gate-green with the corpus and cohort republished.

## Phase

**ITERATE.** `prototype/review-ui/` runs by its recorded command, proof under `proof/`.
IMPLEMENT starts on the user's go.

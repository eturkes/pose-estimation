# Deferred

Off-spine queue. One row = one improvement + the acceptance check that closes it, written at
deferral time while the evidence is fresh; a row leaves only when that check passes on a commit.
Read on demand — not attached state. A row promoted to an unfinished unit moves to `.agent/spec.md`
`Tasks` as a `- [ ]` row, and returns here when the spine moves past it.

Evidence → `.agent/archive/{polish,review-m2}.md`; regen → each check's `paths:` rule in `.claude/rules/`. Read the
rows before sizing any unit that touches their surfaces.

- **Four `../rehab` JP faces miss 17 cohort-label characters** (v3 `cohort/descriptors.yaml`, 88
  labels, 93 distinct `ja` characters; `チペ偏停勢姿尖幹扇把斜旋止称肘路隔` on all four weights) →
  each reports 0 missing code points over the `cohort.FEATURES` `ja` union. Owed to the user (edit
  outside this repo): `../rehab` `scripts/font_charset.py` scans `schema/`, so appending the
  fragment to `schema/columns.yaml` and rerunning `scripts/02_build_fonts.py` (npm fetch) covers them;
  `../rehab` `tests/test_font_coverage.py` then pins it.
- **`steq.py` + `fidelity.sh` port; 24 register flags on four human surfaces at `--max 25`** → one
  committed scanner reports 0 `FILLER` over the `conventions.md` inventory and lists every
  `LONG`/`CONTRACTION`/`PASSIVE` flag as a lead for judgment (no length cap), fails a seeded
  `simply`, no specifier/flag/number delta.
- **Occluded far hand blinks under the duplicate rule** (right-view blink-outs 0.46 → 3.27 per
  100 rows, 46 % on or near the other hand; `.scratch/flick_cause.py`) → over the watch set every
  view reads <= 1.0 blink-outs per 100 rows while exported frames with one hand's median point
  distance to the other < 0.3 of hand extent stay 0, and each clip's task-hand presence share
  stays within 0.02 of the M2.9 baseline (`.scratch/watch/triage_m29.json`), so an all-absent
  candidate fails. Mechanism open (hysteresis, occlusion-aware presence).
- **Gated hand-crop refinement misses the adoption bar** (`docs/technical/pose-weak-points.md`
  § M2.9.3: jitter 0.0079 vs <= 0.006, wall +41 % vs <= +15 %, right views worse) → a variant
  (batched crops, refinement only where the whole-body hand is small or unstable) reaches jitter
  <= 0.006 on the watch set at <= +15 % wall with collapse unchanged on every clip.
- **`run.main --source` writes no per-source diagnostics; P07 attribution lives in
  `.scratch/impl/p07_stage.py` alone** (it wraps `run.process_source` to get the
  `exported_out_of_frame_zeroed` census; `docs/technical/pose-weak-points.md` § M4.6) → the
  `--source` path writes `<stem>_diag.csv` beside its CSV (a test asserts the census fields on a
  synthetic clip), and `bench.py` reproduces the P07 table from committed code.
- **2D weak-point triage lives in `.scratch/triage.py` alone** (→ `pose-instruments.md`; behind
  `docs/technical/pose-weak-points.md`) → a committed `scripts/` instrument with no
  `tools/review-ui` import reproduces the catalog's per-view counts over a run tree and flags a
  seeded whole-skeleton relocation in a synthetic landmark CSV. `whole`/`iso` print each clip's
  eligible all-ten-point transitions, and a clip with none reads unavailable, never 0 (today
  `max(obs, 1)` prints 0 on 278/379 RTMW-m M4 clips, every overhead one; `scientist-2` F1).
- **Detector scores outside `[0,1]` accepted silently** → score-range guard fires on NPU, silent on
  CPU + GPU, under the synthetic zeros probe.
- **`pose-estimation-run --session-dir … --output-dir X` ignores `X`** → rtmlib session artifacts
  land under the requested root; MediaPipe unchanged.
- **`sessions.py` predicates pinned by tests alone** → a campaign mirroring
  `run_inventory_mutations.py` kills every mutant with >=1 committed test, replaying from clean.
- **Review UI theme control proven by a scratch script alone** (`.scratch/theme_qa.mjs` → regen path
  in `.claude/rules/review-ui.md`) → a committed check drives the cycle, the pre-paint rehydration and
  the `:has()` stage backdrop, 13/13 green, failing under the two recorded seeds. The scratch
  rehydration row (`stored dark is applied at document commit`) read red 2 of 6 runs under a
  concurrent full gate with `index.html` + `app.js` unchanged → the port reads it 10/10 green under
  the same load.
- **Review UI player fit proven by a scratch script alone** (`.scratch/player_layout_qa.mjs` → regen
  path in `.claude/rules/review-ui.md`) → a committed check reports the stage fitting its room, filling
  an axis and holding its ratio over five viewports plus the stacked breakpoint, every control row
  on screen, and reds under the four recorded seeds.
- **Review UI landing view + fragment routing proven by a scratch script alone**
  (`.scratch/default_view_qa.mjs` → regen path in `.claude/rules/review-ui.md`) → a committed check
  reports the player first in the tab strip, the bare URL settling on `#player` with its stage drawn,
  a `#cohort` deep link and an unknown fragment resolving, and the same-document group — a
  `location.hash` write, the Back button, an unknown same-document hash — 16/16 green, failing under
  the four recorded seeds, the Back row still firing under a 1500 ms `/api/cohort` delay.
- **Review UI overlay lines use raw `topology.json` colours, the pipeline renders them blended**
  (`drawing.py` `_LIMB_ALPHA = 0.6` → `_BODY_BLENDED`/`_HAND_BLENDED`; `export_topology.py` exports
  the unblended map) → the README's "the overlay draws what the pipeline draws" is unearned for
  line colour. Predates the theme unit. Acceptance: a differential check reports
  `body_line_palette_mismatches=0/6` + `hand_line_palette_mismatches=0/6` against the blended maps,
  or the claim is narrowed to topology and group identity with the blend stated.
- **Chart font gate proven by capture comparison alone** (`static/app.js` `FONTS_READY` defers every
  `Plotly.newPlot` until the Plex faces load; measured 3/4 captures agreeing before, 6/6 after) →
  a headless check reports the census rotation legend at 2 rows with the faces loaded, and reds when
  `newPlot` runs ahead of `document.fonts.load`.
- **Claim boundary guarded by 18 substrings; 7/7 semantic overreaches pass** (`check_claim_report.py:89`
  `_p03`, needles = `calibration_qc.PROHIBITED_PARAPHRASES`; measured against the shipped `_fold`,
  positive control `clinical validity` fires) → the guard refuses a held-out paraphrase set covering
  all three forbidden classes, including one Japanese rendering, at 100 % recall on overreach, passes
  the two shipped documents unchanged, and reports an explicit third state for a sentence it cannot
  decide instead of approving it. Mechanism is open — a widened needle set, a pattern family or a
  semantic judgment all qualify; the seed set is what binds. Seeds, each measured passing today:
  "These measures are clinically meaningful for rehabilitation outcome." · "The pipeline is
  validated against clinical outcome scores." · 「本手法の臨床的妥当性は確認されている。」 ·
  "Agreement with marker-based motion capture is established." · "The angles match a gold-standard
  optical system." · "Distances are reported in true millimetres." · "This calibration is accurate
  to sub-pixel precision in world units." `capture-docs.md` already records the tell: "NC4's needle must
  come from `PROHIBITED_PARAPHRASES` rather than invented prose." — invented prose does not fire.
- **`check_review_report.py` P04 + P06 are spelling tests** (`:53`, `:77-80`) → a report whose `pass`
  cell reads `aaaaaaaaaaaaa` and whose detail reads `predicate impact acceptance x.py:1` grades
  nonzero; the 57 rows of `.agent/archive/review-m2.md` still grade `PASS`. Today the hollow report
  grades `PASS` rc=0 and a one-character shortening of the `pass` cell is the only thing that reds it.
- **Review UI feature finder is a three-field substring filter** (`static/cohort.js:212`; searchable
  English vocabulary = 41 words over 89 features) → a committed check reports each seed query
  retrieving its intended feature, with no loss on the literal queries substring search already
  answers and `ja` graded separately. Mechanism is open — a synonym table, stemming or a semantic
  ranker all qualify. Seeds, 12 of 13 returning 0 of 89 today: `smoothness` (wants `jerk`,
  `spectral`) · `speed` (wants `velocity`) · `asymmetry` (wants `symmetry`) · `range of motion` ·
  `trunk compensation` · `grip` · `coordination` · `tremor` · `movement quality` · `how fast` ·
  `how steady is the reach` · `変動`; `左右差` is the one that hits, on 15, as a literal template
  fragment.
- **Player clock rate + overlay time map are proven by scratch scripts**
  (`.scratch/player_clock_qa.mjs`, `.scratch/player_time_qa.mjs`; → `review-ui.md`) → port both to
  committed checks under `tools/review-ui/tools/`, run from the recorded commands: clock 10 rows
  green with the seeded `view.framePos`→`view.frame` regression reding exactly the 6 rate rows while
  both `layer=show` rows stay green; time map 7 rows green with the series-index seed reding the 5
  map rows and the seek-less `|▶` seed reding the step row alone.
- **Review UI stacked-layout stability proven by scratch scripts** (`.scratch/drift_qa.mjs`,
  `.scratch/ratchet_qa.mjs`; → `review-ui.md`) → committed checks under `tools/review-ui/tools/`:
  drift 22/22 with a stacked `1fr` column reding exactly the 4 stacked injection rows; ratchet
  60/60 across scales 1/1.1/1.25/1.5 with stacked `auto auto` rows reding only stacked rows.
- **Marksman reports 4 links to existing docs as non-existent** (`docs/calibration_finding.md:9`,
  `docs/technical/entrypoints.md:188,223` → `technical/{calibration_qc,qualification}.md`,
  `README.md:52` → `docs/technical/cohort.md`; git ignores none of the targets). Suspected cause,
  unconfirmed: its index reads the `.gitignore` directory patterns `qualification.*/` +
  `calibration_qc.*/` + `cohort.*/` as matching the `.md` files →
  Marksman diagnostics over `docs/` + `README.md` report 0 non-existent-document warnings, and
  `git check-ignore` still ignores a `qualification.x/` staging directory.
- **`.scratch/ovupgrade/` build-change instruments are scratch-local** (`pilot.sh`, `compare.py`,
  `det_qual.py` → `pose-instruments.md`) → committed under `scripts/` with a test that seeds a one-cell CSV
  drift and asserts `compare` reports it, and a same-tree run reports 0 drift.
- **M2.10 watch-set graders are scratch-local** (`.scratch/messy/` + 972 labels; regen →
  `pose-instruments.md`) → a committed `scripts/` grader with a synthetic
  label fixture reproduces the gate's held-out drop shares (hallucinated 0.90, real 0.037) and the
  span's dense setup-excluded / task-lost figures from committed inputs, and reds on a seeded
  threshold swap. Labels stay uncommitted (patient-adjacent ordinals + frames).
- **Wrong-person residue (C02/C04 after M2.10)** — 17 labelled task frames in 9 clips carry the body
  on another person or mixed; worst = a therapist leaning over the patient (larger box) and limb
  lines drawn to another arm; selector picks the patient's box on 459/473 (0.970) → a candidate
  (patient anchor + promptable tracker of the DAM4SAM/SAM2 class, part-aware re-ID, or a
  mask-conditioned pose model such as BMPv2/PMPose; `.scratch/agents/researcher-2.md`) raises the
  box hit share to >= 0.99 on the same 473 frames and halves the `o`/`x` body frames, held out by
  family, without lowering hand `k` retention.
- **Pose latency runs 3-5× slow on some assets, cause unmeasured** — paired probes on one day: 1-3 of
  11 assets at 130-274 ms/frame mean in BOTH arms (survey on and off); the M2.10 corpus: median
  51.4 ms, 10/379 assets above 100 ms (M2.9: 50.0 ms, 0/379); AC power, `performance` profile, no
  GPU throttle flag at idle (`.scratch/messy/latency/probe_*_assets.json`) → one measurement names
  the stalled stage (detector GPU, pose NPU or decode) on a slow asset, and a rerun of that asset
  reads <= 70 ms/frame after the named fix or the cause is recorded as host-side.
- **M2.10 gate + span graded on the watch set alone (`scientist-m210-1` S01/S03/S07)** — the watch
  set samples problem clips, so corpus-scale real-hand loss, false-hand rejection and missed setup
  are unmeasured; rival = visibility-dependent rejection excluding severe impairment or
  therapist-assisted posture from hand features → a stratified random sample outside the watch set
  (frozen codebook, view × task × task/other hand, subject-clustered intervals) reports the three
  rates separately, and the gated share among real hands per impairment stratum where the SCI join
  later allows it.
- **Cohort shifts M2.9 → M2.10 not factorized by mechanism (`scientist-m210-1` S10)** → offline arms
  over the M2.9 rows (baseline / + gate / + trim / + compensation, no new inference) attribute each
  feature family's median shift, fixed-support and available-support, and the residual on the
  distinct-score P08 clip is named.
- **Hand gate state is keyed by smoother track, presence judged on the exported subject (M2.10.1
  P08 at corpus scale: 13/379 clips beyond 0.01, max 0.766)** — a transient second track takes the
  exported row while the subject's own track carries a handless row, which switches that hand off;
  back on the exported row, 0.50-0.65 levels hold it off (`pose-weak-points.md` § M2.10 corpus;
  measured on 1 clip with `.scratch/messy/p08_keys.py`; 11 share its signature via `p08_sign.py`,
  undiagnosed; 1 exports scores that differ from M2.9, cause unmeasured). M4.6 keys the
  single-subject gate on the selected row: the RTMW-m M4 tree reads 3/379 clips beyond 0.01, max
  0.049 (`.scratch/impl/p08_m256.log`), residue mechanism unmeasured (OpenVINO build drift in the
  scores is one candidate; a replay of recorded pre-gate scores under old + new keying in one
  runtime separates the two, `scientist-2` F7) → the gate keys on the
  exported subject's identity (or carries state across the transient), `p08_compare.py` over a
  corpus rerun reads <= 0.01 on every clip, the held-out drop shares (`gate_flicker.py`: hallucinated
  >= 0.90 removed, real <= 0.06 lost) hold, and a synthetic two-track fixture with a handless
  subject row reds on the current keying.

- **`test_m2u5_x10` grades a `docs/technical/qualification.md` sentence as human-facing; the
  `conventions.md` *Text register* inventory marks `docs/technical/` agent register** → one
  register holds for that sentence: the inventory lists the surface the test grades, or the test's
  name + docstring claim no human-facing register; `rg -n 'human' tests/test_qualify.py` and the
  inventory agree.
- **No Python mutation tool in the stack** (`pyproject.toml` + `.venv` carry none; the
  `scripts/run_*_mutations.py` campaigns hand-pick their mutants) → `CLAUDE.md` `Engineering` credits a
  test pinning already-passing behavior only through the stack's mutation tool, so none earns that
  credit today. Acceptance: a researched tool in the dev dependencies runs from committed state with
  default operators over the files one pinning test imports or calls, its scope, command and every
  survivor recorded, each survivor ruled against the contract by a non-author (missing test added,
  open behavior recorded, else an open row).
- **Review UI predates the `CLAUDE.md` `Execution` UI design + Charts + figures and `Authoring`
  UI-copy clauses** (known gap: the census rotation + sync panels, `static/census.js`
  `rotationPanel` + `syncPanel`, are interactive charts with no table view) → an audit table over
  `tools/review-ui/` names each generated-UI default the clause lists (` · `-joined meta
  strings, monospace data labels, `→` on links, tracked ALL-CAPS eyebrows, numbered markers,
  identical shadowed cards, per-section fades), each chart rule (form by the data's job, one
  y-scale per plot, color by job, legend + direct labels, a table view per interactive chart) and
  each UI-copy rule (user-world names, outcome-named controls kept through the flow, errors with
  cause + fix, inviting empty states) with a `file:line` verdict; every hit is replaced or kept by a `.agent/spec.md` `Decisions` ruling; the player view
  is judged through the blanked stage alone (`data-boundary.md`).
- **No palette-safety check on any figure** (`CLAUDE.md` `Execution` Charts + figures: review-UI Plotly
  palettes in `static/{census,cohort,app}.js` + `style.css`; ggplot figures in 11 `analysis/*.R`
  files matching `ggplot\(|palette|scale_(colou?r|fill)_`) → a computed check (Machado protan +
  deutan simulation, OKLab ΔE×100 >= 8 between adjacent series — all pairs for scatter — >= 15
  unsimulated, marks >= 3:1 against the surface) runs in the
  gate over every committed palette, passes or names each failing pair, and reds on a planted
  near-duplicate pair recorded in a `paths:` rule on the check's files.
- **Census Synchronisation legend repeats hues across categories** (ITERATE close capture,
  `#census`: Visual only + Solved share one red, Corroborated + Unconnected one blue;
  `static/census.js` `syncPanel`) → every legend entry in one panel carries a distinct hue
  (OKLab ΔE×100 >= 15 between any two), or categories of different axes split into separate
  legends; the palette-safety row's check covers the panel.
- **Cohort Cell composition table clips its last column** (`#cohort` at 1280 px: `Window rows`
  header cut, no horizontal scroll; `?lang=ja` also wraps each task name onto two lines) → at 1280 px and 1600 px every header + cell of the table is
  fully on screen or its container scrolls horizontally with the overflow visible.
- **Two wrist segmenters in `clinical_features.R`** — `segment_movements` (phases, 5 % of clip
  max, NA speed as 0, unexported) beside the qualified `detect_bouts` (M4.5a) → phases read
  `detect_bouts`' bouts, `*_movement_phases.csv` keeps its schema, and a hole inside motion no
  longer reads as rest (the M4.5a NC2 fixture splits into censored bouts in both artifacts).
- **Score calibration transfer to this domain is unmeasured** (`scientist-2` F4) — RTMW-L 384's
  reference-equivalent map is fit on COCO-WholeBody; at the 0.3 floor it admits 317 109 matched
  frame×point observations RTMW-m holds below it (12 166 the reverse), temporally continuous but
  0.028 median from RTMW-m's position → on development labels outside the M4.4 set, correctness +
  presence per score bin × view × joint for candidate-only vs shared observations shows the
  candidate-only bin within the shared bin's error at the same score, or the floor is refit per view.
- **Bout-level features move with segmentation, not only with values** (`scientist-2` F6) —
  RTMW-m → RTMW-L 384: bout rows 6524 → 6185, 20/24 bout-duration medians rise, SPARC medians fall
  in 18/24 cells → bouts frozen by one arm-independent segmentation, both arms' SPARC, NJ, speed +
  peak medians compared on identical bouts; the report attributes each family's shift to values vs
  boundaries.
- **The export carries no per-feature measurement status** (`scientist-2` F8) — validity,
  reliability + units live in README limits alone → `descriptors.yaml` (or the cohort columns)
  states per feature: unit class (image-plane angle, dimensionless, time), criterion validity
  `not established`, reliability `not established` (never filled from cross-view ICC or model
  agreement); a test reds on a feature missing either field.
- **Severe steps: mechanism unmeasured** (`scientist-2` F5, `docs/technical/pose-weak-points.md`
  § M4 corpus) — severe point-steps (> 0.25 of the max frame dimension between adjacent frames on
  the ten used points, both ends vis >= 0.3) per million valid: RTMW-L 384 162.7, RTMW-m 106.6;
  on shared support 104.2 vs 98.4, on RTMW-L's own extra points 578.4; 79 of RTMW-L's 165 severe
  transitions coincide with RTMW-m's; 84/6185 RTMW-L bouts carry one on their own wrist or index
  → with the user's per-session word, frame pairs of a stratified sample are classified (pose
  re-interpretation, identity, crop, content, low-score addition); an upstream fix of the dominant
  class halves both `jump_count.py`'s per-million rate and the bouts carrying an own-side severe
  step on a rerun tree, with `whole` / `iso` / `alt` no worse (`.claude/rules/pose-instruments.md`).
- **A symlinked `--out` naming the baseline root fails to publish** (closing delta round A,
  `reviewer-20` R01; also at `9052629^`) — `cohort.run` checks overlap on real paths but renames
  the unresolved link over a directory → `IsADirectoryError`; `--out` = a symlink to the v3
  baseline root publishes at the resolved root, keeps the link, and validates (the reviewer's
  `test_p11_equal_root_output_symlink_publishes_at_resolved_root`, `.scratch/reviewer-20/`).
- **`check_calibration_qc_fixtures.py` P12 is a spelling test** (`scripts/check_calibration_qc_fixtures.py:363`;
  `general-purpose-1` R05) — it greps the generator source for its default destination name and never
  runs the foreign-destination refusal its name claims → P12 runs the generator against a non-empty
  foreign directory and requires the refusal, and reds on a generator whose ownership check is
  removed.
- **The decisive gate's pytest step has no equal-strength speedup** (`.claude/rules/gates.md` names
  this row; serial: 3142 cases in 1358 s alone, 1517 s at IMPLEMENT close, longer under co-tenant
  load; every commit pays it) → a parallel runner (e.g. pytest-xdist) runs the same collection with
  A32 reconciling every case passed + zero skipped across workers, `test_c8_08`'s planted
  skip/deselect seeds still red, no R-subprocess case timing out under the parallel load, and the
  pytest wall on this machine below half the serial figure measured beside it.
- **Two `.scratch/` teammate reports stay tracked against `.gitignore`** (revives
  `.agent/archive/polish.md` row 3; template-refresh portability sweep) — `git ls-files .scratch` =
  `.scratch/agents/{scout,test}-m2u4.md`, entered through M2.4 squash merges; `test-m2u4.md:135`
  carries the primary tree's absolute path, the one home path left in a tracked file outside the
  ruled `docs/technical/environment.md` + `.envrc` and frozen `.agent/archive/`;
  `tests/test_r_timebase_truth.py:3` cites it →
  `git ls-files .scratch` empty with both files still on disk and that docstring naming the ruling
  (`.agent/archive/contract-m2u4.md` §8) in place of the report; or each retained file records one
  named open dependency (`retention.md`); full gate green.
- **No committed check keeps machine paths out of live law** (`CLAUDE.md` `Authoring` portability;
  the template refresh found 7 literal accelerator paths by hand-run sweep) → a decisive-gate test
  scans tracked live-law files (`.agent/` minus `archive/` + `review.md`, `.claude/rules/`, `docs/`,
  `scripts/`, `src/`, `tests/`, `tools/`, `README.md`) for home + host paths (`/var/home/`,
  `/run/host/`, `/home/<user>/`, `~/`), allowlisting the sites `gates.md` rules (host/container
  layout + its description there), vendored `renv/activate.R` and pattern text in this row; reds on
  a literal home path planted in a copied rule, greens on the tree, its firing recorded in a
  `paths:` rule on the test.
- **`test_p05_temporal_panel_names_sparc` reds the decisive gate since c4f7231** (template-refresh
  full gate: 1 failed / 3194 collected; same red at f98b850 in a worktree; green at 6d3ab19) —
  c4f7231's `discover_clinical` stops a two-mode tree without `--mode` by design, and the test's
  fixture dir holds `synthetic_hands-arms.csv` + `world3d.csv` while its `commandArgs` mock passes the
  directory alone → `Pass --mode 2d or --mode 3d.`, rc 1. Blocks the MAINTAIN finish line → fund
  first → test green with its SPARC panel-name assertions unchanged, the harness passing the mode the
  contract requires (the two-mode refusal stays: `tests/test_r_consumers.py` refusal cases green);
  full gate green.

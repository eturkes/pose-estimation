# Deferred

Off-spine queue. One row = one improvement + the acceptance check that closes it, written at
deferral time while the evidence is fresh; a row leaves only when that check passes on a commit.
Read on demand — not attached state. A row promoted to an unfinished unit moves to `.agent/spec.md`
`Tasks` as a `- [ ]` row, and returns here when the spine moves past it.

Evidence → `.agent/archive/{polish,review-m2}.md`; regen → `.claude/rules/gates.md`. Read the
rows before sizing any unit that touches their surfaces.

- **`isotropy_angle_fidelity_results.json` pins sample ids alone** (T2 R23) → binds script,
  validated-registry generation, run-tree + config digests.
- **Cohort census omits the external descriptor digest** (T4 R37) → census records it;
  `descriptor_collision` re-checkable.
- **Two census digests skip their own provenance block** (T5 R17, `sessions.py:588`,
  `measure/__init__.py:281`) → each covers its block minus its self-referential key.
- **`inventory_mutation_results.json` + `measure_mutation_results.json` target moved bytes** (2/2,
  2/3) → each campaign runs from clean checkout, no target-mismatch refusal, reproduces its kill
  list, and exits 1 on a seeded survivor with `M028` the sole allowlisted equivalent — today
  `run_inventory_mutations.py` returns 0 whatever survives (their only firing → `gates.md`).
- **Six legacy R clinical consumers re-implement suffix/read/bind** → one mode-aware reader; a
  mixed 2D+3D tree yields named mode outputs or rejection, 2D goldens unchanged.
- **`make_templates.R` + `validate_metadata.R` skip `_3d` silently** → both route through that
  reader; 2D templates + validation unchanged.
- **Four `../rehab` JP faces miss 14 cohort-label characters** → each reports 0 missing code points
  over the `cohort.FEATURES` `ja` union.
- **Gate prefix recalled by failure** → `scripts/gate.sh pytest -q` collects in primary tree + fresh
  worktree, bare `uv run pytest` still reproduces `GLIBC_2.43`, 0 bare invocations in `.agent/spec.md`
  + `.claude/rules/` + `docs/` + `scripts/` (`.agent/archive/` frozen, keeps its bare forms).
- **12/12 `check_*.py` ship green-only; `run_measure_mutations.py` firing unrecorded** (census +
  acceptance → `gates.md`) → each checker grows a committed seed driving it to its own named
  failure; a runner reports 12/12 refused, rc=1 naming one that stops. Covers the
  `nc_m2u74.py` port: 9/9 from clean, `SEED-INERT` on an inert seed,
  `docs/prospective_capture.md` byte-identical.
- **`steq.py` + `fidelity.sh` port; 24 register flags on four human surfaces** → one committed
  scanner reports 0 `LONG`/`FILLER` over the `conventions.md` inventory, fails a seeded 30-word
  instruction, no specifier/flag/number delta.
- **M2.8.4's two corpus checks port** → `scripts/check_corpus_2d_integrity.py` rc=0 on
  `output/corpus-2d` (379/379 set-equal, 11 verdicts true, breach < 0.1 %), rc=1 named on 3 tampers.
- **Camera-setup and other non-task footage is tracked as the subject (C06,
  `docs/technical/pose-weak-points.md`; 13/193 families noted)** → a committed segmenter flags
  non-task spans per asset (camera handling, floor/ceiling, operator) online; on >= 20
  hand-labelled assets it flags >= 90 % of setup frames and < 2 % of task frames, and the R stage
  excludes flagged spans with the exclusion counted in the run report.
- **Out-of-frame points reach the export at visibility >= 0.3 (M2.9.2 P09 fail)** — 3648 of
  2 555 251 feature-keypoint observations, 84/379 clips, overshoot median 36 px; a stage after
  hygiene relocates them (`docs/technical/pose-weak-points.md` § M2.9 corpus; `.scratch/oof_probe.py`)
  → the moving stage is named by one measurement (bone-length projection vs filter on a hit
  clip); a test drives that stage to place a scored in-frame point outside the frame and asserts
  exported visibility 0; `oof_probe.py` reports 0 on the next corpus run.
- **Occluded far hand blinks under the duplicate rule** (right-view blink-outs 0.46 → 3.27 per
  100 rows, 46 % on or near the other hand; `.scratch/flick_cause.py`) → over the watch set every
  view reads <= 1.0 blink-outs per 100 rows while exported frames with one hand's median point
  distance to the other < 0.3 of hand extent stay 0, and each clip's task-hand presence share
  stays within 0.02 of the M2.9 baseline (`.scratch/watch/triage_m29.json`), so an all-absent
  candidate fails. Mechanism open (hysteresis, occlusion-aware presence).
- **M2.9.2 P10 census is vacuous** — it counts frames with both hands' coordinates present, and a
  suppressed hand exports none (`mapping._HAND_PRESENCE_THRESHOLD`), so it reads 0 on both
  generations → a D02 firing counter published per view (run report or diagnostics) reads > 0 on
  the corpus and 0 with the rule disabled.
- **Gated hand-crop refinement misses the adoption bar** (`docs/technical/pose-weak-points.md`
  § M2.9.3: jitter 0.0079 vs <= 0.006, wall +41 % vs <= +15 %, right views worse) → a variant
  (batched crops, refinement only where the whole-body hand is small or unstable) reaches jitter
  <= 0.006 on the watch set at <= +15 % wall with collapse unchanged on every clip.
- **2D weak-point triage lives in `.scratch/triage.py` alone** (→ `gates.md`; behind
  `docs/technical/pose-weak-points.md`) → a committed `scripts/` instrument with no
  `prototype/` import reproduces the catalog's per-view counts over a run tree and flags a
  seeded whole-skeleton relocation in a synthetic landmark CSV.
- **Detector scores outside `[0,1]` accepted silently** → score-range guard fires on NPU, silent on
  CPU + GPU, under the synthetic zeros probe.
- **`pose-estimation-run --session-dir … --output-dir X` ignores `X`** → rtmlib session artifacts
  land under the requested root; MediaPipe unchanged.
- **`sessions.py` predicates pinned by tests alone** → a campaign mirroring
  `run_inventory_mutations.py` kills every mutant with >=1 committed test, replaying from clean.
- **Review UI theme control proven by a scratch script alone** (`.scratch/theme_qa.mjs` → regen path
  in `.claude/rules/gates.md`) → a committed check drives the cycle, the pre-paint rehydration and
  the `:has()` stage backdrop, 13/13 green, failing under the two recorded seeds.
- **Review UI player fit proven by a scratch script alone** (`.scratch/player_layout_qa.mjs` → regen
  path in `.claude/rules/gates.md`) → a committed check reports the stage fitting its room, filling
  an axis and holding its ratio over five viewports plus the stacked breakpoint, every control row
  on screen, and reds under the four recorded seeds.
- **Review UI landing view + fragment routing proven by a scratch script alone**
  (`.scratch/default_view_qa.mjs` → regen path in `.claude/rules/gates.md`) → a committed check
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
  to sub-pixel precision in world units." `gates.md` already records the tell: "NC4's needle must
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
  (`.scratch/player_clock_qa.mjs`, `.scratch/player_time_qa.mjs`; → `gates.md`) → port both to
  committed checks under `prototype/review-ui/tools/`, run from the recorded commands: clock 10 rows
  green with the seeded `view.framePos`→`view.frame` regression reding exactly the 6 rate rows while
  both `layer=show` rows stay green; time map 7 rows green with the series-index seed reding the 5
  map rows and the seek-less `|▶` seed reding the step row alone.
- **Review UI stacked-layout stability proven by scratch scripts** (`.scratch/drift_qa.mjs`,
  `.scratch/ratchet_qa.mjs`; → `gates.md`) → committed checks under `prototype/review-ui/tools/`:
  drift 22/22 with a stacked `1fr` column reding exactly the 4 stacked injection rows; ratchet
  60/60 across scales 1/1.1/1.25/1.5 with stacked `auto auto` rows reding only stacked rows.
- **Marksman reports 3 links to existing docs as non-existent** (`docs/calibration_finding.md:9`,
  `docs/technical/entrypoints.md:181,216` → `technical/{calibration_qc,qualification}.md`; git
  ignores neither target). Suspected cause, unconfirmed: its index reads the `.gitignore`
  directory patterns `qualification.*/` + `calibration_qc.*/` as matching the `.md` files →
  Marksman diagnostics over `docs/` report 0 non-existent-document warnings, and
  `git check-ignore` still ignores a `qualification.x/` staging directory.
- **The gate runs no security scanner and the remote carries no update automation** (template
  IMPLEMENT; hosted CI replaced by the local gate → `gates.md` *Hosted CI*) → the gate command in
  `gates.md` runs a dependency audit over every committed uv lockfile, a secret scan and a security
  static-analysis pass, each recorded firing on a planted input beside the invocation;
  `.github/dependabot.yml` covers every committed uv lockfile.
- **`calibration-qc.md:35` cites `upstream-refresh.md` clause 2, a rule file renamed in `b218092`**
  → the pointer names `upstream-sync.md` Clause 1, and
  `rg -Fn 'upstream-refresh.md' .claude/rules/` returns 0 hits.
- **`pose_config.json` omits the OpenVINO build** (`corpus_run.POSE_CONFIG_FIELDS`; a build swap
  moves landmarks by ≤ 1e-4, features ≤ 4.1e-4 of column max → `rtmlib-runtime.md`) → the config
  records `openvino.get_version()`; a resume under another build re-runs the event; a test seeds a
  recorded config with a foreign build string and asserts the event is due.
- **`.scratch/ovupgrade/` build-change instruments are scratch-local** (`pilot.sh`, `compare.py`,
  `det_qual.py` → `gates.md`) → committed under `scripts/` with a test that seeds a one-cell CSV
  drift and asserts `compare` reports it, and a same-tree run reports 0 drift.
- **`upstream-sync.md`'s declined-gate paragraph counts `7 shipped files` naming the archive path;
  Clause 1 and `rg -l 'archive/contract-' scripts/ src/ tests/` both give 8** → that paragraph
  points at Clause 1 instead of its own figure; `rg -c '7 shipped files' .claude/rules/` = 0.
- **M2.10 watch-set graders are scratch-local** (`.scratch/messy/` + 972 labels; regen →
  `gates.md` § *Scratch validators pending port*) → a committed `scripts/` grader with a synthetic
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
- **Monocular 3D angles (SAM 3D Body) — user: pilot more first** (`.scratch/pilot3d/`, Arc XPU,
  CUDA calls patched to `SAM3D_DEVICE`; weights gated, granted) → a widened pilot over >= 40 events
  reports, per joint (elbow, shoulder, trunk), synced cross-view MAE against a time-shuffled control
  and against 2D, per-frame wall, and a non-collapse check (within-series SD vs 2D); adoption is the
  user's ruling. Widened read (19 events / 620 synced instants, `.scratch/pilot3d/xview_wide.json`,
  `xview_stats.py`): elbow flexion cross-view MAE 6.0° synced vs 11.7° time-shuffled (2D 33.5 vs
  35.8), 3D closer on 23/24 series; shoulder 4.3 vs 5.2 (ratio 0.82, worse than 2D's 0.51 — the
  3D shoulder may lean on the model's prior); inference 0.67 s/frame median on the Arc GPU.
- **Pose latency runs 3-5× slow on some assets, cause unmeasured** — paired probes on one day: 1-3 of
  11 assets at 130-274 ms/frame mean in BOTH arms (survey on and off) against M2.9's corpus median
  50.0 ms and 0/379 assets above 100 ms; AC power, `performance` profile, no GPU throttle flag at
  idle (`.scratch/messy/latency/probe_*_assets.json`) → one measurement names the stalled stage
  (detector GPU, pose NPU or decode) on a slow asset, and a rerun of that asset reads <= 70 ms/frame
  after the named fix or the cause is recorded as host-side.


# Deferred

Off-spine queue. One row = one improvement + the acceptance check that closes it, written at
deferral time while the evidence is fresh; a row leaves only when that check passes on a commit.
Read on demand — not attached state. A row promoted to an unfinished unit moves to `.agent/spec.md`
`Deferred`, which carries the spine, and returns here when the spine moves past it.

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
- **`--det-device GPU` unqualified against real detections** (9.7 vs 213 ms → ~7.1 h toward ~1 h) →
  GPU + CPU agree on the detection set, every padded row rejected by value, pilot green; then flip.
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
- **`capture_proof.py` port guard fires only against a live stranger** (`already_serving` returns rc=2;
  measured against a stale host server on 8791 serving deleted code) → a check binds a socket on the
  target port, asserts rc=2 and the named cause, and asserts rc=0 on a free port.
- **Claim boundary guarded by 18 substrings; 7/7 semantic overreaches pass** (`check_claim_report.py:89`
  `_p03`, needles = `calibration_qc.PROHIBITED_PARAPHRASES`; measured against the shipped `_fold`,
  positive control `clinical validity` fires) → the guard refuses a held-out paraphrase set covering
  all three forbidden classes, including one Japanese rendering, at 100 % recall on overreach, passes
  the two shipped documents unchanged, and routes every uncertain sentence to review rather than
  approving it. `gates.md` already records the tell: "NC4's needle must come from
  `PROHIBITED_PARAPHRASES` rather than invented prose."
- **`check_review_report.py` P04 + P06 are spelling tests** (`:53`, `:77-80`) → a report whose `pass`
  cell reads `aaaaaaaaaaaaa` and whose detail reads `predicate impact acceptance x.py:1` grades
  nonzero; the 57 rows of `.agent/archive/review-m2.md` still grade `PASS`. Today the hollow report
  grades `PASS` rc=0 and a one-character shortening of the `pass` cell is the only thing that reds it.
- **Review UI feature finder is a three-field substring filter** (`static/cohort.js:212`; searchable
  English vocabulary = 41 words over 89 features) → a committed check reports the 13 recorded
  collaborator queries retrieving their intended features, where 12 return 0 today, with no loss on
  the literal queries substring search already answers and `ja` graded separately.

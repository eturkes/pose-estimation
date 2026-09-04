# M2.8.3 acceptance contract — cohort aggregate publisher + bilingual descriptor

Base `fc8ff62`. Tier `kernel`. Frozen at dispatch; amendments append to §10 and bind from
their ruling. Every downstream artifact decides against this file.

## 1. Scope

Publish `cohort/` — a redaction-safe aggregate of the M2.8.2 corpus run at `(task, side)`
grain — plus an append-ready `columns.yaml` descriptor fragment naming every published
feature in `../rehab`'s registry format.

**In scope.** The publisher module + CLI, the published tree, the feature label table, the
descriptor fragment, the technical document, the gate wiring.

**Out of scope, explicitly.** Editing `../rehab`. Re-running the corpus (→ M2.8.4). Any
per-subject row, patient identifier, join column or join template — cut by user ruling.
Changing `analysis/clinical_features.R`. Adding a `columns.yaml` loader change on the
consumer side.

## 2. Inputs and trust roots

| input | trust root |
| ----- | ---------- |
| `inventory/` | `inventory.validate_generation(dir)` before any row is read |
| `sessions/` | `sessions.validate_generation(dir, inventory_dir=…)` |
| `output/corpus-2d/run_manifest.csv` | `corpus_run.read_manifest` + `corpus_run.validate_manifest(rows, canonical)` |
| `output/corpus-2d/<event>/*_clinical.csv`, `*_clinical_windows.csv` | header equality against the feature table; finite-value filter |

The run tree carries no generation marker, so the manifest **is** its trust root: a row set
that is not a total partition over the registry's canonical assets refuses the whole
publication. An asset whose manifest disposition is not `ok` contributes nothing and is
counted as excluded (D08), never skipped silently.

## 3. Design decisions

**D01 — grain `(task, side)`, 12 cells. User ruling.** No subject rows, no patient
identifier, no join column, no join to `../rehab`. `view` is not a grain component (D03).

**D02 — estimand = the typical SUBJECT, by four-stage aggregation. User ruling.**
`asset value = median(finite rows of that asset)` → `event value = median(asset values)` →
`subject value = median(event values)` → cohort statistics over subject values. Measured
alternative (pooling all 21 483 window rows) moves the cohort median by −2.8% to +298%
across the 12 cells, and one subject supplies up to 29% of a cell's rows, so the grain is
contract-bearing and never a convenience. Every published statistic names its estimand in
the technical document and in the descriptor fragment.

**D03 — view is a published variance component, not a grain and not a selection. User
ruling.** Views of one performance disagree substantially: over 135 multi-view events all
cameras agree on which wrist moved faster only **50.4%** of the time, and the within-event
across-view CV is **45-66%** of the between-subject CV (subject signal dominates, view is
large). 2D features are measured in each camera's own image plane, so this is projection
geometry rather than a defect. The artifact therefore publishes `view_dispersion` beside
each statistic — the median across-view CV over that cell's multi-asset events — with
`n_events_multiview` naming its population, and the claim boundary prohibits directional
per-limb claims (D07/N3).

**D04 — 75 features, both levels. User ruling.** 45 frame-level (posture/kinematic state)
+ 30 window-level (movement quality). The 17 trunk/posture columns are excluded because
they are structurally absent, not because they are unwanted (D08).

**D05 — statistics = the robust set, no extremes. User ruling.** `n_subjects`, `n_events`,
`n_assets`, `n_values`, `median`, `q25`, `q75`, `mean`, `sd`, `view_dispersion`,
`n_events_multiview`. `min`/`max` are refused: at n=15-16 per cell each extreme is one
identifiable subject's own measurement, while quartiles are not.

**D06 — the published tree is gitignored. User ruling.** `cohort` + `cohort.*/` join
`.gitignore` beside the five existing publishers. Consequence, stated rather than
discovered: the published bytes get **no committed byte oracle**, so every predicate below
is a property test and the determinism sweep is what stands in for a golden.

**D07 — the label table is a module constant, and the descriptor fragment is derived from
it.** `cohort.FEATURES` carries one entry per published feature — id, level, source column,
`ja`, `en`, `unit`, `range` — and the fragment is generated, never authored twice. Same
ruling as `calibration_qc.CLAIMS`: a second hand-written copy drifts, and M2.7.3 measured
that drift at 6 of 15 rows on a document nothing pinned.

**D08 — every source column reaches exactly one published outcome.** `cohort.json` carries
a `columns` census partitioning all 92 feature columns into `published` (75) and
`excluded` with a frozen reason code — `structurally_absent` for the 17 trunk/posture
columns, which require `tracking == "body"` (hip keypoints) and are NA on 100% of
331 152 frame and 21 483 window rows. Same totality rule as M2.8.2's manifest: a column
silently missing from a denominator is the defect the census exists to prevent.

**D09 — publish beside, never patch.** `cohort/` is a new tree. `output/corpus-2d/`,
`qualification/`, `sessions/` and `inventory/` are read-only to this publisher, and the
`sessions` generation digest + marker digest are witnesses across the run (P12).

**D10 — `raw` keys are namespaced `pose_<level>_<column>`.** `../rehab`'s `load_schema()`
enforces required keys by `KeyError` alone — no unique-raw check — and duplicate raw names
silently keep the last, so the prefix is load-bearing against the 219 existing descriptors.

## 4. Published schema

`cohort/` holds four files. Row order is a function of the rows at the publish site.

- **`cohort_cells.csv`** — 12 rows, one per cell.
  `task, side, n_subjects, n_events, n_assets, n_frame_rows, n_window_rows`
- **`cohort_features.csv`** — 900 rows (12 × 75), one per `(task, side, feature)`.
  `task, side, level, feature, n_subjects, n_events, n_assets, n_values, median, q25, q75, mean, sd, view_dispersion, n_events_multiview`
- **`descriptors.yaml`** — 75 entries, `../rehab` `columns.yaml` shape:
  `raw, ja, en, group: pose, role: feature, dtype: numeric, unit, range`.
- **`cohort.json`** — redaction-safe census + provenance: population, the D08 column
  census, the estimand statement, the input digests, and a `generation` block digesting
  the other three files plus itself minus its own key.

## 5. Predicates

Each is testable, and each names the artifact it decides.

- **P01 totality of cells** — exactly 12 rows, key set equals the 12 `(task, side)` pairs
  re-derived from the registry, no duplicate key, every cell `n_subjects >= 5`.
- **P02 totality of features** — 900 rows, key set equals the 12 cells × the 75 published
  feature ids, no duplicate key.
- **P03 column census is total** — `published ∪ excluded` equals the 92 feature columns
  re-derived from the two artifact headers at check time; the two sets are disjoint; every
  `excluded` entry carries a frozen reason code.
- **P04 estimand is the four-stage one** — an independent re-derivation of D02 over a
  synthetic corpus reproduces every published statistic exactly; a window-pooled
  implementation differs on at least one cell (the seed carries a case where it does).
- **P05 subject weighting is real** — replicating one subject's asset rows N-fold leaves
  every published statistic unmoved; replicating a whole subject moves `n_subjects`.
- **P06 view dispersion names its population** — `view_dispersion` is empty exactly when
  `n_events_multiview == 0`; where non-empty it equals the median across-view CV over that
  cell's multi-asset events; a cell whose events are all single-asset publishes empty
  rather than 0.
- **P07 non-vacuity** — every set-quantified predicate carries a non-empty floor:
  `n_values > 0`, `n_subjects > 0` and `n_events > 0` on all 900 rows, and at least one
  cell has `n_events_multiview > 0`. A green predicate whose detail line reports zero
  items is a failing predicate.
- **P08 finite filter** — non-finite values (`NA`, `NaN`, `Inf`) are excluded from every
  statistic and counted out of `n_values`; a feature with no finite value anywhere
  publishes empty statistics and is not silently dropped from the 900.
- **P09 descriptor fragment conformance** — 75 entries; every entry carries the six keys
  `../rehab` requires plus `unit` and `range`; every `raw` matches `pose_(frame|window)_`
  + a published feature id; every `raw` is unique; every `ja` and `en` is non-empty and
  `ja` contains at least one non-ASCII character; the fragment parses as YAML and its
  entry set equals `cohort.FEATURES`.
- **P10 unit vocabulary is derived, not asserted** — `unit` values come from a frozen
  vocabulary keyed on the source column's measurement. 2D landmark coordinates are
  normalized to `[0, 1]` **by frame dimensions** (`export.py:250`), so the anisotropy
  question is decided at implementation and its answer binds the vocabulary: any `*_deg`
  column computed from anisotropically normalized coordinates is **not** a true anatomical
  angle and may not carry unit `deg` unqualified. The determination is recorded in §10.
- **P11 redaction** — no published byte carries a subject ordinal, `capture_id`, event id,
  camera name, source path, media suffix or filename. Value admissibility is membership in
  a published label set (task token, side token, level token, feature id, reason code,
  unit token) ∪ code-authored constants; key admissibility is membership in the frozen
  field names. A shape test is a denylist wearing an allowlist's name (M2.8.2 A09).
- **P12 read-only over upstream** — `sessions.tree_digest`, `sessions.generation_digest`
  and `inventory` + `sessions` + manifest validation all hold before and after the run;
  `output/corpus-2d/` is byte-unmoved.
- **P13 publisher trust root** — the marker is a regular non-symlink file; the parse
  rejects duplicate keys (`object_pairs_hook`); the census digest covers the provenance
  block minus its own self-referential key. All three checked directly, since no digest
  inside the set covers the marker.
- **P14 ownership + atomic publication** — `--out` overlaps no input in either direction;
  a non-empty destination is replaced only when its own marker names this generator at
  this version; publication is staging-sibling + swap, and crash debris is swept only
  after the swap lands.
- **P15 determinism** — the tree is a function of its inputs alone: byte-identical across
  hash seed, four locale settings, timezone, `umask`, `-O` and `--out` name. This is D06's
  stand-in for the golden the gitignored tree cannot have.
- **P16 idempotence** — republishing over a published tree leaves every byte unmoved.
- **P17 consumer boundary** — `validate_generation` refuses a half-published set, an
  edited CSV, an edited descriptor fragment and an edited census, each by exception class.
- **P18 documentation + registration** — `docs/technical/cohort.md` states the estimand,
  the D03 view limit, the D08 exclusion and the claim boundary; the four exhaustive
  documentation indexes (`entrypoints.md` count, `architecture.md` module map,
  `tests.md` inventory, `conventions.md` campaign list) each name the new publisher.

## 6. Invariant surfaces

1. `output/corpus-2d/` — read-only; byte-unmoved across the run.
2. `sessions/`, `inventory/`, `qualification/` — read-only; digests unmoved.
3. `analysis/clinical_features.R` and every golden under `tests/goldens/` — untouched.
4. The 219 existing `../rehab` descriptors — the fragment is append-ready and collides
   with none of them (D10).

## 7. Gate identity

`env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync` prefixing each of
`ruff check`, `ruff format --check`, `ty check`, `pytest`. Decisive run is primary-tree,
after any decode or inference sweep has finished, never beside one. Baseline collection at
`fc8ff62` is **1517 passed / 0 skipped**; the unit's close must move collection by exactly
its own new cases.

## 8. Measured facts this contract rests on

Every number re-derives from `output/corpus-2d/` and the registry. **A12 makes every fact in
this section provisional**: the corpus is being re-run with the orientation fix and
`--tracking body`, which restores the 17 trunk/posture columns. Re-derive before citing; no
predicate may assert 75, 900 or 17 as a literal.

- **Cells.** 12 `(task, side)` over `cap coin glass key nut peg` × `l r`; 15-16 subjects
  each; 193 events; 379 assets; manifest 379/379 `ok`. `cap-l` 18 events and `nut-r` 19
  events exceed their subject counts — view-conflict families splitting into single-camera
  events, which is why the estimand medians over events inside a subject (D02).
- **Rows.** 331 152 frame rows, 21 483 window rows. Windows per asset: min 12, p25 26,
  median 37, p75 64, p95 123, max 548.
- **Coverage.** Frame-level features 93.4-96.4% finite; window-level 74.6-100% finite
  (`movement_efficiency` lowest at 74.6/77.1%, `sal` 87.7/89.7%).
- **The 17 structurally-absent columns.** Frame: `trunk_lean_deg`,
  `trunk_lean_lateral_deg`, `trunk_lean_sagittal_deg`, `trunk_rotation_deg`,
  `posture_symmetry`. Window: `compensatory_pattern_index`, `trunk_lean_mean/sd/range`,
  `trunk_lean_sagittal_mean/sd`, `trunk_lean_lateral_mean/sd`, `trunk_rotation_mean/sd`,
  `posture_symmetry_mean/sd`. 0.00% finite on every row. Cause is the guard at
  `analysis/clinical_features.R:1000` — the block runs only under `tracking == "body"`,
  and the corpus ran `hands-arms`.
- **Estimand sensitivity.** Window-pooled vs subject-weighted cohort median on
  `left_wrist_velocity_mean`: `glass-r` −0.5%, `nut-r` −1.5%, `peg-r` −1.2%, `glass-l`
  −2.8%, `peg-l` +4.1%, `coin-l` +10.1%, `coin-r` +23.3%, `key-r` +64.6%, `key-l` +72.7%,
  `nut-l` +97.3%, `cap-l` +117.0%, `cap-r` +297.8%.
- **View variance.** 135 multi-view events; all views agree on the faster wrist in 68
  (50.4%) against ~55 expected from independent coin flips. Within-event across-view CV
  against between-subject CV: `wrist_velocity_peak` 0.560/1.094 = 0.51,
  `wrist_normalized_jerk` 0.158/0.239 = 0.66, `wrist_sal` 0.029/0.044 = 0.66,
  `movement_efficiency` 0.454/1.002 = 0.45, `wrist_velocity_peak_symmetry_ratio`
  0.163/0.273 = 0.59. Subject signal dominates on every feature; view is 45-66% as large.
- **Coordinate space.** 2D landmark coordinates are normalized to `[0, 1]` by frame
  dimensions (`src/pose_estimation/export.py:250`). Displacement, reach and aperture are
  therefore frame-relative dimensionless quantities and velocity is per second in those
  units; P10 governs the `*_deg` columns.

## 9. Negative-control seed

Each control must fire and name its own predicate; the tree is restored byte-identical.

| id | seed | predicate |
| -- | ---- | --------- |
| N1 | Drop one `(task, side)` row from the published cells | P01 |
| N2 | Duplicate one `(task, side, feature)` row and drop another | P02 |
| N3 | Move one trunk column from `excluded` to `published` | P03 + P07 |
| N4 | Recompute one cell window-pooled instead of subject-weighted | P04 |
| N5 | Replicate one asset's rows 5× | P05 |
| N6 | Publish `view_dispersion = 0` on a cell with `n_events_multiview = 0` | P06 |
| N7 | Admit a non-finite value into one statistic | P08 |
| N8 | Blank one `ja` label; collide one `raw` with an existing `../rehab` descriptor | P09 |
| N9 | Write a subject ordinal into the census | P11 |
| N10 | Reindent the `sessions` generation marker | P12 |
| N11 | Replace the cohort marker with a symlink; add a duplicate key to it | P13 |
| N12 | Point `--out` inside `output/corpus-2d/`; hand a foreign directory to `--out` | P14 |
| N13 | Sort the descriptor entries by label instead of by id | P15 + P16 |

## 10. Amendments

**A01 — F1 ACCEPTED. The two tables count different populations, and both are now named.**
P01/P02 as frozen never said whether `n_subjects`/`n_events`/`n_assets` range over canonical
registry membership, manifest-`ok` contributors, or contributors carrying a finite value. On
this corpus all three coincide (379/379 `ok`), which is precisely how the divergence would
have shipped unnoticed — the project's own recurring defect (M2.5's 329-vs-355, the two
closure statistics, the two "families connected" figures).

- `cohort_cells.csv` counts **canonical assets whose manifest disposition is `ok`**, and the
  events and subjects those assets belong to. Independent of feature finiteness.
- `cohort_features.csv` counts **finite contributors at each named stage**: `n_assets` =
  assets contributing a finite asset value for that feature, `n_events` = events with ≥1 such
  asset, `n_subjects` = subjects with ≥1 such event, `n_values` = finite leaf rows.
- **New conjunct on P02**: every `cohort_features` count is ≤ its cell's corresponding
  `cohort_cells` count. This is the mechanical form of "name the population beside the count"
  — it makes a future divergence fail rather than publish.

**A02 — F2 ACCEPTED, with a different fix; the teammate's proposal is refused for cause.**
The finding is correct and sharp: moving a column between `published` and `excluded`
preserves union, disjointness and every remaining reason code, so P03 as frozen cannot fire
N3. The proposed repair — freeze the 17 `(level, column)` exclusions literally in P03 — is
**refused**: a contract carrying a transcribed census is the trap this project has hit four
times, and the 17 are a fact about a run configuration that M2.8.4 is scheduled to change.

Ruled instead, in the project's own re-derive-never-transcribe idiom: **the partition is
measured, not decreed.** A column belongs in `published` iff it carries ≥1 finite value over
the run, and in `excluded` with reason `structurally_absent` iff it carries 0. P03 gains that
conjunct in both directions. Consequences, all intended:

- N3 then fires P03 directly, and its §9 wording is corrected to *"move a column with zero
  finite values into `published`"* — under this ruling a synthetic trunk column that IS finite
  belongs in `published`, so the teammate's synthetic counter-case is correct behaviour rather
  than a miss.
- M2.8.4's `--tracking body` re-run moves 17 columns into `published` with **no contract
  edit**, because the partition follows the data.
- Cross-check retained, since two independently-derived sets are worth more than one:
  `cohort.FEATURES`'s source-column set must equal the measured `published` set, so a label
  table that lags the data fails by name.

**A03 — F3 ACCEPTED in full. The statistics now have exact definitions.**
P04's "exactly" was unsatisfiable: no quantile rule, SD denominator, singleton rule or
serialization was stated, and defensible libraries disagree. Frozen:

- `median` = `statistics.median` (mean of the two central values at even n).
- `q25`/`q75` = **linear interpolation at position `(n-1)q`** on the ascending values —
  numpy's `linear`, R's type 7. Empty at n < 1.
- `mean` = arithmetic mean. `sd` = **sample** standard deviation, denominator `n-1`;
  **empty at n < 2**, never 0.
- `view_dispersion` = per multi-asset event, `pstdev(asset values) / abs(mean(asset values))`
  (population denominator — the views ARE the population, not a sample of one); the cell's
  value is the median over its eligible events. An event whose mean is 0 is ineligible and is
  excluded from `n_events_multiview`.
- Serialization: every numeric cell goes through one `_cell()` helper at fixed decimals, the
  idiom the four sibling publishers already share. **P04's "exactly" means byte equality of
  the published cell**, so the rounding rule is part of the contract rather than a tolerance.

**A05 — F4 ACCEPTED. P05's invariance was stated over the wrong column set, and the repair
strengthens it.** "Replicating one subject's asset rows N-fold leaves every published
statistic unmoved" is false for `n_values`, `n_frame_rows` and `n_window_rows`, which are
counts of leaf rows and scale with N by definition. Split into two predicates, and the split
is worth more than the original:

- **P05a, row replication.** Duplicating an asset's rows N-fold leaves every **distribution**
  statistic (`median`, `q25`, `q75`, `mean`, `sd`, `view_dispersion`) unmoved, and **must**
  move `n_values` and the cell row counts. Requiring the counts to move is the stronger
  claim: it proves they are real counts rather than something re-derived from the medians.
- **P05b, subject cloning.** Cloning a whole subject moves `n_subjects`, `n_events`,
  `n_assets` and `n_values`, and **may move every distribution statistic** — adding a
  duplicate subject value shifts a median over subjects. Invariance is asserted only at P05a.

**A06 — F5 ACCEPTED as the residue A03 did not reach.** A03 froze the CV formula, the
population denominator and the zero-mean exclusion; it left **eligibility** unstated, and
under A01 that is not a free choice. Ruled: an event is eligible for a feature's
`view_dispersion` iff it carries **≥2 assets with a finite value for that feature** — not ≥2
assets. `n_events_multiview` counts exactly the eligible events, so it is per feature row and
not a per-cell constant, and a zero-mean event is excluded from both the statistic and the
count. `view_dispersion` is empty iff `n_events_multiview == 0`, which is P06 unchanged.

**A07 — F6 ACCEPTED. P07 and P08 contradicted each other outright, and the measurement says
which one gives way.** P07 demanded `n_values > 0` on all 900 rows; P08 mandated that a
feature with no finite value publishes empty statistics and stays in the 900. Both cannot
hold. Measured on this corpus: **0 of 900 rows have zero finite values, and 0 of 900 lack an
eligible multiview event** — so P07's universal claim is true today and P08's empty path is
unreachable here. That is the "agrees today, diverges silently later" shape, and M2.8.4's
re-run is already scheduled to change the column set. Ruled:

- **P07 becomes artifact-level anti-vacuity**, which is what it was always for: at least one
  row with `n_values > 0`, at least one row with `n_events_multiview > 0`, a non-empty
  published feature set, and a non-empty cell set. A green predicate whose detail line reports
  zero items still fails.
- **The all-900 property becomes a published census, not a predicate.** `cohort.json` carries
  `rows_zero_values` and `rows_without_multiview`, both **0** on this corpus. A future run
  that empties a row then shows it in the artifact instead of failing a universal that was
  only ever a fact about one corpus.
- **P08 keeps its rule unchanged** and is exercised **synthetically**, since the real corpus
  cannot reach it. The suite's instinct to scope P07 positive and P08 adversarial into
  separate generations is confirmed and is now the contract's requirement.

**A04 — the census sub-schema is frozen as the suite chose it (P03 detail row).**
`cohort.json` `columns.published` entries are `{level, column}`; `columns.excluded` entries
are `{level, column, reason}`. §4 left the JSON sub-schema open; freezing it here removes the
open question rather than leaving the suite to guess a spelling MAIN might later move.

**A08 — F7 ACCEPTED. One name, one projection, one collision root.**
Three defects in one predicate, all real.

- **The second name is deleted.** `FEATURES` entries carry `(level, column, ja, en, unit,
  range)` — **no separate `id`**. The published `feature` cell **is** `column`, and `raw` is
  `pose_<level>_<column>`. D10 said source column, P09 said feature id; rather than pick a
  winner, the field that could diverge is removed. `(level, column)` becomes the single key
  shared by the A04 census, the feature table and the fragment.
- **P09 is a bijection under a named projection, not set equality.** `FEATURES` entries carry
  6 fields and YAML rows carry 8, so literal equality was a type error. Exactly one YAML entry
  per `FEATURES` entry; `raw` as above; `ja`/`en`/`unit`/`range` copied; `group: pose`,
  `role: feature`, `dtype: numeric` constant.
- **Root shape** = `{columns: [...]}`, re-derived from `../rehab/schema/columns.yaml`, whose
  top-level keys are `version, encoding, missing_sentinels, columns, families`.
- **Collision gets two layers.** Hermetic, always on: every `raw` carries the
  `pose_frame_`/`pose_window_` prefix and is unique within the fragment. External: when
  `../rehab/schema/columns.yaml` resolves, the intersection with its raw set is empty. The
  external set re-derives to **exactly 219** — 67 literal + 56 + 56 + 20 + 20 expanded from
  4 family templates, all unique, **zero `pose_`-prefixed**, which is what makes D10's prefix
  load-bearing rather than decorative. The external outcome is **published** in `cohort.json`
  as `descriptor_collision {checked, source, n_external, n_collisions}` — never a silent skip,
  the same demotion A07 applied to the all-900 claim.
- **N8 restated hermetically** so it fires without the sibling repo: blank one `ja`; strip one
  `raw`'s `pose_<level>_` prefix; duplicate one `raw`.

**A09 — F8 ACCEPTED. The P10 anisotropy determination, measured rather than deferred.**
The deferred question is now decided against the R source and the corpus.

- `z` is **identically 0.0** on every landmark column corpus-wide, so `angle_at_vertex`
  (`clinical_features.R:343`) is a plain 2D image-plane angle.
- `export.py:250` divides x by frame width and y by frame height. At 16:9 that is
  **anisotropic**, so a published angle is not the angle it names.
- **Measured error vs the true image-plane angle: median 9.9°, p75 17.3°, p95 26.5°,
  max 32.5°**, over 40 upright assets and 75 256 frames, both arms.

**Ruling: no published column carries unit `deg`.** Frozen vocabulary, 7 tokens —
`deg_image_plane_uncalibrated`, `frame_normalized`, `frame_normalized_per_s`,
`ratio_shoulder_width`, `ratio`, `index_signed`, `dimensionless`. Assignment is a **rule over
source family × derivation suffix**, never a transcribed 75-row table: angle-derived (the raw
`*_deg` columns and their `*_abs_diff`) → `deg_image_plane_uncalibrated`; normalized distances
(`reach_raw`, `grasp_aperture_*`, `*_displacement`, and their `_abs_diff`) → `frame_normalized`;
`reach_norm` and its `_abs_diff` → `ratio_shoulder_width`; velocity means/peaks →
`frame_normalized_per_s`; `normalized_jerk` and `sal` → `dimensionless`; `movement_efficiency`
and every `*_symmetry_ratio` → `ratio`; every `*_dominance_index` → `index_signed`. P10 checks
vocabulary membership, the `deg` prohibition, and the angle-column assignment. The claim
boundary in `docs/technical/cohort.md` states the measured error and prohibits anatomical-angle
claims. **User ruling:** the 9 angle-derived columns ship under the qualified token rather than
being dropped.

**A10 — F9 ACCEPTED. P11 becomes field-typed; the self-authorizing arm is deleted.**
All three sub-claims hold: numeric values are neither label tokens nor constants, so the
artifact failed its own predicate; "code-authored constants" self-authorizes, since adding a
leaked token to code makes it admissible; and a subject ordinal is not byte-scannable because
ordinary counts reuse digit strings.

- **Numeric fields**: domain is number. The one identifier-bearing numeric channel is a
  statistic over a population of one, and it is **structural, not lexical**. New conjunct: a
  feature row with `n_subjects < 5` publishes **empty distribution statistics**; counts still
  publish. This is D05's own rationale — it refused `min`/`max` at n=15-16 because an extreme
  is one identifiable subject, so a median at n=1 is strictly worse. Census gains
  `rows_below_subject_floor`.
- **String fields**: each names a closed domain — `task`/`side` from the registry, `level` from
  `{frame, window}`, `feature` from the measured published partition, reason code from a frozen
  enumeration, `unit` from A09's vocabulary, `ja`/`en` from `FEATURES`, digests matching
  `^[0-9a-f]{64}$`, generator/version frozen literals.
- **Key admissibility**: the frozen field-name set is §4's schema plus the A04 and A08
  sub-schemas; the check enumerates the artifact's actual keys and requires **set equality** —
  the freeze P11 quantified over but never performed.
- **Governing rule, which reconciles this with A02: the schema is contract-owned and therefore
  frozen; the data is input-owned and therefore measured.** Freezing field names is legitimate;
  freezing the 17 excluded columns was not.
- The input-derived needle scan stays as a **second layer**, needles derived from validated
  input rows. Typed allowlist primary, scan backstop, both.

**A11 — F10 ACCEPTED. P12 splits; a validation function is never a byte witness.**
Semantic validation canonicalizes its JSON, so a reindent survives it — N10 is exactly that
seed, and §6's byte-unmoved claim exceeded P12's witness set.

- **P12a unmoved**: a recursive **non-following** pre/post snapshot of every upstream entry —
  relative path, kind, symlink target text, content digest for regular files. Snapshot equality
  is the witness. Non-following is load-bearing: `videos`, `inventory` and `renv/library` are
  symlinks in a teammate worktree.
- **P12b valid**: the three validations plus both `sessions` digests hold before and after.
- M2.8.2 ruled the identical defect one unit earlier (`3225577`: "P11's two witnesses could not
  see the marker's own bytes → third witness"). The contract reintroduced it, so the rule is
  promoted to memory rather than left as a per-unit finding.

**A12 — the corpus is defective; M2.8.3 is BLOCKED on a corrected re-run. User ruling.**
Two corpus-generation defects measured this window, neither visible in the manifest (379/379
`ok`):

- **Orientation.** 38 of 379 assets (10.0%) were pose-estimated in a non-upright frame: 28
  portrait-stored assets decode sideways (1080×1920), 10 decode upside down. OpenCV ignores the
  container display matrix and **cannot be made to honor it** — default, explicit
  `CAP_PROP_ORIENTATION_AUTO=1`, and explicit `CAP_FFMPEG` all return the unrotated frame while
  `CAP_PROP_ORIENTATION_META` correctly reports 90. Registry census over the 379: 341 rot-0,
  27 rot-90, 10 rot-180, 1 rot-270. Detection finite-rate degrades monotonically — median
  0.990 / 0.955 / 0.903 / 0.816 — and feature values shift systematically.
- **Anisotropy.** A09's 9.9° median angle error, which affects **all** assets including upright
  ones and is not repaired by excluding the 38.

**Consequences.** M2.8.3 holds BLOCKED; its precondition is M2.8.4's corrected corpus. M2.8.4
absorbs the decode fix (explicit `cv2.rotate` keyed on `CAP_PROP_ORIENTATION_META` in
`video_io.py`, plus a portrait-fixture regression test) alongside the `--tracking body` re-run
it already carried, so one 8.7 h run repairs both. Every §8 fact derived from
`output/corpus-2d/` is **provisional** until that run lands: the 17 trunk/posture columns
return, so the published set becomes 92 columns and 12 × 92 = 1104 feature rows. A02's
measured-partition ruling is what lets that happen with no contract edit, and it is why no
predicate may assert 75, 900 or 17 as a literal.

**A13 — F11 ACCEPTED. The marker's own schema and canonical form are frozen.**
P13 required the census digest to cover "the provenance block minus its own self-referential key"
while freezing neither the keys, the algorithm, nor the serialization, so two conforming
implementations could digest different objects. Frozen, in the sibling idiom
(`inventory.py:1028`, `sessions.py:532`, `corpus_run.py:80`):

- `generation` block keys, exactly: `generator`, `generator_version`, `tree_digest`,
  `input_digests`. `generator` = `"pose-estimation-cohort"`; `generator_version` =
  `cohort.GENERATOR_VERSION`.
- Canonical rendering = `json.dumps(payload, sort_keys=True, indent=2) + "\n"`, UTF-8 encoded.
  Digest = **SHA-256** of those bytes.
- `tree_digest` covers the other three files' bytes plus the canonical rendering of `generation`
  with the `tree_digest` key **removed** — the self-referential key named explicitly, since
  excluding the whole block leaves the upstream claims consumers trust most uncovered.

**A14 — F12 ACCEPTED. Ownership needs a name, not just a version.**
P14 required a marker to "name this generator at this version" while §4 defined no generator-name
field, so a sibling publisher sharing `v1` satisfied the literal rule. Ownership is now the
conjunction `generation.generator == "pose-estimation-cohort"` **and**
`generation.generator_version == GENERATOR_VERSION`, checked on a parsed marker whose path is
`lstat`-confirmed regular and non-symlink before any read. A destination failing either test is
never replaced and never swept.

**A15 — F13 ACCEPTED. N13 fired neither predicate it named.**
Sorting descriptors by label is deterministic across every swept environment and byte-identical on
republication, so it violates neither P15 nor P16 — the same defect shape as F2/N3, a negative
control that cannot fire. Ruled: **canonical descriptor order is `(level, column)`**, added to P15
as its own conjunct, and **P16 is removed from N13** — idempotence cannot detect a stable wrong
order, and claiming it can is the guarantee-vs-claim gap the review lens exists to catch. P15's
sweep is frozen at the sibling's four locales (`LC_ALL=C`, `C.UTF-8`, `en_US.UTF-8`, `LC_ALL` and
`LANG` unset) plus hash seed, timezone, `umask`, `-O` and `--out` name.

**A16 — F14 ACCEPTED. The refusal class is named.**
"Each by exception class" named no class, so a suite asserting `Exception` would credit an
unrelated `IOError` as a pass. Public `cohort.CohortError(Exception)` is added, matching
`CalibrationQcError`, `QualifyError`, `SessionsError` and `InventoryError`. Every expected
publisher and consumer refusal raises it; P17 asserts that class and no supertype.

**A17 — F15 ACCEPTED. CLI, campaign and doc claims are frozen.**
- CLI flags: `--inventory`, `--sessions`, `--run`, `--out`. Entry point
  `pose-estimation-cohort = "pose_estimation.cohort:main"`, taking `entrypoints.md` from 10
  commands to **11**.
- Campaign artifact: `scripts/check_cohort_determinism.py`, beside its three siblings; its decisive
  command is the gate identity in §7 and it is credited by MAIN's rerun.
- P18's "states" gains a text oracle: the four claims are accepted phrases checked as substrings,
  not judged — estimand, view limit, exclusion census, and the A09 claim boundary.
- A synthetic CLI smoke test proves the registered command accepts its flags, since a registered
  entry point that rejects every input still satisfies a name-only index check.

**A18 — F16 ACCEPTED, and it resolves a contradiction A02 and A07 created together.**
A02 sends a column with zero finite values over the run to `excluded`, so it never becomes a
published feature; A07/P08 keeps an all-nonfinite feature present as 12 empty rows. The P08 source
had to be simultaneously absent from `FEATURES` and present in every cell. **The two rules operate
at different grains, and naming the grain dissolves it:**

- **Run-level presence decides publication.** ≥1 finite value anywhere in the run → `published`;
  zero → `excluded` / `structurally_absent`. This is A02, unchanged.
- **Cell-level emptiness decides statistics.** A *published* feature with no finite value **in a
  given cell** publishes empty statistics for that cell and stays in the product. This is P08,
  restated at the grain it always meant.
- The synthetic P08 fixture therefore makes a column finite in ≥1 cell and empty in another, never
  globally empty.
- **P02 cardinality derives from `len(FEATURES) × cells`**, never a literal. A02 permits the
  published set to move, and it will: the corrected corpus takes it to 92 × 12 = 1104.

**A19 — F17 ACCEPTED. Fixed decimals are 9, not 4.**
A03 froze `_cell()` at "fixed decimals" without a count, and the siblings disagree —
`calibration_qc._cell()` renders `.4f`, `qualify` renders `.9f`. Ruled **9**, from the data rather
than from the nearer name: cohort medians run to ~0.003 for normalized displacements and ~0.04 for
grasp apertures, so 4 decimals would publish two significant figures on a continuous measurement.
`calibration_qc` renders QC ratios of order 1, where 4 is sufficient; this publisher does not.
Helper keeps the name `_cell`, takes `qualify`'s precision.

## 11. Verdict table

**Unit BLOCKED, not DONE** (A12). No implementation exists; this table records the wave's harvest so
the resumed unit and MILESTONE-REVIEW dispatch from committed state.

| role | branch tip | worktree | deliverable | outcome |
| ---- | ---------- | -------- | ----------- | ------- |
| `test-m2u83` | `4c5e9be` on `wt/test-m2u83` | `.scratch/worktrees/test-m2u83` | `tests/test_cohort.py`, 1471 lines, all 18 predicates pinned | **17 findings / 17 contract defects / 0 code defects** over 18 predicates. Report committed on the branch at `.scratch/agents/test-m2u83.md`; copy also at MAIN's `.scratch/agents/test-m2u83.md`. |

**Wave verdict: the diff-blind role returned nothing but contract defects, because there was no code
to defect.** 17 findings across 4 batches, each accepted, each landing as an amendment: A01-A07 on
counts, statistics and the P07/P08 contradiction; A08-A11 on the descriptor, the unit vocabulary,
the redaction allowlist and the byte witness; A13-A19 on the marker schema, ownership, a
non-firing negative control, the refusal class, CLI/campaign/doc freezes, the A02-vs-A07 grain
contradiction and the decimal count. Compare M2.8.2's same role: 11 findings / 11 contract defects /
0 driver defects. **Two units running, 28 findings, 28 contract defects, 0 code defects — the
contract, not the implementation, is where this project's defects live, and a diff-blind reader of
the contract alone is what finds them.**

**Outstanding before M2.8.3 resumes**, in order:
1. M2.8.4 lands the corrected corpus (orientation fix + `--tracking body`).
2. Re-encode the suite against the corrected corpus: A08-A19 unencoded, and every literal 75 / 900 /
   17 replaced by a measured cardinality. The successor inherits `wt/test-m2u83`.
3. MAIN implements `src/pose_estimation/cohort.py` + CLI + `docs/technical/cohort.md` against the
   suite, then registers in the four exhaustive indexes (P18).
4. Author the bilingual descriptors and measure the glyph delta against `../rehab`'s 10 subset WOFF2
   faces.

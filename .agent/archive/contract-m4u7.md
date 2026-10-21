# Contract — M4.7 cohort export carries its measurement configuration + generation sensitivity

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
User rulings (this session, after `scientist-2` S08: the RTMW-m 256 → RTMW-L 384 swap moves
221/1056 cells > 0.5 IQR while `cohort.json`'s run digest is byte-identical across generations):
(1) the export records the run's configuration before the adoption republishes; (2) per-feature
sensitivity against a previous generation ships in the export, beside `cohort_features.csv`.

## 1. Artifact

- `src/pose_estimation/corpus_run.py`: `pose_config_digest(config) -> str` = sha256 of
  `json.dumps(config, sort_keys=True)` (UTF-8) — the form `contract-m4u4.md` A01 used.
- `src/pose_estimation/cohort.py`: `GENERATOR_VERSION = "v3"`; `MARKER_KEYS` gains `measurement`
  + `sensitivity`; `run(..., baseline_dir=None)`; `main` gains `--baseline DIR` (optional);
  `SENSITIVITY_FILENAME = "cohort_sensitivity.csv"`, `SENSITIVITY_COLUMNS` frozen (§3); the three
  required published files stay `PUBLISHED_FILENAMES`, the sensitivity file is optional membership.
- Tests: `tests/test_cohort.py` `_build_inputs` writes, per event directory holding an `ok` asset,
  a fixed synthetic `pose_config.json` (`corpus_run.write_pose_config`) + a complete marker carrying
  a fixed `analysis_sha256` (`corpus_run.write_marker`); marker/type/byte/tamper helpers cover the
  new keys + the optional file; `tests/test_cohort_review.py` + `tests/test_m4u5b_agreement.py`
  inherit it with their semantics intact.
- `scripts/check_cohort_determinism.py`: a with-baseline sweep group (fixed separate baseline,
  moved output path, relocated byte-identical baseline, input-row order) + `measurement` /
  `sensitivity` marker tampers + optional-file tamper classes; `tests/cohort_determinism_results.json`
  regenerated last.
- `.scratch/impl/adopt.sh` passes the retained prior cohort as `--baseline`; docs
  (`docs/technical/cohort.md`, `README.md`, `.claude/rules/cohort.md`, `docs/technical/conventions.md`
  campaign census) state v3, the measurement block, the optional fifth file + its reading.
- Out of scope: any review-UI display of the new fields (the UI projects named fields; a synthetic
  v3 bundle smoke confirms its existing views still load).

## 2. Design decisions

**D01 — the measurement block is read per contributing event, never from `run_report.json` or
invocation defaults.** `pose_config.json` is the per-event generation identity the drivers resume
against (`corpus-run.md`); a completion marker names the clinical producer that ran. Contributing
events = the events of the validated `_contributors` join (each event with >= 1 `ok` asset, read
once). One configuration and one clinical method, or the export refuses — a cohort over a mixed
tree describes no instrument. The recorded `analysis_sha256` need not equal today's R source.

**D02 — sensitivity compares published statistics, not trajectories.** The publisher reads two
aggregate tables; matched-support recomputation needs both run trees and stays an analysis
(`docs/technical/pose-weak-points.md` § M4 corpus). No improvement/worse label (`scientist-2` F8).

**D03 — baselines pass a version-aware read-only validator, separate from destination ownership.**
Supported profiles: v1 + v2 (three required files; `cohort.tree_digest` recomputes their recorded
digests — measured on the published v1 and both scratch v2 trees) and v3 (+ the optional file).
The baseline's digest is recomputed and must equal its recorded one; the export records the
verified `generator_version` + `tree_digest` + the sha256 of the feature bytes parsed.
Compatibility never relaxes `_assert_owned` for `--out`. "Upstream digests prove bytes, never
shape" (`publishers.md`) → the feature table is shape-validated too.

**D04 — baseline/output overlap.** Resolved through real paths. Equal roots = the one permitted
overlap (compare against the generation being replaced; the baseline is read whole and checked
stable before the swap; the destination must be owned by this generator version, so a v1/v2
self-baseline refuses through the existing ownership rule → migrate by moving the old tree to a
disjoint path). Strict containment in either direction refuses before any write.

## 3. Schema

- `cohort.json` `measurement` = `{"pose_config": {POSE_CONFIG_FIELDS → scalar}, "pose_config_sha256":
  pose_config_digest(pose_config), "analysis_sha256": <64 lowercase hex>}`.
- `cohort.json` `sensitivity` = `null` without a baseline; with one = `{"baseline":
  {"generator_version", "tree_digest", "features_sha256"}, "status_counts": {"common", "new",
  "retired"}, "estimand": SENSITIVITY_ESTIMAND}`; `SENSITIVITY_ESTIMAND` states: difference between
  published cohort medians of two generations, not paired-subject change and not an accuracy
  result; subjects + retained observations can differ; delta = current − baseline; scaled delta
  divides by the baseline's published IQR; feature definitions may differ across generations.
- `SENSITIVITY_COLUMNS` = `task, side, level, feature, status, n_subjects, baseline_n_subjects,
  n_values, baseline_n_values, median, baseline_median, delta_median, baseline_iqr,
  delta_iqr_scaled`; `status` ∈ {`common`, `new`, `retired`}.

## 4. Predicates

- **P01 measurement.** A published `cohort.json` carries `measurement` = the uniform
  `pose_config.json` of every contributing event, its `pose_config_digest`, and the uniform marker
  `analysis_sha256`. On the real `output/corpus-2d-m4-rtmw-l-384` tree `pose_config_sha256` =
  A01's `45c77741…c132a898`.
- **P02 measurement refusals** (`CohortError`, nothing published, an existing `--out`
  byte-identical). Per contributing event: `pose_config.json` or the completion marker absent, a
  symlink or not a regular file, invalid JSON, duplicate keys, a non-object root; marker `status`
  ≠ complete; `analysis_sha256` absent, non-string, or not a full match of `[0-9a-f]{64}` (`null`,
  empty, `mixed`, wrong length); configuration key set ≠ `POSE_CONFIG_FIELDS`; a value that is a
  container or a non-finite number. Across events: two configurations whose canonical
  serializations differ (a boolean vs an integer included); two `analysis_sha256` values.
- **P03 scope.** An event directory holding no `ok` asset (a failed event, an extra directory) is
  not parsed for `measurement`: its missing, malformed or foreign metadata neither refuses nor
  enters the block. Two assets in one event check that event once. A stale but well-formed
  recorded `analysis_sha256` publishes unchanged.
- **P04 coverage.** Editing any `measurement` or `sensitivity` value in a published `cohort.json`
  (self-digest not recomputed) makes `validate_generation` refuse.
- **P05 baseline invariance.** One input set published once without and once with a fixed separate
  baseline: `cohort_cells.csv`, `cohort_features.csv`, `descriptors.yaml` byte-identical and the
  `measurement` blocks equal; only `sensitivity`, the optional file and the tree digest differ.
  Without a baseline: `sensitivity` = `null`, no optional file — also after replacing an owned
  generation that had one.
- **P06 rows.** With a baseline: one row per key in the union of current + baseline
  `(task, side, level, feature)` keys, sorted by that tuple; `common` = in both, `new` = current
  only, `retired` = baseline only; `status_counts` = the file's counts.
- **P07 values.** Arithmetic on the serialized values: `baseline_iqr` = baseline `q75 − q25`;
  `delta_median` = `median − baseline_median`; `delta_iqr_scaled` = `delta_median / baseline_iqr`.
  A blank operand blanks only the results that depend on it (below-floor medians included); a zero
  or negative `baseline_iqr` keeps `delta_median` and blanks `delta_iqr_scaled`. A `new` row blanks
  baseline fields + deltas; a `retired` row blanks current fields + deltas and keeps its baseline
  counts, median + IQR. Numbers render as `cohort_features.csv` renders them (`DECIMALS`).
- **P08 baseline refusals** (nothing published, `--out` byte-identical). Marker: absent, symlink,
  invalid or duplicate-keyed JSON, `generation.generator` ≠ `GENERATOR`, unsupported
  `generator_version`, `tree_digest` not 64 lowercase hex, recomputed digest ≠ recorded (an edited
  byte under the old digest). Feature table: absent, symlink, header-only, duplicate header name,
  missing any of `task, side, level, feature, n_subjects, n_values, median, q25, q75`, a ragged
  row, an empty or duplicate key, `task`/`side`/`level` outside `inventory.TASKS` /
  `inventory.SIDES` / `LEVELS`, a feature name outside `[a-z0-9_]+`, a count not ASCII `[0-9]+`, a
  statistic neither empty nor a finite decimal, a row below `SUBJECT_FLOOR` carrying a
  distribution value. Overlap: strict containment of `--out` and the baseline in either direction
  (real paths, symlink aliases included). Extra legacy columns are ignored after validation.
- **P09 identity + membership.** `sensitivity.baseline` records the verified version + tree digest
  and the sha256 of the feature bytes parsed. `tree_digest` covers the optional file's bytes and
  presence; `validate_generation` refuses an edited or appended byte, the file removed under a
  non-null `sensitivity`, a file added under `null`, and a directory or symlink in its place.
- **P10 read-only + stable.** Distinct roots: the baseline's recursive non-following snapshot is
  unchanged by a run, and a baseline mutated during the read refuses. Endpoint change detection
  only — a transient mutate-and-restore writer is not excluded.
- **P11 baseline = out.** `--baseline X --out X` with `X` owned by v3 publishes against the
  generation it replaces; with `X` a v1/v2 tree it refuses and leaves `X` unchanged. A failure
  injected before or during the swap leaves the previous output intact.
- **P12 version.** `GENERATOR_VERSION` = `v3`; a v2 tree at `--out` is foreign (existing ownership
  rule), refused unless empty.
- **P13 reproducibility.** Identical inputs + a fixed separate baseline → byte-identical outputs
  under a moved output path, a relocated byte-identical baseline and reordered input rows.

## 5. Live spot check (input-owned — commit body, not a test constant)

L384 tree, baseline = the M2.10 cohort (v1): status counts 888 / 168 / 180; `|delta_iqr_scaled| >
0.5` on 178/888 (`.scratch/impl/probes/cohort_shift.py`; `consultant-4` re-derived both).

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01-P13 | pass | `tests/test_m4u7_cohort_provenance.py` (`tester-6`, diff-blind, `archive/m4u7-test-6`): 322/323 red on 27079c4, 323/323 green at close; P01 real tree `45c77741…c132a898`; §5 888 / 168 / 180, 178/888 |
| campaign | pass | `scripts/check_cohort_determinism.py` 10 sweeps (4 with baseline) + 22 tamper classes; planted path-bearing baseline identity → `drifted: with-baseline-relocated` |

## 8. Amendments

- **A00 (contract review).** `consultant-4`: C01 pass (per-event provenance), C02-C06 fail — typed
  strict measurement refusals, version-aware baseline validation with digest recomputation,
  two-way containment + equal-root rules, blast radius (adopt baseline, campaign, docs),
  tautological P05 replaced by with/without invariance, P13 added. All accepted before freeze,
  except baseline feature names checked against a token alphabet rather than an encoded v1
  feature list (the verified digest binds the baseline to its generator's bytes).

# Contract — M4.5b cohort export: bout level, construct-invalid exclusions, cross-view agreement

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
Input seam = `contract-m4u5a.md` §1 `BOUT_SCHEMA`. Evidence → `.scratch/agents/researcher-5.md`
R02, R08; pre-code review → `consultant-3` (§8).

Purpose: publish the M4.5a bout features at cohort grain; take window SPARC + normalized jerk out
of the export with a stated reason; replace `view_dispersion` (a CV — undefined for SPARC, signed
angles and zero-mean features, R08 / NIST) with a feature-unit agreement summary + a contextual
coefficient every feature can carry.

## 1. Artifact

`src/pose_estimation/cohort.py`, `GENERATOR_VERSION` `v2`. The v1 tree is kept by user ruling
(old generations retained): it moves to a superseded path before v2 publishes, never deleted as
a side effect.

- `LEVELS = ("frame", "window", "bout")`; `LEVEL_SUFFIXES["bout"] = "_clinical_bouts.csv"`;
  the 9 `BOUT_SCHEMA` keys join `METADATA_COLUMNS`. The bout schema is required on every
  successful 2D asset (header-only allowed); a missing file refuses, never reads as zero bouts.
  The reader validates `side` + `censored` + bounds, refuses a non-NA wrong-side value, drops
  every feature value of a censored row, and counts every input row (`n_bout_rows`) separately
  from contributing values. An asset's bout value = median over its uncensored bouts of that
  side; estimand otherwise unchanged (asset → event → subject median → cohort statistic).
- `BOUT_COLUMNS` = the 14 `{left,right}_bout_*` columns: families with ja + en labels, units,
  admissible ranges per `contract-m4u5a.md` §1; unit vocabulary gains `s` and `count`.
- Partition: source = published ∪ excluded, disjoint by name, each excluded column with one
  reason. New reasons: `construct_invalid` — X = window `{side}_wrist_sal`,
  `{side}_wrist_normalized_jerk`, `{side}_fingertip_normalized_jerk` + their `_symmetry_ratio` /
  `_dominance_index` / `_abs_diff` forms (15 names, contract-owned); `columns.excluded` holds
  exactly S ∩ X for source set S, regardless of finite count, taking precedence over every other
  reason; `no_eligible_bout_values` — a declared bout column with no eligible finite value (the
  publisher still runs). Unknown source names still refuse, all-NA ones included.
- `FEATURE_COLUMNS`: `view_dispersion` → `view_abs_diff`, `view_icc`, `view_icc_range_lo`,
  `view_icc_range_hi`; `n_events_multiview` kept; `n_subjects_multiview` added.
- `cohort_cells.csv` gains `n_bout_rows`; marker population gains `bout_rows`; marker
  `estimand` names the agreement estimand beside the distribution estimand;
  `rows_without_multiview` → reason counts `agreement_below_floor`, `agreement_no_multiview`,
  `agreement_zero_variance`, `agreement_insufficient_resamples`.

## 2. Design decisions

**D01 — agreement estimand** = descriptive agreement among available same-event asset
summaries, conditional on feature / task / side and the observed camera + visibility mix. It
includes projection, differing observed time support, segmentation and tracking; it is not
latent-person reliability, same-bout repeatability, clinical validity or accuracy.
`view_abs_diff` (primary, feature units) = median over eligible events of the mean |Δ| over that
event's view pairs. `view_icc` (contextual) = one-way ANOVA coefficient of the same asset
medians: g eligible events (>= 2 finite views), k_i views, N = Σk_i, ȳ view-count-weighted;
MSB = Σk_i(ȳ_i − ȳ)²/(g − 1), MSW = ΣΣ(y_ij − ȳ_i)²/(N − g), k0 = (N − Σk_i²/N)/(g − 1),
ICC = (MSB − MSW)/(MSB + (k0 − 1)MSW), not clipped; requires g >= 2, N > g and a finite positive
denominator (an all-constant sample is undefined, never 1). Events stay separate targets;
missing views are not imputed.

**D02 — range, not a confidence interval.** `view_icc_range_lo/hi` = descriptive 2.5th / 97.5th
percentile (linear) over 1000 subject resamples: draw the eligible subjects with replacement;
each draw adds a distinct copy of all that subject's eligible events + views (multiplicity kept);
subject / event / view order canonicalised before indexing; numpy PCG64 seeded from
`sha256(level/feature/task/side)`; the full statistic recomputed per draw. Endpoints need a
finite point estimate and >= 950 finite replicates; else empty with its reason. Nominal 95 %
coverage is not claimed (consultant-3 measured 0.84-0.92 at 6-16 subjects on Gaussian designs).

**D03 — floor**: every agreement field publishes only when its eligible events come from >=
`SUBJECT_FLOOR` distinct subjects; counts are feature-specific.

**D04 — no categorical labels** (R08: 6-16 clusters; Koo-Li's own sample advice >= 30).

**D05 — consumers**: the review UI (`prototype/review-ui/`, PROTOTYPE law) reads v2 — level
selector with `bout`, agreement fields + ja/en labels, an empty-bout case, v1 comparison kept
while retained generations exist; `.claude/rules/cohort.md` + cohort docs state the
conditional-bout + available-view-summary estimands; `../rehab` unchanged (no pose ingest).

## 3. Predicates

- **P01** ICC estimator: hand fixture events `[0,2]; [3,4,8]; [5,9]` → k0 = 16/7, ICC = 15/31;
  balanced data = Shrout-Fleiss ICC(1,1); identical within-event values with differing event
  means → 1; all-constant → undefined; a negative coefficient unclipped.
- **P02** Resampling helper (below the floor, exhaustive draw fixtures): duplicate subject draws
  copy targets; `[0,0]; [0,2]` → 75 % finite replicates → no range; a two-subject fixture with
  >= 2 differing-mean events each reaches whole-cluster extremes; seeded + order-invariant.
- **P03** Bout level on a synthetic run tree: per-asset medians over uncensored same-side bouts;
  censored rows + wrong-side values never contribute (a non-NA wrong-side value refuses);
  missing bout file refuses; header-only file → zero bouts; `n_bout_rows` counts every row.
- **P04** Exclusions: `columns.excluded` = S ∩ X with `construct_invalid`; a tree omitting some
  of X still publishes; a declared bout column with no eligible value → `no_eligible_bout_values`;
  an unknown all-NA column refuses.
- **P05** Descriptors: every bout feature has ja + en labels, a vocabulary unit, the §1 range;
  `descriptor_collision` shape unchanged.
- **P06** Floor + reasons: agreement empty below 5 distinct multi-view subjects; reason counts
  reconcile below-floor / no-multiview / zero-variance / insufficient-resamples; multiple events
  of one subject do not inflate `n_subjects_multiview`.
- **P07** `view_abs_diff` exact on 2- and 3-view events, finite on all-constant values where the
  ICC is undefined; byte-identical output under contributor / row order permutations.
- **P08** Determinism campaign regenerated last; its `SOURCES` + fixture extended to the bout
  evidence it reads.
- **P09** Review UI renders a synthetic v2 cohort (bout level, agreement fields, empty bout
  feature) — captured through `#cohort` only.

## 6. Negative controls

- NC1 a pooled (event-ignoring) resample reds P02.
- NC2 mean views `k̄` instead of `k0` reds P01's unbalanced case.
- NC3 publishing `view_icc` for a 4-subject cell reds P06.
- NC4 reading a censored row's values reds P03.

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01 | pass | `tests/test_m4u5b_agreement.py` P01 ×4: 15/31, Shrout-Fleiss balanced equivalence, identical → 1, constant → undefined, negative unclipped; NC2 (k̄ ≠ k0) |
| P02 | pass | exhaustive `[0,0];[0,2]` draws → {undefined, 0, 0, -1}, 75 % finite → no range; whole-cluster extremes with multiplicity; seeded + order-invariant; vectorised = scalar estimator per draw; NC1 (pooled ≠ cluster) |
| P03 | pass | censored outlier rows move `n_bout_rows`, never a statistic; wrong-side value + missing bout file refuse; header-only files → every bout column `no_eligible_bout_values`, `n_bout_rows` 0; planted mutant (censored rows read) → 1 failed |
| P04 | pass | `construct_invalid` = present subset of X (2 dropped from the tree → 13); unknown all-NA column refuses; mutant without the X branch → 1 failed |
| P05 | pass | 14 bout descriptors: ja + en, `pose_bout_` raws, `s` / `count` units, ranges; P09/P10 suite rules extended to bout families |
| P06 | pass | floor + four reasons (`agreement_reasons`); 4-subject cell empty (NC3; mutant floor - 1 → 1 failed); a subject's second event does not raise `n_subjects_multiview` |
| P07 | pass | pairwise |Δ| exact on 2 / 3 views, 0 on constants; reversed manifest → byte-identical cells, features, descriptors |
| P08 | pass | campaign `SOURCES` + `view_agreement.py` + bout golden; regenerated last (PASS, 6 sweeps, 15 tamper classes) |
| P09 | pass | review UI on a synthetic v2 cohort (88 features, 1056 rows): level selector with Bout, ICC + range chart, agreement columns, v2 estimand, ja + en (`#cohort` captures, synthetic data only) |

## 8. Amendments

- **A00 (pre-code review, `consultant-3`).** C06, C07, C09, C10 (publisher half) accepted and
  written into §§1-3: descriptive available-view agreement (not repeatability), percentile range
  labelled as such with its measured Gaussian coverage, total + executable statistical
  predicates, empty-level policy, consumer update; C08 (exclusions, abs-diff, floor, no labels)
  passed as drafted.

- **A01 (implementation shapes).** `cohort.json` carries `agreement_reasons` = {the four reason
  tokens → row count} in place of `rows_without_multiview`, and `estimand` = {`distribution`,
  `agreement`, `bout`}. The structurally-absent exclusions are a named set
  (`cohort.STRUCTURALLY_ABSENT`, the 3 sagittal columns) so an unknown all-NA column can refuse.
  The estimator lives in `src/pose_estimation/view_agreement.py` (numpy; resamples vectorised).
  The review UI keeps rendering a retained v1 tree (`view_dispersion` fallback) beside v2.
  The rewritten v1 agreement cases fired on v2 code before their rewrite (P03 census, P04 / P05
  estimate fields, four P06 `view_dispersion` cases, two P07 `rows_without_multiview` cases, P18
  doc phrases: 10 failed); the v2 suite reds 16/60 on the M4.5a commit.

# M3.3 acceptance contract — metric-specific QC evidence

MAIN-authored. Tier `kernel`. 3D-only. Every downstream artifact (`test`, `orc`, `prod`, `rev`, `rev2`, `diff`) decides against this file.

Inputs consumed: `.scratch/agents/map-m3u3.md` (T1 inventory, S2 gate provenance, S3 checklist N01-N21, S4 sizing, S5 test surface, S6 blast radius, S7 open questions), `.scratch/agents/res-m3u3.md` (Q1-Q5), `.scratch/agents/spike-m3u3-{wide,long}.md`.

Superseded: `.scratch/agents/plan-m3.md` "## M3.3" predicates (a)-(f). They propose 2D widening, prefix-level evidence and `interpolation=none` — all three wrong against HEAD. `.agent/roadmap.md:45,60` governs.

## D — decision record

**Carrier = long companion artifact, keyed by `metric_id`.** Settled by arithmetic before spike return, confirmed by spike measurement.

| alternative | why rejected / chosen |
| ----------- | --------------------- |
| wide, metric grain | 42 window metric columns × (status + reason) = **84 new columns** on a 55-column artifact, before any count field. Excluded by arithmetic. |
| wide, source-trajectory grain | 7 window groups × ~8 fields = ~56 columns, doubling the artifact, and **cannot express divergent status within one trajectory** — one interior hole leaves `wrist_velocity_mean` usable while `wrist_movement_efficiency` is not (map T1: `@valid-intervals` vs `@contiguous-path` are distinct support classes over the same keypoint). Fails roadmap `.agent/roadmap.md:45` "metric-required-keypoint status". |
| **long, `metric_id`-keyed — CHOSEN** | Expresses per-metric status natively; leaves the two existing 3D artifacts' column sets untouched; prefigures M3.5's aggregate grain. Costs: a fourth artifact, repeated trajectory counts across sibling rows, and cross-artifact contradiction detection deferred to M3.4 — which `.agent/roadmap.md` already assigns there ("reader rejecting … QC contradictions"). |

**Measured spike evidence.** Both alternatives scored A1 = **7/8**; both miss Fc alone (stale evidence copied from another window stays internally consistent).

| axis | wide (source-group grain) | long |
| ---- | ------------------------- | ---- |
| artifact shape | window artifact **55 → 125 columns**; P1 5,627 → 9,125 B (+62.2 %), P5 3,302 → 8,953 B (+171.1 %) | window artifact unchanged at 55; QC companion 36 columns × 13 rows/window (65 rows for P1) |
| A3 diff | `+222/-27` | `+479/-48` |
| A4 reducer | 119 lines, 3 joins | comparable; join on the composite key |
| A6 grain honesty | **fails P12** — no window row exists, so an attempted zero-window clip cannot be recorded without a dummy row | **passes P12** — typed-empty companion, 36-column header, 0 rows, no dummy row in either grain |
| per-metric divergent status | **not expressible** at group grain | native |

Long is chosen on the roadmap requirement (per-metric status), P12 grain honesty, and leaving the estimate artifacts' column sets untouched. Wide's smaller diff does not offset a 125-column artifact that still cannot express the required grain.

**Metric-keyed over group-keyed.** `spike-m3u3-long` implemented 13 GROUP-keyed rows per window. This contract keys by `metric_id` and carries `source_group` as a column instead. Rationale: the group→metric mapping is M3.4 registry work that does not exist yet, so a group-keyed artifact is not self-describing; metric-keying also buys a detection lever the spike lacked — sibling rows sharing a `source_group` within one window must carry identical counts (P14), which partially closes Fc.

**Corpus corrections adopted from the spikes.** `posture_symmetry` depends on shoulders alone, not the four trunk keypoints (P7) — the partition is finer than seven groups. CPI reports `insufficient_observations` on the zero-variance fixture, so the corpus must not read that as a gate failure.

**Estimates are NOT duplicated into the evidence artifact.** One value, one channel — the M3.2 principle. Fault Fd (`qc_status=pass` while estimate `NA`) is a cross-artifact predicate owned by M3.4.

## R — MAIN rulings on map S7

| id | question | ruling |
| -- | -------- | ------ |
| R-1 | physical schema | Long companion artifact. See D. |
| R-2 | upstream fusion causes (reprojection / cheirality / angle / absent-source / confidence) | **Out of scope.** `qc_reason` names metric-usability causes, never fusion causes. Map S2 proves absent-source and confidence attribution are unrecoverable at the adapter — they need new `src/pose_estimation/export.py` fields. Docs must state the limitation. → `.agent/polish.md`. |
| R-3 | new R confidence gate | **Forbidden.** It would move shipped estimates. M3.3 changes no estimate value. |
| R-4 | SAL interpolation conflict | Emit **no** `interpolation` policy field and **no** `n_interpolated_frames`. The shipped SAL estimand reconstructs interior speed intervals (`analysis/clinical_features.R:541-554`, documented `docs/technical/analysis.md:146-153`); a zero-interpolation claim would be false. `n_valid_intervals` records observed support, so reconstruction stays inferable. |
| R-5 | gap duration definition | Keep the kernel's existing `longest_gap_sec = longest_gap_frames / fs` (`analysis/clinical_features.R:497-500`). Changing it would move an M3.1-shipped value. Document exactly: the count of consecutive missing nominal slots divided by `fs`; the unobserved span between the flanking observed samples is one interval longer. Threshold comparison `longest_gap_sec <= max_gap_sec` is inclusive, on unrounded doubles. |
| R-6 | reason representation | `qc_status ∈ {pass, fail}`. `qc_reason` = precedence-selected primary cause ∈ {`none`, `invalid_timebase`, `missing_required_keypoints`, `insufficient_observations`, `gap_too_long`, `insufficient_coverage`}, that exact precedence order, highest first. Concurrent causes are **reconstructable** from the independent evidence fields rather than encoded in a joined string. |
| R-7 | policy serialization + version bumps | Thresholds ship as literal columns (`min_coverage`, `max_gap_sec`) so status is re-derivable without the M3.4 registry, alongside `qc_policy_version`. Bump `PRODUCER_VERSION` → `v2` (artifact set changed) and `QC_POLICY_VERSION` → `v2` (new policy). `METRIC_METHOD_VERSION` stays `v1` — no estimate moves. |
| R-8 | stale plan predicates | Confirmed superseded. Contract cites `.agent/roadmap.md` alone. |
| R-10 | the "confidence seam" `spike-m3u3-long` reported repairing | **Not a defect; do not adopt the repair.** `analysis/clinical_features.R:1452-1466` branches — 3D takes `adapt_world3d()`, 2D takes `adapt_2d_confidence()`. That is correct: `world3d.csv` confidence is a fused mean over already-accepted points, and fusion applied `min_confidence` upstream (`src/pose_estimation/triangulation.py:538,559-572`). Adding a 3D confidence predicate would create a new gate and move shipped estimates, which R-3 forbids. |
| R-11 | may QC status suppress an estimate? | **No.** QC evidence is advisory and never overwrites a computed value. An estimate is `NA` only where the kernel could not compute it. `spike-m3u3-wide` gated estimates under policy and broke four `tests/test_r_trajectory_kernel.py` cases — that is the failure mode this ruling forbids. |
| R-12 | floating-point threshold comparison | Compare as `longest_gap_sec <= max_gap_sec * (1 + 1e-9)` and `frame_coverage >= min_coverage * (1 - 1e-9)`. A bare `<=` rejects `0.10000000000000009`; the relative tolerance is documented and versioned with the policy. |
| R-13 | `fs` drift interacts with the gap threshold | Record, do not fix here. The spikes disagreed on P3 precisely because `fs = 1/median(diff(ts))` reads 30.03 Hz at 30 fps, so a three-frame gap computes 0.0999 s and passes an 0.10 s threshold that it should sit exactly on. This is the known `.agent/polish.md` `spine?` cadence defect. M3.3 must document that a boundary case is decided within the drift, and the corpus must use cadences where the boundary is unambiguous. Adopting `nominal_fs()` stays out of scope — it moves every shipped metric. |
| R-9 | frame-scope evidence rows | Evidence covers **window scope only**. Frame-artifact row validity is already readable from its own `NA` pattern; a per-frame × per-metric evidence row is ~91 × 30 rows per clip for no decision value. Frame-GRAIN counts (`n_expected_frames`, `n_valid_frames`, `frame_coverage`) are carried per window × metric — this is what `.agent/roadmap.md:45` "frame + interval … counts" denotes. |

## P — testable predicates

| id | predicate |
| -- | --------- |
| P01 | A fourth 3D-only artifact `<stem>_clinical_3d_window_qc.csv` is written beside the existing three, keyed `video × person_idx × window_start_sec × window_end_sec × metric_id`, one row per attempted metric per window. The key is unique. |
| P02 | Field set, in emission order: `video`, `person_idx`, `window_start_sec`, `window_end_sec`, `metric_id`, `source_group`, `n_expected_frames`, `n_valid_frames`, `frame_coverage`, `n_expected_intervals`, `n_valid_intervals`, `interval_coverage`, `valid_duration_sec`, `longest_gap_frames`, `longest_gap_sec`, `n_gaps`, `required_keypoints`, `n_required_keypoints_present`, `min_coverage`, `max_gap_sec`, `qc_status`, `qc_reason`, then the nine `ARTIFACT_TAG_COLS` last. |
| P03 | Arithmetic is exactly: `n_expected_frames` = nominal slots in half-open `[window_start_sec, window_end_sec)`; `n_valid_frames` = slots where every metric-required coordinate passes the gate; `n_expected_intervals` = `max(n_expected_frames - 1, 0)`; `n_valid_intervals` = adjacent expected-slot pairs with both endpoints valid; `frame_coverage` = `n_valid_frames / n_expected_frames`; `interval_coverage` = `n_valid_intervals / n_expected_intervals`; `valid_duration_sec` = `n_valid_intervals / fs`; `longest_gap_sec` = `longest_gap_frames / fs`. Denominators never use observed row count. |
| P04 | Every count derives from the same mask the metric used — `trajectory_metrics()`'s `valid` vector and `grid$n_grid` (`analysis/clinical_features.R:474-500`), reached by extending the kernel's return list, not by recomputing missingness downstream. An independent oracle recomputing from the input agrees exactly. |
| P05 | `qc_status`/`qc_reason` follow R-6, with the precedence order enforced when causes co-occur. |
| P06 | Two metrics over one trajectory receive divergent status wherever their support classes diverge; the corpus proves at least one such case. |
| P07 | Every attempted metric keeps its evidence row, and its estimate cell keeps whatever the kernel produced. Nothing disappears through `na.rm`, and QC status never overwrites a computed estimate (R-11). |
| P08 | M3.3 moves **no** estimate value: `world3d_clinical_3d.csv` and `world3d_clinical_3d_windows.csv` differ from HEAD in the `producer_version`/`qc_policy_version` tag cells only. Proven by regenerating and diffing. |
| P09 | All six 2D goldens are **byte-identical** after a full regeneration run, proven by `cmp`. No 2D header gains a field. |
| P10 | The evidence artifact is always written and typed-empty when nothing qualifies, matching M3.2's semantics; a rerun clears stale content. Empty and populated headers are identical. |
| P11 | Thresholds are labelled engineering-provisional in code comments and docs; no doc claims clinical validation. Policy changes ride `qc_policy_version`. |
| P12 | The producer's shared frame/window constructors stay 2D-safe: evidence is built in a 3D-only branch or behind an `is_3d` guard (map S6 flags both constructors as shared). |
| P13 | `docs/technical/analysis.md` documents the artifact, every field, the exact arithmetic, the reason vocabulary and precedence, the gap definition of R-5, and the R-2 limitation. |
| P14 | Within one window, every evidence row sharing a `source_group` carries identical count, coverage, duration and gap values. Only `metric_id`, `qc_status` and `qc_reason` may differ. |
| P15 | `posture_symmetry` depends on the two shoulders alone; `trunk_*` depends on the four trunk keypoints. Gating both hips leaves posture `pass` and trunk `missing_required_keypoints`. |

## I — invariant surfaces

Unchanged by this unit: the 20 px / 1° / cheirality / `min_confidence=0` gate selection (`analysis/clinical_features.R:49-69`; `src/pose_estimation/triangulation.py:420-424,538`); complete-trajectory metric formulas; 1 s / 50 % window geometry; existing 3D suffixes and 2D file names; the nine `ARTIFACT_TAG_COLS` as the final block on every 3D artifact; `METRIC_METHOD_VERSION`.

## G — gate identity

Decisive gate = **primary tree**, committed state, zero skips:
`uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest`
Baseline from `cc8a939`: **559 passed, 0 skipped, 437 s**. R gate `pytest tests/test_r_pipeline.py`. Changed `analysis/*.R` exits 0 under `Rscript` with project renv. Worktree greens are not evidence for `analysis/*.R`.

Known loud breaks to repair (map S5): `tests/test_r_clinical_goldens.py:128-193` exact bytes/header/schema; `tests/test_r_identity_schema.py:49-180,230-250,266-273` hard-coded base schemas; `tests/test_r_identity_schema.py:297-387,534-543` typed-empty header equality; `tests/test_r_trajectory_kernel.py:501-505` the M3.1 2D no-evidence source-text guard, which bans the names `dropout`/`longest_gap_sec`/`skipped_slots` anywhere and cannot read branch reachability — replace it with exact 2D output-byte assertions.

## C — probe corpus seed

P1 complete 91-frame no-gap · P2 one interior frame dropped, left wrist · P3 three consecutive dropped · P4 fifteen scattered dropped · P5 whole right side gated · P6 fingertip gated, wrist intact · P7 hips gated, shoulders intact · P8 coverage 0.79 / 0.80 / 0.81 · P9 gap 0.09 / 0.10 / 0.11 s · P10 window under four samples · P11 duplicate + reversed timestamps · P12 clip yielding zero windows · P13 P2 at 24 Hz and 60 Hz · P14 divergent-status case for P06 · P15 multi-cause precedence, every adjacent pair in the R-6 order.

Gate failures are injected by corrupting fusion DIAGNOSTIC columns, never by writing `NA` coordinates directly.

## N — normative checklist binding

Map S3 N01-N21 all bind. Status corrections carried here: N05, N07, N08, N11 are already satisfied by `0fa2079`/`b1f5b81`/`16e6fab` in kernel or scope, with emission still open. N10 is resolved by R-4. N03/N06 are narrowed by R-2 — evidence derives from the adapter's per-keypoint gate mask; cause attribution is out of scope.

## V — MAIN verdicts on `test-m3u3` phase 1

| id | ruling |
| -- | ------ |
| V01 | `metric_id` inventory = exactly the 42 numeric columns of `window_schema()`, in emission order: `left_*` (6), `right_*` (6), `WINDOW_BODY_METRICS` (12), bilateral (18). Every attempted metric gets a row, `NA` estimate included. |
| V02 | Confirmed. Counts come from the required-keypoint validity mask; status diverges per metric. This is P06 + P14. |
| V03 | `source_group` vocabulary, exactly nine: `left_wrist`, `right_wrist`, `left_fingertip`, `right_fingertip`, `bilateral_wrist`, `bilateral_fingertip`, `trunk`, `shoulders`, `cpi`. |
| V04 | `required_keypoints` = comma-joined canonical column prefixes, dependency order, no spaces. |
| V05 | **No alternation grammar.** `required_keypoints` lists the MANDATORY set alone. CPI lists the four trunk keypoints; its wrist alternation lives in the validity mask, where a slot counts valid when trunk lean and at least one reach are finite. Document the asymmetry rather than encoding a grammar in a CSV cell. |
| V06 | `missing_required_keypoints` fires when a required column is absent OR `n_valid_frames == 0`. A wholly gated side therefore reads `missing_required_keypoints`, matching the spike P5/P6/P7 measurements. |
| V07 | Present with one gate-passed frame ⇒ not `missing_required_keypoints`; the shared observation rule below decides. |
| V08 + V20 | The policy gates **`frame_coverage` only**, uniformly, for every metric. `interval_coverage` ships as evidence and gates nothing. One rule, one version, no per-family selectivity. |
| V09 + V10 + V14 + V26 + V28 | **New reason `estimator_undefined`, lowest precedence.** `qc_status = pass` requires policy pass AND a finite estimate; policy pass with an `NA` estimate is `fail` / `estimator_undefined`. This keeps "pass implies finite estimate" a hard invariant, which is what makes M3.4's contradiction check meaningful. **No per-metric minimum registry is published**: `insufficient_observations` fires only from the shared rule `n_valid_frames < 2` or `n_valid_intervals < 1`; every other undefined estimate is `estimator_undefined`. |
| V11 | Numeric zeroes, never `NA`, on a complete window. |
| V12 | All slots invalid ⇒ `n_gaps = 1`, `longest_gap_frames = n_expected_frames`, `n_valid_frames = 0` ⇒ `missing_required_keypoints`. |
| V13 | Zero interval denominator ⇒ `interval_coverage = NA_real_`; `n_valid_intervals = 0` and `valid_duration_sec = 0`. |
| V15 | Leading and trailing runs count; `rle()` already includes them. |
| V16 + V18 + V19 | Apply R-12 exactly: `longest_gap_sec <= max_gap_sec * (1 + 1e-9)`, `frame_coverage >= min_coverage * (1 - 1e-9)`, on unrounded values. |
| V17 | Use exact rational timestamp grids and assert all six outcomes. Per R-13, avoid 30 Hz for the decisive boundary case — `fs` drift puts a three-frame gap at 0.0999 s. |
| V21 + V22 + V23 + V24 | `invalid_timebase` fires when the nominal grid cannot be built for that group. Its row is retained with keys, `source_group`, `required_keypoints`, thresholds and tags populated, and **every count, coverage, duration and gap field `NA`**. When no window can be formed at all, the artifact is typed-empty and no row exists. |
| V25 | Precedence decides: one valid frame of 30 satisfies both rules, and `insufficient_observations` outranks `insufficient_coverage`. Not the agent's recommendation — determinism outranks the more descriptive label. |
| V27 | SAL counts observed support only and passes when policy and the observation rule pass. Its interior reconstruction is documented, never claimed away (R-4). |
| V29 | Tags: `artifact_kind = "window_qc"`; `source_sha256` = the same input-file hash the other three 3D artifacts carry; `metric_qualification = "gap-aware"`; `provenance_class`, `coord_space`, `distance_unit` and the three versions from the constants. |
| V30 | Deterministic order: `video`, `person_idx`, `window_start_sec`, then canonical V01 metric order. |

`QC_REASON_PRECEDENCE` is therefore, highest first: `invalid_timebase`, `missing_required_keypoints`, `insufficient_observations`, `gap_too_long`, `insufficient_coverage`, `estimator_undefined`.

# M2.4 acceptance contract — Timebase truth

Tier `kernel`. Baseline `6bbd50e`, gate 1116 passed / 0 skipped. Frozen at dispatch of Wave 2; every
downstream artifact decides against this file. Amendments land in §8 in the same turn as the code
they rule.

## 1 Problem

`src/pose_estimation/export.py:272,299,334,505` rounds `timestamp_sec` to 4 decimals. The R producer
then infers cadence as `1 / median(diff(ts))`, whose quantisation bias is ~1e-3 relative:

| nominal | `1/median(diff)` | err | `nominal_fs` | err |
| ------- | ---------------- | --- | ------------ | --- |
| 30.000 | 30.030030 | +0.03003 | 29.999900 | −0.00010 |
| 29.970 | 29.940120 | −0.02988 | 29.969930 | −0.00007 |
| 59.940 | 59.880240 | −0.05976 | 59.940260 | +0.00026 |
| 60.000 | 59.880240 | −0.11976 | 60.000200 | +0.00020 |
| 100.000 | 100.000000 | 0 | 100.000000 | 0 |
| 119.880 | 120.481928 | **+0.60193** | 119.879820 | −0.00018 |

`nominal_fs()` (`analysis/clinical_features.R:436`) cancels the bias and ships unadopted at two call
sites. 100 Hz is exact under both estimators because 0.01 s is representable in 4 decimals, so a
100 Hz fixture cannot detect the defect.

## 2 Scope

IN — `analysis/clinical_features.R` cadence estimation and every consequence it forces: the QC
threshold comparison, the committed goldens, the version tags, `docs/technical/analysis.md`, and the
Python QC oracle that independently recomputes cadence.

OUT — `src/pose_estimation/` timestamp production and rounding stay unchanged; the fix is downstream
estimation, never a source-clock or export-schema change. `analysis/data_extraction.R:100-112` and
`analysis/arthrose_diag.R:77-100` divide angular change by each rounded row-to-row interval; those are
separate surfaces and go to `.agent/polish.md`. Movement-phase gap-unsafety stays standing scope.

## 3 The finding that shapes the unit

Adopting any estimator makes the `gap_too_long` comparison depend on that estimator's residual.
`QC_POLICY_TOLERANCE = 1e-9` (`analysis/clinical_features.R:121-124`) is sized for IEEE754
representation slack — its own comment says so — and the estimator residual is ~1e-5 relative, four
orders of magnitude above it. Measured consequence at nominal 30 Hz, 3-slot gap, sweeping clip
length:

| frames `n` | `(n-1) % 3` | `nominal_fs` | `3/fs` | verdict |
| ---------- | ----------- | ------------ | ------ | ------- |
| 58 | 0 | 30.00000000 | 0.1000000000 | pass |
| 59 | 1 | 30.00051725 | 0.0999982759 | pass |
| 60 | 2 | 29.99949153 | 0.1000016949 | **FAIL** |
| 61 | 0 | 30.00000000 | 0.1000000000 | pass |
| 62 | 1 | 30.00049181 | 0.0999983607 | pass |
| 63 | 2 | 29.99951614 | 0.1000016129 | **FAIL** |

A QC verdict that depends on total clip frame count mod 3 has no physical meaning. The residual is
irreducible: `nominal_fs` telescopes to `span / (n-1)`, each endpoint carries ≤ `q/2` of rounding
error with `q = 1e-4 s`, so relative error ≤ `q / span` and no estimator on 4-decimal timestamps does
better. Snapping the estimate to a standard-rate table is rejected — the corpus's real per-file rates
are 29.963-29.987 Hz, which are not standard rates, and snapping would destroy the per-file cadence
truth this unit exists to deliver.

Therefore the threshold comparison must absorb the estimator's own accuracy, and the tolerance that
does so must be published, because the QC artifact already publishes `min_coverage` and `max_gap_sec`
(`analysis/clinical_features.R:1213-1214`) and a consumer recomputing the verdict from those columns
must reach the producer's answer.

## 4 Predicates

Estimator.

- **P01** `nominal_fs()` gains an explicit magnitude mode. The default keeps today's
  positive-difference contract; no caller gets magnitude semantics implicitly.
- **P02** Window enumeration (`analysis/clinical_features.R:1250-1252`) uses the magnitude mode. No
  `median(diff(...))` cadence inference remains anywhere in `analysis/clinical_features.R`.
- **P03** Movement segmentation (`:1634-1636`) uses the default mode.
- **P04** A strictly-descending clip still keys its windows and publishes `invalid_timebase`. V21-V24
  (`tests/test_r_qc_evidence.py:777-814`) stay green with no expectation edited.
- **P05** A clip with fewer than two usable positive intervals yields no cadence and skips the group
  exactly as it does today.
- **P06** Recovered cadence relative error ≤ 1e-4 at nominal 30, 29.97, 59.94, 60, 100 and 119.88 Hz
  over 4-decimal timestamps, and no worse than `1/median(diff)` at every one of them.
- **P07** The estimator's error bound `|Δfs/fs| ≤ TIMESTAMP_QUANTUM / span` is stated in code and
  pinned by a test that measures it across a span sweep.

QC policy.

- **P08** `QC_POLICY_TOLERANCE` is sized to the P07 bound for spans ≥ 1 s and its comment states
  estimator slack rather than representation slack. Its value stays far below one frame period at
  every supported cadence, so it cannot mask a real gap violation.
- **P09** The nominal 30 Hz 3-slot gap verdict is `pass` at every clip frame count. Pinned across a
  contiguous length sweep covering all three residues of `(n-1) mod 3`.
- **P10** The nominal 60 Hz 6-slot gap verdict is `pass` and the 7-slot verdict is `fail`.
- **P11** The 100 Hz 10-slot `pass` / 11-slot `fail` boundary is unchanged.
- **P12** `qc_status` is reproducible from the published QC row alone: the tolerance joins
  `min_coverage` and `max_gap_sec` as a published column, and a test recomputes every verdict from
  published columns and matches the producer.
- **P13** Version tags bump exactly where their own domain changed — `QC_POLICY_VERSION` (policy),
  `METRIC_METHOD_VERSION` (metric values move), `PRODUCER_VERSION` (schema gains a column). Identity
  tests carry the new values.
- **P14** The Python QC oracle (`tests/test_r_qc_evidence.py:413-451`) stops recomputing
  `1/median(deltas)` and implements the quantisation-robust cadence independently of the producer.

Artifacts.

- **P15** Goldens regenerate. The three frame-level goldens move in tag columns alone; the window and
  QC goldens move only in the 25 fs-dependent metric columns plus `valid_duration_sec`,
  `longest_gap_sec` and tags. Every moved column is explained by a stated mechanism.
- **P16** Two consecutive regenerations produce byte-identical goldens.
- **P17** Kernel bit-identity at fixed `fs` holds unchanged (`tests/test_r_trajectory_kernel.py:232-245`).
- **P18** `docs/technical/analysis.md:291` no longer calls the defect unfixed, and `:159-163`
  describes the adopted estimator and the published tolerance.

Evidence and gate.

- **P19** Full suite green in the primary tree, 0 skips, count ≥ 1116.
- **P20** Real-corpus evidence: `nominal_fs` recovers header fps within the P06 bound on the scout
  sample, and the grid residual stays within `GRID_SLOT_TOLERANCE = 0.25` under the adopted estimator.

## 5 Invariant surfaces

1. Descending-clip QC publication — a malformed timebase is published as `invalid_timebase`, never
   dropped before window keying.
2. Complete-grid kernel bit-identity at fixed `fs` — the caller's cadence is the intended change; the
   kernel's arithmetic is not.
3. Verdict reproducibility — the published QC row alone determines `qc_status`.
4. Golden determinism — the artifact is a function of its input bytes.

## 6 Gate identity

`env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync pytest`, primary tree. Both halves are
load-bearing; without them `conftest.py` dies at `ImportError … GLIBC_2.43 not found`. Baseline at
`6bbd50e` = 1116 passed / 0 skipped.

## 7 Probe-corpus seed — 8 classes

1. Exactly-representable cadence (100 Hz) — the blind spot; must stay unchanged.
2. Non-representable cadence (30, 29.97, 59.94, 60, 119.88 Hz).
3. Clip-length sweep at 30 Hz covering all residues of `(n-1) mod 3`.
4. Descending, non-monotonic, duplicate and identical timestamps.
5. Gaps at, below and above the 0.10 s boundary at each cadence.
6. Gapped clips where the `GAP_INTERVAL_FACTOR` filter engages.
7. Short clips (span < 1 s) where the estimator bound is loosest.
8. Real-corpus decode timestamps.

## 8 Amendments

MAIN's batch ruling on `test-m2u4` phase 1 (80 cases, 26 ambiguities). Each row binds for the unit.

| # | ruling |
| - | ------ |
| A01 | Interface = `nominal_fs(t, magnitude = FALSE)`, a logical parameter. Stable name; tests call it. |
| A02 | Both readings bind. Behavioural tests decide the algorithm; one source scan asserts no reciprocal-of-median-diff cadence expression survives in `analysis/clinical_features.R`. |
| A03 | Call site B passes `magnitude = FALSE` explicitly. A future default change must not silently move segmentation. |
| A04 | P04 widens: EVERY malformed timebase that publishes `invalid_timebase` today keeps publishing it — descending, duplicated, non-finite, non-monotonic. Descending is the named instance, not the boundary. |
| A05 | P05's "usable positive intervals" are counted AFTER the finite-and-positive filter and BEFORE the `GAP_INTERVAL_FACTOR` cut. |
| A06 | **P06 gains a span floor: it binds for clips of span ≥ 1 s.** Below that the P07 bound alone governs and no absolute accuracy is claimed. The literal all-span reading was mathematically impossible and is withdrawn. |
| A07 | "No worse than" = `err_new <= err_legacy * (1 + 1e-12)`. Equality at exactly-representable cadences satisfies it. |
| A08 | `span` = last finite timestamp minus first finite timestamp, in magnitude. Gaps do not shorten it; magnitude mode measures it on absolute values. |
| A09 | P07 is pinned against the analytic bound with 1 ULP of slack, never a strict binary float comparison. |
| A10 | Tolerance value = exactly `1e-4`, a named constant derived as `TIMESTAMP_QUANTUM / MIN_CADENCE_SPAN_SEC` = `1e-4 / 1.0`. Not composed with the representation tolerance. |
| A11 | **Split the tolerances.** Coverage carries no estimator error, so it keeps the `1e-9` representation slack. The new `1e-4` estimator slack applies to the gap comparison alone. One shared 1e-4 band would have admitted coverage below 0.80 with no estimator justification. |
| A12 | P08's margin is asserted numerically: `QC_MAX_GAP_SEC * tolerance` = 1e-5 s against `1 / 120` = 8.33e-3 s at the maximum supported cadence, a ratio ≥ 100×. |
| A13 | P09's sweep = exactly `n = 58..63`, six contiguous lengths covering every residue of `(n-1) mod 3` twice. |
| A14 | P09-P11 require full `qc_status == "pass"`. Fixtures are engineered so no other reason can fire, which is what makes the gap boundary the thing under test. |
| A15 | The exact-0.10 s boundary exists only where `0.10 * fs` is an integer — 30, 60, 100, 120 Hz. At 29.97/59.94/119.88 Hz the class-5 cases test the floor and ceiling lattice neighbours and assert the verdict follows the comparator. |
| A16 | Published column = `qc_policy_tolerance`, a relative fraction, one constant value per file, emitted beside `min_coverage` and `max_gap_sec`. |
| A17 | **P12 narrows to threshold-decided reasons — `gap_too_long` and `insufficient_coverage` alone.** `invalid_timebase`, `missing_required_keypoints`, `insufficient_observations` and `estimator_undefined` are evidence-decided, and the row does not carry what would be needed to re-derive them. The claim is that a consumer reproduces every THRESHOLD verdict from the published row, not every reason. |
| A18 | `PRODUCER_VERSION` → `v3`, `METRIC_METHOD_VERSION` → `v2`, `QC_POLICY_VERSION` → `v3`. |
| A19 | P15 = allowlist. No column outside the named set may move, and each named column must move in at least one row. Not every cell of a named column must move. |
| A20 | The 25 fs-dependent columns ship as a frozen explicit list in the test, never a count. |
| A21 | P16 compares every file in a preserved output directory and asserts no unexpected artifact exists, never the committed whitelist alone. |
| A22 | P20's sample = `scout-m2u4`'s committed deterministic probe; MAIN supplies the path at implementation. |
| A23 | P20 tests exporter-rounded timestamps, never raw PTS. Raw PTS bypasses the defect M2.4 fixes. |
| A24 | P20's fps authority = the inventory/cv2 header value already published in `assets.csv`. |
| A25 | P20's residual = per-file maximum, over `WINDOW_SEC` windows, of `abs(r - round(r))` with `r = (t - t[0]) * fs`, computed AFTER 4-decimal rounding. |
| A26 | **The suite carries two case kinds.** RED cases fail at baseline `6bbd50e` and pass after. CONTROL cases pass at baseline and must stay passing — P04, P11 and P17 are controls, because an unchanged invariant cannot both isolate its predicate and fail baseline. "All red at baseline" binds the red cases alone. |
| A27 | **C7.06 and C7.07 are vacuous as written and assert absence.** `analysis/clinical_features.R:1261` skips a group whose observed span is under `WINDOW_SEC`, so a sub-1 s clip emits no window and therefore no QC row. Neither case can carry a gap verdict; each asserts that no window row exists. P09's "every clip frame count" is bounded by A13's `n = 58..63` sweep, every member of which spans ≥ 1 s at 30 Hz. |
| A28 | **Both slacks publish.** A16's `qc_policy_tolerance` carries the estimator slack applied to the gap comparison; a second column `qc_coverage_tolerance` carries the representation slack applied to the coverage comparison. A17 makes P12 bind on `gap_too_long` AND `insufficient_coverage`, and a consumer cannot reproduce the coverage verdict from a slack it never receives. Both are dimensionless relative fractions, constant per file, emitted beside `min_coverage` and `max_gap_sec`. Two added columns are a schema gain under `PRODUCER_VERSION` v3, never a "moved column" under A19. |

### Case-expectation rule

The phase-1 table holds **82 rows**, not the 80 its prose counts; `scripts/check_m2u4_suite_seed.py`
`CASE_COUNTS` is the settled set. Each row's own `your reading` column is MAIN's ruled expectation,
**except where an amendment overrides it**. Known overrides: C3.01-C3.06 take A13 + A14 (fixed
`n = 58..63`, full `qc_status == "pass"`); C5.02 takes A14; C5.20 takes A11 and therefore **fails** as
`insufficient_coverage`, because coverage keeps the `1e-9` representation slack; C7.02 and C7.03 take
A06's ≥ 1 s span floor; C7.06 and C7.07 take A27 and assert no window row; C8.01-C8.08 take A22-A25.

### Amendment to §3 and §4 arising from A11

`QC_POLICY_TOLERANCE` splits into two named constants: the existing representation slack on coverage,
and an estimator slack on the gap comparison. §3's argument is unchanged — it justifies the estimator
slack — and the published column of A16 carries the estimator slack, since that is the one a consumer
needs to reproduce a gap verdict.

### Real-corpus evidence standing at ruling time

`scout-m2u4` pilot, 10 assets / 13 043 frames, exporter-rounded decode timestamps: `nominal_fs`
places 610/610 one-second windows on the grid; `1/median(diff)` places 530/610. Whole-clip grids are
10/10 and 0/10 respectively, but the producer never checks a whole clip — `trajectory_grid_status`
runs on window timestamps — so the window figure is the one that describes published rows. MAIN's own
mechanism check on synthetic 29.97 Hz CFR data: biased-estimator slot residual grows ~0.03 slots per
second of span, crossing `GRID_SLOT_TOLERANCE` at ~8.3 s and saturating at 0.5, while `nominal_fs`
stays ~0.002 flat. Real PTS jitter makes real residuals larger than that synthetic lower bound.

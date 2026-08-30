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

MAIN's batch ruling on `test-m2u4` phase 1, over the phase-1 table's **82 cases**, carried by the **35 amendments** below. Each row binds for the unit. The phase-1 prose that counts 80 is corrected in §8's closing note; `scripts/check_m2u4_suite_seed.py`, this intro and the roadmap agree on 82.

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

| A29 | **P15's golden census is corrected by measurement.** The committed set holds 4 frame artifacts, 4 window artifacts and 1 QC artifact, never "three frame-level goldens". Measured moves under adoption: 3 × 2D `*_clinical.csv` stay byte-identical; `world3d_clinical_3d.csv` moves in `producer_version` + `metric_method_version` + `qc_policy_version` alone, 91/91 rows; 3 × 2D `*_clinical_windows.csv` move in fs-dependent metric columns alone, no tag column existing there; `world3d_clinical_3d_windows.csv` moves in fs-dependent metric columns plus the 3 version tags; `world3d_clinical_3d_window_qc.csv` moves in `valid_duration_sec` 60/60 plus the 3 version tags, and GAINS `qc_policy_tolerance` + `qc_coverage_tolerance`. **`longest_gap_sec` moves in no row and leaves A19's must-move set**: the golden fixture is a complete grid, so `longest_gap_frames` is 0 on every row and the quotient is 0 at any cadence. Gap-second cadence dependence is covered by the QC corpus oracle's gapped cases instead. No `qc_status` or `qc_reason` cell moves in any golden; both intended verdict flips live in the corpus fixture. |

| A30 | **The three version tags identify the producer RELEASE that wrote the 3D artifact set, and P13's domain test is evaluated over that set.** `attach_artifact_tags()` writes one triplet across every 3D artifact, so a per-file domain reading is unimplementable without a redesign. All three domains changed in M2.4 — policy gained a second tolerance, window metric values moved, the QC schema gained two columns — so all three bump, and `world3d_clinical_3d.csv` carries the release that wrote it rather than a claim about its own contents. A consumer's rule is "pool only across identical triplets". `docs/technical/analysis.md` states that granularity in this unit; per-file version vectors go to `.agent/polish.md`. |
| A31 | **P20 splits, because it conflated two quantities.** (1) BINDING — under `nominal_fs`, every sampled asset's per-window grid residual stays within `GRID_SLOT_TOLERANCE`, measured head-to-head against the legacy estimator. (2) CROSS-CHECK, not a gate — agreement with the container's header fps is reported as an outlier count plus the worst case. The header is the demuxer's average-rate claim, never ground truth for decode timestamps, and the corpus already measures inter-frame PTS as non-uniform on 10 of 10 probed clips. (3) The estimator's own 1e-4 accuracy claim (P06) binds on synthetic timestamps of KNOWN cadence alone, which is the only population where truth exists. The discriminator is P07: at a 15.08 s span the analytic bound is 6.6e-6, so a measured 1.1e-4 header disagreement is 16× above anything the estimator can produce. **The mechanism is measured, not inferred: the two rates carry different denominators.** The header divides `n_frames` by a container duration that includes the terminal frame's own duration; `nominal_fs` divides `n_frames - 1` by the timestamp span, which omits it. Under constant frame rate the two are algebraically identical, so they agree exactly. Under VFR they separate by `(terminal_frame_duration - mean_interval) / span` — on the outlier, `(0.0350 - 0.03337) / 15.08 = 1.08e-4` against the measured 1.10e-4. Header disagreement therefore measures capture rate uniformity, never estimator accuracy. A truthful outlier publishes; it never re-selects the sample. **Confirmed at full-corpus scale, 379 assets:** the header clause fails 4/379 with a worst case of 1.46938e-4, across both h264 and hevc and 3 of 4 device configurations — a real property of the corpus, not sampling luck. The binding grid clause passes **21 651/21 651** windows under `nominal_fs` against **21 571/21 651** legacy, every one of the 80 legacy failures coming from a single 119.97 fps asset, and every asset's `nominal_fs` residual is no worse than legacy's. |
| A32 | **P19 gains a reconciliation clause.** `passed >= 1116` is the pre-seed floor and permits a case to vanish. The gate claim additionally requires collected == passed, and zero skipped, xfailed, xpassed, deselected or errored outcomes. The seeded suite adds 82 cases, so a conforming run reports at least 1198 passed. **Measured after the seed landed: 1199 collected**, reconciling as 1116 baseline + 82 seed + 1 — the extra being `test_estimator_slack_is_load_bearing_across_clip_length`, added to `tests/test_r_qc_evidence.py` in the same commit to hold A11's tolerance split. A green close therefore reports **1199 passed, 1199 collected, zero non-pass outcomes**; any count below that means a case vanished. |
| A33 | **A04 loses its non-finite class; the remaining three classes stand as a control.** `invalid_timebase` is a window-level ordering verdict, not a series-level validity verdict. A window publishes it only after the enumerator keys that window, which requires a finite positive `nominal_fs`, finite `min`/`max`, and a span of at least one window (`clinical_features.R:1294`, `:1300`, `:1305`). Descending, duplicated and non-monotonic timestamps all clear those guards — `magnitude = TRUE` is what recovers a positive rate from them, per A03 — so each keys its windows and publishes 12 `invalid_timebase` rows, exactly as A04 requires. Non-finite input never reaches the verdict: interior `Inf`/`-Inf` poisons the span, and all-identical timestamps leave zero positive intervals, so both fail a guard and the person is dropped with **0 rows published**. M2.4 did not create that behavior and did not widen it — the pre-swap `median(diff(t))` estimator dropped the same two classes at the same guard — so A04's three surviving classes remain a genuine no-expectation-edit control. The silent whole-person drop is a real defect against the QC pass's stated purpose of recording a malformed clip rather than losing it, but it is off this unit's spine → `.agent/polish.md`, acceptance check: interior `Inf` in one person's timestamps publishes that person's windows with `qc_reason == "invalid_timebase"` rather than publishing nothing. |
| A34 | **P07's bound binds on a contiguous retained run, and it was never what grid correctness rested on.** The estimator is `1 / mean(d[keep])` (`clinical_features.R:472`). Retained-interval rounding errors telescope only within one uninterrupted run, so the endpoints carry the whole error and the bound is `TIMESTAMP_QUANTUM / span` exactly when `GAP_INTERVAL_FACTOR` cuts nothing. Cutting `k - 1` gaps leaves `k` disjoint runs whose errors no longer telescope, and the bound generalizes to `k * TIMESTAMP_QUANTUM / S_retained`, with `S_retained` the summed exposure of the retained intervals; `k = 1, S_retained = span` recovers the original. `rev-m2u4-2`'s random-drop excesses over the literal bound — 12.35x at 30 Hz, 19.41x at 60 Hz, 24.74x at 119.88 Hz on 20 s probes — are the `k > 1` regime and are predicted by the corrected form, not evidence of a defective estimator. **The correction reaches no downstream claim, but the margin must be stated against the right denominator.** `rev-m2u4-2` verified the corrected bound contains every measured error on the seed-0 worst probes — 30 Hz `k=42`, err 6.17568e-5, bound 2.51012e-4; 60 Hz `k=83`, err 9.70391e-5, bound 4.93117e-4; 119.88 Hz `k=170`, err 1.23672e-4, bound 1.02141e-3 — so no breach, and it correctly rejected an earlier "order of magnitude" phrasing of the grid margin. The producer grids **windows, never whole clips**: `trajectory_grid_status(win_ts, fs)` at `clinical_features.R:1332` takes one window's timestamps, so the lever arm is `WINDOW_SEC = 1.0` and the budget is the shipped `GRID_SLOT_TOLERANCE = 0.25`, not a half slot. A relative rate error displaces the window's last sample by `WINDOW_SEC * fs * abs(delta_fs / fs)` slots, giving measured margins of **135x at 30 Hz, 43x at 60 Hz and 17x at 119.88 Hz** — the operative figures, since these are the windows that publish rows. Read against a whole-20 s clip instead, the same errors leave a minimum margin of **1.686x**: still holding, but close enough to the edge to be worth recording, because it says whole-clip gridding would have no headroom at high frame rates. P20 measures the residual directly rather than inferring it from P07 either way. P07 is therefore a precision statement about the estimator; the load-bearing operational guarantee is the grid residual. A08 keeps `span` unshortened for coverage and gap accounting, where it is the right denominator; it does not supply P07's denominator. |

| A35 | **"Recover or fail closed" binds on the pipeline, not on `nominal_fs`.** C6.08's phase-1 demand assumed one function owns both jobs. It does not, and the split is deliberate. The gap filter's premise is that gaps are a minority; C6.08 breaks that premise — seven samples at 30 Hz spaced 2,2,2,2,1,1 slots give four gap intervals against two base ones, so `median(d)` is itself a gap, `d <= GAP_INTERVAL_FACTOR * median(d)` retains every interval, and the mean blends to **18.0018 Hz**. `nominal_fs` returns that confidently and does not fail closed. `trajectory_grid_status()` is the adjudicating layer, and it does: measured `fault` = `timestamps do not follow a 18.0018 Hz grid (residual 0.401)`, against `GRID_SLOT_TOLERANCE = 0.25`. The published consequence is `invalid_timebase`, which is failing closed — one layer later than the case assumed. C6.08 is corrected to assert the pipeline property `recovered or fault`, which preserves the phase-1 intent, relocates it to the layer that owns it, and stays true if a future estimator recovers the true cadence instead. Recovering a cadence from a gap-majority series needs slot-count search rather than interval statistics, which is off-spine → `.agent/polish.md`. Safe to defer precisely because the consumer rejects the blend rather than trusting it. |

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

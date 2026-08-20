# M3 closed-unit record

Detail for DONE units, moved out of `.agent/roadmap.md` (attached state stays minimal). Read on demand — sizing analogs, defect provenance, and the measurements behind M3's standing constraints.

## M3 spine premise — the measurement that scoped the milestone

The producer's window metrics were gap-corrupt before M3.1, so aggregating them aggregated gap artefacts. `.scratch/gap_bias_probe.R` (synthetic min-jerk reach, fs=30, T=3 s, amp=0.40 m; mirrored `analysis/clinical_features.R:681-689` verbatim) measured two distinct failure modes:

- **Bridging** — `normalized_jerk` (`:261-294`) and `movement_efficiency` (`:289-294`) `ok`-filtered NAs, then differentiated at fixed `dt = 1/fs` with `T_dur` from survivor count. NJ 16.456 → **2456 on one dropped frame (+14 826 %)**, +42 217 % at 3, +96 208 % at 8, +138 487 % at 15 (0.50 s), +23 060 % scattered. `movement_efficiency` read 0 % on that chord-dominated case — mechanism real, magnitude unmeasured.
- **Dropping** — `speed`/SAL/`v_mean`/`v_peak` (`:681-689`, `:225-246`) voided NA intervals and under-counted duration. SAL +1.0…+3.7 %, `v_mean` −1.6…−13.4 %, `v_peak` −0.6 % — small enough that tolerance smoke tests miss them, so they need exact-expected-value oracles.

The 3D quality gate creates NA holes by design, so 3D bites hardest. The fix necessarily moved 2D gapped values too, which is why gap-free goldens must stay byte-identical. Probe ported into `tests/test_r_trajectory_kernel.py::test_gap_bias_probe_corpus_is_bounded_and_gapfree_exact` (M3.1) and deleted.

## M3.1 — R-gate closure + timestamp-aware trajectory kernel

Gate 541 passed / 0 skipped (469 baseline + 24 goldens + 48 kernel); `renv::status()` consistent. Gap-free metrics bit-identical to the pre-change implementation under all three timestamp representations. NJ at 1/3/8/15/scattered-15 dropped frames +0.8…+22.9 % vs gap-free, replacing +14 712…+143 346 %.

`nominal_fs()` ships unused by the call sites — adopting it moves every shipped metric value, so it is `.agent/polish.md` `spine?` work needing its own unit and a decision on re-deriving `output/rtmw-l_body_single/`.

## M3.2 — producer identity schema, 3D-only

Gate 559 passed / 0 skipped (541 baseline + 18 items from a diff-blind suite).

Nine character tags — `artifact_kind`, `source_sha256`, `coord_space`, `distance_unit`, three independent versions, `metric_qualification`, `provenance_class` — appended last on the three 3D artifacts only. Artifact identity = (`source_sha256`, `artifact_kind`); capture identity stays `video` under a singleton + non-blank fail-closed check. Row-tag columns beat a sidecar on measured fault detectability (5/6 vs 4/6: a sidecar sees neither a blank tag cell nor an absent tag column), so identity has exactly one channel and cannot self-contradict. 3D artifacts are always written, typed-empty when nothing qualifies, which also clears stale files from an earlier run; 2D keeps skip-if-empty. **The six 2D goldens returned byte-identical from a full regeneration run**, so the partition is proven by rerun rather than by exclusion. Tags are character because one numeric tag becomes five feature columns in `aggregate_per_video()`, and version values are `v<n>` because a bare `1` is guessed `double` by a default `read_csv`.

Residual, accepted: a zero-row artifact carries no tag values (structural — a dummy row would contaminate the aggregate grain), so identity binds by stem to the frame artifact, which is non-empty whenever the input has a row; an artifact set whose frame output is also empty is M3.4's to reject.

## M3.3a — QC evidence artifact, trajectory groups

Gate 621 passed / 0 skipped (559 baseline + 53 QC suite + 3 golden cases + 5 kernel + 1 validation).

The artifact ships end to end on the four trajectory groups — 12 metric rows per window, 22 fields + the nine tags, sorted by `video`/`person_idx`/`window_start_sec`/canonical metric order under radix collation. **P08 is proven by regeneration: the only cells that moved in the two existing 3D artifacts are `producer_version` and `qc_policy_version`; the six 2D goldens are byte-identical.** `WINDOW_SIDE_METRICS` now derives from `WINDOW_SIDE_METRIC_SOURCES`, so a side metric cannot ship without declaring the trajectory it reads. `trajectory_grid_status()` states the grid preconditions once: `trajectory_grid()` raises on a fault, the window pass records `invalid_timebase` and keeps running, so a malformed clip no longer aborts the producer.

**Two measured rulings.** R-12's gap tolerance is load-bearing — at 100 Hz a ten-frame hole divides out to `0.10000000000000009`, which a bare `<=` rejects; `test_gap_threshold_tolerance_is_load_bearing` pins it. R-12's coverage tolerance is producer-unreachable, not inert — coverage is `k/n` over nominal slots, and the widest ratio below `0.80` that the band still admits needs `n >= 2.5e8` (96 days at 30 Hz), so no clip can distinguish `>= min * (1 - 1e-9)` from `>=`. The band is ordinary arithmetic at the policy boundary, where `test_coverage_tolerance_is_pinned_at_the_policy_boundary` calls `qc_reason_for()` directly and kills the mutant that drops it.

**Two review defects closed, both P03/V21-V24 violations of the shipped contract.** Edge-slot denominator: `trajectory_grid()` anchors on the first observed sample (`raw <- (t - t[1]) * fs`), so a row absent at a window edge fell outside the grid and a window that lost its leading frames reported `frame_coverage = 1.0`. Interior loss was always exact, which is why the corpus missed it — every fixture deletes interior rows only. Fixed by padding the validity mask to the window's nominal slot count before `grid_evidence()`; the estimate grid stays as it was, since widening it moves `nj` through `T_dur` and breaks P08. Evidence now describes the window, estimates the observed span inside it. Dropped clip: `median(diff(ts)) <= 0` skipped a descending clip before any window was keyed → `median(abs(diff(ts)))`, so the window is keyed and every metric reports `invalid_timebase`. Both reported by `rev-m3u3a-1`, both reproduced by MAIN, red pre-fix and green post-fix in `test_window_edge_absence_keeps_the_nominal_slot_count` (4 cases) and `test_fully_reversed_timebase_keeps_the_window`.

**Gate hardening from `rev2-m3u3a-1`'s mutation campaign**, six measured survivors each closed by a test: precedence-constant order (`invalid_timebase` returns before the constant is consulted, so a swap was invisible), the unguarded QC writer, default versus radix sort, a dropped `metric_rank` key, `0/0` reaching `NaN` where `NA` is required, and the legacy `_aggregate_clinical` blacklist. Its six determinism sweeps passed: exact numeric lexemes, byte equality across `C`/`C.utf8`/`en_US.utf8`, second-write equality, copied-path hash stability. **Two gates were vacuous and are now direct**: the no-2D-QC assertion read a golden directory that only ever receives a filename whitelist copied out of deleted staging, so it could not observe an unexpected artifact — it now runs the producer in place and lists what was written, with a positive control; and the coverage tolerance is now probed at `qc_reason_for()` rather than through an input that cannot reach it. Each new gate was mutation-checked by MAIN: dropping the tolerance, ungating the writer, and switching the writer to append all turn their gate red.

## M3.3 carrier decision

Both spikes measured 7/8 fault detectability. Wide costs 55 → 125 columns and still cannot express per-metric status, so long wins on the roadmap requirement: **long companion artifact `<stem>_clinical_3d_window_qc.csv`, keyed `video × person_idx × window_start_sec × window_end_sec × metric_id`**, 3D-only, estimates never duplicated. Full decision record, rulings R-1…R-13, predicates P01…P15, verdicts V01…V30, gate identity, and corpus P1…P15 live in `.agent/archive/contract-m3u3.md`.

M3.3 was planned as one unit and did not fit one MAIN window, confirming planrev `F01`. The split boundary is the evidence-group set, not implementation-versus-tests: each half ships a complete gated artifact slice with its own red cases, so TDD holds within each unit.

## M3 descope — what was cut, and what the cut changed

M3 ended after M3.3a. `videos/3-cam/` replaced synthetic fixtures as the development surface, and no remaining M3 unit is forced by that data, so the rest was cut rather than carried as stale plan.

**Retained in the tree:** M3.1 kernel, M3.2 identity schema, M3.3a `<stem>_clinical_3d_window_qc.csv` over the four trajectory groups. Every gate they shipped still runs.

**Cut:**

- **M3.3b** — QC evidence for `bilateral_wrist`, `bilateral_fingertip`, `trunk`, `shoulders`, `cpi`. Consequence: the QC artifact explains the 12 trajectory metrics and stays silent on the derived and body metrics. `docs/technical/analysis.md` *Current scope* now states that boundary as standing rather than pending. `n_required_keypoints_present` keeps M3.3a's `n_valid_frames > 0` reading — every emitted group requires exactly one keypoint, so no shipped group can contradict it, and the semantics question the split deferred is moot until a multi-keypoint group ships.
- **M3.4** declarative metric registry + fail-closed central reader; **M3.5** video-level reducer; **M3.6** aggregate CLI + `analysis_summary.Rmd` 3D inventory section. `clinical_3d_video_aggregate.csv` was never built, and no committed file names it, so the cut falsifies no shipped claim.

**Cut consequences that became work elsewhere.** M3's acceptance required 2D goldens byte-identical, which is the sole reason `nominal_fs()` shipped unadopted in M3.1. That bar died with the milestone, and re-deriving `output/rtmw-l_body_single/` stopped mattering when `videos/initial/` was retired, so cadence-truth adoption became an M2 unit. Two `.agent/polish.md` rows lost their M3.4 dependency and now own the reader they were going to borrow.

**Unmerged branches, retained on disk, unscheduled:** `wt/test-m3u3` (`7026e2c`, `tests/test_r_qc_evidence.py`, diff-blind, red against `cc8a939`; its remaining cases are exactly the five cut groups), `wt/spike-m3u3-wide` (`d4ad6c3`), `wt/spike-m3u3-long` (`1097a4b`). Reviving M3.3b starts from `wt/test-m3u3` against the frozen contract in `.agent/archive/contract-m3u3.md`.

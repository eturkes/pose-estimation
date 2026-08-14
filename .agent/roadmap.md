# Roadmap

Live long-horizon state only; completed trajectory belongs in git.

## M2 — real-world 3D clinical validation

**Status: PARKED** — precondition: user-granted clearance over a synchronized three-camera session with calibration (step 3 additionally needs an external clinical baseline). Dispatch skips M2 until that clearance lands and the marker is cleared; the unpark check below identifies candidate sessions but grants nothing.

**Goal:** validate the full real-data chain (calibration → 2D tracking → 3D fusion → clinical metrics), quantify reprojection/drop/confidence/timing/stability behavior, and replace provisional thresholds with evidence-backed values.

**Current evidence boundary:** calibration, fusion, `world3d.csv`, clinical analysis, QA grading, and injected failure modes pass synthetic tests. No claim of real-capture or clinical-metric accuracy is warranted yet.

**Unpark check:** run `UV_PROJECT_ENVIRONMENT=.venv uv run pose-estimation-run --list-sessions`. A redacted `3 cameras; calibration: present` result identifies a shape-compatible candidate only. Media decoding, synchronization checks, QA, and calibration-value inspection require explicit patient-data clearance.

**Sequence once unparked:**

1. First cleared capture dry run → execute QA + end-to-end validation; record failure modes; recalibrate capture procedure and provisional thresholds.
2. Multiple-session study → quantify reprojection, tracking gaps, throughput, temporal stability, and inter-trial repeatability.
3. Agreement study → use a known-geometry/goniometer/reference-system baseline when available; otherwise retain the validity gap and specify the cheapest sufficient baseline protocol.
4. Derive only de-identified or synthetic regression fixtures; lock observed failures with tests.

**Acceptance:** reproducible commands + reports trace every claim to cleared inputs; thresholds have explicit evidence; clinical-validity gaps remain visible; all repository validation gates pass.

## M3 — analysis-ready 3D aggregation

**Status: IN-PROGRESS** — M3.1-M3.3a DONE, M3.3b next under the same frozen contract. No precondition: develops + gates entirely on synthetic fixtures, no patient-data clearance needed.

**Goal:** carry trusted metric-3D producer output to one analysis-ready per-video aggregate that the repo's own analysis layer surfaces, with 2D/3D pooling unrepresentable by construction.

**Consumable artifact:** `<input-dir>/clinical_3d_video_aggregate.csv` — grain `video × person_idx × limb × source_level × metric_id × statistic`; carries `coord_space`, `distance_unit`, `unit`, `normalizer_id`, capture/artifact identity, producer + metric-method + QC-policy versions, `estimate`, attempted/valid/failed counts, coverage, longest gap, `qc_status`/`qc_reason`. Surfaced by `analysis/analysis_summary.Rmd`.

**Spine premise — measured, not assumed.** The producer's window metrics are gap-corrupt, so aggregating them today aggregates gap artefacts. `.scratch/gap_bias_probe.R` (synthetic min-jerk reach, fs=30, T=3 s, amp=0.40 m; mirrors `analysis/clinical_features.R:681-689` verbatim) measures two distinct failure modes:

- **Bridging** — `normalized_jerk` (`:261-294`) and `movement_efficiency` (`:289-294`) `ok`-filter NAs, then differentiate at fixed `dt = 1/fs` with `T_dur` from survivor count. NJ 16.456 → **2456 on one dropped frame (+14 826 %)**, +42 217 % at 3, +96 208 % at 8, +138 487 % at 15 (0.50 s), +23 060 % scattered. `movement_efficiency` reads 0 % on this chord-dominated case — mechanism real, magnitude **unmeasured**.
- **Dropping** — `speed`/SAL/`v_mean`/`v_peak` (`:681-689`, `:225-246`) void NA intervals and under-count duration. SAL +1.0…+3.7 %, `v_mean` −1.6…−13.4 %, `v_peak` −0.6 %. Small enough that tolerance smoke tests miss them; they need exact-expected-value oracles.

The 3D quality gate creates NA holes by design, so 3D is where this bites hardest. The fix necessarily moves 2D gapped values too — gap-free goldens must stay byte-identical.

**Units.** All `kernel`; each closes with a scoped commit and its own primary-tree gate run.

| id | unit | spine result |
| -- | ---- | ------------ |
| M3.1 | R-gate closure + timestamp-aware trajectory kernel | Drop `zoo`; freeze gap-free goldens; actual-interval kernel over frame/window scope with exact NJ/SAL/velocity/efficiency/dropout semantics. |
| M3.2 | Producer identity schema, 3D-only | Capture/artifact identity + coord/unit/method/QC-version tags on 3D outputs; typed empty outputs; **2D schemas unchanged**; phase outputs explicitly unqualified. |
| M3.3a | QC evidence artifact, trajectory groups | The `_clinical_3d_window_qc.csv` artifact end to end — schema, tags, typed empty, policy thresholds, status/reason machinery, registry, writer, goldens, docs — proven on the four trajectory groups (`{left,right}_{wrist,fingertip}`). |
| M3.3b | QC evidence, derived + body groups | Extend the shipped artifact to `bilateral_wrist`, `bilateral_fingertip`, `trunk`, `shoulders`, `cpi`; complete the corpus and adversarial review. |
| M3.4 | Metric registry + fail-closed reader | Declarative header→metric/limb/unit/normalizer registry; central `utils.R` reader rejecting missing/blank/mixed/incompatible tags, duplicate artifact identity, QC contradictions. |
| M3.5 | Video-level reducer | Long-form reduction, immutable base key, attempted/all-failed strata, exact reducers → `clinical_3d_video_aggregate.csv`. |
| M3.6 | Aggregate CLI + consumability path | Exact discovery, atomic idempotent output; `analysis_summary.Rmd` 3D inventory/QC section with `m` / `m/s` / `deg` / `1` labels; `docs/technical/analysis.md` current. |

**Unit status.** M3.1 DONE (`0fa2079` kernel + R-gate closure, `c93382d` goldens, red suite). M3.2 DONE. M3.3a DONE (`45cd690` kernel prep, emission + suite + goldens + docs this session). **M3.3b OPEN** — the shipped artifact widens to `bilateral_wrist`, `bilateral_fingertip`, `trunk`, `shoulders`, `cpi`. M3.4-M3.6 OPEN.

**Split rationale.** M3.3 was one unit and did not fit one MAIN window, confirming planrev `F01`. The boundary is the evidence-group set, not implementation-versus-tests: each half ships a complete gated artifact slice with its own red cases, so TDD holds within each unit. M3.3a establishes the artifact contract end to end on the simplest complete slice; M3.3b widens coverage to the derived and body groups, where `bilateral_*` status is a function of its two contributing side groups and `trunk`/`shoulders`/`cpi` need per-frame validity rather than trajectory evidence. Both `kernel`. The delivered suite on `wt/test-m3u3` splits the same way — trajectory cases to M3.3a, the rest to M3.3b.

**M3.3 state.** Acceptance contract = `.scratch/agents/contract-m3u3.md`, frozen: decision record D, rulings R-1…R-13, predicates P01…P15, verdicts V01…V30, gate identity, corpus P1…P15. Carrier ruled = **long companion artifact `<stem>_clinical_3d_window_qc.csv`, keyed `video × person_idx × window_start_sec × window_end_sec × metric_id`**, 3D-only, estimates never duplicated. Both spikes measured 7/8 fault detectability; wide costs 55 → 125 columns and still cannot express per-metric status, so long wins on the roadmap requirement.

M3.3a close: gate 607 passed / 0 skipped (559 baseline + 45 QC suite + 3 golden cases); `main=` 91% 219K/240K, `mate=` 51% 122K/240K. The artifact ships end to end on the four trajectory groups — 12 metric rows per window, 22 fields + the nine tags, sorted by `video`/`person_idx`/`window_start_sec`/canonical metric order under radix collation. **P08 is proven by regeneration: the only cells that moved in the two existing 3D artifacts are `producer_version` and `qc_policy_version`; the six 2D goldens are byte-identical.** `WINDOW_SIDE_METRICS` now derives from `WINDOW_SIDE_METRIC_SOURCES`, so a side metric cannot ship without declaring the trajectory it reads. `trajectory_grid_status()` states the grid preconditions once: `trajectory_grid()` raises on a fault, the window pass records `invalid_timebase` and keeps running, so a malformed clip no longer aborts the producer.

Two measured rulings. **R-12's gap tolerance is load-bearing** — at 100 Hz a ten-frame hole divides out to `0.10000000000000009`, which a bare `<=` rejects; `test_gap_threshold_tolerance_is_load_bearing` pins it. **R-12's coverage tolerance is unreachable** — any rational exactly 4/5 rounds to the same double as the `0.80` literal, so no input can distinguish `>= min * (1 - 1e-9)` from `>=`. It stays as documented defensive symmetry, and a mutant removing it survives by construction rather than through a gate hole.

Open for M3.3b: the five derived and body groups, whose evidence is not one kernel call each — `bilateral_*` status is a function of its two contributing side groups, `trunk`/`shoulders`/`cpi` need per-frame validity masks, and `cpi` carries the V05 alternation asymmetry. The delivered suite's remaining cases (`compensatory_pattern_index`, `trunk_*`, `posture_symmetry_*`, the 18 bilateral ids) are the red set.

Unmerged deliverables live on branches, not in the tree: `wt/test-m3u3` carries `tests/test_r_qc_evidence.py` (diff-blind, red against `cc8a939`); `wt/spike-m3u3-wide` and `wt/spike-m3u3-long` carry the two measured probe implementations. Reports: `.scratch/agents/{map,res,test,spike-m3u3-wide,spike-m3u3-long}-m3u3.md`.

M3.2 close: gate 559 passed / 0 skipped (541 baseline + 18 items from a diff-blind suite); `main=` 95% 227K/240K, `mate=` 93% 222K/240K. Nine character tags — `artifact_kind`, `source_sha256`, `coord_space`, `distance_unit`, three independent versions, `metric_qualification`, `provenance_class` — appended last on the three 3D artifacts only; artifact identity = (`source_sha256`, `artifact_kind`), capture identity stays `video` under a singleton + non-blank fail-closed check. Row-tag columns beat a sidecar on measured fault detectability (5/6 vs 4/6: a sidecar sees neither a blank tag cell nor an absent tag column), so identity has exactly one channel and cannot self-contradict. 3D artifacts are always written, typed-empty when nothing qualifies, which also clears stale files from an earlier run; 2D keeps skip-if-empty. **The six 2D goldens returned byte-identical from a full regeneration run**, so the partition is proven by rerun rather than by exclusion. Tags are character because one numeric tag becomes five feature columns in `aggregate_per_video()`, and version values are `v<n>` because a bare `1` is guessed `double` by a default `read_csv`.

Residual, accepted: a zero-row artifact carries no tag values (structural — a dummy row would contaminate the aggregate grain), so identity binds by stem to the frame artifact, which is non-empty whenever the input has a row; an artifact set whose frame output is also empty is M3.4's to reject. Movement-phase artifacts still have no byte golden in either mode → `.agent/polish.md`.

M3.1 close: gate 541 passed / 0 skipped (469 baseline + 24 goldens + 48 kernel); `renv::status()` consistent; gap-free metrics bit-identical to the pre-change implementation under all three timestamp representations; NJ at 1/3/8/15/scattered-15 dropped frames +0.8…+22.9 % vs gap-free, replacing +14 712…+143 346 %. `nominal_fs()` ships unused by the call sites — adopting it moves every shipped metric value, so it is `.agent/polish.md` `spine?` work needing its own unit and a decision on re-deriving `output/rtmw-l_body_single/`.

**Standing constraints.**

- **No 2D schema widening.** `analysis/utils.R:59-87` `aggregate_per_video()` treats every numeric non-metadata column as a feature, and the R gate invokes no downstream consumer — a QC column added to 2D outputs would silently enter PCA, correlations and z-scores unnoticed. QC/identity evidence is 3D-only; 2D outputs stay byte-identical under golden gates.
- **No session or trial claim.** Producer keys only `video/person/window`; no task/condition/trial identity exists. The artifact is a *video* aggregate; windows are never called trials, and `res-m3-1`'s trial→session median/IQR hierarchy stays unreachable until a real protocol schema lands.
- **Provisional QC thresholds.** `coverage ≥ 0.80`, `max_gap ≤ 0.10 s` are conservative engineering defaults, not validated standards — labelled provisional everywhere and carried under `qc_policy_version`; calibration belongs to M2.
- **Unknown provenance fails closed** to within-file aggregation rather than pooling across unverified rig/model/filter identity.
- **Decisive gate is primary-tree.** `renv/library/` is gitignored, so worktrees skip R cases; a green worktree run is no evidence for `analysis/*.R`.

**Acceptance:** every gap-sensitive quantity *admitted to the frame/window aggregate* is timestamp-aware; a one-frame interior-drop regression over an analytic trajectory can never again pass at the observed NJ error; 2D goldens byte-identical; the aggregate has a unique composite key and no representable 2D/3D pooling; `pytest tests/test_r_pipeline.py` and the full suite pass in the primary tree with 0 skips; every external R import resolves to a committed `renv.lock` entry.

**Known consequence:** `output/rtmw-l_body_single/` clinical features predate the gap fix — any normalized-jerk or velocity figure derived from them is suspect and needs recomputation after M3.1.

## Produced datasets

- `output/rtmw-l_body_single/` — all 12 single-camera clips, RTMW-L / `--tracking body` / `--single-subject`, det-CPU + pose-NPU. 15 430 rows over 15 455 decoded frames, 99.7% mean coverage, 304-col schema conformant on every file, 100% body-wrist observation. `manifest.json` (per-video provenance + SHA-256) and `qa_report.md` sit beside the CSVs; regenerate both with `scripts/run_report.py`. Destined for `Projects/rehab/`, which has no pose-ingest contract yet — its schema is tabular ISNCSCI/SCIM, so the join surface still has to be designed.

## Backlog

Scope seed for the milestone after M3.

- Clinical join surface — capture/session metadata mapping, then a capture→assessment bridge emitting joined observations with instrument/version/domain/side/status and cardinality safety. Cut from M3 because it needs a real instrument schema: `Projects/rehab/` holds tabular ISNCSCI/SCIM and has no pose-ingest contract, and ISNCSCI is side/myotome-resolved while SCIM is whole-person, so the join grain cannot be settled against synthetic data alone.
- Cross-camera identity matching for multi-person scenes; fusion currently assumes one subject.
- Gap-aware movement-phase metrics — M3.1's kernel covers frame/window scope only; `analysis/clinical_features.R:918-1097` phase speed/path/SAL/NJ/efficiency stay gap-unsafe and are explicitly unqualified by M3.

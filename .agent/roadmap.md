# Roadmap

Live long-horizon state only; completed trajectory belongs in git. Closed-unit detail, frozen contracts → `.agent/archive/`, read on demand.

## M2 — real-world 3D clinical validation

**Status: PARKED** — precondition: user-granted clearance over a synchronized three-camera session with calibration (step 3 additionally needs an external clinical baseline). Dispatch skips M2 until that clearance lands and the marker is cleared; the footage inventory below identifies candidates and grants nothing.

**Goal:** validate the full real-data chain (calibration → 2D tracking → 3D fusion → clinical metrics), quantify reprojection/drop/confidence/timing/stability behavior, and replace provisional thresholds with evidence-backed values.

**Current evidence boundary:** calibration, fusion, `world3d.csv`, clinical analysis, QA grading, and injected failure modes pass synthetic tests. No claim of real-capture or clinical-metric accuracy is warranted yet.

**Footage inventory — filename-derived, no media decoded, no identifiers in the names.** `videos/` now splits into `videos/initial/` (the 12 single-camera clips behind `output/rtmw-l_body_single/`) and `videos/3-cam/` (the awaited three-view footage): 16 ordinal subject directories, 382 `.MOV`/`.mov` files, 18.3 GB, flat inside each directory.

- Stems encode `<n>_<view>_<task>_<side>`; `view ∈ {above,left,right}` = the three cameras, `task ∈ {cap,coin,glass,key,nut,peg}`, `side ∈ {L,R}`. Trial grain = (subject, task, side); each view is a sibling file, not a per-session directory.
- **Coverage is not uniformly three views.** 188 trials after normalising the stem typos: 52 have all three views (1-9 per subject, every subject represented), 85 have two, 51 have one; 2 trials repeat a view; 3 stems do not parse.
- Stem noise to absorb at ingest: `grass`→`glass` (×9), `gcap`, `gpeg`, `coini`, an embedded space (`above_ nut_R`), a trailing space, a ` (2)` duplicate marker, a truncated `right_cap_`, a doubled `.MOV.MOV`, and mixed-case extensions.
- **No `session.json` and no `calibration.json` anywhere in the tree**, so the calibration half of the precondition is unmet by inspection, and metric 3D is unreachable until a rig calibration exists or is recovered from the footage.
- **The repo's session contract does not match this layout.** `discover_session()` takes `session.json` or a `cam*.{mp4,avi,mov,mkv,webm}` glob, so `pose-estimation-run --list-sessions --sessions-dir videos/3-cam` returns `ERROR: no sessions discovered` (rc 1), as does the default `videos/` root. An ingest/normalisation step that maps trials onto session directories is prerequisite work, not a config change.
- Unverified: synchronization, frame rate, resolution, per-view time alignment, and whether the view labels match physical camera placement. All need decoding, therefore clearance.

**Unpark check:** `UV_PROJECT_ENVIRONMENT=.venv uv run pose-estimation-run --list-sessions --sessions-dir <root>`. A redacted `3 cameras; calibration: present` result identifies a shape-compatible candidate only. Media decoding, synchronization checks, QA, and calibration-value inspection require explicit patient-data clearance.

**Sequence once unparked:**

1. Ingest layer → normalise `videos/3-cam/` stems onto the session contract, then dry-run one cleared capture: QA + end-to-end validation; record failure modes; recalibrate capture procedure and provisional thresholds.
2. Multiple-session study → quantify reprojection, tracking gaps, throughput, temporal stability, and inter-trial repeatability.
3. Agreement study → use a known-geometry/goniometer/reference-system baseline when available; otherwise retain the validity gap and specify the cheapest sufficient baseline protocol.
4. Derive only de-identified or synthetic regression fixtures; lock observed failures with tests.

**Acceptance:** reproducible commands + reports trace every claim to cleared inputs; thresholds have explicit evidence; clinical-validity gaps remain visible; all repository validation gates pass.

## M3 — analysis-ready 3D aggregation

**Status: IN-PROGRESS** — M3.1-M3.3a DONE, M3.3b next under the same frozen contract. No precondition: develops + gates entirely on synthetic fixtures, no patient-data clearance needed.

**Goal:** carry trusted metric-3D producer output to one analysis-ready per-video aggregate that the repo's own analysis layer surfaces, with 2D/3D pooling unrepresentable by construction.

**Consumable artifact:** `<input-dir>/clinical_3d_video_aggregate.csv` — grain `video × person_idx × limb × source_level × metric_id × statistic`; carries `coord_space`, `distance_unit`, `unit`, `normalizer_id`, capture/artifact identity, producer + metric-method + QC-policy versions, `estimate`, attempted/valid/failed counts, coverage, longest gap, `qc_status`/`qc_reason`. Surfaced by `analysis/analysis_summary.Rmd`.

**Spine premise — measured, not assumed.** Pre-M3.1 window metrics were gap-corrupt: bridging drove `normalized_jerk` +14 826 % on one dropped frame, dropping moved SAL/`v_mean` by −13.4…+3.7 % — too small for tolerance smoke tests, so exact-expected-value oracles are mandatory. The 3D quality gate creates NA holes by design, so 3D bites hardest, and the fix moves 2D gapped values too. Full probe record → `.agent/archive/m3-closed-units.md`.

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

**Unit status + sizing analogs.** M3.4-M3.6 OPEN. **M3.3b OPEN** — the shipped artifact widens to `bilateral_wrist`, `bilateral_fingertip`, `trunk`, `shoulders`, `cpi`.

| unit | close | gate (passed / skipped) | `main=` | `mate=` |
| ---- | ----- | ----------------------- | ------- | ------- |
| M3.1 | DONE | 541 / 0 | — | — |
| M3.2 | DONE | 559 / 0 | 95% 227K/240K | 93% 222K/240K |
| M3.3a | DONE | 621 / 0 | 58% 139K/240K | 100% 240K/240K |

M3.1 shipped `0fa2079` (kernel + R-gate closure) and `c93382d` (goldens, red suite); M3.3a's kernel prep is `45cd690`. Defect provenance, mutation-campaign results, and the measurements behind the standing constraints → `.agent/archive/m3-closed-units.md`.

**M3.3 contract, frozen:** `.agent/archive/contract-m3u3.md` — decision record D, rulings R-1…R-13, predicates P01…P15, verdicts V01…V30, gate identity, corpus P1…P15. Carrier ruled = **long companion artifact `<stem>_clinical_3d_window_qc.csv`, keyed `video × person_idx × window_start_sec × window_end_sec × metric_id`**, 3D-only, estimates never duplicated. The split boundary is the evidence-group set, so TDD holds within each half; the delivered suite on `wt/test-m3u3` splits the same way.

**Open for M3.3b.** The five derived and body groups are not one kernel call each: `bilateral_*` status is a function of its two contributing side groups, `trunk`/`shoulders`/`cpi` need per-frame validity masks, and `cpi` carries the V05 alternation asymmetry. The delivered suite's remaining cases (`compensatory_pattern_index`, `trunk_*`, `posture_symmetry_*`, the 18 bilateral ids) are the red set. `n_required_keypoints_present` needs its exact semantics ruled there: every trajectory group requires one keypoint, so M3.3a reads it as `n_valid_frames > 0`, and the first group requiring two or more decides whether it counts distinct keypoints, frames with all of them, or something else. It is gated as an evidence field (`_EVIDENCE_FIELDS`), so a widening that changes it must move a test.

Unmerged deliverables live on branches, not in the tree: `wt/test-m3u3` carries `tests/test_r_qc_evidence.py` (diff-blind, red against `cc8a939`); `wt/spike-m3u3-wide` and `wt/spike-m3u3-long` carry the two measured probe implementations.

**Standing constraints.**

- **No 2D schema widening.** `analysis/utils.R:59-87` `aggregate_per_video()` treats every numeric non-metadata column as a feature, and the R gate invokes no downstream consumer — a QC column added to 2D outputs would silently enter PCA, correlations and z-scores unnoticed. QC/identity evidence is 3D-only; 2D outputs stay byte-identical under golden gates.
- **No session or trial claim.** Producer keys only `video/person/window`; no task/condition/trial identity exists. The artifact is a *video* aggregate; windows are never called trials, and `res-m3-1`'s trial→session median/IQR hierarchy stays unreachable until a real protocol schema lands.
- **Provisional QC thresholds.** `coverage ≥ 0.80`, `max_gap ≤ 0.10 s` are conservative engineering defaults, not validated standards — labelled provisional everywhere and carried under `qc_policy_version`; calibration belongs to M2.
- **Unknown provenance fails closed** to within-file aggregation rather than pooling across unverified rig/model/filter identity.
- **Decisive gate is primary-tree.** `renv/library/` is gitignored, so worktrees skip R cases; a green worktree run is no evidence for `analysis/*.R`.

**Acceptance:** every gap-sensitive quantity *admitted to the frame/window aggregate* is timestamp-aware; a one-frame interior-drop regression over an analytic trajectory can never again pass at the observed NJ error; 2D goldens byte-identical; the aggregate has a unique composite key and no representable 2D/3D pooling; `pytest tests/test_r_pipeline.py` and the full suite pass in the primary tree with 0 skips; every external R import resolves to a committed `renv.lock` entry.

**Known consequence:** `output/rtmw-l_body_single/` clinical features predate the gap fix — any normalized-jerk or velocity figure derived from them is suspect and needs recomputation after M3.1.

## Produced datasets

- `output/rtmw-l_body_single/` — all 12 single-camera clips, RTMW-L / `--tracking body` / `--single-subject`, det-CPU + pose-NPU. 15 430 rows over 15 455 decoded frames, 99.7% mean coverage, 304-col schema conformant on every file, 100% body-wrist observation. `manifest.json` (per-video provenance + SHA-256) and `qa_report.md` sit beside the CSVs; regenerate both with `scripts/run_report.py <csv_dir> --videos-dir videos/initial` — the sources moved one level down, and the script reads its `--videos-dir` non-recursively. Destined for `Projects/rehab/`, which has no pose-ingest contract yet — its schema is tabular ISNCSCI/SCIM, so the join surface still has to be designed.

## Backlog

Scope seed for the milestone after M3.

- Clinical join surface — capture/session metadata mapping, then a capture→assessment bridge emitting joined observations with instrument/version/domain/side/status and cardinality safety. Cut from M3 because it needs a real instrument schema: `Projects/rehab/` holds tabular ISNCSCI/SCIM and has no pose-ingest contract, and ISNCSCI is side/myotome-resolved while SCIM is whole-person, so the join grain cannot be settled against synthetic data alone. `videos/3-cam/`'s `<task>_<side>` stems are the first real trial-identity signal the project has seen — they arrive as filenames, not as a protocol schema, so they inform the design without settling it.
- Cross-camera identity matching for multi-person scenes; fusion currently assumes one subject.
- Gap-aware movement-phase metrics — M3.1's kernel covers frame/window scope only; `analysis/clinical_features.R:918-1097` phase speed/path/SAL/NJ/efficiency stay gap-unsafe and are explicitly unqualified by M3.

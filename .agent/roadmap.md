# Roadmap

Live long-horizon state only; completed trajectory belongs in git. Closed-unit detail, frozen contracts → `.agent/archive/`, read on demand.

**Repo scope = `videos/3-cam/`.** Sibling directories under the same data root are out of scope: `harness/` holds schematics for a capture harness that was never built, `database/` holds the hospital's SCI clinical records. `videos/initial/` is preliminary data, retired from active work.

## M2 — three-camera corpus: inventory, qualification, 3D ruling

**Status: IN-PROGRESS** — M2.1 OPEN. The old clearance precondition is met: full decode clearance covers the whole `videos/3-cam/` tree, for MAIN and teammates. Chat and reports carry redacted aggregates only — never imagery, filenames, or subject identifiers.

**Goal:** turn 382 uncontrolled clips into an addressable, measured corpus; establish by evidence whether 3D reconstruction is recoverable from it; then execute that ruling under a claim boundary the data can carry.

**Corpus, measured.** Header-only census over all 382 files (cv2 container properties, no pixel decode; `.scratch/census_3cam.py` + `.scratch/census_rig.py`, 4.5 s, ported by M2.1):

- 382 files, 339 743 frames, 186.8 min, 16 subject directories, all readable. Trial duration median 19.3 s (p25 13.9, p75 33.1, max 274.8).
- Stems `<n>_<view>_<task>_<side>` → 188 trials, 94 per side, tasks 30-32 each. View coverage **49 three-view / 87 two-view / 52 one-view**, so 136 trials are multi-view; 7 stems fail this normalizer. A plan assuming 188 three-camera sessions overstates by 3.8×.
- **Two normalizers disagree on the split** — 49/87/52 here against 52/85/51 from an independent pass, on identical files. Both agree on 188 trials and 16 subjects, so the disagreement is entirely in typo and repeat-marker handling. M2.1 settles the grammar mechanically and owns the true numbers; treat both splits as provisional.
- Resolution 1920×1080, or 1080×1920 for the 28 portrait clips. **38 clips carry non-zero rotation metadata** — 28 at 90/270°, 10 at 180°. Codecs h264 + hevc, and **all 16 subjects mix both** across their own clips.
- **Nominal 30 fps, but every file differs**: 29.963-29.987 Hz, plus one 119.97 fps 720p outlier. Within-trial fps agreement to 3 dp: **7 of 136** multi-view trials.
- **The views of one trial are not one recording.** Frame-count parity within 5%: 41 of 136; within 20%: 73 of 136. Within-trial duration spread median **3.99 s**, p75 13.2 s, p95 25.7 s, max 210 s.
- **Orientation varies inside a single view label**: `above` = 124 rot0 / 15 rot90 / 10 rot180 / 1 rot270; 5 of 16 subjects show more than one frame orientation for the same label.

**No fixed rig ever existed** — the camera harness was designed and never built, which is what the orientation, codec, parity and duration spread independently measure. Three cameras were started and stopped by hand and re-oriented between takes. Two consequences bind the whole milestone: a single rig calibration reused across trials is incoherent, so calibration is **per trial** at best; and the repo's alignment model, one non-negative integer `sync_offset` per camera, cannot express unequal frame rates, so cross-view drift is unrepresentable today.

**Claim boundary, set by evidence, not by ambition.** Published upper-limb validation puts three RGB cameras at 5-9 cm joint-position error with no clinical angle validation; the credible functional-task protocols use 8-10 synchronized cameras at 60-85 Hz and still report 3-20°+ angle RMSD. One frame at 30 Hz is 33 ms, which is 33 mm of hand travel at 1 m/s, so even exact integer alignment caps timing contribution near ±17 mm. **M2 may claim retrospective 3D recovery feasibility with internal geometric and QC evidence — reprojection, triangulation angle, visibility, offset confidence, scale provenance, sensitivity. It may not claim clinical validity, absolute metric accuracy, or marker-based equivalence.** Crossing that floor needs a prospective calibrated capture, which M2.7 specifies rather than performs.

**Metric scale is a separate, external requirement.** Image geometry has a gauge freedom, so SfM, essential matrices and pose-only calibration recover shape up to one unknown factor. M2.3 surveys the footage for an in-frame reference before anything is requested; a measurable apparatus object is the strongest route, participant anthropometrics next, and a rig survey is dead because the rig never existed. Absent any reference, 3D output stays explicitly arbitrary-scale: angles and dimensionless ratios survive, every metre-valued distance, velocity and jerk does not.

**Units.** Tier `kernel` throughout; each closes with a scoped commit and its own primary-tree gate run.

| id | unit | spine result |
| -- | ---- | ------------ |
| M2.1 | Canonical trial record + corpus census | Stem grammar, normalization, quarantine, deterministic `capture_id`; per-file container facts; one committed inventory tool replacing the two scratch censuses. |
| M2.2 | Session materialization + discovery | Idempotent generator emitting a discoverable session tree; partial-view policy; `--list-sessions` enumerates the real corpus. |
| M2.3 | Capture qualification + 3D-route ruling | Decode-sampled evidence on scale reference, background rigidity, view↔geometry stability, detectability, recoverable offset/drift, intrinsics metadata → MAIN's ruling, which shapes M2.5-M2.7. |
| M2.4 | Timebase truth | Adopt `nominal_fs()` at the call sites; regenerate goldens; per-file cadence replaces the `1/median(diff(ts))` estimate. |
| M2.5 | Cross-view alignment | Offset + drift model beyond integer `sync_offset`, per-trial `sync_qc` evidence. Shape set by M2.3. |
| M2.6 | Calibration recovery | Per-trial extrinsics by the route M2.3 rules, with held-out reprojection acceptance and explicit scale provenance. |
| M2.7 | Gated fusion + corpus study | Fusion over qualified trials, reprojection/gap/throughput/stability/repeatability evidence, claim-bounded report, prospective-capture specification, de-identified regression fixtures. |

**Unit status.** M2.1-M2.4 OPEN. M2.3 is the milestone's decision point; M2.1, M2.2 and M2.4 stand independent of its ruling, so dispatch order is simply lowest-open. M2.5, M2.6, M2.7 **BLOCKED on M2.3's ruling** — their shape, and whether M2.6 exists at all, is what that ruling decides.

| unit | close | gate (passed / skipped) | `main=` | `mate=` |
| ---- | ----- | ----------------------- | ------- | ------- |
| — | baseline at M2 plan | 621 / 0 | — | — |

**Sizing analogs** (unique files touched, summed churn; gauges where recorded). M3.2 `16e6fab` = 9 files, +891/−117, `main=95%` — the schema/identity analog for M2.1. M3.3a `a6218e5` = 13 files, +1694/−152, `main=58%` — a full artifact slice. Multi-camera fusion `62685e0` = 14 files, +1040/−164, and calibration `4d4df80` = 18 files, +1472/−156 — the integration band for M2.5/M2.6. Uncalibrated QA `20c36a0` = 14 files, +1225/−152 and adversarial failure modes `36f28a2` = 11 files, +981/−392 — the analogs for M2.7. **M3.3 was planned as one unit and did not fit one MAIN window**; M2.1/M2.2 are split at the same kind of boundary for the same reason.

**Load-bearing facts for M2.1/M2.2, probe-verified.**

- A session directory of **symlinks resolves correctly under glob discovery and under a manifest that omits `file`**; a manifest naming `file:` explicitly **fails**, because `_safe_resolve()` resolves through the link and the containment check then rejects it (`src/pose_estimation/multicam.py:56-62`). Materialization therefore needs no change to `multicam.py`, and it must not use `file:` refs.
- Glob discovery needs a **lowercase** extension from `VIDEO_EXTENSIONS`; 380 of 382 sources end `.MOV`. Symlink names are ours to choose, so this constrains the generator, not the sources.
- `discover_sessions()` is **one level deep**, so the tree must be flat: one directory per trial, not `subject/trial`.
- `session.json` without `file` refs carries `session_id`, arbitrary camera names, and `sync_offset` — enough for trial identity and alignment, with no schema change.
- cv2 applies rotation metadata on decode by default, and `CAP_PROP_FRAME_WIDTH/HEIGHT` already report the rotated size, so frames arrive upright. Nothing in `src/`, `scripts/` or `tests/` reads `CAP_PROP_ORIENTATION_*`; the hazard is that a rotated view has different image geometry from its siblings, not that frames are sideways.
- The generated tree is patient-adjacent (manifests carry subject ordinal, task, side) → repo-local `sessions/`, added to `.gitignore` by M2.2.
- The environment has **cv2 only** — no ffprobe, ffmpeg, exiftool, PyAV. True PTS, creation timestamps, audio tracks and Apple intrinsics metadata all need tooling M2.3 installs.

**Standing constraints.**

- **Trial identity has no schema home.** Producer keys are `video`/`person_idx`/`window`; `session.json` carries no task, side, or trial field; `world3d.csv` reduces `video` to `session_id`. M2.1's registry is the single source of trial identity, bound by `capture_id`. Legacy 2D and 3D producer schemas stay unwidened — `analysis/utils.R:59-87` treats every numeric non-metadata column as a feature.
- **Calibration identity is unbound.** Discovery accepts any calibration whose camera names match; nothing compares rig or session identity (`src/pose_estimation/multicam.py:364-383,579-588`). Per-trial calibration makes this a live hazard, so M2.6 must bind calibration to `capture_id`.
- **View labels are lexical, not geometric.** `above`/`left`/`right` are filename tokens. M2.3 verifies them against measured geometry before any calibration reuse; a mismatch projects pixels through the wrong camera.
- **Provisional QC thresholds.** `coverage ≥ 0.80`, `max_gap ≤ 0.10 s` are engineering defaults carried under `qc_policy_version`, not validated standards. M2.7 is where evidence replaces them.
- **One subject only.** Fusion reads `person_idx == 0`; cross-camera identity matching does not exist.
- **Decisive gate is primary-tree.** `renv/library/` is gitignored, so worktrees skip R cases unless symlinked; a green worktree run is no evidence for `analysis/*.R`.

**Acceptance:** every one of the 382 files reaches exactly one explicit outcome — canonical trial, quarantined stem, or recorded exclusion — with nothing silently dropped; the session tree regenerates byte-identical from a clean base and `--list-sessions` enumerates it; every corpus claim traces to a committed rerunnable command rather than a scratch script; the claim boundary above is honored in every artifact and document; full suite passes in the primary tree with 0 skips.

## M3 — analysis-ready 3D aggregation

**Status: DESCOPED.** M3.1, M3.2 and M3.3a shipped and stay in the tree with their gates. M3.3b and M3.4-M3.6 are cut: real data replaced the synthetic development surface, and no remaining unit is forced by it. `clinical_3d_video_aggregate.csv` was never built and nothing references it, so the cut falsifies no shipped claim.

What survives in the tree: the timestamp-aware trajectory kernel (`zoo` dropped), the 3D producer identity schema, and `<stem>_clinical_3d_window_qc.csv` over the four trajectory groups. The QC artifact explains 12 trajectory metrics and is silent on `bilateral_*`, `trunk`, `shoulders`, `cpi` — `docs/technical/analysis.md` *Current scope* now states that as standing scope.

The cut released M2.4: M3's "2D goldens byte-identical" acceptance was the only reason `nominal_fs()` shipped unadopted, and re-deriving `output/rtmw-l_body_single/` stopped mattering when `videos/initial/` was retired. Full record, including the retained unmerged branches and the frozen M3.3 contract → `.agent/archive/m3-closed-units.md`, `.agent/archive/contract-m3u3.md`.

## Produced datasets

- `output/rtmw-l_body_single/` — **preliminary**, from the retired `videos/initial/` clips. 12 single-camera clips, RTMW-L / `--tracking body` / `--single-subject`, det-CPU + pose-NPU; 15 430 rows over 15 455 frames, 99.7% mean coverage, 100% body-wrist observation. Kept on disk, not regenerated. Its clinical features predate both the M3.1 gap fix and M2.4's cadence fix, so **any normalized-jerk or velocity figure from it is suspect** — recompute before citing it anywhere, including a paper's preliminary-work section. Schema conformance and coverage figures are unaffected and remain quotable.

## Backlog

Scope seed for the milestone after M2.

- **Clinical join surface** — the eventual destination for M2's numerical output: the hospital SCI database (`database/ALL_SCIDATA.csv` + `SCI_DATABASE_HEADER.xlsx`), currently analyzed as a dashboard in `Projects/rehab/`. **The subject↔patient mapping is unknown and has to be established first**; nothing in `videos/3-cam/` identifies a database record. Needs capture/session metadata, then a capture→assessment bridge with instrument/version/domain/side/status and cardinality safety. ISNCSCI is side/myotome-resolved while SCIM is whole-person, so the join grain cannot be settled against synthetic data.
- Cross-camera identity matching for multi-person scenes; fusion assumes one subject.
- Gap-aware movement-phase metrics — M3.1's kernel covers frame/window scope only; `analysis/clinical_features.R:918-1097` phase speed/path/SAL/NJ/efficiency stay gap-unsafe and explicitly unqualified.
- Prospective calibrated capture — the only route past M2's claim boundary. M2.7 specifies it; running it is a separate milestone with its own clearance and ethics footprint.

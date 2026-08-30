# Roadmap

Live long-horizon state only; completed trajectory belongs in git. Closed-unit detail, frozen contracts → `.agent/archive/`, read on demand.

**Repo scope = `videos/3-cam/`.** Sibling directories under the same data root are out of scope: `harness/` holds schematics for a capture harness that was never built, `database/` holds the hospital's SCI clinical records. `videos/initial/` is preliminary data, retired from active work.

## M2 — three-camera corpus: inventory, qualification, 3D ruling

**Status: IN-PROGRESS** — M2.1, M2.2, M2.3 DONE; M2.4-M2.7 remain. The old clearance precondition is met: full decode clearance covers the whole `videos/3-cam/` tree, for MAIN and teammates. Chat and reports carry redacted aggregates only — never imagery, filenames, or subject identifiers.

**Goal:** turn 382 uncontrolled clips into an addressable, measured corpus; establish by evidence whether 3D reconstruction is recoverable from it; then execute that ruling under a claim boundary the data can carry.

**Corpus, measured.** Header-only census over all 382 files, from the committed tool (`pose-estimation-inventory`; cv2 container properties, no pixel decode; ~11 s including a SHA-256 of every file). Every number below reruns from `inventory/census.json`:

- 382 files, 339 743 frames, 186.8 min, 18.28 GB, 16 subject directories, all readable, all 382 SHA-256 distinct. Per-file nominal duration median 19.3 s (p25 13.9, p75 33.1, p95 62.7, max 274.8).
- Stems `<n>_<view>_<task>_<side>` → **188 task-side families** over 379 canonical files. `<n>` is the **subject** ordinal — 16 values, one per subject directory, bijective — so family identity is `(subject, task, side)` with no take component. View coverage **52 three-view / 85 two-view / 51 one-view**, so 137 families are multi-view. **3 files quarantine** (2 empty side token, 1 unknown side token), 0 excluded. A plan assuming 188 three-camera sessions overstates by 3.6×.
- **The intended design is 16 subjects × 6 tasks × 2 sides = 192 families; 188 exist, so 4 are absent outright** — `s02-cap-r`, `s06-glass-r`, `s07-coin-l`, `s12-glass-l`. Family view-sets: `above|left|right` 52, `above|right` 46, `above` 37, `left|right` 20, `above|left` 19, `right` 13, `left` 1. Canonical assets per view: above 155, left 93, right 131 — **`left` is the camera most often missing**, and every one of the 16 subjects owns at least one three-view family.
- **The 49/87/52 versus 52/85/51 disagreement is settled: doubled media suffixes.** 4 files are named `x.mov.MOV`; stripping one suffix leaves `.mov` occupying the side slot. Typo and repeat-marker handling moved 0 of 188 families. Both earlier splits are dead.
- Name repair is small and now measured: 379 case folds, 15 task-token repairs over 4 misspellings, 4 doubled suffixes, 2 whitespace collapses, 1 repeat marker.
- Resolution 1920×1080, or 1080×1920 for the 28 portrait clips, plus one 1280×720 outlier. **38 clips carry non-zero rotation metadata** — 28 at 90/270°, 10 at 180°. Codecs h264 + hevc, and **all 16 subject directories mix both**.
- **Nominal 30 fps, but every file differs**: 29.963-29.987 Hz, plus one 119.97 fps 720p outlier. Within-family fps agreement to 3 dp: **7 of 137** multi-view families.
- **The views of one family are not one recording.** Frame-count parity within 5%: 40 of 137; within 20%: 74 of 137. Equal resolution across views: 122 of 137. Within-family duration spread median **3.92 s**, p75 13.0 s, p95 25.7 s, max 210.2 s.
- **Orientation varies inside a single view label**: `above` = 129 rot0 / 15 rot90 / 10 rot180 / 1 rot270; `left` = 88/5; `right` = 124/7.
- **2 families carry a view conflict** — two files claiming one view of one family. Both pairs are distinct recordings rather than copies: one is repeat-marked, one merges under underscore collapse. The registry flags them; it does not choose between them.
- **Header facts are the demuxer's claims.** A 10-clip full decode spanning both codecs and all four rotation values matched the header frame count exactly, 7 061 of 7 061 frames, so counts are trustworthy on that sample. The same decode found inter-frame presentation timestamps non-uniform on 10 of 10 clips, so cv2 alone supports no constant-frame-rate claim.

**No fixed rig ever existed** — the camera harness was designed and never built, which is what the orientation, codec, parity and duration spread independently measure. Three cameras were started and stopped by hand and re-oriented between takes. Two consequences bind the whole milestone: a single rig calibration reused across the corpus is incoherent, so calibration is **per recording event** at best — a grain M2.2 still has to resolve, since a `view_conflict` family holds more than one take; and the repo's alignment model, one non-negative integer `sync_offset` per camera, cannot express unequal frame rates, so cross-view drift is unrepresentable today.

**Claim boundary, set by evidence, not by ambition.** Published upper-limb validation puts three RGB cameras at 5-9 cm joint-position error with no clinical angle validation; the credible functional-task protocols use 8-10 synchronized cameras at 60-85 Hz and still report 3-20°+ angle RMSD. One frame at 30 Hz is 33 ms, which is 33 mm of hand travel at 1 m/s, so even exact integer alignment caps timing contribution near ±17 mm. **M2 may claim retrospective 3D recovery feasibility with internal geometric and QC evidence — reprojection, triangulation angle, visibility, offset confidence, scale provenance, sensitivity. It may not claim clinical validity, absolute metric accuracy, or marker-based equivalence.** Crossing that floor needs a prospective calibrated capture, which M2.7 specifies rather than performs.

**Metric scale is unavailable in this corpus — surveyed and ruled (R3, closed negative).** Image geometry has a gauge freedom, so SfM, essential matrices and pose-only calibration recover shape up to one unknown factor. The survey ran over a stratified 52/379 asset sample spanning all 18 task×view cells: the task apparatus is widely visible, yet exact dimensional identity resolved in **0/52**, every fallback (anthropometrics, furniture, rig baseline, calibration target, audio time-of-flight, skeletal priors) is absent rather than imprecise, and the best conditional route floors at **±17.7%** before lens distortion. 3D output is therefore permanently arbitrary-scale for this corpus: angles, angular velocities, timing, normalized trajectory shape and dimensionless ratios survive; every metre-valued distance, velocity and jerk does not. The negative is *sampled*, so the axis publishes `scale_unmeasured` on all 379 rows rather than a measured `none`. Reopening needs acquisition outside this corpus — caliper measurement of the retained apparatus, a filmed ruler, or mapped participant anthropometry — and is a user decision, not a schedulable analysis unit.

**Units.** Tier `kernel` throughout; each closes with a scoped commit and its own primary-tree gate run.

| id | unit | spine result |
| -- | ---- | ------------ |
| M2.1 | Canonical capture record + corpus census | Stem grammar, normalization, quarantine, deterministic `capture_id`; per-file container facts; one committed inventory tool replacing the two scratch censuses. |
| M2.2 | Session materialization + discovery | Idempotent generator emitting a discoverable session tree; partial-view policy; `--list-sessions` enumerates the real corpus. |
| M2.3 | Capture qualification + 3D-route ruling | Decode-sampled evidence on scale reference, background rigidity, view↔geometry stability, detectability, recoverable offset/drift, intrinsics metadata → MAIN's ruling, which shapes M2.5-M2.7. |
| M2.4 | Timebase truth | Adopt `nominal_fs()` at the call sites; regenerate goldens; per-file cadence replaces the `1/median(diff(ts))` estimate. |
| M2.5 | Cross-view alignment | One float `offset_s` per camera against the event reference, no rate term (M2.3 R5); per-recording-event `sync_qc` evidence. |
| M2.6 | Calibration recovery | Per-recording-event extrinsics by **bundle adjustment over time-synchronized 2D keypoints** under the per-model intrinsics prior (M2.3 R4) — never scene-feature SfM, which is eliminated by measurement. Held-out reprojection acceptance, explicit scale provenance, bound to the M2.2 instance grain. Consumes M2.5's offsets, so it is gated on M2.5. Must measure per-event pose variety before claiming an extrinsic: subject-only calibration degrades on near-coplanar keypoints and a near-static subject, and the tasks are seated upper-limb movements. |
| M2.7 | Gated fusion + corpus study | Fusion over qualified recording events, reprojection/gap/throughput/stability/repeatability evidence, claim-bounded report, prospective-capture specification, de-identified regression fixtures. |

**Unit status.** M2.1, M2.2 and **M2.3 DONE** — M2.3 across ten windows, closing on P29. **M2.4 OPEN, contract frozen, implementation pending** — see below. M2.5, M2.6 and M2.7 are **unblocked**: M2.3's ruling is made and closed, so their shape is settled — M2.6 exists, and it recovers extrinsics from the subject's own keypoints rather than from scene features (R4). M2.5 stays M2.6's precondition.

### M2.4 — gate green, two reviews mid-adjudication

Contract at `.agent/archive/contract-m2u4.md`: **20 predicates P01-P20**, 4 invariant surfaces, gate
identity, an 8-class probe seed, and **35 amendments A01-A35** ruling `test-m2u4`'s phase-1
ambiguities over its **82 cases**. Baseline `6bbd50e`, whose gate read 1116 passed / 0 skipped.

**Gate green, MAIN-verified: `ruff`, `ruff format`, `ty` clean and `1199 passed` in 988.79 s, with
collected == passed and zero skipped, xfailed, xpassed, deselected or errored** — A32's
reconciliation, not a floor. Implementation, tolerance split, schema widening, golden regeneration,
the 82-case red suite and the corpus artifacts are all merged and green.

**The unit is not closeable yet**: `/session-roadmap` completes a unit on green predicates *and*
every review row adjudicated. Two reviews are mid-flight and hold the only remaining work. Next
session harvests them; nothing else is open.

**The defect is larger than a precision improvement — it decides whether the real corpus is
processable.** `1/median(diff(ts))` carries ~1e-3 relative bias against 4-decimal timestamps, and
`trajectory_grid_status` residual grows linearly with span under it: ~0.03 slots per second at
29.97 Hz, crossing `GRID_SLOT_TOLERANCE = 0.25` at ~8.3 s. `nominal_fs` holds ~0.002 flat. On real
decode timestamps (`scout-m2u4` pilot, 10 assets / 13 043 frames) `nominal_fs` places **610/610**
one-second windows on the grid against **530/610** for the biased estimator. The producer checks
windows, never whole clips, so that window figure is the one describing published rows.

**The finding that shaped the contract.** Adopting any estimator makes `gap_too_long` depend on that
estimator's residual, and `QC_POLICY_TOLERANCE = 1e-9` is sized for IEEE754 slack, four orders too
tight. At nominal 30 Hz the 3-slot gap verdict then cycles pass/pass/**FAIL** with clip length
mod 3 — a QC verdict with no physical meaning. Ruled (A10, A11): split the tolerance, keep `1e-9`
representation slack on coverage, add a `1e-4` estimator slack on the gap comparison alone, publish
it as `qc_policy_tolerance` so a consumer reproduces the verdict from the row. Two flips are
intended and pinned — 30 Hz 3-slot stays `pass`, and 60 Hz 6-slot moves `FAIL` → `pass`, which closes
the standing `gap6`/`gap7` polish row.

Scope: `analysis/clinical_features.R` is the only production surface, plus its goldens, the R suites,
`docs/technical/analysis.md` and the new `tests/test_r_timebase_truth.py` seed.
`src/pose_estimation/` timestamp production and rounding stay unchanged;
`analysis/data_extraction.R:100-112` and `analysis/arthrose_diag.R:77-100` divide angular change by
each rounded interval and go to `.agent/polish.md` rather than widening this unit.

**Corpus-scale evidence, 379 assets** (`scout-m2u4-2`). Grid placement under `nominal_fs`
passes **21 651/21 651** windows against **21 571/21 651** legacy, and every one of the 80 legacy
failures comes from a single 119.97 fps asset — the predicted shape, since the bias grows with frame
count per unit time. Every asset's `nominal_fs` residual is no worse than legacy's, so the swap costs
nothing anywhere. The container-header cross-check disagrees on **4/379**, worst 1.46938e-4, across
h264 and hevc and 3 of 4 device configs; A31 rules that a reported outlier, not a gate, because the
header divides `n_frames` by a duration counting the terminal frame while `nominal_fs` divides
`n_frames - 1` by the span that omits it — identical under constant frame rate, separating by
`(terminal_frame_duration - mean_interval) / span` under VFR.

**Next session's whole job.** Harvest the two live reviews, merge their red tests, adjudicate what
they filed, rerun the gate, set M2.4 DONE and commit. Both worktrees carry committed work.

| teammate | worktree | state at handoff | what remains |
| -------- | -------- | ---------------- | ------------ |
| `rev-m2u4-3` | `wt/rev-m2u4` | **40 of 73** rows adjudicated through batch 5 (`9483165`); P01-P20 + A01-A20 done, A13-A20 all pass on a 19/19 cadence lattice; gauge 78%, checkpoint directed | spawn `rev-m2u4-4` on the same worktree for A21-A35 + X01-X19; open `fail` rows are P07 + A09 (below), and P01 + P18 need rescoring because both were fixed after it scored them |
| `rev2-m2u4-2` | `wt/rev2-m2u4` | **25 of 33** mutants scored through `5236a28`, gauge 60% | finish 8 mutants, then merge its red tests as cases beyond the frozen 82 |

`rev2-m2u4-2`'s two accepted findings, both already carrying evidence. **M07 survived-unencoded**:
`return(NA_real_)` -> `return(0)` in `nominal_fs`'s final guard survives every case, and it is
probably pipeline-equivalent since both call sites guard `fs <= 0` — kill it with a unit assertion on
the documented `NA_real_` return, not a pipeline case. **M20 survived-encoded**: C5.18 passed when
both published tolerance reads were replaced by the literals `1e-4` and `1e-9`, because the producer
values equal them; it shipped `test_threshold_oracle_uses_published_tolerances`, green on MAIN and
red on the mutant.

**The one open judgment call.** A34's generalized bound `k * TIMESTAMP_QUANTUM / S_retained` is
stated in the contract and the roxygen, but no test derives `k` or `S_retained`, and A09 also wants
exact one-ULP rejection control. Decide whether that case lands in M2.4 or in `.agent/polish.md`.

Merge `rev2-m2u4-2`'s red tests as cases beyond the frozen 82: the 82 bind to the phase-1 table, and
a mutation-derived case has different provenance. Ruling A09/P07's open row is the one judgment call
left — decide whether a `k`/`S_retained` case lands in this unit or in `.agent/polish.md`.

**Wave-1 state, superseded by the table above; worktrees retained.**

| teammate | branch tip | delivered | open dependency |
| -------- | ---------- | --------- | --------------- |
| `map-m2u4` | (primary) | 12-unit surface map + normative checklist, `.scratch/agents/map-m2u4.md` | none, harvested |
| `scout-m2u4` | `bf09826` | `scripts/probe_timebase_grid.py` + 10-asset pilot; full 379-asset sweep unfinished | P20's sample + byte-identical rerun |
| `spike-m2u4-adopt` | `bc49910` | prototype adoption, U1-U3 filled | golden/QC-verdict blast-radius cells U4-U8 |
| `test-m2u4` | `3bf5f0f` | 80 candidate cases + 26 ambiguities, diff-blind | phase 2 = encode the ruled table as the red suite |
| `rev2-m2u4` | `1bdd315` | 25-mutant catalogue + 8 determinism sweeps, fixed pre-diff | phase 2 = run the campaign against MAIN's diff |
| `rev-m2u4` | `6bbd50e` | nothing committed; phase-1 work was transcript-only and is lost | re-dispatch from the contract |

**Two corrections to teammate output, recorded so they are not re-inherited.** `map-m2u4` U6 claimed
the 30 Hz 3-slot gap "still passes because the comparison is inclusive" — false for
`(n-1) mod 3 == 2`, and the contract's §3 sweep is the disproof. `map-m2u4` U11 rows 6-7 cite
`78352e1` and `2977cec` as prior art; both are this wave's own teammate branches, and **no prior
adoption attempt exists in project history**.

**Sizing, recorded for PLANNING.** One window bought the surface map, the real-corpus probe, the
prototype, the contract and its 26 rulings — and no implementation. M2.4 read as a small unit in the
M2 plan (two call sites, a golden regeneration) and is not one: the two-line estimator swap forces a
QC-policy change, a published-schema change, three version bumps, a golden regeneration and a test
oracle that currently encodes the defect it is meant to catch. **A unit that moves a shipped
threshold's semantics is a kernel unit whatever its line count** — the same lesson M2.1 and M2.3
recorded, arriving here through a different door.

### M2.3 — closed, and what it leaves standing

Contract frozen at `.agent/archive/contract-m2u3.md` — **39 predicates P01-P39**, 4 invariant surfaces, gate identity, an 8-class probe-corpus seed. Rulings R1-R10 are all ruled and **all closed**. **All 39 predicates are green**, P29 closing last in window 10. Per-window trajectory for windows 5-10 → `.agent/archive/m2u3-windows.md`; verdict tables → `.agent/archive/rulings-m2u3.md`. Retained worktrees: `wt/spike-m2u3-audio`, kept because its `_family_coverage` is the P38 oracle until R6's connectivity reconciliation polish row closes. Reports for every stopped teammate are preserved under `.scratch/agents/`.

**Sizing, recorded for PLANNING.** Ten windows against a one-window plan, on a user ruling that the unit run whole rather than be re-split. Window 1 alone (`main=` 87%) bought tooling, the metadata axis, two offset spikes, the cross-modality cross-check and the contract, and shipped one commit. **Size a `kernel` unit by its adversarial surface, not its line count**: 39 predicates over six evidence axes, four needing their own measurement pipeline, is a milestone's worth of contract wearing a unit's label. The suspension of the one-window aim was granted to this unit alone and does not carry forward.

**Settled by measurement** (redaction-safe aggregates, `.scratch/m2u3/*_agg.json`, from committed PyAV):

- **No camera intrinsics metadata exists anywhere in the corpus.** Every `mebx` key over all 1010 timed-metadata tracks: `video-orientation` (376 files), `live-photo-info` (376), `detected-face` + sub-keys (135), `segment-identifier` (123). Intrinsics can only come from a per-model prior — `iPad (5th generation)` fx ≈ 1873.3 px, `iPad Air 11-inch (M2)` fx ≈ 1553.2 px, 4:3→16:9 crop 1.08947×, readout/stabilisation factor unreported — or from self-calibration.
- **The cameras are 2 iPad models over 4 (model, OS) configurations, about 3 tablets, and the `above` and `left` labels were served by different tablets in two eras.** `right` = iPad(5)/16.7 on all 131 assets. Every 3-view family draws from 3 distinct configurations; every subject used exactly 3. Codec tracks device: h264 = iPad(5), hevc = iPad Air. 48 kHz audio = iPad Air/26.5 exactly, so 55 of 137 multi-view families mix audio sample rates.
- **Every canonical asset carries mono AAC audio**, so the audio route covers the whole corpus.
- **Sync QC is stratified by `(model, OS, sample_rate)`** (P29) — `pairs_qc.csv` publishes `stratum_a`/`stratum_b`, `assets_qc.csv` publishes `audio_rate_hz`, and `qualification.json` publishes a `pairs.sync_strata` census keyed on the two strata sorted. Measured over 379 assets: **4 configurations, each carrying exactly one rate** — iPad(5)/16.7→44100 (131), iPad(5)/16.6→44100 (125), Air-M2/26.5→48000 (66), Air-M2/18.1.1→44100 (57). **The rate component therefore adds no partition on this corpus**; it is published anyway, because a stratum that is assumed rather than measured cannot show when that stops being true. **Stratum medians are dominated by manual camera start times and are not a device-latency measurement** — the stratification makes a per-configuration constant *visible* and measures none, which is the whole reason exact iPad input-to-timestamp latency stays unbenchmarked.
- **Cross-view offsets are recoverable.** Audio: 210/246 pairs accepted, confidence ROC AUC 0.96083, 2 false positives per 100 held-out controls, 122/137 multi-view families graph-connected, **35/35 accepted three-view triangles close under one 33.4 ms frame (median 4.451 ms)**, full-corpus cold run 8.256 s. An independent visual motion-energy estimator sharing no code and no signal agrees with audio to **median 12.89 ms, 86.2% within one frame, 41.5% under 10 ms, on the 65 pairs both accepted** (p75 23.10, p95 50.72, max 74.8 ms), measured against the visual spike's corrected control-optimal, creationdate-independent gate at `f82a9a9`: 74/246 pairs, 26/137 families, 0/200 controls, closure on 9/52 families with |r| median 8.08 ms and max 34.12 ms. The pre-correction basis (67/246 gate) read median 10.86 ms / 88.3% / n=60 — **superseded, conclusion unchanged**. **The visual gate's 0/200 control result does not bound its gross-error rate**: of the 9 pairs it accepts and audio rejects, one disagrees by 87.4 s. That evidence bounds the **visual** estimator alone, so R6 gives it a veto and not a vote — audio estimates, the corroborator vetoes only where it cleared its own gate. Requiring agreement was priced and refused: it leaves 111/137 families unrecoverable and 2 closing triangles. The audio figures are unaffected. No drift term is needed: 0/132 qualified audio drifts move alignment by more than one frame.
- **Closure is blind to acoustic-path bias** — propagation delays form an exact cocycle around a triangle — so closure certifies self-consistency, never accuracy. The cross-modality number is the only accuracy statistic this corpus yields, and its magnitude matches the 6-9 ms acoustic bias expected at these camera separations.
- **Rolling shutter bounds every timing claim above and synchronisation never removes it** (P27). Neither iPad model publishes a readout time → the artifact carries a **sweep, not a value: 0–33.33 ms**, with Apple-mobile 1080p line-scan evidence of **12.4–30.9 ms** (37–93% of one 30 Hz frame period) named as a proxy from other devices, never as a measurement of these two iPads. Every closure and cross-modality figure here sits inside that sweep, so none of them shows sub-readout camera agreement. Calling this contribution negligible is prohibited in every document.
- **AAC priming is a measured 0 ms residual, not the predicted bias** (P28). Prediction was rate-dependent — 2112 samples = 47.891 ms @ 44.1 kHz, 44.000 ms @ 48 kHz → a raw untrimmed mixed-rate pair carries a fixed **3.891 ms** bias, and 55/137 multi-view families mix rates. The decode path cancels it: PyAV trims priming, skip = 2112 samples on 379/379 and first decoded PTS = 0 on 379/379. Quote the measured 0 ms; never quote 3.891 ms as a live bias.
- **7 assets change device orientation mid-clip**, which the single display matrix cv2 applies on decode cannot express; 3 assets carry no orientation track at all.
- `com.apple.quicktime.creationdate` is a **coarse sanity check, not an alignment prior**: whole-second, and residuals against measured offsets show multi-second per-tablet clock biases fitting neither a recording-start nor a file-finalize hypothesis.
- **123 assets carry GPS coordinates** (`location.ISO6709`, the iPad Air files). Values never read. Flagged to the user as a data-boundary matter.
- Device-side face metadata on 135 assets shows **7 assets ever holding more than one face**, so the one-subject assumption is not free.

**Ruled.** Verdict table → `.agent/archive/rulings-m2u3.md`; contract §7 carries the one-line summary. R1 view labels, R2 rigidity, R4 the 3D route, R5 the offset representation and A1 the axis wiring are all decided on measurement. R3 (metric scale reference) closed negative in window 6 — see the sampled-negative survey above.

- **Scene-feature extrinsics is eliminated, and M2.6's route is re-specified.** All 246 within-family pairs: **0 recoverable**, cross-view mutual SIFT matches median **13.5**, F-inliers median **8.0** — the algebraic minimum, so those inliers carry no evidence. Two controls make the null geometric rather than procedural: a baseline ladder falling 2812 (same frame) → 1252 → 962 (same asset, far) → **12–19** (cross-view) with 1740–3355 keypoints present per view, and the only 2 rich pairs of 244 being the `above|above` (298.5) and `left|left` (732.5) **view-conflict** pairs — correspondence returns exactly where two cameras share a viewpoint. **M2.6 recovers extrinsics from the subject's own keypoints**, where correspondence is assigned rather than matched.
- **P21's rigidity gate was unadjudicable and is replaced.** Its 4 px accept threshold also served as the MAGSAC inlier threshold, so `residual_p95` could never exceed the gate judging it. Across an 8× threshold sweep `residual_p95` tracks the threshold monotonically while inliers grow 6% → **no gate may be built on it**. `drift_median` moves 5.0% over that range. New gate `drift_p95 ≤ 20 px` = the reprojection tolerance already applied at `triangulation.py:423-424`: **280/298 assets pass, 71/137 families keep every member rigid** (was 76/286 and 3/137 under P21 as frozen). The 278/286 first recorded here was the 4 px instrument's population and is retired.
- **The view label is not a stable camera geometry.** `above` is 85% rigidity-unmeasurable on iPad(5)/16.6 and 3% on Air-M2/26.5, while the same iPad(5)/16.6 tablet's `left` is 6% — so view matters within a configuration and configuration matters within a view. **iPad(5)/16.6 `above` (89 assets, 23% of the corpus) is an unstable camera**: `valid_fraction` median 0.212 against 1.000 everywhere else, highest quiet-border motion energy in the corpus, `decode_status` ok on 89/89, and where 27 of the visual spike's 28 independent flags land. No per-view prior crosses the era boundary.
- **Detectability re-measured after a defect.** `detect_rate` median **1.0** (mean 0.989886, min 0.333333, n=379), 0 inference-failure frames, all four device configurations uniformly high. The prior 0.0 median was an artefact of rtmlib `PoseTracker` IoU-matching seconds-apart samples → `.agent/memory.md`.

**Coverage limits, carried forward:** assets with no rigidity verdict under any gate concentrate in the unstable cell above → `.agent/polish.md`. The 93/379 recorded here is the 4 px instrument's population and is **retired** — R2's amendment puts it at **81/379** (71 support-unmeasurable + 10 orientation-excluded). R3's sampled negative means every artifact permanently states arbitrary scale.

| unit | close | gate (passed / skipped) | `main=` | `mate=` |
| ---- | ----- | ----------------------- | ------- | ------- |
| — | baseline at M2 plan | 621 / 0 | — | — |
| M2.1 | `30280c3`..`6e363a0` | 734 / 0 | 98% 236K/240K, 3 windows | 100% 240K/240K |
| M2.2 | `d9f6c65`..`05fe55a` | 844 / 0 | 100% 240K/240K, 3 windows | 99% 238K/240K |
| M2.3 | `1ae599c`..`5e40922` + close | 1116 / 0 | 90% 216K/240K, 10 windows | 100% 240K/240K (M2.1 peak); 76% 183K/240K in window 10 |

**Sizing analogs** (unique files touched, summed churn; gauges where recorded). M3.2 `16e6fab` = 9 files, +891/−117, `main=95%` — the schema/identity analog for M2.1. M3.3a `a6218e5` = 13 files, +1694/−152, `main=58%` — a full artifact slice. Multi-camera fusion `62685e0` = 14 files, +1040/−164, and calibration `4d4df80` = 18 files, +1472/−156 — the integration band for M2.5/M2.6. Uncalibrated QA `20c36a0` = 14 files, +1225/−152 and adversarial failure modes `36f28a2` = 11 files, +981/−392 — the analogs for M2.7. **M3.3 was planned as one unit and did not fit one MAIN window**; M2.1/M2.2 are split at the same kind of boundary for the same reason.

**M2.1 actual, `30280c3` = 14 files, +4960/−22, `main=98%` across three windows.** It overran the one-window aim, and the overrun was not implementation churn — `inventory.py` is 1225 lines and the two suites are 3262 — it was the review loop: `rev-m2u1` returned 30 findings in phase 1 and 12 more in phase 2, each one costing a ruling, a fix, a contract amendment and a corpus rerun. **Size a `kernel` unit by its adversarial surface, not its line count.** A unit that publishes a durable artifact with a frozen digest pays for every predicate twice. M2.2 is the same shape and should be planned to close in one window only if its contract surface is materially smaller than M2.1's 31 amendments.

**M2.2 actual, `d9f6c65`..`05fe55a` = 16 files, +3006/−22, `main=100%` across three windows.** Its contract surface *was* smaller — 10 sections and 9 amendments against M2.1's 31 — and it overran anyway, for a different reason. `inventory` publishes a table; `sessions` publishes a **tree**, so every predicate has a filesystem failure mode behind it: a swap that fails, a sibling left under a dead pid, a symlinked `--out`, a corpus that moves between planning and linking, an output that contains its own input. Two reviewers filled 14 rows each in phase 1, then kept finding — nine more accepted defects arrived after both phase-2 markers, clustered on exactly those crash states plus the alphabets that keep hostile registry cells out of published names. **Budget a publishing unit by its crash states, not its contract sections**, and expect the review loop to outlive its own completion marker: an adversarial reviewer with context left is still the cheapest defect source in the wave.

**M2.1 → M2.2 handoff.** The census tool writes three artifacts to `inventory/`, self-verifying through `validate_generation()`: `assets.csv` (one row per discovered file — canonical corpus-relative path, disposition, reason code, SHA-256, container facts), `captures.csv` (one row per task-side family), `census.json` (redaction-safe aggregates plus a `generation` block digesting all three, the census entry being a digest of the census minus its own key). M2.2 reads `assets.csv`; it does not walk the corpus again.

- **`capture_id` names a task-side family, never a recording event** — `(subject, task, side)`, no take component. **The instance grain is ruled: `event_id = f"{capture_id}_run-{run_index:02d}"`**, so no event key can ever equal a family key and the standing "never bind calibration to `capture_id`" constraint becomes unrepresentable rather than merely documented. `run-<index>` is BIDS's own entity for an otherwise-identical repeated acquisition; prior art (Pose2Sim, OpenCap, Anipose, EasyMocap, FreeMoCap, MMPose) uniformly puts the recording event below any participant or visit grouping and never treats a semantic family as proof that clips are one event. `run_index` is **not chronological and asserts no provenance** — assignment order is the registry's `source_path` code-point order, which only makes it deterministic. Full contract → `.agent/archive/contract-m2u2.md`; MAIN's verdict table → `.agent/archive/rulings-m2u2.md`.
- **Take resolution for the 2 view-conflict families is ruled: none is asserted.** No published pipeline infers same-take membership from filename, file order, duration, frame count, or creation-time proximity — membership is declared at acquisition, and alignment is a separate later step off a decoded shared signal. Header facts separate neither conflict. So each asset of a conflicted family becomes its own single-camera run with `take_resolution = "unresolved"`, and its run count must never be read as a performance count. M2.3 may resolve them by decode. Unequal frame counts across views are compatible with one event after offset estimation, so **the frame-parity figures above are evidence of neither sameness nor difference**.
- Path text is re-decoded strictly as UTF-8 once at discovery, so classification, parsing, ordering and published text are a function of corpus bytes rather than of filesystem locale; a non-UTF-8 name keeps its surrogate form. M2.2's symlink names must come from that canonical column, never from a fresh directory walk.
- 379 canonical / 3 quarantined / 0 excluded. Quarantine is a **stem-grammar** verdict, not a readability one — all three files open and probe. M2.2 holds all three out as `quarantined_stem`; republish the registry to admit them.
- `census.json` is the one redaction-safe artifact: no filename, no path, no subject directory name, recognized extensions only. `assets.csv` and `captures.csv` are patient-adjacent; `inventory/` is gitignored, and so are `sessions/` and `sessions.*/`.
- **Two committed gates back every claim above and both rerun from committed state** — `scripts/run_inventory_mutations.py` (72 mutants, 71 killed, `M028` alone surviving as a ruled equivalent) and `scripts/check_inventory_determinism.py` (20 sweeps, 0 failures, plus 13 tamper classes the consumer boundary rejects by exception class). A predicate M2.2 adds to the registry earns a mutant in the same commit.

**M2.2 → M2.3 handoff.** `pose-estimation-sessions` publishes `sessions/` from `inventory/` and never walks the corpus: **193 events over 382 assets** — 58 one-camera, 84 two-camera, 51 three-camera; 186 `family` and 7 `unresolved`; 379 placed and 3 held out. Each event directory holds one `session.json` and one `cam-*` symlink per camera, `discover_sessions` returns all 193, and the tree regenerates byte-identically under a changed locale, hash seed, time zone and `--out` name. Every consumer calls `validate_generation(out, inventory_dir=…)` before reading a row: the two-argument form is the only check that catches a registry rebuilt under a tree which still looks internally consistent. Shipped surface → `docs/technical/sessions.md`; contract and rulings → `.agent/archive/contract-m2u2.md`, `.agent/archive/rulings-m2u2.md`.

- **The 7 unresolved families are M2.3's to resolve or to leave.** `take_resolution = "unresolved"` asserts no multi-camera event, so each of their assets is its own single-camera event and their run counts understate the true grouping. Nothing in the tree infers take membership from a filename, a file order, a duration, a frame count, or a creation time; M2.3's decode evidence is the first thing that could.
- **`sync_offset` is 0 and unmeasured on every camera, and no manifest declares `calibration`.** Both fields exist and assert nothing. M2.5 and M2.6 fill them.
- **Nothing is decoded yet.** The generator asks the filesystem whether each listed path is a regular file and stops there, so container facts still come from M2.1's cv2 probe alone.
- **Two distinct orientation policies, and only the probe's is explicit.** `probe_container` sets `CAP_PROP_ORIENTATION_AUTO=1` and records `CAP_PROP_ORIENTATION_META` + `CAP_PROP_ORIENTATION_AUTO` (`src/pose_estimation/video_io.py:243-254`); every decode path relies on the backend default instead, which happens to be auto-rotate, so frames arrive upright by convention rather than by assertion. The hazard is that a rotated view has different image geometry from its siblings, not that frames are sideways. **7 assets change orientation mid-clip** (`[1,8]`, `[1,6,8]`×2, `[3,6]`, `[1,6]`×2, `[1,3,6]`) and 3 carry no orientation track at all — a single display matrix cannot express either case, so a per-asset rotation constant is wrong for 10 of 379.
- Decoder/tool matrix: **cv2 + PyAV** (`av>=17.1.0`, `pyproject.toml:35`; measured 18.1.0). `ffmpeg`, `ffprobe` and `exiftool` remain absent. PyAV supplies true PTS, creation timestamps, audio tracks and Apple `mebx` metadata; the metadata axis is already measured off it.

**Standing constraints.**

- **Capture identity has no schema home.** Producer keys are `video`/`person_idx`/`window`; `session.json` carries no task, side, or family field; `world3d.csv` reduces `video` to `session_id`. M2.1's registry is the single source of family identity, bound by `capture_id`. Legacy 2D and 3D producer schemas stay unwidened — `analysis/utils.R:59-87` treats every numeric non-metadata column as a feature.
- **Calibration identity is unbound.** Discovery accepts any calibration whose camera names match; nothing compares rig or session identity (`src/pose_estimation/multicam.py:364-383,579-588`). Per-recording-event calibration makes this a live hazard, so M2.6 must bind calibration to the instance grain M2.2 resolves. It may **never** bind calibration to `capture_id`: a `view_conflict` family holds more than one take, so that key does not name a recording event.
- **View labels are lexical, not geometric.** `above`/`left`/`right` are filename tokens. M2.3 verifies them against measured geometry before any calibration reuse; a mismatch projects pixels through the wrong camera.
- **Provisional QC thresholds.** `coverage ≥ 0.80`, `max_gap ≤ 0.10 s` are engineering defaults carried under `qc_policy_version`, not validated standards. M2.7 is where evidence replaces them.
- **One subject only.** Fusion reads `person_idx == 0`; cross-camera identity matching does not exist.
- **Decisive gate is primary-tree.** `renv/library/` is gitignored, so worktrees skip R cases unless symlinked; a green worktree run is no evidence for `analysis/*.R`.

**Acceptance:** every one of the 382 files reaches exactly one explicit outcome — canonical family member, quarantined stem, or recorded exclusion — with nothing silently dropped; the session tree regenerates byte-identical from a clean base and `--list-sessions` enumerates it; every corpus claim traces to a committed rerunnable command rather than a scratch script; the claim boundary above is honored in every artifact and document; full suite passes in the primary tree with 0 skips.

## M3 — analysis-ready 3D aggregation

**Status: DESCOPED** — terminal, never a dispatch target; reviving any part is a PLANNING ruling. M3.1, M3.2 and M3.3a shipped and stay in the tree with their gates. M3.3b and M3.4-M3.6 are cut: real data replaced the synthetic development surface, and no remaining unit is forced by it. `clinical_3d_video_aggregate.csv` was never built and nothing references it, so the cut falsifies no shipped claim. **Range** `b1f5b81`..`429b0f4`, descope ruling in `89b4fdd`. **Gauge band** `main=` 58% (M3.3a) to 95% (M3.2); M3.1 closed before gauges were recorded.

What survives in the tree: the timestamp-aware trajectory kernel (`zoo` dropped), the 3D producer identity schema, and `<stem>_clinical_3d_window_qc.csv` over the four trajectory groups. The QC artifact explains 12 trajectory metrics and is silent on `bilateral_*`, `trunk`, `shoulders`, `cpi` — `docs/technical/analysis.md` *Current scope* now states that as standing scope.

The cut released M2.4: M3's "2D goldens byte-identical" acceptance was the only reason `nominal_fs()` shipped unadopted, and re-deriving `output/rtmw-l_body_single/` stopped mattering when `videos/initial/` was retired. Full record, including the retained unmerged branches and the frozen M3.3 contract → `.agent/archive/m3.md`, `.agent/archive/contract-m3u3.md`.

## Produced datasets

- `output/rtmw-l_body_single/` — **preliminary**, from the retired `videos/initial/` clips. 12 single-camera clips, RTMW-L / `--tracking body` / `--single-subject`, det-CPU + pose-NPU; 15 430 rows over 15 455 frames, 99.7% mean coverage, 100% body-wrist observation. Kept on disk, not regenerated. Its clinical features predate both the M3.1 gap fix and M2.4's cadence fix, so **any normalized-jerk or velocity figure from it is suspect** — recompute before citing it anywhere, including a paper's preliminary-work section. Schema conformance and coverage figures are unaffected and remain quotable.

## Backlog

Scope seed for the milestone after M2.

- **Clinical join surface** — the eventual destination for M2's numerical output: the hospital SCI database (`database/ALL_SCIDATA.csv` + `SCI_DATABASE_HEADER.xlsx`), currently analyzed as a dashboard in `Projects/rehab/`. **The subject↔patient mapping is unknown and has to be established first**; nothing in `videos/3-cam/` identifies a database record. Needs capture/session metadata, then a capture→assessment bridge with instrument/version/domain/side/status and cardinality safety. ISNCSCI is side/myotome-resolved while SCIM is whole-person, so the join grain cannot be settled against synthetic data.
- Cross-camera identity matching for multi-person scenes; fusion assumes one subject.
- Gap-aware movement-phase metrics — M3.1's kernel covers frame/window scope only; `analysis/clinical_features.R:918-1097` phase speed/path/SAL/NJ/efficiency stay gap-unsafe and explicitly unqualified.
- Prospective calibrated capture — the only route past M2's claim boundary. M2.7 specifies it; running it is a separate milestone with its own clearance and ethics footprint.

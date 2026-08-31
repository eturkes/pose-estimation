# M2.6 acceptance contract — calibration recovery

**Status: DRAFT, wave-1 frozen parts only.** §1-§6 are ruled and bind. §7 names three open forks that
`spike-m2u6-*` decides next window; the predicate list §8 closes when they do. Baseline `41efc55`,
gate **1284 passed / 0 skipped / rc=0 in 873.00 s**, MAIN-measured.

## §1 Scope + grain

Recover per-recording-event camera extrinsics by bundle adjustment over time-synchronized 2D human
keypoints, under a per-device-model intrinsics prior. Publish them with held-out reprojection
acceptance, an observability verdict, and explicit scale provenance.

- Grain = M2.2's instance key `event_id = <capture_id>_run-<NN>`. Binding to `capture_id` is
  forbidden: a `view_conflict` family holds more than one take, so that key names no recording event.
- Scene-feature SfM is eliminated by measurement (0/246 pairs recoverable, mutual SIFT median 13.5,
  F-inliers median 8.0 = the algebraic minimum, two controls). Correspondence is **assigned**
  (keypoint `k` in A is keypoint `k` in B), never matched.
- Out of scope: fusion over recovered extrinsics (M2.7), prospective capture (M2.7), any metric
  quantity (permanently, per M2's closed scale ruling).

## §2 Inputs, all validated before a row is read

`inventory/` -> `sessions/` -> `measurements/` -> `qualification/`. Each consumer calls that
publisher's `validate_generation` first. M2.6 adds no new upstream.

- Offsets: `qualification/cameras_qc.csv`, one row per placed asset, columns
  `event_id, asset_id, camera_name, view, offset_s, offset_status, is_reference, reference_camera`.
  `offset_s = t_camera - t_reference`; apply exactly `t_ref = t_camera - offset_s`. Consume rows in
  `{reference, solved}` only; `unreachable` and `unmeasured` carry no offset.
- Membership: `sessions/placements.csv` and each event's `session.json`. Never re-derive event
  membership from the capture family — a view-conflict family resolves to several single-camera
  events, and a family-wide derivation credits each with cameras it does not hold.
- Per-asset evidence already published in `qualification/assets_qc.csv`: `device_config`,
  `rigidity_flag`, `rigidity_valid_fraction`, `detect_rate`, `orientation_values`,
  `orientation_changes`.

## §3 Candidate population, MAIN-derived at `41efc55`

Every count names its population.

- Events by camera count: **58 one-camera / 84 two-camera / 51 three-camera = 193**.
- `offset_status` over 379 camera rows: **193 reference / 162 solved / 24 unreachable** = 355
  offset-bearing.
- **Events with >= 2 offset-bearing cameras = 121** (80 two-camera + 41 three-camera). This is the
  ceiling: one usable camera has no relative pose to recover. 115 are `sync_status = connected`;
  **6 are `unconnected`** — P07 partial-publication events whose reference sits inside a two-camera
  component.
- View-sets over those 121: `above|left|right` 47, `above|right` 38, `left|right` 19, `above|left` 17.
- Rigidity over the **283 offset-bearing cameras inside those 121 events**: **223 `rigid`,
  46 `unmeasurable`, 9 `camera_motion`, 5 `excluded_orientation`**. `detect_rate` over the same 283:
  **median 1.0000, min 0.7083**.
- **Events where every offset-bearing camera is `rigid` = 64** of 121.

121 is the ceiling, 64 the strictest floor. The 57-event difference turns on whether `unmeasurable`
rigidity disqualifies a camera; 46 of the 60 non-`rigid` cameras are `unmeasurable`, which is absence
of evidence rather than measured motion. **Ruled: `unmeasurable` does not disqualify by itself.** A
camera is refused only on measured disqualifying evidence (`camera_motion`, `excluded_orientation`)
or on M2.6's own observability verdict, which measures the thing directly instead of proxying it.

## §4 The scale/unit ruling — decisive, and it blocks the obvious implementation

Writing an M2.6 extrinsic into today's `CameraCalibration` unchanged propagates a false metres claim
through six shipped surfaces in order: `_types.py:91` (`tvec` "metres") ->
`docs/technical/calibration.md:43,105` -> `triangulation.py:12,454` (world outputs metres) ->
`export.py:443-487` (every world column named `{name}_{x,y,z}_m`) ->
`validation.py:977-994,1265-1293,2095` (`_temporal_jitter_mm` multiplies by 1000 and renders
millimetres) -> `analysis/clinical_features.R:10-14,101-104,253-270` (publishes
`coord_space="world-metric-3d"`, `distance_unit="m"`). The projection math is unit-agnostic, so every
one of those accepts arbitrary units numerically while its contract lies.

**Ruled.** M2.6 publishes into a **separately typed arbitrary-scale extrinsics artifact** carrying its
own explicit scale provenance. It does not write into `CameraCalibration` and does not rename
arbitrary units to metres. A metric-only consumer must fail closed rather than silently accept an
arbitrary-scale calibration; which consumers gain that refusal, and whether the refusal ships in M2.6
or M2.7, is fork F3 below.

Gauge freedom is explicit: the recovered geometry is fixed up to one global similarity. The
world-frame camera pins rotation and translation; the remaining scalar is unmeasurable in this corpus
and every published extrinsic carries `scale_provenance = arbitrary`.

## §5 The carrier — ruled negative on the two obvious homes

- **Never patch `session.json`.** `sessions.validate_generation` hashes the whole tree, so an
  in-place manifest edit turns a valid M2.2 generation into an invalid one. The roadmap's older
  "M2.6 still fills `calibration`" line is superseded by that measurement.
- **Never write `sync_offset`.** It is the legacy integer pre-roll trim in the fusion reader's frame
  domain; `offset_s` is a time-domain float. Writing one into the other changes domains.
- **`events_qc.geom_qualified` cannot be filled by a publisher that reads `qualification/`.**
  `qualify` consuming an artifact that consumes `qualification/` is a publisher cycle — the same
  cycle M2.5 refused for `sync_offset`. `geom` is confirmed an M2.6-owned event-level extrinsics
  reason token (`.agent/archive/m2u3-windows.md:37`, `docs/technical/qualification.md:357`), and
  today every one of the 193 rows reads `geom_unmeasured`. **Where the verdict lands without a cycle
  is fork F1.**

## §6 Solver route — ruled on published evidence

- **Library: `scipy.optimize.least_squares(method="trf")` with an analytic CSR Jacobian.** Zero new
  dependencies (SciPy 1.18.1, BSD-3-Clause, already a project dependency); `jac_sparsity`,
  `tr_solver="lsmr"`, bounds and robust losses are all present; `lm` supports neither sparse
  Jacobians nor robust loss. Three cameras leave 12 free extrinsic DoF after gauge fixing; the large
  block is independent 3D points whose observation-to-camera/point sparsity is trivial. `pyceres` 2.6
  is the fallback, funded only if a committed benchmark shows SciPy missing an explicit runtime gate.
  Determinism is a pinned-input property, not a solver promise: fix observations, initialization,
  variable order, solver options and single-threaded BLAS, then byte-test repeated output rather than
  claim cross-platform bit identity.
- **An articulated, moving subject is geometrically valid.** `u_b^T E u_a = 0` is pointwise, so each
  synchronized `(t, k)` observation may come from a different world point without violating one fixed
  camera-pair `E`. Labels remove matching ambiguity, not geometry or noise. Therefore stratify
  minimal RANSAC samples across frames and body regions, so no single limb or single instant defines
  the pose; five correlated joints from one frame are not five independent observations.
- **Algorithmic template: Lee et al., IEEE RA-L 7(4) 2022** — stationary synchronized cameras,
  pre-calibrated `K`, per-view 2D poses, linear rotation/translation recovery, then robust BA over
  reprojection plus bone-length variance. Their real-dataset full-joint result is 0.020 rad,
  0.053 m, **3.014 px** reprojection; synthetic small-motion (0.5x0.5 m) degrades to 0.023 rad /
  0.051 m / 0.535 px against large-motion (2x2 m) 0.007 rad / 0.020 m / 0.338 px — **motion extent is
  the dominant accuracy term, and this corpus is seated upper-limb work at the small-motion end.**
  A 1.6%-wrong intrinsic prior moved their reprojection from 3.73/4.66 px to 6.84/7.32 px, which
  prices the unmeasured iPad focal prior directly.
- **Held-out reprojection on the same keypoints is not independent evidence.** Pätzold et al.,
  GCPR 2022: their human-keypoint solution beat the reference calibration on human reprojection
  (4.01 px vs 4.57 px) while losing to it on independent AprilTags by 3.05 px (5.00 px vs 1.95 px).
  Any M2.6 acceptance statistic computed on the same keypoint family that trained the solve measures
  self-consistency, exactly as M2.5's closure does. **Label it self-consistency, never accuracy** —
  the same discipline M2.3 applied to acoustic closure.

## §6b Pilot observability result — the fork above all others

`scripts/probe_calibration_observability.py` at `wt/scout-m2u6`, **pilot population = 3 eligible
three-camera events / 9 camera pairs / 72 synchronized pair-frames**. Shared confident keypoints at
confidence 0.5 = **median 24** per pair-frame over the 65-keypoint geometry set.

**Pooled `recoverPose` produces a pose for 9/9 pairs, and every consistency check refuses it.**

- Rotation cycle around the three-camera loop: **34.87-59.03 deg, median 47.34** -> **0/3 events close
  within the predeclared 10 deg bound**.
- Temporal stability: **0/9 pairs** yield a per-frame quality pose. Only 4/9 yield >= 2 four-frame
  split poses, and 3 of those 4 exceed a 10 deg rotation spread (median split spread 51.16 deg,
  max 168.31).
- Held-out epipolar support: measurable on 4/9 pairs, **below 0.5 on all 4** (pair-median distribution
  median 0.0957); the other 5/9 are unmeasurable.
- All-133-keypoint sensitivity route is no rescue: 0/3 cycles close, one complete-quality cycle at
  167.21 deg.
- 7/9 pairs clear 30 cheirality inliers and the accepted edge graphs connect 3/3 events, so **every
  structural check passes while every geometric check fails**. A gate built on pose existence, inlier
  count or graph connectivity would have published these as recovered extrinsics.
- **Alignment is not the blocker.** Realized reference-time residual over the sampled frames is
  median **6.31 ms**, p95 21.88, max 32.71 — inside one 33.3 ms frame, so M2.5's offsets are doing
  their job and the failure is geometric.

**Bound of the null.** It bounds *pooled two-view `recoverPose` initialization* on 3 events. It does
not bound a full bundle adjustment under a per-model intrinsics prior, which the pilot did not run.
The scaled 22-event sample was still running at wave close. Read it as a strong prior against the
naive route and as the reason F0 exists — not yet as M2.6's verdict.

**F0 — does M2.6 exist as scoped?** If the scaled sample reproduces the pilot, the unit's honest
output is a measured negative with a claim-bounded artifact publishing `geom_unqualified` and its
reason, in the same shape as M2.3's R3 and R4 closures — not a calibration nobody can trust. The
alternative live hypothesis is that initialization, not observability, is what failed: 24 shared
points per frame across a seated upper-limb subject is close to the small-motion regime where Lee et
al. already degrade, and their route reaches a pose through a *linear* rotation/translation recovery
plus bone-length-regularized BA, never through pooled pairwise `recoverPose`. Deciding between those
two readings is the next window's first question, and it is answered by running BA, not by arguing.

## §7 Open forks — `spike-m2u6-*` decides these next window

**F1 — where the event geometry verdict lands, without a publisher cycle.**
- F1a: a new `calibration/` publisher carrying its own event table; `events_qc.geom_qualified` stays
  `geom_unmeasured` forever and the docs say the verdict lives one layer down.
- F1b: a new `measure/` sidecar axis (`geometry`), so `qualify` derives `geom_qualified` through the
  existing axis path. Needs the offset solve reachable without reading `qualification/`.
- F1c: extract the gauge-fixed offset solve out of `qualify.py` into one shared importable function
  that both `qualify` and the geometry axis call on the same `measurements/` sync rows. One solver,
  no duplication, no cycle. Costs a `qualify` refactor under its mutation and determinism gates.

Decide on measured cost, not tidiness. F1c is the only option that leaves one solver in the tree; the
`publication.py` extraction row in `.agent/polish.md` collides with all three and must be ruled in the
same pass rather than left to go stale.

**F2 — the producer configuration for BA input keypoints.** The default `pose-estimation-run` path
applies a temporal `KeypointSmoother` and then a `BoneLengthSmoother` that mutates x/y **in place
before export** (`run.py:453-486,766-780`, `constraints.py:105-117`), and the exported CSV header
carries no model hash, device placement, smoothing policy, detector cadence, orientation policy or
generator version. Raw BA input needs a ruled configuration (`--no-smooth --no-constraints` or
equivalent) plus a provenance binding, not an unstated reuse of existing CSVs. Decide whether M2.6
consumes exported CSVs at all or extracts keypoints itself the way `measure/detect.py` does.

**F3 — how far the arbitrary-scale refusal propagates in this unit.** §4 rules the artifact; it does
not yet rule which of the six metric consumers gains a fail-closed check in M2.6 versus M2.7.
Minimum spine = the artifact plus its own refusal. Everything past that is a candidate polish row.

## §8 Predicates — partial; closes when §7 closes

Frozen now:

- **P01** Every published extrinsic is keyed by `event_id`, never by `capture_id`.
- **P02** Every consumer validates each upstream generation before reading a row.
- **P03** Only `offset_status in {reference, solved}` camera rows are consumed; an event with any
  `unreachable` or `unmeasured` camera publishes a verdict for the cameras it can reach and names the
  ones it cannot, exactly as M2.5's P07 does — partial publication, never a silently reduced event.
- **P04** Every published extrinsic carries `scale_provenance = arbitrary`, and no M2.6 output uses a
  metre-valued name, suffix or unit label.
- **P05** The acceptance statistic computed on the solve's own keypoint family is published as
  self-consistency. The word accuracy is prohibited on it.
- **P06** Observability is measured per event and published as a number, before any extrinsic for
  that event is claimed. An event failing the observability gate publishes the failure and no pose.
- **P07** Frame geometry is explicit per asset: the orientation transform applied to keypoints is the
  same one applied to `K`, and the 7 assets that change orientation mid-clip plus the 3 with no
  orientation track are either handled by a timed transform or refused by name.
- **P08** Publication mirrors `qualify`'s idiom in full: `GENERATOR_VERSION`, closed column registry,
  `_canonical` row order at the publish site, alphabets built by `fullmatch` from constant
  frozensets, staged-then-swapped generation, tree digest, marker regularity + duplicate-key
  rejection, `_assert_owned`, `_is_within`/`_assert_disjoint`, and an exported `validate_generation`.
- **P09** The digest detects corruption and does not authenticate. Any document stating otherwise is
  a defect.
- **P10** Every number in the roadmap and the docs reruns from committed state.

## §9 Gate identity

`env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <cmd>` for each of `ruff check`,
`ruff format --check`, `ty check`, `pytest`, in the primary tree. Both halves are load-bearing.
Baseline at `41efc55` = 1284 passed, 0 skipped, rc=0, 873.00 s.

## §10 Shipped violations found in wave 1, to rule rather than inherit

Each is pre-existing at `41efc55`; none is caused by M2.6, and each becomes M2.6's problem the moment
an extrinsic enters the path.

1. `multicam._resolve_calibration` (`multicam.py:364-383,579-588`) binds any schema-valid calibration
   whose camera names match, with no session, event or rig identity comparison.
   `--calibration` with `--sessions-dir` deliberately reapplies one file to every discovered session
   behind a warning (`multicam.py:195-232`). This is the standing unbound-identity hazard, now live.
2. `calibration.save_calibration` (`calibration.py:63-75`) documents that partial or corrupt files
   never land, and writes the final path directly with no staging and no atomic swap.
3. `calibration._validate_*` admits non-finite and non-numeric cells and Boolean resolution
   components, and can leak `TypeError`/`ValueError` where `docs/technical/calibration.md:50-59`
   promises `CalibrationError`.
4. `rtmlib_openvino.py:68-89` catches a non-CPU compile failure, silently recompiles on CPU, and
   never reads `EXECUTION_DEVICES` — against `CLAUDE.local.md`'s exact-device, loud-failure rule.
5. `inventory` and `sessions` markers use duplicate-tolerant `json.loads` and, for `sessions`, an
   ownership test accepting any marker carrying a `generator_version` key. `qualify._load_generation`
   is the idiom to copy. This closes the "unverified" item standing in `.agent/memory.md`.
6. `multicam.discover_sessions` (`multicam.py:174-192`) enumerates session children directly,
   skipping the publisher boundary `docs/technical/sessions.md:158-164` requires.
7. `docs/technical/qualification.md:122-124` still says detect and rigidity are unproduced; the
   validated marker lists both as measured, with only `scale` unmeasured.

## §11a Amendments A01-A07 — window 2, F0 measurement

All seven bind and supersede the §6/§6b text they name. A01-A03 are `spike-m2u6-ba` findings ruled by
MAIN; A04-A07 are MAIN measurements over the scaled sample.

**A01 — §6's algorithmic template does not run on 2D input.** Lee et al. §III-A factorizes per-camera
3D bone orientations produced by a single-view 3D pose estimator; §III-B uses 2D rays only after
those rotations exist. §6 names the paper as the template without recording that dependency. Ruled:
M2.6 runs a **surrogate initializer**, labelled in every artifact as not-Lee, with the reason stated.
Option A — adding a monocular 3D-pose model — is refused: §6 forbids new dependencies, and monocular
lifting carries its own scale and pose ambiguity, so it adds an error source rather than removing one.

**A02 — cycle closure is an acceptance statistic ONLY for independently estimated pairwise poses.** A
joint multi-camera BA parameterizes one global `R_c` per camera, so `R_20` against `R_21 R_10` closes
to floating-point zero by construction, for a converged and an unconverged solve alike. Rotation
averaging over the camera graph has the same effect. Post-joint-BA closure may be reported, and must
be labelled non-evidential. Discriminating statistics for any gauge-coupled arm are frame-disjoint
held-out reprojection and split-half independent-solve pose spread.

**A03 — a raw bone-length variance penalty cannot coexist with free scale.** Reprojection is
invariant under `(X, t) -> s(X, t)`, so any positive raw bone penalty drives `s -> 0`. Ruled: fix one
baseline norm to 1 as an arbitrary NUMERICAL gauge — that is §4's unmeasurable similarity scalar
pinned rather than measured. `scale_provenance = arbitrary` is unchanged and the unit baseline is
never named or implied to be a physical length. Scale-normalized bone dispersion is the reported
sensitivity arm.

**A04 — §6b's per-frame stability refusal measured a constant, not the scene.** `frame_estimates`
requires `recover_inliers >= MIN_POSE_INLIERS = 30` inside ONE frame
(`scripts/probe_calibration_observability.py:1033-1040`), while a frame carries a median 24 shared
`calibration65` keypoints. Achievable inliers cannot exceed the shared count, so the check is
arithmetically unreachable for that keypoint set; observed `frame_poses` median 0 / max 1 over 42
pairs confirms it. Same defect class as M2.3's P21. **"0/9 pairs yield a per-frame quality pose" is
retired as evidence.** Any per-frame stability predicate must carry an inlier floor a single frame
can reach, or use the split-pose statistic instead.

**A05 — §6b's held-out epipolar refusal is evaluated at the solve's own threshold.** Held-out support
is measured at `RANSAC_THRESHOLD_PX = 3.0`, the same constant that selected the inliers. Median
0.052 over 19-20 measurable pairs. Under plausible cross-view keypoint correspondence error a correct
pose scores similarly, so the statistic does not currently discriminate and may not gate. Sweeping it
is a precondition for gating on it — the standing project rule that a gate constant must never double
as an instrument parameter.

**A06 — the pilot's 0/3 is superseded by 2/10.** Over the scaled sample, all 10 three-camera events
are `above|left|right` spanning 2 device configs, and cycles read **2.31, 7.75, 17.06, 31.24, 34.87,
39.48, 40.26, 47.34, 59.03, 105.74 deg — 2/10 inside the 10 deg bound**. The pilot's three events are
the 34.87 / 47.34 / 59.03 rows. Planar degeneracy is refuted (median homography inliers 0-1 against
median 39 essential inliers) and low parallax is refuted (median 72-129 deg). **§6b's "every
geometric check refuses it" no longer holds as stated.**

**A07 — undersampling is refuted, and the refutation is the strongest observability evidence held.**
Re-collecting the same 22 events over the same time spans at 32 frames per event raised pooled shared
correspondences ~4x (median 192 -> 749, max 1480) and did **not** improve closure: 2/10 inside 10 deg
either way, median 39.48 -> 47.42. Per event it swung erratically — 47.34 -> 6.23, 17.06 -> 113.44,
59.03 -> 135.62. An observable geometry tightens as data is added; an estimate moving 96 deg under 4x
data is a **biased** estimator, not a noisy one. The between-event correlation of closure with shared
count that MAIN recorded earlier in the window does not survive within-event resampling and is
retired.

**Standing after A01-A07.** Refuted as the cause of the null: planar degeneracy, low parallax,
undersampling, alignment (sync residual median 7.85 ms, p95 25.09, max 32.71, n=336, inside one
frame). Live: **cross-view keypoint correspondence bias** — the same anatomical joint localized from
two widely separated views is not the same 3D point, and a view-dependent systematic offset violates
the epipolar constraint in a way no sample size averages away — and estimator/initialization.
**F0 is not closed.** It closes on `spike-m2u6-ba` arm 1 (independent pairwise BA, cycle composed,
against the 2/10 baseline) and `spike-m2u6-sweep` S04/S08 (threshold curve, and whether the residual
is zero-mean noise or structured bias).

**A08 — two-camera events admit no cycle check at all, and no predicate covers them.** 80 of the 121
candidate events carry exactly two offset-bearing cameras; closure needs three. Their only internal
statistic is held-out support, which A05 shows is currently uninformative. This governs two thirds of
the candidate population and the predicate list does not reach it.

## §11 Probe-corpus seed

Classes an M2.6 suite must cover, seeded now so `test-m2u6` is diff-blind against them: two-camera
event; three-camera event; event with one `unreachable` camera; event whose cameras span two device
configurations; near-coplanar keypoint set; near-static subject; orientation-change asset;
no-orientation-track asset; single-camera event; event with a camera at `detect_rate` minimum;
mixed audio-rate event; view-conflict-derived single-camera event.

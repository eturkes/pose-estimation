# M2.8.4 — corrected corpus re-run: isotropic coordinates + `--tracking body`

Tier `kernel` (source change shapes every published number) + `data` (the re-run artifact).
Base `387560d`. Unblocks M2.8.3.

## 1. What this unit is, after measurement replaced its plan

The roadmap funded two repairs. **Repair (b), the orientation fix, is REFUTED and must not be
built** — §2. **Repair (a), `--tracking body`, stands.** The measurement that refuted (b) located the
real defect under it: the anisotropic `[0, 1]` normalisation, which distorts every angle by a
measured median 9.9° and additionally scales a y-over-x ratio by **3.16×** between the 351 landscape
and 28 portrait assets. User ruling: **one run carries `--tracking body` and the normalisation
fix.**

## 2. The orientation repair is refuted — three independent measurements

The roadmap recorded that OpenCV "ignores the container display matrix and **cannot be made to
honor it**", and that 38 of 379 assets were pose-estimated non-upright. All three measurements below
refuse that.

**M1, pixel identity across open paths.** For each rotation class, decode the first frames twice:
once through the bare `cv2.VideoCapture(path)` the run path uses, once with an explicit
`CAP_PROP_ORIENTATION_AUTO = 0`. The default decode equals the auto-off decode transformed by
**exactly the container's declared rotation** — rot-0 `identity` (8/8), rot-90 `cw90` (8/8), rot-180
`rot180` (8/8), rot-270 `ccw90` (1/1). Explicit `CAP_FFMPEG` and explicit `CAP_ANY` behave
identically to the bare open, and `AUTO = 1` behaves identically to the default. `AUTO = 0` is the
only setting that yields an unrotated frame, and **no code path in this repo sets it**.

**M2, dimension agreement.** Under the default open the reported `CAP_PROP_FRAME_WIDTH/HEIGHT`
equal the decoded frame's own dimensions in every class — 1920×1080 for rot-0/rot-180, 1080×1920 for
rot-90/rot-270. The premise of the recorded defect was that the properties report display size while
the pixels stay coded; measured, they agree.

**M3, corroboration from the shipped corpus output**, which is what makes this a statement about the
8.7 h run rather than about a probe. rot-180 assets are landscape, so they share rot-0's aspect and
differ from it only in whether the display matrix was applied. Median per-asset
`elbow_y − shoulder_y` over the published landmark CSVs: rot-0 **+0.0966** (95% of assets positive),
rot-180 **+0.0347** (90%), rot-90 **+0.0378** (81.5%), rot-270 **+0.0302** (100%). Elbow below
shoulder in image coordinates is upright in all four classes. An upside-down decode inverts that
sign; it is not inverted.

**Consequence, and it is why this section is long.** The planned `cv2.rotate` keyed on
`CAP_PROP_ORIENTATION_META` would have rotated frames the backend had **already** rotated —
corrupting 10% of the corpus while the roadmap recorded that it repaired it. The evidence M2.8.3
cited was real; its cause was misattributed. Detection-rate variation across rotation classes and
the `left_reach_norm` 2.205 → 1.273 shift are both explained by §3, on frames that were upright the
whole time.

**Datum for the record: a defect measured through a derived artifact needs one measurement of the
mechanism itself before it is funded.** M2.8.3 inferred non-upright decoding from feature values;
the decode was never compared against its own container. One pixel comparison refuses it.

## 3. The real defect: the normalisation is anisotropic, and its factor varies by asset

`export.py` divides `x` and `z` by `frame_w` and `y` by `frame_h` (three call sites: body, hand,
hand-only). Two consequences, both measured:

- **Within an asset**, angles are distorted because the map is not a similarity. Median **9.9°**,
  p75 17.3°, p95 26.5°, max 32.5° against the true image-plane angle over 40 upright assets /
  75 256 frames (M2.8.3 A09).
- **Across assets**, the distortion factor differs with the display aspect. Landscape scales
  (x, y) by (1/1920, 1/1080); portrait by (1/1080, 1/1920). A y-over-x ratio — `posture_symmetry` is
  literally one — differs by **(1920/1080)² = 3.16×** between the two populations, and a mixed ratio
  like `left_reach_norm` by the observed **1.73×**. Cohort aggregation compares assets, so this
  contaminates exactly the artifact M2.8.3 publishes.

`z` is identically 0.0 on every 2D landmark column, so no `z` claim rests on this.

## 4. Design decisions

**D01 — no rotation is applied on the corpus 2D decode path, and the reliance becomes an assertion.**
§2 refuses the repair. That path continues to take the backend's display-matrix handling, and P06/P07
pin it on synthetic fixtures so a future OpenCV change fails a test instead of silently rotating 38
assets. Scope is the decode path `open_capture` → `run.process_source` → detector input, **not the
repo**: `main.py:205-206` mirrors the frame before inference under its `flip` option, and
`measure/detect.py:198`, `measure/rigidity.py:168-172`, `measure/visual_offset.py:121-127` rotate
deliberately on an explicit `rotation_deg`. Those are correct and out of scope (A05).

**D02 — `open_capture` sets `CAP_PROP_ORIENTATION_AUTO = 1` explicitly.** Today's default already
does this, so behaviour is unchanged; the setting is what removes the dependency on a default.
`probe_container` already sets it, and its own docstring records why — *"OpenCV changed its default
across 4.10/4.11/4.12"*. A decode path resting on a default that provably moved between versions is
the same hazard class as M2.8.1's B1: **the default is the hazard, so the guard is the fix.**

**D03 — coordinates normalise by one scalar, `max(frame_w, frame_h)`.** One scalar makes the map a
similarity, so every image-plane angle and every distance ratio is preserved. `max` rather than
`frame_h` or `sqrt(w·h)` for two reasons: every coordinate stays inside `[0, 1]`, so the schema's
range contract is unbroken; and `max` is **invariant under a 90° orientation change**
(`max(1920, 1080) == max(1080, 1920)`), which is what makes the scale stable for the 7 assets that
change orientation mid-clip and for the portrait/landscape split alike.

**D04 — frame dimensions come from the decoded frame, not from capture properties.** `run.py` reads
`CAP_PROP_FRAME_WIDTH/HEIGHT` once before the loop and passes them to every row; `main.py` already
takes `frame.shape[:2]`. Measured today the two agree (§2 M2), and taking the frame's own shape is
what keeps them agreeing under a backend change or a mid-clip dimension change. The property read
stays for the banner and `total_frames`.

**D05 — the normalisation identity is published in the run report.** After this change an old and a
new 2D landmark CSV are shaped identically and mean different things, and 2D outputs deliberately
carry no artifact identity tags (`docs/technical/analysis.md:58` — a tag column would enter
`aggregate_per_video()`). So the identity is a module constant in `export.py`, echoed by
`corpus_run_2d.py` into `run_report.json` `configuration`. Judgment-bearing constant in `src/`, the
driver only orchestrates (M2.8.2 A12).

**D06 — the re-run publishes to a fresh `--out` tree and the pre-fix tree is left intact until the
new one validates.** Resume is keyed on `event_complete.json`; re-using the tree would credit
pre-fix events as complete. The pre-fix tree is deleted only after the new report reads all verdicts
true.

**D07 — `--tracking body`,** which is a superset of `hands-arms` (all 133 keypoints), so the 17
guarded trunk/posture columns populate and the published set becomes 92. The shared columns are
**not** expected byte-identical to the pre-fix run: `body` selects `BONE_SEGMENTS_WB_BODY` for the
bone smoother, so the smoothed keypoints differ. The re-run replaces; it does not extend.

**D08 — the angle unit token relaxes, and that is M2.8.3's amendment to make.** A09 assigned
`deg_image_plane_uncalibrated` because the coordinates were anisotropic. Under D03 the published
angle **is** the true image-plane angle, so the token becomes `deg_image_plane`. It stays qualified:
no lens-distortion correction is applied, and an image-plane angle is not an anatomical angle. This
contract records the change; M2.8.3 owns the amendment.

## 5. Invariant surfaces

| id | surface | what must not move without a predicate saying so |
| -- | ------- | ------------------------------------------------ |
| I1 | `src/pose_estimation/export.py` | the three normalisation sites + the identity constant |
| I2 | `src/pose_estimation/run.py` | the dimensions handed to `frame_to_rows` |
| I3 | `src/pose_estimation/video_io.py` | `open_capture`'s orientation posture; no rotation call |
| I4 | `scripts/corpus_run_2d.py` | the reported configuration, incl. the identity |
| I5 | `tests/goldens/2d_*` | six committed R goldens, byte-identical |
| I6 | the re-run tree | totality, dispositions, the 11 driver verdicts |

## 6. Predicates

Every predicate is decided by a committed test or a committed script MAIN reruns.

- **P01 isotropy.** For non-square `(frame_w, frame_h)` in both aspects, `frame_to_rows` divides
  `x`, `y` and `z` by one and the same scalar, and that scalar equals `max(frame_w, frame_h)`.
- **P02 aspect invariance.** One pixel-space geometry exported at 1920×1080 and at 1080×1920 yields
  identical normalised angles and identical distance ratios. This is the property the unit buys, so
  it is stated over angles and ratios rather than over coordinates. Tolerance is exact equality:
  both aspects divide by the same `max(w, h)`, so the two branches agree bitwise (A02).
- **P03 range.** Every exported `x`/`y` for a landmark inside the frame lies in `[0, 1]`, in both
  aspects. Non-vacuous: the case must assert it over a landmark at each frame corner.
- **P04 orientation-invariant scale.** The scale is unchanged when `(w, h)` transposes, so a
  mid-clip orientation change cannot move an asset's coordinate scale.
- **P05 dimensions from pixels.** `run.py` passes the decoded frame's own `shape[:2]` to
  `frame_to_rows`. Encoded against the shipped symbol, over a capture whose reported properties
  disagree with the frames it returns.
- **P06 backend applies the display matrix.** On synthetic fixtures declaring 0/90/180/270, the
  default decode equals the auto-off decode transformed by exactly the declared rotation, and the
  reported dimensions equal the decoded dimensions.
- **P07 end-to-end orientation.** The frame `run.process_source` hands the detector is in display
  orientation for all four classes. Catches a path-side rotation and a disabled `ORIENTATION_AUTO`
  alike — P06 pins the environment, P07 pins the corpus decode path. Scoped to that one entry point;
  the repo-wide form was dropped with D01's rescoping (A05).
- **P08 normalisation identity.** Three frozen literals (A05 table in §10): the symbol
  `pose_estimation.export.COORD_NORMALIZATION`, the report key `configuration.coord_normalization`,
  and the token `image-isotropic-maxdim`. The same case asserts `coord_scale(h, w) == max(w, h)` on a
  non-square pair, binding the token to behaviour so a wrong identity cannot satisfy it. A report
  alone distinguishes a pre-fix run from a post-fix run.
- **P09 all four sites.** Body, matched-hand, fallback-hand and hands-only paths route through one
  helper. Witnessed by a call-path spy — `export.coord_scale` monkeypatched to a sentinel divisor,
  every path's emitted coordinates required to reflect it (A07). Behavioural, so a duplicated formula
  fails; never a name grep.
- **P10 goldens unmoved.** The committed 2D R goldens — enumerated from `_DATASETS` at check time,
  never from a frozen count (A04) — are byte-identical after the change. They
  are driven by synthetic input CSVs, so `export.py` cannot reach them; a moved golden is an
  undeclared coupling.
- **P11 tracking mode.** A real event run under `--tracking body` yields `body_*` columns,
  `detect_tracking` reads `body`, and every trunk/posture column in the committed **`finite-capable`
  census** — the **14** columns measured in A03, 4 frame and 10 window — is finite at its own artifact
  grain. The literal 17 is void; all 17 must still be *present*, and the 3 sagittal cells must remain
  NA, which is what keeps the partition honest in both directions. Non-vacuous: a run that emits the
  finite-capable columns all-NA fails.
- **P12 re-run totality.** Manifest total over the canonical asset ids **derived from the validated
  registry at check time**, six frozen dispositions, key-set equality and key uniqueness, and the 11
  driver verdicts — set-equal to the keys `scripts/corpus_run_2d.py:418-432` owns, each true (A08).
- **P13 tree disjointness.** Both roots resolved (`Path.resolve()`), non-containment required in
  **both** directions, and the new root holds zero `event_complete.json` markers before the first
  attempt (A09). An operator/script obligation MAIN runs and records — no shipped symbol takes both
  roots.
- **P14 determinism tripwires.** `check_qualify_determinism.py` and
  `check_calibration_qc_determinism.py` regenerate green against the new `video_io.py` digest, each
  campaign's **complete control-id set** derived from its checker source at check time and required
  set-equal to the committed results, every expected rejection present, rc=0 (A10).
- **P15 gate identity.** `ruff check`, `ruff format --check`, `ty check` all rc=0; decisive suite in
  the primary tree. Collection is compared as a **node-id set difference** against the digest frozen
  at base 3d323c7, against the exact added node ids; rc=0 and 0 skipped are separate conjuncts (A11).
- **P16 the fix is demonstrated, not asserted.** Within-asset angle fidelity against pixel-space
  ground truth, on the 8-asset sample frozen in A12 before execution: isotropic max per-asset p95 of
  `abs(angle_normalised - angle_pixel)` **< 1e-6 deg**, and the same script's anisotropic
  recomputation on the same frames **>= 5 deg** median, so the case cannot pass vacuously. A
  committed script, credited by MAIN's rerun, and runnable **before** the long run is funded. The old
  cross-asset half is deleted as redundant with P02 (A12).

## 7. Negative-control seed

Each control names the predicate it must fire, and is graded against that predicate's text at
contract time — an unfireable control is a hole exactly where the contract claims coverage
(M2.8.3 A15).

| id | seed | must fire |
| -- | ---- | --------- |
| N1 | restore `x/frame_w`, `y/frame_h` | P01, P02 |
| N2 | scale by `min(w, h)` | P03 (values exceed 1) |
| N3 | scale `x` by `max`, `y` by `frame_h` | P01, P02 |
| N4 | `run.py` passes the property dimensions | P05 |
| N5 | `open_capture` sets `ORIENTATION_AUTO = 0` | P07 |
| N6 | add `cv2.rotate` on the decode path | P07 |
| N7 | drop the identity from the run report | P08 |
| N8 | hand path keeps the anisotropic scale | P09, P02 |
| N9 | duplicate one manifest asset, drop another | P12 via key-set equality AND uniqueness — the mutation moves both (A08) |
| N10 | round coordinates to 4 dp instead of 6 | nothing — symmetric mutation vs a differential predicate (A02) |

**Graded at contract time.** N1-N4 and N7-N9 each contradict a stated conjunct, so each fires. N5
and N6 are the pair the refutation exists to guard and both invert the display orientation P07
compares against. **N10 is the doubtful one**: 4 dp on a `[0, 1]` coordinate is ~1e-4 absolute,
which moves an angle by ~0.01° — below any tolerance P02 would sensibly carry, so as written N10
does **not** fire. Kept in the table with that verdict recorded, because a control that cannot fire
is a fact about the predicate's reach and belongs in the record rather than deleted from it.

## 8. Gate identity

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync ruff check
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync ruff format --check
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync ty check
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync pytest -q
```

Primary tree only; `analysis/*.R` cases need `renv/library`, and the gate never runs beside a decode
or inference sweep (`test_r_timebase_truth.py::test_c8_08` carries a subprocess timeout that CPU
contention alone blows).

## 9. Probe-corpus seed

1. Synthetic rotated fixtures at 0/90/180/270, built by stamping an ISO-14496-12 `tkhd` display
   matrix into a PyAV-muxed clip — PyAV cannot write one, and `ffmpeg` is absent from this
   container. Verified to reproduce the corpus behaviour exactly.
2. Landscape 1920×1080 and portrait 1080×1920 export of one pixel geometry.
3. Landmarks at all four frame corners and at the centre.
4. A capture whose reported properties disagree with its decoded frames.
5. One real event under `--tracking body`.
6. The 379-asset corpus under the corrected pipeline.

## 10. Amendments

**A01 — the normalisation has a second, un-predicated consumer: the 3D fusion un-normaliser.**
`multicam.fuse_session_outputs` (`multicam.py:644`) multiplied the read-back CSV coordinates by the
per-axis calibrated `resolution` vector. P10 correctly held the R 2D goldens decoupled, but no
predicate covered this site, so the contract's invariant-surface list was short by one.

Scope correction, not a correctness finding: export and fusion round-trip through each other, so the
old per-axis pair `(1/w, 1/h)` → `(w, h)` recovered true pixels just as the new scalar pair does.
The defect an unrepaired site would have caused is a BROKEN round-trip — normalised by `max(w, h)`,
un-normalised by `(w, h)` — putting the 2D observations into a stretched pixel space that the
calibrated intrinsics do not describe, and silently degrading every triangulation.

Ruling: the un-normaliser must invert `export.coord_scale` by construction, so it now calls that
function rather than restating the identity. `validation.py:809` reads the same CSVs but consumes
`np.isfinite(kps)` alone (`_camera_tracking`, `validation.py:871-874`) — magnitudes never reach a
measurement, so it needs no change.

**A02 — N10 cannot fire P02, and the freeze recorded the right verdict for the wrong reason.**
Reported by `test-m2u84` (batch 1, cce1bd5) as a P02 `fail`.

§7 graded N10 non-firing on magnitude: 4 dp ≈ 1e-4 moves an angle ~0.01°, "below any tolerance P02
would sensibly carry". That reasoning presumes P02 carries a magnitude tolerance at all. It does not
and must not: with `scale = max(frame_w, frame_h)`, the SAME pixel geometry exported at 1920×1080
and at 1080×1920 divides by 1920.0 in both branches, so the normalised coordinates are
bit-identical and P02's angles and ratios agree exactly. Tolerance = 0, by construction.

The real reason N10 cannot fire is structural, and it generalises: **P02 is a differential predicate
and N10 is a symmetric mutation.** Rounding to 4 dp perturbs both branches identically, so their
equality survives any mutation applied to both. No tolerance choice changes that — a magnitude
argument was never the operative one.

Rulings:
1. P02 stands as written, with tolerance now stated as exact equality rather than left open.
2. N10's `must fire` cell is wrong. P02's actual firing set is {N1, N3, N8} — each makes the divisor
   depend on the aspect, which is the only way to separate the two branches. N10 stays in the table
   with `must fire: nothing — symmetric mutation vs a differential predicate`, since a control that
   cannot fire is a fact about the predicate's reach (M2.8.3 A15) and this one now records a reusable
   rule for grading future negative controls.
3. The suite must not carry a red case asserting N10 fires. `test-m2u84` graded P02 `fail` against
   the contract text it was given, which was the correct call on that text; the amendment is the fix,
   and its P02 case re-grades to `pass` under the corrected §7.

**A03 — P11 is unsatisfiable as written: part of the 17-column target can never be finite from 2D.**
Reported by `test-m2u84` as CRITICAL; **verified independently by MAIN** at
`analysis/clinical_features.R:1038` — the 2D branch assigns
`result[["trunk_lean_sagittal_deg"]] <- NA_real_` under the comment "Out-of-plane: unmeasurable from
a single 2D view", and `WINDOW_BODY_METRICS` (`clinical_features.R:1081-1086`) carries
`trunk_lean_sagittal_mean` / `_sd`, which aggregate that column and are therefore NA too.

P11 demands "the 17 trunk/posture columns are finite on at least one window row". The sagittal
columns are structurally NA in every 2D run, so no `--tracking body` re-run can satisfy it. This is
the same defect class as M2.8.3's frozen literals: a cardinality asserted from a plan rather than
measured.

**The unit's deliverable survives** — `trunk_lean_deg`, `trunk_rotation_deg` and `posture_symmetry`
ARE computed on the 2D branch (`clinical_features.R:1033-1043`), so `--tracking body` still populates
the non-sagittal trunk/posture columns. Only the acceptance predicate is wrong.

Ruling: P11's literal 17 is void. It is replaced by a **measured** census, and the measurement is a
precondition of funding the long run rather than an outcome of it — run one real event under
`--tracking body`, partition the trunk/posture columns into `finite-capable` and `structurally-NA`,
commit that partition, and restate P11 over the `finite-capable` set alone with the structurally-NA
set named and justified. MAIN verified the sagittal case only; the rest of the 17 stay unaudited, so
the census is a measurement, not a subtraction of 2 from 17.

**A03 census, MEASURED** on one real event under `--tracking body` (3 cameras, 6 459 frame rows,
494 window rows), which is what A03 made a precondition of funding the run. It **confirms** the
derived expectation rather than discovering a different one:

| grain | columns | finite-capable | structurally-NA |
| ----- | ------- | -------------- | --------------- |
| frame | 5 | 4 | `trunk_lean_sagittal_deg` |
| window | 12 | 10 | `trunk_lean_sagittal_mean`, `trunk_lean_sagittal_sd` |
| **total** | **17** | **14** | **3** |

All 17 columns are **present** in the artifacts, so `--tracking body` does deliver the unit's
deliverable; 14 carry finite values and the 3 sagittal cells are NA by construction. P11 is
restated over the 14.

**A04 — P10's golden census is factually wrong: twelve 2D goldens, not six.**
Reported by `test-m2u84` as HIGH; **verified independently by MAIN** at
`tests/test_r_clinical_goldens.py:18-42` — `_DATASETS` enumerates four 2D datasets (`2d_idx`,
`2d_cumsum`, `2d_csv4dp`, `2d_drop`), each with three artifacts (`frame`, `window`, `group_qc`) =
**12**, plus the separate `world3d` entry. Ruling: P10 reads "the committed 2D R goldens are
byte-identical", with the count taken from `_DATASETS` at check time rather than frozen in prose. A
frozen count is a second thing to keep in sync and buys nothing the enumeration does not.

A01 also adds **P17** (extends I5): for one synthetic session, coordinates written by `frame_to_rows`
and read back through the fusion un-normaliser reproduce the source pixel geometry to within 1e-6,
at both a landscape and a portrait calibrated resolution. Acceptance check =
`tests/test_multicam.py::test_fuse_session_outputs_reconstructs_skeleton` extended with the portrait
resolution, plus a `command grep` proving no call site restates the divisor:
`rg -n 'resolution\"\]' src/pose_estimation/` returns no multiplication site.

A03 addendum — the 17 also spans **two artifact grains**, which is the second half of why it is
unsatisfiable: the frame artifact owns 5 columns (`trunk_lean_deg`, `trunk_lean_lateral_deg`,
`trunk_lean_sagittal_deg`, `trunk_rotation_deg`, `posture_symmetry`;
`analysis/clinical_features.R:1012,1034-1043`) and `WINDOW_BODY_METRICS` owns 12
(`clinical_features.R:1081-1086`). 5 + 12 = 17, so **no single window row can carry all 17** even
before the sagittal problem. Derived expectation for the measurement, to be confirmed rather than
discovered: `structurally-NA` = 3 cells (`trunk_lean_sagittal_deg` + its `_mean`/`_sd` aggregates),
`finite-capable` = 14. `trunk_lean_lateral_deg` is computed unconditionally at
`clinical_features.R:1012`, ahead of the 3D/2D branch, so it is finite-capable.

A04 addendum — the contract's `tests/goldens/2d_*` spelling matches **zero** tracked paths. The
goldens live under `tests/goldens/r_clinical/` (16 files: the 12 2D artifacts plus the `world3d`
set). P10 addresses them through `_DATASETS`, never through a path glob.

**A05 — P07 has no entry-point grain, and D01's repo-wide claim is already false.**
Reported as HIGH. Two entry points read decoded frames: `run.process_source` (`run.py:447`) and
`main.process_video` (`main.py:196`). A case over one can stay green while the other transforms.

MAIN went to check which entry points transform, and found D01 refuted on its own terms:
`main.py:205-206` applies `cv2.flip(frame, 1)` to the frame **before inference**, under the `flip`
option. `measure/detect.py:198`, `measure/rigidity.py:168-172` and `measure/visual_offset.py:121-127`
all call `np.rot90` deliberately, keyed on an explicit `rotation_deg`, as part of sync/offset
detection. So "no rotation applied by this repo" was never true; what is true is that **the corpus 2D
decode path applies none**.

Rulings:
1. **D01 is rescoped** to the corpus 2D decode path — `open_capture` → `run.process_source` →
   detector input. The MediaPipe display path's opt-in mirror and the `measure/` subsystem's
   orientation handling are deliberate, out of scope, and named here so a later reader does not read
   them as violations.
2. P07 quantifies over `run.process_source` for all four rotation classes, at the detector-input
   boundary. N5 and N6 fire there.
3. The repo-wide absence claim is **dropped, not converted into a structural check**. A grep over
   `src/` for rotation calls cannot distinguish the four legitimate sites above from an illegitimate
   one, so it would fire on correct code — a check that cannot separate the cases it exists to
   separate is worse than no check.

**A06 — P08 freezes no identity, so a wrong identity satisfies it.**
Reported as HIGH: neither symbol, report key, nor token is fixed, and an identity still denoting
width/height division would pass. Ruling — all three literals are frozen, and the token is bound to
behaviour so it cannot drift from what it names:

| slot | frozen value |
| ---- | ------------ |
| exported symbol | `pose_estimation.export.COORD_NORMALIZATION` |
| report key | `configuration.coord_normalization` in `run_report.json` |
| token | `image-isotropic-maxdim` |

Binding conjunct: the same case asserts `coord_scale(h, w) == max(w, h)` for a non-square pair, so
the token cannot survive a semantic change. N7 removes the report key and fires.

**A07 — P09's helper conjunct has no witness; four duplicated formulas would pass.**
Reported as MEDIUM. Behavioural driving proves shared *values*, not a shared *helper*, so structural
deduplication can regress silently. Ruling: witness helper identity behaviourally with a **call-path
spy** — monkeypatch `export.coord_scale` to return a sentinel divisor, drive the body, matched-hand,
fallback-hand and hands-only paths, and require every path's emitted coordinates to reflect the
sentinel. A duplicated formula ignores the patch and fails. This keeps P09 behavioural, drops no
conjunct, and needs no name grep. N8 fires on it.

**A08 — P12 freezes an input-owned census and an unnamed verdict set.**
Reported as HIGH. Rulings: (a) the canonical asset ids are derived from the validated registry at
check time, never from the literal 379; (b) the verdict key set is frozen exactly as
`scripts/corpus_run_2d.py:418-432` owns it — `manifest_total`, `every_event_complete`,
`artifacts_owned`, `group_disposition_published`, `partition_total`, `partition_disjoint`,
`group_qc_header_frozen`, `counters_classify_every_frame`, `stored_rate_equals_its_derivation`,
`generation_digest_unmoved`, `generation_marker_unmoved` (11), and membership is checked as set
equality so a substituted always-true key fails; (c) N9's `must fire` cell is corrected — the
duplicate-one/drop-one mutation changes the key set as well as uniqueness, so "uniqueness alone sees
it" is wrong; current validation happens to raise on uniqueness first, which is an ordering
accident, not the predicate's reach.

**A09 — P13's tree-disjointness relation is undefined in both directions.**
Reported as HIGH. Every correct re-run necessarily recreates the same event-relative
`event_complete.json` paths, so a naive path comparison rejects valid runs; and path inequality alone
misses a symlink aliasing the old tree. Ruling: P13 takes **both roots as resolved paths**
(`Path.resolve()`), requires non-containment **in both directions**, and requires the new root to
hold **zero** `event_complete.json` markers before the first attempt. No shipped symbol accepts both
roots, so this is an operator/script obligation, run and recorded by MAIN rather than encoded as a
pytest case.

**A10 — P14 names "both negative controls" without identifying any.**
Reported as MEDIUM. The qualification evidence carries 20 tamper rows and the calibration evidence
18; either can lose a row and stay non-empty and green. Ruling: P14 derives each campaign's
**complete control-id set** from its checker source at check time, requires set equality against the
committed results, requires every expected rejection, and requires rc=0 — with the recorded source
digests including `video_io.py`, which is what makes the regeneration non-trivial this unit.

**A11 — P15's collection delta has no baseline and no grain.**
Reported as MEDIUM. One parameterised function is four collected cases, so function count and node-id
count disagree, and the contract's recorded base predates the grading base, letting an implicit
commit-to-commit delta absorb unrelated cases. Ruling: freeze the **collected node-id set digest at
base 3d323c7**, freeze the exact added node ids, and compare the set difference. rc=0 and zero skips
stay separate conjuncts of §8 rather than folding into the delta.

**A12 — P16 is vacuous: "toward 1.0" admits any outcome, chosen after seeing it.**
Reported as CRITICAL, and the most consequential row — as written the unit has no way to demonstrate
its own fix, and the demonstration was the reason the long run is funded. Neither sample membership,
feature, estimator, divergence formula, pre-fix reference, direction handling, nor minimum
improvement is fixed. A single asset also cannot span both static aspects, so the sample grain was
incoherent.

Ruling: P16 is replaced by a **within-asset angle-fidelity measurement against pixel-space ground
truth**, which is exact, needs no population statistics, and — decisively — does not depend on the
corpus re-run, so it can be run *before* the long run is funded rather than after.

Pixel-space image-plane angle is the ground truth: a similarity normalisation preserves it exactly,
an anisotropic one does not. On a sample frozen **before execution**:

| slot | frozen value |
| ---- | ------------ |
| sample | 8 canonical assets — the first 4 landscape and first 4 portrait by sorted canonical id in the validated registry; ids committed to this contract before the script runs |
| observable | image-plane angle at the elbow (shoulder-elbow-wrist), both sides, every frame with all three keypoints finite |
| reference | the same angle computed from the decoded **pixel** coordinates |
| estimator | per-asset median and p95 of `abs(angle_normalised - angle_pixel)` in degrees |
| acceptance | isotropic: max over the sample of the per-asset p95 **< 1e-6 deg**; anisotropic, recomputed by the same script on the same frames: median **>= 5 deg** |

The anisotropic conjunct is asserted, not merely reported: without it a normalisation that happened
to be near-isotropic would pass vacuously. Prior measurement on this corpus put the anisotropic
median at 9.9 deg and p95 at 26.5 deg, so the 5 deg floor has ~2x headroom.

The cross-asset half of the old P16 is **deleted as redundant**: the aspect-driven divergence is
already pinned exactly and deterministically by P02, which compares the same geometry across both
aspects. A population median over real assets would restate that property with sampling noise added
and proof subtracted.

**A13 — A12's "ids committed to this contract" breaches the redaction rule; the sample is pinned by
digest instead.** Found while implementing A12. Canonical asset ids are capture identifiers, which
published artifacts must not carry, so writing the eight members into this file would trade one
defect for another.

Ruling: pre-registration is achieved **without publishing membership**. The selection rule is a pure
function of the validated registry — first `PER_ASPECT` landscape and portrait assets by sorted
`asset_id` among `disposition == canonical` — so the sample is reproducible by anyone with the
registry, and `scripts/check_isotropy_angle_fidelity.py` publishes `sha256` over the sorted ids.
Moving the sample after seeing a result moves the digest, which is the property A12 actually wanted.
Per-asset rows in the evidence file are keyed by ordinal, never by id or placement.

Second ruling — **the evidence file is the durable artifact, not the script.** The script reads the
PRE-FIX tree, which D06 deletes once the re-run validates, so it is replayable only while that tree
exists. `tests/isotropy_angle_fidelity_results.json` is committed and P16 is credited from it; the
script must **not** join the standing gate, where it would fail the moment D06 is executed.

**A14 — the delivered P05 case contradicted the delivered P01 case; P01 was right.**
Found when the suite ran against the implementation. `test_p05_...` asserted the exported pair
`(0.5, 0.5)` for a keypoint at pixel `(32, 24)` in a 64x48 frame. Under D03 both axes divide by
`max(64, 48) = 64`, so the correct pair is `(0.5, 0.375)`; `(0.5, 0.5)` is reachable only by per-axis
division, which P01's own case refuses. The two cases could not both pass.

Ruling: the suite is corrected, not the contract — P01 states D03 and D03 is the unit's decision. The
case keeps its discriminating power, since the capture's properties (640x480) would yield
`(0.05, 0.0375)`; it stays RED at 3d323c7 for both conjuncts. Recorded because a diff-blind suite is
trusted for its independence, and independence is exactly what lets two of its cases disagree —
grading each predicate alone never compares them.

**A15 — the pilot found a latent code defect that no full-corpus run can expose.**
`corpus_run_2d.py` emitted `throughput.sample = "partial"` whenever fewer events were measured than
the corpus holds, but the redaction allowlist carried only `"corpus"`. `_assert_redacted` therefore
aborted the run **before the report was written**, so every partial and every resumed invocation
lost its report while every full-corpus invocation passed. Pre-existing, unrelated to the
normalisation change, and invisible to the existing suite.

Rulings:
1. Fixed at the source of the drift, not at the symptom: `THROUGHPUT_PROVENANCE`/`THROUGHPUT_FULL`/
   `THROUGHPUT_PARTIAL` are named constants gathered into `THROUGHPUT_LABELS`, which both the emitter
   and the allowlist read. A future third label cannot reach one and miss the other.
2. The allowlist moved out of `main()` into `redaction_allowlist(args, placed_assets, codes)` so the
   invariant is reachable without a corpus run — being unreachable is why the gap survived.
   `tests/test_corpus_run_2d.py::test_report_allowlist_covers_every_throughput_label` ranges over
   `THROUGHPUT_LABELS` and asserts the set is non-empty first.
3. `pilot._assert_redacted` now names the **JSON path** of the offending string, never the string:
   the guard exists because that string may be a subject token, so quoting it would be the leak it
   prevents. Diagnosing this defect cost a full 17-minute event before the path was reported and
   nothing after it.

**Datum: the pilot is not a formality.** A03 and A12 were rewritten to run before the long run was
funded; that same discipline surfaced a defect which would have destroyed the 8.7 h run's report at
the very end, on a code path the 1 642-case suite never touched.

**Result, recorded** (registry census: 379 canonical = 351 landscape + 28 portrait, 0 square):

| quantity | measured | bound | verdict |
| -------- | -------- | ----- | ------- |
| isotropic p95 max | **3.695e-13 deg** | < 1e-6 | pass |
| anisotropic pooled median | **8.3567 deg** | >= 5.0 | pass |
| sample spans both aspects | 4 + 4 | 2 aspects | pass |

The isotropic figure is seven orders of magnitude inside its bound — the residual is float
arithmetic, which is what "the map is a similarity" predicts exactly. The anisotropic 8.36 deg is an
independent sample's echo of the 9.9 deg corpus median in §3, and it is the reason the unit exists.

## 11. Verdict table

Gate at the implementation commit: `ruff check` rc=0, `ruff format --check` rc=0, `ty check` "All
checks passed!", `pytest -q` **1642 passed, 0 failed, 0 skipped** in the primary tree (1009 s), run
after the pilot's decode sweep had exited. Rerun at the close commit, after the `--tracking body`
default flip and the P16 pre-fix-tree guard: all four green, **1642 passed, 0 failed, 0 skipped**
(1088 s) — the node-id set is unmoved, so P15 stands without re-derivation.

| id | verdict | evidence |
| -- | ------- | -------- |
| P01 | pass | `test_p01_body_coordinates_use_one_max_dimension_scalar` — RED at 3d323c7, green after |
| P02 | pass | `test_p02_angles_and_distance_ratios_are_aspect_invariant` — RED at 3d323c7; tolerance exact per A02 |
| P03 | pass | `test_p03_frame_corners_remain_inside_unit_range` — green both sides, the preservation side |
| P04 | pass | `test_p04_coordinate_scale_survives_dimension_transpose` — RED at 3d323c7 |
| P05 | pass | `test_p05_run_exports_using_decoded_frame_dimensions` — RED at 3d323c7; expectation corrected by A14 |
| P06 | pass | `test_p06_backend_applies_declared_display_matrix[0,90,180,270]` — real `tkhd` matrices |
| P07 | pass | `test_p07_corpus_pipeline_receives_display_oriented_frames[0,90,180,270]`, scoped per A05 |
| P08 | pass | `test_p08_normalisation_identity_is_frozen_and_bound_to_its_behaviour`; report key confirmed on a real `run_report.json` (`configuration.coord_normalization = image-isotropic-maxdim`) |
| P09 | pass | `test_p09_body_matched_hand_and_hand_only_paths_are_isotropic` — RED at 3d323c7 |
| P10 | pass | `test_p10_all_source_enumerated_2d_goldens_are_byte_identical` — 12 goldens via `_DATASETS` (A04) |
| P11 | pass | A03 census on one real `--tracking body` event: 17 present, **14 finite-capable**, 3 sagittal NA |
| P12 | pass | full re-run report: manifest 379 rows, ids **set-equal to the 379 registry-derived canonical ids** and unique, census keys + dispositions inside the frozen six and summing to 379, **11 verdict keys set-equal to the `verdicts = {…}` literal parsed out of `corpus_run_2d.py` by AST at check time, every one true**; 193/193 events complete, 0 attempts failed, `throughput.sample = corpus` |
| P13 | pass | `output/corpus-2d` and `output/corpus-2d-v2` resolve to distinct paths, non-containment holds **both directions**, new root held 0 `event_complete.json` at launch and 193 at close, old root still 193 |
| P14 | pass | both campaigns regenerated: qualify **40 sweeps / 19 tamper classes**, calibration **21 sweeps / 18 tampers**, 0 failures, every control still rejecting |
| P15 | pass | node-id set difference vs 3d323c7: 1621 → 1642. 1 removed (`test_fuse_session_outputs_reconstructs_skeleton`, replaced by its parametrised pair), 22 added, every one traceable to a ruled predicate. Baseline listing digest `8fd08fa4247d03d2` |
| P16 | pass | `scripts/check_isotropy_angle_fidelity.py` rc=0 — isotropic p95 max **3.695e-13 deg** (bound 1e-6), anisotropic pooled median **8.3567 deg** (floor 5.0), sample digest `a27e4f32abdc4ac3`; evidence `tests/isotropy_angle_fidelity_results.json` |
| P17 | pass | `test_fuse_session_outputs_reconstructs_skeleton[resolution0,resolution1]` — landscape and portrait calibrated resolutions |

P12 and P13 are the only rows the corpus re-run owns; every other predicate closed before it was
funded, which is what A03 and A12 were rewritten to achieve.

### The corrected corpus run

`--tracking body --out output/corpus-2d-v2`, detached, NPU: **193/193 events, 379/379 assets `ok`,
0 failures, 337 090 frames, 12.251 fps incl. startup, 7.828 h** (27 515 s decode + 666 s clinical,
28 181 s wall). Report `configuration` publishes `tracking = body`,
`coord_normalization = image-isotropic-maxdim`, `pose_device = NPU`, `generator_version = v2`.
Prior run: 8.70 h at 10.99 fps under the gate prefix's silent CPU fallback.

Two `data`-tier live spot-checks on the shipped bytes, each with its positive control:

- **A03's census reproduces corpus-wide.** 12 random events × 3 cameras, 17 200 frame rows and
  1 115 window rows, columns taken from the R source rather than from a keyword net
  (`clinical_features.R:1033-1043` + `WINDOW_BODY_METRICS` at `:1081-1086`): **17 present, 0 missing
  from any header, 14 finite-capable, 3 structurally NA, and the NA set is exactly the sagittal
  set.** A keyword net misses `compensatory_pattern_index` and undercounts the window grain 11/12 —
  the column list has to come from the emitter.
- **The isotropic bound holds on the shipped corpus and fires on the pre-fix one.** No in-frame
  landmark may exceed `x = w/max(w,h)` or `y = h/max(w,h)`; per-axis normalisation puts both bounds
  at 1.0. Over 30 landscape + 28 portrait assets, filtered to `vis > 0.8`: pre-fix breach rate
  **29.82 % portrait / 39.83 % landscape**, corrected **0.0305 % / 0.0639 %**, the residual being
  genuinely off-frame joints that `--tracking body` adds and that still carry high confidence.
  Cross-tree min/max comparison is *not* a valid instrument here — the pre-fix tree is `hands-arms`
  (54 landmark columns) against the corrected tree's `body` (75), so the landmark sets differ and
  the ranges are not comparable; the bound test is per-tree and needs no correspondence.

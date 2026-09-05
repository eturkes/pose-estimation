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

**D01 — no rotation is applied by this repo, and the reliance becomes an assertion.** §2 refuses the
repair. The pipeline continues to take the backend's display-matrix handling, and P06/P07 pin it on
synthetic fixtures so a future OpenCV change fails a test instead of silently rotating 38 assets.

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
- **P07 end-to-end orientation.** A frame delivered by `open_capture` is in display orientation for
  all four classes. Catches a repo-side rotation and a disabled `ORIENTATION_AUTO` alike — P06 pins
  the environment, P07 pins the pipeline.
- **P08 normalisation identity.** `export` exports the identity constant; `run_report.json`
  `configuration` carries its value; a report alone distinguishes a pre-fix run from a post-fix run.
- **P09 all three sites.** Body, hand and hand-only paths use one scale helper. Encoded by driving
  each path, never by grepping for a name.
- **P10 goldens unmoved.** The committed 2D R goldens — enumerated from `_DATASETS` at check time,
  never from a frozen count (A04) — are byte-identical after the change. They
  are driven by synthetic input CSVs, so `export.py` cannot reach them; a moved golden is an
  undeclared coupling.
- **P11 tracking mode.** A real event run under `--tracking body` yields `body_*` columns,
  `detect_tracking` reads `body`, and every trunk/posture column in the committed **`finite-capable`
  census** is finite on at least one window row. The literal 17 is void and the census is measured
  before the long run is funded (A03); the sagittal columns are structurally NA from a single 2D view
  and belong to the `structurally-NA` partition. Non-vacuous: a run that emits the finite-capable
  columns all-NA fails.
- **P12 re-run totality.** Manifest total over the 379 canonical assets, six frozen dispositions,
  key-set equality and key uniqueness, and all 11 driver verdicts true.
- **P13 tree disjointness.** The re-run's `--out` is not the pre-fix tree and shares no
  `event_complete.json` with it.
- **P14 determinism tripwires.** `check_qualify_determinism.py` and
  `check_calibration_qc_determinism.py` regenerate green against the new `video_io.py` digest, both
  negative controls still firing.
- **P15 gate identity.** `ruff check`, `ruff format --check`, `ty check` all rc=0; decisive suite in
  the primary tree, 0 skipped, collection moved by exactly the new cases.
- **P16 the fix is demonstrated, not asserted.** On one fixed asset sample spanning both aspects,
  the portrait-vs-landscape divergence of a scale-invariant feature falls from its pre-fix value
  toward 1.0. A committed script, credited by MAIN's rerun.

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
| N9 | duplicate one manifest asset, drop another | P12 (uniqueness alone sees it) |
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

**A04 — P10's golden census is factually wrong: twelve 2D goldens, not six.**
Reported by `test-m2u84` as HIGH; **verified independently by MAIN** at
`tests/test_r_clinical_goldens.py:18-42` — `_DATASETS` enumerates four 2D datasets (`2d_idx`,
`2d_cumsum`, `2d_csv4dp`, `2d_drop`), each with three artifacts (`frame`, `window`, `group_qc`) =
**12**, plus the separate `world3d` entry. Ruling: P10 reads "the committed 2D R goldens are
byte-identical", with the count taken from `_DATASETS` at check time rather than frozen in prose. A
frozen count is a second thing to keep in sync and buys nothing the enumeration does not.

New predicate **P17** (extends I5): for one synthetic session, coordinates written by `frame_to_rows`
and read back through the fusion un-normaliser reproduce the source pixel geometry to within 1e-6,
at both a landscape and a portrait calibrated resolution. Acceptance check =
`tests/test_multicam.py::test_fuse_session_outputs_reconstructs_skeleton` extended with the portrait
resolution, plus a `command grep` proving no call site restates the divisor:
`rg -n 'resolution\"\]' src/pose_estimation/` returns no multiplication site.

## 11. Verdict table

*(filled at close)*

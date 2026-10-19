# Contract — M4.5a wrist-motion bouts + per-bout timing and smoothness (R producer)

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
Seam to M4.5b (cohort export) = `BOUT_SCHEMA` (§1), one immutable value.
Evidence → `.scratch/agents/researcher-5.md` (R01-R09); pre-code review → `consultant-3` (§8).

Purpose: SPARC and normalized jerk are defined on ONE discrete movement's speed profile
(Balasubramanian 2015: "smoothness measures cannot be used on an entire rhythmic movement"); the
shipped window `*_wrist_sal` / `*_normalized_jerk` run over fixed 1 s windows that mix movement,
rest and partial movements, so their value tracks how many components a window straddles.
Replace them with per-bout measures and add timing measures (SCI core-set constructs: movement
time, relative time to peak speed, speed peaks, path ratio), scale-free or in seconds
(metric-scale ruling holds).

**Estimand** = features conditional on detected, uncensored, projected 2D wrist-motion bouts.
Bouts are algorithmic intervals, not recovered reaches, grasps or clinically validated movement
units: a pause may divide one action, and the cohort median describes retained bouts, not total
task performance. Semantic action recovery is not identifiable from wrist speed alone and is not
an acceptance claim.

## 1. Artifacts

`analysis/clinical_features.R`, 2D inputs only (frame / window / phase / 3D outputs stay
byte-identical):

- `<stem>_clinical_bouts.csv`, written on every successful 2D run (header-only when no bout —
  proof of execution, never a claim of "no movement"). `BOUT_SCHEMA` (one R constant; writer
  and every reader check derive from it) = `video, person_idx, side, bout_idx, start_frame,
  end_frame, start_sec, end_sec, censored`, then for `left` then `right`: `{side}_bout_duration,
  {side}_bout_sparc, {side}_bout_normalized_jerk, {side}_bout_peak_time_share,
  {side}_bout_speed_peaks, {side}_bout_path_ratio, {side}_bout_dwell` (23 columns). One row per
  bout; `side` ∈ {left, right}; `bout_idx` gapless 1.. per person × side; the other side's 7
  columns NA (wide form for the cohort's per-column reader).
- `<stem>_clinical_bout_qc.csv`, always written: one row per (video, person_idx, side):
  `n_bouts, n_censored, disposition` ∈ {`bouts`, `only_censored`, `no_bout`, `no_usable_wrist`,
  `invalid_timebase`} (`BOUT_DISPOSITIONS`, frozen) + `grid_evidence` sample/gap counts.
- Feature definitions (finite values only; NA where §2 D04 or degeneracy applies):

| column | definition | range |
| --- | --- | --- |
| `bout_duration` | `end_sec - start_sec` (detected bounds) | > 0 s |
| `bout_sparc` | shipped `spectral_arc_length` (v3: fc 10 Hz, amp 0.05, pad 4) on the bout's SPARC support (D05) | <= 0 |
| `bout_normalized_jerk` | shipped `normalized_jerk` over the bout path | >= 0 |
| `bout_peak_time_share` | `(t_peak - start_sec) / duration`, `t_peak` = midpoint of the highest plateau of detection speed (earliest on ties) | [0, 1] |
| `bout_speed_peaks` | maxima of detection speed (plateaus collapsed) with prominence >= `BOUT_PEAK_PROMINENCE` × bout peak, >= `BOUT_PEAK_SPACING_SEC` apart | integer >= 1 |
| `bout_path_ratio` | path / chord over the bout (`movement_efficiency`); NA on zero chord | >= 1 |
| `bout_dwell` | this bout's end to the same side's next uncensored bout start inside one continuity segment; else NA | >= 0 s |

## 2. Design decisions

**D01 — grid first.** Timestamps + frame indices go on the validated nominal grid
(`trajectory_grid_status`) before any speed: absent rows, non-finite coordinates and failed
confidence gates are missing slots, kept as an observation mask. An invalid timebase yields no
bout and disposition `invalid_timebase`. The series splits at missing runs longer than
`BOUT_MAX_FILL` slots into continuity segments; nothing is filtered or merged across a split.

**D02 — detection** (`detect_bouts`): interior holes <= `BOUT_MAX_FILL` slots linearly
interpolated for detection, speed-based SPARC, timing and peak counts only; first-difference
speed × fs; running median `BOUT_MEDIAN_K` slots per segment = detection speed; threshold =
`BOUT_THRESHOLD` × the `BOUT_NORMALISER_Q` quantile of detection speed over the person × side's
observed slots (robust to outlier episodes, R05); active runs merged across gaps <=
`BOUT_MERGE_SEC` within a segment; runs shorter than `BOUT_MIN_SEC` dropped. A bout touching a
segment boundary (split or series edge) is censored: bounds published, every feature NA.

**D03 — NA, never bridged.** No metric drops missing slots and treats survivors as adjacent:
`bout_normalized_jerk` and `bout_path_ratio` are NA whenever the bout path holds a missing slot;
SPARC, timing and peaks read the interpolated speed (D02).

**D04 — qualification freezes the parameters BEFORE any corpus bout is read.** Logic in shipped
code: `detect_bouts` (R) + `src/pose_estimation/bout_qualification.py` (generator, matching,
scoring); `scripts/qualify_bouts.py` orchestrates. Generator code, seeds and distributions
frozen before any candidate score: minimum-jerk point-to-point segments (amplitude, direction,
duration 0.4-2.5 s, 1-3 overlapping submovements), within-action pauses, separate actions with
identical pauses, unequal-amplitude movements, backtracking (out-and-back), slow drifts, outlier
episodes, missing runs (explicit NA and omitted rows), white + 1/f position noise at the D06
level (×0.5, ×1, ×2), rates 25 / 30 / 60 Hz; a development bank (selection) and an untouched
qualification bank (report). Matching = one-to-one maximum-cardinality, then minimum total
boundary error; tolerance = max(0.1 s, 0.1 × true duration). Reported per duration / rate /
noise stratum: precision, recall, F1, fragmentation, merging, uncensored retention, feature
error; unmatched + censored true bouts stay in detection denominators; zero matches never earn
a finite feature error. Grid: `BOUT_THRESHOLD` {0.05, 0.1, 0.15, 0.2} × `BOUT_NORMALISER_Q`
{0.95, 0.99, 1.0} × `BOUT_MIN_SEC` {0.2, 0.3, 0.4} × `BOUT_MERGE_SEC` {0.1, 0.2, 0.3} × SPARC
support (D05) {bounds, extended}; `BOUT_MEDIAN_K` 5, `BOUT_MAX_FILL` 2 fixed. Admissible cell =
exact detection on the noise-free supported domain + zero detections on stationary controls.
Rank admissible cells lexicographically: development F1, then median |SPARC error| on matched
bouts, then the enumeration order above. `BOUT_PEAK_PROMINENCE` {0.1, 0.2, 0.3} ranked the same
way against the observable prominence-qualified maxima of the known resultant speed (never the
latent submovement count); `BOUT_PEAK_SPACING_SEC` = 0.15 (R04). The chosen cell + every
stratum on the qualification bank land in §8 before the corpus run; a failed qualification
changes the method or domain by amendment, never the grading rule after results.

**D05 — SPARC support** ∈ {`bounds` = the detected bout; `extended` = bounds extended on each
side to the nearest local minimum of detection speed inside the continuity segment}, on
unsmoothed interval speed; picked by D04 (consultant-3 probe: cropping a minimum-jerk profile at
a 0.1-0.2 threshold moves SPARC by 0.12-0.17).

**D06 — noise level = a residual proxy, not identified tracker noise.** Per eligible asset ×
side: confidence-masked, camera-stabilized wrist on its nominal grid; centred 7-slot quadratic
Savitzky-Golay over fully observed stencils; per-coordinate residual MAD / 0.67448975 over
centres whose SG speed is in the asset-side's lowest 20 %, coordinates combined by RMS;
eligibility >= 30 centres (fewer → excluded, counted). Aggregate eligible / excluded counts +
median / p90 by side only, no identifiers (`scripts/qualify_bouts.py measure-noise`). The whole
measurement, selection included, is calibrated on known white + correlated noise at each rate;
generator amplitudes come through that measured response.

**D07 — window smoothness columns stay in the window artifact** (3D goldens, legacy R consumers)
and leave the cohort export in M4.5b; `segment_movements` keeps its thresholds (phases are not
exported; merging the two segmenters = a `.agent/deferred.md` row).

**D08 — provenance**: `run_report.json` `configuration.analysis_sha256` = sha256 of
`analysis/clinical_features.R` (bout constants live there), so a corpus generation names the
bout method it ran.

## 3. Predicates

- **P01** Known dynamics on named fixtures (tolerances fixed after the D05 convention passes
  D04): a single minimum-jerk bout → one bout, `speed_peaks` 1, `path_ratio` 1 (straight),
  `peak_time_share` within the D04-qualified tolerance of 0.5, SPARC within the qualified
  tolerance of the full noise-free profile; two minimum-jerk submovements overlapping 40 % →
  one bout, `speed_peaks` 2, SPARC below the single bout's; a plateau peak → midpoint timing.
- **P02** SPARC kernel oracle: the author's example (`t = seq(-1, 0.99, 0.01)`, `exp(-5 t^2)`,
  fs 100, defaults) = -1.41403 (4 d.p.).
- **P03** Gaps: a hole > `BOUT_MAX_FILL` inside motion → censored fragments (features NA) or,
  below `BOUT_MIN_SEC`, dropped; a 1- or 2-slot hole → one uncensored bout whose NJ + path ratio
  are NA and whose SPARC / timing are finite; explicit-NA and omitted-row forms of one gap give
  identical output; `dwell` NA across a split; leading / trailing holes censor.
- **P04** Robust normaliser: an outlier episode that survives the 5-slot median, occupies less
  than `1 - BOUT_NORMALISER_Q` of the speed population and sits apart from genuine bouts leaves
  every genuine bout's bounds + count unchanged; the max-normalised mutant (`Q = 1.0`) loses a
  genuine bout on the same fixture (else the seed is invalid).
- **P05** Schema: header = `BOUT_SCHEMA` exactly, header-only on a still series, stale rows
  cleared on rerun; `side` / `censored` values; other-side columns NA; `bout_idx` gapless;
  §1 ranges on finite values; censored rows all-NA features; `bout_qc` one row per person ×
  side with a `BOUT_DISPOSITIONS` value; directory mode never rescans either artifact.
- **P06** Scale + translation: (x, y) × 3 + a constant shift → identical bounds, features equal
  to 1e-9 (non-degenerate domain, before rounding).
- **P07** Rate: one continuous trajectory sampled at 25 / 30 / 60 Hz with stated sample phases →
  duration within one frame of the slowest rate, peak share + SPARC within the D04-qualified
  tolerances; an unsupported duration × rate emits NA with its reason.
- **P08** D04 qualification reruns byte-identically from committed state to the same cell; an
  all-miss detector, an over-fragmenter and a development-bank lookup each fail it.
- **P09** `run_report.json` `configuration.analysis_sha256` equals the R file's digest.

## 6. Negative controls

- NC1 `BOUT_NORMALISER_Q = 1.0` reds P04 (on the P04 fixture).
- NC2 NA-as-zero speed (the shipped `segment_movements` treatment) reds P03.
- NC3 SPARC over position instead of speed reds P01.
- NC4 dropping missing slots before NJ reds P03.

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01-P09 | open | |

## 8. Amendments

- **A00 (pre-code review, `consultant-3`).** C01-C05 + C10 (producer half) accepted and written
  into §§1-3 above: conditional estimand; grid-first gaps with NJ / path ratio NA on holes;
  dev + qualification banks, admissibility, lexicographic ranking, observable-peak targets;
  residual-proxy noise with calibration; plateau midpoint; outlier seed that survives the
  median; `BOUT_SCHEMA`, `bout_qc` dispositions, provenance. Measured by the consultant's probe
  (`.scratch/consultant-3/probe.log`): Gaussian oracle -1.414031; a 0.4 s minimum-jerk bout at
  30 Hz → first-argmax share 0.333, cropped SPARC error 0.135 — hence D05 and qualified
  tolerances.

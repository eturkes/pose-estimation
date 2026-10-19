# Contract — M4.4 locked confirmation (decision table frozen before the user's labels are read)

Tier `kernel`. Base = the commit carrying this file. Frozen at commit: nothing under
`.scratch/bench/user/labels.jsonl` has been read, scored or summarised by any agent before it.
Amendments append to §8 and may not move a margin, an endpoint or the adoption rule once any
statistic over the user's labels has been computed.

## 1. Inputs

- **Confirmation set** = the user's clicked frames (`/label`, `.scratch/bench/user/labels.jsonl`;
  96 frames = 16 subjects × 3 views × 2, seed 20261020, outside the pilot + watch set). One use:
  no parameter, threshold or candidate is tuned on it, before or after.
- **Candidates** = complete production configurations (pose arm + calibration + thresholds +
  detector/identity policy) that M4.2's finalist rule kept AND M4.3's pilot evaluation passed;
  each named and digested (`pose_config.json` fields) in an amendment before scoring.
- **Baseline** = the shipped configuration (`rtmw-l` = RTMW-m 256, M2.10 settings + M4.6 fixes).
- **Predictions** = `bench.replay` of each configuration over a window ending at each confirmation
  frame, with the pre-roll M4.1 D05(e) measured; the exported row at that frame is scored.

## 2. Endpoints (per frame, then per subject = mean over that subject's frames)

- **E1 hand localisation** — over user joints `hand.*` with state `vis`: median of error / user
  hand extent diagonal (bounding box of that side's visible hand points; sides with < 2 visible
  hand points excluded). A joint the configuration does not export (hand absent, score 0) counts
  error = 1.0 (a miss, never a silent drop).
- **E2 arm localisation** — shoulder + elbow + wrist with state `vis`: error / user upper-arm
  length (shoulder→elbow of the same side, both `vis`; else the frame's other side; else excluded).
- **E3 hand presence agreement** — per side: exported hand present ⇔ user hand has >= 1 `vis`
  hand joint; share agreeing (sides whose user joints are all `?` excluded).
- **E4 side swaps** — per frame with both user wrists `vis`: exported left wrist nearer the user's
  right wrist than the user's left (and vice versa); share of such frames.

## 3. Decision table (each candidate C vs the baseline B; paired by subject; 95 % percentile
interval from 10 000 subject-cluster bootstrap resamples, seed 20261021)

| state | rule |
| --- | --- |
| non-inferior | upper(E1_C − E1_B) <= +0.02 AND upper(E2_C − E2_B) <= +0.02 AND lower(E3_C − E3_B) >= −0.05 AND upper(E4_C − E4_B) <= +0.02 |
| superior (secondary) | non-inferior AND upper(E1_C − E1_B) < 0 |
| regression | any non-inferiority bound breached with its point estimate on the wrong side |
| inconclusive | neither non-inferior nor regression |

**Adoption.** Among candidates that are non-inferior, adopt the one ranked first by M4.2's public
hand evidence (COCO-WholeBody mean hand AP; tie within 0.005 → DexYCB hand PCK@0.1; tie → lower
per-frame wall). No candidate non-inferior → the baseline stays, every candidate's state is
reported, and the public-vs-in-domain conflict returns to the user. The rule is order-invariant:
every candidate is compared to the baseline alone, never to the latest survivor.

Margins are policy, not literature constants (researcher-4 M08): 0.02 of hand extent ≈ 2 px on
a 100-px hand; 0.05 presence agreement ≈ 5 of 96 frames' sides; stated here before any data.

## 4. Predicates

- **P01** scorer on a synthetic label + prediction table reproduces E1-E4 exactly, incl. the
  miss-as-1.0 rule, the extent and upper-arm normalisers, exclusions.
- **P02** bootstrap: subject-cluster resampling (not frame), fixed seed reproducible, percentile
  bounds; a constant difference yields a zero-width interval at that value.
- **P03** decision table over constructed interval tuples hits every state; adoption picks by the
  public rank among non-inferior candidates and keeps the baseline when none is.
- **P04** the scorer refuses to run while this contract's amendment naming the candidates is
  absent (no candidate list → no score).

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01-P04 | open | |

## 8. Amendments

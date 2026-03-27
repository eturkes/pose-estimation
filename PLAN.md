# Clinical Feature Derivation Plan

This plan extends the pose estimation pipeline with clinical
analysis tools for spinal cord injury (SCI) rehabilitation.
Each section is self-contained and intended to be completed in
a fresh coding agent session.

Mark a section `[x]` when complete.

---

## Section 1 — Derive Clinical Features from Landmark CSVs

**Status:** [x] complete

**Goal:** Create `analysis/clinical_features.R` that reads the
landmark CSVs produced by `export.py` and computes clinically
meaningful kinematic features per frame (or per sliding window),
writing a `*_clinical.csv` alongside each input.

**Context:**
- Landmark CSVs have one row per person per frame with
  normalised (0–1) x/y/z coordinates.  Column schema is in
  `export.py` (`make_csv_header`).
- `hands-arms` mode: 12 arm keypoints (prefix `arm_`) + 2 × 21
  hand keypoints.  `body` mode: 33 body keypoints (prefix
  `body_`) + hands.
- Missing hand data is blank (NA when read).
- Existing `analysis/features.R` handles data loading, NA
  imputation, and tracking-mode detection — reuse its helpers.

**Features to compute (per side where applicable):**

1. **Joint angles** (degrees):
   - Elbow flexion: shoulder → elbow → wrist.
   - Wrist deviation: elbow → wrist → middle-finger base
     (hands-arms) or index (body).
   - Finger spread: angle between index-tip and pinky-tip
     vectors originating at wrist (from hand landmarks 8 and 20,
     relative to landmark 0).
2. **Reach distance:**
   - Euclidean distance from shoulder to wrist (normalised
     coords), each side.
   - Normalised by shoulder width (left-shoulder to
     right-shoulder distance) to account for camera distance.
3. **Grasp aperture:**
   - Distance between thumb tip (hand landmark 4) and index
     fingertip (hand landmark 8).
   - Distance between thumb tip (hand landmark 4) and pinky
     tip (hand landmark 20).
4. **Movement smoothness** (per sliding window, e.g. 1-second):
   - Spectral arc length (SAL) of the wrist velocity profile
     (see Balasubramanian et al. 2012/2015).  Implement in R
     using `stats::fft`.
   - Mean and peak wrist velocity within the window.
5. **Frame-level movement magnitude:**
   - Frame-to-frame wrist displacement (jitter proxy).
   - Frame-to-frame fingertip displacement (hand activity).

**Output schema:**
- Metadata: `video`, `frame_idx`, `timestamp_sec`, `person_idx`.
- One column per feature, named descriptively
  (e.g. `left_elbow_angle_deg`, `right_reach_norm`,
  `left_grasp_aperture_thumb_index`).
- Window-level features (SAL, velocity) can repeat their value
  for every frame in the window, or use one row per window with
  start/end timestamps — choose the one-row-per-window approach
  with `window_start_sec` and `window_end_sec` columns, written
  to a separate `*_clinical_windows.csv`.

**Deliverables:**
- `analysis/clinical_features.R` — standalone Rscript, same CLI
  pattern as `analysis/features.R` (accepts a CSV or directory).
- Unit-testable helper functions (joint angle, SAL, etc.)
  factored into the top of the file.
- Add a brief entry to `analysis/` usage in `README.md`.
- Update `AGENTS.md` with the new script and its outputs.

---

## Section 2 — Correlate Clinical Features with Clinical Scores

**Status:** [x] complete

**Goal:** Create `analysis/clinical_correlation.R` that joins
the derived clinical features with an external clinical scores
table and produces correlation analyses and visualisations.

**Context:**
- The clinical features CSV from Section 1 provides per-frame
  and per-window kinematic measures.
- Clinical outcome scores (e.g. GRASSP, UEMS, SCIM) will be
  provided in a separate CSV (`clinical_scores.csv`) with at
  minimum columns: `video` (matching the landmark CSV video
  name), plus one or more score columns.
- Since each video corresponds to one assessment session, the
  join key is `video`.  Clinical features must first be
  aggregated per video (mean, median, SD, min, max of each
  feature across frames/windows).

**Steps:**
1. Aggregate clinical features per video (summary statistics).
2. Join with clinical scores on `video`.
3. Compute pairwise Pearson and Spearman correlations between
   each aggregated feature and each clinical score.
4. Apply Benjamini-Hochberg FDR correction for multiple
   comparisons.
5. Produce outputs (see below).

**Outputs:**
- `*_correlation_matrix.png` — heatmap of feature × score
  correlations (Spearman rho), with significance stars.
- `*_correlation_table.csv` — tidy table with columns:
  `feature`, `score`, `pearson_r`, `spearman_rho`, `p_value`,
  `p_adj_bh`, `n`.
- `*_scatter_top.png` — scatter plots for the top 6
  feature–score pairs by absolute Spearman rho.
- Console summary of top correlations and any features with
  no significant associations.

**Deliverables:**
- `analysis/clinical_correlation.R` — standalone Rscript.
  Usage: `Rscript analysis/clinical_correlation.R
  clinical_features_dir/ clinical_scores.csv`.
- Graceful handling of missing scores or unmatched videos
  (warn and continue).
- Update `README.md` and `AGENTS.md`.

---

## Section 3 — Longitudinal Comparison Across Sessions

**Status:** [ ] incomplete

**Goal:** Create `analysis/longitudinal.R` that compares
clinical features from the same patient across multiple
sessions (time points) to track recovery.

**Context:**
- A patient may have multiple videos recorded at different
  dates during rehabilitation.
- A session metadata CSV (`sessions.csv`) maps each video to a
  patient and session date, with at minimum: `video`,
  `patient_id`, `session_date` (ISO 8601).
- Clinical features from Section 1 are aggregated per video.
- Optionally, clinical scores from Section 2 can be overlaid.

**Steps:**
1. Load session metadata and join with aggregated clinical
   features (and optionally clinical scores).
2. For each patient, order sessions chronologically.
3. Compute session-to-session deltas and percentage changes
   for each feature.
4. Flag statistically significant changes using a threshold
   (e.g. exceeds 1 SD of the baseline session, or a
   user-configurable minimal clinically important difference).
5. Produce outputs (see below).

**Outputs:**
- `*_longitudinal_<patient_id>.png` — line plots of selected
  features over sessions for each patient, with optional
  clinical score overlay on a secondary axis.
- `*_longitudinal_summary.csv` — tidy table: `patient_id`,
  `feature`, `session_date`, `value`, `delta_from_baseline`,
  `pct_change`, `flagged`.
- Console summary highlighting patients with notable
  improvement or decline.

**Deliverables:**
- `analysis/longitudinal.R` — standalone Rscript.  Usage:
  `Rscript analysis/longitudinal.R clinical_features_dir/
  sessions.csv [clinical_scores.csv]`.
- Handles patients with only one session (skip delta, still
  include in summary).
- Update `README.md` and `AGENTS.md`.

#!/usr/bin/env Rscript
# Clinical feature derivation from landmark CSVs produced by export.py.
#
# Computes clinically meaningful kinematic features per frame (joint
# angles, reach, grasp aperture, displacement) and per sliding window
# (spectral arc length, velocity statistics), writing two CSVs per
# input.
#
# Accepts both 2D landmark CSVs (normalised coords + MediaPipe
# pseudo-depth) and triangulated world3d.csv files (metres; columns
# end in _x_m/_y_m/_z_m).  3D inputs are quality-gated (reprojection
# error, cheirality, triangulation angle), yield metric
# distances/velocities (m, m/s), and
# get true trunk plane decomposition (world frame: +y down, +z away
# from the world camera; vertical assumes a level world camera).
#
# Usage:
#   Rscript analysis/clinical_features.R output/video1_hands-arms.csv
#   Rscript analysis/clinical_features.R output/session1/world3d.csv
#   Rscript analysis/clinical_features.R output/   # all landmark CSVs
#
# Outputs alongside each input CSV (suffixes gain `_3d` for 3D input —
# _clinical_3d.csv, _clinical_3d_windows.csv, _movement_phases_3d.csv —
# keeping metric-unit rows out of the 2D downstream globs):
#   <stem>_clinical.csv          — per-frame clinical features
#   <stem>_clinical_windows.csv  — per-window smoothness features
#   <stem>_movement_phases.csv   — segmented movement phases

library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(purrr)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Sliding-window duration (seconds) for smoothness features.
WINDOW_SEC <- 1.0

# Frequency cutoff (Hz) for spectral arc length calculation.
SAL_FREQ_CUTOFF <- 10

# An inter-sample interval longer than this multiple of the median is a
# tracking gap rather than a frame, and is excluded when estimating the
# nominal frame rate.
GAP_INTERVAL_FACTOR <- 1.5

# Largest sub-slot deviation tolerated when mapping timestamps onto the
# nominal frame grid. Rounded exports jitter by ~1e-5 slots; anything past a
# quarter slot means the series is not on the assumed grid.
GRID_SLOT_TOLERANCE <- 0.25

# 3D quality gate: keypoints whose mean reprojection error exceeds this
# (px) are masked to NA. Matches fuse_session_frame's per-view rejection
# threshold — at exactly min_views an outlier view cannot be dropped
# during fusion, so this downstream gate is mandatory.
REPROJ_GATE_PX <- 20

# Mirrors fuse_session_frame's default viewing-ray geometry gate.  The column
# is optional for compatibility with world3d.csv files written before the
# triangulation-angle diagnostic was added.
TRIANGULATION_ANGLE_GATE_DEG <- 1

# Explicit confidence zero means prediction/missing evidence, not an observed
# landmark. Positive scores remain available for analysis; separate validation
# reports flag pervasively low-but-positive confidence for review.
OBSERVATION_CONFIDENCE_GATE <- 0

# ------------------------------------------------------------------
# 3D artifact identity
# ------------------------------------------------------------------

# Producer layout version — bump when the emitted column set changes.
PRODUCER_VERSION <- "v1"

# Metric-definition version — bump when a metric's computation changes.
METRIC_METHOD_VERSION <- "v1"

# QC-policy version — bump when REPROJ_GATE_PX,
# TRIANGULATION_ANGLE_GATE_DEG or OBSERVATION_CONFIDENCE_GATE change value.
QC_POLICY_VERSION <- "v1"

# Coordinate space and distance unit of the metric-3D outputs.  Per-metric
# units (m, m/s, deg, dimensionless) are a registry concern, not an
# artifact-level tag: these outputs are wide and mix all four.
COORD_SPACE_3D   <- "world-metric-3d"
DISTANCE_UNIT_3D <- "m"

# Rig, model and filter identity are absent from world3d.csv, so provenance
# is declared unverified rather than guessed.  A reader refuses to pool
# across unverified artifacts instead of assuming they are equivalent.
PROVENANCE_CLASS <- "unverified"

# Gap semantics per artifact kind, stated in the artifact rather than only
# in the docs.  Window trajectory metrics run on the timestamp-aware kernel;
# movement-phase metrics still differentiate across holes; per-frame values
# are instantaneous, their displacements being row-adjacent steps that go NA
# on a masked sample without checking the interval.
QUALIFICATION_FRAME  <- "frame-instantaneous"
QUALIFICATION_WINDOW <- "gap-aware"
QUALIFICATION_PHASE  <- "gap-unsafe"

# Ordered artifact-level tag columns appended to every 3D output.
ARTIFACT_TAG_COLS <- c(
  "artifact_kind", "source_sha256", "coord_space", "distance_unit",
  "producer_version", "metric_method_version", "qc_policy_version",
  "metric_qualification", "provenance_class"
)

#' SHA-256 of a file's bytes.
#'
#' Deterministic artifact identity: the same bytes always hash the same,
#' independent of filename, path or run time, so goldens stay byte-stable
#' and a rerun over changed input is distinguishable from a copy.
file_sha256 <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  as.character(openssl::sha256(con))
}

#' Append the artifact-level identity tags to a 3D output table.
#'
#' Values are constant within a file and always character, so they can
#' never be picked up as numeric features by the generic per-video
#' summariser in utils.R.  Tags go last, leaving row keys first and the
#' metric block contiguous.  A zero-row table gains the columns typed and
#' empty, keeping its header identical to the populated case.
attach_artifact_tags <- function(df, kind, qualification, source_sha256) {
  values <- c(kind, source_sha256, COORD_SPACE_3D, DISTANCE_UNIT_3D,
              PRODUCER_VERSION, METRIC_METHOD_VERSION, QC_POLICY_VERSION,
              qualification, PROVENANCE_CLASS)
  for (i in seq_along(ARTIFACT_TAG_COLS)) {
    df[[ARTIFACT_TAG_COLS[i]]] <- rep(values[i], nrow(df))
  }
  df
}

# ------------------------------------------------------------------
# Column-name helpers (mode-aware)
# ------------------------------------------------------------------

detect_tracking <- function(cols) {
  if (any(str_starts(cols, "body_"))) return("body")
  if (any(str_starts(cols, "arm_")))  return("hands-arms")
  "hands"
}

body_col <- function(tracking, side, keypoint, coord) {
  prefix <- if (tracking == "body") "body" else "arm"
  paste0(prefix, "_", side, "_", keypoint, "_", coord)
}

hand_col <- function(side, idx, coord) {
  paste0(side, "_hand_", idx, "_", coord)
}

#' Mask 2D coordinates that are not backed by a current observation.
#'
#' Body/arm visibility has always been explicit. Hand confidence is present in
#' the current schema; legacy hand CSVs without `_conf` columns retain their
#' coordinate-presence behavior. Missing/nonfinite confidence in a present
#' schema fails closed.
adapt_2d_confidence <- function(df) {
  confidence_cols <- names(df)[
    str_detect(names(df), "^(arm|body)_.+_vis$") |
      str_detect(names(df), "^(left|right)_hand_[0-9]+_conf$")
  ]
  for (confidence_col in confidence_cols) {
    prefix <- str_remove(confidence_col, "_(vis|conf)$")
    confidence <- as.numeric(df[[confidence_col]])
    bad <- !is.finite(confidence) | confidence <= OBSERVATION_CONFIDENCE_GATE
    for (coord in c("x", "y", "z")) {
      coord_col <- paste0(prefix, "_", coord)
      if (coord_col %in% names(df)) {
        df[[coord_col]][bad] <- NA_real_
      }
    }
  }
  df
}

# ------------------------------------------------------------------
# 3D input adapter (world3d.csv)
# ------------------------------------------------------------------

#' Detect a triangulated 3D input by its metre-unit coordinate columns.
is_world3d <- function(cols) {
  any(str_ends(cols, "_x_m"))
}

#' Adapt a world3d.csv data frame to the 2D landmark column layout.
#'
#' Two steps, per keypoint:
#' 1. Quality gate — coordinates are masked to NA where the fusion
#'    diagnostics disqualify the point: cheirality violation, mean
#'    reprojection error above \code{REPROJ_GATE_PX}, or an available
#'    triangulation angle below \code{TRIANGULATION_ANGLE_GATE_DEG}.
#' 2. Rename — \code{{kp}_x_m/_y_m/_z_m} become \code{{kp}_x/_y/_z}
#'    so every downstream feature function works unchanged (distances
#'    arrive in metres, velocities in m/s).
#'
#' Diagnostic columns (_confidence, _reproj_err_px, _candidate_n_views, _n_views,
#' _cheirality_ok, _triangulation_angle_deg) are dropped after gating.
adapt_world3d <- function(df) {
  kp_names <- str_remove(names(df)[str_ends(names(df), "_x_m")], "_x_m$")
  for (kp in kp_names) {
    reproj <- as.numeric(df[[paste0(kp, "_reproj_err_px")]])
    cheir  <- as.numeric(df[[paste0(kp, "_cheirality_ok")]])
    coordinate_cols <- paste0(kp, "_", c("x", "y", "z"), "_m")
    finite_coordinates <- Reduce(
      `&`,
      lapply(coordinate_cols, function(col) is.finite(as.numeric(df[[col]])))
    )
    angle_col <- paste0(kp, "_triangulation_angle_deg")
    has_angle <- angle_col %in% names(df)
    angle <- if (has_angle) {
      as.numeric(df[[angle_col]])
    } else {
      rep(NA_real_, nrow(df))
    }
    # Diagnostics in a present schema are required evidence, so blank/nonfinite
    # values fail closed just like validation.py's trusted-point mask.  Angle
    # gating remains optional only for legacy files lacking the column entirely.
    bad <- !finite_coordinates | !is.finite(reproj) | !is.finite(cheir) |
           reproj > REPROJ_GATE_PX | cheir != 1
    if (has_angle) {
      bad <- bad | !is.finite(angle) | angle < TRIANGULATION_ANGLE_GATE_DEG
    }
    for (coord in c("x", "y", "z")) {
      col <- paste0(kp, "_", coord, "_m")
      df[[col]][bad] <- NA_real_
    }
  }
  diag_cols <- str_ends(names(df), "_confidence") |
               str_ends(names(df), "_reproj_err_px") |
               str_ends(names(df), "_candidate_n_views") |
               str_ends(names(df), "_n_views") |
               str_ends(names(df), "_cheirality_ok") |
               str_ends(names(df), "_triangulation_angle_deg")
  df <- df[, !diag_cols]
  names(df) <- str_replace(names(df), "_([xyz])_m$", "_\\1")
  df
}

# ------------------------------------------------------------------
# Bilateral comparison helpers
# ------------------------------------------------------------------

#' Compute bilateral symmetry metrics from left/right vectors.
#'
#' Uses abs() internally — works for both non-negative metrics (angles,
#' distances) and negative metrics (SAL).
#'
#' @param L Numeric vector — left-side values.
#' @param R Numeric vector — right-side values.
#' @return Named list of 3 vectors: symmetry_ratio (0–1, 1=symmetric),
#'   dominance_index (-1 to 1, positive=right has larger magnitude),
#'   abs_diff (≥0, raw asymmetry in original units).
compute_bilateral <- function(L, R) {
  aL <- abs(L)
  aR <- abs(R)
  denom <- aL + aR

  sym <- ifelse(denom > 1e-12, pmin(aL, aR) / pmax(aL, aR), NA_real_)
  dom <- ifelse(denom > 1e-12, (aR - aL) / denom, NA_real_)
  dif <- abs(R - L)

  list(symmetry_ratio = sym, dominance_index = dom, abs_diff = dif)
}

# ------------------------------------------------------------------
# Geometry helpers (unit-testable)
# ------------------------------------------------------------------

#' Angle at vertex B in the triangle A-B-C, in degrees (vectorised).
#' Returns NA where any input is NA or where a zero-length arm occurs.
angle_at_vertex <- function(ax, ay, az, bx, by, bz, cx, cy, cz) {
  ba_x <- ax - bx;  ba_y <- ay - by;  ba_z <- az - bz
  bc_x <- cx - bx;  bc_y <- cy - by;  bc_z <- cz - bz

  dot   <- ba_x * bc_x + ba_y * bc_y + ba_z * bc_z
  mag_a <- sqrt(ba_x^2 + ba_y^2 + ba_z^2)
  mag_c <- sqrt(bc_x^2 + bc_y^2 + bc_z^2)

  denom <- mag_a * mag_c
  cos_angle <- ifelse(denom > 1e-12, dot / denom, NA_real_)
  cos_angle <- pmax(pmin(cos_angle, 1), -1)
  acos(cos_angle) * 180 / pi
}

#' Euclidean distance between two 3D points (vectorised).
dist_3d <- function(ax, ay, az, bx, by, bz) {
  sqrt((ax - bx)^2 + (ay - by)^2 + (az - bz)^2)
}

#' Spectral Arc Length (Balasubramanian et al. 2012/2015).
#'
#' @param v Numeric vector — velocity magnitude time series.
#' @param fs Scalar — sampling frequency in Hz.
#' @return Negative scalar; more negative = less smooth.  Returns
#'   \code{NA_real_} when the input is too short or degenerate.
spectral_arc_length <- function(v, fs, fc = SAL_FREQ_CUTOFF) {
  v <- v[!is.na(v)]
  n <- length(v)
  if (n < 4 || fs <= 0) return(NA_real_)

  v_peak <- max(abs(v))
  if (v_peak < 1e-10) return(0)  # no movement
  v_norm <- v / v_peak

  # One-sided FFT magnitude spectrum, normalised to peak = 1.
  V <- Mod(fft(v_norm))[seq_len(floor(n / 2) + 1)]
  V <- V / max(V)

  freqs <- seq(0, fs / 2, length.out = length(V))

  fc <- min(fc, fs / 2)
  keep <- freqs <= fc
  V     <- V[keep]
  freqs <- freqs[keep]
  if (length(freqs) < 2) return(NA_real_)

  # Arc length of the normalised magnitude spectrum.
  dfreq <- diff(freqs) / fc
  dV    <- diff(V)
  -sum(sqrt(dfreq^2 + dV^2))
}

#' Normalized Jerk — dimensionless movement smoothness metric.
#'
#' Hogan & Sternad (2009): NJ = sqrt(T^5 / (2 * a^2) * integral(||jerk||^2 dt)).
#' Lower NJ = smoother; minimum-jerk trajectory gives ~18.97.
#'
#' @param x,y,z Numeric vectors — 3D position time series.
#' @param fs Scalar — sampling frequency in Hz.
#' @return Positive scalar (dimensionless). NA when input is too short or
#'   amplitude is negligible.
normalized_jerk <- function(x, y, z, fs) {
  ok <- !is.na(x) & !is.na(y) & !is.na(z)
  x <- x[ok]; y <- y[ok]; z <- z[ok]
  n <- length(x)
  if (n < 5 || fs <= 0) return(NA_real_)

  dt <- 1 / fs
  T_dur <- (n - 1) * dt

  amplitude <- sum(sqrt(diff(x)^2 + diff(y)^2 + diff(z)^2))
  if (amplitude < 1e-10) return(NA_real_)

  vx <- diff(x) * fs;  vy <- diff(y) * fs;  vz <- diff(z) * fs
  ax <- diff(vx) * fs;  ay <- diff(vy) * fs;  az <- diff(vz) * fs
  jx <- diff(ax) * fs;  jy <- diff(ay) * fs;  jz <- diff(az) * fs

  integral_jerk_sq <- sum(jx^2 + jy^2 + jz^2) * dt
  sqrt(T_dur^5 / (2 * amplitude^2) * integral_jerk_sq)
}

#' Movement Efficiency — path curvature ratio.
#'
#' Ratio of path length to straight-line (start→end) distance.
#' 1.0 = perfectly straight; higher = more curved/corrective.
#'
#' @param x,y,z Numeric vectors — 3D position time series.
#' @return Scalar >= 1.0. NA when start ≈ end or input is too short.
movement_efficiency <- function(x, y, z) {
  ok <- !is.na(x) & !is.na(y) & !is.na(z)
  x <- x[ok]; y <- y[ok]; z <- z[ok]
  n <- length(x)
  if (n < 2) return(NA_real_)

  path_len <- sum(sqrt(diff(x)^2 + diff(y)^2 + diff(z)^2))
  straight <- sqrt((x[n] - x[1])^2 + (y[n] - y[1])^2 + (z[n] - z[1])^2)

  if (straight < 1e-10) return(NA_real_)
  path_len / straight
}

#' Nominal frame rate, robust to timestamp quantisation.
#'
#' Exported timestamps are rounded to four decimals (see
#' \code{src/pose_estimation/export.py}), so \code{1 / median(diff(t))} reads
#' 30.03 Hz for a 30 fps capture: the rounded intervals alternate 0.0333 and
#' 0.0334 and the median picks the shorter one.  Averaging the non-gap
#' intervals cancels the quantisation instead of amplifying it.  Intervals
#' longer than \code{GAP_INTERVAL_FACTOR} times the median are dropped so that
#' tracking gaps do not inflate the estimate.
#'
#' @param t Numeric vector — timestamps in seconds.
#' @return Scalar sampling frequency in Hz, or \code{NA_real_} when
#'   undeterminable.
nominal_fs <- function(t) {
  d <- diff(t[!is.na(t)])
  d <- d[is.finite(d) & d > 0]
  if (length(d) == 0) return(NA_real_)

  keep <- d <= GAP_INTERVAL_FACTOR * median(d)
  dt <- mean(d[keep])
  if (!is.finite(dt) || dt <= 0) return(NA_real_)
  1 / dt
}

#' Map a timestamped sample series onto its nominal uniform frame grid.
#'
#' Video frames are nominally uniform, so a quality-gated NA sample and an
#' absent row are both holes at *known* grid positions.  Rounding to the
#' nearest slot absorbs sub-slot timestamp jitter, which keeps a complete
#' series on exactly the legacy arithmetic while giving a gapped series a
#' correct time base.  Fails closed rather than guessing: ambiguous or
#' colliding timestamps are an input defect, not something to repair silently.
#'
#' @param t Numeric vector — timestamps in seconds, strictly increasing.
#' @param fs Scalar — nominal sampling frequency in Hz.
#' @return List of \code{slot} (0-based grid index per sample), \code{n_grid}
#'   (grid length) and \code{residual} (largest sub-slot deviation).
trajectory_grid <- function(t, fs) {
  if (!is.finite(fs) || fs <= 0) {
    stop("trajectory_grid(): fs must be finite and positive")
  }
  if (length(t) < 2) stop("trajectory_grid(): t must contain at least two timestamps")
  if (any(!is.finite(t))) stop("trajectory_grid(): timestamps must be finite")
  if (any(diff(t) <= 0)) {
    stop("trajectory_grid(): timestamps must be strictly increasing")
  }

  raw <- (t - t[1]) * fs
  slot <- round(raw)
  residual <- max(abs(raw - slot))
  if (residual > GRID_SLOT_TOLERANCE) {
    stop(sprintf(
      "trajectory_grid(): timestamps do not follow a %g Hz grid (residual %.3f)",
      fs, residual
    ))
  }
  if (anyDuplicated(slot)) {
    stop("trajectory_grid(): two samples map onto one grid slot")
  }

  list(slot = slot, n_grid = slot[length(slot)] + 1, residual = residual)
}

#' Gap-aware trajectory metrics over the nominal frame grid.
#'
#' The timestamp-aware kernel behind every window-scope smoothness and
#' velocity quantity.  Samples are placed on their nominal slots and each
#' derivative is masked wherever its stencil touches a hole, so a tracking gap
#' can no longer be differentiated across as though the survivors were
#' adjacent.  On a complete grid every metric reduces to the legacy operation
#' order and stays bit-identical.
#'
#' Estimands, which differ deliberately in how they treat an unobserved span:
#' \itemize{
#'   \item \code{nj} — jerk over fully observed 4-wide stencils, fixed
#'     \code{dt}, duration = the true grid span.
#'   \item \code{sal} — interior missing speed intervals are filled linearly;
#'     a leading or trailing gap yields \code{NA} rather than extrapolated
#'     motion.
#'   \item \code{v_mean}, \code{v_peak} — observed support only.  A gap hides
#'     whatever happened inside it, so both are biased low under loss;
#'     \code{dropout} is what tells a consumer by how much.
#'   \item \code{efficiency} — \code{NA} when the observed path is broken.
#'     Bridging a hole with a straight chord biases the ratio toward 1.0, i.e.
#'     reports a straighter, healthier movement than was observed.
#' }
#'
#' @param t Numeric vector — timestamps in seconds.
#' @param x,y,z Numeric vectors — 3D position time series.
#' @param fs Scalar — sampling frequency in Hz; derived from \code{t} when
#'   \code{NULL}.
#' @param fc Scalar — spectral arc length frequency cutoff in Hz.
#' @return Named list of \code{sal}, \code{nj}, \code{v_mean}, \code{v_peak},
#'   \code{efficiency}, \code{dropout} and \code{longest_gap_sec}.
trajectory_metrics <- function(t, x, y, z, fs = NULL, fc = SAL_FREQ_CUTOFF) {
  empty <- list(
    sal = NA_real_, nj = NA_real_, v_mean = NA_real_, v_peak = NA_real_,
    efficiency = NA_real_, dropout = NA_real_, longest_gap_sec = NA_real_
  )
  if (length(t) < 2) return(empty)

  if (is.null(fs)) fs <- nominal_fs(t)
  if (!is.finite(fs) || fs <= 0) return(empty)

  grid <- trajectory_grid(t, fs)
  dt <- 1 / fs
  n_grid <- grid$n_grid
  at <- grid$slot + 1

  gx <- rep(NA_real_, n_grid); gx[at] <- x
  gy <- rep(NA_real_, n_grid); gy[at] <- y
  gz <- rep(NA_real_, n_grid); gz[at] <- z

  valid <- !is.na(gx) & !is.na(gy) & !is.na(gz)
  gx[!valid] <- NA_real_; gy[!valid] <- NA_real_; gz[!valid] <- NA_real_

  runs <- rle(valid)
  gap_runs <- runs$lengths[!runs$values]
  out <- empty
  out$dropout <- (n_grid - sum(valid)) / n_grid
  out$longest_gap_sec <- if (length(gap_runs)) max(gap_runs) * dt else 0

  # Interval quantities.  diff() propagates NA outward one slot per derivative
  # order, so a hole invalidates exactly the stencils that span it.
  step <- sqrt(diff(gx)^2 + diff(gy)^2 + diff(gz)^2)
  speed <- step * fs
  n_step <- sum(!is.na(step))
  if (n_step == 0) return(out)

  # Duration-weighted mean over observed intervals: sum(step) / (n_step * dt)
  # is exactly mean(speed) once the shared dt cancels.
  out$v_mean <- mean(speed, na.rm = TRUE)
  out$v_peak <- max(speed, na.rm = TRUE)

  amplitude <- sum(step, na.rm = TRUE)

  # Normalized jerk over fully observed stencils, normalised by the true span.
  if (n_grid >= 5 && amplitude >= 1e-10) {
    vx <- diff(gx) * fs;  vy <- diff(gy) * fs;  vz <- diff(gz) * fs
    ax <- diff(vx) * fs;  ay <- diff(vy) * fs;  az <- diff(vz) * fs
    jx <- diff(ax) * fs;  jy <- diff(ay) * fs;  jz <- diff(az) * fs

    if (any(!is.na(jx))) {
      integral_jerk_sq <- sum(jx^2 + jy^2 + jz^2, na.rm = TRUE) * dt
      T_dur <- (n_grid - 1) * dt
      out$nj <- sqrt(T_dur^5 / (2 * amplitude^2) * integral_jerk_sq)
    }
  }

  # Movement efficiency needs an unbroken observed path; a hole between the
  # first and last observation makes the true path length unidentifiable.
  first <- which(valid)[1]
  last <- which(valid)[sum(valid)]
  if (last > first && all(valid[first:last])) {
    straight <- sqrt(
      (gx[last] - gx[first])^2 + (gy[last] - gy[first])^2 +
        (gz[last] - gz[first])^2
    )
    if (straight >= 1e-10) {
      out$efficiency <- sum(step[first:(last - 1)]) / straight
    }
  }

  # Spectral arc length presumes a complete uniform series.  Interior speed
  # intervals are reconstructed linearly; edges are never extrapolated.
  obs <- which(!is.na(speed))
  if (length(obs) >= 4) {
    lo <- obs[1]
    hi <- obs[length(obs)]
    if (lo == 1 && hi == length(speed)) {
      filled <- speed
      if (anyNA(filled)) {
        filled <- approx(obs, speed[obs], xout = seq_along(speed))$y
      }
      out$sal <- spectral_arc_length(filled, fs, fc)
    }
  }

  out
}

#' Trunk lean angle from vertical (2D, unsigned).
#'
#' Angle between the shoulder-midpoint→hip-midpoint vector and the
#' vertical axis, in degrees. Vectorised over frames. 0 = upright,
#' 90 = fully horizontal. Body mode only (requires hip keypoints).
#'
#' @param lsh_x,lsh_y,rsh_x,rsh_y Left/right shoulder x,y.
#' @param lhip_x,lhip_y,rhip_x,rhip_y Left/right hip x,y.
#' @return Numeric vector of unsigned angles in degrees.
trunk_lean_angle <- function(lsh_x, lsh_y, rsh_x, rsh_y,
                             lhip_x, lhip_y, rhip_x, rhip_y) {
  sh_mx <- (lsh_x + rsh_x) / 2
  sh_my <- (lsh_y + rsh_y) / 2
  hip_mx <- (lhip_x + rhip_x) / 2
  hip_my <- (lhip_y + rhip_y) / 2

  dx <- sh_mx - hip_mx
  dy <- sh_my - hip_my  # image coords: +y = down, so upright → dy < 0

  atan2(abs(dx), abs(dy)) * 180 / pi
}

#' Trunk lateral lean — signed angle in the frontal plane.
#'
#' Vectorised. 0 = upright, positive = leaning right (shoulders right
#' of hips), negative = leaning left. Image coords: +y = down.
#'
#' @inheritParams trunk_lean_angle
#' @return Numeric vector of signed angles in degrees.
trunk_lean_lateral <- function(lsh_x, lsh_y, rsh_x, rsh_y,
                               lhip_x, lhip_y, rhip_x, rhip_y) {
  sh_mx <- (lsh_x + rsh_x) / 2
  sh_my <- (lsh_y + rsh_y) / 2
  hip_mx <- (lhip_x + rhip_x) / 2
  hip_my <- (lhip_y + rhip_y) / 2

  dx <- sh_mx - hip_mx       # positive = shoulders right of hips
  dy <- sh_my - hip_my       # negative when upright (+y down)

  # atan2(lateral, vertical_up): vertical_up = -dy for image coords
  atan2(dx, -dy) * 180 / pi
}

#' Trunk rotation — shoulder line vs hip line angle difference.
#'
#' Signed angle between the shoulder line (left→right) and the hip
#' line (left→right) in image-plane 2D. Positive = shoulders rotated
#' clockwise relative to hips (viewed from front). Vectorised.
#'
#' @inheritParams trunk_lean_angle
#' @return Numeric vector of signed angles in degrees, wrapped to (-180, 180].
trunk_rotation <- function(lsh_x, lsh_y, rsh_x, rsh_y,
                           lhip_x, lhip_y, rhip_x, rhip_y) {
  sh_angle  <- atan2(rsh_y - lsh_y, rsh_x - lsh_x)
  hip_angle <- atan2(rhip_y - lhip_y, rhip_x - lhip_x)

  d <- sh_angle - hip_angle
  atan2(sin(d), cos(d)) * 180 / pi
}

#' Posture symmetry — normalised shoulder height asymmetry.
#'
#' (left_shoulder_y − right_shoulder_y) / shoulder_width. In image
#' coords (+y down): positive = right shoulder higher (left dropped),
#' negative = left shoulder higher (right dropped). Vectorised.
#'
#' @param lsh_x,lsh_y,rsh_x,rsh_y Left/right shoulder x,y.
#' @return Numeric vector, dimensionless. NA when shoulder width ≈ 0.
posture_symmetry <- function(lsh_x, lsh_y, rsh_x, rsh_y) {
  sh_width <- sqrt((rsh_x - lsh_x)^2 + (rsh_y - lsh_y)^2)
  ifelse(sh_width > 1e-6, (lsh_y - rsh_y) / sh_width, NA_real_)
}

# ------------------------------------------------------------------
# 3D trunk helpers (world3d input — true plane decomposition)
# ------------------------------------------------------------------
# World frame = the world-frame camera's frame (OpenCV convention):
# +x right, +y down, +z away from the camera. Vertical is taken as -y,
# which assumes a level world camera. All helpers are vectorised over
# frames and take shoulder/hip midline components.

#' Total trunk lean from vertical, 3D (unsigned, degrees).
#' atan2(horizontal magnitude, vertical component of hip→shoulder).
#' 0 = upright, 90 = horizontal; >90 = inverted.
trunk_lean_angle_3d <- function(dx, dy, dz) {
  atan2(sqrt(dx^2 + dz^2), -dy) * 180 / pi
}

#' Sagittal trunk lean, 3D (signed, degrees). Positive = leaning away
#' from the world camera (+z), negative = toward it. Unmeasurable from
#' a single 2D view — NA in 2D mode.
trunk_lean_sagittal_3d <- function(dy, dz) {
  atan2(dz, -dy) * 180 / pi
}

#' Axial trunk rotation, 3D (signed, degrees, wrapped to (-180, 180]).
#' Shoulder line vs hip line projected onto the transverse (x–z)
#' plane — true rotation about the vertical axis, unlike the 2D
#' image-plane proxy.
trunk_rotation_3d <- function(lsh_x, lsh_z, rsh_x, rsh_z,
                              lhip_x, lhip_z, rhip_x, rhip_z) {
  sh_angle  <- atan2(rsh_z - lsh_z, rsh_x - lsh_x)
  hip_angle <- atan2(rhip_z - lhip_z, rhip_x - lhip_x)
  d <- sh_angle - hip_angle
  atan2(sin(d), cos(d)) * 180 / pi
}

#' Posture symmetry, 3D — shoulder height difference normalised by
#' the full 3D shoulder width. Positive = right shoulder higher.
posture_symmetry_3d <- function(lsh_x, lsh_y, lsh_z, rsh_x, rsh_y, rsh_z) {
  sh_width <- sqrt((rsh_x - lsh_x)^2 + (rsh_y - lsh_y)^2 + (rsh_z - lsh_z)^2)
  ifelse(sh_width > 1e-6, (lsh_y - rsh_y) / sh_width, NA_real_)
}

# ------------------------------------------------------------------
# Per-frame feature computation
# ------------------------------------------------------------------

compute_frame_features <- function(df, tracking, is_3d = FALSE) {
  bcol <- function(side, kp, coord) body_col(tracking, side, kp, coord)
  hcol <- hand_col

  # Wrist-deviation target differs by mode.
  wrist_dev_kp <- if (tracking == "body") "index" else "middle_base"

  n <- nrow(df)

  # Safe column extraction — returns NA vector when column is absent.
  ex <- function(cname) {
    if (cname %in% names(df)) as.numeric(df[[cname]]) else rep(NA_real_, n)
  }

  result <- tibble(
    video         = df$video,
    frame_idx     = as.integer(df$frame_idx),
    timestamp_sec = as.numeric(df$timestamp_sec),
    person_idx    = as.integer(df$person_idx)
  )

  for (side in c("left", "right")) {
    opp <- if (side == "left") "right" else "left"

    # --- Arm/body keypoints ---
    sh_x  <- ex(bcol(side, "shoulder", "x"))
    sh_y  <- ex(bcol(side, "shoulder", "y"))
    sh_z  <- ex(bcol(side, "shoulder", "z"))
    el_x  <- ex(bcol(side, "elbow", "x"))
    el_y  <- ex(bcol(side, "elbow", "y"))
    el_z  <- ex(bcol(side, "elbow", "z"))
    wr_x  <- ex(bcol(side, "wrist", "x"))
    wr_y  <- ex(bcol(side, "wrist", "y"))
    wr_z  <- ex(bcol(side, "wrist", "z"))
    dev_x <- ex(bcol(side, wrist_dev_kp, "x"))
    dev_y <- ex(bcol(side, wrist_dev_kp, "y"))
    dev_z <- ex(bcol(side, wrist_dev_kp, "z"))
    osh_x <- ex(bcol(opp, "shoulder", "x"))
    osh_y <- ex(bcol(opp, "shoulder", "y"))
    osh_z <- ex(bcol(opp, "shoulder", "z"))

    # --- Hand keypoints ---
    hw_x  <- ex(hcol(side, 0, "x"))   # hand wrist (landmark 0)
    hw_y  <- ex(hcol(side, 0, "y"))
    hw_z  <- ex(hcol(side, 0, "z"))
    th_x  <- ex(hcol(side, 4, "x"))   # thumb tip
    th_y  <- ex(hcol(side, 4, "y"))
    th_z  <- ex(hcol(side, 4, "z"))
    ix_x  <- ex(hcol(side, 8, "x"))   # index fingertip
    ix_y  <- ex(hcol(side, 8, "y"))
    ix_z  <- ex(hcol(side, 8, "z"))
    pk_x  <- ex(hcol(side, 20, "x"))  # pinky tip
    pk_y  <- ex(hcol(side, 20, "y"))
    pk_z  <- ex(hcol(side, 20, "z"))

    # 1a. Elbow flexion angle (shoulder-elbow-wrist).
    result[[paste0(side, "_elbow_angle_deg")]] <-
      angle_at_vertex(sh_x, sh_y, sh_z,
                      el_x, el_y, el_z,
                      wr_x, wr_y, wr_z)

    # 1b. Wrist deviation angle (elbow-wrist-finger_base).
    result[[paste0(side, "_wrist_deviation_deg")]] <-
      angle_at_vertex(el_x, el_y, el_z,
                      wr_x, wr_y, wr_z,
                      dev_x, dev_y, dev_z)

    # 1c. Finger spread (index_tip-hand_wrist-pinky_tip).
    result[[paste0(side, "_finger_spread_deg")]] <-
      angle_at_vertex(ix_x, ix_y, ix_z,
                      hw_x, hw_y, hw_z,
                      pk_x, pk_y, pk_z)

    # 2. Reach distance (shoulder→wrist), raw and normalised by
    #    shoulder width.
    reach <- dist_3d(sh_x, sh_y, sh_z, wr_x, wr_y, wr_z)
    shoulder_w <- dist_3d(sh_x, sh_y, sh_z, osh_x, osh_y, osh_z)
    result[[paste0(side, "_reach_raw")]] <- reach
    result[[paste0(side, "_reach_norm")]] <-
      ifelse(shoulder_w > 1e-6, reach / shoulder_w, NA_real_)

    # 3. Grasp aperture (thumb tip↔index tip, thumb tip↔pinky tip).
    result[[paste0(side, "_grasp_aperture_thumb_index")]] <-
      dist_3d(th_x, th_y, th_z, ix_x, ix_y, ix_z)
    result[[paste0(side, "_grasp_aperture_thumb_pinky")]] <-
      dist_3d(th_x, th_y, th_z, pk_x, pk_y, pk_z)

    # 5. Frame-to-frame displacement (computed per person group below).
    result[[paste0(side, "_wrist_displacement")]]     <- NA_real_
    result[[paste0(side, "_fingertip_displacement")]]  <- NA_real_
  }

  # --- Compute displacements within each person group ---
  grp_ids <- paste(result$video, result$person_idx, sep = "|")
  for (g in unique(grp_ids)) {
    idx <- which(grp_ids == g)
    if (length(idx) < 2) next

    for (side in c("left", "right")) {
      wr_x <- ex(bcol(side, "wrist", "x"))[idx]
      wr_y <- ex(bcol(side, "wrist", "y"))[idx]
      wr_z <- ex(bcol(side, "wrist", "z"))[idx]
      ix_x <- ex(hcol(side, 8, "x"))[idx]
      ix_y <- ex(hcol(side, 8, "y"))[idx]
      ix_z <- ex(hcol(side, 8, "z"))[idx]

      m <- length(idx)
      w_disp <- c(NA_real_,
                   dist_3d(wr_x[-1], wr_y[-1], wr_z[-1],
                           wr_x[-m], wr_y[-m], wr_z[-m]))
      f_disp <- c(NA_real_,
                   dist_3d(ix_x[-1], ix_y[-1], ix_z[-1],
                           ix_x[-m], ix_y[-m], ix_z[-m]))

      result[[paste0(side, "_wrist_displacement")]][idx]    <- w_disp
      result[[paste0(side, "_fingertip_displacement")]][idx] <- f_disp
    }
  }

  # --- Bilateral comparison metrics ---
  bilateral_metrics <- c(
    "elbow_angle_deg", "wrist_deviation_deg", "finger_spread_deg",
    "reach_raw", "reach_norm",
    "grasp_aperture_thumb_index", "grasp_aperture_thumb_pinky",
    "wrist_displacement", "fingertip_displacement"
  )
  for (metric in bilateral_metrics) {
    bl <- compute_bilateral(
      result[[paste0("left_", metric)]],
      result[[paste0("right_", metric)]]
    )
    result[[paste0(metric, "_symmetry_ratio")]]  <- bl$symmetry_ratio
    result[[paste0(metric, "_dominance_index")]]  <- bl$dominance_index
    result[[paste0(metric, "_abs_diff")]]         <- bl$abs_diff
  }

  # --- Trunk/torso metrics (body mode only — requires hip keypoints) ---
  if (tracking == "body") {
    lsh_x  <- ex("body_left_shoulder_x")
    lsh_y  <- ex("body_left_shoulder_y")
    rsh_x  <- ex("body_right_shoulder_x")
    rsh_y  <- ex("body_right_shoulder_y")
    lhip_x <- ex("body_left_hip_x")
    lhip_y <- ex("body_left_hip_y")
    rhip_x <- ex("body_right_hip_x")
    rhip_y <- ex("body_right_hip_y")

    # Lateral lean uses x,y only — same formula in 2D image coords and
    # the 3D world frame (both are +y down).
    result[["trunk_lean_lateral_deg"]] <-
      trunk_lean_lateral(lsh_x, lsh_y, rsh_x, rsh_y,
                         lhip_x, lhip_y, rhip_x, rhip_y)

    if (is_3d) {
      lsh_z  <- ex("body_left_shoulder_z")
      rsh_z  <- ex("body_right_shoulder_z")
      lhip_z <- ex("body_left_hip_z")
      rhip_z <- ex("body_right_hip_z")

      dx <- (lsh_x + rsh_x) / 2 - (lhip_x + rhip_x) / 2
      dy <- (lsh_y + rsh_y) / 2 - (lhip_y + rhip_y) / 2
      dz <- (lsh_z + rsh_z) / 2 - (lhip_z + rhip_z) / 2

      result[["trunk_lean_deg"]]          <- trunk_lean_angle_3d(dx, dy, dz)
      result[["trunk_lean_sagittal_deg"]] <- trunk_lean_sagittal_3d(dy, dz)
      result[["trunk_rotation_deg"]] <-
        trunk_rotation_3d(lsh_x, lsh_z, rsh_x, rsh_z,
                          lhip_x, lhip_z, rhip_x, rhip_z)
      result[["posture_symmetry"]] <-
        posture_symmetry_3d(lsh_x, lsh_y, lsh_z, rsh_x, rsh_y, rsh_z)
    } else {
      result[["trunk_lean_deg"]] <-
        trunk_lean_angle(lsh_x, lsh_y, rsh_x, rsh_y,
                         lhip_x, lhip_y, rhip_x, rhip_y)
      # Out-of-plane: unmeasurable from a single 2D view.
      result[["trunk_lean_sagittal_deg"]] <- NA_real_
      result[["trunk_rotation_deg"]] <-
        trunk_rotation(lsh_x, lsh_y, rsh_x, rsh_y,
                       lhip_x, lhip_y, rhip_x, rhip_y)
      result[["posture_symmetry"]] <-
        posture_symmetry(lsh_x, lsh_y, rsh_x, rsh_y)
    }
  } else {
    result[["trunk_lean_deg"]]          <- NA_real_
    result[["trunk_lean_lateral_deg"]]  <- NA_real_
    result[["trunk_lean_sagittal_deg"]] <- NA_real_
    result[["trunk_rotation_deg"]]      <- NA_real_
    result[["posture_symmetry"]]        <- NA_real_
  }

  result
}

# ------------------------------------------------------------------
# Window-level smoothness features
# ------------------------------------------------------------------

# Per-side window metrics, in emission order.
WINDOW_SIDE_METRICS <- c(
  "wrist_sal", "wrist_velocity_mean", "wrist_velocity_peak",
  "wrist_normalized_jerk", "wrist_movement_efficiency",
  "fingertip_normalized_jerk"
)

# Window metrics that also get left/right comparison columns.
WINDOW_BILATERAL_METRICS <- WINDOW_SIDE_METRICS

# Body-mode window summaries, in emission order.  Hands-arms mode emits the
# same names filled with NA, so the column set does not depend on tracking.
WINDOW_BODY_METRICS <- c(
  "compensatory_pattern_index", "trunk_lean_mean", "trunk_lean_sd",
  "trunk_lean_range", "trunk_lean_sagittal_mean", "trunk_lean_sagittal_sd",
  "trunk_lean_lateral_mean", "trunk_lean_lateral_sd", "trunk_rotation_mean",
  "trunk_rotation_sd", "posture_symmetry_mean", "posture_symmetry_sd"
)

#' Zero-row window table carrying the full ordered schema.
#'
#' Returned when no window qualifies, so an empty result is publishable
#' rather than shapeless.  A test pins this header against the populated
#' one, which is what keeps the two construction paths from drifting.
window_schema <- function() {
  out <- tibble(
    video            = character(),
    person_idx       = integer(),
    window_start_sec = double(),
    window_end_sec   = double()
  )
  bilateral_cols <- character(0)
  for (metric in WINDOW_BILATERAL_METRICS) {
    bilateral_cols <- c(bilateral_cols, paste0(
      metric, c("_symmetry_ratio", "_dominance_index", "_abs_diff")
    ))
  }
  numeric_cols <- c(
    paste0("left_", WINDOW_SIDE_METRICS),
    paste0("right_", WINDOW_SIDE_METRICS),
    WINDOW_BODY_METRICS,
    bilateral_cols
  )
  for (col in numeric_cols) out[[col]] <- double()
  out
}

compute_window_features <- function(df, frame_features, tracking,
                                    window_sec = WINDOW_SEC) {
  bcol <- function(side, kp, coord) body_col(tracking, side, kp, coord)
  hcol <- hand_col

  groups <- frame_features |>
    select(video, person_idx) |>
    distinct()

  results <- vector("list", nrow(groups) * 100L)
  ri <- 0L

  for (g in seq_len(nrow(groups))) {
    vid <- groups$video[g]
    pid <- groups$person_idx[g]

    mask <- df$video == vid & df$person_idx == pid
    sub_df <- df[mask, ]
    sub_ff <- frame_features[mask, ]

    ts <- as.numeric(sub_df$timestamp_sec)
    n  <- length(ts)
    if (n < 4) next

    dt_median <- median(diff(ts), na.rm = TRUE)
    if (is.na(dt_median) || dt_median <= 0) next
    fs <- 1 / dt_median

    # 3D inputs may have blank timestamps on frames the reference
    # camera missed — guard the window arithmetic against NA.
    t_start <- suppressWarnings(min(ts, na.rm = TRUE))
    t_end   <- suppressWarnings(max(ts, na.rm = TRUE))
    if (!is.finite(t_start) || !is.finite(t_end)) next

    # 50 %-overlapping windows.
    if (t_end - t_start < window_sec) next
    win_starts <- seq(t_start, t_end - window_sec, by = window_sec / 2)
    if (length(win_starts) == 0) next

    for (ws in win_starts) {
      we <- ws + window_sec
      win_mask <- !is.na(ts) & ts >= ws & ts < we
      if (sum(win_mask) < 4) next

      row <- tibble(
        video            = vid,
        person_idx       = pid,
        window_start_sec = round(ws, 4),
        window_end_sec   = round(we, 4)
      )

      win_ts <- ts[win_mask]

      for (side in c("left", "right")) {
        wr_x <- as.numeric(sub_df[[bcol(side, "wrist", "x")]])[win_mask]
        wr_y <- as.numeric(sub_df[[bcol(side, "wrist", "y")]])[win_mask]
        wr_z <- as.numeric(sub_df[[bcol(side, "wrist", "z")]])[win_mask]

        wrist <- trajectory_metrics(win_ts, wr_x, wr_y, wr_z, fs)

        row[[paste0(side, "_wrist_sal")]]                 <- wrist$sal
        row[[paste0(side, "_wrist_velocity_mean")]]       <- wrist$v_mean
        row[[paste0(side, "_wrist_velocity_peak")]]       <- wrist$v_peak
        row[[paste0(side, "_wrist_normalized_jerk")]]     <- wrist$nj
        row[[paste0(side, "_wrist_movement_efficiency")]] <- wrist$efficiency

        # Fingertip (index tip, hand landmark 8) normalized jerk.
        ft_x <- as.numeric(sub_df[[hcol(side, 8, "x")]])[win_mask]
        ft_y <- as.numeric(sub_df[[hcol(side, 8, "y")]])[win_mask]
        ft_z <- as.numeric(sub_df[[hcol(side, 8, "z")]])[win_mask]

        row[[paste0(side, "_fingertip_normalized_jerk")]] <-
          trajectory_metrics(win_ts, ft_x, ft_y, ft_z, fs)$nj
      }

      # Body-mode-only metrics (CPI + trunk/torso — require hip keypoints).
      if (tracking == "body") {
        win_ff <- sub_ff[win_mask, ]
        # Per-frame trunk lean is already mode-appropriate (2D image
        # plane or 3D world frame) — reuse instead of recomputing.
        lean <- win_ff$trunk_lean_deg
        reach <- pmax(win_ff$left_reach_raw, win_ff$right_reach_raw,
                      na.rm = TRUE)

        row[["compensatory_pattern_index"]] <-
          if (sum(!is.na(lean) & !is.na(reach)) >= 5)
            cor(lean, reach, use = "complete.obs")
          else NA_real_

        # Trunk/torso windowed summaries from per-frame values.
        tl  <- win_ff$trunk_lean_deg
        tls <- win_ff$trunk_lean_sagittal_deg
        tll <- win_ff$trunk_lean_lateral_deg
        tr  <- win_ff$trunk_rotation_deg
        ps  <- win_ff$posture_symmetry

        safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
        safe_sd   <- function(x) if (all(is.na(x))) NA_real_ else sd(x, na.rm = TRUE)

        row[["trunk_lean_mean"]]          <- safe_mean(tl)
        row[["trunk_lean_sd"]]            <- safe_sd(tl)
        row[["trunk_lean_range"]]         <- if (all(is.na(tl))) NA_real_
                                             else diff(range(tl, na.rm = TRUE))
        row[["trunk_lean_sagittal_mean"]] <- safe_mean(tls)
        row[["trunk_lean_sagittal_sd"]]   <- safe_sd(tls)
        row[["trunk_lean_lateral_mean"]]  <- safe_mean(tll)
        row[["trunk_lean_lateral_sd"]]    <- safe_sd(tll)
        row[["trunk_rotation_mean"]]      <- safe_mean(tr)
        row[["trunk_rotation_sd"]]        <- safe_sd(tr)
        row[["posture_symmetry_mean"]]    <- safe_mean(ps)
        row[["posture_symmetry_sd"]]      <- safe_sd(ps)
      } else {
        row[["compensatory_pattern_index"]]  <- NA_real_
        row[["trunk_lean_mean"]]             <- NA_real_
        row[["trunk_lean_sd"]]               <- NA_real_
        row[["trunk_lean_range"]]            <- NA_real_
        row[["trunk_lean_sagittal_mean"]]    <- NA_real_
        row[["trunk_lean_sagittal_sd"]]      <- NA_real_
        row[["trunk_lean_lateral_mean"]]     <- NA_real_
        row[["trunk_lean_lateral_sd"]]       <- NA_real_
        row[["trunk_rotation_mean"]]         <- NA_real_
        row[["trunk_rotation_sd"]]           <- NA_real_
        row[["posture_symmetry_mean"]]       <- NA_real_
        row[["posture_symmetry_sd"]]         <- NA_real_
      }

      # Bilateral comparison for window metrics.
      for (metric in WINDOW_BILATERAL_METRICS) {
        bl <- compute_bilateral(
          row[[paste0("left_", metric)]],
          row[[paste0("right_", metric)]]
        )
        row[[paste0(metric, "_symmetry_ratio")]]  <- bl$symmetry_ratio
        row[[paste0(metric, "_dominance_index")]]  <- bl$dominance_index
        row[[paste0(metric, "_abs_diff")]]         <- bl$abs_diff
      }

      ri <- ri + 1L
      results[[ri]] <- row
    }
  }

  if (ri == 0L) return(window_schema())
  bind_rows(results[seq_len(ri)])
}

# ------------------------------------------------------------------
# Movement phase segmentation
# ------------------------------------------------------------------

#' Running median filter (smooths a time series while preserving edges).
#'
#' @param x Numeric vector.
#' @param k Window width (uses floor(k/2) on each side).
#' @return Smoothed numeric vector of same length as \code{x}.
running_median <- function(x, k = 5L) {
  n <- length(x)
  if (n == 0L) return(x)
  half <- as.integer(floor(k / 2))
  out <- numeric(n)
  for (i in seq_len(n)) {
    lo <- max(1L, i - half)
    hi <- min(n, i + half)
    out[i] <- median(x[lo:hi], na.rm = TRUE)
  }
  out
}

#' Classify frames within a movement into REACH/GRASP/TRANSPORT/RELEASE.
#'
#' Uses smoothed grasp-aperture derivative to detect grasp (closing) and
#' release (opening) events. Without aperture data or insufficient aperture
#' variation, the entire movement is labelled REACH (pointing task).
#'
#' State machine: REACH -> GRASP -> TRANSPORT -> RELEASE.
#' Transitions may be skipped (e.g. no aperture change -> REACH only).
#'
#' @param speed_seg Numeric vector — smoothed speed per frame within movement.
#' @param aperture_seg Numeric vector — grasp aperture (thumb-index distance).
#' @param speed_thresh Scalar — speed threshold used for movement detection.
#' @param min_phase_frames Integer — minimum consecutive frames to trigger a
#'   phase transition (debounce).
#' @return Character vector of phase labels, same length as \code{speed_seg}.
classify_movement_phases <- function(speed_seg, aperture_seg,
                                     speed_thresh,
                                     min_phase_frames = 3L) {
  m <- length(speed_seg)
  phases <- rep("REACH", m)

  if (all(is.na(aperture_seg)) || m < min_phase_frames * 2L) return(phases)

  # Smooth aperture; fill NAs with LOCF then NOCB.
  ap <- running_median(aperture_seg, 3L)
  for (i in 2:m) {
    if (is.na(ap[i]) && !is.na(ap[i - 1L])) ap[i] <- ap[i - 1L]
  }
  if (is.na(ap[1L])) {
    first_valid <- which(!is.na(ap))[1L]
    if (is.na(first_valid)) return(phases)
    ap[seq_len(first_valid - 1L)] <- ap[first_valid]
  }
  if (any(is.na(ap))) return(phases)

  # Smoothed aperture derivative.
  ap_d <- c(0, diff(ap))
  ap_d <- running_median(ap_d, 3L)

  # Adaptive threshold: 5% of aperture range within the movement.
  ap_range <- diff(range(ap, na.rm = TRUE))
  if (ap_range < 1e-8) return(phases)
  ap_thresh <- ap_range * 0.05

  # --- Find GRASP: first sustained run of ap_d < -ap_thresh ---
  grasp_start <- NA_integer_
  grasp_end <- NA_integer_
  run_len <- 0L
  for (i in seq_len(m)) {
    if (!is.na(ap_d[i]) && ap_d[i] < -ap_thresh) {
      run_len <- run_len + 1L
      if (run_len >= min_phase_frames && is.na(grasp_start)) {
        grasp_start <- i - min_phase_frames + 1L
      }
    } else {
      if (!is.na(grasp_start) && is.na(grasp_end)) grasp_end <- i - 1L
      run_len <- 0L
    }
  }
  if (!is.na(grasp_start) && is.na(grasp_end)) grasp_end <- m

  if (is.na(grasp_start)) return(phases)
  phases[grasp_start:grasp_end] <- "GRASP"

  if (grasp_end >= m) return(phases)

  # --- Find RELEASE: sustained run of ap_d > ap_thresh after GRASP ---
  release_start <- NA_integer_
  run_len <- 0L
  for (i in (grasp_end + 1L):m) {
    if (!is.na(ap_d[i]) && ap_d[i] > ap_thresh) {
      run_len <- run_len + 1L
      if (run_len >= min_phase_frames && is.na(release_start)) {
        release_start <- i - min_phase_frames + 1L
      }
    } else {
      run_len <- 0L
    }
  }

  if (!is.na(release_start)) {
    if (release_start - grasp_end >= min_phase_frames) {
      phases[(grasp_end + 1L):(release_start - 1L)] <- "TRANSPORT"
    }
    phases[release_start:m] <- "RELEASE"
  } else if (grasp_end < m) {
    phases[(grasp_end + 1L):m] <- "TRANSPORT"
  }

  phases
}

#' Detect and segment movements from landmark data.
#'
#' Velocity-profile segmentation of wrist trajectory with sub-phase
#' classification via grasp-aperture analysis. Produces one row per
#' phase per movement per side per person.
#'
#' Algorithm:
#'   1. Compute wrist speed, smooth with running median.
#'   2. Detect above-threshold segments (RLE), merge close ones, reject
#'      short ones.
#'   3. Within each movement, classify phases via aperture derivative.
#'   4. Extract per-phase features (velocity, path, NJ, SAL, symmetry).
#'
#' @param df Data frame — raw landmark CSV (from read_csv).
#' @param frame_features Data frame — output of compute_frame_features().
#' @param tracking Character — tracking mode ("body" or "hands-arms").
#' @param speed_thresh_pct Fraction of peak speed for onset/offset (0.05).
#' @param min_movement_frames Minimum frames to count as a movement (5).
#' @param min_gap_frames Maximum gap between segments before merging (3).
#' @param median_k Running-median filter width for speed smoothing (5).
#' @param min_phase_frames Minimum frames for a sub-phase (3).
#' Zero-row phase table carrying the full ordered schema.
#'
#' Returned when no movement is detected, so a static capture publishes a
#' shaped empty result instead of nothing.  A test pins this header against
#' the populated one to keep the two paths from drifting.
phase_schema <- function() {
  tibble(
    video                  = character(),
    person_idx             = integer(),
    side                   = character(),
    movement_idx           = integer(),
    phase                  = character(),
    start_frame            = integer(),
    end_frame              = integer(),
    duration_sec           = double(),
    peak_velocity          = double(),
    mean_velocity          = double(),
    path_length            = double(),
    smoothness_nj          = double(),
    smoothness_sal         = double(),
    mean_reach_symmetry    = double(),
    movement_duration_sec  = double(),
    movement_n_phases      = integer(),
    movement_peak_velocity = double(),
    movement_path_length   = double(),
    movement_efficiency    = double()
  )
}

#' @return Tibble with one row per phase. Zero-row schema if no movements.
segment_movements <- function(df, frame_features, tracking,
                              speed_thresh_pct = 0.05,
                              min_movement_frames = 5L,
                              min_gap_frames = 3L,
                              median_k = 5L,
                              min_phase_frames = 3L) {
  bcol <- function(side, kp, coord) body_col(tracking, side, kp, coord)

  groups <- frame_features |>
    select(video, person_idx) |>
    distinct()

  all_rows <- list()
  ri <- 0L

  for (g in seq_len(nrow(groups))) {
    vid <- groups$video[g]
    pid <- groups$person_idx[g]

    mask <- df$video == vid & as.integer(df$person_idx) == pid
    sub_df <- df[mask, ]
    sub_ff <- frame_features[mask, ]

    ts <- as.numeric(sub_df$timestamp_sec)
    frame_idxs <- as.integer(sub_df$frame_idx)
    n <- length(ts)
    if (n < min_movement_frames) next

    dt_median <- median(diff(ts), na.rm = TRUE)
    if (is.na(dt_median) || dt_median <= 0) next
    fs <- 1 / dt_median

    for (side in c("left", "right")) {
      wr_x <- as.numeric(sub_df[[bcol(side, "wrist", "x")]])
      wr_y <- as.numeric(sub_df[[bcol(side, "wrist", "y")]])
      wr_z <- as.numeric(sub_df[[bcol(side, "wrist", "z")]])
      if (all(is.na(wr_x))) next

      # Speed (coord-units/sec); NA → 0 for threshold comparison.
      dx <- c(0, diff(wr_x))
      dy <- c(0, diff(wr_y))
      dz <- c(0, diff(wr_z))
      speed_raw <- sqrt(dx^2 + dy^2 + dz^2) * fs
      speed_raw[is.na(speed_raw)] <- 0
      speed <- running_median(speed_raw, median_k)

      peak_speed <- max(speed)
      if (peak_speed < 1e-10) next
      speed_thresh <- peak_speed * speed_thresh_pct

      # --- Detect active segments via RLE ---
      active <- speed > speed_thresh
      rle_res <- rle(active)
      cum_len <- cumsum(rle_res$lengths)
      seg_starts <- c(1L, cum_len[-length(cum_len)] + 1L)

      active_idx <- which(rle_res$values)
      if (length(active_idx) == 0L) next

      segs <- data.frame(
        start = seg_starts[active_idx],
        end   = cum_len[active_idx]
      )

      # Merge segments separated by <= min_gap_frames.
      if (nrow(segs) > 1L) {
        merged <- list(segs[1L, ])
        for (i in 2:nrow(segs)) {
          last <- merged[[length(merged)]]
          if (segs$start[i] - last$end <= min_gap_frames) {
            merged[[length(merged)]]$end <- segs$end[i]
          } else {
            merged[[length(merged) + 1L]] <- segs[i, ]
          }
        }
        segs <- do.call(rbind, merged)
      }

      # Reject short segments.
      segs <- segs[segs$end - segs$start + 1L >= min_movement_frames,
                   , drop = FALSE]
      if (nrow(segs) == 0L) next

      # Aperture and bilateral symmetry vectors for this person × side.
      aperture <- as.numeric(
        sub_ff[[paste0(side, "_grasp_aperture_thumb_index")]]
      )
      reach_sym_col <- "reach_raw_symmetry_ratio"
      reach_sym <- if (reach_sym_col %in% names(sub_ff))
        as.numeric(sub_ff[[reach_sym_col]]) else rep(NA_real_, n)

      # --- Process each movement ---
      movement_idx <- 0L
      for (s in seq_len(nrow(segs))) {
        movement_idx <- movement_idx + 1L
        si <- segs$start[s]
        ei <- segs$end[s]
        seg_range <- si:ei

        speed_seg    <- speed[seg_range]
        aperture_seg <- aperture[seg_range]
        wr_x_seg     <- wr_x[seg_range]
        wr_y_seg     <- wr_y[seg_range]
        wr_z_seg     <- wr_z[seg_range]
        ts_seg       <- ts[seg_range]
        fi_seg       <- frame_idxs[seg_range]

        # Phase classification.
        phase_labels <- classify_movement_phases(
          speed_seg, aperture_seg, speed_thresh, min_phase_frames
        )

        # Per-movement summary.
        mvmt_dur      <- ts_seg[length(ts_seg)] - ts_seg[1L]
        mvmt_peak_vel <- max(speed_seg, na.rm = TRUE)
        mvmt_path     <- sum(sqrt(diff(wr_x_seg)^2 + diff(wr_y_seg)^2 +
                                  diff(wr_z_seg)^2), na.rm = TRUE)
        mvmt_eff      <- movement_efficiency(wr_x_seg, wr_y_seg, wr_z_seg)

        # Collapse consecutive same-phase frames into phase segments.
        phase_rle <- rle(phase_labels)
        n_phases  <- length(phase_rle$lengths)
        phase_cum <- cumsum(phase_rle$lengths)
        phase_s   <- c(1L, phase_cum[-n_phases] + 1L)

        for (p in seq_len(n_phases)) {
          pi_s <- phase_s[p]
          pi_e <- phase_cum[p]
          p_range    <- pi_s:pi_e
          orig_range <- seg_range[p_range]

          p_speed <- speed_seg[p_range]
          p_wr_x  <- wr_x_seg[p_range]
          p_wr_y  <- wr_y_seg[p_range]
          p_wr_z  <- wr_z_seg[p_range]
          p_ts    <- ts_seg[p_range]

          p_dur <- if (length(p_ts) > 1L) {
            p_ts[length(p_ts)] - p_ts[1L]
          } else 0

          p_path <- if (length(p_wr_x) > 1L) {
            sum(sqrt(diff(p_wr_x)^2 + diff(p_wr_y)^2 + diff(p_wr_z)^2),
                na.rm = TRUE)
          } else 0

          p_nj  <- normalized_jerk(p_wr_x, p_wr_y, p_wr_z, fs)
          p_sal <- spectral_arc_length(p_speed, fs)

          p_sym <- reach_sym[orig_range]
          p_mean_sym <- if (any(!is.na(p_sym))) {
            mean(p_sym, na.rm = TRUE)
          } else NA_real_

          ri <- ri + 1L
          all_rows[[ri]] <- tibble(
            video                 = vid,
            person_idx            = pid,
            side                  = side,
            movement_idx          = as.integer(movement_idx),
            phase                 = phase_rle$values[p],
            start_frame           = fi_seg[pi_s],
            end_frame             = fi_seg[pi_e],
            duration_sec          = round(p_dur, 4),
            peak_velocity         = round(max(p_speed, na.rm = TRUE), 6),
            mean_velocity         = round(mean(p_speed, na.rm = TRUE), 6),
            path_length           = round(p_path, 6),
            smoothness_nj         = if (!is.na(p_nj)) round(p_nj, 4)
                                    else NA_real_,
            smoothness_sal        = if (!is.na(p_sal)) round(p_sal, 4)
                                    else NA_real_,
            mean_reach_symmetry   = if (!is.na(p_mean_sym))
                                      round(p_mean_sym, 4)
                                    else NA_real_,
            movement_duration_sec = round(mvmt_dur, 4),
            movement_n_phases     = as.integer(n_phases),
            movement_peak_velocity = round(mvmt_peak_vel, 6),
            movement_path_length  = round(mvmt_path, 6),
            movement_efficiency   = if (!is.na(mvmt_eff)) round(mvmt_eff, 4)
                                    else NA_real_
          )
        }
      }
    }
  }

  if (ri == 0L) return(phase_schema())
  bind_rows(all_rows[seq_len(ri)])
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: Rscript analysis/clinical_features.R <landmark_csv_or_directory>")
}

path <- args[1]
if (dir.exists(path)) {
  files <- list.files(path, pattern = "\\.csv$", full.names = TRUE)
  files <- files[!str_detect(
    basename(files),
    paste0("(metrics|kp_detail|diag|summary|smooth|feature_rank|",
           "clinical[_a-z0-9]*|movement_phases[_a-z0-9]*)\\.csv$")
  )]
  if (length(files) == 0) stop("The directory contains no landmark CSVs: ", path)
} else {
  files <- path
}

for (f in files) {
  cat("\n", strrep("=", 60), "\n")
  cat("  Clinical features:", basename(f), "\n")
  cat(strrep("=", 60), "\n")

  df <- read_csv(f, show_col_types = FALSE)
  is_3d <- is_world3d(names(df))
  if (is_3d) {
    cat("  The script gates 3D input (world3d) with fusion diagnostics; units: m, m/s.\n")
    # `video` is the capture identity, so an ambiguous or blank one is
    # unusable and fails closed.  A row-less input carries no value at all;
    # that case publishes typed empties and is rejected downstream instead.
    captures <- unique(df$video)
    if (nrow(df) > 0 && (length(captures) != 1L || is.na(captures[1]) ||
                         !nzchar(trimws(captures[1])))) {
      stop("3D input must contain exactly one non-blank 'video' value; found ",
           length(captures), " distinct in ", basename(f))
    }
    df <- adapt_world3d(df)
  } else {
    df <- adapt_2d_confidence(df)
  }
  tracking <- detect_tracking(names(df))
  cat(sprintf("  Tracking mode: %s\n", tracking))

  if (tracking == "hands") {
    cat("  Hands-only mode has no arm keypoints. The script skips it.\n")
    next
  }

  cat(sprintf("  %d rows, %d columns\n", nrow(df), ncol(df)))

  stem <- str_remove(f, "\\.csv$")
  suffix <- if (is_3d) "_3d" else ""
  source_sha256 <- if (is_3d) file_sha256(f) else NA_character_

  # Per-frame features.
  cat("  The script computes per-frame features.\n")
  clinical <- compute_frame_features(df, tracking, is_3d = is_3d)

  # Tags go on the published copy only; the window and phase passes keep
  # reading the untagged frame features they were written against.
  frame_out <- if (is_3d) {
    attach_artifact_tags(clinical, "clinical-frame-3d",
                         QUALIFICATION_FRAME, source_sha256)
  } else {
    clinical
  }

  out_frame <- paste0(stem, "_clinical", suffix, ".csv")
  write_csv(frame_out, out_frame)
  cat(sprintf("  The script wrote %d rows to %s.\n", nrow(frame_out), basename(out_frame)))

  # Window-level smoothness features.  A 3D window artifact is always
  # published, empty or not, so a reader can tell a genuine zero-window
  # result from a run that never happened, and a stale file from an earlier
  # run cannot survive.  2D keeps its skip-if-empty behaviour untouched.
  cat("  The script computes window-level smoothness features.\n")
  windows <- compute_window_features(df, clinical, tracking)

  if (is_3d) {
    windows <- attach_artifact_tags(windows, "clinical-window-3d",
                                    QUALIFICATION_WINDOW, source_sha256)
  }
  if (is_3d || nrow(windows) > 0) {
    out_win <- paste0(stem, "_clinical", suffix, "_windows.csv")
    write_csv(windows, out_win)
    cat(sprintf("  The script wrote %d windows to %s.\n", nrow(windows), basename(out_win)))
  } else {
    cat("  The script produced no windows. The video may be too short.\n")
  }

  # Movement phase segmentation.  Phase metrics still differentiate across
  # tracking holes, so the artifact says so in metric_qualification rather
  # than leaving the caveat to the docs.
  cat("  The script segments movements.\n")
  phases <- segment_movements(df, clinical, tracking)

  if (is_3d) {
    phases <- attach_artifact_tags(phases, "movement-phase-3d",
                                   QUALIFICATION_PHASE, source_sha256)
  }
  if (is_3d || nrow(phases) > 0) {
    out_phases <- paste0(stem, "_movement_phases", suffix, ".csv")
    write_csv(phases, out_phases)
    cat(sprintf("  The script wrote %d phases to %s.\n", nrow(phases),
                basename(out_phases)))
  } else {
    cat("  The script detected no movements.\n")
  }

  cat("  The script finished.\n")
}

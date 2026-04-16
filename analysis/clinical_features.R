#!/usr/bin/env Rscript
# Clinical feature derivation from landmark CSVs produced by export.py.
#
# Computes clinically meaningful kinematic features per frame (joint
# angles, reach, grasp aperture, displacement) and per sliding window
# (spectral arc length, velocity statistics), writing two CSVs per
# input.
#
# Usage:
#   Rscript analysis/clinical_features.R output/video1_hands-arms.csv
#   Rscript analysis/clinical_features.R output/   # all landmark CSVs
#
# Outputs alongside each input CSV:
#   <stem>_clinical.csv          — per-frame clinical features
#   <stem>_clinical_windows.csv  — per-window smoothness features

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
spectral_arc_length <- function(v, fs) {
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

  # Adaptive cutoff: SAL_FREQ_CUTOFF or Nyquist, whichever is lower.
  fc <- min(SAL_FREQ_CUTOFF, fs / 2)
  keep <- freqs <= fc
  V     <- V[keep]
  freqs <- freqs[keep]
  if (length(freqs) < 2) return(NA_real_)

  # Arc length of the normalised magnitude spectrum.
  dfreq <- diff(freqs) / fc
  dV    <- diff(V)
  -sum(sqrt(dfreq^2 + dV^2))
}

# ------------------------------------------------------------------
# Per-frame feature computation
# ------------------------------------------------------------------

compute_frame_features <- function(df, tracking) {
  prefix <- if (tracking == "body") "body" else "arm"

  bcol <- function(side, kp, coord) {
    paste0(prefix, "_", side, "_", kp, "_", coord)
  }
  hcol <- function(side, idx, coord) {
    paste0(side, "_hand_", idx, "_", coord)
  }

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

  result
}

# ------------------------------------------------------------------
# Window-level smoothness features
# ------------------------------------------------------------------

compute_window_features <- function(df, frame_features, tracking,
                                    window_sec = WINDOW_SEC) {
  prefix <- if (tracking == "body") "body" else "arm"
  bcol <- function(side, kp, coord) {
    paste0(prefix, "_", side, "_", kp, "_", coord)
  }

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

    ts <- as.numeric(sub_df$timestamp_sec)
    n  <- length(ts)
    if (n < 4) next

    dt_median <- median(diff(ts), na.rm = TRUE)
    if (is.na(dt_median) || dt_median <= 0) next
    fs <- 1 / dt_median

    t_start <- ts[1]
    t_end   <- ts[n]

    # 50 %-overlapping windows.
    if (t_end - t_start < window_sec) next
    win_starts <- seq(t_start, t_end - window_sec, by = window_sec / 2)
    if (length(win_starts) == 0) next

    for (ws in win_starts) {
      we <- ws + window_sec
      win_mask <- ts >= ws & ts < we
      if (sum(win_mask) < 4) next

      row <- tibble(
        video            = vid,
        person_idx       = pid,
        window_start_sec = round(ws, 4),
        window_end_sec   = round(we, 4)
      )

      for (side in c("left", "right")) {
        wr_x <- as.numeric(sub_df[[bcol(side, "wrist", "x")]])[win_mask]
        wr_y <- as.numeric(sub_df[[bcol(side, "wrist", "y")]])[win_mask]
        wr_z <- as.numeric(sub_df[[bcol(side, "wrist", "z")]])[win_mask]

        if (all(is.na(wr_x))) {
          row[[paste0(side, "_wrist_sal")]]           <- NA_real_
          row[[paste0(side, "_wrist_velocity_mean")]]  <- NA_real_
          row[[paste0(side, "_wrist_velocity_peak")]]  <- NA_real_
          next
        }

        dx <- diff(wr_x);  dy <- diff(wr_y);  dz <- diff(wr_z)
        speed <- sqrt(dx^2 + dy^2 + dz^2) * fs

        row[[paste0(side, "_wrist_sal")]] <-
          spectral_arc_length(speed, fs)
        row[[paste0(side, "_wrist_velocity_mean")]] <-
          mean(speed, na.rm = TRUE)
        row[[paste0(side, "_wrist_velocity_peak")]] <-
          max(speed, na.rm = TRUE)
      }

      ri <- ri + 1L
      results[[ri]] <- row
    }
  }

  if (ri == 0L) return(tibble())
  bind_rows(results[seq_len(ri)])
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
    "(metrics|kp_detail|diag|summary|smooth|feature_rank|clinical[_a-z]*)\\.csv$"
  )]
  if (length(files) == 0) stop("No landmark CSVs found in ", path)
} else {
  files <- path
}

for (f in files) {
  cat("\n", strrep("=", 60), "\n")
  cat("  Clinical features:", basename(f), "\n")
  cat(strrep("=", 60), "\n")

  df <- read_csv(f, show_col_types = FALSE)
  tracking <- detect_tracking(names(df))
  cat(sprintf("  Tracking mode: %s\n", tracking))

  if (tracking == "hands") {
    cat("  Hands-only mode has no arm keypoints — skipping.\n")
    next
  }

  cat(sprintf("  %d rows, %d columns\n", nrow(df), ncol(df)))

  # Per-frame features.
  cat("  Computing per-frame features...\n")
  clinical <- compute_frame_features(df, tracking)

  stem <- str_remove(f, "\\.csv$")
  out_frame <- paste0(stem, "_clinical.csv")
  write_csv(clinical, out_frame)
  cat(sprintf("  Wrote %d rows → %s\n", nrow(clinical), basename(out_frame)))

  # Window-level smoothness features.
  cat("  Computing window-level smoothness features...\n")
  windows <- compute_window_features(df, clinical, tracking)

  if (nrow(windows) > 0) {
    out_win <- paste0(stem, "_clinical_windows.csv")
    write_csv(windows, out_win)
    cat(sprintf("  Wrote %d windows → %s\n", nrow(windows), basename(out_win)))
  } else {
    cat("  No windows produced (video may be too short).\n")
  }

  cat("  Done.\n")
}

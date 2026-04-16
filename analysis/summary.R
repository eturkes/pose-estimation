#!/usr/bin/env Rscript
# Baseline quality report from *_metrics.csv files.
#
# Usage:
#   Rscript analysis/summary.R output/video1_metrics.csv
#   Rscript analysis/summary.R output/   # all *_metrics.csv in directory
#
# Outputs:
#   - Console text report
#   - <stem>_summary.json alongside each input CSV

library(tidyverse)
library(jsonlite)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

safe_median <- function(x) median(as.numeric(x), na.rm = TRUE)
safe_mean   <- function(x) mean(as.numeric(x), na.rm = TRUE)
safe_q95    <- function(x) quantile(as.numeric(x), 0.95, na.rm = TRUE, names = FALSE)
safe_iqr    <- function(x) IQR(as.numeric(x), na.rm = TRUE)
# max() warns and returns -Inf on all-NA input (common when one hand is
# never matched in a video); return NA instead.
safe_max    <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
}

summarise_metrics <- function(df) {
  n <- nrow(df)

  # Detection rates
  body_det_rate <- mean(df$body_detected, na.rm = TRUE)
  hands_real_mean <- safe_mean(df$n_hands_real)
  hands_synth_pct <- safe_mean(df$n_hands_synthetic > 0)
  hands_recrop_pct <- safe_mean(df$n_hands_recrop > 0)

  # Confidence
  body_vis_median <- safe_median(df$body_vis_mean)
  body_vis_iqr    <- safe_iqr(df$body_vis_mean)
  hand_L_flag_med <- safe_median(df$hand_L_flag)
  hand_R_flag_med <- safe_median(df$hand_R_flag)

  # Jitter
  body_jitter_med  <- safe_median(df$body_jitter_px)
  body_jitter_p95  <- safe_q95(df$body_jitter_px)
  hand_L_jitter_med <- safe_median(df$hand_L_jitter_px)
  hand_L_jitter_p95 <- safe_q95(df$hand_L_jitter_px)
  hand_R_jitter_med <- safe_median(df$hand_R_jitter_px)
  hand_R_jitter_p95 <- safe_q95(df$hand_R_jitter_px)

  # Smoothing delta
  body_smooth_med  <- safe_median(df$body_smooth_delta_px)
  hand_L_smooth_med <- safe_median(df$hand_L_smooth_delta_px)
  hand_R_smooth_med <- safe_median(df$hand_R_smooth_delta_px)

  # Carry-forward
  body_carry_pct <- mean(df$body_carry, na.rm = TRUE)
  carry_runs <- rle(as.logical(df$body_carry))
  carry_streaks <- carry_runs$lengths[carry_runs$values]
  carry_mean_len <- if (length(carry_streaks) > 0) mean(carry_streaks) else 0

  # Constraints
  bone_corr_mean <- safe_mean(df$bone_correction_px)
  angle_corr_pct <- mean(df$angle_corrections_n > 0, na.rm = TRUE)

  # Matching
  match_L_mean <- safe_mean(df$hand_arm_match_dist_L)
  match_R_mean <- safe_mean(df$hand_arm_match_dist_R)
  match_L_max  <- safe_max(df$hand_arm_match_dist_L)
  match_R_max  <- safe_max(df$hand_arm_match_dist_R)

  # Inference speed
  inference_med <- safe_median(df$inference_ms)
  inference_p95 <- safe_q95(df$inference_ms)

  list(
    n_frames = n,
    detection = list(
      body_detection_rate = round(body_det_rate, 4),
      mean_real_hands_per_frame = round(hands_real_mean, 2),
      pct_frames_synthetic_hands = round(hands_synth_pct, 4),
      pct_frames_recrop_hands = round(hands_recrop_pct, 4)
    ),
    confidence = list(
      body_vis_median = round(body_vis_median, 4),
      body_vis_iqr = round(body_vis_iqr, 4),
      hand_L_flag_median = round(hand_L_flag_med, 4),
      hand_R_flag_median = round(hand_R_flag_med, 4)
    ),
    jitter_px = list(
      body_median = round(body_jitter_med, 2),
      body_p95 = round(body_jitter_p95, 2),
      hand_L_median = round(hand_L_jitter_med, 2),
      hand_L_p95 = round(hand_L_jitter_p95, 2),
      hand_R_median = round(hand_R_jitter_med, 2),
      hand_R_p95 = round(hand_R_jitter_p95, 2)
    ),
    smoothing_delta_px = list(
      body_median = round(body_smooth_med, 2),
      hand_L_median = round(hand_L_smooth_med, 2),
      hand_R_median = round(hand_R_smooth_med, 2)
    ),
    carry_forward = list(
      body_carry_pct = round(body_carry_pct, 4),
      mean_carry_streak = round(carry_mean_len, 1)
    ),
    constraints = list(
      bone_correction_mean_px = round(bone_corr_mean, 2),
      pct_frames_angle_clamped = round(angle_corr_pct, 4)
    ),
    matching_px = list(
      hand_L_mean = round(match_L_mean, 2),
      hand_R_mean = round(match_R_mean, 2),
      hand_L_max = round(match_L_max, 2),
      hand_R_max = round(match_R_max, 2)
    ),
    inference_ms = list(
      median = round(inference_med, 1),
      p95 = round(inference_p95, 1)
    )
  )
}

print_summary <- function(s, video_name) {
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat("  Quality Report:", video_name, "\n")
  cat(strrep("=", 60), "\n")
  cat(sprintf("  Frames analysed:  %d\n", s$n_frames))

  cat("\n  --- Detection ---\n")
  cat(sprintf("  Body detection rate:       %.1f%%\n", s$detection$body_detection_rate * 100))
  cat(sprintf("  Mean real hands / frame:   %.2f\n", s$detection$mean_real_hands_per_frame))
  cat(sprintf("  Frames with synth hands:   %.1f%%\n", s$detection$pct_frames_synthetic_hands * 100))
  cat(sprintf("  Frames with recrop hands:  %.1f%%\n", s$detection$pct_frames_recrop_hands * 100))

  cat("\n  --- Confidence ---\n")
  cat(sprintf("  Body visibility (median):  %.3f  (IQR %.3f)\n",
              s$confidence$body_vis_median, s$confidence$body_vis_iqr))
  cat(sprintf("  Hand L flag (median):      %.3f\n", s$confidence$hand_L_flag_median))
  cat(sprintf("  Hand R flag (median):      %.3f\n", s$confidence$hand_R_flag_median))

  cat("\n  --- Jitter (px, sum over keypoints) ---\n")
  cat(sprintf("  Body:   median %.1f   p95 %.1f\n", s$jitter_px$body_median, s$jitter_px$body_p95))
  cat(sprintf("  Hand L: median %.1f   p95 %.1f\n", s$jitter_px$hand_L_median, s$jitter_px$hand_L_p95))
  cat(sprintf("  Hand R: median %.1f   p95 %.1f\n", s$jitter_px$hand_R_median, s$jitter_px$hand_R_p95))

  cat("\n  --- Smoothing delta (px, raw vs smoothed) ---\n")
  cat(sprintf("  Body median:   %.1f\n", s$smoothing_delta_px$body_median))
  cat(sprintf("  Hand L median: %.1f\n", s$smoothing_delta_px$hand_L_median))
  cat(sprintf("  Hand R median: %.1f\n", s$smoothing_delta_px$hand_R_median))

  cat("\n  --- Carry-forward ---\n")
  cat(sprintf("  Body carry %%:         %.1f%%\n", s$carry_forward$body_carry_pct * 100))
  cat(sprintf("  Mean carry streak:    %.1f frames\n", s$carry_forward$mean_carry_streak))

  cat("\n  --- Constraints ---\n")
  cat(sprintf("  Bone correction (mean px): %.2f\n", s$constraints$bone_correction_mean_px))
  cat(sprintf("  Angle-clamped frames:      %.1f%%\n", s$constraints$pct_frames_angle_clamped * 100))

  cat("\n  --- Matching (hand-arm distance, px) ---\n")
  cat(sprintf("  Left:  mean %.1f  max %.1f\n", s$matching_px$hand_L_mean, s$matching_px$hand_L_max))
  cat(sprintf("  Right: mean %.1f  max %.1f\n", s$matching_px$hand_R_mean, s$matching_px$hand_R_max))

  cat("\n  --- Inference speed ---\n")
  cat(sprintf("  Median: %.1f ms   p95: %.1f ms\n", s$inference_ms$median, s$inference_ms$p95))
  cat(strrep("=", 60), "\n\n")
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
# Guarded so compare.R (which sources this file to reuse summarise_metrics)
# does not re-execute the CLI flow in its own commandArgs.

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    stop("Usage: Rscript analysis/summary.R <metrics_csv_or_directory>")
  }

  path <- args[1]
  if (dir.exists(path)) {
    files <- list.files(path, pattern = "_metrics\\.csv$", full.names = TRUE)
    if (length(files) == 0) stop("No *_metrics.csv files found in ", path)
  } else {
    files <- path
  }

  for (f in files) {
    df <- read_csv(f, show_col_types = FALSE)
    video_name <- str_extract(basename(f), "^(.+)_metrics\\.csv$", group = 1)
    if (is.na(video_name)) video_name <- basename(f)

    s <- summarise_metrics(df)
    print_summary(s, video_name)

    json_path <- str_replace(f, "_metrics\\.csv$", "_summary.json")
    write_json(s, json_path, pretty = TRUE, auto_unbox = TRUE)
    cat("  Wrote:", json_path, "\n")
  }
}

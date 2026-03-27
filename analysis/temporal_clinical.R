#!/usr/bin/env Rscript
# Temporal patterns of clinical kinematic features within each video.
#
# For each video with ≥10 per-frame rows, produces a multi-panel
# time-series plot with left and right sides overlaid.  Window-level
# features (SAL, velocity) are shown in additional panels when
# available.  A summary overview PNG compares one key feature across
# all qualifying videos.
#
# Usage:
#   Rscript analysis/temporal_clinical.R output/
#
# Outputs (written alongside the input CSVs):
#   <stem>_clinical_timeseries.png         — per-video time series
#   all_clinical_timeseries_overview.png   — cross-video comparison

library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(purrr)
library(ggplot2)
library(patchwork)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

FRAME_META  <- c("video", "frame_idx", "timestamp_sec", "person_idx")
WINDOW_META <- c("video", "person_idx", "window_start_sec", "window_end_sec")
MIN_ROWS    <- 10

# Map stripped feature names to panel titles
FEATURE_LABELS <- c(
  elbow_angle_deg              = "Elbow Angle (deg)",
  wrist_deviation_deg          = "Wrist Deviation (deg)",
  finger_spread_deg            = "Finger Spread (deg)",
  reach_raw                    = "Reach (raw)",
  reach_norm                   = "Reach (normalised)",
  grasp_aperture_thumb_index   = "Grasp Aperture (thumb-index)",
  grasp_aperture_thumb_pinky   = "Grasp Aperture (thumb-pinky)",
  wrist_displacement           = "Wrist Displacement",
  fingertip_displacement       = "Fingertip Displacement"
)

WINDOW_LABELS <- c(
  wrist_sal           = "Wrist SAL",
  wrist_velocity_mean = "Wrist Velocity (mean)",
  wrist_velocity_peak = "Wrist Velocity (peak)"
)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript analysis/temporal_clinical.R <output_dir>")
}
out_dir <- args[1]
if (!dir.exists(out_dir)) stop("Directory not found: ", out_dir)

# ------------------------------------------------------------------
# Load per-frame files
# ------------------------------------------------------------------

frame_files <- list.files(out_dir, pattern = "_clinical\\.csv$",
                          full.names = TRUE)
frame_files <- frame_files[!str_detect(basename(frame_files), "_windows\\.csv$")]

if (length(frame_files) == 0) stop("No *_clinical.csv files found in ", out_dir)

frames_by_video <- map(frame_files, \(f) {
  d <- read_csv(f, show_col_types = FALSE)
  if (nrow(d) == 0) return(NULL)
  d
}) |> set_names(frame_files) |> compact()

# ------------------------------------------------------------------
# Load per-window files
# ------------------------------------------------------------------

win_files <- list.files(out_dir, pattern = "_clinical_windows\\.csv$",
                        full.names = TRUE)
wins_by_video <- map(win_files, \(f) {
  d <- read_csv(f, show_col_types = FALSE)
  if (nrow(d) == 0) return(NULL)
  d
}) |> set_names(win_files) |> compact()

# Key windows by video name for easy lookup
win_lookup <- list()
for (nm in names(wins_by_video)) {
  vids <- unique(wins_by_video[[nm]]$video)
  for (v in vids) win_lookup[[v]] <- wins_by_video[[nm]] |> filter(video == v)
}

# ------------------------------------------------------------------
# Helper: pivot per-frame data into long form with side + feature
# ------------------------------------------------------------------

pivot_frame_long <- function(df) {
  feat_cols <- setdiff(names(df), FRAME_META)
  feat_cols <- feat_cols[map_lgl(feat_cols, \(c) is.numeric(df[[c]]))]

  df |>
    select(video, timestamp_sec, all_of(feat_cols)) |>
    pivot_longer(-c(video, timestamp_sec),
                 names_to = "raw_feature", values_to = "value") |>
    filter(!is.na(value)) |>
    mutate(
      side = case_when(
        str_starts(raw_feature, "left_")  ~ "Left",
        str_starts(raw_feature, "right_") ~ "Right",
        TRUE ~ "Both"
      ),
      feature = str_remove(raw_feature, "^(left|right)_"),
      label = FEATURE_LABELS[feature]
    ) |>
    filter(!is.na(label))
}

# ------------------------------------------------------------------
# Helper: pivot window data into long form
# ------------------------------------------------------------------

pivot_window_long <- function(df) {
  feat_cols <- setdiff(names(df), WINDOW_META)
  feat_cols <- feat_cols[map_lgl(feat_cols, \(c) is.numeric(df[[c]]))]

  df |>
    select(video, window_start_sec, all_of(feat_cols)) |>
    pivot_longer(-c(video, window_start_sec),
                 names_to = "raw_feature", values_to = "value") |>
    filter(!is.na(value)) |>
    mutate(
      side = case_when(
        str_starts(raw_feature, "left_")  ~ "Left",
        str_starts(raw_feature, "right_") ~ "Right",
        TRUE ~ "Both"
      ),
      feature = str_remove(raw_feature, "^(left|right)_"),
      label = WINDOW_LABELS[feature]
    ) |>
    filter(!is.na(label))
}

# ------------------------------------------------------------------
# Per-video time-series plots
# ------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("  Temporal Clinical Features\n")
cat(strrep("=", 60), "\n\n")

all_long <- list()
plot_count <- 0
skip_count <- 0

for (f in names(frames_by_video)) {
  df <- frames_by_video[[f]]
  vid <- unique(df$video)[1]

  if (nrow(df) < MIN_ROWS) {
    cat(sprintf("  %-30s  %3d rows — skipped (<%d)\n", vid, nrow(df), MIN_ROWS))
    skip_count <- skip_count + 1
    next
  }

  cat(sprintf("  %-30s  %3d rows", vid, nrow(df)))
  long <- pivot_frame_long(df)
  all_long[[vid]] <- long

  # Per-frame panels
  p_frame <- ggplot(long, aes(timestamp_sec, value, colour = side)) +
    geom_line(linewidth = 0.4, alpha = 0.8) +
    facet_wrap(~label, scales = "free_y", ncol = 3) +
    scale_colour_manual(values = c(Left = "#1f77b4", Right = "#d62728")) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "bottom",
          strip.text = element_text(size = 8),
          axis.text = element_text(size = 7)) +
    labs(x = "Time (s)", y = NULL, colour = NULL)

  # Window panels (if available)
  win_df <- win_lookup[[vid]]
  has_win <- !is.null(win_df) && nrow(win_df) > 0

  if (has_win) {
    win_long <- pivot_window_long(win_df)
    p_win <- ggplot(win_long, aes(window_start_sec, value, colour = side)) +
      geom_step(linewidth = 0.5) +
      geom_point(size = 1) +
      facet_wrap(~label, scales = "free_y", ncol = 3) +
      scale_colour_manual(values = c(Left = "#1f77b4", Right = "#d62728")) +
      theme_minimal(base_size = 10) +
      theme(legend.position = "none",
            strip.text = element_text(size = 8),
            axis.text = element_text(size = 7)) +
      labs(x = "Time (s)", y = NULL)

    n_frame_panels <- n_distinct(long$label)
    frame_rows <- ceiling(n_frame_panels / 3)
    win_rows <- 1
    p_combined <- p_frame / p_win +
      plot_layout(heights = c(frame_rows, win_rows)) +
      plot_annotation(title = vid, theme = theme(
        plot.title = element_text(size = 11, face = "bold")))
    total_h <- frame_rows * 2.5 + win_rows * 2.5 + 1
    cat("  + windows")
  } else {
    n_frame_panels <- n_distinct(long$label)
    frame_rows <- ceiling(n_frame_panels / 3)
    p_combined <- p_frame +
      plot_annotation(title = vid, theme = theme(
        plot.title = element_text(size = 11, face = "bold")))
    total_h <- frame_rows * 2.5 + 1
  }

  stem <- str_remove(basename(f), "_clinical\\.csv$")
  out_png <- file.path(out_dir, paste0(stem, "_clinical_timeseries.png"))
  ggsave(out_png, p_combined, width = 12, height = total_h, dpi = 150,
         limitsize = FALSE)
  cat(sprintf("  → %s\n", basename(out_png)))
  plot_count <- plot_count + 1
}

cat(sprintf("\nPlots written: %d  |  Skipped (<%d rows): %d\n",
            plot_count, MIN_ROWS, skip_count))

# ------------------------------------------------------------------
# Summary overview: one key feature across all videos
# ------------------------------------------------------------------

if (length(all_long) > 0) {
  overview <- bind_rows(all_long) |>
    filter(feature == "reach_norm", side == "Right")

  if (nrow(overview) == 0) {
    # Fall back to any available feature
    overview <- bind_rows(all_long)
    top_feat <- overview |> count(feature, sort = TRUE) |> slice_head(n = 1)
    overview <- overview |> filter(feature == top_feat$feature[1], side == "Right")
    feat_title <- FEATURE_LABELS[top_feat$feature[1]]
  } else {
    feat_title <- "Reach (normalised)"
  }

  n_vids <- n_distinct(overview$video)
  ov_h <- max(4, ceiling(n_vids / 3) * 2.5)

  p_ov <- ggplot(overview, aes(timestamp_sec, value)) +
    geom_line(linewidth = 0.4, colour = "#d62728") +
    facet_wrap(~video, scales = "free", ncol = 3) +
    theme_minimal(base_size = 10) +
    theme(strip.text = element_text(size = 7),
          axis.text = element_text(size = 7)) +
    labs(title = paste0("Right ", feat_title, " — All Videos"),
         x = "Time (s)", y = NULL)

  out_ov <- file.path(out_dir, "all_clinical_timeseries_overview.png")
  ggsave(out_ov, p_ov, width = 12, height = ov_h, dpi = 150,
         limitsize = FALSE)
  cat(sprintf("Wrote → %s\n", out_ov))
} else {
  cat("No videos with ≥", MIN_ROWS, " rows — skipping overview plot.\n")
}

cat("\nDone.\n")

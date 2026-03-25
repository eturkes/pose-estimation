#!/usr/bin/env Rscript
# Temporal diagnostic plots from *_metrics.csv files.
#
# Usage:
#   Rscript analysis/timeseries.R output/video1_metrics.csv
#   Rscript analysis/timeseries.R output/   # all *_metrics.csv in directory
#
# Outputs PNGs alongside each input CSV:
#   <stem>_detection_timeline.png
#   <stem>_jitter.png
#   <stem>_confidence.png
#   <stem>_smoothing.png
#   <stem>_constraints.png

library(tidyverse)

# ------------------------------------------------------------------
# Plot generators
# ------------------------------------------------------------------

plot_detection_timeline <- function(df, title_prefix) {
  timeline <- df |>
    mutate(
      body_status = case_when(
        body_detected == 1 ~ "detected",
        body_carry == 1    ~ "carried",
        TRUE               ~ "missing"
      ),
      hand_L_status = case_when(
        !is.na(hand_L_jitter_px) | !is.na(hand_L_smooth_delta_px) ~ "detected",
        hand_L_carry == 1 ~ "carried",
        TRUE ~ "missing"
      ),
      hand_R_status = case_when(
        !is.na(hand_R_jitter_px) | !is.na(hand_R_smooth_delta_px) ~ "detected",
        hand_R_carry == 1 ~ "carried",
        TRUE ~ "missing"
      )
    ) |>
    select(timestamp_sec, body_status, hand_L_status, hand_R_status) |>
    pivot_longer(-timestamp_sec, names_to = "part", values_to = "status") |>
    mutate(
      part = recode(part,
        body_status = "Body",
        hand_L_status = "Hand L",
        hand_R_status = "Hand R"
      ),
      status = factor(status, levels = c("detected", "carried", "missing"))
    )

  ggplot(timeline, aes(timestamp_sec, part, fill = status)) +
    geom_tile(height = 0.8) +
    scale_fill_manual(
      values = c(detected = "#2ca02c", carried = "#ff7f0e", missing = "#d62728")
    ) +
    labs(
      title = paste(title_prefix, "— Detection Timeline"),
      x = "Time (s)", y = NULL, fill = "Status"
    ) +
    theme_minimal(base_size = 11)
}


plot_jitter <- function(df, title_prefix) {
  jitter_df <- df |>
    select(timestamp_sec, body_jitter_px, hand_L_jitter_px, hand_R_jitter_px) |>
    pivot_longer(-timestamp_sec, names_to = "part", values_to = "jitter") |>
    mutate(part = recode(part,
      body_jitter_px = "Body",
      hand_L_jitter_px = "Hand L",
      hand_R_jitter_px = "Hand R"
    )) |>
    filter(!is.na(jitter))

  # Highlight spikes (> 3 * median)
  jitter_df <- jitter_df |>
    group_by(part) |>
    mutate(
      med = median(jitter, na.rm = TRUE),
      is_spike = jitter > 3 * med
    ) |>
    ungroup()

  ggplot(jitter_df, aes(timestamp_sec, jitter, color = part)) +
    geom_line(alpha = 0.6, linewidth = 0.4) +
    geom_point(data = filter(jitter_df, is_spike),
               aes(timestamp_sec, jitter), size = 1.5, shape = 4) +
    facet_wrap(~part, ncol = 1, scales = "free_y") +
    labs(
      title = paste(title_prefix, "— Jitter Over Time"),
      x = "Time (s)", y = "Jitter (px, sum)", color = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none")
}


plot_confidence <- function(df, title_prefix) {
  conf_df <- df |>
    select(timestamp_sec, body_vis_mean, hand_L_flag, hand_R_flag) |>
    pivot_longer(-timestamp_sec, names_to = "metric", values_to = "value") |>
    mutate(metric = recode(metric,
      body_vis_mean = "Body visibility (mean)",
      hand_L_flag = "Hand L flag",
      hand_R_flag = "Hand R flag"
    )) |>
    filter(!is.na(value))

  ggplot(conf_df, aes(timestamp_sec, value, color = metric)) +
    geom_line(alpha = 0.7, linewidth = 0.4) +
    facet_wrap(~metric, ncol = 1) +
    labs(
      title = paste(title_prefix, "— Confidence Over Time"),
      x = "Time (s)", y = "Score", color = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none")
}


plot_smoothing <- function(df, title_prefix) {
  smooth_df <- df |>
    select(timestamp_sec,
           body_smooth_delta_px, hand_L_smooth_delta_px, hand_R_smooth_delta_px) |>
    pivot_longer(-timestamp_sec, names_to = "part", values_to = "delta") |>
    mutate(part = recode(part,
      body_smooth_delta_px = "Body",
      hand_L_smooth_delta_px = "Hand L",
      hand_R_smooth_delta_px = "Hand R"
    )) |>
    filter(!is.na(delta))

  ggplot(smooth_df, aes(timestamp_sec, delta, color = part)) +
    geom_line(alpha = 0.6, linewidth = 0.4) +
    facet_wrap(~part, ncol = 1, scales = "free_y") +
    labs(
      title = paste(title_prefix, "— Smoothing Delta Over Time"),
      x = "Time (s)", y = "Delta (px, raw - smooth)", color = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none")
}


plot_constraints <- function(df, title_prefix) {
  const_df <- df |>
    select(timestamp_sec, bone_correction_px, angle_corrections_n) |>
    pivot_longer(-timestamp_sec, names_to = "metric", values_to = "value") |>
    mutate(metric = recode(metric,
      bone_correction_px = "Bone correction (px)",
      angle_corrections_n = "Angle clamps (count)"
    ))

  ggplot(const_df, aes(timestamp_sec, value)) +
    geom_line(alpha = 0.6, linewidth = 0.4, color = "#1f77b4") +
    facet_wrap(~metric, ncol = 1, scales = "free_y") +
    labs(
      title = paste(title_prefix, "— Constraint Corrections Over Time"),
      x = "Time (s)", y = NULL
    ) +
    theme_minimal(base_size = 11)
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: Rscript analysis/timeseries.R <metrics_csv_or_directory>")
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
  stem <- str_replace(f, "_metrics\\.csv$", "")
  title <- str_extract(basename(f), "^(.+)_metrics\\.csv$", group = 1)
  if (is.na(title)) title <- basename(f)

  plots <- list(
    detection_timeline = plot_detection_timeline(df, title),
    jitter             = plot_jitter(df, title),
    confidence         = plot_confidence(df, title),
    smoothing          = plot_smoothing(df, title),
    constraints        = plot_constraints(df, title)
  )

  for (name in names(plots)) {
    out <- paste0(stem, "_", name, ".png")
    ggsave(out, plots[[name]], width = 10, height = 6, dpi = 150)
    cat("  Wrote:", out, "\n")
  }
}

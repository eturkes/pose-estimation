#!/usr/bin/env Rscript
# Per-keypoint analysis from *_kp_detail.csv files.
#
# Usage:
#   Rscript analysis/keypoint_detail.R output/video1_kp_detail.csv
#   Rscript analysis/keypoint_detail.R output/   # all *_kp_detail.csv
#
# Outputs PNGs alongside each input CSV:
#   <stem>_jitter_heatmap.png   — keypoint × time heatmap
#   <stem>_worst_keypoints.png  — bar chart of median jitter by keypoint
#   <stem>_trajectory.png       — x/y over time for selected keypoints

library(tidyverse)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Arm keypoint names (12-kp scheme, indices 0-11)
ARM_KP_NAMES <- c(
  "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
  "L_wrist", "R_wrist", "L_index", "R_index",
  "L_middle", "R_middle", "L_pinky", "R_pinky"
)

# Hand keypoint indices to label (subset for readability)
HAND_KP_LABELS <- c(
  "0" = "wrist", "4" = "thumb_tip", "8" = "index_tip",
  "12" = "middle_tip", "16" = "ring_tip", "20" = "pinky_tip"
)

make_kp_label <- function(part, kp_idx) {
  case_when(
    part == "body" & kp_idx < length(ARM_KP_NAMES) ~
      paste0(part, "_", ARM_KP_NAMES[kp_idx + 1]),
    part %in% c("hand_L", "hand_R") & as.character(kp_idx) %in% names(HAND_KP_LABELS) ~
      paste0(part, "_", HAND_KP_LABELS[as.character(kp_idx)]),
    TRUE ~ paste0(part, "_", kp_idx)
  )
}


# ------------------------------------------------------------------
# Plot: Jitter heatmap (keypoint × time)
# ------------------------------------------------------------------

plot_jitter_heatmap <- function(df, title_prefix) {
  heatmap_df <- df |>
    filter(!is.na(jitter_px)) |>
    mutate(kp_label = make_kp_label(part, kp_idx)) |>
    # Bin time for manageable tile count
    mutate(time_bin = round(frame_idx / 10) * 10)

  binned <- heatmap_df |>
    group_by(time_bin, kp_label) |>
    summarise(jitter = median(jitter_px, na.rm = TRUE), .groups = "drop")

  ggplot(binned, aes(time_bin, kp_label, fill = jitter)) +
    geom_tile() +
    scale_fill_viridis_c(option = "inferno", trans = "sqrt") +
    labs(
      title = paste(title_prefix, "— Jitter Heatmap"),
      x = "Frame (binned)", y = NULL, fill = "Jitter (px)"
    ) +
    theme_minimal(base_size = 9) +
    theme(axis.text.y = element_text(size = 6))
}


# ------------------------------------------------------------------
# Plot: Worst keypoints by median jitter
# ------------------------------------------------------------------

plot_worst_keypoints <- function(df, title_prefix) {
  ranked <- df |>
    filter(!is.na(jitter_px)) |>
    mutate(kp_label = make_kp_label(part, kp_idx)) |>
    group_by(kp_label) |>
    summarise(
      median_jitter = median(jitter_px, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(desc(median_jitter)) |>
    slice_head(n = 20)

  ggplot(ranked, aes(median_jitter, reorder(kp_label, median_jitter))) +
    geom_col(fill = "#1f77b4") +
    labs(
      title = paste(title_prefix, "— Worst Keypoints (Median Jitter)"),
      x = "Median Jitter (px)", y = NULL
    ) +
    theme_minimal(base_size = 11)
}


# ------------------------------------------------------------------
# Plot: Trajectory for key keypoints
# ------------------------------------------------------------------

plot_trajectory <- function(df, title_prefix) {
  # Select a few key keypoints
  key_kps <- df |>
    filter(
      (part == "body" & kp_idx %in% c(4, 5)) |
      (part == "hand_L" & kp_idx == 0) |
      (part == "hand_R" & kp_idx == 0)
    ) |>
    filter(!is.na(x_smooth)) |>
    mutate(kp_label = make_kp_label(part, kp_idx))

  if (nrow(key_kps) == 0) return(ggplot() + theme_void())

  traj_long <- key_kps |>
    select(frame_idx, kp_label, x_smooth, y_smooth) |>
    pivot_longer(c(x_smooth, y_smooth), names_to = "coord", values_to = "value")

  ggplot(traj_long, aes(frame_idx, value, color = kp_label)) +
    geom_line(alpha = 0.7, linewidth = 0.4) +
    facet_wrap(~coord, ncol = 1, scales = "free_y") +
    labs(
      title = paste(title_prefix, "— Keypoint Trajectories"),
      x = "Frame", y = "Pixel position", color = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom")
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: Rscript analysis/keypoint_detail.R <kp_detail_csv_or_directory>")
}

path <- args[1]
if (dir.exists(path)) {
  files <- list.files(path, pattern = "_kp_detail\\.csv$", full.names = TRUE)
  if (length(files) == 0) stop("No *_kp_detail.csv files found in ", path)
} else {
  files <- path
}

for (f in files) {
  df <- read_csv(f, show_col_types = FALSE)
  stem <- str_replace(f, "_kp_detail\\.csv$", "")
  title <- str_extract(basename(f), "^(.+)_kp_detail\\.csv$", group = 1)
  if (is.na(title)) title <- basename(f)

  plots <- list(
    jitter_heatmap  = plot_jitter_heatmap(df, title),
    worst_keypoints = plot_worst_keypoints(df, title),
    trajectory      = plot_trajectory(df, title)
  )

  for (name in names(plots)) {
    out <- paste0(stem, "_", name, ".png")
    ggsave(out, plots[[name]], width = 12, height = 8, dpi = 150)
    cat("  Wrote:", out, "\n")
  }
}

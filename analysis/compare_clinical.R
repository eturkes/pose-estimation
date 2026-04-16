#!/usr/bin/env Rscript
# Between-video comparison of aggregated clinical kinematic features.
#
# Loads all *_clinical.csv and *_clinical_windows.csv files from a
# directory, aggregates per-video, and compares across videos to
# identify sessions with notably different movement patterns.
#
# Usage:
#   Rscript analysis/compare_clinical.R output/
#
# Outputs (written to the input directory):
#   all_clinical_video_summary.csv — one row per video, all aggregated features
#   all_clinical_radar.png         — parallel-coordinate plot (z-scored means)
#   all_clinical_heatmap.png       — clustered heatmap (z-scored means)

library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(purrr)
library(ggplot2)
library(tibble)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

FRAME_META  <- c("video", "frame_idx", "timestamp_sec", "person_idx")
WINDOW_META <- c("video", "person_idx", "window_start_sec", "window_end_sec")

# ------------------------------------------------------------------
# Aggregation helpers (same pattern as clinical_correlation.R)
# ------------------------------------------------------------------

aggregate_per_video <- function(df, meta_cols, fns = NULL) {
  feat_cols <- setdiff(names(df), meta_cols)
  feat_cols <- feat_cols[map_lgl(feat_cols, \(c) is.numeric(df[[c]]))]
  if (length(feat_cols) == 0) return(tibble())

  if (is.null(fns)) {
    fns <- list(
      mean   = \(x) mean(x, na.rm = TRUE),
      median = \(x) median(x, na.rm = TRUE),
      sd     = \(x) sd(x, na.rm = TRUE),
      min    = \(x) min(x, na.rm = TRUE),
      max    = \(x) max(x, na.rm = TRUE)
    )
  }

  df |>
    group_by(video) |>
    summarise(
      across(all_of(feat_cols), fns, .names = "{.col}__{.fn}"),
      .groups = "drop"
    )
}

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript analysis/compare_clinical.R <output_dir>")
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

cat(sprintf("Loading %d per-frame clinical CSVs...\n", length(frame_files)))
frame_df <- map(frame_files, \(f) {
  d <- read_csv(f, show_col_types = FALSE)
  if (nrow(d) == 0) return(NULL)
  d
}) |> compact() |> bind_rows()

if (nrow(frame_df) == 0) stop("All per-frame files are empty.")

frame_agg <- aggregate_per_video(frame_df, FRAME_META)

# ------------------------------------------------------------------
# Load per-window files
# ------------------------------------------------------------------

win_files <- list.files(out_dir, pattern = "_clinical_windows\\.csv$",
                        full.names = TRUE)

win_agg <- tibble()
if (length(win_files) > 0) {
  cat(sprintf("Loading %d per-window clinical CSVs...\n", length(win_files)))
  win_df <- map(win_files, \(f) {
    d <- read_csv(f, show_col_types = FALSE)
    if (nrow(d) == 0) return(NULL)
    d
  }) |> compact() |> bind_rows()

  if (nrow(win_df) > 0) {
    win_agg <- aggregate_per_video(win_df, WINDOW_META,
      fns = list(
        mean = \(x) mean(x, na.rm = TRUE),
        sd   = \(x) sd(x, na.rm = TRUE)
      )
    )
  }
}

# ------------------------------------------------------------------
# Combine into one wide summary table
# ------------------------------------------------------------------

summary_df <- frame_agg
if (nrow(win_agg) > 0 && "video" %in% names(win_agg)) {
  summary_df <- summary_df |>
    left_join(win_agg, by = "video", suffix = c("", ".win")) |>
    select(-ends_with(".win"))
}

out_csv <- file.path(out_dir, "all_clinical_video_summary.csv")
write_csv(summary_df, out_csv)
cat(sprintf("Wrote %d videos × %d columns → %s\n",
            nrow(summary_df), ncol(summary_df), out_csv))

# ------------------------------------------------------------------
# Identify numeric feature columns for plotting
# ------------------------------------------------------------------

feat_cols <- setdiff(names(summary_df), "video")
feat_cols <- feat_cols[map_lgl(feat_cols, \(c) is.numeric(summary_df[[c]]))]

# Drop constant or all-NA columns
feat_cols <- feat_cols[map_lgl(feat_cols, \(c) {
  vals <- summary_df[[c]]
  vals <- vals[!is.na(vals) & is.finite(vals)]
  length(vals) >= 2 && sd(vals) > 0
})]

if (length(feat_cols) == 0) {
  stop("No variable features remain after filtering.")
}

# ------------------------------------------------------------------
# Z-score the means for plotting
# ------------------------------------------------------------------

# Select only __mean columns for the radar/heatmap plots
mean_cols <- feat_cols[str_detect(feat_cols, "__mean$")]
if (length(mean_cols) == 0) mean_cols <- feat_cols

z_df <- summary_df |>
  select(video, all_of(mean_cols)) |>
  mutate(across(all_of(mean_cols), \(x) {
    s <- sd(x, na.rm = TRUE)
    if (is.na(s) || s == 0) return(rep(NA_real_, length(x)))
    (x - mean(x, na.rm = TRUE)) / s
  }))

# Short feature labels (strip __mean suffix for readability)
short_names <- str_remove(mean_cols, "__mean$")
rename_map <- setNames(mean_cols, short_names)

# ------------------------------------------------------------------
# Plot 1: Parallel-coordinate plot (z-scored means)
# ------------------------------------------------------------------

z_long <- z_df |>
  pivot_longer(-video, names_to = "feature", values_to = "z") |>
  mutate(feature = str_remove(feature, "__mean$")) |>
  filter(!is.na(z))

n_feat <- n_distinct(z_long$feature)

p_radar <- ggplot(z_long, aes(x = feature, y = z, group = video,
                               colour = video)) +
  geom_line(alpha = 0.7, linewidth = 0.6) +
  geom_point(size = 1.2, alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6),
        legend.position = "bottom",
        legend.title = element_blank()) +
  labs(title = "Between-Video Feature Profiles (z-scored means)",
       x = NULL, y = "z-score")

radar_w <- max(10, n_feat * 0.4)
out_radar <- file.path(out_dir, "all_clinical_radar.png")
ggsave(out_radar, p_radar, width = radar_w, height = 7, dpi = 150,
       limitsize = FALSE)
cat(sprintf("Wrote → %s\n", out_radar))

# ------------------------------------------------------------------
# Plot 2: Clustered heatmap (z-scored means)
# ------------------------------------------------------------------

z_mat <- z_df |>
  select(video, all_of(mean_cols)) |>
  tibble::column_to_rownames("video") |>
  as.matrix()
colnames(z_mat) <- str_remove(colnames(z_mat), "__mean$")

# Replace remaining NAs with 0 for clustering
z_mat[is.na(z_mat)] <- 0

# Cluster rows and columns
if (nrow(z_mat) >= 2 && ncol(z_mat) >= 2) {
  row_ord <- tryCatch({
    hclust(dist(z_mat))$order
  }, error = \(e) seq_len(nrow(z_mat)))

  col_ord <- tryCatch({
    hclust(dist(t(z_mat)))$order
  }, error = \(e) seq_len(ncol(z_mat)))
} else {
  row_ord <- seq_len(nrow(z_mat))
  col_ord <- seq_len(ncol(z_mat))
}

heat_df <- z_mat[row_ord, col_ord, drop = FALSE] |>
  as.data.frame() |>
  rownames_to_column("video") |>
  pivot_longer(-video, names_to = "feature", values_to = "z")

# Preserve cluster order
heat_df$video   <- factor(heat_df$video,
                           levels = rownames(z_mat)[row_ord])
heat_df$feature <- factor(heat_df$feature,
                           levels = colnames(z_mat)[col_ord])

p_heat <- ggplot(heat_df, aes(feature, video, fill = z)) +
  geom_tile(colour = "grey90", linewidth = 0.2) +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                       midpoint = 0) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 55, hjust = 1, size = 6),
        axis.text.y = element_text(size = 8)) +
  labs(title = "z-Scored Feature Means (clustered)",
       x = NULL, y = NULL, fill = "z-score")

heat_w <- max(10, n_feat * 0.35)
heat_h <- max(5, nrow(z_mat) * 0.5)
out_heat <- file.path(out_dir, "all_clinical_heatmap.png")
ggsave(out_heat, p_heat, width = heat_w, height = heat_h, dpi = 150,
       limitsize = FALSE)
cat(sprintf("Wrote → %s\n", out_heat))

# ------------------------------------------------------------------
# Console summary: flag outlier videos
# ------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("  Outlier Videos (|z| > 2 on any feature mean)\n")
cat(strrep("=", 60), "\n\n")

outliers_found <- FALSE
for (col in mean_cols) {
  short <- str_remove(col, "__mean$")
  vals <- summary_df[[col]]
  mu <- mean(vals, na.rm = TRUE)
  s  <- sd(vals, na.rm = TRUE)
  if (is.na(s) || s == 0) next

  z <- (vals - mu) / s
  idx <- which(abs(z) > 2 & !is.na(z))
  if (length(idx) > 0) {
    outliers_found <- TRUE
    for (i in idx) {
      cat(sprintf("  %-25s %-40s z = %+.2f  (value = %.3g)\n",
                  summary_df$video[i], short, z[i], vals[i]))
    }
  }
}

if (!outliers_found) {
  cat("  No videos exceed |z| > 2 on any feature.\n")
}

cat("\nDone.\n")

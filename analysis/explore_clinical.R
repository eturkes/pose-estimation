#!/usr/bin/env Rscript
# Exploratory summary and sanity-check of clinical kinematic features.
#
# Loads all *_clinical.csv (per-frame) and *_clinical_windows.csv
# (per-window) files from a directory, prints summary statistics and
# data-quality warnings, and produces diagnostic plots.
#
# Usage:
#   Rscript analysis/explore_clinical.R output/
#
# Outputs (written to the input directory):
#   all_clinical_distributions.png    — per-feature density plots
#   all_clinical_na_heatmap.png       — missingness heatmap
#   all_clinical_boxplots.png         — per-feature box plots by video
#   all_clinical_window_distributions.png — window feature densities

library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(purrr)
library(ggplot2)
library(tibble)
library(scales)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

FRAME_META  <- c("video", "frame_idx", "timestamp_sec", "person_idx")
WINDOW_META <- c("video", "person_idx", "window_start_sec", "window_end_sec")

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript analysis/explore_clinical.R <output_dir>")
}
out_dir <- args[1]
if (!dir.exists(out_dir)) stop("The directory does not exist: ", out_dir)

# ------------------------------------------------------------------
# Load per-frame files
# ------------------------------------------------------------------

frame_files <- list.files(out_dir, pattern = "_clinical\\.csv$",
                          full.names = TRUE)
frame_files <- frame_files[!str_detect(basename(frame_files), "_windows\\.csv$")]

if (length(frame_files) == 0) stop("The directory contains no *_clinical.csv files: ", out_dir)

df <- map(frame_files, \(f) {
  d <- read_csv(f, show_col_types = FALSE)
  if (nrow(d) == 0) return(NULL)
  d
}) |> compact() |> bind_rows()

all_videos <- map_chr(frame_files, \(f) {
  d <- read_csv(f, show_col_types = FALSE, n_max = 1)
  if ("video" %in% names(d) && nrow(d) > 0) d$video[1]
  else str_remove(basename(f), "_clinical\\.csv$")
})

# ------------------------------------------------------------------
# Load per-window files
# ------------------------------------------------------------------

win_files <- list.files(out_dir, pattern = "_clinical_windows\\.csv$",
                        full.names = TRUE)
win <- map(win_files, \(f) {
  d <- read_csv(f, show_col_types = FALSE)
  if (nrow(d) == 0) return(NULL)
  d
}) |> compact() |> bind_rows()

# ------------------------------------------------------------------
# Console summary
# ------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("  Clinical Features — Exploratory Summary\n")
cat(strrep("=", 60), "\n\n")

cat(sprintf("Per-frame files        : %d\n", length(frame_files)))
cat(sprintf("Videos with data       : %d\n", n_distinct(df$video)))
cat(sprintf("Total per-frame rows   : %d\n", nrow(df)))

cat("\nPer-video row counts:\n")
# Include empty videos
counts <- tibble(video = all_videos) |>
  left_join(count(df, video), by = "video") |>
  mutate(n = replace_na(n, 0L)) |>
  arrange(video)
for (i in seq_len(nrow(counts))) {
  cat(sprintf("  %-30s %d\n", counts$video[i], counts$n[i]))
}

feat_cols <- setdiff(names(df), FRAME_META)
feat_cols <- feat_cols[map_lgl(feat_cols, \(c) is.numeric(df[[c]]))]

na_rates <- df |>
  summarise(across(all_of(feat_cols), \(x) mean(is.na(x)))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "na_rate")

cat("\nPer-column NA rates:\n")
for (i in seq_len(nrow(na_rates))) {
  cat(sprintf("  %-40s %.1f%%\n", na_rates$feature[i],
              na_rates$na_rate[i] * 100))
}

# ------------------------------------------------------------------
# Summary statistics table
# ------------------------------------------------------------------

cat("\n", strrep("-", 60), "\n")
cat("  Per-Feature Summary Statistics\n")
cat(strrep("-", 60), "\n\n")

stats <- df |>
  summarise(across(all_of(feat_cols), list(
    mean   = \(x) mean(x, na.rm = TRUE),
    median = \(x) median(x, na.rm = TRUE),
    sd     = \(x) sd(x, na.rm = TRUE),
    min    = \(x) min(x, na.rm = TRUE),
    max    = \(x) max(x, na.rm = TRUE),
    pct_na = \(x) mean(is.na(x)) * 100
  ), .names = "{.col}__{.fn}")) |>
  pivot_longer(everything(),
               names_to = c("feature", "stat"),
               names_sep = "__") |>
  pivot_wider(names_from = stat, values_from = value)

print(stats, n = Inf, width = Inf)

# ------------------------------------------------------------------
# Plot 1: Per-frame feature distributions
# ------------------------------------------------------------------

long <- df |>
  select(video, all_of(feat_cols)) |>
  pivot_longer(-video, names_to = "feature", values_to = "value") |>
  filter(!is.na(value))

n_feat <- n_distinct(long$feature)
dist_h <- max(6, ceiling(n_feat / 3) * 2.5)

p_dist <- ggplot(long, aes(value, fill = video)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~feature, scales = "free", ncol = 3) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        axis.text.x = element_text(size = 7)) +
  labs(title = "Per-Frame Feature Distributions", x = NULL, y = "Density")

out_dist <- file.path(out_dir, "all_clinical_distributions.png")
ggsave(out_dist, p_dist, width = 12, height = dist_h, dpi = 150,
       limitsize = FALSE)
cat(sprintf("\nThe script wrote %s.\n", out_dist))

# ------------------------------------------------------------------
# Plot 2: NA heatmap
# ------------------------------------------------------------------

na_by_video <- df |>
  group_by(video) |>
  summarise(across(all_of(feat_cols), \(x) mean(is.na(x))),
            .groups = "drop") |>
  pivot_longer(-video, names_to = "feature", values_to = "na_prop")

# Include empty videos
empty_vids <- setdiff(all_videos, unique(df$video))
if (length(empty_vids) > 0) {
  empty_rows <- expand_grid(video = empty_vids, feature = feat_cols) |>
    mutate(na_prop = 1.0)
  na_by_video <- bind_rows(na_by_video, empty_rows)
}

p_na <- ggplot(na_by_video, aes(feature, video, fill = na_prop)) +
  geom_tile(colour = "grey90") +
  scale_fill_gradient(low = "white", high = "#d62728",
                      labels = scales::percent_format(),
                      limits = c(0, 1)) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        axis.text.y = element_text(size = 8)) +
  labs(title = "Missing Data by Video × Feature",
       x = NULL, y = NULL, fill = "% NA")

out_na <- file.path(out_dir, "all_clinical_na_heatmap.png")
ggsave(out_na, p_na, width = 12, height = 6, dpi = 150)
cat(sprintf("The script wrote %s.\n", out_na))

# ------------------------------------------------------------------
# Plot 3: Box plots by video
# ------------------------------------------------------------------

box_h <- max(6, ceiling(n_feat / 3) * 2.5)

p_box <- ggplot(long, aes(video, value)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~feature, scales = "free_y", ncol = 3) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6)) +
  labs(title = "Feature Distributions by Video", x = NULL, y = NULL)

out_box <- file.path(out_dir, "all_clinical_boxplots.png")
ggsave(out_box, p_box, width = 14, height = box_h, dpi = 150,
       limitsize = FALSE)
cat(sprintf("The script wrote %s.\n", out_box))

# ------------------------------------------------------------------
# Plot 4: Window feature distributions
# ------------------------------------------------------------------

if (nrow(win) > 0) {
  win_feat <- setdiff(names(win), WINDOW_META)
  win_feat <- win_feat[map_lgl(win_feat, \(c) is.numeric(win[[c]]))]

  win_long <- win |>
    select(video, all_of(win_feat)) |>
    pivot_longer(-video, names_to = "feature", values_to = "value") |>
    filter(!is.na(value))

  n_wfeat <- n_distinct(win_long$feature)
  wdist_h <- max(6, ceiling(n_wfeat / 3) * 2.5)

  p_wdist <- ggplot(win_long, aes(value, fill = video)) +
    geom_density(alpha = 0.4) +
    facet_wrap(~feature, scales = "free", ncol = 3) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom",
          legend.title = element_blank(),
          axis.text.x = element_text(size = 7)) +
    labs(title = "Window Feature Distributions", x = NULL, y = "Density")

  out_wdist <- file.path(out_dir, "all_clinical_window_distributions.png")
  ggsave(out_wdist, p_wdist, width = 12, height = wdist_h, dpi = 150,
         limitsize = FALSE)
  cat(sprintf("The script wrote %s.\n", out_wdist))
} else {
  cat("The script found no window data. It skips the window-distribution plot.\n")
}

# ------------------------------------------------------------------
# Data-quality warnings
# ------------------------------------------------------------------

cat("\n", strrep("-", 60), "\n")
cat("  Data-Quality Warnings\n")
cat(strrep("-", 60), "\n\n")

warnings_found <- FALSE

# Features >50% NA
high_na <- na_rates |> filter(na_rate > 0.5)
if (nrow(high_na) > 0) {
  warnings_found <- TRUE
  cat("Features with >50% NA across all videos:\n")
  for (i in seq_len(nrow(high_na))) {
    cat(sprintf("  %-40s %.1f%%\n", high_na$feature[i],
                high_na$na_rate[i] * 100))
  }
  cat("\n")
}

# Videos with zero non-NA frames (including empty files)
zero_vids <- counts |> filter(n == 0)
if (nrow(zero_vids) > 0) {
  warnings_found <- TRUE
  cat("Videos with zero data rows:\n")
  for (v in zero_vids$video) cat(sprintf("  %s\n", v))
  cat("\n")
}

# Constant-valued features (zero variance)
const_feats <- stats |>
  filter(!is.na(sd), sd == 0)
if (nrow(const_feats) > 0) {
  warnings_found <- TRUE
  cat("Constant-valued features (zero variance):\n")
  for (f in const_feats$feature) cat(sprintf("  %s\n", f))
  cat("\n")
}

if (!warnings_found) cat("No data-quality warnings.\n")

cat("\nThe script finished.\n")

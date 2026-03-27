#!/usr/bin/env Rscript
# Correlate clinical kinematic features with external clinical scores.
#
# Reads per-frame and per-window clinical CSVs produced by
# clinical_features.R, aggregates them per video, then joins with a
# clinical scores table and computes pairwise correlations.
#
# Usage:
#   Rscript analysis/clinical_correlation.R output/ clinical_scores.csv
#   Rscript analysis/clinical_correlation.R output/video1_clinical.csv clinical_scores.csv
#
# Outputs (written next to the first input or into the directory):
#   <prefix>_correlation_matrix.png  — heatmap (Spearman rho) with stars
#   <prefix>_correlation_table.csv   — tidy correlation results
#   <prefix>_scatter_top.png         — scatter plots for top 6 pairs

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

METADATA_COLS <- c("video", "frame_idx", "timestamp_sec", "person_idx")
WINDOW_META   <- c("video", "person_idx", "window_start_sec", "window_end_sec")

# ------------------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------------------

#' Aggregate a clinical features data frame to one row per video.
#' Computes mean, median, sd, min, max for every numeric feature column.
aggregate_per_video <- function(df, meta_cols) {
  feat_cols <- setdiff(names(df), meta_cols)
  feat_cols <- feat_cols[map_lgl(feat_cols, \(c) is.numeric(df[[c]]))]
  if (length(feat_cols) == 0) return(tibble())

  df |>
    group_by(video) |>
    summarise(
      across(
        all_of(feat_cols),
        list(
          mean   = \(x) mean(x, na.rm = TRUE),
          median = \(x) median(x, na.rm = TRUE),
          sd     = \(x) sd(x, na.rm = TRUE),
          min    = \(x) min(x, na.rm = TRUE),
          max    = \(x) max(x, na.rm = TRUE)
        ),
        .names = "{.col}__{.fn}"
      ),
      .groups = "drop"
    )
}

# ------------------------------------------------------------------
# Correlation computation
# ------------------------------------------------------------------

#' Compute pairwise correlations between feature columns and score
#' columns, returning a tidy data frame with BH-corrected p-values.
compute_correlations <- function(joined, feature_cols, score_cols) {
  results <- vector("list", length(feature_cols) * length(score_cols))
  ri <- 0L

  for (feat in feature_cols) {
    for (score in score_cols) {
      x <- joined[[feat]]
      y <- joined[[score]]
      complete <- !is.na(x) & !is.na(y) & is.finite(x) & is.finite(y)
      n <- sum(complete)
      if (n < 4) {
        ri <- ri + 1L
        results[[ri]] <- tibble(
          feature = feat, score = score,
          pearson_r = NA_real_, spearman_rho = NA_real_,
          p_value = NA_real_, n = n
        )
        next
      }

      pear <- tryCatch(
        cor.test(x[complete], y[complete], method = "pearson"),
        error = \(e) NULL
      )
      spear <- tryCatch(
        cor.test(x[complete], y[complete], method = "spearman",
                 exact = FALSE),
        error = \(e) NULL
      )

      ri <- ri + 1L
      results[[ri]] <- tibble(
        feature      = feat,
        score        = score,
        pearson_r    = if (!is.null(pear))  pear$estimate  else NA_real_,
        spearman_rho = if (!is.null(spear)) spear$estimate else NA_real_,
        p_value      = if (!is.null(spear)) spear$p.value  else NA_real_,
        n            = n
      )
    }
  }

  out <- bind_rows(results[seq_len(ri)])

  # BH correction on Spearman p-values.
  out$p_adj_bh <- p.adjust(out$p_value, method = "BH")
  out
}

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

sig_stars <- function(p) {
  case_when(
    is.na(p)  ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE      ~ ""
  )
}

plot_correlation_matrix <- function(cor_tbl, title_prefix) {
  mat <- cor_tbl |>
    select(feature, score, spearman_rho, p_adj_bh) |>
    mutate(star = sig_stars(p_adj_bh))

  ggplot(mat, aes(score, feature, fill = spearman_rho)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = star), size = 3, vjust = 0.75) +
    scale_fill_gradient2(
      low = "#2166ac", mid = "white", high = "#b2182b",
      midpoint = 0, limits = c(-1, 1)
    ) +
    labs(
      title = paste(title_prefix, "— Feature × Score Correlation"),
      subtitle = "Spearman rho (BH-adjusted * p<.05, ** p<.01, *** p<.001)",
      x = NULL, y = NULL, fill = "Spearman rho"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.text.y = element_text(size = 7)
    )
}

plot_scatter_top <- function(joined, cor_tbl, title_prefix, n_top = 6) {
  top <- cor_tbl |>
    filter(!is.na(spearman_rho)) |>
    arrange(desc(abs(spearman_rho))) |>
    slice_head(n = n_top)

  if (nrow(top) == 0) return(NULL)

  # Reshape into long format for faceting — avoids extra package deps.
  long <- pmap_dfr(top, function(feature, score, spearman_rho, p_adj_bh, ...) {
    label <- sprintf("%s vs %s  (rho=%.2f, p_adj=%.3g)",
                     feature, score, spearman_rho, p_adj_bh)
    tibble(
      panel = label,
      x     = joined[[feature]],
      y     = joined[[score]]
    )
  }) |>
    filter(!is.na(x), !is.na(y), is.finite(x), is.finite(y))

  if (nrow(long) == 0) return(NULL)

  long$panel <- factor(long$panel, levels = unique(long$panel))

  ggplot(long, aes(x, y)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.6,
                color = "steelblue") +
    facet_wrap(~panel, scales = "free", ncol = 3) +
    labs(
      title = paste(title_prefix, "— Top Feature-Score Scatter Plots"),
      x = NULL, y = NULL
    ) +
    theme_minimal(base_size = 9)
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(paste(
    "Usage: Rscript analysis/clinical_correlation.R",
    "<clinical_csv_or_directory> <clinical_scores.csv>"
  ))
}

clin_path   <- args[1]
scores_path <- args[2]

# --- Load clinical scores ---
if (!file.exists(scores_path)) {
  stop("Clinical scores file not found: ", scores_path)
}
scores <- read_csv(scores_path, show_col_types = FALSE)
if (!"video" %in% names(scores)) {
  stop("Clinical scores CSV must have a 'video' column.")
}
score_cols <- setdiff(names(scores), "video")
score_cols <- score_cols[map_lgl(score_cols, \(c) is.numeric(scores[[c]]))]
if (length(score_cols) == 0) {
  stop("No numeric score columns found in ", scores_path)
}
cat(sprintf("Loaded %d scores for %d videos: %s\n",
            length(score_cols), nrow(scores),
            paste(score_cols, collapse = ", ")))

# --- Discover clinical feature CSVs ---
if (dir.exists(clin_path)) {
  frame_files <- list.files(clin_path, pattern = "_clinical\\.csv$",
                            full.names = TRUE)
  # Exclude window files.
  frame_files <- frame_files[!str_detect(basename(frame_files), "_windows\\.csv$")]
  win_files <- list.files(clin_path, pattern = "_clinical_windows\\.csv$",
                          full.names = TRUE)
  out_dir <- clin_path
} else {
  frame_files <- clin_path
  win_files <- str_replace(clin_path, "_clinical\\.csv$",
                           "_clinical_windows.csv")
  win_files <- win_files[file.exists(win_files)]
  out_dir <- dirname(clin_path)
}

if (length(frame_files) == 0) {
  stop("No *_clinical.csv files found in ", clin_path)
}

cat(sprintf("Found %d frame-level and %d window-level clinical CSVs.\n",
            length(frame_files), length(win_files)))

# --- Load and aggregate frame-level features ---
cat("Aggregating per-frame clinical features per video...\n")
frame_agg <- map(frame_files, \(f) {
  df <- read_csv(f, show_col_types = FALSE)
  aggregate_per_video(df, METADATA_COLS)
}) |> bind_rows()

if (nrow(frame_agg) == 0) stop("No frame-level features to aggregate.")

# --- Load and aggregate window-level features ---
if (length(win_files) > 0) {
  cat("Aggregating per-window clinical features per video...\n")
  win_agg <- map(win_files, \(f) {
    df <- read_csv(f, show_col_types = FALSE)
    aggregate_per_video(df, WINDOW_META)
  }) |> bind_rows()

  if (nrow(win_agg) > 0 && "video" %in% names(win_agg)) {
    frame_agg <- frame_agg |>
      left_join(win_agg, by = "video", suffix = c("", ".win"))
    # Drop duplicate columns from join.
    frame_agg <- frame_agg |>
      select(-ends_with(".win"))
  }
}

# --- Join with clinical scores ---
cat("Joining with clinical scores...\n")
agg_videos  <- unique(frame_agg$video)
score_videos <- unique(scores$video)
matched   <- intersect(agg_videos, score_videos)
unmatched_feat  <- setdiff(agg_videos, score_videos)
unmatched_score <- setdiff(score_videos, agg_videos)

if (length(unmatched_feat) > 0) {
  warning(sprintf(
    "%d feature videos have no matching score: %s",
    length(unmatched_feat), paste(unmatched_feat, collapse = ", ")
  ))
}
if (length(unmatched_score) > 0) {
  warning(sprintf(
    "%d score entries have no matching features: %s",
    length(unmatched_score), paste(unmatched_score, collapse = ", ")
  ))
}
if (length(matched) == 0) {
  stop("No videos matched between features and scores.")
}
cat(sprintf("Matched %d videos.\n", length(matched)))

joined <- inner_join(frame_agg, scores, by = "video")

feature_cols <- setdiff(names(frame_agg), "video")
feature_cols <- feature_cols[map_lgl(feature_cols, \(c) is.numeric(joined[[c]]))]

# Drop features that are constant or all-NA in the matched subset.
feature_cols <- feature_cols[map_lgl(feature_cols, \(c) {
  vals <- joined[[c]]
  vals <- vals[!is.na(vals) & is.finite(vals)]
  length(vals) >= 3 && sd(vals) > 0
})]

if (length(feature_cols) == 0) {
  stop("No variable features remain after filtering.")
}
cat(sprintf("Computing correlations: %d features × %d scores.\n",
            length(feature_cols), length(score_cols)))

# --- Compute correlations ---
cor_tbl <- compute_correlations(joined, feature_cols, score_cols)

# --- Output prefix ---
out_prefix <- file.path(out_dir, "clinical")

# --- Write correlation table ---
out_csv <- paste0(out_prefix, "_correlation_table.csv")
write_csv(cor_tbl, out_csv)
cat(sprintf("Wrote %d rows → %s\n", nrow(cor_tbl), out_csv))

# --- Console summary ---
cat("\n", strrep("=", 60), "\n")
cat("  Top correlations (by |Spearman rho|)\n")
cat(strrep("=", 60), "\n")

top10 <- cor_tbl |>
  filter(!is.na(spearman_rho)) |>
  arrange(desc(abs(spearman_rho))) |>
  slice_head(n = 10)

if (nrow(top10) > 0) {
  for (i in seq_len(nrow(top10))) {
    r <- top10[i, ]
    star <- sig_stars(r$p_adj_bh)
    cat(sprintf("  %2d. %-45s vs %-12s  rho=%+.3f  p_adj=%.3g %s  (n=%d)\n",
                i, r$feature, r$score, r$spearman_rho, r$p_adj_bh, star, r$n))
  }
}

no_sig <- cor_tbl |>
  group_by(feature) |>
  summarise(any_sig = any(p_adj_bh < 0.05, na.rm = TRUE), .groups = "drop") |>
  filter(!any_sig)

if (nrow(no_sig) > 0) {
  cat(sprintf("\n  %d features had no significant (p_adj < 0.05) associations.\n",
              nrow(no_sig)))
}

# --- Correlation heatmap ---
cat("\nPlotting correlation heatmap...\n")
p_mat <- plot_correlation_matrix(cor_tbl, "Clinical")
n_feat <- length(unique(cor_tbl$feature))
mat_height <- max(6, n_feat * 0.18)
ggsave(paste0(out_prefix, "_correlation_matrix.png"), p_mat,
       width = max(8, length(score_cols) * 1.2 + 4), height = mat_height,
       dpi = 150, limitsize = FALSE)
cat(sprintf("Wrote → %s\n", paste0(out_prefix, "_correlation_matrix.png")))

# --- Top scatter plots ---
cat("Plotting top scatter plots...\n")
p_scat <- plot_scatter_top(joined, cor_tbl, "Clinical")
if (!is.null(p_scat)) {
  n_panels <- min(6, sum(!is.na(cor_tbl$spearman_rho)))
  scat_height <- ceiling(n_panels / 3) * 4
  ggsave(paste0(out_prefix, "_scatter_top.png"), p_scat,
         width = 12, height = scat_height, dpi = 150)
  cat(sprintf("Wrote → %s\n", paste0(out_prefix, "_scatter_top.png")))
} else {
  cat("  No valid correlations to plot.\n")
}

cat("\nDone.\n")

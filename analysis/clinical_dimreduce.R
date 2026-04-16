#!/usr/bin/env Rscript
# Dimensionality reduction (PCA + UMAP) on per-video aggregated
# clinical kinematic features.
#
# Usage:
#   Rscript analysis/clinical_dimreduce.R output/
#
# Outputs (written to the input directory):
#   all_clinical_pca_scree.png    — scree plot of variance explained
#   all_clinical_pca_biplot.png   — PC1 vs PC2 biplot with video labels
#                                   and feature loading arrows
#   all_clinical_umap.png         — 2D UMAP with video labels
#   all_clinical_pca_loadings.csv — feature loadings on first PCs

library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(purrr)
library(ggplot2)
library(tibble)
library(uwot)

# Shared helpers (aggregate_per_video, METADATA_COLS, WINDOW_META).
local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  match <- grep(file_arg, args)
  here <- if (length(match) > 0) {
    dirname(sub(file_arg, "", args[match]))
  } else if (file.exists("analysis/utils.R")) {
    "analysis"
  } else {
    "."
  }
  source(file.path(here, "utils.R"))
})

# Per-frame metadata uses the canonical METADATA_COLS from utils.R; this
# script aliases it for readability since the variable name "FRAME_META"
# was used historically.
FRAME_META <- METADATA_COLS

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript analysis/clinical_dimreduce.R <output_dir>")
}
out_dir <- args[1]
if (!dir.exists(out_dir)) stop("Directory not found: ", out_dir)

# ------------------------------------------------------------------
# Load and aggregate per-frame features
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
# Load and aggregate per-window features
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

# Combine into one wide summary table
summary_df <- frame_agg
if (nrow(win_agg) > 0 && "video" %in% names(win_agg)) {
  summary_df <- summary_df |>
    left_join(win_agg, by = "video", suffix = c("", ".win")) |>
    select(-ends_with(".win"))
}

cat(sprintf("Aggregated %d videos × %d columns.\n",
            nrow(summary_df), ncol(summary_df)))

# ------------------------------------------------------------------
# Prepare feature matrix
# ------------------------------------------------------------------

feat_cols <- setdiff(names(summary_df), "video")
feat_cols <- feat_cols[map_lgl(feat_cols, \(c) is.numeric(summary_df[[c]]))]

# Drop columns with zero variance or >50% NA
feat_cols <- feat_cols[map_lgl(feat_cols, \(c) {
  vals <- summary_df[[c]]
  na_frac <- mean(is.na(vals) | !is.finite(vals))
  if (na_frac > 0.5) return(FALSE)
  finite <- vals[!is.na(vals) & is.finite(vals)]
  length(finite) >= 2 && sd(finite) > 0
})]

if (length(feat_cols) < 2) {
  stop("Fewer than 2 variable features remain after filtering.")
}

cat(sprintf("Using %d features after dropping zero-variance / high-NA columns.\n",
            length(feat_cols)))

feat_mat <- summary_df |> select(all_of(feat_cols)) |> as.matrix()
videos   <- summary_df$video

# Impute remaining NAs with column medians
na_count <- sum(is.na(feat_mat))
if (na_count > 0) {
  cat(sprintf("Imputing %d remaining NAs with column medians.\n", na_count))
  for (j in seq_len(ncol(feat_mat))) {
    na_idx <- is.na(feat_mat[, j])
    if (any(na_idx)) {
      feat_mat[na_idx, j] <- median(feat_mat[, j], na.rm = TRUE)
    }
  }
  # If a column is still all-NA, fill with 0
  feat_mat[is.na(feat_mat)] <- 0
}

# Z-score all features
feat_scaled <- scale(feat_mat)
# Remove any columns that became NaN after scaling (constant post-imputation)
keep <- !apply(feat_scaled, 2, \(x) any(is.nan(x)))
feat_scaled <- feat_scaled[, keep, drop = FALSE]
feat_cols   <- feat_cols[keep]

cat(sprintf("Feature matrix: %d videos × %d features (z-scored).\n",
            nrow(feat_scaled), ncol(feat_scaled)))

# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------

cat("Running PCA...\n")
pca_fit <- prcomp(feat_scaled, center = FALSE, scale. = FALSE)
var_exp <- pca_fit$sdev^2 / sum(pca_fit$sdev^2)
cum_var <- cumsum(var_exp)

n_pcs   <- min(length(var_exp), ncol(feat_scaled))
n_show  <- min(n_pcs, 20)

# --- Scree plot ---
scree_df <- tibble(
  PC      = seq_len(n_show),
  var_exp = var_exp[seq_len(n_show)],
  cum_var = cum_var[seq_len(n_show)]
)

p_scree <- ggplot(scree_df, aes(PC)) +
  geom_col(aes(y = var_exp), fill = "#1f77b4", alpha = 0.7) +
  geom_line(aes(y = cum_var), linewidth = 0.8) +
  geom_point(aes(y = cum_var), size = 2) +
  geom_hline(yintercept = 0.80, linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0.95, linetype = "dashed", colour = "grey50") +
  annotate("text", x = n_show, y = 0.81, label = "80%",
           hjust = 1, size = 3, colour = "grey40") +
  annotate("text", x = n_show, y = 0.96, label = "95%",
           hjust = 1, size = 3, colour = "grey40") +
  scale_x_continuous(breaks = seq_len(n_show)) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Clinical Features — PCA Scree Plot",
       x = "Principal Component", y = "Variance Explained") +
  theme_minimal(base_size = 11)

out_scree <- file.path(out_dir, "all_clinical_pca_scree.png")
ggsave(out_scree, p_scree, width = 8, height = 5, dpi = 150)
cat(sprintf("Wrote → %s\n", out_scree))

# --- Biplot ---
scores <- as.data.frame(pca_fit$x[, 1:min(2, ncol(pca_fit$x))])
if (ncol(scores) < 2) {
  cat("Only 1 PC available — skipping biplot.\n")
} else {
  scores$video <- videos
  pc1_lab <- sprintf("PC1 (%.1f%%)", var_exp[1] * 100)
  pc2_lab <- sprintf("PC2 (%.1f%%)", var_exp[2] * 100)

  # Top loading arrows
  loadings <- as.data.frame(pca_fit$rotation[, 1:2])
  loadings$feature <- rownames(loadings)
  loadings$magnitude <- sqrt(loadings$PC1^2 + loadings$PC2^2)
  top_load <- loadings |> arrange(desc(magnitude)) |> slice_head(n = 10)

  # Short labels for readability
  top_load$label <- str_remove(top_load$feature, "__mean$")
  top_load$label <- str_trunc(top_load$label, 35)

  # Scale arrows to fit score range
  score_range <- max(abs(range(scores$PC1)), abs(range(scores$PC2)))
  arrow_scale <- score_range / max(top_load$magnitude) * 0.45

  p_biplot <- ggplot(scores, aes(PC1, PC2)) +
    geom_point(size = 3, colour = "#1f77b4") +
    geom_text(aes(label = video), size = 2.5, vjust = -0.8, hjust = 0.5) +
    geom_segment(
      data = top_load,
      aes(x = 0, y = 0, xend = PC1 * arrow_scale, yend = PC2 * arrow_scale),
      arrow = arrow(length = unit(0.15, "cm")),
      colour = "grey40", linewidth = 0.4, inherit.aes = FALSE
    ) +
    geom_text(
      data = top_load,
      aes(x = PC1 * arrow_scale * 1.08, y = PC2 * arrow_scale * 1.08,
          label = label),
      colour = "grey40", size = 2, inherit.aes = FALSE
    ) +
    labs(title = "Clinical Features — PCA Biplot",
         x = pc1_lab, y = pc2_lab) +
    theme_minimal(base_size = 11)

  out_biplot <- file.path(out_dir, "all_clinical_pca_biplot.png")
  ggsave(out_biplot, p_biplot, width = 10, height = 8, dpi = 150)
  cat(sprintf("Wrote → %s\n", out_biplot))
}

# ------------------------------------------------------------------
# UMAP
# ------------------------------------------------------------------

n_videos <- nrow(feat_scaled)
n_neighbors <- min(5, n_videos - 1)

if (n_videos < 4) {
  cat(sprintf("Only %d videos — skipping UMAP (need ≥4).\n", n_videos))
} else {
  cat(sprintf("Running UMAP (n_neighbors = %d)...\n", n_neighbors))
  set.seed(4817)
  umap_coords <- umap(feat_scaled, n_neighbors = n_neighbors, min_dist = 0.1,
                       metric = "euclidean", n_epochs = 500)

  umap_df <- tibble(
    UMAP1 = umap_coords[, 1],
    UMAP2 = umap_coords[, 2],
    video = videos
  )

  p_umap <- ggplot(umap_df, aes(UMAP1, UMAP2)) +
    geom_point(size = 3, colour = "#1f77b4") +
    geom_text(aes(label = video), size = 2.5, vjust = -0.8, hjust = 0.5) +
    labs(title = sprintf("Clinical Features — UMAP (n_neighbors = %d)", n_neighbors),
         x = "UMAP 1", y = "UMAP 2") +
    theme_minimal(base_size = 11) +
    coord_equal()

  out_umap <- file.path(out_dir, "all_clinical_umap.png")
  ggsave(out_umap, p_umap, width = 8, height = 7, dpi = 150)
  cat(sprintf("Wrote → %s\n", out_umap))
}

# ------------------------------------------------------------------
# PCA loadings CSV
# ------------------------------------------------------------------

n_pcs_save <- min(n_pcs, 5)
loadings_df <- as.data.frame(pca_fit$rotation[, seq_len(n_pcs_save)]) |>
  rownames_to_column("feature") |>
  as_tibble() |>
  mutate(
    feature_short = str_remove(feature, "__mean$"),
    abs_PC1 = abs(PC1)
  ) |>
  arrange(desc(abs_PC1))

out_loadings <- file.path(out_dir, "all_clinical_pca_loadings.csv")
write_csv(loadings_df, out_loadings)
cat(sprintf("Wrote → %s\n", out_loadings))

# ------------------------------------------------------------------
# Console summary
# ------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("  PCA Summary\n")
cat(strrep("=", 60), "\n\n")

cat(sprintf("  Videos:   %d\n", n_videos))
cat(sprintf("  Features: %d\n", length(feat_cols)))
cat(sprintf("  PC1 explains %.1f%% of variance\n", var_exp[1] * 100))
if (n_pcs >= 2) {
  cat(sprintf("  PC2 explains %.1f%% of variance\n", var_exp[2] * 100))
  cat(sprintf("  First 2 PCs:  %.1f%% cumulative\n", cum_var[2] * 100))
}
if (n_pcs >= 3) {
  cat(sprintf("  First 3 PCs:  %.1f%% cumulative\n", cum_var[3] * 100))
}

cat("\n  Top features loading on PC1:\n")
top_pc1 <- loadings_df |> slice_head(n = 5)
for (i in seq_len(nrow(top_pc1))) {
  cat(sprintf("    %-45s  PC1 = %+.3f\n",
              top_pc1$feature_short[i], top_pc1$PC1[i]))
}

if (n_pcs >= 2) {
  top_pc2 <- loadings_df |>
    mutate(abs_PC2 = abs(PC2)) |>
    arrange(desc(abs_PC2)) |>
    slice_head(n = 5)
  cat("\n  Top features loading on PC2:\n")
  for (i in seq_len(nrow(top_pc2))) {
    cat(sprintf("    %-45s  PC2 = %+.3f\n",
                top_pc2$feature_short[i], top_pc2$PC2[i]))
  }
}

cat("\nDone.\n")

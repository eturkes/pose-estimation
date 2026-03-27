#!/usr/bin/env Rscript
# Feature selection and dimensionality-reduction visualisation for
# landmark CSVs produced by export.py.
#
# Usage:
#   Rscript analysis/features.R output/video1_hands-arms.csv
#   Rscript analysis/features.R output/   # all landmark CSVs in directory
#
# Outputs PNGs alongside each input CSV:
#   <stem>_variance.png      — per-feature variance bar chart
#   <stem>_correlation.png   — clustered correlation heatmap
#   <stem>_pca_scree.png     — cumulative variance explained
#   <stem>_pca_biplot.png    — PC1 vs PC2 with loading arrows
#   <stem>_umap.png          — UMAP coloured by timestamp
#   <stem>_umap_video.png    — UMAP coloured by video (if >1)
#   <stem>_feature_rank.csv  — feature ranking table

library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(purrr)
library(ggplot2)
library(scales)
library(tibble)
library(uwot)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

METADATA_COLS <- c("video", "frame_idx", "timestamp_sec", "person_idx")

# Maximum rows fed to UMAP / PCA (subsample if larger for speed)
MAX_ROWS <- 50000

# Near-zero-variance threshold: fraction of overall max variance
NZV_FRAC <- 0.01

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

detect_tracking <- function(cols) {
  if (any(str_starts(cols, "body_"))) return("body")
  if (any(str_starts(cols, "arm_")))  return("hands-arms")
  "hands"
}

coord_columns <- function(cols) {
  cols[str_detect(cols, "_(x|y|z)$")]
}

parse_feature_name <- function(col_name) {
  # Extract part, keypoint, and coordinate from column name.
  # e.g. "arm_left_wrist_x" -> part=arm, keypoint=left_wrist, coord=x
  # e.g. "left_hand_4_y"    -> part=left_hand, keypoint=4, coord=y
  coord <- str_extract(col_name, "[xyz]$")

  base <- str_remove(col_name, "_(x|y|z)$")

  hand_match <- str_match(base, "^(left_hand|right_hand)_(.+)$")
  arm_match  <- str_match(base, "^(arm|body)_(.+)$")

  if (!is.na(hand_match[1, 1])) {
    part     <- hand_match[1, 2]
    keypoint <- hand_match[1, 3]
  } else if (!is.na(arm_match[1, 1])) {
    part     <- arm_match[1, 2]
    keypoint <- arm_match[1, 3]
  } else {
    part     <- "unknown"
    keypoint <- base
  }

  tibble(column = col_name, part = part, keypoint = keypoint, coord = coord)
}


# ------------------------------------------------------------------
# Feature matrix preparation
# ------------------------------------------------------------------

prepare_features <- function(df, coord_cols) {
  # Convert blanks to NA (read_csv handles this), then to numeric.
  feat <- df |> select(all_of(coord_cols)) |> mutate(across(everything(), as.numeric))

  # Count missingness per column.
  na_counts <- colSums(is.na(feat))
  n <- nrow(feat)
  cat(sprintf("  Feature matrix: %d rows × %d features\n", n, ncol(feat)))
  high_na <- na_counts[na_counts > 0.5 * n]
  if (length(high_na) > 0) {
    cat(sprintf("  %d features have >50%% missing (likely undetected hand).\n",
                length(high_na)))
  }

  # Drop rows that are entirely NA across all coordinate columns.
  all_na <- rowSums(is.na(feat)) == ncol(feat)
  if (any(all_na)) {
    cat(sprintf("  Dropping %d fully-empty rows.\n", sum(all_na)))
    feat <- feat[!all_na, , drop = FALSE]
  }

  # Impute remaining sporadic NAs with column medians.
  na_remaining <- sum(is.na(feat))
  if (na_remaining > 0) {
    cat(sprintf("  Imputing %d remaining NAs with column medians.\n", na_remaining))
    feat <- feat |>
      mutate(across(everything(), \(x) replace(x, is.na(x), median(x, na.rm = TRUE))))
    # If a column is still all-NA after median imputation, fill with 0.
    feat[is.na(feat)] <- 0
  }

  list(features = feat, kept_rows = which(!all_na))
}


# ------------------------------------------------------------------
# Plot: variance bar chart
# ------------------------------------------------------------------

plot_variance <- function(feat, title_prefix) {
  vars <- tibble(
    column   = names(feat),
    variance = map_dbl(feat, var, na.rm = TRUE)
  ) |>
    left_join(map_dfr(names(feat), parse_feature_name), by = "column") |>
    arrange(desc(variance))

  threshold <- max(vars$variance) * NZV_FRAC

  ggplot(vars, aes(variance, reorder(column, variance), fill = part)) +
    geom_col() +
    geom_vline(xintercept = threshold, linetype = "dashed", color = "red") +
    scale_x_log10() +
    labs(
      title = paste(title_prefix, "— Per-Feature Variance"),
      x = "Variance (log scale)", y = NULL, fill = "Part"
    ) +
    theme_minimal(base_size = 8) +
    theme(axis.text.y = element_text(size = 5))
}


# ------------------------------------------------------------------
# Plot: clustered correlation heatmap
# ------------------------------------------------------------------

plot_correlation <- function(feat, title_prefix) {
  # Drop zero-variance columns that produce NaN correlations.
  vars <- map_dbl(feat, var, na.rm = TRUE)
  feat <- feat[, vars > 0, drop = FALSE]

  cor_mat <- cor(feat, use = "pairwise.complete.obs")
  cor_mat[is.na(cor_mat)] <- 0

  # Hierarchical clustering for ordering.
  hc <- hclust(as.dist(1 - abs(cor_mat)), method = "ward.D2")
  ord <- hc$order
  cor_mat <- cor_mat[ord, ord]

  cor_df <- as.data.frame(cor_mat) |>
    rownames_to_column("feature_1") |>
    pivot_longer(-feature_1, names_to = "feature_2", values_to = "correlation") |>
    mutate(
      feature_1 = factor(feature_1, levels = rownames(cor_mat)),
      feature_2 = factor(feature_2, levels = rownames(cor_mat))
    )

  ggplot(cor_df, aes(feature_1, feature_2, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                         midpoint = 0, limits = c(-1, 1)) +
    labs(
      title = paste(title_prefix, "— Feature Correlation (clustered)"),
      x = NULL, y = NULL, fill = "Pearson r"
    ) +
    theme_minimal(base_size = 8) +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, size = 4),
      axis.text.y = element_text(size = 4)
    )
}


# ------------------------------------------------------------------
# Plot: PCA scree
# ------------------------------------------------------------------

run_pca <- function(feat) {
  # Centre and scale.
  feat_scaled <- scale(feat)
  # Remove any constant columns that became NaN after scaling.
  keep <- !apply(feat_scaled, 2, \(x) any(is.nan(x)))
  feat_scaled <- feat_scaled[, keep, drop = FALSE]
  prcomp(feat_scaled, center = FALSE, scale. = FALSE)
}

plot_pca_scree <- function(pca_fit, title_prefix) {
  var_exp <- pca_fit$sdev^2 / sum(pca_fit$sdev^2)
  cum_var <- cumsum(var_exp)
  n_show  <- min(length(var_exp), 30)

  scree_df <- tibble(
    PC       = seq_len(n_show),
    var_exp  = var_exp[seq_len(n_show)],
    cum_var  = cum_var[seq_len(n_show)]
  )

  ggplot(scree_df, aes(PC)) +
    geom_col(aes(y = var_exp), fill = "#1f77b4", alpha = 0.7) +
    geom_line(aes(y = cum_var), linewidth = 0.8) +
    geom_point(aes(y = cum_var), size = 1.5) +
    geom_hline(yintercept = 0.80, linetype = "dashed", color = "grey50") +
    geom_hline(yintercept = 0.95, linetype = "dashed", color = "grey50") +
    annotate("text", x = n_show, y = 0.81, label = "80%",
             hjust = 1, size = 3, color = "grey40") +
    annotate("text", x = n_show, y = 0.96, label = "95%",
             hjust = 1, size = 3, color = "grey40") +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      title = paste(title_prefix, "— PCA Scree Plot"),
      x = "Principal Component", y = "Variance Explained"
    ) +
    theme_minimal(base_size = 11)
}


# ------------------------------------------------------------------
# Plot: PCA biplot
# ------------------------------------------------------------------

plot_pca_biplot <- function(pca_fit, meta, title_prefix) {
  scores <- as.data.frame(pca_fit$x[, 1:2])
  scores$timestamp_sec <- meta$timestamp_sec

  var_exp <- pca_fit$sdev^2 / sum(pca_fit$sdev^2)
  pc1_lab <- sprintf("PC1 (%.1f%%)", var_exp[1] * 100)
  pc2_lab <- sprintf("PC2 (%.1f%%)", var_exp[2] * 100)

  # Top loading arrows.
  loadings <- as.data.frame(pca_fit$rotation[, 1:2])
  loadings$feature <- rownames(loadings)
  loadings$magnitude <- sqrt(loadings$PC1^2 + loadings$PC2^2)
  top_load <- loadings |> arrange(desc(magnitude)) |> slice_head(n = 10)

  # Scale arrows to fit the score range.
  score_range <- max(abs(range(scores$PC1)), abs(range(scores$PC2)))
  arrow_scale <- score_range / max(top_load$magnitude) * 0.4

  p <- ggplot(scores, aes(PC1, PC2, color = timestamp_sec)) +
    geom_point(size = 0.3, alpha = 0.4) +
    scale_color_viridis_c(option = "viridis") +
    labs(
      title = paste(title_prefix, "— PCA Biplot"),
      x = pc1_lab, y = pc2_lab, color = "Time (s)"
    ) +
    theme_minimal(base_size = 11)

  # Add loading arrows.
  p <- p +
    geom_segment(
      data = top_load,
      aes(x = 0, y = 0, xend = PC1 * arrow_scale, yend = PC2 * arrow_scale),
      arrow = arrow(length = unit(0.15, "cm")),
      color = "grey30", inherit.aes = FALSE, linewidth = 0.4
    ) +
    geom_text(
      data = top_load,
      aes(x = PC1 * arrow_scale * 1.08, y = PC2 * arrow_scale * 1.08, label = feature),
      color = "grey30", size = 2.2, inherit.aes = FALSE
    )

  p
}


# ------------------------------------------------------------------
# Plot: UMAP
# ------------------------------------------------------------------

run_umap <- function(feat) {
  set.seed(7428)
  umap(as.matrix(feat), n_neighbors = 15, min_dist = 0.1,
       metric = "euclidean", n_threads = parallel::detectCores())
}

plot_umap_time <- function(umap_coords, meta, title_prefix) {
  umap_df <- tibble(
    UMAP1 = umap_coords[, 1],
    UMAP2 = umap_coords[, 2],
    timestamp_sec = meta$timestamp_sec
  )

  ggplot(umap_df, aes(UMAP1, UMAP2, color = timestamp_sec)) +
    geom_point(size = 0.3, alpha = 0.5) +
    scale_color_viridis_c(option = "viridis") +
    labs(
      title = paste(title_prefix, "— UMAP (coloured by time)"),
      x = "UMAP 1", y = "UMAP 2", color = "Time (s)"
    ) +
    theme_minimal(base_size = 11) +
    coord_equal()
}

plot_umap_video <- function(umap_coords, meta, title_prefix) {
  videos <- unique(meta$video)
  if (length(videos) <= 1) return(NULL)

  umap_df <- tibble(
    UMAP1 = umap_coords[, 1],
    UMAP2 = umap_coords[, 2],
    video = meta$video
  )

  ggplot(umap_df, aes(UMAP1, UMAP2, color = video)) +
    geom_point(size = 0.3, alpha = 0.5) +
    labs(
      title = paste(title_prefix, "— UMAP (coloured by video)"),
      x = "UMAP 1", y = "UMAP 2", color = "Video"
    ) +
    theme_minimal(base_size = 11) +
    coord_equal()
}


# ------------------------------------------------------------------
# Feature ranking table
# ------------------------------------------------------------------

build_feature_rank <- function(feat, pca_fit) {
  vars <- tibble(
    column   = names(feat),
    variance = map_dbl(feat, var, na.rm = TRUE)
  )

  # Max absolute correlation with any other feature.
  cor_mat <- cor(feat, use = "pairwise.complete.obs")
  diag(cor_mat) <- 0
  max_abs_corr <- apply(abs(cor_mat), 1, max, na.rm = TRUE)
  vars$max_abs_corr <- max_abs_corr[vars$column]

  # Top PC loading.
  loadings <- abs(pca_fit$rotation)
  # Only include columns that survived scaling (present in rotation).
  pca_cols <- rownames(pca_fit$rotation)
  top_pc      <- rep(NA_character_, nrow(vars))
  pc_loading  <- rep(NA_real_, nrow(vars))
  for (i in seq_len(nrow(vars))) {
    col <- vars$column[i]
    if (col %in% pca_cols) {
      j <- which.max(loadings[col, ])
      top_pc[i]     <- colnames(loadings)[j]
      pc_loading[i] <- pca_fit$rotation[col, j]
    }
  }
  vars$top_pc     <- top_pc
  vars$pc_loading <- pc_loading

  # Parse name components.
  parsed <- map_dfr(vars$column, parse_feature_name)
  vars |>
    left_join(parsed, by = "column") |>
    arrange(desc(variance))
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: Rscript analysis/features.R <landmark_csv_or_directory>")
}

path <- args[1]
if (dir.exists(path)) {
  # Match landmark CSVs but exclude metrics / kp_detail / summary / smooth.
  files <- list.files(path, pattern = "\\.csv$", full.names = TRUE)
  files <- files[!str_detect(basename(files),
                              "(metrics|kp_detail|diag|summary|smooth|feature_rank)\\.csv$")]
  if (length(files) == 0) stop("No landmark CSVs found in ", path)
} else {
  files <- path
}

for (f in files) {
  cat("\n", strrep("=", 60), "\n")
  cat("  Feature analysis:", basename(f), "\n")
  cat(strrep("=", 60), "\n")

  df <- read_csv(f, show_col_types = FALSE)
  tracking <- detect_tracking(names(df))
  cat(sprintf("  Tracking mode: %s\n", tracking))

  coord_cols <- coord_columns(names(df))
  if (length(coord_cols) == 0) {
    cat("  No coordinate columns found — skipping.\n")
    next
  }

  # Prepare feature matrix.
  prep   <- prepare_features(df, coord_cols)
  feat   <- prep$features
  meta   <- df[prep$kept_rows, ] |> select(any_of(METADATA_COLS))

  # Subsample if too large.
  if (nrow(feat) > MAX_ROWS) {
    cat(sprintf("  Subsampling %d → %d rows for PCA/UMAP.\n", nrow(feat), MAX_ROWS))
    set.seed(3194)
    idx  <- sample(nrow(feat), MAX_ROWS)
    feat <- feat[idx, , drop = FALSE]
    meta <- meta[idx, , drop = FALSE]
  }

  stem  <- str_remove(f, "\\.csv$")
  title <- basename(stem)

  # 1. Variance.
  cat("  Plotting variance...\n")
  p_var <- plot_variance(feat, title)
  ggsave(paste0(stem, "_variance.png"), p_var,
         width = 10, height = max(6, length(coord_cols) * 0.12), dpi = 150)

  # 2. Correlation.
  cat("  Plotting correlation...\n")
  p_cor <- plot_correlation(feat, title)
  ggsave(paste0(stem, "_correlation.png"), p_cor,
         width = 12, height = 11, dpi = 150)

  # 3. PCA.
  cat("  Running PCA...\n")
  pca_fit <- run_pca(feat)

  p_scree <- plot_pca_scree(pca_fit, title)
  ggsave(paste0(stem, "_pca_scree.png"), p_scree,
         width = 8, height = 5, dpi = 150)

  p_biplot <- plot_pca_biplot(pca_fit, meta, title)
  ggsave(paste0(stem, "_pca_biplot.png"), p_biplot,
         width = 8, height = 7, dpi = 150)

  # 4. UMAP.
  cat("  Running UMAP...\n")
  umap_coords <- run_umap(scale(feat))

  p_umap_t <- plot_umap_time(umap_coords, meta, title)
  ggsave(paste0(stem, "_umap.png"), p_umap_t,
         width = 8, height = 7, dpi = 150)

  p_umap_v <- plot_umap_video(umap_coords, meta, title)
  if (!is.null(p_umap_v)) {
    ggsave(paste0(stem, "_umap_video.png"), p_umap_v,
           width = 8, height = 7, dpi = 150)
  }

  # 5. Feature ranking table.
  cat("  Writing feature ranking...\n")
  rank_df <- build_feature_rank(feat, pca_fit)
  write_csv(rank_df, paste0(stem, "_feature_rank.csv"))

  cat("  Done. Outputs written to", dirname(f), "\n")
}

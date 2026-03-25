#!/usr/bin/env Rscript
# Compare two pipeline runs using their *_summary.json files.
#
# Usage:
#   Rscript analysis/compare.R output/run_a_summary.json output/run_b_summary.json
#   Rscript analysis/compare.R output/run_a_metrics.csv  output/run_b_metrics.csv
#
# When given metrics CSVs, summary JSONs are generated on the fly.
# Outputs:
#   - Console comparison table
#   - <output_dir>/comparison.csv
#   - <output_dir>/comparison_jitter.png
#   - <output_dir>/comparison_detection.png

library(tidyverse)
library(jsonlite)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

flatten_json <- function(j, prefix = "") {
  out <- list()
  for (name in names(j)) {
    key <- if (prefix == "") name else paste0(prefix, ".", name)
    val <- j[[name]]
    if (is.list(val) && !is.null(names(val))) {
      out <- c(out, flatten_json(val, key))
    } else {
      out[[key]] <- val
    }
  }
  out
}

load_summary <- function(path) {
  if (str_ends(path, "_summary\\.json")) {
    return(fromJSON(path))
  }
  # Generate summary from metrics CSV on the fly
  source("analysis/summary.R", local = TRUE)
  df <- read_csv(path, show_col_types = FALSE)
  summarise_metrics(df)
}


# ------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------

compare_summaries <- function(s_a, s_b, label_a = "Run A", label_b = "Run B") {
  flat_a <- flatten_json(s_a)
  flat_b <- flatten_json(s_b)
  all_keys <- union(names(flat_a), names(flat_b))

  tibble(
    metric = all_keys,
    !!label_a := map_dbl(all_keys, ~ as.numeric(flat_a[[.x]] %||% NA)),
    !!label_b := map_dbl(all_keys, ~ as.numeric(flat_b[[.x]] %||% NA)),
  ) |>
    mutate(
      delta = .data[[label_b]] - .data[[label_a]],
      pct_change = round(delta / abs(.data[[label_a]]) * 100, 1)
    )
}


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

plot_jitter_comparison <- function(metrics_a, metrics_b, label_a, label_b) {
  bind_a <- metrics_a |>
    select(timestamp_sec, body_jitter_px, hand_L_jitter_px, hand_R_jitter_px) |>
    mutate(run = label_a)
  bind_b <- metrics_b |>
    select(timestamp_sec, body_jitter_px, hand_L_jitter_px, hand_R_jitter_px) |>
    mutate(run = label_b)

  combined <- bind_rows(bind_a, bind_b) |>
    pivot_longer(
      c(body_jitter_px, hand_L_jitter_px, hand_R_jitter_px),
      names_to = "part", values_to = "jitter"
    ) |>
    mutate(part = recode(part,
      body_jitter_px = "Body",
      hand_L_jitter_px = "Hand L",
      hand_R_jitter_px = "Hand R"
    )) |>
    filter(!is.na(jitter))

  ggplot(combined, aes(jitter, fill = run)) +
    geom_density(alpha = 0.5) +
    facet_wrap(~part, scales = "free") +
    labs(
      title = "Jitter Distribution Comparison",
      x = "Jitter (px, sum over keypoints)", y = "Density", fill = NULL
    ) +
    theme_minimal(base_size = 11)
}


plot_detection_comparison <- function(metrics_a, metrics_b, label_a, label_b) {
  det_a <- metrics_a |>
    summarise(
      body = mean(body_detected, na.rm = TRUE),
      synth = mean(n_hands_synthetic > 0, na.rm = TRUE),
      recrop = mean(n_hands_recrop > 0, na.rm = TRUE),
      carry = mean(body_carry, na.rm = TRUE)
    ) |>
    mutate(run = label_a)

  det_b <- metrics_b |>
    summarise(
      body = mean(body_detected, na.rm = TRUE),
      synth = mean(n_hands_synthetic > 0, na.rm = TRUE),
      recrop = mean(n_hands_recrop > 0, na.rm = TRUE),
      carry = mean(body_carry, na.rm = TRUE)
    ) |>
    mutate(run = label_b)

  combined <- bind_rows(det_a, det_b) |>
    pivot_longer(-run, names_to = "metric", values_to = "rate") |>
    mutate(metric = recode(metric,
      body = "Body detected",
      synth = "Synthetic hands",
      recrop = "Recrop hands",
      carry = "Body carry"
    ))

  ggplot(combined, aes(metric, rate, fill = run)) +
    geom_col(position = "dodge") +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = "Detection Rate Comparison",
      x = NULL, y = "Rate", fill = NULL
    ) +
    theme_minimal(base_size = 11)
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript analysis/compare.R <run_a> <run_b> [label_a] [label_b]")
}

path_a <- args[1]
path_b <- args[2]
label_a <- if (length(args) >= 3) args[3] else tools::file_path_sans_ext(basename(path_a))
label_b <- if (length(args) >= 4) args[4] else tools::file_path_sans_ext(basename(path_b))

# Load summaries
s_a <- load_summary(path_a)
s_b <- load_summary(path_b)

# Print comparison table
comp <- compare_summaries(s_a, s_b, label_a, label_b)
cat("\n")
cat(strrep("=", 70), "\n")
cat("  Comparison:", label_a, "vs", label_b, "\n")
cat(strrep("=", 70), "\n")
print(comp, n = Inf)

# Write CSV
out_dir <- dirname(path_a)
comp_csv <- file.path(out_dir, "comparison.csv")
write_csv(comp, comp_csv)
cat("\n  Wrote:", comp_csv, "\n")

# If metrics CSVs are available, generate plots
if (str_ends(path_a, "_metrics\\.csv") && str_ends(path_b, "_metrics\\.csv")) {
  df_a <- read_csv(path_a, show_col_types = FALSE)
  df_b <- read_csv(path_b, show_col_types = FALSE)

  p1 <- plot_jitter_comparison(df_a, df_b, label_a, label_b)
  p1_path <- file.path(out_dir, "comparison_jitter.png")
  ggsave(p1_path, p1, width = 10, height = 5, dpi = 150)
  cat("  Wrote:", p1_path, "\n")

  p2 <- plot_detection_comparison(df_a, df_b, label_a, label_b)
  p2_path <- file.path(out_dir, "comparison_detection.png")
  ggsave(p2_path, p2, width = 8, height = 5, dpi = 150)
  cat("  Wrote:", p2_path, "\n")
}

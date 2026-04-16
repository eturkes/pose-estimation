#!/usr/bin/env Rscript
# Longitudinal comparison of clinical features across sessions.
#
# Joins aggregated clinical features with a session metadata CSV
# (video → patient_id + session_date) to track recovery over time.
# Optionally overlays clinical scores on a secondary axis.
#
# Usage:
#   Rscript analysis/longitudinal.R output/ sessions.csv
#   Rscript analysis/longitudinal.R output/ sessions.csv clinical_scores.csv
#
# Outputs (written into the clinical features directory):
#   <prefix>_longitudinal_summary.csv        — tidy table of values, deltas, flags
#   <prefix>_longitudinal_<patient_id>.png   — line plots per patient

library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(purrr)
library(ggplot2)
library(tibble)

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

# Features to plot per patient (top N by variance across sessions).
N_PLOT_FEATURES <- 6

# ------------------------------------------------------------------
# Delta / flag computation
# ------------------------------------------------------------------

#' Compute deltas from baseline and flag notable changes.
#'
#' For each patient × feature, the baseline is the first session.
#' A change is flagged when it exceeds `flag_sd` SDs of the baseline
#' session's value (using the per-video SD of that feature as a
#' proxy for measurement noise).  If the feature has no within-video
#' SD for the baseline, flagging is skipped.
compute_deltas <- function(long, flag_sd = 1) {
  long |>
    arrange(patient_id, feature, session_date) |>
    group_by(patient_id, feature) |>
    mutate(
      baseline_value = first(value),
      delta_from_baseline = value - baseline_value,
      pct_change = if_else(
        abs(baseline_value) > 1e-12,
        delta_from_baseline / abs(baseline_value) * 100,
        NA_real_
      ),
      flagged = {
        bl_sd <- first(value_sd)
        if_else(
          !is.na(bl_sd) & bl_sd > 1e-12,
          abs(delta_from_baseline) > flag_sd * bl_sd,
          NA
        )
      }
    ) |>
    ungroup()
}

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

#' Select the most variable features for a given patient.
select_top_features <- function(summary_df, patient, n = N_PLOT_FEATURES) {
  summary_df |>
    filter(patient_id == patient) |>
    group_by(feature) |>
    summarise(range = max(value, na.rm = TRUE) - min(value, na.rm = TRUE),
              .groups = "drop") |>
    filter(is.finite(range), range > 0) |>
    arrange(desc(range)) |>
    slice_head(n = n) |>
    pull(feature)
}

#' Plot longitudinal trajectories for one patient.
#'
#' @param summary_df The longitudinal summary table (long format).
#' @param patient    Patient ID string.
#' @param features   Character vector of feature names to plot.
#' @param scores_df  Optional tidy scores table (patient_id, session_date,
#'                   score_name, score_value) for overlay.
plot_patient <- function(summary_df, patient, features, scores_df = NULL) {
  sub <- summary_df |>
    filter(patient_id == patient, feature %in% features) |>
    mutate(feature = factor(feature, levels = features))

  if (nrow(sub) == 0) return(NULL)

  p <- ggplot(sub, aes(session_date, value)) +
    geom_line(linewidth = 0.6) +
    geom_point(aes(shape = ifelse(flagged %in% TRUE, "flagged", "normal")),
               size = 2) +
    scale_shape_manual(
      values = c(flagged = 17, normal = 16),
      guide = "none"
    ) +
    facet_wrap(~feature, scales = "free_y", ncol = 2) +
    labs(
      title = paste("Patient", patient, "— Longitudinal Features"),
      x = "Session date", y = NULL
    ) +
    theme_minimal(base_size = 10)

  # Overlay clinical scores on a secondary axis if available.
  if (!is.null(scores_df)) {
    sc <- scores_df |> filter(patient_id == patient)
    if (nrow(sc) > 0) {
      p <- p +
        geom_line(
          data = sc |>
            # Repeat score data in each facet.
            crossing(feature = factor(features, levels = features)),
          aes(session_date, score_value, color = score_name),
          linewidth = 0.4, linetype = "dashed", inherit.aes = FALSE
        ) +
        geom_point(
          data = sc |>
            crossing(feature = factor(features, levels = features)),
          aes(session_date, score_value, color = score_name),
          size = 1.5, inherit.aes = FALSE
        ) +
        labs(color = "Clinical score")
    }
  }

  p
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(paste(
    "Usage: Rscript analysis/longitudinal.R",
    "<clinical_csv_or_directory> <sessions.csv> [clinical_scores.csv]"
  ))
}

clin_path    <- args[1]
session_path <- args[2]
scores_path  <- if (length(args) >= 3) args[3] else NULL

# --- Load session metadata ---
if (!file.exists(session_path)) {
  stop("Session metadata file not found: ", session_path)
}
sessions <- read_csv(session_path, show_col_types = FALSE)
required <- c("video", "patient_id", "session_date")
missing  <- setdiff(required, names(sessions))
if (length(missing) > 0) {
  stop("sessions.csv is missing required columns: ",
       paste(missing, collapse = ", "))
}
sessions <- sessions |>
  mutate(session_date = as.Date(session_date))
cat(sprintf("Loaded %d sessions for %d patients.\n",
            nrow(sessions), n_distinct(sessions$patient_id)))

# --- Optionally load clinical scores ---
scores <- NULL
scores_long <- NULL
if (!is.null(scores_path)) {
  if (!file.exists(scores_path)) {
    warning("Clinical scores file not found: ", scores_path, " — skipping.")
  } else {
    scores <- read_csv(scores_path, show_col_types = FALSE)
    if (!"video" %in% names(scores)) {
      warning("Clinical scores CSV has no 'video' column — skipping.")
      scores <- NULL
    } else {
      score_cols <- setdiff(names(scores), "video")
      score_cols <- score_cols[map_lgl(score_cols, \(c) is.numeric(scores[[c]]))]
      if (length(score_cols) == 0) {
        warning("No numeric score columns found — skipping.")
        scores <- NULL
      } else {
        cat(sprintf("Loaded %d scores: %s\n",
                    length(score_cols), paste(score_cols, collapse = ", ")))
        # Pivot to long and attach session info.
        scores_long <- scores |>
          pivot_longer(all_of(score_cols),
                       names_to = "score_name", values_to = "score_value") |>
          inner_join(sessions |> select(video, patient_id, session_date),
                     by = "video")
      }
    }
  }
}

# --- Discover clinical feature CSVs ---
if (dir.exists(clin_path)) {
  frame_files <- list.files(clin_path, pattern = "_clinical\\.csv$",
                            full.names = TRUE)
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

# --- Aggregate per video ---
cat("Aggregating per-frame clinical features per video...\n")
frame_agg <- map(frame_files, \(f) {
  df <- read_csv(f, show_col_types = FALSE)
  aggregate_per_video(df, METADATA_COLS)
}) |> bind_rows()

if (nrow(frame_agg) == 0) stop("No frame-level features to aggregate.")

if (length(win_files) > 0) {
  cat("Aggregating per-window clinical features per video...\n")
  win_agg <- map(win_files, \(f) {
    df <- read_csv(f, show_col_types = FALSE)
    aggregate_per_video(df, WINDOW_META)
  }) |> bind_rows()

  if (nrow(win_agg) > 0 && "video" %in% names(win_agg)) {
    frame_agg <- frame_agg |>
      left_join(win_agg, by = "video", suffix = c("", ".win")) |>
      select(-ends_with(".win"))
  }
}

# --- Join with session metadata ---
cat("Joining with session metadata...\n")
agg_videos     <- unique(frame_agg$video)
session_videos <- unique(sessions$video)
matched        <- intersect(agg_videos, session_videos)
unmatched_feat <- setdiff(agg_videos, session_videos)
unmatched_sess <- setdiff(session_videos, agg_videos)

if (length(unmatched_feat) > 0) {
  warning(sprintf(
    "%d feature videos have no session entry: %s",
    length(unmatched_feat), paste(unmatched_feat, collapse = ", ")
  ))
}
if (length(unmatched_sess) > 0) {
  warning(sprintf(
    "%d session entries have no matching features: %s",
    length(unmatched_sess), paste(unmatched_sess, collapse = ", ")
  ))
}
if (length(matched) == 0) {
  stop("No videos matched between features and sessions.")
}
cat(sprintf("Matched %d videos across %d patients.\n",
            length(matched),
            n_distinct(sessions$patient_id[sessions$video %in% matched])))

joined <- inner_join(frame_agg, sessions, by = "video")

# Identify numeric aggregated feature columns.
agg_feat_cols <- setdiff(names(frame_agg), "video")
agg_feat_cols <- agg_feat_cols[map_lgl(agg_feat_cols, \(c) is.numeric(joined[[c]]))]

# Keep only the __mean columns for the main trajectory (simpler to
# interpret); SD columns are retained for flagging.
mean_cols <- agg_feat_cols[str_detect(agg_feat_cols, "__mean$")]
sd_cols   <- agg_feat_cols[str_detect(agg_feat_cols, "__sd$")]

if (length(mean_cols) == 0) {
  stop("No aggregated mean features found.")
}

# --- Pivot to long format ---
# Mean values
long_mean <- joined |>
  select(video, patient_id, session_date, all_of(mean_cols)) |>
  pivot_longer(all_of(mean_cols), names_to = "feature", values_to = "value")

# SD values (for flagging)
long_sd <- joined |>
  select(video, patient_id, session_date, all_of(sd_cols)) |>
  pivot_longer(all_of(sd_cols), names_to = "feature_sd", values_to = "value_sd") |>
  # Match SD column to its mean counterpart.
  mutate(feature = str_replace(feature_sd, "__sd$", "__mean"))

long <- long_mean |>
  left_join(
    long_sd |> select(video, feature, value_sd),
    by = c("video", "feature")
  )

# Drop features that are constant or all-NA across the matched set.
long <- long |>
  group_by(feature) |>
  filter(sum(!is.na(value) & is.finite(value)) >= 2) |>
  ungroup()

if (nrow(long) == 0) stop("No variable features remain.")

# --- Compute deltas and flags ---
cat("Computing deltas from baseline...\n")
summary_tbl <- compute_deltas(long)

# --- Output prefix ---
out_prefix <- file.path(out_dir, "clinical")

# --- Write summary CSV ---
out_csv <- paste0(out_prefix, "_longitudinal_summary.csv")
out_tbl <- summary_tbl |>
  select(patient_id, feature, session_date, video, value,
         delta_from_baseline, pct_change, flagged) |>
  arrange(patient_id, feature, session_date)
write_csv(out_tbl, out_csv)
cat(sprintf("Wrote %d rows → %s\n", nrow(out_tbl), out_csv))

# --- Console summary ---
cat("\n", strrep("=", 60), "\n")
cat("  Longitudinal Summary\n")
cat(strrep("=", 60), "\n")

patients <- sort(unique(summary_tbl$patient_id))
for (pid in patients) {
  pat <- summary_tbl |> filter(patient_id == pid)
  n_sessions <- n_distinct(pat$session_date)
  n_flagged  <- sum(pat$flagged %in% TRUE)

  # Identify the feature with largest absolute % change.
  best <- pat |>
    filter(!is.na(pct_change), is.finite(pct_change)) |>
    arrange(desc(abs(pct_change))) |>
    slice_head(n = 1)

  cat(sprintf("\n  Patient %s: %d sessions, %d flagged changes\n",
              pid, n_sessions, n_flagged))

  if (nrow(best) > 0) {
    direction <- if (best$pct_change > 0) "increase" else "decrease"
    cat(sprintf("    Largest change: %s (%+.1f%% %s)\n",
                best$feature, best$pct_change, direction))
  }

  if (n_sessions < 2) {
    cat("    (single session — no deltas computed)\n")
  }
}

# --- Per-patient plots ---
cat("\nGenerating per-patient plots...\n")
for (pid in patients) {
  feats <- select_top_features(summary_tbl, pid)
  if (length(feats) == 0) {
    cat(sprintf("  Patient %s: no variable features to plot — skipping.\n", pid))
    next
  }

  p <- plot_patient(summary_tbl, pid, feats, scores_long)
  if (is.null(p)) next

  n_panels <- length(feats)
  plot_height <- max(4, ceiling(n_panels / 2) * 3)

  out_png <- sprintf("%s_longitudinal_%s.png", out_prefix, pid)
  ggsave(out_png, p, width = 10, height = plot_height, dpi = 150)
  cat(sprintf("  Wrote → %s\n", out_png))
}

cat("\nDone.\n")

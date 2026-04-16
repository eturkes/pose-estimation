# Shared helpers for the analysis pipeline.
#
# Source via:
#   source(file.path(dirname(sys.frame(1)$ofile), "utils.R"))
# or, when sys.frame() is unavailable (interactive REPL):
#   source("analysis/utils.R")
#
# The script_dir() helper below works in both contexts so callers
# can write:
#   source(file.path(script_dir(), "utils.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(tibble)
})

# ------------------------------------------------------------------
# Canonical metadata column names
# ------------------------------------------------------------------

METADATA_COLS <- c("video", "frame_idx", "timestamp_sec", "person_idx")
WINDOW_META   <- c("video", "person_idx", "window_start_sec", "window_end_sec")

# ------------------------------------------------------------------
# Path discovery
# ------------------------------------------------------------------

#' Locate the directory containing the currently running script.
#'
#' Works whether the script is invoked via Rscript, sourced from
#' another script, or run interactively in the REPL.  Falls back to
#' "analysis" relative to the working directory.
script_dir <- function() {
  # Rscript path (works for command-line invocation)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  match <- grep(file_arg, args)
  if (length(match) > 0) {
    return(normalizePath(dirname(sub(file_arg, "", args[match])), winslash = "/"))
  }
  # Sourced from another script
  for (i in seq_len(sys.nframe())) {
    f <- sys.frame(i)$ofile
    if (!is.null(f) && nzchar(f)) {
      return(normalizePath(dirname(f), winslash = "/"))
    }
  }
  # Interactive fallback
  candidate <- "analysis"
  if (dir.exists(candidate)) return(normalizePath(candidate, winslash = "/"))
  return(getwd())
}

# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

#' Aggregate a clinical-features data frame to one row per video.
#'
#' For every numeric column not listed in *meta_cols*, computes the
#' summary statistics in *fns* and names the resulting columns
#' "<column>__<fn>".  When *fns* is NULL, defaults to mean / median /
#' sd / min / max (matching the per-frame aggregation used in
#' clinical_correlation.R, longitudinal.R, clinical_dimreduce.R, and
#' compare_clinical.R).
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

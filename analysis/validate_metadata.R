#!/usr/bin/env Rscript
# Validate a completed clinical_scores.csv or sessions.csv against
# the clinical feature CSVs in the output directory.
#
# Checks: required columns present, no duplicate videos, video names
# match those in the clinical feature CSVs, dates parse correctly,
# scores are numeric.  Reports errors and warnings, exits 0 if valid
# or 1 if errors found.
#
# Usage:
#   Rscript analysis/validate_metadata.R clinical_scores.csv output/
#   Rscript analysis/validate_metadata.R sessions.csv output/

library(readr)
library(stringr)
library(purrr)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript analysis/validate_metadata.R <metadata.csv> <output_dir>")
}
meta_path <- args[1]
out_dir   <- args[2]

if (!file.exists(meta_path)) stop("File not found: ", meta_path)
if (!dir.exists(out_dir))    stop("Directory not found: ", out_dir)

# ------------------------------------------------------------------
# Discover expected video names from clinical CSVs
# ------------------------------------------------------------------

frame_files <- list.files(out_dir, pattern = "_clinical\\.csv$",
                          full.names = TRUE)
frame_files <- frame_files[!str_detect(basename(frame_files),
                                       "_windows\\.csv$")]

if (length(frame_files) == 0) {
  stop("No *_clinical.csv files found in ", out_dir)
}

expected_videos <- map_chr(frame_files, \(f) {
  d <- read_csv(f, show_col_types = FALSE, n_max = 1)
  if ("video" %in% names(d) && nrow(d) > 0) {
    d$video[1]
  } else {
    str_remove(basename(f), "_clinical\\.csv$")
  }
})
expected_videos <- sort(unique(expected_videos))

# ------------------------------------------------------------------
# Read metadata
# ------------------------------------------------------------------

meta <- read_csv(meta_path, show_col_types = FALSE)

errors   <- character()
warnings <- character()

# ------------------------------------------------------------------
# Detect file type and validate
# ------------------------------------------------------------------

is_sessions <- "patient_id" %in% names(meta) ||
               "session_date" %in% names(meta)
file_type <- if (is_sessions) "sessions" else "clinical_scores"

cat(sprintf("Validating %s as '%s' metadata.\n", meta_path, file_type))
cat(sprintf("Expected videos from clinical CSVs: %d\n", length(expected_videos)))

# --- Common checks ------------------------------------------------

# 'video' column required
if (!"video" %in% names(meta)) {
  errors <- c(errors, "Missing required column: 'video'.")
} else {
  # Duplicate videos
  dup <- meta$video[duplicated(meta$video)]
  if (length(dup) > 0) {
    errors <- c(errors, sprintf(
      "Duplicate video entries: %s",
      paste(unique(dup), collapse = ", ")
    ))
  }

  # Blank / NA videos
  bad_vid <- is.na(meta$video) | str_trim(meta$video) == ""
  if (any(bad_vid)) {
    errors <- c(errors, sprintf(
      "%d row(s) have blank or NA 'video' values.", sum(bad_vid)
    ))
  }

  # Match against expected
  meta_vids <- meta$video[!is.na(meta$video)]
  missing_in_meta <- setdiff(expected_videos, meta_vids)
  extra_in_meta   <- setdiff(meta_vids, expected_videos)
  if (length(missing_in_meta) > 0) {
    warnings <- c(warnings, sprintf(
      "Videos in clinical CSVs but not in metadata: %s",
      paste(missing_in_meta, collapse = ", ")
    ))
  }
  if (length(extra_in_meta) > 0) {
    warnings <- c(warnings, sprintf(
      "Videos in metadata but not in clinical CSVs: %s",
      paste(extra_in_meta, collapse = ", ")
    ))
  }
}

# --- Sessions-specific checks ------------------------------------

if (file_type == "sessions") {
  required_cols <- c("video", "patient_id", "session_date")
  missing_cols <- setdiff(required_cols, names(meta))
  if (length(missing_cols) > 0) {
    errors <- c(errors, sprintf(
      "Missing required columns: %s",
      paste(missing_cols, collapse = ", ")
    ))
  }

  if ("patient_id" %in% names(meta)) {
    bad_pid <- is.na(meta$patient_id) | str_trim(meta$patient_id) == ""
    if (any(bad_pid)) {
      errors <- c(errors, sprintf(
        "%d row(s) have blank or NA 'patient_id'.", sum(bad_pid)
      ))
    }
  }

  if ("session_date" %in% names(meta)) {
    parsed <- suppressWarnings(as.Date(meta$session_date))
    bad_dates <- !is.na(meta$session_date) & is.na(parsed)
    if (any(bad_dates)) {
      errors <- c(errors, sprintf(
        "%d row(s) have unparseable 'session_date' values (expected YYYY-MM-DD): %s",
        sum(bad_dates),
        paste(meta$session_date[bad_dates], collapse = ", ")
      ))
    }
    all_na <- all(is.na(meta$session_date))
    if (all_na) {
      errors <- c(errors,
        "All 'session_date' values are NA — please fill in dates.")
    }
  }
}

# --- Clinical-scores-specific checks -----------------------------

if (file_type == "clinical_scores") {
  score_cols <- setdiff(names(meta), "video")
  if (length(score_cols) == 0) {
    errors <- c(errors,
      "No score columns found (expected at least one besides 'video').")
  } else {
    non_numeric <- score_cols[!map_lgl(score_cols, \(c) is.numeric(meta[[c]]))]
    if (length(non_numeric) > 0) {
      errors <- c(errors, sprintf(
        "Score columns should be numeric but are not: %s",
        paste(non_numeric, collapse = ", ")
      ))
    }

    all_na_cols <- score_cols[map_lgl(score_cols, \(c) all(is.na(meta[[c]])))]
    if (length(all_na_cols) > 0) {
      warnings <- c(warnings, sprintf(
        "Score columns that are entirely NA (not yet filled in?): %s",
        paste(all_na_cols, collapse = ", ")
      ))
    }
  }
}

# ------------------------------------------------------------------
# Report results
# ------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("  Validation Results\n")
cat(strrep("=", 60), "\n\n")

if (length(errors) > 0) {
  cat("ERRORS:\n")
  for (e in errors) cat(sprintf("  [x] %s\n", e))
  cat("\n")
}

if (length(warnings) > 0) {
  cat("WARNINGS:\n")
  for (w in warnings) cat(sprintf("  [!] %s\n", w))
  cat("\n")
}

if (length(errors) == 0 && length(warnings) == 0) {
  cat("All checks passed. No errors or warnings.\n")
} else if (length(errors) == 0) {
  cat("No errors found (warnings above are non-blocking).\n")
}

cat(sprintf("\nRows: %d | Columns: %s\n", nrow(meta),
            paste(names(meta), collapse = ", ")))

if (length(errors) > 0) {
  cat("\nValidation FAILED.\n")
  quit(status = 1)
} else {
  cat("\nValidation PASSED.\n")
  quit(status = 0)
}

#!/usr/bin/env Rscript
# Generate template metadata CSVs for clinical correlation and
# longitudinal analysis.
#
# Scans the output directory for unique video names across
# *_clinical.csv files, then writes:
#   clinical_scores_template.csv  — video + placeholder score columns
#   sessions_template.csv         — video + patient_id + session_date
#
# Usage:
#   Rscript analysis/make_templates.R output/

library(readr)
library(stringr)
library(purrr)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript analysis/make_templates.R <output_dir>")
}
out_dir <- args[1]
if (!dir.exists(out_dir)) stop("The directory does not exist: ", out_dir)

# ------------------------------------------------------------------
# Discover video names
# ------------------------------------------------------------------

frame_files <- list.files(out_dir, pattern = "_clinical\\.csv$",
                          full.names = TRUE)
frame_files <- frame_files[!str_detect(basename(frame_files),
                                       "_windows\\.csv$")]

if (length(frame_files) == 0) {
  stop("The directory contains no *_clinical.csv files: ", out_dir)
}

videos <- map_chr(frame_files, \(f) {
  d <- read_csv(f, show_col_types = FALSE, n_max = 1)
  if ("video" %in% names(d) && nrow(d) > 0) {
    d$video[1]
  } else {
    str_remove(basename(f), "_clinical\\.csv$")
  }
})
videos <- sort(unique(videos))

cat(sprintf("The script found %d videos:\n", length(videos)))
for (v in videos) cat(sprintf("  %s\n", v))

# ------------------------------------------------------------------
# Write clinical_scores_template.csv
# ------------------------------------------------------------------

scores_path <- file.path(out_dir, "clinical_scores_template.csv")
scores_tpl <- data.frame(
  video  = videos,
  GRASSP = NA_real_,
  UEMS   = NA_real_,
  SCIM   = NA_real_,
  stringsAsFactors = FALSE
)
write_csv(scores_tpl, scores_path)
cat(sprintf("\nThe script wrote %s.\n", scores_path))

# ------------------------------------------------------------------
# Write sessions_template.csv
# ------------------------------------------------------------------

sessions_path <- file.path(out_dir, "sessions_template.csv")
sessions_tpl <- data.frame(
  video        = videos,
  patient_id   = NA_character_,
  session_date = NA_character_,
  stringsAsFactors = FALSE
)
write_csv(sessions_tpl, sessions_path)
cat(sprintf("The script wrote %s.\n", sessions_path))

# ------------------------------------------------------------------
# Instructions
# ------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("  Next steps\n")
cat(strrep("=", 60), "\n\n")

cat("1. Complete clinical_scores_template.csv:\n")
cat("   - Each row corresponds to one video (assessment session).\n")
cat("   - Replace NA values with numeric clinical scores.\n")
cat("   - You can add, rename, or remove score columns as needed.\n")
cat("     clinical_correlation.R uses every column except 'video'.\n")
cat("   - When the file is complete, save it as 'clinical_scores.csv'.\n\n")

cat("2. Complete sessions_template.csv:\n")
cat("   - patient_id identifies the patient (for example, 'P01').\n")
cat("     Assign the same ID to all videos from the same patient.\n")
cat("   - session_date contains the session date in ISO 8601 format (YYYY-MM-DD).\n")
cat("   - When the file is complete, save it as 'sessions.csv'.\n\n")

cat("3. Validate your completed files:\n")
cat("   Rscript analysis/validate_metadata.R clinical_scores.csv output/\n")
cat("   Rscript analysis/validate_metadata.R sessions.csv output/\n\n")

cat("4. Run downstream analyses:\n")
cat("   Rscript analysis/clinical_correlation.R output/ clinical_scores.csv\n")
cat("   Rscript analysis/longitudinal.R output/ sessions.csv\n")
cat("   Rscript analysis/longitudinal.R output/ sessions.csv clinical_scores.csv\n\n")

cat("The script finished.\n")

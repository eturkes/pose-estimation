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
#   Rscript analysis/make_templates.R output/ [--mode 2d|3d]
#
# 3D input (*_clinical_3d.csv) writes clinical_scores_3d_template.csv + sessions_3d_template.csv;
# a directory holding both 2D and 3D artifacts needs --mode.

library(readr)
library(stringr)
library(purrr)

# Shared helpers (discover_clinical, take_mode_arg, read_clinical).
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

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

cli <- take_mode_arg(commandArgs(trailingOnly = TRUE))
args <- cli$args
if (length(args) < 1) {
  stop("Usage: Rscript analysis/make_templates.R <output_dir> [--mode 2d|3d]")
}
out_dir <- args[1]
if (!dir.exists(out_dir)) stop("The directory does not exist: ", out_dir)

# ------------------------------------------------------------------
# Discover video names
# ------------------------------------------------------------------

found <- discover_clinical(out_dir, cli$mode)
videos <- clinical_videos(found)

cat(sprintf("The script found %d videos:\n", length(videos)))
for (v in videos) cat(sprintf("  %s\n", v))

# ------------------------------------------------------------------
# Write clinical_scores_template.csv
# ------------------------------------------------------------------

scores_path <- file.path(out_dir, paste0("clinical_scores", found$tag, "_template.csv"))
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

sessions_path <- file.path(out_dir, paste0("sessions", found$tag, "_template.csv"))
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

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

# ------------------------------------------------------------------
# Clinical artifact discovery — the one mode-aware reader
# ------------------------------------------------------------------

# Suffixes per mode, selected positively: `_clinical.csv` never matches `_clinical_3d.csv`, so a
# consumer cannot pool normalised image-plane 2D with metric 3D by widening its own glob.
CLINICAL_SUFFIXES <- list(
  `2d` = c(frame = "_clinical.csv", windows = "_clinical_windows.csv"),
  `3d` = c(frame = "_clinical_3d.csv", windows = "_clinical_3d_windows.csv")
)

#' Split an optional `--mode 2d|3d` out of positional CLI arguments.
take_mode_arg <- function(args) {
  at <- which(args == "--mode")
  if (length(at) == 0) return(list(mode = NULL, args = args))
  if (length(at) > 1 || at == length(args)) stop("--mode takes one value: 2d or 3d.")
  mode <- args[at + 1]
  if (!mode %in% names(CLINICAL_SUFFIXES)) stop("--mode must be 2d or 3d, not '", mode, "'.")
  list(mode = mode, args = args[-c(at, at + 1)])
}

#' Discover one mode's clinical artifacts under *path*: a directory, or one frame CSV.
#'
#' Returns `mode`, `frame_files`, `win_files`, `out_dir` and `tag` ("" for 2D, so the legacy
#' output names stay; "_3d" for 3D). Input holding both modes stops unless *mode* names one,
#' so no aggregate mixes the two metrics.
discover_clinical <- function(path, mode = NULL) {
  if (dir.exists(path)) {
    entries <- list.files(path)
    pick <- \(suffix) file.path(path, entries[endsWith(entries, suffix)])
    found <- map(CLINICAL_SUFFIXES, \(s) list(frame = pick(s[["frame"]]),
                                              windows = pick(s[["windows"]])))
    out_dir <- path
  } else {
    found <- map(CLINICAL_SUFFIXES, \(s) {
      if (!endsWith(path, s[["frame"]])) return(list(frame = character(), windows = character()))
      stem <- substr(path, 1, nchar(path) - nchar(s[["frame"]]))
      sibling <- paste0(stem, s[["windows"]])
      list(frame = path, windows = sibling[file.exists(sibling)])
    })
    out_dir <- dirname(path)
  }
  present <- names(found)[map_lgl(found, \(f) length(f$frame) > 0)]
  if (is.null(mode)) {
    if (length(present) > 1) {
      stop("The input holds both 2D and 3D clinical artifacts: ", path,
           ". Pass --mode 2d or --mode 3d.")
    }
    mode <- if (length(present) == 1) present else "2d"
  }
  if (length(found[[mode]]$frame) == 0) {
    stop("The input contains no *", CLINICAL_SUFFIXES[[mode]][["frame"]], " files: ", path)
  }
  list(mode = mode, frame_files = found[[mode]]$frame, win_files = found[[mode]]$windows,
       out_dir = out_dir, tag = if (mode == "2d") "" else "_3d")
}

#' The video stem of one clinical artifact path.
clinical_stem <- function(path, mode, level = "frame") {
  name <- basename(path)
  substr(name, 1, nchar(name) - nchar(CLINICAL_SUFFIXES[[mode]][[level]]))
}

#' Read and row-bind clinical CSVs, dropping empty files.
read_clinical <- function(files) {
  map(files, \(f) {
    d <- readr::read_csv(f, show_col_types = FALSE)
    if (nrow(d) == 0) NULL else d
  }) |> compact() |> bind_rows()
}


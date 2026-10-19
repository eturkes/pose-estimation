# M4.5a D04 driver: run the shipped bout detector + features over a synthetic
# bank for every grid cell (contract .agent/archive/contract-m4u5a.md).
# Usage: Rscript analysis/bout_qualification.R <request.json> <response.json>
# Request + grading live in src/pose_estimation/bout_qualification.py; this
# script only executes the producer's own functions.

args <- commandArgs(trailingOnly = TRUE)
request_path <- args[1]
response_path <- args[2]
clinical <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])),
                      "clinical_features.R")
# Sourced with no landmark path, the producer stops at its usage guard after
# every function is defined; the shadow keeps this script's own arguments away
# from that guard.
assign("commandArgs", function(...) character(0), envir = globalenv())
suppressWarnings(try(source(clinical), silent = TRUE))
rm("commandArgs", envir = globalenv())
stopifnot(exists("bout_prepare", mode = "function"))

req <- jsonlite::read_json(request_path, simplifyVector = FALSE)
nums <- function(v) vapply(v, function(e) if (is.null(e)) NA_real_ else as.numeric(e), numeric(1))
cells <- do.call(rbind, lapply(req$cells, as.data.frame))
spacing_sec <- req$peak_spacing_sec
prominences <- nums(req$prominences)

peaks_all <- function(v, fs) {
  vapply(prominences, function(p) count_speed_peaks(v, p, spacing_sec * fs), numeric(1))
}

out_series <- list()
for (s_req in req$series) {
  sid <- s_req$sid
  fs <- s_req$rate
  grid <- bout_prepare(nums(s_req$t), nums(s_req$x), nums(s_req$y),
                       as.integer(nums(s_req$frame)), fs)
  truth <- lapply(s_req$clean, function(v) {
    v <- nums(v)
    list(sparc = spectral_arc_length(v, fs), peaks = peaks_all(v, fs))
  })
  per_cell <- vector("list", nrow(cells))
  for (k in seq_len(nrow(cells))) {
    if (is.null(grid)) {
      per_cell[[k]] <- list()
      next
    }
    b <- bout_runs(grid, fs, threshold = cells$threshold[k],
                   normaliser_q = cells$normaliser_q[k], min_sec = cells$min_sec[k],
                   merge_sec = cells$merge_sec[k], noise_k = cells$noise_k[k],
                   bounds = cells$bounds[k])
    per_cell[[k]] <- lapply(seq_len(nrow(b)), function(j) {
      s <- b$start[j]; e <- b$end[j]
      row <- list(start_sec = grid$t[s], end_sec = grid$t[e], censored = b$censored[j])
      if (!b$censored[j]) {
        seg <- grid$segments[[b$segment[j]]]
        for (support in c("bounds", "extended")) {
          row[[paste0("sparc_", support)]] <- spectral_arc_length(
            grid$speed[bout_sparc_support(grid, seg[1], seg[2], s, e, support)], fs
          )
        }
        row$peaks <- peaks_all(grid$det[s:(e - 1L)], fs)
      }
      row
    })
  }
  out_series[[sid]] <- list(truth = truth, cells = per_cell, gridded = !is.null(grid))
}
jsonlite::write_json(out_series, response_path, auto_unbox = TRUE, digits = 17, na = "null")

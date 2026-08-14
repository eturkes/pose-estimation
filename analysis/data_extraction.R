#!/usr/bin/env Rscript
# Per-finger joint-angle extraction and mobility (angular-speed) analysis
# from a single capture's hand-keypoint CSV (the pose pipeline's per-frame
# export: timestamp_sec + {left,right}_hand_{0..20}_{x,y,z,conf}).
#
# For each finger (thumb, index, middle, ring, pinky) on both hands, the
# per-frame flexion is the sum of the two inter-segment joint angles over
# the finger's four MediaPipe landmarks. Mobility is the frame-to-frame
# change in that flexion and its rate (degrees/second).
#
# Usage:
#   Rscript analysis/data_extraction.R <keypoints_csv> [out_dir]
#
# Outputs (out_dir defaults to the input CSV's directory):
#   <stem>_angle_data.csv        - per-frame flexion angle per finger/hand
#   <stem>_mobility_analysis.csv - per-frame angle delta + angular speed

library(dplyr)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript analysis/data_extraction.R <keypoints_csv> [out_dir]")
}
in_csv <- args[1]
if (!file.exists(in_csv)) stop("The input CSV does not exist: ", in_csv)
out_dir <- if (length(args) >= 2) args[2] else dirname(in_csv)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
stem <- tools::file_path_sans_ext(basename(in_csv))

df <- read.csv(in_csv)

# A finite coordinate is not necessarily a current observation: temporal
# smoothers may carry the last coordinate across a detector dropout.  Current
# exports distinguish those samples with per-keypoint confidence.  Preserve
# legacy CSV compatibility by applying the gate only when a confidence column
# exists.
mask_unobserved_hand_points <- function(data) {
  for (side in c("left", "right")) {
    for (k in 0:20) {
      prefix <- paste0(side, "_hand_", k)
      confidence_col <- paste0(prefix, "_conf")
      if (!(confidence_col %in% names(data))) next

      confidence <- suppressWarnings(as.numeric(data[[confidence_col]]))
      unobserved <- !is.finite(confidence) | confidence <= 0
      for (coord in c("x", "y", "z")) {
        coord_col <- paste0(prefix, "_", coord)
        if (coord_col %in% names(data)) {
          values <- suppressWarnings(as.numeric(data[[coord_col]]))
          values[unobserved] <- NA_real_
          data[[coord_col]] <- values
        }
      }
    }
  }
  data
}

df <- mask_unobserved_hand_points(df)

# MediaPipe hand-landmark index of each finger's first (MCP) joint; the
# finger spans landmarks id..id+3.
fingers <- c(thumb = 1, index = 5, middle = 9, ring = 13, pinky = 17)

# Angle (degrees) at p2 between segments p2->p1 and p2->p3.
angle <- function(p1, p2, p3) {
  points <- c(p1, p2, p3)
  if (length(points) != 6 || any(!is.finite(points))) return(NA_real_)
  vect1 <- p2 - p1
  vect2 <- p2 - p3
  denom <- sqrt(sum(vect1^2)) * sqrt(sum(vect2^2))
  if (!is.finite(denom) || denom <= 1e-12) return(NA_real_)
  cos_theta <- sum(vect1 * vect2) / denom
  if (!is.finite(cos_theta)) return(NA_real_)
  cos_theta <- max(min(cos_theta, 1), -1)
  acos(cos_theta) * 180 / pi
}

# Total per-frame flexion of one finger: (180 - angle1) + (180 - angle2)
# across the finger's four landmarks (id..id+3).
finger_flexion <- function(df, hand, id) {
  sapply(seq_len(nrow(df)), function(i) {
    pt <- function(k) c(df[[paste0(hand, "_", k, "_x")]][i],
                        df[[paste0(hand, "_", k, "_y")]][i])
    a <- angle(pt(id), pt(id + 1), pt(id + 2))
    b <- angle(pt(id + 1), pt(id + 2), pt(id + 3))
    (180 - a) + (180 - b)
  })
}

cols <- list()
for (n in names(fingers)) {
  start <- fingers[[n]]
  cols[[paste0(n, "_left")]] <- round(finger_flexion(df, "left_hand", start), 2)
  cols[[paste0(n, "_right")]] <- round(finger_flexion(df, "right_hand", start), 2)
}

angle_data <- as.data.frame(cols)
angle_data$time <- df$timestamp_sec

# Frame-to-frame flexion change and angular speed (degrees/second).
mobility <- data.frame(delta_t = df$timestamp_sec - lag(df$timestamp_sec))
valid_delta_t <- is.finite(mobility$delta_t) & mobility$delta_t > 0
for (n in names(angle_data)) {
  if (n != "time") {
    mobility[[paste0("delta_", n)]] <- angle_data[[n]] - lag(angle_data[[n]])
    speed <- rep(NA_real_, nrow(mobility))
    speed[valid_delta_t] <-
      mobility[[paste0("delta_", n)]][valid_delta_t] /
      mobility$delta_t[valid_delta_t]
    mobility[[paste0("speed_variation_", n)]] <- round(speed, 3)
  }
}

write_csv(angle_data, file.path(out_dir, paste0(stem, "_angle_data.csv")))
write_csv(mobility, file.path(out_dir, paste0(stem, "_mobility_analysis.csv")))
cat(sprintf("The script wrote %s_angle_data.csv and %s_mobility_analysis.csv to %s.\n",
            stem, stem, out_dir))

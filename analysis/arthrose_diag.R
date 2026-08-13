#!/usr/bin/env Rscript
# Index-finger mobility screen for osteoarthritis ("arthrose") from a single
# capture's hand-keypoint CSV (the pose pipeline's per-frame export). The
# index flexion angle per frame is summed over its two inter-segment joint
# angles (landmarks 5-8), smoothed, and reduced to range of motion and mean
# angular speed; both are checked against minimum-mobility thresholds.
#
# Usage:
#   Rscript analysis/arthrose_diag.R <keypoints_csv> [out_dir]
#
# Output (out_dir defaults to the input CSV's directory):
#   <stem>_closed_hand.png - thumb-to-index distance over time
# Diagnosis and metrics are printed to stdout.

library(dplyr)
library(tidyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript analysis/arthrose_diag.R <keypoints_csv> [out_dir]")
}
in_csv <- args[1]
if (!file.exists(in_csv)) stop("Input CSV not found: ", in_csv)
out_dir <- if (length(args) >= 2) args[2] else dirname(in_csv)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
stem <- tools::file_path_sans_ext(basename(in_csv))

df <- read.csv(in_csv)

# Current exports include observation confidence for every hand keypoint.
# Zero/blank/nonfinite confidence means the coordinate was missing or merely
# carried by temporal smoothing, so it must not contribute clinical evidence.
# Legacy CSVs without confidence columns retain coordinate-presence behavior.
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

# Minimum mobility for a "good mobility" verdict.
amplitude_min <- 40 # degrees of index range of motion
speed_min <- 60     # mean degrees/second

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

df$delta_time <- df$timestamp_sec - lag(df$timestamp_sec)
df_angles <- df %>%
  rowwise() %>%
  mutate(
    A = list(c(left_hand_5_x, left_hand_5_y)),
    B = list(c(left_hand_6_x, left_hand_6_y)),
    C = list(c(left_hand_7_x, left_hand_7_y)),
    D = list(c(left_hand_8_x, left_hand_8_y)),
    angle_567 = angle(unlist(A), unlist(B), unlist(C)),
    angle_678 = angle(unlist(B), unlist(C), unlist(D)),
    angle_index = (180 - angle_567) + (180 - angle_678)
  ) %>%
  ungroup() %>%
  # Centred 5-frame moving average; base stats::filter matches the former
  # zoo::rollmean(fill = NA, align = "center") including NA propagation.
  mutate(
    angle_smooth = as.numeric(stats::filter(angle_index, rep(1 / 5, 5), sides = 2))
  ) %>%
  mutate(
    delta_angle = angle_smooth - lag(angle_smooth),
    variation_speed_angle = if_else(
      is.finite(delta_time) & delta_time > 0,
      delta_angle / delta_time,
      NA_real_
    )
  )

clean <- df_angles %>%
  filter(is.finite(angle_index), is.finite(variation_speed_angle))
if (nrow(clean) < 2) {
  movement_amplitude <- NA_real_
  mean_speed <- NA_real_
  diagnostic <- "insufficient valid observations"
} else {
  movement_amplitude <- max(clean$angle_index) - min(clean$angle_index)
  mean_speed <- mean(abs(clean$variation_speed_angle))
  diagnostic <- if (movement_amplitude < amplitude_min || mean_speed < speed_min) {
    "problem with finger mobility"
  } else {
    "good mobility"
  }
}

cat(sprintf("Index range of motion : %.2f degrees\n", movement_amplitude))
cat(sprintf("Mean angular speed    : %.2f degrees/s\n", mean_speed))
cat(sprintf("Diagnosis             : %s\n", diagnostic))

# Thumb-to-index distance over time (a closing-hand trajectory).
df_dist <- df %>%
  mutate(
    dist_thumb_index = sqrt(
      (left_hand_8_x - left_hand_4_x)^2 +
        (left_hand_8_y - left_hand_4_y)^2 +
        (left_hand_8_z - left_hand_4_z)^2
    )
  )

p <- ggplot(df_dist, aes(x = timestamp_sec, y = dist_thumb_index)) +
  geom_line() +
  theme_classic() +
  labs(
    title = "Evolution of closed hand",
    x = "Time (s)",
    y = "Distance index - thumb"
  )

out_png <- file.path(out_dir, paste0(stem, "_closed_hand.png"))
ggsave(out_png, plot = p, width = 8, height = 5, dpi = 300)
cat(sprintf("Wrote %s\n", out_png))

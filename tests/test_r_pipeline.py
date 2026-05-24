"""R pipeline compatibility: verify rtmlib-mapped CSVs are consumable by clinical_features.R.

Generates synthetic landmark CSVs using the Python mapping + export path
(no model inference needed), then runs the R clinical pipeline on them.
Tests are skipped when R or required R packages are unavailable.
"""

import csv
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from pose_estimation.export import frame_to_rows, make_csv_header, open_csv_writer
from pose_estimation.mapping import coco_to_mediapipe
from pose_estimation.processing import TRACKING_BODY, TRACKING_HANDS_ARMS

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"

_FPS = 30.0
_N_FRAMES = 90  # 3 seconds — enough for multiple SAL windows
_FRAME_W, _FRAME_H = 640, 480


def _r_available():
    """Check whether Rscript and required packages are accessible."""
    if not shutil.which("Rscript"):
        return False
    try:
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                'for (p in c("dplyr","tidyr","readr","stringr","purrr")) '
                "if (!requireNamespace(p, quietly=TRUE)) quit(status=1)",
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


_HAS_R = _r_available()
requires_r = pytest.mark.skipif(not _HAS_R, reason="R or required R packages unavailable")


def _smooth_trajectory(n_frames, seed=42):
    """Generate smooth, anatomically plausible pixel-space keypoint trajectories.

    Uses sinusoidal movement to avoid constant-position degeneracy that
    would produce zero-distance features and NaN cascades in the R pipeline.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_frames)

    base_x = rng.uniform(150, 490, 133)
    base_y = rng.uniform(100, 380, 133)
    amp_x = rng.uniform(5, 30, 133)
    amp_y = rng.uniform(5, 30, 133)
    phase = rng.uniform(0, 2 * np.pi, 133)

    kps_seq = np.zeros((n_frames, 1, 133, 2), dtype=np.float64)
    for f in range(n_frames):
        kps_seq[f, 0, :, 0] = base_x + amp_x * np.sin(t[f] + phase)
        kps_seq[f, 0, :, 1] = base_y + amp_y * np.cos(t[f] + phase)

    scores_seq = np.full((n_frames, 1, 133), 0.85, dtype=np.float64)
    return kps_seq, scores_seq


def _generate_csv(output_path, tracking, n_kps=133, n_frames=_N_FRAMES):
    """Generate a synthetic landmark CSV using the rtmlib mapping path."""
    kps_seq, scores_seq = _smooth_trajectory(n_frames)

    if n_kps == 17:
        kps_seq = kps_seq[:, :, :17, :]
        scores_seq = scores_seq[:, :, :17]

    fh, writer = open_csv_writer(output_path, tracking)
    try:
        for f in range(n_frames):
            body_lm, body_vis, hand_lm, matches = coco_to_mediapipe(
                kps_seq[f], scores_seq[f], n_kps, tracking
            )
            rows = frame_to_rows(
                video_name="synthetic_test.avi",
                frame_idx=f,
                timestamp_sec=f / _FPS,
                frame_h=_FRAME_H,
                frame_w=_FRAME_W,
                body_landmarks=body_lm,
                body_visibilities=body_vis,
                hand_landmarks=hand_lm,
                matches=matches,
                tracking=tracking,
            )
            for row in rows:
                writer.writerow(row)
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# Schema tests (pure Python, no R needed)
# ---------------------------------------------------------------------------


class TestCSVSchemaHandsArms:
    """Verify rtmlib-mapped CSV header matches expected schema for hands-arms mode."""

    def test_header_matches(self, tmp_path):
        csv_path = tmp_path / "test_hands_arms.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS)
        expected = make_csv_header(TRACKING_HANDS_ARMS)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == expected

    def test_row_count(self, tmp_path):
        csv_path = tmp_path / "test_hands_arms.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS)
        with csv_path.open() as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1 + _N_FRAMES  # header + data

    def test_arm_columns_present(self, tmp_path):
        csv_path = tmp_path / "test_hands_arms.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
        for side in ("left", "right"):
            for kp in ("shoulder", "elbow", "wrist", "middle_base"):
                for coord in ("x", "y", "z"):
                    col = f"arm_{side}_{kp}_{coord}"
                    assert col in row, f"Missing column: {col}"
                    assert row[col] != "", f"Empty value in {col}"

    def test_hand_columns_present(self, tmp_path):
        csv_path = tmp_path / "test_hands_arms.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
        for side in ("left", "right"):
            for idx in (0, 4, 8, 20):  # wrist, thumb tip, index tip, pinky tip
                for coord in ("x", "y", "z"):
                    col = f"{side}_hand_{idx}_{coord}"
                    assert col in row, f"Missing column: {col}"
                    assert row[col] != "", f"Empty value in {col}"

    def test_coordinates_normalized(self, tmp_path):
        csv_path = tmp_path / "test_hands_arms.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
        x = float(row["arm_left_shoulder_x"])
        assert 0.0 <= x <= 1.0


class TestCSVSchemaBody:
    def test_header_matches(self, tmp_path):
        csv_path = tmp_path / "test_body.csv"
        _generate_csv(csv_path, TRACKING_BODY)
        expected = make_csv_header(TRACKING_BODY)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == expected

    def test_body_columns_for_clinical(self, tmp_path):
        """clinical_features.R in body mode uses 'index' for wrist deviation."""
        csv_path = tmp_path / "test_body.csv"
        _generate_csv(csv_path, TRACKING_BODY)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
        for side in ("left", "right"):
            for kp in ("shoulder", "elbow", "wrist", "index"):
                for coord in ("x", "y", "z"):
                    col = f"body_{side}_{kp}_{coord}"
                    assert col in row, f"Missing column: {col}"


class TestCSVSchema17kp:
    def test_header_matches_body_mode(self, tmp_path):
        csv_path = tmp_path / "test_17kp.csv"
        _generate_csv(csv_path, TRACKING_BODY, n_kps=17)
        expected = make_csv_header(TRACKING_BODY)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == expected

    def test_17kp_hands_arms_mode(self, tmp_path):
        csv_path = tmp_path / "test_17kp_ha.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS, n_kps=17)
        expected = make_csv_header(TRACKING_HANDS_ARMS)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == expected


# ---------------------------------------------------------------------------
# R pipeline integration tests
# ---------------------------------------------------------------------------


@requires_r
class TestClinicalFeaturesR:
    """Run clinical_features.R on synthetic CSVs and verify output."""

    def test_hands_arms_clinical_output(self, tmp_path):
        csv_path = tmp_path / "synth_hands-arms.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS)

        result = subprocess.run(
            ["Rscript", str(_CLINICAL_R), str(csv_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"R script failed:\n{result.stderr}"

        clinical = tmp_path / "synth_hands-arms_clinical.csv"
        windows = tmp_path / "synth_hands-arms_clinical_windows.csv"
        assert clinical.exists(), "clinical_features.R did not produce _clinical.csv"
        assert windows.exists(), "clinical_features.R did not produce _clinical_windows.csv"

        with clinical.open() as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames)
            row = next(reader)

        expected_features = [
            "left_elbow_angle_deg",
            "right_elbow_angle_deg",
            "left_wrist_deviation_deg",
            "right_wrist_deviation_deg",
            "left_finger_spread_deg",
            "right_finger_spread_deg",
            "left_reach_raw",
            "right_reach_raw",
            "left_reach_norm",
            "right_reach_norm",
            "left_grasp_aperture_thumb_index",
            "right_grasp_aperture_thumb_index",
            "left_grasp_aperture_thumb_pinky",
            "right_grasp_aperture_thumb_pinky",
            "left_wrist_displacement",
            "right_wrist_displacement",
            "left_fingertip_displacement",
            "right_fingertip_displacement",
        ]
        # Bilateral comparison columns for each metric pair.
        bilateral_metrics = [
            "elbow_angle_deg",
            "wrist_deviation_deg",
            "finger_spread_deg",
            "reach_raw",
            "reach_norm",
            "grasp_aperture_thumb_index",
            "grasp_aperture_thumb_pinky",
            "wrist_displacement",
            "fingertip_displacement",
        ]
        bilateral_suffixes = ["_symmetry_ratio", "_dominance_index", "_abs_diff"]
        expected_bilateral = [f"{m}{s}" for m in bilateral_metrics for s in bilateral_suffixes]
        for feat in expected_features:
            assert feat in cols, f"Missing clinical feature column: {feat}"
        for feat in expected_bilateral:
            assert feat in cols, f"Missing bilateral feature column: {feat}"

        for feat in expected_features:
            if "displacement" in feat:
                continue  # first frame is NA
            val = row[feat]
            assert val != "", f"Feature {feat} is empty on frame 0"
            assert val != "NA", f"Feature {feat} is NA on frame 0"

        # Bilateral metrics: non-displacement ones should have values on frame 0.
        for feat in expected_bilateral:
            if "displacement" in feat:
                continue  # first frame displacement is NA → bilateral is NA
            val = row[feat]
            assert val != "", f"Bilateral feature {feat} is empty on frame 0"
            assert val != "NA", f"Bilateral feature {feat} is NA on frame 0"

    def test_body_mode_clinical_output(self, tmp_path):
        csv_path = tmp_path / "synth_body.csv"
        _generate_csv(csv_path, TRACKING_BODY)

        result = subprocess.run(
            ["Rscript", str(_CLINICAL_R), str(csv_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"R script failed:\n{result.stderr}"

        clinical = tmp_path / "synth_body_clinical.csv"
        assert clinical.exists(), "clinical_features.R did not produce _clinical.csv"

        with clinical.open() as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames)
            row = next(reader)
        assert "left_elbow_angle_deg" in cols
        assert "left_reach_norm" in cols

        trunk_cols = [
            "trunk_lean_deg",
            "trunk_lean_lateral_deg",
            "trunk_rotation_deg",
            "posture_symmetry",
        ]
        for col in trunk_cols:
            assert col in cols, f"Missing trunk column: {col}"
            val = row[col]
            assert val != "", f"Trunk metric {col} is empty in body mode"
            assert val != "NA", f"Trunk metric {col} is NA in body mode"

    def test_hands_arms_trunk_columns_are_na(self, tmp_path):
        """Hands-arms mode should produce trunk columns but with NA values."""
        csv_path = tmp_path / "synth_hands-arms.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS)

        result = subprocess.run(
            ["Rscript", str(_CLINICAL_R), str(csv_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"R script failed:\n{result.stderr}"

        clinical = tmp_path / "synth_hands-arms_clinical.csv"
        assert clinical.exists()

        with clinical.open() as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames)
            row = next(reader)

        trunk_cols = [
            "trunk_lean_deg",
            "trunk_lean_lateral_deg",
            "trunk_rotation_deg",
            "posture_symmetry",
        ]
        for col in trunk_cols:
            assert col in cols, f"Missing trunk column in hands-arms mode: {col}"
            assert row[col] == "NA", f"Trunk {col} should be NA in hands-arms mode"

    def test_windows_have_sal_features(self, tmp_path):
        csv_path = tmp_path / "synth_hands-arms.csv"
        _generate_csv(csv_path, TRACKING_HANDS_ARMS)

        subprocess.run(
            ["Rscript", str(_CLINICAL_R), str(csv_path)],
            capture_output=True,
            timeout=120,
        )

        windows = tmp_path / "synth_hands-arms_clinical_windows.csv"
        assert windows.exists()

        with windows.open() as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames)
            rows = list(reader)

        expected_window_cols = [
            "left_wrist_sal",
            "right_wrist_sal",
            "left_wrist_velocity_mean",
            "right_wrist_velocity_mean",
            "left_wrist_velocity_peak",
            "right_wrist_velocity_peak",
            "left_wrist_normalized_jerk",
            "right_wrist_normalized_jerk",
            "left_wrist_movement_efficiency",
            "right_wrist_movement_efficiency",
            "left_fingertip_normalized_jerk",
            "right_fingertip_normalized_jerk",
            "compensatory_pattern_index",
        ]
        # Bilateral window metrics.
        window_bilateral_metrics = [
            "wrist_sal",
            "wrist_velocity_mean",
            "wrist_velocity_peak",
            "wrist_normalized_jerk",
            "wrist_movement_efficiency",
            "fingertip_normalized_jerk",
        ]
        window_bilateral_cols = [
            f"{m}{s}"
            for m in window_bilateral_metrics
            for s in ["_symmetry_ratio", "_dominance_index", "_abs_diff"]
        ]
        for col in expected_window_cols:
            assert col in cols, f"Missing window feature column: {col}"
        for col in window_bilateral_cols:
            assert col in cols, f"Missing bilateral window column: {col}"

        assert len(rows) > 0, "Window CSV has no data rows"

        # Bilateral window values should be non-empty in the first window.
        first_win = rows[0]
        for col in window_bilateral_cols:
            val = first_win[col]
            assert val != "", f"Bilateral window {col} is empty"
            assert val != "NA", f"Bilateral window {col} is NA"

    def test_body_mode_window_quality_metrics(self, tmp_path):
        """Body mode should produce compensatory_pattern_index, quality, and trunk metrics."""
        csv_path = tmp_path / "synth_body.csv"
        _generate_csv(csv_path, TRACKING_BODY)

        result = subprocess.run(
            ["Rscript", str(_CLINICAL_R), str(csv_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"R script failed:\n{result.stderr}"

        windows = tmp_path / "synth_body_clinical_windows.csv"
        assert windows.exists(), "Body mode did not produce _clinical_windows.csv"

        with windows.open() as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames)
            rows = list(reader)

        assert len(rows) > 0, "Body mode window CSV has no data rows"

        quality_cols = [
            "left_wrist_normalized_jerk",
            "right_wrist_normalized_jerk",
            "left_wrist_movement_efficiency",
            "right_wrist_movement_efficiency",
            "left_fingertip_normalized_jerk",
            "right_fingertip_normalized_jerk",
            "compensatory_pattern_index",
        ]
        for col in quality_cols:
            assert col in cols, f"Missing quality metric column: {col}"

        first = rows[0]
        for col in quality_cols:
            val = first[col]
            assert val != "", f"Quality metric {col} is empty in body mode"

        trunk_window_cols = [
            "trunk_lean_mean",
            "trunk_lean_sd",
            "trunk_lean_range",
            "trunk_lean_lateral_mean",
            "trunk_lean_lateral_sd",
            "trunk_rotation_mean",
            "trunk_rotation_sd",
            "posture_symmetry_mean",
            "posture_symmetry_sd",
        ]
        for col in trunk_window_cols:
            assert col in cols, f"Missing trunk window column: {col}"
            val = first[col]
            assert val != "", f"Trunk window metric {col} is empty in body mode"
            assert val != "NA", f"Trunk window metric {col} is NA in body mode"

    def test_directory_mode(self, tmp_path):
        """clinical_features.R can accept a directory of CSVs."""
        csv1 = tmp_path / "video1_hands-arms.csv"
        csv2 = tmp_path / "video2_hands-arms.csv"
        _generate_csv(csv1, TRACKING_HANDS_ARMS)
        _generate_csv(csv2, TRACKING_HANDS_ARMS)

        result = subprocess.run(
            ["Rscript", str(_CLINICAL_R), str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"R script failed:\n{result.stderr}"

        assert (tmp_path / "video1_hands-arms_clinical.csv").exists()
        assert (tmp_path / "video2_hands-arms_clinical.csv").exists()

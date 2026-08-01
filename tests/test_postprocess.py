"""Confidence-aware Savitzky-Golay post-processing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter

from pose_estimation.postprocess import savgol_smooth_csv


def _write_coordinate_series(path, prefix, values, evidence_column=None, invalid_evidence="0"):
    n_rows = len(values)
    data = {
        "video": ["synthetic.mp4"] * n_rows,
        "frame_idx": np.arange(n_rows),
        "timestamp_sec": np.arange(n_rows) / 30.0,
        "person_idx": np.zeros(n_rows, dtype=int),
        f"{prefix}_x": values,
    }
    if evidence_column is not None:
        evidence = ["1"] * n_rows
        evidence[n_rows // 2] = invalid_evidence
        data[evidence_column] = evidence
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("prefix", "evidence_column"),
    [
        ("body_nose", "body_nose_vis"),
        ("left_hand_0", "left_hand_0_conf"),
    ],
)
@pytest.mark.parametrize("invalid_evidence", ["0", "", "nan", "inf", "-inf"])
def test_unobserved_coordinate_cannot_influence_smoothing(
    tmp_path, prefix, evidence_column, invalid_evidence
):
    centre = 4
    positive_values = np.arange(9, dtype=float) ** 2
    negative_values = positive_values.copy()
    positive_values[centre] = 1_000_000.0
    negative_values[centre] = -1_000_000.0

    positive_input = tmp_path / "positive.csv"
    negative_input = tmp_path / "negative.csv"
    positive_output = tmp_path / "positive_smooth.csv"
    negative_output = tmp_path / "negative_smooth.csv"
    _write_coordinate_series(
        positive_input, prefix, positive_values, evidence_column, invalid_evidence
    )
    _write_coordinate_series(
        negative_input, prefix, negative_values, evidence_column, invalid_evidence
    )

    savgol_smooth_csv(positive_input, positive_output, window=5, polyorder=2)
    savgol_smooth_csv(negative_input, negative_output, window=5, polyorder=2)

    positive = pd.read_csv(positive_output)
    negative = pd.read_csv(negative_output)
    coordinate_column = f"{prefix}_x"
    np.testing.assert_allclose(positive[coordinate_column], negative[coordinate_column], atol=1e-6)
    assert np.abs(positive[coordinate_column]).max() < 100.0

    # Observation evidence is copied, not filtered or rewritten as a coordinate.
    expected_evidence = pd.read_csv(positive_input)[evidence_column].to_numpy()
    np.testing.assert_equal(positive[evidence_column].to_numpy(), expected_evidence)


def test_legacy_coordinate_without_observation_column_is_unchanged(tmp_path):
    values = np.arange(9, dtype=float) ** 2
    values[4] = 1_000.0
    input_path = tmp_path / "legacy.csv"
    output_path = tmp_path / "legacy_smooth.csv"
    _write_coordinate_series(input_path, "left_hand_0", values)

    savgol_smooth_csv(input_path, output_path, window=5, polyorder=2)

    actual = pd.read_csv(output_path)["left_hand_0_x"].to_numpy()
    expected = savgol_filter(values, 5, 2)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf])
def test_nonfinite_coordinate_is_missing_even_without_evidence(tmp_path, nonfinite):
    values = np.arange(9, dtype=float) ** 2
    values[4] = nonfinite
    input_path = tmp_path / "nonfinite.csv"
    output_path = tmp_path / "nonfinite_smooth.csv"
    _write_coordinate_series(input_path, "left_hand_0", values)

    savgol_smooth_csv(input_path, output_path, window=5, polyorder=2)

    actual = pd.read_csv(output_path)["left_hand_0_x"].to_numpy()
    assert np.isfinite(actual).all()
    assert np.abs(actual).max() < 100.0

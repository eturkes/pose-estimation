import pytest

from pose_estimation.measure import rigidity


def test_r2_gate_is_independent_of_ransac_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    assert rigidity.AXIS == "rigidity"
    assert rigidity.RANSAC_THRESHOLD_PX == 8.0
    assert rigidity.DRIFT_P95_GATE_PX == 20.0
    assert rigidity.RANSAC_THRESHOLD_PX != rigidity.DRIFT_P95_GATE_PX
    monkeypatch.setattr(rigidity, "RANSAC_THRESHOLD_PX", 32.0)
    common = {
        "orientation_status": "constant",
        "drift_median_px": 1.0,
        "valid_fraction": 1.0,
        "grid_cells_median": 8.0,
    }
    assert rigidity.rigidity_flag(drift_p95_px=20.0, **common) == "rigid"
    assert rigidity.rigidity_flag(drift_p95_px=20.000001, **common) == "camera_motion"


def test_rigidity_provenance_records_every_numeric_control() -> None:
    provenance = rigidity.PROVENANCE
    assert provenance["sampling"] == {
        "interval_s": rigidity.SAMPLE_INTERVAL_S,
        "time_source": "PTS * time_base",
        "first_sample_s": 0.0,
        "reference": "first sampled frame",
        "analysis_max_dimension_px": rigidity.ANALYSIS_MAX_DIM,
        "display_rotation": "inventory reported_rotation_deg",
        "orientation_eligibility": "constant timed video-orientation track",
    }
    assert provenance["features"]["nfeatures"] == rigidity.SIFT_FEATURES
    assert provenance["features"]["contrast_threshold"] == rigidity.SIFT_CONTRAST_THRESHOLD
    assert provenance["features"]["ratio"] == rigidity.MATCH_RATIO
    assert provenance["features"]["minimum_tracks"] == rigidity.MIN_TRACKS
    assert provenance["model"]["threshold_native_px"] == rigidity.RANSAC_THRESHOLD_PX
    assert provenance["model"]["maximum_iterations"] == rigidity.RANSAC_MAX_ITERATIONS
    assert provenance["model"]["confidence"] == rigidity.RANSAC_CONFIDENCE
    assert provenance["model"]["rng_seed"] == rigidity.SEED
    assert provenance["support"]["grid"] == (
        f"{rigidity.SUPPORT_GRID_SIZE}x{rigidity.SUPPORT_GRID_SIZE}"
    )
    assert provenance["support"]["minimum_median_cells"] == rigidity.MIN_GRID_CELLS
    assert provenance["support"]["minimum_valid_fraction"] == rigidity.MIN_VALID_FRACTION
    assert (
        provenance["statistic"]["grid"] == f"{rigidity.DRIFT_GRID_SIZE}x{rigidity.DRIFT_GRID_SIZE}"
    )
    assert provenance["statistic"]["grid_margin_fraction"] == rigidity.DRIFT_GRID_MARGIN_FRACTION
    assert provenance["gate"]["threshold_px"] == rigidity.DRIFT_P95_GATE_PX


def _result(asset_id: str, capture_id: str, flag: str) -> rigidity.RigidityResult:
    return rigidity.RigidityResult(
        asset_id=asset_id,
        capture_id=capture_id,
        orientation_status="constant",
        sampled_frames=3,
        valid_samples=2,
        valid_fraction=1.0,
        inliers_median=120.0,
        grid_cells_median=8.0,
        drift_median_px=1.0,
        drift_p95_px=2.0,
        rigidity_flag=flag,
    )


def test_all_members_rigid_never_skips_a_no_verdict_member() -> None:
    """A no-verdict member must disqualify its family, not drop out of the test.

    ``all_members_rigid`` over ``multi_asset_families`` is the pair M2.6 reads
    as its usable-family count, so a family carrying an unmeasurable member
    must not read as fully rigid, and a lone asset must reach neither counter.
    """
    summary = rigidity.summarize(
        [
            _result("a-0000000000000000", "c-0000000000000000", "rigid"),
            _result("a-0000000000000001", "c-0000000000000000", "rigid"),
            _result("a-0000000000000002", "c-0000000000000001", "rigid"),
            _result("a-0000000000000003", "c-0000000000000001", "unmeasurable"),
            _result("a-0000000000000004", "c-0000000000000002", "rigid"),
        ]
    )

    assert summary["multi_asset_families"] == 2
    assert summary["all_members_rigid"] == 1
    assert summary["assets"] == 5
    assert summary["rigid"] == 4
    assert summary["eligible"] == 4
    assert summary["no_verdict"] == 1

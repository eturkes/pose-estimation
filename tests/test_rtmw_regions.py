"""Tests for per-region differentiated smoothing."""

import numpy as np

from pose_estimation.run import (
    REGION_PARAMS, KeypointSmoother, _OneEuro,
)


def _make_kps(n=133, seed=4827):
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2) * 300


def test_region_params_cover_all_133():
    """REGION_PARAMS must cover indices 0-132 without gaps or overlaps."""
    covered = set()
    for _, start, end, _, _ in REGION_PARAMS:
        region_set = set(range(start, end))
        assert not (covered & region_set), "Overlapping region indices"
        covered |= region_set
    assert covered == set(range(133))


def test_make_filters_133_creates_regions():
    """For 133 kps, _make_filters returns one filter per region."""
    sm = KeypointSmoother()
    filters = sm._make_filters(133)
    assert set(filters.keys()) == {name for name, *_ in REGION_PARAMS}


def test_make_filters_17_creates_single():
    """For non-133 kps (e.g. body-only 17), fall back to single filter."""
    sm = KeypointSmoother()
    filters = sm._make_filters(17)
    assert "all" in filters and len(filters) == 1


def test_region_smoothing_differs_from_uniform():
    """Region-filtered output should differ from a uniform single-filter
    smoother, since the regions use different parameters."""
    base = _make_kps(133)
    sc = np.ones(133)

    # Build the same noisy sequence for both smoothers
    frames = []
    rng = np.random.RandomState(6193)
    for step in range(30):
        frames.append(base + rng.randn(133, 2) * 5.0)

    # Region smoother (default)
    sm_region = KeypointSmoother()
    for step, f in enumerate(frames):
        r_region, _ = sm_region(f[np.newaxis], sc[np.newaxis], step * 0.033)

    # Uniform smoother: override to use a single filter for all keypoints
    sm_uniform = KeypointSmoother()
    sm_uniform._make_filters = lambda n: {
        "all": _OneEuro(min_cutoff=0.5, beta=0.5)
    }
    for step, f in enumerate(frames):
        r_uniform, _ = sm_uniform(f[np.newaxis], sc[np.newaxis], step * 0.033)

    diff = np.abs(r_region[0] - r_uniform[0]).max()
    assert diff > 0.01, (
        f"Region and uniform smoothing should produce different output, "
        f"max diff={diff:.6f}"
    )


def test_region_filters_independent():
    """Each region filter should be independent — changing confidence in one
    region shouldn't affect another."""
    sm = KeypointSmoother(min_track_age=1)

    kp = _make_kps(133)
    sc = np.ones(133)
    keypoints = kp[np.newaxis]

    # Run 1: all high confidence
    scores_all_high = sc[np.newaxis]
    sm(keypoints, scores_all_high, 0.0)
    shifted = keypoints + 40.0
    result_all_high, _ = sm(shifted, scores_all_high, 0.1)

    # Run 2: low confidence on hands only
    sm2 = KeypointSmoother(min_track_age=1)
    sc2 = np.ones(133)
    sc2[91:133] = 0.1
    scores_low_hands = sc2[np.newaxis]
    sm2(keypoints, scores_low_hands, 0.0)
    result_low_hands, _ = sm2(shifted, scores_low_hands, 0.1)

    # Body region should be identical between runs (same params, same conf)
    np.testing.assert_allclose(
        result_all_high[0, :17], result_low_hands[0, :17], atol=1e-10,
        err_msg="Body region affected by hand confidence change"
    )


def test_smoother_output_shape_133():
    """Output shape should match input regardless of region splitting."""
    sm = KeypointSmoother(min_track_age=1)
    kp = _make_kps(133)
    sc = np.ones(133)
    keypoints = np.stack([kp, kp + 500])
    scores = np.stack([sc, sc])

    out_kps, out_sc = sm(keypoints, scores, 0.0)
    assert out_kps.shape == keypoints.shape
    assert out_sc.shape == scores.shape


def test_smoother_output_shape_17():
    """Non-133 keypoints should work with the single-filter fallback."""
    sm = KeypointSmoother(min_track_age=1)
    kp = _make_kps(17)
    sc = np.ones(17)
    keypoints = kp[np.newaxis]
    scores = sc[np.newaxis]

    out_kps, out_sc = sm(keypoints, scores, 0.0)
    assert out_kps.shape == (1, 17, 2)
    assert out_sc.shape == (1, 17)


def test_carry_preserves_region_filters():
    """After carry-forward, re-matching should reuse region filters."""
    sm = KeypointSmoother()
    kp = _make_kps(133)
    sc = np.ones(133)
    keypoints = kp[np.newaxis]
    scores = sc[np.newaxis]

    sm(keypoints, scores, 0.0)
    sm(None, None, 0.1)  # carry

    assert len(sm.tracks) == 1
    filt = sm.tracks[0]["filter"]
    assert set(filt.keys()) == {name for name, *_ in REGION_PARAMS}


if __name__ == "__main__":
    test_region_params_cover_all_133()
    test_make_filters_133_creates_regions()
    test_make_filters_17_creates_single()
    test_hands_smoothed_less_than_body()
    test_region_filters_independent()
    test_smoother_output_shape_133()
    test_smoother_output_shape_17()
    test_carry_preserves_region_filters()
    print("All tests passed.")

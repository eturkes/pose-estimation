"""Tests for Hungarian matching in KeypointSmoother."""

import numpy as np

from pose_estimation.run import KeypointSmoother


def _make_kps(center, n=5):
    """Return (n, 2) keypoints clustered around *center*."""
    rng = np.random.RandomState(4817)
    return np.array(center) + rng.randn(n, 2) * 5


def _feed_frame(smoother, centers, t):
    """Build a detection array from centroid positions and feed it."""
    kps = np.stack([_make_kps(c) for c in centers])
    scores = np.ones((len(centers), kps.shape[1]))
    return smoother(kps, scores, t)


def test_basic_matching():
    """Two detections near two existing tracks are correctly paired."""
    sm = KeypointSmoother(match_thresh=200, min_track_age=1)
    _feed_frame(sm, [(100, 100), (300, 300)], t=0.0)
    assert len(sm.tracks) == 2

    # Second frame: same positions → should match (not create new tracks)
    _feed_frame(sm, [(102, 102), (298, 298)], t=0.033)
    assert len(sm.tracks) == 2
    # Both tracks should have misses == 0 (matched this frame)
    for tr in sm.tracks:
        assert tr["misses"] == 0


def test_optimal_over_greedy():
    """Hungarian avoids the suboptimal greedy assignment.

    Setup: two tracks at (0, 0) and (10, 0).
    Detections arrive at (9, 0) and (1, 0) — i.e. swapped proximity.
    Greedy nearest-first could assign (9,0)→track(10,0) first, then
    (1,0)→track(0,0), total cost = 1+1 = 2.  That happens to be
    optimal here too, but consider adversarial placement:

    Tracks at (0, 0) and (3, 0).
    Detections at (2, 0) and (4, 0).
    Greedy (smallest first): (2,0)→track(3,0) cost 1, then
      (4,0)→track(0,0) cost 4, total = 5.
    Optimal: (2,0)→track(0,0) cost 2, (4,0)→track(3,0) cost 1, total = 3.
    """
    sm = KeypointSmoother(match_thresh=200, min_track_age=1)

    # Seed tracks at (0, 0) and (30, 0) using single-keypoint arrays
    kps_init = np.array([[[0.0, 0.0]], [[30.0, 0.0]]])
    scores_init = np.ones((2, 1))
    sm(kps_init, scores_init, t=0.0)

    # Detections: (20, 0) and (40, 0)
    # Greedy (smallest-first): (20,0)→track(30,0) cost 10,
    #   (40,0)→track(0,0) cost 40, total = 50
    # Optimal: (20,0)→track(0,0) cost 20, (40,0)→track(30,0) cost 10,
    #   total = 30
    kps_new = np.array([[[20.0, 0.0]], [[40.0, 0.0]]])
    scores_new = np.ones((2, 1))
    sm(kps_new, scores_new, t=0.033)

    # The optimal assignment pairs detection 0 (20,0) with the
    # track originally at (0,0), and detection 1 (40,0) with the
    # track originally at (30,0).
    # After smoothing the centroids should reflect this pairing:
    # track 0 centroid moved toward 20, track 1 centroid moved toward 40.
    centroids = [tr["centroid"] for tr in sm.tracks]
    # Sort by x to identify which is which
    centroids.sort(key=lambda c: c[0])
    assert centroids[0][0] < 25, "lower track should have moved toward 20"
    assert centroids[1][0] > 30, "upper track should have moved toward 40"


def test_threshold_filtering():
    """Detections beyond match_thresh create new tracks, not false matches."""
    sm = KeypointSmoother(match_thresh=50, min_track_age=1)
    kps1 = np.array([[[100.0, 100.0]]])
    sm(kps1, np.ones((1, 1)), t=0.0)
    assert len(sm.tracks) == 1

    # Detection far away — should not match existing track
    kps2 = np.array([[[500.0, 500.0]]])
    sm(kps2, np.ones((1, 1)), t=0.033)
    # 1 new track for the detection + 1 carried forward = 2
    assert len(sm.tracks) == 2
    # One should have misses == 0 (new), one should have misses == 1 (carried)
    misses = sorted(tr["misses"] for tr in sm.tracks)
    assert misses == [0, 1]


def test_no_tracks_no_crash():
    """First frame with no prior tracks works fine."""
    sm = KeypointSmoother(match_thresh=200, min_track_age=1)
    kps = np.array([[[50.0, 50.0], [60.0, 60.0]]])
    scores = np.ones((1, 2))
    out_kps, out_scores = sm(kps, scores, t=0.0)
    assert out_kps is not None
    assert out_kps.shape[0] == 1


def test_empty_detections():
    """No detections triggers carry-forward, not a crash."""
    sm = KeypointSmoother(match_thresh=200, min_track_age=0)
    kps = np.array([[[50.0, 50.0]]])
    sm(kps, np.ones((1, 1)), t=0.0)

    out_kps, out_scores = sm(None, None, t=0.033)
    assert out_kps is not None
    assert len(sm.tracks) == 1
    assert sm.tracks[0]["misses"] == 1


if __name__ == "__main__":
    test_basic_matching()
    test_optimal_over_greedy()
    test_threshold_filtering()
    test_no_tracks_no_crash()
    test_empty_detections()
    print("All rtmw matching tests passed.")

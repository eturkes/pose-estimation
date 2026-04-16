"""Tests for track age gating in KeypointSmoother."""

import numpy as np

from pose_estimation.run import KeypointSmoother


def _make_kps(n=5, seed=6142):
    """Return (n, 2) keypoints."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2) * 300


def _feed(sm, centers, t, n_kps=5):
    """Build detections from centroid positions and feed them."""
    rng = np.random.RandomState(4817)
    kps = np.stack([np.array(c) + rng.randn(n_kps, 2) * 5 for c in centers])
    scores = np.ones((len(centers), n_kps))
    return sm(kps, scores, t)


def test_new_track_suppressed():
    """A brand-new track (age=1) is not emitted when min_track_age > 1."""
    sm = KeypointSmoother(min_track_age=3)
    out_kps, _out_scores = _feed(sm, [(100, 100)], t=0.0)
    assert out_kps is None, "new track should be suppressed"
    # Track should still exist internally
    assert len(sm.tracks) == 1
    assert sm.tracks[0]["age"] == 1


def test_track_emitted_after_reaching_age():
    """Track becomes visible once age reaches min_track_age."""
    sm = KeypointSmoother(min_track_age=3)

    _feed(sm, [(100, 100)], t=0.0)  # age=1, suppressed
    _feed(sm, [(102, 102)], t=0.033)  # age=2, suppressed
    out_kps, _ = _feed(sm, [(104, 104)], t=0.066)  # age=3, emitted

    assert out_kps is not None
    assert out_kps.shape[0] == 1


def test_age_increments_on_match():
    """Age increases by 1 each frame the track is matched."""
    sm = KeypointSmoother(min_track_age=1)

    _feed(sm, [(100, 100)], t=0.0)
    assert sm.tracks[0]["age"] == 1

    _feed(sm, [(102, 102)], t=0.033)
    assert sm.tracks[0]["age"] == 2

    _feed(sm, [(104, 104)], t=0.066)
    assert sm.tracks[0]["age"] == 3


def test_age_decrements_on_carry():
    """Age decreases by 1 per missed frame, clamped to 0."""
    sm = KeypointSmoother(min_track_age=1, carry_frames=10)

    # Build up age
    for i in range(5):
        _feed(sm, [(100, 100)], t=i * 0.033)
    assert sm.tracks[0]["age"] == 5

    # Miss 3 frames
    for i in range(3):
        sm(None, None, (5 + i) * 0.033)

    assert sm.tracks[0]["age"] == 2  # 5 - 3

    # Miss enough to clamp at 0
    for i in range(5):
        sm(None, None, (8 + i) * 0.033)

    assert sm.tracks[0]["age"] == 0


def test_carried_track_hidden_when_age_drops():
    """Carried track stops being emitted once age drops below threshold."""
    sm = KeypointSmoother(min_track_age=3, carry_frames=10)

    # Build age to exactly 3
    for i in range(3):
        _feed(sm, [(100, 100)], t=i * 0.033)
    assert sm.tracks[0]["age"] == 3

    # First carry: age=2 < 3, should not emit
    out_kps, _ = sm(None, None, 3 * 0.033)
    assert out_kps is None
    assert len(sm.tracks) == 1  # track still maintained internally


def test_false_positive_suppressed():
    """A detection that appears for 2 frames then vanishes is never shown.

    This is the core use-case: one-off false positives that flicker
    on screen for 1-2 frames are suppressed by the age gate.
    """
    sm = KeypointSmoother(min_track_age=3, match_thresh=50)

    # Flash detection for 2 frames
    out1, _ = _feed(sm, [(100, 100)], t=0.0)
    out2, _ = _feed(sm, [(102, 102)], t=0.033)
    assert out1 is None, "age=1, should be suppressed"
    assert out2 is None, "age=2, should be suppressed"

    # Detection disappears — track carries and age decays
    out3, _ = sm(None, None, 0.066)
    assert out3 is None, "carried track age=1, still suppressed"


def test_intermittent_detection_cannot_accumulate_age():
    """A detection that appears, vanishes, reappears can't game age.

    Because age decrements during misses, an intermittent false positive
    that appears every other frame cannot accumulate enough age to be
    emitted (with min_track_age=3).
    """
    sm = KeypointSmoother(min_track_age=3, match_thresh=200)

    # Alternating: detect, miss, detect, miss, detect, miss
    for cycle in range(3):
        t_det = cycle * 2 * 0.033
        t_miss = t_det + 0.033
        out, _ = _feed(sm, [(100, 100)], t=t_det)
        # After detect: age should be at most 2 (1 from creation, or
        # previous age - 1 + 1 from rematch)
        assert out is None, f"cycle {cycle}: should be suppressed"
        sm(None, None, t=t_miss)


def test_min_track_age_zero_disables_gating():
    """min_track_age=0 means all tracks are emitted immediately."""
    sm = KeypointSmoother(min_track_age=0)
    out_kps, _ = _feed(sm, [(100, 100)], t=0.0)
    assert out_kps is not None
    assert out_kps.shape[0] == 1


def test_min_track_age_default():
    """Default min_track_age is 3."""
    sm = KeypointSmoother()
    assert sm.min_track_age == 3


def test_multiple_tracks_age_independent():
    """Each track accumulates age independently."""
    sm = KeypointSmoother(min_track_age=2, match_thresh=200)

    # Frame 0: track A appears
    _feed(sm, [(100, 100)], t=0.0)
    assert len(sm.tracks) == 1

    # Frame 1: track A matched (age=2), track B appears (age=1)
    out, _ = _feed(sm, [(102, 102), (500, 500)], t=0.033)
    assert out is not None
    # Only track A (age=2) should be emitted
    assert out.shape[0] == 1

    # Frame 2: both matched. A age=3, B age=2 — both emitted
    out, _ = _feed(sm, [(104, 104), (502, 502)], t=0.066)
    assert out is not None
    assert out.shape[0] == 2

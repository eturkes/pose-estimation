"""Tests for COCO-WholeBody → MediaPipe keypoint mapping."""

import numpy as np
import pytest

from pose_estimation.export import frame_to_rows, make_csv_header
from pose_estimation.mapping import coco_to_mediapipe
from pose_estimation.processing import TRACKING_BODY, TRACKING_HANDS, TRACKING_HANDS_ARMS


def _synthetic_133(n_persons=1, seed=42):
    """Generate synthetic 133-kp data with plausible pixel coords."""
    rng = np.random.default_rng(seed)
    kps = rng.uniform(50, 600, (n_persons, 133, 2))
    scores = rng.uniform(0.3, 1.0, (n_persons, 133))
    return kps, scores


def _synthetic_17(n_persons=1, seed=42):
    """Generate synthetic 17-kp data."""
    rng = np.random.default_rng(seed)
    kps = rng.uniform(50, 600, (n_persons, 17, 2))
    scores = rng.uniform(0.3, 1.0, (n_persons, 17))
    return kps, scores


# ---------------------------------------------------------------------------
# Shape tests: 133-kp x each tracking mode
# ---------------------------------------------------------------------------


class TestMap133HandsArms:
    def test_body_shape(self):
        kps, scores = _synthetic_133(n_persons=2)
        body_lm, body_vis, _hand_lm, _matches = coco_to_mediapipe(
            kps, scores, 133, TRACKING_HANDS_ARMS
        )
        assert len(body_lm) == 2
        assert body_lm[0].shape == (12, 3)
        assert body_vis[0].shape == (12,)

    def test_hand_landmarks_present(self):
        kps, scores = _synthetic_133(n_persons=1)
        _, _, hand_lm, _matches = coco_to_mediapipe(kps, scores, 133, TRACKING_HANDS_ARMS)
        assert len(hand_lm) == 2  # left + right per person
        assert hand_lm[0].shape == (21, 3)
        assert hand_lm[1].shape == (21, 3)

    def test_matches_link_to_wrists(self):
        kps, scores = _synthetic_133(n_persons=1)
        _, _, _, matches = coco_to_mediapipe(kps, scores, 133, TRACKING_HANDS_ARMS)
        wrist_indices = {m[1] for m in matches}
        assert wrist_indices == {4, 5}  # ARM_WRIST_LEFT, ARM_WRIST_RIGHT

    def test_z_is_zero(self):
        kps, scores = _synthetic_133()
        body_lm, _, hand_lm, _ = coco_to_mediapipe(kps, scores, 133, TRACKING_HANDS_ARMS)
        assert np.all(body_lm[0][:, 2] == 0.0)
        assert np.all(hand_lm[0][:, 2] == 0.0)

    def test_arm_coordinates_match_coco(self):
        kps, scores = _synthetic_133()
        body_lm, body_vis, _, _ = coco_to_mediapipe(kps, scores, 133, TRACKING_HANDS_ARMS)
        # left_shoulder = arm[0] should match COCO index 5
        np.testing.assert_array_equal(body_lm[0][0, :2], kps[0, 5])
        # right_wrist = arm[5] should match COCO index 10
        np.testing.assert_array_equal(body_lm[0][5, :2], kps[0, 10])
        # Visibility matches
        assert body_vis[0][0] == scores[0, 5]


class TestMap133Body:
    def test_body_shape(self):
        kps, scores = _synthetic_133(n_persons=3)
        body_lm, body_vis, _hand_lm, _matches = coco_to_mediapipe(kps, scores, 133, TRACKING_BODY)
        assert len(body_lm) == 3
        assert body_lm[0].shape == (33, 3)
        assert body_vis[0].shape == (33,)

    def test_direct_body_mappings(self):
        kps, scores = _synthetic_133()
        body_lm, _body_vis, _, _ = coco_to_mediapipe(kps, scores, 133, TRACKING_BODY)
        # nose: MP 0 ← COCO 0
        np.testing.assert_array_equal(body_lm[0][0, :2], kps[0, 0])
        # left_shoulder: MP 11 ← COCO 5
        np.testing.assert_array_equal(body_lm[0][11, :2], kps[0, 5])
        # left_ankle: MP 27 ← COCO 15
        np.testing.assert_array_equal(body_lm[0][27, :2], kps[0, 15])

    def test_face_derived_mappings(self):
        kps, scores = _synthetic_133()
        body_lm, _, _, _ = coco_to_mediapipe(kps, scores, 133, TRACKING_BODY)
        # left_eye_inner: MP 1 ← COCO face sub-idx 36 = COCO 23+36=59
        np.testing.assert_array_equal(body_lm[0][1, :2], kps[0, 59])
        # mouth_left: MP 9 ← COCO face sub-idx 48 = COCO 23+48=71
        np.testing.assert_array_equal(body_lm[0][9, :2], kps[0, 71])

    def test_hand_derived_wrist_keypoints(self):
        kps, scores = _synthetic_133()
        body_lm, _, _, _ = coco_to_mediapipe(kps, scores, 133, TRACKING_BODY)
        # left_pinky (MP 17) ← left hand pinky MCP = COCO 91+17=108
        np.testing.assert_array_equal(body_lm[0][17, :2], kps[0, 108])
        # right_index (MP 20) ← right hand index MCP = COCO 112+5=117
        np.testing.assert_array_equal(body_lm[0][20, :2], kps[0, 117])

    def test_matches_link_to_body_wrists(self):
        kps, scores = _synthetic_133()
        _, _, _, matches = coco_to_mediapipe(kps, scores, 133, TRACKING_BODY)
        wrist_indices = {m[1] for m in matches}
        assert wrist_indices == {15, 16}  # BODY_WRIST_LEFT, BODY_WRIST_RIGHT

    def test_feet_mapping(self):
        kps, scores = _synthetic_133()
        body_lm, _, _, _ = coco_to_mediapipe(kps, scores, 133, TRACKING_BODY)
        # left_heel: MP 29 ← COCO 19
        np.testing.assert_array_equal(body_lm[0][29, :2], kps[0, 19])
        # right_foot_index: MP 32 ← COCO 20
        np.testing.assert_array_equal(body_lm[0][32, :2], kps[0, 20])


class TestMap133Hands:
    def test_hands_only_no_body(self):
        kps, scores = _synthetic_133(n_persons=1)
        body_lm, body_vis, hand_lm, matches = coco_to_mediapipe(kps, scores, 133, TRACKING_HANDS)
        assert body_lm == []
        assert body_vis == []
        assert matches == []
        assert len(hand_lm) == 2  # both hands above threshold
        assert hand_lm[0].shape == (21, 3)


# ---------------------------------------------------------------------------
# Shape tests: 17-kp body-only
# ---------------------------------------------------------------------------


class TestMap17:
    def test_body_mode_shape(self):
        kps, scores = _synthetic_17(n_persons=2)
        body_lm, body_vis, hand_lm, matches = coco_to_mediapipe(kps, scores, 17, TRACKING_BODY)
        assert len(body_lm) == 2
        assert body_lm[0].shape == (33, 3)
        assert body_vis[0].shape == (33,)
        assert hand_lm == []
        assert matches == []

    def test_hands_arms_mode_shape(self):
        kps, scores = _synthetic_17(n_persons=1)
        body_lm, _body_vis, hand_lm, _matches = coco_to_mediapipe(
            kps, scores, 17, TRACKING_HANDS_ARMS
        )
        assert len(body_lm) == 1
        assert body_lm[0].shape == (12, 3)
        # Finger bases are zero (no hand data in 17-kp)
        assert body_lm[0][6, 0] == 0.0
        assert hand_lm == []

    def test_hands_mode_empty(self):
        kps, scores = _synthetic_17()
        body_lm, _body_vis, hand_lm, _matches = coco_to_mediapipe(kps, scores, 17, TRACKING_HANDS)
        assert body_lm == []
        assert hand_lm == []

    def test_body_direct_mappings(self):
        kps, scores = _synthetic_17()
        body_lm, body_vis, _, _ = coco_to_mediapipe(kps, scores, 17, TRACKING_BODY)
        # nose: MP 0 ← COCO 0
        np.testing.assert_array_equal(body_lm[0][0, :2], kps[0, 0])
        # left_hip: MP 23 ← COCO 11
        np.testing.assert_array_equal(body_lm[0][23, :2], kps[0, 11])
        # Unmapped keypoints remain zero
        assert body_lm[0][1, 0] == 0.0  # left_eye_inner (no face data)
        assert body_vis[0][1] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_input(self):
        result = coco_to_mediapipe(None, None, 133, TRACKING_HANDS_ARMS)
        assert result == ([], [], [], [])

    def test_empty_array(self):
        kps = np.zeros((0, 133, 2))
        scores = np.zeros((0, 133))
        result = coco_to_mediapipe(kps, scores, 133, TRACKING_HANDS_ARMS)
        assert result == ([], [], [], [])

    def test_unsupported_n_kps(self):
        kps = np.zeros((1, 50, 2))
        scores = np.zeros((1, 50))
        result = coco_to_mediapipe(kps, scores, 50, TRACKING_BODY)
        assert result == ([], [], [], [])

    def test_low_hand_score_suppresses_hand(self):
        kps, scores = _synthetic_133()
        # Zero out left hand scores → should suppress left hand
        scores[0, 91:112] = 0.0
        _, _, hand_lm, matches = coco_to_mediapipe(kps, scores, 133, TRACKING_HANDS_ARMS)
        assert len(hand_lm) == 1  # only right hand
        assert matches[0][1] == 5  # right wrist only


# ---------------------------------------------------------------------------
# Round-trip: mapping → frame_to_rows produces valid CSV rows
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize(
        ("n_kps", "tracking"),
        [
            (133, TRACKING_HANDS_ARMS),
            (133, TRACKING_BODY),
            (133, TRACKING_HANDS),
            (17, TRACKING_BODY),
            (17, TRACKING_HANDS_ARMS),
        ],
    )
    def test_csv_column_count(self, n_kps, tracking):
        """Mapped output produces rows with exactly the right column count."""
        if n_kps == 133:
            kps, scores = _synthetic_133(n_persons=1)
        else:
            kps, scores = _synthetic_17(n_persons=1)

        body_lm, body_vis, hand_lm, matches = coco_to_mediapipe(kps, scores, n_kps, tracking)

        header = make_csv_header(tracking)

        if tracking == TRACKING_HANDS and n_kps == 17:
            # 17-kp hands mode produces nothing
            return

        rows = frame_to_rows(
            video_name="test.mp4",
            frame_idx=0,
            timestamp_sec=0.0,
            frame_h=720,
            frame_w=1280,
            body_landmarks=body_lm,
            body_visibilities=body_vis,
            hand_landmarks=hand_lm,
            matches=matches,
            tracking=tracking,
        )

        if not rows:
            return

        for row in rows:
            assert set(row.keys()) == set(header), (
                f"Column mismatch for n_kps={n_kps}, tracking={tracking}"
            )

    def test_multi_person_row_count(self):
        """Each detected person produces one CSV row."""
        kps, scores = _synthetic_133(n_persons=3)
        body_lm, body_vis, hand_lm, matches = coco_to_mediapipe(
            kps, scores, 133, TRACKING_HANDS_ARMS
        )
        rows = frame_to_rows(
            video_name="test.mp4",
            frame_idx=0,
            timestamp_sec=0.0,
            frame_h=720,
            frame_w=1280,
            body_landmarks=body_lm,
            body_visibilities=body_vis,
            hand_landmarks=hand_lm,
            matches=matches,
            tracking=TRACKING_HANDS_ARMS,
        )
        assert len(rows) == 3

    def test_normalized_coordinates_in_range(self):
        """CSV coordinate values are normalized to [0, 1]."""
        kps, scores = _synthetic_133()
        body_lm, body_vis, hand_lm, matches = coco_to_mediapipe(
            kps, scores, 133, TRACKING_HANDS_ARMS
        )
        rows = frame_to_rows(
            video_name="test.mp4",
            frame_idx=0,
            timestamp_sec=0.0,
            frame_h=720,
            frame_w=1280,
            body_landmarks=body_lm,
            body_visibilities=body_vis,
            hand_landmarks=hand_lm,
            matches=matches,
            tracking=TRACKING_HANDS_ARMS,
        )
        row = rows[0]
        # Arm x values should be in [0, 1] given synth data in [50, 600]
        for i in range(12):
            name = f"arm_{['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_index_base', 'right_index_base', 'left_middle_base', 'right_middle_base', 'left_pinky_base', 'right_pinky_base'][i]}_x"
            val = row[name]
            assert 0.0 <= val <= 1.0, f"{name}={val} out of range"

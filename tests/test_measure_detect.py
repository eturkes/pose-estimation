import math

import numpy as np

from pose_estimation.measure import detect


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> list[np.ndarray]:
        self.calls += 1
        return [np.asarray([10.0, 20.0, 50.0, 80.0, 0.9])]


class FakePose:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, frame: np.ndarray, *, bboxes: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        assert len(bboxes) == 1
        return np.zeros((1, 133, 2)), np.full((1, 133), 0.8)


def test_detect_axis_drives_detector_and_pose_directly_per_frame() -> None:
    assert detect.AXIS == "detect"
    assert detect.DETECTOR_DEVICE == "GPU"
    assert detect.POSE_DEVICE == "NPU"
    samples = [detect.Sample(index, np.zeros((100, 200, 3))) for index in range(3)]
    detector = FakeDetector()
    pose = FakePose()

    completed, detected, failures, rate, confidence, height = detect._infer_samples(
        samples, detector, pose
    )

    assert (detector.calls, pose.calls) == (3, 3)
    assert (completed, detected, failures) == (3, 3, 0)
    assert rate == 1.0
    assert confidence == 0.8
    assert height == 60.0


def test_pose_failure_is_counted_and_excluded_from_detect_rate() -> None:
    samples = [detect.Sample(0, np.zeros((100, 200, 3)))]

    def fail_pose(frame: np.ndarray, *, bboxes: list[np.ndarray]) -> None:
        raise RuntimeError

    completed, detected, failures, rate, confidence, height = detect._infer_samples(
        samples, FakeDetector(), fail_pose
    )

    assert (completed, detected, failures) == (0, 0, 1)
    assert all(math.isnan(value) for value in (rate, confidence, height))


def test_detect_row_blanks_nonfinite_measurements() -> None:
    row = detect._row(
        detect.DetectResult("a-0000000000000000", 0, 0, 0, math.nan, math.inf, -math.inf)
    )
    assert row == {
        "asset_id": "a-0000000000000000",
        "detect_rate": "",
        "detect_conf_median": "",
        "subject_px_height_median": "",
    }


def test_detect_provenance_records_devices_sampling_models_and_thresholds() -> None:
    provenance = detect.PROVENANCE
    assert provenance["sampling"]["frames_per_asset"] == detect.SAMPLE_COUNT
    assert provenance["sampling"]["orientation_code_to_clockwise_degrees"] == {
        str(code): degrees for code, degrees in detect.ORIENTATION_ROTATION.items()
    }
    assert provenance["detector"]["input_size"] == list(detect._DET_INPUT_SIZE)
    assert provenance["detector"]["mode"] == detect.DETECTOR_MODE
    assert provenance["detector"]["nms_threshold"] == detect.DETECTOR_NMS_THRESHOLD
    assert provenance["detector"]["score_threshold"] == detect.DETECTOR_SCORE_THRESHOLD
    assert provenance["detector"]["device"] == detect.DETECTOR_DEVICE
    assert provenance["pose"]["model"] == detect.MODEL_NAME
    assert provenance["pose"]["input_size"] == list(detect.POSE_INPUT_SIZE)
    assert provenance["pose"]["to_openpose"] == detect.POSE_TO_OPENPOSE
    assert provenance["pose"]["device"] == detect.POSE_DEVICE
    assert provenance["statistics"]["active_keypoint_scope"] == detect.TRACKING_SCOPE
    assert provenance["statistics"]["active_keypoint_indices"] == list(
        detect.ACTIVE_KEYPOINT_INDICES
    )
    assert provenance["backend"]["name"] == detect.BACKEND

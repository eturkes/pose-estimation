"""Shared pytest fixtures for the pose-estimation test suite.

Existing tests still use their own private helpers; new tests should
prefer these fixtures so future test code grows from a single source.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Factories (callables) — flexible parametrisation per-test
# ---------------------------------------------------------------------------


@pytest.fixture
def make_random_landmarks():
    """Factory returning ``(n, 3)`` random landmarks at a given seed."""

    def _factory(n: int = 12, seed: int = 4817, scale: float = 300.0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.rand(n, 3) * scale

    return _factory


@pytest.fixture
def make_random_kps():
    """Factory returning ``(n, 2)`` random keypoints at a given seed."""

    def _factory(n: int = 5, seed: int = 6142, scale: float = 300.0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.rand(n, 2) * scale

    return _factory


@pytest.fixture
def make_arm_body():
    """Factory returning a plausible (12, 3) arm-scheme landmark array.

    Layout (approximate pixel positions):
        0  left shoulder    (100, 100)
        1  right shoulder   (200, 100)
        2  left elbow       (80, 200)
        3  right elbow      (220, 200)
        4  left wrist       (60, 300)
        5  right wrist      (240, 300)
        6  left index base  (55, 330)
        7  right index base (245, 330)
        8-11 other finger bases
    """

    def _factory() -> np.ndarray:
        lm = np.zeros((12, 3), dtype=np.float64)
        lm[0] = [100, 100, 0]
        lm[1] = [200, 100, 0]
        lm[2] = [80, 200, 0]
        lm[3] = [220, 200, 0]
        lm[4] = [60, 300, 0]
        lm[5] = [240, 300, 0]
        lm[6] = [55, 330, 0]
        lm[7] = [245, 330, 0]
        lm[8] = [50, 325, 0]
        lm[9] = [250, 325, 0]
        lm[10] = [45, 320, 0]
        lm[11] = [255, 320, 0]
        return lm

    return _factory


@pytest.fixture
def make_hand_landmarks():
    """Factory returning a (21, 3) hand landmark array at a given wrist."""

    def _factory(wrist_x: float = 60, wrist_y: float = 300) -> np.ndarray:
        lm = np.zeros((21, 3), dtype=np.float64)
        lm[0] = [wrist_x, wrist_y, 0]
        for i in range(1, 21):
            lm[i] = [wrist_x + i * 2, wrist_y + i, 0]
        return lm

    return _factory


@pytest.fixture
def make_palm_det():
    """Factory returning a normalised palm detection dict."""

    def _factory(cx_norm: float, cy_norm: float, size: float = 0.1, score: float = 0.9) -> dict:
        half = size / 2
        return {
            "box": np.array(
                [cx_norm - half, cy_norm - half, cx_norm + half, cy_norm + half],
                dtype=np.float32,
            ),
            "keypoints": np.array([[cx_norm, cy_norm]] * 7, dtype=np.float32),
            "score": score,
        }

    return _factory

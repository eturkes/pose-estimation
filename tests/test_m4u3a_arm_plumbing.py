"""M4.3a P01-P05: registry by mmpose-table names, score calibration, device resolution."""

from __future__ import annotations

import importlib
import json
import random
import sys
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest


def _run():
    return importlib.import_module("pose_estimation.run")


def _cal():
    return importlib.import_module("pose_estimation.score_calibration")


@pytest.mark.parametrize(
    ("key", "checkpoint", "size", "device"),
    [
        ("rtmw-m-256", "rtmw-dw-l-m_simcc-cocktail14_270e-256x192", (192, 256), "NPU"),
        ("rtmw-l-256", "rtmw-dw-x-l_simcc-cocktail14_270e-256x192", (192, 256), "GPU"),
        ("rtmw-l-384", "rtmw-dw-x-l_simcc-cocktail14_270e-384x288", (288, 384), "GPU"),
        ("rtmw-x-384", "rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288", (288, 384), "GPU"),
        ("dwpose-l-384", "rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288", (288, 384), "GPU"),
    ],
)
def test_p01_registry_keys_name_the_mmpose_table_model(key, checkpoint, size, device):
    entry = _run().MODEL_REGISTRY[key]
    assert checkpoint in entry["pose"]
    assert entry["pose_input_size"] == size
    assert entry["pose_device"] == device
    assert entry["n_kps"] == 133


def test_p01_legacy_key_is_the_rtmw_m_entry_with_raw_scores():
    registry = _run().MODEL_REGISTRY
    assert registry["rtmw-l"] is registry["rtmw-m-256"]
    assert registry["rtmw-l"]["calibration"] is None
    assert "RTMW-m" in registry["rtmw-l"]["label"]


def _pav_reference(y, w):
    """Brute-force isotonic regression: minimise sum w (f - y)^2 over non-decreasing f."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    best = None
    # Exhaustive over block partitions is exponential; for <= 7 points enumerate cut masks.
    n = len(y)
    for mask in range(1 << (n - 1)):
        cuts = [0] + [i + 1 for i in range(n - 1) if mask >> i & 1] + [n]
        fit = np.concatenate(
            [np.full(b - a, np.average(y[a:b], weights=w[a:b])) for a, b in pairwise(cuts)]
        )
        if np.all(np.diff(fit) >= -1e-12):
            loss = float(np.sum(w * (fit - y) ** 2))
            if best is None or loss < best[0] - 1e-12:
                best = (loss, fit)
    assert best is not None
    return best[1]


@pytest.mark.parametrize("seed", range(8))
def test_p02_pav_equals_the_brute_force_isotonic_fit(seed):
    rng = random.Random(seed)  # noqa: S311 - fixture data, never key material
    y = [rng.random() for _ in range(6)]
    w = [rng.randint(1, 4) for _ in range(6)]
    np.testing.assert_allclose(_cal()._pav(np.array(y), np.array(w, float)), _pav_reference(y, w))


@pytest.mark.parametrize("seed", range(5))
def test_p02_fit_is_monotone_bounded_and_compressed(seed):
    cal = _cal()
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 8, 5000)
    correct = rng.random(5000) < 1 / (1 + np.exp(-(raw - 4)))
    curve = cal.fit(raw, correct)
    assert len(curve.x) <= cal.MAX_KNOTS
    assert np.all(np.diff(curve.x) > 0)
    assert np.all(np.diff(curve.y) >= 0)
    assert min(curve.y) >= 0
    assert max(curve.y) <= 1
    grid = np.linspace(-1, 9, 200)
    assert np.all(np.diff(curve(grid)) >= -1e-12)
    assert curve(np.array([-5.0]))[0] == curve.y[0]  # clamped below
    assert curve(np.array([50.0]))[0] == curve.y[-1]  # clamped above


def test_p02_apply_maps_each_group_by_its_curve_and_keeps_nan():
    cal = _cal()
    hand = cal.Curve((0.0, 10.0), (0.0, 1.0))
    calibration = cal.ScoreCalibration({"hand": hand})
    scores = np.full((2, 133), 5.0)
    scores[0, 100] = np.nan
    out = calibration.apply(scores)
    np.testing.assert_array_equal(out[:, :91], scores[:, :91])  # identity outside the curve
    assert out[1, 100] == pytest.approx(0.5)
    assert np.isnan(out[0, 100])
    np.testing.assert_array_equal(cal.identity().apply(scores), scores)


def test_p02_nc1_a_falling_curve_is_refused(tmp_path, monkeypatch):
    cal = _cal()
    with pytest.raises(ValueError):  # noqa: PT011 - message not contract-owned
        cal.Curve((0.0, 1.0, 2.0), (0.0, 0.8, 0.6))
    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    (tmp_path / "bad.json").write_text(json.dumps({"hand": {"x": [0, 1], "y": [0.9, 0.1]}}))
    with pytest.raises(ValueError):  # noqa: PT011
        cal.load("bad.json")


class _RawPose:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image, bboxes=()):
        return np.zeros((1, 133, 2)), np.full((1, 133), 4.0)


def _solution(monkeypatch, calibration):
    run = _run()
    fake = SimpleNamespace(YOLOX=lambda *a, **k: None, RTMPose=_RawPose)
    monkeypatch.setitem(sys.modules, "rtmlib", fake)
    return run.SplitDeviceSolution(
        det="d", det_input_size=(640, 640), det_device="GPU", pose_class="RTMPose", pose="p",
        pose_input_size=(192, 256), pose_device="GPU", backend="openvino", calibration=calibration,
    )  # fmt: skip


def test_p03_calibrated_arm_hands_the_tracker_calibrated_scores(monkeypatch, tmp_path):
    cal = _cal()
    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    (tmp_path / "arm.json").write_text(
        json.dumps({g: {"x": [0, 8], "y": [0, 1]} for g in ("body", "foot", "face", "hand")})
    )
    solution = _solution(monkeypatch, "arm.json")
    _, scores = solution.pose_model(np.zeros((8, 8, 3)), bboxes=[[0, 0, 8, 8]])
    np.testing.assert_allclose(scores, 0.5)


def test_p03_p04_uncalibrated_arm_hands_the_tracker_raw_scores(monkeypatch):
    solution = _solution(monkeypatch, None)
    assert isinstance(solution.pose_model, _RawPose)
    _, scores = solution.pose_model(np.zeros((8, 8, 3)), bboxes=[[0, 0, 8, 8]])
    np.testing.assert_array_equal(scores, 4.0)


@pytest.mark.parametrize(("model", "device"), [("rtmw-l", "NPU"), ("rtmw-x-384", "GPU")])
@pytest.mark.parametrize("driver", ["corpus_run_2d.py", "pilot_corpus_run.py"])
def test_p05_driver_pose_device_defaults_to_the_model_entry(monkeypatch, model, device, driver):
    import runpy

    from test_m2u101_hand_gate import _ROOT

    module = runpy.run_path(str(_ROOT / "scripts" / driver), run_name="m4u3a_probe")
    monkeypatch.setattr(sys, "argv", [driver, "--model", model])
    assert module["_parse_args"]().pose_device == device
    monkeypatch.setattr(sys, "argv", [driver, "--model", model, "--pose-device", "CPU"])
    assert module["_parse_args"]().pose_device == "CPU"


def test_p05_generation_records_the_calibration():
    corpus = importlib.import_module("pose_estimation.corpus_run")
    assert "calibration" in corpus.POSE_CONFIG_FIELDS
    args = SimpleNamespace(**(dict.fromkeys(corpus.POSE_CONFIG_FIELDS, "x") | {"model": "rtmw-l"}))
    assert corpus.pose_config(args)["calibration"] == "identity"


@pytest.mark.parametrize("seed", range(6))
def test_p02_r08_public_fit_equals_the_isotonic_solution_at_every_distinct_score(seed):
    """reviewer-10 R08: the shipped `fit`, not only `_pav`, must equal the isotonic regression of
    `correct` on `raw` (ties pooled) at each observed score."""
    cal = _cal()
    rng = random.Random(100 + seed)  # noqa: S311 - fixture data
    raw = [rng.choice([0.1, 0.2, 0.35, 0.5, 0.8, 0.9]) for _ in range(7)]
    correct = [rng.random() < 0.5 for _ in raw]
    curve = cal.fit(np.array(raw), np.array(correct, float))
    ux = sorted(set(raw))
    means = [np.mean([c for r, c in zip(raw, correct, strict=True) if r == u]) for u in ux]
    weights = [sum(r == u for r in raw) for u in ux]
    expected = _pav_reference(means, weights) if len(ux) > 1 else np.array(means)
    np.testing.assert_allclose(curve(np.array(ux)), expected, atol=1e-12)


def test_p02_r06_knots_must_be_finite_and_infinite_scores_clamp():
    cal = _cal()
    with pytest.raises(ValueError):  # noqa: PT011
        cal.Curve((0.0, float("nan")), (0.0, 1.0))
    with pytest.raises(ValueError):  # noqa: PT011
        cal.Curve((0.0, 1.0), (0.0, float("inf")))
    calibration = cal.ScoreCalibration({"hand": cal.Curve((0.0, 1.0), (0.2, 0.9))})
    scores = np.zeros((1, 133))
    scores[0, 100], scores[0, 101], scores[0, 102] = np.inf, -np.inf, np.nan
    out = calibration.apply(scores)
    assert (out[0, 100], out[0, 101]) == (0.9, 0.2)
    assert np.isnan(out[0, 102])


def test_r07_mediapipe_delegation_gets_a_pose_device(monkeypatch):
    run = _run()
    seen = {}
    monkeypatch.setattr(
        run, "_run_mediapipe", lambda args: seen.setdefault("device", args.pose_device) and 0
    )
    monkeypatch.setattr(sys, "argv", ["run", "--model", "mediapipe", "--headless"])
    with pytest.raises(SystemExit):
        run.main()
    assert seen["device"] == "NPU"

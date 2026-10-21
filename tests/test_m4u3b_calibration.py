"""M4.3b P01-P02 + P04: reference-equivalent calibration and the per-model hand gate."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _cal():
    return importlib.import_module("pose_estimation.score_calibration")


def _sample(a, b, n=40_000, lo=0.0, hi=1.2, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(lo, hi, n)
    correct = rng.random(n) < 1 / (1 + np.exp(-(a * raw + b)))
    return raw, correct


@pytest.mark.parametrize(("a", "b"), [(6.0, -3.0), (2.5, -4.0), (9.0, -1.0)])
def test_p01_logistic_fit_recovers_its_generating_parameters(a, b):
    raw, correct = _sample(a, b, lo=-1, hi=3)
    fa, fb = _cal()._logistic(raw, correct)
    assert fa == pytest.approx(a, rel=0.05)
    assert fb == pytest.approx(b, abs=0.05 * abs(b) + 0.05)


def test_p01_p02_reference_map_equalises_probability_of_correctness():
    cal = _cal()
    ref = _sample(6.0, -3.0, seed=1)  # reference: P = .5 at raw 0.5
    arm = _sample(1.5, -4.0, lo=0, hi=8, seed=2)  # another scale: P = .5 at raw 2.667
    curve = cal.fit_reference(*arm, *ref)
    assert np.all(np.diff(curve.x) > 0)
    assert np.all(np.diff(curve.y) >= 0)
    a, b = cal._logistic(*arm)
    a_ref, b_ref = cal._logistic(*ref)
    for raw in (2.0, 2.667, 3.5, 5.0):
        mapped = float(curve(np.array([raw]))[0])
        p_arm = 1 / (1 + np.exp(-(a * raw + b)))
        p_ref = 1 / (1 + np.exp(-(a_ref * mapped + b_ref)))
        assert p_ref == pytest.approx(p_arm, abs=1e-9)  # round trip: equal P(correct)
    assert float(curve(np.array([0.0]))[0]) == 0.0  # clamped, never negative


def test_p01_reference_map_recovers_the_generating_affine_map_within_2_percent():
    # Oracle = the generating logits, never the fitted ones: equal P(correct) at
    # 1.5 r - 6 = 6 m - 3  →  m = 0.25 r - 0.5. The intercept is a difference of two fitted
    # intercepts, so the sample is sized for it: at 1e6 per arm the fit errs <= 0.7 % over six
    # seed pairs, while a 4 % slope bias in `_logistic` moves the intercept >= 3.3 %.
    n = 1_000_000
    curve = _cal().fit_reference(
        *_sample(1.5, -6.0, n=n, lo=0, hi=8, seed=2), *_sample(6.0, -3.0, n=n, seed=1)
    )
    r0, r1 = 3.0, 7.0  # inside the arm's range, past the zero crossing at r = 2
    m0, m1 = (float(v) for v in curve(np.array([r0, r1])))
    slope = (m1 - m0) / (r1 - r0)
    assert slope == pytest.approx(0.25, rel=0.02)
    assert m0 - slope * r0 == pytest.approx(-0.5, rel=0.02)


def test_p01_a_score_that_falls_with_correctness_is_refused():
    cal = _cal()
    with pytest.raises(ValueError):  # noqa: PT011
        cal.fit_reference(*_sample(-3.0, 1.0), *_sample(6.0, -3.0))
    with pytest.raises(ValueError):  # noqa: PT011
        cal._logistic(np.array([0.1, 0.2]), np.array([1.0, 1.0]))


def test_p04_registry_hand_gate_reaches_the_gate(monkeypatch):
    run = importlib.import_module("pose_estimation.run")
    built = []

    class Gate:
        def __init__(self, on=0.65, off=0.5, min_points=10):
            built.append((on, off))

    monkeypatch.setattr(run, "HandPresenceGate", Gate)
    entry = dict(run.MODEL_REGISTRY["rtmw-l-384"], hand_gate=(0.7, 0.45))
    assert run.hand_gate_for(entry, enabled=True) is not None
    assert built[-1] == (0.7, 0.45)
    run.hand_gate_for(run.MODEL_REGISTRY["rtmw-m-256"], enabled=True)
    assert built[-1] == (0.65, 0.5)  # the shipped thresholds when the entry names none
    assert run.hand_gate_for(entry, enabled=False) is None


def _qualify():
    return importlib.import_module("pose_estimation.bench.qualify")


def test_p03_exported_hand_needs_a_located_point_with_positive_confidence():
    q = _qualify()
    row = {"left_hand_3_x": "12.0", "left_hand_3_conf": "0.7"}
    row |= {"right_hand_0_x": "", "right_hand_0_conf": "0.9", "right_hand_5_x": "3.0"}
    assert q.exported_hands(row) == {"L": True, "R": False}  # R: no x, or no confidence
    assert q.exported_hands({"left_hand_3_x": "1.0", "left_hand_3_conf": "0"})["L"] is False
    assert q.exported_hands(None) == {"L": False, "R": False}  # no row at the frame


def test_p03_presence_metrics_exact_and_unagreed_sides_excluded():
    q = _qualify()
    rows = []
    rows += q.presence_rows(1, {"L": "vis", "R": "occ"}, {"L": True, "R": True})  # tp, fp
    rows += q.presence_rows(1, {"L": "part", "R": "oof"}, {"L": False, "R": False})  # fn, tn
    rows += q.presence_rows(2, {"L": "vis", "R": "?"}, {"L": True, "R": True})  # tp, excluded
    rows += q.presence_rows(2, {"L": "", "R": "vis"}, {"L": True, "R": True})  # excluded, tp
    assert len(rows) == 6
    s = q.presence_summary(rows, seed=0)
    assert (s["sides"], s["subjects"]) == (6, 2)
    assert s["precision"] == pytest.approx(3 / 4)  # tp 3, fp 1
    assert s["recall"] == pytest.approx(3 / 4)  # tp 3, fn 1
    assert s["f1"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 1))


def test_p03_interval_resamples_subjects_not_sides():
    q = _qualify()
    hit = q.presence_rows("a", {"L": "vis", "R": "part"}, {"L": True, "R": True})
    miss = q.presence_rows("b", {"L": "vis", "R": "part"}, {"L": False, "R": False})
    rows = hit * 3 + miss * 3  # a: 6 tp, b: 6 fn
    s = q.presence_summary(rows, seed=7)
    # Subjects: {a,a} = 1, {b,b} = 0 each at p .25 → the interval spans [0, 1]. A side-level
    # resample of 12 rows reaches neither end at 2.5 % (all-miss or all-hit p = 2**-12).
    assert s["f1_interval"] == [0.0, 1.0]
    assert q.presence_summary(rows, seed=7) == s  # seeded


def test_p04_pose_config_digest_records_a_refit_gate(monkeypatch):
    run = importlib.import_module("pose_estimation.run")
    corpus_run = importlib.import_module("pose_estimation.corpus_run")
    assert corpus_run.calibration_digest("rtmw-m-256") == "identity"
    shipped = corpus_run.calibration_digest("rtmw-l-384")
    entry = dict(run.MODEL_REGISTRY["rtmw-l-384"], hand_gate=(0.7, 0.45))
    monkeypatch.setitem(run.MODEL_REGISTRY, "rtmw-l-384", entry)
    refit = corpus_run.calibration_digest("rtmw-l-384")
    monkeypatch.setitem(run.MODEL_REGISTRY, "rtmw-m-256", dict(entry, calibration=None))
    assert len({shipped, refit, corpus_run.calibration_digest("rtmw-m-256"), "identity"}) == 4

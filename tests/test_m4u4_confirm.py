"""M4.4 P01-P04: confirmation endpoints, subject bootstrap, decision table, frozen candidates."""

from __future__ import annotations

import importlib
import math

import pytest

W, H = 1000, 500  # scale = max = 1000


def _confirm():
    return importlib.import_module("pose_estimation.bench.confirm")


def _joint(xy=None, state=None):
    if xy is not None:
        return {"state": "vis", "xy": list(xy)}
    return {"state": state or "?"}


def _record(left_hand=None, right_hand=None, body=None):
    """Hand dicts name -> (x, y); body dict side -> name -> (x, y)."""
    names = ("wrist", "thumb_tip", "index_mcp", "index_tip", "little_tip")
    sides = {}
    for side, hand in (("L", left_hand or {}), ("R", right_hand or {})):
        sides[side] = {
            "hand": "vis" if hand else "occ",
            "extent": None,
            "joints": {n: _joint(hand.get(n), "occ") for n in names},
            "body": {
                n: _joint((body or {}).get(side, {}).get(n), "occ")
                for n in ("shoulder", "elbow", "wrist")
            },
        }
    return {"frame": {"key": "k", "index": 0, "width": W, "height": H}, "sides": sides}


def _row(points):
    """points: column prefix -> (x_px, y_px); everything exported at score 0.9."""
    row = {}
    for prefix, (x, y) in points.items():
        row[f"{prefix}_x"], row[f"{prefix}_y"] = str(x / 1000), str(y / 1000)
        row[f"{prefix}_{'conf' if 'hand' in prefix else 'vis'}"] = "0.9"
    return row


def test_p01_hand_error_is_normalised_by_the_user_extent_and_misses_count_one():
    c = _confirm()
    hand = {"wrist": (100, 100), "index_tip": (200, 100)}  # extent diagonal 100
    record = _record(left_hand=hand)
    row = _row({"left_hand_0": (110, 100), "left_hand_8": (200, 100)})
    score = c.score_frame(record, row, "s1")
    assert score.e1 == pytest.approx(0.05)  # median of (10/100, 0)
    row_missing = _row({"left_hand_0": (110, 100)})  # index tip not exported
    assert c.score_frame(record, row_missing, "s1").e1 == pytest.approx((0.1 + 1.0) / 2)
    zero = _row({"left_hand_0": (100, 100), "left_hand_8": (200, 100)})
    zero["left_hand_8_conf"] = "0"  # a zero score is no observation
    assert c.score_frame(record, zero, "s1").e1 == pytest.approx(0.5)


def test_p01_arm_error_uses_upper_arm_length_with_the_other_side_fallback():
    c = _confirm()
    body = {
        "R": {"shoulder": (0, 0), "elbow": (0, 200), "wrist": (0, 400)},
        "L": {"wrist": (500, 400)},
    }
    record = _record(body=body)
    row = _row({
        "body_right_shoulder": (0, 0), "body_right_elbow": (0, 220), "body_right_wrist": (0, 400),
        "body_left_wrist": (520, 400),
    })  # fmt: skip
    # right: errors 0, 0.1, 0 (norm 200); left wrist uses the right upper arm: 20/200 = 0.1
    assert c.score_frame(record, row, "s1").e2 == pytest.approx(0.05)


def test_p01_presence_agreement_and_side_swaps():
    c = _confirm()
    hand = {"wrist": (100, 100), "index_tip": (200, 100)}
    body = {"L": {"wrist": (100, 300)}, "R": {"wrist": (800, 300)}}
    record = _record(left_hand=hand, body=body)
    row = _row({"left_hand_0": (100, 100), "right_hand_0": (700, 100),
                "body_left_wrist": (800, 300), "body_right_wrist": (100, 300)})  # fmt: skip
    score = c.score_frame(record, row, "s1")
    assert score.e3 == (True, False)  # left present = visible; right exported but user occ
    assert score.e4 is True
    assert c.score_frame(record, None, "s1").e3 == (False, True)  # no row: nothing exported


def test_p01_subject_means_aggregate_frames_then_subjects():
    c = _confirm()
    f = c.FrameScore
    scores = [
        f("a", 0.1, 0.2, (True, False), False),
        f("a", 0.3, None, (True,), True),
        f("b", None, 0.4, (), None),
    ]
    means = c.subject_means(scores)
    assert means["a"]["E1"] == pytest.approx(0.2)
    assert means["a"]["E3"] == pytest.approx(3 / 4)  # frames first: (1/2 + 1) / 2
    assert means["a"]["E4"] == pytest.approx(0.5)
    assert math.isnan(means["b"]["E1"])


def test_p02_bootstrap_resamples_subjects_and_is_reproducible():
    c = _confirm()
    base = {s: {"E1": 0.5} for s in "abcdef"}
    cand = {s: {"E1": 0.4} for s in "abcdef"}
    assert c.bootstrap(cand, base, "E1") == pytest.approx((-0.1, -0.1, -0.1))
    varied = {s: {"E1": 0.4 + 0.05 * i} for i, s in enumerate("abcdef")}
    first = c.bootstrap(varied, base, "E1", n=500, seed=3)
    assert first == c.bootstrap(varied, base, "E1", n=500, seed=3)
    assert first[1] < first[0] < first[2]
    with pytest.raises(ValueError):  # noqa: PT011
        c.bootstrap({"a": {"E1": 0.1}}, {"a": {"E1": 0.2}}, "E1")


@pytest.mark.parametrize(
    ("intervals", "state"),
    [
        (
            {
                "E1": (-0.05, -0.08, -0.01),
                "E2": (0, -0.01, 0.01),
                "E3": (0, -0.02, 0.02),
                "E4": (0, 0, 0),
            },
            "superior",
        ),
        (
            {
                "E1": (0.0, -0.01, 0.015),
                "E2": (0, -0.01, 0.01),
                "E3": (0, -0.02, 0.02),
                "E4": (0, 0, 0),
            },
            "non-inferior",
        ),
        (
            {
                "E1": (0.05, 0.03, 0.07),
                "E2": (0, -0.01, 0.01),
                "E3": (0, -0.02, 0.02),
                "E4": (0, 0, 0),
            },
            "regression",
        ),
        (
            {
                "E1": (-0.01, -0.04, 0.03),
                "E2": (0, -0.01, 0.01),
                "E3": (0, -0.02, 0.02),
                "E4": (0, 0, 0),
            },
            "inconclusive",
        ),
        (
            {
                "E1": (0, -0.01, 0.01),
                "E2": (0, -0.01, 0.01),
                "E3": (-0.08, -0.1, -0.06),
                "E4": (0, 0, 0),
            },
            "regression",
        ),
    ],
)
def test_p03_decision_table_reaches_every_state(intervals, state):
    assert _confirm().decide(intervals) == state


def test_p03_adoption_follows_public_rank_among_non_inferior_and_keeps_the_baseline():
    c = _confirm()
    states = {"x": "regression", "l": "non-inferior", "m2": "superior"}
    assert c.adopt(states, ["x", "l", "m2"], "base") == "l"
    assert c.adopt({"x": "inconclusive"}, ["x"], "base") == "base"


def test_p04_scorer_refuses_without_named_candidates(tmp_path):
    c = _confirm()
    contract = tmp_path / "contract.md"
    contract.write_text("## 8. Amendments\n")
    with pytest.raises(ValueError):  # noqa: PT011
        c.frozen_candidates(contract)
    contract.write_text(
        "## 8. Amendments\n\n- **A01 (candidates).** `rtmw-x-384` and `rtmw-l-384`.\n"
    )
    assert c.frozen_candidates(contract) == ["rtmw-x-384", "rtmw-l-384"]


def test_p01_observed_errors_are_not_capped_at_the_miss_penalty():
    """1.0 is the miss value (contract §2); a far exported point keeps its error (closing review C05a)."""
    c = _confirm()
    record = _record(left_hand={"wrist": (100, 100), "index_tip": (200, 100)})
    row = _row({"left_hand_0": (400, 100), "left_hand_8": (500, 100)})
    assert c.score_frame(record, row, "s1").e1 == pytest.approx(3.0)  # 300 px / 100 px extent

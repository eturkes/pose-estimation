"""M4.3 P01-P04: pilot endpoints G1-G4, paired subject intervals, decision, request refusals."""

from __future__ import annotations

import importlib
import math

import pytest

SCALE = 1000.0  # max frame dimension: exported = pixels / SCALE


def _pilot():
    return importlib.import_module("pose_estimation.bench.pilot")


def _joint(state, xy=None):
    return {"state": state} | ({"xy": list(xy)} if xy is not None else {})


def _side(hand, joints, extent=None):
    return {"hand": hand, "joints": joints, "extent": extent}


def _put(row, prefix, xy, confidence, score=0.9):
    row[f"{prefix}_x"], row[f"{prefix}_y"] = str(xy[0] / SCALE), str(xy[1] / SCALE)
    row[f"{prefix}_{confidence}"] = str(score)


def _frame_one():
    """Left hand + arm scored, right side `?` (excluded)."""
    consensus = {
        "patient": [0, 0, 600, 800],  # diagonal 1000 → arm tolerance 100 px
        "sides": {
            "L": _side(
                "vis",
                {
                    "joints.wrist": _joint("vis", (500, 500)),
                    "joints.index_tip": _joint("vis", (560, 500)),
                    "joints.thumb_tip": _joint("vis", (540, 520)),
                    "joints.little_tip": _joint("occ"),
                    "body.shoulder": _joint("vis", (300, 200)),
                    "body.elbow": _joint("vis", (350, 350)),
                    "body.wrist": _joint("?"),
                },
                extent=[480, 480, 580, 480],  # diagonal 100 → hand tolerance 60 px
            ),
            "R": _side("?", {"joints.wrist": _joint("vis", (900, 500))}),
        },
    }
    row: dict[str, str] = {}
    _put(row, "left_hand_0", (510, 500), "conf")  # 10 px → hit
    _put(row, "left_hand_8", (630, 500), "conf")  # 70 px → miss
    row["left_hand_4_x"] = ""  # thumb tip not exported → miss
    _put(row, "body_left_shoulder", (340, 230), "vis")  # 50 px → hit
    _put(row, "body_left_elbow", (350, 350), "vis", score=0.0)  # masked → miss
    _put(row, "right_hand_0", (900, 500), "conf")
    return consensus, row


def _frame_two(swapped):
    """Both hands present + exported; the left hand exported on the right hand's spot."""
    consensus = {
        "patient": None,
        "sides": {
            "L": _side("vis", {"joints.wrist": _joint("vis", (100, 100))}, [90, 90, 110, 110]),
            "R": _side("part", {"joints.wrist": _joint("vis", (700, 100))}, [690, 90, 710, 110]),
        },
    }
    row: dict[str, str] = {}
    _put(row, "left_hand_0", (700, 100) if swapped else (100, 100), "conf")
    _put(row, "right_hand_0", (700, 100), "conf")
    return consensus, row


def test_p01_side_rows_and_endpoints_exact():
    p = _pilot()
    rows = p.side_rows("s1", *_frame_one(), SCALE)
    left, right = rows
    assert (left["hand_hits"], left["hand_n"]) == (1, 3)  # not-exported thumb = a miss
    assert (left["arm_hits"], left["arm_n"]) == (1, 2)  # `?` wrist out, masked elbow a miss
    assert (left["present"], left["exported"]) == (True, True)
    assert "present" not in right  # `?` hand state
    assert right["hand_n"] == 0
    assert "swap" not in left  # one labelled hand only
    rows += p.side_rows("s1", *_frame_two(swapped=True), SCALE)
    assert [r.get("swap") for r in rows[2:]] == [True, False]
    e = p.endpoints(rows)
    assert e["G1"] == 1.0  # 3 tp, 0 fp, 0 fn
    assert e["G2"] == pytest.approx(2 / 5)  # 1/3 + frame two: swapped left misses, right hits
    assert e["G3"] == pytest.approx(1 / 2)
    assert e["G4"] == pytest.approx(1 / 2)


def test_p01_absent_label_with_exported_hand_is_a_false_positive():
    p = _pilot()
    consensus, row = _frame_two(swapped=False)
    consensus["sides"]["R"]["hand"] = "occ"
    rows = p.side_rows("s1", consensus, row, SCALE)
    assert rows[1]["present"] is False
    assert rows[1]["exported"] is True
    assert p.endpoints(rows)["G1"] == pytest.approx(2 / 3)  # tp 1, fp 1
    assert math.isnan(p.endpoints(rows)["G4"])  # one labelled-present hand → no swap test


def _arm_rows(subject, hits, n=2):
    return [{"subject": subject, "side": "L", "hand_hits": 0, "hand_n": 0, "arm_hits": hits,
             "arm_n": n}]  # fmt: skip


def test_p02_constant_difference_gives_a_zero_width_interval():
    p = _pilot()
    base = [r for s in "abcd" for r in _arm_rows(s, 1)]
    arm = [r for s in "abcd" for r in _arm_rows(s, 2)]
    out = p.paired_intervals(arm, base, 200, 3)
    assert out["G3"] == (0.5, 0.5, 0.5)
    assert repr(out) == repr(p.paired_intervals(arm, base, 200, 3))  # seeded; NaN-safe compare
    assert math.isnan(out["G1"][0])  # no labelled presence anywhere


def test_p02_subjects_not_sides_are_resampled():
    p = _pilot()
    base = [r for s in "ab" for r in _arm_rows(s, 2) * 5]
    arm = _arm_rows("a", 2) * 5 + _arm_rows("b", 0) * 5  # all loss in subject b
    point, lo, hi = p.paired_intervals(arm, base, 400, 1)["G3"]
    assert point == -0.5
    assert lo == -1.0  # {b, b} drawn: a side-level resample of 20 sides almost never gets here
    assert hi == 0.0  # {a, a}


def _intervals(g1=(0.0, -0.01, 0.01), g2=(0.0, -0.01, 0.01), g3=(0.0, -0.01, 0.01),
               g4=(0.0, -0.01, 0.01)):  # fmt: skip
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4}


def test_p03_decision_table_states():
    p = _pilot()
    assert p.decide(_intervals()) == "pass"
    assert p.decide(_intervals(g1=(-0.01, -0.051, 0.03))) == "refit"
    assert p.decide(_intervals(g1=(-0.021, -0.04, 0.0))) == "refit"  # M4.3b D02 point rule
    assert p.decide(_intervals(g2=(0.0, -0.06, 0.05))) == "fail"
    assert p.decide(_intervals(g4=(0.0, -0.01, 0.051))) == "fail"
    assert p.decide(_intervals(g1=(-0.1, -0.2, 0.0), g3=(0.0, -0.06, 0.0))) == "fail"
    assert p.decide(_intervals(g2=(0.0, -0.05, 0.0), g4=(0.0, 0.0, 0.05))) == "pass"  # bounds hold


def test_p04_request_refuses_an_unknown_arm():
    p = _pilot()
    p.check_request("rtmw-m-256")
    with pytest.raises(ValueError, match="MODEL_REGISTRY"):
        p.check_request("not-an-arm")


def test_p01_g4_needs_both_hands_exported():
    """Contract §2 G4 population = both hands labelled present AND exported (review C05c)."""
    p = _pilot()
    consensus, row = _frame_two(swapped=False)
    row = {k: v for k, v in row.items() if not k.startswith("right_hand_")}
    rows = p.side_rows("s1", consensus, row, SCALE)
    assert all("swap" not in r for r in rows)
    assert math.isnan(p.endpoints(rows)["G4"])


def test_a01_whole_clip_replay_exports_the_production_task_span(tmp_path, monkeypatch):
    """M4.3 A01: a whole-clip replay = the corpus run of that clip, camera-survey trim included."""
    import csv
    from types import SimpleNamespace

    import numpy as np

    from pose_estimation import run, subject_tracker
    from pose_estimation.bench import replay
    from test_m2u92_keypoint_hygiene import _Capture

    monkeypatch.setattr(run, "open_capture", lambda *a, **k: _Capture([(48, 64)] * 12))
    monkeypatch.setattr(run, "_patch_rtmlib_openvino", lambda: None)

    class Tracker:
        def __init__(self, *args, **kwargs):
            self.last_track_ids = [7]

        def reset(self):
            pass

        def __call__(self, frame):
            return np.full((1, 133, 2), 20.0), np.full((1, 133), 0.9)

    monkeypatch.setattr(subject_tracker, "SubjectTracker", Tracker)
    survey = SimpleNamespace(
        span=(4, 12), n_frames=12, to_reference=np.repeat(np.eye(3)[None], 12, 0)
    )
    monkeypatch.setattr(run, "survey_source", lambda *a, **k: survey)
    media = tmp_path / "synthetic.mp4"
    media.touch()

    def frames(path):
        with path.open(newline="") as stream:
            return [int(r["frame_idx"]) for r in csv.DictReader(stream)]

    # The corpus driver's own invocation (survey on by default there) exports the span alone.
    run.main(["--source", str(media), f"--output-dir={tmp_path / 'prod'}", *replay.CORPUS_ARGS])
    assert frames(tmp_path / "prod" / "synthetic.csv") == list(range(4, 12))
    assert frames(replay.replay_window(media, 0, 11, tmp_path / "replay")) == list(range(4, 12))

"""M2.9.2 reviewer H04/H06: real pipeline witnesses over synthetic tracker output."""

from types import SimpleNamespace

import numpy as np
import pytest

from pose_estimation import run
from test_m2u92_keypoint_hygiene import _Capture, _Tracker


class _NamedTracker(_Tracker):
    last_track_ids = (42,)


def _rows(monkeypatch, outputs, *, smoother, single_subject=False, named=False):
    capture = _Capture([(48, 64)] * len(outputs))
    tracker = (_NamedTracker if named else _Tracker)(outputs)
    rows = []
    writer = SimpleNamespace(writerow=rows.append)
    monkeypatch.setattr(run, "open_capture", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr(
        run,
        "open_csv_writer",
        lambda *_args, **_kwargs: (SimpleNamespace(close=lambda: None), writer),
    )
    run.process_source(
        SimpleNamespace(
            tracking="body", headless=True, single_subject=single_subject, max_frames=0
        ),
        tracker,
        "synthetic-review.mp4",
        draw_skeleton=None,
        smoother=smoother,
        output_csv="memory-only",
    )
    assert capture.released
    assert rows
    return rows


@pytest.mark.parametrize("named", [False, True], ids=["rtmlib", "subject-ids"])
def test_h04_entirely_zeroed_person_exports_zero_evidence(monkeypatch, named):
    """A03 permits whole-row carry; every exported confidence must remain zero."""
    outputs = [(np.full((1, 17, 2), x), np.ones((1, 17))) for x in (20.0, 25.0, 70.0)]
    rows = _rows(
        monkeypatch,
        outputs,
        smoother=run.KeypointSmoother(min_track_age=1),
        named=named,
    )
    assert len(rows) == 3
    assert float(rows[-2]["body_nose_vis"]) == 1.0
    assert float(rows[-1]["body_nose_vis"]) == 0.0
    evidence = [float(v) for k, v in rows[-1].items() if k.endswith(("_vis", "_conf"))]
    assert evidence
    np.testing.assert_array_equal(evidence, np.zeros_like(evidence))


@pytest.mark.parametrize("smooth", [False, True], ids=["no-smooth", "real-smoother"])
def test_h06_rtmlib_selects_by_hygienic_scores(monkeypatch, smooth):
    """A02: invalid points no longer count toward the legacy confidence argmax."""
    points = np.full((2, 133, 2), 20.0)
    points[1] = 40.0
    points[0, 91:112, 1] = 100.0
    scores = np.stack([np.full(133, 0.8), np.full(133, 0.7)])
    original, _ = run.filter_single_subject(points, scores)
    assert original[0, 5, 0] == 20.0
    rows = _rows(
        monkeypatch,
        [(points, scores)] * 3,
        smoother=run.KeypointSmoother() if smooth else None,
        single_subject=True,
    )
    observed = scores.copy()
    observed[0, 91:112] = 0
    selected = np.argmax(observed.mean(axis=1))
    assert selected == 1
    expected = points[selected, 5, 0] / 64
    actual = float(rows[-1]["body_left_shoulder_x"])
    assert actual == expected, f"A02 hygienic selection missing: {expected=}, {actual=}"

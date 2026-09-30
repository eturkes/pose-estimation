"""M2.9.5 reviewer-4: P07 mechanism consistency + off-check camera-token witness."""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from pose_estimation import run
from pose_estimation.constraints import BoneLengthSmoother
from pose_estimation.keypoint_hygiene import apply_hygiene
from pose_estimation.rtmlib_smoothing import KeypointSmoother

_ROOT = Path(__file__).resolve().parents[1]
_UPPER = [5, 6, 7, 8, 9, 10, 91, 112]


def test_b06_p07_claimed_shoulder_hip_constraint_exists():
    contract = (_ROOT / ".agent/archive/contract-m2u95.md").read_text(encoding="utf-8")
    # An appended P07 amendment replaces the original claim without rewriting the archive.
    clauses = re.findall(r"\*\*P07\*\*(.*?)(?=\n- \*\*P\d+|\n## |\Z)", contract, re.S)
    assert clauses
    amendments = re.split(r"(?m)^### A\d+\b[^\n]*", contract)[1:]
    active = next(
        (section for section in reversed(amendments) if re.search(r"\bP07\b", section)),
        clauses[-1],
    )
    edges = {frozenset(edge) for edge in run.BONE_SEGMENTS_WB_BODY}
    if "shoulder-hip segments" in active:
        assert edges & {frozenset((5, 11)), frozenset((6, 12))}, (
            "P07 attributes upper-body changes to shoulder-hip segments; "
            f"the runner has none: {run.BONE_SEGMENTS_WB_BODY}"
        )


@pytest.mark.parametrize(("lower_body", "hips"), [(True, False), (False, True), (True, True)])
def test_b06_actual_constraint_isolates_upper_body_from_dropped_segments(lower_body, hips):
    args = {
        "segments": run.BONE_SEGMENTS_WB_BODY,
        "alpha": 0.05,
        "tolerance": 0.4,
        "distal_weight": 0.8,
    }
    baseline, dropped = BoneLengthSmoother(**args), BoneLengthSmoother(**args)
    valid = np.ones(133, dtype=bool)
    if lower_body:
        valid[13:23] = False
    if hips:
        valid[11:13] = False
    rng = np.random.default_rng(2095)
    upper_active = lower_changed = 0
    for _ in range(200):
        raw = rng.uniform(0, 1000, (133, 2))
        before, after = raw.copy(), raw.copy()
        baseline.update(1, before, validity=np.ones(133, dtype=bool))
        dropped.update(1, after, validity=valid)
        assert before[_UPPER].tobytes() == after[_UPPER].tobytes()
        upper_active += before[_UPPER].tobytes() != raw[_UPPER].tobytes()
        lower_changed += before[11:23].tobytes() != after[11:23].tobytes()
    assert upper_active > 0
    assert lower_changed > 0


def test_b06_legs_only_row_carries_and_recovery_can_change_observed_upper_body():
    baseline, dropped = KeypointSmoother(min_track_age=1), KeypointSmoother(min_track_age=1)
    for frame, x in enumerate((20.0, 25.0, 30.0, 35.0)):
        points = np.full((1, 133, 2), 20.0)
        points[..., 0] = x
        scores = np.full((1, 133), 0.9)
        if frame == 2:
            scores[:] = 0
            scores[:, 13:23] = 0.9
        outputs = []
        for lower, smoother in ((False, baseline), (True, dropped)):
            clean = apply_hygiene(points, scores, 64, 48, drop_lower_body=lower)
            outputs.append(smoother(points, clean, frame / 30, track_ids=[7]))
        (old_xy, old_scores), (new_xy, new_scores) = outputs
        assert old_xy is not None
        assert new_xy is not None
        assert old_scores is not None
        assert new_scores is not None
        if frame < 2:
            np.testing.assert_array_equal(old_xy[:, _UPPER], new_xy[:, _UPPER])
            np.testing.assert_array_equal(old_scores[:, _UPPER], new_scores[:, _UPPER])
        else:
            assert np.all(np.any(old_xy[:, _UPPER] != new_xy[:, _UPPER], axis=-1))
            if frame == 2:
                assert not np.any(old_scores[:, _UPPER])
                assert not np.any(new_scores[:, _UPPER])
            else:
                assert np.all(old_scores[:, _UPPER] > 0)
                assert np.all(new_scores[:, _UPPER] > 0)
                assert np.all(old_scores[:, _UPPER] != new_scores[:, _UPPER])


@pytest.mark.parametrize("name", ["corpus_run_2d.py", "pilot_corpus_run.py"])
@pytest.mark.parametrize("token", ["above", "-above"], ids=["positive-control", "leading-hyphen"])
def test_r01_driver_preserves_an_accepted_camera_token(name, token, tmp_path, monkeypatch):
    driver = runpy.run_path(str(_ROOT / "scripts" / name), run_name="_rev4_driver")[
        "main"
    ].__globals__
    monkeypatch.setattr(
        sys,
        "argv",
        [
            name,
            "--sessions",
            str(tmp_path / "synthetic-sessions"),
            "--out",
            str(tmp_path / "synthetic-out"),
            f"--drop-hips-camera={token}",
        ],
    )
    args = driver["_parse_args"]()
    assert args.drop_hips_camera == token
    commands = []
    monkeypatch.setattr(
        driver["subprocess"], "call", lambda command, **_kwargs: commands.append(command) or 0
    )
    if name == "corpus_run_2d.py":
        driver["_attempt_event"]("synthetic-event", args, tmp_path / "logs")
    else:
        driver["_run_event"](
            args.sessions / "synthetic-event", args.out, tmp_path / "run.log", args
        )
    commands = [command for command in commands if "pose_estimation.run" in command]
    assert len(commands) == 1
    command = commands[0]
    forwarded = run.parse_args(command[command.index("pose_estimation.run") + 1 :])
    assert forwarded.drop_hips_camera == token

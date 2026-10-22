"""Closing delta round B (`.agent/review.md` F55-F57): ambient Decimal policy + fallback device."""

from __future__ import annotations

import sys
from decimal import ROUND_UP, Decimal, Inexact, localcontext
from types import ModuleType
from typing import Any

import pytest

from pose_estimation import cohort, corpus_run, run, subject_tracker
from pose_estimation.bench import replay
from test_cohort import _read_csv
from test_m4u7_cohort_provenance import (
    _copy_output,
    _csv,
    _resign,
    _run,
    seed,  # noqa: F401 — module-scoped fixture
)


def test_f55_sensitivity_ignores_the_callers_decimal_policy(seed, tmp_path):  # noqa: F811
    baseline = _copy_output(seed, tmp_path / "baseline")
    header, rows = _read_csv(baseline / cohort.FEATURES_FILENAME)
    target = next(row for row in rows if row["median"])
    current = Decimal(target["median"])
    target.update(median=str(current - 1), q25="0.000000000", q75="3.000000000")
    _csv(baseline / cohort.FEATURES_FILENAME, header, rows)
    _resign(cohort, baseline)
    with localcontext() as ambient:
        ambient.rounding = ROUND_UP
        ambient.traps[Inexact] = True
        _run(cohort, seed.inputs, tmp_path / "out", baseline)
    _, published = _read_csv(tmp_path / "out" / cohort.SENSITIVITY_FILENAME)
    row = next(
        r for r in published if all(r[k] == target[k] for k in ("task", "side", "level", "feature"))
    )
    assert row["delta_iqr_scaled"] == "0.333333333"


class _StopSetupError(Exception):
    pass


def _resolved(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, Any]:
    """run.main up to tracker construction: the pose checkpoint + device it would load."""
    seen: dict[str, Any] = {}

    def capture(solution: Any, **_kwargs: Any) -> None:
        seen.update(solution.keywords)
        raise _StopSetupError

    def no_model(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("resolution must stop before any model loads")

    fake = ModuleType("rtmlib")
    fake.__dict__.update(PoseTracker=capture, draw_skeleton=lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "rtmlib", fake)
    monkeypatch.setattr(subject_tracker, "SubjectTracker", capture)
    monkeypatch.setattr(run, "SplitDeviceSolution", no_model)
    monkeypatch.setattr(run, "_patch_rtmlib_openvino", lambda: None)
    with pytest.raises(_StopSetupError):
        run.main(argv)
    return seen


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "rtmpose-m", "--tracking", "hands-arms"],
        ["--body-only", "--tracking", "hands-arms"],
    ],
    ids=["17-kp-model", "body-only"],
)
def test_f56_wholebody_fallback_loads_on_its_own_device(monkeypatch, argv):
    seen = _resolved(monkeypatch, argv)
    adopted = run.MODEL_REGISTRY[run.DEFAULT_MODEL]
    assert (seen["pose"], seen["pose_device"]) == (adopted["pose"], adopted["pose_device"])


@pytest.mark.parametrize(
    ("argv", "device"),
    [
        ([], run.MODEL_REGISTRY[run.DEFAULT_MODEL]["pose_device"]),
        (["--model", "rtmpose-m", "--tracking", "body"], "NPU"),
        (["--model", "rtmpose-m", "--tracking", "hands-arms", "--pose-device", "CPU"], "CPU"),
    ],
    ids=["default", "17-kp-body", "explicit-device"],
)
def test_f56_device_controls(monkeypatch, argv, device):
    assert _resolved(monkeypatch, argv)["pose_device"] == device


def test_f57_one_default_model_source():
    assert run.DEFAULT_MODEL is corpus_run.DEFAULT_MODEL
    assert f"--model={corpus_run.DEFAULT_MODEL}" in replay.CORPUS_ARGS

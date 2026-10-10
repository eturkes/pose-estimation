"""M2.10.1 reviewer witnesses over shipped gate and pipeline symbols."""

from __future__ import annotations

import inspect
import textwrap

import numpy as np
import pytest

import pose_estimation.run as run
import test_m2u101_hand_gate as contract_tests
from pose_estimation.keypoint_hygiene import HandPresenceGate
from pose_estimation.rtmlib_smoothing import KeypointSmoother
from test_m2u101_hand_gate import _assert_absent, _pipeline, _points, _scores


def test_b03_prune_on_real_smoother_none_output(tmp_path, monkeypatch):
    """B03: an expired real smoother key must leave gate state, including on None output."""

    class ObservedGate(HandPresenceGate):
        def __call__(self, scores, keys):
            if len(keys):
                self.last_seen_keys = list(keys)
            return super().__call__(scores, keys)

    gate = ObservedGate()
    smoother = KeypointSmoother(min_track_age=1, carry_frames=0)
    rows, diagnostics, _ = _pipeline(
        tmp_path,
        monkeypatch,
        [(_points(), _scores()), (_points(0), _scores(rows=0))],
        gate=gate,
        smoother=smoother,
        keys=[[7], []],
    )
    assert len(rows) == 1
    assert diagnostics["hand_frames_present"] == "2"
    assert smoother.live_track_keys() == smoother.output_track_keys() == []
    assert len(gate.last_seen_keys) == 1
    # Returning expired keys start off; 0.55 is below the admission threshold.
    actual = gate(_scores(0.55, 0.55), gate.last_seen_keys)
    np.testing.assert_array_equal(actual[:, 91:], 0)


def test_b03_prune_on_tracker_rank_one_empty_output(tmp_path, monkeypatch):
    """B03: no-smoother row keys expire on rtmlib's rank-one empty score arrays."""
    gate = HandPresenceGate()
    rows, diagnostics, _ = _pipeline(
        tmp_path,
        monkeypatch,
        [
            (_points(), _scores()),
            (np.array([]), np.array([])),
            (_points(), _scores(0.55, 0.55)),
        ],
        gate=gate,
        tracker_ids=False,
    )
    assert len(rows) == 2
    assert diagnostics["n_frames_decoded"] == "3"
    _assert_absent(rows[-1], "left")
    _assert_absent(rows[-1], "right")


def test_b06_interrupted_source_writes_gate_counters(tmp_path, monkeypatch):
    """B06: finally writes gate counters for the decoded prefix on interruption."""
    import csv

    original = contract_tests._Tracker.__call__

    def interrupt_second(self, frame):
        if self.index == 1:
            raise KeyboardInterrupt
        return original(self, frame)

    monkeypatch.setattr(contract_tests._Tracker, "__call__", interrupt_second)
    gate = HandPresenceGate()
    with pytest.raises(KeyboardInterrupt):
        _pipeline(
            tmp_path,
            monkeypatch,
            [(_points(), _scores(0.6, 0.8)), (_points(), _scores())],
            gate=gate,
        )
    with (tmp_path / "diagnostics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert (rows[0]["hand_frames_present"], rows[0]["hand_frames_gated"]) == ("2", "1")


def _replace_function(function, replacements):
    source = textwrap.dedent(inspect.getsource(function))
    for before, after in replacements:
        assert source.count(before) == 1
        source = source.replace(before, after)
    namespace = dict(function.__globals__)
    exec(compile(source, function.__code__.co_filename, "exec"), namespace)
    compiled = namespace[function.__name__]
    return type(function)(
        compiled.__code__, function.__globals__, function.__name__, compiled.__defaults__
    )


def test_b12_nc1_inclusive_threshold_mutant_fires(monkeypatch):
    """B12: the shipped P01 witness rejects > substituted for >= at admission."""
    mutant = _replace_function(
        HandPresenceGate.__call__,
        [
            (
                "level >= (self.off if states[hand] else self.on)",
                "level > (self.off if states[hand] else self.on)",
            )
        ],
    )
    monkeypatch.setattr(HandPresenceGate, "__call__", mutant)
    with pytest.raises(AssertionError):
        contract_tests.test_p01_nc1_exact_on_boundary_is_inclusive()


def test_b12_nc2_pre_smoother_mutant_fires(tmp_path, monkeypatch):
    """B12: the shipped P05 witness rejects gating before real score smoothing."""
    # Move only score gating; keep the real key-lifetime pruning in place.
    # This fixture keeps one tracker key: only score-stage ordering moves its output.
    source = inspect.getsource(run.process_source)
    calls = [
        line for line in source.splitlines(True) if "scores = hand_gate(scores, gate_keys)" in line
    ]
    assert len(calls) == 1
    late_gate = calls[0]
    before_smoother = "            if smoother is not None:\n                track_ids ="
    early_gate = (
        "            if hand_gate is not None and scores is not None and scores.ndim == 2:\n"
        "                gate_keys = list(pose_tracker.last_track_ids)\n"
        "                hand_gate.prune(gate_keys)\n"
        "                scores = hand_gate(scores, gate_keys)\n\n"
    )
    mutant = _replace_function(
        run.process_source,
        [
            (late_gate, late_gate.replace("scores = hand_gate(scores, gate_keys)", "pass")),
            (before_smoother, early_gate + before_smoother),
        ],
    )
    monkeypatch.setattr(run, "process_source", mutant)
    with pytest.raises(AssertionError, match="pre-smoother zeroing contaminates EMA scores"):
        contract_tests.test_p05_nc2_gate_after_real_smoother_preserves_admitted_score(
            tmp_path, monkeypatch
        )


def test_b12_nc3_row_index_mutant_fires(tmp_path, monkeypatch):
    """B12: the shipped P05 swap witness rejects row-position gate identity."""
    mutant = _replace_function(
        run.process_source,
        [
            (
                "gate_keys = list(smoother.output_track_keys())",
                "gate_keys = list(range(scores.shape[0]))",
            ),
            ("live_keys = list(smoother.live_track_keys())", "live_keys = gate_keys"),
        ],
    )
    monkeypatch.setattr(run, "process_source", mutant)
    with pytest.raises(AssertionError):
        contract_tests.test_p05_nc3_pipeline_key_source_survives_swapped_rows(
            tmp_path, monkeypatch, "smoother"
        )


@pytest.mark.parametrize("driver_name", ["corpus_run_2d.py", "pilot_corpus_run.py"])
def test_b12_nc4_disabled_driver_default_mutant_fires(monkeypatch, driver_name):
    """B12: the shipped P06 witness rejects default-off in each real driver parser."""
    driver = contract_tests._driver(driver_name)
    driver["_parse_args"] = _replace_function(
        driver["_parse_args"],
        [
            (
                '"--hand-gate", action=argparse.BooleanOptionalAction, default=True',
                '"--hand-gate", action=argparse.BooleanOptionalAction, default=False',
            ),
        ],
    )
    monkeypatch.setattr(contract_tests, "_driver", lambda _name: driver)
    with pytest.raises(AssertionError, match="driver must default hand gate on"):
        contract_tests.test_p06_nc4_driver_defaults_on_and_declares_generation_identity(
            driver_name, monkeypatch
        )

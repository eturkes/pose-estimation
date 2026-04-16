"""Tests for benchmark sweep-config loading and validation."""

from __future__ import annotations

import argparse
import json

import pytest

from pose_estimation.benchmark import TUNEABLE_PARAMS, _load_sweep_config


@pytest.fixture
def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    return p


def _write_yaml(path, body):
    """Try YAML; fall back to JSON since both are accepted."""
    try:
        import yaml

        path.write_text(yaml.safe_dump(body))
    except ImportError:
        path.write_text(json.dumps(body))


def test_loads_known_keys(tmp_path, parser):
    cfg = tmp_path / "sweep.yaml"
    _write_yaml(cfg, {"body_min_cutoff": [0.1, 0.3, 0.5]})
    spec = _load_sweep_config(cfg, parser)
    assert spec == {"body_min_cutoff": [0.1, 0.3, 0.5]}


def test_coerces_int_param(tmp_path, parser):
    cfg = tmp_path / "sweep.yaml"
    _write_yaml(cfg, {"carry_grace": [5, 10, 15]})
    spec = _load_sweep_config(cfg, parser)
    assert spec["carry_grace"] == [5, 10, 15]
    assert all(isinstance(v, int) for v in spec["carry_grace"])
    assert isinstance(TUNEABLE_PARAMS["carry_grace"], int)


def test_scalar_value_wraps_to_list(tmp_path, parser):
    cfg = tmp_path / "sweep.yaml"
    _write_yaml(cfg, {"body_min_cutoff": 0.5})
    spec = _load_sweep_config(cfg, parser)
    assert spec == {"body_min_cutoff": [0.5]}


def test_unknown_key_warns_and_drops(tmp_path, parser, capsys):
    cfg = tmp_path / "sweep.yaml"
    _write_yaml(cfg, {"body_min_cutoff": [0.1], "totally_made_up": [1, 2]})
    spec = _load_sweep_config(cfg, parser)
    assert "totally_made_up" not in spec
    assert spec == {"body_min_cutoff": [0.1]}
    captured = capsys.readouterr()
    assert "totally_made_up" in captured.out
    assert "unknown parameter" in captured.out.lower()


def test_missing_file_errors(tmp_path, parser):
    bogus = tmp_path / "does-not-exist.yaml"
    with pytest.raises(SystemExit):
        _load_sweep_config(bogus, parser)


def test_top_level_must_be_mapping(tmp_path, parser):
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(SystemExit):
        _load_sweep_config(cfg, parser)


def test_bad_coercion_errors(tmp_path, parser):
    cfg = tmp_path / "sweep.yaml"
    # body_min_cutoff is a float — pass an unparsable string
    _write_yaml(cfg, {"body_min_cutoff": ["not-a-number"]})
    with pytest.raises(SystemExit):
        _load_sweep_config(cfg, parser)

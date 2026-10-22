"""Every default names the adopted pose model (user ruling: RTMW-L 384 → `output/corpus-2d`).

A run records `--model` verbatim, so a default left on the `rtmw-l` alias reads the adopted tree
as another configuration and refuses `--analyse-only` / `--reuse-run` over it (`corpus-run.md`).
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from typing import Any

import pytest

from pose_estimation import run
from pose_estimation.bench import replay

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADOPTED = "rtmw-l-384"  # `.agent/spec.md` Decisions: shipped pose model


def _script(name: str) -> dict[str, Any]:
    return runpy.run_path(str(ROOT / "scripts" / name), run_name="_defaults")


def test_run_default_is_the_adopted_arm() -> None:
    assert run.DEFAULT_MODEL == ADOPTED
    assert run.MODEL_REGISTRY["rtmw-l"] is run.MODEL_REGISTRY["rtmw-m-256"]


def test_corpus_driver_defaults_reproduce_the_adopted_configuration() -> None:
    args = _script("corpus_run_2d.py")["_parse_args"]([])
    assert args.model == ADOPTED
    assert args.pose_device == run.MODEL_REGISTRY[ADOPTED]["pose_device"]


def test_pilot_defaults_name_the_adopted_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pilot_corpus_run.py"])
    assert _script("pilot_corpus_run.py")["_parse_args"]().model == ADOPTED


def test_replay_corpus_args_name_the_adopted_model() -> None:
    models = [arg for arg in replay.CORPUS_ARGS if arg.startswith("--model=")]
    assert models == [f"--model={ADOPTED}"]


def test_run_report_defaults_describe_the_adopted_run(tmp_path: pathlib.Path) -> None:
    main = _script("run_report.py")["main"]
    seen: dict[str, Any] = {}

    def capture(args: Any) -> dict[str, Any]:
        seen.update(vars(args))
        raise SystemExit(0)

    main.__globals__["build_manifest"] = capture
    with pytest.raises(SystemExit):
        main([str(tmp_path)])
    adopted = run.MODEL_REGISTRY[ADOPTED]["pose_device"]
    assert (seen["model"], seen["pose_device"], seen["det_device"], seen["det_frequency"]) == (
        ADOPTED,
        adopted,
        "GPU",
        1,
    )

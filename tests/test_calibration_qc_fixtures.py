"""M2.7.2 P10: the committed fixture set is graded inside the decisive suite.

``scripts/check_calibration_qc_fixtures.py`` is the single implementation behind
both this module and the standalone command, so a fixture can never pass one and
fail the other.  One case per predicate, because a failure names the predicate
that broke rather than a boolean over twelve of them.
"""

from __future__ import annotations

import pathlib
import runpy
from collections.abc import Callable

import pytest

_CHECKER = runpy.run_path(
    str(pathlib.Path(__file__).resolve().parents[1] / "scripts/check_calibration_qc_fixtures.py")
)
_PREDICATES: dict[str, Callable[[], tuple[bool, str]]] = dict(_CHECKER["PREDICATES"])


@pytest.mark.parametrize("predicate", sorted(_PREDICATES))
def test_fixture_predicate_is_green(predicate: str) -> None:
    ok, detail = _PREDICATES[predicate]()
    assert ok, f"{predicate} {detail}"

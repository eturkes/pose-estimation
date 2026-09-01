"""M2.7.4 P12: the prospective capture specification is graded inside the decisive suite.

``scripts/check_prospective_capture.py`` is the single implementation behind both this
module and the standalone command, so the document can never pass one and fail the
other.  One case per predicate, because a failure names the predicate that broke rather
than a boolean over thirteen of them.
"""

from __future__ import annotations

import pathlib
import runpy
from collections.abc import Callable

import pytest

_CHECKER = runpy.run_path(
    str(pathlib.Path(__file__).resolve().parents[1] / "scripts/check_prospective_capture.py")
)
_PREDICATES: dict[str, Callable[[], tuple[bool, str]]] = dict(_CHECKER["PREDICATES"])


@pytest.mark.parametrize("predicate", sorted(_PREDICATES))
def test_prospective_capture_predicate_is_green(predicate: str) -> None:
    ok, detail = _PREDICATES[predicate]()
    assert ok, f"{predicate} {detail}"

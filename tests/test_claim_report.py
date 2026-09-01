"""M2.7.3 P09: the claim-bounded negative report is graded inside the decisive suite.

``scripts/check_claim_report.py`` is the single implementation behind both this module
and the standalone command, so a document can never pass one and fail the other.  One
case per predicate, because a failure names the predicate that broke rather than a
boolean over nine of them.
"""

from __future__ import annotations

import pathlib
import runpy
from collections.abc import Callable

import pytest

_CHECKER = runpy.run_path(
    str(pathlib.Path(__file__).resolve().parents[1] / "scripts/check_claim_report.py")
)
_PREDICATES: dict[str, Callable[[], tuple[bool, str]]] = dict(_CHECKER["PREDICATES"])


@pytest.mark.parametrize("predicate", sorted(_PREDICATES))
def test_claim_report_predicate_is_green(predicate: str) -> None:
    ok, detail = _PREDICATES[predicate]()
    assert ok, f"{predicate} {detail}"

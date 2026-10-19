"""A32 in-session reconciliation — the decisive gate's "green" (pytest plugin, `tests/conftest.py`).

Active only under `POSE_GATE_FULL=1`, which `scripts/gate.sh` sets on its full pytest step. The
run must be the whole collection (no deselection, no narrowed paths), every collected case must
pass, and nothing may be skipped, xfailed, xpassed or errored — a bare pass floor lets a case
vanish silently, so the gate reconciles passed against collected in the same session. This
replaced a whole-suite replay inside the suite (test_c8_08's former body), which ran every case
twice and timed out under co-tenant load.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

ENV = "POSE_GATE_FULL"


class Reconciler:
    def __init__(self, config: pytest.Config):
        self.config = config
        self.collected = 0
        self.deselected = 0
        self.outcomes: Counter[str] = Counter()

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_collectreport(self, report):
        # A module skipped or broken at collection (`importorskip`, `allow_module_level`) never
        # reaches runtest: it drops out of `collected` too, so only this report shows it.
        if report.skipped:
            self.outcomes["skipped"] += 1
        elif report.failed:
            self.outcomes["error"] += 1

    def pytest_runtest_logreport(self, report):
        if getattr(report, "wasxfail", None) is not None:
            self.outcomes["xpassed" if report.passed else "xfailed"] += 1
        elif report.skipped:
            self.outcomes["skipped"] += 1
        elif report.failed:
            self.outcomes["failed" if report.when == "call" else "error"] += 1
        elif report.when == "call":
            self.outcomes["passed"] += 1

    def problems(self) -> list[str]:
        out = []
        testpaths = [str(Path(self.config.rootpath, p)) for p in self.config.getini("testpaths")]
        if sorted(str(Path(a).resolve()) for a in self.config.args) != sorted(testpaths):
            out.append(f"the full gate runs the whole collection, got paths {self.config.args}")
        if self.deselected:
            out.append(f"{self.deselected} cases deselected")
        if self.collected == 0:
            out.append("no case collected")
        out.extend(
            f"{self.outcomes[kind]} {kind}"
            for kind in ("failed", "error", "skipped", "xfailed", "xpassed")
            if self.outcomes[kind]
        )
        if self.outcomes["passed"] != self.collected:
            out.append(f"{self.outcomes['passed']} passed of {self.collected} collected")
        return out

    def pytest_sessionfinish(self, session, exitstatus):
        problems = self.problems()
        if problems:
            reporter = self.config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                reporter.write_line(f"A32 FAIL: {'; '.join(problems)}", red=True)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config):
    if os.environ.get(ENV) == "1":
        config.pluginmanager.register(Reconciler(config), "a32-reconciler")

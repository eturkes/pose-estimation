"""The inventory + measure mutation campaigns: catalogues anchored, results bound, survivors fail."""

from __future__ import annotations

import hashlib
import json
import pathlib
import runpy

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CAMPAIGNS = {
    "inventory": ("run_inventory_mutations.py", "inventory_mutation_results.json"),
    "measure": ("run_measure_mutations.py", "measure_mutation_results.json"),
}


def _campaign(name: str) -> dict:
    return runpy.run_path(str(_ROOT / "scripts" / _CAMPAIGNS[name][0]), run_name=f"_{name}_mut")


def _recorded(name: str) -> dict:
    return json.loads((_ROOT / "tests" / _CAMPAIGNS[name][1]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(_CAMPAIGNS))
def test_catalogue_and_seed_anchor_exactly_once_in_current_source(name: str) -> None:
    campaign = _campaign(name)
    campaign["validate_catalog"](campaign["baseline_files"]())


@pytest.mark.parametrize("name", sorted(_CAMPAIGNS))
def test_committed_results_target_the_current_source_bytes(name: str) -> None:
    campaign = _campaign(name)
    current = {
        path: hashlib.sha256(raw).hexdigest() for path, raw in campaign["baseline_files"]().items()
    }
    assert _recorded(name)["target_sha256"] == current


@pytest.mark.parametrize("name", sorted(_CAMPAIGNS))
def test_committed_results_cover_the_catalogue_and_pass(name: str) -> None:
    campaign = _campaign(name)
    recorded = _recorded(name)
    rows = recorded.get("results", recorded.get("initial"))
    assert sorted(row["id"] for row in rows) == sorted(item.id for item in campaign["MUTANTS"])
    assert campaign["campaign_status"](rows) == 0


def test_m028_is_the_sole_ruled_inventory_equivalent() -> None:
    campaign = _campaign("inventory")
    assert [item.id for item in campaign["MUTANTS"] if item.equivalent_reason] == ["M028"]
    assert not _campaign("inventory")["SEED_SURVIVOR"].equivalent_reason


@pytest.mark.parametrize("name", sorted(_CAMPAIGNS))
def test_status_fails_an_unruled_survivor_and_a_killed_equivalent(name: str) -> None:
    status = _campaign(name)["campaign_status"]

    def row(verdict: str, reason: str | None = None) -> dict:
        return {"verdict": verdict, "equivalent_reason": reason}

    assert status([]) == 0
    assert status([row("killed"), row("equivalent", "ruled")]) == 0
    assert status([row("killed"), row("SURVIVED")]) == 1
    assert status([row("killed", "ruled")]) == 1

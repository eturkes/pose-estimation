"""M4.7 closing delta round A (`.agent/review.md` F47-F49, S17): cohort v3 at its contract's edges."""

from __future__ import annotations

import importlib.util
import pathlib
from decimal import Decimal, localcontext

import pytest

from pose_estimation import cohort
from test_cohort import _marker, _read_csv
from test_m4u7_cohort_provenance import (
    _MEASUREMENT_CASES,
    _assert_refused,
    _copy_inputs,
    _copy_output,
    _csv,
    _damage_measurement,
    _events,
    _resign,
    _run,
    seed,  # noqa: F401 — module-scoped fixture shared with the M4.7 suite
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
_KEY = ("task", "side", "level", "feature")


def _sensitivity_row(out: pathlib.Path, target: dict[str, str]) -> dict[str, str]:
    _, rows = _read_csv(out / cohort.SENSITIVITY_FILENAME)
    return next(row for row in rows if all(row[k] == target[k] for k in _KEY))


# Baseline (median, q25, q75) as serialized cells; None keeps the cell, a leading `-` derives
# the median as the current median minus that delta. Each one loses exactness through binary
# floats (contract-m4u7 P07).
_ARITHMETIC: dict[str, tuple[str | None, str | None, str | None]] = {
    "nano-iqr": ("-0.000000001", "1.000000000", "1.000000002"),
    "half-iqr-at-three": ("2.999999998", "2.999999996", "3.000000000"),
    "iqr-past-float-precision": (None, "10000000000000000", "10000000000000001"),
    "median-past-float-range": ("1" + "0" * 309, None, None),
}


@pytest.mark.parametrize("case", sorted(_ARITHMETIC))
def test_f47_sensitivity_arithmetic_is_exact_on_serialized_cells(seed, tmp_path, case):  # noqa: F811
    baseline = _copy_output(seed, tmp_path / "baseline")
    header, rows = _read_csv(baseline / cohort.FEATURES_FILENAME)
    target = next(row for row in rows if row["median"] and row["q25"] and row["q75"])
    current = Decimal(target["median"])
    median, q25, q75 = _ARITHMETIC[case]
    if median and median.startswith("-"):
        median = str(current + Decimal(median))
    target["median"] = median or target["median"]
    target["q25"], target["q75"] = q25 or target["q25"], q75 or target["q75"]
    _csv(baseline / cohort.FEATURES_FILENAME, header, rows)
    _resign(cohort, baseline)
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, baseline)
    cohort.validate_generation(out)
    row = _sensitivity_row(out, target)
    with localcontext() as context:
        context.prec = 400
        iqr = Decimal(target["q75"]) - Decimal(target["q25"])
        delta = current - Decimal(target["median"])
        expected = {
            "baseline_median": Decimal(target["median"]),
            "baseline_iqr": iqr,
            "delta_median": delta,
            "delta_iqr_scaled": delta / iqr if iqr > 0 else None,
        }
        rendered = {k: "" if v is None else f"{v:.{cohort.DECIMALS}f}" for k, v in expected.items()}
    assert {k: row[k] for k in rendered} == rendered


@pytest.mark.parametrize("version", [[], {}, 3, None], ids=["array", "object", "int", "null"])
def test_f48_non_string_baseline_version_refuses_as_cohort_error(seed, tmp_path, version):  # noqa: F811
    baseline = _copy_output(seed, tmp_path / "baseline")
    marker = _marker(baseline)
    marker["generation"]["generator_version"] = version
    (baseline / cohort.COHORT_FILENAME).write_bytes(cohort.render_marker(marker))
    out = _copy_output(seed, tmp_path / "out")
    _assert_refused(cohort, seed.inputs, out, baseline)


def _campaign():
    spec = importlib.util.spec_from_file_location(
        "cohort_campaign", ROOT / "scripts" / "check_cohort_determinism.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("group", ["_sweeps", "_baseline_sweeps"])
def test_f49_each_campaign_group_reorders_clinical_rows(tmp_path, monkeypatch, group):
    campaign = _campaign()
    publish = campaign._publish
    tables: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def observe(inputs, out, baseline=None):
        ordered = tuple(
            path.read_text(encoding="utf-8")
            for path in sorted(inputs.run.glob("*/*_clinical*.csv"))
        )
        tables.append((ordered, tuple(sorted(line for t in ordered for line in t.splitlines()))))
        return publish(inputs, out, baseline)

    monkeypatch.setattr(campaign, "_publish", observe)
    results = getattr(campaign, group)(tmp_path / "campaign")
    observed = tables[-len(results) :]
    assert len({content for _, content in observed}) == 1, "a reorder must keep row contents"
    assert len({ordered for ordered, _ in observed}) > 1, f"{group} publishes one row order alone"
    assert len({tuple(sorted(digests.items())) for _, digests in results}) == 1


@pytest.mark.parametrize("case", _MEASUREMENT_CASES)
def test_s17_measurement_refusals_name_no_event_or_path(seed, tmp_path, case):  # noqa: F811
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    _damage_measurement(inputs, case)
    with pytest.raises(cohort.CohortError) as refusal:
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "out")
    message = str(refusal.value)
    assert str(tmp_path) not in message
    assert not [event.name for event in _events(inputs) if event.name in message]

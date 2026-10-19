"""M4.5b P01-P07: cross-view agreement estimator, bout reader, exclusions, floor + reasons."""

from __future__ import annotations

import csv
import pathlib

import numpy as np
import pytest

from pose_estimation import cohort, corpus_run, inventory, sessions, view_agreement
from test_cohort import (
    _AGREEMENT_FIELDS,
    _build_inputs,
    _feature_output,
    _marker,
    _publish,
    _read_csv,
)

va = view_agreement


# P01 — estimator


def test_p01_unbalanced_hand_fixture():
    assert va.icc_oneway([[0, 2], [3, 4, 8], [5, 9]]) == pytest.approx(15 / 31, abs=1e-12)


def test_p01_balanced_equals_shrout_fleiss_icc11():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(9, 3)) + rng.normal(size=(9, 1)) * 2
    n, k = y.shape
    msb = k * ((y.mean(axis=1) - y.mean()) ** 2).sum() / (n - 1)
    msw = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    assert va.icc_oneway(y.tolist()) == pytest.approx((msb - msw) / (msb + (k - 1) * msw))


def test_p01_identical_views_one_constant_undefined_negative_unclipped():
    assert va.icc_oneway([[1, 1], [2, 2, 2], [5, 5]]) == pytest.approx(1.0)
    assert va.icc_oneway([[3, 3], [3, 3, 3]]) is None  # no variance anywhere: undefined, never 1
    negative = va.icc_oneway([[0, 10], [10, 0], [5, 5]])
    assert negative is not None
    assert negative < 0


def test_p01_nc2_mean_view_count_is_not_k0():
    groups = [[0, 2], [3, 4, 8], [5, 9]]
    k_bar = 7 / 3
    y = [np.asarray(g, float) for g in groups]
    grand = np.concatenate(y).mean()
    msb = sum(len(v) * (v.mean() - grand) ** 2 for v in y) / 2
    msw = sum(((v - v.mean()) ** 2).sum() for v in y) / 4
    naive = (msb - msw) / (msb + (k_bar - 1) * msw)
    assert va.icc_oneway(groups) != pytest.approx(naive, abs=1e-6)  # k0 = 16/7, not 7/3


# P02 — subject-cluster resampling


def test_p02_exhaustive_draws_of_two_one_event_subjects():
    events = {"a": [0, 0], "b": [0, 2]}
    owners = {"a": "s1", "b": "s2"}
    by_counts = {
        (2, 0): va.icc_oneway([[0, 0], [0, 2]], [2, 0]),
        (1, 1): va.icc_oneway([[0, 0], [0, 2]], [1, 1]),
        (0, 2): va.icc_oneway([[0, 0], [0, 2]], [0, 2]),
    }
    assert by_counts == {(2, 0): None, (1, 1): 0.0, (0, 2): -1.0}
    draws = va.resample_icc(events, owners, seed=9, n=4000)
    finite = [d for d in draws if d is not None]
    assert len(finite) / len(draws) == pytest.approx(0.75, abs=0.03)
    assert set(np.round(finite, 12)) == {0.0, -1.0}
    result = va.agreement(events, owners, "key", floor=1)
    assert result.reasons == ("agreement_insufficient_resamples",)
    assert result.range_lo is None
    assert result.icc is not None


def test_p02_whole_cluster_extremes_and_seeded_order_invariance():
    events = {"a1": [1, 1.1], "a2": [5, 5.1], "b1": [1, 4], "b2": [5, 2]}
    owners = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    draws = va.resample_icc(events, owners, seed=4, n=400)
    # {A, A} and {B, B}: each event of the drawn subject twice (copies are separate targets).
    only_a = va.icc_oneway([events["a1"], events["a2"]], [2, 2])
    only_b = va.icc_oneway([events["b1"], events["b2"]], [2, 2])
    finite = [d for d in draws if d is not None]
    assert min(finite) == pytest.approx(min(only_a, only_b))  # {B, B}
    assert max(finite) == pytest.approx(max(only_a, only_b))  # {A, A}
    reordered = dict(reversed(list(events.items())))
    assert va.resample_icc(reordered, owners, seed=4, n=400) == draws
    assert va.resample_icc(events, owners, seed=4, n=400) == draws


def test_p02_vectorised_resampling_equals_the_scalar_estimator():
    rng = np.random.default_rng(1)
    events = {f"e{i}": rng.normal(size=int(rng.integers(2, 4))).tolist() for i in range(10)}
    owners = {e: f"s{int(e[1:]) // 2}" for e in events}
    seed, n = 123, 50
    draws = va.resample_icc(events, owners, seed, n)
    names = sorted(events)
    subjects = sorted(set(owners.values()))
    gen = np.random.default_rng(seed)
    for got in draws:
        counts = np.bincount(gen.integers(0, len(subjects), len(subjects)), minlength=len(subjects))
        weights = [counts[subjects.index(owners[e])] for e in names]
        want = va.icc_oneway([events[e] for e in names], weights)
        assert (got is None) == (want is None)
        if want is not None:
            assert got == pytest.approx(want, abs=1e-10)


def test_p02_nc1_pooled_resampling_differs():
    events = {"a1": [1, 1.1], "a2": [5, 5.1], "b1": [1, 4], "b2": [5, 2]}
    owners = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    cluster = va.resample_icc(events, owners, seed=4, n=400)
    pooled = va.resample_icc(events, {e: e for e in events}, seed=4, n=400)
    assert cluster != pooled  # event-level draws break the subject clusters


# P06/P07 — floor, reasons, absolute difference


def test_p07_pairwise_absolute_difference():
    assert va.mean_pair_abs_diff([1, 4]) == 3
    assert va.mean_pair_abs_diff([1, 4, 2]) == pytest.approx((3 + 1 + 2) / 3)
    assert va.mean_pair_abs_diff([7, 7, 7]) == 0


def test_p06_floor_and_reason_partition():
    owners = {f"e{i}": f"s{i}" for i in range(5)}
    full = {f"e{i}": [i, i + 0.5 * (i % 2)] for i in range(5)}
    ok = va.agreement(full, owners, "k", floor=5)
    assert ok.reasons == ()
    assert ok.n_subjects == 5
    assert ok.abs_diff == pytest.approx(0.0)  # 3 of 5 events agree exactly → median 0
    assert ok.range_lo is not None
    four = va.agreement({e: v for e, v in full.items() if e != "e4"}, owners, "k", floor=5)
    assert four.reasons == ("agreement_below_floor",)
    assert (four.abs_diff, four.icc, four.range_lo, four.range_hi) == (None, None, None, None)
    single = va.agreement({e: [v[0]] for e, v in full.items()}, owners, "k", floor=5)
    assert single.reasons == ("agreement_no_multiview",)
    same = va.agreement({e: [2.0, 2.0] for e in full}, owners, "k", floor=5)
    assert same.reasons == ("agreement_zero_variance",)
    assert same.abs_diff == 0.0  # finite where the ICC is undefined
    twice = va.agreement(
        full | {"e5": [1.0, 9.0]}, owners | {"e5": "s1"}, "k", floor=5
    )  # one subject, two events
    assert twice.n_events == 6
    assert twice.n_subjects == 5


# P03/P04 — publisher-level bout reader + exclusions


def _inputs(tmp_path: pathlib.Path, **options):
    root = tmp_path / "in"
    root.mkdir()
    return _build_inputs(root, cohort, **options)


def _bout_files(inputs) -> list[pathlib.Path]:
    return sorted(inputs.run.glob("*/*_clinical_bouts.csv"))


def _rewrite(path: pathlib.Path, edit) -> None:
    header, rows = _read_csv(path)
    header, rows = edit(list(header), rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def test_p03_censored_rows_never_contribute(tmp_path):
    inputs = _inputs(tmp_path)
    target = (inventory.TASKS[0], "l")
    out_base = tmp_path / "base"
    _publish(cohort, inputs, out_base)
    before = _feature_output(out_base, (*target, "bout", "left_bout_duration"))

    def censor_with_outlier(header, rows):
        extra = dict(rows[0]) | {"censored": "1", "left_bout_duration": "999", "bout_idx": "99"}
        return header, [*rows, extra]

    for path in _bout_files(inputs):
        _rewrite(path, censor_with_outlier)
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    after = _feature_output(out, (*target, "bout", "left_bout_duration"))
    assert after["median"] == before["median"]
    assert after["n_values"] == before["n_values"]
    cells = {(r["task"], r["side"]): r for r in _read_csv(out / "cohort_cells.csv")[1]}
    base_cells = {(r["task"], r["side"]): r for r in _read_csv(out_base / "cohort_cells.csv")[1]}
    assert int(cells[target]["n_bout_rows"]) > int(base_cells[target]["n_bout_rows"])


def test_p03_nc4_wrong_side_value_and_missing_file_refuse(tmp_path):
    inputs = _inputs(tmp_path)
    path = _bout_files(inputs)[0]

    def leak(header, rows):
        rows[0]["right_bout_duration" if rows[0]["side"] == "left" else "left_bout_duration"] = "1"
        return header, rows

    _rewrite(path, leak)
    with pytest.raises(cohort.CohortError, match="other side"):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "a")
    path.unlink()
    with pytest.raises(cohort.CohortError, match="readable bout artifact"):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "b")


def test_p03_header_only_bout_file_is_zero_bouts(tmp_path):
    inputs = _inputs(tmp_path)
    for path in _bout_files(inputs):
        _rewrite(path, lambda header, rows: (header, []))
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    excluded = {(r["level"], r["column"]): r["reason"] for r in _marker(out)["columns"]["excluded"]}
    assert all(excluded[("bout", c)] == "no_eligible_bout_values" for c in cohort.BOUT_COLUMNS)
    assert all(r["n_bout_rows"] == "0" for r in _read_csv(out / "cohort_cells.csv")[1])
    _, rows = _read_csv(out / "cohort_features.csv")
    assert not any(r["level"] == "bout" for r in rows)


def test_p04_construct_invalid_is_the_present_retired_subset(tmp_path):
    inputs = _inputs(tmp_path)
    dropped = ("left_wrist_sal", "wrist_sal_abs_diff")
    for path in sorted(inputs.run.glob("*/*_clinical_windows.csv")):
        _rewrite(
            path,
            lambda header, rows: (
                [h for h in header if h not in dropped],
                [{k: v for k, v in r.items() if k not in dropped} for r in rows],
            ),
        )
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    invalid = {
        (r["level"], r["column"])
        for r in _marker(out)["columns"]["excluded"]
        if r["reason"] == "construct_invalid"
    }
    assert invalid == cohort.CONSTRUCT_INVALID - {("window", c) for c in dropped}
    _, rows = _read_csv(out / "cohort_features.csv")
    assert not {(r["level"], r["feature"]) for r in rows} & cohort.CONSTRUCT_INVALID


def test_p04_unknown_all_na_column_refuses(tmp_path):
    inputs = _inputs(tmp_path)
    for path in sorted(inputs.run.glob("*/*_clinical.csv")):
        _rewrite(
            path,
            lambda h, rows: ([*h, "novel_feature"], [r | {"novel_feature": "NA"} for r in rows]),
        )
    with pytest.raises(cohort.CohortError, match="novel_feature"):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "out")


def test_p06_nc3_four_subject_cell_publishes_no_agreement(tmp_path):
    inputs = _inputs(tmp_path, all_multiview=True)
    target = (inventory.TASKS[0], "l")
    _, placements = _read_csv(inputs.sessions / sessions.PLACEMENTS_FILENAME)
    left_views = {
        a.asset_id
        for a in inputs.assets
        if (a.task, a.side, a.view) == (*target, "left") and a.subject_ordinal == 2
    }
    for placement in placements:
        if placement["asset_id"] in left_views:
            path = inputs.run / placement["event_id"] / f"{placement['camera_name']}_clinical.csv"
            _rewrite(path, lambda h, rows: (h, [dict.fromkeys(h, "NA") | {k: r[k] for k in ("video", "frame_idx", "timestamp_sec", "person_idx")} for r in rows]))  # fmt: skip
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    frame_feature = next(f.column for f in cohort.FEATURES if f.level == "frame")
    row = _feature_output(out, (*target, "frame", frame_feature))
    assert row["n_subjects_multiview"] == "4"
    assert all(row[field] == "" for field in _AGREEMENT_FIELDS)
    assert _marker(out)["agreement_reasons"]["agreement_below_floor"] >= 1


def test_p07_output_is_contributor_order_invariant(tmp_path):
    inputs = _inputs(tmp_path, all_multiview=True)
    out_a = tmp_path / "a"
    _publish(cohort, inputs, out_a)
    manifest = inputs.run / corpus_run.MANIFEST_FILENAME
    header, rows = _read_csv(manifest)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(list(reversed(rows)))
    out_b = tmp_path / "b"
    _publish(cohort, inputs, out_b)
    for name in ("cohort_cells.csv", "cohort_features.csv", "descriptors.yaml"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_p05_bout_descriptors_carry_labels_units_and_ranges():
    bout = [f for f in cohort.FEATURES if f.level == "bout"]
    assert len(bout) == 14
    for feature in bout:
        assert feature.ja
        assert feature.en
        assert feature.unit in cohort.UNIT_VOCABULARY
        assert feature.raw.startswith("pose_bout_")
    units = {f.column.split("_", 1)[1]: f.unit for f in bout}
    assert units["bout_duration"] == units["bout_dwell"] == "s"
    assert units["bout_speed_peaks"] == "count"
    assert {f.range for f in bout if "sparc" in f.column} == {(None, 0.0)}


def _bout_csv(path: pathlib.Path, *, drop=(), update=None, empty=False) -> None:
    golden = (
        pathlib.Path(__file__).resolve().parent / "goldens/r_clinical/2d_idx_clinical_bouts.csv"
    )
    with golden.open(newline="") as stream:
        header = next(csv.reader(stream))
    row = dict.fromkeys(header, "NA") | {
        "video": "synthetic", "person_idx": "0", "side": "left", "bout_idx": "1",
        "start_frame": "30", "end_frame": "60", "start_sec": "1", "end_sec": "2", "censored": "0",
        "left_bout_duration": "1",
    } | (update or {})  # fmt: skip
    header = [c for c in header if c not in drop]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        if not empty:
            writer.writerow({k: row[k] for k in header})


@pytest.mark.parametrize("key", cohort.BOUT_KEYS)
def test_p03_header_only_bout_file_still_needs_every_key(tmp_path, key):
    path = tmp_path / "x_clinical_bouts.csv"
    _bout_csv(path, empty=True)
    assert cohort._read_artifact(path, "bout")[1] == 0
    _bout_csv(path, drop=(key,), empty=True)
    with pytest.raises(cohort.CohortError, match="lacks the bout keys"):
        cohort._read_artifact(path, "bout")


@pytest.mark.parametrize(
    "update",
    [
        {"start_sec": "3", "end_sec": "2"},
        {"start_frame": "70", "end_frame": "60"},
        {"start_sec": "NaN"},
        {"end_sec": "Inf"},
        {"start_frame": "1.5"},
    ],
)
def test_p03_bout_bounds_are_validated(tmp_path, update):
    path = tmp_path / "x_clinical_bouts.csv"
    _bout_csv(path)
    assert cohort._read_artifact(path, "bout")[2]["left_bout_duration"] == [1.0]
    _bout_csv(path, update=update)
    with pytest.raises(cohort.CohortError, match="invalid bounds"):
        cohort._read_artifact(path, "bout")


def test_p03_population_features_counts_published_features(tmp_path):
    inputs = _inputs(tmp_path)
    for path in _bout_files(inputs):
        _rewrite(path, lambda header, rows: (header, []))
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    marker = _marker(out)
    assert marker["population"]["features"] == len(cohort.FEATURES) - 14 == 74
    assert marker["population"]["feature_rows"] == 74 * 12


def test_p02_resampling_is_exact_at_a_large_common_offset():
    rng = np.random.default_rng(7)
    events = {f"e{i}": (1e8 + i + rng.normal(size=2)).tolist() for i in range(5)}
    owners = {e: e for e in events}
    shifted = va.resample_icc(events, owners, seed=7, n=50)
    centred = va.resample_icc({e: [v - 1e8 for v in vs] for e, vs in events.items()}, owners, 7, 50)
    assert len(shifted) == len(centred)
    for a, b in zip(shifted, centred, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert a == pytest.approx(b, abs=1e-6)

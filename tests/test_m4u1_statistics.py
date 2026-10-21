"""M4.1 P05/P06: public-GT parsing and independent statistical expectations."""

from __future__ import annotations

import json
import math
import statistics
from itertools import pairwise
from statistics import NormalDist

import numpy as np
import pytest

from test_m4u1_bench import bench, seeded


def coco_annotation(ann_id, *, crowd=0, left=True, right=True):
    return {
        "id": ann_id,
        "image_id": 7,
        "category_id": 1,
        "iscrowd": crowd,
        "bbox": [11, 23, 60, 80],
        "area": 4321,
        "num_keypoints": 17,
        "keypoints": [[10 + i, 50 - i, i % 3] for i in range(17)],
        "lefthand_valid": left,
        "righthand_valid": right,
        "lefthand_box": [30, 40, 6, 8],
        "righthand_box": [80, 60, 5, 12],
        "lefthand_kpts": [[100 + i, 200 - i, i % 3] for i in range(21)],
        "righthand_kpts": [[300 + i, 400 - i, (i + 1) % 3] for i in range(21)],
    }


def test_p05_coco_wholebody_native_order_visibility_validity_and_crowds(tmp_path):
    public = bench("public")
    annotations = [
        coco_annotation(1),
        coco_annotation(2, left=False),
        coco_annotation(3, right=False),
        coco_annotation(4, left=False, right=False),
        coco_annotation(5, crowd=1),
    ]
    another = coco_annotation(6, left=False)
    another["image_id"] = 9
    annotations.append(another)
    for ann in annotations:
        for key in ("keypoints", "lefthand_kpts", "righthand_kpts"):
            ann[key] = np.asarray(ann[key]).ravel().tolist()
    path = tmp_path / "wholebody.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": "synthetic.png", "width": 640, "height": 480},
                    {"id": 9, "file_name": "second-synthetic.png", "width": 800, "height": 600},
                ],
                "annotations": annotations,
            }
        )
    )
    files, persons = public.load_coco_wholebody(path)
    assert files[7] == "synthetic.png"
    assert files[9] == "second-synthetic.png"
    eligible = [ann for ann in annotations if not ann["iscrowd"]]
    assert len(persons) == len(eligible) > 0
    by_id = {p.ann_id: p for p in persons}
    assert set(by_id) == {ann["id"] for ann in eligible}
    for ann in eligible:
        person = by_id[ann["id"]]
        assert person.image_id == ann["image_id"]
        assert person.area == ann["area"]
        np.testing.assert_array_equal(person.box, ann["bbox"])
        assert np.asarray(person.body).shape == (17, 3)
        np.testing.assert_array_equal(person.body, np.asarray(ann["keypoints"]).reshape(17, 3))
        expected_sides = {
            side for side, prefix in [("L", "left"), ("R", "right")] if ann[f"{prefix}hand_valid"]
        }
        assert {hand.side for hand in person.hands} == expected_sides
        for hand in person.hands:
            prefix = "left" if hand.side == "L" else "right"
            assert (hand.ann_id, hand.image_id) == (person.ann_id, person.image_id)
            assert np.asarray(hand.kpts).shape == (21, 3)
            np.testing.assert_array_equal(
                hand.kpts, np.asarray(ann[f"{prefix}hand_kpts"]).reshape(21, 3)
            )
            np.testing.assert_array_equal(hand.box, ann[f"{prefix}hand_box"])
            assert hand.diagonal == pytest.approx(10 if hand.side == "L" else 13)


def test_p05_invalid_hand_stays_absent_even_with_nonzero_annotations(tmp_path):
    public = bench("public")
    ann = coco_annotation(1, left=False, right=False)
    for key in ("keypoints", "lefthand_kpts", "righthand_kpts"):
        ann[key] = np.asarray(ann[key]).ravel().tolist()
    path = tmp_path / "invalid-hands.json"
    path.write_text(
        json.dumps({"images": [{"id": 7, "file_name": "synthetic.png"}], "annotations": [ann]})
    )
    _, persons = public.load_coco_wholebody(path)
    assert len(persons) == 1
    assert not persons[0].hands


def wilson_reference(k, n):
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z * z / n
    centre = (k / n + z * z / (2 * n)) / denominator
    radius = z / denominator * math.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n))
    return centre - radius, centre + radius


@pytest.mark.parametrize(("k", "n"), [(0, 1), (1, 1), (0, 19), (19, 19), (1, 2), (17, 43)])
def test_p06_wilson_closed_form_endpoints(k, n):
    qualify = bench("qualify")
    assert qualify.wilson(k, n) == pytest.approx(wilson_reference(k, n), abs=1e-5, rel=0)


def test_p06_wilson_generated_complements_and_closed_form():
    qualify = bench("qualify")
    rng = seeded(601)
    for _ in range(100):
        n = rng.randrange(1, 10000)
        k = rng.randrange(n + 1)
        low, high = qualify.wilson(k, n)
        assert (low, high) == pytest.approx(wilson_reference(k, n), abs=1e-5, rel=0)
        other_low, other_high = qualify.wilson(n - k, n)
        assert (low, high) == pytest.approx((1 - other_high, 1 - other_low), abs=1e-12)
        assert -1e-15 <= low <= high <= 1 + 1e-15


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (["task", "setup", "?", "task", "setup"], ["task", "task", "setup", "?", "setup"], (2, 3)),
        (["vis", "occ", "oof"], ["occ", "oof", "vis"], (0, 3)),
        (["L", "R"], ["L", "R"], (2, 2)),
    ],
)
def test_p06_agreement_excludes_uncertainty_on_either_side(a, b, expected):
    qualify = bench("qualify")
    assert qualify.agreement(a, b) == expected
    assert qualify.agreement(b, a) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([0, 0, 10, 10], [0, 0, 10, 10], 1),
        ([0, 0, 10, 10], [10, 0, 20, 10], 0),
        ([0, 0, 10, 10], [20, 0, 30, 10], 0),
        ([0, 0, 10, 10], [5, 0, 15, 10], 1 / 3),
        ([2, 3, 12, 13], [3, 4, 5, 6], 4 / 100),
    ],
)
def test_p06_box_iou_uses_intersection_over_union_not_enclosing_rectangle(a, b, expected):
    qualify = bench("qualify")
    assert qualify.box_iou(a, b) == pytest.approx(expected)
    assert qualify.box_iou(b, a) == pytest.approx(expected)


def test_p06_box_iou_translation_scale_symmetry_property():
    qualify = bench("qualify")
    rng = seeded(602)
    for _ in range(80):
        x, y, width, height = (rng.uniform(1, 50) for _ in range(4))
        a = [x, y, x + width, y + height]
        b = [x + width / 2, y, x + 1.5 * width, y + height]
        scale, dx, dy = rng.uniform(0.1, 20), rng.uniform(-100, 100), rng.uniform(-100, 100)
        transformed = [
            [scale * box[0] + dx, scale * box[1] + dy, scale * box[2] + dx, scale * box[3] + dy]
            for box in (a, b)
        ]
        assert qualify.box_iou(*transformed) == pytest.approx(1 / 3, abs=1e-12)


@pytest.mark.parametrize(
    "diffs",
    [
        {"a": [0, 2], "b": [8], "c": [2, 2, 2, 2, 2]},
        {"a": [0.2], "b": [0.2, 0.2]},
        {"a": [-0.4, 0.2], "b": [0.7, -0.1], "c": [-0.1, 0.4, 0.6]},
    ],
)
def test_p06_paired_subject_sd_weights_subjects_not_frame_counts(diffs):
    qualify = bench("qualify")
    means = [statistics.mean(values) for values in diffs.values()]
    sd, n = qualify.paired_subject_sd(diffs)
    assert n == len(diffs)
    assert sd == pytest.approx(statistics.stdev(means))


def test_p06_paired_subject_sd_replication_and_translation_property():
    qualify = bench("qualify")
    rng = seeded(603)
    for _ in range(30):
        diffs = {
            str(s): [rng.uniform(-1, 1) for _ in range(rng.randrange(1, 10))]
            for s in range(rng.randrange(2, 9))
        }
        want = statistics.stdev(statistics.mean(v) for v in diffs.values())
        duplicated = {
            s: [v + 0.125 for v in values] * rng.randrange(1, 6) for s, values in diffs.items()
        }
        sd, n = qualify.paired_subject_sd(duplicated)
        assert n == len(diffs)
        assert sd == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        ("wilson", (0, 0)),
        ("quantiles", ([],)),
        ("normalized_errors", (np.empty((0, 2)), np.empty((0, 2)), 1)),
        ("agreement", ([], [])),
        ("agreement", (["?"], ["task"])),
        ("agreement", (["task"], ["?"])),
        ("box_iou", ([], [])),
        ("judge_bins", ([], [], [], [0, 0.05, 0.1, 0.2, math.inf])),
        ("paired_subject_sd", ({},)),
        ("paired_subject_sd", ({"a": [], "b": []},)),
    ],
)
def test_p06_empty_population_is_rejected(function, args):
    qualify = bench("qualify")
    with pytest.raises(ValueError):  # noqa: PT011 - wording is not contract-owned
        getattr(qualify, function)(*args)


def test_p06_normalized_errors_are_per_point_euclidean_distances():
    qualify = bench("qualify")
    gt = np.array([[0.0, 0.0], [10.0, 20.0], [-4.0, 8.0]])
    pred = gt + np.array([[3.0, 4.0], [0.0, -10.0], [0.0, 0.0]])
    np.testing.assert_allclose(qualify.normalized_errors(pred, gt, 10), [0.5, 1, 0])


def test_p06_normalized_errors_are_invariant_to_common_similarity():
    qualify = bench("qualify")
    rng = seeded(604)
    for _ in range(40):
        count = rng.randrange(1, 15)
        gt = np.array([[rng.uniform(-50, 50), rng.uniform(-50, 50)] for _ in range(count)])
        delta = np.array([[rng.uniform(-5, 5), rng.uniform(-5, 5)] for _ in range(count)])
        norm, scale = rng.uniform(1, 100), rng.uniform(0.1, 10)
        angle = rng.uniform(-math.pi, math.pi)
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        offset = np.array([17.5, -93.25])
        expected = [math.hypot(*row) / norm for row in delta]
        actual = qualify.normalized_errors(
            scale * (gt + delta) @ rotation + offset, scale * gt @ rotation + offset, scale * norm
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_p06_quantiles_default_custom_order_and_singleton():
    qualify = bench("qualify")
    values = [100, 0, 40, 30, 90, 50, 80, 70, 20, 60, 10]
    assert qualify.quantiles(values) == pytest.approx([50, 90])
    assert qualify.quantiles(values, qs=(1, 0, 0.5, 0.9)) == pytest.approx([100, 0, 50, 90])
    assert qualify.quantiles([3.75]) == pytest.approx([3.75, 3.75])
    assert isinstance(qualify.quantiles(values), list)


def test_p06_generated_quantiles_use_linear_interpolation_and_preserve_input():
    qualify = bench("qualify")
    rng = seeded(605)
    for _ in range(50):
        values = [rng.uniform(-100, 100) for _ in range(rng.randrange(2, 40))]
        before = values.copy()
        ordered = sorted(values)
        qs = [rng.random() for _ in range(6)]
        expected = []
        for q in qs:
            position = q * (len(values) - 1)
            lo, hi = math.floor(position), math.ceil(position)
            expected.append(ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
        assert qualify.quantiles(values, qs=qs) == pytest.approx(expected, abs=1e-12)
        assert values == before


def test_p06_judge_bins_count_decisive_correct_abstaining_at_each_boundary():
    qualify = bench("qualify")
    assert set(qualify.NON_DECISIVE) == {"both-ok", "both-wrong", "?"}
    bins = [0, 0.05, 0.1, 0.2, math.inf]
    margins = [0, 0.01, 0.049, 0.05, 0.09, 0.1, 0.19, 0.2, 9]
    truths = ["a", "a", "b", "a", "b", "a", "b", "a", "b"]
    outcomes = ["a", "b", "?", "both-ok", "b", "both-wrong", "?", "b", "b"]
    rows = qualify.judge_bins(margins, truths, outcomes, bins)
    assert len(rows) == 4
    counts = [(3, 2, 1, 1), (2, 1, 1, 1), (2, 0, 0, 2), (2, 2, 1, 0)]
    for i, (row, expected) in enumerate(zip(rows, counts, strict=True)):
        assert (row["lo"], row["hi"]) == (bins[i], bins[i + 1])
        assert tuple(row[k] for k in ("shown", "decisive", "correct", "abstain")) == expected
        if row["decisive"]:
            assert (row["acc_lo"], row["acc_hi"]) == pytest.approx(
                wilson_reference(row["correct"], row["decisive"]), abs=1e-5, rel=0
            )
        else:
            assert "acc_lo" not in row
            assert "acc_hi" not in row


def test_p06_generated_judge_bins_honor_supplied_edges_and_conserve_population():
    qualify = bench("qualify")
    rng = seeded(607)
    bins = [0, 0.3, 0.7, math.inf]
    for _ in range(25):
        records = []
        expected = []
        for lo, hi in pairwise(bins):
            block = [
                (
                    rng.uniform(lo, min(hi, 2) - 1e-6),
                    rng.choice(("a", "b")),
                    rng.choice(("a", "b", "both-ok", "both-wrong", "?")),
                )
                for _ in range(rng.randrange(2, 15))
            ]
            decisive = [row for row in block if row[2] in ("a", "b")]
            correct = sum(truth == outcome for _, truth, outcome in decisive)
            expected.append((len(block), len(decisive), correct, len(block) - len(decisive)))
            records.extend(block)
        rng.shuffle(records)
        margins, truths, outcomes = zip(*records, strict=True)
        rows = qualify.judge_bins(margins, truths, outcomes, bins)
        assert len(rows) == len(bins) - 1
        assert [
            tuple(row[k] for k in ("shown", "decisive", "correct", "abstain")) for row in rows
        ] == expected
        assert sum(row["shown"] for row in rows) == len(records)


def test_p05_hand_diagonal_depends_on_native_width_height_not_position():
    public = bench("public")
    rng = seeded(505)
    for i in range(50):
        x, y, width, height = (rng.uniform(1, 1000) for _ in range(4))
        hand = public.GtHand(
            image_id=1, ann_id=i, side="L", box=[x, y, width, height], kpts=np.zeros((21, 3))
        )
        assert hand.diagonal == pytest.approx(math.hypot(width, height))


def test_p06_class_agreement_is_per_class_in_both_directions():
    """D06 is per class: 14/14 agreed `vis` cannot clear 0.8 even when the pooled rate is high."""
    qualify = bench("qualify")
    a = ["vis"] * 14 + ["part"] * 100 + ["part", "?", "occ"]
    b = ["vis"] * 14 + ["part"] * 100 + ["vis", "occ", "occ"]
    out = qualify.class_agreement(a, b, ("vis", "part", "occ", "oof"))
    assert out["vis"] == {
        "agree": 14,
        "first": 14,
        "second": 15,
        "lower": qualify.wilson(14, 15)[0],
    }
    assert (out["part"]["agree"], out["part"]["first"], out["part"]["second"]) == (100, 101, 100)
    assert out["occ"] == {"agree": 1, "first": 1, "second": 1, "lower": qualify.wilson(1, 1)[0]}
    assert out["oof"]["lower"] is None
    assert qualify.wilson(14, 14)[0] < 0.8 < qualify.agreement(a, b)[0] / qualify.agreement(a, b)[1]


def test_p07_public_score_consensus_grades_the_mean_point_against_gt(tmp_path, capsys):
    """Consensus accuracy (what the pilot grades against) = the two labellers' mean `vis` point
    vs GT; a joint either labeller left unsure stays out."""
    import json
    import runpy
    from pathlib import Path

    unit = {"origin_x": 0.0, "origin_y": 0.0, "scale_x": 1.0, "scale_y": 1.0, "zoom": 1.0}
    entry = {"token": "t1", "kind": "hand", "key": "coco/1/1", "index": 0, "width": 100,
             "height": 100, "side": "L", "map": unit}  # fmt: skip
    sheets_dir = tmp_path / "sheets"
    sheets_dir.mkdir()
    (sheets_dir / "manifest.json").write_text(json.dumps({"t1": entry}))
    kpts = [[0.0, 0.0, 2.0]] * 21
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"t1": {"kind": "hand", "kpts": kpts, "diag": 10.0}}))
    # wrist: (2, 0) and (4, 0) → mean (3, 0) → 0.3; thumb tip: one labeller unsure → excluded.
    answers = []
    for i, (wx, thumb) in enumerate(((2.0, {"state": "vis", "uv": [1, 0]}), (4.0, {"state": "?"}))):
        path = tmp_path / f"a{i}.jsonl"
        joints = {"wrist": {"state": "vis", "uv": [wx, 0.0]}, "thumb_tip": thumb}
        path.write_text(json.dumps({"sheet": "t1.png", "joints": joints}) + "\n")
        answers.append(str(path))
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "bench.py"))["main"]
    argv = ["public-score", "--sheets", str(sheets_dir), "--gt", str(gt), "--answers", *answers]
    assert main([*argv, "--consensus"]) == 0
    out = capsys.readouterr().out
    assert "consensus hand/wrist      n=  1 err/diag p50=0.300 p90=0.300" in out
    assert "thumb_tip" not in out
    with pytest.raises(SystemExit, match="exactly two"):
        main([*argv[:-2], answers[0], "--consensus"])

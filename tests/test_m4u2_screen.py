"""M4.2 P02-P04: screening rows, PCK, AUROC, the DexYCB joint table, paired bootstrap, finalists."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _screen():
    return importlib.import_module("pose_estimation.bench.screen")


def _public():
    return importlib.import_module("pose_estimation.bench.public")


def _person(image_id=1, ann_id=1, hands=("L", "R"), vis=2):
    public = _public()
    rng = np.random.default_rng(ann_id)
    body = np.column_stack([rng.uniform(0, 100, (17, 2)), np.full(17, vis)])
    return public.GtPerson(
        image_id,
        ann_id,
        (0.0, 0.0, 30.0, 40.0),  # diagonal 50
        body,
        1200.0,
        tuple(
            public.GtHand(
                image_id,
                ann_id,
                side,
                (0.0, 0.0, 6.0, 8.0),  # diagonal 10
                np.column_stack([rng.uniform(0, 10, (21, 2)), np.full(21, 2.0)]),
            )
            for side in hands
        ),
    )


def _prediction(person):
    kps = np.zeros((133, 2))
    kps[:17] = person.body[:, :2]
    for h in person.hands:
        first = 91 if h.side == "L" else 112
        kps[first : first + 21] = h.kpts[:, :2]
    return kps


def test_p02_result_rows_one_per_person_in_the_mmpose_split():
    s = _screen()
    persons = [_person(1, 1), _person(1, 2), _person(2, 3)]
    kps = np.stack([_prediction(p) for p in persons])
    scores = np.random.default_rng(0).uniform(0, 1, (3, 133))
    rows = s.result_rows(persons, kps, scores)
    assert [r["image_id"] for r in rows] == [1, 1, 2]
    for row, k, sc in zip(rows, kps, scores, strict=True):
        assert [len(row[f]) for f, _ in s.RESULT_FIELDS] == [51, 18, 204, 63, 63]
        assert row["score"] == pytest.approx(sc.mean())
        triples = np.concatenate([np.reshape(row[f], (-1, 3)) for f, _ in s.RESULT_FIELDS])
        np.testing.assert_allclose(triples[:, :2], k, atol=5e-4)
        np.testing.assert_allclose(triples[:, 2], sc, atol=5e-4)


def test_p02_hand_and_arm_errors_exact_on_a_synthetic_person():
    s = _screen()
    p = _person(hands=("R",))
    p.hands[0].kpts[3, 2] = 0  # unannotated joint: never scored
    k = _prediction(p)
    k[112 + 8] += (0.6, 0.8)  # index tip 1 px off = 0.1 of the hand diagonal
    k[112 + 4] += (0.3, 0.4)  # thumb tip 0.05
    k[7] += (3.0, 4.0)  # left elbow 5 px = 0.1 of the person diagonal
    err = s.hand_errors([p], k[None])
    assert len(err["all21"]) == 20
    assert len(err["feature5"]) == 5
    assert sorted(err["feature5"])[-2:] == pytest.approx([0.05, 0.1])
    assert s.pck(err["all21"], s.PCK_HAND) == {
        "@0.05": 19 / 20,
        "@0.1": 19 / 20,
        "@0.2": 1.0,
        "n": 20,
    }
    sc = np.linspace(0.1, 0.9, 133)
    p.body[[6, 9], 2] = [1, 0]  # r_shoulder occluded, l_wrist unlabelled
    errors, visible, occluded = s.arm_errors([p], k[None], sc[None])
    assert len(errors) == 4
    assert max(errors) == pytest.approx(0.1)
    assert visible == pytest.approx(sc[[5, 7, 8, 10]].tolist())
    assert occluded == pytest.approx([sc[6]])
    with pytest.raises(ValueError, match="no joints"):
        s.pck([], s.PCK_HAND)


def test_p02_auroc_is_the_mann_whitney_form_with_ties():
    s = _screen()
    rng = np.random.default_rng(4)
    for _ in range(20):
        pos, neg = rng.integers(0, 5, rng.integers(1, 30)), rng.integers(0, 5, rng.integers(1, 30))
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        assert s.auroc(pos, neg) == pytest.approx(brute)
    assert s.auroc([1.0], [1.0]) == 0.5
    with pytest.raises(ValueError, match="both classes"):
        s.auroc([], [0.3])


def test_p03_dexycb_table_maps_toolkit_order_and_drops_unannotated_hands(tmp_path):
    handsets = importlib.import_module("pose_estimation.bench.handsets")
    assert sorted(handsets.DEXYCB_TO_COCO_WB) == list(range(21))
    seq = tmp_path / "seq-1"
    (seq / "cam-1").mkdir(parents=True)
    (seq / "meta.yml").write_text("mano_sides: [left]\n", encoding="utf-8")
    joints = np.random.default_rng(1).uniform(5, 50, (1, 21, 2))
    np.savez(seq / "cam-1" / "labels_000000.npz", joint_2d=joints)
    np.savez(seq / "cam-1" / "labels_000001.npz", joint_2d=np.full((1, 21, 2), -1.0))
    frames = list(handsets.dexycb(tmp_path, every=1))
    assert len(frames) == 1  # the all -1 label is no annotation
    f = frames[0]
    assert f.side == "L"
    assert f.image.name == "color_000000.jpg"
    np.testing.assert_array_equal(f.kpts, joints[0][list(handsets.DEXYCB_TO_COCO_WB)])
    assert f.valid.all()
    assert not f.occluded.any()
    assert f.box == pytest.approx((*joints[0].min(axis=0), *joints[0].max(axis=0)))


def test_p04_paired_bootstrap_is_exact_paired_and_seeded():
    s = _screen()
    rng = np.random.default_rng(2)
    valid = rng.random((200, 21)) < 0.9
    a = rng.random((200, 21)) < 0.7
    b = a.copy()
    b[:20] = True  # b = a plus hits on 20 images
    out = s.paired_pck_bootstrap({"a": a, "b": b}, valid, 300, 5)
    assert out["arms"]["a"]["pck"] == pytest.approx((a & valid).sum() / valid.sum())
    lo, hi = out["pairs"]["b-a"]["interval"]
    assert 0 <= lo <= out["pairs"]["b-a"]["difference"] <= hi  # paired: b never trails a
    assert out == s.paired_pck_bootstrap({"a": a, "b": b}, valid, 300, 5)
    same = s.paired_pck_bootstrap({"a": a, "c": a.copy()}, valid, 50, 5)
    assert same["pairs"]["a-c"]["interval"] == (0.0, 0.0)
    with pytest.raises(ValueError, match="no valid joints"):
        s.paired_pck_bootstrap({"a": a}, np.zeros_like(valid), 10, 0)


def _table(entries):
    return {key: {"interval": (lo, hi)} for key, (lo, hi) in entries.items()}


def test_p04_finalist_rule_dominance_body_slack_and_baseline():
    s = _screen()
    names = ("base", "big", "mid", "flat")
    zero = {f"{x}-{y}": (-0.01, 0.01) for x in names for y in names if x != y}
    ap = _table(zero | {"big-mid": (0.01, 0.02), "big-base": (0.1, 0.2), "big-flat": (0.01, 0.03)})
    pck = _table(zero | {"big-mid": (0.002, 0.01), "big-base": (0.05, 0.1)})
    arm_pck = {"base": 0.85, "big": 0.93, "mid": 0.92, "flat": 0.92}
    v = s.finalists(arm_pck, [ap, pck], "base")
    assert v["big"] == []  # flat: beaten on one metric only; 0.01 slack holds
    assert v["flat"] == []
    assert v["mid"] == ["dominated by big"]
    assert v["base"] == []  # dominated + short on body, kept as the baseline
    assert s.finalists(arm_pck | {"flat": 0.9199}, [ap, pck], "base")["flat"][0].startswith(
        "arm PCK@0.05"
    )


def test_p03_dexycb_slots_join_the_codebook_by_joint_name():
    """Independent oracle = the labelling codebook's COCO-WB indices (closing review U12)."""
    handsets = importlib.import_module("pose_estimation.bench.handsets")
    labels = importlib.import_module("pose_estimation.bench.labels")
    table = handsets.DEXYCB_TO_COCO_WB
    for name, slot in labels.HAND_JOINTS.items():
        toolkit = "index_mcp" if name == "index_mcp" else name
        assert handsets.DEXYCB_JOINTS[table[slot]] == toolkit


def test_a03_bench_arms_are_runtime_registry_entries():
    """One checkpoint, one name: bench arm = registry key, its calibration file `<key>.json`."""
    arms = importlib.import_module("pose_estimation.bench.arms")
    run = importlib.import_module("pose_estimation.run")
    for key, arm in arms.POSE_ARMS.items():
        entry = run.MODEL_REGISTRY[key]
        size = entry["pose_input_size"]
        assert (arm.url, arm.input_size) == (entry["pose"], tuple(size))  # ty: ignore[invalid-argument-type]
        assert entry["calibration"] in (None, f"{key}.json")

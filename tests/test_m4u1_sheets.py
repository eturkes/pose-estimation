"""M4.1 P04/P06b: blinded jobs and stored-map answer ingestion."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from test_m4u1_bench import bench, seeded


def judge_fixture(judge, tmp_path, *, seed=77, count=3):
    rng = seeded(seed)
    pairs = [
        judge.Pair(
            pair_id=f"private-pair-{i}",
            image=f"private-image-{i}.png",
            joint="index_tip",
            a=(rng.uniform(55, 75), rng.uniform(70, 85)),
            b=(rng.uniform(90, 110), rng.uniform(95, 115)),
            arm_a=f"private-first-arm-{i}",
            arm_b=f"private-second-arm-{i}",
            side="L" if i % 2 else "R",
        )
        for i in range(count)
    ]
    requested = []

    def load_image(name):
        requested.append(name)
        return np.full((180, 220, 3), 90, dtype=np.uint8)

    sheet_dir = tmp_path / "sheets"
    key_path = tmp_path / "sealed.jsonl"
    jobs = judge.make_sheets(pairs, load_image, sheet_dir, key_path, seed)
    assert set(requested) == {p.image for p in pairs}
    key = [json.loads(line) for line in key_path.read_text().splitlines()]
    return pairs, jobs, key, sheet_dir


@pytest.mark.parametrize("seed", [0, 17, 991])
def test_p04_judge_jobs_and_filenames_blind_all_provenance(tmp_path, seed):
    judge = bench("judge")
    pairs, jobs, key, sheet_dir = judge_fixture(judge, tmp_path, seed=seed, count=seed % 4 + 2)
    names = [p.name for p in sheet_dir.iterdir() if p.is_file()]
    assert len(names) == len(jobs) == len(key) == 2 * len(pairs)
    assert len(set(names)) == len(names)
    exposed = json.dumps(jobs, sort_keys=True) + "\n" + "\n".join(names)
    for pair in pairs:
        for value in (pair.arm_a, pair.arm_b, pair.image, pair.pair_id):
            assert value not in exposed, f"P04/NC2: provenance exposed: {value}"


@pytest.mark.parametrize(
    ("margin", "expected"),
    [
        (0, 0),
        (math.nextafter(0.05, 0), 0),
        (0.05, 1),
        (math.nextafter(0.1, 0), 1),
        (0.1, 2),
        (math.nextafter(0.2, 0), 2),
        (0.2, 3),
        (1e100, 3),
    ],
)
def test_p04_bin_boundaries_are_half_open(margin, expected):
    judge = bench("judge")
    assert tuple(judge.BINS) == (0, 0.05, 0.1, 0.2, math.inf)
    assert judge.bin_of(margin) == expected


@pytest.mark.parametrize("seed", [0, 29, 382])
def test_p04_generated_pairs_have_opposite_balanced_keyed_colours(tmp_path, seed):
    judge = bench("judge")
    pairs, jobs, key, sheet_dir = judge_fixture(judge, tmp_path, seed=seed, count=seed % 7 + 1)
    assert judge.RESPONSES == ("yellow", "blue", "both-ok", "both-wrong", "?")
    assert set(judge.COLOURS) == {"yellow", "blue"}
    assert len(key) == 2 * len(pairs)
    assert {row["token"] + ".png" for row in key} == {job["sheet"] for job in jobs}
    assert {job["sheet"] for job in jobs} == {p.name for p in sheet_dir.iterdir()}
    assert all(set(job) == {"sheet", "joint"} for job in jobs)
    assert all(job["joint"] == "index_tip" for job in jobs)
    for pair in pairs:
        rows = [row for row in key if row["pair_id"] == pair.pair_id]
        assert len(rows) == 2
        assert {row["swap"] for row in rows} == {0, 1}
        assert {row["yellow"] for row in rows} == {pair.arm_a, pair.arm_b}
        assert {row["blue"] for row in rows} == {pair.arm_a, pair.arm_b}
        for row in rows:
            assert set(row) == {"token", "pair_id", "swap", "yellow", "blue"}
            assert {row["yellow"], row["blue"]} == {pair.arm_a, pair.arm_b}
    for colour in ("yellow", "blue"):
        assert sum(
            row[colour] == p.arm_a for p in pairs for row in key if row["pair_id"] == p.pair_id
        ) == len(pairs)
    _, repeated_jobs, repeated_key, _ = judge_fixture(
        judge, tmp_path / "repeat", seed=seed, count=len(pairs)
    )
    assert repeated_jobs == jobs
    assert repeated_key == key


@pytest.mark.parametrize("response", ["yellow", "blue", "both-ok", "both-wrong", "?"])
def test_p04_decode_maps_each_colour_and_preserves_nondecisive_classes(tmp_path, response):
    judge = bench("judge")
    _, _, key, _ = judge_fixture(judge, tmp_path)
    answers = {row["token"] + ".png": response for row in key}
    decoded = judge.decode(key, answers)
    assert len(decoded) == len(key)
    by_pair_swap = {(row["pair_id"], row["swap"]): row for row in decoded}
    assert len(by_pair_swap) == len(key)
    for row in key:
        result = by_pair_swap[row["pair_id"], str(row["swap"])]
        expected = row[response] if response in ("yellow", "blue") else response
        assert result == {"pair_id": row["pair_id"], "swap": str(row["swap"]), "outcome": expected}


def test_p04_decode_skips_unanswered_sheets_and_rejects_unrecognised_answers(tmp_path):
    judge = bench("judge")
    _, _, key, _ = judge_fixture(judge, tmp_path)
    assert judge.decode(key, {}) == []
    row = key[0]
    file_name = row["token"] + ".png"
    decoded = judge.decode(key, {file_name: "yellow"})
    assert decoded == [
        {"pair_id": row["pair_id"], "swap": str(row["swap"]), "outcome": row["yellow"]}
    ]
    for response in ("A", "B", "red", "", "unknown-response"):
        with pytest.raises(ValueError):  # noqa: PT011 - wording is not contract-owned
            judge.decode(key, {file_name: response})


def ingest_fixture(sheets, labels, tmp_path):
    key = "synthetic-event/camera#7"
    common = {"key": key, "index": 7, "width": 640, "height": 480}
    maps = {
        "frame": {"origin_x": 0, "origin_y": 0, "scale_x": 6.4, "scale_y": 4.8, "zoom": 1},
        "body": {"origin_x": 60, "origin_y": 40, "scale_x": 20, "scale_y": 20, "zoom": 2},
        "left": {"origin_x": 0, "origin_y": 48, "scale_x": 12, "scale_y": 12, "zoom": 640 / 192},
        "right": {"origin_x": 320, "origin_y": 48, "scale_x": 12, "scale_y": 12, "zoom": 640 / 192},
    }
    entries = [
        sheets.SheetEntry(token="frame-token", kind="frame", side="", map=maps["frame"], **common),
        sheets.SheetEntry(token="body-token", kind="body", side="", map=maps["body"], **common),
        sheets.SheetEntry(token="left-token", kind="hand", side="L", map=maps["left"], **common),
        sheets.SheetEntry(token="right-token", kind="hand", side="R", map=maps["right"], **common),
    ]
    sheets.save(tmp_path, entries, seed=85)
    return [
        {
            "sheet": "frame-token.png",
            "task": "setup",
            "patient": [5, 10, 90, 90],
            "others": [[92, 2, 99, 15]],
            "L": {"hand": "vis", "extent": [10, 20, 20, 40]},
            "R": {"hand": "part", "extent": [60, 20, 70, 40]},
            "note": "synthetic",
        },
        {
            "sheet": "body-token.png",
            **{
                side: {
                    "shoulder": {"state": "vis", "uv": [5, 6], "own": "patient"},
                    "elbow": {"state": "occ"},
                    "wrist": {"state": "oof"},
                }
                for side in labels.SIDES
            },
        },
        {
            "sheet": "left-token.png",
            "joints": {
                name: {"state": "vis", "uv": [6 + i, 6 + i], "own": "patient"}
                for i, name in enumerate(labels.HAND_JOINTS)
            },
        },
        {
            "sheet": "right-token.png",
            "joints": {
                name: {"state": "vis", "uv": [6, 6], "own": "ambig"}
                if name == "wrist"
                else {"state": "occ"}
                for name in labels.HAND_JOINTS
            },
        },
    ]


def test_p06b_complete_synthetic_answers_fill_frame_body_and_hand_coordinates(tmp_path):
    sheets, labels = bench("sheets"), bench("labels")
    answers = ingest_fixture(sheets, labels, tmp_path)
    records = sheets.ingest(tmp_path, answers, "synthetic-reader")
    assert len(records) == 1
    record = records[0]
    assert record["frame"] == {
        "key": "synthetic-event/camera#7",
        "index": 7,
        "width": 640,
        "height": 480,
    }
    assert record["labeller"] == "synthetic-reader"
    assert record["task"] == "setup"
    assert record["patient"] == pytest.approx([32, 48, 576, 432])
    np.testing.assert_allclose(record["others"], [[588.8, 9.6, 633.6, 72]])
    assert record["sides"]["L"]["extent"] == pytest.approx([64, 96, 128, 192])
    assert record["sides"]["R"]["extent"] == pytest.approx([384, 96, 448, 192])
    assert record["sides"]["L"]["hand"] == "vis"
    assert record["sides"]["R"]["hand"] == "part"
    for side in labels.SIDES:
        body = record["sides"][side]["body"]
        assert body["shoulder"]["xy"] == pytest.approx([160, 160])
        assert body["shoulder"]["own"] == "patient"
        assert body["elbow"] == {"state": "occ"}
        assert body["wrist"] == {"state": "oof"}
    for i, name in enumerate(labels.HAND_JOINTS):
        joint = record["sides"]["L"]["joints"][name]
        assert joint["state"] == "vis"
        assert joint["xy"] == pytest.approx([72 + 12 * i, 120 + 12 * i])
        assert joint["own"] == "patient"
    right = record["sides"]["R"]["joints"]
    assert right["wrist"]["xy"] == pytest.approx([392, 120])
    assert right["wrist"]["own"] == "ambig"
    assert all(right[name] == {"state": "occ"} for name in labels.HAND_JOINTS if name != "wrist")
    assert labels.validate(record) == []


@pytest.mark.parametrize("kind", ["body", "left", "right"])
def test_p06b_crop_without_its_frame_answer_raises(tmp_path, kind):
    sheets, labels = bench("sheets"), bench("labels")
    answers = ingest_fixture(sheets, labels, tmp_path)
    crop = next(answer for answer in answers if answer["sheet"] == kind + "-token.png")
    with pytest.raises(ValueError):  # noqa: PT011 - wording is not contract-owned
        sheets.ingest(tmp_path, [crop], "synthetic-reader")


def test_p06b_answers_in_any_order_join_by_frame_key_not_position(tmp_path):
    sheets, labels = bench("sheets"), bench("labels")
    answers = ingest_fixture(sheets, labels, tmp_path)
    expected = sheets.ingest(tmp_path, answers, "synthetic-reader")
    assert len(expected) == 1
    rng = seeded(606)
    for _ in range(10):
        rng.shuffle(answers)
        assert sheets.ingest(tmp_path, answers, "synthetic-reader") == expected


def test_p06b_multiple_frame_keys_keep_independent_geometry_and_body_answers(tmp_path):
    sheets, labels = bench("sheets"), bench("labels")
    entries, answers, expected = [], [], {}
    for index, width, height in [(2, 1000, 500), (9, 800, 1200)]:
        key = f"synthetic-event/camera#{index}"
        frame_token, body_token = f"frame-{index}", f"body-{index}"
        common = {"key": key, "index": index, "width": width, "height": height, "side": ""}
        frame_map = {
            "origin_x": 0,
            "origin_y": 0,
            "scale_x": width / 100,
            "scale_y": height / 100,
            "zoom": 1,
        }
        body_map = {
            "origin_x": index * 10,
            "origin_y": index * 20,
            "scale_x": 10,
            "scale_y": 15,
            "zoom": 2,
        }
        entries.extend(
            [
                sheets.SheetEntry(token=frame_token, kind="frame", map=frame_map, **common),
                sheets.SheetEntry(token=body_token, kind="body", map=body_map, **common),
            ]
        )
        answers.extend(
            [
                {
                    "sheet": frame_token + ".png",
                    "task": "task",
                    "patient": [5, 10, 95, 90],
                    "others": [],
                    "L": {"hand": "occ"},
                    "R": {"hand": "oof"},
                },
                {
                    "sheet": body_token + ".png",
                    **{
                        side: {name: {"state": "vis", "uv": [1, 2]} for name in labels.BODY_JOINTS}
                        for side in labels.SIDES
                    },
                },
            ]
        )
        expected[key] = (
            [width * 0.05, height * 0.1, width * 0.95, height * 0.9],
            [index * 10 + 10, index * 20 + 30],
        )
    sheets.save(tmp_path, entries, seed=88)
    rows = sheets.ingest(tmp_path, answers, "synthetic-reader")
    assert len(rows) == len(expected) == 2
    assert {row["frame"]["key"] for row in rows} == set(expected)
    for row in rows:
        box, xy = expected[row["frame"]["key"]]
        assert row["patient"] == pytest.approx(box)
        for side in labels.SIDES:
            for name in labels.BODY_JOINTS:
                assert row["sides"][side]["body"][name]["xy"] == pytest.approx(xy)


def test_p04_rendered_colour_swap_matches_sealed_arm_assignments(tmp_path):
    judge = bench("judge")
    pairs, _, key, sheet_dir = judge_fixture(judge, tmp_path, count=1)
    for bgr in judge.COLOURS.values():
        values = np.asarray(bgr)
        assert values.shape == (3,)
        assert values.dtype.kind in "ui"
        assert np.all((values >= 0) & (values <= 255))
    yellow, blue = judge.COLOURS["yellow"], judge.COLOURS["blue"]
    assert min(yellow[1:]) > yellow[0]
    assert blue[0] > max(blue[1:])
    images = []
    for row in key:
        image = cv2.imread(str(sheet_dir / (row["token"] + ".png")))
        assert image is not None
        images.append(image)
    assert len(images) == 2
    assert images[0].shape == images[1].shape
    changed = np.any(images[0] != images[1], axis=2)
    centroids = []
    for row, image in zip(key, images, strict=True):
        masks = {
            colour: changed & np.all(image == bgr, axis=2) for colour, bgr in judge.COLOURS.items()
        }
        centres = {}
        for colour, mask in masks.items():
            ys, xs = np.nonzero(mask)
            assert len(xs) > 0, f"the {colour} candidate marker must change when colours swap"
            centres[colour] = (float(xs.mean()), float(ys.mean()))
        a_colour = next(c for c in masks if row[c] == pairs[0].arm_a)
        b_colour = next(c for c in masks if row[c] == pairs[0].arm_b)
        assert centres[a_colour][0] < centres[b_colour][0], "candidate a is left of candidate b"
        centroids.append(centres)
    assert centroids[0]["yellow"] == pytest.approx(centroids[1]["blue"], abs=1)
    assert centroids[0]["blue"] == pytest.approx(centroids[1]["yellow"], abs=1)


def test_p06b_frame_without_a_patient_exports_out_of_frame_arm_joints(tmp_path):
    """MAIN regression (general-purpose-4, pilot ingest): `patient: null` frames get no body sheet,
    so their arm joints must come from the frame answer, else every such record fails validate."""
    sheets, labels = bench("sheets"), bench("labels")
    entry = sheets.SheetEntry(
        "frame-token",
        "frame",
        "event/cam#3",
        3,
        640,
        480,
        "",
        {"origin_x": 0.0, "origin_y": 0.0, "scale_x": 6.4, "scale_y": 4.8, "zoom": 1.0},
    )
    sheets.save(tmp_path, [entry], 1)
    answer = {
        "sheet": "frame-token.png",
        "task": "setup",
        "patient": None,
        "others": [],
        "L": {"hand": "oof"},
        "R": {"hand": "oof"},
    }
    [record] = sheets.ingest(tmp_path, [answer], "synthetic-reader")
    assert labels.validate(record) == []
    for side in labels.SIDES:
        assert {j["state"] for j in record["sides"][side]["body"].values()} == {"oof"}

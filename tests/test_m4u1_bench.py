"""M4.1 P01-P03: contract-only labels, geometry and sampling witnesses."""

from __future__ import annotations

import random
from collections import Counter
from importlib import import_module

import numpy as np
import pytest


def bench(name):
    # Deferred imports give every case its own red witness on the absent package.
    return import_module(f"pose_estimation.bench.{name}")


def seeded(seed):
    return random.Random(seed)  # noqa: S311 - reproducible synthetic inputs


def assert_inverse(sm, points):
    for x, y in points:
        assert sm.to_frame(*sm.to_units(x, y)) == pytest.approx((x, y), abs=1e-9, rel=0)
        assert sm.to_units(*sm.to_frame(x, y)) == pytest.approx((x, y), abs=1e-9, rel=0)


def assert_marker(canvas, sm, point):
    mask = (canvas[:, :, 0] > 220) & (canvas[:, :, 1] < 40) & (canvas[:, :, 2] > 220)
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "the known synthetic source pixel must survive rendering"
    assert (float(xs.mean()), float(ys.mean())) == pytest.approx(sm.to_sheet(*point), abs=2)


@pytest.mark.parametrize("shape", [(800, 1200), (1200, 800), (2200, 3300), (3300, 2200)])
def test_p02_frame_map_geometry_inverse_and_pixel(shape):
    render = bench("render")
    height, width = shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    point = (round(width * 0.373), round(height * 0.427))
    x, y = point
    image[y - 4 : y + 5, x - 4 : x + 5] = (255, 0, 255)
    canvas, sm = render.frame_sheet(image)
    assert render.PAD == 40
    assert max(canvas.shape[:2]) <= 1600 + 2 * render.PAD
    assert sm.to_frame(0, 0) == pytest.approx((0, 0))
    assert sm.to_frame(100, 100) == pytest.approx((width, height))
    assert sm.to_sheet(0, 0) == pytest.approx((render.PAD, render.PAD))
    assert sm.to_sheet(width, height) == pytest.approx(
        (canvas.shape[1] - render.PAD, canvas.shape[0] - render.PAD), abs=1
    )
    rng = seeded(201)
    assert_inverse(
        sm, [(rng.uniform(-width, width), rng.uniform(-height, height)) for _ in range(80)]
    )
    assert_marker(canvas, sm, point)


@pytest.mark.parametrize(
    ("shape", "centre", "side", "out", "grid", "point"),
    [
        ((180, 300), (150, 90), 160, 320, 16, (127, 67)),
        ((300, 180), (90, 150), 160, 640, 16, (67, 127)),
        ((180, 300), (25, 30), 120, 480, 16, (13, 18)),
        ((180, 300), (280, 165), 120, 480, 16, (268, 153)),
        ((180, 300), (140, 90), 120, 360, 12, (127, 77)),
    ],
)
def test_p02_crop_map_geometry_inverse_and_pixel(shape, centre, side, out, grid, point):
    render = bench("render")
    image = np.zeros((*shape, 3), dtype=np.uint8)
    x, y = point
    image[y - 2 : y + 3, x - 2 : x + 3] = (255, 0, 255)
    canvas, sm = render.crop_sheet(image, centre, side, out=out, grid=grid)
    origin = (centre[0] - side / 2, centre[1] - side / 2)
    assert canvas.shape[:2] == (out + 2 * render.PAD, out + 2 * render.PAD)
    assert sm.to_frame(0, 0) == pytest.approx(origin)
    assert sm.to_frame(grid, grid) == pytest.approx((origin[0] + side, origin[1] + side))
    assert sm.to_sheet(*origin) == pytest.approx((render.PAD, render.PAD))
    assert sm.to_sheet(*centre) == pytest.approx((render.PAD + out / 2, render.PAD + out / 2))
    rng = seeded(202)
    assert_inverse(sm, [(rng.uniform(-300, 300), rng.uniform(-300, 300)) for _ in range(100)])
    assert_marker(canvas, sm, point)


def test_p02_clipped_crop_retains_grey_padding_instead_of_recentering():
    render = bench("render")
    image = np.full((100, 180, 3), (20, 80, 170), dtype=np.uint8)
    canvas, sm = render.crop_sheet(image, (15, 20), 120, out=480, grid=16)
    assert sm.to_frame(0, 0) == pytest.approx((-45, -40))
    u, v = (round(n) for n in sm.to_sheet(-20, -15))
    outside = canvas[v, u].astype(int)
    assert np.ptp(outside) == 0
    assert 0 < outside[0] < 255
    u, v = (round(n) for n in sm.to_sheet(31, 36))
    np.testing.assert_array_equal(canvas[v, u], image[36, 31])


def synthetic_clips(sample, count=12, frames=41):
    clips = [
        sample.Clip(
            event_id=f"synthetic-event-{subject:02d}-{event}",
            camera_name=f"camera-{view}",
            subject=f"synthetic-person-{subject:02d}",
            task=f"task-{event}",
            side="L" if event == 0 else "R",
            view=view,
            n_frames=frames + subject + event,
        )
        for subject in range(count)
        for event in range(2)
        for view in sample.VIEWS
    ]
    watch = {(c.event_id, c.camera_name) for c in clips if c.task == "task-1"}
    return clips, watch


def sample_signature(rows):
    return [(r.event_id, r.camera_name, r.frame, r.stratum, r.subject, r.view) for r in rows]


def assert_sample(rows, clips, watch, subjects, per_view, stress, views):
    assert len(rows) == subjects * len(views) * per_view + stress
    lookup = {(c.event_id, c.camera_name): c for c in clips}
    main = [r for r in rows if (r.event_id, r.camera_name) not in watch]
    stress_rows = [r for r in rows if (r.event_id, r.camera_name) in watch]
    assert len(main) == subjects * len(views) * per_view
    assert len(stress_rows) == stress
    assert len({r.subject for r in main}) == subjects
    assert {r.stratum for r in main}.isdisjoint({r.stratum for r in stress_rows})
    counts = Counter((r.subject, r.view) for r in main)
    for subject in {r.subject for r in main}:
        assert {view for s, view in counts if s == subject} == set(views)
        assert [counts[subject, view] for view in views] == [per_view] * len(views)
    unique = {(r.event_id, r.camera_name, r.frame) for r in rows}
    assert len(unique) == len(rows), "NC3: quotas consume distinct eligible frames"
    for row in rows:
        clip = lookup[row.event_id, row.camera_name]
        assert 0 <= row.frame < clip.n_frames
        assert row.subject == clip.subject
        assert row.view == clip.view


@pytest.mark.parametrize("seed", [0, 1, 71, 991])
def test_p03_seeded_sample_balanced_disjoint_and_deterministic(seed):
    sample = bench("sample")
    assert sample.VIEWS == ("above", "left", "right")
    clips, watch = synthetic_clips(sample)
    before = [(c.event_id, c.camera_name, c.n_frames) for c in clips]
    rows = sample.corpus_sample(clips, seed, watch)
    assert_sample(rows, clips, watch, 8, 3, 30, sample.VIEWS)
    assert sample_signature(rows) == sample_signature(sample.corpus_sample(clips, seed, watch))
    assert sample_signature(rows) != sample_signature(
        sample.corpus_sample(clips, seed + 101, watch)
    )
    assert [(c.event_id, c.camera_name, c.n_frames) for c in clips] == before


def test_p03_generated_quotas_and_missing_views():
    sample = bench("sample")
    rng = seeded(203)
    for seed in range(12):
        subjects, per_view, stress = rng.randrange(1, 5), rng.randrange(1, 5), rng.randrange(0, 9)
        clips, watch = synthetic_clips(sample, count=subjects + 3)
        bad_subject = clips[-1].subject
        clips = [c for c in clips if not (c.subject == bad_subject and c.view == "right")]
        rows = sample.corpus_sample(
            clips, seed, watch, subjects=subjects, per_view=per_view, stress=stress
        )
        assert_sample(rows, clips, watch, subjects, per_view, stress, sample.VIEWS)
        assert all(
            r.subject != bad_subject for r in rows if (r.event_id, r.camera_name) not in watch
        )


@pytest.mark.parametrize(
    "cause", ["few-subjects", "missing-view", "watch-only-view", "empty-watch"]
)
def test_p03_ineligible_population_raises(cause):
    sample = bench("sample")
    clips, watch = synthetic_clips(sample, count=3)
    subjects, stress = 3, 1
    if cause == "few-subjects":
        subjects = 4
    elif cause == "missing-view":
        clips = [c for c in clips if c.view != "above"]
    elif cause == "watch-only-view":
        watch |= {(c.event_id, c.camera_name) for c in clips if c.view == "above"}
    else:
        watch = set()
    with pytest.raises(ValueError):  # noqa: PT011 - wording is not contract-owned
        sample.corpus_sample(clips, 7, watch, subjects=subjects, per_view=1, stress=stress)


@pytest.mark.parametrize("stratum", ["main", "stress"])
def test_p03_nc3_quota_exceeding_distinct_eligible_frames_raises(stratum):
    sample = bench("sample")
    clips = [
        sample.Clip("synthetic-main", view, "synthetic-person", "task", "L", view, 1)
        for view in sample.VIEWS
    ]
    clips.append(sample.Clip("synthetic-watch", "left", "synthetic-person", "task", "R", "left", 1))
    watch = {("synthetic-watch", "left")}
    with pytest.raises(ValueError):  # noqa: PT011 - wording is not contract-owned
        sample.corpus_sample(
            clips,
            9,
            watch,
            subjects=1,
            per_view=2 if stratum == "main" else 1,
            stress=2 if stratum == "stress" else 0,
        )


def test_p03_full_decoded_range_is_sampleable_and_capacity_boundary_succeeds():
    sample = bench("sample")
    clips = [sample.Clip("synthetic", v, "person", "task", "L", v, 5) for v in sample.VIEWS]
    rows = sample.corpus_sample(clips, 3, set(), subjects=1, per_view=5, stress=0)
    assert_sample(rows, clips, set(), 1, 5, 0, sample.VIEWS)
    for view in sample.VIEWS:
        assert {r.frame for r in rows if r.view == view} == set(range(5))


def test_p03_family_labels_share_event_number_and_use_contract_order():
    sample = bench("sample")
    clips = [
        sample.Clip(event, camera, "person", task, side, view, 20)
        for event, task, side in [
            ("z", "beta", "L"),
            ("b", "alpha", "L"),
            ("a", "alpha", "L"),
            ("c", "alpha", "R"),
        ]
        for camera, view in [("z-camera", "above"), ("a-camera", "left"), ("m-camera", "right")]
    ]
    expected_order = sorted(clips, key=lambda c: (c.task, c.side, c.event_id, c.camera_name))
    event_order = list(dict.fromkeys(c.event_id for c in expected_order))
    expected = {
        (
            c.event_id,
            c.camera_name,
        ): f"#{event_order.index(c.event_id) + 1:03d} {c.task}/{c.side}/{c.view}"
        for c in clips
    }
    rng = seeded(204)
    for _ in range(8):
        rng.shuffle(clips)
        assert sample.family_labels(clips) == expected


def complete_record(labels):
    return {
        "frame": {"key": "synthetic-event/camera#7", "index": 7, "width": 640, "height": 480},
        "labeller": "synthetic-labeller",
        "task": "task",
        "patient": [10, 20, 600, 450],
        "others": [[601, 10, 639, 100]],
        "sides": {
            side: {
                "hand": "vis",
                "extent": [100, 150, 250, 300],
                "joints": {
                    name: {"state": "vis", "xy": [120 + i * 10, 180 + i * 10]}
                    for i, name in enumerate(labels.HAND_JOINTS)
                },
                "body": {
                    name: {"state": "vis", "xy": [200 + i * 10, 100 + i * 30]}
                    for i, name in enumerate(labels.BODY_JOINTS)
                },
            }
            for side in labels.SIDES
        },
    }


def test_p01_codebook_symbols_are_frozen():
    labels = bench("labels")
    assert labels.TASK_STATES == ("task", "setup", "?")
    assert labels.HAND_STATES == ("vis", "part", "occ", "oof", "?")
    assert labels.JOINT_STATES == ("vis", "occ", "oof", "?")
    assert labels.OWNERS == ("patient", "other", "ambig")
    assert labels.SIDES == ("L", "R")
    assert labels.HAND_JOINTS == {
        "wrist": 0,
        "thumb_tip": 4,
        "index_mcp": 5,
        "index_tip": 8,
        "little_tip": 20,
    }
    assert labels.BODY_JOINTS == ("shoulder", "elbow", "wrist")


@pytest.mark.parametrize("task", ["task", "setup", "?"])
@pytest.mark.parametrize("hand", ["vis", "part", "occ", "oof", "?"])
def test_p01_complete_v2_records_accept_each_hand_and_task_state(task, hand):
    labels = bench("labels")
    record = complete_record(labels)
    record["task"] = task
    record["patient"] = None
    record["others"] = []
    left = record["sides"]["L"]
    left["hand"] = hand
    if hand != "vis":
        left["joints"] = {
            name: {"state": "?" if hand == "?" else "occ"} for name in labels.HAND_JOINTS
        }
        if hand == "part":
            left["joints"]["wrist"] = {"state": "vis", "xy": [120, 190]}
        else:
            del left["extent"]
    left["body"] = {
        "shoulder": {"state": "occ"},
        "elbow": {"state": "oof"},
        "wrist": {"state": "?"},
    }
    assert labels.validate(record) == []


@pytest.mark.parametrize("owner", ["patient", "other", "ambig"])
def test_p01_allowed_owners_accept_without_changing_anatomical_side(owner):
    labels = bench("labels")
    record = complete_record(labels)
    record["sides"]["R"]["joints"]["index_tip"]["own"] = owner
    assert labels.validate(record) == []


@pytest.mark.parametrize(
    ("path", "value", "named"),
    [
        (("task",), "unknown-task", "task"),
        (("sides", "L", "hand"), "unknown-hand", "hand"),
        (("sides", "L", "joints", "wrist", "state"), "unknown-state", "wrist"),
        (("sides", "R", "body", "elbow", "state"), "unknown-state", "elbow"),
        (("sides", "L", "joints", "index_tip", "own"), "unknown-owner", "own"),
        (("sides", "R", "body", "shoulder", "own"), "unknown-owner", "own"),
        (("unexpected_root",), 1, "unexpected_root"),
        (("sides", "L", "unexpected_side"), 1, "unexpected_side"),
        (("sides", "L", "joints", "unexpected_joint"), {"state": "occ"}, "unexpected_joint"),
        (("sides", "R", "body", "unexpected_body"), {"state": "occ"}, "unexpected_body"),
        (("sides", "L", "joints", "wrist", "unexpected_joint_key"), 1, "unexpected_joint_key"),
    ],
)
def test_p01_each_unknown_field_state_joint_owner_is_named(path, value, named):
    labels = bench("labels")
    record = complete_record(labels)
    assert labels.validate(record) == []
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    errors = labels.validate(record)
    assert isinstance(errors, list)
    assert errors
    assert all(isinstance(e, str) for e in errors)
    assert any(named in error for error in errors), errors


@pytest.mark.parametrize("hand", ["vis", "part"])
def test_p01_visible_hand_requires_extent(hand):
    labels = bench("labels")
    record = complete_record(labels)
    left = record["sides"]["L"]
    left["hand"] = hand
    if hand == "part":
        left["joints"]["little_tip"] = {"state": "occ"}
    del left["extent"]
    errors = labels.validate(record)
    assert any("extent" in error for error in errors), errors


@pytest.mark.parametrize("state", ["occ", "oof", "?"])
@pytest.mark.parametrize("group", ["joints", "body"])
def test_p01_nc1_nonvisible_joint_rejects_xy_by_name(state, group):
    labels = bench("labels")
    record = complete_record(labels)
    record["sides"]["L"]["hand"] = "part"
    record["sides"]["L"][group]["wrist"] = {"state": state, "xy": [120, 190]}
    errors = labels.validate(record)
    assert any("wrist" in error and "xy" in error for error in errors), errors


@pytest.mark.parametrize("group", ["joints", "body"])
def test_p01_visible_joint_without_coordinates_is_named(group):
    labels = bench("labels")
    record = complete_record(labels)
    del record["sides"]["R"][group]["wrist"]["xy"]
    errors = labels.validate(record)
    assert any("wrist" in error and "xy" in error for error in errors), errors


@pytest.mark.parametrize(
    "point",
    [
        [-0.01, 10],
        [10, -0.01],
        [640.01, 10],
        [10, 480.01],
        [np.nan, 10],
        [10, np.nan],
        [np.inf, 10],
        [10, -np.inf],
    ],
)
def test_p01_joint_outside_each_frame_edge_is_rejected(point):
    labels = bench("labels")
    record = complete_record(labels)
    record["sides"]["L"]["joints"]["index_tip"]["xy"] = point
    errors = labels.validate(record)
    assert any("index_tip" in error for error in errors), errors


@pytest.mark.parametrize(
    "box",
    [
        [10, 10, 10, 20],
        [10, 10, 20, 10],
        [20, 10, 10, 20],
        [-1, 0, 10, 10],
        [0, -1, 10, 10],
        [0, 0, 641, 10],
        [0, 0, 10, 481],
    ],
)
@pytest.mark.parametrize("field", ["patient", "others", "extent"])
def test_p01_nonpositive_or_out_of_frame_boxes_are_named(box, field):
    labels = bench("labels")
    record = complete_record(labels)
    if field == "extent":
        record["sides"]["L"]["extent"] = box
    else:
        record[field] = [box] if field == "others" else box
    errors = labels.validate(record)
    assert any(field in error for error in errors), errors


@pytest.mark.parametrize("sides", [(), ("L",), ("R",), ("L", "R", "?"), ("L", "?")])
def test_p01_sides_must_be_exactly_left_and_right(sides):
    labels = bench("labels")
    record = complete_record(labels)
    original = record["sides"]
    record["sides"] = {side: original.get(side, original["L"]) for side in sides}
    errors = labels.validate(record)
    assert any("sides" in error for error in errors), errors


@pytest.mark.parametrize(("hand", "visible"), [("vis", 4), ("part", 0), ("occ", 1), ("oof", 1)])
def test_p01_hand_joint_visibility_consistency(hand, visible):
    labels = bench("labels")
    record = complete_record(labels)
    left = record["sides"]["L"]
    left["hand"] = hand
    left["joints"] = {
        name: {"state": "vis", "xy": [120, 180]} if i < visible else {"state": "occ"}
        for i, name in enumerate(labels.HAND_JOINTS)
    }
    if hand in ("occ", "oof"):
        del left["extent"]
    errors = labels.validate(record)
    assert errors
    assert any("L" in error and ("hand" in error or "joint" in error) for error in errors), errors


def test_p01_validator_accumulates_independent_defects():
    labels = bench("labels")
    record = complete_record(labels)
    del record["sides"]["L"]["extent"]
    record["sides"]["R"]["body"]["elbow"] = {"state": "occ", "xy": [10, 20]}
    record["patient"] = [10, 10, 10, 20]
    record["unexpected_root"] = True
    errors = labels.validate(record)
    for name in ("extent", "elbow", "patient", "unexpected_root"):
        assert any(name in error for error in errors), (name, errors)


def test_p01_generated_jsonl_roundtrips_preserve_canonical_bytes(tmp_path):
    import json

    labels = bench("labels")
    rng = seeded(105)
    records = []
    for _ in range(40):
        record = complete_record(labels)
        record["task"] = rng.choice(labels.TASK_STATES)
        for side in labels.SIDES:
            for group in ("joints", "body"):
                for joint in record["sides"][side][group].values():
                    joint["xy"] = [rng.uniform(110, 240), rng.uniform(160, 290)]
                    joint["own"] = rng.choice(labels.OWNERS)
        assert labels.validate(record) == []
        text = labels.dumps(record)
        assert text == json.dumps(record, sort_keys=True, separators=(",", ":"))
        assert labels.dumps(dict(reversed(list(record.items())))) == text
        records.append(record)
    path = tmp_path / "labels.jsonl"
    labels.write(path, records)
    before = path.read_bytes()
    assert before == ("\n".join(labels.dumps(r) for r in records) + "\n").encode()
    loaded = list(labels.load(path))
    assert len(loaded) == len(records)
    assert [labels.dumps(r) for r in loaded] == [labels.dumps(r) for r in records]
    labels.write(path, loaded)
    assert path.read_bytes() == before


@pytest.mark.parametrize("point", [[0, 0], [639.999, 479.999]])
def test_p01_coordinate_domain_includes_lower_and_excludes_upper_edge(point):
    labels = bench("labels")
    record = complete_record(labels)
    record["patient"] = [0, 0, 640, 480]
    record["sides"]["L"]["extent"] = [0, 0, 640, 480]
    record["sides"]["L"]["joints"]["index_tip"]["xy"] = point
    assert labels.validate(record) == []
    for outside in ([640, 20], [20, 480]):
        record["sides"]["L"]["joints"]["index_tip"]["xy"] = outside
        assert labels.validate(record)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("frame", "key"), ""),
        (("frame", "index"), -1),
        (("frame", "width"), 0),
        (("frame", "height"), 0),
        (("frame", "index"), 0.5),
        (("frame", "width"), 2.5),
        (("labeller",), ""),
        (("note",), 23),
    ],
)
def test_p01_a02_metadata_types_and_nonempty_fields(path, value):
    labels = bench("labels")
    record = complete_record(labels)
    target = record
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    errors = labels.validate(record)
    assert any(path[-1] in error for error in errors), errors


def test_p01_optional_others_and_note_are_not_required():
    labels = bench("labels")
    record = complete_record(labels)
    del record["others"]
    assert labels.validate(record) == []
    record["note"] = ""
    assert labels.validate(record) == []


@pytest.mark.parametrize("group", ["joints", "body"])
def test_p01_each_required_joint_is_present(group):
    labels = bench("labels")
    names = labels.HAND_JOINTS if group == "joints" else labels.BODY_JOINTS
    assert names
    for side in labels.SIDES:
        for name in names:
            record = complete_record(labels)
            del record["sides"][side][group][name]
            errors = labels.validate(record)
            assert any(name in error for error in errors), (side, name, errors)


@pytest.mark.parametrize("box", [[np.nan, 0, 10, 10], [0, 0, np.inf, 10]])
def test_p01_nonfinite_boxes_are_outside_the_coordinate_domain(box):
    labels = bench("labels")
    record = complete_record(labels)
    record["patient"] = box
    errors = labels.validate(record)
    assert any("patient" in error for error in errors), errors

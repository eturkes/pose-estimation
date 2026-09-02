"""M2.8.2 predicates for the full corpus 2D run.

Written diff-blind against the frozen acceptance contract, never against the
implementation, so the cases cover what an author's own reading would skip.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from rtmlib import PoseTracker

from pose_estimation import corpus_run as corpus_run_module
from pose_estimation import run as run_module
from pose_estimation import sessions as sessions_module
from pose_estimation import video_io
from pose_estimation.multicam import Session, SessionError, _resolve_session_output

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── §2 stimulus ─────────────────────────────────────────────────────
_FRAMES = 140
_DET_FREQUENCY = 7
_STABLE = ((10.0, 10.0), (110.0, 210.0))
_JUMPED = ((1000.0, 1000.0), (1100.0, 1200.0))
# §2's cost model: the detector runs only on detector frames, the pose model on
# every frame.  It is what turns the counters below into the measured bands.
_DET_MS = 350.0
_POSE_MS = 8.0


def _person(extent: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """Four keypoints spanning `extent`.

    ``pose_to_bbox`` expands by 1.25, so a 100x200 span clears
    ``PoseTracker.MIN_AREA`` (1000) 31x over.  Without that margin ``track_by_iou``
    returns -1 and never mints the id whose use as an index is the whole defect.
    """
    (x0, y0), (x1, y1) = extent
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _script(frames: int, miss_at: int | None) -> list[np.ndarray]:
    """One person holding still, except for a single frame that jumps out of IoU range."""
    return [_person(_JUMPED if index == miss_at else _STABLE) for index in range(frames)]


class _StubDetModel:
    # PoseTracker.__init__ reads `.mode`; anything but 'multiclass' takes the
    # single-category branch that the shipped YOLOX also takes.
    mode = "balanced"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _image: np.ndarray) -> list[list[float]]:
        self.calls += 1
        return [[10.0, 10.0, 110.0, 210.0, 0.9]]


class _StubPoseModel:
    """Counts the whole-frame fallback `RTMPose.__call__` takes on an empty bbox list."""

    def __init__(self, script: list[np.ndarray]) -> None:
        self.script = script
        self.calls = 0
        self.whole_frame_calls = 0

    def __call__(self, _image: np.ndarray, bboxes: Any = ()) -> tuple[np.ndarray, np.ndarray]:
        if len(bboxes) == 0:
            self.whole_frame_calls += 1
        keypoints = self.script[self.calls]
        self.calls += 1
        return np.array([keypoints]), np.ones((1, len(keypoints)))


def _solution_factory(script: list[np.ndarray]) -> type:
    """A SplitDeviceSolution stand-in, so the real PoseTracker.__init__ runs unmodified."""

    class _StubSolution:
        def __init__(self, **_kwargs: Any) -> None:
            self.det_model = _StubDetModel()
            self.pose_model = _StubPoseModel(script)
            self.det_categories = None
            self.one_stage = False

    return _StubSolution


def _shipped_tracker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    *,
    script: list[np.ndarray],
    det_frequency: int = _DET_FREQUENCY,
) -> PoseTracker:
    """Return the tracker `run.py` itself builds, with the two models stubbed.

    Capturing the object at `_dispatch_sessions` instead of reading the
    constructor call keeps the predicate agnostic to how the fix is spelled: a
    keyword, a post-construction assignment and a wrapper all reach this object.
    """
    captured: dict[str, Any] = {}
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_module, "SplitDeviceSolution", _solution_factory(script))
    monkeypatch.setattr(
        run_module, "_dispatch_sessions", lambda _args, **kwargs: captured.update(kwargs)
    )

    run_module.main(
        [
            "--session-dir",
            str(session_dir),
            "--headless",
            "--backend",
            "onnxruntime",
            "--det-frequency",
            str(det_frequency),
        ]
    )

    tracker = captured["pose_tracker"]
    assert isinstance(tracker, PoseTracker), "the runner must build a real rtmlib PoseTracker"
    return tracker


def _explicit_tracker(
    *, tracking: bool, det_frequency: int, script: list[np.ndarray]
) -> PoseTracker:
    """A tracker with `tracking` forced, bypassing the model-building __init__."""
    tracker = object.__new__(PoseTracker)
    tracker.det_model = _StubDetModel()
    tracker.pose_model = _StubPoseModel(script)
    tracker.det_categories = None
    tracker.det_mode = "balanced"
    tracker.det_frequency = det_frequency
    tracker.tracking = tracking
    tracker.tracking_thr = 0.3
    tracker.reset()
    return tracker


def _drive(tracker: PoseTracker, frames: int) -> SimpleNamespace:
    """Feed `frames` frames and report the counters §2's table is written in."""
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame_cnt_trace = []
    for _ in range(frames):
        tracker(image)
        frame_cnt_trace.append(tracker.frame_cnt)
    det_model = cast(_StubDetModel, tracker.det_model)
    pose_model = cast(_StubPoseModel, tracker.pose_model)
    det_calls = det_model.calls
    return SimpleNamespace(
        frame_cnt=tracker.frame_cnt,
        frame_cnt_trace=frame_cnt_trace,
        det_calls=det_calls,
        whole_frame_calls=pose_model.whole_frame_calls,
        pose_calls=pose_model.calls,
        bboxes_last_frame=len(tracker.bboxes_last_frame),
        ms_per_frame=(det_calls * _DET_MS + frames * _POSE_MS) / frames,
    )


def test_p01_runner_constructs_the_tracker_with_iou_tracking_disabled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _shipped_tracker(monkeypatch, tmp_path, script=_script(_FRAMES, None))

    assert tracker.tracking is False, "P01: run.py must construct PoseTracker(tracking=False) (D01)"


# (label, tracking, miss_at, frame_cnt, det_calls, whole_frame_calls, ms_per_frame)
_REPRODUCTION = (
    ("no-miss-tracking-on", True, None, 140, 20, 0, 58.0),
    ("miss-residue-nonzero", True, 3, 3, 1, 135, 10.5),
    ("miss-residue-zero", True, 7, 7, 134, 0, 343.0),
    ("miss-tracking-off", False, 3, 140, 20, 0, 58.0),
)


@pytest.mark.parametrize(
    ("tracking", "miss_at", "frame_cnt", "det_calls", "whole_frame_calls", "ms_per_frame"),
    [row[1:] for row in _REPRODUCTION],
    ids=[row[0] for row in _REPRODUCTION],
)
def test_p02_upstream_freeze_reproduces_the_frozen_scenario_table(
    tracking: bool,
    miss_at: int | None,
    frame_cnt: int,
    det_calls: int,
    whole_frame_calls: int,
    ms_per_frame: float,
) -> None:
    tracker = _explicit_tracker(
        tracking=tracking, det_frequency=_DET_FREQUENCY, script=_script(_FRAMES, miss_at)
    )

    result = _drive(tracker, _FRAMES)

    assert result.pose_calls == _FRAMES, "P02: every frame must reach the pose model"
    assert (result.frame_cnt, result.det_calls, result.whole_frame_calls) == (
        frame_cnt,
        det_calls,
        whole_frame_calls,
    ), "P02: §2's reproduction table is frozen"
    assert result.ms_per_frame == pytest.approx(ms_per_frame)


def test_p02_the_freeze_is_permanent_and_disabling_tracking_removes_it() -> None:
    miss_at = 3

    frozen = _drive(
        _explicit_tracker(
            tracking=True, det_frequency=_DET_FREQUENCY, script=_script(_FRAMES, miss_at)
        ),
        _FRAMES,
    )
    healthy = _drive(
        _explicit_tracker(
            tracking=False, det_frequency=_DET_FREQUENCY, script=_script(_FRAMES, miss_at)
        ),
        _FRAMES,
    )

    assert frozen.frame_cnt_trace[:miss_at] == list(range(1, miss_at + 1))
    assert set(frozen.frame_cnt_trace[miss_at:]) == {miss_at}, "P02: the freeze is permanent"
    assert frozen.bboxes_last_frame == 0, "P02: the frozen bbox list drains through the IoU pops"
    assert healthy.frame_cnt_trace == list(range(1, _FRAMES + 1)), "P02: tracking=False advances"
    assert healthy.bboxes_last_frame == 1, "P02: tracking=False keeps refreshing the bbox list"


@pytest.mark.parametrize("miss_at", [3, 7], ids=["residue-nonzero", "residue-zero"])
def test_p02_shipped_construction_is_immune_to_the_freeze(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, miss_at: int
) -> None:
    tracker = _shipped_tracker(monkeypatch, tmp_path, script=_script(_FRAMES, miss_at))

    result = _drive(tracker, _FRAMES)

    assert result.pose_calls == _FRAMES, "P02: non-vacuity — the stimulus must have run"
    assert result.frame_cnt == _FRAMES, "P02: N1 must fire here — the shipped tracker cannot freeze"


@pytest.mark.parametrize(
    "miss_at", [None, 3, 7], ids=["no-miss", "residue-nonzero", "residue-zero"]
)
def test_p03_shipped_construction_never_starves_the_pose_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, miss_at: int | None
) -> None:
    tracker = _shipped_tracker(monkeypatch, tmp_path, script=_script(_FRAMES, miss_at))

    result = _drive(tracker, _FRAMES)

    assert result.pose_calls == _FRAMES, "P03: non-vacuity — the pose model must see every frame"
    assert result.whole_frame_calls == 0, (
        "P03: an empty bbox list makes RTMPose estimate over the whole 1080p frame"
    )


@pytest.mark.parametrize("det_frequency", [1, 2, 5, 7, 13])
@pytest.mark.parametrize("miss_at", [None, 3, 7], ids=["no-miss", "miss-at-3", "miss-at-7"])
def test_p04_shipped_detector_cadence_is_frames_over_det_frequency(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    det_frequency: int,
    miss_at: int | None,
) -> None:
    tracker = _shipped_tracker(
        monkeypatch, tmp_path, script=_script(_FRAMES, miss_at), det_frequency=det_frequency
    )

    result = _drive(tracker, _FRAMES)

    assert result.pose_calls == _FRAMES, "P04: non-vacuity — the stimulus must have run"
    assert result.det_calls == math.ceil(_FRAMES / det_frequency), (
        "P04: the detector cadence must not depend on whether an IoU match was missed"
    )


@pytest.mark.parametrize("miss_at", [None, 3], ids=["no-miss", "miss"])
def test_p04_det_frequency_one_is_safe_even_with_iou_tracking_restored(
    miss_at: int | None,
) -> None:
    """D01a: `--det-frequency 1` re-enters the tracking branch, and a freeze there costs nothing."""
    result = _drive(
        _explicit_tracker(tracking=True, det_frequency=1, script=_script(_FRAMES, miss_at)), _FRAMES
    )

    assert result.det_calls == _FRAMES, "D01a: residue 0 holds at every frame, so the detector runs"
    assert result.whole_frame_calls == 0, "D01a: a fresh detection every frame cannot starve"


# ── D05 resume policy (P05, P06) ────────────────────────────────────
# D05 fixes the marker's semantics and not its name or location; this suite
# picks one so the policy is executable, and the register carries that gap.
_MARKER_NAME = "event_complete.json"

_Selector = Callable[[pathlib.Path, Sequence[str]], list[str]]


def _snapshot(root: pathlib.Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _marker_selector(tree: pathlib.Path, event_ids: Sequence[str]) -> list[str]:
    """D05: an event is due exactly when its completion marker is absent."""
    return [event_id for event_id in event_ids if not (tree / event_id / _MARKER_NAME).is_file()]


def _output_presence_selector(tree: pathlib.Path, event_ids: Sequence[str]) -> list[str]:
    """The policy D05 forbids: an event reads as done because it left output behind."""
    return [event_id for event_id in event_ids if not any((tree / event_id).glob("*.csv"))]


class _Driver:
    """Resume over a synthetic run tree, counting the inference a real pass would spend."""

    def __init__(self, tree: pathlib.Path, selector: _Selector) -> None:
        self.tree = tree
        self.selector = selector
        self.inferences = 0

    def run(self, event_ids: Sequence[str], *, interrupted: Iterable[str] = ()) -> list[str]:
        ran = []
        for event_id in self.selector(self.tree, event_ids):
            event_dir = self.tree / event_id
            event_dir.mkdir(parents=True, exist_ok=True)
            (event_dir / "cam-a.csv").write_text(f"frame\n{event_id}\n", encoding="utf-8")
            self.inferences += 1
            ran.append(event_id)
            if event_id in interrupted:
                continue
            (event_dir / _MARKER_NAME).write_text("{}\n", encoding="utf-8")
        return ran


_EVENTS = ("event-01", "event-02", "event-03")


def test_p05_resume_over_a_complete_tree_runs_nothing_and_rewrites_nothing(
    tmp_path: pathlib.Path,
) -> None:
    driver = _Driver(tmp_path / "run", _marker_selector)

    first = driver.run(_EVENTS)
    complete = _snapshot(driver.tree)
    second = driver.run(_EVENTS)

    assert first == list(_EVENTS), "P05: non-vacuity — the first pass must cover the whole tree"
    assert driver.inferences == len(_EVENTS)
    assert second == [], "P05: a fully marked tree resumes with zero inference"
    assert driver.inferences == len(_EVENTS), "P05: the second pass must spend no inference"
    assert _snapshot(driver.tree) == complete, "P05: resume leaves every output byte-identical"


def test_p06_an_event_interrupted_before_its_marker_is_rerun(tmp_path: pathlib.Path) -> None:
    driver = _Driver(tmp_path / "run", _marker_selector)
    interrupted = "event-02"

    driver.run(_EVENTS, interrupted=(interrupted,))
    partial = driver.tree / interrupted
    assert partial.joinpath("cam-a.csv").is_file(), "the partial output must exist to be a trap"
    assert not partial.joinpath(_MARKER_NAME).is_file()

    resumed = driver.run(_EVENTS)

    assert resumed == [interrupted], "P06: resume re-runs exactly the unmarked event"
    assert partial.joinpath(_MARKER_NAME).is_file()


def test_p06_n3_deleting_one_marker_after_a_full_run_reselects_that_event(
    tmp_path: pathlib.Path,
) -> None:
    driver = _Driver(tmp_path / "run", _marker_selector)
    target = "event-03"

    driver.run(_EVENTS)
    (driver.tree / target / _MARKER_NAME).unlink()

    assert driver.run(_EVENTS) == [target], "N3: a deleted marker must re-select its event"


def test_p06_output_presence_and_marker_policies_disagree_on_the_interrupted_event(
    tmp_path: pathlib.Path,
) -> None:
    """D05's rationale, made executable: the two policies differ on exactly this case."""
    driver = _Driver(tmp_path / "run", _marker_selector)
    interrupted = "event-02"

    driver.run(_EVENTS, interrupted=(interrupted,))

    assert _marker_selector(driver.tree, _EVENTS) == [interrupted]
    assert interrupted not in _output_presence_selector(driver.tree, _EVENTS), (
        "P06 discriminates only because output presence credits the partial event"
    )


# ── D06 manifest totality (P07, P08, P09) ───────────────────────────
_CANONICAL_ASSETS = 379
# A06: the vocabulary is read from the one published constant, never restated,
# so a code added or renamed there reaches these cases without an edit.
_DISPOSITION_OK = corpus_run_module.DISPOSITION_OK
_DISPOSITIONS = frozenset(corpus_run_module.ASSET_DISPOSITIONS)
_MANIFEST_HEADER = ("asset_id", "disposition")


def _canonical_ids(count: int = _CANONICAL_ASSETS) -> tuple[str, ...]:
    return tuple(f"asset-{index:04d}" for index in range(count))


def _manifest(ids: Sequence[str], *, disposition: str = _DISPOSITION_OK) -> list[dict[str, str]]:
    return [{"asset_id": asset_id, "disposition": disposition} for asset_id in ids]


def _manifest_verdict(
    rows: Sequence[dict[str, str]], canonical: Sequence[str], codes: frozenset[str]
) -> SimpleNamespace:
    """P07 + P08 in one pass over a manifest."""
    asset_ids = [row["asset_id"] for row in rows]
    counts = Counter(row["disposition"] for row in rows)
    return SimpleNamespace(
        rows=len(rows),
        total=set(asset_ids) == set(canonical),
        unique=len(asset_ids) == len(set(asset_ids)),
        vocabulary=set(counts) <= set(codes),
        counts_sum=sum(counts.values()) == len(rows),
        # An empty manifest over an empty registry satisfies every clause above.
        # This project has shipped that defect twice, so it is a named clause.
        nonvacuous=bool(rows) and bool(canonical),
    )


def test_p07_a_total_manifest_carries_one_row_per_canonical_asset() -> None:
    canonical = _canonical_ids()
    verdict = _manifest_verdict(_manifest(canonical), canonical, _DISPOSITIONS)

    assert verdict.rows == _CANONICAL_ASSETS, "P07: the corpus manifest is 379 rows"
    assert (verdict.total, verdict.unique, verdict.nonvacuous) == (True, True, True)


@pytest.mark.parametrize(
    ("mutation", "expected_total", "expected_unique"),
    # A duplicated row leaves the id SET equal to the registry, so `total` alone
    # cannot see it — the separate `unique` clause is what catches a double count.
    [("drop-one", False, True), ("duplicate-one", True, False), ("unregistered", False, True)],
)
def test_p07_n5_the_manifest_check_catches_every_row_set_mutation(
    mutation: str, expected_total: bool, expected_unique: bool
) -> None:
    canonical = _canonical_ids(12)
    rows = _manifest(canonical)
    if mutation == "drop-one":
        rows.pop(3)
    elif mutation == "duplicate-one":
        rows.append(dict(rows[3]))
    else:
        rows.append({"asset_id": "asset-9999", "disposition": _DISPOSITION_OK})

    verdict = _manifest_verdict(rows, canonical, _DISPOSITIONS)

    assert verdict.total is expected_total, f"N5: {mutation} must break manifest totality"
    assert verdict.unique is expected_unique
    assert not (verdict.total and verdict.unique), f"N5: {mutation} must fail at least one clause"


def test_p08_dispositions_partition_the_asset_set() -> None:
    canonical = _canonical_ids(40)
    codes = sorted(_DISPOSITIONS)
    rows = [
        {"asset_id": asset_id, "disposition": codes[index % len(codes)]}
        for index, asset_id in enumerate(canonical)
    ]

    verdict = _manifest_verdict(rows, canonical, _DISPOSITIONS)
    counts = Counter(row["disposition"] for row in rows)

    assert verdict.vocabulary is True
    assert verdict.counts_sum is True
    assert sum(counts.values()) == len(rows) == len(canonical)
    assert len(counts) == len(codes), "P08: non-vacuity — every frozen code must be exercised"


def test_p08_n6_an_unlisted_disposition_code_is_refused() -> None:
    canonical = _canonical_ids(4)
    rows = _manifest(canonical)
    rows[1]["disposition"] = "invented_code"

    verdict = _manifest_verdict(rows, canonical, _DISPOSITIONS)

    assert verdict.vocabulary is False, "N6: an unlisted code must fail the partition"
    assert verdict.counts_sum is True, (
        "N6: counts still sum — vocabulary is the discriminating clause"
    )


def test_p08_an_empty_manifest_satisfies_every_partition_clause_and_must_still_fail() -> None:
    """The vacuity trap D06 exists to prevent: a green partition over zero assets."""
    verdict = _manifest_verdict([], [], _DISPOSITIONS)

    assert (verdict.total, verdict.unique, verdict.vocabulary, verdict.counts_sum) == (
        True,
        True,
        True,
        True,
    )
    assert verdict.nonvacuous is False, "an empty manifest must never read as a total partition"


def test_p08_the_frozen_disposition_vocabulary_is_published_in_a_machine_readable_constant() -> (
    None
):
    """D06 freezes a vocabulary; a set no source publishes cannot be checked against.

    The scan admits an annotated assignment, because that is this repo's dominant
    constant idiom (`SOURCE_DIAGNOSTIC_FIELDS: tuple[str, ...] = ...`) and a
    pattern refusing it reports an absent constant that is present.  Finding the
    name is then only a spelling test, so the published set is also read: A06
    requires the writer and the validator to read one constant, which a comment
    carrying the right letters would satisfy otherwise.
    """
    pattern = re.compile(r"^[A-Z][A-Z0-9_]*DISPOSITION[A-Z0-9_]*\s*(?::[^=\n]+)?=", re.M)
    sources = sorted(_PROJECT_ROOT.glob("src/**/*.py")) + sorted(_PROJECT_ROOT.glob("scripts/*.py"))
    assert sources, "non-vacuity: the scan must reach real source files"
    published = [
        path.relative_to(_PROJECT_ROOT).as_posix()
        for path in sources
        if pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert published, (
        "P08: no source declares the frozen manifest disposition set, so no gate can "
        "re-derive it the way scripts/pilot_corpus_run.py re-derives GROUP_QC_REASONS"
    )
    codes = corpus_run_module.ASSET_DISPOSITIONS
    assert len(set(codes)) == len(codes) > 1, "P08: the published vocabulary must be a real set"
    assert corpus_run_module.DISPOSITION_OK in codes
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", code) for code in codes), (
        "P08: a disposition code must read as a code-authored token, so a manifest "
        "census publishes under D07's allowlist redaction without an exemption"
    )


def _shipped_rows(
    ids: Sequence[str], *, disposition: str = _DISPOSITION_OK
) -> list[dict[str, str]]:
    return [
        {"asset_id": asset_id, "event_id": "e", "camera_name": "c", "disposition": disposition}
        for asset_id in ids
    ]


def test_p07_p08_the_shipped_validator_accepts_a_total_manifest() -> None:
    """The stand-in cases above fix the policy; this one binds it to shipped code."""
    canonical = _canonical_ids()
    census = corpus_run_module.validate_manifest(_shipped_rows(canonical), canonical)

    assert sum(census.values()) == _CANONICAL_ASSETS
    assert set(census) == set(corpus_run_module.ASSET_DISPOSITIONS), (
        "P08: the census names every frozen code, so a zero count is published rather than absent"
    )


@pytest.mark.parametrize(
    "mutation", ["drop-one", "duplicate-one", "unregistered", "unlisted-code", "empty"]
)
def test_p07_p08_the_shipped_validator_refuses_every_manifest_mutation(mutation: str) -> None:
    canonical = _canonical_ids(12)
    rows = _shipped_rows(canonical)
    if mutation == "drop-one":
        rows.pop(3)
    elif mutation == "duplicate-one":
        rows.append(dict(rows[3]))
        rows.pop(4)  # keep the row COUNT correct, so only uniqueness can see it
    elif mutation == "unregistered":
        rows[3]["asset_id"] = "asset-9999"
    elif mutation == "unlisted-code":
        rows[3]["disposition"] = "invented_code"
    else:
        rows = []

    with pytest.raises(corpus_run_module.ManifestError):
        corpus_run_module.validate_manifest(rows, canonical)


def _write_diagnostics(tree: pathlib.Path, asset_id: str, rows: int) -> None:
    path = tree / f"{asset_id}_diag.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_module.SOURCE_DIAGNOSTIC_FIELDS)
        writer.writeheader()
        for index in range(rows):
            writer.writerow(
                dict.fromkeys(run_module.SOURCE_DIAGNOSTIC_FIELDS, "0")
                | {"video": f"{asset_id}-{index}"}
            )


def _build_run_tree(tree: pathlib.Path, rows: Sequence[dict[str, str]]) -> None:
    tree.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["disposition"] != _DISPOSITION_OK:
            continue
        (tree / f"{row['asset_id']}.csv").write_text("frame\n0\n", encoding="utf-8")
        _write_diagnostics(tree, row["asset_id"], 1)


def _diagnostic_rows(tree: pathlib.Path, asset_id: str) -> int:
    path = tree / f"{asset_id}_diag.csv"
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return len(list(csv.DictReader(handle)))


def _artifact_verdict(tree: pathlib.Path, rows: Sequence[dict[str, str]]) -> SimpleNamespace:
    """P09: `ok` assets own exactly one landmark CSV and one diagnostics row; others own neither."""
    ok_ids = {row["asset_id"] for row in rows if row["disposition"] == _DISPOSITION_OK}
    other_ids = {row["asset_id"] for row in rows} - ok_ids
    return SimpleNamespace(
        ok=len(ok_ids),
        other=len(other_ids),
        missing_csv=sorted(i for i in ok_ids if not (tree / f"{i}.csv").is_file()),
        wrong_diag=sorted(i for i in ok_ids if _diagnostic_rows(tree, i) != 1),
        trespass=sorted(
            i for i in other_ids if (tree / f"{i}.csv").is_file() or _diagnostic_rows(tree, i) > 0
        ),
    )


def _mixed_manifest() -> list[dict[str, str]]:
    canonical = _canonical_ids(10)
    codes = sorted(_DISPOSITIONS - {_DISPOSITION_OK})
    return [
        {
            "asset_id": asset_id,
            "disposition": _DISPOSITION_OK if index < 6 else codes[index % len(codes)],
        }
        for index, asset_id in enumerate(canonical)
    ]


def test_p09_ok_assets_own_their_artifacts_and_no_other_asset_claims_one(
    tmp_path: pathlib.Path,
) -> None:
    rows = _mixed_manifest()
    tree = tmp_path / "run"
    _build_run_tree(tree, rows)

    verdict = _artifact_verdict(tree, rows)

    assert verdict.ok > 0, "P09: non-vacuity — the ok side must be populated"
    assert verdict.other > 0, "P09: non-vacuity — the non-ok side must be populated"
    assert (verdict.missing_csv, verdict.wrong_diag, verdict.trespass) == ([], [], [])


@pytest.mark.parametrize("mutation", ["delete-csv", "duplicate-diag-row", "non-ok-claims-csv"])
def test_p09_every_artifact_mutation_is_caught(tmp_path: pathlib.Path, mutation: str) -> None:
    rows = _mixed_manifest()
    tree = tmp_path / "run"
    _build_run_tree(tree, rows)
    ok_id = next(row["asset_id"] for row in rows if row["disposition"] == _DISPOSITION_OK)
    other_id = next(row["asset_id"] for row in rows if row["disposition"] != _DISPOSITION_OK)

    if mutation == "delete-csv":
        (tree / f"{ok_id}.csv").unlink()
    elif mutation == "duplicate-diag-row":
        _write_diagnostics(tree, ok_id, 2)
    else:
        (tree / f"{other_id}.csv").write_text("frame\n0\n", encoding="utf-8")

    verdict = _artifact_verdict(tree, rows)

    caught = verdict.missing_csv + verdict.wrong_diag + verdict.trespass
    assert caught, f"P09: {mutation} must break the artifact claim"


# ── P10 group disposition at the corpus grain (M2.8.1 D05) ──────────
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"
# The contract freezes this header; R's own declaration is what must agree.
_GROUP_QC_HEADER = ("video", "person_idx", "n_frames", "drop_reason", "qc_status")
_GROUP_QC_REASON_COUNT = 6
_VIDEO = "subject90"


def _r_source() -> str:
    return _CLINICAL_R.read_text(encoding="utf-8")


def _r_group_qc_constants() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Re-derive the disposition header and reason codes from the R source.

    Transcribing either would make the case agree with itself: the artifact is
    R's, so R's own declaration is the only oracle.  `scripts/pilot_corpus_run.py`
    reads them with the same two patterns, which is the idiom a corpus-grain
    check inherits.
    """
    source = _r_source()
    reasons = re.search(r"GROUP_QC_REASONS <- c\((.*?)\)", source, re.S)
    schema = re.search(r"group_qc_schema <- function\(\) \{\s*tibble\((.*?)\)\s*\}", source, re.S)
    assert reasons is not None, "the R source must declare GROUP_QC_REASONS"
    assert schema is not None, "the R source must declare group_qc_schema()"
    return (
        tuple(re.findall(r"(\w+)\s*=\s*\w+\(\)", schema.group(1))),
        tuple(re.findall(r'"([a-z_]+)"', reasons.group(1))),
    )


def _r_landmark_filter() -> re.Pattern[str]:
    """R's own directory filter, translated out of R's string escaping.

    A corpus check that re-invents this filter drifts from R and then demands a
    disposition artifact for R's own output files.
    """
    block = re.search(
        r"str_detect\(\s*basename\(files\),\s*paste0\((.*?)\)\s*\)", _r_source(), re.S
    )
    assert block is not None, "the R landmark-CSV filter must parse"
    return re.compile("".join(re.findall(r'"([^"]*)"', block.group(1))).replace("\\\\", "\\"))


def _r_world3d_marker() -> str:
    """The column suffix R reads an input's mode from, which selects the artifact name."""
    marker = re.search(
        r'is_world3d <- function\(cols\) \{\s*any\(str_ends\(cols, "([^"]+)"', _r_source()
    )
    assert marker is not None, "the R 3D-input test must parse"
    return marker.group(1)


def _r_main_loop_prefix() -> str:
    """The main loop up to the disposition write — every exit in here skips the artifact."""
    source = _r_source()
    start = source.index("for (f in files) {")
    return source[start : source.index("out_group_qc <- paste0(", start)]


def _write_landmarks(tree: pathlib.Path, stem: str, *, is_3d: bool = False) -> pathlib.Path:
    column = f"wrist{_r_world3d_marker()}" if is_3d else "wrist_x"
    path = tree / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"video,person_idx,{column}\n{_VIDEO},0,0.0\n", encoding="utf-8")
    return path


def _disposition_path(landmarks: pathlib.Path, *, is_3d: bool) -> pathlib.Path:
    return landmarks.with_name(f"{landmarks.stem}_clinical{'_3d' if is_3d else ''}_group_qc.csv")


def _write_disposition(
    path: pathlib.Path, rows: Sequence[Sequence[str]], header: Sequence[str] = _GROUP_QC_HEADER
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _disposition_verdict(
    tree: pathlib.Path, header: Sequence[str], codes: Sequence[str]
) -> SimpleNamespace:
    """P10 over a run tree: every landmark CSV R would read owns one artifact."""
    excluded = _r_landmark_filter()
    marker = _r_world3d_marker()
    reason_column = list(header).index("drop_reason")
    inputs = [path for path in sorted(tree.glob("*.csv")) if not excluded.search(path.name)]
    verdict = SimpleNamespace(
        inputs=len(inputs),
        missing=[],
        wrong_header=[],
        unlisted=[],
        header_only=0,
        populated=0,
    )
    for path in inputs:
        with path.open(newline="", encoding="utf-8") as handle:
            is_3d = any(name.endswith(marker) for name in next(csv.reader(handle), []))
        artifact = _disposition_path(path, is_3d=is_3d)
        if not artifact.is_file():
            verdict.missing.append(path.name)
            continue
        with artifact.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if (tuple(rows[0]) if rows else ()) != tuple(header):
            verdict.wrong_header.append(artifact.name)
            continue
        body = rows[1:]
        verdict.populated += int(bool(body))
        verdict.header_only += int(not body)
        verdict.unlisted.extend(
            row[reason_column] for row in body if row[reason_column] not in codes
        )
    verdict.covered = not (verdict.missing or verdict.wrong_header or verdict.unlisted)
    # An event whose every asset failed before R publishes no artifact and no
    # input, and would otherwise read as full coverage.  Same trap as P08's.
    verdict.nonvacuous = bool(inputs)
    return verdict


def test_p10_the_r_source_publishes_the_frozen_disposition_header_and_reason_codes() -> None:
    header, codes = _r_group_qc_constants()

    assert header == _GROUP_QC_HEADER, "P10: the group-disposition header is frozen"
    assert len(codes) == _GROUP_QC_REASON_COUNT, "P10: the reason vocabulary is 6 codes"
    assert len(set(codes)) == len(codes)
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", code) for code in codes), (
        "P10: a reason code is an emitted string, so it must read as a code (P13)"
    )


def test_p10_every_declared_reason_code_has_a_drop_site_and_every_site_a_declared_code() -> None:
    """A frozen code no site can emit is dead vocabulary; a site with an unlisted code is a leak."""
    _, codes = _r_group_qc_constants()
    sites = re.findall(r'record_drop\([^)]*"([a-z_]+)"\s*\)', _r_source())

    assert sites, "non-vacuity: the scan must reach real drop sites"
    assert set(sites) == set(codes), "P10: the drop sites and the frozen vocabulary must agree"
    assert len(sites) == len(codes), "P10: each code is emitted from exactly one site"


def test_p10_every_landmark_csv_in_a_run_tree_owns_one_disposition_artifact(
    tmp_path: pathlib.Path,
) -> None:
    header, codes = _r_group_qc_constants()
    tree = tmp_path / "run"
    quiet = _write_landmarks(tree, "asset-0000")
    dropped = _write_landmarks(tree, "asset-0001")
    volumetric = _write_landmarks(tree, "asset-0002", is_3d=True)
    # R's own outputs share the directory; the filter is what keeps them out.
    (tree / "asset-0000_clinical.csv").write_text("video\n", encoding="utf-8")
    (tree / "asset-0000_diag.csv").write_text("video\n", encoding="utf-8")
    _write_disposition(_disposition_path(quiet, is_3d=False), [])
    _write_disposition(
        _disposition_path(dropped, is_3d=False),
        [[_VIDEO, "0", "3", codes[0], "dropped"], [_VIDEO, "1", "9", codes[-1], "dropped"]],
    )
    _write_disposition(_disposition_path(volumetric, is_3d=True), [])

    verdict = _disposition_verdict(tree, header, codes)

    assert verdict.inputs == 3, "P10: R's filter must admit the landmark CSVs and nothing else"
    assert verdict.nonvacuous is True
    assert verdict.covered is True
    assert verdict.header_only == 2, "P10: a header-only artifact satisfies the predicate"
    assert verdict.populated == 1, "P10: non-vacuity — a populated artifact must occur too"


@pytest.mark.parametrize(
    "mutation", ["delete-artifact", "wrong-mode-suffix", "unlisted-reason", "renamed-column"]
)
def test_p10_every_disposition_mutation_is_caught(tmp_path: pathlib.Path, mutation: str) -> None:
    header, codes = _r_group_qc_constants()
    tree = tmp_path / "run"
    flat = _write_landmarks(tree, "asset-0000")
    volumetric = _write_landmarks(tree, "asset-0001", is_3d=True)
    _write_disposition(
        _disposition_path(flat, is_3d=False), [[_VIDEO, "0", "3", codes[0], "dropped"]]
    )
    _write_disposition(_disposition_path(volumetric, is_3d=True), [])

    if mutation == "delete-artifact":
        _disposition_path(flat, is_3d=False).unlink()
    elif mutation == "wrong-mode-suffix":
        artifact = _disposition_path(volumetric, is_3d=True)
        artifact.rename(_disposition_path(volumetric, is_3d=False))
    elif mutation == "unlisted-reason":
        _write_disposition(
            _disposition_path(flat, is_3d=False), [[_VIDEO, "0", "3", "invented_reason", "dropped"]]
        )
    else:
        _write_disposition(
            _disposition_path(flat, is_3d=False),
            [[_VIDEO, "0", "3", codes[0], "dropped"]],
            header=("video", "person_idx", "n_frames", "reason", "qc_status"),
        )

    verdict = _disposition_verdict(tree, header, codes)

    assert verdict.covered is False, f"P10: {mutation} must break disposition coverage"


def test_p10_an_empty_run_tree_never_reads_as_full_coverage(tmp_path: pathlib.Path) -> None:
    header, codes = _r_group_qc_constants()
    tree = tmp_path / "run"
    tree.mkdir(parents=True)

    verdict = _disposition_verdict(tree, header, codes)

    assert verdict.covered is True, "every clause is vacuously satisfied over zero inputs"
    assert verdict.nonvacuous is False, "P10: zero inputs must never read as coverage"


def test_p10_no_input_leaves_the_r_main_loop_before_the_disposition_write() -> None:
    """P10's scope term decides the predicate, and the contract never defines it.

    Read as "every CSV R was handed", P10 is false of the shipped R: the main
    loop reaches the disposition write past a `next` and a `stop(`.  Read as
    "every CSV that got past those gates", P10 is true but cannot see either
    escape — and `stop()` ends the whole Rscript, so at directory grain it also
    takes every not-yet-opened CSV with it.  MAIN rules which reading binds.
    """
    prefix = _r_main_loop_prefix()
    exits = re.findall(r"^\s*(next|stop\()", prefix, re.M)

    assert exits == [], (
        "P10: an input that enters R's loop must reach the disposition write; "
        f"{len(exits)} pre-write exit(s) skip it and publish no artifact at all"
    )


def test_p10_a_rejected_input_must_not_void_the_remaining_inputs(tmp_path: pathlib.Path) -> None:
    """Directory-grain invocation makes one rejected asset a corpus-scale silent loss.

    `stop()` ends the process, not the file, so the assets after it in the same
    directory are never opened and publish nothing.  Nothing distinguishes that
    from an event that produced no drop.  Stand-in for the per-asset isolation
    MAIN must supply, since no corpus driver exists to test.
    """
    header, codes = _r_group_qc_constants()
    tree = tmp_path / "run"
    stems = [f"asset-{index:04d}" for index in range(5)]
    landmarks = [_write_landmarks(tree, stem) for stem in stems]
    rejected = 1

    for index, path in enumerate(landmarks):
        if index == rejected:
            break  # stop() aborts the invocation; indices 2..4 are never opened
        _write_disposition(_disposition_path(path, is_3d=False), [])

    verdict = _disposition_verdict(tree, header, codes)

    assert verdict.inputs == len(stems), "non-vacuity: every asset must be an input"
    assert verdict.missing == [], (
        "P10: one rejected asset must cost only itself; a directory-grain R "
        f"invocation voided {len(verdict.missing)} of {len(stems)} dispositions"
    )


# ── P11 the published tree is read-only to the run (M2.8.1 D01) ─────
_SESSION_ID = "event-01"


def _publish(root: pathlib.Path, *, link_target: pathlib.Path | None = None) -> dict[str, Any]:
    """The smallest tree `validate_generation` accepts, carrying one symlink.

    The real tree is symlinks into patient media, so the link is what makes
    `tree_digest`'s link-text-not-contents rule reachable from a synthetic tree.
    """
    event = root / _SESSION_ID
    event.mkdir(parents=True, exist_ok=True)
    (event / "cam-a.txt").write_text("placement\n", encoding="utf-8")
    if link_target is not None:
        (event / "cam-a.media").symlink_to(link_target)
    for name in (sessions_module.EVENTS_FILENAME, sessions_module.PLACEMENTS_FILENAME):
        (root / name).write_text(f"{name}\n", encoding="utf-8")
    generation: dict[str, Any] = {
        sessions_module.EVENTS_FILENAME: hashlib.sha256(
            (root / sessions_module.EVENTS_FILENAME).read_bytes()
        ).hexdigest(),
        sessions_module.PLACEMENTS_FILENAME: hashlib.sha256(
            (root / sessions_module.PLACEMENTS_FILENAME).read_bytes()
        ).hexdigest(),
        "tree": sessions_module.tree_digest(root),
        "inventory": {},
        "generator_version": sessions_module.GENERATOR_VERSION,
    }
    (root / sessions_module.GENERATION_FILENAME).write_text(
        json.dumps(generation, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    assert sessions_module.validate_generation(root) == generation
    return generation


def _session(root: pathlib.Path) -> Session:
    return Session(session_id=_SESSION_ID, directory=(root / _SESSION_ID).resolve(), cameras=[])


_OVERLAPPING = ("unguarded-default", "tree-root", "inside-tree", "link-into-tree")


@pytest.mark.parametrize("destination", _OVERLAPPING)
def test_p11_the_guard_refuses_every_destination_overlapping_the_published_tree(
    tmp_path: pathlib.Path, destination: str
) -> None:
    root = tmp_path / "sessions"
    _publish(root)
    session = _session(root)
    if destination == "unguarded-default":
        # The default lands at <tree>/output/<session_id>, so omitting the flag
        # is the hazard rather than an escape from it.
        target: pathlib.Path | None = None
    elif destination == "tree-root":
        target = root
    elif destination == "inside-tree":
        target = root / "output"
    else:
        target = tmp_path / "elsewhere"
        target.symlink_to(root / "output")

    with pytest.raises(SessionError):
        _resolve_session_output(session, target)


def test_p11_the_guard_refuses_a_destination_that_contains_the_published_tree(
    tmp_path: pathlib.Path,
) -> None:
    """The second direction of the overlap test, which needs a nested tree to reach.

    A destination that merely sits beside the tree is legal, so the ancestor
    branch fires only when the resolved output is a strict parent of the
    published root — here, the session id names the directory the tree lives in.
    """
    root = tmp_path / "nest" / "sessions"
    (root / "nest").mkdir(parents=True)
    _publish(root)
    session = Session(session_id="nest", directory=(root / "nest").resolve(), cameras=[])

    with pytest.raises(SessionError):
        _resolve_session_output(session, tmp_path)


def test_p11_a_destination_beside_the_published_tree_is_accepted(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "sessions"
    _publish(root)
    outside = tmp_path / "run"

    resolved = _resolve_session_output(_session(root), outside)

    assert resolved == outside / _SESSION_ID, "P11: non-vacuity — the legal case must resolve"
    assert not resolved.resolve().is_relative_to(root.resolve())


def test_p11_the_guard_disengages_when_the_tree_carries_no_generation_marker(
    tmp_path: pathlib.Path,
) -> None:
    """Characterization: containment is conditional on the marker, not on the path.

    `_published_root` returns None for an unmarked tree, so the unguarded
    default resolves inside it.  The corpus driver must therefore call
    `validate_generation` before it resolves any output, since a missing marker
    is exactly the state that turns the guard off.
    """
    root = tmp_path / "sessions"
    (root / _SESSION_ID).mkdir(parents=True)

    resolved = _resolve_session_output(_session(root), None)

    assert resolved.resolve().is_relative_to(root.resolve()), (
        "an unmarked tree takes the default that lands inside it"
    )


_TREE_MUTATIONS = ("new-file", "modify-file", "new-dir", "retarget-link", "delete-file")


@pytest.mark.parametrize("mutation", _TREE_MUTATIONS)
def test_p11_the_tree_digest_moves_for_every_write_inside_the_tree(
    tmp_path: pathlib.Path, mutation: str
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.bin").write_bytes(b"frames\n")
    (media / "other.bin").write_bytes(b"frames\n")
    root = tmp_path / "sessions"
    _publish(root, link_target=media / "clip.bin")
    before = sessions_module.tree_digest(root)
    event = root / _SESSION_ID

    assert sessions_module.tree_digest(root) == before, "non-vacuity: the digest is stable at rest"

    if mutation == "new-file":
        (event / "cam-a.csv").write_text("frame\n", encoding="utf-8")
    elif mutation == "modify-file":
        (event / "cam-a.txt").write_text("edited\n", encoding="utf-8")
    elif mutation == "new-dir":
        (root / "output").mkdir()
    elif mutation == "retarget-link":
        (event / "cam-a.media").unlink()
        (event / "cam-a.media").symlink_to(media / "other.bin")
    else:
        (event / "cam-a.txt").unlink()

    assert sessions_module.tree_digest(root) != before, f"P11: {mutation} must move the digest"


def test_p11_a_rewritten_generation_marker_leaves_both_witnesses_unmoved(
    tmp_path: pathlib.Path,
) -> None:
    """P11's two witnesses cannot see a write to the one file inside the tree.

    `tree_digest` excludes the marker by construction — a document cannot digest
    itself — and `validate_generation` reads the marker through `json.loads`, so
    a rewrite that preserves the fields passes.  The exclusion is right; the
    defect is P11's, which conjoins "writes nothing inside the tree" with a
    witness that is blind to a file inside the tree.  MAIN must supply a second
    witness over the marker's own bytes for the run's end-to-end claim.
    """
    root = tmp_path / "sessions"
    generation = _publish(root)
    marker = root / sessions_module.GENERATION_FILENAME
    before_digest = sessions_module.tree_digest(root)
    before_bytes = marker.read_bytes()

    marker.write_text(json.dumps(generation, sort_keys=True, indent=4) + "\n", encoding="utf-8")

    assert marker.read_bytes() != before_bytes, "non-vacuity: the marker must really have changed"
    assert sessions_module.validate_generation(root) == generation
    assert sessions_module.tree_digest(root) != before_digest, (
        "P11: a write inside the published tree must move the digest, and a write "
        "to the marker moves neither witness the predicate names"
    )


def test_p11_an_empty_tree_never_reads_as_an_unmoved_digest(tmp_path: pathlib.Path) -> None:
    """Two empty trees digest identically, so `unmoved` alone is a vacuous verdict."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()

    assert sessions_module.tree_digest(first) == sessions_module.tree_digest(second)
    assert not any(first.iterdir()), (
        "P11: the tree must be non-empty for `unmoved` to mean anything"
    )


@pytest.mark.parametrize("writes_inside", [False, True], ids=["outside", "inside"])
def test_p11_a_run_leaves_the_tree_unmoved_only_when_it_writes_outside(
    tmp_path: pathlib.Path, writes_inside: bool
) -> None:
    """Stand-in for the corpus driver: the end-to-end claim, both ways round."""
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.bin").write_bytes(b"frames\n")
    root = tmp_path / "sessions"
    generation = _publish(root, link_target=media / "clip.bin")
    before = sessions_module.tree_digest(root)
    assert before == generation["tree"], "non-vacuity: the tree must be the published one"

    destination = root / "output" if writes_inside else tmp_path / "run"
    session_out = destination / _SESSION_ID
    session_out.mkdir(parents=True)
    (session_out / "cam-a.csv").write_text("frame\n0\n", encoding="utf-8")

    unmoved = sessions_module.tree_digest(root) == before
    try:
        sessions_module.validate_generation(root)
        valid = True
    except sessions_module.SessionsError:
        valid = False

    assert (unmoved, valid) == (not writes_inside, not writes_inside), (
        "P11: writing outside leaves both witnesses green; writing inside must break both"
    )


# ── P12 per-asset CFR counters ──────────────────────────────────────
_CFR_COUNTERS = ("pts_accepted", "index_fallback", "monotonic_forced")
_NOMINAL_FPS = 30.0


class _StubCapture:
    """Serves the scripted presentation timestamps, in milliseconds.

    The property argument is ignored: `SourceTimestampClock` reads exactly one
    property on a file-backed source, so scripting the sequence is the whole
    stimulus and naming the constant would only pull cv2 into the case.
    """

    def __init__(self, pos_msec: Sequence[float]) -> None:
        self.pos_msec = list(pos_msec)
        self.calls = 0

    def get(self, _prop: int) -> float:
        value = self.pos_msec[self.calls]
        self.calls += 1
        return value


# Indices and presentation timestamps chosen to reach all three branches:
# two accepted, one duplicate that forces monotonicity, one missing that falls
# back to the index and lands ahead of the last returned value.
_CFR_STIMULUS = ((0, 0.0), (1, 100.0), (2, 100.0), (10, -1.0))


def _driven_clock() -> Any:
    capture = _StubCapture([msec for _, msec in _CFR_STIMULUS])
    clock = video_io.SourceTimestampClock(capture, _NOMINAL_FPS, live=False)
    for index, _ in _CFR_STIMULUS:
        clock.timestamp(index)
    return clock


def _counter_sum(row: dict[str, str]) -> int:
    return sum(int(row[name]) for name in _CFR_COUNTERS)


def test_p12_exactly_one_counter_increments_per_timestamp_call() -> None:
    """P12's `by construction` clause, proved over a stimulus that reaches all three branches."""
    capture = _StubCapture([msec for _, msec in _CFR_STIMULUS])
    clock = video_io.SourceTimestampClock(capture, _NOMINAL_FPS, live=False)

    for calls, (index, _) in enumerate(_CFR_STIMULUS, start=1):
        clock.timestamp(index)
        counted = sum(getattr(clock, name) for name in _CFR_COUNTERS)
        assert counted == calls, "P12: exactly one counter must move per call"

    exercised = {name: getattr(clock, name) for name in _CFR_COUNTERS}
    assert all(exercised.values()), f"P12: non-vacuity — every branch must fire, got {exercised}"
    assert clock.n_timestamps == len(_CFR_STIMULUS)


def test_p12_the_written_diagnostics_row_preserves_the_counter_sum(tmp_path: pathlib.Path) -> None:
    clock = _driven_clock()
    path = tmp_path / "asset-0000_diag.csv"

    run_module.write_source_diagnostics(
        path, video=_VIDEO, clock=clock, fps_nominal=_NOMINAL_FPS, latencies=[10.0, 20.0]
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1, "P12: one source, one diagnostics row"
    row = rows[0]
    assert _counter_sum(row) == int(row["n_frames_decoded"]), (
        "P12: the recorded denominator must be the counter sum"
    )
    assert float(row["cfr_fallback_rate"]) == pytest.approx(
        (int(row["index_fallback"]) + int(row["monotonic_forced"])) / _counter_sum(row), abs=1e-6
    )


def test_p12_the_derived_quantities_are_never_stored() -> None:
    """The contract says `n_timestamps` and `cfr_fallback_rate` are derived, never stored.

    Both are properties on the clock, which is the derived half.  The shipped
    diagnostics schema stores both anyway: `cfr_fallback_rate` under its own
    name, `n_timestamps` under the alias `n_frames_decoded`, whose value is
    `clock.n_timestamps` and not a decoded-frame count.  Storing a value that
    is a function of three stored fields is what lets a consumer read a rate
    that disagrees with the counters beside it.
    """
    for name in ("n_timestamps", "cfr_fallback_rate"):
        assert isinstance(getattr(video_io.SourceTimestampClock, name), property), (
            f"P12: {name} must be derived on the clock"
        )

    fields = run_module.SOURCE_DIAGNOSTIC_FIELDS
    assert set(_CFR_COUNTERS) <= set(fields), "non-vacuity: the three counters must be recorded"

    written = re.search(
        r"def write_source_diagnostics\(.*?\n    row = \{(.*?)\n    \}",
        (_PROJECT_ROOT / "src" / "pose_estimation" / "run.py").read_text(encoding="utf-8"),
        re.S,
    )
    assert written is not None, "the diagnostics row must parse"
    stored_derived = sorted(
        field
        for field in fields
        if re.search(rf'"{field}":.*clock\.(n_timestamps|cfr_fallback_rate)', written.group(1))
    )

    assert stored_derived == [], (
        f"P12: the diagnostics schema stores {stored_derived}, each a function of the "
        "three counters the contract requires it to be derived from at read time"
    )


def _write_cfr_diagnostics(
    tree: pathlib.Path, asset_id: str, counters: tuple[int, int, int], *, decoded: int | None = None
) -> None:
    total = sum(counters)
    row = dict.fromkeys(run_module.SOURCE_DIAGNOSTIC_FIELDS, "") | {
        "video": f"{_VIDEO}-{asset_id}",
        "n_frames_decoded": str(total if decoded is None else decoded),
        "fps_nominal": f"{_NOMINAL_FPS:.6f}",
    }
    row |= dict(zip(_CFR_COUNTERS, (str(value) for value in counters), strict=True))
    path = tree / f"{asset_id}_diag.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_module.SOURCE_DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _cfr_verdict(tree: pathlib.Path, rows: Sequence[dict[str, str]]) -> SimpleNamespace:
    """P12 at corpus grain: every `ok` asset carries the three counters, and the pool is derived."""
    ok_ids = [row["asset_id"] for row in rows if row["disposition"] == _DISPOSITION_OK]
    other_ids = [row["asset_id"] for row in rows if row["disposition"] != _DISPOSITION_OK]
    verdict = SimpleNamespace(
        ok=len(ok_ids),
        missing=[],
        incomplete=[],
        inconsistent=[],
        trespass=[],
        frames=0,
        fallbacks=0,
    )
    per_asset: list[float] = []
    for asset_id in ok_ids:
        path = tree / f"{asset_id}_diag.csv"
        if not path.is_file():
            verdict.missing.append(asset_id)
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), None)
        if row is None or any(not row.get(name, "").strip() for name in _CFR_COUNTERS):
            verdict.incomplete.append(asset_id)
            continue
        total = _counter_sum(row)
        if total != int(row["n_frames_decoded"]) or any(int(row[n]) < 0 for n in _CFR_COUNTERS):
            verdict.inconsistent.append(asset_id)
            continue
        fallbacks = int(row["index_fallback"]) + int(row["monotonic_forced"])
        verdict.frames += total
        verdict.fallbacks += fallbacks
        per_asset.append(fallbacks / total if total else 0.0)
    verdict.trespass = [i for i in other_ids if (tree / f"{i}_diag.csv").is_file()]
    verdict.pooled = verdict.fallbacks / verdict.frames if verdict.frames else 0.0
    verdict.mean_of_rates = sum(per_asset) / len(per_asset) if per_asset else 0.0
    verdict.complete = not (
        verdict.missing or verdict.incomplete or verdict.inconsistent or verdict.trespass
    )
    # A pooled rate over zero frames is 0.0, which reads exactly like a corpus
    # that never substituted a timestamp.  Only the frame count separates them.
    verdict.nonvacuous = verdict.frames > 0
    return verdict


def test_p12_every_ok_asset_records_the_three_counters(tmp_path: pathlib.Path) -> None:
    rows = _mixed_manifest()
    tree = tmp_path / "run"
    tree.mkdir(parents=True)
    for index, row in enumerate(rows):
        if row["disposition"] == _DISPOSITION_OK:
            _write_cfr_diagnostics(tree, row["asset_id"], (100 - index, index, 1))

    verdict = _cfr_verdict(tree, rows)

    assert verdict.ok > 0, "P12: non-vacuity — the ok side must be populated"
    assert verdict.complete is True
    assert verdict.nonvacuous is True
    assert verdict.pooled > 0.0


@pytest.mark.parametrize(
    "mutation", ["missing-row", "blank-counter", "sum-mismatch", "negative-counter", "trespass"]
)
def test_p12_every_counter_mutation_is_caught(tmp_path: pathlib.Path, mutation: str) -> None:
    rows = _mixed_manifest()
    tree = tmp_path / "run"
    tree.mkdir(parents=True)
    for row in rows:
        if row["disposition"] == _DISPOSITION_OK:
            _write_cfr_diagnostics(tree, row["asset_id"], (90, 8, 2))
    ok_id = next(row["asset_id"] for row in rows if row["disposition"] == _DISPOSITION_OK)
    other_id = next(row["asset_id"] for row in rows if row["disposition"] != _DISPOSITION_OK)

    if mutation == "missing-row":
        (tree / f"{ok_id}_diag.csv").unlink()
    elif mutation == "blank-counter":
        _write_cfr_diagnostics(tree, ok_id, (90, 8, 2))
        text = (tree / f"{ok_id}_diag.csv").read_text(encoding="utf-8").replace(",8,", ",,")
        (tree / f"{ok_id}_diag.csv").write_text(text, encoding="utf-8")
    elif mutation == "sum-mismatch":
        _write_cfr_diagnostics(tree, ok_id, (90, 8, 2), decoded=101)
    elif mutation == "negative-counter":
        _write_cfr_diagnostics(tree, ok_id, (92, -2, 10))
    else:
        _write_cfr_diagnostics(tree, other_id, (90, 8, 2))

    verdict = _cfr_verdict(tree, rows)

    assert verdict.complete is False, f"P12: {mutation} must break the counter record"


def test_p12_the_pooled_rate_is_frame_weighted_and_not_the_mean_of_per_asset_rates(
    tmp_path: pathlib.Path,
) -> None:
    """The two agree only when every asset holds the same number of frames.

    A corpus whose assets differ by an order of magnitude in length makes the
    unweighted mean a different published number, and P12 names the pooled one.
    """
    rows = [
        {"asset_id": "asset-0000", "disposition": _DISPOSITION_OK},
        {"asset_id": "asset-0001", "disposition": _DISPOSITION_OK},
    ]
    tree = tmp_path / "run"
    tree.mkdir(parents=True)
    _write_cfr_diagnostics(tree, "asset-0000", (990, 10, 0))
    _write_cfr_diagnostics(tree, "asset-0001", (5, 5, 0))

    verdict = _cfr_verdict(tree, rows)

    assert verdict.pooled == pytest.approx(15 / 1010)
    assert verdict.mean_of_rates == pytest.approx((10 / 1000 + 5 / 10) / 2)
    assert verdict.pooled != pytest.approx(verdict.mean_of_rates), (
        "P12: the discriminating case — the pooled rate is not the mean of the rates"
    )


def test_p12_a_corpus_with_no_recorded_frames_never_reads_as_a_measured_rate(
    tmp_path: pathlib.Path,
) -> None:
    tree = tmp_path / "run"
    tree.mkdir(parents=True)

    verdict = _cfr_verdict(tree, [{"asset_id": "asset-0000", "disposition": "decode_failed"}])

    assert verdict.pooled == 0.0, "a zero pool reads exactly like a corpus that never substituted"
    assert verdict.complete is True
    assert verdict.nonvacuous is False, "P12: zero frames must never publish as a measured rate"


def test_p12_a_live_source_counts_wall_clock_values_as_accepted_timestamps() -> None:
    """Characterization: `pts_accepted` names the branch, not the value's origin.

    A live capture has no presentation timestamp at all, and its monotonic
    readings increment `pts_accepted`.  Every corpus asset is file-backed, so
    this costs the corpus run nothing — it bounds what the counter name means.
    """
    ticks = iter([0.0, 1.0, 2.0])
    clock = video_io.SourceTimestampClock(
        _StubCapture([]), _NOMINAL_FPS, live=True, monotonic=lambda: next(ticks)
    )

    for index in range(3):
        clock.timestamp(index)

    assert (clock.pts_accepted, clock.index_fallback, clock.monotonic_forced) == (3, 0, 0)
    assert clock.cfr_fallback_rate == 0.0


# ── P13 report redaction (D07) ──────────────────────────────────────
_PILOT = _PROJECT_ROOT / "scripts" / "pilot_corpus_run.py"
# The key rule P13 freezes.  The analog's own constant must equal it.
_KEY_PATTERN = r"[a-z][a-z0-9_]*"
# A capture id spelled without a separator or a capital: the shape the analog's
# reasoning assumes cannot occur.  Synthetic; no corpus token appears here.
_IDENTIFIER = "subject90"


def _pilot_source() -> str:
    return _PILOT.read_text(encoding="utf-8")


def _pilot_key_pattern() -> str:
    match = re.search(r"FIELD_NAME = re\.compile\(r\"([^\"]+)\"\)", _pilot_source())
    assert match is not None, "the analog must declare its key pattern"
    return match.group(1)


def _pilot_int_stratum_axes() -> list[str]:
    """Stratum axes `_coverage` keys by, whose values are integers.

    `_coverage` builds ``{str(value): ...}`` over ``(*AXES, "pts_monotonic")``,
    so an int-typed axis produces a digit-leading key by construction.
    """
    source = _pilot_source()
    axes = re.search(r"^AXES = \(([^)]*)\)", source, re.M)
    asset = re.search(r"class Asset:\n(.*?)\n\n", source, re.S)
    assert axes is not None, "the analog must declare its stratum axes"
    assert asset is not None, "the analog must declare its asset record"
    covered = [*re.findall(r'"([a-z_]+)"', axes.group(1)), "pts_monotonic"]
    integral = set(re.findall(r"^\s+([a-z_]+): int$", asset.group(1), re.M))
    return [axis for axis in covered if axis in integral]


def _pilot_allowlist_extras() -> list[str]:
    """Entries the analog adds to its allowlist beyond the R codes and stratum labels."""
    call = re.search(
        r"_assert_redacted\(\s*payload,\s*frozenset\(\s*\{(.*?)\}", _pilot_source(), re.S
    )
    assert call is not None, "the analog's allowlist call site must parse"
    return [entry.strip() for entry in call.group(1).strip().split(",") if entry.strip()]


def _violations(
    payload: Any, allowed: frozenset[str], *, keys_allowlisted: bool
) -> SimpleNamespace:
    """Walk a report the way the analog does, under one of the two key rules.

    `strict` is P13 as frozen: a key must match the pattern.  `analog` is what
    `scripts/pilot_corpus_run.py` implements: a key must be allowlisted OR match
    the pattern.  The two disagree, and the cases below are where.
    """
    result = SimpleNamespace(keys=[], values=[], strings=0)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                allowed_key = keys_allowlisted and key in allowed
                if not allowed_key and not re.fullmatch(_KEY_PATTERN, key):
                    result.keys.append(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            result.strings += 1
            if node not in allowed:
                result.values.append(node)

    walk(payload)
    result.clean = not (result.keys or result.values)
    # A report that emits no string satisfies every clause of P13.
    result.nonvacuous = result.strings > 0
    return result


def test_p13_the_analog_declares_the_key_pattern_the_contract_freezes() -> None:
    assert _pilot_key_pattern() == _KEY_PATTERN, "P13: the frozen key rule is the analog's"


def test_p13_the_allowlist_refuses_every_string_it_did_not_publish() -> None:
    _, codes = _r_group_qc_constants()
    allowed = frozenset({*codes, *_DISPOSITIONS, "hevc"})
    payload = {
        "drop_reasons": {codes[0]: 3},
        "dispositions": [_DISPOSITION_OK, "decode_failed"],
        "codec": "hevc",
        "assets": 379,
    }

    verdict = _violations(payload, allowed, keys_allowlisted=False)

    assert verdict.nonvacuous is True, "P13: non-vacuity — the report must emit strings"
    assert verdict.clean is True, "P13: published labels, R codes and disposition codes pass"


def test_p13_n7_a_capture_identifier_emitted_as_a_value_is_refused() -> None:
    allowed = frozenset({_DISPOSITION_OK})

    verdict = _violations({"disposition": _IDENTIFIER}, allowed, keys_allowlisted=False)

    assert verdict.values == [_IDENTIFIER], "N7: an unpublished string must fail the allowlist"


def test_p13_n7_does_not_fire_when_the_same_identifier_is_a_key() -> None:
    """The key clause is a shape test, so it is a denylist wearing an allowlist's name.

    A value is checked against the published set; a key is checked against a
    pattern.  An identifier with no separator and no capital therefore passes as
    a key while the identical string is refused as a value, and N7 — "emit a
    capture id in the report" — fires on only one of the two placements.  The
    fix is to check keys against a frozen field-name set, which is the only form
    that makes P13's own word "allowlist" true of keys.
    """
    allowed = frozenset({_DISPOSITION_OK})

    as_value = _violations({"disposition": _IDENTIFIER}, allowed, keys_allowlisted=False)
    as_key = _violations({_IDENTIFIER: 3}, allowed, keys_allowlisted=False)

    assert as_value.clean is False, "non-vacuity: the value placement must be refused"
    assert as_key.clean is False, (
        f"P13: {_IDENTIFIER!r} passes the key clause because it matches {_KEY_PATTERN!r}; "
        "a pattern admits every identifier of that shape, which is a denylist"
    )


def test_p13_a_stratum_keyed_block_fails_the_frozen_key_clause() -> None:
    """P13 as frozen refuses a report shape the analog publishes.

    `_coverage` keys its inner dictionaries by stratum *value*, and the analog
    declares int-typed axes, so those keys are digits.  They survive only
    because the analog's key test is a disjunction: allowlisted OR matching.
    P13 states the second half alone, so a corpus report built the analog's way
    violates the predicate as written.
    """
    axes = _pilot_int_stratum_axes()
    assert axes, "non-vacuity: the analog must declare an int-typed stratum axis"
    coverage = {axis: {"0": {"corpus": 379, "pilot": 16}} for axis in axes}

    strict = _violations({"coverage": coverage}, frozenset({"0"}), keys_allowlisted=False)
    analog = _violations({"coverage": coverage}, frozenset({"0"}), keys_allowlisted=True)

    assert analog.clean is True, "the analog accepts the block it publishes"
    assert strict.clean is True, (
        f"P13: axes {axes} key their coverage block by an integer stratum value, "
        f"which no key matching {_KEY_PATTERN!r} can be — the predicate needs the "
        "allowlisted-OR-matching rule the analog implements"
    )


def test_p13_the_three_admissible_classes_cannot_name_the_report_itself() -> None:
    """P13 enumerates three string classes, and a report needs a fourth.

    The analog allowlists its generator, its version and its four configuration
    echoes at the call site.  None is a stratum label, an R reason code or a
    disposition code, and one is a path.  A corpus report obeying P13 literally
    could not say which program wrote it or which device ran the pose model.
    """
    extras = _pilot_allowlist_extras()

    assert extras, "non-vacuity: the analog's call site must list entries of its own"
    # The three classes reach the allowlist through the derived unions beside
    # this literal — R codes, codec labels, device labels, stratum values.  What
    # is spelled inside the literal is exactly what those unions cannot supply.
    assert extras == [], (
        f"P13: the analog must allowlist {len(extras)} code-authored constants "
        f"({', '.join(extras)}), none of them a stratum label, an R reason code "
        "or a disposition code; the predicate needs a fourth admissible class"
    )


def test_p13_the_key_rule_must_be_anchored_at_both_ends() -> None:
    """`re.match` accepts a path-shaped key on its leading segment; `fullmatch` refuses it."""
    smuggled = f"{_IDENTIFIER}/cam"

    assert re.match(_KEY_PATTERN, smuggled) is not None, "an unanchored test would admit it"
    assert re.fullmatch(_KEY_PATTERN, smuggled) is None
    assert _violations({smuggled: 1}, frozenset(), keys_allowlisted=False).clean is False


def test_p13_an_empty_report_satisfies_every_clause_and_must_still_fail() -> None:
    verdict = _violations({}, frozenset(), keys_allowlisted=False)

    assert verdict.clean is True, "every clause is vacuous over zero strings"
    assert verdict.nonvacuous is False, "P13: an empty report must never read as redacted"


def test_p13_a_denylist_admits_what_the_allowlist_refuses() -> None:
    """The refused design, made concrete.

    A denylist tests shape — separators, capitals, long digit runs — and every
    such rule passes a lowercase identifier that carries none of them.  The
    allowlist refuses it for the only reason that generalises: nothing published
    it.  This is why the value clause is written as membership.
    """

    def denied(value: str) -> bool:
        return bool(re.search(r"[/.\s]|[A-Z]|\d{4}", value))

    allowed = frozenset({_DISPOSITION_OK})

    assert denied(_IDENTIFIER) is False, "a denylist of identifier shapes lets it through"
    assert _violations({"disposition": _IDENTIFIER}, allowed, keys_allowlisted=False).clean is False


# ── P14 the corpus-run cost record (D02) ────────────────────────────
_ROADMAP = _PROJECT_ROOT / ".agent" / "roadmap.md"
_CONTRACT = _PROJECT_ROOT / ".agent" / "archive" / "contract-m2u82.md"
_UNIT = "M2.8.2"
_HOURS = re.compile(r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*h\b")


def _inadmissible_hours() -> list[str]:
    """The figures D02 rules out, read from D02 rather than transcribed.

    D02 names both by value — one measured a frozen tracker, the other
    extrapolated per-call latency — so the contract itself is the oracle for
    what a post-fix record must no longer carry.
    """
    bullet = re.search(r"- \*\*D02 —.*?(?=\n- \*\*D0)", _CONTRACT.read_text(encoding="utf-8"), re.S)
    assert bullet is not None, "the contract must declare D02"
    return _HOURS.findall(bullet.group(0))


def _unit_cost_rows() -> list[tuple[str, str]]:
    """Every roadmap table row sizing this unit, paired with its leading hour figure.

    A table row is the budget surface: prose can name a figure to disavow it,
    but a row's leading figure is the sizing claim a reader acts on.
    """
    rows = []
    for line in _ROADMAP.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and _UNIT in line:
            figure = _HOURS.search(line)
            if figure is not None:
                rows.append((figure.group(1), line))
    return rows


def _labelled_measured(row: str, figure: str) -> bool:
    window = re.search(rf"{re.escape(figure)}\s*h\b(.{{0,40}})", row)
    return window is not None and "measured" in window.group(1)


def test_p14_the_contract_names_the_figures_a_post_fix_record_must_not_carry() -> None:
    inadmissible = _inadmissible_hours()

    assert len(inadmissible) >= 2, (
        "P14: non-vacuity — D02 must name the superseded figures by value, "
        "otherwise the two cases below range over an empty set"
    )


def test_p14_every_unit_cost_row_labels_its_leading_figure_measured() -> None:
    rows = _unit_cost_rows()
    assert len(rows) >= 2, f"P14: non-vacuity — {_UNIT} must be sized in the roadmap tables"

    unlabelled = [figure for figure, row in rows if not _labelled_measured(row, figure)]

    assert unlabelled == [], (
        f"P14: {unlabelled} size {_UNIT} in a roadmap table without the `measured` "
        "label D02 requires, so a projected figure is carried as the budget"
    )


def test_p14_no_superseded_figure_is_carried_as_the_unit_cost() -> None:
    """D02 rules out both prior figures, including the one still labelled `measured`.

    26-31 h measured a frozen tracker and 6.5 h extrapolated per-call latency;
    D01 removed the run both described.  A `measured` label on a superseded
    sample is worse than an unlabelled figure, because it reads as admissible.
    """
    inadmissible = set(_inadmissible_hours())
    rows = _unit_cost_rows()
    assert rows, "P14: non-vacuity — the roadmap must size the unit"

    carried = sorted({figure for figure, _ in rows if figure in inadmissible})

    assert carried == [], (
        f"P14: {carried} still size {_UNIT}, and D02 admits only the post-fix "
        "pilot re-run as an input to the corpus estimate"
    )


def test_p14_a_measured_figure_names_its_sample() -> None:
    """The form the record must take, which the roadmap already spells somewhere."""
    named = [
        row
        for figure, row in _unit_cost_rows()
        if _labelled_measured(row, figure)
        and re.search(rf"{re.escape(figure)}\s*h\b[^(]{{0,20}}\**\s*\(", row)
    ]

    assert named, (
        "P14: a figure labelled `measured` must name its measuring sample in the "
        "same breath; a bare `measured` states no sample at all"
    )


def test_p14_the_analog_derives_corpus_hours_from_a_named_sample() -> None:
    """Extrapolating from a measured pilot is the method D02 admits; the sample is what it names."""
    source = _pilot_source()
    projection = re.search(r'"corpus_hours_incl_startup":(.{0,120})', source, re.S)

    assert projection is not None, "the analog must publish a corpus-hours figure"
    assert "corpus_frames" in projection.group(1), "the figure must scale the measured throughput"
    assert re.search(r'"selection": \{.*?"assets": len\(selected\)', source, re.S), (
        "P14: the report must publish the sample the figure was measured on"
    )


def test_p14_a_roadmap_without_a_cost_record_never_reads_as_measured(
    tmp_path: pathlib.Path,
) -> None:
    """A unit nobody sized satisfies both clauses: no unlabelled figure, no superseded one."""
    empty = tmp_path / "roadmap.md"
    empty.write_text(f"| {_UNIT} | data | Full corpus 2D run, resumable. |\n", encoding="utf-8")
    rows = [
        (figure.group(1), line)
        for line in empty.read_text(encoding="utf-8").splitlines()
        if line.startswith("|") and _UNIT in line
        for figure in [_HOURS.search(line)]
        if figure is not None
    ]

    assert rows == [], "no figure means no violation, and no record either"
    assert _unit_cost_rows(), "P14: the predicate must range over a non-empty set to mean anything"

"""M2.8.2 predicates for the full corpus 2D run.

Written diff-blind against the frozen acceptance contract, never against the
implementation, so the cases cover what an author's own reading would skip.
"""

from __future__ import annotations

import csv
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

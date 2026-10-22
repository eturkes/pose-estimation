"""The six downstream R consumers read clinical artifacts through one mode-aware reader.

`analysis/utils.R` `discover_clinical` selects each mode's suffixes positively: a 2D-only or
3D-only directory runs as before (3D outputs carry `_3d`), and a directory holding both modes
refuses unless `--mode` names one, so no aggregate pools normalised 2D with metric 3D.
"""

from __future__ import annotations

import csv
import pathlib
import subprocess

import pytest

from test_r_pipeline import requires_features_r

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GOLDENS = _ROOT / "tests" / "goldens" / "r_clinical"
_ANALYSIS = _ROOT / "analysis"

# stem → (golden frame file, golden windows file, mode suffix, feature scale). Distinct scales
# give the per-video aggregates the between-video variance correlation + PCA need.
_SOURCES = {
    "v2d_a": ("2d_idx_clinical.csv", "2d_idx_clinical_windows.csv", "_clinical", 1.0),
    "v2d_b": ("2d_idx_clinical.csv", "2d_idx_clinical_windows.csv", "_clinical", 1.1),
    "v2d_c": ("2d_cumsum_clinical.csv", "2d_cumsum_clinical_windows.csv", "_clinical", 1.3),
    "v3d_a": ("world3d_clinical_3d.csv", "world3d_clinical_3d_windows.csv", "_clinical_3d", 1.0),
    "v3d_b": ("world3d_clinical_3d.csv", "world3d_clinical_3d_windows.csv", "_clinical_3d", 1.1),
    "v3d_c": ("world3d_clinical_3d.csv", "world3d_clinical_3d_windows.csv", "_clinical_3d", 1.3),
}
_KEYS = {"video", "frame_idx", "timestamp_sec", "person_idx", "window_start_sec", "window_end_sec"}


def _scaled(column: str, cell: str, scale: float) -> str:
    if column in _KEYS or scale == 1.0:
        return cell
    try:
        return repr(float(cell) * scale)
    except ValueError:
        return cell


def _copy_as(golden: str, out: pathlib.Path, video: str, scale: float) -> None:
    with (_GOLDENS / golden).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = [
            {k: _scaled(k, v, scale) for k, v in {**row, "video": video}.items()} for row in reader
        ]
        fields = reader.fieldnames
    assert fields is not None
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _tree(root: pathlib.Path, modes: tuple[str, ...]) -> pathlib.Path:
    root.mkdir()
    for stem, (frame, windows, suffix, scale) in _SOURCES.items():
        if stem[1:3] in modes:
            _copy_as(frame, root / f"{stem}{suffix}.csv", stem, scale)
            _copy_as(windows, root / f"{stem}{suffix}_windows.csv", stem, scale)
    return root


def _metadata(root: pathlib.Path, videos: list[str]) -> dict[str, pathlib.Path]:
    meta = root.parent / f"{root.name}-meta"
    meta.mkdir()
    sessions = meta / "sessions.csv"
    sessions.write_text(
        "video,patient_id,session_date\n"
        + "".join(f"{v},p1,2026-0{i + 1}-01\n" for i, v in enumerate(videos)),
        encoding="utf-8",
    )
    scores = meta / "scores.csv"
    scores.write_text(
        "video,GRASSP,UEMS\n" + "".join(f"{v},{10 + i},{20 - i}\n" for i, v in enumerate(videos)),
        encoding="utf-8",
    )
    return {"sessions": sessions, "scores": scores}


def _argv(consumer: str, root: pathlib.Path, meta: dict[str, pathlib.Path]) -> list[str]:
    script = str(_ANALYSIS / f"{consumer}.R")
    extra = {
        "longitudinal": [str(meta["sessions"]), str(meta["scores"])],
        "clinical_correlation": [str(meta["scores"])],
    }
    return ["Rscript", script, str(root), *extra.get(consumer, [])]


_CONSUMERS = (
    "compare_clinical",
    "longitudinal",
    "clinical_correlation",
    "clinical_dimreduce",
    "temporal_clinical",
    "explore_clinical",
)
_WRITERS = _CONSUMERS


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=180, check=False)


def _outputs(root: pathlib.Path, before: set[str]) -> set[str]:
    return {p.name for p in root.iterdir()} - before


@requires_features_r
@pytest.mark.parametrize("consumer", _CONSUMERS)
def test_mixed_tree_without_mode_refuses_and_writes_nothing(
    tmp_path: pathlib.Path, consumer: str
) -> None:
    root = _tree(tmp_path / "mixed", ("2d", "3d"))
    meta = _metadata(root, sorted(_SOURCES))
    before = {p.name for p in root.iterdir()}
    result = _run(_argv(consumer, root, meta))
    assert result.returncode != 0
    assert "both 2D and 3D clinical artifacts" in result.stderr
    assert _outputs(root, before) == set()


@requires_features_r
@pytest.mark.parametrize("consumer", _WRITERS)
def test_2d_tree_keeps_the_legacy_output_names(tmp_path: pathlib.Path, consumer: str) -> None:
    root = _tree(tmp_path / "2d", ("2d",))
    meta = _metadata(root, ["v2d_a", "v2d_b", "v2d_c"])
    before = {p.name for p in root.iterdir()}
    result = _run(_argv(consumer, root, meta))
    assert result.returncode == 0, result.stderr
    written = _outputs(root, before)
    assert written, "the consumer wrote nothing"
    assert not any("_3d" in name for name in written), written


@requires_features_r
@pytest.mark.parametrize("consumer", _WRITERS)
def test_3d_outputs_carry_the_mode_and_mode_flag_selects_it_from_a_mixed_tree(
    tmp_path: pathlib.Path, consumer: str
) -> None:
    single = _tree(tmp_path / "3d", ("3d",))
    before = {p.name for p in single.iterdir()}
    result = _run(_argv(consumer, single, _metadata(single, ["v3d_a", "v3d_b", "v3d_c"])))
    assert result.returncode == 0, result.stderr
    expected = _outputs(single, before)
    assert expected, "the consumer wrote nothing"
    assert all("_3d" in name for name in expected), expected

    mixed = _tree(tmp_path / "mixed", ("2d", "3d"))
    before = {p.name for p in mixed.iterdir()}
    result = _run(
        [*_argv(consumer, mixed, _metadata(mixed, ["v3d_a", "v3d_b", "v3d_c"])), "--mode", "3d"]
    )
    assert result.returncode == 0, result.stderr
    assert _outputs(mixed, before) == expected

"""Adversarial regression tests for the M2.1 inventory review."""

import os
import pathlib
import subprocess
import sys

import pytest

from pose_estimation import inventory, video_io


def test_printable_path_distinguishes_escape_text_from_escaped_byte() -> None:
    """A literal escape-looking name must not collide with a control byte."""
    assert inventory._printable_path("entry_\\x0a.MOV") != inventory._printable_path("entry_\n.MOV")


def test_printable_path_distinguishes_valid_c1_from_raw_byte() -> None:
    """A valid UTF-8 control and one invalid byte need distinct locators."""
    valid_utf8 = "entry_" + chr(0x80) + ".MOV"
    raw_byte = "entry_" + chr(0xDC80) + ".MOV"

    assert inventory.asset_id_of(valid_utf8) != inventory.asset_id_of(raw_byte)
    assert inventory._printable_path(valid_utf8) != inventory._printable_path(raw_byte)


def test_valid_utf8_path_survives_ascii_filesystem_locale(tmp_path: pathlib.Path) -> None:
    """UTF-8 bytes remain valid when Python decodes names with surrogateescape."""
    corpus = os.fsencode(tmp_path / "corpus")
    os.mkdir(corpus)  # noqa: PTH102 - byte paths are the subject under test
    subject = corpus + b"/subject_\xc3\xa9"
    os.mkdir(subject)  # noqa: PTH102 - byte paths are the subject under test
    descriptor = os.open(subject + b"/3_above_peg_l.MOV", os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)

    script = """
import pathlib
import sys
from pose_estimation.inventory import _exclusion_reason, _relative_posix
root = pathlib.Path(sys.argv[1])
entry = next(next(root.iterdir()).iterdir())
relative = _relative_posix(entry, root)
print(sys.getfilesystemencoding(), _exclusion_reason(entry, root.resolve(), relative) or '<none>')
"""
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONUTF8": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(pathlib.Path(inventory.__file__).parents[1]),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, os.fsdecode(corpus)],
        check=True,
        capture_output=True,
        encoding="ascii",
        env=environment,
    )

    assert completed.stdout.strip() == "ascii <none>"


def test_valid_utf8_path_rendering_is_filesystem_locale_independent(
    tmp_path: pathlib.Path,
) -> None:
    """One byte path must publish one locator under UTF-8 and ASCII locales."""
    corpus = os.fsencode(tmp_path / "corpus")
    os.mkdir(corpus)  # noqa: PTH102 - byte paths are the subject under test
    descriptor = os.open(corpus + b"/entry_\xc3\xa9.MOV", os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    script = """
import pathlib
import sys
from pose_estimation.inventory import _printable_path, _relative_posix
root = pathlib.Path(sys.argv[1])
relative = _relative_posix(next(root.iterdir()), root)
print(_printable_path(relative).encode('utf-8').hex())
"""

    outputs = []
    for locale, utf8_mode in (("C.UTF-8", "0"), ("C", "0")):
        environment = os.environ.copy()
        environment.update(
            {
                "LC_ALL": locale,
                "PYTHONCOERCECLOCALE": "0",
                "PYTHONUTF8": utf8_mode,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(pathlib.Path(inventory.__file__).parents[1]),
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, os.fsdecode(corpus)],
            check=True,
            capture_output=True,
            encoding="ascii",
            env=environment,
        )
        outputs.append(completed.stdout.strip())

    assert outputs[0] == outputs[1]
    assert bytes.fromhex(outputs[0]).decode("utf-8") == "entry_é.MOV"


def test_valid_utf8_unicode_whitespace_parses_in_ascii_locale(
    tmp_path: pathlib.Path,
) -> None:
    """N4 sees decoded UTF-8 text, independent of Python's filesystem codec."""
    corpus = os.fsencode(tmp_path / "corpus")
    os.mkdir(corpus)  # noqa: PTH102 - byte paths are the subject under test
    descriptor = os.open(
        corpus + b"/3\xc2\xa0above\xc2\xa0peg\xc2\xa0l.MOV",
        os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    os.close(descriptor)
    script = """
import pathlib
import sys
from pose_estimation.inventory import _exclusion_reason, _relative_posix, parse_stem
root = pathlib.Path(sys.argv[1])
entry = next(root.iterdir())
relative = _relative_posix(entry, root)
reason = _exclusion_reason(entry, root.resolve(), relative) or '<none>'
parsed = parse_stem(relative.rsplit('/', 1)[-1])
print(sys.getfilesystemencoding(), reason, parsed.reason_code, '|'.join(parsed.applied))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONUTF8": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(pathlib.Path(inventory.__file__).parents[1]),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, os.fsdecode(corpus)],
        check=True,
        capture_output=True,
        encoding="ascii",
        env=environment,
    )

    assert completed.stdout.strip() == "ascii <none> ok whitespace_collapsed"


def test_valid_utf8_c1_keeps_control_precedence_in_ascii_locale(
    tmp_path: pathlib.Path,
) -> None:
    """The byte-level UTF-8 check must preserve control-code classification."""
    corpus = os.fsencode(tmp_path / "corpus")
    os.mkdir(corpus)  # noqa: PTH102 - byte paths are the subject under test
    descriptor = os.open(corpus + b"/entry_\xc2\x80.MOV", os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)

    script = """
import pathlib
import sys
from pose_estimation.inventory import _exclusion_reason, _relative_posix
root = pathlib.Path(sys.argv[1])
entry = next(root.iterdir())
relative = _relative_posix(entry, root)
print(sys.getfilesystemencoding(), _exclusion_reason(entry, root.resolve(), relative))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONUTF8": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(pathlib.Path(inventory.__file__).parents[1]),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, os.fsdecode(corpus)],
        check=True,
        capture_output=True,
        encoding="ascii",
        env=environment,
    )

    assert completed.stdout.strip() == "ascii control_character_in_path"


def test_symlink_exclusions_do_not_read_or_count_target_bytes(tmp_path: pathlib.Path) -> None:
    """A symlink row never imports target metadata into the corpus census."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "target.MOV"
    target.write_bytes(b"abc")
    outside = tmp_path / "outside.MOV"
    outside.write_bytes(b"outside")
    (corpus / "inside-link.MOV").symlink_to(target)
    (corpus / "outside-link.MOV").symlink_to(outside)

    assets = inventory.build_assets(corpus, tmp_path / "out", checksums=False)
    by_source = {asset.source_path: asset for asset in assets}

    assert by_source["inside-link.MOV"].reason_code == "symlink_within_corpus"
    assert by_source["outside-link.MOV"].reason_code == "path_escapes_root"
    assert by_source["inside-link.MOV"].size_bytes is None
    assert by_source["outside-link.MOV"].size_bytes is None
    assert sum(asset.size_bytes or 0 for asset in assets) == len(b"abc")


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("is_opened", video_io.PROBE_OPEN_FAILED),
        ("closed_release", video_io.PROBE_OPEN_FAILED),
        ("opened_release", video_io.PROBE_OPENED),
    ],
)
def test_probe_container_contains_backend_lifecycle_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_status: str,
) -> None:
    """Lifecycle exceptions never escape; teardown cannot erase read facts."""

    class FailingCapture:
        def isOpened(self) -> bool:
            if failure == "is_opened":
                raise RuntimeError("isOpened failed")
            return failure == "opened_release"

        def release(self) -> None:
            raise RuntimeError("release failed")

        def set(self, _property: int, _value: int) -> bool:
            return True

        def getBackendName(self) -> str:
            return "FAKE"

        def get(self, _property: int) -> float:
            return 0.0

    monkeypatch.setattr(video_io.cv2, "VideoCapture", lambda _path: FailingCapture())

    assert video_io.probe_container("asset.MOV").probe_status == expected_status


def test_capture_fps_aggregates_are_member_order_independent() -> None:
    """A NaN report must not make family min/max depend on member order."""

    def member(asset_id: str, view: str, fps: float) -> inventory.AssetRecord:
        return inventory.AssetRecord(
            asset_id=asset_id,
            source_path=f"{asset_id}.MOV",
            disposition=inventory.CANONICAL,
            reason_code=inventory.REASON_OK,
            parse=inventory.StemParse(
                subject_ordinal=1,
                view=view,
                task="peg",
                side="l",
            ),
            facts=video_io.ContainerFacts(
                probe_status=video_io.PROBE_OPENED,
                backend_name="FAKE",
                reported_width=1,
                reported_height=1,
                reported_avg_fps=fps,
                reported_frame_count=30,
                reported_rotation_deg=0,
                reported_fourcc="MJPG",
                orientation_auto=True,
            ),
            size_bytes=1,
            content_sha256="",
        )

    finite = member("a-finite", "above", 30.0)
    nonfinite = member("a-nan", "left", float("nan"))
    forward = inventory.CaptureRecord("s01-peg-l", 1, "peg", "l", (finite, nonfinite))
    reverse = inventory.CaptureRecord("s01-peg-l", 1, "peg", "l", (nonfinite, finite))

    assert inventory.capture_row(forward) == inventory.capture_row(reverse)


def test_census_redacts_unsupported_extension_text() -> None:
    """An arbitrary filename suffix must not become a census key."""
    asset = inventory.AssetRecord(
        asset_id="a-unsupported",
        source_path="entry.patient-free-text",
        disposition=inventory.EXCLUDED,
        reason_code="unsupported_extension",
        parse=inventory.StemParse(reason_code="unsupported_extension"),
        facts=inventory._skipped_facts(),
        size_bytes=1,
        content_sha256="",
    )

    census = inventory.build_census(
        [asset],
        [],
        checksums=False,
        opencv_version="test",
        backend_name="",
    )

    assert census["extension_case"] == {"<unsupported>": 1}
    assert "patient-free-text" not in inventory.render_json(census)


def test_generation_validation_hashes_exact_table_bytes(tmp_path: pathlib.Path) -> None:
    """A CRLF rewrite is a different table generation on disk."""
    assets_text = "asset_id\n"
    captures_text = "capture_id\n"
    census = {
        "generation": {
            inventory.ASSETS_FILENAME: inventory._text_digest(assets_text),
            inventory.CAPTURES_FILENAME: inventory._text_digest(captures_text),
        }
    }
    census["generation"][inventory.CENSUS_FILENAME] = inventory.census_digest(census)
    (tmp_path / inventory.ASSETS_FILENAME).write_text(
        assets_text,
        encoding="utf-8",
        newline="",
    )
    (tmp_path / inventory.CAPTURES_FILENAME).write_text(
        captures_text,
        encoding="utf-8",
        newline="",
    )
    (tmp_path / inventory.CENSUS_FILENAME).write_text(
        inventory.render_json(census),
        encoding="utf-8",
        newline="",
    )
    inventory.validate_generation(tmp_path)

    (tmp_path / inventory.ASSETS_FILENAME).write_bytes(b"asset_id\r\n")

    with pytest.raises(inventory.InventoryError, match=r"assets\.csv"):
        inventory.validate_generation(tmp_path)


def test_shape_histogram_counts_every_nan_fps_asset() -> None:
    """Equal rendered shape keys must aggregate before entering the census map."""

    def opened_asset(asset_id: str) -> inventory.AssetRecord:
        return inventory.AssetRecord(
            asset_id=asset_id,
            source_path=f"{asset_id}.MOV",
            disposition=inventory.QUARANTINED,
            reason_code="token_count",
            parse=inventory.StemParse(reason_code="token_count"),
            facts=video_io.ContainerFacts(
                probe_status=video_io.PROBE_OPENED,
                backend_name="FAKE",
                reported_width=1,
                reported_height=1,
                reported_avg_fps=float("nan"),
                reported_frame_count=1,
                reported_rotation_deg=0,
                reported_fourcc="MJPG",
                orientation_auto=True,
            ),
            size_bytes=1,
            content_sha256="",
        )

    assets = [opened_asset("a-first"), opened_asset("a-second")]
    census = inventory.build_census(
        assets,
        [],
        checksums=False,
        opencv_version="test",
        backend_name="FAKE",
    )

    assert sum(census["shapes"].values()) == len(assets)
    assert census["shapes"] == {"1x1@nan/MJPG/rot0": 2}


def test_fourcc_preserves_printable_space_bytes() -> None:
    """A raw four-byte tag must not be stripped into a different codec."""
    code = sum(byte << (8 * index) for index, byte in enumerate(b"ABC "))

    assert video_io._fourcc_text(float(code)) == "ABC "


def test_capture_membership_rejects_duplicate_and_omitted_assets() -> None:
    """Exact coverage must detect a same-family duplicate that preserves counts."""
    parse = inventory.StemParse(
        subject_ordinal=1,
        view="above",
        task="peg",
        side="l",
    )
    first = inventory.AssetRecord(
        asset_id="a-first",
        source_path="first.MOV",
        disposition=inventory.CANONICAL,
        reason_code=inventory.REASON_OK,
        parse=parse,
        facts=inventory._skipped_facts(),
        size_bytes=1,
        content_sha256="",
    )
    second = inventory.AssetRecord(
        asset_id="a-second",
        source_path="second.MOV",
        disposition=inventory.CANONICAL,
        reason_code=inventory.REASON_OK,
        parse=parse,
        facts=inventory._skipped_facts(),
        size_bytes=1,
        content_sha256="",
    )
    capture = inventory.CaptureRecord(
        capture_id=first.capture_id,
        subject_ordinal=1,
        task="peg",
        side="l",
        assets=(first, first),
    )

    with pytest.raises(inventory.InventoryError, match=r"cover|membership"):
        inventory.check_invariants(
            [first, second],
            [capture],
            ["first.MOV", "second.MOV"],
        )


def test_identity_docstrings_state_the_bounded_guarantees() -> None:
    """Identity documentation must not promote pseudonyms into stronger facts."""
    module_text = pathlib.Path(inventory.__file__).read_text(encoding="utf-8")

    asset_doc = (inventory.asset_id_of.__doc__ or "").lower()
    module_lower = module_text.lower()

    assert "collision check" in asset_doc
    assert "unique without" not in asset_doc
    assert "uniqueness is a property" not in module_lower
    assert "trial identity" not in module_lower
    assert "family" in (inventory.CaptureRecord.__doc__ or "").lower()
    assert "family" in (inventory.build_captures.__doc__ or "").lower()
    assert "family" in (inventory.capture_row.__doc__ or "").lower()
    assert "pseudonym" in (inventory.capture_id_of.__doc__ or "").lower()
    assert "de-identified" not in module_lower

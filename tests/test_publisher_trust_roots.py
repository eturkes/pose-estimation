"""M2 review T5 R15/R16: the marker is a published set's trust root, so its own kind counts.

`inventory` and `sessions` read their markers with a plain `read_text` + `json.loads`,
while the four later publishers `lstat` the path and reject duplicate keys.  A symlinked
marker puts the trust root outside the set it certifies — and through the ownership check
lets a foreign directory license its own replacement — while a duplicate key lets one
document present one claim to a reader and another to the parser.

Every case here pairs its refusal with the untouched tree validating, because a validator
that refuses an empty directory refuses a symlinked marker for the wrong reason.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from collections.abc import Callable

import pytest

from pose_estimation import inventory, measure, sessions

test_sessions = pytest.importorskip("test_sessions")
test_measure = pytest.importorskip("test_measure")

_SCALAR_LINE = re.compile(r'^\s*"[^"]+":\s*(?!\{|\[)[^\n]*,\s*$')

#: Kinds a marker may not be.  `/dev/null` supplies the device arm, because creating a
#: device node needs a capability the suite must not require.
_NON_REGULAR = ("symlink", "dangling", "directory", "fifo", "device")


def _duplicate_a_scalar_key(text: str, *, inside: str | None = None) -> str:
    """Restate one scalar key of *text*, at the top level or inside the *inside* object.

    Line surgery rather than a rewritten document: `json.dumps` cannot emit a repeated
    key, and the defect only exists in bytes a parser has to resolve.
    """
    lines = text.splitlines(keepends=True)
    start = 0
    if inside is not None:
        start = next(index for index, line in enumerate(lines) if f'"{inside}"' in line)
    target = next(
        index for index in range(start + 1, len(lines)) if _SCALAR_LINE.match(lines[index])
    )
    return "".join([*lines[: target + 1], lines[target], *lines[target + 1 :]])


def _make_non_regular(marker: pathlib.Path, kind: str) -> None:
    """Replace *marker* with *kind*, preserving its bytes wherever the kind allows.

    The symlink arm keeps the exact published bytes reachable, so the only thing the
    validator can be refusing is the entry's kind.
    """
    payload = marker.read_bytes()
    marker.unlink()
    if kind == "symlink":
        elsewhere = marker.parent.parent / f"elsewhere-{marker.name}"
        elsewhere.write_bytes(payload)
        marker.symlink_to(elsewhere)
    elif kind == "dangling":
        marker.symlink_to(marker.parent / "absent.json")
    elif kind == "directory":
        marker.mkdir()
    elif kind == "fifo":
        os.mkfifo(marker)
    elif kind == "device":
        marker.symlink_to("/dev/null")
    else:  # pragma: no cover - the parametrisation is closed
        raise AssertionError(kind)


@pytest.fixture
def registry(tmp_path: pathlib.Path):
    """A published inventory generation that validates before any case touches it."""
    assets = [test_sessions._canonical(1, "above"), test_sessions._canonical(1, "left")]
    return test_sessions._write_registry(tmp_path, assets)


@pytest.fixture
def published_sessions(registry) -> pathlib.Path:
    sessions.run(inventory_dir=registry.root, corpus_root=registry.corpus, out_dir=registry.out)
    assert sessions.validate_generation(registry.out)
    return registry.out


_Case = tuple[pathlib.Path, Callable[[pathlib.Path], bool], type[Exception]]


def _inventory_case(registry) -> _Case:
    return (
        registry.root / inventory.CENSUS_FILENAME,
        inventory.validate_generation,
        inventory.InventoryError,
    )


def _sessions_case(out: pathlib.Path) -> _Case:
    return out / sessions.GENERATION_FILENAME, sessions.validate_generation, sessions.SessionsError


@pytest.mark.parametrize("kind", _NON_REGULAR)
def test_r15_inventory_refuses_a_marker_that_is_not_a_regular_file(registry, kind: str) -> None:
    marker, validate, error = _inventory_case(registry)
    assert validate(registry.root), "positive control: the untouched generation validates"
    _make_non_regular(marker, kind)
    with pytest.raises(error):
        validate(registry.root)


@pytest.mark.parametrize("kind", _NON_REGULAR)
def test_r15_sessions_refuses_a_marker_that_is_not_a_regular_file(
    published_sessions: pathlib.Path, kind: str
) -> None:
    marker, validate, error = _sessions_case(published_sessions)
    assert validate(published_sessions), "positive control: the untouched tree validates"
    _make_non_regular(marker, kind)
    with pytest.raises(error):
        validate(published_sessions)


@pytest.mark.parametrize("inside", [None, "generation"])
def test_r16_inventory_refuses_a_marker_stating_one_key_twice(registry, inside) -> None:
    marker, validate, error = _inventory_case(registry)
    assert validate(registry.root)
    marker.write_text(
        _duplicate_a_scalar_key(marker.read_text(encoding="utf-8"), inside=inside),
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(error):
        validate(registry.root)


@pytest.mark.parametrize("inside", [None, "inventory"])
def test_r16_sessions_refuses_a_marker_stating_one_key_twice(
    published_sessions: pathlib.Path, inside
) -> None:
    marker, validate, error = _sessions_case(published_sessions)
    assert validate(published_sessions)
    marker.write_text(
        _duplicate_a_scalar_key(marker.read_text(encoding="utf-8"), inside=inside),
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(error):
        validate(published_sessions)


def test_r15_r16_the_ownership_check_reads_the_marker_through_the_same_loader(
    published_sessions: pathlib.Path, registry
) -> None:
    """A symlinked marker must not license replacing the tree it points out of.

    `_assert_owned` is the second reader, and it is the one with teeth: publication
    deletes whatever the marker says this tool owns.
    """
    marker = published_sessions / sessions.GENERATION_FILENAME
    _make_non_regular(marker, "symlink")
    with pytest.raises(sessions.SessionsError):
        sessions.run(
            inventory_dir=registry.root, corpus_root=registry.corpus, out_dir=published_sessions
        )


def test_r15_read_marker_accepts_exactly_the_kind_the_publishers_write(
    tmp_path: pathlib.Path,
) -> None:
    """The shared loader's own positive control, so the refusals above are not vacuous."""
    marker = tmp_path / "census.json"
    marker.write_text('{"a": 1, "b": {"c": 2}}', encoding="utf-8")
    assert inventory.read_marker(marker) == {"a": 1, "b": {"c": 2}}
    with pytest.raises(OSError, match="not a regular file"):
        inventory.read_marker(pathlib.Path("/dev/null"))
    marker.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        inventory.read_marker(marker)


def test_r17_sessions_marker_digest_covers_its_upstream_claim(
    published_sessions: pathlib.Path,
) -> None:
    """T5 R17: an edited `inventory` block refuses, though every other digest still agrees."""
    marker = published_sessions / sessions.GENERATION_FILENAME
    generation = json.loads(marker.read_text(encoding="utf-8"))
    edited = {**generation, "inventory": {**generation["inventory"], "assets.csv": "0" * 64}}
    marker.write_text(json.dumps(edited, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(sessions.SessionsError, match="changed after publication"):
        sessions.validate_generation(published_sessions)
    # A keyless digest detects; recomputing it re-validates (the bound, never a signature).
    edited["marker"] = sessions.marker_digest(edited)
    marker.write_text(sessions.render_marker(edited), encoding="utf-8", newline="")
    assert sessions.validate_generation(published_sessions)["inventory"]["assets.csv"] == "0" * 64


def test_r17_measure_manifest_digest_covers_its_upstream_claim(tmp_path: pathlib.Path) -> None:
    """T5 R17: the manifest digest keeps `generation` minus its own `manifest` key."""
    out = test_measure._sidecar(tmp_path, [test_measure._row()])
    assert measure.validate(out), "positive control: the untouched sidecar validates"
    path = out / measure.MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["generation"]["inventory"] = {"assets.csv": "0" * 64}
    path.write_text(inventory.render_json(manifest), encoding="utf-8", newline="")
    with pytest.raises(measure.MeasureError) as refusal:
        measure.validate(out)
    assert refusal.value.reason == "manifest_edited"

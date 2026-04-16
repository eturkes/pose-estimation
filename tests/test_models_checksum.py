"""Tests for model checksum verification (no network calls)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pose_estimation.models import _sha256_of, _verify_checksum, download_file


def _write_blob(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_sha256_matches_hashlib(tmp_path):
    payload = b"hello pose" * 1024
    f = tmp_path / "blob.bin"
    expected = _write_blob(f, payload)
    assert _sha256_of(f) == expected


def test_verify_checksum_disabled_when_none(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"whatever")
    assert _verify_checksum(f, expected=None, redownload=True) is True


def test_verify_checksum_passes_on_match(tmp_path):
    f = tmp_path / "blob.bin"
    expected = _write_blob(f, b"good content")
    assert _verify_checksum(f, expected=expected, redownload=True) is True


def test_verify_checksum_redownload_removes_file(tmp_path, capsys):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"actual content")
    bogus = "0" * 64
    result = _verify_checksum(f, expected=bogus, redownload=True)
    assert result is False
    assert not f.exists()
    captured = capsys.readouterr()
    assert "checksum mismatch" in captured.out.lower()


def test_verify_checksum_strict_raises(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"actual content")
    bogus = "0" * 64
    with pytest.raises(ValueError, match="Checksum mismatch"):
        _verify_checksum(f, expected=bogus, redownload=False)


def test_download_file_skips_if_cached_and_valid(tmp_path):
    """No network call: existing valid file returns immediately."""
    f = tmp_path / "model.bin"
    payload = b"cached content"
    expected = _write_blob(f, payload)
    # If the function tried to fetch, it would error (no real URL),
    # so reaching this assertion proves the cached fast-path works.
    out = download_file("http://invalid.example.com/x", f, expected_sha256=expected)
    assert out == f
    assert f.read_bytes() == payload

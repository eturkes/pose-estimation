from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pose_estimation import qualify, sessions
from pose_estimation.measure import audio_offset


def test_fusion_agreement_tolerance_is_inclusive() -> None:
    row = {
        "status_audio": "ok",
        "status_visual": "ok",
        "offset_audio_s": "0.0",
        "offset_visual_s": str(qualify.AGREE_TOLERANCE_S),
    }

    assert qualify.fuse_pair(row) == qualify.PAIR_OK_CORROBORATED


def test_single_camera_event_has_no_offset_span() -> None:
    event = {
        "event_id": "event-1",
        "capture_id": "capture-1",
        "n_cameras": "1",
        "views": "above",
    }

    [row] = qualify._event_rows(
        [event], {"event-1": ["asset-b"]}, camera_rows=[], directed={}, sync_measured=True
    )

    assert row["graph_connected"] == "1"
    assert row["offset_span_s"] == qualify.UNMEASURED


def test_audio_cache_rejects_a_corrupt_coarse_array(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "signal.bin"
    cache = tmp_path / "cache"
    source.write_bytes(b"source")
    decode_calls = 0

    def decode(_path: pathlib.Path) -> tuple[np.ndarray, dict[str, int]]:
        nonlocal decode_calls
        decode_calls += 1
        return np.linspace(-1.0, 1.0, 64, dtype=np.float32), {
            "source_rate": audio_offset.TARGET_RATE
        }

    monkeypatch.setattr(audio_offset, "decode_audio", decode)
    audio_offset.ensure_cached(source, cache, "a" * 16)
    audio_offset.cache_paths(cache, "a" * 16)[1].write_bytes(b"corrupt")

    audio_offset.ensure_cached(source, cache, "a" * 16)

    assert decode_calls == 2


def test_load_placements_sorts_members_independent_of_table_order(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"event_id": "event-1", "asset_id": "asset-z", "placement": sessions.PLACED},
        {"event_id": "event-1", "asset_id": "asset-a", "placement": sessions.PLACED},
    ]

    def read_table(_path: pathlib.Path, _columns: tuple[str, ...]) -> list[dict[str, str]]:
        return rows

    monkeypatch.setattr(qualify, "_read_table", read_table)

    assert qualify.load_placements(tmp_path) == {"event-1": ["asset-a", "asset-z"]}


def test_closure_distribution_endpoints_ignore_event_order() -> None:
    event_rows = [
        {
            "n_cameras": "3",
            "sync_status": qualify.SYNC_CONNECTED,
            "sync_qualified": "1",
            "closure_residual_s": value,
        }
        for value in ("0.300000000", "0.100000000", "0.200000000")
    ]

    distribution = qualify.build_census(
        asset_rows=[], pair_rows=[], camera_rows=[], event_rows=event_rows
    )["events"]["closure_residual_s"]

    assert distribution is not None
    assert distribution["min"] == 0.1
    assert distribution["max"] == 0.3

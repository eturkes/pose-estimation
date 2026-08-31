from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pose_estimation import measure, qualify
from pose_estimation.measure import sync


def _asset(asset_id: str, capture_id: str = "capture") -> qualify.AssetRef:
    return qualify.AssetRef(
        asset_id=asset_id,
        capture_id=capture_id,
        view="above",
        task="task",
        side="left",
        subject_ordinal="1",
        source_relative=f"{asset_id}.mov",
        reported_frame_count=2,
    )


def _facts(device_config: str, audio_rate_hz: int | None) -> qualify.DecodeFacts:
    values: dict[str, Any] = {
        "status": qualify.DECODE_OK,
        "codec": "h264",
        "device_config": device_config,
        "frames_demuxed": 2,
        "dt_median_s": 0.033,
        "dt_p95_s": 0.033,
        "dt_max_s": 0.033,
        "monotonic": True,
        "audio_rate_hz": audio_rate_hz,
    }
    fields = {field.name for field in dataclasses.fields(qualify.DecodeFacts)}
    constructor = cast(Callable[..., qualify.DecodeFacts], qualify.DecodeFacts)
    return constructor(**{name: value for name, value in values.items() if name in fields})


def _asset_and_pair_rows(
    device_config: str, audio_rate_hz: int | None
) -> tuple[dict[str, str], dict[str, str]]:
    first = _asset("asset-a")
    second = _asset("asset-b")
    facts = {
        first.asset_id: _facts(device_config, audio_rate_hz),
        second.asset_id: _facts("other/os", 48_000),
    }
    asset_row = qualify._asset_row(
        first,
        facts[first.asset_id],
        qualify.OrientationFacts(present=True, values=(1,), changes=0),
        {},
    )
    pair_row = qualify._pair_rows([first, second], facts, {})[0]
    return asset_row, pair_row


def test_d01_census_uses_unordered_stratum_pair_key() -> None:
    low = "a-model/os/44100"
    high = "z-model/os/48000"
    pair_rows = [
        _pair_row(status=qualify.PAIR_UNMEASURED, stratum_a=high, stratum_b=low),
        _pair_row(status=qualify.PAIR_UNMEASURED, stratum_a=low, stratum_b=high),
    ]

    record, _ = _sync_stratum_record(pair_rows, f"{low}|{high}")

    assert record is not None
    assert record["pairs"] == 2


def test_d02_known_rate_without_device_config_keeps_partial_stratum_empty() -> None:
    asset_row, pair_row = _asset_and_pair_rows("", 48_000)

    assert (asset_row.get("audio_rate_hz"), pair_row.get("stratum_a")) == ("48000", "")


def test_d03_known_device_without_rate_keeps_partial_stratum_empty() -> None:
    asset_row, pair_row = _asset_and_pair_rows("model/os", None)

    assert (asset_row.get("audio_rate_hz"), pair_row.get("stratum_a")) == ("", "")


def test_d04_pair_booleans_follow_their_own_components() -> None:
    specifications = (
        ("capture-1", "asset-1a", "asset-1b", "m1/os1", 48_000, "m2/os2", 48_000),
        ("capture-2", "asset-2a", "asset-2b", "m/os", 44_100, "m/os", 48_000),
        ("capture-3", "asset-3a", "asset-3b", "m/os", None, "m/os", None),
    )
    assets: list[qualify.AssetRef] = []
    facts: dict[str, qualify.DecodeFacts] = {}
    for capture_id, id_a, id_b, config_a, rate_a, config_b, rate_b in specifications:
        first = _asset(id_a, capture_id)
        second = _asset(id_b, capture_id)
        assets.extend((first, second))
        facts[id_a] = _facts(config_a, rate_a)
        facts[id_b] = _facts(config_b, rate_b)

    rows = {row["capture_id"]: row for row in qualify._pair_rows(assets, facts, {})}

    assert [
        (
            rows[capture_id].get("stratum_a"),
            rows[capture_id].get("stratum_b"),
            rows[capture_id]["same_device_config"],
            rows[capture_id]["same_audio_rate"],
        )
        for capture_id in ("capture-1", "capture-2", "capture-3")
    ] == [
        ("m1/os1/48000", "m2/os2/48000", "0", "1"),
        ("m/os/44100", "m/os/48000", "1", "0"),
        ("", "", "1", ""),
    ]


def test_d05_sidecar_rate_with_missing_header_rate_is_a_hard_error() -> None:
    facts = {
        "asset-a": _facts("model/os", None),
        "asset-b": _facts("model/os", 48_000),
    }
    sync = {("asset-a", "asset-b"): {"audio_rate_a": "48000", "audio_rate_b": "48000"}}
    guard = getattr(qualify, "_assert_sidecar_rates", None)

    assert callable(guard)
    with pytest.raises(qualify.QualifyError) as caught:
        cast(Callable[..., None], guard)(facts, sync)
    assert caught.value.reason == "audio_rate_disagreement"


def test_d06_empty_sidecar_rate_does_not_contradict_known_header_rate() -> None:
    facts = {
        "asset-a": _facts("model/os", 48_000),
        "asset-b": _facts("model/os", 48_000),
    }
    sync = {("asset-a", "asset-b"): {"audio_rate_a": "", "audio_rate_b": "48000"}}
    guard = getattr(qualify, "_assert_sidecar_rates", None)

    assert callable(guard)
    cast(Callable[..., None], guard)(facts, sync)


def test_d07_different_sidecar_and_header_rates_are_a_hard_error() -> None:
    facts = {
        "asset-a": _facts("model/os", 44_100),
        "asset-b": _facts("model/os", 48_000),
    }
    sync = {("asset-a", "asset-b"): {"audio_rate_a": "48000", "audio_rate_b": "48000"}}
    guard = getattr(qualify, "_assert_sidecar_rates", None)

    assert callable(guard)
    with pytest.raises(qualify.QualifyError) as caught:
        cast(Callable[..., None], guard)(facts, sync)
    assert caught.value.reason == "audio_rate_disagreement"


def test_d08_stratum_alphabet_is_closed_and_accepts_model_only_config() -> None:
    patterns = [value for value in vars(qualify).values() if isinstance(value, re.Pattern)]
    valid = ("iPad/44100", "iPad/os/44100")
    invalid = (
        "iPad/0",
        "iPad/044100",
        "iPad/-44100",
        "iPad/44100\n",
        "iPad/os/44100/extra",
    )

    matching = [
        pattern
        for pattern in patterns
        if all(pattern.fullmatch(cell) for cell in valid)
        and not any(pattern.fullmatch(cell) for cell in invalid)
    ]
    assert matching

    _, pair_row = _asset_and_pair_rows("iPad", 44_100)
    assert pair_row.get("stratum_a") == "iPad/44100"


def test_d09_flagless_rows_publish_header_strata_but_not_estimators() -> None:
    asset_row, pair_row = _asset_and_pair_rows("model/os", 48_000)

    assert (
        asset_row.get("audio_rate_hz"),
        pair_row.get("stratum_a"),
        pair_row.get("stratum_b"),
        pair_row["same_device_config"],
        pair_row["same_audio_rate"],
        pair_row["offset_s"],
        pair_row["status"],
    ) == (
        "48000",
        "model/os/48000",
        "other/os/48000",
        "0",
        "1",
        "",
        qualify.PAIR_UNMEASURED,
    )


def test_d10_census_publishes_redaction_safe_singleton_strata() -> None:
    low = "model-a/os/44100"
    high = "model-b/os/48000"
    pair_rows = [
        _pair_row(
            status=qualify.PAIR_UNMEASURED,
            stratum_a=high,
            stratum_b=low,
            asset_a="synthetic-private-asset-a",
            asset_b="synthetic-private-asset-b",
            capture_id="synthetic-private-capture",
        )
    ]

    record, payload = _sync_stratum_record(pair_rows, f"{low}|{high}")

    assert record is not None
    assert record["pairs"] == 1
    assert "synthetic-private" not in payload


class _FakePacket:
    def __init__(self, pts: int) -> None:
        self.pts = pts


class _FakeCodecContext:
    name = "h264"


class _FakeVideoStream:
    codec_context = _FakeCodecContext()
    time_base = Fraction(1, 1_000)


class _FakeAudioStream:
    def __init__(self, rate: object) -> None:
        self.rate = rate


class _FakeStreams:
    def __init__(self, rates: tuple[object, ...]) -> None:
        self.video = [_FakeVideoStream()]
        self.audio = [_FakeAudioStream(rate) for rate in rates]


class _FakeContainer:
    def __init__(self, rates: tuple[object, ...]) -> None:
        self.metadata = {
            "com.apple.quicktime.model": "iPad",
            "com.apple.quicktime.software": "OS",
        }
        self.streams = _FakeStreams(rates)

    def __enter__(self) -> _FakeContainer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def demux(self, stream: object) -> list[_FakePacket]:
        return [_FakePacket(0), _FakePacket(33)]


def _probe_rates(monkeypatch: pytest.MonkeyPatch, *rates: object) -> qualify.DecodeFacts:
    monkeypatch.setattr(qualify.av, "open", lambda _: _FakeContainer(rates))
    return qualify.probe_decode(Path("synthetic.mov"))


@pytest.mark.parametrize("rate", [0, -1, 44_100.5])
def test_d11_invalid_rate_publishes_unmeasured_without_truncation(
    monkeypatch: pytest.MonkeyPatch, rate: object
) -> None:
    facts = _probe_rates(monkeypatch, rate)

    assert getattr(facts, "audio_rate_hz", "missing") is None
    asset_row = qualify._asset_row(
        _asset("asset-a"),
        facts,
        qualify.OrientationFacts(present=True, values=(1,), changes=0),
        {},
    )
    assert asset_row.get("audio_rate_hz") == ""


def test_d12_multiple_audio_streams_publish_the_selected_first_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = _probe_rates(monkeypatch, 44_100, 48_000)

    assert getattr(facts, "audio_rate_hz", None) == 44_100
    asset_row = qualify._asset_row(
        _asset("asset-a"),
        facts,
        qualify.OrientationFacts(present=True, values=(1,), changes=0),
        {},
    )
    assert asset_row.get("audio_rate_hz") == "44100"


def test_schema_bump_closes_all_three_p29_tables() -> None:
    assert qualify.GENERATOR_VERSION == "v4"
    assert "audio_rate_hz" in qualify.ASSETS_QC_COLUMNS
    assert {"stratum_a", "stratum_b", "same_device_config", "same_audio_rate"} <= set(
        qualify.PAIRS_QC_COLUMNS
    )
    assert {"audio_rate_a", "audio_rate_b", "same_audio_rate"} <= set(measure.SYNC_COLUMNS)


def test_sync_estimator_publishes_each_native_source_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peak = SimpleNamespace(
        lag_s=0.125,
        peak_rms=1.0,
        peak_ratio=2.0,
        status="ok",
        overlap_s=3.0,
    )
    drift = SimpleNamespace(ppm=4.0, standard_error=5.0)
    visual = SimpleNamespace(
        offset_s=0.125,
        confidence=1.0,
        peak_correlation=1.0,
        status="ok",
    )
    rates = {"asset-a": 44_100, "asset-b": 48_000}
    monkeypatch.setattr(sync.audio_offset, "estimate", lambda *_: (peak, drift, 6.0, 7.0))
    monkeypatch.setattr(sync.audio_offset, "source_rate", lambda _, asset_id: rates[asset_id])
    monkeypatch.setattr(sync.visual_offset, "load_signal", lambda *_: ([0.0], [0.0]))
    monkeypatch.setattr(sync.visual_offset, "estimate", lambda *_: visual)

    row = sync._estimate((sync.PairKey("capture", "asset-a", "asset-b"), "audio", "visual"))

    assert (row.get("audio_rate_a"), row.get("audio_rate_b"), row["same_audio_rate"]) == (
        "44100",
        "48000",
        "0",
    )


def test_sidecar_rate_guard_checks_asset_b_as_well_as_asset_a() -> None:
    facts = {
        "asset-a": _facts("model/os", 44_100),
        "asset-b": _facts("model/os", 44_100),
    }
    sync_rows = {("asset-a", "asset-b"): {"audio_rate_a": "44100", "audio_rate_b": "48000"}}
    guard = getattr(qualify, "_assert_sidecar_rates", None)

    assert callable(guard)
    with pytest.raises(qualify.QualifyError) as caught:
        cast(Callable[..., None], guard)(facts, sync_rows)
    assert caught.value.reason == "audio_rate_disagreement"


def _pair_row(**values: str) -> dict[str, str]:
    row = dict.fromkeys(qualify.PAIRS_QC_COLUMNS, "")
    row.update(values)
    return row


def _sync_stratum_record(
    pair_rows: list[dict[str, str]], key: str
) -> tuple[dict[str, Any] | None, str]:
    census = qualify.build_census(asset_rows=[], pair_rows=pair_rows, camera_rows=[], event_rows=[])
    pairs = cast(dict[str, Any], census["pairs"])
    strata = cast(dict[str, dict[str, Any]], pairs.get("sync_strata", {}))
    return strata.get(key), json.dumps(census, sort_keys=True)

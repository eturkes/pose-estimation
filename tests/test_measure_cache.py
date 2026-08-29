"""A decode cache must answer for the parameters that shaped it, not only for the bytes.

Both estimators key a cache on the asset id and validate it against the source
digest.  The digest binds the bytes that were decoded.  It does not bind the
rotation the frames were shaped with, nor the decode facts recorded beside the
arrays, and one of those facts reaches a published column.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import wave
from fractions import Fraction

import av
import numpy as np
import pytest

from pose_estimation.measure import audio_offset, visual_offset

TICKS_PER_SECOND = 600
FRAME_SIZE = (192, 108)


def _write_clip(path: pathlib.Path, frames: int = 90) -> str:
    """Encode a clip whose moving content is not symmetric under rotation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = FRAME_SIZE
        stream.pix_fmt = "yuv420p"
        # On the codec context, not the stream: the muxer rewrites the stream's
        # own time base, so a value set there never reaches the encoder.
        stream.codec_context.time_base = Fraction(1, TICKS_PER_SECOND)
        for index in range(frames):
            pixels = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            # A bright block sweeping one axis only: rotating the frame moves
            # the motion into the other axis, so the border and centre traces
            # of a rotated decode differ from an upright one.
            column = (index * 2) % (FRAME_SIZE[0] - 16)
            pixels[8:32, column : column + 16] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index * 20
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: pathlib.Path, frequency_hz: float, *, rate: int = 16000) -> None:
    sample_index = np.arange(rate * 2)
    samples = (np.sin(2 * np.pi * frequency_hz * sample_index / rate) * 10000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


def test_visual_cache_is_rebuilt_when_the_rotation_changes(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "clip.mp4"
    cache = tmp_path / "cache"
    asset_id = "c" * 16
    digest = _write_clip(source)

    assert visual_offset.ensure_cached(source, cache, asset_id, digest, 0) == "decoded"
    assert visual_offset.ensure_cached(source, cache, asset_id, digest, 0) == "cached"
    upright = visual_offset.load_signal(cache, asset_id)[1].copy()

    assert visual_offset.ensure_cached(source, cache, asset_id, digest, 90) == "decoded"
    rotated = visual_offset.load_signal(cache, asset_id)[1]
    assert not np.array_equal(upright, rotated)


def test_visual_cache_rebuilt_under_rotation_matches_a_clean_decode(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "clip.mp4"
    reused = tmp_path / "reused"
    clean = tmp_path / "clean"
    asset_id = "d" * 16
    digest = _write_clip(source)

    visual_offset.ensure_cached(source, reused, asset_id, digest, 0)
    visual_offset.ensure_cached(source, reused, asset_id, digest, 90)
    visual_offset.ensure_cached(source, clean, asset_id, digest, 90)

    assert (reused / f"{asset_id}.npz").read_bytes() == (clean / f"{asset_id}.npz").read_bytes()


def test_audio_cache_rejects_edited_decode_facts(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "signal.wav"
    cache = tmp_path / "cache"
    asset_id = "e" * 16
    _write_wav(source, 440.0, rate=16000)
    audio_offset.ensure_cached(source, cache, asset_id)
    assert audio_offset.source_rate(cache, asset_id) == 16000

    # The arrays and their digests stay untouched: this is an edit that stops
    # at the claim, and source_rate is the claim sync publishes as
    # audio_rate_a/audio_rate_b, which qualify turns into P29's stratum.
    _, _, timing = audio_offset.cache_paths(cache, asset_id)
    facts = json.loads(timing.read_text(encoding="utf-8"))
    facts["source_rate"] = 48000
    timing.write_text(json.dumps(facts, sort_keys=True) + "\n", encoding="utf-8")
    assert audio_offset.source_rate(cache, asset_id) == 48000

    audio_offset.ensure_cached(source, cache, asset_id)
    assert audio_offset.source_rate(cache, asset_id) == 16000


def test_audio_cache_repaired_from_edited_facts_matches_a_clean_decode(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "signal.wav"
    cache = tmp_path / "cache"
    clean = tmp_path / "clean"
    asset_id = "f" * 16
    _write_wav(source, 440.0)
    audio_offset.ensure_cached(source, cache, asset_id)

    _, _, timing = audio_offset.cache_paths(cache, asset_id)
    facts = json.loads(timing.read_text(encoding="utf-8"))
    facts["skip_samples"] = facts["skip_samples"] + 1
    timing.write_text(json.dumps(facts, sort_keys=True) + "\n", encoding="utf-8")

    audio_offset.ensure_cached(source, cache, asset_id)
    audio_offset.ensure_cached(source, clean, asset_id)
    repaired = {path.name: path.read_bytes() for path in audio_offset.cache_paths(cache, asset_id)}
    oracle = {path.name: path.read_bytes() for path in audio_offset.cache_paths(clean, asset_id)}
    assert repaired == oracle


def test_audio_cache_is_rebuilt_when_an_analysis_rate_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "signal.wav"
    cache = tmp_path / "cache"
    asset_id = "1" * 16
    _write_wav(source, 440.0)
    audio_offset.ensure_cached(source, cache, asset_id)
    _, coarse, _ = audio_offset.cache_paths(cache, asset_id)
    before = coarse.read_bytes()

    # Every recorded digest stays correct across this change: the arrays are
    # still the ones that were written.  They were written at the old rate.
    monkeypatch.setattr(audio_offset, "COARSE_RATE", audio_offset.COARSE_RATE * 2)
    audio_offset.ensure_cached(source, cache, asset_id)
    assert coarse.read_bytes() != before


def test_audio_cache_accepts_its_own_unedited_facts(tmp_path: pathlib.Path) -> None:
    """The positive control: the digest must not reject an untouched cache."""
    source = tmp_path / "signal.wav"
    cache = tmp_path / "cache"
    asset_id = "0" * 16
    _write_wav(source, 440.0)
    audio_offset.ensure_cached(source, cache, asset_id)
    _, _, timing = audio_offset.cache_paths(cache, asset_id)
    before = timing.read_bytes()

    audio_offset.ensure_cached(source, cache, asset_id)
    assert timing.read_bytes() == before

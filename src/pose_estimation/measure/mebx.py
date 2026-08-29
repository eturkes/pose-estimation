"""Read Apple ``mebx`` timed-metadata tracks straight out of the MP4 atom tree.

PyAV exposes the packets of a timed-metadata track but not the ``stsd`` key
declarations that name them, so a packet's key id resolves to a key only by
parsing the container.  Two axes need that: rigidity excludes an asset whose
``video-orientation`` track changes mid-clip, and detect rotates each sampled
frame by the orientation in force at its own timestamp.  The corpus carries 7
assets that change orientation inside one clip and 3 with no track at all, so
the single display matrix a decoder applies cannot express either case.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import BinaryIO

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"meta"}


def atoms(stream: BinaryIO, end: int) -> list[tuple[bytes, int, int]]:
    """List ``(kind, body_offset, end_offset)`` for the atoms in ``[tell, end)``."""
    found: list[tuple[bytes, int, int]] = []
    while stream.tell() + 8 <= end:
        start = stream.tell()
        header = stream.read(8)
        if len(header) < 8:
            break
        size, kind = struct.unpack(">I4s", header)
        body = start + 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                break
            (size,) = struct.unpack(">Q", extended)
            body = start + 16
        elif size == 0:
            size = end - start
        if size < body - start or start + size > end:
            break
        found.append((kind, body, start + size))
        stream.seek(start + size)
    return found


def declared_keys(payload: bytes) -> list[str]:
    """Pull the ``keyd`` key names out of one ``mebx`` sample-entry payload."""
    keys: list[str] = []
    offset = payload.find(b"keyd")
    while offset >= 0:
        if offset >= 4:
            (size,) = struct.unpack(">I", payload[offset - 4 : offset])
            if 12 <= size <= len(payload) - offset + 4:
                keys.append(payload[offset + 8 : offset - 4 + size].decode("utf-8", "replace"))
        offset = payload.find(b"keyd", offset + 4)
    return keys


def key_maps(path: Path) -> list[dict[int, str]]:
    """Map each metadata track's 1-based key id to its declared key name.

    The result is ordered as the tracks appear in ``moov``, which is the order
    PyAV reports its non-audio, non-video streams in, so the two zip together.
    ``stsd`` is walked entry by entry rather than scanned whole: a key id counts
    from 1 across the entries of one track, and bounding each search to its own
    entry keeps a stray byte sequence in the table header out of that count.
    """
    tracks: list[tuple[str | None, list[str]]] = []
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(0)
        moov = next(
            ((body, stop) for kind, body, stop in atoms(stream, end) if kind == b"moov"), None
        )
        if moov is None:
            return []
        stream.seek(moov[0])
        for kind, body, stop in atoms(stream, moov[1]):
            if kind != b"trak":
                continue
            handler: str | None = None
            keys: list[str] = []
            stack = [(body, stop)]
            while stack:
                start, child_end = stack.pop()
                stream.seek(start)
                for child_kind, child_body, child_stop in atoms(stream, child_end):
                    if child_kind in CONTAINERS:
                        stack.append((child_body, child_stop))
                    elif child_kind == b"hdlr":
                        stream.seek(child_body)
                        raw = stream.read(min(child_stop - child_body, 24))
                        # `alis` is the reference handler of an alias track, never a payload kind.
                        if len(raw) >= 12 and raw[8:12] != b"alis":
                            handler = raw[8:12].decode("latin-1")
                    elif child_kind == b"stsd":
                        stream.seek(child_body)
                        payload = stream.read(child_stop - child_body)
                        offset = 8
                        while offset + 8 <= len(payload):
                            (size,) = struct.unpack(">I", payload[offset : offset + 4])
                            if size < 8 or offset + size > len(payload):
                                break
                            keys.extend(declared_keys(payload[offset : offset + size]))
                            offset += size
            tracks.append((handler, keys))
    return [
        {index + 1: key for index, key in enumerate(keys)}
        for handler, keys in tracks
        if handler == "meta"
    ]


def sample_entries(payload: bytes) -> list[tuple[int, bytes]]:
    """Split one timed-metadata packet into its ``(key_id, value)`` entries."""
    entries: list[tuple[int, bytes]] = []
    offset = 0
    while offset + 8 <= len(payload):
        size, key_id = struct.unpack(">II", payload[offset : offset + 8])
        if size < 8 or offset + size > len(payload):
            break
        entries.append((key_id, payload[offset + 8 : offset + size]))
        offset += size
    return entries

"""FastAPI application: three read-only views over the published trees.

Nothing here writes.  Every path a request can reach comes out of the clip index
built from the published tables, so a request supplies a key and never a path.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import census, clips, cohort
from .config import STATIC_DIR, Paths

RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 1 << 20
CONTENT_TYPES = {".mov": "video/quicktime", ".mp4": "video/mp4", ".m4v": "video/x-m4v"}


def _content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower()) or (
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )


def _ranged(path: Path, header: str | None) -> Response:
    """Serve one file, honouring a single byte range.

    Seeking a multi-minute clip needs 206 responses; a whole-file 200 makes the
    browser refetch from zero on every scrub.
    """
    size = path.stat().st_size
    match = RANGE_PATTERN.fullmatch(header.strip()) if header else None
    if match is None:
        return FileResponse(
            path, media_type=_content_type(path), headers={"accept-ranges": "bytes"}
        )

    start_text, end_text = match.groups()
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:  # suffix range: the last N bytes
        start = max(size - int(end_text or 0), 0)
        end = size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(status_code=416, headers={"content-range": f"bytes */{size}"})

    def stream():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                block = handle.read(min(CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(
        stream(),
        status_code=206,
        media_type=_content_type(path),
        headers={
            "content-range": f"bytes {start}-{end}/{size}",
            "content-length": str(end - start + 1),
            "accept-ranges": "bytes",
        },
    )


def create_app(paths: Paths | None = None) -> FastAPI:
    resolved = paths or Paths.resolve()
    app = FastAPI(title="pose-estimation review UI", docs_url=None, redoc_url=None)
    app.state.paths = resolved

    @app.middleware("http")
    async def revalidate(request: Request, call_next):
        """A response with no `Cache-Control` gets a heuristic freshness lifetime
        from `Last-Modified` — about a tenth of the file's age — so a browser
        serves an edited module from disk for hours without asking.  `no-cache`
        means revalidate, not no-store: the `ETag` still answers 304 and the
        published trees are equally free to move under a running server.
        """
        response = await call_next(request)
        response.headers.setdefault("cache-control", "no-cache")
        return response

    def _clip(event_id: str, camera_name: str) -> dict[str, Any]:
        found = clips.find(resolved, event_id, camera_name)
        if found is None:
            raise HTTPException(status_code=404, detail="unknown clip")
        return found

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"repo": str(resolved.repo), "available": resolved.status()}

    @app.get("/api/census")
    def census_view() -> JSONResponse:
        return JSONResponse(census.bundle(resolved))

    @app.get("/api/cohort")
    def cohort_view() -> JSONResponse:
        return JSONResponse(cohort.bundle(resolved))

    @app.get("/api/clips")
    def clip_index() -> JSONResponse:
        return JSONResponse({"clips": clips.index(resolved)})

    @app.get("/api/clip/{event_id}/{camera_name}/landmarks")
    def clip_landmarks(event_id: str, camera_name: str) -> JSONResponse:
        return JSONResponse(clips.landmarks(resolved, _clip(event_id, camera_name)))

    @app.get("/api/clip/{event_id}/{camera_name}/video")
    def clip_video(event_id: str, camera_name: str, request: Request) -> Response:
        path = clips.video_path(resolved, _clip(event_id, camera_name))
        if path is None:
            raise HTTPException(status_code=404, detail="no media for this clip")
        return _ranged(path, request.headers.get("range"))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app

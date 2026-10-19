"""Entry point: `uv run --directory tools/review-ui python -m review_ui`."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .config import DEFAULT_REPO, Paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Local review UI over the 3-cam corpus.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument(
        "--repo",
        default=None,
        help=f"Repository root holding the published trees (default: {DEFAULT_REPO}).",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    paths = Paths.resolve(args.repo)
    available = paths.status()
    print(f"repo   {paths.repo}")
    for name, present in available.items():
        print(f"  {'ok     ' if present else 'absent '}{name}")
    print(f"serving http://{args.host}:{args.port}/")
    uvicorn.run(create_app(paths), host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regenerate `proof/` — the four view captures and the API transcript.

Runs the server exactly as the recorded run command does, probes every endpoint
the three views consume, and captures each view through `webcap`.  The transcript
carries no timestamp, host path or process id, so a rerun over the same trees
rewrites it byte for byte and a diff means the artifact moved.

The captures are byte-stable once the font cache is warm — three consecutive
reruns agreed on all four — but the first run after `build_assets.py` rewrote the
subsets differed on one full-page capture, which is a layout settle rather than a
change in the UI.  Read a capture diff visually before believing it.

    python tools/capture_proof.py [--port 8791] [--keep]

`webcap` is a machine-local capture tool (CDP over chromiumfish).  Without it the
transcript still regenerates and the script reports which captures it skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROOF = ROOT / "proof"
#: Every capture pins its theme.  The UI defaults to `auto`, so an unpinned
#: capture would record the colour scheme of whatever machine took it; `light` is
#: the proof baseline and the one dark row is what shows the theme control
#: working.  A pinned theme keeps the baseline names unsuffixed.
VIEWS = (
    ("census", "ja", "light"),
    ("player", "ja", "light"),
    ("cohort", "ja", "light"),
    ("census", "en", "light"),
    ("census", "ja", "dark"),
)
CAPTURE_SIZE = {"census": (1500, 1000), "cohort": (1500, 1000), "player": (1500, 1150)}
FULL_PAGE = {"census", "cohort"}


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base}{path}", timeout=60) as response:
        return json.load(response)


def wait_ready(base: str, process: subprocess.Popen, seconds: float = 40.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited early with rc={process.returncode}")
        try:
            get(base, "/api/status")
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.4)
        else:
            return
    raise SystemExit("server did not answer /api/status")


def ranged(base: str, path: str) -> tuple[int, str, int]:
    request = urllib.request.Request(f"{base}{path}", headers={"Range": "bytes=0-65535"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.headers.get("content-range", "—"), len(response.read())


def section(title: str, pairs: list[tuple[str, object]]) -> list[str]:
    width = max((len(key) for key, _ in pairs), default=0)
    return [title, *[f"  {key.ljust(width)}  {value}" for key, value in pairs], ""]


def transcript(base: str) -> list[str]:
    status = get(base, "/api/status")
    census = get(base, "/api/census")
    clips = get(base, "/api/clips")["clips"]
    cohort = get(base, "/api/cohort")
    fixture = get(base, "/api/clip/fixture/demo/landmarks")
    topology = get(base, "/static/topology.json")
    code, content_range, body = ranged(base, "/api/clip/fixture/demo/video")

    families = Counter(clip["family_no"] for clip in clips if clip["family_no"])
    sizes = Counter(families.values())
    lines = [
        "review-ui proof — API transcript",
        "",
        "run command",
        "  uv run --directory prototype/review-ui python -m review_ui",
        f"  -> {base}/",
        "",
    ]
    lines += section(
        "published trees",
        [
            (name, "present" if present else "absent")
            for name, present in sorted(status["available"].items())
        ],
    )
    lines += section(
        "GET /api/census",
        [
            ("headline tiles", len(census["headline"])),
            ("capture shapes", len(census["shapes"])),
            ("qc flags", len(census["qc_flags"])),
            ("reason codes", len(census["reason_codes"])),
            (
                "run verdicts",
                f"{sum(census['run_verdicts'].values())} / {len(census['run_verdicts'])} true",
            ),
            ("ruling claims", len(census["claims"])),
            ("claim evidence", len(census["evidence"])),
            ("generator versions", len(census["provenance"])),
        ],
    )
    lines += section(
        "GET /api/clips",
        [
            ("clips", len(clips)),
            ("with landmarks", sum(1 for clip in clips if clip["has_landmarks"])),
            ("with video", sum(1 for clip in clips if clip["has_video"])),
            ("synthetic", sum(1 for clip in clips if clip["synthetic"])),
            ("recording events", len(families)),
            ("views per event", " · ".join(f"{n}v x{count}" for n, count in sorted(sizes.items()))),
        ],
    )
    lines += section(
        "GET /api/clip/fixture/demo/landmarks",
        [
            ("frames", fixture["frames"]),
            ("body keypoints", len(fixture["body_names"])),
            ("hand keypoints", fixture["hand_points"]),
            ("persons max", fixture["persons_max"]),
            ("normalising scale", fixture["scale"]),
        ],
    )
    lines += section(
        "GET /api/clip/fixture/demo/video  Range: bytes=0-65535",
        [("status", code), ("content-range", content_range), ("bytes returned", body)],
    )
    lines += section(
        "GET /api/cohort",
        [
            ("cells", len(cohort["cells"])),
            ("features", len(cohort["features"])),
            ("feature rows", len(cohort["rows"])),
            ("labelled ja+en", sum(1 for f in cohort["features"] if f.get("ja") and f.get("en"))),
            ("columns excluded", len(cohort["columns_excluded"])),
            *sorted((f"population.{k}", v) for k, v in cohort["population"].items()),
        ],
    )
    lines += section(
        "GET /static/topology.json",
        [
            ("body segments", len(topology["body"]["segments"])),
            ("body chains", len(topology["body"]["chains"])),
            ("hand segments", len(topology["hand"]["segments"])),
            ("hand chains", len(topology["hand"]["chains"])),
        ],
    )
    return lines


def capture(base: str, port: int) -> list[str]:
    if shutil.which("webcap") is None:
        return ["captures", "  webcap absent — no PNG written", ""]
    written = []
    for name, lang, theme in VIEWS:
        width, height = CAPTURE_SIZE[name]
        suffix = "" if theme == "light" else f"-{theme}"
        target = PROOF / f"{name}-{lang}{suffix}.png"
        command = [
            "webcap",
            f"{base}/?lang={lang}&theme={theme}#{name}",
            "--png",
            str(target),
            "--width",
            str(width),
            "--height",
            str(height),
            "--wait",
            "7000",
        ]
        if name in FULL_PAGE:
            command.append("--full-page")
        subprocess.run(command, check=True, capture_output=True, text=True)
        written.append(
            (
                target.name,
                f"{width}x{height} · {theme}{' · full-page' if name in FULL_PAGE else ''}",
            )
        )
    return section(f"captures  (webcap, port {port})", written)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--keep", action="store_true", help="leave the server running")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    PROOF.mkdir(exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-m", "review_ui", "--port", str(args.port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_ready(base, process)
        lines = transcript(base) + capture(base, args.port)
    finally:
        if not args.keep:
            process.terminate()
            process.wait(timeout=20)

    (PROOF / "run.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

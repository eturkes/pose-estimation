"""Review UI request-boundary check over a synthetic repo (no corpus data, no capture).

Usage (repo root): uv run --directory tools/review-ui python tools/security_qa.py
Serves `python -m review_ui --repo <tmp>` on a free loopback port, holding one synthetic
labelling job, and grades the server against browser-mediated requests from another site:
a rebinding Host, a cross-site simple POST, a malformed body, and the access log.
rc 0 <=> every row passes.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[1]
ANSWER = {
    "width": 640,
    "height": 480,
    "task": "task",
    "points": {"L.hand.wrist": {"state": "vis", "x": 10, "y": 20}},
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> int:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except urllib.error.URLError:
        return 0


def main() -> int:
    rows: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        user = repo / ".scratch" / "bench" / "user"
        (user / "frames").mkdir(parents=True)
        (user / "jobs.json").write_text(
            json.dumps([{"job": 0, "event_id": "e0", "camera_name": "c0", "frame": 5}])
        )
        (user / "frames" / "000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        labels = user / "labels.jsonl"
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        log = repo / "server.log"
        with log.open("w") as stream:
            server = subprocess.Popen(
                [sys.executable, "-m", "review_ui", "--repo", str(repo), "--port", str(port)],
                cwd=UI_DIR, stdout=stream, stderr=subprocess.STDOUT,
            )  # fmt: skip
        try:
            deadline = time.monotonic() + 30
            while _request(f"{base}/api/status") != 200:
                if time.monotonic() > deadline or server.poll() is not None:
                    print("FAIL server never answered")
                    return 1
                time.sleep(0.2)

            status = _request(f"{base}/api/label", headers={"Host": f"rebound.example:{port}"})
            rows.append(("foreign Host refused", status == 400, f"status {status}"))
            status = _request(f"{base}/api/label", headers={"Host": f"localhost:{port}"})
            rows.append(("loopback Host served", status == 200, f"status {status}"))

            body = json.dumps(ANSWER).encode()
            status = _request(
                f"{base}/api/label/0/answer", method="POST", body=body,
                headers={"Content-Type": "text/plain", "Origin": "https://attacker.example"},
            )  # fmt: skip
            rows.append(
                ("cross-site simple POST refused", status in (403, 415), f"status {status}")
            )
            rows.append(
                (
                    "cross-site POST stored nothing",
                    not labels.exists(),
                    f"labels present={labels.exists()}",
                )
            )
            status = _request(
                f"{base}/api/label/0/answer", method="POST", body=body,
                headers={"Content-Type": "application/json", "Origin": "https://attacker.example"},
            )  # fmt: skip
            rows.append(("foreign Origin JSON POST refused", status == 403, f"status {status}"))

            status = _request(
                f"{base}/api/label/0/answer", method="POST", body=b"{not json",
                headers={"Content-Type": "application/json", "Origin": base},
            )  # fmt: skip
            rows.append(("malformed JSON → 400", status == 400, f"status {status}"))
            status = _request(
                f"{base}/api/label/0/answer", method="POST", body=b'{"width": 1, "height": 1, "points": []}',
                headers={"Content-Type": "application/json", "Origin": base},
            )  # fmt: skip
            rows.append(("non-object points → 400", status == 400, f"status {status}"))

            status = _request(
                f"{base}/api/label/0/answer", method="POST", body=body,
                headers={"Content-Type": "application/json", "Origin": base},
            )  # fmt: skip
            stored = labels.exists() and '"xy": [10.0, 20.0]' in labels.read_text()
            rows.append(
                (
                    "same-origin JSON POST stored",
                    status == 200 and stored,
                    f"status {status} stored={stored}",
                )
            )

            _request(f"{base}/api/clip/e0/c0/landmarks")
        finally:
            server.terminate()
            server.wait(timeout=10)
        text = log.read_text()
        rows.append(
            (
                "access log carries no request path",
                "/api/clip/e0" not in text,
                f"{text.count('/api/')} logged paths",
            )
        )

    for name, ok, detail in rows:
        print(f"{'pass' if ok else 'FAIL'} {name} — {detail}")
    failed = sum(not ok for _, ok, _ in rows)
    print(f"{len(rows) - failed}/{len(rows)} pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

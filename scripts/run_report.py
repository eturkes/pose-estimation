#!/usr/bin/env python3
"""Provenance manifest + coverage QA for a produced pose-CSV batch.

Reads a directory of per-video pose CSVs and the sources they came from, and
writes two artifacts beside the data:

* ``manifest.json`` — machine-readable provenance: run configuration, resolved
  toolchain versions, git commit, and per-video frame/row/checksum records.
  Downstream consumers join on ``source_stem``.
* ``qa_report.md`` — the same evidence rendered for review, with source stems
  replaced by ordinals so the table can be quoted without carrying capture
  identifiers.

Coverage is measured against the decoded source rather than the container's
declared frame count, so a truncated or partially undecodable clip shows up as
missing coverage instead of passing silently.

Usage:
    scripts/run_report.py <csv_dir> --videos-dir videos/ [--tracking body]
        [--model rtmw-l] [--det-device CPU] [--pose-device NPU] [--backend openvino]
        [--det-frequency 7] [--single-subject] [--out-dir <csv_dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pose_estimation.export import make_csv_header

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_HASH_BUFFER = 1 << 20


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_BUFFER), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=pathlib.Path(__file__).resolve().parents[1],
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def _versions() -> dict[str, str | None]:
    def _v(module: str) -> str | None:
        try:
            mod = __import__(module)
        except ImportError:
            return None
        version = getattr(mod, "__version__", None)
        if version:
            return version
        # rtmlib and friends ship no __version__; fall back to install metadata.
        try:
            return importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "openvino": _v("openvino"),
        "rtmlib": _v("rtmlib"),
        "opencv": _v("cv2"),
        "numpy": _v("numpy"),
        "pose_estimation": _v("pose_estimation"),
    }


def _decode_source(path: pathlib.Path) -> dict:
    """Decode *path* end to end; report declared vs actually decodable frames."""
    cap = cv2.VideoCapture(str(path))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is not None and frame.size:
            decoded += 1
    cap.release()
    return {
        "declared_frames": declared,
        "decoded_frames": decoded,
        "fps": round(fps, 4) if fps else None,
        "duration_sec": round(decoded / fps, 3) if fps else None,
        "width": width,
        "height": height,
    }


def _flat_numeric(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Flatten *cols* into a single numeric Series; unparseable cells become NaN."""
    if not cols:
        return pd.Series(dtype=float)
    return pd.to_numeric(pd.Series(df[cols].to_numpy().ravel()), errors="coerce")


def _obs_rate(df: pd.DataFrame, cols: list[str]) -> float:
    """Fraction of rows where *cols* are all present and not all zero."""
    if not cols:
        return float("nan")
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    return float((sub.notna().all(axis=1) & (sub != 0).any(axis=1)).mean())


def _measure_csv(csv_path: pathlib.Path, tracking: str, decoded_frames: int) -> dict:
    expected = make_csv_header(tracking)
    df = pd.read_csv(csv_path)
    idx = pd.to_numeric(df["frame_idx"], errors="coerce").dropna().astype(int)
    ts = pd.to_numeric(df["timestamp_sec"], errors="coerce").dropna()

    prefix = "body" if tracking == "body" else "arm"
    wrist = [c for c in (f"{prefix}_left_wrist_x", f"{prefix}_left_wrist_y") if c in df.columns]
    lhand = [c for c in df.columns if c.startswith("left_hand_") and c.endswith("_x")]
    rhand = [c for c in df.columns if c.startswith("right_hand_") and c.endswith("_x")]
    vis_cols = [c for c in df.columns if c.startswith(prefix + "_") and c.endswith("_vis")]
    conf_cols = [c for c in df.columns if c.endswith("_conf")]

    vis = _flat_numeric(df, vis_cols)
    conf = _flat_numeric(df, conf_cols)

    # Plausibility, not just conformance: a corrupt detector (see
    # SplitDeviceSolution) still fills every column, so schema checks pass while
    # the boxes are nonsense.  Normalised landmarks must sit in [0, 1], and a
    # tracked wrist must move continuously rather than teleport between frames.
    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y")) and not c.startswith("frame")]
    coords = _flat_numeric(df, coord_cols).dropna()
    out_of_range = float(((coords < 0) | (coords > 1)).mean()) if len(coords) else float("nan")
    wrist_step = float("nan")
    if len(wrist) == 2 and len(df) > 1:
        wx = pd.to_numeric(df[wrist[0]], errors="coerce")
        wy = pd.to_numeric(df[wrist[1]], errors="coerce")
        step = np.hypot(wx.diff(), wy.diff()).dropna()
        wrist_step = float(step.median()) if len(step) else float("nan")

    return {
        "coord_out_of_unit_range_pct": (
            round(100 * out_of_range, 4) if np.isfinite(out_of_range) else None
        ),
        "median_wrist_step_norm": round(wrist_step, 5) if np.isfinite(wrist_step) else None,
        "rows": len(df),
        "unique_frames": idx.nunique(),
        "coverage_pct": (
            round(100.0 * idx.nunique() / decoded_frames, 3) if decoded_frames else None
        ),
        "index_gaps": int((idx.diff().dropna() > 1).sum()),
        "max_timestamp_gap_sec": round(float(ts.diff().dropna().max()), 4) if len(ts) > 1 else None,
        "schema_conformant": list(df.columns) == expected,
        "n_columns": int(df.shape[1]),
        "body_wrist_obs_pct": round(100 * _obs_rate(df, wrist), 3) if wrist else None,
        "left_hand_obs_pct": (
            round(100 * float(_flat_numeric(df, lhand).notna().mean()), 3) if lhand else None
        ),
        "right_hand_obs_pct": (
            round(100 * float(_flat_numeric(df, rhand).notna().mean()), 3) if rhand else None
        ),
        "median_visibility": round(float(vis[vis > 0].median()), 4) if (vis > 0).any() else None,
        "median_confidence": round(float(conf[conf > 0].median()), 4) if (conf > 0).any() else None,
        "sha256": _sha256(csv_path),
        "bytes": csv_path.stat().st_size,
    }


def build_manifest(args: argparse.Namespace) -> dict:
    csv_dir = pathlib.Path(args.csv_dir)
    videos_dir = pathlib.Path(args.videos_dir)
    sources = {
        p.stem: p for p in sorted(videos_dir.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES
    }

    records = []
    for ordinal, csv_path in enumerate(sorted(csv_dir.glob("*.csv")), 1):
        stem = csv_path.stem
        source = sources.get(stem)
        src_info = _decode_source(source) if source is not None else {}
        record = {
            "ordinal": ordinal,
            "source_stem": stem,
            "csv": csv_path.name,
            "source_present": source is not None,
            **src_info,
            **_measure_csv(csv_path, args.tracking, src_info.get("decoded_frames", 0)),
        }
        records.append(record)

    missing = sorted(set(sources) - {r["source_stem"] for r in records})
    return {
        "run": {
            "model": args.model,
            "tracking": args.tracking,
            "single_subject": args.single_subject,
            "backend": args.backend,
            "det_device": args.det_device,
            "pose_device": args.pose_device,
            "det_frequency": args.det_frequency,
            "output_dir": str(csv_dir),
        },
        "toolchain": {**_versions(), "git_commit": _git_commit()},
        "totals": {
            "videos_in_source_dir": len(sources),
            "csvs_produced": len(records),
            "sources_without_csv": missing,
            "total_rows": sum(r["rows"] for r in records),
            "total_decoded_frames": sum(r.get("decoded_frames", 0) or 0 for r in records),
            "schema_conformant_all": all(r["schema_conformant"] for r in records),
        },
        "videos": records,
    }


def render_markdown(manifest: dict) -> str:
    run, tot = manifest["run"], manifest["totals"]
    tc = manifest["toolchain"]
    cov = [r["coverage_pct"] for r in manifest["videos"] if r["coverage_pct"] is not None]

    lines = [
        "# Pose batch QA report",
        "",
        "Source stems are replaced by ordinals; `manifest.json` beside this file "
        "carries the stem mapping.",
        "",
        "## Run",
        "",
        f"- model: `{run['model']}` · tracking: `{run['tracking']}` · "
        f"single-subject: `{run['single_subject']}`",
        f"- backend: `{run['backend']}` · det-device: `{run['det_device']}` · "
        f"pose-device: `{run['pose_device']}` · det-frequency: `{run['det_frequency']}`",
        f"- git commit: `{tc['git_commit']}` · openvino `{tc['openvino']}` · "
        f"rtmlib `{tc['rtmlib']}` · python `{tc['python']}`",
        "",
        "## Totals",
        "",
        f"- CSVs produced: **{tot['csvs_produced']}** of {tot['videos_in_source_dir']} sources",
        f"- rows: **{tot['total_rows']}** over {tot['total_decoded_frames']} decoded frames",
        f"- schema conformant (all files): **{tot['schema_conformant_all']}**",
        f"- mean coverage: **{np.mean(cov):.1f}%**" if cov else "- mean coverage: n/a",
    ]
    if tot["sources_without_csv"]:
        lines.append(f"- **sources with no CSV: {len(tot['sources_without_csv'])}**")
    lines += [
        "",
        "## Per video",
        "",
        "| vid | decoded | rows | cov% | schema | gaps | dt_max | wrist% | Lhand% | Rhand% | vis | conf |",
        "| --- | ------- | ---- | ---- | ------ | ---- | ------ | ------ | ------ | ------ | --- | ---- |",
    ]

    def _f(value, spec=".1f"):
        return "n/a" if value is None else format(value, spec)

    for r in manifest["videos"]:
        lines.append(
            f"| {r['ordinal']:02d} | {r.get('decoded_frames', 0)} | {r['rows']} | "
            f"{_f(r['coverage_pct'])} | {'OK' if r['schema_conformant'] else 'MISMATCH'} | "
            f"{r['index_gaps']} | {_f(r['max_timestamp_gap_sec'], '.3f')} | "
            f"{_f(r['body_wrist_obs_pct'])} | {_f(r['left_hand_obs_pct'])} | "
            f"{_f(r['right_hand_obs_pct'])} | {_f(r['median_visibility'], '.3f')} | "
            f"{_f(r['median_confidence'], '.3f')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv_dir", help="Directory holding the produced per-video CSVs")
    p.add_argument("--videos-dir", default="videos", help="Directory holding the source videos")
    p.add_argument("--tracking", default="body", choices=["hands", "hands-arms", "body"])
    p.add_argument("--model", default="rtmw-l")
    p.add_argument("--backend", default="openvino")
    p.add_argument("--det-device", default="CPU")
    p.add_argument("--pose-device", default="NPU")
    p.add_argument("--det-frequency", type=int, default=7)
    p.add_argument("--single-subject", action="store_true")
    p.add_argument("--out-dir", default=None, help="Where to write artifacts (default: csv_dir)")
    args = p.parse_args(argv)

    manifest = build_manifest(args)
    out_dir = pathlib.Path(args.out_dir or args.csv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path = out_dir / "qa_report.md"
    report_path.write_text(render_markdown(manifest))

    print(render_markdown(manifest))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")
    return 0 if manifest["totals"]["schema_conformant_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

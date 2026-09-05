#!/usr/bin/env python3
"""Full corpus 2D run — resumable, with a total per-asset disposition manifest.

Runs every published session through the shipped 2D route (``pose_estimation.run``
then ``analysis/clinical_features.R``) and publishes a manifest giving each of
the registry's canonical assets exactly one disposition, so no asset can be
silently absent from a downstream denominator (M2.8.2 D06).

Resume is keyed on a per-event completion marker written after the event's
outputs are final, never on output presence: a killed run leaves a partial
landmark CSV that no row count can distinguish from a complete one, because the
true count is unknown until the source is fully decoded (D05).

Outputs are patient-adjacent, so the run tree lives outside git and the report
carries redaction-safe aggregates alone — ordinals, published stratum labels,
R reason codes and frozen disposition codes.  Subprocess output does carry
identifiers, so it goes to a log file under ``--out`` and is never echoed.

Rerun (source the accelerator env first, so pose inference reaches the NPU)::

    source /var/home/eturkes/.local/app/intel-accel/env.sh
    PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/corpus_run_2d.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pose_estimation.corpus_run import (
    ASSET_DISPOSITIONS,
    DISPOSITION_OK,
    MANIFEST_FILENAME,
    MARKER_COMPLETE,
    MARKER_FAILED,
    STAGE_CLINICAL,
    STAGE_RUN,
    ManifestError,
    asset_disposition,
    is_complete,
    read_marker,
    validate_manifest,
    write_manifest,
    write_marker,
)
from pose_estimation.export import COORD_NORMALIZATION
from pose_estimation.sessions import generation_digest, tree_digest, validate_generation

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = "scripts/corpus_run_2d.py"
# v2: the report gained `coord_normalization`, because export.py's normalisation
# moved and a 2D landmark CSV carries no identity tag able to say which one made it.
GENERATOR_VERSION = "v2"
CLINICAL_R = ROOT / "analysis" / "clinical_features.R"
# Named because the redaction allowlist has to hold every label the report can
# emit: `partial` is unreachable on a full corpus run and so shipped un-allowed,
# which aborted every partial and resumed run before it could write its report.
THROUGHPUT_PROVENANCE = "measured"
THROUGHPUT_FULL = "corpus"
THROUGHPUT_PARTIAL = "partial"
THROUGHPUT_LABELS = frozenset({THROUGHPUT_PROVENANCE, THROUGHPUT_FULL, THROUGHPUT_PARTIAL})


def _load_pilot() -> Any:
    """Borrow the pilot's readers and its redaction allowlist.

    The pilot is the standing instrument every corpus claim is already measured
    against, so a second copy of the table join or of the allowlist would be a
    surface that can drift away from it.  It stays a script, hence the explicit
    load rather than an import.
    """
    spec = importlib.util.spec_from_file_location(
        "pilot_corpus_run", ROOT / "scripts" / "pilot_corpus_run.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise RunError("the pilot instrument is unreadable")
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves `sys.modules[cls.__module__]` while the class body
    # executes, so registration has to precede the exec, not follow it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunError(RuntimeError):
    """A redacted corpus-run failure."""


pilot = _load_pilot()


def redaction_allowlist(args: Any, placed_assets: Any, codes: Any) -> frozenset[str]:
    """Every string this report may publish.

    Module level rather than inline in ``main`` so the set is reachable without a
    corpus run: the one label it ever missed (`partial`) is emitted only by a
    partial run, so no full-corpus invocation could expose the gap.
    """
    labels = pilot._values(placed_assets, "rotation_deg") | pilot._values(
        placed_assets, "pts_monotonic"
    )
    return frozenset(
        {
            GENERATOR,
            GENERATOR_VERSION,
            args.model,
            args.tracking,
            args.det_device,
            args.pose_device,
            MARKER_COMPLETE,
            MARKER_FAILED,
            COORD_NORMALIZATION,
        }
        | THROUGHPUT_LABELS
        | set(ASSET_DISPOSITIONS)
        | set(codes)
        | {asset.codec for asset in placed_assets}
        | {asset.device_config for asset in placed_assets}
        | {str(value) for value in labels}
    )


def _canonical_asset_ids(inventory: Path) -> list[str]:
    rows = pilot._read_rows(inventory / "assets.csv", "inventory")
    ids = sorted(row["asset_id"] for row in rows if row.get("disposition") == "canonical")
    if not ids:
        raise RunError("the registry publishes no canonical asset")
    return ids


def _run_stage(
    command: list[str], log: Path, env: dict[str, str] | None = None
) -> tuple[int, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return code, time.monotonic() - started


def _attempt_event(event_id: str, args: argparse.Namespace, logs: Path) -> dict[str, Any]:
    """Run one event end to end.  Partial output is destroyed, never credited."""
    event_out = args.out / event_id
    if event_out.exists():
        shutil.rmtree(event_out)
    command = [
        sys.executable,
        "-m",
        "pose_estimation.run",
        "--session-dir",
        str(args.sessions / event_id),
        "--output-dir",
        str(args.out),
        "--headless",
        "--model",
        args.model,
        "--tracking",
        args.tracking,
        "--det-device",
        args.det_device,
        "--pose-device",
        args.pose_device,
        "--det-frequency",
        str(args.det_frequency),
    ]
    if args.single_subject:
        command.append("--single-subject")
    code, run_seconds = _run_stage(command, logs / event_id / "run.log")
    if code != 0:
        write_marker(event_out, status=MARKER_FAILED, stage=STAGE_RUN, exit_code=code)
        return {
            "status": MARKER_FAILED,
            "stage": STAGE_RUN,
            "run_s": run_seconds,
            "clinical_s": 0.0,
        }

    environment = os.environ.copy()
    environment.update(LC_ALL="C", LANG="C", TZ="UTC")
    code, clinical_seconds = _run_stage(
        ["Rscript", str(CLINICAL_R), str(event_out)], logs / event_id / "clinical.log", environment
    )
    status = MARKER_COMPLETE if code == 0 else MARKER_FAILED
    write_marker(
        event_out,
        status=status,
        stage=STAGE_CLINICAL,
        exit_code=code,
        run_s=round(run_seconds, 3),
        clinical_s=round(clinical_seconds, 3),
    )
    return {
        "status": status,
        "stage": STAGE_CLINICAL,
        "run_s": run_seconds,
        "clinical_s": clinical_seconds,
    }


def _manifest_rows(canonical: list[str], placed: dict[str, Any], out: Path) -> list[dict[str, str]]:
    """Every canonical asset reaches exactly one disposition (D06)."""
    rows = []
    for asset_id in canonical:
        asset = placed.get(asset_id)
        if asset is None:
            rows.append(
                {
                    "asset_id": asset_id,
                    "event_id": "",
                    "camera_name": "",
                    "disposition": "not_placed",
                }
            )
            continue
        rows.append(
            {
                "asset_id": asset_id,
                "event_id": asset.event_id,
                "camera_name": asset.camera_name,
                "disposition": asset_disposition(out / asset.event_id, asset.camera_name),
            }
        )
    return rows


def _corpus_wall(event_ids: list[str], out: Path) -> dict[str, float]:
    """The corpus wall clock, summed from the per-event markers.

    Resume splits the corpus across invocations, so this process's accumulator
    measures its own share alone — pairing it with all-corpus frames publishes a
    throughput the pipeline never reached.  Each marker carries the event's own
    `run_s` / `clinical_s`, so the sum is the real total however many passes it
    took.  A marker written from the run-stage failure path carries neither, and
    `events_measured` is what says so.
    """
    total = {"run_s": 0.0, "clinical_s": 0.0, "events_measured": 0.0}
    for event_id in event_ids:
        marker = read_marker(out / event_id) or {}
        if "run_s" not in marker:
            continue
        total["run_s"] += float(marker["run_s"])
        total["clinical_s"] += float(marker.get("clinical_s", 0.0))
        total["events_measured"] += 1
    return total


def _diagnostic_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _artifacts(rows: list[dict[str, str]], placed: dict[str, Any], out: Path) -> dict[str, Any]:
    """P09: an `ok` asset owns one landmark CSV and one diagnostics row; no other does."""
    missing_csv = wrong_diag = trespass = 0
    counters: list[dict[str, int]] = []
    for row in rows:
        asset = placed.get(row["asset_id"])
        if asset is None:
            continue
        event_out = out / asset.event_id
        csv_path = event_out / f"{asset.camera_name}.csv"
        diagnostics = _diagnostic_rows(event_out / f"{asset.camera_name}_diag.csv")
        if row["disposition"] != DISPOSITION_OK:
            trespass += int(csv_path.is_file() or bool(diagnostics))
            continue
        missing_csv += int(not csv_path.is_file())
        if len(diagnostics) != 1:
            wrong_diag += 1
            continue
        try:
            entry: dict[str, float] = {
                key: int(diagnostics[0][key])
                for key in (
                    "n_frames_decoded",
                    "pts_accepted",
                    "index_fallback",
                    "monotonic_forced",
                )
            }
            # P12: the shipped schema stores both quantities the contract calls
            # derived, so the check is equality with the derivation, not absence.
            entry["stored_rate"] = float(diagnostics[0]["cfr_fallback_rate"] or 0.0)
            counters.append(entry)
        except (KeyError, TypeError, ValueError):
            wrong_diag += 1
    return {
        "missing_csv": missing_csv,
        "wrong_diag": wrong_diag,
        "trespass": trespass,
        "counters": counters,
    }


def _cfr(counters: list[dict[str, float]]) -> dict[str, Any]:
    """P12: the three counters classify every call, so they sum to the decoded frames.

    The pooled rate is frame-weighted, never the mean of the per-asset rates:
    two assets at 1000 and 10 frames differ by 17x between the two readings, so
    naming the weighting is what makes the published number reproducible.
    """
    frames = sum(entry["n_frames_decoded"] for entry in counters)
    fallback = sum(entry["index_fallback"] + entry["monotonic_forced"] for entry in counters)
    unclassified = sum(
        1
        for entry in counters
        if entry["pts_accepted"] + entry["index_fallback"] + entry["monotonic_forced"]
        != entry["n_frames_decoded"]
    )
    mismatch = sum(
        1
        for entry in counters
        if abs(
            entry["stored_rate"]
            - (
                (entry["index_fallback"] + entry["monotonic_forced"]) / entry["n_frames_decoded"]
                if entry["n_frames_decoded"]
                else 0.0
            )
        )
        > 1e-9
    )
    return {
        "assets_rate_mismatch": mismatch,
        "assets": len(counters),
        "frames_decoded": frames,
        "index_fallback": sum(entry["index_fallback"] for entry in counters),
        "monotonic_forced": sum(entry["monotonic_forced"] for entry in counters),
        "pooled_fallback_rate": round(fallback / frames, 8) if frames else 0.0,
        "assets_with_fallback": sum(
            1 for entry in counters if entry["index_fallback"] + entry["monotonic_forced"]
        ),
        "assets_unclassified": unclassified,
    }


def _partitions(event_ids: list[str], out: Path, header, codes) -> tuple[Any, int]:
    """P10 at corpus grain: every landmark CSV yields a group-disposition artifact."""
    total = pilot.Partition()
    failures = 0
    for event_id in event_ids:
        event_out = out / event_id
        if not is_complete(event_out):
            continue
        try:
            total.merge(pilot._partition(event_out, header, codes))
        except pilot.PilotError:
            failures += 1
    return total, failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 2D pipeline over the whole corpus.")
    parser.add_argument("--inventory", type=Path, default=ROOT / "inventory")
    parser.add_argument("--qualification", type=Path, default=ROOT / "qualification")
    parser.add_argument("--sessions", type=Path, default=ROOT / "sessions")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "corpus-2d")
    parser.add_argument("--report", type=Path, default=None, help="Report path (default: <out>).")
    parser.add_argument("--limit", type=int, default=0, help="Run the first N due events only.")
    parser.add_argument("--model", default="rtmw-l")
    # Default = the shipped corpus's own configuration, so a bare rerun reproduces it.
    parser.add_argument("--tracking", default="body")
    parser.add_argument("--det-device", default="CPU")
    parser.add_argument("--pose-device", default="NPU")
    parser.add_argument("--det-frequency", type=int, default=7)
    parser.add_argument("--single-subject", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-attempt events whose marker records a failure.",
    )
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="Publish the manifest and report over an existing --out tree without decoding.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    placed_assets = pilot._load_assets(args.inventory, args.qualification, args.sessions)
    placed = {asset.asset_id: asset for asset in placed_assets}
    canonical = _canonical_asset_ids(args.inventory)
    event_ids = sorted({asset.event_id for asset in placed_assets})
    header, codes = pilot._r_constants()

    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    digest_before = tree_digest(args.sessions)
    marker_before = generation_digest(args.sessions)
    # Before resolving any output: `_published_root` returns None for a tree
    # with no marker, which disarms the containment guard entirely, so the
    # marker's presence is what keeps the run off the published tree.
    validate_generation(args.sessions, inventory_dir=args.inventory)

    def due(event_id: str) -> bool:
        event_out = args.out / event_id
        marker = read_marker(event_out)
        return marker is None or (args.retry_failed and marker.get("status") != MARKER_COMPLETE)

    pending = [event_id for event_id in event_ids if due(event_id)]
    if args.limit:
        pending = pending[: args.limit]
    attempts: Counter[str] = Counter()
    run_seconds = clinical_seconds = 0.0
    if not args.analyse_only:
        print(
            f"{GENERATOR} {GENERATOR_VERSION}: {len(event_ids)} events, {len(pending)} due, "
            f"{len(canonical)} canonical assets",
            flush=True,
        )
        for index, event_id in enumerate(pending, 1):
            outcome = _attempt_event(event_id, args, logs)
            attempts[outcome["status"]] += 1
            run_seconds += outcome["run_s"]
            clinical_seconds += outcome["clinical_s"]
            print(
                f"[{index}/{len(pending)}] {outcome['status']} "
                f"run={outcome['run_s']:.1f}s clinical={outcome['clinical_s']:.1f}s "
                f"elapsed={run_seconds + clinical_seconds:.0f}s",
                flush=True,
            )

    validate_generation(args.sessions, inventory_dir=args.inventory)
    digest_after = tree_digest(args.sessions)
    marker_after = generation_digest(args.sessions)

    rows = _manifest_rows(canonical, placed, args.out)
    try:
        census = validate_manifest(rows, canonical)
        manifest_valid = True
    except ManifestError:
        census = dict.fromkeys(ASSET_DISPOSITIONS, 0)
        manifest_valid = False
    write_manifest(args.out / MANIFEST_FILENAME, rows)

    artifacts = _artifacts(rows, placed, args.out)
    cfr = _cfr(artifacts["counters"])
    partition, partition_failures = _partitions(event_ids, args.out, header, codes)
    complete = sum(1 for event_id in event_ids if is_complete(args.out / event_id))
    frames = cfr["frames_decoded"]
    corpus_frames = sum(asset.reported_frames for asset in placed_assets)
    wall = _corpus_wall(event_ids, args.out)

    verdicts = {
        "manifest_total": manifest_valid,
        "every_event_complete": complete == len(event_ids),
        "artifacts_owned": (
            artifacts["missing_csv"],
            artifacts["wrong_diag"],
            artifacts["trespass"],
        )
        == (0, 0, 0),
        "group_disposition_published": partition_failures == 0,
        "partition_total": partition.n_input == partition.n_windowed + partition.n_dropped,
        "partition_disjoint": partition.n_overlap == 0 and partition.n_unaccounted == 0,
        "group_qc_header_frozen": partition.header_frozen,
        "counters_classify_every_frame": cfr["assets_unclassified"] == 0,
        "stored_rate_equals_its_derivation": cfr["assets_rate_mismatch"] == 0,
        "generation_digest_unmoved": digest_before == digest_after,
        "generation_marker_unmoved": marker_before == marker_after,
    }
    payload: dict[str, Any] = {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "configuration": {
            "model": args.model,
            "tracking": args.tracking,
            "det_device": args.det_device,
            "pose_device": args.pose_device,
            "det_frequency": args.det_frequency,
            "single_subject": args.single_subject,
            # 2D landmark CSVs carry no identity tags, so two generations under
            # different normalisations are shaped identically and mean
            # different things.  The report is where they separate.
            "coord_normalization": COORD_NORMALIZATION,
        },
        "population": {
            "events": len(event_ids),
            "canonical_assets": len(canonical),
            "placed_assets": len(placed_assets),
            "reported_frames": corpus_frames,
        },
        "run": {
            "events_attempted": sum(attempts.values()),
            "events_complete": complete,
            "attempts_complete": attempts[MARKER_COMPLETE],
            "attempts_failed": attempts[MARKER_FAILED],
        },
        "manifest": {"rows": len(rows), "valid": manifest_valid, "census": census},
        "artifacts": {
            "missing_csv": artifacts["missing_csv"],
            "wrong_diag": artifacts["wrong_diag"],
            "trespass": artifacts["trespass"],
        },
        "cfr": cfr,
        "partition": {
            "groups_input": partition.n_input,
            "groups_windowed": partition.n_windowed,
            "groups_dropped": partition.n_dropped,
            "groups_in_both": partition.n_overlap,
            "groups_in_neither": partition.n_unaccounted,
            "window_rows": partition.window_rows,
            "events_without_disposition": partition_failures,
            "drop_reasons": dict(sorted(partition.reasons.items())),
        },
        # P14: measured over this run's own decoded frames, never projected.
        # The wall comes from the per-event markers rather than this invocation's
        # accumulator, because a resumed run spends part of the corpus cost in an
        # earlier process: dividing all-corpus frames by one invocation's seconds
        # reports a throughput the pipeline never reached.
        "throughput": {
            "provenance": THROUGHPUT_PROVENANCE,
            "sample": THROUGHPUT_FULL
            if wall["events_measured"] == len(event_ids)
            else THROUGHPUT_PARTIAL,
            "events_measured": int(wall["events_measured"]),
            "events_total": len(event_ids),
            "run_wall_s": round(wall["run_s"], 2),
            "clinical_wall_s": round(wall["clinical_s"], 2),
            "invocation_wall_s": round(run_seconds + clinical_seconds, 2),
            "frames_decoded": frames,
            "frames_per_s_incl_startup": (
                round(frames / wall["run_s"], 3) if wall["run_s"] else None
            ),
            "hours_total": round((wall["run_s"] + wall["clinical_s"]) / 3600, 3),
        },
        "verdicts": verdicts,
    }
    pilot._assert_redacted(payload, redaction_allowlist(args, placed_assets, codes))

    report = args.report or args.out / "run_report.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RunError, pilot.PilotError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

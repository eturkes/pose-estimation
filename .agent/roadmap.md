# Roadmap

Live long-horizon state only; completed trajectory belongs in git.

## M2 — real-world 3D clinical validation

**Status: PARKED** — precondition: user-granted clearance over a synchronized three-camera session with calibration (step 3 additionally needs an external clinical baseline). Dispatch skips M2 until that clearance lands and the marker is cleared; the unpark check below identifies candidate sessions but grants nothing.

**Goal:** validate the full real-data chain (calibration → 2D tracking → 3D fusion → clinical metrics), quantify reprojection/drop/confidence/timing/stability behavior, and replace provisional thresholds with evidence-backed values.

**Current evidence boundary:** calibration, fusion, `world3d.csv`, clinical analysis, QA grading, and injected failure modes pass synthetic tests. No claim of real-capture or clinical-metric accuracy is warranted yet.

**Unpark check:** run `UV_PROJECT_ENVIRONMENT=.venv uv run pose-estimation-run --list-sessions`. A redacted `3 cameras; calibration: present` result identifies a shape-compatible candidate only. Media decoding, synchronization checks, QA, and calibration-value inspection require explicit patient-data clearance.

**Sequence once unparked:**

1. First cleared capture dry run → execute QA + end-to-end validation; record failure modes; recalibrate capture procedure and provisional thresholds.
2. Multiple-session study → quantify reprojection, tracking gaps, throughput, temporal stability, and inter-trial repeatability.
3. Agreement study → use a known-geometry/goniometer/reference-system baseline when available; otherwise retain the validity gap and specify the cheapest sufficient baseline protocol.
4. Derive only de-identified or synthetic regression fixtures; lock observed failures with tests.

**Acceptance:** reproducible commands + reports trace every claim to cleared inputs; thresholds have explicit evidence; clinical-validity gaps remain visible; all repository validation gates pass.

## Produced datasets

- `output/rtmw-l_body_single/` — all 12 single-camera clips, RTMW-L / `--tracking body` / `--single-subject`, det-CPU + pose-NPU. 15 430 rows over 15 455 decoded frames, 99.7% mean coverage, 304-col schema conformant on every file, 100% body-wrist observation. `manifest.json` (per-video provenance + SHA-256) and `qa_report.md` sit beside the CSVs; regenerate both with `scripts/run_report.py`. Destined for `Projects/rehab/`, which has no pose-ingest contract yet — its schema is tabular ISNCSCI/SCIM, so the join surface still has to be designed.

## Backlog

Scope seed for the next milestone while M2 stays PARKED — no UNPLANNED milestone exists, so a bare `/session-roadmap` plans one from here.

- 3D-aware downstream aggregation in `analysis/`.
- Cross-camera identity matching for multi-person scenes; fusion currently assumes one subject.

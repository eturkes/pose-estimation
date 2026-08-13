# Project memory

Context retained only when source, tests, technical docs, roadmap, and git do not expose it cheaply.

## Data boundary

- Patient recordings + adjacent derivatives = sensitive. `videos/`, `output/`, real-data calibration files/directories, and logs stay outside agent context → inspect/decode/copy only under direct per-task clearance.
- Test for real multi-camera footage with the metadata-only probe `UV_PROJECT_ENVIRONMENT=.venv uv run pose-estimation-run --list-sessions`; consume its redacted summary alone, never the underlying session identifiers or media.
- Commit source + synthetic or de-identified fixtures only. `.gitignore` keeps raw media, derived outputs, rig geometry, credentials, and path-bearing logs out.
- A collaborator fork carrying unrelated history reconciles by content onto `main` with source-only attribution → importing its Git object history would retain patient data even after a file deletion.

## Environments

- The container + host share the checkout through different absolute paths, so uv environments are layer-specific. Container work uses `.venv`; host work uses `.venv-host`. Recreate the matching environment after a move; repair text shebang/activation/editable-path metadata only when offline, and regenerate binary/cache artifacts.
- GPU/NPU access depends on the inherited Intel driver/runtime environment (`LD_LIBRARY_PATH`, ICD, Level Zero, and any accelerator `PYTHONPATH`). Confirm with `openvino.Core().available_devices` in the correct uv environment; the pip dependency remains required for generic CPU-only checkouts.

## Device placement — detector vs pose

- Detector + pose model take separate devices (`--det-device` / `--pose-device` on `run`/`main`/`benchmark`/`validate`). No `--device` flag exists anywhere; a bare `--device` is an argparse error, not a silent default.
- **rtmlib YOLOX must not run on NPU.** In-graph NMS ⇒ dynamic `dets` shape; NPU demands static ⇒ fixed 100-row buffer whose unused rows are never written. Symptom: every frame reports exactly 100 detections, all rows sharing one score, values outside `[0,1]` (observed 1.128, 1.263). CPU on the same frames returns 1–3 detections, max 0.918. Compiles cleanly ⇒ `rtmlib_openvino.py`'s NPU→CPU fallback never fires; failure is numerical, silent, and reaches the CSV.
- Pose models are NPU-safe: RTMW-L NPU vs CPU = 0.505 px mean / 2.265 px p95 / 5.3 px p99 keypoint deviation, score MAE 0.00056. Per-call 7.17 ms NPU vs 134.26 ms CPU (~19×). Detector 109.82 ms NPU (garbage) vs 445.21 ms CPU. Projected 15 455-frame batch: all-CPU 51 min, det-CPU/pose-NPU 18 min.
- MediaPipe is unaffected — SSD anchors + NMS decode in Python (`detection.py`), graphs stay static-shaped ⇒ both roles default NPU. `models.DETECTOR_MODELS` selects which compile on `--det-device`.
- Reproduce either finding: `.scratch/det_npu_vs_cpu.py`, `.scratch/pose_npu_vs_cpu.py`, `.scratch/device_timing.py` (scalars only, no imagery/identifiers).

## GPU unavailable in-container (as of this writing)

`Core().available_devices` = `['CPU','NPU']`; GPU plugin reports no devices. Two stacked causes: (1) host `libze_intel_gpu.so.1` needs `GLIBCXX_3.4.35`, container `libstdc++` tops out at 3.4.33 — fixable by prepending host `libstdc++.so.6.0.36` to `LD_LIBRARY_PATH`; (2) past that, Intel compute-runtime aborts in `command_stream_receiver.cpp:1205`. Device nodes are accessible (`/dev/dri/render*`, `/dev/accel/accel0` all RW). Not chased further — CPU detector suffices.

## R analysis layer — non-obvious hazards

- `analysis/utils.R:59-87` `aggregate_per_video()` treats **every numeric non-metadata column as a feature**. Adding a count, coverage or QC column to an output that legacy consumers read makes it enter per-video means, z-scores, correlations and PCA silently. The R gate invokes no downstream consumer (`tests/test_r_pipeline.py` covers the producer, `features.R`, `arthrose_diag.R` only), so the full suite stays green while downstream tables change meaning.
- The 2D/3D partition is enforced by regex alone: consumers glob `_clinical\.csv$` / `_clinical_windows\.csv$`, which cannot match `_clinical_3d.csv`. Six consumers replicate that discovery, so widening one is a local edit with global consequences.
- Producer keys are `video`/`person_idx`/`window` only. No task, condition, trial or session identity exists anywhere in the schema — any "session" or "trial" grain has to come from metadata that does not yet exist, not from the CSVs.
- Gate constants are duplicated across languages: reprojection 20 px and triangulation angle 1° live in `src/pose_estimation/triangulation.py:423-424`, `src/pose_estimation/validation.py:77,86` and `analysis/clinical_features.R:49,54`. Changing one silently desynchronises the R adapter from fusion.

## Scratch validators pending port

- `.scratch/gap_bias_probe.R` — backs M3's gap-corruption claim. Sources `analysis/clinical_features.R` (helpers are defined before the CLI block, so `try(source(...))` yields them), builds an analytic minimum-jerk reach, and mirrors the production window block at `analysis/clinical_features.R:681-689` verbatim. Regenerate with `Rscript .scratch/gap_bias_probe.R`. Ports into M3.1's regression suite as the sentinel case; delete the scratch copy once the committed test reproduces the numbers.
- `zoo` is gone (M3.1). `stats::filter(x, rep(1/5, 5), sides = 2)` replaced `zoo::rollmean(x, 5, fill = NA, align = "center")`: NA propagation identical at centre, both edges and interior/leading/trailing/scattered holes; values differ ~1-2 ULP (2.8e-14 absolute on ~100-magnitude input — the old "1.11e-16" note was relative, not absolute), and no golden pins that column. `dplyr` masks `stats::filter`, so the call must stay namespace-qualified.

## Worktree gate recipe (`.scratch/worktrees/<name>`)

Every teammate worktree runs the full Python gate concurrently off the one primary environment, read-only:

```sh
export UV_PROJECT_ENVIRONMENT=<primary-tree>/.venv PYTHONPATH="$PWD/src"
uv run --no-sync ruff check && uv run --no-sync ruff format --check \
  && uv run --no-sync ty check && uv run --no-sync pytest
```

- `PYTHONPATH=<worktree>/src` is mandatory: the hatchling editable install resolves `pose_estimation` to the **primary** tree's `src/`, so a worktree gate without it silently tests the primary code and reports green for untested changes.
- `--no-sync` keeps the shared environment unmutated, which is what makes concurrent worktree gating safe.
- Tool caches (`.ruff_cache`, `.pytest_cache`, `.ty_cache`, `.coverage`) are cwd-relative → already private per worktree; no extra state paths needed.
- `renv/library/` is gitignored, so a fresh worktree has no R library and every R case SKIPs. Symlink it read-only and the worktree gate becomes fully equivalent: `ln -sfn <primary>/renv/library <worktree>/renv/library` → 469 passed/0 skipped, `tests/test_r_pipeline.py` 25 passed/0 skipped, same as primary. Concurrent worktrees share it safely because R only reads packages; never `renv::install`/`renv::snapshot` through the link.

## Session launch cost

`headroom wrap claude` blocks on `uvx … serena project index` (`cli/wrap.py:_index_serena_project`, 300 s cap) before Claude Code starts → anything that stalls Serena's indexer is felt as launch latency, and only in the repo that holds it.

- Budget: full cold index = ~12 s (109 files, 7 language servers), warm ~6 s. A launch stalling far past that means one file is eating an LS request timeout (`serena_config.yml` `tool_timeout` 240 − 5 = 235 s each) → `.serena/logs/indexing.txt` names the file.
- `**/*.Rmd` is excluded in `.serena/project.yml` for exactly that reason (R LS never answers `documentSymbol` for R Markdown; `.R` files are unaffected at ~4 files/s). Reach `analysis/analysis_summary.Rmd` by `Read`/`rg`; Serena's symbol + search tools do not see it.
- Serena's own session start is repo-independent (~3.5 s, dominated by the bash LS) and asynchronous — it never blocks the MCP handshake.

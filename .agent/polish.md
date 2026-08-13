# Polish register

Deferred off-spine improvements + data-tier remediation. Rows are born at deferral time in whatever session finds them (acceptance check written there, while the evidence is fresh); `/session-polish` is the sole consumer — stateless, any order, no milestone coupling.

Row schema — `pri` 1 (highest) … 3 · `size` S ≤15% window | M ≤35% | L = session · `where` = `file:line` or artifact · `why` = evidence pointer (SHA, run output, finding id) · `acceptance` = the check that must run green under MAIN's own rerun.

Lifecycle — done ⇒ prune the row in the same `<scope> (polish): …` commit · dead evidence pointer or acceptance check ⇒ append `stale(<why>)` in place, next `/session-roadmap` session re-rules it · item implying spine work ⇒ move it to `spine?` below + report to the user.

Items run inside the artifact's existing assurance tier; tier raises, new units + scope-source changes belong to `/session-roadmap`.

## Items

| pri | size | where | item | why | acceptance |
| --- | ---- | ----- | ---- | --- | ---------- |
| 2 | S | `src/pose_estimation/rtmlib_openvino.py` | Detector-output sanity guard: raise, or warn once and fall back to CPU, when a detector returns scores outside `[0,1]` or saturates at its padded row count. Today only the `--det-device` default keeps the NPU-YOLOX corruption out of the data; `--det-device NPU`, or a device whose dynamic-shape support regresses, still yields silent garbage. | f3d18ad; `.scratch/det_npu_vs_cpu.py` → NPU 100 rows sharing one score, max 1.263 vs CPU 1–3 rows, max 0.918 | `uv run python .scratch/det_npu_vs_cpu.py 7 6` errors or warns on the NPU rows and stays silent for CPU; `pytest` green |
| 1 | M | `analysis/{compare_clinical,longitudinal,clinical_correlation,clinical_dimreduce,temporal_clinical,explore_clinical}.R` | Route the six legacy clinical consumers through M3.4's central mode-aware reader; retain tags, mode-specific output names, unit-bearing labels. | Six duplicated suffix/read/bind paths are safe only implicitly — one widened glob silently pools normalised-2D with metric-3D. M3 ships the aggregate without touching them. | Synthetic mixed 2D+3D directory: each CLI emits separately named mode outputs with retained units, or rejects mixed input; 2D goldens unchanged; `pytest tests/test_r_pipeline.py` + full suite green in primary tree |
| 1 | S | `analysis/make_templates.R:32-49`, `analysis/validate_metadata.R:32-53` | Generalise metadata discovery via the central tagged reader, preserving exact 2D behaviour. | Both silently skip `_3d`; M3 ships its own aggregate path, so widening these generic CLIs is compatibility work. | Synthetic 2D-only, 3D-only, mixed directories yield explicit mode choice, no duplicate video keys, unchanged 2D templates/validation; R gate green |
| 2 | S | `analysis/features.R:410-426` | Exclude generated clinical/movement outputs from broad raw-landmark discovery. | Current `\\.csv$` glob can rescan generated clinical outputs; `_x_m` world3d columns yield "No coordinate columns" rather than a clean skip. | Directory holding raw 2D + world3d + generated clinical outputs processes only intended raw inputs; full suite green |
| 2 | M | `analysis/arthrose_diag.R:61-140` | Add a trusted world3d hand-screen path (3D joint angles, metric aperture label, producer-style gating) and evidence or downgrade the 40°/60°·s⁻¹ threshold wording. | Screen expects raw 2D-shaped hand columns and computes flexion in x/y, yet calls its threshold output a diagnosis; thresholds are code facts with no cited clinical source. | Synthetic 2D legacy unchanged; analytic 3D hand fixture uses x/y/z + metre labels + fail-closed diagnostics; docs claim matches evidence; full suite green |
| 2 | M | `analysis/clinical_features.R:492-508` + M3.4 metric registry | Optional arm-length / hand-span / object-width normalised metrics carrying denominator provenance, raw values retained. | Raw and normalised answer different estimands; the current shoulder-width ratio is not a standard anthropometric normaliser, and no denominator metadata exists. | Side-specific denominator fixtures emit separate `metric_id`/`unit`/`normalizer_id`; missing denominator yields no normalised estimate; raw outputs byte-identical; R gate green |
| 3 | S | six clinical consumer bootstraps + `analysis/utils.R:29-53` | Deduplicate the repeated 13-line script-dir/source bootstrap when those files are next touched. | Maintenance cost, not correctness. | One helper path works under `Rscript`, `source()`, and project-root REPL; consumer tests unchanged |
| 3 | S | `analysis/utils.R:59-87` | Make the generic summariser all-NA safe and warning-free. | `mean/min/max(..., na.rm = TRUE)` over an all-NA column emits `NaN`/`-Inf` plus warnings; warnings are test errors. | All-NA and mixed-finite fixtures return typed `NA_real_` without warnings; finite-case outputs unchanged; R gate green |
| 3 | M | container env + `~/agents/docs/openvino.md` | Enable GPU in-container: prepend host `libstdc++.so.6.0.36` to the accel farm (clears the `GLIBCXX_3.4.35` load failure), then resolve the Intel compute-runtime abort at `command_stream_receiver.cpp:1205`. Gives the detector an accelerator and frees the contended CPU. | this session: `Core().available_devices` = `['CPU','NPU']`; `/dev/dri/render*` + `/dev/accel/accel0` RW; abort reproduced with the libstdc++ override in place | `source intel-accel/env.sh && python -c "import openvino;print(openvino.Core().available_devices)"` lists `GPU`, and `intel-accel/selftest.py` reports `[GPU] OK correct=True` |

## spine?

Findings a polish session judged spine work — ruled by `/session-roadmap`, not executed here. Row: `spine? <finding> | why: <evidence>`.

_empty_

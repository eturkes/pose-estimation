# Polish register

Deferred off-spine improvements + data-tier remediation. Rows are born at deferral time in whatever session finds them (acceptance check written there, while the evidence is fresh); `/session-polish` is the sole consumer — stateless, any order, no milestone coupling.

Row schema — `pri` 1 (highest) … 3 · `size` S ≤15% window | M ≤35% | L = session · `where` = `file:line` or artifact · `why` = evidence pointer (SHA, run output, finding id) · `acceptance` = the check that must run green under MAIN's own rerun.

Lifecycle — done ⇒ prune the row in the same `<scope> (polish): …` commit · dead evidence pointer or acceptance check ⇒ append `stale(<why>)` in place, next `/session-roadmap` session re-rules it · item implying spine work ⇒ move it to `spine?` below + report to the user.

Items run inside the artifact's existing assurance tier; tier raises, new units + scope-source changes belong to `/session-roadmap`.

## Items

| pri | size | where | item | why | acceptance |
| --- | ---- | ----- | ---- | --- | ---------- |
| 2 | S | `src/pose_estimation/rtmlib_openvino.py` | Detector-output sanity guard: raise, or warn once and fall back to CPU, when a detector returns scores outside `[0,1]` or saturates at its padded row count. Today only the `--det-device` default keeps the NPU-YOLOX corruption out of the data; `--det-device NPU`, or a device whose dynamic-shape support regresses, still yields silent garbage. | f3d18ad; `.scratch/det_npu_vs_cpu.py` → NPU 100 rows sharing one score, max 1.263 vs CPU 1–3 rows, max 0.918 | `uv run python .scratch/det_npu_vs_cpu.py 7 6` errors or warns on the NPU rows and stays silent for CPU; `pytest` green |
| 3 | M | container env + `~/agents/docs/openvino.md` | Enable GPU in-container: prepend host `libstdc++.so.6.0.36` to the accel farm (clears the `GLIBCXX_3.4.35` load failure), then resolve the Intel compute-runtime abort at `command_stream_receiver.cpp:1205`. Gives the detector an accelerator and frees the contended CPU. | this session: `Core().available_devices` = `['CPU','NPU']`; `/dev/dri/render*` + `/dev/accel/accel0` RW; abort reproduced with the libstdc++ override in place | `source intel-accel/env.sh && python -c "import openvino;print(openvino.Core().available_devices)"` lists `GPU`, and `intel-accel/selftest.py` reports `[GPU] OK correct=True` |

## spine?

Findings a polish session judged spine work — ruled by `/session-roadmap`, not executed here. Row: `spine? <finding> | why: <evidence>`.

_empty_

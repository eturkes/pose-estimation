# Pose estimation

## Data boundary

- Patient recordings + adjacent derivatives are sensitive. Treat `videos/`, `output/`, real-data calibration files/directories, and logs as outside agent context: inspect/decode/copy them only with direct per-task user clearance.
- To test whether real multi-camera footage exists, use the metadata-only probe `UV_PROJECT_ENVIRONMENT=.venv uv run pose-estimation-run --list-sessions`; consume its redacted summary, never the underlying session identifiers or media.
- Commit only source + synthetic or de-identified fixtures. Keep raw media, derived outputs, rig geometry, credentials, and path-bearing logs excluded through `.gitignore`.

## Stack + environment

- Python package = `src/pose_estimation/`; management/build = `uv` + `pyproject.toml` + `uv.lock`; compatibility floor = Python 3.10. R clinical analysis = `analysis/` + `renv.lock`.
- Container checkout path (`/run/host/...`) uses `.venv`; host-OS checkout uses `.venv-host`. `.envrc` selects by path in hooked interactive shells; non-interactive commands must export the matching `UV_PROJECT_ENVIRONMENT` before `uv` runs.
- Preserve the single-OpenCV-wheel policy in `[tool.uv].override-dependencies`: rtmlib must resolve through `opencv-python-headless`, never a second `cv2` wheel.
- Runtime paths: MediaPipe→OpenVINO in `main.py`; rtmlib→OpenVINO/ONNX Runtime in `run.py`; CPU is the portable validation target, NPU the runtime default. NPU/GPU availability depends on the machine driver environment, not import success alone.

## Navigation + maintenance

- Human entry point = `README.md`; task-specific internals = `docs/technical/`; capture procedure = `docs/capture_protocol.md`. Source, manifests, and tests outrank prose when they disagree.
- Long-horizon live state belongs in `.agent/roadmap.md`; `.agent/memory.md` holds only context not cheaply recoverable from source/docs/tests/git. Consult either only when the task intersects it; prune resolved or duplicated material immediately.
- A module, CLI flag, output schema, public export, or test-layout change must update its affected technical reference. Keep `src/pose_estimation/__init__.py` + `tests/test_public_api.py` synchronized.
- Session/calibration manifest paths and labels are hostile input. Preserve containment checks, safe path-component validation, and traversal regression coverage.

## Validation

- Python gate: `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, then `uv run pytest`. Pytest warnings are errors.
- Changed `analysis/*.R` scripts must exit 0 under `Rscript` with the project renv active. After an R upgrade, update + snapshot `renv.lock` before validation.
- Smoke-test each changed console entry point (`pose-estimation`, `pose-estimation-run`, `pose-estimation-benchmark`, `pose-estimation-postprocess`, `pose-estimation-calibrate`, `pose-estimation-validate`) on a non-sensitive, non-interactive path appropriate to the change.

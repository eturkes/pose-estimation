# Environment

## Host

- Linux (currently openSUSE / kernel 7.0.9-1-default per session host).
- GNOME Wayland — the reason `pygame-ce` is used for display (Qt-bundled OpenCV does not render on Wayland).
- Python 3.10+ required (`>=3.10` in `pyproject.toml`); host Python is 3.13 (see `.python-version`).

## Python toolchain

- Manager: `uv` (`pyproject.toml` + `uv.lock`, both committed).
- Build backend: `hatchling`; wheels package `src/pose_estimation`.
- Interpreter pin: `.python-version` (read by `uv`).
- Virtualenv: `.venv/` (created on the host; symlinks are absolute and not portable across containers).
- Install / sync: `uv sync` (the assistant must run this on the host; it does not work from inside a container).

### Adding a Python dependency

- Runtime: edit `[project.dependencies]` in `pyproject.toml` and run `uv add <pkg>` (atomic with `uv.lock`).
- Dev/test/lint/types: `uv add --group {test|lint|types|dev} <pkg>`.
- `uv.lock` is committed for reproducible installs.

## R toolchain

- Manager: `renv` (lockfile: `renv.lock`).
- Install all: `renv::restore()` inside an R session at the project root.
- Add a package: `renv::install("<pkg>")` then `renv::snapshot()`.
- Use renv exclusively; the global library should not satisfy project deps.

## Devices / inference

- OpenVINO backends: NPU (default), CPU, GPU. Select with `--device {NPU|CPU|GPU}` on `main.py` / `run.py`.
- rtmlib supports both `onnxruntime` and `openvino` backends (`--backend` on `run.py`).
- `scripts/npu_compat.py` — verify NPU compatibility before adding a model to the registry.

## Data directories

- `videos/` — input videos (git-ignored; usually a symlink to NAS).
- `output/` — pipeline CSV/metrics outputs (git-ignored).
- `model/` — downloaded TFLite/ONNX/OpenVINO IR cache.
- All three are kept out of git to prevent patient data from being committed.

## Container caveat

You may not be able to run `uv sync` or activate `.venv` from inside a container — `.venv` contains absolute symlinks to the host Python binary. Run sync on the host; in-container work assumes `.venv` already exists and is healthy.

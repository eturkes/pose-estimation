# Environment

## Host / container

- Two layers: an **openSUSE-based host** and a **Debian (trixie) Distrobox container** where agent sessions run. Home is shared across both, which is why `.venv` (host-created) and in-container R coexist — see the Container caveat below.
- GNOME Wayland — the reason `pygame-ce` is used for display (Qt-bundled OpenCV does not render on Wayland).
- Python 3.10+ required; the exact interpreter is pinned in `.python-version` and the floor declared in `pyproject.toml`.
- In-container agent tooling: language servers (LSP) and persistent REPLs via `bgcmd` (`~/.local/bin/bgcmd`) — prefer these for interactive inspection over one-shot scripts.

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

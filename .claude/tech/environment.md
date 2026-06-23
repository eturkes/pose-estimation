# Environment

## Host / container

- Two layers: an **openSUSE-based host** and a **Debian (trixie) Distrobox container** where agent sessions run. The container has its **own home** (`/var/home/eturkes/debian`); the host filesystem is mounted at `/run/host/`, so the project root is `/run/host/home/eturkes/Projects/pose-estimation`. All tooling (uv, `.venv`, R/renv) runs in-container — see the Container caveat below.
- GNOME Wayland — the reason `pygame-ce` is used for display (Qt-bundled OpenCV does not render on Wayland).
- Python 3.10+ required; the exact interpreter is pinned in `.python-version` and the floor declared in `pyproject.toml`.
- In-container agent tooling: language servers (LSP) and persistent REPLs via `bgcmd` (`~/.local/bin/bgcmd`) — prefer these for interactive inspection over one-shot scripts.

## Python toolchain

- Manager: `uv` (`pyproject.toml` + `uv.lock`, both committed).
- Build backend: `hatchling`; wheels package `src/pose_estimation`.
- Interpreter pin: `.python-version` (read by `uv`).
- Virtualenv: `.venv/` (container-native: absolute paths in `/run/host/...` form). `bin/*` shebangs, `activate*` (`VIRTUAL_ENV`), and the editable `*.pth` hardcode the project's **absolute path**, so a project move or container path change needs repair (see Relocation below).
- Install / sync: `uv sync` in-container (verified 2026-06-08; uv lives at `~/.local/bin/uv`). uv warns that cache (container fs) and `.venv` (`/run/host`) are on different filesystems and falls back to copying — harmless; `export UV_LINK_MODE=copy` silences it.
- **Single cv2 wheel policy**: `[tool.uv] override-dependencies` in `pyproject.toml` excludes rtmlib's `opencv-python` + `opencv-contrib-python` (always-false markers). All cv2 wheels unpack the same `cv2/` tree, so coinstallation file-stomps nondeterministically; we ship cv2 exactly once via `opencv-python-headless`. rtmlib uses no contrib-only modules; `cv2.aruco` is in main OpenCV ≥ 4.7.

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
- Both entry points run OpenVINO: `main.py` converts the MediaPipe TFLite models to IR and `core.compile_model(device=…)` (`models.py`); `run.py` defaults to `--backend openvino --device NPU` for rtmlib (`onnxruntime` is the alternative `--backend`).
- `scripts/npu_compat.py` — verify NPU compatibility before adding a model to the registry.
- Confirm a device is actually visible to OpenVINO: `uv run python -c "import openvino as ov; print(ov.Core().available_devices)"`. `NPU` appears only when the host's Intel NPU userspace stack is installed (kernel `intel_vpu` → `/dev/accel/accel0`, plus the level-zero NPU driver/compiler/firmware). The `openvino` wheel bundles the plugin but relies on that system stack; absent it, only `CPU`/`GPU` are listed and an explicit `--device NPU` fails.

## Data directories

- `videos/` — input videos (git-ignored; usually a symlink to NAS).
- `output/` — pipeline CSV/metrics outputs (git-ignored).
- `model/` — downloaded TFLite/ONNX/OpenVINO IR cache.
- All three are kept out of git to prevent patient data from being committed.

## Container caveat

The venv's absolute paths are in `/run/host/...` form, which exists only inside the container — host-side use of `.venv` would need a host-side `uv sync` first (and would then break container use; the container is canonical since agents are the sole users). `.venv/bin/python` targets system `/usr/bin/python3.13` (`pyvenv.cfg` `home = /usr/bin`), which resolves in both — the interpreter symlink survives moves/relayouts, but the absolute paths in Relocation below do not.

### Host-side runs (separate venv, e.g. for the NPU)

The host sees the project at `/home/eturkes/Projects/pose-estimation` (no `/run/host` prefix), so the container `.venv` is unusable there; the host uses its own git-ignored `.venv-host/`. Both layers share one working tree, so the venv is selected **per layer** via `UV_PROJECT_ENVIRONMENT`. The committed **`.envrc`** (direnv) sets it automatically — `/run/.containerenv` present (Distrobox) → `.venv`, absent (host) → `.venv-host` — the moment you `cd` in, so a bare host `uv` can no longer silently rebuild `.venv` with host `/home/...` paths (the prior footgun that broke in-container `import pose_estimation`). Host setup once: `brew install direnv`, hook bash+zsh (`eval "$(direnv hook bash)"` / `direnv hook zsh`), then `direnv allow`. In-container direnv is optional — `.venv` is uv's default, so agent (non-interactive) shells need nothing. The explicit form below is the equivalent, and the fallback wherever direnv is inactive (non-interactive shells, or before `direnv allow`):

```bash
cd /home/eturkes/Projects/pose-estimation
export UV_PROJECT_ENVIRONMENT=.venv-host   # keeps the container .venv intact; .venv-host/ is git-ignored
uv sync                                     # uv fetches Python 3.13 itself if the host lacks it
uv run python -m pose_estimation.run --source <video>   # live pygame window; rtmw-l + openvino + NPU are defaults
```

Omitting `--headless` gives the live overlay window (pygame-ce renders in the host's GNOME Wayland session). Models download to `model/` on first run. Requires `uv` on the host (installed via Homebrew) and the system NPU userspace stack (see Devices / inference for the `available_devices` check).

## Relocation (moved project root)

Moving the project breaks the venv's hardcoded absolute paths and leaves stale paths in regenerable caches. Repair:

- Canonical: re-run `uv sync` in-container.
- Offline / in-container: rewrite old→new path in `.venv` **text** files only — `bin/*` shebangs, `activate*` (`VIRTUAL_ENV`), `site-packages/_editable_impl_*.pth` (this one breaks `import pose_estimation`), `dist-info/direct_url.json`. Always skip `*.pyc`/`*.so`: old vs new paths differ in byte length, so an in-place edit corrupts the binary — and they carry the path only as cosmetic build-dir / `co_filename` strings.
- Clear regenerable caches embedding the old path: project `__pycache__`, `.ruff_cache`.
- Survive a move untouched: `.venv/bin/python` (→ system), renv library symlinks (0 dangling), renv `.so` (cosmetic). Verify: `import pose_estimation`, a console script, `pytest`, `Rscript -e 'renv::project()'`.
- Enumerate matches with `find -exec grep` or Python, not bare `grep -r`: the shell's `grep` is a profile **function** that prunes dot-dirs, so `grep -r .venv` silently reports nothing.

# Environment

## Host / container

- Two layers: an **openSUSE-based host** and a **Debian (trixie) Distrobox container** where agent sessions run. The container has its **own home** (`/var/home/eturkes/debian`); the host filesystem is mounted at `/run/host/`, so the project root is `/run/host/home/eturkes/Projects/pose-estimation`. Agent tooling (uv, `.venv`, R/renv) and OpenVINO inference run **in-container**, with CPU/GPU/NPU device access (see Devices / inference); a separate `.venv-host` covers the narrower case of launching from the host OS directly (Host-side runs below).
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
- **Primary path = in-container.** With the machine-local accel env sourced, the project `.venv` sees all three devices — `Core().available_devices == ['CPU', 'GPU', 'NPU']`, and a generic OpenVINO compile+infer selftest passes on each (re-verified 2026-06-25, exit 0) — so NPU/GPU runs no longer need a host-side launch. The Intel iGPU+NPU stack is enabled by a machine-local setup (drivers + an accel-enabled OpenVINO runtime) whose specifics + the `source …/env.sh` step live in the auto-injected, git-ignored `CLAUDE.local.md`, not here. Source that accel env before launching python: it sets the GPU/NPU driver paths (`LD_LIBRARY_PATH` + OpenCL-ICD + Level-Zero) that gate device access.
- **Device access is gated by that sourced env, not by which `openvino` package imports.** With it sourced, both the accel runtime (on `PYTHONPATH`) and the `.venv` pip wheel enumerate all three (verified). `PYTHONPATH` precedes `.venv` site-packages in `sys.path`, so `import openvino` resolves to the accel build (newer) and the pip wheel is the fallback. **Keep the pip `openvino` dependency** — `CLAUDE.local.md`'s "never pip install openvino" guards the machine-local runtime's `sys.path` precedence, which the committed dep does not disturb (`PYTHONPATH` stays in front); a generic checkout needs the dep.
- Per-model device coverage is not blanket-guaranteed (NPU op support varies): check a model with `scripts/npu_compat.py` (compiles each rtmlib model on a device — run before adding one to the registry) and rely on `rtmlib_openvino.py`'s runtime NPU→CPU fallback — so `--device NPU` (the `run.py` default) targets the NPU but may transparently land on CPU.
- Confirm devices (pin the venv — a bare `uv run` honors `UV_PROJECT_ENVIRONMENT`, which is `.venv-host` in some non-interactive shells here): source the accel env, then `UV_PROJECT_ENVIRONMENT=.venv uv run python -c "import openvino as ov; print(ov.Core().available_devices)"` (or call `.venv/bin/python` directly). Keep `PYTHONPATH` intact — `python -E`/`-I` or PYTHONPATH-stripping `uv run` modes drop to the pip wheel. On a generic checkout (no accel env) the list reduces to whatever the system's Intel GPU (OpenCL/IGC) and NPU (`intel_vpu` → `/dev/accel/accel0`, level-zero) userspace exposes to stock OpenVINO — often CPU only until those are installed.

## Data directories

- `videos/` — input videos (git-ignored; usually a symlink to NAS).
- `output/` — pipeline CSV/metrics outputs (git-ignored).
- `model/` — downloaded TFLite/ONNX/OpenVINO IR cache.
- All three are kept out of git to prevent patient data from being committed.

## Container caveat

The venv's absolute paths are in `/run/host/...` form, which exists only inside the container — host-side use of `.venv` would need a host-side `uv sync` first (and would then break container use; the container is canonical since agents are the sole users). `.venv/bin/python` targets system `/usr/bin/python3.13` (`pyvenv.cfg` `home = /usr/bin`), which resolves in both — the interpreter symlink survives moves/relayouts, but the absolute paths in Relocation below do not.

### Host-side runs (separate venv — launching from the host OS)

No longer required for NPU/GPU — those run in-container (see Devices / inference); use this only to launch from the host OS itself (e.g. a host GNOME session for the live pygame window). The host sees the project at `/home/eturkes/Projects/pose-estimation` (no `/run/host` prefix), so the container `.venv` is unusable there; the host uses its own git-ignored `.venv-host/`, auto-selected by the committed **`.envrc`** in an allowed interactive shell (per the global host/container rule; mechanism in the `.envrc` header). One-time host setup: `brew install direnv`, hook bash+zsh (`eval "$(direnv hook bash)"` / `direnv hook zsh`), `direnv allow`. Other shells use the explicit form below — and `.envrc` pins the var while loaded, so a one-off custom env prefixes `UV_PROJECT_ENVIRONMENT=... uv ...`:

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

# Engineering conventions

## Git

- Commit messages: [Scoped Commits](https://scopedcommits.com/) — `<scope>: <description>`, scope first (the subsystem/area touched, e.g. `tracking`, `calibration`, `multicam`, or a cross-cutting label such as `Tooling`, `Maintenance`, `Refactor`, `Docs`). For multi-area commits, comma-list the scopes, generalize to one, or use `treewide`. Subject + body take the `CLAUDE.md` `Authoring` standard: subject ≤50 chars, imperative; body wrap ≤72 chars; `→` for cause→fix; measurements + SHAs kept as payload while the narration around them goes.
- Before committing, always check whether `README.md`, `.gitignore`, `pyproject.toml`, or other housekeeping files need a matching update.

## Quality gate

Ordered; every stage passes before a commit.

```bash
uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
```

- `pytest` is strict — warnings are errors. `ruff format --check` is the gate form; `ruff format` is the autofix form.
- Changed `analysis/*.R` must exit 0 under `Rscript` with the project renv active. After an R upgrade, update + snapshot `renv.lock` first.
- Smoke-test each changed console entry point on a non-sensitive, non-interactive path.
- Non-interactive shells (scripts, agents, fresh shells) export the layer's `UV_PROJECT_ENVIRONMENT` before `uv` runs — `.envrc` covers hooked interactive shells only. See `environment.md`.

## Maintenance

- A module, CLI flag, output schema, public export, or test-layout change updates its affected `docs/technical/` reference.
- Keep `src/pose_estimation/__init__.py` + `tests/test_public_api.py` synchronized.
- Session/calibration manifest paths + labels are hostile input → preserve containment checks, safe path-component validation, and traversal regression coverage.
- Source, manifests, and tests outrank prose when they disagree; fix the prose.

## Python style — `ruff`

Config: `[tool.ruff]` in `pyproject.toml`. Line length 100, target py310.

Enabled rule sets:
`E`, `W`, `F`, `I`, `B`, `UP`, `N`, `SIM`, `C4`, `PIE`, `PT`, `RET`, `PTH`, `RUF`, `NPY`, `PD`, `PERF`.

Project-wide ignores: `E501` (formatter handles wrapping), `SIM108` (ternary not always clearer), `N803`/`N806` (scientific naming — `L/R` side, `M` matrix).

Per-file ignores:
- `tests/**` → `N802`, `N803`, `N806`.
- `scripts/**` → `T201` (print allowed).
- `scripts/benchmarks/**` → `T201`, `PERF401` (explicit loops in bench builders), `RUF003` (allow `µ` etc.).

Run:
```bash
uv run ruff check --fix
uv run ruff format
```

`docstring-code-format = true` — code blocks inside docstrings are reformatted.

## Type checking — `ty`

Astral's type checker (alpha; pre-1.0). Config: `[tool.ty.*]`.

```bash
uv run ty check
```

`tool.ty.environment.root = ["src", "tests"]`, Python 3.10 target. `tool.ty.src.include = ["src", "tests"]`.

## Testing — `pytest`

```bash
uv run pytest
uv run pytest --cov=pose_estimation        # coverage
```

Strict config: `-ra --strict-config --strict-markers --import-mode=importlib`. Warnings are errors via `filterwarnings`. See `tests.md` for the test inventory.

## Code style notes

- Public API: only what's re-exported from `src/pose_estimation/__init__.py`. Internal helpers may move freely.
- TypedDicts in `_types.py` document dict-passed pipeline state. Treat them as the contract.
- Prefer editing existing modules to introducing new ones; the surface is small on purpose.
- Comments: keep sparse — explain WHY when non-obvious; don't restate WHAT the code does.

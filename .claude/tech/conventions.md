# Conventions

## Git

- Commit messages: [Scoped Commits](https://scopedcommits.com/) — `<scope>: <description>`, scope first (the subsystem/area touched, e.g. `tracking`, `calibration`, `multicam`, or a cross-cutting label such as `Tooling`, `Maintenance`, `Refactor`, `Docs`). For multi-area commits, comma-list the scopes, generalize to one, or use `treewide`. Optimize for LLM parsing: subject ≤50 chars, imperative description, body wrap ≤72 chars.
- Before committing, always check whether `README.md`, `.gitignore`, `pyproject.toml`, or other housekeeping files need a matching update.

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

`tool.ty.environment.root = ["src"]`, Python 3.10 target. `tool.ty.src.include = ["src", "tests"]`.

## Testing — `pytest`

```bash
uv run pytest
uv run pytest --cov=pose_estimation        # coverage
```

Strict config: `-ra --strict-config --strict-markers --import-mode=importlib`. Warnings are errors via `filterwarnings`. See `tech/tests.md` for the test inventory.

## Code style notes

- Public API: only what's re-exported from `src/pose_estimation/__init__.py`. Internal helpers may move freely.
- TypedDicts in `_types.py` document dict-passed pipeline state. Treat them as the contract.
- Prefer editing existing modules to introducing new ones; the surface is small on purpose.
- Comments: keep sparse — explain WHY when non-obvious; don't restate WHAT the code does.

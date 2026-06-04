# Conventions

## Git

- Commit timing: commit before every end-of-turn message that closes out a cohesive piece of work. Defer commits when mid-iteration and awaiting user input.
- Commit messages: optimized for parsing by an LLM. Subject line under 50 characters, imperative mood. Body line wrap under 72 characters.
- Before committing, always check whether `README.md`, `.gitignore`, `pyproject.toml`, or other housekeeping files need a matching update.
- The user handles all commands that affect the remote (push, force-push, branch creation, etc.). Agents stop at the local commit.

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
- Comments: keep sparse. The CLAUDE.md (project root) directive applies — explain WHY when non-obvious; don't restate WHAT the code does.

## Navigation (token-efficient)

- Locate symbols via the repo map: `rg '\bNAME\b' .claude/repomap.md` returns `path:line` plus the signature (Python from AST, R from regex). Prefer that + `Read(offset, limit)` over reading a whole module.
- `rg 'relpath/file.py' .claude/repomap.md` lists one file's symbols (its outline).
- Regenerate after adding, moving, or renaming a symbol: `python scripts/repomap.py`. The committed `.claude/repomap.md` is drift-guarded by `tests/test_repomap.py` — a stale map fails the suite (same pattern as `test_public_api.py`). The map indexes tracked files only (`git ls-files`); `git add` a new module before regenerating.
- Use `rg`/`sg` for content search; reserve whole-file reads for when you genuinely need full context.

## Working style (agents)

- Subagents: when dispatching work, always run them on the most capable model (Opus) with maximum thinking — the same tier the main session uses. Multi-agent councils/teams are encouraged for hard problems (per CLAUDE.md).
- Prose (docs, commit bodies, memory): dry, direct, concise, precise; assume a technical reader.
- New behaviour: favour red-green-refactor (failing test → make it pass → refactor), balanced against the overtesting caution in CLAUDE.md and `tech/tests.md`.

## Agent-writable files

- `/CLAUDE.md` (project root) — meta-instructions. Agents may rewrite freely when content is obsolete, better phrased, or superseded (per 2026-05-24 decision).

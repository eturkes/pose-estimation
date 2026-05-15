# .claude/ Manifest

Entry point for agent sessions. Load files on demand using the "when to load" hints.

## Layout

```
.claude/
├── INDEX.md          # this file
├── tech/             # project-specific technical reference
├── memory/           # cross-session learning (append-only)
└── prompts/          # reusable session prompts
```

## tech/ — load on demand

| File | When to load |
|------|--------------|
| `tech/architecture.md` | Modifying or reasoning about pipeline modules, public API, frame data flow. |
| `tech/entrypoints.md`  | Touching CLI behaviour, `main.py`, `run.py`, `benchmark.py`, or `postprocess.py`. |
| `tech/tracking-modes.md` | Anything mode-sensitive: keypoint counts, column prefixes, mode-specific constants. |
| `tech/analysis.md`     | Working on `analysis/*.R` scripts or downstream clinical features. |
| `tech/optimization.md` | Tuning pipeline parameters, running parameter sweeps, or micro-benchmarks. |
| `tech/tests.md`        | Adding/running tests; mapping test files to the modules they cover. |
| `tech/environment.md`  | Dependency, `uv`, `.venv`, R/`renv`, NPU/OpenVINO setup questions. |
| `tech/conventions.md`  | Git messages, ruff/ty/pytest configuration, code style. |

## memory/ — append-only, agent-writable

| File | Use |
|------|-----|
| `memory/decisions.md`  | Record architectural decisions (ADR-lite). Append on decision; never silently revise. |
| `memory/lessons.md`    | Mistakes + corrective rule. Append after a recovered failure. |
| `memory/scratchpad.md` | Transient session notes. Safe to prune; not a source of truth. |

## prompts/

| File | Use |
|------|-----|
| `prompts/kickoff.md` | Reusable kickoff prompt for fresh sessions. Paste into a new session to bootstrap context. |

## Authoring rules

- Always prefer editing `tech/*.md` over `AGENTS.md` (which is now a pointer).
- Always cross-reference with `path/to/file.py:line` so navigation is cheap.
- Always keep tech notes accurate to current code; correct drift the moment you spot it.
- Always append to `memory/` rather than overwrite, unless pruning redundant entries.
- You must request user approval before editing `/CLAUDE.md` (project root meta-instructions).
- README.md is the public/human-discoverable face; keep it concise but informative.

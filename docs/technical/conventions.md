# Engineering conventions

## Git

- Commit messages: [Scoped Commits](https://scopedcommits.com/) — `<scope>: <description>`, scope first (the subsystem/area touched, e.g. `tracking`, `calibration`, `multicam`, or a cross-cutting label such as `Tooling`, `Maintenance`, `Refactor`, `Docs`). For multi-area commits, comma-list the scopes, generalize to one, or use `treewide`. Subject + body take the `CLAUDE.md` `Authoring` standard: subject = `<scope>: <cause> → <fix>`, imperative, one line — the cause→fix shape sets the length, and the log runs 45-95 chars, so no 50-char cap applies; body wrap ≤72 chars; measurements + SHAs kept as payload while the narration around them goes.
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

### Auxiliary campaigns

These committed campaigns run outside the ordered stages. Run each applicable campaign after the
quality gate, and never before it: a `ruff format` pass moves source bytes, which invalidates every
committed determinism digest and every mutation patch anchor. Run every mutation campaign alone. It
rewrites `src/` in place and restores it at the end, so a concurrent job in the same tree reads
mutated bytes.

- **Registry.** Run `uv run python scripts/run_inventory_mutations.py` and `uv run python scripts/check_inventory_determinism.py`. The mutation campaign holds 72 mutants over `inventory.py` and `video_io.py`: 71 are killed, and `M028` is a ruled equivalent. The determinism campaign runs 20 sweeps and 13 tamper classes at the consumer boundary. Rerun both before you quote a corpus-registry claim. A new registry predicate earns a mutant in the same commit.
- **Qualification.** Run `uv run python scripts/check_qualify_determinism.py`. It passes 40 sweeps across both publication modes and 19 consumer-boundary tamper classes. It refuses to run when source bytes move; run `rm -f tests/qualify_determinism_results.json` first for an intentional regeneration.
- **M2.5 alignment.** Run `uv run python scripts/check_m2u5_determinism.py` and `uv run python scripts/run_m2u5_mutations.py`. The first command passes D06-D09. The second command kills all 25 mutants through `tests/test_m2u5_mutants.py`.
- **Calibration-QC determinism.** Run `uv run python scripts/check_calibration_qc_determinism.py`. It passes 21 publication sweeps and 18 consumer-boundary tamper classes in 21 seconds. It refuses to run when source bytes move; run `rm -f tests/calibration_qc_determinism_results.json` first for an intentional regeneration.
- **Calibration-QC mutation.** Run `uv run python scripts/run_calibration_qc_mutations.py`. It kills all 51 publisher mutants through `tests/test_calibration_qc_mutants.py` in under three minutes.

## Maintenance

- A module, CLI flag, output schema, public export, or test-layout change updates its affected `docs/technical/` reference.
- Keep `src/pose_estimation/__init__.py` + `tests/test_public_api.py` synchronized.
- Session/calibration manifest paths + labels are hostile input → preserve containment checks, safe path-component validation, and traversal regression coverage.
- Source, manifests, and tests outrank prose when they disagree; fix the prose.

## Text register

Both registers + the human-facing/code-surface rule live in `CLAUDE.md` `Authoring`; this section owns the repo inventory alone. ASD-STE100 applies to this surface:

- `README.md`, `docs/capture_protocol.md` — the shipped docs. `docs/technical/` is internal → agent register, as is every artifact left unlisted here.
- `analysis/analysis_summary.Rmd` prose outside chunks; operator-visible `cat`/`message`/`warning`/`stop` text + rendered plot/table labels in `analysis/*.R`.
- `argparse` `help=`/`description=`/`epilog=` and console `print()` text under `src/pose_estimation/` + `scripts/`; every Markdown renderer — `validation.py` `_render_markdown`/`_render_qa_markdown`, `scripts/run_report.py`, `scripts/benchmarks/aggregate.py`.

Payload stays code surface wherever a person also reads it: the `to_json` half of `ValidationReport`/`QAReport` beside their Markdown renderers, `manifest.json` beside the run report, CSV + JSON field names, `ARTIFACT_TAG_COLS` values (`coord_space`, `distance_unit`, `metric_qualification` ⊇ `gap-aware`/`gap-unsafe`), and the artifact-name suffixes consumers glob (`_clinical.csv`, `_clinical_windows.csv`, `_clinical_3d.csv`).

The `notes` lists on `Verdict`, `ValidationReport` + `QAReport` cross the boundary: both Markdown renderers list them under `## Notes`, so they take the human register, while `tests/test_validation.py` matches tokens inside them (`legacy schema`, `candidate-view`, `UNVALIDATED`, `reach_raw`, `RMS unassessed`) — retain those tokens when rewording.

Identifiers, flag names, units, defaults, and paths stay verbatim in both registers — a register change never moves a claim.

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
- Comment policy + the agent-legibility bounds live in `CLAUDE.md` `Engineering`.

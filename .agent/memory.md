# Project memory

Context retained only when source, tests, technical docs, roadmap, and git do not expose it cheaply.

## Data boundary

- Patient recordings + adjacent derivatives = sensitive. `videos/`, `output/`, real-data calibration files/directories, and logs stay outside agent context → inspect/decode/copy only under direct per-task clearance.
- Test for real multi-camera footage with the metadata-only probe `UV_PROJECT_ENVIRONMENT=.venv uv run pose-estimation-run --list-sessions`; consume its redacted summary alone, never the underlying session identifiers or media.
- Commit source + synthetic or de-identified fixtures only. `.gitignore` keeps raw media, derived outputs, rig geometry, credentials, and path-bearing logs out.
- A collaborator fork carrying unrelated history reconciles by content onto `main` with source-only attribution → importing its Git object history would retain patient data even after a file deletion.

## Environments

- The container + host share the checkout through different absolute paths, so uv environments are layer-specific. Container work uses `.venv`; host work uses `.venv-host`. Recreate the matching environment after a move; repair text shebang/activation/editable-path metadata only when offline, and regenerate binary/cache artifacts.
- GPU/NPU access depends on the inherited Intel driver/runtime environment (`LD_LIBRARY_PATH`, ICD, Level Zero, and any accelerator `PYTHONPATH`). Confirm with `openvino.Core().available_devices` in the correct uv environment; the pip dependency remains required for generic CPU-only checkouts.

## Worktree gate recipe (`.scratch/worktrees/<name>`)

Every teammate worktree runs the full Python gate concurrently off the one primary environment, read-only:

```sh
export UV_PROJECT_ENVIRONMENT=<primary-tree>/.venv PYTHONPATH="$PWD/src"
uv run --no-sync ruff check && uv run --no-sync ruff format --check \
  && uv run --no-sync ty check && uv run --no-sync pytest
```

- `PYTHONPATH=<worktree>/src` is mandatory: the hatchling editable install resolves `pose_estimation` to the **primary** tree's `src/`, so a worktree gate without it silently tests the primary code and reports green for untested changes.
- `--no-sync` keeps the shared environment unmutated, which is what makes concurrent worktree gating safe.
- Tool caches (`.ruff_cache`, `.pytest_cache`, `.ty_cache`, `.coverage`) are cwd-relative → already private per worktree; no extra state paths needed.
- The R gate is primary-tree-only: `renv/library/` is gitignored, so it never lands in a worktree and the R-library-dependent cases report SKIP where the primary tree reports PASS (17 at present: worktree 452 passed/17 skipped vs primary 469 passed/0 skipped). A green worktree gate is therefore no evidence for changed `analysis/*.R` → route those to a primary-tree run.

## Session launch cost

`headroom wrap claude` blocks on `uvx … serena project index` (`cli/wrap.py:_index_serena_project`, 300 s cap) before Claude Code starts → anything that stalls Serena's indexer is felt as launch latency, and only in the repo that holds it.

- Budget: full cold index = ~12 s (109 files, 7 language servers), warm ~6 s. A launch stalling far past that means one file is eating an LS request timeout (`serena_config.yml` `tool_timeout` 240 − 5 = 235 s each) → `.serena/logs/indexing.txt` names the file.
- `**/*.Rmd` is excluded in `.serena/project.yml` for exactly that reason (R LS never answers `documentSymbol` for R Markdown; `.R` files are unaffected at ~4 files/s). Reach `analysis/analysis_summary.Rmd` by `Read`/`rg`; Serena's symbol + search tools do not see it.
- Serena's own session start is repo-independent (~3.5 s, dominated by the bash LS) and asynchronous — it never blocks the MCP handshake.

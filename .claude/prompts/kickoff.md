# Session kickoff prompt

Paste this prompt at the start of a fresh agent session on this project. The bracketed `<TASK>` line at the end is the only part the user (or steering message) needs to customise.

---

You are continuing work on the **pose-estimation** project. Bootstrap your context as follows before doing anything else:

1. Always read `/CLAUDE.md` first — it is the project-root meta-instructions document and overrides default behaviour. You may rewrite it whenever content is obsolete, better phrased, or superseded.
2. Always read `.claude/INDEX.md` next — it is the manifest of project-specific tech notes and memory files, with hints on when to load each.
3. Load only the `.claude/tech/*.md` files relevant to the task at hand (the INDEX names them). For broad bug-fixing or refactor work, load `architecture.md` + `tests.md` + the module-specific note.
4. Skim the **top** of `.claude/memory/decisions.md` and `.claude/memory/lessons.md` to inherit prior context. Entries are newest-first.
5. When you reach a non-trivial decision, append to `.claude/memory/decisions.md`. When you recover from a mistake worth remembering, append to `.claude/memory/lessons.md`. Phrase lessons positively ("always X", "first X then Y") — avoid "do not"/"never" framings.
6. Whenever code changes invalidate any `.claude/tech/*.md` content, fix the affected note in the same change. Drift is the most common failure mode for this repo.
7. Treat `AGENTS.md` as a pointer; it should remain a short redirect to `CLAUDE.md` + `.claude/INDEX.md`. Project-specific tech belongs under `.claude/tech/`, never inlined into `AGENTS.md`.
8. Push back on ambiguous or under-specified requests with concrete clarifying questions before acting. CLAUDE.md authorises this explicitly.
9. Use TaskCreate for any plan with three or more distinct steps; mark each task `completed` the moment its work is done.

Tooling reminders:
- Python: `uv` + `.venv/` (created on host). `uv run ruff check --fix`, `uv run ruff format`, `uv run ty check`, `uv run pytest`.
- R: `renv`; install with `renv::restore()`.
- Display: `pygame-ce` (Wayland-compatible). OpenCV is `opencv-python-headless`.
- Devices: NPU default; `--device {NPU|CPU|GPU}` on `main.py` / `run.py`. rtmlib supports `--backend {onnxruntime|openvino}`.
- Data dirs (`videos/`, `output/`, `model/`) are git-ignored — keep patient data out of commits.

Then proceed with: <TASK>

---

## Tips for whoever pastes this

- The prompt is intentionally short. CLAUDE.md and `.claude/INDEX.md` carry the heavy context; the kickoff only points at them.
- Replace `<TASK>` with the concrete request, or leave it blank to have the agent wait for follow-up.
- If you want the agent to skip a step (e.g. "don't auto-edit tech notes"), append the override after the task line; CLAUDE.md treats user instructions as final say.

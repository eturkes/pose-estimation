# Agents — start here

This project's agent context is split for selective loading:

1. **`/CLAUDE.md`** — meta-instructions (how you operate, memory, decisions, etc.). Read first. Agent-writable.
2. **`.claude/INDEX.md`** — manifest of project-specific tech notes and memory files. Read second. Load `.claude/tech/*.md` on demand from there.
3. **`/session`** (`.claude/commands/session.md`) — slash command that bootstraps a fresh session. Run `/session <TASK>` to override the roadmap, or `/session` alone to continue it.

`README.md` is the human-facing project description; it links to specific tech notes when more detail is needed.

> This file is intentionally a pointer. Project tech reference lives under `.claude/tech/`; never inline it back here.

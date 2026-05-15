# Architectural decisions

Append-only log of decisions that future sessions must respect. Always add new entries to the **top** so the newest is read first when the file is loaded selectively.

## Entry schema

```
## YYYY-MM-DD — <short title>

**Context.** What problem prompted the decision.
**Decision.** What was chosen.
**Alternatives considered.** Briefly, what was rejected and why.
**Consequences.** Constraints this places on future work; how to reverse if needed.
**References.** Files, tests, or commits that encode the decision.
```

---

## 2026-05-16 — Split project context into `CLAUDE.md` (meta) + `.claude/notes/` (tech)

**Context.** A new project-root `CLAUDE.md` introduces meta-instructions for AI agents (memory system, LLM-optimised docs, token efficiency). The existing `AGENTS.md` mixed meta with project-specific tech reference; size was growing and drift from code was accumulating.
**Decision.** Keep `CLAUDE.md` at the project root as the agent meta-instructions document. Move project-specific technical reference into `.claude/tech/*.md` for selective loading. Replace `AGENTS.md` with a pointer for tools that follow the AGENTS.md convention. Add `.claude/memory/` (decisions, lessons, scratchpad) and `.claude/prompts/kickoff.md`.
**Alternatives considered.** (a) Symlink `AGENTS.md → CLAUDE.md`: rejected — mixes meta and tech context and clutters `CLAUDE.md` (which is owner-approval-only). (b) Keep both side by side: rejected — guaranteed drift between two parallel docs. (c) Fold everything into one mega-file: rejected — defeats selective loading and token efficiency.
**Consequences.** All project tech notes live under `.claude/tech/`. `AGENTS.md` is a thin pointer; do not grow it. `CLAUDE.md` modifications require explicit user approval. When `tech/*` content drifts from code, fix it in the same change that introduced the drift.
**References.** `.claude/INDEX.md`, `AGENTS.md`, `/CLAUDE.md`, `.claude/prompts/kickoff.md`.

# Scratchpad

Transient working notes — anything from "current investigation" to "half-finished idea". Safe to prune at session boundaries. Never treat as a source of truth; promote stable findings to `tech/`, decisions to `decisions.md`, lessons to `lessons.md`.

## How to use

- Start each session by skimming the most-recent entries (top of file).
- Append `## YYYY-MM-DD HH:MM — <session topic>` and write freely.
- When closing a session, either prune the entry or summarise it into a longer-lived file.

---

## 2026-06-01 — CLAUDE.md alignment audit + R-env maintenance

User revised CLAUDE.md; this session propagated the changes and acted on the maintenance directive. Full record in `decisions.md` and `lessons.md` (both 2026-06-01). Highlights:

- **Doc propagation** (single canonical home each, no duplication): `environment.md` (Debian-Distrobox/openSUSE two-layer model + LSP/`bgcmd` tooling; dropped drift-prone version literals); `INDEX.md` authoring rule (entries omit version numbers); `conventions.md` "Working style (agents)" (subagent-model + dry prose + red-green-refactor); pruned incidental versions from the R-4.6 migration decision. `kickoff.md` left untouched (the subagent directive already lives in CLAUDE.md).
- **Maintenance:** removed orphaned renv R-4.5 tree (111 dangling links); `renv::restore()` healed 5 dangling R-4.6 links (0 remain); reinstalled R-graphics apt sysreqs a container rebuild had dropped (ragg/ggplot2 verified loading); apt cache cleaned.
- **Deferred / offered:** a persistent setup script for the R sysreqs (depends on container-recreation cadence); agent-oriented-languages exploration (CLAUDE.md pointer — forward-looking, no doc change needed yet).

**Project status:** Phase 1 ✓ (stability), Phase 2 ✓ (clinical metrics), Phase 3 pending (3D pipeline, awaiting footage), Phase 4 ongoing (maintenance).

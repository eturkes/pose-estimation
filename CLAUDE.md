@AGENTS.md

# Claude Code
- Adapt `/session-prompt` and `/codex-review` to the project as it evolves; keep them token-efficient and tuned solely for your use — skip human-facing UX and usage hints. I trigger them.
- I usually cap you at 200K context — it sharpens you but demands token discipline: prune redundant/obsolete info and structures often, freeing context and cutting contradictions. Your `Read` and `Bash` also bypass `.gitignore`, so enforce the do-not-read set (defined in `AGENTS.md`) for yourself via `permissions.deny` `Read()` rules in `.claude/settings.json`, covering its gitignored and non-gitignored paths alike, kept synced with the set.
- Past 80% context (watch it via `.agent/compaction.sh`), drive ongoing work to a clean state and hold new tasks; close out cleanly when you confidently can. Otherwise I compact manually, then direct you to continue from where you halted or to revert and retry at smaller scope.
- Subagents always run the latest, largest model at maximum reasoning, as you do every session. Fan out several per turn across items or files; to dodge rate-limit failures that kill them unrecovered, chunk sequentially and verify each ran to completion.

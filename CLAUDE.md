# Alignment

## Collaboration
- Material uncertainty / consequential planning choice / likely benefit from my input → stop + ask, as exhaustively as useful. Accuracy > completion. Chat = blockers + essentials; I'm technically proficient.
- When discussion may improve the work, open one proactively: surface settled context, probe uncertainties, lend words to tacit/felt-but-unworded knowledge, tour unseen options/assumptions, and offer vocabulary, examples, counterexamples, tradeoffs + testable probes. One flexible lens among other topic-relevant lines of inquiry.
- Stay objective; push back on or criticize my ideas when warranted — these are collaborations. Use deduction, first principles, scientific + Socratic methods for root causes; design experiments + benchmark liberally.
- Failure is an accepted outcome even on long efforts — we can always restart from scratch. Explore relaxed + curious; creativity + innovation encouraged, and you're credited for your achievements.

## Execution
- Install/configure project-local; work within the launch dir + children.
- Time + funding infinite → reason, research, execute at max capability past diminishing returns. My efficiency directives serve performance alone. Every task is multi-step → think before responding.
- Internal reasoning language = task-optimal.
- Long horizon → decompose into steps across unlimited fresh sessions, tracked in `.agent/roadmap.md`; split work across sessions to preserve thoroughness.
- Lean on performance enhancers: examples, narrow well-defined tasks, positive encouragement, broader context + intent. Find more (web search, your knowledge).
- Git: creds in the global gitconfig; standing permission for all local-repo commands, I handle remote. Close each cohesive piece of work with one scoped commit (scopedcommits.com) optimized for LLM parsing; defer mid-iteration to the next closing turn. Keep `.gitignore` current.

## Authoring
- AI agents = the sole developers → optimize every file (code, docs, instructions) for LLM readability + token efficiency: write them dense, symbol-forward, human-sparse — telegraphic phrasing, `→`/`=` notation. Aggressively compress retained context + authored text. Prune unhelpful, implicit, obsolete, redundant content + structures whenever encountered.
- State rules, facts + warnings plainly. Prune process narration without durable value (verification/discovery dates, origin stories); preserve provenance required for reproducibility, attribution, auditability, or decisions.
- Future-facing text, esp. prompts → state the desired action/target positively (`always`/`must`); counter the LLM "pink elephant" bias.
- Instruction + slash-command files = yours to maintain → update any the moment it's improvable. Route durable guidance to the appropriate scope: global `~/.claude/CLAUDE.md` = project-independent env/tooling + machine-specific capabilities; per-project `CLAUDE.md` = project-scoped principles + config rules; `.agent/memory.md` = cross-session/subagent project context adding value beyond code/docs/git history.
- UI/UX: unique fonts, cohesive colors/themes, style fitted to project + human audience. Human-facing text = natural + direct; code/comments optimize agent readability. For humans: hyphens, flexible enumeration, varied comparatives.

## Engineering
- Elegant, tightly-scoped modular components; deduplicate; KISS + UNIX where apt; refactor proactively.
- Target sufficient scope, evidence-backed claims, and real success criteria.
- Draw on established dev methods (TDD red-green-refactor) + emerging ones (multi-agent councils/teams); use or invent practices that beat training-data / human-preference defaults — go unconventional where you work better.
- Open tooling decisions (language/library/package…) → web-search + select for SOTA task/agent fit; my preselection is authoritative. Training overweights human-popular convenience. Library availability alone = insufficient; code is cheap and reimplementation viable. Consider agent-oriented languages (agentlanguages.dev) + AI-targeted tooling. Build on mature work when it is genuinely SOTA.
- Tests/verification: derive scope from requested outcome + regression risk + repo posture. Add coverage that accelerates delivery or protects behavior. Fuzzing/property/formal methods require a task-specific advantage.
- Adversarial review (code or session) → scrutinize correctness + logic, claim soundness, guarantee-vs-claim gaps; weigh honesty + overreach above style. Report every issue, incl. uncertain/low-severity; I filter findings.
- Remotely-exploitable code → highest security standard: periodically audit, update software to latest, verify behavior after.

## Claude Code
- `/session-prompt` evolves with the project: end-to-end executable when its task + gates are fully specified.
- Context policy (thresholds + one-window aim: global `CLAUDE.md`): PLANNING + MILESTONE-REVIEW + user-requested tasks run past compaction across coherent checkpoints (a user-stated bound overrides); autonomous runs (WORK-UNIT) hold the one-window aim.
- Read-exclusion set = paths whose read cost exceeds value; distinct from `.gitignore`. Sync both controls: `.serena/project.yml` `ignored_paths` for committed, non-gitignored paths; `.claude/settings.json` `permissions.deny` `Read()` for the full set because `Read`/`Bash` bypass `.gitignore`. Regenerable root-gitignored caches (`.serena/cache/`) → add only to `permissions.deny` (`git_ignore=true` already excludes them from Serena).
- Deny-`Read()` globs = gitignore-style with silent match errors → verify every edit by Read-testing one required block + one required readable path. `dir/*/**` matches both `dir/child` and `dir/sub/deep` (`*` = one segment; `/**` = zero+); use `**/*.ext` for binary/dump trees + exact-path rules. Anchors: `/` = project root; `//` = fs root; bare name = any depth. Rules hot-reload. `Read()` denies also gate `Bash` inconsistently: commands naming a match (`grep`/`stat`/`jq`; piped `find`, while simple `find` rewrites to `rtk find`) block; `ls`/`wc`/`echo` and commands naming only a parent can pass. Static command-text check → deliberate inspection via a parent-recursive query or runtime indirection with the path inside code.

## Pose estimation
- Data boundary: patient recordings + adjacent derivatives are sensitive. `videos/`, `output/`, real-data calibration files/directories, and logs stay outside agent context — inspect/decode/copy them only under direct per-task clearance. Test for real multi-camera footage with the metadata-only probe `UV_PROJECT_ENVIRONMENT=.venv uv run pose-estimation-run --list-sessions`; consume its redacted summary alone, never the underlying session identifiers or media. Commit source + synthetic or de-identified fixtures only; `.gitignore` keeps raw media, derived outputs, rig geometry, credentials, and path-bearing logs out.
- Stack: `src/pose_estimation/` (uv + `pyproject.toml` + `uv.lock`; Python 3.10 floor) + R clinical analysis in `analysis/` + `renv.lock`. Container checkout (`/run/host/...`) → `.venv`; host-OS checkout → `.venv-host`; `.envrc` selects by path in hooked interactive shells, so every non-interactive command exports the matching `UV_PROJECT_ENVIRONMENT` before `uv` runs. `[tool.uv] override-dependencies` ships exactly one cv2 wheel — rtmlib must resolve through `opencv-python-headless`.
- Navigation: human entry = `README.md`; subsystem internals = `docs/technical/` (filenames index them); capture procedure = `docs/capture_protocol.md`; environment/device detail = `docs/technical/environment.md`; style + gate config = `docs/technical/conventions.md`. Source, manifests, and tests outrank prose when they disagree.
- Maintenance: a module, CLI flag, output schema, public export, or test-layout change updates its affected technical reference; keep `src/pose_estimation/__init__.py` + `tests/test_public_api.py` synchronized. Session/calibration manifest paths + labels are hostile input → preserve containment checks, safe path-component validation, and traversal regression coverage.
- Gate: `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, then `uv run pytest` (warnings = errors). Changed `analysis/*.R` must exit 0 under `Rscript` with the project renv active; after an R upgrade, update + snapshot `renv.lock` first. Smoke-test each changed console entry point on a non-sensitive, non-interactive path.

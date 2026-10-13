---
paths:
  - "README.md"
  - "docs/*.md"
  - "docs/technical/conventions.md"
  - "analysis/*.Rmd"
  - "analysis/*.R"
  - "src/pose_estimation/**/*.py"
  - "scripts/**/*.py"
---

# Text register — scratch scanners pending port

Scratch-local validators = temporary encodings (global `CLAUDE.md` `Engineering`); each port = a `.agent/deferred.md` row.

- `.scratch/steq.py` — ASD-STE100 register scan over the human-facing surface (inventory: `docs/technical/conventions.md` → *Text register*). Drops fences/tables/headings/frontmatter, joins wrapped lines into blocks so a sentence is measured whole, splits on `.!?`, flags `LONG` (> `--max`), `FILLER`, `CONTRACTION` (also fires on possessive `'s`), `PASSIVE` (be-verb + participle heuristic). Code-file mode samples quoted `help=`/`description=`/`title=` strings only. **Every flag is a lead for judgment, never a verdict**: global `Authoring` asks for short sentences and keeps text caps out of checks, so `LONG` at any `--max` marks a sentence to read. Measured over the four shipped surfaces: 48 flags at `--max 20`, 24 at `--max 25`. Of the 24, both `CONTRACTION` hits are possessives (`instrument's`, `solve's`) and at least 4 of 21 `PASSIVE` hits are predicate adjectives (`is untested`, `is unmeasured`, `is unaffected`, `is closed`) — so **23 of 24 are heuristic output awaiting a human**, and several true passives are mandated by the claim boundary's own "may not be claimed" phrasing.
- `.scratch/fidelity.sh <base-ref> <file>…` — pairs with it: diffs the multiset of format specifiers, `--flags`, backticked spans, file names and numbers between a base ref and the working tree. A register-only edit must show no delta; every delta needs an explanation. Caught the p-value reformat (`p<.05` → `p < 0.05`) and confirmed 14 R files invariant.

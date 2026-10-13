---
paths:
  - ".agent/spec-lint"
  - ".agent/spec.md"
  - "tests/test_spec_lint.py"
---

# Spec lint

- `.agent/spec-lint` = a byte copy of `~/.local/app/agents/claude/spec-lint` (template
  `Session flow`); fixes land upstream and reach this repo at the next refresh (`upstream-sync.md`).
  The decisive gate runs it through `tests/test_spec_lint.py`: the live `.agent/spec.md` lints rc 0
  with empty output, and three planted specs (a Phase history line, a stray `Tasks` line, a bolded
  Phase line 1) each lint rc 1 naming their finding.
- Firing, measured (`P pytest tests/test_spec_lint.py`, 4 cases): each mutant of `.agent/spec-lint`
  that neutralises one check's `bad()` call reds exactly its own planted case — history check 1,
  `Tasks` row check 1, Phase line-1 check 1 — and a history line appended to the live spec reds
  `test_live_spec_lints_clean` alone.

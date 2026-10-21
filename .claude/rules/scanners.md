---
paths:
  - "scripts/gate.sh"
  - "scripts/install-tools.sh"
  - "scripts/scanner_seeds.sh"
  - "pyproject.toml"
  - "uv.lock"
  - "renv.lock"
  - "tools/review-ui/uv.lock"
  - "tools/cocowb-eval/uv.lock"
  - ".github/dependabot.yml"
---

# Security scanners — what runs, what it covers, how it fires

- **Dependency audit** = `uv audit --locked` over the three uv lockfiles (root, `tools/review-ui/`,
  `tools/cocowb-eval/`; `--preview-features audit-command` silences the experimental banner) + `osv-scanner scan source
  -L renv.lock` (122 CRAN packages; Dependabot has no renv ecosystem, so this step is its only
  update signal). The vendored review-UI JS (Plotly, fonts) carries no lockfile and no audit.
- **Secret scan** = betterleaks 1.9.0 (maintained by gitleaks' original author; MIT), default rules,
  `--redact`, live validation off (it is opt-in, `--validation`): `git .` over every commit + `dir`
  over `git ls-files -co --exclude-standard`. The AWS rule needs the key pair — a lone `AKIA…` id
  scans clean.
- **Static analysis** = ruff `S` (flake8-bandit) inside the ruff step + shellcheck over tracked
  `*.sh`. Ignored by rule, each with its reason in `pyproject.toml`: S603/S607 everywhere (list argv,
  PATH tools), S101 in tests + five invariant-assert modules, S105/S106/S102 in tests, S311 in
  scripts. `.agent/spec-lint` stays outside shellcheck: a template file, and its awk body reads as
  SC2016.
- **Binaries are pinned + checksummed** in `scripts/install-tools.sh` → `.tools/bin/` (git-ignored);
  bump = new version + its release sha256 in that script. Python-side tools (ruff, ty,
  shellcheck-py, check-jsonschema) come from `uv.lock`. Scanners need network (OSV, PyPI advisories).
- **Firing, measured** (`scripts/scanner_seeds.sh`, a full-gate step): urllib3 2.7.0 lock →
  `uv audit` rc 1; renv.lock with commonmark 1.7 → osv-scanner rc 1 `RSEC-2023-6`; a run-time AWS key
  pair in a file and in a commit → betterleaks rc 1 `aws-access-token` (dir + git); `shell=True`
  on a built string → ruff rc 1 `S602` under the repo config (silent before `S` joined `select`);
  unquoted `$1` → shellcheck rc 1 `SC2086`. The seeds assemble the key at run time, so no literal
  secret is committed and the repo's own scan stays clean. A seed that stops firing fails the gate.
- Dependabot schema: `check-jsonschema --builtin-schema vendor.dependabot` (static step); an
  unknown `package-ecosystem` reds it.

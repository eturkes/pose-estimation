"""Every documented root-project command carries its prefix, so none is recalled by failure.

A bare `uv run`, `python`, `pytest` or `pose-estimation-*` imports the host OpenVINO build through
the inherited `PYTHONPATH` and dies at `GLIBC_2.43` (`.claude/rules/gates.md`). Accepted forms:
`P <cmd>` / `$P <cmd>`, `scripts/gate.sh <cmd>`, the spelled-out gate prefix, the accelerator
recipe (`PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python …`), a subproject's own
`uv run --directory tools/<name> …` or its `UV_PROJECT_ENVIRONMENT`, the host layer's
`.venv-host`, stdlib `python3 .scratch/…`, and a span the prose calls bare.
`.agent/archive/` is frozen and keeps its bare forms.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Iterator

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_ENV = r"(?:[A-Z_][A-Z0-9_]*=\S+\s+)*"
_COMMAND = re.compile(
    rf"^{_ENV}(?:"
    r"uv run(?!\s+(?:--quiet\s+)?--directory)\s+\S"
    r"|python3?\s+(?!\.scratch/|-m review_ui\b|tools/|-I\b)(?!-[cmE]$)\S"
    r"|pytest\s+\S"
    r"|pose-estimation(?:-[a-z0-9-]+)?\s+[^\s`|]"
    r")"
)
_ACCEPTED_BEFORE = re.compile(r"(?:\bP|\$P|scripts/gate\.sh(?:\s+--fast)?|\bbare)\s*`?$")
_PREFIX = "env -u LD_LIBRARY_PATH"


def surfaces(root: pathlib.Path = _ROOT) -> list[pathlib.Path]:
    """The row's scope: spec, rules, docs, scripts."""
    return [
        root / ".agent" / "spec.md",
        *sorted((root / ".claude" / "rules").glob("*.md")),
        *sorted((root / "docs").rglob("*.md")),
        *sorted(p for p in (root / "scripts").rglob("*") if p.suffix in (".py", ".sh")),
    ]


def _joined(lines: list[tuple[int, str]]) -> Iterator[tuple[int, str]]:
    """Join shell continuations (a trailing `\\`, doubled inside a Python string) into one line."""
    pending: tuple[int, str] | None = None
    for number, line in lines:
        if pending is not None:
            number, line = pending[0], f"{pending[1]} {line.strip()}"
        if line.rstrip().endswith("\\"):
            pending = (number, line.rstrip().rstrip("\\"))
            continue
        pending = None
        yield number, line
    if pending is not None:
        yield pending


def _spans(path: pathlib.Path, text: str) -> Iterator[tuple[int, str, str]]:
    """(line, candidate command, text before it): code spans, fenced lines, script comments."""
    lines = text.splitlines()
    if path.suffix != ".md":
        bodies = []
        for number, line in enumerate(lines, 1):
            body = re.sub(r"^(?:#+|\"\"\"|\'\'\')\s*", "", line.strip())
            bodies.append((number, re.sub(r"^(?:Run|Rerun|Usage|Example)s?:\s*", "", body)))
        for number, body in _joined(bodies):
            yield number, body, ""
    fenced, block, prose = False, [], []
    for number, line in enumerate(lines, 1):
        if path.suffix == ".md" and line.strip().startswith("```"):
            if fenced:
                for row, command in _joined(block):
                    yield row, command.strip(), ""
                block = []
            fenced = not fenced
            prose.append("")
            continue
        if fenced:
            block.append((number, line))
            prose.append("")
        else:
            prose.append(line)
    # Inline spans may wrap across lines; fenced bodies are blanked so their backticks stay out.
    joined = "\n".join(prose)
    for match in re.finditer(r"`([^`]+)`", joined):
        row = joined.count("\n", 0, match.start()) + 1
        before = joined[joined.rfind("\n", 0, match.start()) + 1 : match.start()]
        yield row, " ".join(match.group(1).split()), before


def bare_invocations(path: pathlib.Path, text: str) -> list[tuple[int, str]]:
    found = []
    for number, span, before in _spans(path, text):
        command = re.sub(r"^\$\s+", "", span.strip())
        if command.startswith(_PREFIX) or _ACCEPTED_BEFORE.search(before):
            continue
        # A subproject's own environment or the host layer (host glibc matches the host build).
        if re.match(r"UV_PROJECT_ENVIRONMENT=\S*(?:tools/|\.venv-host)", command):
            continue
        if _COMMAND.match(command):
            found.append((number, command))
    return found


def test_no_surface_carries_a_bare_invocation() -> None:
    files = surfaces()
    assert len(files) > 50, "the surface list is vacuous"
    hits = [
        f"{path.relative_to(_ROOT)}:{line}: {command}"
        for path in files
        for line, command in bare_invocations(path, path.read_text(encoding="utf-8"))
    ]
    assert hits == []


def test_scanner_flags_each_bare_form_and_accepts_each_prefixed_one() -> None:
    markdown = pathlib.Path("seed.md")
    bare = [
        "Run `python scripts/check_claim_report.py` first.",
        "Then `pose-estimation-cohort --out cohort` publishes.",
        "```bash\nuv run pytest -q tests/x.py\n```",
        "```sh\npython -m pose_estimation.run --source v.mp4\n```",
    ]
    for text in bare:
        assert bare_invocations(markdown, text), text
    assert bare_invocations(pathlib.Path("seed.py"), '"""Run:\n    python scripts/seed.py\n"""')
    accepted = [
        "Run `P python scripts/check_claim_report.py` first.",
        "`scripts/gate.sh pytest tests/x.py` and `$P pytest -q`.",
        "`env -u LD_LIBRARY_PATH PYTHONPATH=$PWD/src uv run --no-sync pytest`",
        '```bash\nPYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/corpus_run_2d.py\n```',
        "`uv run --directory tools/review-ui python -m review_ui` serves it.",
        "`python3 .scratch/after_table.py a.json b.json` reads JSON.",
        "A bare `uv run pytest` fails; `pytest` is strict; `python -m` resolves; `uv run` syncs.",
        "`pose-estimation-cohort` publishes the aggregate.",
        "`UV_PROJECT_ENVIRONMENT=.venv-host uv run python -m pose_estimation.run` on the host.",
        "`UV_PROJECT_ENVIRONMENT=<primary>/tools/review-ui/.venv uv run --no-sync` serves it.",
    ]
    for text in accepted:
        assert bare_invocations(markdown, text) == [], text

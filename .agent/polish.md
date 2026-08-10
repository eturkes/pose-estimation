# Polish register

Deferred off-spine improvements + data-tier remediation. Rows are born at deferral time in whatever session finds them (acceptance check written there, while the evidence is fresh); `/session-polish` is the sole consumer — stateless, any order, no milestone coupling.

Row schema — `pri` 1 (highest) … 3 · `size` S ≤15% window | M ≤35% | L = session · `where` = `file:line` or artifact · `why` = evidence pointer (SHA, run output, finding id) · `acceptance` = the check that must run green under MAIN's own rerun.

Lifecycle — done ⇒ prune the row in the same `<scope> (polish): …` commit · dead evidence pointer or acceptance check ⇒ append `stale(<why>)` in place, next `/session-roadmap` session re-rules it · item implying spine work ⇒ move it to `spine?` below + report to the user.

Items run inside the artifact's existing assurance tier; tier raises, new units + scope-source changes belong to `/session-roadmap`.

## Items

| pri | size | where | item | why | acceptance |
| --- | ---- | ----- | ---- | --- | ---------- |

_empty_

## spine?

Findings a polish session judged spine work — ruled by `/session-roadmap`, not executed here. Row: `spine? <finding> | why: <evidence>`.

_empty_

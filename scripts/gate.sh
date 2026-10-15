#!/usr/bin/env bash
# gate.sh — the gate command (.claude/rules/gates.md).
#   gate.sh                    full gate: spec-lint, ruff check, ruff format --check, ty check, pytest
#   gate.sh --fast [pytest…]   fast gate: the four static steps + the named tests, test_c8_08 deselected
#   gate.sh <cmd…>             <cmd> under the gate prefix
# Every step runs from the repo root under the prefix; an exported UV_PROJECT_ENVIRONMENT passes
# through untouched (worktree recipe). A failed step leaves the later ones running, so one run
# reports every failure; rc 1 names them.
set -u
root=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$root" || exit 2

# shellcheck disable=SC2317  # reached through step
prefixed() { env -u LD_LIBRARY_PATH PYTHONPATH="$root/src" uv run --no-sync "$@"; }

failed=()
step() {
	local name=$1 t0=${EPOCHREALTIME/,/.} rc
	shift
	"$@"
	rc=$?
	# EPOCHREALTIME + printf %f follow the locale's decimal mark; awk formats in C.
	printf 'gate: %-22s rc=%-3d %9s s\n' "$name" "$rc" "$(awk "BEGIN { printf \"%.2f\", ${EPOCHREALTIME/,/.} - $t0 }")"
	((rc == 0)) || failed+=("$name")
}

static() {
	step spec-lint sh .agent/spec-lint
	step ruff-check prefixed ruff check -q
	step ruff-format prefixed ruff format --check -q
	step ty-check prefixed ty check
}

case ${1-} in
--fast)
	shift
	static
	# test_c8_08 replays the whole suite in a subprocess, so it never belongs to a fast run.
	(($#)) && step pytest prefixed pytest -q --durations=5 \
		--deselect tests/test_r_timebase_truth.py::test_c8_08 "$@"
	;;
'')
	static
	step pytest prefixed pytest -q --durations=15
	;;
*)
	prefixed "$@"
	exit
	;;
esac

((${#failed[@]} == 0)) && { echo "gate: PASS"; exit 0; }
echo "gate: FAIL ${failed[*]}"
exit 1

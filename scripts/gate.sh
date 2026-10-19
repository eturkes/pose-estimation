#!/usr/bin/env bash
# gate.sh — the gate command (.claude/rules/gates.md).
#   gate.sh                    full gate: static steps, scanners, pytest
#   gate.sh --fast [pytest…]   fast gate: static steps + the named tests, test_c8_08 deselected
#   gate.sh <cmd…>             <cmd> under the gate prefix
# Static = spec-lint, ruff check (security rules S included), ruff format --check, ty check, shell
# lint, Dependabot schema. Scanners = dependency audit (both uv.lock via uv audit, renv.lock via osv-scanner), secret
# scan (history + working tree), scanner seeds (each scanner fires on a planted input); they need
# network + one run of scripts/install-tools.sh.
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
	local sh
	mapfile -t sh < <(git ls-files '*.sh')
	step shellcheck prefixed shellcheck "${sh[@]}"
	step dependabot prefixed check-jsonschema -q --builtin-schema vendor.dependabot .github/dependabot.yml
}

tools=$root/.tools/bin
# shellcheck disable=SC2317,SC2329  # audit + secrets are reached through step
audit() {
	local rc=0
	uv audit --locked --preview-features audit-command -q || rc=1
	uv audit --locked --preview-features audit-command -q --directory prototype/review-ui || rc=1
	"$tools/osv-scanner" scan source -L renv.lock >/dev/null || rc=1
	return "$rc"
}
# History (every commit) + every tracked or untracked, not-ignored file in the working tree.
# shellcheck disable=SC2317,SC2329
secrets() {
	local leaks=("$tools/betterleaks" --no-banner --redact --log-level warn)
	"${leaks[@]}" git . || return 1
	git ls-files -z -co --exclude-standard | xargs -0 "${leaks[@]}" dir
}
scanners() {
	[[ -x $tools/betterleaks && -x $tools/osv-scanner ]] ||
		echo "gate: scanners missing — run scripts/install-tools.sh" >&2
	step audit audit
	step secrets secrets
	step scanner-seeds scripts/scanner_seeds.sh
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
	scanners
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

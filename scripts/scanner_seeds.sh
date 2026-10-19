#!/usr/bin/env bash
# scanner_seeds.sh — prove each gate scanner fires: plant one known-bad input per scanner in a temp
# dir, run the scanner exactly as scripts/gate.sh does, demand a nonzero rc AND the named finding.
# rc 0 = every seed fired; rc 1 = names each scanner that stayed silent. Needs network (OSV, PyPI).
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tools="$root/.tools/bin"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
silent=()

expect() { # name needle cmd…  — fires ⟺ rc != 0 and output contains needle
	local name=$1 needle=$2 out rc
	shift 2
	out=$("$@" 2>&1)
	rc=$?
	if ((rc != 0)) && [[ $out == *"$needle"* ]]; then
		printf 'seed: %-14s fired (rc=%d, %s)\n' "$name" "$rc" "$needle"
	else
		printf 'seed: %-14s SILENT (rc=%d, wanted %s)\n' "$name" "$rc" "$needle"
		silent+=("$name")
	fi
}

# uv audit: a lock pinning urllib3 2.7.0 (PYSEC-2026-4175..4177, fixed 2.8.0).
mkdir "$tmp/uvseed"
printf '[project]\nname = "seed"\nversion = "0"\nrequires-python = ">=3.10"\ndependencies = ["urllib3==2.7.0"]\n' \
	>"$tmp/uvseed/pyproject.toml"
uv lock --quiet --directory "$tmp/uvseed"
expect uv-audit urllib3 uv audit --locked --preview-features audit-command --directory "$tmp/uvseed"

# osv-scanner on renv.lock: commonmark 1.7 (RSEC-2023-6, fixed 1.8).
mkdir "$tmp/renvseed"
cat >"$tmp/renvseed/renv.lock" <<'EOF'
{"R": {"Version": "4.4.0", "Repositories": [{"Name": "CRAN", "URL": "https://cloud.r-project.org"}]},
 "Packages": {"commonmark": {"Package": "commonmark", "Version": "1.7", "Source": "Repository", "Repository": "CRAN"}}}
EOF
expect osv-renv RSEC-2023-6 "$tools/osv-scanner" scan source -L "$tmp/renvseed/renv.lock"

# betterleaks: an AWS key pair assembled at run time (no literal in this file), in a working-tree
# file and in a committed one. The key id alone does not fire: the rule wants the secret beside it.
rand() { LC_ALL=C tr -dc "$1" </dev/urandom | head -c "$2"; }
mkdir "$tmp/leak"
printf 'aws_access_key_id = AKIA%s\naws_secret_access_key = %s\n' "$(rand A-Z2-7 16)" \
	"$(rand 'A-Za-z0-9/+' 40)" >"$tmp/leak/config.ini"
leaks=("$tools/betterleaks" --no-banner --redact --report-format json --report-path -)
expect secrets-dir aws-access-token "${leaks[@]}" dir "$tmp/leak"
git -C "$tmp/leak" init -q && git -C "$tmp/leak" add config.ini &&
	git -C "$tmp/leak" -c user.name=seed -c user.email=seed@invalid commit -qm seed
expect secrets-git aws-access-token "${leaks[@]}" git "$tmp/leak"

# ruff security rules under the repo's own config: shell=True on a built string (S602).
printf 'import subprocess\n\n\ndef f(x):\n    subprocess.call("ls " + x, shell=True)\n' >"$tmp/seed_s602.py"
expect ruff-S S602 env -u LD_LIBRARY_PATH uv run --no-sync --directory "$root" \
	ruff check --no-cache --config "$root/pyproject.toml" "$tmp/seed_s602.py"

# Shell lint seed: an unquoted expansion (SC2086).
# shellcheck disable=SC2016  # the seed's unexpanded $1 is the point
printf '#!/bin/sh\nrm -rf $1/*\n' >"$tmp/seed.sh"
expect shellcheck SC2086 env -u LD_LIBRARY_PATH uv run --no-sync --directory "$root" shellcheck "$tmp/seed.sh"

((${#silent[@]} == 0)) && { echo "seed: all fired"; exit 0; }
echo "seed: SILENT ${silent[*]}"
exit 1

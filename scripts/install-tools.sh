#!/usr/bin/env bash
# install-tools.sh — fetch the gate's pinned non-Python scanners into .tools/bin (git-ignored).
# Python-side tools (ruff, ty, shellcheck-py) come from uv.lock; these two ship as Go binaries.
# Each download is checked against the sha256 recorded here; a mismatch aborts before install.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
bin="$root/.tools/bin"
mkdir -p "$bin"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fetch() { # name version url sha256 [tar-member]
	local name=$1 version=$2 url=$3 sum=$4 member=${5-}
	if [[ -x "$bin/$name" && $("$bin/$name" --version 2>&1 || "$bin/$name" version 2>&1) == *"$version"* ]]; then
		echo "install-tools: $name $version present"
		return
	fi
	curl -fsSL -o "$tmp/$name.dl" "$url"
	echo "$sum  $tmp/$name.dl" | sha256sum -c --quiet
	if [[ -n $member ]]; then
		tar xzf "$tmp/$name.dl" -C "$tmp" "$member"
		install -m 0755 "$tmp/$member" "$bin/$name"
	else
		install -m 0755 "$tmp/$name.dl" "$bin/$name"
	fi
	echo "install-tools: $name $version installed"
}

fetch betterleaks 1.9.0 \
	https://github.com/betterleaks/betterleaks/releases/download/v1.9.0/betterleaks_1.9.0_linux_x64.tar.gz \
	f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7 betterleaks
fetch osv-scanner 2.7.0 \
	https://github.com/google/osv-scanner/releases/download/v2.7.0/osv-scanner_linux_amd64 \
	160c6253e2922e9e0a6a9771c5c4c40a4b3c6090b34a3cdd5e80c96cc2e4218b

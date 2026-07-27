#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${IN_NIX_SHELL:-}" ]]; then
  exec nix-shell -p cargo rustc --run "bash .auto/checks.sh"
fi

cargo test --quiet

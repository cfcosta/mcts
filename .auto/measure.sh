#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep all Rust tools supplied by NixOS rather than relying on host PATH state.
if [[ -z "${IN_NIX_SHELL:-}" ]]; then
  exec nix-shell -p cargo rustc --run "bash .auto/measure.sh"
fi

# This fixed-work driver mirrors representative cases from benches/mcts.rs.
# Unlike Criterion's long statistical runs, it completes quickly enough for
# an autonomous loop while retaining broad end-to-end and hot-path coverage.
taskset -c 6 cargo run \
  --release \
  --quiet \
  --manifest-path .auto/bench/Cargo.toml

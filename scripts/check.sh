#!/bin/sh
# All local gates. Run before pushing.
set -e
cd "$(dirname "$0")/.."
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
sh scripts/lint-arch.sh

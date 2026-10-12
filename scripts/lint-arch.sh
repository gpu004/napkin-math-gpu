#!/bin/sh
# Custom architecture lints. Each violation prints what is wrong and how to
# fix it; exits 1 when anything is found.
cd "$(dirname "$0")/.." || exit 1
fail=0

hit() {
  printf 'Violation: %s\nFix: %s\n' "$1" "$2"
  fail=1
}

line_comments=$(grep -rn --include='*.rs' -E '(^|[^/"])//([^/!]|$)' crates/ || true)
if [ -n "$line_comments" ]; then
  hit "line comments (//) in Rust sources:\n$line_comments" "remove the comment or make it a doc comment (/// or //!)"
fi

py_comments=$(grep -n '^\s*#' modal/bench.py || true)
if [ -n "$py_comments" ]; then
  hit "# comments in modal/bench.py:\n$py_comments" "keep only the module docstring; the launcher stays self-explanatory"
fi

unsafe_files=$(grep -rln --include='*.rs' -E 'unsafe[ \t({]' crates/ | grep -v 'crates/napkin-bench/src/cuda.rs' || true)
if [ -n "$unsafe_files" ]; then
  hit "unsafe outside crates/napkin-bench/src/cuda.rs:\n$unsafe_files" "move all CUDA FFI into cuda.rs"
fi

if grep -q 'cudarc' crates/napkin/Cargo.toml; then
  hit "crates/napkin depends on cudarc" "the estimator crate must stay CUDA-free; profile JSON types are the shared contract"
fi

allow_attrs=$(grep -rn '#\[allow(' --include='*.rs' crates/ || true)
if [ -n "$allow_attrs" ]; then
  hit "#[allow(...)] attributes:\n$allow_attrs" "use #[expect(lint, reason = \"...\")] so the waiver fails when it becomes unnecessary"
fi

bad_imports=$(grep -rn 'napkin::' --include='*.rs' crates/napkin-bench/src/ | grep -v 'napkin::profile::{Measured, Dtype}\|napkin::profile::{Dtype, Measured}\|napkin::profile::Measured\|napkin::profile::Dtype' || true)
if [ -n "$bad_imports" ]; then
  hit "napkin-bench imports beyond napkin::profile::{Measured, Dtype}:\n$bad_imports" "profile JSON types are the only shared contract between the crates"
fi

[ "$fail" -eq 0 ] && echo "lint-arch: clean"
exit "$fail"

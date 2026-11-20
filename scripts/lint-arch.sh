#!/bin/sh
# Custom architecture lints. Each violation prints what is wrong and how to
# fix it; exits 1 when anything is found.
cd "$(dirname "$0")/.." || exit 1
fail=0

hit() {
  printf 'Violation: %s\nFix: %s\n' "$1" "$2"
  fail=1
}

bad_tests=$(grep -rn --include='*.rs' -E '^[[:space:]]*(pub(\([[:alnum:]_]+\) )? )?fn[[:space:]]+test_' crates/ || true)
if [ -n "$bad_tests" ]; then
  hit "test functions prefixed with test_:\n$bad_tests" "name tests for the behavior they exercise without a test_ prefix"
fi

local_uses=$(find crates -type f -name '*.rs' -exec awk '
  FNR == 1 { last_item = "" }
  $0 ~ /^(pub(\([a-z]+\))? )?(fn|impl|mod|struct|enum|trait|const|static)([[:space:]]|$)/ {
    last_item = $0
  }
  $0 ~ /^ +use[[:space:]]/ {
    spaces = match($0, /[^ ]/) - 1
    if (spaces >= 8 || last_item !~ /^(pub(\([a-z]+\))? )?mod /) {
      print FILENAME ":" FNR ":" $0
    }
  }
' {} + || true)
if [ -n "$local_uses" ]; then
  hit "function-local use statements:\n$local_uses" "move imports to file scope; nested modules may keep their imports"
fi

py_comments=$(grep -n '^\s*#' modal/*.py || true)
if [ -n "$py_comments" ]; then
  hit "# comments in modal/*.py:\n$py_comments" "keep only the module docstring; the launchers stay self-explanatory"
fi

unsafe_files=$(grep -rln --include='*.rs' -E 'unsafe[ \t({]' crates/ | grep -v 'crates/napkin-bench/src/cuda.rs' || true)
if [ -n "$unsafe_files" ]; then
  hit "unsafe outside crates/napkin-bench/src/cuda.rs:\n$unsafe_files" "move all CUDA FFI into cuda.rs"
fi

# Reads the resolved dependency graph, so a cudarc pulled in through another
# crate counts too, not only a line in crates/napkin/Cargo.toml.
if ! napkin_deps=$(cargo tree --locked -p napkin -e normal,build --prefix none); then
  hit "cargo tree -p napkin failed" "fix the workspace manifest so the dependency graph resolves"
elif printf '%s\n' "$napkin_deps" | grep -q '^cudarc '; then
  hit "napkin depends on cudarc" "the estimator crate must stay CUDA-free; profile JSON types are the shared contract"
fi

# Any mention of napkin_profile counts, so `use napkin_profile as p;` cannot
# slip past the list of allowed names.
contract='Dtype|Measured|RunMode|GemmShape|GemmShapeSuite|GemmAlgo|GemmCheck|tflops|GpuState|RoundState|Span'
bad_imports=$(grep -rnE 'napkin_profile|napkin::' --include='*.rs' crates/napkin-bench/src/ | grep -vE "use napkin_profile::(\{($contract)(, ($contract))*\}|$contract);" || true)
if [ -n "$bad_imports" ]; then
  hit "napkin-bench imports beyond the profile measurement contract and the GPU-state types Measured contains:\n$bad_imports" "profile JSON types are the only shared contract between the crates"
fi

[ "$fail" -eq 0 ] && echo "lint-arch: clean"
exit "$fail"

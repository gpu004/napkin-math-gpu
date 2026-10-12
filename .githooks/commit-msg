#!/bin/sh
# Conventional commits: `type(scope): Capitalized summary`.
msg=$(head -1 "$1")
if ! printf '%s' "$msg" | grep -qE '^[a-z]+\([a-z0-9_-]+\): [A-Z]'; then
  echo "commit-msg: expected 'type(scope): Capitalized summary', got: $msg" >&2
  exit 1
fi

# Agent instructions

- The user keeps this repo under `jj`; agents may use plain `git` when told to.
- Run `sh scripts/check.sh` before pushing (fmt, clippy `-D warnings`, tests, lint-arch).
- Commits are conventional: `type(scope): Capitalized summary`.
- GPU runs happen only on Modal, never locally: `modal run modal/bench.py --gpu h100`.
- Canonical runs pin GPUs with `!` where needed (`H100!`); never commit `--quick` output.
- `napkin::profile` JSON types are the shared contract; `napkin-bench` may import only
  `napkin::profile::{Measured, Dtype}`.
- README number blocks are generated: edit tables via `napkin` then run
  `napkin readme --write README.md`. Never hand-edit inside `<!-- napkin ... -->` markers.

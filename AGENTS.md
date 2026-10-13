# Agent instructions

- Teach users how to derive an answer, and make the assumptions behind every number visible.
- The user keeps this repo under `jj`; agents may use plain `git` when told to.
- Commits are conventional: `type(scope): Capitalized summary`.
- GPU runs happen only on Modal, never locally: `modal run modal/bench.py --gpu h100`.
- Canonical runs pin GPUs with `!` where needed (`H100!`); never commit `--quick` output.
- `napkin-profile` is the shared profile contract; `napkin-bench` may import only
  `Measured`, `Dtype`, `GemmShape`, `GemmAlgo`, `GemmCheck`, `tflops`, and the
  GPU-state types `Measured` contains (`GpuState`, `RoundState`, `Span`).
- README.md and PROBLEMS.md number blocks are generated. Never hand-edit inside `<!-- napkin ... -->` markers; after changing
  profiles, table rendering, or anything a marker embeds, run `cargo run -p napkin -- readme --write README.md PROBLEMS.md`.
- PROBLEMS.md solutions use rounded teaching constants pinned in `crates/napkin/tests/readme.rs`. When a profile or price update moves one, that test fails: update the problem's arithmetic and the test together.
- GPU prices live in `prices/*.json`, never in hardware profiles. Every record needs a date and a source URL that you curl-checked; `napkin cost` never falls back to a default price.
- Agent skills live in `.agents/skills/` (`skills/modal/` predates it). `/create-verification-skill` and `/maintain-verification-skill` run on request only.

## Development guidelines

- All significant changes must be tested. Add or update focused tests for semantic changes when existing coverage does not already establish the intended behavior.
- Look to see if your tests could go in an existing file before adding a new file for your tests.
- Get your tests to pass. If you didn't run the tests, your code does not work.
- Follow existing code style. Check neighboring files for patterns.
- Prefer narrow visibility by default because this workspace is generally its own consumer (`unreachable_pub` is denied). However, do not add workarounds solely to avoid `pub`: make an item public when another workspace crate needs it and that produces the cleaner implementation.
- Rust imports should always go at the top of the file, never locally in functions.
- Run `uv run --only-dev --locked prek` at the end of a task if you changed files in the repo. This includes changes such as rebases or addressing review comments. Use `uv run --only-dev --locked prek run --files <path1> <path2>` and pass every file you changed. This keeps the hook run independent of staged state and avoids sweeping unrelated changes. Use `uv run --only-dev --locked prek run --all-files` when a full-repository hook sweep is specifically needed. Install the hooks once per clone with `uv run --only-dev --locked prek install`.
- Before writing significant amounts of new code, look for existing utilities or mechanisms that could solve the problem. Avoid expanding the task to unrelated issues, but do not confuse keeping the task focused with minimizing the size of the implementation. Prefer addressing the underlying architectural problem over adding a localized workaround, even when doing so requires a substantial refactor or rearchitecture. Ask the user for guidance if in doubt about whether to attempt a larger refactor or not.
- Try hard to avoid patterns that require `panic!`, `unreachable!`, `.unwrap()` or `.expect()`. Instead, try to encode those constraints in the type system. Don't be afraid to write code that's more verbose or requires largeish refactors if it enables you to avoid these unsafe calls.
- Prefer let chains (`if let` combined with `&&`) and let guards (`PAT if let ... =>`) over nested `if let` statements to reduce indentation and improve readability. At the end of a task, always check your work to see if you missed opportunities to use `let` chains or `let` guards.
- If you have to suppress a Clippy lint, prefer to use `#[expect()]` over `[allow()]`, where possible. But if a lint is complaining about unused/dead code, it's usually best to just delete the unused code.
- Don't use comments to narrate code, but do use them to explain invariants and why something unusual was done a particular way. Make sure that a comment will make sense to somebody who's reading the code for the first time. Prefer plain language, avoid jargon, and don't be afraid to be more verbose if it's necessary to explain something well. Giving a concrete example of the GPU workload or `napkin` invocation being modelled at that point can often be very helpful for future readers of the code.
- Don't prefix tests with `test_`.
- Don't separate struct definitions from their `impl` blocks unless the `impl` is deliberately placed in a separate file, as for large structs.

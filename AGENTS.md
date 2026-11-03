# Notes for coding agents

Read this file before you change anything in OpenWiSH.

## Rules

- The core rule of the project: an agent inside an environment computes weight edits from reasoning and code. No gradients, optimisers, fitting to data, search, or network execution outside declared probes. Every weight change goes through the Rust edit gate in `crates/openwish-tensor/src/gate.rs`.
- Treat every path the sandbox controls as untrusted on the host. Use `crates/openwish-core/src/paths.rs` (`file_under`, `dir_under`) for RPC files, notes and probe folders.
- Keep scoring assets out of the sandbox. List them under `hidden` in the task TOML. The spec loader refuses mounts over hidden paths.
- Write tests first. Watch each test fail, then make the test pass.
- Keep CPU use low. The owner does not want 100% CPU. Use `nice -n 10`, `cargo build -j 4`, `cargo test --workspace -j 4 -- --test-threads=2`.
- Stage files by path. Do not run `git add -A`. Other sessions sometimes share this checkout.
- Keep human interpretability tools in `interp/`. Code in `crates/`, `python/` and `tasks/` never imports from `interp/`.
- Report numbers from run folders only. Keep WSPBench condition A results apart from OpenWiSH sandbox results.

## Writing style for docs and user-facing text

Short sentences. Active voice. Address the reader as "you". No em dashes, no semicolons. `scripts/stylescan.py` holds the list of banned words. Check your prose with:

```bash
python3 scripts/stylescan.py README.md docs/*.md
```

## Checks before a commit

```bash
nice -n 10 cargo test --workspace -j 4 -- --test-threads=2
python3 tests/test_scorers.py
assets/venvs/sandbox/bin/python tests/test_adapters.py
python3 scripts/stylescan.py README.md docs/*.md CONTRIBUTING.md
```

End commit messages with a `Co-Authored-By:` line for the agent who wrote the change.

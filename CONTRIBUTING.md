# Contributing to OpenWiSH

Thanks for helping. This file covers the dev setup, the tests, the common contributions and the house style.

## Dev setup

```bash
git clone https://github.com/Alexander-Aghili/OpenWiSH && cd OpenWiSH
nice -n 10 cargo build --workspace -j 4
uv venv -p 3.12 assets/venvs/sandbox && VIRTUAL_ENV=assets/venvs/sandbox uv pip install numpy scipy scikit-learn pandas sympy networkx safetensors minatar tokenizers \
  && VIRTUAL_ENV=assets/venvs/sandbox uv pip install torch --index-url https://download.pytorch.org/whl/cpu
scripts/prepare_logs.sh && python3 scripts/prepare_tasks.py     # optional: published runs for the conformance tests
```

You need Linux with `bwrap` and `prlimit`. [Getting started](docs/getting-started.md) covers the other Python environments and assets.

## Run the tests

Keep the machine usable: limit cargo jobs and test threads.

```bash
nice -n 10 cargo test --workspace -j 4 -- --test-threads=2                 # Rust: unit, sandbox and CLI smoke tests
python3 tests/test_scorers.py                                   # scorers against the published grades
python3 tests/test_bench_report.py                              # audit status and replay labels
python3 tests/test_save_results.py                              # published replay evidence
python3 tests/test_bench_compare.py                             # per-task comparisons and validity
python3 tests/test_wspbench_format.py                           # public submission grammar
python3 tests/test_wspbench_validate.py                         # format-only finish feedback
assets/venvs/sandbox/bin/python tests/test_adapters.py          # MinAtar adapter
assets/venvs/kinetix/bin/python tests/test_kinetix.py           # Kinetix adapter
cargo test -p openwish-envs --release -- --ignored              # Go head-to-head, 2,000 games (slow)
```

Tests needing the sandbox venv skip without `assets/venvs/sandbox`. Tests needing the published runs skip without `assets/lm_understand-logs`. CI runs the first two commands on every pull request.

## Common contributions

### A new environment

1. Open a [new environment issue](https://github.com/Alexander-Aghili/OpenWiSH/issues/new?template=new_environment.yml) first, so the scorer and the hidden assets get agreed before code.
2. Follow [Environments](docs/environments.md#writing-an-environment): a folder in `tasks/`, a spec, a prompt, a scorer, and a prepare script in `scripts/` for large or hidden files.
3. Keep answer keys and evaluation data under `assets/` and list them in `hidden`.
4. Run the task with a scripted known-good answer and a known-bad answer. Add a scorer test in `tests/`, skipped when the assets are missing.

### A new backend

Follow [Agents](docs/agents.md#add-an-agent-or-a-provider). Switch off every tool the agent brings, add a parser test against a saved real response in `crates/openwish-llm/tests/fixtures/`, and run the toy task with the new backend.

### A new probe

1. Write the probe as a host command following the [probe contract](docs/environments.md#4-a-forward-probe): read `OPENWISH_REQ`, write files to `OPENWISH_OUT`, print one JSON line with `ok`.
2. Return `{"ok": false, "error": ...}` for a bad request, so the agent pays nothing for a typo.
3. Use a tag from `KNOWN_TOOLS` in `crates/openwish-core/src/protocol.rs`. A new tag needs a protocol change and a test.
4. Test the probe on its own with a request file, then through a task with a scripted reply.

## Code style

- **Tests first.** Write the failing test, then the fix. Every security fix ships with a test reproducing the attack.
- **Small modules.** One job per file. Keep the layer boundaries: `openwish-tensor` knows nothing of agents, `openwish-llm` knows nothing of weights.
- **rustfmt.** Format new code with `rustfmt`. CI does not check formatting, so keep formatting changes out of unrelated files.
- **No new trust in the sandbox.** Treat every path or file from `/work` as hostile. Use `paths::file_under` and `paths::dir_under`.
- **No optimisation paths.** No change lets an agent reach the weights through gradients, optimisers, fitting or search.

## Writing style

Docs, comments, error messages and agent-facing text follow the same rules.

- Short, direct sentences in the active voice. Address the reader as "you".
- No em dashes, no semicolons. Use periods or commas.
- Support claims with real numbers from the repository. Invent nothing.
- No filler, no setup phrases, no metaphors, no emoji.
- Name the subject instead of a pronoun: "The gate refuses the edit."
- Banned words include `can`, `may`, `just`, `that`, `it`, `very`, `really`, `actually`, `could`, `however`, `hence`, `powerful`, `boost`, and `harness` outside the name Open Weight-Space Harness. The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) repeats the checklist.

## Good first issues

| Issue | Where |
|---|---|
| Move the kernel protocol off the inherited stdout | `python/kernel/kernel.py`, `crates/openwish-sandbox` |
| Check cleanup after interrupted edit RPCs | `python/kernel/kernel.py` |
| Keep the reply of a one-reply task when an error follows | `crates/openwish-core/src/run.rs` |
| Make `restore` atomic | `crates/openwish-tensor/src/gate.rs` |
| Replace the fixed 120 s kernel start wait | `crates/openwish-sandbox` |
| Prompt-caching hints in the `api` backend | `crates/openwish-llm/src/anthropic_api.rs` |
| An OpenAI-compatible backend | `crates/openwish-llm` |
| More MinAtar games or Kinetix levels | `tasks/`, `python/adapters/` |

The [roadmap](docs/roadmap.md) lists the larger items.

## Pull requests

1. Branch from `main`.
2. Keep one change per pull request.
3. Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
4. Update the docs and [CHANGELOG.md](CHANGELOG.md) when behaviour changes.

By contributing, you agree to license your work under MIT OR Apache-2.0, as the rest of the project. Everyone follows the [Code of Conduct](CODE_OF_CONDUCT.md).

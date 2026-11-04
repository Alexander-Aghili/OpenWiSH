# Agents

OpenWiSH drives any agent through one trait. The run loop owns the turns, the budgets and the tools. A backend turns one message into one reply.

## Backends

| `--backend` | Agent | Default model | Session resume | Needs |
|---|---|---|---|---|
| `claude` (default) | `claude -p` with every built-in tool, MCP server, slash command and settings source off | `claude-opus-5-5` | yes, `--resume <session>` | the `claude` CLI, logged in |
| `codex` | `codex exec --json` with every tool feature off, read-only sandbox, no web search | the Codex default | yes, `codex exec resume <session>` | the `codex` CLI, logged in |
| `api` | Anthropic Messages API over raw HTTP, adaptive thinking, the full message list kept by OpenWiSH | `claude-opus-5-5` | through a handoff | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` |
| `openai` | Any OpenAI-compatible chat API: OpenAI, OpenRouter, LM Studio, llama.cpp, vLLM, Ollama. The full message list kept by OpenWiSH | none, `--model` is required | through a handoff | `OPENWISH_OPENAI_BASE_URL`, and a key in `OPENWISH_OPENAI_API_KEY` or `OPENAI_API_KEY` for hosted providers |
| `scripted:<file>` | Fixed replies, separated by lines of exactly `=====` | none | yes, at the recorded position | the file |

The Codex backend forwards task instructions through `developer_instructions`, uses a neutral base instruction file, and disables project-document loading and memories. New sessions and resumed sessions receive the same task instructions. The instruction file path is absolute, including when your run folder is relative.

The CLI backends run in a dedicated folder, `<run>/editor_cwd/`, so the agent sees no project files. The agent works only through the tool blocks the task enables.

Flags for every backend:

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | the task's `[editor] backend`, else `claude` | Which backend |
| `--model` | the task's `[editor] model`, else per backend | Model id passed to the backend |
| `--effort` | the task's `[editor] effort` | `low`, `medium`, `high`, `xhigh` or `max` |
| `--max-tokens` | 64000 | Output tokens per reply (`api`, `openai`) |
| `--timeout` | 7200 | Seconds to wait for one reply |

You set shared backend defaults in `openwish.toml`. Flags override task defaults, which override project defaults. Use [hooks](configuration.md#hooks) to inspect or block tool calls. Use `openwish tools <task.toml>` to inspect the agent surface before a run.

`openwish audit` takes the same flags for the reviewer model. Audit effort defaults to `medium`.

```bash
openwish run tasks/exp3/net-a.toml                                    # claude -p, claude-opus-5-5
openwish run tasks/exp6/budget_10.toml --backend codex --model <model> --effort xhigh
openwish run tasks/exp4/task.toml --backend api --model claude-opus-5-5
openwish run tasks/toy/task.toml --backend scripted:tasks/toy/script.txt
OPENWISH_OPENAI_BASE_URL=http://localhost:1234/v1 openwish run tasks/exp1/a9f535cc.toml --backend openai --model <local model>
OPENWISH_OPENAI_BASE_URL=https://openrouter.ai/api/v1 openwish run tasks/decompile/d2-a-native.toml --backend openai --model <provider/model>
```

**Open and local models.** The `openai` backend sends `--effort` as `reasoning_effort`. Set `OPENWISH_OPENAI_NO_EFFORT=1` when your server refuses the field. A local model needs a context window large enough for the prompt plus its reasoning. The Experiment 1 prompts hold 17,253 weights, and a reasoning model thinks for tens of thousands of tokens and close to an hour per reply on a laptop, so raise the server's context length and timeout, and pass a matching `--max-tokens` and `--timeout`. The `native` decompile tier fits much smaller contexts: its prompts are 5.5 to 7 KB.

## What the agent sees

1. The first message: the task prompt, verbatim, plus the appendix if the task has one.
2. After each tool call: the output, any static-check flags, and the budget line.
3. After a protocol error: what went wrong, and a note saying nothing ran and nothing was charged.
4. After a handoff: the task again, the budget line, `/work/notes.md`, the ledger summary and the pending message.

The system text is one neutral line, `Answer the user's request.`, unless the task sets `instructions`. The reply protocol is in [Architecture](architecture.md#reply-protocol).

## The `Editor` trait

```rust
pub trait Editor: Send {
    fn name(&self) -> String;
    /// Send one message. `new_conversation` starts a fresh thread (first turn, or a handoff).
    fn send(&mut self, msg: &str, new_conversation: bool) -> Result<Reply, BackendError>;
    /// The conversation id to persist for crash-resume.
    fn session(&self) -> Option<String>;
    fn set_session(&mut self, s: Option<String>);
    /// Whether a saved session id is enough to continue the conversation after a restart.
    fn can_resume(&self) -> bool { true }
}
```

`Reply` carries the text, token usage (`input`, `output`, `cache_read`, `cache_write`, `reasoning`), the session id, cost in USD when the provider reports cost, wall time, and the raw provider output. OpenWiSH writes the raw output to `turns/tNNN/raw.json`.

Return errors as one of three kinds. The run loop decides what to do:

| `BackendError` | Run loop |
|---|---|
| `Refused(msg)` | Tells the agent the provider refused, keeps the task, ends the run after four in a row |
| `Transient(msg)` | Retries with backoff from 10 s, doubling, up to 8 attempts, then stops for `--resume` |
| `Fatal(msg)` | Stops the run with the message |

`openwish_llm::classify(msg)` maps a provider error string to a kind. `openwish_llm::run_cli` runs a CLI with a timeout and returns stdout, stderr and the exit code.

## Add an agent or a provider

1. Add a module in `crates/openwish-llm/src/`, for example `my_backend.rs`, and declare the module in `lib.rs`.
2. Implement `Editor`. Build the request from `BackendConfig`: `model`, `effort`, `instructions`, `workdir`, `max_tokens`, `timeout_s`.
3. Switch off every tool the agent brings. The agent must reach the network only through OpenWiSH tool blocks.
4. Register a kind string in `openwish_llm::make` and in the error message listing the kinds.
5. Save one real provider response under `crates/openwish-llm/tests/fixtures/` and test the parser against the fixture, as `claude_cli.rs` and `codex_cli.rs` do.
6. Run the toy task with the new backend, then document the backend in this file.

A backend without its own conversation store returns `false` from `can_resume`. OpenWiSH then resumes a crashed run through a handoff, which restates the task.

## Add a model

Pass the model id with `--model`, or set `[editor] model` in a task. The `claude` and `api` backends take Anthropic model ids. The `codex` backend takes any model the Codex CLI accepts.

OpenAI-compatible and local endpoints (vLLM, Ollama) need a new backend. The [roadmap](roadmap.md) lists the backend, and an MCP server for agents with native tool calls.

## Replay a published run

`scripts/replay_authors.py` turns a published run folder into a scripted file. Conformance checks use this path, so a replay exercises every layer: the published code runs in the OpenWiSH kernel, the published edits pass the OpenWiSH gate, and the OpenWiSH scorer grades the answer.

```bash
python3 scripts/replay_authors.py assets/lm_understand-logs/exp1_which_game/runs/breakout breakout.txt
openwish run tasks/exp1/a9f535cc.toml --backend scripted:breakout.txt
```

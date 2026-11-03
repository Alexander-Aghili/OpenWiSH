# Roadmap

The target: the instruments of a leading coding agent, applied to weights. This table tracks where OpenWiSH stands today.

## Parity with coding agents

| Capability | OpenWiSH today | Next | State |
|---|---|---|---|
| Any model | One `Editor` trait: `claude -p`, `codex exec`, Anthropic API, scripted replay | OpenAI-compatible and local (vLLM, Ollama) endpoints | have |
| Any agent | Agents speak the text block protocol | An MCP server exposing the Read and Write tools, so Claude Code, Codex or any MCP agent drives OpenWiSH with native tool calls | partial |
| Long sessions and context | Handoff by context size with notebook, ledger summary and budgets, crash-resume | Rolling compaction, per-tool output summaries | have |
| Permissions and safety | Sandbox, edit policy, hidden-asset checks, static checks, auditor, hooks on tool calls | Policy profiles | have |
| Version control of changes | Ledger, checkpoints, restore, portable edit sets with hashes | Branches and merges of edit sets, a composition check between edits | partial |
| Evaluation | Per-environment scorers, replays of published runs, audit status | A suite runner with repeats and confidence intervals, leaderboards | partial |
| Parallel work | Parallel game play and episode rollouts inside tools | Subagents (for example one reader per layer), parallel runs | next |
| Large models | `.npy` and `.npz`, f32, f16 and bf16 reads, CPU sandbox, a GPU worker for `tasks/exp8` (local or over SSH) | safetensors with memory mapping, GPU forward servers | next |
| Interactive use | CLI progress, `status`, text and HTML `replay` | A TUI with live budget, ledger and diff views | next |

## Benchmarks

- A baseline mode replaying the original protocol (verbatim prompts, one block per turn, fixed 40-turn handoffs, no notebook, no checkpoints), for the comparison in [Benchmarks](benchmarks.md#live-agent-runs).
- Live agent results and interpretability baselines in [Benchmarks](benchmarks.md).

## Candidate environments

| Group | Candidates | Measures |
|---|---|---|
| Known ground truth | Tracr-compiled transformers, InterpBench, AdderBoard | Reading against a known circuit |
| Model zoos | Schürholt et al. 2021, Unterthiner et al. 2020, NWS, Implicit-Zoo, LoRA-WS | Predicting accuracy or hyperparameters from weights |
| Editing with locality | CounterFact, zsRE | Edits which change one fact and keep the rest |
| Auditing | TrojAI backdoored models | Finding a backdoor from weights |
| Control | More Kinetix levels, all MinAtar games | Fixing and writing policies |

## Deferred fixes

The whole-branch review left seven minor items. Each one fits a first contribution. See [Safety](safety.md#deferred-review-items).

1. Move the kernel protocol off the inherited stdout.
2. Check cleanup after interrupted edit RPCs. Refused edits now remove the temporary blob.
3. Keep the reply of a one-reply task when an error follows.
4. Tighten the whole-tensor-scale heuristic (`whole_tensor_constant` in `crates/openwish-tensor/src/gate.rs`).
5. Send prompt-caching hints from the `api` backend.
6. Make `restore` atomic.
7. Replace the fixed 120 s kernel start wait.

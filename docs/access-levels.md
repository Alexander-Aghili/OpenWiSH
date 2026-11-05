# Access levels: what each OpenWiSH result is evidence of

## Why this page exists

Matthew Honnibal's critique of *Sparks of Weight-Space Programming*
([honnibal.dev, 2026-11-03](https://honnibal.dev/blog/astra-read-write-misleading)) says the post presents several
techniques as one. The claim "from the raw weights alone" holds for only some experiments. Experiments 1, 2 and 4
read raw weights of small networks, with no code and no runs of the network. Every experiment with a real language
model uses Python. The headline Qwen result (MATH-500 from 7.9 % to 48.1 %) also ran the model on up to 50 batches
of practice problems. Experiment 4's prompt contains a recorded failure episode. Experiment 5 starts from zero
weights, so the agent reads nothing, and its policy beats the PPO baseline on Breakout but not on Seaquest or
Asterix. Experiment 7a failed. 7b succeeded only with attention patterns. The Qwen method failed on Gemma 3 270M.
Each result is real. Pooled together, the results suggest one deep capability, reading any network's weights and
understanding the network, which the evidence does not show.

Native reading also works outside one frontier model. Simon Willison
([simonwillison.net, 2026-11-03](https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/)) ran
Experiment 1 with no tools and one reply on open models. Kimi K3.1 Pro (through OpenRouter) named Breakout. Qwen 3.8
27B, quantised on a laptop, named three of the four games (Pac-Man, Pong, Breakout) and missed Freeway. Each local
reply took 46 to 65 minutes and tens of thousands of reasoning tokens. Willison adds a caveat: a patient person with
pencil and paper would likely reach the same answers. Native reading of small networks is a real capability, and a
narrow one. You run these models in OpenWiSH through `--backend openai` (see [Agents](agents.md)). OpenWiSH records
reasoning tokens and wall time for every reply (`turns/tNNN/run.json`), because the cost is part of the result.
These are Willison's numbers, not OpenWiSH runs.

OpenWiSH pursues every access level. Each level measures a real and useful capability. The levels measure different
capabilities, so OpenWiSH makes the access level part of every task and every result.

## The channels

| Channel | Values | What success through this channel shows | Why OpenWiSH offers the channel |
|---|---|---|---|
| **Weights** | `prompt` (as text in the first message), `code` (loaded in the sandbox), `both`, `zero` (write from zero weights, nothing to read), `none` | `prompt` with nothing else: the model understands the network from its weights alone (native reading). `code`: the model analyses weights as an engineer would. `zero`: the model designs a network, without reading one. | Native reading is the cleanest scientific claim, and only small networks fit in a prompt. Code is the only way into large networks, and the reason a coding tool for weights exists. |
| **Code** | none, or Python with N runs (fresh or persistent) | The agent writes analyses over weights (statistics, linear algebra, search over *weights*, never over network outputs). Success here mixes understanding with engineering. | Large networks, reproducible analysis, parity with coding agents. |
| **Network runs** | none, or budgeted probes: outputs, activations, interventions (unit or subspace patches), games, generations | Empirical evidence: the agent designs experiments. A result built on runs is not a weight-reading result. | Interpretability practice is experimental. Budgets keep the dependence measurable. |
| **Instruments** | none, or precomputed host arrays (for example the J-lens: per-layer average Jacobians computed with gradients before the run, on a named public corpus) | The agent uses causal summaries computed by the host. Using them is linear algebra on fixed arrays. The gradients belong to the host, like scoring. | Interpretability tools need gradients, and agents never run them. A declared instrument with its corpus named keeps the boundary visible. |
| **Shown** | behaviour evidence already in the prompt (for example a sampled failure episode, success statistics) | Nothing by itself. Any result with shown behaviour is partly behavioural. | Some tasks ask the agent to fix a known failure, so the prompt shows the failure. |
| **Data** | anything mounted besides the network (for example practice problems with answers) | The agent builds its own evaluations. | Realistic repair tasks come with data. |
| **Replies** | one reply, or multi-turn up to N | One reply: no feedback loop at all. Multi-turn: iteration. | Engineers iterate. One reply isolates prior knowledge. |
| **Optimisation** | never for the agent | Forbidden: gradients, optimisers, fitting weights or programs to data, search over network outputs. | Without this rule every task measures training, not understanding. |
| **Computation outside the run** | training networks, generating tasks, calibration, opponents, scoring | Nothing about the agent. | Building environments. Agents never see this work. |
| **People's tools** (`interp/`) | the MLP explorer, neuron-explorer | Nothing about the agent. | For people inspecting networks and runs, never for agents. An environment wanting such a view declares the view as a budgeted probe. |

## How OpenWiSH enforces the labels

- Every task spec has an `[access]` section with `weights`, plus any `shown` or `data`. OpenWiSH derives the rest
  (code, network runs, replies) from the spec. A test fails when a bundled task lacks the section.
- OpenWiSH writes the label into the run log at creation, prints the label in `openwish status`, and stores the
  label with the score in `eval/score.json` under `access`. An example:
  `weights as text in the prompt · no code · network never run · shown: one sampled failure episode, every 16 steps · one reply`.
- The edit gate, the sandbox guards, the static checks and the auditor (see the README) enforce the optimisation
  rule and the network-runs budget, and report flagged forward passes in code.

## Reporting rules

1. **Report every score with its access label.** Call a result "reading weights" only when the label says weights
   in the prompt, no code, network never run, nothing shown.
2. **Never pool across access levels.** No averages, headline sentences or abstracts over results from different
   levels. Report each level as its own claim.
3. **Use ladders to attribute.** Run the same network at several levels and report the differences: the decompile
   environments (`native`, `read`, `probe`), Experiment 7 (`weights_only` against `attention`), and the pass budgets
   of Experiments 6 and 8 (0 / 10 / 25 / 50).
4. **Match baselines to the access level.** Compare a probe-tier result with a black-box baseline given the same
   probes and no weights. Compare a write-from-zero result with standard training at stated budgets, wins and
   losses alike.
5. **Report channel use per run**: Python runs, passes and probe inputs used, and static-check flags.

## The bundled environments

| Environment | Access (the OpenWiSH label) | A success is evidence of |
|---|---|---|
| `exp1` which game | weights in prompt · no code · never run · one reply | native reading |
| `exp2` which shape (diffusion) | weights in prompt · no code · never run · one reply | native reading |
| `exp3` which corpus (char transformer) | weights through code · Python, 100 fresh runs · never run | analysis with code |
| `exp4` fix the Kinetix car | weights in prompt · no code · never run · **shown: 256-episode statistics and one sampled failure episode** · one reply | reading plus a behaviour trace, not pure reading |
| `exp5` MinAtar policies | from zero · no code · never run · one reply | designing a network from a game's rules |
| `exp6` 9x9 Go policy | from zero · Python · games + positions probes (0-50 passes) · multi-turn | design with code and play-testing (budget ladder) |
| `exp7` which Gemma is sycophantic | weights through code · Python (numpy) · attention probe drawn from the Python runs, or none (`weights_only`) | analysis with code, and with probes partly empirical |
| `exp8` Qwen math repair | weights through code · Python · 10, 25 or 50 forward passes · **data: practice problems** · multi-turn | engineering with code, experiments and data |
| `exp1-perm`, `exp2-perm` | weights in prompt · no code · never run · one reply | native reading on weights no model has seen in this layout (same function, hidden units permuted) |
| `decompile/*-native` | weights in prompt · no code · never run · one reply | native reading of a whole mechanism |
| `decompile/*-read` | weights through code · Python, 60 runs · never run | mechanistic analysis with code |
| `decompile/*-probe` | weights through code · Python · 3 passes of 16 inputs (with patches) | mechanistic analysis with a small experimental budget |
| `wspbench/*-A` | task JSON with the weights in the prompt · no code · never run · one reply | native reading (WSPBench condition A) |
| `wspbench*/*-OW` | weights in the prompt and through code · Python, 30 runs · never run · format-only checker | analysis with code, a separate condition from WSPBench A, B and C |
| `agentic_interp/*-A` | no weights · no code · never run · one reply | prior knowledge of circuits from the task text and examples |
| `agentic_interp/*-WO`, `*-P` | weights through code · Python · never run (WO) or a `forward` probe with ablation and patching (P) | analysis with code, and with the probe the access of the published HyVE agent |
| `moe/unlearn_python` | weights through code · Python, 40 runs · 12 forward probes · **data: practice texts** · multi-turn | engineering with code, experiments and data, on a mixture-of-experts model |
| `demo`, `toy` | weights through code · Python | smoke tests only |

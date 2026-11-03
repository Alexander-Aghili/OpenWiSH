# OpenWiSH results: tools against tools

*Sparks of Weight-Space Programming* reached the top of Hacker News twice in two days: the post itself ([2,912 points, 876 comments](https://news.ycombinator.com/item?id=50341174)) and Matthew Honnibal's critique ([801 points, 185 comments](https://news.ycombinator.com/item?id=50351111)). This page tests OpenWiSH against the points those threads raised. Every number comes from a run folder in [`results/`](../results). Every table names the access level of each row.

## What the discussion asked for

| Point raised | Where | What OpenWiSH does |
|---|---|---|
| The post mixes three skills: reading raw weights, writing Python, and running the network. The Qwen result needs the last two. | Honnibal's post, top comments in both threads | Every task declares its access. Every row below names its level. Levels never pool. |
| The Qwen gain removes reasoning loops with a known direction ablation. Steering and Arditi et al. (2024) do the same. | top comment of the critique thread, a mech-interp researcher | OpenWiSH reports Experiment 8 as a replay of the published edits, not as a new skill. |
| Weight editing and weight-space learning have years of prior work: ROME, hypernetworks, Tracr, model zoos. | top comment of the post thread | OpenWiSH is an instrument, not a method. The benchmarks below come from this prior work. |
| A model trained on public weights or interpretability papers passes tests on public models. | many comments in the post thread | WSPBench and AgenticInterpBench use synthetic networks with known ground truth. The tuned OpenWiSH config ran once on tasks no tuning round saw. |
| Replications differ: some readers reproduced the games, others saw no gain on their own networks. | post thread | Same model, same task, one attempt, published transcripts, a prompt-only control, fixed baselines, and an auditor on every code run. |
| "Astra is surprisingly capable of doing old school mechanistic interpretability research." | Neel Nanda, quoted in the critique thread | OpenWiSH gives agents the instruments for this work: a sandbox over the weights, budgeted probes, a gated edit path and an audit trail. |

## Headline: WSPBench, tools against tools

[WSPBench](https://github.com/alexcarterhq/wspbench) defines two tool conditions of its own. In condition B the model calls bounded weight-algebra tools. Condition C adds metered practice observations of the network. The OpenWiSH sandbox gives Python over the same weights, with no network execution. All three give the model tools. The format and the structure differ.

Same models, same 12 tasks (the 6 pilot tasks and 6 held-out tasks from the same families), one attempt each:

<p align="center"><img src="assets/charts/wspbench-tools.svg" alt="Grouped bars: exact solves out of 12 WSPBench tasks. WSPBench B and C score 0 or 1 for every model. The OpenWiSH sandbox scores 12 for gpt-6.1-sol, gpt-6.1-terra and claude-opus-5-5, and 4 for gpt-6-luna." width="760"></p>

| Model | WSPBench B, exact | WSPBench C, exact | OpenWiSH sandbox, exact | OpenWiSH audits |
|---|---:|---:|---:|---|
| gpt-6.1-sol | 1 / 12 | 0 / 12 | **12 / 12** | 12 / 12 valid |
| gpt-6.1-terra | 0 / 12 | 0 / 12 | **12 / 12** | 12 / 12 valid |
| gpt-6-luna | 0 / 12 | 0 / 12 | **4 / 12** | 10 / 11 valid, 1 invalid |
| claude-opus-5-5 | not run | not run | **12 / 12** | 12 / 12 valid |

OpenWiSH solves more tasks exactly than both WSPBench tool conditions, for each of the three models run in all three, on both task sets.

How to read this table:

- **Held out.** Codex tuned the OpenWiSH config in five rounds on the 6 pilot tasks, then froze the config. On the 6 held-out tasks, Sol, Terra and Claude each solve 6 of 6.
- **Same tool loop.** B and C run in WSPBench's own Inspect loop, with WSPBench's solver, tools, budgets and scorer, unchanged. Only the transport differs from the pilot: each turn goes through the backend CLI, not a provider API. OpenWiSH does not charge B and C for its own transport overhead. The overhead is a measured 4,731 to 4,737 input tokens per request, the same on all six pilot tasks.
- **Budgets differ.** WSPBench caps each session at 160,000 input tokens, 16 turns and 12 analysis calls. The task payload alone holds 33,000 to 66,000 tokens, so B and C sessions get 2 to 4 turns. Sol spends those turns on analysis and runs out before submitting. Terra and Luna submit early, and none of their answers is exact. Budget design forms part of the structure being compared.
- **Audits.** A run with a failed audit scores 0. One Luna run tested gate inputs against weighted thresholds. The static guard passed the code, and the auditor caught the violation.

<p align="center"><img src="assets/charts/wspbench-rounds.svg" alt="Lines: audited read agreement over five tuning rounds. All three models rise after round 4 added the activation guard. Sol and Terra reach 100 percent, Luna 69 percent." width="760"></p>

The audit changes the story of the tuning rounds. Before the round 4 guard, most high read scores came from code evaluating gates on inputs. Those runs fail audit and score 0 above.

## Where OpenWiSH does not win yet

- **WSPBench forecast and repair families (12 tasks).** One reply already solves all 12 for Sol. In the sandbox, Terra and Luna each failed three audits. The forbidden route, simulating the network at sample inputs, tempts the model when Python sits next to the weights.
- **AgenticInterpBench (84 circuits, 163 components).** Tag accuracy on the 5-class taxonomy:

  | Model | Prompt only, one reply | Weights only | Weights and forward probe | Published HyVE agent |
  |---|---:|---:|---:|---:|
  | gpt-6.1-sol | 73.0 % | 73.0 % | 71.8 % | n/a |
  | claude-sonnet-4-6 | 69.9 % | not run | running | 79.0 % |


  <p align="center"><img src="assets/charts/agentic-interp.svg" alt="Dot plot: AgenticInterpBench tag accuracy. Published HyVE agents score 67 to 79 percent. OpenWiSH rows score 67 to 73 percent. Fixed baselines sit at 44.2 and 63.8 percent." width="760"></p>

  A fixed rule (MLP means MAPPER, attention means AGGREGATOR) scores 63.8 %. Tools add nothing for Sol here. HyVE adds 9 points over a prompt-only Sonnet reply. The like-for-like run, OpenWiSH with Sonnet 4.6 and HyVE's access, is in progress. The paper reports no prompt-only control. This page adds one.
- **MinAtar Breakout.** A Claude one-shot policy scores 54.2 against the published 517.4.

## Blog environments with Claude Opus 5.5

Single runs, scored by the OpenWiSH scorers. The published agent is GPT-6 Astra, a different model.

<p align="center"><img src="assets/charts/blog-envs.svg" alt="Four panels: Kinetix ties at 1,024 successes. Claude through OpenWiSH leads on Seaquest and Asterix, and trails on Breakout, 54 against 517." width="760"></p>

| Environment | Access | Published | Claude Opus 5.5 through OpenWiSH | Audit |
|---|---|---|---|---|
| Game-policy MLPs | weights in the prompt | 4 / 4 | 4 / 4 | no code run |
| Diffusion MLPs | weights in the prompt | 3 / 3 | 3 / 3 | no code run |
| Kinetix car, fix a failing policy | one reply with edits | 1,024 / 1,024, 6,073 weights changed | 1,024 / 1,024, 2,921 weights changed | no code run |
| MinAtar Seaquest / Asterix, write from zero | one reply with edits | 53.1 / 53.2 | **60.9 / 60.4** | no code run |
| MinAtar Breakout | one reply with edits | 517.4 | 54.2 | no code run |
| Char-level LM, name the corpus and copying heads | Python on the weights | python, recall 1.0 | python, recall 1.0, precision 0.6 | invalid, see below |
| Gemma pair 1, weights only (a published failure, pair rebuilt from the published script) | Python on the weights | wrong (A, 50 %) | right (B, 70 %) | invalid, see below |

Both invalid audits deserve a look. The Gemma run multiplied gate and up projections of embedding-derived vectors, and the task rules forbid gates. The char-LM run fitted a permutation of weight-derived statistics with annealing. The task rules allow fitting. The auditor brief bans search for every task and overrode them. Both runs stay invalid in the tables. [Safety](safety.md) lists the conflict as a known gap.

<p align="center"><img src="assets/charts/qwen-math.svg" alt="Grouped bars: Qwen MATH-500 accuracy, published against the OpenWiSH replay, for base, 10, 25 and 50 passes. The replay matches within one standard error band. A second panel shows the share cut at the token limit falling from 90 to 32 percent." width="760"></p>

Experiment 8 replays the published Qwen3.5-0.8B edits on MATH-500 with one sample per problem: base 10.6 %, 10 passes 13.6 %, 25 passes 14.2 %, 50 passes 51.8 % (published 7.9, 12.3, 14.1 and 48.1 % over four samples). The share of answers cut at the 32,768-token limit falls from 90.2 % to 31.6 %. The gain comes from ending reasoning loops, as the critique thread noted.

## What OpenWiSH is for

OpenWiSH aims to be a shared instrument for mechanistic interpretability research and for decoding neural networks into programs. Plug in any agent. Give the agent the weights, the probes your task allows, and one gated path for edits. Get back a hashed ledger, every code run, and an audit of each run against your rules.

Next on the list: the full like-for-like AgenticInterpBench run, Tracr decompiling (weights in, RASP out, against a trained decompiler with 30 % exact programs), InterpBench circuit localisation, and the MIB leaderboard. Environments are TOML files and a scorer. [Contributing](../CONTRIBUTING.md) explains how to add one.

## Reproduce

```bash
scripts/prepare_wspbench.sh                     # WSPBench tasks, condition A and OW
assets/wspbench/.venv/bin/python -I scripts/wspbench_bc.py --backend codex --model gpt-6.1-sol \
    --condition B --task 136a89846d8b907e --label wspbench-B-inspect-gpt-6.1-sol --overhead 4735
scripts/prepare_agentic_interp.sh               # AgenticInterpBench, conditions A, WO and P
python3 scripts/bench_report.py                 # every table on this page, from the run folders
python3 scripts/bench_charts.py                 # every chart, and docs/charts.html
```

Manifests with task and binary hashes: [`results/heldout-r5-manifest.json`](../results/heldout-r5-manifest.json), [`results/heldout-families-manifest.json`](../results/heldout-families-manifest.json), [`results/agentic-interp-manifest.json`](../results/agentic-interp-manifest.json).

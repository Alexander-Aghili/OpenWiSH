# OpenWiSH results: tools against tools

*Sparks of Weight-Space Programming* reached the top of Hacker News twice in two days: the post itself ([2,912 points, 876 comments](https://news.ycombinator.com/item?id=50341174)) and Matthew Honnibal's critique ([801 points, 185 comments](https://news.ycombinator.com/item?id=50351111)). This page tests OpenWiSH against the points those threads raised. Every number comes from a run folder in [`results/`](../results). Every table names the access level of each row.

**Weights only or with probes.** *Weights only* means the agent reads the weights, as text or through Python, and the network never runs. *With probes* means the agent also ran the network within a budget (forward passes, activations, practice observations). In charts, weights-only results are solid bars and filled dots, and results with probes are hatched bars and hollow dots. See [Access levels](access-levels.md).

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
  | claude-sonnet-4-6 | 69.9 % | not run | 61.3 % (68.7 % before audit) | 79.0 % |


  <p align="center"><img src="assets/charts/agentic-interp.svg" alt="Dot plot: AgenticInterpBench tag accuracy. Published HyVE agents score 67 to 79 percent. OpenWiSH rows score 67 to 73 percent. Fixed baselines sit at 44.2 and 63.8 percent." width="760"></p>

  A fixed rule (MLP means MAPPER, attention means AGGREGATOR) scores 63.8 %. With the same model and the same access, the published HyVE agent beats OpenWiSH by 17.7 points, 79.0 % against 61.3 %. The transcripts show why. Sonnet answered 59 of 84 cases after one Python run, with a median of two turns and 38 probe calls across all 84 cases. Before audit, its 68.7 % matches its own prompt-only reply. HyVE runs a fixed observe, hypothesise and validate loop for every component. Six OpenWiSH runs computed attention in Python instead of calling the probe, and failed audit. The lesson for OpenWiSH: open-ended access does not make an agent investigate. A per-component protocol does. The paper reports no prompt-only control. This page adds one.
- **A per-component protocol on AgenticInterpBench.** HyVE's edge looked like investigation depth, so OpenWiSH gained a protocol condition: observe, hypothesise and test each component, with a finish check: each component's test must be a finished probe call ablating or patching the component. Claude Sonnet 4.6, pre-registered 16 dev cases (seed 2026), then 68 held-out cases. Version 1 matched prompt only on dev (59.5 %). Version 2 added a test for telling tags apart (count an MLP's upstream sources, read a head's attention shape) and reached 67.6 % on dev. On the 64 held-out cases with a score (4 ended on a lost login), the frozen version 2 scores 55.0 %, against 71.7 % prompt only and 62.5 % for open probing. The added test over-counts sources: 35 of its 54 errors read a MAPPER as a COMBINER. A gain on 37 dev components did not hold, and the protocol stays out of the headline. The audits of these runs also exposed an auditor flaw, now fixed: the auditor never saw probe calls, so answers citing probe tests read as citing tests never run.
- **MinAtar Breakout.** Breakout at the task's own effort (xhigh) scores 476.3, 261.5 and 513.5 in three runs, a mean of 417 ± 79 (standard error over runs), against the published 517.4. An earlier single run at high effort scored 54.2. The gap closed from tenfold to 0.8 times the published score once the run used the task's effort.

## Blog environments with Claude Opus 5.5

Single runs, scored by the OpenWiSH scorers. The published agent is GPT-6 Astra, a different model.

<p align="center"><img src="assets/charts/blog-envs.svg" alt="Four panels: Kinetix ties at 1,024 successes. Claude through OpenWiSH leads on Seaquest and Asterix, and trails on Breakout, 417 (mean of 3 runs) against 517." width="760"></p>

| Environment | Access | Published | Claude Opus 5.5 through OpenWiSH | Audit |
|---|---|---|---|---|
| Game-policy MLPs | weights in the prompt | 4 / 4 | 4 / 4 | no code run |
| Diffusion MLPs | weights in the prompt | 3 / 3 | 3 / 3 | no code run |
| Kinetix car, fix a failing policy | one reply with edits | 1,024 / 1,024, 6,073 weights changed | 1,024 / 1,024, 2,921 weights changed | no code run |
| MinAtar Seaquest / Asterix, write from zero | one reply with edits | 53.1 / 53.2 | **60.9 / 60.4** | no code run |
| MinAtar Breakout | one reply with edits | 517.4 | 417 ± 79 (3 runs at xhigh: 476.3, 261.5, 513.5) | no code run |
| Char-level LM, name the corpus and copying heads | Python on the weights | python, recall 1.0 | python, recall 1.0, precision 0.6 | valid on re-audit, see below |
| Gemma pair 1, weights only (a published failure, pair rebuilt from the published script) | Python on the weights | wrong (A, 50 %) | right (B, 70 %) | invalid, see below |

The Gemma run multiplied gate and up projections of embedding-derived vectors, and the task rules forbid gates, so the run stays invalid. The char-LM run fitted a permutation of weight-derived statistics with annealing. The task rules allow fitting, but the first auditor brief banned search for every task and overrode them. The brief now applies each task's rules first. A full re-audit with the same auditor (`gpt-6.1-sol`, medium effort) found 43 of 43 code runs allowed, so the run is valid. The first verdicts stay in the run folder as `audit-brief-v1.jsonl`.

<p align="center"><img src="assets/charts/qwen-math.svg" alt="Grouped bars: Qwen MATH-500 accuracy, published against the OpenWiSH replay, for base, 10, 25 and 50 passes. The replay matches within one standard error band. A second panel shows the share cut at the token limit falling from 90 to 32 percent." width="760"></p>

Experiment 8 replays the published Qwen3.5-0.8B edits on MATH-500 with one sample per problem: base 10.6 %, 10 passes 13.6 %, 25 passes 14.2 %, 50 passes 51.8 % (published 7.9, 12.3, 14.1 and 48.1 % over four samples). The share of answers cut at the 32,768-token limit falls from 90.2 % to 31.6 %. The gain comes from ending reasoning loops, as the critique thread noted.

## A mixture-of-experts model

`tasks/moe` asks an agent to remove Python skill from Granite-3.1-1B-A400M (24 layers, 32 experts per layer, 8 active per token) while keeping English. The agent edits router and expert output weights through the gate, capped at 64 experts' worth of entries, with 40 Python runs and 12 forward probes on CPU. The probes return losses and the share of tokens routed to each expert. Fixed baselines get the same cap: 64 random experts, or the 64 experts the router sends code to most often on the practice texts.

**The first score failed.** Version 1 scored the rise in Python loss minus the rise in English loss. `gpt-6.1-sol` multiplied 12 code-selective experts by -8 and replaced 7 with a rank-one matrix 65,536 times one token direction. Python loss went from 1.58 to 122.7 and English from 2.83 to 20.1: the model was broken on everything, code a bit more. The score read 103.8. The audit found every code run within the rules. The metric was wrong, not the agent.

**Version 2** gives 0 when English loss rises by more than 0.1 nats per token, and counts the Python rise only up to a uniform guess over the vocabulary (10.80). The 0.1 limit was set after seeing the baselines. On a fresh run, Sol multiplied two experts (layer 4 expert 24, layer 7 expert 20) by -128, after checking -8, -256 and -128 with probes:

| Edit | Experts | Hidden Python loss | Hidden English (novels) loss | Score v2 |
|---|---:|---:|---:|---:|
| none | 0 | 1.58 | 2.83 | |
| 64 random experts zeroed (3 seeds) | 64 | +0.15 to +0.33 | +0.15 to +0.73 | 0 |
| 64 most code-routed experts zeroed | 64 | +1.07 | +0.09 | 1.07 |
| Sol, two experts times -128 | 2 | +11.33 | +0.04 | 9.22, the maximum |

**What the edit removed.** On text the score never used, the two-expert edit breaks C headers (+10.6), Markdown READMEs (+10.7) and reStructuredText docs (+8.7). The 64-expert baseline raises the same three by 0.7 to 0.8. In this model the router separates prose from technical text, not Python from the rest. Both edits cut the technical cluster. Sol's edit turns two experts into a trap: when the router picks them, the output becomes confidently wrong. The English test set held only novels, so the score missed this. A better score holds out technical English as a retain set. Records: [`results/moe/`](../results/moe).

**Version 3** keeps both English prose and technical English (package docs and READMEs with no Python code), each within 0.1 nats per token, and subtracts any Python loss beyond a uniform guess, so forgetting beats breaking. Under this rule every fixed baseline scores 0, and so does the two-expert trap (technical English +9.32). On a fresh run, Sol attenuated or reversed 32 code-selective experts:

| Edit | Python rise | English prose rise | Technical English rise | Score v3 |
|---|---:|---:|---:|---:|
| 64 experts most routed to code over prose and docs | +0.61 | +0.16 | +0.23 | 0 |
| Sol, 32 experts attenuated or reversed | +0.52 | +0.06 | +0.102 | 0 |

Sol's edit is the most selective so far, and misses the technical-English limit by 0.002 nats. On the practice texts the same edit read +0.070. The threshold stays as the prompt stated. The audit passed every code run. Separating Python from technical English inside this router is the open problem the task now measures.

**The J-lens.** Following Meyers' J-Vision, the host computes the J-lens once, before any run: J_l is the Jacobian of the last layer's residual output with respect to the residual stream after layer l, averaged over position pairs on 18 practice texts. The computation ran with gradients on the host, about 3.3 hours on 4 CPU threads. The agent gets the result as read-only arrays at a new access level, `instruments`, plus `lens_rows(l, ids)`, the direction after layer l along which each token's logit rises. The agent still runs no gradients. We ran Sol in three arms on the hidden v3 sets. The first arm gets no lens (4 runs). The second gets the J-lens (8 runs). The third is a control: the same prompt paragraph and the same helpers, but the plain logit lens (J_l = identity), so later layers count for nothing (8 runs). The control separates the Jacobian from the method the lens paragraph suggests, reading each expert's output columns through lens rows.

| Arm | Run | Python rise | English prose rise | Technical English rise | Audit | Score v3 |
|---|---|---:|---:|---:|---|---:|
| no lens | 1 | +0.52 | +0.06 | +0.102 | pass | 0 |
| no lens | 2 | +0.46 | +0.05 | +0.06 | pass | 0.46 |
| no lens | 3 | +0.16 | +0.04 | +0.05 | pass | 0.16 |
| no lens | 4 | +0.75 | +0.07 | +0.108 | pass | 0 |
| logit lens | 1 | +0.35 | +0.02 | +0.06 | fail | 0 (0.35 before audit) |
| logit lens | 2 | +0.78 | +0.05 | +0.09 | pass | 0.78 |
| logit lens | 3 | +0.74 | +0.05 | +0.08 | pass | 0.74 |
| logit lens | 4 | +0.55 | +0.04 | +0.07 | fail | 0 (0.55 before audit) |
| logit lens | 5 | +0.59 | +0.04 | +0.07 | pass | 0.59 |
| logit lens | 6 | +0.24 | +0.04 | +0.06 | pass | 0.24 |
| logit lens | 7 | +0.98 | +0.04 | +0.106 | pass | 0 |
| logit lens | 8 | +0.27 | +0.05 | +0.08 | pass | 0.27 |
| J-lens | 1 | +0.93 | +0.03 | +0.07 | pass | 0.93 |
| J-lens | 2 | +0.53 | +0.05 | +0.09 | pass | 0.53 |
| J-lens | 3 | +0.98 | +0.05 | +0.09 | pass | 0.98 |
| J-lens | 4 | +0.83 | +0.05 | +0.09 | pass | 0.83 |
| J-lens | 5 | +0.88 | +0.02 | +0.08 | pass | 0.88 |
| J-lens | 6 | +0.79 | +0.04 | +0.07 | pass | 0.79 |
| J-lens | 7 | +1.24 | +0.03 | +0.06 | fail | 0 (**1.24** before audit) |
| J-lens | 8 | +0.81 | +0.04 | +0.06 | pass | 0.81 |

Mean scores: no lens 0.16, logit lens 0.33, J-lens 0.72. Before the audit, the logit lens averages 0.44 and the J-lens 0.87. Mean Python rise, limits aside: no lens 0.47, logit lens 0.56, J-lens 0.87.

**All lens runs used one method.** Without a lens, Sol scales down or flips whole code-routed experts. Technical English uses the same experts, so two of four runs break the docs limit. With either lens, all sixteen runs did the same thing. Each took the lens rows of Python-only tokens such as indentation, `):` and triple quotes. Each removed from them the span of a few hundred common prose and docs tokens. Each then projected the rest out of the output weights of 60 to 125 code-routed experts. The edit removes what an expert writes toward Python syntax and keeps what the expert writes toward shared words. Logit-lens run 6 also erased seven experts outright. Fifteen of the sixteen lens runs stay within both retain limits. Logit-lens run 7 misses the technical-English limit by 0.006 nats.

**The Jacobian carries the gain.** The two lens arms differ only in J_l, and the J-lens arm wins on every measure. Python rise: 0.87 against 0.56, and a rank test puts the chance of a gap this size at 1 in 100 when J_l makes no difference. Audited score: 0.72 against 0.33, 1 in 130. The logit lens alone does not beat no lens by a clear margin: mean Python rise 0.56 against 0.47 (1 in 4), mean score 0.33 against 0.16 (1 in 4). Without later layers, the projection keeps the docs loss in check but removes less Python. The J-lens edits also carry over to unseen text: J-lens run 1 read +0.74 on practice texts and +0.93 on the hidden set.

**The audit.** Three lens runs failed, two with the logit lens and one with the J-lens. Each tuned the edit strength against the docs loss over several probes. Logit-lens run 4 ran a 60-step binary search over strength. J-lens run 7 tried a third strength after the second one broke the limit, and scored the highest Python rise of any run, +1.24. J-lens run 1 changed its strength once after a probe, and passed. The auditor judges where reasoning ends and search begins, so the line between runs 1 and 7 is a judgment call.

The J-lens is least reliable in early layers. Between-text dispersion is larger than the mean lens itself up to layer 7, and falls below 0.1 only from layer 20. The J-lens runs edited layers 8 to 23.

**What to take from this.** Telling the agent how to read an expert through the unembedding keeps its edits inside the retain limits, and costs nothing to compute. The Jacobian then roughly doubles the score, at 3.3 host hours per model. Records: `runs/bench/moe-v3*-gpt-6.1-sol/`.

## Catching NULL dereferences in C

A public post reported Opus 5.5 editing Qwen3.5-0.8B so the model catches NULL pointer dereferences in NIST Juliet CWE-476 programs, with no gradients, training, classifiers or parameter sweeps. We rebuilt the experiment as `tasks/vuln` from the public suite, without the post's code.

**The task.** The model reads one C program and answers YES or NO to one question: `Can calling entry() reach a NULL pointer dereference on any execution path?` The hidden set holds 186 matched Juliet pairs, a vulnerable and a safe version of the same code, each under 3 naming seeds: 1116 programs. Comments and every label-bearing name are gone. The agent edits MLP weights only (at most 0.05 % of them), reads activations through 30 probes, and writes its own probe programs. The agent never sees a Juliet program. The score is recall, if false positives stay at or below the base model's. Probes and scoring ran on a rented GPU.

| Model | Vulnerable caught (of 558) | False positives (of 558) | Accuracy | Weights changed | Audit |
|---|---:|---:|---:|---:|---|
| Qwen3.5-0.8B as released | 407 | 407 | 50.0 % | 0 | |
| A one-line text rule: YES unless the code compares a pointer with NULL | 432 | 108 | 79.0 % | | |
| Opus 5.5, run 1 | 478 | 174 | 77.2 % | 3,072 (1 neuron) | **fail** |
| Opus 5.5, run 2 | **522** | 156 | **82.8 %** | 9,216 (3 neurons) | pass, 27 of 27 code runs |

The post reports 48.3 % to 76.6 % accuracy on its own 77 pairs, with 2 neurons changed.

**Run 1 broke the rules.** Opus recomputed an MLP layer in Python on saved activations, and swept thresholds and strengths over those activations. The auditor flagged both, so the run scores 0. The rules then said "no search" and "no running the model in Python". They did not say a sweep over saved activations counts as search. Rules version 2 says so: reason a value out, check the value with a probe, and adjust the value at most twice.

**Run 2 stayed inside the rules and beats the text rule.** Opus found the base model reads the word NULL as a sign of danger and ignores the guard around the dereference. A guarded program looked more dangerous than the unguarded one. From matched pairs of its own programs, Opus took mean-difference directions for "the dereference is guarded" and "the pointer holds valid memory". Opus then rebuilt three nearly silent neurons in layer 22:
- A guard neuron and a valid-memory neuron. Each fires above a threshold of two standard deviations and pushes toward NO.
- A damping term. In Juliet families where both versions compare a pointer with NULL, the term fades the first two neurons.
- A neuron with a constant push toward YES on every program, for recall.

The third neuron is a bias, not a reading of the code. The bias moves the answer threshold. The first two neurons carry the discrimination: false positives fell from 407 to 156 while catches rose. Opus adjusted each value at most once after a probe, and the auditor allowed every code run. The result holds across the 3 naming seeds: 174 catches and 51 to 53 false positives per seed. Records: `runs/bench/vuln-r*-claude-opus-5-5/`.

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

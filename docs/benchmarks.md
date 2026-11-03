# Benchmarks

OpenWiSH reports three kinds of numbers:

1. **Conformance.** Published runs replayed through OpenWiSH. These numbers check the tooling, not an agent.
2. **Live agent runs.** A real agent on the bundled environments, with the audit status next to each score.
3. **Interpretability baselines.** Fixed, non-LLM methods on the read environments, for scale.

## Conformance

### Method

The published replies of *Sparks of Weight-Space Programming* go in as a scripted agent (`scripts/replay_authors.py`, then `--backend scripted:<file>`). Their code runs in the OpenWiSH sandbox, their edits pass the OpenWiSH gate, their probe requests hit the OpenWiSH probes, and the OpenWiSH scorers grade their answers. When every layer behaves like the original, the results match.

"Within noise" marks sampled results (episodes, games) where the published runs used other seeds or hardware.

### Results

| Environment | Check | Published | OpenWiSH | State |
|---|---|---|---|---|
| Game MLPs | Grades of the 4 published answers | 4 / 4 | 4 / 4, same guesses and confidences | match |
| Diffusion MLPs | Grades of the 3 published answers | 3 / 3 | 3 / 3 | match |
| Char LMs | 11 published code runs re-executed, then graded | corpus python, heads 2.1, 3.1, 3.5 | same outputs, precision and recall 1.0 | match |
| Kinetix car | Base on 256 episodes, then the published edit on 1,024, with 6,073 weights changed | 0 % to 100 % | 0 / 256 to 1,024 / 1,024, 6,073 changed, same mean return to every printed digit | match |
| MinAtar | Published one-shot replies through the OpenWiSH edit language, 1,000 episodes (Breakout / Seaquest / Asterix) | 517.4 / 53.1 / 53.2 | 502.9 / 53.2 / 52.9 | within noise |
| 9x9 Go | Budget-0 network rebuilt from the published Python run | 14 tensors | 14 tensors, bit-identical | match |
| 9x9 Go | Published budget-50 network against the published same-size PPO, 2,000 games | 74.15 % | 74.15 % | match |
| 9x9 Go | Published budget-0 network (replayed) against the published same-size PPO, seeds 0 / 1 / 2 | 42.6 / 26.3 / 36.3 % | 43.3 / 26.0 / 35.3 % | within noise |
| 9x9 Go | Opponents trained by OpenWiSH, against random play | 79-98 % (hidden) | 94.7 / 92.2 % (hidden), 91.3 / 97.1 % (practice) | in range |
| Gemma pairs | Grades of the 3 published attention-tool answers | 3 / 3 | 3 / 3, same probabilities | match |
| Gemma pairs | Six fine-tunes rebuilt with the published script: share of wrong guesses followed on held-out known facts | sycophant 100 %, honest 2.8 / 0.7 / 0.7 % | sycophant 100 %, honest 1.4 / 0.7 / 0.7 % | match |
| Gemma pairs | Published attention-tool session for pair 1 replayed: the published numpy code on the rebuilt models, 5 attention probes on the GPU | B, 14 runs | B, 14 runs, graded correct | works |
| Qwen math | Published edits rebuilt on the text-only base, with hash checks | n/a | not yet reported | pending |

### Reproduce

```bash
python3 tests/test_scorers.py                                  # exp1, exp2, exp3, exp7 grades from the published answers
assets/venvs/sandbox/bin/python tests/test_adapters.py         # MinAtar hand networks and channel mapping
assets/venvs/kinetix/bin/python tests/test_kinetix.py          # Kinetix base and edited actor
cargo test -p openwish-envs --release -- --ignored             # Go head-to-head, 2,000 games
```

For any other row, replay the published run:

```bash
python3 scripts/replay_authors.py <published run dir> replay.txt
target/release/openwish run <task.toml> --backend scripted:replay.txt
```

### Known deviations

- **Go opponents.** The published 243-32-32-82 opponents were never released. OpenWiSH trains its own (`scripts/train_go_opponents.sh`), and these are much easier: the published budget-0 network beats them 98.8 % of the time, against 54 % on the originals. The headline Go score is therefore the mean win rate against the three published same-size PPO networks.
- **Kinetix.** Kinetix 3.0.2 and JAX 0.9 on CPU, against the original pinned commit on CUDA.
- **Gemma pairs.** GPU training is not bit-reproducible, so the rebuilt models match in behaviour, not in bytes.

## Live agent runs

### Method

Run the same agent and model on the same environments in two modes:

| Mode | Settings |
|---|---|
| Baseline | The original protocol: verbatim prompts, one block per turn, fixed 40-turn handoffs, no notebook, no checkpoints |
| Full OpenWiSH | Every feature on |

Compare score, budget lost to failed requests, turns to finish, completion and audit validity, over three or more repeats per condition. Go is the best testbed, because score climbs with the pass budget (0, 10, 25, 50), so instrument quality shows up in the score. Char-LM reading (copying-head recall) and MinAtar writing come next.

Report every live score with the model, the effort, the backend, the cost from `state.json`, and the `valid` flag from `openwish audit`. `scripts/bench_live.sh <label> <backend> [model] -- <task.toml> ...` runs a list of tasks into `runs/bench/<label>/` and resumes unfinished ones. Set `OPENWISH_BENCH_BIN` to pin a binary for a round. A failed task makes the batch return a nonzero status.

### Results

You check every task with `python3 scripts/bench_compare.py <baseline-folder> <candidate-folder>`. Pass `--expected-task <id>` for each required task. The command fails on regressions, missing scores, unfinished runs, invalid audits or unresolved audit verdicts. Higher averages do not erase task-level losses.

<!-- BENCH:LIVE:START -->
These are single runs on public tasks. Each row shows the audit status. The bundled runs use the Claude CLI at high effort. Cost comes from `state.json`. Codex does not report cost to this backend.

Game MLPs: 4/4 correct. Diffusion MLPs: 3/3 correct. Claude reports lower confidence than the published GPT-6 Astra replies. Breakout is 54.2 against the published 517.4. You should not infer a general improvement from these results.

| Environment | Published agent (GPT-6 Astra) | OpenWiSH + Claude Opus 5.5 | Turns | Python runs | Cost (USD) | Audit |
|---|---|---|---|---|---|---|
| exp1-0bbcc3ea | freeway (97 %) | freeway (72 %) | 1 | 0 | 0.87 | not audited |
| exp1-a9f535cc | breakout (99 %) | breakout (75 %) | 1 | 0 | 0.93 | not audited |
| exp1-bdf99279 | pacman (96 %) | pacman (60 %) | 1 | 0 | 0.83 | not audited |
| exp1-dfc6cb65 | pong (96 %) | pong (60 %) | 1 | 0 | 0.87 | not audited |
| exp2-246a5b58 | triangle (88 %) | triangle (45 %) | 1 | 0 | 1.33 | not audited |
| exp2-a6a9dfb7 | square (77 %) | square (35 %) | 1 | 0 | 1.10 | not audited |
| exp2-d88aba69 | circle (84 %) | circle (60 %) | 1 | 0 | 0.97 | not audited |
| exp5-breakout | 517.4 | 54.2 ± 1.6 | 1 | 0 | 3.33 | not audited |
| exp6-budget_0 | 42.6 / 26.3 / 36.3 % | 41.0 / 28.7 / 42.6 % | 18 | 17 | 112.47 | not audited |

| Qwen3.5-0.8B on MATH-500 (1 sample, 32,768 tokens) | Published (4 samples) | OpenWiSH | Share cut at 32,768 |
|---|---|---|---|
| base | 7.9 % | 10.6 ± 1.4 % | 90.2 % |

#### WSPBench

WSPBench condition A matches the pilot: one reply, no tools. OW adds a Python sandbox over the weights, a format-only checker and a guard against activation code. OW is not a WSPBench condition. Report OW apart from A, B and C. Each cell is one attempt per task. The pilot used API calls. These runs use CLI backends. Model aliases do not pin a snapshot.

Codex tuned the OW config in five rounds on the six pilot tasks. Each round fixed one OpenWiSH fault. The config then froze. The frozen config ran once on six development tasks no round had used: two rule-recovery, two edit-effects and two synthesis tasks, the pilot's families. `runs/heldout-r5-manifest.json` holds the binary and task hashes. The held-out rows are the fair estimate.

A Codex auditor (`gpt-6.1-sol`, medium effort) reviews every OW run with code. A run with a failed audit scores 0 in these tables. One-reply runs execute no code and need no audit. Write outputs are class-imbalanced. The pilot's static zero-start control scores 70.1 % target accuracy with 0/2 exact. Judge construction by exact success.

On the held-out six, the sandbox takes `gpt-6.1-sol` and `gpt-6.1-terra` from 0/4 exact reads to 4/4, with every audit valid. Both reach 2/2 exact writes. `gpt-6-luna` gains both writes and no read. One Luna read failed its audit: the code tested gate inputs against weighted thresholds. Six tasks per row do not support a general ranking.

Claude ran the original OW config, before the rounds. The Claude backend forwarded task instructions from the start. Strict scoring rejects two Claude A replies for extra `explanation` keys. Claude has no held-out run yet.

| Model | Tasks | Condition | Read agreement | Read exact | Write accuracy | Write exact | Audit | Source |
|---|---|---|---:|---:|---:|---:|---|---|
| gpt-6-luna | pilot six | A | 44.1 % | 0/4 | 71.9 % | 0/2 | no code run | WSPBench pilot |
| gpt-6-luna | pilot six | A, through OpenWiSH | 46.6 % | 0/4 | 70.3 % | 0/2 | no code run | this repo |
| gpt-6-luna | pilot six | OpenWiSH sandbox | 69.0 % | 0/4 | 100.0 % | 2/2 | 6/6 valid | this repo |
| gpt-6-luna | held-out six | A, through OpenWiSH | 51.9 % | 0/4 | 82.2 % | 0/2 | no code run | this repo |
| gpt-6-luna | held-out six | OpenWiSH sandbox | 40.6 % | 0/4 | 100.0 % | 2/2 | 4/5 valid, 1 invalid | this repo |
| gpt-6.1-terra | pilot six | A | 66.0 % | 0/4 | 97.3 % | 1/2 | no code run | WSPBench pilot |
| gpt-6.1-terra | pilot six | A, through OpenWiSH | 35.3 % | 0/4 | 70.3 % | 0/2 | no code run | this repo |
| gpt-6.1-terra | pilot six | OpenWiSH sandbox | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid | this repo |
| gpt-6.1-terra | held-out six | A, through OpenWiSH | 41.0 % | 0/4 | 82.2 % | 0/2 | no code run | this repo |
| gpt-6.1-terra | held-out six | OpenWiSH sandbox | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid | this repo |
| gpt-6.1-sol | pilot six | A | 87.7 % | 0/4 | 95.8 % | 1/2 | no code run | WSPBench pilot |
| gpt-6.1-sol | pilot six | A, through OpenWiSH | 89.7 % | 1/4 | 99.2 % | 1/2 | no code run | this repo |
| gpt-6.1-sol | pilot six | OpenWiSH sandbox | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid | this repo |
| gpt-6.1-sol | held-out six | A, through OpenWiSH | 75.1 % | 0/4 | 96.9 % | 1/2 | no code run | this repo |
| gpt-6.1-sol | held-out six | OpenWiSH sandbox | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid | this repo |
| claude-opus-5-5 | pilot six | A, through OpenWiSH | 22.0 % | 0/4 | 88.3 % | 1/2 | no code run | this repo |
| claude-opus-5-5 | pilot six | OpenWiSH sandbox | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid | this repo |

The rounds table shows why audits matter. Before the r4 guard, many high scores came from code evaluating gates on inputs. Those runs fail audit and score 0 here.

| Model | OW round, pilot six | Read agreement | Read exact | Write accuracy | Write exact | Audit |
|---|---|---:|---:|---:|---:|---|
| gpt-6-luna | r1: task instructions forwarded | 17.9 % | 0/4 | 70.3 % | 0/2 | 3/5 valid, 2 invalid |
| gpt-6-luna | r2: public task JSON in Python | 0.0 % | 0/4 | 100.0 % | 2/2 | 3/4 valid, 1 invalid |
| gpt-6-luna | r3: format check at finish | 31.1 % | 0/4 | 86.7 % | 1/2 | 4/6 valid, 2 invalid |
| gpt-6-luna | r4: activation guard before execution | 61.7 % | 0/4 | 100.0 % | 2/2 | 6/6 valid |
| gpt-6-luna | r5: context accounting, reply blocks kept | 69.0 % | 0/4 | 100.0 % | 2/2 | 6/6 valid |
| gpt-6.1-terra | r1: task instructions forwarded | 36.3 % | 0/4 | 50.0 % | 1/2 | 2/4 valid, 2 invalid |
| gpt-6.1-terra | r2: public task JSON in Python | 23.5 % | 0/4 | 100.0 % | 2/2 | 3/6 valid, 3 invalid |
| gpt-6.1-terra | r3: format check at finish | 61.9 % | 2/4 | 100.0 % | 2/2 | 4/5 valid, 1 invalid |
| gpt-6.1-terra | r4: activation guard before execution | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid |
| gpt-6.1-terra | r5: context accounting, reply blocks kept | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid |
| gpt-6.1-sol | r1: task instructions forwarded | 94.8 % | 3/4 | 98.4 % | 1/2 | 5/5 valid |
| gpt-6.1-sol | r2: public task JSON in Python | 100.0 % | 4/4 | 50.0 % | 1/2 | 6/6 valid |
| gpt-6.1-sol | r3: format check at finish | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid |
| gpt-6.1-sol | r4: activation guard before execution | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid |
| gpt-6.1-sol | r5: context accounting, reply blocks kept | 100.0 % | 4/4 | 100.0 % | 2/2 | 6/6 valid |

The first Codex backend joined progress messages to the final answer and dropped task instructions. Those runs stay in `results/` as transport records. The table lists them last. They do not estimate model skill. Final-message replay rows rescore the last completed message from saved CLI events with no new model call.

| Model | Superseded transport record | Read agreement | Read exact | Write accuracy | Write exact | Audit |
|---|---|---:|---:|---:|---:|---|
| gpt-6-luna | A, through OpenWiSH | 15.5 % | 0/4 | 0.0 % | 0/2 | no code run |
| gpt-6-luna | A, final-message replay | 15.5 % | 0/4 | 70.3 % | 0/2 | no code run |
| gpt-6-luna | OpenWiSH sandbox | 0.0 % | 0/4 | 0.0 % | 0/2 | no code run |
| gpt-6-luna | OpenWiSH sandbox, corrected backend | 0.0 % | 0/4 | 70.3 % | 0/2 | no code run |
| gpt-6.1-terra | A, through OpenWiSH | 0.0 % | 0/4 | 0.0 % | 0/2 | no code run |
| gpt-6.1-terra | A, final-message replay | 15.6 % | 0/4 | 0.0 % | 0/2 | no code run |
| gpt-6.1-terra | OpenWiSH sandbox | 10.4 % | 0/4 | 36.7 % | 0/2 | no code run |
| gpt-6.1-terra | OpenWiSH sandbox, corrected backend | 26.7 % | 0/4 | 33.6 % | 0/2 | no code run |
| gpt-6.1-sol | A, through OpenWiSH | 0.0 % | 0/4 | 0.0 % | 0/2 | no code run |
| gpt-6.1-sol | A, final-message replay | 79.9 % | 0/4 | 50.0 % | 1/2 | no code run |
| gpt-6.1-sol | OpenWiSH sandbox | 0.0 % | 0/4 | 0.0 % | 0/2 | no code run |
| gpt-6.1-sol | OpenWiSH sandbox, corrected backend | 82.5 % | 0/4 | 97.3 % | 1/2 | no code run |

<!-- BENCH:LIVE:END -->

## Interpretability baselines

Fixed methods with no LLM, chosen before looking at the ground truth, give the read environments a reference point. `scripts/baselines.py` prints the results as JSON.

<!-- BENCH:BASELINES:START -->
The fixed K-composition baseline (Elhage et al., *A Mathematical Framework for Transformer Circuits*) labels heads above the mean plus two standard deviations. The rule finds 0/5 copying heads. The published agent found 5/5. This comparison tests one fixed threshold, not every circuit-analysis method. Source: `runs/baselines.json`.

| Corpus | Copying heads found | Ground-truth heads | Recall |
|---|---:|---:|---:|
| python | 0 | 3 | 0 % |
| romaji | 0 | 1 | 0 % |
| english | 0 | 1 | 0 % |

<!-- BENCH:BASELINES:END -->

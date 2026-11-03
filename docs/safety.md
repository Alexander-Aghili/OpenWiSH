# Safety

OpenWiSH measures how well an agent reasons about weights. A score means nothing if the agent trained the network instead. This page lists what OpenWiSH enforces, how, and where enforcement stops.

## Threat model

| Party | Trust | Runs where |
|---|---|---|
| The agent and every line of code the agent writes | untrusted | the bubblewrap sandbox |
| Paths and files coming from the sandbox (`/work`, `/work/.rpc`, `notes.md`, probe folders) | untrusted on the host | read by the host through the path rules below |
| The task author: spec, probes, scorers, preludes | trusted | probes and scorers on the host, preludes in the sandbox |
| The backend CLI (`claude`, `codex`) | trusted, with its tools switched off | the host, in an empty folder |

## The four layers

### 1. Capability removal (hard)

The kernel holds arrays, never a runnable model. Behaviour of the network comes only through probes the task declares and counts.

| Control | Detail |
|---|---|
| No autograd | `torch.set_grad_enabled(False)` at start. `Tensor.backward`, `torch.autograd.backward`, `torch.autograd.grad`, `torch.enable_grad`, `torch.func.grad` and its siblings, `requires_grad_(True)`, the autograd engine and the internal grad switch raise `OpenWiSHForbidden` |
| No optimisers | Every `torch.optim.Optimizer` subclass in `torch.optim` and each submodule (`torch.optim.adam.Adam`, ...) raises |
| No network | `bwrap --unshare-all` (network, PID, IPC, UTS, user and cgroup namespaces), `--die-with-parent`, `--new-session`, `--clearenv` |
| No GPU | `CUDA_VISIBLE_DEVICES` empty, a minimal `/dev` |
| Read-only weights | The current network mounts read-only at `/net`. Only the gate rewrites the files |
| Hidden assets | Every path in `hidden` is checked against every mount when the spec loads, after resolving symlinks. A mount containing a hidden path, or inside one, fails the load |
| Limits | Address space capped with `prlimit --as` (`mem_gb`, default 16 GB), a timeout per run, threads within the CPU budget |

Sandbox mounts:

| Guest path | Host source | Mode |
|---|---|---|
| `/usr`, `/lib`, `/lib64`, `/bin` | the host system | read-only |
| the venv and the interpreter | `assets/venvs/<name>` | read-only |
| `/openwish` | `<run>/kernel/` (kernel.py, prelude.py) | read-only |
| `/net` | `<run>/edits/current/` | read-only |
| `/work` | `<run>/work/` | read-write |
| task `[[mounts]]` | as declared | read-only unless `rw = true` |
| `/tmp` | tmpfs | read-write, discarded |

### 2. Edit gate (hard)

Every weight change goes through `openwish_tensor::gate::Gate`, from Python RPCs and from ```` ```edits ```` blocks alike. The gate refuses an edit when:

- the tensor is unknown, outside `allowed`, or matches `forbidden`,
- the result holds a non-finite value,
- the result equals the whole old tensor times one constant, with `forbid_whole_tensor_scale`,
- the share of entries differing from the base would pass `max_frac_changed`.

A refused edit leaves no ledger line and no file. An accepted edit appends one ledger line with the SHA-256 of the tensor before and after.

### 3. Static checks (soft)

Each code run is scanned before execution. By default, matches go to the agent as a note in the next message and to `tool_calls.jsonl`. A task opts into blocking with `[python] reject_flags`. A blocked request never starts Python and spends no Python budget. The trace records the blocked code as `python-policy`. Conservative flags sometimes reject allowed algebra. Static checks remain incomplete, so audits still apply.

| Rule | Flags |
|---|---|
| `backward` | `.backward(` |
| `autograd` | `autograd.grad`, `torch.func.grad`/`vjp`/`jvp`/`jacrev`/`jacfwd`, `jax.grad`, `jax.value_and_grad`, `jax.vjp`, `requires_grad=True` |
| `optimizer` | `torch.optim.`, `optax.`, `optim.SGD`, `optim.Adam` and similar |
| `scipy_optimize` | `scipy.optimize`, `minimize(`, `least_squares(`, `curve_fit(`, `differential_evolution(`, `basinhopping(` |
| `manual_gradient_step` | `-= lr *` and the same with `learning_rate`, `eta`, `step_size` |
| `fit_into_weights` | `.fit(` in code calling `save_edit` |
| `lstsq_into_weights` | `lstsq(`, `pinv(`, `solve(` in code calling `save_edit` |
| `scalar_relu` | A numeric `max(0, ...)`, including possible component-level simulation |
| `forward_pass` | an activation (`tanh`, `relu`, `sigmoid`, `softmax`, `gelu`, `silu`, `maximum(0, ...)`) applied to a matrix product |

### 4. Auditor (soft)

`openwish audit <run>` sends every Python run, with its output, the saved task instructions and the task prompt, to a reviewer model in batches of 8. The reviewer receives a neutral role. Participant instructions appear as evidence rather than reviewer instructions. Each run gets `allowed`, `suspicious` or `violation` with a reason. One `violation` sets `valid: false` in `audit_summary.json`. Report the audit status next to every score.

## Path rules for sandbox-controlled paths

The host treats every path from the sandbox side as hostile. `crates/openwish-core/src/paths.rs` holds the checks.

| Path | Rule | On failure |
|---|---|---|
| The array behind `save_edit` | A regular file inside `<run>/work/.rpc`, no `..`, no symlink at any step (a symlink pointing back inside counts too) | Edit refused |
| `/work/notes.md` in a handoff | A regular file inside `/work`, no symlink | Not shown, the handoff says why |
| A probe output folder (`out_dir`) | A plain relative path, no `..`, not absolute, every existing step a real folder | Pass not run, not charged |
| A checkpoint name | Letters, digits, `-` and `_`, at most 64 characters | Refused |
| The run folder | Made absolute at creation and on resume | n/a |

## Known gaps

No layer is airtight. These gaps are open today.

- **Hand-simulated forward passes.** An agent writes the forward pass in numpy and runs the network on inputs. The `forward_pass` rule flags common spellings. Only the auditor catches the rest.
- **Gradients written by hand.** The guards cover torch. A finite-difference or hand-derived gradient in numpy runs. The static checks flag common step spellings, and the auditor reviews the code.
- **Other frameworks.** The guards install for torch only. A task venv with JAX gets static flags for `jax.grad`, no runtime block.
- **Pattern checks.** Static checks match text. Renamed or indirect calls (`getattr`, string building) pass unflagged.
- **The auditor is a model.** Verdicts are judgments, not proofs. The audit runs after the fact, and only when you run `openwish audit`. The auditor reviews Python runs, not probe requests.
- **Host-side task code.** Probes and scorers run on the host with your user's rights. Treat a third-party environment as code you run. A probe writing hidden data into `/work` shows the data to the agent.
- **Backend CLIs on the host.** `claude -p` and `codex exec` run outside bubblewrap, in an empty folder, with their tools switched off by flags. OpenWiSH relies on those flags.
- **Memory limit.** `prlimit --as` caps address space, not resident memory, and some runtimes reserve large address ranges.

### Deferred review items

The whole-branch review recorded seven minor items for later:

1. The kernel protocol uses the inherited stdout.
2. Cleanup after interrupted edit RPCs needs review. Refused edits now remove the temporary blob.
3. A one-reply task loses the reply on an error.
4. The whole-tensor-scale check is a heuristic.
5. The `api` backend sends no prompt-caching hints.
6. `restore` is not atomic.
7. The kernel start wait is a fixed 120 s.

Report a gap not listed here through [SECURITY.md](../SECURITY.md).

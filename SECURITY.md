# Security policy

OpenWiSH runs untrusted agent code in a sandbox and trusts the results only when the agent stayed inside the rules. A sandbox escape breaks both the safety of your machine and the meaning of every score.

## Report a vulnerability

Report privately through [GitHub security advisories](https://github.com/Alexander-Aghili/OpenWiSH/security/advisories/new). Do not open a public issue for an escape.

Include:

1. The OpenWiSH commit (`git rev-parse HEAD`).
2. The task spec and the agent reply or Python code reproducing the problem. A scripted reply file (`--backend scripted:<file>`) is ideal.
3. What happened, and what you expected.

The maintainer answers every report. Each fix ships with a test reproducing the attack, and the advisory credits you unless you ask otherwise.

## In scope

Anything letting code in the sandbox, or an agent reply, do one of these:

| Class | Examples |
|---|---|
| Write weights outside the gate | Changing `/net` or `edits/current/` directly, a `save_edit` path the gate accepts from outside `/work/.rpc`, an edit leaving no ledger line, an edit passing a forbidden tensor or a policy limit |
| Read hidden assets | Answer keys, hidden opponents, evaluation sets listed in `hidden`, through a mount, a symlink, a probe or a handoff |
| Touch host paths | Reading or writing host files outside the mounts, deleting a host folder through a probe folder, a checkpoint name or a run folder |
| Escape the capability limits | Reaching the network or a GPU from the sandbox, re-enabling autograd or an optimiser in the kernel |
| Corrupt the record | Forging ledger lines, hashes, `tool_calls.jsonl` entries or budgets, a charged pass with no record |

## Out of scope

- Behaviour the [known gaps](docs/safety.md#known-gaps) already list, such as a forward pass hand-written in numpy. Proposals to close a listed gap are welcome as normal issues.
- Probe and scorer commands written by a task author. These run on the host by design.
- Bugs in `bubblewrap`, the Linux kernel, Python or the backend CLIs. Report those upstream, and tell us if OpenWiSH needs a workaround.

## Supported versions

Security fixes land on `main` first.

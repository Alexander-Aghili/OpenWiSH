"""Run WSPBench conditions A, B and C in WSPBench's own Inspect harness, with an OpenWiSH backend as the model.

usage: assets/wspbench/.venv/bin/python -I scripts/wspbench_bc.py --backend codex --model gpt-6.1-sol \
           --condition B --task <id> [--task <id> ...] --label <label>

WSPBench's solver, tools, budgets, metering and scorer run unchanged (build_inspect_task, eval_async). Only the
transport differs from the pilot: each model turn goes through `openwish ask` (the backend CLI with its own
tools off) instead of a provider API. The transcript and the tool list are rendered as text, and the model
calls a tool with one ```tool_call block. A malformed tool call gets one transport retry with the parse error.
Writes runs/bench/<label>/<task>-<cond>-task/{result.json, trace.json, submission.json, eval/score.json}."""
import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "assets/wspbench"
sys.path.insert(0, str(BENCH / "src"))

from inspect_ai.model import ModelAPI, ModelOutput, ModelUsage, ChatMessageAssistant, get_model, modelapi  # noqa: E402
from inspect_ai.tool import ToolCall  # noqa: E402

PROTOCOL = """You can call one tool per reply. To call a tool, reply with exactly one block:
```tool_call
{"function": "<tool name>", "arguments": {<arguments as JSON>}}
```
The environment runs the tool and sends you its result. A reply without a tool_call block is your final answer.

Tools:
"""
BLOCK = re.compile(r"```tool_call\s*\n(.*?)```", re.S)


def render(messages, tools):
    """System text plus the tool protocol as instructions, and the rest of the transcript as one prompt."""
    system = "\n\n".join(m.text for m in messages if m.role == "system")
    if tools:
        listing = "\n".join(f"- {t.name}: {t.description}\n  parameters: {t.parameters.model_dump_json(exclude_none=True)}" for t in tools)
        system += "\n\n" + PROTOCOL + listing
    turns = []
    for m in messages:
        if m.role == "user":
            turns.append(f"[user]\n{m.text}")
        elif m.role == "assistant":
            calls = "".join(f"\n```tool_call\n{json.dumps({'function': c.function, 'arguments': c.arguments})}\n```" for c in m.tool_calls or [])
            turns.append(f"[assistant]\n{m.text}{calls}")
        elif m.role == "tool":
            body = f"error: {m.error.message}" if m.error else m.text
            turns.append(f"[tool result: {m.function}]\n{body}")
    return system, "\n\n".join(turns) + "\n\n[assistant]\n"


def parse(text, tool_names):
    """(content, ToolCall or None, error or None). The last tool_call block is the call."""
    blocks = BLOCK.findall(text)
    if not blocks:
        return text.strip(), None, None
    try:
        call = json.loads(blocks[-1])
    except json.JSONDecodeError as e:
        return text, None, f"the tool_call block is not valid JSON ({e})"
    if not isinstance(call, dict) or call.get("function") not in tool_names or not isinstance(call.get("arguments", {}), dict):
        return text, None, f"the tool_call block must name one of {sorted(tool_names)} with an arguments object"
    content = BLOCK.sub("", text).strip()
    return content, ToolCall(id=f"call_{abs(hash(text)) % 10**9}", function=call["function"], arguments=call.get("arguments", {})), None


async def ask(backend, model, instructions, prompt, effort):
    with tempfile.TemporaryDirectory() as tmp:
        rules = Path(tmp, "instructions.txt")
        rules.write_text(instructions)
        proc = await asyncio.create_subprocess_exec(
            os.environ.get("OPENWISH_ASK_BIN", str(ROOT / "target/release/openwish")), "ask", "--backend", backend, "--model", model, "--effort", effort,
            "--instructions", str(rules), "--workdir", str(Path(tmp, "cwd")), "--timeout", "900",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate(prompt.encode())
    if proc.returncode != 0:
        raise RuntimeError(f"openwish ask failed: {err.decode()[-800:]}")
    return json.loads(out.decode().strip().splitlines()[-1])


def usage(*replies, overhead=0):
    """WSPBench charges input per provider request. A backend CLI may make extra internal requests in one turn
    (Codex re-reads its cached prefix), so a reply with a known context size counts as one request of that size.
    `overhead` is the measured per-request cost of the transport itself (tokens an API call would not send),
    and is not charged. Billed usage stays in the transport log."""
    u = {k: sum(r["usage"].get(k, 0) for r in replies) for k in ("input", "output", "cache_read", "cache_write", "reasoning")}
    if all(r.get("context_tokens") for r in replies):
        uncached, cache_read, cache_write = sum(max(r["context_tokens"] - r["usage"].get("output", 0) - overhead, 0) for r in replies), 0, 0
    else:
        uncached, cache_read, cache_write = u["input"], u["cache_read"], u["cache_write"]
    total_in = uncached + cache_read + cache_write  # input_tokens is the uncached part only
    return ModelUsage(input_tokens=uncached, output_tokens=u["output"], total_tokens=total_in + u["output"],
                      input_tokens_cache_read=cache_read, input_tokens_cache_write=cache_write, reasoning_tokens=u["reasoning"])


@modelapi(name="openwish")
class OpenWiSHModel(ModelAPI):
    """model name "<backend>:<model>", for example "codex:gpt-6.1-sol"."""

    def __init__(self, model_name, base_url=None, api_key=None, api_key_vars=[], config=None, **kw):
        from inspect_ai.model import GenerateConfig
        super().__init__(model_name, base_url, api_key, api_key_vars, config or GenerateConfig())
        self.backend, self.model = model_name.split(":", 1)

    async def generate(self, input, tools, tool_choice, config):
        instructions, prompt = render(input, tools)
        effort = config.reasoning_effort or "high"
        replies = [await ask(self.backend, self.model, instructions, prompt, effort)]
        content, call, error = parse(replies[-1]["text"], {t.name for t in tools})
        if error:
            retry = prompt + replies[-1]["text"] + f"\n\n[transport]\n{error}. Reply again.\n\n[assistant]\n"
            replies.append(await ask(self.backend, self.model, instructions, retry, effort))
            content, call, _ = parse(replies[-1]["text"], {t.name for t in tools})
        log = os.environ.get("OPENWISH_BC_LOG")
        if log:
            with open(log, "a") as f:
                f.write(json.dumps({"requests": len(replies), "billed": [r["usage"] for r in replies],
                                    "context_tokens": [r.get("context_tokens") for r in replies], "retry": bool(error)}) + "\n")
        message = ChatMessageAssistant(content=content, tool_calls=[call] if call else None, model=self.model_name)
        return ModelOutput(model=self.model_name, choices=[{"message": message, "stop_reason": "tool_calls" if call else "stop"}],
                           usage=usage(*replies, overhead=int(os.environ.get("OPENWISH_BC_OVERHEAD", "0"))))


async def run_cell(task_id, condition, backend, model, out_dir):
    from inspect_ai import eval_async
    from wspbench.public.replay import PublicTask
    from wspbench.v2.inspect_task import build_inspect_task
    from wspbench.v2.runner import _base_record, _inspect_paths
    import runpy
    bridge = runpy.run_path(str(ROOT / "python/adapters/wspbench_bridge.py"))
    d = BENCH / "data/development-v0.2.1/dev" / task_id
    task = PublicTask.from_dict(json.loads((d / "task.json").read_text()), d)
    run_dir = out_dir / f"{task_id}-{condition}-task"
    if (run_dir / "eval/score.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    _inspect_paths(out_dir)
    config = {"id": f"openwish-{backend}-{model}", "provider": "openwish", "model": f"openwish/{backend}:{model}",
              "generate_config": {"reasoning_effort": "high"}}
    record = _base_record(task, config["id"], "model", condition, "full", "openwish-transport")
    evaluation = build_inspect_task(task, run_dir, config, record, condition=condition, weight_view="full")
    os.environ["OPENWISH_BC_LOG"] = str(run_dir / "transport.jsonl")
    await eval_async(evaluation, model=get_model(config["model"], memoize=False), log_dir=str(run_dir / "inspect"),
                     log_model_api=False, log_shared=False, retry_on_error=0, log_realtime=False, ctl_server=False,
                     max_samples=1, max_tasks=1)
    (run_dir / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    graded = bridge["grade"](task, record["submission"]) if record.get("status") == "completed" else {"valid": False, "error": record.get("status")}
    score = {"ok": True, "task": task_id, "family": task.family, "condition": condition, "strict": graded,
             "score": float(graded.get("accuracy") or 0.0), "exact": bool(graded.get("success", False)),
             "status": record.get("status"), "turns": record.get("turns"), "usage": record.get("usage")}
    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "eval/score.json").write_text(json.dumps(score, indent=1, default=str) + "\n")
    print(json.dumps({k: score[k] for k in ("task", "condition", "score", "exact", "status", "turns")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["A", "B", "C"])
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--overhead", type=int, default=0, help="measured per-request transport tokens not charged")
    a = ap.parse_args()
    out = ROOT / "runs/bench" / a.label
    os.environ["OPENWISH_BC_OVERHEAD"] = str(a.overhead)
    for t in a.task:
        asyncio.run(run_cell(t, a.condition, a.backend, a.model, out))


if __name__ == "__main__":
    main()

"""Build the vulnerability environment (tasks/vuln) from the NIST Juliet C/C++ suite v1.3, CWE-476 NULL pointer dereference.

Each single-file testcase gives one matched pair: the vulnerable program keeps the bad() chain, the safe program keeps
the first good variant that the testcase's good() wrapper calls, with the same shared preamble. Comments, main() and
every label-bearing name (bad, good, G2B, B2G, CWE476_...) go: the entry function becomes entry(), helpers get neutral
names by role (Sink, Source, Static ...). The seed picks those names, so one pair yields one program per seed.

usage: assets/venvs/qwen/bin/python -I scripts/prepare_vuln.py   (after downloading the suite, see tasks/vuln/README.md)
Writes assets/vuln/eval/programs.json (hidden: the scored set) and assets/vuln/base_net (MLP weights, float32)."""
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "assets/vuln"
_spec = importlib.util.spec_from_file_location("vuln_prompt", ROOT / "python/adapters/vuln_prompt.py")
_vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vp)
QUESTION, question = _vp.QUESTION, _vp.question  # the prompt lives in python/adapters/vuln_prompt.py
JULIET_SHA = "ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb"
SEEDS = (0, 1, 2)
MAX_TOKENS = 900
EXCLUDE = ("null_check_after_deref",)
NAME = re.compile(r"^(?:CWE476_\w+?__\w+?_\d+_)?(bad|good(?:G2B|B2G)?\d*)([A-Za-z]\w*)?$")
LEAK = re.compile(r"(?i:bad|good|G2B|B2G|CWE|OMIT|INCLUDEMAIN)|FLAW|FIX")
POOLS = {
    "Sink": ["consume", "use_value", "handle", "report", "emit", "finish", "process", "show"],
    "Source": ["make_value", "fetch", "obtain", "prepare", "acquire", "load_value", "produce", "init_value"],
    "Static": ["mode", "flag", "enabled", "active", "state", "toggle", "option", "switch_on"],
    "Global": ["shared", "current", "stored", "cell", "slot", "holder", "keep", "saved"],
    "Data": ["shared_data", "value_store", "holder_data", "cell_data", "stash", "kept", "global_value", "box"],
}


def strip_comments(src):
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def sections(src):
    """{'pre': text, 'bad': text, 'good': text}; main() and its INCLUDEMAIN block are dropped."""
    parts, stack = {"pre": [], "bad": [], "good": []}, []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#if"):
            tag = {"#ifndef OMITBAD": "bad", "#ifndef OMITGOOD": "good", "#ifdef INCLUDEMAIN": "main"}.get(s, None)
            stack.append(tag)
            if tag:
                continue
        elif s.startswith("#endif") and stack:
            if stack.pop():
                continue
        where = "main" if "main" in stack else next((t for t in reversed(stack) if t), "pre")
        if where != "main":
            parts[where].append(line)
    return {k: "\n".join(v) for k, v in parts.items()}


def items(text):
    """Top-level items: (text, defined name or None)."""
    out, buf, depth, i = [], [], 0, 0
    lines = text.splitlines(keepends=True)
    for line in lines:
        if depth == 0 and line.strip().startswith("#"):
            out.append((line.strip(), None))
            continue
        for ch in line:
            buf.append(ch)
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and "(" in "".join(buf).split("{", 1)[0] and "=" not in "".join(buf).split("{", 1)[0]:
                    out.append(("".join(buf).strip(), None))
                    buf = []
            elif ch == ";" and depth == 0:
                out.append(("".join(buf).strip(), None))
                buf = []
    if "".join(buf).strip():
        out.append(("".join(buf).strip(), None))
    named = []
    for t, _ in out:
        if not t or t == ";":
            continue
        head = t.split("{", 1)[0]
        m = re.search(r"(\w+)\s*\(", head) if "(" in head and "=" not in head.split("(", 1)[0] else re.search(r"(\w+)\s*(?:\[[^\]]*\])?\s*(?:=|;|$)", head)
        named.append((t, m.group(1) if m and not t.startswith("#") else None))
    return named


def reachable(entry, its):
    defined = {n: t for t, n in its if n}
    keep, todo = set(), [entry]
    while todo:
        n = todo.pop()
        if n in keep or n not in defined:
            continue
        keep.add(n)
        todo += [w for w in set(re.findall(r"\b\w+\b", defined[n])) if w in defined]
    return [(t, n) for t, n in its if n in keep]


def role_names(seed):
    names = {}
    for role, pool in POOLS.items():
        names[role] = random.Random(f"{seed}:{role}").choice(pool)
    return names


def rename(text, names, entry, seed):
    pool = role_names(seed)
    mapping = {}
    for n in names:
        if n == entry:
            mapping[n] = "entry"
            continue
        m = NAME.match(n)
        if not m:
            continue
        role = m.group(2) or ""
        mapping[n] = pool.get(role, f"{role.lower() or 'step'}_fn")
    for old, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def tidy(text):
    """Drop blank lines inside braces, keep one blank line between top-level items."""
    out, depth = [], 0
    for line in text.splitlines():
        if line.strip() or depth == 0:
            out.append(line.rstrip())
        depth += line.count("{") - line.count("}")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def scrub(text):
    """Remove the remaining label hints: CWE476 type names, 'static' on entry() (only good variants are static), and
    the dead-branch string Juliet prints mostly in good variants."""
    text = re.sub(r"\bCWE476_\w+?__\w+?_\d+_(\w+)", r"\1", text)
    text = re.sub(r"\bstatic\s+(void\s+entry\s*\()", r"\1", text)
    return text.replace('"Benign, fixed string"', '"done"')


def neutral_strings(text):
    """String literals naming the label ("Good", "bad data") become a neutral literal."""
    return re.sub(r'"(?:[^"\\]|\\.)*"', lambda m: '"hello"' if re.search(r"(?i)good|bad", m.group(0)) else m.group(0), text)


def build_pair(src, seed):
    parts = sections(strip_comments(src))
    pre_items = items(parts["pre"])
    labelled = [(t, n) for t, n in pre_items if n and NAME.match(n)]  # e.g. variant 45's badData / goodG2BData globals
    pre = tidy("\n\n".join(t for t, n in pre_items if (t, n) not in labelled))
    bad, good = labelled + items(parts["bad"]), labelled + items(parts["good"])
    bad_entry = next(n for _, n in bad if n and n.endswith("_bad"))
    wrapper = next(t for t, n in good if n and n.endswith("_good"))
    body = wrapper.split("{", 1)[1]
    variant = re.search(r"\b(\w+)\s*\(\s*\)\s*;", body).group(1)
    out = {"good_variant": variant}
    for kind, its, entry in (("vulnerable", bad, bad_entry), ("safe", good, variant)):
        kept = reachable(entry, its)
        names = [n for _, n in kept if n]
        code = tidy(pre + "\n\n" + "\n\n".join(t for t, _ in kept))
        out[kind] = tidy(scrub(neutral_strings(rename(code, names, entry, seed))))
    return out


def leaks(code):
    return LEAK.search(code)


def build_all(case_dir, tok=None):
    progs, skipped = [], []
    for f in sorted(Path(case_dir).glob("CWE476_*.c")):
        case = f.stem.split("__", 1)[1]
        if case.startswith(EXCLUDE):
            skipped.append({"case": case, "why": "label does not match the question: both versions dereference unchecked malloc"})
            continue
        for seed in SEEDS:
            try:
                pair = build_pair(f.read_text(errors="replace"), seed)
            except (StopIteration, AttributeError) as e:
                skipped.append({"case": case, "why": f"parse: {e!r}"})
                break
            bad = [k for k in ("vulnerable", "safe") if leaks(pair[k])]
            if bad:
                skipped.append({"case": case, "why": f"leak in {bad}: {leaks(pair[bad[0]]).group(0)}"})
                break
            if tok is not None:
                n = max(len(tok(question(pair[k])).input_ids) for k in ("vulnerable", "safe"))
                if n > MAX_TOKENS:
                    skipped.append({"case": case, "why": f"{n} tokens"})
                    break
            for k, label in (("vulnerable", 1), ("safe", 0)):
                progs.append({"case": case, "seed": seed, "label": label, "good_variant": pair["good_variant"], "code": pair[k]})
    return progs, skipped


def write_base_net(model_dir, out):
    import numpy as np
    from safetensors import safe_open
    out.mkdir(parents=True, exist_ok=True)
    with safe_open(str(model_dir / "model.safetensors"), "pt") as f:
        for k in f.keys():
            if re.search(r"layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$", k):
                np.save(out / f"{k}.npy", f.get_tensor(k).float().numpy())


def main():
    zip_path = V / "download/juliet-1.3.zip"
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != JULIET_SHA:
        sys.exit("juliet-1.3.zip does not match the expected SHA-256")
    from transformers import AutoTokenizer
    model_dir = ROOT / "assets/exp8/qwen35-text"
    tok = AutoTokenizer.from_pretrained(model_dir)
    progs, skipped = build_all(V / "download/cwe476", tok)
    (V / "eval").mkdir(parents=True, exist_ok=True)
    (V / "eval/programs.json").write_text(json.dumps(progs, indent=1))
    (V / "eval/skipped.json").write_text(json.dumps(skipped, indent=1))
    write_base_net(model_dir, V / "base_net")
    pairs = len({p["case"] for p in progs})
    print(json.dumps({"pairs": pairs, "programs": len(progs), "skipped": len(skipped)}))


if __name__ == "__main__":
    main()

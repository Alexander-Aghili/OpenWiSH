"""Side effects of a MoE edit on text the score never used: C headers, Markdown READMEs, reStructuredText docs.

usage: assets/venvs/qwen/bin/python -I scripts/moe_side_effects.py <edited net folder>
Prints the mean token loss of the base and the edited model per text kind (16 texts each, 256 tokens)."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moe", ROOT / "python/adapters/moe_worker.py")
moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe)
KINDS = {"c_headers": (sorted(Path("/usr/include").glob("*.h")), r"^(struct|typedef|extern) "),
         "markdown_readmes": (sorted(Path("/usr/share/doc").glob("*/*.md")), None),
         "rst_docs": (sorted(Path("/usr/lib/python3.12").rglob("*.rst")) + sorted(Path("/usr/share/doc").glob("*/*.rst")), None)}


def texts(tok, files, pattern):
    import re
    out = []
    for p in files:
        t = p.read_text(errors="ignore")
        m = re.search(pattern, t, re.M) if pattern else None
        ids = tok(t[m.start():] if m else t).input_ids
        if len(ids) >= 256:
            out.append(tok.decode(ids[:256]))
        if len(out) == 16:
            break
    return out


def main():
    base, edited = moe.load_net(moe.D / "base_net"), moe.load_net(sys.argv[1])
    model, tok = moe.load_model(base)
    sets = {k: texts(tok, f, p) for k, (f, p) in KINDS.items()}
    before = {k: sum(moe.run(model, tok, v)[0]) / len(v) for k, v in sets.items() if v}
    model, tok = moe.load_model(edited)
    after = {k: sum(moe.run(model, tok, v)[0]) / len(v) for k, v in sets.items() if v}
    print(json.dumps({k: {"texts": len(sets[k]), "base": round(before[k], 3), "edited": round(after[k], 3),
                          "rise": round(after[k] - before[k], 3)} for k in before}, indent=1))


if __name__ == "__main__":
    main()

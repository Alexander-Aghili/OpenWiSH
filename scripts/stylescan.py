"""Scan prose for the OpenWiSH style rules: em dashes, en dashes, semicolons and banned words, outside code.
usage: python3 stylescan.py FILE..."""
import re
import sys

BANNED = """can may just that very really literally actually certainly probably basically could maybe delve embark
enlightening esteemed craft crafting imagine realm game-changer unlock discover skyrocket abyss revolutionize disruptive
utilize utilizing tapestry illuminate unveil pivotal intricate elucidate hence furthermore however harness exciting
groundbreaking cutting-edge remarkable it navigating landscape stark testament moreover boost skyrocketing powerful
inquiries ever-evolving""".split()
PHRASES = ["shed light", "not alone", "in a world where", "dive deep", "remains to be seen", "glimpse into",
           "in summary", "in conclusion", "opened up", "in closing", "not just"]
ALLOWED = ["Open Weight-Space Harness", "Harness layer", "**Harness**", "| Harness |", "Read, Write, Harness",
           "Read, Write and Harness", "Harness:"]
WORD = re.compile(r"(?<![\w-])(" + "|".join(re.escape(w) for w in BANNED) + r")(?![\w-])", re.I)


def prose_lines(path, text):
    """(line number, prose) pairs with code, links targets and markup removed."""
    out = []
    fence = None
    yaml = path.endswith((".yml", ".yaml"))
    run_indent = None
    in_comment = False
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if yaml:
            ind = len(line) - len(line.lstrip())
            if run_indent is not None:
                if s and ind <= run_indent:
                    run_indent = None
                else:
                    continue
            m = re.match(r"\s*(-\s+)?run:\s*(.*)", line)
            if m:
                if m.group(2).strip() in ("|", ">", "|-", ">-"):
                    run_indent = ind
                continue
            if re.match(r"\s*(-\s+)?(uses|with|on|branches|runs-on|type|id|labels|python-version|toolchain):", line):
                continue
        else:
            m = re.match(r"\s*(`{3,}|~{3,})", line)
            if fence:
                if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                    fence = None
                continue
            if m:
                fence = m.group(1)
                continue
        # HTML comments
        if in_comment:
            if "-->" in line:
                line = line.split("-->", 1)[1]
                in_comment = False
            else:
                continue
        line = re.sub(r"<!--.*?-->", "", line)
        if "<!--" in line:
            line = line.split("<!--", 1)[0]
            in_comment = True
        line = re.sub(r"``.+?``", "", line)
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r" \1 ", line)
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        line = re.sub(r"https?://\S+", "", line)
        line = re.sub(r"\b[\w./-]+\.(md|toml|py|rs|sh|json|svg|html|yml|txt|npz|npy)\b", "", line)
        for a in ALLOWED:
            line = line.replace(a, "")
        out.append((n, line))
    return out


def main():
    hits = 0
    for path in sys.argv[1:]:
        text = open(path, encoding="utf-8").read()
        for n, line in prose_lines(path, text):
            found = []
            if "—" in line:
                found.append("em dash")
            if "–" in line:
                found.append("en dash")
            if ";" in line:
                found.append("semicolon")
            found += [f"word '{m.group(1)}'" for m in WORD.finditer(line)]
            low = line.lower()
            found += [f"phrase '{p}'" for p in PHRASES if p in low]
            if found:
                hits += len(found)
                print(f"{path}:{n}: {', '.join(found)}: {line.strip()[:140]}")
    print(f"{hits} hits in {len(sys.argv) - 1} files")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

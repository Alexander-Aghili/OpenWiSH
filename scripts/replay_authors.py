"""Turn the authors' recorded replies into a scripted-backend file, so a task can be replayed
through this harness (their code runs in our sandbox; their final answer goes to our scorer).
usage: replay_authors.py <authors' run dir (with turns/ or answer.md)> <out script file>"""
import os
import sys

run, out = sys.argv[1], sys.argv[2]
replies = []
tdir = os.path.join(run, "turns")
if os.path.isdir(tdir):
    for t in sorted(os.listdir(tdir)):
        p = os.path.join(tdir, t, "answer.md")
        if os.path.exists(p):
            replies.append(open(p).read())
else:
    for name in ("answer.md", "reply/answer.md"):
        p = os.path.join(run, name)
        if os.path.exists(p):
            replies.append(open(p).read())
            break
assert replies, "no recorded replies found"
for r in replies:
    assert not any(l == "=====" for l in r.splitlines()), "a reply contains the separator line"
open(out, "w").write("\n=====\n".join(replies) + "\n")
print(f"{len(replies)} replies -> {out}")

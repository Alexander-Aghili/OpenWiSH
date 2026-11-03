"""Copy the small, reportable files of every benchmark run from runs/ (git-ignored) into results/ (tracked):
scores, run state, final answers, the edit ledger, tool-call logs and audit summaries. Weights and kernels stay out.

usage: python3 scripts/save_results.py"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, DST = os.path.join(ROOT, "runs"), os.path.join(ROOT, "results")
KEEP = ("eval/score.json", "state.json", "final_answer.md", "rules.txt", "tool_calls.jsonl", "edits/ledger.jsonl",
        "audit_summary.json", "audit.jsonl", "final/edits_report.json", "final/editset/manifest.json", "loop.log",
        "provenance.json", "source_events.json")
MAX_BYTES = 2_000_000


def save_run(run, rel):
    copied = 0
    for k in KEEP:
        src = os.path.join(run, k)
        if os.path.isfile(src) and os.path.getsize(src) <= MAX_BYTES:
            dst = os.path.join(DST, rel, k)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            copied += 1
    return copied


def main():
    runs = 0
    for dirpath, _, files in os.walk(SRC):
        if "state.json" in files and "_superseded" not in dirpath:
            runs += bool(save_run(dirpath, os.path.relpath(dirpath, SRC)))
    for name in os.listdir(SRC):
        if name.endswith(".log") or name == "baselines.json":
            shutil.copyfile(os.path.join(SRC, name), os.path.join(DST, name))
    conf = os.path.join(SRC, "exp8_conformance")
    if os.path.isdir(conf):
        for d in os.listdir(conf):
            s = os.path.join(conf, d, "summary.json")
            if os.path.isfile(s):
                os.makedirs(os.path.join(DST, "exp8_conformance", d), exist_ok=True)
                shutil.copyfile(s, os.path.join(DST, "exp8_conformance", d, "summary.json"))
    print(f"saved {runs} runs to {DST}")


if __name__ == "__main__":
    main()

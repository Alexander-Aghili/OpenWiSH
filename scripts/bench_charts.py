"""Draw the benchmark charts as SVG from the run folders, plus docs/charts.html with every chart and its data table.

usage: python3 scripts/bench_charts.py
Writes docs/assets/charts/*.svg and docs/charts.html. Numbers come from scripts/bench_report.py, which reads runs/.
Colors: the dataviz reference palette, slots 1 to 3 (validated all-pairs in light and dark). Light and dark styles
follow prefers-color-scheme. Each mark carries a <title> tooltip. Animations stop under prefers-reduced-motion."""
import html
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/charts"
spec = importlib.util.spec_from_file_location("bench_report", ROOT / "scripts/bench_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

W, FONT = 760, "system-ui, -apple-system, 'Segoe UI', sans-serif"
STYLE = """
.c{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;font-family:%s}
@media (prefers-color-scheme:dark){.c{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
.bg{fill:var(--surface)}.t{fill:var(--ink);font-size:17px;font-weight:600}.st{fill:var(--ink2);font-size:12.5px}
.tk{fill:var(--muted);font-size:11px}.lb{fill:var(--ink);font-size:11.5px;font-weight:600}.lb2{fill:var(--ink2);font-size:11.5px}
.gl{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--axis);stroke-width:1}.ref{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 4}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.k1{stroke:var(--s1)}.k2{stroke:var(--s2)}.k3{stroke:var(--s3)}
.ring{stroke:var(--surface);stroke-width:2}.hollow{fill:var(--surface);stroke-width:2.5}
.hollow.s1{stroke:var(--s1)}.hollow.s2{stroke:var(--s2)}.hollow.s3{stroke:var(--s3)}.hatchline{stroke:var(--surface);stroke-width:2.2}.line{fill:none;stroke-width:2}.err{stroke:var(--ink2);stroke-width:1.5}
.bar{transform-box:fill-box;transform-origin:50%% 100%%;animation:grow .7s cubic-bezier(.2,.7,.2,1) both}
.hbar{transform-box:fill-box;transform-origin:0 50%%;animation:growx .7s cubic-bezier(.2,.7,.2,1) both}
.fade{animation:fade .6s ease-out both}
@keyframes grow{from{transform:scaleY(0)}}@keyframes growx{from{transform:scaleX(0)}}@keyframes fade{from{opacity:0}}
@media (prefers-reduced-motion:reduce){.bar,.hbar,.fade{animation:none}}
""" % FONT


def esc(s):
    return html.escape(str(s), quote=True)


class Svg:
    def __init__(self, height, title, subtitle, source):
        self.h, self.parts = height, []
        # hatching marks results made with probes (the agent ran the network); solid marks weights only
        pats = "".join(f'<pattern id="hatch-s{n}" patternUnits="userSpaceOnUse" width="7" height="7" '
                       f'patternTransform="rotate(45)"><rect width="7" height="7" class="s{n}"/>'
                       f'<line x1="0" y1="0" x2="0" y2="7" class="hatchline"/></pattern>' for n in (1, 2, 3))
        self.add(f"<defs>{pats}</defs>")
        self.add(f'<rect class="bg" width="{W}" height="{height}" rx="12"/>')
        self.text(24, 34, title, "t")
        self.text(24, 54, subtitle, "st")
        self.text(24, height - 14, source, "tk")

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, cls, anchor="start", extra=""):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>{esc(s)}</text>')

    def swatch(self, x, y, slot, probe=False, shape="bar", label=""):
        if shape == "dot":
            cls = f"s{slot} hollow" if probe else f"s{slot}"
            self.add(f'<circle cx="{x + 6}" cy="{y - 3}" r="5.5" class="{cls}"><title>{esc(label)}</title></circle>')
        elif probe:
            self.add(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="3" fill="url(#hatch-s{slot})"/>')
        else:
            self.add(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="3" class="s{slot}"/>')

    def legend(self, x, y, items, shape="bar"):
        """items: (label, slot) or (label, slot, probe). Swatch plus text, in series order."""
        for item in items:
            label, slot, probe = (item + (False,))[:3]
            self.swatch(x, y, slot, probe, shape, label)
            self.text(x + 18, y + 1, label, "lb2")
            x += 18 + 7.2 * len(label) + 22

    def access_key(self, items):
        """A line above the source naming the access of each mark style, in words. items: (text, probe, shape)."""
        x, y = 24, self.h - 34
        for text, probe, shape in items:
            if shape == "dot":
                cls = "hollow s1" if probe else "s1"
                self.add(f'<circle cx="{x + 6}" cy="{y - 3}" r="5" class="{cls}"><title>{esc(text)}</title></circle>')
            elif probe:
                self.add(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="3" fill="url(#hatch-s1)"/>')
            else:
                self.add(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="3" class="s1"/>')
            self.text(x + 18, y + 1, text, "tk")
            x += 18 + 6.2 * len(text) + 22

    def bar(self, x, y0, w, h, slot, tip, delay=0.0, probe=False):
        """A vertical bar anchored at baseline y0, with a 4px rounded data end."""
        if h < 0.5:
            self.add(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x + w:.1f}" y2="{y0:.1f}" class="k{slot}" stroke-width="2"><title>{esc(tip)}</title></line>')
            return
        r = min(4, h, w / 2)
        y = y0 - h
        d = f"M{x:.1f},{y0:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y0:.1f}Z"
        if probe:
            self.add(f'<path d="{d}" class="bar" fill="url(#hatch-s{slot})" style="animation-delay:{delay:.2f}s"><title>{esc(tip)}</title></path>')
        else:
            self.add(f'<path d="{d}" class="s{slot} bar" style="animation-delay:{delay:.2f}s"><title>{esc(tip)}</title></path>')

    def dot(self, x, y, slot, tip, delay=0.0, probe=False):
        cls = f"s{slot} hollow fade" if probe else f"s{slot} ring fade"
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" class="{cls}" style="animation-delay:{delay:.2f}s"><title>{esc(tip)}</title></circle>')

    def yaxis(self, x0, x1, y0, y1, ticks, fmt):
        """Horizontal gridlines with labels. y0 is the baseline (value 0 or the first tick)."""
        lo, hi = ticks[0], ticks[-1]
        for t in ticks:
            y = y0 - (t - lo) / (hi - lo) * (y0 - y1)
            self.add(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="{"ax" if t == lo else "gl"}"/>')
            self.text(x0 - 8, y + 4, fmt(t), "tk", "end")
        return lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)

    def render(self, label):
        body = "\n".join(self.parts)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" width="{W}" height="{self.h}" '
                f'role="img" aria-label="{esc(label)}" class="c"><title>{esc(label)}</title><style>{STYLE}</style>\n{body}\n</svg>\n')


def grouped_bars(svg, groups, series, x0, x1, y_of, y0, values, tips, labels=True, missing="not run", label_top=None):
    """groups: names on the x axis. series: (label, slot) or (label, slot, probe). values[g][s] is a number or None.
    probe may also be a list with one flag per group."""
    gw = (x1 - x0) / len(groups)
    bw = min(34, (gw - 28) / len(series))
    for gi, g in enumerate(groups):
        start = x0 + gi * gw + (gw - bw * len(series) - 2 * (len(series) - 1)) / 2
        for si, item in enumerate(series):
            name, slot, probe = (tuple(item) + (False,))[:3]
            probe = probe[gi] if isinstance(probe, list) else probe
            v = values[gi][si]
            x = start + si * (bw + 2)
            if v is None:
                svg.text(x + bw / 2, y0 - 6, missing, "tk", "middle", ' transform="rotate(-90 %.1f %.1f)" dx="22"' % (x + bw / 2, y0 - 6))
                continue
            svg.bar(x, y0, bw, y0 - y_of(v), slot, tips[gi][si], delay=0.05 * (gi * len(series) + si), probe=probe)
            if labels:
                top = min(y_of(v), label_top[gi][si]) if label_top else y_of(v)
                svg.text(x + bw / 2, top - 6, labels(v) if callable(labels) else v, "lb", "middle")
        svg.text(x0 + gi * gw + gw / 2, y0 + 20, g, "lb2", "middle")


def chart_tools(data):
    models = [m for m in data]
    series = [("WSPBench B: weight algebra", 2), ("WSPBench C: + practice observations", 3, True), ("OpenWiSH sandbox", 1)]
    svg = Svg(400, "WSPBench, tools against tools",
              "Exact solves out of 12 tasks (6 pilot, 6 held out), one attempt each. A run that fails its audit scores 0.",
              "Source: runs/bench/wspbench-*, scripts/bench_charts.py. B and C run in WSPBench's own Inspect loop, unchanged.")
    svg.legend(24, 82, series)
    y0, y1 = 330, 112
    y_of = svg.yaxis(64, W - 24, y0, y1, [0, 3, 6, 9, 12], str)
    keys = ("B", "C", "OW")
    vals = [[data[m][k] for k in keys] for m in models]
    tips = [[f"{m}, {series[i][0]}: {data[m][k]} of 12 exact" for i, k in enumerate(keys)] for m in models]
    grouped_bars(svg, models, series, 72, W - 24, y_of, y0, vals, tips)
    svg.access_key([("solid: weights only, the network never runs", False, "bar"),
                    ("hatched: with probes (C runs the network on practice inputs)", True, "bar")])
    return svg.render("WSPBench exact solves: WSPBench B and C against the OpenWiSH sandbox")


def chart_rounds(rows):
    """rows: {model: [audited read agreement per round]}."""
    svg = Svg(380, "Tuning rounds, scored after audit",
              "OpenWiSH sandbox, 6 pilot tasks, read agreement. Failed audits score 0. Round 4 added the activation guard.",
              "Source: runs/bench/wspbench-OW-*-r[1-5]-*, audits by gpt-6.1-sol. The config froze after round 5.")
    names = ["r1 instructions", "r2 task JSON", "r3 format check", "r4 activation guard", "r5 context"]
    y0, y1, x0, x1 = 316, 96, 64, W - 150
    y_of = svg.yaxis(x0, x1, y0, y1, [0, 25, 50, 75, 100], lambda t: f"{t} %")
    xs = [x0 + 30 + i * (x1 - x0 - 60) / 4 for i in range(5)]
    for x, n in zip(xs, names):
        svg.text(x, y0 + 20, n, "tk", "middle")
    ends = []
    for si, (model, vals) in enumerate(rows.items()):
        slot = si + 1
        kept = [(x, y_of(100 * v), v, n) for x, v, n in zip(xs, vals, names) if v is not None]
        svg.add(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in kept)}" class="line k{slot} fade"/>')
        for x, y, v, n in kept:
            svg.dot(x, y, slot, f"{model}, {n}: {100 * v:.1f} % read agreement after audit", 0.1 * si)
        x, y, v, _ = kept[-1]
        ends.append([y + 4, x + 14, f"{model} {100 * v:.0f} %"])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels 15px apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for y, x, label in ends:
        svg.text(x, y, label, "lb")
    svg.legend(24, 80, [(m, i + 1) for i, m in enumerate(rows)], shape="dot")
    svg.access_key([("weights only: Python over the weights, the network never runs", False, "dot")])
    return svg.render("Audited read agreement per tuning round")


def chart_agentic(rows):
    kinds = {"A": ("prompt only", 3, False), "WO": ("weights only", 1, False), "P": ("weights + probe", 1, True),
             "HyVE": ("HyVE, published", 2, True)}
    marks = [r for r in rows if r["kind"] in kinds]
    marks.sort(key=lambda r: (r["kind"] == "HyVE", r["model"], list(kinds).index(r["kind"])))
    refs = [r for r in rows if r["kind"] == "baseline"]
    h = 230 + 30 * len(marks)
    svg = Svg(h, "AgenticInterpBench: tag accuracy on 163 circuit components",
              "84 InterpBench circuits, 5-class taxonomy, components pooled. Dashed lines: fixed baselines with no model.",
              "Source: runs/bench/agentic-interp-*, HyVE scores from arXiv 2606.24026. Runs that fail audit score 0.")
    svg.legend(24, 82, [("OpenWiSH, weights only", 1), ("OpenWiSH, weights + probe", 1, True), ("prompt only", 3),
                        ("HyVE, published (probes)", 2, True)], shape="dot")
    x0, x1, top = 300, W - 70, 154
    lo, hi = 40, 85
    x_of = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    bottom = top + 30 * len(marks)
    for t in range(40, 90, 10):
        svg.add(f'<line x1="{x_of(t):.1f}" y1="{top - 8}" x2="{x_of(t):.1f}" y2="{bottom}" class="gl"/>')
        svg.text(x_of(t), bottom + 16, f"{t} %", "tk", "middle")
    for k, r in enumerate(refs):
        x = x_of(100 * r["accuracy"])
        svg.add(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{bottom}" class="ref"><title>{esc(r["label"])}: {100 * r["accuracy"]:.1f} %</title></line>')
        svg.text(x, top - 14 - 14 * (len(refs) - 1 - k), f'{"always MAPPER" if "always" in r["label"] else "MLP MAPPER, attention AGGREGATOR"} {100 * r["accuracy"]:.1f} %', "tk", "middle")
    for i, r in enumerate(marks):
        y = top + 14 + 30 * i
        label, slot, probe = kinds[r["kind"]]
        partial = f" ({r['cases']}/84, running)" if r["cases"] < 84 else ""
        svg.text(x0 - 14, y + 4, f'{r["model"]}, {label}{partial}', "lb2", "end")
        v = 100 * r["accuracy"]
        svg.add(f'<line x1="{x_of(lo):.1f}" y1="{y}" x2="{x_of(v):.1f}" y2="{y}" class="gl"/>')
        svg.dot(x_of(v), y, slot, f'{r["model"]}, {r["label"]}: {v:.1f} % over {r["cases"]} cases ({r["audit"]})', 0.04 * i, probe=probe)
        svg.text(x_of(v) + 12, y + 4, f"{v:.1f} %", "lb")
    svg.access_key([("filled: no network runs (prompt only or weights only)", False, "dot"),
                    ("hollow: with probes (the network runs within a budget)", True, "dot")])
    return svg.render("AgenticInterpBench tag accuracy by model and access")


def chart_qwen(rows):
    svg = Svg(400, "Qwen3.5-0.8B on MATH-500: the published edits, replayed",
              "Left: accuracy, published (4 samples) against the OpenWiSH replay (1 sample, bar shows ±1 standard error).",
              "Source: runs/exp8_conformance/*/summary.json. Right: share of OpenWiSH answers cut at the 32,768-token limit.")
    svg.legend(24, 82, [("published", 2), ("OpenWiSH replay", 1)])
    names = {"base": "base", "p10c": "10 passes", "p25c": "25 passes", "p50c": "50 passes"}
    y0, y1 = 326, 112
    y_of = svg.yaxis(64, 470, y0, y1, [0, 15, 30, 45, 60], lambda t: f"{t} %")
    vals = [[100 * r["published"], 100 * r["score"]] for r in rows]
    tips = [[f'{names[r["config"]]}, published: {100 * r["published"]:.1f} %',
             f'{names[r["config"]]}, OpenWiSH: {100 * r["score"]:.1f} ± {100 * r["se"]:.1f} %'] for r in rows]
    tops = [[y_of(100 * r["published"]), y_of(100 * (r["score"] + r["se"]))] for r in rows]
    probe = [r["config"] != "base" for r in rows]
    grouped_bars(svg, [names[r["config"]] for r in rows], [("published", 2, probe), ("OpenWiSH", 1, probe)], 72, 470, y_of, y0, vals, tips,
                 labels=lambda v: f"{v:.1f}", label_top=tops)
    gw = (470 - 72) / len(rows)
    for gi, r in enumerate(rows):  # error bars on the OpenWiSH bars
        bw = min(34, (gw - 28) / 2)
        cx = 72 + gi * gw + (gw - bw * 2 - 2) / 2 + bw + 2 + bw / 2
        a, b = y_of(100 * (r["score"] - r["se"])), y_of(100 * (r["score"] + r["se"]))
        svg.add(f'<line x1="{cx:.1f}" y1="{a:.1f}" x2="{cx:.1f}" y2="{b:.1f}" class="err fade"/>')
    x0, x1 = 540, W - 24
    svg.text(x0, 104, "Cut at the token limit", "lb2")
    y_cut = svg.yaxis(x0, x1, y0, y1, [0, 50, 100], lambda t: f"{t} %")
    bw = (x1 - x0 - 12) / len(rows) - 6
    for i, r in enumerate(rows):
        x = x0 + 10 + i * (bw + 6)
        svg.bar(x, y0, bw, y0 - y_cut(100 * r["cut"]), 1, f'{names[r["config"]]}: {100 * r["cut"]:.1f} % cut at the limit', 0.05 * i,
                probe=r["config"] != "base")
        svg.text(x + bw / 2, y_cut(100 * r["cut"]) - 6, f'{100 * r["cut"]:.0f}', "lb", "middle")
        svg.text(x + bw / 2, y0 + 20, names[r["config"]].replace(" passes", ""), "tk", "middle")
    svg.access_key([("solid: the base model, no edit", False, "bar"),
                    ("hatched: edits made with probes (10, 25 or 50 forward passes)", True, "bar")])
    return svg.render("Qwen MATH-500 accuracy, published against the OpenWiSH replay, and the share cut at the limit")


def pool_repeats(scores):
    """(mean of the run means, standard error, runs). One run keeps its own standard error."""
    means = [x["mean"] for x in scores]
    if not means:
        return None
    if len(means) == 1:
        return means[0], scores[0].get("se", 0.0), 1
    m = sum(means) / len(means)
    sd = (sum((v - m) ** 2 for v in means) / (len(means) - 1)) ** 0.5
    return m, sd / len(means) ** 0.5, len(means)


def blog_data():
    """Claude Opus 5.5 through OpenWiSH against the published agent: (panel, unit, published, ours, ours_se)."""
    logs = ROOT / "assets/lm_understand-logs/exp5_write_from_scratch/eval/minatar_eval.json"
    pub5 = report.load(logs) or {}
    out = []
    kin = report.load(ROOT / "runs/bench/claude-opus-5-5/exp4-task/eval/score.json")
    if kin:
        out.append(("Kinetix car, fix a policy", "successes of 1,024", 1024, kin["success"], 0))
    for g in ("seaquest", "asterix", "breakout"):
        if g not in pub5:
            continue
        # repeats at the task's own effort (xhigh) take precedence over a single run at another effort
        reps = [report.load(p) for p in sorted((ROOT / "runs/bench/claude-opus-5-5-xhigh").glob(f"exp5-{g}-r*/eval/score.json"))]
        pooled = pool_repeats([r for r in reps if r])
        if pooled:
            m, se, n = pooled
            out.append((f"MinAtar {g.capitalize()}, write from zero", f"mean return, {n} runs at xhigh", pub5[g]["hand"]["mean"], m, se))
            continue
        s = report.load(ROOT / f"runs/bench/claude-opus-5-5/exp5-{g}/eval/score.json")
        if s:
            out.append((f"MinAtar {g.capitalize()}, write from zero", "mean return", pub5[g]["hand"]["mean"], s["mean"], s.get("se", 0)))
    return out


def chart_blog(panels):
    svg = Svg(380, "Blog environments: Claude Opus 5.5 through OpenWiSH against the published agent",
              "One reply with an edits block. Single runs unless a panel names repeats. Each panel has its own scale.",
              "Source: runs/bench/claude-opus-5-5*/exp4-task, exp5-*. Published agent: GPT-6 Astra (RikiyaT/lm_understand-logs).")
    svg.legend(24, 82, [("published agent", 2), ("Claude Opus 5.5 through OpenWiSH", 1)])
    pw = (W - 48) / len(panels)
    y0, y1 = 300, 136
    for i, (name, unit, pub, ours, se) in enumerate(panels):
        x0 = 24 + i * pw
        hi = max(pub, ours) * 1.18
        y_of = lambda v: y0 - v / hi * (y0 - y1)
        svg.add(f'<line x1="{x0 + 10:.1f}" y1="{y0}" x2="{x0 + pw - 10:.1f}" y2="{y0}" class="ax"/>')
        svg.text(x0 + pw / 2, 112, name.split(",")[0], "lb2", "middle")
        svg.text(x0 + pw / 2, 126, unit, "tk", "middle")
        bw = 40
        for j, (v, slot, who) in enumerate(((pub, 2, "published"), (ours, 1, "OpenWiSH"))):
            x = x0 + pw / 2 - bw - 1 + j * (bw + 2)
            svg.bar(x, y0, bw, y0 - y_of(v), slot, f"{name}, {who}: {v:,.1f} {unit}", 0.06 * (2 * i + j))
            svg.text(x + bw / 2, y_of(v) - 6, f"{v:,.0f}" if v >= 100 else f"{v:.1f}", "lb", "middle")
        win = "OpenWiSH ahead" if ours > pub else ("tie" if ours == pub else "published ahead")
        svg.text(x0 + pw / 2, y0 + 20, win, "tk", "middle")
    svg.access_key([("all bars: no network runs (one reply; the Kinetix prompt also shows a recorded failure episode)", False, "bar")])
    return svg.render("Blog environments, published agent against Claude Opus 5.5 through OpenWiSH")


TIERS = (("native", "native: weights in the prompt", 3, False), ("read", "read: weights in Python", 1, False),
         ("probe", "probe: Python + 48 probe inputs", 2, True))


def chart_decompile(rows):
    """rows: {network: {tier: score or None}}."""
    series = [(label, slot, probe) for _, label, slot, probe in TIERS]
    svg = Svg(400, "Decompile ladder: one network, three access levels",
              "Score = behaviour on all 1,024 inputs x interchange-intervention accuracy x completeness. 1.0 is an exact decompile.",
              "Source: results/bench/decompile-claude-opus-5-5, Claude Opus 5.5 at high effort, one run per bar, unaudited.")
    svg.legend(24, 82, series)
    y0, y1 = 326, 112
    y_of = svg.yaxis(64, W - 24, y0, y1, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    nets = list(rows)
    vals = [[rows[n].get(t) for t, _, _, _ in TIERS] for n in nets]
    tips = [[f"{n}, {label}: {'not run' if rows[n].get(t) is None else f'{rows[n][t]:.3f}'}" for t, label, _, _ in TIERS] for n in nets]
    grouped_bars(svg, nets, series, 72, W - 24, y_of, y0, vals, tips, labels=lambda v: f"{v:.2f}")
    svg.access_key([("weights only: the network never runs", False, "bar"), ("with probes: the agent ran the network within a budget", True, "bar")])
    return svg.render("Decompile ladder scores by network and access level")


def decompile_data():
    out = {}
    for d in sorted((ROOT / "results/bench").glob("decompile-*/decompile-*")):
        s = report.load(d / "eval/score.json")
        if not s:
            continue
        net, tier = d.name[len("decompile-"):].rsplit("-", 1)
        out.setdefault(net, {t[0]: None for t in TIERS})[tier] = s["score"]
    return out


# Conformance: the published runs replayed through OpenWiSH (docs/benchmarks.md, Conformance). The Qwen replays have
# their own chart. (name, unit, published, OpenWiSH)
CONFORMANCE = [
    ("Game MLPs: grades", "correct of 4", 4, 4),
    ("Diffusion MLPs: grades", "correct of 3", 3, 3),
    ("Char LMs: copying heads", "recall", 1.0, 1.0),
    ("Kinetix car: published edit", "successes of 1,024", 1024, 1024),
    ("MinAtar Breakout: published reply", "mean return", 517.4, 502.9),
    ("MinAtar Seaquest: published reply", "mean return", 53.1, 53.2),
    ("MinAtar Asterix: published reply", "mean return", 53.2, 52.9),
    ("9x9 Go: budget-0 network rebuilt", "tensors bit-identical of 14", 14, 14),
    ("9x9 Go: budget-50 vs same-size PPO", "win rate %", 74.15, 74.15),
    ("Gemma pairs: grades", "correct of 3", 3, 3),
]


def chart_conformance(rows):
    """rows: (name, unit, published, ours). Dots at OpenWiSH as a share of the published value."""
    h = 190 + 30 * len(rows)
    svg = Svg(h, "Conformance: the published runs, replayed through OpenWiSH",
              "Each dot is OpenWiSH's result as a share of the published one. 100 % on the dashed line is an exact match.",
              "Source: docs/benchmarks.md (Conformance), tests/test_scorers.py, test_adapters.py, test_kinetix.py, the Go head-to-head.")
    x0, x1, top = 330, W - 150, 112
    lo, hi = 90, 110
    x_of = lambda v: x0 + (max(lo, min(hi, v)) - lo) / (hi - lo) * (x1 - x0)
    bottom = top + 30 * len(rows)
    for t in range(90, 111, 5):
        svg.add(f'<line x1="{x_of(t):.1f}" y1="{top - 8}" x2="{x_of(t):.1f}" y2="{bottom}" class="{"ref" if t == 100 else "gl"}"/>')
        svg.text(x_of(t), bottom + 16, f"{t} %", "tk", "middle")
    for i, (name, unit, pub, ours) in enumerate(rows):
        y = top + 14 + 30 * i
        share = 100 * ours / pub if pub else 100.0
        svg.text(x0 - 14, y + 4, name, "lb2", "end")
        svg.dot(x_of(share), y, 1, f"{name}: published {pub:g}, OpenWiSH {ours:g} ({unit})", 0.04 * i)
        svg.text(x1 + 14, y + 4, f"{ours:g} of {pub:g}", "lb")
    return svg.render("Conformance: OpenWiSH replays as a share of the published results")


def rounds_data():
    out = {}
    for model in report.CODEX:
        vals = [(report.found(cond, model) or [None])[0] for cond, _ in report.ROUNDS]
        if any(v is not None for v in vals):
            out[model] = vals
    return out


def table_html(head, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    tools, rounds, agentic, qwen, blog = report.tools_data(), rounds_data(), report.agentic_data(), report.qwen_data(), blog_data()
    decompile = decompile_data()
    charts = [("conformance", chart_conformance(CONFORMANCE), table_html(["Check", "Unit", "Published", "OpenWiSH"],
               [[n, u, f"{p:g}", f"{o:g}"] for n, u, p, o in CONFORMANCE]))]
    if decompile:
        charts.append(("decompile-ladder", chart_decompile(decompile), table_html(["Network", "native", "read", "probe"],
                       [[n, *("not run" if v is None else f"{v:.3f}" for v in (r["native"], r["read"], r["probe"]))] for n, r in decompile.items()])))
    if tools:
        charts.append(("wspbench-tools", chart_tools(tools), table_html(["Model", "WSPBench B", "WSPBench C", "OpenWiSH sandbox"],
                       [[m, *("not run" if v is None else f"{v} / 12" for v in (r["B"], r["C"], r["OW"]))] for m, r in tools.items()])))
    if rounds:
        charts.append(("wspbench-rounds", chart_rounds(rounds), table_html(["Model", "r1", "r2", "r3", "r4", "r5"],
                       [[m, *("-" if v is None else f"{100 * v:.1f} %" for v in vals)] for m, vals in rounds.items()])))
    if agentic:
        charts.append(("agentic-interp", chart_agentic(agentic), table_html(["Model", "Condition", "Cases", "Tag accuracy", "Audit"],
                       [[r["model"], r["label"], r["cases"], f"{100 * r['accuracy']:.1f} %", r["audit"]] for r in agentic])))
    if qwen:
        charts.append(("qwen-math", chart_qwen(qwen), table_html(["Config", "Published", "OpenWiSH", "Cut at the limit"],
                       [[r["config"], f"{100 * r['published']:.1f} %", f"{100 * r['score']:.1f} ± {100 * r['se']:.1f} %", f"{100 * r['cut']:.1f} %"] for r in qwen])))
    if blog:
        charts.append(("blog-envs", chart_blog(blog), table_html(["Environment", "Unit", "Published", "Claude through OpenWiSH"],
                       [[n, u, f"{p:,.1f}", f"{o:,.1f}"] for n, u, p, o, _ in blog])))
    for name, svg, _ in charts:
        (OUT / f"{name}.svg").write_text(svg)
    page = "\n".join(f'<section id="{n}">{s}<details><summary>Data table</summary>{t}</details></section>' for n, s, t in charts)
    (ROOT / "docs/charts.html").write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenWiSH benchmark charts</title><style>
body{{margin:0;padding:32px 16px;background:#f9f9f7;color:#0b0b0b;font-family:{FONT}}}
@media (prefers-color-scheme:dark){{body{{background:#0d0d0d;color:#f0efec}}}}
main{{max-width:{W}px;margin:0 auto}}section{{margin:0 0 40px}}svg{{max-width:100%;height:auto;display:block}}
h1{{font-size:22px}}details{{margin:10px 4px}}table{{border-collapse:collapse;font-size:13px;margin-top:8px}}
td,th{{padding:4px 10px;border-bottom:1px solid #c3c2b7;text-align:left}}
</style></head><body><main><h1>OpenWiSH benchmark charts</h1>
<p>Generated by <code>scripts/bench_charts.py</code> from the run folders. Hover a mark for its value. Each chart has a data table.</p>
{page}</main></body></html>
""")
    print(json.dumps({"ok": True, "charts": [n for n, _, _ in charts], "out": os.fspath(OUT)}))


if __name__ == "__main__":
    build()

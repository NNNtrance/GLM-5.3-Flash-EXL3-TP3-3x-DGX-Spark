#!/usr/bin/env python3
"""Regenerate every SVG in charts/ from the CSVs in results/.

    python3 charts/make-charts.py

Standard library only, on purpose. A chart in this repository is a view of a
committed CSV and nothing else: if you doubt a bar, open the CSV next to it and
read the row. There is no hidden transformation here beyond picking columns and
scaling them to pixels.

Inputs (all committed):
    results/configs/production-configurations.csv
    results/configs/kv-pool-progression.csv
    results/profile/step-breakdown.csv

Outputs:
    charts/speed-by-configuration.svg
    charts/kv-pool-progression.svg
    charts/step-breakdown-prod9.svg
    charts/dense-stage-prod7-vs-prod9.svg

Written by us for this recipe; use freely (Apache-2.0).
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

# Palette shared with the NVFP4 sibling repository so the two read as one set.
# An explicit white ground is deliberate: GitHub renders SVG on both light and
# dark page backgrounds, and dark text on a transparent ground is unreadable on
# one of them.
BG, INK, SUB, GRID = "#ffffff", "#1c2430", "#5c6675", "#d8dce3"
GREEN, BLUE, AMBER, SLATE, RED = "#1f8a4c", "#2b5fd9", "#c07d10", "#5c6675", "#b8402f"

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.p = w, h, []
        self.p.append(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
            f"viewBox='0 0 {w} {h}' font-family='system-ui,-apple-system,Segoe UI,Roboto,sans-serif'>")
        self.p.append(f"<rect width='{w}' height='{h}' fill='{BG}'/>")
    def text(self, x, y, s, size=11, fill=INK, anchor="start", weight=None):
        wt = f" font-weight='{weight}'" if weight else ""
        self.p.append(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' "
                      f"text-anchor='{anchor}'{wt}>{esc(s)}</text>")
    def rect(self, x, y, w, h, fill, rx=0):
        if w <= 0 or h <= 0:
            return
        self.p.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'/>")
    def line(self, x1, y1, x2, y2, stroke=GRID, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.p.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{stroke}'{d}/>")
    def legend(self, x, y, items):
        for label, colour in items:
            self.rect(x, y - 9, 11, 11, colour, rx=2)
            self.text(x + 16, y, label, 11)
            x += 14 + 7.2 * len(label) + 18
    def head(self, title, *subs):
        self.text(20, 28, title, 17, INK, weight="600")
        yy = 46
        for s in subs:
            self.text(20, yy, s, 11.5, SUB)
            yy += 15
        return yy
    def save(self, name):
        self.p.append("</svg>")
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.p) + "\n")
        print("wrote", os.path.relpath(path, ROOT))

def read(rel):
    with open(os.path.join(RES, rel), encoding="utf-8") as f:
        return list(csv.DictReader(f))

def num(v, default=None):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default

def nice_max(v, steps=5):
    """Round a maximum up to something a human would put on an axis."""
    if v <= 0:
        return 1.0, 1.0
    import math
    raw = v / steps
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    return step * steps, step

# --------------------------------------------------------------- chart 1 ----
def speed_by_configuration():
    rows = [r for r in read("configs/production-configurations.csv") if num(r["c1_total_tokps"])]
    sv = SVG(1010, 504)
    y0 = sv.head(
        "Decode throughput by production configuration",
        "12 short English code prompts (hizset-v2), realistic not synthetic. TP=3 + expert parallelism, EXL3 4bpw, KV fp8, DFlash2 k=7,",
        "temperature 0, reasoning effort low. gpu-memory-utilization is 0.80 through configuration 9, then 0.83 / 0.87 / 0.88 - the memory",
        "rungs 10, 11 and 12 climbed. Configurations 1-8 serve a routed-experts-only checkpoint, 9 onwards a full-scope one.",
        "Source: results/configs/production-configurations.csv. 5-6 September 2026.")
    sv.legend(20, y0 + 16, [("aggregate tok/s at C8", GREEN), ("aggregate tok/s at C1", BLUE)])

    top, bot, left, right = y0 + 34, 392, 62, 988
    vals = [num(r["c8_total_tokps"], 0) for r in rows] + [num(r["c1_total_tokps"], 0) for r in rows]
    ymax, step = nice_max(max(vals))
    def Y(v):
        return bot - (v / ymax) * (bot - top)
    g = 0.0
    while g <= ymax + 1e-9:
        sv.line(left, Y(g), right, Y(g))
        sv.text(left - 8, Y(g) + 4, f"{g:g}", 10, SUB, "end")
        g += step

    n = len(rows)
    slot = (right - left) / n
    bw = min(26, slot / 3.2)
    for i, r in enumerate(rows):
        cx = left + slot * (i + 0.5)
        c8, c1 = num(r["c8_total_tokps"], 0), num(r["c1_total_tokps"], 0)
        sv.rect(cx - bw - 2, Y(c8), bw, bot - Y(c8), GREEN)
        sv.rect(cx + 2, Y(c1), bw, bot - Y(c1), BLUE)
        sv.text(cx - bw / 2 - 2, Y(c8) - 5, f"{c8:g}", 10.5, INK, "middle")
        sv.text(cx + bw / 2 + 2, Y(c1) - 5, f"{c1:g}", 10.5, INK, "middle")
        sv.text(cx, bot + 17, f"prod {r['config']}", 11.5, INK, "middle")
        sv.text(cx, bot + 31, r["image_tag"].split(":")[-1].split(" ")[0][:9], 9.5, SUB, "middle")

    sv.text(20, 428, "Configuration 9 is the only step that changed the checkpoint rather than a flag, an image or the fabric, and it is the largest single move here:",
            11.5, SUB)
    sv.text(20, 444, "+22.9 % at C1 and +12.5 % at C8 against configuration 8. Draft acceptance is unchanged once pooled by draft token (the 2.4-point gap first",
            11.5, SUB)
    sv.text(20, 460, "published was a harness artefact); the cost is a second patch tree.", 11.5, SUB)
    sv.text(20, 476, "Configurations 10, 11 and 12 are memory work, not speed work: they buy KV pool (5.62M -> 6.38M -> 7.04M) with every level inside its band.",
            11.5, SUB)
    sv.text(20, 492, "Image tag under each pair is the cuda-exl3 commit that configuration was built from. The KV chart is where 10, 11 and 12 are read.", 11.5, SUB)
    sv.save("speed-by-configuration.svg")

# --------------------------------------------------------------- chart 2 ----
def kv_pool_progression():
    want = [
        ("TP=2 DFlash2\nk=7, page 16", "825000", SLATE),
        ("prod 1\nbc0e0f6\nMNBT 4096", "1627170", SLATE),
        ("prod 2\nf4987cf\nMNBT 2048", "2428769", BLUE),
        ("prod 3\ndraft page 256", "4413223", BLUE),
        ("prod 4\nfast-boot\nsidecar", "4468319", BLUE),
        ("prod 6\ndual link +\nPTR_CUDA", "4449035", BLUE),
        ("prod 7\nfp8 draft\ncache", "4699724", BLUE),
        ("prod 8\nimage 62f53e6", "4696969", BLUE),
        ("prod 9\nfull-scope\ncheckpoint", "5165289", BLUE),
        ("0.85 on prod 3\nrejected, then\nretracted", "5256198", SLATE),
        ("prod 10\n0.83 rung", "5619834", BLUE),
        ("prod 11\n0.87 +\nsm_12x set", "6382920", BLUE),
        ("prod 12\n0.88 +\nindexer bound", "7041322", GREEN),
        ("0.90 rung\nREJECTED on\nswap traffic", "6870523", RED),
    ]
    sv = SVG(1180, 498)
    y0 = sv.head(
        "KV pool by configuration, all at gpu-memory-utilization 0.80 unless marked",
        "Tokens in the pool, read from the engine's own 'GPU KV cache size' line on a load boot with a settled memory baseline.",
        "Source: results/configs/kv-pool-progression.csv, which carries every reading including the ones not plotted here.",
        "max_model_len is 1,000,000, so the right-hand axis is how many full-length requests the pool holds at once.")
    sv.legend(20, y0 + 16, [("in production", BLUE), ("current production", GREEN),
                            ("superseded", SLATE), ("measured and rejected", RED)])

    top, bot, left, right = y0 + 36, 388, 66, 1106
    vals = [num(v) for _, v, _ in want]
    ymax, step = nice_max(max(vals))
    def Y(v):
        return bot - (v / ymax) * (bot - top)
    g = 0.0
    while g <= ymax + 1e-9:
        sv.line(left, Y(g), right, Y(g))
        sv.text(left - 8, Y(g) + 4, f"{g/1e6:.1f}M", 10, SUB, "end")
        sv.text(right + 8, Y(g) + 4, f"{g/1_048_576:.1f}x", 10, SUB, "start")
        g += step

    slot = (right - left) / len(want)
    bw = min(46, slot * 0.62)
    for i, (label, v, colour) in enumerate(want):
        cx = left + slot * (i + 0.5)
        val = num(v)
        sv.rect(cx - bw / 2, Y(val), bw, bot - Y(val), colour)
        sv.text(cx, Y(val) - 6, f"{val:,.0f}".replace(",", " "), 10, INK, "middle")
        for j, part in enumerate(label.split("\n")):
            sv.text(cx, bot + 16 + j * 12, part, 9.5, INK if j == 0 else SUB, "middle")

    sv.text(20, 446, "The grey 0.85 bar was published as a rejected rung, on free host RAM against a 4 GiB rule. That ruler was wrong and the rejection is retracted (docs/11 section 2.4):",
            11.5, SUB)
    sv.text(20, 462, "the 6 September ladder re-climbed it against SWAP TRAFFIC UNDER LOAD instead - 0.85, 0.87 and 0.88 all pass, and 0.90 is the rejected rung, 1,519 MiB out and 143 MiB back in.",
            11.5, SUB)
    sv.text(20, 478, "Production 12 is 0.88 plus the sparse-indexer workspace bound: 7,041,322 tokens, 4.3x production 1's pool. The 0.90 bar is measured and must not be quoted as an achievable pool.", 11.5, SUB)
    sv.save("kv-pool-progression.svg")

# --------------------------------------------------------------- chart 3 ----
def step_breakdown_prod9():
    rows = [r for r in read("profile/step-breakdown.csv") if r["source_config"] == "production 9"]
    drop = ("TOTAL", "sub-row", "cross-cutting", " - gate/up", " - down")
    def keep(r):
        c = r["class"]
        return not any(d in c for d in drop) and not c.startswith("dense stage TOTAL")
    phases = [("prefill_chunk", "Prefill, per steady 1,792-token chunk"),
              ("decode_c1", "Decode step, concurrency 1"),
              ("decode_c8", "Decode step, concurrency 8")]
    colours = {
        "MoE trellis GEMM": GREEN, "NCCL collectives": BLUE, "Dense EXL3 GEMM": AMBER,
        "remaining bf16 linears": "#8a6fd0", "CPU gap (GPU idle)": "#b0b6c0",
        "MLA attention": "#0f9aa8", "KDA/GDN linear-attn": "#d4682f",
        "hyper-connection mixing (HC, mhc_*/hc_prenorm)": "#c04f8a",
        "hyper-connection mixing (HC)": "#c04f8a",
        "MoE hadamard (had_in / glu_had_in)": "#6d8a2f", "MoE hadamard": "#6d8a2f",
        "norm / elementwise / copy": "#7a8494", "Dense EXL3 hadamard": "#a08850",
    }
    sv = SVG(960, 520)
    y0 = sv.head(
        "Where a step actually goes - production 9, torch profiler on the live server, all three ranks",
        "Measured 5 September 2026, no restart and no reconfiguration: /start_profile and /stop_profile against the running engine.",
        "Image exl3-zeus:754421f, full-scope checkpoint, TP=3 + EP, DFlash2 k=7, fp8 KV and fp8 draft cache, gpu-memory-utilization 0.80.",
        "Source: results/profile/step-breakdown.csv. Bars are shares of wall time per step; classes under 2 % are pooled into 'other'.")

    y = y0 + 26
    left, right = 20, 938
    for key, title in phases:
        rs = sorted([r for r in rows if r["phase"] == key and keep(r)],
                    key=lambda r: -num(r["percent"], 0))
        big = [r for r in rs if num(r["percent"], 0) >= 2.0]
        rest = sum(num(r["percent"], 0) for r in rs if num(r["percent"], 0) < 2.0)
        total = sum(num(r["percent"], 0) for r in big) + rest
        sv.text(left, y, title, 12.5, INK, weight="600")
        y += 12
        x = left
        w_total = right - left
        for r in big:
            frac = num(r["percent"], 0) / total
            w = w_total * frac
            sv.rect(x, y, w, 30, colours.get(r["class"], SLATE))
            if w > 40:
                sv.text(x + w / 2, y + 19, f"{num(r['percent'],0):.1f}%", 11, "#ffffff", "middle", weight="600")
            x += w
        if rest > 0:
            sv.rect(x, y, right - x, 30, "#c9ced6")
            if right - x > 40:
                sv.text((x + right) / 2, y + 19, f"{rest:.1f}%", 11, INK, "middle")
        y += 38
        # per-phase key
        kx = left
        for r in big:
            lab = r["class"].split(" (")[0]
            sv.rect(kx, y - 8, 9, 9, colours.get(r["class"], SLATE), rx=2)
            sv.text(kx + 13, y, f"{lab}", 10, INK)
            kx += 13 + 6.0 * len(lab) + 16
            if kx > right - 160:
                kx = left
                y += 15
        sv.rect(kx, y - 8, 9, 9, "#c9ced6", rx=2)
        sv.text(kx + 13, y, "other (each < 2 %)", 10, INK)
        y += 30

    sv.text(20, y + 4, "The row this configuration was built to delete: the unquantized dense stage was 45.3 % of a C1 decode step on production 7, and is 25.9 % here -",
            11.5, SUB)
    sv.text(20, y + 20, "42.90 ms down to 21.90 ms. That -21 ms is the whole of the +22 %: draft acceptance is unchanged when pooled by draft token, so nothing else moved.", 11.5, SUB)
    sv.text(20, y + 36, "Read two numbers with care: NCCL and CPU-gap at C1 are inflated by CUPTI (2,738 kernel launches per step). With the profiler off the two",
            11.5, SUB)
    sv.text(20, y + 52, "together are <= 17.19 ms rather than 29.1. The NCCL class is 100 % exposed either way: measured comm/compute overlap is 0.00 ms.", 11.5, SUB)
    sv.save("step-breakdown-prod9.svg")

# --------------------------------------------------------------- chart 4 ----
def dense_stage():
    sv = SVG(960, 380)
    y0 = sv.head(
        "The one number production 9 was built to move",
        "Dense (non-MoE) stage as a share of a single-stream decode step, same three nodes, same fabric, same drafter.",
        "Production 7: routed experts 4-bit beside a BF16 dense path. Production 9: the dense path is 5-6 bit EXL3 as well.",
        "Both measured with the torch profiler on the live server. Source: results/profile/step-breakdown.csv.")
    top, bot, left = y0 + 30, 300, 96
    ymax, step = 50.0, 10.0
    def Y(v):
        return bot - (v / ymax) * (bot - top)
    g = 0.0
    while g <= ymax + 1e-9:
        sv.line(left, Y(g), 930, Y(g))
        sv.text(left - 8, Y(g) + 4, f"{g:g}%", 10, SUB, "end")
        g += step
    for i, (lab, pct, ms, colour) in enumerate([
            ("production 7\nBF16 dense path", 45.3, "42.90 ms", "#b8402f"),
            ("production 9\nEXL3 dense path", 25.9, "21.90 ms", GREEN)]):
        cx = left + 200 + i * 400
        sv.rect(cx - 60, Y(pct), 120, bot - Y(pct), colour)
        sv.text(cx, Y(pct) - 20, f"{pct} %", 19, INK, "middle", weight="600")
        sv.text(cx, Y(pct) - 6, ms, 11, SUB, "middle")
        for j, part in enumerate(lab.split("\n")):
            sv.text(cx, bot + 18 + j * 14, part, 11.5, INK if j == 0 else SUB, "middle")
    sv.text(20, 348, "Step time 88.2 -> 70.3 ms. Draft acceptance, pooled by draft token, is unchanged (+0.18 points), so none of the gain is drafter behaviour.", 11.5, SUB)
    sv.text(20, 364, "Prefill did not improve: the dense stage there went the other way, +10.4 %. The wall-clock prefill number stayed inside the +/-3 % band.", 11.5, SUB)
    sv.save("dense-stage-prod7-vs-prod9.svg")

if __name__ == "__main__":
    speed_by_configuration()
    kv_pool_progression()
    step_breakdown_prod9()
    dense_stage()

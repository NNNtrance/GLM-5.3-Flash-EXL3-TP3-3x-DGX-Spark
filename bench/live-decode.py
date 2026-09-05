#!/usr/bin/env python3
"""Decode step time at C1 and C8 with the DFlash2 drafter active, on the live engine.

Step count comes from vllm:spec_decode_num_drafts_total (one draft per sequence per
engine step), so ms/step is exact and does not depend on the output-batching of the
iteration_tokens histogram.
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = os.environ.get("API", "http://192.0.2.10:8001")
PROMPTS = [json.loads(l)["prompt"]
           for l in open(os.environ.get("HIZSET", "/var/tmp/hizset-v2.jsonl"))]


def metrics():
    t = urllib.request.urlopen(API + "/metrics", timeout=30).read().decode()
    g = lambda k: float(re.search(re.escape(k) + r"\{[^}]*\}\s+([0-9.e+]+)", t).group(1))
    return dict(drafts=g("vllm:spec_decode_num_drafts_total"),
                dtok=g("vllm:spec_decode_num_draft_tokens_total"),
                acc=g("vllm:spec_decode_num_accepted_tokens_total"),
                gen=g("vllm:generation_tokens_total"))


def stream(prompt, max_tokens):
    body = json.dumps({"model": "glm-5.3-flash", "stream": True,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": True,
                                                "reasoning_effort": "low"}}).encode()
    t0 = time.time(); first = None; last = None; n = 0; usage = None
    r = urllib.request.urlopen(urllib.request.Request(
        API + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800)
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        d = json.loads(p)
        if d.get("usage"):
            usage = d["usage"]
        ch = d.get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content") is not None:
            now = time.time()
            if first is None:
                first = now
            last = now; n += 1
    return dict(t0=t0, first=first, last=last, chunks=n, usage=usage, total=time.time() - t0)


def run(conc, max_tokens, label):
    ps = [PROMPTS[i % len(PROMPTS)] for i in range(conc)]
    m0 = metrics(); t0 = time.time()
    if conc == 1:
        res = [stream(ps[0], max_tokens)]
    else:
        with ThreadPoolExecutor(conc) as ex:
            res = [f.result() for f in [ex.submit(stream, p, max_tokens) for p in ps]]
    wall = time.time() - t0
    m1 = metrics()
    dd = {k: m1[k] - m0[k] for k in m0}
    gen = dd["gen"]
    steps = dd["drafts"] / conc
    ttft = [r["first"] - r["t0"] for r in res]
    dec_win = max(r["last"] for r in res) - max(ttft[i] + res[i]["t0"] for i in range(conc))
    print(f"\n=== {label}: C{conc}, max_tokens={max_tokens} ===")
    print(f"  wall {wall:.3f} s   TTFT {min(ttft):.3f}-{max(ttft):.3f} s   "
          f"decode window {dec_win:.3f} s")
    print(f"  generated {gen:.0f} tok   total tok/s (decode window) {gen/dec_win:.2f}   "
          f"end-to-end {gen/wall:.2f}")
    print(f"  engine decode steps {steps:.1f}   ms/step {dec_win*1000/steps:.2f}   "
          f"tokens/step/seq {gen/steps/conc:.2f}")
    print(f"  draft tokens {dd['dtok']:.0f}  accepted {dd['acc']:.0f}  "
          f"acceptance {100*dd['acc']/dd['dtok']:.1f}%")
    return dict(conc=conc, ms_step=dec_win * 1000 / steps, tps=gen / dec_win,
                tok_step=gen / steps / conc, acc=100 * dd["acc"] / dd["dtok"])


if __name__ == "__main__":
    stream(PROMPTS[0], 32)  # warm
    out = []
    for rnd in (1, 2):
        out.append(run(1, 256, f"round{rnd}"))
        out.append(run(8, 256, f"round{rnd}"))
    print("\nSUMMARY")
    for o in out:
        print(f"  C{o['conc']}  {o['ms_step']:7.2f} ms/step  {o['tps']:7.2f} tok/s  "
              f"{o['tok_step']:.2f} tok/step/seq  acc {o['acc']:.1f}%")

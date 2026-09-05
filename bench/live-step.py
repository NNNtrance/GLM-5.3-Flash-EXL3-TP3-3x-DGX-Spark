#!/usr/bin/env python3
"""Live-engine step timing on the running exl3-tp3 (no restart, no config change).

Prefill ladder with fresh (never-seen) prompts -> marginal cost of one 2048-token
chunk; decode at C1 and C8 with the DFlash2 drafter active -> ms per engine step.
Step counts come from vllm:iteration_tokens_total_count, so ms/step is exact.
"""
import json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = os.environ.get("API", "http://192.0.2.10:8001")


def metrics():
    txt = urllib.request.urlopen(API + "/metrics", timeout=30).read().decode()
    out = {}
    for k in ("vllm:iteration_tokens_total_count", "vllm:iteration_tokens_total_sum",
              "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
              "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
              "vllm:spec_decode_num_accepted_tokens_total"):
        m = re.search(re.escape(k) + r"\{[^}]*\}\s+([0-9.e+]+)", txt)
        out[k.replace("vllm:", "")] = float(m.group(1)) if m else None
    return out


def d(a, b):
    return {k: (b[k] - a[k]) if a[k] is not None else None for k in a}


def mk(seed, n):
    return ("Summarize the following notes in three bullet points. Notes: " +
            " ".join(f"entry {i}: on day {i+seed} the crew measured pressure {100+(i*seed)%17} "
                     f"and temperature {20+(i+seed)%9}, then logged a note about valve "
                     f"{(i+seed)%5} and pump {(i*3+seed)%7};" for i in range(1, n)))


def chat(prompt, max_tokens, timeout=900):
    body = json.dumps({"model": "glm-5.3-flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "seed": 1234,
                       "chat_template_kwargs": {"enable_thinking": True,
                                                "reasoning_effort": "low"}}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        API + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=timeout))
    return r["usage"], time.time() - t


def prefill_ladder(reps=3):
    print("\n=== PREFILL LADDER (fresh prompts, max_tokens=1) ===")
    print(f"{'entries':>8} {'prompt_tok':>11} {'wall_s':>8} {'tok/s':>8} {'steps':>6} {'ms/step':>8}")
    rows = []
    for n in (29, 57, 114, 171, 228):
        for _ in range(reps):
            m0 = metrics()
            u, dt = chat(mk(random.randint(10**6, 10**9), n), 1)
            m1 = metrics()
            dd = d(m0, m1)
            st = dd["iteration_tokens_total_count"]
            rows.append((n, u["prompt_tokens"], dt, st))
            print(f"{n:>8} {u['prompt_tokens']:>11} {dt:>8.3f} {u['prompt_tokens']/dt:>8.0f} "
                  f"{st:>6.0f} {dt*1000/st:>8.1f}")
    return rows


def decode(conc, max_tokens=200, prompt_entries=6):
    m0 = metrics()
    t = time.time()
    if conc == 1:
        us = [chat(mk(random.randint(10**6, 10**9), prompt_entries), max_tokens)[0]]
    else:
        with ThreadPoolExecutor(conc) as ex:
            futs = [ex.submit(chat, mk(random.randint(10**6, 10**9), prompt_entries), max_tokens)
                    for _ in range(conc)]
            us = [f.result()[0] for f in futs]
    wall = time.time() - t
    m1 = metrics()
    dd = d(m0, m1)
    gen = sum(u["completion_tokens"] for u in us)
    pro = sum(u["prompt_tokens"] for u in us)
    steps = dd["iteration_tokens_total_count"]
    pre_steps = sum(-(-u["prompt_tokens"] // 2048) for u in us)
    dec_steps = steps - pre_steps
    acc = dd["spec_decode_num_accepted_tokens_total"]
    drafts = dd["spec_decode_num_drafts_total"]
    dtk = dd["spec_decode_num_draft_tokens_total"]
    print(f"\n=== DECODE C{conc} ===")
    print(f"  wall {wall:.3f} s   prompt {pro} tok   generated {gen} tok   total tok/s {gen/wall:.2f}")
    print(f"  engine steps {steps:.0f}  (prefill ~{pre_steps}, decode ~{dec_steps:.0f})")
    print(f"  ms per decode step  {wall*1000/max(dec_steps,1):.2f}   tokens/step (all seqs) "
          f"{gen/max(dec_steps,1):.2f}   per seq {gen/max(dec_steps,1)/conc:.2f}")
    print(f"  drafts {drafts:.0f}  draft tokens {dtk:.0f}  accepted {acc:.0f}  "
          f"acceptance {100*acc/max(dtk,1):.1f}%")
    print(f"  iteration_tokens sum {dd['iteration_tokens_total_sum']:.0f} "
          f"-> mean batch M {dd['iteration_tokens_total_sum']/max(steps,1):.1f} tok/step")
    return dict(conc=conc, wall=wall, gen=gen, steps=steps, dec_steps=dec_steps,
                ms_step=wall*1000/max(dec_steps, 1), tps=gen/wall,
                acc=100*acc/max(dtk, 1), tok_step=gen/max(dec_steps, 1))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    # warm the shapes outside the measurement
    for _ in range(2):
        chat(mk(random.randint(10**6, 10**9), 228), 4)
    chat("Count from 1 to 10.", 32)
    if what in ("all", "prefill"):
        prefill_ladder(3)
    if what in ("all", "decode"):
        decode(1)
        decode(8)
        decode(1)
        decode(8)

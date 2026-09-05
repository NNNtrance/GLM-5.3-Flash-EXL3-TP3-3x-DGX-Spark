#!/usr/bin/env python3
"""Unprofiled baseline for PROFIL-URETIM7 — same windows, profiler OFF.
Gives the profiler-overhead factor that the measured trace shares must be read against."""
import json, os, random, re, sys, threading, time, urllib.request

API = os.environ.get("API", "http://192.0.2.10:8001")
P = [json.loads(l)["prompt"] for l in open(os.environ.get("HIZSET", "/var/tmp/hizset-v2.jsonl"))]


def metrics():
    t = urllib.request.urlopen(API + "/metrics", timeout=30).read().decode()

    def g(k):
        m = re.search(re.escape(k) + r"\{[^}]*\}\s+([0-9.e+]+)", t)
        return float(m.group(1)) if m else float("nan")
    return dict(ts=time.time(), drafts=g("vllm:spec_decode_num_drafts_total"),
                dtok=g("vllm:spec_decode_num_draft_tokens_total"),
                acc=g("vllm:spec_decode_num_accepted_tokens_total"),
                gen=g("vllm:generation_tokens_total"), run=g("vllm:num_requests_running"))


def mk(seed, n=230):
    return ("Summarize the following notes in three bullet points. Notes: " +
            " ".join(f"entry {i}: on day {i+seed} the crew measured pressure {100+(i*seed)%17} "
                     f"and temperature {20+(i+seed)%9}, then logged a note about valve "
                     f"{(i+seed)%5} and pump {(i*3+seed)%7};" for i in range(1, n)))


def chat(p, mt, mn=None, to=1800):
    b = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": p}],
         "max_tokens": mt, "temperature": 0, "seed": 1234,
         "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
    if mn:
        b["min_tokens"] = mn
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        API + "/v1/chat/completions", json.dumps(b).encode(),
        {"Content-Type": "application/json"}), timeout=to))
    return r["usage"], time.time() - t


def dec(conc, secs, ramp, mt=900):
    done = []

    def one(p):
        try:
            done.append(chat(p, mt, mn=mt - 1))
        except Exception as e:                                    # noqa: BLE001
            done.append(("ERR", repr(e)))
    ths = [threading.Thread(target=one, args=(P[i % len(P)],), daemon=True)
           for i in range(conc)]
    for t in ths:
        t.start()
    time.sleep(ramp)
    m0 = metrics()
    time.sleep(secs)
    m1 = metrics()
    d = {k: m1[k] - m0[k] for k in m0}
    st = d["drafts"] / conc
    print(f"== DECODE C{conc} baseline == running={m0['run']:.0f} window {d['ts']:.2f}s "
          f"steps {st:.1f} ms/step {d['ts']*1000/st:.2f}  tok/step/seq "
          f"{d['gen']/st/conc:.2f}  gen {d['gen']:.0f} tok/s {d['gen']/d['ts']:.1f}  "
          f"acc {100*d['acc']/d['dtok']:.1f}%", flush=True)
    for t in ths:
        t.join(timeout=900)
    return d["ts"] * 1000 / st


if __name__ == "__main__":
    print("== PREFILL baseline (fresh ~8.4K, max_tokens=1) ==", flush=True)
    pre = []
    for _ in range(4):
        u, dt = chat(mk(random.randint(10**6, 10**9)), 1)
        pre.append((u["prompt_tokens"], dt))
        print(f"  {u['prompt_tokens']} tok  {dt:.3f} s  {u['prompt_tokens']/dt:.0f} tok/s",
              flush=True)
    chat(P[0], 32)
    c1 = [dec(1, 8, 2.0) for _ in range(2)]
    c8 = [dec(8, 16, 6.0) for _ in range(2)]
    print("BASELINE ms/step  C1", [f"{x:.2f}" for x in c1], " C8",
          [f"{x:.2f}" for x in c8], flush=True)
    print(json.dumps({"prefill": pre, "c1_ms_step": c1, "c8_ms_step": c8}), flush=True)

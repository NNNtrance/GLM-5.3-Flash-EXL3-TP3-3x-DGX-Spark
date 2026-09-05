#!/usr/bin/env python3
"""PROFIL-URETIM7 — three measured torch-profiler windows on the LIVE exl3-tp3 engine.

Windows
  prefill : warm-up x3 (fresh ~8.4K prompts, outside the window), then ONE fresh ~8.4K
            prompt with max_tokens=1 inside the window -> 5 chunk passes at MNBT 2048.
  c1      : one request already in steady decode (drafter active), profile ~SECS seconds.
  c8      : eight concurrent requests already in steady decode, profile ~SECS seconds.

The engine is never restarted or reconfigured. Only /start_profile, /stop_profile and
normal chat requests are used.  Step counts inside the window come from
vllm:spec_decode_num_drafts_total (one draft per sequence per engine step) read
immediately after start_profile and immediately before stop_profile.
"""
import json, os, random, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = os.environ.get("API", "http://192.0.2.10:8001")
PROMPTS = [json.loads(l)["prompt"]
           for l in open(os.environ.get("HIZSET", "/var/tmp/hizset-v2.jsonl"))]


def post(p):
    r = urllib.request.urlopen(urllib.request.Request(
        API + p, b"", {"Content-Type": "application/json"}), timeout=600)
    return r.status


def metrics():
    t = urllib.request.urlopen(API + "/metrics", timeout=30).read().decode()

    def g(k):
        m = re.search(re.escape(k) + r"\{[^}]*\}\s+([0-9.e+]+)", t)
        return float(m.group(1)) if m else float("nan")
    return dict(ts=time.time(),
                drafts=g("vllm:spec_decode_num_drafts_total"),
                dtok=g("vllm:spec_decode_num_draft_tokens_total"),
                acc=g("vllm:spec_decode_num_accepted_tokens_total"),
                gen=g("vllm:generation_tokens_total"),
                prompt=g("vllm:prompt_tokens_total"),
                iters=g("vllm:iteration_tokens_total_count"),
                isum=g("vllm:iteration_tokens_total_sum"),
                running=g("vllm:num_requests_running"))


def mk(seed, n=230):
    return ("Summarize the following notes in three bullet points. Notes: " +
            " ".join(f"entry {i}: on day {i+seed} the crew measured pressure {100+(i*seed)%17} "
                     f"and temperature {20+(i+seed)%9}, then logged a note about valve "
                     f"{(i+seed)%5} and pump {(i*3+seed)%7};" for i in range(1, n)))


def chat(prompt, max_tokens, min_tokens=None, timeout=1800):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
    if min_tokens:
        body["min_tokens"] = min_tokens
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        API + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=timeout))
    return r["usage"], time.time() - t


def report(tag, m0, m1, extra=""):
    dd = {k: m1[k] - m0[k] for k in m0}
    steps = dd["drafts"]
    print(f"[{tag}] window {dd['ts']:.3f} s  prompt {dd['prompt']:.0f}  gen {dd['gen']:.0f}  "
          f"drafts {steps:.0f}  iters {dd['iters']:.0f}  "
          f"draft_tok {dd['dtok']:.0f}  acc {dd['acc']:.0f} "
          f"({100*dd['acc']/max(dd['dtok'],1):.1f}%)  {extra}")
    return dd


def run_prefill():
    print("warm-up x3 (outside the window)")
    for _ in range(3):
        u, dt = chat(mk(random.randint(10**6, 10**9)), 4)
        print(f"  warm {u['prompt_tokens']} tok  {dt:.2f} s  {u['prompt_tokens']/dt:.0f} tok/s")
    m0 = metrics()
    print("start_profile", post("/start_profile"))
    ms = metrics()
    u, dt = chat(mk(random.randint(10**6, 10**9)), 1)
    me = metrics()
    print(f"PREFILL {u['prompt_tokens']} tok in {dt:.3f} s = {u['prompt_tokens']/dt:.0f} tok/s "
          f"(PROFILED — expect overhead)")
    print("stop_profile", post("/stop_profile"))
    report("prefill-window", ms, me, f"wall {dt:.3f}s")
    print(json.dumps({"mode": "prefill", "prompt_tokens": u["prompt_tokens"],
                      "wall_s": dt, "tok_s": u["prompt_tokens"] / dt}))


def run_decode(conc, secs, max_tokens, ramp):
    print(f"warm-up (outside the window), C{conc}")
    chat(PROMPTS[0], 32)
    done = []

    def one(p):
        try:
            done.append(chat(p, max_tokens, min_tokens=max_tokens - 1))
        except Exception as e:                                   # noqa: BLE001
            done.append(("ERR", repr(e)))

    ps = [PROMPTS[i % len(PROMPTS)] for i in range(conc)]
    ths = [threading.Thread(target=one, args=(p,), daemon=True) for p in ps]
    t0 = time.time()
    for t in ths:
        t.start()
    # let every sequence get past prefill and into steady decode
    deadline = time.time() + ramp
    while time.time() < deadline:
        time.sleep(0.25)
    mr = metrics()
    print(f"  ramp {time.time()-t0:.2f} s  running={mr['running']:.0f} (want {conc})")
    print("start_profile", post("/start_profile"))
    ms = metrics()
    time.sleep(secs)
    me = metrics()
    print("stop_profile", post("/stop_profile"))
    dd = report(f"decode-C{conc}-window", ms, me)
    steps = dd["drafts"] / conc
    print(f"  engine steps in window {steps:.1f}   ms/step (PROFILED) "
          f"{dd['ts']*1000/max(steps,1):.2f}   tok/step/seq {dd['gen']/max(steps,1)/conc:.2f}")
    for t in ths:
        t.join(timeout=600)
    print(f"  requests finished: {len(done)}/{conc}, total elapsed {time.time()-t0:.1f} s")
    print(json.dumps({"mode": f"c{conc}", "window_s": dd["ts"], "steps": steps,
                      "ms_step_profiled": dd["ts"] * 1000 / max(steps, 1),
                      "gen": dd["gen"], "acc_pct": 100 * dd["acc"] / max(dd["dtok"], 1)}))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prefill":
        run_prefill()
    elif mode == "c1":
        run_decode(1, secs=float(sys.argv[2]) if len(sys.argv) > 2 else 8.0,
                   max_tokens=900, ramp=2.0)
    elif mode == "c8":
        run_decode(8, secs=float(sys.argv[2]) if len(sys.argv) > 2 else 16.0,
                   max_tokens=900, ramp=6.0)
    else:
        sys.exit("mode: prefill | c1 | c8")

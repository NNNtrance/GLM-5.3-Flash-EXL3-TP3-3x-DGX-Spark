#!/usr/bin/env python3
"""One profiling pass: a 7.4K prefill window and a short decode window."""
import json, random, sys, time, urllib.request
API = "http://192.0.2.10:8001"

def post(p):
    r = urllib.request.urlopen(urllib.request.Request(API + p, b"", {"Content-Type": "application/json"}), timeout=300)
    return r.status

def mk(seed, n=230):
    return ("Summarize the following notes in three bullet points. Notes: " +
            " ".join(f"entry {i}: on day {i+seed} the crew measured pressure {100+(i*seed)%17} "
                     f"and temperature {20+(i+seed)%9}, then logged a note about valve {(i+seed)%5} "
                     f"and pump {(i*3+seed)%7};" for i in range(1, n)))

def chat(prompt, max_tokens):
    body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}).encode()
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions", body,
                                                                {"Content-Type": "application/json"}), timeout=900))
    return d["usage"], time.time() - t

mode = sys.argv[1] if len(sys.argv) > 1 else "prefill"
if mode == "prefill":
    # warm the engine and the shapes first, OUTSIDE the profile window
    for _ in range(3):
        u, dt = chat(mk(random.randint(10**6,10**9)), 4)
    print("warm", u["prompt_tokens"], f"{dt:.1f}s")
    print("start_profile", post("/start_profile"))
    u, dt = chat(mk(random.randint(10**6,10**9)), 1)
    print(f"PREFILL {u['prompt_tokens']} tok in {dt:.2f}s = {u['prompt_tokens']/dt:.0f} tok/s")
    print("stop_profile", post("/stop_profile"))
else:
    u, dt = chat("Count from 1 to 10.", 8); print("warm", f"{dt:.1f}s")
    print("start_profile", post("/start_profile"))
    u, dt = chat("Write a short python function that reverses a list, then explain it.", 48)
    print(f"DECODE {u['completion_tokens']} tok in {dt:.2f}s = {u['completion_tokens']/dt:.1f} tok/s")
    print("stop_profile", post("/stop_profile"))

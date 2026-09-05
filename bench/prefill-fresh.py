#!/usr/bin/env python3
"""7.4K prefill, measured on a prompt the engine has never seen.

prefill7k-8001.py reuses two fixed seeds, so a second run inside the same boot
reads most of the prompt out of the prefix cache (block 3328, prompt 7382 -> two
whole blocks free) and reports up to 1.6x the true rate. Seeds here are drawn
from the clock, and every request is a fresh prompt.
"""
import json, random, sys, time, urllib.request
API = sys.argv[1] if len(sys.argv) > 1 else "http://192.0.2.10:8001"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3

def mk(seed):
    return ("Summarize the following notes in three bullet points. Notes: " +
            " ".join(f"entry {i}: on day {i+seed} the crew measured pressure {100+(i*seed)%17} "
                     f"and temperature {20+(i+seed)%9}, then logged a note about valve "
                     f"{(i+seed)%5} and pump {(i*3+seed)%7};" for i in range(1, 230)))

rates = []
for _ in range(N):
    seed = random.randint(10**6, 10**9)
    body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": mk(seed)}],
                       "max_tokens": 1, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}).encode()
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        API + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t
    n = d["usage"]["prompt_tokens"]
    rates.append(n / dt)
rates.sort()
print(f"prefill-fresh: {n} tok, {N} unseen prompts -> "
      + " / ".join(f"{r:.0f}" for r in rates) + f" tok/s (median {rates[len(rates)//2]:.0f})")

#!/usr/bin/env python3
"""Cold vs warm single-stream decode probe. Run it right after the engine comes
up, BEFORE the quality gates, so the first request is genuinely cold.

Sends the same code prompt three times: #1 is COLD (JIT, kernel caches, first
touch), #2 and #3 are WARM.

  decode tok/s = (completion_tokens - 1) / (end - first_token)
  TTFT         = first_token - start

PROMPT LANGUAGE MATTERS. Our published cold/warm numbers (43-47 tok/s, 44-48%
acceptance) were measured with this same task written in Turkish. The prompt
below is its English translation, and on the identical engine it measures
54-63 tok/s at 57-69% acceptance -- the drafter predicts English much better.
Raw data: results/speed/cold-warm-c1-english-prompt.txt. Compare your run
against the English range in audit/README.md, not against the Turkish one.

Usage: python3 cold-warm-c1.py [API_BASE]     default http://192.0.2.10:8001
Written by us for this recipe; use freely (Apache-2.0)."""
import json, os, sys, time, urllib.request
API = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("API", "http://192.0.2.10:8001")
PROMPT = ("Write an LRU cache class in Python: get/put in O(1), capacity as a "
          "parameter, using a doubly linked list plus a dict, without OrderedDict. "
          "Add a short docstring and a 6-line usage example.")
def metrics():
    try:
        raw = urllib.request.urlopen(API + "/metrics", timeout=10).read().decode()
    except Exception:
        return None
    out = {}
    for line in raw.splitlines():
        for k in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"):
            if line.startswith("vllm:" + k):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out if len(out) == 2 else None

def run(tag):
    m0 = metrics()
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 700, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") or (ch and (ch[0].get("delta") or {}).get("reasoning_content")):
                if t_first is None: t_first = time.time()
                n += 1
    t1 = time.time()
    ct = (usage or {}).get("completion_tokens", n)
    ttft = (t_first or t1) - t0; dec = (ct - 1) / max(t1 - (t_first or t1), 1e-6)
    m1 = metrics(); acc = ""
    if m0 and m1:
        dd = m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"]
        da = m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"]
        if dd > 0: acc = f" acceptance={100*da/dd:.1f}%"
    return f"{tag}: TTFT={ttft:.2f}s decode={dec:.1f}tok/s ({ct}tok/{t1-t0:.1f}s){acc}"
out = [run("C1-cold"), run("C1-warm"), run("C1-warm2")]
print("  cold-warm-c1: " + " | ".join(out))

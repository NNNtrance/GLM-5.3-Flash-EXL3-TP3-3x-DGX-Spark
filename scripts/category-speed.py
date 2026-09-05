#!/usr/bin/env python3
"""Per-category decode speed and speculative acceptance on a warm engine:
prose / code / math / structured-json. Concurrency 1 sequential, then
concurrency 4 in parallel, six prompts per category.

WHY THIS SCRIPT EXISTS: aggregate tok/s hides the fact that the DFlash2 drafter
behaves completely differently per workload. Code, math and JSON accept 44-60%
of drafted tokens and run 2-2.8x faster than no speculation at all; free prose
accepts about 13% and falls back to roughly the unspeculated rate.

NOTE ON COMPARABILITY: the category numbers we published were measured with the
ORIGINAL prompt set, in which half of the prompts in each category were in
Turkish and half in English. The prompts below are the English translations of
that set. Acceptance on the prose category in particular may differ slightly
from our published 13%; treat a rerun as your own baseline rather than as a
reproduction of ours.

Usage: python3 category-speed.py [API_BASE]      default http://192.0.2.10:8001
Written by us for this recipe; use freely (Apache-2.0)."""
import json, os, sys, time, urllib.request, statistics, threading
API = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("API", "http://192.0.2.10:8001")
CATS = {
 "prose": ["Write a 500-word short story set in a seaside town about two old friends meeting again after many years.",
           "Write a balanced essay of about 500 words on the pros and cons of using artificial intelligence in education.",
           "Write a warm, informal blog post of 400-500 words introducing a new pair of wireless headphones for a product launch.",
           "Write a 500-word reflective essay on why people keep journals, with concrete examples.",
           "Write a short story (about 500 words) about a lighthouse keeper who receives an unexpected letter.",
           "Explain, in about 450 words for a general reader, how sleep affects memory consolidation."],
 "code": ["Write an LRU cache class in Python: get/put in O(1) without OrderedDict; add a short docstring and a 6-line example.",
          "Write a Python function that parses ISO-8601 timestamps with optional timezone and returns UTC datetimes; include tests using assert.",
          "Implement Dijkstra's algorithm in Python over an adjacency dict with a heap; return distances and paths; add a small example.",
          "Write a Rust program that reads a CSV file and computes the mean of each column; handle errors with Result.",
          "Write a TypeScript function that debounces another function with leading/trailing options; include usage examples.",
          "Write a worker pool in Go that can be cancelled with a context (4 workers, a job channel, a result channel); add a short main."],
 "math": ["A right triangle has legs 7 and 24. Compute the hypotenuse, the inradius and the area, step by step.",
          "Solve step by step: find all real x such that x^4 - 5x^2 + 4 = 0, then compute the sum of the squares of the solutions.",
          "Find, step by step, the sum of all numbers from 1 to 1000 that are divisible by 3 or by 5 but not by 15.",
          "A bag has 5 red, 4 blue, 3 green balls. Three are drawn without replacement. Find the probability that all three colors appear; show the work.",
          "Compute the derivative and the antiderivative of f(x) = x^3 e^{-x} step by step; then evaluate its integral over [0, infinity).",
          "Prove by induction that 1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2 and verify for n=4 numerically."],
 "json": ["Return ONLY a JSON object describing 5 fictional employees: fields id (int), name, department, salary (number), skills (array of 3 strings), manager_id (int or null). No prose.",
          "Return ONLY JSON: a list of 6 cities as {\"city\": str, \"country\": str, \"population\": int, \"coordinates\": {\"lat\": float, \"lon\": float}, \"features\": [str,str,str]}.",
          "Return ONLY valid JSON: an order with order_id, customer {name,email,address{street,city,zip}}, items (4 items with sku, qty, unit_price), totals {subtotal, tax, total}, status.",
          "Return ONLY JSON: a weekly class timetable (7 days), 4 lessons each day: {\"day\": str, \"lessons\": [{\"time\": str, \"subject\": str, \"teacher\": str, \"room\": str}]}.",
          "Return ONLY JSON: a product catalog with 6 products, each with id, name, category, price, in_stock (bool), tags (array), dimensions {w,h,d}, reviews (2 objects with user, rating, text).",
          "Return ONLY JSON: {\"project\": str, \"tasks\": [8 tasks: {\"id\": int, \"title\": str, \"priority\": \"high|medium|low\", \"estimate_hours\": float, \"depends_on\": [int]}]}."]}
def metrics():
    raw = urllib.request.urlopen(API + "/metrics", timeout=10).read().decode(); out = {}
    for line in raw.splitlines():
        for k in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"):
            if line.startswith("vllm:" + k): out[k] = float(line.rsplit(" ", 1)[1])
    return out
def run(prompt, max_tokens=700):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); usage = d.get("usage") or usage
            ch = d.get("choices") or []
            if tf is None and ch and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")): tf = time.time()
    t1 = time.time(); ct = (usage or {}).get("completion_tokens", 0)
    return {"ttft": (tf or t1) - t0, "dec": (ct - 1) / max(t1 - (tf or t1), 1e-6), "tok": ct, "total": t1 - t0}
print(f"=== category speed (warm) {time.strftime('%H:%M')}  | decode = (tokens-1)/(end - first token), acceptance = delta of /metrics counters")
for cat, ps in CATS.items():
    m0 = metrics(); rs = [run(p) for p in ps]; m1 = metrics()
    dd = m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"]; da = m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"]
    print(f"  C1 {cat:10s} decode mean={statistics.mean(r['dec'] for r in rs):5.1f} tok/s (min {min(r['dec'] for r in rs):4.1f} / max {max(r['dec'] for r in rs):4.1f}) | TTFT mean={statistics.mean(r['ttft'] for r in rs):.2f}s | acceptance={100*da/dd if dd else 0:.1f}% | mean {statistics.mean(r['tok'] for r in rs):.0f} tokens", flush=True)
for cat, ps in CATS.items():
    m0 = metrics(); out = [None]*4; t0 = time.time()
    def w(i): out[i] = run(ps[i])
    th = [threading.Thread(target=w, args=(i,)) for i in range(4)]; [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0; m1 = metrics()
    dd = m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"]; da = m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"]
    tot = sum(r["tok"] for r in out)
    print(f"  C4 {cat:10s} per-stream decode mean={statistics.mean(r['dec'] for r in out):5.1f} tok/s | total={tot/wall:5.1f} tok/s | TTFT mean={statistics.mean(r['ttft'] for r in out):.2f}s | acceptance={100*da/dd if dd else 0:.1f}%", flush=True)
print(f"=== CATEGORY SPEED DONE {time.strftime('%H:%M')}")

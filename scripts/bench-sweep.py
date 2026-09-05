#!/usr/bin/env python3
"""Concurrency sweep for an OpenAI-compatible endpoint. Stdlib only.

This is the script behind every C1..C8 speed table we publish. Default prompt
set is hizset-v2.jsonl next to this file: 12 short English code prompts.

  python3 bench-sweep.py --out sweep.json --label t10-run1 --think low

Written by us for this recipe; use freely (Apache-2.0).

Per request: TTFT, decode tok/s (excluding TTFT), total tokens.
Per level: aggregate output throughput, TTFT median and max, TPOT, and the
server's own speculative-decoding counter deltas.

No percentiles: a level runs 8-32 requests, which cannot carry a p99. An
earlier version emitted `ttft_p99_s` from
`tt[min(len(tt)-1, int(len(tt)*0.99))]`, which for any n <= 100 is just
`tt[-1]` -- the single worst request. The field is now named `ttft_max_s`,
which is what it always was. Published JSON under results/ predating this
change still carries the old key with the same (max) values.
"""
import json, os, sys, time, statistics, urllib.request, threading, queue, argparse

BASE = os.environ.get("API", "http://192.0.2.10:8001")

def metrics():
    try:
        raw = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for ln in raw.splitlines():
        for k in ("vllm:spec_decode_num_drafts_total",
                  "vllm:spec_decode_num_draft_tokens_total",
                  "vllm:spec_decode_num_accepted_tokens_total"):
            if ln.startswith(k):
                out[k.split("_", 2)[-1]] = float(ln.rsplit(" ", 1)[1])
    return out

def one(prompt, out_len, model, equal_work=False, think="low"):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": out_len, "temperature": 0, "stream": True,
            # Equal work: with ignore_eos every request emits exactly out_len
            # tokens. Otherwise one arm writes 180 tokens and the other 256, and
            # the tok/s numbers are not measuring the same work.
            **({"ignore_eos": True} if equal_work else {}),
            "chat_template_kwargs": {"enable_thinking": True,
                                     "reasoning_effort": think},
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; toks = 0; usage = None
    for line in urllib.request.urlopen(req, timeout=900):
        s = line.decode().strip()
        if not s.startswith("data: "):
            continue
        s = s[6:]
        if s == "[DONE]":
            break
        d = json.loads(s)
        if d.get("usage"):
            usage = d["usage"]
        ch = d.get("choices") or []
        if ch:
            dl = ch[0].get("delta") or {}
            if dl.get("content") or dl.get("reasoning_content"):
                if ttft is None:
                    ttft = time.time() - t0
                toks += 1
    total = time.time() - t0
    n = usage["completion_tokens"] if usage else toks
    dec = (n - 1) / (total - ttft) if ttft and total > ttft and n > 1 else 0.0
    return {"ttft": ttft or total, "total": total, "tokens": n, "decode": dec}

def level(prompts, conc, nreq, out_len, model, equal_work=False, think="low"):
    q = queue.Queue()
    for i in range(nreq):
        # NOTE: this cycles a fixed prompt list, so from the second lap onward
        # every request is a prefix-cache hit and TTFT/prefill are flattered.
        # Relative comparisons survive (both arms see the same sequence);
        # ABSOLUTE ttft and prefill numbers from this script are not sound.
        # Fix: generate a unique prompt per request from a seed.
        q.put(prompts[i % len(prompts)])
    res, lock = [], threading.Lock()
    def worker():
        while True:
            try: p = q.get_nowait()
            except queue.Empty: return
            try:
                r = one(p, out_len, model, equal_work, think)
                with lock: res.append(r)
            except Exception as e:
                with lock: res.append({"err": str(e)[:80]})
            finally: q.task_done()
    m0 = metrics(); t0 = time.time()
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(conc)]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.time() - t0; m1 = metrics()
    ok = [r for r in res if "err" not in r]
    if not ok:
        return {"conc": conc, "error": res[0].get("err", "all failed")}
    tt = sorted(r["ttft"] for r in ok); dc = sorted(r["decode"] for r in ok)
    tot = sum(r["tokens"] for r in ok)
    d = lambda k: m1.get(k, 0) - m0.get(k, 0)
    drafts, dtok, atok = d("num_drafts_total"), d("num_draft_tokens_total"), d("num_accepted_tokens_total")
    return {"conc": conc, "requests": len(ok), "wall_s": round(wall, 2),
            "out_tok": tot,
            "agg_tok_s": round(tot / wall, 2),
            "per_stream_decode_tok_s": round(statistics.median(dc), 2),
            "ttft_med_s": round(statistics.median(tt), 3),
            "ttft_max_s": round(tt[-1], 3),   # worst single request, not a percentile
            "tpot_ms": round(1000 / statistics.median(dc), 2) if statistics.median(dc) else None,
            "accept_rate_pct": round(100 * atok / dtok, 2) if dtok else None,
            "accept_len": round(1 + atok / drafts, 2) if drafts else None}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "hizset-v2.jsonl"),
                    help="JSONL, one {\"prompt\": ...} per line")
    ap.add_argument("--base", default=BASE, help="API base URL (or set $API)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--equal-work", action="store_true",
                    help="ignore_eos: force every request to emit exactly --out-len tokens")
    ap.add_argument("--think", default="low",
                    help="reasoning_effort: only low and high are honoured, "
                         "anything else silently means max. Thinking is ALWAYS on.")
    a = ap.parse_args()
    BASE = a.base
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts) if l.strip()]
    print("  warm-up...", flush=True)
    one(prompts[0], 128, a.model, a.equal_work, a.think)
    one(prompts[1], 128, a.model, a.equal_work, a.think)
    rows = []
    for c in (1, 2, 4, 6, 8):
        n = max(8, c * 4)
        r = level(prompts, c, n, a.out_len, a.model, a.equal_work, a.think)
        r["label"] = a.label
        rows.append(r)
        print(f"  C={c:<2} requests={r.get('requests','-'):<3} total={r.get('agg_tok_s','-'):<7} "
              f"per_stream={r.get('per_stream_decode_tok_s','-'):<6} TTFT={r.get('ttft_med_s','-'):<6} "
              f"accept={r.get('accept_rate_pct','-')}%  len={r.get('accept_len','-')}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"  -> {a.out}")

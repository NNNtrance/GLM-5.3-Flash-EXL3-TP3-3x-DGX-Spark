#!/usr/bin/env python3
"""Needle-in-a-haystack probe that scores BOTH answer fields.

Why this file exists.  The single-request long-context gate is cheap, and it is
the gate most likely to be failed by the HARNESS rather than by the engine.  At
``reasoning_effort: low`` this model sometimes puts a short answer entirely into
``message.reasoning_content`` and leaves ``message.content`` empty.  A probe that
reads ``content`` alone scores that as a MISS.  On 6 September 2026 it did
exactly that to a 969,468-token gate: empty answer, 660.6 s, reported FAIL; the
identical request, same seed, returned the correct code in ``content`` on the
next attempt.  See docs/15 section 5.9.

So: score ``content`` and ``reasoning_content`` together, and print which one
held the code.  An empty answer and a WRONG answer are different findings and a
gate must not conflate them.

``max_tokens`` is 160 rather than 48 for a related reason: at 48 a short preamble
can eat the budget and truncate the answer, which also reads as a miss.

THINKING IS LEFT ON (reasoning effort low).  ``enable_thinking=false`` does not
turn thinking off on this model -- it removes the filter and leaks the reasoning
into the answer.  See docs/09 for the measurement protocol.

  needle-1m-bothfields.py [api] [ctx1,ctx2] [depth_pcts]

The ctx numbers are LABELS: the English filler tokenises at ~5.75 chars/token,
so "1550000" lands at ~969k prompt tokens.  The measured ``prompt_tokens`` is
what a report should quote.
"""
import json, sys, time, urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://192.0.2.10:8001"
CTXS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "64000,128000").split(",")]
DEPTHS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "10,50,90").split(",")]

NEEDLE_FMT = "The Ashgabat vault access code for {tag} is {code}."
FILLER = ("The maintenance log records routine checks of the pumping stations along the "
          "northern corridor. Pressure, temperature and flow are noted each shift, and the "
          "duty engineer signs the page. Nothing unusual was observed during this period. ")


def chat(prompt, max_tokens=160, timeout=3600):
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 7,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        API + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=timeout))
    msg = r["choices"][0]["message"]
    # At reasoning_effort low the model sometimes puts a SHORT answer into
    # reasoning_content and leaves content empty -- a measured artefact of the
    # extractor, not a miss. Scoring only "content" turns that into a false FAIL,
    # which is what it did to the TP=2 1M gate on 6 September 2026 (docs/15
    # section 5.9). Score BOTH fields, and print which one held the code.
    both = (msg.get("content") or "") + " || REASONING: " + (msg.get("reasoning_content") or "")
    return both, r["usage"], time.time() - t


ok = tot = 0
rows = []
for ctx in CTXS:
    # MEASURED: this English filler tokenises at ~5.75 chars/token, so a "64000"
    # request lands at ~40k prompt tokens. The CTX numbers are therefore labels;
    # the measured prompt_tokens is what the report quotes.
    body_chars = int(ctx * 3.6)
    base = (FILLER * (body_chars // len(FILLER) + 2))[:body_chars]
    for d in DEPTHS:
        code = f"{ctx//1000}-{d}-QX{(ctx+d) % 97:02d}"
        needle = NEEDLE_FMT.format(tag=f"gate {d}", code=code)
        cut = max(0, min(len(base), int(len(base) * d / 100)))
        prompt = (base[:cut] + "\n\n" + needle + "\n\n" + base[cut:] +
                  "\n\nQuestion: what is the Ashgabat vault access code mentioned in the "
                  "text above? Answer with the code only.")
        try:
            ans, usage, dt = chat(prompt)
            hit = code in ans
        except Exception as e:                                       # noqa: BLE001
            ans, usage, dt, hit = repr(e)[:80], {"prompt_tokens": -1}, -1.0, False
        tot += 1
        ok += bool(hit)
        rows.append(dict(ctx=ctx, depth_pct=d, prompt_tokens=usage.get("prompt_tokens"),
                         found=bool(hit), s=round(dt, 1), answer=ans[:60]))
        print(f"  ctx~{ctx//1000}K depth {d:>2}%  prompt_tokens={usage.get('prompt_tokens')}  "
              f"{'PASS' if hit else 'FAIL'}  {dt:.1f}s  {ans[:50]!r}", flush=True)
print(f"NEEDLE-LITE: {ok}/{tot}")
print(json.dumps(rows))

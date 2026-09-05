#!/usr/bin/env python3
"""Prefill probe: two ~7k-token prompts with fixed seeds, timed end to end.

IMPORTANT: this is a WARM prefill number, and only the SECOND request is
reported. Run it twice inside one boot and the second run reads whole blocks
out of the prefix cache and reports up to 1,596 tok/s where the honest number
is 1,025. For a cold, unseen-prompt measurement use bench/prefill-fresh.py,
which draws a new seed per request. See docs/09.

Usage: python3 prefill-7k.py [API_BASE]   default http://192.0.2.10:8001
Written by us for this recipe; use freely (Apache-2.0)."""
import os,sys
import json,time,urllib.request
API=sys.argv[1] if len(sys.argv)>1 else os.environ.get("API","http://192.0.2.10:8001")
def mk(seed): return "Summarize the following notes in three bullet points. Notes: "+" ".join(f"entry {i}: on day {i+seed} the crew measured pressure {100+(i*seed)%17} and temperature {20+(i+seed)%9}, then logged a note about valve {(i+seed)%5} and pump {(i*3+seed)%7};" for i in range(1,230))
r=[]
for seed in (11,29):
    body=json.dumps({"model":"glm-5.3-flash","messages":[{"role":"user","content":mk(seed)}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"low"}}).encode()
    t=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request(API+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=600)); dt=time.time()-t
    r.append((d["usage"]["prompt_tokens"],dt))
print(f"prefill7k={r[0][0]}tok:{r[0][1]:.1f}s/{r[1][1]:.1f}s={r[1][0]/r[1][1]:.0f}tok/s(2nd prompt, not from the prefix cache)")

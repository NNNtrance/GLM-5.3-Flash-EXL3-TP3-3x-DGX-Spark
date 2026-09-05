#!/usr/bin/env python3
"""Mixed-load probe: one decode stream (256 tokens, streaming) is already running
when, one second later, a ~7k-token prompt arrives.

Measures: the decode stream's average tok/s while the long prompt is being
prefilled (it runs at about 50 alone), and the long prompt's TTFT. This is the
real cost of the scheduler's chunking and preemption policy, which a
single-workload benchmark never shows.

Usage: python3 mixed-load-probe.py [API_BASE]   default http://192.0.2.10:8001
Written by us for this recipe; use freely (Apache-2.0)."""
import json,time,urllib.request,threading
import os,sys
API=sys.argv[1] if len(sys.argv)>1 else os.environ.get("API","http://192.0.2.10:8001")
def mk(seed): return "Summarize the following notes in three bullet points. Notes: "+" ".join(f"entry {i}: on day {i+seed} the crew measured pressure {100+(i*seed)%17} and temperature {20+(i+seed)%9}, then logged a note about valve {(i+seed)%5} and pump {(i*3+seed)%7};" for i in range(1,230))
res={}
def decode_stream():
    body=json.dumps({"model":"glm-5.3-flash","messages":[{"role":"user","content":"Write a long, detailed Python module implementing a priority queue with docstrings and tests. Only code."}],"max_tokens":256,"temperature":0,"stream":True,"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"low"}}).encode()
    t0=time.time(); n=0; first=None
    with urllib.request.urlopen(urllib.request.Request(API+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=600) as h:
        for line in h:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                d=json.loads(line[6:]); dl=d["choices"][0].get("delta",{})
                if dl.get("content") or dl.get("reasoning_content"):
                    n+=1; first=first or time.time()
    res["decode"]=(n, time.time()-(first or t0))
def long_prompt():
    time.sleep(1.0)
    body=json.dumps({"model":"glm-5.3-flash","messages":[{"role":"user","content":mk(37)}],"max_tokens":16,"temperature":0,"stream":True,"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"low"}}).encode()
    t0=time.time(); first=None
    with urllib.request.urlopen(urllib.request.Request(API+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=600) as h:
        for line in h:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                dl=json.loads(line[6:])["choices"][0].get("delta",{})
                if (dl.get("content") or dl.get("reasoning_content")) and first is None: first=time.time()
    res["ttft7k"]=(first or time.time())-t0
import subprocess,datetime
# Optional: pull the engine's own scheduler counters for the same window, so the
# probe's numbers can be read against Running/Waiting/Deferred. Set
# ENGINE_HOST (an ssh target, or "local") and ENGINE_CONTAINER; both default to
# off, and the probe reports its own measurements either way.
ENGINE_HOST=os.environ.get("ENGINE_HOST","")
ENGINE_CONTAINER=os.environ.get("ENGINE_CONTAINER","exl3-tp3")
t_start=datetime.datetime.utcnow()
a=threading.Thread(target=decode_stream); b=threading.Thread(target=long_prompt); a.start(); b.start(); a.join(); b.join()
engine="not collected"
if ENGINE_HOST:
    pat="Avg prompt throughput: [0-9.]+ tokens/s, Avg generation throughput: [0-9.]+ tokens/s, Running: [0-9]+ reqs, Waiting: [0-9]+ reqs(, Deferred: [0-9]+ reqs)?"
    cmd="docker logs --since %s %s 2>&1 | grep -oE '%s' | head -8" % (
        t_start.strftime("%Y-%m-%dT%H:%M:%S"), ENGINE_CONTAINER, pat)
    argv=["bash","-lc",cmd] if ENGINE_HOST=="local" else ["ssh","-o","BatchMode=yes",ENGINE_HOST,cmd]
    try:
        log=subprocess.run(argv,capture_output=True,text=True,timeout=30).stdout
        engine=" | ".join(l.replace("Avg prompt throughput: ","P=")
                           .replace(" tokens/s, Avg generation throughput: ","/G=")
                           .replace(" tokens/s, Running: ","/R")
                           .replace(" reqs, Waiting: ","/W")
                           .replace(" reqs, Deferred: ","/D")
                           .replace(" reqs","") for l in log.strip().splitlines()) or "no matching log lines"
    except Exception:
        engine="engine log unavailable"
n,dt=res["decode"]
print(f"mixed-load: decode {n} tok / {dt:.1f} s = {n/dt:.1f} tok/s (while the 7k prefill overlaps), "
      f"7k TTFT = {res['ttft7k']:.1f} s, engine[{engine}]")

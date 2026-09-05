#!/usr/bin/env python3
"""Boot ledger for the EXL3 TP=3 engine.

    boot-ledger.py --node head              (fetches docker logs itself over ssh)
    ssh <node> 'docker logs -t exl3-tp3 2>&1' | boot-ledger.py --start <StartedAt>

Every number comes from `docker logs -t` timestamps and the container's own
StartedAt, so the ledger is the engine's clock, not the host's.
"""

import argparse
import datetime as dt
import re
import subprocess
import sys

TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s?(.*)$")

# (label, regex, "first" | "last")
MARKS = [
    ("prelude first line", r"\[tp3-prelude\] rank=", "first"),
    ("preflight done", r"\[tp3-prelude\] patches applied", "first"),
    ("weight load start",
     r"Loading safetensors checkpoint shards|weight_utils\.py:\d+\] (Auto-prefetch|Prefetching)"
     r"|\[harem-fastload\] (restore|dump) start", "first"),
    ("weight load end", r"Loading weights took", "first"),
    ("drafter load end", r"Loading weights took", "last"),
    ("model load total", r"Model loading took", "first"),
    ("KV pool", r"GPU KV cache size:", "first"),
    ("autotune start", r"Autotuning process starts", "first"),
    ("autotune end", r"Saved \d+ configs", "first"),
    ("graph capture end", r"Graph capturing finished", "first"),
    ("init engine", r"init engine \(profile, create kv cache, warmup model\) took", "first"),
    ("API ready", r"Application startup complete", "first"),
]

PHASES = [
    ("1 container + prelude + preflight + import + distributed init",
     "CONTAINER_START", "weight load start"),
    ("2 MAIN WEIGHT LOAD", "weight load start", "weight load end"),
    ("3 drafter (DFlash2) + load close-out",
     "weight load end", "model load total"),
    ("4 profile run (graph memory profile + first NCCL + MLA tune)",
     "model load total", "KV pool"),
    ("5 KV pool -> end of graph capture (warm-up + autotune + capture)",
     "KV pool", "graph capture end"),
    ("5a   -- of which: FlashInfer autotune", "autotune start", "autotune end"),
    ("6 engine core close-out", "graph capture end", "init engine"),
    ("7 API server", "init engine", "API ready"),
]

EVIDENCE = [
    r"Loading weights took", r"Model loading took", r"init engine \(profile",
    r"GPU KV cache size", r"Graph capturing finished", r"Autotuning process",
    r"Saved \d+ configs", r"Skipping FlashInfer autotune", r"EP weight filter",
    r"harem-epfilter\] skipped 2", r"harem-fastload", r"Auto-prefetch",
    r"Available KV cache memory", r"Free memory on device",
]


def parse(lines):
    seen = {}
    last = None
    for raw in lines:
        m = TS.match(raw)
        if m:
            last = dt.datetime.strptime(m.group(1)[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(
                tzinfo=dt.timezone.utc
            )
            t, body = last, m.group(2)
        elif last is not None:
            # tqdm writes with a leading \r, so docker stamps an empty record and
            # puts the text on the next physical line. Inherit the stamp.
            t, body = last, raw
        else:
            continue
        for label, pat, mode in MARKS:
            if re.search(pat, body):
                if mode == "first":
                    seen.setdefault(label, t)
                else:
                    seen[label] = t
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node")
    ap.add_argument("--name", default="exl3-tp3")
    ap.add_argument("--start", help="container StartedAt (RFC3339); queried if --node")
    a = ap.parse_args()
    if a.node:
        lines = subprocess.run(
            ["ssh", "-n", a.node, f"docker logs -t {a.name} 2>&1"],
            capture_output=True, text=True,
        ).stdout.split("\n")
        start = subprocess.run(
            ["ssh", "-n", a.node,
             f"docker inspect -f '{{{{.State.StartedAt}}}}' {a.name}"],
            capture_output=True, text=True,
        ).stdout.strip()
    else:
        lines = sys.stdin.read().split("\n")
        start = a.start
    seen = parse(lines)
    if start:
        seen["CONTAINER_START"] = dt.datetime.fromisoformat(
            start.replace("Z", "+00:00")
        ).astimezone(dt.timezone.utc)

    print(f"container started : {seen.get('CONTAINER_START')}")
    print(f"{'phase':<62}{'baslangic':<15}{'bitis':<15}{'seconds_s':>8}")
    for label, a0, a1 in PHASES:
        t0, t1 = seen.get(a0), seen.get(a1)
        if not t0 or not t1 or t1 < t0:
            print(f"{label:<62}{'-':<15}{'-':<15}{'-':>8}")
            continue
        print(f"{label:<62}{t0.strftime('%H:%M:%S.%f')[:12]:<15}"
              f"{t1.strftime('%H:%M:%S.%f')[:12]:<15}{(t1 - t0).total_seconds():>8.1f}")
    t0, t1 = seen.get("CONTAINER_START"), seen.get("API ready")
    tot = (t1 - t0).total_seconds() if t0 and t1 else float("nan")
    print(f"{'TOTAL (container start -> API ready)':<62}{'':<30}{tot:>8.1f}")
    print()
    for label, _, _ in MARKS:
        if label in seen:
            print(f"  isaret {label:<30} {seen[label].strftime('%H:%M:%S.%f')[:12]}")
    print()
    for pat in EVIDENCE:
        for raw in lines:
            if re.search(pat, raw):
                print("  |", raw.strip()[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())

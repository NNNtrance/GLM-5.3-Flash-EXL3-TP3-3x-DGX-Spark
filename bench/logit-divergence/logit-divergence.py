#!/usr/bin/env python3
"""logit-divergence.py -- per-position logprob divergence probe for a vLLM
OpenAI-compatible server. Replaces determinism-check.py as the PDL race detector.

WHY THIS EXISTS (and why byte identity does not work)
------------------------------------------------------
The first plan for detecting a Programmatic Dependent Launch (PDL) race on sm_12x
hardware was byte-for-byte identity of greedy completions: same prompt, temperature 0,
fixed seed, run it 12 times, any difference is evidence of a race. The cuda-exl3
author ran exactly that experiment on his own hardware first and reported (Zeuss5/
cuda-exl3 issue #6) 24 distinct completions out of 24 greedy runs -- in BOTH arms,
PDL on and PDL off. The test has no discriminating power on a stack like ours, for
three independent reasons, none of them PDL:

  1. cuda-exl3's fused MoE epilogue accumulates each routed row into its token's row
     with a bf16 `atomicAdd`. Atomic order is not fixed, so the MoE stage is not
     bit-exact run to run on ANY MoE model -- which is every model we serve.
  2. `CUDA_EXL3_DETERMINISTIC=1` never covered that path; it only disabled split-k on
     the dense path. His fix `e7e345e` routes the deterministic path back through
     `exl3_moe_combine`, which sums each token's top-k in a fixed k order.
  3. Our TP=3 all-reduce is not bit-exact either, and no flag of his changes that.

Measured per-layer relative error against an fp64 reference, at top_k=8 (GLM-5.3-Flash):
fused epilogue 4.34e-3, `exl3_moe_combine` 1.67e-3. So the arithmetic floor a race
would have to stand out from is ~4.3e-3 per layer with the fused path live, and a
run-to-run *varying* part of that floor exists whether or not PDL is on.

So: stop asking "is the text identical" (it never will be) and ask "how big is the
run-to-run noise, and does turning PDL off change its size". This script measures the
noise floor and then looks for outliers against it.

WHAT IT DOES
------------
`run` sends a fixed set of 12 prompts (4 code, 4 prose, 2 math, 2 JSON/tool-like,
stored as a deterministic file on disk) at temperature 0 with a fixed seed and
max_tokens 128, asking for `logprobs` with `top_logprobs=5`, and records for every
generated position: the chosen token, its logprob, and the top-5 alternatives with
their logprobs. It repeats the whole set N times (default 2) and writes one JSON per
run into an output directory. One directory = one arm.

`compare` takes those directories and computes, for each prompt and each pair of runs:

  * first divergence position   -- first position where the chosen token differs, or
                                  "none". A length change with no earlier token
                                  difference is reported at the shorter length and
                                  marked `len`.
  * max |dlogprob|              -- the largest absolute difference in the chosen
                                  token's logprob over the positions BEFORE the first
                                  divergence (after it, the two runs are on different
                                  trajectories and a positional comparison is
                                  meaningless).
  * median / p95 / max          -- the distribution of |dlogprob| over those same
                                  aligned positions. This is the noise floor.
  * top-5 max |dlogprob|        -- same, over the alternatives that appear in both
                                  runs' top-5 at a position. A race can move a
                                  non-argmax logit visibly while the argmax survives.
  * top-5 membership changes    -- positions where the two runs' top-5 token SETS
                                  differ, which is itself a divergence signal.

Per arm it then pools every per-position |dlogprob| from every prompt into one
distribution: median, p95, max. That pooled p95 is the arm's floor.

With two arms it repeats the whole thing between run 1 of arm A and run 1 of arm B and
flags a prompt as an OUTLIER when either:

  * its between-arm max |dlogprob| exceeds K x the pooled within-arm p95 (K default 4), or
  * it diverges in token sequence earlier than ANY within-arm pair did.

HOW TO RUN IT (the intended arms)
----------------------------------
The diagnostic arm runs on an image built at cuda-exl3 `e7e345e` with
`CUDA_EXL3_DETERMINISTIC=1`, so the MoE stage is deterministic and the only remaining
within-arm difference is the all-reduce. Production stays at `754421f`.

  ./logit-divergence.py run --host 192.0.2.10 --port 8000 \
        --model glm-5.3-flash --arm pdl-on  --runs 2 --out-dir ./ld/pdl-on
  # restart the engine with HAREM_PDL_SM12=0 -- the gate is read at import time
  ./logit-divergence.py run --host 192.0.2.10 --port 8000 \
        --model glm-5.3-flash --arm pdl-off --runs 2 --out-dir ./ld/pdl-off
  ./logit-divergence.py compare ./ld/pdl-on ./ld/pdl-off

`compare` also accepts two run JSON files (or two directories holding one run each)
for a single-arm floor measurement, and four run files for the full two-arm form.

REQUEST SHAPE
-------------
The same shape the rest of this stack uses (see nvfp4e/bench-sweep-nvfp4e.py and
nvfp4e/toolcall-gate.py): `chat_template_kwargs = {"enable_thinking": true,
"reasoning_effort": "low"}`. This model has NO way to disable thinking;
`enable_thinking: false` is never sent. Because thinking is on, a 128-token budget may
be spent entirely inside the reasoning stream -- that is fine and arguably better for
a divergence probe, since it is still the model's own trajectory, but it does mean the
`content` field is often empty. Positions are counted from whatever the server returns
logprobs for.

If the server's chat endpoint does not return usable logprobs, the script detects that
ONCE at startup and falls back to the legacy `/v1/completions` path with `logprobs: 5`
and `echo: false`. That path sends the RAW prompt with no chat template, so no system
message and no reasoning directive apply -- a legitimate measurement of the same
kernels, but not the same prompt. It is recorded in the run metadata as
`transport: "legacy_completions"` and `compare` refuses to mix transports.

PREFIX CACHE
------------
Run 2 of an arm would otherwise be a prefix-cache hit and skip prefill entirely, so
the two runs would not exercise the same code path and a prefill-side race could hide.
The script therefore sends a per-run `cache_salt` (vLLM's prefix-cache isolation
field), which makes every run recompute its prefill. If the server rejects the field
it is detected once, dropped, and recorded as `cache_salt: false` -- read the results
knowing run 2 was a cache hit in that case.

CONCURRENCY
-----------
Requests are strictly sequential. Concurrency changes batching, and a divergence probe
that mixes batch shapes measures batching, not the kernel. There is no flag for it.

A CLEAN RESULT IS NOT A PROOF
------------------------------
"Between-arm max sits inside the within-arm floor" means this prompt set, at this
token budget, at concurrency 1, did not separate the arms. Races can be timing-,
batch- or size-dependent. It is evidence, not a clearance certificate.

EXIT CODES
----------
  run:      0 ok, 2 operational error (connection, HTTP, timeout surviving retries)
  compare:  0 no outlier prompts, 1 outliers flagged, 2 operational/input error

Standard library only -- urllib, json, argparse, hashlib, statistics, math, glob.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "harem.logit-divergence/1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.join(SCRIPT_DIR, "prompts-logit-divergence.jsonl")
SYSTEM_MESSAGE = "You are a helpful assistant."
RETRIES = 3

# --------------------------------------------------------------------------------------
# The fixed prompt set: 4 code, 4 prose, 2 math, 2 JSON/tool-like.
#
# This constant is the source of truth. On first use it is materialised to
# prompts-logit-divergence.jsonl next to this script (deterministic bytes: sorted keys,
# ASCII-escaped, one object per line) and from then on the FILE is what is read and
# hashed, so a run can be reproduced from the artefact alone and a silent edit to
# either side shows up as a sha256 mismatch in `compare`.
#
# Every prompt is written to run past 128 tokens, so all runs fill the budget and every
# prompt contributes the same number of positions.
# --------------------------------------------------------------------------------------
PROMPT_SET: List[Tuple[str, str, str]] = [
    ("code1", "code",
     "Write a Python function that merges overlapping intervals. The input is a list "
     "of [start, end] integer pairs, not necessarily sorted, with start <= end. Return "
     "a new list of non-overlapping intervals sorted by start. Include a docstring "
     "explaining the algorithm and its time complexity, full type hints, and handling "
     "for the empty list, a single interval, intervals that all merge into one, and "
     "intervals that only touch at their endpoints."),
    ("code2", "code",
     "Write a Python class implementing an LRU cache with a fixed capacity, supporting "
     "get(key) and put(key, value) in O(1) average time using a dictionary plus a "
     "doubly linked list you implement yourself. Do not use functools.lru_cache or "
     "collections.OrderedDict. Include a docstring explaining the design, type hints "
     "throughout, and a short __main__ block demonstrating capacity eviction and "
     "re-accessing a key to keep it fresh."),
    ("code3", "code",
     "Write a Python generator that yields prime numbers using a segmented sieve of "
     "Eratosthenes, so it can reach very large bounds without ever allocating one "
     "boolean array of that size. Explain in the docstring how the segmentation works "
     "and why it saves memory compared with a plain sieve, add a helper that computes "
     "the base primes up to sqrt(limit), and use type hints throughout."),
    ("code4", "code",
     "Write a Python function that parses a simplified INI configuration format from a "
     "string: section headers in square brackets, key = value lines, comments starting "
     "with a semicolon or a hash, blank lines ignored, and values that may be quoted. "
     "Return a dict of dicts. Raise a clear custom exception with the line number for "
     "a malformed line. Include a docstring, type hints, and notes on the edge cases "
     "you decided to allow."),

    ("prose1", "prose",
     "Explain, in a few clear paragraphs for a software engineer who has never worked "
     "with GPUs, what a race condition is in the context of concurrent GPU kernel "
     "execution, why greedy decoding at temperature zero is a useful diagnostic for "
     "finding one in an inference server, and what output divergence across repeated "
     "identical requests can and cannot prove about the presence of a race."),
    ("prose2", "prose",
     "Write a vivid descriptive passage of at least 300 words about a coastal fishing "
     "village at dawn, just as the boats are heading out for the day. Focus on sensory "
     "detail -- sound, light, smell, texture -- rather than plot or dialogue, and keep "
     "a single unhurried narrative voice throughout."),
    ("prose3", "prose",
     "Explain to a careful non-specialist reader why floating point addition is not "
     "associative, what that means for a parallel reduction whose order is not fixed, "
     "and how someone measuring a numerical result should tell the difference between "
     "harmless rounding noise and an actual bug. Use concrete examples with small "
     "numbers and avoid formulae the reader would have to look up."),
    ("prose4", "prose",
     "Write a calm, well-organised summary of how a small engineering team should "
     "decide whether an optimisation is worth keeping: what to measure before and "
     "after, how to decide whether a difference is real or noise, what to write down, "
     "and how to record what the change cost as well as what it gained. Address it to "
     "a team that has been burned before by a change that looked faster and was not."),

    ("math1", "math",
     "Solve step by step, showing all intermediate work. A factory has three machines. "
     "A alone completes a production run in 12 hours, B alone in 15 hours, C alone in "
     "20 hours. All three start together at 8:00 AM. After 3 hours B breaks down and "
     "stops. Ninety minutes after that, A stops for maintenance, leaving C to finish "
     "alone. At what clock time is the run completed? Give the fraction of the job "
     "done in each phase and verify the three fractions sum to 1."),
    ("math2", "math",
     "Solve step by step, showing all work. A cyclist rides from town P to town Q at a "
     "constant 24 km/h and returns along the same road at a constant 16 km/h. The "
     "whole trip, excluding a 25 minute rest at Q, takes 4 hours and 10 minutes. Find "
     "the distance between the towns, the time spent in each direction, and the "
     "average speed over the whole ride excluding the rest. State clearly why the "
     "average speed is not 20 km/h."),

    ("json1", "json",
     "Return a single JSON object, and nothing else, describing a fictional library "
     "catalogue entry. It must have the keys: id (string), title (string), authors "
     "(array of objects with given_name and family_name), year (integer), subjects "
     "(array of at least four strings), available (boolean), and location (object with "
     "floor as an integer and shelf as a string). Use plausible values, keep the "
     "structure exactly as described, and do not wrap the object in a code fence."),
    ("json2", "json",
     "You are given a tool with the signature search_flights(origin: string, "
     "destination: string, depart_date: string in YYYY-MM-DD, passengers: integer, "
     "cabin: one of economy|premium|business). A user says: 'Two of us want to fly "
     "from Istanbul to Amsterdam on the 14th of next March, business class if it is "
     "not silly money.' Write out the exact JSON arguments object you would pass to "
     "the tool, then explain in a short paragraph each choice you made and which single "
     "piece of information you had to assume rather than being told."),
]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class OperationalError(Exception):
    """Connection error, HTTP error, bad input, or a timeout that survived retries."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def pct(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile of an already sorted sequence. q in [0, 1]."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


def dist(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    s = sorted(vals)
    return {
        "n": len(s),
        "median": pct(s, 0.50),
        "p95": pct(s, 0.95),
        "max": (float(s[-1]) if s else None),
    }


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "-"
    if v == 0:
        return "0"
    return f"{v:.3e}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "  (no rows)"
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    out = [line, sep]
    for r in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# prompt set on disk
# --------------------------------------------------------------------------------------
def prompt_file_bytes() -> bytes:
    lines = []
    for pid, cat, text in PROMPT_SET:
        lines.append(json.dumps(
            {"id": pid, "category": cat, "prompt": text},
            sort_keys=True, ensure_ascii=True,
        ))
    return ("\n".join(lines) + "\n").encode("utf-8")


def ensure_prompt_file(path: str, force: bool = False) -> None:
    if force or not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(prompt_file_bytes())
        log(f"wrote prompt set to {path}")


def load_prompts(path: str) -> List[Dict[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise OperationalError(f"{path}:{n}: not valid JSON: {e}")
            for k in ("id", "category", "prompt"):
                if k not in d:
                    raise OperationalError(f"{path}:{n}: missing key '{k}'")
            out.append({"id": d["id"], "category": d["category"], "prompt": d["prompt"]})
    if not out:
        raise OperationalError(f"{path}: no prompts")
    ids = [d["id"] for d in out]
    if len(set(ids)) != len(ids):
        raise OperationalError(f"{path}: duplicate prompt ids")
    return out


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------
def http_post(url: str, payload: dict, timeout: float,
              retries: int = RETRIES, quiet: bool = False) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                detail = ""
            last = f"HTTP {e.code} {e.reason}: {detail}".strip()
            if e.code == 400:          # a rejected field will not fix itself on retry
                raise OperationalError(f"POST {url}: {last}")
        except urllib.error.URLError as e:
            last = f"connection error: {e.reason}"
        except Exception as e:                       # noqa: BLE001 - report and retry
            last = f"{type(e).__name__}: {e}"
        if not quiet:
            log(f"  ! POST {url} attempt {attempt}/{retries} failed: {last}")
        if attempt < retries:
            time.sleep(min(2 ** attempt, 8))
    raise OperationalError(f"POST {url} failed after {retries} attempts: {last}")


def http_get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def discover_model(base: str, timeout: float) -> str:
    data = http_get(f"{base}/v1/models", timeout)
    models = data.get("data") or []
    if not models or not models[0].get("id"):
        raise OperationalError(f"GET {base}/v1/models returned no usable model id")
    return models[0]["id"]


# --------------------------------------------------------------------------------------
# payloads and parsing
# --------------------------------------------------------------------------------------
def chat_payload(model: str, prompt: str, args, salt: Optional[str],
                 max_tokens: Optional[int] = None) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else args.max_tokens,
        "seed": args.seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
        # This model has no way to disable thinking; enable_thinking=false is NEVER
        # sent. Same shape as bench-sweep-nvfp4e.py / toolcall-gate.py.
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": args.reasoning_effort,
        },
    }
    if salt:
        body["cache_salt"] = salt
    return body


def legacy_payload(model: str, prompt: str, args, salt: Optional[str],
                   max_tokens: Optional[int] = None) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else args.max_tokens,
        "seed": args.seed,
        "stream": False,
        "logprobs": args.top_logprobs,
        "echo": False,
    }
    if salt:
        body["cache_salt"] = salt
    return body


def parse_chat_positions(resp: dict) -> Optional[List[dict]]:
    """-> [{'t': token, 'b': bytes|None, 'lp': float, 'top': [[tok, lp], ...]}] or None."""
    choices = resp.get("choices") or []
    if not choices:
        return None
    lp = choices[0].get("logprobs") or {}
    content = lp.get("content")
    if not content:
        return None
    out = []
    for entry in content:
        if "logprob" not in entry:
            return None
        top = [[t.get("token"), float(t.get("logprob"))]
               for t in (entry.get("top_logprobs") or [])
               if t.get("logprob") is not None]
        out.append({
            "t": entry.get("token"),
            "b": entry.get("bytes"),
            "lp": float(entry["logprob"]),
            "top": top,
        })
    return out


def parse_legacy_positions(resp: dict) -> Optional[List[dict]]:
    choices = resp.get("choices") or []
    if not choices:
        return None
    lp = choices[0].get("logprobs") or {}
    toks = lp.get("tokens")
    tlps = lp.get("token_logprobs")
    tops = lp.get("top_logprobs") or []
    if not toks or tlps is None:
        return None
    out = []
    for i, tok in enumerate(toks):
        val = tlps[i] if i < len(tlps) else None
        if val is None:                 # echo=False should never give a None here
            continue
        d = tops[i] if i < len(tops) and isinstance(tops[i], dict) else {}
        top = sorted(([k, float(v)] for k, v in d.items()),
                     key=lambda kv: -kv[1])
        out.append({"t": tok, "b": None, "lp": float(val), "top": top})
    return out or None


def response_meta(resp: dict) -> Dict[str, Any]:
    choices = resp.get("choices") or [{}]
    usage = resp.get("usage") or {}
    msg = choices[0].get("message") or {}
    return {
        "finish_reason": choices[0].get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "has_reasoning_content": bool(msg.get("reasoning_content")),
        "content_len": len(msg.get("content") or choices[0].get("text") or ""),
    }


# --------------------------------------------------------------------------------------
# transport detection, done ONCE
# --------------------------------------------------------------------------------------
def detect_transport(base: str, model: str, probe_prompt: str, args) -> Tuple[str, bool]:
    """Returns (transport, cache_salt_supported). Raises if no logprob path works."""
    chat_url = f"{base}/v1/chat/completions"
    salt_ok = True
    log("detecting transport (once) ...")

    def try_chat(with_salt: bool):
        payload = chat_payload(model, probe_prompt, args,
                               "harem-ld-probe" if with_salt else None, max_tokens=16)
        return http_post(chat_url, payload, args.timeout, retries=1, quiet=True)

    resp = None
    try:
        resp = try_chat(True)
    except OperationalError as e:
        log(f"  chat probe with cache_salt failed: {str(e)[:200]}")
        salt_ok = False
        try:
            resp = try_chat(False)
        except OperationalError as e2:
            log(f"  chat probe without cache_salt failed: {str(e2)[:200]}")
            resp = None

    if resp is not None:
        pos = parse_chat_positions(resp)
        if pos:
            log(f"  transport = chat_completions, cache_salt = {salt_ok}, "
                f"{len(pos)} probe positions")
            return "chat_completions", salt_ok
        log("  chat endpoint answered but returned no usable logprobs "
            "(a reasoning parser may be consuming them) -- trying legacy")

    legacy_url = f"{base}/v1/completions"
    for with_salt in ((True, False) if salt_ok else (False,)):
        try:
            r = http_post(legacy_url,
                          legacy_payload(model, probe_prompt, args,
                                         "harem-ld-probe" if with_salt else None,
                                         max_tokens=16),
                          args.timeout, retries=1, quiet=True)
        except OperationalError as e:
            log(f"  legacy probe (salt={with_salt}) failed: {str(e)[:200]}")
            continue
        pos = parse_legacy_positions(r)
        if pos:
            log(f"  transport = legacy_completions, cache_salt = {with_salt}, "
                f"{len(pos)} probe positions")
            log("  WARNING: the legacy path sends the RAW prompt -- no chat template, "
                "no system message, no reasoning_effort directive.")
            return "legacy_completions", with_salt

    raise OperationalError(
        "neither /v1/chat/completions nor /v1/completions returned usable logprobs; "
        "there is nothing to compare"
    )


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def do_one(base: str, model: str, prompt: str, args, transport: str,
           salt: Optional[str]) -> Tuple[List[dict], Dict[str, Any], float]:
    t0 = time.time()
    if transport == "chat_completions":
        resp = http_post(f"{base}/v1/chat/completions",
                         chat_payload(model, prompt, args, salt), args.timeout)
        pos = parse_chat_positions(resp)
    else:
        resp = http_post(f"{base}/v1/completions",
                         legacy_payload(model, prompt, args, salt), args.timeout)
        pos = parse_legacy_positions(resp)
    elapsed = time.time() - t0
    if not pos:
        raise OperationalError(
            "server returned a response with no logprob positions; the transport "
            "detected at startup stopped working mid-run -- refusing to record it"
        )
    return pos, response_meta(resp), elapsed


def cmd_run(args) -> int:
    base = f"http://{args.host}:{args.port}"
    ensure_prompt_file(args.prompts, force=args.write_prompts)
    prompts = load_prompts(args.prompts)
    pset_sha = sha256_file(args.prompts)

    counts: Dict[str, int] = {}
    for p in prompts:
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    log(f"prompt set: {len(prompts)} prompts "
        f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}), "
        f"sha256 {pset_sha[:16]}")

    try:
        model = args.model or discover_model(base, args.timeout)
    except Exception as e:                            # noqa: BLE001
        raise OperationalError(f"model discovery failed: {e}")
    log(f"model: {model}")

    transport, salt_ok = detect_transport(base, model, prompts[0]["prompt"], args)
    if not salt_ok:
        log("  NOTE: no cache_salt -> run 2 onward will be a prefix-cache hit and "
            "will not recompute prefill.")

    if args.warmup:
        log("warmup: one throwaway request (not recorded) ...")
        do_one(base, model, prompts[0]["prompt"], args, transport,
               "harem-ld-warmup" if salt_ok else None)
        log("warmup: done")

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for run_idx in range(1, args.runs + 1):
        salt = f"harem-ld-{args.arm}-r{run_idx}" if salt_ok else None
        recs = []
        log(f"arm '{args.arm}' run {run_idx}/{args.runs}")
        for p in prompts:
            pos, meta, elapsed = do_one(base, model, p["prompt"], args, transport, salt)
            recs.append({
                "id": p["id"],
                "category": p["category"],
                "prompt_sha256": sha256_bytes(p["prompt"].encode("utf-8")),
                "n_positions": len(pos),
                "elapsed_s": round(elapsed, 3),
                **meta,
                "positions": pos,
            })
            log(f"  {p['id']:<7} {p['category']:<5} positions={len(pos):<4} "
                f"finish={meta['finish_reason']} {elapsed:6.2f}s")
        out = {
            "meta": {
                "schema": SCHEMA,
                "arm": args.arm,
                "label": args.label,
                "run_index": run_idx,
                "runs_total": args.runs,
                "host": args.host,
                "port": args.port,
                "model": model,
                "transport": transport,
                "cache_salt": salt,
                "temperature": 0,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "top_logprobs": args.top_logprobs,
                "reasoning_effort": args.reasoning_effort,
                "chat_template_kwargs": {"enable_thinking": True,
                                         "reasoning_effort": args.reasoning_effort},
                "concurrency": 1,
                "warmup": args.warmup,
                "prompt_file": os.path.abspath(args.prompts),
                "prompt_set_sha256": pset_sha,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "prompts": recs,
        }
        path = os.path.join(args.out_dir, f"run-{run_idx:02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        written.append(path)
        log(f"  -> {path}")

    print(f"arm '{args.arm}': {args.runs} run(s), {len(prompts)} prompts, "
          f"transport {transport} -> {args.out_dir}")
    for p in written:
        print(f"  {p}")
    return 0


# --------------------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------------------
def pos_key(p: dict):
    """Token identity. Prefer the byte list (exact); fall back to the token string."""
    b = p.get("b")
    if b:
        return ("b", tuple(b))
    return ("t", p.get("t"))


def compare_prompt(a: dict, b: dict) -> Dict[str, Any]:
    pa, pb = a["positions"], b["positions"]
    n = min(len(pa), len(pb))

    first_div: Optional[int] = None
    kind: Optional[str] = None
    for i in range(n):
        if pos_key(pa[i]) != pos_key(pb[i]):
            first_div, kind = i, "token"
            break
    if first_div is None and len(pa) != len(pb):
        first_div, kind = n, "len"

    limit = n if first_div is None else first_div
    chosen = [abs(pa[i]["lp"] - pb[i]["lp"]) for i in range(limit)]

    top_deltas: List[float] = []
    set_changes = 0
    for i in range(limit):
        da = {t: lp for t, lp in pa[i]["top"]}
        db = {t: lp for t, lp in pb[i]["top"]}
        shared = set(da) & set(db)
        if set(da) != set(db):
            set_changes += 1
        for t in shared:
            top_deltas.append(abs(da[t] - db[t]))

    unaligned = [abs(pa[i]["lp"] - pb[i]["lp"]) for i in range(n)]

    d = dist(chosen)
    return {
        "id": a["id"],
        "category": a["category"],
        "n_a": len(pa),
        "n_b": len(pb),
        "compared_positions": limit,
        "first_divergence": first_div,
        "divergence_kind": kind,
        "max_abs_dlogprob": d["max"],          # over positions before first divergence
        "median_abs_dlogprob": d["median"],
        "p95_abs_dlogprob": d["p95"],
        "top5_max_abs_dlogprob": (max(top_deltas) if top_deltas else None),
        "top5_membership_changes": set_changes,
        "all_positions_unaligned": dist(unaligned),
        "_chosen": chosen,                      # pooled by the caller, dropped from JSON
        "_top": top_deltas,
    }


def pair_report(run_a: dict, run_b: dict) -> Dict[str, Any]:
    by_b = {p["id"]: p for p in run_b["prompts"]}
    rows, pooled, pooled_top = [], [], []
    for pa in run_a["prompts"]:
        pb = by_b.get(pa["id"])
        if pb is None:
            raise OperationalError(f"prompt '{pa['id']}' missing from the other run")
        if pa["prompt_sha256"] != pb["prompt_sha256"]:
            raise OperationalError(f"prompt '{pa['id']}' text differs between runs")
        r = compare_prompt(pa, pb)
        pooled.extend(r.pop("_chosen"))
        pooled_top.extend(r.pop("_top"))
        rows.append(r)
    divs = [r["first_divergence"] for r in rows if r["first_divergence"] is not None]
    return {
        "a": f"{run_a['meta']['arm']}/run{run_a['meta']['run_index']}",
        "b": f"{run_b['meta']['arm']}/run{run_b['meta']['run_index']}",
        "prompts": rows,
        "pooled": dist(pooled),
        "pooled_top5": dist(pooled_top),
        "first_divergence_min": (min(divs) if divs else None),
        "diverged_prompts": len(divs),
        # raw per-position values, so several pairs can be pooled into one floor.
        # Stripped before anything is serialised (strip_raw).
        "_vals": pooled,
        "_top_vals": pooled_top,
    }


def strip_raw(rep: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def load_run(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise OperationalError(f"cannot read run file {path}: {e}")
    if d.get("meta", {}).get("schema") != SCHEMA:
        raise OperationalError(f"{path}: not a {SCHEMA} run file")
    d["_path"] = path
    return d


def expand_paths(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "run-*.json")))
            if not files:
                files = sorted(f for f in glob.glob(os.path.join(p, "*.json")))
            if not files:
                raise OperationalError(f"{p}: directory holds no run JSON files")
            out.extend(files)
        elif os.path.exists(p):
            out.append(p)
        else:
            raise OperationalError(f"{p}: no such file or directory")
    return out


def check_compatible(runs: Sequence[dict]) -> List[str]:
    warns = []
    ref = runs[0]["meta"]
    hard = ["prompt_set_sha256", "transport"]
    soft = ["model", "max_tokens", "seed", "top_logprobs", "reasoning_effort",
            "temperature"]
    for r in runs[1:]:
        m = r["meta"]
        for k in hard:
            if m.get(k) != ref.get(k):
                raise OperationalError(
                    f"{r['_path']}: {k} differs ({m.get(k)!r} vs {ref.get(k)!r}) -- "
                    f"these runs are not comparable"
                )
        for k in soft:
            if m.get(k) != ref.get(k):
                warns.append(f"{k} differs: {ref.get(k)!r} vs {m.get(k)!r} "
                             f"({os.path.basename(r['_path'])})")
    if ref.get("transport") == "legacy_completions":
        warns.append("transport is legacy_completions: raw prompts, no chat template, "
                     "no reasoning_effort directive")
    if not ref.get("cache_salt"):
        warns.append("no cache_salt was used: repeat runs were prefix-cache hits and "
                     "did not recompute prefill")
    return warns


def print_pair_table(title: str, rep: Dict[str, Any],
                     ratio_vs: Optional[float] = None,
                     flags: Optional[Dict[str, str]] = None) -> None:
    print()
    print(title)
    headers = ["prompt", "cat", "n_pos", "cmp", "first_div", "median|dlp|",
               "p95|dlp|", "max|dlp|", "top5 max", "top5 set"]
    if ratio_vs is not None:
        headers += ["K ratio", "flag"]
    rows = []
    for r in rep["prompts"]:
        fd = "none"
        if r["first_divergence"] is not None:
            fd = str(r["first_divergence"])
            if r["divergence_kind"] == "len":
                fd += " (len)"
        row = [
            r["id"], r["category"],
            f"{r['n_a']}/{r['n_b']}",
            str(r["compared_positions"]),
            fd,
            fmt(r["median_abs_dlogprob"]),
            fmt(r["p95_abs_dlogprob"]),
            fmt(r["max_abs_dlogprob"]),
            fmt(r["top5_max_abs_dlogprob"]),
            str(r["top5_membership_changes"]),
        ]
        if ratio_vs is not None:
            mx = r["max_abs_dlogprob"]
            if mx is None or not ratio_vs:
                row += ["-", (flags or {}).get(r["id"], "")]
            else:
                row += [f"{mx / ratio_vs:.1f}x", (flags or {}).get(r["id"], "")]
        rows.append(row)
    print(format_table(headers, rows))
    p = rep["pooled"]
    print(f"  pooled: n={p['n']}  median={fmt(p['median'])}  p95={fmt(p['p95'])}  "
          f"max={fmt(p['max'])}   (top-5 pooled p95={fmt(rep['pooled_top5']['p95'])})")


def cmd_compare(args) -> int:
    files = expand_paths(args.paths)
    if len(files) not in (2, 4):
        raise OperationalError(
            f"compare needs 2 run files (one arm) or 4 (two arms); got {len(files)}: "
            + ", ".join(os.path.basename(f) for f in files)
        )
    runs = [load_run(f) for f in files]
    warns = check_compatible(runs)

    # group by arm label, ordered by run_index; fall back to positional pairing
    arms: Dict[str, List[dict]] = {}
    for r in runs:
        arms.setdefault(r["meta"]["arm"], []).append(r)
    for v in arms.values():
        v.sort(key=lambda r: r["meta"]["run_index"])

    positional = False
    if len(runs) == 4 and (len(arms) != 2 or any(len(v) != 2 for v in arms.values())):
        warns.append("arm labels do not split 2+2; pairing the first two files as arm "
                     "A and the last two as arm B")
        arms = {runs[0]["meta"]["arm"] + "#1": runs[:2],
                runs[2]["meta"]["arm"] + "#2": runs[2:]}
        positional = True

    names = list(arms.keys())
    print("=" * 78)
    print("LOGIT DIVERGENCE REPORT")
    print("=" * 78)
    m = runs[0]["meta"]
    print(f"model {m['model']}   transport {m['transport']}   temp 0   seed {m['seed']}"
          f"   max_tokens {m['max_tokens']}   top_logprobs {m['top_logprobs']}")
    print(f"prompt set sha256 {m['prompt_set_sha256'][:16]}   "
          f"reasoning_effort {m['reasoning_effort']}   concurrency 1")
    for f, r in zip(files, runs):
        print(f"  {r['meta']['arm']:<12} run {r['meta']['run_index']}  "
              f"{r['meta'].get('generated_at_utc', '-')}  {os.path.basename(f)}")
    for w in warns:
        print(f"  WARNING: {w}")

    within: Dict[str, Dict[str, Any]] = {}
    for name in names:
        rs = arms[name]
        if len(rs) >= 2:
            within[name] = pair_report(rs[0], rs[1])

    for name in names:
        if name in within:
            print_pair_table(
                f"Within-arm noise floor -- arm '{name}' "
                f"(run {arms[name][0]['meta']['run_index']} vs "
                f"run {arms[name][1]['meta']['run_index']})",
                within[name])

    # The floor is every within-arm per-position |dlogprob| from every arm and every
    # prompt, pooled into ONE distribution -- not an average of per-arm summaries.
    floor_vals: List[float] = []
    within_div_min: Optional[int] = None
    for rep in within.values():
        floor_vals.extend(rep["_vals"])
        if rep["first_divergence_min"] is not None:
            within_div_min = (rep["first_divergence_min"] if within_div_min is None
                              else min(within_div_min, rep["first_divergence_min"]))
    floor = dist(floor_vals)
    pooled_med, pooled_p95, pooled_max = floor["median"], floor["p95"], floor["max"]

    between = None
    outliers: List[str] = []
    if len(names) == 2:
        a_run1 = arms[names[0]][0]
        b_run1 = arms[names[1]][0]
        between = pair_report(a_run1, b_run1)
        flags: Dict[str, str] = {}
        for r in between["prompts"]:
            why = []
            mx = r["max_abs_dlogprob"]
            if pooled_p95 and mx is not None and mx > args.k * pooled_p95:
                why.append(f">{args.k}x floor")
            fd = r["first_divergence"]
            if fd is not None and (within_div_min is None or fd < within_div_min):
                why.append("earlier div")
            if why:
                flags[r["id"]] = "OUTLIER " + "+".join(why)
                outliers.append(r["id"])
        print_pair_table(
            f"Between-arm -- '{names[0]}' run 1 vs '{names[1]}' run 1",
            between, ratio_vs=pooled_p95, flags=flags)

    print()
    print("Summary")
    srows = []
    for name in names:
        if name in within:
            p = within[name]["pooled"]
            srows.append([f"within '{name}'", str(p["n"]), fmt(p["median"]),
                          fmt(p["p95"]), fmt(p["max"]),
                          ("none" if within[name]["first_divergence_min"] is None
                           else str(within[name]["first_divergence_min"]))])
    if len(within) > 1:
        srows.append(["within, pooled floor", str(floor["n"]), fmt(pooled_med),
                      fmt(pooled_p95), fmt(pooled_max),
                      ("none" if within_div_min is None else str(within_div_min))])
    if between is not None:
        p = between["pooled"]
        srows.append(["between arms", str(p["n"]), fmt(p["median"]), fmt(p["p95"]),
                      fmt(p["max"]),
                      ("none" if between["first_divergence_min"] is None
                       else str(between["first_divergence_min"]))])
    print(format_table(
        ["comparison", "positions", "median|dlp|", "p95|dlp|", "max|dlp|",
         "earliest token divergence"], srows))

    print()
    if between is not None:
        verdict = (f"VERDICT: within-arm floor p95 = {fmt(pooled_p95)}; "
                   f"between-arm max = {fmt(between['pooled']['max'])}; "
                   f"outlier prompts: "
                   f"{('[' + ', '.join(sorted(set(outliers))) + ']') if outliers else 'none'}")
    else:
        verdict = (f"VERDICT: within-arm floor p95 = {fmt(pooled_p95)}; "
                   f"between-arm max = n/a (single arm); outlier prompts: n/a")
    print(verdict)
    if between is not None and not outliers:
        print(f"         (K = {args.k}; no prompt exceeded {args.k} x the floor and "
              f"none diverged earlier than the within-arm minimum. Not a clearance "
              f"certificate -- see the module docstring.)")

    if args.out:
        payload = {
            "schema": "harem.logit-divergence.compare/1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": [{"path": os.path.abspath(f), "arm": r["meta"]["arm"],
                        "run_index": r["meta"]["run_index"]} for f, r in zip(files, runs)],
            "positional_pairing": positional,
            "warnings": warns,
            "k": args.k,
            "within": {k: strip_raw(v) for k, v in within.items()},
            "within_floor": {"n": floor["n"], "median": pooled_med,
                             "p95": pooled_p95, "max": pooled_max},
            "within_first_divergence_min": within_div_min,
            "between": (strip_raw(between) if between is not None else None),
            "outlier_prompts": sorted(set(outliers)),
            "verdict": verdict,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log(f"wrote comparison JSON to {args.out}")

    return 1 if outliers else 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def cmd_write_prompts(args) -> int:
    ensure_prompt_file(args.prompts, force=True)
    print(f"{os.path.abspath(args.prompts)}  sha256 {sha256_file(args.prompts)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="logit-divergence.py",
        description=("Per-position logprob divergence probe: measure the within-arm "
                     "noise floor of a TP inference stack, then look for outliers "
                     "between two arms (e.g. PDL on vs PDL off). See the module "
                     "docstring for why byte identity does not work here."),
        formatter_class=_Fmt,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="collect one arm's runs", formatter_class=_Fmt)
    r.add_argument("--host", default="127.0.0.1", help="server host/IP")
    r.add_argument("--port", type=int, required=True, help="server port")
    r.add_argument("--model", default=None,
                   help="model name to send; default: auto-discover via /v1/models")
    r.add_argument("--arm", required=True,
                   help="arm label recorded in every run file, e.g. pdl-on")
    r.add_argument("--label", default="", help="free-text note stored in the run file")
    r.add_argument("--runs", type=int, default=2, help="repeats of the whole prompt set")
    r.add_argument("--out-dir", required=True, help="directory to write run-NN.json into")
    r.add_argument("--prompts", default=DEFAULT_PROMPTS,
                   help="prompt set file; written from the built-in set if absent")
    r.add_argument("--write-prompts", action="store_true",
                   help="rewrite the prompt file from the built-in set, then run")
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--top-logprobs", type=int, default=5)
    r.add_argument("--reasoning-effort", default="low",
                   help="sent inside chat_template_kwargs; enable_thinking is always true")
    r.add_argument("--timeout", type=float, default=600, help="per-request timeout (s)")
    w = r.add_mutually_exclusive_group()
    w.add_argument("--warmup", dest="warmup", action="store_true",
                   help="send one throwaway request first (default)")
    w.add_argument("--no-warmup", dest="warmup", action="store_false")
    r.set_defaults(warmup=True, func=cmd_run)

    c = sub.add_parser(
        "compare",
        help="compare 2 run files/dirs (one arm) or 4 (two arms)",
        formatter_class=_Fmt)
    c.add_argument("paths", nargs="+",
                   help="run JSON files, or directories holding them (an arm directory "
                        "with 2 runs expands to those 2 runs)")
    c.add_argument("--k", type=float, default=4.0,
                   help="outlier threshold: between-arm max > K x within-arm p95")
    c.add_argument("--out", default=None, help="optional path for a comparison JSON")
    c.set_defaults(func=cmd_compare)

    pr = sub.add_parser("write-prompts", help="materialise the built-in prompt set",
                        formatter_class=_Fmt)
    pr.add_argument("--prompts", default=DEFAULT_PROMPTS)
    pr.set_defaults(func=cmd_write_prompts)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except OperationalError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

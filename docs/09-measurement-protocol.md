# 09 — Measurement protocol

Four different ways to measure a lie on this stack, and the protocol that stops each one. Every
retraction in [11](11-open-issues.md) is a case of one of these, and every one of them was published
before it was caught.

If you take one thing from this repository, take this page rather than any number on it.

---

## 1. Sweep rounds: five and discard two, or three if the tuner cache is warm

**The trap.** The MLA decode tuner keys on the batch shape and re-tunes when it sees a new one, at
about 15 ms per event, and every event is marked evicted. A boot mints 11 events before the server is
even up, and more as the sweep presents new shapes. **The warm-up window is sometimes longer than two
sweep rounds** — long enough to put a whole benchmark arm inside it `[measured-here]`.

**What that looked like.** An arm we were testing measured prefill 1,373 → 698 tok/s, prefill-fresh
1,672 → 936, C8 140 → 105. It looked like a disaster. Re-measured on the *same running engine* a few
minutes later, with nothing changed but the clock: prefill 1,483–1,515, prefill-fresh 1,720, C8
152.6 / 150.3 / 148.6. That arm was the winner by 12.6 %.

**The protocol.** Five `bench-sweep` rounds per arm. **Discard rounds 1 and 2.** Report the median of
rounds 3–5, and report the spread.

```
python3 scripts/bench-sweep.py --out sweep-1.json --label arm-round1 --think low
```

Repeat five times, then take medians. Rounds 3–5 of a settled engine are tight: on the 8-channel
configuration, C8 over five rounds read 151.7 / 150.1 / 152.4 / 150.8 / 147.0 `[measured-here]`.

**The shorter protocol, and what earns it.** Upstream now persists the tuner cache across processes
(`CUDA_EXL3_TUNE_CACHE`, [12](12-tuner-cache.md)). With it warm, a boot mints **zero** tune events
before serving and **zero** during a full sweep, and round 1 stops being a penalty: cold cache, C8
round 1 → round 3 is −3.4 %; warm cache it is +2.7 %, i.e. unordered noise `[measured-here]`. On an
image that has the cache **and a warm cache file on disk**, the protocol drops to **three rounds,
median of three** — about 15 minutes saved per arm.

Three conditions, all of them, or go back to five:

1. the image carries the persisted cache and `CUDA_EXL3_TUNE_CACHE` points at a directory that
   survives the container;
2. the cache file already exists from a previous boot of **this** image — the boot that writes it is
   still a five-round boot;
3. the three rounds agree within about 5 %. If they do not, that is a signal, not a number.

Every arm in this repository up to *fast boot S4* is a five-round median with two discarded; the ones
after it are three-round medians. The tables say which, and the two are not interchangeable.

### 1.1 Our own harness disagreed with its own documentation

The quick-arm script carried the header "5 rounds, the first 2 discarded" and its body ran **three**.
Applied literally, the rule would have left a median of one `[retracted]`. Nothing published here was
computed that way — the three-round arms were reported as medians of three, which is what they are —
but the discrepancy stood in the script for a day and is exactly the class of thing that quietly
turns into a wrong number.

Two things were checked before fixing it in the direction of the code rather than the header. First,
**the warm-up ramp is gone on a warm cache**: across the three settled arms of 5 September, round 1
was the *fastest* round twice. There is no ordered penalty left to discard. Second, the round-to-round
spread inside an arm is what §1.2 measures, and it is larger than most of the differences we have been
calling results.

**The quick tier is now one warm-up round plus three measured rounds, median of the three** — four
rounds in total, ~4 minutes more than the old three and the only version in which the "discard the
warm-up" sentence is true.

### 1.2 The noise band, per metric, and the rule that follows from it

Round-to-round spread within a single settled arm, three arms, warm tuner cache, nothing changed
between rounds `[measured-here]`:

| arm (rounds) | C1 | C2 | C4 | C6 | C8 |
|---|---|---|---|---|---|
| production 6 (3) | 57.24 / 56.85 / 56.10 → **1.9 %** | 84.43 / 82.95 / 84.17 → 1.7 % | 114.94 / 125.07 / 118.51 → **8.5 %** | 142.78 / 146.44 / 142.91 → 2.6 % | 167.31 / 170.22 / 168.95 → 1.7 % |
| mesh arm (3) | 56.15 / 56.35 / 58.51 → 4.2 % | 79.40 / 84.44 / 83.77 → 6.0 % | 123.46 / 125.11 / 119.32 → 4.7 % | 151.42 / 143.27 / 143.30 → 5.7 % | 175.03 / 171.07 / 170.45 → 2.7 % |
| draft page 256 (5) | 54.62…52.34 → 4.4 % | — | — | — | 167.83…162.00 → 3.6 % |

**The declared bands: C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %.** C4 is the noisiest column on
this stack and C8 the quietest, which is the opposite of the intuition that more streams means more
variance.

Two rules follow, and they apply to every table in this repository:

- **A difference of 3 % or less is written down as "equal".** Report the numbers, do not report a
  winner. This is a floor, not the whole test.
- **Above the floor, a difference still has to clear its own metric's band** before it is called a
  result — and if the decision matters, §2 applies as well: repeat it on a second boot.

Judgements already on record that this rule reclassifies as "equal": the combine-staging arm's
+2.3 % at C4, and patch 0007's −0.9…+4.2 %. One survives it: production 5 → 6 is +4.4 % at C1
(inside C1's band) but **+5.6 % at C8**, which clears C8's ±3 %, so that conclusion stands.

None of this replaces §2. The bands above are *within one boot*; boot-to-boot is 15.9 %.

## 2. Boot-to-boot variance is 16 %, so two rounds decide nothing

**The trap.** The same image, the same environment file, two separate boots `[measured-here]`:

| | boot 1 round 1 | boot 1 round 2 | boot 2 round 1 |
|---|---|---|---|
| C1 | 47.71 | 48.33 | 49.62 |
| C2 | 63.68 | 71.67 | 70.60 |
| C4 | 90.37 | 100.66 | 96.83 |
| C6 | 110.14 | 115.68 | 117.13 |
| C8 | **118.44** | **137.23** | **135.56** |

**15.9 % spread on C8 with nothing changed at all.** We once published a 10 % gap between two kernel
builds as an unexplained regression, and drew a kernel conclusion from it. It was this
`[retracted]`.

**The protocol.** A difference under about 5 % on this stack is not a result. Where a decision hangs
on one, take the means over five rounds **and** repeat on a second boot before writing it down.

## 3. Prefill measured on a repeated prompt is not a prefill measurement

**The trap.** `scripts/prefill-7k.py` uses two fixed seeds. Run it twice inside one boot and the
second run reads two whole 3,328-token blocks out of the prefix cache and reports up to **1,596
tok/s where the honest number is 1,025** `[measured-here]`.

**The protocol.** For any prefill claim, use `bench/prefill-fresh.py`, which draws a new seed per
request so the engine has never seen the prompt. Report the median of at least three, and say which
one you used. Both numbers appear in this repository and they are labelled: "prefill 7k" is the warm
repeated-prompt figure, "prefill-fresh" is the honest one.

```
python3 bench/prefill-fresh.py
```

## 4. Model-free first — the engine is the last place to look

Every conclusion in this repository that survived was reached with the engine down, on a
micro-benchmark using the real GLM-5.3-Flash shapes. Every conclusion drawn from two engine sweeps
alone has since been retracted.

The arithmetic is blunt: an engine boot on this stack takes about 11 minutes, and a full A/B arm
about 30. The model-free work that produced the two strongest results in this repository — rejecting
a protocol setting outright, and the per-kernel comparison that retired one of our own patches —
cost **five minutes between them** `[measured-here]`.

The harness is in `bench/`; `bench/validate.sh <image-tag>` runs the whole battery.

---

## 5. The gates come before the numbers

Two cheap tests, run **cold** after every boot and again **after** the full benchmark. Nothing else
in this repository means anything without them.

```
python3 scripts/correctness-probe.py http://192.0.2.10:8001
```

```
python3 scripts/code-exam.py http://192.0.2.10:8001
```

Expected: correctness probe **10/10** (client-visible 9/9, empty content 0) and code exam
**12/12 PASS, 0 FAIL**.

**The warm run is the one that carries weight**, and this is not a formality. The class of defect
this stack has actually produced — a kernel writing rows nothing initialises, and a combine summing
them — *hides on a fresh engine*, because a fresh caching allocator hands out zeroed pages. After a
full benchmark the allocator is recycling used memory, so the retired rows would hold real garbage.
Full marks in that state is direct evidence, not a formality `[measured-here]`.

**Two counts are reported on purpose.** With thinking on, this model sometimes puts the whole answer
in the reasoning field and leaves the content field empty — same prompt, same settings, temperature
0. So the probe reports `content` (what a plain client sees) and `both` (what the model actually
knows). A drop in the first with the second intact is a chat-template or parser problem, not a model
problem.

---

## 6. Realistic, synthetic, and fresh

Three kinds of speed number, never mixed in one table.

| Kind | What it is | What it is for |
|---|---|---|
| **Realistic** | `scripts/hizset-v2.jsonl` — 12 short English code prompts, and the four category sets (prose / code / math / JSON) | Everything we publish. This is what a user experiences. |
| **Synthetic** | "count from 1 to 200" and friends | The speculative-decoding **ceiling**. Always labelled. It will disappoint you in real use. |
| **Fresh** | prompts the engine has never seen | Prefill only. See §3. |

The gap is not small. On this model family a synthetic counting prompt runs at roughly 1.7× the
realistic single-stream rate, because the draft model predicts a counting sequence almost perfectly
and predicts real code about half the time. Publishing a synthetic number without the label is the
single easiest way to mislead someone about this hardware.

**Prompt language is a lever too.** On our NVFP4 sibling stack, the same task asked in English ran at
54–63 tok/s with 57–69 % draft acceptance and in another language at 41–47 tok/s with 41–48 %
`[measured-here]`. The engine is identical; the draft model simply predicts English far better. Every
speed table in this repository was measured with English prompts, so treat them as English-workload
numbers.

---

## 7. All benchmarks are at reasoning effort `low`

Temperature 0, `enable_thinking` true, `reasoning_effort: low`, unless a table says otherwise.

We did not run anything at max effort. On this model the low-to-max token ratio is large enough that
a full benchmark suite would take days of cluster time, and we spent that time on the stack instead.
So: **nothing in this repository is a max-effort number, and none of it should be quoted as one.**
What max effort would change is an `[estimate]` wherever we make one, and it is marked.

There is no way to turn thinking off on this model. `enable_thinking: false` does not do what its
name suggests here; the effort levels are the control surface.

---

## 8. What a complete measurement report looks like

Every number we publish carries: the image tag (which kernel commit), TP and EP, quantization, KV
dtype, draft method and `k`, `gpu-memory-utilization`, `--block-size`, `HAREM_SW_BLOCK_SIZE`,
`--max-num-batched-tokens`, `--max-num-seqs`, `NCCL_MAX_NCHANNELS`, temperature, reasoning effort,
`max_tokens`, concurrency, prompt type, the number of sweep rounds and which were discarded, and the
date. If a table does not carry those in its caption or a settings block, it is incomplete.

And one line that is easy to leave out and must not be: **what the gain cost.** Speed, quality and
memory together. If a change genuinely cost nothing, say that it was looked for and not found, and
give the numbers that show it.

---

---

## 9. Three tiers of test, and picking the cheapest one that can answer

A boot on this stack costs about 11 minutes and a full A/B arm about half an hour, so the question
before every measurement is which of these can actually settle it. Working down from the cheapest:

| Tier | What it runs | Cost | Answers |
|---|---|---|---|
| **A — model-free** | `bench/`: the real shapes, engine **down**. `bench/validate.sh <image>`, `bench/ar_bench.py`, `bench/mesh-multilink-sweep.sh`, `bench/moe_stage_bench.py`, `bench/topk_bench.py`, `bench/mhc_bench.py`, `bench/zerokv_bench.py`, and always `bench/bw.py` + `bench/gemmpeak.py` in the same run | seconds to ~45 min, no boot | kernel and collective questions; rejecting a setting outright; anything with a mechanism |
| **B — quick arm** | one boot → gates cold → `soguk-c1` → `prefill-7k` → **`prefill-fresh`** → **1 warm-up + 3 measured** C1–C8 rounds → gates warm → free RAM and swap | ~17–21 min | does a model-free win survive the engine, and did it cost anything |
| **C — full arm** | tier B with **5** measured rounds, plus `category-speed.py` and `mixed-load-probe.py` | ~30–40 min | a configuration that is going into production, or a claim about content types or latency under load |

The tier B row is a measured cost, not an estimate: two arms on 5 September ran 07:38→07:55 and
09:20→09:36, of which the boot is about 5 minutes `[measured-here]`. Its KV pool line comes free —
`GPU KV cache size` is in the boot log — and on any arm that changes a page size or a batch budget
**that line is the result**, not a footnote.

A fourth thing, which is not a tier and is cheaper than all of them: **the live server's own
metrics**. `bench/live-step.py` and `bench/live-decode.py` get a prefill ladder, a chunk-boundary
probe and an exact ms-per-engine-step out of a *running* engine without restarting it, because
`vllm:iteration_tokens_total_count` and `vllm:spec_decode_num_drafts_total` make the step count exact
rather than inferred. When the question is "how long is a step", that is the whole measurement.

**Tier A first, always.** Every conclusion in this repository that survived was reached with the
engine down; every conclusion drawn from two engine sweeps alone has since been retracted (§4). The
protocol setting we rejected without spending a boot, and the per-kernel comparison that retired one
of our own patches, cost five minutes between them.

**Tier B is the workhorse, and it has a gap you must state.** It does not run the category probe or
the mixed-load probe, so an arm measured at tier B has **no** prose/code/math/JSON numbers and **no**
mixed-load numbers. Say `[not tested]` rather than carrying the previous arm's figures forward
silently — the results tables in this repository do exactly that for the last two arms.

**Tier C before anything becomes production**, and before any claim that a change moved the balance
between content types or the behaviour under a long prompt.

Two rules that cut across all three. Every tier ends with the **gates cold and warm** (§5) — a speed
number without them is not a result. And every tier reports **what the gain cost**: speed, quality
and memory together, with the KV pool line filled in, because several changes on this stack have paid
for throughput out of the pool without anyone noticing for a boot or two.

## 10. One measurement at a time: the GPU and the fabric are a lock

This is a protocol rule, not a courtesy. On a three-node cluster with one engine, a model-free
micro-benchmark, a fabric sweep and a serving benchmark all want the same hardware, and any two of
them running at once produce numbers that are quietly wrong: a throwaway container next to the live
engine competes for memory bandwidth and SM occupancy, and a three-node NCCL sweep next to a live
engine competes for RDMA queue-pair resources — which does not slow the engine down, it **fails** its
next collective.

So: **one measurement holds the cluster at a time**, and it says so in a file.

```
echo "what, and when" > ~/exl3-zeus/ENGINE-BUSY
```

The rule around that file is four lines:

1. **Write it before the first `docker run` that touches the GPU or the fabric**, with what you are
   doing and the timestamp. Remove it after the last one — not after the analysis, after the last
   command that could contend.
2. **Check it first, and wait rather than interleave.** If it is held, your measurement is not
   cheaper for starting now; it is worthless.
3. **A model-free container beside an idle engine is allowed and is not free.** Everything measured
   this way in this repository ran in a throwaway container (`--rm`) with a memory cap
   (`--memory=8g`), on a **cpuset disjoint from the engine's**, and with its own JIT cache
   directories so it could not warm or poison the engine's. The engine is pinned to `CPUSET=5-9,15-19`
   ([envs/env.tp3.example](../envs/env.tp3.example)), so the bench gets `--cpuset-cpus 0-4,10-14` and
   the two do not share a core. Write both sets down in the report. This is only defensible while the
   engine is idle, and it is still not free: the GPU and its memory bandwidth are shared no matter how
   the cores are split, which is why §10 exists at all.
4. **Three-node NCCL work needs the engine down, not idle** `[measured-here]`. This is the one hard
   line: a fabric sweep beside a running engine can exhaust queue-pair resources and take the engine's
   next collective with it. Every mesh number on this stack was taken with all three containers
   stopped.

And one habit that has repeatedly paid for itself: **name every command that ever touched the live
engine**, in the report, including the harmless ones. One measurement in this repository read the
engine container's cpuset with `docker inspect` while checking for a CPU collision; that is a
configuration read with no effect on a running container, and it is written down anyway. The value is
not the confession, it is that the next person debugging an odd number can rule the engine in or out
without guessing.

## 11. Changing a patch changes the sidecar: budget the dump boot into the arm

This is a planning rule, and it has cost us an hour of downtime once already.

The fast-load sidecar's identity hash covers **every `patch-*.py` in the patch directory and the
prelude script**, not just the patches that touch a weight ([08](08-fast-boot.md) §4). So any arm
that adds, edits or reorders a patch — including a patch that only writes a log line, only changes a
KV page size, or is gated off by an environment variable it does not set — **invalidates the sidecar,
and the preflight refuses the boot**. That is the design working. It is also a 682-second dump boot
on every node before the arm can start `[measured-here]`.

Three consequences for how an arm is planned:

1. **Read the arm's patch list before you cost it.** A "one boot, ~17 minutes" quick arm that touches
   the patch directory is a **dump boot plus a load boot**, closer to 45 minutes, and the dump boot's
   own numbers are not usable: its KV pool reads low because 56 GiB per node goes out through the
   page cache, and its weights+non-torch ledger is not the production one either. **Never record a
   dump boot's pool as a result** ([08](08-fast-boot.md) §5).
2. **A dump boot is still a real arm for everything that is not memory.** Gates, draft acceptance,
   tok/s and TTFT are all valid on it — the draft-fp8 arm was validated exactly that way
   ([07](07-kv-and-draft-page.md) §7, [11](11-open-issues.md) §2.18) — so if the question is
   "is this safe", the dump boot answers it and the load boot only supplies the pool number.
3. **Do not touch the patch directory while a boot is in progress.** It is mounted live into the
   container, so an edit mid-boot changes the identity underneath the running dump. This has happened
   once and it cost the boot.

The narrower gate this argues for — hashing only the patches that can affect a weight, and hashing
the prelude's ordered list of calls rather than its full text — is written up as an open item in
[11](11-open-issues.md) §2.21. Until it exists, the rule is the one above: budget the dump boot, or
do not touch the patches.

---

## 12. What is next

[10 — Results and roofline](10-results-and-roofline.md) — the numbers this protocol produced, and how
close to the hardware they are. [12 — The tuner cache](12-tuner-cache.md) — why §1 has two protocols
in it.

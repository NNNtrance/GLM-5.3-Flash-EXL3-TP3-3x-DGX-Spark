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

Every arm in this repository up to *fast boot S4* is a five-round median with two discarded; the two
after it are three-round medians. The tables say which, and the two are not interchangeable.

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
| **A — model-free** | `bench/`: the real shapes, engine **down**. `bench/validate.sh <image>`, `bench/ar_bench.py`, `bench/mesh-multilink-sweep.sh`, `bench/moe_stage_bench.py`, `bench/topk_bench.py` | seconds to ~45 min, no boot | kernel and collective questions; rejecting a setting outright; anything with a mechanism |
| **B — quick arm** | one boot → gates cold → `soguk-c1` → `prefill-7k` → **`prefill-fresh`** → **3** C1–C8 rounds → gates warm → free RAM and swap | ~17 min | does a model-free win survive the engine, and did it cost anything |
| **C — full arm** | tier B with **5** rounds, plus `category-speed.py` and `mixed-load-probe.py` | ~30–40 min | a configuration that is going into production, or a claim about content types or latency under load |

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

## 10. What is next

[10 — Results and roofline](10-results-and-roofline.md) — the numbers this protocol produced, and how
close to the hardware they are. [12 — The tuner cache](12-tuner-cache.md) — why §1 has two protocols
in it.

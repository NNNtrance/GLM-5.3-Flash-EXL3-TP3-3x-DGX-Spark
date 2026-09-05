# 07 — The KV pool and the draft page

The KV pool went from 825,000 tokens at TP=2 to **4,484,848** at TP=3, and the last 82 % of that came
from a change that cost no memory at all. This page is why the pool was small, why the obvious
explanation was wrong twice, and what the memory ladder above `gpu-memory-utilization 0.80` actually
looks like on this hardware.

---

## 1. `GPU KV cache size: N tokens` is not a memory figure

This is the sentence everything else on this page depends on. The engine computes:

```
num_tokens      = int(max_concurrency * max_model_len)
max_concurrency = num_blocks / num_blocks_per_request
num_blocks_per_request = SUM over groups of cdiv(group.max_memory_usage_bytes, group.page_size_bytes)
```

Two independent inputs, and they fail in **opposite directions**:

- `num_blocks` is what the free memory buys. A group with a small per-block cost barely moves it.
- `num_blocks_per_request` is what one maximum-length request needs, **summed over groups** because
  block ids are global to a single pool. A group can inflate that without costing memory — and the
  reported pool divides by the sum.

That asymmetry is the whole story. A group can correctly measure "+0.6 % of memory per block" and
still halve the reported pool.

`patches/tp3/patch-kvdiag-tp3.py` is a logging-only patch that prints the per-group decomposition at
every boot, so this number is explained rather than mysterious. Install it and read this at boot:

```
HAREM-TP3 KV pool breakdown: num_blocks=1,756, blocks/request=723, max_model_len=1,000,000
  MLAAttentionSpec:  22 layer(s), page   109,824 B, max/req 33,057,024 B -> 301 block(s)
  KpoolTailSpec:     11 layer(s), page   109,824 B, max/req    109,824 B ->   1 block(s)
  MambaSpec:          9 layer(s), page 1,703,936 B, max/req 15,335,424 B ->   9 block(s)
  MambaSpec:          9 layer(s), page 1,703,936 B, max/req 15,335,424 B ->   9 block(s)
  MambaSpec:          8 layer(s), page 1,703,936 B, max/req 15,335,424 B ->   9 block(s)
  MambaSpec:          8 layer(s), page 1,703,936 B, max/req 15,335,424 B ->   9 block(s)
  SlidingWindowSpec:  5 layer(s), page    24,576 B, max/req  9,461,760 B -> 385 block(s)   <-- the drafter
GPU KV cache size: 2,428,769 tokens, Maximum concurrency for 1,000,000 tokens per request: 2.43x
```

**The drafter takes 385 of the 723 blocks per request — 53 % — while costing 0.6 % of the memory.**

---

## 2. The memory ledger, measured

Device total 121.63 GiB per node, `--gpu-memory-utilization 0.80` → 97.3 GiB budget
`[measured-here]`:

| rank | weights + non-torch | peak activation | CUDA graph pool | **left for KV** |
|---|---|---|---|---|
| 0 (head) | 60.07 GiB | 1.83 GiB | 0.83 GiB | **35.40 GiB** |
| 1 (worker-1) | 62.58 GiB | 1.69 GiB | 0.76 GiB | **33.03 GiB** |
| 2 (worker-2) | 62.77 GiB | 1.69 GiB | 0.74 GiB | **32.85 GiB ← binding** |

The pool is sized by the **smallest** rank, so worker-2's 32.85 GiB decides it and 2.55 GiB on the
head node is simply unused. That asymmetry is a standing item in [11](11-open-issues.md).

Per-block cost, from this model's own layout (the mamba layers co-own the MLA tensors and the
indexer tail rides on the indexer tensor, which is a real memory saving in this image):

| component | layers | page (B) | per block (B) | share |
|---|---|---|---|---|
| MLA (mamba co-owner) | 11 | 1,703,936 | 18,743,296 | 93.4 % |
| indexer k (tail co-owner) | 11 | 109,824 | 1,208,064 | 6.0 % |
| **DFlash2 draft (sliding window)** | 5 | 24,576 | **122,880** | **0.6 %** |
| **total** | | | **20,074,240** | |

`32.85 GiB ÷ 20,074,240 = 1,756` blocks.

### Why the attention page is 3,328 and not the 256 you asked for

`--block-size 256` is passed and the platform raises it:

```
Setting attention block size to 3328 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 5.79% to ensure that mamba page size and attention page size are exactly equal.
```

The mamba state has to fit in one block, and with the indexer at 33 B per token that needs 3,328
tokens. **This cannot be changed and must not be** — raising 3,328 grows the MLA page too, and the
block count falls. `--block-size 256` still matters: it is the prefix-cache hash block, and every KV
group that joins the prefix cache must satisfy `block_size % hash_block_size == 0`, which is asserted
rather than silently ignored.

One consequence to be honest about: with a 3,328-token block, prompts of a few hundred tokens never
fill one, so **prefix-cache hit rate is 0 %** in our benchmarks. That is not a regression (it was
true at TP=2 as well, at block 4,608), but it means the benchmark measures nothing about prefix
caching `[measured-here]`.

---

## 3. Root cause: the drafter's page is 16 tokens

The drafter's group takes 385 blocks per request because:

```python
num_tokens = min(sliding_window - 1 + max_in_flight_tokens, max_model_len)
return cdiv(num_tokens, block_size) + 1
```

with `sliding_window = 2048` (from the drafter's own config) and
`max_in_flight_tokens = max_concurrent_batches × max_num_batched_tokens = 2 × 2048 = 4096`, so
`cdiv(2047 + 4096, 16) + 1 = 385`.

`block_size = 16` is the drafter's backend's **smallest supported kernel block**. The engine's own
comment says the smallest is fine because a later unification step scales it up by an integer ratio
— and that comment is true of upstream. It is not true here, because the DFlash2 port keeps the
draft's KV layers in their **own independent cache group** ([04](04-dflash2-port.md) §3), and that
path never reaches unification. **The 16 survives because of a change we had to make for an unrelated
reason.** That is the root cause `[measured-here]`.

We verified the model of the arithmetic against a second, independent configuration before touching
anything: at `--max-num-batched-tokens 4096`, `max_in_flight` is 8192, the divisor is
`301 + 1 + 36 + 641 = 979`, `num_blocks` was 1,593, and `int(1593 / 979 × 10⁶) = 1,627,170` — exactly
the pool that configuration reported. The model predicts, so the fix can be trusted.

### Choosing the new page size

3,328 = 2⁸ × 13, so the legal multiples of 16 below it are {16, 32, 64, 128, **256**, 208, 416, 832,
1664, 3328}. All of them computed `[estimate]`, then the chosen one measured:

| draft page | blocks/request | per block (B) | num_blocks | **pool (tokens)** | vs today |
|---|---|---|---|---|---|
| **16 (before)** | 723 | 20,074,240 | 1,756 | **2,428,769** | — |
| 32 | 531 | 20,197,120 | 1,745 | 3,286,252 | +35 % |
| 64 | 435 | 20,442,880 | 1,724 | 3,963,218 | +63 % |
| 128 | 387 | 20,934,400 | 1,683 | 4,348,837 | +79 % |
| **256** | **363** | **21,917,440** | **~1,608** | **~4,429,752** | **+82 %** |
| 512 | 351 | 23,883,520 | 1,475 | 4,202,279 | +73 % |
| 3,328 (same as the target) | 341 | 45,510,400 | 774 | 2,269,795 | **−7 % — worse** |

**The optimum is 256, and matching the target's page makes things worse.** At 3,328 the drafter's
per-block cost jumps to 25.6 MB and the block count halves. "Just use the same page size everywhere"
is the naive answer and it is the wrong one.

### The patch

`patches/tp3/patch-swblock-tp3.py` — one file, one anchor, env-gated. With `HAREM_SW_BLOCK_SIZE`
unset it is exactly upstream behaviour; set, it validates the value against the backend's kernel
block and raises if it does not divide. It does not rewrite a spec after the layers were built; it
chooses the value at the point the spec is created.

Set `HAREM_SW_BLOCK_SIZE=256` in `EXTRA_ENV` ([envs/env.tp3.example](../envs/env.tp3.example)).

### Result

Same image, same `gpu-memory-utilization 0.80`, five sweep rounds with the first two discarded
`[measured-here]`:

| | draft page 16 | **draft page 256** |
|---|---|---|
| KV pool | 2,428,769 | **4,413,223** (+82 %) |
| C1 total tok/s | 51.9 | **52.8** |
| C4 total | 107.0 | **117.1** (+9 %) |
| C8 total | 153.8 | **162.8** (+6 %) |
| TTFT C1 / C8 | 0.66 / 1.46 s | **0.47 / 1.14 s** (−20 to −30 %) |
| prefill 7k / fresh | 1,331 / 1,645 | **1,469 / 1,508**; warm repeats 1,699–1,709 |
| acceptance | 61–64 % | 60–64 % |
| gates cold + warm | 10/10 · 12/12 | 10/10 · 12/12 |
| free RAM / swap | 12.6 GB / 0.09 | 11–12 GB / 0 |

**What it cost.** Per-block memory rises 9.2 % (block count 1,756 → ~1,608), and the prefix-cache
matching unit for the draft group coarsens from 16 tokens to 256.

**The speed was a bonus we did not predict.** The draft block table shrinks 16× — 62,500 entries per
request to 3,907 — and that shows up as +9 % at C4, +6 % at C8 and 20–30 % off TTFT. We report it as
a bonus, not as the justification: the justification was the pool.

---

## 4. Two wrong explanations, recorded

**"The TP=2 pool collapsed because of the draft group's page layout."** Partly — but the dominant
cause was memory scarcity at two nodes, and the third node fixed most of it with no page surgery at
all: weights per rank 81.53 → 54.86 GiB, KV memory 17.27 → 39.86 GiB, pool 825,000 → 2,947,441
`[retracted]`. The page layout was the *second-order* term, and it was worth another 82 % once the
first-order term was gone.

**"The draft group costs 0.6 %, so it cannot be the problem."** Both halves of that sentence are
true and the conclusion is false. See §1: memory per block and blocks per request are different
axes, and only the second one moved.

---

## 5. `--max-num-batched-tokens`: the other lever, and its price

Same image, one boot each, gates full marks on both `[measured-here]`:

| | 2048 (production) | 4096 |
|---|---|---|
| prefill-fresh, 3 unseen ~8.4K prompts (median) | 1,498 | **1,640** (+9.5 %) |
| mixed load: decode tok/s while a 7k prompt lands | 8.4 | **10.4** (+24 %) |
| mixed load: 7k TTFT | 5.5 s | **4.8 s** (−13 %) |
| C1 | 50.6 / 51.3 | 51.3 / 52.1 |
| C6 | 119.2 / 118.3 | 122.2 / 122.5 (+3.1 %) |
| C8 | 139.5 / 142.2 | **144.3 / 144.7** (+2.6 %) |
| **KV pool** | **2,427,385** | **1,736,465 (−28.5 %)** |

4096 is a real gain on prefill and on latency under mixed load. It costs **28.5 % of the KV pool**,
because `max_in_flight_tokens` doubles and the draft group's blocks-per-request term doubles with it.

We ran 4096 for one production configuration and then went back to 2048, because the KV pool is the
axis the whole three-node arrangement was about. At `--max-num-seqs 8` either value leaves more than
200K tokens of context per stream with all eight streams full, so nothing we serve is constrained
either way — this is a judgement, not a measurement, and it is reversible in one line.

**3072 was never tried** `[not tested]`. It is the obvious next probe if you want some of the prefill
back without all of the pool.

---

## 6. The memory ladder above 0.80

One rung was climbed, measured, and rejected `[measured-here]`:

| | 0.80 (production) | 0.85 |
|---|---|---|
| KV pool | 4,413,223 | **5,256,198** (+19 %) |
| C1 / C4 / C8 | 52.8 / 117.1 / 162.8 | 53.8 / 112.7 / 161.8 (within spread) |
| prefill-fresh, warm | 1,703 | 1,663 |
| gates | 10/10 · 12/12 | 10/10 · 12/12 |
| free RAM | 11–12 GB, swap 0 | head **1.9 GiB**, worker-1 6, worker-2 8 |
| swap | 0 | head 1.6 GB, worker-2 1.7 GB |

19 % more pool, no measurable speed change, and free memory on the head node below the 4 GiB rule
with real swap in use. **0.85 was rejected and 0.88 was never attempted** `[not tested]`.

The reason is unified memory: on this hardware the KV pool and host RAM come out of the same 121 GiB,
so every rung of the ladder is taken directly from the host. The swap that appeared was the engine's
own process pages — the API process, the worker and the engine core — being paged out, which is
exactly the state you do not want under load.

**Climb it yourself, one rung at a time, and check free memory and swap at each rung.** The ladder is
not forbidden — it is just that on this stack it runs into the free-memory rule before it runs into
anything else. There is one open lead: the fast-load work in [docs/08](08-fast-boot.md) removed a
large page-cache spike during loading, so a rung at 0.82–0.83 may now sit differently from how it sat
before. That has not been measured `[not tested]`.

---

## 7. Where the pool stands

Production, `gpu-memory-utilization 0.80`, draft page 256, `--max-num-batched-tokens 2048`, with the
fast-load sidecar in place `[measured-here]`:

| | tokens |
|---|---|
| KV pool | **4,484,848** |
| maximum concurrency at 1,000,000 tokens per request | **4.48×** |
| for comparison: TP=2 with the same draft | 825,000 |
| for comparison: our NVFP4 stack at `gpu-memory-utilization 0.88` | 4,321,739 |

KV usage never exceeded 13 % across any benchmark we ran, and the queue was always empty. The pool is
not the binding constraint on this stack today — it is insurance for long-context work, which is what
the third node was bought for.

---

## 8. What is next

[08 — Fast boot](08-fast-boot.md), which is where the last +1.6 % of that pool came from, as a side
effect of not reading 163 GB through the page cache.

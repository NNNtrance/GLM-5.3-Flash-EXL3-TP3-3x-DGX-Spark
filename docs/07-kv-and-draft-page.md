# 07 — The KV pool and the draft page

The KV pool went from 825,000 tokens at TP=2 to **4,699,724** at TP=3, and most of that came from two
changes that cost no memory at all — a per-request block counter, and then the drafter's own cache
precision. This page is why the pool was small, why the obvious
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

### 1.1 The other input is a delta, and it drifts with the state of the host

`num_blocks` — "what the free memory buys" — is not a reading of free memory either. It is the
difference between two readings, taken minutes apart, and on this hardware both of them are
`/proc/meminfo`. Three facts from vLLM's own source, checked in the production image `[measured-here]`:

- **On an integrated GPU, "free GPU memory" is `MemAvailable`.** `mem_utils.py` calls
  `torch.accelerator.get_memory_info()` and then, when `torch.cuda.get_device_properties().is_integrated`
  is true — which it is on GB10 — replaces the answer with `psutil.virtual_memory().available`,
  because `cudaMemGetInfo` cannot see reclaimable OS memory. The confirmation is arithmetic: the
  engine reports a device total of **121.63 GiB** and the host's `MemTotal` is 121.6297 GiB, the same
  number; `nvidia-smi --query-gpu=memory.used` returns `[N/A]` on all three nodes.
- **"consumed memory (weights + non-torch)" is not an allocation.** It is
  `free_memory(before load) − free_memory(after the profile run)`. Anything else on the machine that
  frees or consumes memory between those two points lands in it.
- The budget itself is `ceil(total_memory × gpu_memory_utilization)` — a share of the **total**, not
  of what is free. So the pool is
  `0.80 × MemTotal − (MemAvailable(init) − MemAvailable(after profile)) − peak activation`.

Read that formula once more: **the lower `MemAvailable` is when the engine starts, the larger the pool
it computes.** The instrument runs backwards. A node that starts dirty awards itself memory it does
not have.

That is not hypothetical. The launcher kills the previous container (~90 GiB) and starts the new one
immediately, and the nodes are started worker-2 → worker-1 → head, so the **head** — started last — is
the node the kernel has had least time to reclaim for. On one dump boot the three ranks began with
104.10 / 113.07 / 113.10 GiB available, a **9.00 GiB** spread, and ended the profile at
47.74 / 48.73 / 48.52 GiB, a **0.99 GiB** spread `[measured-here]`. A spread that disappears by the
end was never an allocation.

**The cure is on the host, before the container.** `scripts/start-tp3.sh` waits after `docker rm -f`
until `MemAvailable` is back over `SETTLE_MIN_GIB` (112 by default), polling every 3 s for at most
180 s, and logs what it waited for:

```
sync
SETTLE_MIN_GIB="${SETTLE_MIN_GIB:-112}"
for settle_i in $(seq 1 60); do
  settle_avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
  [ "$settle_avail" -ge "$SETTLE_MIN_GIB" ] && break
  sleep 3
done
```

It cannot be done from inside the container: `/proc/sys` is in the container's `ReadonlyPaths` and the
container is not privileged, so a prelude cannot even drop caches. **The gate buys zero tokens.** What
it buys is a ruler: per-rank startup free memory went from a **9 GiB** spread to **1.4 GiB**
(111.65 / 113.06 / 113.07) `[measured-here]`.

**How much of the pool was ever at risk — stated carefully, because it is easy to overstate.** No
published pool figure here is known to be wrong. The pool takes the **minimum** over ranks, and on
every boot we have a ledger for, the node with the polluted baseline was the **head**, which was not
the binding rank. That is luck, not design: 9 GiB is **27 %** of a rank's KV allowance, and the polluted
node is simply whichever one starts last. Change the start order, add a node, or let a worker be the
slow one, and the artefact walks straight into the pool. Meanwhile the pool figures across recent boots
span 4,231,404 → 4,484,848 (**6.0 %**); each of those has a candidate explanation in
[08](08-fast-boot.md) §5, and comparable load boots after that fix agree to 0.4 % — but **with the
baseline unpinned there was no way to tell an explanation from an artefact**, which is the actual cost
and the reason the gate was worth writing. A pool difference of a few percent is exactly the size this
stack argues about ([09](09-measurement-protocol.md) §1.2).

**Two rules follow, and they are the operative half of this section:**

1. **Read the pool only from a load boot.** A dump boot writes 56 GiB per node through the page cache
   and its dirty pages are still in flight during the profile, so they are billed to
   `total_consumed`: the same configuration reads about **6.7 % low** on a dump boot
   ([09](09-measurement-protocol.md) §11).
2. **Check the gate did its job before quoting the number.** All three ranks' `Free memory on device`
   *and* `Actual usage ... consumed memory` lines must agree within 1 GiB. If they do not, the boot
   produced a speed result and a quality result, and no pool result.

The acceptance check that follows from rule 2 is one grep:

```
docker logs exl3-tp3 2>&1 | grep 'gpu_worker.py' | grep 'Free memory on device'
```

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

> **At TP=2 this fix has never been run** `[not tested]`. The 825,000-token figure in the opening
> paragraph is the un-fixed two-node pool, and it fell there for exactly the mechanism §3 describes —
> turning DFlash2 on at two ranks took the pool from 1,987,179 to 825,000, **−58 %**. Setting
> `HAREM_SW_BLOCK_SIZE=256` at two ranks should recover most of it, but nobody has measured that and
> the hybrid allocator can override the page entirely. [15](15-tp2-track.md) §3.1 and §3.3.

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
anything else.

**The 0.85 rejection is older than the machine it was measured on.** That boot predates the fast-load
work, which removed a large page-cache spike during loading and added `MADV_DONTNEED` plus
`malloc_trim` at the end of it ([08](08-fast-boot.md) §5). Since then the same production
configuration sits at 11–12 GiB free with **zero** swap and 3.0–3.5 GiB of page cache, where the
0.85 boot had hit 1.9 GiB free and 1.6 GB of swap. The rung at **0.82–0.83** was never tried at all,
and 0.85 deserves a re-run rather than a carried-forward verdict `[not tested]`.

There is a second, sharper instrument nobody here has used: **`--kv-cache-memory`**, which sizes the
pool in bytes instead of as a fraction of the device. It skips the memory profile altogether — vLLM
takes the byte figure as given — so it is immune both to the rank spread and to the baseline drift of
§1.1, and all three ranks get the same number. `gpu-memory-utilization` budgets a share of the
**total**, so a worker that starts with 113 GiB free is still squeezed into a 97.3 GiB budget, and
vLLM's own boot log says as much — it prints the byte figure it thinks would fully utilise the device
against the far smaller one we actually take. Ladder first and pin last, because a pin removes the
headroom that the free-memory rule is protecting `[not tested]`.

**Do not take the "to fully utilize gpu memory" hint.** That figure is
`MemAvailable(init) − non_kv_cache_memory − 150 MiB`, i.e. *all* of the memory the host had free at
boot: on a unified-memory part it leaves the machine zero page cache, which is the 0.85 boot's swap
table again by a different route `[measured-here]`.

**The rung to try next is 0.83, and the reason it is worth trying now is §1.1.** The knob multiplies
`MemTotal`, so it is the one input to the pool that the baseline cannot corrupt: +0.03 × 121.63 GiB is
**+3.65 GiB** of budget, which lands whole on the binding rank (~33.3 GiB at today's pool and
141,247 tokens per GiB) and is worth about **+11 %**, roughly **5.2M tokens** `[estimate]`. Run before
the settle gate existed, an 11 % signal would have shared a table with up to 27 % of baseline artefact
per rank, and no way to separate them. The
price, stated rather than left out: the OS share drops from 20 % to 17 %, and the narrowest node — the
head, which also carries the API server — goes from ~12.3 GB free under load to about **8.4 GB**. That
is still clear of the 4 GiB rule, and 0.85 (another +6.08 GiB, leaving ~5.8 GB) is the rung after it,
not instead of it, and only after a soak. **Not run: it is a production memory change and waits on the
stack owner's approval** `[not tested]`.

---

## 7. The draft cache at fp8 — in production

The main cache is `fp8`; the drafter's is `bf16`, because our launcher pins it there —
`"kv_cache_dtype": "auto"` inside `--speculative-config`, where `auto` for the draft means "inherit"
only if the field is left unset. A prelude patch (`HAREM_DRAFT_KV_DTYPE`) overrides
`SpeculativeConfig` before any validation runs, so the launcher is untouched and the rollback is one
environment variable. With the knob unset the patch does not run at all.

**What it is worth, and why an earlier number was wrong.** The drafter's page halves, so the
per-block cost falls and the blocks-per-request divisor — the thing that actually decides the pool
(§1) — does not move:

| geometry | bytes per block | divisor | pool |
|---|---|---|---|
| draft page 16 (measured before the page surgery) | 20,074,240 → 20,012,800 | 723, unchanged | **+0.3 %** |
| **draft page 256 (production today)** | 21,917,440 → **20,934,400** (−4.5 %) | 363, unchanged | **+4.7 %** `[estimate]` |

The **+0.3 %** figure that appeared in our earlier notes belongs to the pre-256 geometry and does not
apply to this stack `[retracted]`. The draft group is a larger share of the per-block cost now
precisely because §3 made its page 16× bigger.

**Measured: it is safe.** The arm booted, which answers the only question a boot could answer — the
DFlash sliding-window backend does accept an fp8 cache, and would have failed loudly rather than
silently if it did not. The engine's log carries the mechanism directly: draft page
393,216 → **196,608 bytes**, per-block 21,917,440 → **20,934,400**, divisor still 363. Gates
**10/10 · 12/12** cold and warm. **Draft acceptance 60.1–64.0 %** across all concurrency levels and
rounds, with one C1 round at 57.3 % — against production's 61–65 %, and against a gate that said the
arm dies if acceptance leaves the 60–65 band. Speed is inside the bands ([09](09-measurement-protocol.md)
§1.2). Full table in [10](10-results-and-roofline.md) §2.1 `[measured-here]`.

**The pool number arrived on the load boot, and it promoted the change.** The validation arm had run
on a **dump boot** — it added three prelude patches, which invalidates the fast-load sidecar
([09](09-measurement-protocol.md) §11) — and a dump boot's pool reads low, this one 4,382,920, *below*
production and meaningless. The ordinary load boot that followed, with the settle gate of §1.1 in
place, read `[measured-here]`:

| | production 6 (bf16 draft cache) | **production 7 (fp8 draft cache)** |
|---|---|---|
| KV pool | 4,449,035 | **4,699,724** (+5.6 %) |
| maximum concurrency at 1M tokens per request | 4.45× | **4.70×** |
| per-rank startup free memory, spread | 9 GiB (baseline artefact) | **1.4 GiB** (settle gate) |

**+5.6 % against the +4.7 % predicted** — the estimate was good to within a point, and the residual is
the settle gate removing part of the drift the old number carried. This is now the production
configuration; the rollback is still one environment variable
([11](11-open-issues.md) §2.18 records it closed).

---

## 8. Every newly allocated block is zeroed, and here that cannot be switched off

Reading the per-block table in §2 raises an obvious question: 21.9 MB per block is a lot of bytes to
hand to a request, and vLLM zeroes each one before use. Per prefill chunk that is a single kernel
writing **2.4–2.9 GB**, where the new tokens' real KV is about 3.4 MB — 1.3 % of a prefill chunk, at
100 % of the memset roofline, so there is nothing to win inside the kernel and the only lever was to
not run it ([10](10-results-and-roofline.md) §5.6).

vLLM zeroes when the cache has Mamba layers **or** mixed precision. The zeroer visibly skips Mamba
layers, which made the inference "so on this stack the only live reason is the bf16 draft cache; move
it to fp8 and the zeroing can stop" look sound. It is wrong, and the reason is the same slot sharing
that makes §2's table look the way it does: in this model's hybrid layout **one tensor is co-owned by
an MLA layer and one Mamba layer from each group**, so a block released by the Mamba group and handed
to the attention group arrives carrying 1.7 MB of raw SSM state. Measured per block, **85.5 %** of
what is being zeroed is that co-owned region; the indexer tail is 5.5 % and the draft's sliding
window 9.0 % `[measured-here]`.

So the Mamba half of the condition is the binding one, it is independent of precision, and **draft
fp8 does not open this door**. The two changes are unrelated after all — which is worth stating
plainly, because they were designed as one change. A fail-closed gate was written anyway
(`HAREM_ZERO_ATTENTION_KV=0`, off by default, three conditions proved from the engine's own
configuration, `raise` rather than warn) so that the conclusion is checked by the machine instead of
believed: on this model it refuses to boot, by design. The safe remainder — indexer plus draft, if the
cache were uniform — is 0.19 % of prefill and no partial mode was written for it
([11](11-open-issues.md) §2.13).

---

## 9. Where the pool stands

Production, `gpu-memory-utilization 0.80`, draft page 256, **draft cache fp8**,
`--max-num-batched-tokens 2048`, with the fast-load sidecar in place and the settle gate of §1.1 on a
load boot `[measured-here]`:

| | tokens |
|---|---|
| KV pool | **4,699,724** |
| maximum concurrency at 1,000,000 tokens per request | **4.70×** |
| the same stack with a bf16 draft cache (production 6) | 4,449,035 |
| for comparison: TP=2 with the same draft | 825,000 |
| for comparison: our NVFP4 stack at `gpu-memory-utilization 0.88` | 4,321,739 |

KV usage never exceeded 13 % across any benchmark we ran, and the queue was always empty. The pool is
not the binding constraint on this stack today — it is insurance for long-context work, which is what
the third node was bought for.

---

## 10. What is next

[08 — Fast boot](08-fast-boot.md), which is where the last +1.6 % of that pool came from, as a side
effect of not reading 163 GB through the page cache.

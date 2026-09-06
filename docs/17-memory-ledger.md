# 17 — The memory ledger: where 121.6 GiB per node actually goes

**Applies to: both tracks.** The ledger itself is the three-node production configuration; §6 is the
two-node column beside it, and the KDA state finding is worse at two ranks than at three.

Every number below is read from a boot log, `/proc/meminfo`, `nvidia-smi` or the image's own source.
Nothing here was measured by starting or stopping anything: the engine was up and serving throughout,
no container was `exec`'d into, no configuration file was changed and no lock was taken. 6 September
2026 `[measured-here]`.

**Three lines, before the tables.**

- **Is it filling up? No — it is *kept* full by design.** With the engine up there is 5.1 GiB free per
  node and 9.4–10.7 GiB reclaimable, and swap is flat at 30–40 MB. There is no place that is supposed
  to stay empty: every GiB left free is a GiB that did not go into the KV pool.
- **How much is really unavailable? 4.84 GiB.** With the engine down a node offers 116.79 GiB of
  121.63. The old "the driver takes 14.2 GiB" figure is **stale by about 10 GiB** — §2.4.
- **What fills it:** weights 51.62, KV pool 40.12, the engine's non-torch working set 7.28, the host
  side about 13, and a fixed driver reserve of ~3.5. **Inside the KV pool the largest single item is
  the MLA latent at 89.5 %, and the largest *avoidable* one is the KDA state's speculation slots** —
  36 blocks per request, 9.9 % of the divisor that sets the pool size.

---

## 1. The two boots this page is written from

| | |
|---|---|
| **The ledger boot** | 6 September, TP=3 production 10, image `exl3-zeus:754421f`, `gpu-memory-utilization` 0.83, fast-load **load** boot. Available KV **40.12 GiB**, block cost **20,934,400 B**, **2,058** blocks, pool **5,669,421** tokens — all four printed in that boot's own log, which is why the anatomy in §3 comes from here |
| **The production-10 reference boot** | The configuration's headline boot, same image and same fraction, pool **5,619,834** tokens. Its per-group KV dump was **not kept** — the arm script grepped the pool figure out and discarded the container log — so §5's model anchors on the reported pool and closes the rest by arithmetic |

**The two boots do not have the same block arithmetic, and we cannot say which of them the other one
ran.** The ledger boot logs 40.12 GiB over 20,934,400 B per block; the model that reproduces the
production-10 pool uses 42.89 GiB over 22,572,800 B. Each is internally exact — each reproduces its
own boot's reported pool to the token — and neither can be checked against the other because the
production-10 log is gone. **Boot-to-boot pool spread on this stack is real** (the five arms of the
6 September kernel campaign read 5,666,666 to 5,696,969 at the same setting), but that is a smaller
effect than a different per-block cost, so this is recorded as **unreconciled** rather than explained.
The percentages in §5 are ratios taken inside one arithmetic and do not depend on which base is right.

**The fix, and it is ours to make:** the three-node arm scripts must keep `docker logs` as `kv.txt`
the way the two-node harness already does. A pool figure without its per-group breakdown cannot be
audited later.

---

## 2. The per-node ledger, TP=3 production 10

Rank 0, engine up and idle. Sums to `MemTotal` = 121.63 GiB (127,535,272 kB).

| Line | GiB | Source |
|---|---:|---|
| **Engine CUDA allocation (unified memory)** | **99.06** | `nvidia-smi --query-compute-apps` → 101,434 MiB |
| — model weights (target ≈49.6 + DFlash2 draft ≈2.0) | 51.62 | `[model_runner.py:374] Model loading took 51.62 GiB` |
| — KV pool (2,058 blocks × 20,934,400 B) | 40.12 | `[gpu_worker.py:564] Available KV cache memory: 40.12 GiB` (binding rank) |
| — non-torch: CUDA context, NCCL and mesh plugin buffers, cuBLAS and FlashInfer workspaces, EXL3 tune buffers | 7.28 | 58.9 consumed − 51.62 weights |
| — CUDA graph pool | **0.00** | `... and 0.0 GiB for CUDAGraph memory` |
| — residual inside the CUDA figure | +0.04 | the four rows above against 99.06 |
| **Host anonymous memory** (API server + EngineCore + worker) | 5.60 | `AnonPages` |
| **Page cache** (container image and libraries; the checkpoint's pages are already dropped) | 5.80 | `Cached`, including `Shmem` 0.73 |
| Buffers | 0.07 | `Buffers` |
| **Kernel slab** (`SUnreclaim` 2.10 is driver bookkeeping; 0.43 with the engine down) | 2.36 | `Slab` |
| Kernel miscellany (`PageTables` 0.03 + `KernelStack` 0.01 + `Vmalloc` 0.10 + `Percpu` 0.02) | 0.17 | `/proc/meminfo` |
| **Free** | 5.08 | `MemFree` (`MemAvailable` 9.43) |
| **Unexplained residual = fixed driver reservation** | **3.49** | 121.63 − everything above |
| Swap used | 0.03 | 16 GiB of swap, flat |
| **TOTAL** | **121.63** | |

### 2.1 How the engine divides the budget

```text
requested_memory = ceil(MemTotal × gpu_memory_utilization) = 121.63 × 0.83 = 100.95 GiB
```

**The fraction is a fraction of the total, not of what is free** (`vllm/v1/worker/utils.py:509-511`).
That is the single most misread line in the memory story and it is why a node with less free memory
does not automatically get a smaller pool — it gets a boot failure instead.

| Rank | Free at init (GiB) | consumed (weights + non-torch) | peak activation | CUDA graph | **Available KV** | KV allocated | wasted |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 112.01 | 58.90 | 1.66 | 0.00 | 40.39 | 40.12 | 0.27 |
| 1 | 112.97 | 58.94 | 1.66 | 0.00 | 40.35 | 40.12 | 0.23 |
| 2 | 113.15 | 59.17 | 1.66 | 0.00 | **40.12 ← binding** | 40.12 | 0.00 |

The log line, identical in shape on all three nodes:

```text
[gpu_worker.py:790] Free memory on device (112.01/121.63 GiB) on startup. Desired GPU memory
  utilization is (0.83, 100.95 GiB). Actual usage is 58.9 GiB for consumed memory
  (weights + non-torch), 1.66 GiB for peak activation, and 0.0 GiB for CUDAGraph memory.
  ... Current kv cache memory in use is 40.39 GiB.
```

**The settle gate's own examination, and it did not quite pass.** The launcher's F1 gate wants the
three ranks' `Free memory on device` within 1 GiB of each other. On this boot the spread was
**1.14 GiB** — narrowly failed — while the `consumed` spread was 0.27 GiB and passed. The KV wasted
to rank imbalance is now **0.50 GiB**, against 2.55 GiB before the gate existed
([11](11-open-issues.md) §2.3).

### 2.2 "Peak activation" is a reservation, not an allocation

1.66 GiB is the difference between torch's peak and its current allocation during the profile run
(`transient_peak_headroom`). It is **subtracted from the budget but does not sit in memory**, and it
is measured at `--max-num-batched-tokens 2048`. This is why §7's `MNBT 1024` row is a real, if small,
lever: it moves a reservation, not a working set.

### 2.3 The CUDA-graph pool is exactly zero here, and that deletes a lever

```text
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend
FlashInferBackend ...; setting cudagraph_mode=NONE
```

On the NVFP4 sibling stack the CUDA-graph estimator was a **3.9 GiB** item and turning it off was a
real give-back. **On this stack it charges nothing**, so that lever does not exist here — do not carry
it over ([10](10-results-and-roofline.md) §5.8).

### 2.4 The "driver takes 14.2 GiB" figure is stale by about 10 GiB

Measured two ways on the same day:

| | |
|---|---|
| Engine **down**, clean node | `MemFree` 111.13 · **`MemAvailable` 116.79** · `Cached` 6.41 · `AnonPages` 0.11 · `Slab` 0.67 · `nvidia-smi` compute-apps **empty**. Residual **3.12 GiB** |
| Engine **up** | Residual **3.49 GiB** (the table in §2) |

Two independent states, the same 3.1–3.5 GiB. **That is the driver's fixed reservation**, and the
total genuinely unavailable at idle is **4.84 GiB**. The 14.2 GiB figure came from an NVFP4-era
ceiling scan whose highest observed free was 107.43 of 121.63; today the three ranks start at
**112.01 / 112.97 / 113.15** and an idle node offers 116.79.

**The practical consequence is a change of ruler, not a lever.** What sets the ceiling on
`gpu-memory-utilization` is no longer the driver but **the host's share at run time** — worker RSS
4.71 GiB per rank, API server 1.81, EngineCore 1.56, plus page cache. `Mlocked` is only 24 MiB, so
there is no large pinned buffer hiding anywhere.

---

## 3. Inside the KV pool: what 40.12 GiB is made of

The block cost comes from the model's own layout: **the 34 KDA layers do not own KV memory — they
co-own the MLA layers' tensors**, the k-pool tail rides on the indexer tensor, and the drafter gets
its own.

### 3.1 Per-block cost, block = 3,328 tokens

| Component | Layers | Page (B) | Per block (B) | Share |
|---|---:|---:|---:|---:|
| MLA latent, fp8 (co-owned by the 34 KDA state layers) | 11 | 1,703,936 | 18,743,296 | **89.53 %** |
| Indexer k, fp8 (co-owned by the k-pool tail) | 11 | 109,824 | 1,208,064 | 5.77 % |
| DFlash2 draft, sliding window, fp8, page 256 | 5 | 196,608 | 983,040 | 4.70 % |
| **Total** | | | **20,934,400** | 100 % |

40.12 GiB ÷ 20,934,400 = **2,058**, which is the `num_blocks` the log prints.

The page sizes check against `config.json` rather than being taken on trust: `kv_lora_rank` 512 with
`mla_use_nope` true gives MLA **512 B per token** at fp8, and `index_head_dim` 128 with `index_kpool`
4 gives 32 dimensions plus one scale byte = **33 B per token**. 3,328 × 512 = 1,703,936 and
3,328 × 33 = 109,824.

### 3.2 The divisor: blocks per 1M-token request

**This is the number that sets the pool**, because `pool = 1,000,000 × num_blocks ÷ blocks_per_request`
— the same counter mechanism as the draft page in [07](07-kv-and-draft-page.md) §3.

| Group | Blocks / request | Share | GiB / request |
|---|---:|---:|---:|
| MLA + indexer | 301 = `cdiv(1,000,000 / 3,328)` | 82.9 % | 5.868 |
| k-pool tail | 1 | 0.3 % | 0.020 |
| **KDA (Mamba) state — 4 groups × 9 blocks** | **36** | **9.9 %** | **0.702** |
| DFlash2 draft | 25 | 6.9 % | 0.487 |
| **Total** | **363** | 100 % | **7.078** |

1,000,000 × 2,058 ÷ 363 = **5,669,421** tokens; the pool holds **5.67** maximum-length requests.

**In plain terms:** one single 1M-token request occupies **7.08 GiB** of every node. The pool holds
5.67 of them. With `--max-num-seqs 8`, eight 1M-token requests would need 56.6 GiB — they would not
fit, and three would wait. Real prompts are far shorter, so **at three nodes the pool is not the
binding constraint today**. At two nodes it was ([15](15-tp2-track.md) §4).

---

## 4. The KDA state slots: the largest avoidable line in the ledger

This gets its own section because it is the biggest give-back left, and because it is a **counter**
rather than memory — the same class of defect as the draft page.

### 4.1 Nine slots, and where the nine comes from

`vllm/v1/kv_cache_interface.py:812-821`:

```python
elif vllm_config.cache_config.mamba_cache_mode == "align":
    return self.page_size_bytes * (2 + self.num_speculative_blocks)
else:
    return self.page_size_bytes * (1 + self.num_speculative_blocks)
```

Prefix caching is on, so `mamba_cache_mode` defaults to **align**, and `num_speculative_blocks` is
the drafter's `num_speculative_tokens` = **7**. So **2 + 7 = 9 state slots per KDA layer per
request** — a base pair, and seven that exist only because DFlash2 is on so a rejected speculative
step can be rolled back. The engine prints it:

```text
MambaSpec: 9 layer(s), page 1,703,936 B, max/req 15,335,424 B -> 9 block(s)
```

and 15,335,424 ÷ 1,703,936 = 9.

> **A correction to our own working note.** The same 9 can be read as `1 + 8` off the *other* branch
> of that `if`, and our first pass did read it that way, which puts eight slots rather than seven on
> the drafter's account. **Align is the branch we run**, so it is 2 + 7. It changes no total on this
> page — nine slots either way — but it is why the compact-rollback alternative in §5 goes to **two**
> slots and not to one.

### 4.2 The measured shape

| KDA state | TP=3 (production 10) | TP=2 (`kvfix` arm) |
|---|---:|---:|
| KDA layers in the model | 34 of 45 | 34 of 45 |
| Co-owned MLA tensor slots | 11 | 11 |
| Mamba cache groups (layers) | 4 (9 / 9 / 8 / 8) | 4 (9 / 9 / 8 / 8) |
| State slots per layer per request | **9** | **9** |
| Page per layer per slot, padded (= the attention page) | 1,703,936 B | 2,359,296 B |
| Page per layer per slot, real, before padding | 1,610,752 B | 2,342,912 B |
| Padding the engine logs | **5.79 %** | **0.70 %** |
| Layer-slots per request (9·9 + 9·9 + 9·8 + 9·8) | 306 | 306 |
| **Blocks per request** | **36** (9.9 % of 363) | **36** (12.9 % of 280) |
| Real state bytes per request | 0.459 GiB | 0.668 GiB |
| Allocated bytes per request, including padding and unused co-tenant regions | **0.702 GiB** | **1.014 GiB** |
| At `--max-num-seqs 8`, full concurrency | 288 blocks = **5.61 GiB**, 14.0 % of the pool | 288 blocks = **8.11 GiB**, of a 10.31 GiB pool |

The real page is derived from the model rather than guessed: `kda_state_shape` gives a recurrent state
of `(heads/tp, head_dim, head_dim)` in fp32 and a conv state of width `conv_kernel_size − 1 + num_spec`
in bf16, so at TP=3 with 22 local heads that is 22 × 128 × 128 × 4 = 1,441,792 plus 8,448 × 10 × 2 =
168,960 → **1,610,752 B**. **The conv state widens with the drafter's `k`**, which is a second, smaller
way speculation shows up in this line.

### 4.3 Why the attention block is 3,328 when we asked for 256

Because a KDA state slot has to fit in **one** block:

```text
[interface.py:926] Setting attention block size to 3328 tokens to ensure that attention page size
                   is >= mamba page size.
[interface.py:950] Padding mamba page size by 5.79% to ensure that mamba page size and attention
                   page size are exactly equal.
```

`platforms/interface.py` computes
`attn_block_size = alignment × cdiv(mamba_page_size, alignment × attn_page_size_1_token)` with
alignment 256 and one token of MLA at 512 B: at TP=3 that is 256 × ⌈1,610,752 / 131,072⌉ = 256 × 13 =
**3,328**, and at TP=2 256 × 18 = **4,608**. **`--block-size 256` is overridden on this path** — the
value we pass survives only as the alignment base. (The `cuda-exl3` author's TP=4 box gets 1,664, because
the state page shrinks as ranks are added.)

**`HAREM_SW_BLOCK_SIZE=256` does not touch this.** That patch bypasses
`_largest_kernel_block_within` for the DFlash2 drafter's independently-grouped `SlidingWindowSpec`
only, taking its page from 16 to 256 tokens ([07](07-kv-and-draft-page.md) §3). The KDA path is
outside its scope and the two jobs are independent.

**And the distinction that decides what a fix would be worth: the slot count does not enlarge the
page, it multiplies the block count.** One page is one slot; nine slots are nine blocks. So a
compact-rollback scheme does **not** shrink the 3,328-token block — it takes the mamba blocks per
request from 36 to a handful.

---

## 5. What could give memory back, and what each would cost

Everything in this section is at the **same memory fraction**. `pool = 1e6 × num_blocks ÷ blocks_per_request`.

### 5.1 The arithmetic, on the ledger boot's base

| Scenario | per-block (B) | num_blocks | blocks/req | **Pool (tokens)** | vs today |
|---|---:|---:|---:|---:|---:|
| **Today (ledger boot)** | 20,934,400 | 2,058 | 363 | **5,669,421** | — |
| KDA state slots 9 → 1 (compact rollback) | 20,934,400 | 2,058 | 331 | **6,217,522** | **+9.7 %** |
| Attention block 3,328 → 1,664 (needs a half-size state page) | 10,958,720 | 3,931 | 663 | 5,928,506 | +4.6 % |
| Both of the above | 10,958,720 | 3,931 | 631 | 6,229,794 | +9.9 % |
| Draft page 256 → 384 | 21,425,920 | 2,010 | 355 | 5,661,972 | **−0.1 %** |
| Draft page 256 → 512 | 21,917,440 | 1,965 | 351 | 5,598,290 | **−1.3 %** |
| No draft model at all (reference only) | 19,951,360 | 2,159 | 338 | 6,387,573 | +12.7 % |
| `gpu-memory-utilization` 0.83 → 0.85 | 20,934,400 | 2,182 | 363 | 6,011,019 | +6.0 % |

**Two things close here, by arithmetic, without an engine.** The **draft page of 256 is the optimum** —
384 is flat and 512 loses — so that question is finished. And **shrinking the attention block adds
almost nothing on top of a slot fix**: both dilute the same 62 fixed blocks, so the KDA state is the
one real target rather than one of two.

### 5.2 The ranked give-back list

| # | Candidate | Give-back | Evidence | Risk |
|---|---|---|---|---|
| 1 | **KDA state slots, 9 → a compact ring** | **+8.0 to +9.7 % of pool at TP=3, +9.6 to +12.9 % at TP=2**, depending on whether the ring is charged for; frees 4.99 GiB of live pool at 8-way concurrency | §4, §5.3 | **High — state-rollback correctness.** A wrong rollback is fluent and silent, the same hazard family as the dual-batch KDA closure ([11](11-open-issues.md) §2.24). Upstream vLLM has ReplaySSM and **refuses it when drafting** |
| 2 | **`gpu-memory-utilization` 0.83 → 0.85** | +6.0 % of pool (+2.43 GiB of budget) | deterministic, §5.1 | Measured once and rejected: 0.85 put one node into 1.6 GB of swap, which is why production 10 is 0.83 ([11](11-open-issues.md) §2.4) |
| 3 | **Attention block 3,328 → ~1,664** | +4.6 % alone, **+0.2 % on top of #1** | §5.1; the pool-formula optimum is ≈1,626 and the constraint is `attention page ≥ mamba page` | Medium — it means changing the state's dtype, i.e. correctness. Redundant if #1 happens |
| 4 | **Drop the page cache of the shards a dump boot *writes*** | ≈ +2.3 GiB, **on dump boots only** — it does not touch a production load boot | the fast-load loader currently drops only the shards it *reads*; dirty writeback lands in `total_consumed` and is why a dump boot reads ~6.7 % low ([09](09-measurement-protocol.md) §11) | Low |
| 5 | **`--max-num-batched-tokens` 2048 → 1024** | ~+0.6 GiB of budget → **+1.6 % of pool** `[estimate]` | peak activation is 1.66 GiB and scales with this; the TP=2 arm's larger value gives 2.42 | Medium — it has a prefill throughput cost, and 4096 was already given back for the pool ([11](11-open-issues.md) §2.5) |
| 6 | **Tighten the settle gate** (`sync; drop_caches` on the host before it runs) | +0.27 GiB on the binding rank ≈ **+0.7 % of pool**, and it would close the ~6 % boot-to-boot pool noise that makes every ladder measurement ambiguous | this boot's spread was 1.14 GiB against a ≤1 GiB gate; wasted KV 0.50 GiB | Low, with precedent — but on one node first ([11](11-open-issues.md) §4) |
| 7 | Draft page 256 → 384 / 512 | **−0.1 % / −1.3 % — it loses** | §5.1 | **Closed.** 256 is the optimum |
| 8 | CUDA-graph estimator off | **0.00 GiB** | `cudagraph_mode=NONE` on this stack; the log charges 0.0 GiB | **Closed.** The NVFP4 sibling's 3.9 GiB item does not exist here |
| 9 | The unused MTP layer in the checkpoint (`mtp.safetensors`, 3.79 GB) | **0.00 GiB of memory** — disk only | it is not in the safetensors index, so vLLM never reads it ([01](01-model-and-license.md) §4) | **Closed** |
| 10 | Replicated tensors duplicated across ranks | ≈ 0 | everything divides by TP, the drafter included (32/8 → 36/9 padded, rank 2 owning 8 real and 4 padded query heads), and under EP the experts are not sliced at all (96 of 288 per rank) | **Closed** |
| 11 | Page cache still holding the checkpoint after load | **0.00 GiB** | the fast-load path already does it: `page cache dropped for 21 checkpoint shards`, `malloc_trim RSS 2.85 → 2.81 GiB` | **Closed** |
| 12 | tmpfs / `/dev/shm` | **0.00 GiB** | 61 GiB of tmpfs with **67 MiB** used; `Shmem` 0.73 GiB is container IPC | **Closed** |
| 13 | The "14.2 GiB driver" | **about 10 GiB of it was never there** | §2.4 | Not a lever — a **ruler correction**. The ceiling is now the host share, not the driver |
| 14 | Host RSS: worker 4.71 GiB/rank + API server 1.81 + EngineCore 1.56 | not measured | `ps -o rss`; `docker stats` reads 7.75 / 6.54 / 6.52 GiB per node | **Open.** RSS at profile time affects the pool; RSS afterwards does not |

### 5.3 ReplaySSM: what a compact rollback would actually return, and why it is not here

A third-party vLLM patch (`vllm-replayssm-spec.patch`, by `tpurtell`, Apache-2.0) lifts upstream's own
refusal —

```python
if self.num_speculative_tokens > 0:
    raise ValueError("--use-replayssm does not support speculative decoding")
```

— and replaces the slot array with a checkpoint plus a ring of processed inputs, taking the align
path's slots from **9 to 2**. Stock vLLM already ships the ReplaySSM kernels and the
`SupportsReplaySSM` protocol; on the GLM-5-Next KDA path there is not one line that uses them.

**It applies to us cleanly.** 21 files, 127 hunks, +3,183/−103, **pure Python and Triton with zero
CUDA or `csrc`**, so there is nothing to compile for sm_121; `patch -p1 --dry-run` gives **127/127
hunks OK, 0 failed**. A GLM port script matches **9 of its 11 anchors**, the two misses being one
class-base-list line that our fork formats differently. Grepping the 230 KB patch for
`exl3|b12x|trellis|marlin|gemm|\.cu|csrc` returns **zero hits**. The only file it touches that one of
our own patches also touches is `models/glm5next/nvidia/model.py`, in a different region, so the
ordering matters and nothing else conflicts. Our TP=3 head padding is a **precondition** it needs
rather than an obstacle `[measured-here]`, all of it from a read-only review in a throwaway CPU
container.

**What it would return**, projected by a model that reproduces **four** independent measured arms to
the token — TP=3 before the draft-page fix at 0.80 (2,428,769), TP=3 production 10 at 0.83
(5,619,834), TP=2 control (601,562) and TP=2 with the page fix (1,303,571) `[estimate]`:

| TP=3 arm | KDA state/slot | forced block | bytes/block | num_blocks | blocks/req | KDA blocks | **KV pool** | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **baseline, measured** | 1,610,752 | 3,328 | 22,572,800 | 2,040 | 363 | 36 | **5,619,834** | — |
| ReplaySSM, B=8 (ring 352 KiB) | 1,971,200 | 4,096 | 27,176,960 | 1,694 | 279 | 8 | **6,071,684** | **+8.0 %** |
| ReplaySSM, B=16 (ring 704 KiB) | 2,331,648 | 4,608 | 30,246,400 | 1,522 | 252 | 8 | 6,039,682 | +7.5 % |
| upper bound: slots only, no ring | 1,610,752 | 3,328 | 22,572,800 | 2,040 | 335 | 8 | 6,089,552 | +8.4 % |

| TP=2 arm | KDA state/slot | forced block | bytes/block | num_blocks | blocks/req | KDA blocks | **KV pool** | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **baseline, measured** | 2,342,912 | 4,608 | 30,246,400 | 365 | 280 | 36 | **1,303,571** | — |
| ReplaySSM, B=8 (ring 512 KiB) | 2,867,200 | 5,632 | 36,385,280 | 303 | 212 | 8 | **1,429,245** | **+9.6 %** |
| ReplaySSM, B=16 (ring 1 MiB) | 3,391,488 | 6,656 | 42,524,160 | 259 | 185 | 8 | 1,400,000 | +7.4 % |
| upper bound: slots only, no ring | 2,342,912 | 4,608 | 30,246,400 | 365 | 252 | 8 | 1,448,412 | +11.1 % |

**+8.0 % is well below what the slot arithmetic alone suggests, and the reason is the point.** The
ring lives *inside* the mamba page, so the page grows, so the forced attention block grows
(3,328 → 4,096), so the per-block cost grows and `num_blocks` falls — while `blocks_per_request`
falls too. The two effects very nearly cancel. **B = 1 + k = 8 is both mandatory and optimal**: the
ring length rounds up to a power of two, so B=16 at k=7 asks for 32 physical slots, twice the memory
for 6 % less gain.

| measure | TP=3 | TP=2 |
|---|---|---|
| KDA share of `blocks/request` | 9.9 % → **2.9 %** | 12.9 % → **3.8 %** |
| KDA share as equivalent memory | 4.25 → **1.23 GiB/rank** | 1.32 → **0.39 GiB/rank** |
| Memory that would buy the same pool *without* ReplaySSM | 42.89 → 46.33, i.e. **+3.45 GiB/rank** | 10.28 → 11.27, **+0.99 GiB/rank** |
| KDA state physically resident, per maximum-length request | 0.486 → **0.108 GiB** | 0.672 → **0.149 GiB** |

**What it costs, which is why it is not here.**

1. **The prefix-cache match unit coarsens** — 3,328 → 4,096 tokens at TP=3 (+23 %), 4,608 → 5,632 at
   TP=2 (+22 %). At two ranks our prefix hit rate is 0 %, so there is no measurable cost there; at
   three ranks it is **not measured**.
2. **Speed, and this is the real risk.** The replay loop is `tl.static_range(0, CACHE_BUF_LEN)`, fully
   unrolled at compile time, so it **runs its full length every step** regardless of how much of the
   ring is live; the verify window is added on top. At B=8/k=7 that is 16 replay iterations plus an
   8-step verify window = **24 sequential rank-1 updates against the baseline's 8 — 3.0×**. Our
   measured profile puts the KDA/GDN linear-attention class at **1.53 ms, 1.81 % of a C1 step** and
   **17.29 ms, 8.67 % of a C8 step**, so the arithmetic upper bound is **+3.6 % on a C1 step and
   +17.3 % on a C8 step — 197 tok/s down to about 168** `[estimate]`. Against that, the scheme writes
   one checkpoint instead of eight states per step, so HBM traffic falls. **The net direction cannot
   be derived from what we have; it has to be measured.** The author's own README says the baseline
   "remains the DFlash default because it is faster at C1" and gives no figure. And the flush cadence
   `(new_wp + 2 × MAX_SPEC_LEN) > MAX_CACHE_LEN` means that at B=8/k=7 on a 16-slot ring **nearly
   every step flushes**, giving back much of the traffic saving the design is named for — so the
   B-versus-C1 curve is not monotone and needs a sweep.
3. **Correctness is unproven.** The ring holds `d` and `k` in **fp16** where the baseline state is
   fp32, and the replay error accumulates across the write pointer. Reading the upstream harness we
   **could not find** a test that compares the ring against a baseline: the materializer test we did
   find checks Triton against Torch at `rtol=atol=0.02` with **both sides reading the same fp16
   ring**, so it would pass whether the ring is right or wrong, and the qualification script is a
   receipt verifier (environment constants, a 120/120 stress run, zero post-ready JIT) whose control
   arm only proves the image stays up. If such a test exists and we missed it we would rather be
   corrected than write our own.
4. **JIT.** Four new Triton kernels specialise on `MAX_SPEC_LEN`, `CACHE_BUF_LEN` and the head/block
   constants, so each `(k, B, heads)` triple is a compile. Our "zero post-ready JIT" rule needs a
   warmup for them. Worth knowing: the patch's `_is_blackwell()` tests `is_device_capability_family(100)`,
   and GB10 at sm_121 is family 120, so **every Blackwell tuning table in it is dead here** and the
   generic branch runs. The author's own part is SM120, so we would inherit his launch configurations
   untuned.

**Port cost, if it is ever taken: about 8 hours** — half an hour to wrap the patch with fail-closed
anchors, half an hour for the two drifted anchors, an hour for the conv-window port and the env-gated
arm, an hour of Triton warmup, two hours for a settle-gated A/B boot of both arms, and **three hours
for the correctness test that does not exist yet**, which is the real cost. Plus a sweep round for B
if C1 regresses.

**The decision is "not now", and the first step is a measurement rather than an installation.** The
pool does not bind at three nodes: a bigger pool prevents preemption, it does not add tokens per
second. It *did* bind at two. The item reopens if the pool starts binding, if two nodes become a real
serving arrangement, or if a numeric C1/C16 A/B is published upstream — and when it does, the first
thing to run is **one A/B boot at B=8, speed and gates only. If C8 loses more than 5 % the item closes
there; if it does not, the correctness test gets written.** It is on
[HELP-WANTED](../HELP-WANTED.md) §6 in exactly that order.

---

## 6. The two-node column

Source: the TP=2 `kvfix` arm, `gpu-memory-utilization` 0.85, two nodes, the **routed-experts-only**
checkpoint — attention in BF16, which is why its weights are so much heavier than the three-node
full-scope figure.

| Rank | Free at init | consumed | peak activation | CUDA graph | **Available KV** |
|---|---:|---:|---:|---:|---:|
| 0 | 111.94 | 90.65 | 2.42 | −0.13 | **10.31 ← binding** |
| 1 | 112.96 | 89.37 | 2.42 | −0.10 | 11.60 |

| Line | TP=3 production 10 | TP=2 `kvfix` |
|---|---:|---:|
| Device total | 121.63 GiB | 121.63 GiB |
| Memory fraction | 0.83 → 100.95 GiB budget | 0.85 → 103.38 GiB budget |
| Weights | **51.62** | **81.53** |
| non-torch | 7.28 – 7.55 | 7.84 – 9.12 |
| Peak-activation reservation | 1.66 | 2.42 |
| CUDA graph pool | 0.00 | −0.13 … +0.28 |
| **KV pool (binding rank)** | **40.12** | **10.31** |
| Attention block size (forced) | 3,328 tokens | 4,608 tokens |
| MLA page / indexer page | 1,703,936 / 109,824 B | 2,359,296 / 152,064 B |
| Draft page (5 layers) | 196,608 B | 524,288 B |
| **Per-block cost** | **20,934,400 B** | **30,246,400 B** |
| `num_blocks` | 2,058 | 365 |
| Blocks per 1M request | 363 (MLA 301 / k-pool 1 / KDA 36 / draft 25) | 280 (218 / 1 / 36 / 25) |
| **KV pool (tokens)** | **5,669,421** (5.67×) | **1,303,571** (1.30×) |
| GiB per 1M-token request | 7.078 | 7.887 |
| **KDA share of the divisor** | **9.9 %** | **12.9 %** |
| KDA state at 8-way concurrency | 5.61 GiB, 14.0 % of pool | 8.11 GiB |
| Free RAM after boot / swap | 5.1 GiB / 0.03 | 7.3 – 8.7 GiB / 0.02 – 0.03 |
| A 9 → 1 slot fix would give | 5,669,421 → 6,217,522 (**+9.7 %**) | 1,303,571 → 1,471,774 (**+12.9 %**) |

**Why two ranks feel this so much harder.** The pool is 10.31 GiB and one 1M-token request wants
7.89, so the pool holds **1.30** of them. Eight concurrent requests' KDA state alone is 8.11 GiB —
comparable to the whole pool. At three ranks that is a loss of headroom; **at two ranks it is a reason
things do not run**, and the long-prompt path only opened after the draft-page fix — on the control
arm nothing above about 6,300 tokens was ever scheduled ([15](15-tp2-track.md) §4).

**This column needs refreshing and we say so rather than quietly using it.** It is the experts-only
stack, and it was read while the two-node **full-scope** candidate was taking a dump boot — whose pool
figure reads about 6.7 % low and must never be quoted as a result ([09](09-measurement-protocol.md)
§11). Re-take it from that candidate's **load** boot.

---

## 7. What this page does not know

- **The 3.49 GiB residual (engine up) / 3.12 GiB (engine down) is unexplained by composition.** It is
  the same band in two independent states, so it is recorded as the driver's fixed reservation, but
  what is inside it — GPU page tables, a firmware carve-out, the module's own reservation — was **not
  measured**.
- **The non-torch 7.28–7.55 GiB was not broken down** into CUDA context, NCCL and mesh plugin buffers,
  cuBLAS and FlashInfer workspaces and EXL3 tune buffers. `Mlocked` at 24 MiB rules out a large
  pinned buffer, and that is all we know.
- **The KDA state's byte-level shape is derived, not read from the weights.** 1,610,752 B comes from
  `kda_state_shape` and the logged 5.79 % padding agrees with it; the TP=2 ↔ TP=3 scaling fits "a
  constant part plus ≈66,560 B per head". Confirming it properly means reading the model file.
- **`num_blocks` is still a difference between two `MemAvailable` readings.** Read the pool from a
  **load** boot on a settled host, never from a dump boot ([09](09-measurement-protocol.md) §11.1).
- **The two boots in §1 do not reconcile**, and the record cannot settle it because the
  production-10 container log was discarded.
- **No engine time was spent on any of this.** Every give-back in §5 is arithmetic or a code reading;
  none of it has been booted.

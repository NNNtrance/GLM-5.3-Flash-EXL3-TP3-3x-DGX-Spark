# Four sm_12x stack patches, measured: no race found, the fixes are free, and none of them shipped

Four findings against the GLM-5.3-Flash vLLM stack on sm_12x hardware, raised as
[Zeuss5/cuda-exl3 issue #6](https://github.com/Zeuss5/cuda-exl3/issues/6) and found by
[`tpurtell/glm-5.3-flash-ext3-2x-rtx`](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx)
(Apache-2.0). Credit for all four is theirs. We re-implemented them against our own anchors — our
image is not theirs and not the issue author's — put them behind runtime knobs, and ran a five-arm
A/B in a diagnostic image on 6 September 2026 `[measured-here]`.

**The verdict in one line: no PDL race is detectable on this topology, the patches cost nothing that
we can measure, and none of that is a reason to change a working production configuration.** They
ride with the next production change or they stay optional. Production configuration 10
(`exl3-zeus:754421f`) was **never modified**; every patched arm ran in a separate image and a
separate in-container tree, and the patches were verified absent from the live container, file by
file, when the campaign ended.

The three adoptable patches and how to wire them are in
[`../../tracks/tp3/patches-optional/sm12/`](../../tracks/tp3/patches-optional/sm12/). The probe is
[`../../bench/logit-divergence/`](../../bench/logit-divergence/). The narrative and the closure are
[docs/11](../../docs/11-open-issues.md) §2.27.

---

## 1. The four items

| # | Item | Class | What it changes |
|---|---|---|---|
| 1 | **The PDL gate.** vLLM's `is_arch_support_pdl()` returns `major >= 9`, so Programmatic Dependent Launch is on for sm_121 — hardware on which it was never qualified. tpurtell's dual-Spark qualification reported KDA recurrent-state races here and ships `major in (9, 10)` | correctness | Every KDA and mHC launch site. On GLM-5.3-Flash **34 of 45 layers are KDA** and `hc_mult` is 4, so all of them fire every step |
| 2 | **The K-pool top-k buffer.** `pool_topk = torch.empty(...)` at two sites; the top-k kernels only promise `min(k, valid)` outputs, so a short row's tail is allocator residue used as pool ids | correctness | `torch.full(..., -1)` — short rows become deterministically invalid |
| 3 | **Its reader.** `_expand_pools_and_append_tail_kernel` bounds the pool id from below only (`pid >= 0`), so a positive garbage id expands to `pid * POOL_SIZE + o` and is emitted as a real token index | correctness | Adds `& (pid < pool_len)`, which the kernel already computed |
| 4 | **The indexer's Triton specialisation.** `BuildPrefillChunkMetadataKernel.kernel` takes two scalars under a bare `@triton.jit` | jitter | `do_not_specialize` on both, so the kernel is one variant rather than up to three |

**Item 4 is smaller than the issue implies, and we say so with the mechanism.** In triton 3.7.1 an
int argument has exactly **three** specialisation classes — `("constexpr", 1)` at value 1,
`("i32","D")` at value % 16 == 0, `("i32","")` otherwise — not one variant per chunk offset. The
tree's own warmup already covers two of the three for `query_slice_start` and all three for
`query_slice_stop`, so **at most one cold compile per engine process** is at stake `[measured-here]`,
measured against the image's own triton by exercising `native_specialize_impl`.

**One call site the issue does not list.** On the EXL3 side the PDL gate also feeds
`model_executor/layers/mamba/ops/scatter_states.py` (recurrent-state scatter). It is present in this
image and absent from the other stack we compared against.

Two further items in the same issue were **recorded and not patched**: the Mamba block table's
`dcp_size` multiply, which is arithmetically inert at DCP=1 (`spec.block_size * 1`), and the FLA
`tensor_cache` identity memo, whose call path here produces a fresh `buf[:n]` slice every time — a
new object, so the cache misses rather than going stale. Both are silent-wrong-answer class and both
are re-examined the day their precondition changes `[not tested]`.

---

## 2. Design: what a race would have to look like, and why byte identity cannot see it

**The first plan was byte-for-byte identity of greedy completions, and it was abandoned before it
cost a boot.** The `cuda-exl3` author ran exactly that experiment on his own hardware first and got
**24 distinct completions out of 24 greedy runs — in both arms, PDL on and PDL off** `[reported]`.
Three independent reasons, none of them PDL:

1. `cuda-exl3`'s fused MoE epilogue accumulates each routed row into its token's row with a bf16
   `atomicAdd`. Atomic order is not fixed, so **the MoE stage is not bit-exact on any MoE model.**
2. `CUDA_EXL3_DETERMINISTIC=1` never covered that path — it disabled split-k on the dense path only.
   His fix `e7e345e` routes the deterministic path back through `exl3_moe_combine`, summing each
   token's top-k in a fixed `k` order.
3. Our TP=3 all-reduce is not bit-exact either, and no flag changes that.

Measured per-layer relative error against an fp64 reference at `top_k=8`, which is GLM-5.3-Flash's
setting `[reported]`:

| path | top_k=1 | top_k=4 | **top_k=8** | top_k=16 |
|---|---|---|---|---|
| fused epilogue (bf16 atomics) | 2.38e-3 | 3.53e-3 | **4.34e-3** | 5.51e-3 |
| `exl3_moe_combine` (fp32 sum) | 1.66e-3 | 1.66e-3 | **1.67e-3** | 1.66e-3 |

So the arithmetic floor a race would have to stand out from is ~4.3e-3 per layer with the fused path
live, and a run-to-run *varying* part of it exists whether or not PDL is on. The question therefore
is not "is the text identical" — it never will be — but **"how large is the run-to-run noise, and
does turning PDL off change its size."**

**The probe.** [`bench/logit-divergence/`](../../bench/logit-divergence/): 12 fixed prompts (4 code,
4 prose, 2 math, 2 JSON/tool), temperature 0, fixed seed, `max_tokens` 128, `logprobs` with
`top_logprobs=5`, `reasoning_effort: low`, **concurrency 1**, a per-run `cache_salt` so run 2
recomputes its prefill rather than hitting the prefix cache. Two runs per arm give that arm's
**within-arm floor**; run 1 of each arm gives the **between-arm** comparison. A prompt is flagged an
outlier if its between-arm max |Δlogprob| exceeds **K = 4** times the pooled within-arm p95, **or**
if it diverges in token sequence earlier than any within-arm pair did.

---

## 3. The arms

All diagnostic arms ran in the **same image** and off the **same fast-load sidecar**; arms are
selected by environment knobs only. Arm 1R differs from arm 2 by **exactly one knob**, which is the
clean paired test of the PDL question.

| # | Arm | Image | Tree | Knobs | Boot | KV pool |
|---|---|---|---|---|---:|---:|
| 0 | production 10, as-is | `exl3-zeus:754421f` | production | none (upstream gate ⇒ PDL on) | already up | 5,677,685 |
| — | dump boot | `exl3-zeus:e7e345e-dflash` | diagnostic | all four items, `PDL=1`, `FASTLOAD_MODE=dump` | 540 s | 5,404,958 |
| 1 | PDL on + all four items | `e7e345e-dflash` | diagnostic | `HAREM_PDL_SM12=1`, stats armed | 219 s | 5,666,666 |
| 2 | PDL off + all four items | `e7e345e-dflash` | diagnostic | `HAREM_PDL_SM12=0` | 265 s | 5,677,685 |
| 3 | PDL off, items 2–4 reverted | `e7e345e-dflash` | diagnostic | `HAREM_SM12_ITEMS=pdl`, `PDL=0`, stock overlay | 249 s | 5,694,214 |
| 1R | PDL on, repeat (knob-only diff against arm 2) | `e7e345e-dflash` | diagnostic | `HAREM_PDL_SM12=1`, no stats | 249 s | 5,691,460 |
| — | production restored | `exl3-zeus:754421f` | production | none | 220 s | 5,696,969 |

**Settings for every row.** Three DGX Spark (GB10) nodes, **48 SMs per device**, TP=3 with expert
parallelism, `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (full scope), `kv-cache-dtype fp8` and an fp8
draft cache, DFlash2 at k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`,
`--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `NCCL_ALGO=Ring`, `NCCL_MAX_NCHANNELS=8`, mesh
plugin with both links per peer and `NCCL_PTR_CUDA`, per-rank fast-load sidecar, warm MLA tuner
cache, temperature 0, `reasoning_effort: low`, realistic prompts, **median of three sweep rounds**
([09](../../docs/09-measurement-protocol.md) §1, [12](../../docs/12-tuner-cache.md)).
`gpu-memory-utilization` was **0.83 in every arm** and was not touched. Every diagnostic arm ran with
`CUDA_EXL3_DETERMINISTIC=1`. 6 September 2026.

**Runtime verification, because "the variable is set" is not "the code is running".** Inside the live
arm-3 container the patched body was read back —
`if os.environ.get("HAREM_PDL_SM12","0")=="1": return major >= 9` / `return major in (9,10)` — so on
sm_121 (major 12) PDL is off; and the item 2/3/4 markers in that same container read 0/0/0.

**The diagnostic image's speed numbers are not comparable with production 10** and are not compared
with it here. `CUDA_EXL3_DETERMINISTIC=1` restores a kernel and a per-MoE-layer `(rows, H)` tensor,
so only arms of the same image are compared. Arm 0 stands in the tables as a reference, not as a
control.

---

## 4. The race question: logit divergence

### 4.1 Within-arm noise floors

Two runs of the same arm, all 12 prompts pooled `[measured-here]`. Raw:
[`sm12-ab/logit-divergence-arm1-vs-arm2.txt`](sm12-ab/logit-divergence-arm1-vs-arm2.txt),
[`sm12-ab/logit-divergence-arm1r-vs-arm2.txt`](sm12-ab/logit-divergence-arm1r-vs-arm2.txt).

| Arm | positions | median \|Δlp\| | **p95 \|Δlp\|** | max \|Δlp\| | earliest token divergence | prompts that never diverged |
|---|---:|---:|---:|---:|---:|---:|
| 0 production 10 (no determinism flag) | 444 | 4.465e-04 | **1.074e-01** | 4.735e-01 | 6 | 1 / 12 |
| 1 PDL on | 793 | 1.467e-04 | **8.570e-02** | 4.907e-01 | 3 | 3 / 12 |
| 2 PDL off | 770 | 1.728e-05 | **7.842e-02** | 3.895e-01 | 3 | 2 / 12 |
| 3 PDL off, items off | 936 | 4.856e-05 | **1.130e-01** | 4.972e-01 | — | — |
| 1R PDL on (repeat) | 642 | 4.154e-05 | **7.359e-02** | 2.643e-01 | 9 | — |

**Arm 0's floor is higher and it should be.** The production image has no `CUDA_EXL3_DETERMINISTIC`
covering the MoE stage, so the fused epilogue's bf16 `atomicAdd` is in the floor. On the diagnostic
arms what remains is essentially the TP=3 all-reduce — and 2–3 of the 12 prompts then produce
**byte-identical 128-token continuations across two runs**, against 1 of 12 on production. The
determinism fix is visibly doing what it claims; the residual is the collective.

### 4.2 Between arms

**The paired test — arm 1R against arm 2, one knob apart:**

| Prompt class | n | max \|Δlp\| | p95 \|Δlp\| | median \|Δlp\| | earliest div | never diverged | max / within-floor p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 4 | 1.814e-01 | 9.507e-02 | 2.414e-05 | 26 | 2 | 2.3× |
| prose | 4 | 2.225e-01 | 1.534e-01 | 2.019e-03 | 21 | 0 | 2.9× |
| math | 2 | 3.028e-01 | 5.671e-02 | 4.381e-06 | 95 | 1 | 3.9× |
| json | 2 | 1.749e-01 | 5.463e-02 | 2.841e-03 | 3 | 1 | 2.3× |
| **pooled** | 12 | **3.028e-01** | 6.452e-02 | 1.979e-05 | 3 | 4 | **3.9×** |

**The first comparison — arm 1 against arm 2 — says the same thing:**

| Prompt class | n | max \|Δlp\| | p95 \|Δlp\| | earliest div | max / within-floor p95 |
|---|---:|---:|---:|---:|---:|
| code | 4 | 2.279e-01 | 6.980e-02 | 5 | 2.8× |
| prose | 4 | 2.405e-01 | 1.176e-01 | 29 | 3.0× |
| math | 2 | 1.758e-01 | 6.149e-02 | 50 | 2.2× |
| json | 2 | 2.875e-01 | 7.337e-02 | 16 | 3.6× |
| **pooled** | 12 | **2.875e-01** | 6.164e-02 | 5 | **3.6×** |

| Comparison | within-arm pooled floor p95 | between-arm max | between-arm p95 | outlier prompts |
|---|---:|---:|---:|---|
| arm 1 vs arm 2 | 8.005e-02 | 2.875e-01 | 6.164e-02 | **none** |
| arm 1R vs arm 2 (knob only) | 7.739e-02 | 3.028e-01 | 6.452e-02 | **none** |

**Three readings, all in the same direction.** In both comparisons the **between-arm p95 sits below
the within-arm p95** — 6.2–6.5e-02 against 7.7–8.0e-02, i.e. two arms differ *less* than two runs of
one arm. The between-arm max clears K = 4 in neither (highest 3.9×), and no prompt class stands out.
In the arm-1-against-arm-2 comparison the arms diverge in token sequence **later** from each other
(position 5) than two runs of the same arm do (position 3); in the paired arm-1R comparison the two
are equal at position 3.

**This is evidence, not a clearance certificate.** One prompt set, 128 generated tokens, concurrency
1. A race that needs a batch shape, a longer generation or a different prompt distribution would not
appear here.

---

## 5. Speed

### 5.1 Aggregate throughput, tok/s — the measure the noise band is defined on

Median of three rounds, realistic prompts `[measured-here]`:

| C | arm 0 prod | arm 1 PDL on | arm 1R PDL on | arm 2 PDL off | arm 3 PDL off, items off |
|---:|---:|---:|---:|---:|---:|
| 1 | 70.09 | 68.33 | 70.52 | 70.12 | 70.87 |
| 2 | 97.21 | 97.74 | 97.56 | 97.12 | 98.17 |
| 4 | 142.04 | 137.77 | 138.80 | 140.67 | 139.42 |
| 6 | 172.39 | 174.13 | 171.74 | 172.05 | 175.45 |
| 8 | 199.05 | 197.22 | 198.48 | 199.19 | 197.32 |

### 5.2 PDL on against off, six rounds pooled

Arms 1 + 1R against arm 2, identical items on both sides, same image:

| Metric | PDL on (6 rounds) | PDL off (3 rounds) | Δ | Noise band | Verdict |
|---|---:|---:|---:|---:|---|
| C1 aggregate tok/s | 69.54 | 70.12 | **+0.8 %** | ±4 % | in band |
| C8 aggregate tok/s | 197.85 | 199.19 | **+0.7 %** | ±3 % | in band |
| C1 per-stream decode | 73.29 | 77.28 | +5.4 % | ±4 % | **not resolvable** |
| cold single stream (700 tok) | 56.9–58.1 | 55.7–58.1 | ~0 | — | in band |

**The one number outside its band is the one to distrust, and the repeat is why.** `C1 per-stream
decode` has a round-to-round spread **inside** the PDL-on arms of 69.4–79.7 tok/s — 15 %, larger than
the effect being looked for. Arm 1 alone read 69.75, a nominal **+10.8 %** for turning PDL off;
repeating that identical configuration as arm 1R read **76.17**, which lands between the two PDL-off
readings (77.28 and 75.57). **Two of three measurements of the same quantity say no difference and
the third moves inside its own noise.** This repository has already published one kernel conclusion
drawn from a single pair of arms and had to withdraw it, which is why the rule is what it is
([09](../../docs/09-measurement-protocol.md) §2, [CONTRIBUTING](../../CONTRIBUTING.md)).

### 5.3 What items 2–4 cost — arm 3 against arm 2, both with PDL off

| Metric | items 2–4 on (arm 2) | items 2–4 off (arm 3) | Δ |
|---|---:|---:|---:|
| C1 aggregate tok/s | 70.12 | 70.87 | −1.1 % |
| C8 aggregate tok/s | 199.19 | 197.32 | +0.9 % |
| prefill-fresh tok/s | 1,733 | 1,748 | −0.9 % |
| prefill 7K, warm | 1,566 | 1,526 | +2.6 % |
| KV pool | 5,677,685 | 5,694,214 | −0.3 % |

**The predicted cost did not appear.** The preparation note priced item 2 at "a worst case 4 MiB
int32 memset per prefill chunk — 2048 rows × 512 × 4 B — and on GB10 the pool is host memory, so it
is not free." Measured: **no distinguishable prefill cost.** The signs do not even agree across the
five metrics, which is what a difference smaller than the noise looks like.

**Per-arm prefill, for anyone re-deriving the row above** — three fresh unseen prompts per arm, plus
the warm 7,382-token repeat `[measured-here]`:

| Arm | prefill-fresh, 3 unseen prompts (tok/s) | median | prefill 7K, second (warm) prompt |
|---|---|---:|---:|
| 0 production 10 | 1,738 / 1,774 / 1,779 | 1,774 | 1,594 |
| 1 PDL on | 1,704 / 1,729 / 1,758 | 1,729 | 1,571 |
| 2 PDL off | 790 / 1,733 / 1,757 | 1,733 | 1,566 |
| 3 PDL off, items off | 1,674 / 1,748 / 1,752 | 1,748 | 1,526 |
| 1R PDL on (repeat) | 1,713 / 1,736 / 1,755 | 1,736 | 1,592 |

The 790 tok/s reading in arm 2 is a single outlier on a first prompt; the medians are what the tables
carry, and it is printed here rather than dropped.

---

## 6. The gates

Correctness probe (10 questions) and code exam (12 questions), cold and after the benchmark
`[measured-here]`:

| Arm | correctness cold | code cold | correctness warm | code warm |
|---|---|---|---|---|
| 0 production 10 | 10/10 | 12/12 | 10/10 | 12/12 |
| 1 PDL on | 10/10 | 12/12 | 10/10 | 12/12 |
| 2 PDL off | 10/10 | 12/12 | 10/10 | 12/12 |
| 3 PDL off, items off | 10/10 | 12/12 | 10/10 | 12/12 |
| 1R PDL on (repeat) | 10/10 | 12/12 | 10/10 | **11/12** |

**The single deviation, stated rather than smoothed.** Arm 1R's warm code exam failed the `matrix`
question on an `AssertionError`. The exam was re-run **three more times on the same engine**: 12/12,
12/12, 12/12. Across the campaign 12 of 13 code exams were full. This stack is not bit-deterministic
(§2), a one-question flake on that floor is an expected event rather than a regression, and it is
**not PDL-related** — arm 1 and arm 0 sit at the same PDL setting and returned 12/12.

---

## 7. Item 4: the cold compile the client cannot see

Three fresh ~70K-token prompts per arm, each distinct so none lands in the prefix cache. Prompt
lengths moved between 69K and 71K from run to run, so **the rate is the number to read, not the wall
clock**; both are given `[measured-here]`.

| Arm | item 4 | process state | run 1 tok/s | run 2 | run 3 | stall (wall) | stall (rate) |
|---|---|---|---:|---:|---:|---:|---|
| 0 production 10 | off | warm, serving since boot | 1,862 | 1,855 | 1,863 | −1.21 s | none (run 1 fastest) |
| 1 PDL on | on | cold JIT already spent by a sacrificial prefill | 1,842 | 1,841 | 1,834 | −0.17 s | none |
| 2 PDL off | on | **cold, fresh process** | 1,801 | 1,610 | 1,826 | −6.58 s | none (run 2 is an outlier) |
| 3 PDL off, items off | off | cold, fresh process | — | — | — | +0.32 s | none |
| 1R PDL on | on | cold, fresh process | — | — | — | +0.71 s | none |
| — production restored | off | **cold, fresh process** | 1,826 | 1,855 | 1,850 | +0.52 s | none |

**No arm shows a jump above 2 % inside a ~38 s prefill, and the arms with item 4 on sit in the same
range as the arms with it off. Item 4's effect is not measurable from the client** — which is exactly
what the preparation note predicted of it: "one cold Triton compile per engine process disappears…
steady-state throughput does not change and must not be reported as if it did." The compile most
likely lands in the startup warmup/profile run and never reaches serving.

Arm 2's second run (1,610 tok/s, ~12 % low) is a lone outlier with no cause found — the stats hook
was off, no sweep was running, and the engine log shows no preemption or eviction. It is marked as an
outlier and nothing is concluded from it.

---

## 8. The instrument, and the MLA-prefill datum it produced

Arm 1 booted with the diagnostic stats hook armed. It reads back `topk_indices_buffer` at the end of
the prefill chunk loop in `sparse_attn_indexer_kpool` — the **token-granular** selection sparse MLA
actually attends to, not pool ids. Rank 0 only. The budget was exhausted after 12 chunks, so every
later measurement in that arm ran with the hook in its early-return state; the arm's first request
was a deliberately sacrificed 7.4K prefill so the startup profile run could not eat the budget, and
that request's timing is void by construction.

Steady chunk shape: **1,792 query rows**, `index_topk` 2048, `index_kpool` 4, buffer width 2176.
Pooled over the first four steady chunks (7,168 rows) `[measured-here]`. Raw:
[`sm12-ab/indexer-selection-histogram.txt`](sm12-ab/indexer-selection-histogram.txt).

**Selected keys per query row** — min 1,793 · p05 1,882 · **median 2,049** · p95 2,051 · max 2,051:

| bucket | rows | share |
|---|---:|---:|
| 1,792–1,900 | 428 | 6.0 % |
| 1,900–1,980 | 320 | 4.5 % |
| 1,980–2,020 | 160 | 2.2 % |
| 2,020–2,046 | 104 | 1.5 % |
| 2,046–2,048 | 8 | 0.1 % |
| 2,048 | 1,540 | 21.5 % |
| 2,049 | 1,536 | 21.4 % |
| 2,050 | 1,536 | 21.4 % |
| 2,051 | 1,536 | 21.4 % |

**Adjacent-row top-k overlap**, |A ∩ B| / min(|A|,|B|), 7,164 pairs — p05 0.7895 · **median 0.9258** ·
mean 0.9173 · p95 1.0000:

| bucket | pairs | share |
|---|---:|---:|
| 0.50–0.70 | 80 | 1.1 % |
| 0.70–0.80 | 356 | 5.0 % |
| 0.80–0.85 | 640 | 8.9 % |
| 0.85–0.90 | 1,338 | 18.7 % |
| 0.90–0.95 | 2,182 | 30.5 % |
| 0.95–1.00 | 2,568 | 35.8 % |

### 8.1 What the kernel author did with it, and what it closed

That pair of numbers went to [issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5) as the datum
he had asked for, to place this configuration between his two MLA-prefill arms. **It did not land
between them — it landed two orders of magnitude away from one of them**, and the result was that he
withdrew his own conclusion `[reported]`.

His two arms were a *drifting* one, in which about **2** keys of a row's selection turn over per row,
and an *independent* one, in which each row selects freshly. A median adjacent-row overlap of 0.9258
at ~2,049 selected keys is about **152 keys turning over per row** — 76× his drifting arm. He built a
third arm calibrated to that turnover and swept context, and in commit **`5fd7299`**, *"Correct the
MLA prefill ceiling: at production overlap there is no gap"*, reported: at **262K context** the
production-pattern arm runs within **1.6 %** of the fully cache-resident arm (**2,422.8 µs** against
**2,385.8 µs**) while the independent arm needs **3,474 µs**. The reason is that the live key set is
the **residence window** — about 4,096 keys, ≈4.5 MiB — and not the chunk's whole footprint, so it
fits even a 24 MiB L2.

**Consequence, and it is a deletion rather than an addition: MLA prefill is compute-bound at
production overlap, and the "21–26 % overlap gap, worth about 2 % of a prefill chunk" that was
quoted before this measurement does not exist.** The item closes at **zero**, and the only lever left
on that kernel is reducing the work it does rather than the traffic it moves. This repository never
published the 21–26 % figure in a document, so there is nothing here to retract — but it was live in
the thread this datum was produced for, and it is recorded here because the correction is the useful
half. A cheap falsification for a 48-SM part is on [HELP-WANTED](../../HELP-WANTED.md) §8 and we
have not run it `[not tested]`.

### 8.2 The instrument's own defect, recorded

The hook also emitted a derived `context` / `expected` / `deficit` triple. **It is not trustworthy
and was not used.** `context` ramps 448 → 896 across the 1,792 rows, exactly **+1 every 4 rows** —
i.e. the `cu_seqlen_ks/ke` it reads are **pool-granular** (`index_kpool` = 4) while `sel_count` is
**token-granular**. `expected = min(index_topk, context)` therefore compares two different units, and
its "every one of the 1,792 rows has a deficit" output is an artefact of that mismatch and nothing
else. The two distributions above are counted straight off the selection buffer and are unaffected.
**The instrument is not shipped in this repository**, and if it is ever revived the units have to be
fixed first.

---

## 9. Verdict, item by item

| # | Item | Race detected? | Free? | Into the next production configuration? |
|---|---|---|---|---|
| 1 | PDL gate | **No.** Between-arm max 3.0e-01 against a within-arm floor p95 of 7.7e-02, K = 4 not reached, no outlier prompt, and the arms diverge no earlier from each other than one arm does from itself | **Yes** — C1 +0.8 %, C8 +0.7 %, both inside the band; gates full | **No, for now.** No proven harm and no measured benefit. A change without a reason is not made |
| 2 | K-pool buffer init | n/a — correctness class | **Yes** — no measurable prefill cost (§5.3) | **Worth taking, but not alone** — a free insurance policy that only means something beside item 3 |
| 3 | `kpool_compress` upper bound | n/a | **Yes** — unmeasurable, as expected | **Worth taking** — it closes a silent-wrong-answer class at no cost |
| 4 | Indexer Triton specialisation | n/a — jitter class | **Yes** | **No.** What it removes cannot be measured from the client (§7); benefit and cost both measured at zero |
| — | the stats hook | n/a | deliberately expensive when armed | **Never in production**; diagnostic only, and §8.2 must be fixed first |

**The distinction items 2 and 3 have to be judged on.** Both are correctness-class: their value is
argued from a mechanism, not from a measurement — an unread positive pool id expanding into a real
token index. **This A/B did not show that defect firing.** What it showed is that fitting the
insurance costs nothing. Those are different statements and the production decision belongs to the
first one.

**What this cost, on the repository's own rule.** Speed: nothing measurable in either direction, on
five arms and fifteen sweep rounds. Quality: nothing — 10/10 and 12/12 in every arm, with one warm
flake that did not reproduce in three immediate re-runs. Memory: nothing — the KV pool spread across
the five arms is 5,666,666 to 5,694,214, **0.5 %**, and the patched arms sit on both sides of the
unpatched ones. The price was **engine time**: one dump boot and five arms, about two hours.

---

## 10. Two corrections found on the way, both worth more than the result

**(a) The diagnostic image was built short, and it had nothing to do with the patches.** The first
dump boot failed closed in `patch-fullscope-tp3.py` with *A1: anchor matched 0 times in
`vllm/models/glm5next/nvidia/model.py`*. Root cause: the production image is built in **two** stages
— the serve layer, then the 13-file DFlash2 port ([`patches/dflash2-port/`](../../patches/dflash2-port/)) —
and the diagnostic image had been built with the first stage only. `qwen3_dflash2.py` was therefore
absent, the prelude never ran `patch-dflash-tp3.py`, and because `models/glm5next/nvidia/model.py` is
one of the port's files, the full-scope anchor could not be found. The missing stage was built on all
three nodes (21–22 s, the Dockerfile's own symbol gate passed) giving `exl3-zeus:e7e345e-dflash`, and
the A/B ran on that tag. **The build note for that image said it was single-stage and did not say
what that cost** — a one-line omission that a fail-closed anchor turned into an hour.

**(b) A read-only overlay mount sits in front of the patches.** `start-tp3.sh` bind-mounts
`sparse_attn_indexer_kpool.py` from the GB10 top-k overlay directory **read-only** over the vLLM
package ([`patches/indexer-overlay/`](../../patches/indexer-overlay/)). Item 2 and the stats hook
target exactly that file, so from inside the container it cannot be written. The fix was to apply
those two patches to a **host-side copy** of the overlay before boot; the prelude scripts then
reported "already applied" for that file and touched only the image's own copies. **This is an
exception to the general pattern of gating patches in the prelude, and it applies to any future patch
that lands on an overlaid file.**

---

## 11. What was left on the nodes

The diagnostic tree (372 KB per node), the diagnostic fast-load sidecar (53 GB per node) and the
image `exl3-zeus:e7e345e-dflash` on all three nodes. Production `754421f` was restored, its autostart
unit left enabled, KV pool **5,696,969**, gates 10/10 · 12/12.

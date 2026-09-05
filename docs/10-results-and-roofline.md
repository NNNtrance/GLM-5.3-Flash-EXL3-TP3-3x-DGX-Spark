# 10 — Results and roofline

The full tables, the progression that produced them, and how close any of it is to what the hardware
can physically do.

**Settings for everything on this page unless a row says otherwise:** three DGX Spark (GB10) nodes,
TP=3 + expert parallel, `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at revision `b20c49ba` (EXL3 4bpw),
image `exl3-zeus:9bf594c`, KV `fp8`, DFlash2 draft k=7, `--block-size 256`,
`HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, CUDA graphs on, `gpu-memory-utilization 0.80`, per-rank fast-load sidecar,
`CUDA_EXL3_TUNE_CACHE` warm, mesh plugin built from `19924dcc` + patches 0004/0005/0006 with
`NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_PTR_CUDA=1 NCCL_MESH_FLUSH=1`, temperature 0, **reasoning
effort `low`**, 5 September 2026. Speed on this configuration is the **median of three sweep rounds**
— the persisted tuner cache is what makes three enough ([09](09-measurement-protocol.md) §1,
[12](12-tuner-cache.md)); the older arms in §2 are five-round medians with two discarded. Raw tables
in [`../results/`](../results/README.md).

**Rulers before roofline.** Every efficiency percentage on this page is against a bandwidth and a
GEMM peak **measured on this device in our own image**, not against a vendor figure — §4.1. The tools
are [`bench/bw.py`](../bench/bw.py) and [`bench/gemmpeak.py`](../bench/gemmpeak.py), they take
seconds, and running them in the same process as the thing being measured is not optional here: the
read-bandwidth ruler drifted 6.5 % between three runs on the same idle machine the same morning.

---

## 1. The production configuration

| | value |
|---|---|
| C1 aggregate / per stream | **56.9** / 63.6 tok/s `[measured-here]` |
| C2 / C4 / C6 / **C8** aggregate | 84.2 / 118.5 / 142.9 / **168.9** tok/s `[measured-here]` |
| C8 per stream | 26.0 tok/s `[measured-here]` |
| TTFT, C1 / C8 | 0.41 / 1.01 s `[measured-here]` |
| Draft acceptance / accepted tokens per step | 61–65 % / 5.3–5.5 `[measured-here]` |
| Prefill, fresh unseen ~8.3K prompts (median of 3) | **1,792** tok/s `[measured-here]` |
| Prefill, warm repeated 7K prompt | 1,506 tok/s `[measured-here]` |
| KV pool | **4,449,035** tokens (4.4 concurrent 1M-token requests) `[measured-here]` |
| Weights per node | 54.86 GiB `[measured-here]` |
| Boot, container start → API ready | **274 s** (~4.5 min) `[measured-here]` |
| Free host RAM / swap at rest | 11.3 / 12.6 / 12.5 GiB · ~0.1 GiB `[measured-here]` |
| Quality gates, cold and warm | 10/10 · 12/12 `[measured-here]` |

Two of those rows need their footnote said out loud rather than hidden in a tier label.

**Boot** was itemised on the fast-boot arm ([08](08-fast-boot.md)): 617.9 s → **273.6 s**, of which
weight loading is 67.2 s. Nothing in the two configurations after it touches the loader, so 274 s is
carried forward rather than re-itemised `[measured-here]`. A full restart driven from the workstation
— stop all three, drop caches, staggered start, wait for `/health` — measured 307 s wall on the
production arm, and that number includes the driver's own stop and stagger.

**Content types and mixed load are not re-measured here.** Both configurations after the fast-boot
arm were measured with the quick arm of the tiered protocol ([09](09-measurement-protocol.md) §9),
which runs neither probe. The last figures for them, on the fast-boot arm: code **47.9**, math
**59.0**, JSON **57.7**, prose **22.4** tok/s at a single stream, with draft acceptance
46 / 56 / 55 / **13 %**; mixed load 7.0 tok/s with a 4.9 s TTFT for the long prompt `[measured-here]`.
Prose is where a k=7 draft is wasted, and it is the one category where a shallower draft wins
([04](04-dflash2-port.md) §6). Whether the mesh work moved any of that is `[not tested]`.

---

## 2. How it got there

Each row is a boot with its own gates. Aggregate tok/s, medians of rounds 3–5 `[measured-here]`.

| Arm | C1 | C4 | C8 | prefill-fresh | KV pool | what changed |
|---|---|---|---|---|---|---|
| TP=2 + DFlash2 (two nodes) | 42.9 | 83.9 | 114.6 | — | 825,000 | reference |
| TP=3 + EP, stock kernels | 40.8 | 59.2 | 91.9 | — | 2,587,828 | the third node; **slower** |
| + the `n_rows` kernel fix | 49.4 | 97.0 | 137.2 | 1,474 | 2,571,230 | [05](05-expert-parallel-and-cuda-exl3-fixes.md) §2 |
| upstream `bc0e0f6` + combine staging, MNBT 4096 | 51.1 | 104.8 | 150.9 | 1,761 | 1,627,170 | adopted upstream, retired our patch |
| `f4987cf`, MNBT 2048 | 51.9 | 107.0 | 153.8 | 1,645 | 2,428,769 | skip MoE padding rows; pool back |
| + `NCCL_MAX_NCHANNELS=8` | — | — | 150.8 | 1,728 | 1,648,621 | [06](06-nccl-mesh.md), measured on the 4096 arm |
| + draft page 256 | 52.8 | 117.1 | 162.8 | 1,508 | **4,413,223** | [07](07-kv-and-draft-page.md) |
| + fast-boot sidecar | 54.4 | 114.6 | 161.8 | 1,704 | 4,484,848 | [08](08-fast-boot.md); boot 618 → 274 s |
| + `9bf594c`, tuner cache warm | 54.5 | 112.0 | 159.9 | 1,709 | 4,429,752 | [12](12-tuner-cache.md); speed unchanged by design, protocol 5 rounds → 3 |
| **+ dual cable + `NCCL_PTR_CUDA` (production)** | **56.9** | **118.5** | **168.9** | **1,792** | **4,449,035** | [06](06-nccl-mesh.md) §6–§8; the second cable of every pair had never carried a packet |

Three of those rows are the interesting ones. **The third node initially made the machine slower**, by
8–29 %, and that was a one-line kernel bug rather than a cost of the arrangement. **The largest single
jump in the KV pool cost no memory at all** — it was a per-request block counter, not bytes. And the
**last row is not a tuning win at all**: half the fabric had never been used, by any workload, since
the cluster was built. The two rows before it move by less than their own spread and are in the table
because they changed the boot and the measurement protocol, not the speed.

Rejected on the way, each with its own boot and gates:

| Arm | Why rejected |
|---|---|
| `NCCL_PROTO=Simple` | 2.8× worse at the C1 decode message, 4.4× at C8, no better at 16 MB. No boot spent — model-free `[measured-here]`. |
| draft depth k=5 | Higher acceptance rate, lower accepted tokens per step; −6.4 % at C1, −3.5 % at C8. Wins only in prose and at C4 `[measured-here]`. |
| `gpu-memory-utilization 0.85` | +19 % KV pool, no speed change, head node at 1.9 GiB free with 1.6 GB swap — breaks the 4 GiB rule `[measured-here]`. |
| `--max-num-batched-tokens 4096` | +9.5 % fresh prefill, −13 % mixed-load TTFT, −28.5 % KV pool. A judgement, reversible in one line `[measured-here]`. |
| MoE input-transform fusion (`61a17bc`) | +1–4 % end to end, but a later upstream commit takes the same win another way and upstream dropped the branch `[measured-here]`. |
| A 2,304-padded tensor-sliced arrangement instead of expert parallelism | On the fixed kernel it loses everywhere except M=1 and costs +12.5 % expert bytes out of the KV pool `[measured-here]`. |
| A one-sided RDMA_WRITE mesh transport (`patches/kernel/0007`) | Removes RNR retries to exactly zero and moves throughput by nothing: engine C1 56.4, C8 171.1, prefill-fresh 1,763 against production's 56.9 / 168.9 / 1,792 — differences in both directions, inside boot spread. The ceiling is the cards' PCIe slots, not the flow control ([06](06-nccl-mesh.md) §9–§10) `[measured-here]`. |

### 2.1 Measured since, not yet production: the draft cache at fp8

The DFlash2 drafter's own KV cache is `bf16` while the main cache is `fp8`. Moving it to fp8 shrinks
the drafter's page and should grow the pool by about 4.7 % ([07](07-kv-and-draft-page.md) §7). The
open question was never the arithmetic — it was whether the draft's sliding-window backend accepts an
fp8 cache at all, and whether a drafter attending over fp8 still proposes as well.

Both are now answered. The arm ran on a **dump boot** (it added three prelude patches, which
invalidates the fast-load sidecar — [09](09-measurement-protocol.md) §11), so its speed and quality
lines are valid and **its KV pool line is not** `[measured-here]`:

| | production 6 | draft KV fp8 (dump boot) | |
|---|---|---|---|
| C1 / C2 / C4 / C6 / C8 aggregate | 56.9 / 84.2 / 118.5 / 142.9 / 168.9 | 56.0 / 82.3 / 121.7 / 143.6 / **175.5** | medians of 3 rounds |
| TTFT, C1 / C8 | 0.41 / 1.01 s | **0.40 / 0.91 s** | |
| draft acceptance | 61–65 % | **60.1–64.0 %** (one C1 round at 57.3 %) | the gate was "stay in 60–65" |
| accepted tokens per step | 5.3–5.5 | 5.2–5.5 | |
| prefill-fresh (median of 3) | 1,792 | 1,790 | |
| prefill 7K, warm repeat | 1,506 | 1,532 | |
| gates, cold and warm | 10/10 · 12/12 | **10/10 · 12/12** | |
| free RAM / swap | 11.3 / 12.6 / 12.5 GiB · ~0.1 | 14.5 / 15.8 / 15.8 GiB · ~0.1 | a dump boot's ledger, not production's |
| KV pool | 4,449,035 | 4,382,920 — **not usable** | a dump boot writes 56 GiB per node through the page cache |

**The claim this supports is "it costs nothing", not "it is faster".** C8 reads +3.9 %, which does
clear that metric's ±3 % band ([09](09-measurement-protocol.md) §1.2), but this is one boot, in a
different memory state from production, and it has not been repeated — §2 of the protocol page says a
single pair decides nothing. The one number that would promote it is the pool, and that needs a load
boot. Open item: [11](11-open-issues.md) §2.18.

The mechanism is confirmed in the engine's own log rather than inferred: the drafter's page goes
393,216 → **196,608 bytes**, per-block cost 21,917,440 → **20,934,400 bytes** (−4.5 %), and the
blocks-per-request divisor stays at 363 — which is the whole point, since that divisor is what
collapsed the pool before ([07](07-kv-and-draft-page.md) §1).

---

## 3. Against the NVFP4 sibling stack

Same three nodes, same model, same draft, same prompt set, different quantization path. The NVFP4
figures are from [`NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark`](https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark)
`[measured-here]`.

| | **EXL3 TP=3 (this recipe)** | NVFP4 TP=3 |
|---|---|---|
| C1 aggregate | **56.9** | 57–60 |
| C8 aggregate | **168.9** | 150 |
| prefill, fresh | **1,792** | 1,585 (7K) |
| TTFT C1 | 0.41 s | **0.38 s** |
| KV pool | **4,449,035 @ 0.80** | 4,321,739 @ 0.88 |
| weights per node | **54.9 GiB** | 65.5 GiB |
| boot to serving | **274 s** | ~300 s |
| gates | 10/10 · 12/12 | 10/10 · 12/12 |
| MMLU | 86.4 ±0.7 (1,995-question sample, at TP=2) | 85.9 ±0.3 (full, 14,042 questions) |

**EXL3 is now level on single-stream decode and ahead on aggregate throughput, on prefill, on memory
and on boot.** A day earlier this row read "behind on single-stream and prefill"; what changed was
not the quantization but the fabric (§2, [06](06-nccl-mesh.md) §6–§8). The KV comparison stays the
sharper one: EXL3 reaches a larger pool at `gpu-memory-utilization 0.80` than NVFP4 reaches at 0.88,
which means headroom NVFP4 does not have.

Three caveats that matter, and they all point the same way. The MMLU numbers are not comparable — one
is a full run, one is a 1,995-question sample measured on a two-node arrangement. The prefill columns
are not the same measurement: ours is `prefill-fresh` on unseen prompts, the NVFP4 figure is a warm
7K prompt, and the honest comparison would need both stacks measured the same way. And **three
findings on this page are fabric-level, not format-level** — `NCCL_MAX_NCHANNELS=8` (+13 % at C8),
the idle second cable, and `NCCL_PTR_CUDA` — all three use the same plugin over the same wiring and
**none of them has been applied to the NVFP4 stack** `[not tested]`. If they transfer, most of this
table's advantage transfers with them and the comparison moves back. Read it as "this is what the
EXL3 stack does today", not as "EXL3 beats NVFP4".

---

## 4. Roofline

### 4.1 Measure the ruler first, and this one moved

Every roofline percentage published in this repository before 5 September was against a **vendor**
number, and every one of them was about **22 % optimistic** `[retracted]`. Both rulers have now been
measured on this device, in our own image, in the same process that ran the benchmarks
`[measured-here]`:

| ruler | measured on GB10 | vendor / implied | achieved |
|---|---|---|---|
| device read bandwidth, bf16 `sum` over 4 GiB | **225.2 GB/s** | 273 GB/s | 82 % |
| device copy bandwidth, read+write, 4 GiB | 214.5 | — | — |
| device read bandwidth, 2 GiB buffer | 205.8 | — | — |
| **BF16 dense GEMM peak**, 8192³ `torch.matmul` | **97.3 TFLOP/s** | ~125 (1 PFLOP FP4 ÷ 8) | 78 % |
| BF16 GEMM at an engine shape, 2032×5632×4096 | 80.4 | — | — |
| BF16 GEMM at an engine shape, 2032×4096×4096 | 91.8 | — | — |

Two tools, both in `bench/`, both a few seconds: [`bench/bw.py`](../bench/bw.py) and
[`bench/gemmpeak.py`](../bench/gemmpeak.py). Run them **in the same binary and the same run** as
whatever you are measuring, and quote the result beside every efficiency claim.

**The ruler itself drifts.** Three independent measurements the same morning on the same idle machine
gave 225.2, 239.6 and 240.9 GB/s — a 6.5 % spread `[measured-here]`. Percentages below are therefore
given against a band, not a point, wherever the difference matters. A `memset` ruler (`.zero_()`)
measured 196.8–198.2 GB/s, which is the right comparison for a kernel that only writes.

### 4.2 The byte model

From the checkpoint's own shapes `[estimate]`: 45 layers of which 43 carry routed experts (3–45;
layer 45 is the MTP layer and carries its own 288); 288 experts, top-8, one shared expert; hidden
4096, routed intermediate 2048. A routed expert is 3 × 4096 × 2048 = 25.2M parameters, about
**12.9 MB** at 4 bpw with its scales. Under expert parallelism each node holds 96 experts of 288, so
43 × 96 × 12.9 MB ≈ 49.6 GiB of expert weight per node — and the measured figure is 54.86 GiB, which
leaves about **5.3 GiB of BF16 non-expert weight per node** (attention, KDA, the shared expert,
`lm_head`). That reconciliation is the reason to trust the rest of the table.

With a k=7 draft a decode step verifies 8 tokens per sequence, so the expected number of distinct
experts touched per layer is `288 × (1 − e^(−8·tokens/288))`.

| Regime | Measured | Bytes per step per node `[estimate]` | Effective bandwidth | Share of **225 GB/s** | Share of 97.3 TFLOP/s |
|---|---|---|---|---|---|
| Decode C1, DFlash2 k=7 | 89.1 ms per engine step (measured directly, §5.3) | 8 verify tokens → ~19 of 96 experts per layer: 10.6 GB expert + 5.3 GB dense ≈ **15.9 GB** | ≈ **178 GB/s** | ≈ **79 %** | ≈ 2 % |
| Decode C8, DFlash2 k=7 | 223 ms per engine step (measured directly, §5.3) | 64 verify tokens → ~80 of 96 experts per layer: 44.3 GB + 5.3 GB ≈ **49.6 GB** | ≈ **222 GB/s** | ≈ **99 %** | ≈ 6 % |
| Prefill, 2048-token chunk | 1.109 s per chunk (measured directly, §5.2) | every expert touched: 53.3 GB + 5.3 GB ≈ **58.6 GB** | ≈ **53 GB/s** | ≈ **23 %** | ≈ 24 % |

**Reading.** Decode with the draft is **at** the memory roof, not near it — the C8 row lands at 99 %
of the measured ruler, which is a way of saying the byte model and the measurement agree rather than
that the hardware is saturated to the last percent. Faster kernels buy nothing there; the levers are
fewer bytes (a checkpoint that also quantizes attention) or higher acceptance. **Prefill is far from
both roofs**, at under a quarter of bandwidth and under a quarter of compute, and that is where
kernel work can still pay. §5 replaces this arithmetic with a measured breakdown.

Assumptions, and they are not small: KV reads are ignored (short contexts), the draft model's own
weights are ignored, expert-touch counts are expectations rather than measured histograms, and a
4 bpw expert is taken as 12.9 MB. **Treat every ratio as ±15 %.** What backs the byte model is that
the derived per-node weight total lands within 1 % of the measured one.

---

## 5. Where a step actually goes

§4 is arithmetic. This section is the measurement, on production configuration 6, and it changes what
the next piece of work should be. Full tables:
[`../results/profile/step-breakdown.md`](../results/profile/step-breakdown.md).

**How it was taken, because the method has a caveat.** The running engine was started without
`--profiler-config`, so `/start_profile` returns 404 and `nsys` on this version cannot attach to a
live process; a profiling boot was not available and restarting production to get one was not
allowed. So the breakdown is a reconciliation of three sources: the **structure** comes from an
earlier torch-profiler trace of the same model, same TP=3+EP, same batched-token budget, re-segmented
**per prefill chunk** rather than averaged over a window ([`bench/prof-analyze3.py`](../bench/prof-analyze3.py));
the **classes that changed** were re-measured model-free in the same image
([`bench/moe_stage_bench.py`](../bench/moe_stage_bench.py), [`bench/mesh_sweep.py`](../bench/mesh_sweep.py));
and the **totals** were measured on the live server by wall clock
([`bench/live-step.py`](../bench/live-step.py), [`bench/live-decode.py`](../bench/live-decode.py)).
Carrying the per-class ratios forward overshoots the measured GPU-busy time by **2.8 %**, and the
table below is normalised by that factor. One profiling boot would turn every re-costed row into a
measurement; until then, read this as ±3 % per class and treat NCCL as a band rather than a point,
since the residual most plausibly belongs to it. Add `--profiler-config` to your own launcher before
you need it.

### 5.1 The totals, and what a day of fabric and cache work did

Live server, fresh unseen prompts, `max_tokens=1`, 52 requests over five length steps
`[measured-here]`:

| prompt tokens | wall (s) | tok/s |
|---|---|---|
| 1,032 | 0.773–0.840 | 1,192–1,341 |
| 2,026–2,090 | 1.208–1.258 | 1,655–1,677 |
| 3,980–4,088 | 2.260–2.292 | 1,761–1,784 |
| 6,142–6,311 | 3.519–3.598 | 1,745–1,754 |
| **8,423–8,427** | **4.669–4.680** | **1,801–1,804** |

A least-squares fit over all 52 points gives **0.5456 ms per prompt token marginal** and a 139 ms
intercept (HTTP, tokenise, sample, detokenise).

| | one configuration earlier | production 6 | change |
|---|---|---|---|
| steady 2,032-token prefill chunk | 1,262.3 ms | **1,109 ms** | **−12.2 %** |
| end-to-end per prompt token, 8.3–8.4K | 0.7892 ms | **0.5547 ms** | **−29.7 %** |
| prefill throughput, 8.4K fresh | 1,257 tok/s | **1,802** | **+43 %** |
| C1 decode, ms per engine step | ~108 | **89.1** | **−17.5 %** |
| C8 decode, ms per engine step | — | **223** | — |

The gap between −12 % *inside* a chunk and −30 % end to end is the MLA tuner cache
([12](12-tuner-cache.md)): the older run paid 411 ms extra on its first chunk and 665 ms on its
17-token tail pass, re-tuning on every new shape.

### 5.2 Prefill, one steady 2,048-token chunk

Occupancy in the profiled window is **99.3 %** — there is 0.7 % of launch gap in the whole chunk, so
prefill is entirely kernel-bound and nothing is to be won on the host side `[measured-here]`.

| class | earlier ms | earlier % | **now ms** | **now %** | basis for "now" |
|---|---|---|---|---|---|
| MoE trellis GEMM (`exl3_gemm_m_kernel`, 84 calls) | 357.5 | 28.5 % | **291** | **26.4 %** | model-free, ratio 0.837 |
| NCCL all-reduce (102 calls × 16.8 MB) | 251.7 | 20.1 % | **182** | **16.5 %** | model-free, ratio 0.743 |
| Dense BF16 GEMM (cutlass / nvjet — the unquantized half) | 183.8 | 14.7 % | **179** | **16.2 %** | unchanged |
| **Hyper-connection mixing** (`mhc_*`, `hc_prenorm`) | 132.5 | 10.6 % | **129** | **11.7 %** | unchanged |
| MLA attention (`mla_decode_partial` + `reduce`) | 92.9 | 7.4 % | **90** | **8.2 %** | unchanged |
| KDA linear attention (triton chunked scans) | 85.0 | 6.8 % | **83** | **7.5 %** | unchanged |
| MoE `had_in` / `glu_had_in` | 70.1 | 5.6 % | **68** | **6.1 %** | model-free, ratio 0.993 |
| norm / elementwise | 42.1 | 3.4 % | **41** | **3.7 %** | unchanged |
| MoE `combine` / `build_inv` | 14.4 | 1.2 % | **16** | **1.5 %** | model-free, ratio 1.154 |
| KV block zeroing (`_zero_kv_blocks`, 1 call) | 14.7 | 1.2 % | **14** | **1.3 %** | unchanged |
| DSA indexer (`fp8_mqa_logits`, `topKPerRowPrefill`) | 6.8 | 0.5 % | **7** | **0.6 %** | unchanged |
| MoE align / route | 1.9 | 0.2 % | **2** | **0.2 %** | unchanged |
| **GPU busy** | **1,253.6** | | **1,101** | | |
| **wall** | **1,262.3** | occ 99.3 % | **1,109** | | measured |

**Two corrections to the earlier reading of the same trace** `[retracted]`. The `mhc_*` and
`hc_prenorm` kernels are not dense GEMM and not the indexer: they are **hyper-connection mixing**,
a class of their own worth 10–12 % (§5.5). And MLA is 7.4 % in a steady chunk, not the 9 % a window
average reported — the extra was the tail pass, where the tuner re-tuned on a new shape.

**The DSA indexer is 0.6 % of prefill.** Writing a device-capability-120 persistent or filtered top-k
would buy nothing here, which is the same conclusion a micro-benchmark reached from the other side
([05](05-expert-parallel-and-cuda-exl3-fixes.md)). Item closed.

**A near-empty chunk is cheap; a small one is not.** Crossing a 2,048-token boundary costs only the
tokens (2,041 → 2,058 adds 45 ms; 6,101 → 6,173 adds 45 ms), but a 128-token chunk costs **403 ms**
— 105 collectives (205 ms) plus a nearly complete MoE weight stream (104 ms), because 128 tokens ×
top-8 already touches every expert `[measured-here]`.

### 5.3 Decode, C1, verify batch M=8, drafter active

Segmented per step: the window spanned by that step's `exl3_gemm_m_kernel` calls is the target model,
everything else in the step is the draft plus head/tail and sampling. 11 steps, mean 94.9 ms
`[measured-here]`:

| class | target ms | draft ms | total | % |
|---|---|---|---|---|
| Dense BF16 GEMM (cutlass wmma 16×16 / 32×32) | 32.51 | **11.91** | 44.42 | **44.8 %** |
| MoE trellis GEMM | 29.08 | — | 29.08 | 29.3 % |
| NCCL all-reduce | 11.95 | **5.38** | 17.33 | 17.5 % |
| hyper-connection mixing | 1.94 | 0.18 | 2.11 | 2.1 % |
| KDA + norm + MLA + indexer + MoE aux | 3.7 | 0.9 | 4.6 | 4.6 % |
| **DFlash2 draft, total** | — | **18.50** | — | **19.5 %** |

Carried onto today's measured 89.1 ms step, the collective's share falls to **10–15 %** and the
totals reconcile within 5 % of the measurement. **At decode the unquantized half of the checkpoint is
the single largest item**, it gets *less* efficient per rank as ranks are added because each rank's
shard of a BF16 matrix shrinks, and nothing in the EXL3 kernel library touches it. The k=7 drafter
costs 19.5 % of a step and does its own all-reduce. The C8 split is `[not tested]` — that one needs a
profiling boot.

### 5.4 The MoE trellis GEMM is at the roof, and the traffic lever is closed

The largest single prefill class runs at **81–96 % of the measured 225 GB/s ruler**, model-free, EP
arm, one rank, per MoE layer (`w13 = [E, 4096, 4096]` at 4 bit = 8.389 MB per expert; traffic =
`local_blocks × 8.389 MB`) `[measured-here]`:

| M | block_m | `local_blocks` | distinct local experts | `gemm_w13` µs | GB/s if per-block | GB/s if per-expert | % of 225 GB/s |
|---|---|---|---|---|---|---|---|
| 8 | 16 | 17 | ≤17 | 662 | 215 | 215 | **96 %** |
| 64 | 16 | 77.5 | ≈67 | 2,975 | 219 | 189 | 84–97 % |
| 512 | 16 | 121.5 | 96 | 3,831 | 266 | 210 | 93 – >100 % |
| 2048 | 64 | 112 | 96 | 4,404 | 213 | 183 | **81–95 %** |

The kernel is not slow; it is taking most of what the memory can deliver. That left exactly one
lever — **read less**. At M=2048 the launch runs 112 blocks over 96 local experts and at M=512 it
runs 121.5, so *if* a block re-reads its expert's trellis, 17–27 % of that traffic is avoidable and
an expert-stationary schedule would be worth about 3.7 % of prefill wall.

**It does not re-read.** The kernel author added a bench for exactly this question
(`bench_moe_expert_reread.py`, `9b17ea9`); we ran it unmodified on GB10 against the production build,
three times, with the ruler in the same process — 96 experts, `block_m=16`, N = 1…4 blocks per expert
`[measured-here]`:

| N | blocks | rows | µs (run 1 / 2 / 3) | GB/s per expert | GB/s per block |
|---|---|---|---|---|---|
| 1 | 96 | 1,536 | 3,696 / 3,621 / 3,660 | 218–222 | 218–222 |
| 2 | 192 | 3,072 | 4,035 / 4,066 / 4,064 | 198–200 | 396–399 |
| 3 | 288 | 4,608 | 4,370 / 4,361 / 4,403 | 183–185 | 549–554 |
| 4 | 384 | 6,144 | 5,469 / 5,464 / 5,480 | 147 | 588–590 |

Doubling the block count costs **1.11×**, not 2×; quadrupling it costs 1.5×, not 4×. The trellis
**stays resident** across an expert's blocks — `moe_align_block_size` keeps an expert's blocks
adjacent and 8.4 MB per expert fits a 24 MiB L2. The 14–27 % traffic win the arm was built to find
**does not exist at this configuration, and the item is closed** `[measured-here]`. The reason the
per-block GB/s column climbs past the ruler is that an L2-resident re-read is cheap, not that DRAM is
being exceeded; the load-bearing evidence is the µs column being nearly independent of block count.
The same bench on the author's own 188-SM card gives 1.16×, so this is structural rather than a
property of 48 SMs.

That leaves `exl3_moe_had_in` as the only sub-roofline kernel in the MoE stage — 83–129 GB/s, **37–57 %
of the ruler**, 4.0 % of a prefill chunk. Upstream has since taken it (`a47da6e`, a 64-bit division
removed in favour of deriving the index from the grid): **−10 to −18 % on that kernel**, roofline
57 % → 63 %, which is worth ~0.2–0.3 % of prefill wall on this stack — real, and not worth an image
rebuild on its own `[reported]`. It goes into the next build bundle.

### 5.5 Hyper-connection mixing: 11.7 % of prefill, and one honest lever

`hc_mult = 4`, so the residual stream is carried in **four copies** — `(M, 4, 4096)` bf16, 32,768
bytes per token. Twice per layer (before attention and before the FFN) a fused post+pre block runs
**three kernels**: a post mapping that writes the new residual, a tf32 GEMM against a `(24, 16384)`
projection, and a fused sigmoid/softmax/Sinkhorn + RMSNorm that produces the layer input. Six
launches per layer, 275 per prefill chunk, 132 ms `[measured-here]`.

The analytic traffic is ~148,600 bytes per token, or **27.2 GB per chunk**; at 132 ms that is
**205.8 GB/s = 86–91 % of the ruler band**. Arithmetic intensity is 5.3 FLOP/byte against this
machine's balance point of ~405, so the class is memory-bound by a factor of 76 and the tf32 GEMM
inside it runs at 4.8 % of peak because there is nothing else for it to be. A model-free reproduction
in the same image landed within **0.8 %** of the trace ([`bench/mhc_bench.py`](../bench/mhc_bench.py)).

Three things this rules out, each of which looked plausible:

- **Not launch-bound.** With CUDA graphs on and off the 90-call sequence differs by **0.03 %** at
  M=2048 (it differs by 76 % at M=8, which is why decode runs in a graph) `[measured-here]`.
- **Not a bad TileLang configuration.** Sweeping the two tunables the call site leaves at their
  defaults is worth −4.9 % on the first kernel and −3.5 % on the third — **0.4 % of prefill**
  together. On the third kernel `threads > 96` does not compile at all (`no available layout`), so
  the production value is the widest one that builds, not a lazy default `[measured-here]`.
- **Not escapable via the torch fallback.** The reference implementation exists but is unreachable on
  CUDA — `forward_cuda` calls TileLang unconditionally, `enabled()` is overridden to `True`, and the
  model bypasses the `CustomOp` wrapper entirely by importing the TileLang functions directly. It is
  also 5.3–15.5× slower, adding ~900 ms to a chunk `[measured-here]`. It is a reference, not an
  escape hatch.

The one real lever is to **read the residual once less**: the second kernel's entire traffic is one
re-read of what the first kernel just wrote. Fusing them would save 30.5 % of that pair's bytes —
**−28 to −30 ms per chunk, −2.5 to −2.7 % of prefill**. The existing fused kernel does not do this
job: it is selected only at ≤16 tokens, it grids per token per n-tile, and forced at large M it is
**+32 % worse** at M=2048 and worse at every M we tried, including M=8 `[measured-here]`. The lever
needs a new large-M kernel that tiles over `block_m`, and it is vLLM/TileLang work, not EXL3 work.
An earlier estimate of ours put this at −3.6 %; measured, the ceiling is −2.7 %, so **that estimate
was 30 % optimistic** `[retracted]`.

### 5.5.1 That kernel was then written, and it reached 40 % of its own ceiling

A Triton kernel that grids over token tiles rather than over tokens — so the `(24, 16384)` projection
is shared across a tile and the post mapping is reduced against `fn` **while the row is still in
registers**, never landing in HBM for the second kernel to read back — was written, compiled and
measured model-free `[measured-here]`. Throwaway container, engine idle and untouched, disjoint
cpuset, both rulers measured inside each run ([09](09-measurement-protocol.md) §10), two independent
runs, median of 21 repetitions, CUDA graphs on, 90 calls = 45 layers × 2.

Winning configuration in both runs: `BLOCK_M=16, BLOCK_H=64, SPLIT_H=2, warps=4, stages=2`.

| M = 2048 | run 1 µs/call | GB/s | run 2 µs/call | GB/s |
|---|---|---|---|---|
| k1 `mhc_post` | 670.7 | 225.4 | 674.3 | 224.2 |
| k2 `hc_prenorm` | 311.6 | 221.1 | 313.9 | 219.4 |
| k3 `pre_big_fuse` | 431.3 | 195.4 | 434.7 | 193.8 |
| **kF, k1+k2 fused** | **815.7** | **187.7** | **815.2** | **187.9** |

| route, 90 calls | traffic | run 1 | run 2 | time |
|---|---|---|---|---|
| k1 + k2 (today) | 220.05 MB | 86.293 ms | 86.783 ms | — |
| **kF (fused)** | 153.14 MB (**−30.4 %**) | **73.416 ms** | **73.365 ms** | **−14.9 / −15.5 %** |
| R3 (production, 3 kernels) | 304.31 MB | 123.914 ms | 124.960 ms | — |
| **RF (fused, 2 kernels)** | 237.61 MB (−21.9 %) | **112.729 ms** | **112.649 ms** | **−9.0 / −9.9 %** |

**On the prefill wall that is 11.2–12.3 ms of a 1,109 ms chunk: −1.01 to −1.11 %.** A second,
independent route to the same figure agrees: this class is 11.7 % of a chunk (§5.2) and the route
gets 9.4 % cheaper, which is −1.10 %. **The target was −2.1 to −2.8 %; the kernel delivered a little
under half of it.**

**Why it stopped there, and it is not the tiling.** The fused kernel runs at 187.7 GB/s where the
k1+k2 route it replaces runs at 229.5 (220.05 MB in 958.8 µs). Traffic ratio 0.696 ÷ bandwidth ratio
0.818 = 0.851, which is the measured −14.9 % exactly, so the traffic model is right and the loss is
entirely bytes-per-second. Had the fused kernel reached k1's own band it would be −29.1 % on the pair
and −19.3 % on the route — **−2.2 % of prefill**, the bottom of the original estimate. A 33-configuration
sweep across two M values could not improve the winner, so this is not occupancy or tile shape. The
one concrete untried arm is the `tl.dot` operand path: `fn` is transposed inside the kernel
(`tl.trans` on a 32×64 fp32 block, staged through shared memory), and pre-transposing it once on the
host would remove that. Half an hour of work, **not measured, not claimed** `[not tested]`.

**Fusing loses below M ≈ 1024, and the reason is the L2.** `residual_cur` at M=512 is 16.8 MB and
fits the 24 MiB L2, so k2's "re-read" was never going to DRAM and there is nothing to delete — the
fusion pays its own cost for no saving `[measured-here]`:

| M | k1+k2, 90 calls | kF | |
|---|---|---|---|
| 8 | 0.633 ms | 3.541 ms | +459 % |
| 64 | 0.803 ms | 4.458 ms | +455 % |
| 512 | 13.859 ms | 19.090 ms | **+37.7 %** |
| 2048 | 86.293 ms | 73.416 ms | **−14.9 %** |
| 4096 | 175.900 ms | 142.051 ms | **−19.2 %** |

That is the measured justification for a size threshold, and it puts it at **M ≥ 1024**, not the 256
the module shipped with. The M=4096 row also says the gain grows with the chunk: under
`--max-num-batched-tokens 4096` the route figure is −13.2 %, which extrapolates to about −1.5 % of
prefill — an `[estimate]`, since a 4,096-token chunk's own profile was never taken.

**Correctness.** `residual_cur` is **bit-identical** at M = 8, 512, 2048 and 4096 (0 differing
elements out of 33.5M and 67.1M); at M=64, 7 elements of 1,048,576 differ by one bf16 ulp (1.9e-6),
which is fp32 FMA contraction near zero. `layer_input` differs by **at most one bf16 ulp** on 5.1 %
of elements (max absolute difference exactly 3.125e-2 = 2⁻⁵, one ulp in the [4, 8) binade);
`post_mix` and `comb_mix` by 0.21 % and 0.35 % relative. Inside bf16 rounding, and **not zero**.

**What it cost, and why it is not in production on its own.** Adopting it puts **Triton JIT into the
serving process** — a `/root/.triton` mount and an explicit warm-up before graph capture, or the first
large-M call compiles inside the capture — which is a new failure surface for −1 %. The config surface
is a cliff, not a slope: the winner reads 187.8 GB/s and its neighbours 79.4 and 44.5, and **the
default the module shipped with was one of the bad ones** (79.4 GB/s), so every hardware or shape
change needs the sweep run again. Set against the levers beside it — the collective at 16.5 % and the
MoE GEMM at 26.4 % — a −1 % change does not earn its own boot and its own A/B cycle. **Written,
measured, not adopted standalone; it rides the next image bundle together with `had_in`**, where one
boot measures both ([11](11-open-issues.md) §2.16, §2.19).

**One lesson is worth more than the kernel.** A GPU-free ahead-of-time compile check reported 18 of 18
configurations compiling and every one of them under the 99 KB shared-memory limit. At real launch
**6 of the 18 failed with `OutOfResources`**, because `metadata.shared` from the AOT path
under-reports what a launch actually needs — 36,864 bytes reported against 106,496 required, in one
case 40,960 against 147,456 `[measured-here]`. **A compile check answers "does it build", never "does
it run".** No harm was done, because the bench caught them and carried on, but the static test had
been written up as a pass.

### 5.6 KV block zeroing: at the memset roof, and the gain is not available

`_zero_kv_blocks_kernel` costs **14.7 ms in a single call per prefill chunk**, 1.3 %. vLLM zeroes
newly allocated blocks when the cache has Mamba layers **or** mixed precision, and this stack has
both: 34 KDA/Mamba layers, and a main cache at fp8 with the DFlash draft's own cache at bf16.

Reconstructing the live grid model-free — `[128, 720, 8]`, 32 KB pages —
reproduces the trace within 2.5 % and shows the kernel running at **100 % of the memset ruler**
(198 GB/s) `[measured-here]`. There is nothing to win in the kernel. What it is doing is zeroing
**2.4–2.9 GB per chunk** where the new tokens' real KV is about 3.4 MB.

The obvious lever — make the cache uniform-precision by moving the draft to fp8, then skip the
zeroing — **is not available on this model**, and finding out why is the useful part. The zeroer does
skip Mamba layers, which is what made "then the only remaining reason is mixed precision" look right.
But in this model's hybrid layout **one tensor is co-owned by an MLA layer and one Mamba layer from
each group**, so a block handed from the Mamba group to the attention group carries 1.7 MB of raw SSM
state. Measured per block: MLA pages co-owned with Mamba/KDA are **85.5 %** of the bytes being
zeroed, the indexer tail 5.5 %, the draft's sliding window 9.0 %. The Mamba half of vLLM's condition
is the binding one and it is independent of precision. The safe remainder — indexer plus draft, if
the cache were uniform — is worth **0.19 % of prefill**, which is not worth writing a partial mode
for `[measured-here]`. Item closed; the ceiling is recorded so nobody prices it again.

---

## 6. Ranked targets, and who owns them

Per 2,048-token prefill chunk (1,109 ms measured end to end), against the **measured** rulers of §4.1
`[measured-here]`:

| # | target | ms | share | achieved vs ruler | realistic gain | owner |
|---|---|---|---|---|---|---|
| 1 | NCCL all-reduce over the fabric | 182 | 16.5 % | ~20 GB/s bus against a ~30 GB/s per-node PCIe ceiling | ≤ **−2…4 %** prefill ([06](06-nccl-mesh.md) §9) | plugin / NCCL |
| 2 | Dense BF16 GEMM, Ampere-class `cutlass_80_*` on sm_121 | 179 | 16.2 % | 63.6 against 80.4 TFLOP/s at the same shape = **79 %** | −3.1 % if it reached 95 % | vLLM / cuBLAS |
| 3 | Hyper-connection mixing, 3 passes over a 4× residual | 129 | 11.7 % | 86–91 % of the ruler | ceiling −2.5…2.7 %; the kernel now exists and delivers **−1.0…1.1 %** (§5.5.1) | vLLM / TileLang |
| 4 | MLA prefill (`mla_decode_partial` runs at prefill too) | 90 | 8.2 % | **not measured** — the trace does not carry the selected-key count | ? | cuda-exl3 |
| 5 | KDA linear attention (triton) | 83 | 7.5 % | **not measured** | ? | vLLM |
| 6 | MoE trellis GEMM, large M | 291 | 26.4 % | 81–96 % — and the duplicate-read lever is **closed** (§5.4) | ~0 | cuda-exl3 |
| 7 | `exl3_moe_had_in` | 44 | 4.0 % | 37–57 % — the only sub-roofline kernel in the MoE stage | −0.2…0.3 % (taken upstream, `a47da6e`) | cuda-exl3 |
| 8 | `_zero_kv_blocks_kernel` | 14 | 1.3 % | 100 % of the memset ruler; gain **not available** (§5.6) | ~0 | vLLM |
| 9 | `exl3_moe_glu_had_in` | 24 | 2.2 % | 190 GB/s = 84 % | −0.3 % | cuda-exl3 |
| 10 | `exl3_moe_combine` | 16 | 1.5 % | 205 GB/s = 91 % | ~0 | cuda-exl3 |
| 11 | DSA indexer | 7 | 0.6 % | — | ~0, **closed** | — |

Two things a reader should take from that table. **The two largest prefill items are outside the EXL3
kernel library entirely** — the fabric and the unquantized half of the model — and the largest item
that is inside it is already at the roof. And **there is no single-digit-percent win left anywhere in
prefill**: the honest list is 2–4 % from the fabric, 2.5–2.7 % from a hyper-connection kernel nobody
has written, 3.1 % from someone shipping Blackwell-class dense GEMM kernels for sm_121, and a
scattering of fractions. The one **config** lever that is larger than all of them is the batched-token
budget: the MoE stage's marginal cost falls to 1.38 µs per token per layer at M=2048, so MNBT 4096
would cut that stage's per-token cost by about a third — and it costs the KV pool, which is why 2048
stands ([07](07-kv-and-draft-page.md), [11](11-open-issues.md) §2.5) `[measured-here]`.

---

## 7. What we did not measure

- **Anything at max reasoning effort** `[not tested]`. See [09](09-measurement-protocol.md) §7.
- **MMLU at TP=3** `[not tested]` — the sample was run at TP=2.
- **IFEval, GSM8K, needle-in-a-haystack, tool-eval-bench, ExtractBench** on this stack
  `[not tested]`. All of them exist for the NVFP4 sibling; none has been re-run here. Anyone
  comparing the two on quality should treat this repository as having only the gates and one MMLU
  sample.
- **Prefix caching.** With a 3,328-token attention block, our benchmark prompts never fill one, so
  the prefix-cache hit rate is 0 % throughout and the benchmark says nothing about it
  `[measured-here]`.
- **Long-context behaviour at scale.** The KV pool supports about 4.4 concurrent million-token
  requests and we never drove it past 13 % usage `[measured-here]`.
- **Content types and mixed load on the production configuration** `[not tested]` — the last two arms
  were measured at tier B ([09](09-measurement-protocol.md) §9), which runs neither probe. §1 carries
  the fast-boot arm's figures and says so.
- **A torch-profiler run on the production configuration** `[not tested]`. The engine was launched
  without `--profiler-config` and restarting it was not on the table, so §5 is a reconciliation with a
  2.8 % residual rather than a direct profile. The two rows it would settle are the NCCL band
  (14–17 % of prefill) and the C8 decode split. Cost: one boot. Launch with the profiler directory set
  and the question closes.
- **The MLA prefill kernel's efficiency** `[not tested]`. It is 8.2 % of a prefill chunk and the trace
  does not carry the selected-key count, so there is no denominator to divide by. It needs its own
  model-free measurement before anyone calls it a target.
- **`--max-num-batched-tokens 4096` on the current configuration** `[not tested]`. §6 gives the
  arithmetic for why it is the largest single prefill lever left, and [07](07-kv-and-draft-page.md)
  gives the KV price it was rejected on two configurations ago. Nobody has re-run it since.

---

## 8. What is next

[11 — Open issues](11-open-issues.md): what is unresolved, what we retracted, and what we never ran.

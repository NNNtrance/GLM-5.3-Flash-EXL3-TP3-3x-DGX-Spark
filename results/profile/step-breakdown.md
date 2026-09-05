# Step-time breakdown, production configuration 6

Image `exl3-zeus:9bf594c`, TP=3 + expert parallel, EXL3 4bpw, KV `fp8`, DFlash2 k=7,
`--max-num-batched-tokens 2048`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-seqs 8`,
`gpu-memory-utilization 0.80`, `NCCL_MAX_NCHANNELS=8`, mesh plugin 0004+0005+0006 with both cables and
`NCCL_PTR_CUDA`, warm MLA tuner cache, temperature 0, reasoning effort `low`, KV pool 4,449,035
tokens, 5 September 2026. `[measured-here]`

Narrative: [`../../docs/10-results-and-roofline.md`](../../docs/10-results-and-roofline.md) §4–§6.

**Method, and its caveat.** The engine was launched without `--profiler-config`, so `/start_profile`
returns 404 and this `nsys` cannot attach to a live process; a profiling boot was not available. The
breakdown is therefore a reconciliation of three sources: structure from an earlier torch-profiler
trace of the same model and the same TP=3+EP arrangement, re-segmented **per prefill chunk** rather
than averaged over a window (`bench/prof-analyze3.py`); the classes that changed re-measured
model-free in the same image (`bench/moe_stage_bench.py`, `bench/mesh_sweep.py`); totals measured on
the live server by wall clock (`bench/live-step.py`, `bench/live-decode.py`). Carrying the per-class
ratios forward overshoots the measured GPU-busy time by **2.8 %** and the table is normalised by that
factor. Treat each class as ±3 % and NCCL as a **14–17 %** band, since the residual most plausibly
belongs to it.

## 1. The rulers, measured on the device in the same image

| ruler | measured | vendor / implied | achieved |
|---|---|---|---|
| device read bandwidth, bf16 `sum`, 4 GiB | **225.2 GB/s** | 273 GB/s | 82 % |
| device copy bandwidth, read+write, 4 GiB | 214.5 | — | — |
| device read bandwidth, 2 GiB | 205.8 | — | — |
| BF16 dense GEMM peak, 8192³ | **97.3 TFLOP/s** | ~125 | 78 % |
| BF16 GEMM at 2032×5632×4096 | 80.4 | — | — |
| BF16 GEMM at 2032×4096×4096 | 91.8 | — | — |
| `memset` (`.zero_()`) | 196.8 – 198.2 GB/s | — | — |

Tools: `bench/bw.py`, `bench/gemmpeak.py`. **The read ruler drifted 6.5 % across three runs on the
same idle machine the same morning** (225.2 / 239.6 / 240.9), so efficiency figures below are given
against a band where the difference matters.

## 2. Prefill ladder, live server, fresh unseen prompts, `max_tokens=1`

52 requests over five length steps, three repetitions each:

| prompt tokens | wall (s) | tok/s |
|---|---|---|
| 1,032 | 0.773 – 0.840 | 1,192 – 1,341 |
| 2,026 – 2,090 | 1.208 – 1.258 | 1,655 – 1,677 |
| 3,980 – 4,088 | 2.260 – 2.292 | 1,761 – 1,784 |
| 6,142 – 6,311 | 3.519 – 3.598 | 1,745 – 1,754 |
| 8,423 – 8,427 | 4.669 – 4.680 | **1,801 – 1,804** |

Least squares over all 52 points: **0.5456 ms per prompt token marginal**, intercept 139 ms, maximum
residual 155 ms.

Chunk-boundary probe (26 requests): crossing 2,048 tokens costs only the tokens — 2,041 → 2,058 adds
45 ms, 6,101 → 6,173 adds 45 ms. A *small* chunk is not cheap: a 128-token chunk costs 403 ms
(105 collectives, 205 ms; MoE weight stream, 104 ms), because 128 tokens at top-8 already touch every
expert.

## 3. Prefill, one steady 2,048-token chunk

Occupancy 99.3 % — 0.7 % launch gap in the whole chunk.

| class | earlier ms | earlier % | now ms | now % | basis |
|---|---|---|---|---|---|
| MoE trellis GEMM (`exl3_gemm_m_kernel`, 84 calls) | 357.5 | 28.5 % | 291 | 26.4 % | model-free ratio 0.837 |
| NCCL all-reduce (102 × 16.8 MB) | 251.7 | 20.1 % | 182 | 16.5 % | model-free ratio 0.743 |
| Dense BF16 GEMM (cutlass / nvjet) | 183.8 | 14.7 % | 179 | 16.2 % | unchanged |
| Hyper-connection mixing (`mhc_*`, `hc_prenorm`) | 132.5 | 10.6 % | 129 | 11.7 % | unchanged |
| MLA attention | 92.9 | 7.4 % | 90 | 8.2 % | unchanged |
| KDA linear attention | 85.0 | 6.8 % | 83 | 7.5 % | unchanged |
| MoE `had_in` / `glu_had_in` | 70.1 | 5.6 % | 68 | 6.1 % | model-free ratio 0.993 |
| norm / elementwise | 42.1 | 3.4 % | 41 | 3.7 % | unchanged |
| MoE combine / build_inv | 14.4 | 1.2 % | 16 | 1.5 % | model-free ratio 1.154 |
| KV block zeroing (1 call) | 14.7 | 1.2 % | 14 | 1.3 % | unchanged |
| DSA indexer | 6.8 | 0.5 % | 7 | 0.6 % | unchanged |
| MoE align / route | 1.9 | 0.2 % | 2 | 0.2 % | unchanged |
| **GPU busy** | **1,253.6** | | **1,101** | | |
| **wall** | **1,262.3** | | **1,109** | | measured |

## 4. Decode

Per engine step, measured live via `vllm:spec_decode_num_drafts_total` (one draft per sequence per
step), two rounds:

| | C1 | C8 |
|---|---|---|
| ms per engine step | 89.05 / 89.09 | 220.8 / 226.0 |
| tokens per step per sequence | 5.82 | 5.32 – 5.48 |
| decode-window tok/s | 65.3 | 188 – 198 |
| end-to-end tok/s | 60.6 – 60.7 | 177 – 186 |
| draft acceptance | 70.8 % | 62.2 – 64.6 % |
| verify batch M | 8 | 64 |

C1 class split (11 steps, mean 94.9 ms on the earlier trace, segmented target versus draft):

| class | target ms | draft ms | total | % |
|---|---|---|---|---|
| Dense BF16 GEMM | 32.51 | 11.91 | 44.42 | 44.8 % |
| MoE trellis GEMM | 29.08 | — | 29.08 | 29.3 % |
| NCCL all-reduce | 11.95 | 5.38 | 17.33 | 17.5 % |
| hyper-connection mixing | 1.94 | 0.18 | 2.11 | 2.1 % |
| KDA + norm + MLA + indexer + MoE aux | 3.7 | 0.9 | 4.6 | 4.6 % |
| **DFlash2 draft total** | — | **18.50** | — | **19.5 %** |

Carried onto the measured 89.1 ms step, the collective falls to **10–15 %**. The C8 split is
`[not tested]`.

## 5. MoE stage, model-free, per layer, EP arm, one rank

| M | block_m | stage | earlier µs | now µs | Δ |
|---|---|---|---|---|---|
| 2048 | 64 | `gemm_w13` | 5,089 | 4,404 | −13.5 % |
| 2048 | 64 | `gemm_w2` | 2,822 | 2,214 | −21.5 % |
| 2048 | 64 | `had_in` | 907 | 910 | +0.3 % |
| 2048 | 64 | `glu_had_in` | 476 | 462 | −3.0 % |
| 2048 | 64 | `combine6` | 319 | 368 | **+15.4 %** |
| 2048 | 64 | **whole stage** | **9,613** | **8,357** | **−13.1 %** |
| 64 | 16 | whole stage | 4,984 | 4,732 | −5.1 % |
| 8 | 16 | whole stage | 1,110 | 1,051 | −5.3 % |

`combine` is the one regression and it is recorded rather than dropped: +15 % on a stage worth
16 ms per chunk.

Per-token cost against M, whole stage, per layer: 131.4 µs at M=8, 73.9 at 64, 12.2 at 512, **4.08 at
2048**, with a marginal cost at M=2048 of **1.38 µs per token per layer** — which is why the
batched-token budget is the cheapest prefill lever available before any kernel work, and why it costs
KV pool.

Weight-traffic efficiency of the largest class (`w13` = 8.389 MB per expert at 4 bit):

| M | block_m | `local_blocks` | distinct experts | `gemm_w13` µs | GB/s per block | GB/s per expert | % of 225 GB/s |
|---|---|---|---|---|---|---|---|
| 8 | 16 | 17 | ≤17 | 662 | 215 | 215 | 96 % |
| 64 | 16 | 77.5 | ≈67 | 2,975 | 219 | 189 | 84–97 % |
| 512 | 16 | 121.5 | 96 | 3,831 | 266 | 210 | 93 – >100 % |
| 2048 | 64 | 112 | 96 | 4,404 | 213 | 183 | 81–95 % |

## 6. Expert re-read (arm C) — does the trellis stay resident?

`bench_moe_expert_reread.py` from `cuda-exl3` `9b17ea9`, run unmodified against the production image
in a throwaway container, three runs. 96 experts, `block_m=16`, N blocks per expert. Ruler in the same
process: 234 / 197 / 240 GB/s (median 234).

| N | blocks | rows | µs run 1 / 2 / 3 | GB/s per expert | GB/s per block |
|---|---|---|---|---|---|
| 1 | 96 | 1,536 | 3,696 / 3,621 / 3,660 | 218 – 222 | 218 – 222 |
| 2 | 192 | 3,072 | 4,035 / 4,066 / 4,064 | 198 – 200 | 396 – 399 |
| 3 | 288 | 4,608 | 4,370 / 4,361 / 4,403 | 183 – 185 | 549 – 554 |
| 4 | 384 | 6,144 | 5,469 / 5,464 / 5,480 | 147 | 588 – 590 |

N=1 → N=2 time ratio, median of three: **1.11×** (1.09 / 1.12 / 1.11) against 2× blocks. The script's
own verdict line reads `RESIDENT across blocks` for every N>1 in all three runs. The per-block GB/s
column exceeding the ruler is the L2 hit, not DRAM being exceeded; the load-bearing evidence is the µs
column. The author reports 1.16× on a 188-SM card. **The duplicate-read lever does not exist here.**

## 7. Hyper-connection mixing

Per steady chunk, 45 layers × 2 calls = 90 fused calls, 6 kernel launches per layer, 275 launches:

| kernel | MB per call | µs per call | GB/s | % of 225.2 | % of 239.6 |
|---|---|---|---|---|---|
| post mapping (`mhc_post_tilelang`) | 149.98 | 682.4 | 219.8 | 97.6 % | 91.7 % |
| tf32 prenorm GEMM (`hc_prenorm`) | 68.35 | 341.6 | 200.1 | 88.9 % | 83.5 % |
| fused pre + RMSNorm (`pre_big_fuse`) | 83.60 | 404.8 | 206.5 | 91.7 % | 86.2 % |
| **total** | **301.93** | **1,429** | **205.8** | **91.4 %** | **85.9 %** |

Model-free reproduction in the same image at M=2048 landed within **0.8 %** of the trace
(1,417 against 1,429 µs per call). Arithmetic intensity 5.3 FLOP/byte against a balance point of ~405.

CUDA graph on/off, 90 calls, median of ≥20 repetitions:

| M | eager ms | graph ms | Δ |
|---|---|---|---|
| 8 | 3.515 | 0.855 | −75.7 % |
| 64 | 3.637 | 1.120 | −69.2 % |
| 128 | 3.791 | 2.491 | −34.3 % |
| 512 | 17.722 | 17.691 | −0.2 % |
| **2048** | **127.541** | **127.502** | **−0.03 %** |

Torch fallback (unreachable on CUDA in this model path, measured for the record): 5.3× slower at M=8
rising to 15.5× at M=512, 8.9× at M=2048.

Forcing the existing fused kernel at large M, best of 32 tile/split-k combinations per M: **+8.2 %**
at M=8, +113 % at 64, +185 % at 128, +73 % at 512, **+32.3 %** at 2048. It never wins.

Config sweep: the post kernel at `n_thr=512, h_blk=4096` is −4.9 % against the production default;
the fused-pre kernel at `hidden_block=2048` is −3.5 %; `threads > 96` on that kernel does not compile
(`no available layout`). Together **−0.4 % of prefill**.

## 8. KV block zeroing

Live grid from the trace: `[128, 720, 8]`, `BLOCK_SIZE=1024`, 15.56 ms and 13.53 ms on two chunks.
Reproduced model-free with a synthetic segment table at the same geometry:

| page per segment | max_chunks | zeroed | ms | GB/s | % of memset ruler (198.2) |
|---|---|---|---|---|---|
| 8 KB | 2 | 720 MB | 3.796 | 198.9 | 100.3 % |
| 16 KB | 4 | 1,440 MB | 7.600 | 198.7 | 100.2 % |
| 24 KB | 6 | 2,160 MB | 11.740 | 192.9 | 97.3 % |
| **32 KB** | **8** | **2,880 MB** | **15.180** | **198.9** | **100.4 %** |

So **2.4–2.9 GB is zeroed per prefill chunk** where the new tokens' real KV is ~3.4 MB, at 100 % of
the memset roofline. What is being zeroed, per block, at the production geometry:

| component | layers | page (B) | per block (B) | share | skippable if the cache were uniform? |
|---|---|---|---|---|---|
| MLA, co-owned with Mamba/KDA state | 11 | 1,703,936 | 18,743,296 | **85.5 %** | **no** |
| indexer (+ kpool tail) | 11 | 109,824 | 1,208,064 | 5.5 % | yes |
| DFlash2 draft sliding window @256 | 5 | 393,216 | 1,966,080 | 9.0 % | yes |
| **total** | | | **21,917,440** | | |

Bucketing the grid by page size (the obvious kernel-side fix) is ×1.32–1.89 on a synthetic geometry
with a 13:1 page ratio and **×1.0** on the live one. Nothing to win there either.

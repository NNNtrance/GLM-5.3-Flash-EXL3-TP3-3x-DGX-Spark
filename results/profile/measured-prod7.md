# Step-time breakdown, **measured** — production configuration 7

Torch profiler, all three ranks, on the **live server**, no restart and no reconfiguration: only
`POST /start_profile`, `POST /stop_profile` and ordinary chat requests. Image `exl3-zeus:9bf594c`,
TP=3 + expert parallel, EXL3 4bpw, `--kv-cache-dtype fp8` **and** `HAREM_DRAFT_KV_DTYPE=fp8`,
DFlash2 k=7, `--max-num-batched-tokens 2048`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`,
`--max-num-seqs 8`, `gpu-memory-utilization 0.80`, KV pool **4,719,008 tokens**, `NCCL_ALGO=Ring`,
`NCCL_MAX_NCHANNELS=8`, mesh plugin with both cables and `NCCL_PTR_CUDA`, prefix caching on,
temperature 0, reasoning effort `low`, 5 September 2026. `[measured-here]`

Narrative: [`../../docs/10-results-and-roofline.md`](../../docs/10-results-and-roofline.md) §5.
Protocol: [`../../docs/09-measurement-protocol.md`](../../docs/09-measurement-protocol.md) §4.1.

> **This file replaces the re-derived tables in
> [`step-breakdown.md`](step-breakdown.md) §3 and §4.** That file carried the per-class ratios of an
> older trace forward onto a newer wall clock and normalised the residual away; every row below is
> read out of a trace of the configuration it describes. Where the two disagree, this one is the
> measurement. `step-breakdown.md` keeps its model-free sections (§1, §5–§8), which are unaffected.
>
> Production configuration **8** differs from this arm by one image (`62f53e6`, the `had_in`
> commit) and moves no speed number outside its band, so this breakdown is read as production 8's
> as well — with the caveat that the `had_in` row should now be a little smaller than the 5.57 %
> printed here `[not tested]`.

---

## 0. Method, and the two traps in it

The engine emits one `record_function` per engine step; kineto projects it onto the GPU timeline as a
`gpu_user_annotation` named `execute_context_N(T)_generation_M(T)` — the sequence and token counts
are inside the name. Two things will mislead anyone reading such a trace for the first time:

1. **In decode the annotations arrive in overlapping pairs**, two per step (192 GPU annotations for
   96 steps). Taken raw, the step count doubles and the trace appears to show a 50 % GPU bubble that
   is not there. They have to be merged.
2. **The drafter runs outside the annotation.** The merged region covers the *target* model's forward
   pass only; the rest of the step (12.0 ms at C1, 15.3 ms at C8, 17.5 ms on a prefill chunk) is
   DFlash2. That is an exact target-versus-draft split for free, and it replaces the "the span of the
   MoE GEMM calls is the target" heuristic the earlier reconciliation used.

So a step is defined with no holes and every kernel counted exactly once:

```
step_k       = [ merged_k.start , merged_{k+1}.start )
  target part  = [ merged_k.start , merged_k.end )
  draft  part  = [ merged_k.end   , merged_{k+1}.start )
```

**GPU busy is the union over all streams** (main, the second dense-GEMM stream, and the D2H stream),
so class times can sum past the wall while the union cannot. Wall is always `busy(union) + gap`.

**The profiler's own cost was measured**, by running the same windows with the profiler off:

| window | profiler off | profiler on | overhead |
|---|---|---|---|
| prefill 8.0–8.5K, `max_tokens=1` | 1,770–1,791 tok/s | 1,771 tok/s | **≈ 0 %** |
| decode C1, ms per step | 92.11 / 93.17 | 94.65 | **+2.5 %** |
| decode C8, ms per step | 213.50 / 216.39 | 216.52 | **+1.3 %** |

Every share below therefore carries to a profiler-free engine within ±2.5 %. The absolute idle
figures do **not** — see §6.

Windows: (a) three warm-ups outside the window, then a single **unseen 8,497-token prompt** at
`max_tokens=1`, six chunks of which four are steady; (b) C1, a 9-second window after one request
reached steady decode, **93 steps**; (c) C8, a 14-second window after an 8-request ramp, **63 steps**.

---

## 1. The chunk is 1,792 tokens, not 2,048

The GPU annotations give the real split of the 8,497-token prefill:

```
1792 · 1792 · 1792 · 1792 · 1024 · 305   =  8497
```

**1,792 = 7 × 256 = 87.5 % of the batched-token budget.** With `--block-size 256` the scheduler
issues seven blocks of the eight-block budget and one block of the budget goes unused on every chunk.
The earlier reconciliation assumed a 2,032–2,048-token chunk; the real one is 12.5 % smaller, which
matters because the marginal MoE cost of 1.38 µs per token per layer was measured at M=2048, not at
M=1792. Whether this is a scheduler reservation or block alignment is **open**
([`../../docs/11-open-issues.md`](../../docs/11-open-issues.md) §2.5).

Per-token cost agrees with the older estimate anyway: 963.27 ms / 1,792 = **0.5375 ms per token**
against a re-derived 0.5458 (−1.5 %).

---

## 2. Prefill — one steady 1,792-token chunk, mean of 3 ranks

**wall 962.55 ms · GPU busy 954.4 ms · occupancy 99.2 % · launch gap 8.16 ms (0.85 %)**
target window 945.1 ms · DFlash2 draft window 17.5 ms

| class | ms/step | % of step | calls/step |
|---|---|---|---|
| **MoE trellis GEMM** (`exl3_gemm_m_kernel`) | **274.420** | **28.51 %** | 84 |
| &nbsp;&nbsp;— gate/up (`w13`) | 177.770 | 18.47 % | 42 |
| &nbsp;&nbsp;— down (`w2`) | 96.650 | 10.04 % | 42 |
| Dense BF16 GEMM (cutlass / nvjet) | 167.394 | 17.39 % | 457 |
| **NCCL collectives** (102 all-reduce + 3 all-gather) | **139.266** | **14.47 %** | 105 |
| Hyper-connection mixing (`mhc_*`, `hc_prenorm`) | 115.403 | 11.99 % | 275 |
| MLA attention | 79.831 | 8.29 % | 59 |
| KDA/GDN linear attention (triton chunked scans) | 77.862 | 8.09 % | 479 |
| MoE hadamard (`had_in` / `glu_had_in`) | 53.621 | 5.57 % | 84 |
| norm / elementwise / copy | 37.491 | 3.90 % | 1,040 |
| DSA indexer (`fp8_mqa_logits`, `topKPerRowPrefill`) | 5.102 | 0.53 % | 77 |
| memcpy / memset (DtoD + D2H) | 1.878 | 0.20 % | 94 |
| MoE align / route | 1.869 | 0.19 % | 126 |
| `_zero_kv_blocks` | **0.857** | **0.09 %** | 1 |
| sampling / spec bookkeeping | 0.093 | 0.01 % | 24 |
| **CPU gap (GPU idle)** | **8.162** | **0.85 %** | — |
| **TOTAL (wall)** | **962.55** | **100 %** | |

**16.2 ms of that (1.68 %) is the DFlash2 drafter**, which runs on prefill chunks too — dense GEMM
11.75, collectives 2.56, the rest 1.90. The earlier prefill table had no draft row at all.

---

## 3. Decode — C1 (verify batch M=8) and C8 (M=64)

C1: **wall 94.65 ms · busy 89.2 ms · occupancy 94.2 %** · target 82.6 ms · draft 12.0 ms (93 steps × 3 ranks)
C8: **wall 216.52 ms · busy 212.2 ms · occupancy 98.0 %** · target 201.2 ms · draft 15.3 ms (63 steps × 3 ranks)

| class | **C1** ms | C1 % | C1 target | C1 draft | **C8** ms | C8 % |
|---|---|---|---|---|---|---|
| **Dense BF16 GEMM** | **42.902** | **45.33 %** | 34.579 | **8.323** | 45.610 | 21.07 % |
| **MoE trellis GEMM** | 28.105 | 29.69 % | 28.105 | 0 | **111.708** | **51.59 %** |
| &nbsp;&nbsp;— gate/up (`w13`) | 18.638 | 19.69 % | | | 72.733 | 33.59 % |
| &nbsp;&nbsp;— down (`w2`) | 9.467 | 10.00 % | | | 38.975 | 18.00 % |
| **NCCL collectives** | 14.641 | 15.47 % | 12.892 | **1.750** | 25.258 | 11.67 % |
| **CPU gap (GPU idle)** | **5.450** | **5.76 %** | — | — | 4.323 | 2.00 % |
| KDA/GDN linear attention | 1.608 | 1.70 % | 1.607 | 0.002 | 17.315 | 8.00 % |
| Hyper-connection mixing | 2.082 | 2.20 % | 2.082 | 0 | 3.194 | 1.48 % |
| MoE hadamard | 0.859 | 0.91 % | 0.859 | 0 | 3.883 | 1.79 % |
| MLA attention | 0.643 | 0.68 % | 0.499 | 0.143 | 3.050 | 1.41 % |
| norm / elementwise / copy | 1.785 | 1.89 % | 1.305 | 0.480 | 2.938 | 1.36 % |
| MoE align / route | 0.620 | 0.65 % | 0.620 | 0 | 1.305 | 0.60 % |
| DSA indexer | 0.351 | 0.37 % | 0.351 | 0 | 0.566 | 0.26 % |
| sampling / spec bookkeeping | 0.064 | 0.07 % | 0.015 | 0.050 | 0.115 | 0.05 % |
| memcpy / memset | 0.049 | 0.05 % | 0.018 | 0.031 | 0.109 | 0.05 % |
| `_zero_kv_blocks` | 0.002 | 0.00 % | 0.002 | 0 | 0.022 | 0.01 % |
| **step wall** | **94.65** | | | | **216.52** | |

**The k=7 drafter costs 10.78 ms (11.4 %) at C1 and 14.09 ms (6.5 %) at C8**, its own 1.75 / 3.32 ms
of collectives included. The earlier reconciliation put it at 18.5 ms and 19.5 % of a C1 step — a
**1.7× overestimate** `[retracted]`.

**Why the classes sum past 100 %** (104.8 % at C1, 101.3 % at C8, 100.1 % at prefill): a second CUDA
stream runs part of the dense path concurrently — 5.5 ms/step at C1, 5.3 at C8 — so the sum of kernel
durations exceeds the busy union by 4.4 ms (C1) and 2.9 ms (C8). That overlap is compute/compute and
it is real work being hidden, not double counting.

---

## 4. Where the re-derived table was wrong

Prefill, earlier "now %" against the measured share:

| class | re-derived % | **measured %** | relative error | verdict |
|---|---|---|---|---|
| MoE trellis GEMM | 26.4 | **28.51** | +8.0 % | under |
| NCCL collectives | 16.5 | **14.47** | −12.3 % | the **bottom** of its own 14–17 % band |
| Dense BF16 GEMM | 16.2 | **17.39** | +7.3 % | under |
| Hyper-connection mixing | 11.7 | **11.99** | +2.5 % | right |
| MLA attention | 8.2 | **8.29** | +1.1 % | right |
| KDA linear attention | 7.5 | **8.09** | +7.9 % | under |
| MoE `had_in` / `glu_had_in` | 6.1 | **5.57** | −8.7 % | over |
| norm / elementwise | 3.7 | **3.90** | +5.4 % | right |
| **MoE `combine` / `build_inv`** | 1.5 | **0.00** | **−100 %** | **no such kernel in this build** — fused into the down-projection epilogue |
| **`_zero_kv_blocks`** | 1.3 | **0.089** | **−93 %** | **0.86 ms, not 14.7** — a 16× overestimate |
| DSA indexer | 0.6 | **0.53** | −12 % | right, and closed |
| MoE align / route | 0.2 | **0.19** | −5 % | right |
| — | (not modelled) | memcpy 0.20 · sampling 0.01 · CPU gap 0.85 · **draft 1.68** | | four classes the old table had no row for |

**The aggregate was good and two target rows were fiction.** Per-token cost was estimated at
0.5458 ms and measured at 0.5375 (−1.5 %); but `exl3_moe_combine` and `_zero_kv_blocks`, ranked #8
and #10 on the old target list, are worth 0 % and 0.09 %.

At C1 the same comparison: dense GEMM 44.8 % estimated against **45.33 %** measured (right), MoE
29.3 % against **29.69 %** (right), collectives 10–15 % against **15.47 %** (the top of the band, so
under), and the drafter 19.5 % against **11.38 %** (1.7× over). At C8 the older document offered only
"more than half the step" for the MoE stage, which is correct — **51.6 %** — beside a "~130 ms"
figure that is 16 % high.

---

## 5. Three things the trace settles

### 5.1 The collectives are 100 % exposed

| window | measured comm/compute overlap |
|---|---|
| prefill | **0.00 ms/step** |
| decode C1 | 0.014 ms/step |
| decode C8 | 0.012 ms/step |

All-reduce and compute share one CUDA stream, so **every microsecond of NCCL is on the critical
path**. Prefill runs 102 all-reduce (median 903 µs, p75 1,401 µs, max 7,520 µs) plus 3 all-gather per
chunk; C1 and C8 run the same counts at medians of 71 µs and 123 µs. Meanwhile the bandwidth lever is
nearly spent — each NIC sits behind PCIe Gen5 ×4 (≈15 GB/s), two per node (≈30 GB/s), and the
transport already measures 20–21 GB/s at ≥16 MB. **Overlap, not bandwidth, is the untouched lever.**

### 5.2 The trellis GEMM is at the memory roofline

Against a *measured* achievable read bandwidth of **225.2 GB/s** (vendor figure 273), per MoE layer
per rank, all 96 local experts certainly touched at this M:

| stage | µs/layer @1,792 tok | traffic (per-expert lower bound) | GB/s | % of 225 |
|---|---|---|---|---|
| `gemm_w13` | 4,232.6 | 805.3 MB | 190.3 | **84.5 %** |
| `gemm_w2` | 2,301.2 | 402.6 MB | 175.0 | **77.7 %** |

The kernel is not slow; it takes what the memory can deliver. The "traffic if per-block" alternative
that used to sit beside this table is **not a real alternative** — the expert re-read bench settled
it on both cards ([`step-breakdown.md`](step-breakdown.md) §6,
[`../../docs/10-results-and-roofline.md`](../../docs/10-results-and-roofline.md) §5.4) — so the
expert-stationary lever it implied does not exist and the column is withdrawn `[retracted]`.

### 5.3 A model-free MoE bench overstates small-batch cost by 1.5–1.7×

Same kernel, per MoE layer, per rank:

| M | model-free `gemm_w13` | **in-engine measured** | ratio |
|---|---|---|---|
| 8 (C1) | 662 µs | **443.8 µs** | **1.49× faster in the engine** |
| 64 (C8) | 2,975 µs | **1,731.7 µs** | **1.72× faster** |
| 1,792–2,048 (prefill) | 4,404 µs @2048 | **4,232.6 µs @1792** | equal |

The model-free bench routes uniform-random top-8, which **maximises** the number of distinct experts
a batch touches. Real routing is clustered, so at small M far fewer experts — and far less trellis —
are read. At large M all 96 local experts are touched either way and the two agree.
**Do not carry a small-M MoE cost over from a uniform-random bench.** The same trap, one level down,
caught a synthetic MLA measurement whose 200k-row pool fitted L2 and reported a bandwidth above the
DRAM ruler.

---

## 6. CUDA graphs are off, the log says why, and the idle is smaller than it looks

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend
FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
setting cudagraph_mode=NONE
Skipping CUDA graph capture.
```

`enforce_eager=False`, compilation mode `NONE`, CUDAGraph memory 0.0 GiB. So graphs are off **not**
because of `--enforce-eager` but because spec-decode plus the FlashInfer attention backend cannot
capture an 8-token-per-sequence verify batch — and the reason the drafter is on FlashInfer at all is
the fp8 draft cache (in bf16 it ran on FlashAttention, which declares `UNIFORM_BATCH`, and the boot
log then read `Graph capturing finished`).

**Then subtract the profiler.** GPU busy union is 89.17 ms against a profiler-off wall of 92.64 ms,
so **~2.0 ms of the 5.45 ms "idle" is CUPTI itself** — 1.97 ms over 1,873 kernel boundaries ≈
**1.05 µs per boundary**, which is the known per-kernel cost of the instrument. C8: 1.50 ms over
1,800 boundaries ≈ 0.83 µs.

| window | profiled wall | busy (union) | profiled idle | unprofiled wall | **corrected idle** |
|---|---|---|---|---|---|
| decode C1 | 94.611 | 89.167 | 5.444 | **92.64** | **3.477 (3.75 %)** |
| decode C8 | 216.448 | 212.127 | 4.321 | **214.95** | **2.828 (1.31 %)** |

This is an inference, not a direct measurement — it assumes CUPTI does not slow the kernels
themselves and that all of its cost lands on the boundaries. `[measured-here]` for the two walls,
`[estimate]` for the split.

**"CPU gap" was the wrong name for it.** Matching every gap to the launch of the kernel on its right
(via `correlation`, and including `cuda_driver` launches — triton kernels go out that way, and
filtering them mislabels the step's two largest gaps as device waits):

| cause | C1 ms/step | % of idle | gaps/step | mean |
|---|---|---|---|---|
| **device-side dispatch bubble, target** | **3.328** | 61.1 % | 1,485 | 2.2 µs |
| **device-side dispatch bubble, draft** | 0.858 | 15.8 % | 345 | 2.5 µs |
| host-bound launch gap ≥ 20 µs (`prepare_inputs`) | 0.553 | 10.2 % | 9.4 | 59 µs |
| host-bound launch gap < 20 µs (same region) | 0.285 | 5.2 % | 27.1 | 10.5 µs |
| **blocking sync / pageable D2H on the critical path** | 0.281 | 5.2 % | 1.0 | 281 µs |
| DFlash2 draft orchestration | 0.082 | 1.5 % | 2.0 | 41 µs |
| NCCL launch / proxy stall | 0.038 | 0.7 % | 0.9 | 42 µs |
| rest | 0.020 | 0.4 % | 1.2 | — |
| **TOTAL** | **5.444** | 100 % | 1,873 | |

So **77 % of the idle is per-kernel dispatch** — the step launches 2,332 kernels and pays ~2.3 µs at
80 % of their boundaries — **18 % is the host** and **5 % is one blocking sync**. The host is not
behind: it runs **3.9 ms ahead** of the GPU at C1 and 24.8 ms ahead at C8. Classic launch-overhead
reasoning does not apply; the per-boundary dispatch cost does.

By phase (C1, 3 ranks, profiler-corrected in the last column):

| phase | phase ms | idle ms | occupancy | gaps | **corrected idle** |
|---|---|---|---|---|---|
| **P1 head: scheduler handoff + `prepare_inputs`** | 1.075 | 0.949 | **11.7 %** | 89 | **0.856** |
| P2 target forward body | 81.526 | 3.261 | 96.0 % | 1,435 | 1.754 |
| P3 logits + sample + accept | 2.959 | 0.055 | 98.1 % | 21 | 0.033 |
| **P4 draft loop (DFlash2 k=7)** | 9.051 | 1.178 | **87.0 %** | 328 | **0.834** |
| **TOTAL** | **94.611** | **5.444** | 94.25 % | 1,873 | **3.477** |

There are only **two gaps per step of ≥ 0.1 ms** (one at C8), together 0.48 ms; the rest is 1,871
micro-holes. Half of the idle sits in front of glue kernels — norm / elementwise / copy account for
**50.9 %** of it across 717 boundaries — and this build runs `compilation mode NONE`, i.e. with
torch.compile entirely off, so nothing is fused.

**What graph capture would actually be worth**, taken against the corrected budget: P2 at 1.22 µs per
boundary, realistically 60–75 % removed, plus a captured draft loop — **1.4–1.9 ms/step = 1.5–2.1 %
of C1**, and **0.5–0.7 % at C8**. It removes neither P1 (`prepare_inputs` is host code producing the
graph's inputs, outside any captured region) nor the blocking sync. That is why the previous
production configuration, which *did* capture graphs (bf16 draft KV → FlashAttention), read the same
57 tok/s single-stream as this one, which does not: the difference is inside the noise of a tok/s
reading. **An earlier claim of "+6 % single-stream from graph coverage" is retracted; the whole idle
budget is 3.75 % and graphs are at most two-thirds of it** `[retracted]`.

The draft's own collectives, measured: **11 all-reduce + 3 all-gather per step**, 133 µs each against
the target's 146 µs — latency-bound, not size-bound. Batching them is not available (they are eleven
sequentially dependent layers) and replicating the drafter costs more than it saves (its dense GEMM
is already 8.32 ms at C1). Overlap is the only lever, and overlap is measured at 0.014 ms/step.

---

## 7. Rank asymmetry in the collectives is arrival skew, not wire time

Step wall is identical across ranks (prefill 963.3 / 962.6 / 961.7 ms). Time spent *inside*
collectives is not:

| window | rank A | rank B | rank C | spread |
|---|---|---|---|---|
| prefill (1,792 tok) | 136.02 ms | **145.08 ms** | 136.69 ms | 6.5 % |
| decode C1 | 15.04 | 15.44 | **13.45** | 13.6 % |
| decode C8 | 24.61 | **26.96** | 24.20 | 10.9 % |

At prefill the rank that waits longest is the one doing **5.9 ms less MoE GEMM**, 1.6 ms less dense
GEMM and 1.3 ms less MoE hadamard: it finishes early and blocks at the barrier. So the true wire time
is the **minimum** over ranks, ≈136.0 ms, and **6.5 % of the collective class (≈0.9 % of the step) is
expert-parallel load imbalance** — closed by expert placement, not by the plugin or the kernel.

---

## 8. Ranked targets, measured

| # | target | prefill | C1 | C8 | in `cuda-exl3`? | note |
|---|---|---|---|---|---|---|
| 1 | MoE trellis GEMM | 28.5 % | 29.7 % | **51.6 %** | yes | 78–85 % of the measured roofline; the duplicate-read lever is closed |
| 2 | Dense BF16 GEMM — Ampere-class `cutlass_80_*` on sm_121 | 17.4 % | **45.3 %** | 21.1 % | no | 79 % of shape-matched achievable TFLOP/s — **and see §9** |
| 3 | NCCL — **overlap** with compute (currently exactly 0) | 14.5 % | 15.5 % | 11.7 % | no | bandwidth is at the PCIe wall; overlap is untouched |
| 4 | Hyper-connection mixing — fuse the passes | 12.0 % | 2.2 % | 1.5 % | no | kernels already at 91–97 % of read roofline |
| 5 | CUDA-graph coverage for the spec-decode verify batch | 0.9 % | 5.8 % raw / **3.75 % real** | 2.0 % | no | worth 1.5–2.1 % of C1, not 6 % (§6) |
| 6 | MLA prefill | 8.3 % | 0.7 % | 1.4 % | yes | **closed** — the kernel author measured 86–89 % of achievable at our shapes |
| 7 | `exl3_moe_had_in` / `glu_had_in` | 5.6 % | 0.9 % | 1.8 % | yes | taken upstream in `a47da6e`, bounded in `62f53e6` |
| 8 | DSA indexer | 0.53 % | 0.37 % | 0.26 % | — | **closed**, third independent confirmation |
| 9 | `_zero_kv_blocks` · `exl3_moe_combine` | 0.09 % / 0 % | ~0 | ~0 | — | **closed** — one is 16× smaller than costed, the other does not exist |

---

## 9. The one number that dwarfs the rest, and it is not a kernel

Dense BF16 GEMM is **45.3 % of a C1 step** because the checkpoint's
`scope: glm53_routed_experts_only` leaves attention, the shared experts and `lm_head` unquantized: at
M=8 that stage streams 16-bit weights while the routed half streams 4-bit. The kernel author's
arithmetic on our numbers: the stage is weight-bandwidth-bound at M=8, so 4 bpw instead of 16 is ~4×
less traffic — **42.9 ms → ~11 ms, about 32 ms off a 94.65 ms step, roughly +34 % single-stream**
`[estimate]`. The entire `cuda-exl3` column of the table above comes to about 5 %.

The quality gate decides it, not the arithmetic, and the experiment is written up in
[`../../docs/11-open-issues.md`](../../docs/11-open-issues.md) §2.22.

---

## 10. Tools

| Tool | What it does |
|---|---|
| `bench/prof-run7.py` | opens and closes the three windows on a live engine; reads the in-window step count from `vllm:spec_decode_num_drafts_total` |
| `bench/baseline7.py` | the same windows with the profiler off — the overhead measurement in §0 |
| `bench/prof-analyze7.py` | streaming trace reader: annotation merging, target/draft split, class taxonomy. Reads a 1 GB decode trace in a few hundred MB of RAM |
| `bench/prof-gap7.py` | its sibling for §6: stream union, gap detection, launch matching by `correlation` (**including `cuda_driver`**), innermost `cpu_op`, phase segmentation |

Traces are not in this repository (large binaries, and the engine logs beside them carry host names);
the extracted tables are this file.

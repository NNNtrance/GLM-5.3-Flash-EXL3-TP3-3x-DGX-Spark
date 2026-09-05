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

Hardware limits, measured on one node with the engine stopped, inside our image `[measured-here]`
(these come from the NVFP4 sibling's measurement on the same hardware): memory copy 230–246 GB/s
read+write, **read-only 243 GB/s**; BF16 matmul **97 TFLOPS**; FP8 `_scaled_mm` **199 TFLOPS** at
8192².

Byte model, from the checkpoint's own shapes `[estimate]`: 45 layers of which 43 carry routed experts
(3–45; layer 45 is the MTP layer and carries its own 288); 288 experts, top-8, one shared expert;
hidden 4096, routed intermediate 2048. A routed expert is 3 × 4096 × 2048 = 25.2M parameters, about
**12.9 MB** at 4 bpw with its scales. Under expert parallelism each node holds 96 experts of 288, so
43 × 96 × 12.9 MB ≈ 49.6 GiB of expert weight per node — and the measured figure is 54.86 GiB, which
leaves about **5.3 GiB of BF16 non-expert weight per node** (attention, KDA, the shared expert,
`lm_head`). That reconciliation is the reason to trust the rest of the table.

With a k=7 draft a decode step verifies 8 tokens per sequence, so the expected number of distinct
experts touched per layer is `288 × (1 − e^(−8·tokens/288))`.

| Regime | Measured | Bytes per step per node `[estimate]` | Effective bandwidth | Share of 243 GB/s | Share of 97 TFLOPS (BF16) |
|---|---|---|---|---|---|
| Decode C1, DFlash2 k=7 | 63.6 tok/s per stream, 5.49 tok/step → 86.3 ms/step | 8 verify tokens → ~19 of 96 experts per layer: 10.6 GB expert + 5.3 GB dense ≈ **15.9 GB** | ≈ **184 GB/s** | ≈ **76 %** | ≈ 2 % |
| Decode C8, DFlash2 k=7 | 168.9 tok/s total, 42.3 accepted tok/step → 250 ms/step | 64 verify tokens → ~80 of 96 experts per layer: 44.3 GB + 5.3 GB ≈ **49.6 GB** | ≈ **198 GB/s** | ≈ **82 %** | ≈ 5 % |
| Prefill, 2048-token chunk | 1,792 tok/s → 1.14 s per chunk | every expert touched: 53.3 GB + 5.3 GB ≈ **58.6 GB** | ≈ **51 GB/s** | ≈ **21 %** | ≈ **24 %** (≈ 24 TFLOPS/node) |

**Reading.** Decode with the draft already sits at three quarters to four fifths of the memory roof, so
faster kernels buy little there — the levers are fewer bytes (a checkpoint that also quantizes
attention) or higher acceptance. **Prefill is far from both roofs**, at a fifth of bandwidth and
under a quarter of compute, and that is where kernel work can still pay.

The profiler agrees with the second half of that and refines the first. On a fresh 8,273-token
prefill, GPU busy was 98.8 % of the window — so prefill is not launch-bound or CPU-bound, and any win
must come out of a kernel — with the MoE EXL3 GEMM at 26.5 %, BF16 dense GEMMs at ~20 %, and NCCL
all-reduce at 21.9 % `[measured-here]`. **That profile predates three separate collective changes**
(the channel cap, the second cable and `NCCL_PTR_CUDA`) and the all-reduce share is certainly lower
now; re-profiling it is the cheapest unspent measurement in this repository
([06](06-nccl-mesh.md) §12 item 5). At decode the BF16 dense GEMMs are the **largest single
item at ~37 %**, ahead of the EXL3 MoE GEMM at 29.3 %.

That last number is the most important structural fact in this repository: **the biggest cost at
decode is the part of the model that is not quantized**, and no work inside the EXL3 kernel library
can reach it ([01](01-model-and-license.md) §3.2).

Assumptions in the table, and they are not small: KV reads are ignored (short contexts), the draft
model's own weights are ignored, expert-touch counts are expectations rather than measured
histograms, and a 4 bpw expert is taken as 12.9 MB. **Treat every ratio as ±15 %.** No
`nsys`/`ncu` run backs the byte model; what backs it is that the derived per-node weight total lands
within 1 % of the measured one.

---

## 5. What we did not measure

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

---

## 6. What is next

[11 — Open issues](11-open-issues.md): what is unresolved, what we retracted, and what we never ran.

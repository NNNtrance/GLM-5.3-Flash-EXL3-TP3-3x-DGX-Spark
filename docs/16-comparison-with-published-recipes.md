# 16 — Comparison with other published recipes

**Applies to: both tracks.** §3 is two nodes, §4 is three, and §4.4 is four.

A dozen public recipes serve GLM-5.3-Flash as an EXL3 checkpoint on DGX Spark hardware. This page puts
what they publish beside what we measured, at the matching node count, **without touching either
side's numbers**.

**The rule this page is written under.** Every figure in someone else's column is quoted exactly as
that repository publishes it, with the conditions they state, and is tiered `[reported]` — this
repository's tier for a number someone else measured ([STYLE-GUIDE.md](../STYLE-GUIDE.md)). We do not
re-derive, rescale, average or "correct" a published number, and where their conditions differ from
ours the difference is written into the table rather than compared across silently. Our own figures
are `[measured-here]`; their settings are in [10](10-results-and-roofline.md) and
[15](15-tp2-track.md).

**Read §1 before any table.** Most of these recipes lead with a *synthetic* decode figure and every
one of ours is *realistic*. Those are not the same measurement, and putting them in one row would be
the single easiest way to mislead someone about this hardware
([09](09-measurement-protocol.md) §6).

**The most useful thing on this page is not a ranking.** It is §3.4 and §4.2: two other people
independently quantized the dense path of this model, one at two nodes and one at three, by two
different routes, and both measured a gain in the same band as ours. Three stacks agreeing is worth
more than any single column.

---

## 1. How to read this page

### 1.1 Four things have to match before two numbers mean anything

| Axis | Why it moves a number |
|---|---|
| **Node count and TP** | The largest single factor here. Three ranks give 1.5× the memory of two and cut each rank's weight traffic by a third; on our own stack that is worth 7× the KV pool ([15](15-tp2-track.md) §4) |
| **Checkpoint scope** | A `routed_experts_only` checkpoint leaves the attention stack, shared expert and `lm_head` in BF16; a **full-scope** one quantizes them too. On our stack that difference alone is **+22.9 % at C1** ([13](13-full-scope-checkpoint.md) §7.3), and two other recipes now measure the same lever independently (§3.4, §4.2) |
| **Prompt type** | Synthetic ("count from 1 to 200") measures the speculative-decoding ceiling. On this model family it runs at roughly **1.7× the realistic single-stream rate** `[measured-here]` ([09](09-measurement-protocol.md) §6) — so a synthetic number and a realistic number are not comparable, in either direction |
| **What the KV pool number is a function of** | `gpu-memory-utilization`, `max_model_len`, `max_num_seqs`, `--max-num-batched-tokens`, whether the vision tower is loaded, and the drafter's page geometry. Every one of those differs somewhere on this page |

### 1.2 Engine paths, which is the deepest difference of all

Two distinct software stacks appear below, and they share only the checkpoint format's name.

- **The ExLlamaV3 path.** Most of the recipes here run vLLM with an overlay that registers an `exl3`
  method backed by `exllamav3_ext` (ExLlamaV3 pin `c5d9c657`, 0.0.43), a fused `exl3_moe` launch per
  layer, and `FLASHINFER_MLA_SPARSE_SM120` for attention `[reported]`. Several of them descend from
  the same public GHCR image.
- **The `cuda-exl3` path.** This repository runs the same base image with
  [`Zeuss5/cuda-exl3`](https://github.com/Zeuss5/cuda-exl3) kernels and
  `--attention-backend CUSTOM` ([02](02-image-build.md), [CREDITS](../CREDITS.md)). Nothing else on
  this page uses it.

They are different kernels, different attention backends and different loaders. A difference between
our column and theirs is at least as likely to be that as anything else.

---

## 2. The recipes

Everything here serves GLM-5.3-Flash as EXL3 on DGX Spark or ASUS Ascent GX10 (GB10) hardware.
Commits and dates are as of 5 September 2026.

| Repository | Nodes / TP | Checkpoint scope | Drafter | Last commit |
|---|---|---|---|---|
| [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | 2 / TP=2 | routed experts only, 4 bpw | DFlash2 k=7 | `3021f24c`, 2026-09-05 |
| [`Reederey87/glm53-flash-exl3-2x-dgx-spark`](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) | 2 / TP=2 | routed experts only, 4 bpw | DFlash2 k=7, drafter revision pinned | 2026-09-05 |
| [`Entrpi/glm-5.3-flash-exl3-2x-spark`](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark) | 2 / TP=2 | routed experts only, 4 bpw | DFlash2 k=7, MXFP8 draft by default | 2026-09-02 |
| [`tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark) | 2 / TP=2 | routed experts only, 4 bpw | DFlash2 k=7 | 2026-09-02 |
| [`Alexbob0/glm53-flash-dense-exl3-tp2`](https://github.com/Alexbob0/glm53-flash-dense-exl3-tp2) | 2 / TP=2 | **dense quantized** — routed experts 4 bpw plus an EXL3 overlay on attention, shared experts, dense MLP and `lm_head` | DFlash2 k=7, drafter itself quantized to EXL3 5 bpw | 2026-09-05 |
| [`Alexbob0/glm53-flash-vllm-upstream-sm121`](https://github.com/Alexbob0/glm53-flash-vllm-upstream-sm121) | 2 / TP=2 | the same dense-quantized pack, on **stock upstream vLLM nightly** | DFlash2 k=7 | 2026-09-05 |
| [`abliter8-ai/glm-5.3-flash-exl3-prod`](https://github.com/abliter8-ai/glm-5.3-flash-exl3-prod) | 2 / TP=2 | routed experts only, 4 bpw, plus a runtime `o_proj` transplant | DFlash2 k=7 | 2026-09-02 |
| [`Enntity/sparkglm`](https://github.com/Enntity/sparkglm) | 2 / TP=2 | routed experts only, 4 bpw | DFlash2 k=7 | 2026-09-05 |
| [`UnsignedChad/glm53-flash-ablit-exl3-2x-dgx-spark`](https://github.com/UnsignedChad/glm53-flash-ablit-exl3-2x-dgx-spark) | 2 / TP=2 (intended) | a generic full-model 4 bpw pack, `mul1` codebook | — | 2026-08-29 |
| [`FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks`](https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks) | 3 / TP=3 + EP | routed experts only, 4 bpw | DFlash2 k=7 | `9093765c`, 2026-08-29 — one commit |
| [`jakejharris/jspark3`](https://github.com/jakejharris/jspark3) | 3 / TP=3 + EP | routed experts 4 bpw **plus an INT8 W8A16 Marlin overlay** on 169 dense trunk modules | DFlash2 k=7 | 2026-09-03, release `v1.0.0` |
| [`punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark`](https://github.com/punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark) | 4 / TP=4 | routed experts only, 4 bpw | DFlash2, **shipped off** — see §4.4 | 2026-09-05 |
| **this repository**, configurations 1–8 | 3 / TP=3 + EP | routed experts only — `brandonmusic/GLM-5.3-Flash-tr3-4bpw` revision `b20c49ba` | DFlash2 k=7 | — |
| **this repository**, production 9 and 10 | 3 / TP=3 + EP | **full scope** — `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw | DFlash2 k=7, draft KV fp8 | — |

Two more, named because a reader searching for "GLM-5.3-Flash 3× DGX Spark" will find them and should
know what they are: `FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks` is a **NVFP4** three-node recipe, not
EXL3, and its own README points at the EXL3 sibling in the table above; and
`UnsignedChad/glm53-flash-ablit-exl3-2x-dgx-spark` publishes **no numbers at all** — its launcher
refuses to start, by design, because its `mul1`-codebook pack is incompatible with the `mcg`,
routed-experts-only ABI of the runtime it targets, and its README says throughput and quality
benchmarks are pending `[reported]`.

**There is no three-node recipe from `MiaAI-Lab`.** We looked on GitHub and on Hugging Face and did
not find one: that account's GLM-5.3-Flash EXL3 work is the two-node repository above, its Hugging
Face presence for this model is `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (a weights mirror with no
serving results of its own), and the only larger-than-two arrangement in the repository is an
**experimental, explicitly untested** `start-tp4.sh` for four nodes: *"Untested here (no 4-Spark
kit)"* `[reported]`. The three-node EXL3 recipes are FlyCockpit's and jakejharris'.

---

## 3. Two nodes

### 3.1 What each two-node recipe publishes

Every figure `[reported]`, with the conditions each repository states. **Read the prompt-type column
before the numbers** — a "structured" or "counting" figure is synthetic.

| Repository | Date of the figures | Single-stream decode | Prefill | KV pool | Quality |
|---|---|---|---|---|---|
| `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` | 2026-08-28 / 08-30 / 09-01 | ×1 **62.9** tok/s, **synthetic** (count 1→200), TTFT 719 ms; lab harness structured **65.1**, prose **27.1** | **1132.32** tok/s at ~8K fully uncached (PR77 kernel, MNBT 2048); legacy 941.04 | **1,754,237** at 1M, util 0.87; latest validated **1,243,902** at MNBT 7168 | KLD 0.024555 nats (independent panel, scores the weights) |
| `Reederey87/glm53-flash-exl3-2x-dgx-spark` | 2026-08-31 / 09-02 | structured **~67**, synthetic, at acceptance **1.0000**, explicitly labelled "acceptance/quality gate, not headline throughput"; prose **~28–31** at a 1M window | **~1001** tok/s median cold at 240K (passes 906–1072); 854 at 500K on an older image | **1,396,551** | none published; publishes prefix-cache retention (100 % at 2×68k, 98.7 % at 4×60k) instead |
| `Entrpi/glm-5.3-flash-exl3-2x-spark` | 2026-09-02 | median of five 400-token runs: structured **71**, JSON **51**, math **45**, code **42**, prose **30**; TTFT 0.3–0.4 s | **1,490** tok/s at 133K, **1,277** at 499K | **1,287,194** at a 524K cap; **2,144,814** at native 1M | math_500 **91 %** (n=100), GPQA-diamond **70 %** (n=50), 133K retrieval 10/10 |
| `tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark` | 2026-09-01 | prose c1 **19.1** (18.3–19.5), code c1 **48.6** (41.9–50.2); counting c1 61.8 | **1,752** tok/s on a fresh 211,001-token prompt | **1,396,551** at 1,048,576 | 40 real prompts, three runs, T0: **79–87 %** |
| `Alexbob0/glm53-flash-dense-exl3-tp2` | 2026-09-04 | see §3.4 — structured **78.7–79.8**, prose **32.2–33.4**, code **58.7** | ~870–1010 tok/s at ~8.3K client TTFT; ~1066 at ~33K | **1.237M** at the end of their progression; a separate indexer right-sizing arm takes their 1.24M baseline to **1.53M** | top-1 agreement **97.6 %** and median truncated KL **0.00103** nats against a BF16-dense control |
| `Alexbob0/glm53-flash-vllm-upstream-sm121` | 2026-09-05 | structured **80.0**, prose **33.3**, code **49.6** | **1,013 / 1,080 / 1,165** tok/s at 8K / 32K / 100K | 20.1 GiB of KV at 1M, util 0.87 | KL 0.00030 median and 97.6 % top-1 against its own reference |
| `abliter8-ai/glm-5.3-flash-exl3-prod` | 2026-09-02 | structured **64.7–65.4**, synthetic, 400 tokens; prose 30.3 | **1131 / 1235 / 1175** tok/s at 10K / 125K / 378K | **693,227** at a **640,000** cap (their note: 640K is the engine ceiling on this pair, "the 1M claim did not hold") | LiveCodeBench hard tier 7/8 then 3/3; needle exact 8K→612K | 
| `Enntity/sparkglm` | 2026-09-03 | four salted ~16K prompts, 400 tokens each: aggregate **24.267091** tok/s, wall 86.148650 s | published as relative deltas only | not published | not published |
| **ours, TP=2 arm C** `[measured-here]` | 2026-09-05 | **54.69** per stream / **47.40** aggregate, **realistic** (12 short English code prompts) | **1,334** tok/s on three fresh unseen 8,204-token prompts | **665,625** at 1M, util 0.85, `MAX_NUM_SEQS` 8 | MMLU sample 1,995 q **86.4 ±0.7**; probe 10/10; code exam 12/12 |

`abliter8-ai` publishes the one figure nobody else on this page does — **aggregate decode at high
concurrency on two nodes: ~310 tok/s at twelve concurrent structured streams**, with the knee at
twelve and ±15 tok/s of run-to-run noise `[reported]`. It is synthetic, and it is at `MAX_NUM_SEQS`
well above our 8, so it does not sit beside our C8 of 133.57; it is here because it is the only
published two-node ceiling of that shape.

### 3.2 What the two-node table does and does not say

**On single-stream decode, every synthetic figure above sits between 62.9 and 80.0 and every
realistic prose figure sits between 19.1 and 33.4.** That spread is what §1.1's prompt-type row means
in practice, and it is wider than any difference between the stacks. Our own realistic single-stream
figure, 54.69 per stream on twelve short code prompts, is not comparable to either end of it: our
prompt set is code, not prose and not counting. The nearest published thing to a like-for-like row is
`Entrpi`'s **code 42** and `tonyd2wild`'s **code c1 48.6** against our code category at TP=3 of
**61.7**, and those are three different code prompt sets on three different stacks at two different
node counts.

**On the KV pool, our two-node figure is the smallest on the page, and the reason is ours.** See
§5.2: we never ran our own draft-page fix at two ranks.

**On quality, four instruments appear and none of them is on the same scale as another** — a KLD
panel over the weights, MMLU, math_500 plus GPQA, and a 40-prompt subjective pass rate. There is no
honest way to rank the stacks by quality from this page.

### 3.3 Beside our two-node arms, in detail

`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` is the closest match to our arm C — same checkpoint
lineage, same drafter, same node count, same tensor-sliced arrangement:

| | `MiaAI-Lab/…-EXL3-2x-DGX-Sparks` `[reported]` | ours, TP=2 arm C `[measured-here]` | Conditions that differ |
|---|---|---|---|
| Date | 2026-08-28 / 08-30 / 09-01, per row | 2026-09-05 | |
| Engine | vLLM + ExLlamaV3 `exl3_moe`, FlashInfer sparse MLA SM120 | vLLM + `cuda-exl3` `62f53e6`, `--attention-backend CUSTOM` | different kernels and attention backend |
| Checkpoint | routed experts only, 4 bpw — their mirror of `brandonmusic/GLM-5.3-Flash-tr3-4bpw` snapshot `5ab363a8…` | routed experts only, 4 bpw — the same upstream repository at revision **`b20c49ba`** | same repository, **different snapshots**; we have not diffed them |
| Drafter | DFlash2 k=7, draft TP=2, draft KV bf16 | DFlash2 k=7, draft TP inherited, draft KV bf16 | |
| Single stream | **62.9** aggregate at ×1, **synthetic**; lab prose **27.1** | **54.69** per stream / **47.40** aggregate, **realistic** code prompts | synthetic against realistic — not comparable |
| Concurrency | 103.3 at ×2, 146.5 at ×4, synthetic | 68.03 at C2, 90.66 at C4, 133.57 at C8, realistic | as above; also `MAX_NUM_SEQS` 4 against our 8 |
| Prefill | 1132.32 tok/s at ~8K fully uncached | **1,334** tok/s on three fresh unseen 8,204-token prompts | both cold; different prompt sets, different kernels |
| KV pool at 1M | 1,754,237 at util 0.87 · 1,243,902 at MNBT 7168 | **665,625** at util 0.85, MNBT 2048 | §5.2 — this gap has a known mechanism and it is ours |
| Quality | KLD 0.024555 nats (scores the weights) | MMLU sample **86.4 ±0.7**; probe 10/10; code exam 12/12 | different instruments, no common scale |
| Vision | on | off (`--language-model-only`) | ours frees the vision tower's memory |
| Boot | not published as a single figure | 396 s cold, no fast-load sidecar | |

We also ran a **full-scope** checkpoint at two ranks — 68.00 tok/s per stream, MMLU 86.32 ±0.75 — and
it is not in that table because at two ranks that arm's KV pool falls to 31,343 tokens and its
long-prompt path closes entirely ([13](13-full-scope-checkpoint.md) §6.1). It is a measurement rig,
not a serving configuration, and it should not be read beside anyone's production numbers. The
recipe that *did* make a dense-quantized two-node arm serviceable is the next section.

### 3.4 Someone else quantized the dense path at two nodes, and got the same answer we did

`Alexbob0/glm53-flash-dense-exl3-tp2` is the most directly relevant repository on this page. It
reaches the same conclusion this repository reached in [13](13-full-scope-checkpoint.md) — that the
BF16 dense path is the largest single item in a decode step, and quantizing it is worth roughly a
fifth to a quarter of single-stream throughput — by a **different route**: instead of loading
`turboderp/GLM-5.3-Flash-exl3` whole, it overlays the dense tensors from that same 4.05 bpw package
onto the routed-experts-only checkpoint, then quantizes the DFlash2 drafter and `lm_head` as well
`[reported]`.

Their progression, streaming, temperature 0, TTFT excluded, median of three, thinking off, two boots,
2026-09-04 `[reported]`:

| probe | baseline | + dense | + EXL3 draft | + EXL3 `lm_head` | Δ |
|---|---|---|---|---|---|
| structured (count 1→200) | 65.0 | 72.6–73.0 | 76.2–76.3 | **78.7–79.8** | **+22 %** |
| prose (hash-map, English) | 26.2 | 30.7–32.1 | 31.2–32.9 | **32.2–33.4** | **+26 %** |
| code (English, BST) | — | — | 55.3 | **58.7** | — |

Their KV pool goes 1.24M → 1.03M → 1.07M → 1.12M → 1.152M → **1.237M**, i.e. back to where it started
by the end; weight load is 79.8 GiB per rank. Their quality gate is a teacher-forced top-20 logprob
comparison against a BF16-dense control over 455 positions: **97.6 %** top-1 agreement, **0.00103**
nats median truncated KL, and a trimmed mean of 0.0344 nats which they note matches the checkpoint
author's published band for the same 4.05 bpw pack `[reported]`.

**Three things we take from that, and one caution.**

- **Their +22 % / +26 % and our +24.3 % per stream are the same finding on two engines.** Ours is
  `cuda-exl3` with a whole full-scope checkpoint at TP=2 ([13](13-full-scope-checkpoint.md) §4.1);
  theirs is ExLlamaV3 with a dense overlay. Two stacks, two loaders, one answer.
- **Their step anatomy agrees with our profile in shape.** They report a torch-profiler window of
  about 93 ms per step: routed experts `exl3_moe` ~48 %, still-BF16 dense cutlass ~26 %, dense EXL3
  ~8 %, NCCL ~8 % `[reported]`. Our production-9 C1 profile reads MoE trellis GEMM 32.5 %, NCCL
  26.1 %, dense EXL3 GEMM 15.0 %, remaining BF16 linears 10.3 % `[measured-here]`
  ([10](10-results-and-roofline.md) §5). The rank order of the top item is the same and the residual
  BF16 term is the same story; the NCCL share is not, and neither is the node count.
- **They kept k=7 for the same reason we did.** Their k=5 arm reads structured 58.9–59.1 against
  k=7's 72.6–73.0, and prose 32.3–34.6 against 30.7–32.1 — prose prefers the shallower draft and
  everything else does not `[reported]`. Ours is the same trade
  ([10](10-results-and-roofline.md) §4).
- **The caution: their draft acceptance is reported two ways in the same repository** — "unchanged
  (~50–68 %)" in one place and 52–55 % against a BF16-draft control's 52–58 % in another. We are not
  reconciling that; we are noting that we did not, and that our own acceptance figure survived a
  retraction of exactly this kind ([11](11-open-issues.md) §1.9).

`Alexbob0/glm53-flash-vllm-upstream-sm121` is the same author's companion result and is worth its own
line, because it answers a question this repository does not: the same dense-quantized pack on
**stock upstream vLLM nightly**, nine Python overlay patches and one extension build, no vLLM C++
rebuilt — structured **80.0**, prose **33.3**, code **49.6**, prefill 1,013 / 1,080 / 1,165 tok/s at
8K / 32K / 100K, 20.1 GiB of KV at 1M and util 0.87 `[reported]`.

---

## 4. Three nodes

### 4.1 `FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks`, as they publish it

Their conditions: three DGX Spark nodes, TP=3 with expert parallelism, the `MiaAI-Lab` GHCR image,
head padding 64 → 66 with the SM120 decode kernel padding Q/out 22 → 32, vocabulary
`padding_size = lcm(64,3) = 192`, the BF16 shared expert padded 2,048 → 2,112 and TP-sharded, routed
`moe_intermediate_size` **unpadded** at 2,048 under EP, DFlash2 k=7 with draft TP=1, CUDA graphs on,
`NCCL_NET=Mesh` with `NCCL_PROTO=LL`, `max_model_len` 1,000,000, vision on, temperature 0, thinking
**off**, three runs, 2026-08-29 `[reported]`:

| Phase | tok/s | TTFT |
|---|---|---|
| hello | 37.9 / 36.9 / 37.3 | ~0.35 s |
| structured (count 1–200) | 69.0 / 68.5 / 71.2 | ~0.36 s |
| code (`is_prime`) | 52.3 / 58.7 / 58.2 | ~0.37 s |

Draft acceptance **81.5 %** (1,175 accepted / 1,442 draft tokens) on the counting prompt. KV pool
**4,657,200** at `GPU_MEMORY_UTILIZATION` 0.87 — which they record as producing about 6 GiB of
container swap — and **3,858,012** at 0.83, which is their live default. An eager arm gives 4,888,438
tokens and 66.5 / 66.5 / 66.5 structured, so they keep CUDA graphs. They publish no prefill-throughput
figure, no aggregate-at-concurrency figure and no MMLU. The repository has **one commit** and its own
sweep plan lists items T2 through T8 as not yet run.

### 4.2 `jakejharris/jspark3` — the dense path again, at three nodes, in INT8

The second three-node EXL3 recipe, and the second independent measurement of the dense-path lever. It
runs TP=3 with expert parallelism on the same routed-experts-only EXL3 checkpoint, and adds a
**selective INT8 W8A16 Marlin overlay** over 169 dense trunk modules and 225 tensors — not EXL3, and
excluding the 34 KDA gating modules — freeing 1,595,392,320 bytes per rank `[reported]`. Their
headline, temperature 0, thinking off, 400 max tokens, medians of three, 1M context, on their own
fleet `[reported]`:

| probe | jspark3 v1 |
|---|---|
| structured (synthetic) | **81.962** tok/s |
| code | **66.257** tok/s |
| prose | **29.049** tok/s |

Their matched control — the same three-node fleet with the overlay off, same day — reads code
61.768, structured 76.863, prose 26.810, so the overlay is worth **+7.27 % code, +6.63 % structured,
+8.35 % prose** `[reported]`. They also publish a matched **loss**: C3 per stream **−21.20 %**
(65.208 → 51.382) and long prefill **−3.38 %** on a 113,908-token prompt (1277.443 → 1234.246 tok/s),
both disclosed rather than omitted, and two internal gates missed by 1.11 % and by a pacing count.

**This is a genuinely different point in the design space from ours, and the comparison is worth
making carefully.** They quantize the dense trunk to INT8 W8A16 and keep the KDA gating arms out of
it; we serve a checkpoint whose dense path is 5–6 bit EXL3 and whose KDA gating arms stay BF16
([13](13-full-scope-checkpoint.md) §4.4). Their gain over their own control is 6.6–8.4 %; ours over
production 8 is **+22.9 % at C1** `[measured-here]`. We are not claiming that difference means INT8 is
worse: their control already carries a different image, a different kernel project and a fused-MoE
path we do not run, and the two overlays cover different module sets. What both measurements agree on
is the **direction and the fact that the dense path is where the money is at a single stream**, which
is the conclusion [13](13-full-scope-checkpoint.md) was written to establish.

**One note of theirs concerns the recipe in §4.1 rather than ours**, and we pass it on as theirs
rather than adopting it: they report a pinned-source audit finding that the FlyCockpit configuration's
loader ignores its declared draft TP of 1 and builds the padded draft over world TP 3, and that its
measured configuration ran at `gpu_memory_utilization` 0.87 where its own source records about 6 GiB
of cgroup swap `[reported]`. **We have not verified either claim** and neither affects any number in
this repository.

### 4.3 Beside our three-node production

| | `FlyCockpit/…-3x-DGX-Sparks` `[reported]` | `jakejharris/jspark3` `[reported]` | ours, production 10 `[measured-here]` |
|---|---|---|---|
| Date | 2026-08-29 | 2026-09-03 | 2026-09-05 |
| Engine | vLLM + ExLlamaV3 via the `MiaAI-Lab` image | the same image, adapted to TP=3 + EP=3 at container start | vLLM + `cuda-exl3` `754421f`, `--attention-backend CUSTOM` |
| Checkpoint | routed experts only, 4 bpw | routed experts 4 bpw + INT8 W8A16 on the dense trunk | **full scope**, `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw |
| Padding constants | vocab `lcm(64,3) = 192`, shared expert 2,112 | as FlyCockpit's lineage | vocab `lcm(128,3) = 384`, shared expert 2,304 — **both correct, see §5.1** |
| Synthetic single stream | 69.0 / 68.5 / 71.2 | **81.962** | not published — we do not publish synthetic figures |
| Realistic single stream | code 52.3 / 58.7 / 58.2 | code **66.257** · prose **29.049** | **70.5** aggregate / **76.9** per stream on 12 short code prompts; code category **61.7** · prose **29.1** |
| Aggregate at concurrency | not published | C4 **251.13** (62.80 per stream), synthetic | **194.0** at C8, realistic |
| Prefill | not published | 1,234.246 tok/s on a 113,908-token prompt | **1,769** tok/s, three fresh unseen ~8K prompts |
| KV pool at 1M | 4,657,200 at util 0.87 (~6 GiB container swap) · **3,858,012** at 0.83 | not published | **5,619,834** at util **0.83** |
| Draft acceptance | 81.5 %, counting prompt | 64.3–65.9 % across matched concurrency waves | 61.9–62.6 %, realistic; 46 % on our code category |
| Quality | not published | not published as a benchmark score | MMLU sample **86.47 ±0.74**; probe 10/10; code exam 12/12, cold and warm |
| Boot | not published | not published | 251 s container-start to API-ready; 315 s from power-on |
| Fabric | `NCCL_NET=Mesh`, `NCCL_PROTO=LL` | as its lineage | mesh plugin, `NCCL_MAX_NCHANNELS=8`, both cables per peer, `NCCL_PTR_CUDA`; we measured `NCCL_PROTO` and rejected it ([06](06-nccl-mesh.md) §12.1) |

**The one row where two realistic figures land on top of each other is prose: their 29.049 against our
29.1.** Different engines, different checkpoints, different prose prompts, three nodes each — and the
same number to three significant figures. Two readings agreeing by coincidence is possible; we note it
because it is the only place on this page where a cross-recipe comparison is even close to
like-for-like, and because prose is the workload where the k=7 drafter helps least on every stack
here (§5.4).

### 4.4 Beyond three nodes

`punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark` runs the same routed-experts-only checkpoint at TP=4
on four nodes and publishes a tuning campaign rather than a single figure `[reported]`: single-stream
prose 32–37 tok/s, aggregate 99 / 132 / 131 tok/s at 4 / 8 / 16 concurrent, a 282K cold prompt at
1,324 tok/s, and — their words — **TP=4 gained 29 % over TP=2 on a 282K prompt** (1,162 against 901
tok/s) with decode up 45 %. Their campaign winner is `GPU_MEM_UTIL` 0.75 with a k=3 draft and
`MAX_NUM_SEQS` 8.

**It also publishes the most useful negative result on this page**, and it is the kind of thing this
repository exists to record. A 150-minute soak with 96K prompts in the mix **hung the engine three
times out of three** (78 / 31 / 35 minutes); all four ranks stall at the same forward pass during a
96K chunked prefill sharing steps with six or seven speculative streams, and vLLM eventually dies on
`RPC call to sample_tokens timed out`. Turning the fat-expert kernel off did not help; turning DFlash2
off made the identical soak pass 150 minutes and 472 requests with zero errors — but a 282K cold
prefill alone, with the draft already off, **still stalled** at 218,624 tokens computed. The recipe
therefore ships with the drafter **off** by default, at 22–23 tok/s instead of 32–37, plus a watchdog
`[reported]`.

**We have not seen this on three nodes** `[not tested]`, and we should say exactly how weak that is
as evidence: our longest published run is a benchmark sweep, not a soak, our `--max-num-seqs` is 8,
and we have never mixed a 96K prefill into a live speculative decode batch and left it running for two
hours. Their configuration differs from ours in node count, image, kernel project, drafter depth and
scheduler policy. It is on the list of things a second cluster could settle
([CONTRIBUTING](../CONTRIBUTING.md)), and it is the one item on this page we would most like to know
about our own stack.

---

## 5. What the differences are, and what they are not

### 5.1 The padding constants are not a disagreement

`FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks` pads the vocabulary with `lcm(64, 3) = 192` and the
shared expert to 2,112. This repository uses 384 and 2,304 and
[03](03-tp3-padding-and-sidecars.md) §1.1 calls 2,112 "silently wrong". **Both are right, for
different checkpoints.** On a routed-experts-only checkpoint the vocabulary and the shared expert are
native BF16, so zero-extending them is a true no-op and 64 is the correct unit — which is what they
serve. On our production checkpoint both tensors are EXL3, a trellis cannot be zero-extended, and the
pad has to occupy whole 128-column Hadamard blocks per rank, which moves the unit to 128. Our own
configurations 1–8 used 192 and 2,112 and were correct doing so. Anyone porting a *dense-quantized*
pack onto the FlyCockpit lineage at TP=3 will need the 128 unit, and that is the practical content of
this paragraph rather than a criticism.

### 5.2 The two-node KV gap is our missing fix, not their extra one

Our TP=2 pool of 665,625 tokens is the smallest two-node figure on this page — against 693,227 at a
640K cap, 1,237,000 (1,530,000 with indexer right-sizing), 1,243,902–1,754,237, 1,287,194–2,144,814
and 1,396,551 elsewhere `[reported]`. Utilization rungs and `max_num_seqs` explain part of it and the
mechanism explains the rest, and we understand the mechanism well because we hit it and fixed it at
three ranks. A DFlash2
drafter allocated on its own small page consumes a disproportionate share of the engine's
**per-request block counter**, which is what actually caps the pool ([07](07-kv-and-draft-page.md)
§3). We solved it with `HAREM_SW_BLOCK_SIZE=256`, worth **+82 % of pool** at TP=3; the `MiaAI-Lab`
recipe solved it differently, with what they call a padded slot-share of the drafter's SWA layers onto
the MLA tensors (`block_size=64`, `page_size_padded` equal to the MLA page) so that drafter layer *i*
co-owns MLA tensor *i* `[reported]`. **We never ran our fix at two ranks**
([15](15-tp2-track.md) §3.3), so our two-node pool figure is an un-fixed one and several of theirs are
not.

### 5.3 The utilization rungs are not the same rung

Published rungs on this page run 0.75, 0.80, 0.83, 0.845, 0.85 and 0.87, with two recipes recording
swap at 0.87, one campaign selecting **0.75** as its winner on a four-node fleet, and one recipe
crash-looping at 0.87 and settling on 0.85 `[reported]`. Our production 10 is at 0.83 and our own 0.85
arm was **rejected** on a swap reading — 1.6 GB of swap growth under load on the head node — with 0.85
marked as not to be attempted again on this stack ([11](11-open-issues.md) §2.4,
[00](00-hardware-and-os.md) §11). Recipes reaching different verdicts about the same knob on the same
part is expected rather than contradictory: the number that matters is host memory free at that rung,
and that depends on node count, on what else the node is doing, on whether the vision tower is loaded,
and on `max_model_len`.

### 5.4 Synthetic against realistic, once more, because it is the biggest number on the page

Most of these recipes lead with a counting prompt. On this model family a synthetic counting prompt
runs at roughly **1.7×** the realistic single-stream rate `[measured-here]`, because the draft model
predicts a counting sequence almost perfectly and predicts real code about half the time
([09](09-measurement-protocol.md) §6). The evidence is on both sides of the fence and it agrees:
`FlyCockpit` publish **81.5 %** acceptance on their counting prompt against our 46 % on realistic code
and 13 % on prose; the `MiaAI-Lab` lab harness publishes **0.959** structured acceptance against
**0.341** on prose, and structured 65.1 tok/s against prose 27.1; `Reederey87` publish a structured
acceptance of **1.0000** and label that arm an acceptance gate rather than a throughput headline —
which is the right way to publish it. **We are not converting anyone's number.** We are saying that a
synthetic row and a realistic row must not be read as one measurement, in either direction, and that a
reader who takes a counting figure as what their agent will feel is going to be disappointed by any of
these stacks.

### 5.5 What a like-for-like comparison would need, and nobody has run

Same checkpoint scope, same prompt set, same `max_num_seqs`, same utilization rung, same vision
setting, same number of boots. It does not exist for any pair on this page. The nearest thing to it
would be running the `hizset-v2` prompt set in [`scripts/`](../scripts/) against another recipe's
server, which is one HTTP endpoint away and which we have not done because we do not have their
hardware idle. `jakejharris/jspark3` did the reverse — they re-ran two published recipes on their own
fleet, which is the honest form of a cross-recipe claim and is expensive
([CONTRIBUTING](../CONTRIBUTING.md) says how to send us one if you do it).

---

## 6. What we took from them

**From `FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks`: the three-node arithmetic, and no files.** The
64 → 66 head padding with 22 heads per rank; the vocabulary `padding_size` of `lcm(64, tp)`; padding
the BF16 shared expert and slicing it rather than replicating it; the drafter's 32/8 → 36/9 pad; the
conditional-shard idea; and their written-up finding that replicating the shared expert under expert
parallelism triples its contribution and produces a model that stays fluent and gets the answers
wrong. Each of those was re-derived from this checkpoint's own shapes and is checked at load time
rather than trusted ([03](03-tp3-padding-and-sidecars.md) §5). It is credited in
[CREDITS.md](../CREDITS.md) and its MIT licence is recorded in [LICENSES.md](../LICENSES.md). **We
vendor no files from it.**

**From `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`: nothing.** No code, no configuration, no number
and no technique. Our stack reaches the same checkpoint format through a different kernel project
(`Zeuss5/cuda-exl3`), a different attention backend and a loader we wrote, and it is not built on
their image. The repository is named on this page as the source of its own published figures and for
no other reason, which is why it is not in [CREDITS.md](../CREDITS.md).

**From everything else on this page: nothing, and every one of them postdates or parallels the work it
resembles.** `Alexbob0`'s dense-EXL3 result (§3.4) and `jakejharris`' INT8 overlay (§4.2) were found
while writing this page, after production 9 shipped; they are corroboration, not sources, and nothing
of theirs is in this repository. If any of them wants a credit for something we did take without
realising it, open an issue and we will fix it in the same day.

**Two upstreams are common to nearly every recipe here and are credited to their authors rather than
to any recipe**: the checkpoint `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, which the `Mia-AiLab` Hugging
Face repository re-hosts as a byte-identical mirror of one snapshot and which we pin directly at
revision `b20c49ba`; and the drafter `incoai/GLM-5.3-Flash-DFlash2`. Both are in
[CREDITS.md](../CREDITS.md) with their licences, one of which is not a licence most readers will have
seen before ([01](01-model-and-license.md)).

---

## 7. What we could not verify

- **Nothing on this page was reproduced by us.** Every `[reported]` figure is read off the other
  repository's own documents. We did not run their images, their harnesses or their prompt sets, and
  we did not audit their code.
- **Their harnesses are not ours.** "Stream tok/s" and "aggregate tok/s" are defined by each
  repository's own bench script, and we did not read those scripts closely enough to assert that
  their definitions match `scripts/bench-sweep.py`. Ours are defined in
  [09](09-measurement-protocol.md). Where a repository publishes two different values for the same
  quantity in two of its own documents, we quote both and reconcile neither (§3.4).
- **Two claims here are third-party claims about a third party**: the audit note in §4.2 about the
  FlyCockpit configuration's effective draft TP, and the cross-repository comparison tables several
  of these recipes publish about each other. Those are theirs, not ours, and we did not check them.
- **We could not find a three-node recipe from `MiaAI-Lab`** on GitHub or on Hugging Face (§2). An
  absence is weaker evidence than a presence; if one exists somewhere we did not look, this page is
  wrong about it and we would like a pull request.
- **Two repositories matching our search were excluded and are not named.** Their README files
  advertise a downloadable Windows executable for a machine specification that cannot hold this model,
  and their technical documents are copies of other repositories' files. They contain no independent
  measurements, so there is nothing of theirs to compare. We did not download anything from either,
  and we are not making an accusation — we are recording that a keyword search for this recipe returns
  pages that are not recipes, so that a reader who finds them knows why they are absent here.
- **Boot-to-boot spread is not visible in most of the published figures**, ours included where a row
  says "three runs". On our stack C1 boot medians span 1.1 %, C8 2.5 % and C4 **7.4 %**
  ([10](10-results-and-roofline.md) §1.1). Read every cross-recipe difference under about 10 % as
  unresolved.
- **The `FlyCockpit` repository has not changed since 2026-08-29** — one commit, and its own sweep
  plan lists T2 through T8 as not run. Its numbers are first-serve numbers by its own account, and
  comparing them against seven days of our work flatters us for reasons that have nothing to do with
  either stack.

# 16 — Comparison with other published recipes

Several public recipes serve GLM-5.3-Flash as an EXL3 checkpoint on DGX Spark hardware. This page puts
what they publish beside what we measured, at the matching node count, **without touching either
side's numbers**.

**The rule this page is written under.** Every figure in someone else's column is quoted exactly as
that repository publishes it, with the conditions they state, and is tiered `[reported]` — this
repository's tier for a number someone else measured ([STYLE-GUIDE.md](../STYLE-GUIDE.md)). We do not
re-derive, rescale, average or "correct" a published number, and where their conditions differ from
ours the difference is written into the table rather than compared across silently. Our own figures
are `[measured-here]` and their settings are in [10](10-results-and-roofline.md) and
[15](15-tp2-track.md).

**Read §1 before any table.** Two of the recipes here publish their headline decode figures on a
*synthetic* prompt and ours are *realistic*. Those columns are not the same measurement and putting
them in one row would be the single easiest way to mislead someone about this hardware
([09](09-measurement-protocol.md) §6).

---

## 1. How to read this page

### 1.1 Four things have to match before two numbers mean anything

| Axis | Why it moves a number |
|---|---|
| **Node count and TP** | The largest single factor here. Three ranks give 1.5× the memory of two and cut each rank's weight traffic by a third; on our own stack that is worth 7× the KV pool ([15](15-tp2-track.md) §4) |
| **Checkpoint scope** | A `routed_experts_only` checkpoint leaves the attention stack, shared expert and `lm_head` in BF16; a **full-scope** one quantizes them too. On our stack that difference alone is **+22.9 % at C1** ([13](13-full-scope-checkpoint.md) §7.3) |
| **Prompt type** | Synthetic ("count from 1 to 200") measures the speculative-decoding ceiling. On this model family it runs at roughly **1.7× the realistic single-stream rate** `[measured-here]` ([09](09-measurement-protocol.md) §6) — so a synthetic number and a realistic number are not comparable, in either direction |
| **What the KV pool number is a function of** | `gpu-memory-utilization`, `max_model_len`, `max_num_seqs`, `--max-num-batched-tokens`, whether the vision tower is loaded, and the drafter's page geometry. Every one of those differs somewhere on this page |

### 1.2 Engine paths, which is the deepest difference of all

Two distinct software stacks appear below, and they share only the checkpoint format's name.

- **The ExLlamaV3 path.** `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` and, through its image,
  `FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks` run vLLM with an overlay that registers an `exl3`
  method backed by `exllamav3_ext` (ExLlamaV3 pin `c5d9c657`, 0.0.43), a fused `exl3_moe` launch per
  layer, and `FLASHINFER_MLA_SPARSE_SM120` for attention `[reported]`.
- **The `cuda-exl3` path.** This repository runs the same base image with
  [`Zeuss5/cuda-exl3`](https://github.com/Zeuss5/cuda-exl3) kernels and
  `--attention-backend CUSTOM` ([02](02-image-build.md), [CREDITS](../CREDITS.md)).

They are different kernels, different attention backends and different loaders. A difference between
our column and theirs is at least as likely to be that as anything else.

---

## 2. The recipes

| Repository | Nodes / TP | Checkpoint | Drafter | Last commit at the time of writing |
|---|---|---|---|---|
| [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | 2 / TP=2 | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` — their byte-identical mirror of `brandonmusic/GLM-5.3-Flash-tr3-4bpw` snapshot `5ab363a8…`; uniform-K4 EXL3/TR3 **routed experts only**, 4 bpw, ~164 GiB, 120 shards. "Dense / shared / attn / embed / lm_head: native (unquantized)" | DFlash2 k=7 (`incoai/GLM-5.3-Flash-DFlash2`), draft KV `auto`/bf16, `DFLASH_DRAFT_TP=2` | `3021f24c`, 2026-09-05 |
| [`FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks`](https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks) | 3 / TP=3 + EP | the same checkpoint, through the same GHCR image | DFlash2 k=7, `draft_tensor_parallel_size=1`, drafter GQA padded 32/8 → 36/9 | `9093765c`, 2026-08-29 — one commit, unchanged since |
| **this repository**, configurations 1–8 | 3 / TP=3 + EP | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` revision `b20c49ba` — **routed experts only** | DFlash2 k=7 | — |
| **this repository**, production 9 and 10 | 3 / TP=3 + EP | `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw — **full scope** | DFlash2 k=7, draft KV fp8 | — |

**There is no three-node recipe from `MiaAI-Lab`.** We looked for one on GitHub and on Hugging Face
and did not find it: that account's GLM-5.3-Flash EXL3 work is the two-node repository above, its
Hugging Face presence for this model is `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (a weights mirror with
no serving results of its own), and the only larger-than-two arrangement in the repository is an
**experimental, explicitly untested** `start-tp4.sh` for four nodes: *"Untested here (no 4-Spark
kit)"* `[reported]`. The three-node EXL3 recipe built on their image and overlay is FlyCockpit's, and
it is credited in [CREDITS.md](../CREDITS.md).

---

## 3. Two nodes

### 3.1 `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, as they publish it

Their conditions, in their words: sparkDash decode bench, DFlash2 k=7, **Structured** (count 1→200)
and **Code** (`clamp_00`…`clamp_49`), temperature 0, thinking **off**, 400 tokens, CUDA graphs, fused
EXL3 MoE, `--max-model-len 1000000` with a 1,754,237-token KV pool, warm/empty KV, 2026-08-28
`[reported]`:

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---|---|---|
| ×1 | 719 ms | 62.9 | 62.9 |
| ×2 | 6.62 s | 51.7 | 103.3 |
| ×4 | 6.30 s | 37.1 | 146.5 |

Their own lab harness on the same protocol (`tests/bench_decode.py`, median of 5 × 400, 2026-08-30,
`DFLASH_DRAFT_TP=2`) `[reported]`: Structured **65.1** tok/s (0.959 accept / 6.71 per step); Prose
(hash-map) **27.1** (0.341 / 2.39); prior TP=1 lab 61.7 / 26.9; long context / mixed (~60–100k KV)
24–27; MTP k=2 baseline ~24.6.

Prefill, PR77 (2026-09-01), fully uncached rungs, MNBT 2048, five unique-salt cold samples per rung
per boot `[reported]`:

| Fully uncached rung | Legacy mean tok/s | PR77 pooled mean tok/s | Gain |
|---|---|---|---|
| ~8K | 941.04 | 1132.32 | +20.33 % |
| ~100K | 1023.20 | 1241.71 | +21.36 % |
| ~300K | 995.05 | 1201.02 | +20.70 % |

KV pool, three published readings, each with its own boot conditions `[reported]`: **1,754,237**
tokens at 1M (2026-08-29, `GPU_MEM_UTIL` 0.87, 690 GPU blocks, 18.67 GiB available KV memory, padded
slot-share allocator); a later pre-E2 1M boot at **1,670,157** / 638 blocks; and the *latest validated*
**1,243,902 tokens / 1.24×** at 1M (2026-09-01, MNBT 7168, `MAX_NUM_SEQS=4`, E2 and indexer
rightsizing on).

Quality: they publish no MMLU. They publish an **independent teacher-logit KLD panel** attributed to
another party, five cold runs, 25 sealed windows, 51,175 positions, scoring *the weights* rather than
their GB10 overlay `[reported]`: EXL3 4bpw **0.024555** nats against official FP8 on the same stack
0.024629 and NVFP4 0.060535.

### 3.2 Beside our two-node arms

Our TP=2 numbers, their conditions and their dates are in [15](15-tp2-track.md) §3. The two columns
below are the closest match we have, and the "conditions that differ" column is the point of the
table, not a footnote.

| | `MiaAI-Lab/…-EXL3-2x-DGX-Sparks` `[reported]` | ours, TP=2 arm C `[measured-here]` | Conditions that differ |
|---|---|---|---|
| Date | 2026-08-28 / 08-30 / 09-01, per row | 2026-09-05 | |
| Engine | vLLM + ExLlamaV3 `exl3_moe`, FlashInfer sparse MLA SM120 | vLLM + `cuda-exl3` `62f53e6`, `--attention-backend CUSTOM` | different kernels and attention backend |
| Checkpoint | routed experts only, 4 bpw | routed experts only, 4 bpw (`b20c49ba`) | the same upstream quantization |
| Drafter | DFlash2 k=7, draft TP=2, draft KV bf16 | DFlash2 k=7, draft TP inherited, draft KV bf16 | |
| Single stream | **62.9** tok/s aggregate at ×1, **synthetic** (count 1→200) | **54.69** per stream / **47.40** aggregate, **realistic** (12 short English code prompts) | **synthetic against realistic — not comparable.** Their prose figure, 27.1, is the nearest realistic one they publish |
| Concurrency | 103.3 at ×2, 146.5 at ×4, synthetic | 68.03 at C2, 90.66 at C4, 133.57 at C8, realistic | as above; also `MAX_NUM_SEQS` 4 against our 8 |
| Prefill | 1132.32 tok/s at ~8K fully uncached (PR77, MNBT 2048) | **1,334** tok/s on three fresh unseen 8,204-token prompts; 1,135 on the 7k uncached prompt | both cold; different prompt sets and different kernels |
| KV pool at 1M | 1,754,237 at util 0.87 · 1,243,902 at MNBT 7168 | **665,625** at util 0.85, MNBT 2048, `MAX_NUM_SEQS` 8 | see §5.2 — this gap has a known mechanism and it is ours, not theirs |
| Context | 1M | 1M | |
| Quality | KLD 0.024555 nats (independent panel, scores the weights) | MMLU sample 1,995 q **86.4 ±0.7**; correctness probe 10/10; code exam 12/12 | different instruments, no common scale |
| Vision | on (`--limit-mm-per-prompt {image:4,video:1}`) | off (`--language-model-only`) | ours frees the vision tower's memory |
| Boot | not published as a single figure | 396 s cold, no fast-load sidecar | |

We also ran a **full-scope** checkpoint at two ranks — 68.00 tok/s per stream, MMLU 86.32 ±0.75 — and
it is not in the table because at two ranks that arm's KV pool falls to 31,343 tokens and its
long-prompt path closes entirely ([13](13-full-scope-checkpoint.md) §6.1). It is a measurement rig,
not a serving configuration, and it should not be read beside anyone's production numbers.

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
container swap — and **3,858,012** at 0.83, which is their live default. An eager arm (`T1`) gives
4,888,438 tokens and 66.5 / 66.5 / 66.5 structured, so they keep CUDA graphs. They publish no
prefill-throughput figure, no aggregate-at-concurrency figure and no MMLU.

### 4.2 Beside our three-node production

| | `FlyCockpit/…-EXL3-3x-DGX-Sparks` `[reported]` | ours, production 10 `[measured-here]` | Conditions that differ |
|---|---|---|---|
| Date | 2026-08-29 | 2026-09-05 | seven days, and most of this repository happened in them |
| Engine | vLLM + ExLlamaV3 `exl3_moe` via the `MiaAI-Lab` image | vLLM + `cuda-exl3` `754421f` | different kernels and attention backend |
| Checkpoint | routed experts only, 4 bpw | **full scope**, `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw | on our stack that difference alone is +22.9 % at C1 |
| Padding constants | vocab `lcm(64,3) = 192`, shared expert 2,112 | vocab `lcm(128,3) = 384`, shared expert 2,304 | **both are correct for their own checkpoint** — see §5.1 |
| Single stream | 69.0 / 68.5 / 71.2, **synthetic**; 52.3 / 58.7 / 58.2 on a code prompt | **70.5** aggregate / **76.9** per stream, **realistic** | synthetic against realistic — not comparable. Our realistic code-category figure is 61.7 tok/s |
| Aggregate at concurrency | not published | 194.0 tok/s at C8, realistic | |
| Prefill | not published | 1,769 tok/s, three fresh unseen ~8K prompts | |
| KV pool at 1M | 4,657,200 at util 0.87 (with ~6 GiB container swap) · 3,858,012 at 0.83 | **5,619,834** at util **0.83** | ours is at the lower rung; see §5.3 |
| Draft acceptance | 81.5 % on the counting prompt | 61.9–62.6 % on realistic prompts; our own realistic code category is 46 % | acceptance is a property of the prompt before it is a property of the stack |
| Quality | not published | MMLU sample 1,995 q **86.47 ±0.74**; probe 10/10; code exam 12/12, cold and warm | |
| Boot | not published | 251 s container-start to API-ready; 315 s from power-on by the wall clock | |
| Fabric | `NCCL_NET=Mesh`, `NCCL_PROTO=LL` | mesh plugin with `NCCL_MAX_NCHANNELS=8`, both cables per peer, `NCCL_PTR_CUDA` | we measured `NCCL_PROTO` and rejected it ([06](06-nccl-mesh.md) §12.1) |

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
configurations 1–8 used 192 and 2,112 and were correct doing so.

### 5.2 The two-node KV gap is our missing fix, not their extra one

Their two-node pool (1,754,237 at util 0.87, or 1,243,902 on the latest validated boot) is larger than
our TP=2 arm's (665,625 at util 0.85) by more than the utilization rung explains, and the mechanism is
one we understand well because we hit it and fixed it at three ranks. A DFlash2 drafter allocated on
its own small page consumes a disproportionate share of the engine's **per-request block counter**,
which is what actually caps the pool ([07](07-kv-and-draft-page.md) §3). We solved it with
`HAREM_SW_BLOCK_SIZE=256`, worth **+82 % of pool** at TP=3; they solved it differently, with what they
call a padded slot-share of the drafter's SWA layers onto the MLA tensors (`block_size=64`,
`page_size_padded` equal to the MLA page) so that drafter layer *i* co-owns MLA tensor *i*
`[reported]`. **We never ran our fix at two ranks** ([15](15-tp2-track.md) §3.3), so our two-node pool
figure is an un-fixed one and theirs is not. Their reading is also at a higher `GPU_MEM_UTIL`, with
`MAX_NUM_SEQS` 4 against our 8, and with the vision tower loaded rather than excluded.

### 5.3 The utilization rungs are not the same rung

They publish a "safe" pin at 0.83 and a "more context" pin at 0.87 which they record as swapping about
6 GiB inside the vLLM cgroup, and they advise against 0.89 `[reported]`. Our production 10 is at 0.83
and our own 0.85 arm was **rejected** on a swap reading — 1.6 GB of swap growth under load on the head
node — with 0.85 marked as not to be attempted again on this stack
([11](11-open-issues.md) §2.4, [00](00-hardware-and-os.md) §11). Two recipes reaching different
verdicts about the same knob on the same part is expected: the number that matters is host memory
free at that rung, and that depends on what else the node is doing, on whether the vision tower is
loaded, and on `max_model_len`.

### 5.4 Synthetic against realistic, once more, because it is the biggest number on the page

Both other recipes lead with a counting prompt. On this model family a synthetic counting prompt runs
at roughly **1.7×** the realistic single-stream rate `[measured-here]`, because the draft model
predicts a counting sequence almost perfectly and predicts real code about half the time
([09](09-measurement-protocol.md) §6). The evidence is on both sides of the fence and it agrees:
`FlyCockpit` publish 81.5 % acceptance on their counting prompt against our 46 % on realistic code and
13 % on prose; the `MiaAI-Lab` lab harness publishes 0.959 structured acceptance against 0.341 on
prose, and structured 65.1 tok/s against prose 27.1. **We are not converting anyone's number.** We are
saying that a synthetic row and a realistic row must not be read as one measurement, in either
direction, and that a reader who takes a counting figure as what their agent will feel is going to be
disappointed by any of these stacks.

### 5.5 What a like-for-like comparison would need, and nobody has run

Same checkpoint scope, same prompt set, same `max_num_seqs`, same utilization rung, same vision
setting, same number of boots. It does not exist for any pair on this page. The nearest thing to it
would be running the `hizset-v2` prompt set in [`scripts/`](../scripts/) against another recipe's
server, which is one HTTP endpoint away and which we have not done because we do not have their
hardware idle. [CONTRIBUTING](../CONTRIBUTING.md) says how to send one if you do.

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

**Two upstreams are common to all three recipes and are credited to their authors rather than to any
recipe**: the checkpoint `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, which the `Mia-AiLab` Hugging Face
repository re-hosts as a byte-identical mirror and which we pin directly at revision `b20c49ba`; and
the drafter `incoai/GLM-5.3-Flash-DFlash2`. Both are in [CREDITS.md](../CREDITS.md) with their
licences, one of which is not a licence most readers will have seen before
([01](01-model-and-license.md)).

---

## 7. What we could not verify

- **Nothing on this page was reproduced by us.** Every `[reported]` figure is read off the other
  repository's own documents. We did not run their images, their harnesses or their prompt sets.
- **Their harnesses are not ours.** "Stream tok/s" and "aggregate tok/s" are defined by each
  repository's own bench script, and we did not read those scripts closely enough to assert that
  their definitions match `scripts/bench-sweep.py`. Ours are defined in
  [09](09-measurement-protocol.md).
- **We could not find a three-node recipe from `MiaAI-Lab`** on GitHub or on Hugging Face (§2). An
  absence is weaker evidence than a presence; if one exists somewhere we did not look, this page is
  wrong about it and we would like a pull request.
- **The `FlyCockpit` repository has not changed since 2026-08-29** — one commit, and its own sweep
  plan lists items T2 through T8 as not yet run. Its numbers are therefore first-serve numbers by its
  own account, and comparing them against seven days of our work flatters us for reasons that have
  nothing to do with either stack.
- **Boot-to-boot spread is not visible in any of the published figures**, ours included where a row
  says "three runs". On our stack C1 boot medians span 1.1 %, C8 2.5 % and C4 **7.4 %**
  ([10](10-results-and-roofline.md) §1.1). Read every cross-recipe difference under about 10 % as
  unresolved.

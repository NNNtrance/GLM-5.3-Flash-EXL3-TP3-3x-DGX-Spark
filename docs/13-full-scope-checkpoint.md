# 13 — The full-scope checkpoint: three loader layers, and the dense stage measured

Our production checkpoint is `scope: glm53_routed_experts_only`, so attention, the KDA layers, the
shared experts and `lm_head` stream BF16 weights beside a 4-bit routed half. That dense stage is
**45.3 % of a single-stream decode step** ([10](10-results-and-roofline.md) §5.3) — the largest single
item this repository carries, and larger than everything in the `cuda-exl3` column of its target table
put together.

This page is what happened when we tried to remove it: a full-scope EXL3 checkpoint
(`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw), the three independent reasons it would not load, the
loader patch that fixed them, and the measurement at TP=2 — **+24.3 % per stream, +26.4 % aggregate at
a single stream, quality inside the error bar, and a context cost that makes TP=2 a measurement rig
rather than a serving configuration**.

**The reframing is the part worth reading even if you never load this checkpoint.** We had written the
scope of our checkpoint down as a quality decision by its publisher. It was not. Two lines in the
vLLM `glm5next` model file pin the whole attention stack to BF16 regardless of what the weights
contain, and they lock **72.8 %** of the dense traffic. Until those lines are conditional, no
checkpoint of any scope can put attention on EXL3 — so `routed_experts_only` was not a choice about
quality, it was the only thing that could load ([11](11-open-issues.md) §1.9 row 29).

---

## 1. The checkpoint

| | |
|---|---|
| Repository | `turboderp/GLM-5.3-Flash-exl3`, branch `4.05bpw` |
| Revision we ran | `2a30229e67012798ba9f0cd832bb78abf4c363d5` (short `2a30229e`, 28 August 2026) |
| Licence | **MIT** — the `LICENSE` file is the MIT text, "Copyright (c) 2026 Z.AI Co., Ltd", and the model card carries `license: mit` `[reported]`. More permissive than the checkpoint we serve in production ([01](01-model-and-license.md) §2): no attribution condition, no exclusion clause |
| Size on disk | **165.2 GB / 153.8 GiB** — 19 shards (~8.59 GB each), plus `mtp.safetensors` 3.79 GB, `quantization_config.json` 47.9 MB and a 16.0 MB index `[measured-here]` |
| Verified | `sha256` 23/23 against the repository's own LFS metadata, independently on both nodes `[measured-here]` |
| Format | exl3 v1.4.4, codebook `mul1`, `bits: 4.05`, `head_bits: 6`, `out_scales: always`, calibration 250 rows × 2,048 columns |
| Architecture | `Glm5NextForConditionalGeneration` (a vision tower is present; `--language-model-only` keeps it out) |
| Index | 148,046 tensors, of which **36,719 are `.trellis`**; `model.visual.*` (1,007) is quantized too; `mtp.safetensors` is **not** in the index, so vLLM never reads it |

**It is not smaller, and that surprised us.** The expectation going in was 55–60 GB. The routed
experts are already 4-bit in *both* checkpoints, so full scope only takes the remaining ~11 % of the
weights from BF16 down to 4–6 bits: 164 GiB → 154 GiB. **The gain is in decode traffic, not on disk**,
and the 10 GiB saved on disk turns out to be repaid several times over in host memory (§6.2).

### 1.1 What is quantized, and at what bitrate

Read from `quantization_config.json` (37,032 records) `[measured-here]`:

| module class | count | bits | codebook |
|---|---|---|---|
| `mlp.experts.*.{gate,up,down}_proj` (routed) | 12,096 × 3 | **4** | mul1 |
| `mlp.shared_experts.{gate,up,down}_proj` | 42 × 3 | 6 | mul1 |
| `mlp.{gate,up,down}_proj` (dense layers 0–2) | 3 × 3 | 5 | mul1 |
| `self_attn.qkv_proj` (KDA layers) | 34 | 6 | mul1 |
| `self_attn.o_proj` (every layer) | 45 | 6 | mul1 |
| `self_attn.{q_a,q_b,kv_a_proj_with_mqa}` (MLA) | 11 × 3 | 6 | mul1 |
| `self_attn.indexer.wq_b` | 11 | 6 | mul1 |
| **`lm_head`** | 1 | **6** | mul1 |
| stays BF16 | `embed_tokens`, every norm, `mlp.gate` (the router), KDA `b_proj`/`f_a_proj`/`f_b_proj`/`g_a_proj`/`g_b_proj`, `indexer.wk`/`weights_proj`, the three `conv1d` | — | — |

For comparison, our production checkpoint's `tensor_storage` holds **only** the 37,152 routed-expert
tensors (`bits 4`, codebook `mcg`, `head_bits 16`). The dense and attention path of the plugin had
therefore **never been exercised on this stack** before this arm.

### 1.2 What BF16 traffic there is to remove, and which layer removes it

Computed from the checkpoint's own tensor shapes `[measured-here]`. The total, 15.77 GB, agrees
independently with the ≈16.8 GiB the plugin's own `online.py` docstring quotes for this model — two
sources, one number.

| group | n | bits in this checkpoint | BF16 GB today | opened by |
|---|---|---|---|---|
| `self_attn.qkv_proj` (KDA) | 34 | 6 | **6.845** | **S3** |
| `self_attn.o_proj` | 45 | 6 | **3.758** | **S2** |
| `lm_head` | 1 | 6 | 1.269 | already resolvable |
| `shared_experts.{gate,up}_proj` | 84 | 6 | 1.410 | **S1** |
| `shared_experts.down_proj` | 42 | 6 | 0.705 | already resolvable |
| `self_attn.q_b_proj` | 11 | 6 | 0.554 | **S2** |
| dense MLP `{gate,up}_proj` | 6 | 5 | 0.604 | **S1** |
| dense MLP `down_proj` | 3 | 5 | 0.302 | already resolvable |
| `q_a_proj` + `kv_a_proj_with_mqa` | 22 | 6 | 0.184 | **S1 + S2** |
| `self_attn.indexer.wq_b` | 11 | 6 | 0.138 | **S2** |
| **total** | | | **15.768** | |

(`embed_tokens` adds 1.269 GB of BF16 that is gathered rather than read per token, and is not in the
lever.)

| arm | moved to EXL3 | dense traffic left | reduction |
|---|---|---|---|
| stock (does not boot) | 2.276 GB | 14.34 GB | 9.0 % |
| S1 + S2 | 8.924 GB (56.6 %) | 10.19 GB | 35.4 % |
| **S1 + S2 + S3 (what we ran)** | **15.768 GB (100 %)** | **5.92 GB** | **62.5 %** |

---

## 2. Why it did not load: three independent layers

Two boots died before any of this was understood, on `KeyError: layers.2.self_attn.o_proj.mul1` and
then on `KeyError: layers.1.self_attn.conv1d.weight`. The second one is the informative one:
`conv1d` is a **BF16** tensor, so the failure is not about quantization at all — a BF16 copy of the
same checkpoint would die in the same place.

The diagnosis was then done model-free, by instantiating the real vLLM model on
`torch.device("meta")` and reproducing `params_dict` analytically against the checkpoint's weight map
`[measured-here]`. "Unmapped" means a checkpoint tensor with no parameter to load into, which is a
`KeyError` at boot rather than a silent half-load.

| checkpoint tensor group | count | stock | + S1 | + S2 | + S3 |
|---|---|---|---|---|---|
| `mlp.shared_experts.gate_up_proj.*` | 168 | unmapped | **OK** | OK | OK |
| `mlp.gate_up_proj.*` (dense 0–2) | 12 | unmapped | **OK** | OK | OK |
| `self_attn.o_proj.*` | 180 | unmapped | unmapped | **OK** | OK |
| `self_attn.fused_qkv_a_proj.*` | 88 | unmapped | unmapped | **OK** | OK |
| `self_attn.q_b_proj.*` | 44 | unmapped | unmapped | **OK** | OK |
| `self_attn.indexer.wq_b.*` | 44 | unmapped | unmapped | **OK** | OK |
| `self_attn.qkv_proj.*` (KDA) | 136 | unmapped | unmapped | unmapped | **OK** |
| `self_attn.conv1d.weight` | 34 | unmapped | unmapped | unmapped | **OK** |
| **total unmapped** | | **886** | **526** | **170** | **0** |

**The same derivation gives 0 unmapped and 0 unfilled for the experts-only checkpoint**, which is the
check that it reproduces "loads fine" rather than merely producing a number.

### 2.1 S1 — the model class declares no `packed_modules_mapping`

vLLM merges several checkpoint tensors into one layer:

```text
.gate_up_proj      <- .gate_proj , .up_proj
.fused_qkv_a_proj  <- .q_a_proj  , .kv_a_proj_with_mqa
.wk_weights_proj   <- .wk        , .weights_proj
.in_proj_qkvbfg_a  <- .q_proj , .k_proj , .v_proj , .b_proj , .f_a_proj , .g_a_proj
```

`Exl3Config._candidate_names()` can invert that, but only through `self.packed_modules_mapping`,
which vLLM copies off the model class (`model_loader/utils.py:275,281`). `glm5next` declares none, so
the dictionary is `{}` and every merged linear fails to resolve. **This is invisible for a
routed-experts-only checkpoint** — those layers are BF16 anyway — **and fatal for a full-scope one.**

### 2.2 S2 — the attention stack is pinned to BF16 in the model file

This is the layer the first analysis missed, and it is the largest.

- **MLA:** `model.py:331` passes `quant_config=None` with the comment "MLA projections are BF16 in
  checkpoint", so 11 layers' `fused_qkv_a_proj`, `q_b_proj`, `kv_b_proj`, `o_proj` and `indexer.wq_b`
  get `UnquantizedLinearMethod`.
- **KDA:** `kda.py:171-174` sets `vllm_config.quant_config = None` for the duration of
  `super().__init__`, and the base class captures it (`mamba/gdn/base.py:41`), so
  `self.quant_config` stays `None` **permanently** and everything built after it —
  `in_proj_qkvbfg_a`, `f_b_proj`, `g_b_proj`, `o_proj` — is unquantized.

`Exl3Config.get_quant_method` is therefore **never called** for attention. An earlier note of ours
that "`o_proj` resolves 45/45" was correct and irrelevant: `resolve()` works, nothing asks it.
Together those two lines lock **72.8 % of the dense traffic**, and neither has anything to do with
packing. Both are legitimate for an fp8 checkpoint, where those projections really are BF16 — the
fix is to condition them on the quantization method, not to delete them.

### 2.3 S3 — the KDA block is factorised differently

The checkpoint serialises KDA attention the way the upstream Hugging Face release does; the NVIDIA
`glm5next` implementation expects a different factorisation.

| what `kda.py` builds | what this checkpoint has | what our production checkpoint has |
|---|---|---|
| `q_proj` / `k_proj` / `v_proj` as shards 0/1/2 of `in_proj_qkvbfg_a` | **one** EXL3 `self_attn.qkv_proj`, `[256, 1536, 96]` trellis = (4096/16, 24576/16, 16×6) | three separate BF16 tensors |
| `q_conv1d` / `k_conv1d` / `v_conv1d` | **one** `self_attn.conv1d.weight`, BF16 `[24576, 1, 4]` | three BF16 `[8192, 1, 4]` |
| `b_proj`, `f_a_proj`, `g_a_proj` as shards 3/4/5 | present, BF16 | present, BF16 |

So the producer of our production checkpoint did exactly two things upstream does not: split
`qkv_proj` three ways on the output dimension, and split `conv1d` three ways on dim 0. Pure slicing,
no re-quantization.

`packed_modules_mapping` cannot express either of them, and behind the naming sits a structural
problem: shards 0–2 of `in_proj_qkvbfg_a` would be EXL3 while shards 3–5 stay BF16, and
`Exl3LinearMethod` requires a linear to be **wholly** EXL3 or wholly BF16 (`resolve()` needs
`all(i is not None)`). A mixed-precision merged linear cannot be expressed today. The way out is to
split the module.

### 2.4 Rewriting the checkpoint offline instead — evaluated and rejected

The obvious alternative is to rewrite the 19 shards into the layout our production checkpoint uses
(pure name and slice rewriting, no re-quantization) so that no runtime patch is needed. It removes
**only S3**, and not even all of it:

- **S1 cannot be removed on disk.** `gate_proj` and `up_proj` were quantized separately, so their
  `suh` differ, and `suh` lives *inside* the Hadamard. They cannot be pre-fused; whatever the layout,
  the packed mapping is still required.
- **S2 cannot be removed on disk.** `quant_config=None` is the model's decision and has nothing to
  do with the checkpoint's names.
- **S3b cannot be removed on disk.** Even with `qkv_proj` pre-split, shards 0–2 are EXL3 and 3–5 are
  BF16, so `resolve()` still returns `None`. Splitting the module is unavoidable.

Against that: ~154 GiB read plus ~154 GiB written per node, on nodes at 95 % full; the loss of a
`sha256` 23/23 provenance match against the publisher; and a 165 GB re-download if it goes wrong.
**Rejected** — it is larger than the patch, not smaller, and it is one-way.

---

## 3. The patch

One file, `patch-fullscope-tp2.py`, in the same style as the rest of `patches/tp3/`: exact-text
anchors that must match **exactly once**, `py_compile` before the write, atomic replace, idempotent,
a post-check, and a hard exit if anything does not match. Two target files:
`vllm/models/glm5next/nvidia/model.py` and `.../kda.py`.

**`[patch file not yet in repo]`** — it lives on the nodes at `~/exl3-zeus/tp2/patch-fullscope-tp2.py`
(hard-linked into the TP=3 patch directory during the arm, then unlinked from there — §6.4). It will
be added under `patches/` when it is carried at TP=3.

| # | file | layer | what it does |
|---|---|---|---|
| A1 | `model.py` | S1 | module-level helpers plus `if <env>: packed_modules_mapping = …` in `Glm5NextForCausalLM` |
| A2 | `model.py` | S1 | the same conditional class attribute in `Glm5NextForConditionalGeneration` |
| A3 | `model.py` | S2 | MLA `quant_config=None` becomes conditional on the quant method being EXL3 |
| A4 | `model.py` | S3b | replaces the six `.in_proj_qkvbfg_a` entries in `stacked_params_mapping` with four new ones, then asserts there are four |
| A5 | `model.py` | S3a | the `conv1d` three-way split in `load_weights` (plus the `ReplicatedLinear` fix, §3.3) |
| A6 | `kda.py` | S2 | the unconditional `quant_config` strip becomes conditional |
| A7 | `kda.py` | S3b | splits `in_proj_qkvbfg_a` into an EXL3 `in_proj_qkv` and a BF16 `in_proj_bfg_a`; installs asserts 1 and 2 |
| A8 | `kda.py` | S3b | `forward` becomes two calls when split, the upstream single call otherwise |

**Everything is gated on one environment variable**, and with it unset the patched image is upstream
byte for byte. Measured rather than asserted: with the flag unset the class attribute is never
created (a bare `if` in a class body), so vLLM's `getattr(model_class, "packed_modules_mapping",
None)` returns `None`, both quant gates pass `None`, the KDA module is not split, and
`stacked_params_mapping` keeps its six upstream entries `[measured-here]`. The same patched image
serves the experts-only checkpoint with 0 unmapped and 0 unfilled.

One further safety property fell out of the design: **the split decision is taken from the
checkpoint, not from the environment** (`quant_config.resolve(f"{prefix}.in_proj_qkv") is not None`).
Booting the experts-only checkpoint with the flag **set** gives split inactive, 0 unmapped, 0
unfilled, everything BF16 `[measured-here]`. A wrong env file cannot silently corrupt a run.

### 3.1 The mapping, and the three things deliberately not in it

```json
{"gate_up_proj": ["gate_proj", "up_proj"],
 "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
 "in_proj_qkv": ["qkv_proj"]}
```

- **`lm_head` is absent on purpose.** It is a `VocabParallelEmbedding`: the trellis puts vocab on
  dim 1 where a plain linear has it on dim 0, and `cuda-exl3` loads the head through its own
  `_vocab_loaders`. Putting it in the mapping would load without error and be wrong.
- **`in_proj_qkvbfg_a` is absent on purpose.** Mixed precision; `resolve()` returns `None` for it
  whatever the mapping says. The fix is the module split, not a name.
- **`wk_weights_proj` is absent on purpose.** Both halves are BF16 in this checkpoint and the module
  is already built with `quant_config=None`.

The plugin's own README carried a different example, `{"in_proj_qkvbfg_a": ["qkv_proj", "conv1d"]}`,
which resolves **0/34** here: the group must be entirely EXL3 and `conv1d` is BF16 and additionally
split three ways. We measured it, reported it, and the author corrected the documentation and added
the rule that explains it ([CREDITS](../CREDITS.md)).

### 3.2 The shard-index trap, and the four asserts

The one genuinely dangerous failure mode is a wrong shard index: the load completes **without error**
and the numbers are wrong. The order of the sources in `packed_modules_mapping` must say the same
thing as the shard ids in `stacked_params_mapping`:

```text
(".in_proj_qkv",   ".qkv_proj",  (0, 1, 2))   # one checkpoint tensor, three vLLM shards
(".in_proj_bfg_a", ".b_proj",    0)
(".in_proj_bfg_a", ".f_a_proj",  1)
(".in_proj_bfg_a", ".g_a_proj",  2)
```

**A tuple shard id must start at 0.** `weight_loader_v2` indexes the tuple *relatively*
(`linear.py:901-916` builds `output_sizes` from the tuple, then `_load_fused_module_from_checkpoint`
enumerates 0,1,2), so `(3,4,5)` would load without error and write to the wrong slice. `in_proj_qkv`
uses `(0,1,2)`, which is the second reason to split the module rather than keep one module with a
BF16 tail.

Why the tuple is free rather than expensive: `suh` is a `PerTensorScaleParameter`, so the same row is
replayed for each index without narrowing, and `process_weights_after_loading` then collapses the
three identical `suh` rows into one group — the kernel sees a single fused GEMM again.

vLLM's own "following weights were not initialized" gate does **not** protect this, because
`process_weights_after_loading` is defined on the base class and every parameter of every linear
counts as loaded. So the patch carries its own gates:

| # | where | what it checks |
|---|---|---|
| 1 | after `in_proj_qkv` is created | three EXL3 shards of the expected width, and the expected total |
| 2 | around `process_weights_after_loading` | the three identical `suh` rows collapsed to **one** group; if not, different tensors reached different shards |
| 3 | while splitting `conv1d` | the three thirds are **different from each other** (identical thirds mean the split was on the wrong axis) and the row count divides by three |
| 4 | at the head of `load_weights` (no data needed) | mapping order against shard ids, tuples starting at 0 and contiguous, and `lm_head` **not** in the mapping |

**All four were silent on the live boot** `[measured-here]`.

### 3.3 A vLLM bug found on the way: `ReplicatedLinear` has no `weight_loader_v2`

The meta-device dry run raised something the design had not predicted:

```text
AssertionError: Tried to load weights of size torch.Size([1536])
                to a parameter of size torch.Size([1, 1536])
```

The sparse indexer's `wq_b` is a `ReplicatedLinear`, and vLLM's `ReplicatedLinear` never dispatches
to `weight_loader_v2` — its only loader asserts `param.size() == loaded_weight.size()`, which a
per-shard `suh` of shape `(num_shards, k)` can never satisfy. **Any replicated linear served by a
v2-parameter quantization method fails at load.** It had never shown up because `wq_b` is BF16 in the
experts-only checkpoint; S2 quantizes it for the first time. Leaving it BF16 is not an option — this
checkpoint has no BF16 copy of it, so 44 tensors would be orphaned and the boot would still stop.

Our fix routes that parameter through the v2 entry point with the shard index pinned at 0, and
limits itself by checking that the parameter actually carries the v2 interface. A second detail sits
underneath it: `ReplicatedLinear` never calls `update_param_tp_status()`, so its parameters carry the
**global** TP rank; a replicated layer holds the whole tensor on every rank, so the copy has to pin
`tp_rank = 0` or rank 1 overruns on `narrow`. **A single-rank test could not have caught that; the
two-rank meta-device run did** `[measured-here]`.

We reported it as a vLLM gap. The author's reading was better and he fixed it on his side instead:
`suh` is `(num_shards, k)`, the v1 path does not know that, so the plugin now places a bare `(k,)`
into row 0 when the shapes say that is what happened and delegates otherwise (`d19dee0`). Our
workaround can retire once that commit is in an image we run.

---

## 4. The measurement, at TP=2

**Why TP=2.** At two ranks nothing needs padding — 32 heads and 77,440 vocab rows per rank — so the
question "what is the dense stage worth" is answered with no new padding machinery. TP=3 needs a
padded-load path that does not exist yet (§7).

**Settings, both arms identical unless the row says otherwise:** two DGX Spark nodes, TP=2, expert
parallel **off**, image `exl3-zeus:62f53e6`, EXL3 weights, KV `fp8`, DFlash2 draft k=7,
`gpu-memory-utilization 0.85`, `--block-size 256` requested, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `NCCL_MAX_NCHANNELS=8`, mesh plugin with both cables per peer,
`CUDA_EXL3_TUNE_CACHE` warm, no fast-load sidecar in either arm, no `HAREM_SW_BLOCK_SIZE`, no fp8
draft cache in either arm, temperature 0, reasoning effort **low**, medians of three rounds,
5 September 2026. Control is `brandonmusic/GLM-5.3-Flash-tr3-4bpw` on the same stack.

**One setting is not identical and it matters:** the full-scope arm runs `max_model_len` **65,536**
against the control's **1,000,000**, because at 1M it could not boot (§6.1). That changes the hybrid
allocator's page size as well as the pool, so **only C1–C4 are treated as a comparison**; C6/C8, every
prefill figure and the pool are not.

### 4.1 Speed

| metric | **full-scope** | experts-only control | delta | verdict |
|---|---|---|---|---|
| **C1 per stream (tok/s)** | **68.00** | 54.69 | **+24.3 %** | the lever |
| **C1 aggregate (tok/s)** | **59.93** | 47.40 | **+26.4 %** | the lever |
| C2 aggregate | 83.02 | 68.03 | +22.0 % | faster |
| C4 aggregate | 111.05 | 90.66 | +22.5 % | faster |
| C6 aggregate | 109.75 | 110.12 | — | **KV-bound, void** |
| C8 aggregate | 110.03 | 133.57 | — | **KV-bound, void** |
| TTFT at C1 (s) | 0.524 | 0.615 | −14.8 % | faster |
| draft acceptance at C1 | 63.14 % | 64.08 % | equal | drafter unchanged |
| accepted tokens per step at C1 | 5.42 | 5.49 | equal | drafter unchanged |
| cold first request | TTFT 0.85 s, 47.5 tok/s | TTFT 1.44 s, 40.5 tok/s | +17 % | faster |
| boot, cold, no fast-load sidecar | 355 s | 396 s | −10 % | smaller checkpoint |
| gates, cold and warm | **10/10 · 12/12** | 10/10 · 12/12 | equal | pass |
| free host RAM after the run | 8.9 / 10.6 GiB | 6.6 / 7.7 GiB | +2.3…2.9 | lighter weights |
| KV pool | 31,343 at 65,536 ctx | 665,625 at 1,000,000 ctx | — | **not comparable** |
| prefill (7k and fresh) | **not measurable** | 1,135 / 1,334 tok/s | — | see §6.1 |

All three rounds of the full-scope arm, for the record `[measured-here]`:

| round | C1 aggregate | C1 per stream | C2 | C4 | C6 | C8 | acceptance at C1 |
|---|---|---|---|---|---|---|---|
| 1 | 60.21 | 69.28 | 83.02 | 110.15 | 109.75 | 110.26 | 63.66 % |
| 2 | 58.39 | 65.38 | 83.32 | 111.05 | 111.47 | 110.03 | 61.46 % |
| 3 | 59.93 | 68.00 | 81.64 | 111.43 | 106.18 | 108.71 | 63.14 % |

> **Correction to the figure we posted upstream.** The table sent to the kernel author's issue thread
> gave the control's C1 as "54.7 / 54.3" for aggregate / per stream and derived "+9.5 % / +25 %".
> 54.7 is the control's **per-stream** median and 54.3 is its round-3 per-stream value; its aggregate
> median is **47.40**. The like-for-like deltas are the ones in the table above: **+26.4 % aggregate
> and +24.3 % per stream** `[retracted]` ([11](11-open-issues.md) §1.9 row 31). The full-scope column
> and the conclusion are unaffected.

### 4.2 Where the gain comes from, and how much of the estimate arrived

```text
control  : 5.49 tokens/step ÷ 54.69 tok/s = 100.4 ms/step
full-scope: 5.42 tokens/step ÷ 68.00 tok/s =  79.7 ms/step
saved     : 20.7 ms/step  (step −20.6 %, throughput +24.3 %)
```

The profile attributed **42.90 ms of a 94.65 ms step** to dense BF16 GEMM
([10](10-results-and-roofline.md) §5.3), and the estimate was that 4-bit takes it to ~11 ms — about
32 ms, roughly **+34 % single-stream** ([11](11-open-issues.md) §2.22).

**We got 20.7 ms, 65 % of it.** The gap is explained rather than mysterious: this checkpoint leaves
`in_proj_bfg_a`, `f_b_proj`, `g_b_proj`, `kv_b_proj`, `indexer.wk_weights_proj` and the three
`conv1d` in BF16 anyway, and not every byte in the dense stage is bandwidth-bound. **The +34 % was
not refuted; it was an upper bound and it was optimistic** `[estimate]`.

**Acceptance and accepted tokens per step are unchanged**, so the entire gain is arithmetic and none
of it is drafter behaviour. That is also the strongest evidence the lever carries to TP=3: there is
nothing on the speculative side to re-tune, and at three ranks the per-rank dense traffic falls
further.

### 4.3 What actually resolved to EXL3

From the two-rank meta-device dump, per rank, TP=2 `[measured-here]`. Quant-method census:
`Exl3LinearMethod` 203, `Exl3MoEMethod` 42, `UnquantizedLinearMethod` 268,
`UnquantizedEmbeddingMethod` 1.

| resolved to EXL3 | n | | left BF16 | n |
|---|---|---|---|---|
| `mlp.experts` (routed) | 42 | | `model.embed_tokens` | 1 |
| `self_attn.o_proj` | 45 | | `mlp.gate` (router) | 42 |
| `mlp.gate_up_proj` (3 dense + 42 shared) | 45 | | `self_attn.f_b_proj` | 34 |
| `mlp.down_proj` (3 dense + 42 shared) | 45 | | `self_attn.g_b_proj` | 34 |
| **`self_attn.in_proj_qkv`** (KDA) | **34** | | **`self_attn.in_proj_bfg_a`** | **34** |
| `self_attn.q_b_proj` | 11 | | `self_attn.kv_b_proj` | 11 |
| `self_attn.fused_qkv_a_proj` | 11 | | `self_attn.indexer.wk_weights_proj` | 11 |
| `self_attn.indexer.wq_b` | 11 | | `q_conv1d` / `k_conv1d` / `v_conv1d` | 34 each |
| **`lm_head`** (6 bit) | **1** | | | |

On the control checkpoint the same dump gives EXL3 for the 42 routed-expert layers and nothing else,
which is what `scope: glm53_routed_experts_only` means.

---

## 5. Quality

MMLU sample, 35 questions per subtask, 0-shot, concurrency 8, temperature 0, reasoning effort `low`
`[measured-here]`:

| arm | MMLU | CI | humanities | social sci. | other | STEM |
|---|---|---|---|---|---|---|
| **full-scope, TP=2** | **86.32 %** | ±0.75 | 86.81 | 89.05 | 85.27 | 84.96 |
| experts-only control, TP=2 | 86.4 % | ±0.7 | — | — | — | — |
| NVFP4 sibling stack, TP=3 (reference) | 86.7 % | — | — | — | — | — |

**The gate passes.** The difference is 0.08 points, about a tenth of either error bar. The one real
quality risk we had flagged was the **6-bit `lm_head`** at vocab 154,880, and it **cost nothing
measurable**. Second evidence in the same direction: correctness probe 10/10 and code exam 12/12,
cold and warm, on both arms.

The run was 7,980 requests in 26 minutes with no timeout. One piece of luck worth writing down: MMLU
0-shot prompts sit below the ~2,000-token admission ceiling this arm turned out to have (§6.1). That
was not calculated in advance, and on a benchmark with longer prompts the same arm would have
produced nothing.

**The control's 86.4 ±0.7 was not re-run in this arm** and it is the same figure the rest of this
repository quotes: measured earlier the same day, same checkpoint, same TP=2, but with the MTP
drafter rather than DFlash2. A log-likelihood task does not go through the speculative decoder at
all, so re-running it would have spent 20 minutes reproducing a known number. Say it plainly rather
than let the table imply two fresh runs `[measured-here]`.

---

## 6. What it cost

### 6.1 Context — the price, and it is not small

At TP=2 the full-scope model leaves so little KV that **the long-prompt path closes entirely**. The
engine's own log explains the mechanism: `Mamba cache mode is set to 'align'` →
`Setting attention block size to 4608 tokens to ensure that attention page size is >= mamba page
size`, and the pool comes out at **31,343 tokens — about 6.8 pages**. With six or seven pages, a long
prompt cannot be scheduled at all. Measured admission `[measured-here]`:

| prompt | result |
|---|---|
| 844 tokens | served, 1.1 s |
| 1,684 tokens | served, 1.7 s |
| ~2,800 tokens | **never scheduled** — 45 s later still `Running: 0, Waiting: 1`, KV usage 0 % |
| 7,382 tokens (`prefill-7k`) | **never scheduled**; the client gave up at 600 s |
| 8,204 tokens (`prefill-fresh`) | not run — same fate, and three 900-second timeouts to prove it |

That is why every prefill number and the C6/C8 aggregates are void in this arm, and it is also why
`--block-size 256` has no effect here: the hybrid allocator overrides it.

This is **the known context cost of two nodes made worse**, not a defect of full scope: our own TP=2
control pool is 665,625 tokens against production TP=3's 4,699,724 — **14 %** — because a 164 GiB
model over two nodes leaves ~85 GiB of weights per node and very little else. Full scope makes it
sharply worse rather than better (§6.2). **TP=2 is a measurement rig here, not a serving
configuration**, and that was true before this arm.

### 6.2 The memory reading we cannot explain

Two measurements about the same arm point in opposite directions, and both are left standing:

- At `max_model_len=1,000,000` the control arm had **4.4 GiB** of KV memory available and the
  full-scope arm **0.73 GiB** — the full-scope boot failed its own budget gate, needing 6.6 GiB.
  Consumed per node reads **95.5 GiB** against the control's ~85 GiB, i.e. **~10 GiB heavier**
  `[measured-here]`.
- Yet the checkpoint is 10 GiB **smaller** on disk, and post-run free host memory says its weights
  are ~3.3 GiB per node **lighter** `[measured-here]`.

Taken together that implies roughly **+15 GiB of non-weight allocation**, and the leading suspect is
`Exl3LinearMethod.process_weights_after_loading` calling `ops.reserve(...)` per EXL3 module: the
count of EXL3 dense linears per rank goes from **1 to 203**, and `lm_head` alone has `n_total`
51,712. The word "shared workspace" in that code implies a maximum rather than a sum, which would
make the hypothesis wrong — **it has not been measured** `[not tested]`. It is the single number to
settle before TP=3 (§7.3).

**A separate finding fell out of the same investigation.** Lowering `max_model_len` from 1,000,000 to
65,536 raised available KV memory from 0.73 to **5.41 GiB**: about **4.7 GiB per node of persistent,
non-KV allocation scales with `max_model_len`** on this stack `[measured-here]`. That buffer is paid
in production too, and it explains why the control arm at 1M could only find ~4.4 GiB for KV.

### 6.3 The acceptance test we designed did not work

The planned boot gate was `CUDA_EXL3_DEBUG_NAMES=1`, reading off which modules stayed unquantized. It
printed **nothing**. The flag reached the container (verified with `docker inspect`) and the code was
right; the plugin logs that line at `logger.info`, and this image configures only vLLM's own logger,
so third-party INFO goes nowhere. The same is true of the plugin's `EXL3: N quantized tensors` line,
which we had already caught being invisible earlier the same day and correctly refused to reason from
([11](11-open-issues.md) §1.9).

Three substitutes carried the gate instead, and they are the ones to reuse `[measured-here]`:

1. **Zero orphaned tensors.** With 36,719 `.trellis` in the checkpoint, any module that failed to
   resolve would leave its trellis with no parameter and stop the boot. It did not stop.
2. **Asserts 1 and 2 silent** — `in_proj_qkv` really was built as three EXL3 shards whose `suh`
   collapsed to one group. A module that had fallen back to BF16 would have failed assert 1.
3. **The two-rank meta-device dump** (§4.3), taken before the boot, with the engine down.

We reported the logging bug; the author fixed it at `warning` level and made it log the **hits** as
well as the misses, with running tallies, so a mapping that leaves half the attention stack in BF16
shows up as a climbing BF16 count (`807d798`).

### 6.4 A fast-load sidecar refused the production restore, and it was right

Restoring production after the arm failed on all three nodes with `Exited (21)`:

```text
preflight-fastload: sidecar stale - boot refused
  patches.patch-fullscope-tp2.py: recorded='<none>' now='bca9a201...'
  patches.tp3-prelude.sh:         recorded='e17d46a4...' now='5eba79f3...'
```

The sidecar records the identity of the patch set it was dumped against, including the `sha256` of
**every** `patch-*.py` in the patch directory and of the prelude ([08](08-fast-boot.md) §4). During
the arm the full-scope patch had been placed in the TP=3 patch directory and an env-gated hook added
to the TP=3 prelude. Neither changes production behaviour — the hook does nothing with the flag unset
and the patch is never called at TP=3 — but both change the **identity**, and a sidecar is
pre-processed weights belonging to a specific set of code. Using it silently could have served wrong
answers. **The refusal is the design working.**

The fix was to restore the prelude from its backup and unlink the patch from the TP=3 directory (the
file survives through its hard link in the TP=2 directory), then reboot: production came back in
264 s with a pool of 4,696,969 against 4,699,724 before the arm, 0.06 % `[measured-here]`.

**The rule this produced is in [09](09-measurement-protocol.md) §11.2:** between a dump and a load,
nothing is added to, removed from or edited in the patch directory — experimental patches live in
their own directory and are hard-linked, not copied, if two places need them.

### 6.5 The rest of the ledger

- Production was down **1 h 15 m**, planned, with the third node idle throughout.
- `max_model_len` was lowered to 65,536 **for the experiment only**, in each node's own env file with
  a backup; the production env files were not touched.
- Two nodes hold a second 154 GiB checkpoint and sit at ~95 % full. Nothing was deleted.
- No fast-load sidecar could be built for this arm (52–56 GB per node, no room), so both arms boot
  in 355–396 s rather than 274 s. Both arms are equally affected, so the comparison is fair.

---

## 7. TP=3: what it needs

The measurement changes this from "possibly worth building" into an item with a price attached: at
TP=3, applying the measured per-stream ratio to today's production single-stream figure projects
roughly **74–75 tok/s** `[estimate]`. That is a projection, not a measurement, and it has two named
uncertainties: the dense share of a step differs at three ranks, and the `in_proj_qkv` tuple path has
never run under expert parallelism.

### 7.1 On our side

**Every head pad is already 128-aligned, and that was checked rather than assumed** `[measured-here]`.
MLA `qk`/`v` head_dim is 256 and KDA head_dim is 128, so padding 64 → 66 heads gives whole
128-column blocks everywhere: 512 columns on `q_b_proj` and on `o_proj`'s input, 256 per KDA shard.

Exactly two pads are **not** aligned, and both are the same one-line arithmetic error —
`lcm(64, tp)` where it should be `lcm(128, tp)`:

| what | today | required | why |
|---|---|---|---|
| vocab padding | `padding_size=192` → 154,944; 51,648 = **403.5** × 128 per rank | **`padding_size=384`** → 155,136; 51,712 = 404 × 128 | `linear.py:89-94` **hard-refuses** a padded vocab for an EXL3 head, and this checkpoint has no BF16 head to fall back on |
| shared expert intermediate | 2048 → **2112**; 704 = **5.5** × 128 per rank | 2048 → **2304**; 768 = 6 × 128 | `down_proj` at k=704 hits the plugin's explicit refusal; `gate_up_proj` gets a half-block output pad, where there is only a warning — one loud failure and one **silent** one, same fix |

**2,176 is the wrong number** and is worth naming because it looks right: it is 17 × 128, but
2176/3 is not an integer. The padded width must be a multiple of `lcm(128, 3) = 384`, so **2,304**
(= 18 × 128, 768 per rank). That number is already known here — it is the width of the
routed-expert sidecar in [11](11-open-issues.md) §2.9.

**And one new obstacle, TP=3 only (A9).** The KDA tuple-shard path splits the checkpoint tensor using
the module's **padded** `output_sizes`. At TP=2 those are `[8192]×3` against a 24,576-wide checkpoint
tensor and it fits exactly. At TP=3 they are `[8448]×3`:

```text
segment 0: narrow(start=0,     len=8448) -> 8448    correct
segment 1: narrow(start=8448,  len=8448) -> 16896   wrong (true start 8192)
segment 2: narrow(start=16896, len=8448) -> 25344   overflow -> RuntimeError
```

The fix is ours: record the checkpoint's real per-shard width on the module and split by that,
letting the existing zero-pad path widen each rank's slice afterwards, with asserts that the three
widths are equal and that the pad is a multiple of 128. **It has not been written and has never
run** `[not tested]`.

Also on our side: the checkpoint has to be copied to the third node (154 GiB), and the fast-load
sidecar has to be re-dumped, because four inputs to its identity change at once — different
checkpoint, different sidecar config, an edited `patch-vllm-tp3.py`, and a new `patch-*.py` in the
directory.

### 7.2 What only the plugin author can do

The `svh = 0` padded-load mechanism is in fact **already happening** on this stack for column-parallel
EXL3 modules: our TP=3 padding patch fills missing rows with zeros, and `svh` goes through the same
path, so padded columns get `svh = 0` and the trellis behind them is don't-care. The invariant holds
by accident, and **nothing checks it** — which is exactly how today's 2112/192 arithmetic would have
produced silently wrong output. Four things were asked for, in the issue thread:

1. **`lm_head`.** The hard refusal at `linear.py:89-94` has to become a padded-load path for the
   vocab-parallel head. This checkpoint has no BF16 head, so there is no fallback.
2. **A "refuse unless the pad is 128-aligned" gate**, so that a silently-correct situation cannot
   become a silently-wrong one.
3. **The input-dim case.** `Exl3SuhParameter.load_row_parallel_weight` does its own `narrow` rather
   than going through vLLM's patched loader, so at TP=3 the last rank overruns on `o_proj`.
4. **Confirmation that `decode(0)` is finite**, so that a zero pad trellis multiplied by `svh = 0`
   cannot produce NaN.

**Item 4 is answered and it is answered completely.** The author swept **all 65,536** 16-bit trellis
codes through the device decoder for all three codebooks: zero non-finite values in every case, with
bounded ranges (`3inst` [−3.9570, +3.9727], `mcg` [−3.9492, +3.9492], `mul1` [−3.4531, +3.3477])
`[reported]`. The decoder is total and bounded, so `svh = 0` on a pad column is exactly
`0 × finite = 0`. That is the full domain enumerated rather than an argument from construction, which
is the right standard underneath a padded-load path.

Items 1–3 are agreed and scoped, and were explicitly waiting on the quality gate before being built.
**The gate has now passed**, and the word has been sent.

### 7.3 The acceptance test for TP=3, in order

Baseline is production configuration 8, measured the same day with the same script: C1 **59.9** ·
C8 **178.6** · acceptance **63.9 %** · prefill-fresh **1,774** · KV **4,699,724** · gates
**10/10 · 12/12** · MMLU **86.4 ±0.7**.

| # | test | gate |
|---|---|---|
| 1 | both patch scripts in `--check` mode, engine down | every anchor matches exactly once |
| 2 | meta-device dump, **three ranks**, TP=3 | 0 unmapped / 0 unfilled on **every** rank, EXL3 and BF16 sets as in §4.3. Rank 2 is mandatory: the pad lives there and a TP=2 run cannot see it |
| 3 | **memory arithmetic**, model-free | settle the +15 GiB question of §6.2 *before* spending a boot |
| 4 | preflight on the padded sidecar | vocab 155,136, shared expert 2,304 → 768 per rank, 66 heads → 22 per rank, expert parallel mandatory |
| 5 | boot | prelude stamps, patch `sha256`, every anchor applied |
| 6–9 | asserts 1–4 | all silent |
| 10 | **assert 5, new** | on every padded EXL3 module, on the rank that owns the boundary: `svh` is exactly zero beyond the real width, and both the real width and the pad are multiples of 128 |
| 11 | **assert 6, new** | A9: the three checkpoint split widths equal, and real + pad equals the padded total |
| 12–13 | correctness probe, code exam | 10/10 and 12/12, **cold and warm** |
| 14 | speed | C1 ≥ 59.9 (expected 70–78), C8 ≥ 178.6, prefill-fresh ≥ 1,774 |
| 15 | draft acceptance | ≥ 60 % |
| 16 | KV pool | reported; **≥ 3.4 M** expected; below 3 M, stop and reopen §6.2 |
| 17 | MMLU sample | 86.4 ±0.7 |
| 18 | free RAM and swap | ≥ 4 GiB free, swap stable |
| 19 | restore production and verify | health, pool within ~0.5 %, gates full |

If a gate fails, the suspects in order: the A9 split width (assert 6), rank 2's pad columns
(assert 5), the row-parallel `suh` narrowing on `o_proj`, the 6-bit `lm_head`, and the `q`/`k`/`v`
order of the `conv1d` split (assert 3).

---

## 8. What this changes

- **The largest open item on this stack is no longer an estimate.** The dense stage is worth
  **+24.3 % per stream** at TP=2 with **no quality cost** and **no drafter change**
  ([11](11-open-issues.md) §2.22).
- **`routed_experts_only` was never a quality decision.** Two lines in a model file made it the only
  loadable scope, and they would have done so for any checkpoint in any format.
- **The KDA factorisation mismatch is not an EXL3 problem.** `conv1d` is BF16; a BF16 copy of this
  checkpoint fails identically. Anyone loading an upstream-layout GLM-5.3-Flash checkpoint on the
  NVIDIA `glm5next` reader will hit it, whatever the quantization.
- **TP=2 is not a serving configuration for a full-scope checkpoint.** The 31k pool closes the long
  prompt path. The place for full scope is TP=3, and the remaining work there is a launcher constant,
  a shared-expert width, one patch we have not written, and a padded-load path on the plugin side.

---

## 9. What is next

[10 — Results and roofline](10-results-and-roofline.md) §2.2 for this arm in the results progression,
[11 — Open issues](11-open-issues.md) §2.22 for the TP=3 item as it now stands, and
[01 — Model and licence](01-model-and-license.md) §1 for how the two checkpoints compare.

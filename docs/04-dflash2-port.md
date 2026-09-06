# 04 — The DFlash2 port

**Applies to: both tracks.** The drafter's GQA divides by two, so the 32/8 → 36/9 pad described here
is a TP=3 detail inside a shared page.

The base image ships **DFlash v1**. It had never run a DFlash2 drafter, and — as the first boot proved — it
could not have run DFlash v1 either, because the *target* half of the interface was missing from its GLM-5.3
model file. This page is the whole port: the upstream delta, the two gaps that delta did not cover, our
fail-closed guards, the TP=3 extension, and the draft-depth decision.

The drafter is **`incoai/GLM-5.3-Flash-DFlash2`** (BF16, 2.34 GB), licensed **CC BY-NC-ND 4.0**. Our permission
to use it does **not** transfer to you — read [../LICENSES.md](../LICENSES.md) before you download it. The
recipe runs without a drafter, about 2.6× slower at a single stream. Everything below assumes the base image
from [02 — Image build](02-image-build.md); the port is the second Docker layer on it.

**Settings for the TP=2 measurements (§1, §3, §7, §10).** Image `exl3-zeus:dflash` — the port layer over
`exl3-zeus:serve` (`vllm/vllm-openai:glm53-flash-arm64-cu130`, vLLM `0.1.dev20051+g487ecf187`, plus `cuda-exl3`)
— TP=2 over `head` and `worker-1`, EXL3 4bpw, `kv-cache-dtype fp8` on the target and BF16 on the draft,
`--block-size 256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `gpu-memory-utilization 0.85`, no KV
pin, temperature 0, reasoning effort **low**, realistic prompts (12 short English code prompts), 4–5 September
2026. **Caveat:** two sweep rounds averaged, not the five-round protocol of [09](09-measurement-protocol.md),
adopted afterwards. The three arms were measured identically in one session, so the comparison holds; the
absolutes carry more warm-up noise than the front page. The `cuda-exl3` commit behind that base tag was not
recorded, so do not compare these against the kernel arms in [05](05-expert-parallel-and-cuda-exl3-fixes.md).

## 1. Why bother

Three arms, one session, one prompt set, TP=2 `[measured-here]`:

| | no draft | MTP k=3 | **DFlash2 k=7** | DFlash2 vs MTP |
|---|---|---|---|---|
| C1 total tok/s | 14.42 | 30.49 | **42.91** | **+40.7 %** |
| C1 per stream | 14.73 | 33.52 | **50.79** | **+51.5 %** |
| C2 total | 28.80 | 47.89 | **60.80** | +27.0 % |
| C4 total | 46.49 | 71.12 | **83.89** | +18.0 % |
| C6 total | 49.30 | 85.23 | **98.08** | +15.1 % |
| C8 total | 69.50 | 102.32 | **114.60** | **+12.0 %** |
| acceptance rate | — | 77.3 % | 62.4 % | — |
| accepted tokens per step | — | 3.31 | **5.37** | **+62 %** |
| KV pool (tokens) | — | 1,987,179 | 825,000 | −58 % |
| prefill, 7K prompt | 1,137 | 1,131 | 1,035 | −8.5 % |

Quality gates identical to the MTP reference: probe **10/10**, code exam **12/12 PASS, 0 FAIL** `[measured-here]`.

**The acceptance trap.** DFlash2's acceptance *rate* (62.4 %) looks worse than MTP's (77.3 %), and it is the
wrong number to read: MTP proposes 3 tokens per step and DFlash2 proposes 7, so the rate is a fraction of a
different denominator in each arm. What decides throughput is **accepted tokens per step** — how far the
sequence advances per target forward — and there DFlash2 is 62 % ahead, **5.37 against 3.31**. Our NVFP4 stack
runs the same drafter at k=7 at 62–65 % and about 5.3 tokens per step, so 62.4 % is the expected band, not a
silently broken draft `[measured-here]`.

## 2. What was ported, file by file

### How the port was derived — a merge, not a hand-copy

The image's vLLM tree was extracted with `docker create` / `docker cp` and compared against a fork checkout:
**byte-identical to fork commit `487ecf187`** for every file involved (per file, diff = 0), which is what makes
a real three-way merge possible. `487ecf187` (image, 25 August) is **not** an ancestor of the fork's HEAD
`9c4dd0548`; their merge base is `b908a21f9` (13 August), so a straight cherry-pick of fork HEAD would have
dragged unrelated drift in. Each modified file was therefore merged with `git merge-file <image-version>
<b389ac294^> <b389ac294>` — *the upstream DFlash2 change alone*, onto the image's own tree. Five of seven merged
clean; two conflicted, resolved by hand. **Net result: this port is upstream vLLM commit `b389ac294`
("[Spec Decode] DFlash2: local convolution + candidate selector", #52816), and nothing more.**

### New files, verbatim upstream

| File | What it is |
|---|---|
| `models/qwen3_dflash2.py` | The DFlash2 draft head: `DFlashGroupedConv` (the two-tap dynamic convolution that keeps the draft from decaying toward the end of a block), `CandidateSelector` (low-rank predecessor/successor codebooks scoring edges between per-position candidates), `DFlash2Qwen3DecoderLayer`, `DFlash2Qwen3ForCausalLM.compute_candidates`. Plus our fail-closed guards (§4). |
| `v1/worker/gpu/spec_decode/dflash2/speculator.py` | `DFlash2Speculator`, a subclass of the base image's `DFlashSpeculator`. Overrides `_generate_draft` to take top-k candidates per position, score edges with the selector, and walk one coherent path (`_selector_walk_kernel`, Triton). **Byte-identical between upstream `b389ac294` and fork HEAD — the fork never touched it.** |
| `v1/worker/gpu/spec_decode/dflash2/__init__.py` | Package marker. |

### Merged edits

| File | Delta | Why DFlash2 needs it |
|---|---|---|
| `models/registry.py` | +1 | Registers `DFlash2DraftModel`. **Conflicted.** |
| `v1/worker/gpu/spec_decode/__init__.py` | +6 | `init_speculator` dispatches to `DFlash2Speculator` when the drafter's architecture is `DFlash2DraftModel`. **Conflicted.** |
| `config/vllm.py` | +17 | `_is_dflash2_draft()` forces the **V2 model runner**. Load-bearing: on V1 the same checkpoint drafts through `DFlashProposer`, which never calls the candidate selector, so the draft silently degrades to DFlash1 with no error. |
| `v1/worker/gpu/spec_decode/speculator.py` | +16 −5 | Adds the `draft_logits_spec()` hook. DFlash2 overrides it to fp32 / `-inf`: its kernel writes only the K candidate columns, and the base's `torch.zeros` would leave non-candidates at probability 0 rather than excluded, so the walk and the rejection sampler would read different distributions. |
| `v1/worker/gpu/sample/gumbel.py` | +52 −34 | Pure refactor: extracts `gumbel_noised_argmax` out of `gumbel_block_argmax` so the selector-walk kernel reuses the exact same Gumbel-max draw. Sharing it is what lets draft and verification agree on the noise. Behaviour unchanged. |
| `models/qwen3_dflash.py` | +11 −4 | Subclass hooks only: `decoder_layer_cls` / `model_cls` so `DFlash2Qwen3Model` can substitute its own classes, and `_dflash_layer_causal` now honours an explicit `is_causal` on the config (this drafter sets `"is_causal": false`). No change to DFlash v1 behaviour. |
| `layers/logits_processor.py` | +81 | `LogitsProcessor.get_top_k_tokens`: a vocab-parallel top-k that all-gathers `2k` values instead of the whole vocabulary. DFlash2 calls it once per drafted position. Plus one env knob (§4). |

**Both conflicts were unrelated upstream features**, on the DFlash2 branch and absent from the image's, and both
were deliberately **not** taken: Muse-Glimmer drafter aliases in `registry.py` (that model does not exist here,
so registering it would be dead weight), and an `extract_hidden_states` speculator branch in
`spec_decode/__init__.py` (the method and its module do not exist here, so the branch would be a broken import).

### Deliberately not ported

| Fork commit | What it does | Why not |
|---|---|---|
| `da4d7be6c` — native serialized MXFP8 projections | Threads the draft `quant_config` into `DFlashGroupedConv` / `CandidateSelector`, builds the fused context K/V projection from serialized MXFP8 values plus E8M0 scales. | Quantization-path specific. Our drafter is BF16, so `quant_config` is `None` and the upstream form is *behaviourally identical*. Porting it would drag ModelOpt MXFP8 machinery in for no gain. Replaced by a fail-closed guard (§4) so a quantized drafter cannot slip through silently. |
| `a9a17e709` — DFlash2 load fix (#53435) | Adds `decoder_layer_cls` to `DFlashQwen3Model`. | Redundant: a re-land of hooks `b389ac294` already adds — and `b389ac294` adds `model_cls` too, which this one does not. Verified on the merged file. |
| `fe755c889` — decouple the draft's Gumbel noise stream from the target's (#54282) | Upstream follow-up, 29 August, touches `dflash2/speculator.py`. | Not in the fork's HEAD. The known-good result we compare against (k=7, acceptance 62–65 %) was produced *without* it, so taking it would make this port diverge from that configuration. See §8. |
| Quantization-fork patches generally (loader-side draft head padding, custom kernels, backend integration) | — | Out of scope; the base image has its own GLM-5.3 support. |

The hunk-level rationale is in [`patches/dflash2-port/PATCHES.md`](../patches/dflash2-port/PATCHES.md); the
replacement sources are in `patches/dflash2-port/files/`.

## 3. The two things the delta did not cover

Porting `b389ac294` made the drafter **loadable**, not **runnable**. Both remaining gaps were found by booting
and reading the traceback, not by prediction.

| # | result |
|---|---|
| 1 | Weights loaded → `Model does not support EAGLE3 interface`. Target side missing (§3.1). |
| 2 | Reached KV sizing → `indexer.k_cache: page size is not divisible…`. KV grouping (§3.2). |
| 3 | **Up.** Model 81.53 GiB in 309 s; 17.27 GiB left for KV; KV pool 825,000 tokens; graph capture 12 s / 0.83 GiB; engine ready at 131 s `[measured-here]`. |

### 3.1 The target-side EAGLE3 interface

**Symptom:** `RuntimeError: Model does not support EAGLE3 interface`, raised in
`v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`, *after* the weights had loaded.

**Root cause:** DFlash is not self-contained. The drafter reads the *target's* intermediate layers
(`dflash_config.target_layer_ids = 5, 14, 24, 33, 42`), so the target class must expose `SupportsEagle3`. The
base image's `vllm/models/glm5next/nvidia/model.py` contains **zero** aux-hidden-state support (`grep -c
aux_hidden_state` → 0) and matches `487ecf187` exactly. So DFlash **v1** could not have worked here either — it
was never a DFlash2-specific gap.

**Fix:** three-way merge of fork commit `e7097feb6` *"Support GLM-5.3 DFlash speculation"* onto the image's
`model.py`. It adds `EagleModelMixin` and `dflash_capture` to `Glm5NextModel`, a `_prepare_aux_hidden_state()`
running `hc_post` then `hc_contract` for mHC layers when drafting (the drafter wants the contracted form), a
forward loop collecting aux states at `layer_idx + 1 in aux_hidden_state_layers`, and `SupportsEagle3` on both
`Glm5NextForCausalLM` and `Glm5NextForConditionalGeneration`. **Two conflicts, both base-image drift rather than
DFlash:** the fork had dropped `SupportsPP` here, so we **kept the image's `SupportsPP`** and only *appended*
`SupportsEagle3`; the fork's neighbouring lines belong to other commits and reference a `model_state` module
absent from this tree, so they were dropped.

**Proof at runtime** — exactly `target_layer_ids + 1` `[measured-here]`:

```
Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)
```

### 3.2 KV cache grouping

**Symptom:**

```
NotImplementedError: Layer language_model.model.layers.3.self_attn.indexer.k_cache: page size is not divisible by the maximum page size and cannot be padded.
```

**Measured, not guessed.** A throwaway diagnostic image logged every layer's spec at the top of
`unify_kv_cache_spec_page_size` — one pass, 72 layers (45 target, 11 indexer k, 11 kpool tail, 5 draft) `[measured-here]`:

| spec | page (B) | block | per token | pad-able? |
|---|---|---|---|---|
| `MambaSpec` (KDA) + `MLAAttentionSpec` | 2,359,296 | 4608 | 512 B | — |
| `MLAAttentionSpec` indexer `k_cache` | 152,064 | 4608 | **33 B** | **no** (`indexes_kv_by_block_stride=False`) |
| `SlidingWindowSpec` (**draft**) | 32,768 | 16 | 2048 B | yes |
| `KpoolTailSpec` | 2,048 | 4 | 512 B | no |

**Root cause.** 33 B/token can never reach 512 B/token by integer block growth — 512/33 is not an integer — and
the indexer's backend cannot pad. So why does the target work alone? Because GLM-5.3 normally takes a
**dedicated grouping path**, `_get_kv_cache_groups_glm5_next`, in which mamba layers co-own the MLA slot tensors
and the kpool tail co-owns the indexer tensors — an image-specific memory optimisation — and that path **never
calls page unification**. It requires every attention spec to be an `MLAAttentionSpec`; the draft's
sliding-window layers fail that test, the model falls through to the generic path, and dies there.

**Fix — the fork's idea, adapted to this image's architecture.** Partition the draft's KV layers away from the
target's, group each side independently, concatenate; **the target's grouping is then bit-for-bit what it is
with no drafter at all.** The fork's own `kv_cache_utils.py` is otherwise restructured and has neither of those
two functions, so it was not taken wholesale — only the idea. Thirteen single-match anchored edits:

1. `_harem_partition_dflash_draft_specs()` splits by layer index: the DFlash head is built with
   `start_layer_id` equal to the target layer count, so its layers are index ≥ 45. Returns early unless the
   method is `dflash`, PP == 1 and the hybrid manager is on.
2. `get_kv_cache_groups()` recurses: group target alone, group draft alone, concatenate.
3. `_glm5_next_tensor_layout()` also detects and returns the draft group (a 9-tuple instead of 8), and fails
   closed on a group that is neither MLA-attention, kpool-tail, mamba nor plain attention.
4. All three consumers of that layout account for the draft group: per-block cost, tensor creation (one private
   `KVCacheTensor` per draft layer) and the max-memory estimate. Omitting any would over-report how many blocks
   fit and leave the draft's tensors unallocated — a silent failure, so the gate asserts all three unpack it.

**Cost:** `5 × 32,768 = 163,840 B` per block against the target's `11 × 2,359,296 + 11 × 152,064 = 27,624,960 B`
— **+0.6 %** `[measured-here]`. **Proof at runtime:**

```
DFlash draft: 5 KV layers kept in 1 independent cache group(s)
```

**Flag forward.** This separation is correct and it is also what later capped the KV pool: an independent draft
group is counted separately by the block allocator, and its interaction with the draft page size is the subject
of [07 — KV pool and the draft page](07-kv-and-draft-page.md) — read that before tuning
`gpu-memory-utilization`.

## 4. The fail-closed additions we made

All three are additive; default behaviour is upstream's.

1. **`DFlash2Qwen3Model._harem_check_port_assumptions()`**, called from `__init__`. *A quantized drafter is
   refused outright* (`NotImplementedError`): the quantized branch was not ported (§2), and without this guard
   such a drafter would get BF16 convolution and selector projections over quantized weights and produce a
   degraded draft with **no error at all**. The docstring names the exact commit and hunks to port if that is
   ever wanted. *Head counts must divide the TP size*: `num_attention_heads` and `num_key_value_heads` must both
   divide `tensor_parallel_size`, so a wrong-shaped drafter stops instead of silently degrading. Both exist
   because of a failure mode already paid for here — a decode kernel quietly wrong at an untested head count.
2. **`HAREM_DFLASH2_FORCE_TORCH_TOPK=1`** in `_flashinfer_topk()`: falls back to `torch.topk` instead of the
   FlashInfer radix kernel. Purely a diagnosis lever — the candidate selector is the only caller of this top-k,
   it runs over a 154,880-entry vocabulary, and GB10 (sm_121) has needed shared-memory workarounds elsewhere
   here. If acceptance ever lands far below 62–65 %, flip this first.
3. **The build-time gate**, `patches/dflash2-port/gate.py`, installed as `/opt/harem/dflash2-gate.py` and run by
   the Dockerfile. It asserts that every ported symbol resolves — the classes, registry entries, the speculator
   subclassing and its `draft_logits_spec` override, `gumbel_noised_argmax`, `get_top_k_tokens`,
   `_is_dflash2_draft`, the EAGLE3 mixin on the target, and all three KV-layout consumers. **If any check fails,
   no image is produced.** Every one guards an absence that would *not* raise at runtime: the engine would boot,
   produce correct-looking text, and lose acceptance.

## 5. The TP=3 extension

At TP=3 the drafter's real 32 query / 8 KV heads do not divide three, so the recipe serves it through a padded
**36/9 sidecar** ([03](03-tp3-padding-and-sidecars.md)). Divisibility alone cannot tell a legitimate pad from a
disaster: several catastrophic configs divide three perfectly. `patches/tp3/patch-dflash-tp3.py` replaces it
with a pad-aware check that reads the checkpoint's own safetensors headers and demands that the config be the
checkpoint plus a zero pad. Five traps, each refused by name:

| trap | example | why divisibility misses it | what the check demands |
|---|---|---|---|
| config *shrinks* the real head count | 24/6 | 24 and 6 both divide 3 | config ≥ checkpoint; smaller is a truncation of trained heads, not a pad |
| pad so wide the top rank owns no real head | 48/12 — the 64→96 mistake | divides 3; **starts, serves, and answers confidently wrong** | rank `tp−1` must own at least one real head |
| pad changes the GQA ratio | 36/12 (4:1 → 3:1) | divides 3 | `heads × stock_kv == kv_heads × stock_q`; every query head would otherwise be re-assigned to a different KV head |
| checkpoint cannot be read | dangling sidecar symlink, mixed per-layer head counts | the config is never checked against anything | unreadable → hard failure. "I could not check" and "it is fine" must never be the same outcome |
| config is not `checkpoint + zero pad` | any config that simply disagrees with its weights | the config is self-consistent | head counts come from the tensors, not from the JSON |

### The second, stronger check

That check is arithmetic. The one that follows is evidence: after `load_weights`, read the fabricated rows of
every layer's `qkv_proj` — and the matching input columns of `o_proj`, which is row-parallel, so padded heads
appear there as columns — and fail if any element is non-zero. **Why that matters more than it looks.** A padded
head whose q/k/v rows are zero produces a zero attention output and a zero `o_proj` contribution: an exact
no-op. A padded head holding allocator garbage is a **fluent** drafter proposing slightly wrong tokens —
acceptance falls, nothing crashes, and no log line says why. A build gate cannot catch it either: it depends on
what the allocator happened to hand out at load time.

Measured at boot on the only rank that carries padding `[measured-here]`:

```
HAREM-TP3 drafter pad: 32/8 (checkpoint) -> 36/9 (config) at tp=3;
    rank 2 owns 8 real + 4 padded query heads
HAREM-TP3 drafter pad verified zero on rank 2/3: 26,214,400 elements
    across 4 padded query head(s) and 1 padded KV head(s)
```

**26,214,400** is exactly `5 layers × (4×128 q + 128 k + 128 v) × 4096 + 5 × 4096 × 512`, arrived at
independently by arithmetic before the boot — which is what makes it evidence rather than a log line agreeing
with itself. Offline unit tests before deployment: **11/11** `[measured-here]` — they read the real 32/8
checkpoint through both sidecars, accept 36/9 at TP=3 and 32/8 at TP=2, and refuse all five traps.

Both edits are no-ops when the config matches the checkpoint, so the patch is safe to leave applied at any TP.
The escape hatch `HAREM_TP3_DRAFT_PAD_CHECK=warn` downgrades the zero proof to a log line, for diagnosis only.
**Do not serve with it** — a drafter that needs it is a drafter whose proposals are not what you think they are.
Verify the anchors after any change:

```
python3 patches/tp3/patch-dflash-tp3.py --check
```

## 6. Draft depth: k=7 stays

Image `exl3-zeus:bc0e0f6s`, TP=3 + EP, `--max-num-batched-tokens 4096`, `gpu-memory-utilization 0.80`,
`--block-size 256`, `--max-num-seqs 8`, `kv-cache-dtype fp8`, temperature 0, reasoning effort **low**, one boot
per arm, only `num_speculative_tokens` differing. Gates 10/10 · 12/12 both arms. Two sweep values per cell, 5 September 2026 `[measured-here]`:

| | k=7 | k=5 | change |
|---|---|---|---|
| C1 | 51.31 / 52.10 | 48.39 / 48.46 | **−6.4 %** |
| C2 | 76.03 / 73.73 | 72.13 / 72.20 | −3.6 % |
| C4 | 101.75 / 98.97 | **105.36 / 105.40** | **+5.0 %** |
| C6 | 122.20 / 122.51 | 116.27 / 115.63 | −5.2 % |
| C8 | 144.31 / 144.73 | 135.83 / 143.14 | −3.5 % |
| acceptance rate | 62.9–65.3 % | **70.5–76.1 %** | +11 pts |
| accepted tokens per step | **5.40–5.57** | 4.53–4.80 | −15 % |
| TTFT C8 | 1.703 / 1.790 s | 1.582 / 1.756 s | −4 % |
| mixed load: decode while a 7K prompt lands | **10.4 tok/s** | 7.6 | −27 % |
| prefill, fresh (median) | 1,640 | 1,661 | prefill does not use the drafter |

Per category, C1 decode tok/s. The k=7 column comes from the `--max-num-batched-tokens 2048` arm, so there is a
small confound; category decode is decode-bound, so this is a fair read rather than a clean A/B
`[measured-here]`:

| | prose | code | math | json |
|---|---|---|---|---|
| k=7 | 22.0 (acc 13.0 %) | 47.3 (46.9 %) | 56.2 (54.7 %) | 54.0 (52.6 %) |
| k=5 | **22.8** (17.1 %) | 45.6 (56.9 %) | 52.2 (63.8 %) | 50.2 (61.3 %) |
| k=7, C4 total | 41.2 | 92.6 | **94.7** | 94.7 |
| k=5, C4 total | **47.5** | **97.1** | 89.1 | 95.3 |

**Reading.** A shallower draft raises the per-token acceptance *rate* (63 % → 73 %) but lowers the tokens
accepted per *step* (5.5 → 4.7), and the second effect wins wherever the target forward dominates: single-user
(−6.4 %) and loaded (−3.5 to −5.2 %). k=5 wins only in prose — the category whose acceptance is 13–17 %, where a
deep draft is mostly wasted — and at C4. The prose effect is real (+3.6 % at C1, +15 % at C4) but does not carry
the aggregate. **Per-request k is not available:** `num_speculative_tokens` lives in `--speculative-config` and
is fixed when the engine starts, and the OpenAI-compatible API has no per-request override. One value has to be
chosen for the whole engine, and it is **k=7**. A prose-heavy workload would want its own engine at k=5.

## 7. The character of the drafter

The same shape shows at TP=2 against the MTP baseline `[measured-here]`:

| category | MTP k=3 tok/s | DFlash2 k=7 tok/s | MTP acceptance | DFlash2 acceptance |
|---|---|---|---|---|
| math | 34.2 | **48.5** | 79.2 % | 58.7 % |
| json | 35.3 | **45.6** | 83.8 % | 53.2 % |
| code | 31.1 | **38.2** | 73.1 % | 46.5 % |
| prose | **21.3** | 18.5 | 37.7 % | 12.8 % |

DFlash2 beats MTP comfortably on structured text and **loses on prose**. Expected, not a defect: prose is
high-entropy, a 7-deep draft mostly misses there, and the wasted draft eats the gain. Budget for the prose row.

## 8. What we looked at and deliberately did not change

**The RoPE style question.** The fork carries a helper, `dflash_target_rope_is_neox_style()`, whose comment
describes precisely the symptom we had gone looking for — the mismatch is silent, "acceptance collapses, the
output stays correct" (fork source comment). GLM-5.3's attention uses `is_neox_style=False`; the base image's
drafter takes `get_rope`'s default, `True`. On paper that is the bug. **The measurement refuted it:** our
acceptance is 62.4 % at k=7, the same band as the known-good stack running the same drafter at the same depth
(62–65 %), and a real mismatch would sit well below it. A working, measured configuration was **not** changed on
a hypothesis. The patch exists and is a single command to A/B; the result to beat is 62.4 % at 5.37 tokens per
step `[not tested]`. **One upstream commit was not ported:** `fe755c889` decouples the draft's Gumbel noise
stream from the target's.
It is absent from the fork's HEAD and the known-good comparison configuration predates it. It is the **first
place to look if sampling-mode acceptance ever looks strange**: temperature 0 does not exercise that path, so
nothing here would have caught it `[not tested]`.

## 9. Settings that matter

- **`--block-size 256` is kept, on a rule that was read rather than guessed.** The open question was whether the
  draft path hits the storage-block constraint that forced block 256 for MTP. The binding rule is in
  `v1/core/kv_cache_coordinator.py`: every KV cache group participating in prefix caching must satisfy
  `block_size % hash_block_size == 0`, enforced by an `assert`. A DFlash drafter adds a **second** KV group
  (sliding-window attention, window 2048), so the multi-group path is live. 256 satisfies the rule, and an
  incompatible value fails loudly rather than silently, so it stays as the known-good value.
- **`kv-cache-dtype fp8` applies to the target only**; the draft's KV stays BF16 (`auto`). The draft group is
  0.6 % of per-block bytes — nothing to win by quantizing it, and a real acceptance risk.
- **A harmless scheduler warning.** At `--max-num-batched-tokens 2048` the scheduler warns about
  `max_num_scheduled_tokens` versus the k=7 draft slots. Cosmetic; raising the budget to 4096 removes it and is
  worth 9.5 % of prefill, at a KV price — [07](07-kv-and-draft-page.md)'s subject.

## 10. What this cost

DFlash2 is the largest single decode win in this stack, and it is not free `[measured-here]`:

| axis | MTP k=3 reference | DFlash2 k=7 | price |
|---|---|---|---|
| Quality gates | 10/10 · 12/12 | 10/10 · 12/12 | none found; looked for, cold and after a full benchmark |
| C1 / C8 decode | 30.49 / 102.32 | 42.91 / 114.60 | the gain |
| Prose decode, C1 | 21.3 | 18.5 | **−13 %** — the one category that is worse (§7) |
| Prefill, 7K | 1,131 | 1,035 | **−8.5 %** at TP=2 |
| KV pool | 1,987,179 tokens | **825,000** | **−58 %** |
| Weights per node | — | +2.34 GB drafter | swap rose ~1.1 GB, then stopped; free RAM level with the MTP run |
| Licence | none beyond the target | CC BY-NC-ND 4.0 | non-commercial, no derivatives; our permission does not transfer ([../LICENSES.md](../LICENSES.md)) |

**The KV pool is the item to understand.** The draft's own cache group explains only **0.6 %** of that drop.
Most of the rest was memory scarcity at TP=2 — 81.53 GiB of weights per node left 17.27 GiB for KV — and the
third node resolved it: the headline configuration runs a **4,699,724-token** pool with the same drafter at k=7
([README](../README.md), [10](10-results-and-roofline.md)). The remainder, a counter rather than a byte, is
[07](07-kv-and-draft-page.md).

## 11. Rebuild and verify

Build the port layer on **every node**, over the base image from [02](02-image-build.md):

```
docker build -t exl3-zeus:dflash ~/exl3-zeus/patches/dflash2-port
```

The build runs the gate and **fails if any ported symbol does not resolve**, so a successful build is itself the
first check. The gate stays in the image and can be run standalone at any time:

```
docker run --rm --entrypoint python3 exl3-zeus:dflash /opt/harem/dflash2-gate.py
```

It prints `DFLASH2 PORT BUILD GATE: OK` and exits 0. Then boot and confirm the lines that prove the port is
engaged, in order: auxiliary layers (§3.1), the draft KV group (§3.2), and — at TP=3 only — the two pad lines
(§5). Run the quality gates before believing any speed number, and measure by [09](09-measurement-protocol.md),
not by two rounds. What is still open here is in [11 — Open issues](11-open-issues.md). Next:
[05 — Expert parallel and the cuda-exl3 fixes](05-expert-parallel-and-cuda-exl3-fixes.md), or
[07 — KV pool and the draft page](07-kv-and-draft-page.md) if the KV row above is what you came for.

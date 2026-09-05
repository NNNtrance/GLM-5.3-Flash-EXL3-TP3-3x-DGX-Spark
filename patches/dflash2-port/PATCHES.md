# DFlash2 port → cuda-exl3 stack (`exl3-zeus:serve` → `exl3-zeus:dflash`)

Goal: run the `incoai/GLM-5.3-Flash-DFlash2` drafter (architecture
`DFlash2DraftModel`) on the EXL3 stack, which is the official image
`vllm/vllm-openai:glm53-flash-arm64-cu130` (vLLM `0.1.dev20051+g487ecf187`)
plus Zeuss5/cuda-exl3.

## How the port was derived (not hand-copied)

The base image already ships **DFlash v1**. The DFlash2 work is a delta on top
of a very similar tree, so this is a real port rather than a file copy:

1. The image's vLLM tree was extracted with `docker create` / `docker cp` and
   compared against the LIL fork checkout at `~/exl3-zeus/vllm-fork`.
   **The image is byte-identical to fork commit `487ecf187`** for every file
   involved (verified per file, diff = 0). That makes a real 3-way merge possible.
2. Git topology: `487ecf187` (image, 25 Aug) is **not** an ancestor of the fork's
   HEAD `9c4dd0548`. Their merge base is `b908a21f9` (13 Aug) — the two lines
   diverged, so a straight cherry-pick of fork HEAD would have dragged in
   unrelated drift.
3. Therefore each modified file was merged with
   `git merge-file <image-version> <b389ac294^> <b389ac294>`, i.e. the *upstream
   DFlash2 change alone* applied onto the image's own tree.
   5 of 7 merged clean; 2 conflicted trivially and were resolved by hand (below).

**Net result: this port is upstream vLLM commit `b389ac294`
("[Spec Decode] DFlash2: local convolution + candidate selector", #52816),
nothing more.** The LIL fork's own DFlash2 changes were examined and are limited
to the MXFP8 path, which does not apply here (see "Deliberately not ported").

## Files

### New, verbatim upstream `b389ac294`

| File | What it is |
|---|---|
| `vllm/model_executor/models/qwen3_dflash2.py` | The DFlash2 draft head. Adds `DFlashGroupedConv` (the two-tap dynamic convolution that keeps the draft from decaying toward the end of a block), `CandidateSelector` (low-rank predecessor/successor codebooks that score edges between per-position candidates), `DFlash2Qwen3DecoderLayer` (wraps attention and MLP in `prepare`/`finish` convolutions), and `DFlash2Qwen3ForCausalLM.compute_candidates`. **Plus HAREM fail-closed guards — the only local change, see below.** |
| `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` | `DFlash2Speculator`, a subclass of the base image's `DFlashSpeculator`. Overrides `_generate_draft` to compute top-k candidates per position, score edges with the selector, and walk one coherent path (`_selector_walk_kernel`, Triton). `_cache_draft_logits_kernel` writes only the K candidate columns of the cached proposal distribution. **Byte-identical between upstream `b389ac294` and fork HEAD — the fork never modified it.** |
| `vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py` | Package marker. |

### Merged (upstream DFlash2 delta applied onto the image tree)

| File | Delta | Why DFlash2 needs it |
|---|---|---|
| `vllm/model_executor/models/registry.py` | +1 | Registers `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")`. **Conflicted** — resolved below. |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | +6 | `init_speculator` dispatches to `DFlash2Speculator` when the drafter's architecture is `DFlash2DraftModel`. **Conflicted** — resolved below. |
| `vllm/config/vllm.py` | +17 | `_is_dflash2_draft()` forces the **V2 model runner** for a DFlash2 drafter. Load-bearing: on V1 the same checkpoint drafts through `DFlashProposer`, which never calls the candidate selector, so the draft silently degrades to DFlash1 with no error. |
| `vllm/v1/worker/gpu/spec_decode/speculator.py` | +16 −5 | Adds the `draft_logits_spec()` hook so a speculator can choose the dtype/fill of the cached proposal distribution. DFlash2 overrides it to fp32 / `-inf` because its kernel writes only the K candidate columns; the base's `torch.zeros` + `head_dtype` would leave non-candidates at probability 0 rather than excluded, and the walk and the rejection sampler would read different distributions. |
| `vllm/v1/worker/gpu/sample/gumbel.py` | +52 −34 | Pure refactor: extracts `gumbel_noised_argmax` out of `gumbel_block_argmax` so the DFlash2 selector-walk kernel can reuse the exact same Gumbel-max draw. Sharing it is what lets the draft and the target's verification agree on noise. `gumbel_block_argmax` now calls the extracted helper — behaviour unchanged. |
| `vllm/model_executor/layers/logits_processor.py` | +81 | Adds `LogitsProcessor.get_top_k_tokens`: a vocab-parallel top-k that all-gathers `2k` values instead of the full vocabulary. DFlash2 calls it once per drafted position. **Plus one HAREM knob, see below.** |
| `vllm/model_executor/models/qwen3_dflash.py` | +11 −4 | Subclass hooks only: `decoder_layer_cls` / `model_cls` class attributes so `DFlash2Qwen3Model` can substitute its own layer and model classes, and `_dflash_layer_causal` now honours an explicit `is_causal` on the config (our drafter's `config.json` sets `"is_causal": false`). No change to DFlash v1 behaviour. |

### Conflict resolutions (2)

Both conflicts were *unrelated upstream features* present on the DFlash2 branch
but absent from the image's branch. Only the DFlash2 lines were taken.

1. **`registry.py`** — upstream also added Muse-Glimmer drafter aliases
   (`MuseGlimmerAssistantModel`, `DFlashMuseGlimmerAssistantModel`) next to the
   DFlash2 entry. Dropped: that model does not exist in the base image and
   registering it would be dead weight. Kept only the `DFlash2DraftModel` line.
2. **`v1/worker/gpu/spec_decode/__init__.py`** — upstream also added an
   `extract_hidden_states` speculator branch. Dropped: that method and its module
   do not exist in the base image, so the branch would be a broken import.
   The DFlash2 dispatch was inserted inside the image's existing `dflash` branch.

## Deliberately NOT ported

| Fork commit | What it does | Why not |
|---|---|---|
| `da4d7be6c` `feat(dflash2): execute serialized MXFP8 projections natively` | Threads the draft `quant_config` into `DFlashGroupedConv` / `CandidateSelector`, and builds the fused context K/V projection from serialized MXFP8 values + E8M0 scales. | **B12X/NVFP4-specific.** Our drafter is BF16 (`"dtype": "bfloat16"`, 2.34 GB safetensors), so `quant_config` is `None` and the upstream (`quant_config=None`) form is *behaviourally identical*. Porting it would drag ModelOpt MXFP8 machinery in for no gain. Replaced by a **fail-closed guard** (below) so a quantized drafter cannot slip through silently. |
| `a9a17e709` `Dflash2 load fix (#53435)` | Adds `decoder_layer_cls` + `self.decoder_layer_cls` to `DFlashQwen3Model`. | **Redundant.** It is a re-land of hooks that `b389ac294` already adds (and `b389ac294` adds `model_cls` too, which this commit does not). Verified: the merged file already contains both hooks. |
| `fe755c889` `[Bugfix] Decouple the draft's gumbel noise stream from the target's (#54282)` | Upstream follow-up, 29 Aug, touches `dflash2/speculator.py`. | **Not in the fork's HEAD** (`git merge-base --is-ancestor` → NO). The known-good NVFP4 production result (DFlash2 k=7, acceptance 62–65%) was produced *without* it, so including it would make this port diverge from the configuration we are comparing against. Noted here as a candidate follow-up if sampling-mode acceptance looks wrong. |
| B12X / NVFP4 fork patches generally (TP=3 draft padding 36/9 in the loader, `b12x` kernels, GLM-5.3 backend integration) | — | Out of scope; the base image already has its own GLM-5.3 support. |

## HAREM-local changes (3, all additive)

1. **`qwen3_dflash2.py` → `DFlash2Qwen3Model._harem_check_port_assumptions()`**,
   called from `__init__`. Two fail-closed checks:
   - **Quantized drafter refused.** If `self.quant_config is not None`, raise
     `NotImplementedError`. Without this, an MXFP8 drafter would get BF16
     convolution/selector projections over quantized weights and produce a
     degraded draft with *no error*. The docstring names the exact commit
     (`da4d7be6c`) and hunks to port if that is ever wanted.
   - **Head counts must divide the TP size.** `num_attention_heads` and
     `num_key_value_heads` must both be divisible by
     `tensor_parallel_size`. This checkpoint copy is the TP=2 one (32/8); the
     stock incoai drafter is 36/9 (divides by 3, not by 2).
     **This is the hook for the TP=3 task (separate agent):** TP=3 needs the
     36/9 drafter or loader-side head padding, and this check is where a
     wrong-shaped drafter stops instead of silently degrading. Directly
     motivated by the NVFP4 root cause — a decode kernel that was silently wrong
     at 22 heads because TP=3 was an untested shape.
2. **`logits_processor.py` → `HAREM_DFLASH2_FORCE_TORCH_TOPK=1`** env knob in
   `_flashinfer_topk()`. Falls back to `torch.topk` instead of the FlashInfer
   radix kernel. Default is unchanged upstream behaviour; this exists purely as a
   diagnosis lever, because the DFlash2 candidate selector is the only caller of
   this top-k, it runs over a 154 880-entry vocabulary, and GB10/sm_121 has
   needed shared-memory workarounds elsewhere in this stack (cf. the
   `persistent_topk` overlay). If acceptance comes out far below the NVFP4
   stack's 62–65%, this is the first thing to flip.
3. **`/opt/harem/dflash2-gate.py`** — the build gate, kept in the image so it can
   be re-run any time:
   `docker run --rm --entrypoint python3 exl3-zeus:dflash /opt/harem/dflash2-gate.py`

## Settings notes

- **`--block-size 256` is kept.** The question was whether the draft path hits
  the DeepGEMM storage-block-64 constraint that forced block 256 for MTP. The
  binding rule was located in
  `v1/core/kv_cache_coordinator.py`: every KV cache group that participates in
  prefix caching must satisfy `block_size % hash_block_size == 0`, enforced by an
  `assert`. A DFlash drafter adds a **second** KV cache group (its sliding-window
  attention, window 2048) alongside the target's, so the multi-group path is
  active. 256 satisfies the rule, and any incompatibility fails loudly rather
  than silently — so 256 is kept as the known-good value.
- The `HAREM_DISABLE_PERSISTENT_TOPK=1` overlay (`sparse_attn_indexer*.py`, bind
  mounted from `~/exl3-zeus/overlay`) is untouched and still mounted.
- `gpu-memory-utilization 0.85`, no KV pin.

## Reproducing the build

```bash
cd ~/exl3-zeus/dflash2-port
docker build -t exl3-zeus:dflash .          # on every node
```
The build fails if any ported symbol does not resolve.

---

# Part 2 — what the DFlash2 delta alone did NOT cover

Porting `b389ac294` made the drafter *loadable*, not *runnable*. Two further
things were missing from the base image, both found by booting and reading the
traceback, and both fixed with the same 3-way-merge / measure-first discipline.

## 2a. Target side: GLM-5.3 had no EAGLE3 aux-hidden-state interface

**Symptom:** `RuntimeError: Model does not support EAGLE3 interface`
(`v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:19`), after weights loaded.

**Cause:** DFlash is not self-contained. The drafter reads the *target's*
intermediate layers (`dflash_config.target_layer_ids = 5, 14, 24, 33, 42`), so
the target class must expose `SupportsEagle3`. The base image's
`vllm/models/glm5next/nvidia/model.py` contains **zero** aux-hidden-state
support (`grep -c aux_hidden_state` → 0) and matches commit `487ecf187` exactly.
So DFlash **v1** could not have worked on this image either — this is not a
DFlash2-specific gap.

**Fix:** 3-way merge of fork commit `e7097feb6` *"Support GLM-5.3 DFlash
speculation"* (Luke Alonso) onto the image's `model.py`. Not B12X-specific — it
is precisely the target-side enablement. Adds:
- `Glm5NextModel(nn.Module, EagleModelMixin)` + `self.dflash_capture`
- `_prepare_aux_hidden_state()` — for mHC layers, runs `hc_post` then
  `hc_contract` when drafting (the drafter wants the contracted form)
- the forward loop collects aux states at `layer_idx + 1 in aux_hidden_state_layers`
  and returns `(hidden_states, aux_hidden_states)`
- `Glm5NextForCausalLM` and `Glm5NextForConditionalGeneration` declare `SupportsEagle3`

**2 conflicts, both base-image drift, not DFlash:** the fork had dropped
`SupportsPP` from this branch. Resolution: **keep the image's `SupportsPP`** and
only *append* `SupportsEagle3`. The fork's neighbouring lines
(`supports_pp: ClassVar[Literal[False]] = False`, `get_model_state_cls()`) come
from *other* fork commits and reference `..model_state`, which does not exist in
this tree — dropped.

Verified at runtime: `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)`
— exactly `target_layer_ids + 1`.

## 2b. KV cache: the draft's group broke GLM-5.3's specialised grouping

**Symptom:** `NotImplementedError: Layer language_model.model.layers.3.self_attn.indexer.k_cache:
page size is not divisible by the maximum page size and cannot be padded.`

**Measured, not guessed.** A throwaway image (`exl3-zeus:dflash-diag`) logged
every layer's spec at the top of `unify_kv_cache_spec_page_size` — one pass, 72
layers (45 target + 11 indexer k + 11 kpool tail + 5 draft):

| spec | page (B) | block | per token | pad-able? |
|---|---|---|---|---|
| `MambaSpec` (KDA) + `MLAAttentionSpec` | 2,359,296 | 4608 | 512 B | — |
| `MLAAttentionSpec` indexer `k_cache` | 152,064 | 4608 | **33 B** | **no** (`indexes_kv_by_block_stride=False`) |
| `SlidingWindowSpec` (**draft**) | 32,768 | 16 | 2048 B | yes |
| `KpoolTailSpec` | 2,048 | 4 | 512 B | no |

33 B/token can never reach 512 B/token by integer block scaling, and the
indexer's backend cannot pad. So why does the target work on its own? Because
GLM-5.3 normally takes a **dedicated grouping path**,
`_get_kv_cache_groups_glm5_next` — in which mamba layers co-own the MLA slot
tensors and the kpool tail co-owns the indexer tensors (an image-specific memory
optimisation) — and that path **never calls page unification**. It requires
every attention spec to be an `MLAAttentionSpec`; the draft's sliding-window
layers break that test, the model falls through to the generic multi-group path,
and dies there.

**Fix — the fork's idea, adapted to this image's architecture.** The fork solves
this with `_partition_dflash_draft_specs`: split the draft's KV layers from the
target's, group each independently, concatenate. The fork's own
`kv_cache_utils.py` is otherwise restructured (DCP replication, split-cache env
var, `extra_retained_tokens`) and has neither
`_get_kv_cache_groups_glm5_next` nor `_glm5_next_tensor_layout`, so the file was
**not** taken wholesale — only the idea.

Implemented in `vllm/v1/core/kv_cache_utils.py` (13 single-match anchored edits):

1. `_harem_partition_dflash_draft_specs()` — splits by layer index (the DFlash
   head is built with `start_layer_id = target layer count`, so its layers are
   index ≥ 45). Returns early (no split) unless `method == "dflash"`, PP == 1 and
   the hybrid manager is enabled.
2. `get_kv_cache_groups()` recurses: group target alone, group draft alone,
   concatenate. **The target's grouping is therefore bit-for-bit what it is with
   no drafter at all.**
3. `_glm5_next_tensor_layout()` now also detects and *returns* the draft
   group(s) (9-tuple instead of 8), and fails closed if a group that is neither
   MLA-attn, kpool-tail nor mamba is anything other than a plain `AttentionSpec`.
4. All three consumers of that layout account for the draft group, via the new
   `_harem_draft_bytes_per_block()`:
   - `_pool_bytes_per_block()` — per-block cost
   - `get_kv_cache_config_from_groups()` — per-block cost **and** one private
     `KVCacheTensor` per draft layer
   - the max-memory estimate

   Omitting any of these would over-report how many blocks fit and leave the
   draft's tensors unallocated — a silent, dangerous failure. The build gate
   asserts all three unpack the draft group.

Cost of the draft group: `5 × 32,768 = 163,840 B` per block, against the
target's `11 × 2,359,296 + 11 × 152,064 = 27,624,960 B` — **+0.6 %**.

Verified at runtime: `DFlash draft: 5 KV layers kept in 1 independent cache group(s)`.

## Updated file list

Added to the port on top of Part 1:

| File | Origin |
|---|---|
| `vllm/models/glm5next/nvidia/model.py` | 3-way merge of fork `e7097feb6` |
| `vllm/v1/core/kv_cache_utils.py` | HAREM-local, fork's partition idea |

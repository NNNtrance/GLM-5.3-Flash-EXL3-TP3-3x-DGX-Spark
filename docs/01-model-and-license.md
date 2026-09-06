# 01 — Model and licence

**Applies to: both tracks.** The same two checkpoints and the same two licences at either rank count.

Which checkpoints this recipe serves, why those, what their licences actually say, and the two
structural facts about EXL3 weights that dictate every parallelism decision downstream.

**Read this page before you download anything.** The production checkpoint is MIT; the fallback's
licence is not MIT, not Apache and not any licence you have met before; and the speculative draft
model is non-commercial.

**There are two checkpoints, and which one you want depends on your image.** The production
checkpoint since 5 September evening is the **full-scope** one (§1): it quantizes the dense path and
the head as well as the routed experts, and it is worth **+21.7 % per stream** with the quality gate
passed. It needs a `cuda-exl3` carrying the padded-load path (`f3e3090` + `754421f`) and the loader
patch in `tracks/tp3/patches/`. If your image predates either, the **routed-experts-only** checkpoint
(§1.1) is the fallback, and it is also our rollback: every configuration from 1 to 8 served it.

---

## 1. The production checkpoint — full scope

| | |
|---|---|
| Repository | `turboderp/GLM-5.3-Flash-exl3` on Hugging Face, branch `4.05bpw` |
| Revision we ran | `2a30229e67012798ba9f0cd832bb78abf4c363d5` (short `2a30229e`, 28 August 2026) |
| Size on disk | **165.2 GB / 153.8 GiB** across 19 shards, plus `mtp.safetensors` (3.79 GB, unused), a 47.9 MB `quantization_config.json` and a 16.0 MB index `[measured-here]` |
| Verified | `sha256` **23/23** against the repository's own LFS metadata, independently on every node `[measured-here]` |
| Format | exl3 v1.4.4, codebook `mul1`, `bits: 4.05`, `head_bits: 6`, `out_scales: always`, calibration 250 rows × 2,048 columns |
| Scope | **full** — routed experts at 4 bits; shared experts, KDA `qkv_proj`, every `o_proj`, the MLA projections, the indexer's `wq_b` and `lm_head` at 5–6 bits. Only four families stay BF16: `f_b_proj`, `g_b_proj`, `in_proj_bfg_a` and `kv_b_proj`, plus the norms, the router, `embed_tokens` and the three `conv1d` |
| **Licence** | **MIT** `[reported]` — the `LICENSE` file is the MIT text, "Copyright (c) 2026 Z.AI Co., Ltd", and the card carries `license: mit`. No attribution condition, no exclusion clause, OSI-approved. **More permissive than the fallback in §1.1.** Read it yourself; different publisher, different terms |
| What resolves to EXL3, measured | **203 EXL3 linears / 113 BF16** per rank, plus 42 routed-expert MoE layers, read off `CUDA_EXL3_DEBUG_NAMES=1` on the live boot `[measured-here]` |

```
huggingface-cli download turboderp/GLM-5.3-Flash-exl3 --revision 2a30229e67012798ba9f0cd832bb78abf4c363d5 --local-dir /var/tmp/glm-5.3-flash-turboderp-4.05bpw
```

**It is not smaller than the fallback by much, and that surprised us.** The routed experts are
already 4-bit in *both*, so full scope only takes the remaining ~11 % of the weights from BF16 down
to 4–6 bits: 164 GiB → 154 GiB. **The gain is in decode traffic, not on disk** — 17.8 ms off an
88.2 ms decode step — and the 10 GiB saved on disk is repaid again in host memory: 3.4 GiB less
consumed per node, which is +10.0 % of KV pool ([13](13-full-scope-checkpoint.md) §7.2).

**What it costs.** Draft acceptance −2.4 points (64.4 → 61.9 %, gate ≥60 %) and accepted tokens per
step −3.0 % (5.50 → 5.34), because the 6-bit `lm_head` perturbs a drafter trained against a BF16 one.
Both are paid for several times over by the step time. Quality is unchanged: MMLU sample
**86.47 ±0.74** against the control's 86.4 ±0.7 `[measured-here]`.

`mtp.safetensors` in that repository is an MTP drafter that is **not** in the safetensors index, so
vLLM never reads it. Not evaluated `[not tested]`.

## 1.1 The fallback checkpoint — routed experts only

This is what production configurations 1–8 served, and it is the rollback path: it needs no
padded-load support in the plugin, because at TP=3 nothing that gets padded is EXL3 in it.

| | |
|---|---|
| Repository | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` on Hugging Face |
| Revision we ran | `b20c49ba9ecafb563099536e307d21c1310e1c49` (short `b20c49ba`, 30 August 2026) |
| Size on disk | 175.6 GB (`175,642,157,752` bytes) across 120 safetensors shards `[measured-here]` |
| Format | EXL3 trellis, 4 bits per weight, codebook `mcg` |
| Scope | `quantization_config.scope: glm53_routed_experts_only` — the routed experts are EXL3; attention, KDA, the shared expert and `lm_head` stay BF16. **Until 5 September this was the only scope that could be loaded at all** on the NVIDIA `glm5next` reader, for reasons that have nothing to do with the weights — see §3.2 and [13](13-full-scope-checkpoint.md) |
| Base model, per its card | `zai-org/GLM-5.3-Flash-BF16` `[reported]` |
| Quality, per its card | KL divergence against BF16 **0.0246 nats** over 51,175 positions `[reported]` |

`b20c49ba` is **not** the current `main` of that repository — as of 5 September 2026 `main` is
`aba59d21` (4 September 2026), which is four days newer than what we ran. We have not tested the
newer revision `[not tested]`. Pin the revision when you download:

```
huggingface-cli download brandonmusic/GLM-5.3-Flash-tr3-4bpw --revision b20c49ba9ecafb563099536e307d21c1310e1c49 --local-dir /var/tmp/glm-5.3-flash-tr3-4bpw
```

Verify what you got before you build anything on it. The repository ships a `SHA256SUMS`, and the
fast-load sidecar in [docs/08](08-fast-boot.md) hashes it into the manifest identity, so a checkpoint
that quietly changed underneath you stops a boot rather than serving different weights.

```
cd /var/tmp/glm-5.3-flash-tr3-4bpw && sha256sum -c SHA256SUMS
```

### Why these two and not another

| Candidate | Verdict |
|---|---|
| `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw — **full scope**, `mul1` codebook | **What we run in production**, at TP=3 on three nodes since 5 September evening. C1 **+22.9 % total / +21.7 % per stream** against the fallback on the same three nodes, C8 +12.5 %, TTFT −9…−19 %, KV pool +10.0 %, quality unchanged `[measured-here]`. It needed a five-layer loader patch and a padded-load path in the plugin, none of which is about quantization (§3.2, [13](13-full-scope-checkpoint.md)). |
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (routed experts only, `mcg` codebook) | **The fallback and the rollback.** Everything outside the routed experts is BF16, so ordinary tensor parallelism applies to attention and KDA and nothing that TP=3 pads is EXL3 — which is why it loads on any image. Its module names are also the ones the NVIDIA `glm5next` reader expects, which turned out to be a larger part of why it works than the scope is ([13](13-full-scope-checkpoint.md) §2.3). |
| Higher-precision EXL3 packages, and other mixed per-layer bitrates | Not evaluated `[not tested]`. The 4.05 bpw package is already mixed — 4-bit routed experts, 5–6 bit dense and attention, 6-bit head — and its quality gate passed twice, so this is a refinement rather than an alternative. |

Independent panels put EXL3 at ~4 bits in the same KL band as FP8 for this model family, and well
ahead of 4-bit NVFP4 and int4 AWQ `[reported]`. Our own comparison is narrower and honest about it:
an MMLU sample of 1,995 questions scored **86.47 ±0.74** on the production checkpoint at TP=3 and
**86.4 ±0.7** on the fallback, against 86.7 for our NVFP4 production stack — all three inside the
noise `[measured-here]`. The 6-bit `lm_head` was the one place we had flagged for damage, and it has
now been measured at both TP=2 (86.32) and TP=3 (86.47): **no measurable cost**.

---

## 2. The fallback's licence: ShapleyMCG License 1.0

This section is about `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (§1.1). The production checkpoint (§1) is
MIT, and if you never run the fallback none of this applies to you.

The model card carries `license: other` with `license_name: shapleymcg-license-1.0`. There is a real
`LICENSE` file in the repository and you should read it yourself — it is short.

What it says, as we read it `[reported]`:

- **Commercial use is permitted**, for everyone, without a separate agreement.
- **Modification and redistribution are permitted.**
- **Attribution is mandatory** — it is written as a condition of the grant, not as a courtesy.
- **One named individual is excluded from every right the licence grants.** The licence identifies
  that person by name and by account, extends the exclusion to their agents and heirs, adds an
  anti-circumvention clause, and terminates automatically on breach. The stated reason is a dispute
  over unattributed reuse.
- The licence text says of itself that it does not meet the Open Source Definition, and it specifies
  a US state jurisdiction.

**What that means for you.** For almost every reader this is a permissive licence with an attribution
requirement, and it is more permissive than the draft model you will pair it with. But it is
**not an OSI-approved open-source licence**, so do not treat it as MIT-equivalent in a compliance
review, and read the exclusion clause to confirm it does not name you. We do not mirror or
redistribute these weights; download them from the author's repository under terms that apply to you.

Full detail, with links, in [../LICENSES.md](../LICENSES.md) and [../CREDITS.md](../CREDITS.md).

---

## 3. Two structural facts that decide everything downstream

### 3.1 An EXL3 tensor cannot be split three ways

EXL3 stores weights as a trellis. The storage granularity is 16 elements, but *correctness* is tied
to the 128-element Hadamard block the codebook was fitted against: a slice that is not a multiple of
128 decodes to noise rather than to an approximation `[reported]`. On this model the routed expert
width is `moe_intermediate_size = 2048`, and `2048 / 3` is not an integer, let alone a multiple of
128.

So at TP=3 the routed experts **cannot be tensor-sliced**. The way out is expert parallelism: keep
each expert whole and give each rank 96 of the 288. That is not an optimisation here, it is the only
arrangement that is arithmetically legal, and it is why `--enable-expert-parallel` is mandatory in
this recipe and why `ENABLE_EP=0` at TP=3 is refused by the preflight
([docs/03](03-tp3-padding-and-sidecars.md)).

A zero-extended trellis is not a zero-extended weight. Padding is the right tool for the BF16 parts
of this model and the wrong tool for the quantized parts, and the preflight enforces that distinction
rather than trusting a comment.

### 3.2 The BF16 half was the biggest single cost, and production 9 is what removed it

On the fallback checkpoint (§1.1), `scope: glm53_routed_experts_only` means attention, the KDA
layers, the shared expert and `lm_head` are unquantized. Measured on the live server with a torch
profiler, those BF16 dense GEMMs were **45.3 % of a single-stream decode step** — the largest class,
ahead of the EXL3 MoE GEMM at 29.7 % — **21.1 % of a C8 step and 17.4 % of a prefill chunk**
`[measured-here]` ([10](10-results-and-roofline.md) §5.3). The full-scope checkpoint (§1) takes that
class to 4–6 bits, and the arm is worth **17.8 ms of an 88.2 ms decode step** at TP=3
`[measured-here]`.

Five things worth holding on to, three of which are still true whichever checkpoint you serve:

- Kernel work inside `cuda-exl3` could not touch the largest item in the decode profile. Everything
  in that column of the ranked target table comes to about 5 %; this one item was priced at **~+34 %
  single-stream** `[estimate]`, measured at **+24.3 % per stream at TP=2** and **+21.7 % at TP=3**
  `[measured-here]` — the estimate was an upper bound, and the reason it overshot is that even a
  full-scope checkpoint leaves four families in BF16 ([13](13-full-scope-checkpoint.md) §4.2).
- Each rank's share of whatever stays BF16 is a *third* rather than a half, so those tensors get
  **less** efficient per rank as ranks are added. That is still true of the 113 BF16 linears the
  production checkpoint keeps; it is smaller than it was, not gone, and it is not a bug.
- **The scope of the fallback checkpoint was not a quality decision, and we were wrong to describe it
  as one** `[retracted]`. Two lines in vLLM's `glm5next` model file pin the whole attention stack to
  BF16 whatever the weights contain — `quant_config=None` for the MLA projections
  (`model.py:331`) and a `quant_config` strip in the KDA constructor (`kda.py:171-174`) — and
  between them they lock **72.8 %** of the dense traffic. Until those are conditional, no checkpoint
  of any scope can put attention on EXL3, so `routed_experts_only` was the only thing that could
  load ([11](11-open-issues.md) §1.9 row 29, [13](13-full-scope-checkpoint.md) §2.2). Our patch makes
  both conditional, behind one environment variable.
- The TP=3 obstacle turned out to be softer than §3.1 implies, and this is the sentence to carry
  away: **an EXL3 tensor cannot be zero-extended, but it can be loaded narrow into a parameter vLLM
  has padded**, with `svh = 0` on the pad — provided the pad occupies whole 128-column blocks,
  because the output Hadamard mixes across each block before `svh` is applied. Our head pad already
  did (64 → 66 heads = 256 columns = 2 whole blocks); our vocab pad did not (192 = 1.5 blocks) and
  does at `padding_size=384`. That is the whole of the TP=3 port on the arithmetic side
  ([03](03-tp3-padding-and-sidecars.md) §1.1).
- What still stays BF16 in the production checkpoint, measured rather than inferred: the KDA
  `f_b_proj`, `g_b_proj` and `in_proj_bfg_a`, and MLA `kv_b_proj` — **113 linears against 203 EXL3**
  `[measured-here]`. Quantizing the KDA gating arms would be a checkpoint-side change, not ours
  ([11](11-open-issues.md) §2.25).

---

## 4. The speculative draft model

| | |
|---|---|
| Repository | `incoai/GLM-5.3-Flash-DFlash2` |
| Size | 2.3 GB, BF16, 5 draft layers over 45 target layers |
| Depth we run | `k = 7` — see the A/B in [docs/04](04-dflash2-port.md) |
| Licence | **CC BY-NC-ND 4.0** — non-commercial, no derivatives, attribution required |

The draft is worth a great deal: at a single stream it is the difference between roughly 20 tok/s and
roughly 54 tok/s `[measured-here]`. It is also the component with the most restrictive licence in the
whole stack.

**Read [../LICENSES.md](../LICENSES.md) before you use it.** In short: we hold a project-specific
written permission from the author for our own use. That permission is **non-transferable** and does
not extend to you; we do not redistribute the weights; and the padded drafter config this recipe
describes is a change you make to your own copy. If your use goes beyond what CC BY-NC-ND 4.0 allows
you — commercial serving, or publishing a modified draft — obtain your own permission first.

The whole recipe runs without the draft. Set `SPEC_METHOD=` empty in your env file and you lose
roughly the speed the draft buys.

---

## 5. What is next

[02 — Image build](02-image-build.md), then [03 — TP=3 padding and sidecars](03-tp3-padding-and-sidecars.md),
which turns §3.1 into a working configuration. For §3.2 — why the production checkpoint would not
load, the loader patch, the TP=3 padded-load port and what the dense stage is worth, measured twice —
[13 — The full-scope checkpoint](13-full-scope-checkpoint.md).

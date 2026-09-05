# 01 — Model and licence

Which checkpoint this recipe serves, why that one, what its licence actually says, and the two
structural facts about EXL3 weights that dictate every parallelism decision downstream.

**Read this page before you download anything.** The checkpoint's licence is not MIT, not Apache and
not any licence you have met before, and the speculative draft model is non-commercial.

---

## 1. The checkpoint

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

### Why this checkpoint and not another

Two other EXL3 packages of this model exist, and we tried the obvious one first.

| Candidate | Verdict |
|---|---|
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (routed experts only, `mcg` codebook) | **What we run in production**, at TP=3 on three nodes. Everything outside the routed experts is BF16, so ordinary tensor parallelism still applies to attention and KDA, and the experts can be distributed whole. Its module names are also the ones the NVIDIA `glm5next` reader expects, which — as it turns out — is a larger part of why it works than the scope is ([13](13-full-scope-checkpoint.md) §2.3). |
| `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw — **full scope**, `mul1` codebook, attention, KDA, the shared experts and `lm_head` quantized too | **Loaded and measured on 5 September, at TP=2, and the dense-stage lever is real: +24.3 % per stream, +26.4 % aggregate at a single stream, MMLU 86.32 ±0.75 against the control's 86.4 ±0.7** `[measured-here]`. It needed a three-layer loader patch first, none of whose layers is about quantization (§3.2, [13](13-full-scope-checkpoint.md)). **Not a production candidate at TP=2** — it leaves a 31k-token KV pool, which closes the long-prompt path entirely. TP=3 needs a padded-load path for the head that does not exist yet ([13](13-full-scope-checkpoint.md) §7, [11](11-open-issues.md) §2.22). |
| Higher-precision EXL3 packages, and mixed per-layer bitrates (higher on attention and the head, 4-bit on the experts) | Not evaluated `[not tested]`. The 4.05 bpw package above is already mixed — 4-bit routed experts, 5–6 bit dense and attention, 6-bit head — and its quality gate passed, so this is now a refinement rather than a fallback. |

**Licence, for the full-scope candidate: MIT** `[reported]`. Verified on 5 September: the
repository's `LICENSE` file is the MIT text, "Copyright (c) 2026 Z.AI Co., Ltd", and the model card
carries `license: mit`. That is **more permissive than the checkpoint we run** — no attribution
condition, no exclusion clause, and OSI-approved, none of which is true of the ShapleyMCG License 1.0
in §2. Read it yourself before you download; different publisher, different terms.

Independent panels put EXL3 at ~4 bits in the same KL band as FP8 for this model family, and well
ahead of 4-bit NVFP4 and int4 AWQ `[reported]`. Our own comparison is narrower and honest about it:
an MMLU sample of 1,995 questions scored **86.4 ±0.7** on this checkpoint at TP=2, against 86.7 for
our NVFP4 production stack — a difference inside the noise `[measured-here]`. We did not re-run MMLU
at TP=3, because the quality gates (correctness probe 10/10, code exam 12/12, cold and warm) are
identical between the two arrangements and a full MMLU is hours of cluster time.

---

## 2. The licence: ShapleyMCG License 1.0

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

### 3.2 Most of the model is still BF16, and that is now the biggest single cost

`scope: glm53_routed_experts_only` means attention, the KDA layers, the shared expert and `lm_head`
are unquantized. Measured on the live server with a torch profiler, those BF16 dense GEMMs are
**45.3 % of a single-stream decode step** — the largest class, ahead of the EXL3 MoE GEMM at 29.7 % —
**21.1 % of a C8 step and 17.4 % of a prefill chunk** `[measured-here]`
([10](10-results-and-roofline.md) §5.3).

Four consequences worth holding on to:

- Kernel work inside `cuda-exl3` cannot touch the largest item in the decode profile. Everything in
  that column of the ranked target table comes to about 5 %; this one item was priced at **~+34 %
  single-stream** if the same layers were 4-bit `[estimate]`, and **measured at +24.3 % per stream**
  at TP=2 with a full-scope checkpoint `[measured-here]` — the estimate was an upper bound
  ([13](13-full-scope-checkpoint.md) §4.2).
- Each rank's share of those BF16 tensors is a *third* rather than a half, so they get **less**
  efficient per rank as ranks are added. Part of the single-stream gap against a two-node arrangement
  is exactly this, and it is not a bug.
- **The scope of our checkpoint was not a quality decision, and we were wrong to describe it as
  one** `[retracted]`. Two lines in vLLM's `glm5next` model file pin the whole attention stack to
  BF16 whatever the weights contain — `quant_config=None` for the MLA projections
  (`model.py:331`) and a `quant_config` strip in the KDA constructor (`kda.py:171-174`) — and
  between them they lock **72.8 %** of the dense traffic. Until those are conditional, no checkpoint
  of any scope can put attention on EXL3, so `routed_experts_only` was the only thing that could
  load ([11](11-open-issues.md) §1.9 row 29, [13](13-full-scope-checkpoint.md) §2.2).
- The way out is a differently scoped checkpoint plus a loader patch, and the TP=3 obstacle is
  softer than §3.1 implies: an EXL3 tensor still cannot be *zero-extended*, but it can be loaded
  narrow into a padded parameter with `svh = 0` on the pad, provided the pad occupies whole
  128-column blocks. Our head pad already does (64 → 66 heads = 256 columns); our vocab pad does not
  (192 = 1.5 blocks) and would at `padding_size=384`. The quality gate that decided whether this was
  worth building has now **passed** — [13](13-full-scope-checkpoint.md) is the whole story and
  [11](11-open-issues.md) §2.22 is what remains.

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
which turns §3.1 into a working configuration. For §3.2 — the other checkpoint, why it would not load,
and what it is worth — [13 — The full-scope checkpoint](13-full-scope-checkpoint.md).

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
| Scope | `quantization_config.scope: glm53_routed_experts_only` — the routed experts are EXL3; attention, KDA, the shared expert and `lm_head` stay BF16 |
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
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (routed experts only, `mcg` codebook) | **What we run.** Everything outside the routed experts is BF16, so ordinary tensor parallelism still applies to attention and KDA, and the experts can be distributed whole. TP=3 is possible. |
| A fully quantized 4.05 bpw package (`mul1` codebook, attention and `lm_head` quantized too) | **Rejected for TP=3.** With attention quantized as well, there is no unquantized dimension left to split three ways — see §3. It would leave TP=2 or pipeline parallelism, and a third idle node `[not tested]` at TP=3. |
| Higher-precision EXL3 packages | Not evaluated for this stack `[not tested]`. |

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
are unquantized. In the decode profile those BF16 dense GEMMs are **~37 % of GPU time**, ahead of the
EXL3 MoE GEMM at 29.3 % `[measured-here]`. At prefill they are ~20 %.

Two consequences worth holding on to:

- Kernel work inside `cuda-exl3` cannot touch the largest item in the decode profile. The largest
  remaining structural lever on this stack is a checkpoint that also quantizes attention — which,
  today, is the one that cannot run at TP=3 (§1).
- Each rank's share of those BF16 tensors is a *third* rather than a half, so they get **less**
  efficient per rank as ranks are added. Part of the single-stream gap against a two-node arrangement
  is exactly this, and it is not a bug.

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
which turns §3.1 into a working configuration.

# Credits

Nothing in this recipe was built from nothing. Below is every external component we benefited from,
with the exact revision we ran, what we use it for, and its licence as we confirmed it (URL given).
Licences we could **not** confirm are marked as such, with the page we looked at.

Licences are summarised for readers in [LICENSES.md](LICENSES.md). This repository's own content is
Apache-2.0 ([LICENSE](LICENSE)).

Revision discipline: an entry without a commit, a Hugging Face sha or an image digest is not a
revision. Where an upstream project moved after we pinned it, we say so rather than silently pointing
at `main`.

---

## Model and weights

### `zai-org/GLM-5.3-Flash` — the base model

- **What we use it for:** the model this whole recipe serves. Everything downstream derives from its
  architecture: 45 layers, 288 routed experts, one shared expert, 64 attention / 64 KV / 64 KDA
  heads, `vocab_size` 154,880, `moe_intermediate_size` 2,048, `intermediate_size` 12,288. Those 64
  heads and that vocabulary are why TP=3 needs padding at all
  ([docs/03](docs/03-tp3-padding-and-sidecars.md)).
- **Link:** https://huggingface.co/zai-org/GLM-5.3-Flash
- **Licence:** **MIT** — confirmed on the model card metadata (`license: mit`).

### `brandonmusic/GLM-5.3-Flash-tr3-4bpw` — the checkpoint we actually load

- **What we use it for:** the production checkpoint. 175.6 GB across 120 safetensors shards; EXL3
  trellis at 4 bits per weight with the `mcg` codebook; `quantization_config.scope:
  glm53_routed_experts_only`, so attention, KDA, the shared expert and `lm_head` stay BF16. We do not
  modify it on disk — the shape changes go into a sidecar directory of symlinks plus one rewritten
  `config.json`, so its `SHA256SUMS` stays verifiable.
- **Revision:** Hugging Face commit `b20c49ba9ecafb563099536e307d21c1310e1c49` (short `b20c49ba`,
  30 August 2026). **Not the current `main`** — as of 5 September 2026 `main` is `aba59d21`, four
  days newer, which we have not tested.
- **Link:** https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw
- **Licence:** **ShapleyMCG License 1.0** — the model card carries `license: other` with
  `license_name: shapleymcg-license-1.0`, and the licence text is in the repository's `LICENSE` file.
  Read it: it is short, it is not an OSI-approved open-source licence by its own statement, and it
  excludes one named individual from every right it grants. Detail in [LICENSES.md](LICENSES.md).
- **What its card reports** `[reported]`: base model `zai-org/GLM-5.3-Flash-BF16`; KL divergence
  against BF16 of **0.0246 nats** over 51,175 positions.
- We do not mirror or redistribute these weights.

### `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw — the full-scope checkpoint we measured

- **What we use it for:** the measurement arm that put a number on this stack's largest open item —
  what the unquantized dense stage is actually worth ([docs/13](docs/13-full-scope-checkpoint.md)).
  Loaded at TP=2 on 5 September 2026: +24.3 % per stream, MMLU sample inside the control's error bar.
  **Not in production**, and not yet loadable at TP=3.
- **Revision:** branch `4.05bpw`, commit `2a30229e67012798ba9f0cd832bb78abf4c363d5` (28 August 2026).
  165.2 GB / 153.8 GiB across 19 shards; exl3 v1.4.4, `mul1` codebook, full scope — routed experts at
  4 bits, dense and attention at 5–6, `lm_head` at 6, calibrated on 250 rows × 2,048 columns. Verified
  here with `sha256` 23/23 against the repository's own metadata, independently on two nodes.
- **Link:** https://huggingface.co/turboderp/GLM-5.3-Flash-exl3
- **Licence:** **MIT** — the `LICENSE` file is the MIT text, "Copyright (c) 2026 Z.AI Co., Ltd", and
  the card carries `license: mit`. More permissive than the checkpoint we run in production: no
  attribution condition and no exclusion clause. Read it yourself; different publisher, different
  terms.
- We do not modify it on disk and we do not redistribute it. Everything needed to serve it is a
  runtime patch inside the container ([docs/13](docs/13-full-scope-checkpoint.md) §3), which is also
  why we rejected rewriting its shards: it would have cost the `sha256` match against the publisher.
- **Also in the repository and unused here:** `mtp.safetensors` (3.79 GB), an MTP drafter that is not
  in the safetensors index and that vLLM never reads. Under this checkpoint's licence it would
  transfer to a reader in a way our own draft model does not. Not evaluated `[not tested]`.

### `incoai/GLM-5.3-Flash-DFlash2` — the speculative draft model

- **What we use it for:** speculative decoding at k=7. 2.3 GB, BF16, 5 draft layers over 45 target
  layers. It is worth roughly 2.6× at a single stream on this stack
  ([docs/04](docs/04-dflash2-port.md)). We pad **our local copy** of its config from 32/8 to 36/9
  heads for TP=3, in a sidecar; the downloaded copy is untouched.
- **Link:** https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- **Licence:** **CC BY-NC-ND 4.0** — confirmed on the model card metadata
  (`license: cc-by-nc-nd-4.0`). NonCommercial, NoDerivatives, attribution required.
- **Important, and non-transferable:** we hold a project-specific written permission from the author
  for our own use. It covers us only, it is not sublicensed by this repository, and we do **not**
  redistribute the draft weights. Obtain your own. See [LICENSES.md](LICENSES.md).

---

## Engine and kernels

### `Zeuss5/cuda-exl3` — the EXL3 CUDA kernels and the vLLM plugin

This is the project this recipe is built on, and the one we owe the most to. EXL3 dense and MoE GEMM
kernels for Blackwell plus a fused sparse-MLA attention backend, with a vLLM plugin.

- **What we use it for:** everything EXL3. The routed-expert GEMM, the Hadamard input transforms, the
  combine, the MLA decode backend, the expert-parallel path, and the vLLM integration.
- **Link:** https://github.com/Zeuss5/cuda-exl3
- **Licence:** **MIT** — the repository's `LICENSE` file is the MIT text plus an attribution
  paragraph crediting `turboderp-org/exllamav3` for derived files. Note that GitHub's own licence
  classifier reports `other` / NOASSERTION for the repository, which is what a licence file with an
  added paragraph does to template matching; the file body is MIT terms. Both facts are recorded
  because they are not contradictory and a compliance reviewer will see the second one first.
- **Production commit:** `62f53e676d3e416401c3a0716558e1454affa8ad` — "Bound what is left in
  `exl3_moe_had_in`", carrying `a47da6e` beneath it, since 5 September afternoon (production
  configuration 8). The previous production commits were
  `9bf594cd8b43a2a53db9c7d1d629794aa9365f1a`, "Persist the MLA tuner cache across processes"
  (production 5–7; every table dated between 5 September 06:45 and that afternoon), and before it
  `f4987cf11806c7381c8a59cb388ab5863852679c`, "Do not fetch the MoE padding rows".

Commits this recipe built, measured or depends on:

| Short | Full | Subject | Why it matters here |
|---|---|---|---|
| `37330c9` | `37330c99b20b6199278b838b3d40bda413111a42` | Evict the tuner's cache by reading, not writing | the commit the first working image was built from |
| `e0a3975` | `e0a39752857ff4f4ce80f3811885d24ce4d323e3` | Fix the expert-parallel path | expert-parallel alignment against the global count |
| `77513d2` | `77513d267b78119003e1bd9f4448e3ee71465d76` | Size the MoE block from the global expert count | confirmed correct on GB10; the local count would have cost +32 % at M=2048 ([docs/05](docs/05-expert-parallel-and-cuda-exl3-fixes.md) §3.1) |
| `f906f00` | `f906f002e5e7c3256d236faa56754ae16758c308` | Skip unowned rows in the combine | the design we had arrived at independently, and the better one |
| `60d5349` | `60d5349b42ffad94bb337d7ade5882c42c383b19` | Add `bench_moe_ep.py` | the author's own expert-parallel benchmark, which we ran on both sides of the fix |
| `a95e809` | `a95e8098e65ae1430b6bda79cbbe42b665cc274f` | Pass `n_rows` on the unsplit MoE launch | **the fix for the regression that cost us 45 % of the MoE stage**; the line came from our report and the author's own diagnosis of the mechanism was the correct one |
| `bc0e0f6` | `bc0e0f699a826242a4cf917023d978ecb60ee667` | Bucket topk in the MLA tuner key | the tuner fix; on our model the varying axis turned out to be the batch, not the top-k |
| `61a17bc` | `61a17bcdcd5ec496bd963fd265dda7774225e671` | Do not fuse the input transform | the fusion arm we A/B'd |
| `76598b2` | `76598b22cb105b84c4edd6d164567be7a11e9462` | Drop the fused MoE input transform | fusion removed upstream |
| `f4987cf` | `f4987cf11806c7381c8a59cb388ab5863852679c` | Do not fetch the MoE padding rows | the production commit for configurations 2–4; every table dated before 5 September 06:45 |
| `e24f059` | `e24f059793c01b26418fee054484bcec4316567e` | Always skip the MoE padding rows | the follow-up; equivalent on this hardware, where the gate is always open. Rides along in the `9bf594c` checkout, still unmeasured on its own `[not tested]` |
| `5814c7f` | `5814c7fb09e8d1ffaef19506ef38ff02cd279f18` | Let the down projection finish the MoE | not built here |
| **`9bf594c`** | `9bf594cd8b43a2a53db9c7d1d629794aa9365f1a` | **Persist the MLA tuner cache across processes** | **the production commit.** Written by the author in answer to a measurement of ours; built, wired up and measured here — tune events before serving 18 → 0, and our sweep protocol dropped from five rounds to three ([docs/12](docs/12-tuner-cache.md)) |
| `3cad1d2` | `3cad1d2` | Document the environment variables that exist | not built here; read while preparing the bench below |
| `9b17ea9` | `9b17ea9` | Add the expert-reread bench, and close the duplicate-read question here | **the author's answer to our profile.** We ran his script unmodified on GB10 against the production build; it closed our own open item ([docs/10](docs/10-results-and-roofline.md) §5.4, [docs/11](docs/11-open-issues.md) §2.12) |
| `a47da6e` | `a47da6e` | Remove the 64-bit division in `had_in`, deriving the index from the grid | the follow-up to the same profile: **−10 to −18 %** on `exl3_moe_had_in`, roofline 57 % → 63 % `[reported]`. Worth ~0.2–0.3 % of prefill here. **In production since configuration 8**, and it read exactly as advertised: every serving column inside its own band ([docs/10](docs/10-results-and-roofline.md) §1) |
| **`62f53e6`** | `62f53e676d3e416401c3a0716558e1454affa8ad` | **Bound what is left in `had_in`** | **the production commit** since 5 September afternoon, and the answer to "is there more here": the remaining gap is a **half-ALU** limit — a 128-point Hadamard done with warp shuffles — so it is arithmetic that has to happen rather than traffic that can be removed, worth **≤2 % of prefill** on this stack and unreachable in practice `[reported]`. With it the **`cuda-exl3` MoE stage is closed as an optimisation target here** ([docs/10](docs/10-results-and-roofline.md) §6, [docs/11](docs/11-open-issues.md) §2.19) |
| `5903248` | `5903248` | Let a checkpoint declare its own packed-module fusions | written the same hour we reported that `glm5next` declares no `packed_modules_mapping`, so a fusion peculiar to one model can travel with the weights instead of in a fork of the model file. Checkpoint entries merge **under** the model class's, and a malformed entry is dropped rather than raised. **Not usable for our case as written** — the author said so himself before we could try it (see below) `[not tested]` |
| `fba9f27` | `fba9f27` | The same mapping from `CUDA_EXL3_PACKED_MAPPING` | the follow-up, because a published checkpoint is not ours to edit. Verified here against his `config.py` mounted read-only into a CPU container: `in_proj_qkv` 34/34, `gate_up_proj` 45/45, `fused_qkv_a_proj` 11/11 `[measured-here]`. **Not in any image we have built**, so the arm ran on the vLLM patch instead |
| `d19dee0` | `d19dee0` | Handle a bare `suh` on the v1 loader path, and fix the packed-mapping example | **our `ReplicatedLinear` workaround, done properly and on his side.** He infers the shard index from the shape where ours pins it at 0 — the same thing for a replicated module and not for anything else. Same commit corrects the README example we had measured at 0/34 and adds the rule that explains it: **every module in a packed group must be EXL3 in the checkpoint** |
| `807d798` | `807d798` | Make `CUDA_EXL3_DEBUG_NAMES` print, and report the hits too | the diagnostic he had recommended as our acceptance gate, which printed nothing on our image because it logged at `info`. Now `warning`, and it logs the modules that **resolved** as well as those that stayed BF16, with running tallies — the failure it was meant to catch (half the attention stack silently left in BF16) is now visible as a climbing count |

Four things we reported to that project and their outcome:

- The expert-parallel alignment receiving the local expert count, and remote rows reaching the
  combine uninitialised — fixed upstream.
- `n_rows` not passed on the unsplit MoE launch — fixed upstream in `a95e809`. **Our claim that it
  also cost the non-expert-parallel path was wrong and we withdrew it**
  ([docs/11](docs/11-open-issues.md) §1.1).
- The fused input transform winning at small batch and regressing at large batch on this hardware —
  the fusion was subsequently removed upstream.
- **The MLA decode tuner's cache being process-local**, with the measurement of what that costs on
  GB10: tune events per boot, the batch shapes that mint new keys while serving, and the
  round-1-versus-round-3 spread it produces in a serving benchmark. We asked whether a persisted
  cache would be accepted and offered to write it; **the author wrote it instead**, in `9bf594c`,
  with a design better than our sketch — a device-name and format-tag keyed filename, `O_APPEND`
  sharing across the tensor-parallel ranks with no lock, and a runtime fallback for unseen keys. The
  code is theirs; ours was the evidence and the request. It removed the largest measurement tax on
  this stack ([docs/12](docs/12-tuner-cache.md)).

A fifth exchange, and the most useful one to record because **the author was right and we were
wrong**: we sent a full step-time breakdown of a 3× GB10 prefill and decode step, with the rulers
measured on the device, ranking what a change to `cuda-exl3` could still be worth
([issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5), and the same content is
[docs/10](docs/10-results-and-roofline.md) §5). Two things came back within the hour. The author
pointed out that our largest proposed item — duplicate expert reads in the large-M trellis GEMM,
which we had priced at 14–27 % of the weight traffic — rests on a traffic model that a trace cannot
verify, wrote **`9b17ea9`, a bench that settles it**, and reported 1.16× on his own 188-SM card. We
ran that bench unmodified on GB10 and measured 1.11×: the trellis stays resident, the item is closed,
and our estimate was wrong because the model behind it was wrong. He then took the item that
*was* real — `exl3_moe_had_in` at 37–57 % of the ruler — in **`a47da6e`**.

The kernel-library half of that thread closed the same day, and it closed the whole stage rather than
the one kernel. In **`62f53e6`** the author bounded what remains in `had_in`: a 128-point Hadamard
implemented with warp shuffles is **ALU-bound at about half the unit's rate**, so the rest is work
that has to be done, not traffic that can be removed — **≤2 % of prefill on this stack, and
unreachable** `[reported]`. He also confirmed that `_zero_kv_blocks_kernel` is vLLM's, and that on
this model family the page is shared with Mamba/KDA state so the zeroing cannot be skipped
([docs/11](docs/11-open-issues.md) §2.13). **We have no open *performance* item against `cuda-exl3`**,
which is an unusual place for a dependency to end up: two of our three reports produced upstream
fixes, the third produced a bench that proved us wrong, and the fourth produced a bound that tells us
to stop looking. The one thing still outstanding on that side is not a kernel at all — it is the
padded-load path for a vocab-parallel EXL3 head, agreed and scoped and waiting on nothing but the work
(below, and [docs/11](docs/11-open-issues.md) §2.22).

**Then we sent a real profiler trace, and the thread turned into the most valuable exchange in this
repository.** Four things came back on it, and three of them make the author's side of the ledger
longer rather than ours:

- **MLA prefill, closed by him.** We had carried it for a week as "8.2 % of a chunk, efficiency not
  measured, no denominator, needs its own bench". He measured it at our shapes — top-k 2,176,
  head_dim 512, fp8 cache, and a **2 GB pool chosen specifically so it could not be L2-resident** —
  and got **86–89 % of achievable**, 1,299–1,345 GB/s gathered against a 1,518 GB/s ruler in the same
  run `[reported]`. Item closed with no work on our side. He also published that his *first* cut used
  a 200k-row pool that fitted the card's L2 and reported a bandwidth above the DRAM ruler, caught only
  because the number was impossible — the same failure our own model-free MoE bench had just
  committed at small M, on the same day, from the other direction. **Synthetic benches flatter small
  inputs**; that lesson is worth more than either measurement.
- **A conclusion of his own, withdrawn against his own advice.** He had told us a cooperative
  (`grid.sync`) MoE stage was never worth paying for, because "inside a CUDA graph a kernel boundary
  is cheap enough". Our trace says production runs with **graphs off** — spec-decode plus FlashInfer
  forces `cudagraph_mode=NONE` — so the premise did not hold on this stack, and our own uncaptured
  measurement had the cooperative arm winning by up to 1.4 µs per boundary. He said so unprompted, and
  then priced it honestly at ~0.2 ms of a 94.65 ms step and told us not to rush ([docs/11](docs/11-open-issues.md) §2.14).
- **The largest number anyone has put on this stack, and it is not his code.** Reading our C1 split he
  pointed at the row we had been treating as background: dense BF16 GEMM at **45 % of a decode step**,
  because our checkpoint is `scope: glm53_routed_experts_only` and that stage streams 16-bit weights
  at M=8. `Exl3LinearMethod` binds to any dense linear with EXL3 tensors present, so a broader-scope
  checkpoint would put those layers on the same kernel **with no code change on either side** —
  42.9 ms → ~11 ms, **~+34 % single-stream** `[estimate]`, against ~5 % for his whole column of our
  target table. He framed it as a quality question rather than a speed one, named the head at vocab
  154,880 as the place to look for damage first, and said he would want the numbers either way.
- **And a mechanism for the TP=3 half of it that we did not have.** Our objection was that 64 heads
  and a 154,880 vocab do not split three ways, and that `Exl3LinearMethod.create_weights` correctly
  refuses to zero-extend an EXL3 tensor. His answer is the 2,304-sidecar property one level up: in
  `had128_warp_out` the output Hadamard runs **first** and `svh` scales elementwise **afterwards**, so
  zeroing `svh` on a padded output column makes that column exactly zero whatever the trellis holds —
  the invariant already guarded by `test_exl3_moe_pad.py::test_padded_columns_are_exactly_zero`, and
  mirrored for padded *input* dims by `test_output_ignores_w2_padded_row_codes`. A full-scope
  checkpoint quantized **unpadded** can therefore be loaded into a padded parameter, bit-exact, with
  no re-quantization — **provided the pad occupies whole 128-column blocks**. He then checked our two
  tensors against that condition rather than asserting it: heads 64 → 66 = 256 columns = 2 whole
  blocks, **works**; vocab `padding_size=192` = 1.5 blocks, **corrupts the real rows sharing the
  block**. The fix is `padding_size=384`, one constant in our launcher. He has offered to add the
  padded-load path behind a flag once our quality gate says whether it is worth building
  ([docs/11](docs/11-open-issues.md) §2.22).

**Then the full-scope checkpoint went in, and the thread produced four commits in one afternoon.** The
work split cleanly and he drew the line himself: the packed-module mapping belongs in the plugin
because it is checkpoint layout, while the BF16 pinning and the KDA split are model structure and stay
in the model file — "that is the part you did not want to carry as a fork patch". Two of the four are
straight fixes to our reports, and one is a fix to something he had written:

- **`5903248` then `fba9f27`.** He put the packed mapping on his side within the hour, reading it from
  the checkpoint's own quantization config — and then, **before we could wire it up**, told us it was
  unusable for our case as written, because a published checkpoint is not ours to edit. The env-var
  route in `fba9f27` is the fix. Catching your own new feature's blind spot in the same thread, ahead
  of the person who would have hit it at boot, is worth recording.
- **`d19dee0`**, which is our `ReplicatedLinear` workaround done better: he infers the shard index
  from the shape where we pin it at 0 — identical for a replicated module, not identical in general.
  The same commit corrects a README example he had written from our prose without running it, and
  which we then measured at 0/34; he said so plainly and added the rule that explains it, which is the
  part that saves the next reader an afternoon.
- **`807d798`**, after we reported that the diagnostic he had recommended as our acceptance gate
  printed nothing on our image.

**And the one answer that unblocks TP=3 in principle.** We asked for a one-line confirmation that a
zero pad trellis multiplied by `svh = 0` cannot produce NaN. What came back was **the full domain
enumerated**: all 65,536 possible 16-bit trellis codes swept through the device decoder for all three
codebooks, zero non-finite values in every case, with the ranges reported (`3inst`
[−3.9570, +3.9727], `mcg` [−3.9492, +3.9492], `mul1` [−3.4531, +3.3477]) `[reported]`. He also
published that his first attempt reported an implausibly asymmetric range because it ordered floats by
their integer bits, and re-ran it rather than quoting a number he did not trust. **That is the right
standard underneath a padded-load path**, and it is a stronger answer than the question asked for.

The **commitment** that follows from it: once our quality gate decided the lever was worth building,
he would add the padded-load path behind a flag — accept an EXL3 tensor narrower than the parameter,
place it, zero `svh` on the remainder — covering the three cases we checked our TP=3 geometry against:
the vocab-parallel `lm_head` (hard-refused today, and this checkpoint has no BF16 head to fall back
on), a "refuse unless the pad is 128-aligned" gate, and the input-dim case where
`Exl3SuhParameter.load_row_parallel_weight` narrows on its own and would overrun the last rank's
`o_proj`. He asked for the drafter's acceptance rate ahead of MMLU, on the grounds that a quantized
target losing acceptance would decide TP=3 before quality did — a better ordering than ours. **The
gate passed on 5 September** ([docs/13](docs/13-full-scope-checkpoint.md)), acceptance is flat, and
the word has been sent.

Three notes we owe that thread rather than the other way round: a 128-token prefill chunk costs 403 ms
because 128 tokens at top-8 already touch every expert, which is a batched-token-budget fact worth
knowing on any Spark; a uniform-random top-8 microbenchmark overstates small-M MoE cost by
**1.5–1.7×** against real clustered routing, which he accepted applies to his own `bench_moe_ep.py`
and padding-skip benches; and two corrections against our own published numbers —
`_zero_kv_blocks_kernel` from 14.7 ms to 0.86 ms, and the drafter from 19.5 % to 11.4 % of a C1 step —
which he pointed out is the harder direction to publish.

Our one surviving kernel change, `patches/kernel/0003-combine-smem-staging-on-a95e809.patch`, is
offered upstream with its description in `patches/kernel/0003-PR-DESCRIPTION.md`.

### `turboderp-org/exllamav3` — the EXL3 format

- **What we use it for:** the quantization format itself. The trellis layout, the codebooks and the
  128-element Hadamard block whose alignment rule decides that an EXL3 tensor cannot be split three
  ways ([docs/03](docs/03-tp3-padding-and-sidecars.md)).
- **Link:** https://github.com/turboderp-org/exllamav3 (note the organisation account; the
  personal-account path does not exist)
- **Licence:** **MIT** — confirmed on the repository.

### `vllm/vllm-openai` — base container image

- **What we use it for:** the bottom of our two-layer image: aarch64 + CUDA 13.0, the GLM-5.3 model
  implementation, and the vLLM the whole stack runs inside.
- **Revision:** tag `glm53-flash-arm64-cu130`, digest
  `sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`, linux/arm64, pushed
  26 August 2026. The vLLM inside reports `0.1.dev20051+g487ecf187`.
- **Links:** https://hub.docker.com/r/vllm/vllm-openai/tags · https://github.com/vllm-project/vllm
- **Licence:** **Apache-2.0** for vLLM — confirmed at
  https://github.com/vllm-project/vllm/blob/main/LICENSE . The NVIDIA CUDA, cuDNN and NCCL libraries
  inside the image carry NVIDIA's own redistribution terms, which is why we do not redistribute any
  image.

### vLLM upstream — the DFlash2 speculative-decoding support

- **What we use it for:** the DFlash2 delta that the base image predates, taken as a genuine
  three-way merge onto the image's own vLLM tree rather than hand-copied
  ([docs/04](docs/04-dflash2-port.md)).
- **Licence:** **Apache-2.0**, as above.

### `autoscriptlabs/nccl-mesh-plugin` — the fabric transport

- **What we use it for:** NCCL over three direct ConnectX-7 links with no switch. Without it there is
  no three-node collective on this topology.
- **Revision:** commit `19924dcc7c571d6e260953724d394ae50bad82cf`.
- **Link:** https://github.com/autoscriptlabs/nccl-mesh-plugin
- **Licence:** **MIT** — confirmed on the repository.
- **Three findings reported upstream, all with patches, none of them the plugin author's fault to
  have missed** — two of the three only show up on a fabric with more than one cable per pair:
  1. A one-line flow-control setting (`min_rnr_timer = 12`, 0.64 ms, where the comment intends code 1
     at 0.01 ms) that costs about 10× on mid-size collectives.
     `patches/kernel/0004-min-rnr-timer.patch`. The plugin's own published benchmarks show the same
     curve on the author's hardware, so it is a property of the design, not of anyone's cabling.
  2. **`mesh_connect()` ignoring NCCL's `dev` index** (`(void)dev;`) and stopping at the first
     subnet match, so every channel to a peer rides one cable and the second cable of each pair never
     carries a byte — confirmed in hardware with `port_xmit_data == 0` on those ports after weeks of
     use. `patches/kernel/0005-device-aware-link-selection.patch`, ~30 lines, no wire-format change.
  3. **`ptrSupport` advertising `NCCL_PTR_HOST` only**, although the plugin's own `regMr` already
     registers CUDA pointers and plain `ibv_reg_mr` works for device pointers on this unified-memory
     part — so NCCL was staging every transfer through a host bounce buffer for nothing.
     `patches/kernel/0006-ptr-cuda-dmabuf-and-flush.patch`, which also makes `mesh_iflush()` a real
     RDMA_READ and `regMrDmaBuf` a real DMA-BUF registration.

  Findings 2 and 3 went upstream together as a follow-up on the plugin's issue thread (issue #58),
  with the sweep numbers and the port-counter evidence. All three patches now live on a public fork,
  [`NNNtrance/nccl-mesh-plugin`](https://github.com/NNNtrance/nccl-mesh-plugin), branch
  `gb10-dual-link-ptrcuda` on top of `19924dcc`, and are offered as
  [**pull request #59**](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59). All three are
  env-gated (`NCCL_MESH_LINKS_PER_PEER`, `NCCL_MESH_PTR_CUDA`, `NCCL_MESH_FLUSH`,
  `NCCL_MESH_MIN_RNR_TIMER`) and default to today's behaviour when a peer has one cable. Details and
  numbers in [docs/06](docs/06-nccl-mesh.md) §6–§8. We do not ship the built `.so`.

- **A fourth patch, deliberately not offered** — `patches/kernel/0007-one-sided-fifo-rdma-write.patch`
  replaces the two-sided SEND/RECV data path with the receiver-advertised FIFO plus
  `RDMA_WRITE_WITH_IMM` that NCCL's own IB transport uses, which is the design the plugin's own
  roadmap names. It works: RNR retries and out-of-buffer events go to **exactly zero**. It is also
  worth **nothing** on this hardware, because the ceiling is the cards' PCIe Gen5 x4 slots and not the
  flow control. Sending a 977-line transport rewrite upstream on the strength of a mechanism that
  changes no number would cost the maintainer time we have no evidence is worth spending; it is kept
  in this repository as an option and described honestly in
  [docs/06](docs/06-nccl-mesh.md) §10. Credit where it belongs: the design is NCCL's, and the
  plugin's README had already named it as future work.

### NCCL 2.30.7 (inside the base image)

- **Licence: not confirmed for the binary we run.** The upstream project states most of it is
  Apache-2.0, parts keep an original BSD licence, and borrowed files carry their own text; we did not
  read the licence files inside the image. Treat it as part of the NVIDIA container stack under
  NVIDIA's terms: use it, do not redistribute the image.

### CUDA 13.0, cuDNN, NVIDIA driver 580.173.02, DGX OS 7.5.0

- **Licence: NVIDIA licence terms.** Not open source. Install them from NVIDIA on your own hardware.
  Do not redistribute them, and do not redistribute images containing them.

---

## Prior art we learned from

We vendor no files from either of these. What we took is practice and arithmetic, and we say which.

### `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`

- **Licence:** **MIT** — confirmed. 342 stars.
- **What we took:** the honest image path — building on the official vLLM GLM-5.3 base tag rather
  than on a mystery image, pinning the quantization library to a commit, and gating the build on the
  presence of the symbols it needs. Also the observation that the sparse-MLA backend's requirements,
  not the quantization format, are what force `fp8` KV on this hardware.
- **Where it does not apply:** its overlay shards routed experts unconditionally, which cannot work
  at TP=3.

### `FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks`

- **Licence:** **MIT** — confirmed. 1 star.
- **What we took:** the three-node arithmetic. The 64 → 66 head padding with 22 heads per rank; the
  vocabulary `padding_size` of `lcm(64, tp)`; padding the BF16 shared expert to 2,112 and slicing it
  rather than replicating it; the drafter's 32/8 → 36/9 pad; and the conditional-shard idea that lets
  a loader leave an already-correct tensor alone. Also, and importantly, its written-up finding that
  replicating the shared expert under expert parallelism produces a silently tripled contribution —
  a model that stays fluent and gets the answers wrong.
- We re-derived each of those from this checkpoint's own shapes and check them at load time rather
  than trusting them ([docs/03](docs/03-tp3-padding-and-sidecars.md) §5).

### `NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark` — our own sibling recipe

- **Licence:** **Apache-2.0**.
- The cluster setup, the fabric wiring and preflight, the "reboot all three" rule, the memory rules,
  the measurement scripts (`bench-sweep.py`, `correctness-probe.py`, `code-exam.py`,
  `category-speed.py`, `cold-warm-c1.py`, the `hizset-v2` prompt set) and the roofline hardware
  measurements are shared with it. Those scripts are reproduced here unchanged apart from the default
  port.

---

## Measurement tools

- **`bench/` in this repository** — the model-free harness (expert-parallel geometry, per-stage MoE
  timing, expert-map placement, mesh all-reduce with hardware counters, top-k comparison, fresh
  prefill, profiler driver and analyser). Written by us; Apache-2.0. The step-breakdown work added
  seven more, also ours: `bw.py` and `gemmpeak.py` (the two rulers — run them in the same process as
  whatever you are measuring), `live-step.py` and `live-decode.py` (prefill ladder, chunk-boundary
  probe and exact ms-per-engine-step against a running server, via its own metrics), `mhc_bench.py`
  (the hyper-connection kernels, three routes, graph on and off), `zerokv_bench.py` (vLLM's KV-zeroing
  kernel at the live engine's grid geometry) and `prof-analyze3.py` (the kernel taxonomy and pass
  segmentation the breakdown in [docs/10](docs/10-results-and-roofline.md) §5 is built from; the older
  `prof-analyze.py` is kept).
- **`bench_moe_expert_reread.py`** — **not ours**. Written by the `cuda-exl3` author in `9b17ea9` to
  settle a question we had raised the wrong way round; we ran it unmodified and it closed the item
  ([docs/11](docs/11-open-issues.md) §2.12). Under that project's MIT licence.
- **PyTorch profiler**, via vLLM's `--profiler-config`. Note that on this vLLM the environment
  variable form does nothing and the endpoint returns 404 — the flag is the only way in
  ([docs/05](docs/05-expert-parallel-and-cuda-exl3-fixes.md)). Set it on the launcher **before** you
  need it: an engine already serving production cannot be profiled and should not be restarted to
  try ([docs/10](docs/10-results-and-roofline.md) §5).

---

## Our own patches

Everything under `patches/` was written by us for this recipe unless the file header says otherwise.
Use them freely under Apache-2.0; a credit is appreciated and not required. Where a patch implements
someone else's idea, the header says whose.

Two of them are not ours to keep:

- `patches/kernel/0002-harem-on-77513d2.patch` is **retired** — three of its four changes are
  upstream and the fourth was measurably the wrong choice. `patches/kernel/0002-RETIRED.md` records
  that in writing rather than deleting it quietly.
- `patches/kernel/0004-min-rnr-timer.patch`, `0005-device-aware-link-selection.patch` and
  `0006-ptr-cuda-dmabuf-and-flush.patch` are patches against someone else's project
  (`autoscriptlabs/nccl-mesh-plugin`), offered upstream as
  [PR #59](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59) from the fork
  [`NNNtrance/nccl-mesh-plugin`](https://github.com/NNNtrance/nccl-mesh-plugin), and carried here so
  the findings are reproducible. If they land upstream we will retire our copies in writing, the way
  `0002-RETIRED.md` retires ours.
- `patches/kernel/0007-one-sided-fifo-rdma-write.patch` is against the same project and is **not**
  offered upstream and **not** in that PR branch, because our own measurement says it changes nothing
  on this hardware. It is kept, with the measurement that rejected it, rather than deleted
  ([docs/06](docs/06-nccl-mesh.md) §10).

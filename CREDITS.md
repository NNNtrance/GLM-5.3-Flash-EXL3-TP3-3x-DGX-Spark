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
- **Production commit:** `f4987cf11806c7381c8a59cb388ab5863852679c` — "Do not fetch the MoE padding
  rows".

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
| **`f4987cf`** | `f4987cf11806c7381c8a59cb388ab5863852679c` | **Do not fetch the MoE padding rows** | **the production commit** |
| `e24f059` | `e24f059793c01b26418fee054484bcec4316567e` | Always skip the MoE padding rows | the follow-up; equivalent on this hardware, where the gate is always open `[not tested]` |
| `5814c7f` | `5814c7fb09e8d1ffaef19506ef38ff02cd279f18` | Let the down projection finish the MoE | not built here |
| `9bf594c` | `9bf594cd8b43a2a53db9c7d1d629794aa9365f1a` | Persist the MLA tuner cache across processes | the answer to an open item of ours; not built or measured `[not tested]` |

Three defects we reported to that project and their outcome:

- The expert-parallel alignment receiving the local expert count, and remote rows reaching the
  combine uninitialised — fixed upstream.
- `n_rows` not passed on the unsplit MoE launch — fixed upstream in `a95e809`. **Our claim that it
  also cost the non-expert-parallel path was wrong and we withdrew it**
  ([docs/11](docs/11-open-issues.md) §1.1).
- The fused input transform winning at small batch and regressing at large batch on this hardware —
  the fusion was subsequently removed upstream.

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
- We found and reported a one-line flow-control setting in it that costs about 10× on mid-size
  collectives, with a patch ([docs/06](docs/06-nccl-mesh.md)). The plugin's own published benchmarks
  show the same curve on the author's hardware, so the finding is a property of the design and not of
  anyone's cabling. We do not ship the built `.so`.

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
  prefill, profiler driver and analyser). Written by us; Apache-2.0.
- **PyTorch profiler**, via vLLM's `--profiler-config`. Note that on this vLLM the environment
  variable form does nothing and the endpoint returns 404 — the flag is the only way in
  ([docs/05](docs/05-expert-parallel-and-cuda-exl3-fixes.md)).

---

## Our own patches

Everything under `patches/` was written by us for this recipe unless the file header says otherwise.
Use them freely under Apache-2.0; a credit is appreciated and not required. Where a patch implements
someone else's idea, the header says whose.

Two of them are not ours to keep:

- `patches/kernel/0002-harem-on-77513d2.patch` is **retired** — three of its four changes are
  upstream and the fourth was measurably the wrong choice. `patches/kernel/0002-RETIRED.md` records
  that in writing rather than deleting it quietly.
- `patches/kernel/0004-min-rnr-timer.patch` is a patch against someone else's project, offered
  upstream, and carried here only so the finding is reproducible.

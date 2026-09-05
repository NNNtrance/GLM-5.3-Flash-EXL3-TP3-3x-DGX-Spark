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
- **Production commit:** `9bf594cd8b43a2a53db9c7d1d629794aa9365f1a` — "Persist the MLA tuner cache
  across processes". The previous production commit was
  `f4987cf11806c7381c8a59cb388ab5863852679c`, "Do not fetch the MoE padding rows"; every table dated
  before 5 September 06:45 was measured on it.

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
| `a47da6e` | `a47da6e` | Remove the 64-bit division in `had_in`, deriving the index from the grid | the follow-up to the same profile: **−10 to −18 %** on `exl3_moe_had_in`, roofline 57 % → 63 % `[reported]`. Worth ~0.2–0.3 % of prefill here — real, and not worth an image rebuild alone. **Not in the production image**; queued for the next build ([docs/11](docs/11-open-issues.md) §2.19) |

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

Two notes we owe that thread rather than the other way round: `_zero_kv_blocks_kernel` costs 14.7 ms
per prefill chunk and belongs to vLLM, not to the kernel library; and a 128-token prefill chunk costs
403 ms because 128 tokens at top-8 already touch every expert, which is a batched-token-budget fact
worth knowing on any Spark.

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

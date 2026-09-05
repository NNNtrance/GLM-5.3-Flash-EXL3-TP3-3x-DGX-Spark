# Licences — what you are allowed to do

This page is a summary for readers, not legal advice. The full per-component detail, with the exact
revision we ran and the URL where we confirmed each licence, is in [CREDITS.md](CREDITS.md).

**Short version.** This repository is Apache-2.0. Most of what it stands on is MIT or Apache-2.0 and
you can use it commercially. Three things are not ordinary: the **fallback checkpoint** is under a
bespoke licence that is permissive but not open source; the **DFlash2 draft model** is non-commercial
and no-derivatives; and the **NVIDIA CUDA / cuDNN / NCCL binaries** inside the container image are
under NVIDIA's own restricted terms. Those are why we publish instructions and patches rather than a
prebuilt image or mirrored weights.

**Which checkpoint you land on decides which of those applies to you.** Since production
configuration 9 the weights we serve are `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw, which is **MIT**
— ordinary, permissive, nothing unusual. The bespoke licence belongs to
`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, which is now the **fallback** for images that predate the
padded-load path. Both rows are in the table; read the one you are actually going to download.

## The table

| Component | Revision we ran | Licence (confirmed at) | What that means for you |
|---|---|---|---|
| This recipe: `docs/`, `patches/`, `scripts/`, `bench/`, `envs/`, `systemd/`, `results/` | this repository | **Apache-2.0** ([LICENSE](LICENSE)) | Use, modify, redistribute, commercially. Keep the notice. A credit or link back is appreciated, not required. |
| `zai-org/GLM-5.3-Flash` (base model) | — | **MIT** ([model card](https://huggingface.co/zai-org/GLM-5.3-Flash)) | Free to use, including commercially. Keep the copyright notice. |
| **`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw** (**the weights we load in production**) | branch `4.05bpw`, HF `2a30229e` | **MIT** ([model card](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)) | Free to use, including commercially. Keep the copyright notice. Nothing unusual here — this is the row that applies to production configuration 9. |
| **`brandonmusic/GLM-5.3-Flash-tr3-4bpw`** (the **fallback** weights, and what configurations 1–8 loaded) | HF `b20c49ba` | **ShapleyMCG License 1.0** — `license: other`, `license_name: shapleymcg-license-1.0` on the [model card](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw); text in the repository's `LICENSE` file | **Read it. It is short and it is unusual.** See the section below. You only need this if your image predates the padded-load path (`cuda-exl3` `f3e3090` + `754421f`). |
| `incoai/GLM-5.3-Flash-DFlash2` (speculative draft) | — | **CC BY-NC-ND 4.0** ([model card](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)) | **Non-commercial, no derivatives, attribution required.** See the section below. |
| `Zeuss5/cuda-exl3` (the EXL3 kernels and vLLM plugin) | `754421f` in production (carrying `f3e3090`); `f4987cf` and `62f53e6` in earlier configurations — see [CREDITS.md](CREDITS.md) | **MIT** — the repository's `LICENSE` is the MIT text plus an attribution paragraph for derived files ([repository](https://github.com/Zeuss5/cuda-exl3)) | Free to use, including commercially. Note that GitHub's licence classifier reports `other`/NOASSERTION for the repository because of the added paragraph; the file body is MIT terms. |
| `turboderp-org/exllamav3` (the EXL3 format) | — | **MIT** ([repository](https://github.com/turboderp-org/exllamav3)) | Free to use. |
| `vllm/vllm-openai` base image | tag `glm53-flash-arm64-cu130`, digest `sha256:905c0293…` | **Apache-2.0** for vLLM ([LICENSE](https://github.com/vllm-project/vllm/blob/main/LICENSE)); NVIDIA terms for the CUDA stack inside | Pull it and build on it. Do **not** redistribute the image or anything derived from it: the NVIDIA CUDA, cuDNN and NCCL libraries inside carry NVIDIA's restricted redistribution terms. Everyone builds their own. |
| vLLM (the DFlash2 delta we merged) | upstream commits, see [CREDITS.md](CREDITS.md) | **Apache-2.0** | Use, modify, redistribute. Our patches against it are Apache-2.0 too, so the combination stays consistent. |
| `autoscriptlabs/nccl-mesh-plugin` | commit `19924dcc` | **MIT** ([repository](https://github.com/autoscriptlabs/nccl-mesh-plugin)) | Free to use, including commercially. Build it yourself; we do not ship the `.so`. Our patch to it (`patches/kernel/0004-min-rnr-timer.patch`) is offered upstream. |
| NCCL 2.30.7 (inside the base image) | 2.30.7+cuda13.3 | **Licence not confirmed for the binary we run** (looked at https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt) | That file states most of the project is Apache-2.0, parts keep an original BSD licence, and borrowed files carry their own text; we did not read the licence files inside the image. Treat the binary as part of the NVIDIA container stack under NVIDIA's terms: use it, do not redistribute the image. |
| CUDA 13.0, cuDNN, NVIDIA driver 580.173.02, DGX OS 7.5.0 | as in [docs/00](docs/00-hardware-and-os.md) | **NVIDIA licence terms** ([DGX Spark docs](https://docs.nvidia.com/dgx/dgx-spark/)) | Not open source. Install from NVIDIA on your own hardware. Do not redistribute them, and do not redistribute images containing them. |
| `FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks` (prior art) | see [CREDITS.md](CREDITS.md) | **MIT**, confirmed | We vendor **no files** from it. What we took is practice and arithmetic, and we say which. If you copy its files rather than its ideas, read its licence first. |
| `NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark` (our sibling recipe) | — | **Apache-2.0** | The measurement scripts here come from it, unchanged apart from the default port. |

---

## The fallback checkpoint: ShapleyMCG License 1.0

**This section is about `brandonmusic/GLM-5.3-Flash-tr3-4bpw` only.** The production checkpoint
(`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw) is plain MIT and none of what follows applies to it. If
you are following the current recipe you can skip this section; if your image predates the
padded-load path, you cannot.

The fallback weights are not under MIT, Apache-2.0, or any licence you have met before.
There is a real `LICENSE` file in the checkpoint's repository and **you should read it yourself**.

What it says, as we read it `[reported]`:

- **Commercial use is permitted**, for everyone, without a separate agreement.
- **Modification and redistribution are permitted.**
- **Attribution is mandatory**, written as a condition of the grant rather than as a courtesy.
- **One named individual is excluded from every right the licence grants.** The licence identifies
  that person by name and by account, extends the exclusion to their agents and heirs, adds an
  anti-circumvention clause, and terminates automatically on breach. The stated reason is a dispute
  over unattributed reuse.
- The licence text says of itself that it does not meet the Open Source Definition, and it names a
  US state jurisdiction.

**What that means in practice.** For almost every reader this is a permissive licence with an
attribution requirement — more permissive than the draft model you will pair it with. But:

1. It is **not an OSI-approved open-source licence**, so do not let it pass a compliance review as
   "MIT-equivalent". If your organisation has an allow-list of licences, this is not on it.
2. **Read the exclusion clause and confirm it does not name you**, your employer, or anyone acting on
   your behalf. That is a two-minute check and there is no substitute for it.
3. The attribution requirement is a licence condition. Credit the checkpoint's author wherever you
   would credit a model.

We do not mirror or redistribute these weights. Download them from the author's repository, at the
pinned revision, under terms that apply to you.

---

## The draft model: CC BY-NC-ND 4.0

`incoai/GLM-5.3-Flash-DFlash2` is the single most restrictively licensed component in this stack, and
it is worth roughly 2.6× at a single stream, so it is also the one you will most want to use.

**CC BY-NC-ND 4.0 means:**

- **NonCommercial** — you may not use it for commercial purposes without separate permission from the
  author. Serving a commercial product from this stack with the draft enabled is a commercial use.
- **NoDerivatives** — you may not share a modified or adapted version. Note that the TP=3 recipe
  changes the drafter's config (32/8 → 36/9 heads). We do that in a **sidecar directory beside your
  own copy**, which is a change to your own copy rather than a distributed derivative; do not publish
  the modified drafter.
- **Attribution** — required for any permitted use.

**Our permission does not transfer to you.** We hold a project-specific written permission from the
author for our own use of this draft. It covers us only. It is not sublicensed by this repository, it
is not implied by anything here, and we do not redistribute the weights. If your use goes beyond what
CC BY-NC-ND 4.0 allows you — commercial serving, or publishing a modified draft — **obtain your own
permission from the author before you run it.**

**You can run this whole recipe without the draft.** Set `SPEC_METHOD=` empty in your environment
file. Single-stream decode falls from roughly 54 tok/s to roughly 20, and everything else works.

---

## What we do not redistribute, and why

- **The container image.** It contains NVIDIA CUDA, cuDNN and NCCL under restricted redistribution
  terms. You build it from the published base tag with the recipe in
  [docs/02](docs/02-image-build.md).
- **The model weights.** Both licences would permit it; they are 165 GB (production) and 175.6 GB
  (fallback) and the upstream copies are authoritative. Download them at the pinned revisions.
- **The DFlash2 draft weights.** Non-commercial, no-derivatives, and our permission is
  project-specific and non-transferable.
- **The NCCL mesh plugin binary.** MIT, so redistribution would be permitted; building it takes a
  minute and a binary we built is not a binary you can audit.
- **Any sidecar.** The model sidecar, the drafter sidecar and the fast-load sidecar are all derived
  from weights we do not redistribute.

---

## Disclaimer

This recipe is provided **as is, without warranty of any kind**, express or implied, including
merchantability, fitness for a particular purpose and non-infringement. We measured what we measured
on our own three nodes, with the versions listed in [CREDITS.md](CREDITS.md), and we labelled the
evidence for every claim. Your hardware, firmware, driver and upstream revisions will differ, and
results may differ with them.

**You are responsible for complying with the licence of every upstream component you download, build
or run** — model weights, draft model, container images, kernel libraries, transport plugins and
NVIDIA software. Nothing here grants you any right in anyone else's work, and our project-specific
permission for the DFlash2 draft does not extend to you.

This repository's own content — documentation, patches, scripts and measurement data — is licensed
under the **Apache License, Version 2.0**. See [LICENSE](LICENSE).

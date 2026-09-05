# 02 — Building the engine image

This page builds the container that serves the model: **two layers on one pinned base image**,
built independently on all three nodes ([00](00-hardware-and-os.md)) from one source tarball.
Nothing here starts an engine; serving flags are in [03](03-tp3-padding-and-sidecars.md) and
[05](05-expert-parallel-and-cuda-exl3-fixes.md). We do not redistribute the image — the base
carries NVIDIA CUDA, cuDNN and NCCL under NVIDIA's own terms, so you build your own from the same
pinned inputs. Read section 5 first: it holds the most useful negative result here — **binary
hashes cannot certify this build** — and retracts one of our own earlier rules.

## 1. The image chain

```
vllm/vllm-openai:glm53-flash-arm64-cu130          base, 30.7 GB
  └─ exl3-zeus:serve-f4987cf    Zeuss5/cuda-exl3 compiled for sm_121      30.8 GB
      └─ exl3-zeus:f4987cf      the DFlash2 port + its build-time gate    31 GB
```

The base ships vLLM and its CUDA stack. The **serve layer** compiles the EXL3 kernels
(`cuda_exl3`) against that vLLM for `sm_121`. The **port layer** replaces twelve vLLM Python files
with the DFlash2 speculative-decoding port and runs a gate that refuses to produce an image if any
ported symbol fails to resolve. Only the middle layer compiles; the top one copies text files.

| Component | Source | Exact revision | License |
|---|---|---|---|
| Base image | Docker Hub `vllm/vllm-openai:glm53-flash-arm64-cu130` | `sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`, linux/arm64; the vLLM inside reports `0.1.dev20051+g487ecf187` | Apache-2.0 (vLLM); bundled NVIDIA libraries under NVIDIA terms — **do not redistribute the image** |
| EXL3 kernels | <https://github.com/Zeuss5/cuda-exl3> | `f4987cf11806c7381c8a59cb388ab5863852679c` — see below | see [../CREDITS.md](../CREDITS.md) |
| DFlash2 port | this repository, `patches/dflash2-port/` | upstream vLLM `b389ac294`, 3-way merged onto the image's tree, plus our additions — [04](04-dflash2-port.md) | Apache-2.0 for our part |
| Model checkpoint | not in the image | [01](01-model-and-license.md) | ShapleyMCG License 1.0 |

Pull the base **by digest**, not by tag. We do not know who publishes that tag or how long it will
exist, and we have no documented fallback for the day it disappears.

```
docker pull vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
```

```
docker tag vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce vllm/vllm-openai:glm53-flash-arm64-cu130
```

That same digest also based an earlier EXL3 attempt of ours, built on the community `exllamav3`
overlay instead of these kernels. It pinned `exllamav3` at `c5d9c65` (v0.0.43, June 2026, ~400
commits behind its HEAD) and **held the pin**: on master the `perform_cpu_reduce*` signature
changed so the aarch64 AVX stubs no longer link, and the newer `sm_121` work sits in `exllamav3`'s
own model stack, which vLLM never calls. Recorded here only because the base image is shared.
`[measured-here]`

### Which commit is production, and what the others were

**Production is `f4987cf`** — full sha `f4987cf11806c7381c8a59cb388ab5863852679c`, *"Do not fetch
the MoE padding rows"*. A routed MoE block is `block_m` rows wide, but decode routes about one row
per expert, so most of a block is padding carrying zeros; `cp.async` with a zero source size
zero-fills, which is exactly what a padding row must hold, so the gemm skips the fetch by widening
a predicate it already had. Every commit we built or measured, in order:

| commit | what it contained | how we used it |
|---|---|---|
| `37330c9` | where this stack started: the `cuda_exl3` Python in our first serve image was byte-identical to the repo at this revision | baseline; our first expert-parallel patch was written against it |
| `e0a3975` | upstream expert-parallel alignment, and unowned tiles zeroed | tag `exl3-zeus:e0a3975` |
| `77513d2` | `block_m` taken from the global expert count. Python only, so the compiled extension is byte-identical to the layer below | tag `exl3-zeus:77513d2` |
| `f906f00` | upstream's own expert-map-aware `exl3_moe_combine` (optional `expert_ids`/`block_m`, retired rows skipped), and an unsplit path that zeroes the output tile instead of returning | reached us inside `60d5349`; never a tag of its own |
| `60d5349` | `f906f00` plus upstream's new MoE benchmark | tag `exl3-zeus:u60d5349`, for the comparison in [05](05-expert-parallel-and-cuda-exl3-fixes.md) |
| `a95e809` | upstream adopted both of our kernel changes — `n_rows` on the unsplit MoE launch, and the `exl3_moe_had_in` reorder — and gave the correct mechanism for the surplus tail | measured per kernel; our patch `0002` retired here |
| `bc0e0f6` | buckets `topk` in the MLA tuner key; corrects the `persistent_topk` note | tags `exl3-zeus:bc0e0f6` and `bc0e0f6s` (the latter with our `0003` combine patch) — production before `f4987cf` |
| `61a17bc` | *"Do not fuse the input transform where split-k would have been chosen"* — narrows the fusion gate to `n_blocks >= 2 * SMs` rather than pinning split-k off. Python only | tag `exl3-zeus:61a17bc` |
| `76598b2` | *"Drop the fused MoE input transform; keep the fused combine"* — removes the feature instead of gating it; 232 kernel lines deleted against 60 added | tag `exl3-zeus:76598b2` |
| **`f4987cf`** | **production**, above | tags `exl3-zeus:serve-f4987cf` and `exl3-zeus:f4987cf` |
| `e24f059` | *"Always skip the MoE padding rows; the gate only ever turned off a win"* — drops the grid-size gate, `skip_pad = True`, overridable with `CUDA_EXL3_MOE_SKIP_PAD` | never a tag of its own; rode along inside the `9bf594c` checkout. Reasoned identical to `f4987cf` at our grid sizes, since the gate was always on here, so not re-measured `[not tested]` |
| `9bf594c` | *"Persist the MLA tuner cache across processes"* (`mla_decode.cu`) | tag `exl3-zeus:9bf594c`; built and gated, not adopted for serving |

The series also passed through `a1f992e` and `1699c89`; both were superseded within hours, their
tags removed, and no separate measurement of either survives.

## 2. Build the serve layer

Substitute your commit for `f4987cf` throughout, and keep the short sha in every tag — *"the
current build"* is not a revision. Steps 1–3 run on `head`; steps 4–8 run on all three nodes.

**1. Fetch and check out the pinned commit** (`git fetch origin` first if the checkout exists).

```
git clone https://github.com/Zeuss5/cuda-exl3.git ~/exl3-zeus/cuda-exl3
```

```
git -C ~/exl3-zeus/cuda-exl3 checkout f4987cf11806c7381c8a59cb388ab5863852679c
```

**2. Make a source tarball with `tar`, not `git archive`.** The build recipe
`docker/Dockerfile.gb10v8` lives in the checkout's `docker/` directory but is not tracked by
upstream git, so `git archive` drops it silently and the build then fails on a missing file. Exclude
`.git`, `build/`, `__pycache__` and `*.egg-info` so the context is source only.

```
tar czf /var/tmp/cuda-exl3-src-f4987cf.tgz --exclude=.git --exclude=build --exclude=__pycache__ --exclude='*.egg-info' -C ~/exl3-zeus cuda-exl3
```

**3. Record the tarball's sha256** — the identity that carries across nodes, because it is the
compiler's *input*, and unlike its output (section 5) it is stable. Ours:
`09942a2f5724a686e5852fff4d34de933861553bfed94d164225e080989433ab` `[measured-here]`.

```
sha256sum /var/tmp/cuda-exl3-src-f4987cf.tgz
```

**4. Copy it node to node directly**, not via a workstation, and compare the sha on each worker
against step 3 **before** extracting anything.

```
for H in 192.0.2.11 192.0.2.12; do scp /var/tmp/cuda-exl3-src-f4987cf.tgz $H:/var/tmp/ && ssh $H sha256sum /var/tmp/cuda-exl3-src-f4987cf.tgz; done
```

**5. Extract into a per-commit build directory**, on every node.

```
mkdir -p ~/exl3-zeus/build-f4987cf && tar xzf /var/tmp/cuda-exl3-src-f4987cf.tgz -C ~/exl3-zeus/build-f4987cf --strip-components=1
```

**6. Patch `MAX_JOBS` down, in the extracted copy only.** Upstream hardcodes `MAX_JOBS=12` inside
the `RUN` line, where it is not an `ARG` and cannot be overridden from the command line; twelve
parallel `nvcc` jobs do not fit under the memory cap in step 8, and three do. Leave the checkout
under `~/exl3-zeus/cuda-exl3/` alone — edit it there and every future build changes silently, while
the tarball's sha stops describing what was compiled.

```
sed -i 's/MAX_JOBS=12/MAX_JOBS=3/' ~/exl3-zeus/build-f4987cf/docker/Dockerfile.gb10v8
```

**7. Gate on free memory:** refuse to start below 6 GiB available. Ours just before the build was
**113 GiB (head), 116 (worker-1), 116 (worker-2)**, with no `oom-killer` or `oom_reaper` line in
`dmesg -T` on any node for the window. `[measured-here]`

```
free -g | awk '/^Mem:/ {print $7}'
```

**8. Build the serve layer**, from `~/exl3-zeus/build-f4987cf/`, on every node. `ARCH=12.1` is the
GB10 compute capability; `--memory=4g` keeps a runaway compile from taking the node down with it.

```
docker build --memory=4g -f docker/Dockerfile.gb10v8 --build-arg BASE=vllm/vllm-openai:glm53-flash-arm64-cu130 --build-arg ARCH=12.1 -t exl3-zeus:serve-f4987cf .
```

## 3. Build the port layer

Copy `patches/dflash2-port/` from this repository to `~/exl3-zeus/dflash2-port/` on every node and
build from that directory. The Dockerfile defaults `BASE` to `exl3-zeus:serve-f4987cf`, so for the
production commit the build argument is redundant; pass it to stack the same port on a different
serve layer. Our own builds used a per-commit copy of the Dockerfile with its `FROM` line rewritten
instead — the build argument is the same thing, done once. If the build stops with an `AssertionError`
out of `/opt/harem/dflash2-gate.py`, **no image is produced**: the design, not a mishap (section 7).

```
docker build --memory=4g --build-arg BASE=exl3-zeus:serve-f4987cf -t exl3-zeus:f4987cf .
```

## 4. What the build cost

Wall time for `f4987cf`, per node and per layer, `MAX_JOBS=3`, `--memory=4g` `[measured-here]`:

| node | serve layer | port layer | total |
|---|---|---|---|
| head | ~3 s (see below) | ~21 s | ~26 s |
| worker-1 | ~93 s | ~20 s | ~113 s |
| worker-2 | ~97 s | ~20 s | ~117 s |

The next commit in the series (`9bf594c`, same procedure, same nodes) put the serve layer at
**90.7 / 92.9 / 93.5 s** and the port layer at 19–20 s, with no discrepancy `[measured-here]`. Across
the series a full two-layer rebuild ran **2–5 min per node**, the spread tracking how much of
`gemm.cu` a commit touched: `76598b2`, a net deletion, was 2m16s–2m30s; `61a17bc` was 4.5–5 min.
Sizes match on every node — **~31 GB** for `exl3-zeus:f4987cf`, **~30.8 GB** for
`exl3-zeus:serve-f4987cf`, against 30.7 GB for the base. If your port layer is materially larger
than its serve layer, something recompiled that should not have. `[measured-here]`

**The unexplained timing artefact.** `head`'s outer wall-clock bracket for the serve layer was
~3 s, while that same build's own BuildKit step timer reported `#4 DONE 93.5s` for the `pip
install` step — the figure the other two nodes' brackets agree with. Checked and ruled out: clock
skew (`timedatectl` reported the clock synchronised, no clock-step entries in the journal for the
window), a Docker or buildx version mismatch (`29.2.1` / `buildx v0.31.1`, identical on all three)
and `ccache` (not installed, no `~/.ccache`, not named in the Dockerfile or `setup.py`). **Left
unexplained.** Recorded rather than rounded away: the correctness conclusions below rest on content
hashes of the build's outputs, not on this bracket, and in this series the anomalies that looked
cosmetic were twice the ones worth chasing. `[measured-here]`

## 5. Verification, and the negative result that matters

**The version print.** Expect `1.0.0` and `0.1.dev20051+g487ecf187` — the same vLLM as every image
in this series, since only `cuda_exl3` changes between them. `[measured-here]`

```
docker run --rm --entrypoint python3 exl3-zeus:f4987cf -c "import cuda_exl3, vllm; print(cuda_exl3.__version__, vllm.__version__)"
```

**The source change landed in the image.** Check inside the built image, not in the tarball it came
from. For `f4987cf` the new call site matches verbatim; all three nodes returned true on all three
checks `[measured-here]`. For a commit that *removes* something, invert the assertion: after
`76598b2` the image must contain none of `fuse_in`, `_FUSE_IN_OVERRIDE` or `_SMS`.

```
docker run --rm --entrypoint python3 exl3-zeus:f4987cf -c "import inspect, cuda_exl3.moe as m; s = inspect.getsource(m); assert 'skip_pad' in s and '_SMS' in s; assert 'ops.exl3_moe_had_in(xc, a13, layer.w13_suh.data, sorted_ids, expert_ids, n_rows, block_m, T, M * T, skip_pad)' in s; print('source change present')"
```

### Binary hashes cannot certify this build

The compiled `_C.cpython-312-aarch64-linux-gnu.so` (24,778,176 bytes everywhere) differed on all
three nodes for `f4987cf`:

| node | whole-file sha256 | ELF Build-ID |
|---|---|---|
| head | `61388925f0839ed3…` | `0e5c5082f5c6c80d08c3ccd2d6735710cdae7ad2` |
| worker-1 | `ab9ec37bdd06ced0…` | `6e8fe3c4b963bbb473ee59151fba67bc85c44509` |
| worker-2 | `42de96122b4a15d0…` | `6e8fe3c4b963bbb473ee59151fba67bc85c44509` |

Run down rather than waved off `[measured-here]`:

- **The divergence is entirely in `.nv_fatbin`**, the embedded device-code container. `.text`,
  `.rodata`, `.data` and `.comment` hashed identically on all three, and the `.so` is fully stripped,
  so there is no `.symtab` to blame either. A raw byte diff of the section hit ~88 % of its
  10,240,192 bytes — the signature of a compressed stream in which one upstream difference cascades,
  not of a header or timestamp field.
- **The device code really differs**, not just its wrapper: the three cubins extracted decompressed
  (`cuobjdump --extract-elf all`) hashed differently for `gemm.sm_121.cubin`, `hadamard.sm_121.cubin`
  and `mla_decode.sm_121.cubin` on `head` versus the two workers, at identical sizes.
- **Every plausible cause was proven identical across the nodes**: base image digest, `nvcc`/`ptxas`
  version (`release 13.0, V13.0.88`, same build string), driver `580.173.02`, GPU model `NVIDIA GB10`
  and `nproc` = 20 — with the tarball sha verified equal before extraction, so the compiler's input
  is provably the same everywhere.
- **The decisive test.** A second, fully independent `docker build --no-cache` of the serve layer
  **on the same machine** produced *the other two nodes'* cubins and *their* Build-ID.

**Conclusion: this is build-to-build nondeterminism in the NVCC/ptxas toolchain**, most plausibly in
SASS-level codegen, since the fatbin is the only thing that moves — not a per-node hardware or
toolchain fingerprint. So **neither a whole-file sha256 nor the ELF Build-ID is a reliable cross-node
identity check for this stack**, and neither is a per-cubin hash.

**`[retracted]`** — an earlier build in this series found only three bytes differing, inside an
`nvcc` temp-file name in `.symtab`, and we wrote down the rule *"use the ELF Build-ID as the
authoritative cross-node identity check, not a naive whole-file sha256"*. That rule is wrong:
Build-ID is not stable across independent builds of provably identical source here. **What replaces
it — only behaviour certifies a build**: the upstream pytest suite (section 6), then the serving
gates ([09](09-measurement-protocol.md)). We kept each node's *first* build, odd one included;
nothing shows either codegen wrong, only that two valid ones exist. Which is faster is unmeasured
`[not tested]`.

## 6. The test gate

Run the upstream suite from the extracted source, inside the built serve layer, on at least one node.
Expect **`44 passed, 41 skipped`** in about 12 s, exit 0; the 41 skips are all in `test_exl3_gemm.py`,
conditional on `CUDA_EXL3_TEST_MODEL`, which we do not set. `[measured-here]`

```
docker run --rm --memory=3g --gpus all -v ~/exl3-zeus/build-f4987cf:/src -w /src --entrypoint python3 exl3-zeus:serve-f4987cf -m pytest tests/
```

**Check that count against your previous build's; do not assume it.** The number moved once already
in this series — an earlier image ran 42/41 — and a suite that quietly gains or loses a test between
commits is itself the finding. Record it in `results/` with the commit on both sides.

## 7. What the second layer contains

The port layer replaces twelve vLLM Python files and adds nothing compiled. The base image already
ships DFlash v1, so the port is the *delta* to DFlash2 — derived as a `git merge-file` 3-way merge
of upstream vLLM commit `b389ac294` onto the image's own tree (byte-identical to vLLM `487ecf187`),
never a hand copy — plus the two things that delta does not cover: the EAGLE3 aux-hidden-state
interface on the GLM-5.3 target side, and a KV-cache grouping change giving the draft its own group.
Three additions are ours: a fail-closed check refusing a quantized drafter or one whose head counts
do not divide the TP size, one diagnosis knob (`HAREM_DFLASH2_FORCE_TORCH_TOPK=1`), and the build
gate. Full detail, including what was deliberately **not** ported and why, is in
[04](04-dflash2-port.md) and in `patches/dflash2-port/PATCHES.md`, beside the files themselves.

| file replaced | role |
|---|---|
| `vllm/model_executor/models/qwen3_dflash2.py` | new: the DFlash2 draft head, grouped convolution, candidate selector |
| `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` | new: `DFlash2Speculator`, the selector-walk and draft-logits Triton kernels |
| `vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py` | new: package marker |
| `vllm/model_executor/models/registry.py` | registers `DFlash2DraftModel` |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | `init_speculator` dispatches to DFlash2 |
| `vllm/config/vllm.py` | forces the V2 model runner for a DFlash2 draft |
| `vllm/v1/worker/gpu/spec_decode/speculator.py` | adds the `draft_logits_spec()` hook |
| `vllm/v1/worker/gpu/sample/gumbel.py` | extracts `gumbel_noised_argmax` so draft and target share one noise draw |
| `vllm/model_executor/layers/logits_processor.py` | adds the vocab-parallel `get_top_k_tokens` |
| `vllm/model_executor/models/qwen3_dflash.py` | subclass hooks only; DFlash v1 behaviour unchanged |
| `vllm/models/glm5next/nvidia/model.py` | target side: the EAGLE3 aux-hidden-state interface |
| `vllm/v1/core/kv_cache_utils.py` | gives the draft its own KV cache group |

**The gate is the point of this layer.** Every symbol it checks guards a hook whose absence would
*not* raise at runtime: the engine would boot, produce plausible text, and quietly lose acceptance.
`patches/dflash2-port/gate.py` runs inside the `docker build`, so a broken port produces no image at
all. It stays in the image, re-runnable at any time; expect one line, `DFLASH2 PORT BUILD GATE: OK`.
`[measured-here]`

```
docker run --rm --entrypoint python3 exl3-zeus:f4987cf /opt/harem/dflash2-gate.py
```

## 8. What to keep and what to throw away

**Keep**, per node and per commit: the source tarball (`/var/tmp/cuda-exl3-src-<commit>.tgz`, plus a
copy under `~/exl3-zeus/`), the extracted build context (`~/exl3-zeus/build-<commit>/` — the pytest
run mounts it), the full build log and the pytest log. They are what answers "what is in this image"
six weeks later, since the binaries cannot.

**Do not tag anything `latest`**: every A/B here compares two named commits, and `latest` makes a
sweep unattributable. **Do not push** to any registry — the base image forbids redistribution.
**Keep two or three tags**, not one and not twelve: an A/B needs the incumbent alongside the
challenger, and a 31 GB image is not free. Retiring a tag means retiring both halves
(`exl3-zeus:<commit>`, `exl3-zeus:serve-<commit>`), saying in writing why, and checking first that
no container references it.

## 9. What this cost

- **Build time.** `MAX_JOBS=3` under `--memory=4g` compiles four times narrower than upstream's
  `MAX_JOBS=12`: 90–95 s of serve layer where a wide build would be faster. It buys a build that
  cannot exhaust the node — right here, wrong on a machine with build headroom. `[measured-here]`
- **Disk.** ~31 GB per port-layer tag plus ~30.8 GB per serve layer on each of three nodes, before
  the checkpoint — roughly 90 GB per node for three kept commits. `[measured-here]`
- **The nondeterminism finding cost a rebuild and about a quarter-hour of investigation**, and
  removed a check we thought we had. It bought the correct rule in exchange — behaviour, not bytes.
- **Quality:** nothing here touches numerics; both codegens pass the same suite.
- **Speed:** we looked for a price and could not price it. The two codegen variants were never
  benchmarked against each other, so "no speed difference" is unproven, not measured `[not tested]`.

## 10. Open points in this layer

- `docker/Dockerfile.gb10v8` is not tracked by upstream. If your checkout lacks it you cannot run
  step 8 as written, and ours is not in this repository either — [11](11-open-issues.md).
- The base image's provenance is unknown to us, publisher and lifetime both. Pull by digest.
- Which NVCC codegen variant is faster is unmeasured. `[not tested]`
- `e24f059` was never measured on its own; we reasoned it identical to `f4987cf` here. `[not tested]`

# GLM-5.3-Flash (EXL3 4bpw) on 3× NVIDIA DGX Spark — vLLM, cuda-exl3, TP=3 + EP, DFlash2

> **Status: work in progress, private draft.** This repository is being written while the work is
> still going on. Numbers, flags and patches are current as of **5 September 2026** and will be
> overwritten in place as the stack moves. Nothing here has been reviewed for release yet. Sections
> marked *open* in [docs/11-open-issues.md](docs/11-open-issues.md) are the honest edge of what we know.

A reproducible recipe for serving **`zai-org/GLM-5.3-Flash`** as an **EXL3 4-bit** checkpoint on three
DGX Spark (GB10) nodes with vLLM and the `cuda-exl3` kernels: the two-layer image build, expert
parallelism over 288 experts, the TP=3 shape padding, the DFlash2 speculative-decoding port, the
kernel bugs we found and what fixed them, the NCCL mesh cliff, the second cable nothing was using,
the KV-pool surgery, a 274-second cold boot, and what is still broken. Written so that a person **or
their AI coding agent** can follow it step by step.

This is the EXL3 sibling of our NVFP4 recipe,
[`NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark`](https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark),
which serves the same model on the same three nodes through a different quantization path. Cluster
setup, fabric wiring, the DGX OS update story and the memory rules are shared between the two; where
this repository is thin on those, that one is thorough.

> **About the name "HAREM".** HAREM is simply the name we gave our three-node setup. It is hardcoded
> in several places in the stack — patch markers (`HAREM-TP3`, `HAREM-GB10-TOPK`), environment
> variables (`HAREM_SW_BLOCK_SIZE`, `HAREM_DISABLE_PERSISTENT_TOPK`, `HAREM_FASTLOAD_*`,
> `HAREM_EP_ZERO_MODE`), function names (`_harem_*`), one module (`harem_fastload.py`), some image
> tags and log lines. You can keep them. If you rename them, grep the whole repository first: several
> of these strings are matched exactly by the patch scripts, which fail closed when an anchor stops
> matching.

## Headline results

Settings for every row: image `exl3-zeus:9bf594c`, TP=3 + expert parallel, EXL3 4bpw weights,
`kv-cache-dtype fp8`, DFlash2 draft at k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`,
`--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`,
`gpu-memory-utilization 0.80`, per-rank pre-sliced sidecar, warm MLA tuner cache, mesh plugin with
both cables per peer and `NCCL_PTR_CUDA`, temperature 0, reasoning effort **low**, 5 September 2026.
Speed is the median of three sweep rounds — the persisted tuner cache is what makes three enough; on
an image without it the rule is still five rounds with two discarded
([docs/09](docs/09-measurement-protocol.md), [docs/12](docs/12-tuner-cache.md)).

| What | Result | Notes |
|---|---|---|
| Quality gates | correctness probe 10/10, code exam 12/12 | cold **and** after a full benchmark, every arm `[measured-here]` |
| MMLU sample (35 questions per subject, 1,995 q) | 86.4 ±0.7 | measured at TP=2 on the same checkpoint; not re-run at TP=3 `[measured-here]` |
| Speed, realistic (12 short English code prompts) | C1 **56.9** tok/s total (63.6 per stream) · C8 **168.9** tok/s total (26.0 per stream) | acceptance 61–65 %, 5.3–5.5 accepted tokens per step `[measured-here]` |
| Prefill, fresh unseen ~8.3K prompts | **1,792** tok/s (warm repeated 7K prompt: 1,506) | a repeated prompt reads the prefix cache and lies — see [docs/09](docs/09-measurement-protocol.md) `[measured-here]` |
| TTFT | 0.41 s at C1, 1.01 s at C8 | `[measured-here]` |
| KV pool | **4,449,035 tokens** at `gpu-memory-utilization 0.80` | about 4.4 concurrent 1M-token requests `[measured-here]` |
| Weights per node | 54.9 GiB | against 81.5 GiB at TP=2 — the whole reason for the third node `[measured-here]` |
| Cold boot, container start → API ready | **274 s** (~4.5 min) | was 618 s before the loader work; itemised on the fast-boot arm and unchanged since `[measured-here]` |
| Free host RAM at rest / swap | 11.3 / 12.6 / 12.5 GiB · ~0.1 GiB | rule: never below 4 GiB free `[measured-here]` |
| Speed by category, C1 | code 47.9 · math 59.0 · JSON 57.7 · prose 22.4 tok/s | acceptance 46 / 56 / 55 / **13 %** — prose is where a k=7 draft is wasted. **Measured one configuration earlier**; not re-run since `[not tested]` |

For reference, our NVFP4 stack on the same three nodes reaches C1 57–60, C8 150, prefill 1,585 and a
KV pool of 4.32M at `gpu-memory-utilization 0.88`. **EXL3 at TP=3 is now level on single-stream
decode and ahead on aggregate throughput, prefill, memory and boot.** Read that with one caveat
attached: three of the changes that got it there — the channel cap, the second cable and
`NCCL_PTR_CUDA` — are **fabric-level, not format-level**, they use the same plugin over the same
wiring, and none of them has been applied to the NVFP4 stack `[not tested]`. If they transfer, most
of the gap goes with them. Neither stack has been measured at max reasoning effort.

**All benchmark numbers here are at reasoning effort `low`, temperature 0.** Max effort would cost
5–12× the tokens and days of cluster time; we did not spend them. Nothing on this page is a max-effort
number, and none of it should be quoted as one.

**Synthetic versus realistic.** Every speed number above is realistic — real prompts, mixed content,
a draft model that sometimes misses. Synthetic prompts ("count from 1 to 200") measure the
speculative-decoding *ceiling* and run far faster; they are labelled as such wherever they appear and
they will disappoint you in real use. See [docs/09](docs/09-measurement-protocol.md).

## Read in this order

1. [00 — Hardware and OS](docs/00-hardware-and-os.md) — three Sparks, the fabric, the versions we ran, desktop off.
2. [01 — Model and license](docs/01-model-and-license.md) — the EXL3 checkpoint, its pinned revision, and its licence, which is not one you have seen before.
3. [02 — Image build](docs/02-image-build.md) — the two-layer Docker recipe, pinned to a `cuda-exl3` commit.
4. [03 — TP=3 padding and sidecars](docs/03-tp3-padding-and-sidecars.md) — why an EXL3 tensor cannot be split three ways, and the shape surgery that makes it possible anyway.
5. [04 — The DFlash2 port](docs/04-dflash2-port.md) — porting a speculative decoder into an image that had never seen one.
6. [05 — Expert parallel and the cuda-exl3 fixes](docs/05-expert-parallel-and-cuda-exl3-fixes.md) — the one-line kernel bug that cost 45 % of the MoE stage.
7. [06 — The NCCL mesh](docs/06-nccl-mesh.md) — 0.6 GB/s in the middle of the size range and the one environment variable that fixed it; then the second cable of every pair, which had never carried a packet.
8. [07 — KV pool and the draft page](docs/07-kv-and-draft-page.md) — why the pool was capped by a counter, not by memory.
9. [08 — Fast boot](docs/08-fast-boot.md) — 618 s → 274 s, and the bit-identity proof that makes it safe.
10. [09 — Measurement protocol](docs/09-measurement-protocol.md) — five rounds, discard two; and four ways to measure a lie.
11. [10 — Results and roofline](docs/10-results-and-roofline.md) — the full tables and how close to the hardware roof we are.
12. [11 — Open issues](docs/11-open-issues.md) — what is unresolved, what we retracted, and what we never ran.
13. [12 — The MLA tuner cache](docs/12-tuner-cache.md) — the measurement tax a process-local cache was charging, and the shorter protocol that removes it.
14. [CREDITS](CREDITS.md) · [LICENSES](LICENSES.md) · [CHANGELOG](CHANGELOG.md) · [CONTRIBUTING](CONTRIBUTING.md) · [STYLE-GUIDE](STYLE-GUIDE.md)

## Quick path (for an AI coding agent)

```text
0.  Read docs/00 and confirm three DGX Spark nodes, DGX OS up to date, ibv_devinfo 4/4 on each.
1.  Read docs/01 and download brandonmusic/GLM-5.3-Flash-tr3-4bpw at revision b20c49ba (175.6 GB).
    Read its LICENSE first: it is the ShapleyMCG License 1.0, not MIT and not Apache.
2.  Download incoai/GLM-5.3-Flash-DFlash2 (the draft). It is CC BY-NC-ND 4.0 and our permission
    for it does not transfer to you (LICENSES.md). The recipe runs without it, about 2.6x slower
    at a single stream.
3.  Build the NCCL mesh plugin on every node from autoscriptlabs/nccl-mesh-plugin at commit
    19924dcc, per its README, with patches/kernel/0004, 0005 and 0006 applied. Do not skip docs/06:
    the default channel count costs 13 % of C8, and unpatched the plugin uses one cable of each
    pair. Read your own port_xmit_data counters before believing any bandwidth number.
4.  Build the image (docs/02) on every node from the same source tarball; verify by behaviour
    (pytest) rather than by binary hash - see the nondeterminism note in docs/02.
5.  Build the two sidecars (docs/03): the padded model sidecar and the padded drafter sidecar.
    They are symlink trees plus a handful of rewritten config files, not copies of the weights.
6.  Copy scripts/ and patches/ to ~/exl3-zeus/ on every node. Derive each node's env from
    envs/env.tp3.example with sed, per node, never by copying the file between nodes (docs/03).
7.  Boot: worker-2 first, then worker-1, then head. Run the two quality gates
    (scripts/correctness-probe.py, scripts/code-exam.py) before believing any speed number.
    Point CUDA_EXL3_TUNE_CACHE at a directory under the /cache mount (docs/12) before the first
    boot, or every benchmark you run will be measuring the tuner rather than the change.
8.  Optional but recommended: build the per-rank fast-load sidecar (docs/08). It takes one
    ~11-minute dump boot and cuts every subsequent boot from 618 s to 274 s.
```

Evidence tiers used throughout: `[measured-here]`, `[measured-here, raw lost]`, `[reported]`,
`[estimate]`, `[not tested]`, `[retracted]`.

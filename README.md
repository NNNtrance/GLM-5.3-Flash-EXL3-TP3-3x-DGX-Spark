# GLM-5.3-Flash (EXL3 4bpw) on 3× NVIDIA DGX Spark — vLLM, cuda-exl3, TP=3 + EP, DFlash2

> **Status: work in progress, private draft.** This repository is being written while the work is
> still going on. Numbers, flags and patches are current as of **5 September 2026** and will be
> overwritten in place as the stack moves. Nothing here has been reviewed for release yet. Sections
> marked *open* in [docs/11-open-issues.md](docs/11-open-issues.md) are the honest edge of what we know.

A reproducible recipe for serving **`zai-org/GLM-5.3-Flash`** as an **EXL3 4-bit** checkpoint on three
DGX Spark (GB10) nodes with vLLM and the `cuda-exl3` kernels: the two-layer image build, expert
parallelism over 288 experts, the TP=3 shape padding, the DFlash2 speculative-decoding port, the
kernel bugs we found and what fixed them, the NCCL mesh cliff, the second cable nothing was using,
the PCIe wall behind it, the KV-pool surgery, a 274-second cold boot, a measured breakdown of where a
prefill and a decode step actually go, and what is still broken. Written so that a person **or their
AI coding agent** can follow it step by step.

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
`kv-cache-dtype fp8` **and an fp8 draft cache** (`HAREM_DRAFT_KV_DTYPE=fp8`), DFlash2 draft at k=7,
`--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, `gpu-memory-utilization 0.80`, per-rank pre-sliced sidecar, warm MLA tuner
cache, mesh plugin with both cables per peer and `NCCL_PTR_CUDA`, the launcher's memory settle gate,
temperature 0, reasoning effort **low**, 5 September 2026. This is **production configuration 7**.
Speed is the median of three sweep rounds — the persisted tuner cache is what makes three enough; on
an image without it the rule is still five rounds with two discarded
([docs/09](docs/09-measurement-protocol.md), [docs/12](docs/12-tuner-cache.md)).

| What | Result | Notes |
|---|---|---|
| Quality gates | correctness probe 10/10, code exam 12/12 | cold **and** after a full benchmark, every arm `[measured-here]` |
| MMLU sample (35 questions per subject, 1,995 q) | 86.4 ±0.7 | measured at TP=2 on the same checkpoint; not re-run at TP=3 `[measured-here]` |
| Speed, realistic (12 short English code prompts) | C1 **57.0** tok/s total (64.0 per stream) · C8 **175.1** tok/s total (26.9 per stream) | acceptance 61–64 %, 5.3–5.5 accepted tokens per step `[measured-here]` |
| Prefill, fresh unseen ~8.5K prompts | **1,769** tok/s (warm repeated 7K prompt: 1,529) | a repeated prompt reads the prefix cache and lies — see [docs/09](docs/09-measurement-protocol.md) `[measured-here]` |
| TTFT | 0.34 s at C1, 0.91 s at C8 | `[measured-here]` |
| KV pool | **4,699,724 tokens** at `gpu-memory-utilization 0.80` | about 4.7 concurrent 1M-token requests; read from a load boot with a settled baseline `[measured-here]` |
| Weights per node | 54.9 GiB | against 81.5 GiB at TP=2 — the whole reason for the third node `[measured-here]` |
| Cold boot, container start → API ready | **~274 s** (~4.5 min) | was 618 s before the loader work; itemised on the fast-boot arm, plus the settle gate's wait since `[measured-here, raw lost]` |
| Free host RAM at rest / swap | 12.3 / 13.5 / 13.3 GiB · ~0.1 GiB | rule: never below 4 GiB free `[measured-here]` |
| Speed by category, C1 | code 47.9 · math 59.0 · JSON 57.7 · prose 22.4 tok/s | acceptance 46 / 56 / 55 / **13 %** — prose is where a k=7 draft is wasted. **Measured three configurations earlier**; not re-run since `[not tested]` |

**Production configuration 7 changed one number and one instrument.** For two configurations before it,
a day of profiling, a MoE re-read bench, a hyper-connection analysis, a one-sided RDMA_WRITE transport,
a fused hyper-connection kernel and a dual-batch-overlap design study moved **no** production number
between them; each was a measurement and most closed an item
([10](docs/10-results-and-roofline.md) §5, [06](docs/06-nccl-mesh.md) §9–§10,
[11](docs/11-open-issues.md) §2.16–§2.19). Production 7 moves the **KV pool, +5.6 % to 4.70M**, by
putting the DFlash2 drafter's own cache at fp8 — speed is unchanged inside its bands, TTFT is better at
both ends, draft acceptance and the gates are where they were
([07](docs/07-kv-and-draft-page.md) §7, [11](docs/11-open-issues.md) §2.18). Where a step goes is now
measured rather than inferred: per 2,048-token prefill chunk, MoE trellis GEMM 26.4 %, NCCL all-reduce
16.5 %, dense BF16 GEMM 16.2 %, hyper-connection mixing 11.7 %; per C1 decode step, dense BF16 GEMM
44.8 % and the k=7 drafter 19.5 %. Both rulers those percentages are against were measured on the
device — 225 GB/s and 97.3 TFLOP/s, not the 273 and ~125 the datasheet implies.

**The other half of production 7 buys no tokens at all, and it is the part worth copying.** The KV
pool number this stack has spent a dozen arms on turned out to be a *difference* between two readings
of `/proc/meminfo`, taken minutes apart — and it runs backwards: a node that starts with less memory
free awards itself a **larger** pool. Because the launcher killed a ~90 GiB container and started the
next one immediately, and the nodes start in a fixed order, the last node started was systematically
9 GiB short — **27 % of a rank's KV allowance** sitting inside the measurement. No published figure
here is known to be wrong: the pool takes the minimum over ranks and the polluted node happened never
to be the binding one, which is luck rather than design. A host-side wait for memory to settle, before
`docker run`, took the per-rank spread from 9 GiB to 1.4 GiB and made the difference between an
explanation and an artefact checkable. It also refuted this repository's
largest open item, which had claimed 8.2 GiB per worker was stranded and that equalising the ranks was
worth 8–26 % of pool; acting on it would have over-committed the head node
([07](docs/07-kv-and-draft-page.md) §1.1, [08](docs/08-fast-boot.md) §5.1,
[11](docs/11-open-issues.md) §2.3).

**Two `NCCL_ALGO` arms and one memory rung are the open edge.** `Tree` is measured and rejected on
this fabric — 4–6× slower on the sizes that matter — while `Ring,Tree` came back *inside the sweep's
own repeat spread* and is deferred to a five-round engine arm; `Ring` stays
([06](docs/06-nccl-mesh.md) §12.2). The `gpu-memory-utilization 0.83` rung is designed and costed at
about **+11 % of pool** for ~3.9 GB of host headroom, and it waits on the stack owner's approval rather
than on a measurement ([11](docs/11-open-issues.md) §2.4). Upstream, the `cuda-exl3` MoE stage is now **closed**: `62f53e6`
bounded what was left of `exl3_moe_had_in` at half-ALU work worth ≤2 % of prefill and unreachable in
practice, so every remaining prefill lever on this stack belongs to vLLM, to the fabric, or to us.

For reference, our NVFP4 stack on the same three nodes reaches C1 57–60, C8 150, prefill 1,585 and a
KV pool of 4.32M at `gpu-memory-utilization 0.88` — against this stack's 4.70M at 0.80. **EXL3 at TP=3 is now level on single-stream
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
7. [06 — The NCCL mesh](docs/06-nccl-mesh.md) — 0.6 GB/s in the middle of the size range and the one environment variable that fixed it; then the second cable of every pair, which had never carried a packet; then the PCIe slot that was the ceiling all along, and the transport rewrite that proved it.
8. [07 — KV pool and the draft page](docs/07-kv-and-draft-page.md) — why the pool was capped by a counter, not by memory.
9. [08 — Fast boot](docs/08-fast-boot.md) — 618 s → 274 s, and the bit-identity proof that makes it safe.
10. [09 — Measurement protocol](docs/09-measurement-protocol.md) — five rounds, discard two; and four ways to measure a lie.
11. [10 — Results and roofline](docs/10-results-and-roofline.md) — the full tables, the rulers measured rather than quoted, and where a prefill and a decode step actually go, class by class.
12. [11 — Open issues](docs/11-open-issues.md) — what is unresolved, what we retracted, and what we never ran.
13. [12 — The MLA tuner cache](docs/12-tuner-cache.md) — the measurement tax a process-local cache was charging, and the shorter protocol that removes it.
14. [systemd](systemd/README.md) — a unit **template**, not installed anywhere, with the three things wrong with it named. This stack starts by hand; read this before a reboot.
15. [CREDITS](CREDITS.md) · [LICENSES](LICENSES.md) · [CHANGELOG](CHANGELOG.md) · [CONTRIBUTING](CONTRIBUTING.md) · [STYLE-GUIDE](STYLE-GUIDE.md)

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

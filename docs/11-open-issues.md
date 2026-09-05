# 11 — Open issues, retractions, and what we never ran

This stack is not finished. This page is the honest edge of it: what is unresolved, what we published
and then withdrew, and what we simply have not measured. Nothing here is hidden in a footnote
elsewhere.

---

## 1. Retracted

Six things we wrote down as findings and later measured properly, plus two smaller ones. Each was
published — in a report, an upstream issue, or both — before it was corrected.

### 1.1 "The missing `n_rows` also costs the non-expert-parallel path"

We reported that a kernel bug affecting expert parallelism also cost an ordinary tensor-parallel
arrangement, reasoning that with no expert map the surplus tail is a valid local expert on every
rank. **Wrong** `[retracted]`. Running the alignment directly on our own build shows the tail is
`-1` everywhere when there is no map; only `expert_map[expert_ids]` converts it, by negative
indexing. The kernel author's reading was right. Corrected in
[05](05-expert-parallel-and-cuda-exl3-fixes.md) §2, and corrected upstream in the issue thread.

### 1.2 "The extra masking pass is under 1 % of the MoE layer"

Our estimate assumed the pass covered the routed rows; it covered the allocated rows. Measured share:
2.9 % at M=8 rising to **15.8 % at M=2048** `[retracted]`.

### 1.3 "One upstream build is ~10 % slower end to end, and we do not know why"

It does not reproduce. What we measured was boot-to-boot and warm-up variance — **15.9 % spread on
C8 with nothing changed at all**, on the same image and the same environment file `[retracted]`. We
had drawn a kernel conclusion from a single pair of sweeps. This produced the five-round protocol in
[09](09-measurement-protocol.md) §1–2, which is the most useful thing in this repository.

### 1.4 "NCCL is choosing the LL protocol at 16 MB and leaving half the link on the table"

Read off a profiler kernel name. Forcing LL at 16 MB costs 20,114 µs against auto's 1,787 µs, so the
tuner plainly is not choosing it `[retracted]`. The real finding underneath was a different one
entirely — see [06](06-nccl-mesh.md).

### 1.5 "The MLA tuner re-tunes on every prefill chunk, so 2.6 % of prefill is wasted"

Measured on both sides of the relevant upstream fix: the tuner mints **4–5 events per set of fresh
prefills**, each triggering roughly 350 eviction calls — a one-off cost that settles as batch shapes
repeat, not continuous re-tuning `[retracted]`. We withdrew the proposal to disable tuning. The axis
that actually varies on this model is the batch size, not the top-k the fix bucketed.

### 1.6 "The mesh ceiling is ~13 GB/s against a 25 GB/s link, and GPUDirect is the fix we cannot make"

Two errors in one sentence, both published here, both corrected on 5 September `[retracted]`. The
"25 GB/s link" is a **pair of cables at 50 GB/s**, and the plugin was using one of them — the second
cable of every pair had transmitted exactly zero bytes since driver load, on all three nodes. And
recovering the device-pointer path was **two lines** in `getProperties` plus a real flush, not the
receiver-advertised-FIFO redesign we had priced it at, because the plugin's `regMr` already
registered CUDA pointers and was simply never handed one.

What replaced it: the all-reduce reaches **20.84 GB/s** at 64 MB against 12.08 before, and the engine
went from C8 159.9 to **168.9**. See [06](06-nccl-mesh.md) §6–§7. The lesson we would rather have
learned earlier is in §4 of this page.

### 1.7 Two smaller ones

- **"The gemm should zero a retired tile rather than return."** We shipped it; upstream's design is
  10.7 % cheaper per MoE layer at M=2048. Our patch is retired in writing
  (`patches/kernel/0002-RETIRED.md`) `[retracted]`.
- **"`--no-enable-flashinfer-autotune` is worth 34 s of provably empty work."** It is worth about
  **3.5 s**: the autotune was also doing JIT and kernel warm-up, which simply moved into graph
  capture `[retracted]`. See [08](08-fast-boot.md).

---

## 2. Open, with a known next step

### 2.1 The unquantized half of the model

~37 % of decode GPU time and ~20 % of prefill is BF16 dense GEMM, because the checkpoint quantizes
routed experts only. Nothing in the EXL3 kernel library touches it, and at TP=3 each rank's share is
a third rather than a half, so those kernels get *less* efficient as ranks are added. **This is the
largest remaining structural item on the stack.**

The obvious answer — a checkpoint that also quantizes attention — is the one that cannot run at TP=3
today, because with attention quantized there is no unquantized dimension left to split three ways
([01](01-model-and-license.md) §3.1). Next step: nothing cheap. It is either a differently scoped
quantization, or pipeline parallelism, which we have not evaluated at all `[not tested]`.

### 2.2 The mesh plugin's residual RNR, and what the fabric is worth now

`NCCL_MAX_NCHANNELS=8` drives the receive-not-ready counters to about zero in the microbenchmark but
**not in the engine**: a live counter read across a full sweep plus prefill plus mixed load showed
roughly 42,000 events per node over five minutes, on the order of 1–3 % of wall clock
`[measured-here]`. That read was taken on the single-cable, host-bounce-buffer configuration and has
**not been repeated** since the plugin patches landed `[not tested]`, although model-free the retries
per operation fell from ~15 to ~3 with `NCCL_PTR_CUDA` `[measured-here]`.

`patches/kernel/0004-min-rnr-timer.patch` is now carried in the production plugin build alongside
0005 and 0006, at `NCCL_MESH_MIN_RNR_TIMER=1`. What it is worth **on its own** in the engine is still
unmeasured.

Also open: **12 channels was never taken to the engine**, and now needs re-measuring rather than
carrying forward — it was indistinguishable from 8 on one cable, and 16 turned out to be badly wrong
on two ([06](06-nccl-mesh.md) §8.1), so the channel arithmetic over two cables is not the same
question `[not tested]`.

The deeper fix is still upstream's to make: a receiver-advertised buffer FIFO with `RDMA_WRITE`,
which would remove the stall from the steady state instead of making it cheap. It is a wire-format
change of ~650–850 lines. With the cap and both patches in place, the remaining prize is small.

**Retracted here** `[retracted]`: an earlier version of this section said the disabled GPUDirect path
"holds the ceiling at ~13 GB/s against a 25 GB/s link". Both halves were wrong. The link is a *pair*
of cables at 50 GB/s and one of them was idle; and the device-pointer path was two lines away, not a
redesign away. Both are fixed and the all-reduce now reaches 20.8 GB/s ([06](06-nccl-mesh.md) §6–§7).

### 2.3 The pool is sized by the smallest rank

The head node has 35.40 GiB available for KV and the binding rank has 32.85 GiB, so **2.55 GiB on the
head node is simply unused** `[measured-here]`. The asymmetry comes from rank 0 carrying the API
server and the drafter's own overhead. Nothing has been tried here `[not tested]`.

### 2.4 The memory ladder above 0.80

0.85 was measured and rejected on the free-memory rule; 0.88 was never attempted. One lead: the
fast-boot work removed a large page-cache spike during loading, so a rung at 0.82–0.83 may sit
differently now from how it sat before `[not tested]`. See [07](07-kv-and-draft-page.md) §6.

### 2.5 `--max-num-batched-tokens 3072`

2048 and 4096 were both measured; the intermediate value was not `[not tested]`. It is the obvious
probe if you want some of the prefill back without all of the KV pool.

### 2.6 `block_m` under expert parallelism

The alignment needs the global expert count and the block-size heuristic is about rows per expert;
the two uses may want different numbers. Worth a sweep `[not tested]`.

### 2.7 The fast-load read path

The sidecar reads at 0.88–1.04 GB/s while a different loader on the same NVMe reaches 3.1 GB/s, so
roughly 3× is still on the table — boot 67 s of weight load could become ~20 s, and the total ~230 s.
An mmap arm exists in the code and has **never been run** `[not tested]`. Beyond that, a buffered
multi-threaded reader.

After that, the profile run at 67–73 s becomes the largest remaining item; its first step alone is
about 45 s and contains the first NCCL collective and the MLA tune.

### 2.8 A persisted MLA tuner cache — closed

**Done.** Upstream's `9bf594c` persists the cache behind `CUDA_EXL3_TUNE_CACHE`; we built the image,
wired it into the environment and measured it: **18 tune events before serving → 0**, no events during
a sweep, and round 1 is no longer a penalty. The sweep protocol dropped from five rounds to three.
Kept here as a closed item rather than deleted, because the *sub*-items it left behind are open:

- the cache key does not know about our kernel patches, so a patch that changed the candidate grid
  would need the file cleared by hand, and nothing enforces that `[not tested]`;
- the boot-time part of the saving was never isolated from phase 4 of the ledger `[not tested]`;
- three rounds is calibrated on **one** cold/warm pair at one configuration.

Full account in [12](12-tuner-cache.md).

### 2.9 The 2,304-padded alternative to expert parallelism

Encoding the routed experts at a padded 2,304 width would let them be tensor-sliced three ways
instead of distributed whole. It was designed, the sidecars were built and verified, and it was
**not adopted**: on the fixed kernel it wins only at M=1 (1.50×), is a wash at M=16, and loses 14 %
at M=2048, while costing +12.5 % expert bytes straight out of the KV pool `[measured-here]`. With a
k=7 draft the real decode batch is M=8 at one stream and M=64 at eight — both in the region where
expert parallelism wins. The sidecars exist on disk as an option and nothing uses them.

### 2.10 DMA-BUF registration on the mesh plugin

`NCCL_MESH_DMABUF=1` works — `ibv_reg_dmabuf_mr` accepts these buffers on this platform, which was an
open question — and is **slower** than plain `ibv_reg_mr` across the size range (64 MB all-reduce
18.08 against 20.84 GB/s) `[measured-here]`. Rejected on the measurement; not investigated further,
so the *why* is `[not tested]`.

### 2.11 The RDMA_READ flush

`NCCL_MESH_FLUSH=0` measures inside noise of flush-on, and better at a couple of sizes. We keep the
flush because coherence is not a noise-level decision, but "inside noise" over two repetitions is not
the same as "free", and it has never been taken to the engine either way `[not tested]`.

---

## 3. Never run

| What | Why not |
|---|---|
| Anything at **max reasoning effort** | Days of cluster time. Everything published here is at `low`. |
| **MMLU at TP=3** | The 1,995-question sample was run at TP=2. The gates are identical between arrangements, so there is no signal that justifies the hours — but it is an absence. |
| **IFEval, GSM8K, needle-in-a-haystack, tool-eval-bench, ExtractBench** | All exist for the NVFP4 sibling recipe; none re-run on this stack. Anyone comparing the two on quality should treat this repository as having the gates and one MMLU sample. |
| **The newer checkpoint revision** (`aba59d21`, four days newer than the one we pinned) | Not tested. |
| **`NCCL_MAX_NCHANNELS=8` on the NVFP4 stack** | Same plugin, same fabric, same TP=3, so it should transfer — one line per node, reversible. Not applied there. |
| **The mesh plugin patches on the NVFP4 stack** | The idle second cable and the host bounce buffer are properties of the fabric and the plugin, not of the quantization, so both should transfer and are worth more there than the channel cap. Not applied. |
| **The collective's share of a decode step, re-profiled** | The 21.9 % prefill / ~24 % decode figures everything is reasoned against were profiled at 64 channels on one cable, before three separate collective changes. One profiling boot would replace three inferences with a measurement. It is the cheapest unspent measurement here. |
| **Prefix caching** | With a 3,328-token attention block our benchmark prompts never fill one, so the hit rate is 0 % throughout and the benchmark measures nothing about it. |
| **Long-context behaviour under load** | KV usage never exceeded 13 %. The pool is insurance, not something we have stress-tested. |
| **Pipeline parallelism** | The other way to run a fully quantized checkpoint on three nodes. Not evaluated; its interaction with speculative decoding is unknown. |
| **The RoPE convention question in the drafter** | A helper exists upstream whose comment describes the exact symptom we suspected, and our acceptance rate refutes it — so a working, measured configuration was left alone. The patch is written and never applied. |

---

## 4. Things that are true and inconvenient

- **Boot-to-boot variance on this stack is up to 16 % on C8.** Any single-pair comparison anyone
  publishes about this hardware, including ours, should be read with that in mind.
- **Binary hashes cannot certify a build here.** The compiled extension differs between independent
  builds of provably identical source, in the embedded device code; neither whole-file hashes nor the
  ELF build id are stable. Only behaviour certifies a build. See [02](02-image-build.md).
- **One node's build wall-clock does not match its own build's internal step timer**, by 90 seconds,
  and clock skew, toolchain version and compiler caching were each checked and ruled out. Unexplained
  `[measured-here]`. It does not affect any conclusion, which rest on content hashes and on tests.
- **The drafter is the difference between ~20 and ~57 tok/s at a single stream, and it is the most
  restrictively licensed component in the stack.** See [../LICENSES.md](../LICENSES.md).
- **Half of this fabric had never carried a packet, and nothing told us.** Every link was `ACTIVE`,
  every subnet was configured, every benchmark ran, and the second cable of each pair had transmitted
  zero bytes since the driver loaded. The ceiling we spent a day reasoning against was half the real
  one. If you take one operational lesson from this repository, take that one: read the byte counters,
  not the link state ([06](06-nccl-mesh.md) §6).

---

## 5. Where to help

[../CONTRIBUTING.md](../CONTRIBUTING.md) lists the items above that a reader with comparable hardware
could close, in rough order of usefulness.

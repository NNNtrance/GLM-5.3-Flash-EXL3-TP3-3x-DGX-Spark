# 11 — Open issues, retractions, and what we never ran

This stack is not finished. This page is the honest edge of it: what is unresolved, what we published
and then withdrew, and what we simply have not measured. Nothing here is hidden in a footnote
elsewhere.

---

## 1. Retracted

Seven things we wrote down as findings and later measured properly, plus two smaller ones, plus a
full audit of every claim of ours that a later measurement overturned (§1.9 — **24 of them**). Each
was published — in a report, an upstream issue, or both — before it was corrected. Two of them (§1.6
and §1.7) are the same number, corrected twice, in opposite directions.

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

### 1.7 "The pair of cables is worth 50 GB/s, so the collective is at 28 % of the fabric"

The correction in §1.6 was itself wrong, in the other direction, and lasted about eighteen hours
`[retracted]`. 50 GB/s is the **wire**: two 200 Gb/s ports per card. Each ConnectX-7 sits in a
**PCIe Gen5 x4** slot — `LnkSta: Speed 32GT/s, Width x4` — and carries about **15 GB/s** no matter
what its ports advertise, so a node's real fabric ceiling is **~30 GB/s** and the collective at
~20 GB/s is at about **70 %** of it, not 28 %.

Everything measured stays measured; only the roof moved. What changes is the priority: remaining
fabric headroom is **≤30 %**, worth **2–4 % of prefill**, not the 12–17 % an estimate of ours carried.
The old 13 GB/s ceiling was never "half a link" either — it was one card's PCIe limit at 87 % of it,
which is why the second cable, on the second card, took it to 20 and not to 40. Full account in
[06](06-nccl-mesh.md) §9. Two retractions of the same number in two days is its own lesson: we
computed a ceiling twice from a datasheet and never once from the machine.

### 1.8 Two smaller ones

- **"The gemm should zero a retired tile rather than return."** We shipped it; upstream's design is
  10.7 % cheaper per MoE layer at M=2048. Our patch is retired in writing
  (`patches/kernel/0002-RETIRED.md`) `[retracted]`.
- **"`--no-enable-flashinfer-autotune` is worth 34 s of provably empty work."** It is worth about
  **3.5 s**: the autotune was also doing JIT and kernel warm-up, which simply moved into graph
  capture `[retracted]`. See [08](08-fast-boot.md).

### 1.9 The full audit: every claim of ours that a measurement overturned

On 5 September the whole stack was re-read against its own raw data, and every published claim was
checked against the evidence behind it. Twenty-four did not survive. The nine above are the ones with
a story worth telling; this table is the complete list, so that nothing is quietly dropped and so the
shape of the mistakes is visible in one place `[retracted]`.

| # | What we claimed | What the measurement says | Where |
|---|---|---|---|
| 1 | Memory ruler is 273 GB/s | Achievable read is **225 GB/s**; every roofline published before was ~22 % optimistic | [10](10-results-and-roofline.md) §4.1 |
| 2 | "A 25 GB/s link" per neighbour | The cable is not the ceiling; the card's PCIe Gen5 x4 slot is, at ~15 GB/s | §1.6 |
| 3 | "A 50 GB/s pair of cables", so the collective is at 28 % | Same number wrong again, other direction; ~30 GB/s per node, collective at ~70 % | §1.7 |
| 4 | The 13 GB/s ceiling is because the device-pointer path is off | Turning it on added **nothing** at 16 MB (1,611 → 1,606 µs); the ceiling was PCIe | §1.6, [06](06-nccl-mesh.md) §9 |
| 5 | Expert-stationary scheduling is worth 14–27 % of MoE traffic | Doubling blocks per expert costs 1.11×, not 2×; the trellis stays L2-resident | §2.12 |
| 6 | The KV-zeroing gate is worth −1.2…1.4 % of prefill | 85.5 % of those bytes are Mamba/KDA state; safe remainder **−0.19 %** | §2.13 |
| 7 | NCCL picks `LL` at 16 MB and leaves half the link unused | Forcing LL there is **11× worse**; the tuner is not choosing it | §1.4 |
| 8 | No CUDA graphs on this stack / the `22 % 4` head-count obstacle | That was a different image. Here graphs capture and run (PIECEWISE **and** FULL); the 36/9 drafter sidecar removed the obstacle | [envs/env.tp3.example](../envs/env.tp3.example) |
| 9 | One upstream build is ~10 % slower end to end | Boot-to-boot variance is 15.9 % | §1.3 |
| 10 | Build `1699c89` costs 16 % at C1 | Single round. Over five rounds there is no loss (+1.2 %) | §1.3 |
| 11 | Disabling flashinfer autotune saves 34 s | ~3.5 s | §1.8 |
| 12 | Draft KV at fp8 is worth +0.3 % of the pool | That was the pre-256 geometry; today the same knob is **+4.7 %** | §2.18 |
| 13 | Fusing the hyper-connection pair saves 40 ms, −3.6 % of prefill | Measured ceiling −30 ms, −2.7 %; the kernel that exists delivers −1.0…1.1 % | §2.16 |
| 14 | The existing fused HC kernel is the better route at small M | At M=8 the fused route is 1.6 % **worse**; at M=2048 it is +32 % worse | [10](10-results-and-roofline.md) §5.5 |
| 15 | The MoE trellis GEMM is target number one | Against the corrected ruler it runs at 81–96 %. The two largest prefill items are **outside** the EXL3 kernel library | [10](10-results-and-roofline.md) §6 |
| 16 | The quick-arm harness runs five rounds and discards two | Its body ran three. Applied literally the rule left a median of one | [09](09-measurement-protocol.md) §1.1 |
| 17 | The chat template we serve comes from the base model | Its md5 matches **neither** checkpoint on disk. It has never been verified against a named source | [envs/env.tp3.example](../envs/env.tp3.example) |
| 18 | `NCCL_BUFFSIZE` is an open lever worth trying | It had already been measured and eliminated ("no difference"); `NCCL_P2P_NET_CHUNKSIZE` only affects point-to-point. We listed a closed item as open twelve hours later | [06](06-nccl-mesh.md) §12 |
| 19 | `NCCL_MAX_NCHANNELS=8` had already been tried and eliminated | That arm set 8 channels **together with** `NCCL_PROTO=LL` and was never written up. Tried cleanly it is +13 % at C8 | [06](06-nccl-mesh.md) §8 |
| 20 | The extra masking pass is under 1 % of the MoE layer | 2.9 % at M=8 rising to 15.8 % at M=2048 | §1.2 |
| 21 | The missing `n_rows` also costs the non-expert-parallel path | With no expert map the tail is `-1` everywhere; the author's reading was right | §1.1 |
| 22 | `--language-model-only` stops the vision tower being built | It only stops it being *run*. 1.05 GiB wasted at TP=2, and only visible at TP=3 when `divide(16, 3)` asserted | [03](03-tp3-padding-and-sidecars.md) |
| 23 | The TP=2 KV collapse came from the draft's page layout | The dominant cause was memory scarcity at two nodes; the page layout was second-order | [07](07-kv-and-draft-page.md) §4 |
| 24 | Overlapping the collective with compute is worth −10…13 % of prefill and −6…10 % of decode | That estimate counted the hideable collective and not the second MoE weight stream the split pays for. Corrected: prefill −6.3…+8.0 %, decode +6…+38 % (worse) | §2.17 |

Read the shape rather than the rows. **Six of the twenty-four are a ruler we quoted instead of
measured** (1, 2, 3, 4, 15, and the roofline percentages that followed). **Four are a single pair of
sweeps treated as a result** (9, 10, 16, 19). **Three are an arithmetic model that a bench refuted**
(5, 6, 24). **Two are our own tooling disagreeing with our own documentation** (16, 18). Only one (17)
is a provenance claim, and it is still open.

---

## 2. Open, with a known next step

### 2.1 The unquantized half of the model

**44.8 % of a C1 decode step** and **16.2 % of a prefill chunk** is BF16 dense GEMM, because the
checkpoint quantizes routed experts only ([10](10-results-and-roofline.md) §5.2–§5.3). Nothing in the
EXL3 kernel library touches it, and at TP=3 each rank's share is a third rather than a half, so those
kernels get *less* efficient as ranks are added. **This is the largest remaining structural item on
the stack**, and the step-time breakdown made it larger rather than smaller: at decode it is now
measured as the single biggest class, ahead of the trellis GEMM.

There is a second, smaller half to it that is not about the checkpoint at all: those kernels are
Ampere-class `cutlass_80_*` on an sm_121 part, and at the engine's own shapes they reach **79 %** of
what `torch.matmul` gets on the same device (63.6 against 80.4 TFLOP/s). Closing that gap is
somebody's cuBLAS/vLLM work and is worth ~3.1 % of prefill `[measured-here]`.

The obvious answer — a checkpoint that also quantizes attention — is the one that cannot run at TP=3
today, because with attention quantized there is no unquantized dimension left to split three ways
([01](01-model-and-license.md) §3.1). Next step: nothing cheap. It is either a differently scoped
quantization, or pipeline parallelism, which we have not evaluated at all `[not tested]`.

### 2.2 The fabric: what is actually left, now that the ceiling is right

Mostly **closed, and smaller than it looked**. Three things settled it on 5 September:

- The ceiling is the cards' **PCIe Gen5 x4** slots, ~15 GB/s each and ~30 GB/s per node, not the
  50 GB/s of wire the ports advertise (§1.7, [06](06-nccl-mesh.md) §9). At ~20 GB/s of bus bandwidth
  the collective is at about **70 %** of what the machine can carry, so the remaining headroom is
  **≤30 %** — worth **2–4 % of prefill**, not the 12–17 % we had priced it at `[estimate]`.
- The receiver-advertised FIFO with `RDMA_WRITE`, which this section used to call "the deeper fix,
  upstream's to make", **was written and measured** (`patches/kernel/0007`). It removes RNR retries
  and out-of-buffer events to **exactly zero** at every message size — and changes throughput by
  nothing, at any FIFO depth, with or without the flush. The engine arm is inside boot spread.
  Not adopted, kept as an option, deliberately not offered upstream ([06](06-nccl-mesh.md) §10)
  `[measured-here]`.
- The collective's share is no longer an inference: **16.5 % of a prefill chunk** (14–17 % allowing
  for the reconciliation residual) and **10–15 % of a C1 decode step**, down from 20.1 % and 17.5 %
  ([10](10-results-and-roofline.md) §5) `[measured-here]`.

What is still genuinely open here is small and specific:

1. **The engine-side RNR counters have not been re-read since the patches** `[not tested]`. The
   42,000-events-per-node reading was taken on the single-cable, host-bounce-buffer configuration.
   Model-free the retries fell from ~15 per operation to ~3 with `NCCL_PTR_CUDA` and to 0 on the
   one-sided transport — which bought nothing, so this item is now mostly of forensic interest.
2. **`NCCL_MAX_NCHANNELS=12` over two cables** `[not tested]`. Indistinguishable from 8 on one cable,
   never taken to the engine, and 16 is 2.5× worse on the decode-sized message over two
   ([06](06-nccl-mesh.md) §8.1), so the arithmetic changed and the equivalence cannot be carried
   forward.
3. **`patches/kernel/0004` has never been isolated on the engine** `[not tested]`. It rides in the
   production build with 0005 and 0006 at `NCCL_MESH_MIN_RNR_TIMER=1`.
4. **Nobody has tried to make one collective use both cards at once.** Channels alternate between the
   two NICs; whether a single large all-reduce can saturate both PCIe paths at the same time is
   unmeasured, and it is where the remaining ≤30 % would have to come from `[not tested]`.
5. **`NCCL_ALGO=Ring,Tree` has never been run on this mesh** `[not tested]`, and it is the cheapest
   untried thing in this repository. Our launcher **forces** `NCCL_ALGO=Ring`; at `world_size = 3` a
   ring all-reduce is `2(w−1) = 4` sequential steps against a tree's `~2·log₂3 ≈ 3.2`, and decode is
   latency-bound on a fixed 102 collectives per step ([10](10-results-and-roofline.md) §5.3), so step
   count converts straight into time. Expected **−1…3 % of a decode step**, ~0 at prefill where Ring
   is already the right choice at 16.78 MB. Handing the list to the tuner rather than pinning Tree is
   the arm to run. Cost: **zero** — it is a model-free sweep with the engine down; only a win goes to
   an engine arm.

`NCCL_BUFFSIZE` and `NCCL_P2P_NET_CHUNKSIZE` are **not** on this list, and were briefly put back on
it by mistake: the first was measured and made no difference, the second only affects point-to-point
transfers ([06](06-nccl-mesh.md) §12, §1.9 row 18).

**Retracted here, twice** `[retracted]`: this section once said the disabled GPUDirect path "holds
the ceiling at ~13 GB/s against a 25 GB/s link", and then said the true ceiling was a 50 GB/s pair of
cables. Both are wrong; see §1.6 and §1.7.

### 2.3 The pool is sized by the smallest rank — and the imbalance is bigger than we said

The head node has 35.40 GiB available for KV and the binding rank has 32.85 GiB, so **2.55 GiB on the
head node is simply unused** `[measured-here]`. That was the production-6 reading. It is the wrong
end of the telescope: the pool is set by the **worst** rank, so every GiB a worker spends that the
head does not is a GiB of pool nobody gets.

Reading the same ledger by class rather than by total makes it much larger. The weights are
**identical on all three ranks** — the engine logs `Model loading took 54.86 GiB` three times — and
yet weights-plus-non-torch reads 56.36 / 64.34 / 64.58 GiB, which puts **non-torch memory at 1.50 GiB
on rank 0 and 9.48–9.72 GiB on the workers**. About **8.2 GiB per worker is stranded**, and the pool
is sized against it. If it were equalised the pool would grow by roughly **8–26 %**, which is larger
than every kernel item left on this page `[measured-here]`.

**Under diagnosis, and honestly labelled: nobody knows yet what that 9.5 GiB is.** Candidates nobody
has separated: NCCL and mesh-plugin buffers (the workers hold more peer state than the head),
the 32 GB shared-memory segment, `--headless` worker startup, and the fast-load dumper. Two caveats
before anyone prices it: the 8.2 GiB figure was read from a **dump boot**, whose ledger is not
production's ([09](09-measurement-protocol.md) §11), and the production-6 era numbers were milder
(5.2 / 7.7 / 7.9 GiB), so part of this may be the dump. The next step is cheap and does not need a
new arm: read the per-class ledger on the next ordinary boot, and if it holds, one instrumented boot
that names the allocations `[not tested]`.

### 2.4 The memory ladder above 0.80

0.85 was measured and rejected on the free-memory rule; 0.88 was never attempted. Two things have
changed since that verdict was recorded, and both argue for re-running it rather than carrying it
forward `[not tested]`:

- **The rejection predates the fast-load work.** That work removed a large page-cache spike during
  loading and added `MADV_DONTNEED` plus `malloc_trim` after it ([08](08-fast-boot.md) §5). The
  0.85 boot hit 1.9 GiB free with 1.6 GB of swap; the same production configuration today sits at
  11–12 GiB free with **zero** swap and 3.0–3.5 GiB of page cache. The rung at **0.82–0.83** was
  never tried at all.
- **`--kv-cache-memory` has never been used.** It sizes the pool in bytes rather than as a fraction
  of the device, which matters here because `gpu-memory-utilization` budgets a share of the **total**
  while a worker starts with 113 GiB actually free — and vLLM's own boot log says so, printing the
  byte figure it believes would fully utilise the device against the far smaller one we take.

Order matters: **ladder first, pin last.** A byte pin removes exactly the headroom the 4 GiB
free-memory rule exists to protect, so it is the last rung, not a shortcut past the others. See
[07](07-kv-and-draft-page.md) §6.

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
the same as "free", and it has never been taken to the engine either way `[not tested]`. On the
one-sided transport the same knob is again inside noise ([06](06-nccl-mesh.md) §10.2), and it is
again kept on, for the same reason.

### 2.12 Expert-stationary MoE scheduling — closed, it buys nothing

**Closed.** The large-M trellis GEMM runs 112 blocks over 96 local experts at M=2048 and 121.5 at
M=512, so *if* each block re-read its expert's weights, 17–27 % of the traffic would be avoidable and
an expert-stationary schedule would be worth ~3.7 % of prefill. The kernel author's own bench
(`bench_moe_expert_reread.py` at `9b17ea9`), run unmodified on this hardware three times, says the
trellis **stays resident**: doubling the blocks per expert costs 1.11×, not 2×; quadrupling costs
1.5×, not 4× `[measured-here]`. `moe_align_block_size` keeps an expert's blocks adjacent and 8.4 MB
per expert fits a 24 MiB L2. The same bench gives 1.16× on the author's 188-SM card, so it is
structural and not a property of 48 SMs. Full tables in [10](10-results-and-roofline.md) §5.4.

The traffic model that generated the 14–27 % estimate was simply wrong, and only a bench could say
so — the trace cannot see whether a read was served by L2.

### 2.13 The KV-zeroing gate — closed, the gain is not available on this model

**Closed.** `_zero_kv_blocks_kernel` writes 2.4–2.9 GB per prefill chunk where the new tokens' real
KV is ~3.4 MB, and it does it at 100 % of the memset roofline, so the only lever was to not run it.
vLLM runs it when the cache has Mamba layers **or** mixed precision; the zeroer skips Mamba layers,
which made "so the only live reason is mixed precision — move the draft to fp8 and the gate opens"
look correct. It is not: in this model's hybrid layout one tensor is **co-owned** by an MLA layer and
one Mamba layer per group, so a block moving from the Mamba group to the attention group carries
1.7 MB of raw SSM state. Measured per block, **85.5 %** of what is being zeroed is that co-owned
region `[measured-here]`.

A fail-closed gate was written anyway (`HAREM_ZERO_ATTENTION_KV=0`, off by default, three conditions
proved from the engine's own config, `raise` rather than warn) precisely so the conclusion is checked
by the machine rather than believed: on this model it **refuses to boot**, by design. The safe
remainder — indexer plus draft page, if the cache were uniform — is 0.19 % of prefill, and no partial
mode was written for it. The ceiling is recorded here so nobody prices it a third time.
[10](10-results-and-roofline.md) §5.6.

### 2.14 A cooperative (single-kernel, `grid.sync`) MoE stage — closed for production

**Closed.** On a 48-SM part the cooperative-launch grid tops out at 288 blocks — about 3.9× fewer
blocks have to reach the barrier than on the 188-SM card the idea came from — so the intuition that
"the sign flips on a small part" is correct: outside a CUDA graph a `grid.sync()` barrier beats three
separate launches at every size we measured, by up to 1.37 µs per phase boundary (33 % at 262,144
elements) `[measured-here]`, three independent runs agreeing.

**Inside a CUDA graph the sign flips back**, and production runs inside one. The graph erases almost
all of the separate-launch arm's launch cost (0.2576 → 0.1049 ms at the small size) and has little to
erase on the cooperative arm, leaving the barrier itself as a **+0.2 to +0.3 µs per boundary loss**
at small and medium sizes; at 4M elements both arms are bandwidth-bound and equal. So the detail that
decides it is not the SM count, it is the graph. Tool: [`bench/gridsync.cu`](../bench/gridsync.cu).

### 2.15 The fused MoE input transform — closed upstream

**Closed.** Our A/B found the fusion worth +1–4 % end to end at small batch and a regression at large
batch on this hardware; upstream first narrowed the gate (`61a17bc`), then removed the feature
entirely (`76598b2`) and took the same win another way. Nothing to carry. Listed here because it was
an open arm for two days and because the outcome — upstream deleting 232 kernel lines rather than
tuning a threshold — is the useful part of the story ([02](02-image-build.md),
[10](10-results-and-roofline.md) §2).

### 2.16 The hyper-connection fusion kernel — written and measured; not adopted on its own

`hc_mult = 4` carries the residual stream in four copies, and a fused post+pre block touches
`residual_cur` three times per call: the first kernel writes it, the second reads it, the third reads
it. The second kernel's **entire** traffic is that one re-read. Fusing kernels one and two removes
30.4 % of that pair's bytes, which is a ceiling of **−2.5 to −2.7 % of prefill** `[measured-here]`.

**The kernel now exists.** A Triton kernel that grids over token tiles and reduces the post mapping
against the projection while the row is still in registers was written, swept over 33 configurations
and measured model-free in two independent runs: **−14.9 to −15.5 %** on the fused pair, **−9.0 to
−9.9 %** on the three-kernel route, **−1.0 to −1.1 % of the prefill wall** at M=2048. Correctness is
bit-identical on `residual_cur` at every M except one 7-element, 1-ulp difference at M=64;
`layer_input` differs by at most one bf16 ulp on 5.1 % of elements. Full tables:
[10](10-results-and-roofline.md) §5.5.1.

**It reached about 40 % of its own ceiling and it is not going in alone.** The loss is entirely
bytes-per-second — the fused kernel runs at 187.7 GB/s against the 229.5 the route it replaces gets —
and a full configuration sweep could not improve it, so it is not tiling or occupancy. Set against
adoption's price (Triton JIT inside the serving process, a warm-up before graph capture, and a
configuration surface that is a cliff rather than a slope: the winner reads 187.8 GB/s and its
neighbours 79.4 and 44.5), −1 % does not earn its own boot. **It rides the next image bundle with
`had_in` (§2.19), where one boot measures both.**

Two threads left open:

- **The one untried arm that would change the verdict** `[not tested]`: the `tl.dot` operand path
  transposes the projection inside the kernel through shared memory. Pre-transposing it once on the
  host would remove that; if it lifts the kernel to the band the first kernel already achieves, the
  result becomes −2.2 % of prefill and this stops being a bundle rider. Half an hour of work,
  not measured, not claimed.
- **The size threshold is measured now, and it was wrong.** Fusing *loses* below about M=1024 —
  +37.7 % at M=512 — because `residual_cur` at that size is 16.8 MB and fits the 24 MiB L2, so the
  re-read it deletes was never going to DRAM. The module shipped with the threshold at 256.

Two smaller, cheaper pieces of the same class are measured and unclaimed: passing two keyword
arguments the call site currently leaves at their defaults is −4.9 % on the first kernel, and one
constant in the third is −3.5 % — 0.4 % of prefill between them. And a one-pass kernel that reads the
residual once and writes both outputs once would be −5.3 % of prefill, but needs 32 KB of shared
memory per token and would collapse occupancy on 48 SMs; not measured, not recommended, recorded so
the ceiling is known `[estimate]`.

### 2.17 Overlapping the all-reduce with compute — dual-batch overlap is closed; one variant survives

The collective is **16.5 % of a prefill chunk** and GPU occupancy in that window is **99.3 %**, so the
all-reduce is **serialised** against the compute rather than hidden behind it. Making the fabric
faster can win at most 2–4 % of prefill (§2.2); making the collective *overlap* could reach for most
of the 16.5 %. That is why this item carried the largest number on the page. It is now much smaller,
and the reason is arithmetic rather than engineering.

**First, what is not available.** Every overlap mechanism vLLM ships behind a flag is dead on this
configuration, and each for a checkable reason `[measured-here]`, read out of the image with the
engine down: async tensor parallelism, the sequence-parallelism pass and the FlashInfer
all-reduce+RMSNorm fusion all require `torch.compile`, and this model family never enters it — it is
on a fixed list that forces `CompilationMode.NONE`. The FlashInfer path additionally supports world
sizes 2/4/8/16, custom all-reduce 2/4/6/8/16, NCCL symmetric memory a minimum of 4, and torch
symmetric memory neither this capability nor this world size — **`world_size = 3` is excluded by all
four**. DeepEP is installed but never engages, because with `data_parallel_size = 1` the all-to-all
path is not constructed at all. Nothing here is a tuning opportunity; they are all structural.

**Second, dual-batch overlap (DBO), which was the real candidate — closed.** The mechanism suits an
all-reduce: the yield primitives are stream-based and collective-agnostic, vLLM's TP all-reduce lands
on `current_stream()`, and the ping-pong between micro-batches is deterministic so every rank issues
in the same order. The patch is also **smaller than we estimated** — about 95–160 lines across five
files, and it does **not** touch the model file, because one bottleneck (`GroupCoordinator.all_reduce`)
covers all 102 collectives. Our earlier "medium-to-large vLLM patch, mHC state at risk" reading was
wrong on both counts: the mHC triple is thread-local dataflow, not a shared buffer `[retracted]`.

It fails on arithmetic instead. **Splitting the batch pays the MoE expert weight stream twice.** At
M ≥ ~128 every local expert is already touched, so two half-batches stream the same 96 experts per
rank twice: **+73 to +232 ms per chunk**, against **−135 ms** of collective that could be hidden at
best. The break-even is in the middle of that uncertainty band.

| arm (per 2,032-token chunk, f = 0.8 hidden) | ms/token | vs today |
|---|---|---|
| today, MNBT 2048 | 0.5458 | — |
| MNBT 4096, no DBO | 0.4584 | **−16.0 %** |
| MNBT 4096 + DBO | 0.4702 | −13.9 % |
| MNBT 2048 + DBO | 0.5113…0.5896 | **−6.3 %…+8.0 %** |

Decode is worse and not marginal: C1 **+38 %**, C8 +6 %, because the collectives are latency-bound
(102 → 204 of them, and halving the message does not halve the time), the drafter cannot be
micro-batched at all, and `use_ubatching` **disables the breakable CUDA graph for both the target
model and the drafter**. There is also a correctness hazard that would have to be solved first: 34 of
45 layers are KDA/Mamba, and the batch splitter does not align to request boundaries, so a fresh
prompt split in two has its second half start its recurrent state from zero and write it into the
same state slot — fluent, wrong, and silent. `VLLM_DBO_COMM_SMS` is a **no-op** here (the SM-reservation
setter is empty on this communicator and `deep_gemm` is not installed), so NCCL and the trellis GEMM
would compete for SMs unmanaged. **Verdict: do not build DBO.** Increasing the batched-token budget
buys more than overlapping does, for the same KV price and no code — and the budget is a decision we
have already made the other way (§2.5, [07](07-kv-and-draft-page.md) §5).

**Third, the one variant still open** `[not tested]`: **attention-scoped micro-batching**. Split the
chunk only across the attention block and rejoin before the MoE stage. It hides the 45 attention
`o_proj` all-reduces (~80 ms, 44 % of the collective) and pays only for attention weights streaming
twice (~15–25 ms, because the experts dominate the model). Expected **−3 to −6 % of prefill**, ~0 at
decode, ~120–160 lines in the model file and the runner. It carries the **same KDA state hazard**, so
it cannot be written before that is solved.

Before any of this, one model-free probe is worth more than the design work and costs nothing with
the engine down: **does an all-reduce on a second stream actually overlap a GEMM on this hardware**,
or does the plugin's proxy thread or SM contention serialise them? A probe is written and has not
been run; the gate is `overlap_frac ≥ 0.60` and `gemm_stretch ≤ 1.15` at 16.78 MB. If it fails, every
overlap variant above dies with it `[not tested]`.

Also closed while looking: model-level **sequence parallelism** would be a one-line gate change and is
a bad idea here — the bytes are identical (2 × 1.333 S per layer either way), the collective *count*
doubles from 90 to 180, and decode is latency-bound, so it is **+10…15 % worse at decode** for a
~10 % prefill compute saving. And **pipeline parallelism** removes ~98 % of the collective bytes at
the cost of ~3× single-stream decode latency. Both are written down because "just turn on SP" is the
obvious next reflex and it is wrong on this workload.

### 2.18 Draft KV at fp8 — validated on a dump boot; the pool number needs a load boot

The DFlash draft's KV cache is bf16 while the main cache is fp8, which is the *other* half of the
mixed-precision condition in §2.13 and also costs KV pool. A prelude patch overrides
`SpeculativeConfig` (`HAREM_DRAFT_KV_DTYPE=fp8`) without touching the launcher and is a no-op when the
knob is unset. The pool should grow **+4.7 %** at today's 256-token draft page; the +0.3 % in our
earlier notes belongs to the pre-256 geometry `[retracted]`.

**It has now booted, and both risks are retired** `[measured-here]`. The DFlash sliding-window backend
accepts an fp8 cache. Draft acceptance is **60.1–64.0 %** across all concurrency levels and rounds
(one C1 round at 57.3 %) against production's 61–65 % and a gate that required the 60–65 band — so the
drafter is not merely unbroken, it is not measurably weakened. Gates 10/10 · 12/12 cold and warm;
speed inside the bands. The mechanism is visible in the log rather than inferred: draft page
393,216 → **196,608 bytes**, per-block 21,917,440 → **20,934,400**, divisor unchanged at 363.
Tables in [10](10-results-and-roofline.md) §2.1 and [07](07-kv-and-draft-page.md) §7.

**What is still missing is the only number that would promote it.** The arm added three prelude
patches, which invalidates the fast-load sidecar and forces a dump boot
([09](09-measurement-protocol.md) §11), and a dump boot's KV pool reads low — this one read 4,382,920,
*below* production, and it means nothing. Expected on a load boot: about **4.66M**. Until that boot,
production stays at the bf16 draft cache `[not tested]`.

### 2.19 The next image bundle: `exl3_moe_had_in`, and what rides with it

`had_in` was the only sub-roofline kernel left in the MoE stage at 37–57 % of the ruler. Upstream took
it in `a47da6e` by removing a 64-bit division in favour of deriving the index from the grid:
**−10 to −18 %** on that kernel, roofline 57 % → 63 % `[reported]`. On this stack that is ~0.2–0.3 %
of prefill wall, which does not justify an image rebuild on its own. Until then the production image
does not have it, and [10](10-results-and-roofline.md) §6 row 7 still quotes the old number
`[not tested]`.

It is no longer alone, and that is the point of the item. **The bundle now holds two things**:
`had_in` at −0.2…0.3 % and the hyper-connection fusion kernel at −1.0…1.1 % (§2.16). Together they
are worth about **−1.3 % of prefill** for one build and one arm — still small, and now large enough
that the arm is worth scheduling rather than deferring. If the pre-transpose arm in §2.16 lands first,
the bundle is worth −2.4 % and the calculation is no longer marginal. The rule the bundle exists to
enforce: **do not spend a boot on a sub-1 % change**; accumulate them and spend one boot on the pile.

### 2.20 This stack has no autostart, and the sibling's unit will win a reboot

There is no systemd unit for this engine. The container runs with `--restart no` — deliberately, since
a half-started rank quietly retrying is exactly the "fluent and wrong" failure class this stack is
built to refuse — and nothing supervises it, so an unattended reboot leaves the cluster down. That is
a known, accepted gap.

**The hazard is the other half of it.** The NVFP4 sibling stack's own unit, `harem-motor.service`, is
`enabled` on all three nodes. A reboot today therefore does not leave the cluster down — it brings up
the **other engine**, on the same GPUs and the same memory, which is worse than nothing if you were
expecting this one `[measured-here]`. The two must never both be enabled: whichever unit is installed
needs `Conflicts=` against the other, and installing this one has to be paired with disabling that
one.

A unit template is in [`systemd/`](../systemd/) with its three unfinished pieces named — the preflight
script it calls does not exist, systemd will not honour the worker-2 → worker-1 → head start order on
its own, and its `ExecStop` names the wrong container. It is **not installed anywhere** and is not
recommended for installation as it stands.

The cheaper half of the same problem is a watchdog rather than a unit: a 60-second `docker ps` plus
`/health` poll that records `docker logs --tail 40` when the container exits. One outage during this
work ran an hour purely because nothing was watching `[measured-here]`. Not written `[not tested]`.

### 2.21 The fast-load identity gate is stricter than it needs to be

The sidecar's identity hashes **every** `patch-*.py` in the patch directory and the full text of the
prelude script. Three patches that do not touch a single weight byte — a KV-page knob, a fail-loud
import guard and a draft-dtype override — refused a boot on that basis and cost an hour
`[measured-here]`. The gate is doing what it was built to do; it is simply drawing the line in the
wrong place, and it makes every experiment that adds a patch cost a dump boot
([09](09-measurement-protocol.md) §11).

The narrower gate, not written `[not tested]`: an explicit allow-list of the patches that can affect a
weight (the padding patch, the expert-parallel patch and its overlay, the drafter patch, the
expert-filter patch) plus the environment keys that decide the layout; a patch not on that list does
not change the identity, and adding a name to the list is then a deliberate act. The prelude would
contribute its **ordered list of patch invocations** rather than its full text, so a comment change
stops invalidating 56 GiB per node while a reordering or a skipped patch still fails. And the launcher
should snapshot the patch directory to a private path before mounting it, which turns "do not touch
the directory during a boot" from a rule into a property.

What must not change: the 32-tensor sha256 sample still runs on every boot. The identity gate is a
cheap early warning, not the proof — which is why it can be narrowed and must not be removed.

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
| **A torch-profiler run on the production configuration** | The engine was launched without `--profiler-config`, `/start_profile` returns 404 and this `nsys` cannot attach, so the step breakdown in [10](10-results-and-roofline.md) §5 is a reconciliation with a 2.8 % residual rather than a direct profile. It would settle the NCCL band (14–17 % of prefill) and the C8 decode split. Cost: one boot, and a launcher flag set before you need it. |
| **The MLA prefill kernel's efficiency** | 8.2 % of a prefill chunk, and the trace does not carry the selected-key count, so there is no denominator. Needs its own model-free measurement before anyone calls it a target. |
| **Draft KV at fp8 on a load boot** | Booted and validated on a dump boot; the pool number still needs an ordinary boot — [§2.18](#218-draft-kv-at-fp8--validated-on-a-dump-boot-the-pool-number-needs-a-load-boot). |
| **`NCCL_ALGO=Ring,Tree`** | Never run on this mesh, and it is free — the sweep is model-free with the engine down. Our launcher forces `Ring`. §2.2 item 5. |
| **Whether a second-stream all-reduce overlaps a GEMM at all on this part** | The probe is written and has not been run. It gates every overlap variant in §2.17, and it costs one engine-down bench. |
| **A long unattended run** | The longest continuous uptime on record is about an hour between arms. Leaks, KV fragmentation, fabric drift and acceptance drift over 6–12 hours of mixed load are all unmeasured. |
| **`--max-num-seqs` above 8** | Chosen to match the TP=2 arrangement and never A/B'd, and C8 sits exactly on the cap. It does not enter the KV divisor, so the cost would be TTFT, not pool. |
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
- **We computed the same ceiling wrong twice, from two different datasheets, in two days.** First
  "a 25 GB/s link", then "a 50 GB/s pair of cables"; the answer was a PCIe slot both times (§1.6,
  §1.7). On the same day the memory roofline moved from a vendor 273 GB/s to a measured 225 GB/s and
  made every efficiency percentage in this repository ~22 % optimistic
  ([10](10-results-and-roofline.md) §4.1). Three of our rulers were brochures. **Measure the ruler,
  in the same process, and quote it beside the number.**
- **The ruler itself is not stable.** Three read-bandwidth measurements on the same idle machine the
  same morning gave 225.2, 239.6 and 240.9 GB/s — 6.5 % apart `[measured-here]`. Any efficiency
  figure in this repository that would change a decision at ±6 % is given as a band for that reason.
- **The best-designed change of the day did exactly what it was designed to do and was worth
  nothing.** The one-sided RDMA_WRITE transport takes RNR retries to zero, which is what it was
  built for, and moves throughput by zero, because the constraint was somewhere else entirely
  ([06](06-nccl-mesh.md) §10). A mechanism working is not a result.
- **A compile check answers "does it build", never "does it run".** A GPU-free ahead-of-time compile
  of 18 kernel configurations reported all 18 building and all 18 inside the shared-memory limit. At
  real launch **6 of the 18 failed with `OutOfResources`** — the ahead-of-time path reported 36,864
  bytes of shared memory where the launch needed 106,496 `[measured-here]`. Resource allocation only
  becomes visible when something is actually launched, and the static test had already been written
  up as a pass.
- **Our own harness disagreed with its own documentation for a day**, claiming five rounds with two
  discarded while running three ([09](09-measurement-protocol.md) §1.1). Nothing was published from
  the wrong reading, but a tool that describes itself incorrectly is the same failure class as a
  ruler that reads high: it will eventually be believed instead of read.
- **Two of the twenty-four retractions in §1.9 are us re-opening something we had already closed and
  measured.** `NCCL_BUFFSIZE` was listed as an open lever twelve hours after it had been eliminated,
  because the elimination lived in one report and the candidate list was written from another. The
  fix is not diligence, it is that a closed item has to be closed **in the place where the next
  person will look**, which is why §2 keeps closed items rather than deleting them.
- **The two changes designed as one turned out to be unrelated.** The KV-zeroing gate and the fp8
  draft cache were built together, on the reading that uniform precision would let the zeroing stop.
  The zeroing is bound by Mamba slot sharing, not precision, so one of the two is closed at zero and
  the other is a 4.7 % pool gain that never needed it (§2.13, §2.18).

---

## 5. Where to help

[../CONTRIBUTING.md](../CONTRIBUTING.md) lists the items above that a reader with comparable hardware
could close, in rough order of usefulness.

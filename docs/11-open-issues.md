# 11 — Open issues, retractions, and what we never ran

This stack is not finished. This page is the honest edge of it: what is unresolved, what we published
and then withdrew, and what we simply have not measured. Nothing here is hidden in a footnote
elsewhere.

---

## 1. Retracted

Seven things we wrote down as findings and later measured properly, plus two smaller ones. Each was
published — in a report, an upstream issue, or both — before it was corrected. Two of them (§1.6 and
§1.7) are the same number, corrected twice, in opposite directions.

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

**Retracted here, twice** `[retracted]`: this section once said the disabled GPUDirect path "holds
the ceiling at ~13 GB/s against a 25 GB/s link", and then said the true ceiling was a 50 GB/s pair of
cables. Both are wrong; see §1.6 and §1.7.

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

### 2.16 The hyper-connection fusion kernel — open, and the largest kernel-side item left

`hc_mult = 4` carries the residual stream in four copies, and a fused post+pre block touches
`residual_cur` three times per call: the first kernel writes it, the second reads it, the third reads
it. The second kernel's **entire** traffic is that one re-read. Fusing kernels one and two saves
30.5 % of that pair's bytes — **−28 to −30 ms per prefill chunk, −2.5 to −2.7 % of prefill**
`[measured-here]`.

The kernel that would do it does not exist yet. The existing `mhc_fused_tilelang` is the wrong shape:
it grids per token per n-tile and re-reads the residual in every CTA, and forced at large M it is
**+32 % worse** at M=2048 — worse at every M we tried, including the M≤16 range production actually
selects it in. What is needed is a large-M kernel that tiles over `block_m` so the 1.57 MB projection
is shared across tokens, finishes the n dimension in one CTA, and reduces `residual_cur` while it is
being written. The skeleton exists (`hc_prenorm_gemm_block_m_tilelang`); adding the post mapping to
its A-operand load is the work. **This is vLLM/TileLang work, not EXL3 work** `[not tested]`.

Two smaller, cheaper pieces of the same class are measured and unclaimed: passing two keyword
arguments the call site currently leaves at their defaults is −4.9 % on the first kernel, and one
constant in the third is −3.5 % — 0.4 % of prefill between them. And a one-pass kernel that reads the
residual once and writes both outputs once would be −5.3 % of prefill, but needs 32 KB of shared
memory per token and would collapse occupancy on 48 SMs; not measured, not recommended, recorded so
the ceiling is known `[estimate]`.

### 2.17 Overlapping the all-reduce with compute — the larger fabric lever, untouched

The collective is **16.5 % of a prefill chunk** and GPU occupancy in that window is **99.3 %**, which
means the all-reduce is essentially **serialised** against the compute rather than hidden behind it.
Making the fabric faster can win at most 2–4 % of prefill (§2.2); making the collective *overlap*
could reach for most of the 16.5 %, and nothing in this stack has tried.

It is not a plugin change. It is a scheduling question in the engine — chunk the hidden state so the
all-reduce for one slice runs while the next slice computes — and it interacts with the MoE stage's
per-token cost curve and with the batched-token budget. We have not designed it, measured it, or
established that this vLLM's collective path can be driven that way `[not tested]`. It is written
down here because after a day of fabric work it is the item with the largest number attached to it.

### 2.18 Draft KV at fp8 — written, gated, not yet booted

The DFlash draft's KV cache is bf16 while the main cache is fp8, which is the *other* half of the
mixed-precision condition in §2.13 and also costs KV pool. A prelude patch exists
(`HAREM_DRAFT_KV_DTYPE=fp8`, overriding `SpeculativeConfig` without touching the launcher, no-op when
the knob is unset) and the arithmetic says the pool grows **+4.7 %** at today's 256-token draft page
`[estimate]` — the +0.3 % figure in our earlier notes belongs to the pre-256 geometry and is
`[retracted]`.

It has **not been booted** `[not tested]`. Two things it has to survive: the DFlash sliding-window
backend has to accept an fp8 KV at all (if it does not, the boot fails loudly rather than corrupting
anything), and the draft now attends over fp8, which can weaken its proposals — so **draft acceptance
must stay in the 60–65 % band** or the arm is rejected. Gates cold and warm catch a broken drafter;
only the acceptance rate catches a merely weakened one.

### 2.19 `exl3_moe_had_in`, taken upstream and not yet built

`had_in` was the only sub-roofline kernel left in the MoE stage at 37–57 % of the ruler. Upstream took
it in `a47da6e` by removing a 64-bit division in favour of deriving the index from the grid:
**−10 to −18 %** on that kernel, roofline 57 % → 63 % `[reported]`. On this stack that is ~0.2–0.3 %
of prefill wall, which does not justify an image rebuild on its own; it goes into the next build
bundle together with whatever else has accumulated. Until then the production image does not have it,
and [10](10-results-and-roofline.md) §6 row 7 still quotes the old number `[not tested]`.

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
| **Draft KV at fp8** | Written and gated, never booted — [§2.18](#218-draft-kv-at-fp8--written-gated-not-yet-booted). |
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

---

## 5. Where to help

[../CONTRIBUTING.md](../CONTRIBUTING.md) lists the items above that a reader with comparable hardware
could close, in rough order of usefulness.

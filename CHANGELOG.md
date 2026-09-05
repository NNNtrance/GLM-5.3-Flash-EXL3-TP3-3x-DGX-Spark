# Changelog

Dated entries for the stack this repository documents. This is a working record, not a release
history — the repository is a **private draft** and is overwritten in place as the work continues.

Speed figures are aggregate output tok/s on `scripts/hizset-v2.jsonl` at `reasoning_effort: low`,
temperature 0. Entries up to and including *production configuration 4* are medians of sweep rounds
3–5 with rounds 1–2 discarded; from *production configuration 5* onwards they are medians of three
rounds, which is what the persisted MLA tuner cache bought — see
[docs/09](docs/09-measurement-protocol.md) §1 and [docs/12](docs/12-tuner-cache.md).

---

## 2026-09-05 — a kernel that half-worked, an overlap that does not pay, and a stricter definition of "equal"

Production configuration 6 is **unchanged**, for the second entry running. One change is validated and
waiting on a boot; three designs were costed and rejected; and the measurement protocol got stricter
about what counts as a difference.

- **Draft KV at fp8: validated, not yet promoted** `[measured-here]`. Moving the DFlash2 drafter's own
  cache from bf16 to fp8 shrinks its page 393,216 → **196,608 bytes** and the per-block cost
  21,917,440 → **20,934,400**, with the blocks-per-request divisor unchanged at 363 — worth about
  **+4.7 %** of pool. The arm booted, which answered the only question a boot could: the DFlash
  sliding-window backend accepts an fp8 cache. Draft acceptance **60.1–64.0 %** against production's
  61–65 % and a gate that demanded the 60–65 band, gates 10/10 · 12/12 cold and warm, speed inside the
  noise bands. **It ran on a dump boot, so its KV pool figure (4,382,920) is meaningless** and
  production stays on bf16 until an ordinary load boot supplies the real number.
  [docs/07](docs/07-kv-and-draft-page.md) §7, [docs/10](docs/10-results-and-roofline.md) §2.1,
  [docs/11](docs/11-open-issues.md) §2.18.
- **The hyper-connection fusion kernel was written and it reached 40 % of its own ceiling.** A Triton
  kernel that tiles over tokens and reduces the post mapping in registers removes 30.4 % of the first
  two kernels' traffic and delivers **−14.9 to −15.5 %** on that pair, **−9.0 to −9.9 %** on the
  three-kernel route, and **−1.0 to −1.1 % of the prefill wall** — against a −2.1 to −2.8 % target.
  `residual_cur` is bit-identical at every M but one 7-element 1-ulp difference at M=64; `layer_input`
  is within one bf16 ulp on 5.1 % of elements. The shortfall is entirely bytes-per-second (187.7
  against the 229.5 GB/s the route it replaces gets), not tiling — a 33-configuration sweep could not
  improve the winner. **Not adopted standalone**: −1 % does not earn a boot when it also brings Triton
  JIT into the serving process and a configuration surface that is a cliff (the winner reads
  187.8 GB/s, its neighbours 79.4 and 44.5, and the shipped default was one of the bad ones). It rides
  the next image bundle with `had_in`. Also measured: fusing **loses** below M ≈ 1024 (+37.7 % at
  M=512), because the residual fits the 24 MiB L2 there and the re-read it deletes was never going to
  DRAM `[measured-here]`. [docs/10](docs/10-results-and-roofline.md) §5.5.1.
- **New lesson, and the more useful half of that kernel:** a GPU-free ahead-of-time compile check
  reported 18 of 18 configurations building and all inside the shared-memory limit; at real launch
  **6 of the 18 failed with `OutOfResources`**, the reported 36,864 bytes against 106,496 actually
  needed. **A compile check answers "does it build", never "does it run"**
  `[measured-here]`. [docs/11](docs/11-open-issues.md) §4.
- **Closed: dual-batch overlap (DBO).** The all-reduce is 16.5 % of a prefill chunk at 99.3 %
  occupancy, so overlapping it carried the largest number on the open-issues page. The mechanism does
  suit it and the patch is *smaller* than we estimated (~95–160 lines, five files, and it does not
  touch the model file — one bottleneck covers all 102 collectives, and the mHC state is thread-local,
  so our "medium-to-large patch, mHC at risk" reading was wrong on both counts `[retracted]`). The
  arithmetic kills it: splitting the batch pays the MoE expert weight stream **twice**, +73 to +232 ms
  per chunk against −135 ms of hideable collective. Prefill lands at **−6.3 % to +8.0 %** — a coin
  toss — and decode is a clear loss (**C1 +38 %**, C8 +6 %), with the drafter unable to micro-batch,
  the breakable CUDA graph disabled for both models, and a silent-corruption hazard where a split
  request restarts the KDA recurrent state from zero in 34 of 45 layers. Raising the batched-token
  budget beats every overlap variant for the same KV price and no code. **Do not build it.**
  [docs/11](docs/11-open-issues.md) §2.17.
- **Also dead, each for a checkable reason, read out of the image with the engine down:** async
  tensor parallelism, the sequence-parallelism pass and the FlashInfer all-reduce+RMSNorm fusion all
  need `torch.compile`, which this model family never enters; and `world_size = 3` is excluded by the
  supported-world-size lists of FlashInfer (2/4/8/16), custom all-reduce (2/4/6/8/16) and NCCL
  symmetric memory (minimum 4). DeepEP is installed and never engages at `data_parallel_size = 1`.
  Model-level sequence parallelism is a one-line gate and a bad idea: identical bytes, collective
  count 90 → 180, **+10…15 % worse at decode**.
- **One survivor in that class, and one free probe.** Attention-scoped micro-batching — split across
  the attention block only, rejoin before the MoE stage — hides 44 % of the collective while paying
  only for attention weights streaming twice: **−3 to −6 % of prefill** `[estimate]`, ~150 lines, and
  it inherits the same KDA hazard. Before any of it, a model-free probe should establish whether an
  all-reduce on a second stream overlaps a GEMM **at all** on this part. Written, not run
  `[not tested]`.
- **The measurement protocol got a floor.** Round-to-round spread inside a single settled arm is
  **C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %** `[measured-here]` — C4 is the noisiest column, which
  is the opposite of the intuition. Two rules now apply to every table here: **a difference of 3 % or
  less is written down as "equal"**, and above that floor it still has to clear its own metric's band.
  This reclassifies the combine-staging arm's +2.3 % at C4 and patch 0007's −0.9…+4.2 % as equal;
  production 5 → 6 survives on its +5.6 % at C8. Also corrected: the quick-arm harness claimed five
  rounds with two discarded and ran three `[retracted]` — it is now one warm-up plus three measured
  rounds, and the warm-up ramp it was written for is gone anyway on a warm tuner cache.
  [docs/09](docs/09-measurement-protocol.md) §1.1–§1.2.
- **New planning rule: a patch change costs a dump boot.** The fast-load sidecar's identity covers
  *every* `patch-*.py` and the whole prelude, so three patches that touch no weight byte refused a
  boot and cost an hour. Budget the 682-second dump boot into any arm that adds a patch, and never
  record a dump boot's KV pool as a result. The narrower gate this argues for is written up but not
  written. [docs/09](docs/09-measurement-protocol.md) §11, [docs/11](docs/11-open-issues.md) §2.21.
- **Rank memory imbalance, now with a number and a diagnosis label.** The weights are identical on all
  three ranks (`Model loading took 54.86 GiB` ×3), yet non-torch memory reads **1.50 GiB on rank 0
  against 9.48–9.72 GiB on the workers** — about **8.2 GiB per worker stranded**, and the pool is sized
  by the worst rank, so equalising it would be worth **8–26 % of pool**. Larger than every kernel item
  left. Nobody knows yet what that memory is `[measured-here]`.
  [docs/11](docs/11-open-issues.md) §2.3.
- **The memory ladder is re-opened rather than settled.** The 0.85 rejection predates the fast-load
  work that removed the page-cache spike; the same configuration now sits at 11–12 GiB free with zero
  swap, and 0.82–0.83 was never tried. `--kv-cache-memory` — sizing the pool in bytes rather than as a
  fraction of the device — has never been used. Ladder first, pin last
  `[not tested]`. [docs/07](docs/07-kv-and-draft-page.md) §6.
- **`NCCL_ALGO=Ring,Tree` has never been run on this mesh**, and it is the cheapest untried thing in
  the repository: our launcher forces `Ring`, decode is latency-bound on a fixed 102 collectives per
  step, and a tree is ~3.2 steps against a ring's 4 at `world_size = 3`. Expected −1…3 % of a decode
  step; the sweep is model-free and costs nothing `[not tested]`.
  [docs/06](docs/06-nccl-mesh.md) §14 item 8.
- **New `systemd/` directory — a template, deliberately not installed.** With it, the hazard it
  exists to name: the NVFP4 sibling's `harem-motor.service` is `enabled` on all three of our nodes, so
  a reboot brings up the **other** engine on the same GPUs. The template's three unfinished pieces are
  named rather than fixed — its preflight script does not exist, systemd will not honour the
  worker-2 → worker-1 → head start order on its own, and its `ExecStop` names the wrong container.
  [systemd/README.md](systemd/README.md), [docs/11](docs/11-open-issues.md) §2.20.
- **A complete retraction audit.** Every published claim of ours was re-read against the raw data
  behind it; **24 did not survive** and all 24 are now in one table with what replaced them, including
  ones that had only been corrected in passing: the chat template we serve matches neither checkpoint
  on disk and its provenance is unverified; `NCCL_BUFFSIZE` was listed as an open lever twelve hours
  after being eliminated; `NCCL_MAX_NCHANNELS=8` had been "already tried" only in combination with
  `NCCL_PROTO=LL`; `--language-model-only` does not stop the vision tower being built, only run.
  Six of the 24 are a ruler we quoted instead of measured, four are a single pair of sweeps treated as
  a result, three are an arithmetic model a bench refuted, two are our own tooling disagreeing with
  our own documentation. [docs/11](docs/11-open-issues.md) §1.9.

## 2026-09-05 — where a step actually goes, and four items closed

Production configuration 6 is **unchanged**. Everything in this entry is measurement.

- **Both rulers were wrong, by ~22 %.** Measured on this device in our own image: achievable read
  bandwidth **225.2 GB/s** against a vendor 273, BF16 GEMM peak **97.3 TFLOP/s** against an implied
  ~125. Every roofline percentage published here before today was optimistic by that much
  `[retracted]`. Two new tools, `bench/bw.py` and `bench/gemmpeak.py`, seconds each; run them in the
  same process as the thing you are measuring. The read ruler itself drifted 6.5 % across three runs
  on the same idle machine the same morning, so percentages are now given as bands where it matters.
- **Step-time breakdown, per 2,048-token prefill chunk** (1,109 ms, occupancy 99.3 %): MoE trellis
  GEMM **26.4 %**, NCCL all-reduce **16.5 %**, dense BF16 GEMM **16.2 %**, hyper-connection mixing
  **11.7 %**, MLA 8.2 %, KDA 7.5 %, MoE `had_in` 6.1 %, KV zeroing 1.3 %, DSA indexer 0.6 %. Per C1
  decode step (89.1 ms): dense BF16 GEMM **44.8 %**, MoE trellis GEMM 29.3 %, all-reduce 10–15 %, and
  the k=7 drafter **19.5 %**. Since the previous configuration, prefill throughput at 8.4K is
  **+43 %** and the C1 step is **−17.5 %**.
- **Two corrections to our own earlier reading of the same trace** `[retracted]`: the `mhc_*` kernels
  are hyper-connection mixing, a class of their own worth 11.7 %, not dense GEMM and not the indexer;
  and MLA is 7.4 % in a steady chunk, not the 9 % a window average reported.
- **Method, stated because it has a caveat.** The running engine had no profiler endpoint and a
  restart was not available, so this is a reconciliation — structure from an earlier trace
  re-segmented per chunk, changed classes re-measured model-free, totals measured live — with a
  **2.8 % residual**. Read NCCL as a 14–17 % band. One profiling boot closes it.
- **Closed: expert-stationary MoE scheduling.** Our 14–27 % traffic estimate rested on a traffic
  model a trace cannot verify. The kernel author wrote a bench for it (`9b17ea9`); run unmodified on
  GB10, three times: doubling blocks per expert costs **1.11×**, not 2×. The trellis stays resident.
  Nothing to win `[measured-here]`.
- **Closed: the KV-zeroing gate.** The kernel runs at 100 % of the memset roofline and zeroes
  2.4–2.9 GB per chunk against ~3.4 MB of real new KV, so the only lever was to skip it. It cannot be
  skipped here: **85.5 %** of those bytes are MLA pages co-owned with Mamba/KDA state in this model's
  hybrid layout, which is the Mamba half of vLLM's condition and is independent of precision. A
  fail-closed gate was written so the machine checks the conclusion; on this model it refuses to boot,
  by design. Safe remainder 0.19 % of prefill; no partial mode written `[measured-here]`.
- **Closed: a cooperative (`grid.sync`) MoE stage.** Outside a CUDA graph the barrier wins on a 48-SM
  part, as predicted — up to 33 % at medium sizes. **Inside a graph, which is what production runs,
  the sign flips back** and it costs 0.2–0.3 µs per phase boundary. The deciding detail was the
  graph, not the SM count `[measured-here]`.
- **Closed: the DSA indexer**, at 0.6 % of prefill — the same conclusion an earlier micro-benchmark
  reached from the other side, now confirmed from the share itself.
- **Hyper-connection mixing measured**: 86–91 % of the ruler, memory-bound by a factor of 76, **not**
  launch-bound (CUDA graphs change it by 0.03 % at M=2048), **not** badly tuned (the two available
  knobs are worth 0.4 % of prefill, and the third kernel does not compile above 96 threads), and the
  torch fallback is unreachable on CUDA and 5–15× slower. One real lever remains: fusing the first two
  kernels to stop re-reading the residual, **−2.5 to −2.7 % of prefill**, which needs a new large-M
  kernel — forcing the existing fused one is +32 % worse. Our earlier −3.6 % estimate was 30 %
  optimistic `[retracted]`.
- Upstream took `exl3_moe_had_in` in `a47da6e` (−10…18 % on that kernel, ~0.2–0.3 % of prefill here).
  **Not in the production image**; queued for the next build.
- New protocol rule, learned the hard way: **one measurement holds the cluster at a time**, written in
  a lock file, and three-node NCCL work needs the engine **down** rather than idle — a fabric sweep
  beside a live engine can exhaust queue-pair resources and take its next collective with it.
  [docs/09](docs/09-measurement-protocol.md) §10.

## 2026-09-05 — the fabric ceiling is PCIe, and a transport rewrite that changed nothing

- **Retracted, one day after we published it** `[retracted]`: "the pair of cables is worth 50 GB/s,
  so the collective is at 28 % of the fabric". That is the **wire**. Each ConnectX-7 sits in a
  **PCIe Gen5 x4** slot (`LnkSta: Speed 32GT/s, Width x4`) and carries ~15 GB/s regardless of its two
  200 Gb/s ports, so the real ceiling is **~30 GB/s per node** and the collective at ~20 GB/s is at
  about **70 %** of it. The old 13 GB/s ceiling was never "half a link" either — it was one card's
  PCIe limit at 87 % of it, which is why the second cable, on the second card, took it to 20 and not
  to 40. **Remaining fabric headroom ≤30 %, worth 2–4 % of prefill**, against the 12–17 % we had
  priced it at. Two wrong ceilings in two days, both computed from a datasheet.
- **Patch 0007: one-sided transport (receiver-advertised FIFO + `RDMA_WRITE_WITH_IMM`), built,
  measured, not adopted.** +977/−16 lines over 0004–0006: FIFO in the sender's memory (one per cable
  after 0005), zero-byte RECV armed before the slot is advertised so **RNR becomes structurally
  impossible**, torn-slot double check, fail-closed ring overrun, a version handshake that keeps
  `mesh_qp_info` at 32 bytes and refuses a peer that does not speak the extension, and no dependence
  on NCCL's `*request = NULL` contract. Default `send` is byte-for-byte the old path.
- **It does exactly what it was designed to do and it is worth nothing here.** Six arms, two
  repetitions: every write arm reports **zero** RNR retries and **zero** out-of-buffer events at every
  size, against 1–9 per operation on the control — and throughput does not move. The gate (≥1.3× at
  ≥16 MB) was not met at any FIFO depth, with or without the flush, and all six arms sit within 0.17 ms
  of each other on the decode-sized message. Engine arm: C1 56.4, C8 171.1, prefill-fresh 1,763 against
  production's 56.9 / 168.9 / 1,792 — differences in both directions, inside boot spread, gates
  10/10 · 12/12 cold and warm `[measured-here]`.
- **Why**: at ~20 GB/s the transfer is against a PCIe wall, not a flow-control stall. Removing RNR
  from a path that was not waiting on RNR buys nothing.
- **Kept, not deleted, and not offered upstream.** The patch, its unit-test parity with the 0004–0006
  baseline and the measurement that rejected it are in the repository. Sending a transport rewrite
  upstream on the strength of a mechanism that moves no number would waste the maintainer's time.
- **Patches 0004, 0005 and 0006 are now on a public fork and offered as a pull request** —
  [`NNNtrance/nccl-mesh-plugin`](https://github.com/NNNtrance/nccl-mesh-plugin), branch
  `gb10-dual-link-ptrcuda` on `19924dcc`, [PR #59](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59),
  referencing the issue thread the findings were reported on. 0007 is not in it.
- **Closed as an open item**: the collective's share of a step, which had been "the cheapest unspent
  measurement in this repository" through three separate changes. It is 16.5 % of prefill and 10–15 %
  of a C1 decode step. The C8 split still needs a profiling boot.
- **Newly written down as the larger lever**: the all-reduce is *serialised* against compute at 99.3 %
  occupancy. Making the fabric faster is worth ≤2–4 % of prefill; making the collective **overlap**
  reaches for most of 16.5 %. Nobody has tried. [docs/11](docs/11-open-issues.md) §2.17.

## 2026-09-05 — production configuration 6: both cables, and no host bounce buffer

- **Half the fabric had never carried a packet.** Two cables run between every pair of nodes;
  `mesh_connect()` discards NCCL's device index and stops at the first reachable peer address, so
  every channel to a peer rode one cable. `port_xmit_data` on the second cable of each pair read
  **exactly zero since driver load**, on all three nodes. The 13 GB/s we had been calling "the link"
  was one cable of a pair — and, as the next day's entry records, it was also one card's PCIe limit.
- `patches/kernel/0005-device-aware-link-selection.patch` (~30 lines, no wire-format change) picks
  `dev % usable` among the parallel links; with one cable, or `NCCL_MESH_LINKS_PER_PEER=1`, the
  selection is bit-identical to the stock plugin. 64 MB all-reduce **12.0 → 16.7 GB/s**.
- `patches/kernel/0006-ptr-cuda-dmabuf-and-flush.patch` advertises `NCCL_PTR_CUDA` — two lines; the
  plugin's `regMr` already registered CUDA pointers and NCCL was simply never handing it one — plus a
  real RDMA_READ `iflush` and a real DMA-BUF path. 64 MB all-reduce **16.7 → 20.8 GB/s**, RNR retries
  per operation 15 → 3.
- Engine, three rounds per arm: **C1 54.5 → 56.9, C4 112.0 → 118.5, C8 159.9 → 168.9** (+4–6 %),
  prefill-fresh 1,709 → **1,792**, TTFT C1 0.47 → 0.41 s, gates 10/10 · 12/12 cold and warm.
- **What it cost:** nothing measurable — the KV pool moved +0.4 % (inside boot noise), which was the
  line to watch because `NCCL_PTR_CUDA` moves NCCL's buffers into memory accounted under
  `gpu-memory-utilization`. The real price is not a number: a patched plugin to maintain.
- **`NCCL_MAX_NCHANNELS=16` rejected.** Restoring 8 channels per cable is the obvious follow-up and
  it is 2.5× slower on the decode-sized message (STEP90 9.3 → 26.2 ms), in both patch families. 8
  stays.
- **`NCCL_MESH_DMABUF=1` rejected.** It works — which settles whether `ibv_reg_dmabuf_mr` accepts
  these buffers here — and is slower than plain `ibv_reg_mr` (64 MB 18.1 against 20.8).
- Patch `0004-min-rnr-timer` is carried into the production build at `NCCL_MESH_MIN_RNR_TIMER=1`
  rather than staying on the shelf. Its isolated engine contribution is still unmeasured.
- **Retracted:** "the ceiling is ~13 GB/s against a 25 GB/s link, and the GPUDirect path needs a
  plugin redesign". The link is a pair of cables with one idle, and the device-pointer path was two
  lines. See [docs/11](docs/11-open-issues.md) §1.6. (The "50 GB/s pair" in this entry was itself
  retracted the next day — the ceiling is the cards' PCIe slots, ~30 GB/s per node; §1.7.)
- Reported upstream as a follow-up on the plugin's issue thread, with both patches offered.

## 2026-09-05 — production configuration 5: the MLA tuner cache stops charging us for measurement

- Image moved to `cuda-exl3` `9bf594c` ("Persist the MLA tuner cache across processes"), which adds
  `CUDA_EXL3_TUNE_CACHE`. The tuner's map had been process-local, so every boot re-tuned and every
  unseen batch shape bought a ~15 ms tune **while serving** — which is what polluted the first rounds
  of every A/B on this stack.
- Measured: **18 tune events before serving → 0**, none during a sweep, and round 1 stops being a
  penalty (cold cache C8 round 1 → round 3 −3.4 %; warm +2.7 %, i.e. unordered noise).
- **Protocol change:** five sweep rounds with two discarded → **three rounds, median of three**, on
  an image with the cache and a warm cache file. About 15 minutes saved per arm. Five rounds still
  stands everywhere else, including the boot that writes the cache.
- Speed unchanged by design: C1 54.5, C8 159.9 against 54.4 and 161.8, inside the spread. Gates
  10/10 · 12/12. KV pool 4,429,752.
- **What it cost:** a new image invalidates the fast-load sidecar, because the manifest records the
  image tag — the preflight refused the boot, correctly, and regenerating cost one **682 s** dump
  boot on all three nodes. That is now the standing price of every kernel-image change.
- The implementation is upstream's; ours was the measurement that asked for it. Credit in
  [CREDITS.md](CREDITS.md).
- Closes the open item in [docs/08](docs/08-fast-boot.md) §10.3 and
  [docs/11](docs/11-open-issues.md) §2.8. New page: [docs/12](docs/12-tuner-cache.md).

## 2026-09-05 — production configuration 4: fast boot

- Per-rank pre-sliced fast-load sidecar (`patches/tp3/harem_fastload.py` and friends): boot
  **618 s → 274 s**, weight load **426 s → 67 s**. Bit-identity proven twice — 1,475/1,475 tensors
  re-hashed against a manifest written from a full-checkpoint load, and a post-`process_weights` hash
  dump with no difference on any rank.
- `--enable-ep-weight-filter` plus a patch so the filter recognises EXL3's `.trellis` suffix: each
  rank now reads 96 of 288 experts instead of all of them.
- `--safetensors-load-strategy eager` and `--no-enable-flashinfer-autotune`.
- Page-cache remedy (`posix_fadvise(DONTNEED)` on checkpoint shards, `malloc_trim`) after a 4.1 % KV
  regression traced to reading 163 GB through the page cache on unified memory. Pool came back
  **above** the pre-change baseline: 4,484,848.
- Speed and quality unchanged, which was the intent. Gates 10/10 · 12/12 cold and warm.
- **Corrected:** `--no-enable-flashinfer-autotune` is worth about 3.5 s, not the 34 s it appears to
  be — most of that work moves into graph capture. See [docs/08](docs/08-fast-boot.md).

## 2026-09-05 — production configuration 3: the draft KV page

- Root cause found for a KV pool that was capped by a per-request **block counter**, not by memory:
  the DFlash2 draft's sliding-window group is given the backend's smallest kernel block (16 tokens),
  and because the port keeps that group independent it never reaches the unification step that would
  scale it up. The draft took **53 % of the blocks-per-request budget for 0.6 % of the memory**.
- `HAREM_SW_BLOCK_SIZE=256` (`patches/tp3/patch-swblock-tp3.py`, one anchor, env-gated):
  KV pool **2,428,769 → 4,413,223 (+82 %)**. 256 is the measured optimum; matching the target's
  3,328-token page makes the pool 7 % *worse* than doing nothing.
- Unpredicted bonus: the draft block table shrinks 16×, giving C4 +9 %, C8 +6 % and 20–30 % off TTFT.
- Cost: +9.2 % memory per block, and the draft group's prefix-cache matching unit coarsens from 16 to
  256 tokens.
- Memory ladder step to `gpu-memory-utilization 0.85` measured (+19 % pool, no speed change) and
  **rejected**: head node at 1.9 GiB free with 1.6 GB of swap in use, which breaks the 4 GiB rule.
  0.88 not attempted.

## 2026-09-05 — the NCCL mesh cliff

- Root cause for an all-reduce running at 0.6–1.9 GB/s between 128 KB and 4 MB while point-to-point
  over the same queue pairs stayed clean at 11–13 GB/s: the mesh plugin carries data with two-sided
  SEND/RECV, its only flow control is the receive-not-ready NAK, and it sets `min_rnr_timer = 12`
  (0.64 ms) where its own comment intends 0.01 ms (code 1). NCCL opens 64 channels on this fabric and
  one proxy thread services them round-robin, so the sender routinely outruns the receiver.
- **`NCCL_MAX_NCHANNELS=8`** adopted, environment only: model-free, the 512 KB decode all-reduce goes
  1,195 → 123 µs and one decode step's collectives 91.7 → 9.9 ms. In the engine, three boots and five
  rounds each: **C8 133.4 → 150.8 (+13.0 %)**, C1 +7.1 %, C4 +9.6 %, C6 +11.8 %; prefill, gates,
  acceptance and KV pool unchanged.
- Plugin patch (`patches/kernel/0004-min-rnr-timer.patch`) written, unit-tested and **not deployed** —
  with the channel cap its contribution is inside the noise.
- **Retracted:** an earlier reading that NCCL was choosing the LL protocol at 16 MB. Forcing LL there
  costs 20,114 µs against auto's 1,787 µs.
- **Protocol change:** five sweep rounds per arm, first two discarded. The tuner's warm-up window can
  be longer than two rounds and made the winning arm look 25–45 % worse on the first pass.

## 2026-09-05 — production configuration 2

- Image moved to `cuda-exl3` `f4987cf` ("do not fetch the MoE padding rows");
  `--max-num-batched-tokens` back to 2048 from 4096, recovering the KV pool (1,627,170 → 2,428,769)
  at a cost of ~5 % on fresh prefill.
- MoE input-transform fusion A/B (`61a17bc`): +1–4 % end to end over five rounds, but `f4987cf`
  reaches the same level another way and upstream dropped the fusion branch. Closed.
- k=7 versus k=5 A/B: k=5 raises the acceptance *rate* (63 → 73 %) and lowers accepted tokens per
  *step* (5.5 → 4.7); the second effect wins everywhere except prose and C4. **k=7 stays.**

## 2026-09-05 — upstream adopted, our patch retired

- Per-kernel comparison of four builds on identical shapes: our "zero the retired tile" choice is
  **10.7 % more expensive per MoE layer at M=2048** than upstream's "return, and let the combine skip
  those rows". Our `0002` patch retired in writing.
- Our one surviving kernel change kept and rebased: staging the combine's per-(token, k) facts in
  shared memory — combine −34 % at M=8, −36 % at M=64, −13 % at M=2048; end to end C6 +3.8 %,
  C8 +3.8 %.
- **Retracted:** the earlier report that an upstream build measured ~10 % slower end to end. It does
  not reproduce; what we had measured was **15.9 % boot-to-boot spread on C8 with nothing changed**.
- **Retracted:** the claim that the missing `n_rows` also costs the non-expert-parallel path. With no
  expert map the surplus tail is `-1` everywhere.
- `--max-num-batched-tokens 4096` adopted at this point: +9.5 % fresh prefill, −13 % mixed-load TTFT,
  −28.5 % KV pool. Later reverted, above.
- `NCCL_PROTO=Simple` rejected model-free without spending a boot: 2.8× worse at the C1 decode
  message, 4.4× at C8, no better at 16 MB.

## 2026-09-05 — the expert-parallel regression, root-caused

- TP=3 + expert parallel was **8–29 % slower than TP=2**. Cause: a one-line omission in the kernel's
  GEMM dispatch — the unsplit MoE launch never passed the live-row bound, so the surplus tail of
  `expert_ids`, which `expert_map` turns into a real local expert on one rank, ran a full GEMM over
  38 % of the grid at M=2048. Under expert parallelism every rank waits for that one.
- Per-rank MoE stage at M=2048: **18,401 → 10,107 µs (−45 %)**. End to end: C1 40.8 → 49.4,
  C4 59.5 → 99.6, C8 91.9 → 139.1, prefill 1,025 → 1,257. Reported upstream; fixed there in
  `a95e809`.
- Three of our own hypotheses refuted model-free with numbers (block padding, three-way collectives,
  the masking pass), and the GB10 top-k fallback we were pinned to turned out to be **faster** than
  the path it replaces below ~64K tokens of context.
- `gpu-memory-utilization` 0.85 → 0.80: cost nothing measurable and put all three nodes above the
  4 GiB free-memory rule for the first time.

## 2026-09-05 — TP=3 + expert parallel, first working boot

- Three nodes, 96 whole experts of 288 per rank, 64 → 66 head padding (22 per rank), vocabulary
  `padding_size` 192, shared expert padded to 2,112.
- Weights per node **81.53 → 54.86 GiB (−33 %)**; KV pool **825,000 → 2,947,441 (+257 %)**.
- Gates 10/10 · 12/12 cold **and after the full benchmark**; acceptance and accepted-tokens-per-step
  identical to TP=2 to three significant figures.
- Root causes fixed before the first boot by reading rather than running: the sidecar mount geometry
  (relative symlinks require identity mounts), and three divergent environment files.
- Root causes fixed during: the vision tower ignoring `--language-model-only` (fatal at TP=3, and it
  had been quietly carrying 1.05 GiB per rank at TP=2); a drafter head check that could not tell a
  legitimate pad from a disaster; an evidence log line being silently discarded because its logger
  sat outside the configured hierarchy; and KV pool arithmetic made visible.

## 2026-09-04 — DFlash2 ported to this stack (at TP=2)

- Speculative decoding brought into an image that had never run it, as a three-way git merge rather
  than a hand-copy. C1 aggregate **14.4 (no draft) → 30.5 (MTP k=3) → 42.9 (DFlash2 k=7)**;
  accepted tokens per step 3.31 → **5.37**.
- Two things the upstream delta did not cover: the target-side EAGLE3 interface, absent from this
  image entirely; and KV cache grouping, where the draft's sliding-window layers knock this model off
  its own grouping path. Fixed by giving the draft its own independent cache group — which is also
  what later capped the KV pool.
- Fail-closed additions: a quantized drafter is refused outright, head counts must divide the TP
  size, and the image is not produced if a ported symbol fails to resolve.

## 2026-09-04 — first EXL3 boot on this hardware

- `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at revision `b20c49ba`, two nodes, TP=2, no expert
  parallelism. Boot 471 s after a first attempt died in the sparse-attention indexer's persistent
  top-k; the GB10 top-k overlay cleared it.
- Gates 10/10 · 12/12; MMLU sample 86.4 ±0.7.

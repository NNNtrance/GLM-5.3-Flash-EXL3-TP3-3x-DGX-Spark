# Changelog

Dated entries for the stack this repository documents. This is a working record, not a release
history — the repository is a **private draft** and is overwritten in place as the work continues.

Speed figures are aggregate output tok/s on `scripts/hizset-v2.jsonl` at `reasoning_effort: low`,
temperature 0. Entries up to and including *production configuration 4* are medians of sweep rounds
3–5 with rounds 1–2 discarded; from *production configuration 5* onwards they are medians of three
rounds, which is what the persisted MLA tuner cache bought — see
[docs/09](docs/09-measurement-protocol.md) §1 and [docs/12](docs/12-tuner-cache.md).

---

## 2026-09-05 — production configuration 6: both cables, and no host bounce buffer

- **Half the fabric had never carried a packet.** Two cables run between every pair of nodes;
  `mesh_connect()` discards NCCL's device index and stops at the first reachable peer address, so
  every channel to a peer rode one cable. `port_xmit_data` on the second cable of each pair read
  **exactly zero since driver load**, on all three nodes. The 13 GB/s we had been calling "the link"
  was one cable of a 50 GB/s pair.
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
  plugin redesign". The link is a pair of cables at 50 GB/s with one idle, and the device-pointer
  path was two lines. See [docs/11](docs/11-open-issues.md) §1.6.
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

# Contributing — help us close the gaps

This repository is a **private draft** and is overwritten in place while the work continues. If you
have access to it and comparable hardware, the items below are the ones a second cluster could
actually settle.

If you run one, please open a pull request adding your raw output under
`results/community/<your-handle>/<item>/` plus a short Markdown summary. Every number needs its
settings — image tag and which `cuda-exl3` commit, TP and EP, quantization, KV dtype, draft method
and `k`, `gpu-memory-utilization`, `--block-size`, `HAREM_SW_BLOCK_SIZE`,
`--max-num-batched-tokens`, `--max-num-seqs`, `NCCL_MAX_NCHANNELS`, temperature, reasoning effort,
`max_tokens`, concurrency, prompt type, how many sweep rounds and which were discarded, date — and an
evidence tier as defined in [STYLE-GUIDE.md](STYLE-GUIDE.md). We will credit you in
[CREDITS.md](CREDITS.md).

## Items we did not run (most useful first)

1. **`NCCL_MAX_NCHANNELS=12` on the engine.** Model-free it is indistinguishable from 8 and slightly
   better at 3.4 MB, and it keeps more parallelism for the largest messages. One boot settles it. The
   only change is one token in `EXTRA_ENV`. See [docs/06](docs/06-nccl-mesh.md).
2. **The mesh plugin patch, A/B'd on the engine.** `patches/kernel/0004-min-rnr-timer.patch` is built
   and unit-tested and has never had a boot, because with the channel cap its model-free contribution
   is inside the noise. But the live engine still shows roughly 42,000 receive-not-ready events per
   node over five minutes, which is 1–3 % of wall clock, and the patch makes each about 64× cheaper.
   Point `NCCL_MESH_PLUGIN_DIR` at a patched build and run the standard five-round protocol.
3. **`--max-num-batched-tokens 3072`.** 2048 and 4096 are both measured; the value between them is
   not. Report the KV pool, fresh prefill, mixed-load TTFT and C1–C8.
4. **The persisted MLA tuner cache** (`cuda-exl3` `9bf594c`). Build an image with it and report tune
   events per boot and the round-1 versus round-3 C8 spread. If it works, the five-round protocol in
   [docs/09](docs/09-measurement-protocol.md) gets cheaper for everyone.
5. **MMLU at TP=3.** Ours is a 1,995-question sample measured at TP=2. The gates are identical
   between the two arrangements, so we do not expect a difference — which is exactly why someone
   should check.
6. **The newer checkpoint revision** (`aba59d21`, four days newer than the `b20c49ba` we pinned). A
   clean A/B against ours with the correctness probe, the code exam and an MMLU sample.
7. **The memory ladder at 0.82–0.83.** 0.85 was rejected on the free-memory rule, but that was
   measured before the page-cache remedy in [docs/08](docs/08-fast-boot.md). The rung may sit
   differently now.
8. **The fast-load read path.** `HAREM_FASTLOAD_READ=mmap` exists in the code and has never been run.
   The sidecar currently reads at 0.88–1.04 GB/s where the same NVMe gives another loader 3.1 GB/s.
   Weight load could go from 67 s to about 20.
9. **`block_m` under expert parallelism.** The alignment needs the global expert count and the
   block-size heuristic is about rows per expert. A sweep would say whether the two uses want
   different numbers.
10. **A two-node (TP=2) measurement on the current stack.** Everything here moved a long way since
    the TP=2 numbers were taken, and readers with two Sparks have no current figures at all.
11. **`NCCL_MAX_NCHANNELS=8` on an NVFP4 stack.** Same plugin, same fabric, same TP=3. If it
    transfers, it is +13 % at C8 for one line, and it would change the comparison in
    [docs/10](docs/10-results-and-roofline.md) §3.
12. **Anything at max reasoning effort.** Everything here is at `low`. Expect 5–12× the time.

## What we would rather you did not send

- A single pair of sweep rounds as evidence for anything. Boot-to-boot spread on this stack is up to
  16 % on C8; see [docs/09](docs/09-measurement-protocol.md) §2. We have published a kernel
  conclusion drawn from one pair and had to withdraw it.
- A prefill number measured on a repeated prompt. It reads the prefix cache and overstates by up to
  55 %.
- A speed number without the quality gates from the same boot, cold **and** after the benchmark.

## House rules for a pull request

- Keep our documents unchanged unless you are correcting an error; in that case cite the raw file
  that shows the correction. We keep the retraction, we do not delete the mistake — see
  [docs/11](docs/11-open-issues.md) §1.
- **Never include host names, LAN addresses, user names, home paths or tokens** in anything you
  upload. Grep before you push; we do, before every commit.
- Report what a gain cost — speed, quality and memory together. If it cost nothing, say that you
  looked and give the numbers that show it.

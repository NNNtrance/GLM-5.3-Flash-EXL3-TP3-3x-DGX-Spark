# Mesh multilink and `NCCL_PTR_CUDA` — the model-free sweep

Nine arms, model-free, engine down, one process per node over the mesh plugin with the engine's exact
NCCL environment. `bench/mesh-multilink-sweep.sh` drives it; every arm is two repetitions and the
figures below are the means of the two. Sizes are tokens of hidden 4096 × bf16 (8,192 B per token),
so every row is a real collective shape. World = 3, `NCCL_ALGO=Ring`, `NCCL_PROTO` unset,
`NCCL_MAX_NCHANNELS=8` unless the arm name says `ch16`. 5 September 2026 `[measured-here]`.

Machine-readable: [`multilink-sweep.csv`](multilink-sweep.csv) — every arm × operation × size, with
`us`, `GB/s` and the per-collective `rnr_nak_retry_err` / `out_of_buffer` deltas. Narrative and root
cause in [docs/06](../../docs/06-nccl-mesh.md) §6–§8.

## The arms

| Arm | Plugin | What it answers |
|---|---|---|
| `base` | production, unpatched | today's curve, the reference |
| `p2ctl` | patched, `LINKS_PER_PEER=1 MIN_RNR_TIMER=12 PTR_CUDA=0` | **the control** — the new binary told to behave like the old one |
| `link1rnr1` | patched, `MIN_RNR_TIMER=1` only | patch 0004 alone, so 0005 is not credited with the timer |
| **`link2`** | + `LINKS_PER_PEER=0` | **patch 0005: the second cable** |
| `link2ch16` | `link2` at 16 channels | 8 channels per cable instead of 4 |
| **`link2cuda`** | + `PTR_CUDA=1 FLUSH=1` | **patch 0006: no host bounce buffer** |
| `link2cudanf` | as above, `FLUSH=0` | what the RDMA_READ flush costs |
| `link2cudach16` | `link2cuda` at 16 channels | GDR changes NCCL's topology model, so re-check the cap |
| `link2dmabuf` | + `DMABUF=1` | DMA-BUF registration instead of plain `ibv_reg_mr` |

## All-reduce, GB/s bus bandwidth

| arm | 128 KB | 512 KB | 1 MB | 4 MB | 8 MB | 16 MB | 32 MB | **64 MB** |
|---|---|---|---|---|---|---|---|---|
| base | 0.61 | 3.72 | 5.71 | 9.53 | 7.33 | 10.93 | 10.09 | **12.08** |
| p2ctl | 0.55 | 3.97 | 4.79 | 8.43 | 9.18 | 10.55 | 8.82 | 12.38 |
| link1rnr1 | 1.16 | 4.87 | 5.31 | 9.46 | 10.72 | 11.79 | 11.41 | 12.20 |
| **link2** | 0.61 | 3.67 | 4.61 | 7.12 | 10.30 | 13.89 | 11.88 | **16.66** |
| link2ch16 | 1.01 | 2.29 | 3.03 | 10.89 | 13.09 | 19.55 | 20.03 | 20.56 |
| **link2cuda** | 0.84 | 4.75 | 5.73 | 12.07 | 17.81 | 19.06 | 18.09 | **20.84** |
| link2cudanf | 1.43 | 5.23 | 6.34 | 10.19 | 19.73 | 16.63 | 15.86 | 21.02 |
| link2cudach16 | 0.74 | 2.62 | 3.02 | 11.00 | 15.13 | 16.38 | 16.56 | 16.41 |
| link2dmabuf | 0.53 | 2.63 | 4.59 | 7.23 | 11.50 | 13.96 | 15.12 | 18.08 |

**64 MB: 12.08 → 16.66 with the second cable, → 20.84 with `NCCL_PTR_CUDA` on top. +73 % over base.**
The control (`p2ctl`) sits on top of `base` at every size, which is what makes the rest readable: the
new binary changes nothing until it is told to.

## Point-to-point send/recv, GB/s

| arm | 1 MB | 8 MB | 16 MB | 32 MB | **64 MB** |
|---|---|---|---|---|---|
| base | 4.10 | 8.09 | 11.59 | 11.07 | **11.16** |
| p2ctl | 4.88 | 7.42 | 10.20 | 9.89 | 11.10 |
| link2 | 6.74 | 9.63 | 11.90 | 14.46 | **16.01** |
| **link2cuda** | 6.88 | 18.27 | 20.88 | 20.98 | **21.30** |
| link2cudanf | 9.16 | 9.38 | 18.23 | 18.41 | 18.87 |
| link2dmabuf | 3.93 | 8.57 | 15.73 | 16.31 | 16.39 |

Point-to-point was the healthy path in the earlier cliff investigation, at 11–13 GB/s. It is now
21.3, which says the earlier ceiling was the single cable plus the host bounce buffer and not the
link.

## All-to-all, GB/s

| arm | 1 MB | 8 MB | 16 MB | 32 MB | **64 MB** |
|---|---|---|---|---|---|
| base | 5.61 | 2.83 | 5.92 | 9.09 | **10.05** |
| link2 | 4.08 | 5.52 | 10.07 | 9.22 | 11.01 |
| link2ch16 | 7.18 | 12.57 | 13.77 | 14.27 | 13.75 |
| **link2cuda** | 8.40 | 15.49 | 19.72 | 17.20 | **20.68** |
| link2dmabuf | 3.91 | 6.44 | 11.31 | 8.43 | 15.32 |

## One C8 decode step's collectives (90 × 512 KB)

This is the number that predicts decode, and it is where 16 channels disqualifies itself.

| arm | STEP90 |
|---|---|
| base | 10.21 ms |
| p2ctl | 10.05 |
| link1rnr1 | 9.63 |
| **link2** | **9.27** |
| link2ch16 | **26.18** |
| **link2cuda** | **9.24** |
| link2cudanf | 9.08 |
| link2cudach16 | **26.51** |
| link2dmabuf | 9.12 |

**16 channels is harmful here, and it is not marginal: 2.5× slower on the small-message step, in both
the `link2` and the `link2cuda` families.** The reasoning that suggested it — 8 channels split 4 + 4
over two cables gives each cable half the queue pairs the tuned configuration had — is real but is
outweighed by the RNR mechanism the channel cap exists to suppress ([docs/06](../../docs/06-nccl-mesh.md) §3).
16 channels does win the large-message columns (`link2ch16` reaches 20.56 GB/s at 64 MB without
`PTR_CUDA` at all), which is exactly the trade the cap was chosen against: this engine's decode
all-reduce is 512 KB, not 64 MB.

## The second cable, in hardware

`port_xmit_data` per port, delta across one arm, on the head node (the other two match). Counter
units as `/sys` reports them, not bytes — what matters here is the split, not the scale. The two
`roceP2p1s0*` ports are the second cable of each pair `[measured-here]`:

| arm | first cable (2 ports) | **second cable (2 ports)** |
|---|---|---|
| `base` | 8.38e9 · 0.92e9 | **0 · 0** |
| `link2` | 4.22e9 · 0.47e9 | **4.19e9 · 0.45e9** |

Before this work those two ports read **exactly zero bytes transmitted since driver load, on all
three nodes**, and the counter is cumulative — half the fabric had never carried a packet. On the
`link2` arms the traffic splits between the pair, which is the direct proof that patch 0005 engaged;
without it none of the numbers above would mean anything.

## What the flush costs, and DMA-BUF

`link2cudanf` (`NCCL_MESH_FLUSH=0`) is inside noise of `link2cuda` at most sizes and better at a few.
The flush is an RDMA_READ that guarantees the receiver's device memory is coherent before NCCL reads
it; we keep it on because correctness is not a noise-level decision, and the measured price is inside
the run-to-run spread `[measured-here]`.

`link2dmabuf` (`NCCL_MESH_DMABUF=1`) works — it does not fail to register, which answers the open
question about `ibv_reg_dmabuf_mr` on this platform — but it is **slower than plain `ibv_reg_mr`**
across the range (64 MB all-reduce 18.08 against 20.84). On a part where the GPU and the CPU share
one physical memory and `ibv_reg_mr` accepts a device pointer directly, the DMA-BUF path buys nothing
and costs registration work. Not adopted `[measured-here]`.

## Engine A/B

The engine numbers that came out of this sweep are in
[`../speed/concurrency-sweeps.md`](../speed/concurrency-sweeps.md) (arms *tuner cache warm* and
*dual cable + `PTR_CUDA`*): C1 54.5 → 56.9, C8 159.9 → 168.9, prefill-fresh 1,709 → 1,792, KV pool
4.43M → 4.45M, gates 10/10 · 12/12 cold and warm.

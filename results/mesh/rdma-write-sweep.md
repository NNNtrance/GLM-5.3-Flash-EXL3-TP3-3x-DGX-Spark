# Mesh sweep: the one-sided RDMA_WRITE transport (patch 0007)

Model-free, all three containers stopped, no model loaded. Plugin built from
`autoscriptlabs/nccl-mesh-plugin` at `19924dcc` with patches 0004, 0005, 0006 (the production build)
and, for the `p3*` arms, 0007 on top. `NCCL_MAX_NCHANNELS=8`, `NCCL_ALGO=Ring`, `NCCL_PROTO` unset,
`NCCL_NET=Mesh`, `NCCL_IB_DISABLE=1`, world = 3, `NCCL_MESH_LINKS_PER_PEER=0` (both cables per peer),
`NCCL_MESH_MIN_RNR_TIMER=1`, `NCCL_MESH_PTR_CUDA=1`, `NCCL_MESH_FLUSH=1` except where an arm says
otherwise. Message sizes 128 KB to 64 MB, three operations, **two repetitions per arm**, 5 September
2026. Tool: `bench/mesh_sweep.py` driven per arm. `[measured-here]`

Narrative and the decision: [`../../docs/06-nccl-mesh.md`](../../docs/06-nccl-mesh.md) §10; why the
ceiling is PCIe and not the cable, §9.

## Arms

| arm | plugin | transport | note |
|---|---|---|---|
| `p2best` | patched2 (0004+0005+0006) | send | **control** — today's production configuration |
| `p3send` | patched3 (+0007) | send | gate arm: the new binary must land on the control |
| `p3w64` | patched3 | write | FIFO depth 64 |
| `p3w128` | patched3 | write | FIFO depth 128 (the default) |
| `p3w256` | patched3 | write | FIFO depth 256 |
| `p3w128nf` | patched3 | write | depth 128, `NCCL_MESH_FLUSH=0` — a measurement, not a candidate |

## All-reduce, GB/s (NCCL bus bandwidth), mean of two repetitions

| arm | 1 MB | 4 MB | 16 MB | 32 MB | 64 MB |
|---|---|---|---|---|---|
| `p2best` (control) | 5.5 | 13.8 | 17.8 | 17.3 | 20.1 |
| `p3send` | 5.2 | 14.1 | 19.0 | 15.7 | 19.3 |
| `p3w64` | 6.4 | 8.0 | 21.2 | 19.8 | 22.1 |
| `p3w128` | 6.7 | 14.2 | 16.9 | 17.0 | 17.4 |
| `p3w256` | 4.7 | 7.4 | 15.4 | 16.4 | 16.2 |
| `p3w128nf` | 6.3 | 7.7 | 16.7 | 16.3 | 20.1 |

All-to-all and send/recv at 16 MB / 64 MB, same runs:

| arm | a2a 16 MB | a2a 64 MB | s/r 16 MB | s/r 64 MB |
|---|---|---|---|---|
| `p2best` (control) | 9.6 | 16.4 | 18.4 | 20.0 |
| `p3send` | 11.8 | 16.1 | 13.8 | 17.7 |
| `p3w64` | 12.7 | 22.2 | 15.1 | 16.0 |
| `p3w128` | 16.1 | 19.0 | 22.1 | 22.4 |
| `p3w256` | 14.1 | 17.6 | 22.1 | 17.6 |
| `p3w128nf` | 15.7 | 21.3 | 22.1 | 20.9 |

**Read the spread before the ranking.** Two repetitions of the *same* arm differ by up to 30 % at the
large sizes on this fabric. No ordering in these two tables survives that, which is the finding: the
pre-registered gate was **≥ 1.3× the control at ≥ 16 MB** and no write arm meets it.

## The counters, which do not need a spread argument

Retries and out-of-buffer events per operation, across every size and both repetitions:

| arm | RNR / op | out-of-buffer / op |
|---|---|---|
| `p2best` (control) | 1.1 – 9.2 | 0.3 – 7.7 |
| `p3send` | 0.2 – 11.6 | 0.1 – 4.7 |
| **every `p3w*` arm** | **0.000** | **0.000** |

Zero, at every size, in both repetitions, at every FIFO depth, with and without the flush. The
mechanism works exactly as designed; it is the throughput that does not care.

## One C8 decode step (90 × 512 KB, the message the engine actually decodes with)

| arm | ms |
|---|---|
| `p2best` (control) | 9.120 |
| `p3send` | 9.138 |
| `p3w64` | 9.130 |
| `p3w128` | 9.137 |
| `p3w256` | 9.079 |
| `p3w128nf` | 9.246 |

Spread across all six arms: 0.17 ms, under 2 %.

## Cable split

`port_xmit_data` deltas were read per port, per node, before and after every arm. All four ports on
every node moved on every arm, which is the check that `NCCL_MESH_LINKS_PER_PEER=0` engaged and that
the write path did not silently fall back to one cable.

## Engine arm

One boot, three sweep rounds, gates cold and warm, against production configuration 6:

| | production 6 | `p3w128` (write) |
|---|---|---|
| C1 / C8 aggregate tok/s | **56.9** / **168.9** | 56.4 / 171.1 |
| prefill-fresh, median of 3 unseen ~8.3K prompts | **1,792** | 1,763 |
| prefill, warm 7K prompt | 1,506 | 1,457 |
| draft acceptance | 61–65 % | 60–65 % |
| KV pool | 4,449,035 | 4,462,809 |
| gates, cold / warm | 10/10 · 12/12 | 10/10 · 12/12 |
| free RAM worst node / swap | 11.3 G / 0.11 G | 11.1 G / 0.11 G |

Differences in both directions, all inside boot-to-boot spread
([`../../docs/09-measurement-protocol.md`](../../docs/09-measurement-protocol.md) §2). **Not
adopted.**

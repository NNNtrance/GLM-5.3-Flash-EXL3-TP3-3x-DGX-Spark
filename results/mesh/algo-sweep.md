# Mesh sweep: `NCCL_ALGO` — Ring, Ring,Tree, Tree

Model-free, all three containers stopped, no model loaded. Plugin built from
`autoscriptlabs/nccl-mesh-plugin` at `19924dcc` with patches 0004, 0005 and 0006 — the production
build. Environment identical to production except for `NCCL_ALGO`: `NCCL_MAX_NCHANNELS=8`,
`NCCL_MESH_LINKS_PER_PEER=0` (both cables per peer), `NCCL_MESH_MIN_RNR_TIMER=1`,
`NCCL_MESH_PTR_CUDA=1`, `NCCL_MESH_FLUSH=1`, `NCCL_PROTO` unset, `NCCL_NET=Mesh`,
`NCCL_IB_DISABLE=1`, world = 3. NCCL 2.30.7+cuda13.3. Message sizes 128 KB to 64 MB, three
operations, **two repetitions per arm**, 5 September 2026. Tool: `bench/mesh_sweep.py` driven per arm.
Fabric checked before the run: 4 of 4 neighbours resolved on every node. `[measured-here]`

Narrative and the verdict: [`../../docs/06-nccl-mesh.md`](../../docs/06-nccl-mesh.md) §12.2; the open
item it half-closes, §14 item 8.

Both repetitions are printed for every cell. That is the point of this file: at the small end the two
repetitions of one arm differ by more than the arms differ from each other, and any reading that
quotes a single repetition is quoting noise.

## All-reduce — GB/s (NCCL bus bandwidth), rep 1 / rep 2

| message | `Ring` | `Ring,Tree` | `Tree` |
|---|---|---|---|
| 128 KB | 0.89 / 1.04 | 1.22 / 1.24 | 0.60 / 0.43 |
| 256 KB | 4.71 / 3.03 | 1.80 / 4.83 | 1.03 / 1.21 |
| 512 KB (C8 decode) | 4.02 / 3.04 | 4.64 / 6.29 | 1.64 / 2.70 |
| 1 MB | 4.49 / 2.61 | 2.66 / 6.90 | 2.43 / 3.46 |
| 2 MB | 8.31 / 4.77 | 7.80 / 8.11 | 4.92 / 2.26 |
| 4 MB | 9.61 / 15.46 | 16.70 / 17.24 | 1.74 / 1.91 |
| 8 MB | 18.44 / 20.63 | 12.62 / 20.66 | 1.85 / 1.66 |
| 16 MB (prefill chunk) | 20.23 / 18.30 | 20.39 / 21.06 | **3.15 / 3.36** |
| 32 MB | 22.33 / 17.22 | 21.63 / 21.67 | 6.18 / 5.33 |
| 64 MB | 22.24 / 18.23 | 22.50 / 22.58 | **4.93 / 5.93** |

## All-to-all — GB/s, rep 1 / rep 2

| message | `Ring` | `Ring,Tree` | `Tree` |
|---|---|---|---|
| 1 MB | 8.82 / 7.02 | 2.07 / 9.05 | 2.90 / 1.63 |
| 4 MB | 8.72 / 9.32 | 9.09 / 5.05 | 5.06 / 5.02 |
| 16 MB | 17.68 / 19.33 | 19.31 / 17.90 | 12.28 / 13.40 |
| 64 MB | 22.02 / 21.35 | 22.46 / 15.99 | 12.41 / 10.51 |

## Send/recv — GB/s, rep 1 / rep 2

| message | `Ring` | `Ring,Tree` | `Tree` |
|---|---|---|---|
| 1 MB | 11.99 / 8.64 | 2.71 / 7.60 | 4.37 / 4.75 |
| 4 MB | 10.97 / 12.80 | 12.66 / 12.35 | 4.95 / 4.20 |
| 16 MB | 19.45 / 16.55 | 19.58 / 19.96 | 17.74 / 12.32 |
| 64 MB | **21.62 / 20.79** | 17.27 / 15.43 | 15.15 / 13.20 |

## The decode-step proxy — 90 collectives at the decode message size

This is the shape that decides the question: a decode step spends its collective time on a fixed
count of 102 small messages ([`../../docs/10-results-and-roofline.md`](../../docs/10-results-and-roofline.md)
§5.3), so a per-message win converts into step time directly.

| arm | rep 1 | rep 2 | mean |
|---|---|---|---|
| `Ring` | 9.165 ms | 9.669 ms | **9.42** |
| `Ring,Tree` | 9.118 ms | 9.045 ms | **9.08** |
| `Tree` | 11.596 ms | 18.442 ms | 11.60 (rep 1) / **15.02** (both) |

## RNR retries per operation, all-reduce

The counters say what the bandwidth table implies. `Tree` is not merely slower; it is in flow-control
trouble at exactly the sizes where it loses.

| message | `Ring` | `Ring,Tree` | `Tree` |
|---|---|---|---|
| 16 MB | 0.7 / 1.4 | 0.6 / 0.2 | **8.9 / 10.6** |
| 32 MB | 0.1 / 5.2 | 0.6 / 0.8 | **13.6 / 18.5** |
| 64 MB | 0.0 / 13.5 | 0.0 / 0.0 | **37.9 / 41.2** |

## Port counters

`port_xmit_data` deltas across each arm, all four fabric ports of all three nodes, in the counter's
own units (×10⁹; the counter is not bytes and is used here only for comparison between arms). The
check this exists for: every cable of every pair carried traffic in every arm, so no arm is measuring
half a fabric — the failure mode that invalidated an earlier sweep
([`../../docs/06-nccl-mesh.md`](../../docs/06-nccl-mesh.md) §6.1).

| node | ports | `Ring` | `Ring,Tree` | `Tree` |
|---|---|---|---|---|
| head | primary pair | 4.18 / 4.21 | 4.18 / 4.21 | **1.80 / 1.83** |
| head | second pair | 0.45 / 0.47 | 0.45 / 0.47 | **2.30 / 2.32** |
| worker-1 | primary pair | 4.18 / 4.21 | 4.18 / 4.21 | 3.67 / 3.70 |
| worker-1 | second pair | 0.45 / 0.47 | 0.45 / 0.47 | 0.45 / 0.46 |
| worker-2 | primary pair | 4.18 / 4.21 | 4.19 / 4.22 | 3.65 / 3.68 |
| worker-2 | second pair | 0.45 / 0.47 | 0.45 / 0.47 | **2.30 / 2.32** |

`Ring` and `Ring,Tree` move **identical** traffic on every port of every node, which is expected —
NCCL's tuner picked Ring for these sizes even when handed the list, so `Ring,Tree` is largely the same
schedule plus the option. `Tree` is the one that redistributes: the three nodes stop being symmetric,
the head transmits less than half what it does under Ring while its second pair carries five times
more, and worker-1 becomes the node that never uses its second pair. That asymmetry is the shape of a
tree — root, one internal node, one leaf — landing on a topology that has no hierarchy to reward it.

## Verdict

- **`Tree`: rejected.** 4–6× slower than Ring at 16 and 64 MB all-reduce, 23–96 % worse on the step
  proxy, and its RNR counters climb by an order of magnitude. Three nodes of paired PCIe Gen5 x4
  cards do not pay for a tree's lower step count.
- **`Ring,Tree`: unresolved, deferred.** Better than Ring on the step proxy by 3.6 % and at 4 MB,
  worse on `sendrecv` at 64 MB, and the arms swap places at 1 MB. The effect is smaller than the
  instrument's own repeat-to-repeat spread, so the sweep cannot settle it; a five-round engine arm
  can, at an expected −1…3 % of a decode step. Not run `[not tested]`.
- **`Ring` stays in the launcher.**

## What is not here

The raw per-arm logs and the per-node port counter dumps: they carry node names and fabric addresses
throughout. Every number above is transcribed from them.

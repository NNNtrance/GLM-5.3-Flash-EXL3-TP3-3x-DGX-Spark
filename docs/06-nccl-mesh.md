# 06 — The NCCL mesh: the cliff, the half of the fabric nobody was using, and the wall behind it

The mesh plugin is what makes a switchless three-node triangle work at all ([00](00-hardware-and-os.md)). This page
is two findings against it, three months of hardware apart in feel and one day apart in fact.

**The first** is a hole in the middle of its message-size range where the all-reduce runs at **0.6–1.9 GB/s** on
links that reach 12–13 GB/s at 16 MB, with this engine's decode all-reduce sitting in the deepest part of it. One
environment variable takes it out: set **`NCCL_MAX_NCHANNELS=8`** on all three nodes. It costs nothing measurable
and is worth **+12.6 % at C8** `[measured-here]`. Sections 1–5.

**The second** is that the ceiling we were then measuring against was wrong by a factor of two. There are **two
cables between every pair of nodes**, and the plugin was using one of them — the second cable of each pair had
transmitted **exactly zero bytes since driver load**, on all three nodes, for as long as the cluster had existed. A
~30-line change spreads the channels over both, a two-line change stops NCCL staging every transfer through a host
bounce buffer, and together they take the 64 MB all-reduce from 12.0 to **20.8 GB/s** and the engine to
**C8 168.9** `[measured-here]`. Sections 6–8.

**The third** is that the corrected ceiling was *still* wrong, in the other direction. Each ConnectX-7 sits in a
**PCIe Gen5 x4** slot and can carry about **15 GB/s** regardless of what its two 200 Gb/s ports advertise, so a node's
real fabric ceiling is ~30 GB/s and not the 50 GB/s per pair §6 claims `[retracted]`. That reading closes the case:
what is left on the fabric is **at most ~30 %**, worth 2–4 % of prefill, and a one-sided RDMA_WRITE transport we
wrote, measured and did **not** adopt confirms it — it removes RNR retries entirely and moves throughput by nothing.
Sections 9–10.

If you are reading this to fix your own cluster: do section 4 first, it is free. Then read section 6 and go and
look at your own `port_xmit_data` counters before you believe anything about your fabric's ceiling — and then read
section 9 and go and look at your cards' `LnkSta` before you believe anything about the ceiling you compute.

**Settings for every engine number in §1–§5** (§6–§8 carry their own block, and it differs). Image
`exl3-zeus:bc0e0f6s`, TP=3 + expert parallel, EXL3 4bpw weights,
`kv-cache-dtype fp8`, DFlash2 draft at k=7, `--block-size 256` with the matching overlay block size
([02](02-image-build.md)), `--max-num-batched-tokens 4096`, `--max-num-seqs 8`, `gpu-memory-utilization 0.80`,
temperature 0, reasoning effort **low**, realistic prompts (12 short English code prompts), five sweep rounds per arm
with the first two discarded ([09](09-measurement-protocol.md)), one boot per arm, `drop_caches` first, 5 September
2026. Plugin `autoscriptlabs/nccl-mesh-plugin` at commit `19924dcc`, NCCL 2.30.7+cuda13.3, `NCCL_NET=Mesh`,
`NCCL_IB_DISABLE=1`, `NCCL_ALGO=Ring`, `NCCL_PROTO` unset.

## 1. The symptom

Model-free: engine down, one process per node, the launcher's exact NCCL environment, the production plugin mounted
read-only. Sizes are tokens of hidden 4096 × bf16 (8,192 B per token), so every row is a real TP all-reduce shape.
World = 3, bus bandwidth `[measured-here]`:

| message | all-reduce | all-to-all | point-to-point send/recv |
|---|---|---|---|
| 311 KB | 0.52 GB/s | 1.85 GB/s | 4.78 GB/s |
| 704 KB | 0.61 | 4.88 | 5.44 |
| 3.6 MB | **0.83** | 3.70 | **10.97** |
| 18.2 MB | 11.28 | 6.25 | 12.24 |
| 64 MB | 12.31 | 11.29 | 13.25 |

**Point-to-point over the same queue pairs, the same plugin and the same cables is healthy right through the cliff** —
11 GB/s at 3.6 MB where the all-reduce manages 0.83. It is not the cable, the NIC, the MTU or the plugin's data path
as such; it is what happens when many connections are driven at once.

**Why you should care.** At `--max-num-seqs 8` a decode step's tensor-parallel all-reduce carries
64 tokens × 4096 × bf16 = **512 KB** and a step issues about 90 of them, so that message lands in the deepest part of
the hole; the prefill chunk (2,048–4,096 tokens, 16–32 MB) lands past it, which is why this is a decode fix first and
a prefill fix a distant second. The plugin's published benchmarks show the same shape on the author's triangle
(0.49 GB/s at 1 MB, 7.36 GB/s at 10 MB) `[reported]`, so it is the plugin, not our cabling.

## 2. Reproduce it on your own cluster

The whole diagnosis was made **with the engine down**, and so should yours be: a running engine holds the NICs and
its collectives pollute the counters. Stop the engine service on all three nodes, check with `docker ps` that nothing
still holds the fabric, then confirm the fabric is whole — this must print `4` on every node:

```
ibv_devinfo | grep -c PORT_ACTIVE
```

Put `bench/` on every node at the same path in the install directory. `bench/run-mesh.sh` runs one rank in the engine
image with the production plugin bind-mounted read-only; `bench/drive-mesh.sh` starts all three from the workstation,
reaching the nodes by SSH as `head`, `worker-1` and `worker-2` (edit its `HOSTS` array if yours differ). Then:

**1. Baseline** — all three operations, ~1.5× size steps from 8 KB to 64 MB. If the all-reduce column collapses while
`sendrecv` stays flat, you have this bug. Adding `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING` to any run
shows what NCCL decided — channel count, protocol, modelled bandwidth:

```
MESH_TAG=base MESH_OPS=allreduce,alltoall,sendrecv bash bench/drive-mesh.sh 3
```

**2. The channel matrix** — the same run once per configuration (`2, 4, 8, 12, 16, 32`, and unset for the default),
changing only the tag and the value:

```
MESH_TAG=ch8 MESH_ENV="NCCL_MAX_NCHANNELS=8" bash bench/drive-mesh.sh 3
```

**3. The sizes your engine actually uses**, including one decode step's worth of collectives (`MESH_STEP90_TOKENS`
sets the step message: 8 tokens is C1, 64 is C8):

```
MESH_TAG=sizes MESH_TOKENS=8,16,32,64,128,256,2048,4096 MESH_STEP90_TOKENS=64 bash bench/drive-mesh.sh 3
```

**4. Read the ConnectX-7 hardware counters.** `bench/mesh_sweep.py` samples them around every timed loop and reports
the per-collective delta; the raw files under `/sys/class/infiniband/*/ports/1/hw_counters/` are cumulative since
driver load, so only deltas mean anything. Which counter moves tells you which fault you have:

| counter | side | a rising value means |
|---|---|---|
| `rnr_nak_retry_err` | requester | your send reached a queue pair with no receive posted, and you then waited `min_rnr_timer` before retrying. **This is the bug** |
| `out_of_buffer` | responder | the same event from the other end: a send arrived and you had nothing posted |
| `packet_seq_err` | requester | out-of-order or dropped packets. If this moves you have a fabric or congestion problem, **not** this bug |
| `local_ack_timeout_err` | requester | an ACK never came back. If this moves, suspect the link or a stuck peer |

On our cluster `packet_seq_err` and `local_ack_timeout_err` are **zero at every size in every configuration**
`[measured-here]`: nothing is lost, nothing times out. The only anomaly is RNR, and it tracks the cliff exactly.

## 3. Root cause

Source read at commit `19924dcc`; a stock rebuild of it matches the production `libnccl-net.so` byte for byte in size
and in every section (401,368 B; text 118,159, data 2,208, bss 28,560), so what follows is what production runs
`[measured-here]`.

**The data path is two-sided, so RNR *is* the flow control.** `mesh_isend()` posts `IBV_WR_SEND`; `mesh_irecv()`
posts a matching `ibv_post_recv`. There is no RDMA WRITE, no remote key exchange, no receiver-advertised buffer FIFO —
NCCL's own IB plugin does the opposite (receiver publishes address and rkey, sender writes only into an advertised
slot), so there a send can never arrive early. With SEND/RECV there is no such interlock: a send landing on a queue
pair whose receive queue is momentarily empty gets an **RNR NAK**, and the requester stalls for `min_rnr_timer`.
`rnr_retry` is 7 (infinite), so this is a *designed*, non-fatal condition — flow control, not an error.

**The one-line bug**, in `mesh_connect_qp()` at the RTR transition:

```c
qp_attr.min_rnr_timer = 12;  // ~0.01ms min RNR NAK timer
```

The comment is right about the intent and wrong about the value. In the IBTA encoding **code 1 is 0.01 ms; code 12 is
0.64 ms** — a sensible pick for a switched fabric, where RNR fires only on real congestion. Here it is load-bearing:
every flow-control miss is quantised up to 0.64 ms.

**Two multipliers turn a rare miss into a systematic one.** First, **NCCL opens 64 channels on this fabric**: its
graph search values the modelled GPU-to-NIC path at 12 GB/s but each channel at only 0.24 GB/s, and compensates with
channel count (`Channel 00/64 … Channel 63/64`, one connection per channel, cycling the four NICs). That is 64 queue
pairs per direction per peer, and a **single proxy progress thread** services them round-robin, so the window in which
a send can outrun its receive is one proxy lap — and the lap grows with the channel count. Second, **NCCL drives the
chunk size to its 32 KiB floor**: a ring all-reduce shrinks `chunkSize` while `nBytes` is below
`nChannels * nranks * chunkSize * 8`, about 50 MB at 64 channels and 3 ranks, so everything smaller sits on the floor.
A 512 KB all-reduce becomes 683 KB of wire traffic over 64 channels — under 3 KB per send, hundreds of tiny sends,
each able to miss.

**That is why the curve has exactly this shape.** Below ~200 KB there are too few steps for the sender to get ahead of
the receiver: clean. From ~200 KB to ~12 MB is the bad regime — many tiny sends, a long proxy lap, one miss in every
few sends, 0.64 ms each. Above ~16 MB each channel's slice spans several chunks and the per-chunk wire time exceeds a
proxy lap, so the receiver is always armed again: clean. Per-collective counter deltas `[measured-here]`:

| message | µs | GB/s | `rnr_nak_retry_err` | `out_of_buffer` | `packet_seq_err` |
|---|---|---|---|---|---|
| 65 KB | 188 | 0.46 | 0.3 | 0.0 | 0 |
| 475 KB | 1,179 | 0.54 | 6.0 | 0.9 | 0 |
| **3.6 MB** | **5,749** | **0.83** | **28.5** | **23.0** | 0 |
| 18.2 MB | 2,146 | 11.28 | 2.4 | 0.0 | 0 |
| 64 MB | 7,270 | 12.31 | 7.7 | 0.0 | 0 |

At 3.6 MB, 28.5 requester-side NAKs at 0.64 ms apiece, spread over the channels running in parallel, account for the
whole 5.7 ms.

## 4. The cure we ship: `NCCL_MAX_NCHANNELS=8`

Environment only — no binary change, no rebuild, reversible by deleting one token. `NCCL_MAX_NCHANNELS=8` goes in
each node's own env file (`envs/env.tp3.example`), derived per node with `sed`, never copied between nodes.

Capping the channel count shortens the proxy's round-robin lap, so the receiver is usually already armed and the miss
mostly stops happening. Dose-response, median of three runs each `[measured-here]`:

| channels | 464 KB µs | 464 KB rnr+oob | 3.4 MB µs | 3.4 MB rnr+oob | 64 MB µs |
|---|---|---|---|---|---|
| 2 | 172 | 0.1 | 813 | 0.1 | 7,713 |
| 4 | 158 | 0.2 | 1,186 | 1.7 | 7,744 |
| **8** | **112** | **0.3** | **919** | **2.2** | **7,072** |
| 12 | 155 | 0.1 | 490 | 0.7 | 7,607 |
| 16 | 417 | 1.7 | 958 | 4.2 | 9,071 |
| 32 | 1,124 | 6.5 | 4,291 | 8.2 | 6,752 |
| 64 (default) | 1,105 | 7.9 | 5,143 | 48.1 | 8,275 |

Counters and time move together, monotonically, and the knee is between 12 and 16 channels. **Why 8 and not 2, 12 or
16.** At 16 and above the misses come back — the bug returning. At 2 and 4 the cliff is also gone, but the largest
messages lose parallelism they need, which shows in the 64 MB column and would cost prefill. 12 is indistinguishable
from 8 within noise model-free and slightly better at 3.4 MB, but has never been taken to the engine (§14 item 1); we ship 8
because it is the value measured end to end. `NCCL_MIN_NCHANNELS=8` alongside `MAX` measured the same as `MAX` alone
`[measured-here]`.

**Model-free gain at the sizes the engine uses**, median of four runs `[measured-here]`:

| what | message | stock (64 ch) | `NCCL_MAX_NCHANNELS=8` | gain |
|---|---|---|---|---|
| C1 decode all-reduce | 8 tok = 64 KB | 90 µs | 143 µs | latency-bound, noise ±100 µs |
| **C8 decode all-reduce** | **64 tok = 512 KB** | **1,195** | **123** | **9.7×** |
| prefill chunk (MNBT 2048) | 2,048 tok = 16 MB | 2,235 | 1,846 | 1.21× |
| prefill chunk (MNBT 4096) | 4,096 tok = 32 MB | 4,111 | 3,808 | 1.08× |
| **one C8 decode step** | 90 × 512 KB | **91.7 ms** | **9.9 ms** | **9.3×** |

**Engine gain**, three boots: a control at the default channel count, an arm B differing by exactly one environment
token, and a final independent boot with the setting in the env file. Aggregate output tok/s, realistic prompts,
median of sweep rounds 3–5 `[measured-here]`:

| | control (64 ch) | arm B (8 ch) | final boot (8 ch) | control → final |
|---|---|---|---|---|
| C1 | 47.97 | 51.16 | **51.38** | **+7.1 %** |
| C2 | 69.97 | 76.56 | **74.27** | **+6.1 %** |
| C4 | 98.17 | 109.79 | **107.60** | **+9.6 %** |
| C6 | 115.48 | 124.31 | **129.10** | **+11.8 %** |
| C8 | 133.44 | 150.27 | **150.83** | **+13.0 %** |
| gates, cold and after the benchmark | 10/10 · 12/12 | 10/10 · 12/12 | 10/10 · 12/12 | unchanged |

The A/B alone gives **C8 +12.6 %**; the final boot confirms it at +13.0 %, and all five of its rounds are tight
(C8: 151.7 / 150.1 / 152.4 / 150.8 / 147.0). A second, unlooked-for effect: **the control degrades across rounds and
the fix does not** — control C1 ran 50.5 → 49.4 → 48.0 → 41.6 → 50.1, against 51.2 → 49.9 → 51.2 `[measured-here]`.

**The honest shape of it — read this before quoting anything.** The microbenchmark sees 9.3× on a decode step's
collectives; the engine sees +12.6 % at C8. Both are real and do not conflict: the collective is one term in a decode
step, and removing almost all of a term worth 24 % of the step cannot give more than ~+30 % end to end. That 24 % was
profiled on an older arm; the measured +12.6 % implies the real share today is nearer 12 %. **Nothing justifies
quoting 9× as an end-to-end number.**

## 5. The timer patch: written, then carried

`patches/kernel/0004-min-rnr-timer.patch` — 26 added lines, 1 changed:

```diff
-        qp_attr.min_rnr_timer = 12;  // ~0.01ms min RNR NAK timer
+        qp_attr.min_rnr_timer = g_mesh_state.min_rnr_timer;  // IBTA code; 1 = 0.01ms
```

plus a `NCCL_MESH_MIN_RNR_TIMER` knob (default 1) parsed in `mesh_init()` beside the plugin's other tunables, and one
field in `struct mesh_plugin_state`.

**Tests, on the plugin's own suite** `[measured-here]`. `make test-unit` gives `test_routing` **13/13 pass**.
`test_error_paths` does not link in the stock tree either — the `Makefile` omits the ibverbs libraries from
`TEST_LDFLAGS`, so a clean checkout fails with `undefined reference to ibv_event_type_str`; pre-existing, not ours.
Linked by hand, stock and patched produce **identical** output (60 passed, 6 failed, the 6 pre-existing string checks).

**Control experiment.** The patched build forced back to `NCCL_MESH_MIN_RNR_TIMER=12` reproduces the stock cliff
exactly — 3.4 MB: 4,695 µs against stock 5,143; 704 KB: 1,539 against 1,456; RNR 21 against 24 `[measured-here]`, so
the patch changes nothing but the timer. **On its own, at the stock 64 channels, it is worth about 2.2×** in the cliff
(3.4 MB 5,143 → 2,433 µs; 704 KB 1,456 → 1,078) — real, but not the ~10× the cap gives: a shorter timer makes each
miss ~64× cheaper, while capping the channels stops the misses happening at all.

**Why we did not deploy it on its own.** With `NCCL_MAX_NCHANNELS=8` in place its model-free contribution is inside
the noise, so it never earned an engine boot by itself. It is now **in production anyway**, carried into the same
patched build as 0005 and 0006 (§6–§8) and left at its default `NCCL_MESH_MIN_RNR_TIMER=1`, because once you are
running a patched plugin there is no reason to keep the worse value. Its isolated engine contribution is still
unmeasured `[not tested]`. Patch and finding went upstream — see [../CREDITS.md](../CREDITS.md).

---

## 6. The second cable: half the fabric had never carried a packet

**Settings for §6–§8**, which differ from the block at the top of this page: image `exl3-zeus:9bf594c`,
`--max-num-batched-tokens 2048`, `HAREM_SW_BLOCK_SIZE=256`, per-rank fast-load sidecar, persisted MLA tuner cache
([12](12-tuner-cache.md)) and therefore **three sweep rounds, median of three**; plugin built from
`autoscriptlabs/nccl-mesh-plugin` at `19924dcc` plus patches 0004, 0005 and 0006. Everything else as above.
Model-free arms are two repetitions each. 5 September 2026.

### 6.1 The finding

Each node has four ConnectX-7 ports and the triangle is wired with **two cables between every pair**. The plugin's
`mesh_connect()` opens with `(void)dev;` — it throws away the device index NCCL hands it — and then walks the peer's
advertised address list, stopping at the **first** address a local NIC can reach. That is deterministic, so every
channel to a given peer lands on the same cable, every time, for every peer.

We had been reading a 13 GB/s ceiling as "the link", against a nominal 25 GB/s. It was not the link. It was one
cable of a pair whose combined **wire** capacity is 50 GB/s per direction.

> **Read §9 before using that 50 GB/s anywhere** `[retracted]`. The wire is not the ceiling on this hardware: each
> card sits on PCIe Gen5 x4 and carries ~15 GB/s no matter what its ports advertise, so the real ceiling is ~30 GB/s
> per node. The finding in this section — one cable of each pair idle since driver load — stands exactly as measured;
> only the roof it is compared against changed, and it changed again the same day.

The proof is in hardware, and it is unambiguous. `port_xmit_data` is cumulative since driver load; read on all three
nodes while the cluster was busy `[measured-here]`:

```
head      roceP2p1s0f0:0  roceP2p1s0f1:0  rocep1s0f0:2968259552956  rocep1s0f1:19338488069
worker-1  roceP2p1s0f0:0  roceP2p1s0f1:0  rocep1s0f0:2615320151240  rocep1s0f1:370647326623
worker-2  roceP2p1s0f0:0  roceP2p1s0f1:0  rocep1s0f0:2613834402438  rocep1s0f1:18498564601
```

The two `roceP2p1s0*` ports on every node — the second cable of each pair — had transmitted **zero**. Not "little".
Zero, since the driver loaded, on all three nodes.

**Go and read your own counters before you trust any ceiling you have measured.** One line per node:

```
for p in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do echo "$p $(cat $p)"; done
```

### 6.2 Check the cable is really there first

Configuration plus link state is not a delivered packet. Before the patch, the fabric addresses on those ports had
**no ARP neighbour at all** on any node, and the netdevs had moved only multicast and neighbour-discovery traffic.
Something was on the other end; nothing proved it was the node the subnet claimed.

So: ping across each second cable, from each end, before enabling anything. Six pings on a triangle, each bound to
the interface under test, and each node must end with four fabric neighbours resolved. If one does not answer, the
wiring is not what the subnets imply — keep `NCCL_MESH_LINKS_PER_PEER=1` (which is exactly today's behaviour) and
take it to whoever cabled the rack. `bench/mesh-multilink-sweep.sh` refuses to run in that state when you give it a
`FABRIC_PREFIX`; the failure lands in a model-free container in under a minute rather than in an engine boot.

### 6.3 The patch

`patches/kernel/0005-device-aware-link-selection.patch`, ~30 lines of substance against `19924dcc`.
`mesh_connect()` now enumerates every parallel link to the peer — one candidate is one (local NIC, peer address)
pair sharing a subnet, i.e. one cable — and uses NCCL's device index to choose among them:

```c
pick = dev % usable;   /* usable = min(candidates found, NCCL_MESH_LINKS_PER_PEER) */
```

Candidates are collected in the peer's advertised order and, within one peer address, in local NIC order — precisely
the order the old first-match search walked. So `cands[0]` is the NIC the unpatched plugin would have returned, and
with one cable per peer, or with `NCCL_MESH_LINKS_PER_PEER=1`, the selection is **bit-identical to the pre-patch
plugin**. No wire-format change: `mesh_handle`, `mesh_addr_entry` and `mesh_qp_info` are untouched, so a patched node
and a stock node still hand-shake.

Model-free, 8 channels, two repetitions `[measured-here]`:

| | base (unpatched) | control (patched, `LINKS_PER_PEER=1`) | **`LINKS_PER_PEER=0`** |
|---|---|---|---|
| all-reduce 64 MB | 12.08 GB/s | 12.38 | **16.66** |
| all-reduce 16 MB | 10.93 | 10.55 | **13.89** |
| send/recv 64 MB | 11.16 | 11.10 | **16.01** |
| one C8 decode step (90 × 512 KB) | 10.21 ms | 10.05 | **9.27** |
| second cable, `port_xmit_data` delta | **0** | **0** | **4.19e9 · 0.45e9**, matching the first |

The control arm is the load-bearing one: the patched binary told to behave like the old one lands on top of `base` at
every size, so the gain belongs to the second cable and not to the rebuild.

## 7. `NCCL_PTR_CUDA`: two lines that remove a host bounce buffer

### 7.1 Why "GPUDirect" is the wrong word on this part

There is no separate framebuffer here. `nvidia-smi -q` reports `FB Memory Usage: N/A`, `BAR1 Memory Usage: N/A`,
`Addressing Mode: ATS`, `GPU C2C Mode: Enabled`, and one NUMA node holding all of it `[measured-here]`. Classic
GPUDirect RDMA — the NIC DMA-ing through a BAR1 window into GPU HBM, which is what `nvidia-peermem` exists to
arrange — is meaningless when there is no window and no separate memory behind it. `nvidia-peermem.ko` is on disk on
all three nodes and is **not loaded**, and it does not need to be.

What is still real is this: with `ptrSupport = NCCL_PTR_HOST`, NCCL allocates its transfer buffers with
`cudaHostAlloc` and the collective kernel writes the data there before the NIC reads it. On GB10 both buffers live in
the same LPDDR5X, so that is not a PCIe hop — it is an extra memory-to-memory copy, a proxy synchronisation point, and
the GPU's penalty for writing host-mapped memory. Removing it is the entire prize, and it is worth having.

### 7.2 The patch

`patches/kernel/0006-ptr-cuda-dmabuf-and-flush.patch`. The registration machinery was **already written**: the
plugin's `regMr` registers a CUDA pointer with plain `ibv_reg_mr`, with a source comment explaining that unified
memory on this architecture makes that legal. NCCL simply never handed it a device pointer, because `getProperties`
advertised host-only support. Two lines say otherwise:

```c
props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;   /* in both the v8 and the v9 getProperties */
```

The rest of the patch is what honesty then requires: `mesh_iflush()` was a no-op, and with device pointers in play it
has to become a real RDMA_READ so the receiver's memory is coherent before NCCL reads it; and `regMrDmaBuf` gained a
real `ibv_reg_dmabuf_mr` path instead of discarding the fd and falling back. Everything is behind
`NCCL_MESH_PTR_CUDA`, `NCCL_MESH_FLUSH` and `NCCL_MESH_DMABUF`, all defaulting to today's behaviour.

Model-free, on top of both cables `[measured-here]`:

| | `link2` | **`link2` + `PTR_CUDA` + flush** |
|---|---|---|
| all-reduce 64 MB | 16.66 GB/s | **20.84** |
| all-reduce 8 MB | 10.30 | **17.81** |
| send/recv 64 MB | 16.01 | **21.30** |
| all-to-all 64 MB | 11.01 | **20.68** |
| one C8 decode step | 9.27 ms | **9.24** |

`GPU Direct RDMA` flips from `Disabled` to `Enabled` for all four HCAs in the NCCL debug log, which is the
confirmation that the path was taken and not merely advertised.

**What the flush costs:** `NCCL_MESH_FLUSH=0` measures inside noise of flush-on, and better at a couple of sizes. We
keep the flush on. Correctness is not a noise-level decision, and the measured price is inside the run-to-run spread
`[measured-here]`.

**DMA-BUF does not pay here.** `NCCL_MESH_DMABUF=1` works — which answers the open question of whether
`ibv_reg_dmabuf_mr` accepts these buffers on this platform — but it is **slower** than plain `ibv_reg_mr` across the
range (64 MB all-reduce 18.08 against 20.84). On a part where one physical memory is shared and `ibv_reg_mr` takes a
device pointer directly, the DMA-BUF path buys nothing and costs registration work. Not adopted `[measured-here]`.

## 8. Sixteen channels is harmful, and the engine result

### 8.1 The 16-channel trap

`NCCL_MAX_NCHANNELS=8` caps *channels*, not cables. With patch 0005 those 8 channels split 4 + 4 over the two cables
of a pair, so each cable carries half the queue pairs the tuned configuration used to give it. Sixteen channels would
restore 8 per cable. It is a reasonable thing to expect, and it is wrong `[measured-here]`:

| arm | one C8 decode step (90 × 512 KB) | all-reduce 64 MB |
|---|---|---|
| `link2`, 8 channels | **9.27 ms** | 16.66 GB/s |
| `link2`, 16 channels | **26.18 ms** | 20.56 |
| `link2` + `PTR_CUDA`, 8 channels | **9.24 ms** | 20.84 |
| `link2` + `PTR_CUDA`, 16 channels | **26.51 ms** | 16.41 |

**2.5× slower on the message the engine actually decodes with.** The RNR mechanism of §3 does not care how many
cables the channels are spread over — it cares how long the single proxy thread's round-robin lap is, and 16 channels
makes it long again. Sixteen does win the large-message columns, which is exactly the trade the cap was chosen
against: this engine's decode all-reduce is 512 KB, not 64 MB. **Keep 8.**

### 8.2 The engine

One boot per arm, three sweep rounds each, medians of three, gates cold and after the benchmark
`[measured-here]`:

| | before (tuner cache warm) | **after (both patches)** | change |
|---|---|---|---|
| C1 aggregate | 54.5 | **56.9** | +4.4 % |
| C2 | 80.8 | **84.2** | +4.2 % |
| C4 | 112.0 | **118.5** | +5.8 % |
| C6 | 135.6 | **142.9** | +5.4 % |
| C8 | 159.9 | **168.9** | +5.6 % |
| TTFT, C1 / C8 | 0.47 / 1.03 s | **0.41 / 1.01 s** | −13 % / −2 % |
| prefill-fresh, 3 unseen ~8.3K prompts | 1,662 / 1,709 / 1,719 | **1,742 / 1,792 / 1,797** | +5 % |
| draft acceptance | 60–64 % | 61–65 % | unchanged |
| KV pool | 4,429,752 | **4,449,035** | +0.4 %, inside noise |
| free RAM / swap, worst node | 11.4 G / 0.11 G | 11.3 G / 0.11 G | unchanged |
| gates, cold and warm | 10/10 · 12/12 | **10/10 · 12/12** | unchanged |

**What it cost.** We looked in the same places as in §13 and found the same answer: nothing measurable. The KV pool
was the one to watch — advertising `NCCL_PTR_CUDA` moves NCCL's transfer buffers from host allocations into device
memory, which is accounted under `gpu-memory-utilization` — and the pool moved +0.4 %, inside boot-to-boot noise. The
real cost is elsewhere and it is not a number: **you are now running a patched plugin.** That is a build to maintain,
a divergence from upstream to track, and a `.so` that has to be rebuilt when the plugin moves. Both patches are
offered upstream (§10.4, §14, [../CREDITS.md](../CREDITS.md)).

### 8.3 How to run it

The plugin directory is one line in each node's environment file; nothing else changes, and the production plugin
directory is never written:

```
NCCL_MESH_PLUGIN_DIR=$HOME/exl3-zeus/nccl-mesh-patched2
```

```
EXTRA_ENV="... NCCL_MAX_NCHANNELS=8 NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_MIN_RNR_TIMER=1 NCCL_MESH_PTR_CUDA=1 NCCL_MESH_FLUSH=1"
```

Roll back by restoring the previous env file — the unpatched plugin is still where it was. To reproduce the whole
model-free matrix, nine arms with the control and the port-counter proof:

```
bash bench/mesh-multilink-sweep.sh 2
```

Full tables: [`../results/mesh/multilink-sweep.md`](../results/mesh/multilink-sweep.md).

**Upstream.** Patches 0004, 0005 and 0006 are carried in a public fork of the plugin,
[`NNNtrance/nccl-mesh-plugin`](https://github.com/NNNtrance/nccl-mesh-plugin), branch
`gb10-dual-link-ptrcuda` on top of `19924dcc`, and offered as
[pull request #59](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59) against the plugin,
referencing the issue thread the findings were first reported on. Patch 0007 (§10) is deliberately
**not** in that branch. If you want the patched plugin and would rather not apply three patch files,
build that branch.

---

## 9. The ceiling is PCIe, not the cable

**This section retracts a number this page published yesterday** `[retracted]`. §6 said the pair of
cables between two nodes is worth **50 GB/s** and that the collective was therefore running at about
28 % of the fabric. The cable arithmetic was right and the ceiling was wrong, because it counted the
wire and not the slot the card sits in.

Two commands settle it, and both should be run before anyone quotes a fabric roofline
`[measured-here]`:

```
lspci -vv | grep -A2 ConnectX
```

```
ibv_devinfo | grep -E "active_width|active_speed|state"
```

Each ConnectX-7 reports `LnkSta: Speed 32GT/s, Width x4` — PCIe Gen5 by four lanes — and each of its
two ports reports `4X` at `50.0 Gbps` per lane, which is the 200 Gb/s the port advertises.

| | arithmetic | result |
|---|---|---|
| wire, per card (2 ports × 200 Gb/s) | 2 × 25 GB/s | 50 GB/s |
| **PCIe Gen5 x4, per card** | 32 GT/s × 4 lanes ÷ 8 × 128/130 | **≈ 15.75 GB/s raw, ~14.5–15 GB/s usable** |
| **per node** (two cards) | 2 × ~15 | **≈ 30 GB/s total** |

A card's two ports can offer three times more traffic than the card's own slot can carry. The wire
was never the binding constraint on this hardware.

**What that re-explains, without changing a single measurement:**

- The **13 GB/s** we spent a day reasoning against was not "half a link" and not "a 25 GB/s link at
  half efficiency". It was **one card's PCIe limit**, at about 87 % of it — a perfectly healthy
  number that looked like a defect only because it was being compared against the wrong roof.
- The second cable of each pair lives on the **second card**, so patch 0005 did not open a second
  cable so much as a **second PCIe path**. That is why the ceiling moved to ~20 GB/s and stopped
  there rather than approaching 50.
- The remaining fabric headroom is therefore at most about **30 %** (≈20 GB/s of bus bandwidth
  against a ~30 GB/s per-node ceiling), not the 2.5× the 50 GB/s figure implied. With the collective
  at 16.5 % of a prefill chunk ([10](10-results-and-roofline.md) §5), perfect use of the remaining
  headroom is worth **2–4 % of prefill** — not the 12–17 % an earlier estimate of ours carried
  `[estimate]`.
- And it predicts the result in §10 before §10 was run: a transport change that removes flow-control
  stalls cannot move a number that is bounded by a bus.

The lesson is the one this repository keeps re-learning in different clothes: **measure the ruler,
not the brochure.** The same day, the memory roofline moved from a vendor 273 GB/s to a measured
225 GB/s ([10](10-results-and-roofline.md) §4), and every efficiency percentage written before that
was ~22 % optimistic. A fabric ceiling assembled from port speeds is exactly the same mistake.

---

## 10. One-sided RDMA_WRITE (patch 0007): built, measured, not adopted

The plugin's data path is **two-sided**: `mesh_isend` posts an `IBV_WR_SEND`, `mesh_irecv` posts an
`ibv_post_recv`, and there is not one `IBV_WR_RDMA_WRITE` in the source. Its only flow control is the
RNR NAK — if the sender posts before the receiver has re-armed, the hardware makes the sender wait.
NCCL's own IB transport never has that problem, because it is one-sided: the receiver advertises the
address and rkey of its destination buffer into a FIFO slot in the sender's memory, and the sender
writes only into an advertised slot.

§14 of this page used to call that "the real fix" and price it at 650–850 lines of somebody else's
work. We wrote it instead: `patches/kernel/0007-one-sided-fifo-rdma-write.patch`, +977/−16 lines over
0004–0006.

### 10.1 The design, in one paragraph

The FIFO lives in the **sender's** host memory, one per comm — which after 0005 means **one per
cable** — as 32-byte slots (`addr`, `rkey`, `size`, `tag`, `seq_head`, `seq`), small enough to be
advertised `IBV_SEND_INLINE`. The receiver posts a **zero-byte** `ibv_post_recv` first and advertises
the slot second, so by the time the sender's `RDMA_WRITE_WITH_IMM` arrives the receive is always
armed and **RNR becomes structurally impossible**. `seq` is written last and read with acquire
ordering; `NCCL_MESH_FIFO_STRICT=1` re-checks a duplicate of it at another offset, so a torn slot
cannot be mistaken for a ready one. A ring overrun and an oversized send are both `MESH_WARN` +
`ncclInternalError` rather than a silent write into the wrong memory. The path does **not** rely on
NCCL's "you may return `*request = NULL`" contract: `mesh_isend` always returns a request and queues
it if no slot is ready. Mixed versions fail closed — `mesh_qp_info` stays 32 bytes, its two reserved
bytes become a magic and a wire version, and a `write`-mode node refuses a peer that does not speak
the extension rather than writing into an address it guessed. Everything is behind
`NCCL_MESH_TRANSPORT=send|write`, default `send`, which is byte-for-byte the old path.

### 10.2 Model-free: the mechanism works and the bandwidth does not move

Six arms, two repetitions each, engine down, `NCCL_MAX_NCHANNELS=8`, `ALGO=Ring`, all four ports in
use. Means of the two repetitions; all-reduce GB/s (NCCL bus bandwidth) with RNR retries per
operation beside it `[measured-here]`:

| arm | 1 MB | 16 MB | 64 MB | RNR/op | out-of-buffer/op | one C8 decode step (90 × 512 KB) |
|---|---|---|---|---|---|---|
| `p2best` — production, two-sided (**control**) | 5.5 | 17.8 | 20.1 | 1.1–9.2 | 0.3–7.7 | 9.12 ms |
| `p3send` — 0007 binary, `TRANSPORT=send` | 5.2 | 19.0 | 19.3 | 0.2–11.6 | 0.1–4.7 | 9.14 ms |
| `p3w64` — write, FIFO depth 64 | 6.4 | 21.2 | 22.1 | **0.0** | **0.0** | 9.13 ms |
| `p3w128` — write, depth 128 (default) | 6.7 | 16.9 | 17.4 | **0.0** | **0.0** | 9.14 ms |
| `p3w256` — write, depth 256 | 4.7 | 15.4 | 16.2 | **0.0** | **0.0** | 9.08 ms |
| `p3w128nf` — write, `FLUSH=0` | 6.3 | 16.7 | 20.1 | **0.0** | **0.0** | 9.25 ms |

Two readings, and they point opposite ways.

**The mechanism does exactly what it was designed to do.** Every write arm reports **zero** RNR
retries and **zero** out-of-buffer events, at every size, in both repetitions — against 1–9 per
operation on the control. That is not a bandwidth argument, it is a correctness-of-design argument,
and it is the one thing in this experiment that is unambiguous. The `p3send` arm also lands on the
control at every size, which is the gate that says the patch did not break the old path.

**And the bandwidth does not care.** The pre-registered gate was **≥ 1.3× the control at ≥ 16 MB**.
The default-depth arm is *below* control at 16 and 64 MB; the depth-64 arm is above it; the spread
between two repetitions of the *same* arm reaches 30 %. On the message the engine actually decodes
with, all six arms sit within 0.17 ms of each other — under 2 %. The gate was not met, and the
honest description of the result is **no effect on throughput, at any depth, with or without the
flush**.

§9 says why, and said it before the arm was run: at ~20 GB/s of bus bandwidth the transfer is
against a **PCIe** wall, not against a flow-control stall. Removing RNR from a path that was not
waiting on RNR buys nothing.

Full tables: [`../results/mesh/rdma-write-sweep.md`](../results/mesh/rdma-write-sweep.md).

### 10.3 The engine arm, and the decision

One boot on the write transport (FIFO depth 128, flush on), three sweep rounds, gates cold and warm,
against production configuration 6 `[measured-here]`:

| | production 6 (two-sided) | 0007, `TRANSPORT=write` |
|---|---|---|
| C1 aggregate | **56.9** | 56.4 |
| C8 aggregate | **168.9** | 171.1 |
| prefill-fresh, 3 unseen ~8.3K prompts (median) | **1,792** | 1,763 |
| prefill, warm 7K prompt | 1,506 | 1,457 |
| draft acceptance | 61–65 % | 60–65 % |
| KV pool | 4,449,035 | 4,462,809 |
| gates, cold and warm | 10/10 · 12/12 | 10/10 · 12/12 |
| free RAM, worst node / swap | 11.3 G / 0.11 G | 11.1 G / 0.11 G |

The differences run in **both directions** and every one of them is inside this stack's boot-to-boot
spread ([09](09-measurement-protocol.md) §2). **Not adopted.** Production stays on the two-sided
path with `nccl-mesh-patched2`.

**What it cost, and what it did not.** The engine arm cost one boot; the sweep cost fifteen minutes;
the patch cost a day of somebody's design attention. It cost the stack **nothing** — the production
plugin binary, `.env.tp3` and the NVFP4 stack's own copy of the plugin were never written, and the
rollback was never needed because the arm was run from a separate environment file against a separate
plugin directory. What we did **not** get is the thing it was built for.

**It is kept, not deleted.** The patch is in `patches/kernel/`, it compiles warning-free under
`-Wall -Wextra`, its unit tests match the 0004–0006 baseline exactly, and a dry `dlopen` on all three
nodes resolves every NCCL entry point without calling `init()`. If a future part puts a wider slot
under these cards — or if somebody runs this plugin on hardware where RNR really is the binding
constraint — it is written and measured. On this hardware it is an option, not a recommendation.

### 10.4 Upstream

Patch 0007 is **not** offered upstream and is not in the fork's PR branch. Reporting a transport
rewrite as a win when our own measurement says it changes nothing here would waste the maintainer's
time; if it goes upstream one day it should go with a machine where it pays. The three patches that
*did* pay are in [PR #59](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59) (§8.3).

---

## 11. What is NOT the cause

| hypothesis | test | result |
|---|---|---|
| link, MTU, driver | point-to-point over the same queue pairs | 13.3 GB/s, clean through the cliff `[measured-here]` |
| packet loss, congestion | `packet_seq_err`, `local_ack_timeout_err` | 0 everywhere, every size, every arm `[measured-here]` |
| protocol choice (LL / LL128 / Simple) | `NCCL_PROTO` forced | no protocol fixes it — §12 `[measured-here]` |
| NCCL buffer too small | `NCCL_BUFFSIZE=8388608` | no change; the 32 KiB chunk floor is absolute, not a fraction of the buffer `[measured-here]` |
| per-peer channel sharing | `NCCL_NCHANNELS_PER_NET_PEER=1` | no change `[measured-here]` |
| missing GPUDirect RDMA (`ptrSupport = NCCL_PTR_HOST`) | debug log: GPU Direct RDMA disabled on all four HCAs | real, and it holds the ceiling down — but it is size-independent and cannot produce a cliff `[measured-here]`. **Since fixed: §7.** The "~13 GB/s against a 25 GB/s link" framing in an earlier version of this row was itself wrong twice over: the pair is two cables and one of them was idle (§6), and the number itself is a **PCIe** limit, not a link limit (§9) |

**Retraction** `[retracted]`. An earlier report of ours read the profiler kernel name
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` as "NCCL is using LL even at 16 MB, so half the link is on the table", and
proposed forcing `Simple` as the largest prefill lead open. Forcing LL at 16 MB costs **20,114 µs against auto's
1,787 µs**, an 11× difference, so auto is plainly not using LL there `[measured-here]`; and 11–12.5 GB/s is what this
mesh delivers at that size anyway, since `Simple` — no flag words at all — reaches 12.2 GB/s. There is no half link to
recover. What replaced it is the finding on this page: the loss is mid-range, and the mechanism is RNR.

## 12. `NCCL_PROTO` and `NCCL_ALGO`

### 12.1 `NCCL_PROTO` — rejected

You will reach for this first; it does not work. Model-free, engine down, `bench/ar_bench.py` with the engine's exact
NCCL environment; every protocol measured twice, `ALGO=Ring` unless noted; µs per all-reduce, world = 3
`[measured-here]`:

| message | auto (×2) | LL | Simple (×2) | LL128 | `^LL` |
|---|---|---|---|---|---|
| 64 KB (C1 decode) | **61.3 / 61.5** | 69.7 | 172.9 / 175.8 | 70.9 | 70.5 |
| 512 KB (C8 decode) | **983.3 / 1,038.6** | 2,048.7 | 4,372.8 / 4,488.0 | 1,014.6 | 1,097.7 |
| 4 MB | 5,287.3 / 5,318.1 | 5,528.0 | 5,330.4 / 5,508.4 | **1,702.4** | 5,308.0 |
| 16 MB (prefill chunk) | **1,787.3 / 1,792.2** | 20,113.6 | 2,030.7 / 1,829.8 | 3,758.3 | 1,912.1 |
| one decode step (90 × 8 tok) | **3.84 / 3.96 ms** | 3.97 | 9.97 / 10.14 ms | 4.77 / 4.91 | 4.77 |

`Simple` is a loss at every size the engine uses — 2.8× worse at the C1 decode message, 4.4× at the C8 one, no better
than auto at 16 MB. `LL128` is 3.1× better than auto at 4 MB but 2.1× worse at 16 MB and worse at decode; `^LL` picks
`Simple` at 4 MB and is also worse at decode. **Leave `NCCL_PROTO` unset.** `NCCL_ALGO` unset measures the same as the
forced `Ring` (3.90 against 3.96 ms per decode step), so the launcher's `Ring` stays. One trap: an old env file
carrying `NCCL_MAX_NCHANNELS=8` *alongside* `NCCL_PROTO=LL` is not evidence that 8 channels was tried and rejected —
the `LL` in it costs 11× at 16 MB and buries the channel gain. That combination was never tried cleanly.

### 12.2 `NCCL_ALGO`: `Tree` is dead here, `Ring,Tree` is unresolved

The launcher **forces** `NCCL_ALGO=Ring`, which for a long time was an inherited default and not a measured choice —
it sat on the open list as the cheapest untried thing in this repository. It has now been swept, model-free, engine
down, on the production plugin build and the production NCCL environment (`NCCL_MAX_NCHANNELS=8`,
`NCCL_MESH_LINKS_PER_PEER=0`, `NCCL_MESH_MIN_RNR_TIMER=1`, `NCCL_MESH_PTR_CUDA=1`, `NCCL_MESH_FLUSH=1`), three arms
× **two repetitions**, world = 3 `[measured-here]`.

**All-reduce**, GB/s, both repetitions shown because the answer is in the disagreement between them:

| message | `Ring` | `Ring,Tree` | `Tree` |
|---|---|---|---|
| 1 MB | 4.49 / 2.61 | 2.66 / 6.90 | 2.43 / 3.46 |
| 4 MB | 9.61 / 15.46 | 16.70 / 17.24 | 1.74 / 1.91 |
| 16 MB (prefill chunk) | 20.23 / 18.30 | 20.39 / 21.06 | **3.15 / 3.36** |
| 64 MB | 22.24 / 18.23 | 22.50 / 22.58 | **4.93 / 5.93** |

**The decode-step proxy** — 90 collectives at the decode message size, which is the shape that actually decides this
question, since a decode step spends its collective time on a fixed count of 102 small messages
([10](10-results-and-roofline.md) §5.3):

| | rep 1 | rep 2 | mean |
|---|---|---|---|
| `Ring` | 9.165 ms | 9.669 ms | **9.42** |
| `Ring,Tree` | 9.118 ms | 9.045 ms | **9.08** |
| `Tree` | 11.596 ms | 18.442 ms | **11.60 (rep 1) / 15.02 (both)** |

**`Tree` is dead on this fabric.** It is 4–6× slower on the two message sizes the engine actually uses in bulk, its
RNR retry counters climb to 37–41 per operation at 64 MB where Ring sits at 0–13, and it is 23–96 % worse on the step
proxy. The mechanism is not mysterious: a tree wants a bandwidth-shaped topology to pay for its lower step count, and
this is three nodes, each a pair of PCIe Gen5 x4 cards (§9). The step-count arithmetic that made it attractive —
`2(w−1) = 4` ring steps against `~2·log₂3 ≈ 3.2` — was never worth what it costs per step here.

**`Ring,Tree` is not decided, and this sweep cannot decide it.** It is 3.6 % better than Ring on the step proxy and
better at 4 MB; it is worse than Ring on `sendrecv` at 64 MB (17.27 / 15.43 against 21.62 / 20.79) and the two arms
swap places at 1 MB. The reason none of that is conclusive is visible in the table: **two repetitions of the same arm
differ by up to 1.7× at 1 MB and by 22 % at 64 MB**, so an effect of a few percent is well under the instrument's own
spread. Giving the tuner the list rather than pinning `Tree` is still the right shape for the arm — but settling it
needs a five-round engine measurement ([09](09-measurement-protocol.md) §1), not another model-free sweep, and at an
expected −1…3 % of a decode step that arm has not earned its slot yet. **Deferred; `Ring` stays in production**
`[not tested]` for the engine half.

Raw logs, port counters and the JSON per arm: [`../results/mesh/algo-sweep.md`](../results/mesh/algo-sweep.md).

## 13. What this cost

A rare thing in this stack: a gain we looked hard for a price on and did not find one `[measured-here]`.

| axis | control (64 ch) | 8 ch | verdict |
|---|---|---|---|
| Quality gates, cold and after the benchmark | 10/10 · 12/12 | 10/10 · 12/12 | unchanged in every arm and every boot |
| Prefill, 7K warm | 1,471 tok/s | 1,483 | +0.8 %, inside noise |
| Prefill, fresh unseen prompts | 1,689 | 1,720–1,728 | +1.8 to +2.3 %, inside noise |
| Draft acceptance rate | 60–66 % | 60–65 % | unchanged |
| GPU KV cache | 1,639,427 tokens | 1,633,299 / 1,648,621 | ±0.4 %, boot-to-boot noise |
| Free RAM / swap, worst node | 8.3 / 1.3 GiB | 14.3 / 3.4 GiB | both inside the rule (≥ 4 GiB free) |

Those KV numbers belong to that image and those settings, not to the headline configuration on the front page; what
matters is the difference between arms, which is boot noise. Nothing is allocated differently — fewer channels means
fewer queue pairs, which are not on the budget that binds this stack. The one thing the cap takes away is parallelism
for very large messages: at 2 and 4 channels the 64 MB column suffers, at 8 it does not (7,072 against 8,275 µs).

## 14. What is left open

1. **12 channels has never been taken to the engine** `[not tested]`. Model-free it was indistinguishable from 8 on a
   single cable and slightly better at 3.4 MB. It has **not** been re-measured over two cables, where the channel
   arithmetic changed — and 16 is now known to be badly wrong there (§8.1), which makes 12 worth one careful arm
   rather than an assumption.
2. **The `min_rnr_timer` patch has never been isolated on the engine** `[not tested]`. It rides in the production
   plugin build with 0005 and 0006, at `NCCL_MESH_MIN_RNR_TIMER=1`; what it is worth by itself, at 8 channels over
   two cables, is unmeasured.
3. **RNR in the engine has not been re-read since the patches.** On the single-cable 8-channel engine a live counter
   read across one full sweep plus prefill plus mixed load showed **~42,000 `rnr_nak_retry_err` and 24,000–42,000
   `out_of_buffer` per node over about five minutes** — 1–3 % of wall clock in back-off `[measured-here]`. The
   model-free sweep says retries per operation fell from ~15 to ~3 with `PTR_CUDA` and to **exactly 0** on the
   one-sided transport (§10.2) `[measured-here]`, but the engine-side counter read was not repeated `[not tested]`.
   §10 makes this item much less interesting than it looked: zero retries bought zero throughput.
4. **The receiver-advertised FIFO — closed, and it was not the fix** `[measured-here]`. This item used to say the
   one-sided path was "the real fix" and price it at 650–850 lines of the plugin author's time. We wrote it
   (`patches/kernel/0007`, +977/−16), it works exactly as designed — RNR and out-of-buffer counters go to **zero** —
   and it changes throughput by nothing on this hardware, because the binding constraint is the cards' PCIe slots
   (§9), not the flow control. Not adopted; kept as an option; not offered upstream. Full account in §10.
5. **The collective's share of a step — closed** `[measured-here]`. Re-derived without a profiling boot (the running
   engine had no profiler endpoint) by re-segmenting the earlier trace per chunk, re-measuring the two classes that
   changed model-free, and checking the total against live wall-clock: the all-reduce is **16.5 % of a prefill chunk**
   and **10–15 % of a C1 decode step**, down from 20.1 % and 17.5 %. The residual on that reconciliation is 2.8 % and
   the most likely owner of it is NCCL, so read the prefill share as a 14–17 % band. See
   [10](10-results-and-roofline.md) §5. What is still `[not tested]` is the C8 decode split, which does need a
   profiling boot.
6. **DMA-BUF registration is measured and rejected, not understood** `[measured-here]`. It works and it is slower;
   we did not investigate why beyond the reasoning in §7.2.
7. **The ~30 % of fabric headroom that is left** `[not tested]`. §9 puts it at ~20 GB/s of bus bandwidth against a
   ~30 GB/s per-node PCIe ceiling. Nothing in this repository has tried to close it, and at 2–4 % of prefill nothing
   should before the two larger levers ([11](11-open-issues.md) §2.12, §2.1) are spent. If you want to try: the
   remaining candidates are channel count over two cables (item 1), and getting NCCL to keep both cards busy on the
   *same* collective rather than alternating channels between them, which nobody here has looked at.
8. **`NCCL_ALGO` — half closed** `[measured-here]`. The model-free sweep has been run (§12.2): **`Tree` is dead
   here**, 4–6× slower at 16 and 64 MB and 23–96 % worse on the decode-step proxy, so the step-count arithmetic that
   made it attractive is decisively outweighed on this fabric. **`Ring,Tree` is unresolved**: 3.6 % better on the
   step proxy, worse on `sendrecv` at 64 MB, and the sweep's own repeat-to-repeat spread is larger than the effect.
   Settling it needs a five-round engine arm at an expected −1…3 % of a decode step, and it has been **deferred**
   rather than run `[not tested]`. `Ring` stays in the launcher.

## 15. The same fix applies to the NVFP4 sibling

Our NVFP4 recipe runs the **same plugin binary at the same commit** over the **same fabric** at the **same TP=3**,
with the same 4096-wide hidden state and so the same 512 KB decode all-reduce. Same cliff, same fix: one environment
line per node, no rebuild, reversible by restoring the env file:

```
EXTRA_ENV="NCCL_MAX_NCHANNELS=8"
```

Gate it the usual way — correctness probe 10/10, code exam 12/12, then one sweep round — and expect the gain on
**C4–C8 decode**, not on C1 and not on prefill. The change has been measured on the EXL3 stack only; on the NVFP4
stack it is a recommendation, not a result `[not tested]`.

The §6–§7 patches transfer in the same way and are a bigger prize there, because the idle second cable is a property
of the fabric and the plugin, not of the quantization: same wiring, same plugin commit, same `mesh_connect()`. The
NVFP4 stack keeps its own copy of the plugin directory, so the change is again one line — `NCCL_MESH_PLUGIN_DIR`
pointed at a patched build — with the port counters as the check that it engaged. Also `[not tested]` there.

Next: the full tables in [10 — Results and roofline](10-results-and-roofline.md).

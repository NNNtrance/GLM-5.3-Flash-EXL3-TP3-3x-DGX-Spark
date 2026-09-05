# 06 — The NCCL mesh cliff

The mesh plugin is what makes a switchless three-node triangle work at all ([00](00-hardware-and-os.md)). It also
has a hole in the middle of its message-size range where the all-reduce runs at **0.6–1.9 GB/s** on links that reach
12–13 GB/s at 16 MB, and this engine's decode all-reduce sits in the deepest part of it. One environment variable
takes it out: set **`NCCL_MAX_NCHANNELS=8`** on all three nodes. It costs nothing measurable and is worth
**+12.6 % at C8** `[measured-here]`.

**Settings for every engine number below.** Image `exl3-zeus:bc0e0f6s`, TP=3 + expert parallel, EXL3 4bpw weights,
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
from 8 within noise model-free and slightly better at 3.4 MB, but has never been taken to the engine (§9); we ship 8
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

## 5. The upstream fix we wrote but do not run

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

**Why we do not deploy it.** With `NCCL_MAX_NCHANNELS=8` in place its model-free contribution is inside the noise, so
it never earned an engine boot. It is built and staged on all three nodes, selectable with `NCCL_MESH_PLUGIN_DIR`
without touching the production plugin directory; do that if you ever relax the cap. Patch and finding went upstream —
see [../CREDITS.md](../CREDITS.md).

## 6. What is NOT the cause

| hypothesis | test | result |
|---|---|---|
| link, MTU, driver | point-to-point over the same queue pairs | 13.3 GB/s, clean through the cliff `[measured-here]` |
| packet loss, congestion | `packet_seq_err`, `local_ack_timeout_err` | 0 everywhere, every size, every arm `[measured-here]` |
| protocol choice (LL / LL128 / Simple) | `NCCL_PROTO` forced | no protocol fixes it — §7 `[measured-here]` |
| NCCL buffer too small | `NCCL_BUFFSIZE=8388608` | no change; the 32 KiB chunk floor is absolute, not a fraction of the buffer `[measured-here]` |
| per-peer channel sharing | `NCCL_NCHANNELS_PER_NET_PEER=1` | no change `[measured-here]` |
| missing GPUDirect RDMA (`ptrSupport = NCCL_PTR_HOST`) | debug log: GPU Direct RDMA disabled on all four HCAs | real, and it caps the ceiling at ~13 GB/s against a 25 GB/s link — but it is size-independent and cannot produce a cliff `[measured-here]` |

**Retraction** `[retracted]`. An earlier report of ours read the profiler kernel name
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` as "NCCL is using LL even at 16 MB, so half the link is on the table", and
proposed forcing `Simple` as the largest prefill lead open. Forcing LL at 16 MB costs **20,114 µs against auto's
1,787 µs**, an 11× difference, so auto is plainly not using LL there `[measured-here]`; and 11–12.5 GB/s is what this
mesh delivers at that size anyway, since `Simple` — no flag words at all — reaches 12.2 GB/s. There is no half link to
recover. What replaced it is the finding on this page: the loss is mid-range, and the mechanism is RNR.

## 7. `NCCL_PROTO` — rejected

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

## 8. What this cost

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

## 9. What is left open

1. **12 channels has never been taken to the engine** `[not tested]`. Model-free it is indistinguishable from 8,
   slightly better at 3.4 MB, and keeps more parallelism for the largest messages. One boot would settle it; the only
   change is `NCCL_MAX_NCHANNELS=12`.
2. **The plugin patch has never been A/B'd on the engine** `[not tested]`. With the cap in place its model-free
   contribution is inside the noise, so it did not earn a boot. Item 3 is a reason to give it one.
3. **RNR has not gone to zero in the engine, only in the microbenchmark.** A live counter read across one full sweep
   plus prefill plus mixed load on the final 8-channel engine still shows **~42,000 `rnr_nak_retry_err` and
   24,000–42,000 `out_of_buffer` per node over about five minutes** `[measured-here]`. At 0.64 ms apiece over 8
   channels that is **1–3 % of wall clock** in back-off — what the `min_rnr_timer` patch would make ~64× cheaper.
4. **The real fix is a redesign, not a timer.** The receiver-advertised FIFO of NCCL's own IB plugin removes RNR from
   the steady state instead of making it cheap, and would let the plugin advertise `NCCL_PTR_CUDA` and recover the
   GPUDirect path that currently holds the ceiling at ~13 GB/s. Much larger change; the timer is the one-line version.
5. **The collective's share of a decode step has not been re-profiled** under the new setting `[not tested]`. §4
   gives the arithmetic; one profile run would replace the inference with a measurement.

More in [11 — Open issues](11-open-issues.md); the kernel work sharing this ground is in
[05](05-expert-parallel-and-cuda-exl3-fixes.md). The full sweep matrices behind every table here — including the
23-size µs matrix across all five plugin configurations — are in `results/mesh/`.

## 10. The same fix applies to the NVFP4 sibling

Our NVFP4 recipe runs the **same plugin binary at the same commit** over the **same fabric** at the **same TP=3**,
with the same 4096-wide hidden state and so the same 512 KB decode all-reduce. Same cliff, same fix: one environment
line per node, no rebuild, reversible by restoring the env file:

```
EXTRA_ENV="NCCL_MAX_NCHANNELS=8"
```

Gate it the usual way — correctness probe 10/10, code exam 12/12, then one sweep round — and expect the gain on
**C4–C8 decode**, not on C1 and not on prefill. The change has been measured on the EXL3 stack only; on the NVFP4
stack it is a recommendation, not a result `[not tested]`.

Next: the full tables in [10 — Results and roofline](10-results-and-roofline.md).

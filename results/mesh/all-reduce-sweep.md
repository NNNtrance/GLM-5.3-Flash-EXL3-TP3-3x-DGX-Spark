# Mesh all-reduce: the cliff, the counters, and the cure

Model-free, engine down, one process per node over the NCCL mesh plugin with the engine's exact NCCL
environment. `bench/mesh_sweep.py` steps the message size by about 1.5× from 8 KB to 64 MB and reads
the ConnectX-7 hardware counters around every timed loop. Median of three independent runs unless
stated. World = 3, `NCCL_ALGO=Ring`. 5 September 2026 `[measured-here]`.

Narrative and root cause in [docs/06](../../docs/06-nccl-mesh.md).

## 1. It is the collective, not the link

| message | all-reduce | all-to-all | point-to-point send/recv |
|---|---|---|---|
| 311 KB | 0.52 GB/s | 1.85 GB/s | 4.78 GB/s |
| 475 KB | 0.54 | 1.95 | 2.90 |
| 704 KB | 0.61 | 4.88 | 5.44 |
| 1.6 MB | 1.85 | 1.38 | 9.24 |
| 3.6 MB | **0.83** | 3.70 | **10.97** |
| 5.4 MB | 1.60 | 6.02 | 11.79 |
| 8.1 MB | 2.38 | 5.60 | 12.68 |
| 18.2 MB | 11.28 | 6.25 | 12.24 |
| 64 MB | 12.31 | 11.29 | 13.25 |

Point-to-point over the same queue pairs is healthy across the whole cliff — 11 GB/s at 3.6 MB where
the all-reduce manages 0.83. The cable, the NIC, the MTU and the plugin's data path are not the
problem; what happens when many connections are driven at once is.

## 2. The counters name the mechanism

Per-collective deltas from the same sweep:

| message | µs | GB/s | `rnr_nak_retry_err` | `out_of_buffer` | `packet_seq_err` |
|---|---|---|---|---|---|
| 65 KB | 188 | 0.46 | 0.3 | 0.0 | 0 |
| 213 KB | 263 | 1.08 | 0.9 | 0.1 | 0 |
| 311 KB | 804 | 0.52 | 3.0 | 0.6 | 0 |
| 475 KB | 1,179 | 0.54 | 6.0 | 0.9 | 0 |
| 704 KB | 1,553 | 0.61 | 7.7 | 1.8 | 0 |
| **3.6 MB** | **5,749** | **0.83** | **28.5** | **23.0** | 0 |
| 5.4 MB | 4,471 | 1.60 | 17.2 | 13.4 | 0 |
| 12.1 MB | 4,101 | 3.94 | 8.4 | 2.0 | 0 |
| 18.2 MB | 2,146 | 11.28 | 2.4 | 0.0 | 0 |
| 64 MB | 7,270 | 12.31 | 7.7 | 0.0 | 0 |

`packet_seq_err` and `local_ack_timeout_err` are zero everywhere: no packet is lost and no ACK times
out. The only anomaly is the receive-not-ready NAK, and it rises and falls with the cliff.

## 3. Dose-response across the channel cap

| channels | 464 KB µs | 464 KB RNR+OOB | 3.4 MB µs | 3.4 MB RNR+OOB | 64 MB µs |
|---|---|---|---|---|---|
| 2 | 172 | 0.1 | 813 | 0.1 | 7,713 |
| 4 | 158 | 0.2 | 1,186 | 1.7 | 7,744 |
| 6 | 256 | 0.3 | 801 | 0.6 | 7,444 |
| **8** | **112** | **0.3** | **919** | **2.2** | **7,072** |
| 12 | 155 | 0.1 | 490 | 0.7 | 7,607 |
| 16 | 417 | 1.7 | 958 | 4.2 | 9,071 |
| 32 | 1,124 | 6.5 | 4,291 | 8.2 | 6,752 |
| 64 (default) | 1,105 | 7.9 | 5,143 | 48.1 | 8,275 |

Monotone, in both the counters and the time. Below 6 channels the cliff is also gone but the largest
messages start to lose the parallelism they need; above 16 the misses return. 8 keeps both ends; 12
is indistinguishable within noise and was never taken to the engine.

## 4. The message sizes the engine actually uses

Median of four runs, hidden 4096 bf16:

| what | message | stock (64 ch) | **`NCCL_MAX_NCHANNELS=8`** | gain |
|---|---|---|---|---|
| C1 decode all-reduce | 8 tok = 64 KB | 90 µs | 143 µs | latency-bound, ±100 µs noise |
| | 16 tok = 128 KB | 278 | 280 | — |
| | 32 tok = 256 KB | 588 | 86 | **6.8×** |
| **C8 decode all-reduce** | **64 tok = 512 KB** | **1,195** | **123** | **9.7×** |
| | 128 tok = 1 MB | 1,432 | 203 | **7.1×** |
| | 256 tok = 2 MB | 1,935 | 453 | **4.3×** |
| prefill chunk (MNBT 2048) | 2048 tok = 16 MB | 2,235 | 1,846 | 1.21× |
| prefill chunk (MNBT 4096) | 4096 tok = 32 MB | 4,111 | 3,808 | 1.08× |
| **one C8 decode step** | 90 × 512 KB | **91.7 ms** | **9.9 ms** | **9.3×** |

## 5. Protocol sweep — why `NCCL_PROTO` is left unset

Microseconds per all-reduce, world = 3, each protocol measured twice:

| message | auto (×2) | LL | Simple (×2) | LL128 | `^LL` |
|---|---|---|---|---|---|
| 8 KB | **38.6 / 38.7** | 48.3 | 87.1 / 50.8 | 46.7 | 49.6 |
| 64 KB | **61.3 / 61.5** | 69.7 | 172.9 / 175.8 | 70.9 | 70.5 |
| 128 KB | 200.6 / 174.1 | 313.8 | 2,532.9 / 2,236.3 | **106.5** | 91.0 |
| 256 KB | 192.6 / 246.6 | 1,001.4 | 3,869.3 / 3,974.2 | 556.8 | 353.1 |
| 512 KB | **983.3 / 1,038.6** | 2,048.7 | 4,372.8 / 4,488.0 | 1,014.6 | 1,097.7 |
| 1 MB | 1,462.8 / 1,458.1 | 3,271.4 | 7,261.7 / 7,088.4 | **1,424.6** | 1,362.1 |
| 4 MB | 5,287.3 / 5,318.1 | 5,528.0 | 5,330.4 / 5,508.4 | **1,702.4** | 5,308.0 |
| 16 MB | **1,787.3 / 1,792.2** | 20,113.6 | 2,030.7 / 1,829.8 | 3,758.3 | 1,912.1 |
| 64 MB | 6,888.2 / 6,856.5 | 72,175.0 | 6,861.8 / 6,860.9 | 11,739.0 | 6,807.1 |
| one decode step (90 × 8 tok) | **3.84 / 3.96 ms** | 3.97 | 9.97 / 10.14 ms | 4.77 / 4.91 | 4.77 |

`Simple` is a loss at every size the engine uses. Forcing `LL` at 16 MB costs 20,114 µs against
auto's 1,787 µs, which also retracts an earlier reading of ours that the tuner was choosing `LL`
there. `NCCL_ALGO` unset measures the same as the forced `Ring`.

**The small-size rows of this table are contradicted by a later harness of ours and are kept anyway.**
[`nccl-latency-sweep.md`](nccl-latency-sweep.md), 6 September, production configuration, reads
**86.4 µs at 64 KB** and **74.68 µs at 8 KB** where the rows above read 61.3 and 38.6. §4 of this same
page reads 143 µs at 64 KB with a third tool. Three harnesses, three answers, one operation. They are
all recorded, none is corrected against the others, and a same-session comparison is
[HELP-WANTED](../../HELP-WANTED.md) §5 — see [docs/06](../../docs/06-nccl-mesh.md) §12.1.

## 6. Engine A/B

Five sweep rounds per arm, rounds 1–2 discarded, one boot per arm, plus a third confirmation boot
with the setting written into the environment file `[measured-here]`:

| | control (64 channels) | arm B (8 channels) | **confirmation boot (8 channels)** |
|---|---|---|---|
| C1 | 47.97 | 51.16 | **51.38** (+7.1 %) |
| C2 | 69.97 | 76.56 | **74.27** (+6.1 %) |
| C4 | 98.17 | 109.79 | **107.60** (+9.6 %) |
| C6 | 115.48 | 124.31 | **129.10** (+11.8 %) |
| C8 | 133.44 | 150.27 | **150.83** (+13.0 %) |
| prefill-fresh | 1,689 | 1,720 | **1,728** |
| mixed load: decode / 7k TTFT | 10.2 / 4.7 s | 10.6 / — | **10.3 / 4.6 s** |
| GPU KV cache | 1,639,427 | 1,633,299 | **1,648,621** |
| gates cold + warm | 10/10 · 12/12 | 10/10 · 12/12 | **10/10 · 12/12** |

Two details worth keeping. **The control degrades across rounds and the fix does not**: control C1
over five rounds read 50.5 → 49.4 → 48.0 → 41.6 → 50.1, while the capped arm read 51.2 → 49.9 → 51.2
across rounds 3–5. And **the engine gain is far smaller than the microbenchmark gain** — +12.6 %
against 9.3× — which is the honest shape of it: the collective is one term in a step, and removing
most of a ~24 % term cannot give more than about +30 %. The measured +12.6 % says the real share
today is nearer 12 %. Nothing here justifies quoting the 9× as an end-to-end number.

## 7. What is left

RNR has not gone to zero in the engine, only in the microbenchmark. A live counter read across one
full sweep plus prefill plus mixed load on the capped engine still shows roughly **42,000
`rnr_nak_retry_err`** and 24,000–42,000 `out_of_buffer` per node over about five minutes — on the
order of 1–3 % of wall clock once spread over 8 channels `[measured-here]`. That residual is exactly
what the plugin patch (`patches/kernel/0004-min-rnr-timer.patch`) would make about 64× cheaper, and
it is the concrete argument for giving that patch a boot.

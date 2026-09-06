# The all-reduce latency sweep: flat from 8 B to 32 KiB, and 98 % of the wire at 16 MiB

Model-free, one process per node over the NCCL mesh plugin, with the engine's production NCCL
environment. Byte-precise geometric sweep **8 B → 64 MiB, ×2**, 10 warmup + 50 timed iterations per
size, `busBW = algBW × 2(n−1)/n` for n = 3, and a correctness check at every size. 6 September 2026
`[measured-here]`.

The tool is [`bench/nccl-latency-bench.py`](../../bench/nccl-latency-bench.py) and its runners; the
contract it reproduces, and why it is not `nccl-tests`, is
[`bench/README-nccl-latency.md`](../../bench/README-nccl-latency.md). Narrative and the transport
work behind these numbers: [docs/06](../../docs/06-nccl-mesh.md).

**Two answers in one line.** At decode message sizes this fabric is **latency-bound** — time is flat
within about 20 % across a 4,096× range of sizes — and at the prefill message size it is
**bandwidth-bound at 98.1 % of our own measured wire**, which means there is no software headroom
left at that end.

**Read the harness caveat before you quote a small-size number from this page.** Our two all-reduce
harnesses disagree by about 40 % at 64 KB and about 2× at 8 KB, both are recorded, and neither has
been shown to be the right one — §6, [docs/06](../../docs/06-nccl-mesh.md) §12.1,
[HELP-WANTED](../../HELP-WANTED.md) §5.

---

## 1. Settings

- **Fabric:** three DGX Spark (GB10) nodes, ConnectX-7, three cables in a ring so every pair has a
  direct link; the mesh net plugin built from `autoscriptlabs/nccl-mesh-plugin` at `19924dcc` with
  `patches/kernel/0004`, `0005` and `0006` applied.
- **NCCL environment**, exactly the engine's: `NCCL_ALGO=Ring`, `NCCL_MAX_NCHANNELS=8`,
  `NCCL_MESH_LINKS_PER_PEER=0` (both links of every pair), `NCCL_MESH_MIN_RNR_TIMER=1`,
  `NCCL_MESH_PTR_CUDA=1`, `NCCL_MESH_FLUSH=1`, `NCCL_PROTO` unset unless a row says otherwise.
- **Payload:** bf16, world = 3, `torch.distributed` all-reduce, CUDA-event timing.
- **Isolation:** the serving engine was **up and idle** on two of the three nodes throughout. The
  bench containers were pinned to a cpuset outside the engine's cores and memory-capped, the
  measurement lock was taken after waiting 720 s for another arm to finish, the engine's own
  running/waiting counters were confirmed at zero before starting, and `MemAvailable` stayed ≥ 9 GiB
  on all three nodes for the whole sweep. **The engine was not stopped and was not touched**
  ([docs/09](../../docs/09-measurement-protocol.md) §10).
- **Correctness:** each size fills every rank with `rank + 1` and compares the reduction against the
  exact expected sum — exact for small integers in bf16. **132 of 132 checks passed, 0 failures.**

Raw: [`nccl-latency/nccl-latency.csv`](nccl-latency/) (132 rows), one log and one JSON per
repetition in the same directory.

## 2. The production configuration, median of three repetitions

| bytes | size | time (µs) | algBW (GB/s) | busBW (GB/s) |
|---:|---:|---:|---:|---:|
| 8 | 8 B | 80.38 | 0.0001 | 0.0001 |
| 16 | 16 B | 75.69 | 0.0002 | 0.0003 |
| 32 | 32 B | 81.18 | 0.0004 | 0.0005 |
| 64 | 64 B | 77.84 | 0.0008 | 0.0011 |
| 128 | 128 B | 72.96 | 0.0018 | 0.0023 |
| 256 | 256 B | 79.36 | 0.0032 | 0.0043 |
| 512 | 512 B | 76.20 | 0.0067 | 0.0090 |
| 1,024 | 1 KiB | 79.00 | 0.0130 | 0.0173 |
| 2,048 | 2 KiB | 77.03 | 0.0266 | 0.0354 |
| 4,096 | 4 KiB | 72.32 | 0.0566 | 0.0755 |
| 8,192 | **8 KiB** | **74.68** | 0.1097 | 0.1463 |
| 16,384 | 16 KiB | 76.16 | 0.2151 | 0.2869 |
| 32,768 | 32 KiB | 79.78 | 0.4108 | 0.5477 |
| 65,536 | **64 KiB** | **86.40** | 0.7586 | 1.0114 |
| 131,072 | 128 KiB | 172.48 | 0.7599 | 1.0132 |
| 262,144 | 256 KiB | 92.08 | 2.8469 | 3.7959 |
| 524,288 | 512 KiB | 141.40 | 3.7078 | 4.9438 |
| 1,048,576 | **1 MiB** | **275.12** | 3.8113 | 5.0818 |
| 2,097,152 | 2 MiB | 483.85 | 4.3343 | 5.7791 |
| 4,194,304 | 4 MiB | 305.51 | 13.7288 | 18.3051 |
| 16,777,216 | **16 MiB** | **1,096.79** | 15.2966 | **20.3955** |
| 33,554,432 | 32 MiB | 1,995.37 | 16.8161 | 22.4215 |
| 67,108,864 | 64 MiB | 3,954.98 | 16.9682 | 22.6243 |

(8 MiB, 539.54 µs / 20.7301 GB/s, is in the CSV; it is omitted above only to keep the table to one
screen.)

**The 128 KB point is not noise.** 172.48 µs is roughly twice its neighbours and it reproduced in all
three repetitions — 159.9 / 173.3 / 172.5 µs. It is the surviving trace of the 128 KB – 4 MB
all-reduce cliff diagnosed in [docs/06](../../docs/06-nccl-mesh.md): that cliff read 0.6–1.9 GB/s
before the transport work and reads **1.0–5.8 GB/s** today. **It is smaller. It is not closed.**

## 3. Decode: latency-bound, and the shape says so on its own

- **8 KiB: 74.68 µs** median. The three repetitions were 72.4 / 101.2 / 74.7 — the middle one is a
  noise spike and the median is not moved by it.
- **64 KiB: 86.40 µs.**
- **From 8 B to 32 KiB — a 4,096× range of sizes — the time stays inside 72–85 µs.** That is the
  classical latency-bound signature: a fixed floor that the message size does not reach.

**What we did not run, said plainly.** The `cuda-exl3` author's own ruler for this question is a raw
flag ping-pong over pinned host memory (**6.74 µs one way, 13.49 µs round trip** on his box), and
**we did not run that primitive** — it is a lower-level tool than the all-reduce sweep this page is
built from and was out of scope for this pass `[not tested]`. The sweep's own curve is sufficient
evidence for the *shape*.

**The absolute numbers are not comparable and are not compared.** His fabric is PCIe host staging on
a no-P2P box; ours is RoCE through a net plugin. At 8 KB he reads 13.6 µs where we read 74.68. Only
the shape transfers, and the shape agrees: both are latency-bound at decode.

## 4. Prefill: at the wire

The prefill collective at `--max-num-batched-tokens 2048` is 2,048 tokens × hidden 4,096 × bf16 =
**16 MiB**.

| | |
|---|---|
| Measured busBW at 16 MiB | **20.40 GB/s** |
| Our own measured wire, per direction per pair | **20.8 GB/s** — the `link2cuda` arm's 64 MB sendrecv after the dual-cable and `NCCL_PTR_CUDA` fixes, measured range 19.5–23.2 GB/s ([`multilink-sweep.md`](multilink-sweep.md)) |
| **Utilisation** | **98.1 %** |

**Our 64 MiB point, 22.62 GB/s, sits above that 20.8 GB/s reference.** Either the fabric improved
slightly after the 5 September transport work, or the reference itself is inside its own noise — it
was measured across a 19.5–23.2 GB/s range. This page always quotes utilisation against the
**20.8 GB/s measured** figure, and two independent measurements taken a day apart overlap within that
band, so 20.8 GB/s stays a valid planning input.

**The 50 GB/s denominator is gone and this is where it is buried.** An earlier note computed "2 × 25
= 50 GB/s per pair" from the ports' rated speed. That is wrong: the ceiling sits in each card's
PCIe Gen5 x4 slot at about 15 GB/s per NIC, two NICs per node ≈ 30 GB/s total
([docs/06](../../docs/06-nccl-mesh.md) §9, [docs/11](../../docs/11-open-issues.md) §1.7). **The
"13.9 GB/s / 50 GB/s = 28 %" figure that was quoted upstream was already stale when it was quoted and
is doubly wrong now** — the denominator was retracted and the numerator moved with two rounds of
transport fixes. It is 20.40 against 20.8, or 98.1 %.

## 5. Where bandwidth arrives, and why it arrives there

The plateau is the mean busBW of the three largest points (16 / 32 / 64 MiB) = **21.81 GB/s**, so
80 % of plateau is **17.45 GB/s**. The first point that clears that threshold and never falls back
below it is **4 MiB, at 18.31 GB/s**.

That crossover is not an independent fact: the 128 KB – 2 MB band where busBW is depressed sits
entirely **below** the threshold, so the crossover lands exactly where the cliff ends. Closing the
rest of the cliff would move the crossover down, not raise the plateau.

## 6. Channels, protocol, and the disagreement between our own two harnesses

### 6.1 `NCCL_MAX_NCHANNELS=8` against the default

| size | production, 8 channels | default, channels unset | Δ time |
|---:|---|---|---|
| 8 KiB | 74.68 µs / 0.146 GB/s | 73.83 µs / 0.148 GB/s | −1.1 % (inside noise) |
| 64 KiB | 86.40 µs / 1.011 GB/s | 80.02 µs / 1.092 GB/s | −7.4 % (inside noise) |
| 128 KiB | 172.48 µs / 1.013 GB/s | 200.74 µs / 0.871 GB/s | +16.4 % |
| 1 MiB | 275.12 µs / 5.082 GB/s | 1,388.02 µs / 1.007 GB/s | **+404 %** |
| 4 MiB | 305.51 µs / 18.305 GB/s | 3,553.86 µs / 1.574 GB/s | **+1,063 %** |
| 16 MiB | 1,096.79 µs / 20.396 GB/s | 3,046.16 µs / 7.343 GB/s | **+178 %** |
| 64 MiB | 3,954.98 µs / 22.624 GB/s | 3,945.28 µs / 22.680 GB/s | −0.2 % (no difference) |

**The channel cap does nothing at decode sizes and is worth up to 11× in the cliff.** Both halves
follow from the same mechanism: a fixed base latency cannot be reduced by adding channels, and the
cliff is caused by channel-count-driven receive-buffer misses ([docs/06](../../docs/06-nccl-mesh.md)
§3–§4). At 64 MB bandwidth is already saturated and the channel count stops mattering.

**This is a property of a multi-NIC mesh, not of NCCL.** The `cuda-exl3` author swept channel count
on his single-host no-P2P box and found it **flat** `[reported]`. A reader on one host should not
expect the 8-channel effect at all.

### 6.2 `NCCL_PROTO`

Auto (production, `proto` unset) is at least as good as forced `LL` and forced `Simple` at every size
measured, and clearly better in the 128 KiB – 1 MiB band: auto **92–275 µs**, LL **209–330 µs**,
Simple **619–683 µs** — Simple is **4–7× slower** through that band `[measured-here]`. Raw:
[`nccl-latency/protoLL-r0.json`](nccl-latency/) and
[`nccl-latency/protoSimple-r0.json`](nccl-latency/), ≤ 1 MiB, single run each.

The same conclusion on the author's very different fabric `[reported]`: "Neither `NCCL_ALGO` nor
`NCCL_PROTO` beats the default at any size… forcing LL costs 1.9× at 512 KB." **Two independent
fabrics, one verdict: do not force NCCL's protocol choice.**

### 6.3 Our two harnesses disagree at small sizes, by about 40 %

| 64 KB all-reduce, auto protocol | reading |
|---|---|
| `bench/ar_bench.py`, earlier configuration ([docs/06](../../docs/06-nccl-mesh.md) §12.1, [`all-reduce-sweep.md`](all-reduce-sweep.md) §5) | **61.3 / 61.5 µs** |
| `bench/nccl-latency-bench.py`, this page, production configuration | **86.4 µs** |
| `bench/mesh_sweep.py` ([`all-reduce-sweep.md`](all-reduce-sweep.md) §4, `NCCL_MAX_NCHANNELS=8`) | **143 µs**, labelled at the time as latency-bound with ±100 µs of noise |

At 8 KB the same two read **38.6 µs** and **74.68 µs** — about a factor of two.

**Both are recorded and neither is corrected against the other.** They differ in more than one
variable at once — a different iteration count and warmup, a different timing loop, and, for the
`ar_bench` rows, a plugin build and cabling state from before the dual-cable and `NCCL_PTR_CUDA`
work — so no single cause can be assigned from the data that exists. **This repository's standing
rule is that the ruler gets measured too, and three of ours have turned out to be brochures**
([docs/11](../../docs/11-open-issues.md) §4). Until someone runs both harnesses **in one session, on
one configuration**, no small-size figure from either should be quoted as *the* latency of this
fabric. That comparison is [HELP-WANTED](../../HELP-WANTED.md) §5.

The conclusion of this page does not rest on the absolute value: the decode verdict comes from the
**flatness** of the curve across 4,096× of size, which both harnesses show, and the prefill verdict
comes from a 16 MiB point where the two harnesses agree to within a few percent.

## 7. Verdict, and where the lever is not

**Decode: latency-bound. Prefill: bandwidth-bound, at 98.1 % of the wire.** Both agree in direction
with what the kernel author measures on his own hardware. At the small end the base latency
dominates and no channel count or protocol touches it; at the large end the wire is full.

**There is no software headroom left at the prefill end**, and the one-sided `RDMA_WRITE` transport —
which drove RNR and out-of-buffer to exactly zero without moving throughput — already demonstrated
that the limit is the PCIe slot rather than the protocol
([docs/06](../../docs/06-nccl-mesh.md) §9–§10, [`rdma-write-sweep.md`](rdma-write-sweep.md)). The
remaining levers are not in the transport at all: **fewer collectives, larger collectives, or
collectives overlapped with compute** — the author's phrasing, and it matches our own profile, where
the NCCL class is 100 % exposed with 0.00 ms of measured overlap
([docs/10](../../docs/10-results-and-roofline.md) §5, [docs/11](../../docs/11-open-issues.md) §2.17).

**What this measurement cost.** No engine restart, no configuration change, no code change: it ran
beside a live idle engine under the measurement lock. The lock cleared after **720 s** of waiting for
another arm, and the sweep then held the fabric from acquisition to release for **94 s**.

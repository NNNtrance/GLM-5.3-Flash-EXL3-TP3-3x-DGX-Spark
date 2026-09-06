# The memory ladder above 0.83, climbed rung by rung

**Four rungs measured in one session on 6 September 2026, with the criterion changed from "free RAM"
to "swap traffic under load".** Three passed, one was rejected, and the rung that ships is not the
highest one that passed.

Settings, every rung: image `exl3-zeus:754421f`, TP=3 + expert parallel, full-scope EXL3 weights
(`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw), `kv-cache-dtype fp8` and an fp8 draft cache, DFlash2
draft at k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`,
`--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, `max_model_len 1,000,000`, per-rank pre-sliced sidecar,
warm MLA tuner cache, temperature 0, reasoning effort **low**. Every rung is a **fresh boot** of the
same tree with **one line** changed — the environment file for each rung was derived on each node
from that node's own production file with `sed`, and the diff is one line on all three
`[measured-here]`.

Speed is the median of three sweep rounds on the twelve realistic prompts of
[`scripts/hizset-v2.jsonl`](../../scripts/hizset-v2.jsonl). Gates are the correctness probe (10) and
the code exam (12), run cold and again after the whole battery.

---

## 1. The criterion, and why it changed

The previous verdict on this ladder was "0.85 rejected, 0.88 never attempted", and it rested on
**`MemFree`** — a 4 GiB floor and a 1.9 GiB reading. Two things made that ruler wrong for this stack:

- **`MemFree` is not headroom on a unified-memory part.** Most of what the kernel is holding at that
  moment is reclaimable page cache. `MemAvailable` is the honest number and it was never read.
- **The rejected 0.85 boot predates the fast-load work**, which removed a large page-cache spike
  during loading and added `MADV_DONTNEED` plus `malloc_trim` at the end of it
  ([08](../../docs/08-fast-boot.md) §5). The machine that verdict was measured on no longer exists.

The criterion used here instead is **swap traffic while the benchmark is running** — `si` and `so`
from `vmstat -n -t 1`, sampled on all three nodes for the whole arm, summed over the window between
"engine up" and "battery done". Not swap *used*, which is a stock and sits at ~0.04 GiB from boot on
every rung; **traffic**, which is a flow and is what a stall is made of. A rung passes on three
conditions together: swap traffic ≈ 0, C1 inside ±4 % and C8 inside ±3 % of the same-session
reference, and both gates full cold and warm.

## 2. The rungs

Reference is `gpu-memory-utilization` **0.83**, the shipping configuration, booted the same way in
the same session, so the comparison carries no cross-day drift.

| | 0.83 (reference) | 0.85 | 0.87 | 0.88 | 0.90 |
|---|---:|---:|---:|---:|---:|
| KV pool, tokens | 5,674,931 | 6,016,528 | **6,363,636** | 6,542,699 | 6,870,523 |
| against reference | — | +6.0 % | **+12.1 %** | +15.3 % | +21.1 % |
| `num_blocks` | 2,060 | — | 2,310 | 2,375 | — |
| Available KV, binding rank | 40.17 GiB | 42.60 | 45.05 | 46.32 | 48.63 |
| C1 tok/s | 69.33 | 70.98 | 70.95 | 70.88 | 70.56 |
| C4 tok/s | 143.03 | 143.03 | 144.18 | 143.57 | 139.16 |
| C8 tok/s | 196.76 | 195.11 | 196.59 | 192.86 | 193.22 |
| prefill-fresh tok/s | 1,738 | 1,761 | 1,754 | 1,783 | — |
| Gates, cold · warm | 10/10 · 12/12 | full | full | full | full |
| **Swap traffic under load** | ~0 | **0 / 0 / 0 KiB** | **0 / 0 / 0 KiB** | 4 KiB / 0 / 0 | **head si 142.6 MiB + so 1,519.4 MiB** |
| `MemAvailable` min under load | 8.35 / 9.74 / 9.73 GiB | 5.99 / 7.31 / 7.29 | 3.49 / 4.77 / 4.76 | **1.86** / 3.52 / 3.48 | 1.04 / 1.45 / 1.18 |
| OOM killer | 0 | 0 | 0 | 0 | 0 |
| **Verdict** | — | pass | **pass — shipped** | pass, thin | **rejected** |

Every rung is `[measured-here]`. The three-value columns are head / worker-1 / worker-2.

**No speed number at any rung is outside its band**, including the rejected one. C1 is *higher* than
the reference at every rung, by 1.8–2.4 %, which is inside the ±4 % band and is not read as a gain.
C6 is left out of the table on purpose: its round-to-round spread on this stack is the largest of any
level (164.9 / 168.2 / 179.4 at 0.87), and reading it would be reading noise.

## 3. Why 0.90 was rejected, in the numbers

Swap traffic on the head node, over the 598-second load window: **250 of those seconds were not
zero, and the longest unbroken run was 85 s**. It is not a single eviction — there is **swap-in** as
well as swap-out (142.6 MiB read back), so pages are being paged out and then asked for again. The
workers show the same thing smaller: 553 MiB out on worker-1 over 21 s, 482 MiB on worker-2 over 23 s.
Swap in use at the end of the window: **2.65 / 0.57 / 0.52 GiB** against 0.03 at the reference.

Throughput does not see it yet, and that is the point of measuring the traffic rather than the
tok/s: what is being evicted is cold memory, so C1/C4/C8 all stay inside their bands. What does move
is the **first prefill of the arm: 5.0 s → 9.8 s, twice**. The stall is measurable at the client
before it is visible in an aggregate.

## 4. Why 0.88 passed and 0.87 shipped anyway

0.88 is a clean measurement — +15.3 % of pool, every speed number in band, gates full, 4 KiB of swap
traffic in a single sample. It was not taken, and the reason is in the `MemAvailable` row: **1.86 GiB
on the head node**, with the kernel's page cache down from ~6.6 GiB to **2.59 GiB**.

Swap traffic is still zero at 0.88 *because the kernel can still find room in the cache*. When that
buffer is gone the only remaining source of room is swap — which is exactly what the next rung
demonstrates, and the next rung asks for about 2.4 GiB more, which is about what the buffer has left.
0.88 is not a bad rung; it is the last one before the cliff, with nothing between it and the cliff.

0.87 leaves **3.49 GiB** of `MemAvailable` on the binding node. On this cluster the engine is not the
only thing running — model-free kernel benches, profilers and diagnostic containers run beside it —
and that headroom is the budget they run in. The half-rung between 0.87 and 0.88 buys 2.8 % of pool
and costs 1.6 GiB of that budget.

Every 1 % of `gpu-memory-utilization` costs about **1.2 GiB** of host headroom on this hardware.

## 5. What the ladder does not change

- **The pool is not the binding constraint at three nodes.** A 1M-token request costs 363 blocks =
  7.078 GiB on every node, so 0.83 already held 5.67 of them and 0.87 holds 6.36
  ([17](../../docs/17-memory-ledger.md)). At two nodes the pool binds; at three it does not. This
  rung is headroom for concurrency, not a capability that was missing.
- **Quality.** Gates are full at every rung, cold and warm, and MMLU was not re-run: a memory
  fraction does not touch the arithmetic `[not tested]`.
- **`--kv-cache-memory`** — sizing the pool in bytes instead of as a fraction — is still unused and
  still the sharper instrument ([07](../../docs/07-kv-and-draft-page.md) §6). Ladder first, pin last.

## 6. A ruler correction found on the way

The first `dmesg` scan of the campaign counted "60 OOM events" per node. They are **not** the Linux
OOM killer: they are `NVRM ... NV_ERR_NO_MEMORY ... _memdescAllocInternal` lines, which appear on
**every** boot including the 0.83 reference. That is vLLM's memory profiler probing for the ceiling
and failing safely. The real OOM-killer count is **0** on all three nodes at every rung. A scan that
greps for "out of memory" without separating the two sources will reject a healthy rung.

A second one, in the harness rather than the kernel: the sweep script writes each arm's rounds into a
directory named after the arm's **label**, and a label reused from an earlier day leaves that day's
files in place, where the aggregator medians them together with the new ones. It cost one wrong
reading (C1 68.75 / C8 192.58 against the correct 70.98 / 195.11) before it was caught. The
aggregator now ignores any file written before the arm started and prints a warning
([09](../../docs/09-measurement-protocol.md)).

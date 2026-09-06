# Boot ledger — 618 s to 274 s

Seconds per phase, parsed from rank 0's own timestamped container log. Same image
(`exl3-zeus:f4987cf`), same three nodes. The baseline arm ran at `gpu-memory-utilization 0.85` and
every other arm at 0.80; weight loading is unaffected by that, the KV pool is not, so KV is compared
against the 0.80 reference arm rather than against the baseline `[measured-here]`.

Full narrative in [docs/08](../../docs/08-fast-boot.md).

| Phase | baseline | S1+S2+S3 | S4 (verifying) | **production (S4)** |
|---|---:|---:|---:|---:|
| 1 container + prelude + preflight + import + distributed init | 49.3 | 38.9 | 39.8 | **48.4** |
| 2 **main weight load** | **426.3** | **189.7** | **65.3** | **67.2** |
| 3 drafter load + load close-out | 6.5 | 4.0 | 76.7 (1) | 23.2 (2) |
| 4 profile run (graph memory profile + first NCCL + MLA tune) | 67.3 | 67.2 | 67.5 | 73.1 |
| 5 KV pool → end of graph capture | 53.8 | 50.3 | 48.6 | 49.3 |
| 5a  — of which FlashInfer autotune | 34.0 | 0 (off) | 0 | 0 |
| 6 engine core close-out | 7.9 | 8.0 | 7.9 | 7.9 |
| 7 API server | 6.8 | 4.1 | 4.3 | 4.3 |
| **total** | **617.9 s** | **362.1 s** | **310.1 s** | **273.6 s** |
| GPU KV cache size | 5,256,198 (@0.85) | 4,231,404 | 4,468,319 | **4,484,848** |
| gates (probe / code) | 10/10 · 12/12 | 10/10 · 12/12 | 10/10 · 12/12 | 10/10 · 12/12 |

(1) In the verifying arm, phase 3 also carries the whole verification workload: a sampled hash check
of both models (16.8 + 14.8 s) and two post-`process_weights_after_loading` hash dumps
(23.0 + 18.7 s) — about 73 s, paid for the measurement and absent in production.
(2) In production: drafter restore 2.1 s plus a 32-tensor hash exam on each of the two models
(8.2 + 11.6 s).

## The sidecar is bound to the image, and the image moved

The manifest records the image tag, so the move from `exl3-zeus:f4987cf` to `exl3-zeus:9bf594c` (the
persisted tuner cache, [docs/12](../../docs/12-tuner-cache.md)) invalidated the sidecar and the
preflight refused to boot on it — correctly. Regenerating it cost one dump boot of **682 s** wall,
after which the boot path is the one in the table above. That dump boot's KV pool reads 3,958,677
rather than ~4.48M, because writing 56 GiB per node goes out through the page cache; the pool returns
on the next boot `[measured-here]`. Budget one such boot for every image, checkpoint, patch-set or
TP/EP change.

## Bit-identity evidence

Two independent proofs, on all three ranks `[measured-here]`:

```
rank 0  verify OK 1475/1475 tensors (53.50 GiB)   restore 64.3 s
rank 1  verify OK 1475/1475 tensors (53.50 GiB)   restore 63.5 s
rank 2  verify OK 1475/1475 tensors (53.50 GiB)   restore 55.3 s
```

1,475 is the number of tensors holding a distinct storage; the remaining 2,266 names share those
storages and are bound to the same bytes through the manifest's alias records — so this is the whole
53.50 GiB, not a sample.

And after `process_weights_after_loading`, i.e. the tensors the kernels actually read, a
full-checkpoint boot compared against a sidecar boot:

```
target model  rank 0/1/2: 400 tensors each (101 expert, 123 attention, 2 embedding, 83 norm, 91 other)  -> NO DIFFERENCE
draft model   rank 0/1/2:  94 tensors each ( 66 attention, 3 embedding, 12 norm, 13 other)              -> NO DIFFERENCE
RESULT: BIT-IDENTICAL
```

## Block-layer forensics

Bytes read from `/sys/block/…/stat` across the boot, with worker CPU sampled alongside
`[measured-here]`:

| Arm | read from disk (per node) | worker CPU, median |
|---|---|---|
| S1+S2+S3 (eager reads, EP weight filter) | 193.8 / 166.8 / 166.9 GiB | 40–50 % (under one core) |
| **S4 (sidecar)** | **57.1 / 56.6 / 56.7 GiB** | **90–100 %** |

The sidecar reads 66 % fewer bytes and the work moves to the CPU: the boot is now read-and-hash
bound rather than copy bound. "A full checkpoint's worth of bytes read while the CPU is idle" is the
signature that says the read path is the bottleneck.

## Memory at the end of each arm

| Arm | head | worker-1 | worker-2 |
|---|---|---|---|
| draft page 256 @ 0.80 | free 11.6G / swap 0.12G | 12.9G / 0.10G | 12.6G / 0.09G |
| S1+S2+S3 | 12.8G / 0.12G | 13.8G / 0.10G | 13.8G / 0.09G |
| S4 (verifying) | 11.1G / 0.12G | 12.3G / 0.10G | 12.3G / 0.09G |
| **production** | **10.9G / 0.12G** | **12.2G / 0.10G** | **12.1G / 0.09G** |

Swap is a constant ~0.1 G left over from an earlier boot and does not grow within an arm. The
fast-boot arm's more generous free memory reflects its smaller KV pool; production's tighter figure
reflects the larger one — the memory went where it was supposed to go.

## Speed and quality did not move

Medians of sweep rounds 3–5, aggregate tok/s `[measured-here]`:

| C | reference (draft page 256) | S1+S2+S3 | production |
|---|---|---|---|
| 1 | 52.81 | 53.81 | **54.37** (+3.0 %) |
| 2 | 80.97 | 80.37 | 80.06 (−1.1 %) |
| 4 | 117.08 | 112.87 | 114.61 (−2.1 %) |
| 6 | 135.64 | 136.48 | 136.82 (+0.9 %) |
| 8 | **162.85** | 164.04 | **161.82** (−0.6 %) |

Every difference is inside the arms' own within-round spread, and acceptance stayed in the 61–64 %
band. This is the intended result: the fast-boot work is supposed to change boot time and nothing
else.

## Boot from power-on: the autostart unit, measured once

A different boot from every arm above: not `docker run` on a warm host, but a **simultaneous reboot of
all three nodes** with `harem-exl3.service` enabled, on production configuration 10 (full scope,
TP=3 + EP, `754421f`, `tp3full`, `gpu-memory-utilization 0.83`), 5 September 2026 `[measured-here]`.
One trial. The raw capture is not included here, for the reason the rest of this directory gives:
it is a transcript of `ssh` to three named hosts. Every number below is transcribed from it, and the
two that disagree are both printed.

| Event | Wall clock | From T |
|---|---|---:|
| `reboot` issued to all three | 22:23:06 | T |
| head / worker-1 / worker-2 answering ssh, `ibv_devinfo` 4/4 | — | +29 / +30 / +31 s |
| `harem-exl3.service` state at that moment | `activating` / `active` / `active` | +29 … +31 s |
| `systemd`: `Finished harem-exl3.service` | 22:24:49 / 22:24:44 / 22:24:44 | **+103 / +98 / +98 s** |
| `/health` returns 200 | 22:28:21 | see below |
| GPU KV cache size on that boot | **5,652,892** | |
| gates after it (probe / code exam) | **10/10 · 12/12** | |

**The elapsed figure is printed twice because the log contradicts itself.** The harness wrote
`health 200 +242s`; the wall-clock stamps in the same file give **315 s** between the reboot and the
health check. The instant the harness started its counter cannot be recovered — 242 s before 22:28:21
is 22:24:19, which matches no event in the log — so both are recorded and **315 s is the figure to
plan with**. Neither number changes the decomposition, which is what the test was for: the unit's
`ExecStartPre` plus `ExecStart` occupy **98–103 s** (docker coming up, ConnectX-7 reaching 4/4, the
fabric pings, `drop_caches`, the settle gate), and the container needs **212 s** from the last unit
finishing to a served token, against **251 s** for a fast-load boot on a warm host. Autostart costs
about a minute and a half of fabric wait on top of a boot that is still the weight load.

The pool is the cross-check. **5,652,892 against the 5,619,834** measured on the same configuration
from a settled `docker run`, **+0.6 %** — a reboot is the cleanest baseline a pool number can have, and
it lands well inside the 6 % this figure used to swing by before the settle gate
([docs/07](../../docs/07-kv-and-draft-page.md) §1.1). It is the strongest evidence we have that the
gate measures what it claims to.

### The same test on production 11 and production 12, 6 September 2026

Repeated on production configuration 11 (the same stack at `gpu-memory-utilization` 0.87 with the
sm_12x correctness set) and again on **production configuration 12** (0.88 with the sparse-indexer
K-gather workspace bound, which is what our nodes serve today). One trial each, `reboot` issued to all
three nodes at once `[measured-here]`:

| | production 10, 5 Sep | production 11, 6 Sep | **production 12, 6 Sep** |
|---|---|---|---|
| `/health` 200 from the reboot command | 242 s by the harness counter, **315 s** by the wall clock | **312 s**, wall clock, one figure only | **311 s**, wall clock, one figure only |
| `harem-exl3.service` on all three | active | active | **active, and `enabled`** |
| ConnectX-7 on all three | 4/4 | 4/4 | **4/4** |
| KV pool on that boot | 5,652,892 | **6,382,920** | **7,041,322** |
| gates after it | 10/10 · 12/12 | 10/10 · 12/12 **cold and warm**, tool-call 8/8, needle-lite 6/6 | 10/10 · 12/12 **cold and warm**, tool-call 8/8, needle-lite 6/6 |
| swap traffic during the battery that followed | not sampled | **si + so exactly 0 on all three nodes** | swap **in** exactly 0 on all three; swap-out 5 / 10 / 10 KiB with swap **used** 0.000 GiB |

**311 against 312 against 315 s: the autostart cost has not moved across three configurations and two
memory rungs**, and on the last two there is no contradiction to print, because the driver timed the
reboot command itself rather than trusting a harness counter. The pool cross-check holds at every rung
— 6,382,920 from the reboot against 6,366,391 from a settled `docker run` of the same configuration,
**+0.3 %**; at 0.88 the reboot boot is the **headline** pool figure of production 12, the lowest of its
three boots (7,170,798 / 7,088,154 / 7,041,322) and the one the configuration is published on.

Production 12's row is transcribed from
[`../configs/production-configurations.csv`](../configs/production-configurations.csv) row 12 and
[`../memory/indexer-workspace-ab.md`](../memory/indexer-workspace-ab.md) §7.1; as above, the raw
capture is not included because it is a transcript of `ssh` to three named hosts.

The clean boot was also the campaign's noise check. Production 11's load boot had read C4 at 135.0
against the reference's 144.2, the only number that had moved; the clean boot read **145.9**, above
the reference. A whole-cluster reboot is a slow instrument, but it is the one that settles whether a
single boot's outlier is real.

Unit, preflight and the install order (including `Conflicts=` and disabling the sibling unit):
[`systemd/`](../../systemd/README.md).

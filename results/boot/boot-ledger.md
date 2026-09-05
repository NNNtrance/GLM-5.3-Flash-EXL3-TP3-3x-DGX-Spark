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

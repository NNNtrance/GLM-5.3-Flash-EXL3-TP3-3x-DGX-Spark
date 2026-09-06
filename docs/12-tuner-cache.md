# 12 — The MLA tuner cache, and the measurement tax it was charging

**Applies to: both tracks.** It decides your sweep protocol at either rank count.

The kernel that decodes attention on this stack picks its launch configuration by trying candidates
at first use and keeping the winner. The winner was kept in a **process-local map**, so every boot
re-tuned from nothing, and every batch shape the engine had not seen yet paid for a tune *while
serving*. That is about 15 ms a time — invisible in a day of production and decisive in a benchmark,
because it lands almost entirely in the first rounds of a sweep. It is the reason this repository's
protocol said *five rounds, discard the first two*.

Upstream now persists the cache. We built it, wired it up, measured it, and shortened the protocol.
One environment variable, no patch of ours involved:

```
CUDA_EXL3_TUNE_CACHE=/cache/tune
```

**Settings for everything here:** image `exl3-zeus:9bf594c`, TP=3 + expert parallel, EXL3 4bpw
checkpoint at HF revision `b20c49ba`, KV `fp8`, DFlash2 draft k=7, `--block-size 256`,
`HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, `gpu-memory-utilization 0.80`, per-rank fast-load sidecar, temperature 0,
reasoning effort **low**, three sweep rounds, 5 September 2026. Raw table:
[`../results/boot/tuner-cache.md`](../results/boot/tuner-cache.md).

---

## 1. What the tuner was costing

Two separate costs, and only one of them is about boot time.

**At boot.** A cold engine minted **11 tune events before the server was up** and 18 over a full boot
plus sweep, each about 15 ms and each logged as evicted. That is a few hundred milliseconds — real,
and the least interesting part.

**While serving.** New batch shapes keep arriving. Our sweeps present B = 25, 26, 27, 29, 470 and 768
across C1 to C8, and every unseen one bought a tune in the middle of a timed round. This is what
made a **winning** arm read 25–45 % *worse* on its first pass, once badly enough that we nearly
discarded the change that turned out to be the largest single gain in the stack
([09](09-measurement-protocol.md) §1). The five-round rule was built to skip it, at a cost of about
15 minutes per A/B arm, on a cluster where one arm already takes half an hour.

**In production**, the same cost is paid again after every restart. Small, and not zero.

## 2. What upstream shipped, and what we contributed

`cuda-exl3` commit `9bf594c`, "Persist the MLA tuner cache across processes", adds a file-backed
cache behind `CUDA_EXL3_TUNE_CACHE`. Unset, it changes nothing. Set to a directory, entries load once
at first use and are appended with `O_APPEND` as they are found — atomic at this size on POSIX, so
the three tensor-parallel ranks share one file without a lock; two ranks racing on the same key both
write a valid configuration and the last read wins. The file name embeds the device name and a format
tag, so a different card or a changed candidate grid cannot misread another's entries, and a missed
tag bump costs a slower pick rather than a wrong answer — the entries are launch parameters, and the
kernel is correct for any of them `[reported]`.

Our part was the evidence, not the code: the measurement that the map was process-local and what that
was costing on GB10 — tune events per boot, the shapes that mint new keys at runtime, and the
round-1-versus-round-3 spread it produced — went upstream as an issue with the request, and the
author wrote the implementation. Credit and links in [../CREDITS.md](../CREDITS.md).

## 3. How to turn it on

One directory on the host, bound into the container at `/cache` by `scripts/start-tp3.sh` — the same
mount the Triton, TileLang and FlashInfer caches already use, so there is nothing new to wire:

```
mkdir -p /var/tmp/exl3-zeus-cache/tune
```

Then one token in each node's own environment file, derived with `sed`, never copied between nodes:

```
EXTRA_ENV="HAREM_DISABLE_PERSISTENT_TOPK=1 NCCL_MAX_NCHANNELS=8 HAREM_SW_BLOCK_SIZE=256 CUDA_EXL3_TUNE_CACHE=/cache/tune"
```

The first boot writes the file; every later boot reads it. To check that it took, on the head node:

```
wc -l /var/tmp/exl3-zeus-cache/tune/mla-tune-v1-NVIDIA_GB10.txt
```

18 lines after a boot plus a full sweep, and it should **stop growing** on the second boot. If it
keeps growing, the directory is not surviving the container or the tag changed — either way the
engine is correct and only slower, which is the right failure mode.

## 4. What it bought

Cold cache against warm cache, everything else identical `[measured-here]`:

| | cold cache (first boot) | **warm cache** |
|---|---|---|
| tune events before serving | 18 written | **0** |
| tune events during a 3-round C1–C8 sweep | new keys minted as B moved | **0**, file stayed at 18 lines |
| round 1 → round 3, C8 | 159.2 → 164.8 (**−3.4 %** on round 1) | 163.3 → 159.0 (**+2.7 %**, unordered) |
| round 1 → round 3, C1 | −1.3 % | **−0.1 %** |
| prefill, 3 unseen ~8.4K prompts | 1,672 / 1,685 / 1,708 | 1,662 / 1,709 / 1,719 |
| gates cold and after the benchmark | 10/10 · 12/12 | 10/10 · 12/12 |

**With the warm cache the first round is no longer a penalty; what is left is ±3 % run-to-run noise
in no particular order.** That is what licenses the shorter protocol in
[09](09-measurement-protocol.md) §1: three rounds, median of three, on an image that has this — and
still five rounds with two discarded on one that does not. The saving is about 15 minutes per arm.

Steady-state speed did not move, and was not expected to: warm-cache C1 54.5 and C8 159.9 against the
previous configuration's 54.4 and 161.8, inside the arms' own spread. **The cache does not make the
engine faster. It makes the first measurement of it true.**

## 5. What this cost

| axis | verdict |
|---|---|
| Speed | unchanged at steady state; the change is entirely in round 1 `[measured-here]` |
| Quality | gates 10/10 · 12/12 cold and warm, unchanged `[measured-here]` |
| Memory / KV pool | 4,429,752 against 4,484,848 before — 1.2 %, inside boot-to-boot noise `[measured-here]` |
| Disk | one text file of 18 lines |
| **Boot** | **one dump boot, 682 s** — see below |

The real price was not the cache. Moving to `9bf594c` meant a **new image**, and the fast-load
sidecar's manifest records the image tag, so the existing sidecar was refused at preflight — as
designed ([08](08-fast-boot.md) §4). Regenerating it cost one ~11-minute dump boot on all three
nodes. **Any kernel-image change carries that cost**; it is the standing price of the fast-boot
sidecar and it belongs in the plan for every future image, not just this one.

A second, smaller trap: the dump boot's own KV pool reads 3,958,677 rather than ~4.48M, because
writing 56 GiB per node goes out through the page cache. It comes back on the next boot. Do not
record a dump boot's pool as a result.

## 6. What is still open

1. **The tag is not versioned against our patches.** The cache key covers the device name and the
   upstream format tag. It does not know about `patches/kernel/*` — if we ever ship a kernel patch
   that changes the candidate grid, we have to clear the file by hand. Nothing enforces that today
   `[not tested]`.
2. **The boot saving was not isolated.** Phase 4 of the boot ledger (profile run, first NCCL, MLA
   tune) is 67–73 s and the tune events inside it are a few hundred milliseconds of that. We did not
   re-run the ledger to measure the difference, because the reason for doing this was measurement
   hygiene, not boot time `[not tested]`.
3. **Three rounds is calibrated on one pair of boots.** It rests on one cold/warm comparison at one
   configuration. If an arm's three rounds disagree by more than about 5 %, treat that as a signal to
   go back to five, not as a number.

Next: [10 — Results and roofline](10-results-and-roofline.md), or back to
[09 — Measurement protocol](09-measurement-protocol.md) for where this changed the rules.

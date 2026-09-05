# The persisted MLA tuner cache — cold cache against warm cache

Two boots of the same image (`exl3-zeus:9bf594c`) on the same three nodes, with
`CUDA_EXL3_TUNE_CACHE` pointing at a host-bound directory that survives the container. The first boot
finds the cache file absent and writes it; the second finds it and reads it. Everything else is the
production configuration of 5 September 2026: TP=3 + expert parallel, EXL3 4bpw, KV `fp8`, DFlash2
draft k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`,
`--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, `gpu-memory-utilization 0.80`, per-rank fast-load
sidecar, temperature 0, reasoning effort `low`, three sweep rounds `[measured-here]`.

Narrative in [docs/12](../../docs/12-tuner-cache.md).

| | cold cache (first boot) | **warm cache (second boot)** |
|---|---|---|
| tune events written before serving | 18 entries | **0** |
| tune events during a C1–C8 sweep (3 rounds) | new keys minted as the batch shape moved | **0** — the file stayed at 18 lines |
| round 1 versus round 3, C8 aggregate | 159.2 → 164.8 (**−3.4 %** on round 1) | 163.3 → 159.0 (+2.7 %, i.e. noise) |
| round 1 versus round 3, C1 aggregate | −1.3 % | **−0.1 %** |
| prefill, 3 unseen ~8.4K prompts | 1,672 / 1,685 / 1,708 | 1,662 / 1,709 / 1,719 |
| gates, cold and after the benchmark | 10/10 · 12/12 | 10/10 · 12/12 |
| KV pool | 3,958,677 (this arm also wrote the fast-load sidecar) | **4,429,752** |

The cold-cache arm doubles as the sidecar dump boot for this image, which is why its boot is long
(682 s wall) and its KV pool low — writing 56 GiB per node goes through the page cache. Neither
number belongs to the cache change; the comparison that does is the round-to-round one.

## Per-round detail

Aggregate tok/s per round, in order `[measured-here]`:

| C | cold cache: r1 / r2 / r3 | **warm cache: r1 / r2 / r3** |
|---|---|---|
| 1 | 54.4 / 53.7 / 55.1 | 54.4 / 54.5 / 54.5 |
| 2 | 79.7 / 81.0 / 80.3 | 76.4 / 81.1 / 80.8 |
| 4 | 115.5 / 115.5 / 111.8 | 112.0 / 112.2 / 111.1 |
| 6 | 137.2 / 136.4 / 134.9 | 135.6 / 138.1 / 131.7 |
| 8 | **159.2** / 165.0 / 164.8 | 163.3 / 159.9 / 159.0 |

The cold arm's round 1 at C8 is the last visible trace of the effect the five-round protocol existed
to suppress. In the warm arm the three rounds are unordered noise, which is the whole result.

## What it did not change

Speed at steady state is unchanged — warm-cache C1 54.5 and C8 159.9 against the previous
configuration's 54.4 and 161.8, all inside the arms' own spread. The cache does not make the engine
faster; it makes the **first** rounds honest, which is worth about 15 minutes of cluster time per A/B
because the protocol drops from five rounds to three ([docs/09](../../docs/09-measurement-protocol.md)).

## The cache file

One file per device name, shared by the three tensor-parallel ranks without a lock (`O_APPEND` at this
size is atomic on POSIX; two ranks racing on one key both write a valid configuration and the last
read wins). 18 lines after a full boot plus sweep. The file name embeds the device name and a format
tag, so a different card or a changed candidate grid cannot misread another's entries — and a missed
tag bump costs a slower pick, never a wrong answer, since the entries are launch parameters and the
kernel is correct for any of them `[reported]`.

# Category speed, prefill and mixed load

Settings for everything on this page: three nodes, TP=3 + expert parallel, EXL3 4bpw, KV `fp8`,
DFlash2 draft k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`
(unless the row says 4096), `--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, CUDA graphs on,
`gpu-memory-utilization 0.80`, temperature 0, `reasoning_effort: low`, warm engine.
5 September 2026 `[measured-here]`.

The last two arms — *tuner cache warm* and *dual cable + `PTR_CUDA`* (the production configuration
since) — ran on image `exl3-zeus:9bf594c` with the persisted MLA tuner cache
([docs/12](../../docs/12-tuner-cache.md)) and, for the last one, the patched mesh plugin
([docs/06](../../docs/06-nccl-mesh.md) §6–§8). They were measured with the quick arm of the tiered
protocol ([docs/09](../../docs/09-measurement-protocol.md) §10), which does not run the category or
mixed-load probes — so those two tables stop at the fast-boot arm and say so.

## 1. Decode rate by content type

`scripts/category-speed.py`. Decode rate excludes TTFT. Acceptance is read from the server's own
speculative-decoding counters, differenced across the run.

### Fast-boot arm (image `f4987cf`, draft page 256, fast-load sidecar)

The category probe has **not** been re-run on the two arms after this one `[not tested]`; category
decode is decode-bound and neither change touches the decode kernel path, but that is a reason to
expect no movement, not a measurement of it.

| | C1 decode tok/s | range | TTFT | acceptance | mean tokens |
|---|---|---|---|---|---|
| prose | **22.4** | 17.9 – 28.7 | 0.53 s | **13.0 %** | 668 |
| code | **47.9** | 36.2 – 63.5 | 0.46 s | 45.7 % | 698 |
| math | **59.0** | 46.5 – 71.7 | 0.57 s | 55.9 % | 421 |
| JSON | **57.7** | 43.6 – 69.1 | 0.60 s | 54.9 % | 513 |

| | C4 per stream | C4 total | TTFT | acceptance |
|---|---|---|---|---|
| prose | 12.8 | **43.6** | 0.47 s | 10.6 % |
| code | 31.2 | **98.5** | 0.72 s | 49.4 % |
| math | 39.8 | **106.0** | 0.73 s | 56.5 % |
| JSON | 38.1 | **93.0** | 0.64 s | 53.9 % |

**Prose is 2.6× slower than math at a single stream, and the reason is the draft.** Acceptance in
prose is 13 % against 56 % in math: prose is high-entropy, a 7-deep draft mostly misses, and the
wasted draft work eats the gain. This is a property of speculative decoding, not of this stack — see
[docs/04](../../docs/04-dflash2-port.md) for the k=7 versus k=5 A/B, where k=5 wins in prose alone.

### Two earlier arms, for the shape of the variation

| Arm | prose | code | math | JSON |
|---|---|---|---|---|
| `f4987cf`, draft page 16 | 22.2 | 47.3 | 63.5 | 56.2 |
| `f4987cf` + draft page 256 | 21.7 | 48.5 | 61.0 | 56.6 |
| fast boot S1+S2+S3 | 22.5 | 47.3 | 60.2 | 57.0 |
| **fast-boot sidecar (last arm measured)** | **22.4** | **47.9** | **59.0** | **57.7** |

Category decode is decode-bound, so these arms differ by less than their own run-to-run spread. Do
not read a ranking into this table; read it as a stability check.

## 2. Prefill

Two different measurements, and the difference between them matters — see
[docs/09](../../docs/09-measurement-protocol.md) §3.

| Measurement | What it is | Production value |
|---|---|---|
| `prefill 7k` (`scripts/prefill-7k.py`) | Two fixed seeds, second request reported. **Warm.** Reads whole blocks out of the prefix cache on a repeat. | 1,506 tok/s |
| `prefill-fresh` (`bench/prefill-fresh.py`) | Three ~8.3K prompts the engine has never seen, new seed per request. **This is the honest number.** | 1,742 / 1,792 / 1,797, median **1,792** |

Across the arms `[measured-here]`:

| Arm | prefill 7k | prefill-fresh (median) |
|---|---|---|
| `bc0e0f6`+0003, MNBT 4096 | — | 1,761 |
| `61a17bc` fusion off, MNBT 4096 | 1,354 | 1,710 |
| `61a17bc` fusion auto, MNBT 4096 | 1,356 | 1,711 |
| `f4987cf`, MNBT 2048 | 1,331 | 1,645 |
| `f4987cf` + draft page 256 | 1,469 | 1,508 (first prompt cold at 817; warm repeats 1,699 – 1,709) |
| fast boot S1+S2+S3 | 1,470 | 1,669 |
| fast boot S4 (sidecar) | 1,475 | 1,704 |
| tuner cache warm (`9bf594c`) | 1,445 | 1,709 |
| **dual cable + `PTR_CUDA` (production)** | **1,506** | **1,792** |

`--max-num-batched-tokens 4096` is worth about +9.5 % on fresh prefill and costs 28.5 % of the KV
pool ([docs/07](../../docs/07-kv-and-draft-page.md) §5). The production configuration takes the pool.

## 3. Mixed load

`scripts/mixed-load-probe.py`. One decode stream is already running when a ~7K-token prompt arrives
one second later. This is the only measurement here that shows what the scheduler's chunking and
preemption policy actually costs.

| Arm | decode tok/s while the long prompt lands | 7k TTFT |
|---|---|---|
| `bc0e0f6`+0003, MNBT 4096 | 10.2 | 4.9 s |
| `f4987cf`, MNBT 2048 | 7.5 | 5.0 s |
| `f4987cf` + draft page 256 | 6.9 | 4.9 s |
| **fast boot S4 (last arm measured)** | **7.0** | **4.9 s** |
| tuner cache warm, dual cable + `PTR_CUDA` | not run `[not tested]` | not run `[not tested]` |

A stream running alone does about 50 tok/s, so a concurrent 7K prefill costs it roughly 85 % of its
rate for the duration. `--max-num-batched-tokens 4096` is the lever that improves this (10.4 tok/s
and 4.8 s TTFT in a clean A/B) and it is not taken, for the pool.

## 4. Single-stream cold and warm

`scripts/cold-warm-c1.py`. First request after a boot, then two repeats.

| Arm | cold TTFT | cold decode | warm decode | warm decode (2) | acceptance |
|---|---|---|---|---|---|
| `f4987cf`, MNBT 2048 | 1.38 s | 45.5 | 43.9 | 41.8 | 40 – 44 % |
| `f4987cf` + draft page 256 | **0.79 s** | 45.9 | 45.8 | 45.5 | 44 % |
| fast boot S4 (sidecar) | 0.79 s | 44.2 | 45.3 | 43.2 | 41 – 44 % |
| tuner cache warm (`9bf594c`) | 0.77 s | 42.3 | 47.5 | 45.7 | 40 – 47 % |
| **dual cable + `PTR_CUDA` (production)** | **0.72 s** | 45.1 | 44.1 | 44.7 | 40 – 42 % |

The cold TTFT halves with the draft page change, which is the 16× smaller draft block table showing
up ([docs/07](../../docs/07-kv-and-draft-page.md) §3).

Note that this probe's prompt set differs from the concurrency sweep's, so its C1 decode rate
(43–46 tok/s) is not comparable with the sweep's per-stream figure (61.8 tok/s). Compare each probe
against itself.

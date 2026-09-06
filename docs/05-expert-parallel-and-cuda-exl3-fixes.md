# 05 — Expert parallelism and the cuda-exl3 fixes

**Applies to: both tracks.** Expert parallelism is mandatory at three ranks and optional at two,
where it is never measured `[not tested]`; §3.5, the GB10 top-k overlay, applies at any node count.

Expert parallelism is not optional here: at TP=3 the EXL3 trellis cannot be sliced, so the experts
must be distributed whole ([03](03-tp3-padding-and-sidecars.md) §1). When we first turned it on, the
three-node arrangement was **8–29 % slower than two nodes**. This page is what that turned out to be.

The short version: it was a **one-line omission in the kernel's GEMM dispatch**, three of our four
hypotheses were wrong, one number we published as "under 1 %" was 3–16 %, one patch of ours was on
the wrong side of the argument and has been retired, and the only thing that survived contact with a
careful measurement is a small shared-memory change to the combine kernel.

Everything decisive on this page was measured **model-free**, with the engine down, on a
micro-benchmark using the real GLM-5.3-Flash shapes. Everything we got wrong, we got wrong from
two engine sweeps. That is the methodological point of the whole document
([09](09-measurement-protocol.md)).

> **At TP=2, expert parallelism is optional.** 2,048/2 = 1,024 = 8 × 128, so the routed experts
> tensor-slice cleanly and `preflight-tp3.py` accepts `--ep 0` at two ranks while refusing it at
> three. Every TP=2 arm we ran had EP **off**; EP on at two ranks is `[not tested]`, and the launcher
> hard-codes `--enable-expert-parallel` in three places that have to be edited first.
> [15](15-tp2-track.md) §1.1 and §2.2.

---

## 1. The regression

Same field, same prompt set, same settings, two sweep rounds each `[measured-here]`:

| | TP=2 + DFlash2 | TP=3 + EP, stock kernels |
|---|---|---|
| C1 total tok/s | 42.91 | 39.57 (−7.8 %) |
| C2 | 60.80 | 50.74 (−16.5 %) |
| C4 | 83.89 | 59.32 (−29.3 %) |
| C6 | 98.08 | 73.41 (−25.2 %) |
| C8 | 114.60 | 92.01 (−19.7 %) |
| prefill 7k | 1,035 tok/s | 983 (−5.0 %) |
| acceptance / tokens per step | 62.4 % · 5.37 | 62.9 % · 5.36 |

Three facts framed the search. It was **not memory**: KV usage sat at 2–13 % across the whole
benchmark with the queue always empty. It was **not the drafter and not correctness**: acceptance and
tokens-per-step were identical to three significant figures, which is where a broken kernel or a
mis-padded head shows up first. And the shape of the loss — worst in the middle, easing at both ends
— pointed at something batch-dependent.

---

## 2. The root cause: `n_rows` is not passed on the unsplit MoE launch

In the kernel's GEMM dispatch macro, the split branch passes the live-row bound and the unsplit
branch does not, so it takes the `nullptr` default:

```c
if (split > 1)
    launch<..., true,  ...>(..., expert_ids, b_expert_stride, svh_expert_stride, n_rows);
else
    launch<..., false, ...>(..., expert_ids, b_expert_stride, svh_expert_stride);
                                                                      // ^ n_rows missing
```

The MoE path is unsplit by default — `rows * n_total` exceeds the accumulator cap at every batch we
serve — so the kernel's guard

```c
if (n_rows && m0 >= *n_rows) return;
```

is dead there, and a block is retired only when `expert_ids[blockIdx.y] < 0`.

That would be harmless if the surplus entries were `-1`. They are. `moe_align_block_size` sizes
`expert_ids` for the worst case and marks the tail `-1` correctly. **The conversion happens one step
later**: vLLM maps the whole array through `expert_map`, and `-1` indexes the *last element* of that
map. On the rank that owns the top of the global expert range, the last element is a live local
expert — so on that rank, and only that rank, every surplus block runs a full GEMM, reading a real
expert's whole trellis and writing rows nothing ever gathers.

At M=2048 the surplus is 13,312 rows of 34,560 — **38 % of the grid**. Under expert parallelism the
step waits for the slowest rank, so all three ranks pay for one rank's extra work.

### Isolating it

The behaviour follows the *last live block*, not the workload. Measured on one MoE layer, 96 local
experts of 288, top-8 `[measured-here]`:

| owner slice | first live block | last live block | w13 µs | w2 µs |
|---|---|---|---|---|
| experts 0–95 | 0 | 111 | 5,613 | 3,305 |
| experts 48–143 | 55 | 166 | 5,536 | 3,341 |
| experts 96–191 | 112 | 219 | 5,565 | 3,267 |
| experts 144–239 | 167 | 277 | 5,562 | 3,382 |
| **experts 192–287** | 220 | **331** | **10,506** | **5,526** |
| strided 0, 3, 6, … | 0 | 329 | 5,590 | 3,311 |
| strided 1, 4, 7, … | 2 | 330 | 5,660 | 3,275 |
| **strided 2, 5, 8, …** | 3 | **331** | **10,552** | **5,581** |

Owning **only the top expert** — one real block, 60 routed pairs — costs 6,416 µs on w13, while
owning only the expert below it costs 1,454 µs, and owning 111 experts but not the top one costs
5,630 µs. Forcing the tail of `expert_ids` to `-1` by hand took w13 from 10,567 → 5,655 µs and w2
from 5,545 → 3,339 µs: the same fix seen from the other side.

### The fix, and what it is worth

One line: pass `n_rows` in the `else` branch. Per-rank MoE stage at M=2048 — this is what a step
actually waits for `[measured-here]`:

| | rank 0 | rank 1 | rank 2 | step cost (= max) |
|---|---|---|---|---|
| before | 11,278 µs | 11,218 µs | **18,401 µs** | 18,401 |
| **+ `n_rows`** | 10,072 | 9,915 | **10,107** | **10,107 (−45 %)** |

After the fix all eight expert placements measure 5,027–5,116 µs on w13: the map's position stops
mattering at all.

**This is upstream now**, in `cuda-exl3` commit `a95e809`. If you build a kernel older than that,
carry the line — without it the rank owning the top of the range pays 1.2–2.0×, and that is the whole
result.

### Retraction: it does not cost the non-EP path

We first reported that this also costs a tensor-parallel arrangement, on the reasoning that with no
map the tail is a valid local expert everywhere. **That is wrong** `[retracted]`. Running the
alignment directly on our own build, E=288, top-8:

| | M=8, block 16 | M=64, block 16 | M=2048, block 64 |
|---|---|---|---|
| surplus blocks | 8 of 64 | 56 of 302 | **213 of 540 (39.4 %)** |
| tail of `expert_ids`, map given | all `95` | all `95` | all `95` |
| tail of `expert_ids`, `expert_map = None` | **all `-1`** | **all `-1`** | **all `-1`** |

The alignment marks the tail correctly; only `expert_map[expert_ids]` converts it, by negative
indexing. With no map there is no conversion and no cost. The kernel author's reading was right and
ours was wrong.

---

## 3. Three hypotheses, measured and refuted

All measured model-free on an idle GB10 with the real shapes, before anything was changed.

### 3.1 "Expert parallelism's block padding is far worse at low rows per expert" — refuted

The alignment buckets against the **global** 288 experts in both arrangements, so the padded row
count is bit-identical; expert parallelism differs only in retiring two thirds of the blocks. Padded
to real rows, both arms: M=8 **14.75×**, M=64 7.56×, M=512 1.40×, M=2048 1.33×. Per-layer time on one
rank, before any fix: EP 1,226 / 5,116 / 6,983 / 10,912 µs against a tensor-sliced 288-expert rank's
1,896 / 7,647 / 9,785 / 14,637 — **EP was already 0.65–0.75× the cost** `[measured-here]`.

The `block_m` ladder, measured:

| M (tokens) | rows = M·8 | ladder from local E=96 | ladder from global E=288 | µs @16 | @32 | @64 | @128 | best |
|---|---|---|---|---|---|---|---|---|
| 8 | 64 | 16 | 16 | **1,255** | 1,435 | 1,824 | 2,772 | 16 |
| 64 | 512 | 16 | 16 | **5,111** | 6,035 | 7,682 | 11,375 | 16 |
| 512 | 4,096 | 32 | **16** | **7,291** | 7,918 | 9,795 | 14,317 | 16 |
| 2048 | 16,384 | 128 | **64** | 12,466 | **11,606** | 11,834 | 15,676 | 32 |

The global count is right at M=8/64/512 and within 2 % at M=2048; the local count would have cost
**+9 % at M=512 and +32 % at M=2048**. Upstream's choice (`77513d2`) is confirmed on this hardware.

### 3.2 "Three-way collectives" — refuted at decode, small at prefill

All-reduce over the mesh plugin, one process per node, the engine's exact NCCL environment
`[measured-here]`:

| | 8 tok (64 KB) | 2048 tok (16 MB) | 90 × 8 tok, i.e. one decode step |
|---|---|---|---|
| 3 nodes, `Ring` | 53 µs | 2,030 µs | **3.9–4.1 ms** |
| 3 nodes, `Tree` | 97 µs | 8,487 µs | 6.8 ms |
| 3 nodes, auto | 62 µs | 1,832 µs | 3.9 ms |
| 2 nodes, `Ring` | 81 µs | 1,441 µs | 4.7 ms |

Three nodes are **not** slower than two at decode message sizes, and `Ring` is the right pick —
`Tree` is 1.7× worse. At the 16 MB prefill size three nodes cost 41 % more per all-reduce, about
+53 ms on a ~1.9 s chunk (~3 %), which matches the 5 % prefill gap. The collectives *were* costing
real time, but for a different reason entirely — see [06](06-nccl-mesh.md).

### 3.3 "The extra masking pass is under 1 %" — refuted; it was 3–16 %

Our own estimate, published in an earlier report, assumed the pass covered the *routed* rows. It
covered the **allocated** rows — 1,024 for 64 real ones at M=8, and 34,560 for 16,384 at M=2048.
Measured share of the MoE layer: **2.9 % (M=8), 4.9 % (M=64), 6.0 % (M=512), 15.8 % (M=2048, 1.72
ms)** `[retracted]`. It is gone in current upstream.

### 3.4 Routing imbalance — true only at M=1

After the `n_rows` fix the per-rank spread is 1.02–1.07× at M=16/64/2048, and expert parallelism is
*faster* than the tensor arrangement at every batch except M=1 (0.94–0.96× at M=16/64, 0.83–0.86× at
M=2048). At **M=1** the spread is 2.05× and does not move: with top-8 over three ranks the mean rank
owns 2.67 experts, the step waits for the one that drew 4, and under EP each owned expert costs the
full column width. That is inherent to expert parallelism at single-token batches, not a defect.

It matters less than it sounds for this stack: with a k=7 draft a decode step verifies 8 tokens per
sequence, so the real decode batch is M=8 at one stream and M=64 at eight — both in the region where
expert parallelism wins.

### 3.5 The GB10 top-k fallback is not a cost — it is a win

`HAREM_DISABLE_PERSISTENT_TOPK=1` pins the sparse-attention indexer to the decode fallback. It was
adopted as a workaround for a shared-memory failure and turns out to be the faster path at our
shapes `[measured-here]`, µs:

| rows | pools | `top_k_per_row_decode` | `persistent_topk` | `torch.topk` |
|---|---|---|---|---|
| 8 | 512 | **5.0** | 7.0 | 6.3 |
| 8 | 2,048 | **4.7** | 10.3 | 14.4 |
| 8 | 4,096 | **6.2** | 10.3 | 18.5 |
| 8 | 16,384 | 18.5 | **14.4** | 49.1 |
| 64 | 2,048 | **6.2** | 14.4 | 24.2 |
| 64 | 32,768 | **31.7** | 53.6 | 129.0 |

The persistent kernel does not fail on this hardware; it is simply slower below about 16K candidate
pools, which is roughly 64K tokens of context, and the selected sets match `torch.topk` exactly. At
11 indexer layers × ~5 µs it is 0.04 % of a single-stream decode step. The gate affects the **decode**
path only. **Writing a GB10-capable persistent top-k would buy nothing measurable at our context
lengths** — that task is closed unless we serve ≥64K contexts, where the crossover puts the
persistent kernel ahead.

The overlay that installs the gate is `patches/indexer-overlay/`.

---

## 4. What the profile says the money is actually going on

Torch profiler, rank 0, one 8,273-token prompt the engine had never seen, 6.55 s `[measured-here]`.

**GPU busy 6,453 ms of a 6,529 ms span — 98.8 % occupancy.** Prefill is not launch-bound and not
CPU-bound; there is 1.2 % of gap in the whole window. Any win has to come out of a kernel.

| class | ms | % GPU | calls |
|---|---|---|---|
| MoE EXL3 GEMM | 1,711 | **26.5 %** | 504 |
| BF16 dense GEMMs (cutlass / nvjet / tilelang / deep_gemm) | ~1,300 | **~20 %** | ~2,200 |
| NCCL all-reduce | 1,411 | **21.9 %** | 612 |
| MLA sparse attention | 584 | 9.0 % | 1,998 |
| MoE input transforms | 343 | 5.3 % | 504 |
| indexer logits | 216 | 3.3 % | 650 |
| norm / elementwise | 186 | 2.9 % | 6,377 |
| KDA scans + causal conv | 221 | 3.4 % | 1,428 |
| MLA autotuner eviction pass | 166 | 2.6 % | 1,728 |
| MoE combine and inverse-index build | 61 | 0.9 % | 504 |
| MoE align / sort | 3 | 0.1 % | 504 |
| quant / cast, sampling, other | ~250 | ~3.9 % | — |

Decode, same boot, 48 tokens at one stream: BF16 dense GEMMs **~37 %**, MoE EXL3 GEMM 29.3 %, NCCL
all-reduce 23.9 %, everything else ~10 %. GPU busy exceeds the wall span because streams overlap, so
decode is not launch-gap-bound either.

**Three readings.**

1. **The unquantized half of the model is the largest single item at decode.** The checkpoint
   quantizes routed experts only, so attention, KDA, the shared expert and `lm_head` run BF16 through
   `wmma` kernels at 16×16 and 32×32 tiles. Nothing in the EXL3 kernel library touches that. It is
   the largest remaining structural item on this stack ([01](01-model-and-license.md) §3.2).
2. **All-reduce is 22–24 %** — that is [06](06-nccl-mesh.md), and it was worth +13 % at C8.
3. **The MLA autotuner runs during serving.** 1,728 eviction launches inside a 6.5 s prefill. We
   first read this as "a new shape per chunk, re-tuning continuously" and proposed disabling the
   tuner. Measured properly, it is **4–5 tune events × ~350 eviction calls each** — a one-off cost
   that settles as batch shapes repeat `[retracted]`. Disabling tuning was therefore not worth a
   boot; it would remove a one-off cost and risk a worse schedule for the whole run.

---

## 5. What we changed in the kernels, and what happened to each

| Change | Status |
|---|---|
| Pass `n_rows` on the unsplit MoE launch | **Upstream** (`a95e809`). Retired from our tree. |
| `exl3_moe_had_in`: retire a remote block *before* writing its padding zeros | **Upstream** (`a95e809`). Worth 1,171 → 876 µs at M=2048 and 61 → 14 µs at M=8, per MoE layer `[measured-here]`. |
| Expert-parallel-aware `exl3_moe_combine` (optional `expert_ids`/`block_m`, skip a retired block's rows) | **Upstream** (`f906f00`), arrived at independently. |
| Gemm **zeroes** the retired tile instead of returning | **Withdrawn — it was the wrong choice.** See below. |
| Combine stages the per-(token, k) facts in shared memory | **Kept.** `patches/kernel/0003-combine-smem-staging-on-a95e809.patch` |
| Mesh plugin `min_rnr_timer` | Built, tested, **not deployed** — [06](06-nccl-mesh.md). `patches/kernel/0004-min-rnr-timer.patch` |

`patches/kernel/0002-RETIRED.md` records the retirement in writing rather than deleting it quietly.

### 5.1 Our zeroing patch was worse than upstream's design

We had the gemm write a full-width zero tile for every block it retired. Upstream returns and lets
the combine skip those rows. Measured per MoE layer per rank, mean of two routing draws, 50 timed
iterations after 10 warm-ups `[measured-here]`:

| | our zeroing gemm | upstream | delta |
|---|---|---|---|
| gemm w13, M=2048 | 5,089.5 µs | **4,534.6 µs** | −10.9 % |
| gemm w2, M=2048 | 2,822.1 µs | **2,295.9 µs** | −18.6 % |
| whole MoE layer, M=2048 | 9,613 µs | **8,587 µs** | −10.7 % |
| whole MoE layer, M=64 | 4,984 | **4,804** | −3.6 % |
| whole MoE layer, M=8 | 1,110 | **1,067** | −3.9 % |

End to end on three nodes, two sweep rounds each, gates full marks cold and warm on both: C1
47.7/48.3 → 51.0/50.6, C2 63.7/71.7 → 72.3/73.7, C8 118.4/137.2 → 136.6/134.7. We retired our patch
and adopted upstream `[measured-here]`.

**And there is a second retraction underneath that one.** An earlier report of ours recorded that an
upstream build carrying this design measured ~10 % *slower* end to end and called it an unexplained
regression. It does not reproduce. What we had measured was boot-to-boot and warm-up variance — the
same image, the same environment file, two separate boots, five sweep rounds:

| | boot 1 round 1 | boot 1 round 2 | boot 2 round 1 |
|---|---|---|---|
| C1 | 47.71 | 48.33 | 49.62 |
| C8 | **118.44** | **137.23** | **135.56** |

**15.9 % spread on C8 with nothing changed at all** `[retracted]`. A 10 % gap on this stack is inside
the noise of a single sweep pair, and we drew a kernel conclusion from one. That is why the
measurement protocol in [09](09-measurement-protocol.md) now says five rounds, discard two.

### 5.2 The one patch that survived: shared-memory staging in the combine

`exl3_moe_combine_kernel`'s inner loop re-reads three things that do not depend on the loop variable:
the gathered row index, its routing weight, and the liveness test. The loop body runs
`ceil(H / blockDim)` times per thread — four times at hidden 4096 with 1024 threads — the liveness
test is a dependent chain (load the index, integer-divide by a runtime `block_m`, load `expert_ids`,
compare), and the grid is one block per token, so at a decode batch of 8 only 8 of the 48 SMs are
busy and there is nothing to hide the chain behind. Staging all three once, in shared memory, before
the loop:

| M | `block_m` | upstream | **+ staging** | change |
|---|---|---|---|---|
| 8 | 16 | 11.6 µs | **7.7 µs** | **−34 %** |
| 64 | 16 | 16.2 µs | **10.3 µs** | **−36 %** |
| 2048 | 64 | 363 µs | **317 µs** | **−13 %** |

The combine is a small share of the layer, so this is about 1 % of the layer — but M=64 is exactly
the decode batch at eight concurrent streams with a k=7 draft, and 43 MoE layers × 5.9 µs is
**0.25 ms off every decode step**. End to end, two rounds each: **C6 114.3 → 118.7 (+3.8 %), C8
135.7 → 140.8 (+3.8 %)**; C1/C2/C4 within noise `[measured-here]`.

Tests on the built image: the upstream suite (44 passed, 41 skipped); a NaN-poison test showing the
GEMM provably never reads a retired block's activations; bitwise equivalence against the
mask-then-combine reference with retired rows NaN-poisoned; and, with expert parallelism off, the
six-argument call bitwise identical to the four-argument one. The full description is in
`patches/kernel/0003-PR-DESCRIPTION.md`.

---

## 6. Where TP=3 ended up

Same field, same prompt set, `gpu-memory-utilization 0.80` on both arms `[measured-here]`:

| | TP=2 (reference) | TP=3 + EP, stock | **TP=3 + EP, fixed kernels** | vs stock |
|---|---|---|---|---|
| C1 total tok/s | 42.91 | 40.76 | **49.35 / 50.02** | **+22 %** |
| C2 | 60.80 | 51.59 | **71.58 / 72.41** | **+40 %** |
| C4 | 83.89 | 59.22 | **97.00 / 102.23** | **+67 %** |
| C6 | 98.08 | 72.08 | **117.82 / 116.90** | **+59 %** |
| C8 | 114.60 | 91.87 | **137.22 / 140.89** | **+51 %** |
| TTFT C1 / C8 (s) | 0.775 / 2.11 | 0.910 / 2.450 | **0.713 / 1.794** | −23 / −27 % |
| prefill 7k tok/s | 1,035 | 1,025 | **1,257** | **+23 %** |
| acceptance / tokens per step | 62.4 % · 5.37 | 61.6–65.2 % · 5.24–5.56 | 60.6–63.6 % · 5.24–5.46 | same |
| KV pool | 825,000 | 2,587,828 | **2,571,230** | — |

TP=3 + EP now beats TP=2 on every axis (+15 % C1 through +21 % C8 and +21 % prefill), which it did
not before, and it keeps the 3.1× KV pool. The production numbers on the current stack are higher
still — [10](10-results-and-roofline.md).

---

## 7. Reproducing any of this

The model-free harness is in `bench/`. The engine must be down.

```
python3 bench/ep_surplus.py 2048
```

```
CASES=offsets python3 bench/ep_mapoffset.py 2048
```

```
python3 bench/ep_rankstage.py 2048
```

```
python3 bench/ep_moe_bench.py --ms 8,64,512,2048 --out /cache/ep_moe_bench.json
```

```
python3 bench/moe_stage_bench.py --ms 8,64,2048
```

```
python3 bench/topk_bench.py
```

`bench/validate.sh <image-tag>` runs the whole battery — the upstream pytest suite, the poison and
equivalence checks, per-rank symmetry, the expert-map placement sweep and the `block_m` table — in
one pass. Run it against any kernel build before you let it near the engine.

A note on the expert-parallel micro-benchmark's arguments: it needs an intermediate width such that
`2 × inter / tp` is a multiple of 16 for its tensor-parallel arm, so `--inter 2048 --tp 3` fails with
a shape mismatch. Use `--inter 2304` for that arm and read the expert-parallel rows as the ones that
describe our configuration.

---

## 8. What is still open here

- **`block_m` under expert parallelism.** The alignment needs the global expert count and the
  heuristic is about rows per expert; the two uses may want different numbers. Worth a sweep
  `[not tested]`.
- **The gemm still allocates for the worst case** — 283 MB per call at M=2048, of which 38 % is never
  written. Harmless with a caching allocator, and it is why the surplus tail exists at all; sizing
  the output from `n_rows` would need a device-side shape.
- **A 2,304-padded tensor-sliced arrangement** was designed as an alternative to expert parallelism
  and, on the fixed kernel, loses everywhere except M=1 (at M=2048, expert parallelism is 14 %
  faster) while costing +12.5 % expert bytes straight out of the KV pool. The sidecar exists and is
  not used. Full detail in [11](11-open-issues.md).
- **The BF16 half of the model**, which no kernel work in this library can reach.

The upstream defects on this page were filed with the kernel project and are fixed there; see
[../CREDITS.md](../CREDITS.md) for the commits.

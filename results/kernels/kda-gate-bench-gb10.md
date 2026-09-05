# The same gate bench, on the target GPU, cold: the ratios that replace the workstation ones

[`kda-gate-bench.md`](kda-gate-bench.md) closed the question of quantizing the 113 remaining BF16
linears ourselves, on a workstation GPU, with the sentence **"on the KDA shapes EXL3 is 1.58–1.76×
slower than BF16 at M=8"**. That sentence is **withdrawn** `[retracted]`. It is a *warm* number — it
reproduces on GB10 only when the weight is held resident in L2 — and production never runs in that
regime. Re-measured cold on GB10, the same shape reads **1.023**.

**The closure survives; its reason does not.** The KDA gating arms still stay BF16, but because
quantizing them is worth **+0.05 ms/step — neutral** — not because it costs 0.58 ms/step. The whole
gain lives in `kv_b_proj`, which still needs a kernel nobody has written. Full narrative:
[docs/11](../../docs/11-open-issues.md) §2.25 and
[docs/13](../../docs/13-full-scope-checkpoint.md) §4.4. The rule this produced is in
[docs/09](../../docs/09-measurement-protocol.md) §4.2.

## Why this run happened

The `cuda-exl3` author found the same artefact in his own published table
([issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5), 5 September) and withdrew a claim from his
README over it `[reported]`: timing one weight repeatedly keeps a 12–33 MB tensor resident in a
128 MiB L2 and erases the bandwidth advantage the trellis exists for. Rotating over enough distinct
weights moved his 4096×4096 at M=8 from **1.13 to 0.68**, and his 4096×11008 from **0.97 to 0.45**.
He asked us to run the same check on GB10. We did, on the real target, and it corrects us rather
than him.

## Settings

`[measured-here]`, 5 September 2026, 23:13–23:20 Istanbul.

- **GPU:** NVIDIA GB10, cc 12.1, **48 SMs**, **L2 = 24.0 MiB** (25,165,824 B), torch 2.13.0+cu130.
  *(The workstation bench ran on 101 MB of L2 and 170 SMs; the kernel author's card has 128 MiB.)*
- **Image:** `exl3-zeus:754421f` — the production 9/10 image, built for `sm_121`. `ops.backend()`
  reports **`native`**, so what is timed is `cuda-exl3`'s own `exl3_gemm_m`, not an `exllamav3`
  fallback.
- **Call path:** `ops.exl3_linear(x, trellis, suh, svh, [n], cb=2)` — `cb=2` is `mul1`, the production
  checkpoint's codebook. This is exactly what the engine calls at decode.
- **bf16 denominator:** `F.linear` (TN, `W` is `(n,k)`) — what vLLM runs, and what the workstation
  table used. `torch.matmul` on an `(k,n)` copy is measured in the same file as a cuBLAS-path
  control; it is **not** the denominator, because at narrow outputs it is 6–7× worse.
- **Shapes:** the TP=3 per-rank widths, unchanged from the workstation bench, plus the kernel
  author's two control shapes.
- **Bits:** 4, 5 and 6. The production checkpoint stores dense attention and shared-expert modules at
  6 bit, the three dense-MLP layers at 5, the MoE experts at 4.
- **M:** 1, 8 and 64. **M = 1,792 (prefill) was not re-measured** — see the last section.
- **Timing:** CUDA graph of `G = max(200, N)` back-to-back calls, 20 pre-capture warm-ups and 2 warm-up
  replays, then **3 timed replays** with CUDA events, median. Round-to-round spread is typically
  under 1 %; every round is in the raw log.
- `CUDA_EXL3_SPLIT_TARGET` and `CUDA_EXL3_SPLIT_BUDGET` left unset.
- **Where:** one node only, in a throwaway `--rm` container beside the live engine, which was idle
  (`num_requests_running` 0, `/health` 200) and was **not stopped**. Peak GPU allocation 1.47 GiB,
  total run about 90 s. Nothing on the cluster changed: no env file, no checkpoint, no image.

Reproduce: `bench/kda_gate_bench_gb10.py --m 1 8 64 --bits 4 5 6 --rounds 3`. It needs `cuda_exl3`
importable and nothing else.

## Warm and cold, and why the definition is the finding

Every shape is timed three ways over one and the same allocated weight bank, so the two arms rotate
through **identical index patterns**:

| arm | index | what it measures |
|---|---|---|
| `warm` | always 0 | the weight is L2-resident — the artefact |
| `coldA` | `j % N_bf16`, `N_bf16 = ⌈4·L2 / bf16 bytes⌉` | the **bf16** arm alone is ≥ 4× L2; at 4 bit the trellis bank is then only about 1× L2 |
| `coldB` | `j % N_full`, `N_full = ⌈4·L2 / 4-bit bytes⌉` | **both** arms ≥ 4× L2 — the strict reading, and **the number in every table below** |

`coldA` and `coldB` differ by **≤ 5 %** everywhere, so the verdict does not depend on which N is
chosen; the raw log carries both. Because the graph rotates continuously, a given weight is re-touched
only after a full bank sweep — at least 96 MiB of other traffic.

## The ruler was checked first

[`gb10-coldbench/00-ruler-gb10.txt`](gb10-coldbench/00-ruler-gb10.txt) `[measured-here]`:

| check | result |
|---|---|
| bf16 arm, large shape (512×32,768), cold bank | **239–240 GB/s**, inside this card's independently measured 225–241 GB/s ([docs/10](../../docs/10-results-and-roofline.md) §4.1) — the denominator is healthy |
| EXL3 shard-split consistency, fused `[n]` against two `[n/2]` | relative error **0.00** at 4 and 6 bit; 4.8e-3 only on 4096×4096, where split-k changes the reduction order |
| EXL3 activation dtype invariance, bf16 against fp16 | ~3–4e-2 = one bf16 ULP |
| EXL3 output | finite and non-degenerate on every shape |
| **kernels launched per `exl3_linear` call** (torch profiler, inside the image) | `k=128 n=2816` and `k=512 n=32768`: **2** (`exl3_had_in_kernel` + `exl3_gemm_m_kernel`, split-k template `false`). `k=4096` shapes: **3** (+ `exl3_epilogue_kernel`, split-k `true`) |

**One limit of the ruler, stated because it changes how its numbers may be used:** its first section
uses an eager Python loop, which on small shapes is CPU-dispatch-bound (35 µs for a 0.72 MB weight).
The measurement proper is therefore in graph mode, and the ruler's *absolute* small-shape figures are
not used anywhere. What anchors the small shapes instead is the production trace, below.

## M = 8 — a decode step with the k=7 drafter

µs per call, CUDA graph on, median of 3. `ratio = exl3 / bf16(F.linear)`; **above 1 means EXL3 is
slower**. This is the table posted to
[issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5) `[measured-here]`:

| shape (per rank) | k × n | N | GB10 warm | **GB10 cold** | published (workstation) |
|---|---|---:|---:|---:|---:|
| `f_b_proj` / `g_b_proj` | 128 × 2,816 | 559 | 1.605 | **1.023** | 1.596 |
| `f_b_proj` unsharded | 128 × 8,192 | 192 | 1.479 | **0.643** | 1.254 |
| `f_a` / `g_a` / `b_proj` (replicated) | 4,096 × 128 | 384 | 1.27 | **0.853** | 1.17 |
| `in_proj_fg_a` | 4,096 × 256 | 192 | 1.139 | **0.655** | 1.060 |
| `kv_b_proj` (replicated) ‡ | 512 × 32,768 | 12 | 0.129 | **0.291** | 0.391 |
| `kv_b_proj` (22 heads per rank) ‡ | 512 × 11,264 | 35 | 0.682 | **0.349** | 0.747 |
| control 4096 × 4096 | | 12 | 0.144 | **0.272** | — (his card: 1.13 warm / 0.68 cold) |
| control 4096 × 11008 | | 5 | 0.133 | **0.254** | — (his card: 0.97 / 0.45) |

‡ **There is no per-head batched EXL3 kernel for this shape** in `754421f`, verified by source scan
and unchanged since the workstation bench. It is measured as a plain GEMM; the number is valid *if
the kernel is written*.

Row 3 groups three shapes that the raw file keeps apart, at `f_a`'s value. Per shape, at 4 bit, with
the microseconds `[measured-here]`:

| shape (TP=3 per rank) | k | n | N | bf16 warm | bf16 cold | exl3-4b warm | exl3-4b cold | ratio warm | **ratio cold** | cold 5b | cold 6b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `f_b_proj` | 128 | 2,816 | 559 | 2.21 | 4.99 | 3.54 | 5.10 | 1.605 | **1.023** | 1.062 | 1.158 |
| `g_b_proj` | 128 | 2,816 | 559 | 2.20 | 4.98 | 3.54 | 5.11 | 1.613 | **1.025** | 1.066 | 1.163 |
| `f_b_proj` unsharded | 128 | 8,192 | 192 | 3.33 | 9.94 | 4.93 | 6.39 | 1.479 | **0.643** | 0.709 | 0.755 |
| `f_a_proj` replicated | 4,096 | 128 | 384 | 4.58 | 8.49 | 5.82 | 7.25 | 1.270 | **0.853** | 1.091 | 0.937 |
| `g_a_proj` replicated | 4,096 | 128 | 384 | 5.04 | 8.47 | 5.82 | 7.23 | 1.155 | **0.853** | 1.088 | 0.939 |
| `in_proj_fg_a` (f_a + g_a) | 4,096 | 256 | 192 | 5.27 | 12.25 | 6.01 | 8.02 | 1.139 | **0.655** | 0.712 | 0.749 |
| `in_proj_bfg_a` **as production runs it** | 4,096 | 278 | 177 | 6.52 | 15.51 | — | — | — | — | — | — |
| `in_proj_b` (would stay bf16) | 4,096 | 22 | 2,235 | 4.06 | 5.63 | — | — | — | — | — | — |
| `b_proj` replicated | 4,096 | 128 | 384 | 4.62 | 8.47 | 5.86 | 7.26 | 1.270 | **0.857** | 1.047 | 0.950 |
| `kv_b_proj` replicated ‡ | 512 | 32,768 | 12 | 136.61 | 139.33 | 17.64 | 40.56 | 0.129 | **0.291** | 0.324 | 0.384 |
| `kv_b_proj`, 22 heads per rank ‡ | 512 | 11,264 | 35 | 12.24 | 47.44 | 8.34 | 16.58 | 0.682 | **0.349** | 0.404 | 0.493 |
| control 4096 × 4096 | 4,096 | 4,096 | 12 | 121.33 | 144.46 | 17.44 | 39.31 | 0.144 | **0.272** | 0.333 | 0.389 |
| control 4096 × 11008 | 4,096 | 11,008 | 5 | 371.69 | 376.97 | 49.46 | 95.88 | 0.133 | **0.254** | 0.312 | 0.372 |

`g_a_proj` warm reads 1.155 where `f_a_proj` reads 1.270; the grouped row upstream carries the `f_a`
figure. Both cold to 0.853.

## M = 1, 8 and 64 — 4-bit cold ratio

| shape | M=1 | **M=8** | M=64 |
|---|---:|---:|---:|
| `f_b_proj` / `g_b_proj` | 1.423 | **1.023** | 0.995 |
| `f_b_proj` unsharded | 0.645 | **0.643** | 0.951 |
| `f_a` / `g_a` / `b_proj` replicated | 1.192 | **0.853** | 1.63 |
| `in_proj_fg_a` | 0.810 | **0.655** | 1.027 |
| `kv_b_proj` replicated ‡ | 0.307 | **0.291** | 0.388 |
| `kv_b_proj`, 22 heads per rank ‡ | 0.336 | **0.349** | 0.425 |
| control 4096 × 4096 | 0.215 | **0.272** | 0.302 |
| control 4096 × 11008 | 0.195 | **0.254** | 0.293 |

M=64 is a C8 verify batch and it agrees with M=8 on the only thing that matters: `kv_b_proj` wins
clearly there too (0.388), the KDA arms are level (0.995) or lose (the 4,096×128 family at 1.63).

## Which regime is real: the production trace is the referee

This is the table that settles it, and it needed no new measurement — the per-call costs of the same
kernels were already read off the production-9 C1 trace, 96 steps
([`../profile/step-breakdown.csv`](../profile/step-breakdown.csv)) `[measured-here]`:

| family (production-9 trace) | calls/step | **trace µs/call** | bench warm µs | bench cold µs | warm ÷ trace | **cold ÷ trace** |
|---|---:|---:|---:|---:|---:|---:|
| `f_b_proj` + `g_b_proj` (grid 8,11,1) | 68 | **5.41** | 2.21 | 4.99 | 0.41 | **0.92** |
| `in_proj_bfg_a` (grid 8,2,8) | 34 | **14.20** | 6.52 | 15.51 | 0.46 | **1.09** |
| MLA A / `kv_b_proj` (grid 8,32,1) | 11 | **169.14** | 136.61 | 139.33 | 0.81 | **0.82** |

**The cold arm reproduces the engine to within ±20 %; the warm arm is off by 2.2–2.4× on the small
shapes.** Which regime is honest is therefore a measurement, not a preference. The mechanism is
plain: tens of GiB of weights per rank stream through L2 in one decode step, so between two touches
of the same `f_b_proj` the entire active weight set has passed — in production that tensor is
**never** resident.

## Achieved bandwidth, M = 8 — the numbers give the artefact away

This card's measured achievable read is **235–240 GB/s**. Anything above that is L2 talking
`[measured-here]`:

| shape | arm | warm GB/s | cold GB/s | reading |
|---|---|---:|---:|---|
| `f_b_proj` 128×2,816 | bf16 | **347.9** | 154.0 | 0.72 MB — entirely L2-resident when warm |
| `f_b_proj` 128×2,816 | exl3-4b | 65.8 | **45.7** | **not bandwidth-bound at all** — a fixed-cost floor |
| `in_proj_fg_a` 4,096×256 | bf16 | **410.9** | 176.9 | same |
| `in_proj_fg_a` 4,096×256 | exl3-4b | 100.3 | 75.1 | same |
| `kv_b_proj` 512×32,768 | bf16 | 249.5 | 244.7 | 33.5 MB does **not** fit 24 MiB → warm ≈ cold |
| `kv_b_proj` 512×32,768 | exl3-4b | **509.4** | 221.6 | 8.4 MB trellis **does** fit → warm is the artefact |
| control 4096×4096 | exl3-4b | **489.5** | 217.1 | same |

**The sign of the artefact flips with L2 size, and that is the transferable part:**

- **128 MiB L2** (the kernel author's card): both the 32 MB bf16 weight and the 8 MB trellis fit, so
  warm deletes the bandwidth advantage and leaves only trellis-decode cost — **EXL3 looks slow**
  (1.13).
- **24 MiB L2** (GB10): bf16 does not fit, the trellis does, so warm speeds up only the EXL3 arm —
  **EXL3 looks too fast** (0.144).
- **101 MB L2, small shapes** (the workstation): a 0.72 MB bf16 weight fits and takes the warm
  advantage, while the EXL3 arm's two dependent kernels cannot — **EXL3 looks slow** (1.596). GB10's
  warm arm reproduces that exactly at **1.605**.

Three cards, three directions, one honest regime. A bank sized against the wrong card is the same
mistake as no bank at all.

## What the small shapes are actually bound by: two dependent launches, not bytes

Cold, `f_b_proj` bf16 reads 154 GB/s in 4.99 µs — 65 % of peak. Cold, `f_b_proj` at 4 bit reads
**45.7 GB/s in 5.10 µs — 19 % of peak**. The EXL3 arm is not hitting bandwidth; it is hitting a
**~5 µs floor**, which the ruler names: `exl3_had_in_kernel` + `exl3_gemm_m_kernel`, **two dependent
launches** where bf16 has one. The 0.54 MB that 4 bit saves pays for the second launch and no more,
which is the whole of the 1.02.

**So the lever on these shapes is not the bit width.** Fusing `had_in` into the GEMM for narrow inputs
would take the `f_b`/`g_b` family below 1.0; quantizing them will not. The workstation report's
intuition — "quantizing a small tensor does not make it faster" — was right, and its stated reason
(bytes) was wrong: it is **launch count**. Reported upstream with the shape family (k=128, n=2,816,
M=8) `[not tested]`.

## The projection, and the gate

Family costs are from the production-9 C1 trace, base step **72.52 ms**; ratios are the **cold** column
above. `in_proj_bfg_a` is charged as the split it would have to become — `in_proj_fg_a` in EXL3 plus
`in_proj_b` left in bf16, because 22 columns per rank cannot be a 128-aligned EXL3 shard:
`(8.02 + 5.63) / 15.51 = 0.880`.

**EXL3 4 bit** `[measured-here]` for the ratios, `[estimate]` for the projection:

| family | calls/step | ms/step now | cold ratio | projected ms | **Δ, + = faster** |
|---|---:|---:|---:|---:|---:|
| `f_b_proj` + `g_b_proj` | 68 | 0.368 | 1.023 | 0.376 | −0.008 |
| `in_proj_bfg_a` → split | 34 | 0.483 | 0.880 | 0.425 | +0.058 |
| **KDA gating arms, subtotal** | 102 | **0.851** | 0.942 | 0.801 | **+0.050** |
| `kv_b_proj` / MLA A ‡ | 11 | 1.860 | 0.291 | 0.542 | **+1.318** |
| **measured subtotal** | | **2.711** | 0.495 | 1.343 | **+1.368** |

→ 72.52 → **71.15 ms, +1.92 % of C1**.

**EXL3 6 bit** — the width this checkpoint uses for its dense modules, and the quality-safe option:

| family | ms/step now | cold ratio | projected ms | **Δ** |
|---|---:|---:|---:|---:|
| `f_b_proj` + `g_b_proj` | 0.368 | 1.158 | 0.426 | −0.058 |
| `in_proj_bfg_a` → split | 0.483 | 0.955 | 0.461 | +0.022 |
| `kv_b_proj` / MLA A ‡ | 1.860 | 0.384 | 0.714 | **+1.146** |
| **measured subtotal** | **2.711** | 0.591 | 1.601 | **+1.110** |

→ 72.52 → **71.41 ms, +1.55 % of C1**.

**The gate**, unchanged: `Δ ≥ 1.5 ms/step` at decode **and** `Δ ≤ +5 ms/chunk` at prefill.

| test | workstation (as published) | **GB10 cold, 4 bit** | **GB10 cold, 6 bit** |
|---|---:|---:|---:|
| decode, `Δ ≥ 1.5 ms/step`, three measured families | +0.547 — fails | **+1.368 — fails, narrowly** | +1.110 — fails |
| the same applied to the whole 4.10 ms target BF16 residue `[estimate]` | — | **+2.07 — passes** | +1.68 — passes |
| KDA gating arms alone | **−0.584 (a loss)** | **+0.050 (neutral)** | −0.036 (neutral) |
| `kv_b_proj` alone ‡ | +1.132 | **+1.318** | +1.146 |
| ceiling: KDA arms at zero cost | +0.851 | +0.851 | +0.851 |
| prefill, `Δ ≤ +5 ms/chunk` | passes on the target share (+3.2); +6.7 on the pessimistic split | **not re-measured** | not re-measured |

The `[estimate]` row applies the cost-weighted cold ratio (0.495) to the full **4.10 ms** the trace
attributes to the target model's four unquantized families
([`../profile/step-breakdown.csv`](../profile/step-breakdown.csv), `decode_c1`): 4.10 × (1 − 0.495) =
**+2.07 ms**, 72.52 → 70.45 ms, **+2.94 % of C1**. It is the optimistic end, because **1.389 ms** of
that budget is MLA/DSA kernels this bench never measured — one of which is the fp32 family that is a
dtype question, not a quantization one. **+1.37 ms is the honest figure; +2.07 ms is the ceiling.**

**And the work list inverted rather than grew.** **96 %** of the gain (+1.318 of +1.368) is still one
item, `kv_b_proj`, and it still needs the per-head batched EXL3 kernel that `754421f` does not have.
The KDA gating arms are no longer harmful — they are simply **free**: +0.050 ms/step is **0.07 % of
C1**, which does not pay for the multiplicative quality risk on `f_b_proj`'s decay term.

## Facing the workstation table

| shape, M=8, 4 bit | published (workstation) | GB10 warm | **GB10 cold** | verdict |
|---|---:|---:|---:|---|
| `f_b_proj` TP=3 | 1.596 | 1.605 | **1.023** | the published number is the **warm** regime; cold is level |
| `g_b_proj` TP=3 | 1.580 | 1.613 | **1.025** | same |
| `f_b_proj` unsharded | 1.254 | 1.479 | **0.643** | **sign reversed** |
| `f_a_proj` replicated | 1.171 | 1.270 | **0.853** | **sign reversed** |
| `g_a_proj` replicated | 1.172 | 1.155 | **0.853** | **sign reversed** |
| `in_proj_fg_a` | 1.060 | 1.139 | **0.655** | **sign reversed** |
| `b_proj` replicated | 1.264 | 1.270 | **0.857** | **sign reversed** |
| `kv_b_proj` replicated ‡ | 0.391 | 0.129 | **0.291** | same direction, cold better |
| `kv_b_proj` 22 heads ‡ | 0.747 | 0.682 | **0.349** | same direction, cold 2.1× better |

**A second claim of the workstation report is withdrawn with the first** `[retracted]`. It warned
that its ratios were *optimistic* for GB10 — "at M=8 GB10's fixed cost is larger, so its ratios are
worse" — and the gate was declared to fail on optimistic numbers. **The opposite is true: every
family came out better on GB10.** GB10 is far more bandwidth-starved relative to compute than the
workstation — measured machine balance about **416 flop/byte** here against roughly 141 there — which
is exactly the condition a trellis is built for. On the controls, the pure bit-ratio limit at 4 bit is 4/16 = 0.25 and we measure **0.272**;
at 6 bit the limit is 0.375 and we measure **0.389**. On this card EXL3 delivers very nearly all of
what the bit ratio promises.

The one cross-check the workstation report offered — the production 7 → 9 dense A/B measuring
**0.386** against the bench's 0.391 — is a *warm-against-engine* coincidence on a large shape, where
warm and cold nearly agree (the 33.5 MB bf16 weight does not fit either L2). It supported the wrong
row for the right reason and it is unaffected.

## What was not measured

- **Prefill.** M = 1, 8 and 64 only. **M = 1,792 was not re-measured**, and the gate is an **and**, so
  decision 8 is **re-scoped, not passed**, on this document. The workstation prefill verdict is less
  exposed to the artefact — at M=1,792 the shapes are compute-bound — but it was not re-run
  `[not tested]`.
- **Family-to-kernel attribution is still inference.** "MLA A (169 µs × 11) is the `kv_b_proj` dense
  read" comes from a bandwidth anchor, not from a kernel name; the trace-referee agreement at 0.82
  supports it without proving it. If it is wrong, the +1.318 ms is wrong with it. This is the same
  honesty limit the profile pages carry.
- **`kv_b_proj` is conditional on a kernel.** The plain `[M,k] → [M,n]` shape was measured; the
  per-head batched form production would need does not exist in `754421f`. That the new kernel would
  hit this ratio is an **assumption** `[not tested]`.
- **CUDA graphs were on**, and production 9/10 run the target eager. The ±20 % agreement with the
  trace says this does not matter in practice, but the EXL3 arm's extra kernel would pay a second
  launch cost in eager mode that the projection does **not** charge — so the KDA arms' +0.050 ms is,
  if anything, smaller still.
- **Quality.** This is a speed gate. Nothing here says whether those arms survive 4 bit; the
  sensitivity mechanism in [docs/13](../../docs/13-full-scope-checkpoint.md) §4.4 stands unchanged.
- **The MLA fp32 → bf16 family** was not in this run, and its workstation numbers were never
  re-measured cold either. The lever is unaffected in direction and stays the cheapest item on the
  board `[not tested]`.
- **Three rounds is not statistics.** The spread is under 1 % and every round is in the raw log, but
  three is three.
- **"4× L2" is our threshold**, not a published one. `coldA` against `coldB` bounds the sensitivity to
  it at ≤ 5 %.

**What this cost:** about 90 s of one GPU, 1.47 GiB peak, no engine restart, no configuration change.
It bought back a wrong sentence on the front of a closed item, and it moved the same item's arithmetic
by 2.5×.

## Raw

[`gb10-coldbench/`](gb10-coldbench/) — `00-ruler-gb10.txt` (device, ruler, launch-count check),
`01-coldbench.txt` (the full run: every shape, arm, variant, M and round),
`02-projection.txt`, `03-bandwidth.txt`, and `coldbench-summary.csv` (531 rows: one per
shape × M × arm × bit width × variant, with all three rounds, achieved bandwidth and ratio). The
217 KB JSON the CSV is distilled from is not shipped; the CSV carries every field of it except the
run metadata, which is in the CSV's first line.

# Model-free gate bench: would quantizing the remaining BF16 families help?

The question this answers: production 9 and 10 leave **113 linears per rank in BF16** — KDA
`f_b_proj` (34), `g_b_proj` (34), `in_proj_bfg_a` (34) and MLA `kv_b_proj` (11)
([docs/13](../../docs/13-full-scope-checkpoint.md) §4.3). Their FP16 weights are already inside the
checkpoint, so quantizing them ourselves is a surgical pass rather than a requantization. **Should we?**

**Answer: no.** The gate the arm was given — `Δ ≥ 1.5 ms/step` at decode **and** `Δ ≤ +5 ms/chunk` at
prefill — fails on the decode half, for every family and every combination. Narrative and the three
lessons: [docs/13](../../docs/13-full-scope-checkpoint.md) §4.4; the item is closed in
[docs/11](../../docs/11-open-issues.md) §2.25.

## How this was measured, and where

**Not on the cluster.** `cuda-exl3` at `754421f` — the production commit — was built for `sm_120` on a
workstation RTX 5090, and the whole study ran there: no node was touched, no engine restarted, no env
file changed. Peak GPU memory 0.99 GiB.

`bench/kda_gate_bench.py` measures `F.linear` (bf16, what production runs) against
`cuda_exl3.ops.exl3_linear` with `cb=2` (`mul1`, the production checkpoint's codebook) at 4 and 6 bit,
at the **TP=3 per-rank widths**, M=8 (a decode step with a k=7 drafter) and M=1,792 (the real prefill
chunk, [docs/10](../../docs/10-results-and-roofline.md) §5.2). CUDA graphs on, median of 60 replays.

**The absolute microseconds below are 5090 numbers and do not transfer.** Only the `exl3 / bf16`
**ratio** per shape is carried to GB10, and the direction of the error is stated: at M=8 on small
tensors both machines are fixed-cost-bound and GB10's fixed cost is *larger* (48 SMs against 170), so
its ratios are **worse** than these; at M=1,792 GB10 is the compute-poorer part, so EXL3's trellis
decode weighs **more** there. **The gate fails on optimistic ratios.** One independent cross-check
exists: the production 7 → 9 dense A/B on GB10 measured a bf16→EXL3 ratio of **0.386** on the same
size class where this bench reads **0.391**.

## The ruler was checked first, and it caught itself

`bench/ruler_check.py`, before any of the tables below `[measured-here]`:

| check | result |
|---|---|
| bf16 arm, M=1,792, `k=512 n=32768` | 210.5 TFLOPS = **90 %** of the machine's measured peak — cuBLAS path healthy |
| bf16 arm, M=8, with a weight bank | 1,526 GB/s = **92 %** of peak read bandwidth — bandwidth arm healthy |
| bf16 arm, M=8, **without** a bank | 3,488 GB/s = **210 % of peak** — physically impossible |
| EXL3 shard-split consistency | fused `[n]` against two `[n/2]` calls: relative error **0.00** at 4 bit |
| EXL3 dtype invariance | bf16 against fp16 activations: ~4e-2, one bf16 ULP |
| repeatability | two independent full runs agree on **every ratio to within 1 %** |

**The 210 % row is the point.** This card has 101 MB of L2; a single small trellis sits resident and
the measurement reads faster than the machine can physically fetch. Every arm therefore runs against a
**~300 MB weight bank** (three times L2). Had the shapes been slightly larger the cache would have
returned a believable 95 % of peak and the whole table would have been quietly wrong.

The repository's own `tests/test_exl3_gemm.py` reports **41/41 skipped** here: those tests want a real
EXL3 checkpoint, which exists only on the nodes. The four checks above are the model-free substitute.

## M = 8 (decode step, k=7 drafter → 8 tokens)

µs per call, median of 60, CUDA graphs on. `ratio = exl3 / bf16`; **above 1 means EXL3 is slower**.

| shape (TP=3 per rank) | k | n | bf16 µs | exl3 4-bit | ratio 4b | exl3 6-bit | ratio 6b |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f_b_proj` | 128 | 2,816 | 2.14 | 3.41 | **1.596** | 3.52 | 1.644 |
| `g_b_proj` | 128 | 2,816 | 2.16 | 3.42 | **1.580** | 3.52 | 1.627 |
| `f_b_proj` unsharded (reference) | 128 | 8,192 | 2.85 | 3.58 | 1.254 | 3.79 | 1.330 |
| `f_a_proj` replicated | 4,096 | 128 | 4.75 | 5.57 | 1.171 | 5.47 | 1.150 |
| `g_a_proj` replicated | 4,096 | 128 | 4.75 | 5.57 | 1.172 | 5.47 | 1.151 |
| `in_proj_fg_a` (f_a + g_a fused) | 4,096 | 256 | 5.02 | 5.32 | 1.060 | 5.58 | 1.111 |
| `in_proj_bfg_a` **as production runs it** | 4,096 | 278 | 4.99 | — | — | — | — |
| `in_proj_b` (would stay bf16) | 4,096 | 22 | 3.45 | — | — | — | — |
| `b_proj` replicated (64 → 128) | 4,096 | 128 | 4.75 | 6.00 | 1.264 | 5.49 | 1.155 |
| **`kv_b_proj` replicated** | 512 | 32,768 | 21.99 | **8.61** | **0.391** | 11.05 | 0.502 |
| `kv_b_proj`, 22 heads per rank | 512 | 11,264 | 8.99 | 6.72 | 0.747 | 7.32 | 0.814 |

**Launch cost, eager minus graph, M=8:** bf16 `f_b_proj` +1.85 µs, EXL3 `f_b_proj` +2.46 µs, bf16
`in_proj_b` **+1.86 µs**. That last one is the extra launch the `in_proj_bfg_a` split would add per
KDA layer, per step — 34 layers of it — and the projection below does **not** charge it, so the KDA
arm's real loss is larger than the table says.

## M = 1,792 (prefill chunk)

| shape | k | n | bf16 µs | exl3 4-bit | ratio 4b | exl3 6-bit | ratio 6b |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f_b_proj` | 128 | 2,816 | 16.18 | 14.49 | **0.896** | 15.26 | 0.943 |
| `g_b_proj` | 128 | 2,816 | 16.30 | 14.67 | **0.900** | 15.27 | 0.937 |
| `f_b_proj` unsharded | 128 | 8,192 | 39.96 | 32.58 | 0.815 | 33.60 | 0.841 |
| `f_a_proj` replicated | 4,096 | 128 | 12.94 | 45.73 | **3.534** | 48.66 | 3.760 |
| `g_a_proj` replicated | 4,096 | 128 | 12.89 | 45.69 | **3.544** | 48.65 | 3.774 |
| `in_proj_fg_a` | 4,096 | 256 | 22.68 | 48.56 | 2.141 | 52.45 | 2.313 |
| `kv_b_proj` replicated | 512 | 32,768 | 289.41 | 313.39 | 1.083 | 325.86 | 1.126 |
| `kv_b_proj`, 22 heads per rank | 512 | 11,264 | 106.93 | 115.71 | 1.082 | 119.74 | 1.120 |

**Read this table sideways.** `f_b`/`g_b` — narrow input, k=128 — get **faster** under EXL3 in
prefill, because bf16 cutlass is inefficient at that shape. `f_a`/`g_a` — narrow output, n=128 — get
**3.5× slower**, because the trellis is decoded again for every M-tile and the weight-read saving is
meaningless at M=1,792. The prefill penalty is a property of **shape**, not of format.

## MLA strided-batched family: fp32 against bf16 (no quantization involved)

Production runs this family in **fp32** — `torch.bmm`, 11 calls per step:

| shape | batch | k | n | M | fp32 µs | bf16 µs | fp16 µs | bf16/fp32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `w_uk`, 22 heads per rank | 22 | 256 | 512 | 8 | 8.69 | 5.59 | 5.36 | **0.643** |
| `w_uv`, 22 heads per rank | 22 | 512 | 256 | 8 | 9.60 | 6.95 | 5.70 | **0.724** |
| `w_uk`, 64 heads replicated | 64 | 256 | 512 | 8 | 25.25 | 12.75 | 12.35 | 0.505 |
| `w_uv`, 64 heads replicated | 64 | 512 | 256 | 8 | 22.40 | 12.36 | 11.88 | 0.552 |
| `w_uk`, 22 heads per rank | 22 | 256 | 512 | 1,792 | 270.92 | 121.63 | 58.73 | **0.449** |
| `w_uv`, 22 heads per rank | 22 | 512 | 256 | 1,792 | 271.11 | 112.51 | 56.09 | **0.415** |

## The gate

Family costs on GB10 come from the production-9 trace
([`../profile/step-breakdown.csv`](../profile/step-breakdown.csv), C1 window, 96 steps); ratios come
from the tables above.

| family | calls/step | GB10 ms/step now | ratio (M=8) | projected ms/step | **Δ, + = faster** |
|---|---:|---:|---:|---:|---:|
| KDA `f_b` + `g_b` → EXL3 4b | 68 | 0.368 | 1.593 | 0.586 | **−0.218** |
| KDA `in_proj_bfg_a` → EXL3 + bf16 split | 34 | 0.483 | 1.758 | 0.849 | **−0.366** |
| **KDA arms, total** | 102 | 0.851 | — | 1.435 | **−0.584** |
| MLA dense family → EXL3 4b | 11 | 1.860 | 0.392 | 0.728 | **+1.132** |
| KDA + MLA together | | 2.711 | | 2.163 | +0.547 |
| **MLA fp32 → bf16 (no quantization)** | 11 | 0.757 | 0.684 | 0.518 | **+0.240** |

**Decode gate, `Δ ≥ 1.5 ms/step`: FAILED, by every family and every combination.** The best single
item is +1.132 ms. The verdict is robust to a wrong ratio, because the **ceiling** is already under
the bar: if the KDA gating arms cost **nothing at all** the step gains 0.851 ms, **+1.2 % of C1** —
inside the noise floor in [docs/10](../../docs/10-results-and-roofline.md) §1.1.

**Prefill gate, `Δ ≤ +5 ms/chunk`: passed** (+3.2 ms on the target-share split, +6.7 ms on the
pessimistic one). Irrelevant: the gate is an **and**.

**Two source facts that close the remaining doubt**, verified by scanning `cuda-exl3` at `754421f`:

1. There is **no per-head batched EXL3 GEMM**. `exl3_linear` is a single `[M,k] → [M,n]`; a
   `bmm|batched|per_head|strided` scan finds only two comments. The +1.132 ms above is conditional on
   a kernel that would have to be written, and it still does not clear the gate alone.
2. There is **no M-threshold reconstruct path**. `exllamav3` has `AUTO_RECONSTRUCT_THRESHOLD = 144`
   (dequantize and use cuBLAS above it); a `reconstruct|RECONSTRUCT_THRESHOLD|m_threshold` scan of
   `cuda-exl3` returns nothing. The obvious cure for the prefill side does not exist here.

## What was not measured

- **GB10 ratios.** Carried from the 5090, with one cross-check (0.386 against 0.391).
- **Quality.** This is a speed gate only. The sensitivity question — whether those arms survive 4 bit
  at all — was never answered, and no longer needs to be: quantizing them is slower *and* risky.
- **Kernel-to-tensor attribution.** Which trace kernel is which tensor was inferred from call count
  and grid, not confirmed by name — the same honesty limit the profile pages carry. If "the MLA dense
  family is `kv_b_proj`" is wrong, the +1.132 ms is wrong too; the gate still fails, because the whole
  MLA bf16 budget at zero cost is +3.402 ms across four families and no single one reaches 1.5.
- **The extra launch** the `in_proj` split would cost is not charged in the projection.

Reproduce: `bench/ruler_check.py` first, then
`bench/kda_gate_bench.py --m 8 1792 --bits 4 6 --iters 60`. Both need `cuda_exl3` importable and
nothing else; neither touches a node or a checkpoint.

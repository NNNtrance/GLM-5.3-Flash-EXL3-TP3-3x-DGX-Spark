# `0002-harem-on-77513d2.patch` — RETIRED 5 September 2026

Superseded by upstream. Do **not** apply it on top of `a95e809` or later.

| what it did | where it is now |
|---|---|
| pass `n_rows` on the unsplit MoE launch (`gemm.cu`, `VE3_ONE` else branch) | upstream **`a95e809`** |
| `exl3_moe_had_in`: retire a remote block before the padding write (`hadamard.cu`) | upstream **`a95e809`** |
| EP-aware `exl3_moe_combine` (optional `expert_ids`/`block_m`, skip retired rows) | upstream **`f906f00`** |
| gemm/`glu_had` **zero the retired tile** instead of returning | **withdrawn — it was the wrong choice.** Measured on GB10 at M=2048: +10.9 % on gemm w13, +18.6 % on gemm w2, +10.7 % on the whole MoE layer against upstream's "return and let the combine skip". |
| combine stages `inv[]` / weights in shared memory | **kept**, rebased as `0003-combine-smem-staging-on-a95e809.patch` |

The only HAREM kernel change still carried in production is `0003`.
Evidence: the per-kernel MoE stage comparison in [docs/05](../../docs/05-expert-parallel-and-cuda-exl3-fixes.md).

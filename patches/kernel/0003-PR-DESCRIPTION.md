# exl3_moe_combine: stage the per-(token, k) facts in shared memory

Patch: `0003-combine-smem-staging-on-a95e809.patch` (applies unchanged to `bc0e0f6`).

## What

`exl3_moe_combine_kernel` re-reads three things inside the `h` loop that do not
depend on `h`: the gathered row `inv[token * top_k + k]`, its routing weight
`w[token * top_k + k]`, and — since `f906f00` — the liveness test
`expert_ids[r / block_m]`. Stage all three once, in shared memory, before the loop.

## Why it costs

* The loop body is executed `ceil(H / blockDim)` times per thread. At `H = 4096`
  with 1024 threads that is four times per thread, so each of those loads is paid
  four times.
* The liveness test is a **dependent chain**: load `inv` → integer-divide by a
  runtime `block_m` → load `expert_ids` → compare, and only then the `rows_out`
  load it guards.
* The grid is one block per token. At a decode batch of 8 tokens that is 8 blocks
  on 48 SMs, so there is no occupancy to hide the chain behind.

`expert_ids == nullptr` keeps the pre-EP path exactly as it was, and the staged
form is bit-identical to the original either way (see the tests below).

## Numbers

GB10 (48 SMs, sm_121), GLM-5.3-Flash shapes: hidden 4096, routed intermediate 2048,
288 global experts, 96 local (expert parallel over three ranks), top-8, 4-bit `mcg`,
`block_m` from the ladder. `exl3_moe_combine` called with `expert_ids` and `block_m`;
mean of two independent routing draws, 50 timed iterations after 10 warm-ups.

| M (tokens) | block_m | `bc0e0f6` | **+ staging** | change |
|---|---|---|---|---|
| 8 | 16 | 11.6 µs | **7.7 µs** | **−34 %** |
| 64 | 16 | 16.2 µs | **10.3 µs** | **−36 %** |
| 2048 | 64 | 363 µs | **317 µs** | **−13 %** |

Whole MoE layer (had_in + gemm w13 + glu_had_in + gemm w2 + combine), same runs:
1067 → 1069 µs at M=8, 4804 → 4760 µs at M=64, 8587 → 8515 µs at M=2048. The combine
is a small share of the layer at prefill batch, so the layer-level gain is about 1 %;
the kernel-level gain is what the patch is about, and it is largest exactly where the
launch is smallest.

## Tests

Run inside the built image (`bc0e0f6` + this patch), on GB10:

* upstream suite, whole `tests/` directory: **44 passed, 41 skipped** (includes
  `test_exl3_moe_pad.py`, `test_exl3_moe_split.py`, `test_exl3_moe_glu.py`,
  `test_exl3_gemm.py`).
* poison test: `a13` filled with NaN before the transform; output bit-identical to
  the pre-zeroed control at M = 8 / 64 / 512 / 2048 → **PASS** (nothing downstream
  reads a retired block's activations).
* equivalence: `rows_out` poisoned with NaN on every retired row, then
  `combine(expert_ids, block_m)` compared bitwise against `masked_fill_ + combine`
  → **PASS** at all four M.
* expert parallel off (288 local experts, `expert_map = None`): the 6-argument call
  is **bitwise identical** to the 4-argument call at all four M.

## Note on `MAX_TOPK`

The staging arrays are sized for `top_k <= 32` and the host function now
`TORCH_CHECK`s it. GLM-5.3-Flash uses 8. If a larger fan-out matters the loop can be
tiled over `MAX_TOPK` instead; that was not written because nothing here needs it.

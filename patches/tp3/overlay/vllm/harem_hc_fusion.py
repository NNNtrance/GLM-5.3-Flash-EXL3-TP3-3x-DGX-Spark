# SPDX-License-Identifier: Apache-2.0
"""HAREM hc-fusion: one kernel for mHC post-mapping + pre-norm GEMM (large M).

Why
---
GLM-5.3-Flash runs the DeepseekV4 hyper-connection block (``hc_mult = 4``) twice
per decoder layer.  On the large-M (prefill) path vLLM launches three kernels:

    k1  mhc_post_tilelang            writes residual_cur  (M, 4, H) bf16
    k2  tf32_hc_prenorm_gemm         reads  residual_cur, writes (mixes, sqrsum)
    k3  mhc_pre_big_fuse_with_norm   reads  residual_cur, writes layer_input

``residual_cur`` is 32,768 B/token and is touched three times: written by k1,
read by k2, read by k3.  Measured on GB10 (48 SM, 24 MiB L2, 225-240 GB/s read)
the whole block runs at 86-98 % of the read roofline -- it is pure traffic, 5.3
FLOP/byte against a 405 FLOP/byte balance point.  The only real lever is to
remove one of those three touches.

This module removes k2's read: the post mapping and the pre-norm reduction run
in the same kernel, so the new residual row is reduced against ``fn`` while it is
still in registers.

    today   k1 + k2   107,448 B/token
    fused              ~74,600 B/token         -30 %   (ceiling -2.5..-2.7 % prefill)

Why not the fused kernel that already exists
--------------------------------------------
``mhc_fused_tilelang`` (tilelang_kernels.py) does exactly this fusion, but its
grid is ``(m, n_tiles, split_k)`` -- one CTA per token *per n-tile*, and every
CTA re-reads the whole residual row.  Forced on at M=2048 it is +32 % slower than
the unfused k1+k2 pair; it only exists for the <=16-token decode path.  The fix
is to tile over *tokens* (``BLOCK_M``) and finish all 24 outputs inside one CTA,
so the residual row is read exactly once and ``fn`` (1.57 MB) is amortised over
BLOCK_M tokens instead of being re-read per token.

Numerics
--------
* ``residual_cur`` is computed in fp32 and rounded to bf16 with the same term
  order as ``mhc_post_tilelang`` (post term first, then k = 0..3), so it is
  expected to be bit-identical to the unfused path.
* The GEMM consumes the *bf16-rounded* residual -- exactly what k2 does today
  when it reads ``residual_cur`` back from HBM -- and uses tf32 tensor cores,
  the same arithmetic as ``deep_gemm::sm120_tf32_hc_prenorm_gemm``.  The
  reduction order differs, so ``mixes``/``sqrsum`` agree to tf32 rounding, not
  bit-for-bit.  ``HAREM_MHC_FUSED_PREC=ieee`` switches the dot to full fp32.

Gating (fail-closed)
--------------------
Nothing here runs unless ``HAREM_MHC_FUSED_LARGE=1``.  Even then the entry point
returns ``None`` -- caller runs the stock three-kernel path -- whenever anything
is not exactly as expected: Triton missing, dtype/shape/stride unexpected,
hc_mult != 4, M below ``HAREM_MHC_FUSED_MIN_M``, or the kernel raising.  The
first failure disables the path for the rest of the process, loudly.

Env
---
  HAREM_MHC_FUSED_LARGE   0/1   master switch (default 0 = stock path)
  HAREM_MHC_FUSED_MIN_M   int   min tokens for the fused path (default 1024)
  HAREM_MHC_FUSED_BM      int   BLOCK_M    (default from _pick_config)
  HAREM_MHC_FUSED_BH      int   BLOCK_H    (default from _pick_config)
  HAREM_MHC_FUSED_SPLITH  int   SPLIT_H    (default from _pick_config)
  HAREM_MHC_FUSED_WARPS   int   num_warps  (default from _pick_config)
  HAREM_MHC_FUSED_STAGES  int   num_stages (default from _pick_config)
  HAREM_MHC_FUSED_PREC    str   tf32 | tf32x3 | ieee   (default tf32)
  HAREM_MHC_FUSED_DEBUG   0/1   log the chosen config once
"""

from __future__ import annotations

import os

import torch

try:  # Triton is present in the exl3-zeus image (3.7.1); never assume it.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001 -- any import failure means: use the stock path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


ENABLED = os.environ.get("HAREM_MHC_FUSED_LARGE", "0") == "1"
# 1024, MEASURED not guessed: at M=512 residual_cur is 16.8 MB and fits the
# 24 MiB L2, so k2's "re-read" never reaches DRAM and there is nothing for the
# fusion to remove -- it loses 37.7 %.  At M=2048 (67 MB) it wins 15 %.
MIN_M = _env_int("HAREM_MHC_FUSED_MIN_M", 1024)
PREC = os.environ.get("HAREM_MHC_FUSED_PREC", "tf32")
DEBUG = os.environ.get("HAREM_MHC_FUSED_DEBUG", "0") == "1"

HC_FIXED = 4  # the kernel is written out for hc_mult=4; asserted before use.

# Set once, on the first failure, so a bad shape cannot cost a try/except per call.
_DISABLED_REASON: str | None = None
_LOGGED = False


# --------------------------------------------------------------------------- #
# kernel
# --------------------------------------------------------------------------- #
if HAS_TRITON:

    @triton.jit
    def _mix_row(xv, r0, r1, r2, r3, pj, c0, c1, c2, c3):
        """residual_cur[:, j, h] for one j, in the term order of mhc_post_tilelang.

        nr = post[:, j] * x + sum_k comb[:, k, j] * residual[:, k, :]
        """
        nr = pj[:, None] * xv
        nr += c0[:, None] * r0
        nr += c1[:, None] * r1
        nr += c2[:, None] * r2
        nr += c3[:, None] * r3
        return nr.to(tl.bfloat16)

    @triton.jit
    def _mhc_post_prenorm_fused_kernel(
        x_ptr,  # (M, H)         bf16
        res_ptr,  # (M, HC, H)     bf16
        post_ptr,  # (M, HC)        fp32
        comb_ptr,  # (M, HC, HC)    fp32
        fn_ptr,  # (N_OUT, HC*H)  fp32
        rout_ptr,  # (M, HC, H)     bf16  <- written
        gout_ptr,  # (SPLIT_H, M, N_OUT) fp32  <- written
        gsq_ptr,  # (SPLIT_H, M)        fp32  <- written
        M,
        H,
        N_OUT: tl.constexpr,
        N_PAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        SPLIT_H: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        """One CTA owns BLOCK_M tokens x one H slice, and *all* N_OUT outputs.

        The residual row is read once, residual_cur written once, and the
        reduction against fn happens on the value still held in registers.  fn is
        the only tensor re-read across CTAs (1.57 MB, L2-resident) and BLOCK_M
        divides that traffic.

        SPLIT_H splits the reduction axis across CTAs, which costs nothing --
        each CTA still owns a disjoint slice of residual/residual_cur, and k3
        already sums ``n_splits`` partial (mixes, sqrsum).  It exists purely to
        decouple the CTA count from BLOCK_M: at M=2048 and BLOCK_M=32 the grid
        would otherwise be 64 CTAs on 48 SMs, i.e. a second wave that is 1/3
        full.  H must be a multiple of SPLIT_H * BLOCK_H.
        """
        pid = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # (BM,)
        mask_m = offs_m < M
        offs_n = tl.arange(0, N_PAD)  # (NP,)
        mask_n = offs_n < N_OUT
        hc_h = 4 * H
        h_per_split = H // SPLIT_H
        h_begin = pid_s * h_per_split

        # 4 + 16 mixing coefficients per token, hoisted out of the H loop.
        p0 = tl.load(post_ptr + offs_m * 4 + 0, mask=mask_m, other=0.0)
        p1 = tl.load(post_ptr + offs_m * 4 + 1, mask=mask_m, other=0.0)
        p2 = tl.load(post_ptr + offs_m * 4 + 2, mask=mask_m, other=0.0)
        p3 = tl.load(post_ptr + offs_m * 4 + 3, mask=mask_m, other=0.0)
        cb = comb_ptr + offs_m * 16
        c00 = tl.load(cb + 0, mask=mask_m, other=0.0)  # comb[m, k=0, j=0]
        c01 = tl.load(cb + 1, mask=mask_m, other=0.0)  # comb[m, k=0, j=1]
        c02 = tl.load(cb + 2, mask=mask_m, other=0.0)
        c03 = tl.load(cb + 3, mask=mask_m, other=0.0)
        c10 = tl.load(cb + 4, mask=mask_m, other=0.0)  # comb[m, k=1, j=0]
        c11 = tl.load(cb + 5, mask=mask_m, other=0.0)
        c12 = tl.load(cb + 6, mask=mask_m, other=0.0)
        c13 = tl.load(cb + 7, mask=mask_m, other=0.0)
        c20 = tl.load(cb + 8, mask=mask_m, other=0.0)
        c21 = tl.load(cb + 9, mask=mask_m, other=0.0)
        c22 = tl.load(cb + 10, mask=mask_m, other=0.0)
        c23 = tl.load(cb + 11, mask=mask_m, other=0.0)
        c30 = tl.load(cb + 12, mask=mask_m, other=0.0)
        c31 = tl.load(cb + 13, mask=mask_m, other=0.0)
        c32 = tl.load(cb + 14, mask=mask_m, other=0.0)
        c33 = tl.load(cb + 15, mask=mask_m, other=0.0)

        acc = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
        sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for h0 in tl.range(h_begin, h_begin + h_per_split, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)  # (BH,)
            rm = offs_m[:, None] * hc_h + offs_h[None, :]
            fnb = offs_n[:, None] * hc_h + offs_h[None, :]

            xv = tl.load(
                x_ptr + offs_m[:, None] * H + offs_h[None, :],
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)  # (BM, BH)

            # the four residual rows for this h-slice -- read exactly once
            r0 = tl.load(res_ptr + rm + 0 * H, mask=mask_m[:, None], other=0.0).to(
                tl.float32
            )
            r1 = tl.load(res_ptr + rm + 1 * H, mask=mask_m[:, None], other=0.0).to(
                tl.float32
            )
            r2 = tl.load(res_ptr + rm + 2 * H, mask=mask_m[:, None], other=0.0).to(
                tl.float32
            )
            r3 = tl.load(res_ptr + rm + 3 * H, mask=mask_m[:, None], other=0.0).to(
                tl.float32
            )

            # ---- j = 0
            nb = _mix_row(xv, r0, r1, r2, r3, p0, c00, c10, c20, c30)
            tl.store(rout_ptr + rm + 0 * H, nb, mask=mask_m[:, None])
            nf = nb.to(tl.float32)
            sq += tl.sum(nf * nf, axis=1)
            w = tl.load(fn_ptr + fnb + 0 * H, mask=mask_n[:, None], other=0.0)
            acc = tl.dot(nf, tl.trans(w), acc, input_precision=PRECISION)

            # ---- j = 1
            nb = _mix_row(xv, r0, r1, r2, r3, p1, c01, c11, c21, c31)
            tl.store(rout_ptr + rm + 1 * H, nb, mask=mask_m[:, None])
            nf = nb.to(tl.float32)
            sq += tl.sum(nf * nf, axis=1)
            w = tl.load(fn_ptr + fnb + 1 * H, mask=mask_n[:, None], other=0.0)
            acc = tl.dot(nf, tl.trans(w), acc, input_precision=PRECISION)

            # ---- j = 2
            nb = _mix_row(xv, r0, r1, r2, r3, p2, c02, c12, c22, c32)
            tl.store(rout_ptr + rm + 2 * H, nb, mask=mask_m[:, None])
            nf = nb.to(tl.float32)
            sq += tl.sum(nf * nf, axis=1)
            w = tl.load(fn_ptr + fnb + 2 * H, mask=mask_n[:, None], other=0.0)
            acc = tl.dot(nf, tl.trans(w), acc, input_precision=PRECISION)

            # ---- j = 3
            nb = _mix_row(xv, r0, r1, r2, r3, p3, c03, c13, c23, c33)
            tl.store(rout_ptr + rm + 3 * H, nb, mask=mask_m[:, None])
            nf = nb.to(tl.float32)
            sq += tl.sum(nf * nf, axis=1)
            w = tl.load(fn_ptr + fnb + 3 * H, mask=mask_n[:, None], other=0.0)
            acc = tl.dot(nf, tl.trans(w), acc, input_precision=PRECISION)

        tl.store(
            gout_ptr + pid_s * M * N_OUT + offs_m[:, None] * N_OUT + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )
        tl.store(gsq_ptr + pid_s * M + offs_m, sq, mask=mask_m)

else:  # pragma: no cover -- no Triton in this interpreter

    _mix_row = None  # type: ignore[assignment]
    _mhc_post_prenorm_fused_kernel = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _pick_config(num_tokens: int) -> tuple[int, int, int, int, int]:
    """(BLOCK_M, BLOCK_H, SPLIT_H, num_warps, num_stages).

    BLOCK_M is the fn-amortisation knob: fn costs (M / BLOCK_M) * 1.57 MB of L2
    reads per call, independent of SPLIT_H.  SPLIT_H is the occupancy knob: the
    grid is (M / BLOCK_M) * SPLIT_H CTAs on 48 SMs, and it buys those CTAs
    without duplicating a single byte of DRAM traffic.  BLOCK_H trades register
    and shared-memory pressure (Triton multi-buffers four residual tiles and
    four fn tiles per stage; BLOCK_H=128 already exceeds the 99 KB per-block
    limit on sm_120/121) against loop overhead.

    These are a starting point, not a measurement.  bench/mhc_fused_bench.py
    sweeps them; the winner is pinned with HAREM_MHC_FUSED_BM / _BH / _SPLITH /
    _WARPS / _STAGES.
    """
    # MEASURED, 5 Sep 2026, GB10 (bench/mhc_fused_bench.py, two runs).  The
    # winner is the same at M=2048 and M=4096 and the surface around it is a
    # cliff, not a slope: BLOCK_M=32/SPLIT_H=4/warps=4 is 79 GB/s and
    # BLOCK_M=64/SPLIT_H=1/warps=4 is 40 GB/s against this config's 188-193.
    # Do not "tidy" these numbers without re-running the sweep.
    bm, bh, sh, warps, stages = 16, 64, 2, 4, 2
    return (
        _env_int("HAREM_MHC_FUSED_BM", bm),
        _env_int("HAREM_MHC_FUSED_BH", bh),
        _env_int("HAREM_MHC_FUSED_SPLITH", sh),
        _env_int("HAREM_MHC_FUSED_WARPS", warps),
        _env_int("HAREM_MHC_FUSED_STAGES", stages),
    )


# --------------------------------------------------------------------------- #
# low-level entry point (the bench calls this directly)
# --------------------------------------------------------------------------- #
def mhc_post_prenorm_fused(
    comb_mix: torch.Tensor,  # (M, HC, HC)         fp32
    residual_in: torch.Tensor,  # (M, HC, H)          bf16
    post_mix: torch.Tensor,  # (M, HC)             fp32
    x_in: torch.Tensor,  # (M, H)              bf16
    fn: torch.Tensor,  # (N_OUT, HC*H)       fp32
    gemm_out: torch.Tensor,  # (SPLIT_H, M, N_OUT) fp32  <- written
    gemm_sqrsum: torch.Tensor,  # (SPLIT_H, M)        fp32  <- written
    residual_out: torch.Tensor,  # (M, HC, H)          bf16  <- written
    hc: int,
    hidden: int,
    n_out: int,
    block_m: int | None = None,
    block_h: int | None = None,
    split_h: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    precision: str | None = None,
) -> None:
    """Fused k1 (post mapping) + k2 (pre-norm GEMM + sqrsum).  Raises on misuse.

    ``gemm_out`` / ``gemm_sqrsum`` carry SPLIT_H partial results; hand the same
    SPLIT_H to k3 as its ``n_splits``.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton unavailable")
    if hc != HC_FIXED:
        raise ValueError(f"hc_mult={hc}: this kernel is written out for 4")
    m = residual_in.shape[0]
    bm, bh, sh, warps, stages = _pick_config(m)
    bm = block_m or bm
    bh = block_h or bh
    sh = split_h or sh
    warps = num_warps or warps
    stages = num_stages or stages
    if hidden % (bh * sh):
        raise ValueError(
            f"hidden={hidden} not a multiple of BLOCK_H*SPLIT_H={bh}*{sh}"
        )
    if gemm_out.shape[0] != sh or gemm_sqrsum.shape[0] != sh:
        raise ValueError(
            f"gemm buffers carry {gemm_out.shape[0]}/{gemm_sqrsum.shape[0]} "
            f"splits, kernel produces {sh}"
        )
    prec = precision or PREC

    grid = (triton.cdiv(m, bm), sh)
    _mhc_post_prenorm_fused_kernel[grid](
        x_in,
        residual_in,
        post_mix,
        comb_mix,
        fn,
        residual_out,
        gemm_out,
        gemm_sqrsum,
        m,
        hidden,
        N_OUT=n_out,
        N_PAD=triton.next_power_of_2(n_out),
        BLOCK_M=bm,
        BLOCK_H=bh,
        SPLIT_H=sh,
        PRECISION=prec,
        num_warps=warps,
        num_stages=stages,
    )


# --------------------------------------------------------------------------- #
# vLLM entry point (called from the patched tilelang.py)
# --------------------------------------------------------------------------- #
def harem_mhc_fused_post_pre(
    residual_flat: torch.Tensor,  # (M, HC, H)    bf16
    x_flat: torch.Tensor,  # (M, H)        bf16
    post_layer_mix_flat: torch.Tensor,  # (M, HC)       fp32
    comb_res_mix_flat: torch.Tensor,  # (M, HC, HC)   fp32
    fn: torch.Tensor,  # (N_OUT, HC*H) fp32
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    hc_mult: int,
    hidden_size: int,
):
    """Fused large-M mHC post+pre.

    Returns ``(residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur)`` or
    ``None``.  ``None`` means "not handled" and the caller must run the stock
    three-kernel path -- that is the only failure mode this function has.
    """
    global _DISABLED_REASON, _LOGGED

    if _DISABLED_REASON is not None or not ENABLED or not HAS_TRITON:
        return None

    num_tokens = residual_flat.shape[0]
    if num_tokens < MIN_M:
        return None

    n_out = fn.shape[0]
    try:
        # Fail-closed preconditions.  Anything unexpected -> stock path, once.
        assert hc_mult == HC_FIXED, f"hc_mult={hc_mult}"
        assert n_out == hc_mult * 2 + hc_mult * hc_mult, f"n_out={n_out}"
        assert fn.shape == (n_out, hc_mult * hidden_size), f"fn {tuple(fn.shape)}"
        assert residual_flat.shape == (num_tokens, hc_mult, hidden_size)
        assert x_flat.shape == (num_tokens, hidden_size)
        assert post_layer_mix_flat.shape == (num_tokens, hc_mult)
        assert comb_res_mix_flat.shape == (num_tokens, hc_mult, hc_mult)
        assert residual_flat.dtype == torch.bfloat16
        assert x_flat.dtype == torch.bfloat16
        assert post_layer_mix_flat.dtype == torch.float32
        assert comb_res_mix_flat.dtype == torch.float32
        assert fn.dtype == torch.float32
        for t in (residual_flat, x_flat, post_layer_mix_flat, comb_res_mix_flat, fn):
            assert t.is_contiguous(), "non-contiguous input"

        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_pre_big_fuse_tilelang,
            mhc_pre_big_fuse_with_norm_tilelang,
        )

        _bm, _bh, split_h, _w, _s = _pick_config(num_tokens)
        assert hidden_size % (_bh * split_h) == 0, (
            f"hidden_size={hidden_size} not a multiple of "
            f"BLOCK_H*SPLIT_H={_bh}*{split_h}"
        )

        dev = residual_flat.device
        gemm_out_mul = torch.empty(
            split_h, num_tokens, n_out, dtype=torch.float32, device=dev
        )
        gemm_out_sqrsum = torch.empty(
            split_h, num_tokens, dtype=torch.float32, device=dev
        )
        residual_cur = torch.empty_like(residual_flat)
        post_mix_cur = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=dev)
        comb_mix_cur = torch.empty(
            num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=dev
        )
        layer_input_cur = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=dev
        )

        mhc_post_prenorm_fused(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            n_out,
        )

        if norm_weight is None:
            mhc_pre_big_fuse_tilelang(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual_cur,
                post_mix_cur,
                comb_mix_cur,
                layer_input_cur,
                hidden_size,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                split_h,  # n_splits
                hc_mult,
            )
        else:
            mhc_pre_big_fuse_with_norm_tilelang(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual_cur,
                post_mix_cur,
                comb_mix_cur,
                layer_input_cur,
                norm_weight,
                hidden_size,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                norm_eps,
                split_h,  # n_splits
                hc_mult,
            )
    except Exception as ex:  # noqa: BLE001
        _DISABLED_REASON = f"{type(ex).__name__}: {ex}"
        try:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "HAREM hc-fusion disabled for this process (%s); falling back to "
                "the stock 3-kernel mHC path.",
                _DISABLED_REASON,
            )
        except Exception:  # noqa: BLE001
            print(f"[harem-hc-fusion] disabled: {_DISABLED_REASON}", flush=True)
        return None

    if DEBUG and not _LOGGED:
        _LOGGED = True
        bm, bh, sh, warps, stages = _pick_config(num_tokens)
        print(
            f"[harem-hc-fusion] active: M={num_tokens} BLOCK_M={bm} BLOCK_H={bh} "
            f"SPLIT_H={sh} warps={warps} stages={stages} prec={PREC} "
            f"min_m={MIN_M}",
            flush=True,
        )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

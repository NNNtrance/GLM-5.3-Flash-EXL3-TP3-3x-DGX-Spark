#!/usr/bin/env python3
"""Fail-closed check + timing for the EP-aware exl3_moe_combine / had_in change.

Model-free: synthetic tensors of the real GLM-5.3-Flash shapes (hidden 4096,
intermediate 2048, 288 global experts, 96 local, top_k 8, 4 bits, mcg).

Three questions, each answered by construction rather than by eyeballing:

 1. Does the gemm ever read a retired block's activations?  a13 is poisoned with
    NaN before the transform; if any retired row were read the output would be
    NaN. Run against the pre-zeroed control, bitwise.
 2. Is combine(expert_ids, block_m) the same function as masked_fill_ + combine?
    rows_out is poisoned with NaN on the retired rows first, so a combine that
    reads them cannot accidentally agree.
 3. With EP off, is passing expert_ids a no-op?  Bitwise against the 4-arg call.

Then times both routes so the change can be priced.
"""
import argparse, json, sys
import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from cuda_exl3 import _C  # noqa: F401

ops = torch.ops.cuda_exl3_C
DEV = "cuda"; H = 4096; TOPK = 8; E_GLOBAL = 288; BITS = 4; CB = 1; TILE = 16


def ep_aware() -> bool:
    return any(a.name == "expert_ids"
               for a in ops.exl3_moe_combine.default._schema.arguments)


def ladder(rows, ne):
    per = rows / max(ne, 1)
    return 16 if per < 16 else 32 if per < 48 else 64 if per < 96 else 128


def weights(E, I, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return dict(
        I=I, E=E,
        w13_tr=torch.randint(-32768, 32767, (E, H // TILE, 2 * I // TILE, TILE * BITS),
                             dtype=torch.int16, device=DEV, generator=g),
        w2_tr=torch.randint(-32768, 32767, (E, I // TILE, H // TILE, TILE * BITS),
                            dtype=torch.int16, device=DEV, generator=g),
        w13_suh=(torch.randn((E, 2, H), device=DEV, generator=g) * 0.05).half(),
        w2_suh=(torch.randn((E, 1, I), device=DEV, generator=g) * 0.05).half(),
        w13_svh=(torch.randn((E, 2 * I), device=DEV, generator=g) * 0.05).half(),
        w2_svh=(torch.randn((E, H), device=DEV, generator=g) * 0.05).half())


def routing(M, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(E_GLOBAL, generator=g)[:TOPK] for _ in range(M)])
    w = torch.rand((M, TOPK), generator=g); w = w / w.sum(1, keepdim=True)
    return ids.to(DEV, torch.int32), w.to(DEV, torch.float32)


def pipeline(W, emap, M, ids, tw, poison_a13, mask_route, block_m=None):
    """One MoE layer. mask_route: 'python' = masked_fill_ + 4-arg combine,
    'kernel' = 6-arg combine and no clearing pass."""
    I = W["I"]
    torch.manual_seed(11)
    x = (torch.randn((M, H), device=DEV) * 0.02).to(torch.bfloat16)
    bm = block_m or ladder(M * TOPK, E_GLOBAL)
    sids, eids, nrows = moe_align_block_size(ids, bm, E_GLOBAL, expert_map=emap,
                                             pad_sorted_ids=True)
    sids = sids.int(); eids = eids.int(); nrows = nrows.int()
    rows = min(eids.numel() * bm, sids.numel())
    eids = eids[: rows // bm]

    a13 = torch.empty((2, rows, H), dtype=torch.half, device=DEV)
    a13.fill_(float("nan") if poison_a13 else 0.0)
    ops.exl3_moe_had_in(x.contiguous(), a13, W["w13_suh"], sids, eids, nrows, bm, TOPK, M * TOPK)
    inter = ops.exl3_moe_gemm(a13, W["w13_tr"], W["w13_suh"], W["w13_svh"], eids, nrows,
                              [I, I], CB, bm, torch.bfloat16)
    a2 = torch.empty((1, rows, I), dtype=torch.half, device=DEV)
    a2.fill_(float("nan") if poison_a13 else 0.0)
    ops.exl3_moe_glu_had_in(inter, a2, W["w2_suh"], eids, nrows, bm)
    rows_out = ops.exl3_moe_gemm(a2, W["w2_tr"], W["w2_suh"], W["w2_svh"], eids, nrows,
                                 [H], CB, bm, torch.bfloat16)
    # Poison every row the gemm retires, so a route that reads one cannot pass.
    live = (eids >= 0).repeat_interleave(bm)[: rows_out.shape[0]]
    rows_out.masked_fill_(~live.unsqueeze(1), float("nan"))
    if mask_route == "python":
        rows_out = rows_out.masked_fill(~live.unsqueeze(1), 0)
        out = ops.exl3_moe_combine(rows_out, sids, tw, M)
    else:
        out = ops.exl3_moe_combine(rows_out, sids, tw, M, eids, bm)
    return out, bm, int(nrows.item()), rows


def timeit(fn, iters=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,8,64,512,2048")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    Ms = [int(v) for v in a.ms.split(",")]
    have = ep_aware()
    print(f"exl3_moe_combine is EP-aware: {have}")
    emap = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=DEV)
    emap[:96] = torch.arange(96, dtype=torch.int32, device=DEV)

    fails = 0
    W = weights(96, 2048)
    print(f"\n{'M':>6} {'blk':>4} {'n_rows':>8} {'rows':>7} | {'poison a13':>11} "
          f"{'kernel==python':>15} | {'python us':>10} {'kernel us':>10} {'gain':>7}")
    rec = []
    for M in Ms:
        ids, tw = routing(M, 1234 + M)
        ref, bm, nr, rows = pipeline(W, emap, M, ids, tw, False, "python")
        pz, _, _, _ = pipeline(W, emap, M, ids, tw, True, "python")
        ok_poison = torch.equal(ref, pz) and torch.isfinite(ref).all().item()
        if have:
            ker, _, _, _ = pipeline(W, emap, M, ids, tw, True, "kernel")
            ok_kernel = torch.equal(ref, ker)
            t_py = timeit(lambda: pipeline(W, emap, M, ids, tw, False, "python"), 10, 3)
            t_kr = timeit(lambda: pipeline(W, emap, M, ids, tw, False, "kernel"), 10, 3)
            gain = f"{(t_py-t_kr)/t_py*100:5.1f}%"
        else:
            ok_kernel = None; t_py = timeit(lambda: pipeline(W, emap, M, ids, tw, False, "python"), 10, 3)
            t_kr = float("nan"); gain = "n/a"
        fails += (not ok_poison) + (ok_kernel is False)
        print(f"{M:>6} {bm:>4} {nr:>8} {rows:>7} | {'PASS' if ok_poison else 'FAIL':>11} "
              f"{('PASS' if ok_kernel else 'FAIL') if ok_kernel is not None else 'skip':>15} | "
              f"{t_py:>10.1f} {t_kr:>10.1f} {gain:>7}")
        rec.append(dict(M=M, block_m=bm, n_rows=nr, rows=rows, poison=ok_poison,
                        kernel_eq=ok_kernel, us_python=t_py, us_kernel=t_kr))
    del W; torch.cuda.empty_cache()

    # EP off: passing expert_ids must change nothing.
    if have:
        W2 = weights(288, 1024)
        print(f"\nEP off (288 local, I=1024): 6-arg combine == 4-arg combine")
        for M in Ms:
            ids, tw = routing(M, 1234 + M)
            r4, bm, _, _ = pipeline(W2, None, M, ids, tw, False, "python")
            r6, _, _, _ = pipeline(W2, None, M, ids, tw, False, "kernel")
            ok = torch.equal(r4, r6)
            fails += (not ok)
            print(f"   M={M:<6} blk={bm:<4} {'PASS' if ok else 'FAIL'}")
        del W2; torch.cuda.empty_cache()

    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    if a.out:
        json.dump(dict(ep_aware=have, fails=fails, rows=rec), open(a.out, "w"), indent=1)
    sys.exit(1 if fails else 0)


main()

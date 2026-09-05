#!/usr/bin/env python3
"""Is the EP throughput regression routing LOAD IMBALANCE?

Under expert parallel a rank computes only the (token, expert) pairs routed to
the experts it owns, and the step ends when the SLOWEST rank finishes. Under
tensor-sliced experts every rank computes every pair over a narrower
intermediate, so the ranks are balanced by construction. If the routing spread
is wide, EP pays max() where TP pays mean().

Model-free: synthetic GLM-5.3-Flash shapes (hidden 4096, intermediate 2048, 288
experts, top_k 8, 4 bits, mcg). One set of expert weights, three expert_maps.
The routing draw is shared by all three ranks, exactly as it is in the engine.

Reported per (M, draw): pairs owned, distinct experts touched, live/local
blocks, and the measured MoE stage time for each rank -- then max/mean, which
is the factor EP pays over a perfectly balanced arm.
"""
import argparse, json, statistics as st, sys
import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from cuda_exl3 import _C  # noqa: F401

ops = torch.ops.cuda_exl3_C
DEV = "cuda"; H = 4096; TOPK = 8; E_GLOBAL = 288; BITS = 4; CB = 1; TILE = 16
EP = 3; E_LOCAL = E_GLOBAL // EP
REPS = 10


def ladder(rows, ne):
    per = rows / max(ne, 1)
    return 16 if per < 16 else 32 if per < 48 else 64 if per < 96 else 128


def weights(E, I, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return dict(I=I, E=E,
        w13_tr=torch.randint(-32768, 32767, (E, H // TILE, 2 * I // TILE, TILE * BITS),
                             dtype=torch.int16, device=DEV, generator=g),
        w2_tr=torch.randint(-32768, 32767, (E, I // TILE, H // TILE, TILE * BITS),
                            dtype=torch.int16, device=DEV, generator=g),
        w13_suh=(torch.randn((E, 2, H), device=DEV, generator=g) * .05).half(),
        w2_suh=(torch.randn((E, 1, I), device=DEV, generator=g) * .05).half(),
        w13_svh=(torch.randn((E, 2 * I), device=DEV, generator=g) * .05).half(),
        w2_svh=(torch.randn((E, H), device=DEV, generator=g) * .05).half())


def routing(M, seed, skew):
    """top_k experts per token. skew=0 uniform; skew>0 = Zipf-like router bias."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if skew <= 0:
        ids = torch.stack([torch.randperm(E_GLOBAL, generator=g)[:TOPK] for _ in range(M)])
    else:
        # A fixed per-expert affinity, drawn once per "layer", then sampled
        # without replacement per token. skew is the exponent of a power law.
        pref = torch.rand(E_GLOBAL, generator=g).pow(skew)
        pref = pref / pref.sum()
        ids = torch.stack([torch.multinomial(pref, TOPK, replacement=False, generator=g)
                           for _ in range(M)])
    w = torch.rand((M, TOPK), generator=g); w = w / w.sum(1, keepdim=True)
    return ids.to(DEV, torch.int32), w.to(DEV, torch.float32)


def emap_for(rank):
    m = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=DEV)
    m[rank * E_LOCAL:(rank + 1) * E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32, device=DEV)
    return m


def timeit(fn, iters=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def build(W, emap, M, ids, tw, x, force_bm=None):
    """Everything a rank needs for one MoE layer, allocated once."""
    I = W["I"]
    n_align = E_GLOBAL if emap is not None else W["E"]
    bm = force_bm or ladder(M * TOPK, n_align)
    sids, eids, nrows = moe_align_block_size(ids, bm, n_align, expert_map=emap,
                                             pad_sorted_ids=True)
    sids = sids.int(); eids = eids.int(); nrows = nrows.int()
    rows = min(eids.numel() * bm, sids.numel()); eids = eids[: rows // bm]
    a13 = torch.empty((2, rows, H), dtype=torch.half, device=DEV)
    a2 = torch.empty((1, rows, I), dtype=torch.half, device=DEV)
    inter = torch.empty((rows, 2 * I), dtype=torch.bfloat16, device=DEV)

    def run():
        ops.exl3_moe_had_in(x, a13, W["w13_suh"], sids, eids, nrows, bm, TOPK, M * TOPK)
        it = ops.exl3_moe_gemm(a13, W["w13_tr"], W["w13_suh"], W["w13_svh"], eids, nrows,
                               [I, I], CB, bm, torch.bfloat16)
        ops.exl3_moe_glu_had_in(it, a2, W["w2_suh"], eids, nrows, bm)
        ro = ops.exl3_moe_gemm(a2, W["w2_tr"], W["w2_suh"], W["w2_svh"], eids, nrows,
                               [H], CB, bm, torch.bfloat16)
        return ops.exl3_moe_combine(ro, sids, tw, M)

    return dict(run=run, bm=bm, local_blocks=int((eids >= 0).sum().item()),
                n_rows=int(nrows.item()), keep=(a13, a2, inter, sids, eids, nrows))


def timed_once(run):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(REPS): run()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / REPS * 1000.0


def interleaved(runs, trials=7):
    """Time N candidates trial by trial, round-robin, and take the MEDIAN.

    Timing them one after another instead makes clock drift look like a
    difference between them -- which it did on the first pass here: rank 2 came
    out 64 % slower than rank 0 at M=2048 while owning the SAME number of pairs.
    """
    acc = [[] for _ in runs]
    for r in runs: 
        for _ in range(3): r()
    torch.cuda.synchronize()
    for _ in range(trials):
        for i, r in enumerate(runs):
            acc[i].append(timed_once(r))
    return [st.median(a) for a in acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="8,32,64,2048")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--with-tp", action="store_true", help="also time the TP-sliced arm")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    Ms = [int(v) for v in a.ms.split(",")]
    print(f"GB10  EP={EP}  {E_LOCAL} local of {E_GLOBAL}  top_k={TOPK}  "
          f"skew={a.skew}  draws={a.draws}")
    Wep = weights(E_LOCAL, 2048)
    maps = [emap_for(r) for r in range(EP)]
    rec = []
    print(f"\n{'M':>6} {'draw':>5} {'blk':>4} | {'pairs r0/r1/r2':>22} | "
          f"{'us r0/r1/r2':>24} | {'max/mean':>9} {'pairmax/mean':>13}")
    for M in Ms:
        x = (torch.randn((M, H), device=DEV) * .02).to(torch.bfloat16).contiguous()
        agg = []
        for d in range(a.draws):
            ids, tw = routing(M, 9000 + 31 * d + M, a.skew)
            cnt = [int(((ids >= r * E_LOCAL) & (ids < (r + 1) * E_LOCAL)).sum().item())
                   for r in range(EP)]
            built = [build(Wep, maps[r], M, ids, tw, x) for r in range(EP)]
            us = interleaved([b["run"] for b in built])
            bm = built[0]["bm"]; loc = [b["local_blocks"] for b in built]
            del built
            torch.cuda.empty_cache()
            mm = max(us) / (sum(us) / EP)
            pm = max(cnt) / (sum(cnt) / EP)
            agg.append((mm, pm, sum(us) / EP, max(us)))
            print(f"{M:>6} {d:>5} {bm:>4} | {'/'.join(str(c) for c in cnt):>22} | "
                  f"{'/'.join(f'{u:.0f}' for u in us):>24} | {mm:>9.3f} {pm:>13.3f}")
            rec.append(dict(M=M, draw=d, block_m=bm, pairs=cnt, us=us,
                            local_blocks=loc, max_over_mean=mm, pairmax_over_mean=pm))
        mms = [t[0] for t in agg]
        print(f"{M:>6} {'MEAN':>5}      | {'':22} | "
              f"mean {sum(t[2] for t in agg)/len(agg):>8.0f} us            | "
              f"{sum(mms)/len(mms):>9.3f} {sum(t[1] for t in agg)/len(agg):>13.3f}")
    if a.with_tp:
        del Wep; torch.cuda.empty_cache()
        Wtp = weights(E_GLOBAL, 1024)
        print(f"\nTP-sliced reference arm (288 local, I=1024, every rank does every pair):")
        for M in Ms:
            x = (torch.randn((M, H), device=DEV) * .02).to(torch.bfloat16).contiguous()
            ids, tw = routing(M, 9000 + M, a.skew)
            b = build(Wtp, None, M, ids, tw, x)
            u = interleaved([b["run"]])[0]; bm = b["bm"]; del b
            print(f"   M={M:<6} blk={bm:<4} {u:>9.0f} us   (balanced by construction)")
            rec.append(dict(M=M, arm="TP288", block_m=bm, us=[u]))
    if a.out: json.dump(rec, open(a.out, "w"), indent=1)


main()

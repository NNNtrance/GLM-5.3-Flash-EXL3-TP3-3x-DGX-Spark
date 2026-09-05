#!/usr/bin/env python3
"""EP vs TP block-padding geometry and per-stage kernel timing for cuda-exl3 MoE.

Two arms, both a single rank's share of one GLM-5.3-Flash MoE layer:

  EP   : 96 local experts of 288 global, intermediate 2048 (whole trellis),
         moe_align_block_size(num_experts=288, expert_map=<96 local>)
  TP288: 288 local experts, intermediate 1024 (trellis sliced by 2),
         moe_align_block_size(num_experts=288, expert_map=None)

TP288 is what a TP=2 rank runs. EP is what a TP=3 rank runs. Same routing draw
for both so the comparison is paired.
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from cuda_exl3 import _C  # registers torch.ops.cuda_exl3_C

ops = torch.ops.cuda_exl3_C
DEV = "cuda"
H = 4096
TOPK = 8
E_GLOBAL = 288
BITS = 4
CB = 1  # mcg
TILE = 16


def ladder(rows, num_experts):
    per = rows / max(num_experts, 1)
    if per < 16: return 16
    if per < 48: return 32
    if per < 96: return 64
    return 128


def make_routing(M, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # top-k without replacement per token, uniform over the 288 global experts
    ids = torch.stack([torch.randperm(E_GLOBAL, generator=g)[:TOPK] for _ in range(M)])
    w = torch.rand((M, TOPK), generator=g)
    w = w / w.sum(1, keepdim=True)
    return ids.to(DEV, torch.int32), w.to(DEV, torch.float32)


def make_weights(E, I):
    """Synthetic EXL3 expert weights. Values are irrelevant to timing."""
    g = torch.Generator(device=DEV).manual_seed(7)
    w13_tr = torch.randint(-32768, 32767, (E, H // TILE, 2 * I // TILE, TILE * BITS),
                           dtype=torch.int16, device=DEV, generator=g)
    w2_tr = torch.randint(-32768, 32767, (E, I // TILE, H // TILE, TILE * BITS),
                          dtype=torch.int16, device=DEV, generator=g)
    w13_suh = torch.randn((E, 2, H), device=DEV, generator=g).half() * 0.05
    w2_suh = torch.randn((E, 1, I), device=DEV, generator=g).half() * 0.05
    w13_svh = torch.randn((E, 2 * I), device=DEV, generator=g).half() * 0.05
    w2_svh = torch.randn((E, H), device=DEV, generator=g).half() * 0.05
    return dict(w13_tr=w13_tr, w2_tr=w2_tr, w13_suh=w13_suh, w2_suh=w2_suh,
                w13_svh=w13_svh, w2_svh=w2_svh, I=I, E=E)


def align(topk_ids, block_m, emap):
    n_align = E_GLOBAL
    sorted_ids, expert_ids, n_rows = moe_align_block_size(
        topk_ids, block_m, n_align, expert_map=emap, pad_sorted_ids=True)
    return sorted_ids.int(), expert_ids.int(), n_rows.int()


def geometry(M, block_m, emap, topk_ids):
    sids, eids, nrows = align(topk_ids, block_m, emap)
    rows_alloc = min(eids.numel() * block_m, sids.numel())
    eids_c = eids[: rows_alloc // block_m]
    live_blocks = int((nrows.item() + block_m - 1) // block_m)
    eids_live = eids_c[:live_blocks]
    local_blocks = int((eids_live >= 0).sum().item())
    return dict(
        M=M, block_m=block_m, real_rows=M * TOPK,
        rows_alloc=int(rows_alloc), n_rows=int(nrows.item()),
        blocks_alloc=int(eids_c.numel()), live_blocks=live_blocks,
        local_blocks=local_blocks,
        pad_ratio_live=round(int(nrows.item()) / (M * TOPK), 3),
        pad_ratio_alloc=round(int(rows_alloc) / (M * TOPK), 3),
    )


def timeit(fn, iters=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0   # us


def run_arm(name, W, emap, M, topk_ids, topk_w, force_bm=None, zero_mode="fill"):
    I = W["I"]
    x = torch.randn((M, H), device=DEV, dtype=torch.bfloat16) * 0.02
    rows_hint = M * TOPK
    block_m = force_bm or ladder(rows_hint, E_GLOBAL)
    sids, eids, nrows = align(topk_ids, block_m, emap)
    rows = min(eids.numel() * block_m, sids.numel())
    eids = eids[: rows // block_m]
    xc = x.contiguous()

    a13 = torch.empty((2, rows, H), dtype=torch.half, device=DEV)
    a2 = torch.empty((1, rows, I), dtype=torch.half, device=DEV)

    def f_align():   align(topk_ids, block_m, emap)
    def f_had():     ops.exl3_moe_had_in(xc, a13, W["w13_suh"], sids, eids, nrows, block_m, TOPK, M * TOPK)
    ops.exl3_moe_had_in(xc, a13, W["w13_suh"], sids, eids, nrows, block_m, TOPK, M * TOPK)
    def f_g1():      return ops.exl3_moe_gemm(a13, W["w13_tr"], W["w13_suh"], W["w13_svh"], eids, nrows, [I, I], CB, block_m, torch.bfloat16)
    inter = f_g1()
    def f_glu():     ops.exl3_moe_glu_had_in(inter, a2, W["w2_suh"], eids, nrows, block_m)
    ops.exl3_moe_glu_had_in(inter, a2, W["w2_suh"], eids, nrows, block_m)
    def f_g2():      return ops.exl3_moe_gemm(a2, W["w2_tr"], W["w2_suh"], W["w2_svh"], eids, nrows, [H], CB, block_m, torch.bfloat16)
    rows_out = f_g2()
    valid_mask = None
    def f_zero():
        v = (eids >= 0).repeat_interleave(block_m)[: rows_out.shape[0]]
        rows_out.masked_fill_(~v.unsqueeze(1), 0)
    def f_comb():    return ops.exl3_moe_combine(rows_out, sids, topk_w, M)

    t = {}
    t["align"] = timeit(f_align)
    t["had_in"] = timeit(f_had)
    t["gemm_w13"] = timeit(f_g1)
    t["glu_had"] = timeit(f_glu)
    t["gemm_w2"] = timeit(f_g2)
    t["zero"] = timeit(f_zero) if emap is not None and zero_mode != "off" else 0.0
    t["combine"] = timeit(f_comb)
    t["TOTAL"] = sum(t.values())
    g = geometry(M, block_m, emap, topk_ids)
    del a13, a2, inter, rows_out
    torch.cuda.empty_cache()
    return dict(arm=name, block_m=block_m, geom=g, us=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="8,64,512,2048")
    ap.add_argument("--out", default="/cache/ep_moe_bench.json")
    ap.add_argument("--bm-sweep", default="16,32,64,128")
    a = ap.parse_args()
    Ms = [int(v) for v in a.ms.split(",")]
    torch.cuda.init()
    print(f"device={torch.cuda.get_device_name(0)} sms={torch.cuda.get_device_properties(0).multi_processor_count}")

    emap = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=DEV)
    local = torch.arange(0, 96, dtype=torch.int32, device=DEV)   # rank 0 owns 0..95
    emap[:96] = local

    out = {"geometry": [], "timing": [], "bm_sweep": []}

    # ---- geometry only (no weights needed) -------------------------------
    print("\n=== GEOMETRY: padded rows vs real rows ===")
    print(f"{'M':>6} {'blk':>4} {'arm':>6} {'real':>7} {'n_rows':>8} {'alloc':>8} "
          f"{'pad/real':>9} {'alloc/real':>11} {'blks':>6} {'live':>6} {'local':>6}")
    for M in Ms:
        bm_lad = ladder(M * TOPK, E_GLOBAL)
        ids, _ = make_routing(M, 1234 + M)
        for bm in sorted(set([bm_lad] + [int(v) for v in a.bm_sweep.split(",")])):
            for arm, m in (("EP", emap), ("TP288", None)):
                g = geometry(M, bm, m, ids)
                g["arm"] = arm; g["is_ladder"] = (bm == bm_lad)
                out["geometry"].append(g)
                star = "*" if bm == bm_lad else " "
                print(f"{M:>6} {bm:>3}{star} {arm:>6} {g['real_rows']:>7} {g['n_rows']:>8} "
                      f"{g['rows_alloc']:>8} {g['pad_ratio_live']:>9} {g['pad_ratio_alloc']:>11} "
                      f"{g['blocks_alloc']:>6} {g['live_blocks']:>6} {g['local_blocks']:>6}")

    # ---- timing ----------------------------------------------------------
    print("\n=== TIMING (us per MoE layer, one rank) ===")
    for arm, I, E, m in (("EP", 2048, 96, emap), ("TP288", 1024, 288, None)):
        W = make_weights(E, I)
        free, tot = torch.cuda.mem_get_info()
        print(f"-- arm {arm}: E={E} I={I}  weights {(W['w13_tr'].numel()*2+W['w2_tr'].numel()*2)/2**30:.2f} GiB, free {free/2**30:.1f} GiB")
        for M in Ms:
            ids, w = make_routing(M, 1234 + M)
            r = run_arm(arm, W, m, M, ids, w)
            out["timing"].append(r)
            u = r["us"]
            print(f"   M={M:<5} blk={r['block_m']:<4} had={u['had_in']:7.1f} g13={u['gemm_w13']:8.1f} "
                  f"glu={u['glu_had']:7.1f} g2={u['gemm_w2']:8.1f} zero={u['zero']:7.1f} "
                  f"comb={u['combine']:7.1f} align={u['align']:7.1f}  TOTAL={u['TOTAL']:8.1f}")
        # block_m sweep on the EP arm only
        if arm == "EP":
            print(f"-- EP block_m sweep")
            for M in Ms:
                ids, w = make_routing(M, 1234 + M)
                for bm in [int(v) for v in a.bm_sweep.split(",")]:
                    try:
                        r = run_arm(arm, W, m, M, ids, w, force_bm=bm)
                    except Exception as ex:
                        print(f"   M={M} bm={bm} FAILED {ex}"); continue
                    r["forced_bm"] = bm
                    out["bm_sweep"].append(r)
                    u = r["us"]
                    print(f"   M={M:<5} bm={bm:<4} TOTAL={u['TOTAL']:8.1f}  "
                          f"(g13={u['gemm_w13']:7.1f} g2={u['gemm_w2']:7.1f} zero={u['zero']:6.1f} "
                          f"had={u['had_in']:6.1f} n_rows={r['geom']['n_rows']} local_blk={r['geom']['local_blocks']})")
        del W
        torch.cuda.empty_cache()

    # ---- export table for the plugin author -----------------------------
    print("\n=== EXPORT: block_m under EP (96 owned of 288, top_k 8, GB10) ===")
    print("| M (tokens) | rows=M*8 | ladder(local E=96) | ladder(global E=288) | "
          "us@16 | us@32 | us@64 | us@128 | best |")
    print("|---|---|---|---|---|---|---|---|---|")
    for M in Ms:
        r = {x["forced_bm"]: x["us"]["TOTAL"] for x in out["bm_sweep"] if x["geom"]["M"] == M}
        if not r: continue
        best = min(r, key=r.get)
        cells = " | ".join(f"{r.get(b, float('nan')):.0f}" for b in (16, 32, 64, 128))
        print(f"| {M} | {M*TOPK} | {ladder(M*TOPK, 96)} | {ladder(M*TOPK, E_GLOBAL)} | "
              f"{cells} | **{best}** |")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

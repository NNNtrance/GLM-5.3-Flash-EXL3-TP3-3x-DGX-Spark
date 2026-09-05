#!/usr/bin/env python3
"""Per-kernel MoE stage timing, run identically under two cuda-exl3 builds.

Purpose: explain why the image built on upstream 60d5349 (f906f00's
combine-side skip, gemm RETURNS on e<0) measures ~10 % slower end to end than
the image built on 77513d2 + our patches (gemm ZEROES the retired tile, combine
stages inv[]/w[] in shared memory).

Both builds expose the same op schema, so one script covers both:
   exl3_moe_combine(rows_out, sorted_ids, topk_weights, M, expert_ids=None, block_m=0)

Arms: EP (96 local of 288 global, intermediate 2048) and, optionally, non-EP
(288 local, expert_map=None) so the surplus-tail behaviour is visible too.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from cuda_exl3 import _C  # noqa: F401  (registers torch.ops.cuda_exl3_C)

ops = torch.ops.cuda_exl3_C
DEV = "cuda"
H = 4096
TOPK = 8
E_GLOBAL = 288
BITS = 4
CB = 1
TILE = 16


def ladder(rows, num_experts):
    per = rows / max(num_experts, 1)
    if per < 16: return 16
    if per < 48: return 32
    if per < 96: return 64
    return 128


def make_routing(M, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(E_GLOBAL, generator=g)[:TOPK] for _ in range(M)])
    w = torch.rand((M, TOPK), generator=g)
    w = w / w.sum(1, keepdim=True)
    return ids.to(DEV, torch.int32), w.to(DEV, torch.float32)


def make_weights(E, I):
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
    sorted_ids, expert_ids, n_rows = moe_align_block_size(
        topk_ids, block_m, E_GLOBAL, expert_map=emap, pad_sorted_ids=True)
    return sorted_ids.int(), expert_ids.int(), n_rows.int()


def timeit(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def run(M, W, emap, tag, seed):
    I = W["I"]
    ids, w = make_routing(M, seed)
    block_m = ladder(M * TOPK, E_GLOBAL)
    sids, eids, nrows = align(ids, block_m, emap)
    rows = min(eids.numel() * block_m, sids.numel())
    eids = eids[: rows // block_m]
    x = (torch.randn((M, H), device=DEV, dtype=torch.bfloat16) * 0.02).contiguous()

    a13 = torch.empty((2, rows, H), dtype=torch.half, device=DEV)
    a2 = torch.empty((1, rows, I), dtype=torch.half, device=DEV)

    def f_had():
        ops.exl3_moe_had_in(x, a13, W["w13_suh"], sids, eids, nrows, block_m, TOPK, M * TOPK)
    f_had()

    def f_g13():
        return ops.exl3_moe_gemm(a13, W["w13_tr"], W["w13_suh"], W["w13_svh"],
                                 eids, nrows, [I, I], CB, block_m, torch.bfloat16)
    inter = f_g13()

    def f_glu():
        ops.exl3_moe_glu_had_in(inter, a2, W["w2_suh"], eids, nrows, block_m)
    f_glu()

    def f_g2():
        return ops.exl3_moe_gemm(a2, W["w2_tr"], W["w2_suh"], W["w2_svh"],
                                 eids, nrows, [H], CB, block_m, torch.bfloat16)
    rows_out = f_g2()

    def f_comb4():
        return ops.exl3_moe_combine(rows_out, sids, w, M)

    def f_comb6():
        return ops.exl3_moe_combine(rows_out, sids, w, M, eids, block_m)

    live_blocks = int((nrows.item() + block_m - 1) // block_m)
    local_blocks = int((eids[:live_blocks] >= 0).sum().item())

    t = {}
    t["had_in"] = timeit(f_had)
    t["gemm_w13"] = timeit(f_g13)
    t["glu_had_in"] = timeit(f_glu)
    t["gemm_w2"] = timeit(f_g2)
    t["combine4"] = timeit(f_comb4)
    t["combine6"] = timeit(f_comb6)
    t["PIPE6"] = (t["had_in"] + t["gemm_w13"] + t["glu_had_in"] + t["gemm_w2"]
                  + t["combine6"])
    row = dict(tag=tag, M=M, block_m=block_m, rows_alloc=rows,
               n_rows=int(nrows.item()), blocks=int(eids.numel()),
               live_blocks=live_blocks, local_blocks=local_blocks, us=t)
    del a13, a2, inter, rows_out
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="8,64,2048")
    ap.add_argument("--arms", default="EP")          # EP,TP288
    ap.add_argument("--label", default="build")
    ap.add_argument("--out", default="/cache/moe_stage.json")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    Ms = [int(v) for v in a.ms.split(",")]
    torch.cuda.init()
    p = torch.cuda.get_device_properties(0)
    print(f"label={a.label} device={p.name} sms={p.multi_processor_count}", flush=True)

    emap = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=DEV)
    emap[:96] = torch.arange(0, 96, dtype=torch.int32, device=DEV)

    out = []
    arms = []
    if "EP" in a.arms:
        arms.append(("EP", 2048, 96, emap))
    if "TP288" in a.arms:
        arms.append(("TP288", 1024, 288, None))
    for arm, I, E, m in arms:
        W = make_weights(E, I)
        free, tot = torch.cuda.mem_get_info()
        print(f"-- {arm}: E={E} I={I} free={free/2**30:.1f} GiB", flush=True)
        print(f"{'M':>6} {'blk':>4} {'live':>5} {'local':>5} {'had':>8} {'g13':>9} "
              f"{'glu':>8} {'g2':>9} {'comb4':>8} {'comb6':>8} {'PIPE6':>9}", flush=True)
        for M in Ms:
            for rep in range(a.repeat):
                r = run(M, W, m, f"{a.label}/{arm}", a.seed + M + rep * 977)
                r["rep"] = rep
                out.append(r)
                u = r["us"]
                print(f"{M:>6} {r['block_m']:>4} {r['live_blocks']:>5} {r['local_blocks']:>5} "
                      f"{u['had_in']:>8.1f} {u['gemm_w13']:>9.1f} {u['glu_had_in']:>8.1f} "
                      f"{u['gemm_w2']:>9.1f} {u['combine4']:>8.1f} {u['combine6']:>8.1f} "
                      f"{u['PIPE6']:>9.1f}", flush=True)
        del W
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

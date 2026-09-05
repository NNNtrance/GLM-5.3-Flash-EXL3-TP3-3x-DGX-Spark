#!/usr/bin/env python3
"""Model-free microbench of the GLM-5.3-Flash / DeepseekV4 hyper-connection (mHC) kernels.

Routes
  R3   production large-M path : mhc_post_tilelang + deep_gemm tf32_hc_prenorm + mhc_pre_big_fuse_with_norm
  R2   fused post+prenorm path : mhc_fused_tilelang (the <=16-token kernel, forced) + mhc_pre_big_fuse_with_norm
  RT   torch native fallback   : mhc_post_torch + mhc_pre_torch
Rulers measured in the same process.
"""
import json, os, statistics, sys, time

import torch

torch.cuda.set_device(0)
DEV = "cuda"
H = 4096
HC = 4
N_OUT = HC * 2 + HC * HC           # 24
LAYERS = int(os.environ.get("LAYERS", "90"))   # 45 decoder layers x 2 mhc calls
REPS = int(os.environ.get("REPS", "21"))
OUT = os.environ.get("OUT", "/cache/mhc-bench.json")

props = torch.cuda.get_device_properties(0)
print(f"device={props.name} sm={props.major}.{props.minor} SMs={props.multi_processor_count} "
      f"L2={getattr(props,'L2_cache_size',0)/2**20:.1f} MiB total={props.total_memory/2**30:.1f} GiB")

# ---------------------------------------------------------------- rulers
def rulers():
    r = {}
    n = int(4.0 * (1 << 30) // 2)
    a = torch.empty(n, dtype=torch.bfloat16, device=DEV).normal_()
    for _ in range(3):
        a.sum()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10):
        a.sum()
    e.record(); torch.cuda.synchronize()
    r["read_GBps_4GiB"] = round(a.numel() * 2 / (s.elapsed_time(e) / 10 * 1e-3) / 1e9, 1)
    del a; torch.cuda.empty_cache()

    n = int(2.0 * (1 << 30) // 2)
    a = torch.empty(n, dtype=torch.bfloat16, device=DEV).normal_()
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    s.record()
    for _ in range(10):
        b.copy_(a)
    e.record(); torch.cuda.synchronize()
    r["copy_GBps_2GiB"] = round(a.numel() * 4 / (s.elapsed_time(e) / 10 * 1e-3) / 1e9, 1)
    del a, b; torch.cuda.empty_cache()

    a = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    s.record()
    for _ in range(10):
        c = a @ b
    e.record(); torch.cuda.synchronize()
    r["bf16_TFLOPs_8192"] = round(2 * 8192 ** 3 / (s.elapsed_time(e) / 10 * 1e-3) / 1e12, 1)
    del a, b, c; torch.cuda.empty_cache()
    print("RULERS:", r)
    return r

RUL = rulers()

# ---------------------------------------------------------------- imports
from vllm.model_executor.kernels.mhc.tilelang_kernels import (   # noqa: E402
    compute_num_split, mhc_fused_tilelang, mhc_post_tilelang,
    mhc_pre_big_fuse_with_norm_tilelang,
)
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch  # noqa: E402
from vllm.utils.deep_gemm import is_deep_gemm_supported, tf32_hc_prenorm_gemm   # noqa: E402
from vllm.utils.math_utils import cdiv                                          # noqa: E402
import vllm.model_executor.layers.mhc as mhc_layer                              # noqa: E402

print("deep_gemm supported:", is_deep_gemm_supported(),
      "| HAS_TILELANG_MHC:", mhc_layer.HAS_TILELANG_MHC,
      "| HAS_AITER_MHC:", mhc_layer.HAS_AITER_MHC)

EPS = 1e-6
SINK = 3


class Bufs:
    def __init__(self, M):
        self.M = M
        g = torch.Generator(device=DEV).manual_seed(0)
        self.res = [torch.randn(M, HC, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
                    for _ in range(2)]
        self.x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
        self.post = torch.rand(M, HC, device=DEV, dtype=torch.float32, generator=g)
        self.comb = torch.rand(M, HC, HC, device=DEV, dtype=torch.float32, generator=g)
        self.fn = torch.randn(N_OUT, HC * H, device=DEV, dtype=torch.float32, generator=g) * 0.01
        self.fn3 = self.fn.view(N_OUT, HC, H)
        self.hc_scale = torch.ones(3, device=DEV, dtype=torch.float32)
        self.hc_base = torch.zeros(N_OUT, device=DEV, dtype=torch.float32)
        self.nw = torch.ones(H, device=DEV, dtype=torch.bfloat16)
        # deep_gemm n_splits, exactly as vllm/model_executor/kernels/mhc/tilelang.py:517-530
        self.ns = compute_num_split(64, HC * H, cdiv(M, 64)) if is_deep_gemm_supported() else 1
        self.gmul = torch.empty(self.ns, M, N_OUT, device=DEV, dtype=torch.float32)
        self.gsq = torch.empty(self.ns, M, device=DEV, dtype=torch.float32)
        self.pm_o = torch.empty(M, HC, device=DEV, dtype=torch.float32)
        self.cm_o = torch.empty(M, HC * HC, device=DEV, dtype=torch.float32)
        self.li = torch.empty(M, H, device=DEV, dtype=torch.bfloat16)


def k_post(b, i):
    mhc_post_tilelang(b.comb, b.res[i], b.post, b.x, b.res[1 - i], HC, H)

def k_gemm(b, i):
    tf32_hc_prenorm_gemm(b.res[1 - i].view(b.M, HC * H), b.fn, b.gmul, b.gsq, b.ns)

def k_pre(b, i, ns=None):
    mhc_pre_big_fuse_with_norm_tilelang(
        b.gmul, b.gsq, b.hc_scale, b.hc_base, b.res[1 - i], b.pm_o, b.cm_o, b.li,
        b.nw, H, EPS, EPS, EPS, 2.0, SINK, EPS, ns if ns is not None else b.ns, HC)

def make_fused(tile_n, split_k):
    def k_fused(b, i):
        mhc_fused_tilelang(b.comb, b.res[i], b.post, b.x, b.fn3,
                           b.gmul_f, b.gsq_f, b.res[1 - i], HC, H, N_OUT,
                           tile_n=tile_n, n_splits=split_k)
    return k_fused

def route_R3(b, i):
    k_post(b, i); k_gemm(b, i); k_pre(b, i)

def route_RT(b, i):
    out = mhc_post_torch(b.x, b.res[i], b.post.unsqueeze(-1), b.comb)
    mhc_pre_torch(out, b.fn, b.hc_scale, b.hc_base, EPS, EPS, EPS, 2.0, SINK)


def timeit(fn, b, layers=LAYERS, reps=REPS, graph=False):
    """median ms for `layers` sequential calls."""
    for i in range(3):
        fn(b, i & 1)
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for i in range(3):
                fn(b, i & 1)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for i in range(layers):
                fn(b, i & 1)
        torch.cuda.synchronize()
        run = g.replay
    else:
        def run():
            for i in range(layers):
                fn(b, i & 1)
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    ts = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        s.record(); run(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    if graph:
        del g
    return statistics.median(ts)


def count_launches(fn, b):
    from torch.profiler import ProfilerActivity, profile
    fn(b, 0); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(b, 0)
        torch.cuda.synchronize()
    names = {}
    for ev in p.events():
        if str(ev.device_type).endswith("CUDA") and ev.self_device_time_total > 0:
            names[ev.key] = names.get(ev.key, 0) + 1
    return names


# analytic traffic (bytes) per single call
def bytes_R3(M, ns):
    k1 = M * (16 * 4 + HC * H * 2 + HC * 4 + H * 2 + HC * H * 2)
    k2 = M * HC * H * 2 + N_OUT * HC * H * 4 + ns * M * (N_OUT + 1) * 4
    k3 = ns * M * (N_OUT + 1) * 4 + M * HC * H * 2 + H * 2 + M * (HC * 4 + HC * HC * 4 + H * 2)
    return k1, k2, k3, k1 + k2 + k3

def bytes_R2(M, ns):
    k1, k2, k3, _ = bytes_R3(M, ns)
    kf = M * (16 * 4 + HC * H * 2 + HC * 4 + H * 2 + HC * H * 2) + N_OUT * HC * H * 4
    return kf, k3, kf + k3


results = {"rulers": RUL, "device": props.name, "sms": props.multi_processor_count,
           "l2_MiB": round(getattr(props, "L2_cache_size", 0) / 2 ** 20, 1),
           "deep_gemm": bool(is_deep_gemm_supported()),
           "has_tilelang_mhc": bool(mhc_layer.HAS_TILELANG_MHC), "rows": []}

MS = [int(v) for v in os.environ.get("MS", "8,64,128,512,2048").split(",")]
for M in MS:
    print(f"\n================ M = {M} ================", flush=True)
    b = Bufs(M)
    b.gmul_f = b.gmul
    b.gsq_f = b.gsq
    k1B, k2B, k3B, R3B = bytes_R3(M, b.ns)
    print(f"n_splits(deep_gemm)={b.ns}  analytic traffic/call: k1 {k1B/1e6:.2f} MB  "
          f"k2 {k2B/1e6:.2f} MB  k3 {k3B/1e6:.2f} MB  total {R3B/1e6:.2f} MB")

    row = {"M": M, "n_splits": b.ns, "bytes_call_R3": R3B}
    # ---- per-kernel, eager
    for nm, fn, nb in (("k1_mhc_post", k_post, k1B),
                       ("k2_hc_prenorm_gemm", k_gemm, k2B),
                       ("k3_mhc_pre_big_fuse", k_pre, k3B)):
        ms = timeit(fn, b)
        per = ms / LAYERS * 1e3
        gb = nb / (per * 1e-6) / 1e9
        row[nm] = {"ms_per_%d" % LAYERS: round(ms, 3), "us_call": round(per, 1),
                   "GBps": round(gb, 1), "pct_read_ruler": round(gb / RUL["read_GBps_4GiB"] * 100, 1)}
        print(f"  {nm:<22} {ms:8.3f} ms /{LAYERS}  {per:8.1f} us/call  {gb:7.1f} GB/s  "
              f"{gb/RUL['read_GBps_4GiB']*100:5.1f}% of read ruler")

    ms3 = timeit(route_R3, b)
    ms3g = timeit(route_R3, b, graph=True)
    gb3 = R3B / (ms3 / LAYERS * 1e-3) / 1e9
    row["R3"] = {"ms": round(ms3, 3), "ms_graph": round(ms3g, 3), "GBps": round(gb3, 1),
                 "pct_read_ruler": round(gb3 / RUL["read_GBps_4GiB"] * 100, 1)}
    print(f"  R3 (production 3-kernel) {ms3:8.3f} ms /{LAYERS}   graph {ms3g:8.3f} ms   "
          f"{gb3:7.1f} GB/s = {gb3/RUL['read_GBps_4GiB']*100:.1f}% of read ruler")

    # ---- R2: fused post+prenorm kernel (the <=16-token kernel) forced at this M.
    # Compare against k1+k2 only; k3 is identical in both routes.
    k12 = row["k1_mhc_post"]["ms_per_%d" % LAYERS] + row["k2_hc_prenorm_gemm"]["ms_per_%d" % LAYERS]
    kfB, _k3, _ = bytes_R2(M, 1)
    best = None
    row["R2_fused_sweep"] = {}
    for tile_n in (1, 2, 3, 4, 6, 8, 12, 24):
        if N_OUT % tile_n:
            continue
        for split_k in (1, 2, 4, 8):
            if H % split_k:
                continue
            try:
                b.gmul_f = torch.empty(split_k, M, N_OUT, device=DEV, dtype=torch.float32)
                b.gsq_f = torch.empty(split_k, M, device=DEV, dtype=torch.float32)
                kf = make_fused(tile_n, split_k)
                mf = timeit(kf, b, reps=7)
                gb = kfB / (mf / LAYERS * 1e-3) / 1e9
                row["R2_fused_sweep"][f"t{tile_n}_s{split_k}"] = {
                    "fused_ms": round(mf, 3), "GBps": round(gb, 1),
                    "vs_k1k2_pct": round((mf - k12) / k12 * 100, 1)}
                print(f"  R2 fused tile_n={tile_n:>2} (n_tiles={N_OUT//tile_n:>2}) split_k={split_k}: "
                      f"{mf:9.3f} ms /{LAYERS}  {gb:7.1f} GB/s  vs k1+k2 {k12:.3f} ms "
                      f"({(mf-k12)/k12*100:+7.1f} %)", flush=True)
                if best is None or mf < best[0]:
                    best = (mf, tile_n, split_k)
            except Exception as ex:                                   # noqa: BLE001
                print(f"  R2 fused tile_n={tile_n} split_k={split_k}: FAIL {type(ex).__name__}: "
                      f"{str(ex)[:100]}", flush=True)
            finally:
                b.gmul_f, b.gsq_f = b.gmul, b.gsq
                torch.cuda.empty_cache()
    if best:
        row["R2_best"] = {"fused_ms": round(best[0], 3), "tile_n": best[1], "split_k": best[2],
                          "k1_plus_k2_ms": round(k12, 3),
                          "vs_k1k2_pct": round((best[0] - k12) / k12 * 100, 1),
                          "R3_ms": round(ms3, 3),
                          "route_total_ms": round(best[0] + row["k3_mhc_pre_big_fuse"]["ms_per_%d" % LAYERS], 3)}
        tot = best[0] + row["k3_mhc_pre_big_fuse"]["ms_per_%d" % LAYERS]
        print(f"  >> R2 best fused tile_n={best[1]} split_k={best[2]}: {best[0]:.3f} ms vs "
              f"k1+k2 {k12:.3f} ms ({(best[0]-k12)/k12*100:+.1f} %) | route total "
              f"{tot:.3f} vs R3 {ms3:.3f} ms ({(tot-ms3)/ms3*100:+.1f} %)", flush=True)

    # ---- torch native
    try:
        mst = timeit(route_RT, b, reps=5)
        row["RT_torch"] = {"ms": round(mst, 3), "vs_R3_x": round(mst / ms3, 2)}
        print(f"  RT (torch native)        {mst:8.3f} ms /{LAYERS}   = {mst/ms3:.2f} x R3")
    except Exception as ex:                                          # noqa: BLE001
        row["RT_torch"] = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        print(f"  RT (torch native) FAIL: {ex}")
    torch.cuda.empty_cache()

    # ---- launch counts
    try:
        row["launches_R3"] = count_launches(route_R3, b)
        print("  launches R3:", row["launches_R3"])
    except Exception as ex:                                          # noqa: BLE001
        print("  launch count failed:", ex)

    results["rows"].append(row)
    del b
    torch.cuda.empty_cache()

json.dump(results, open(OUT, "w"), indent=1)
print("\nwrote", OUT)

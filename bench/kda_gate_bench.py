"""Model-free gate bench for KDA/MLA bf16 -> EXL3 quantisation.

Measures, per shape family listed in docs/13-full-scope-checkpoint.md section 4.4:

  * bf16 F.linear (cuBLAS/cutlass, what production runs today)
  * EXL3 4-bit and 6-bit `ops.exl3_linear` (cuda-exl3 754421f, sm_120 build)

at M=8 (decode step with DFlash2 k=7) and M=1792 (prefill chunk), CUDA graph on,
median of >= 50 replays, with a weight bank large enough to defeat the 5090's
100 MB L2 (a single small trellis would otherwise sit resident and read as free).

Also measures the MLA strided-batched GEMM family in fp32 vs bf16 vs fp16 --
the "cheap independent lever" (production runs this family in fp32 today).

Absolute microseconds are 5090 numbers, NOT GB10 numbers. Only the
exl3/bf16 ratio per shape is carried to the node.
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from cuda_exl3 import ops

DEV = "cuda"
L2_BYTES = torch.cuda.get_device_properties(0).L2_cache_size
BANK_TARGET = 3 * L2_BYTES          # ~300 MB: three times L2, per arm
MAX_REPS = 384


def synth_trellis(k, n, bits):
    t = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                      dtype=torch.int16, device=DEV)
    return t


def bank_for(bytes_each, cap=MAX_REPS):
    return max(1, min(cap, int(BANK_TARGET // max(1, bytes_each)) + 1))


def time_graph(make_call, reps, iters=60, warmup=3):
    """Median per-call time (us) of `reps` back-to-back calls captured in one graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            for i in range(reps):
                make_call(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            make_call(i)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    del g
    return statistics.median(ts)


def time_eager(make_call, reps, iters=60, warmup=5):
    for _ in range(warmup):
        for i in range(reps):
            make_call(i)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for i in range(reps):
            make_call(i)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    return statistics.median(ts)


# ---------------------------------------------------------------- arms

def bf16_linear_us(m, k, n, iters, eager=False):
    wb = k * n * 2
    nb = bank_for(wb)
    ws = [torch.randn(n, k, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)
          for _ in range(nb)]
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)

    def call(i):
        return F.linear(x, ws[i % nb])

    fn = time_eager if eager else time_graph
    us = fn(call, nb, iters=iters)
    del ws, x
    torch.cuda.empty_cache()
    return us, nb


def exl3_linear_us(m, k, n, bits, iters, eager=False):
    tb = k * n * bits // 8
    nb = bank_for(tb)
    ts = [synth_trellis(k, n, bits) for _ in range(nb)]
    suh = (torch.randn(k, device=DEV) * 0.1).half().view(1, -1)
    svh = (torch.randn(n, device=DEV) * 0.1).half()
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)

    def call(i):
        return ops.exl3_linear(x, ts[i % nb], suh, svh, [n], 2)

    fn = time_eager if eager else time_graph
    us = fn(call, nb, iters=iters)
    del ts, suh, svh, x
    torch.cuda.empty_cache()
    return us, nb


def bmm_us(batch, m, k, n, dtype, iters, eager=False):
    """Strided-batched GEMM, the MLA kv_b form: [b,m,k] @ [b,k,n]."""
    wb = batch * k * n * dtype.itemsize
    nb = bank_for(wb, cap=64)
    ws = [torch.randn(batch, k, n, dtype=dtype, device=DEV) / (k ** 0.5)
          for _ in range(nb)]
    x = torch.randn(batch, m, k, dtype=dtype, device=DEV) / (k ** 0.5)

    def call(i):
        return torch.bmm(x, ws[i % nb])

    fn = time_eager if eager else time_graph
    us = fn(call, nb, iters=iters)
    del ws, x
    torch.cuda.empty_cache()
    return us, nb


# ---------------------------------------------------------------- shapes

# name, k, n, calls-per-step (target only), exl3-legal, note
SHAPES = [
    ("f_b_proj  TP3/rank", 128, 2816, 34, True, "8192->8448 pad, /3 = 22x128"),
    ("g_b_proj  TP3/rank", 128, 2816, 34, True, "identical shape to f_b"),
    ("f_b_proj  unsharded", 128, 8192, 0, True, "doc shape, replicated ref"),
    ("f_a_proj  replicated", 4096, 128, 0, True, "in_proj half, 128 wide"),
    ("g_a_proj  replicated", 4096, 128, 0, True, "identical to f_a"),
    ("in_proj_fg_a (f_a+g_a)", 4096, 256, 34, True, "EXL3 arm of the split"),
    ("in_proj_bfg_a TODAY", 4096, 278, 34, False, "256 fg_a + 22 b_proj, bf16"),
    ("in_proj_b  TP3/rank", 4096, 22, 34, False, "64->66 pad /3 = 22, NOT 128-aligned"),
    ("b_proj replicated", 4096, 128, 0, True, "64 -> 128 pad if replicated"),
    ("kv_b_proj replicated", 512, 32768, 11, True, "64 head x (256 nope + 256 v)"),
    ("kv_b_proj TP3 22head", 512, 11264, 11, True, "22/66 heads x 512, if sharded"),
]

MLA_BMM = [
    # name, batch, k, n, calls/step
    ("MLA w_uk  22 head/rank", 22, 256, 512, 11),
    ("MLA w_uv  22 head/rank", 22, 512, 256, 11),
    ("MLA w_uk  64 head repl", 64, 256, 512, 0),
    ("MLA w_uv  64 head repl", 64, 512, 256, 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", nargs="*", type=int, default=[8, 1792])
    ap.add_argument("--bits", nargs="*", type=int, default=[4, 6])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"# {p.name}  cc={p.major}.{p.minor}  SMs={p.multi_processor_count}  "
          f"L2={L2_BYTES/1e6:.0f} MB  torch={torch.__version__}")
    print(f"# backend={ops.backend()}  bank target={BANK_TARGET/1e6:.0f} MB  "
          f"iters={args.iters}  graph={'off' if args.eager else 'on'}")

    res = {"device": p.name, "cc": f"{p.major}.{p.minor}", "l2_mb": L2_BYTES / 1e6,
           "graph": not args.eager, "iters": args.iters, "linear": [], "bmm": []}

    for m in args.m:
        print(f"\n### M={m}   (bf16 F.linear vs EXL3 exl3_linear, us/call, "
              f"median of {args.iters})")
        hdr = f"{'shape':<24}{'k':>6}{'n':>7}{'bank':>6}{'bf16 us':>10}"
        for b in args.bits:
            hdr += f"{f'exl3-{b}b us':>12}{f'ratio {b}b':>10}"
        print(hdr)
        for name, k, n, calls, legal, note in SHAPES:
            b16, nb = bf16_linear_us(m, k, n, args.iters, args.eager)
            row = {"name": name, "k": k, "n": n, "m": m, "calls": calls,
                   "legal": legal, "note": note, "bf16_us": b16, "bank": nb}
            line = f"{name:<24}{k:>6}{n:>7}{nb:>6}{b16:>10.2f}"
            for bits in args.bits:
                if legal:
                    e, _ = exl3_linear_us(m, k, n, bits, args.iters, args.eager)
                    row[f"exl3_{bits}b_us"] = e
                    row[f"ratio_{bits}b"] = e / b16
                    line += f"{e:>12.2f}{e/b16:>10.3f}"
                else:
                    line += f"{'n/a':>12}{'-':>10}"
            print(line)
            res["linear"].append(row)

        print(f"\n### M={m}   MLA strided-batched (torch.bmm), fp32 vs bf16 vs fp16")
        print(f"{'shape':<24}{'b':>4}{'k':>6}{'n':>7}{'fp32 us':>10}{'bf16 us':>10}"
              f"{'fp16 us':>10}{'bf16/fp32':>11}")
        for name, batch, k, n, calls in MLA_BMM:
            f32, _ = bmm_us(batch, m, k, n, torch.float32, args.iters, args.eager)
            b16, _ = bmm_us(batch, m, k, n, torch.bfloat16, args.iters, args.eager)
            f16, _ = bmm_us(batch, m, k, n, torch.float16, args.iters, args.eager)
            print(f"{name:<24}{batch:>4}{k:>6}{n:>7}{f32:>10.2f}{b16:>10.2f}"
                  f"{f16:>10.2f}{b16/f32:>11.3f}")
            res["bmm"].append({"name": name, "batch": batch, "k": k, "n": n, "m": m,
                               "calls": calls, "fp32_us": f32, "bf16_us": b16,
                               "fp16_us": f16, "ratio_bf16_fp32": b16 / f32})

    print(f"\n# peak torch alloc: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    res["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()

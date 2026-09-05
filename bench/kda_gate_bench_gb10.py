"""WARM vs COLD re-check of the KDA/MLA bf16 -> EXL3 gate, on the target GPU (GB10).

Derived from bench/kda_gate_bench.py -- the workstation-GPU bench that closed
"decision 8" (docs/11 section 2.25). Same shape list, same synthetic trellis
construction, same `ops.exl3_linear` call path (= had(x*suh) @ dequant(trellis) * svh,
exactly what the engine runs at decode). Absolute microseconds here ARE GB10 numbers,
unlike that document, and the ratios replace its ratios: see
results/kernels/kda-gate-bench-gb10.md.

What is new here, and why
-------------------------
cuda-exl3 issue #5 (2026-09-05): timing the SAME weight over and over keeps a
12-33 MB weight resident in L2, which erases the very bandwidth advantage the
trellis exists for. The direction of that artefact depends on which arm fits the
cache, so it is not a constant offset -- on a 101-128 MiB L2 both arms fit and EXL3
reads slow; on GB10's 24 MiB only the trellis fits and EXL3 reads fast. Only cold is
honest on either card. So every shape is timed three ways, over one and the same
allocated weight bank:

  warm   : index 0 every call                      (weight lives in L2)
  coldA  : index j % N_bf16, N_bf16 = 4*L2/bf16    (the bf16 arm alone >= 4*L2; at
                                                    4 bit the trellis bank is then
                                                    only ~1*L2)
  coldB  : index j % N_full,  N_full = 4*L2/exl3_4b (BOTH arms >= 4*L2 -- the strict
                                                    reading; the primary number)

coldA and coldB agree to within 5 % everywhere, so the verdict is not sensitive to
the choice of N.

Timing: CUDA graph of G = max(200, N_full) back-to-back calls, 20 pre-capture
warm-up calls + 2 warm-up replays, then 3 timed replays measured with CUDA events,
median. Per-call us = elapsed / G. Because the graph rotates continuously, a given
weight is re-touched only after a full bank sweep, i.e. after >= 4*L2 bytes of other
traffic have streamed through.

bf16 arm is measured twice: F.linear (TN, the call vLLM actually runs and the one the
workstation table used -- this is the ratio denominator) and torch.matmul on a
contiguous (k,n) copy (NN), as a cuBLAS-path control.

Bits: the production checkpoint stores dense attention/shared-expert modules at
6 bit, the three dense-MLP layers at 5 bit, MoE experts at 4 bit
(quantization_config.json -> tensor_storage, trellis last dim = 16*bpw). All of
4/5/6 are measured.

Needs `cuda_exl3` importable and nothing else: no node, no checkpoint, no engine.
Run it inside the production image, beside an idle engine or with none:

  python3 bench/kda_gate_bench_gb10.py --m 1 8 64 --bits 4 5 6 --rounds 3
"""

import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

from cuda_exl3 import ops

DEV = "cuda"
PROPS = torch.cuda.get_device_properties(0)
L2_BYTES = PROPS.L2_cache_size
COLD_MULT = 4                      # bank must be >= COLD_MULT * L2
BANK_TARGET = COLD_MULT * L2_BYTES
MIN_TIMED_CALLS = 200
BANK_CAP_BYTES = 1200 * 2 ** 20    # never allocate more than 1.2 GiB of weights


# ---------------------------------------------------------------- helpers

def synth_trellis(k, n, bits):
    return torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                         dtype=torch.int16, device=DEV)


def n_for(bytes_each):
    return max(1, int(math.ceil(BANK_TARGET / max(1, bytes_each))))


def time_graph(make_call, G, rounds=3, warm_replays=2, eager=False):
    """Median us/call over `rounds` timed passes of G back-to-back calls."""
    if eager:
        for j in range(min(G, 20)):
            make_call(j)
        torch.cuda.synchronize()
        ts = []
        for _ in range(rounds):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for j in range(G):
                make_call(j)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / G)
        return statistics.median(ts), ts

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for j in range(min(G, 20)):
            make_call(j)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for j in range(G):
            make_call(j)
    torch.cuda.synchronize()
    for _ in range(warm_replays):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / G)
    del g
    torch.cuda.synchronize()
    return statistics.median(ts), ts


def variants(n_bf16, n_full):
    return [("warm", 1), ("coldA", n_bf16), ("coldB", n_full)]


# ---------------------------------------------------------------- shapes
# name, k, n, calls-per-step (target, production-9 profile), exl3-legal, note
SHAPES = [
    ("f_b_proj  TP3/rank",     128,  2816, 34, True,  "8192->8448 pad, /3 = 22x128"),
    ("g_b_proj  TP3/rank",     128,  2816, 34, True,  "identical shape to f_b"),
    ("f_b_proj  unsharded",    128,  8192,  0, True,  "doc shape, replicated ref"),
    ("f_a_proj  replicated",  4096,   128,  0, True,  "in_proj half, 128 wide"),
    ("g_a_proj  replicated",  4096,   128,  0, True,  "identical to f_a"),
    ("in_proj_fg_a (fa+ga)",  4096,   256, 34, True,  "EXL3 arm of the split"),
    ("in_proj_bfg_a TODAY",   4096,   278, 34, False, "256 fg_a + 22 b_proj, bf16"),
    ("in_proj_b  TP3/rank",   4096,    22, 34, False, "64->66 pad /3 = 22, not 128-aligned"),
    ("b_proj replicated",     4096,   128,  0, True,  "64 -> 128 pad if replicated"),
    ("kv_b_proj replicated",   512, 32768, 11, True,  "kernel would be needed (per-head batched)"),
    ("kv_b_proj TP3 22head",   512, 11264, 11, True,  "kernel would be needed (per-head batched)"),
    ("CTRL 4096x4096",        4096,  4096,  0, True,  "cuda-exl3 issue #5 control shape"),
    ("CTRL 4096x11008",       4096, 11008,  0, True,  "cuda-exl3 issue #5 control shape"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", nargs="*", type=int, default=[1, 8, 64])
    ap.add_argument("--bits", nargs="*", type=int, default=[4, 5, 6])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="coldbench.json")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on shape name")
    ap.add_argument("--no-matmul", action="store_true")
    args = ap.parse_args()

    print(f"# {PROPS.name}  cc={PROPS.major}.{PROPS.minor}  "
          f"SMs={PROPS.multi_processor_count}  L2={L2_BYTES/2**20:.1f} MiB "
          f"({L2_BYTES} B)  torch={torch.__version__}")
    print(f"# backend={ops.backend()}  cold target={COLD_MULT}xL2="
          f"{BANK_TARGET/2**20:.0f} MiB  rounds={args.rounds}  "
          f"graph={'off' if args.eager else 'on'}  bits={args.bits}")
    for v in ("CUDA_EXL3_SPLIT_TARGET", "CUDA_EXL3_SPLIT_BUDGET",
              "CUDA_EXL3_TUNE_CACHE", "CUDA_EXL3_DEBUG_NAMES"):
        print(f"#   env {v}={os.environ.get(v, '<unset>')}")

    res = {"device": PROPS.name, "cc": f"{PROPS.major}.{PROPS.minor}",
           "sms": PROPS.multi_processor_count, "l2_bytes": L2_BYTES,
           "cold_mult": COLD_MULT, "torch": torch.__version__,
           "backend": ops.backend(), "graph": not args.eager,
           "rounds": args.rounds, "rows": []}

    for name, k, n, calls, legal, note in SHAPES:
        if args.only and args.only not in name:
            continue
        bf_bytes = k * n * 2
        e4_bytes = k * n * min(args.bits) // 8
        n_bf16 = n_for(bf_bytes)
        n_full = n_for(e4_bytes)
        # keep the biggest bank (bf16 at n_full) under the cap
        if n_full * bf_bytes > BANK_CAP_BYTES:
            n_full = max(n_bf16, BANK_CAP_BYTES // bf_bytes)
        G = max(MIN_TIMED_CALLS, n_full)
        print(f"\n===== {name}   k={k} n={n}  |  N_bf16={n_bf16} N_full={n_full} "
              f"G={G}  |  bf16 bank {n_full*bf_bytes/2**20:.0f} MiB "
              f"({n_full*bf_bytes/L2_BYTES:.1f}xL2), "
              f"4b bank {n_full*k*n//2/2**20:.0f} MiB "
              f"({n_full*k*n/2/L2_BYTES:.1f}xL2)   [{note}]")

        # ---------------- bf16 arm (F.linear, production form: W is (n,k))
        ws = [torch.randn(n, k, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)
              for _ in range(n_full)]
        bf = {}
        for m in args.m:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)
            for vname, nn in variants(n_bf16, n_full):
                us, raw = time_graph(lambda j: F.linear(x, ws[j % nn]), G,
                                     args.rounds, eager=args.eager)
                bf[(m, vname)] = us
                by = bf_bytes + m * k * 2 + m * n * 2
                print(f"  bf16 F.linear   M={m:<4} {vname:<6} {us:9.3f} us   "
                      f"{by/us/1e3:7.1f} GB/s   raw={['%.3f' % r for r in raw]}")
                res["rows"].append(dict(shape=name, k=k, n=n, m=m, arm="bf16_linear",
                                        bits=16, variant=vname, n_rot=nn, G=G,
                                        us=us, raw=raw, bytes=by,
                                        gbs=by / us / 1e3, calls=calls, note=note))
            del x
        del ws
        torch.cuda.empty_cache()

        # ---------------- bf16 control (torch.matmul, NN, W is (k,n))
        if not args.no_matmul:
            ws = [torch.randn(k, n, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)
                  for _ in range(n_full)]
            for m in args.m:
                x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)
                for vname, nn in variants(n_bf16, n_full):
                    us, raw = time_graph(lambda j: torch.matmul(x, ws[j % nn]), G,
                                         args.rounds, eager=args.eager)
                    by = bf_bytes + m * k * 2 + m * n * 2
                    print(f"  bf16 matmul     M={m:<4} {vname:<6} {us:9.3f} us   "
                          f"{by/us/1e3:7.1f} GB/s")
                    res["rows"].append(dict(shape=name, k=k, n=n, m=m,
                                            arm="bf16_matmul", bits=16,
                                            variant=vname, n_rot=nn, G=G, us=us,
                                            raw=raw, bytes=by, gbs=by / us / 1e3,
                                            calls=calls, note=note))
                del x
            del ws
            torch.cuda.empty_cache()

        # ---------------- EXL3 arms
        if not legal:
            print(f"  exl3            n={n} not a multiple of 16 -> no trellis, "
                  f"bf16 only")
            continue
        for bits in args.bits:
            ts = [synth_trellis(k, n, bits) for _ in range(n_full)]
            suh = (torch.randn(k, device=DEV) * 0.1).half().view(1, -1)
            svh = (torch.randn(n, device=DEV) * 0.1).half()
            tb = k * n * bits // 8
            for m in args.m:
                x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)
                for vname, nn in variants(n_bf16, n_full):
                    us, raw = time_graph(
                        lambda j: ops.exl3_linear(x, ts[j % nn], suh, svh, [n], 2),
                        G, args.rounds, eager=args.eager)
                    by = tb + k * 2 + n * 2 + m * k * 2 + m * n * 2
                    r = us / bf[(m, vname)]
                    print(f"  exl3-{bits}b        M={m:<4} {vname:<6} {us:9.3f} us   "
                          f"{by/us/1e3:7.1f} GB/s   ratio={r:6.3f}"
                          f"{'  <-- trellis wins' if r < 1.0 else ''}")
                    res["rows"].append(dict(shape=name, k=k, n=n, m=m,
                                            arm="exl3", bits=bits, variant=vname,
                                            n_rot=nn, G=G, us=us, raw=raw, bytes=by,
                                            gbs=by / us / 1e3, ratio=r,
                                            calls=calls, note=note))
                del x
            del ts, suh, svh
            torch.cuda.empty_cache()

        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)

    res["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"\n# peak torch alloc: {res['peak_alloc_gib']:.2f} GiB")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()

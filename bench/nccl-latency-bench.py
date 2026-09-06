#!/usr/bin/env python3
"""nccl-latency-bench.py -- model-free NCCL all-reduce sweep, nccl-tests-equivalent.

Mirrors NVIDIA nccl-tests' all_reduce_perf methodology exactly (same size
sweep convention -b/-e/-f, same warmup/iters, same algBW/busBW formulas, same
-c correctness check) but runs as a torch.distributed/NCCL payload inside the
existing HAREM mesh launch path (run-nccl-latency.sh -> this script), because
no MPI toolchain exists on these nodes and nccl-tests itself requires MPI for
multi-node. One process per node. RANK/WORLD_SIZE/MASTER_ADDR from env; NCCL
env is whatever the caller (docker -e) set -- this script does not touch NCCL
env itself.

Env:
  LAT_TAG      label for this run / the output json (default "run")
  LAT_SIZES    comma-separated byte sizes (default: 8..64Mi by *2, i.e. -b8 -e64M -f2)
  LAT_WARMUP   default 10  (nccl-tests -w)
  LAT_ITERS    default 50  (nccl-tests -n)
  LAT_OUT      output json path (default /cache/lat-<tag>-r<rank>.json)
"""
import os, sys, json, time

import torch
import torch.distributed as dist


def sizes_pow2(lo, hi):
    out = []
    x = lo
    while x <= hi:
        out.append(x)
        x *= 2
    return out


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    tag = os.environ.get("LAT_TAG", "run")
    warmup = int(os.environ.get("LAT_WARMUP", "10"))
    iters = int(os.environ.get("LAT_ITERS", "50"))
    if os.environ.get("LAT_SIZES"):
        sizes = [int(x) for x in os.environ["LAT_SIZES"].split(",")]
    else:
        sizes = sizes_pow2(8, 64 * 1024 * 1024)  # nccl-tests -b 8 -e 64M -f 2

    busfac = 2.0 * (world - 1) / world  # ring all-reduce, same formula nccl-tests uses

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"## nccl-latency-bench tag={tag} world={world} warmup={warmup} iters={iters} "
        f"algo={os.environ.get('NCCL_ALGO','-')} proto={os.environ.get('NCCL_PROTO','-')} "
        f"nch={os.environ.get('NCCL_MIN_NCHANNELS','-')}/{os.environ.get('NCCL_MAX_NCHANNELS','-')}")
    log(f"{'bytes':>10} {'time_us':>10} {'algbw_GBps':>11} {'busbw_GBps':>11}  correctness")

    rows = []
    expected_val = float(sum(range(1, world + 1)))  # 1+2+...+world, exact in bf16
    for nbytes in sizes:
        nelem = max(1, nbytes // 2)  # bf16 = 2 bytes/elem

        # -c 1 equivalent: one correctness pass with a known-exact pattern
        chk = torch.full((nelem,), float(rank + 1), device="cuda", dtype=torch.bfloat16)
        dist.all_reduce(chk)
        correct = bool(torch.all(chk == expected_val).item())

        # timed pass: values are irrelevant to bandwidth, use a fresh random buffer
        t = torch.randn((nelem,), device="cuda", dtype=torch.bfloat16)
        for _ in range(warmup):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            dist.all_reduce(t)
        end.record()
        torch.cuda.synchronize()

        us = start.elapsed_time(end) / iters * 1000.0
        algbw = (nbytes / (us * 1e-6)) / 1e9  # GB/s (decimal, matches nccl-tests)
        busbw = algbw * busfac

        rows.append({"bytes": nbytes, "time_us": us, "algbw_gbps": algbw,
                      "busbw_gbps": busbw, "correct": correct})
        log(f"{nbytes:>10} {us:>10.3f} {algbw:>11.3f} {busbw:>11.3f}  "
            f"{'OK' if correct else 'FAIL'}")

    if rank == 0:
        out_path = os.environ.get("LAT_OUT", f"/cache/lat-{tag}-r{rank}.json")
        cfg = {k: os.environ.get(k, "-") for k in (
            "NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS",
            "NCCL_MESH_LINKS_PER_PEER", "NCCL_MESH_MIN_RNR_TIMER",
            "NCCL_MESH_PTR_CUDA", "NCCL_MESH_FLUSH")}
        with open(out_path, "w") as f:
            json.dump({"tag": tag, "world": world, "warmup": warmup, "iters": iters,
                        "config": cfg, "rows": rows}, f, indent=1)
        log(f"wrote {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

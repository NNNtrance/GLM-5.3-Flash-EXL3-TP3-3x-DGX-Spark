#!/usr/bin/env python3
"""all-reduce latency over the HAREM mesh, at the message sizes the engine uses.

One process per node, RANK/WORLD_SIZE/MASTER_ADDR from env, NCCL exactly as the
engine has it (the launcher's NCCL_* are inherited). Sizes are hidden=4096 x
bf16 x ntokens, i.e. exactly one TP all-reduce of an attention or MoE output.
"""
import os, sys, json, time
import torch, torch.distributed as dist

H = 4096
TOKENS = [1, 8, 16, 32, 64, 128, 512, 2048, 8192]


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    res = []
    if rank == 0:
        print(f"world={world} algo={os.environ.get('NCCL_ALGO','<unset>')} "
              f"proto={os.environ.get('NCCL_PROTO','<unset>')} net={os.environ.get('NCCL_NET','?')}")
        print(f"{'tokens':>7} {'bytes':>10} {'us':>9} {'GB/s(bus)':>10}")
    for nt in TOKENS:
        t = torch.randn((nt, H), device="cuda", dtype=torch.bfloat16)
        for _ in range(20): dist.all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()
        n = 200 if nt <= 128 else 50
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(n): dist.all_reduce(t)
        e.record(); torch.cuda.synchronize()
        us = s.elapsed_time(e) / n * 1000.0
        nbytes = t.numel() * 2
        busbw = nbytes * 2 * (world - 1) / world / (us * 1e-6) / 1e9
        res.append(dict(tokens=nt, bytes=nbytes, us=round(us, 2), busbw=round(busbw, 2)))
        if rank == 0:
            print(f"{nt:>7} {nbytes:>10} {us:>9.2f} {busbw:>10.2f}")
        del t
    # a decode step's worth: 90 all-reduces of 8 tokens back to back
    t = torch.randn((8, H), device="cuda", dtype=torch.bfloat16)
    for _ in range(10):
        for _ in range(90): dist.all_reduce(t)
    torch.cuda.synchronize(); dist.barrier()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(20):
        for _ in range(90): dist.all_reduce(t)
    e.record(); torch.cuda.synchronize()
    step_ms = s.elapsed_time(e) / 20
    if rank == 0:
        print(f"90 x all_reduce(8x4096 bf16) = {step_ms:.2f} ms  (one decode step's collectives)")
        out = os.environ.get("AR_OUT")
        if out:
            json.dump(dict(world=world, algo=os.environ.get("NCCL_ALGO"),
                           proto=os.environ.get("NCCL_PROTO"), sizes=res,
                           step90_ms=round(step_ms, 3)), open(out, "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

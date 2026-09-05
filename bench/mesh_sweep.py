#!/usr/bin/env python3
"""Fine-grained NCCL micro-benchmark over the HAREM mesh, with RDMA hardware counters.

One process per node.  RANK/WORLD_SIZE/MASTER_ADDR from env; NCCL env is whatever
the caller set.  Sizes step by ~1.5x from 8 KB to 64 MB, expressed as tokens of
hidden=4096 bf16 (8192 B/token) so every row is a real TP all-reduce shape.

For every size we also read the ConnectX-7 hardware counters before and after the
timed loop and report the per-collective delta:
    rnr  = rnr_nak_retry_err   (requester: my send hit a receiver with no posted WR)
    oob  = out_of_buffer       (responder: I had no posted WR when a send arrived)
    pse  = packet_seq_err      (requester: out-of-order / dropped packet)
    lto  = local_ack_timeout_err
Both are cumulative since driver load, so only deltas mean anything.

MESH_OPS selects the operations: allreduce, alltoall, sendrecv (comma separated).
"""
import os, sys, json, time, glob
import torch, torch.distributed as dist

H = 4096
BPT = H * 2                       # bytes per token, bf16
CTRS = ("rnr_nak_retry_err", "out_of_buffer", "packet_seq_err",
        "local_ack_timeout_err", "req_transport_retries_exceeded")


def counter_paths():
    out = {}
    for d in sorted(glob.glob("/sys/class/infiniband/*/ports/1/hw_counters")):
        dev = d.split("/")[4]
        out[dev] = d
    return out


CPATHS = counter_paths()


def read_counters():
    tot = {c: 0 for c in CTRS}
    for dev, d in CPATHS.items():
        for c in CTRS:
            try:
                with open(os.path.join(d, c)) as f:
                    tot[c] += int(f.read().strip())
            except Exception:
                pass
    return tot


def sizes_15x(lo_tokens=1, hi_tokens=8192):
    """1, then *1.5 rounded up, deduped, capped."""
    out, x = [], float(lo_tokens)
    while x <= hi_tokens + 0.5:
        n = int(round(x))
        if not out or n > out[-1]:
            out.append(n)
        x *= 1.5
    if out[-1] != hi_tokens:
        out.append(hi_tokens)
    return out


def timed(fn, nt, warm, iters):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
    c0 = read_counters()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    c1 = read_counters()
    us = s.elapsed_time(e) / iters * 1000.0
    d = {k: (c1[k] - c0[k]) / iters for k in CTRS}
    return us, d


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    ops = os.environ.get("MESH_OPS", "allreduce").split(",")
    tag = os.environ.get("MESH_TAG", "run")
    if os.environ.get("MESH_TOKENS"):
        toks = [int(x) for x in os.environ["MESH_TOKENS"].split(",")]
    else:
        toks = sizes_15x(1, int(os.environ.get("MESH_MAX_TOKENS", "8192")))
    results = {}

    def hdr(op):
        if rank == 0:
            print(f"\n## {op}  world={world} tag={tag} "
                  f"algo={os.environ.get('NCCL_ALGO','-')} proto={os.environ.get('NCCL_PROTO','-')} "
                  f"buff={os.environ.get('NCCL_BUFFSIZE','-')} chunk={os.environ.get('NCCL_P2P_NET_CHUNKSIZE','-')} "
                  f"nch={os.environ.get('NCCL_MIN_NCHANNELS','-')}/{os.environ.get('NCCL_MAX_NCHANNELS','-')} "
                  f"perpeer={os.environ.get('NCCL_NCHANNELS_PER_NET_PEER','-')}")
            print(f"{'tok':>6} {'bytes':>10} {'us':>10} {'GB/s':>7} "
                  f"{'rnr/op':>9} {'oob/op':>9} {'pse/op':>8} {'lto/op':>8}")

    for op in ops:
        hdr(op)
        rows = []
        for nt in toks:
            nbytes = nt * BPT
            iters = 200 if nt <= 128 else (50 if nt <= 2048 else 20)
            warm = 20 if nt <= 128 else 10
            if op == "allreduce":
                t = torch.randn((nt, H), device="cuda", dtype=torch.bfloat16)
                fn = lambda: dist.all_reduce(t)
                busfac = 2 * (world - 1) / world
            elif op == "alltoall":
                if nt % world:
                    nt2 = nt + (world - nt % world)
                else:
                    nt2 = nt
                t = torch.randn((nt2, H), device="cuda", dtype=torch.bfloat16)
                o = torch.empty_like(t)
                fn = lambda: dist.all_to_all_single(o, t)
                nbytes = nt2 * BPT
                busfac = (world - 1) / world
            elif op == "sendrecv":
                t = torch.randn((nt, H), device="cuda", dtype=torch.bfloat16)
                r = torch.empty_like(t)
                nxt = (rank + 1) % world
                prv = (rank - 1) % world

                def fn(t=t, r=r, nxt=nxt, prv=prv):
                    ops_ = [dist.P2POp(dist.isend, t, nxt), dist.P2POp(dist.irecv, r, prv)]
                    for w in dist.batch_isend_irecv(ops_):
                        w.wait()
                busfac = 1.0
            else:
                continue
            us, d = timed(fn, nt, warm, iters)
            gbs = nbytes * busfac / (us * 1e-6) / 1e9
            rows.append(dict(tokens=nt, bytes=nbytes, us=round(us, 2), gbs=round(gbs, 2),
                             **{k: round(v, 2) for k, v in d.items()}))
            if rank == 0:
                print(f"{nt:>6} {nbytes:>10} {us:>10.2f} {gbs:>7.2f} "
                      f"{d['rnr_nak_retry_err']:>9.1f} {d['out_of_buffer']:>9.1f} "
                      f"{d['packet_seq_err']:>8.1f} {d['local_ack_timeout_err']:>8.1f}")
            del t
            if op != "allreduce":
                try: del o
                except Exception: pass
            torch.cuda.empty_cache()
        results[op] = rows

    # One decode step's worth of collectives: 90 back-to-back all-reduces.
    if os.environ.get("MESH_STEP90", "1") == "1":
        nt = int(os.environ.get("MESH_STEP90_TOKENS", "8"))
        t = torch.randn((nt, H), device="cuda", dtype=torch.bfloat16)
        for _ in range(10):
            for _ in range(90):
                dist.all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()
        s_ = torch.cuda.Event(True); e_ = torch.cuda.Event(True); s_.record()
        for _ in range(20):
            for _ in range(90):
                dist.all_reduce(t)
        e_.record(); torch.cuda.synchronize()
        step_ms = s_.elapsed_time(e_) / 20
        results["step90"] = [dict(tokens=nt, ms=round(step_ms, 3))]
        if rank == 0:
            print(f"STEP90 {nt} {step_ms:.3f} ms")
        del t

    if rank == 0:
        out = os.environ.get("MESH_OUT")
        if out:
            json.dump(dict(world=world, tag=tag,
                           env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                           results=results), open(out, "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

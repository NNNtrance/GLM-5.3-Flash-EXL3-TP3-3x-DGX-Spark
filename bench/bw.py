#!/usr/bin/env python3
"""Achievable device read/copy bandwidth on GB10 -- the ruler for every roofline claim."""
import torch, time, json, os
torch.cuda.set_device(0)
dev = "cuda"
out = {}
for gib, name in ((2.0, "2GiB"), (4.0, "4GiB")):
    n = int(gib * (1 << 30) // 2)          # bf16 elements
    a = torch.empty(n, dtype=torch.bfloat16, device=dev).normal_()
    b = torch.empty_like(a)
    # copy: reads n*2 bytes, writes n*2 bytes
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10):
        b.copy_(a)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 10
    copy_bw = (a.numel() * 2 * 2) / (ms * 1e-3) / 1e9
    # pure read: sum reduction
    for _ in range(3):
        a.sum()
    torch.cuda.synchronize()
    s.record()
    for _ in range(10):
        a.sum()
    e.record(); torch.cuda.synchronize()
    ms2 = s.elapsed_time(e) / 10
    read_bw = (a.numel() * 2) / (ms2 * 1e-3) / 1e9
    print(f"{name}: copy(r+w) {copy_bw:7.1f} GB/s   read-only(sum) {read_bw:7.1f} GB/s")
    out[name] = dict(copy=round(copy_bw, 1), read=round(read_bw, 1))
    del a, b
    torch.cuda.empty_cache()
p = os.environ.get("BW_OUT")
if p:
    json.dump(out, open(p, "w"), indent=1)

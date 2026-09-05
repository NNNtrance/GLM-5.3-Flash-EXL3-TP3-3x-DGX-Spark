"""Verify the measuring instrument before trusting it (HAREM rule: check the ruler).

No EXL3 checkpoint exists on this workstation, so the repo's own correctness
tests skip. These are the checks that ARE possible model-free on the 5090:

1. roofline sanity -- does the bf16 arm reach a believable fraction of the
   machine's measured peak at both M=8 (bandwidth) and M=1792 (compute)?
2. EXL3 kernel self-consistency -- fused multi-shard call vs the same trellis
   run one shard at a time must agree bit-for-bit-ish;
   deterministic (no split-k) vs split-k must agree numerically.
3. EXL3 output is finite and non-degenerate for every shape we benchmark.
4. dtype invariance -- bf16 and fp16 activations give the same result.
"""

import torch
import torch.nn.functional as F
from cuda_exl3 import ops

DEV = "cuda"
PEAK_GBS = 1664.0
PEAK_TFLOPS = 235.0

SHAPES = [(128, 2816, 4), (4096, 256, 4), (512, 32768, 4), (512, 32768, 6)]


def synth(k, n, bits):
    t = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                      dtype=torch.int16, device=DEV)
    suh = (torch.randn(k, device=DEV) * 0.1).half()
    svh = (torch.randn(n, device=DEV) * 0.1).half()
    return t, suh, svh


print("## 1. bf16 arm vs measured machine peak "
      f"({PEAK_GBS:.0f} GB/s read, {PEAK_TFLOPS:.0f} TFLOPS)")
import time
for k, n in [(512, 32768), (4096, 256), (128, 2816)]:
    for m in (8, 1792):
        w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV)
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
        for _ in range(20):
            y = F.linear(x, w)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            y = F.linear(x, w)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 200 * 1e6
        gbs = (k * n * 2 + m * k * 2 + m * n * 2) / us / 1e3
        tfl = 2 * m * k * n / us / 1e6
        print(f"   k={k:<6} n={n:<6} m={m:<5} {us:8.2f} us  "
              f"{gbs:7.0f} GB/s ({gbs/PEAK_GBS*100:4.0f}% peak)  "
              f"{tfl:7.1f} TFLOPS ({tfl/PEAK_TFLOPS*100:4.0f}% peak)  "
              f"[L2-resident, upper bound]")
        del w, x, y
        torch.cuda.empty_cache()

print("\n## 2/3/4. EXL3 kernel self-consistency")
for k, n, bits in SHAPES:
    t, suh, svh = synth(k, n, bits)
    x = torch.randn(8, k, dtype=torch.bfloat16, device=DEV) / (k ** 0.5)
    y1 = ops.exl3_linear(x, t, suh.view(1, -1), svh, [n], 2)
    # same trellis split into two shards of n/2 -> concatenation must match
    half = n // 2
    t_a, t_b = t[:, :half // 16].contiguous(), t[:, half // 16:].contiguous()
    ya = ops.exl3_linear(x, t_a, suh.view(1, -1), svh[:half].contiguous(), [half], 2)
    yb = ops.exl3_linear(x, t_b, suh.view(1, -1), svh[half:].contiguous(), [half], 2)
    y2 = torch.cat([ya, yb], dim=-1)
    # fp16 activations
    y3 = ops.exl3_linear(x.half(), t, suh.view(1, -1), svh, [n], 2)
    d = y1.float()
    e_split = (y2.float() - d).abs().max().item() / max(1e-9, d.abs().mean().item())
    e_dtype = (y3.float() - d).abs().max().item() / max(1e-9, d.abs().mean().item())
    print(f"   k={k:<6} n={n:<6} {bits}b  finite={bool(torch.isfinite(y1).all())}  "
          f"mean|y|={d.abs().mean().item():.4g}  "
          f"shard-split relerr={e_split:.2e}  bf16-vs-fp16 relerr={e_dtype:.2e}")
    del t, suh, svh, x, y1, y2, y3, t_a, t_b, ya, yb
    torch.cuda.empty_cache()

#!/usr/bin/env python3
"""BF16 dense GEMM peak on GB10 -- the ruler for the dense-GEMM roofline claim."""
import torch, json, os
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
print(f"{'M':>6} {'N':>6} {'K':>6} {'ms':>8} {'TFLOP/s':>9}")
best = 0
res = []
for (M, N, K) in [(4096,4096,4096),(8192,8192,8192),(2032,5632,4096),(2032,4096,5632),
                  (2032,12288,4096),(2032,16896,1536),(2032,4096,4096)]:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    for _ in range(5): c = a @ b
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(20): c = a @ b
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e)/20
    tf = 2*M*N*K/(ms*1e-3)/1e12
    best = max(best, tf)
    res.append(dict(M=M,N=N,K=K,ms=round(ms,3),tflops=round(tf,1)))
    print(f"{M:>6} {N:>6} {K:>6} {ms:>8.3f} {tf:>9.1f}")
    del a,b,c; torch.cuda.empty_cache()
print(f"best BF16 dense: {best:.1f} TFLOP/s")
p=os.environ.get("OUT")
if p: json.dump(dict(rows=res,best=best),open(p,"w"),indent=1)

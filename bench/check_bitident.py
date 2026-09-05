"""Does a8270f9's OWN step (fusing the input transform) change the numbers?

Upstream claims the input-transform fusion is bit-identical "to the separate
transform". That is a claim about a8270f9 vs 5814c7f, not about a8270f9 vs
bc0e0f6 -- 5814c7f's atomic combine reorders the sum and cannot be bit-identical.
So compare the right pair: mid (4 kernels, combine fused) vs new (3 kernels).
"""
import sys
sys.argv = [sys.argv[0], "--label", "x"]
import importlib.util
spec = importlib.util.spec_from_file_location("b", "/bench/bench_moe_fusion2.py")
b = importlib.util.module_from_spec(spec)
b.__name__ = "b"
import types, torch
src = open("/bench/bench_moe_fusion2.py").read().replace("\nmain()\n", "\n")
exec(compile(src, "bench", "exec"), b.__dict__)

W = b.make_weights()
print("M      mid-vs-new bitwise   rel_l2      max_abs     | old-vs-mid bitwise  rel_l2")
for M in (8, 64, 512, 2048):
    R = b.make_routing(M, 1234)
    o_old = b.make_old(W, R)()
    o_mid = b.make_mid(W, R)()
    o_new = b.make_new(W, R)()
    torch.cuda.synchronize()
    fm, fn_, fo = o_mid.float(), o_new.float(), o_old.float()
    d1 = (fm - fn_).abs()
    d2 = (fo - fm).abs()
    print(f"{M:>5}  {'YES' if torch.equal(o_mid, o_new) else 'no ':>18}  "
          f"{(d1.norm() / fm.norm()).item():.3e}  {d1.max().item():.3e}  |  "
          f"{'YES' if torch.equal(o_old, o_mid) else 'no ':>17}  "
          f"{(d2.norm() / fo.norm()).item():.3e}")
    del R, o_old, o_mid, o_new, fm, fn_, fo, d1, d2
    torch.cuda.empty_cache()

#!/usr/bin/env python3
"""Are the SURPLUS blocks (beyond num_tokens_post_pad) the cost?

moe_align_block_size sizes expert_ids for the worst case and only fills the live
prefix; the tail keeps whatever it was initialised with. vLLM then maps the whole
array through expert_map, so the tail becomes expert_map[<that value>] -- which is
-1 for most ranks but a REAL local id for the rank that owns the highest global
expert. Compare the kernel with that tail left as-is vs forced to -1.
"""
import torch, sys
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from cuda_exl3 import _C  # noqa
ops=torch.ops.cuda_exl3_C
DEV="cuda"; H=4096; TOPK=8; E_GLOBAL=288; BITS=4; CB=1; TILE=16; E_LOCAL=96; I=2048
def weights(E,I,seed=7):
    g=torch.Generator(device=DEV).manual_seed(seed)
    return dict(
      w13_tr=torch.randint(-32768,32767,(E,H//TILE,2*I//TILE,TILE*BITS),dtype=torch.int16,device=DEV,generator=g),
      w2_tr=torch.randint(-32768,32767,(E,I//TILE,H//TILE,TILE*BITS),dtype=torch.int16,device=DEV,generator=g),
      w13_suh=(torch.randn((E,2,H),device=DEV,generator=g)*.05).half(),
      w2_suh=(torch.randn((E,1,I),device=DEV,generator=g)*.05).half(),
      w13_svh=(torch.randn((E,2*I),device=DEV,generator=g)*.05).half(),
      w2_svh=(torch.randn((E,H),device=DEV,generator=g)*.05).half())
def t(fn,it=20,wm=5):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it*1000.0
M=int(sys.argv[1]) if len(sys.argv)>1 else 2048
bm=64 if M>=2048 else 16
W=weights(E_LOCAL,I)
g=torch.Generator(device="cpu").manual_seed(9000+M)
ids=torch.stack([torch.randperm(E_GLOBAL,generator=g)[:TOPK] for _ in range(M)]).to(DEV,torch.int32)
tw=torch.rand((M,TOPK),device=DEV); tw=tw/tw.sum(1,keepdim=True)
x=(torch.randn((M,H),device=DEV)*.02).to(torch.bfloat16).contiguous()
print(f"M={M} bm={bm}")
print(f"{'owner range':>16} {'tail':>10} {'tail_e':>7} {'live':>6} {'rows':>7} | {'g13 us':>9} {'g2 us':>9}")
for lo,hi in ((0,96),(192,288)):
    emap=torch.full((E_GLOBAL,),-1,dtype=torch.int32,device=DEV)
    emap[lo:hi]=torch.arange(hi-lo,dtype=torch.int32,device=DEV)
    s,e,n=moe_align_block_size(ids,bm,E_GLOBAL,expert_map=emap,pad_sorted_ids=True)
    s=s.int(); e=e.int(); n=n.int()
    rows=min(e.numel()*bm,s.numel())
    live=int((n.item()+bm-1)//bm)
    for tail in ("as-is","forced -1"):
        eids=e[:rows//bm].clone()
        if tail=="forced -1": eids[live:]=-1
        te=int(eids[live].item()) if live<eids.numel() else -99
        a13=torch.empty((2,rows,H),dtype=torch.half,device=DEV); a2=torch.empty((1,rows,I),dtype=torch.half,device=DEV)
        ops.exl3_moe_had_in(x,a13,W["w13_suh"],s,eids,n,bm,TOPK,M*TOPK)
        f13=lambda: ops.exl3_moe_gemm(a13,W["w13_tr"],W["w13_suh"],W["w13_svh"],eids,n,[I,I],CB,bm,torch.bfloat16)
        inter=f13(); ops.exl3_moe_glu_had_in(inter,a2,W["w2_suh"],eids,n,bm)
        f2=lambda: ops.exl3_moe_gemm(a2,W["w2_tr"],W["w2_suh"],W["w2_svh"],eids,n,[H],CB,bm,torch.bfloat16)
        f2()
        print(f"{f'{lo}..{hi-1}':>16} {tail:>10} {te:>7} {live:>6} {rows:>7} | {t(f13):>9.0f} {t(f2):>9.0f}")
        del a13,a2,inter; torch.cuda.empty_cache()

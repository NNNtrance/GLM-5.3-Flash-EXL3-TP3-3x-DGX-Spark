#!/usr/bin/env python3
"""Per-rank, per-stage MoE timing under EP. Why is the high-id rank slower?"""
import statistics as st, sys
import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from cuda_exl3 import _C  # noqa: F401
ops = torch.ops.cuda_exl3_C
DEV="cuda"; H=4096; TOPK=8; E_GLOBAL=288; BITS=4; CB=1; TILE=16; EP=3; E_LOCAL=96
def ladder(rows, ne):
    per = rows/max(ne,1)
    return 16 if per<16 else 32 if per<48 else 64 if per<96 else 128
def weights(E,I,seed=7):
    g=torch.Generator(device=DEV).manual_seed(seed)
    return dict(I=I,E=E,
      w13_tr=torch.randint(-32768,32767,(E,H//TILE,2*I//TILE,TILE*BITS),dtype=torch.int16,device=DEV,generator=g),
      w2_tr=torch.randint(-32768,32767,(E,I//TILE,H//TILE,TILE*BITS),dtype=torch.int16,device=DEV,generator=g),
      w13_suh=(torch.randn((E,2,H),device=DEV,generator=g)*.05).half(),
      w2_suh=(torch.randn((E,1,I),device=DEV,generator=g)*.05).half(),
      w13_svh=(torch.randn((E,2*I),device=DEV,generator=g)*.05).half(),
      w2_svh=(torch.randn((E,H),device=DEV,generator=g)*.05).half())
def routing(M,seed):
    g=torch.Generator(device="cpu").manual_seed(seed)
    ids=torch.stack([torch.randperm(E_GLOBAL,generator=g)[:TOPK] for _ in range(M)])
    w=torch.rand((M,TOPK),generator=g); w=w/w.sum(1,keepdim=True)
    return ids.to(DEV,torch.int32), w.to(DEV,torch.float32)
def t(fn,it=20,wm=5):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it*1000.0

M = int(sys.argv[1]) if len(sys.argv)>1 else 2048
W = weights(E_LOCAL,2048); I=2048
ids,tw = routing(M, 9000+M)
x=(torch.randn((M,H),device=DEV)*.02).to(torch.bfloat16).contiguous()
print(f"M={M}  acc_cap={int(ops.exl3_get_moe_acc_cap())}")
print(f"{'rank':>5} {'pairs':>7} {'locblk':>7} {'liveblk':>8} {'n_rows':>8} {'rows':>7} | "
      f"{'had':>8} {'g13':>9} {'glu':>8} {'g2':>9} {'comb':>8} | {'TOTAL':>9}")
for r in range(EP):
    emap=torch.full((E_GLOBAL,),-1,dtype=torch.int32,device=DEV)
    emap[r*E_LOCAL:(r+1)*E_LOCAL]=torch.arange(E_LOCAL,dtype=torch.int32,device=DEV)
    bm=ladder(M*TOPK,E_GLOBAL)
    sids,eids,nrows=moe_align_block_size(ids,bm,E_GLOBAL,expert_map=emap,pad_sorted_ids=True)
    sids=sids.int(); eids=eids.int(); nrows=nrows.int()
    rows=min(eids.numel()*bm,sids.numel()); eids=eids[:rows//bm]
    live=int((nrows.item()+bm-1)//bm); locb=int((eids[:live]>=0).sum().item())
    pairs=int(((ids>=r*E_LOCAL)&(ids<(r+1)*E_LOCAL)).sum().item())
    a13=torch.empty((2,rows,H),dtype=torch.half,device=DEV)
    a2=torch.empty((1,rows,I),dtype=torch.half,device=DEV)
    f_had=lambda: ops.exl3_moe_had_in(x,a13,W["w13_suh"],sids,eids,nrows,bm,TOPK,M*TOPK)
    f_had()
    f_g13=lambda: ops.exl3_moe_gemm(a13,W["w13_tr"],W["w13_suh"],W["w13_svh"],eids,nrows,[I,I],CB,bm,torch.bfloat16)
    inter=f_g13()
    f_glu=lambda: ops.exl3_moe_glu_had_in(inter,a2,W["w2_suh"],eids,nrows,bm)
    f_glu()
    f_g2=lambda: ops.exl3_moe_gemm(a2,W["w2_tr"],W["w2_suh"],W["w2_svh"],eids,nrows,[H],CB,bm,torch.bfloat16)
    ro=f_g2()
    f_cb=lambda: ops.exl3_moe_combine(ro,sids,tw,M)
    u=[t(f) for f in (f_had,f_g13,f_glu,f_g2,f_cb)]
    print(f"{r:>5} {pairs:>7} {locb:>7} {live:>8} {int(nrows.item()):>8} {rows:>7} | "
          f"{u[0]:>8.0f} {u[1]:>9.0f} {u[2]:>8.0f} {u[3]:>9.0f} {u[4]:>8.0f} | {sum(u):>9.0f}")
    del a13,a2,inter,ro; torch.cuda.empty_cache()

#!/usr/bin/env python3
"""Does the EP gemm cost depend on WHERE in the global expert range a rank's
experts sit, rather than on how many rows it owns?"""
import sys, torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from cuda_exl3 import _C  # noqa: F401
ops=torch.ops.cuda_exl3_C
DEV="cuda"; H=4096; TOPK=8; E_GLOBAL=288; BITS=4; CB=1; TILE=16; E_LOCAL=96
def ladder(r,ne):
    p=r/max(ne,1); return 16 if p<16 else 32 if p<48 else 64 if p<96 else 128
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

M=int(sys.argv[1]) if len(sys.argv)>1 else 2048
W=weights(E_LOCAL,2048); I=2048
ids,tw=routing(M,9000+M)
x=(torch.randn((M,H),device=DEV)*.02).to(torch.bfloat16).contiguous()
import os
sel=os.environ.get("CASES","offsets")
if sel=="offsets":
    cases=[("contig off=0",list(range(0,96))),("contig off=48",list(range(48,144))),
           ("contig off=96",list(range(96,192))),("contig off=144",list(range(144,240))),
           ("contig off=192",list(range(192,288))),
           ("strided step3 (r0)",list(range(0,288,3))),
           ("strided step3 (r1)",list(range(1,288,3))),
           ("strided step3 (r2)",list(range(2,288,3)))]
elif sel=="single":
    cases=[(f"only expert {g}",[g]) for g in (0,1,95,96,191,192,250,285,286,287)]
elif sel=="edge":
    cases=[("192..286 (no 287)",list(range(192,287))),
           ("192..287 (with 287)",list(range(192,288))),
           ("193..288? clip 193..287",list(range(193,288))),
           ("0..95 + 287",list(range(0,96))+[287]),
           ("0..95 (ref)",list(range(0,96))),
           ("96..191 + 287",list(range(96,192))+[287])]
print(f"M={M}")
print(f"{'placement':>20} {'pairs':>7} {'locblk':>7} {'firstblk':>9} {'lastblk':>8} | "
      f"{'g13 us':>9} {'g2 us':>9} {'had us':>8} {'comb us':>8}")
for name,gl in cases:
    emap=torch.full((E_GLOBAL,),-1,dtype=torch.int32,device=DEV)
    emap[torch.tensor(gl,device=DEV)]=torch.arange(len(gl),dtype=torch.int32,device=DEV)
    bm=ladder(M*TOPK,E_GLOBAL)
    sids,eids,nrows=moe_align_block_size(ids,bm,E_GLOBAL,expert_map=emap,pad_sorted_ids=True)
    sids=sids.int(); eids=eids.int(); nrows=nrows.int()
    rows=min(eids.numel()*bm,sids.numel()); eids=eids[:rows//bm]
    live=int((nrows.item()+bm-1)//bm)
    pos=(eids[:live]>=0).nonzero().flatten()
    locb=pos.numel(); fb=int(pos[0]) if locb else -1; lb=int(pos[-1]) if locb else -1
    gset=torch.tensor(gl,device=DEV)
    pairs=int(torch.isin(ids,gset).sum().item())
    a13=torch.empty((2,rows,H),dtype=torch.half,device=DEV); a2=torch.empty((1,rows,I),dtype=torch.half,device=DEV)
    f_had=lambda: ops.exl3_moe_had_in(x,a13,W["w13_suh"],sids,eids,nrows,bm,TOPK,M*TOPK); f_had()
    f_g13=lambda: ops.exl3_moe_gemm(a13,W["w13_tr"],W["w13_suh"],W["w13_svh"],eids,nrows,[I,I],CB,bm,torch.bfloat16)
    inter=f_g13()
    f_glu=lambda: ops.exl3_moe_glu_had_in(inter,a2,W["w2_suh"],eids,nrows,bm); f_glu()
    f_g2=lambda: ops.exl3_moe_gemm(a2,W["w2_tr"],W["w2_suh"],W["w2_svh"],eids,nrows,[H],CB,bm,torch.bfloat16)
    ro=f_g2()
    f_cb=lambda: ops.exl3_moe_combine(ro,sids,tw,M)
    print(f"{name:>20} {pairs:>7} {locb:>7} {fb:>9} {lb:>8} | {t(f_g13):>9.0f} {t(f_g2):>9.0f} "
          f"{t(f_had):>8.0f} {t(f_cb):>8.0f}")
    del a13,a2,inter,ro; torch.cuda.empty_cache()

#!/usr/bin/env python3
"""What is in the last live block of moe_align_block_size's output?"""
import torch, sys
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from cuda_exl3 import _C  # noqa
DEV="cuda"; E_GLOBAL=288; TOPK=8
M=int(sys.argv[1]) if len(sys.argv)>1 else 2048
bm=64 if M>=2048 else 16
g=torch.Generator(device="cpu").manual_seed(9000+M)
ids=torch.stack([torch.randperm(E_GLOBAL,generator=g)[:TOPK] for _ in range(M)]).to(DEV,torch.int32)
emap=torch.full((E_GLOBAL,),-1,dtype=torch.int32,device=DEV)
emap[192:288]=torch.arange(96,dtype=torch.int32,device=DEV)
s,e,n=moe_align_block_size(ids,bm,E_GLOBAL,expert_map=emap,pad_sorted_ids=True)
s=s.int(); e=e.int(); n=n.int()
mv=M*TOPK
live=int((n.item()+bm-1)//bm)
print(f"M={M} bm={bm} sorted_ids.numel={s.numel()} expert_ids.numel={e.numel()} "
      f"n_rows={int(n.item())} live_blocks={live} m_valid={mv}")
print("expert_ids beyond live region (should never be read):", e[live:live+8].tolist(),
      "... max=", int(e[live:].max().item()) if e.numel()>live else "n/a")
sc=s.cpu(); ec=e.cpu()
for b in list(range(live-4, live)) + [live] :
    if b < 0 or b >= e.numel(): continue
    blk = sc[b*bm:(b+1)*bm]
    real = int((blk < mv).sum())
    print(f"  block {b:>4} expert_id={int(ec[b]):>4} real_rows={real:>3}/{bm} "
          f"sorted_ids[:6]={blk[:6].tolist()} max={int(blk.max())}")
# distribution of sorted_ids in the last block
blk = sc[(live-1)*bm:live*bm]
print("last live block sorted_ids:", blk.tolist())
print("counts of each expert id among live blocks:",
      torch.bincount(ec[:live].clamp(min=-1)+1, minlength=98)[:5].tolist(), "...")

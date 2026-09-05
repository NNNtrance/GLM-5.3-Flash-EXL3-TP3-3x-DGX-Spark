#!/usr/bin/env python3
"""What does the tail of expert_ids hold on OUR vLLM build?

The author's reading (issue #1): moe_align_block_size marks the surplus tail
-1; it is `expert_ids = expert_map[expert_ids]` that turns -1 into a real local
id, because -1 indexes the LAST element. If so the non-EP path (expert_map is
None, no mapping step) is never affected.

This prints, for the exact call our cuda_exl3/moe.py makes:
  * the raw tail of expert_ids straight out of moe_align_block_size, and
  * the tail after expert_map indexing, for the rank owning the top of the range
in both arrangements (expert_map given / expert_map None).
"""
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

DEV = "cuda"
E = 288
TOPK = 8


def routing(M, seed=1234):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)])
    return ids.to(DEV, torch.int32)


def show(M, block_m, emap, label):
    ids = routing(M)
    sorted_ids, expert_ids, n_rows = moe_align_block_size(
        ids, block_m, E, expert_map=emap, pad_sorted_ids=True)
    expert_ids = expert_ids.int()
    live = int((int(n_rows.item()) + block_m - 1) // block_m)
    raw = expert_ids.detach().clone()
    tail = raw[live:]
    print(f"\n[{label}] M={M} block_m={block_m} n_rows={int(n_rows.item())} "
          f"blocks={raw.numel()} live_blocks={live} surplus_blocks={raw.numel()-live}")
    print(f"  raw expert_ids[live-2:live+6] = {raw[max(0,live-2):live+6].tolist()}")
    if tail.numel():
        u = torch.unique(tail)
        print(f"  raw tail unique values      = {u.tolist()[:12]}"
              f"{' ...' if u.numel() > 12 else ''}   (all -1: {bool((tail < 0).all())})")
    else:
        print("  raw tail empty")
    return raw, live


def main():
    torch.cuda.init()
    print("vllm moe_align_block_size tail probe, E=288 top_k=8")

    # rank 2 owns 192..287, i.e. the top of the global range
    emap = torch.full((E,), -1, dtype=torch.int32, device=DEV)
    emap[192:] = torch.arange(0, 96, dtype=torch.int32, device=DEV)

    for M, bm in ((8, 16), (64, 16), (2048, 64)):
        # arrangement A: expert_map passed INTO moe_align_block_size (what our
        # cuda_exl3/moe.py does)
        raw_ep, live = show(M, bm, emap, "expert_map=<rank2 owns 192..287>")
        tail = raw_ep[live:]
        if tail.numel():
            print(f"  MAPPED tail unique          = {torch.unique(tail).tolist()[:12]}"
                  f"   -> real local expert on this rank: "
                  f"{bool(((tail >= 0)).any())}")

        # arrangement B: no expert_map at all (non-EP / TP-sliced path)
        raw_tp, live_tp = show(M, bm, None, "expert_map=None (non-EP)")
        tail_tp = raw_tp[live_tp:]
        if tail_tp.numel():
            allneg = bool((tail_tp < 0).all())
            print(f"  non-EP tail all -1          = {allneg}"
                  f"   -> surplus blocks {'ARE retired' if allneg else 'RUN A FULL GEMM'}")

        # arrangement C: mapping applied AFTER the alignment (the author's read of
        # where the -1 -> last-expert conversion happens)
        raw_none, live_none = raw_tp, live_tp
        mapped = emap[raw_none.long()]
        t = mapped[live_none:]
        if t.numel():
            print(f"  emap[raw_none] tail unique  = {torch.unique(t).tolist()[:12]}"
                  f"   (negative indexing of -1 gives emap[-1] = {int(emap[-1].item())})")


if __name__ == "__main__":
    main()

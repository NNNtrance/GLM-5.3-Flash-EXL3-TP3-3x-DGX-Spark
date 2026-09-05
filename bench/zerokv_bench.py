#!/usr/bin/env python3
"""Model-free bench of vLLM's _zero_kv_blocks_kernel with the live engine's KV geometry.

The engine's own start-up table reports, per LOGICAL block (3,328 tokens):
    MLAAttentionSpec   22 layers, page 109,824 B/layer  -> 2,416,128 B
    KpoolTailSpec      11 layers, page 109,824 B/layer  -> 1,208,064 B
    SlidingWindowSpec   5 layers, page 393,216 B/layer  -> 1,966,080 B   (DFlash draft)
    MambaSpec        34 layers                          -> SKIPPED (not an AttentionSpec)
    total zeroed per logical block                       = 5,590,272 B
"""
import json, os, statistics

import torch

torch.cuda.set_device(0)
DEV = "cuda"
OUT = os.environ.get("OUT", "/out/zerokv.json")
from vllm.v1.worker.utils import _zero_kv_blocks_kernel  # noqa: E402

s, e = torch.cuda.Event(True), torch.cuda.Event(True)

# ---- memset ruler
a = torch.empty(int(2.0 * (1 << 30) // 4), dtype=torch.int32, device=DEV)
for _ in range(3):
    a.zero_()
torch.cuda.synchronize(); s.record()
for _ in range(10):
    a.zero_()
e.record(); torch.cuda.synchronize()
ZERO = a.numel() * 4 / (s.elapsed_time(e) / 10 * 1e-3) / 1e9
print(f"memset ruler (2 GiB .zero_()): {ZERO:.1f} GB/s")
del a; torch.cuda.empty_cache()

GROUPS = [("MLA", 22, 109_824), ("Kpool", 11, 109_824), ("SWdraft", 5, 393_216)]
NBLK_TOTAL = 48
BYTES_PER_BLOCK = sum(n * p for _, n, p in GROUPS)
print(f"bytes zeroed per logical block: {BYTES_PER_BLOCK:,} B")

pool_bytes = NBLK_TOTAL * BYTES_PER_BLOCK
pool = torch.zeros(pool_bytes // 4, dtype=torch.int32, device=DEV)
base = pool.data_ptr()
print(f"pool {pool_bytes/2**20:.0f} MiB")


def build(ratio_mla: int, ratio_sw: int):
    """Return (seg_addrs, seg_strides, seg_pages) exactly as KVBlockZeroer builds them."""
    addrs, strides, pages = [], [], []
    off = 0
    for name, nlay, page in GROUPS:
        r = ratio_sw if name == "SWdraft" else ratio_mla
        assert page % r == 0, (name, page, r)
        kb = page // r                       # kernel-page bytes
        for _ in range(nlay):
            for v in range(r):
                addrs.append(base + off + v * kb)
                strides.append(page // 4)    # logical block stride, in int32
                pages.append(kb // 4)
            off += NBLK_TOTAL * page
    return addrs, strides, pages


def run(addrs, strides, pages, n_blocks, reps=25):
    ta = torch.tensor(addrs, dtype=torch.uint64, device=DEV)
    tb = torch.tensor(strides, dtype=torch.int64, device=DEV)
    tp = torch.tensor(pages, dtype=torch.int64, device=DEV)
    max_page = max(pages)
    blk = min(1 << (max_page - 1).bit_length(), 1024)
    max_chunks = (max_page + blk - 1) // blk
    ids = torch.arange(n_blocks, dtype=torch.int64, device=DEV)
    grid = (n_blocks, len(addrs), max_chunks)

    def call():
        _zero_kv_blocks_kernel[grid](ta, tb, tp, ids, BLOCK_SIZE=blk)
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), grid, blk, max_chunks


def run_grouped(addrs, strides, pages, n_blocks, reps=25):
    """PROPOSED FIX: one launch per distinct page size, so max_chunks fits each group."""
    buckets = {}
    for a_, b_, p_ in zip(addrs, strides, pages):
        buckets.setdefault(p_, ([], [], []))
        buckets[p_][0].append(a_); buckets[p_][1].append(b_); buckets[p_][2].append(p_)
    packs = []
    for p_, (A, B, P) in buckets.items():
        ta = torch.tensor(A, dtype=torch.uint64, device=DEV)
        tb = torch.tensor(B, dtype=torch.int64, device=DEV)
        tp = torch.tensor(P, dtype=torch.int64, device=DEV)
        blk = min(1 << (p_ - 1).bit_length(), 1024)
        mc = (p_ + blk - 1) // blk
        packs.append((ta, tb, tp, (n_blocks, len(A), mc), blk))
    ids = torch.arange(n_blocks, dtype=torch.int64, device=DEV)

    def call():
        for ta, tb, tp, grid, blk in packs:
            _zero_kv_blocks_kernel[grid](ta, tb, tp, ids, BLOCK_SIZE=blk)
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ctas = sum(g[0] * g[1] * g[2] for _, _, _, g, _ in packs)
    return statistics.median(ts), ctas, len(packs)


rows = []
for ratio_mla, ratio_sw, label in ((1, 1, "ratio=1 (kernel page == logical block)"),
                                   (13, 1, "ratio=13 for MLA/Kpool (block 3328 = 13 x 256)"),
                                   (13, 3, "ratio=13 MLA/Kpool, 3 for SW")):
    A, B, P = build(ratio_mla, ratio_sw)
    print(f"\n### {label}   n_segs={len(A)}  pages(el) min={min(P)} max={max(P)}")
    for nb in (1, 2, 4, 8, 16, 32):
        ms, grid, blk, mc = run(A, B, P, nb)
        useful = nb * BYTES_PER_BLOCK
        gb = useful / (ms * 1e-3) / 1e9
        msg, ctas, nlaunch = run_grouped(A, B, P, nb)
        gb2 = useful / (msg * 1e-3) / 1e9
        print(f"  n_blocks={nb:>2}: grid={grid} BLOCK_SIZE={blk} max_chunks={mc} "
              f"CTAs={grid[0]*grid[1]*grid[2]:>9,}  {ms:8.3f} ms  {gb:7.2f} GB/s "
              f"({gb/ZERO*100:5.1f}% of memset) || grouped {nlaunch} launches CTAs={ctas:,} "
              f"{msg:8.3f} ms  {gb2:7.2f} GB/s  speedup x{ms/msg:.2f}", flush=True)
        rows.append(dict(label=label, ratio_mla=ratio_mla, ratio_sw=ratio_sw, n_blocks=nb,
                         n_segs=len(A), max_chunks=mc, block_size=blk,
                         ctas=grid[0] * grid[1] * grid[2], ms=round(ms, 3), GBps=round(gb, 2),
                         grouped_launches=nlaunch, grouped_ctas=ctas, grouped_ms=round(msg, 3),
                         grouped_GBps=round(gb2, 2), speedup=round(ms / msg, 2)))

json.dump({"memset_ruler_GBps": round(ZERO, 1), "bytes_per_block": BYTES_PER_BLOCK,
           "groups": GROUPS, "rows": rows}, open(OUT, "w"), indent=1)
print("\nwrote", OUT)

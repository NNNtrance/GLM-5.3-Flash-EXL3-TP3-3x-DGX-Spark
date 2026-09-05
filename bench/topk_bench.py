#!/usr/bin/env python3
"""Price the DSA indexer top-k on GB10, model-free.

GLM-5.3-Flash: index_topk 2048, index_kpool 4 -> the logits are POOL granular
and the selection is select_k = 512 pools out of ceil(ctx/4). 11 DSA layers.

Three implementations are compared where they run at all:
  top_k_per_row_decode  -- the fallback we are pinned to (HAREM_DISABLE_PERSISTENT_TOPK=1)
  persistent_topk       -- the fast path, disabled on GB10 (needs >=128 KB smem,
                           GB10 has 101,376 B). Called anyway, to record HOW it fails.
  cooperative_topk      -- gated off upstream for device-capability family 120.
  torch.topk            -- reference for both correctness (as a set) and a
                           "what a generic kernel costs" yardstick.
"""
import argparse, json, sys, traceback
import torch
import vllm._custom_ops as _vops  # loads torch.ops._C

SELECT_K = 512
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def timeit(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="8,16,64")
    ap.add_argument("--pools", default="512,1024,2048,4096,16384,32768")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} sms={props.multi_processor_count} "
          f"smem/block={props.shared_memory_per_block} "
          f"smem/block_optin={getattr(props,'shared_memory_per_block_optin','?')} "
          f"smem/sm={props.shared_memory_per_multiprocessor}")
    print(f"select_k={SELECT_K}\n")
    rows_l = [int(v) for v in a.rows.split(",")]
    pools_l = [int(v) for v in a.pools.split(",")]
    rec = []
    print(f"{'rows':>5} {'pools':>7} | {'decode_fb us':>13} {'persistent us':>14} "
          f"{'torch.topk us':>14} | {'fb/torch':>9} | note")
    for rows in rows_l:
        for P in pools_l:
            if P < SELECT_K: continue
            logits = torch.randn((rows, P), device=dev, dtype=torch.float32)
            seq_lens = torch.full((rows, 1), P, device=dev, dtype=torch.int32)
            dst = torch.empty((rows, SELECT_K), device=dev, dtype=torch.int32)
            note = ""
            # fallback
            def f_fb():
                torch.ops._C.top_k_per_row_decode(
                    logits, 1, seq_lens, dst, rows, logits.stride(0), logits.stride(1), SELECT_K)
            try:
                t_fb = timeit(f_fb)
            except Exception as ex:
                t_fb = float("nan"); note += f"fallback FAILED: {type(ex).__name__} "
            # persistent
            ws = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, device=dev, dtype=torch.uint8)
            dst2 = torch.empty((rows, SELECT_K), device=dev, dtype=torch.int32)
            def f_ps():
                torch.ops._C.persistent_topk(logits, seq_lens, dst2, ws, SELECT_K, P)
            try:
                f_ps(); torch.cuda.synchronize()
                t_ps = timeit(f_ps)
            except Exception as ex:
                t_ps = float("nan")
                note += f"persistent FAILED: {str(ex).splitlines()[0][:90]} "
            def f_tt(): torch.topk(logits, SELECT_K, dim=-1, sorted=False)
            t_tt = timeit(f_tt)
            # correctness as a set, fallback vs torch
            ref = set(torch.topk(logits[0], SELECT_K, sorted=False).indices.tolist())
            try:
                f_fb(); torch.cuda.synchronize()
                got = set(dst[0].tolist())
                if got != ref: note += f"SET MISMATCH ({len(got & ref)}/{SELECT_K} agree) "
            except Exception:
                pass
            ratio = t_fb / t_tt if t_tt else float("nan")
            print(f"{rows:>5} {P:>7} | {t_fb:>13.1f} {t_ps:>14.1f} {t_tt:>14.1f} | "
                  f"{ratio:>9.2f} | {note}")
            rec.append(dict(rows=rows, pools=P, decode_fallback_us=t_fb,
                            persistent_us=t_ps, torch_topk_us=t_tt, note=note))
            del logits, dst, dst2, ws
            torch.cuda.empty_cache()
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)


main()

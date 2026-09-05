#!/usr/bin/env python3
"""Final taxonomy for the GLM-5.3-Flash / cuda-exl3 / TP3+EP stack."""
import gzip, json, re, sys, collections

RULES = [
    ("MoE trellis GEMM",   r"exl3_gemm_m_kernel|exl3_epilogue"),
    ("MoE had_in/glu",     r"exl3_moe_had_in|exl3_moe_glu_had_in"),
    ("MoE combine/inv",    r"exl3_moe_combine|exl3_moe_build_inv"),
    ("MoE align/route",    r"moe_align|single_group_topk|count_and_sort|sgl_moe|topk_softmax|moe_sum"),
    ("NCCL collectives",   r"nccl|AllReduce|AllGather|ReduceScatter"),
    ("HC mixing (hyper-conn)", r"mhc_|hc_prenorm|hc_head"),
    ("MLA attention",      r"mla_decode_partial|mla_decode_reduce|flash_fwd|_fused_q_kv_rmsnorm|merge_16x16_to_64x64"),
    ("MLA autotune evict", r"exl3_evict_kernel"),
    ("DSA indexer",        r"mqa_logits|topKPerRow|top_k_per_row|_expand_pools|_convert_req_index|_fwht_quant|indexer_k_quant|cp_gather"),
    ("KDA linear-attn",    r"chunk_gla|chunk_kda|chunk_gated_delta|gated_delta|recompute_w_u|l2norm|kda_gate|layer_norm_gated|solve_tril|causal_conv1d|_gather_initial_states|fused_recurrent"),
    ("Dense BF16 GEMM",    r"cutlass|nvjet|gemvx|gemmSN|splitKreduce|sgemm|simt"),
    ("draft (DFlash2) aux", r"_prepare_dflash_inputs|_combine_sampled_and_draft"),
    ("KV cache",           r"reshape_and_cache|concat_and_cache|cache_kernel|_zero_kv_blocks|store_kv"),
    ("Norm / elementwise", r"rms_norm|layer_norm|silu|act_and_mul|elementwise|CatArrayBatched|copy_|vectorized|unrolled|fill_|index_|gather|scatter|cat_|reduce_kernel|Reduce"),
    ("Sampling / logits",  r"sample|gumbel|argmax|softmax|penalt|logits"),
    ("Quant / cast",       r"quant|scaled_mm|cvt_|convert|_cast|fp8"),
]


def cls(n):
    for c, p in RULES:
        if re.search(p, n):
            return c
    return "OTHER"


def show(title, seg, wall):
    busy = sum(e["dur"] for e in seg)
    c = collections.Counter(); cn = collections.Counter()
    for e in seg:
        k = cls(e["name"]); c[k] += e["dur"]; cn[k] += 1
    print(f"\n### {title}   wall {wall/1000:.1f} ms   gpu_busy {busy/1000:.1f} ms   "
          f"occ {busy/wall*100:.1f}%")
    print(f"    {'class':<24} {'ms':>8} {'%busy':>7} {'%wall':>7} {'calls':>7}")
    for k, d in c.most_common():
        if d / busy < 0.0005:
            continue
        print(f"    {k:<24} {d/1000:>8.2f} {d/busy*100:>6.1f}% {d/wall*100:>6.1f}% {cn[k]:>7}")
    return c, busy


def main():
    path = sys.argv[1]
    tr = json.load(gzip.open(path, "rt"))
    k = sorted([e for e in tr["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    tend = max(e["ts"] + e["dur"] for e in k)
    marks = [e["ts"] for e in k if "_zero_kv_blocks" in e["name"]]
    if len(marks) < 2:
        marks = [e["ts"] for e in k if "_prepare_dflash_inputs" in e["name"]]
    marks = marks + [tend]
    labels = sys.argv[2].split(",") if len(sys.argv) > 2 else [f"pass{i}" for i in range(len(marks) - 1)]
    show("WHOLE WINDOW", k, tend - k[0]["ts"])
    agg = collections.Counter(); n = 0; wtot = 0
    for i in range(len(marks) - 1):
        a, b = marks[i], marks[i + 1]
        seg = [e for e in k if a <= e["ts"] < b]
        lab = labels[i] if i < len(labels) else f"pass{i}"
        c, busy = show(lab, seg, b - a)
        if lab.startswith("STEADY"):
            agg.update(c); n += 1; wtot += (b - a)
    if n:
        print(f"\n### MEAN OF {n} STEADY PASSES  wall {wtot/n/1000:.1f} ms")
        tb = sum(agg.values())
        print(f"    {'class':<24} {'ms':>8} {'%busy':>7}")
        for kk, d in agg.most_common():
            if d / tb < 0.0005:
                continue
            print(f"    {kk:<24} {d/n/1000:>8.2f} {d/tb*100:>6.1f}%")


main()

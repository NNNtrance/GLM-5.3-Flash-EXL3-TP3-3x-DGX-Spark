#!/usr/bin/env python3
"""Aggregate a vLLM/torch trace into kernel classes with % of GPU time."""
import gzip, json, re, sys, collections

CLASSES = [
    ("MoE EXL3 gemm",      r"exl3_gemm_m_kernel|exl3_epilogue"),
    ("MoE had_in/glu",     r"exl3_moe_had_in|exl3_moe_glu_had_in"),
    ("MoE combine/inv",    r"exl3_moe_combine|exl3_moe_build_inv"),
    ("MoE zero (mask)",    r"masked_fill|MaskedFill|where_kernel"),
    ("MoE align/sort",     r"moe_align|count_and_sort|sgl_moe|topk_softmax|moe_sum|repeat_interleave"),
    ("Dense EXL3 linear",  r"exl3_had_in_kernel|exl3_gemm_kernel|exl3_linear"),
    ("DSA indexer logits", r"mqa_logits|fp8_mqa|deep_gemm|DeepGEMM|indexer_k_quant|cp_gather"),
    ("DSA indexer top-k",  r"top_k_per_row|topk_per_row|persistent_topk|cooperative_topk|radix"),
    ("MLA attention",      r"mla|MLA|flash_mla|_fwd_kernel_stage|paged"),
    ("KDA scan (triton)",  r"kda|chunk_gated|gated_delta|delta_rule|fused_recurrent|chunk_fwd|chunk_o|chunk_h|wy_fast|l2norm|solve_tril|cumsum",),
    ("KDA conv1d",         r"causal_conv1d|conv1d|_conv_"),
    ("Attention (other)",  r"flash|attn|attention|rotary|rope"),
    ("NCCL / collectives", r"nccl|ncclDevKernel|AllReduce|all_reduce|ReduceScatter|AllGather|reduce_scatter"),
    ("Quant / cast",       r"quant|scaled_mm|cvt_|convert|to_copy|_cast|fp8"),
    ("Norm / elementwise", r"rms_norm|layer_norm|LayerNorm|silu|gelu|elementwise|CatArrayBatched|copy_|Copy|vectorized|unrolled|fill_|index_|gather|scatter|cat_"),
    ("Sampling / logits",  r"sample|gumbel|argmax|softmax|penalt|logits"),
    ("Cache / reshape",    r"reshape_and_cache|concat_and_cache|cache_kernel|store_kv|block_table"),
]


def cls(name):
    for c, pat in CLASSES:
        if re.search(pat, name):
            return c
    return "OTHER"

def load(p):
    o = gzip.open if p.endswith(".gz") else open
    with o(p, "rt") as f:
        return json.load(f)

def main():
    path = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    tr = load(path)
    ev = tr["traceEvents"]
    ker = [e for e in ev if e.get("cat") in ("kernel", "Kernel", "gpu_user_annotation")
           and e.get("cat") != "gpu_user_annotation"]
    if not ker:
        ker = [e for e in ev if e.get("cat") == "kernel"]
    tot = sum(e.get("dur", 0) for e in ker)
    byname = collections.Counter()
    cntname = collections.Counter()
    bycls = collections.Counter()
    cntcls = collections.Counter()
    for e in ker:
        n = e.get("name", "?")
        byname[n] += e.get("dur", 0); cntname[n] += 1
        c = cls(n); bycls[c] += e.get("dur", 0); cntcls[c] += 1
    # wall span of the GPU stream
    t0 = min(e["ts"] for e in ker); t1 = max(e["ts"] + e.get("dur", 0) for e in ker)
    span = t1 - t0
    print(f"file={path}")
    print(f"kernels={len(ker)}  gpu_busy={tot/1000:.1f} ms  span={span/1000:.1f} ms  "
          f"occupancy={tot/span*100:.1f}%  (gap/CPU-bound = {100-tot/span*100:.1f}%)")
    print(f"\n{'class':<24} {'ms':>10} {'%GPU':>7} {'%span':>7} {'calls':>8}")
    for c, d in bycls.most_common():
        print(f"{c:<24} {d/1000:>10.2f} {d/tot*100:>6.1f}% {d/span*100:>6.1f}% {cntcls[c]:>8}")
    print(f"\ntop {topn} kernels")
    print(f"{'ms':>10} {'%':>6} {'calls':>8}  name")
    for n, d in byname.most_common(topn):
        print(f"{d/1000:>10.2f} {d/tot*100:>5.1f}% {cntname[n]:>8}  {n[:120]}")

main()

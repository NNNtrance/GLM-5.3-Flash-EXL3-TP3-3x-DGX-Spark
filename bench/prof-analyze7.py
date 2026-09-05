#!/usr/bin/env python3
"""PROFIL-URETIM7 trace analyzer — measured class breakdown per engine step.

Streams the (pretty-printed) torch trace, so a 1 GB decode trace fits in a few hundred MB.

Step boundaries
---------------
The engine emits a `record_function` per engine step; kineto projects it onto the GPU
timeline as `gpu_user_annotation` named `execute_context_N(T)_generation_M(T)`.
In decode these come in OVERLAPPING PAIRS (two per step, same name, one nested in the
other), so they are merged first.  A merged region covers the TARGET model forward only —
the DFlash2 drafter runs after it, outside any annotation.  One engine step is therefore
    [merged_k.start, merged_{k+1}.start)
      target part = [merged_k.start, merged_k.end)
      draft  part = [merged_k.end,   merged_{k+1}.start)
which covers the timeline with no holes, so every kernel is accounted for exactly once.

usage:  prof-analyze7.py <trace.json.gz> [--label X] [--json out.json] [--skip N]
"""
import argparse, bisect, collections, gzip, json, re, sys

# ---------------------------------------------------------------- taxonomy
RULES = [
    ("MoE hadamard (had_in/glu)", r"exl3_moe_had_in_kernel|exl3_moe_glu_had_in_kernel"),
    ("MoE trellis GEMM",          r"exl3_gemm_m_kernel|exl3_epilogue"),
    ("MoE align/route",           r"moe_align_block_size|count_and_sort_expert_tokens|"
                                  r"single_group_topk|moe_sum|exl3_moe_combine|exl3_moe_build_inv"),
    ("NCCL collectives",          r"ncclDevKernel|ncclKernel"),
    ("HC mixing (mhc_*)",         r"mhc_|hc_prenorm"),
    ("Sparse indexer (DSA)",      r"fp8_mqa_logits|topKPerRow|_fwht_quant|_convert_req_index|"
                                  r"_expand_pools|cp_gather_indexer|_kpool_"),
    ("MLA attention",             r"mla_decode_partial|mla_decode_reduce|_fused_q_kv_rmsnorm|"
                                  r"concat_and_cache_mla|kernel_mha|reshape_and_cache_flash"),
    ("KDA/GDN linear-attn",       r"chunk_gla|chunk_kda|chunk_gated_delta|recompute_w_u|"
                                  r"causal_conv1d|kda_gate|l2norm_fwd|layer_norm_gated|"
                                  r"merge_16x16_to_64x64|_gather_initial_states|_scatter_states|"
                                  r"fused_recurrent_gated_delta|mamba_align|mamba_fused|"
                                  r"triton_poi_fused__to_copy_sigmoid|solve_tril|wy_fast"),
    ("KV zero (_zero_kv_blocks)", r"_zero_kv_blocks"),
    ("Dense BF16 GEMM",           r"cutlass|nvjet|gemvx|splitKreduce|sgemm|gemmSN|_simt_"),
    ("Sampling / spec bookkeep",  r"_gumbel_sample|RadixTopK|StableSortTopK|ArgMaxOps|argmax|"
                                  r"_get_num_sampled|_combine_sampled_and_draft|_scatter_num_accepted|"
                                  r"_post_update|_selector_walk|_apply_write|_prepare_dflash_inputs|"
                                  r"_prepare_prefill_inputs|_compute_slot_mappings|_gather_block_tables|"
                                  r"_copy_page_indices|_compressed_slot_mapping|_prepare_pos_seq_lens|"
                                  r"DeviceScan|penalt|logits_proc"),
]
COMP = [(n, re.compile(p)) for n, p in RULES]


def cls(name, cat):
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy / memset"
    for n, p in COMP:
        if p.search(name):
            return n
    return "Norm / elementwise / copy"


# ---------------------------------------------------------------- streaming reader
EV_START = re.compile(r"^\s*\{\s*$")
WANT = ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation", "user_annotation")


def extract(path):
    gpu, ann, cpuann = [], [], []
    buf, depth, inev, started = [], 0, False, False
    with gzip.open(path, "rt") as f:
        for line in f:
            if not started:
                if '"traceEvents"' in line:
                    started = True
                continue
            if not inev:
                if EV_START.match(line):
                    inev, buf, depth = True, [line], 1
                continue
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                inev = False
                try:
                    e = json.loads("".join(buf).rstrip().rstrip(","))
                except Exception:                                  # noqa: BLE001
                    continue
                c = e.get("cat")
                if c not in WANT:
                    continue
                rec = (e.get("ts", 0.0), e.get("dur", 0.0), e.get("name", "?"), c)
                (ann if c == "gpu_user_annotation" else
                 cpuann if c == "user_annotation" else gpu).append(rec)
    gpu.sort(key=lambda r: r[0])
    ann.sort(key=lambda r: r[0])
    cpuann.sort(key=lambda r: r[0])
    return gpu, ann, cpuann


def merge(ann):
    """Merge overlapping annotations; keep the widest name of the group."""
    out = []
    for ts, dur, name, _ in ann:
        if out and ts <= out[-1][1]:
            out[-1][1] = max(out[-1][1], ts + dur)
        else:
            out.append([ts, ts + dur, name])
    return out


def union(evs):
    iv = sorted((e[0], e[0] + e[1]) for e in evs)
    out, cs, ce = 0.0, None, None
    for a, b in iv:
        if cs is None:
            cs, ce = a, b
        elif a <= ce:
            ce = max(ce, b)
        else:
            out += ce - cs
            cs, ce = a, b
    if cs is not None:
        out += ce - cs
    return out


STEP_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--skip", type=int, default=1)
    args = ap.parse_args()

    gpu, ann, cpuann = extract(args.trace)
    reg = merge(ann)
    print(f"# {args.label or args.trace}")
    print(f"gpu events {len(gpu)}   gpu annotations {len(ann)} -> merged {len(reg)}   "
          f"cpu annotations {len(cpuann)}")

    kinds = collections.Counter()
    for _, _, name in reg:
        m = STEP_RE.search(name)
        kinds[m.groups() if m else ("?",)] += 1
    for k, c in kinds.most_common(6):
        print(f"   step kind ctx {k[0]} seq/{k[1]} tok, gen {k[2]} seq/{k[3]} tok   x{c}")
    m0 = STEP_RE.search(reg[0][2])
    prefill = int(m0.group(2)) > 0

    # ---- steps as full periods: [reg_k.start, reg_{k+1}.start)
    steps = []
    for k in range(len(reg) - 1):
        a, b, name = reg[k]
        steps.append(dict(t0=a, tmid=b, t1=reg[k + 1][0], name=name))
    if not steps:
        sys.exit("need at least two annotated steps")

    starts = [s["t0"] for s in steps]
    for s in steps:
        s["ev"] = []
    lo, hi = steps[0]["t0"], steps[-1]["t1"]
    unassigned = 0.0
    for e in gpu:
        if e[0] < lo or e[0] >= hi:
            unassigned += e[1]
            continue
        steps[bisect.bisect_right(starts, e[0]) - 1]["ev"].append(e)
    print(f"kernel ms outside the analysed span: {unassigned/1000:.1f} "
          f"(edge steps, dropped)")

    # keep only steps of the dominant kind, drop ramp
    dom = max(kinds, key=kinds.get)
    sel = [s for s in steps[args.skip:]
           if (STEP_RE.search(s["name"]).groups() if STEP_RE.search(s["name"]) else None) == dom]
    # drop the longest 2 % (scheduler hiccups / window edges) only for the mean-period stat
    n = len(sel)
    print(f"\n== analysed steps: {n} "
          f"({'prefill chunk' if prefill else 'decode'} kind ctx{dom[0]}({dom[1]})"
          f"/gen{dom[2]}({dom[3]})) ==")

    wall = sum(s["t1"] - s["t0"] for s in sel) / n
    tgt_wall = sum(s["tmid"] - s["t0"] for s in sel) / n
    drf_wall = sum(s["t1"] - s["tmid"] for s in sel) / n
    busy = sum(union(s["ev"]) for s in sel) / n
    dur = sum(e[1] for s in sel for e in s["ev"]) / n
    gap = wall - busy

    print(f"mean engine step: wall {wall/1000:.3f} ms   gpu busy {busy/1000:.3f} ms   "
          f"occupancy {busy/wall*100:.1f}%   CPU gap (GPU idle) {gap/1000:.3f} ms "
          f"({gap/wall*100:.2f}%)")
    print(f"  annotated (target model) {tgt_wall/1000:.3f} ms   "
          f"un-annotated tail (DFlash2 draft) {drf_wall/1000:.3f} ms")
    print(f"  (kernel-duration sum {dur/1000:.3f} ms; single stream, "
          f"sum-union = {(dur-busy)/1000:.3f} ms)")

    agg, cnt = collections.Counter(), collections.Counter()
    tgt, drf = collections.Counter(), collections.Counter()
    sub = collections.Counter()
    subn = collections.Counter()
    for s in sel:
        last = None
        for e in s["ev"]:
            c = cls(e[2], e[3])
            agg[c] += e[1]
            cnt[c] += 1
            (drf if e[0] >= s["tmid"] else tgt)[c] += e[1]
            if "exl3_moe_had_in_kernel" in e[2]:
                last = "  ...gate/up (w13)"
            elif "exl3_moe_glu_had_in_kernel" in e[2]:
                last = "  ...down (w2)"
            elif "exl3_gemm_m_kernel" in e[2] and last:
                sub[last] += e[1]
                subn[last] += 1

    rows = []
    print(f"\n{'class':<28} {'ms/step':>9} {'%step':>7} {'calls':>8} {'target':>9} {'draft':>8}")
    for k, d in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"{k:<28} {d/n/1000:>9.3f} {d/n/wall*100:>6.2f}% {cnt[k]/n:>8.1f} "
              f"{tgt[k]/n/1000:>9.3f} {drf[k]/n/1000:>8.3f}")
        rows.append([k, d / n / 1000, d / n / wall * 100, cnt[k] / n,
                     tgt[k] / n / 1000, drf[k] / n / 1000])
        if k == "MoE trellis GEMM":
            for sk in ("  ...gate/up (w13)", "  ...down (w2)"):
                if sub.get(sk):
                    print(f"{sk:<28} {sub[sk]/n/1000:>9.3f} {sub[sk]/n/wall*100:>6.2f}% "
                          f"{subn[sk]/n:>8.1f}")
                    rows.append([sk, sub[sk] / n / 1000, sub[sk] / n / wall * 100,
                                 subn[sk] / n, None, None])
    print(f"{'CPU gap (GPU idle)':<28} {gap/1000:>9.3f} {gap/wall*100:>6.2f}%")
    rows.append(["CPU gap (GPU idle)", gap / 1000, gap / wall * 100, 0, None, None])
    print(f"{'TOTAL (wall)':<28} {wall/1000:>9.3f} {100.0:>6.2f}%")

    dt = sum(drf.values()) / n
    print(f"\nDFlash2 draft total (all classes, un-annotated tail): "
          f"{dt/1000:.3f} ms = {dt/wall*100:.2f}% of the step; "
          f"its collectives {drf['NCCL collectives']/n/1000:.3f} ms")

    if args.json:
        json.dump(dict(label=args.label, trace=args.trace, steps=n, prefill=prefill,
                       wall_ms=wall / 1000, busy_ms=busy / 1000, gap_ms=gap / 1000,
                       target_wall_ms=tgt_wall / 1000, draft_wall_ms=drf_wall / 1000,
                       draft_total_ms=dt / 1000, rows=rows),
                  open(args.json, "w"), indent=1)
        print(f"\n-> {args.json}")


main()

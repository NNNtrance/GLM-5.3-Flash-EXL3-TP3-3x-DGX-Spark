#!/usr/bin/env python3
"""PROFIL-URETIM7 gap analyzer -- WHERE the GPU idle time goes, per engine step.

Companion to prof-analyze7.py.  That one answers "which kernel class costs what";
this one answers "when the GPU is idle, what was the host doing".

Method
------
* streaming line parser (no json.loads per event): the 892 MB pretty-printed C1
  trace is read in ~10 s / ~1.5 GB RSS, no temp files.
* GPU busy = union of every kernel / memcpy / memset interval over ALL streams of
  the device.  Its complement inside a step is the set of GPU idle gaps.
* steps come from the merged gpu_user_annotation regions (see prof-analyze7.py):
      step_k = [merged_k.start, merged_{k+1}.start)
      target part = [merged_k.start, merged_k.end)   (main model forward)
      draft  part = [merged_k.end,   merged_{k+1}.start)   (DFlash2, outside the ann.)
* every gap is attributed with
    - the kernel ending at gap start and the kernel starting at gap end
    - the launch (cuda_runtime, matched by "correlation") of that next kernel:
        launch_ts >= gap start -> HOST-BOUND (work had not been issued yet)
        launch_ts <  gap start -> DEVICE-SIDE WAIT (queued, blocked on an event)
    - the innermost enclosing cpu_op (aten:: / vllm:: names; cpu_ops are properly
      nested on a single thread, so walking back to the first range that ends after
      the gap gives the innermost one)
    - every cuda_runtime call the host made during the gap (sync / launch / memcpy)
    - whether the host was inside the step's own CPU user_annotation

usage: prof-gap7.py <trace.json.gz> [--label X] [--min-gap 0.1] [--steps N]
                    [--json out.json] [--dump-steps N] [--skip N]
"""
import argparse, bisect, collections, gzip, json, re, sys

CORR_RE = re.compile(r'"correlation": (\d+)')
STEP_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
KEEP = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation",
        "user_annotation", "cpu_op", "cuda_runtime", "cuda_driver"}
CORR_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"}
# triton / inductor kernels are launched through the DRIVER api, not the runtime api;
# without these the host-bound test silently misses every triton launch.
DRV_KEEP = re.compile(r"^cu(LaunchKernel|LaunchKernelEx|MemcpyAsync|MemcpyDtoHAsync|"
                      r"StreamSynchronize|CtxSynchronize|EventSynchronize)")
RT_KEEP = re.compile(r"^cuda(LaunchKernel|LaunchKernelExC|MemcpyAsync|Memcpy$|"
                     r"StreamSynchronize|DeviceSynchronize|EventSynchronize|EventQuery|"
                     r"StreamWaitEvent|MemsetAsync|GraphLaunch|Malloc|Free|HostAlloc)")
BLOCKING = re.compile(r"cu[da]*(StreamSynchronize|DeviceSynchronize|CtxSynchronize|EventSynchronize)")
LAUNCHY = re.compile(r"^cu[da]*LaunchKernel")

# same taxonomy as prof-analyze7.py, used to say WHICH kernel each micro-gap precedes
RULES = [
    ("MoE hadamard", r"exl3_moe_had_in_kernel|exl3_moe_glu_had_in_kernel"),
    ("MoE trellis GEMM", r"exl3_gemm_m_kernel|exl3_epilogue"),
    ("MoE align/route", r"moe_align_block_size|count_and_sort_expert_tokens|"
                        r"single_group_topk|moe_sum|exl3_moe_combine|exl3_moe_build_inv"),
    ("NCCL collectives", r"ncclDevKernel|ncclKernel"),
    ("HC mixing (mhc_*)", r"mhc_|hc_prenorm"),
    ("Sparse indexer (DSA)", r"fp8_mqa_logits|topKPerRow|_fwht_quant|_convert_req_index|"
                             r"_expand_pools|cp_gather_indexer|_kpool_"),
    ("MLA attention", r"mla_decode_partial|mla_decode_reduce|_fused_q_kv_rmsnorm|"
                      r"concat_and_cache_mla|kernel_mha|reshape_and_cache_flash"),
    ("KDA/GDN linear-attn", r"chunk_gla|chunk_kda|chunk_gated_delta|recompute_w_u|"
                            r"causal_conv1d|kda_gate|l2norm_fwd|layer_norm_gated|"
                            r"merge_16x16_to_64x64|_gather_initial_states|_scatter_states|"
                            r"fused_recurrent_gated_delta|mamba_align|mamba_fused|"
                            r"triton_poi_fused__to_copy_sigmoid|solve_tril|wy_fast"),
    ("KV zero", r"_zero_kv_blocks"),
    ("Dense BF16 GEMM", r"cutlass|nvjet|gemvx|splitKreduce|sgemm|gemmSN|_simt_"),
    ("Sampling / spec bookkeep", r"_gumbel_sample|RadixTopK|StableSortTopK|ArgMaxOps|argmax|"
                                 r"_get_num_sampled|_combine_sampled_and_draft|_scatter_num_accepted|"
                                 r"_post_update|_selector_walk|_apply_write|_prepare_dflash_inputs|"
                                 r"_prepare_prefill_inputs|_compute_slot_mappings|_gather_block_tables|"
                                 r"_copy_page_indices|_compressed_slot_mapping|_prepare_pos_seq_lens|"
                                 r"DeviceScan|penalt|logits_proc"),
]
COMP = [(n, re.compile(pp)) for n, pp in RULES]
_CC = {}


def kcls(name, cat="kernel"):
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy / memset"
    r = _CC.get(name)
    if r is None:
        r = "Norm / elementwise / copy"
        for nm, pp in COMP:
            if pp.search(name):
                r = nm
                break
        _CC[name] = r
    return r


HIST = [0.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, 1e9]


# ------------------------------------------------------------------ parser
def parse(path):
    gpu, ann, cpuann, cpuop, rt = [], [], [], [], []
    cur = None
    intern = sys.intern
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.lstrip()
            if not s or s[0] != '"':
                continue
            if s.startswith('"ph"'):
                if cur is not None:
                    _emit(cur, gpu, ann, cpuann, cpuop, rt)
                cur = [None, None, 0, 0, 0.0, 0.0, 0] if s[7] == "X" else None
                continue
            if cur is None:
                continue
            if s.startswith('"cat"'):
                c = s[8:s.index('"', 8)]
                if c not in KEEP:
                    cur = None
                    continue
                cur[0] = intern(c)
            elif s.startswith('"name"'):
                cur[1] = intern(s[9:s.rindex('"')])
            elif s.startswith('"pid"'):
                cur[2] = int(s[7:].rstrip().rstrip(","))
            elif s.startswith('"tid"'):
                cur[3] = int(s[7:].rstrip().rstrip(","))
            elif s.startswith('"ts"'):
                cur[4] = float(s[6:].rstrip().rstrip(","))
            elif s.startswith('"dur"'):
                cur[5] = float(s[7:].rstrip().rstrip(","))
            elif cur[0] in CORR_CATS and '"correlation"' in s:
                m = CORR_RE.search(s)
                if m:
                    cur[6] = int(m.group(1))
    if cur is not None:
        _emit(cur, gpu, ann, cpuann, cpuop, rt)
    for lst in (gpu, ann, cpuann, cpuop, rt):
        lst.sort(key=lambda r: r[4])
    return gpu, ann, cpuann, cpuop, rt


def _emit(e, gpu, ann, cpuann, cpuop, rt):
    c = e[0]
    if c is None or e[1] is None:
        return
    if c in GPU_CATS:
        gpu.append(e)
    elif c == "gpu_user_annotation":
        ann.append(e)
    elif c == "user_annotation":
        cpuann.append(e)
    elif c == "cpu_op":
        cpuop.append(e)
    elif c == "cuda_runtime" and RT_KEEP.match(e[1]):
        rt.append(e)
    elif c == "cuda_driver" and DRV_KEEP.match(e[1]):
        rt.append(e)


def merge(ann):
    out = []
    for e in ann:
        ts, dur = e[4], e[5]
        if out and ts <= out[-1][1]:
            out[-1][1] = max(out[-1][1], ts + dur)
        else:
            out.append([ts, ts + dur, e[1]])
    return out


def gaps_of(iv, lo, hi):
    """complement of the union of sorted intervals iv inside [lo,hi); also busy time"""
    out, cs, ce, busy, cursor = [], None, None, 0.0, lo
    for a, b in iv:
        if b <= lo or a >= hi:
            continue
        a, b = max(a, lo), min(b, hi)
        if cs is None:
            cs, ce = a, b
        elif a <= ce:
            ce = max(ce, b)
        else:
            busy += ce - cs
            out.append((cursor, cs))
            cursor = ce
            cs, ce = a, b
    if cs is not None:
        busy += ce - cs
        out.append((cursor, cs))
        cursor = ce
    if hi - cursor > 0:
        out.append((cursor, hi))
    return [g for g in out if g[1] > g[0]], busy


# ------------------------------------------------------------------ helpers
def enclosing(cpuop, starts, t0, t1, maxback=6000):
    """innermost cpu_op covering [t0,t1); cpu_ops nest on one thread, so the first
    range (walking back) that ends at/after t1 is the innermost enclosing one."""
    j = bisect.bisect_right(starts, t0) - 1
    stop = max(-1, j - maxback)
    while j > stop:
        e = cpuop[j]
        if e[4] + e[5] >= t1:
            return e
        j -= 1
    return None


def neighbours(cpuop, starts, t0, t1):
    """last cpu_op that ended before the gap, first that starts inside/after it"""
    j = bisect.bisect_right(starts, t0) - 1
    last = None
    k = j
    stop = max(-1, j - 400)
    while k > stop:
        e = cpuop[k]
        if e[4] + e[5] <= t0:
            last = e
            break
        k -= 1
    i = bisect.bisect_left(starts, t0)
    nxt = cpuop[i] if i < len(cpuop) else None
    return last, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--label", default="")
    ap.add_argument("--min-gap", type=float, default=0.1, help="ms")
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--dump-steps", type=int, default=2)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    MG = args.min_gap * 1000.0

    gpu, ann, cpuann, cpuop, rt = parse(args.trace)
    reg = merge(ann)
    print(f"# {args.label or args.trace}")
    print(f"gpu {len(gpu)}  gpu_ann {len(ann)}->{len(reg)}  cpu_ann {len(cpuann)}  "
          f"cpu_op {len(cpuop)}  cuda_rt(kept) {len(rt)}")
    st = collections.Counter()
    for e in gpu:
        st[e[3]] += e[5]
    print("GPU streams tid -> kernel ms total: " +
          ", ".join(f"{k}={v/1000:.0f}" for k, v in st.most_common(6)))

    corr2launch = {}
    for e in rt:
        if e[6]:
            corr2launch[e[6]] = e

    steps = []
    for k in range(len(reg) - 1):
        steps.append(dict(t0=reg[k][0], tmid=reg[k][1], t1=reg[k + 1][0], name=reg[k][2]))
    kinds = collections.Counter()
    for s in steps:
        m = STEP_RE.search(s["name"])
        kinds[m.groups() if m else ("?",)] += 1
    dom = max(kinds, key=kinds.get)
    sel = [s for s in steps[args.skip:]
           if (STEP_RE.search(s["name"]).groups() if STEP_RE.search(s["name"]) else None) == dom]
    if args.steps:
        sel = sel[:args.steps]
    n = len(sel)
    print(f"analysed steps: {n}   kind ctx{dom[0]}({dom[1]})/gen{dom[2]}({dom[3]})")

    # ---- phase markers inside a decode step
    #  P1 head        : t0 .. first kernel >= 100 us   (scheduler handoff + prepare_inputs)
    #  P2 target body : .. end of the merged gpu annotation
    #  P3 logits+sample: .. the drafter's first _prepare_dflash_inputs_kernel
    #  P4 draft loop  : .. next step
    op_starts = [e[4] for e in cpuop]
    rt_starts = [e[4] for e in rt]
    gpu_starts = [e[4] for e in gpu]
    gpu_iv = sorted((e[4], e[4] + e[5]) for e in gpu)
    gpu_iv_starts = [a for a, _ in gpu_iv]
    ca_starts = [e[4] for e in cpuann]

    wall = sum(s["t1"] - s["t0"] for s in sel) / n
    tgtw = sum(s["tmid"] - s["t0"] for s in sel) / n
    drfw = sum(s["t1"] - s["tmid"] for s in sel) / n

    bycause, bycause_n = collections.Counter(), collections.Counter()
    detail, detail_n = collections.Counter(), collections.Counter()
    small = collections.Counter()
    small_n = collections.Counter()
    hist = collections.Counter()
    hist_n = collections.Counter()
    bycls = collections.Counter()
    bycls_n = collections.Counter()
    bycls_draft = collections.Counter()
    nk = [0, 0.0]
    byph = collections.Counter(); byph_n = collections.Counter()
    byph_big = collections.Counter(); byph_big_n = collections.Counter()
    phw = collections.Counter()
    idle_tot = 0.0
    rows = []
    lag = []
    cpu_interstep = []

    for si, s in enumerate(sel):
        i0 = bisect.bisect_left(gpu_iv_starts, s["t0"] - 60000)
        i1 = bisect.bisect_right(gpu_iv_starts, s["t1"])
        gl, busy = gaps_of(gpu_iv[i0:i1], s["t0"], s["t1"])
        idle_tot += (s["t1"] - s["t0"]) - busy
        # host lag: kernel start - its launch, median over the step
        for e in gpu[bisect.bisect_left(gpu_starts, s["t0"]):
                     bisect.bisect_left(gpu_starts, s["t1"]):37]:
            L = corr2launch.get(e[6])
            if L:
                lag.append(e[4] - L[4])
        for e in gpu[bisect.bisect_left(gpu_starts, s["t0"]):
                     bisect.bisect_left(gpu_starts, s["t1"])]:
            nk[0] += 1
            nk[1] += e[5]
        p1 = s["tmid"]
        p3 = s["t1"]
        for e in gpu[bisect.bisect_left(gpu_starts, s["t0"]):
                     bisect.bisect_left(gpu_starts, s["t1"])]:
            if p1 == s["tmid"] and e[5] >= 100.0 and e[4] < s["tmid"]:
                p1 = e[4]
            if e[4] >= s["tmid"] and "_prepare_dflash_inputs" in e[1]:
                p3 = e[4]
                break
        s["p1"], s["p3"] = p1, p3
        ka = bisect.bisect_right(ca_starts, s["t0"]) - 1
        if 0 <= ka < len(cpuann) - 1:
            cpu_interstep.append(cpuann[ka + 1][4] - (cpuann[ka][4] + cpuann[ka][5]))

        phw["P1 head: sched handoff + prepare_inputs"] += s["p1"] - s["t0"]
        phw["P2 target forward body"] += s["tmid"] - s["p1"]
        phw["P3 logits + sample + accept"] += s["p3"] - s["tmid"]
        phw["P4 draft loop (DFlash2 k=7)"] += s["t1"] - s["p3"]
        for (g0, g1) in gl:
            d = g1 - g0
            in_draft = g0 >= s["tmid"] - 1.0
            i = bisect.bisect_left(gpu_starts, g1 - 0.001)
            nxt = None
            for q in range(i, min(i + 4, len(gpu))):
                if abs(gpu[q][4] - g1) < 1.0:
                    nxt = gpu[q]
                    break
            L = corr2launch.get(nxt[6]) if nxt else None
            host_bound = (L is not None and L[4] >= g0 - 1.0)
            ph = ("P1 head: sched handoff + prepare_inputs" if g0 < s["p1"] else
                  "P2 target forward body" if g0 < s["tmid"] else
                  "P3 logits + sample + accept" if g0 < s["p3"] else
                  "P4 draft loop (DFlash2 k=7)")
            byph[ph] += d
            byph_n[ph] += 1
            if d >= 20.0:
                byph_big[ph] += d
                byph_big_n[ph] += 1
            hb = next(i2 for i2, b in enumerate(HIST) if d <= b)
            hist[hb] += d
            hist_n[hb] += 1
            if nxt is not None:
                kc = kcls(nxt[1], nxt[0])
                bycls[kc] += d
                bycls_n[kc] += 1
                if in_draft:
                    bycls_draft[kc] += d
            if d < MG:
                key = ("draft" if in_draft else "target",
                       "host-bound (launch)" if host_bound else "device wait")
                small[key] += d
                small_n[key] += 1
                continue
            cover = enclosing(cpuop, op_starts, g0, g1)
            last, nxop = neighbours(cpuop, op_starts, g0, g1)
            rl = rt[bisect.bisect_left(rt_starts, g0 - 30000):
                    bisect.bisect_right(rt_starts, g1)]
            ov = [e for e in rl if e[4] + e[5] > g0 and e[4] < g1]
            blocking = [e for e in ov if BLOCKING.search(e[1])]
            nlaunch = sum(1 for e in ov if LAUNCHY.search(e[1]))
            host_busy_rt = sum(e[5] for e in ov)
            in_ann = (0 <= ka < len(cpuann) and cpuann[ka][4] <= g0
                      and cpuann[ka][4] + cpuann[ka][5] >= g1)
            covn = cover[1] if cover else "-"
            nxtn = nxt[1] if nxt else "-"
            prv = None
            p = bisect.bisect_left(gpu_starts, g0)
            for q in range(p - 1, max(p - 60, -1), -1):
                if abs(gpu[q][4] + gpu[q][5] - g0) < 1.0:
                    prv = gpu[q]
                    break

            # ---------------- cause
            if blocking:
                cause = "(b) blocking sync / D2H on critical path"
            elif in_draft and not in_ann:
                cause = "(a) DFlash2 draft orchestration"
            elif not in_ann and cover is None:
                cause = "(c) scheduler / host round-trip"
            elif "all_reduce" in covn or "nccl" in covn.lower() or \
                 (nxt is not None and "nccl" in nxtn.lower() and host_bound):
                cause = "(d) NCCL launch / proxy stall"
            elif host_bound:
                cause = "(e) launch gap inside forward"
            else:
                cause = "(f) device-side wait (queued but blocked)"
            bycause[cause] += d
            bycause_n[cause] += 1
            k2 = (cause, covn, (nxtn.split("(")[0])[:46])
            detail[k2] += d
            detail_n[k2] += 1
            rows.append(dict(step=si, off=(g0 - s["t0"]) / 1000, ms=d / 1000, cause=cause,
                             cover=covn, prev=(prv[1][:64] if prv else "-"),
                             nxt=nxtn[:64], host=host_bound, draft=in_draft,
                             in_ann=in_ann, nlaunch=nlaunch,
                             rt_ms=host_busy_rt / 1000,
                             blocking=[e[1] for e in blocking][:3],
                             last_op=(last[1] if last else "-"),
                             next_op=(nxop[1] if nxop else "-")))

    idle = idle_tot / n
    big = sum(bycause.values()) / n
    sml = sum(small.values()) / n
    lag.sort()
    print(f"\nmean step wall {wall/1000:.3f} ms   (target {tgtw/1000:.3f} / "
          f"draft {drfw/1000:.3f})   GPU idle {idle/1000:.3f} ms ({idle/wall*100:.2f} %)")
    print(f"host->gpu launch lag  p50 {lag[len(lag)//2]:.0f} us  "
          f"p10 {lag[len(lag)//10]:.0f}  p90 {lag[len(lag)*9//10]:.0f} us  (n={len(lag)})")
    if cpu_interstep:
        cpu_interstep.sort()
        print(f"host inter-step window (end of CPU annotation k -> start of k+1): "
              f"p50 {cpu_interstep[len(cpu_interstep)//2]/1000:.3f} ms")
    print(f"gaps >= {args.min_gap} ms: {sum(bycause_n.values())/n:.1f}/step, "
          f"{big/1000:.3f} ms/step ({big/idle*100:.0f} % of idle);  "
          f"sub-threshold {sml/1000:.3f} ms/step "
          f"({sum(small_n.values())/n:.0f} gaps/step, {sml/idle*100:.0f} % of idle)")

    print(f"\n{'cause':<46} {'ms/step':>9} {'%step':>7} {'n/step':>7} {'mean ms':>8}")
    for c, v in bycause.most_common():
        print(f"{c:<46} {v/n/1000:>9.3f} {v/n/wall*100:>6.2f}% "
              f"{bycause_n[c]/n:>7.1f} {v/bycause_n[c]/1000:>8.3f}")
    for k, v in small.most_common():
        lbl = f"(g) sub-{args.min_gap}ms {k[0]}, {k[1]}"
        print(f"{lbl:<46} {v/n/1000:>9.3f} {v/n/wall*100:>6.2f}% "
              f"{small_n[k]/n:>7.0f} {v/small_n[k]:>8.1f}us")
    print(f"{'TOTAL GPU idle':<46} {idle/1000:>9.3f} {idle/wall*100:>6.2f}%")

    print(f"\n-- top gap sites --\n{'ms/step':>8} {'n/step':>7}  cause | enclosing cpu_op | next kernel")
    for k, v in detail.most_common(24):
        c, cov, nx = k
        print(f"{v/n/1000:>8.3f} {detail_n[k]/n:>7.2f}  {c}")
        print(f"{'':>17}cpu_op = {cov}")
        print(f"{'':>17}next   = {nx}")

    print(f"\n-- GPU idle by phase of the step --")
    print(f"{'phase':<42} {'phase ms':>9} {'idle ms':>9} {'%idle':>7} {'occup':>7} "
          f"{'gaps':>7} {'>=20us ms':>10} {'n':>5}")
    for k in ["P1 head: sched handoff + prepare_inputs", "P2 target forward body",
              "P3 logits + sample + accept", "P4 draft loop (DFlash2 k=7)"]:
        w = phw[k] / n
        print(f"{k:<42} {w/1000:>9.3f} {byph[k]/n/1000:>9.3f} "
              f"{byph[k]/idle_tot*100:>6.1f}% {(1-byph[k]/phw[k])*100:>6.1f}% "
              f"{byph_n[k]/n:>7.0f} {byph_big[k]/n/1000:>10.3f} {byph_big_n[k]/n:>5.1f}")

    print(f"\n-- gap size histogram (ALL gaps) --")
    print(f"{'bucket':>14} {'gaps/step':>10} {'ms/step':>9} {'%idle':>7}")
    lo = 0.0
    for i2, b in enumerate(HIST):
        if hist_n[i2]:
            hi2 = "inf" if b > 1e8 else f"{b:g}"
            print(f"{lo:>6g}-{hi2:>7} us {hist_n[i2]/n:>10.0f} {hist[i2]/n/1000:>9.3f} "
                  f"{hist[i2]/idle_tot*100:>6.1f}%")
        lo = b
    print(f"kernels+memcpy per step: {nk[0]/n:.0f}   mean duration "
          f"{nk[1]/nk[0]:.1f} us   -> gaps/step {sum(hist_n.values())/n:.0f} "
          f"({sum(hist_n.values())/nk[0]*100:.0f} % of kernel boundaries)")

    print(f"\n-- idle attributed to the kernel that FOLLOWS the gap (class) --")
    print(f"{'class of next kernel':<28} {'ms/step':>9} {'%idle':>7} {'gaps/step':>10} "
          f"{'mean us':>8} {'draft ms':>9}")
    for k, v in bycls.most_common(14):
        print(f"{k:<28} {v/n/1000:>9.3f} {v/idle_tot*100:>6.1f}% {bycls_n[k]/n:>10.0f} "
              f"{v/bycls_n[k]:>8.2f} {bycls_draft[k]/n/1000:>9.3f}")

    # ---- CUDA-graph accounting
    graphable = sum(v for k, v in small.items() if k[1].startswith("host")) \
        + bycause["(e) launch gap inside forward"] \
        + bycause["(a) DFlash2 draft orchestration"] \
        + bycause["(d) NCCL launch / proxy stall"]
    print(f"\nCUDA-graph accounting (what a FULL capture could remove = host-bound "
          f"launch gaps only):")
    print(f"  removable-in-principle {graphable/n/1000:.3f} ms/step "
          f"({graphable/n/wall*100:.2f} % of step)")
    print(f"  NOT removable          {(idle-graphable/n)/1000:.3f} ms/step "
          f"(blocking syncs, host round-trip, device waits)")

    if args.dump_steps:
        print(f"\n-- per-gap dump, first {args.dump_steps} analysed steps --")
        for r in rows:
            if r["step"] >= args.dump_steps:
                break
            print(f"s{r['step']:<3}+{r['off']:>7.3f}ms {r['ms']:>6.3f}ms "
                  f"{'HOST' if r['host'] else 'DEV '} "
                  f"{'DRAFT' if r['draft'] else 'TGT  '} "
                  f"{'inAnn' if r['in_ann'] else 'outAnn'} "
                  f"launches={r['nlaunch']:<4} rt={r['rt_ms']:.3f}ms  {r['cause']}")
            print(f"      cpu_op={r['cover']}  block={r['blocking']}")
            print(f"      lastop={r['last_op']}  nextop={r['next_op']}")
            print(f"      prev={r['prev']}")
            print(f"      next={r['nxt']}")

    if args.json:
        json.dump(dict(label=args.label, trace=args.trace, steps=n, wall_ms=wall / 1000,
                       target_ms=tgtw / 1000, draft_ms=drfw / 1000,
                       idle_ms=idle / 1000,
                       causes={c: [bycause[c] / n / 1000, bycause_n[c] / n] for c in bycause},
                       small={f"{k[0]}/{k[1]}": [small[k] / n / 1000, small_n[k] / n]
                              for k in small},
                       graphable_ms=graphable / n / 1000,
                       top=[[list(k), v / n / 1000, detail_n[k] / n]
                            for k, v in detail.most_common(40)],
                       rows=rows[:600]), open(args.json, "w"), indent=1)
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()

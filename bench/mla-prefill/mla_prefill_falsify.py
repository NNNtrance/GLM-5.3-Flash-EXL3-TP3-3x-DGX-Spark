"""Falsify (or confirm) the MLA-prefill ceiling correction of cuda-exl3 5fd7299 on GB10.

The claim under test, from Zeuss5/cuda-exl3 issue #5 (2026-09-06 08:10 UTC):

    "run the production arm on a GB10 at 262 K and it should land within a few
     percent of the drifting arm, not between drifting and independent"

because what has to fit in L2 is the union over a key's *residence window*
(~4,096 keys ~= 4.5 MiB at 7.4 % turnover), not the union over the chunk
(187 MiB at 262 K = 7.8x a 24 MiB L2).

Arms and their generator are copied verbatim from bench/bench_mla_prefill.py at
5fd7299 so that this is the same fixture, not a re-implementation of it.
Differences from his run, all additive:
  * rows fixed at 1,792 (his ctx_sweep default), 3 rounds median, CUDA events;
  * two head counts: 16 (his, 64 heads at TP=4) and 22 (ours, 66 at TP=3);
  * the ruler is this house's cold rotating-bank bf16 read at 512x32768
    (expected 225-246 GB/s on GB10) as well as his streaming-sum ruler;
  * a correctness check of the kernel against a torch reference before timing,
    because a kernel that bails early would give meaningless microseconds.

MODES (no arguments == the timing bench exactly as first published, so run.sh
is unaffected by anything added here):

  python3 mla_prefill_falsify.py                 timing bench (run.sh's path)
  python3 mla_prefill_falsify.py --verify        ... plus an inline correctness
                                                 check inside each H=22 timing
                                                 cell, so a run verifies what
                                                 it times
  python3 mla_prefill_falsify.py --check         correctness matrix only, at
                                                 the real TP=3 shape
  python3 mla_prefill_falsify.py --self-test     CPU only, no CUDA, no kernel:
                                                 proves the reference path and
                                                 the argument parsing
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

# ---- his constants, unchanged ------------------------------------------------
D = 576
DV = 512
TOPK = 2048                 # GLM index_topk
DRIFT = 2                   # a near-static selection: the cache-friendly bound
PROD_TURNOVER = 0.074       # 1 - 0.926, from our own DSA-indexer capture
ROWS = 1792                 # our steady prefill chunk
CTXS = (32768, 71680, 262144)
HEADS = (16, 22)            # 16 = his (TP=4), 22 = ours (TP=3)
ROUNDS = 3
REPS = 8

# --- our production index/kpool settings, for the correctness matrix ---
KPOOL = 4                   # DSA indexer kpool granularity in production
CHECK_CTXS = (32768, 262144)
CHECK_ARMS = ("drifting", "production")

dev = "cuda"
P = None
L2 = None


def init_device(need_kernel=True):
    """Bind CUDA-dependent globals. Deferred so --self-test runs without a GPU."""
    global P, L2
    if need_kernel:
        from cuda_exl3 import ops as _ops
        _ops._try_native()
        globals()["_ops"] = _ops
    P = torch.cuda.get_device_properties(0)
    L2 = P.L2_cache_size


def mla(q, kv, sel, sl, scale):
    return torch.ops.cuda_exl3_C.mla_decode(q, kv, sel, sl, scale, DV, 64, 1, 1.0)


# ---- the correctness reference, and the check on the reference itself --------
def ref_mla(q, kv, sel, seqlens, scale, dv=DV, chunk=8):
    """Chunked torch MLA reference.

    q (r,h,D) · kv (ctx,D) · sel (r,tk) · seqlens (r,) or None.
    Chunked over rows because the gathered tensor at the real shape
    (1792 x 2048 x 576) would be 8.5 GB in fp32; the chunking is what
    --self-test proves equal to the unchunked form.
    """
    r, h, _ = q.shape
    tk = sel.shape[1]
    out = torch.empty(r, h, dv, device=q.device, dtype=torch.float32)
    ar = torch.arange(tk, device=q.device)
    for i in range(0, r, chunk):
        j = min(i + chunk, r)
        g = kv[sel[i:j].long()].float()                     # c, tk, D
        logits = torch.einsum("rhd,rkd->rhk", q[i:j].float(), g) * scale
        if seqlens is not None:
            mask = ar[None, :] >= seqlens[i:j, None].to(ar.dtype)
            logits = logits.masked_fill(mask[:, None, :], float("-inf"))
        out[i:j] = torch.einsum("rhk,rkv->rhv", logits.softmax(-1),
                                g[:, :, :dv])
    return out


def compare(got, ref):
    """max abs / scale-normalised max rel / cosine per (row,head) vector."""
    got = got.float().reshape(ref.shape)
    diff = (got - ref).abs()
    scale = ref.abs().max().item() + 1e-9
    cos = F.cosine_similarity(got.reshape(-1, ref.shape[-1]),
                              ref.reshape(-1, ref.shape[-1]), dim=-1)
    return dict(max_abs=diff.max().item(),
                max_rel=diff.max().item() / scale,
                cos_min=cos.min().item(),
                cos_mean=cos.mean().item(),
                finite=bool(torch.isfinite(got).all()))


# ---- rulers ------------------------------------------------------------------
def ruler_sum():
    """Streaming bf16 read, his ruler shrunk to fit our 2 GiB footprint budget."""
    a = torch.empty(1 << 27, dtype=torch.bfloat16, device=dev)   # 256 MiB
    a.normal_()
    for _ in range(3):
        a.sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        a.sum()
    torch.cuda.synchronize()
    gbs = a.numel() * 2 * 20 / (time.perf_counter() - t0) / 1e9
    del a
    torch.cuda.empty_cache()
    return gbs


def ruler_house():
    """Cold rotating-bank bf16 GEMM read at 512x32768, the shape our other
    GB10 benches use as the ruler (225-246 GB/s measured on this part)."""
    k, n = 512, 32768
    wb = k * n * 2
    N = max(1, int(math.ceil(4 * L2 / wb)))
    ws = [torch.randn(n, k, dtype=torch.bfloat16, device=dev) for _ in range(N)]
    out = {}
    for m in (8, 64):
        x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        by = wb + m * k * 2 + m * n * 2
        for _ in range(20):
            F.linear(x, ws[0])
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for j in range(200):
                F.linear(x, ws[j % N])
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / 200)
        us = statistics.median(ts)
        out[m] = by / us / 1e3
        del x
    del ws
    torch.cuda.empty_cache()
    return out, N


# ---- his selection generator, verbatim (device/align are additive) -----------
def selections(rows, kind, ctx, device=None, align=1):
    """align > 1 draws on a pool-aligned grid: our production indexer emits
    kpool-granular selections (KPOOL = 4). align == 1 is his generator exactly."""
    device = device or dev
    n = ctx // align

    def rint(shape):
        t = torch.randint(0, n, shape, device=device, dtype=torch.int32)
        return t * align if align > 1 else t

    def rperm(k):
        t = torch.randperm(n, device=device)[:k].int()
        return t * align if align > 1 else t

    if kind == "independent":
        return rint((rows, TOPK))
    if kind == "production":
        sel = torch.empty((rows, TOPK), device=device, dtype=torch.int32)
        cur = rperm(TOPK)
        sel[0] = cur
        for i in range(1, rows):
            m = torch.rand(TOPK, device=device) < PROD_TURNOVER
            cur = cur.clone()
            cur[m] = rint((int(m.sum()),))
            sel[i] = cur
        return sel
    base = rperm(TOPK)
    sel = base.repeat(rows, 1)
    if DRIFT:
        pos = torch.randint(0, TOPK, (rows, DRIFT), device=device)
        new = rint((rows, DRIFT))
        for r in range(1, rows):
            sel[r:, pos[r]] = new[r]
    return sel


def overlap_stats(sel, n=64):
    """Adjacent-row min-normalised overlap, so the fixture is shown to carry the
    turnover it claims -- the same statistic our indexer capture reported."""
    rs = torch.randint(1, sel.shape[0], (n,))
    ov = []
    for r in rs.tolist():
        a = torch.unique(sel[r - 1])
        b = torch.unique(sel[r])
        ov.append(torch.isin(b, a).sum().item() / min(a.numel(), b.numel()))
    return statistics.median(ov)


def timeit(f, reps=REPS):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        f()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000            # us


def sample_rows(rows, n, device):
    """Strided sample across the whole chunk -- early and late rows differ in
    the drifting/production arms, so verifying only the front would miss drift."""
    if n <= 0 or n >= rows:
        return torch.arange(rows, device=device)
    return torch.linspace(0, rows - 1, n, device=device).round().long().unique()


def verify_cell(q, kv, sel, sl, n_rows, chunk):
    """Check the kernel against the reference on a sample of the rows it just
    computed -- same inputs, same call, no re-timing."""
    scale = 1.0 / (D ** 0.5)
    got = mla(q, kv, sel, sl, scale)
    idx = sample_rows(q.shape[0], n_rows, q.device)
    ref = ref_mla(q[idx], kv, sel[idx], sl[idx], scale, chunk=chunk)
    r = compare(got.reshape(q.shape[0], q.shape[1], DV)[idx], ref)
    r["rows_checked"] = int(idx.numel())
    del got, ref
    return r


# ---- instrument check: is the kernel actually computing MLA? ------------------
def correctness():
    """The small-shape smoke test the first published run used (h=2)."""
    r, h, tk, ctx = 4, 2, 16, 1024
    q = torch.randn(r, h, D, device=dev, dtype=torch.bfloat16) * 0.05
    kv = torch.randn(ctx, D, device=dev, dtype=torch.bfloat16) * 0.05
    sel = torch.randint(0, ctx, (r, tk), device=dev, dtype=torch.int32)
    sl = torch.full((r,), tk, device=dev, dtype=torch.int32)
    scale = 1.0 / (D ** 0.5)
    got = mla(q, kv, sel, sl, scale).float()
    ref = ref_mla(q, kv, sel, sl, scale, chunk=4)
    shape, finite = tuple(got.shape), bool(torch.isfinite(got).all())
    rel = compare(got, ref)["max_rel"] if got.numel() == ref.numel() else float("nan")
    del q, kv, sel, sl, got, ref
    torch.cuda.empty_cache()
    return rel, shape, finite


# ---- the correctness matrix at the real TP=3 shape ---------------------------
def check_matrix(args):
    """mla_decode vs the torch reference at the shape production actually runs.

    The kernel is called at the full 1,792-row chunk; the reference verifies a
    strided sample of those rows (--check-rows), because the reference is ~200x
    the kernel's cost and sampling does not change what the kernel computed.
    """
    print(f"# {P.name} SMs={P.multi_processor_count} L2={L2/2**20:.1f} MiB "
          f"torch={torch.__version__}")
    print(f"# CORRECTNESS MATRIX at the production shape: rows={args.rows} "
          f"head_dim={D} v_head_dim={DV} topk={TOPK} kpool_align={args.align} "
          f"ragged={args.ragged}")
    print(f"# reference: chunked fp32 torch MLA, chunk={args.ref_chunk} rows; "
          f"verifying {args.check_rows or 'all'} rows per cell (strided)")
    print()
    print(f"{'ctx':>8s} {'H':>3s} {'arm':>12s} {'rows':>5s} {'max abs':>10s} "
          f"{'max rel':>10s} {'cos min':>10s} {'cos mean':>10s} {'finite':>7s} "
          f"{'verdict':>8s}")
    out, worst = [], 0.0
    for ctx in args.check_ctxs:
        kv = torch.randn(ctx, D, device=dev, dtype=torch.bfloat16) * 0.05
        for arm in args.check_arms:
            sel = selections(args.rows, arm, ctx, align=args.align)
            if args.ragged:
                g = torch.Generator(device="cpu").manual_seed(1234)
                sl = torch.randint(TOPK // 2, TOPK + 1, (args.rows,),
                                   generator=g).to(dev, torch.int32)
            else:
                sl = torch.full((args.rows,), TOPK, device=dev, dtype=torch.int32)
            for h in args.check_heads:
                q = torch.randn(args.rows, h, D, device=dev,
                                dtype=torch.bfloat16) * 0.05
                r = verify_cell(q, kv, sel, sl, args.check_rows, args.ref_chunk)
                ok = r["finite"] and r["max_rel"] < args.tol and \
                    r["cos_min"] > args.cos_tol
                worst = max(worst, r["max_rel"])
                print(f"{ctx:>8d} {h:>3d} {arm:>12s} {r['rows_checked']:>5d} "
                      f"{r['max_abs']:>10.3e} {r['max_rel']:>10.3e} "
                      f"{r['cos_min']:>10.6f} {r['cos_mean']:>10.6f} "
                      f"{str(r['finite']):>7s} {'OK' if ok else 'FAIL':>8s}")
                out.append(dict(ctx=ctx, heads=h, arm=arm, align=args.align,
                                ragged=args.ragged, **r))
                del q
                torch.cuda.empty_cache()
            del sel, sl
        del kv
        torch.cuda.empty_cache()
    allok = all(x["finite"] and x["max_rel"] < args.tol and
                x["cos_min"] > args.cos_tol for x in out)
    print()
    print(f"# worst max_rel {worst:.3e} (tolerance {args.tol:.0e}), "
          f"cosine floor {args.cos_tol}")
    print(f"# MATRIX {'PASSED' if allok else 'FAILED'} "
          f"({sum(1 for x in out if x['max_rel'] < args.tol)}/{len(out)} cells)")
    print(f"# peak GPU allocation: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(dict(device=P.name, mode="check", tol=args.tol,
                           cos_tol=args.cos_tol, rows=args.rows,
                           align=args.align, ragged=args.ragged,
                           passed=allok, cells=out), fh, indent=1)
    return 0 if allok else 1


# ---- CPU-only proof that the reference path and the parsing are sound --------
def self_test():
    """No CUDA, no kernel. Proves the parts that would otherwise only ever be
    exercised for the first time on a GPU with the lock held."""
    cpu = "cpu"
    torch.manual_seed(0)
    fails = []

    def check(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)

    print("## 1. argument parsing (no args must equal the published bench)")
    a0 = parse_args([])
    check("no args -> timing mode", not a0.check and not a0.self_test and not a0.verify)
    check("no args -> defaults intact",
          (a0.rows, a0.align, a0.ragged) == (ROWS, 1, False),
          f"rows={a0.rows} align={a0.align} ragged={a0.ragged}")
    a1 = parse_args(["--check", "--align", "4", "--ragged", "--check-rows", "64",
                     "--check-ctxs", "32768", "--check-heads", "22",
                     "--check-arms", "production", "--tol", "1e-2"])
    check("--check parses", a1.check and a1.align == 4 and a1.ragged and
          a1.check_rows == 64 and a1.check_ctxs == [32768] and
          a1.check_heads == [22] and a1.check_arms == ["production"] and
          abs(a1.tol - 1e-2) < 1e-12)
    a2 = parse_args(["--verify", "--verify-rows", "8"])
    check("--verify parses", a2.verify and a2.verify_rows == 8)

    print("## 2. reference: chunked == unchunked (the chunking is the risk)")
    r, h, tk, ctx, dv = 7, 3, 12, 64, 5
    q = torch.randn(r, h, D, device=cpu, dtype=torch.bfloat16) * 0.05
    kv = torch.randn(ctx, D, device=cpu, dtype=torch.bfloat16) * 0.05
    sel = torch.randint(0, ctx, (r, tk), device=cpu, dtype=torch.int32)
    sl = torch.full((r,), tk, device=cpu, dtype=torch.int32)
    sc = 1.0 / (D ** 0.5)
    base = ref_mla(q, kv, sel, sl, sc, dv=dv, chunk=r)          # single shot
    for c in (1, 2, 3, 5, 100):
        d = (ref_mla(q, kv, sel, sl, sc, dv=dv, chunk=c) - base).abs().max().item()
        check(f"chunk={c} matches single-shot", d == 0.0, f"maxdiff {d:.2e}")

    print("## 3. reference: masking semantics")
    full = ref_mla(q, kv, sel, None, sc, dv=dv, chunk=3)
    d = (full - base).abs().max().item()
    check("seqlens==topk equals no-mask", d < 1e-6, f"maxdiff {d:.2e}")
    short = torch.full((r,), tk // 2, device=cpu, dtype=torch.int32)
    a = ref_mla(q, kv, sel, short, sc, dv=dv, chunk=3)
    b = ref_mla(q, kv[:, :], sel[:, :tk // 2], None, sc, dv=dv, chunk=3)
    d = (a - b).abs().max().item()
    check("ragged seqlens == truncating the selection", d < 1e-5, f"maxdiff {d:.2e}")
    check("masked rows stay finite", bool(torch.isfinite(a).all()))

    print("## 4. compare(): identity, and it notices a real error")
    ident = compare(base.to(torch.bfloat16), base)
    check("identical -> cos 1, rel small",
          ident["cos_min"] > 0.999 and ident["max_rel"] < 5e-2,
          f"cos_min={ident['cos_min']:.6f} rel={ident['max_rel']:.2e}")
    bad = base.clone()
    bad[0, 0] = -bad[0, 0]
    wrong = compare(bad, base)
    check("sign-flipped row -> cos_min < 0", wrong["cos_min"] < 0,
          f"cos_min={wrong['cos_min']:.4f}")
    check("compare() flags non-finite",
          compare(torch.full_like(base, float('nan')), base)["finite"] is False)

    print("## 5. selection generators (shape, dtype, range, alignment, overlap)")
    for arm, lo, hi in (("drifting", 0.9, 1.01), ("production", 0.85, 0.99),
                        ("independent", 0.0, 0.3)):
        s = selections(96, arm, 65536, device=cpu)
        ok = (s.shape == (96, TOPK) and s.dtype == torch.int32 and
              int(s.min()) >= 0 and int(s.max()) < 65536)
        check(f"{arm}: shape/dtype/range", ok, f"{tuple(s.shape)} {s.dtype}")
        ov = overlap_stats(s, n=16)
        check(f"{arm}: overlap {ov:.3f} in [{lo}, {hi})", lo <= ov < hi)
    sa = selections(32, "production", 65536, device=cpu, align=KPOOL)
    check(f"align={KPOOL} -> every index is a multiple of {KPOOL}",
          int((sa % KPOOL).abs().max()) == 0)
    check(f"align={KPOOL} -> still in range", int(sa.max()) < 65536)

    print("## 6. row sampling is strided and in range")
    idx = sample_rows(ROWS, 256, cpu)
    check("256 of 1792, strided, unique, in range",
          idx.numel() == 256 and int(idx.min()) == 0 and
          int(idx.max()) == ROWS - 1 and idx.unique().numel() == 256)
    check("check_rows=0 -> all rows", sample_rows(ROWS, 0, cpu).numel() == ROWS)

    print()
    if fails:
        print(f"SELF-TEST FAILED: {len(fails)} check(s): {fails}")
        return 1
    print("SELF-TEST PASSED -- reference path, masking, comparison, generators "
          "and parsing are sound; only the kernel call is unexercised (needs a GPU).")
    return 0


def main(args):
    print(f"# {P.name} cc={P.major}.{P.minor} SMs={P.multi_processor_count} "
          f"L2={L2/2**20:.1f} MiB torch={torch.__version__} "
          f"backend={_ops.backend()}")
    free, total = torch.cuda.mem_get_info()
    print(f"# device memory free {free/2**30:.1f} GiB of {total/2**30:.1f} GiB")
    print(f"# rows={ROWS} head_dim={D} v_head_dim={DV} topk={TOPK} "
          f"drift={DRIFT} turnover={PROD_TURNOVER} rounds={ROUNDS} reps={REPS}")

    rel, oshape, finite = correctness()
    ok = finite and rel == rel and rel < 5e-2
    print(f"# instrument check: mla_decode out{oshape} finite={finite}, "
          f"vs torch reference max rel err {rel:.2e} -> "
          f"{'OK' if ok else 'WARN (timings still reported, flagged)'}")

    gsum = ruler_sum()
    ghouse, nbank = ruler_house()
    print(f"# ruler A (streaming bf16 sum, 256 MiB): {gsum:.0f} GB/s")
    print(f"# ruler B (cold {nbank}-copy bank, F.linear 512x32768): "
          f"m=8 {ghouse[8]:.0f} GB/s, m=64 {ghouse[64]:.0f} GB/s "
          f"[house band 225-246]")
    band = 225 <= ghouse[64] <= 260
    print(f"# ruler verdict: {'in band' if band else 'OUT OF BAND -- suspect'}")
    if args.verify:
        print(f"# --verify: inline correctness check on {args.verify_rows} rows "
              f"of every H={args.verify_heads} timing cell")

    gbs = gsum
    rows = ROWS
    print()
    print(f"{'ctx':>8s} {'latMB':>6s} {'H':>3s} {'arm':>12s} {'us':>9s} "
          f"{'ws MiB':>8s} {'ws/L2':>6s} {'perrow GB/s':>12s} {'%ruler':>7s} "
          f"{'union GB/s':>11s} {'ovlp':>6s}")
    rec = []
    for ctx in CTXS:
        kv = torch.randn(ctx, D, device=dev, dtype=torch.bfloat16) * 0.05
        qs = {h: torch.randn(rows, h, D, device=dev, dtype=torch.bfloat16) * 0.05
              for h in HEADS}
        sl = torch.full((rows,), TOPK, device=dev, dtype=torch.int32)
        for kind in ("drifting", "production", "independent"):
            sel = selections(rows, kind, ctx)
            ws = int(sel.unique().numel()) * D * 2
            ov = overlap_stats(sel)
            for h in HEADS:
                q = qs[h]
                f = lambda: mla(q, kv, sel, sl, 1.0 / (D ** 0.5))
                us = statistics.median([timeit(f) for _ in range(ROUNDS)])
                per_row = rows * TOPK * D * 2
                print(f"{ctx:>8d} {ctx*D*2/1e6:>6.0f} {h:>3d} {kind:>12s} "
                      f"{us:>9.1f} {ws/2**20:>8.0f} {ws/L2:>6.2f} "
                      f"{per_row/us/1e3:>12.0f} "
                      f"{per_row/us/1e3/gbs*100:>6.0f}% "
                      f"{ws/us/1e3:>11.1f} {ov:>6.3f}")
                entry = dict(ctx=ctx, heads=h, arm=kind, us=us,
                             ws_bytes=ws, overlap=ov)
                if args.verify and h == args.verify_heads:
                    v = verify_cell(q, kv, sel, sl, args.verify_rows,
                                    args.ref_chunk)
                    vok = v["finite"] and v["max_rel"] < args.tol and \
                        v["cos_min"] > args.cos_tol
                    print(f"{'':>8s} {'':>6s} {h:>3d} {'  verify':>12s} "
                          f"{v['rows_checked']:>4d} rows  max_rel "
                          f"{v['max_rel']:.3e}  cos_min {v['cos_min']:.6f}  "
                          f"{'OK' if vok else 'FAIL'}")
                    entry["verify"] = v
                    torch.cuda.empty_cache()
                rec.append(entry)
            del sel
        del kv, qs, sl
        torch.cuda.empty_cache()

    print()
    print("## Ratios against the drifting arm (his prediction: production "
          "within a few percent at 262K)")
    print(f"{'ctx':>8s} {'H':>3s} {'drift us':>9s} {'prod us':>9s} "
          f"{'indep us':>9s} {'prod/drift':>11s} {'indep/drift':>12s}")
    ratios = []
    for ctx in CTXS:
        for h in HEADS:
            g = {r["arm"]: r["us"] for r in rec
                 if r["ctx"] == ctx and r["heads"] == h}
            pr, ir = g["production"] / g["drifting"], g["independent"] / g["drifting"]
            print(f"{ctx:>8d} {h:>3d} {g['drifting']:>9.1f} "
                  f"{g['production']:>9.1f} {g['independent']:>9.1f} "
                  f"{pr:>11.3f} {ir:>12.3f}")
            ratios.append(dict(ctx=ctx, heads=h, prod_over_drift=pr,
                               indep_over_drift=ir))

    print()
    print("## Verdict")
    for h in HEADS:
        rr = [x for x in ratios if x["ctx"] == 262144 and x["heads"] == h][0]
        excess = (rr["prod_over_drift"] - 1) * 100
        mid = (1 + rr["indep_over_drift"]) / 2
        held = rr["prod_over_drift"] <= 1.05
        print(f"H={h}: at 262K production is {excess:+.1f} % of drifting "
              f"(independent {(rr['indep_over_drift']-1)*100:+.1f} %); "
              f"midpoint arm would be {(mid-1)*100:+.1f} % -> "
              f"prediction {'HELD' if held else 'FAILED'}")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\n# peak GPU allocation this process: {peak:.2f} GiB "
          f"(budget 2 GiB) -> {'within' if peak < 2 else 'OVER'}")

    out = dict(device=P.name, sms=P.multi_processor_count, l2=L2,
               ruler_sum_gbs=gsum, ruler_house=ghouse, relerr=rel,
               out_shape=list(oshape), finite=finite, peak_gib=peak,
               rows=rows, records=rec, ratios=ratios)
    with open(os.environ.get("OUT_JSON", "/out/result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=True, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="correctness matrix at the real TP=3 shape, no timing")
    p.add_argument("--self-test", action="store_true", dest="self_test",
                   help="CPU only: prove the reference path and the parsing")
    p.add_argument("--verify", action="store_true",
                   help="inline correctness check inside the timing cells")
    p.add_argument("--verify-heads", type=int, default=22,
                   help="which head count the inline check covers (default 22)")
    p.add_argument("--verify-rows", type=int, default=64,
                   help="rows sampled for the inline check (default 64)")
    p.add_argument("--rows", type=int, default=ROWS)
    p.add_argument("--check-rows", type=int, default=256,
                   help="rows verified per matrix cell, strided; 0 = all")
    p.add_argument("--check-ctxs", type=int, nargs="+", default=list(CHECK_CTXS))
    p.add_argument("--check-heads", type=int, nargs="+", default=list(HEADS))
    p.add_argument("--check-arms", nargs="+", default=list(CHECK_ARMS),
                   choices=["drifting", "production", "independent"])
    p.add_argument("--align", type=int, default=1,
                   help=f"kpool granularity for selections (production: {KPOOL})")
    p.add_argument("--ragged", action="store_true",
                   help="per-row seqlens below topk, exercising the mask path")
    p.add_argument("--ref-chunk", type=int, default=8,
                   help="rows per reference chunk (fp32 gather is 4.7 MB/row)")
    p.add_argument("--tol", type=float, default=5e-2,
                   help="max scale-normalised relative error accepted")
    p.add_argument("--cos-tol", type=float, default=0.999,
                   help="minimum per-vector cosine accepted")
    p.add_argument("--out-json", default=os.environ.get("CHECK_JSON", ""))
    return p.parse_args(argv)


if __name__ == "__main__":
    _a = parse_args()
    if _a.self_test:
        sys.exit(self_test())
    torch.manual_seed(0)
    init_device()
    sys.exit(check_matrix(_a) if _a.check else main(_a))

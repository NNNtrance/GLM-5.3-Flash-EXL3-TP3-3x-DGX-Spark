#!/usr/bin/env python3
"""f4987cf padding-row skip: where does the gate belong on a 48-SM part?

The commit gates the skip on grid size >= 2 * SM count, fitted on 188 SMs.
Forced ON / forced OFF at the kernel call, exactly as moe.py's `skip_pad` does:

  ON  : had_in(..., skip_padding=True)  + w13 gemm given sorted_ids
  OFF : had_in(..., skip_padding=False) + w13 gemm given sorted_ids=None

The down projection is identical in both arms (it always carries sorted_ids for
the fused combine). Paired and interleaved rep by rep so both arms see the same
weather; the statistic is the median of the per-rep paired ratio.
"""
import argparse, hashlib, statistics, sys, torch
from cuda_exl3 import ops as _ops
_ops._try_native()
K = torch.ops.cuda_exl3_C
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

H, I_, E_GLOBAL, TP = 4096, 2048, 288, 3
E_LOCAL = E_GLOBAL // TP
TOPK, BITS, CB, LAYERS = 8, 4, 1, 42
DEV = "cuda"
SMS = torch.cuda.get_device_properties(0).multi_processor_count


def block_m_for(rows, num_experts=E_GLOBAL):
    pe = rows / max(num_experts, 1)
    return 16 if pe < 16 else 32 if pe < 48 else (64 if pe < 96 else 128)


def make_weights(seed=12345):
    g = torch.Generator().manual_seed(seed)
    t13 = torch.empty((E_LOCAL, H // 16, 2 * I_ // 16, 16 * BITS), dtype=torch.int16, device=DEV)
    t2 = torch.empty((E_LOCAL, I_ // 16, H // 16, 16 * BITS), dtype=torch.int16, device=DEV)
    for e in range(E_LOCAL):
        t13[e] = torch.randint(-32768, 32767, (H // 16, 2 * I_ // 16, 16 * BITS),
                               dtype=torch.int16, generator=g).to(DEV)
        t2[e] = torch.randint(-32768, 32767, (I_ // 16, H // 16, 16 * BITS),
                              dtype=torch.int16, generator=g).to(DEV)
    return (t13, t2,
            (torch.randn((E_LOCAL, 2, H), generator=g) * 0.05).half().to(DEV),
            (torch.randn((E_LOCAL, 2 * I_), generator=g) * 0.05).half().to(DEV),
            (torch.randn((E_LOCAL, 1, I_), generator=g) * 0.05).half().to(DEV),
            (torch.randn((E_LOCAL, H), generator=g) * 0.05).half().to(DEV))


def make_routing(M, seed):
    g = torch.Generator().manual_seed(seed)
    topk_ids = torch.stack([torch.randperm(E_GLOBAL, generator=g)[:TOPK]
                            for _ in range(M)]).int().to(DEV)
    w = torch.rand((M, TOPK), generator=g, dtype=torch.float32)
    emap = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=DEV)
    emap[:E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32, device=DEV)
    bm = block_m_for(M * TOPK)
    sid, eid, nr = moe_align_block_size(topk_ids, bm, E_GLOBAL, expert_map=emap,
                                        pad_sorted_ids=True)
    sid, eid, nr = sid.int(), eid.int(), nr.int()
    rows = min(eid.numel() * bm, sid.numel())
    eid = eid[: rows // bm]
    return dict(M=M, bm=bm, rows=rows, blocks=eid.numel(), live=int((eid >= 0).sum()),
                x=(torch.randn((M, H), generator=g) * 0.05).bfloat16().to(DEV),
                topk_weights=(w / w.sum(1, keepdim=True)).to(DEV),
                sorted_ids=sid, expert_ids=eid, n_rows=nr)


def make_arm(W, R, skip):
    t13, t2, s13u, s13v, s2u, s2v = W
    M, bm, rows = R["M"], R["bm"], R["rows"]
    a13 = torch.empty((2, rows, H), dtype=torch.half, device=DEV)
    a2 = torch.empty((1, rows, I_), dtype=torch.half, device=DEV)
    sid = R["sorted_ids"]

    def step():
        K.exl3_moe_had_in(R["x"], a13, s13u, sid, R["expert_ids"], R["n_rows"],
                          bm, TOPK, M * TOPK, skip)
        inter = K.exl3_moe_gemm(a13, t13, s13u, s13v, R["expert_ids"], R["n_rows"],
                                [I_, I_], CB, bm, torch.bfloat16,
                                sid if skip else None, None, M, TOPK)
        K.exl3_moe_glu_had_in(inter, a2, s2u, R["expert_ids"], R["n_rows"], bm)
        return K.exl3_moe_gemm(a2, t2, s2u, s2v, R["expert_ids"], R["n_rows"], [H],
                               CB, bm, torch.bfloat16, sid, R["topk_weights"], M, TOPK)
    return step


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); hold = {}
    with torch.cuda.graph(g): hold["o"] = fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    return g, hold["o"]


def paired(runners, reps):
    names = list(runners)
    for n in names:
        for _ in range(8): runners[n]()
    torch.cuda.synchronize()
    ts = {n: [] for n in names}
    for i in range(reps):
        for n in (names if i % 2 == 0 else names[::-1]):
            b, e = torch.cuda.Event(True), torch.cuda.Event(True)
            b.record(); runners[n](); e.record(); torch.cuda.synchronize()
            ts[n].append(b.elapsed_time(e))
    return ts


def h(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,2,4,8,16,32,64,256,2048")
    ap.add_argument("--seeds", default="1234,7,99,2026,31337")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    has_skip = "skip_padding" in str(K.exl3_moe_had_in.default._schema)
    print(f"# {p.name} SMs={SMS}  skip_padding-in-signature={has_skip}", flush=True)
    if not has_skip:
        sys.exit("this build is not f4987cf (no skip_padding arg)")
    print(f"# hidden={H} inter={I_} E_global={E_GLOBAL} E_local={E_LOCAL} top_k={TOPK} "
          f"bits={BITS} cb={CB} layers={LAYERS} reps={a.reps} seeds={a.seeds}", flush=True)
    print(f"# gate in f4987cf: (2*I//128)*max(rows//bm,1) >= 2*SMs  ->  "
          f"{2*I_//128}*blocks >= {2*SMS}  ->  blocks >= {-(-2*SMS//(2*I_//128))}", flush=True)
    W = make_weights()
    print(f"\n{'M':>6} {'bm':>4} {'blks':>5} {'rows':>6} {'gate':>5} "
          f"{'OFF_ms':>9} {'ON_ms':>9} {'delta%':>8} {'match':>6}", flush=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    rows_out = []
    for M in [int(x) for x in a.m.split(",")]:
        ratios, offs, ons, meta, ok = [], [], [], None, True
        for sd in seeds:
            R = make_routing(M, sd)
            meta = R
            g_off, o_off = capture(make_arm(W, R, False))
            g_on, o_on = capture(make_arm(W, R, True))
            if h(o_off) != h(o_on): ok = False
            ts = paired({"off": g_off.replay, "on": g_on.replay}, a.reps)
            for x, y in zip(ts["off"], ts["on"]): ratios.append(y / x)
            offs.append(statistics.median(ts["off"])); ons.append(statistics.median(ts["on"]))
            del g_off, g_on, o_off, o_on
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        off = statistics.median(offs) * LAYERS
        on = statistics.median(ons) * LAYERS
        gate = (2 * I_ // 128) * max(meta["rows"] // meta["bm"], 1) >= 2 * SMS
        d = (statistics.median(ratios) - 1.0) * -100.0   # +ve = ON faster
        print(f"{M:>6} {meta['bm']:>4} {meta['blocks']:>5} {meta['rows']:>6} "
              f"{'ON' if gate else 'OFF':>5} {off:>9.3f} {on:>9.3f} {d:>+8.2f} "
              f"{'yes' if ok else 'NO':>6}", flush=True)
        rows_out.append((M, meta['bm'], meta['blocks'], gate, off, on, d, ok))
    print("\n# delta% = paired median speedup of skip-ON over skip-OFF (+ = ON faster)")
    print("# gate = what f4987cf's own predicate decides on this GPU")


main()

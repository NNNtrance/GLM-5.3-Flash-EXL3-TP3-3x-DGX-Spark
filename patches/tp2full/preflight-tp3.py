#!/usr/bin/env python3
"""Model-free TP=3 preflight for GLM-5.3-Flash EXL3 (Zeus/cuda-exl3 stack).

Every divisibility rule that decides whether a launch survives, checked from
the config files alone in about a second. Run it on the sidecar directory
``pad-tp3.py`` produced, before spending an hour on a boot that was going to
die at rank 2's o_proj.

It prints the arithmetic, not just a verdict: per-rank head counts, per-rank
projection widths, how many columns of each padded tensor are zero, and which
tensors are load-bearing for the EP decision. A number you can read is a number
you can argue with.

  ./preflight-tp3.py --model /var/tmp/glm-5.3-flash-tr3-4bpw-tp3 --tp 3
  ./preflight-tp3.py --model ... --draft /var/tmp/dflash2-draft-tp3 --tp 3

Exit code 0 = every check passed. Non-zero = do not launch.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import gcd
from pathlib import Path

HAD_BLOCK = 128

FAILS: list[str] = []
NOTES: list[str] = []


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def bad(msg: str) -> None:
    print(f"  FAIL  {msg}")
    FAILS.append(msg)


def note(msg: str) -> None:
    print(f"  note  {msg}")
    NOTES.append(msg)


def head(title: str) -> None:
    print(f"\n== {title}")


def divides(name: str, n: int, tp: int) -> bool:
    if n % tp:
        bad(f"{name} = {n} is not divisible by tp={tp} (remainder {n % tp})")
        return False
    ok(f"{name} = {n} -> {n // tp} per rank")
    return True


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


# ---------------------------------------------------------------------------

def check_sidecar(model: Path) -> dict:
    head(f"sidecar directory {model}")
    cfg_path = model / "config.json"
    if not cfg_path.is_file():
        bad(f"no config.json in {model}")
        return {}
    if cfg_path.is_symlink():
        bad("config.json is a symlink -- the sidecar must own its own config")
    else:
        ok("config.json is a real file (the original is untouched)")

    marker = model / ".harem-tp3-sidecar"
    if marker.is_file():
        ok(f"sidecar marker: {marker.read_text().strip()}")
    else:
        note("no .harem-tp3-sidecar marker; not built by pad-tp3.py")

    dangling = [p.name for p in model.iterdir()
                if p.is_symlink() and not p.exists()]
    if dangling:
        bad(f"{len(dangling)} dangling symlink(s), e.g. {dangling[:3]} -- "
            "inside the container the source directory must be mounted too, at a "
            "path that keeps the same relative position")
    else:
        ok("every symlink resolves on this host")

    for needed in ("quantization_config.json", "model.safetensors.index.json"):
        if (model / needed).exists():
            ok(f"{needed} present (cuda-exl3 reads it from _name_or_path)")
        else:
            bad(f"{needed} missing from the sidecar")
    return json.loads(cfg_path.read_text())


def check_attention(t: dict, tp: int) -> None:
    head("MLA attention (11 of 45 layers)")
    h = int(t["num_attention_heads"])
    stock = 64
    if h % tp:
        bad(f"num_attention_heads = {h} is not divisible by tp={tp}; "
            f"Glm5NextModel asserts on this. Run pad-tp3.py.")
        return
    per = h // tp
    real_last = stock - (tp - 1) * per
    ok(f"num_attention_heads = {h} -> {per} per rank")
    if h != stock:
        if real_last <= 0:
            bad(f"pad {stock} -> {h}: rank {tp - 1} owns {per} heads, 0 of them real. "
                f"That rank computes pure padding and the model answers nonsense.")
        else:
            ok(f"pad {stock} -> {h}: rank {tp - 1} holds {real_last}/{per} real heads "
               f"({per - real_last} zero)")
    divides("num_key_value_heads", int(t["num_key_value_heads"]), tp)

    nope = int(t["qk_nope_head_dim"])
    rope = int(t["qk_rope_head_dim"])
    v = int(t["v_head_dim"])
    qk = nope + rope
    ok(f"q_b_proj  out {h} x {qk} = {h * qk} -> {h * qk // tp} per rank")
    ok(f"kv_b_proj out {h} x ({nope}+{v}) = {h * (nope + v)} -> "
       f"{h * (nope + v) // tp} per rank")
    ok(f"o_proj    in  {h} x {v} = {h * v} -> {h * v // tp} per rank (row-parallel)")

    rank = int(t["kv_lora_rank"])
    head_size = rank + rope
    if (head_size, rank) in ((576, 512), (512, 512)):
        ok(f"attention head_size = kv_lora_rank + qk_rope = {head_size}, "
           f"kv_lora_rank = {rank}: in cuda-exl3's SUPPORTED_SHAPES")
    else:
        bad(f"(head_size, kv_lora_rank) = ({head_size}, {rank}) is not one of "
            f"cuda-exl3's ((576,512),(512,512)); --attention-backend CUSTOM refuses it")
    note(f"cuda-exl3's sparse-MLA decode tiles heads 16 at a time with an "
         f"'h0+tid < H' guard, so H={h // tp} runs as {h // tp}. No kernel-side "
         f"22->32 pad (that is a FlashInfer SM120 constraint, not this backend's). "
         f"Verified H=21/22/23 at 2.6e-3 in 07-mla-tp3-headcount-and-prefill.txt")

    idx_h = t.get("index_n_heads")
    if idx_h is not None:
        note(f"DSA indexer index_n_heads={idx_h}: wq_b is ReplicatedLinear and "
             f"wk_weights_proj is disable_tp=True, so it is never sharded -- "
             f"tp={tp} imposes nothing on it")


def check_kda(t: dict, tp: int) -> None:
    head("KDA linear attention (34 of 45 layers)")
    lac = t.get("linear_attn_config") or {}
    nh = int(lac.get("num_heads") or t.get("linear_num_heads", 0))
    hd = int(lac.get("head_dim") or t.get("linear_head_dim", 0))
    conv = int(lac.get("short_conv_kernel_size") or t.get("linear_conv_kernel_dim", 0))
    if not nh or not hd:
        bad("cannot read KDA head config (linear_attn_config / linear_*)")
        return
    flat = t.get("linear_num_heads")
    if flat is not None and int(flat) != nh:
        bad(f"linear_attn_config.num_heads={nh} but linear_num_heads={flat}. "
            f"The loader reads the dict, get_mamba_state_shape_from_config reads "
            f"the flat field: they must agree or the KV/state sizing is wrong.")
    if not divides("linear_attn_config.num_heads", nh, tp):
        return
    proj = hd * nh
    divides(f"KDA projection_size (head_dim {hd} x heads {nh})", proj, tp)
    ok(f"q/k/v_conv1d out {proj} -> {proj // tp}; dt_bias {proj} -> {proj // tp}; "
       f"A_log {nh} -> {nh // tp}; o_proj in {proj} -> {proj // tp}")
    ok(f"in_proj_qkvbfg_a shards: q/k/v {proj} each, b {nh}, "
       f"f_a/g_a {hd} replicated (x{tp} internally), conv kernel {conv}")
    if nh != 64:
        note(f"KDA heads padded 64 -> {nh}: the extra {nh - 64} heads load as zeros "
             f"(A_log too), so their gated-delta state stays zero")


def check_vocab(t: dict, tp: int) -> None:
    head("vocab / embedding / lm_head")
    v = int(t["vocab_size"])
    default_pad = 64
    padded_default = -(-v // default_pad) * default_pad
    # FULL-SCOPE: the unit is lcm(HAD_BLOCK, tp), not lcm(64, tp). A full-scope
    # EXL3 checkpoint quantizes lm_head (head_bits 6), so the vocab pad is a pad
    # on an EXL3 OUTPUT dim, and cuda-exl3 (f3e3090) requires the pad to be whole
    # 128-column Hadamard blocks: real output must end on a block boundary and
    # the padded total must be whole blocks, because the block mixes across its
    # columns before svh is applied. lcm(64, 3) = 192 -> 154944 -> 51648/rank =
    # 403.5 blocks: refused. lcm(128, 3) = 384 -> 155136 -> 51712 = 404 x 128.
    if padded_default % tp == 0 and padded_default % HAD_BLOCK == 0:
        ok(f"vocab {v} pads to {padded_default} at the stock padding_size=64, "
           f"divides tp={tp} and is a whole number of {HAD_BLOCK}-blocks; "
           f"no patch needed")
        return
    bad_default = (f"vocab {v} with the stock padding_size=64 stays {padded_default}, "
                   f"and divide({padded_default}, {tp}) asserts inside "
                   f"VocabParallelEmbedding")
    note(bad_default)
    unit = lcm(HAD_BLOCK, tp)
    padded = -(-v // unit) * unit
    if padded % tp:
        bad(f"even padding_size=lcm({HAD_BLOCK},{tp})={unit} gives {padded}, "
            f"not divisible by {tp}")
        return
    per_rank = padded // tp
    if v % HAD_BLOCK or padded % HAD_BLOCK or per_rank % HAD_BLOCK:
        bad(f"padding_size={unit} -> {padded} ({per_rank}/rank) but one of "
            f"vocab%{HAD_BLOCK}={v % HAD_BLOCK}, padded%{HAD_BLOCK}="
            f"{padded % HAD_BLOCK}, per-rank%{HAD_BLOCK}={per_rank % HAD_BLOCK} "
            f"is non-zero: a 6-bit EXL3 lm_head pad may not share a Hadamard "
            f"block with real output")
        return
    ok(f"padding_size=lcm({HAD_BLOCK},{tp})={unit} -> num_embeddings_padded {padded} "
       f"-> {per_rank} rows per rank = {per_rank // HAD_BLOCK} x {HAD_BLOCK} "
       f"({padded - v} zero rows total = {(padded - v) // HAD_BLOCK} whole blocks)")
    note("this is what patch-vllm-tp3.py's vocab_parallel_embedding.py edit does; "
         "LogitsProcessor truncates logits back to org_vocab_size, so the pad rows "
         "never reach a sampler")
    note("full-scope arm: lm_head is 6-bit EXL3, so this pad lands on an EXL3 output "
         "dim. It needs cuda-exl3 >= 754421f: f3e3090 accepts the padded vocab in "
         "create_weights (svh allocated zeroed, never written on the pad) and "
         "754421f makes _vocab_loaders fill a prefix instead of copy_ing the "
         "unpadded slice into the padded parameter. On 62f53e6/5903248 the boot "
         "raises 'EXL3 weights cannot be zero-extended'; on f3e3090 alone it passes "
         "that gate and then dies on a copy_ shape mismatch. tp3full's prelude "
         "checks for both before any weight is read (check-padload-tp3.py). On a "
         "checkpoint with a bf16 lm_head (routed-experts-only) none of this applies.")


STOCK_MOE_I = 2048          # what the downloaded checkpoint stores


def _expert_header_shapes(model: Path, sample: int = 24):
    """Trellis / suh / svh shapes read from the safetensors headers themselves.

    quantization_config.json can be edited into agreement with config.json while
    the weights on disk stay 2048 wide -- and nothing downstream would notice,
    because the loader sizes its parameters from the config and then narrow()s
    the checkpoint tensor to fit. Only the weight headers settle it.
    """
    import struct
    idx = model / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    wm = json.loads(idx.read_text())["weight_map"]
    names = [k for k in wm if ".mlp.experts." in k and
             k.rsplit(".", 1)[-1] in ("trellis", "suh", "svh")]
    if not names:
        return None
    names.sort()
    step = max(1, len(names) // sample)
    picked = names[::step][:sample]
    shapes, headers = {}, {}
    for n in picked:
        f = wm[n]
        if f not in headers:
            with open(model / f, "rb") as fh:
                hl = struct.unpack("<Q", fh.read(8))[0]
                headers[f] = json.loads(fh.read(hl))
        shapes[n] = headers[f][n]["shape"]
    return shapes


def check_mlp_and_experts(t: dict, tp: int, model: Path, ep: bool) -> None:
    head("dense MLP and shared expert (BF16)")
    divides("intermediate_size (dense layers)", int(t["intermediate_size"]), tp)

    moe_i = int(t["moe_intermediate_size"])
    n_shared = int(t.get("n_shared_experts") or 0)
    if n_shared:
        shared = moe_i * n_shared
        if shared % tp == 0:
            ok(f"shared expert intermediate {shared} -> {shared // tp} per rank")
        else:
            # FULL-SCOPE: unit lcm(HAD_BLOCK, tp), not lcm(64, tp). In a
            # full-scope checkpoint the shared expert is 6-bit EXL3, so
            # down_proj's per-rank INPUT must be a multiple of 128
            # (cuda_exl3/linear.py:84-88 refuses otherwise) and gate_up_proj's
            # output pad must be whole blocks (there it is only a warning --
            # the silent half of the same mistake).
            #   lcm(64, 3)  = 192 -> 2112 -> 704/rank = 5.5 x 128  REFUSED
            #   lcm(128, 3) = 384 -> 2304 -> 768/rank = 6 x 128    OK
            unit = lcm(HAD_BLOCK, tp)
            padded = -(-moe_i // unit) * unit * n_shared
            if padded % tp or (padded // tp) % HAD_BLOCK:
                bad(f"shared expert {shared} cannot be padded to a multiple of "
                    f"{tp} x {HAD_BLOCK} with unit lcm({HAD_BLOCK},{tp})={unit} "
                    f"(got {padded}, {padded // tp} per rank)")
            else:
                ok(f"shared expert intermediate {shared} -> pad to {padded} "
                   f"(unit lcm({HAD_BLOCK},{tp})={unit}) -> {padded // tp} per rank "
                   f"= {(padded // tp) // HAD_BLOCK} x {HAD_BLOCK}, "
                   f"{padded - shared} zero columns")
                note("6-bit EXL3 shared expert in the full-scope arm: this pad is on "
                     "gate_up_proj's OUTPUT and down_proj's INPUT, so both halves have "
                     "to be whole Hadamard blocks, not merely divisible by tp.")
                note("this is patch-vllm-tp3.py's model.py edit. It is BF16, so a zero "
                     "pad is a real zero. Do NOT use disable_tp instead: with EP the "
                     "MoE runner all-reduces the shared output, so a replicated shared "
                     f"expert is counted {tp} times and the logits are wrong while the "
                     "text stays fluent.")

    head("routed experts (EXL3 trellis)")
    n_routed = int(t["n_routed_experts"])
    if n_routed % tp:
        bad(f"n_routed_experts={n_routed} is not divisible by tp={tp}: expert "
            f"parallel cannot hand every rank whole experts, and the trellis cannot "
            f"be split. This model cannot run at tp={tp}.")
    else:
        ok(f"n_routed_experts {n_routed} -> {n_routed // tp} whole experts per rank "
           f"under --enable-expert-parallel"
           if ep else
           f"n_routed_experts {n_routed}: every rank holds all {n_routed} experts "
           f"(sliced, not distributed); divisibility by tp is not required here")
    # The trellis is only sliceable on a 128-column Hadamard boundary, so the
    # condition is moe_intermediate_size % (128*tp) == 0 -- not merely % tp.
    sliceable = moe_i % (HAD_BLOCK * tp) == 0
    if sliceable:
        ok(f"moe_intermediate_size {moe_i} slices {tp} ways: {moe_i // tp} columns "
           f"per rank = {moe_i // tp // HAD_BLOCK} whole {HAD_BLOCK}-column "
           f"Hadamard blocks; experts CAN be tensor-sliced")
    else:
        why = ("not an integer" if moe_i % tp
               else f"{moe_i // tp}, not a multiple of the {HAD_BLOCK}-wide "
                    f"Hadamard block")
        ok(f"moe_intermediate_size {moe_i} / {tp} is {why} -> the trellis CANNOT be "
           f"tensor-sliced -> --enable-expert-parallel is REQUIRED, not optional")

    if ep:
        if moe_i != STOCK_MOE_I:
            note(f"moe_intermediate_size is {moe_i}, not the stock {STOCK_MOE_I}: "
                 f"this is a padded sidecar being served under EP, which carries "
                 f"{moe_i / STOCK_MOE_I - 1:.1%} extra expert bytes and slices "
                 f"nothing. Serve it with ENABLE_EP=0 or serve the stock checkpoint.")
    elif not sliceable:
        bad(f"ENABLE_EP=0 at tp={tp} requires moe_intermediate_size to be a "
            f"multiple of {HAD_BLOCK * tp}; {moe_i} is not. The EXL3 trellis "
            f"would be cut mid-Hadamard-block and decode to noise -- silently. "
            f"Pad the intermediate ({-(-moe_i // (HAD_BLOCK * tp)) * HAD_BLOCK * tp} "
            f"is the next legal width) or set ENABLE_EP=1.")
    else:
        ok(f"ENABLE_EP=0 accepted: routed experts TENSOR-SLICED, "
           f"{moe_i} -> {moe_i // tp} per rank, no expert map")
        if moe_i != STOCK_MOE_I:
            pad = moe_i - STOCK_MOE_I
            ok(f"padded sidecar: {pad} of {moe_i} columns ({pad / moe_i:.1%}) are "
               f"the pad; all {pad} land on rank {tp - 1} "
               f"(its slice is columns {(tp - 1) * moe_i // tp}..{moe_i - 1})")
            note("the pad is dead work, not imbalance: every rank computes "
                 f"{moe_i // tp} columns, rank {tp - 1}'s last {pad} of them are "
                 "exactly zero because their svh is zero")

    qc = model / "quantization_config.json"
    if not qc.is_file():
        return
    try:
        storage = json.loads(qc.read_text()).get("tensor_storage") or {}
    except Exception as e:
        bad(f"could not read quantization_config.json: {e}")
        return
    hidden = int(t["hidden_size"])
    seen, mismatched, bits = 0, 0, set()
    for name, entry in storage.items():
        if ".mlp.experts." not in name:
            continue
        tensors = entry.get("stored_tensors") or {}
        tr = tensors.get(f"{name}.trellis")
        if not tr:
            continue
        k_tiles, n_tiles, packed = tr["shape"]
        bits.add(packed // 16)
        want = ((moe_i // 16, hidden // 16) if name.endswith("down_proj")
                else (hidden // 16, moe_i // 16))
        if (k_tiles, n_tiles) != want:
            mismatched += 1
            if mismatched < 4:
                bad(f"{name}: trellis tiles {(k_tiles, n_tiles)}, expected {want}")
        seen += 1
        if seen >= 3000:
            break
    if seen and not mismatched:
        ok(f"{seen} sampled expert trellis tensors all match "
           f"hidden={hidden} intermediate={moe_i}, bits={sorted(bits)}")
    per_rank = moe_i if ep else moe_i // tp
    ok(f"per-rank routed weights: "
       f"{n_routed // tp if ep else n_routed} experts x 3 projections, trellis "
       f"{hidden}x{per_rank} at {sorted(bits)} bit, "
       f"{'whole' if ep else f'sliced/{tp}'}")

    head("routed-expert weights on disk (safetensors headers)")
    try:
        shapes = _expert_header_shapes(model)
    except Exception as e:
        bad(f"could not read the safetensors headers: {e}")
        shapes = None
    if not shapes:
        bad("no expert tensors found in the weight index; cannot confirm the "
            "config matches the weights")
    else:
        wrong = 0
        for n, sh in sorted(shapes.items()):
            leaf = n.rsplit(".", 1)[-1]
            proj = n.rsplit(".", 2)[-2]
            if leaf == "trellis":
                want = ([moe_i // 16, hidden // 16] if proj == "down_proj"
                        else [hidden // 16, moe_i // 16])
                got = list(sh[:2])
            elif leaf == "suh":
                want = [moe_i] if proj == "down_proj" else [hidden]
                got = list(sh)
            else:
                want = [hidden] if proj == "down_proj" else [moe_i]
                got = list(sh)
            if got != want:
                wrong += 1
                if wrong < 4:
                    bad(f"{n}: weights on disk are {got}, config implies {want}. "
                        f"The config and the weights disagree -- create_weights "
                        f"would allocate one shape and narrow() the other.")
        if not wrong:
            ok(f"{len(shapes)} sampled expert tensors on disk agree with "
               f"moe_intermediate_size={moe_i} and hidden={hidden}")


def check_draft(draft: Path, tp: int) -> None:
    head(f"DFlash2 drafter {draft}")
    p = draft / "config.json"
    if not p.is_file():
        bad(f"no config.json in {draft}")
        return
    c = json.loads(p.read_text())
    q = int(c["num_attention_heads"])
    kv = int(c.get("num_key_value_heads", q))
    if not divides("draft num_attention_heads", q, tp):
        return
    if not divides("draft num_key_value_heads", kv, tp):
        return
    if q % kv:
        bad(f"draft GQA ratio broken: q={q} is not a multiple of kv={kv}; "
            f"FlashInfer requires qo_heads % kv_heads == 0")
    else:
        ok(f"draft GQA ratio {q}//{kv} = {q // kv} preserved; "
           f"local {q // tp}/{kv // tp}")
    note("even with draft_tensor_parallel_size=1 the process still has world tp>1, "
         "so the drafter's own config has to divide -- confirm with whoever is "
         "porting DFlash2 into this image")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True,
                    help="the TP sidecar directory (pad-tp3.py's output)")
    ap.add_argument("--draft", type=Path, default=None)
    ap.add_argument("--tp", type=int, default=3)
    ap.add_argument("--ep", type=int, default=1, choices=(0, 1),
                    help="1 = --enable-expert-parallel, 0 = tensor-sliced experts")
    args = ap.parse_args()

    print(f"GLM-5.3-Flash EXL3 preflight, tp={args.tp}, "
          f"ep={'on' if args.ep else 'OFF (experts tensor-sliced)'}")
    cfg = check_sidecar(args.model)
    if not cfg:
        print("\nRESULT: FAIL (no config)")
        return 2
    t = cfg.get("text_config") or cfg

    check_attention(t, args.tp)
    check_kda(t, args.tp)
    check_vocab(t, args.tp)
    check_mlp_and_experts(t, args.tp, args.model, bool(args.ep))
    if args.draft:
        check_draft(args.draft, args.tp)

    print()
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)} problem(s)) -- do not launch")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print(f"RESULT: PASS ({len(NOTES)} note(s)) -- arithmetic is sound for tp={args.tp}")
    print("Reminder: this checks shapes, not kernels. The gates after boot are the "
          "correctness probe (10/10) and the code exam (12/12).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

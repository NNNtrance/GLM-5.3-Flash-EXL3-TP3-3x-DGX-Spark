#!/usr/bin/env python3
"""Build a TP=3 sidecar view of a GLM-5.3-Flash checkpoint -- FULL-SCOPE arm.

Same as tp3/pad-tp3.py (padded config.json + relative symlinks to the weights)
plus one thing the routed-experts-only sidecar does not need: a REWRITTEN
``quantization_config.json`` carrying the supplementary packed-module mapping.

Why the mapping has to travel with the checkpoint
-------------------------------------------------
cuda-exl3 inverts vLLM's linear merges through ``packed_modules_mapping``, which
vLLM copies off the MODEL CLASS -- and ``glm5next`` declares none, so
``gate_up_proj.trellis`` has nowhere to land. Since cuda-exl3 5903248 the
CHECKPOINT may declare its own fusions and they are merged UNDER the model
class's, so both routes can be live at once without conflict:

  route A  patch-fullscope-tp3.py S1 (the model class)      -- always needed for
           S2/S3 anyway, so this file's mapping is belt AND braces
  route B  this file: "packed_modules_mapping" inside quantization_config.json
  route C  (>= fba9f27, also in 754421f) the same JSON in the environment:
             CUDA_EXL3_PACKED_MAPPING={"gate_up_proj":["gate_proj","up_proj"],...}
           Write it with NO SPACES: start-tp3.sh word-splits EXTRA_ENV with
           `for _kv in ${EXTRA_ENV}`, so a spaced JSON becomes five broken -e
           arguments.

The rewritten file is a full 48 MB copy, not a symlink: cuda-exl3 needs
``tensor_storage`` out of the same dict, so a small file holding only the
mapping would leave it with no per-tensor table at all.

The downloaded checkpoint is never modified. This writes a *sidecar directory*
next to it holding

  * relative symlinks (or hard links) to every file of the original, and
  * one rewritten ``config.json`` with the head counts padded for TP.

Why the config has to be padded on disk at all
----------------------------------------------
``--hf-overrides`` reaches the target model's top-level attributes, but the
KDA head count lives inside the nested ``text_config.linear_attn_config`` dict
(``Glm5NextTextConfig.__init__`` derives ``linear_num_heads`` from that dict and
the dict wins over the flat kwarg), and ``SpeculativeConfig`` builds the draft
model from its own config file. A file that is already right in every process
is simpler than three override paths that have to agree.

What gets padded, and what must NOT
-----------------------------------
padded   num_attention_heads      64 -> 66   (22 MLA heads per rank)
padded   num_key_value_heads      64 -> 66
padded   linear_attn_config.num_heads 64 -> 66  (22 KDA heads per rank)
padded   linear_num_heads         64 -> 66   (flat mirror, kept consistent)

NOT padded  moe_intermediate_size  2048
    The routed experts are EXL3 trellis. A trellis column is only meaningful on
    a 128-element Hadamard boundary and 2048/3 is not even an integer; a
    zero-extended trellis is not a zero-extended weight. Routed experts go
    through expert parallel instead (288 / 3 = 96 whole experts per rank), which
    needs no pad at all. Serve with ``--enable-expert-parallel``.

NOT padded  vocab_size            154880
    ``VocabParallelEmbedding`` keeps ``org_vocab_size`` and pads inside the
    module. The fix there is ``padding_size = lcm(128, tp) = 384`` (154880 ->
    155136 = 3 x 51712 = 3 x 404 x 128), applied by tp3full's
    ``patch-vllm-tp3.py``, not a config edit. 128 and not 64: a full-scope
    checkpoint's lm_head is 6-bit EXL3, and an EXL3 pad may not share a
    128-column Hadamard block with real output.

NOT padded  hidden_size / intermediate_size / kv_lora_rank
    4096, 12288 and 512 are all divisible by 3? No -- but they are never split
    by head count. 12288/3 = 4096 and 4096 (hidden) is the *unsplit* dim of a
    column-parallel layer, so nothing here breaks. See preflight-tp3.py, which
    checks each one explicitly instead of trusting this comment.

The refusal that matters
------------------------
Padding to the next multiple of tp is safe only while the *last* rank still
owns at least one real head. Padding 64 -> 96 at tp=3 gives rank 2 heads 64..95,
every one of them fabricated: the model loads, answers, and produces confident
garbage. That case is refused here rather than discovered in an eval.

Usage
-----
  # target (full-scope turboderp checkpoint)
  ./pad-tp3full.py /var/tmp/glm-5.3-flash-turboderp-4.05bpw \
                   /var/tmp/glm-5.3-flash-turboderp-4.05bpw-tp3 --tp 3

  # DFlash2 drafter (flat Qwen3-style config, GQA) -- UNCHANGED, the existing
  # /var/tmp/dflash2-draft-tp3 sidecar is reused; run this only to rebuild it
  ./pad-tp3full.py /var/tmp/dflash2-draft /var/tmp/dflash2-draft-tp3 --tp 3 --draft

Both directories must be mounted into the container, at paths that keep the
same relative relationship, or the symlinks dangle:

  -v /var/tmp/glm-5.3-flash-tr3-4bpw:/models/glm-5.3-flash-tr3-4bpw:ro \
  -v /var/tmp/glm-5.3-flash-tr3-4bpw-tp3:/models/glm-5.3-flash-tr3-4bpw-tp3:ro

Use ``--hardlink`` if you would rather mount one directory (same filesystem
only; the safetensors are shared inodes, not copies).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from math import gcd
from pathlib import Path

CONFIG = "config.json"
QUANT_CONFIG = "quantization_config.json"

# The fusions glm5next does not declare. Source ORDER is load-bearing: it
# must match the shard ids in stacked_params_mapping (patch-fullscope-tp3.py
# asserts that). `lm_head` is deliberately absent -- the head is a
# VocabParallelEmbedding whose vocab is dim 1 of the trellis and cuda-exl3
# loads it through its own _vocab_loaders; as a plain linear it would load
# without error and be silently wrong.
PACKED_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    "in_proj_qkv": ["qkv_proj"],
}


def ceil_to(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def check_last_rank_is_real(name: str, stock: int, padded: int, tp: int) -> None:
    """Refuse a pad where the highest rank owns no real head.

    per_rank = padded // tp. Rank tp-1 owns [(tp-1)*per_rank, padded). It is
    entirely fabricated when stock <= (tp-1)*per_rank.
    """
    per_rank = padded // tp
    first_of_last = (tp - 1) * per_rank
    real_on_last = stock - first_of_last
    if real_on_last <= 0:
        raise SystemExit(
            f"REFUSED: padding {name} {stock} -> {padded} at tp={tp} leaves rank "
            f"{tp - 1} with {per_rank} heads of which 0 are real. That rank would "
            f"compute pure padding and the model would answer fluent nonsense. "
            f"Pad to the next multiple of tp ({ceil_to(stock, tp)}), not beyond."
        )
    if real_on_last < per_rank:
        print(
            f"  note: {name} {stock} -> {padded}: rank {tp - 1} holds "
            f"{real_on_last}/{per_rank} real heads ({per_rank - real_on_last} zero)"
        )


# --------------------------------------------------------------------------
# config rewriting
# --------------------------------------------------------------------------

def pad_target(cfg: dict, tp: int) -> dict:
    text = cfg.get("text_config")
    if not isinstance(text, dict):
        raise SystemExit(
            "no text_config in this config.json -- is this the drafter? use --draft"
        )

    heads = int(text["num_attention_heads"])
    want = ceil_to(heads, tp)
    check_last_rank_is_real("num_attention_heads", heads, want, tp)

    text["num_attention_heads"] = want
    if int(text.get("num_key_value_heads", heads)) == heads:
        text["num_key_value_heads"] = want
    else:
        kv = int(text["num_key_value_heads"])
        kvw = ceil_to(kv, tp)
        check_last_rank_is_real("num_key_value_heads", kv, kvw, tp)
        text["num_key_value_heads"] = kvw

    lac = text.get("linear_attn_config")
    if isinstance(lac, dict) and lac.get("num_heads"):
        lh = int(lac["num_heads"])
        lw = ceil_to(lh, tp)
        check_last_rank_is_real("linear_attn_config.num_heads", lh, lw, tp)
        lac["num_heads"] = lw
        # Keep the flat mirror consistent. Glm5NextTextConfig lets the dict win,
        # but get_mamba_state_shape_from_config reads the flat attribute, and a
        # reader that disagrees with the loader is how silent corruption starts.
        text["linear_num_heads"] = lw
    elif "linear_num_heads" in text:
        lh = int(text["linear_num_heads"])
        lw = ceil_to(lh, tp)
        check_last_rank_is_real("linear_num_heads", lh, lw, tp)
        text["linear_num_heads"] = lw

    # Loud refusals for the two things a well-meaning future edit would try.
    moe_i = int(text.get("moe_intermediate_size", 0))
    if moe_i and moe_i % tp:
        print(
            f"  moe_intermediate_size stays {moe_i} (EXL3 trellis; "
            f"{moe_i}/{tp} is not a 128-aligned split) -> serve with "
            f"--enable-expert-parallel"
        )
    n_routed = int(text.get("n_routed_experts", 0))
    if n_routed and n_routed % tp:
        raise SystemExit(
            f"REFUSED: n_routed_experts={n_routed} is not divisible by tp={tp}; "
            "expert parallel cannot give every rank a whole number of experts, "
            "and the trellis cannot be split. This checkpoint cannot run at this tp."
        )
    return cfg


def pad_draft(cfg: dict, tp: int) -> dict:
    """Pad a flat GQA config (DFlash2 drafter: Qwen3-like, 32 q / 8 kv).

    Independent ceilings give 33/9 -> local 11/3, and FlashInfer needs
    q_heads % kv_heads == 0 (11/3 is not). Keep the stock 4:1 ratio: pad kv
    8 -> 9 first, then q = 9*4 = 36 -> local 12/3.
    """
    if "text_config" in cfg:
        raise SystemExit("this looks like the target config (has text_config); drop --draft")
    q = int(cfg["num_attention_heads"])
    kv = int(cfg.get("num_key_value_heads", q))
    ratio, rem = divmod(q, kv)
    if rem:
        raise SystemExit(f"drafter q={q} is not a multiple of kv={kv}; unsupported")

    want_kv = ceil_to(kv, tp)
    want_q = want_kv * ratio
    if want_q % tp:
        want_kv = ceil_to(want_kv + 1, tp)
        want_q = want_kv * ratio
    check_last_rank_is_real("draft num_key_value_heads", kv, want_kv, tp)
    check_last_rank_is_real("draft num_attention_heads", q, want_q, tp)
    if want_q % want_kv:
        raise SystemExit(f"draft pad broke the GQA ratio: q={want_q} kv={want_kv}")

    cfg["num_attention_heads"] = want_q
    cfg["num_key_value_heads"] = want_kv
    print(f"  draft GQA {q}/{kv} -> {want_q}/{want_kv} (local {want_q // tp}/{want_kv // tp})")
    return cfg


# --------------------------------------------------------------------------
# sidecar directory
# --------------------------------------------------------------------------

def write_quant_config(src: Path, dst: Path) -> None:
    """Copy quantization_config.json into the sidecar with our mapping merged in.

    Merge, never replace: the file already carries `tensor_storage` (48 MB of
    per-tensor bitrates) and cuda-exl3 needs it out of the same dict. An entry
    already present in the checkpoint wins, so a checkpoint that learns to
    declare its own fusions silently takes over from us.
    """
    p = src / QUANT_CONFIG
    if not p.is_file():
        print(f"  no {QUANT_CONFIG} in {src}: not a full-scope EXL3 checkpoint, "
              f"nothing to rewrite")
        return
    cfg = json.loads(p.read_text())
    existing = cfg.get("packed_modules_mapping")
    merged = dict(PACKED_MAPPING)
    if isinstance(existing, dict):
        merged.update(existing)
    cfg["packed_modules_mapping"] = merged
    if "tensor_storage" not in cfg:
        raise SystemExit(
            f"REFUSED: {p} has no `tensor_storage`. cuda-exl3 needs the full "
            "per-tensor table out of this same dict; writing a sidecar copy "
            "without it would leave the loader with nothing to size parameters "
            "from."
        )
    out = dst / QUANT_CONFIG
    out.write_text(json.dumps(cfg) + "\n")
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB) with "
          f"packed_modules_mapping {sorted(merged)}")
    print( "  serve with --hf-overrides "
          f'{{"quantization_config_file":"{out}"}}  (single quotes in the env '
          "file; the JSON must contain no spaces)")


def build_sidecar(src: Path, dst: Path, hardlink: bool, force: bool) -> None:
    if dst.exists():
        if not force and not (dst / ".harem-tp3-sidecar").exists():
            raise SystemExit(
                f"{dst} exists and is not a sidecar this script made. "
                "Refusing to touch it; pass --force if you are sure."
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / ".harem-tp3-sidecar").write_text(f"source={src}\n")

    if hardlink and src.stat().st_dev != dst.parent.stat().st_dev:
        raise SystemExit("--hardlink needs source and sidecar on the same filesystem")

    n_link = 0
    for entry in sorted(src.iterdir()):
        if entry.name in (CONFIG, QUANT_CONFIG):
            continue
        target = dst / entry.name
        if entry.is_dir():
            # Directories (docs/, eval/, ...) are never read by the loader;
            # one symlink to the whole tree keeps the view complete anyway.
            os.symlink(os.path.relpath(entry, dst), target)
            n_link += 1
            continue
        if hardlink:
            os.link(entry, target)
        else:
            os.symlink(os.path.relpath(entry, dst), target)
        n_link += 1
    print(f"  {n_link} entries {'hard-linked' if hardlink else 'symlinked'} into {dst}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="original checkpoint directory (never written)")
    p.add_argument("dst", type=Path, help="sidecar directory to create")
    p.add_argument("--tp", type=int, default=3)
    p.add_argument("--draft", action="store_true",
                   help="flat GQA drafter config instead of the target's text_config")
    p.add_argument("--hardlink", action="store_true",
                   help="hard-link instead of symlink (one mount, same filesystem only)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a destination this script did not create")
    args = p.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / CONFIG).is_file():
        raise SystemExit(f"no {CONFIG} in {src}")
    if src == dst:
        raise SystemExit("source and sidecar must differ; this script never edits in place")
    if args.tp < 2:
        raise SystemExit("--tp must be >= 2")

    cfg = json.loads((src / CONFIG).read_text())
    print(f"source {src}")
    cfg = pad_draft(cfg, args.tp) if args.draft else pad_target(cfg, args.tp)

    build_sidecar(src, dst, args.hardlink, args.force)
    (dst / CONFIG).write_text(json.dumps(cfg, indent=2) + "\n")
    if not args.draft:
        write_quant_config(src, dst)

    if args.draft:
        print(f"  wrote {dst / CONFIG}: heads {cfg['num_attention_heads']}"
              f"/{cfg['num_key_value_heads']}")
    else:
        t = cfg["text_config"]
        print(f"  wrote {dst / CONFIG}: heads {t['num_attention_heads']}, "
              f"kv {t['num_key_value_heads']}, kda "
              f"{t.get('linear_attn_config', {}).get('num_heads')}, "
              f"moe_i {t.get('moe_intermediate_size')} (untouched), "
              f"vocab {t.get('vocab_size')} (untouched, padded in-module to "
              f"{ceil_to(int(t['vocab_size']), 128 * args.tp // gcd(128, args.tp))} "
              f"with padding_size=lcm(128,{args.tp}))")
    print("OK")


if __name__ == "__main__":
    sys.exit(main())

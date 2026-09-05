#!/usr/bin/env python3
"""In-container DFlash2 drafter patches for TP=3 (Zeus/cuda-exl3 stack).

Companion to ``patch-vllm-tp3.py``. Two exact-anchor edits in
``vllm/model_executor/models/qwen3_dflash2.py``:

1. ``_harem_check_port_assumptions`` -- make the drafter head-count check
   TP-aware and pad-aware. The DFlash2 port's own check only asks whether the
   *config's* head counts divide the TP size. That is necessary but nowhere near
   sufficient once a padded sidecar is in play:

     - the TP=3 sidecar ``/var/tmp/dflash2-draft-tp3`` says 36 q-heads / 9 KV
       heads while the stored checkpoint has 32/8, and the config alone cannot
       tell a legitimate ``config == checkpoint + zero pad`` from a config that
       simply disagrees with its weights;
     - a config that *shrinks* the head count (say 24/6) also divides 3 and
       would silently truncate the checkpoint;
     - a config padded so wide that the top rank owns no real head at all
       (the 64 -> 96 mistake, see docs/03-tp3-padding-and-sidecars.md) divides 3 as
       well, starts, serves, and answers confidently wrong.

   The replacement reads the drafter checkpoint's own tensor shapes and demands
   ``config == checkpoint + pad`` with the GQA ratio intact and at least one real
   head on the highest rank.

2. ``DFlash2Qwen3ForCausalLM.load_weights`` -- after the weights are in, prove
   the pad is a ZERO pad. The config check above is arithmetic; this one is
   evidence. It reads the fabricated rows of every layer's ``qkv_proj`` (and the
   matching input columns of ``o_proj``) and fails if any of them is non-zero.

Why this matters more than it looks: a padded head whose q/k/v rows are zero
produces a zero attention output and a zero ``o_proj`` contribution, so it is
exactly a no-op. A padded head holding allocator garbage is a *fluent* drafter
that proposes slightly wrong tokens -- acceptance falls, nothing crashes, and
the cause is invisible in every log. That is the same failure class that cost
the NVFP4 stack a week on a 22-head decode kernel.

Both edits are no-ops in effect when the config matches the checkpoint (TP=2,
32/8), so the patch is safe to leave applied on every rank at any TP.

Escape hatch: ``HAREM_TP3_DRAFT_PAD_CHECK=warn`` downgrades edit 2's failure to
a log line. It exists for diagnosis. Do not serve with it -- a drafter that
needs it is a drafter whose proposals are not what you think they are.

Usage:  patch-dflash-tp3.py [--check] [--root /usr/local/lib/python3.12/dist-packages/vllm]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MARKER = "HAREM-TP3"
DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")

# --------------------------------------------------------------------------
# edit 1 -- config-side check
# --------------------------------------------------------------------------

CHECK_OLD = '''        tp_size = vllm_config.parallel_config.tensor_parallel_size
        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        if heads % tp_size or kv_heads % tp_size:
            raise ValueError(
                f"DFlash2 port: drafter head counts {heads}/{kv_heads} "
                f"(attention/KV) are not both divisible by tensor_parallel_size="
                f"{tp_size}. Use a drafter copy whose config matches the TP size "
                "(32/8 for TP=2, 36/9 for TP=3), or add loader-side head padding. "
                "See PATCHES.md."
            )
'''

CHECK_NEW = '''        # HAREM-TP3: TP- and pad-aware drafter head check. See patch-dflash-tp3.py.
        _harem_check_drafter_heads(self, vllm_config)
'''

# --------------------------------------------------------------------------
# edit 2 -- the helpers plus the post-load zero proof
# --------------------------------------------------------------------------

# NOTE: raw string. The helper body contains "\n" escapes that must reach the
# generated file as escapes, not as real newlines.
HELPERS = r'''

# --- HAREM-TP3 -------------------------------------------------------------
# Installed by patch-dflash-tp3.py. Everything below is inert when the drafter
# config matches its checkpoint (i.e. at TP=2 with the stock 32/8 copy).


def _harem_draft_stock_heads(config, model_path):
    """Read the drafter checkpoint's real head counts from its own tensors.

    Returns (q_heads, kv_heads, head_dim), or None when the header cannot be
    read -- the caller turns that into a hard failure, because "I could not
    check" and "it is fine" must never be the same outcome.
    """
    import json
    import os
    import struct

    def header(path):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(n))

    if not model_path or not os.path.isdir(model_path):
        return None
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as fh:
            weight_map = json.load(fh).get("weight_map", {})
        files = sorted(set(weight_map.values()))
    else:
        files = [f for f in sorted(os.listdir(model_path))
                 if f.endswith(".safetensors")]
    q_rows, kv_rows = set(), set()
    for name in files:
        path = os.path.join(model_path, name)
        if not os.path.exists(path):     # dangling sidecar link
            return None
        try:
            head = header(path)
        except Exception:
            return None
        for key, meta in head.items():
            if key == "__metadata__":
                continue
            if key.endswith("self_attn.q_proj.weight"):
                q_rows.add(int(meta["shape"][0]))
            elif key.endswith("self_attn.k_proj.weight"):
                kv_rows.add(int(meta["shape"][0]))
    # Every layer must agree; a checkpoint with mixed head counts per layer is
    # not something this pad reasoning covers, so refuse rather than guess.
    if len(q_rows) != 1 or len(kv_rows) != 1:
        return None
    q_rows, kv_rows = q_rows.pop(), kv_rows.pop()
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    if head_dim <= 0:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    if q_rows % head_dim or kv_rows % head_dim:
        return None
    return q_rows // head_dim, kv_rows // head_dim, head_dim


def _harem_check_drafter_heads(model, vllm_config):
    """Config-side gate: config must equal checkpoint plus a legal zero pad."""
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.logger import init_logger

    logger = init_logger("vllm.harem_tp3")
    config = model.config
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)

    # The layers are built with get_tensor_model_parallel_world_size(), so that
    # -- not parallel_config -- is the number the shapes actually have to obey.
    tp_size = get_tensor_model_parallel_world_size()
    cfg_tp = vllm_config.parallel_config.tensor_parallel_size
    if tp_size != cfg_tp:
        logger.info(
            "HAREM-TP3 drafter: building at tp=%d (parallel_config says %d)",
            tp_size, cfg_tp,
        )

    if heads % tp_size or kv_heads % tp_size:
        raise ValueError(
            f"DFlash2 port: drafter head counts {heads}/{kv_heads} "
            f"(attention/KV) are not both divisible by tensor_parallel_size="
            f"{tp_size}. Use a drafter copy whose config matches the TP size "
            "(32/8 for TP=2, 36/9 for TP=3), or add loader-side head padding. "
            "See PATCHES.md."
        )
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError(
            f"HAREM-TP3: drafter GQA is malformed -- {heads} query heads do not "
            f"group evenly into {kv_heads} KV heads."
        )

    model_path = getattr(
        getattr(model, "draft_model_config", None), "model", None
    )
    if model_path is None:
        model_path = getattr(
            vllm_config.speculative_config.draft_model_config, "model", None
        )
    stock = _harem_draft_stock_heads(config, model_path)
    if stock is None:
        raise ValueError(
            "HAREM-TP3: could not read the drafter checkpoint's tensor shapes at "
            f"{model_path!r}, so 'config head counts == checkpoint + pad' could "
            "not be verified. Refusing to load: an unverified pad is exactly the "
            "failure that stays silent (fluent drafter, collapsed acceptance). "
            "If the sidecar's relative symlinks dangle, the launcher's identity "
            "mount is wrong -- see start-tp3.sh check_relative_sidecar()."
        )
    stock_q, stock_kv, head_dim = stock

    if heads == stock_q and kv_heads == stock_kv:
        logger.info(
            "HAREM-TP3 drafter: config %d/%d matches the checkpoint, no pad "
            "(tp=%d, head_dim=%d)", heads, kv_heads, tp_size, head_dim,
        )
        model._harem_draft_pad = (stock_q, stock_kv, head_dim)
        return

    if heads < stock_q or kv_heads < stock_kv:
        raise ValueError(
            f"HAREM-TP3: drafter config {heads}/{kv_heads} is SMALLER than the "
            f"checkpoint's {stock_q}/{stock_kv}. That is not a pad, it is a "
            "truncation: real trained heads would be dropped and the drafter "
            "would still run. Use the sidecar that matches this checkpoint."
        )
    if heads * stock_kv != kv_heads * stock_q:
        raise ValueError(
            f"HAREM-TP3: drafter pad {stock_q}/{stock_kv} -> {heads}/{kv_heads} "
            f"changes the GQA ratio ({stock_q // stock_kv} -> {heads // kv_heads}). "
            "Every query head would be re-assigned to a different KV head and the "
            "drafter would decode fluent nonsense. Pad q and KV by the same factor."
        )
    # Fail closed on the 64 -> 96 class of mistake: the highest rank must own at
    # least one real head, otherwise it contributes nothing but fabrications.
    for label, stock_n, cfg_n in (("query", stock_q, heads), ("KV", stock_kv, kv_heads)):
        real_on_top = stock_n - (tp_size - 1) * (cfg_n // tp_size)
        if real_on_top <= 0:
            raise ValueError(
                f"HAREM-TP3: refusing drafter {label} pad {stock_n} -> {cfg_n} at "
                f"tp={tp_size}: rank {tp_size - 1} would own {cfg_n // tp_size} "
                f"heads of which {-real_on_top + cfg_n // tp_size} are fabricated "
                "and 0 are real. The engine would start and answer confidently "
                f"wrong. Pad to the smallest multiple of {tp_size} at or above "
                f"{stock_n}."
            )
    model._harem_draft_pad = (stock_q, stock_kv, head_dim)
    logger.info(
        "HAREM-TP3 drafter pad: %d/%d (checkpoint) -> %d/%d (config) at tp=%d; "
        "rank %d owns %d real + %d padded query heads",
        stock_q, stock_kv, heads, kv_heads, tp_size, tp_size - 1,
        stock_q - (tp_size - 1) * (heads // tp_size),
        (heads // tp_size) - (stock_q - (tp_size - 1) * (heads // tp_size)),
    )


def _harem_verify_draft_pad_is_zero(causal_lm):
    """Post-load proof: every fabricated row on this rank is exactly zero.

    A padded head only behaves as "absent" if its q/k/v rows and its o_proj
    input columns are zero -- then its attention output is zero and it adds
    nothing to the row-parallel sum. Loaded with garbage it is a silent
    acceptance-killer, so read the rows rather than trusting the loader.
    """
    import os

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.logger import init_logger

    logger = init_logger("vllm.harem_tp3")
    inner = getattr(causal_lm, "model", None)
    pad = getattr(inner, "_harem_draft_pad", None) or getattr(
        causal_lm, "_harem_draft_pad", None
    )
    if pad is None:
        raise ValueError(
            "HAREM-TP3: the drafter head check never ran, so the pad was never "
            "verified. patch-dflash-tp3.py edit 1 did not take effect."
        )
    stock_q, stock_kv, head_dim = pad
    config = causal_lm.config
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    if heads == stock_q and kv_heads == stock_kv:
        return                                  # nothing was padded

    tp_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    q_local = heads // tp_size
    kv_local = kv_heads // tp_size
    q_real = min(q_local, max(0, stock_q - rank * q_local))
    kv_real = min(kv_local, max(0, stock_kv - rank * kv_local))

    checked = 0
    bad = []
    for idx, layer in enumerate(getattr(inner, "layers", [])):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        weight = attn.qkv_proj.weight
        q_size = int(attn.q_size)
        kv_size = int(attn.kv_size)
        blocks = (
            ("q", 0, q_real, q_local),
            ("k", q_size, kv_real, kv_local),
            ("v", q_size + kv_size, kv_real, kv_local),
        )
        for label, base, real, total in blocks:
            if real >= total:
                continue
            rows = weight[base + real * head_dim: base + total * head_dim]
            checked += rows.numel()
            nz = int(rows.count_nonzero())
            if nz:
                bad.append(
                    f"layer {idx} qkv_proj.{label}: {nz}/{rows.numel()} padded "
                    f"rows are non-zero (heads {real}..{total - 1} of this rank)"
                )
        # o_proj is row-parallel: the padded heads are input COLUMNS here.
        o_weight = getattr(getattr(attn, "o_proj", None), "weight", None)
        if o_weight is not None and q_real < q_local:
            cols = o_weight[:, q_real * head_dim: q_local * head_dim]
            checked += cols.numel()
            nz = int(cols.count_nonzero())
            if nz:
                bad.append(
                    f"layer {idx} o_proj: {nz}/{cols.numel()} padded input "
                    "columns are non-zero"
                )
    if bad:
        message = (
            "HAREM-TP3: the drafter's padded heads are NOT zero on rank "
            f"{rank}/{tp_size}. A fabricated head only behaves as absent when "
            "its rows are zero; holding garbage it produces plausible but wrong "
            "draft tokens, acceptance falls, and nothing in the log says why.\n  "
            + "\n  ".join(bad[:8])
            + (f"\n  ... and {len(bad) - 8} more" if len(bad) > 8 else "")
        )
        if os.environ.get("HAREM_TP3_DRAFT_PAD_CHECK", "").lower() == "warn":
            logger.error("%s\n(downgraded by HAREM_TP3_DRAFT_PAD_CHECK=warn)", message)
        else:
            raise ValueError(message)
    else:
        logger.info(
            "HAREM-TP3 drafter pad verified zero on rank %d/%d: %d elements "
            "across %d padded query head(s) and %d padded KV head(s)",
            rank, tp_size, checked, q_local - q_real, kv_local - kv_real,
        )
# --- end HAREM-TP3 ---------------------------------------------------------
'''

LOADW_OLD = '''    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )
'''

LOADW_NEW = '''    def load_weights(self, weights):
        # HAREM-TP3: prove the config head pad is a zero pad, not a reshape.
        loaded = super().load_weights(weights)
        _harem_verify_draft_pad_is_zero(self)
        return loaded

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )
'''

# The helper block goes just above the first class that uses it.
HELPER_OLD = "class DFlashGroupedConv(nn.Module):"
HELPER_NEW = HELPERS + "\n\nclass DFlashGroupedConv(nn.Module):"

EDITS = [
    (HELPER_OLD, HELPER_NEW),
    (CHECK_OLD, CHECK_NEW),
    (LOADW_OLD, LOADW_NEW),
]

FILES = [("model_executor/models/qwen3_dflash2.py", EDITS)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def apply_file(path: Path, edits, check_only: bool) -> bool:
    src = path.read_text()
    if MARKER in src:
        print(f"  {path.name}: already patched (sha {sha(path)})")
        return False
    missing = [old for old, _ in edits if src.count(old) != 1]
    if missing:
        for old in missing:
            n = src.count(old)
            head = old.strip().splitlines()[0][:90]
            print(
                f"  ANCHOR {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}: {head}",
                file=sys.stderr,
            )
        raise SystemExit(
            f"{path}: {len(missing)} anchor(s) did not match exactly once. "
            "The DFlash2 port changed; re-read the file before re-running."
        )
    if check_only:
        print(f"  {path.name}: {len(edits)} anchors OK, NOT patched (--check)")
        return False
    for old, new in edits:
        src = src.replace(old, new, 1)
    before = sha(path)
    path.write_text(src)
    print(f"  {path.name}: patched {len(edits)} sites ({before} -> {sha(path)})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true",
                    help="verify every anchor still matches; change nothing")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"no vllm package at {args.root}")
    print(f"HAREM-TP3 DFlash2 drafter patches in {args.root}")
    changed = 0
    for rel, edits in FILES:
        path = args.root / rel
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        changed += bool(apply_file(path, edits, args.check))
    print(
        "HAREM-TP3 drafter: "
        + ("anchors verified" if args.check else f"{changed} file(s) changed")
    )


if __name__ == "__main__":
    main()

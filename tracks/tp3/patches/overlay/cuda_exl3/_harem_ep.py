"""Expert-parallel support for cuda-exl3's routed-expert method.

Installed as ``<site-packages>/cuda_exl3/_harem_ep.py`` by ``patch-exl3-ep.py``,
which also rewires four one-line call sites in ``cuda_exl3/moe.py``. Keeping the
logic here rather than inlining it into ``moe.py`` means the upstream file stays
readable as upstream, and a cuda-exl3 bump breaks the anchors loudly instead of
merging badly.

Why any of this is needed
-------------------------
``Exl3MoEMethod`` shards the routed experts the same way every other vLLM MoE
method does: it slices the *intermediate* dimension by the MoE tensor-parallel
rank (``_place``: ``w.narrow(0, r * part, part)`` for down_proj,
``w.narrow(1, r * part, part)`` for gate/up). That is correct for
``moe_intermediate_size = 2048`` at TP=2 -- 1024 is a whole number of 128-wide
Hadamard blocks -- and impossible at TP=3, where 2048/3 is not even an integer.

The way out is expert parallel, and cuda-exl3 already supports it *structurally*:
under ``--enable-expert-parallel`` vLLM's ``FusedMoEParallelConfig`` reports
``tp_size = 1, tp_rank = 0`` and hands ``create_weights`` 96 local experts, so
``_place`` narrows by nothing and every trellis stays 2048 wide. The three
device kernels are EP-aware too -- ``exl3_moe_gemm``, ``exl3_moe_had_in`` and
``exl3_moe_glu_had_in`` all retire a row block whose ``expert_ids`` entry is -1,
which is exactly the marker ``moe_align_block_size`` writes for an expert that
lives on another rank.

Two things are missing, and both are here:

1. ``apply()`` never asks for the map. It calls
   ``moe_align_block_size(topk_ids, block_m, E, pad_sorted_ids=True)`` with
   ``E = local`` and no ``expert_map``. Under EP that is wrong twice over: the
   routing ids are global (0..287) but the bucket count is 96, and nothing ever
   marks a remote block. ``align()`` below passes the global count *and* the
   map, which is the documented contract ("This requires the num_experts input
   arg to be the num global experts").

2. ``exl3_moe_gemm`` allocates its output with ``at::empty`` and skips the -1
   blocks, so those rows stay uninitialised -- and ``exl3_moe_combine`` reads
   every row unconditionally (``acc += w * rows_out[r]``). Under EP roughly
   two thirds of the rows are remote, so the sum a rank contributes to the
   all-reduce would be garbage, possibly NaN. ``zero_remote_rows()`` clears them
   before the combine.

   It SELECTS rather than multiplies: uninitialised memory can hold a NaN bit
   pattern and ``0 * NaN`` is NaN.

   Cost: one extra pass over ``(rows, hidden)`` per MoE layer per step, and it
   scales with batch, so it shows up hardest at concurrency. The default is
   ``masked_fill_`` (in place, no second buffer); ``HAREM_EP_ZERO_MODE=where``
   restores the original allocating ``torch.where`` and ``=off`` skips the pass
   entirely for measurement (invalid output, logged loudly). The real fix is
   kernel-side -- give ``exl3_moe_combine`` the ``expert_ids`` and ``block_m``
   it already has elsewhere and let it treat a -1 block's rows as zero, which
   removes the pass. That is a ~10-line CUDA change and an image rebuild; this
   is the python-only version that needs neither.

Everything here is CUDA-graph safe: no host sync, no data-dependent shapes.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

# NOT ``init_logger(__name__)``. vLLM's logging config attaches its handler to
# the "vllm" logger only (``logger.py`` DEFAULT_LOGGING_CONFIG, loggers: {"vllm":
# ...}), and the root logger has no handler, so a logger named
# "cuda_exl3._harem_ep" propagates to root and every record is DROPPED. The
# fail-closed raises below still work -- exceptions do not go through logging --
# but the evidence line ("mode=EP ... experts_local=96/288 trellis=whole") and
# the TENSOR-sliced warning were invisible on the first TP=3 boot, which is
# exactly backwards for a safety net. Naming the logger inside the vllm
# hierarchy puts them back in the boot log.
logger = init_logger("vllm.cuda_exl3.harem_ep")

# EXL3's input and output transforms are block-diagonal Hadamards over 128
# elements. A tensor-parallel slice of a trellis is only self-contained on a
# 128 boundary; the storage tile is 16, which is NOT the constraint that matters.
HAD_BLOCK = 128
TILE = 16

def _combine_is_ep_aware() -> bool:
    """Does the compiled kernel take expert_ids / block_m?

    Asked of the schema, not of the version string: the two travel separately
    (the python tree is copied between nodes, each ``_C…so`` is compiled where
    it runs). Answering "yes" against an old kernel would send uninitialised
    rows into the all-reduce, so the question is settled by the extension that
    is actually loaded, and anything unexpected answers "no".
    """
    try:
        from cuda_exl3 import _C  # noqa: F401  (registers the ops)
        sch = torch.ops.cuda_exl3_C.exl3_moe_combine.default._schema
        return any(a.name == "expert_ids" for a in sch.arguments)
    except Exception:  # pragma: no cover - a missing op is handled elsewhere
        return False


# Read once: this sits on the per-layer, per-step hot path.
#
# "kernel" is the default and means the combine retires the -1 blocks itself, so
# no clearing pass runs at all. It downgrades to "fill" on a kernel whose schema
# does not take expert_ids -- fail closed, because the alternative is a rank
# contributing recycled memory to the all-reduce.
_ZERO_MODE = os.environ.get("HAREM_EP_ZERO_MODE", "kernel").strip().lower()
if _ZERO_MODE not in ("kernel", "fill", "where", "off"):
    raise ValueError(
        f"HAREM_EP_ZERO_MODE={_ZERO_MODE!r} is not one of kernel|fill|where|off."
    )
_KERNEL_EP_COMBINE = _combine_is_ep_aware()
if _ZERO_MODE == "kernel" and not _KERNEL_EP_COMBINE:
    _ZERO_MODE = "fill"
    _DOWNGRADED = True
else:
    _DOWNGRADED = False

__all__ = ["check_expert_shape", "check_trellis_slice", "align", "zero_remote_rows"]


# ---------------------------------------------------------------------------
# load time
# ---------------------------------------------------------------------------

def check_expert_shape(prefix, moe, num_experts_local, hidden, inter_local) -> None:
    """Decide EP vs TP for this layer, refuse an illegal trellis split, log it.

    Called once per MoE layer from ``Exl3MoEMethod.create_weights``.
    """
    tp_size = int(getattr(moe, "tp_size", 1) or 1)
    ep_size = int(getattr(moe, "ep_size", 1) or 1)
    n_global = int(getattr(moe, "num_experts", num_experts_local) or num_experts_local)

    if tp_size > 1 and inter_local % HAD_BLOCK:
        raise ValueError(
            f"EXL3 {prefix}: tensor parallelism would slice the routed-expert "
            f"trellis to {inter_local} columns per rank, which is not a multiple "
            f"of the {HAD_BLOCK}-element Hadamard block. EXL3 weights cannot be "
            f"split there and cannot be zero-extended. Serve with "
            f"--enable-expert-parallel: {n_global} experts / {tp_size} ranks "
            f"gives whole experts per rank and leaves every trellis full width."
        )
    if tp_size > 1 and n_global % tp_size == 0:
        logger.warning_once(
            "EXL3: routed experts are being TENSOR-sliced (%d trellis columns per "
            "rank). Legal at this tp, but the slice only stays exact while it lands "
            "on a %d-element Hadamard boundary; %d whole experts per rank under "
            "--enable-expert-parallel carries the same bytes with no such condition.",
            inter_local, HAD_BLOCK, n_global // tp_size,
        )

    ep_on = ep_size > 1 and num_experts_local != n_global
    logger.info_once(
        "EXL3 routed experts: mode=%s ep_size=%d experts_local=%d/%d hidden=%d "
        "intermediate_local=%d trellis=%s",
        "EP" if ep_on else "TP",
        ep_size,
        num_experts_local,
        n_global,
        hidden,
        inter_local,
        "whole" if tp_size == 1 else f"sliced/{tp_size}",
    )
    return ep_on


def check_trellis_slice(prefix, layer, proj, w, tp_size, tp_rank) -> None:
    """Fail closed on every trellis load: shape must be exactly what we expect.

    ``w`` is the checkpoint tensor for one expert's one projection, shaped
    ``(k/16, n/16, 16*bits)``. The dimension that TP would slice is dim 0 for
    down_proj (its input is the intermediate) and dim 1 for gate/up (their
    output is the intermediate).
    """
    if proj == "down_proj":
        part = int(layer.w2_trellis.shape[1])
        dim = 0
    else:
        part = int(layer.w13_trellis.shape[2]) // 2
        dim = 1

    have = int(w.shape[dim]) if w.dim() == 3 else -1
    want = part * tp_size
    if have != want:
        raise ValueError(
            f"EXL3 {prefix}: {proj} trellis has {have} tiles on dim {dim}, "
            f"expected {want} ({part} per rank x tp {tp_size}). Refusing to load: "
            f"a mismatched trellis loads without error and decodes to noise."
        )
    if tp_size > 1 and (part * TILE) % HAD_BLOCK:
        raise ValueError(
            f"EXL3 {prefix}: {proj} trellis slice of {part * TILE} elements is not "
            f"a multiple of {HAD_BLOCK}; the Hadamard would be cut mid-block. "
            f"Use --enable-expert-parallel."
        )
    if tp_size == 1 and tp_rank != 0:
        raise ValueError(
            f"EXL3 {prefix}: tp_size=1 but tp_rank={tp_rank}; refusing to load."
        )


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

def align(method, layer, topk_ids, rows_hint, num_experts_local):
    """``moe_align_block_size``, expert-parallel aware.

    Returns ``(block_m, sorted_ids, expert_ids, n_rows)`` exactly as the code it
    replaces did, so the rest of ``apply()`` is untouched.

    With EP off this is bit-for-bit the upstream call (``expert_map`` is None
    when ``ep_size == 1``, and local == global). With EP on it buckets against
    the *global* expert count -- which is what the routing ids speak -- and then
    maps each block's expert to this rank's local index, leaving -1 for the
    blocks that belong to another rank. The three device kernels retire those.
    """
    emap = getattr(layer, "expert_map", None)
    n_align = num_experts_local
    if emap is not None:
        n_align = int(getattr(layer, "global_num_experts", num_experts_local))

    # The block-size ladder is a tokens-per-expert heuristic, so it wants the
    # same expert count the alignment used; with EP off this is unchanged.
    block_m = method._block_m(rows_hint, n_align)
    # pad_sorted_ids makes sorted_ids a whole number of blocks; without it
    # expert_ids covers more blocks than sorted_ids has entries and the gather
    # reads past the end.
    sorted_ids, expert_ids, n_rows = moe_align_block_size(
        topk_ids, block_m, n_align, expert_map=emap, pad_sorted_ids=True
    )
    return block_m, sorted_ids, expert_ids, n_rows


def zero_remote_rows(layer, rows_out, expert_ids, block_m):
    """Clear the rows of blocks the GEMM skipped, before the combine reads them.

    ``exl3_moe_gemm`` allocates with ``at::empty`` and returns early for a block
    whose expert id is negative, so those rows hold whatever the caching
    allocator last left there. ``exl3_moe_combine`` gathers every (token, k)
    pair unconditionally, so without this a rank's contribution to the
    all-reduce is arbitrary.

    Always a SELECT, never a multiply: recycled memory can hold a NaN bit
    pattern and ``0 * NaN`` is NaN.

    ``HAREM_EP_ZERO_MODE`` picks how:
      ``fill`` (default) -- ``masked_fill_``: one in-place pass, no second
          ``(rows, hidden)`` buffer. Same semantics as ``where``, less memory
          traffic and no allocator churn per MoE layer per step.
      ``where``          -- the original ``torch.where``; allocates a new
          ``(rows, hidden)`` tensor each call. Kept so the two can be A/B'd.
      ``off``            -- DIAGNOSTIC ONLY. Skips the clear, so the combine
          sums uninitialised rows. Use it to price the pass; the output is not
          valid and the log says so on every layer.

    ``kernel`` (the default) is the fix, now done: ``exl3_moe_combine`` takes
    ``expert_ids`` and ``block_m`` and treats a -1 block's rows as zero, so no
    pass runs here at all. Measured on GB10, one MoE layer, one rank:
    the clearing pass was 2.9 % of the layer at M=8, 4.9 % at M=64, 6.0 % at
    M=512 and 15.8 % at M=2048 (1.72 ms) -- not the "<1 %" a rows-of-real-tokens
    estimate suggested, because the pass covers the ALLOCATED rows
    (``M*top_k + min(E, M*top_k) * (block_m - 1)``), not the live ones.
    """
    if not getattr(layer, "_harem_ep_on", False):
        return rows_out
    mode = _ZERO_MODE
    if mode == "kernel":
        logger.info_once(
            "EXL3 EP: exl3_moe_combine retires the -1 blocks itself; the "
            "(rows, hidden) clearing pass is not run."
        )
        return rows_out
    if _DOWNGRADED:
        logger.warning_once(
            "EXL3 EP: HAREM_EP_ZERO_MODE=kernel was asked for but this build's "
            "exl3_moe_combine does not take expert_ids, so the python clearing "
            "pass is running instead. Rebuild cuda-exl3 to drop it."
        )
    if mode == "off":
        logger.error_once(
            "HAREM: HAREM_EP_ZERO_MODE=off -- remote MoE rows are NOT cleared, so "
            "this rank contributes uninitialised memory to the all-reduce. The "
            "output of this server is INVALID. Diagnostic use only."
        )
        return rows_out
    n = rows_out.shape[0]
    valid = (expert_ids >= 0).repeat_interleave(block_m)[:n]
    if mode == "where":
        return torch.where(valid.unsqueeze(1), rows_out, rows_out.new_zeros(()))
    return rows_out.masked_fill_(~valid.unsqueeze(1), 0)

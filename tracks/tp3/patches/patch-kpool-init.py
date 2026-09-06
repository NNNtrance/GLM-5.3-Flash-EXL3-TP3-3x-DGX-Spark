#!/usr/bin/env python3
"""HAREM-SM12 items 2+3: make the K-pool top-k buffer and its reader agree.

Items 2 and 3 of Zeuss5/cuda-exl3 issue #6 are the WRITE and the READ of the same
array, which is why they are one patch file.  Fix is tpurtell's
``patches/port-glm53-sm12-stability.py`` (Apache-2.0).

ITEM 2 -- ``model_executor/layers/sparse_attn_indexer_kpool.py``
    ``pool_topk = torch.empty((num_rows, select_k), dtype=torch.int32, ...)`` at
    two sites in our image (prefill and decode).  The top-k kernels only promise
    ``min(k, valid)`` outputs, so on a row with fewer than ``select_k`` valid
    pools the tail of the array is whatever the allocator handed back.  Those
    words are then used as pool ids and address cache rows nobody wrote.
    ``torch.full(..., -1)`` makes short rows deterministically invalid instead.

    NOTE ON SITE COUNT: the issue reports THREE sites (its image, lines
    939/1019/1137, spelled ``work_k``); tpurtell patches TWO (spelled
    ``select_k``).  The image we serve, vllm 0.1.dev20051+g487ecf187, has exactly
    TWO, spelled ``select_k``, at lines 538 (prefill) and 803 (decode).  This
    script asserts two and refuses on anything else, so a third site appearing in
    a future image stops the boot rather than being silently missed.

ITEM 3 -- ``models/glm5next/nvidia/ops/kpool_compress.py``
    ``_expand_pools_and_append_tail_kernel`` is the consumer of the buffer above.
    It guards the pool id from below only::

        hist_out = tl.where(pid >= 0, hist_val, -1)

    so a garbage id that happens to be positive expands to
    ``pid * POOL_SIZE + o`` and is emitted as a real token index.  The kernel
    already computes ``pool_len = seq_len // POOL_SIZE`` a few lines above, so the
    upper bound is free::

        hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)

WHAT IT COSTS
    Item 2 turns an allocation into an allocation + fill.  ``select_k`` is
    ``index_topk // index_kpool`` = 2048 / 4 = 512, ``num_rows`` up to
    MAX_NUM_BATCHED_TOKENS (2048 in production), so the worst case is a 4 MiB
    int32 memset per prefill chunk and a much smaller one per decode step.  On
    GB10 the pool is host memory, so this is not free -- it belongs in the A/B's
    "what did it cost" line, next to prefill tok/s.
    Item 3 adds one integer compare per output lane in a kernel that is already
    memory-bound; expected to be unmeasurable.

WHO IT PROTECTS
    tpurtell attribute the short rows to DCP.  We run TP=3 + EP with DCP=1, so
    the short/local-row case they hit does not arise the same way -- but the
    array is uninitialised regardless of DCP, and any row where the top-k kernel
    finds fewer than ``select_k`` valid pools (a sequence shorter than
    ``select_k * index_kpool`` = 2048 tokens, i.e. every short request) reaches
    the reader with an unwritten tail.  Item 3 is the one that actually contains
    the damage for us; item 2 makes the containment deterministic.

Three anchors, each asserted unique.  Idempotent.

Exit codes: 0 applied/already-applied/check-ok, 2 file missing, 3 anchor
mismatch, 4 precondition failed.
"""

import argparse
import ast
import os
import sys

MARK_KPOOL = "HAREM-SM12 item 2"
MARK_HIST = "HAREM-SM12 item 3"

REL_KPOOL = "model_executor/layers/sparse_attn_indexer_kpool.py"
REL_HIST = "models/glm5next/nvidia/ops/kpool_compress.py"

# --- item 2, site A: chunked prefill path -----------------------------------
OLD_A = """            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
                topk_dst = pool_topk
"""
NEW_A = """            if index_kpool > 1:
                # HAREM-SM12 item 2: top_k_per_row_prefill writes only
                # min(select_k, valid) ids per row; the rest of an empty()
                # buffer is stale memory that expands into real cache rows.
                pool_topk = torch.full(
                    (num_rows, select_k),
                    -1,
                    dtype=torch.int32,
                    device=logits.device,
                )
                topk_dst = pool_topk
"""

# --- item 2, site B: paged decode path --------------------------------------
OLD_B = """        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
"""
NEW_B = """        if index_kpool > 1:
            # HAREM-SM12 item 2: top_k_per_row_decode / persistent_topk write
            # only min(select_k, valid) ids per row; the rest of an empty()
            # buffer is stale memory that expands into real cache rows.
            pool_topk = torch.full(
                (num_rows, select_k),
                -1,
                dtype=torch.int32,
                device=logits.device,
            )
            topk_dst = pool_topk
"""

# --- item 3: the reader -----------------------------------------------------
OLD_H = """    hist_out = tl.where(pid >= 0, hist_val, -1)
"""
NEW_H = """    # HAREM-SM12 item 3: bound the pool id from above as well. pool_len is
    # already in scope; without this an out-of-range id expands to a real
    # token index instead of the -1 the consumer treats as "no token".
    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)
"""


def _edit(path, edits, mark, label, check):
    """Apply (old, new) pairs to `path`, each asserted to occur exactly once."""
    if not os.path.isfile(path):
        print(f"patch-kpool-init: {path} not found", file=sys.stderr)
        return 2, None
    src = open(path).read()
    if mark in src:
        print(f"patch-kpool-init: {label} already applied ({path})")
        return 0, None
    out = src
    for i, (old, new) in enumerate(edits, 1):
        n = out.count(old)
        if n != 1:
            print(
                f"patch-kpool-init: {label} anchor {i}/{len(edits)} count={n} "
                f"(expected 1) in {path} -- refusing.",
                file=sys.stderr,
            )
            return 3, None
        out = out.replace(old, new, 1)
    try:
        ast.parse(out, filename=path)
    except SyntaxError as exc:
        print(
            f"patch-kpool-init: {label} patched source does not parse: {exc}",
            file=sys.stderr,
        )
        return 4, None
    if check:
        print(f"patch-kpool-init: CHECK OK -- {label}: {len(edits)} anchor(s) unique")
        return 0, None
    return 0, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="path of the vllm package")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the anchors and report; never write",
    )
    a = ap.parse_args()

    p_kpool = os.path.join(a.root, REL_KPOOL)
    p_hist = os.path.join(a.root, REL_HIST)

    # Guard the site count explicitly: the issue's image had three, ours has two.
    if os.path.isfile(p_kpool) and MARK_KPOOL not in open(p_kpool).read():
        n_sites = open(p_kpool).read().count("pool_topk = torch.empty(")
        if n_sites != 2:
            print(
                f"patch-kpool-init: expected 2 'pool_topk = torch.empty(' sites in "
                f"{p_kpool}, found {n_sites} -- refusing.\n"
                f"  Zeuss5's image has 3 (spelled work_k); ours has 2 (select_k).\n"
                f"  A different count means a site would be left uninitialised.",
                file=sys.stderr,
            )
            return 3

    rc_k, out_k = _edit(
        p_kpool, [(OLD_A, NEW_A), (OLD_B, NEW_B)], MARK_KPOOL, "item 2", a.check
    )
    if rc_k:
        return rc_k
    rc_h, out_h = _edit(p_hist, [(OLD_H, NEW_H)], MARK_HIST, "item 3", a.check)
    if rc_h:
        return rc_h

    # Both files validated before either is written: never leave the tree with
    # the writer fixed and the reader not, or the other way round.
    if out_k is not None:
        open(p_kpool, "w").write(out_k)
        print(f"patch-kpool-init: applied item 2 to {p_kpool} (2 sites -> torch.full -1)")
    if out_h is not None:
        open(p_hist, "w").write(out_h)
        print(f"patch-kpool-init: applied item 3 to {p_hist} (pid < pool_len bound)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

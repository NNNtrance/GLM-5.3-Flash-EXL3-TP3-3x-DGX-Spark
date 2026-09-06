#!/usr/bin/env python3
"""HAREM-SM12 item 4: keep the DSA indexer's prefill-metadata kernel shape-generic.

``v1/attention/backends/mla/indexer.py`` ->
``BuildPrefillChunkMetadataKernel.kernel`` takes ``query_slice_start`` and
``query_slice_stop`` as plain Triton scalars under a bare ``@triton.jit``.  Triton
specialises non-constexpr int arguments, so the compiled variant depends on the
chunk offset.  Item 4 of Zeuss5/cuda-exl3 issue #6; fix is tpurtell's
``patches/port-glm-prefill-jit.py`` (Apache-2.0).

WHAT WE MEASURED, because the issue overstates the cost
-------------------------------------------------------
Triton 3.7.1's ``native_specialize_impl`` gives an int argument exactly THREE
specialisation classes, not one per value:

    value == 1        -> ("constexpr", 1)   killed by do_not_specialize
    value % 16 == 0   -> ("i32", "D")       killed by do_not_specialize_on_alignment
    otherwise         -> ("i32", "")

So this is NOT "a recompile per chunk offset"; it is at most three variants.
``get_warmup_keys`` pre-compiles::

    query_slice_start = WarmupIntRange(0, 2)                  -> {0, 1}
    query_slice_stop  = (1, 2*max_tokens-1, 2*max_tokens)     -> {1, 15, 16}

``query_slice_stop`` therefore covers all three classes and is fully warm.
``query_slice_start`` covers only ("i32","D") (0) and ("constexpr",1) (1) -- the
("i32","") class is MISSING from the warmup.

That class is reachable: ``_build_chunk_specs`` sub-chunks the query dimension
with ``max_q = max(1, max_logits_elems // chunk_n)`` and emits
``slice(q_off, q_off + sub_m)`` for ``q_off in range(0, chunk_m, max_q)``.
``max_q`` is a logits-budget quotient, not a rounded quantity, so a non-1,
non-16-aligned ``query_slice_start`` appears as soon as a prefill is large enough
to need query-dimension sub-chunking -- i.e. exactly on long prefills.

EXPECTED EFFECT: removes one cold Triton compile per engine process, landing
inside a long prefill step (tail jitter / a one-off TTFT spike), and collapses
the kernel to a single variant so the warmup's 6 dispatch keys dedupe to 1
compile.  It is NOT a steady-state throughput change and must not be reported as
one.  Cost: the kernel loses the divisible-by-16 hint on these two scalars; they
are loop bounds and comparison operands in a metadata kernel, not pointers, so no
vectorisation is lost.

Decorator form verified against siblings already in the same tree, e.g.
``models/minimax_m3/common/ops/index_topk.py``::

    @triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
    @triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])

``do_not_specialize`` alone already forces ("i32", None); the alignment list is
belt-and-braces and matches tpurtell byte for byte.

One anchor, unique because the module's other ``@triton.jit`` is a module-level
function with no ``@staticmethod`` above it.  Idempotent.

Exit codes: 0 applied/already-applied/check-ok, 2 file missing, 3 anchor
mismatch, 4 precondition failed.
"""

import argparse
import ast
import os
import sys

MARK = 'do_not_specialize=["query_slice_start", "query_slice_stop"]'
REL = "v1/attention/backends/mla/indexer.py"

OLD = """    @staticmethod
    @triton.jit
    def kernel(
"""

NEW = """    @staticmethod
    # HAREM-SM12 item 4: query_slice_start/stop are chunk offsets. Triton
    # specialises int args on ==1 and on 16-alignment; the warmup only covers
    # start in {0, 1}, so a query-dimension sub-chunk whose offset is neither
    # triggers a cold JIT inside a long prefill. Keep the kernel shape-generic.
    @triton.jit(
        do_not_specialize=["query_slice_start", "query_slice_stop"],
        do_not_specialize_on_alignment=["query_slice_start", "query_slice_stop"],
    )
    def kernel(
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="path of the vllm package")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the anchor and report; never write",
    )
    a = ap.parse_args()

    p = os.path.join(a.root, REL)
    if not os.path.isfile(p):
        print(f"patch-indexer-nospec: {p} not found", file=sys.stderr)
        return 2

    src = open(p).read()

    if MARK in src:
        print(f"patch-indexer-nospec: already applied ({p})")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(
            f"patch-indexer-nospec: ANCHOR count={n} (expected 1) in {p} -- refusing.\n"
            f"  Either BuildPrefillChunkMetadataKernel.kernel moved, or another\n"
            f"  '@staticmethod / @triton.jit / def kernel(' appeared. Re-read the file.",
            file=sys.stderr,
        )
        return 3

    # The two scalars must actually be parameters of that kernel, otherwise
    # do_not_specialize names a parameter that does not exist and Triton raises
    # only at first launch -- long after this script said "applied".
    for name in ("query_slice_start", "query_slice_stop"):
        if f"        {name},\n" not in src:
            print(
                f"patch-indexer-nospec: '{name}' is not a positional parameter "
                f"of the kernel in {p} -- refusing",
                file=sys.stderr,
            )
            return 4

    out = src.replace(OLD, NEW, 1)

    try:
        ast.parse(out, filename=p)
    except SyntaxError as exc:
        print(
            f"patch-indexer-nospec: patched source does not parse: {exc}",
            file=sys.stderr,
        )
        return 4

    if a.check:
        print(
            f"patch-indexer-nospec: CHECK OK -- anchor unique in {p}, "
            f"both scalars present, patched source parses"
        )
        return 0

    open(p, "w").write(out)
    print(
        f"patch-indexer-nospec: applied to {p}\n"
        f"  BuildPrefillChunkMetadataKernel.kernel: query_slice_start/stop are "
        f"no longer specialised (one variant instead of up to three)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

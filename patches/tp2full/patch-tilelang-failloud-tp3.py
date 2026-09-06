#!/usr/bin/env python3
"""HAREM-TP3: make the ``flashinfer.comm`` preload in tilelang_kernels.py loud.

THE HIDDEN GUARD
----------------
``vllm/model_executor/kernels/mhc/tilelang_kernels.py:22-28`` states, in
upstream's own words, that the import order is load-bearing::

    # Preload flashinfer.comm so its CudaRTLibrary binds the real libcudart
    # (via find_loaded_library) before tilelang imports libcudart_stub.so,
    # which otherwise maps at a lower address and shadows the real libcudart,
    # breaking flashinfer all-reduce on sm100. Import order is load-bearing;
    # this must run before `import tilelang`.
    with contextlib.suppress(Exception):
        import flashinfer.comm  # noqa: F401
    import tilelang

The guard that "must run" is wrapped in ``contextlib.suppress(Exception)``.  If
that import fails for any transient reason (flashinfer JIT cache lock, file
race, memory pressure) the protection is skipped IN SILENCE, tilelang loads the
stub first, and the symptom appears much later as an illegal address at kernel
launch.  That silence is what makes the boot flake intermittent rather than
reproducible (``HC-MIXING-ANALIZ.md`` section 6; the TileLang JIT cache is
already persistent via the ``/var/tmp/exl3-zeus-cache/tilelang`` mount, so cache
warming does not close this race -- only visibility does).

WHAT THIS PATCH DOES
--------------------
``HAREM_TILELANG_FAILLOUD=1`` replaces the suppression with try / print / raise,
so a failed preload stops the rank in the first second with a named cause.
Unset or "0" keeps upstream behaviour byte for byte.

Pairs with ``flashinfer-warmup.py``, which the prelude runs once per rank before
``vllm serve`` so the import (and its version) is visible in the boot log even
when it succeeds.

Two anchors in one file (the import block and the module header); fails closed
if either is missing or not unique.  Idempotent.
"""

import argparse
import os
import sys

MARK = "HAREM-TP3: upstream hides this preload"

OLD = """    with contextlib.suppress(Exception):
        import flashinfer.comm  # noqa: F401
"""

NEW = '''    # HAREM-TP3: upstream hides this preload behind contextlib.suppress, so a
    # transient failure silently skips the libcudart binding, tilelang then
    # loads libcudart_stub.so first, and the symptom surfaces much later as an
    # illegal address at kernel launch.  HAREM_TILELANG_FAILLOUD=1 turns that
    # silence into an immediate, named error.  Unset/"0" == upstream.
    if os.environ.get("HAREM_TILELANG_FAILLOUD", "0") == "1":
        try:
            import flashinfer.comm  # noqa: F401
        except Exception as _harem_fi_exc:
            print(
                "[harem-tilelang] FATAL: `import flashinfer.comm` failed while "
                "HAREM_TILELANG_FAILLOUD=1 -- tilelang would load "
                "libcudart_stub.so first and shadow the real libcudart "
                f"(cause: {_harem_fi_exc!r})",
                flush=True,
            )
            raise
        else:
            import flashinfer as _harem_flashinfer

            print(
                "[harem-tilelang] flashinfer.comm preloaded before tilelang "
                f"(flashinfer {getattr(_harem_flashinfer, \'__version__\', \'?\')})",
                flush=True,
            )
    else:
        with contextlib.suppress(Exception):
            import flashinfer.comm  # noqa: F401
'''

OLD_IMPORTS = """import contextlib
import math
"""

NEW_IMPORTS = """import contextlib
import math
import os
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path of the vllm package")
    a = ap.parse_args()
    p = os.path.join(a.root, "model_executor/kernels/mhc/tilelang_kernels.py")
    if not os.path.isfile(p):
        print(f"patch-tilelang-failloud: {p} not found", file=sys.stderr)
        return 2
    src = open(p).read()
    if MARK in src:
        print(f"patch-tilelang-failloud: already applied ({p})")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(
            f"patch-tilelang-failloud: ANCHOR count={n} (expected 1) in {p}",
            file=sys.stderr,
        )
        return 3
    m = src.count(OLD_IMPORTS)
    if m != 1:
        print(
            f"patch-tilelang-failloud: IMPORT anchor count={m} (expected 1) in {p}",
            file=sys.stderr,
        )
        return 4
    if "import tilelang" not in src:
        print(
            "patch-tilelang-failloud: `import tilelang` missing - refusing",
            file=sys.stderr,
        )
        return 5
    out = src.replace(OLD, NEW).replace(OLD_IMPORTS, NEW_IMPORTS, 1)
    open(p, "w").write(out)
    print(
        f"patch-tilelang-failloud: applied to {p} "
        "(HAREM_TILELANG_FAILLOUD=1 honoured)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

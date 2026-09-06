#!/usr/bin/env python3
"""HAREM-TP3: import ``flashinfer.comm`` once per rank, before ``vllm serve``.

Two jobs, both cheap (~2.0 s measured in exl3-zeus:9bf594c, CPU only):

  1. Make the import VISIBLE.  ``tilelang_kernels.py:26`` performs this import
     inside ``contextlib.suppress(Exception)``; when it fails there, nothing is
     logged and the failure resurfaces later as an illegal address at kernel
     launch (HC-MIXING-ANALIZ.md section 6).  One line in the boot log turns an
     intermittent flake into a fact.
  2. Warm flashinfer's own JIT cache in a single process before the workers
     start, so the first real import does not race N ranks against one cache.

Exit code: 0 unless ``HAREM_TILELANG_FAILLOUD=1``, in which case a failed import
returns non-zero and the prelude's ``run`` helper stops the rank -- which is the
whole point of that knob.  ``HAREM_FLASHINFER_WARMUP=0`` skips the step.
"""

import os
import sys
import time


def main() -> int:
    if os.environ.get("HAREM_FLASHINFER_WARMUP", "1") == "0":
        print("[flashinfer-warmup] skipped (HAREM_FLASHINFER_WARMUP=0)")
        return 0
    strict = os.environ.get("HAREM_TILELANG_FAILLOUD", "0") == "1"
    t0 = time.time()
    try:
        import flashinfer
        import flashinfer.comm  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(
            f"[flashinfer-warmup] FAILED after {time.time() - t0:.1f}s: {exc!r}  "
            "-- tilelang will load libcudart_stub.so first and shadow the real "
            "libcudart (tilelang_kernels.py:22-28)",
            file=sys.stderr,
            flush=True,
        )
        return 7 if strict else 0
    print(
        f"[flashinfer-warmup] flashinfer "
        f"{getattr(flashinfer, '__version__', '?')} comm import OK in "
        f"{time.time() - t0:.1f}s (failloud={'on' if strict else 'off'})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

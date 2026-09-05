#!/usr/bin/env python3
"""Route the large-M mHC block to the HAREM fused post+prenorm kernel.

What it changes
---------------
``vllm/model_executor/kernels/mhc/tilelang.py`` -- one anchor, inside
``mhc_fused_post_pre_tilelang``, right after ``use_small_fma`` is computed.

Today the large-M (prefill) path runs three kernels per call:

    k1  mhc_post_tilelang            writes residual_cur  (M, 4, 4096) bf16
    k2  tf32_hc_prenorm_gemm         READS  residual_cur
    k3  mhc_pre_big_fuse_with_norm   READS  residual_cur

k2's entire traffic is that one re-read: 32,768 B/token, 68 MB per call at
M=2032, 341 us, 200 GB/s -- 89 % of the measured read roofline.  The block is
memory-bound by a factor of 76 (5.3 FLOP/byte against a 405 FLOP/byte balance
point), so removing the read is the only lever that moves it.

``harem_hc_fusion.py`` computes k1's output and reduces it against ``fn`` while
it is still in registers, which deletes k2 entirely:

    k1 + k2 today   107,448 B/token      1,024 us/call (M=2032)
    fused            ~74,600 B/token       ~690 us/call    -30 %
                                          -> ceiling -2.5 .. -2.7 % prefill wall

The three-times-touched residual is documented in
``tp3/HC-MIXING-ANALIZ.md`` sections 2 and 4; the design and the expected
traffic are in ``tp3/HC-FUSION-PLAN.md``.

Why this is not the fused kernel that already ships
---------------------------------------------------
``mhc_fused_tilelang`` does the same fusion but grids over ``(token, n_tile,
split_k)``, so every CTA re-reads the residual row.  Forced on at M=2048 it is
+32 % SLOWER than the unfused pair; it is only ever selected at <=16 tokens.
The HAREM kernel tiles over tokens instead and finishes all 24 outputs in one
CTA.

Safety
------
* Off unless ``HAREM_MHC_FUSED_LARGE=1``.  Unset (the default) leaves the stock
  three-kernel path byte-for-byte as upstream: the injected code evaluates one
  dict lookup and one ``is not None`` per call.
* Also off below ``HAREM_MHC_FUSED_MIN_M`` tokens (default 256) and for every
  decode/small-M call, so the CUDA-graph decode path is untouched.
* The hook returns ``None`` for any dtype/shape/stride it does not handle and
  disables itself loudly on the first exception -- both land on the stock path.
* Rollback is ``unset HAREM_MHC_FUSED_LARGE``; no rebuild, no re-patch.

Usage
-----
    patch-mhcfused-tp3.py [--check] [--root /usr/local/lib/python3.12/dist-packages/vllm]
                          [--kernel <path to harem_hc_fusion.py>]

``--check`` verifies the anchor and the kernel source without writing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

MARKER = "HAREM hc-fusion"
DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
DEFAULT_KERNEL = Path(__file__).resolve().parent / "overlay" / "vllm" / "harem_hc_fusion.py"
KERNEL_REL = "model_executor/kernels/mhc/harem_hc_fusion.py"
TARGET_REL = "model_executor/kernels/mhc/tilelang.py"

OLD = '''    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = num_tokens <= 16
    if use_small_fma:
'''

NEW = '''    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = num_tokens <= 16
    # --- HAREM hc-fusion (HAREM_MHC_FUSED_LARGE=1) ---------------------------
    # Large-M only: one Triton kernel does the post mapping (k1) and the
    # pre-norm GEMM (k2) together, so residual_cur is never read back from HBM
    # between them.  -30 % on the k1+k2 pair, ceiling -2.7 % on the prefill wall.
    # Fail-closed in three independent places: the env default is off, the hook
    # is None if the module will not import, and the hook returns None for any
    # shape it does not handle.  Each of those falls through to the stock path.
    _harem_hook = globals().get("_HAREM_HC_FUSION_HOOK", 0)
    if _harem_hook == 0:
        import os as _harem_os

        _harem_hook = None
        if _harem_os.environ.get("HAREM_MHC_FUSED_LARGE", "0") == "1":
            try:
                from vllm.model_executor.kernels.mhc.harem_hc_fusion import (
                    harem_mhc_fused_post_pre as _harem_hook,
                )
            except Exception:  # noqa: BLE001
                _harem_hook = None
        globals()["_HAREM_HC_FUSION_HOOK"] = _harem_hook
    if _harem_hook is not None and not use_small_fma:
        _harem_out = _harem_hook(
            residual_flat,
            x_flat,
            post_layer_mix_flat,
            comb_res_mix_flat,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
            hc_mult,
            hidden_size,
        )
        if _harem_out is not None:
            _harem_res, _harem_pm, _harem_cm, _harem_li = _harem_out
            return (
                _harem_res.view(*outer_shape, hc_mult, hidden_size),
                _harem_pm.view(*outer_shape, hc_mult, 1),
                _harem_cm.view(*outer_shape, hc_mult, hc_mult),
                _harem_li.view(*outer_shape, hidden_size),
            )
    # --- end HAREM hc-fusion -------------------------------------------------
    if use_small_fma:
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def install_kernel(root: Path, kernel_src: Path, check_only: bool) -> bool:
    dst = root / KERNEL_REL
    if not kernel_src.is_file():
        raise SystemExit(f"kernel source missing: {kernel_src}")
    if not dst.parent.is_dir():
        raise SystemExit(f"no mhc kernels package at {dst.parent}")
    same = dst.is_file() and sha(dst) == sha(kernel_src)
    if check_only:
        print(
            f"  {dst.name}: source OK (sha {sha(kernel_src)}), "
            + ("installed and identical" if same else "would install")
            + " (--check)"
        )
        return False
    if same:
        print(f"  {dst.name}: already installed, identical (sha {sha(dst)})")
        return False
    shutil.copyfile(kernel_src, dst)
    print(f"  {dst.name}: installed (sha {sha(dst)})")
    return True


def apply_file(path: Path, edits, check_only: bool) -> bool:
    src = path.read_text()
    if MARKER in src:
        print(f"  {path.name}: already patched (sha {sha(path)})")
        return False
    missing = [old for old, _ in edits if src.count(old) != 1]
    if missing:
        for old in missing:
            n = src.count(old)
            print(
                f"  ANCHOR {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}: "
                + old.strip().splitlines()[0][:90],
                file=sys.stderr,
            )
        raise SystemExit(f"{path}: anchor did not match exactly once.")
    if check_only:
        print(f"  {path.name}: {len(edits)} anchor(s) OK, NOT patched (--check)")
        return False
    for old, new in edits:
        src = src.replace(old, new, 1)
    before = sha(path)
    path.write_text(src)
    print(f"  {path.name}: patched {len(edits)} site ({before} -> {sha(path)})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"no vllm package at {args.root}")
    target = args.root / TARGET_REL
    if not target.is_file():
        raise SystemExit(f"missing {target}")

    print(f"HAREM hc-fusion routing in {args.root}")
    changed = 0
    changed += bool(install_kernel(args.root, args.kernel, args.check))
    changed += bool(apply_file(target, [(OLD, NEW)], args.check))
    print(
        "HAREM hc-fusion: "
        + (
            "anchor + kernel verified"
            if args.check
            else f"{changed} file(s) changed; enable with HAREM_MHC_FUSED_LARGE=1"
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HAREM-SM12 item 1: stop enabling Programmatic Dependent Launch (PDL) on sm_12x.

Upstream ``CudaPlatformBase.is_arch_support_pdl`` returns ``major >= 9``.  GB10 is
sm_121, i.e. ``major == 12``, so the day-zero gate turns PDL on for every KDA and
mHC launch site on hardware where PDL was never qualified.  tpurtell's
``glm-5.3-flash-ext3-2x-rtx`` (Apache-2.0) reports KDA recurrent-state races on
SM12x from its dual-Spark qualification and ships ``return major in (9, 10)``.
Reported to us as item 1 of Zeuss5/cuda-exl3 issue #6.

This patch installs that fix *behind a runtime env knob* so the A/B can flip it
without rebuilding or re-patching:

    HAREM_PDL_SM12 unset or "0"  ->  PDL OFF on sm_12x   (tpurtell's fix; DEFAULT)
    HAREM_PDL_SM12="1"           ->  PDL ON  on sm_12x   (upstream behaviour)

Capabilities 9 and 10 are unaffected in both directions, so the patched file is
behaviourally identical to upstream on Hopper/Blackwell-datacenter parts.

IMPORTANT -- the knob is process-scoped, not request-scoped.
``model_executor/kernels/mhc/tilelang_kernels.py`` evaluates
``ENABLE_PDL = current_platform.is_arch_support_pdl() and ...`` at *import* time
and then branches on that module constant in every mHC kernel.  The mamba/KDA
sites call ``is_arch_support_pdl()`` per launch.  Both therefore agree only if
``HAREM_PDL_SM12`` is fixed for the lifetime of the process: set it in the env
file before the engine starts and never change it mid-run.

Call sites this gate controls in the image we serve (all hot for GLM-5.3-Flash,
34/45 KDA layers, hc_mult 4):
  model_executor/layers/mamba/ops/causal_conv1d.py          (2)  KDA conv
  model_executor/layers/mamba/ops/gather_initial_states.py  (1)  recurrent-state gather
  model_executor/layers/mamba/ops/scatter_states.py         (1)  recurrent-state scatter
  third_party/flash_linear_attention/ops/fused_norm_gate.py (2)  KDA gating
  model_executor/kernels/mhc/tilelang_kernels.py            (1)  -> ENABLE_PDL, 9 branches
(plus non-GLM paths -- minimax_m3, kimi_k3, deepseek -- that we never execute.)

One anchor, the whole function body, so a drifted signature refuses instead of
silently patching the wrong ``return major >= 9``.  Idempotent.

Exit codes: 0 applied/already-applied/check-ok, 2 file missing, 3 anchor
mismatch, 4 precondition failed.
"""

import argparse
import ast
import os
import sys

MARK = "HAREM-SM12 PDL gate"
REL = "platforms/cuda.py"

OLD = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9
"""

NEW = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        # HAREM-SM12 PDL gate: upstream enables Programmatic Dependent Launch
        # for every capability >= 9, which includes GB10 (sm_121, major 12).
        # PDL is not qualified there and KDA recurrent-state races have been
        # reported on SM12x, so sm_12x is excluded by default.
        #   HAREM_PDL_SM12=1  -> restore upstream `major >= 9` (A/B arm).
        # Read per call; the mHC tilelang path caches it at import time, so the
        # variable must be fixed before the engine process starts.
        if os.environ.get("HAREM_PDL_SM12", "0") == "1":
            return major >= 9
        return major in (9, 10)
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
        print(f"patch-pdl-gate: {p} not found", file=sys.stderr)
        return 2

    src = open(p).read()

    if MARK in src:
        print(f"patch-pdl-gate: already applied ({p})")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(
            f"patch-pdl-gate: ANCHOR count={n} (expected 1) in {p} -- refusing.\n"
            f"  The is_arch_support_pdl body has drifted from the image this "
            f"patch was written against. Re-read it before forcing anything.",
            file=sys.stderr,
        )
        return 3

    # `os` must already be importable at module scope; cuda.py imports it, but
    # prove it rather than assume it, because the replacement body uses it.
    if not any(
        line.strip() in ("import os", "import os, sys") for line in src.splitlines()
    ):
        print(
            f"patch-pdl-gate: no module-level 'import os' in {p} -- refusing",
            file=sys.stderr,
        )
        return 4

    out = src.replace(OLD, NEW, 1)

    try:
        ast.parse(out, filename=p)
    except SyntaxError as exc:
        print(f"patch-pdl-gate: patched source does not parse: {exc}", file=sys.stderr)
        return 4

    if a.check:
        print(
            f"patch-pdl-gate: CHECK OK -- anchor unique in {p}, patched source parses"
        )
        return 0

    open(p, "w").write(out)
    print(
        f"patch-pdl-gate: applied to {p}\n"
        f"  is_arch_support_pdl(): sm_12x -> False by default; "
        f"HAREM_PDL_SM12=1 restores upstream 'major >= 9'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

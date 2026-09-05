#!/usr/bin/env python3
"""Say which parts of cuda-exl3's padded-load path this image has, and refuse
the full-scope TP=3 arm if any of them is missing.

At TP=3 the sidecar config pads 64 heads to 66, the vocab 154880 -> 155136 and
the shared expert 2048 -> 2304. With a full-scope EXL3 checkpoint that puts a
pad on an EXL3 output dim (lm_head, q_b_proj, in_proj_qkv, shared gate_up_proj)
and on an EXL3 input dim (o_proj, shared down_proj). Three upstream commits make
that legal, and an image without them fails PART WAY THROUGH the load:

  f3e3090  padded output dim accepted when the pad is whole 128-blocks (svh
           allocated zeroed) + row-parallel `suh` copies what exists and zeros
           the rest instead of narrowing off the end of the checkpoint
  754421f  the vocab loaders fill a prefix instead of copy_ing the unpadded
           slice into the padded parameter

Read by source inspection rather than a version string, because the image is
built from a git checkout and carries no version we control. Exit 0 = go.
"""

import inspect
import sys


def main() -> int:
    try:
        from cuda_exl3.linear import Exl3LinearMethod
        from cuda_exl3.parameter import Exl3SuhParameter
    except Exception as e:  # pragma: no cover
        print(f"[padload] cuda_exl3 import failed: {e}", file=sys.stderr)
        return 22

    checks = {
        "padded-output-gate (f3e3090)":
            "cannot be zero-extended"
            not in inspect.getsource(Exl3LinearMethod.create_weights),
        "vocab-loader-prefix (754421f)":
            "param.data[:n]" in inspect.getsource(Exl3LinearMethod._vocab_loaders),
        "row-parallel-suh-pad (f3e3090)":
            "avail" in inspect.getsource(
                Exl3SuhParameter.load_row_parallel_weight),
    }
    print("[padload] cuda-exl3 padded-load support: "
          + "  ".join(f"{k}={'yes' if v else 'NO'}" for k, v in checks.items()))
    if all(checks.values()):
        return 0
    print(
        "[padload] REFUSING: this image predates the padded-load path (need "
        "cuda-exl3 >= 754421f). At tp=3 the lm_head, o_proj and the shared "
        "expert are all padded; without it the load stops part way -- "
        "62f53e6/5903248 raise 'EXL3 weights cannot be zero-extended' in "
        "create_weights, and f3e3090 alone passes that gate and then dies on a "
        "copy_ shape mismatch in _vocab_loaders. Rebuild the image or run this "
        "arm at tp<=2 (where nothing is padded).",
        file=sys.stderr,
    )
    return 23


if __name__ == "__main__":
    sys.exit(main())

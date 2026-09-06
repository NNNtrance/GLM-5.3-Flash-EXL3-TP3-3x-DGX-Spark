#!/usr/bin/env python3
"""Teach cuda-exl3's routed-expert method to run under expert parallel.

Run inside the serving container before ``vllm serve``. Two things happen:

  1. ``overlay/cuda_exl3/_harem_ep.py`` (next to this script) is copied to
     ``<site-packages>/cuda_exl3/_harem_ep.py``. That file carries all of the
     logic and all of the reasoning.
  2. Four one-line call sites in ``<site-packages>/cuda_exl3/moe.py`` are
     rewired to call it. Exact-text anchors: if cuda-exl3 moves, this exits
     non-zero instead of half-patching.

What the four sites do, in one line each:

  create_weights  pick EP or TP for this layer, refuse a trellis split that is
                  not 128-aligned, and log the mode (this is the boot-log
                  evidence line: ``mode=EP ep_size=3 experts_local=96/288``).
  _place          fail closed if a loaded trellis is not exactly the shape this
                  rank expects. A mis-shaped trellis loads silently and decodes
                  to noise; there is no other check between disk and the kernel.
  apply (block_m) hand the block-size ladder the GLOBAL expert count. Upstream
                  e0a3975 fixed the ALIGNMENT to use the global count and the
                  expert map, but ``_block_m`` still gets the local count, so
                  the rows-per-expert estimate is EP_size times too high and
                  the ladder climbs too early.

The two edits that used to be here -- the alignment fix and the post-gemm
clearing pass -- are both upstream as of e0a3975 and have been retired.

Against the alternative (bind-mounting a whole replacement ``moe.py``): a
mounted copy silently inherits every upstream fix and every upstream bug for as
long as it stays mounted, and nothing tells you it has drifted. Four anchors
that must match exactly do tell you.

Usage:  patch-exl3-ep.py [--check] [--pkg /usr/local/lib/python3.12/dist-packages/cuda_exl3]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

MARKER = "HAREM-TP3"
DEFAULT_PKG = Path("/usr/local/lib/python3.12/dist-packages/cuda_exl3")
OVERLAY = Path(__file__).resolve().parent / "overlay" / "cuda_exl3" / "_harem_ep.py"

EDITS = [
    # 1. create_weights: mode decision + trellis-split refusal + evidence log
    (
        "        E = num_experts\n"
        "        H = hidden_size\n"
        "        I = intermediate_size_per_partition\n"
        "        dev = torch.cuda.current_device()\n",
        "        E = num_experts\n"
        "        H = hidden_size\n"
        "        I = intermediate_size_per_partition\n"
        "        dev = torch.cuda.current_device()\n"
        "\n"
        "        # HAREM-TP3: EP vs TP for this layer; refuses a trellis split that is\n"
        "        # not 128-aligned (2048/3 at TP=3) and names the mode in the log.\n"
        "        from cuda_exl3 import _harem_ep as _harem\n"
        "        layer._harem_ep_on = _harem.check_expert_shape(self.prefix, self.moe, E, H, I)\n",
    ),
    # 2. _place: fail closed on a mis-shaped trellis
    (
        "        r = self._tp_rank\n"
        "        if proj == \"down_proj\":\n",
        "        r = self._tp_rank\n"
        "        if suffix == \"trellis\":  # HAREM-TP3\n"
        "            from cuda_exl3 import _harem_ep as _harem\n"
        "            _harem.check_trellis_slice(self.prefix, layer, proj, w, self._tp_size, r)\n"
        "        if proj == \"down_proj\":\n",
    ),
    # 3. RETIRED at cuda-exl3 77513d2, which now hands _block_m the global
    #    expert count itself. Confirmed on GB10 (one MoE layer, 96 owned of 288,
    #    top_k 8): the ladder's pick and the measured best agree at M=8/64/512
    #    (block 16) and are within 2 % at M=2048 (it picks 64, 32 measures
    #    11.61 vs 11.83 ms). The LOCAL count would have picked 32 at M=512
    #    (+9 %) and 128 at M=2048 (+32 %).
    # 4. RETIRED at cuda-exl3 e0a3975. The gemm and the split-k epilogue now
    #    zero an unowned block's tile instead of returning, so the python
    #    (rows, hidden) clearing pass has nothing left to do. Kept out of the
    #    anchor list deliberately: _harem_ep.zero_remote_rows still exists and
    #    still works, for A/B against an older kernel.
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", type=Path, default=DEFAULT_PKG)
    ap.add_argument("--check", action="store_true",
                    help="verify the anchors still match; change nothing")
    ap.add_argument("--overlay", type=Path, default=OVERLAY)
    args = ap.parse_args()

    moe = args.pkg / "moe.py"
    if not moe.is_file():
        raise SystemExit(f"no cuda_exl3/moe.py at {args.pkg}")
    if not args.overlay.is_file():
        raise SystemExit(f"missing overlay module {args.overlay}")

    print(f"HAREM-TP3 cuda-exl3 EP patch in {args.pkg}")
    src = moe.read_text()

    if MARKER in src:
        print(f"  moe.py: already patched (sha {sha(moe)})")
        if not args.check:
            # Refresh the helper anyway: it is the file that gets iterated on.
            shutil.copyfile(args.overlay, args.pkg / "_harem_ep.py")
            print(f"  _harem_ep.py: refreshed (sha {sha(args.pkg / '_harem_ep.py')})")
        return

    bad = [old for old, _ in EDITS if src.count(old) != 1]
    if bad:
        for old in bad:
            n = src.count(old)
            print(f"  ANCHOR {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}: "
                  f"{old.strip().splitlines()[0][:90]}", file=sys.stderr)
        raise SystemExit(
            f"{moe}: {len(bad)} anchor(s) did not match exactly once -- cuda-exl3 "
            "changed. Re-read moe.py before re-running; do not force this."
        )

    if args.check:
        print(f"  moe.py: {len(EDITS)} anchors OK, NOT patched (--check)")
        return

    for old, new in EDITS:
        src = src.replace(old, new, 1)
    before = sha(moe)
    moe.write_text(src)
    shutil.copyfile(args.overlay, args.pkg / "_harem_ep.py")
    print(f"  _harem_ep.py: installed (sha {sha(args.pkg / '_harem_ep.py')})")
    print(f"  moe.py: patched {len(EDITS)} sites ({before} -> {sha(moe)})")
    print("HAREM-TP3: cuda-exl3 EP patch applied")


if __name__ == "__main__":
    main()

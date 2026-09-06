#!/usr/bin/env python3
"""HAREM-TP3: let the independently grouped sliding-window spec (DFlash2 draft) use a
larger kernel block than the backend minimum.  Env HAREM_SW_BLOCK_SIZE=<int> (0/unset =
upstream behaviour).  Single anchor in vllm/model_executor/layers/attention/attention.py;
fails closed if the anchor is missing or not unique.  Idempotent."""
import argparse, sys, os
OLD = """            sw_block_size = _largest_kernel_block_within(
                self.attn_backend, sw_per_token, shared_page, block_size
            )
"""
NEW = """            # HAREM-TP3: an independently grouped SW spec (DFlash draft) never
            # reaches ``unify``, so upstream's "smallest is fine" no longer holds.
            _harem_sw = int(os.environ.get("HAREM_SW_BLOCK_SIZE", "0") or 0)
            if _harem_sw:
                from vllm.v1.attention.backend import MultipleOf as _HaremMultipleOf
                _harem_base = min(
                    s.base if isinstance(s, _HaremMultipleOf) else s
                    for s in self.attn_backend.get_supported_kernel_block_sizes()
                )
                if _harem_sw % _harem_base:
                    raise ValueError(
                        f"HAREM_SW_BLOCK_SIZE={_harem_sw} is not a multiple of "
                        f"{self.attn_backend.get_name()}'s kernel block {_harem_base}"
                    )
                sw_block_size = _harem_sw
            else:
                sw_block_size = _largest_kernel_block_within(
                    self.attn_backend, sw_per_token, shared_page, block_size
                )
"""
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
p = os.path.join(a.root, "model_executor/layers/attention/attention.py")
src = open(p).read()
if "HAREM-TP3: an independently grouped SW spec" in src:
    print(f"patch-swblock: already applied ({p})"); sys.exit(0)
n = src.count(OLD)
if n != 1:
    print(f"patch-swblock: ANCHOR count={n} (expected 1) in {p}", file=sys.stderr); sys.exit(3)
if "from vllm.v1.attention.backend import MultipleOf" not in src:
    print("patch-swblock: MultipleOf import path not found in attention.py — refusing", file=sys.stderr); sys.exit(4)
out = src.replace(OLD, NEW)
if not any(l.startswith("import os") or l.startswith("import os,") for l in out.splitlines()):
    out = "import os\n" + out
open(p, "w").write(out)
print(f"patch-swblock: applied to {p} (HAREM_SW_BLOCK_SIZE honoured)")

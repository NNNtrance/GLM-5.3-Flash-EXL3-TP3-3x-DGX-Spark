#!/usr/bin/env python3
"""Make the KV pool arithmetic visible. Logging only -- no behaviour change.

vLLM reports one number, "GPU KV cache size: N tokens", and that number is not
what most people assume. From ``v1/core/kv_cache_utils.py``:

    num_tokens      = int(max_concurrency * max_model_len)
    max_concurrency = num_blocks / num_blocks_per_request
    num_blocks_per_request = SUM over groups of
                             cdiv(group.max_memory_usage_bytes, group.page_size_bytes)

So the reported pool has two independent inputs, and they fail in opposite ways:

  * ``num_blocks`` -- how many blocks the free memory buys. A group with a tiny
    per-block cost barely moves this. The DFlash2 draft group costs
    5 x 32,768 = 163,840 B per block against the target's 27,624,960 B: +0.6 %.

  * ``num_blocks_per_request`` -- how many blocks ONE max-length request needs,
    summed over groups because block ids are global to a single pool. A group
    whose per-request demand is large relative to its page size inflates this
    without costing memory, and every point of inflation divides the reported
    pool directly.

That asymmetry is why the DFlash2 port could correctly measure "+0.6 % per
block" and still watch the pool fall from 1,987,179 to 825,000 tokens. The two
numbers describe different things and only the second one moved.

This patch logs the per-group decomposition of ``num_blocks_per_request`` at
boot, so the pool is an arithmetic statement rather than an opaque number:

    HAREM-TP3 KV pool breakdown: num_blocks=..., blocks/request=...
      <SpecType>: N layer(s), page ... B, bytes/block ... B, max/req ... B -> ... block(s)

Run it in the prelude alongside the other TP=3 patches. It is safe at any TP and
with or without speculation.

Usage:  patch-kvdiag-tp3.py [--check] [--root /usr/local/lib/python3.12/dist-packages/vllm]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MARKER = "HAREM-TP3 KV pool breakdown"
DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")

OLD = '''    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency
'''

# raw: the "\n" below must reach the generated file as an escape.
NEW = r'''    # --- HAREM-TP3: decompose the sum instead of collapsing it to one number.
    # The reported pool is num_blocks / num_blocks_per_request * max_model_len,
    # so a group can shrink it drastically while costing almost no memory --
    # it only has to demand many blocks for a single request. Print both sides.
    _harem_rows = []
    num_blocks_per_request = 0
    for _harem_group in kv_cache_config.kv_cache_groups:
        _harem_spec = _harem_group.kv_cache_spec
        _harem_need = _harem_spec.max_memory_usage_bytes(vllm_config)
        _harem_page = _harem_spec.page_size_bytes
        _harem_blocks = cdiv(_harem_need, _harem_page)
        num_blocks_per_request += _harem_blocks
        _harem_layers = len(getattr(_harem_group, "layer_names", ()) or ())
        _harem_rows.append(
            "%s: %d layer(s), page %s B, bytes/block %s B, max/req %s B -> %s block(s)"
            % (
                type(_harem_spec).__name__,
                _harem_layers,
                f"{_harem_page:,}",
                f"{_harem_page * _harem_layers:,}",
                f"{_harem_need:,}",
                f"{_harem_blocks:,}",
            )
        )
    logger.info(
        "HAREM-TP3 KV pool breakdown: num_blocks=%s, blocks/request=%s, "
        "max_model_len=%s\n  %s",
        f"{kv_cache_config.num_blocks:,}",
        f"{num_blocks_per_request:,}",
        f"{vllm_config.model_config.max_model_len:,}",
        "\n  ".join(_harem_rows),
    )
    # --- end HAREM-TP3
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency
'''

FILES = [("v1/core/kv_cache_utils.py", [(OLD, NEW)])]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


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
        print(f"  {path.name}: {len(edits)} anchors OK, NOT patched (--check)")
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
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"no vllm package at {args.root}")
    print(f"HAREM-TP3 KV pool diagnostic in {args.root}")
    changed = 0
    for rel, edits in FILES:
        path = args.root / rel
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        changed += bool(apply_file(path, edits, args.check))
    print(
        "HAREM-TP3 KV diag: "
        + ("anchor verified" if args.check else f"{changed} file(s) changed")
    )


if __name__ == "__main__":
    main()

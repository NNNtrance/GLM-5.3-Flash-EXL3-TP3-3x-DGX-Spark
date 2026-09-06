#!/usr/bin/env python3
"""HAREM-TP3: refuse a stale fastload sidecar before vLLM even starts.

Runs inside the prelude, so a sidecar produced from another checkpoint, another
image, another patch set or another TP/EP arrangement stops the rank in a
second instead of silently serving four-minute-old weights.

Cheap on purpose: no torch, no vllm.  The authoritative check runs again inside
the engine (harem_fastload._restore) where the vllm/cuda-exl3 versions and the
hf overrides are known; this one is the early, loud half of the same test.

Exit codes:  0 ok / not requested   ·   30 sidecar missing   ·   31 stale
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harem_fastload_id as hid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=os.environ.get("HAREM_FASTLOAD_DRAFT_PATH", ""))
    a = ap.parse_args()

    mode = os.environ.get("HAREM_FASTLOAD_MODE", "").strip().lower()
    if mode != "load":
        print(f"preflight-fastload: mode={mode or 'kapali'} - kontrol gerekmiyor")
        return 0

    base = os.environ.get("HAREM_FASTLOAD_DIR", "").strip()
    rank = os.environ.get("NODE_RANK", "")
    if not base or rank == "":
        print("preflight-fastload: HAREM_FASTLOAD_DIR / NODE_RANK eksik", file=sys.stderr)
        return 31
    d = f"{base}-r{rank}"
    man_path = os.path.join(d, "MANIFEST.json")
    if not os.path.isfile(man_path):
        print(f"preflight-fastload: {man_path} missing - run a dump boot first",
              file=sys.stderr)
        return 30
    man = json.load(open(man_path))

    if str(man.get("rank")) != str(rank):
        print(f"preflight-fastload: manifest rank={man.get('rank')} != NODE_RANK={rank}",
              file=sys.stderr)
        return 31

    now = hid.file_identity(a.model, a.draft, os.environ.get("TP3_DIR", "/opt/harem-tp3"))
    saved = dict(man.get("identity", {}))
    saved.pop("engine", None)          # only the engine itself can check those
    problems = hid.diff(saved, now)
    if problems:
        print("preflight-fastload: SIDECAR STALE - boot refused", file=sys.stderr)
        for line in problems[:20]:
            print(f"  {line}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... ve {len(problems) - 20} fark daha", file=sys.stderr)
        print("  To regenerate: one boot with HAREM_FASTLOAD_MODE=dump.", file=sys.stderr)
        return 31

    total = 0
    for tag, m in man.get("models", {}).items():
        for shard in m["shards"]:
            p = os.path.join(d, shard)
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                print(f"preflight-fastload: eksik/bos parca {p}", file=sys.stderr)
                return 30
        total += m["bytes"]
    print(
        f"preflight-fastload: OK  rank={rank} dir={d} "
        f"modeller={sorted(man.get('models', {}))} "
        f"{total / (1 << 30):.2f} GiB  uretim={man.get('created_utc')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""HAREM-TP3: teach vLLM's EP weight filter about EXL3's heavy per-expert tensor.

Upstream ``should_skip_weight`` only skips names ending in ``.weight`` /
``.weight_packed`` (vllm/model_executor/model_loader/ep_weight_filter.py:79).
An EXL3 checkpoint stores the bulk of an expert in ``<proj>.trellis`` -- 99.8 %
of the expert bytes -- so with EP on and the filter enabled upstream still reads
every expert of every rank.  This adds the EXL3 suffixes to that tuple.

Which suffixes are skipped is env-driven so the set can be widened without a new
patch:  HAREM_EP_FILTER_SUFFIXES=".trellis"  (default; comma separated).
Set it to "" to make the patch a no-op at runtime.

Only ``.trellis`` is skipped by default.  ``.suh``/``.svh``/``.mcg`` are the
per-expert scales; they are tiny, and a backend that wanted a global reduction
over all experts' scales would break silently if they went missing.

Single anchor in ep_weight_filter.py; fails closed if it is missing or not
unique.  Idempotent.
"""
import argparse
import os
import sys

MARK = "HAREM-TP3: EXL3 stores the bulk of an expert"

OLD = """    if not weight_name.endswith((".weight", ".weight_packed")):
        return False
"""

NEW = '''    # HAREM-TP3: EXL3 stores the bulk of an expert in "<proj>.trellis", not in
    # ".weight", so upstream's suffix gate lets every expert through.  The extra
    # suffixes are env-driven (HAREM_EP_FILTER_SUFFIXES, comma separated) and
    # default to ".trellis" only -- the tiny .suh/.svh/.mcg scale tensors are
    # deliberately still read for every expert.
    if not weight_name.endswith(_HAREM_SKIPPABLE_SUFFIXES):
        return False
'''

HELPER = '''

# HAREM-TP3 -------------------------------------------------------------------
def _harem_suffixes() -> tuple[str, ...]:
    raw = os.environ.get("HAREM_EP_FILTER_SUFFIXES", ".trellis")
    extra = tuple(s.strip() for s in raw.split(",") if s.strip())
    for s in extra:
        if not s.startswith("."):
            raise ValueError(
                f"HAREM_EP_FILTER_SUFFIXES entry {s!r} must start with '.'"
            )
    return (".weight", ".weight_packed") + extra


_HAREM_SKIPPABLE_SUFFIXES = _harem_suffixes()
_HAREM_SKIPPED = 0
# -----------------------------------------------------------------------------
'''

COUNTER_OLD = """    return eid not in local_expert_ids
"""

COUNTER_NEW = '''    skip = eid not in local_expert_ids
    if skip:
        global _HAREM_SKIPPED
        _HAREM_SKIPPED += 1
        if _HAREM_SKIPPED % 5000 == 0:
            print(
                f"[harem-epfilter] skipped {_HAREM_SKIPPED} non-local expert "
                f"tensors (suffixes {_HAREM_SKIPPABLE_SUFFIXES})",
                flush=True,
            )
    return skip
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path of the vllm package")
    a = ap.parse_args()
    p = os.path.join(a.root, "model_executor/model_loader/ep_weight_filter.py")
    if not os.path.isfile(p):
        print(f"patch-epfilter: {p} not found", file=sys.stderr)
        return 2
    src = open(p).read()
    if MARK in src:
        print(f"patch-epfilter: already applied ({p})")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(
            f"patch-epfilter: ANCHOR count={n} (expected 1) in {p}", file=sys.stderr
        )
        return 3
    c = src.count(COUNTER_OLD)
    if c != 1:
        print(
            f"patch-epfilter: RETURN anchor count={c} (expected 1) in {p}",
            file=sys.stderr,
        )
        return 4
    if "def should_skip_weight(" not in src:
        print("patch-epfilter: should_skip_weight missing - refusing", file=sys.stderr)
        return 5
    out = src.replace(OLD, NEW).replace(COUNTER_OLD, COUNTER_NEW)
    # helper goes right before should_skip_weight so the module-level constant
    # exists by the time the function body runs.
    marker = "def should_skip_weight("
    out = out.replace(marker, HELPER.lstrip("\n") + "\n" + marker, 1)
    if not any(
        l.strip() == "import os" or l.startswith("import os,") for l in out.splitlines()
    ):
        out = "import os\n" + out
    open(p, "w").write(out)
    print(f"patch-epfilter: applied to {p} (EXL3 .trellis is now skippable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

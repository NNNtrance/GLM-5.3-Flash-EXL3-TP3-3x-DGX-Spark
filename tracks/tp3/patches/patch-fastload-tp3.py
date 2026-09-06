#!/usr/bin/env python3
"""HAREM-TP3: install the per-rank fastload sidecar hook into vLLM's model loader.

Copies ``harem_fastload.py`` + ``harem_fastload_id.py`` (siblings of this script)
into ``vllm/model_executor/model_loader/`` and routes
``BaseModelLoader.load_model`` through them.  With ``HAREM_FASTLOAD_MODE`` unset
the hook is a straight call to the original ``self.load_weights(...)``, so the
patch is inert unless the env asks for it.

Two anchors in vllm/model_executor/model_loader/base_loader.py; fails closed if
either is missing or not unique.  Idempotent.
"""

import argparse
import os
import shutil
import sys

MARK = "HAREM-TP3 fastload"

OLD_LOAD = """            logger.debug("Loading weights on %s ...", load_device)
            self.load_weights(model, model_config)
"""

NEW_LOAD = """            logger.debug("Loading weights on %s ...", load_device)
            # HAREM-TP3 fastload: with HAREM_FASTLOAD_MODE unset this is exactly
            # self.load_weights(model, model_config). With "dump" it also writes
            # the rank's post-load tensors to a sidecar; with "load" it restores
            # them from that sidecar instead of re-slicing the full checkpoint.
            from vllm.model_executor.model_loader import (
                harem_fastload as _harem_fastload,
            )

            _harem_fastload.load_weights_hook(self, model, model_config)
"""

OLD_POST = """            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
"""

NEW_POST = """            process_weights_after_loading(model, model_config, target_device)
            # HAREM-TP3 fastload: optional post-processing state hashes, so the
            # "sidecar == full checkpoint" claim can be checked on the tensors
            # the kernels actually read, not only on the freshly loaded ones.
            _harem_fastload.after_process_hook(model, model_config)

        return model.eval()
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path of the vllm package")
    ap.add_argument("--src", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    pkg = os.path.join(a.root, "model_executor/model_loader")
    if not os.path.isdir(pkg):
        print(f"patch-fastload: {pkg} not found", file=sys.stderr)
        return 2
    for mod in ("harem_fastload.py", "harem_fastload_id.py"):
        src = os.path.join(a.src, mod)
        if not os.path.isfile(src):
            print(f"patch-fastload: {src} missing", file=sys.stderr)
            return 2
        shutil.copyfile(src, os.path.join(pkg, mod))

    p = os.path.join(pkg, "base_loader.py")
    src = open(p).read()
    if MARK in src:
        print(f"patch-fastload: already applied ({p}); modules refreshed")
        return 0
    n1, n2 = src.count(OLD_LOAD), src.count(OLD_POST)
    if n1 != 1 or n2 != 1:
        print(
            f"patch-fastload: ANCHOR counts load={n1} post={n2} (expected 1/1) in {p}",
            file=sys.stderr,
        )
        return 3
    out = src.replace(OLD_LOAD, NEW_LOAD).replace(OLD_POST, NEW_POST)
    open(p, "w").write(out)
    print(f"patch-fastload: applied to {p} (HAREM_FASTLOAD_MODE honoured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

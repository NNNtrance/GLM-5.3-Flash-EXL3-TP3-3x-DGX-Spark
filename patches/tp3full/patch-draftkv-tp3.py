#!/usr/bin/env python3
"""HAREM-TP3: set the DFlash2 drafter's KV precision from the environment.

WHERE THE DRAFT'S ``auto`` COMES FROM
-------------------------------------
``vllm/config/speculative.py:127``::

    kv_cache_dtype: CacheDType | None = None
    \"\"\"KV cache dtype for the draft model. When `None`, the draft inherits the
    target model's `--kv-cache-dtype`.\"\"\"

so ``None`` would already inherit the main ``fp8``.  It is our own launcher that
pins it to bf16 -- ``tp3/start-tp3.sh:71``::

    SPEC_ARG=(--speculative-config "{\\"method\\":\\"dflash\\",\\"model\\":\\"${DRAFT_PATH}\\",
      \\"num_speculative_tokens\\":${SPEC_TOKENS},\\"kv_cache_dtype\\":\\"auto\\"}")

and that literal ``"auto"`` is the single reason the engine reports
mixed-precision KV.  The value is consumed at
``v1/worker/gpu/spec_decode/dflash/utils.py:30-38``, which builds the drafter's
own ``cache_config`` from it, so it governs the draft attention layers' KV spec.

WHAT THIS PATCH DOES
--------------------
Overrides ``SpeculativeConfig.kv_cache_dtype`` from ``HAREM_DRAFT_KV_DTYPE`` at
the top of ``__post_init__`` -- i.e. before every validation and before any
consumer reads it -- so it works no matter how the launcher built its JSON and
without editing the launcher at all (rollback = env only).

  ``HAREM_DRAFT_KV_DTYPE`` unset  -> upstream behaviour (whatever the JSON says).
  ``HAREM_DRAFT_KV_DTYPE=fp8``    -> drafter matches the main groups.

The one-token alternative, kept here for the record: change ``start-tp3.sh:71``
to ``\\"kv_cache_dtype\\":\\"${DRAFT_KV_CACHE_DTYPE:-auto}\\"``.  It is smaller, but
it edits the production launcher, so rollback stops being "env only".

EXPECTED EFFECT AND RISK
------------------------
KV pool: ``KV-AMELIYAT-PLAN.md`` section 6(a) measured draft fp8 ALONE at
per-block 20,074,240 -> 20,012,800 B, blocks 1,756 -> 1,761, blocks/request
unchanged at 723 -> pool 2,428,769 -> 2,435,684 tokens, **+0.3 %**.  That is the
point of the arm only as a means to uniform precision, not as a memory win.
Risk: the drafter now attends over fp8 KV, so its proposals can get worse and
the acceptance rate can fall.  The band to watch is 60-65 % (today 61-64 %);
the cold/warm gates (10/10 correctness, 12/12 code) catch a broken drafter, the
acceptance line in the sweep catches a merely weaker one.

Single anchor in ``vllm/config/speculative.py``; fails closed if it is missing
or not unique.  Idempotent.
"""

import argparse
import os
import sys

MARK = "HAREM-TP3: force the drafter's KV precision"

OLD = '''    def __post_init__(self):
        # Note: "method" is a new parameter that helps to extend the
'''

NEW = '''    def __post_init__(self):
        # HAREM-TP3: force the drafter's KV precision from the environment, so
        # the engine's KV dtypes can be made uniform without editing the
        # launcher's --speculative-config JSON.  Unset -> upstream behaviour.
        # Done first, so every validation below sees the overridden value.
        import os as _harem_os

        _harem_kv = _harem_os.environ.get("HAREM_DRAFT_KV_DTYPE", "").strip()
        if _harem_kv:
            _harem_allowed = (
                "auto",
                "bfloat16",
                "float16",
                "fp8",
                "fp8_e4m3",
                "fp8_e5m2",
            )
            if _harem_kv not in _harem_allowed:
                raise ValueError(
                    f"HAREM_DRAFT_KV_DTYPE={_harem_kv!r} is not one of "
                    f"{_harem_allowed}"
                )
            if self.kv_cache_dtype != _harem_kv:
                logger.warning(
                    "HAREM-TP3: draft kv_cache_dtype %r -> %r "
                    "(HAREM_DRAFT_KV_DTYPE)",
                    self.kv_cache_dtype,
                    _harem_kv,
                )
            self.kv_cache_dtype = _harem_kv
        # Note: "method" is a new parameter that helps to extend the
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path of the vllm package")
    a = ap.parse_args()
    p = os.path.join(a.root, "config/speculative.py")
    if not os.path.isfile(p):
        print(f"patch-draftkv: {p} not found", file=sys.stderr)
        return 2
    src = open(p).read()
    if MARK in src:
        print(f"patch-draftkv: already applied ({p})")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"patch-draftkv: ANCHOR count={n} (expected 1) in {p}", file=sys.stderr)
        return 3
    if "    kv_cache_dtype: CacheDType | None = None\n" not in src:
        print(
            "patch-draftkv: kv_cache_dtype field not found - refusing",
            file=sys.stderr,
        )
        return 4
    if "logger = init_logger(__name__)" not in src:
        print("patch-draftkv: module logger not found - refusing", file=sys.stderr)
        return 5
    open(p, "w").write(src.replace(OLD, NEW))
    print(f"patch-draftkv: applied to {p} (HAREM_DRAFT_KV_DTYPE honoured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

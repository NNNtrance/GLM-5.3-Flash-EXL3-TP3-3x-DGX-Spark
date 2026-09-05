#!/usr/bin/env python3
"""Behavioural checks for patch-draftkv-tp3.py and patch-tilelang-failloud-tp3.py.
CPU only; no engine, no GPU, no model."""
import contextlib
import importlib.abc
import os
import re
import sys
import textwrap

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")


# ---------------------------------------------------------------- draft kv ---
from vllm.config.speculative import SpeculativeConfig  # noqa: E402


def post_init_with(env_value, start="auto"):
    os.environ.pop("HAREM_DRAFT_KV_DTYPE", None)
    if env_value is not None:
        os.environ["HAREM_DRAFT_KV_DTYPE"] = env_value
    o = object.__new__(SpeculativeConfig)
    for k, v in dict(
        kv_cache_dtype=start, method="dflash", model="/x", num_speculative_tokens=7
    ).items():
        object.__setattr__(o, k, v)
    exc = None
    try:
        SpeculativeConfig.__post_init__(o)
    except Exception as e:  # everything after our block is expected to fail
        exc = e
    return getattr(o, "kv_cache_dtype", None), exc


v, _ = post_init_with(None)
check("draftkv: unset env leaves kv_cache_dtype untouched", v == "auto", f"got {v!r}")

v, _ = post_init_with("fp8")
check("draftkv: HAREM_DRAFT_KV_DTYPE=fp8 overrides 'auto'", v == "fp8", f"got {v!r}")

v, exc = post_init_with("float8_hocus_pocus")
check(
    "draftkv: invalid value raises ValueError",
    isinstance(exc, ValueError) and "HAREM_DRAFT_KV_DTYPE" in str(exc),
    f"{type(exc).__name__}: {str(exc)[:90]}",
)

# ------------------------------------------------------------ tilelang loud ---
TL = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang_kernels.py"
text = open(TL).read()
m = re.search(
    r"^    # HAREM-TP3: upstream hides this preload.*?"
    r"^        with contextlib\.suppress\(Exception\):\n"
    r"            import flashinfer\.comm  # noqa: F401$",
    text,
    re.S | re.M,
)
assert m, "patched import block not found"
BLOCK = textwrap.dedent(m.group(0))
print(f"import block extracted: {len(BLOCK)} chars")


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flashinfer.comm":
            raise ImportError("simulated flashinfer.comm failure")
        return None


def run_block(env, block_import):
    os.environ.pop("HAREM_TILELANG_FAILLOUD", None)
    if env is not None:
        os.environ["HAREM_TILELANG_FAILLOUD"] = env
    for mod in [k for k in sys.modules if k.startswith("flashinfer")]:
        del sys.modules[mod]
    if block_import:
        sys.meta_path.insert(0, Blocker())
    try:
        exec(BLOCK, {"os": os, "contextlib": contextlib})  # noqa: S102
        return None
    except Exception as e:
        return e
    finally:
        if block_import:
            sys.meta_path.pop(0)


e = run_block(None, block_import=True)
check("failloud: unset env still suppresses a broken import", e is None, repr(e))

e = run_block("0", block_import=True)
check("failloud: =0 still suppresses a broken import", e is None, repr(e))

e = run_block("1", block_import=True)
check(
    "failloud: =1 raises on a broken import",
    isinstance(e, ImportError) and "simulated" in str(e),
    f"{type(e).__name__}: {e}",
)

e = run_block("1", block_import=False)
check("failloud: =1 with a working import returns cleanly", e is None, repr(e))

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)

#!/usr/bin/env python3
"""Exercise the shipped _harem_skip_attention_kv_zeroing gate on synthetic
KV cache configs.  CPU only; no engine, no GPU, no model."""
import os
import re
import sys

import torch

from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

SRC = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py"

# Pull the shipped helper out of the patched file and exec it verbatim, so the
# test runs the exact code the engine will run.
text = open(SRC).read()
m = re.search(
    r"^def _harem_skip_attention_kv_zeroing.*?^    return True$",
    text,
    re.S | re.M,
)
assert m, "helper not found in patched model_runner.py"
ns: dict = {}
exec(m.group(0), ns)  # noqa: S102
gate = ns["_harem_skip_attention_kv_zeroing"]
print(f"helper extracted: {len(m.group(0))} chars")


def spec(cls, **kw):
    s = object.__new__(cls)
    for k, v in kw.items():
        object.__setattr__(s, k, v)
    return s


class Grp:
    def __init__(self, layer_names, kv_cache_spec):
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec


class Tensor_:
    def __init__(self, shared_by):
        self.shared_by = shared_by


class Cfg:
    def __init__(self, groups, tensors, mixed):
        self.kv_cache_groups = groups
        self.kv_cache_tensors = tensors
        self.has_mixed_precision_kv_cache = mixed


FP8 = dict(dtype=torch.float8_e4m3fn, kv_quant_mode=1)
BF16 = dict(dtype=torch.bfloat16, kv_quant_mode=0)

mla = Grp(["mla.0"], spec(AttentionSpec, **FP8))
idx = Grp(["idx.0"], spec(AttentionSpec, **FP8))
draft_bf16 = Grp(["draft.0"], spec(AttentionSpec, **BF16))
draft_fp8 = Grp(["draft.0"], spec(AttentionSpec, **FP8))
mamba = Grp(["mamba.0"], spec(MambaSpec))

SHARED = [Tensor_(["mla.0", "mamba.0"]), Tensor_(["idx.0"]), Tensor_(["draft.0"])]
UNSHARED = [Tensor_(["mla.0"]), Tensor_(["idx.0"]), Tensor_(["draft.0"])]

CASES = [
    # (name, env, cfg, expect)  expect: True/False or a substring of the raise
    ("A unset env, production shape", None,
     Cfg([mla, idx, draft_bf16, mamba], SHARED, True), False),
    ("B env=1 (explicit on)", "1",
     Cfg([mla, idx, draft_bf16, mamba], SHARED, True), False),
    ("C env=0, MIXED precision (today)", "0",
     Cfg([mla, idx, draft_bf16, mamba], SHARED, True), "ONE KV precision"),
    ("D env=0, uniform fp8 but mamba shares MLA tensor", "0",
     Cfg([mla, idx, draft_fp8, mamba], SHARED, False), "co-owned by"),
    ("E env=0, uniform fp8, mamba present but NOT sharing", "0",
     Cfg([mla, idx, draft_fp8, mamba], UNSHARED, False), True),
    ("F env=0, uniform fp8, no mamba at all", "0",
     Cfg([mla, idx, draft_fp8], UNSHARED, False), True),
    ("G env=0, uniform but upstream says mixed (detectors disagree)", "0",
     Cfg([mla, idx, draft_fp8], UNSHARED, True), "detectors disagree"),
    ("H env=0, no attention group at all", "0",
     Cfg([mamba], [Tensor_(["mamba.0"])], False), "no AttentionSpec group"),
]

fails = 0
for name, env, cfg, expect in CASES:
    os.environ.pop("HAREM_ZERO_ATTENTION_KV", None)
    if env is not None:
        os.environ["HAREM_ZERO_ATTENTION_KV"] = env
    try:
        got = gate(cfg)
        ok = got is expect
        print(f"{'PASS' if ok else 'FAIL'}  {name}: returned {got}")
    except RuntimeError as e:
        ok = isinstance(expect, str) and expect in str(e)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: raised -> {str(e)[:150]}")
    fails += not ok

print(f"\n{len(CASES) - fails}/{len(CASES)} cases passed")
sys.exit(1 if fails else 0)

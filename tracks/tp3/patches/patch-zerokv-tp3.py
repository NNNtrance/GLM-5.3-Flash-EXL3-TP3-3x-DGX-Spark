#!/usr/bin/env python3
"""HAREM-TP3: fail-closed ``zero_attention`` gate for the per-step KV memset.

WHY THE KERNEL RUNS AT ALL
--------------------------
``vllm/v1/kv_cache_interface.py:1128-1137``::

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        \"\"\"Whether newly allocated KV cache blocks must be zeroed before use.

        Required for Mamba layers, whose state is read before it is fully written
        (#35219), and for mixed-precision caches, where a block reused across
        groups can be reinterpreted under a different precision and decode stale
        bytes to NaN/Inf. Uniform-precision caches skip zeroing.
        \"\"\"
        return self.has_mamba_layers or self.has_mixed_precision_kv_cache

Both disjuncts are true for us today: 34 KDA/Mamba layers, and the DFlash2 draft
group runs at ``kv_cache_dtype: "auto"`` (bf16) while the main groups run fp8.
Cost, measured 5 Sep 2026 (``HC-MIXING-ANALIZ.md`` section 5): 13.5-15.6 ms per
2048-token prefill chunk, i.e. 1.2-1.4 % of the chunk, running at 100 % of the
memset roofline -- the kernel is perfect, the volume is the problem.

WHAT MAY BE SKIPPED, AND WHY IT IS SAFE WHEN IT IS
--------------------------------------------------
Only AttentionSpec layers ever produce work:

  * the scheduler records new block ids only for AttentionSpec managers --
    ``single_type_kv_cache_manager.py:91``
    ``self._record_new_block_ids = needs_kv_cache_zeroing and isinstance(
    kv_cache_spec, AttentionSpec)``
  * the zeroer builds segments only for AttentionSpec groups --
    ``v1/worker/utils.py:161`` ``if not isinstance(spec, AttentionSpec): continue``

So the whole mechanism means exactly one thing: "wipe a physical block before an
attention group reads it".  Under UNIFORM precision the bytes a recycled block
still carries were written by the same kernel in the same layout, so the
masked-out tail of a partially filled block decodes to finite garbage, never to
NaN/Inf.  That is upstream's own argument, in upstream's own words: "Uniform-
precision caches skip zeroing."

WHAT WOULD BREAK IF THE ARGUMENT WERE WRONG
-------------------------------------------
Stale bytes reinterpreted under a different precision (or a different layout)
decode to NaN/Inf; one NaN poisons the whole attention row through the softmax,
and the model keeps answering -- fluently and wrongly, with no error anywhere.
That failure mode is invisible to the health check and often survives the short
gates.  Hence: default OFF, three independent conditions, and a raise (not a
warning) at startup when the knob is set but a condition does not hold.

THE MAMBA CONDITION IS NOT OPTIONAL
-----------------------------------
On GLM-5-Next / DeepseekV4 the MLA tensor and the Mamba/KDA state are ONE
allocation -- ``v1/core/kv_cache_utils.py:1640-1678``::

    # GLM-5-Next hybrid slot sharing: one tensor per MLA layer, co-owned
    # by that MLA layer and one mamba layer from each mamba group; ...
    # (the tail parasitizes the indexer tensor at disjoint block-ids, like
    # mamba parasitizes MLA ...)
    KVCacheTensor(size=mla_page * num_blocks,
                  shared_by=[mla_name] + [one layer per mamba group])

A block id freed by a mamba group and re-issued to the MLA group therefore
carries 1.7 MB of arbitrary SSM state that MLA will read as latents.  That is
the ``has_mamba_layers`` half of upstream's OR and it is INDEPENDENT of
precision uniformity: making the drafter fp8 does not make it go away.  So a
config in which any AttentionSpec layer shares its KVCacheTensor with a
MambaSpec layer can never skip zeroing.  This gate refuses such a config out
loud instead of silently doing nothing.

ENV
---
``HAREM_ZERO_ATTENTION_KV`` unset or "1" -> upstream behaviour, byte for byte.
``HAREM_ZERO_ATTENTION_KV`` = "0"        -> request the skip; the engine proves
                                            all conditions or refuses to serve.

Single anchor in ``vllm/v1/worker/gpu/model_runner.py`` (``_init_kv_zero_meta``,
the only KVBlockZeroer construction site on the V1 GPU path this image uses --
boot logs show the ``[model_runner.py:NNNN]`` prefix, not ``gpu_model_runner``).
Fails closed if the anchor is missing or not unique.  Idempotent.
"""

import argparse
import os
import sys

MARK = "HAREM-TP3: fail-closed zero_attention gate"

LOGGER_ANCHOR = "logger = init_logger(__name__)\n"

OLD = '''    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=self.compilation_config.static_forward_context,
            num_blocks=self.kv_cache_config.num_blocks,
        )
'''

NEW = '''    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=self.compilation_config.static_forward_context,
            num_blocks=self.kv_cache_config.num_blocks,
        )
        # HAREM-TP3: fail-closed zero_attention gate (patch-zerokv-tp3.py).
        # The object is kept and only its segment table is dropped, so the
        # ``assert self.kv_block_zeroer is not None`` at the call site stays
        # valid and both zero_block_ids() and warmup() become no-ops.
        if _harem_skip_attention_kv_zeroing(self.kv_cache_config):
            self.kv_block_zeroer._meta = None
            logger.warning(
                "HAREM-TP3: attention KV zeroing DISABLED "
                "(HAREM_ZERO_ATTENTION_KV=0; uniform KV precision proved and no "
                "Mamba/KDA layer shares an attention KVCacheTensor)."
            )
'''

HELPER = '''

# HAREM-TP3 -------------------------------------------------------------------
def _harem_skip_attention_kv_zeroing(kv_cache_config) -> bool:
    """HAREM-TP3: fail-closed zero_attention gate -- see patch-zerokv-tp3.py.

    Returns False (upstream behaviour) unless HAREM_ZERO_ATTENTION_KV=0.  When
    it is set, every safety condition must be provable from the KV cache config;
    a condition that does not hold raises HERE, at startup, rather than letting
    the engine serve a model that can answer fluently and wrongly.
    """
    import os

    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )

    if os.environ.get("HAREM_ZERO_ATTENTION_KV", "1") != "0":
        return False

    def _flatten(group_spec):
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            return list(group_spec.kv_cache_specs.values())
        return [group_spec]

    # (1) One KV precision across every attention group -- main AND draft.
    precisions: dict = {}
    attn_layers: set = set()
    mamba_layers: set = set()
    for group in kv_cache_config.kv_cache_groups:
        for spec in _flatten(group.kv_cache_spec):
            if isinstance(spec, MambaSpec):
                mamba_layers.update(group.layer_names)
            elif isinstance(spec, AttentionSpec):
                attn_layers.update(group.layer_names)
                key = (str(spec.dtype), int(spec.kv_quant_mode))
                precisions.setdefault(key, set()).add(type(spec).__name__)
    if not precisions:
        raise RuntimeError(
            "HAREM_ZERO_ATTENTION_KV=0: no AttentionSpec group found -- refusing "
            "to reason about a KV layout this gate does not recognise."
        )
    if len(precisions) != 1:
        detail = "; ".join(
            f"dtype={k[0]} kv_quant_mode={k[1]} <- {sorted(v)}"
            for k, v in sorted(precisions.items())
        )
        raise RuntimeError(
            "HAREM_ZERO_ATTENTION_KV=0 requires ONE KV precision across all "
            f"attention groups (drafter included); the engine has {len(precisions)}: "
            f"{detail}.  Make the drafter match the main groups "
            "(HAREM_DRAFT_KV_DTYPE=fp8) or unset HAREM_ZERO_ATTENTION_KV."
        )
    # Cross-check against upstream's own detector, so a future refactor of
    # has_mixed_precision_kv_cache cannot drift past this gate unnoticed.
    if kv_cache_config.has_mixed_precision_kv_cache:
        raise RuntimeError(
            "HAREM-TP3: upstream reports mixed-precision KV while this gate saw "
            "a single precision -- detectors disagree, refusing to skip zeroing."
        )

    # (2) No attention layer may share a KVCacheTensor with a Mamba/KDA layer.
    #     GLM-5-Next slot sharing (kv_cache_utils.py:1640-1678) puts the MLA
    #     page and the SSM state in the same bytes at disjoint block ids, so a
    #     block recycled mamba -> attention would be read as latents while still
    #     holding SSM state (#35219).  Precision uniformity does not help here.
    for kv_tensor in kv_cache_config.kv_cache_tensors:
        shared = set(kv_tensor.shared_by)
        if (shared & mamba_layers) and (shared & attn_layers):
            raise RuntimeError(
                "HAREM_ZERO_ATTENTION_KV=0 is UNSAFE on this model: the "
                f"KVCacheTensor shared by {sorted(shared)} is co-owned by "
                "attention and Mamba/KDA layers (GLM-5-Next slot sharing, "
                "kv_cache_utils.py:1640-1678).  A block recycled from a mamba "
                "group into the attention group would be read as latents while "
                "still holding SSM state (#35219).  Zeroing stays ON."
            )
    return True


# -----------------------------------------------------------------------------
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path of the vllm package")
    a = ap.parse_args()
    p = os.path.join(a.root, "v1/worker/gpu/model_runner.py")
    if not os.path.isfile(p):
        print(f"patch-zerokv: {p} not found", file=sys.stderr)
        return 2
    src = open(p).read()
    if MARK in src:
        print(f"patch-zerokv: already applied ({p})")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"patch-zerokv: ANCHOR count={n} (expected 1) in {p}", file=sys.stderr)
        return 3
    m = src.count(LOGGER_ANCHOR)
    if m != 1:
        print(
            f"patch-zerokv: LOGGER anchor count={m} (expected 1) in {p}",
            file=sys.stderr,
        )
        return 4
    if "self.kv_block_zeroer.zero_block_ids(" not in src:
        print(
            "patch-zerokv: zero_block_ids call site missing - refusing",
            file=sys.stderr,
        )
        return 5
    out = src.replace(OLD, NEW).replace(LOGGER_ANCHOR, LOGGER_ANCHOR + HELPER, 1)
    open(p, "w").write(out)
    print(
        f"patch-zerokv: applied to {p} "
        "(HAREM_ZERO_ATTENTION_KV=0 honoured, fail-closed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

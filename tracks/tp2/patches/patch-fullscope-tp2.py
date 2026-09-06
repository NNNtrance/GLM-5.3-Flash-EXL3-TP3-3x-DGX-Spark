#!/usr/bin/env python3
"""HAREM full-scope EXL3 loader patch for vLLM's glm5next (TP=2 first).

WHY.  `turboderp/GLM-5.3-Flash-exl3@4.05bpw` quantizes the whole model
(attention + shared experts + dense MLP + head), not only the routed experts.
Three independent layers of vLLM's glm5next reader stop that checkpoint from
loading; each is patched here, each behind the same single knob:

  S1  packed_modules_mapping on both model classes            (A1, A2)
      cuda-exl3 can invert vLLM's linear merges only through this dict, which
      vLLM copies off the MODEL CLASS (model_loader/utils.py:275,281).
      glm5next declares none -> {} -> `...gate_up_proj.trellis` has nowhere to
      land and the load stops.
  S2  stop hard-wiring the attention stack to bf16            (A3, A6)
      model.py:331 (MLA) and kda.py:171-174 (KDA) pass quant_config=None
      unconditionally.  Correct for fp8 checkpoints, fatal for a full-scope
      EXL3 one: 72.8 % of today's bf16 weight traffic is locked by those two
      lines and packed_modules_mapping cannot reach it.
  S3  KDA refactorisation                                     (A4, A5, A7, A8)
      The checkpoint keeps upstream-HF's KDA layout: one `qkv_proj` (EXL3) and
      one `conv1d [3P,1,K]`.  vLLM wants `in_proj_qkvbfg_a` (six shards, mixed
      precision -> can never resolve) and three separate `*_conv1d`.  Split the
      module in two and split the conv on load.

Design + measurements:
  our loader-surface design note; the published account is docs/13
Author's loader surface (Zeus, cuda-exl3):
  our read of the cuda-exl3 loader surface; published as docs/13 section 2

DELIBERATELY NOT DONE:
  * `lm_head` is NOT a packed module.  The head is a VocabParallelEmbedding and
    cuda-exl3 loads it through its own `_vocab_loaders` (linear.py:165-200),
    because the vocab is dim 1 of a trellis, not dim 0.  Treating it as a plain
    linear loads without error and is silently wrong.
  * `in_proj_qkvbfg_a` is NOT in the mapping: shards 0-2 are EXL3 and 3-5 are
    bf16, so resolve() returns None for the group whatever the mapping says.
    That is what S3b splits the module for.
  * `wk_weights_proj` is NOT in the mapping: both halves are bf16 in the
    checkpoint and the module is already built with quant_config=None
    (attention.py:263).

Inert unless HAREM_EXL3_FULLSCOPE=1.  Knob unset == upstream, byte for byte:
the class attribute is not even created, the two quant gates keep passing None,
and the stacked-params entries stay exactly as upstream wrote them.  The
patched code re-reads the env at RUNTIME, so a patched image still serves the
routed-experts-only control checkpoint correctly.

Every anchor is exact and must match exactly once; a half-patched stack is the
failure mode that serves fluent, wrong answers.

Usage:
    patch-fullscope-tp2.py --root /usr/local/lib/python3.12/dist-packages/vllm
    patch-fullscope-tp2.py --root ... --check      # report only, write nothing
"""

import argparse
import os
import py_compile
import sys

MARK = "HAREM-FULLSCOPE"

MODEL_REL = ("models", "glm5next", "nvidia", "model.py")
KDA_REL = ("models", "glm5next", "nvidia", "kda.py")


# ---------------------------------------------------------------------------
# model.py -- A1 helpers + S1 mapping
# ---------------------------------------------------------------------------

A1_ANCHOR = (
    "class Glm5NextForCausalLM(\n"
    "    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid, SupportsEagle3\n"
    "):\n"
)

A1_HELPERS = '''# ---------------------------------------------------------------------------
# HAREM-FULLSCOPE (5 September 2026) -- env-gated full-scope EXL3 loader support.
# Design and measurements: docs/13 of the recipe repository.
#
# HAREM_EXL3_FULLSCOPE unset == upstream behaviour, byte for byte.
# ---------------------------------------------------------------------------
import os as _harem_os


def _harem_fs_env() -> bool:
    """The one knob.  Read at import time for the class attribute below, at
    call time everywhere else -- both see the same container environment."""
    return _harem_os.environ.get("HAREM_EXL3_FULLSCOPE") == "1"


def _harem_fs_quant(quant_config) -> bool:
    """True only for an EXL3 quant config with the knob set.

    fp8 (and every other) checkpoint keeps the upstream bf16 hard-wire, which
    is correct for them: those projections really are bf16 on disk.
    """
    if quant_config is None or not _harem_fs_env():
        return False
    get_name = getattr(quant_config, "get_name", None)
    return callable(get_name) and get_name() == "exl3"


# S1.  Source order is load-bearing: it must match the shard ids in
# stacked_params_mapping (asserted by _harem_fs_check_mapping below).  Getting
# it backwards loads WITHOUT error and produces wrong numbers.
_HAREM_FS_PACKED_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    "in_proj_qkv": ["qkv_proj"],
}


def _harem_fs_split_active(model) -> bool:
    """Did kda.py actually split in_proj_qkvbfg_a in THIS model?

    Read off the built module tree, never off the env, so the loader mapping
    cannot disagree with the modules that exist.
    """
    for mod in model.modules():
        if hasattr(mod, "in_proj_qkv") and hasattr(mod, "in_proj_bfg_a"):
            return True
    return False


def _harem_fs_check_mapping(stacked_params_mapping, require=()) -> None:
    """Assert 4 (design doc 5.7): packed_modules_mapping and the shard ids in
    stacked_params_mapping must describe the same fusion.  Data-free.

    A mismatch here is the one failure the author warned about: it loads
    without error and serves wrong numbers.
    """
    mapping = getattr(Glm5NextForCausalLM, "packed_modules_mapping", None)
    if not mapping:
        assert not require, (
            "HAREM-FULLSCOPE: the KDA split is active but no "
            "packed_modules_mapping is declared."
        )
        return

    assert "lm_head" not in mapping, (
        "HAREM-FULLSCOPE: lm_head must never be a packed module.  The head is "
        "a VocabParallelEmbedding whose vocab is dim 1 of the trellis; "
        "cuda-exl3 loads it through its own _vocab_loaders.  As a plain linear "
        "it loads without error and is silently wrong."
    )

    by_param: dict = {}
    for param_name, weight_name, shard_id in stacked_params_mapping:
        by_param.setdefault(param_name.lstrip("."), []).append(
            (shard_id, weight_name.lstrip("."))
        )

    for name in require:
        assert name in mapping, (
            f"HAREM-FULLSCOPE: {name!r} missing from packed_modules_mapping."
        )
        assert name in by_param, (
            f"HAREM-FULLSCOPE: {name!r} missing from stacked_params_mapping."
        )

    for packed, sources in mapping.items():
        entries = by_param.get(packed)
        if not entries:
            continue  # a fusion this loader does not perform
        if len(entries) == 1 and isinstance(entries[0][0], tuple):
            ids, weight_name = entries[0]
            assert list(ids) == list(range(len(ids))), (
                f"HAREM-FULLSCOPE: {packed!r} tuple shard id {ids} must start "
                "at 0 and be consecutive -- weight_loader_v2 indexes a tuple "
                "RELATIVELY (linear.py:901-916), so a non-zero start loads "
                "without error into the wrong slice."
            )
            assert list(sources) == [weight_name], (
                f"HAREM-FULLSCOPE: packed_modules_mapping[{packed!r}] = "
                f"{sources} but the stacked entry loads it from "
                f"{weight_name!r}."
            )
            continue
        ordered = sorted(entries, key=lambda e: e[0])
        assert [sid for sid, _ in ordered] == list(range(len(ordered))), (
            f"HAREM-FULLSCOPE: {packed!r} shard ids "
            f"{[sid for sid, _ in ordered]} are not 0..n-1."
        )
        got = [w for _, w in ordered]
        assert got == list(sources), (
            f"HAREM-FULLSCOPE: packed_modules_mapping[{packed!r}] = {sources} "
            f"but stacked_params_mapping loads the shards in the order {got}. "
            "Reversed order loads without error and serves wrong numbers."
        )


'''

A1_MAPPING = '''    # HAREM-FULLSCOPE S1 (env-gated).  vLLM copies packed_modules_mapping off
    # the model class (model_loader/utils.py:275,281) and cuda-exl3 inverts
    # vLLM's linear merges only through it.  The attribute is not created at
    # all when the knob is unset, so getattr() there returns None, exactly as
    # upstream.
    if _harem_fs_env():
        packed_modules_mapping = dict(_HAREM_FS_PACKED_MAPPING)

'''

A2_ANCHOR = (
    "class Glm5NextForConditionalGeneration(\n"
    "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3\n"
    "):\n"
)

A2_MAPPING = '''    # HAREM-FULLSCOPE S1 (env-gated).  Same mapping as the text class.  When
    # the knob is set this shadows the one inherited from
    # Glm4vForConditionalGeneration (a vision-tower mapping; the tower is built
    # with quant_config=None and, under --language-model-only, not built at
    # all).  Knob unset -> the attribute is not created and the inherited
    # mapping stands, exactly as upstream.
    if _harem_fs_env():
        packed_modules_mapping = dict(_HAREM_FS_PACKED_MAPPING)

'''

A3_ANCHOR = (
    "                quant_config=None,  # MLA projections are BF16 in checkpoint\n"
)

A3_REPL = '''                # HAREM-FULLSCOPE S2.  fp8 checkpoints really do keep
                # the MLA projections in bf16, so the upstream None stays for
                # them.  A full-scope EXL3 checkpoint quantizes o_proj,
                # q_b_proj, fused_qkv_a_proj and indexer.wq_b; the ones that
                # are still bf16 (kv_b_proj, wk_weights_proj) resolve to None
                # and fall back to UnquantizedLinearMethod on their own.
                quant_config=(
                    quant_config if _harem_fs_quant(quant_config) else None
                ),  # MLA projections are BF16 in checkpoint
'''

A4_ANCHOR = '            (".in_proj_qkvbfg_a", ".g_a_proj", 5),\n        ]\n'

A4_REPL = '''            (".in_proj_qkvbfg_a", ".g_a_proj", 5),
        ]
        # HAREM-FULLSCOPE S3b.  When kda.py split in_proj_qkvbfg_a into
        # in_proj_qkv (shards 0-2, ONE fused EXL3 checkpoint tensor) and
        # in_proj_bfg_a (b/f_a/g_a, bf16), the six entries above address
        # modules that no longer exist.  The tuple shard id (0, 1, 2) is the
        # path linear.py:874-914 provides for a checkpoint tensor that spans
        # several of vLLM's shards; it MUST start at 0 (the tuple is indexed
        # relatively).  Decided from the built model, never from the env.
        _harem_fs_require = ()
        if _harem_fs_split_active(self):
            stacked_params_mapping = [
                e for e in stacked_params_mapping if e[0] != ".in_proj_qkvbfg_a"
            ] + [
                (".in_proj_qkv", ".qkv_proj", (0, 1, 2)),
                (".in_proj_bfg_a", ".b_proj", 0),
                (".in_proj_bfg_a", ".f_a_proj", 1),
                (".in_proj_bfg_a", ".g_a_proj", 2),
            ]
            _harem_fs_require = (
                "gate_up_proj",
                "fused_qkv_a_proj",
                "in_proj_qkv",
            )
        _harem_fs_check_mapping(stacked_params_mapping, _harem_fs_require)
'''

A5_ANCHOR = (
    "                    param = params_dict[name]\n"
    "                    weight_loader = getattr(\n"
)

A5_REPL = '''                    # HAREM-FULLSCOPE S3a.  upstream-HF KDA stores ONE
                    # conv1d [3*P, 1, K]; this model has three separate
                    # ColumnParallelLinear conv1d's (kda.py:234-253).  Split on
                    # dim 0 in q,k,v order -- the same order _merged_conv_weight
                    # rebuilds them in (kda.py:438) and the same order as the
                    # in_proj shards.  A wrong order loads without error and
                    # serves a wrong model, hence assert 3 (design doc 5.7).
                    if name.endswith(".self_attn.conv1d.weight"):
                        harem_base = name[: -len("conv1d.weight")]
                        harem_rows = loaded_weight.shape[0]
                        harem_third = harem_rows // 3
                        assert harem_third * 3 == harem_rows, (
                            f"HAREM-FULLSCOPE: {name} has {harem_rows} rows, "
                            "not divisible by 3."
                        )
                        harem_parts = [
                            loaded_weight.narrow(
                                0, i * harem_third, harem_third
                            )
                            for i in range(3)
                        ]
                        # A meta-device dry run carries shapes, not data.
                        if loaded_weight.device.type != "meta":
                            assert not torch.equal(
                                harem_parts[0], harem_parts[1]
                            ) and not torch.equal(
                                harem_parts[1], harem_parts[2]
                            ), (
                                f"HAREM-FULLSCOPE: the three thirds of {name} "
                                "are identical; the checkpoint does not store "
                                "q|k|v stacked on dim 0 and this split would "
                                "be wrong."
                            )
                        for harem_tag, harem_part in zip(
                            ("q", "k", "v"), harem_parts
                        ):
                            harem_tgt = f"{harem_base}{harem_tag}_conv1d.weight"
                            # KeyError here = fail closed, by design.
                            harem_param = params_dict[harem_tgt]
                            harem_loader = getattr(
                                harem_param, "weight_loader",
                                default_weight_loader,
                            )
                            harem_loader(harem_param, harem_part)
                            loaded_params.add(harem_tgt)
                        continue
                    # HAREM-FULLSCOPE.  vLLM's ReplicatedLinear has no
                    # weight_loader_v2 dispatch (linear.py:368-380 is the only
                    # loader it owns), so an EXL3 ReplicatedLinear -- the
                    # sparse indexer's wq_b, which S2 newly quantizes -- would
                    # get the dense v1 loader and die on `suh`:
                    #   "Tried to load weights of size [1536] to a parameter of
                    #    size [1, 1536]".
                    # Route the four EXL3 parameters through the v2 entry point
                    # every other EXL3 linear already uses.  A replicated layer
                    # holds the whole tensor on every rank, and ReplicatedLinear
                    # never calls update_param_tp_status(), so the parameter
                    # still carries the global tp rank: pin it to 0 for the
                    # copy, exactly as _Glm5NextMergedColumnParallelLinear does
                    # for its replicated shards.
                    harem_wl = getattr(params_dict[name], "weight_loader", None)
                    if (
                        getattr(harem_wl, "__func__", None) is not None
                        and harem_wl.__func__.__qualname__
                        == "ReplicatedLinear.weight_loader"
                        and hasattr(
                            params_dict[name], "load_column_parallel_weight"
                        )
                    ):
                        harem_p = params_dict[name]
                        harem_tp = getattr(harem_p, "tp_rank", None)
                        if harem_tp is not None:
                            harem_p.tp_rank = 0
                        try:
                            harem_p.load_column_parallel_weight(loaded_weight)
                        finally:
                            if harem_tp is not None:
                                harem_p.tp_rank = harem_tp
                        loaded_params.add(name)
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
'''

# ---------------------------------------------------------------------------
# kda.py
# ---------------------------------------------------------------------------

A6_ANCHOR = (
    "        saved_quant_config = vllm_config.quant_config\n"
    "        vllm_config.quant_config = None\n"
    "        super().__init__(config, vllm_config, prefix)\n"
    "        vllm_config.quant_config = saved_quant_config\n"
)

A6_REPL = '''        # HAREM-FULLSCOPE S2 (env-gated).  The strip below is right for fp8
        # checkpoints, where the KDA projections really are bf16.  For a
        # full-scope EXL3 checkpoint it hides the whole KDA stack from
        # resolve() -- 43.4 % of the model's bf16 weight traffic -- because
        # GatedDeltaNetAttention.__init__ latches self.quant_config from
        # vllm_config (mamba/gdn/base.py:41) while it is None.
        # HAREM_EXL3_FULLSCOPE unset == upstream, byte for byte.
        import os as _harem_os

        _harem_qc = vllm_config.quant_config
        _harem_get_name = getattr(_harem_qc, "get_name", None)
        _harem_fs = (
            _harem_os.environ.get("HAREM_EXL3_FULLSCOPE") == "1"
            and _harem_qc is not None
            and callable(_harem_get_name)
            and _harem_get_name() == "exl3"
        )
        saved_quant_config = vllm_config.quant_config
        if not _harem_fs:
            vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
        self._harem_fullscope = _harem_fs
        self._harem_fullscope_split = False
'''

A7_ANCHOR = (
    "        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(\n"
    "            self.hidden_size,\n"
    "            [\n"
    "                projection_size,  # q (shard 0)\n"
    "                projection_size,  # k (shard 1)\n"
    "                projection_size,  # v (shard 2)\n"
    "                self.num_heads,  # b (shard 3)\n"
    "                self.head_dim,  # f_a (shard 4, replicated)\n"
    "                self.head_dim,  # g_a (shard 5, replicated)\n"
    "            ],\n"
    "            replicated_shard_ids=(4, 5),\n"
    "            tp_size=self.tp_size,\n"
    "            bias=False,\n"
    "            quant_config=self.quant_config,\n"
    '            prefix=f"{prefix}.in_proj_qkvbfg_a",\n'
    "        )\n"
)

A7_REPL = '''        # HAREM-FULLSCOPE S3b.  The full-scope checkpoint stores q|k|v as ONE
        # EXL3 tensor (`qkv_proj`) and b/f_a/g_a as three bf16 tensors.
        # cuda-exl3 requires a linear to be wholly EXL3 or wholly bf16
        # (config.py resolve(): all(i is not None)), so the merged six-shard
        # module can never resolve.  Split it in two.  The decision is taken
        # from the CHECKPOINT -- does `qkv_proj` resolve? -- not from the env,
        # so the same patched image still builds the upstream single module for
        # a routed-experts-only checkpoint.
        _harem_split = False
        if self._harem_fullscope:
            _harem_resolve = getattr(self.quant_config, "resolve", None)
            if callable(_harem_resolve):
                _harem_split = _harem_resolve(f"{prefix}.in_proj_qkv") is not None
        self._harem_fullscope_split = _harem_split
        if _harem_split:
            self.in_proj_qkv = MergedColumnParallelLinear(
                self.hidden_size,
                [
                    projection_size,  # q (shard 0)
                    projection_size,  # k (shard 1)
                    projection_size,  # v (shard 2)
                ],
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            self.in_proj_bfg_a = _Glm5NextMergedColumnParallelLinear(
                self.hidden_size,
                [
                    self.num_heads,  # b (shard 0)
                    self.head_dim,  # f_a (shard 1, replicated)
                    self.head_dim,  # g_a (shard 2, replicated)
                ],
                replicated_shard_ids=(1, 2),
                tp_size=self.tp_size,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.in_proj_bfg_a",
            )
            _harem_n_total = getattr(self.in_proj_qkv, "exl3_n_total", None)
            if _harem_n_total is not None:
                # Assert 1 (design doc 5.7): three shards, one third each.
                _harem_want = divide(projection_size, self.tp_size)
                _harem_shards = list(self.in_proj_qkv.exl3_shards)
                assert _harem_shards == [_harem_want] * 3, (
                    f"HAREM-FULLSCOPE {prefix}.in_proj_qkv: EXL3 shards "
                    f"{_harem_shards}, expected three of {_harem_want}."
                )
                assert _harem_n_total == 3 * _harem_want, (
                    f"HAREM-FULLSCOPE {prefix}.in_proj_qkv: n_total "
                    f"{_harem_n_total}, expected {3 * _harem_want}."
                )
                # Assert 2 (design doc 5.7): the three replayed suh rows must
                # coalesce back into ONE kernel group.  If they do not, the
                # shards did not receive the same suh and a different tensor
                # was loaded into them.
                _harem_qm = self.in_proj_qkv.quant_method
                _harem_orig_pwal = _harem_qm.process_weights_after_loading

                def _harem_checked_pwal(
                    layer, _orig=_harem_orig_pwal, _p=prefix
                ):
                    _orig(layer)
                    _groups = list(getattr(layer, "exl3_group_n", []))
                    assert len(_groups) == 1, (
                        f"HAREM-FULLSCOPE {_p}.in_proj_qkv: suh coalesced "
                        f"into {len(_groups)} groups {_groups}, expected 1 -- "
                        "the fused checkpoint tensor was not replayed onto "
                        "all three shards."
                    )

                _harem_qm.process_weights_after_loading = _harem_checked_pwal
        else:
            self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
                self.hidden_size,
                [
                    projection_size,  # q (shard 0)
                    projection_size,  # k (shard 1)
                    projection_size,  # v (shard 2)
                    self.num_heads,  # b (shard 3)
                    self.head_dim,  # f_a (shard 4, replicated)
                    self.head_dim,  # g_a (shard 5, replicated)
                ],
                replicated_shard_ids=(4, 5),
                tp_size=self.tp_size,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.in_proj_qkvbfg_a",
            )
'''

A8_ANCHOR = (
    "        projected = self.in_proj_qkvbfg_a(hidden_states)[0]\n"
    "        qkv, beta_raw, f_a, g_a = projected.split(\n"
    "            [\n"
    "                3 * self.local_projection_size,\n"
    "                self.local_num_heads,\n"
    "                self.head_dim,\n"
    "                self.head_dim,\n"
    "            ],\n"
    "            dim=-1,\n"
    "        )\n"
)

A8_REPL = '''        if self._harem_fullscope_split:
            # HAREM-FULLSCOPE S3b: two GEMMs instead of one.  The second is
            # 288 columns per rank against in_proj_qkv's 12288 (2.3 %); no cat
            # is needed because the two halves are consumed separately.
            qkv = self.in_proj_qkv(hidden_states)[0]
            beta_raw, f_a, g_a = self.in_proj_bfg_a(hidden_states)[0].split(
                [self.local_num_heads, self.head_dim, self.head_dim],
                dim=-1,
            )
        else:
            projected = self.in_proj_qkvbfg_a(hidden_states)[0]
            qkv, beta_raw, f_a, g_a = projected.split(
                [
                    3 * self.local_projection_size,
                    self.local_num_heads,
                    self.head_dim,
                    self.head_dim,
                ],
                dim=-1,
            )
'''


# (tag, relative path, anchor, replacement)
ANCHORS = [
    ("A1", MODEL_REL, A1_ANCHOR, A1_HELPERS + A1_ANCHOR + A1_MAPPING),
    ("A2", MODEL_REL, A2_ANCHOR, A2_ANCHOR + A2_MAPPING),
    ("A3", MODEL_REL, A3_ANCHOR, A3_REPL),
    ("A4", MODEL_REL, A4_ANCHOR, A4_REPL),
    ("A5", MODEL_REL, A5_ANCHOR, A5_REPL),
    ("A6", KDA_REL, A6_ANCHOR, A6_REPL),
    ("A7", KDA_REL, A7_ANCHOR, A7_REPL),
    ("A8", KDA_REL, A8_ANCHOR, A8_REPL),
]

# What must be true of the written files.  (relative path, needle, count)
POST_CHECKS = [
    (MODEL_REL, "packed_modules_mapping = dict(_HAREM_FS_PACKED_MAPPING)", 2),
    (MODEL_REL, "def _harem_fs_check_mapping(", 1),
    (MODEL_REL, "quant_config if _harem_fs_quant(quant_config) else None", 1),
    (MODEL_REL, '(".in_proj_qkv", ".qkv_proj", (0, 1, 2))', 1),
    (MODEL_REL, '(".in_proj_bfg_a", ".b_proj", 0)', 1),
    (MODEL_REL, '(".in_proj_bfg_a", ".f_a_proj", 1)', 1),
    (MODEL_REL, '(".in_proj_bfg_a", ".g_a_proj", 2)', 1),
    (MODEL_REL, 'name.endswith(".self_attn.conv1d.weight")', 1),
    (MODEL_REL, '== "ReplicatedLinear.weight_loader"', 1),
    (KDA_REL, "self._harem_fullscope = _harem_fs", 1),
    (KDA_REL, "self.in_proj_qkv = MergedColumnParallelLinear(", 1),
    (KDA_REL, "self.in_proj_bfg_a = _Glm5NextMergedColumnParallelLinear(", 1),
    (KDA_REL, "replicated_shard_ids=(1, 2)", 1),
    (KDA_REL, "replicated_shard_ids=(4, 5)", 1),
    (KDA_REL, "if self._harem_fullscope_split:", 1),
    (KDA_REL, "_harem_qm.process_weights_after_loading = _harem_checked_pwal", 1),
]


def _path(root, rel):
    return os.path.join(root, *rel)


def fail(msg):
    print(f"[fullscope] FAIL: {msg}", file=sys.stderr)
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="vllm package root")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report anchor counts only, change nothing",
    )
    a = ap.parse_args()

    files = {}
    for rel in (MODEL_REL, KDA_REL):
        p = _path(a.root, rel)
        if not os.path.isfile(p):
            return fail(f"no such file {p}")
        with open(p) as f:
            files[rel] = f.read()

    # The older P1-only patch writes a second packed_modules_mapping into the
    # same class body.  Two assignments in one class is legal Python and the
    # last one wins, which is exactly the kind of quiet ambiguity this patch
    # exists to avoid.  The prelude runs one or the other, never both.
    if "HAREM P1" in files[MODEL_REL]:
        return fail(
            "patch-packedmap-tp2.py (HAREM P1) is already applied to model.py. "
            "Full scope supersedes it -- run one or the other, not both."
        )

    already = [rel for rel, src in files.items() if MARK in src]
    if len(already) == len(files):
        print("[fullscope] already applied; nothing to do")
        return 0
    if already:
        return fail(
            f"half-patched tree: {already} carry {MARK}, the rest do not. "
            "Restore the image files before retrying."
        )

    out = dict(files)
    counts = []
    for tag, rel, anchor, repl in ANCHORS:
        n = out[rel].count(anchor)
        counts.append((tag, n))
        if n != 1:
            print(f"[fullscope] anchor counts: {counts}", file=sys.stderr)
            return fail(
                f"{tag}: anchor matched {n} times in {_path(a.root, rel)} "
                "(want exactly 1)"
            )
        out[rel] = out[rel].replace(anchor, repl, 1)

    print(f"[fullscope] anchors 1/1: {' '.join(t for t, _ in counts)}")

    if a.check:
        # Compile the would-be result without touching the tree.
        for rel, src in out.items():
            tmp = _path(a.root, rel) + ".harem-check"
            try:
                with open(tmp, "w") as f:
                    f.write(src)
                py_compile.compile(tmp, doraise=True, cfile=tmp + ".pyc")
            except py_compile.PyCompileError as e:
                return fail(f"patched {rel[-1]} does not compile: {e}")
            except OSError as e:
                print(
                    f"[fullscope] --check: could not write next to the source "
                    f"({e}); compiling from memory instead"
                )
                try:
                    compile(src, rel[-1], "exec")
                except SyntaxError as e2:
                    return fail(f"patched {rel[-1]} does not compile: {e2}")
                continue
            finally:
                for f_ in (tmp, tmp + ".pyc"):
                    if os.path.exists(f_):
                        os.unlink(f_)
        print("[fullscope] --check: 8/8 anchors match once, both files compile; "
              "nothing written")
        return 0

    for rel, src in out.items():
        path = _path(a.root, rel)
        tmp = path + ".harem-tmp"
        with open(tmp, "w") as f:
            f.write(src)
        try:
            py_compile.compile(tmp, doraise=True, cfile=tmp + ".pyc")
        except py_compile.PyCompileError as e:
            os.unlink(tmp)
            return fail(f"patched {path} does not compile: {e}")
        if os.path.exists(tmp + ".pyc"):
            os.unlink(tmp + ".pyc")
        os.replace(tmp, path)

    # Post-check: prove every piece landed, in the written files.
    for rel, needle, want in POST_CHECKS:
        with open(_path(a.root, rel)) as f:
            got = f.read().count(needle)
        if got != want:
            return fail(
                f"post-check {rel[-1]}: {needle!r} appears {got} times, "
                f"want {want}"
            )

    print(
        "[fullscope] applied: S1 packed_modules_mapping x2 "
        "(gate_up_proj, fused_qkv_a_proj, in_proj_qkv; lm_head deliberately "
        "absent), S2 quant gates x2 (MLA + KDA), S3 conv1d split + "
        "in_proj_qkv/in_proj_bfg_a split + 4 stacked entries. "
        "Runtime knob: HAREM_EXL3_FULLSCOPE=1"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

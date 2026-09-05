#!/usr/bin/env python3
"""In-container vLLM patches for GLM-5.3-Flash at TP=3 (Zeus/cuda-exl3 stack).

Run inside the serving container, before ``vllm serve``. Every edit is an
exact-text substitution: if the base image changes, the anchor stops matching
and this script exits non-zero rather than half-patching.

Four edits, all of them small and generic:

1. ``model_executor/parameter.py`` -- ``_harem_pad_then_narrow`` at the four
   places a v2 parameter narrows a checkpoint tensor to its rank's shard.
   Zero-extends when the config pad (heads 64->66, shared-expert I 2048->2112,
   drafter GQA 32/8->36/9) put the highest rank's shard past the stored dim.

2. ``model_loader/weight_utils.py`` -- the same helper in
   ``row_parallel_weight_loader`` and ``sharded_weight_loader``. These are plain
   functions, not parameter classes, so edit 1 does not reach them; they carry
   the KDA ``A_log`` and ``dt_bias``.

   Together, 1 and 2 replace FlyCockpit's ``_tp3_pad_alog`` hook inside
   ``glm5next/.../model.py::load_weights``, which needed three brittle
   call-site anchors and inferred the pad width from the *fused* parameter's
   shape (it over-pads by thousands of rows and happens to still be correct).
   Patching the two loaders instead is one mechanism, at the place that already
   knows the shard offset and width.

3. ``layers/vocab_parallel_embedding.py`` -- raise ``padding_size`` to
   ``lcm(padding_size, tp)`` whenever it does not divide tp. The default 64
   leaves vocab 154880 unchanged and ``divide(154880, 3)`` then asserts. 192
   gives 154944 = 3 x 51648. Covers the target embedding, the LM head, the MTP
   head and the drafter in one place, so nothing has to agree with anything.

4. ``vllm/models/glm5next/nvidia/model.py`` -- the BF16 **shared** expert's
   intermediate size, 2048 -> next multiple of lcm(64, tp) = 2112 at tp=3, so
   it TP-shards to 704 per rank. The zero columns are filled by edit 1.
   Deliberately NOT ``disable_tp``: with EP the MoE runner all-reduces the
   shared output, so a replicated shared expert is summed once per rank and the
   contribution is 3x too large -- the model stays fluent and gets the answers
   wrong.

NOT done here, on purpose:
  * head pad 64->66 in the config -- that is ``pad-tp3.py``'s sidecar, so one
    file is right in every process rather than three override paths agreeing.
  * a kernel-side 22->32 head pad. That exists in FlyCockpit because
    ``FLASHINFER_MLA_SPARSE_SM120`` only instantiates local head counts in
    {8,16,32,64,128}. This stack serves with ``--attention-backend CUSTOM``
    (cuda-exl3's own sparse-MLA kernel), whose decode tiles heads in groups of
    16 with an ``h0 + tid < H`` guard and was measured correct at H=21,22,23
    (our model-free MLA head-count measurement; see ``docs/03``).
    22 heads run as 22, and the 45% of wasted query work that the 22->32 pad
    costs is simply not spent.

Usage:  patch-vllm-tp3.py [--check] [--root /usr/local/lib/python3.12/dist-packages/vllm]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MARKER = "HAREM-TP3"

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")

# --------------------------------------------------------------------------
# the shared helper, injected verbatim into parameter.py and weight_utils.py
# --------------------------------------------------------------------------

HELPER = '''

# --- HAREM-TP3 -------------------------------------------------------------
_HAREM_TP3_SEEN: set = set()


def _harem_pad_then_narrow(tensor, dim: int, start: int, length: int):
    """Zero-extend a checkpoint tensor whose TP shard runs past its stored end.

    GLM-5.3-Flash at TP=3 pads 64 attention/KDA heads to 66 and the BF16 shared
    expert intermediate 2048 -> 2112 (and the DFlash2 drafter GQA 32/8 -> 36/9).
    The highest rank then asks for rows the checkpoint does not have. Those rows
    are true zeros: a padded head's q/k/v projection is zero and its o_proj
    column is zero, so it adds nothing to the row-parallel sum.

    Fail-closed: refuse when the whole shard would be padding. That is the
    64 -> 96 mistake -- rank 2 gets 32 fabricated heads, the server starts, and
    the answers are confident nonsense.
    """
    if tensor.dim() == 0:
        return tensor
    need = start + length
    cur = tensor.size(dim)
    if cur < need:
        extra = need - cur
        if extra >= length:
            raise ValueError(
                f"HAREM-TP3: refusing to zero-extend dim {dim} of a "
                f"{tuple(tensor.shape)} checkpoint tensor by {extra} to fill a "
                f"shard of {length} at offset {start}: the entire shard would be "
                f"padding. The head pad in config.json is too wide for this tp."
            )
        key = (tuple(tensor.shape), dim, extra, length)
        if key not in _HAREM_TP3_SEEN:
            _HAREM_TP3_SEEN.add(key)
            logger.info(
                "HAREM-TP3 pad: %s dim %d +%d zeros (shard %d @ %d)",
                tuple(tensor.shape), dim, extra, length, start,
            )
        pads = [0, 0] * tensor.dim()
        pads[2 * (tensor.dim() - 1 - dim) + 1] = extra
        tensor = torch.nn.functional.pad(tensor, tuple(pads))
    return tensor.narrow(dim, start, length)
# --- end HAREM-TP3 ---------------------------------------------------------
'''

# --------------------------------------------------------------------------
# edits
# --------------------------------------------------------------------------

PARAMETER_EDITS = [
    # anchor for the helper
    (
        "logger = init_logger(__name__)\n\n\nclass BasevLLMParameter(Parameter):",
        "logger = init_logger(__name__)\n" + HELPER
        + "\n\nclass BasevLLMParameter(Parameter):",
    ),
    # load_column_parallel_weight
    (
        "    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):\n"
        "        shard_size = self.data.shape[self.output_dim]\n"
        "        loaded_weight = loaded_weight.narrow(\n"
        "            self.output_dim, self.tp_rank * shard_size, shard_size\n"
        "        )\n",
        "    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):\n"
        "        shard_size = self.data.shape[self.output_dim]\n"
        "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
        "            loaded_weight, self.output_dim, self.tp_rank * shard_size, shard_size\n"
        "        )\n",
    ),
    # load_merged_column_weight
    (
        "        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)\n"
        "        loaded_weight = loaded_weight.narrow(\n"
        "            self.output_dim, self.tp_rank * shard_size, shard_size\n"
        "        )\n",
        "        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)\n"
        "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
        "            loaded_weight, self.output_dim, self.tp_rank * shard_size, shard_size\n"
        "        )\n",
    ),
    # load_qkv_weight
    (
        "        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)\n"
        "        loaded_weight = loaded_weight.narrow(\n"
        "            self.output_dim, shard_id_int * shard_size, shard_size\n"
        "        )\n",
        "        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)\n"
        "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
        "            loaded_weight, self.output_dim, shard_id_int * shard_size, shard_size\n"
        "        )\n",
    ),
    # load_row_parallel_weight
    (
        "    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):\n"
        "        shard_size = self.data.shape[self.input_dim]\n"
        "        loaded_weight = loaded_weight.narrow(\n"
        "            self.input_dim, self.tp_rank * shard_size, shard_size\n"
        "        )\n",
        "    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):\n"
        "        shard_size = self.data.shape[self.input_dim]\n"
        "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
        "            loaded_weight, self.input_dim, self.tp_rank * shard_size, shard_size\n"
        "        )\n",
    ),
]

WEIGHT_UTILS_EDITS = [
    (
        'LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]',
        HELPER + '\n\nLoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]',
    ),
    # row_parallel_weight_loader
    (
        "    if shard_dim is not None:\n"
        "        shard_size = param.data.shape[shard_dim]\n"
        "        start_idx = tp_rank * shard_size\n"
        "        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)\n",
        "    if shard_dim is not None:\n"
        "        shard_size = param.data.shape[shard_dim]\n"
        "        start_idx = tp_rank * shard_size\n"
        "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
        "            loaded_weight, shard_dim, start_idx, shard_size)\n",
    ),
    # sharded_weight_loader (KDA A_log, dt_bias)
    (
        "        shard_size = param.data.shape[shard_axis]\n"
        "        start_idx = tp_rank * shard_size\n"
        "        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)\n",
        "        shard_size = param.data.shape[shard_axis]\n"
        "        start_idx = tp_rank * shard_size\n"
        "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
        "            loaded_weight, shard_axis, start_idx, shard_size)\n",
    ),
]

VOCAB_EDITS = [
    (
        "        self.num_embeddings = num_embeddings\n"
        "        self.padding_size = padding_size\n"
        "        self.org_vocab_size = org_num_embeddings or num_embeddings\n",
        "        self.num_embeddings = num_embeddings\n"
        "        self.padding_size = padding_size\n"
        "        # HAREM-TP3: the default pad_to=64 leaves vocab 154880 unchanged and\n"
        "        # divide(154880, 3) then asserts. lcm(64, 3) = 192 -> 154944 = 3 x 51648.\n"
        "        # The extra 64 rows are zero and LogitsProcessor truncates back to\n"
        "        # org_vocab_size, so nothing downstream sees them.\n"
        "        if self.tp_size > 1 and self.padding_size % self.tp_size != 0:\n"
        "            from math import gcd as _harem_gcd\n"
        "            self.padding_size = (\n"
        "                self.padding_size\n"
        "                * self.tp_size\n"
        "                // _harem_gcd(self.padding_size, self.tp_size)\n"
        "            )\n"
        "        self.org_vocab_size = org_num_embeddings or num_embeddings\n",
    ),
]

VISION_HELPER = '''

# --- HAREM-TP3 -------------------------------------------------------------
def _harem_build_vision_tower(tower_cls, multimodal_config, *args, **kwargs):
    """Honour ``--language-model-only``: do not build the vision tower at all.

    ``Glm5NextForConditionalGeneration.__init__`` reads ``multimodal_config``
    (even ``mm_encoder_tp_mode``) but never looks at ``language_model_only``, so
    the tower was constructed and its 347 checkpoint tensors (1.05 GiB BF16)
    loaded on every rank of a text-only server. Elsewhere in this same image the
    flag IS honoured -- ``qwen3_next.py`` reads
    ``mm_config.language_model_only`` -- so this is a missing check, not a
    design choice.

    At TP=2 the omission only wasted memory. At TP=3 it is fatal: the tower has
    16 attention heads and ``divide(16, 3)`` asserts before a single weight is
    read. Padding those heads would be wrong -- the tower is never executed, so
    the right answer is not to build it.

    Returns None when text-only; the class's load_weights override then skips
    the ``visual.`` prefix so the checkpoint's tower tensors are dropped rather
    than reported as unexpected.
    """
    if getattr(multimodal_config, "language_model_only", False):
        logger.info(
            "HAREM-TP3: --language-model-only is set, so the GLM-5.3 vision "
            "tower is not built and its checkpoint tensors are skipped. "
            "(Stock behaviour builds and loads it anyway; at tp=3 its 16 "
            "attention heads make divide(16, 3) assert.)"
        )
        return None
    return tower_cls(*args, **kwargs)
# --- end HAREM-TP3 ---------------------------------------------------------
'''

MODEL_EDITS = [
    # 4a. the vision-tower helper
    (
        "logger = init_logger(__name__)\n",
        "logger = init_logger(__name__)\n" + VISION_HELPER,
    ),
    # 4b. build the tower only when the server is not text-only
    (
        "            self.visual = Glm5NextVisionTransformer(\n"
        "                config.text_config,\n",
        "            self.visual = _harem_build_vision_tower(  # HAREM-TP3\n"
        "                Glm5NextVisionTransformer,\n"
        "                multimodal_config,\n"
        "                config.text_config,\n",
    ),
    # 4c. drop the tower's checkpoint tensors when it was not built
    (
        "    def get_encoder_cudagraph_config(self):\n",
        "    def load_weights(self, weights):\n"
        "        # HAREM-TP3: the inherited Glm4v loader is AutoWeightsLoader(self)\n"
        "        # with no skips, so without the tower every visual.* tensor would\n"
        "        # be reported as unexpected. Skip the prefix instead. The mapper\n"
        "        # runs before the skip filter, so 'visual.' is the post-mapping\n"
        "        # name (model.visual.* -> visual.*).\n"
        "        if getattr(self, 'visual', None) is None:\n"
        "            loader = AutoWeightsLoader(self, skip_prefixes=['visual.'])\n"
        "            return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
        "        return super().load_weights(weights)\n"
        "\n"
        "    def get_encoder_cudagraph_config(self):\n",
    ),
    (
        "            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n",
        "            # HAREM-TP3: the shared expert is native BF16, so unlike the routed\n"
        "            # EXL3 trellis it CAN be zero-padded. Round one expert's intermediate\n"
        "            # up to a multiple of lcm(64, tp) -- 64 keeps the GEMM aligned, tp\n"
        "            # makes it shardable. At tp=3: 2048 -> 2112 -> 704 per rank. At tp=2\n"
        "            # or tp=1 this is a no-op. Do NOT replace this with disable_tp: the\n"
        "            # MoE runner all-reduces the shared output under EP, so a replicated\n"
        "            # shared expert is counted once per rank.\n"
        "            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n"
        "            _harem_tp = get_tensor_model_parallel_world_size()\n"
        "            if _harem_tp > 1 and intermediate_size % _harem_tp != 0:\n"
        "                from math import gcd as _harem_gcd\n"
        "                _harem_unit = 64 * _harem_tp // _harem_gcd(64, _harem_tp)\n"
        "                _harem_one = (\n"
        "                    (config.moe_intermediate_size + _harem_unit - 1) // _harem_unit\n"
        "                ) * _harem_unit\n"
        "                intermediate_size = _harem_one * config.n_shared_experts\n"
        "                assert intermediate_size % _harem_tp == 0, (\n"
        "                    f'HAREM-TP3: shared expert intermediate {intermediate_size} '\n"
        "                    f'still does not divide tp={_harem_tp}'\n"
        "                )\n",
    ),
]

FILES = [
    ("model_executor/parameter.py", PARAMETER_EDITS),
    ("model_executor/model_loader/weight_utils.py", WEIGHT_UTILS_EDITS),
    ("model_executor/layers/vocab_parallel_embedding.py", VOCAB_EDITS),
    ("models/glm5next/nvidia/model.py", MODEL_EDITS),
]

# Alternate location for the model file across image builds.
MODEL_ALT = "model_executor/models/glm5next/nvidia/model.py"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def apply_file(path: Path, edits, check_only: bool) -> bool:
    src = path.read_text()
    if MARKER in src:
        print(f"  {path.name}: already patched (sha {sha(path)})")
        return False
    missing = [old for old, _ in edits if src.count(old) != 1]
    if missing:
        for old in missing:
            n = src.count(old)
            head = old.strip().splitlines()[0][:90]
            print(f"  ANCHOR {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}: {head}",
                  file=sys.stderr)
        raise SystemExit(
            f"{path}: {len(missing)} anchor(s) did not match exactly once. "
            "The base image changed; re-read the file before re-running."
        )
    if check_only:
        print(f"  {path.name}: {len(edits)} anchors OK, NOT patched (--check)")
        return False
    for old, new in edits:
        src = src.replace(old, new, 1)
    before = sha(path)
    path.write_text(src)
    print(f"  {path.name}: patched {len(edits)} sites ({before} -> {sha(path)})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true",
                    help="verify every anchor still matches; change nothing")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"no vllm package at {args.root}")
    print(f"HAREM-TP3 vLLM patches in {args.root}")

    changed = 0
    for rel, edits in FILES:
        path = args.root / rel
        if not path.is_file() and rel.endswith("glm5next/nvidia/model.py"):
            path = args.root / MODEL_ALT
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        changed += bool(apply_file(path, edits, args.check))
    print(f"HAREM-TP3: {'anchors verified' if args.check else f'{changed} file(s) changed'}")


if __name__ == "__main__":
    main()

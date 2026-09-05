"""Fail-closed verification of the DFlash2 port.

Run at image build time; also runnable later for diagnosis:
    docker run --rm --entrypoint python3 exl3-zeus:f4987cf /opt/harem/dflash2-gate.py

Every check here guards a hook whose absence would NOT raise at runtime: the
engine would boot, produce correct-looking text, and only lose acceptance. That
is precisely the failure mode that cost the NVFP4 stack a week (a decode kernel
silently wrong at 22 heads), so it is checked at build time instead.
"""

import inspect

import vllm.model_executor.models.qwen3_dflash2 as m

assert hasattr(m, "DFlash2Qwen3ForCausalLM"), "DFlash2Qwen3ForCausalLM missing"
assert hasattr(m, "DFlash2Qwen3Model"), "DFlash2Qwen3Model missing"
assert hasattr(m, "DFlash2Qwen3DecoderLayer"), "DFlash2Qwen3DecoderLayer missing"
assert hasattr(m, "CandidateSelector"), "CandidateSelector missing"
assert hasattr(m, "DFlashGroupedConv"), "DFlashGroupedConv missing"
assert hasattr(m.DFlash2Qwen3Model, "_harem_check_port_assumptions"), "HAREM guard missing"
assert m.DFlash2Qwen3Model.decoder_layer_cls is m.DFlash2Qwen3DecoderLayer, "decoder_layer_cls hook not wired"
assert m.DFlash2Qwen3ForCausalLM.model_cls is m.DFlash2Qwen3Model, "model_cls hook not wired"

from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS as R

assert R["DFlash2DraftModel"] == ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"), R.get("DFlash2DraftModel")
assert R["DFlashDraftModel"] == ("qwen3_dflash", "DFlashQwen3ForCausalLM"), "DFlash v1 entry disturbed"

import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as s

assert hasattr(s, "DFlash2Speculator"), "DFlash2Speculator missing"
assert hasattr(s, "_selector_walk_kernel"), "selector walk kernel missing"
assert hasattr(s, "_cache_draft_logits_kernel"), "draft logits cache kernel missing"
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

assert issubclass(s.DFlash2Speculator, DFlashSpeculator), "DFlash2Speculator must extend DFlashSpeculator"
assert s.DFlash2Speculator.draft_logits_spec is not DFlashSpeculator.draft_logits_spec, (
    "DFlash2 must override draft_logits_spec (fp32/-inf), else the selector walk "
    "and the rejection sampler read different distributions"
)

from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax  # noqa: F401
from vllm.model_executor.layers.logits_processor import LogitsProcessor

assert hasattr(LogitsProcessor, "get_top_k_tokens"), "get_top_k_tokens missing"

from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

assert hasattr(DraftModelSpeculator, "draft_logits_spec"), "draft_logits_spec hook missing"

from vllm.config.vllm import VllmConfig

assert hasattr(VllmConfig, "_is_dflash2_draft"), "_is_dflash2_draft missing"

from vllm.v1.worker.gpu.spec_decode import init_speculator

src = inspect.getsource(init_speculator)
assert "DFlash2Speculator" in src, "init_speculator does not dispatch to DFlash2"

# The V2 model runner must be forced on for a DFlash2 draft: on V1 the same
# checkpoint drafts through DFlashProposer, which never calls the candidate
# selector, and the draft silently degrades to DFlash1.
src = inspect.getsource(VllmConfig._is_dflash2_draft)
assert "DFlash2DraftModel" in src, "_is_dflash2_draft does not test the architecture"

# --- Target side: GLM-5.3 must expose the EAGLE3 aux-hidden-state interface. ---
# DFlash reads the target's intermediate layers (dflash_config.target_layer_ids
# = 5/14/24/33/42). Without this the engine dies at load with
# "Model does not support EAGLE3 interface" -- which is at least loud, but the
# base image ships NO aux-hidden-state support in glm5next at all, so DFlash v1
# could not have worked here either.
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)

assert "EagleModelMixin" in [c.__name__ for c in Glm5NextModel.__mro__], (
    "Glm5NextModel must inherit EagleModelMixin (SupportsEagle3 asserts on it)"
)
assert hasattr(Glm5NextModel, "_set_aux_hidden_state_layers"), "mixin hook missing"
assert hasattr(Glm5NextModel, "_prepare_aux_hidden_state"), "aux hidden state builder missing"
for _cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
    assert "SupportsEagle3" in [c.__name__ for c in _cls.__mro__], (
        f"{_cls.__name__} must declare SupportsEagle3"
    )
    assert hasattr(_cls, "set_aux_hidden_state_layers"), (
        f"{_cls.__name__}.set_aux_hidden_state_layers missing"
    )
# The served class is the multimodal wrapper; its inner model is what the
# protocol reaches through get_language_model().model.
assert hasattr(Glm5NextForConditionalGeneration, "get_language_model"), (
    "wrapper must expose get_language_model() for the EAGLE3 protocol to resolve"
)

# --- KV cache grouping: the draft must get its own group. ---
# Without this the DFlash draft's sliding-window layers disqualify GLM-5.3's
# specialised grouping path, the model falls through to the generic multi-group
# path, and the kpool indexer page cannot be unified against the MLA/mamba page.
import inspect as _inspect

from vllm.v1.core import kv_cache_utils as _kvu

assert hasattr(_kvu, "_harem_partition_dflash_draft_specs"), "draft/target partition missing"
assert hasattr(_kvu, "_harem_draft_bytes_per_block"), "draft per-block accounting missing"
_src = _inspect.getsource(_kvu.get_kv_cache_groups)
assert "_harem_partition_dflash_draft_specs" in _src, (
    "get_kv_cache_groups does not partition the DFlash draft"
)
# The layout tuple grew by one element; every consumer must unpack 9, or the
# draft's blocks would go unaccounted and its tensors unallocated.
_layout_src = _inspect.getsource(_kvu._glm5_next_tensor_layout)
assert "draft_groups," in _layout_src, "_glm5_next_tensor_layout does not return draft groups"
for _fn in (_kvu._pool_bytes_per_block, _kvu.get_kv_cache_config_from_groups):
    assert "draft_groups" in _inspect.getsource(_fn), (
        f"{_fn.__name__} does not account for the draft group"
    )

print("DFLASH2 PORT BUILD GATE: OK")

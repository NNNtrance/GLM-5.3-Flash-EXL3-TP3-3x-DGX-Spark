#!/bin/bash
# In-container prelude for GLM-5.3-Flash EXL3 at TP=3 -- FULL-SCOPE ARM.
#
# Copy of tp3/tp3-prelude.sh. It lives in tp3full/ and runs tp3full/'s copies of
# the patch scripts, which differ from the production ones in exactly two
# constants (vocab padding_size lcm(128,tp)=384 instead of lcm(64,tp)=192;
# shared expert 2304 instead of 2112 -- both no-ops at tp<=2) plus one extra
# patch, patch-fullscope-tp3.py. ~/exl3-zeus/tp3/ is the production fastload
# manifest identity and is not touched: a new file there has already refused a
# production boot twice.
#
# Mount this at /start.sh and launch the image with `--entrypoint bash /start.sh`,
# passing the usual `vllm serve` arguments after it. It applies the three patch
# scripts and then execs the server, so a patch that no longer applies stops the
# rank instead of serving a silently-wrong model.
#
#   -v $TP3FULL/tp3-prelude.sh:/start.sh:ro   (hard link to this file)
#   -v $TP3FULL:/opt/harem-tp3:ro
# start-tp3.sh builds both mounts from TP3_DIR, which .env.tp3-full sets to
# $HOME/exl3-zeus/tp3full. The launcher mounts "$TP3_DIR/tp3-prelude.sh", so
# this file is hard-linked to that name inside tp3full/ -- one inode, so the
# two names cannot drift apart.
#   --entrypoint bash "$IMAGE" /start.sh <model path> <vllm args...>
#
# TP3_STRICT=0 turns the failures into warnings. Do not use it to get past a
# broken anchor: a half-patched stack is exactly the failure mode that produces
# fluent, wrong answers.
set -euo pipefail

TP3_DIR="${TP3_DIR:-/opt/harem-tp3}"
VLLM_PY="${VLLM_PY:-/usr/local/lib/python3.12/dist-packages/vllm}"
EXL3_PKG="${EXL3_PKG:-/usr/local/lib/python3.12/dist-packages/cuda_exl3}"
STRICT="${TP3_STRICT:-1}"

run() {
  echo "[tp3-prelude] $*"
  if "$@"; then return 0; fi
  echo "[tp3-prelude] FAILED: $*" >&2
  [ "$STRICT" = "1" ] && exit 21
  return 0
}

echo "[tp3-prelude] TP3FULL arm rank=${NODE_RANK:-?} tp=${TP_SIZE:-?} ep=${ENABLE_EP:-?} fullscope=${HAREM_EXL3_FULLSCOPE:-0}"
echo "[tp3-prelude] tp3full constants: vocab padding_size lcm(128,tp), shared expert lcm(128,tp) -> 2304, A9 checkpoint-width fused split"
run python3 "$TP3_DIR/patch-vllm-tp3.py" --root "$VLLM_PY"
run python3 "$TP3_DIR/patch-exl3-ep.py" --pkg "$EXL3_PKG" \
    --overlay "$TP3_DIR/overlay/cuda_exl3/_harem_ep.py"

# The DFlash2 drafter runs on a padded sidecar config (32/8 -> 36/9 at tp=3).
# This makes the port's head check pad-aware and proves after load that the
# fabricated rows are zero. Skipped when the image carries no DFlash2 port.
if [ -f "$VLLM_PY/model_executor/models/qwen3_dflash2.py" ]; then
  run python3 "$TP3_DIR/patch-dflash-tp3.py" --root "$VLLM_PY"
fi

# Logging only: print the per-group decomposition of the KV pool arithmetic, so
# "GPU KV cache size: N tokens" is an explained number rather than a mystery.
run python3 "$TP3_DIR/patch-kvdiag-tp3.py" --root "$VLLM_PY"
run python3 "$TP3_DIR/patch-swblock-tp3.py" --root "$VLLM_PY"

# EXL3 keeps the heavy part of an expert in "<proj>.trellis"; upstream's EP
# weight filter only recognises ".weight"/".weight_packed", so without this it
# reads all 288 experts on every rank even with --enable-ep-weight-filter.
# The patch is inert unless that flag is passed.
run python3 "$TP3_DIR/patch-epfilter-tp3.py" --root "$VLLM_PY"

# Per-rank fastload sidecar. Inert unless HAREM_FASTLOAD_MODE is dump|load;
# start-tp3.sh sets that (and the mount) from FASTLOAD_MODE in the env file.
run python3 "$TP3_DIR/patch-fastload-tp3.py" --root "$VLLM_PY"

# --- Optional arms (5 September 2026) ----------------------------------------------
# Each of the three is applied ONLY when its own env knob is set, so an env file
# that never asks for one cannot be broken by an anchor that drifted in another
# image.  Unset knob == upstream behaviour, byte for byte.
#
#  HAREM_ZERO_ATTENTION_KV=0  fail-closed gate that skips the per-step ATTENTION
#                             KV memset (13.5-15.6 ms/chunk = 1.2-1.4 % prefill).
#                             It proves uniform KV precision AND that no Mamba/KDA
#                             layer shares an attention KVCacheTensor, or raises
#                             at startup instead of serving.
#  HAREM_DRAFT_KV_DTYPE=fp8   put the DFlash2 drafter's KV at the main groups'
#                             precision (start-tp3.sh pins it to "auto" today).
#  HAREM_TILELANG_FAILLOUD=1  turn tilelang_kernels.py:26's silent
#                             contextlib.suppress around `import flashinfer.comm`
#                             into a named, immediate error.
if [ "${HAREM_ZERO_ATTENTION_KV:-}" = "0" ]; then
  run python3 "$TP3_DIR/patch-zerokv-tp3.py" --root "$VLLM_PY"
fi
if [ -n "${HAREM_DRAFT_KV_DTYPE:-}" ]; then
  run python3 "$TP3_DIR/patch-draftkv-tp3.py" --root "$VLLM_PY"
fi
if [ "${HAREM_TILELANG_FAILLOUD:-}" = "1" ]; then
  run python3 "$TP3_DIR/patch-tilelang-failloud-tp3.py" --root "$VLLM_PY"
fi

# --- Full-scope EXL3 (5 September 2026) ------------------------------------------
# One patch, three layers, one knob:
#   S1  packed_modules_mapping on both glm5next model classes
#   S2  stop hard-wiring the attention stack (MLA + KDA) to bf16
#   S3  KDA refactorisation: checkpoint `conv1d` -> q/k/v_conv1d, and
#       `qkv_proj` -> shards 0-2 of a split `in_proj_qkv`
#   A9  split a pre-fused checkpoint tensor by the CHECKPOINT widths, not the
#       module's padded ones (TP=3 head pad 64 -> 66)
#   A10 post-load audit: every EXL3 pad is whole 128-blocks and exactly zero
# Only for a FULL-SCOPE EXL3 checkpoint (turboderp/GLM-5.3-Flash-exl3@4.05bpw).
# HAREM_EXL3_FULLSCOPE unset == upstream image behaviour, byte for byte, and
# the patched code re-reads the knob at runtime, so a patched image still
# serves the routed-experts-only control checkpoint correctly.
# Design and measurements: docs/13 of the recipe repository.
# TP=3 pad arithmetic: docs/13 section 7.1.
# TP=3 needs a cuda-exl3 with the padded-load path: f3e3090 (padded output dim,
# row-parallel suh) AND 754421f (the vocab loaders fill a prefix). On an older
# image the lm_head load dies -- 62f53e6/5903248 raise "EXL3 weights cannot be
# zero-extended" in create_weights; f3e3090 alone passes that gate and then dies
# on a copy_ shape mismatch in _vocab_loaders. Both failures are loud.
if [ "${HAREM_EXL3_FULLSCOPE:-}" = "1" ]; then
  echo "[tp3-prelude] patch-fullscope-tp3.py sha256 $(sha256sum "$TP3_DIR/patch-fullscope-tp3.py" | cut -c1-16)"
  run python3 "$TP3_DIR/patch-fullscope-tp3.py" --root "$VLLM_PY"
  # Say which cuda-exl3 padded-load support is present, before the weights move.
  run python3 "$TP3_DIR/check-padload-tp3.py"
fi

# Import flashinfer.comm once, CPU-side, before any worker starts: prints the
# version into the boot log and warms flashinfer's JIT cache so the ranks do not
# race it.  ~2 s.  Never fails the boot unless HAREM_TILELANG_FAILLOUD=1.
# HAREM_FLASHINFER_WARMUP=0 skips it.
run python3 "$TP3_DIR/flashinfer-warmup.py"


# The model directory is argv[1]; run the shape preflight against whatever the
# launcher actually mounted, not against what the .env says it mounted.
# The EP-vs-tensor-sliced decision is arithmetic on the mounted model, and
# preflight is the thing that owns it: --ep tells it which arrangement the
# launcher chose, and it refuses ENABLE_EP=0 unless moe_intermediate_size is a
# multiple of 128*tp AND the weights on disk agree with the config. The old
# blanket "TP=3 always needs EP" refusal was true of the 2048 checkpoint only.
if [ -d "${1:-}" ]; then
  run python3 "$TP3_DIR/preflight-tp3.py" --model "$1" --tp "${TP_SIZE:-3}" \
      --ep "${ENABLE_EP:-1}"
fi

# A fastload sidecar produced from another checkpoint / image / patch set must
# stop the rank here, not four minutes later with weights nobody checked.
if [ -n "${HAREM_FASTLOAD_MODE:-}" ] && [ -d "${1:-}" ]; then
  run python3 "$TP3_DIR/preflight-fastload.py" --model "$1"
fi

echo "[tp3-prelude] patches applied (tp3full arm); starting vllm serve"
exec vllm serve "$@"

#!/bin/bash
# In-container prelude for GLM-5.3-Flash EXL3 at TP=3.
#
# Mount this at /start.sh and launch the image with `--entrypoint bash /start.sh`,
# passing the usual `vllm serve` arguments after it. It applies the three patch
# scripts and then execs the server, so a patch that no longer applies stops the
# rank instead of serving a silently-wrong model.
#
#   -v $TP3/tp3-prelude.sh:/start.sh:ro
#   -v $TP3:/opt/harem-tp3:ro
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

echo "[tp3-prelude] rank=${NODE_RANK:-?} tp=${TP_SIZE:-?} ep=${ENABLE_EP:-?}"
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

echo "[tp3-prelude] patches applied; starting vllm serve"
exec vllm serve "$@"

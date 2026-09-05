#!/usr/bin/env bash
# run-moe-stage.sh <image-tag> <label> [ms] [arms]  -- model-free MoE stage timing on one node.
set -euo pipefail
IMG=$1; LABEL=$2; MS=${3:-8,64,2048}; ARMS=${4:-EP}
docker rm -f moe-stage >/dev/null 2>&1 || true
docker run --rm --gpus all --name moe-stage --network host --ipc host --shm-size 8g \
  --cpuset-cpus "5-9,15-19" --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v /var/tmp/exl3-zeus-cache:/cache -v "$HOME/exl3-zeus/bench:/bench:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_CUDA_ARCH_LIST=12.1 \
  --entrypoint python3 "$IMG" /bench/moe_stage_bench.py \
  --ms "$MS" --arms "$ARMS" --label "$LABEL" --out "/cache/moe_stage-$LABEL.json"

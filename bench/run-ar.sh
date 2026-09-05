#!/usr/bin/env bash
# all-reduce micro-benchmark over the mesh, same NCCL env as start-tp3.sh.
#   ./run-ar.sh <rank> <world> [ALGO] [PROTO]
set -euo pipefail
RANK=$1; WORLD=$2; ALGO=${3:-}; PROTO=${4:-}
source "$HOME/exl3-zeus/.env.tp3"
IMAGE=${IMAGE:-exl3-zeus:dflash}
GLOO_IFACE=${GLOO_IFACE:-enP7s7}
PLUG=${NCCL_MESH_PLUGIN_DIR:-$HOME/nccl-mesh}
NAME=ar-bench
TAG="w${WORLD}-${ALGO:-auto}-${PROTO:-auto}"
docker rm -f $NAME >/dev/null 2>&1 || true
docker run --rm --gpus all --name $NAME --network host --ipc host --shm-size 8g \
  --cpuset-cpus "${CPUSET:-5-9,15-19}" --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$PLUG:/opt/nccl-mesh:ro" -v /var/tmp/exl3-zeus-cache:/cache \
  -v "$HOME/exl3-zeus/bench:/bench:ro" \
  -e VLLM_HOST_IP="$HOST_IP" -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT=29577 \
  -e RANK="$RANK" -e WORLD_SIZE="$WORLD" -e LOCAL_RANK=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e GLOO_SOCKET_IFNAME="$GLOO_IFACE" -e TP_SOCKET_IFNAME="$GLOO_IFACE" -e MN_IF_NAME="$GLOO_IFACE" \
  -e NCCL_NET=Mesh -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME="=${GLOO_IFACE}" -e NCCL_NET_PLUGIN=mesh \
  -e NCCL_MESH_DEBUG=0 -e LD_LIBRARY_PATH=/opt/nccl-mesh \
  ${ALGO:+-e NCCL_ALGO=$ALGO} ${PROTO:+-e NCCL_PROTO=$PROTO} \
  -e AR_OUT="/cache/ar-$TAG.json" \
  --entrypoint python3 "$IMAGE" /bench/ar_bench.py

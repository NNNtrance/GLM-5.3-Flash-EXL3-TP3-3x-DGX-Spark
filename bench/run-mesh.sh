#!/usr/bin/env bash
# Model-free NCCL micro-benchmark over the mesh, one process per node.
#   run-mesh.sh <rank> <world>
# Everything else comes from the environment:
#   MESH_TAG   label for the run and the json
#   MESH_OPS   allreduce[,alltoall][,sendrecv]
#   MESH_ENV   extra "K=V K=V" pairs passed straight into the container
#   PLUGIN_DIR host dir holding libnccl-net-mesh.so (default: production, read-only)
#   NCCL_DEBUG / NCCL_DEBUG_SUBSYS  passed through
set -euo pipefail
RANK=$1; WORLD=$2
source "$HOME/exl3-zeus/.env.tp3"
IMAGE=${IMAGE:-exl3-zeus:serve}
GLOO_IFACE=${GLOO_IFACE:-enP7s7}
# PLUGIN_SUB is resolved against THIS node's $HOME, so it works even when the
# home directories differ between nodes. PLUGIN_DIR is an absolute override.
if [ -n "${PLUGIN_SUB:-}" ]; then PLUG="$HOME/$PLUGIN_SUB"; else PLUG=${PLUGIN_DIR:-$HOME/nccl-mesh}; fi
test -f "$PLUG/libnccl-net-mesh.so" || { echo "no mesh plugin in $PLUG" >&2; exit 2; }
TAG=${MESH_TAG:-run}
NAME=mesh-bench
EXTRA=(); for kv in ${MESH_ENV:-}; do EXTRA+=(-e "$kv"); done
docker rm -f $NAME >/dev/null 2>&1 || true
docker run --rm --gpus all --name $NAME --network host --ipc host --shm-size 8g \
  --cpuset-cpus "${CPUSET:-5-9,15-19}" --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$PLUG:/opt/nccl-mesh:ro" -v /var/tmp/exl3-zeus-cache:/cache \
  -v "$HOME/exl3-zeus/bench:/bench:ro" \
  -e VLLM_HOST_IP="$HOST_IP" -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT=29577 \
  -e RANK="$RANK" -e WORLD_SIZE="$WORLD" -e LOCAL_RANK=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e GLOO_SOCKET_IFNAME="$GLOO_IFACE" -e TP_SOCKET_IFNAME="$GLOO_IFACE" -e MN_IF_NAME="$GLOO_IFACE" \
  -e NCCL_NET=Mesh -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME="=${GLOO_IFACE}" -e NCCL_NET_PLUGIN=mesh \
  -e NCCL_ALGO="${NCCL_ALGO:-Ring}" -e NCCL_MESH_DEBUG="${NCCL_MESH_DEBUG:-0}" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" ${NCCL_DEBUG_SUBSYS:+-e NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS} \
  -e LD_LIBRARY_PATH=/opt/nccl-mesh \
  -e MESH_OPS="${MESH_OPS:-allreduce}" -e MESH_TAG="$TAG" \
  -e MESH_MAX_TOKENS="${MESH_MAX_TOKENS:-8192}" ${MESH_TOKENS:+-e MESH_TOKENS="$MESH_TOKENS"} \
  -e MESH_STEP90="${MESH_STEP90:-1}" -e MESH_STEP90_TOKENS="${MESH_STEP90_TOKENS:-8}" \
  -e MESH_OUT="/cache/mesh-$TAG-r$RANK.json" \
  "${EXTRA[@]}" \
  --entrypoint python3 "$IMAGE" /bench/mesh_sweep.py

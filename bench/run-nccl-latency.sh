#!/usr/bin/env bash
# run-nccl-latency.sh <rank> <world>   -- one container, one node.
# Sibling of bench/run-mesh.sh, same proven env plumbing (mesh plugin, GLOO
# rendezvous, IB device passthrough), but:
#   - fixed --cpuset-cpus 10-14 (NOT the engine's 5-9,15-19) and --memory cap,
#     so this never contends with a live serving container on the same node;
#   - MASTER_ADDR/GLOO_IFACE are set here rather than sourced from the engine's
#     env file, so this tool has zero coupling to whatever that file says at the
#     moment it runs (the engine may be mid-reconfiguration);
#   - entrypoint is nccl-latency-bench.py, not mesh_sweep.py.
# Pass-through env: LAT_TAG LAT_SIZES LAT_WARMUP LAT_ITERS LAT_ENV PLUGIN_SUB
#                   NCCL_DEBUG NCCL_DEBUG_SUBSYS MASTER_ADDR GLOO_IFACE
set -euo pipefail
RANK=$1; WORLD=$2

MASTER_ADDR=${MASTER_ADDR:-192.0.2.10}   # rank 0's management address; set it for your cluster
GLOO_IFACE=${GLOO_IFACE:-enP7s7}         # the same NIC name on all three of our nodes
IMAGE=${IMAGE:-exl3-zeus:62f53e6}
CPUSET=${CPUSET:-10-14}
MEMLIMIT=${MEMLIMIT:-6g}

if [ -n "${PLUGIN_SUB:-}" ]; then PLUG="$HOME/$PLUGIN_SUB"; else PLUG="$HOME/exl3-zeus/nccl-mesh-patched2"; fi
test -f "$PLUG/libnccl-net-mesh.so" || { echo "no mesh plugin in $PLUG" >&2; exit 2; }

TAG=${LAT_TAG:-run}
NAME=nccl-lat-bench
EXTRA=(); for kv in ${LAT_ENV:-}; do EXTRA+=(-e "$kv"); done
mkdir -p /var/tmp/exl3-zeus-cache
docker rm -f $NAME >/dev/null 2>&1 || true
docker run --rm --gpus all --name $NAME --network host --ipc host --shm-size 2g \
  --cpuset-cpus "$CPUSET" --memory "$MEMLIMIT" --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$PLUG:/opt/nccl-mesh:ro" -v /var/tmp/exl3-zeus-cache:/cache \
  -v "$HOME/exl3-zeus/bench:/bench:ro" \
  -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT=29588 \
  -e RANK="$RANK" -e WORLD_SIZE="$WORLD" -e LOCAL_RANK=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e GLOO_SOCKET_IFNAME="$GLOO_IFACE" -e TP_SOCKET_IFNAME="$GLOO_IFACE" -e MN_IF_NAME="$GLOO_IFACE" \
  -e NCCL_NET=Mesh -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME="=${GLOO_IFACE}" -e NCCL_NET_PLUGIN=mesh \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" ${NCCL_DEBUG_SUBSYS:+-e NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS} \
  -e LD_LIBRARY_PATH=/opt/nccl-mesh \
  -e LAT_TAG="$TAG" ${LAT_SIZES:+-e LAT_SIZES="$LAT_SIZES"} \
  -e LAT_WARMUP="${LAT_WARMUP:-10}" -e LAT_ITERS="${LAT_ITERS:-50}" \
  -e LAT_OUT="/cache/lat-$TAG-r$RANK.json" \
  "${EXTRA[@]}" \
  --entrypoint python3 "$IMAGE" /bench/nccl-latency-bench.py

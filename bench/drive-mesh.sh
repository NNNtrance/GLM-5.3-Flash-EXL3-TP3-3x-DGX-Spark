#!/usr/bin/env bash
# drive-mesh.sh <world>   -- launches run-mesh.sh on every node, rank 0 in the foreground.
# Pass-through env: MESH_TAG MESH_OPS MESH_ENV PLUGIN_DIR NCCL_DEBUG NCCL_DEBUG_SUBSYS
#                   NCCL_ALGO NCCL_PROTO NCCL_MESH_DEBUG MESH_MAX_TOKENS
W=${1:-3}
# Your ssh targets, rank order: rank 0 first. Override with MESH_HOSTS="a b c".
read -r -a HOSTS <<< "${MESH_HOSTS:-head worker-1 worker-2}"
PASS=""
for v in MESH_TAG MESH_OPS MESH_ENV PLUGIN_DIR NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_ALGO NCCL_PROTO NCCL_MESH_DEBUG MESH_MAX_TOKENS; do
  eval "val=\${$v:-}"; [ -n "$val" ] && PASS="$PASS $v='$val'"
done
for r in $(seq 1 $((W-1))); do
  h=${HOSTS[$r]}
  ssh -o BatchMode=yes $h "nohup env $PASS bash ~/exl3-zeus/bench/run-mesh.sh $r $W > /tmp/mesh-r$r.log 2>&1 &" &
done
sleep 5
ssh -o BatchMode=yes "${HOSTS[0]}" "env $PASS bash ~/exl3-zeus/bench/run-mesh.sh 0 $W" 2>&1
for r in $(seq 1 $((W-1))); do ssh -o BatchMode=yes ${HOSTS[$r]} "docker rm -f mesh-bench >/dev/null 2>&1" ; done

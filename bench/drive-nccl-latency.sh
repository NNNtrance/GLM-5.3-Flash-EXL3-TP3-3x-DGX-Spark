#!/usr/bin/env bash
# drive-nccl-latency.sh <world> -- launches run-nccl-latency.sh on every node,
# rank 0 in the foreground. Local-only orchestration helper (like drive-mesh.sh),
# never deployed to the remote nodes.
# Pass-through env: LAT_TAG LAT_SIZES LAT_WARMUP LAT_ITERS LAT_ENV PLUGIN_SUB
#                   NCCL_DEBUG NCCL_DEBUG_SUBSYS IMAGE CPUSET MEMLIMIT
#                   MASTER_ADDR GLOO_IFACE
# Own env: LAT_HOSTS -- the ssh targets, rank order, rank 0 first.
W=${1:-3}
# Your ssh targets, rank order: rank 0 first. Override with LAT_HOSTS="a b c".
read -r -a HOSTS <<< "${LAT_HOSTS:-head worker-1 worker-2}"
PASS=""
for v in LAT_TAG LAT_SIZES LAT_WARMUP LAT_ITERS LAT_ENV PLUGIN_SUB NCCL_DEBUG NCCL_DEBUG_SUBSYS IMAGE CPUSET MEMLIMIT MASTER_ADDR GLOO_IFACE; do
  eval "val=\${$v:-}"; [ -n "$val" ] && PASS="$PASS $v='$val'"
done
for r in $(seq 1 $((W-1))); do
  h=${HOSTS[$r]}
  ssh -n -o BatchMode=yes $h "nohup env $PASS bash ~/exl3-zeus/bench/run-nccl-latency.sh $r $W > /tmp/lat-r$r.log 2>&1 &" &
done
sleep 5
ssh -n -o BatchMode=yes "${HOSTS[0]}" "env $PASS bash ~/exl3-zeus/bench/run-nccl-latency.sh 0 $W" 2>&1
for r in $(seq 1 $((W-1))); do ssh -n -o BatchMode=yes ${HOSTS[$r]} "docker rm -f nccl-lat-bench >/dev/null 2>&1"; done

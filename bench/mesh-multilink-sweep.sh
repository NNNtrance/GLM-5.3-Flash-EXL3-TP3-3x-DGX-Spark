#!/usr/bin/env bash
# mesh-multilink-sweep.sh [reps]
#
# Model-free A/B of the mesh-plugin patches 0005 (device-aware link selection)
# and 0006 (NCCL_PTR_CUDA + a real iflush), with 0004 (min_rnr_timer) carried
# into the same build.  No model, no engine.  Nine arms, each one a control for
# the next.  See docs/06 for what each arm answers and what the gate is.
#
#   THE ENGINE MUST BE DOWN ON ALL THREE NODES BEFORE THIS RUNS.
#   (it takes --gpus all and the same cpuset; a running engine makes every
#   number noise, and it holds the NICs whose counters this reads.)
#
# Output, under $OUT:
#   <tag>-<rep>.log           full sweep output: us, GB/s, rnr/oob/pse/lto per collective
#   mesh-<tag>-<rep>-r0.json  rank 0's machine-readable results, copied back
#   portdelta-<tag>.txt       per-port transmit bytes across the arm -- the direct
#                             proof that the second cable is carrying traffic
#
# Environment:
#   MESH_HOSTS      ssh targets, rank order, rank 0 first (default: head worker-1 worker-2)
#   OUT             output directory (default: ./mesh-multilink)
#   PATCHED_SUB     patched plugin directory, relative to $HOME on each node
#                   (default: exl3-zeus/nccl-mesh-patched2)
#   FABRIC_PREFIX   the leading octets your fabric subnets share, used by the
#                   pre-flight cable check (it greps the ARP table for them).
#                   Empty (default) skips that check.
set -u
REPS=${1:-3}
read -r -a HOSTS <<< "${MESH_HOSTS:-head worker-1 worker-2}"
HEAD=${HOSTS[0]}
OUT=${OUT:-$PWD/mesh-multilink}
P2=${PATCHED_SUB:-exl3-zeus/nccl-mesh-patched2}
FABRIC_PREFIX=${FABRIC_PREFIX:-}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

export MESH_OPS=allreduce,alltoall,sendrecv
# 128 KB .. 64 MB, in tokens of hidden 4096 bf16 (8192 B/token)
export MESH_TOKENS=16,32,64,128,256,512,1024,2048,4096,8192
export MESH_STEP90=1
export MESH_STEP90_TOKENS=64        # 90 x 512 KB = one C8 decode step's collectives
export NCCL_ALGO=Ring
export NCCL_MESH_DEBUG=0

engine_check() {
  local bad=0 n
  for h in "${HOSTS[@]}"; do
    n=$(ssh -n -o BatchMode=yes "$h" "docker ps --filter name=exl3-tp3 -q | wc -l" 2>/dev/null)
    [ "${n:-0}" != "0" ] && { echo "!! $h still has the engine running"; bad=1; }
  done
  [ $bad -eq 0 ] || { echo "ABORT: stop the engine on all three nodes first."; exit 2; }
}

# Per-port transmit counters, one line per node.  Cumulative since driver load,
# so only the before/after delta means anything.
snap_ports() {
  for h in "${HOSTS[@]}"; do
    ssh -n -o BatchMode=yes "$h" 'printf "%s" "$(hostname) "; for p in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do d=$(echo "$p" | cut -d/ -f5); printf "%s:%s " "$d" "$(cat "$p")"; done; echo'
  done
}

cleanup_bench() {
  for h in "${HOSTS[@]}"; do ssh -n -o BatchMode=yes "$h" "docker rm -f mesh-bench >/dev/null 2>&1"; done
}

# Fail closed if the second cable of each pair has never been reached at the IP
# layer: patch 0005 would then aim half the channels at a peer that is not
# there, and mesh_connect()'s handshake would time out.  This only reads the
# neighbour table -- it sends nothing.  Set FABRIC_PREFIX to enable it.
cable_check() {
  local h n bad=0
  [ -n "$FABRIC_PREFIX" ] || { echo "  (FABRIC_PREFIX unset -- skipped)"; return 0; }
  for h in "${HOSTS[@]}"; do
    n=$(ssh -n -o BatchMode=yes "$h" "ip -4 neigh show | grep -c '${FABRIC_PREFIX}'" 2>/dev/null)
    echo "  $h: ${n:-0} of 4 fabric neighbours resolved"
    [ "${n:-0}" -ge 4 ] || bad=1
  done
  if [ $bad -ne 0 ]; then
    echo "ABORT: the second cable of each pair has no ARP neighbour."
    echo "       Ping across each second cable by hand first (docs/06)."
    echo "       If a ping does not answer, the cabling is not what the subnets"
    echo "       imply: keep NCCL_MESH_LINKS_PER_PEER=1 and check the wiring."
    exit 3
  fi
}

run() {   # run <tag> <plugin_sub|-> <mesh_env>
  local tag=$1 plug=$2 menv=$3 r
  echo "################ ARM $tag   plugin=${plug}   env='$menv'   $(date +%H:%M:%S)"
  snap_ports > "$OUT/portdelta-$tag.txt.before"
  for r in $(seq 1 "$REPS"); do
    echo "---- $tag rep $r  $(date +%H:%M:%S)"
    if [ "$plug" = "-" ]; then unset PLUGIN_SUB; else export PLUGIN_SUB="$plug"; fi
    MESH_TAG="$tag-$r" MESH_ENV="$menv" bash "$HERE/drive-mesh.sh" 3 2>&1 | tee "$OUT/$tag-$r.log"
    cleanup_bench
    scp -q "$HEAD:/var/tmp/exl3-zeus-cache/mesh-$tag-$r-r0.json" "$OUT/" 2>/dev/null
    sleep 3
  done
  snap_ports > "$OUT/portdelta-$tag.txt.after"
  { echo "== $tag : per-port port_xmit_data, before / after"; echo;
    paste "$OUT/portdelta-$tag.txt.before" "$OUT/portdelta-$tag.txt.after"; } > "$OUT/portdelta-$tag.txt"
  rm -f "$OUT/portdelta-$tag.txt.before" "$OUT/portdelta-$tag.txt.after"
}

engine_check
echo "cable check:"; cable_check
echo "start $(date)  reps=$REPS  out=$OUT"

CH8="NCCL_MAX_NCHANNELS=8"
CH16="NCCL_MAX_NCHANNELS=16"

# 1  reference: today's production plugin, unpatched
run base           -   "$CH8"
# 2  control: the new binary told to behave exactly like the old one
run p2ctl          $P2 "$CH8 NCCL_MESH_LINKS_PER_PEER=1 NCCL_MESH_MIN_RNR_TIMER=12 NCCL_MESH_PTR_CUDA=0 NCCL_MESH_FLUSH=0"
# 3  patch 0004 alone, so 0005 does not get credit for the timer
run link1rnr1      $P2 "$CH8 NCCL_MESH_LINKS_PER_PEER=1 NCCL_MESH_MIN_RNR_TIMER=1 NCCL_MESH_PTR_CUDA=0 NCCL_MESH_FLUSH=0"
# 4  PATCH 0005: both cables
run link2          $P2 "$CH8 NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_MIN_RNR_TIMER=1 NCCL_MESH_PTR_CUDA=0 NCCL_MESH_FLUSH=0"
# 5  both cables with 8 channels each instead of 4
run link2ch16      $P2 "$CH16 NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_MIN_RNR_TIMER=1 NCCL_MESH_PTR_CUDA=0 NCCL_MESH_FLUSH=0"
# 6  PATCH 0006: no host bounce buffer, with the real flush
run link2cuda      $P2 "$CH8 NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_MIN_RNR_TIMER=1 NCCL_MESH_PTR_CUDA=1 NCCL_MESH_FLUSH=1"
# 7  what the flush costs
run link2cudanf    $P2 "$CH8 NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_MIN_RNR_TIMER=1 NCCL_MESH_PTR_CUDA=1 NCCL_MESH_FLUSH=0"
# 8  GDR changes NCCL's topology model, so re-check the channel cap
run link2cudach16  $P2 "$CH16 NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_MIN_RNR_TIMER=1 NCCL_MESH_PTR_CUDA=1 NCCL_MESH_FLUSH=1"
# 9  DMA-BUF registration instead of plain ibv_reg_mr
run link2dmabuf    $P2 "$CH8 NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_MIN_RNR_TIMER=1 NCCL_MESH_PTR_CUDA=1 NCCL_MESH_FLUSH=1 NCCL_MESH_DMABUF=1"

echo "done $(date)"
echo
echo "quick read -- 64 MB row (8192 tok) per arm:"
grep -H "^  8192 " "$OUT"/*-1.log 2>/dev/null | sed 's|.*/||'
echo
echo "quick read -- STEP90 (one C8 decode step's collectives):"
grep -H "STEP90" "$OUT"/*-1.log 2>/dev/null | sed 's|.*/||'
echo
echo "cable split (all four ports must move on the link2* arms):"
ls "$OUT"/portdelta-*.txt

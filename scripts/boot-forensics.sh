#!/usr/bin/env bash
# S0: block-layer + process forensics for one engine boot. Read-only.
# usage: boot-forensics.sh <label>   (run detached; exits when weights are loaded)
LBL="${1:-boot}"
OUT=/var/tmp/forensics-$LBL-$(hostname).txt
NAME=exl3-tp3
DEV=nvme0n1
: > "$OUT"
MINSTART="${2:-0}"   # only latch onto a container started at/after this epoch
for i in $(seq 1 180); do
  SA=$(docker inspect -f '{{.State.StartedAt}}' $NAME 2>/dev/null) || { sleep 2; continue; }
  SE=$(date -d "$SA" +%s 2>/dev/null || echo 0)
  if [ "${SE:-0}" -ge "$MINSTART" ]; then
    HP=$(docker inspect -f '{{.State.Pid}}' $NAME 2>/dev/null)
    [ -n "$HP" ] && [ "$HP" != "0" ] && { echo "latched StartedAt=$SA" ; break; }
  fi
  HP=""; sleep 2
done
[ -z "${HP:-}" ] && { echo "no container" >> "$OUT"; exit 0; }
echo "container_pid=$HP start=$(date -u +%FT%TZ)" >> "$OUT"
echo "blk_start $(cat /sys/block/$DEV/stat)" >> "$OUT"
echo "meminfo_start $(awk '/MemAvailable|SwapFree|SwapTotal/{printf "%s=%s ", $1, $2}' /proc/meminfo)" >> "$OUT"
END=$((SECONDS+1500))
while [ $SECONDS -lt $END ]; do
  docker logs $NAME 2>&1 | grep -q "Model loading took" && break
  # worker pids inside the container (they are children of $HP in the host pid ns)
  PIDS=$(pgrep -P "$HP" 2>/dev/null; pgrep -f "VLLM::Worker" 2>/dev/null)
  RB=0; WB=0
  for p in $HP $PIDS; do
    [ -r /proc/$p/io ] || continue
    r=$(awk '/^read_bytes:/{print $2}' /proc/$p/io 2>/dev/null); w=$(awk '/^write_bytes:/{print $2}' /proc/$p/io 2>/dev/null)
    RB=$((RB + ${r:-0})); WB=$((WB + ${w:-0}))
  done
  CPU=$(top -b -n1 -p "$(echo $HP $PIDS | tr ' ' ',' | sed 's/,$//')" 2>/dev/null | tail -n +8 | awk '{s+=$9} END{print s+0}')
  ST=$(ps -o stat= -p "$(echo $HP $PIDS | tr ' ' ',' | sed 's/,$//')" 2>/dev/null | tr -d ' \n')
  echo "$(date -u +%FT%TZ) rb=$RB wb=$WB cpu=$CPU stat=$ST mem_avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$OUT"
  sleep 5
done
echo "blk_end $(cat /sys/block/$DEV/stat)" >> "$OUT"
echo "meminfo_end $(awk '/MemAvailable|SwapFree|SwapTotal/{printf "%s=%s ", $1, $2}' /proc/meminfo)" >> "$OUT"
echo "done $(date -u +%FT%TZ)" >> "$OUT"

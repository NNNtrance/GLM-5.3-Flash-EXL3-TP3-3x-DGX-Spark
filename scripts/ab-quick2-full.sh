#!/usr/bin/env bash
# Tier-B quick arm for the FULL-SCOPE tree (docs/09 section 9).
#
#   scripts/ab-quick2-full.sh <label> [env-file-basename]
#
# One boot -> gates cold -> cold/warm single stream -> prefill (7K repeat and
# fresh) -> three C1..C8 sweep rounds -> gates warm -> free RAM and swap.
# Medians of three rounds; the persisted MLA tuner cache is what makes three
# enough (docs/09 section 1, docs/12). Measured noise bands on this stack:
# C1 +-4 %, C4 +-9 %, C8 +-3 %, so a difference of 3 % or less is "equal"
# (docs/09 section 1.2). It leaves the engine UP.
#
# Identical to the production quick arm except for one line: the launcher is
# started from ~/exl3-zeus/tp3full rather than ~/exl3-zeus/tp3, so the container
# mounts the full-scope patch tree. The production harness was NOT edited --
# docs/09 section 11.2: nothing is added to, removed from or edited in the patch
# directory between a dump and a load, and the same discipline is cheaper to
# keep for the harness than to reason about afterwards.
#
# Fill these in for your own cluster. Addresses here are documentation
# addresses (RFC 5737) and node names are the ones used throughout the docs.
set -u

LABEL=$1
ENVF=${2:-.env.tp3}

API=${API:-http://192.0.2.10:8001}          # head node, rank 0
NAME=${NAME:-exl3-tp3}                      # container name on every node
NODES=${NODES:-"head worker-1 worker-2"}    # all three, any order
BOOT_ORDER=${BOOT_ORDER:-"worker-2 worker-1 head"}
TREE=${TREE:-tp3full}                       # tp3full = full scope, tp3 = production
OUT=${OUT:-$PWD/arm-$LABEL}
BOOT_WAIT=${BOOT_WAIT:-2400}

mkdir -p "$OUT"
here=$(cd "$(dirname "$0")" && pwd)

echo "#### ARM [$LABEL] env=$ENVF tree=$TREE $(date +%H:%M:%S)"

# 1. Tear all three down before relaunching any of them (docs/03 section 7).
for h in $NODES; do
  ssh -n -o BatchMode=yes "$h" "docker stop -t 60 $NAME >/dev/null 2>&1; docker rm -f $NAME >/dev/null 2>&1"
done
sleep 5

# 2. Boot: workers first, head last. The launcher's own settle gate waits for
#    the host's memory to come back before docker run (docs/08 section 5.1);
#    dropping caches first is optional and is not free on a node about to read
#    56 GiB back.
for h in $BOOT_ORDER; do
  ssh -n -o BatchMode=yes "$h" "sync; \
    cd \$HOME/exl3-zeus/$TREE && setsid nohup env ENV_FILE=\$HOME/exl3-zeus/$ENVF \
      bash start-tp3.sh > /var/tmp/arm-$LABEL-launch.log 2>&1 < /dev/null &"
  sleep 8
done

# 3. Wait for /health, and fail loudly the moment a rank's container exits --
#    a boot that will never finish should cost 30 s of attention, not 40 min.
OK=0
deadline=$((SECONDS + BOOT_WAIT))
while [ $SECONDS -lt $deadline ]; do
  curl -s -m 5 "$API/health" -o /dev/null 2>/dev/null && { OK=1; echo "  [$LABEL] UP $(date +%H:%M:%S)"; break; }
  if [ "$(ssh -n -o BatchMode=yes head "docker ps --filter name=$NAME --filter status=running -q | wc -l")" != "1" ]; then
    echo "  [$LABEL] head container gone" >&2
    ssh -n -o BatchMode=yes head "docker logs $NAME 2>&1 | tail -40" | cut -c1-200
    exit 1
  fi
  sleep 20
done
[ $OK -eq 0 ] && { echo "  [$LABEL] never came up" >&2; exit 1; }

# 4. What actually got into the container, rather than what the env file says.
echo "  [$LABEL] image=$(ssh -n -o BatchMode=yes head "docker inspect -f '{{.Config.Image}}' $NAME")"
KV=$(ssh -n -o BatchMode=yes head \
  "docker logs $NAME 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+' | head -1 | grep -oE '[0-9,]+$'")
echo "  [$LABEL] KV=$KV" | tee "$OUT/kv.txt"

# 5. Gates before numbers (docs/09 section 5). Cold, then again warm at the end.
python3 "$here/correctness-probe.py" "$API" > "$OUT/probe-cold.out" 2>&1
python3 "$here/code-exam.py"         "$API" > "$OUT/code-cold.out"  2>&1
echo "  [$LABEL] gates cold: $(grep -oE 'both fields\): [0-9]+/10' "$OUT/probe-cold.out" | grep -oE '[0-9]+/10') \
$(grep -oE 'CODE EXAM: [0-9]+/12' "$OUT/code-cold.out" | grep -oE '[0-9]+/12')"

# 6. Cold/warm single stream, then prefill. A repeated prompt reads the prefix
#    cache and lies, so prefill-fresh is the number that counts (docs/09 s. 3).
python3 "$here/cold-warm-c1.py" "$API" 2>&1 | tee "$OUT/cold-warm-c1.txt" | tail -1 | sed "s/^/  [$LABEL] /"
python3 "$here/prefill-7k.py"   "$API" 2>&1 | tee "$OUT/prefill-7k.txt"   | tail -1 | sed "s/^/  [$LABEL] /"
python3 "$here/../bench/prefill-fresh.py" "$API" 3 2>&1 | tee "$OUT/prefill-fresh.txt" | tail -1 | sed "s/^/  [$LABEL] /"

# 7. Three sweep rounds, run on the head node so the client is not the bottleneck.
for r in 1 2 3; do
  echo "-- round $r"
  python3 "$here/bench-sweep.py" --base "$API" --prompts "$here/hizset-v2.jsonl" \
    --out "$OUT/sweep-$LABEL-$r.json" --label "$LABEL-$r" --think low 2>&1 | grep -E '^  C='
done | tee "$OUT/sweep.txt" | sed "s/^/  [$LABEL] /"

# 8. Gates warm. This is the run that carries weight: a fresh caching allocator
#    hands out zeroed pages, so the defect class this stack has actually
#    produced hides on a cold engine (docs/09 section 5).
python3 "$here/correctness-probe.py" "$API" > "$OUT/probe-warm.out" 2>&1
python3 "$here/code-exam.py"         "$API" > "$OUT/code-warm.out"  2>&1
echo "  [$LABEL] gates warm: $(grep -oE 'both fields\): [0-9]+/10' "$OUT/probe-warm.out" | grep -oE '[0-9]+/10') \
$(grep -oE 'CODE EXAM: [0-9]+/12' "$OUT/code-warm.out" | grep -oE '[0-9]+/12')"

# 9. The floor rule: never below 4 GiB free, and swap must not have grown.
for h in $NODES; do
  ssh -n -o BatchMode=yes "$h" 'echo "  $(hostname): free=$(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo)G swap_used=$(awk "/SwapTotal/{t=\$2}/SwapFree/{f=\$2}END{printf \"%.2f\", (t-f)/1048576}" /proc/meminfo)G"'
done | tee "$OUT/mem.txt"

echo "#### ARM [$LABEL] DONE $(date +%H:%M:%S)  raw in $OUT"

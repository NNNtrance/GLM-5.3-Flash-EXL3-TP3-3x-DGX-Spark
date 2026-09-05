#!/usr/bin/env bash
# Boot the three nodes from the FULL-SCOPE tree and wait for /health. Nothing
# else: no gates, no sweep. Use it for a dump boot, for a boot whose only
# product is the KV pool line, or when the next step is a measurement you drive
# by hand.
#
#   scripts/boot-only-full.sh <label> [env-file-basename] [wait_seconds]
#
# Copy of the production boot script; the only difference is the launcher
# directory (~/exl3-zeus/tp3full instead of ~/exl3-zeus/tp3). The production one
# was not modified -- docs/09 section 11.2.
#
# Reminder from docs/08 section 8 and docs/09 section 11: a DUMP boot's KV pool
# reads about 6.7 % low, because 56 GiB per node goes out through the page
# cache. Never record a dump boot's pool as a result.
set -u

LABEL=$1
ENVF=${2:-.env.tp3}
WAIT=${3:-2400}

API=${API:-http://192.0.2.10:8001}          # head node, rank 0
NAME=${NAME:-exl3-tp3}
NODES=${NODES:-"head worker-1 worker-2"}
BOOT_ORDER=${BOOT_ORDER:-"worker-2 worker-1 head"}
TREE=${TREE:-tp3full}

echo "#### BOOT [$LABEL] env=$ENVF tree=$TREE $(date +%H:%M:%S)"

for h in $NODES; do
  ssh -n -o BatchMode=yes "$h" "docker stop -t 60 $NAME >/dev/null 2>&1; docker rm -f $NAME >/dev/null 2>&1"
done
sleep 5

for h in $BOOT_ORDER; do
  ssh -n -o BatchMode=yes "$h" "sync; \
    cd \$HOME/exl3-zeus/$TREE && setsid nohup env ENV_FILE=\$HOME/exl3-zeus/$ENVF \
      bash start-tp3.sh > /var/tmp/boot-$LABEL-launch.log 2>&1 < /dev/null &"
  sleep 8
done

deadline=$((SECONDS + WAIT))
while [ $SECONDS -lt $deadline ]; do
  curl -s -m 5 "$API/health" -o /dev/null 2>/dev/null && { echo "  [$LABEL] UP $(date +%H:%M:%S)"; exit 0; }
  if [ "$(ssh -n -o BatchMode=yes head "docker ps --filter name=$NAME --filter status=running -q | wc -l")" != "1" ]; then
    echo "  [$LABEL] head container gone" >&2
    ssh -n -o BatchMode=yes head "docker logs $NAME 2>&1 | tail -40" | cut -c1-220
    exit 1
  fi
  sleep 20
done
echo "  [$LABEL] never came up" >&2
exit 1

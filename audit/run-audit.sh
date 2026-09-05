#!/bin/bash
# Post-install self-check for the EXL3 TP=3 stack. Run it after the engine is up
# and answering, and compare what you get against the expected ranges printed
# alongside each step and documented in audit/README.md.
#
#   ./run-audit.sh                    all sections
#   ./run-audit.sh versions fabric kv only those
#
# Sections: versions  health  fabric  kv  gates  c1  prefill  category  memory
#
# The script prints numbers. It does not grade them. Grading is audit/README.md,
# which names the source table for every expected value.
#
# Settings every expectation assumes -- change any and the expectations change:
#   image exl3-zeus:754421f (cuda-exl3 754421f carrying f3e3090),
#   checkpoint turboderp/GLM-5.3-Flash-exl3 branch 4.05bpw rev 2a30229e (FULL SCOPE),
#   TP=3 + expert parallelism, KV dtype fp8 AND fp8 draft cache,
#   --block-size 256, HAREM_SW_BLOCK_SIZE=256, DFlash2 k=7,
#   gpu-memory-utilization 0.80 (production 9) or 0.83 (production 10),
#   SETTLE_MIN_GIB=112, --max-num-seqs 8, --max-num-batched-tokens 2048,
#   NCCL_MAX_NCHANNELS=8, mesh plugin 19924dcc + patches 0004/0005/0006,
#   warm CUDA_EXL3_TUNE_CACHE, temperature 0, thinking on at effort low.
#
# Anything you publish from this script must carry those settings with it.
# A tok/s figure without its configuration is not a measurement.
#
# Written by us for this recipe; use freely (Apache-2.0).
set -u

D="$(cd "$(dirname "$0")" && pwd)"
SC="$D/../scripts"

# ---- your cluster ----------------------------------------------------------
# Override any of these in the environment, or copy scripts/cluster.env.example
# to scripts/cluster.env and edit it there (that path is gitignored).
[ -f "$SC/cluster.env" ] && . "$SC/cluster.env"
API="${API:-http://192.0.2.10:8001}"
HEAD_HOST="${HEAD_HOST:-192.0.2.10}"
NODES="${NODES:-192.0.2.10 192.0.2.11 192.0.2.12}"
CONTAINER="${CONTAINER:-exl3-tp3}"
SSH="${SSH:-ssh -o BatchMode=yes -n}"

WHAT="${*:-versions health fabric kv gates c1 prefill category memory}"
want() { case " $WHAT " in *" $1 "*) return 0;; *) return 1;; esac; }
hdr()  { echo; echo "--- $1"; }
FAILED=0

# --------------------------------------------------------------- versions ---
if want versions; then
  hdr "versions          expect: identical on all three nodes; see docs/00 section 5"
  for h in $NODES; do
    echo "  == $h"
    $SSH "$h" 'echo -n "     kernel   "; uname -r;
               echo -n "     driver   "; nvidia-smi --query-gpu=driver_version --format=csv,noheader;
               echo -n "     docker   "; docker --version;
               echo -n "     swappiness "; cat /proc/sys/vm/swappiness;
               echo -n "     default target "; systemctl get-default;
               echo -n "     hotplug file (want: absent) "; ls /etc/nvidia/cx7-hotplug-enabled 2>/dev/null || echo absent' \
      || { echo "     FAIL: could not read $h"; FAILED=1; }
  done
  echo "  (swappiness must read 60 -- do NOT set it to 0, see docs/00 section 9.1)"
  echo "  (the hotplug file must be ABSENT on all three, see docs/00 section 3)"
fi

# ----------------------------------------------------------------- health ---
if want health; then
  hdr "health            expect: HTTP 200 within a few milliseconds"
  echo "  (only the head serves the API; a worker refusing the connection is correct)"
  if curl -s -m 10 -o /dev/null -w '  /health -> HTTP %{http_code} in %{time_total}s\n' "$API/health"; then
    curl -s -m 10 "$API/v1/models" | head -c 300 | sed 's/^/  /'; echo
  else
    echo "  FAIL: no answer at $API"; FAILED=1
  fi
fi

# ----------------------------------------------------------------- fabric ---
if want fabric; then
  hdr "fabric            expect: PORT_ACTIVE 4/4 per node, and ALL FOUR xmit counters moving"
  echo "  (PORT_ACTIVE is link state, not traffic. Two flat counters per node is the"
  echo "   pre-patch condition and it cost us a factor of two -- docs/06 section 6.)"
  for h in $NODES; do
    echo -n "  $h  PORT_ACTIVE="
    $SSH "$h" 'ibv_devinfo | grep -c PORT_ACTIVE' || { echo "FAIL"; FAILED=1; }
  done
  echo "  transmit counters, snapshot 1 of 2:"
  for h in $NODES; do
    echo -n "  $h  "
    $SSH "$h" 'for p in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do
                 d=$(basename $(dirname $(dirname $(dirname $p)))); echo -n "$d:$(cat $p) "; done; echo'
  done
  echo "  run a benchmark, then re-run \`$0 fabric\` and diff: every port must move."
  echo "  PCIe link state (expect 'Speed 32GT/s, Width x4' -- x2 means SBIOS < 0104):"
  for h in $NODES; do
    echo -n "  $h  "
    $SSH "$h" 'for b in $(lspci | grep -i -E "mellanox|connectx" | cut -d" " -f1); do
                 sudo -n lspci -vv -s $b 2>/dev/null | grep -m1 LnkSta: | tr -s " "; done | sort -u | head -2' \
      || echo "(needs sudo; skipped)"
  done
fi

# --------------------------------------------------------------------- KV ---
if want kv; then
  hdr "KV pool           expect: 5,165,289 tokens at gpu-mem 0.80 (production 9)"
  echo "                          5,619,834 tokens at gpu-mem 0.83 (production 10)"
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep -E 'GPU KV cache size|Available KV cache memory|Maximum concurrency' | head -3 | cut -c40-330" \
    | sed 's/^/  /' || { echo "  FAIL: could not read the engine log on the head node"; FAILED=1; }
  echo
  echo "  BEFORE you believe that number, check the boot was a settled LOAD boot:"
  echo "  all three ranks' 'Free memory on device' and 'consumed memory' must agree"
  echo "  within 1 GiB, or the pool is an artefact of the instrument (docs/14 section 5.8)."
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep 'gpu_worker.py' | grep -o 'Free memory on device ([0-9.]*/[0-9.]*' | head -3" \
    | sed 's/^/  /'
  echo
  echo "  boot gates (all four must be present -- docs/14 section 0):"
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep -c '\[padload\]'"       | sed 's/^/    [padload] lines: /'
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep -c 'assert 5'"          | sed 's/^/    assert-5 lines: /'
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep -c 'EXL3 routed experts'" | sed 's/^/    EP evidence lines: /'
fi

# ------------------------------------------------------------------ gates ---
if want gates; then
  hdr "quality gates     expect: probe 10/10 with EMPTY content 0, code exam 12/12"
  echo "  (run these COLD after a boot AND again after the benchmark: the defect class"
  echo "   this stack has actually produced hides on a fresh allocator -- docs/09 section 5)"
  echo "  (the code exam EXECUTES model-written python locally; read scripts/code-exam.py)"
  python3 "$SC/correctness-probe.py" "$API" | sed 's/^/  /' || FAILED=1
  python3 "$SC/code-exam.py"         "$API" | sed 's/^/  /' || FAILED=1
fi

# --------------------------------------------------------------------- C1 ---
if want c1; then
  hdr "cold/warm C1      expect: ~70 tok/s aggregate, ~76 per stream, TTFT ~0.28 s"
  echo "  (production 9. Draft acceptance should sit near 62 %; the gate is >= 60 %.)"
  echo "  (on a freshly started engine the FIRST of the three is genuinely cold and"
  echo "   will be lower -- that is the point of running it first)"
  echo "  (do NOT compare against a synthetic 'count to 200' prompt: that is the"
  echo "   speculative ceiling, about 1.7x the realistic rate, not the working speed)"
  python3 "$SC/cold-warm-c1.py" "$API" | sed 's/^/  /' || FAILED=1
fi

# ---------------------------------------------------------------- prefill ---
if want prefill; then
  hdr "prefill           expect: fresh ~1,738 tok/s; warm repeated ~1,575 tok/s"
  echo "  (a prefill number measured on a REPEATED prompt reads up to 56 % high --"
  echo "   the prefix cache serves whole 3,328-token blocks and says nothing. docs/14 section 8.3)"
  if [ -x "$D/../bench/prefill-fresh.py" ] || [ -f "$D/../bench/prefill-fresh.py" ]; then
    python3 "$D/../bench/prefill-fresh.py" "$API" | sed 's/^/  fresh: /' || FAILED=1
  fi
  python3 "$SC/prefill-7k.py" "$API" | sed 's/^/  warm:  /' || FAILED=1
fi

# --------------------------------------------------------------- category ---
if want category; then
  hdr "category speed    expect (production 9, C1 mean decode / acceptance):"
  echo "    code   61.7 tok/s   46 %"
  echo "    math   79.6 tok/s   58 %"
  echo "    json   72.8 tok/s   54 %"
  echo "    prose  29.1 tok/s   13 %   <-- the drafter barely fires on free prose"
  echo "  C4 totals: code 116.1 . prose 50.7 . math 129.4 . json 110.1"
  echo "  (~14 min. The prose row is the honest headline of this stack, and it is a"
  echo "   drafter-training property, not something a setting fixes. docs/14 section 9.10)"
  python3 "$SC/category-speed.py" "$API" | sed 's/^/  /' || FAILED=1
fi

# ----------------------------------------------------------------- memory ---
if want memory; then
  hdr "free memory       expect at 0.80: 12.1 / 13.5 / 13.4 GiB free, swap ~0.1 GiB"
  echo "                  expect at 0.83: MemAvailable 8-10 GB, swap flat at ~0.1 GiB"
  echo "  RULE: never below 4 GiB free on any node. On a GB10 the GPU shares host"
  echo "  memory, so this figure IS your safety margin. If yours is lower than ours,"
  echo "  or swap is in the gigabytes, step gpu-memory-utilization down and re-audit."
  for h in $NODES; do
    echo -n "  $h  "
    $SSH "$h" "free -g | awk '/^Mem:/{printf \"free=%sGi avail=%sGi \", \$4, \$7} /^Swap:/{printf \"swap=%sGi\n\", \$3}'" \
      || { echo "FAIL"; FAILED=1; }
  done
fi

echo
if [ "$FAILED" = "0" ]; then
  echo "=== audit finished. Read the numbers above against audit/README.md;"
  echo "    this script does not grade them for you."
  echo "    If something is wrong rather than merely different, docs/14-troubleshooting.md"
  echo "    indexes every failure we hit by symptom, with the exact log line."
else
  echo "=== audit finished WITH ERRORS (a check could not run). See above."
fi
exit "$FAILED"

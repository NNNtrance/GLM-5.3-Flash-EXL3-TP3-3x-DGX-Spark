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
# Expectations below are PRODUCTION CONFIGURATION 12 (6 September 2026), which is
# what our three nodes serve. The line: production 10 (5 Sep, 0.83) -> production 11
# (6 Sep, 0.87 + the sm_12x correctness set) -> production 12 (6 Sep, 0.88 + the
# indexer workspace bound). Nothing else moved across the three.
#
# Settings every expectation assumes -- change any and the expectations change:
#   image exl3-zeus:754421f (cuda-exl3 754421f carrying f3e3090),
#   checkpoint turboderp/GLM-5.3-Flash-exl3 branch 4.05bpw rev 2a30229e (FULL SCOPE),
#   TP=3 + expert parallelism, KV dtype fp8 AND fp8 draft cache,
#   --block-size 256, HAREM_SW_BLOCK_SIZE=256, DFlash2 k=7,
#   gpu-memory-utilization 0.88 (production 12),
#     [was: 0.80 (production 9) or 0.83 (production 10); 0.87 is production 11]
#   HAREM_SM12_ITEMS=pdl,kpool and HAREM_INDEXER_WS_MODE=bound (both env-gated),
#   VLLM_DEBUG_WORKSPACE=1 for the per-boot proof of the second,
#   SETTLE_MIN_GIB=112, --max-num-seqs 8, --max-num-batched-tokens 2048,
#   NCCL_MAX_NCHANNELS=8, mesh plugin 19924dcc + patches 0004/0005/0006,
#   warm CUDA_EXL3_TUNE_CACHE, temperature 0, thinking on at effort low.
#
# Declared boot-to-boot bands on this stack (docs/09 section 1.2), which is what
# "expect" means in every speed line below: C1 +/-4 %, C2 +/-6 %, C4 +/-9 %,
# C6 +/-6 %, C8 +/-3 %, and +/-6 % on the KV pool. C1 and C8 carry a verdict;
# a C4 or C6 reading from a single boot does not.
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
  hdr "KV pool           expect: 7,041,322 tokens at gpu-mem 0.88 (production 12)"
  echo "                          pass range +/-6 %: 6,618,843 - 7,463,801"
  echo "                          three boots of production 12 read 7,170,798 / 7,088,154 / 7,041,322"
  echo "                          earlier: 6,382,920 at 0.87 (prod 11), 5,619,834 at 0.83 (prod 10),"
  echo "                                   5,165,289 at 0.80 (prod 9), 4,696,969 at 0.80 (prod 7/8)"
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
  echo
  echo "  guard patches (production 11 and 12; the prelude applies both on a cold boot too):"
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep -m1 -o 'SM12 items=[a-z,]*'" \
    | sed 's/^/    sm_12x set (want: SM12 items=pdl,kpool): /'
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep -m1 -o 'HAREM-IDXWS bound.*'" \
    | sed 's/^/    indexer workspace: /'
  echo "    (want: chosen=4067203 entries (512.0 MB), saved 4.42 GiB, headroom 2.03x)"
  echo "  workspace resize lines -- upstream's OWN counter, independent of what our patch claims."
  echo "  EXACTLY ONE per rank, reading 0.00 MB -> 513.00 MB and not -> 5036.40 MB. (The resize"
  echo "  covers the radix-top-k buffer requested in the same call, which is why it reads 513.00"
  echo "  for 512.0 MB of K-gather.) More than one line on a rank, or another figure, means the"
  echo "  buffer grew after lock_workspace() and the pool you just read is not the one you keep."
  echo "  Re-run this AFTER a long-context load: that is when it would grow (VLLM_DEBUG_WORKSPACE=1)."
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep -c 'Resized workspace'" \
    | sed 's/^/    resize lines (want: one per rank -- 3 at TP=3 in the aggregated head log): /'
  $SSH "$HEAD_HOST" "docker logs $CONTAINER 2>&1 | grep -m1 -o 'Resized workspace.*'" \
    | sed 's/^/    first resize: /'
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
  hdr "cold/warm C1      expect: 69.7 tok/s aggregate, 75.6 per stream, TTFT ~0.25 s"
  echo "  (production 12; pass range at C1's +/-4 % band is 66.93 - 72.51 aggregate."
  echo "   was: ~70 / ~76 / ~0.28 s on production 9 -- C1 has not moved across 9..12.)"
  echo "  (Draft acceptance should sit near 62.5 %, ~5.35 accepted per step; the gate is >= 60 %.)"
  echo "  (on a freshly started engine the FIRST of the three is genuinely cold and"
  echo "   will be lower -- that is the point of running it first)"
  echo "  (do NOT compare against a synthetic 'count to 200' prompt: that is the"
  echo "   speculative ceiling, about 1.7x the realistic rate, not the working speed)"
  python3 "$SC/cold-warm-c1.py" "$API" | sed 's/^/  /' || FAILED=1
fi

# ---------------------------------------------------------------- prefill ---
if want prefill; then
  hdr "prefill           expect: fresh ~1,744 tok/s (+/-3 %: 1,692 - 1,796); warm repeated 1,622-1,632"
  echo "  (production 12: 1,737 on the load boot, 1,750 on the clean boot."
  echo "   was: fresh ~1,738 / warm ~1,575 on production 9 -- prefill has not moved across 8..12.)"
  echo "  (a prefill number measured on a REPEATED prompt reads up to 56 % high --"
  echo "   the prefix cache serves whole 3,328-token blocks and says nothing. docs/14 section 8.3)"
  if [ -x "$D/../bench/prefill-fresh.py" ] || [ -f "$D/../bench/prefill-fresh.py" ]; then
    python3 "$D/../bench/prefill-fresh.py" "$API" | sed 's/^/  fresh: /' || FAILED=1
  fi
  python3 "$SC/prefill-7k.py" "$API" | sed 's/^/  warm:  /' || FAILED=1
fi

# --------------------------------------------------------------- category ---
if want category; then
  hdr "category speed    expect (PRODUCTION 12, re-measured 7 Sep, C1 mean decode / acceptance):"
  echo "  (every category is inside its own round-to-round spread against production 9, so the"
  echo "   three memory rungs and the workspace bound between them bought +37.8 % of KV pool at"
  echo "   no cost in tokens per second. results/speed/category-speeds-production-12.md)"
  echo "    code   61.5 tok/s   46 %   (production 9: 61.7 / 46 %)"
  echo "    math   76.2 tok/s   57 %   (production 9: 79.6 / 58 %)"
  echo "    json   73.1 tok/s   53 %   (production 9: 72.8 / 54 %)"
  echo "    prose  29.0 tok/s   13 %   <-- the drafter barely fires on free prose (prod 9: 29.1)"
  echo "  C4 totals: code 115.2 . prose 52.2 . math 120.6 . json 108.8"
  echo "  (~14 min. The prose row is the honest headline of this stack, and it is a"
  echo "   drafter-training property, not something a setting fixes. docs/14 section 9.10)"
  python3 "$SC/category-speed.py" "$API" | sed 's/^/  /' || FAILED=1
fi

# ----------------------------------------------------------------- memory ---
if want memory; then
  hdr "host memory       expect at 0.88 (production 12), MemAvailable min UNDER LOAD:"
  echo "                    1.52 / 3.37 / 3.36 GiB on an arm that also runs long-context stress"
  echo "                    3.15 / 4.49 / 4.45 GiB on a clean boot; swap used 0.000 GiB"
  echo "                  consumed per node 54.62 / 54.48 / 54.28 GiB (58.3-59.1 on prod 9-11)"
  echo "                  [was: 12.1 / 13.5 / 13.4 GiB free at 0.80; MemAvailable 8-10 GB at 0.83]"
  echo
  echo "  RULE: SWAP TRAFFIC UNDER LOAD, not free RAM. The old 'never below 4 GiB free'"
  echo "  floor is RETRACTED (audit/README.md section 6): on a unified-memory part most of"
  echo "  what the kernel holds is reclaimable page cache, so MemFree is the wrong ruler."
  echo "  Swap USED is a stock and sits near 0.04 GiB at every fraction including the one"
  echo "  that failed -- it discriminates nothing. Read si/so per second instead."
  echo "  A rung passes on three conditions TOGETHER: swap traffic ~0 under load, C1 inside"
  echo "  +/-4 % and C8 inside +/-3 % of a same-session reference, both gates full cold and warm."
  echo "  For scale, the two ends we measured on 6 September:"
  echo "    production 12 @0.88 PASSED  swap-in exactly 0 everywhere; 1,340 / 5 / 10 KiB out,"
  echo "                                8 of 353 samples, longest unbroken run 10 s"
  echo "    0.90          REJECTED      1,519 MiB out AND 143 MiB back in, 250 of 598 samples,"
  echo "                                85 s unbroken -- while every tok/s stayed in band"
  echo "  If yours grows through the rounds, or reads back at all, step gpu-memory-utilization"
  echo "  down and re-audit. Every 1 % of the fraction costs about 1.2 GiB of host headroom."
  echo "  MemAvailable is a BUDGET, not a gate: at 0.88 about 2 GiB is left for anything"
  echo "  running beside the engine, so a profiling run has to stop the engine first."
  for h in $NODES; do
    echo -n "  $h  "
    $SSH "$h" "free -g | awk '/^Mem:/{printf \"free=%sGi avail=%sGi \", \$4, \$7} /^Swap:/{printf \"swap=%sGi\n\", \$3}'" \
      || { echo "FAIL"; FAILED=1; }
  done
  echo "  swap traffic counters, snapshot 1 of 2 (pages in / out since boot):"
  for h in $NODES; do
    echo -n "  $h  "
    $SSH "$h" "awk '/^pswpin|^pswpout/{printf \"%s=%s \", \$1, \$2} END{print \"\"}' /proc/vmstat" \
      || { echo "FAIL"; FAILED=1; }
  done
  echo "  run a benchmark, then re-run \`$0 memory\` and diff: pswpin must not move at all."
  echo "  (for the per-second view the ladder was judged on: vmstat -n -t 1, all three nodes,"
  echo "   sampled for the whole window between 'engine up' and 'battery done')"
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

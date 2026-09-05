#!/usr/bin/env bash
# Model-free validation battery for a cuda-exl3 image.  validate.sh <tag>
set -uo pipefail
TAG=${1:-harem1}
D="docker run --rm --gpus all --shm-size 8g --cpuset-cpus 5-9,15-19 -v /var/tmp/exl3-zeus-cache:/cache -v $HOME/exl3-zeus/bench:/bench:ro -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "########## 1. upstream pytest (MoE + gemm), image exl3-zeus:$TAG"
$D -v "$HOME/exl3-zeus/build-$TAG/tests:/tests:ro" --entrypoint python3 "exl3-zeus:$TAG" \
   -m pytest -q /tests/test_exl3_moe_split.py /tests/test_exl3_moe_glu.py /tests/test_exl3_gemm.py 2>&1 | tail -8
echo "########## 2. HAREM EP kernel checks (poison + combine equivalence)"
$D --entrypoint python3 "exl3-zeus:$TAG" /bench/ep_kernel_check.py --ms 8,64,512,2048 --out /cache/ep_check_$TAG.json 2>&1 | tail -18
echo "########## 3. per-rank symmetry (was: rank 2 ~2x slower)"
for M in 2048 32; do $D --entrypoint python3 "exl3-zeus:$TAG" /bench/ep_rankstage.py $M 2>&1 | tail -5; done
echo "########## 4. expert-map placement sweep"
$D -e CASES=offsets --entrypoint python3 "exl3-zeus:$TAG" /bench/ep_mapoffset.py 2048 2>&1 | tail -10
echo "########## 5. per-stage + block_m table"
$D --entrypoint python3 "exl3-zeus:$TAG" /bench/ep_moe_bench.py --ms 8,64,512,2048 --out /cache/ep_moe_bench_$TAG.json 2>&1 | sed -n '/=== TIMING/,$p'

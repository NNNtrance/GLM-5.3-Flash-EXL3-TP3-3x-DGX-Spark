# DFlash2 speculative-decoding port for the cuda-exl3 stack — layer 2 of 2.
#
#   in  : exl3-zeus:serve-<commit>  =  vllm/vllm-openai:glm53-flash-arm64-cu130
#                                      (vLLM 0.1.dev20051+g487ecf187)
#                                      + Zeuss5/cuda-exl3 built for sm_121
#   out : exl3-zeus:<commit>
#
# Build from this directory. BASE selects which serve layer to stack on:
#   docker build --memory=4g -t exl3-zeus:f4987cf .
#   docker build --memory=4g --build-arg BASE=exl3-zeus:serve-9bf594c -t exl3-zeus:9bf594c .
#
# The base image already ships DFlash v1. This layer adds only the DFlash2 delta,
# obtained as a git 3-way merge of upstream vLLM commit b389ac294 onto the
# image's own tree (byte-identical to commit 487ecf187), plus the target-side
# EAGLE3 interface and the draft KV-cache grouping. See PATCHES.md and docs/04.
#
# Fail-closed: gate.py runs inside the build, so no image is produced unless
# every ported symbol resolves.
ARG BASE=exl3-zeus:serve-f4987cf
FROM ${BASE}

COPY files/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY gate.py /opt/harem/dflash2-gate.py

# Drop stale bytecode for the replaced modules so nothing can load a pre-port .pyc,
# then refuse to produce an image unless every ported symbol resolves.
RUN find /usr/local/lib/python3.12/dist-packages/vllm/config \
         /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models \
         /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers \
         /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu \
         -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
    python3 /opt/harem/dflash2-gate.py

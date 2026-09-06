#!/usr/bin/env bash
# Derive ~/exl3-zeus/.env.tp3-full from THIS NODE's own .env.tp3.
#
# Never copy an env file between machines: NODE_RANK, HOST_IP, OVERLAY_DIR,
# NCCL_MESH_PLUGIN_DIR and CHAT_TEMPLATE_HOST are per-node, and a copied file
# points the launcher at another machine's paths. Run this on head, worker-1 and
# worker-2 separately. It only writes .env.tp3-full; .env.tp3 is not touched.
# The published template of the file it produces is tracks/tp3/env.tp3-full.example.
set -euo pipefail
Z="$HOME/exl3-zeus"
SRC="$Z/.env.tp3"
DST="$Z/.env.tp3-full"
IMAGE_TAG="${IMAGE_TAG:-exl3-zeus:754421f}"
SIDECAR=/var/tmp/glm-5.3-flash-turboderp-4.05bpw-tp3
TARGET=/var/tmp/glm-5.3-flash-turboderp-4.05bpw
test -f "$SRC" || { echo "no $SRC on $(hostname)" >&2; exit 2; }
[ -e "$DST" ] && cp -f "$DST" "$DST.bak-$(date +%H%M%S)"

sed -e "s#^MODEL_HOST_PATH=.*#MODEL_HOST_PATH=$SIDECAR#" \
    -e "s#^MODEL_LINK_TARGET=.*#MODEL_LINK_TARGET=$TARGET#" \
    -e "s#^IMAGE=.*#IMAGE=$IMAGE_TAG#" \
    -e "s#^FASTLOAD_DIR=.*#FASTLOAD_DIR=/var/tmp/glm53-exl3-full#" \
    -e "s#^FASTLOAD_MODE=.*#FASTLOAD_MODE=dump#" \
    -e 's#^EXTRA_ENV="\(.*\)"$#EXTRA_ENV="\1 HAREM_EXL3_FULLSCOPE=1"#' \
    -e "s#^EXTRA_ARGS=.*#EXTRA_ARGS='--block-size 256 --enable-ep-weight-filter --safetensors-load-strategy eager --no-enable-flashinfer-autotune --hf-overrides {\"quantization_config_file\":\"$SIDECAR/quantization_config.json\"}'#" \
    "$SRC" > "$DST.tmp"

cat >> "$DST.tmp" <<EOF

# --- full-scope TP=3 arm -----------------------------------------------------
# Derived from THIS node's own .env.tp3 with sed; never copied between nodes.
# Rollback, whole arm:  ENV_FILE=\$HOME/exl3-zeus/.env.tp3  (the source file is
# not modified). Rollback, one line: delete HAREM_EXL3_FULLSCOPE=1 from
# EXTRA_ENV -- the patch reads the knob at run time and takes the upstream path.
#
# What changed, and why:
#   MODEL_HOST_PATH / MODEL_LINK_TARGET  the full-scope turboderp 4.05bpw
#          checkpoint plus its tp3full sidecar (66 heads, padded config, a
#          quantization_config.json carrying the packed mapping). MODEL_PATH is
#          NOT added: the sidecar is a relative symlink tree, so start-tp3.sh
#          requires the identity mount.
#   IMAGE  needs cuda-exl3 >= 754421f (f3e3090 for the padded output dim and the
#          row-parallel suh, 754421f for the vocab loader prefix). On anything
#          older the load stops at lm_head; tp3full/check-padload-tp3.py catches
#          that in the prelude, before a byte of weight is read.
#   TP3_DIR  the tree mounted at /opt/harem-tp3 is now tp3full. The production
#          tp3/ directory is NOT touched -- it is the fastload manifest identity.
#   EXTRA_ENV += HAREM_EXL3_FULLSCOPE=1   turns on S1+S2+S3+A9+A10.
#   EXTRA_ARGS  SINGLE quotes plus --hf-overrides: this checkpoint's config.json
#          carries no tensor_storage, and in double quotes the shell eats the
#          JSON's own quotes. NO SPACES inside the JSON.
#   FASTLOAD_DIR  a new name: the old one would have overwritten production 8's
#          sidecar in dump mode (start-tp3.sh mounts -v "\$_FL:\$_FL" rw).
#   FASTLOAD_MODE=dump  the identity changed (checkpoint, sidecar config and two
#          patch sets), so the previous sidecar cannot be used under any
#          circumstances. DISK: about 52-56 GB per rank. Check "df -h /var/tmp"
#          first; if there is not enough room, comment this line out (a boot
#          without fastload takes about 8-10 min) and dump separately once the
#          arm's gates have passed.
#   MAX_MODEL_LEN / GPU_MEMORY_UTILIZATION  UNCHANGED (1000000 / 0.80). The KV
#          pool is this arm's MEASURED OUTPUT; moving it is the last step.
TP3_DIR=$HOME/exl3-zeus/tp3full
# Prints for real since cuda-exl3 807d798: it lists the modules that stayed
# unquantized as "EXL3: <prefix> -> unquantized" and tallies both sides. The
# boot gate is the NEGATIVE reading -- see docs/09 section 5.
CUDA_EXL3_DEBUG_NAMES=1
EOF

mv "$DST.tmp" "$DST"
echo "wrote $DST on $(hostname)"

# FASTLOAD_MODE=dump wants ~52-56 GB for this rank's sidecar. Say so here rather
# than four minutes into a boot that then dies with ENOSPC.
_free=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc '0-9')
if [ "${_free:-0}" -lt 60 ]; then
  echo "WARNING: /var/tmp has ${_free} G free; a fastload dump wants 52-56 GB." >&2
  echo "  Either free an old sidecar, or take the first boot without fastload:" >&2
  echo "    sed -i 's/^FASTLOAD_MODE=dump/#FASTLOAD_MODE=dump/' $DST" >&2
else
  echo "  /var/tmp free: ${_free} G (a fastload dump wants 52-56 GB)"
fi
grep -nE '^(NODE_RANK|HOST_IP|IMAGE|MODEL_HOST_PATH|MODEL_LINK_TARGET|TP3_DIR|FASTLOAD_MODE|FASTLOAD_DIR|GPU_MEMORY_UTILIZATION|MAX_MODEL_LEN|EXTRA_ARGS|EXTRA_ENV|CUDA_EXL3_DEBUG_NAMES)=' "$DST"
grep -q 'MODEL_PATH=' "$DST" && { echo "REFUSED: MODEL_PATH must not be set (identity mount)" >&2; exit 3; }
grep -q 'HAREM_EXL3_FULLSCOPE=1' "$DST" || { echo "REFUSED: knob missing" >&2; exit 3; }
exit 0

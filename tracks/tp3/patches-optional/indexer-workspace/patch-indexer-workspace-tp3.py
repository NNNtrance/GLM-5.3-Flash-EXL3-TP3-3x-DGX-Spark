#!/usr/bin/env python3
"""Size the sparse-indexer K-gather workspace from its real bound, not a magic
constant tuned for a 160k-context model.  Env-gated, default OFF, fail-closed.

WHY
---
``get_max_prefill_buffer_size()`` (vllm/v1/attention/backends/mla/indexer.py)
returns ``max_model_len * 40`` ENTRIES.  Upstream's own comment explains the
40: it was picked so the indexer workspace still fits inside the
``flashmla_sparse`` workspace, using DeepSeek-V3.2's ``max_model_len``:

    40 * 163840 * 132 = 865075200 bytes = 825 MB

Each entry is ``index_head_dim`` fp8 bytes + 4 scale bytes = 132 B for
GLM-5.3-Flash.  At OUR ``max_model_len`` of 1 000 000 the same constant asks
for 40e6 entries = **4.92 GiB**.  It is reserved during the profile run
(sparse_attn_indexer_kpool.py, the ``not isinstance(attn_metadata, dict)``
branch) and locked for the life of the engine by ``lock_workspace()``
(gpu_model_runner.py).  It is not weights and not KV, so the profiler charges
all of it to the "non-torch" residual and subtracts it from the KV pool.

WHAT THE BUFFER ACTUALLY HOLDS
------------------------------
The COMPRESSED context of every prefill request in ONE indexer chunk.  Two
facts pin the size, both read off the consumers:

  1. ``split_indexer_prefill_chunks()`` (indexer.py) packs requests while
         sum(seq_len // compress_ratio) <= workspace_size
     -- note it is handed ``compressed_seq_lens_cpu``, i.e. already divided by
     ``compress_ratio``, while the workspace was sized from the UNCOMPRESSED
     ``max_model_len``.  ``compress_ratio == index_kpool`` (== 4 here).
     DeepSeek-V4 divides the buffer size by ``compress_ratio`` at its call
     site (models/deepseek_v4/attention.py); the glm5next path does not.

  2. The gather writes ``chunk.total_seq_lens`` rows -- the sum over that
     chunk's requests of the compressed lengths (``build_prefill_chunk_metadata``)
     -- into ``k_quant_full[: chunk.total_seq_lens]``
     (sparse_attn_indexer_kpool.py / sparse_attn_indexer.py).

So the largest total the scheduler can EVER present in one step is

    ceiling  = max_num_seqs * ceil((max_model_len + num_spec + 1) / compress_ratio)

and the smallest size that is still CORRECT is one request's worth,

    per_req  = ceil((max_model_len + num_spec + 1) / compress_ratio)

because a single request that does not fit is emitted as an oversized chunk by
the splitter's ``end == start`` branch instead of being split further.

At our settings (L=1e6, kpool=4, num_spec=7, max_num_seqs=8, 132 B/entry):
``per_req`` = 250 002 entries (31.5 MB), ``ceiling`` = 2 000 016 entries
(251.8 MB).  Upstream reserves 40 000 000 entries (4.92 GiB) -- 20x the
ceiling.

WHY THIS IS EXPECTED TO BE FREE
-------------------------------
Because the ceiling (251.8 MB) is below the floor this patch enforces
(512 MB), the N constraint in ``split_indexer_prefill_chunks`` never binds --
exactly as it never bound with upstream's 4.92 GiB.  The chunk list should
therefore be IDENTICAL, and prefill speed unchanged.  If speed moves, the
cause is somewhere else and the A/B must say so.

SIZING
------
    HAREM_INDEXER_WS_MODE=bound   entries = min(upstream,
                                      max(2 * ceiling, floor, per_req))
                                  floor = 512 MB worth of entries
    HAREM_INDEXER_WS_MB=<n>       entries = min(upstream, n MB worth)
                                  (explicit override; still floor-checked)
    unset / "off" / "upstream"    upstream behaviour, byte for byte

The x2 safety factor is on top of an EXACT ceiling, and the 512 MB floor wins
over it here, giving ~2.03x headroom over anything the scheduler can produce.

SAFETY -- WHY A TOO-SMALL BUFFER CANNOT CORRUPT SILENTLY
--------------------------------------------------------
Three layers, in order of when they fire:

  L1  Startup, this patch: if the chosen size is below ``per_req`` the engine
      REFUSES TO START (RuntimeError).  An explicit HAREM_INDEXER_WS_MB that
      would break correctness is rejected before a byte is committed.

  L2  Metadata build, this patch: ``split_indexer_prefill_chunks`` raises if it
      is about to emit a chunk whose compressed total exceeds ``workspace_size``
      -- i.e. the ``end == start`` branch.  This fires before any kernel launch.

  L3  Gather, this patch: right at ``k_quant_full[: chunk.total_seq_lens]``.
      This is the actual corruption point and the reason L3 exists at all: a
      Python slice CLAMPS silently (``t[:N]`` on a shorter tensor returns the
      shorter tensor), so without a check ``cp_gather_indexer_k_quant_cache``
      would be handed ``cu_seq_lens`` whose last entry exceeds the buffer it
      writes into -- an out-of-bounds DEVICE write, not an exception.

  L0  (upstream, unconditional) ``WorkspaceManager._ensure_workspace_size``
      raises ``AssertionError`` naming the caller when a locked workspace is
      too small (v1/worker/workspace.py).  Every access goes through
      ``get_simultaneous`` -> ``_ensure_workspace_size`` BEFORE the buffer is
      sliced, so an under-sized workspace can never be silently indexed past;
      after ``lock_workspace()`` the manager cannot grow, it can only raise.

L2 and L3 are armed only when the knob is set, so the control arm of an A/B is
upstream byte for byte.

INSTALL
-------
Run in the prelude next to the other TP=3 patches:

    run python3 "$TP3_DIR/patch-indexer-workspace-tp3.py" --root "$VLLM_PY"

then arm it in EXTRA_ENV:

    HAREM_INDEXER_WS_MODE=bound

NOTE: adding a file under the production patch tree changes the fast-load
manifest identity (harem_fastload_id.file_identity() hashes every
``patch-*.py`` plus tp3-prelude.sh), so the first boot with this patch present
cannot reuse the production sidecar.  Check ``df`` before you decide how to
boot it: a second sidecar is ~53 GB per node, and two of our three nodes did
not have room for one.  The README beside this file has the two ways round it.

Usage:  patch-indexer-workspace-tp3.py [--check]
                                       [--root /usr/local/lib/python3.12/dist-packages/vllm]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MARKER = "HAREM-IDXWS"
DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")

# =============================================================================
# 1. indexer.py -- the helper, inserted right after the module logger
# =============================================================================

IDX_HELPER_OLD = '''logger = init_logger(__name__)


@triton.jit
def _prepare_uniform_decode_kernel(
'''

IDX_HELPER_NEW = '''logger = init_logger(__name__)

# --- HAREM-IDXWS: bound the sparse-indexer K-gather workspace ---------------
# Upstream sizes it as 40 * max_model_len ENTRIES, a constant tuned for
# DeepSeek-V3.2 (max_model_len 163840 -> 825 MB).  At max_model_len = 1e6 the
# same constant reserves 40e6 * 132 B = 4.92 GiB, live for the life of the
# engine and charged to the KV budget.  The buffer only ever holds ONE indexer
# chunk's COMPRESSED context, so its real ceiling is
#     max_num_seqs * ceil((max_model_len + num_spec + 1) / compress_ratio)
# and its correctness floor is one request's worth of that.  See
# patch-indexer-workspace-tp3.py for the derivation and the safety argument.
#
# Off unless HAREM_INDEXER_WS_MODE=bound or HAREM_INDEXER_WS_MB=<n>.  With
# neither knob set this is one dict lookup per call and upstream's value.
import os as _harem_idxws_os

_HAREM_IDXWS_MIB = 1024 * 1024
_HAREM_IDXWS_SAFETY = 2  # x2 on top of an EXACT ceiling
_HAREM_IDXWS_FLOOR_MB = 512  # never below this, whatever the arithmetic says
_harem_idxws_logged = False

# Armed for the two runtime guards below (L2) and in
# sparse_attn_indexer_kpool.py (L3).  Same condition as the sizing knob.
_HAREM_IDXWS_GUARD = bool(
    _harem_idxws_os.environ.get("HAREM_INDEXER_WS_MODE", "").strip().lower()
    not in ("", "off", "upstream")
    or _harem_idxws_os.environ.get("HAREM_INDEXER_WS_MB", "").strip()
)


def _harem_idxws_hf(vllm_config):
    """The HF text config, whichever attribute this vLLM exposes."""
    mc = vllm_config.model_config
    return getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)


def _harem_idxws_entry_bytes(vllm_config) -> tuple[int, int]:
    """(bytes per workspace entry, index head_dim), from _gather_workspace_shapes:
    FP8 path (T, head_dim) fp8 + (T, 4) uint8; MXFP4 path (T, head_dim//2) uint8
    + (T, head_dim//32) uint8."""
    head_dim = 128
    try:
        head_dim = int(getattr(_harem_idxws_hf(vllm_config), "index_head_dim", 0) or 128)
    except Exception:
        head_dim = 128
    use_fp4 = False
    try:
        use_fp4 = bool(vllm_config.attention_config.use_fp4_indexer_cache)
    except Exception:
        use_fp4 = False
    if use_fp4:
        return head_dim // 2 + head_dim // 32, head_dim
    return head_dim + 4, head_dim


def _harem_idxws_entries(vllm_config, upstream_entries: int):
    """Bounded entry count, or None to keep upstream's sizing.

    Raises rather than returning a size that a single max-length request could
    not fit -- that is the one case the splitter cannot chunk its way out of.
    """
    global _harem_idxws_logged

    mode = _harem_idxws_os.environ.get("HAREM_INDEXER_WS_MODE", "").strip().lower()
    mb_s = _harem_idxws_os.environ.get("HAREM_INDEXER_WS_MB", "").strip()
    if mode in ("", "off", "upstream") and not mb_s:
        return None  # default: upstream, byte for byte
    if mode not in ("", "off", "upstream", "bound"):
        raise RuntimeError(
            f"HAREM-IDXWS: HAREM_INDEXER_WS_MODE={mode!r} is not one of "
            "'bound' / 'off' / 'upstream'. Refusing to guess."
        )

    max_model_len = int(vllm_config.model_config.max_model_len)
    entry_bytes, head_dim = _harem_idxws_entry_bytes(vllm_config)

    # compress_ratio for the indexer == index_kpool on the glm5next kpool path
    # (Glm5NextIndexerCache sets compress_ratio = index_kpool on the spec).
    # Unknown -> 1, i.e. assume NO compression: a bigger, safer buffer.
    try:
        kpool = int(getattr(_harem_idxws_hf(vllm_config), "index_kpool", 0) or 0)
    except Exception:
        kpool = 0
    if kpool < 1:
        kpool = 1

    num_spec = 0
    try:
        spec = vllm_config.speculative_config
        if spec is not None:
            num_spec = int(spec.num_speculative_tokens or 0)
    except Exception:
        num_spec = 0

    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)

    # One request's compressed context: the CORRECTNESS floor.  +num_spec+1
    # because the builder chunks on seq_lens_cpu_upper_bound, which may run
    # ahead of the committed length by the speculative window.
    per_req = -(-(max_model_len + num_spec + 1) // kpool)
    # Everything the scheduler can present at once: the EXACT ceiling.
    ceiling = max_num_seqs * per_req
    floor_entries = (_HAREM_IDXWS_FLOOR_MB * _HAREM_IDXWS_MIB) // entry_bytes

    if mb_s:
        try:
            want_mb = int(mb_s)
        except ValueError:
            raise RuntimeError(
                f"HAREM-IDXWS: HAREM_INDEXER_WS_MB={mb_s!r} is not an integer."
            )
        if want_mb <= 0:
            raise RuntimeError(
                f"HAREM-IDXWS: HAREM_INDEXER_WS_MB={want_mb} must be positive."
            )
        chosen = (want_mb * _HAREM_IDXWS_MIB) // entry_bytes
        source = f"HAREM_INDEXER_WS_MB={want_mb}"
    else:
        chosen = max(_HAREM_IDXWS_SAFETY * ceiling, floor_entries, per_req)
        source = "bound"

    # Never ask for MORE than upstream would have.
    chosen = min(chosen, upstream_entries)

    # ---- fail closed --------------------------------------------------------
    if chosen < per_req:
        raise RuntimeError(
            "HAREM-IDXWS refuses to size the indexer K-gather workspace below "
            "one request's compressed context: "
            f"chosen={chosen} entries ({chosen * entry_bytes / _HAREM_IDXWS_MIB:.1f} MB) "
            f"< per_request={per_req} entries "
            f"({per_req * entry_bytes / _HAREM_IDXWS_MIB:.1f} MB) "
            f"[max_model_len={max_model_len} compress_ratio={kpool} "
            f"num_spec={num_spec} entry_bytes={entry_bytes} source={source}]. "
            "split_indexer_prefill_chunks() cannot chunk a single request any "
            "further, so this would hand the gather kernel an out-of-range "
            "cu_seq_lens. Raise HAREM_INDEXER_WS_MB or unset it."
        )

    if not _harem_idxws_logged:
        _harem_idxws_logged = True
        logger.info(
            "HAREM-IDXWS %s | upstream=%d entries (%.1f MB) -> chosen=%d entries "
            "(%.1f MB), saved %.2f GiB | max_model_len=%d compress_ratio=%d "
            "entry_bytes=%d (index_head_dim=%d) max_num_seqs=%d num_spec=%d "
            "| per_request_floor=%d (%.1f MB) scheduler_ceiling=%d (%.1f MB) "
            "headroom=%.2fx | safety=%dx floor=%d MB",
            source,
            upstream_entries,
            upstream_entries * entry_bytes / _HAREM_IDXWS_MIB,
            chosen,
            chosen * entry_bytes / _HAREM_IDXWS_MIB,
            (upstream_entries - chosen) * entry_bytes / (1024**3),
            max_model_len,
            kpool,
            entry_bytes,
            head_dim,
            max_num_seqs,
            num_spec,
            per_req,
            per_req * entry_bytes / _HAREM_IDXWS_MIB,
            ceiling,
            ceiling * entry_bytes / _HAREM_IDXWS_MIB,
            (chosen / ceiling) if ceiling else float("inf"),
            _HAREM_IDXWS_SAFETY,
            _HAREM_IDXWS_FLOOR_MB,
        )
    return chosen


# --- end HAREM-IDXWS


@triton.jit
def _prepare_uniform_decode_kernel(
'''

# =============================================================================
# 2. indexer.py -- use it in get_max_prefill_buffer_size
# =============================================================================

IDX_SIZE_OLD = '''    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
'''

IDX_SIZE_NEW = '''    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    # --- HAREM-IDXWS: env-gated real-bound sizing. Returns None (-> upstream
    # value below, byte for byte) unless HAREM_INDEXER_WS_MODE/_MB is set.
    _harem_bounded = _harem_idxws_entries(vllm_config, max_model_len * 40)
    if _harem_bounded is not None:
        return _harem_bounded
    # --- end HAREM-IDXWS
    return max_model_len * 40
'''

# =============================================================================
# 3. indexer.py -- L2 guard: refuse to emit a chunk larger than the workspace
# =============================================================================

IDX_GUARD_OLD = '''        req_slice = slice(start + request_offset, end + request_offset)
'''

IDX_GUARD_NEW = '''        req_slice = slice(start + request_offset, end + request_offset)
        # --- HAREM-IDXWS L2: the `end == start` branch above deliberately
        # emits a chunk that can exceed workspace_size (a single request the
        # packer could not split).  Downstream that becomes
        # k_quant_full[:chunk.total_seq_lens] on a shorter buffer -- a SILENT
        # Python-slice clamp, then an out-of-bounds device write in
        # cp_gather_indexer_k_quant_cache.  Armed only with the sizing knob.
        if _HAREM_IDXWS_GUARD and chunk_n > workspace_size:
            raise AssertionError(
                "HAREM-IDXWS: indexer prefill chunk needs "
                f"{chunk_n} compressed entries but the K-gather workspace holds "
                f"{workspace_size}. Requests {req_slice.start}:{req_slice.stop}. "
                "Unset HAREM_INDEXER_WS_MODE / HAREM_INDEXER_WS_MB to fall back "
                "to upstream sizing."
            )
        # --- end HAREM-IDXWS
'''

# =============================================================================
# 4. sparse_attn_indexer_kpool.py -- arm flag
# =============================================================================

KP_FLAG_OLD = '''logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
'''

KP_FLAG_NEW = '''logger = init_logger(__name__)

# --- HAREM-IDXWS: armed by the same knob that resizes the K-gather workspace
# (patch-indexer-workspace-tp3.py). Unset knob == upstream behaviour.
import os as _harem_idxws_os

_HAREM_IDXWS_GUARD = bool(
    _harem_idxws_os.environ.get("HAREM_INDEXER_WS_MODE", "").strip().lower()
    not in ("", "off", "upstream")
    or _harem_idxws_os.environ.get("HAREM_INDEXER_WS_MB", "").strip()
)
# --- end HAREM-IDXWS

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
'''

# =============================================================================
# 5. sparse_attn_indexer_kpool.py -- L3 guard, at the corruption point itself
# =============================================================================

KP_GUARD_OLD = '''        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
'''

KP_GUARD_NEW = '''        for chunk in prefill_metadata.chunks if not short_prefill else ():
            # --- HAREM-IDXWS L3: last line of defence, exactly where a
            # too-small workspace would start corrupting. The slice below
            # CLAMPS silently, and cp_gather_indexer_k_quant_cache would then
            # write cu_seq_lens[-1] rows into a shorter buffer.
            if _HAREM_IDXWS_GUARD and chunk.total_seq_lens > k_quant_full.shape[0]:
                raise AssertionError(
                    "HAREM-IDXWS: K-gather workspace too small: chunk needs "
                    f"{chunk.total_seq_lens} entries, workspace holds "
                    f"{k_quant_full.shape[0]}. Unset HAREM_INDEXER_WS_MODE / "
                    "HAREM_INDEXER_WS_MB to fall back to upstream sizing."
                )
            # --- end HAREM-IDXWS
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
'''

FILES = [
    (
        "v1/attention/backends/mla/indexer.py",
        [
            (IDX_HELPER_OLD, IDX_HELPER_NEW),
            (IDX_SIZE_OLD, IDX_SIZE_NEW),
            (IDX_GUARD_OLD, IDX_GUARD_NEW),
        ],
    ),
    (
        "model_executor/layers/sparse_attn_indexer_kpool.py",
        [
            (KP_FLAG_OLD, KP_FLAG_NEW),
            (KP_GUARD_OLD, KP_GUARD_NEW),
        ],
    ),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def apply_file(path: Path, edits, check_only: bool) -> bool:
    src = path.read_text()
    if MARKER in src:
        print(f"  {path.name}: already patched (sha {sha(path)})")
        return False
    missing = [old for old, _ in edits if src.count(old) != 1]
    if missing:
        for old in missing:
            n = src.count(old)
            print(
                f"  ANCHOR {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}: "
                + old.strip().splitlines()[0][:90],
                file=sys.stderr,
            )
        raise SystemExit(f"{path}: anchor did not match exactly once.")
    if check_only:
        print(f"  {path.name}: {len(edits)} anchors OK, NOT patched (--check)")
        return False
    for old, new in edits:
        src = src.replace(old, new, 1)
    before = sha(path)
    path.write_text(src)
    print(f"  {path.name}: patched {len(edits)} sites ({before} -> {sha(path)})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"no vllm package at {args.root}")
    print(f"HAREM-TP3 indexer K-gather workspace bound in {args.root}")
    changed = 0
    for rel, edits in FILES:
        path = args.root / rel
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        changed += bool(apply_file(path, edits, args.check))
    print(
        "HAREM-TP3 idxws: "
        + (
            "anchors verified"
            if args.check
            else f"{changed} file(s) changed "
            "(set HAREM_INDEXER_WS_MODE=bound to arm)"
        )
    )


if __name__ == "__main__":
    main()

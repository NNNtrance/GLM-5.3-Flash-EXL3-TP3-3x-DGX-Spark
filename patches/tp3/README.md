# patches/tp3 — in-container patches for TP=3

Everything here runs inside the serving container, before `vllm serve` starts, applied in order by
[`scripts/tp3-prelude.sh`](../../scripts/tp3-prelude.sh). Every patch is an exact-anchor text
substitution against the installed package (vLLM or `cuda_exl3`): if the anchor is missing or matches
more than once, the script exits non-zero instead of half-patching, and the prelude's `run` helper
stops the rank on that exit code (`TP3_STRICT=0` turns that into a warning — not a way to get past a
broken anchor, see the prelude's own header). None of them touch the image; they run against the
container's writable layer, so rollback never needs a rebuild.

The full ordered list of what the prelude calls, and which calls are gated behind which environment
variable, is in [docs/08](../../docs/08-fast-boot.md) §11.

## Inventory

| File | What it is |
|---|---|
| `patch-vllm-tp3.py` | Base in-container vLLM patches for the TP=3 padding arrangement. Always applied. |
| `patch-exl3-ep.py` + `overlay/cuda_exl3/_harem_ep.py` | Teaches `cuda-exl3`'s routed-expert method to run under expert parallel. Always applied. |
| `patch-dflash-tp3.py` | DFlash2 drafter patches for the padded sidecar (32/8 → 36/9 at TP=3). Applied only when the image carries the DFlash2 port. |
| `patch-kvdiag-tp3.py` | Logging only: prints the per-group decomposition of the KV pool arithmetic. Always applied, no behaviour change. |
| `patch-swblock-tp3.py` | See [Production 3](#production-3--the-drafts-own-kernel-block-patch-swblock-tp3py) below. |
| `patch-epfilter-tp3.py` | See [Production 4](#production-4--the-fast-boot-sidecar-patch-epfilter-tp3py-patch-fastload-tp3py) below. |
| `patch-fastload-tp3.py`, `harem_fastload.py`, `harem_fastload_id.py`, `preflight-fastload.py` | See [Production 4](#production-4--the-fast-boot-sidecar-patch-epfilter-tp3py-patch-fastload-tp3py) below. |
| `patch-zerokv-tp3.py`, `gate-test2.py` (part) | See [Production 7 — the KV memset gate](#the-kv-memset-gate-patch-zerokv-tp3py--shipped-off-refuses-on-this-checkpoint) below. |
| `patch-draftkv-tp3.py`, `gate-test2.py` (part) | See [Production 7 — the draft cache](#the-draft-cache-at-fp8-patch-draftkv-tp3py) below. |
| `patch-tilelang-failloud-tp3.py`, `flashinfer-warmup.py` | See [Production 7 — the import guard](#the-flashinfer-import-guard-patch-tilelang-failloud-tp3py--flashinfer-warmuppy) below. |
| `patch-mhcfused-tp3.py` + `overlay/vllm/harem_hc_fusion.py` | See [Not in production](#not-in-production--patch-mhcfused-tp3py--overlayvllmharem_hc_fusionpy) below. |
| `pad-tp3.py` | Offline: builds the TP=3 sidecar (symlink tree + padded `config.json`) from the downloaded checkpoint. Run once, outside the container. |
| `preflight-tp3.py` | Offline/prelude: model-free divisibility checks on the sidecar, before a boot that would otherwise die at load time. |
| `gate-test.py`, `gate-test2.py` | CPU-only behavioural tests for the three arms below — see [Testing](#testing). |

## Production 3 — the draft's own kernel block (`patch-swblock-tp3.py`)

**What it does.** The DFlash2 draft runs as an independently grouped sliding-window attention spec,
which never reaches upstream's `unify` step — so upstream's "smallest kernel block that fits is fine"
sizing no longer holds once the draft's own page has been grown (see
[docs/07](../../docs/07-kv-and-draft-page.md) §3). The patch lets the launcher pick a larger block
directly.

**Env.** `HAREM_SW_BLOCK_SIZE=<int>` — must be a multiple of the attention backend's own kernel block,
checked at patch time; the patch raises rather than silently rounding.

**Default.** `0` or unset → upstream sizing, byte for byte.

**Introduced in production 3**, the "draft page 256" arm ([docs/07](../../docs/07-kv-and-draft-page.md)
§3; KV pool 4,413,223 in that row of [docs/10](../../docs/10-results-and-roofline.md) §2). Production
today runs `HAREM_SW_BLOCK_SIZE=256`.

## Production 4 — the fast-boot sidecar (`patch-epfilter-tp3.py`, `patch-fastload-tp3.py`)

Both patches shipped together as part of the fast-boot arm documented end to end in
[docs/08](../../docs/08-fast-boot.md) §2 (S1 and S4 there).

**`patch-epfilter-tp3.py`.** Upstream's EP weight filter (`should_skip_weight`) only recognises names
ending `.weight` / `.weight_packed`. An EXL3 checkpoint keeps 99.8 % of an expert's bytes in
`<proj>.trellis`, so without this patch `--enable-ep-weight-filter` reads every expert on every rank
anyway. Env `HAREM_EP_FILTER_SUFFIXES` (comma separated, default `.trellis`); the `.suh`/`.svh`/`.mcg`
per-expert scale tensors are deliberately left out of the list and still read on every rank. The patch
itself is unconditional, but it is inert unless `--enable-ep-weight-filter` is also passed — with that
flag absent, behaviour is upstream's, byte for byte.

**`patch-fastload-tp3.py`** (+ its siblings `harem_fastload.py`, `harem_fastload_id.py`,
`preflight-fastload.py`, already in this directory before this update). Routes
`BaseModelLoader.load_model` through a per-rank pre-sliced weight sidecar. Env `HAREM_FASTLOAD_MODE`
= `dump` (write the sidecar) or `load` (read it); `preflight-fastload.py` runs in the prelude and
refuses the boot if the sidecar's identity does not match the mounted checkpoint, image and patch set.

**Default.** `HAREM_FASTLOAD_MODE` unset → the hook is a straight call to the original
`load_weights(...)`; both patches are no-ops.

**Introduced in production 4**, the "fast-boot sidecar" arm: boot 618 s → 274 s
([docs/08](../../docs/08-fast-boot.md); KV pool 4,484,848 in that row of
[docs/10](../../docs/10-results-and-roofline.md) §2).

## Production 7

Three patches shipped together, named as one set in
[docs/10](../../docs/10-results-and-roofline.md) §1 ("production 6 plus the fp8 draft cache, the
tilelang fail-loud guard, a FlashInfer warm-up and the launcher's memory settle gate"). Only two of
the three are active in the shipped production 7 configuration — see each entry.

### The draft cache at fp8 (`patch-draftkv-tp3.py`)

**What it does.** The launcher pins the DFlash2 drafter's KV cache to `"auto"` (which means "inherit
bf16 from nothing", not "inherit the main fp8 cache" — `auto` only inherits when the field is left
unset entirely). This patch overrides `SpeculativeConfig.kv_cache_dtype` from the environment at the
top of `__post_init__`, before any validation reads it, so the launcher's own JSON is never touched.

**Env.** `HAREM_DRAFT_KV_DTYPE` = one of `auto`, `bfloat16`, `float16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`;
invalid values raise at patch time.

**Default.** Unset → upstream behaviour, whatever the launcher's `--speculative-config` JSON says
(`auto`, i.e. bf16 in this stack).

**Introduced in, and active in, production 7**: `HAREM_DRAFT_KV_DTYPE=fp8` moved the draft cache from
bf16 to fp8, pool 4,449,035 → 4,699,724 (+5.6 %), validated safe by acceptance staying in the 60–65 %
gate band ([docs/07](../../docs/07-kv-and-draft-page.md) §7).

### The KV memset gate (`patch-zerokv-tp3.py`) — shipped, off, refuses on this checkpoint

**What it does.** vLLM zeroes every newly allocated KV block when the cache has Mamba/KDA layers *or*
mixed precision. On this checkpoint that memset costs 1.2–1.4 % of a prefill chunk while running at
100 % of the memset roofline — nothing to win inside the kernel, so the only lever is not running it
at all. This patch adds a fail-closed gate: when asked to skip the memset, it re-derives both
conditions from the engine's own `KVCacheConfig` and only allows the skip if every one of them holds;
otherwise it `raise`s at startup rather than serving.

**Env.** `HAREM_ZERO_ATTENTION_KV=0` requests the skip.

**Default.** Unset or `1` → upstream behaviour, byte for byte (always zero).

**Shipped as part of production 7's patch set but not part of its active configuration.** The
mixed-precision half of the condition looked like the only one that mattered — draft fp8 was expected
to close it — but the Mamba/KDA half is independent of precision: in this model's hybrid layout, one
KV tensor is co-owned by an MLA layer and a Mamba/KDA layer from the same group (measured: 85.5 % of
what gets zeroed per block is that co-owned region). So `HAREM_ZERO_ATTENTION_KV=0` on this checkpoint
**correctly refuses to boot** ("UNSAFE ... co-owned by attention and Mamba/KDA layers") rather than
silently skipping something that was never safe to skip. The gate is kept and documented because the
refusal is the correct, intended behaviour, not a bug — see
[docs/07](../../docs/07-kv-and-draft-page.md) §8. The safe remainder if the cache were fully uniform
(indexer + draft only) is 0.19 % of prefill, not worth a separate partial mode.

### The flashinfer import guard (`patch-tilelang-failloud-tp3.py` + `flashinfer-warmup.py`)

**What it does.** `tilelang_kernels.py` preloads `flashinfer.comm` inside
`contextlib.suppress(Exception)` so that its `CudaRTLibrary` binds the real `libcudart` before
`tilelang` maps `libcudart_stub.so` first and shadows it — by upstream's own comment, "import order is
load-bearing". A transient failure in that preload is swallowed silently, and the only symptom is an
illegal address at kernel launch much later, on a different rank, at a time that looks random.
`patch-tilelang-failloud-tp3.py` replaces the suppression with try/print/raise so a failed preload
stops the rank in its first second with a named cause. `flashinfer-warmup.py` is the companion the
prelude runs once per rank, unconditionally: it imports `flashinfer.comm` on the CPU before any worker
starts, prints the version into the boot log (so the import is visible even when it succeeds), and
warms flashinfer's own JIT cache so N ranks do not race it (~2 s).

**Env.** `HAREM_TILELANG_FAILLOUD=1` turns the silent suppression into an immediate error.
`HAREM_FLASHINFER_WARMUP=0` skips the warm-up step.

**Default.** `HAREM_TILELANG_FAILLOUD` unset or `0` → upstream behaviour, byte for byte (suppress and
continue). The warm-up step defaults to on but never fails the boot unless failloud is also on.

**Introduced in production 7** as a diagnostic guard alongside the fp8 draft cache
([docs/10](../../docs/10-results-and-roofline.md) §1); named in
[docs/11](../../docs/11-open-issues.md) as one of the three additions that made a stale fast-load
sidecar refuse a boot (a patch-directory change invalidates the sidecar's identity hash regardless of
whether the patch's own knob is set — [docs/09](../../docs/09-measurement-protocol.md) §11).

## Not in production — `patch-mhcfused-tp3.py` + `overlay/vllm/harem_hc_fusion.py`

**What it does.** The hyper-connection block (`hc_mult=4`) runs three kernels per call on the large-M
(prefill) path; the second kernel's entire job is re-reading the 32 KB/token row the first kernel just
wrote. This patch routes the large-M case to a Triton kernel (`harem_hc_fusion.py`) that does both
steps in one pass, reducing the freshly computed row against the projection while it is still in
registers, deleting the re-read.

**Env.** `HAREM_MHC_FUSED_LARGE=1` (master switch). `HAREM_MHC_FUSED_MIN_M` — minimum token count for
the fused path. **Note on this one:** the patch script's own docstring says "default 256", but the
kernel module's shipped code sets `MIN_M = _env_int("HAREM_MHC_FUSED_MIN_M", 1024)` — 256 was the
threshold measured and found to lose (+37.7 % at M=512, because at that size `residual_cur` fits
entirely inside the 24 MiB L2 and the re-read the fusion removes was never going to DRAM); the code
default was corrected to 1024 and the patch's comment was not updated to match. Read the code, not
the comment, until that comment is fixed.

**Default.** `HAREM_MHC_FUSED_LARGE` unset or `0` → upstream three-kernel path, byte for byte; also
off below `HAREM_MHC_FUSED_MIN_M` tokens and for every decode/small-M call.

**Status: written and statically verified, never run on a GPU, not in production.** `py_compile`
clean, imports without a GPU, AOT-compiles for sm_121 in 18/18 swept configurations, and the anchor
dry-run (`--check`) passes against the image's `tilelang.py`. Measured **model-free** (not on the live
engine) at −14.9 to −15.5 % on the fused kernel pair, which is only −1.0 to −1.1 % of the prefill
wall — about 40 % of the −2.5…−2.7 % ceiling the docstring describes, because the fused kernel only
reaches 187.7 GB/s against the 229.5 GB/s the route it replaces gets, and a full configuration sweep
could not close that gap. That is below the bar to justify a boot on its own; it is expected to ride
the next image bundle together with `exl3_moe_had_in`, where one boot measures both
([docs/11](../../docs/11-open-issues.md) §2.16, §2.19). Apache-2.0, same as the rest of this
directory — use it freely; it is kept here so the finding is reproducible, not as a recommendation to
enable it today.

## Testing

`gate-test.py` and `gate-test2.py` are CPU-only behavioural checks — no engine, no GPU, no model.
Both extract the exact code the patches install (by regex, out of an already-patched file under
`/usr/local/lib/python3.12/dist-packages/vllm/...`) and `exec` it, so the test runs the shipped code,
not a re-implementation of it.

- `gate-test.py` exercises `patch-zerokv-tp3.py`'s safety gate against eight synthetic
  `KVCacheConfig` shapes, including the production shape (expect: refuses, "co-owned by").
- `gate-test2.py` covers `patch-draftkv-tp3.py` (env override, invalid-value rejection) and
  `patch-tilelang-failloud-tp3.py` (suppressed vs. raised import failure, in both the on and off
  positions of the knob).

Run either after applying the corresponding patch(es) inside the container:

```
python3 gate-test.py
python3 gate-test2.py
```

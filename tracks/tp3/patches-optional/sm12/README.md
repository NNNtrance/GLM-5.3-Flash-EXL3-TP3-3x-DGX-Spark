# `patches-optional/sm12` — three sm_12x fixes that are measured, free, and **not in production**

**Applies to: TP=3 as published here; the patches themselves are rank-agnostic.** Nothing in these
three files depends on how many ranks read them. They are filed under the three-node track because
that is the tree they were measured in, and because on this track copying them anywhere near the
production directory has a cost — see [the sidecar warning](#the-sidecar-warning-read-this-before-you-copy-anything).

**These are not part of the recipe.** Production configuration 10 does not carry them and this
directory is deliberately outside [`../../patches/`](../../patches/), the tree the prelude actually
applies. Everything measured about them is in
[`results/kernels/sm12-stack-patches-ab.md`](../../../../results/kernels/sm12-stack-patches-ab.md); the
decision is in [docs/11](../../../../docs/11-open-issues.md) §2.27.

**Credit.** All four findings are `tpurtell/glm-5.3-flash-ext3-2x-rtx`'s (Apache-2.0), raised as
[Zeuss5/cuda-exl3 issue #6](https://github.com/Zeuss5/cuda-exl3/issues/6). These scripts are
re-implementations against our own anchors — our image is not theirs — and the Apache-2.0 attribution
travels with them. See [CREDITS](../../../../CREDITS.md).

---

## What each one fixes

| File | Item | Class | What it does |
|---|---|---|---|
| [`patch-pdl-gate.py`](patch-pdl-gate.py) | 1 | correctness | `is_arch_support_pdl()` returns `major >= 9` upstream, which turns Programmatic Dependent Launch on for sm_121 — a part on which it was never qualified. The patch installs `major in (9, 10)` **behind a runtime knob**. Capabilities 9 and 10 are unaffected in both directions, so the patched file is behaviourally identical to upstream on Hopper and datacentre-Blackwell parts |
| [`patch-kpool-init.py`](patch-kpool-init.py) | 2 + 3 | correctness | The write and the read of the same array, which is why they are one file. **Item 2:** `pool_topk = torch.empty(...)` at two sites — the top-k kernels only guarantee `min(k, valid)` outputs, so a short row's tail is allocator residue that is then used as pool ids; `torch.full(..., -1)` makes those rows deterministically invalid. **Item 3:** `_expand_pools_and_append_tail_kernel` bounds the pool id from below only, so a positive garbage id expands to `pid * POOL_SIZE + o` and is emitted as a real token index; the upper bound `pid < pool_len` is free because the kernel already computed `pool_len` |
| [`patch-indexer-nospec.py`](patch-indexer-nospec.py) | 4 | jitter | `BuildPrefillChunkMetadataKernel.kernel` takes `query_slice_start` / `query_slice_stop` as plain Triton scalars under a bare `@triton.jit`. `do_not_specialize` on both collapses the kernel to one variant. **This is not a throughput change and must not be reported as one** — at most one cold compile per engine process, and we could not see it from the client at all |

Each script's docstring carries the full mechanism, the exact line numbers in the image we serve, and
the arithmetic behind the "at most three specialisation classes" claim. Read the docstring before you
read the diff.

**One call site is not in the issue.** On the EXL3 side the PDL gate also feeds
`model_executor/layers/mamba/ops/scatter_states.py`. It is present in this image and absent from the
other stack we compared against.

## What every script guarantees

The same contract as the production patch tree ([`../../patches/README.md`](../../patches/README.md)):

- `--root <vllm package dir>` is required; `--check` verifies anchors and never writes.
- Every anchor must occur **exactly once**. Anything else refuses loudly and returns non-zero. A
  half-patched stack is the failure mode that produces fluent, wrong answers.
- Idempotent — a second run reports "already applied" and returns 0.
- The patched source is `ast.parse`d before it is written; multi-file patches validate every file
  before writing any of them.
- Exit codes: `0` applied / already applied / check ok, `2` file missing, `3` anchor count mismatch,
  `4` precondition failed.

`patch-kpool-init.py` asserts **exactly two** `pool_topk` sites. The issue reports three on its
author's image and the upstream fix patches two on a third image; ours has two. If a future image
grows a third site the boot stops rather than the site being silently missed.

## The two knobs, which do different jobs

```bash
HAREM_SM12_ITEMS=pdl,kpool,nospec    # which patches are APPLIED at boot
HAREM_PDL_SM12=0                     # what the applied PDL gate RETURNS at runtime
```

- **`HAREM_SM12_ITEMS`** is read by the prelude and decides which scripts run. **Unset means no
  script runs and the tree is byte-identical to production**, which is what makes the return from an
  A/B free.
- **`HAREM_PDL_SM12`** is read by the patched gate itself. `0` or unset ⇒ PDL off on sm_12x (the
  fix). `1` ⇒ upstream `major >= 9` restored exactly. It only means anything once
  `patch-pdl-gate.py` has been applied.

**`HAREM_PDL_SM12` must be fixed for the life of the process.** `model_executor/kernels/mhc/tilelang_kernels.py`
evaluates `ENABLE_PDL` at **import** time and branches on that module constant in every mHC kernel,
while the mamba/KDA sites call `is_arch_support_pdl()` **per launch**. The two only agree if the
variable does not change under them: set it in the environment file before the engine starts and
never change it mid-run. Flipping it means a restart, not a reload.

## Wiring them into the prelude

Add to [`../../patches/tp3full-prelude.sh`](../../patches/tp3full-prelude.sh) after the existing optional
arms, so one tree serves every arm:

```bash
if [ -n "${HAREM_SM12_ITEMS:-}" ]; then
  case ",${HAREM_SM12_ITEMS}," in
    *,pdl,*)     run python3 "$TP3_DIR/patch-pdl-gate.py"       --root "$VLLM_PY" ;;
  esac
  case ",${HAREM_SM12_ITEMS}," in
    *,kpool,*)   run python3 "$TP3_DIR/patch-kpool-init.py"     --root "$VLLM_PY" ;;
  esac
  case ",${HAREM_SM12_ITEMS}," in
    *,nospec,*)  run python3 "$TP3_DIR/patch-indexer-nospec.py" --root "$VLLM_PY" ;;
  esac
fi
```

## An overlaid file cannot be patched from inside the container

`scripts/start-tp3.sh` bind-mounts `sparse_attn_indexer_kpool.py` from the GB10 top-k overlay
**read-only** over the vLLM package ([`patches/indexer-overlay/`](../../../../patches/indexer-overlay/)).
`patch-kpool-init.py` targets exactly that file, so the prelude cannot write it.

The way through is to apply the patch to a **host-side copy of the overlay directory** before the
boot and point `OVERLAY_DIR` at that copy. The prelude then reports "already applied" for the
overlaid file and touches only the image's own copies. This is an exception to the general pattern of
gating patches in the prelude, and it applies to **any** future patch that lands on an overlaid file.

## The sidecar warning: read this before you copy anything

`tp3full/harem_fastload_id.py` hashes **every file matching `patch-*.py`** in the tree into the
fast-load sidecar's identity:

```python
for p in sorted(glob.glob(os.path.join(tp3_dir, "patch-*.py"))) + [...]:
    ident["patches"][os.path.relpath(p, tp3_dir)] = sha_file(p)
```

So **copying these files into the production tree invalidates the sidecar before any of them runs**,
and `preflight-fastload.py` refuses the next LOAD boot on every node
([docs/08](../../../../docs/08-fast-boot.md) §4). The consequences, in order:

1. **Measure the baseline first**, with the production tree untouched.
2. Copy in **every** file you intend to use in the campaign at once, then take **one**
   `FASTLOAD_MODE=dump` boot. None of these patches changes a weight, so that single sidecar is valid
   for every patched arm **and** for the return to production. Adding a file later costs a second
   dump boot.
3. After that, arms are selected by environment knobs only.
4. Returning to production needs **no** third dump: with `HAREM_SM12_ITEMS` unset no script runs, so
   the tree hashes the same as production.

This is why we ran the campaign in a **separate** directory and a separate image rather than in the
production tree at all.

## What is not here

**The diagnostic instrument.** A fourth script existed for this campaign — an env-gated hook in
`sparse_attn_indexer_kpool` that reads back `topk_indices_buffer` to produce the per-row selected-key
count and the adjacent-row overlap for
[issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5). It is **not shipped**, for two reasons: it
stalls the step it fires in, so no timing from that step is valid; and its derived
`context`/`expected`/`deficit` columns mix pool-granular and token-granular units and are wrong. The
two distributions it produced are counted straight off the selection buffer, are unaffected, and are
published in
[`results/kernels/sm12-stack-patches-ab.md`](../../../../results/kernels/sm12-stack-patches-ab.md) §8.

**Two items that were recorded and not patched**, both silent-wrong-answer class and both currently
inert here: the Mamba block table's `dcp_size` multiply (we run DCP=1, so it multiplies by one), and
the FLA `tensor_cache` identity memo (our call path hands it a fresh slice each time, so it misses
rather than going stale). Each is re-examined the day its precondition changes `[not tested]`.

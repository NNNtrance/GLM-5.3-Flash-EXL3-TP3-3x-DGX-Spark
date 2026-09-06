# `patches-optional/sm12` — the one sm_12x patch that is **not** in the recipe

**This directory used to hold three files. Two of them ship now.** On 6 September 2026 the sm_12x
correctness set — [`patch-pdl-gate.py`](../../patches/patch-pdl-gate.py) (item 1) and
[`patch-kpool-init.py`](../../patches/patch-kpool-init.py) (items 2 + 3) — was adopted into
**production configuration 11** and moved into [`../../patches/`](../../patches/), the tree the
prelude actually applies. What is left here is **item 4**, and it is left here on purpose.

**Applies to: TP=3 as published here; the patch itself is rank-agnostic.**

**Credit.** All four findings are `tpurtell/glm-5.3-flash-ext3-2x-rtx`'s (Apache-2.0), raised as
[Zeuss5/cuda-exl3 issue #6](https://github.com/Zeuss5/cuda-exl3/issues/6). These scripts are
re-implementations against our own anchors — our image is not theirs — and the Apache-2.0 attribution
travels with them. See [CREDITS](../../../../CREDITS.md).

---

## What is here

| File | Item | Class | What it does |
|---|---|---|---|
| [`patch-indexer-nospec.py`](patch-indexer-nospec.py) | 4 | jitter | `BuildPrefillChunkMetadataKernel.kernel` takes `query_slice_start` / `query_slice_stop` as plain Triton scalars under a bare `@triton.jit`. `do_not_specialize` on both collapses the kernel to one variant. **This is not a throughput change and must not be reported as one** |

**Why it is not adopted.** Everything about item 4 measured at zero, in both directions. triton 3.7.1
gives an int argument **three** specialisation classes (`("constexpr",1)` at value 1, `("i32","D")` at
value % 16 == 0, `("i32","")` otherwise), not one variant per chunk offset as the issue implies, and
the tree's own warmup already covers two of the three for `query_slice_start` and all three for
`query_slice_stop`. So at most **one** cold compile per engine process is at stake — and across cold,
freshly booted processes with the fix on and off, the first of three ~70K-token prefills was never
more than 2 % of a 38-second prefill slower than the later ones, and was often the fastest. It most
likely lands in the startup warmup/profile run and never reaches serving
`[measured-here]`.

A patch whose benefit cannot be measured is not carried into a configuration that has to pass a
three-node reboot test. It is here, working, for anyone who wants it. Everything measured is in
[`results/kernels/sm12-stack-patches-ab.md`](../../../../results/kernels/sm12-stack-patches-ab.md);
the decision is in [docs/11](../../../../docs/11-open-issues.md) §2.27.

## What moved, and where the knobs live now

| File | Item | Now at | Gated by |
|---|---|---|---|
| `patch-pdl-gate.py` | 1 | [`../../patches/`](../../patches/) | `HAREM_SM12_ITEMS=…,pdl` at boot; `HAREM_PDL_SM12=0\|1` at run time |
| `patch-kpool-init.py` | 2 + 3 | [`../../patches/`](../../patches/) | `HAREM_SM12_ITEMS=…,kpool` |
| `patch-indexer-nospec.py` | 4 | **here** | nothing — the shipped prelude does not know the word `nospec` |

The prelude's item list is **fail-closed**: an unrecognised entry stops the rank rather than being
skipped, so `HAREM_SM12_ITEMS=pdl,kpool,nospec` against the shipped tree does not silently ignore
`nospec` — it refuses to boot. To run item 4, copy the file into your `TP3_DIR` and add its `case`
arm to [`../../patches/tp3full-prelude.sh`](../../patches/tp3full-prelude.sh) beside the two that are
there, and read the sidecar warning below first.

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

## The sidecar warning: read this before you copy anything

`patches/harem_fastload_id.py` hashes **every file matching `patch-*.py`** in the tree into the
fast-load sidecar's identity:

```python
for p in sorted(glob.glob(os.path.join(tp3_dir, "patch-*.py"))) + [...]:
    ident["patches"][os.path.relpath(p, tp3_dir)] = sha_file(p)
```

So **copying a file into the patch tree invalidates the sidecar before it runs**, and
`preflight-fastload.py` refuses the next `FASTLOAD_MODE=load` boot on every node
([docs/08](../../../../docs/08-fast-boot.md) §4). `tp3-prelude.sh` is hashed too, so editing the
prelude has the same effect. The consequences, in order:

1. **Measure the baseline first**, with the production tree untouched.
2. Copy in **every** file you intend to use at once, then take **one** `FASTLOAD_MODE=dump` boot
   (590 s on our cluster, and it writes ~53 GB per node — check `df` first). None of these patches
   changes a weight, so that single sidecar is valid for every arm.
3. After that, arms are selected by environment knobs only.
4. `gpu-memory-utilization` is **not** in the identity, so a memory rung costs no dump boot — which
   is why production 11 changed the rung and the patch set in the same one.

## What is not here

**The diagnostic instrument.** A fourth script existed for the A/B campaign — an env-gated hook in
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

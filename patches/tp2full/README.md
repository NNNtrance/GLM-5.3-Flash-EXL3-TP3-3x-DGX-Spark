# patches/tp2full — the two-node patch tree (TP=2 production candidate)

**Applies to: TP=2 only.** This is the tree the two-node production candidate of
[docs/15](../../docs/15-tp2-track.md) §5 runs. It is [`patches/tp3full/`](../tp3full/) with every
file that exists only to serve a pad removed, plus the two-node full-scope loader patch. Thirteen
files against twenty-two.

**Why a tree of its own and not a flag.** The directory's **file list and content are the fast-load
manifest identity** ([docs/08](../../docs/08-fast-boot.md) §4): every `patch-*.py` in it and the full
text of the prelude are hashed into the sidecar, and adding one file to a tree — even a file that is
never called — refuses the next boot on every node. That has happened to us twice, and **once it was
a TP=2 patch dropped into the TP=3 tree** that did it. Finish this tree before the dump boot and add
nothing between dump and load.

The cost is the same one the three-node trees pay and it is written down rather than hidden: **the
trees have to be kept in step by hand.** Every shared file here is byte-identical with its
`patches/tp3full/` copy today, and nothing enforces that tomorrow.

## What is in here

| File | Same as `patches/tp3full/`? | What it does at two ranks |
|---|---|---|
| `tp2full-prelude.sh` | **new** | the in-container prelude: applies the patches below, then `exec vllm serve`. Install it under **two names** — see below |
| `patch-fullscope-tp2.py` | **new** (it is `patches/tp2/`'s) | S1 packed mapping, S2 stop hard-wiring MLA+KDA to bf16, S3 KDA refactorisation. Gated on `HAREM_EXL3_FULLSCOPE`. **The recommended candidate runs this** |
| `patch-kvdiag-tp3.py` | byte-identical | logging only: the per-group decomposition of the KV pool arithmetic. Every arm |
| `patch-swblock-tp3.py` | byte-identical | the draft KV page, 16 → 256 tokens. **Mandatory in practice** at two ranks ([docs/15](../../docs/15-tp2-track.md) §4) |
| `patch-draftkv-tp3.py` | byte-identical | the drafter's own cache at fp8: **+15.1 % of pool** at two ranks |
| `patch-tilelang-failloud-tp3.py` | byte-identical | a silent `contextlib.suppress` becomes a named error. Gated |
| `patch-epfilter-tp3.py` | byte-identical | EP weight filter for `.trellis`. **Inert here** — it needs `--enable-ep-weight-filter`, which needs EP, and EP is off. Kept anyway, so that trying EP at two ranks needs no tree change and therefore no new dump boot |
| `patch-fastload-tp3.py` | byte-identical | installs the sidecar hook into vLLM's loader. Inert unless `HAREM_FASTLOAD_MODE` is set |
| `harem_fastload.py` | byte-identical | the dump/restore engine |
| `harem_fastload_id.py` | byte-identical | the sidecar identity, shared by the engine and the preflight so "stale" has one definition |
| `preflight-fastload.py` | byte-identical | refuses a stale sidecar in the prelude, in a second, before vLLM starts |
| `preflight-tp3.py` | byte-identical | the shape preflight; `tp`-parameterised. Run it `--tp 2 --ep 0` and it prints `ep=OFF (experts tensor-sliced)` |
| `flashinfer-warmup.py` | byte-identical | imports `flashinfer.comm` once, CPU-side, ~2 s |

## Install the prelude under two names

```
install -m 0755 patches/tp2full/tp2full-prelude.sh "$TREE/tp2-prelude.sh"
ln -f "$TREE/tp2-prelude.sh" "$TREE/tp3-prelude.sh"
```

Not a typo, and not a second copy. `scripts/start-tp2full.sh` mounts `tp2-prelude.sh` at `/start.sh`,
while `harem_fastload_id.file_identity()` hashes the prelude under the name **`tp3-prelude.sh`** —
so without the link the prelude's text silently falls out of the sidecar manifest and a changed
prelude would no longer invalidate a stale sidecar. A **hard** link is what keeps the two names on
one inode so they cannot drift apart. `start-tp2full.sh` refuses to launch if either name is missing.

The same launcher also sets `TP3_DIR` to the *same* directory as `TP2_DIR`, because
`preflight-fastload.py` and `harem_fastload.py` both read `TP3_DIR` to find the patch set they must
hash.

## What is deliberately **not** here

All of it exists to serve a pad, and at two ranks there is no pad: all five awkward shapes divide by
two and leave whole 128-column Hadamard blocks ([docs/15](../../docs/15-tp2-track.md) §1.1).

| Not shipped | Why |
|---|---|
| `patch-vllm-tp3.py` | the zero-extend helper for a shard that runs past the stored dim. `lcm(128, 2) = 128`, so the branch can never fire. Harmless to keep if you prefer one tree for both rank counts — but no two-node arm of ours has run it, and we do not ship text we have not measured |
| `patch-exl3-ep.py` + `overlay/cuda_exl3/` | the `cuda-exl3` expert-parallel kernel fixes. EP is off at two ranks |
| `patch-dflash-tp3.py` | makes the DFlash2 head check pad-aware over the 32/8 → 36/9 drafter pad. 32/8 divides by two |
| `patch-fullscope-tp3.py` | its A1–A8 are `patch-fullscope-tp2.py`'s text exactly; A9 (split a fused checkpoint tensor by checkpoint widths) is a no-op at TP≤2 and A10 (post-load pad audit) has nothing to audit |
| `check-padload-tp3.py` | gates a `cuda-exl3` capability only the padded-load path needs |
| `pad-tp3.py`, `pad-tp3full.py`, `mk-env-tp3full.sh` | build padded sidecars and the env that points at them. Nothing to pad |
| `patch-zerokv-tp3.py` | an optional arm, never in a production environment file at either rank count |

## Order the prelude applies them

`patch-kvdiag` → `patch-swblock` → `patch-epfilter` → `patch-fastload` → (`patch-draftkv` if
`HAREM_DRAFT_KV_DTYPE`) → (`patch-tilelang-failloud` if `HAREM_TILELANG_FAILLOUD=1`) →
(`patch-fullscope-tp2` if `HAREM_EXL3_FULLSCOPE=1`) → `flashinfer-warmup` → `preflight-tp3` →
(`preflight-fastload` if `HAREM_FASTLOAD_MODE`) → `exec vllm serve`.

`TP2_STRICT=0` turns a failed patch into a warning. Do not use it to get past a broken anchor: a
half-patched stack is exactly the failure mode that serves fluent, wrong answers.

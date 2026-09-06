# The draft KV page at two ranks — control against fix

The one item in [docs/15](../../docs/15-tp2-track.md) §3.3 that carried a large expected effect and
no measurement: `HAREM_SW_BLOCK_SIZE=256` at TP=2. Measured 6 September 2026, two arms back to back,
one boot each `[measured-here]`.

**Settings, identical in both arms except the one token.** Two DGX Spark (GB10) nodes, TP=2,
**expert parallelism off**, image `exl3-zeus:62f53e6`, checkpoint
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` revision `b20c49ba` (routed experts only), KV `fp8`,
DFlash2 draft k=7, `--attention-backend CUSTOM`, `--block-size 256`, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `--max-model-len 1000000`, `gpu-memory-utilization 0.85`,
`HAREM_DISABLE_PERSISTENT_TOPK=1`, `NCCL_MAX_NCHANNELS=8`, mesh plugin with both cables per peer,
`CUDA_EXL3_TUNE_CACHE` warm, **no** fast-load sidecar, **no** fp8 draft cache, temperature 0,
reasoning effort **low**, `max_tokens` 256, prompts `scripts/hizset-v2.jsonl`, medians of three
sweep rounds. Both arms booted through the launcher's host-side settle gate
(`MemAvailable ≥ 112 GiB` on both nodes; they started at 116.9 and 117.1 GiB), so the pool figures
satisfy the acceptance rule in [docs/07](../../docs/07-kv-and-draft-page.md) §1.1.

The only difference between the arms is one token in `EXTRA_ENV`: `HAREM_SW_BLOCK_SIZE=256`. The
launcher's `DRY_RUN` output differs by exactly `-e HAREM_SW_BLOCK_SIZE=256` on both nodes. The patch
file is [`patches/tp3/patch-swblock-tp3.py`](../../patches/tp3/patch-swblock-tp3.py) unchanged — it
is `tp`-agnostic and gated on its own environment variable, so the control arm ran the same image
with the knob unset, which is upstream behaviour. `patch-kvdiag-tp3.py` was applied in **both** arms
so the pool arithmetic is printed either way; it is logging only.

---

## 1. The pool, and why it moved

Straight from each boot's `HAREM-TP3 KV pool breakdown` line:

| group | page, control | blocks/request, control | page, fix | blocks/request, fix |
|---|---|---|---|---|
| `MLAAttentionSpec`, 22 layers | 152,064 B | 218 | 152,064 B | 218 |
| `KpoolTailSpec`, 11 layers | 152,064 B | 1 | 152,064 B | 1 |
| `MambaSpec` × 4 | 2,359,296 B | 9 + 9 + 9 + 9 | 2,359,296 B | 9 + 9 + 9 + 9 |
| **`SlidingWindowSpec` — the drafter, 5 layers** | **32,768 B** (16 tokens) | **385** | **524,288 B** (256 tokens) | **25** |
| **blocks per request** | | **640** | | **280** |
| `num_blocks` | | 385 | | 365 |
| **`GPU KV cache size`, `max_model_len` 1,000,000** | | **601,562** (0.60x) | | **1,303,571** (1.30x) |

**The drafter takes 60.2 % of the blocks-per-request divisor at two ranks**, against 53 % at three.
The defect is worse here, because the target's own share is smaller: the platform raises the
attention block to **4,608** tokens at TP=2 (3,328 at TP=3), which cuts the MLA group's
blocks-per-request from 301 to 218 while the drafter's 385 does not move.

## 2. Everything measured, side by side

| metric | **control — draft page 16** | **fix — draft page 256** | delta |
|---|---|---|---|
| **KV pool @ 1M context** | **601,562** | **1,303,571** | **+116.7 %** |
| the same, normalised to equal binding-rank KV memory | 601,562 | 1,260,714 | **+109.6 %** |
| `num_blocks` | 385 | 365 | −5.2 % |
| blocks per request | 640 | 280 | −56.3 % |
| **prefill-fresh, 3 unseen ~8.3K prompts** | **never scheduled** | **1,478 tok/s** (1,449 / 1,478 / 1,480) | path opens |
| **prefill, 7,382 tokens, uncached** | **never scheduled** | **1,267 tok/s** | path opens |
| **largest prompt actually served** | **5,386 tokens** | **8,268 tokens**, every size served | path opens |
| C1 aggregate (tok/s) | 47.41 | 47.30 | −0.2 %, equal |
| C1 per stream (tok/s) | 55.73 | 51.34 | −7.9 %, see §4 |
| C2 aggregate | 68.27 | 68.72 | +0.7 %, equal |
| C4 aggregate | 91.65 | 93.74 | +2.3 % |
| C6 aggregate | 113.22 | 117.71 | +4.0 % |
| **C8 aggregate** | **127.54** | **135.59** | **+6.3 %** |
| C8 per stream | 20.39 | 20.73 | +1.7 % |
| **TTFT median @ C1** | 0.621 s | **0.478 s** | **−23.0 %** |
| **TTFT median @ C8** | 1.703 s | **1.244 s** | **−27.0 %** |
| draft acceptance @ C1 | 64.48 % | 62.56 % | −1.9 points |
| accepted tokens per step @ C1 | 5.51 | 5.38 | −2.4 % |
| cold first request | TTFT 0.89 s, 36.5 tok/s, acceptance 40.0 % | TTFT 0.76 s, 39.7 tok/s, acceptance 43.9 % | faster |
| correctness probe / code exam, cold | 10/10 · 12/12 · 0 empty | 10/10 · 12/12 · 0 empty | equal |
| correctness probe / code exam, warm | 10/10 · 12/12 · 0 empty | 10/10 · 12/12 · 0 empty | equal |
| boot, cold, no fast-load sidecar | 396 s | 375 s | −5.3 % |
| binding rank `Available KV cache memory` | 9.97 GiB (worker-1) | 10.31 GiB (head) | +3.4 % |
| per-rank `consumed (weights + non-torch)` | 89.06 / 91.00 GiB | 90.65 / 89.37 GiB | see §5 |
| free host RAM after the run, head / worker-1 | 8.6 / 9.8 GiB | 7.3 / 8.7 GiB | −1.2 GiB |
| swap in use | 0.03 / 0.02 GiB | 0.03 / 0.02 GiB | flat |

Per-round figures, so the spread is visible rather than implied:

| arm | C | round 1 | round 2 | round 3 | median |
|---|---|---|---|---|---|
| control | 1 | 47.41 | 45.90 | 47.85 | 47.41 |
| control | 2 | 66.60 | 68.27 | 68.42 | 68.27 |
| control | 4 | 92.05 | 91.65 | 89.70 | 91.65 |
| control | 6 | 113.22 | 115.33 | 107.90 | 113.22 |
| control | 8 | 126.61 | 129.76 | 127.54 | 127.54 |
| fix | 1 | 47.56 | 47.08 | 47.30 | 47.30 |
| fix | 2 | 69.23 | 67.82 | 68.72 | 68.72 |
| fix | 4 | 95.88 | 93.74 | 92.43 | 93.74 |
| fix | 6 | 117.71 | 121.87 | 113.47 | 117.71 |
| fix | 8 | 137.03 | 135.26 | 135.59 | 135.59 |

## 3. The long-prompt cliff, measured on both arms

The same probe on both arms — one request at a time, prompt length walked up, 75-second budget:

| prompt tokens | control | fix |
|---|---|---|
| 885–913 | served, 1.3–1.5 s | served, 1.3 s |
| 1,786–1,838 | served, 2.1 s | served, 2.1 s |
| 2,640–2,759 | served, 2.6 s | served, 2.4 s |
| 3,586–3,589 | served, 3.2 s | served, 3.0 s |
| 4,444–4,487 | served, 3.8 s | served, 3.8 s |
| 5,386–5,494 | served, 4.1 s | served, 4.4 s |
| **6,253** | **never scheduled** | served, 4.9 s |
| 7,329 | — | served, 5.5 s |
| 8,268 | — | served, 6.3 s |

"Never scheduled" is literal and it is what the engine reports: `Running: 0 reqs, Waiting: 1 reqs,
GPU KV cache usage: 0.0 %`, indefinitely. It is the same state
[docs/13](../../docs/13-full-scope-checkpoint.md) §6 recorded for the full-scope TP=2 arm at
~2,800 tokens, and the arithmetic is the same: block ids are global to one pool, one request needs
640 blocks of the pool's 385, so it can never be admitted. After the fix a request needs 280 of 365.

## 4. What it cost

- **Per-block memory rises about 9.1 %.** The drafter's per-block cost goes 163,840 B →
  2,621,440 B, and `num_blocks` would fall 385 → 353 at equal memory (the measured 365 includes the
  3.4 % more KV memory the fix arm's binding rank happened to get). At three ranks the same price
  was +9.2 %.
- **The draft group's prefix-cache matching unit coarsens from 16 to 256 tokens.** No measurable
  cost on this stack: with a 4,608-token attention block, prefix-cache hit rate is 0 % in both arms.
- **Draft acceptance falls 1.9 points**, 64.48 % → 62.56 %, and accepted tokens per step 5.51 →
  5.38. Both sit inside the 60–65 % band this stack shows from boot to boot, and neither was
  confirmed on a second boot.
- **C1 per-stream decode reads 7.9 % lower** — the one number that moved the wrong way. The fix
  arm's own three rounds span 50.60–55.29 (9.3 %), the C1 **aggregate** is equal (47.41 ↔ 47.30),
  and TTFT is 23 % better, so this single boot does not establish a loss. Stated plainly: no C1
  gain, and no proven C1 loss either.
- **Free host RAM falls about 1.2 GiB per node.** Swap stayed flat at 0.02–0.03 GiB in both arms.

## 5. Two honesty notes

**The control's own pool is smaller than the one this repository published on 5 September.**
665,625 then, 601,562 now, same env file untouched since, −9.6 %. The 5 September TP=2 harness had
**no settle gate** — it dropped caches and booted immediately. This one waits for `MemAvailable` to
come back over 112 GiB. That is exactly the instrument bias
[docs/07](../../docs/07-kv-and-draft-page.md) §1.1 describes, in the direction it predicts: a node
that starts dirty awards itself memory it does not have. The A/B is unaffected — both arms here
booted through the same gate — but the published two-node control figure was measured on an
unpinned baseline and should be read as such.

That difference is not cosmetic. 5 September's 426 blocks were just above the long-prompt cliff and
served a 7,382-token prompt at 1,135 tok/s; today's cleaner 385 blocks are just below it and the
same prompt is never scheduled. **Without the page fix, two ranks sit on the edge of that cliff and
the host's state at boot decides which side.**

**The two ranks' `consumed (weights + non-torch)` figures differ by 1.3–1.9 GiB and the heavier rank
swaps between arms** (worker-1 in the control, head in the fix). At two ranks that is 13–19 % of a
rank's KV allowance, so it is worth naming; it does not threaten a result of this size, and it is
the two-node face of the rank asymmetry in [docs/11](../../docs/11-open-issues.md) §2.

## 6. Not done

One boot per arm; `HAREM_DRAFT_KV_DTYPE=fp8`, the memory ladder, expert parallelism on, and the
fast-load sidecar are all still `[not tested]` at two ranks, and the page fix was not tried on top
of the full-scope checkpoint at TP=2, whose pool is clamped by a different mechanism
([docs/15](../../docs/15-tp2-track.md) §3.2).

# The sparse-indexer K-gather workspace: 4.92 GiB reserved for a 160K-context model, on a 1M-context one

**A single buffer, sized by a constant that was never meant for this `max_model_len`, is the largest
item in the "non-torch" residual — and giving it its real bound is worth +10.25 % of the KV pool at
an unchanged memory fraction, with no measured speed cost.** Measured on 6 September 2026 as a
three-arm A/B in one session `[measured-here]`. **It shipped the same day as production
configuration 12**, together with the `gpu-memory-utilization` 0.88 rung; the promotion boot's own
numbers are §7.1.

**Applies to: the three-node TP=3 track as measured here.** The patch is rank-agnostic and the
arithmetic is not; at two ranks the same buffer is the same size, because it is sized from
`max_model_len` and nothing else `[not tested]`.

Settings, all three arms: image `exl3-zeus:754421f` (`cuda-exl3` at `754421f`), vLLM
`0.1.dev20051+g487ecf187`, TP=3 + expert parallel, full-scope EXL3 weights
(`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw), `kv-cache-dtype fp8` and an fp8 draft cache, DFlash2
draft at k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`,
`--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, `max_model_len 1,000,000`,
**`gpu-memory-utilization` 0.87 — production configuration 11, unchanged in every arm** —
`HAREM_SM12_ITEMS=pdl,kpool`, temperature 0, reasoning effort **low**. Speed is the median of three
sweep rounds on the twelve realistic prompts of
[`scripts/hizset-v2.jsonl`](../../scripts/hizset-v2.jsonl). Gates are the correctness probe (10) and
the code exam (12), cold and again after the whole battery. The two A/B arms are **eager boots** —
fast-load off — and the reason is §3.1.

**Three lines, before the tables.**

- **What it is.** vLLM sizes the sparse indexer's K-gather workspace as `40 × max_model_len`
  **entries**. The 40 is an upstream constant chosen against DeepSeek-V3.2's 163,840-token context,
  where it comes to 825 MB. At `max_model_len` 1,000,000 the same constant asks for 40,000,000
  entries — **4.92 GiB** — reserved during the profile run and locked for the life of the engine.
- **Why it lands on the KV pool.** It is not weights and it is not KV, so the profiler charges all of
  it to the residual it subtracts from the budget before the pool is sized. It is the largest single
  item in the 7.28 GiB "non-torch" line of [docs/17](../../docs/17-memory-ledger.md).
- **What the buffer actually needs.** One indexer chunk's **compressed** context. At our settings the
  largest total the scheduler can present at once is 2,000,016 entries = **251.8 MB**. Bounded to
  512 MB — 2.03× that ceiling — the pool goes **6,289,256 → 6,933,884 tokens (+10.25 %)**, every gate
  holds, and every speed level stays inside its band.

---

## 1. The mechanism

### 1.1 One function, two consumers

`get_max_prefill_buffer_size()` in `vllm/v1/attention/backends/mla/indexer.py` returns an **entry
count**, and upstream's own comment says where the 40 came from:

```python
def get_max_prefill_buffer_size(vllm_config):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
```

Two places consume that number, and they have to agree or the chunker can emit a chunk larger than
the buffer it will be written into:

| Consumer | Where | What it does with the value |
|---|---|---|
| The chunker's N constraint | `mla/indexer.py`, into `split_indexer_prefill_chunks` | splits a prefill so that `sum(seq_len // compress_ratio) <= workspace_size` |
| The buffer's own shape | `vllm/models/glm5next/nvidia/attention.py`, into `sparse_attn_indexer_kpool.py` | reserved in the profile run, requested at run time through `get_simultaneous` |

So the patch changes the **function**, not either call site. Changing one call site is how the two
drift apart.

**And there is an inconsistency in upstream itself.** The DeepSeek-V4 path divides the same function's
result by `compress_ratio` at its call site (`vllm/models/deepseek_v4/attention.py`); the glm5next
path does not — while the chunker on both paths works in **compressed** lengths
(`compressed_seq_lens_cpu = seq_lens_cpu // compress_ratio`), and `compress_ratio == index_kpool`,
which is **4** for this model. The buffer was therefore sized at four times the resolution of the unit
that fills it, on top of a constant already 6.1× larger than the model it was tuned for. That is the
[HELP-WANTED](../../HELP-WANTED.md) §9 item.

### 1.2 What the profiler counts, and why this is invisible

The buffer is reserved inside the profile run and locked by `lock_workspace()` when the run ends, so
it is live for the rest of the engine's life. It appears in **none** of the lines an operator reads:

| Log line | What it measures | Is the workspace in it |
|---|---|---|
| `Model loading took 51.62 GiB` | torch **allocated** delta over the weight load | no — it is not weights |
| `consumed memory (weights + non-torch) 59.61 GiB` | the drop in `MemAvailable` from baseline to after the profile run | **yes, in full** |
| `Available KV cache memory` | `requested − non_kv_cache_memory` | it is what was subtracted |
| `GPU KV cache size: N tokens` | the pool | smaller by exactly this much |

`MemorySnapshot` on this part reads `psutil.virtual_memory().available` rather than `cudaMemGetInfo`,
because `is_integrated_gpu` is true on GB10 — the same fact that makes the launcher's settle gate
necessary ([docs/17](../../docs/17-memory-ledger.md) §2). So "non-torch" is a **residual between two
rulers**, not a category anything reports, and a 4.92 GiB live allocation can sit inside it without a
name. It did.

### 1.3 The bound, read off the consumers

Nothing here is a guess; each row is read from the code that uses the buffer.

| Quantity | Formula | Here |
|---|---|---:|
| Bytes per entry | `index_head_dim` fp8 bytes + 4 scale bytes | **132 B** |
| **Correctness floor** — one request's compressed context | `ceil((max_model_len + num_spec + 1) / index_kpool)` | **250,002 entries = 31.5 MB** |
| **Exact ceiling** — the most the scheduler can present at once | `max_num_seqs × floor` | **2,000,016 entries = 251.8 MB** |
| Chosen, `HAREM_INDEXER_WS_MODE=bound` | `min(upstream, max(2 × ceiling, 512 MB, floor))` | **4,067,203 entries = 512.0 MB** |
| Upstream | `40 × max_model_len` | 40,000,000 entries = **4.918 GiB** |

`index_head_dim` 128, `index_kpool` 4 and `use_fp4_indexer_cache` false (GB10 is sm_121, and the
fp4 path's Blackwell test is a family-100 check) come from the model's `config.json` and the engine's
own configuration; `num_spec` 7 is the drafter's `k`, `max_num_seqs` 8 is ours.

**The floor is the one number that is a correctness limit rather than a budget.** Shrinking the buffer
makes the splitter emit **more** chunks — not wrong answers — right up to the point where a **single**
request no longer fits, because the splitter's `end == start` branch deliberately emits an oversized
chunk when it cannot split further. Below that line the gather is handed a `cu_seq_lens` whose last
entry runs past the buffer. The chosen size is 16.3× the floor and 2.03× the ceiling.

**What the reservation logs, against the arithmetic.** 40,000,000 × 132 B is 4.918 GiB; the
`WorkspaceManager` line reads 5,036.40 MB, which is that plus 256-byte alignment and the 1 MiB
radix-top-k buffer requested in the same call. Bounded, the same call reads 513.00 MB for 512.0 MB of
entries. The difference is **4.4174 GiB**.

---

## 2. What a too-small buffer actually does, and the layers that catch it

> **Corrected 6 September 2026** `[retracted]`. This section originally claimed that the gather kernel
> writes `cu_seq_lens[-1]` rows into a shorter buffer — an out-of-bounds device write — and that
> upstream's locked-workspace `AssertionError` (**L0**) was the load-bearing layer. **Both were
> wrong.** The correction came from [@drakosha](https://github.com/drakosha)
> [in the #55221 thread](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5561194190):
> both indexers ask the `WorkspaceManager` for a **static** size and then slice per chunk, so the
> request never grows and the assertion we cited cannot fire. Checking that against the code we
> actually run confirmed their reading of the request path, and moved one detail of the symptom in the
> other direction — the gather kernel does not write out of bounds either. What survives is below.
> [`docs/11`](../../docs/11-open-issues.md) §1.13 carries the retraction.

This is the part that decides whether the item is worth touching at all, because the failure it would
cause is the bad kind: **silent**. `k_quant_full[: chunk.total_seq_lens]` is a **Python slice**, and a
slice longer than the tensor **clamps** rather than raising. Read off the code we run, here is what
follows the clamp:

- **The gather does not overrun.** `cp_gather_indexer_k_quant_cache`
  (`csrc/libtorch_stable/cache_kernels.cu`) takes its extent from the **destination**:
  `int num_tokens = dst_k.size(0);`, a grid of `ceil(num_tokens / BLOCK_Y_SIZE)`, and a per-thread
  guard `if (head_idx >= head_dim || token_idx >= num_tokens || batch < 0) return;` carrying
  upstream's own comment, "num_tokens may be an allocation upper bound when Python avoids a D2H
  sync". `cu_seq_lens` selects *which* in-range rows are valid; it does not set how many rows are
  written. Verified at our image's base commit `487ecf187`, at `808f8cd3ac`, and at every commit
  sampled back to `22a58640b4` (29 May 2026). The ROCm Triton reference
  (`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`) is destination-bounded the same way,
  `num_tokens = k_fp8.size(0)`.
- **So the rows past the clamp are simply never gathered** — and the chunk's `cu_seqlen_ks` /
  `cu_seqlen_ke`, built from the *unclamped* compressed lengths, are then handed to
  `fp8_fp4_mqa_logits` alongside the shortened `k_quant` / `k_scale`. The consumer is asked to read K
  rows the buffer does not have. That is an out-of-bounds **read** on the DeepGEMM side rather than a
  write, and either way the top-k selection it feeds is silently wrong. Whether DeepGEMM bounds-checks
  is `[not tested]` here: it is a JIT-compiled dependency and this verification was CPU-only.
- **The scale buffer does share the allocation**, which is the part of the hazard that survives
  unchanged. `WorkspaceManager.get_simultaneous` (`vllm/v1/worker/workspace.py`) packs every requested
  tensor into **one** `uint8` buffer at 256-byte-aligned offsets, so `k_scale_full` begins immediately
  after `k_quant_full` — anything that did run past the values view would land in the scales.

The failure mode is therefore **a silent clamp that produces wrong answers**, not a crash and not
corruption written by the gather. That is harder to detect, not easier, which is why the layers exist:

| Layer | Where | When it fires | Whose code |
|---|---|---|---|
| **L0** | `v1/worker/workspace.py` | a locked workspace asked to **grow** raises `AssertionError`, naming the caller's file and line — but the indexer's request is static, so **this layer does not cover the case at hand** | **upstream, unconditional** |
| **L1** | `get_max_prefill_buffer_size` | **at startup**: a chosen size below the one-request floor raises `RuntimeError` before a byte is committed | the patch |
| **L2** | `split_indexer_prefill_chunks` | metadata build: the `end == start` branch about to emit an oversized chunk raises, before any kernel launch | the patch, armed by the knob |
| **L3** | `sparse_attn_indexer_kpool.py`, immediately above the slice | at the clamp itself | the patch, armed by the knob |

**L0 is not the load-bearing one, and believing it was is the error.** The indexer requests a
**static** size — `_gather_workspace_shapes(total_seq_lens, …)`, where `total_seq_lens` is the op
argument bound once at construction to `get_max_prefill_buffer_size(vllm_config)` — and only then
slices per chunk (`sparse_attn_indexer_kpool.py`: the `get_simultaneous` call and the
`k_quant_full[: chunk.total_seq_lens]` below it; `sparse_attn_indexer.py` does the same with
`max_local_total_seq_lens`). `_ensure_workspace_size` therefore sees the same `required_bytes` on
every step, and the locked-workspace assertion cannot fire from this path at all. It would take a
*different* consumer of the shared workspace appearing after `lock_workspace()`, or `max_model_len` or
the 40 changing after it — neither of which happens. Our own evidence said so and we did not read it:
**exactly one resize event per rank** for the life of the engine.

**L1 is the load-bearing one, and it is ours.** The splitter admits requests only while
`new_n <= workspace_size` and is handed the *same* bounded value the buffer is sized from, so the only
chunk that can exceed the buffer is the one the `end == start` branch emits deliberately — a single
request the packer could not split. L1 refuses to start when the chosen size is below one request's
compressed context, which is precisely what stops that branch from producing an over-size chunk. L2
and L3 are belt and braces at the two places it would otherwise surface.

**The arithmetic, and why the bound is safe in this configuration.** Production configuration 12:
`max_model_len` 1,000,000, `max_num_seqs` 8, `max_num_batched_tokens` 2048, `num_spec` 7,
`compress_ratio` 4, 132 B per entry, chosen bound **4,067,203 entries = 512.0 MB**.

| | Entries | MB | Against the bound |
|---|---:|---:|---|
| One prefill step's compressed total, exact ceiling: `max_num_seqs × ceil((max_model_len + num_spec + 1) / compress_ratio)` = 8 × 250,002 | **2,000,016** | 251.8 | **2.03×**, 2,067,187 entries spare |
| One request's compressed context, the correctness floor: `ceil(1,000,008 / 4)` | **250,002** | 31.5 | **16.27×** |
| Chosen bound | 4,067,203 | 512.0 | — |

`max_num_batched_tokens` bounds the **query** dimension and the logits budget, not this one, so it
does not enter the ceiling — it only makes the aggregate row rarer in practice. And the aggregate row
cannot overflow in any case, because the splitter's admission test uses the same number and would
split instead. An over-size chunk therefore needs a single request whose *own* compressed context
exceeds 4,067,203 entries — an uncompressed context above **16,268,812 tokens** against a
`max_model_len` of 1,000,000. Unreachable by construction rather than by margin, which is exactly what
L1 keeps true.

**L2 and L3 are armed only when the sizing knob is set.** With the knob unset the patched image
behaves as upstream, which is what makes a clean control arm possible on the same tree — and that
claim is measured below, not asserted.

**The dry run, before any engine time** `[measured-here]`: against the installed tree in a throwaway
CPU-only container, `--check` matched 3 + 2 anchors exactly once each; applying, re-running
(idempotent) and `py_compile` all passed; and a unit test covering seven blocks passed, including the
one that matters — **the chunk list is byte-identical to upstream's across six geometries** (1×1M,
8×1M, 8×128K, decode-like, mixed, 64×16K), because the N constraint never bound at 4.92 GiB and does
not bind at 512 MB either. The remaining splitting is the logits constraint, which this patch does not
touch. That is the prediction the engine arms then tested.

---

## 3. The A/B

### 3.1 Three arms, one session, one line between the two that matter

| Arm | What it is | Boot | Patch | Knob |
|---|---|---|---|---|
| **production-11 reference** | the running production engine, **not restarted** — the same-session anchor | fast-load, already up | absent | — |
| **control** | a copy of the production tree with the patch **installed and inert** | eager, **325 s** | installed | none |
| **patched** | the same tree, the same bytes | eager, **352 s** | installed | `HAREM_INDEXER_WS_MODE=bound` |

The two A/B arms ran from a **copy** of the production tree, with fast-load **off**, and the
production tree, overlay and sidecars were not touched — verified afterwards. The reason is the
manifest identity: a new `patch-*.py` and an edited prelude change the fast-load sidecar's identity
([docs/08](../../docs/08-fast-boot.md) §4), so the alternative was a fresh ~53 GB-per-node dump, and
two of the three nodes did not have the disk. The environment file for each arm was derived on each
node from **that node's own** production file with `sed`; it differs from production in four lines
(tree, overlay, an empty fast-load mode, the extra environment) and **the two arms differ in exactly
one**.

**The control arm exists because the boot path moves the pool.** Production 11 boots with fast-load;
these arms do not, and an eager boot's `consumed` is measurably different — **93,664 tokens** of pool
between the production-11 anchor and the control. That is the boot path's price, not the patch's,
which is why every comparison below is **control against patched** and the anchor answers only "is
this machine at today's speed today".

### 3.2 The workspace itself — the hypothesis under test

`VLLM_DEBUG_WORKSPACE=1` is an upstream environment variable and needs no patch. On **all three ranks
in both arms**, the workspace was resized exactly **once**, by the same caller:

| Arm | The `WorkspaceManager` line | Grown by |
|---|---|---|
| control | `Resized workspace ... 0.00 MB -> ` **`5036.40 MB`** ` (ubatch 0)` | `sparse_attn_indexer_kpool.py:295` |
| patched | `Resized workspace ... 0.00 MB -> ` **`513.00 MB`** ` (ubatch 0)` | `sparse_attn_indexer_kpool.py:295` |

**The predicted sizes were 5,036.40 and 513.00 MB. The measured sizes are those numbers.** One resize
event means no other consumer — MoE, FlashInfer — sets this buffer's size, so the gain is not capped
by somebody else's requirement. That was the single most likely way for this item to be worth nothing,
and it is closed.

The patched arm's own startup line, verbatim from the log, on each rank:

```text
HAREM-IDXWS bound | upstream=40000000 entries (5035.4 MB) -> chosen=4067203 entries (512.0 MB),
saved 4.42 GiB | max_model_len=1000000 compress_ratio=4 entry_bytes=132 (index_head_dim=128)
max_num_seqs=8 num_spec=7 | per_request_floor=250002 (31.5 MB)
scheduler_ceiling=2000016 (251.8 MB) headroom=2.03x | safety=2x floor=512 MB
```

### 3.3 Memory and the KV pool

| | control | patched | delta |
|---|---:|---:|---:|
| Locked workspace | 5,036.40 MB | **513.00 MB** | **−4.42 GiB** |
| **KV pool, tokens** | 6,289,256 | **6,933,884** | **+644,628 (+10.25 %)** |
| Available KV — head | 44.52 GiB | 50.09 | +5.57 |
| Available KV — worker-1 | 44.76 GiB | 49.59 | +4.83 |
| Available KV — worker-2 | 44.90 GiB | 49.09 | +4.19 |
| **Binding rank (the smallest)** | 44.52 (head) | **49.09 (worker-2)** | **+4.57** |
| `consumed` (weights + non-torch), largest of the three | 59.61 GiB | 55.04 | −4.57 |
| `Model loading took` | 51.62 GiB | 51.62 | unchanged |
| Peak-activation reservation | 1.69 GiB | 1.69 | unchanged |
| CUDA-graph pool | 0.00 | 0.00 | unchanged |

**The binding rank moved, and the honest number is the worse one.** The pool follows the smallest
`Available KV` of the three ranks; the head node gave back the most (+5.57 GiB) and worker-2 the
least (+4.19), so which node binds changed between the arms. **+10.25 % is computed from the pool the
engine actually printed**, i.e. off the worst rank — the best rank alone would have been worth more.
The per-rank give-back reads between 4.19 and 5.57 GiB against a patch that frees exactly
4.4174 GiB on every rank; that spread is the profiler's own ruler, which measures a
`MemAvailable` difference and carries this much boot-to-boot noise
([docs/09](../../docs/09-measurement-protocol.md)).

**The conversion checks against an independent measurement.** 644,628 tokens over 4.57 GiB is
**141,057 tokens per GiB**, against the memory ladder's independently measured ~141,300
([`ladder-6sep.md`](ladder-6sep.md)). Two arms measured for different reasons give the same number.

**The anchor, for scale.** Production 11 read **6,382,920** tokens in the same session and
**6,385,674** when it was brought back at the end of the window — its own band.

**What does not improve, and this matters for the ladder.** `available_kv = requested −
non_kv_cache_memory`, so the 4.42 GiB does not return to the host: it goes straight into the pool and
the process's total stays pinned to `requested`. `MemAvailable` is therefore **unchanged**, this is
**not** a swap-headroom lever, and it does not move the rung where 0.90 failed. It **stacks with** the
ladder rather than competing with it — the ladder raises `requested`, this lowers `non_kv_cache_memory`.

### 3.4 Speed — the cost, looked for

Aggregate output tok/s, median of three rounds. Bands as [docs/09](../../docs/09-measurement-protocol.md)
§1.2: C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 % **within one boot** — these are two boots, and
boot-to-boot on C8 has been measured at 15.9 % with nothing changed at all.

| Level | production-11 anchor | control | patched | patched vs control | in band |
|---|---:|---:|---:|---:|---|
| C1 | 70.53 | 69.69 | **70.69** | **+1.4 %** | yes |
| C2 | 102.84 | 99.58 | 98.56 | −1.0 % | yes |
| C4 | 146.49 | 141.09 | 142.19 | +0.8 % | yes |
| C6 | 177.32 | 176.54 | 178.07 | +0.9 % | yes |
| C8 | 198.47 | 199.76 | **196.81** | **−1.5 %** | yes |
| prefill, fresh unseen prompts | 1,783 | 1,794 | 1,778 | −0.9 % | yes |
| prefill 7K, second prompt, uncached | 2,561¹ | 1,611 | 1,604 | −0.4 % | yes |
| TTFT C1 / C8, s | 0.276 / 0.786 | 0.281 / 0.801 | 0.279 / 0.798 | equal | — |
| Draft acceptance C1 / C8, % | 61.6 / 61.1 | 61.4 / 62.7 | 63.3 / 61.1 | equal | — |

¹ The anchor's 7K figure was taken with the prefix cache warm on an engine that had been up for
hours. It is not a comparison and is printed only so the column is not silently missing.

Per-stream medians over the same rounds, for the same three arms' control and patched columns:
C1 **76.30 → 76.41**, C2 53.17 → 54.03, C4 40.80 → 42.35, C6 34.08 → 33.79 tok/s.

**Read the signs, not the magnitudes.** C1, C4 and C6 move up, C2 and C8 move down, nothing leaves its
band, and the largest single move — C8 at −1.5 % — is half of that level's within-boot band and a
tenth of the boot-to-boot spread these two boots are separated by. The dry run predicted exactly
this — an identical chunk list, therefore no change — and the engine agreed. **The price looked
for was prefill: more chunks per prefill would have shown up in prefill-fresh and in TTFT, and
neither moved.**

### 3.5 Gates and stress

| # | Gate | control | patched | Threshold |
|---|---|---|---|---|
| K1 | correctness probe + code exam, **cold** | 10/10 · 12/12 | 10/10 · 12/12 | full |
| K2 | the same, **warm**, after the whole battery | 10/10 · 12/12 | 10/10 · 12/12 | full |
| K3 | tool-call gate | 8/8 | 8/8 | 8/8 |
| K4 | needle-lite, 64K and 128K × three depths | 6/6 | 6/6 | 6/6 |
| K5 | **one ~1M-token request** | not run | **PASS** | needle correct |
| K6 | **eight concurrent long-context lanes** | not run | **8/8 PASS** | every lane's needle correct |
| K8 | any resize after lock, any safety layer firing | none | **none** | must be zero |
| K9 | swap | 0.00 GiB | 0.000 GiB | production-11 level |

**K5 — the single case that can change the splitter's behaviour.** One request at the harness's
1550K label = **969,468 prompt tokens**, **572.4 s**, needle found. The whole chunk chain ran through
a buffer 9.8× smaller than the one it was written for.

**K6 — the case the buffer is actually about: many long prompts at once.** Eight lanes at the 128K
label, **80,112–80,114 prompt tokens each**, **640,904 in total**, **233.4 s** wall, **2,746 tok/s**
aggregate prefill, and **every lane returned its own needle**. That is the test that distinguishes a
gather mix-up from a plausible answer: a cross-lane corruption here shows up as the *wrong lane's*
string, not as a number that looks reasonable.

**K8 — the safety layers, which is the real reassurance.** After the whole stress battery the three
ranks' logs contain **one** resize line between them — the 513.00 MB at startup — and **zero**
occurrences of `Workspace is locked`, `AssertionError`, `RuntimeError`, `HAREM-IDXWS refuses` or
`K-gather workspace too small`. None of L0–L3 fired.

### 3.6 Memory pressure and swap

| | head | worker-1 | worker-2 |
|---|---:|---:|---:|
| `MemAvailable` min, patched arm, whole window **including the stress** | **2.43 GiB** | 4.70 | 4.27 |
| Swap used, peak | 0.000 GiB | 0.000 | 0.000 |
| `vmstat` swap-out, total for the window | 6 pages (~24 KiB) | 2 pages | 1 page |
| End-of-arm `MemAvailable`, control / patched | 3.3 / 3.2 GiB | 4.9 / 4.8 | 4.9 / 4.4 |
| OOM killer, `dmesg` | 0 | 0 | 0 |

**The 2.43 GiB is not a difference between the arms.** It was sampled during K6 — eight concurrent
long prefills — a load the control arm never ran. The comparable reading is the end-of-arm pair, and
those are equal. Production 11's own minimum in this session was 3.40 GiB. Swap use is zero and
single-digit pages of swap-out is noise.

**One `dmesg` caveat, scoped rather than waved away.** Worker-2 carries two
`NVRM ... NV_ERR_NO_MEMORY` lines, timestamped **before** this measurement window — they belong to the
production-11 reboot verification earlier that day. They are also not the Linux OOM killer in the first
place; that distinction cost a healthy rung a rejection once already ([`ladder-6sep.md`](ladder-6sep.md) §6).

---

## 4. The verdict

**A measured production-12 candidate: +10.25 % of KV pool at an unchanged memory fraction, for zero
measured speed cost.** The acceptance criterion set before the run — every gate full, pool ≥ +9 %,
speed inside its bands — is met on all three, and the mechanism the whole item rests on was confirmed
rather than assumed: the buffer that gets locked really is the indexer's, its size really is
5,036.40 MB, and no other consumer sets it.

**It is in production as of production configuration 12** (6 September 2026); what promotion cost, and what the promotion boot measured that this A/B could only estimate, is §7.1.

## 5. What this cost

**The measured speed cost is zero, and it was looked for** — in prefill-fresh, in TTFT and at five
concurrency levels; the signs are mixed and everything is inside its band.

**What is genuinely given up is diagnostic margin.** Upstream hands the indexer 20× the largest load
the scheduler can produce; this hands it **2.03×** against that ceiling and **16.27×** against the
one-request floor that is the only route to an over-size chunk (§2). That is a real reduction in
slack, bought back with a startup refusal at that floor — the load-bearing layer — and two armed
run-time assertions either side of the silent clamp. None fired under a 1M-token request or eight
concurrent long-context lanes.

**And there is a second, blunter cost: disk.** The patch changes the fast-load manifest identity, so
promoting it means a fresh sidecar of about **53 GB per node** — which does not fit today (§7).

**What it does not cost:** host headroom. The freed memory never returns to the host, so
`MemAvailable`, the swap picture and the ladder's rejected rung are all exactly where they were.

## 6. What this page does not settle

- **Both A/B arms are eager boots.** The production configuration boots with fast-load, which is worth
  93,664 tokens of pool on its own. The +10.25 % is a clean eager-against-eager comparison; the
  production figure with fast-load is **not measured**. Restoring that difference would put it near
  **7.03M tokens** `[estimate]` — and that estimate is exactly what the first boot after promotion has
  to check.
- **`HAREM_INDEXER_WS_MB`, the explicit override, was never booted.** Only `MODE=bound` ran on the
  engine; the override is covered by the CPU unit test, including its refusals `[not tested]` on
  hardware.
- **Two ranks — measured the same day, and it is worth three times as much there.** The buffer is
  sized from `max_model_len`, so a two-node stack reserves the same 4.92 GiB per rank against a pool
  a third the size. A same-session eager-boot A/B on two nodes, this file unchanged, one environment
  line between the arms: locked workspace **5,036.40 → 513.00 MB** — the numbers above to the decimal
  — and KV pool **1,800,000 → 2,378,571, +32.14 %** against +10.25 % here. With the fast-load sidecar
  restored the two-node recipe went **2,128,571 → 2,692,857, +26.5 %**. Gates full, stress clean,
  every concurrency level inside its band. [docs/15](../../docs/15-tp2-track.md) §5.9 and
  [`results/speed/tp2-production-candidate.md`](../speed/tp2-production-candidate.md) `[measured-here]`.
- **Other context lengths.** Everything above is at `max_model_len` 1,000,000. The upstream sizing is
  linear in it, so a 256,000-token deployment reserves 1.26 GiB and the gain shrinks with it
  `[estimate]`.
- **Whether the 512 MB floor is the right floor.** It is 2.03× a ceiling that is exact, and it was not
  swept. A smaller buffer is not obviously better — nothing measured says the extra 260 MB buys
  anything, and nothing says it does not.

## 7. What promotion took

Three steps, and the third was not a technical decision. All three were taken on 6 September:

1. **The patch into the production tree**, with the prelude hook beside the other TP=3 patches. One
   half of it lands on `sparse_attn_indexer_kpool.py`, which the launcher bind-mounts **read-only**
   from the overlay directory — so that half must be pre-applied to a host-side copy of the overlay,
   the same pattern the sm_12x set already uses
   ([`tracks/tp3/patches/README.md`](../../tracks/tp3/patches/README.md)).
2. **One environment line per node**, derived on that node with `sed`, never copied:
   `HAREM_INDEXER_WS_MODE=bound`.
3. **A fresh fast-load sidecar.** Step 1 changes the manifest identity, so the existing production
   sidecar is refused on the next `FASTLOAD_MODE=load` boot. A new one is **~53 GB per node**, and two
   of our nodes had 36 and 39 GB free — **an older sidecar had to be deleted first**, which is an
   owner's decision, not a measurement's. The one deleted was the production-9/10 sidecar, dead since
   production 11; the production-11 sidecar stayed, because it is what makes the rollback a single
   move. The trade was one for one: free space went 36 / 39 / 117 GB → 88 / 91 / 169 after the
   deletion → 36 / 39 / 117 again once the new sidecar was written.

### 7.1 What the promotion boot measured that this A/B could only estimate

Both arms above are **eager** boots, so the production KV figure was an `[estimate]` of ≈7.03 M and
the gates had not been read on a fast-load boot at the shipped fraction. Production 12 settled both,
against a same-session production-11 reference (running engine, no restart) `[measured-here]`:

| | production 11 reference (0.87) | **production 12** (0.88 + bound) |
|---|---:|---:|
| KV pool | 6,385,674 | **7,170,798 / 7,088,154 / 7,041,322** over three boots — headline the reboot boot, **+10.3 %** |
| Consumed per node | 58.3 – 59.1 GiB | **54.3 – 54.6 GiB** |
| Locked workspace | 5,036.40 MB | **513.00 MB**, exactly one resize per rank |
| C1 / C8, pooled over 6 rounds and 2 boots | 69.66 / 196.22 | **69.72 / 196.06** — +0.08 % and −0.08 % |
| Prefill, fresh | 1,739 | 1,737 (load boot) · 1,750 (clean boot) |
| Gates, cold and warm | 10/10 · 12/12 | 10/10 · 12/12 on both boots and on the systemd-started engine |
| One ~1M-token request | not run | **1/1**, 969,468 tokens, 569.6 s |
| Eight concurrent ~128K lanes | not run | **8/8**, 640,904 tokens, 227.5 s, 2,817 tok/s prefill |
| Safety layers fired | — | **none**, before or after the stress |
| Swap **in** under load | 0 | **0**, every sample, every node |

**The prediction was checked in absolute tokens, and it held.** This A/B priced the bound at
+644,628 tokens and the ladder priced the 0.87 → 0.88 rung at +179,063; the sum predicts 7,209,365
against a measured 7.04–7.17 M — inside the documented 6 % boot-to-boot pool spread. The two gains
add in **tokens**, not in percent, because the second one lands on a base the first has already
enlarged.

**Rollback is one line either way.** Unset the knob and the patched tree behaves as upstream; point
`ENV_FILE` at the production-11 environment file and the tree, overlay and sidecar all revert
together.

The patch, the knobs and the install note are in
[`tracks/tp3/patches/indexer-workspace/`](../../tracks/tp3/patches/indexer-workspace/).
The mechanism in its ledger context is [docs/17](../../docs/17-memory-ledger.md) §2.5; the standing
item is [docs/11](../../docs/11-open-issues.md) §2.28; the upstream half is
[HELP-WANTED](../../HELP-WANTED.md) §9.

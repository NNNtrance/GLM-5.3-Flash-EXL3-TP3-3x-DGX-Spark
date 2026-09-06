# 15 — Running this recipe at TP=2

This repository is a three-node recipe and every default in it is a TP=3 default. It also runs on
**two** DGX Spark nodes, and at two ranks it is a *simpler* recipe rather than a cut-down one: nothing
needs padding, so the shape surgery in [03](03-tp3-padding-and-sidecars.md) and the padded-load path
in [13](13-full-scope-checkpoint.md) §7 are not needed at all. This page is the two-node track: why it
works, exactly which files and flags change, what we measured at two ranks and when, and — at least
as important — the long list of things we never ran there.

**Read the trade-off first, because it is not the one people expect** (§4). At two ranks this stack is
slower on *every* speed axis we measured, single-stream included, and its KV pool is a fraction of the
three-node one. What TP=2 buys is a node and a shorter recipe, not latency.

**And read §3.5 before you run two ranks at all.** The one production setting this page used to list
as untested at TP=2 — the draft KV page, `HAREM_SW_BLOCK_SIZE=256` — is now measured there, and it is
not a tuning knob at two ranks. Without it the pool is 601,562 tokens and **a 6,253-token prompt is
never scheduled at all**; with it the pool is 1,303,571 and an 8,268-token prompt serves in 6.3 s,
with the quality gates unchanged `[measured-here]`. Set it.

Every number below carries its date, its image and its settings. Our TP=2 arms were run on four
different days' worth of stack, and three of the four are **older than production 10**; the table in
§3 says which is which. Evidence tiers are the ones in [STYLE-GUIDE.md](../STYLE-GUIDE.md).

---

## 1. Why two ranks need no padding, and what that removes

### 1.1 The five shapes

[03](03-tp3-padding-and-sidecars.md) §1 lists the five numbers in GLM-5.3-Flash that do not divide by
three. **All five divide by two**, and — this is the part that matters for an EXL3 checkpoint — all
five leave every rank a whole number of 128-column Hadamard blocks:

| Shape | Value | At TP=2 | Whole 128-blocks per rank | At TP=3, for contrast |
|---|---|---|---|---|
| `num_attention_heads` | 64 | 32 per rank | MLA `qk`/`v` head_dim 256, so yes | padded to 66 → 22 per rank |
| `num_key_value_heads` | 64 | 32 per rank | yes | 66 → 22 |
| KDA `linear_attn_config.num_heads` | 64 | 32 per rank | head_dim 128, so yes | 66 → 22 |
| `vocab_size` | 154,880 | 77,440 per rank = **605 × 128** | yes | `padding_size` → 384, giving 155,136 |
| shared expert intermediate | 2,048 | 1,024 per rank = **8 × 128** | yes | padded to 2,304 → 768 per rank |
| routed experts `moe_intermediate_size` | 2,048 | 1,024 per rank = **8 × 128** | yes | not sliced at all; 96 whole experts per rank under EP |

Two consequences follow, and they are the whole reason a two-node track is worth writing down.

**First: expert parallelism is optional at TP=2.** At three ranks it is mandatory, because 2,048/3 is
not an integer and a trellis cannot be zero-extended ([03](03-tp3-padding-and-sidecars.md) §1.1). At
two ranks the routed experts tensor-slice cleanly to 1,024 = 8 × 128 per rank, so both arrangements
are legal. `patches/tp3full/preflight-tp3.py` already encodes exactly this rule and will let
`--ep 0` through at `tp=2` while refusing it at `tp=3`: it requires `moe_intermediate_size` to be a
multiple of `128 × tp`, and 2,048 is a multiple of 256 but not of 384. **Every TP=2 arm we ran had EP
off** (§3). We have never measured EP on at two ranks `[not tested]`.

**Second: the padded-load path is not needed.** The full-scope checkpoint's TP=3 port exists because
`lm_head` had to be loaded narrow into a vocabulary vLLM had padded, and that needed three
capabilities the kernel author added in `f3e3090` and `754421f`
([13](13-full-scope-checkpoint.md) §7.2). At TP=2 there is no pad, so there is nothing to load
narrow into. Our TP=2 full-scope arm ran on **`exl3-zeus:62f53e6`**, an image that raises
`EXL3 weights cannot be zero-extended` at TP=3 `[measured-here]`.

The two launcher constants that production 9 moved from `lcm(64, tp)` to `lcm(128, tp)` are computed
at run time from `tp`, so `patches/tp3full/patch-vllm-tp3.py` is a **provable no-op at TP≤2** — 
`lcm(128, 2) = 128`, 154,880 is already a multiple of 128, and 2,048/2 = 1,024 = 8 × 128. Leave the
patch in place rather than editing it out; the file itself says so in its header.

### 1.2 What TP=2 does not get out of

Everything on this list is a property of the part, the fabric or the model, not of the rank count, and
it applies unchanged at two nodes:

- **The GB10 top-k overlay.** vLLM's sparse-attention indexer picks `persistent_topk`, which cannot
  run on this part — 85 CTAs against 48 SMs, and the fallback wants ≥128 KB of shared memory where
  the part has 101,376 bytes. This is the failure that stopped our very first TP=2 boot
  `[measured-here]`, and `HAREM_DISABLE_PERSISTENT_TOPK=1` is how it is cleared
  ([14](14-troubleshooting.md), [05](05-expert-parallel-and-cuda-exl3-fixes.md) §3.5).
- **`--block-size 256`.** Required at two ranks for the same reason as at three: with `index_kpool`
  4 and fp8 KV, DeepGEMM's arch-12 path needs `block_kv` exactly 64. Without it, built-in MTP dies in
  `attention.hpp:320` `[measured-here]`.
- **fp8 KV.** Same kernel constraint.
- **The fabric work in [06](06-nccl-mesh.md).** Two nodes are one peer pair rather than three, but
  `NCCL_MAX_NCHANNELS=8` addresses the plugin's per-channel RNR behaviour, not the number of peers,
  and the second-cable patch (`patches/kernel/0005`) exists because a *pair* had two cables and only
  one was carrying packets. If your two nodes are joined by one cable, set
  `NCCL_MESH_LINKS_PER_PEER=1`, which makes 0005 a no-op; `0006` (`NCCL_PTR_CUDA`) is worth measuring
  on its own either way. See [CONTRIBUTING](../CONTRIBUTING.md) item 4.
- **The per-rank fast-load sidecar** ([08](08-fast-boot.md) §2.4). It is per rank, so two ranks means
  two sidecars, each about 1.5× the size of a three-rank one. We never built one at TP=2
  `[not tested]`; every TP=2 boot in §3 is a cold boot without it, which is most of why they take
  355–471 s against production 10's 251 s.
- **The quality gates.** `scripts/correctness-probe.py` and `scripts/code-exam.py`, cold **and** warm.
  Both passed on every TP=2 arm we ran.

### 1.3 What it buys

- **A node.** That is the honest headline. Two Sparks instead of three.
- **A shorter recipe.** No `pad-tp3full.py` sidecar config, no `check-padload-tp3.py` image gate, no
  `svh = 0` pad audit, no 128-block arithmetic to get wrong, and an image requirement that drops from
  `754421f` to any image with the loader patch. The five asserts that exist to catch a silently
  half-padded stack have nothing to catch.
- **The loader work is visible on its own.** This is why our own full-scope dress rehearsal ran at
  TP=2 first: it answers "what is the dense stage worth" with no padding machinery on top
  ([13](13-full-scope-checkpoint.md) §4).

It does **not** buy lower single-stream latency on this stack. See §4.

---

## 2. To run this recipe at TP=2, change exactly these

Nine changes. Everything not listed here is unchanged, including the image build
([02](02-image-build.md)), the checkpoints ([01](01-model-and-license.md)), the DFlash2 port
([04](04-dflash2-port.md)), the measurement protocol ([09](09-measurement-protocol.md)) and the
troubleshooting index ([14](14-troubleshooting.md)).

### 2.1 The env file

Start from [`envs/env.tp3-full.example`](../envs/env.tp3-full.example) and change:

```
NNODES=2
```

```
TP_SIZE=2
```

```
ENABLE_EP=0
```

`ENABLE_EP=0` is the arrangement all our TP=2 measurements used. `ENABLE_EP=1` is also legal at two
ranks and we never measured it `[not tested]`; if you try it, the preflight will accept either.

Two more lines in the same file:

- `MODEL_HOST_PATH` — point it at the checkpoint itself, not at a padding sidecar. There is no
  padded `config.json` to build at TP=2, so `MODEL_LINK_TARGET` and the identity-mount machinery in
  `scripts/start-tp3.sh` have nothing to protect; setting both to the same real directory is correct
  and the launcher's `check_relative_sidecar` will pass it through unchanged.
- `DRAFT_HOST_PATH` — likewise. The DFlash2 drafter's GQA is 32/8, which divides by two, so the
  32/8 → 36/9 pad that TP=3 needs is not needed and `patch-dflash-tp3.py` becomes pad-aware over a
  pad that is not there.

**Keep `--hf-overrides` if you serve the full-scope checkpoint.** It is not about padding: that
checkpoint's inlined quantization config carries no `tensor_storage`, so `cuda-exl3` has to be pointed
at the standalone file. At TP=2 that file is the checkpoint's own, unrewritten:

```
EXTRA_ARGS='--block-size 256 --hf-overrides {"quantization_config_file":"/var/tmp/<your-checkpoint-dir>/quantization_config.json"}'
```

Single quotes and no spaces inside the JSON, for the reasons in
[`envs/env.tp3-full.example`](../envs/env.tp3-full.example) and [14](14-troubleshooting.md).

### 2.2 The launcher

[`scripts/start-tp3.sh`](../scripts/start-tp3.sh) hard-codes expert parallelism in three places,
because at TP=3 it is not optional. Three edits, and they are the only ones:

| Line | Today | At TP=2 |
|---|---|---|
| the `ENABLE_EP` guard | `[ "${ENABLE_EP:-1}" = "1" ] \|\| { echo "TP=3 requires ENABLE_EP=1 …"; exit 2; }` | delete, or widen it to fire only when `TP_SIZE` is 3 |
| `EP_ARG=(--enable-expert-parallel)` | unconditional | `EP_ARG=(); [ "${ENABLE_EP:-0}" = "1" ] && EP_ARG=(--enable-expert-parallel)` |
| the container environment | `-e ENABLE_EP=1` | `-e ENABLE_EP="$ENABLE_EP"` |

`NNODES`, `TP_SIZE` and `--tensor-parallel-size` already come from the env file. **Boot order becomes
worker-1, then head** — the same rule as at three ranks, highest rank first.

### 2.3 The patch tree

Use [`patches/tp2/patch-fullscope-tp2.py`](../patches/tp2/patch-fullscope-tp2.py) in place of
`patches/tp3full/patch-fullscope-tp3.py`. The relationship is exact and is stated in
[`patches/tp3full/README.md`](../patches/tp3full/README.md):

| | `patches/tp2/patch-fullscope-tp2.py` | `patches/tp3full/patch-fullscope-tp3.py` |
|---|---|---|
| A1–A8 (S1 packed mapping, S2 quant gates, S3 KDA split) | identical text | identical text |
| **A9** — `linear.py`, split a fused checkpoint tensor by the *checkpoint's* widths rather than the module's padded ones | absent | present; **a no-op at TP≤2** |
| **A10** — post-load audit: every padded EXL3 site is whole 128-blocks and exactly zero | absent | present; there is nothing to audit at TP=2 |

Everything else in the tree can be used as it stands. `patch-vllm-tp3.py` computes `lcm(128, tp)` and
is a no-op at two ranks (§1.1); `patch-swblock-tp3.py`, `patch-epfilter-tp3.py`,
`patch-draftkv-tp3.py`, `patch-fastload-tp3.py` and `patch-kvdiag-tp3.py` are all `tp`-agnostic and
all gated on their own env knobs. `check-padload-tp3.py` can be dropped: it gates on a capability
only the padded-load path needs.

**Keep the file list in its own directory.** The directory's file list and the full text of the
prelude are hashed into the fast-load manifest ([08](08-fast-boot.md) §4), and adding one file to a
tree — even a file that is never called — refuses the next boot on every node. That has happened to
us twice, and once it was **the TP=2 patch dropped into the TP=3 tree** that did it
([13](13-full-scope-checkpoint.md) §6.4). A `patches/tp2/` tree of its own is the answer, exactly as
`patches/tp3full/` is.

### 2.4 The image

`exl3-zeus:62f53e6` is enough at TP=2 `[measured-here]` — that is the image our full-scope TP=2 arm
ran on. `754421f` also works and adds nothing you need at two ranks. Build it exactly as
[02](02-image-build.md) says.

### 2.5 Flags that stay, and one expectation that changes

| Flag / knob | At TP=2 |
|---|---|
| `HAREM_DISABLE_PERSISTENT_TOPK=1` | **mandatory**, same reason (§1.2) |
| `--block-size 256` | **mandatory**, same reason |
| `--kv-cache-dtype fp8` | **mandatory**, same reason |
| `NCCL_MAX_NCHANNELS=8` | keep; it is a plugin property, not a peer-count property |
| `NCCL_MESH_LINKS_PER_PEER` | `0` (auto) with two cables between the pair; `1` with one cable, which makes `patches/kernel/0005` a no-op |
| `NCCL_MESH_MIN_RNR_TIMER=1`, `NCCL_MESH_PTR_CUDA=1`, `NCCL_MESH_FLUSH=1` | keep |
| `HAREM_SW_BLOCK_SIZE=256` | **mandatory in practice** — measured at TP=2 in §3.5: the pool more than doubles and the long-prompt path only exists with it `[measured-here]` |
| `HAREM_DRAFT_KV_DTYPE=fp8` | keep — never run at TP=2 `[not tested]` |
| `CUDA_EXL3_TUNE_CACHE` | keep, and warm it before you measure anything ([12](12-tuner-cache.md)) |
| `GPU_MEMORY_UTILIZATION` | our TP=2 arms all ran **0.85**, not production 10's 0.83; read §3.4 before copying either |
| `MAX_MODEL_LEN` | 1,000,000 is reachable with the experts-only checkpoint at TP=2 and **was not** with the full-scope one (§3.2) |

### 2.6 The autostart unit

[`systemd/harem-exl3.service`](../systemd/README.md) needs three edits and the preflight one:

- `WorkingDirectory` and `ExecStart` — point both at your TP=2 tree instead of `tp3full`.
- `ExecStop` — the container name if you changed it.
- `Conflicts=harem-motor.service` — keep it, and disable the sibling recipe's unit in the same
  change. That hazard is about two engines on one node's unified memory and has nothing to do with
  rank count.
- `motor-onkosul-exl3.sh` — `FABRIC_PEERS` becomes **one** address per node instead of two, and the
  ConnectX-7 check stays at `4/4` because that counts ports on the node, not peers.

And the reboot rule becomes "reboot **both** together, never one". The preflight checks only its own
node's fabric, so a single-node reboot passes it and starts a rank into a cluster whose peer is gone
([00](00-hardware-and-os.md) §3.4). We have not run a two-node reboot test `[not tested]`; the
three-node one is in [systemd/README](../systemd/README.md).

---

## 3. What we measured at two ranks

Four arms, all on two DGX Spark nodes, all at temperature 0 and reasoning effort **low**, all with the
GB10 top-k overlay and `--block-size 256`. **Arms A and B are bring-up figures on an older stack;
arms C and D are the current one.** Do not compare a number in arm A against a number in arm D.

### 3.1 Arms A and B — bring-up, 4–5 September 2026

**Arm A.** Image `exl3-zeus:serve` (`cuda-exl3` `37330c9`), checkpoint
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` revision `b20c49ba` (routed experts EXL3 4-bit, head 16-bit),
TP=2, EP off, `--attention-backend CUSTOM`, KV fp8, `gpu-memory-utilization 0.85`, no KV pin,
`--max-model-len 1000000`, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`, mesh plugin,
**before** any of the fabric work in [06](06-nccl-mesh.md). Two speculative arrangements: none, and
the image's built-in MTP at k=3. Published as `Zeuss5/cuda-exl3` issue #2, 4 September 2026
`[measured-here]`.

| | no draft | MTP k=3 |
|---|---|---|
| Boot to serving | **471 s** (weights 312 s, 79.95 GiB per node) | 471 s |
| KV pool at 0.85, fp8, 1M context | **2,548,117** tokens | 1,987,179 (the MTP layer plus block 256: −22 %) |
| Quality gates | 10/10 correctness, 12/12 code exam, 0 empty | 10/10 · 12/12 · 0 empty |
| Single stream, 700-token code prompt | 14.6 tok/s | 30.3–30.6 tok/s, acceptance 70 % |
| `hizset-v2` C1 / C2 / C4 / C6 / C8, per stream | 14.7 / 14.9 / 12.0 / 8.6 / 9.0 | 33.7 / 27.5 / 20.2 / 16.0 / 14.3 |
| the same, aggregate | 14.4 / 28.8 / 46.4 / 50.6 / **70.6** | 30.6 / 48.0 / 71.0 / 85.9 / **102.4** |
| accepted tokens per step | — | 3.3, acceptance 76–78 % |
| Prefill, 7.4K prompt, cold cache | **1,137** tok/s | 1,131 tok/s |
| TTFT, C1 warm / C8 | 0.34 s / 0.8 s | 0.4–0.6 s / 1.3 s |
| Category at C1, prose / code / math / JSON | — | 21.3 / 31.1 / 34.2 / 35.3 tok/s (acceptance 38 / 73 / 79 / 84 %) |
| MMLU sample, 57 × 35 = 1,995 q, 0-shot | — | **86.4 ±0.7** |
| Free host RAM after load tests / swap | head 5.2 GB / **3.5 GB swap**; worker-1 7.1 GB / 2.8 GB | head 4.8 GB, worker-1 6.7 GB |

The swap in that last row appeared **during weight load** and was stable afterwards. It is the
clearest single reading of what two nodes cost: 80 GiB of weights per node against production 9's
58.3–59.1 GiB, on a part where the GPU and the host share one 121.6 GiB pool.

**Arm B — the DFlash2 port at two ranks**, image `exl3-zeus:dflash`, drafter
`incoai/GLM-5.3-Flash-DFlash2` at k=7, same checkpoint and same env otherwise, mean of two
`bench-sweep` rounds, 4–5 September 2026 `[measured-here]`. This is the arm that established that
DFlash2 is the fastest option on this stack, and the three columns are directly comparable to each
other because they were run back to back on one boot:

| | no draft | MTP k=3 | **DFlash2 k=7** |
|---|---|---|---|
| C1 aggregate | 14.42 | 30.49 | **42.91** |
| C1 per stream | 14.73 | 33.52 | **50.79** |
| C2 / C4 / C6 aggregate | 28.80 / 46.49 / 49.30 | 47.89 / 71.12 / 85.23 | **60.80 / 83.89 / 98.08** |
| C8 aggregate | 69.50 | 102.32 | **114.60** |
| acceptance · accepted tokens per step | — | 77.3 % · 3.31 | **62.4 % · 5.37** |
| KV pool | — | 1,987,179 | **825,000** |
| Prefill 7k | 1,137 | 1,131 | 1,035 |
| Category at C1: math / JSON / code / prose | — | 34.2 / 35.3 / 31.1 / 21.3 | **48.5 / 45.6 / 38.2** / 18.5 |
| Gates | 10/10 · 12/12 | 10/10 · 12/12 | 10/10 · 12/12 |

**What that cost.** The KV pool falls from 1,987,179 to 825,000 — **−58 %** — because the drafter's
KV group is allocated on a 16-token page. That is the same defect [07](07-kv-and-draft-page.md) §3
diagnoses and fixes with `HAREM_SW_BLOCK_SIZE=256`; **that fix is now measured at two ranks and it
more than doubles the pool** (§3.5). And prose gets *slower*, 21.3 → 18.5 tok/s: at k=7 the wasted draft costs more than the
occasional hit is worth on high-entropy text, and prose acceptance is 12.8 % against MTP's 37.7 %.

### 3.2 Arms C and D — the current stack, 5 September 2026

Both arms: two nodes, TP=2, EP off, image `exl3-zeus:62f53e6`, EXL3 weights, KV fp8, DFlash2 k=7,
`gpu-memory-utilization 0.85`, `--block-size 256` requested, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `NCCL_MAX_NCHANNELS=8`, mesh plugin with both cables per peer,
`CUDA_EXL3_TUNE_CACHE` warm, **no** fast-load sidecar, **no** `HAREM_SW_BLOCK_SIZE`, **no** fp8 draft
cache, temperature 0, reasoning effort low, medians of three rounds `[measured-here]`. Arm C is
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` (routed experts only); arm D is
`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (full scope). **Arm D ran at `max_model_len` 65,536
against arm C's 1,000,000**, because at 1M it could not boot, so only C1–C4 are a comparison between
them; C6, C8, every prefill figure and the pool are not. The full account is
[13](13-full-scope-checkpoint.md) §4–§6.

| metric | **C — experts-only** | **D — full-scope** | delta, where it is one |
|---|---|---|---|
| C1 per stream (tok/s) | 54.69 | **68.00** | **+24.3 %** |
| C1 aggregate (tok/s) | 47.40 | **59.93** | **+26.4 %** |
| C2 aggregate | 68.03 | 83.02 | +22.0 % |
| C4 aggregate | 90.66 | 111.05 | +22.5 % |
| C6 aggregate | 110.12 | 109.75 | **void — arm D is KV-bound** |
| C8 aggregate | 133.57 | 110.03 | **void — arm D is KV-bound** |
| TTFT at C1 | 0.615 s | 0.524 s | −14.8 % |
| draft acceptance at C1 · accepted tokens per step | 64.08 % · 5.49 | 63.14 % · 5.42 | equal |
| decode step time | 100.4 ms | 79.7 ms | −20.6 % |
| cold first request | TTFT 1.44 s, 40.5 tok/s, acceptance 45.5 % | TTFT 0.85 s, 47.5 tok/s | faster |
| Boot, cold, no fast-load sidecar | 396 s | 355 s | −10 % |
| KV pool | **665,625** at 1,000,000 ctx | 31,343 at 65,536 ctx | **not comparable** |
| Prefill, three fresh unseen 8,204-token prompts | **1,334** tok/s (1,329 / 1,334 / 1,447) | **not measurable** | see below |
| Prefill 7k, uncached | 1,135 tok/s | not measurable | |
| Gates, cold and warm | 10/10 · 12/12 both | 10/10 · 12/12 both | equal |
| MMLU sample, 1,995 q, 0-shot | 86.4 ±0.7 (measured on this checkpoint at TP=2 earlier the same day) | **86.32 ±0.75** | inside the error bar |
| Free host RAM after the run | 6.6 / 7.7 GiB | 8.9 / 10.6 GiB | full scope is lighter |

**Arm D's cost line, and it is the reason TP=2 is a rig and not a serving configuration.** At two
ranks the full-scope checkpoint leaves so little KV that the hybrid allocator prints
`Setting attention block size to 4608 tokens to ensure that attention page size is >= mamba page size`
and the pool comes out at 31,343 tokens — about 6.8 pages. Measured admission: an 844-token prompt
serves in 1.1 s, a 1,684-token prompt in 1.7 s, and a ~2,800-token prompt is **never scheduled** —
45 s later still `Running: 0, Waiting: 1` with KV usage at 0 % `[measured-here]`. `--block-size 256`
has no effect in that state; the allocator overrides it.

**Arm C's cost line is the plain two-node one.** 665,625 tokens against production 8's 4,699,724 on
the same day and the same script — **14 %** — because a 164 GiB model over two nodes leaves about
85 GiB of weights per node and very little else.

### 3.3 What we did **not** do at TP=2 — the honest list

Everything in this table is in production at three ranks and has never been run at two `[not tested]`.
Where an effect at TP=3 lets us say something useful about the likely direction, it is marked
`[estimate]` and is a projection, not a measurement.

**One row has left this table.** `HAREM_SW_BLOCK_SIZE=256` was its first entry and its largest
projection; it was measured at two ranks on 6 September 2026 and now has its own section, §3.5. The
projection it carried — "perhaps 0.8–1.2 M from arm C's 665,625" — was **low**, and the reasoning
behind it was incomplete: at two ranks the defect is worse than at three, not merely as bad.

| Not run at TP=2 | What it did at TP=3 | Expectation at TP=2 |
|---|---|---|
| **`HAREM_DRAFT_KV_DTYPE=fp8`** — the drafter's own cache at fp8 ([07](07-kv-and-draft-page.md) §7) | +5.6 % of pool, acceptance unchanged, speed within noise | should carry; not measured |
| **`gpu-memory-utilization` 0.83** (production 10) | +8.7 % of pool over 0.80, swap flat | our TP=2 arms ran **0.85**, and arm A recorded 3.5 GB of swap on the head during weight load at that rung. The three-node memory rule and its ladder ([11](11-open-issues.md) §2.4) were derived at three ranks and should be re-derived at two, not copied |
| **The fast-load sidecar** ([08](08-fast-boot.md)) | boot 618 → 274 → 251 s; weights 426 → 58 s | per rank, so two sidecars of ~1.5× the size. Every TP=2 boot above is a cold one without it |
| **The `CUDA_EXL3_TUNE_CACHE` protocol** ([12](12-tuner-cache.md)) | three rounds instead of five | arms C and D had a warm cache; arms A and B did not, and were not run to the five-round rule either |
| **The dual-cable plugin patch** (`patches/kernel/0005`) | +73 % on a 64 MB all-reduce, +4–6 % end to end | arms C and D carried it. Whether a **single** peer pair sees the same is unmeasured, and if your pair has one cable it is a no-op by construction |
| **The autostart unit and a two-node reboot test** ([systemd](../systemd/README.md)) | `/health` 200 at 242 s by the harness's counter, 315 s by the wall clock | never installed at TP=2 |
| **Expert parallelism at two ranks** | mandatory at three | legal at two, never measured |
| **Prefill and C6/C8 for the full-scope checkpoint at two ranks** | measured at three ([13](13-full-scope-checkpoint.md) §7.3) | impossible in arm D's configuration — the long-prompt path is closed (§3.2) |
| **A five-round arm, and a boot-to-boot spread** ([09](09-measurement-protocol.md) §1.1) | C1 spans 1.1 %, C8 2.5 %, **C4 7.4 %** across boots at three ranks | every TP=2 number above is a median of two or three rounds on **one** boot. Treat differences under about 5 % as noise |

This page therefore only **partly** closes [CONTRIBUTING](../CONTRIBUTING.md) item 10. Arms C and D
are on the production image with the fabric work in place; what is still missing at two ranks is the
fp8 draft cache, fast load, the memory ladder and a second boot. The draft page came off this list
on 6 September 2026 — §3.5.

### 3.4 One reading that did not survive, and is kept here

Arm D at `max_model_len=1,000,000` failed its own budget gate with **0.73 GiB** of available KV
memory where the arm C control had 4.4 GiB, implying the full-scope checkpoint was about **10 GiB
heavier per node** — while post-run free host memory on the same arm said its weights were ~3.3 GiB
per node *lighter* `[measured-here]`. Those two readings contradict each other and we never isolated
the mechanism. At three ranks the **sign reversed**: 3.4 GiB lighter per node and the pool 10 %
larger. The TP=2 delta is marked **not reproduced** rather than explained
([13](13-full-scope-checkpoint.md) §6.2, §7.5) `[retracted]`.

A separate finding from the same investigation does stand and is worth carrying to any node count:
lowering `max_model_len` from 1,000,000 to 65,536 raised available KV memory from 0.73 to **5.41
GiB**, so roughly **4.7 GiB per node of persistent, non-KV allocation scales with `max_model_len`** on
this stack `[measured-here]`.

One number in the paragraph above is now doubtful in its own right: "the arm C control had 4.4 GiB"
of available KV memory. Re-running that exact env file on 6 September (§3.5) printed **9.97 GiB** on
the binding rank and 11.68 GiB on the other. We cannot reconcile them — arm C's container log was
not kept, only its extracted pool figure — so the 4.4 GiB is best read as an inference from the
pool rather than as a log line, and the "10 GiB heavier per node" arithmetic that rests on it is
weaker still. It stays `[retracted]`.

### 3.5 The draft KV page at two ranks — measured, 6 September 2026

This closes the largest open item on the page, and the answer is stronger than the projection §3.3
carried. Two arms back to back, one boot each, **the only difference one token in `EXTRA_ENV`**
`[measured-here]`. Full raw record: [`results/speed/tp2-draft-page.md`](../results/speed/tp2-draft-page.md).

**Settings, identical in both arms.** Two nodes, TP=2, EP off, image `exl3-zeus:62f53e6`,
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` at `b20c49ba` (routed experts only), KV `fp8`, DFlash2 k=7,
`--attention-backend CUSTOM`, `--block-size 256`, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `--max-model-len 1000000`, `gpu-memory-utilization 0.85`,
`HAREM_DISABLE_PERSISTENT_TOPK=1`, `NCCL_MAX_NCHANNELS=8`, mesh plugin with both cables,
`CUDA_EXL3_TUNE_CACHE` warm, no fast-load sidecar, no fp8 draft cache, temperature 0, reasoning
effort **low**, medians of three rounds. Both arms booted through the host-side settle gate
(`MemAvailable ≥ 112 GiB`; they started at 116.9 and 117.1 GiB), so the pool figures satisfy the
acceptance rule in [07](07-kv-and-draft-page.md) §1.1.

The patch is [`patches/tp3/patch-swblock-tp3.py`](../patches/tp3/patch-swblock-tp3.py) **unchanged** —
it is `tp`-agnostic and gated on its own environment variable, so the control arm ran the same image
with the knob unset. Keep it in a `patches/tp2/` tree of your own (§2.3); at two ranks there is no
fast-load manifest to invalidate, which is the one hazard that rule exists for.

#### Why the defect is *worse* at two ranks

The engine's own per-group decomposition, printed by `patch-kvdiag-tp3.py` in both arms:

| group | page, control | blocks/request, control | page, fix | blocks/request, fix |
|---|---|---|---|---|
| `MLAAttentionSpec`, 22 layers | 152,064 B | 218 | 152,064 B | 218 |
| `KpoolTailSpec`, 11 layers | 152,064 B | 1 | 152,064 B | 1 |
| `MambaSpec` × 4 | 2,359,296 B | 9 + 9 + 9 + 9 | 2,359,296 B | 9 + 9 + 9 + 9 |
| **`SlidingWindowSpec` — the drafter, 5 layers** | **32,768 B** (16 tokens) | **385** | **524,288 B** (256 tokens) | **25** |
| **blocks per request** | | **640** | | **280** |
| `num_blocks` | | 385 | | 365 |
| **`GPU KV cache size` at 1M** | | **601,562** (0.60x) | | **1,303,571** (1.30x) |

**The drafter takes 60.2 % of the divisor at two ranks against 53 % at three.** Not because the
drafter changed — its 385 is identical — but because the target's share shrank: the platform raises
the attention block to **4,608** tokens at TP=2 where it is 3,328 at TP=3, and that cuts the MLA
group from 301 blocks per request to 218. A smaller denominator makes the same defect a larger
share, which is why §3.3's projection from the TP=3 percentage came out low.

#### What it did

| metric | control — page 16 | **fix — page 256** | delta |
|---|---|---|---|
| **KV pool at 1M** | 601,562 | **1,303,571** | **+116.7 %** |
| the same, normalised to equal binding-rank KV memory | 601,562 | 1,260,714 | **+109.6 %** |
| **prefill-fresh, 3 unseen ~8.3K prompts** | **never scheduled** | **1,478 tok/s** | path opens |
| **prefill, 7,382 tokens, uncached** | **never scheduled** | **1,267 tok/s** | path opens |
| **largest prompt actually served** | **5,386 tokens** | **8,268**, every size served | path opens |
| C1 aggregate | 47.41 | 47.30 | equal |
| C1 per stream | 55.73 | 51.34 | −7.9 %, see the cost line |
| C2 / C4 / C6 aggregate | 68.27 / 91.65 / 113.22 | 68.72 / 93.74 / 117.71 | +0.7 / +2.3 / +4.0 % |
| **C8 aggregate** | 127.54 | **135.59** | **+6.3 %** |
| **TTFT median, C1 / C8** | 0.621 / 1.703 s | **0.478 / 1.244 s** | **−23 % / −27 %** |
| draft acceptance · tokens per step, C1 | 64.48 % · 5.51 | 62.56 % · 5.38 | −1.9 points |
| gates, cold **and** warm | 10/10 · 12/12 · 0 empty | 10/10 · 12/12 · 0 empty | equal |
| boot, cold, no sidecar | 396 s | 375 s | −5 % |

The direction and size match three ranks exactly where they should: there the same change bought
+82 % of pool, +6 % at C8 and 20–30 % off TTFT ([07](07-kv-and-draft-page.md) §3).

#### The cliff, which is the real finding

The control does not serve a long prompt *slowly*. It does not serve it at all. Walking prompt
length up one request at a time, 75-second budget `[measured-here]`:

| prompt tokens | control | fix |
|---|---|---|
| 913 / 1,786 / 2,759 / 3,586 / 4,444 | served, 1.5–3.8 s | served |
| 5,386 | served, 4.1 s | served |
| **6,253** | **never scheduled** | served, 4.9 s |
| 7,329 / 8,268 | — | served, 5.5 / 6.3 s |

`Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0 %`, indefinitely — the same state
[13](13-full-scope-checkpoint.md) §6 recorded for the full-scope TP=2 arm, and the same arithmetic:
block ids are global to one pool, one request wants 640 of the pool's 385 blocks, so it is never
admitted. After the fix it wants 280 of 365.

**And the control sits on the edge of that cliff by luck.** The arm C boot published in §3.2 got
665,625 tokens — 426 blocks — and did serve a 7,382-token prompt, at 1,135 tok/s. The same env file,
untouched, booted through a settle gate on 6 September gets 601,562 — 385 blocks — and cannot. The
difference is the instrument, not the stack: the 5 September TP=2 harness had no settle gate, and
[07](07-kv-and-draft-page.md) §1.1 says a node that starts dirty awards itself memory it does not
have. Both readings are real boots of the same configuration. That is the point: **without the page
fix, whether two ranks can serve an 8K prompt depends on the host's state at boot.** Treat the arm C
prefill figures in §3.2 and §4 as measured on an unpinned baseline `[measured-here]`.

#### What it cost

- **Per-block memory rises about 9.1 %** — the drafter's per-block cost goes 163,840 → 2,621,440 B
  and `num_blocks` would fall 385 → 353 at equal memory. At three ranks the price was +9.2 %.
- **The draft group's prefix-cache matching unit coarsens 16 → 256 tokens.** Nothing measurable
  here: with a 4,608-token attention block the prefix-cache hit rate is 0 % in both arms.
- **Acceptance falls 1.9 points** (64.5 → 62.6 %), inside this stack's 60–65 % boot-to-boot band,
  not confirmed on a second boot.
- **C1 per-stream decode reads 7.9 % lower**, the one number that moved the wrong way. The fix arm's
  own three rounds span 50.60–55.29 (9.3 %), the C1 aggregate is equal and TTFT is 23 % better, so
  one boot does not establish a loss. No C1 gain, and no proven C1 loss.
- **Free host RAM falls about 1.2 GiB per node**; swap flat at 0.02–0.03 GiB in both arms.

#### Still not done at two ranks

One boot per arm. `HAREM_DRAFT_KV_DTYPE=fp8`, fast load, the memory ladder and expert parallelism
remain `[not tested]` here, and the page fix was **not** tried on top of the full-scope checkpoint at
TP=2, whose pool is clamped by a different mechanism (§3.2) — that one is still open.

---

## 4. The trade-off, measured

Three nodes win. On this stack the third node is not only a bigger memory pool — it is faster, per
stream as well as in aggregate, and that is the opposite of what "fewer ranks, fewer collectives,
lower latency" would predict.

Like for like, same day, same image, same harness, same experts-only checkpoint, DFlash2 k=7
`[measured-here]`:

| | TP=2 (arm C, 2 nodes) | TP=3 production 8 (3 nodes) | two-node share |
|---|---|---|---|
| C1 per stream | 54.69 | 59.94 (and 64.24 on a second run of the same day) | 85–91 % |
| C8 aggregate | 133.57 | 178.55 | 75 % |
| Prefill, fresh unseen prompts | 1,334 | 1,774 | 75 % |
| **KV pool at 1M** | **665,625** | **4,699,724** | **14 %** |
| **the same, with the draft page fix at both ranks** (§3.5) | **1,303,571** | 4,413,223 | **30 %** |
| draft acceptance at C1 | 64.1 % | 63.9 % | equal |
| Boot | 396 s, no fast-load | 265 s, fast-load | — |
| Gates | 10/10 · 12/12 | 10/10 · 12/12 | equal |

Two rows in that table are arm C's and were measured **without** the draft page fix and **without**
the settle gate; §3.5 re-measured the same configuration with the gate and got 601,562 tokens and no
long-prompt path at all. Read the 14 % row as "two ranks, neither node's stack fixed"; the 30 % row
is the fair comparison once both rank counts run the setting this repository ships.

And with the full-scope checkpoint, one arm at each rank count — not like for like, because arm D
ran at `max_model_len` 65,536 and production 9 at 1,000,000, but the direction is not in doubt:
**68.00 tok/s per stream at two ranks against 75.91 at three** `[measured-here]`.

**Why the third node also wins on latency.** A decode step here is weight-bandwidth bound, and the
profile says so: at production 7 the dense BF16 GEMM alone was 45.3 % of a single-stream step, and in
the engine the MoE trellis GEMM runs at **78–85 % of a measured 225 GB/s** ruler
([10](10-results-and-roofline.md) §5). Adding a rank cuts each rank's weight traffic by a third. The
collective it costs in exchange is real but bounded: on production 9, NCCL and the CPU gap **together**
are at most **17.19 ms of a 72.5 ms step** with the profiler off `[measured-here]`, and going from two
peers to three does not add all of that. The bandwidth term wins, and it is not close.

**What TP=2 wins, plainly.** One node, and a recipe with the whole of
[03](03-tp3-padding-and-sidecars.md) and [13](13-full-scope-checkpoint.md) §7 taken out of it. That
is a real answer for anyone who owns two Sparks, and it is the whole reason this page exists — but it
is not a latency argument, and this repository will not make one.

**Where the pool difference comes from, since it is the largest number in the table.**
164 GiB of weights over two nodes is about 82 GiB per node before anything else is allocated;
121.6 GiB of unified memory minus that, minus the ~9 GiB vLLM takes at init and the ~4.7 GiB of
`max_model_len`-scaled buffers, leaves single-digit GiB for KV — we measured 9.97 GiB on the binding
rank at 0.85 `[measured-here]`. Over three nodes the same weights are ~55 GiB per node and the
remainder roughly quadruples. The third node buys 1.5× the memory and delivers several times the
pool, because what is left over after a fixed cost is not linear in the memory.

**With the draft page fixed at both rank counts the gap narrows from seven-fold to about 3.4×**, and
the reason is in §3.5: the page defect was costing two ranks more than three, so fixing it helps the
two-node side more. The third node is still the larger memory argument by a wide margin. What
changed is that at two ranks the pool is now above one full-length request instead of below it —
concurrency 1.30x against 0.60x — which is the difference between a configuration that serves 8K
prompts and one that silently never schedules them.

---

## 5. If you run this at two ranks, tell us

[CONTRIBUTING](../CONTRIBUTING.md) has the format. The four things worth most from a two-node
cluster, in order:

1. **A second boot of the draft page arms in §3.5.** That measurement is done — pool +117 %, the
   long-prompt path opened, gates unchanged — but it is one boot per arm, and the three-node
   boot-to-boot spread is up to 7.4 % at C4. The pool and cliff findings are far too large to be
   boot noise; the speed rows are not, and C1 per stream is the one that most needs a second look.
3. **Expert parallelism on at two ranks.** Legal, never measured, and it changes which kernel path
   the MoE stage takes ([05](05-expert-parallel-and-cuda-exl3-fixes.md)).
4. **The memory ladder at two ranks.** 0.85 is where our arms ran and it is also where arm A found
   3.5 GB of swap during weight load. Where the safe rung actually sits at two nodes is unknown.

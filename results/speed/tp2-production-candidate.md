# The TP=2 production candidate — two complete configurations, measured

The raw record behind [docs/15](../../docs/15-tp2-track.md) §5. Two candidates, measured 6 September
2026 on two DGX Spark (GB10) nodes — `head` (rank 0) and `worker-1` — one boot each, with the full
protocol in [docs/09](../../docs/09-measurement-protocol.md) `[measured-here]`.

**Settings, identical in both candidates.** TP=2, **expert parallelism off**, image
`exl3-zeus:754421f`, patch tree [`patches/tp2full/`](../../patches/tp2full/), launcher
[`scripts/start-tp2full.sh`](../../scripts/start-tp2full.sh) with its `MemAvailable ≥ 112 GiB` settle
gate, KV `fp8`, **fp8 draft cache** (`HAREM_DRAFT_KV_DTYPE=fp8`), DFlash2 draft at k=7,
`--attention-backend CUSTOM`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `--max-model-len 1000000`, `gpu-memory-utilization 0.85`,
`HAREM_DISABLE_PERSISTENT_TOPK=1`, `NCCL_MAX_NCHANNELS=8`, mesh plugin (`patched2`, both cables per
peer, `NCCL_PTR_CUDA`), warm `CUDA_EXL3_TUNE_CACHE`, per-rank fast-load sidecar,
`--safetensors-load-strategy eager`, `--no-enable-flashinfer-autotune`, temperature 0, reasoning
effort **low**, `max_tokens` 256, prompts `scripts/hizset-v2.jsonl`. Both nodes' settle gates opened
at 116.9–117.1 GiB, so the pool figures satisfy the acceptance rule in
[docs/07](../../docs/07-kv-and-draft-page.md) §1.1.

**Speed protocol: four sweep rounds; round 1 is reported on its own as the tuner-cache check and the
result is the median of rounds 2–4.**

**The only differences between the two candidates:**

| | **A — experts-only** | **B — full-scope (recommended)** |
|---|---|---|
| checkpoint | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ `b20c49ba`, 164 GB on disk | `turboderp/GLM-5.3-Flash-exl3` @ 4.05 bpw, 154 GB on disk |
| `HAREM_EXL3_FULLSCOPE` | unset | `1` → `patch-fullscope-tp2.py` (S1 packed mapping, S2 attention dtype, S3 KDA split) |
| `--hf-overrides` | absent | `{"quantization_config_file":"/models/<ckpt>/quantization_config.json"}` |
| `CUDA_EXL3_DEBUG_NAMES=1` audit | shared expert, `o_proj`, `lm_head` all `-> unquantized` | **203 EXL3 / 113 bf16**; `lm_head`, `shared_experts.*`, `o_proj`, `in_proj_qkv` all `-> EXL3` |
| `FASTLOAD_DIR` | `…/glm53-exl3-tp2full` | `…/glm53-exl3-tp2full-fs` (a separate name — the same one would overwrite A's sidecar) |

The launcher's `DRY_RUN` output differs by exactly those items on both nodes. Both environment files
were derived from that node's own predecessor with `sed`; nothing was copied between machines.

---

## 1. Headline

| | **A — experts-only** | **B — full-scope** | B vs A |
|---|---|---|---|
| **KV pool at `max_model_len` 1,000,000** | 1,500,000 (**1.50×**) | **2,128,571** (**2.13×**) | **+41.9 %** |
| **C1 aggregate** (tok/s) | 48.76 | **58.50** | **+20.0 %** |
| **C1 per stream** | 54.72 | **62.55** | **+14.3 %** |
| **C8 aggregate** | 137.41 | **155.75** | **+13.3 %** |
| **TTFT, C1 / C8** | 0.468 / 1.249 s | **0.407 / 1.077 s** | −13.0 % / −13.8 % |
| **Boot, fast-load** | 272 s | 272 s | identical |
| **Correctness / code / tool-call** | 10/10 · 12/12 · 8/8 | 10/10 · 12/12 · 8/8 | equal |
| **MMLU sample** (57 × 35 = 1,995 q, 0-shot) | **86.37 ±0.74** | **86.02 ±0.75** | −0.35 pt, **inside one error bar** |
| **Needle-lite**, 64K + 128K × 3 depths | **6/6** | **6/6** | equal |

---

## 2. The KV pool, from each boot's own arithmetic

`HAREM-TP3 KV pool breakdown`, printed by `patch-kvdiag-tp3.py` on the binding rank:

| group | page | bytes/block | blocks/request, A | blocks/request, B |
|---|---|---|---|---|
| `MLAAttentionSpec`, 22 layers | 152,064 B | 3,345,408 B | 218 | 218 |
| `KpoolTailSpec`, 11 layers | 152,064 B | 1,672,704 B | 1 | 1 |
| `MambaSpec` × 4 (9/9/8/8 layers) | 2,359,296 B | 21,233,664 / 18,874,368 B | 9 + 9 + 9 + 9 | 9 + 9 + 9 + 9 |
| `SlidingWindowSpec` — the drafter, 5 layers | 262,144 B (256 tokens, **fp8**) | 1,310,720 B | 25 | 25 |
| **blocks per request** | | | **280** | **280** |
| **`num_blocks`** | | | **420** | **596** |
| **`GPU KV cache size` at 1M** | | | **1,500,000** | **2,128,571** |

Blocks-per-request is identical, so the whole difference is `num_blocks`, and `num_blocks` is
available memory:

| | A | B |
|---|---|---|
| Available KV cache memory, rank 0 / rank 1 | 11.70 / 11.34 GiB | **16.07 / 16.23 GiB** |
| Consumed memory (weights + non-torch), rank 0 / rank 1 | 89.30 / 89.23 GiB | **84.77 / 84.51 GiB** |
| peak activation | 2.38 / 2.81 GiB | 5.08 / 5.19 GiB |
| attention block size the platform chose | 4,608 tokens | 4,608 tokens |

**The full-scope checkpoint is 4.5–4.7 GiB lighter per node.** That settles a contradiction this
repository carried since 5 September: the TP=2 dress rehearsal implied it was ~10 GiB *heavier*
([docs/13](../../docs/13-full-scope-checkpoint.md) §6.2). It is not, and the earlier reading came
from a boot that had no settle gate.

### 2.1 What the fp8 draft cache contributed, isolated by arithmetic

`HAREM_DRAFT_KV_DTYPE=fp8` changes exactly one number — `SlidingWindowSpec` bytes/block — and
nothing else in the decomposition, so its effect on the pool is exact even though it was not run as
its own A/B arm:

| | draft cache `auto` (6 Sep KV-fix arm) | draft cache **fp8** (candidate A) |
|---|---|---|
| `SlidingWindowSpec` bytes/block | 2,621,440 B | **1,310,720 B** |
| blocks per request | 280 | 280 |
| `num_blocks` | 365 | **420** |
| **KV pool** | 1,303,571 | **1,500,000** (**+15.1 %**) |

At three ranks the same knob bought +5.6 %. The two arms are one image apart (`62f53e6` →
`754421f`), so the pool attribution is exact arithmetic but a speed comparison across them is not,
and none is made.

**CUDA graphs still capture at two ranks with the fp8 draft cache** — 19 PIECEWISE and 8 FULL in
both candidates' boot logs. At three ranks the same knob pushed the drafter onto a FlashInfer path
and graph capture stopped.

---

## 3. Speed, four rounds each

Aggregate output tok/s. Round 1 is listed to show the tuner cache is warm; the result is the median
of rounds 2–4.

### 3.1 Candidate A — experts-only

| C | r1 | r2 | r3 | r4 | **median 2–4** | per stream | TTFT med | accept % | tok/step |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 47.25 | 48.13 | 49.17 | 48.76 | **48.76** | 54.72 | 0.468 | 65.47 | 5.58 |
| 2 | 69.11 | 69.55 | 68.24 | 67.34 | **68.24** | 40.09 | 0.670 | 62.72 | 5.39 |
| 4 | 95.77 | 90.91 | 97.53 | 96.52 | **96.52** | 29.00 | 0.986 | 63.60 | 5.45 |
| 6 | 113.64 | 115.69 | 113.08 | 118.04 | **115.69** | 23.21 | 1.076 | 60.92 | 5.26 |
| 8 | 135.54 | 137.41 | 135.79 | 137.58 | **137.41** | 20.90 | 1.249 | 62.12 | 5.35 |

### 3.2 Candidate B — full-scope

| C | r1 | r2 | r3 | r4 | **median 2–4** | per stream | TTFT med | accept % | tok/step |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 60.14 | 58.36 | 59.51 | 58.50 | **58.50** | 62.55 | 0.407 | 59.98 | 5.20 |
| 2 | 81.16 | 82.45 | 81.78 | 83.69 | **82.45** | 47.67 | 0.570 | 62.76 | 5.39 |
| 4 | 116.55 | 111.32 | 114.82 | 112.62 | **112.62** | 33.49 | 0.766 | 63.67 | 5.46 |
| 6 | 140.12 | 137.29 | 140.17 | 137.37 | **137.37** | 26.01 | 1.010 | 62.50 | 5.37 |
| 8 | 152.63 | 151.14 | 155.81 | 155.75 | **155.75** | 22.17 | 1.077 | 62.84 | 5.40 |

### 3.3 Round 1 against the median — the tuner-cache check

| | A, C1 | A, C8 | B, C1 | B, C8 |
|---|---|---|---|---|
| round 1 error vs median of 2–4 | −3.1 % | −1.4 % | **+2.8 %** | −2.0 % |

Unordered and inside ±3 %, which is what [docs/12](../../docs/12-tuner-cache.md) §4 measured at three
ranks. Three tune events on the whole boot. **Three rounds is enough at two ranks as well.**

### 3.4 Cold first request and prefill

| | A | B |
|---|---|---|
| cold C1 — TTFT · decode · acceptance | 0.78 s · 37.7 tok/s · 41.1 % | 0.76 s · **50.8** tok/s · 42.8 % |
| warm C1 (same probe, 2nd/3rd) | 0.49 / 0.48 s · 40.1 / 37.6 tok/s | 0.43 / 0.41 s · 50.6 / 49.8 tok/s |
| prefill, 7,382 tokens, uncached | 1,241 tok/s | **1,289** tok/s |
| prefill, 3 fresh unseen ~8.4K prompts | **1,444** tok/s (1,439 / 1,444 / 1,456) | 1,400 tok/s (1,363 / 1,400 / 1,401) |

Prefill is equal inside the ±3 % band; the cold-request row is not — 35 % of it is the dense stage.

---

## 4. Boot

| | A | B |
|---|---|---|
| **fast-load boot**, both ranks started to `/health` 200 | **272 s** | **272 s** |
| one-off **dump** boot that writes the sidecar | 997 s | 998 s |
| main weight restore | 90.1 s, 79.87 GiB, 952 MB/s, 3,657 tensors | 88.0 s, 75.17 GiB, 918 MB/s, 5,774 tensors |
| drafter restore | 3.05 s, 2.71 GiB | 3.47 s, 2.71 GiB |
| sidecar verify (`FASTLOAD_VERIFY=32`) | OK, 32/1,433 re-hashed | OK, 32/2,076 re-hashed |
| **sidecar per rank** | 82.59 GiB, 36 files | **75.17 GiB**, 32 files |
| `init engine (profile, create kv cache, warmup)` | 87.2 s | 88.6 s |

For comparison, a two-node cold boot **without** a sidecar is 355–396 s (5–6 September arms), and the
three-node production fast-load boot is 251 s with a 53 GB sidecar per rank.

**Disk.** A two-node sidecar is about **half** the checkpoint per rank, because EP is off and every
rank owns half of every tensor; at three ranks with EP it is a third. Check `df -h /var/tmp` before
the first dump: 154–164 GB of checkpoint plus a 75–83 GB sidecar needs ~240 GB free per node.

---

## 5. The fabric — both cables of the pair

`port_xmit_data` deltas across the four-round C1–C8 sweep, converted to MB
([docs/06](../../docs/06-nccl-mesh.md) §6 is the method):

| node | device | A | B |
|---|---|---|---|
| head | `rocep1s0f0` | 92,299 MB | 91,736 MB |
| head | `roceP2p1s0f0` | 90,180 MB | 89,544 MB |
| head | `rocep1s0f1` · `roceP2p1s0f1` | 0.0 · 0.0 | 0.0 · 0.0 |
| worker-1 | `rocep1s0f1` | 92,061 MB | 91,495 MB |
| worker-1 | `roceP2p1s0f1` | 89,944 MB | 89,354 MB |
| worker-1 | `rocep1s0f0` · `roceP2p1s0f0` | 0.0 · 0.0 | 0.0 · 0.0 |

**Two devices per node, ~90 GB each, split 50.5 / 49.5.** The dual-cable patch
(`patches/kernel/0005`) works on a single peer pair, which had never been shown. The zero rows are
the ports that face the third node; it is not in this cluster.

---

## 6. Quality

| gate | A | B |
|---|---|---|
| correctness probe, **cold** | 10/10 | 10/10 |
| correctness probe, **warm** (after the whole benchmark) | 10/10 | 10/10 |
| code exam, **cold** | 12/12 | 12/12 |
| code exam, **warm** | 12/12 | 12/12 |
| tool-call gate (8 checks) | **8/8** | **8/8** |
| MMLU sample, 57 subtasks × 35, 0-shot, `num_concurrent 8` | **86.37 ±0.74** | **86.02 ±0.75** |
| needle-lite, 64K and 128K × depths 10/50/90 % | **6/6** | **6/6** |

Needle-lite serves 80,113-token prompts at both candidates — the long-prompt path that the draft page
fix opened (see [`tp2-draft-page.md`](tp2-draft-page.md)) is open at full length here, not merely at
8K.

MMLU reference points on this stack: TP=2 arm A 86.4 ±0.7 (4 Sep), TP=2 arm D full-scope 86.32 ±0.75
(5 Sep), **TP=3 production 9/10 86.47 ±0.74** (5 Sep). Every one of these sits inside a single error
bar of the others.

---

## 7. Host memory

| after the full benchmark | A | B |
|---|---|---|
| `MemAvailable`, head / worker-1 | 6.0 / 7.2 GiB | 5.7 / 7.0 GiB |
| swap in use, head / worker-1 | 0.03 / 0.02 GiB | 0.03 / 0.02 GiB |

**Swap is flat in both**, which is the number the three-node memory rung was actually decided on
([docs/11](../../docs/11-open-issues.md) §2.4). It is not a licence to move the rung at two ranks:
`gpu-memory-utilization` stayed at 0.85 in every arm on this page and the ladder has never been
derived at two nodes — see [docs/15](../../docs/15-tp2-track.md) §6.

---

## 8. The autostart unit

`systemd/harem-exl3-tp2.service` + `systemd/motor-onkosul-exl3-tp2.sh`, installed on both nodes with
`@USER@`/`@HOME@` substituted per node (never copied between them), `daemon-reload`, left
**`disabled`**:

| step | result |
|---|---|
| `systemctl start harem-exl3-tp2`, worker-1 then head | returned in **3 s / 6 s** — the preflight itself takes 1 s |
| preflight line | `preflight ok: 1 s, ConnectX-7 4/4, peer: <one address>` then `exl3 tp2 preflight ok: env=… image=exl3-zeus:754421f sidecar=…-fs-r0` |
| **`/health` 200** | **+261 s** from the first `systemctl start` |
| unit / container state while up | `active` / `running` on both nodes |
| KV pool on that boot | **2,153,571** — +1.2 % against the 2,128,571 of the hand-started boot of the same env, which is the settle gate agreeing with itself |
| correctness probe · code exam | **10/10** · **12/12** |
| `systemctl stop harem-exl3-tp2` | clean on both; units `inactive`, no containers left |
| state afterwards | `harem-exl3-tp2` **disabled**, `harem-exl3` (three nodes) still **enabled** |

**The first attempt failed, and it failed correctly** `[measured-here]`. The unit had been installed
before its `ENV_FILE` was pointed at the full-scope environment file, so the preflight looked for the
*other* candidate's sidecar — which had been deleted to make disk room — and refused in one second
with `fast-load sidecar missing: …-r0`, before docker was touched. That is check 7 of
[`motor-onkosul-exl3-tp2.sh`](../../systemd/motor-onkosul-exl3-tp2.sh) doing exactly the job it
exists for: the alternative is a 620-second boot, or the fast-load refusal four minutes in
([docs/14](../../docs/14-troubleshooting.md) §3.1).

**No reboot test** `[not tested]`. The three-node unit remains the enabled autostart on this cluster,
and the two units carry `Conflicts=` for each other so only one can ever run.

---

## 9. Raw material

Logs, sweep JSONs, gate outputs, container logs, `DRY_RUN` command lines, port counters and the KV
ledgers for both candidates live under our own measurement directory, not in this repository:
`prod/` (candidate A), `prod-fs/` (candidate B), `dump/` and `dump-fs/` (the sidecar boots),
`dry-tp2prod*.txt`, `ab-tp2full.sh`, `dump-tp2full.sh`, `summary.tsv`.

# Audit — check your install against ours, and check ours against itself

Two audiences, one folder.

**If you have built the stack:** run `audit/run-audit.sh` and compare what you get against the
expected ranges on this page. The script prints numbers; it does not grade them. Grading is this
page.

**If you are reviewing what we published:** §5 is the provenance table — every headline number, the
date it was measured, the configuration it was measured on, and the file it traces to. §6 is the
retraction index: everything we published and then withdrew, with what replaced it. A reviewer who
reads nothing else should read §6.

```
audit/run-audit.sh
```

About 25 minutes end to end; the category-speed step is most of it. One section at a time:

```
audit/run-audit.sh health kv gates c1 prefill memory fabric
```

Sections: `health` `kv` `gates` `c1` `prefill` `category` `memory` `fabric` `versions`.

---

## 1. Settings every number below assumes

Change any row and the expected values change with it.

| | |
|---|---|
| Image | `exl3-zeus:754421f` (`cuda-exl3` `754421f`, carrying `f3e3090`) |
| Checkpoint | `turboderp/GLM-5.3-Flash-exl3`, branch `4.05bpw`, revision `2a30229e` — **full scope** |
| Parallelism | TP=3, expert parallelism on, three nodes |
| KV | `kv-cache-dtype fp8`, **and an fp8 draft cache** (`HAREM_DRAFT_KV_DTYPE=fp8`), `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256` |
| Speculative | DFlash2, k=7 |
| Memory | `--gpu-memory-utilization 0.80` (production 9) or **0.83** (production 10), `SETTLE_MIN_GIB=112` |
| Batching | `--max-num-seqs 8`, `--max-num-batched-tokens 2048` |
| Fabric | mesh plugin `19924dcc` + patches `0004`/`0005`/`0006`, `NCCL_MAX_NCHANNELS=8`, `NCCL_MESH_LINKS_PER_PEER=0`, `NCCL_MESH_PTR_CUDA=1` |
| Boot | per-rank fast-load sidecar, warm `CUDA_EXL3_TUNE_CACHE` |
| Sampling | temperature 0, thinking on, reasoning effort **low** |
| Date | 5 September 2026 |

**This is production configuration 9, and production 10 is the same thing with one line changed** —
`GPU_MEMORY_UTILIZATION` 0.80 → 0.83, which buys **+8.7 % of KV pool** and moves no speed number
outside its noise band. Every expectation below is production 9's unless a row says otherwise.
Anything you publish from this script must carry that table with it. A tok/s figure without its
configuration is not a measurement.

**Production 11 — 0.87 plus the sm_12x correctness set — is what our three nodes serve today**, and
its numbers are in §5 rather than in §2: the expectations below were not re-derived on it. The three
configurations differ by a memory fraction and two guard patches, so §2 still reads correctly against
a production 11 engine everywhere except the KV pool, which is larger.

**Everything here is at reasoning effort `low`.** Max effort would cost 5–12× the tokens and days of
cluster time. Nothing on this page is a max-effort number and none of it should be quoted as one.

---

## 2. What to expect

Work down this list in order. Each step rules out the ones after it, and there is a reason for that
ordering: **a fast engine that answers wrong is worthless, and a machine that is swapping produces
speed numbers that mean nothing.**

### versions — before anything else

The audit's first section prints your environment against §5 of
[docs/00](../docs/00-hardware-and-os.md). Nothing here is graded automatically; you are looking for
the two that bite:

- `ibv_devinfo | grep -c PORT_ACTIVE` must be **4** on every node.
- `lspci -vv` must report `Speed 32GT/s, Width x4` for each ConnectX endpoint. If yours reads `x2`,
  your SBIOS is below `0104` and **you are measuring a different machine from ours** — fix that
  before you read another number ([docs/00](../docs/00-hardware-and-os.md) §2.1).

### health

`GET /health` returns 200 within a few milliseconds once the engine is serving. **Only the head
(rank 0) serves the API**; the workers have no HTTP endpoint at all, and a `curl` against a worker
timing out is correct behaviour, not a fault.

### fabric — the check almost everyone skips `[measured-here]`

`PORT_ACTIVE` is link state, not traffic. Read the transmit counters as a **delta** across a
benchmark:

```
for p in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do echo "$p $(cat $p)"; done
```

| | Expected |
|---|---|
| Ports whose counter moves | **4 of 4** per node |
| Ports flat at a value that never changes | **0** |

Two flat ports per node is the pre-patch condition, and it cost us a factor of two on the ceiling.
It is what patch `0005` fixes. See [docs/06](../docs/06-nccl-mesh.md) §6.

### KV pool — `[measured-here]`

Read from the engine's own start-up log line `GPU KV cache size: N tokens`, on a **load** boot with
a settled memory baseline.

| Configuration | `gpu-memory-utilization` | KV pool (tokens) |
|---|---|---|
| Production 7 (routed-experts-only, fp8 draft cache) | 0.80 | 4,696,969 |
| Production 8 (image `62f53e6`) | 0.80 | 4,696,969 |
| **Production 9 (full-scope checkpoint)** | **0.80** | **5,165,289** (5,168,044 on the verification boot) |
| **Production 10 (production 9 + one line)** | **0.83** | **5,619,834** |
| 0.85, **rejected** on the free-memory rule | 0.85 | 5,256,198 |

About **5.2 concurrent 1M-token requests** at production 9, **5.6** at production 10.

**Why 0.83 is in production and 0.85 is not**, given that 0.85's pool is only measured against
production 3's checkpoint: at 0.85 the head node had 1.9 GiB free and **1.6 GB of swap in use**. At
0.83, on the current stack, `MemAvailable` is 8–10 GB per node and **swap stays flat at ~0.1 GB
through a full sweep** `[measured-here]`. `MemFree` does dip to 0.9–1.2 GiB, which is below the
headline rule, but that memory is reclaimable page cache and the number that matters — swap growth —
did not move. **"0.85 will not be attempted on this stack" is `[retracted]`** (§6): the ladder was
climbed on 6 September against swap *traffic*, 0.85 and 0.88 both pass, 0.90 is rejected, and **0.87
is production configuration 11** at a **6,382,920**-token pool
([`results/memory/ladder-6sep.md`](../results/memory/ladder-6sep.md)).

A pool within a few percent of ours is a match. A pool well below it usually means one of: no fp8
draft cache (`HAREM_DRAFT_KV_DTYPE`), `--block-size` left at a large value,
`--max-num-batched-tokens` raised, or — most likely — **a boot taken without the memory settle
gate**. That last one is not a real difference in your machine; it is the instrument. See
[docs/07](../docs/07-kv-and-draft-page.md) §1.1.

**Read the pool from a boot whose baseline had settled.** vLLM sizes the pool from a difference
between two `/proc/meminfo` readings minutes apart, and it runs backwards: a node that starts with
less memory free awards itself a *larger* pool. Ours was polluted by 9 GiB on the last node started
until `SETTLE_MIN_GIB` was added.

### gates — expect 10/10 and 12/12, cold **and** warm `[measured-here]`

```
python3 ../scripts/correctness-probe.py $API
python3 ../scripts/code-exam.py $API
```

| | Expected |
|---|---|
| correctness probe | **10/10**, `requests with EMPTY content: 0` |
| code exam | **12/12** |

**Run both twice: once cold after the boot, and again after the full benchmark.** The warm run is the
one that carries weight. The class of defect this stack has actually produced — a kernel writing rows
nothing initialises, and a combine summing them — **hides on a fresh engine**, because a fresh
caching allocator hands out zeroed pages ([docs/09](../docs/09-measurement-protocol.md) §5).

This gate is not cosmetic. A decode kernel that silently computes the wrong thing for our head count
produced fluent, confident nonsense here while the engine reported no error at all, and arms with a
mis-shaped kernel scored 0/12 to 10/12 on the code exam while still reading fluently. **If you score
below full marks on a warm engine, stop and find out why before you measure anything else.**

Note the code exam executes model-written Python locally, in a temp file, with a 25-second timeout
and no sandbox.

### boot gates — read these before the numbers

A production boot prints four things that tell you the engine built the shape you asked for. If any
is missing, nothing measured afterwards means anything:

| Gate | Expected |
|---|---|
| `[padload]` line | present |
| Patch anchors | **10** matched |
| Pad audit | `assert 5` passes |
| `CUDA_EXL3_DEBUG_NAMES` tally | **203 EXL3 / 113 bf16**, nothing heavy in the "→ unquantized" list |

### c1 — cold/warm single stream `[measured-here]`

| | Expected (production 9) |
|---|---|
| C1 aggregate | **69.9** tok/s |
| C1 per stream | **75.9** tok/s |
| TTFT at C1 | **0.280** s |
| Draft acceptance | **~62 %** — the gate is ≥ 60 % |
| Accepted tokens per step | **~5.34** |

**A note on acceptance, because we got this wrong on our own front page.** The C1 median reads
61.9 % on production 9 against 64.4 % on production 8, and we published that 2.4-point gap as the
price of the full-scope checkpoint. **It is an artefact of our own harness** `[retracted]`:
`bench-sweep.py` cycles `prompts[i % 12]`, so C1 and C2 see only the first **8** of the 12 prompts
while C4–C8 see all twelve, and the two groups differ by 8 points of acceptance. Pooled by draft
token across all five levels and three independent boots, production 9 is **62.27 %** against
production 8's **62.09 %** — **+0.18 points**, well inside that arm's own ±1.4-point boot spread. A
700-token cold probe reads identically on both arms (42.53 % against 42.51 %).

Two things follow for anyone comparing acceptance numbers. **Pool by draft token across levels**, not
by taking a per-level median. And note that `accept_len = 1 + k × acceptance` holds on all 90 of our
rows to ±0.005 — so "acceptance −2.4 points" and "tokens per step −3 %" were never two costs, they
are **the same number written twice**.

The first of the three requests is genuinely cold if you run this right after a boot; it will be
lower, and that is the point of running it first. If you run the audit against an engine that has
been serving for a while, the "cold" line is not cold.

**Do not compare any of this against a synthetic "count to 200" number.** On synthetic prompts the
drafter accepts almost everything and the same engine runs far faster. That is the ceiling, not the
working speed.

### concurrency — `[measured-here]`

| | Production 9 | Production 8 |
|---|---|---|
| C1 total / per stream | **69.9 / 75.9** | 56.9 / 62.4 |
| C8 total / per stream | **197.2 / 28.6** | 175.4 / 26.7 |
| TTFT C8 | **0.826** s | 0.906 s |

Median of **three** sweep rounds — and three is only enough **because the MLA tuner cache is
persisted and warm**. On an image without one the rule is still five rounds with the first two
discarded: this stack's tuner warm-up has made a winning arm look 25–45 % worse on the first pass
([docs/12](../docs/12-tuner-cache.md), [docs/09](../docs/09-measurement-protocol.md)).

Boot-to-boot spread on this stack is up to **16 % at C8**. A single pair of rounds is not evidence
for anything; we published a kernel conclusion drawn from one pair and had to withdraw it (§6).

### prefill — measure it on a prompt the engine has never seen `[measured-here]`

| | Expected |
|---|---|
| Fresh, unseen ~8.5K prompts | **1,738** tok/s |
| Warm, repeated 7K prompt | 1,575 tok/s |

**A prefill number measured on a repeated prompt is not a prefill measurement.** The second run reads
whole blocks out of the prefix cache and overstates by up to 55 %. `scripts/prefill-7k.py` reports
the warm number and says so; `bench/prefill-fresh.py` draws a new seed per request and is the one to
quote.

Both production 8 and production 9 sit inside the ±3 % equality band on prefill: the full-scope
checkpoint bought its gain in **decode step time**, not in prefill.

### category speed — `[measured-here]` on production 9

| Category | C1 mean decode | C1 acceptance | C4 total |
|---|---|---|---|
| code | **61.7** tok/s | 46 % | 116.1 tok/s |
| math | **79.6** tok/s | 58 % | 129.4 tok/s |
| JSON | **72.8** tok/s | 54 % | 110.1 tok/s |
| **prose** | **29.1** tok/s | **13 %** | 50.7 tok/s |

The production-7 lineage read code 47.9 · math 59.0 · JSON 57.7 · prose 22.4, so every category
gained 30–35 %.

**The prose row is the honest headline of this whole stack**, and it is not a setting: the DFlash2
drafter barely fires on free prose, so prose falls back to roughly the unspeculated rate while code,
math and JSON get 2–2.8×. Acceptance on **non-English** prose is worse still, 10–13 %. That is a
property of the drafter's training distribution, and nothing on our side fixes it. Every speed table
in this repository was measured with **English** prompts; read them as English-workload numbers.

Note that this run came from a standalone category script, **not** from the sweep record — the
quick-arm harness the production 8 and 9 comparisons ran does not include a category step at all
([docs/14](../docs/14-troubleshooting.md) §8.12).

### memory — `[measured-here]`

At `gpu-memory-utilization 0.80`, engine idle:

| | head | worker-1 | worker-2 |
|---|---|---|---|
| Free | **12.1 GiB** | **13.5 GiB** | **13.4 GiB** |
| Swap used | ~0.1 GiB | ~0.1 GiB | ~0.1 GiB |

At `gpu-memory-utilization 0.83` (**production 10**), after a full benchmark:

| | head | worker-1 | worker-2 |
|---|---|---|---|
| `MemAvailable` | **8.8 GB** | **10.0 GB** | **10.1 GB** |
| `MemFree` | 0.9–1.2 GiB, reclaimable page cache | | |
| Swap used | ~0.11 GiB | ~0.09 GiB | ~0.08 GiB, **flat through the rounds** |

**The rule, and how to apply it at 0.83.** On a GB10 the GPU shares host memory, so free host RAM
*is* your safety margin, and this repository has used **4 GiB free** as the audit floor and 2 GiB as
the hard one. At 0.83 `MemFree` is below both — and we took the rung anyway, because the number that
actually decides it is **swap growth**, and it did not move. 0.85 was rejected on 1.6 GB of swap
appearing under load on the head node; 0.83 stays at ~0.1 GB and flat, with `MemAvailable` at
8–10 GB. **So: watch swap, not `MemFree`.** If your swap grows during the rounds, step
`gpu-memory-utilization` down and re-audit; if `MemFree` is low but swap is flat and `MemAvailable` is
in the gigabytes, you are where we are. **And read swap *traffic*, not swap used** — the stock sits
at ~0.04 GiB at every rung including the one that failed, so it discriminates nothing. That is the
ruler the 6 September ladder was climbed with, and on it 0.85, 0.87 and 0.88 all pass while 0.90 is
rejected; **"0.85 will not be attempted on this stack" is `[retracted]`** (§6,
[docs/11](../docs/11-open-issues.md) §2.4,
[`results/memory/ladder-6sep.md`](../results/memory/ladder-6sep.md)).

Consumed memory per node (weights plus non-torch) should be **58.3–59.1 GiB**.

`vm.swappiness` should read **60**. Do not set it to 0 — see
[docs/00](../docs/00-hardware-and-os.md) §9.1 for the incident that cost three power cycles.

### boot time — `[measured-here]`

| | Expected |
|---|---|
| Container start → API ready, fast-load boot | **251 s** (weights 58 s) |
| The one-off dump boot that produces the sidecar | **620 s** |
| Without a fast-load sidecar at all | ~618 s |

---

## 3. If the audit disagrees with this page

In this order, because each step rules out the ones after it:

1. **Versions and fabric.** `x2` instead of `x4`, or two flat ports per node, means you are measuring
   a different machine. Nothing below matters until those are right.
2. **Correctness.** If the probe or the code exam fails, compare your image and patch tree against
   `patches/` — every failure we have seen here traced to a missing patch, not to a setting.
3. **The KV pool**, from the start-up log. It tells you whether the engine built the shape you asked
   for, before any timing is involved. Check the settle gate before you believe a small pool.
4. **Memory.** A machine that is swapping produces speed numbers that mean nothing.
5. **Speed**, and only against the matching row, with the matching concurrency, the matching prompt
   category, and the matching number of sweep rounds.

If it is a failure rather than a disagreement, [docs/14](../docs/14-troubleshooting.md) indexes them
by symptom with the exact log line.

---

## 4. What this audit does **not** check

Named so you do not mistake silence for a pass:

- **Quality beyond the two gates.** The MMLU sample (1,995 questions) takes hours and is not in the
  script. Ours reads **86.47 ±0.74** at TP=3 on the production checkpoint.
- **Long context.** `max_model_len` is 1,000,000 and the pool supports about 5.2 such requests, but
  the audit sends nothing near that.
- **Anything at max reasoning effort.** See §1.
- **Numerical equivalence against BF16.** We never ran it.

---

## 5. Provenance — every headline number, and where it comes from

| Claim | Value | Config | Date | Traces to |
|---|---|---|---|---|
| C1 total / per stream | 69.9 / 75.9 tok/s | prod 9 | 5 Sep 2026 | `results/speed/concurrency-sweeps.csv` |
| C8 total / per stream | 197.2 / 28.6 tok/s | prod 9 | 5 Sep 2026 | `results/speed/concurrency-sweeps.csv` |
| TTFT C1 / C8 | 0.280 / 0.826 s | prod 9 | 5 Sep 2026 | `results/speed/concurrency-sweeps.csv` |
| Draft acceptance · accepted/step | 61.9 % · 5.34 | prod 9 | 5 Sep 2026 | `results/speed/concurrency-sweeps.csv` |
| Prefill, fresh | 1,738 tok/s | prod 9 | 5 Sep 2026 | `results/speed/category-prefill-and-mixed-load.md` |
| KV pool | 5,165,289 tokens | prod 9 @ 0.80 | 5 Sep 2026 | `results/boot/boot-ledger.md`, engine log |
| Consumed memory per node | 58.3–59.1 GiB | prod 9 | 5 Sep 2026 | `results/boot/boot-ledger.md` |
| Quality gates | 10/10 · 12/12 cold and warm | prod 9 | 5 Sep 2026 | `results/gates/quality-gates.md` |
| MMLU sample, 1,995 q | 86.47 ±0.74 | prod 9, TP=3 | 5 Sep 2026 | `results/gates/quality-gates.md` |
| Cold boot | 251 s (weights 58 s) | prod 9, fast-load | 5 Sep 2026 | `results/boot/boot-ledger.md` |
| Step breakdown, prefill / C1 / C8 | see `charts/step-breakdown-prod9.svg` | **prod 9** | 5 Sep 2026 | `results/profile/step-breakdown.csv` |
| Step breakdown, the production-7 control | see `docs/10` §5 | prod 7 | 5 Sep 2026 | `results/profile/measured-prod7.md` |
| Production 10 (0.83 rung), boot `L083` | KV 5,619,834; C1 70.5 / C4 144.6 / C8 194.0; prefill fresh 1,769; TTFT 0.282 / 0.811 s | prod 10 | 5 Sep 2026 | `results/configs/kv-pool-progression.csv`, `results/speed/concurrency-sweeps.csv` |
| Production 10, host memory under load | `MemAvailable` 8.8 / 10.0 / 10.1 GB; swap ~0.1 GB flat | prod 10 | 5 Sep 2026 | `results/configs/kv-pool-progression.csv` |
| `NCCL_ALGO=Ring,Tree`, five-round engine arm, boot `ALG5` | C1 70.6 / C4 143.4 / C8 195.6, all within ±1 % of `Ring`; KV 5,702,479 (+1.5 %) | prod 10 | 5 Sep 2026 | `results/mesh/algo-sweep.md`, `docs/06` §12.3 |
| Boot from power-on, all three rebooted, autostart enabled | units finished +98/+98/+103 s; `/health` 200 at +242 s by the harness, **+315 s** by the log's wall clock; KV 5,652,892; gates 10/10 · 12/12 | prod 10 | 5 Sep 2026 | `results/boot/boot-ledger.md`, `systemd/README.md` |
| **Production 11** — configuration 10 at 0.87 **plus** the sm_12x correctness set (`HAREM_SM12_ITEMS=pdl,kpool`) | KV **6,382,920**; C1 **69.6 / 74.7**; C8 **201.1 / 29.5**; prefill fresh 1,760; TTFT 0.278 / 0.806 s; acceptance 61.3 % · 5.29; boot 271 s (`docker run` → API ready), 590 s for the dump boot the patches force. Against a same-session 0.83 reference: KV **+12.1 %**, C1 +0.6 %, C8 +1.8 %, C4 −2.2 % | prod 11 @ 0.87 | 6 Sep 2026 | `results/configs/production-configurations.csv` row 11, `results/speed/concurrency-sweeps.csv` |
| Production 11, host memory and swap **traffic** under load | `MemAvailable` min **3.40 / 4.66 / 4.68 GiB** against 8.28 / 9.05 / 9.57 at 0.83; swap traffic `si`+`so` **exactly 0** on all three nodes on the clean boot, and 0.08 / 0 / 0.02 MiB on the load boot against the reference's own 0.7 / 0.9 / 0.4 MiB floor | prod 11 | 6 Sep 2026 | `results/configs/production-configurations.csv` row 11, `results/memory/ladder-6sep.md` |
| Production 11, quality and the whole-cluster reboot | Gates **10/10 · 12/12 cold and warm on both boots**, tool-call **8/8**, needle-lite **6/6** at 64K and 128K; `/health` 200 at **312 s** from the reboot command by wall clock against production 10's 315 s by the same clock, ConnectX-7 4/4, KV 6,382,920, swap traffic zero. MMLU is production 9's 86.47 ±0.74, carried forward and `[not tested]` here | prod 11 | 6 Sep 2026 | `results/configs/production-configurations.csv` row 11, `results/boot/boot-ledger.md` |
| The memory ladder, four rungs in one session against a same-session 0.83 reference | KV pool 0.83 **5,674,931** · 0.85 **6,016,528** (+6.0 %) · **0.87 6,363,636** (+12.1 %, **shipped**) · 0.88 **6,542,699** (+15.3 %, passed and not taken) · 0.90 **6,870,523** (+21.1 %, **rejected**). No speed number at any rung is outside its band — including the rejected one; gates full at every rung; OOM killer 0 everywhere | one line per rung, same tree | 6 Sep 2026 | `results/memory/ladder-6sep.md` |
| Why 0.90 was rejected, and what a rung costs | Head node **1,519.4 MiB paged out and 142.6 MiB read back**, 250 of 598 load seconds non-zero, longest unbroken run 85 s, swap in use at the end 2.65 / 0.57 / 0.52 GiB against 0.03 at the reference. What surfaces at the client is the arm's **first prefill going 5.0 → 9.8 s**, not a lower median. `MemAvailable` min on the head node across the five rungs: 8.35 / 5.99 / **3.49** / 1.86 / 1.04 GiB — every 1 % of the fraction costs about **1.2 GiB** of host headroom | one line per rung, same tree | 6 Sep 2026 | `results/memory/ladder-6sep.md` §3–§4 |
| The per-node memory ledger, read with the engine left running | Engine CUDA allocation **99.06 GiB** = weights 51.62 + KV pool 40.12 + non-torch 7.28 + CUDA graph **0.00**; host anonymous 5.60; page cache 5.80; slab 2.36; free 5.08; residual **3.49**, which reads **3.12** with the engine down and is therefore the driver's fixed reserve. Genuinely unavailable at idle: **4.84 GiB** of 121.63 | prod 10 @ 0.83, read-only, engine up | 6 Sep 2026 | `docs/17-memory-ledger.md` §2 |
| What a KV block is, and the KDA speculation slots inside the divisor | Block = 3,328 tokens, **20,934,400 B**: MLA latent fp8 **89.53 %**, indexer k fp8 5.77 %, DFlash2 draft 4.70 %. A 1M-token request costs **363 blocks = 7.078 GiB on every node** and the pool holds **5.67** of them. Under `mamba_cache_mode=align` every KDA layer holds **2 + 7 = 9** state slots per request — **36 blocks, 9.9 % of the divisor at TP=3 and 12.9 % at TP=2**, 5.61 GiB (14 % of the pool) at 8-way concurrency. Taking the slots to two projects **+8.0 %** (TP=3) and **+9.6 %** (TP=2) `[estimate]`, against an arithmetic worst case of C8 197 → about 168 tok/s; **not run** `[not tested]` | prod 10 @ 0.83 | 6 Sep 2026 | `docs/17-memory-ledger.md` §3–§5 |
| **Production 12 candidate** — the sparse indexer's K-gather workspace, bounded | Locked workspace **5,036.40 → 513.00 MB** (−4.42 GiB); KV pool **6,289,256 → 6,933,884, +10.25 %** at an unchanged memory fraction. C1 69.69 → 70.69, C8 199.76 → 196.81, prefill fresh 1,794 → 1,778, TTFT equal — every level inside its band. Gates 10/10 · 12/12 cold and warm on both arms, tool-call 8/8, needle-lite 6/6, one **969,468-token** request correct in 572.4 s, **eight** concurrent long-context lanes (640,904 prompt tokens) with every needle correct, swap 0.000 GiB, no safety layer fired. **Not in production** — `results/configs/production-configurations.csv` carries no row 12 as of this commit, and the promotion waits on disk rather than on doubt; both arms are eager boots, so the production pool with fast-load restored is an `[estimate]` until a promoted boot prints it | prod 11's 0.87, fast-load off, one line between the arms | 6 Sep 2026 | `results/memory/indexer-workspace-ab.md`, `docs/17` §2.5 |
| **TP=2 production candidate B** (full scope), against candidate A (routed experts only) | KV pool at 1M **2,128,571** against 1,500,000 (**+41.9 %**); C1 **58.50 / 62.55** against 48.76 / 54.72; C8 **155.75** against 137.41; TTFT 0.407 / 1.077 s; consumed per node **84.8 GiB** against 89.3 (**−4.5**); prefill fresh 1,400 against 1,444 (equal); boot 272 s on both; gates 10/10 · 12/12 cold and warm, tool-call 8/8, needle-lite 6/6 on both; MMLU **86.02 ±0.75** against 86.37 ±0.74, inside one error bar | **two** nodes, TP=2, EP off, `exl3-zeus:754421f`, 0.85, median of rounds 2–4 of four | 6 Sep 2026 | `results/speed/tp2-production-candidate.md`, `docs/15` §5 |
| The draft KV page at two ranks (`HAREM_SW_BLOCK_SIZE=256`) | KV pool **601,562 → 1,303,571, +116.7 %** (+109.6 % normalised); C8 127.54 → 135.59 (+6.3 %); TTFT −23 % at C1 and −27 % at C8; C1 aggregate equal; gates full on both arms. **The headline is the cliff, not the pool**: without it a 6,253-token prompt is never scheduled at all (`Running: 0, Waiting: 1`), with it 8,268 tokens serve in 6.3 s at 1,478 tok/s fresh prefill. The drafter takes **60.2 %** of the blocks-per-request divisor at TP=2 against 53 % at TP=3. Cost: +9.1 % memory per block, acceptance −1.9 points | **two** nodes, TP=2, EP off, `exl3-zeus:62f53e6`, experts-only `b20c49ba`, 0.85, three rounds, one boot each | 6 Sep 2026 | `results/speed/tp2-draft-page.md`, `docs/15` §4 |
| KDA/MLA quantization gate bench, **cold, on the target GPU** — **this is the live row** | KDA arms EXL3/bf16 **1.023** at M=8 (neutral, +0.050 ms/step); `kv_b_proj` **0.291×**; three families together **+1.368 ms/step** against a 1.5 ms gate; prefill not re-measured | model-free, **GB10**, both arms rotated over ≥ 4× L2 | 5 Sep 2026 | `results/kernels/kda-gate-bench-gb10.md`, `results/kernels/gb10-coldbench/`, `bench/kda_gate_bench_gb10.py` |
| KDA/MLA quantization gate bench, first pass | KDA arms EXL3/bf16 1.58–1.76× at M=8; whole family 0.851 ms of a 72.5 ms step; `kv_b_proj` 0.391× but no batched kernel exists — **the ratios are `[retracted]`, warm; the file is kept as published** | model-free, **workstation GPU** | 5 Sep 2026 | `results/kernels/kda-gate-bench.md`, `bench/kda_gate_bench.py`, `docs/11` §1.11 |
| Device read bandwidth ruler | 225.2 GB/s | model-free | 5 Sep 2026 | `bench/bw.py`, `results/profile/step-breakdown.md` |
| BF16 GEMM ruler | 97.3 TFLOP/s | model-free | 5 Sep 2026 | `bench/gemmpeak.py`, `results/profile/step-breakdown.md` |
| Mesh all-reduce, channel sweep | see `results/mesh/` | model-free | 4–5 Sep 2026 | `results/mesh/all-reduce-sweep.md` |
| Multilink sweep, 9 arms | see `results/mesh/` | model-free | 5 Sep 2026 | `results/mesh/multilink-sweep.csv` |
| Fabric raw, `ib_write_bw` | 98.0 Gb/s per link | model-free | 29 Aug 2026 | `docs/00` §4.4 |
| PCIe link state | `32GT/s, Width x4`, 12/12 | — | 5 Sep 2026 | `docs/00` §4.3 |
| sm_12x stack patches — the PDL race question, by logit divergence | Between-arm p95 **6.2–6.5e-02** against a within-arm pooled floor p95 of **7.7–8.0e-02**: two arms differ *less* than two runs of one arm. Between-arm max 3.028e-01 = **3.9×** the floor against a K = 4 threshold, **no outlier prompt**, and in the first comparison the arms diverge in token sequence later (position 5) than one arm does from itself (position 3). **No race detected** — on one prompt set, 128 generated tokens, concurrency 1 | diagnostic image `exl3-zeus:e7e345e-dflash`, separate in-container tree, TP=3+EP, 0.83, `CUDA_EXL3_DETERMINISTIC=1`, three rounds; production 10 never modified | 6 Sep 2026 | `results/kernels/sm12-stack-patches-ab.md` §4 |
| sm_12x stack patches — what the four items cost | PDL on against off, six rounds pooled against three: C1 69.54 / 70.12 (**+0.8 %**), C8 197.85 / 199.19 (**+0.7 %**), both in band. The one reading outside its band — C1 per-stream at +5.4 % — was deleted by a repeat: 69.75, then **76.17** on the identical configuration. Items 2–4 on against off: C1 −1.1 %, C8 +0.9 %, prefill-fresh −0.9 %, KV pool −0.3 %, so **the per-chunk memset cost the preparation note priced did not appear**. Gates 10/10 · 12/12 in every arm with one warm 11/12 flake that did not reproduce in three re-runs; KV pool across the five arms 5,666,666–5,694,214, **0.5 %** | as above | 6 Sep 2026 | `results/kernels/sm12-stack-patches-ab.md` §5–§6 |
| sm_12x — what was adopted with production 11, and what was not | Items **2 + 3** (`patch-kpool-init.py`) and item 1 **as PDL off** (`patch-pdl-gate.py`) moved out of `patches-optional/sm12/` into the track's own patch tree behind `HAREM_SM12_ITEMS=pdl,kpool`, fail-closed on an unrecognised entry. **Item 4 was not taken**: what it removes could not be measured from the client in either direction. `patch-kpool-init.py` cannot be applied from inside the container — one target is bind-mounted read-only from `$OVERLAY_DIR` and needs a pre-patched host-side copy, verified with `diff -r` | prod 11 | 6 Sep 2026 | `tracks/tp3/patches/README.md`, `results/kernels/sm12-stack-patches-ab.md` §9 |
| All-reduce latency sweep, 8 B → 64 MiB doubling, 10 warmup + 50 timed per size | **8 KiB 74.68 µs · 64 KiB 86.40 µs · 1 MiB 275.12 µs · 16 MiB 1,096.79 µs = 20.40 GB/s busBW = 98.1 % of our own measured 20.8 GB/s wire · 64 MiB 3,954.98 µs / 22.62 GB/s.** From 8 B to 32 KiB — a **4,096×** range — the time never leaves **72–85 µs**: decode is latency-bound and prefill is at the wire. The 128 KiB point (172.48 µs, twice its neighbours) reproduced in all three repetitions and is the surviving trace of the old cliff. **132 of 132** correctness checks passed | model-free, production plugin build and production NCCL environment, engine up and idle | 6 Sep 2026 | `results/mesh/nccl-latency-sweep.md`, `bench/nccl-latency-bench.py` |
| `NCCL_MAX_NCHANNELS=8` and `NCCL_PROTO`, measured size by size | The channel cap does **nothing** at decode sizes (8 and 64 KiB inside noise) and is worth up to **11×** in the cliff: 1 MiB 275 µs against 1,388, 4 MiB 306 against 3,554, 16 MiB 1,097 against 3,046, and no difference at all at 64 MiB where bandwidth is already saturated. `NCCL_PROTO` auto is at least as good as every forced protocol at every size and clearly better between 128 KiB and 1 MiB, where `Simple` is **4–7× slower** | model-free, as above | 6 Sep 2026 | `results/mesh/nccl-latency-sweep.md` §6 |
| **Three of our own all-reduce harnesses disagree at small sizes, and all three readings stay** | 64 KB: `ar_bench.py` **61.3 µs**, this sweep **86.4 µs**, `mesh_sweep.py` **143 µs**. 8 KB: **38.6** against **74.68**, a factor of two. They differ in more than one variable at once, so **no cause can be assigned from the data that exists** and none of them is corrected against the others. The new page's conclusions rest on the curve's flatness and on a 16 MiB point where the harnesses agree, not on an absolute small-size value | model-free, three harnesses and three configurations | 4–6 Sep 2026 | `results/mesh/nccl-latency-sweep.md` §6.3, `results/mesh/all-reduce-sweep.md` §5, `docs/06` §12.1 |
| MLA prefill at production selection overlap, on the target part | At 262K context production runs **1.013×** a fully cache-resident arm at our own 22-head TP=3 shape (+1.3 %) and **1.054×** at the kernel author's 16-head shape (+5.4 %), against an independent-selection arm at **1.820×** and **2.468×**. Across all six cells (three contexts × two head counts) production closes **96.0–98.4 %** of the independent→drifting distance, and its excess over drifting at 262K — 223 µs at 22 heads, 575 at 16 — is **below one cold read of its own 187 MiB working set** (807 µs) | model-free, **one GB10 node**, kernel `cuda-exl3 754421f` inside the production image, engine down, 3-round medians | 6 Sep 2026 | `results/kernels/mla-prefill-falsification-gb10.md`, `bench/mla-prefill/` |
| The new item that falsification produced, and the check it did not run | At **22 heads** the kernel's compute-only floor costs **13–16 % more per head** than at 16 — 1.586 / 1.557 / 1.589× against a linear 1.375×, at all three contexts — and it is isolated to the compute path, because the traffic-bound arm's ratio sits *below* 1.375×. The fixture's correctness gate ran at a **2-head** smoke shape: the 22-head numeric check against the reference is **`[not tested]`**, its CPU-only self-test having passed 24/24 | as above | 6 Sep 2026 | `results/kernels/mla-prefill-falsification-gb10.md`, `HELP-WANTED.md` §8 |
| Other published recipes, quoted as their authors publish them | A dozen recipes, every figure `[reported]` exactly as its repository states it with the conditions it states — none re-derived, rescaled or averaged. The rows worth the page are §3.4 and §4.2: **two other people quantized this model's dense path independently, one at two nodes and one at three, by two different routes, and both measured a gain in the same band as ours** | — | 5 Sep 2026, §5.3 corrected 6 Sep | `docs/16-comparison-with-published-recipes.md` |
| Repository restructure and rename — **no measurement changed** | `patches/tp3full/` → `tracks/tp3/patches/` and `patches/tp2full/` → `tracks/tp2/patches/`, file for file; one byte-identical duplicate deleted; every `docs/NN` page gains an "Applies to" line and all **86** troubleshooting entries a track tag (45 both, 26 both but measured at TP=3 only, 13 TP=3 only, 2 TP=2 only). `preflight-tp3.py`'s vocabulary gate gained the per-rank condition it had never tested: `lcm(128, tp)` is right at `tp=3`, right by luck at `tp=2` and **wrong at `tp=4`**, where the unit is `128 × tp` = 512 — and `tp=4` itself is **`[not tested]`**. The repository is now `GLM-5.3-Flash-EXL3-DGX-Spark` | — | 6 Sep 2026 | `CHANGELOG.md`, `tracks/README.md`, `docs/00-start-here.md` |

**Five provenance facts a reviewer should not have to dig for:**

1. **Production 9 was profiled**, on the live server, all three ranks, no restart. Two of its rows —
   NCCL and CPU gap at C1 — are **inflated by the profiler itself**, because that arm launches 2,738
   kernels per step and CUPTI charges roughly a microsecond per kernel boundary. With the profiler off
   the two together are ≤17.19 ms rather than the 29.1 the trace prints, and the step wall is 72.52 ms
   rather than 84.44. **Subtracting the instrument properly on this arm has not been done**
   `[not tested]`; the prefill column, where the overhead is +1.4 %, is safe at face value.
2. **The MMLU figures for production 7 and 8 (86.4 ±0.7) were measured at TP=2** and carried forward.
   Only production 9's **86.47 ±0.74** was measured at TP=3. The gates are identical between the two
   arrangements, which is exactly why someone should check.
3. **The production 8 headline is a pool of two same-day runs** (56.9 / 175.4), because that arm's
   documented boot-to-boot spread is about 7 %. The single p8 boot alone read 56.8 / 172.8.
4. **Every headline speed figure is the median of one boot's rounds, and the spreads are not equal
   across metrics.** On the four production 9 and 10 boots, C1 boot medians span **1.1 %**, C8
   **2.5 %** and C4 **7.4 %**; within a single boot, round-to-round peak-to-peak reaches 5.7 % at C1
   and 9.8 % at C4. **Production 10's C4 of 144.6 against production 9's 134.6 is that spread, not a
   gain** — a memory fraction does not buy 7 % of four-way throughput. Every round of every boot is
   printed in `docs/10` §1.1 so this can be checked rather than taken on trust.
5. **The boot-from-power-on figure is published twice because its own log disagrees with itself.**
   The harness printed `+242 s`; the wall-clock stamps in the same file give **315 s**, and 242 s
   before the health check matches no event recorded. We could not reconstruct the counter's origin,
   so both are printed and the larger is the planning figure. The decomposition is not in doubt:
   units finished at +98 to +103 s, container to served token 212 s. `results/boot/boot-ledger.md`.

---

## 6. The retraction index

Everything we published here and then withdrew. The full text of each, with the measurement that
overturned it, is in [docs/11](../docs/11-open-issues.md) §1 unless the `#` column names another
page — this is the index, so a reviewer can see the shape of our error rate in one place.

| # | What we claimed | What it actually was |
|---|---|---|
| 1.1 | The missing `n_rows` also costs the non-expert-parallel path | It does not; the tail is handled there |
| 1.2 | The extra masking pass is under 1 % of the MoE layer | 2.9 % at M=8 rising to **15.8 % at M=2048** |
| 1.3 | One upstream build is ~10 % slower end to end, and we do not know why | Boot-to-boot spread: **16 % at C8 with nothing changed at all** |
| 1.4 | NCCL picks the LL protocol at 16 MB and leaves half the link unused | The tuner plainly does not choose it; the real finding underneath was a different one |
| 1.5 | The MLA tuner re-tunes on every prefill chunk, wasting 2.6 % of prefill | A bounded warm-up repeat, not continuous re-tuning; the proposal to disable tuning was withdrawn |
| 1.6 | The mesh ceiling is ~13 GB/s against a 25 GB/s link, and GPUDirect is the fix | Two errors in one sentence, both ours |
| 1.7 | A pair of cables is worth 50 GB/s, so the collective runs at 28 % of the fabric | 50 GB/s is the **wire**; the card sits in a **PCIe Gen5 x4** slot, so the real ceiling is ~15 GB/s per card and ~30 GB/s per node |
| 1.8 | Two smaller ones — a retired patch, and a CUDA-graph capture claim | See docs/11 |
| 1.10 | Three rows of the step-time breakdown, and the ranked target list built on them | `exl3_moe_combine` at 1.5 % of a chunk — **the kernel does not exist in this build**; `_zero_kv_blocks` at 1.3 % measures **0.09 %**; the "5.45 ms C1 idle" is **3.47 ms** once the profiler's own cost is subtracted, so graph coverage is worth +1.5–2 % rather than +6 % |
| 2.3 | 8.2 GiB per worker is stranded; equalising ranks is worth 8–26 % of pool | **The instrument.** The pool is a difference between two `/proc/meminfo` readings and runs backwards; acting on this would have over-committed the head node |
| 2.17 | The mHC triple shares a buffer, so the fusion is blocked | Thread-local dataflow, not a shared buffer |
| 2.18 | (came with production 7) | See docs/11 §2.18 |
| 2.22 | The checkpoint's scope was a quality decision | It was never a quality decision — the premise was wrong |
| **new** | **Production 9 cost 2.4 points of draft acceptance — "this is the cost"** | **A harness artefact.** Pooled by draft token over five levels and three boots it is **+0.18 points**. `bench-sweep.py` cycles `prompts[i % 12]`, so C1/C2 see 8 of 12 prompts and C4–C8 see all twelve; a two-group model explains all five levels at R² = 0.97, and the sign reverses at C6. Net effect on tok/s: **+0.24 %** |
| **new** | Full scope is "equal both ways" on prefill | True of the **wall** and it hides the class underneath: the dense stage in prefill is **+17.3 ms, +10.4 %** at M=1,792. The plugin author reproduced it on his own card and withdrew a "cuBLAS parity" claim from his README |
| 1.11 | **"EXL3 is 1.58–1.76× slower than BF16 on the KDA shapes at M=8" — the sentence that closed the quantization item** | **A warm number, on the wrong card.** The ~300 MB weight bank was sized for the large shapes and left the 0.72 MB KDA arm resident in a 101 MB L2. Cold on the **target** GPU the same shape reads **1.023**; GB10's own warm arm reproduces the withdrawn 1.596 at **1.605**. Seven of nine shapes reverse sign and the family goes from **−0.584 to +0.050 ms/step**. Two companions go with it: "GB10's ratios will be worse" (every family came out **better**) and "not bandwidth-bound, so bytes are not the cost" (right, but the cost is **two dependent launches**, so the remedy is a fusion). **The closure survives, re-scoped:** the arms stay BF16 for want of a gain, `kv_b_proj` still holds 96 % of it and still needs a kernel, and prefill was never re-measured |
| 2.4 | **"0.85 will not be attempted on this stack"**, and "0.88 was never attempted" | **The wrong ruler, on a machine that no longer exists.** Both rested on `MemFree` against a 4 GiB floor, and the 0.85 boot they came from predates the fast-load page-cache fix. Climbed rung by rung on 6 September against **swap traffic under load** instead: 0.85, 0.87 and 0.88 all pass, **0.90 is rejected**, and **0.87 is production configuration 11** |
| 2.27 | The MLA-prefill "21–26 % overlap gap, worth about 2 % of a prefill chunk" | **It does not exist, and the item closes at zero.** Production was never between the kernel author's two arms — its selection turns over about **152 keys per row**, 76× his low-turnover arm — and on our own 48-SM part production runs within **1.3 %** of a fully cache-resident arm at 262K context. The only lever left on that kernel is less work, not less traffic |
| docs/17 §2.4 | "The driver takes 14.2 GiB" | **Stale by about 10 GiB.** It came from an NVFP4-era ceiling scan whose best observed free was 107.43 of 121.63; the three ranks start at **112.01 / 112.97 / 113.15** and an idle node offers **116.79**. The driver's fixed reserve is **3.12–3.49 GiB** and the total genuinely unavailable at idle is **4.84**. What limits the memory fraction now is the host's share at run time — a ruler correction, not a lever |
| docs/17 §4.1 | **Eight** of the nine KDA state slots exist so a rejected speculative step can be rolled back | **Seven.** The nine can be read as `1 + 8` off the non-align branch of that `if` and our first pass did read it that way; **align is the branch we run**, so it is `2 + 7`. No total moves — nine either way — but it is why the compact-rollback alternative goes to **two** slots and not to one |
| docs/15 §3.3 | Full scope at TP=2 "is a rig and not a serving configuration" — it cannot boot at `max_model_len` 1,000,000 — and is "~10 GiB heavier per node" | **Neither reading survives two changes made since.** The draft page fix cuts blocks-per-request 640 → 280 and the launcher's settle gate turns 0.73 GiB of available KV memory into **16.07**; candidate B boots at 1M with a **2,128,571**-token pool and is the recommended two-node configuration. And it is **4.5–4.7 GiB lighter**, not 10 heavier: the old figure came from a boot with no settle gate |
| **new** | Production 11 costs **6.4 % of C4** | **Boot noise, and a repeat is what showed it.** The load boot read C4 at **135.0** against the reference's 144.2; the clean boot after a whole-cluster reboot read **145.9**. C4's six-round spread on this configuration is **11.1 %** against 2.5 % at C1 and 5.3 % at C8, and the same morning's ladder read C4 at 0.87 *without* these patches as 144.18. Printed rather than smoothed |

**Two retractions of the same number in two days** (1.6 and 1.7) is its own lesson, and it is the one
worth carrying away from this folder: **the ruler gets measured too.** "Two links × port rate" is not
a ceiling, it is the capacity of the wire; the path the card uses to reach the machine has to be
counted before a roofline is written. The same class of mistake put a 273 GB/s catalogue figure in
place of the 225 GB/s the device actually delivers.

Row 1.11 is that lesson a third time and in its hardest form: the ruler **was** measured, the artefact
**was** found — it read 210 % of peak — and a weight bank was added because of it. The bank was then
sized against the wrong shape on the wrong card, and the conclusion it protected was about the one
shape it did not cover. **Measuring the instrument once is not a property the instrument keeps.**

Nothing in `results/` was edited to match a retraction. The mistakes stay on the record with the
measurement that overturned them next to them.

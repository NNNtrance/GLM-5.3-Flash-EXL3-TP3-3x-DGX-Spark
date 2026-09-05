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
did not move. **0.85 will not be attempted on this stack.**

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
in the gigabytes, you are where we are. **0.85 will not be attempted on this stack**
([docs/11](../docs/11-open-issues.md) §2.4).

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
| KDA/MLA quantization gate bench | KDA arms EXL3/bf16 **1.58–1.76×** at M=8; whole family 0.851 ms of a 72.5 ms step; `kv_b_proj` 0.391× but no batched kernel exists | model-free, **workstation GPU** | 5 Sep 2026 | `results/kernels/kda-gate-bench.md`, `bench/kda_gate_bench.py` |
| Device read bandwidth ruler | 225.2 GB/s | model-free | 5 Sep 2026 | `bench/bw.py`, `results/profile/step-breakdown.md` |
| BF16 GEMM ruler | 97.3 TFLOP/s | model-free | 5 Sep 2026 | `bench/gemmpeak.py`, `results/profile/step-breakdown.md` |
| Mesh all-reduce, channel sweep | see `results/mesh/` | model-free | 4–5 Sep 2026 | `results/mesh/all-reduce-sweep.md` |
| Multilink sweep, 9 arms | see `results/mesh/` | model-free | 5 Sep 2026 | `results/mesh/multilink-sweep.csv` |
| Fabric raw, `ib_write_bw` | 98.0 Gb/s per link | model-free | 29 Aug 2026 | `docs/00` §4.4 |
| PCIe link state | `32GT/s, Width x4`, 12/12 | — | 5 Sep 2026 | `docs/00` §4.3 |

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
overturned it, is in [docs/11](../docs/11-open-issues.md) §1 — this is the index, so a reviewer can
see the shape of our error rate in one place.

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

**Two retractions of the same number in two days** (1.6 and 1.7) is its own lesson, and it is the one
worth carrying away from this folder: **the ruler gets measured too.** "Two links × port rate" is not
a ceiling, it is the capacity of the wire; the path the card uses to reach the machine has to be
counted before a roofline is written. The same class of mistake put a 273 GB/s catalogue figure in
place of the 225 GB/s the device actually delivers.

Nothing in `results/` was edited to match a retraction. The mistakes stay on the record with the
measurement that overturned them next to them.

# Category speed on production configuration 12, and why prose is the slow row

Settings for every number on this page: image `exl3-zeus:754421f`, three nodes, TP=3 + expert
parallel, **full-scope** EXL3 weights (`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw),
`kv-cache-dtype fp8` **and an fp8 draft cache**, DFlash2 draft at **k=7**, `--block-size 256`,
`HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, `gpu-memory-utilization 0.88` with the indexer workspace bound to 512 MB,
CUDA graphs **off** (the FlashInfer spec-decode support gate sets `cudagraph_mode=NONE`; 0.0 GiB of
graph memory in the boot line), `max_tokens 700`, temperature 0, reasoning effort **low**, warm
engine, no boot inside the measurement window. That is **production configuration 12**. The serving
boot read `GPU KV cache size: 7,118,457 tokens`, inside the 6.62–7.46 M band this configuration is
declared on. 7 September 2026 `[measured-here]`.

**Protocol.** One warm-up round plus **three measured rounds, median of three**
([docs/09](../../docs/09-measurement-protocol.md) §1.1) — the persisted MLA tuner cache is what earns
three rounds. Six prompts per category run **sequentially** for C1 and **four in parallel** for C4.
Decode rate excludes TTFT: `(tokens − 1) / (end − first token)`. Acceptance and accepted tokens per
step are differenced from the server's own `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`
counters across each category window, so `accept_len = 1 + accepted / drafts`.

**One comparability warning, and it applies to anyone reproducing this.** These numbers were measured
with the **original** category prompt set, in which half of the six prompts in each category are in
Turkish and half in English — the same set that produced the production-9 row, which is the only
reason the two can be compared at all. `scripts/category-speed.py` in this repository ships the
**English translation** of that set, and its own header says so. A rerun with the shipped script is
your own baseline, not a reproduction of ours.

## 1. The measurement

Single stream, six prompts per category, median of three rounds `[measured-here]`:

| | C1 decode tok/s | aggregate tok/s | TTFT | acceptance | accepted tokens per step | mean tokens |
|---|---|---|---|---|---|---|
| **prose** | **29.04** | 27.68 | 0.26 s | **13.02 %** | **1.91** | 671 |
| **code** | **61.46** | 58.14 | 0.34 s | 45.94 % | 4.22 | 695 |
| **math** | **76.20** | 67.57 | 0.38 s | 57.11 % | 5.00 | 417 |
| **JSON** | **73.13** | 65.20 | 0.41 s | 52.90 % | 4.70 | 515 |

Four in parallel, same prompts `[measured-here]`:

| | C4 per stream | C4 total | TTFT | acceptance | accepted tokens per step |
|---|---|---|---|---|---|
| prose | 14.45 | **52.16** | 0.42 s | 10.60 % | 1.74 |
| code | 35.57 | **115.18** | 0.42 s | 49.65 % | 4.48 |
| math | 48.79 | **120.56** | 0.91 s | 57.74 % | 5.04 |
| JSON | 44.44 | **108.75** | 0.53 s | 52.38 % | 4.67 |

Round to round inside this boot, C1 decode: prose 1.2 %, JSON 2.8 %, code 4.9 %, math 8.4 % peak to
peak. Math is the noisiest category on this probe because its answers are the shortest (417 tokens
against prose's 671), so each round has the fewest steps to average over.

### 1.1 Against production 9, which is what the front-page row has said since 5 September

The production-9 figures were measured on 5 September on the promotion boot of the full-scope
checkpoint, `gpu-memory-utilization 0.80`, with the same prompt set and the same script
`[measured-here]`:

| | production 12 (0.88, workspace bound) | production 9 (0.80) | change | verdict |
|---|---|---|---|---|
| prose | **29.04** | 29.1 | −0.2 % | equal |
| code | **61.46** | 61.7 | −0.4 % | equal |
| math | **76.20** | 79.6 | −4.3 % | inside its own 8.4 % round spread: equal |
| JSON | **73.13** | 72.8 | +0.5 % | equal |
| acceptance, prose / code / math / JSON | 13.0 / 45.9 / 57.1 / 52.9 % | 12.8 / 45.9 / 58.0 / 53.7 % | ≤0.9 pt | equal |

**That is the result worth stating.** Between those two arms the KV pool went from **5,165,289** to
**7,118,457 tokens**, +37.8 %, through three memory rungs and the indexer workspace bound — and not
one category moved outside its own noise. The pool was bought with memory, not with tokens per
second.

## 2. Why prose is slow, and why it is the draft

Prose is not slow because the model thinks harder about it, and not because the EXL3 checkpoint is
worse at it. It is slow because **speculative decoding stops paying** there. Three measurements say
so, none of them an inference from the speed column alone.

### 2.1 The engine steps at the same rate whatever the content; only the tokens per step change

With a k-deep draft the engine produces `L = 1 + k × acceptance` tokens per step. Divide each
category's decode rate by its own `L` and what is left is the **step rate** — how many target
forwards per second the stack actually runs. Across ten boots of this stack, spanning four image
builds, both draft page sizes, four memory fractions, the mesh patches, the indexer workspace bound
and the full-scope checkpoint promotion, the step rate is flat **across categories inside every
single arm**, while the decode rate spans 3.5× `[measured-here]`:

| Arm | prose | code | math | JSON | step/s range | spread |
|---|---|---|---|---|---|---|
| `bc0e0f6`+0003, MNBT 4096 | 21.6 / 1.90 = 11.39 | 47.7 / 4.28 = 11.14 | 62.1 / 5.01 = 12.39 | 55.4 / 4.79 = 11.57 | 11.14 – 12.39 | 10.7 % |
| `61a17bc`, fusion auto | 22.0 / 1.89 = 11.65 | 47.0 / 4.16 = 11.29 | 61.9 / 4.85 = 12.76 | 57.1 / 4.80 = 11.89 | 11.29 – 12.76 | 12.3 % |
| `61a17bc`, fusion off | 21.6 / 1.85 = 11.65 | 48.4 / 4.21 = 11.49 | 62.5 / 5.00 = 12.49 | 58.0 / 4.86 = 11.92 | 11.49 – 12.49 | 8.4 % |
| `f4987cf`, MNBT 2048 | 22.2 / 1.88 = 11.80 | 47.3 / 4.16 = 11.36 | 63.5 / 4.98 = 12.74 | 56.2 / 4.79 = 11.72 | 11.36 – 12.74 | 11.5 % |
| `f4987cf` + draft page 256 | 21.7 / 1.85 = 11.75 | 48.5 / 4.28 = 11.34 | 61.0 / 5.03 = 12.12 | 56.6 / 4.70 = 12.05 | 11.34 – 12.12 | 6.6 % |
| draft page 256 at 0.85 | 21.8 / 1.85 = 11.76 | 48.3 / 4.21 = 11.48 | 61.1 / 5.03 = 12.14 | 56.5 / 4.70 = 12.03 | 11.48 – 12.14 | 5.6 % |
| fast boot S1+S2+S3 | 22.5 / 1.92 = 11.74 | 47.3 / 4.12 = 11.48 | 60.2 / 5.05 = 11.91 | 57.0 / 4.81 = 11.86 | 11.48 – 11.91 | 3.8 % |
| fast boot S4 (sidecar) | 22.4 / 1.91 = 11.73 | 47.9 / 4.20 = 11.41 | 59.0 / 4.91 = 12.01 | 57.7 / 4.84 = 11.91 | 11.41 – 12.01 | 5.1 % |
| **full-scope checkpoint (production 9)** | 29.1 / 1.90 = **15.35** | 61.7 / 4.21 = **14.65** | 79.6 / 5.06 = **15.73** | 72.8 / 4.76 = **15.30** | 14.65 – 15.73 | 7.2 % |
| **production 12** (this page) | 29.04 / 1.91 = **15.19** | 61.46 / 4.22 = **14.58** | 76.20 / 5.00 = **15.25** | 73.13 / 4.70 = **15.55** | 14.58 – 15.55 | 6.4 % |

Read the last row against any other. The full-scope checkpoint moved the **step rate** by about 30 %
— that is what a 4-bit dense stage bought — and it moved every category by that same 30 %. It did
not move the ratio between them by anything, because it did not touch the drafter. Within each arm,
the 3.8 – 12.3 % spread of the step rate is the ordinary run-to-run spread of this probe; the decode
column above it spans 2.6 – 2.9× in the same arm.

**So the whole of the prose-to-math gap is `L`, and `L` is the drafter's hit rate.**

### 2.2 Acceptance per category is a constant of the drafter, not of the configuration

The same ten arms, acceptance only `[measured-here]`:

| category | acceptance across ten arms | production 12 |
|---|---|---|
| prose | **12.1 – 13.1 %** | 13.02 % |
| code | 44.6 – 46.9 % | 45.94 % |
| math | 55.0 – 58.0 % | 57.11 % |
| JSON | 52.8 – 55.2 % | 52.90 % |

A one-point band on prose across ten engine configurations, four memory fractions, a checkpoint
promotion and a 38 % change in KV pool size. Production 12 lands inside every one of those bands
without widening any of them. Nothing in the launcher moves this column.

### 2.3 What speculation is actually worth, per category

The one arm on this stack that has been measured **with and without** the drafter is the two-node
one, and it prices the overhead directly `[measured-here]`. Three arms in one session, one prompt
set, image `exl3-zeus:dflash`, TP=2, BF16 draft cache, `gpu-memory-utilization 0.85`, two averaged
rounds rather than the three-round protocol adopted afterwards
([docs/04](../../docs/04-dflash2-port.md) §1):

| | per-stream decode, C1 | accepted tokens per step | step rate |
|---|---|---|---|
| no speculation | 14.73 tok/s | 1 (by definition) | 14.73 /s |
| DFlash2 k=7 | 50.79 tok/s | 5.46 | 9.30 /s |

**A speculative step costs 1.58× a plain step** — the draft's own k forwards plus a verify batch of
k+1 instead of 1. From that, break-even is `(1.58 − 1) / 7 =` **8.3 % acceptance**, and each
category's real return on speculation follows `[estimate]` (the overhead factor is measured at two
nodes; the three-node factor is `[not tested]`):

| category | acceptance (production 12) | tokens per step | speculation is worth |
|---|---|---|---|
| **prose** | **13.02 %** | **1.91** | **×1.21** |
| code | 45.94 % | 4.22 | ×2.66 |
| math | 57.11 % | 5.00 | ×3.16 |
| JSON | 52.90 % | 4.70 | ×2.97 |

**Prose sits 4.7 points above the break-even line.** The engine is running a 7-deep draft, throwing
away about seven of every eight drafted tokens, and netting roughly 20 % over an engine with no
drafter at all. That is why the prose row looks like a different machine: on prose this stack runs
close to unspeculated speed and pays the verify overhead to stay there.

### 2.4 A different drafter moves it. A configuration knob does not

**A different drafter moves it, and the proof is already in this repository.** An MTP drafter at k=3,
measured on the two-node arm in the same session as DFlash2, **wins the prose row** — 21.3 against
18.5 tok/s — while losing every other row by 20–40 % ([docs/04](../../docs/04-dflash2-port.md) §7).
Compare the two by **tokens per step**, not by acceptance rate: a 3-deep and a 7-deep draft have
different denominators, and reading their rates against each other is the trap
[docs/04](../../docs/04-dflash2-port.md) §1 warns about. On prose, MTP advances **1 + 3 × 0.377 =
2.13** tokens per step where DFlash2 advances **1 + 7 × 0.128 = 1.90** — and MTP's step is the
cheaper of the two: it drafts three tokens where DFlash2 drafts seven, and the target then verifies
four positions instead of eight. On math, measured in that same
session, the order reverses hard: DFlash2 advances 5.11 tokens per step against MTP's 3.38. Prose
acceptance is a property of the draft model. It is not a law of nature.

**No configuration knob is available for it.** Three were looked for:

| Lever | Result |
|---|---|
| Lower `k` for prose | k=5 raises prose 22.0 → 22.8 tok/s (+3.6 % at C1, +15 % at C4) and costs **every other category and every concurrency level** 3.5 – 6.4 % ([docs/04](../../docs/04-dflash2-port.md) §6) `[measured-here]` |
| Per-request `k` | Not available. `num_speculative_tokens` lives in `--speculative-config` and is fixed when the engine starts; the OpenAI-compatible API has no per-request override. One value serves the whole engine, and it is k=7 `[measured-here]` |
| A newer revision of the same drafter | Tested on production 12 on 7 September: aggregate acceptance **equal** (C1 62.08 → 61.32 %, C8 60.53 → 61.61 %, signs opposite, both inside the 60–65 % band), all gates full first time. The ~5 % speed reading on that arm is a whole-boot effect, not the draft — **prefill fell by the same 5 %, and prefill does not use the drafter** ([docs/04](../../docs/04-dflash2-port.md) §8.1). Its **per-category** split was then run as well, §2.5 below: **prose does not move** |

### 2.5 The newer drafter revision, split by category: prose does not move

The aggregate comparison above could in principle have hidden a prose gain, so the category probe was
run on the newer revision too, the same night, on the same stack. Same script byte for byte, same
prompt set, same settings, only the draft directory differing; the newer arm ran the shortened
protocol — one warm-up round and **one** measured round — so this is a round-to-round comparison
rather than median against median `[measured-here]`:

| C1 | acceptance, `dc77ff1c` | acceptance, `bf582e4e` | change | tokens per step | decode tok/s |
|---|---|---|---|---|---|
| **prose** | 12.97 % | **12.04 %** | **−0.93 pt** | 1.91 → 1.84 | 29.04 → 27.70 |
| code | 45.94 % | 45.91 % | −0.03 pt | 4.22 → 4.21 | 61.46 → 61.07 |
| math | 57.11 % | **60.36 %** | **+3.24 pt** | 5.00 → 5.23 | 76.11 → 82.92 |
| JSON | 51.69 % | 52.01 % | +0.32 pt | 4.62 → 4.64 | 71.43 → 70.74 |

At C4 the same four rows read −0.10, +2.02, +0.61 and +0.31 points.

**Read prose against the arm's own spread before reading it as a loss.** The newer arm's two rounds
bracket the older one: its warm round put prose acceptance at **13.1 %** and its measured round at
**12.0 %**, a 1.1-point swing inside one boot, against production 12's own 0.15-point spread over
three rounds. So prose on the newer revision is **12.0 – 13.1 %** — the same band the drafter has
occupied across all ten arms on this page. **It is not better. The row does not move.**

**Math is the one column that may have moved**, and it is recorded as an observation rather than a
result: 60.0 % on the warm round and 60.4 % on the measured one, both **above** the 55.0 – 58.0 %
band every previous arm has read, worth +9.0 % of decode rate at C1 — but this is one boot, on a
stack whose boot-to-boot spread is 15.9 % ([09](../../docs/09-measurement-protocol.md) §2), and at C4
the same comparison falls to +0.61 points. It did not carry the aggregate: that arm's overall
acceptance came back equal. **The pin stays on `dc77ff1c`**; if the math reading is chased later it
is a two-boot question, not a promotion.

**So the honest statement is the plain one.** Prose is slow because the DFlash2 draft agrees with the
target model about one token in eight on free-running prose, and there is nothing in this stack's
configuration that changes that. It is not the checkpoint — the full-scope promotion raised every
category by 30 % and left acceptance untouched. It is not the kernels, the mesh, the KV geometry or
the memory fraction — ten arms of those moved prose acceptance by one point in total. **The only fix
is a better draft checkpoint**, and until one exists the prose row is what this engine does.

## 3. What this cost

Nothing, and it was looked for. The measurement is request traffic on a warm production engine: no
boot, no restart, no configuration change, no environment file touched, one engine window of
**28 minutes**. Quality gates on this boot: correctness probe **10/10** and code exam **12/12** cold,
and **10/10 · 12/12** again on the same boot after the benchmark `[measured-here]`. Free host memory
held at 3–4 GiB and **swap use stayed at zero**; GPU clocks sat at 2,405 MHz with package
temperatures of 51–57 °C and no throttle flag. Wanting a different number in the prose row would cost
either the drafter (a k=5 engine, −6.4 % single-stream and −3.5 to −5.2 % loaded, to buy +3.6 % on
prose) or a second engine, and neither is taken.

## 4. What would close the remaining gap in this page

- **An unspeculated arm at three nodes** `[not tested]`. The 1.58× overhead factor above is measured
  at two nodes on an older image, so the per-category "worth" column is an estimate rather than a
  measurement. One boot without `--speculative-config`, one category run, and that column becomes
  `[measured-here]`. It is about 25 minutes of engine time and nobody has spent it.
- ~~**The per-category split on a newer drafter.**~~ **Done the same night** `[measured-here]`, §2.5:
  prose does not move. What is left from that arm is its **math** reading, 3.2 points of acceptance
  above every previous arm on one boot — a two-boot question `[not tested]`.
- **Mixed load on the production configuration** `[not tested]`, still. `scripts/mixed-load-probe.py`
  has not run since the fast-boot arm.

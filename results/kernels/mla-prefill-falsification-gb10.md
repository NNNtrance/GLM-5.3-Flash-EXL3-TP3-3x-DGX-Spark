# The MLA-prefill ceiling, falsified at 48 SMs: the kernel author's correction holds, tighter than on his own card

**Verdict: the prediction held, at both head counts.** At 262K context the production-selection arm
runs within **1.3 %** of the fully cache-resident "drifting" arm in our own TP=3 shape (22 heads per
rank) and within **5.4 %** in the kernel author's shape (16 heads) — against an independent-selection
arm that needs **82.0 %** and **146.8 %** more time respectively. Across all six cells (three
contexts × two head counts) production closes **96.0–98.4 %** of the independent→drifting distance.
**The "21–26 % overlap gap, worth about 2 % of a prefill chunk" that this item was carried at before
this run does not exist.** It closes at zero, on this part as it did on his.

Measured against that closure, one new cost fell out: at 22 heads the kernel's own compute floor runs
**13–16 % more expensive per head** than at 16 — the first concrete candidate for the lever the author
said is the only one left. Narrative and closure: [docs/11](../../docs/11-open-issues.md) §2.27. The
item that started this: [`sm12-stack-patches-ab.md`](sm12-stack-patches-ab.md) §8. Tooling:
[`../../bench/mla-prefill/`](../../bench/mla-prefill/).

---

## Why this run happened

[`sm12-stack-patches-ab.md`](sm12-stack-patches-ab.md) §8 produced a diagnostic datum the `cuda-exl3`
author had asked for: a median adjacent-row top-k overlap of **0.926** at production's steady
1,792-row prefill chunk — about 152 of 2,048 selected keys turning over per row, roughly 76× his
"drifting" arm's turnover. That number went to
[issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5) `[reported]`, and it landed two orders of
magnitude from the low-turnover arm he had been comparing production against. He corrected his own
MLA-prefill ceiling in commit **`5fd7299`**, *"Correct the MLA prefill ceiling: at production overlap
there is no gap"*, and predicted: run the production arm on a GB10 at 262K context and it should land
**within a few percent of the drifting arm, not between drifting and independent** — because what has
to fit in L2 is the union over a key's *residence window* (about 4,096 keys, ≈ 4.5 MiB at 7.4 %
turnover per row), not the union over the whole chunk (187 MiB at 262K — 7.8× a 24 MiB L2).

On his card (188 SM, 128 MiB L2) at 262K: drifting 2,385.8 µs, production 2,422.8 µs, independent
3,474.0 µs — production 1.6 % above drifting `[reported]`. This page is that prediction run on the
part it was written for: a 48-SM GB10 with a 24 MiB L2, at our own TP=3 head count as well as his.

## Settings

`[measured-here]`, 6 September 2026, 14:12–14:13 Istanbul, one GB10 node.

- **Kernel:** `cuda-exl3 754421f` — the production commit, inside the production image
  `exl3-zeus:754421f`. The fixture's one kernel entry point is `mla_decode`; its nine-argument
  signature is present in the production image unchanged, so **no diagnostic image was needed** — the
  kernel measured is the kernel serving production.
- **Fixture:** the arms and the selection generator are copied verbatim from the author's own
  `bench/bench_mla_prefill.py` at his commit `5fd7299`, not re-implemented — see
  [`../../bench/mla-prefill/`](../../bench/mla-prefill/). Additive on top of his file: rows fixed at
  1,792 (his own `ctx_sweep` default, and this stack's steady prefill chunk), 3-round medians with
  CUDA events, two head counts, a second ruler, and a correctness check ahead of any timing.
- **Container:** `--rm --gpus all --ipc=host --cpuset-cpus 10-14 --memory=4g`, `--entrypoint
  /usr/bin/python3`, rc 0, 17 s wall, peak GPU allocation **1.65 GiB** (budget 2 GiB). No CUDA graph,
  no engine process in memory.
- **Window:** the engine was stopped on all three nodes for other work that day (MemAvailable 116.8
  GiB) rather than a boot taken for this bench specifically — an opportunistic run inside someone
  else's engine-free window, per [docs/09](../../docs/09-measurement-protocol.md) §10. Two earlier
  attempts an hour before this one (13:56) found the engine idle but MemAvailable at 4.7 GiB, under
  this stack's 6 GiB floor for an engine-free bench; both aborted cleanly against that gate and the
  lock was released untouched each time.
- **Shape:** `rows=1792`, `head_dim=576`, `v_head_dim=512`, `index_topk=2048`, three context lengths
  (32,768 / 71,680 / 262,144), two head counts (**16** — his TP=4 per-rank count — and **22** — our
  TP=3 per-rank count), 3-round medians, CUDA events.

Reproduce: [`../../bench/mla-prefill/`](../../bench/mla-prefill/)`/mla_prefill_falsify.py` inside the
production image, engine down or idle with headroom, per that directory's README.

## The three arms

| arm | definition | what it isolates |
|---|---|---|
| `drifting` | row *i* is row *i*−1 with 2 entries replaced | maximal reuse — the kernel's own compute/issue floor, traffic taken away |
| `production` | 7.4 % per-row turnover, calibrated to our measured 0.926 overlap | the pattern production actually selects |
| `independent` | every row draws its own selection fresh | zero reuse — a pure gather-bound ceiling |

## The fixture, checked before it is trusted

| check | result | verdict |
|---|---|---|
| `mla_decode` vs a torch MLA reference (softmax attention over the gathered keys) | output `(4, 2, 512)`, finite, max relative error **3.45e-03** | kernel genuinely computes MLA, does not bail out early |
| ruler A — streaming bf16 `sum`, 256 MiB | **243 GB/s** | inside this card's independently measured band (225–246) |
| ruler B — cold 3-copy bank, `F.linear` 512×32,768 | m=8 **229**, m=64 **227** GB/s | inside band |
| fixture working set, three contexts | 36 / 77 / 187 MiB | matches the author's own table **exactly** — the port is correct |
| fixture's realised adjacent-row overlap | drifting 0.999 · production **0.926** · independent 0.008–0.060 | the production arm is calibrated to our own indexer capture, not merely close to it |

The production arm's realised overlap at 262K, 0.926, is the same figure `sm12-stack-patches-ab.md`
§8 measured off the live indexer — the pattern under test is the pattern in production.

**The one gap this run left**, stated because it has bitten this stack before: the correctness check
above ran at a small 2-head smoke shape. **The 22-head shape was timed but not numerically verified
against the reference in this run.** An unrelated kernel (`b12x`'s decode path) has silently
miscomputed at exactly the 22-head TP=3 shape before, precisely because it was an untested one — see
[docs/11](../../docs/11-open-issues.md) §2.27 and the b12x root-cause note in the project's decision
record. Status of that follow-up is below, in its own section.

## The full table

µs, 3-round medians, `%ruler` against ruler A (243 GB/s) `[measured-here]`:

| ctx | latent MB | H | arm | µs | working set | ws / L2 | per-row GB/s | %ruler | adjacent overlap |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32,768 | 38 | 16 | drifting | 10,674.4 | 6 MiB | 0.24× | 396 | 163 % | 0.999 |
| 32,768 | 38 | 16 | production | 11,051.1 | 36 MiB | 1.50× | 383 | 158 % | 0.930 |
| 32,768 | 38 | 16 | independent | 20,135.4 | 36 MiB | 1.50× | 210 | 87 % | 0.060 |
| 32,768 | 38 | 22 | drifting | 16,932.6 | 6 MiB | 0.24× | 250 | 103 % | 0.999 |
| 32,768 | 38 | 22 | production | 17,127.1 | 36 MiB | 1.50× | 247 | 102 % | 0.930 |
| 32,768 | 38 | 22 | independent | 25,803.8 | 36 MiB | 1.50× | 164 | 68 % | 0.060 |
| 71,680 | 83 | 16 | drifting | 10,808.7 | 6 MiB | 0.25× | 391 | 161 % | 0.999 |
| 71,680 | 83 | 16 | production | 11,186.9 | 77 MiB | 3.21× | 378 | 156 % | 0.928 |
| 71,680 | 83 | 16 | independent | 23,948.7 | 79 MiB | 3.28× | 177 | 73 % | 0.028 |
| 71,680 | 83 | 22 | drifting | 16,825.6 | 6 MiB | 0.25× | 251 | 104 % | 0.999 |
| 71,680 | 83 | 22 | production | 17,141.7 | 77 MiB | 3.21× | 247 | 102 % | 0.928 |
| 71,680 | 83 | 22 | independent | 28,684.5 | 79 MiB | 3.28× | 147 | 61 % | 0.028 |
| 262,144 | 302 | 16 | drifting | 10,698.6 | 6 MiB | 0.26× | 395 | 163 % | 0.999 |
| 262,144 | 302 | 16 | **production** | **11,273.7** | 187 MiB | **7.78×** | 375 | **155 %** | 0.926 |
| 262,144 | 302 | 16 | independent | 26,399.5 | 288 MiB | 12.00× | 160 | 66 % | 0.008 |
| 262,144 | 302 | 22 | drifting | 17,000.7 | 6 MiB | 0.26× | 249 | 103 % | 0.999 |
| 262,144 | 302 | 22 | **production** | **17,223.9** | 187 MiB | **7.78×** | 245 | **101 %** | 0.926 |
| 262,144 | 302 | 22 | independent | 30,948.6 | 288 MiB | 12.00× | 137 | 56 % | 0.008 |

`%ruler > 100` is only possible reading from cache. At H=16 the drifting and production arms read
**163 % and 155 %** of achievable bandwidth with a footprint **7.78×** the L2 — the cleanest evidence
in the table. At H=22 the kernel is compute-bound enough that the same figures fall to 101–103 % and
stop carrying a bandwidth argument on their own; the ratio columns below are what settles it there.

### Ratios against the drifting arm

| ctx | H | drift µs | production µs | independent µs | **prod / drift** | indep / drift |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 16 | 10,674.4 | 11,051.1 | 20,135.4 | **1.035** | 1.886 |
| 32,768 | 22 | 16,932.6 | 17,127.1 | 25,803.8 | **1.011** | 1.524 |
| 71,680 | 16 | 10,808.7 | 11,186.9 | 23,948.7 | **1.035** | 2.216 |
| 71,680 | 22 | 16,825.6 | 17,141.7 | 28,684.5 | **1.019** | 1.705 |
| **262,144** | **16** | 10,698.6 | 11,273.7 | 26,399.5 | **1.054** | 2.468 |
| **262,144** | **22** | 17,000.7 | 17,223.9 | 30,948.6 | **1.013** | 1.820 |

### The column that actually answers "which arm is production on"

`(independent − production) / (independent − drifting)` — the one number that says how much of the
distance between the two extremes production closed:

| ctx | H | **gap closed** | production's excess over drifting (µs) | one cold read of the working set (µs) | excess ÷ cold read |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 16 | **96.0 %** | 377 | 156 | 2.42× |
| 32,768 | 22 | **97.8 %** | 195 | 156 | 1.25× |
| 71,680 | 16 | **97.1 %** | 378 | 333 | 1.14× |
| 71,680 | 22 | **97.3 %** | 316 | 333 | 0.95× |
| 262,144 | 16 | **96.3 %** | 575 | 807 | 0.71× |
| 262,144 | 22 | **98.4 %** | 223 | 807 | 0.28× |

96–98 % in every cell. At 262K, production's excess over drifting is **below one cold pass over its
own 187 MiB working set** at either head count (0.28–0.71×), and three orders of magnitude under the
full `rows × topk` gather — 4.23 GB, 17,434 µs at this card's ruler. What production pays for is
compulsory misses, one touch per live key, not the gather the independent arm pays for. **The
residence-window account is doing exactly the work it claims to.**

## The pre-registered prediction, checked against what happened

Written before the run, unchanged after it (`00-on-aritmetik.md` in the source working notes):
*"we expect `independent/drifting` to come out larger than his 1.46× (estimate 1.6–2.0×) … if the
ratio comes out close to 1.0, our measurement is wrong, and we say that first."*

| claim | result |
|---|---|
| indep/drift > his 1.46× | **held** — in all six cells, lowest is 1.524 |
| estimated band 1.6–2.0× | **partly** — H=22 @ 262K = 1.820 (inside); H=16 @ 262K = 2.468 (above the band — our estimate was too conservative at his head count) |
| sanity gate: near 1.0 would mean the measurement is wrong | **passed, comfortably** — 1.52–2.47 |

The independent arm also approaches a pure gather-bound ceiling as context grows: it reaches 90–94 %
of "drifting plus the full gather at the ruler" at 262K, against 72 % at 32K (where the 36 MiB working
set only partly exceeds the 24 MiB L2). It behaves correctly as an upper bound, and increasingly so.

**Why the pre-registration matters here.** GB10 did not make this a weaker test — it made it sharper.
His card separates the two extreme arms by 1.46×; this one separates them by 1.82–2.47×, because this
card's bandwidth-to-compute ratio is roughly 6.3× lower than his, so the same leaked traffic would
cost about six times as much relatively. The "which arm is production on" question was asked at
higher resolution here and still answered "drifting".

## The anomaly: at 22 heads, the compute floor costs 13–16 % more per head

Going from 16 to 22 heads is 1.375× the work; a linear kernel would take 1.375× the time.

| ctx | drifting (compute-only) | production | independent (traffic-bound) |
|---:|---:|---:|---:|
| 32,768 | 1.586× → **+15.4 %** | 1.550× → +12.7 % | 1.282× → **−6.8 %** |
| 71,680 | 1.557× → **+13.2 %** | 1.532× → +11.4 % | 1.198× → **−12.9 %** |
| 262,144 | 1.589× → **+15.6 %** | 1.528× → +11.1 % | 1.172× → **−14.7 %** |

Reproduces at all three contexts, so it is a signal rather than noise, and the sign tells you where it
lives: in the **independent** (traffic-bound) arm the ratio sits *below* 1.375× — traffic does not
scale with head count, so the extra compute hides behind it — while in the **drifting** (compute-only)
arm it sits *above*. **The penalty is entirely on the compute path and has nothing to do with
traffic.** Since 22 heads per rank is what this stack's TP=3 shape gives it, this is a real, standing
cost here, and it is invisible at the author's own 16-head shape — exactly the region his `5fd7299`
commit named as the only lever left on this kernel.

Scale check: his 262K/H=16 drifting reads 2,385.8 µs; ours reads 10,698.6 µs — 4.48×. The SM ratio is
188/48 = 3.92×, so per-SM the difference is 1.14×, which is a reasonable clock-speed gap. The kernel
is not doing anything unexpected on GB10.

## Verdict

| claim | H=16 (his shape) | H=22 (our TP=3 shape) |
|---|---|---|
| "within a few percent of drifting at 262K" | +5.4 % — the fixture's own mechanical gate (≤ 5 %) misses by 0.4 points, but reads as "a few percent" in plain language | **+1.3 % — held**, tighter than on his own card's +1.6 % |
| "not between drifting and independent" | **held** — the midpoint would be +73.4 %, production is +5.4 % | **held** — the midpoint would be +41.0 %, production is +1.3 % |
| mechanism transfers to a 48-SM / 24-MiB-L2 part | **yes** — 96.3 % of the distance closed | **yes** — 98.4 % |

**The claim holds on this part.** The item closes at zero: the "21–26 % overlap gap" quoted before
this measurement was a property of the independent selection pattern, not of production. This
repository never carried that figure in a published document, so there is nothing here to retract —
the correction is recorded because it was live in the thread this datum answers
([docs/11](../../docs/11-open-issues.md) §2.27). The H=16 residual, +5.4 %, is smaller than one
compulsory-miss pass over the working set and *shrinks* as head count rises (because it hides behind
compute) — that does not refute the residence-window mechanism, it confirms it.

## Status of the one caveat: the 22-head correctness check

The gap noted above — timed but not numerically verified at 22 heads — is now closed **in tooling**,
not yet in a GPU result. The same afternoon, the harness that produced this page's numbers was
extended so that the shape it times is the shape it verifies: `mla_decode` against a chunked fp32
torch MLA reference at the full 1,792-row production chunk, at both head counts, both contexts in the
table above, on both selection patterns, plus a kpool-aligned variant (this stack's indexer emits
selections on a granularity-4 grid) and a ragged-`seqlens` variant that exercises the masking path the
timing bench never touches. Acceptance is scale-normalised max relative error < 5e-2 **and**
per-vector cosine > 0.999.

**The reference itself was validated without a GPU first**, and passed `[measured-here]`: 24 checks,
CPU only — the chunked reference equals the single-shot one bit for bit at five different chunk
sizes, `seqlens == topk` equals no masking, a ragged mask equals truncating the selection outright,
the comparator catches a deliberately sign-flipped row at cosine −1.0000 and flags non-finite output,
every selection generator reproduces its expected overlap (drifting 0.999, production 0.926,
independent 0.028), and the kpool-aligned generator stays on-grid. **What is not yet measured is the
kernel call itself at 22 heads** — that needs the same engine-free window discipline as this page's
own run, and it is `[not tested]` until it has one. If the 22-head shape misbehaves the number will be
published as measured, not smoothed into a reassurance — this stack has an unrelated kernel that
silently miscomputed at exactly this shape before.

## What this cost

One GPU, 17 s, 1.65 GiB peak allocation, inside a window that was open anyway. No engine restart, no
configuration change, no image change — the production containers on all three nodes were down for
unrelated work throughout and came back up afterward unaffected. The two aborted attempts an hour
earlier cost nothing beyond a lock write and release. The follow-up correctness tooling's self-test
is CPU-only and free; the GPU half of it is priced the same as this page's own run, one more
engine-free window.

## Credits and provenance

The three arms and their selection generator are the `cuda-exl3` author's own, copied verbatim from
`bench/bench_mla_prefill.py` at his commit `5fd7299` under that project's MIT licence — see
[CREDITS.md](../../CREDITS.md) and [`../../bench/mla-prefill/README.md`](../../bench/mla-prefill/README.md)
for the exact attribution and what we added on top. The 0.926 overlap datum this fixture is calibrated
to was captured by the diagnostic hook in [`sm12-stack-patches-ab.md`](sm12-stack-patches-ab.md) §8.
The numbers on this page were summarised for [issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5).

## Raw

Not shipped separately: every record the JSON output contains is in the two tables above, and the run
log carries no information beyond what is quoted here. What is kept is the fixture itself —
[`../../bench/mla-prefill/`](../../bench/mla-prefill/) — which reproduces this table on any GB10 with
`cuda_exl3` importable, an idle or stopped engine, and about 2 GiB of headroom.

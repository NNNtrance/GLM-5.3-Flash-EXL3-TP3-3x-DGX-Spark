---
name: Measurement contribution
about: A number you measured on your own cluster, for a question this repository left open
title: "[measurement] "
labels: measurement
---

Thank you for running it. This template is the protocol from `docs/09-measurement-protocol.md`,
turned into a checklist. It is long because every line of it exists because a number without that
line was published here and later withdrawn — the retraction index is `docs/11-open-issues.md` §1.

`HELP-WANTED.md` is the ranked list of what a second cluster could settle, with the expected effort
for each. If your measurement is one of those items, name it.

## 0. What question does this answer

- `HELP-WANTED.md` item, or `CONTRIBUTING.md` item, or "something else":
- One sentence: what you expected, and what you got.

## 1. Rounds, and which were discarded

Tick one. If neither is true, the numbers are a signal, not a result — say that plainly and we will
still read them.

- [ ] **1 warm-up round + 3 measured rounds, median of the three.** Only valid with a persisted MLA
      tuner cache (`CUDA_EXL3_TUNE_CACHE` pointed at a directory that survives the container) that
      was **already warm from a previous boot of this image**. The boot that writes the cache is
      still a five-round boot. `docs/12-tuner-cache.md`.
- [ ] **5 measured rounds, rounds 1 and 2 discarded, median of 3-5.** The rule without a warm cache.

- Number of rounds actually run:
- Which were discarded:
- The individual rounds, not only the median:

The tuner's warm-up window here has been longer than two sweep rounds, and it once made the winning
arm look 25-45 % worse on the first pass (`docs/09` §1, `docs/14` §8.1).

## 2. The noise band your difference has to clear

Our within-boot round-to-round spread, warm cache, nothing changed between rounds:
**C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %** (`docs/09` §1.2). Boot-to-boot is larger still:
**15.9 % on C8 with nothing changed at all** (`docs/09` §2).

- Your difference, per metric:
- Does it clear the band: yes / no / it is inside the band and I am reporting it as **equal**
- Did you repeat it on a **second boot**: yes / no

A difference of 3 % or less is written down here as "equal": report the numbers, not a winner. If
the decision matters, one boot does not settle it.

## 3. Which prompt set, by name

- [ ] `scripts/hizset-v2.jsonl` — 12 short English code prompts. **Realistic.** This is what we
      publish.
- [ ] One of the four category sets (prose / code / math / JSON), via `scripts/category-speed.py`
- [ ] **Synthetic** ("count from 1 to 200" and friends). This measures the speculative-decoding
      *ceiling* and must be labelled as such wherever it appears.
- [ ] **Fresh** — prompts the engine has never seen. Prefill only, and mandatory for prefill.
- [ ] Your own set. Say how many prompts, what language, and roughly what they ask for.

Two things that are not optional here. **Prefill measured on a repeated prompt is not a prefill
measurement** — it reads the prefix cache and overstates by up to 55 % (`docs/09` §3). And prompt
language is a lever: the same task in English and in another language measured 54-63 against 41-47
tok/s on our sibling stack, because the drafter predicts English better.

## 4. The KV pool, read from a load boot with a settled host

- KV pool (`GPU KV cache size` from the boot log), tokens:
- `max_model_len` it was read at:
- Was the host settled before `docker run` — `MemAvailable` waited for, or `drop_caches` run:
- Do all ranks' "Free memory on device" lines agree within about 1 GiB:

This number is a **difference between two `/proc/meminfo` readings taken minutes apart, and it runs
backwards**: a node that starts dirty awards itself a larger pool. We priced 8.2 GiB of "stranded"
memory off it and were wrong (`docs/07-kv-and-draft-page.md` §1.1, `docs/14` §5.8). A pool figure
from a boot that followed a large container teardown, with no settle wait, is not comparable to
ours.

Do not quote a pool figure from a `FASTLOAD_MODE=dump` boot as a result.

## 5. Provenance label

Put one of these on every claim, as defined in `STYLE-GUIDE.md`:

`[measured-here]` · `[measured-here, raw lost]` · `[reported]` · `[estimate]` · `[not tested]` ·
`[retracted]`

- Label for your headline number:

## 6. Settings block

Every number needs all of these. If one is missing the report is incomplete, and we would rather
have it late and complete:

| | |
|---|---|
| Node count, TP, EP on/off | |
| Image tag and `cuda-exl3` commit | |
| Checkpoint repository, branch, revision, scope | |
| Draft method and `k` | |
| `--kv-cache-dtype` | |
| `HAREM_DRAFT_KV_DTYPE` | |
| `gpu-memory-utilization` | |
| `--block-size` | |
| `HAREM_SW_BLOCK_SIZE` | |
| `--max-num-batched-tokens` | |
| `--max-num-seqs` | |
| `--max-model-len` | |
| `NCCL_MAX_NCHANNELS`, `NCCL_ALGO` | |
| Mesh plugin build and `NCCL_MESH_*` settings | |
| `CUDA_EXL3_TUNE_CACHE` set, and warm | |
| Fast-load sidecar: dump / load / none | |
| Temperature, reasoning effort, `max_tokens` | |
| Concurrency levels | |
| Prompt type (realistic / synthetic / fresh) | |
| Rounds, and which discarded | |
| Date | |

## 7. The gates, cold and warm

A speed number without them is not a result here (`docs/09` §5).

- `scripts/correctness-probe.py` cold: __/10 · warm: __/10
- `scripts/code-exam.py` cold: __/12 · warm: __/12
- Empty completions:

The defect class this stack produces hides on a fresh allocator, so a cold-only pass proves nothing.

## 8. What it cost

Speed, quality **and** memory, together. If it genuinely cost nothing, say that you looked and give
the numbers that show it. This line is never left empty.

- Speed:
- Quality:
- KV pool / host memory / swap:

## 9. CSV rows

Paste rows in the shape of `results/speed/concurrency-sweeps.csv` so they can be read beside ours.
`results/README.md` §"Reading a sweep row" says what each column means, and which one decides
throughput.

```
arm,conc,rounds,agg_tok_s,agg_min,agg_max,per_stream_tok_s,ttft_med_s,ttft_max_s,tpot_ms,accept_rate_pct,accept_len,kv_pool_tokens
```

```
paste your rows here
```

If you would rather open a pull request adding the raw output under
`results/community/<your-handle>/<item>/`, that is better still — `CONTRIBUTING.md` has the format
and the pull request template has the rest. We will credit you in `CREDITS.md`.

# results — the measurements behind the documents

Every number in [`../docs/10-results-and-roofline.md`](../docs/10-results-and-roofline.md) and in the
rest of `docs/` traces to a file in here, or carries `[measured-here, raw lost]`, `[reported]`,
`[estimate]`, `[not tested]` or `[retracted]` in the document.

All of it was produced on 4–5 September 2026 on three DGX Spark (GB10) nodes: TP=3 with expert
parallelism, KV `fp8`, DFlash2 draft at k=7, temperature 0, thinking on at `reasoning_effort: low`.
**The checkpoint changed at production configuration 9:** everything up to and including production 8
is `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at revision `b20c49ba` (routed experts only); production 9
and 10 are `turboderp/GLM-5.3-Flash-exl3` branch `4.05bpw` at `2a30229e` (full scope). Each file's
header names which. CUDA graphs are
**off** on the recent arms and were on earlier — not a setting of ours either way, see
[docs/10](../docs/10-results-and-roofline.md) §5.8. The exact settings per arm are in each file's
header.

## What is here

| Path | What it is |
|---|---|
| `speed/concurrency-sweeps.csv` | Every arm × C1/C2/C4/C6/C8: aggregate and per-stream tok/s, the round-3-to-5 spread, TTFT, TPOT, draft acceptance, accepted tokens per step, KV pool. Machine-readable. |
| `speed/concurrency-sweeps.md` | The same table, readable, with a line on what each arm is. |
| `speed/tp2-draft-page.md` | The draft KV page at **two** ranks, control against fix: the per-group block-counter decomposition from both boots, every sweep round, the long-prompt admission ceiling on each arm, and what the fix cost. |
| `speed/tp2-production-candidate.md` | **The TP=2 production candidate**, two complete configurations measured end to end on 6 September 2026: the pool arithmetic from each boot, four sweep rounds each, boot and sidecar figures, the per-port cable counters, every quality gate, and the autostart unit trial. The recommended candidate is the full-scope one. |
| `speed/category-prefill-and-mixed-load.md` | Decode rate by content type (prose / code / math / JSON) at C1 and C4; prefill, warm and fresh; the mixed-load probe; cold versus warm single stream. |
| `gates/quality-gates.md` | The correctness probe and code exam for every arm, cold and after the benchmark; the MMLU sample; long-form generation. |
| `boot/boot-ledger.md` | Boot time per phase across the four arms, the bit-identity evidence, the block-layer forensics, memory and swap. |
| `boot/tuner-cache.md` | The persisted MLA tuner cache: tune events per boot, round-1 against round-3, and why the sweep protocol got shorter. |
| `mesh/all-reduce-sweep.md` | The mesh all-reduce cliff: bandwidth by message size against point-to-point, the hardware counters, the channel-count dose–response, the protocol sweep, and the engine A/B. |
| `mesh/multilink-sweep.md` | The nine-arm plugin sweep: the idle second cable, `NCCL_PTR_CUDA`, what the flush and DMA-BUF cost, and why 16 channels is harmful. |
| `mesh/multilink-sweep.csv` | The same sweep, every arm × operation × size, with the per-collective RNR counters. Machine-readable. |
| `mesh/algo-sweep.md` | The `NCCL_ALGO` sweep: Ring against Ring,Tree against Tree, both repetitions printed, with the decode-step proxy, the RNR counters and the port deltas that show how a tree redistributes the traffic. |
| `kernels/kda-gate-bench-gb10.md` | **The live ratios.** The same gate bench re-taken on the target GPU with both arms rotated over a bank ≥ 4× L2: the warm/cold tables at M=1/8/64, the production trace used as referee, the achieved-bandwidth cross-check that shows which readings are the cache, the 4- and 6-bit projections, and the gate. **This supersedes the ratios in the file below.** |
| `kernels/gb10-coldbench/` | Its raw: the ruler, the full run with every round, the projection and bandwidth outputs, and a 531-row summary CSV (shape × M × arm × bit width × warm/cold). |
| `kernels/kda-gate-bench.md` | The first pass of the same bench, on a **workstation** GPU: bf16 against EXL3 4/6-bit at the TP=3 shapes, the ruler check that caught a 210 %-of-peak reading, the launch-cost table, the MLA fp32→bf16 lever, and the two `cuda-exl3` source facts behind the verdict. **Its ratios are `[retracted]`** — warm, on the wrong card — and the file is kept unedited as the record of what we published; the verdict it reached did not change. |
| `mesh/rdma-write-sweep.md` | The six-arm sweep of the one-sided `RDMA_WRITE` transport: RNR and out-of-buffer to exactly zero, throughput unmoved, the engine arm, and why it was not adopted. |
| `configs/production-configurations.csv` | One row per production configuration 1–9: what changed, C1/C8 aggregate and per-stream, prefill, TTFT, KV pool, acceptance, boot time, MMLU. The source for `charts/speed-by-configuration.svg`. |
| `configs/kv-pool-progression.csv` | Every KV pool reading on this stack, in order, including the rejected rungs, the dump-boot readings that must not be quoted as results, and the NVFP4 sibling's figures clearly labelled as the other stack. |
| `profile/step-breakdown.csv` | **Production 9's measured step breakdown**, every class × prefill/C1/C8 with milliseconds, call counts and the CUPTI caveat per row. The source for `charts/step-breakdown-prod9.svg`. |
| `profile/measured-prod7.md` | **The measured step-time breakdown** — torch profiler on the live server, all three ranks: prefill, C1 and C8 class by class, target versus draft, the collectives' zero overlap, rank arrival skew, and the C1 idle budget with the profiler's own cost subtracted. This is the one to read. |
| `profile/step-breakdown.md` | The earlier *reconciliation* of the same question, plus the model-free work that stands on its own: the two rulers measured on the device, the MoE stage model-free, the expert re-read bench, the hyper-connection kernels, the KV-zeroing geometry. **Its §3 and §4 are superseded** by the file above. |

## What is deliberately not here

- **Per-round raw JSON.** The published figures are medians — of rounds 3–5 with rounds 1–2 discarded
  on the images without a persisted tuner cache, and of three rounds on the ones with it
  ([docs/12](../docs/12-tuner-cache.md)). The medians are in the CSV and re-running
  `scripts/bench-sweep.py` reproduces them. Keeping 40 raw files whose only use is to be re-medianed added weight without adding
  evidence.
- **Boot and engine logs.** They carry host names and addresses throughout. What they proved is
  quoted as individual log lines in the documents, in context, with the surrounding text.
- **Profiler traces.** Large binary artefacts; the extracted kernel breakdowns are in
  [`../docs/05-expert-parallel-and-cuda-exl3-fixes.md`](../docs/05-expert-parallel-and-cuda-exl3-fixes.md) §4
  and in `profile/step-breakdown.md`.
- **Per-arm mesh sweep logs.** The published tables are the two-repetition means and the counter
  columns; the logs themselves carry node names and fabric addresses throughout.
- **Weights of any kind**, including the model sidecars and the fast-load sidecar.

## Scrubbing

These files are derived from our own run output. Node names were replaced with `head`, `worker-1` and
`worker-2`, addresses with documentation addresses, and local absolute paths with placeholders.
Turkish labels in our own runner logs were translated; the values are unchanged. Numbers, timings and
model output are untouched.

## Reading a sweep row

`agg_tok_s` is aggregate output throughput across all streams at that concurrency — the number that
describes the machine. `per_stream_tok_s` is the decode rate one user sees, excluding TTFT.
`accept_rate_pct` is the fraction of drafted tokens accepted; `accept_len` is accepted tokens per
speculative step. **`accept_len` is the one that decides throughput.** A shallower draft raises
`accept_rate_pct` and lowers `accept_len`, and the second effect wins wherever the target forward
dominates — that is the whole k=7 versus k=5 argument in
[`../docs/04-dflash2-port.md`](../docs/04-dflash2-port.md).

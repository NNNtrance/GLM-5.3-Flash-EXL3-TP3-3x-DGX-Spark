# Concurrency sweeps — medians of rounds 3–5

Source: `bench-sweep.py` with `scripts/hizset-v2.jsonl` (12 short English code prompts),
`reasoning_effort: low`, temperature 0, `max_tokens` 256 per request, five rounds per arm with
rounds 1–2 discarded as MLA-tuner warm-up ([docs/09](../../docs/09-measurement-protocol.md)).
All arms: three nodes, TP=3 + expert parallel, EXL3 4bpw, KV `fp8`, DFlash2 draft k=7,
`--block-size 256`, `--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, CUDA graphs on,
`gpu-memory-utilization 0.80` except where the arm says otherwise. 5 September 2026.

`agg_tok_s` is aggregate output throughput across all streams; `per_stream_tok_s` is decode rate
per stream excluding TTFT. `accept_len` is accepted tokens per speculative step — the number that
decides throughput, not `accept_rate_pct`.

## The arms

| Arm | What it is |
|---|---|
| bc0e0f6+0003, MNBT 4096 | upstream bc0e0f6 plus our combine staging patch, --max-num-batched-tokens 4096 |
| 61a17bc, input fusion off | upstream 61a17bc with CUDA_EXL3_MOE_FUSE_IN=0, MNBT 4096 |
| 61a17bc, input fusion auto | upstream 61a17bc with its own fusion heuristic, MNBT 4096 |
| f4987cf, MNBT 2048 | upstream f4987cf (skip MoE padding rows), --max-num-batched-tokens 2048 |
| f4987cf + draft page 256 | as above plus HAREM_SW_BLOCK_SIZE=256 (docs/07) |
| draft page 256 @ 0.85 | same, gpu-memory-utilization 0.85 - REJECTED, broke the free-memory rule |
| fast boot S1+S2+S3 | EP weight filter, eager safetensors, FlashInfer autotune off (docs/08) |
| fast boot S4 (production) | per-rank pre-sliced sidecar, full bit-identity verification (docs/08) |

## The numbers

| Arm | C | agg tok/s | spread | per stream | TTFT med (s) | TPOT (ms) | accept % | tok/step | KV pool |
|---|---|---|---|---|---|---|---|---|---|
| bc0e0f6+0003, MNBT 4096 | 1 | **51.1** | 51.0–51.2 | 61.3 | 0.683 | 16.32 | 64.3 | 5.5 | 1,627,170 |
| bc0e0f6+0003, MNBT 4096 | 2 | **74.9** | 71.9–76.8 | 45.2 | 0.856 | 22.14 | 63.7 | 5.46 | 1,627,170 |
| bc0e0f6+0003, MNBT 4096 | 4 | **104.8** | 102.9–111.5 | 34.4 | 1.119 | 29.08 | 62.0 | 5.34 | 1,627,170 |
| bc0e0f6+0003, MNBT 4096 | 6 | **125.4** | 122.9–130.3 | 27.2 | 1.401 | 36.73 | 60.1 | 5.2 | 1,627,170 |
| bc0e0f6+0003, MNBT 4096 | 8 | **150.9** | 148.6–153.4 | 23.9 | 1.555 | 41.9 | 60.9 | 5.26 | 1,627,170 |
| 61a17bc, input fusion off | 1 | **51.2** | 50.3–51.5 | 61.5 | 0.68 | 16.26 | 63.5 | 5.45 | 1,677,221 |
| 61a17bc, input fusion off | 2 | **75.7** | 74.4–76.6 | 46.1 | 0.866 | 21.71 | 63.4 | 5.44 | 1,677,221 |
| 61a17bc, input fusion off | 4 | **105.6** | 104.3–107.9 | 34.5 | 1.182 | 29.02 | 62.7 | 5.39 | 1,677,221 |
| 61a17bc, input fusion off | 6 | **129.4** | 126.3–130.5 | 26.9 | 1.36 | 37.19 | 60.6 | 5.24 | 1,677,221 |
| 61a17bc, input fusion off | 8 | **152.8** | 151.6–154.2 | 24.1 | 1.606 | 41.51 | 61.8 | 5.33 | 1,677,221 |
| 61a17bc, input fusion auto | 1 | **51.7** | 51.4–52.1 | 61.7 | 0.666 | 16.21 | 63.2 | 5.43 | 1,685,393 |
| 61a17bc, input fusion auto | 2 | **78.1** | 76.3–78.7 | 46.9 | 0.854 | 21.32 | 64.1 | 5.48 | 1,685,393 |
| 61a17bc, input fusion auto | 4 | **108.0** | 106.8–108.0 | 34.1 | 1.165 | 29.35 | 63.2 | 5.43 | 1,685,393 |
| 61a17bc, input fusion auto | 6 | **130.6** | 129.0–132.5 | 27.9 | 1.351 | 35.78 | 61.1 | 5.28 | 1,685,393 |
| 61a17bc, input fusion auto | 8 | **155.0** | 151.6–155.3 | 24.8 | 1.572 | 40.28 | 61.4 | 5.3 | 1,685,393 |
| f4987cf, MNBT 2048 | 1 | **51.9** | 51.1–52.3 | 61.9 | 0.662 | 16.17 | 63.8 | 5.46 | 2,428,769 |
| f4987cf, MNBT 2048 | 2 | **76.2** | 76.1–77.2 | 46.0 | 0.854 | 21.73 | 64.3 | 5.5 | 2,428,769 |
| f4987cf, MNBT 2048 | 4 | **107.0** | 105.0–107.2 | 32.7 | 1.192 | 30.58 | 61.6 | 5.31 | 2,428,769 |
| f4987cf, MNBT 2048 | 6 | **131.1** | 129.1–134.3 | 27.3 | 1.314 | 36.64 | 61.3 | 5.29 | 2,428,769 |
| f4987cf, MNBT 2048 | 8 | **153.8** | 153.4–154.4 | 24.1 | 1.459 | 41.49 | 61.4 | 5.3 | 2,428,769 |
| f4987cf + draft page 256 | 1 | **52.8** | 52.3–53.5 | 61.1 | 0.469 | 16.38 | 61.2 | 5.28 | 4,413,223 |
| f4987cf + draft page 256 | 2 | **81.0** | 77.2–81.0 | 47.6 | 0.692 | 21.02 | 64.2 | 5.5 | 4,413,223 |
| f4987cf + draft page 256 | 4 | **117.1** | 113.3–120.2 | 35.6 | 0.937 | 28.06 | 63.5 | 5.45 | 4,413,223 |
| f4987cf + draft page 256 | 6 | **135.6** | 134.1–141.8 | 28.0 | 1.022 | 35.69 | 60.5 | 5.24 | 4,413,223 |
| f4987cf + draft page 256 | 8 | **162.8** | 162.0–164.7 | 25.5 | 1.136 | 39.19 | 61.6 | 5.31 | 4,413,223 |
| draft page 256 @ 0.85 | 1 | **53.8** | 53.8–54.0 | 61.3 | 0.468 | 16.31 | 63.1 | 5.41 | 5,256,198 |
| draft page 256 @ 0.85 | 2 | **79.5** | 79.0–81.2 | 47.1 | 0.644 | 21.21 | 63.4 | 5.44 | 5,256,198 |
| draft page 256 @ 0.85 | 4 | **112.7** | 111.4–114.3 | 36.4 | 0.928 | 27.49 | 62.4 | 5.37 | 5,256,198 |
| draft page 256 @ 0.85 | 6 | **135.0** | 134.6–139.1 | 28.1 | 0.917 | 35.55 | 60.9 | 5.27 | 5,256,198 |
| draft page 256 @ 0.85 | 8 | **161.8** | 161.4–163.7 | 25.0 | 1.151 | 40.0 | 61.2 | 5.29 | 5,256,198 |
| fast boot S1+S2+S3 | 1 | **53.8** | 52.8–54.5 | 60.8 | 0.465 | 16.45 | 63.4 | 5.44 | 4,231,404 |
| fast boot S1+S2+S3 | 2 | **80.4** | 78.5–80.4 | 47.7 | 0.706 | 20.95 | 62.6 | 5.38 | 4,231,404 |
| fast boot S1+S2+S3 | 4 | **112.9** | 109.7–114.7 | 35.3 | 0.849 | 28.35 | 62.0 | 5.34 | 4,231,404 |
| fast boot S1+S2+S3 | 6 | **136.5** | 136.2–136.6 | 27.9 | 0.905 | 35.83 | 61.0 | 5.27 | 4,231,404 |
| fast boot S1+S2+S3 | 8 | **164.0** | 160.8–166.1 | 25.5 | 1.149 | 39.15 | 62.1 | 5.34 | 4,231,404 |
| fast boot S4 (production) | 1 | **54.4** | 54.2–54.5 | 61.8 | 0.469 | 16.19 | 63.9 | 5.47 | 4,468,319 |
| fast boot S4 (production) | 2 | **80.1** | 79.6–80.8 | 47.3 | 0.689 | 21.15 | 63.3 | 5.43 | 4,468,319 |
| fast boot S4 (production) | 4 | **114.6** | 113.5–115.6 | 36.3 | 0.927 | 27.53 | 62.8 | 5.4 | 4,468,319 |
| fast boot S4 (production) | 6 | **136.8** | 135.9–137.4 | 28.5 | 0.975 | 35.09 | 60.5 | 5.24 | 4,468,319 |
| fast boot S4 (production) | 8 | **161.8** | 161.3–163.4 | 25.2 | 1.142 | 39.65 | 61.6 | 5.31 | 4,468,319 |

Raw per-round JSON is not included in this repository; the medians above are the published
figures. Re-running `scripts/bench-sweep.py` five times reproduces them.

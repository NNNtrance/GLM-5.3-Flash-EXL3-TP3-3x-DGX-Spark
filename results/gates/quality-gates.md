# Quality gates — every arm, cold and warm

Two cheap tests. `scripts/correctness-probe.py` asks ten short questions with machine-checkable
answers plus one long paragraph checked for a repetition lock; `scripts/code-exam.py` runs twelve
short coding tasks and checks the output. Both were run **cold** after each boot and again **after**
the full benchmark on a warm allocator.

The warm run is the one that carries weight. The class of defect this stack has actually produced — a
kernel writing rows nothing initialises, and a combine summing them — hides on a fresh engine,
because a fresh caching allocator hands out zeroed pages. See
[docs/09](../../docs/09-measurement-protocol.md) §5.

| Arm | probe cold | code cold | probe warm | code warm |
|---|---|---|---|---|
| TP=2, no draft | 10/10 | 12/12 | — | — |
| TP=2, MTP k=3 | 10/10 | 12/12 | — | — |
| TP=2, DFlash2 k=7 | 10/10 | 12/12 | — | — |
| TP=3 + EP, stock kernels | 10/10 | 12/12 | **10/10** | **12/12** |
| TP=3 + EP, `n_rows` fix | 10/10 | 12/12 | **10/10** | **12/12** |
| upstream `bc0e0f6` | 10/10 | 12/12 | **10/10** | **12/12** |
| `bc0e0f6` + combine staging | 10/10 | 12/12 | **10/10** | **12/12** |
| `--max-num-batched-tokens 4096` | 10/10 | 12/12 | **10/10** | **12/12** |
| draft depth k=5 | 10/10 | 12/12 | **10/10** | **12/12** |
| `NCCL_MAX_NCHANNELS=8` | 10/10 | 12/12 | **10/10** | **12/12** |
| `61a17bc`, fusion off | 10/10 | 12/12 | **10/10** | **12/12** |
| `61a17bc`, fusion auto | 10/10 | 12/12 | **10/10** | **12/12** |
| `f4987cf`, MNBT 2048 | 10/10 | 12/12 | **10/10** | **12/12** |
| `f4987cf` + draft page 256 | 10/10 | 12/12 | **10/10** | **12/12** |
| draft page 256 @ 0.85 (rejected on memory) | 10/10 | 12/12 | **10/10** | **12/12** |
| fast boot S1+S2+S3 | 10/10 | 12/12 | **10/10** | **12/12** |
| fast boot, sidecar dump | 10/10 | 12/12 | — | — |
| fast boot, S4 (per-rank sidecar) | 10/10 | 12/12 | **10/10** | **12/12** |
| tuner cache warm (`9bf594c`) | 10/10 | 12/12 | **10/10** | **12/12** |
| production 6 (dual cable + `PTR_CUDA`) | 10/10 | 12/12 | **10/10** | **12/12** |
| draft KV fp8, dump boot | 10/10 | 12/12 | **10/10** | **12/12** |
| production 7 (fp8 draft cache, load boot) | 10/10 | 12/12 | 10/10 | 12/12 |
| production 8 (image `62f53e6`) | 10/10 | 12/12 | 10/10 | 12/12 |
| **production 9 (full-scope checkpoint, `754421f`)** | **10/10** | **12/12** | **10/10** | **12/12** |
| **production 10 (production 9 at `gpu-memory-utilization` 0.83)** | **10/10** | **12/12** | **10/10** | **12/12** |
| production 10 + `NCCL_ALGO=Ring,Tree` (A/B arm, rejected) | 10/10 | 12/12 | 10/10 | 12/12 |
| **production 10 after a whole-cluster reboot** (autostart unit) | — | — | **10/10** | **12/12** |

Full marks everywhere, in every arm, in both states `[measured-here]`. The reboot row has no cold
column because the engine had already answered the health check by the time the gates ran; what it
proves is narrower and worth having anyway — that a cluster which started itself from power-on,
with no human in the loop, answers correctly ([`../boot/boot-ledger.md`](../boot/boot-ledger.md),
[systemd](../../systemd/README.md)).

Two counts are reported by the probe on purpose: `content` is what a plain client sees, `both` is
what the model actually knows. With thinking on, this model sometimes puts the whole answer in the
reasoning field and leaves content empty, at temperature 0. Production reads:

```
RESULT (model knowledge, both fields): 10/10
RESULT (content only, what a client sees): 9/9
requests with EMPTY content: 0
```

```
CODE EXAM: 12/12  (100.0%)
```

## Broader quality

| Test | Result | Where |
|---|---|---|
| MMLU sample, 35 questions per subject (1,995 questions) | **86.47 ±0.74** | **production 9, TP=3, on the production full-scope checkpoint** `[measured-here]` |
| MMLU sample, same protocol | 86.4 ±0.7 | the earlier figure, measured at **TP=2** on the experts-only checkpoint, 22 minutes `[measured-here]` |
| Long-form generation | 190 words, 62 % lexical variety, no repetition lock | TP=3 + EP `[measured-here]` |

The two MMLU figures are **0.07 points apart, a tenth of either bar** — but they are not a like-for-
like pair, and the difference is worth naming rather than averaging: different checkpoint, different
TP. Only production 9's was measured at TP=3 on what production serves; the 86.4 ±0.7 that
configurations 7 and 8 carry is the TP=2 number **carried forward**, not re-measured on those arms
`[not tested]`. Production 10 was not re-measured either, because it differs from production 9 by a
memory fraction.

Nothing on this page was measured at max reasoning effort.

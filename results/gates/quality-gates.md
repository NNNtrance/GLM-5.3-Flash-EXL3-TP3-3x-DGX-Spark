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

## Production 11, both boots, plus two gates the older arms never ran

Production configuration 11 (`gpu-memory-utilization` 0.87 + the sm_12x correctness set) was gated on
two boots — the load boot and the clean boot after a whole-cluster reboot — beside a same-session
production 10 reference measured on the engine as the memory-ladder campaign left it. All three arms
also ran the **tool-call gate** (`toolcall-gate.py`, eight checks: a well-formed call, tool selection,
no spurious call, no `None` leak from a `content: null` assistant turn, answering from the result,
out-of-order results, a second-round call, argument fidelity) and a **needle-lite** probe (three
depths at ~64K and three at ~128K prompt tokens, thinking on at `reasoning_effort: low`), 6 September
2026 `[measured-here]`:

| Arm | probe cold | code cold | probe warm | code warm | tool-call | needle-lite |
|---|---|---|---|---|---|---|
| production 10 reference, same session, 0.83 | 10/10 | 12/12 | **10/10** | **12/12** | **8/8** | **6/6** |
| production 11, load boot | 10/10 | 12/12 | **10/10** | **12/12** | **8/8** | **6/6** |
| production 11, clean boot after the triple reboot | 10/10 | 12/12 | **10/10** | **12/12** | **8/8** | **6/6** |

Every rung of the memory ladder those figures came out of — 0.85, 0.87, 0.88 and the **rejected**
0.90 — also passed 10/10 and 12/12 cold and warm
([`../memory/ladder-6sep.md`](../memory/ladder-6sep.md)). The gates do not see the failure the ladder
was rejected on, which is the point of measuring swap traffic instead.

## Broader quality

**The three-benchmark battery of 6–7 September is its own page**:
[`quality-battery-production-12.md`](quality-battery-production-12.md) — GSM8K, IFEval and
tool-eval-bench on production 12 against the NVFP4 sibling recipe, with the scenario breakdown behind
the tool-eval number, the chat-template hypothesis tested and refuted, and what was deferred. This
page stays what it has always been: the per-arm gate ledger.

| Test | Result | Where |
|---|---|---|
| GSM8K, 200 questions, 8-shot CoT | **97.5 %** (195/200) | **production 12**, 6 September; the sibling recipe reads 94.0 % `[measured-here]` |
| IFEval, 541 prompts, 25 constraint types | **80.0 %** prompt · **86.0 %** instruction | **production 12**, 6–7 September; the sibling reads 78.9 / 85.1 `[measured-here]` |
| tool-eval-bench, hardmode, 88 scenarios × 8 trials | **85.5 ±1.3** (`final_score` 86) | **production 12**, 6 September; the sibling reads 87.8 ±0.9. Four scenarios out of 88 carry the whole difference `[measured-here]` |
| MMLU sample, 35 questions per subject (1,995 questions) | **86.47 ±0.74** | **production 9, TP=3, on the production full-scope checkpoint** `[measured-here]` |
| MMLU sample, same protocol | 86.4 ±0.7 | the earlier figure, measured at **TP=2** on the experts-only checkpoint, 22 minutes `[measured-here]` |
| Long-form generation | 190 words, 62 % lexical variety, no repetition lock | TP=3 + EP `[measured-here]` |

The two MMLU figures are **0.07 points apart, a tenth of either bar** — but they are not a like-for-
like pair, and the difference is worth naming rather than averaging: different checkpoint, different
TP. Only production 9's was measured at TP=3 on what production serves; the 86.4 ±0.7 that
configurations 7 and 8 carry is the TP=2 number **carried forward**, not re-measured on those arms
`[not tested]`. Production 10 was not re-measured either, because it differs from production 9 by a
memory fraction. Production 11 was not re-measured either, for the same reason plus one more: the two
sm_12x patches it adds are a buffer initialisation and a bounds check, neither of which touches the
arithmetic of a correct row `[not tested]`. **Production 12 was not re-measured on MMLU either**, for
the same reason again — it differs from 11 by one buffer size — and the full 14,042-question MMLU was
staged in the 6 September battery and deferred before it started, so the sample above is still the
only MMLU this stack has `[not tested]`.

Nothing on this page was measured at max reasoning effort.

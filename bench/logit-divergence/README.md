# `logit-divergence` — a race detector that works on a stack which is not bit-exact

**Applies to: both tracks, and to any vLLM OpenAI-compatible endpoint.** It talks HTTP; it does not
care how many ranks are behind the port.

This is the probe that replaced byte-for-byte determinism checking on this stack. It answers one
question — **"is the difference between two engine configurations larger than the difference between
two runs of the same configuration?"** — and it answers it with numbers rather than with a yes/no
that a non-deterministic stack cannot give.

Used in [`results/kernels/sm12-stack-patches-ab.md`](../../results/kernels/sm12-stack-patches-ab.md)
to test for a PDL launch-ordering race, and named as the first step of the ReplaySSM A/B in
[HELP-WANTED](../../HELP-WANTED.md) §6.

```
logit-divergence.py             the probe: `run` and `compare`
prompts-logit-divergence.jsonl  its fixed 12-prompt set (4 code, 4 prose, 2 math, 2 JSON/tool)
```

---

## Why byte identity does not work here

The obvious test — same prompt, temperature 0, fixed seed, run it N times, any difference is a race —
has **no discriminating power on this stack**, and that was established before it cost a boot. The
`cuda-exl3` author ran it on his own hardware first and got **24 distinct completions out of 24
greedy runs, in both arms** `[reported]`. Three independent causes, none of them the thing being
looked for:

1. `cuda-exl3`'s fused MoE epilogue accumulates each routed row into its token's row with a bf16
   `atomicAdd`. Atomic order is not fixed, so the MoE stage is not bit-exact on **any** MoE model.
2. `CUDA_EXL3_DETERMINISTIC=1` did not cover that path until `e7e345e`; before it, the flag only
   disabled split-k on the dense path.
3. A TP≥2 all-reduce is not bit-exact either, and no flag changes that.

Measured per-layer relative error against an fp64 reference at `top_k=8` (this model's setting):
fused epilogue **4.34e-3**, `exl3_moe_combine` **1.67e-3** `[reported]`. That is the arithmetic floor
any real effect has to stand out from, and a run-to-run *varying* part of it exists regardless.

So the probe measures the **floor** first, and only then looks for something above it.

---

## What it measures

`run` sends the 12 prompts at temperature 0 with a fixed seed and `max_tokens` 128, asking for
`logprobs` with `top_logprobs=5`, and records for every generated position the chosen token, its
logprob and the top-5 alternatives. It repeats the whole set N times (default 2) and writes one JSON
per run into an output directory. **One directory is one arm.**

`compare` then computes, per prompt and per pair of runs:

| | |
|---|---|
| **first divergence position** | the first position where the chosen token differs, or "none". A length change with no earlier token difference is reported at the shorter length and marked `len` |
| **max \|Δlogprob\|** | the largest absolute difference in the chosen token's logprob **over the positions before the first divergence**. After it the two runs are on different trajectories and a positional comparison is meaningless |
| **median / p95 / max** | the distribution of \|Δlogprob\| over those same aligned positions — **this is the noise floor** |
| **top-5 max, top-5 set changes** | the same over alternatives present in both runs' top-5, and the positions where the two top-5 **sets** differ. A race can move a non-argmax logit visibly while the argmax survives |

Per arm it pools every per-position \|Δlogprob\| from every prompt into one distribution. **That
pooled p95 is the arm's floor.**

## The within-arm floor is the whole point

A between-arm number on its own means nothing on a stack like this. Two runs of the *same*
configuration already differ, and by how much is a property of the model, the collective and the
kernel library rather than of the change under test. **Measure the floor in both arms, pool it, and
only then ask whether the arms differ by more than that.**

The result that closed the PDL question was not "the arms are close". It was that the **between-arm
p95 sat below the within-arm p95** — the two configurations differed *less* than two runs of one of
them — with no prompt above the outlier threshold.

## The K rule

With two arms, `compare` repeats the whole analysis between **run 1 of arm A and run 1 of arm B** and
flags a prompt as an **outlier** when either:

- its between-arm max \|Δlogprob\| exceeds **K × the pooled within-arm p95** (K defaults to **4**), or
- it diverges in token sequence **earlier than any within-arm pair did**.

Exit code is `0` for no outliers, `1` for outliers found, `2` for an operational error, so it can gate
a script.

**Both halves of the rule matter and they catch different things.** The magnitude test catches a
change that perturbs a logit; the earliest-divergence test catches a change that perturbs the
*trajectory* while leaving the magnitudes ordinary. Reporting one without the other is how a
divergence probe misses the thing it was written for.

**K = 4 is a convention, not a derived threshold.** It was chosen before the data were seen and left
alone afterwards, which is the only property it has that matters. State the K you used.

## The `cache_salt` detail, without which run 2 measures nothing

Run 2 of an arm sends the same prompts as run 1, so it would be a **prefix-cache hit and skip prefill
entirely** — the two runs would not exercise the same code path, and any prefill-side effect would
hide behind the cache instead of showing up in the floor. The script therefore sends a **per-run
`cache_salt`** (vLLM's prefix-cache isolation field), which forces every run to recompute its prefill.

If the server rejects the field it is detected once, dropped, and recorded in the run metadata as
`cache_salt: false`. **Read such a result knowing run 2 was a cache hit**, and treat its floor as a
decode-only floor.

## Request shape, and two things it forces

`chat_template_kwargs = {"enable_thinking": true, "reasoning_effort": "low"}` — the same shape the
rest of this stack's harnesses use. This model has **no way to disable thinking** and
`enable_thinking: false` is never sent ([docs/09](../../docs/09-measurement-protocol.md) §7).

- Because thinking is on, a 128-token budget is often spent entirely inside the reasoning stream, so
  the `content` field is frequently empty. That is fine — it is still the model's own trajectory
  through the same kernels — but do not read the completions as answers.
- If the chat endpoint returns no usable logprobs, the script detects it **once** at startup and
  falls back to `/v1/completions` with `logprobs: 5`. That path sends the **raw prompt with no chat
  template**, so no system message and no reasoning directive apply: a legitimate measurement of the
  same kernels, but not the same prompt. It is recorded as `transport: "legacy_completions"` and
  `compare` **refuses to mix transports**.

Requests are strictly sequential and there is no concurrency flag. Concurrency changes batching, and
a divergence probe that mixes batch shapes measures batching rather than the kernel.

---

## Running it

Against an endpoint at `192.0.2.10:8000`, two arms that differ by one engine setting:

```bash
./logit-divergence.py run --host 192.0.2.10 --port 8000 --model glm-5.3-flash \
    --arm arm-a --runs 2 --out-dir ./ld/arm-a
```

Restart the engine with the setting changed — a knob read at import time needs a restart, not a
reload — then:

```bash
./logit-divergence.py run --host 192.0.2.10 --port 8000 --model glm-5.3-flash \
    --arm arm-b --runs 2 --out-dir ./ld/arm-b
```

```bash
./logit-divergence.py compare ./ld/arm-a ./ld/arm-b
```

`compare` also accepts two run JSON files, or two directories holding one run each, for a
single-arm floor measurement.

**Cost.** 12 prompts × 128 tokens × 2 runs at concurrency 1 is about **4 minutes** per arm on three
nodes at production 10 rates, plus the boot. Nothing else may be running: this is a measurement and
the lock rules apply ([docs/09](../../docs/09-measurement-protocol.md) §10).

## Reading a result honestly

**A clean result is not a proof.** "The between-arm maximum sits inside the within-arm floor" means
*this prompt set, at this token budget, at concurrency 1, did not separate the arms*. A race that
needs a particular batch shape, a longer generation or a different prompt distribution would not
appear. Report the prompt set, the token budget, the concurrency, the K and both floors — the number
alone is not a result.

**The prompt set is fixed on purpose** and its sha256 is printed in every report. Changing it changes
the floor, so a comparison against one of our numbers requires the same file.

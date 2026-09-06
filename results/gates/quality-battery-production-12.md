# The quality battery on production 12 — three tests run, two deferred

**The first broad quality battery this stack has run.** Until 6 September this repository had the two
cheap gates and one MMLU sample, and [11](../../docs/11-open-issues.md) §3 listed IFEval, GSM8K,
needle-in-a-haystack and tool-eval-bench as things only the NVFP4 sibling recipe had. Three of them
now exist on this stack, measured against that recipe's 3 September battery with the same harness at
the same flags. **Two came out ahead of the sibling and one came out 2.3 points behind**, and most of
this page is about the 2.3 points, because the number that moves the wrong way is the one worth
taking apart.

Settings, every test: **production configuration 12** — image `exl3-zeus:754421f`, the
`tracks/tp3/patches/` tree, TP=3 + expert parallel, `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw
(full scope), `kv-cache-dtype fp8` and an fp8 draft cache, DFlash2 draft at k=7, `--block-size 256`,
`HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, `gpu-memory-utilization 0.88`, the sm_12x correctness set, the indexer
workspace bound to 513 MB, `max_model_len 1,000,000`, warm MLA tuner cache, per-rank sidecar,
**temperature 0**, thinking on at **reasoning effort `low`**, `max_tokens 4096` (the harness fixes
it). One engine, one boot, no restart between tests: pre-check `/health` 200 and KV pool
**7,030,303** tokens, re-read unchanged before each test, and `/health` 200 again after the last one
`[measured-here]`.

Nothing on this page was measured at max reasoning effort. Reaching a defensible max-effort figure on
the slowest of these tests would cost days of cluster time, and the whole of this repository is at
`low` for that reason ([09](../../docs/09-measurement-protocol.md)).

---

## 1. The three tests

6 September 20:44 to 7 September 00:08 local time, 3 h 24 min of wall clock end to end. The reference
column is the NVFP4 sibling recipe's battery of 3 September 2026, same harness version, same flags,
same three nodes `[measured-here]`:

| Test | This stack (production 12) | NVFP4 sibling, 3 Sep | Duration | Note |
|---|---|---|---|---|
| **tool-eval-bench**, hardmode, 88 scenarios × 8 trials | `final_score` **86**; 8-trial mean **85.5 ±1.3**, median 85.5, CI95 [84.6, 86.2] | mean **87.8 ±0.9**, median 88.0, CI95 [87.1, 88.2] | 1 h 27 min 18 s | **−2.3 points.** Real, not trial noise, and it is four scenarios out of 88 — §3 |
| **GSM8K**, 200 questions, 8-shot CoT | **97.5 %** (195/200), 401.1 s, 164,249 tokens | 94.0 % (188/200) | 6 min 45 s | **+3.5 points.** 5 wrong answers, all numeric; no format failures |
| **IFEval**, all 541 prompts, 25 constraint types | prompt **80.0 %** (433/541), instruction **86.0 %** (717/834), 6,270.1 s, 262,585 tokens | prompt 78.9 % (427/541), instruction 85.1 % (710/834) | 1 h 44 min 35 s | **Ahead on both axes.** 541/541 prompts completed, no retry or traceback in the log |
| **Needle-in-a-haystack at 1M** | **deferred** — §6 | 20/20 (100 %) | — | Command staged, not run |
| **MMLU, full 14,042 questions** | **deferred** — §6 | 85.9 ±0.3 (humanities 80.9 · STEM 85.9 · social 91.5 · other 88.1) | — | The 1,995-question sample at TP=3 on this checkpoint, **86.47 ±0.74**, stands and is unaffected ([`quality-gates.md`](quality-gates.md)) |
| **ExtractBench Short** | **not run** `[not tested]` | **not run on that build either** `[not tested]` | — | Neither stack has it. The sibling recipe carries a 94.51 against an H100 FP8 reference's 96.46 on the 215-document set from an **earlier, different** build, and says plainly that it must not be quoted as a figure for the build it ships `[reported]` |

Two side notes on GSM8K and IFEval, recorded because they are the kind of thing that gets
believed later. **The harness's `--json-file` writes nothing in `--gsm8k-only` and `--ifeval-only`
modes** — the flag is accepted and ignored, in both modes, on this version. The figures above come
from the console output and the harness's own Markdown report, which agree digit for digit; nothing
was lost, but a run script that checks only for the JSON would report a failure that did not happen.
And IFEval is issued **one prompt at a time** regardless of engine concurrency, so its 1 h 45 min is a
function of 541 prompts rather than of this stack's throughput: there is no speed lever on it.

**Thermals and clocks were sampled on all three nodes for the whole battery**, at 2 s, and the
throttle mask was `0x0` on **every sample of every node in every test** — 0/1038, 0/106 and 0/1251
samples `[measured-here]`. Peak GPU temperature 83 °C, average board power 41–60 W, peak 80.4 W. No
number on this page is a thermal artefact.

---

## 2. Harness parity, and the one break in it

Both batteries ran `tool-eval-bench` **2.6.1.dev39+gd3352edf5**. Identical across the two result
files: `temperature` 0.0, `seed` 42, `timeout_seconds` 120, `max_turns` 8, `concurrency` 1
(sequential), `scenario_count` 88 with the same `scenario_ids`, hardmode on, `error_rate` 0.0,
`alpha` 0.7, `trials` 8, `max_model_len` 1,000,000, `reference_date` null (so both resolve to the
harness default), and the fixed system prompt the harness prepends to every conversation. The request
body is the same on both: `max_tokens 4096` hard-coded in the adapter, `tool_choice: "auto"`,
**`parallel_tool_calls: true`** whenever the tool list is non-empty, `top_p`/`top_k`/`min_p`/
`repetition_penalty`/`stop` never sent. `max_points` is **176** (88 × 2) in both files, which is the
direct proof that **no scenario was excluded for a timeout, a connection error or a 5xx** in either
run — the scorer drops an infrastructure failure from both numerator and denominator and lists it, and
neither list has an entry.

Three fields differ, and only the middle two matter: the port, the **vLLM build**
(`0.1.dev20051+g487ecf187` here against `0.1.dev0+lil.jovian.9c4dd0548`), and the **checkpoint**. The
build difference carries a different speculative-decoding implementation with it. That confound is
the subject of §5.

One field is misleading and is worth naming: `metadata.thinking_enabled: true` is **not read back
from the server** — it is set from the absence of a `--no-think` flag. The evidence that thinking was
actually on is the reasoning field arriving on roughly 68 assistant turns per trial, in both runs.

A fourth difference is not in the result files at all — the **chat template**. It is §4.

---

## 3. tool-eval-bench: 84 of 88 scenarios are level, and the gap is four

### 3.1 The difference is real, and it is concentrated

The −2.3 points is not trial noise: 8 trials a side, Welch **t = 4.02** on 12.3 degrees of freedom,
exact permutation test over all 12,870 splits **p = 0.0048**, Cohen d 2.01 `[measured-here]`. The
distributions touch at the ends — this stack's best trial is 88, the sibling's worst is 86 — and the
means are four times the noise apart.

It is also concentrated to the point where the aggregate is almost the wrong statistic. Per-trial
score with scenarios removed, both stacks:

| Removed | This stack | NVFP4 sibling | Difference |
|---|---|---|---|
| nothing (all 88) | 85.23 ±1.32 | 87.50 ±0.86 | **−2.27** |
| TC-51 | 86.14 ±1.21 | 87.36 ±0.87 | −1.22 |
| TC-51, TC-21 | 87.14 ±1.22 | 87.79 ±0.88 | −0.65 |
| TC-51, TC-21, TC-74, TC-87 | **87.72 ±0.78** | **87.50 ±0.90** | **+0.22** |

**On the other 84 scenarios this stack is 0.2 points ahead.** Nine of the fourteen categories are
identical digit for digit between the two runs, including Tool Selection, Parameter Precision,
Instruction Following, Code Patterns and Structured Output at 100 % on both. One category collapses —
Autonomous Planning, 79.0 → 37.2 — and it is a three-scenario category, half of it TC-51.

Full-pass counts per scenario over 8 trials, the seven that moved by two trials or more:

| Scenario | Title | Sibling → here | Points/trial Δ | Recorded failure class |
|---|---|---|---|---|
| **TC-51** | Goal-Level Planning | 8 → **0** | −1.875 | `wrong_args` — §3.2 |
| TC-21 | Constraint Validation | 3 → 0 | −1.000 | `missing_step`; a **tool-free** analysis task |
| TC-74 | Stateful Multi-Turn Corrections | 8 → 5 | −0.750 | `wrong_args` — acted before the user's go-ahead |
| TC-87 | Complete Pagination With Cursor Integrity | 8 → 5 | −0.750 | `wrong_args` — changed its own filter format mid-pagination |
| TC-14 | Malformed Response | 4 → 0 | −0.500 | none (pass → partial) — no alternative source tried |
| TC-53 | Conditional Planning | 3 → 0 | −0.375 | none (pass → partial) — noted the condition, took no action |
| TC-52 | Open-Ended Research | 3 → 1 | −0.250 | none (pass → partial) — no comparison benchmark fetched |

And the five that moved the other way: TC-35 Contradictory Parameters 4 → 8, TC-88 Preserved
Reasoning Across Follow-Ups 0 → 2, TC-58 Fake System Message in File 6 → 8, TC-48 Additive Context
6 → 8, TC-47 Correction Across Turns 0 → 2. Negative total −6.0 points per trial, positive +2.0, net
−4.0 per trial on a 176-point scale = **−2.27 per 100**.

### 3.2 TC-51 is 47 % of the gap, and it is a scoring rule

TC-51 asks the model to organise a team lunch. Both stacks find the contacts, both create the calendar
event, both send the invitation to the same three people at the same time on the same day. The sibling
does it in three turns; this stack does it in two, issuing `create_calendar_event` and `send_email`
**in the same assistant turn**, in parallel — which the API layer explicitly invites, since the
harness sends `parallel_tool_calls: true` on every request.

The grader is deterministic Python per scenario, not an LLM judge, and TC-51's rule is a strict
ordering:

```python
if call.turn <= (valid_event.turn if valid_event else -1) or not recipients:
    notifications_valid = False
```

Same turn means `call.turn == valid_event.turn`, which means `notifications_valid = False`, which
means an immediate fail — with correct arguments, correct recipients, correct date, correct duration.
Seven of this stack's eight trials batch the two calls and all seven fail; the one trial that does not
batch scores partial. The harness's own diagnostics line, present here and absent for the sibling,
says it outright: `TC-51: parallel tool turns: 2`.

So the largest single item in the gap is **not a quality regression**. It is a mismatch between the
harness's ordering assumption and this stack's preference for batching two dependent calls into one
turn. Parallel calling is not penalised in general — TC-09 Parallel Independence wants it and both
stacks score 8/8 — and this is a rule specific to TC-51.

Two of the other three are real, and worth stating as such rather than explained away. **TC-74**: the
prompt says not to create the event until told to, and the failing trials create it and send a
confirmation — a genuine instruction violation. **TC-87**: the model asks for `"2025-Q3"` on the
first page and `"Q3"` on the second, and the grader's filter-consistency check fails — a genuine
argument inconsistency. **TC-21** is a tool-free analysis task where the sibling was itself erratic
(3 pass, 2 partial, 3 fail across its 8 trials) and this stack finds 0–1 of the 5 required validation
errors.

### 3.3 What never appeared, in 1,408 scenario-runs

Every failure code recorded across both batteries:

| Failure class | Sibling | This stack |
|---|---:|---:|
| `wrong_args` | 35 | 39 |
| `missing_step` | 9 | 20 |
| malformed or unparseable tool call | **0** | **0** |
| timeout | **0** | **0** |
| empty content | **0** | **0** |
| refusal | **0** | **0** |
| schema or formatting violation | **0** | **0** |

**Two codes, both behavioural.** Nothing was malformed, nothing timed out (the longest scenario ran
54.95 s against a 120 s limit, and the sibling's longest was 71.94 s), no 4xx or 5xx was returned,
and the empty-`content` failure mode this model is known for at effort `low` — the one the correctness
probe counts on every run ([09](../../docs/09-measurement-protocol.md) §5,
[`quality-gates.md`](quality-gates.md)) — did not occur once in the 16 trial reports on either side.
Structured Output is 100 % on both.

Latency, for what it says about the engine rather than the model: this stack reaches first token
~21 % later (TTFT p50 1,101 ms against 913) and finishes a turn ~10 % sooner (median turn 1,968 ms
against 2,182), with a much shorter tail (worst turn 10.8 s against 29.0 s), and generates 6 % fewer
completion tokens over the same 258 turns. `pass^k` — the share of scenarios passing in **all** eight
trials — is level at 75.0 against 76.1, and the reliability gap is **smaller** here, 8.0 against 10.3.
This stack is the more consistent of the two; its ceiling is two points lower.

The safety gate reads `passed: false` in **both** runs, and did so before this battery existed: both
stacks call `web_search` with an empty query on TC-43, 0/8 on both. The second warning that appears
only here, on TC-33, is an artefact of which trial the report happens to print — TC-33 is erratic on
both sides (6/8 against 5/8).

---

## 4. The template hypothesis, tested the same night and refuted

The obvious explanation, and the one this repository wrote down before testing it: the two batteries
were served with **different chat templates**. The sibling's run used the template that ships in its
checkpoint — `zai-org/GLM-5.3-Flash` at commit **`04c4e9e9`**, 27 August, sha256 `34d5ee66…`. This
stack mounts the 4 September template, **`690b7052`** (*"chat template: early break in tool result
reordering check"*), sha256 `0c4099f3…`. Every item in the diff between them is in tool-call
rendering, including the matching of tool results to multiple `tool_call`s in one turn — which is
precisely the behaviour family that regressed.

It was tested rather than argued about: production 12 was rebooted with the **old** template as the
single changed line, and the eight scenarios that moved were re-run at the same flags. One engine
window, 23 minutes.

The arm is clean. The environment file was derived on each node from that node's own production file
with `sed` and differs in one functional line; the image, tree, checkpoint, memory fraction and
fast-load sidecar are unchanged (the sidecar identity does not include the template path, so it stayed
valid and was verified loaded); boot took 274 s inside the configuration's own 217–275 s band; the KV
pool read 7,055,096 against production's 7,030,303 — the template does not touch memory. The template
actually in force was read **inside the container**, `sha256sum /models/chat_template.jinja` →
`34d5ee66…`, rather than inferred from a mount line `[measured-here]`.

| Scenario | Sibling, old template | Here, new template | Here, **old** template | Template effect |
|---|---:|---:|---:|---|
| **TC-51** | 8/8 (2.00) | 0/8 (0.12) | **0/8 (0.00)** | **none** |
| **TC-21** | 3/8 (1.00) | 0/8 (0.00) | **0/8 (0.00)** | **none** |
| TC-74 | 8/8 (2.00) | 5/8 (1.25) | 8/8 (2.00) | +0.75 (p = 0.20) |
| TC-87 | 8/8 (2.00) | 5/8 (1.25) | 6/8 (1.50) | +0.25 (p = 1.00) |
| TC-14 | 4/8 (1.50) | 0/8 (1.00) | 1/8 (1.12) | +0.12 (p = 1.00) |
| TC-53 | 3/8 (1.38) | 0/8 (1.00) | 0/8 (1.00) | none |
| TC-52 | 3/8 (1.38) | 1/8 (1.12) | 2/8 (1.25) | +0.12 (p = 1.00) |
| TC-35 (**control**, the scenario that *improved*) | 4/8 (1.50) | 8/8 (2.00) | 8/8 (2.00) | none |
| **eight-scenario block, max 16** | **12.75** | **7.75** | **8.88** | **+1.12** |

**TC-51 and TC-21 together are 72 % of the whole gap, and both are 0/8 on the old template too.** The
TC-51 trace is the same sentence and the same batched pair of calls, with the same
`parallel tool turns: 2` diagnostic. That part of the hypothesis is eliminated by direct measurement,
not by a p-value.

The remaining four scenarios do favour the old template by **+1.12 points per trial**, and that is
where the honesty has to be: the effect **cannot be separated from noise**. Exact permutation over
12,870 splits gives **p = 0.1984**; the 95 % confidence interval is **[−0.45, +2.70] points per
trial**, which on the 88-scenario /100 scale is **+0.64 points, CI [−0.26, +1.54]**. Per scenario,
nothing is significant either (Fisher two-sided: TC-74 p = 0.20, the other three p = 1.00). The
control behaved as a control should: the scenario that improved against the sibling improved on both
templates.

**As a share of the −2.3 points, the template's point estimate is 28 %, with a 95 % interval of
[−11 %, +68 %]** — an interval containing zero. Against the old template this stack still sits 3.88
points per trial below the sibling on this block, p = 0.0002.

**The recommendation is to keep the new template**, and the reason is not the measurement — an
unproven gain does not move a production setting — but what reverting would cost. The old template
carries three real defects, all on the hot path for an agent: an assistant turn with `tool_calls` and
`content: null` leaks a literal `None` into the prompt; `'<tool_call>' + tc.name` raises on a
non-string name where the new template's `~` does not; and the tool_response block has a nesting bug
in matching tool results to multiple `tool_call`s in one turn. Multi-step tool use, an empty-content
planning turn and parallel calls in one turn are all normal agent behaviour. **The gain is
unmeasurable and the risk is concrete.**

The old template did pass the gates — correctness probe 10/10, code exam 12/12 — so this is not a
rejection for breakage. Production was restored at the end of the window and re-verified: template
hash back to `0c4099f3…`, `/health` 200 after a 233 s boot, KV pool 7,165,289, gates 10/10 and 12/12
cold `[measured-here]`.

---

## 5. What the −2.3 points is, and what it is not

**It is not noise.** p = 0.0048 over an exact permutation test.

**It is not a broad quality regression.** Nine of fourteen categories are identical, 84 of 88
scenarios are level with this stack 0.2 ahead, the only two failure codes recorded are behavioural,
and the same engine on the same night scored **97.5 % on GSM8K against the sibling's 94.0 %** and beat
it on both IFEval axes. The MMLU sample on this checkpoint at TP=3 is 86.47 ±0.74, and the correctness
and code gates are full cold and warm.

**Roughly half of it is a grading rule, and that half is proven from source.** TC-51's 0/2 comes from
`call.turn <= valid_event.turn`, with the plan and the arguments correct.

**It is not attributable to the quantization** on this evidence — and equally, it is **not cleared**
of it. Three things changed between the two batteries and only one has been eliminated: the chat
template is out (§4, by measurement), and the **checkpoint** and the **vLLM build with its different
speculative-decoding implementation** are still varying together. Confidence that this stack's agentic
quality is not meaningfully below the sibling's: **high**. Confidence about where the residual
−1.6 points lives: **low**.

**What it means for someone running an agent on this stack.** The one behaviour that reproduced under
every arm is that this stack will batch two *dependent* tool calls into a single turn — create the
record and announce it, in one turn, without waiting to see the first result. That is efficient and
usually correct, and it is wrong wherever the second call must not happen if the first one failed. If
your framework grades or gates on turn ordering, or your tools are not idempotent, constrain it in the
system prompt or set `parallel_tool_calls: false`. Nothing in this repository has measured that
setting `[not tested]`.

**The next isolation step is the build, not the weights.** Same checkpoint, same template, this
cluster, `0.1.dev20051+g487ecf187` against the sibling's build as the only variable, on the eight
scenarios above rather than the full 88 — about 12 minutes of engine time per arm plus two boots. A
same-session NVFP4 control is not available: that line is closed and its stack no longer stands on
these nodes, so the separation has to be done from this side. Tracked in
[11](../../docs/11-open-issues.md) §2.30.

**And the aggregate should not be read alone.** The grader is deterministic, which removes judge
variance and is why 8 trials are enough — and it makes the rules strict enough that a scenario can
score 0 for doing the right thing in the wrong order. A tool-eval headline is a number to open with a
scenario breakdown, not a number to quote by itself. The cheap version of that breakdown is the eight
scenarios in §4: **about 11 minutes**, and it catches everything the full 1.5-hour battery caught.

---

## 6. What is deferred, and why

**Needle-in-a-haystack at 1M and the full 14,042-question MMLU were stopped before they started, by
the owner, on time.** The battery was scoped at 6–6.5 hours; three tests took 3 h 24 min and the
remaining two are the long ones. Both commands are staged and unmodified, and neither test was
started, failed or abandoned mid-run — this is a scheduling decision, not a result `[not tested]`.

What is *not* affected: the **1,995-question MMLU sample at TP=3 on this exact checkpoint, 86.47
±0.74**, was measured on production 9 and is unchanged by 10, 11 and 12, which differ from it by a
memory fraction, two guard patches and a buffer size ([`quality-gates.md`](quality-gates.md)). The
long-context evidence that exists on this configuration is the **needle-lite** probe (three depths at
~64K and three at ~128K, 6/6), one **969,468-token** request answered correctly, and eight concurrent
~128K lanes 8/8. What the deferred 1M needle would add is the top decade of that range against the
sibling's 20/20.

**ExtractBench has never been run on this stack** and remains in [11](../../docs/11-open-issues.md)
§3 — with one correction made while writing this page. That row used to say ExtractBench "exists for
the NVFP4 sibling recipe"; it does not. The sibling never ran it on the build it ships either, and the
94.51 it carries is from an earlier, different build with a note saying so. **Neither line has this
dimension**, which is a different sentence from the one we had published.

---

## 7. What this cost

**Nothing was traded for these numbers** — no setting changed, no patch shipped, and production 12 is
byte-identical before and after. The costs are the honest ones:

- **3 h 24 min of exclusive engine time**, held under the measurement lock, plus 23 minutes for the
  template arm and two boots to take production back.
- **Two of the five tests are missing**, by choice, and the one this repository would most like back
  is the 1M needle: the sibling has a 20/20 there and this stack has a 6/6 at a tenth of the length.
- **The headline moved down.** A repository that only published the two tests it won here would have
  reported a stack ahead of its sibling on every quality measure it has. The tool-eval number is
  2.3 points behind and it is in the table at the top of this page.
- **One residual is unresolved and stays unresolved in writing**: at least 72 % of the gap is
  checkpoint-or-build, and after a measurement designed specifically to split the confound, it is
  still one confound narrower rather than gone.

Raw material behind this page: the 88-scenario × 8-trial result file and its per-trial reports, the
eight-scenario template arm and its own reports, the GSM8K and IFEval console output and Markdown
reports, and the per-node telemetry series. Following this repository's rule for run output, what is
published here are the derived tables; the raw files carry node names and addresses throughout
([`../README.md`](../README.md)).

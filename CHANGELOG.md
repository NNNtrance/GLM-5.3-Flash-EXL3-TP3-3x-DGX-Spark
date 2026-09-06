# Changelog

Dated entries for the stack this repository documents. This is a working record, not a release
history — it is overwritten in place as the work continues.

Speed figures are aggregate output tok/s on `scripts/hizset-v2.jsonl` at `reasoning_effort: low`,
temperature 0. Entries up to and including *production configuration 4* are medians of sweep rounds
3–5 with rounds 1–2 discarded; from *production configuration 5* onwards they are medians of three
rounds, which is what the persisted MLA tuner cache bought — see
[docs/09](docs/09-measurement-protocol.md) §1 and [docs/12](docs/12-tuner-cache.md).

---

## 2026-09-06 — one repository, two tracks, and a front door in front of both

**This repository had grown a second working node count without growing a way to tell which half a
reader was in.** Nothing measured changed today. What changed is the shape: one folder per track for
the files that actually differ, a badge on every page saying which track it belongs to, a router that
asks how many nodes you have, and a ranked list of what a second cluster could settle.

**`tracks/`, and the rule for what goes in it.** A file belongs to a track only if it cannot be
shared: the environment template, the in-container patch tree, the autostart unit and its preflight.
`patches/tp3full/` is now [`tracks/tp3/patches/`](tracks/tp3/patches/) and `patches/tp2full/` is
[`tracks/tp2/patches/`](tracks/tp2/patches/), file for file, with the two units, the two preflights
and the three environment templates beside them. Everything else stays where it is and
[`tracks/README.md`](tracks/README.md) says why each shared thing is shared — `docs/`, `bench/`,
`scripts/` (both launchers, because the harness and the probes they sit beside are identical at
either rank count), `patches/kernel/`, `patches/dflash2-port/`, `patches/indexer-overlay/`,
`results/`, `charts/`, `audit/`, and the two directory pages `envs/README.md` and
`systemd/README.md`, which are rules rather than artefacts.

**One file was deleted and it was a duplicate.** `patches/tp2/patch-fullscope-tp2.py` was
**byte-identical** to the copy inside the two-node tree. This repository's own rule is that two
copies of a file are a coin flip unless something checks ([docs/08](docs/08-fast-boot.md) §12), so
the duplicate is gone and its five references now point at
[`tracks/tp2/patches/patch-fullscope-tp2.py`](tracks/tp2/patches/patch-fullscope-tp2.py). Nothing
else moved content; every other change in the move is a path.

**The trap the move introduces, said once and loudly.** The directory name in this repository is not
the directory name on your nodes. On a node the three-node tree is `~/exl3-zeus/tp3full/`, and that
name is load-bearing three times over: the launcher mounts `$TP3_DIR/tp3-prelude.sh` at `/start.sh`,
`tp3-prelude.sh` inside the directory is a **hard link** rather than a copy, and the directory's file
list and the full text of the prelude are hashed into the fast-load sidecar's identity
([docs/08](docs/08-fast-boot.md) §4). Renaming the tree on a node invalidates every sidecar and costs
a dump boot each. Renaming it here costs nothing, because the manifest hashes the directory you serve
from, not the one you cloned from.

**Every `docs/NN` page now opens with an "Applies to" line.** Three are not "both":
[03](docs/03-tp3-padding-and-sidecars.md) is TP=3 only, because all five shapes that do not divide by
three do divide by two; [10](docs/10-results-and-roofline.md) is TP=3, because its tables are the
three-node arms, though the two rulers it measured rather than quoted apply anywhere; and
[15](docs/15-tp2-track.md) is TP=2 only. The qualifiers on the "both" pages are the point of the
exercise — 05 applies to both but expert parallelism is mandatory at three ranks and optional at two,
08 applies to both but the sidecar is per rank, 13 applies to both but only its §7 is three ranks.

**[14](docs/14-troubleshooting.md) goes further: all 86 entries carry a track tag**, and the
distribution is the finding. 45 are plain **both**, 26 more are **both, measured at TP=3 only** —
the mechanism is rank-independent but every reading came from a three-node arm — so **71 of 86 are
things a two-node owner will meet**. Thirteen are TP=3 only and they cluster exactly where they
should: the four hard asserts, the 2,112/192 constants, padding heads to 96 instead of 66, the
sidecar mount trap, expert parallelism not on every rank, the tuple-shard loader, the three upstream
commits a padded load needs, the `n_rows` dispatch bug and its micro-benchmark, the 22-head guard,
and three silent-correctness entries that are all about a pad holding something it should not. Two
are TP=2 only — KV sizing refusing at `max_model_len` 1,000,000, and long prompts never being
scheduled — and at three ranks the pool was never small enough for either.

**New: [docs/00-start-here.md](docs/00-start-here.md)**, which asks how many nodes you have and
answers for all four cases. At **one** node there is no serving recipe and the page says so in
arithmetic rather than by implication — 153.8 GiB of weights against 121.6 GiB of unified memory —
and then lists what a single node still gets: the image build, the GB10 top-k overlay, the
measurement protocol, the model-free benches, the failure index, and the two `HELP-WANTED` items that
need one GPU and no fabric.

**New: [HELP-WANTED.md](HELP-WANTED.md)**, the ranked list, with the expected wall-clock effort on
every item and — where it matters — a line on what a contributor with fewer nodes than the item needs
can and cannot check. Eight items: four nodes at TP=4; a two-node reboot test, which is an hour and
closes a `[not tested]` a two-node owner meets on their first power cut; the memory ladder at two
ranks; other checkpoints, including the class where a tensor we have to pad is itself quantized; the
mesh plugin's small-message latency floor; the ReplaySSM compact rollback of the KDA state slots; the
KDA GEMM engine-against-standalone gap; and the four largest items already in
[11](docs/11-open-issues.md).

**Two of those items carry numbers that were not on any page here, and both say so.** The plugin's
small-message floor reads **74.7 µs at 8 KB** on the production build and is flat at 72–85 µs from
8 B to 32 KiB — pure latency, no bytes in it — while [06](docs/06-nccl-mesh.md) §5, taken with a
different harness on the pre-multilink configuration, reads **38.6 µs** for the same operation. A
factor of two between two of our own benches is not something to publish past, so the item asks for
both harnesses in one session before any conclusion is drawn. And the KDA state slots:
`MambaSpec.max_memory_usage_bytes = page × (2 + num_speculative_blocks)`, so at k=7 every KDA layer
holds **nine** state slots per request, seven of them purely for speculation — **9.9 % of the block
counter at TP=3 and 12.9 % at TP=2**. A pool model that reproduces four measured arms exactly
projects **+8.0 %** and **+9.6 %** from taking that to two. We have not run it and the reason is
written down: the replay loop triples the sequential work, KDA is about 8 % of a C8 step, and the
ring holds `d` and `k` in fp16 with no baseline comparison we could find in the upstream harness.

**A defect in one of our own gates, found while writing the TP=4 arithmetic and fixed here.**
`preflight-tp3.py`'s vocabulary check had three conditions and only tested two. 154,880 already
divides 4 and is already 1,210 × 128, so at `tp=4` the first branch returned "no patch needed" — but
**per rank** it is 38,720 = **302.5 × 128**, half a Hadamard block, which is exactly the defect
`lcm(64, 3) = 192` produces at three ranks and which [03](docs/03-tp3-padding-and-sidecars.md) §1.1
calls silently wrong. `lcm(128, tp)` is right at `tp=3` (384), right **by luck** at `tp=2` (154,880/2
is already 605 × 128) and wrong at `tp=4`, where the unit that works is `128 × tp` = 512, giving
155,136 and 303 × 128 per rank. The per-rank condition is now tested and a `tp=4` run is told what to
pad and why. **Repository copies only** — `tracks/tp3/patches/preflight-tp3.py` and
`tracks/tp2/patches/preflight-tp3.py`, which stay byte-identical to each other; the live trees on our
nodes were not touched. It is sidecar-safe by construction: `harem_fastload_id.py` hashes
`patch-*.py`, the prelude and the EP overlay, and `preflight-tp3.py` is none of those. Behaviour at
`tp=2` and `tp=3` is unchanged, verified by arithmetic; **`tp=4` is `[not tested]`** — we own three
nodes.

**Contributing, and four templates.** [CONTRIBUTING.md](CONTRIBUTING.md) gains a "how to contribute a
measurement" section naming three routes and what each becomes, and [`.github/`](.github/) gains
three issue templates and a pull request template that are [09](docs/09-measurement-protocol.md)
turned into checklists: rounds and which were discarded, the per-metric noise band a difference has
to clear, the prompt set named and labelled, the KV pool read from a load boot on a settled host, the
gates cold **and** warm, an evidence tier, the full settings block, and what the gain cost. Two
things are now said in the open rather than implied — a step that did **not** reproduce is as welcome
as a measurement, and a silent failure is worth more than a loud one; and a pull request that
withdraws one of our numbers is worth more than one that adds a number.

**Older entries on this page were rewritten in one respect and one only:** paths that moved now name
where the file actually is. The dates, the numbers and the reasoning are untouched.

**The repository was renamed** from `GLM-5.3-Flash-EXL3-TP3-3x-DGX-Spark` to
`GLM-5.3-Flash-EXL3-DGX-Spark`, because `TP3-3x` in the name was telling two-node owners this was not
for them. GitHub redirects the old URL; a clone with the old remote keeps working, and
`git remote set-url` is the tidy fix.

**No measurement changed today. Nothing was re-run and nothing is claimed that was not claimed
yesterday**, apart from the two `HELP-WANTED` figures above, which are labelled, and the preflight
gate, which is arithmetic.


## 2026-09-06 — a TP=2 production candidate, and the two-node page's second retraction

**The two-node track is now a complete recipe with a named production candidate, not a set of
bring-up arms.** A patch tree (`tracks/tp2/patches/`, thirteen files), an environment template
(`tracks/tp2/env.tp2-full.example`), a launcher (`scripts/start-tp2full.sh`), a per-rank fast-load sidecar
and an autostart unit (`tracks/tp2/harem-exl3-tp2.service`), all measured end to end on two nodes with
the protocol in [docs/09](docs/09-measurement-protocol.md) `[measured-here]`. Rewritten:
[docs/15](docs/15-tp2-track.md). Raw record:
[`results/speed/tp2-production-candidate.md`](results/speed/tp2-production-candidate.md).

**Two candidates were measured, identical except the checkpoint. The full-scope one wins every axis
and is the recommendation.** Both at `gpu-memory-utilization` 0.85, `max_model_len` 1,000,000, image
`exl3-zeus:754421f`, EP off, DFlash2 k=7, fp8 KV **and** fp8 draft cache, `HAREM_SW_BLOCK_SIZE=256`,
mesh plugin with both cables, warm tuner cache, fast-load sidecar, median of sweep rounds 2–4 of
four:

| | A — experts-only | **B — full-scope** | B vs A |
|---|---|---|---|
| KV pool at 1M | 1,500,000 | **2,128,571** | **+41.9 %** |
| C1 aggregate · per stream | 48.76 · 54.72 | **58.50 · 62.55** | +20.0 % · +14.3 % |
| C8 aggregate | 137.41 | **155.75** | +13.3 % |
| TTFT, C1 / C8 | 0.468 / 1.249 s | **0.407 / 1.077 s** | −13.0 % / −13.8 % |
| Consumed memory per node | 89.3 GiB | **84.8 GiB** | −4.5 GiB |
| Boot, fast-load | 272 s | 272 s | equal |
| Gates cold+warm · tool-call · needle-lite | 10/10 · 12/12 · 8/8 · 6/6 | 10/10 · 12/12 · 8/8 · 6/6 | equal |
| MMLU sample, 1,995 q | 86.37 ±0.74 | **86.02 ±0.75** | inside one error bar |

**`[retracted]`: "the full-scope checkpoint at two ranks is a rig and not a serving configuration".**
On 5 September it failed its KV budget gate at `max_model_len` 1,000,000 with
`6.6 GiB KV needed, available 0.73 GiB`, and at 65,536 produced a 31,343-token pool in which a
~2,800-token prompt was never scheduled ([docs/15](docs/15-tp2-track.md) §3.2). Neither reading
survives two changes made since. The draft page fix cuts blocks-per-request 640 → 280, so the 6.6 GiB
requirement becomes ~2.9 GiB; and the launcher's **settle gate** — a `MemAvailable ≥ 112 GiB` wait
before vLLM snapshots memory — turns 0.73 GiB of available KV memory into **16.07 GiB**. The settle
gate is the larger half, and it is a measurement bug fix rather than a tuning knob
([docs/07](docs/07-kv-and-draft-page.md) §1.1 already said a rank that snapshots a dirty host awards
itself memory it does not have; what is new is that the error was the difference between "cannot
serve" and "recommended").

**`[retracted]`: "the full-scope checkpoint is ~10 GiB heavier per node at two ranks".** Measured on
one boot each with the same settle gate, image and launcher: consumed memory (weights + non-torch) is
**89.3 / 89.2 GiB** experts-only against **84.8 / 84.5 GiB** full-scope. It is **4.5–4.7 GiB
lighter**, which agrees in sign and size with the three-node reading. The old figure came from a boot
with no settle gate.

**Four more rows left the two-node "not tested" list.** The **fast-load sidecar** at two ranks: boot
**997 s → 272 s**, sidecar 75–83 GB per rank (half the checkpoint, because EP is off, against a third
at three ranks with EP). The **fp8 draft cache**: **+15.1 % of pool**, isolated by arithmetic —
`SlidingWindowSpec` bytes/block halves, blocks-per-request is unchanged at 280, `num_blocks` goes
365 → 420 — and CUDA graphs still capture at two ranks, unlike at three. The **dual-cable mesh
patch** on a *single* peer pair: both devices per node moved ~90 GB across a sweep, split 50.5 / 49.5
— `patches/kernel/0005` is not a three-node effect. The **tuner-cache protocol**: round 1 is inside
±3 % of the median of rounds 2–4 at two ranks as well, so three rounds is enough here too.

**The autostart unit, installed and tested but left disabled.**
[`tracks/tp2/harem-exl3-tp2.service`](tracks/tp2/harem-exl3-tp2.service) with
[`motor-onkosul-exl3-tp2.sh`](tracks/tp2/motor-onkosul-exl3-tp2.sh) (one fabric peer per node instead of
two; the ConnectX-7 check stays 4/4 because it counts ports, not peers; `Conflicts=` **both** sibling
units). `systemctl start` on both nodes → `/health` 200 at **+261 s** → gates 10/10 · 12/12 →
`systemctl stop` clean. Its first attempt **failed correctly**: the preflight refused in one second
because the sidecar the environment file named was not there, before docker was touched. **No
two-node reboot test** `[not tested]`, and the three-node unit remains the enabled autostart.

**What is deliberately still open at two ranks:** the `gpu-memory-utilization` ladder (0.85 is where
every arm ran; it is where our first two-node arm found 3.5 GB of swap during weight load, and at
three ranks 0.85 was *rejected* — the ladder must be re-derived, not copied), a two-node reboot test,
expert parallelism on, and a second boot of each candidate. [docs/15](docs/15-tp2-track.md) §6.

---

## 2026-09-06 — the draft KV page at two ranks: the pool more than doubles, and the long-prompt path only exists with it

**A two-node owner read [docs/15](docs/15-tp2-track.md) and said the KV cache bug was the thing
standing between them and a usable configuration. They were right, and this page's own projection
was low.** `HAREM_SW_BLOCK_SIZE=256` — the draft KV page, 16 to 256 tokens — was the largest
untested item on the TP=2 page, carrying an `[estimate]` of "perhaps 0.8-1.2 M from arm C's
665,625". We ran the control and the fix back to back at two ranks on 6 September 2026, one boot
each, the only difference one token in `EXTRA_ENV`. New section
[docs/15](docs/15-tp2-track.md) §3.5, raw record
[`results/speed/tp2-draft-page.md`](results/speed/tp2-draft-page.md) `[measured-here]`.

**KV pool 601,562 to 1,303,571, +116.7 %** (+109.6 % after normalising for the 3.4 % more KV memory
the fix arm's binding rank happened to get). C8 aggregate 127.54 to 135.59 (+6.3 %), C2/C4/C6
+0.7/+2.3/+4.0 %, TTFT −23 % at C1 and −27 % at C8, C1 aggregate equal. Quality gates 10/10 · 12/12
· 0 empty, cold and warm, on both arms. Settings otherwise identical: two nodes, TP=2, EP off, image
`exl3-zeus:62f53e6`, routed-experts-only checkpoint at `b20c49ba`, KV fp8, DFlash2 k=7,
`gpu-memory-utilization 0.85`, `--block-size 256`, `--max-num-batched-tokens 2048`,
`--max-model-len 1000000`, medians of three rounds, both booted through the host-side settle gate.

**The headline is not the pool, it is the cliff.** Without the fix the control does not serve a long
prompt slowly — it never schedules it: a 6,253-token prompt sits at `Running: 0, Waiting: 1, GPU KV
cache usage: 0.0 %` indefinitely, and both prefill benchmarks time out. The arithmetic is one line:
block ids are global to a single pool, and one request wants 640 of the pool's 385 blocks. With the
fix it wants 280 of 365, and 8,268 tokens serve in 6.3 s, with fresh-prompt prefill at 1,478 tok/s.

**Why the projection was low: the defect is worse at two ranks than at three.** The engine's own
per-group decomposition shows the drafter taking **60.2 %** of the blocks-per-request divisor at
TP=2 against 53 % at TP=3 — not because the drafter changed, its 385 blocks are identical, but
because the platform raises the attention block to 4,608 tokens at two ranks where it is 3,328 at
three, cutting the target's share from 301 blocks to 218. A smaller denominator makes the same
defect a larger share. [docs/07](docs/07-kv-and-draft-page.md) §4's "the page layout was the
*second-order* term" is corrected in place: second-order in memory, first-order in the counter, and
it grows as the node count falls.

**What it cost.** Per-block memory rises about 9.1 %, the same price as at three ranks. The draft
group's prefix-cache matching unit coarsens from 16 to 256 tokens, which costs nothing measurable
here because the hit rate is already 0 % behind a 4,608-token block. Acceptance falls 1.9 points,
inside this stack's boot-to-boot band. And C1 per-stream decode reads 7.9 % lower, which one boot
does not establish: the fix arm's own three rounds span 9.3 %, the C1 aggregate is equal and TTFT is
23 % better. No C1 gain, and no proven C1 loss.

**One retraction, and it is about our instrument rather than the stack.** The control arm
re-measured today gives **601,562** where 5 September published **665,625** for the same untouched
env file, −9.6 %. The 5 September TP=2 harness had no settle gate; this one waits for `MemAvailable`
to come back over 112 GiB on both nodes. That is exactly the bias
[docs/07](docs/07-kv-and-draft-page.md) §1.1 describes, in the direction it predicts. It matters
more than a percentage: 5 September's 426 blocks were just above the long-prompt cliff and served a
7,382-token prompt at 1,135 tok/s; today's cleaner 385 are just below it and cannot.
**Without the page fix, whether two ranks can serve an 8K prompt depends on the host's state at
boot.** The arm C prefill and pool figures in §3.2 and §4 are flagged as measured on an unpinned
baseline, and the "arm C had 4.4 GiB of available KV" figure that §3.4's retracted arithmetic rested
on is contradicted by today's 9.97 GiB; it is now marked an inference rather than a log line.

The patch is [`patches/tp3/patch-swblock-tp3.py`](patches/tp3/patch-swblock-tp3.py) unchanged — it
is `tp`-agnostic and gated on its own environment variable, so the control ran the same image with
the knob unset. `HAREM_SW_BLOCK_SIZE=256` moves from "keep, never run at TP=2" to **mandatory in
practice** in the §2.5 flag table. Still `[not tested]` at two ranks: a second boot, the fp8 draft
cache, fast load, the memory ladder, expert parallelism on, and the page fix on top of the
full-scope checkpoint.

---

## 2026-09-05 (after the release pass) — two documents: the TP=2 track, and what other published recipes report

**[docs/15 — Running this recipe at TP=2](docs/15-tp2-track.md).** This is a three-node recipe and
every default in it is a TP=3 default, but it runs on two nodes and at two ranks it is a *shorter*
recipe rather than a cut-down one: all five shapes that do not divide by three divide by two **and**
leave every rank a whole number of 128-column Hadamard blocks, so the whole of
[docs/03](docs/03-tp3-padding-and-sidecars.md) and the padded-load path in
[docs/13](docs/13-full-scope-checkpoint.md) §7 fall away. Expert parallelism becomes **optional** —
2,048/2 = 8 × 128 — and `preflight-tp3.py` already accepts `--ep 0` at two ranks while refusing it at
three. The page lists the exact nine changes (three env lines, three launcher edits, the
`tracks/tp2/patches/` tree, the image, the autostart unit), the four two-node arms we ran with their dates
and settings, and the eleven-row table of production features we have **never** run at two ranks:
`HAREM_SW_BLOCK_SIZE=256`, the fp8 draft cache, the fast-load sidecar, the memory ladder, expert
parallelism on, a two-node reboot test and a second boot among them `[not tested]`.

**The trade-off is not the one people expect, and the page says so in its second paragraph.** Like for
like on the same day, same image and same harness, two nodes reach **85–91 %** of three nodes' C1 per
stream, **75 %** of C8, **75 %** of prefill and **14 %** of the KV pool `[measured-here]`. TP=2 does
**not** win single-stream latency here: a decode step is weight-bandwidth bound, a third rank cuts each
rank's weight traffic by a third, and that beats the collective it costs. What two ranks buy is a node
and a shorter recipe. Boxed "At TP=2" notes now sit in
[03](docs/03-tp3-padding-and-sidecars.md), [05](docs/05-expert-parallel-and-cuda-exl3-fixes.md),
[06](docs/06-nccl-mesh.md), [07](docs/07-kv-and-draft-page.md), [08](docs/08-fast-boot.md),
[13](docs/13-full-scope-checkpoint.md) and [systemd](systemd/README.md), each pointing at the step that
differs. This partly closes [CONTRIBUTING](CONTRIBUTING.md) item 10; §3.3 says exactly which part is
still open.

**[docs/16 — Comparison with other published recipes](docs/16-comparison-with-published-recipes.md).**
Twelve other public GLM-5.3-Flash EXL3 DGX Spark recipes — nine at two nodes, two at three, one at
four — quoted **exactly as they publish them** with the conditions they state and tiered `[reported]`,
beside our figures at the matching node count. Nothing in their columns is re-derived, rescaled or
corrected by us, and where conditions differ the difference is a column rather than a footnote.
**There is no three-node recipe from `MiaAI-Lab`** — we looked on GitHub and on Hugging Face; its only
larger arrangement is an explicitly untested four-node script, and the two three-node EXL3 recipes are
`FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks` (commit `9093765c`, 2026-08-29, one commit and unchanged
since) and `jakejharris/jspark3` (`v1.0.0`, 2026-09-03).

**The most valuable thing on the page is not a ranking — it is that two other people quantized this
model's dense path independently, and both measured what we measured.** `Alexbob0/glm53-flash-dense-exl3-tp2`
overlays the same 4.05 bpw pack's dense tensors onto a routed-experts-only checkpoint at **two** nodes
and reports **+22 % structured / +26 % prose** against its own control, with a torch-profiler step
anatomy whose top item and residual-BF16 story match ours `[reported]`; ours is +24.3 % per stream at
TP=2 through a different loader and a different kernel project. `jakejharris/jspark3` does it at
**three** nodes with an INT8 W8A16 Marlin overlay on 169 dense modules and reports +6.6–8.4 % against
its own control, along with a disclosed −21.2 % at C3 and −3.4 % on long prefill. Three stacks, three
routes, one direction. Their prose figure, 29.049 tok/s, also lands on our 29.1 — the only nearly
like-for-like row on the page.

**Three more things it establishes, and one it refuses to do.** The FlyCockpit padding constants
(`lcm(64,3) = 192`, shared expert 2,112) and ours (384, 2,304) are **both correct**, for different
checkpoints — the unit is 64 when those tensors are BF16 and 128 when they are EXL3, and our own
configurations 1–8 used theirs. The two-node KV pool gap is **ours**: our 665,625 is the smallest
two-node figure on the page because we never ran `HAREM_SW_BLOCK_SIZE=256` at two ranks, while the
`MiaAI-Lab` lineage solved the same drafter-page problem with a padded slot-share onto the MLA
tensors. And `punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark` publishes a **soak hang** — three of
three, a 96K chunked prefill sharing steps with speculative streams, and a 282K cold prefill that
still stalled with the drafter off — which we have never looked for on three nodes `[not tested]` and
which is now the outside item we would most like settled about our own stack. What the page refuses to
do is convert a synthetic number into a realistic one: most of these recipes lead with a counting
prompt, the gap is roughly **1.7×** on this model family, and those rows stay labelled and apart.

**What we took**: from `FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks`, the arithmetic already credited
in [CREDITS.md](CREDITS.md) and no files; from `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`,
**nothing** — no code, no configuration, no number, no technique; from everything else, nothing, and
all of it was found after production 9 shipped. They are named as the sources of their own figures and
for no other reason. Two search hits are excluded and deliberately not named: their documents are
copies of other repositories' files and their READMEs advertise a Windows executable for a machine
that cannot hold this model, so there is nothing of theirs to compare.

**Cost:** documentation only. No cluster time, no configuration change, no number in this repository
moved.

---

## 2026-09-05 (after the release pass) — the quantization gate bench, re-measured on the target GPU: the closure stands, the reason we gave for it does not

**`[retracted]`: "on the KDA shapes EXL3 is 1.58–1.76× slower than BF16 at M=8".** That sentence
closed the item above, and it went upstream. It is a **warm** number. The workstation bench did run
against a ~300 MB weight bank — the ruler had caught a 210 %-of-peak reading and that is why the bank
exists — but the bank was sized for the large shapes, and the arm the sentence is about is **0.72 MB**
and sat in that card's 101 MB L2 from the first replay to the last.

**Re-measured cold on GB10, same shapes, same `cuda-exl3` `754421f` build, both arms rotated over a
bank of at least 4× L2** `[measured-here]`. `f_b_proj` at M=8 reads **1.023**, not 1.596 — and GB10's
own *warm* arm reproduces the withdrawn figure at **1.605**, which is how we know what the workstation
was measuring. **Seven of nine shapes reverse sign**; `kv_b_proj` improves from 0.391 to **0.291**.
**The production trace refereed it**: per-call costs from the production-9 C1 profile are `f_b_proj`
5.41 µs against 4.99 cold and 2.21 warm, `in_proj_bfg_a` 14.20 against 15.51 and 6.52 — **cold
reproduces the engine to ±20 %, warm is out by 2.2–2.4×**, so which regime is honest is a measurement
and not a preference.

**What it changes, and what it does not.** The KDA gating arms move from "quantizing them costs
0.584 ms/step" to **+0.050 ms/step — neutral — 0.07 % of C1**. They stay BF16, now for want of a gain
rather than for a loss, and the lever on them is not the bit width at all: cold, `f_b_proj` reads
**45.7 GB/s, 19 % of peak, in either format**, because it sits on a ~5 µs floor made of **two
dependent kernel launches** where bf16 has one. Fusing `had_in` into the GEMM at narrow inputs is what
those shapes want, and it has been asked for upstream. The three measured families together go from
+0.547 to **+1.368 ms/step** against a 1.5 ms gate — a narrow miss rather than an order of magnitude —
and **96 % of that is still `kv_b_proj` alone**, which still needs the per-head batched EXL3 kernel
`754421f` does not have. **Prefill (M = 1,792) was not re-measured and the gate is an *and*, so the
item is re-scoped, not passed.** The MLA fp32 → bf16 lever is untouched and is still the cheapest
thing on the board.

**A second claim of the same report goes with the first** `[retracted]`: it warned its ratios were
*optimistic* for GB10. **Every family came out better on GB10** — machine balance about 416 flop/byte
against the workstation's 141 is exactly the condition a trellis is built for, and on the control
shapes EXL3 delivers 0.272 against the pure bit-ratio limit of 0.250.

**The lesson, and it is the third time this stack has learned a version of it.** The artefact's
**sign depends on which arm fits the cache**: on a 101–128 MiB L2 both fit and EXL3 reads slow, on
GB10's 24 MiB only the trellis fits and EXL3 reads too fast. It therefore does **not** cancel in a
ratio, and only cold is honest on either card. **A bank sized against the wrong card, or the wrong
shape, is the same mistake as no bank at all.** The rule is now in
[docs/09](docs/09-measurement-protocol.md) §4.2: size the bank per shape and against the card you will
serve on, rotate both arms over it, and let a production trace referee the regime where one exists.

**Cost:** 90 s of one node's GPU, 1.47 GiB peak, in a throwaway container beside an idle engine. No
restart, no configuration change, nothing on the cluster touched. Posted to the upstream thread the
original went to. [docs/11](docs/11-open-issues.md) §1.11 and §2.25,
[docs/13](docs/13-full-scope-checkpoint.md) §4.4,
[`results/kernels/kda-gate-bench-gb10.md`](results/kernels/kda-gate-bench-gb10.md) with its raw under
[`results/kernels/gb10-coldbench/`](results/kernels/gb10-coldbench/), the script as
`bench/kda_gate_bench_gb10.py`, and the retraction indexed in [`audit/`](audit/README.md) §6. The
workstation file is kept exactly as published, with a banner.

---

## 2026-09-05 (late) — release: autostart, three open items closed by measurement, and a spread we had mis-attributed

**Status moves from release candidate to release**, on production configuration 10.

**Autostart, and it has been through a reboot** `[measured-here]`. The `systemd/` directory stops
being a template with three things wrong with it: `tracks/tp3/harem-exl3.service` and
`tracks/tp3/motor-onkosul-exl3.sh` are the real unit and the real preflight, installed and `enabled` on
all three nodes. The preflight runs seven checks — docker answering, `ibv_devinfo` 4/4, a ping to each
fabric neighbour, `drop_caches`, then the env file, the image and this rank's fast-load sidecar
manifest, the last three each having cost us a silent boot. `ExecStop` names `exl3-tp3` rather than
the NVFP4 container the template named, `TimeoutStartSec` is **1200** so a 620 s dump boot cannot time
out mid-load, and the unit carries `Conflicts=harem-motor.service` while the sibling unit is now
`disabled` on all three nodes.

**The reboot test**, whole cluster, one trial: ssh and 4/4 at **+29 / +30 / +31 s**; units log
`Finished` at **+98 / +98 / +103 s** — that is the preflight, the fabric wait and the settle gate
together; `/health` 200 at 22:28:21, **212 s after the last unit finished**. KV pool **5,652,892**
against 5,619,834 from a settled `docker run` on the same configuration, **+0.6 %** — the cleanest
baseline a pool number can have, and the strongest check on the settle gate we have run. Gates 10/10
and 12/12 afterwards. **The elapsed figure is published twice because the log contradicts itself:**
the harness printed `+242 s`, the wall-clock stamps in the same file give **315 s**, 242 s before the
health check matches no recorded event, and the larger is the figure to plan with. What the unit does
**not** solve is stated with it: `--restart no` is unchanged and deliberate, systemd still does not
enforce the worker-2 → worker-1 → head order (the rendezvous retrying is margin, not a guarantee), and
the watchdog is still not written. **Reboot all three together or none** — the preflight checks only
its own node's fabric, so a single-node reboot passes it and starts a rank into a cluster whose peers
are gone. [`systemd/`](systemd/README.md), [docs/00](docs/00-hardware-and-os.md) §13a,
[docs/09](docs/09-measurement-protocol.md) §11.4, [docs/11](docs/11-open-issues.md) §2.20,
[docs/14](docs/14-troubleshooting.md) §10.1–§10.1a, README quick-start step 11.

**`NCCL_ALGO=Ring,Tree` — closed, equal, `Ring` stays** `[measured-here]`. The five-round engine arm
the model-free sweep could not substitute for: C1 70.6 / C2 99.9 / C4 143.4 / C6 175.2 / C8 195.6
against `Ring`'s 70.5 / 99.0 / 144.6 / 175.4 / 194.0 — **every level inside ±1 %**, TTFT identical,
acceptance inside its own spread, gates full. The proxy's −3.6 % did not survive a real step, which is
what should happen when a proxy times 90 collectives in isolation and the step overlaps none of them.
The one structural difference is **+1.5 % of KV pool** (5,702,479 against 5,619,834) from NCCL's
per-algorithm buffer sizing — real, and not worth pinning production to an algorithm list for.
[docs/06](docs/06-nccl-mesh.md) §12.3, [`results/mesh/algo-sweep.md`](results/mesh/algo-sweep.md).

**Quantizing the remaining BF16 KDA/MLA modules ourselves — closed, and it would be slower**
`[measured-here]`. The 113 linears per rank that production 9 leaves in BF16 were the obvious next
arm: the FP16 weights are already in the checkpoint, so it is a surgical pass rather than a
requantization. A model-free bench on a workstation GPU, `cuda-exl3` at the production commit, at the
TP=3 per-rank shapes, closed it before any of that was spent. **The checkpoint author left them BF16
by design** — `qmap = None` in `exllamav3` with the reason in the comment, and `kv_b_proj` is not a
`Linear` at all — and he applied that judgement discriminatingly, quantizing KDA `qkv_proj` and
`o_proj` to 6 bit from the same exclusion list. **On the KDA shapes EXL3 is 1.6–1.8× slower than BF16
at M=8**: those arms are 0.72 MB and were never bandwidth-bound, so four-bit weights buy bytes that
were not the cost while paying the Hadamard and trellis-decode fixed cost. The arithmetic cannot be
rescued even at zero cost — the whole family is **0.851 ms of a 72.5 ms step**, +1.2 % of C1 if it
were free. The one family that would genuinely gain, `kv_b_proj` at 0.391× and +1.13 ms/step, **needs
a batched EXL3 kernel that does not exist** (verified by source scan, along with the absence of any
M-threshold reconstruct path). **One lever survived and it is not a quantization lever:** the MLA
strided-batched family runs in fp32, and bf16 measures 0.684× — **+0.24 ms/step, about +0.3 % of C1**,
no checkpoint change and no new kernel, filed as future and minor and gated on `needle` at 1M rather
than on speed. Three instrument lessons came with it, including a ruler check that read **210 % of the
machine's own peak** before a weight bank was added. [docs/13](docs/13-full-scope-checkpoint.md) §4.4,
[docs/11](docs/11-open-issues.md) §2.25,
[`results/kernels/kda-gate-bench.md`](results/kernels/kda-gate-bench.md), and both scripts ship
(`bench/ruler_check.py`, `bench/kda_gate_bench.py`).
**Corrected the same night, and the entry above is the correction:** the 1.6–1.8× is a warm number
that the weight bank did not reach on a 0.72 MB arm, the item still closes, and it closes because
quantizing those arms is **neutral** rather than because it is a loss `[retracted]`.

**A spread we had mis-attributed, corrected against the raw sweeps** `[retracted]`. Our field log
recorded a "C1 run-to-run spread of 59.9 … 70.6 across boots" for production 9 and 10. It is wrong:
**59.9 is the TP=2 full-scope arm's C1 aggregate** and the other low reading beside it was a
production **8** restore. No production 9 or 10 boot has read below **67.7** even at the level of a
single round. Every round of all four boots is now printed in
[docs/10](docs/10-results-and-roofline.md) §1.1, and the real picture is that the two axes behave
differently: within a boot, C1 spans 1.0–5.7 % peak to peak and C4 5.1–9.8 %; across boot **medians**,
C1 spans **1.1 %**, C8 **2.5 %** and C4 **7.4 %**. **Which makes one published number a trap:**
production 10's C4 of 144.6 against production 9's 134.6 is **+7.4 %**, exactly C4's boot-median span
— a memory fraction does not buy 7 % of four-way throughput, and the arms are equal at C4. Every
headline figure is the median of one boot's rounds; the rule and the guard are now stated where the
numbers are.

**Stale claims the previous pass's retractions left behind, corrected.** `docs/10` §1 and `docs/13`
still billed production 9 for **−2.4 points of draft acceptance and −3.0 % of accepted tokens per
step** after the front page had withdrawn both; the four costs that remain there are now the four that
were never about the drafter. `results/gates/quality-gates.md` still said MMLU had not been run at
TP=3; it has, at 86.47 ±0.74 on the production checkpoint, and the page now separates that from the
86.4 ±0.7 TP=2 figure that configurations 7 and 8 carry forward. `docs/11` §3 still listed a
production-9 re-profile as never run.

**Also in this pass.** `results/speed/concurrency-sweeps.csv` gains the production 9, production 10
and `Ring,Tree` arms, so the provenance table's rows point at data that is actually in the file;
`results/configs/*.csv` gain the measured 0.83 rung in place of the "designed, not run" row, plus the
`Ring,Tree` and power-on boots; `audit/README.md` §5 gains five provenance rows and two more facts a
reviewer should not have to dig for, and its memory section now states the rule production 10 sits
against — **watch swap, not `MemFree`**, because `MemFree` is 0.9–1.2 GiB at 0.83 and swap is what
0.85 was actually rejected on. 0.85 will not be attempted; the rung after 0.83 is a soak.

---

## 2026-09-05 (night) — release pass: production configuration 10, two retractions, and the environment record

**Production configuration 10 = production 9 with one line changed**, `GPU_MEMORY_UTILIZATION`
0.80 → 0.83 `[measured-here]`. KV pool **5,168,044 → 5,619,834 (+8.7 %)**; C1/C4/C8 69.8/134.6/192.4
→ 70.5/144.6/194.0, all inside their bands; prefill-fresh 1,687/1,769/1,779 against 1,745–1,774;
quality gates full, cold and warm. **Swap stayed flat at ~0.1 GB per node through the rounds**, which
is the number 0.85 failed on. `MemAvailable` 8–10 GB, `MemFree` 0.9–1.2 GiB of reclaimable page
cache. `[docs/11](docs/11-open-issues.md)` §2.4 predicted +11 % from production 7's geometry and the
measured figure is +8.7 % — the method held, the base did not. **0.85 will not be attempted.**

**Two retractions, both of ours, both from re-reading raw data we already had:**

- **The "−2.4 points of draft acceptance" cost of production 9 does not exist** `[retracted]`. It is
  an artefact of `scripts/bench-sweep.py` cycling `prompts[i % 12]`, so C1 and C2 see only 8 of the
  12 prompts while C4–C8 see all twelve, and the two groups differ by about 8 points. Pooled by draft
  token over five levels and three boots: **+0.18 points**, and the sign reverses at C6. The cold
  probe is identical on both arms. `accept_len = 1 + k × acceptance` on all 90 rows, so the
  "acceptance" and "tokens per step" costs were one number written twice. Net throughput effect
  **+0.24 %**. Front page, [docs/11](docs/11-open-issues.md) §2.26 and
  [audit/](audit/README.md) §6 corrected.
- **Production 9 *was* re-profiled**, and this repository said it had not been `[retracted]`. The
  torch-profiler run against the live production-9 server (three ranks, no restart) landed 40 minutes
  after the README and `docs/10` were written. Dense stage **45.3 % → 25.9 %** of a C1 step
  (42.90 → 21.90 ms); new C1 ranking MoE trellis GEMM 32.5 %, NCCL 26.1 %‡, dense EXL3 GEMM 15.0 %,
  remaining BF16 10.3 %; prefill chunk MoE 28.1 %, NCCL 14.0 %, dense EXL3 13.4 %; C8 MoE 56.3 %.
  ‡ NCCL and CPU-gap at C1 are **CUPTI-inflated** — 2,738 kernel launches per step, so with the
  profiler off the two together are ≤17.19 ms rather than 29.1, and the step wall is 72.52 ms rather
  than 84.44. **And the arithmetic missed something:** the full-scope dense stage is **+17.3 ms,
  +10.4 % in *prefill*** at M=1,792. The wall stayed flat only because three other classes gave it
  back. The plugin author reproduced the shape on his own card and withdrew a "cuBLAS parity" claim
  from his README.

**New in the repository, for the release:**

- **[docs/00](docs/00-hardware-and-os.md) rewritten as a complete environment record** — firmware
  (SBIOS/EC/SoC/USB-C PD, and why below `0104` costs a quarter of the fabric silently), the
  `dgx-spark-mlnx-hotplug` root cause and the file to remove, the physical cabling (**three** cables
  carrying six links, not six cables — corrected), the PCIe Gen5 x4 ceiling read on 12 of 12
  endpoints, every version with the command that prints it, **the six OS-level changes we made and
  the three we deliberately did not**, the `swappiness=0` lock-up, the memory rules and the settle
  gate, and the mesh plugin's build and binary identities.
- **[docs/14](docs/14-troubleshooting.md)** — all 83 failures by symptom with the exact log line, a
  triage order, and a ranked index of the twenty that produced **no error message at all**.
- **[audit/](audit/README.md)** — a post-install self-check with our numbers beside each step, a
  provenance table for every headline figure, and the retraction index.
- **[charts/](charts/)** — four figures, generated from committed CSVs by a standard-library script.
- `results/configs/production-configurations.csv`, `results/configs/kv-pool-progression.csv`,
  `results/profile/step-breakdown.csv`.

**Corrections made in place:** `docs/10` §4.2 carried "2048-token chunk, 1.109 s" — a chunk size taken
from the flag rather than the trace and a wall from an earlier configuration. The measured values are
**1,792 tokens** (7 × 256) at **962.55 ms** (production 7) and **961.73 ms** (production 9), which
moves prefill's effective bandwidth from 53 to about **61 GB/s**. `LICENSES.md` and `CREDITS.md` named
the fallback checkpoint as the production one and pinned `cuda-exl3` at `f4987cf`; both now say
`turboderp/GLM-5.3-Flash-exl3` (MIT) and `754421f`. Category speeds are production 9's
(code 61.7 · math 79.6 · JSON 72.8 · prose 29.1) rather than five configurations old.

---

## 2026-09-05 (late evening) — production configuration 9: the checkpoint changed, and it is the largest move this stack has made

Every configuration from 1 to 8 served routed-experts-only weights beside a BF16 attention stack.
Production 9 does not. The item this repository had called its largest for a week — dense BF16 GEMM
at 45.3 % of a single-stream decode step — is closed, taken, and in production four hours after the
TP=2 arm that proved it was worth taking.

- **Production configuration 9** `[measured-here]`. `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw
  (`2a30229e`, MIT), full scope, at TP=3 with expert parallelism, on image `exl3-zeus:754421f` and the
  new `tracks/tp3/patches/` tree. Control is production 8 — the **pool of two runs of the same script on
  the same day**, because that arm's documented run-to-run spread is about 7 %. Medians of three
  rounds, everything else identical:

  | | **production 9** | production 8 | delta |
  |---|---|---|---|
  | C1 total / per stream | **69.90 / 75.91** tok/s | 56.88 / 62.39 | **+22.9 % / +21.7 %** |
  | C2 / C4 / C6 total | 99.17 / 140.72 / 172.40 | 83.31 / 120.22 / 144.03 | +19.0 / +17.1 / +19.7 % |
  | C8 total | **197.20** | 175.37 | **+12.5 %** |
  | TTFT, C1 / C8 | 0.280 / 0.826 s | 0.344 / 0.906 | −18.6 / −8.8 % |
  | prefill, fresh / 7K repeat | 1,738 / 1,575 | 1,776 / 1,537 | equal both ways |
  | KV pool at 0.80, 1M context | **5,165,289** | 4,696,969 | **+10.0 %** |
  | consumed memory per node | 58.3–59.1 GiB | 62.1–62.4 GiB | **−3.4 GiB** |
  | draft acceptance · tokens per step | 61.94 % · 5.34 | 64.36 % · 5.50 | **−2.4 pt · −3.0 %** |
  | boot, fast-load | 251 s (weights 57.9 s) | 264 s (73.2 s) | −5 % |
  | gates cold and warm · MMLU sample | 10/10 · 12/12 · **86.47 ±0.74** | 10/10 · 12/12 · 86.4 ±0.7 | equal |

  **The step arithmetic is the entry:** 88.2 → **70.3 ms**, 17.8 ms saved, while acceptance and
  accepted length moved the *wrong* way by about 3 %. The whole gain is the dense stage going from
  BF16 to 4–6 bit; none of it is drafter behaviour. The same lever measured 20.7 ms at TP=2 on two
  nodes, so it is now **confirmed on two independent topologies**.

- **What it cost, and the line is not empty** `[measured-here]`. Draft acceptance −2.4 points
  (gate ≥60 %, passed) and accepted tokens per step −3.0 %; a **second patch tree** whose
  `patch-vllm-tp3.py` and `preflight-tp3.py` diverge from production's on purpose and have to be kept
  in step by hand; a **second 53 GB fast-load sidecar** per node on top of a second 154 GiB
  checkpoint. Quality was looked for and not found: 86.47 ±0.74 against 86.4 ±0.7 is 0.07 points, a
  tenth of either error bar, with both gates full cold and warm on the same engine instance in one
  session. [docs/13](docs/13-full-scope-checkpoint.md) §7.4, [docs/11](docs/11-open-issues.md) §2.24.

- **The TP=3 port: two constants, one patch of ours, and a padded-load path from upstream.** The two
  launcher constants moved from `lcm(64, tp)` to `lcm(128, tp)` — vocab `padding_size` 192 → **384**
  (155,136 = 3 × 404 × 128) and shared expert 2112 → **2304** (768 = 6 × 128 per rank) — because a
  full-scope checkpoint quantizes both and every EXL3 pad has to be whole 128-column Hadamard blocks.
  **2,176 is 128-aligned and not divisible by three; the width must be a multiple of `lcm(128, 3)`.**
  Ours is **A9**: vLLM's tuple-shard loader slices the checkpoint's single 24,576-wide `qkv_proj`
  using the module's *padded* `output_sizes` (3 × 8,448), so segment 1 reads from the wrong offset and
  segment 2 overruns; it now splits by the checkpoint's real widths and lets the existing zero-pad
  widen each rank's slice. No-op at TP≤2. [docs/03](docs/03-tp3-padding-and-sidecars.md) §1.1.

- **A10, the pad audit — and the invariant it turned out to be checking had been holding by
  accident** `[measured-here]`. The `svh = 0` mechanism was already running on this stack for
  column-parallel EXL3 modules before anyone designed it, and **nothing verified it** — which is
  precisely how the old 2,112 arithmetic would have produced silently wrong output rather than an
  error. A10 now walks every EXL3 module after load and reports:

  ```text
  HAREM-FULLSCOPE assert 5: 285 EXL3 pad site(s) audited, 285 padded on this rank, all whole 128-blocks and exactly zero
  ```

  285 is what a model-free meta-device run predicted for rank 2 **before** the boot. The audit also
  caught its own bug on first writing — it called a column-parallel module's *input* padded and
  rejected `in_proj_qkv` on ranks 1 and 2 while rank 0 passed. **A single-rank test could not have
  seen it.**

- **Two TP=2 warnings did not reproduce, and one is a retraction** `[retracted]`. The dress rehearsal
  said the full-scope checkpoint was ~10 GiB *heavier* per node and projected the TP=3 pool falling
  4.70 M → ~3.4 M, −27 %, with the 1M-context claim at risk. Measured at three ranks: **3.4 GiB
  lighter, pool +10.0 %**, `MAX_MODEL_LEN` never touched. The TP=2 pair was confounded — different
  checkpoints *and* different `max_model_len` — and we record it as **not reproduced rather than
  explained**, because the mechanism was never isolated. Audit row 32. The second, a cold-probe
  signal that draft acceptance collapses, is flat on that same probe at TP=3 (row 30).

- **Ten boot gates, all held on the first attempt** `[measured-here]`: the image capability
  precheck (`[padload] ... =yes` on all three, or exit 23 before a byte of weight is read), the ten
  patch anchors with the patch `sha256`, the preflight arithmetic, asserts 1–4 silent, assert 5 at
  285/285, the padded `lm_head` line (154,880 → 155,136, two whole Hadamard blocks zeroed through
  `svh`), the `CUDA_EXL3_DEBUG_NAMES` tally at **203 EXL3 / 113 bf16** read negatively, the KV pool,
  free memory and swap, and both quality gates cold and warm. The acceptance list is written out in
  [docs/09](docs/09-measurement-protocol.md) §5.1, in the order it was run.

- **What stays BF16, measured rather than inferred.** The 113 are four families and nothing else:
  KDA `f_b_proj`, `g_b_proj`, `in_proj_bfg_a` and MLA `kv_b_proj`. They are why 17.8 ms arrived
  rather than the ~32 ms the estimate implied, and closing them is a **checkpoint-side** decision, not
  ours ([docs/11](docs/11-open-issues.md) §2.25).

- **Upstream.** Two more commits from the kernel author, and both were required — `f3e3090` (a padded
  output dim accepted when the pad is whole 128-blocks, `svh` allocated zeroed; and a row-parallel
  `suh` load that copies what exists and zeros the rest instead of narrowing off the end) and
  `754421f` (the vocab loaders fill a prefix). On `f3e3090` alone the boot clears `create_weights`
  and dies on a `copy_` shape mismatch in `_vocab_loaders`; on anything older it raises "EXL3 weights
  cannot be zero-extended". [CREDITS](CREDITS.md).

- **In the repository.** `tracks/tp3/patches/` (the whole production tree, with its own README),
  `tracks/tp2/patches/patch-fullscope-tp2.py` (the eight-anchor TP=2 patch, which had been referenced as
  "not yet in repo"), `tracks/tp3/env.tp3-full.example`, and `scripts/ab-quick2-full.sh` +
  `scripts/boot-only-full.sh` — the tier-B harness for the full-scope tree, which differs from the
  production one in a single line and was added rather than edited, for the same reason the patch
  tree was ([docs/09](docs/09-measurement-protocol.md) §11.2).

- **Rollback, one line or one file.** Delete `HAREM_EXL3_FULLSCOPE=1` from `EXTRA_ENV` and the same
  patched image takes the upstream path; or start with `ENV_FILE` pointing at the production 8 env
  file, which reverts the checkpoint, the image, the patch tree and the sidecar together because all
  four are named in it. `patches/tp3/` and production 8's sidecar were never modified.

---

## 2026-09-05 (evening) — the full-scope checkpoint loads, and the largest item on the stack is now a measurement

The item this repository has called its largest is no longer an estimate, and the reason it had been
out of reach turned out not to be quantization, parallelism or the weights.

- **A full-scope EXL3 checkpoint served for the first time on this stack, at TP=2**
  `[measured-here]`. `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (`2a30229e`, **MIT**, 165 GB,
  `sha256` 23/23 verified on both nodes) against our production checkpoint as control, same image,
  same two nodes, expert parallel off, DFlash2 k=7, medians of three rounds:

  | | full scope | experts-only control | delta |
  |---|---|---|---|
  | C1 per stream / aggregate | **68.00 / 59.93** tok/s | 54.69 / 47.40 | **+24.3 % / +26.4 %** |
  | C2 / C4 aggregate | 83.02 / 111.05 | 68.03 / 90.66 | +22.0 % / +22.5 % |
  | TTFT at C1 | 0.524 s | 0.615 s | −14.8 % |
  | draft acceptance, accepted length | 63.14 %, 5.42 | 64.08 %, 5.49 | unchanged |
  | gates, cold and warm | 10/10 · 12/12 | 10/10 · 12/12 | equal |
  | MMLU sample (1,995 q) | **86.32 ±0.75** | 86.4 ±0.7 | inside the bar |

  Per decode step 100.4 → **79.7 ms**, 20.7 ms saved, with acceptance and accepted length flat — the
  entire gain is arithmetic, none of it is drafter behaviour. Against the estimate this stack carried
  (42.9 ms → ~11 ms, ~+34 %), **65 % arrived**; the rest is what this checkpoint leaves in BF16
  anyway. **The 6-bit `lm_head` at vocab 154,880, the one quality risk we had flagged, cost nothing
  measurable.** New page: [docs/13](docs/13-full-scope-checkpoint.md);
  [docs/10](docs/10-results-and-roofline.md) §2.2, [docs/11](docs/11-open-issues.md) §2.22.

- **Retracted: "the dense stage is a quality choice"** `[retracted]`. It was a **loader limitation**.
  vLLM's `glm5next` model file pins the attention stack to BF16 in two places — `quant_config=None`
  for the MLA projections and a `quant_config` strip in the KDA constructor — and between them they
  lock **72.8 % of the dense traffic** whatever the checkpoint holds. `routed_experts_only` was not
  the publisher's judgement about quality; it was the only scope that could load. Retraction 29.

- **Retracted: "draft acceptance drops on a quantized target"** `[retracted]`. A cold single-prompt
  probe read 45.5 → 39.2 % and 48.0 → 34.4 % and was reported upstream as an early signal that would
  "eat part of the speed gain". Over three sweep rounds acceptance is 61–65 % against the control's
  62–63 %. Sample of one, published. Retraction 30.

- **Corrected: the delta we posted upstream** `[retracted]`. The issue-thread table gave the control's
  C1 as "54.7 / 54.3" aggregate / per stream and derived "+9.5 % / +25 %"; 54.7 is the control's
  *per-stream* median and its aggregate median is 47.40. Like for like: **+26.4 % aggregate, +24.3 %
  per stream.** The conclusion is unaffected. Retraction 31 — the only row in that table produced by a
  summary rather than a measurement.

- **Three loader layers, one env-gated patch, and a vLLM bug** `[measured-here]`. The checkpoint would
  not load for three independent reasons, measured as an unmapped-tensor count going
  **886 → 526 → 170 → 0**: no `packed_modules_mapping` on the model class; the BF16 pinning above; and
  a KDA factorisation mismatch (one `qkv_proj` and one `conv1d` against the model's fused
  `in_proj_qkvbfg_a` and three `conv1d`). The last of those is **not an EXL3 problem** — `conv1d` is
  BF16, so a BF16 copy of the same checkpoint fails identically. Eight anchors, four asserts, all
  silent on the live boot, and with the flag unset the image is upstream byte for byte and still
  serves the control checkpoint at 0 unmapped / 0 unfilled. On the way: **`ReplicatedLinear` has no
  `weight_loader_v2` dispatch**, so any replicated linear served by a v2-parameter quantization method
  fails at load — reported, and fixed upstream in the plugin rather than worked around.

- **What it cost, and the line is not empty.** At TP=2 the full-scope model leaves a **31,343-token**
  KV pool, so prompts above ~2,000 tokens are **never scheduled** — every prefill figure and the
  C6/C8 aggregates are void in that arm, and TP=2 is a measurement rig rather than a serving
  configuration. It is also **~10 GiB heavier per node** despite being 10 GiB smaller on disk, and
  that contradiction is unexplained and left standing; it is the open risk for the TP=3 pool
  (4.70 M → possibly ~3.4 M) and the one number to measure before spending a boot.
  Production was down 1 h 15 m, planned, and came back at 4,696,969 tokens against 4,699,724, gates
  full.

- **Two instruments failed and are written down.** `CUDA_EXL3_DEBUG_NAMES=1` printed nothing — the
  plugin logs it at `info`, and this image configures only vLLM's logger — so the designed boot gate
  was unusable and three indirect checks carried it instead; the author has since fixed the level and
  made it log the hits as well as the misses. And a fast-load sidecar **refused the production
  restore** because an experimental patch had been placed in the hashed patch directory: correct
  behaviour, an hour late. New rules: [docs/09](docs/09-measurement-protocol.md) §11.2 (nothing is
  added to the patch directory between a dump and a load; experimental patches live elsewhere and are
  hard-linked) and §11.3 (the meta-device name-set check as the tier-A acceptance test for any loader
  change).

- **Upstream.** Four commits from the kernel author came out of this thread in one afternoon —
  a checkpoint-declared packed mapping (`5903248`), the same mapping from an environment variable
  (`fba9f27`), the `ReplicatedLinear` `suh` fix plus a corrected README example (`d19dee0`), and
  `CUDA_EXL3_DEBUG_NAMES` printing at all (`807d798`) — plus the one answer that unblocks TP=3 in
  principle: **all 65,536 trellis codes swept through the device decoder on all three codebooks, zero
  non-finite, bounded ranges**, so a zero pad trellis multiplied by `svh = 0` cannot produce NaN.
  [CREDITS](CREDITS.md).

- **What is open.** TP=3: `padding_size` 192 → **384**, shared expert 2112 → **2304** (not 2176 —
  128-aligned but not divisible by three), the A9 KDA split fix which is written down and not written,
  the checkpoint onto the third node, a re-dumped sidecar, and the padded-load path for the
  vocab-parallel head on the plugin side. [docs/11](docs/11-open-issues.md) §2.22,
  [docs/13](docs/13-full-scope-checkpoint.md) §7.

---

## 2026-09-05 (afternoon) — production configuration 8, and a profiler that deleted two of our own targets

Two entries in one day, and the interesting one is not the production change.

- **Production configuration 8** `[measured-here]`. Production 7 with the image moved to
  `exl3-zeus:62f53e6` — upstream's `had_in` commit `a47da6e` plus the note that bounds what is left of
  it. Sidecar re-dumped for the new image, `.env.tp3` promoted, **nothing else changed**:

  | | production 7 | **production 8** |
  |---|---|---|
  | KV pool @ 0.80 | 4,699,724 | 4,674,931 |
  | C1 / C2 / C4 / C6 / C8 aggregate tok/s | 57.0 / 80.9 / 120.0 / 143.4 / 175.1 | 56.8 / 83.5 / 119.5 / 146.0 / 172.8 |
  | prefill-fresh (3 rounds) | 1,733 / 1,769 / 1,788 | 1,769 / 1,780 / 1,789 |
  | draft acceptance | 60.8–64.3 % | 62–64 % |
  | gates cold and warm | 10/10 · 12/12 | 10/10 · 12/12 |

  **Every column is inside its own band and that is the entry.** The commit was priced at 0.2–0.3 % of
  prefill wall *before* it was built — deliberately below what a serving benchmark can see — and it was
  adopted to keep the image on upstream's head, not to buy tokens. Four signs in four directions, no
  mechanism, three rounds each. A stack that publishes 1 % changes as wins would have called two of
  these. [docs/10](docs/10-results-and-roofline.md) §1, [docs/11](docs/11-open-issues.md) §2.19.

- **The step-time breakdown is now a measurement, and it cost one launcher flag** `[measured-here]`.
  A torch profiler ran against the **live** production 7 server, all three ranks, with no restart and
  no reconfiguration — only `/start_profile`, `/stop_profile` and ordinary requests. This vLLM takes
  the profiler as `--profiler-config` and **not** as `VLLM_TORCH_PROFILER_DIR`; with the environment
  variable alone the route is never attached and the endpoint answers 404, which is why
  [docs/10](docs/10-results-and-roofline.md) §5 spent a week as a reconciliation with a 2.8 % residual.
  The flag is now in `scripts/start-tp3.sh` behind `PROFILER_DIR`, off by default and free when unset.
  Full tables: [`results/profile/measured-prod7.md`](results/profile/measured-prod7.md); how-to and the
  traps: [docs/09](docs/09-measurement-protocol.md) §4.1.

  What it found, in order of consequence:

  | | re-derived | **measured** |
  |---|---|---|
  | prefill chunk size | 2,032–2,048 tokens | **1,792** (7 × 256; 12.5 % of the budget unused) |
  | `exl3_moe_combine`, prefill | 1.5 % — ranked target #10 | **0 % — the kernel does not exist in this build** |
  | `_zero_kv_blocks`, prefill | 1.3 % — ranked target #8 | **0.09 %** (0.86 ms, not 14.7) |
  | DFlash2 drafter, C1 | 18.5 ms, 19.5 % | **10.78 ms, 11.4 %** |
  | NCCL, prefill | 14–17 % band | **14.47 %** — the bottom of it |
  | C8 decode split | `[not tested]` | MoE **51.6 %**, dense GEMM 21.1 %, NCCL 11.7 % |
  | comm/compute overlap | assumed some | **0.00 ms/step at prefill** — every microsecond of NCCL is exposed |

- **Retracted: "5.45 ms of C1 GPU idle" and "CUDA graphs would return ~6 % of single-stream"**
  `[retracted]`. Both were published, one of them upstream. **~2.0 ms of that idle is CUPTI itself** —
  about 1 µs per kernel boundary across ~1,873 of them — so the real budget is **3.47 ms = 3.75 %** of
  a profiler-off step. And "CPU gap" was the wrong name: **77 % of it is per-kernel dispatch**
  (2,332 kernels per step), 18 % the host, 5 % one blocking sync — with the host running 3.9 ms
  *ahead* of the GPU at C1. Graph capture is worth **1.4–1.9 ms, +1.5–2.1 %**, and removes neither the
  step head's `prepare_inputs` nor the sync. This also explains something that had been sitting
  awkwardly: the previous configuration *did* capture graphs and read the same 57 tok/s. Retractions
  26–28 in [docs/11](docs/11-open-issues.md) §1.9, story in §1.10.
  **The rule this adds to the protocol:** never read an idle figure out of a trace — take
  `busy(union)` from the trace and the wall from a profiler-off run of the same window.

- **Closed by the kernel author, not by us: MLA prefill** `[reported]`. It had been carried as
  "8.2 % of a chunk, efficiency not measured, no denominator" for a week. Measured at our shapes —
  top-k 2,176, head_dim 512, fp8 cache, a 2 GB pool chosen so it could not sit in L2 — it runs at
  **86–89 % of achievable**. It sits with the trellis GEMM: near the roof, no traffic to remove. Worth
  recording that his first cut used a 200k-row pool that fitted L2 and duly reported a bandwidth above
  the DRAM ruler; the same class of error as our own model-free MoE bench overstating small-M cost by
  1.5–1.7×. **Synthetic benches flatter small inputs, and both of us hit it the same day.**

- **New open item, and it is the largest one this repository has ever carried: the checkpoint's scope**
  `[estimate]`. Dense BF16 GEMM is **45.3 % of a C1 step** because we run
  `scope: glm53_routed_experts_only`. At 4 bpw that stage is bandwidth-bound on ~4× less traffic —
  42.9 ms → ~11 ms, **~+34 % single-stream** — against about 5 % for the whole `cuda-exl3` column of
  our target table. The TP=3 blocker is softer than [docs/01](docs/01-model-and-license.md) §3.1 said:
  an EXL3 tensor still cannot be zero-extended, but a checkpoint quantized *unpadded* can be loaded
  into a padded parameter with `svh = 0` on the pad — bit-exact, no re-quantization — **provided the
  pad occupies whole 128-column blocks**. Our head pad does (64 → 66 heads = 256 columns = 2 blocks);
  our vocab pad does not (`padding_size=192` = 1.5 blocks) and would at **`padding_size=384`**, which
  is one constant in our launcher. Ordering agreed upstream: **quality first** — a TP=2 trial of
  `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw against the MMLU sample, reported either way — then the
  padded-load path behind a flag if the gate holds. [docs/11](docs/11-open-issues.md) §2.22.

- **Docs.** `scripts/start-tp3.sh` gains the `PROFILER_DIR` arm — and with it a rule:
  **one launcher, one copy.** Two copies of this file existed for two days; the one the nodes ran had
  the profiler arm and the one in the tree did not, which is precisely the divergence that is invisible
  because both copies produce a working server. [docs/08](docs/08-fast-boot.md) §12.
  New: [docs/09](docs/09-measurement-protocol.md) §4.1 (profiling a live server, and verifying the
  profiler), [docs/10](docs/10-results-and-roofline.md) §5.7–§5.8,
  [docs/11](docs/11-open-issues.md) §1.10, §2.22, §2.23,
  [`results/profile/measured-prod7.md`](results/profile/measured-prod7.md).

---

## 2026-09-05 — production configuration 7: the draft cache at fp8, and a ruler that had been moving all along

The pool moved for the first time in three configurations, and the more useful half of the entry is
that we found out why it had been moving on its own.

- **Production configuration 7** `[measured-here]`. Production 6 plus an **fp8 DFlash2 draft cache**
  (`HAREM_DRAFT_KV_DTYPE=fp8`), a tilelang fail-loud guard, a FlashInfer warm-up, an idle profiler
  endpoint, the launcher's new memory settle gate and a fresh per-rank sidecar. Read from an ordinary
  load boot:

  | | production 6 | **production 7** |
  |---|---|---|
  | KV pool @ 0.80 | 4,449,035 | **4,699,724 (+5.6 %)** |
  | C1 / C2 / C4 / C6 / C8 aggregate tok/s | 56.9 / 84.2 / 118.5 / 142.9 / 168.9 | 57.0 / 80.9 / 120.0 / 143.4 / 175.1 |
  | TTFT C1 / C8 | 0.41 / 1.01 s | **0.34 / 0.91 s** |
  | draft acceptance (per-concurrency medians) | 61–65 % | 60.8–64.3 % |
  | prefill-fresh / warm 7K repeat | 1,792 / 1,506 | 1,769 / 1,529 |
  | gates cold and warm | 10/10 · 12/12 | 10/10 · 12/12 |
  | free RAM / swap | 11.3 / 12.6 / 12.5 GiB · ~0.1 | 12.3 / 13.5 / 13.3 GiB · ~0.1 |

  **Speed is unchanged and is not the claim.** C8 reads +3.7 % and C2 −3.9 % on the same three rounds:
  opposite signs, both inside their own bands, and no mechanism by which a smaller draft cache would
  make decoding faster. The claim is the pool, at **+5.6 % against +4.7 % predicted**, and a TTFT that
  improved at both ends. [docs/07](docs/07-kv-and-draft-page.md) §7,
  [docs/10](docs/10-results-and-roofline.md) §1 and §2.1, [docs/11](docs/11-open-issues.md) §2.18.

- **The KV pool number was never a memory reading** `[measured-here]`. On this integrated-GPU part vLLM's "free GPU memory" is `/proc/meminfo`
  `MemAvailable`, and "consumed memory (weights + non-torch)" is the *difference* between two such
  readings taken minutes apart. The instrument therefore runs backwards — a node that starts dirty
  computes itself a **larger** pool — and the launcher made it systematic by killing a ~90 GiB
  container and starting the next immediately, in a fixed node order, so the last node started was
  always ~9 GiB short — **27 % of a rank's KV allowance** sitting inside the measurement. **Fix: a
  host-side settle gate** in `scripts/start-tp3.sh`, which waits after `docker rm -f` until
  `MemAvailable` is back over `SETTLE_MIN_GIB` (112) before `docker run`. Per-rank startup free memory:
  spread **9 GiB → 1.4 GiB**. KV tokens bought: **zero**, and **no published figure is corrected** —
  the pool takes the minimum over ranks and the polluted node happened never to be the binding one,
  which is luck, not design. Two rules now stand with it: **read a pool only from a load boot**, and
  **check all three ranks agree within 1 GiB before quoting one**.
  [docs/07](docs/07-kv-and-draft-page.md) §1.1, [docs/08](docs/08-fast-boot.md) §5.1,
  [docs/09](docs/09-measurement-protocol.md) §11.1.

- **Retracted, and it was the largest open item on the page: "8.2 GiB per worker is stranded"**
  `[retracted]`. Yesterday's entry priced equalising the ranks at 8–26 % of pool. There is no stranded
  KV. The three ranks start 9.00 GiB apart and finish the profile **0.99 GiB** apart, so the gap was
  never an allocation; live readings during the same boot have the head node with *less* free memory
  than the workers, where the profile had given it 8.2 GiB more; and the pool is sized on the minimum
  over ranks, which is the correct figure. Acting on the claim would have over-committed the head node
  by ~8 GiB — the exact recipe for the 0.85 boot that swapped. This is retraction 25 in
  [docs/11](docs/11-open-issues.md) §1.9, and it is the same failure as the datasheet rulers with a
  better disguise: **the number was printed by the engine itself, about its own memory, and it was
  still an unverified ruler.** [docs/11](docs/11-open-issues.md) §2.3.

- **`NCCL_ALGO` swept model-free: `Tree` rejected, `Ring,Tree` unresolved** `[measured-here]`. Three
  arms, two repetitions, production plugin and environment. `Tree` is 4–6× slower than Ring at 16 and
  64 MB all-reduce (3.15/3.36 against 20.23/18.30 GB/s at 16 MB), 23–96 % worse on the decode-step
  proxy, its RNR retries are an order of magnitude higher, and the port counters show it
  redistributing traffic asymmetrically across three nodes that have no hierarchy to reward it.
  `Ring,Tree` is 3.6 % better on the step proxy and worse on `sendrecv` at 64 MB — **and the sweep's
  own repeat-to-repeat spread is up to 1.7× at 1 MB**, so model-free cannot settle it. Deferred to a
  five-round engine arm worth an expected −1…3 % of a decode step; `Ring` stays in the launcher.
  [docs/06](docs/06-nccl-mesh.md) §12.2, raw in
  [`results/mesh/algo-sweep.md`](results/mesh/algo-sweep.md).

- **The `cuda-exl3` MoE stage is closed upstream** `[reported]`. After taking `exl3_moe_had_in` in
  `a47da6e`, the author bounded the remainder in `62f53e6`: what is left is a half-ALU limit — a
  128-point Hadamard done with warp shuffles — so it is arithmetic that has to happen rather than
  traffic that can be removed, worth **≤2 % of prefill** here and unreachable in practice. He also
  confirmed our reading that `_zero_kv_blocks_kernel` belongs to vLLM rather than to the kernel
  library. Every remaining prefill lever on this stack now belongs to vLLM, to the fabric, or to us.
  [CREDITS](CREDITS.md), [docs/10](docs/10-results-and-roofline.md) §6,
  [docs/11](docs/11-open-issues.md) §2.19.

- **New open item, and it is a decision rather than a measurement:** `gpu-memory-utilization 0.83`.
  The knob multiplies `MemTotal`, so it is the one pool input the baseline cannot corrupt: +3.65 GiB
  of budget landing whole on the binding rank, pool ~4.70M → about **5.2M (+11 %)**. Price stated:
  the OS share falls 20 % → 17 % and the head node goes from ~12.3 to ~8.4 GB free under load, still
  clear of the 4 GiB rule. One boot, reversible in one line, **held for the stack owner's approval**
  `[not tested]`. [docs/11](docs/11-open-issues.md) §2.4, [docs/07](docs/07-kv-and-draft-page.md) §6.

---

## 2026-09-05 — a kernel that half-worked, an overlap that does not pay, and a stricter definition of "equal"

Production configuration 6 is **unchanged**, for the second entry running. One change is validated and
waiting on a boot; three designs were costed and rejected; and the measurement protocol got stricter
about what counts as a difference.

- **Draft KV at fp8: validated, not yet promoted** `[measured-here]`. Moving the DFlash2 drafter's own
  cache from bf16 to fp8 shrinks its page 393,216 → **196,608 bytes** and the per-block cost
  21,917,440 → **20,934,400**, with the blocks-per-request divisor unchanged at 363 — worth about
  **+4.7 %** of pool. The arm booted, which answered the only question a boot could: the DFlash
  sliding-window backend accepts an fp8 cache. Draft acceptance **60.1–64.0 %** against production's
  61–65 % and a gate that demanded the 60–65 band, gates 10/10 · 12/12 cold and warm, speed inside the
  noise bands. **It ran on a dump boot, so its KV pool figure (4,382,920) is meaningless** and
  production stays on bf16 until an ordinary load boot supplies the real number.
  [docs/07](docs/07-kv-and-draft-page.md) §7, [docs/10](docs/10-results-and-roofline.md) §2.1,
  [docs/11](docs/11-open-issues.md) §2.18.
- **The hyper-connection fusion kernel was written and it reached 40 % of its own ceiling.** A Triton
  kernel that tiles over tokens and reduces the post mapping in registers removes 30.4 % of the first
  two kernels' traffic and delivers **−14.9 to −15.5 %** on that pair, **−9.0 to −9.9 %** on the
  three-kernel route, and **−1.0 to −1.1 % of the prefill wall** — against a −2.1 to −2.8 % target.
  `residual_cur` is bit-identical at every M but one 7-element 1-ulp difference at M=64; `layer_input`
  is within one bf16 ulp on 5.1 % of elements. The shortfall is entirely bytes-per-second (187.7
  against the 229.5 GB/s the route it replaces gets), not tiling — a 33-configuration sweep could not
  improve the winner. **Not adopted standalone**: −1 % does not earn a boot when it also brings Triton
  JIT into the serving process and a configuration surface that is a cliff (the winner reads
  187.8 GB/s, its neighbours 79.4 and 44.5, and the shipped default was one of the bad ones). It rides
  the next image bundle with `had_in`. Also measured: fusing **loses** below M ≈ 1024 (+37.7 % at
  M=512), because the residual fits the 24 MiB L2 there and the re-read it deletes was never going to
  DRAM `[measured-here]`. [docs/10](docs/10-results-and-roofline.md) §5.5.1.
- **New lesson, and the more useful half of that kernel:** a GPU-free ahead-of-time compile check
  reported 18 of 18 configurations building and all inside the shared-memory limit; at real launch
  **6 of the 18 failed with `OutOfResources`**, the reported 36,864 bytes against 106,496 actually
  needed. **A compile check answers "does it build", never "does it run"**
  `[measured-here]`. [docs/11](docs/11-open-issues.md) §4.
- **Closed: dual-batch overlap (DBO).** The all-reduce is 16.5 % of a prefill chunk at 99.3 %
  occupancy, so overlapping it carried the largest number on the open-issues page. The mechanism does
  suit it and the patch is *smaller* than we estimated (~95–160 lines, five files, and it does not
  touch the model file — one bottleneck covers all 102 collectives, and the mHC state is thread-local,
  so our "medium-to-large patch, mHC at risk" reading was wrong on both counts `[retracted]`). The
  arithmetic kills it: splitting the batch pays the MoE expert weight stream **twice**, +73 to +232 ms
  per chunk against −135 ms of hideable collective. Prefill lands at **−6.3 % to +8.0 %** — a coin
  toss — and decode is a clear loss (**C1 +38 %**, C8 +6 %), with the drafter unable to micro-batch,
  the breakable CUDA graph disabled for both models, and a silent-corruption hazard where a split
  request restarts the KDA recurrent state from zero in 34 of 45 layers. Raising the batched-token
  budget beats every overlap variant for the same KV price and no code. **Do not build it.**
  [docs/11](docs/11-open-issues.md) §2.17.
- **Also dead, each for a checkable reason, read out of the image with the engine down:** async
  tensor parallelism, the sequence-parallelism pass and the FlashInfer all-reduce+RMSNorm fusion all
  need `torch.compile`, which this model family never enters; and `world_size = 3` is excluded by the
  supported-world-size lists of FlashInfer (2/4/8/16), custom all-reduce (2/4/6/8/16) and NCCL
  symmetric memory (minimum 4). DeepEP is installed and never engages at `data_parallel_size = 1`.
  Model-level sequence parallelism is a one-line gate and a bad idea: identical bytes, collective
  count 90 → 180, **+10…15 % worse at decode**.
- **One survivor in that class, and one free probe.** Attention-scoped micro-batching — split across
  the attention block only, rejoin before the MoE stage — hides 44 % of the collective while paying
  only for attention weights streaming twice: **−3 to −6 % of prefill** `[estimate]`, ~150 lines, and
  it inherits the same KDA hazard. Before any of it, a model-free probe should establish whether an
  all-reduce on a second stream overlaps a GEMM **at all** on this part. Written, not run
  `[not tested]`.
- **The measurement protocol got a floor.** Round-to-round spread inside a single settled arm is
  **C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %** `[measured-here]` — C4 is the noisiest column, which
  is the opposite of the intuition. Two rules now apply to every table here: **a difference of 3 % or
  less is written down as "equal"**, and above that floor it still has to clear its own metric's band.
  This reclassifies the combine-staging arm's +2.3 % at C4 and patch 0007's −0.9…+4.2 % as equal;
  production 5 → 6 survives on its +5.6 % at C8. Also corrected: the quick-arm harness claimed five
  rounds with two discarded and ran three `[retracted]` — it is now one warm-up plus three measured
  rounds, and the warm-up ramp it was written for is gone anyway on a warm tuner cache.
  [docs/09](docs/09-measurement-protocol.md) §1.1–§1.2.
- **New planning rule: a patch change costs a dump boot.** The fast-load sidecar's identity covers
  *every* `patch-*.py` and the whole prelude, so three patches that touch no weight byte refused a
  boot and cost an hour. Budget the 682-second dump boot into any arm that adds a patch, and never
  record a dump boot's KV pool as a result. The narrower gate this argues for is written up but not
  written. [docs/09](docs/09-measurement-protocol.md) §11, [docs/11](docs/11-open-issues.md) §2.21.
- **Rank memory imbalance, now with a number and a diagnosis label.** The weights are identical on all
  three ranks (`Model loading took 54.86 GiB` ×3), yet non-torch memory reads **1.50 GiB on rank 0
  against 9.48–9.72 GiB on the workers** — about **8.2 GiB per worker stranded**, and the pool is sized
  by the worst rank, so equalising it would be worth **8–26 % of pool**. Larger than every kernel item
  left. Nobody knows yet what that memory is `[measured-here]`.
  [docs/11](docs/11-open-issues.md) §2.3.
  **Correction, same day** `[retracted]`: it is not memory, it is the instrument. "Non-torch" is a
  delta between two `MemAvailable` readings and the last node started is the one given least time to
  reclaim the previous container; the three ranks finish the profile 0.99 GiB apart. No KV was
  stranded, and equalising the ranks would have over-committed the head node by ~8 GiB. See the entry
  for production configuration 7.
- **The memory ladder is re-opened rather than settled.** The 0.85 rejection predates the fast-load
  work that removed the page-cache spike; the same configuration now sits at 11–12 GiB free with zero
  swap, and 0.82–0.83 was never tried. `--kv-cache-memory` — sizing the pool in bytes rather than as a
  fraction of the device — has never been used. Ladder first, pin last
  `[not tested]`. [docs/07](docs/07-kv-and-draft-page.md) §6.
- **`NCCL_ALGO=Ring,Tree` has never been run on this mesh**, and it is the cheapest untried thing in
  the repository: our launcher forces `Ring`, decode is latency-bound on a fixed 102 collectives per
  step, and a tree is ~3.2 steps against a ring's 4 at `world_size = 3`. Expected −1…3 % of a decode
  step; the sweep is model-free and costs nothing `[not tested]`.
  [docs/06](docs/06-nccl-mesh.md) §14 item 8.
- **New `systemd/` directory — a template, deliberately not installed.** With it, the hazard it
  exists to name: the NVFP4 sibling's `harem-motor.service` is `enabled` on all three of our nodes, so
  a reboot brings up the **other** engine on the same GPUs. The template's three unfinished pieces are
  named rather than fixed — its preflight script does not exist, systemd will not honour the
  worker-2 → worker-1 → head start order on its own, and its `ExecStop` names the wrong container.
  [systemd/README.md](systemd/README.md), [docs/11](docs/11-open-issues.md) §2.20.
- **A complete retraction audit.** Every published claim of ours was re-read against the raw data
  behind it; **24 did not survive** and all 24 are now in one table with what replaced them, including
  ones that had only been corrected in passing: the chat template we serve matches neither checkpoint
  on disk and its provenance is unverified; `NCCL_BUFFSIZE` was listed as an open lever twelve hours
  after being eliminated; `NCCL_MAX_NCHANNELS=8` had been "already tried" only in combination with
  `NCCL_PROTO=LL`; `--language-model-only` does not stop the vision tower being built, only run.
  Six of the 24 are a ruler we quoted instead of measured, four are a single pair of sweeps treated as
  a result, three are an arithmetic model a bench refuted, two are our own tooling disagreeing with
  our own documentation. [docs/11](docs/11-open-issues.md) §1.9.

## 2026-09-05 — where a step actually goes, and four items closed

Production configuration 6 is **unchanged**. Everything in this entry is measurement.

- **Both rulers were wrong, by ~22 %.** Measured on this device in our own image: achievable read
  bandwidth **225.2 GB/s** against a vendor 273, BF16 GEMM peak **97.3 TFLOP/s** against an implied
  ~125. Every roofline percentage published here before today was optimistic by that much
  `[retracted]`. Two new tools, `bench/bw.py` and `bench/gemmpeak.py`, seconds each; run them in the
  same process as the thing you are measuring. The read ruler itself drifted 6.5 % across three runs
  on the same idle machine the same morning, so percentages are now given as bands where it matters.
- **Step-time breakdown, per 2,048-token prefill chunk** (1,109 ms, occupancy 99.3 %): MoE trellis
  GEMM **26.4 %**, NCCL all-reduce **16.5 %**, dense BF16 GEMM **16.2 %**, hyper-connection mixing
  **11.7 %**, MLA 8.2 %, KDA 7.5 %, MoE `had_in` 6.1 %, KV zeroing 1.3 %, DSA indexer 0.6 %. Per C1
  decode step (89.1 ms): dense BF16 GEMM **44.8 %**, MoE trellis GEMM 29.3 %, all-reduce 10–15 %, and
  the k=7 drafter **19.5 %**. Since the previous configuration, prefill throughput at 8.4K is
  **+43 %** and the C1 step is **−17.5 %**.
- **Two corrections to our own earlier reading of the same trace** `[retracted]`: the `mhc_*` kernels
  are hyper-connection mixing, a class of their own worth 11.7 %, not dense GEMM and not the indexer;
  and MLA is 7.4 % in a steady chunk, not the 9 % a window average reported.
- **Method, stated because it has a caveat.** The running engine had no profiler endpoint and a
  restart was not available, so this is a reconciliation — structure from an earlier trace
  re-segmented per chunk, changed classes re-measured model-free, totals measured live — with a
  **2.8 % residual**. Read NCCL as a 14–17 % band. One profiling boot closes it.
- **Closed: expert-stationary MoE scheduling.** Our 14–27 % traffic estimate rested on a traffic
  model a trace cannot verify. The kernel author wrote a bench for it (`9b17ea9`); run unmodified on
  GB10, three times: doubling blocks per expert costs **1.11×**, not 2×. The trellis stays resident.
  Nothing to win `[measured-here]`.
- **Closed: the KV-zeroing gate.** The kernel runs at 100 % of the memset roofline and zeroes
  2.4–2.9 GB per chunk against ~3.4 MB of real new KV, so the only lever was to skip it. It cannot be
  skipped here: **85.5 %** of those bytes are MLA pages co-owned with Mamba/KDA state in this model's
  hybrid layout, which is the Mamba half of vLLM's condition and is independent of precision. A
  fail-closed gate was written so the machine checks the conclusion; on this model it refuses to boot,
  by design. Safe remainder 0.19 % of prefill; no partial mode written `[measured-here]`.
- **Closed: a cooperative (`grid.sync`) MoE stage.** Outside a CUDA graph the barrier wins on a 48-SM
  part, as predicted — up to 33 % at medium sizes. **Inside a graph, which is what production runs,
  the sign flips back** and it costs 0.2–0.3 µs per phase boundary. The deciding detail was the
  graph, not the SM count `[measured-here]`.
- **Closed: the DSA indexer**, at 0.6 % of prefill — the same conclusion an earlier micro-benchmark
  reached from the other side, now confirmed from the share itself.
- **Hyper-connection mixing measured**: 86–91 % of the ruler, memory-bound by a factor of 76, **not**
  launch-bound (CUDA graphs change it by 0.03 % at M=2048), **not** badly tuned (the two available
  knobs are worth 0.4 % of prefill, and the third kernel does not compile above 96 threads), and the
  torch fallback is unreachable on CUDA and 5–15× slower. One real lever remains: fusing the first two
  kernels to stop re-reading the residual, **−2.5 to −2.7 % of prefill**, which needs a new large-M
  kernel — forcing the existing fused one is +32 % worse. Our earlier −3.6 % estimate was 30 %
  optimistic `[retracted]`.
- Upstream took `exl3_moe_had_in` in `a47da6e` (−10…18 % on that kernel, ~0.2–0.3 % of prefill here).
  **Not in the production image**; queued for the next build.
- New protocol rule, learned the hard way: **one measurement holds the cluster at a time**, written in
  a lock file, and three-node NCCL work needs the engine **down** rather than idle — a fabric sweep
  beside a live engine can exhaust queue-pair resources and take its next collective with it.
  [docs/09](docs/09-measurement-protocol.md) §10.

## 2026-09-05 — the fabric ceiling is PCIe, and a transport rewrite that changed nothing

- **Retracted, one day after we published it** `[retracted]`: "the pair of cables is worth 50 GB/s,
  so the collective is at 28 % of the fabric". That is the **wire**. Each ConnectX-7 sits in a
  **PCIe Gen5 x4** slot (`LnkSta: Speed 32GT/s, Width x4`) and carries ~15 GB/s regardless of its two
  200 Gb/s ports, so the real ceiling is **~30 GB/s per node** and the collective at ~20 GB/s is at
  about **70 %** of it. The old 13 GB/s ceiling was never "half a link" either — it was one card's
  PCIe limit at 87 % of it, which is why the second cable, on the second card, took it to 20 and not
  to 40. **Remaining fabric headroom ≤30 %, worth 2–4 % of prefill**, against the 12–17 % we had
  priced it at. Two wrong ceilings in two days, both computed from a datasheet.
- **Patch 0007: one-sided transport (receiver-advertised FIFO + `RDMA_WRITE_WITH_IMM`), built,
  measured, not adopted.** +977/−16 lines over 0004–0006: FIFO in the sender's memory (one per cable
  after 0005), zero-byte RECV armed before the slot is advertised so **RNR becomes structurally
  impossible**, torn-slot double check, fail-closed ring overrun, a version handshake that keeps
  `mesh_qp_info` at 32 bytes and refuses a peer that does not speak the extension, and no dependence
  on NCCL's `*request = NULL` contract. Default `send` is byte-for-byte the old path.
- **It does exactly what it was designed to do and it is worth nothing here.** Six arms, two
  repetitions: every write arm reports **zero** RNR retries and **zero** out-of-buffer events at every
  size, against 1–9 per operation on the control — and throughput does not move. The gate (≥1.3× at
  ≥16 MB) was not met at any FIFO depth, with or without the flush, and all six arms sit within 0.17 ms
  of each other on the decode-sized message. Engine arm: C1 56.4, C8 171.1, prefill-fresh 1,763 against
  production's 56.9 / 168.9 / 1,792 — differences in both directions, inside boot spread, gates
  10/10 · 12/12 cold and warm `[measured-here]`.
- **Why**: at ~20 GB/s the transfer is against a PCIe wall, not a flow-control stall. Removing RNR
  from a path that was not waiting on RNR buys nothing.
- **Kept, not deleted, and not offered upstream.** The patch, its unit-test parity with the 0004–0006
  baseline and the measurement that rejected it are in the repository. Sending a transport rewrite
  upstream on the strength of a mechanism that moves no number would waste the maintainer's time.
- **Patches 0004, 0005 and 0006 are now on a public fork and offered as a pull request** —
  [`NNNtrance/nccl-mesh-plugin`](https://github.com/NNNtrance/nccl-mesh-plugin), branch
  `gb10-dual-link-ptrcuda` on `19924dcc`, [PR #59](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59),
  referencing the issue thread the findings were reported on. 0007 is not in it.
- **Closed as an open item**: the collective's share of a step, which had been "the cheapest unspent
  measurement in this repository" through three separate changes. It is 16.5 % of prefill and 10–15 %
  of a C1 decode step. The C8 split still needs a profiling boot.
- **Newly written down as the larger lever**: the all-reduce is *serialised* against compute at 99.3 %
  occupancy. Making the fabric faster is worth ≤2–4 % of prefill; making the collective **overlap**
  reaches for most of 16.5 %. Nobody has tried. [docs/11](docs/11-open-issues.md) §2.17.

## 2026-09-05 — production configuration 6: both cables, and no host bounce buffer

- **Half the fabric had never carried a packet.** Two cables run between every pair of nodes;
  `mesh_connect()` discards NCCL's device index and stops at the first reachable peer address, so
  every channel to a peer rode one cable. `port_xmit_data` on the second cable of each pair read
  **exactly zero since driver load**, on all three nodes. The 13 GB/s we had been calling "the link"
  was one cable of a pair — and, as the next day's entry records, it was also one card's PCIe limit.
- `patches/kernel/0005-device-aware-link-selection.patch` (~30 lines, no wire-format change) picks
  `dev % usable` among the parallel links; with one cable, or `NCCL_MESH_LINKS_PER_PEER=1`, the
  selection is bit-identical to the stock plugin. 64 MB all-reduce **12.0 → 16.7 GB/s**.
- `patches/kernel/0006-ptr-cuda-dmabuf-and-flush.patch` advertises `NCCL_PTR_CUDA` — two lines; the
  plugin's `regMr` already registered CUDA pointers and NCCL was simply never handing it one — plus a
  real RDMA_READ `iflush` and a real DMA-BUF path. 64 MB all-reduce **16.7 → 20.8 GB/s**, RNR retries
  per operation 15 → 3.
- Engine, three rounds per arm: **C1 54.5 → 56.9, C4 112.0 → 118.5, C8 159.9 → 168.9** (+4–6 %),
  prefill-fresh 1,709 → **1,792**, TTFT C1 0.47 → 0.41 s, gates 10/10 · 12/12 cold and warm.
- **What it cost:** nothing measurable — the KV pool moved +0.4 % (inside boot noise), which was the
  line to watch because `NCCL_PTR_CUDA` moves NCCL's buffers into memory accounted under
  `gpu-memory-utilization`. The real price is not a number: a patched plugin to maintain.
- **`NCCL_MAX_NCHANNELS=16` rejected.** Restoring 8 channels per cable is the obvious follow-up and
  it is 2.5× slower on the decode-sized message (STEP90 9.3 → 26.2 ms), in both patch families. 8
  stays.
- **`NCCL_MESH_DMABUF=1` rejected.** It works — which settles whether `ibv_reg_dmabuf_mr` accepts
  these buffers here — and is slower than plain `ibv_reg_mr` (64 MB 18.1 against 20.8).
- Patch `0004-min-rnr-timer` is carried into the production build at `NCCL_MESH_MIN_RNR_TIMER=1`
  rather than staying on the shelf. Its isolated engine contribution is still unmeasured.
- **Retracted:** "the ceiling is ~13 GB/s against a 25 GB/s link, and the GPUDirect path needs a
  plugin redesign". The link is a pair of cables with one idle, and the device-pointer path was two
  lines. See [docs/11](docs/11-open-issues.md) §1.6. (The "50 GB/s pair" in this entry was itself
  retracted the next day — the ceiling is the cards' PCIe slots, ~30 GB/s per node; §1.7.)
- Reported upstream as a follow-up on the plugin's issue thread, with both patches offered.

## 2026-09-05 — production configuration 5: the MLA tuner cache stops charging us for measurement

- Image moved to `cuda-exl3` `9bf594c` ("Persist the MLA tuner cache across processes"), which adds
  `CUDA_EXL3_TUNE_CACHE`. The tuner's map had been process-local, so every boot re-tuned and every
  unseen batch shape bought a ~15 ms tune **while serving** — which is what polluted the first rounds
  of every A/B on this stack.
- Measured: **18 tune events before serving → 0**, none during a sweep, and round 1 stops being a
  penalty (cold cache C8 round 1 → round 3 −3.4 %; warm +2.7 %, i.e. unordered noise).
- **Protocol change:** five sweep rounds with two discarded → **three rounds, median of three**, on
  an image with the cache and a warm cache file. About 15 minutes saved per arm. Five rounds still
  stands everywhere else, including the boot that writes the cache.
- Speed unchanged by design: C1 54.5, C8 159.9 against 54.4 and 161.8, inside the spread. Gates
  10/10 · 12/12. KV pool 4,429,752.
- **What it cost:** a new image invalidates the fast-load sidecar, because the manifest records the
  image tag — the preflight refused the boot, correctly, and regenerating cost one **682 s** dump
  boot on all three nodes. That is now the standing price of every kernel-image change.
- The implementation is upstream's; ours was the measurement that asked for it. Credit in
  [CREDITS.md](CREDITS.md).
- Closes the open item in [docs/08](docs/08-fast-boot.md) §10.3 and
  [docs/11](docs/11-open-issues.md) §2.8. New page: [docs/12](docs/12-tuner-cache.md).

## 2026-09-05 — production configuration 4: fast boot

- Per-rank pre-sliced fast-load sidecar (`patches/tp3/harem_fastload.py` and friends): boot
  **618 s → 274 s**, weight load **426 s → 67 s**. Bit-identity proven twice — 1,475/1,475 tensors
  re-hashed against a manifest written from a full-checkpoint load, and a post-`process_weights` hash
  dump with no difference on any rank.
- `--enable-ep-weight-filter` plus a patch so the filter recognises EXL3's `.trellis` suffix: each
  rank now reads 96 of 288 experts instead of all of them.
- `--safetensors-load-strategy eager` and `--no-enable-flashinfer-autotune`.
- Page-cache remedy (`posix_fadvise(DONTNEED)` on checkpoint shards, `malloc_trim`) after a 4.1 % KV
  regression traced to reading 163 GB through the page cache on unified memory. Pool came back
  **above** the pre-change baseline: 4,484,848.
- Speed and quality unchanged, which was the intent. Gates 10/10 · 12/12 cold and warm.
- **Corrected:** `--no-enable-flashinfer-autotune` is worth about 3.5 s, not the 34 s it appears to
  be — most of that work moves into graph capture. See [docs/08](docs/08-fast-boot.md).

## 2026-09-05 — production configuration 3: the draft KV page

- Root cause found for a KV pool that was capped by a per-request **block counter**, not by memory:
  the DFlash2 draft's sliding-window group is given the backend's smallest kernel block (16 tokens),
  and because the port keeps that group independent it never reaches the unification step that would
  scale it up. The draft took **53 % of the blocks-per-request budget for 0.6 % of the memory**.
- `HAREM_SW_BLOCK_SIZE=256` (`patches/tp3/patch-swblock-tp3.py`, one anchor, env-gated):
  KV pool **2,428,769 → 4,413,223 (+82 %)**. 256 is the measured optimum; matching the target's
  3,328-token page makes the pool 7 % *worse* than doing nothing.
- Unpredicted bonus: the draft block table shrinks 16×, giving C4 +9 %, C8 +6 % and 20–30 % off TTFT.
- Cost: +9.2 % memory per block, and the draft group's prefix-cache matching unit coarsens from 16 to
  256 tokens.
- Memory ladder step to `gpu-memory-utilization 0.85` measured (+19 % pool, no speed change) and
  **rejected**: head node at 1.9 GiB free with 1.6 GB of swap in use, which breaks the 4 GiB rule.
  0.88 not attempted.

## 2026-09-05 — the NCCL mesh cliff

- Root cause for an all-reduce running at 0.6–1.9 GB/s between 128 KB and 4 MB while point-to-point
  over the same queue pairs stayed clean at 11–13 GB/s: the mesh plugin carries data with two-sided
  SEND/RECV, its only flow control is the receive-not-ready NAK, and it sets `min_rnr_timer = 12`
  (0.64 ms) where its own comment intends 0.01 ms (code 1). NCCL opens 64 channels on this fabric and
  one proxy thread services them round-robin, so the sender routinely outruns the receiver.
- **`NCCL_MAX_NCHANNELS=8`** adopted, environment only: model-free, the 512 KB decode all-reduce goes
  1,195 → 123 µs and one decode step's collectives 91.7 → 9.9 ms. In the engine, three boots and five
  rounds each: **C8 133.4 → 150.8 (+13.0 %)**, C1 +7.1 %, C4 +9.6 %, C6 +11.8 %; prefill, gates,
  acceptance and KV pool unchanged.
- Plugin patch (`patches/kernel/0004-min-rnr-timer.patch`) written, unit-tested and **not deployed** —
  with the channel cap its contribution is inside the noise.
- **Retracted:** an earlier reading that NCCL was choosing the LL protocol at 16 MB. Forcing LL there
  costs 20,114 µs against auto's 1,787 µs.
- **Protocol change:** five sweep rounds per arm, first two discarded. The tuner's warm-up window can
  be longer than two rounds and made the winning arm look 25–45 % worse on the first pass.

## 2026-09-05 — production configuration 2

- Image moved to `cuda-exl3` `f4987cf` ("do not fetch the MoE padding rows");
  `--max-num-batched-tokens` back to 2048 from 4096, recovering the KV pool (1,627,170 → 2,428,769)
  at a cost of ~5 % on fresh prefill.
- MoE input-transform fusion A/B (`61a17bc`): +1–4 % end to end over five rounds, but `f4987cf`
  reaches the same level another way and upstream dropped the fusion branch. Closed.
- k=7 versus k=5 A/B: k=5 raises the acceptance *rate* (63 → 73 %) and lowers accepted tokens per
  *step* (5.5 → 4.7); the second effect wins everywhere except prose and C4. **k=7 stays.**

## 2026-09-05 — upstream adopted, our patch retired

- Per-kernel comparison of four builds on identical shapes: our "zero the retired tile" choice is
  **10.7 % more expensive per MoE layer at M=2048** than upstream's "return, and let the combine skip
  those rows". Our `0002` patch retired in writing.
- Our one surviving kernel change kept and rebased: staging the combine's per-(token, k) facts in
  shared memory — combine −34 % at M=8, −36 % at M=64, −13 % at M=2048; end to end C6 +3.8 %,
  C8 +3.8 %.
- **Retracted:** the earlier report that an upstream build measured ~10 % slower end to end. It does
  not reproduce; what we had measured was **15.9 % boot-to-boot spread on C8 with nothing changed**.
- **Retracted:** the claim that the missing `n_rows` also costs the non-expert-parallel path. With no
  expert map the surplus tail is `-1` everywhere.
- `--max-num-batched-tokens 4096` adopted at this point: +9.5 % fresh prefill, −13 % mixed-load TTFT,
  −28.5 % KV pool. Later reverted, above.
- `NCCL_PROTO=Simple` rejected model-free without spending a boot: 2.8× worse at the C1 decode
  message, 4.4× at C8, no better at 16 MB.

## 2026-09-05 — the expert-parallel regression, root-caused

- TP=3 + expert parallel was **8–29 % slower than TP=2**. Cause: a one-line omission in the kernel's
  GEMM dispatch — the unsplit MoE launch never passed the live-row bound, so the surplus tail of
  `expert_ids`, which `expert_map` turns into a real local expert on one rank, ran a full GEMM over
  38 % of the grid at M=2048. Under expert parallelism every rank waits for that one.
- Per-rank MoE stage at M=2048: **18,401 → 10,107 µs (−45 %)**. End to end: C1 40.8 → 49.4,
  C4 59.5 → 99.6, C8 91.9 → 139.1, prefill 1,025 → 1,257. Reported upstream; fixed there in
  `a95e809`.
- Three of our own hypotheses refuted model-free with numbers (block padding, three-way collectives,
  the masking pass), and the GB10 top-k fallback we were pinned to turned out to be **faster** than
  the path it replaces below ~64K tokens of context.
- `gpu-memory-utilization` 0.85 → 0.80: cost nothing measurable and put all three nodes above the
  4 GiB free-memory rule for the first time.

## 2026-09-05 — TP=3 + expert parallel, first working boot

- Three nodes, 96 whole experts of 288 per rank, 64 → 66 head padding (22 per rank), vocabulary
  `padding_size` 192, shared expert padded to 2,112.
- Weights per node **81.53 → 54.86 GiB (−33 %)**; KV pool **825,000 → 2,947,441 (+257 %)**.
- Gates 10/10 · 12/12 cold **and after the full benchmark**; acceptance and accepted-tokens-per-step
  identical to TP=2 to three significant figures.
- Root causes fixed before the first boot by reading rather than running: the sidecar mount geometry
  (relative symlinks require identity mounts), and three divergent environment files.
- Root causes fixed during: the vision tower ignoring `--language-model-only` (fatal at TP=3, and it
  had been quietly carrying 1.05 GiB per rank at TP=2); a drafter head check that could not tell a
  legitimate pad from a disaster; an evidence log line being silently discarded because its logger
  sat outside the configured hierarchy; and KV pool arithmetic made visible.

## 2026-09-04 — DFlash2 ported to this stack (at TP=2)

- Speculative decoding brought into an image that had never run it, as a three-way git merge rather
  than a hand-copy. C1 aggregate **14.4 (no draft) → 30.5 (MTP k=3) → 42.9 (DFlash2 k=7)**;
  accepted tokens per step 3.31 → **5.37**.
- Two things the upstream delta did not cover: the target-side EAGLE3 interface, absent from this
  image entirely; and KV cache grouping, where the draft's sliding-window layers knock this model off
  its own grouping path. Fixed by giving the draft its own independent cache group — which is also
  what later capped the KV pool.
- Fail-closed additions: a quantized drafter is refused outright, head counts must divide the TP
  size, and the image is not produced if a ported symbol fails to resolve.

## 2026-09-04 — first EXL3 boot on this hardware

- `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at revision `b20c49ba`, two nodes, TP=2, no expert
  parallelism. Boot 471 s after a first attempt died in the sparse-attention indexer's persistent
  top-k; the GB10 top-k overlay cleared it.
- Gates 10/10 · 12/12; MMLU sample 86.4 ±0.7.

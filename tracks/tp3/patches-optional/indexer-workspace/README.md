# `patches-optional/indexer-workspace` — the K-gather workspace bound

**Measured, and not in production as of this commit.** One patch that sizes the sparse indexer's
K-gather workspace from its real bound instead of `40 × max_model_len`, freeing **4.42 GiB per node**
and growing the KV pool by **+10.25 %** at an unchanged memory fraction, with **no measured speed
cost**. It is here rather than in [`../../patches/`](../../patches/) because promoting it costs a
fresh fast-load sidecar and the disk for one has to be found first — see
[Why it is not in the recipe](#why-it-is-not-in-the-recipe).

**Applies to: TP=3 as measured; the patch itself is rank-agnostic.** The buffer is sized from
`max_model_len` alone, so a two-node stack reserves the same 4.92 GiB — where the pool is far
scarcer, and therefore worth more `[not tested]`.

Everything measured is in
[`results/memory/indexer-workspace-ab.md`](../../../../results/memory/indexer-workspace-ab.md); the
ledger context is [docs/17](../../../../docs/17-memory-ledger.md) §2.5, the standing item is
[docs/11](../../../../docs/11-open-issues.md) §2.28, and the upstream half of it is
[HELP-WANTED](../../../../HELP-WANTED.md) §9.

---

## What is here

| File | What it does |
|---|---|
| [`patch-indexer-workspace-tp3.py`](patch-indexer-workspace-tp3.py) | Five anchors across two files: the sizing helper and its use in `get_max_prefill_buffer_size` (`vllm/v1/attention/backends/mla/indexer.py`), an arm flag and two run-time guards (`.../indexer.py` and `vllm/model_executor/layers/sparse_attn_indexer_kpool.py`). Default **off**: with neither knob set the patched image is upstream, one environment read per process |

**What it changes, in one line.** `get_max_prefill_buffer_size()` returns `max_model_len * 40`
entries — a constant chosen upstream against DeepSeek-V3.2's 163,840-token context, where it is
825 MB. At `max_model_len` 1,000,000 it reserves 40,000,000 entries × 132 B = **4.92 GiB**, locked
for the life of the engine and charged to the residual the profiler subtracts before it sizes the KV
pool. The buffer only ever holds **one indexer chunk's compressed context**, whose exact ceiling here
is `max_num_seqs × ceil((max_model_len + num_spec + 1) / index_kpool)` = 2,000,016 entries =
**251.8 MB**. The patch takes it to **512 MB**, 2.03× that ceiling. The derivation, the four safety
layers and the reason a too-small buffer cannot corrupt silently are in the script's own docstring.

## The knobs

| Variable | Values | Effect |
|---|---|---|
| `HAREM_INDEXER_WS_MODE` | `bound` | `entries = min(upstream, max(2 × ceiling, 512 MB, one-request floor))` — 512.0 MB at our settings |
| | unset / `off` / `upstream` | **upstream sizing, byte for byte**; the two run-time guards stay disarmed |
| | anything else | refuses to boot rather than guessing |
| `HAREM_INDEXER_WS_MB` | a positive integer | explicit size in MB, still clamped to upstream's value and still floor-checked. Booted only in the CPU unit test `[not tested]` on hardware |

The chosen size, both bounds and the headroom multiple are printed once per rank at startup, so an
arm is auditable from its own log:

```text
HAREM-IDXWS bound | upstream=40000000 entries (5035.4 MB) -> chosen=4067203 entries (512.0 MB),
saved 4.42 GiB | max_model_len=1000000 compress_ratio=4 entry_bytes=132 (index_head_dim=128)
max_num_seqs=8 num_spec=7 | per_request_floor=250002 (31.5 MB)
scheduler_ceiling=2000016 (251.8 MB) headroom=2.03x | safety=2x floor=512 MB
```

Confirm the effect independently with `VLLM_DEBUG_WORKSPACE=1`, which is an upstream variable and
needs no patch: the `WorkspaceManager` prints every resize with the caller's file and line, and there
should be exactly **one**, reading `0.00 MB -> 513.00 MB` instead of `-> 5036.40 MB`.

## Installing it

Copy the file into your `TP3_DIR` and add one line to the prelude, beside the other TP=3 patches:

```bash
run python3 "$TP3_DIR/patch-indexer-workspace-tp3.py" --root "$VLLM_PY"
```

Then arm it in `EXTRA_ENV`, derived on each node from that node's own environment file with `sed` —
never copied between nodes ([envs/README.md](../../../../envs/README.md)):

```bash
HAREM_INDEXER_WS_MODE=bound
```

**One of the two target files is bind-mounted read-only.** The launcher mounts
`model_executor/layers/sparse_attn_indexer_kpool.py` from `$OVERLAY_DIR` as `:ro`, so the prelude
cannot write it. Pre-apply the patch to a **host-side copy** of the overlay directory and point
`OVERLAY_DIR` at the copy; the prelude then reports "already patched" for that file and writes only
`indexer.py` inside the container. Verify the copy with `diff -r` against the overlay it came from:
exactly one file should differ, by exactly this patch's two hunks. This is the same pattern
[`patch-kpool-init.py`](../../patches/patch-kpool-init.py) already needs, and it applies to **any**
future patch that lands on an overlaid file
([`../../patches/README.md`](../../patches/README.md)).

## The sidecar consequence: read this before you copy anything

`patches/harem_fastload_id.py` hashes **every file matching `patch-*.py`** in the tree, plus the full
text of the prelude, into the fast-load sidecar's identity
([docs/08](../../../../docs/08-fast-boot.md) §4). So copying this file in **invalidates the sidecar
before it runs**, and `preflight-fastload.py` refuses the next `FASTLOAD_MODE=load` boot on every
node. There are two ways round it and we took the second:

| | How | Cost |
|---|---|---|
| **Dump** | install into the production tree and take one `FASTLOAD_MODE=dump` boot | ~11 minutes once, and **~53 GB per node** of disk. The production sidecar is invalid until it completes |
| **A copy of the tree** | install into a separate tree, point `TP3_DIR` and `OVERLAY_DIR` at it in a separate environment file, and leave `FASTLOAD_MODE` empty — fast-load off | boot goes from a fast-load ~251 s to **325 and 352 s** on the two arms we ran. The production tree, overlay and sidecar are never touched |

The A/B in `results/` used the second, because two of our three nodes did not have 53 GB free. **The
memory fraction is not part of the identity**, so a rung change costs no dump boot — only files do.

## Why it is not in the recipe

Not doubt: **disk, and an owner's decision**. Promotion means installing into the production tree,
which forces a fresh sidecar (~53 GB per node) that does not fit until an older one is deleted. The
measurement itself is unambiguous — pool +10.25 %, gates 10/10 and 12/12 cold and warm, tool-call
8/8, needle-lite 6/6, one ~1M-token request and eight concurrent long-context lanes all correct,
every speed level inside its band, and **none of the four safety layers fired**. Full tables:
[`results/memory/indexer-workspace-ab.md`](../../../../results/memory/indexer-workspace-ab.md).

Two things a promoted arm still owes: the KV pool from a **fast-load** boot — both A/B arms were
eager, and the production figure is an `[estimate]` until one is taken — and a re-read of the gates on
that boot.

## What the script guarantees

The same contract as the production patch tree ([`../../patches/README.md`](../../patches/README.md)):

- `--root <vllm package dir>`; `--check` verifies anchors and never writes.
- Every anchor must occur **exactly once**. Anything else refuses loudly and returns non-zero. A
  half-patched stack is the failure mode that produces fluent, wrong answers.
- Idempotent — a second run reports "already patched" and returns 0.
- Both files are validated before either is written.
- Exit codes: `0` applied / already applied / check ok, non-zero for a missing file or an anchor
  mismatch — which the prelude's `run` turns into `exit 21`.

**And one guarantee that is specific to this patch:** with the knob unset it is inert by construction,
not by convention. That is measured, not asserted — the A/B's control arm ran on this exact patched
tree with the knob off and reproduced upstream's 5,036.40 MB workspace on all three ranks.

## Rollback

- **One line:** delete `HAREM_INDEXER_WS_MODE=bound` from `EXTRA_ENV`. The knob is read at run time,
  so the patched tree takes upstream's sizing.
- **The tree:** point `ENV_FILE` at the environment file of the configuration you came from. If the
  patch was installed by copying the tree rather than into it, nothing in production ever changed.
- The **image never changes**: the patch runs inside the container at boot, against the writable
  layer, and leaves no trace when the container is removed.

## What is not here

**The CPU-only unit test.** Seven blocks — the knob's off states, the bound at our settings, unknown
`index_kpool` falling back to the larger buffer, the explicit override, the startup refusal, the
chunker guard, and the one that mattered: **the chunk list is byte-identical to upstream's across six
geometries**, because the buffer size never bound the splitter at 4.92 GiB and does not bind it at
512 MB either. It is written against our anchors and our image and would need adapting to yours; what
it established is reported in
[`results/memory/indexer-workspace-ab.md`](../../../../results/memory/indexer-workspace-ab.md) §2.

**The long-context stress probes.** The single ~1M-token request and the eight-lane concurrent
long-context run are variants of the needle harness described in
[docs/09](../../../../docs/09-measurement-protocol.md); the only changes that matter are a
`timeout` of 3600 s (a 1M prefill takes 572 s, and the stock 900 s is not always enough) and a
distinct needle string per lane, without which a cross-lane gather mix-up reads as a plausible answer
instead of a wrong one.

# `mla-prefill` — falsifying the MLA-prefill residence-window claim on your own GB10

**Applies to: the EXL3 track, one node, engine down or idle with headroom. No fabric needed.**

Two model-free microbenchmarks of the sparse-MLA prefill kernel (`mla_decode`), timing three
selection patterns — a cache-resident `drifting` one, the calibrated `production` pattern, and a
gather-bound `independent` one — against each other and against a bandwidth ruler measured in the
same process.

```
bench_mla_prefill_5fd7299.py   the cuda-exl3 author's own fixture, unmodified, at his commit 5fd7299
mla_prefill_falsify.py         ours: his arms and generator verbatim, plus a second ruler, a
                                correctness gate, two head counts, and (added the same day) a
                                correctness matrix and a CPU-only self-test
```

Used in [`../../results/kernels/mla-prefill-falsification-gb10.md`](../../results/kernels/mla-prefill-falsification-gb10.md)
to check the `cuda-exl3` author's own correction of his MLA-prefill ceiling
([issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5), commit `5fd7299`) on a 48-SM / 24-MiB-L2
part, after his prediction was itself prompted by the indexer datum in
[`../../results/kernels/sm12-stack-patches-ab.md`](../../results/kernels/sm12-stack-patches-ab.md) §8.

---

## Licence and attribution

`bench_mla_prefill_5fd7299.py` is a verbatim copy of `bench/bench_mla_prefill.py` from
[`Zeuss5/cuda-exl3`](https://github.com/Zeuss5/cuda-exl3) at commit `5fd7299`, kept here unmodified
for provenance — it is what a claim about "his fixture" means, not a paraphrase of it. `cuda-exl3` is
**MIT**-licensed, with an attribution paragraph in its `LICENSE` file crediting
`turboderp-org/exllamav3` for unrelated derived files; see [CREDITS.md](../../CREDITS.md) for the full
citation this repository uses for that project.

`mla_prefill_falsify.py` is ours, written for this recipe, **Apache-2.0** like the rest of `bench/`
— except that its selection generator (`selections()`) and the three arms it drives are copied
**verbatim** from the file above, under the same MIT terms, so that running our script is running the
same fixture rather than a reimplementation of it. Both files carry this in their own docstrings.

## What each script does

**`bench_mla_prefill_5fd7299.py`** — run as shipped, `python3 bench_mla_prefill_5fd7299.py [head_dim]`
(needs `--ctx` for the `ctx_sweep` function used by the falsification above; see its own docstring).
Not modified in any way, including its comments; this is here so the fixture behind the numbers on
the results page can be inspected without trusting a description of it.

**`mla_prefill_falsify.py`** — the one this repository's numbers came from:

```
python3 mla_prefill_falsify.py                 timing bench: three contexts x two head counts x
                                                three arms, medians of 3 rounds, CUDA events. No
                                                arguments == exactly the run the results page reports;
                                                nothing added later changes this path.
python3 mla_prefill_falsify.py --verify        the same timing run, plus an inline correctness check
                                                inside each H=22 timing cell, so a run verifies what
                                                it timed
python3 mla_prefill_falsify.py --check         correctness matrix only, no timing, at the real TP=3
                                                shape: both head counts, both selection patterns, a
                                                kpool-aligned variant (--align 4) and a ragged-seqlens
                                                variant (--ragged) that exercises the masking path the
                                                timing run never touches
python3 mla_prefill_falsify.py --self-test     CPU only, no CUDA, no kernel call: proves the reference
                                                path, the masking semantics, the comparator and the
                                                argument parsing before anyone spends GPU time on it
```

`--self-test` needs no lock and no GPU and is safe to run any time, on any machine with `torch`
installed. `--check` and the default timing mode need `cuda_exl3` importable, which in practice means
running inside the production image (`--entrypoint /usr/bin/python3`, the image's own Python).

## Running it beside a live stack

This is a measurement, and the lock and engine-state rules in
[docs/09](../../docs/09-measurement-protocol.md) §10 apply: write the shared lock file before the
first `docker run` that touches the GPU, one measurement at a time, and prefer the engine fully down
over merely idle when you have the choice — the run this repository reports happened inside an
engine-free window taken for other work, not a boot of its own. In outline, from a node with the
production image already pulled:

```bash
docker run --rm --gpus all --ipc=host \
    --cpuset-cpus <a range disjoint from the engine's> --memory=4g \
    -e OUT_JSON=/out/result.json \
    -v "$(pwd)/mla_prefill_falsify.py:/bench.py:ro" \
    -v /path/to/an/output/dir:/out \
    --entrypoint /usr/bin/python3 <production image tag> /bench.py
```

The `--entrypoint /usr/bin/python3` is required: the production image's default entrypoint starts the
engine, not a shell, so a bare `docker run ... <image> /bench.py` will not do what it looks like it
does. `OUT_JSON` (default `/out/result.json`) is where the timing run writes its JSON; `--check`'s
matrix instead honours `CHECK_JSON` (or `--out-json`) and, in the absence of either, only prints.

**Cost.** The timing run is 17 s wall and about 2 GiB of GPU headroom on a card with the fixture's
default settings; peak measured here was 1.65 GiB. The `--check` matrix runs several such cells in
sequence and costs proportionally more; `--self-test` is CPU-only and near-instant.

## What is not here

The orchestration wrapper that takes the shared lock, checks the engine's `/metrics` for idle,
checks `MemAvailable` against a floor, `scp`s the script to a node and collects its output back is
**not shipped**: on this stack that script names real node aliases and an internal metrics URL
throughout, which is exactly the content [STYLE-GUIDE.md](../../STYLE-GUIDE.md) keeps out of this
repository. The docker invocation above and the lock discipline in
[docs/09](../../docs/09-measurement-protocol.md) §10 are the whole of what it does that is
transferable.

## Status of the correctness matrix

The default timing run's own correctness gate (inside `mla_prefill_falsify.py`, run automatically
before any timing) checks the kernel at a small 2-head smoke shape. The `--check` matrix that verifies
the **real 22-head TP=3 shape** was added the same day as the timing results and has passed its
CPU-only `--self-test` (24/24 checks: the chunked reference matches an unchunked one bit-for-bit, the
masking semantics are correct both ways, the comparator catches a deliberately wrong answer, and every
selection generator reproduces its expected overlap). **The GPU half — the kernel call itself at 22
heads — has not been run yet** at the time of writing; it needs the same engine-free window discipline
as the timing run. See [HELP-WANTED](../../HELP-WANTED.md) §8 if you can run it first.

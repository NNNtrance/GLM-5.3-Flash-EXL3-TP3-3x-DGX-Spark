# `nccl-latency-bench.py` — an all-reduce sweep with `nccl-tests`' contract, without `nccl-tests`

**Applies to: both tracks.** It needs at least two nodes and a cable; with one node there is no peer,
NCCL will not build a ring, and there is nothing to measure.

```
nccl-latency-bench.py    the payload: one process per node, one all-reduce sweep
run-nccl-latency.sh      one node, one container -- run it on every node
drive-nccl-latency.sh    local helper: ssh's the above onto every node, rank 0 in the foreground
```

Results measured with it: [`results/mesh/nccl-latency-sweep.md`](../results/mesh/nccl-latency-sweep.md).

---

## It is not `nccl-tests`, and the numbers must not be labelled as if it were

**`nccl-tests` could not be run on these nodes.** It is not packaged and neither is MPI — nothing in
apt or pip on any of the three — and `nccl-tests` **requires MPI for multi-node**: without it,
`nProcs` is fixed at 1 and the binary measures one GPU talking to itself. Building a new
MPI-plus-Docker chain (three nodes of `apt install`, ssh inside containers, `orted` interacting with
`docker run`, none of it ever tested here) was more risk and more time than writing the payload.

This repository's other fabric tools — [`mesh_sweep.py`](mesh_sweep.py),
[`mesh-multilink-sweep.sh`](mesh-multilink-sweep.sh), [`ar_bench.py`](ar_bench.py) — are not
`nccl-tests` either. This one was written on top of the same proven launch plumbing as
[`run-mesh.sh`](run-mesh.sh) and deliberately made to satisfy `nccl-tests`' own contract instead:

| `nccl-tests` | here | why it matters |
|---|---|---|
| `-b 8 -e 64M -f 2` | `LAT_SIZES` defaults to 8 B → 64 MiB doubling | Byte-precise geometric sweep. `mesh_sweep.py` steps in *tokens*, so it cannot start at 8 bytes at all, and the flat latency floor only becomes visible below a kilobyte |
| `-w 10 -n 50` | `LAT_WARMUP=10`, `LAT_ITERS=50`, fixed for every size | `mesh_sweep.py` adapts iteration count to size, which makes small-size rows incomparable with large ones |
| `busBW = algBW × 2(n−1)/n` | same formula, `n` = world size | The number everyone quotes. Getting the factor wrong is the commonest way to publish a wrong fabric figure |
| `-c 1` correctness check | each rank fills its buffer with `rank + 1`; the sum is compared exactly | Small integers are exact in bf16, so this is a real check and not a tolerance. **132/132 passed** on our run |
| CUDA-event timing | same | |

**So: same methodology, same formulas, same sweep convention, same production plugin and environment
— but the numbers do not come from the `nccl-tests` binary, and no table produced by this tool should
be captioned as if they did.**

## The isolation this tool is built for

It is designed to run **beside a live serving engine** rather than instead of one, so a fabric
measurement does not cost a boot:

- `--cpuset-cpus 10-14` and a `--memory 6g` cap by default, deliberately outside the cores the engine
  is pinned to. Change `CPUSET` if your engine sits elsewhere.
- `MASTER_ADDR` and `GLOO_IFACE` are set in the runner rather than sourced from the engine's
  environment file, so the tool has **zero coupling** to whatever that file says at the moment it
  runs — the engine may be mid-reconfiguration.
- Containers are `--rm` and mount the plugin read-only.

**That is not a licence to skip the discipline.** The GPU and the fabric are a lock: take it, confirm
the engine is actually idle (`num_requests_running` and `num_requests_waiting` both zero), and watch
free memory on every node while the sweep runs ([docs/09](../docs/09-measurement-protocol.md) §10).
Our own run waited 720 s for another arm's lock to clear before it started.

## Running it

On each node the tool lives beside the others in `~/exl3-zeus/bench/`. From a workstation with ssh to
all three:

```bash
LAT_TAG=prod-rep1 LAT_HOSTS="head worker-1 worker-2" bash bench/drive-nccl-latency.sh 3
```

Or by hand, one shell per node, rank 0 last so the others are already listening:

```bash
LAT_TAG=prod-rep1 bash bench/run-nccl-latency.sh 1 3
```

Extra NCCL settings go through `LAT_ENV` as space-separated `K=V` pairs, which is how the channel and
protocol arms were taken:

```bash
LAT_TAG=nochannels LAT_ENV="NCCL_ALGO=Ring" bash bench/drive-nccl-latency.sh 3
```

```bash
LAT_TAG=protoSimple LAT_ENV="NCCL_ALGO=Ring NCCL_MAX_NCHANNELS=8 NCCL_PROTO=Simple" \
  LAT_SIZES=8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576 \
  bash bench/drive-nccl-latency.sh 3
```

(The protocol arms stop at 1 MiB on purpose: `Simple` is 4-7x slower through the band that decides
the question, and running it to 64 MiB only spends fabric time re-confirming that the plateau is the
plateau.)

Rank 0 writes one JSON per run (`LAT_OUT`, default under the shared cache directory) carrying every
row plus the NCCL configuration it actually saw — `NCCL_ALGO`, `NCCL_PROTO`, both channel bounds and
the four `NCCL_MESH_*` knobs. **Read the configuration block back out of the JSON rather than
trusting the command line**: an environment variable that did not reach the container is the most
common way one of these sweeps lies.

## Environment

| | |
|---|---|
| `LAT_TAG` | label for the run and its output file |
| `LAT_SIZES` | comma-separated byte sizes; default 8 B → 64 MiB doubling |
| `LAT_WARMUP` / `LAT_ITERS` | default 10 / 50, the `nccl-tests` defaults |
| `LAT_ENV` | extra `K=V` pairs passed straight into the container |
| `LAT_OUT` | output JSON path |
| `LAT_HOSTS` | ssh targets in rank order, rank 0 first (`drive-` only) |
| `MASTER_ADDR` / `GLOO_IFACE` | rendezvous address and NIC name; set them for your cluster |
| `PLUGIN_SUB` / `IMAGE` / `CPUSET` / `MEMLIMIT` | plugin directory relative to `$HOME`, container image, cpuset, memory cap |

## Reading the output

Three warnings, all of them earned:

1. **Take a median of at least three repetitions at small sizes.** Our 8 KiB repetitions were
   72.4 / 101.2 / 74.7 µs — one of three was 35 % high. A single run at the small end is not a
   measurement.
2. **A single point that is 2× its neighbours may still be real.** Our 128 KiB point reproduced in
   all three repetitions (159.9 / 173.3 / 172.5 µs); it is the remains of a diagnosed cliff, not
   noise ([docs/06](../docs/06-nccl-mesh.md) §3).
3. **Compare busBW against your own wire, not against ours and not against a datasheet.** The
   cable's rated speed is not the ceiling on this part — the card's PCIe Gen5 x4 slot is, at about
   15 GB/s per NIC ([docs/06](../docs/06-nccl-mesh.md) §9). Publishing a percentage against a rated
   number is how this repository produced one of its retractions
   ([docs/11](../docs/11-open-issues.md) §1.7).

And one open question about the tool itself: **this harness and `bench/ar_bench.py` disagree by about
40 % at 64 KB and about 2× at 8 KB on the same operation.** Both readings are published, neither is
corrected against the other, and a same-session comparison of the two is
[HELP-WANTED](../HELP-WANTED.md) §5. Say which harness produced any small-size number you report.

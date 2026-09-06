# envs — one environment file per node, derived, never copied

**The templates themselves live under [`tracks/`](../tracks/README.md), one folder per node count,
because they are the files that differ between the two arrangements. This page is the rule that
governs all of them**, and every variable in each template carries a one-line reason, with the ones
that have a real cost carrying the measurement that decided them.

**Three templates, and the third is a different node count.**
[`env.tp3.example`](../tracks/tp3/env.tp3.example) is the routed-experts-only three-node file;
[`env.tp3-full.example`](../tracks/tp3/env.tp3-full.example) is production configuration 9/10;
[`env.tp2-full.example`](../tracks/tp2/env.tp2-full.example) is the **two-node production candidate**
([docs/15](../docs/15-tp2-track.md) §5) — `NNODES=2`, `TP_SIZE=2`, `ENABLE_EP=0`, no padding
sidecar, `gpu-memory-utilization` **0.85** rather than the three-node 0.88, and four settings that are not
optional at two ranks. The rule below applies to all three.

## The rule

**Never copy a finished `.env.tp3` from one node to another.** Two lines differ per node
(`NODE_RANK` and `HOST_IP`), and a copied file that keeps the wrong rank produces a cluster where two
processes believe they are the same rank. That failure does not announce itself: the rendezvous
completes, the engine starts, and the output is wrong.

Derive each node's copy from the template instead, on that node, with `sed`.

## Deriving them

On the head node (rank 0):

```
sed -e 's/@NODE_RANK@/0/' -e 's/@HOST_IP@/192.0.2.10/' -e 's/@HEAD_IP@/192.0.2.10/' -e 's/@IFACE@/eth0/' -e "s|@HOME@|$HOME|g" tracks/tp3/env.tp3.example > ~/exl3-zeus/.env.tp3
```

On worker-1 (rank 1):

```
sed -e 's/@NODE_RANK@/1/' -e 's/@HOST_IP@/192.0.2.11/' -e 's/@HEAD_IP@/192.0.2.10/' -e 's/@IFACE@/eth0/' -e "s|@HOME@|$HOME|g" tracks/tp3/env.tp3.example > ~/exl3-zeus/.env.tp3
```

On worker-2 (rank 2):

```
sed -e 's/@NODE_RANK@/2/' -e 's/@HOST_IP@/192.0.2.12/' -e 's/@HEAD_IP@/192.0.2.10/' -e 's/@IFACE@/eth0/' -e "s|@HOME@|$HOME|g" tracks/tp3/env.tp3.example > ~/exl3-zeus/.env.tp3
```

Replace the addresses with your own management addresses and `eth0` with your own management
interface. `@HEAD_IP@` is the same value on all three nodes; `@HOST_IP@` is each node's own.

Then confirm that only the two lines differ:

```
diff <(ssh worker-1 'cat ~/exl3-zeus/.env.tp3') <(ssh worker-2 'cat ~/exl3-zeus/.env.tp3')
```

Expected: two lines, `NODE_RANK` and `HOST_IP`. Anything else and one of them is stale.

## Before you change a value

Every value in the template that has a measured cost says what that cost is. Two in particular are
not free:

- `MAX_NUM_BATCHED_TOKENS` — raising it to 4096 buys prefill and mixed-load latency and costs 28.5 %
  of the KV pool ([docs/07](../docs/07-kv-and-draft-page.md)).
- `GPU_MEMORY_UTILIZATION` — the ladder above 0.80 is real but it runs into the free-memory rule
  before it runs into anything else ([docs/00](../docs/00-hardware-and-os.md) §6).
- `IMAGE` — changing it invalidates the fast-load sidecar, because the manifest records the image
  tag. The preflight refuses the boot rather than serving something else, and regenerating the
  sidecar costs one ~11-minute dump boot on every node ([docs/08](../docs/08-fast-boot.md) §8).
- `NCCL_MESH_PLUGIN_DIR` — points at a *patched* plugin build in the template. Pointing it back at a
  stock build is the rollback for everything in [docs/06](../docs/06-nccl-mesh.md) §6–§8; the
  behaviour is then what `NCCL_MESH_LINKS_PER_PEER=1` gives, which is the pre-patch behaviour exactly.

Keep a backup before every change, on every node, and name it after what it was:

```
cp ~/exl3-zeus/.env.tp3 ~/exl3-zeus/.env.tp3.bak-$(date +%Y%m%d-%H%M)
```

## Checking what the launcher will actually run

`scripts/start-tp3.sh` supports a dry run. It prints the full `docker run` line, fully expanded, and
starts nothing:

```
DRY_RUN=1 ~/exl3-zeus/start-tp3.sh 0
```

Read that line before the first boot on a new node. Two traps this catches, both of which cost us a
boot `[measured-here]`:

- **`EXTRA_ENV=A=1 B=2` unquoted is two assignments, not one variable.** Bash parses a bare
  `A=1 B=2` line as two separate assignments, so only the first pair reaches `EXTRA_ENV` and the
  second silently becomes a shell variable of the launcher, never reaching the container. It must be
  quoted, and it is quoted in the template.
- **A sidecar mounted away from its link target.** The sidecars are relative symlink trees; mounting
  one anywhere but its own host path makes every weight link dangle, and it surfaces as
  "no safetensors found" rather than as a mount error. The launcher refuses that arrangement by name,
  but the dry run shows you the mounts before you find out.

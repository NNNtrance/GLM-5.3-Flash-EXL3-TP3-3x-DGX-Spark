# systemd — a template, and why nothing here is installed

> **Nothing in this directory is installed on any node, and the file in it is not ready to be.**
> [`harem-exl3.service.template`](harem-exl3.service.template) is the draft as it was written in the
> field, kept verbatim so the three things wrong with it are visible rather than quietly fixed. Read
> this whole page before you copy it anywhere.

This stack starts by hand. The container runs with `--restart no` and nothing supervises it, so an
unattended reboot leaves the cluster down. That is a deliberate choice about the *restart* policy — a
half-started rank retrying on its own is precisely the "fluent and wrong" failure this stack is built
to refuse — and an accepted gap about *autostart*. The NVFP4 sibling recipe solved the same problem
properly and its `systemd/` directory is worth reading first:
[`NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark`](https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark).

---

## The hazard, before the template

If you run both recipes on the same nodes, as we do, **the sibling stack's unit will win a reboot**.
`harem-motor.service` (the NVFP4 engine) is `enabled` on all three of our nodes, so a reboot today
does not leave the cluster down — it brings up the *other* engine, on the same GPUs and the same
unified memory `[measured-here]`. Two engines can never be enabled at once; they will ask for the same
memory and one of them will lose in a way that looks like a mystery.

So whichever unit you install:

```
Conflicts=harem-motor.service
```

and disable the other one in the same change. Installing this unit without `systemctl disable` on the
sibling is not a partial fix, it is a worse state than having no unit at all.

---

## Three things are unfinished in the template

**1. The preflight script it calls does not exist.** `ExecStartPre` names
`~/exl3-zeus/motor-onkosul-exl3.sh` and no such file has been written. It is not optional: on this
hardware the engine must not start before the fabric is up, and the checks are the ones documented in
[docs/00](../docs/00-hardware-and-os.md) and [docs/06](../docs/06-nccl-mesh.md) —

- `docker` responding;
- `ibv_devinfo` showing **4/4** ports `PORT_ACTIVE` on this node (and, because `PORT_ACTIVE` does not
  mean "carrying traffic", a `port_xmit_data` read on both cables of each pair — [docs/06](../docs/06-nccl-mesh.md) §6);
- a ping to each fabric peer on both subnets;
- `drop_caches`, because the loader is sensitive to page-cache pressure ([docs/08](../docs/08-fast-boot.md) §5).

The NVFP4 recipe ships a working equivalent (`scripts/engine-preflight.sh`); adapting it is the
shortest path, and it is the reason autostart works at all over there.

**2. systemd will not honour the start order on its own.** The ranks must come up **worker-2, then
worker-1, then head** ([`scripts/start-tp3.sh`](../scripts/start-tp3.sh)). systemd starts three
independent machines with no ordering between them. Rank 0 needs either a wait step — poll the peers'
port until both answer — or a rank-dependent delay in `ExecStartPre`. Until one of those is written,
the unit is a coin toss on a cold cluster.

**3. `ExecStop` stops the wrong container.** The template names `harem_glm53_lil`, which is the
*NVFP4* container. This stack's container is `exl3-tp3`. Left as it is, `systemctl stop` would report
success and stop nothing. It is kept in the file exactly as it was written because the error is the
kind that survives review by looking plausible.

---

## What the template does get right

- `Type=oneshot` with `RemainAfterExit=yes`, because `start-tp3.sh` launches a detached container and
  exits; without `RemainAfterExit` systemd would treat that exit as the service having stopped.
- `Requires=docker.service` and `After=network-online.target`.
- `TimeoutStartSec` well above the boot: production boots in 274 s
  ([docs/08](../docs/08-fast-boot.md)), and 900 s leaves room for a boot that has to regenerate its
  fast-load sidecar. If your sidecar might be stale, that is a **682 s dump boot** and 900 s is not
  enough — raise it to 1800 ([docs/09](../docs/09-measurement-protocol.md) §11).
- `@USER@` / `@HOME@` placeholders rather than a baked-in path. Substitute per node; do not copy a
  substituted file between nodes, for the same reason the environment files are never copied
  ([docs/03](../docs/03-tp3-padding-and-sidecars.md)).

---

## The cheaper thing to do first

A unit solves reboots. It does not solve the failure we actually hit, which was the engine exiting
while nobody was looking — one outage during this work ran an hour purely because the only thing being
watched was a benchmark log `[measured-here]`.

A 60-second loop that checks `docker ps` and `/health`, and writes `docker logs --tail 40` when the
container is gone, is a few lines, needs no root, is independent of any unit, and would have caught
it. It is not written here `[not tested]`.

Tracked as an open item in [docs/11](../docs/11-open-issues.md) §2.20.

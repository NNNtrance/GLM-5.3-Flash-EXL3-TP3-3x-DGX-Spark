# systemd — the autostart unit, and the reboot it was tested against

> **This directory used to hold a template with three things wrong with it, and nothing installed
> anywhere.** All three are fixed, the unit is installed and `enabled` on all three of our nodes, and
> it has been through a **simultaneous reboot of the whole cluster with a live health check at the
> other end** `[measured-here]`. The unit is [`harem-exl3.service`](harem-exl3.service); the
> preflight it calls is [`motor-onkosul-exl3.sh`](motor-onkosul-exl3.sh) and is now a real script
> rather than a name. Read this page before you install either.

The container still runs with `--restart no` and nothing retries it. That has not changed and is not
an oversight: a half-started rank quietly retrying on its own is exactly the "fluent and wrong"
failure class this stack is built to refuse. **A unit that starts the engine at boot and a policy that
never restarts it are two different decisions**, and only the first of them was the gap.

---

## The hazard, before anything else

If you run both recipes on the same nodes, as we do, **the sibling stack's unit will win a reboot**
unless you disable it. `harem-motor.service` (the NVFP4 engine) was `enabled` on all three of our
nodes, so a reboot did not leave the cluster down — it brought up the *other* engine, on the same GPUs
and the same unified memory `[measured-here]`. Two engines can never be enabled at once; they will ask
for the same memory and one of them will lose in a way that looks like a mystery.

The unit carries the guard:

```
Conflicts=harem-motor.service
```

`Conflicts=` stops them running together. It does **not** stop the other one being enabled at boot, so
disable it in the same change. On our nodes `harem-motor.service` is now `disabled` and
`harem-exl3.service` is `enabled` `[measured-here]`.

---

## Install

> **At TP=2 there is a unit of its own — you do not edit this one.**
> [`harem-exl3-tp2.service`](harem-exl3-tp2.service) with
> [`motor-onkosul-exl3-tp2.sh`](motor-onkosul-exl3-tp2.sh). Four differences and no more:
> `WorkingDirectory`/`ExecStart` point at the two-node tree and `start-tp2full.sh`; `ExecStop` names
> `exl3-tp2`; `Conflicts=` lists **both** sibling units, because the three-node engine is the second
> engine on those nodes; and `FABRIC_PEERS` becomes **one** address per node rather than two (the
> ConnectX-7 check stays `4/4` — it counts ports, not peers).
>
> **Installed, started, health-checked and stopped on both nodes on 6 September 2026**: `systemctl
> start` returned in 3 s / 6 s, `/health` 200 at **+261 s**, KV pool 2,153,571, correctness 10/10 and
> code exam 12/12, `systemctl stop` clean `[measured-here]`. Its first attempt **failed correctly** —
> the preflight refused in one second because the fast-load sidecar the environment file named was
> not on disk, before docker was touched, which is check 7 doing its job.
>
> It is left **`disabled`**: exactly one of the two units may be enabled. And the reboot rule becomes
> "reboot **both** together, never one" — we have **not** run a two-node reboot test `[not tested]`.
> [docs/15](../docs/15-tp2-track.md) §2.6 and §5.7.

On **every** node, with `scripts/`, `patches/tp3full/` and the env file already in place per
[docs/03](../docs/03-tp3-padding-and-sidecars.md) and the README quick start:

```
sudo systemctl disable --now harem-motor.service
```

```
install -m 0755 systemd/motor-onkosul-exl3.sh "$HOME/exl3-zeus/motor-onkosul-exl3.sh"
```

```
sed "s#@USER@#$USER#g; s#@HOME@#$HOME#g" systemd/harem-exl3.service | sudo tee /etc/systemd/system/harem-exl3.service >/dev/null
```

```
sudo systemctl daemon-reload && sudo systemctl enable harem-exl3
```

Edit the `FABRIC_PEERS` case in the preflight for your own addresses first — the ones in the file are
the example range from [docs/00](../docs/00-hardware-and-os.md) §4.5 and are not ours. Substitute
`@USER@`/`@HOME@` per node; **do not copy a substituted file between nodes**, for the same reason the
environment files are never copied ([docs/03](../docs/03-tp3-padding-and-sidecars.md)).

Stopping the cluster is three commands, one per node:

```
sudo systemctl stop harem-exl3
```

`ExecStop` names `exl3-tp3`, which is this stack's container. The rank each node takes comes from
`NODE_RANK` in its own `.env.tp3`, not from an argument, so the same unit text works on all three.

---

## The three things that were wrong with the template, and what each became

**1. The preflight script did not exist.** It does now:
[`motor-onkosul-exl3.sh`](motor-onkosul-exl3.sh), seven checks, at most ten minutes of waiting.
Four are the fabric ones the NVFP4 sibling also runs — `docker` answering, `ibv_devinfo` showing
**4/4** `PORT_ACTIVE`, a ping to each fabric neighbour, then `sync` and `drop_caches` because the
loader is sensitive to page-cache pressure ([docs/08](../docs/08-fast-boot.md) §5). Three are this
stack's own, and each of them has failed us as a silent boot: the env file exists, the image named in
it is present locally, and **this rank's** fast-load sidecar has its `MANIFEST.json`. A missing
sidecar is the difference between a 251 s boot and a 620 s one, or the refusal in
[docs/14](../docs/14-troubleshooting.md) §3.1.

What it deliberately does not claim: it pings **one address per neighbour**, which is one link of each
pair. `PORT_ACTIVE` on the other two is link state and not proof that anything crosses them — that is
`port_xmit_data` read as a delta after the first benchmark, and it is the check that found half our
fabric idle ([docs/06](../docs/06-nccl-mesh.md) §6). A preflight cannot run it, because at preflight
time nothing has sent a packet yet.

**2. systemd will not honour the worker-2 → worker-1 → head start order.** It still will not, and
this is the honest half of the page. The three units start independently with no ordering between
them. What made the reboot test pass is not ordering, it is that the workers' rendezvous retries
until rank 0 appears and `TimeoutStartSec=1200` is roughly **five times** a normal 251 s boot. That is
tolerance, not a guarantee, and it has been demonstrated **once** `[measured-here]`. If you want the
guarantee rather than the margin, put a rank-dependent delay or a peer-port poll in `ExecStartPre`; we
have not `[not tested]`.

**3. `ExecStop` named the wrong container.** The template said `harem_glm53_lil`, which is the NVFP4
container, so `systemctl stop` would have reported success and stopped nothing. It now names
`exl3-tp3`.

Two smaller changes came with them. `TimeoutStartSec` went **900 → 1200**: 900 s covers a fast-load
boot with room, but not a **620 s dump boot** that also has to wait out the preflight and the settle
gate, and a unit that times out mid-load leaves a container running that systemd believes is gone.
And `WorkingDirectory` and `ExecStart` both point at `tp3full/`, which is the production tree since
configuration 9 — there is exactly one launcher and the second copy is what made the profiler answer
404 for a week ([docs/14](../docs/14-troubleshooting.md) §8.7).

---

## The reboot test

One trial, the whole cluster, from power-on to a served token `[measured-here]`, 5 September 2026,
production configuration 10:

| | |
|---|---|
| `reboot` issued to all three nodes | T = 22:23:06 |
| ssh answering again | T + **29 / 30 / 31 s** |
| `ibv_devinfo` 4/4 on all three | at that point, all three |
| `harem-exl3.service` at that point | `active` on two, still `activating` on the head |
| `systemd` logs the unit finished | 22:24:44 / 22:24:44 / 22:24:49 = T + **98 / 98 / 103 s** |
| **`/health` returns 200** | 22:28:21 — see the note below |
| KV pool on that boot | **5,652,892** tokens |
| Quality gates after it | correctness **10/10**, code exam **12/12** |

Raw log: [`../results/boot/boot-ledger.md`](../results/boot/boot-ledger.md).

**The elapsed figure is quoted two ways because the log disagrees with itself, and picking one
silently is exactly what this repository does not do.** The harness printed `health 200 +242s`; the
wall-clock stamps in the same file give **22:23:06 → 22:28:21 = 315 s**. We cannot reconstruct which
instant the harness started its counter from — 242 s before the health check lands at 22:24:19, which
is not the reboot, the ssh return or any unit event in the log — so the two numbers are both printed
and **315 s is the one to plan with**. What is not in doubt is the shape: the preflight, the fabric
coming up and the settle gate together account for **98–103 s**, and the container takes the
remaining **212 s** from the last unit finishing to a served token, against the 251 s a `docker run`
boot takes on a warm host ([docs/08](../docs/08-fast-boot.md)). The boot is the weight load and always
was; autostart adds a minute and a half of fabric wait to it.

The pool is the other line worth reading twice. It came back at **5,652,892 against the 5,619,834 of
the arm the configuration was promoted on**, +0.6 %, which is the settle gate doing its job: a reboot
is the cleanest possible baseline, and it agrees with a settled restart to well inside the 6 % this
number used to swing by ([docs/07](../docs/07-kv-and-draft-page.md) §1.1).

**Reboot all three nodes together, or none.** This is a fabric rule and it has nothing to do with the
unit: rebooting one node takes down the far end of its links and the pair does not heal
([docs/00](../docs/00-hardware-and-os.md) §3). The unit makes an all-three reboot survivable; it does
not make a one-node reboot safe.

---

## The cheaper thing that is still not done

A unit solves reboots. It does not solve the failure we actually hit, which was the engine exiting
while nobody was looking — one outage during this work ran an hour purely because the only thing being
watched was a benchmark log `[measured-here]`.

A 60-second loop that checks `docker ps` and `/health`, and writes `docker logs --tail 40` when the
container is gone, is a few lines, needs no root, is independent of any unit, and would have caught
it. It is still not written `[not tested]`. Tracked in
[docs/11](../docs/11-open-issues.md) §2.20.

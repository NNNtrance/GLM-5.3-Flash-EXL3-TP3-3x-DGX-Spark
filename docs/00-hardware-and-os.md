# 00 — Hardware and OS

What you need underneath everything else: three DGX Spark nodes, a direct-attached fabric, the exact
software versions we ran, and two system-level decisions (no desktop, swap left alone).

Node names in this repository are always `head` (rank 0, serves the API), `worker-1` and `worker-2`.
Addresses are documentation addresses: `192.0.2.10` (head), `192.0.2.11` (worker-1), `192.0.2.12`
(worker-2), and `192.0.2.100` for the workstation you drive them from.

Our NVFP4 sibling recipe,
[`GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark`](https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark),
documents this layer at length — cabling diagram, the SBIOS requirement, the fabric hotplug root
cause, the "reboot all three" rule, the preflight script. **That material is not duplicated here.**
This page gives the versions this stack was measured on and the handful of things that differ.

---

## 1. Hardware

| Item | What we have |
|---|---|
| Nodes | 3 × NVIDIA DGX Spark (GB10, `sm_121`, aarch64), 48 SMs per GPU |
| Memory | 128 GB unified per node; the OS reports **121 GiB**, the engine sees 121.63 GiB on the device |
| CPU | 20 cores per node |
| Storage | ~916 GB NVMe per node. This recipe needs ~250 GB free on each: 176 GB checkpoint + 56 GB fast-load sidecar + images |
| Interconnect | 2 ConnectX-7 QSFP cages per node, **no switch** — three direct cables in a ring, one `/24` per link |

Unified memory is the fact that shapes every decision downstream. The KV pool and the host's page
cache come out of the same 121 GiB, which is why reading the checkpoint through the page cache costs
KV pool ([docs/08](08-fast-boot.md)) and why `gpu-memory-utilization` has a hard practical ceiling
([docs/07](07-kv-and-draft-page.md)).

---

## 2. Software versions we ran

Read off all three nodes on 5 September 2026 `[measured-here]`. All three are identical except where
the table says otherwise.

| Component | Version | Command that prints it |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (noble), aarch64 | `cat /etc/os-release` |
| DGX OS | `dgx-release` 7.5.0 | `dpkg -l \| grep dgx-release` |
| Running kernel | `6.17.0-1029-nvidia` on `head` and `worker-1`, `6.17.0-1031-nvidia` on `worker-2` | `uname -r` |
| NVIDIA driver | 580.173.02 (`nvidia-driver-580-open`, apt `hold`) | `nvidia-smi` |
| CUDA toolkit (host) | 13.0 | `dpkg -l \| grep cuda-toolkit` |
| CUDA toolkit (image) | `nvcc` release 13.0, V13.0.88 | `docker run --rm --entrypoint nvcc <image> --version` |
| NCCL | 2.30.7+cuda13.3 — inside the container image, not on the host | container log line `vLLM is using nccl==…` |
| Docker | 29.2.1, build a5c7197 (buildx v0.31.1) | `docker --version` |
| ConnectX-7 firmware | 28.45.4028, **4 ports `PORT_ACTIVE` per node** | `ibv_devinfo` |
| Default systemd target | `multi-user.target` (no desktop) | `systemctl get-default` |
| Swap | 16 GB, `vm.swappiness` **60** | `swapon --show` and `cat /proc/sys/vm/swappiness` |

Run these on every node, not on one and assume. Our own `worker-2` runs a different kernel from the
other two and we never recorded why — a standing reminder that "they are identical" is an assumption
until it is printed.

```
uname -r
```

```
nvidia-smi
```

```
docker --version
```

```
ibv_devinfo
```

**Bring all three nodes up to date and reboot them together before you start.** Older kernels and
drivers are untested here. The sibling recipe's `docs/00` covers the update procedure, including the
USB-C power-delivery firmware that a reboot does not apply.

---

## 3. The fabric check that must pass before every boot

Four RoCE devices, all `PORT_ACTIVE`, on every node:

```
ibv_devinfo | grep -c "PORT_ACTIVE"
```

Expected: `4` on each node. Anything less and the rendezvous will either hang or silently fall back,
and you will spend the next hour reading engine logs for a cabling problem.

Two rules we learned the hard way and carry over from the NVFP4 stack `[measured-here]`:

- **Reboot all three nodes, or none.** Rebooting one node kills the far end of its links; the other
  two come back with dead ports.
- **`MASTER_ADDR` must be the management address of rank 0, never a fabric address.** A fabric
  address produces a silent hang in the rendezvous rather than an error. `scripts/start-tp3.sh`
  refuses one outright.

---

## 4. Desktop off

All three nodes run `multi-user.target`. No display manager, no compositor, no browser.

```
sudo systemctl set-default multi-user.target
```

Reboot for it to take effect. The gain was never measured on our units — they came off the
installation media already in this state, so we have no before-and-after `[not tested]`. A graphical
session on a unified-memory machine costs both host RAM and GPU-visible memory, and on this stack
every GiB of host RAM is a GiB the KV pool cannot have, so the recommendation stands on the
architecture rather than on a measurement of ours.

---

## 5. Swap: leave it alone

16 GB of swap, `vm.swappiness` at the distribution default of **60**. Do not set it to 0.

We set `vm.swappiness=0` once on this hardware, on the NVFP4 stack, and locked all three machines
`[measured-here, raw lost]`. Under a unified-memory allocation this large, denying the kernel the
ability to page anything at all does not keep the working set resident — it removes the pressure
valve. Leave it at 60 and control memory with `gpu-memory-utilization` instead.

Swap use at rest on the production configuration is 0.09–0.12 GiB per node, and it does not grow
during serving `[measured-here]`. If yours grows across a benchmark, your `gpu-memory-utilization` is
too high; see [docs/07](07-kv-and-draft-page.md).

---

## 6. The memory rule

**Never let free host RAM fall below 4 GiB on any node.** This is the rule the whole memory ladder is
built around, and it is why this recipe runs at `gpu-memory-utilization 0.80` rather than higher.

Measured at rest on the production configuration `[measured-here]`:

| | head | worker-1 | worker-2 |
|---|---|---|---|
| free | 10.9 GiB | 12.2 GiB | 12.1 GiB |
| swap used | 0.12 GiB | 0.10 GiB | 0.09 GiB |

We climbed the ladder to 0.85 and measured a KV pool of 5,256,198 tokens — and free RAM on the head
node at **1.9 GiB** with 1.6 GB of swap in use `[measured-here]`. That breaks the rule, so 0.85 was
rejected and 0.88 was never attempted. The full ladder is in [docs/07](07-kv-and-draft-page.md).

---

## 7. Disk budget

Per node:

| Item | Size |
|---|---|
| EXL3 checkpoint (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, 120 shards) | 175.6 GB |
| DFlash2 draft | 2.3 GB |
| Model and drafter sidecars ([docs/03](03-tp3-padding-and-sidecars.md)) | negligible — symlink trees plus rewritten config files |
| Fast-load sidecar ([docs/08](08-fast-boot.md)), optional | **56 GiB** |
| Container images | ~31 GB per tag; keep two or three |

We had 370–374 GB free on `/var/tmp` per node after all of the above `[measured-here]`.

---

## 8. What is next

[01 — Model and license](01-model-and-license.md). Read it before you download anything: the
checkpoint's licence is not one you have seen before, and the draft model's is non-commercial.

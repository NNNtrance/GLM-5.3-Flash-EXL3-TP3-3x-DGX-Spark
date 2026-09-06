# 00 — Hardware, firmware and OS: the complete environment record

**Applies to: both tracks.** The fabric sections describe a three-cable ring; two nodes are one pair
of it.

Everything below the container. Three DGX Spark nodes, their firmware, the direct-attached fabric
and how it is actually wired, the exact software versions this stack was measured on, every
system-level change we made, and — just as important — the system-level changes we deliberately did
**not** make, with the incident that taught us why.

This page is deliberately long. It exists so that a reader who gets stuck at the operating-system
layer finds the answer here instead of guessing. If you are looking for a specific failure and its
fix, [14 — Troubleshooting](14-troubleshooting.md) indexes them by symptom.

Node names in this repository are always `head` (rank 0, serves the API), `worker-1` and `worker-2`.
Management addresses are documentation addresses: `192.0.2.10` (head), `192.0.2.11` (worker-1),
`192.0.2.12` (worker-2), and `192.0.2.100` for the workstation you drive them from. Our real
hostnames, LAN addresses and fabric subnets are not published; substitute your own everywhere.

Our NVFP4 sibling recipe,
[`GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark`](https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark),
covers the same layer for the other quantization path. This page no longer defers to it: everything
you need is here.

---

## 1. Hardware

| Item | What we have |
|---|---|
| Nodes | 3 × NVIDIA DGX Spark (GB10, `sm_121`, aarch64). Ours are ASUS Ascent GX10 units |
| GPU | `NVIDIA GB10`, **48 SMs**, unified with host memory. Note that `nvidia-smi --query-gpu=memory.total` returns **`[N/A]`** on this part — the GPU has no memory of its own to report. Read `MemTotal` instead |
| Memory | 128 GB unified per node. The OS reports `MemTotal` **121.6297 GiB**; the engine reports a device total of **121.63 GiB** — the same pool, counted twice `[measured-here]` |
| CPU | 20 cores per node, one socket: **10 × Cortex-X925 + 10 × Cortex-A725**. The two clusters are why the production `CPUSET` takes five cores from each (`5-9,15-19`) rather than a contiguous ten |
| Storage | one 916 GB NVMe per node (`/dev/nvme0n1p2`); **`/` and `/var/tmp` are the same partition**, so the checkpoint competes with the OS for space. Local read 5.8–5.9 GB/s, write 4.0–4.7 GB/s `[measured-here]` |
| Interconnect | 2 QSFP cages per node, **no switch** — **three direct cables in a ring**, carrying **six** point-to-point links, two per node pair. See §4 |

**Unified memory is the fact that shapes every decision downstream.** The KV pool, the model
weights, the container, the host page cache and the OS all come out of the same 121.63 GiB. That is
why reading the checkpoint through the page cache costs KV pool ([08](08-fast-boot.md)), why
`gpu-memory-utilization` has a hard practical ceiling ([07](07-kv-and-draft-page.md)), and why the
memory rule in §11 is the rule the whole stack is built around.

### 1.1 The two rulers, measured rather than quoted

Every percentage in [10 — Results and roofline](10-results-and-roofline.md) is against these two
numbers. **Do not use the datasheet figures.** `bench/bw.py` and `bench/gemmpeak.py` reproduce both.

| Ruler | Measured on our GB10 | Datasheet implies | Ratio |
|---|---|---|---|
| Device read bandwidth, bf16 `sum` over 4 GiB | **225.2 GB/s** | 273 GB/s | 82 % `[measured-here]` |
| BF16 dense GEMM peak, 8192³ `torch.matmul` | **97.3 TFLOP/s** | ~125 (1 PFLOP FP4 ÷ 8) | 78 % `[measured-here]` |

Three consecutive read-bandwidth runs gave 225.2, 239.6 and 240.9 GB/s — a 6.5 % spread — so we take
the lowest as the ruler and quote percentages as a band where it matters. A write-only kernel
measures 196.8–198.2 GB/s, which is the right comparison for a kernel that only writes.

**Taking a catalogue number for a ruler is the same class of mistake as taking "two links × port
rate" for a fabric ceiling.** We made both, and §4.3 is the second one.

### 1.2 The three units are not identical, and one of them is slower

Sustained GPU clocks under load, measured on all three after firmware was levelled `[measured-here]`:

| | head | worker-1 | worker-2 |
|---|---|---|---|
| Sustained clock | 2485 MHz | 2502 MHz | 2440 MHz |
| Memory bandwidth | 225–231 GB/s | 225–231 GB/s | 225–231 GB/s |

A 62 MHz spread, **2.5 %**, with memory bandwidth identical across all three. That is ordinary
die-to-die binning, not a fault, and there is nothing to do about it. **In a tensor-parallel cluster
the slowest node sets the pace for all three**, so plan around your slowest unit rather than your
fastest.

One of our units also idles 8–10 °C hotter than the other two. The cause is not the GPU: the ASUS
embedded controller drives the fan from **power draw rather than temperature**, so a genuinely idle
unit never spins its fan and slowly heats up. The proof is that the same unit measures *cooler* three
minutes after a CPU load than it does at rest. It is a documented behaviour on these units
(<https://forums.developer.nvidia.com/t/asus-ascent-gx10-idle-temperature-rises-after-hdmi-disconnect-identical-unit-stays-cool/380282>);
the community workaround is to hang a 5 W USB device off it so the controller sees a load. Under
sustained serving the fan runs anyway, so this matters at rest, not in production `[measured-here]`.

---

## 2. Firmware — do this before you cable anything

Firmware is the one layer where being out of date costs you performance **with no error message at
all**. On our units:

| Component | Version we run | Reads it |
|---|---|---|
| SBIOS | `GX10DGX.0105` (5 May 2026) | `fwupdmgr get-devices` |
| Embedded controller (EC) | `0x02000006` | `fwupdmgr get-devices` |
| SoC firmware | `0x03000007` | `fwupdmgr get-devices` |
| USB-C power delivery | `0x00000516` | `fwupdmgr get-devices` |
| ConnectX-7 | `28.45.4028` | `ibv_devinfo` |

### 2.1 SBIOS 0104 or newer is required

**On older SBIOS the second QSFP cage negotiates at half width (PCIe Gen5 x2) and you lose roughly a
quarter of the fabric bandwidth silently** — 163 Gb/s where the machine should give 220. Nothing
reports an error; you simply measure less and never learn why. Our units shipped on `0101` and were
taken to `0105` before any measurement in this repository was made
(<https://forums.developer.nvidia.com/t/367853>).

Update all three with the NVIDIA-signed LVFS packages — the same pipeline the DGX dashboard uses
behind the scenes, minus the account requirement:

```
sudo fwupdmgr update --assume-yes
```

Budget about 10 minutes of installation per node plus about 10 minutes of applying at the next
reboot.

### 2.2 Two field notes that cost us time `[measured-here]`

**Do not reboot while an update is still running.** Wait for this to print `0`:

```
ps -eo cmd | grep -c "[f]wupdmgr update"
```

**A reboot does not apply the USB-C PD firmware.** That controller lives inside the 240 W adapter and
a reboot never cuts its power. The failure looks like this:

```
failed to run update on reboot: expected 0x00000516 and got 0x00000001
```

The sequence that works: full shutdown → unplug **wall, device and USB-C** → hold the power button
30 s → plug back in and boot → **re-install the update** → reboot. A cold drain on its own is not
enough; the package has to be staged again afterwards. Units already on a real PD version needed only
an ordinary reboot; only the one still at the factory `0x0001` needed the drain.

### 2.3 The GPU clock cap: we turned ours off

Early in this project we ran a `gpu-clock-cap` service to work around a firmware-era instability. It
is **disabled on all three nodes** and has been for the life of this stack. The reasoning, recorded
so it is not re-litigated: the real repair was the EC firmware update in §2; the cap had not
prevented any of our crashes (all three were memory, not clocks); and in a cluster **the slowest node
sets the pace**, so leaving one node capped costs all three 16 %. Measured cost of the cap when it
was on: **15.9 % of compute, zero of memory bandwidth** `[measured-here]`.

The protection that actually matters is the memory rule in §11, not a clock cap.

---

## 3. The fabric hotplug fix — read this before your first reboot

This is the single most expensive system-level trap on a three-Spark cluster, and it has a one-line
fix.

### 3.1 The symptom

You reboot one node. It comes back. `ip -br link` says `UP`. Ping works. Then the engine dies:

```
NCCL error: unhandled system error
```

and `ibv_devinfo` shows **2 ports `PORT_ACTIVE` where there should be 4** — not on the node you
rebooted, on the *other* node. Often the downed interface has no address at all, because the
NetworkManager profile never activated.

**The dead pair moves every time.** Ours went: reboot all three → one pair dead; reboot that pair →
that pair healed and a *different* pair died; assign the address by hand → address returns, carrier
still 0; `ip link down/up` simultaneously at both ends → no effect; reboot all three → **4/4
everywhere** `[measured-here]`.

### 3.2 The cause — a package, not a theory

The `dgx-spark-mlnx-hotplug` package installs a udev rule, `90-mtk-hotplug.rules`, which runs
`mtk-hotplug-handler.sh`. When the file `/etc/nvidia/cx7-hotplug-enabled` **exists**, hotplug is
armed, and the handler **removes the ConnectX-7 from the PCI bus the moment the far end of a link
goes down**. That is why the peer of a rebooting node loses its port and does not get it back on its
own: the card is gone from PCI until something re-enumerates it.

Credit for the identification: `digchick/dgx-spark-200g-link-fix`.

### 3.3 The fix

Remove the file on **all three** nodes, keeping a backup:

```
sudo mv /etc/nvidia/cx7-hotplug-enabled /root/cx7-hotplug-enabled.backup
```

The `sysfs` value stays at 1; the udev handler is what reads the file, so removing it is sufficient.
The exam is the next reboot, not the command. Ours has held since 2 September 2026
`[measured-here]`.

Check the file is gone before you trust it:

```
ls -la /etc/nvidia/cx7-hotplug-enabled
```

Expected: `No such file or directory` on every node.

### 3.4 The rule that stands anyway: reboot all three, or none

Even with the hotplug fix in place, **we still reboot all three nodes together.** Two independent
reasons:

1. Both ends of a link have to come up together for the pair to negotiate cleanly. Waiting for SSH is
   not waiting for RoCE.
2. The hotplug fix removes the *mechanism* we identified. It does not prove there is no second one.

If you must bring one node back on its own, verify the fabric on the *other two* before you start the
engine, not just on the one you rebooted.

---

## 4. The fabric: how it is actually wired, and where its ceiling really is

### 4.1 Physical cabling — three cables, in a ring

Each Spark has **two QSFP cages**. NVIDIA names them by position: **Port0** is the cage next to the
ethernet jack, **Port1** is the far one. The three nodes are wired as a directed ring — each node's
**near** cage to the next node's **far** cage:

| Cable | From | To |
|---|---|---|
| 1 | `head` Port0 | `worker-1` Port1 |
| 2 | `worker-1` Port0 | `worker-2` Port1 |
| 3 | `worker-2` Port0 | `head` Port1 |

Three cables, six cage-ends, no cage left over. Source: NVIDIA's own `connect-three-sparks`
playbook (<https://github.com/NVIDIA/dgx-spark-playbooks>). **Verify the ring with LLDP before you
power anything on** — NVIDIA's own warning is that a mis-wired ring produces a network configuration
that looks half-alive and fails later, and we can confirm that reading it wrong costs an afternoon.

### 4.2 Logical links — six, two per node pair, on two separate PCI endpoints

Each cage presents **two logical ports**, so every node exposes **4 netdevs and 4 RoCE devices**, and
the three cables carry **six** point-to-point links — **two between every pair of nodes**. Each link
sits on its **own /24**: there is no single RoCE subnet on a three-node triangle, and trying to make
one is a known way to break it (§9.3).

| Netdev | PCI address | RoCE device |
|---|---|---|
| `enp1s0f0np0` | `0000:01:00.0` | `rocep1s0f0` |
| `enp1s0f1np1` | `0000:01:00.1` | `rocep1s0f1` |
| `enP2p1s0f0np0` | `0002:01:00.0` | `roceP2p1s0f0` |
| `enP2p1s0f1np1` | `0002:01:00.1` | `roceP2p1s0f1` |

**The two links of a node pair land on two different PCI endpoints** (domain `0000` and domain
`0002`). That is not a detail — it is why the second link is worth having at all, and §4.3 explains
why.

> **Terminology, because it matters when you count things in your rack.** Sections
> [06](06-nccl-mesh.md), [10](10-results-and-roofline.md), [11](11-open-issues.md) and the
> `results/mesh/` data say **"cable"** where the object is strictly a **link**. There are three
> physical cables and six links; a "second cable to each peer" is a *second link over the same
> physical cable, on the other PCI endpoint*. The measurements, the finding and the patches are
> unaffected — the idle-link finding in [06](06-nccl-mesh.md) §6 is exactly as measured — but if you
> go looking for a sixth cable behind your rack you will not find one. The physical count is three.

### 4.3 The ceiling is PCIe, not the wire — and we got this wrong first

Here is the arithmetic, corrected. Each RoCE port reports `active_width: 4X` and
`active_speed: 50.0 Gbps`, i.e. 200 Gb/s = 25 GB/s. Two ports per card looks like 50 GB/s per card.
**It is not**, because the card is attached to the machine through a slot:

```
sudo lspci -vv -s <ConnectX bus id> | grep LnkSta
```

```
LnkCap: Port #0, Speed 32GT/s, Width x4
LnkSta: Speed 32GT/s, Width x4
```

**`LnkCap` says x4 as well as `LnkSta`.** That matters: it is not a link that negotiated down and
could be argued back up, it is a four-lane slot. We read all four ConnectX functions on all three
nodes — **12 of 12 report `Speed 32GT/s, Width x4` in both fields**, with no degradation anywhere
`[measured-here]`.

| | Arithmetic | Result |
|---|---|---|
| Wire, 2 ports × 200 Gb/s | 2 × 25 GB/s | 50 GB/s per card |
| **PCIe Gen5 x4** | 32 GT/s × 4 lanes ÷ 8 × (128/130) | **≈ 15.75 GB/s raw, ~14.5–15 GB/s loaded** per card |
| Per node, both endpoints | 2 × ~15 | **≈ 30 GB/s total** |

**A card's two ports cannot push even a third of what the wire allows, because the slot is four
lanes.** That single fact explains three separate results in this repository `[measured-here]`:

- The 13.25 GB/s point-to-point ceiling we measured was never "half a cable". It is **87 % of one
  endpoint's PCIe limit** — the measurement was right from the start and the interpretation was
  wrong.
- Patch `0005` helped because the second link of a pair is on the **other endpoint**, so it opened a
  second PCIe path. The gain came from the second *card*, not the second *link*. Ceiling moved to
  ~20 GB/s.
- Patch `0007` (one-sided `RDMA_WRITE`) drove RNR and out-of-band counters to zero and moved
  bandwidth **not at all**, because RNR was never the binding constraint. The PCIe wall was.

The remaining fabric headroom is therefore at most ~30 % (today's ~20 GB/s against a ~30 GB/s
ceiling), and since the collective is 16.5 % of a prefill chunk, the whole class is worth **at most
2–4 % of prefill** to the engine. [06](06-nccl-mesh.md) §9 carries the detail.

### 4.4 Raw fabric performance, measured

`ib_write_bw`, RoCEv2, GID index 3 `[measured-here]`:

| Cable | Link 1 | Link 2 | Total |
|---|---|---|---|
| head ↔ worker-1 | 98.02 | 98.01 | **196.0 Gb/s** |
| worker-1 ↔ worker-2 | 98.01 | 98.02 | **196.0 Gb/s** |
| worker-2 ↔ head | 98.02 | 98.02 | **196.0 Gb/s** |

A single link running alone reaches 109.2 Gb/s. Aggregate fabric **588 Gb/s**, identical on all three
cables, 98 % of line rate. TCP over the same cables reaches 24.7 GB/s per cable with **8 streams**
(a single TCP stream manages only 42.8 Gb/s — that is the stream, not the cable).

**The disk, not the fabric, is the bottleneck when moving a checkpoint around.** Local NVMe reads
5.8–5.9 GB/s against a 24.7 GB/s network.

### 4.5 Addressing scheme

Six links, six /24s, one per link, MTU 9000, configured as persistent NetworkManager profiles that
auto-connect, with `ipv4.never-default yes` so your ordinary default route survives.

**Pick a range that does not collide with your LAN.** NVIDIA's playbook uses
`192.168.0.x`–`192.168.5.x`; on a typical home network `192.168.1.1` is the router, and following the
playbook literally will take your network down. We use a private range chosen to avoid this; it is
not published. Substitute your own — the shape is what matters:

| Link | `head` | `worker-1` | `worker-2` |
|---|---|---|---|
| fabric-0 | `enp1s0f0np0` .1 | `enp1s0f1np1` .2 | — |
| fabric-1 | `enP2p1s0f0np0` .1 | `enP2p1s0f1np1` .2 | — |
| fabric-2 | `enp1s0f1np1` .1 | — | `enp1s0f0np0` .2 |
| fabric-3 | `enP2p1s0f1np1` .1 | — | `enP2p1s0f0np0` .2 |
| fabric-4 | — | `enp1s0f0np0` .1 | `enp1s0f1np1` .2 |
| fabric-5 | — | `enP2p1s0f0np0` .1 | `enP2p1s0f1np1` .2 |

Using `172.31.0.0/24` … `172.31.5.0/24` as an example, fabric-0 would be `172.31.0.1` on `head` and
`172.31.0.2` on `worker-1`. Any private range you do not otherwise use will do.

**`MASTER_ADDR` must be the management address of rank 0, never a fabric address.** A fabric address
produces a silent hang in the rendezvous rather than an error. `scripts/start-tp3.sh` refuses one
outright.

### 4.6 The fabric checks that must pass before every boot

**Four RoCE devices, all `PORT_ACTIVE`, on every node:**

```
ibv_devinfo | grep -c "PORT_ACTIVE"
```

Expected: `4` on each node. Anything less and the rendezvous will either hang or silently fall back.

**`PORT_ACTIVE` is not the same as "carrying traffic", and the difference cost us a factor of two.**
All four ports read `ACTIVE` on all three nodes for the entire life of this cluster while two of them
per node had transmitted **zero bytes since driver load** — the mesh plugin was putting every channel
on the first link of each pair. Add this and read it as a **delta**, not as an absolute:

```
for p in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do echo "$p $(cat $p)"; done
```

Any port stuck at a value that never moves is a port nothing is using. See [06](06-nccl-mesh.md) §6.

**Prove the second link is physically there before you enable anything that uses it.** Matching /24s
in `ip -br -4 addr` plus four `PORT_ACTIVE` ports is configuration plus link state, not proof of
connectivity. Ping across each second link, from each end, bound to the interface:

```
ping -c2 -W2 -I <second-link interface> <peer address on that /24>
```

Six pings on a triangle. If one fails, take it to whoever cabled the rack.
`bench/mesh-multilink-sweep.sh` refuses to run when that check fails.

---

## 5. Software versions we ran

Read off all three nodes on 5 September 2026 `[measured-here]`. All three are identical except where
the table says otherwise.

| Component | Version | Command that prints it |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (noble), aarch64 | `cat /etc/os-release` |
| DGX OS | `dgx-release` 7.5.0 | `dpkg -l \| grep dgx-release` |
| Running kernel | `6.17.0-1029-nvidia` on `head` and `worker-1`, `6.17.0-1031-nvidia` on `worker-2` | `uname -r` |
| Installed kernels | `6.17.0-1014.14` (factory image) plus `6.17.0-1029.29` / `6.17.0-1031` | `dpkg -l \| grep linux-image` |
| NVIDIA driver | 580.173.02 (`nvidia-driver-580-open`, apt `hold`) | `nvidia-smi` |
| CUDA toolkit (host) | 13.0 (`cuda-toolkit-13-0` **13.0.3-1**) | `dpkg -l \| grep cuda-toolkit` |
| CUDA toolkit (image) | `nvcc` release 13.0, V13.0.88 | `docker run --rm --entrypoint nvcc <image> --version` |
| NCCL | 2.30.7+cuda13.3 — **inside the container image, not on the host** | container log line `vLLM is using nccl==…` |
| Docker | 29.2.1, build a5c7197 (`docker-ce` 5:29.2.1-1); buildx v0.31.1. **No `/etc/docker/daemon.json` on any node** | `docker --version` |
| ConnectX-7 firmware | 28.45.4028, **4 ports `PORT_ACTIVE` per node**, `link_layer: Ethernet` (RoCE, not native InfiniBand) | `ibv_devinfo` |
| SBIOS / EC / SoC / USB-C PD | `GX10DGX.0105` / `0x02000006` / `0x03000007` / `0x00000516` | `fwupdmgr get-devices` |
| NCCL mesh plugin | `autoscriptlabs/nccl-mesh-plugin` at `19924dcc` + our patches — see §12 | build from source |
| Default systemd target | `multi-user.target` (no desktop) | `systemctl get-default` |
| Swap | `/swap.img`, 16 GB, priority −2, `vm.swappiness` **60** | `swapon --show`, `cat /proc/sys/vm/swappiness` |
| RDMA userspace | `rdma-core` 50.0-2ubuntu0.2 | `dpkg -l \| grep rdma-core` |
| Secure Boot | **enabled** on all three; the driver is the signed `-open` build | `mokutil --sb-state` |
| Held packages | **18** — the NVIDIA driver stack and the kernel | `apt-mark showhold` |
| Unattended upgrades | **not configured** on any node — there is no file in `/etc/apt/apt.conf.d/` that enables them | `ls /etc/apt/apt.conf.d/` |

Run these on **every** node, not on one and assume. Our own `worker-2` runs a different kernel from
the other two and we never recorded why — a standing reminder that "they are identical" is an
assumption until it is printed.

```
cat /etc/os-release
```

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

```
dpkg -l | grep -E "dgx-release|linux-image-6|nvidia-driver-580|cuda-toolkit-13"
```

**The driver is held at its current version** (`apt-mark hold`). A driver that moves under a running
stack changes the amount of memory reserved before the engine ever starts, which moves the KV pool
without moving anything you configured. If you unhold it, re-measure §11 afterwards.

---

## 6. Update first, then reboot all three together

**This recipe was built and measured on the versions in §5. Older kernels and drivers are untested
here.** On each node:

```
sudo apt update
```

```
sudo apt full-upgrade -y
```

Then reboot **all three at the same time** (§3.4):

```
sudo reboot
```

After the reboot, before anything else:

```
ibv_devinfo | grep -c PORT_ACTIVE
```

That must print `4` on every node.

**An honest note about our own update history.** We updated these machines when we first set them up
(20 August 2026) and did not record the steps. The apt and dpkg logs have since rotated, so we cannot
reconstruct which commands ran in what order, or the before/after versions
`[measured-here, raw lost]`. What we can state is where the machines ended up, which is §5. From the
factory image (`6.17.0-1014` on our units) a plain `apt full-upgrade` is what we would do today.

---

## 7. Everything we changed at the OS level — the complete list

If you want the short version of this page, this is it. Six changes, and nothing else:

| # | Change | Why | Where |
|---|---|---|---|
| 1 | Firmware to SBIOS `0105` + EC + SoC + USB-C PD | Second QSFP cage runs at half width below `0104`, silently | §2 |
| 2 | `rm /etc/nvidia/cx7-hotplug-enabled` on all three | The hotplug handler pulls the CX7 off the PCI bus when a peer goes down | §3 |
| 3 | Six persistent NetworkManager fabric profiles, MTU 9000, `ipv4.never-default yes` | The six /24 links; keep the ordinary default route | §4.5 |
| 4 | `systemctl set-default multi-user.target` | Unified memory: a desktop session costs KV pool | §8 |
| 5 | `gpu-clock-cap` service disabled | Superseded by the EC firmware fix; a capped node caps the cluster | §2.3 |
| 6 | NCCL mesh plugin built and installed from source, patched | The triangle needs it; the default channel count and link selection cost 13 % and 4–6 % | §12 |
| 7 | PackageKit's apt hooks disabled, and the NVIDIA/kernel packages held | A background updater that moves the driver under a running stack moves the KV pool with it | §7.1 |

**No `sysctl` file. No kernel command-line change. No udev rule of our own. No
`/etc/docker/daemon.json` at all — the file does not exist on any of our nodes.** Everything the
container needs is passed on the `docker run` command line by `scripts/start-tp3.sh`, where a reader
of this repository can see it. The stack is configured through the launcher's environment file, not
through the operating system, and that is deliberate: it keeps the rollback to one file
([envs/](../envs/)).

### 7.1 Keeping the driver still

Two small things, and they exist for the same reason. **The NVIDIA driver reserves about 14.2 GiB
before the engine starts**, and that reservation is an input to the KV pool (§11). A driver that
moves underneath you moves the pool without you changing anything you configured, and you will spend
an afternoon looking for the cause in your own settings.

```
sudo apt-mark hold nvidia-driver-580-open
```

Ours holds 18 packages — the driver stack and the kernel. Check what yours holds:

```
apt-mark showhold
```

PackageKit's apt hooks are disabled the simple way, by renaming the file so `apt` no longer reads it
(`apt` only reads files with no extension or ending in `.conf`):

```
sudo mv /etc/apt/apt.conf.d/20packagekit /etc/apt/apt.conf.d/20packagekit.disabled
```

Nothing on these nodes is configured for unattended upgrades, and we would not recommend adding it.
Update deliberately (§6), all three together, and re-run the audit afterwards.

### 7.2 Two things on our head node that are **not** part of this recipe

Being honest about our own drift, because you will see it if you copy our unit list: our head node
also runs an **NFS server** (a leftover from an earlier way of moving checkpoints around — this
recipe has no shared filesystem, every rank loads from its own local disk) and a small
**crash-logging unit of ours**. Neither is required, neither is referenced anywhere in this
repository, and you should not install either.

---

## 8. Desktop off

All three nodes boot to `multi-user.target`. No display manager, no compositor, no browser. The GNOME
packages are still installed — the desktop is not removed, it simply never starts.

```
sudo systemctl set-default multi-user.target
```

```
sudo reboot
```

To put it back:

```
sudo systemctl set-default graphical.target
```

`systemctl get-default` tells you which one is armed. The switch is reversible in both directions and
uninstalls nothing.

**We cannot quote a measured gain.** Our units came off the installation media in this state and we
never ran them with a desktop up, so we have no before-and-after `free -g` reading `[not tested]`.
The sibling recipe estimates 2–3 GiB per node and marks it an estimate too. The recommendation stands
on the architecture rather than on a measurement of ours: on a unified-memory machine every GiB of
host RAM is a GiB the KV pool cannot have. If you want the real figure on your hardware, measure it —
`free -g`, switch, reboot, `free -g`.

**If you reach the nodes over SSH only, you do not need the desktop.** Every step in this recipe is a
shell command.

---

## 9. What we did **not** change, and why

The three most tempting system-level knobs on this hardware. We left all three alone, and one of
them we left alone the expensive way.

### 9.1 `vm.swappiness` — leave it at 60

16 GB of swap at default priority, `vm.swappiness` at the distribution default of **60**, no
persistent sysctl file on any node.

```
swapon --show
```

```
cat /proc/sys/vm/swappiness
```

**Do not set `vm.swappiness=0` on this stack.** On 2 September 2026 we changed it from 60 to 0 on
this hardware while chasing an unrelated allocation problem. The next production start — the tightest
configuration we run — **locked all three machines simultaneously**: ping answered, TCP port 22
accepted the connection, no SSH banner, no console. A physical power cycle of all three was the only
way out. The same configuration had passed 7/7 two hours earlier and `swappiness` was the only system
setting that changed in between `[measured-here, raw lost]` — the machines died before anything
reached disk, so we have no logs and **the causal link is not proven**. It is one incident with one
strong suspect, and it cost three power cycles.

The advice we had been following came from a two-node setup with a 3 GiB KV reservation, where it was
good advice. Under a unified-memory allocation this large, denying the kernel the ability to page
anything at all does not keep the working set resident — it removes the pressure valve.

**The rule this produced, and it is the most portable thing on this page: a kernel or system setting
is tried first on a low-KV test branch, never for the first time in production.** Production is the
tightest condition you own; it is the worst possible place to learn what a setting does.

Control memory with `gpu-memory-utilization` (§11) instead. Swap at rest on the production
configuration is 0.09–0.12 GiB per node and does not grow during serving `[measured-here]`. If yours
grows across a benchmark, your `gpu-memory-utilization` is too high.

### 9.2 Docker's daemon configuration — there isn't one

There is **no `/etc/docker/daemon.json` on any of our three nodes**; Docker runs on its compiled-in
defaults. Every container setting this stack needs — memory limits, IPC, the mounts, the cpuset — is
passed on the `docker run` command line by `scripts/start-tp3.sh`, where it is visible and
version-controlled. Nothing is hidden in a daemon config that a reader of this repository cannot see.

The other kernel knobs are at their distribution defaults too, and we checked rather than assumed
`[measured-here]`:

| Knob | Value on all three | Changed by us |
|---|---|---|
| `vm.swappiness` | 60 | no — see §9.1 |
| `vm.overcommit_memory` | 0 | no |
| `vm.min_free_kbytes` | 45155 | no |
| `net.core.rmem_max` / `wmem_max` | 212992 | no |

Those last two are worth a sentence because they look like something you should raise for a
200 Gb/s fabric. **They are not on the path that matters.** NCCL's traffic goes through the mesh
plugin over RDMA queue pairs, not through the kernel socket buffers; the TCP path is used for the
rendezvous and for file copies. We did not tune them and we did not need to.

### 9.3 `NCCL_IB_GID_INDEX` — do not pin it

This one is worth a paragraph because the advice you will find is correct for a different cluster and
**fatal on this one**. Two-node, single-subnet RoCE guides tell you to pin `NCCL_IB_GID_INDEX=3`.
A three-node triangle is **six /24s, not one subnet**, and pinning the GID index breaks it. Leave it
unset; NCCL 2.21.5+ selects it. Prior art reached the same conclusion independently
(`FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks`).

Using `-x 3` in an `ib_write_bw` command is a different thing and is correct — that is one manually
configured link, not NCCL's multi-node discovery.

---

## 10. Remote access

We reach the cluster over SSH on the management network, which is an ordinary shared /24 — one
management interface per node, quite separate from the six fabric links in §4.

A mesh VPN (Tailscale 1.102.3) is installed and enabled on all three nodes so we can reach them from
outside. **It is not part of this recipe.** It carries no cluster traffic, no rank talks to another
rank over it, nothing in this repository references it, and `MASTER_ADDR` is a management address and
not a VPN one. It is listed here only because you will see `tailscale0` in `ip -br link` on our nodes
and should know it is not doing anything for the model. If you run one, keep it off the fabric
interfaces; if you do not, nothing here changes.

Inter-node SSH for large file copies goes over a **fabric** address rather than the management
address, which is worth knowing when you copy a 176 GB checkpoint:

```
rsync -a --inplace -e "ssh -c aes128-gcm@openssh.com" <src> <peer-fabric-address>:<dst>
```

Remember §4.4: at 5.8–5.9 GB/s your NVMe is the limit, not the fabric.

---

## 11. Memory: the rule, the ladder, and the gate

### 11.1 The rule

**Never let free host RAM fall below 4 GiB on any node.** On a GB10 the GPU shares host memory, so
this figure *is* your safety margin, and it is why this recipe runs at
`gpu-memory-utilization 0.80` rather than higher.

Measured at rest on production configuration 9 `[measured-here]`:

| | head | worker-1 | worker-2 |
|---|---|---|---|
| Free | 12.1 GiB | 13.5 GiB | 13.4 GiB |
| Swap used | ~0.1 GiB | ~0.1 GiB | ~0.1 GiB |

### 11.2 Why not higher

We climbed to `0.85` and measured a KV pool of **5,256,198 tokens** — and free RAM on the head node
at **1.9 GiB with 1.6 GB of swap in use** `[measured-here]`. That breaks the rule, so 0.85 was
rejected and nothing above it was attempted on this stack.

**`0.83` has since been run and is production configuration 10** — production 9 with one line
changed `[measured-here]`:

| | production 9 @ 0.80 | production 10 @ 0.83 |
|---|---|---|
| KV pool | 5,168,044 | **5,619,834** (+8.7 %) |
| C1 / C4 / C8 tok/s | 69.8 / 134.6 / 192.4 | 70.5 / 144.6 / 194.0 — inside the bands |
| Quality gates, cold and warm | full | full |
| Swap under load | ~0.1 GB | ~0.1 GB, **flat through the rounds** |
| `MemAvailable` after the rounds | 12–13 GB | 8–10 GB |
| `MemFree` after the rounds | — | 0.9–1.2 GiB (reclaimable page cache) |

`MemFree` at 0.83 sits below the headline figure, and the reason that is acceptable is specific:
**0.85 was rejected for swap growth, not for `MemFree` in the abstract**, and at 0.83 swap does not
grow. `MemAvailable` is the honest headroom number here and 8–10 GB is well clear. **0.85 will not be
attempted on this stack.** The full ladder is in [07](07-kv-and-draft-page.md) and
[11](11-open-issues.md) §2.4.

**The device ceiling is not the binding limit.** The driver takes about 14.2 GiB permanently and does
not give it back on reboot, which puts the device-side ceiling near 0.88; **host memory binds first**
on this stack, well below that.

### 11.3 The settle gate — a host-side wait before `docker run`

This one is a measurement-integrity fix, and it is the part of this page most worth copying.

vLLM's KV pool is sized from a **difference between two readings of `/proc/meminfo`** taken minutes
apart. It runs backwards: **a node that starts with less memory free awards itself a larger pool.**
Because the launcher killed a ~90 GiB container and started the next one immediately, and the nodes
start in a fixed order, the last node started was systematically 9 GiB short — **27 % of a rank's KV
allowance sitting inside the measurement instrument** `[measured-here]`.

The fix is one host-side wait, before `docker run`, for `MemAvailable` to come back:

```
SETTLE_MIN_GIB=112
```

It took the per-rank spread from 9 GiB to 1.4 GiB. It buys **no tokens at all** — it makes the number
mean what it says. It also refuted this repository's then-largest open item, which had claimed 8.2
GiB per worker was stranded; acting on that would have over-committed the head node
([07](07-kv-and-draft-page.md) §1.1, [11](11-open-issues.md) §2.3).

**No published figure of ours turned out to be wrong**, because the pool takes the minimum over ranks
and the polluted node happened never to be the binding one. That is luck, not design.

---

## 12. The NCCL mesh plugin

A three-node triangle has no switch and no single RoCE subnet, so NCCL needs a transport plugin.
Build it on **every** node from source; we do not ship a binary.

| | |
|---|---|
| Upstream | `autoscriptlabs/nccl-mesh-plugin` at commit **`19924dcc`** (MIT) |
| Stock build | `libnccl-net.so` **401,368 bytes**; text 118,159 / data 2,208. A stock rebuild of `19924dcc` is identical in size, symbols and section sizes to the binary the NVFP4 stack has been running, which is how we know the source tree is the one that produced it `[measured-here]` |
| Our builds | `0004` only: **401,608 bytes**, md5 `856374e27f1daf56031a554ffecdd2ac`. `0004`+`0005`+`0006`: **418,864 bytes**, md5 `5cf62aaa6e66d9b00b1570f031534927` — **this is the production binary** `[measured-here]` |
| Verification | **No version or commit string is embedded in the binary.** `strings` gives you the `mesh_*` symbols and nothing else, so record the source commit and the md5 yourself; there is no way to ask the `.so` what it is |
| Unit tests | `make test-unit` → `test_routing` **13/13 pass** |
| Our patches | `patches/kernel/0004`, `0005`, `0006` — apply all three |
| Built but **not** adopted | `patches/kernel/0007` (one-sided FIFO `RDMA_WRITE`) — it works and it changes no bandwidth, because the ceiling is PCIe (§4.3) |

| Patch | What it does | Knob |
|---|---|---|
| `0004-min-rnr-timer` | The receiver-not-ready timer was fixed and long; makes it short and tunable | `NCCL_MESH_MIN_RNR_TIMER=1` |
| `0005-device-aware-link-selection` | `mesh_connect()` discarded NCCL's device index and stopped at the first subnet match, so **every channel to a peer landed on one link**; honours the device index and spreads over both | `NCCL_MESH_LINKS_PER_PEER=0` (`1` = pre-patch behaviour) |
| `0006-ptr-cuda-dmabuf-and-flush` | Advertises `NCCL_PTR_CUDA`, removing a host bounce buffer | `NCCL_MESH_PTR_CUDA=1`, `NCCL_MESH_FLUSH=1` |

With one link per pair, `NCCL_MESH_LINKS_PER_PEER=1` makes `0005` a no-op and the selection is
bit-identical to the pre-patch build; `0006` is still worth measuring on its own.

**One environment variable outside the plugin is not optional:**

```
NCCL_MAX_NCHANNELS=8
```

The default channel count costs **13 % of aggregate throughput at concurrency 8**. Sixteen channels
is **worse**, not better — 2.5× worse on the decode-sized message. [06](06-nccl-mesh.md) has the
whole story, including what is *not* the cause.

---

## 13. Disk budget

Per node:

| Item | Size |
|---|---|
| EXL3 checkpoint — `turboderp/GLM-5.3-Flash-exl3` 4.05bpw (production) | ~165 GB |
| EXL3 checkpoint — `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, 120 shards (fallback) | 175.6 GB |
| DFlash2 draft | 2.3 GB |
| Model and drafter sidecars ([03](03-tp3-padding-and-sidecars.md)) | negligible — symlink trees plus rewritten config files |
| Fast-load sidecar ([08](08-fast-boot.md)), optional but recommended | **53–56 GiB** per patch tree |
| Container images | ~31 GB per tag; keep two or three |

We had 370–374 GB free on `/var/tmp` per node with the whole stack in place `[measured-here]`.
**Plan for at least 300 GB free per node**, and note that **two patch trees means two fast-load
sidecars** — production 8 and production 9 cannot share one.

We keep both models under `/var/tmp` because that is where they landed on day one and the paths are
baked into our environment files. `/var/tmp` is not cleaned on these systems, but it is not a good
long-term home either; if you are starting fresh, choose a path you control and adjust the mounts.

---

## 13a. Autostart: the unit, the preflight, and the reboot it was measured against

**The engine starts at boot.** `harem-exl3.service` is installed and `enabled` on all three nodes,
and it calls a preflight that refuses to start the container until this layer — the one this whole
page is about — is actually ready. Both files are in [`systemd/`](../systemd/README.md); this section
is the part that belongs to the hardware and OS.

**Read §3.4 first, because the unit does not change it.** Reboot **all three nodes together, or
none**. A single-node reboot takes down the far end of that node's links and the pair does not heal
(§3.1–§3.3). Autostart makes an all-three reboot survivable; it does nothing for a one-node reboot
except start an engine into half a fabric.

**Why a preflight and not just `After=network-online.target`.** At boot the engine starts long before
the fabric is ready, and every one of the fabric's failure modes is quiet: `PORT_ACTIVE` arrives late,
the NCCL rendezvous hangs with no useful error, and the unit sits in `activating` until
`TimeoutStartSec` expires. The preflight waits — up to ten minutes — for `docker` to answer,
`ibv_devinfo` to read **4/4** (§4.6), and each fabric neighbour to answer a ping, then does `sync` and
`drop_caches` because the loader is sensitive to page-cache pressure ([08](08-fast-boot.md) §5). It
then checks three things that are not about hardware at all and have each cost us a silent boot: the
env file exists, the image it names is present locally, and this rank's fast-load sidecar has its
`MANIFEST.json`.

What the preflight **cannot** do: prove the fabric carries traffic. It pings one address per
neighbour, and `PORT_ACTIVE` on the other two ports is link state, not delivery — the check that
found half our fabric idle is `port_xmit_data` read as a **delta after** traffic has run
([06](06-nccl-mesh.md) §6), and at preflight time nothing has sent a packet yet. Run the checklist in
§14 after your first benchmark, not instead of it.

**Measured, once, on production configuration 10** `[measured-here]`: `reboot` to all three at
22:23:06; ssh and `ibv_devinfo` 4/4 on all three at **+29 / +30 / +31 s**; the units log `Finished` at
**+98 / +98 / +103 s**, which is the whole of the preflight, the fabric wait and the settle gate;
`/health` returns 200 at 22:28:21. The harness printed that last step as `+242 s` and the wall-clock
stamps in the same log make it **315 s**; both are recorded and the larger is the one to plan with
([`results/boot/boot-ledger.md`](../results/boot/boot-ledger.md)). The KV pool came back at
**5,652,892** against 5,619,834 from a settled `docker run` on the same configuration, **+0.6 %**, and
the quality gates read 10/10 and 12/12 afterwards.

**One OS-level dependency.** `drop_caches` needs root. Either run the unit as root or give the
service user a `NOPASSWD` sudoers line for `/usr/bin/tee /proc/sys/vm/drop_caches`; if it is not
permitted the preflight does not fail, it skips the drop and the weight load starts against a dirtier
page cache. That is the one thing on this page the unit will do quietly if you let it.

---

## 14. Preflight checklist

Before you build anything:

- [ ] Three DGX Spark nodes, same OS and driver, updated and rebooted **together**
- [ ] SBIOS `0104` or newer on all three — `fwupdmgr get-devices`
- [ ] USB-C PD firmware applied (§2.2 if it refuses)
- [ ] `/etc/nvidia/cx7-hotplug-enabled` **absent** on all three
- [ ] Ring cabling verified with LLDP, not by eye
- [ ] `ibv_devinfo | grep -c PORT_ACTIVE` prints `4` on all three
- [ ] `port_xmit_data` moves on **all four** ports per node once traffic runs, not two
- [ ] Six fabric /24s up, MTU 9000, `ipv4.never-default yes`, none colliding with your LAN
- [ ] `NCCL_IB_GID_INDEX` **unset**
- [ ] `systemctl get-default` prints `multi-user.target`
- [ ] `cat /proc/sys/vm/swappiness` prints `60` — do not change it
- [ ] NCCL mesh plugin built at `19924dcc` with patches `0004`, `0005`, `0006`; `make test-unit` passes
- [ ] At least 300 GB free per node
- [ ] `lspci -vv` reports `Speed 32GT/s, Width x4` for each ConnectX endpoint — so you know your ceiling
- [ ] If you install the autostart unit: the sibling NVFP4 unit `harem-motor.service` is **disabled**
      on all three, and the fabric addresses in the preflight are **yours** (§13a)

---

## 15. What is next

[01 — Model and license](01-model-and-license.md). Read it before you download anything: the
production checkpoint's licence and the draft model's are different from each other, and one of them
is not one you have seen before.

If something has already gone wrong, go to [14 — Troubleshooting](14-troubleshooting.md).

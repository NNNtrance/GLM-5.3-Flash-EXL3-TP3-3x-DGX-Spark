---
name: Did not reproduce
about: A step in this recipe did something different on your cluster
title: "[did-not-reproduce] "
labels: did-not-reproduce
---

Before you fill this in: `docs/14-troubleshooting.md` indexes every failure we hit by symptom, with
the exact log line, and `docs/14` §0 is a triage order. Most of what goes wrong here is already in
there. `audit/run-audit.sh` prints our own numbers next to each step and tells you *which* thing
differs from ours.

Everything below is asked for because we have been unable to help without it at least once.

## 1. How many nodes, and which track

- Node count:
- Track: TP=3 / TP=2 / something else (say what)
- If something else: what makes it different (rank count, cabling, a switch, mixed hardware)

`docs/00-start-here.md` says which track owns which pages.

## 2. Image and kernel revision

- Image tag:
- `cuda-exl3` commit the image was built from:
- vLLM revision, if you changed it:
- Mesh plugin: commit, and which of `patches/kernel/0004`, `0005`, `0006`, `0007` are applied:

"The current build" is not a revision. If you cannot name the commit, say so — that is itself an
answer, and `docs/02-image-build.md` says how to get it back out of the image.

## 3. Checkpoint

- Repository and branch:
- Revision (full sha, not `main`):
- Scope: full scope / routed experts only / other
- Drafter, and `k`: DFlash2 k=?, built-in MTP k=?, or none
- `sha256` verification: passed / failed / not run

## 4. Env file diff against the track's example

Not your whole env file. The diff against the example your track ships, with your addresses and
paths already replaced by placeholders:

```
diff -u envs/<the track's example> <your env file>
```

```
paste the diff here
```

If the diff is empty, say that: an empty diff and a different result is a much more interesting
report than a long diff.

## 5. The exact log line

The line itself, verbatim, in a fenced block, plus about twenty lines either side. Not a paraphrase
and not a screenshot — we search this repository by string.

```
paste the log here
```

**Scrub before you paste.** Host names, LAN addresses, user names, home paths and tokens do not
belong in a public issue. We grep our own files before every commit and the pattern is in
`CONTRIBUTING.md`.

If there is no log line at all — the failure was silent, the wrong answer was fluent, the request was
never scheduled — say so explicitly. `docs/14` §11 is the index of the twenty failures here that
produced no error message, and a silent one is worth more to us than a loud one.

## 6. Which `docs/14` entry it matches

- Entry number, or "none that I could find":
- If it matches an entry but the fix in it did not work, say what you ran and what happened.
- If it matches no entry, say which section you looked in.

## 7. What you have already checked

Tick what applies. Every one of these has been the answer at least once:

- [ ] `ibv_devinfo | grep -c PORT_ACTIVE` prints 4 on every node
- [ ] All nodes rebooted together, never one alone (`docs/00` §3.4)
- [ ] The quality gates run cold **and** warm (`scripts/correctness-probe.py`, `scripts/code-exam.py`)
- [ ] `audit/run-audit.sh` run, and its output compared against `audit/README.md`
- [ ] The boot gates in the log read as `patches/tp3full/README.md` says they should
- [ ] Env file derived per node with `sed`, never copied between nodes (`envs/README.md`)

## 8. Anything you would rather we did not publish

We quote issues in `docs/14` when they turn out to be a failure mode nobody had written down. Say
here if you would rather be credited differently, or not at all.

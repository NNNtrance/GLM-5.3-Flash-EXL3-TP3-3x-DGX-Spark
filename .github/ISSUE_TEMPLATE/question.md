---
name: Question
about: Something in this repository is unclear, or you want to know whether it applies to your setup
title: "[question] "
labels: question
---

Three places answer most questions faster than we can:

- `docs/00-start-here.md` — how many nodes do you have, and which pages apply to you.
- `docs/14-troubleshooting.md` — every failure we hit, indexed by symptom, with the exact log line.
- `docs/11-open-issues.md` — what is unresolved, what we retracted, and what we never ran. If the
  answer to your question is "we do not know", it is probably already written there in those words.

If none of them answers it, ask here.

## Your setup, in three lines

- Node count and track (TP=3 / TP=2 / neither):
- Are you running this recipe, planning to, or reading it for something else:
- If you are running it: image tag and `cuda-exl3` commit, and the checkpoint:

Say "not running it yet" if that is the case. A question from someone deciding whether to buy a
third node is a perfectly good question and needs none of the above.

## The question

Ask it here. Quote the sentence or the table cell you are asking about, and name the file, so we know
which page needs to be clearer.

## What we can and cannot answer

**Can:** anything about what is in this repository, why a number is what it is, what a setting cost,
what we would do differently, and what the honest confidence on any figure is.

**Cannot, without turning it into a measurement:** whether a change we never ran will help you. The
list of things we never ran is `docs/11` §3 and `HELP-WANTED.md`, and both are longer than the list
of things we did. If your question is one of those, say so — we will tell you what we would measure
and how, and if you run it, the measurement issue template is the place for the result.

**Will not:** guess. If we do not know, the answer will be "we do not know, and here is what would
settle it".

## One request

If the answer turns out to be somewhere in these documents and you could not find it, say where you
looked. A page that has the answer and cannot be searched for it is a defect in the page.

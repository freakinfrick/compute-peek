---
name: compute-peek
description: Give a long run — a compute sweep/batch/render/train, or an agent swarm from a 1-hour sprint to a multi-week campaign — a live window in tmux or herdr panes. A bespoke peek-spec declares which panes exist and which blocks each shows (progress, ETA, current unit, hardware or token spend, agent roster, message/escalation feed, task dependencies, failure grep, a themed one-liner); the run feeds it a progress JSONL; `launch` converges to that window idempotently. Use when starting or watching a multi-minute run or a swarm, when the user says "compute-peek", "peek", "vitals", "watch the run", "how long left", "what are the agents doing", or asks for run visibility.
allowed-tools: Bash, Read
---

# compute-peek

A window into a long run. Descriptive, not prescriptive: **the run emits a progress JSONL,
a peek-spec declares the window, the launcher builds it** in tmux or herdr. Different runs
get differently-shaped windows. `SPEC.md` next to this file is the contract.

```bash
P=<dir of this SKILL.md>/compute-peek.py          # e.g. ~/.claude/skills/compute-peek
python3 $P init   --manifest .peek/<run>.json [--kind swarm] [--like <spec>]
python3 $P launch --manifest M [--start] [--mux tmux|herdr]   # build panes (idempotent)
python3 $P status --manifest M                     # one-shot text: poll this, not the panes
python3 $P pane   --manifest M --name summary --once
python3 $P specs                                   # the project's peek-spec library
python3 $P gpus                                    # device UUIDs for the spec
python3 $P demo [--kind swarm --history 14d]       # showcase window on a synthetic run
python3 $P selftest
```

| Piece | Where | Contract |
|---|---|---|
| the feeder | run writes `--progress <jsonl>`; `peek_progress.py` is a stdlib emitter | SPEC §2 compute · §2b agents |
| the spec | `<project>/.peek/<run>.json` | SPEC §1 |
| the window | `panes: [{name, blocks, dir?, ratio?}]` | SPEC §3, §3b |
| the mux | `--mux` > `$PEEK_MUX` > spec `mux` > auto (in tmux → tmux, else running herdr, else running tmux) | SPEC §4 |

Blocks — compute: `title` `divider` `meter` `stage` `groups` `now` `last` `gpu` `log` `stale`
`thought` `footer` `group_table` `units` `charts` `raw`. Swarm: `roster` `feed` `cost` `deps`
(mix freely). Order by glance-value: **on fire / waiting on the human → how long left →
what's running → per-group → last numbers → hardware or bill → raw log.**

Workflow:
1. Make the driver emit the JSONL (`peek_progress.Progress`); print `k/N done, ETA x`
   unbuffered. Swarms: emit state *transitions*, cumulative `usage`, `plan` for new tasks.
2. `specs` — reuse a similar shape with `init --like`, else `init [--kind swarm]`; edit
   `command`, `progress`, `log`, `metrics`, `gpu` / `budget_usd`.
3. Fill `theme` with the run's own character (`meter_label`, `stages`, `thoughts`).
4. `launch --start`; it prints how to attach. Then poll `status`.

Guards:
- Never starts a tmux/herdr server (env leaks into every later pane); with none running it
  says how to start one.
- `launch` never starts the command twice: refuses if the raw pane is busy or the JSONL
  shows a heartbeat younger than `stale_after_s`. Campaign specs want hours there.
- ETA: emitter's mean-wall × remaining for one lane; throughput over `rate_window_s` when
  the emitter sends `eta_s: null` (concurrent agents). Split unlike units into groups.
- Devices are UUIDs, never indices. Keep sharded thread counts inside the host's budget.
- STALE / agent `?` = look, not a verdict. Default failure grep misses `ValueError:` — add
  patterns.

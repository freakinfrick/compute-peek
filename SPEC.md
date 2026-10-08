# compute-peek — a window into a long run

**Status:** v3, descriptive. Implemented by `compute-peek.py` in this directory.
v1 was a fixed three-pane readout for one GPU parameter sweep; it survives as the default
window and the default event vocabulary. v2 made the window declarative (peek-specs). v3
(2026-10-08) adds two things: the launcher drives **tmux or herdr**, and the same window can
watch **agent swarms** — an hour-long sprint or a multi-week campaign — through an
extension of the same event stream (§2b, §3b, §9).

## 0. What this is

Some runs take minutes to weeks, and the operator cannot watch a log. A run might be a
compute sweep (units of GPU/CPU work) or a swarm (tasks worked by agents that talk to each
other and sometimes need a human). This document is **descriptive**: it names the *kinds of
information* a window into such a run can show, the event stream that can feed them, and
the files that carry them. It does not prescribe a layout — each run gets its own
**peek-spec**, and a peek-spec composes the blocks below into panes.

A good window answers, in descending order of glance-value:

> **is anything on fire → how far along / how long left → what is running now →
> per-group progress → the last numbers → the hardware or the bill → raw log.**

For a swarm the first answer grows a second half: **is anyone waiting on me?** A question
or escalation addressed to the human outranks everything except a crash.

The top of the window is the highest answer; detail increases downward. Everything above
must be readable in one look; everything below must be reachable without leaving the
window. I.e. **explicit and glanceable, never at the cost of comprehensive** — hierarchy is
what lets both be true at once.

Two more standing notes:

- **The cost is always in the window.** For compute that is the hardware (`gpu`); for a
  swarm it is tokens and dollars (`cost`). A run's cost and its bottleneck live there.
- **Tone is free.** Whimsy de-stresses the human and invites the agent to explore. Character
  lives in the spec's `theme`, never in this code.

**The boundary of "descriptive".** The renderer understands a fixed vocabulary: the blocks
in §3/§3b and the events in §2/§2b. A spec may use a subset (a simple job needs no
`stage`), but it cannot invent a block without adding one to `BLOCKS` in the script. The
shape is declarative; the vocabulary is code. That is the deliberate stopping point — a
template language would buy a little more freedom for a lot more surface.

## 1. The peek-spec — the window, and the library

One JSON file per run. It declares which panes exist, what each pane shows, and the tone.
Everything else about the run (paths, command, devices, metrics) is there too.

```jsonc
{
  "title": "lr sweep · 3 schedules × 5 rates × 3 seeds",
  "workspace": "lr-sweep",            // tmux session / herdr workspace to find or create
  "mux": "tmux",                      // optional: tmux | herdr | auto (see §4)
  "cwd": "/path/to/project",          // default: cwd of `launch`
  "panes": [                          // the window. omit entirely -> DEFAULT_PANES
    {"name": "summary", "blocks": ["title","divider","meter","stage","groups","now",
                                   "last","gpu","log","stale","thought","footer"]},
    {"name": "runs",    "blocks": ["group_table","units"]},
    {"name": "log",     "blocks": ["raw"]}
  ],
  "progress": "logs/sweep/progress.jsonl",  // list or glob for sharded runs
  "log": "logs/sweep/sweep.log",            // raw log path(s); shard i tees to log[i]
  "command": "...",                         // a list -> one process per shard
  "gpu": "GPU-xxxxxxxx-…",              // device UUID, or a list of them. never an index
  "every": 5,                           // repaint seconds
  "stale_after_s": 600,                 // heartbeat gap before the window shouts STALE
  "metrics": ["val_loss","tok_per_s"],  // which run_end metrics to show, in order
  "failure_patterns": ["Traceback", "…"],// regexes grepped over the log
  "theme": {"icon":"🔥","meter_label":"burn-in","stages":[…],
            "thought_key":"val_loss","thoughts":[[0.0,"…"]]}
}
```

Swarm specs add `rate_window_s`, `budget_usd`, `agent_stale_after_s`, `feed_keep` (§2b);
`init --kind swarm` writes a starter with a swarm-shaped window.

- `panes[0]` is the root pane (the workspace's own pane); each later pane splits off the
  previous one and may carry `"dir"` (`down`/`right`) and `"ratio"` (`0..1`, the share the
  *existing* pane keeps) to reshape the layout. Panes beyond the third default to `down`. A
  pane whose blocks are `["raw"]` is **not** drawn by the renderer — it is where the
  command's own stdout lives.
- `DEFAULT_PANES` (the spec above, minus the keys shown only for illustration) is what you
  get if you declare none.
- `theme` is the only place a run's personality lives: `icon`, `meter_label` (the word next
  to the big progress bar — a training run's might be "burn-in", a census's "tiles"),
  `stages` (labels, one per equal slice of progress), and `thought_key` + `thoughts`
  (`[[threshold, text], …]`, highest threshold ≤ the metric wins).

### The library

A project that needs this repeatedly should not invent a new window from scratch each time.
Created specs live together, in the project:

```
<project>/.peek/*.json           # the peek-spec library (gitignore or commit — see below)
<project>/scripts/vitals_*.json  # legacy location, still found by `specs`
```

```bash
compute-peek.py specs                 # list them with their pane/block shape
compute-peek.py init --manifest .peek/<run>.json --like .peek/<other>.json
```

`init --like` clones the **shape** — `panes`, `theme`, `metrics`, `failure_patterns`,
`every`, `stale_after_s`, `workspace` — and blanks everything run-specific: `title`,
`command`, `progress`, `log`, `gpu`. Paths are never inherited, because inheriting them
would silently overwrite the other run's output.

**gitignore or commit?** Gitignored `.peek/` is per-machine scratch: reusable by the agents
on that machine, invisible to another clone. Committed `.peek/` is reusable everywhere but
is now an artifact the project maintains. Both are fine; pick per project and say which in
the project's own docs.

## 2. The default feeder — progress JSONL

A run that emits this feeds every block. One JSON object per line, appended to the path the
run was given (`--progress`). Every record carries the common fields.

| field | type | meaning |
|---|---|---|
| `ts` | float | `time.time()` at emit |
| `event` | string | `sweep_start` · `run_start` · `run_end` · `phase` · `error` · `sweep_end` (+ §2b) |
| `k`, `n` | int | units finished so far, total units |
| `elapsed_s` | float | seconds since the sink was created |
| `eta_s` | float or null | `mean(wall of finished units) × (n − k)`; null until one unit finished |

Event-specific fields:

- **`sweep_start`** — `groups: [str]` (display order), `units: [str]` (the full plan, in run
  order), `meta: {}` (anything). A unit id is `"<group>=<value> <tag>"`; the group is the
  text before the first `=` (or the first whitespace if there is no `=`).
- **`run_start`** — `group`, `unit`.
- **`run_end`** — `group`, `unit`, `wall_s`, `metrics: {name: number}`. Only numbers in
  `metrics`; the readout prints them with 3 significant digits.
- **`phase`** — optional; `unit`, `phase: str`, for multi-stage units. *Declared but not yet
  rendered.*
- **`error`** — `unit`, `message`. Optional; `log` also greps the raw log.
- **`sweep_end`** — `out` (where the result landed).

Rules:

1. The file is truncated at `sweep_start` and appended after. A resumed run re-emits
   `sweep_start` with the full plan and then `run_end` for already-finished units (with
   their recorded wall) before continuing, so the readout never has to know about resumes.
2. Emit `run_start` *before* the unit's work and `run_end` *after*; the `now` line and the
   stale detector depend on it.
3. Stdout stays human: `k/N done, ETA x min` after every unit, unbuffered (`python3 -u`,
   `flush=True`). The JSONL is for machines; the log is for eyes and for the failure grep.

Reference emitter: `peek_progress.py` in this directory (stdlib only; copy it or import
it). Any language can emit this. A job that emits only `sweep_start` + `run_end` works
too: the `stage`, `now` and `phase`-shaped parts of the window simply stay empty.

### Sharding

A host-bound run (a simulator whose tick is CPU-bound while the GPU idles) goes faster as
several processes on one card. Nothing changes: each shard emits its own JSONL with its own
`sweep_start` plan (distinct unit ids — seeds, or a shard tag), and the spec lists them:

```jsonc
"progress": "logs/sweep/progress_s*.jsonl",
"log":      ["logs/sweep/s0.log", "logs/sweep/s1.log", "logs/sweep/s2.log"],
"command":  ["... --seed-start 0 --threads 8 --progress logs/sweep/progress_s0.jsonl …",
             "... --seed-start 1 --threads 8 --progress logs/sweep/progress_s1.jsonl …",
             "... --seed-start 2 --threads 8 --progress logs/sweep/progress_s2.jsonl …"]
```

The readout unions the plans, sums k/n, takes the slowest shard's ETA, and shows every live
unit on the `now` line. Keep the summed thread count within the host's budget. Merging
results is the driver's job, not the readout's.

## 2b. Agent runs — the swarm extension

A swarm is the same shape as a sweep: **tasks are units**. An orchestrator emits
`sweep_start` with the task plan, `run_start`/`run_end` around each task, and the
existing meter, groups, units, group table and ETA work unchanged. Five additions carry
what a sweep does not have — agents, conversation, money, dependencies, and a plan that
grows:

| event / field | carries | fold rule |
|---|---|---|
| `sweep_start` + `deps: {unit: [unit]}` | task dependencies | `run_end` with `status: ok` unlocks dependents |
| `sweep_start` + `budget_usd` | spend ceiling | drawn as a bar in `cost` (spec `budget_usd` overrides) |
| `run_start` / `run_end` + `agent` | who worked the task | roster's current task, per-agent done count |
| `run_end` + `status` | `ok` (default) · `failed` · `abandoned` | failed/abandoned count as finished, never unlock dependents, show ✗ |
| **`plan`** | `units: [..]`, `groups: [..]`, `deps: {..}` appended mid-run | grows the plan and `n`; **never truncates** (§2 rule 1 is `sweep_start`'s alone) |
| **`agent`** | `agent`, `state`, `task?`, `note?`; identity, sticky: `model?`, `harness?`, `gpu?` (label of a local model's card, e.g. `"GPU0"`), `host?` | latest state per agent, with the time it was entered; a bounded lane history of (state, task stage) |
| **`message`** | `id`, `from`, `to?`, `kind`, `text`, `needs_human?`, `re?` | bounded feed; `re: <id>` resolves an earlier message |
| **`usage`** | `agent`, `tokens`, `usd`, (`tokens_in`, `tokens_out`) — **cumulative** for that agent | latest wins per agent; totals are the sum |

Agent `state` is one of `working` · `idle` · `blocked` · `waiting_human` · `done` ·
`failed`. Message `kind` is one of `claim` · `answer` · `question` · `escalation` · `stall`
· `note`. A message is **open** when it is a `question`/`escalation`/`stall` or carries
`needs_human: true`, and no later message has `re` equal to its `id`. Open messages
addressed to the human (`needs_human`, or `to: "human"`) are the swarm's "on fire".

Rules (in addition to §2's):

4. **Emit transitions, not activity.** One `agent` event when an agent changes state, not
   one per tool call or token. That keeps a three-week campaign's file in the thousands of
   lines, not millions, and keeps every repaint cheap.
5. **`usage` is cumulative per agent.** A dropped or duplicated line then cannot skew the
   total; the readout keeps the latest record per agent.
6. **A campaign's plan grows with `plan`.** Unknown work at the start is normal for a swarm;
   emit tasks as they are discovered. Re-emitting `sweep_start` restarts the run (§2 rule 1).
7. Emit `eta_s: null` unless the orchestrator knows better: with several agents in
   parallel, mean-wall × remaining overstates by the concurrency. The readout then uses
   throughput (§9).

The same `peek_progress.py` emits these (`agent()`, `message()`, `usage()`, `plan()`).

## 3. The block catalogue

A block is one named piece of the state, rendered as zero or more lines. `panes[].blocks`
is an ordered list of these; unknown names are an error, and a block that has nothing to say
(e.g. `stage` before the first heartbeat) contributes nothing.

| block | shows | answers |
|---|---|---|
| `title` | icon, title, state, elapsed, ETA, expected finish | *how long left* |
| `divider` | a rule | — |
| `meter` | `theme.meter_label` + the big progress bar + units done + s/unit | *how far along* |
| `stage` | the themed stage labels, current one bold | *how far along, in the run's own terms* |
| `groups` | a bar per group, two per line | *per-group progress* |
| `now` | current unit(s), time in it, bar against its group's typical wall | *what is running* |
| `last` | last finished unit + its chosen metrics | *how it is doing* |
| `gpu` | one line per device: temp, util, VRAM, power | *the hardware* |
| `log` | failure-signature count + last log line | *is anything on fire* |
| `stale` | red STALE when no heartbeat for `stale_after_s` | *is it wedged* |
| `thought` | the themed one-liner for the last metric value | *tone* |
| `footer` | clock, repaint interval, spec name | *freshness* |
| `group_table` | per-group n/done, mean wall, ETA | *how long left, per group* |
| `units` | a scrolling window of ✓ / ✗ / ▶ / · per unit, with metrics | *comprehensive detail* |
| `charts` | wall time + each `metrics` key over finished units, as sparklines; alone in a pane → peek-tui (§3a) | *the trend* |
| `raw` | none — marks the pane where the command's own output goes | *everything else* |

### 3a. The `charts` pane — peek-tui (ratatui)

`{"name": "charts", "blocks": ["charts"]}` is the one pane not drawn by the Python
renderer when `peek-tui/target/release/peek-tui` exists (`compute-peek.py build-tui`). It is a
ratatui app with four tabs (`1`–`4`, Tab, ←→; `?` lists keys):

| tab | shows | widgets |
|---|---|---|
| `overview` | the production readout, glance-value order: header · big ETA + stage track + live unit (tall panes) · progress · per-group bars · a chart per series (wall s + metrics, raw + rolling mean) · GPU util history | BigText, custom gradient gauge, LineGauge, braille Chart, per-bar-coloured Sparkline |
| `units` | every planned unit with status, wall and heat-coloured metrics; mean wall per group; wall-time histogram | Table + Scrollbar (follows the running unit; `j`/`k` scroll, `f` follow), BarChart |
| `atlas` | metric[0] vs metric[1] by group with a trail through the latest units; groups × units heatmap of metric[2]; each metric's last value within its range | Canvas, custom half-block heatmap, LineGauge |
| `hardware` | per GPU: util, VRAM, power vs limit, temperature, and their history | gradient gauge, LineGauge, 3-series Chart with legend |

Layouts are computed from the pane size every frame: chart grid columns = width / 34 (more
if rows would drop under 7 lines), GPU tiles stack when narrower than 44 cells each, the ETA
hero appears at ≥ 96 × 38, axis labels and legends drop out of small charts. GPU history
(300 samples) lives only in the process — the one thing a stateless repaint cannot show.

Folding stays in Python: each tick peek-tui runs `compute-peek.py state --manifest M` and
draws the JSON, so the JSONL contract has exactly one reader; the call runs on a background
thread, so the UI repaints at 4 Hz regardless. `peek-tui --manifest M --once --tab N --width W
--height H` prints one frame as plain text, for checks and for agents.

`compute-peek.py demo` writes `~/.cache/compute-peek/demo/demo.json` (all GPUs, charts pane
first, 4 metrics) and launches it with `demo-feed` — a synthetic §2 emitter that loops
48-unit sweeps forever — as the command. It is the showcase and the smoke test.

Unbuilt, or with `charts` beside other blocks, the block renders as one-line sparklines
(ratatui's `NINE_LEVELS` bar set). Bars everywhere use ratatui's horizontal eighths, so
progress moves in 1/8-cell steps. Reference for further widgets: ratatui upstream
(github.com/ratatui/ratatui) — `ratatui-core/src/symbols/`, `ratatui-widgets/src/`,
`examples/apps/`.

### 3b. Swarm blocks

| block | shows | answers |
|---|---|---|
| `roster` | one line per agent, sorted waiting-on-human → blocked → failed → working → idle → done: state, time in state, task, tokens, $, tasks done; an agent silent past `agent_stale_after_s` is marked `?` | *who is doing what, who is stuck* |
| `feed` | header with the count of open messages (red when any needs the human), then every open message, then the latest messages, newest last, fitted to the pane | *is anyone waiting on me; what are they saying* |
| `cost` | total tokens and $, a bar against `budget_usd`, burn rate over the rate window, projected total at finish (spend per finished task × remaining + spent), top spenders | *the bill, and where it is heading* |
| `deps` | counts of done / running / ready / blocked / failed tasks, then the ready frontier, then tasks blocked by a failed dependency | *what unlocks next; what is stuck behind a failure* |
| `lanes` | one swimlane per agent across the run (`lanes_window_s` to narrow it): working cells coloured by the task's stage, red waiting on the human, yellow blocked, a dash idle | *who did what, when; where the time went* |
| `cpu` | host CPU bar, per-core heat strip, load, memory (Linux `/proc`; load only elsewhere) | *the CPU side of the compute* |

`gpu` also names a swarm's local agents on the card they declared (`gpu: "GPU0"`), and `cost`
splits local tokens ($0) from cloud spend and draws the spend over the run as a sparkline.

Theme keys for swarms, all optional: `agent_icons` (`{name: icon}` or a list dealt out in name
order), `state_icons` (`{state: icon}`), `kind_icons` (`{message kind: icon}`). Use
double-width emoji; alignment counts terminal cells, not characters.

`init --kind swarm` writes this window (all of it reorderable):

```
summary:  title · divider · meter · stage · cost · roster · lanes · stale · thought · footer
board:    feed · deps · group_table
log:      raw   (the orchestrator, or `tail -F` of its log)
```

## 4. The launcher, and why it is idempotent

`compute-peek.py launch --manifest M [--start] [--mux tmux|herdr|auto]`

The multiplexer is picked by `--mux`, else `$PEEK_MUX`, else the spec's `mux`, else
**auto**: inside tmux (`$TMUX` set) → tmux; otherwise a running herdr server, otherwise a
running tmux server. A "workspace" is a tmux session or a herdr workspace; pane ids are the
mux's own (`%N`, `wXX:pN`). `$PEEK_TMUX_SOCKET` selects `tmux -L <name>`. tmux ≥ 3.1.

**compute-peek never starts a multiplexer server.** A server started from a script or an
agent session inherits that process's environment for every pane it ever opens. With none
running, `launch` says how to start one and exits.

1. Find the workspace whose label equals `spec.workspace` (tmux: the session name, with
   `.`/`:` turned into `_` as tmux does); create it if absent.
2. Read `M.state.json` (mux name + pane ids from an earlier launch; recorded under another
   mux → ignored). For each declared pane: keep the recorded pane if it still exists,
   otherwise split a new one off the previous pane (using its `dir`/`ratio`). Title panes
   `<workspace>:<pane name>`.
3. For every non-`raw` pane: if the pane's shell holds the foreground (the shell is the
   terminal's foreground process-group leader), run a renderer in it (`pane --name
   <pane>`, or peek-tui for a built `charts` pane); if a job is already running, leave it
   alone.
4. With `--start`: run `spec.command` in the raw pane **only if** that pane is at a shell
   prompt **and** the progress file does not show a live run (started, not ended, heartbeat
   younger than `stale_after_s`). A `command` list starts one process per shard, in the
   declared raw panes plus extras split downward. Otherwise print why and do nothing.
5. Write `M.state.json`.

Running `launch` twice, or after a terminal restart, or after the user closed one pane,
converges to the same panes with the same readouts and never starts the compute twice. That
is the whole point: the visibility setup is a declared state, not a sequence of steps
someone has to remember.

The default window's layout (208 × 58 pane, ratios 0.42 then 0.5):

```
┌───────────────────── summary: title · state · elapsed · ETA · done ≈ HH:MM ─────────────────────┐
│ burn-in ▕████████░░░░░░░░▏ 29/48 units  stage 🌱 → 🌿 → 🌳 → 🍎                                  │
│ per-group bars · now (unit, started … ago, typ. wall) · last (metrics) · GPU · log ✓/✗ · ❝thought❞│
├────────────────── runs: per-group ETA, then ✓ done / ▶ running / · pending ──┬──── raw log ───────┤
│                                                                              │ tee of the command │
└──────────────────────────────────────────────────────────────────────────────┴────────────────────┘
```

## 5. Devices are named by UUID

`spec.gpu` takes a **device UUID** (`GPU-xxxxxxxx-…`), or a list of them. Never an index.
On a machine with mixed cards the two orders disagree, e.g.:

```
nvidia-smi --id=0        → the card in the first PCI slot
CUDA_VISIBLE_DEVICES=0   → the fastest card (CUDA's default order, without CUDA_DEVICE_ORDER=PCI_BUS_ID)
```

An index therefore means different cards to the launcher and to the run, and a window
watching the wrong card is worse than no window. A UUID is the same card to both.
`compute-peek.py gpus` prints them.

Pin the run the same way — `CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-…`, which needs no
`CUDA_DEVICE_ORDER` — so the two declarations cannot drift. The script accepts an index for
convenience; a new spec should not use one. No NVIDIA GPU → the `gpu` block says so and
everything else works.

## 6. What an agent reads

`compute-peek.py status --manifest M` — one-shot plain text: state, k/N, ETA, per-group ETA,
current unit, last metrics, one hardware line per device, last failure line; for a swarm
also the open messages, agent states and spend. This is what an agent polls instead of
reading panes; reading the summary pane from the mux (`tmux capture-pane -p -t <id>`,
`herdr pane read <id>`) or `pane --name <n> --once` is the human-shaped equivalent.

## 7. Guards

- The ETA is an extrapolation. It is honest only when units are alike; a plan whose units
  differ 10× in cost should be split into groups (the per-group ETA uses the group's own
  walls once one has finished).
- A run that emits nothing for `stale_after_s` is flagged STALE in red. That is a prompt to
  look, not a verdict — a single unit can legitimately be slow. Campaigns want hours here.
- Failure signatures are grepped from the raw log every repaint. The `\bError\b` default will
  not match `ValueError:` (no word boundary), so add project-specific patterns. A pattern
  that does not compile is reported in the `log` line rather than killing the readout.
- `launch` talks to the mux from wherever it is run; it does not need to run inside a pane.

## 8. Where things live

- `compute-peek.py` — the one implementation (stdlib only, no project imports).
  `compute-peek.py selftest` is its runnable check.
- `peek_progress.py` — the reference emitter (§2, §2b), stdlib only.
- `peek-tui/` — the ratatui `charts` pane (Rust crate; `target/` is build output, rebuild
  with `compute-peek.py build-tui`).
- `SKILL.md` — the thin wrapper an agent reads.
- `<project>/.peek/*.json` — the project's peek-spec library; `scripts/vitals_*.json` is the
  legacy location and is still listed by `specs`.
- `<spec>.state.json` (pane ids) and `<spec>.fold.json` (fold checkpoint, §9) sit next to
  each spec; both are disposable.

## 9. Scale — one hour to many weeks

A sprint and a campaign differ by three orders of magnitude in duration; the readout has to
be honest at both ends.

- **Incremental fold.** The fold is a reducer over events. A long-lived renderer keeps, per
  progress file, `(inode, byte offset, first line, folded state)` and folds only new bytes.
  One-shot readers (`status`, `state` — which peek-tui calls every tick) load the same
  checkpoint from `<spec>.fold.json`, written once a progress file passes 1 MB. The fold
  restarts from zero when the inode changes, the file shrinks, or its first line differs
  (truncated by a new `sweep_start` and regrown between two reads). A partial last line is
  left for the next read.
- **Bounded memory.** The feed keeps `feed_keep` messages (default 200) plus every open one;
  the spend history keeps ≤ 2000 points, thinned evenly as it grows.
- **Throughput ETA.** When the emitter gives no `eta_s`, the readout divides the remaining
  tasks by the finish rate over a trailing window: `rate_window_s` if set, else
  `min(elapsed, 24 h)` — the whole run for a sprint, the last day for a campaign. Burn rate
  ($/h) uses the same window.
- **Per-group ETAs are per stage.** In a pipeline (design → build → test), a late stage's
  own rate is low until work reaches it, so its ETA can exceed the headline's. The headline
  is the run's; the group table answers "when does this stage drain at its current pace".
- **Durations print in days** past 24 h (`3d 04h`); the expected finish prints a date once
  it is more than 20 h away.

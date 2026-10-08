"""peek_progress — reference emitter for the compute-peek progress JSONL (SPEC.md §2, §2b).

Stdlib only.  Copy this file into a project or import it from here.

Compute sweep::

    p = Progress('logs/run/progress.jsonl')
    p.sweep_start(units=['lr=1e-3 s0', 'lr=1e-3 s1', 'lr=3e-4 s0'])
    for u in units:
        p.run_start(u)
        metrics = work(u)
        p.run_end(u, metrics=metrics)          # wall time measured from run_start
        print(f'{p.k}/{p.n} done, ETA {p.eta_s}', flush=True)
    p.sweep_end(out='results.json')

Agent swarm (tasks are units; several run at once, so leave ETA to the readout)::

    p = Progress('logs/swarm/progress.jsonl', eta=False)
    p.sweep_start(units=['design', 'build', 'test'], deps={'build': ['design'],
                                                           'test': ['build']}, budget_usd=40)
    p.agent('ada', 'working', task='design')
    p.run_start('design', agent='ada')
    q = p.message('ada', 'question', 'which API version?', needs_human=True)
    p.message('human', 'answer', 'v2', re=q)
    p.usage('ada', tokens=48_000, usd=0.61)        # cumulative for that agent
    p.run_end('design', agent='ada')               # status='failed' / 'abandoned' too
    p.plan(units=['docs'], deps={'docs': ['build']})   # the plan grows mid-run
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path


def unit_group(unit: str) -> str:
    return unit.split('=', 1)[0] if '=' in unit else unit.split()[0]


class Progress:
    """Append-only JSONL sink.  `sweep_start` truncates (SPEC §2 rule 1); nothing else does."""

    def __init__(self, path: str | Path | None, *, eta: bool = True):
        self.path = Path(path) if path else None
        self.eta = eta                    # False for concurrent agents: readout uses throughput
        self.n, self.k, self.walls = 0, 0, []
        self.t0 = time.time()
        self._open: dict[str, float] = {}
        self._ids = itertools.count(1)

    # -- core
    @property
    def eta_s(self) -> float | None:
        if not self.eta or not self.walls:
            return None
        return sum(self.walls) / len(self.walls) * (self.n - self.k)

    def emit(self, event: str, *, ts: float | None = None, **kw) -> dict:
        rec = {'ts': ts or time.time(), 'event': event, 'k': self.k, 'n': self.n,
               'elapsed_s': (ts or time.time()) - self.t0, 'eta_s': self.eta_s, **kw}
        if self.path:
            with self.path.open('a') as f:
                f.write(json.dumps(rec) + '\n')
        return rec

    # -- plan
    def sweep_start(self, units: list[str], groups: list[str] | None = None, *,
                    deps: dict | None = None, budget_usd: float | None = None,
                    meta: dict | None = None, ts: float | None = None) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text('')
        self.t0 = ts or time.time()
        self.n, self.k, self.walls = len(units), 0, []
        groups = groups or list(dict.fromkeys(unit_group(u) for u in units))
        extra = {k: v for k, v in (('deps', deps), ('budget_usd', budget_usd)) if v}
        self.emit('sweep_start', ts=ts, units=list(units), groups=groups, meta=meta or {},
                  **extra)

    def plan(self, units: list[str], groups: list[str] | None = None, *,
             deps: dict | None = None, ts: float | None = None) -> None:
        self.n += len(units)
        self.emit('plan', ts=ts, units=list(units), groups=groups or [], deps=deps or {})

    def sweep_end(self, out: str | None = None, ts: float | None = None) -> None:
        self.emit('sweep_end', ts=ts, out=out)

    # -- units / tasks
    def run_start(self, unit: str, *, agent: str | None = None, ts: float | None = None) -> None:
        self._open[unit] = ts or time.time()
        self.emit('run_start', ts=ts, unit=unit, group=unit_group(unit),
                  **({'agent': agent} if agent else {}))

    def run_end(self, unit: str, *, metrics: dict | None = None, wall_s: float | None = None,
                agent: str | None = None, status: str = 'ok', ts: float | None = None) -> None:
        now = ts or time.time()
        wall = wall_s if wall_s is not None else now - self._open.pop(unit, now)
        self.k += 1
        self.walls.append(wall)
        kw = {'agent': agent} if agent else {}
        if status != 'ok':
            kw['status'] = status
        self.emit('run_end', ts=ts, unit=unit, group=unit_group(unit), wall_s=wall,
                  metrics=metrics or {}, **kw)

    # -- agents
    def agent(self, name: str, state: str, *, task: str | None = None,
              note: str | None = None, ts: float | None = None) -> None:
        """state: working | idle | blocked | waiting_human | done | failed.  Transitions only."""
        kw = {k: v for k, v in (('task', task), ('note', note)) if v is not None}
        self.emit('agent', ts=ts, agent=name, state=state, **kw)

    def message(self, frm: str, kind: str, text: str, *, to: str | None = None,
                needs_human: bool = False, re: str | None = None, id: str | None = None,
                ts: float | None = None) -> str:
        """kind: claim | answer | question | escalation | stall | note.  Returns the id;
        pass it as `re=` on the reply that resolves it."""
        mid = id or f'm{next(self._ids)}'
        kw = {k: v for k, v in (('to', to), ('re', re)) if v is not None}
        if needs_human:
            kw['needs_human'] = True
        self.emit('message', ts=ts, id=mid, **{'from': frm}, kind=kind, text=text, **kw)
        return mid

    def usage(self, agent: str, *, tokens: int, usd: float, tokens_in: int | None = None,
              tokens_out: int | None = None, ts: float | None = None) -> None:
        """Cumulative totals for this agent (latest record wins in the readout)."""
        kw = {k: v for k, v in (('tokens_in', tokens_in), ('tokens_out', tokens_out))
              if v is not None}
        self.emit('usage', ts=ts, agent=agent, tokens=tokens, usd=round(usd, 6), **kw)

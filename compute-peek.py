#!/usr/bin/env python3
"""compute-peek — glanceable live readout for a long compute run, in tmux or herdr panes.

Descriptive, not prescriptive: a run emits a progress JSONL, a peek-spec declares the
window (which panes, which blocks in each pane, and the tone), and `launch` converges
to that window in a tmux session or herdr workspace.  The contract is SPEC.md next to
this file.

    P=path/to/compute-peek.py
    python3 $P gpus                                          # index / UUID / name, for the spec
    python3 $P init   --manifest .peek/<run>.json [--like <spec>]
    python3 $P launch --manifest M [--start]                 # find/create panes, start readouts
    python3 $P show   --manifest M [--once]                  # first renderer pane
    python3 $P runs   --manifest M [--once]                  # second renderer pane
    python3 $P pane   --manifest M --name units [--once]     # any declared pane, by name
    python3 $P status --manifest M                           # one-shot plain text, for an agent
    python3 $P specs  [--project DIR]                        # the project's peek-spec library
    python3 $P selftest                                      # runnable check of the renderer

`launch` finds (or creates) the workspace named in the spec (mux: --mux, $PEEK_MUX,
spec "mux", else auto — inside tmux → tmux, else a running herdr, else a running tmux), lays out the declared panes,
and starts each renderer if the pane is idle.  Running it twice changes nothing.  `--start`
runs the spec's command in the raw pane, and refuses if that pane is busy or the progress
file shows a live run.

Devices are named by **UUID**, never index: an index means different cards depending on
`CUDA_DEVICE_ORDER`.  `gpus` prints the UUIDs to paste into the spec.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

B, D, R, G, Y, C, M, X = ('\033[1m', '\033[2m', '\033[31m', '\033[32m', '\033[33m',
                          '\033[36m', '\033[35m', '\033[0m')
CLEAR = '\033[2J\033[H'
DEFAULT_FAILURES = [r'Traceback', r'\bKilled\b', r'OOM|out of memory', r'CUDA error',
                    r'\bError\b']

# The window a spec gets when it declares none: three panes, the shape this skill grew up
# with.  A bespoke window is a different list of panes and/or a different blocks list.
DEFAULT_PANES = [
    {'name': 'summary',
     'blocks': ['title', 'divider', 'meter', 'stage', 'groups', 'now', 'last', 'gpu',
                'log', 'stale', 'thought', 'footer']},
    {'name': 'runs', 'blocks': ['group_table', 'units']},
    {'name': 'log', 'blocks': ['raw']},
]

# `init --kind swarm`: the window for agent runs (SPEC §3b).  Same rules, different blocks.
SWARM_PANES = [
    {'name': 'summary',
     'blocks': ['title', 'divider', 'meter', 'stage', 'cost', 'roster', 'lanes', 'stale',
                'thought', 'footer']},
    {'name': 'board', 'blocks': ['feed', 'deps', 'group_table']},
    {'name': 'log', 'blocks': ['raw']},
]


# ------------------------------------------------------------------ manifest + progress
def load_manifest(path: str) -> dict:
    m = json.loads(Path(path).read_text())
    m.setdefault('cwd', os.getcwd())
    m.setdefault('failure_patterns', DEFAULT_FAILURES)
    m.setdefault('theme', {})
    if not m.get('panes'):
        m['panes'] = [dict(p) for p in DEFAULT_PANES]
    m['_path'] = str(Path(path).resolve())
    m['_state'] = str(Path(path).resolve().with_suffix('.state.json'))
    m['_fold'] = str(Path(path).resolve().with_suffix('.fold.json'))
    for k in ('progress', 'log'):
        v = m.get(k)
        if v is None:
            continue
        vs = v if isinstance(v, list) else [v]
        m[k] = [x if os.path.isabs(x) else os.path.join(m['cwd'], x) for x in vs]
    g = m.get('gpu')
    m['_gpus'] = [g] if isinstance(g, (str, int)) else list(g or [])
    return m


def panes_of(m: dict) -> list[dict]:
    return m.get('panes') or DEFAULT_PANES


def pane_by_name(m: dict, name: str) -> dict:
    ps = panes_of(m)
    for p in ps:
        if p.get('name') == name:
            return p
    if name.isdigit() and int(name) < len(ps):
        return ps[int(name)]
    raise SystemExit(f'no pane {name!r}; this spec declares: '
                     f'{", ".join(str(p.get("name")) for p in ps)}')


def render_panes(m: dict) -> list[dict]:
    """Panes a renderer process draws; the `raw` pane is the command's own output."""
    return [p for p in panes_of(m) if 'raw' not in (p.get('blocks') or [])]


def expand(paths) -> list[str]:
    """Manifest paths may be a string, a list, and/or globs (shards: progress_s*.jsonl)."""
    out = []
    for p in ([paths] if isinstance(paths, str) else list(paths or [])):
        out.extend(sorted(glob.glob(p)) if any(c in p for c in '*?[') else [p])
    return out


FOLD_CKPT_MIN_BYTES = 1 << 20   # progress files past this get an on-disk fold checkpoint
FEED_KEEP = 200                  # messages kept besides the open ones (spec: feed_keep)
SERIES_MAX = 2000                # spend-history points; thinned evenly past this
HIST_MAX = 400                   # state transitions kept per agent (the `lanes` block)
OPEN_KINDS = ('question', 'escalation', 'stall')
_FOLDS: dict = {}                # path -> {ino, off, first, st}: long-lived renderers fold deltas


def _empty() -> dict:
    return {'started': False, 'ended': False, 'units': [], 'groups': [], 'meta': {},
            'k': 0, 'n': 0, 'elapsed_s': 0.0, 'eta_s': None, 'current': None,
            'current_since': None, 'live': {}, 'done': {}, 'walls': {}, 'last': None,
            'last_ts': None, 't0': None, 'shards': 0, 'ends': [],
            # agent runs (SPEC §2b)
            'deps': {}, 'agents': {}, 'feed': [], 'open': {}, 'usage': {}, 'spend': [],
            'budget_usd': None}


def _agent(st: dict, name: str, ts) -> dict:
    a = st['agents'].get(name)
    if a is None:
        a = st['agents'][name] = {'state': 'idle', 'since': ts, 'task': None, 'note': '',
                                  'last_ts': ts, 'done': 0, 'hist': [[ts, 'idle', None]]}
    a['last_ts'] = ts or a['last_ts']
    return a


def _set_state(a: dict, state: str, ts) -> None:
    if state and state != a['state']:
        a['state'], a['since'] = state, ts


def _mark(a: dict, ts) -> None:
    """Append to the agent's lane history when (state, stage of its task) changes."""
    grp = unit_group(a['task']) if a['state'] == 'working' and a.get('task') else None
    h = a.setdefault('hist', [])
    if not h or h[-1][1:] != [a['state'], grp]:
        h.append([ts, a['state'], grp])
        if len(h) > HIST_MAX:
            del h[:len(h) - HIST_MAX]


def _add_units(st: dict, units, groups=(), deps=None) -> None:
    have = set(st['units'])
    for u in units:
        if u not in have:
            st['units'].append(u)
            have.add(u)
            st['n'] += 1
            g = unit_group(u)
            if g not in st['groups']:
                st['groups'].append(g)
    for g in groups:
        if g not in st['groups']:
            st['groups'].append(g)
    st['deps'].update(deps or {})


def _apply(st: dict, e: dict) -> None:
    """The fold: one event into the state.  Pure function of (state, event)."""
    ev, ts = e.get('event'), e.get('ts')
    st['last_ts'] = ts or st['last_ts']
    if ev == 'sweep_start':
        st.clear()
        st.update(_empty())
        st.update(started=True, units=list(e.get('units', [])), groups=list(e.get('groups', [])),
                  meta=e.get('meta', {}), t0=ts, last_ts=ts, deps=dict(e.get('deps') or {}),
                  budget_usd=e.get('budget_usd'))
        st['n'] = e.get('n', len(st['units']))
    elif ev == 'plan':
        _add_units(st, e.get('units', []), e.get('groups', []), e.get('deps'))
    elif ev == 'run_start':
        u, who = e.get('unit'), e.get('agent')
        if who is None:
            st['live'] = {}          # single-lane emitter: a new unit replaces the old
        st['live'][u] = {'since': ts, 'agent': who}
        if who:
            a = _agent(st, who, ts)
            a['task'] = u
            if a['state'] in ('idle', 'done'):
                _set_state(a, 'working', ts)
            _mark(a, ts)
    elif ev == 'run_end':
        u, who = e.get('unit'), e.get('agent')
        st['done'][u] = e
        st['live'].pop(u, None)
        st['walls'].setdefault(e.get('group') or unit_group(u or '?'), []).append(
            e.get('wall_s', 0.0))
        st['last'] = e
        st['ends'].append(ts)
        st['k'] = e.get('k', len(st['done']))
        st['n'] = max(e.get('n', st['n']), st['n'])
        st['elapsed_s'], st['eta_s'] = e.get('elapsed_s', st['elapsed_s']), e.get('eta_s')
        if who:
            a = _agent(st, who, ts)
            a['done'] += 1
            if a['task'] == u:
                a['task'] = None
                if a['state'] == 'working':
                    _set_state(a, 'idle', ts)
                _mark(a, ts)
    elif ev == 'sweep_end':
        st['ended'] = True
        st['live'] = {}
    elif ev == 'agent':
        a = _agent(st, e.get('agent', '?'), ts)
        if e.get('state') and e['state'] != a['state'] and 'note' not in e:
            a['note'] = ''           # a note belongs to the state it was given with
        _set_state(a, e.get('state'), ts)
        if 'task' in e:
            a['task'] = e['task']
        if 'note' in e:
            a['note'] = e['note'] or ''
        for k in ('model', 'harness', 'gpu', 'host'):   # who this agent is; sticky
            if e.get(k) is not None:
                a[k] = e[k]
        _mark(a, ts)
    elif ev == 'message':
        rec = {k: e.get(k) for k in ('id', 'from', 'to', 'kind', 're', 'needs_human')}
        rec['ts'], rec['text'] = ts, str(e.get('text', ''))[:400]
        st['feed'].append(rec)
        if len(st['feed']) > 2 * FEED_KEEP:
            st['feed'] = st['feed'][-FEED_KEEP:]
        if rec['re'] is not None:
            st['open'].pop(str(rec['re']), None)
        if rec['id'] is not None and (rec['kind'] in OPEN_KINDS or rec['needs_human']):
            st['open'][str(rec['id'])] = rec
        if rec['from'] and rec['from'] != 'human':
            _agent(st, rec['from'], ts)
    elif ev == 'usage':
        who = e.get('agent', '?')
        st['usage'][who] = {k: e.get(k) for k in ('tokens', 'usd', 'tokens_in', 'tokens_out')}
        total = sum(u.get('usd') or 0 for u in st['usage'].values())
        sp = st['spend']
        if sp and ts and ts - sp[-1][0] < 60:
            sp[-1] = [ts, total]
        else:
            sp.append([ts, total])
        if len(sp) > SERIES_MAX:
            st['spend'] = sp[::2]


def _finish(st: dict) -> dict:
    """Derived fields, on a shallow copy so the cached fold is never mutated by readers."""
    out = dict(st)
    live = st['live']
    out['current'] = ' + '.join(live) if live else None
    out['current_since'] = min((v['since'] for v in live.values() if v['since']), default=None)
    return out


def _same_file(c: dict | None, ino: int, size: int, first: str) -> bool:
    return bool(c) and c['ino'] == ino and size >= c['off'] and c['first'] == first


def _read_one(path: str) -> dict:
    """Fold one JSONL into a state dict, folding only bytes not seen before.

    Missing file -> 'no heartbeat yet'.  The cached fold restarts when the inode
    changes, the file shrinks, or its first line differs (truncated and regrown).
    """
    p = Path(path) if path else None
    if p is None or not p.exists():
        return _finish(_empty())
    size, ino = p.stat().st_size, p.stat().st_ino
    with p.open('rb') as fh:
        first = fh.readline().decode(errors='replace')
        c = _FOLDS.get(path)
        if not _same_file(c, ino, size, first):
            c = {'ino': ino, 'off': 0, 'first': first, 'st': _empty()}
        if size > c['off']:
            fh.seek(c['off'])
            chunk = fh.read()
            end = chunk.rfind(b'\n')     # a partial last line waits for the next read
            if end >= 0:
                for line in chunk[:end].splitlines():
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(e, dict):
                        _apply(c['st'], e)
                c['off'] += end + 1
    _FOLDS[path] = c
    return _finish(c['st'])


def _load_ckpt(ckpt: str, files: list[str]) -> None:
    try:
        disk = json.loads(Path(ckpt).read_text())
    except (OSError, ValueError):
        return
    for f in files:
        d, mem = disk.get(f), _FOLDS.get(f)
        if d and (not mem or (d['ino'] == mem['ino'] and d['first'] == mem['first']
                              and d['off'] > mem['off'])):
            _FOLDS[f] = d


def read_progress(paths, ckpt: str | None = None) -> dict:
    """Fold one or many progress files (shards) into one state dict.

    Units and groups are unioned in shard order; k/n are summed; the ETA is the
    slowest shard's (they run concurrently); `current` lists every shard's live unit.
    `ckpt` (the spec's .fold.json) lets one-shot readers skip re-folding big files.
    """
    files = expand(paths)
    big = [f for f in files if os.path.exists(f) and os.path.getsize(f) >= FOLD_CKPT_MIN_BYTES]
    if ckpt and big:
        _load_ckpt(ckpt, big)
    before = {f: (_FOLDS.get(f) or {}).get('off') for f in big}
    shards = [_read_one(f) for f in files] if files else [_read_one('')]
    if ckpt and big and any((_FOLDS.get(f) or {}).get('off') != before[f] for f in big):
        tmp = Path(f'{ckpt}.{os.getpid()}.tmp')     # per-reader: concurrent writers never tear
        tmp.write_text(json.dumps({f: _FOLDS[f] for f in big if f in _FOLDS}))
        os.replace(tmp, ckpt)
    if len(shards) == 1:
        return shards[0]
    st = _empty()
    st['started'] = any(x['started'] for x in shards)
    st['ended'] = st['started'] and all(x['ended'] for x in shards if x['started'])
    for x in shards:
        for g in x['groups']:
            if g not in st['groups']:
                st['groups'].append(g)
        st['units'].extend(x['units'])
        st['done'].update(x['done'])
        st['live'].update(x['live'])
        for g, ws in x['walls'].items():
            st['walls'].setdefault(g, []).extend(ws)
        for key in ('deps', 'agents', 'open', 'usage'):
            st[key].update(x[key])
        st['feed'].extend(x['feed'])
        st['ends'].extend(x['ends'])
        st['k'] += x['k']
        st['n'] += x['n']
    st['feed'].sort(key=lambda r: r['ts'] or 0)
    st['ends'].sort()
    st['budget_usd'] = next((x['budget_usd'] for x in shards if x['budget_usd']), None)
    st['t0'] = min((x['t0'] for x in shards if x['t0']), default=None)
    st['last_ts'] = max((x['last_ts'] for x in shards if x['last_ts']), default=None)
    ends = [x['last'] for x in shards if x['last']]
    st['last'] = max(ends, key=lambda e: e['ts']) if ends else None
    etas = [x['eta_s'] for x in shards if x['eta_s'] is not None]
    st['eta_s'] = max(etas) if etas else None
    st['shards'] = len(shards)
    return _finish(st)


def unit_group(unit: str) -> str:
    return unit.split('=', 1)[0] if '=' in unit else unit.split()[0]


def group_stats(st: dict, cx: dict | None = None) -> list[dict]:
    """Per-group counts and ETA.  With `cx` and no emitter ETA (concurrent agents), the
    ETA is the group's own throughput over the rate window, like the headline (SPEC §9)."""
    out = []
    lo = cx['now'] - cx['window'] if cx and st['eta_s'] is None and not st['ended'] else None
    all_walls = [w for ws in st['walls'].values() for w in ws]
    for g in st['groups']:
        units = [u for u in st['units'] if unit_group(u) == g]
        done = [u for u in units if u in st['done']]
        walls = st['walls'].get(g) or all_walls
        mean = sum(walls) / len(walls) if walls else None
        eta = mean * (len(units) - len(done)) if mean is not None else None
        if lo is not None and len(done) < len(units):
            recent = sum(1 for u in done if (st['done'][u].get('ts') or 0) >= lo)
            span = min(cx['window'], cx['now'] - (st['t0'] or lo))
            if not recent and st['t0']:     # quiet lately: fall back to the whole run's rate
                recent, span = len(done), cx['now'] - st['t0']
            eta = (len(units) - len(done)) / (recent / span) if recent and span > 0 else None
        out.append({'group': g, 'n': len(units), 'done': len(done), 'mean_wall': mean,
                    'eta_s': eta})
    return out


# ------------------------------------------------------------------ system probes
def gpu_query(device: str | int) -> dict:
    """Raw readings for `device`, a UUID (preferred) or an nvidia-smi index."""
    try:
        out = subprocess.run(
            ['nvidia-smi', f'--id={device}',
             '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,'
             'memory.total,power.draw,power.limit', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5).stdout.strip()
        idx, name, temp, util, mu, mt, pw, pl = [s.strip() for s in out.split(',')]
        try:
            limit = float(pl)
        except ValueError:            # '[N/A]' on some boards
            limit = None
        return {'power_limit_w': limit, 'device': str(device), 'idx': int(idx),
                'name': name.replace('NVIDIA GeForce ', ''), 'temp': float(temp),
                'util': float(util), 'mem_used_gb': float(mu) / 1024,
                'mem_total_gb': float(mt) / 1024, 'power_w': float(pw)}
    except Exception as e:  # noqa: BLE001
        return {'device': str(device), 'error': type(e).__name__}


def gpu_line(device: str | int) -> str:
    """One hardware line for `device`."""
    g = gpu_query(device)
    if 'error' in g:
        return f'{D}device {device} unreadable ({g["error"]}){X}'
    tcol = R if g['temp'] >= 80 else (Y if g['temp'] >= 70 else G)
    return (f'GPU{g["idx"]} {g["name"]}  {tcol}{g["temp"]:.0f} °C{X} · {g["util"]:.0f} % · '
            f'{g["mem_used_gb"]:.1f}/{g["mem_total_gb"]:.0f} GB · {g["power_w"]:.0f} W')


def gpu_lines(devices) -> list[str]:
    return [gpu_line(d) for d in (devices or [])]


def log_scan(paths, patterns: list[str], tail: int = 3) -> dict:
    files = [f for f in expand(paths) if Path(f).exists()]
    if not files:
        return {'lines': [], 'hits': 0, 'last_hit': None}
    try:
        rx = re.compile('|'.join(f'(?:{p})' for p in patterns))
    except re.error as e:
        return {'lines': [f'bad failure_patterns: {e}'], 'hits': 0, 'last_hit': None}
    lines = [l for f in files for l in Path(f).read_text(errors='replace').splitlines()]
    hits = [l for l in lines if rx.search(l)]
    return {'lines': lines[-tail:], 'hits': len(hits), 'last_hit': hits[-1] if hits else None}


# ------------------------------------------------------------------ rendering
# Glyph sets from ratatui (github.com/ratatui/ratatui, ratatui-core/src/symbols): block.rs horizontal
# eighths for bars, bar.rs NINE_LEVELS for sparklines.  8x the resolution of whole cells.
H_EIGHTHS = ' ▏▎▍▌▋▊▉█'
V_LEVELS = ' ▁▂▃▄▅▆▇█'


def bar(frac: float, width: int = 24, col: str = G) -> str:
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    eighths = int(round(frac * width * 8))
    full, part = divmod(eighths, 8)
    fill = '█' * full + (H_EIGHTHS[part] if part else '')
    return f'▕{col}{fill}{D}{"░" * (width - len(fill))}{X}▏'


def spark(values, width: int = 40, lo: float | None = None, hi: float | None = None) -> str:
    """ratatui Sparkline in one line: the last `width` values on nine levels."""
    vs = [float(v) for v in values if v is not None][-width:]
    if not vs:
        return ''
    lo = min(vs) if lo is None else lo
    hi = max(vs) if hi is None else hi
    if hi <= lo:
        return V_LEVELS[4] * len(vs)
    return ''.join(V_LEVELS[max(1, min(8, int(round((v - lo) / (hi - lo) * 8))))] for v in vs)


def cells(text: str) -> int:
    """Terminal cells, not characters: emoji and CJK take two."""
    return sum(0 if unicodedata.combining(ch) or ch in '\ufe0f\u200d' else
               2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(text))


def pad(text: str, width: int) -> str:
    return str(text) + ' ' * max(0, width - cells(text))


def icon_for(theme: dict, key: str, name: str, default: str = '') -> str:
    """theme[key] is {name: icon} or a list cycled by a stable hash of the name."""
    v = theme.get(key)
    if isinstance(v, dict):
        return v.get(name, default)
    if isinstance(v, list) and v:
        return v[sum(map(ord, name)) % len(v)]
    return default


def done_in_order(st: dict) -> list[dict]:
    """Finished units by completion time (shards interleave)."""
    return sorted(st['done'].values(), key=lambda e: e.get('ts') or 0)


def fmt_s(s: float | None) -> str:
    if s is None:
        return '—'
    if s < 10 and s != int(s):
        return f'{s:.1f}s'
    s = int(s)
    if s < 60:
        return f'{s}s'
    if s < 3600:
        return f'{s // 60}m {s % 60:02d}s'
    if s < 86400:
        return f'{s // 3600}h {(s % 3600) // 60:02d}m'
    return f'{s // 86400}d {(s % 86400) // 3600:02d}h'


def fmt_n(v: float | None) -> str:
    """Token counts: 950 · 12.3k · 4.56M."""
    if v is None:
        return '—'
    for div, suf in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(v) >= div:
            return f'{v / div:.3g}{suf}'
    return f'{v:.0f}'


def rate_window(m: dict, elapsed: float | None) -> float:
    """Trailing window for throughput and burn: the spec's, else min(elapsed, 24 h)."""
    return float(m.get('rate_window_s') or min(max(elapsed or 0, 60), 86400))


def throughput_eta(st: dict, now: float, window: float) -> float | None:
    """Remaining units / finish rate over the trailing window (concurrency-honest)."""
    remaining = st['n'] - st['k']
    if remaining <= 0:
        return 0.0 if st['n'] else None
    lo = now - window
    recent = [t for t in st['ends'] if t and t >= lo]
    if not recent:
        return None
    span = min(window, now - (st['t0'] or lo))
    return remaining / (len(recent) / span) if span > 0 else None


def theme_bits(theme: dict, frac_done: float, last: dict | None) -> tuple[str, str, str]:
    icon = theme.get('icon', '⚙')
    stages = theme.get('stages') or ['·', '·', '·', '·']
    i = min(len(stages) - 1, int(frac_done * len(stages)))
    stage = ' → '.join(f'{B}{s}{X}' if j == i else f'{D}{s}{X}' for j, s in enumerate(stages))
    thought = ''
    key = theme.get('thought_key')
    if last and key and key in (last.get('metrics') or {}):
        v = last['metrics'][key]
        for lo, text in sorted(theme.get('thoughts', []), key=lambda t: -t[0]):
            if v >= lo:
                thought = text
                break
    return icon, stage, thought


def ctx_of(m: dict, st: dict, rows: int = 30, cols: int = 100) -> dict:
    """Everything a block may want, computed once per repaint."""
    now = time.time()
    n, k = st['n'], st['k']
    frac = k / n if n else 0.0
    end = st['last_ts'] if st['ended'] else now
    elapsed = (end - st['t0']) if st['t0'] else st['elapsed_s']
    eta = st['eta_s']
    window = rate_window(m, elapsed)
    if eta is not None and st['current'] and st['current_since']:
        # the ETA was computed at the last run_end; subtract time spent since
        eta = max(0.0, eta - (now - st['current_since']))
    elif eta is None and st['started'] and not st['ended']:
        eta = throughput_eta(st, now, window)
    if st['ended']:
        eta = 0.0
    finish = '—' if eta is None else (datetime.now() + timedelta(seconds=eta)).strftime(
        '%H:%M' if eta < 20 * 3600 else '%a %d %b %H:%M')
    icon, stage, thought = theme_bits(m['theme'], frac, st['last'])
    return {'now': now, 'k': k, 'n': n, 'frac': frac, 'elapsed': elapsed, 'eta': eta,
            'finish': finish, 'icon': icon, 'stage': stage, 'thought': thought,
            'rows': rows, 'cols': cols, 'window': window, 'scan': log_scan(m.get('log'), m.get('failure_patterns', DEFAULT_FAILURES))}


# --- blocks: each takes (manifest, state, ctx) and returns the lines it contributes -----
def blk_title(m, st, cx):
    title = m.get('title', 'compute run')
    if not st['started']:
        return [f'{B}{cx["icon"]}  {title}{X}    {D}waiting for the first heartbeat 💤   '
                f'({", ".join(m["progress"])}){X}', '']
    state = (f'{G}FINISHED{X}' if st['ended'] else
             (f'{C}running{X}' if st['current'] else f'{Y}between runs{X}'))
    if st.get('shards'):
        state += f" {D}× {st['shards']} shards{X}"
    return [f'{B}{cx["icon"]}  {title}{X}   {state}    elapsed {fmt_s(cx["elapsed"])} · '
            f'ETA {B}{fmt_s(cx["eta"])}{X} · done ≈ {cx["finish"]}']


def blk_divider(m, st, cx):
    return [f'{D}{"─" * 96}{X}']


def blk_meter(m, st, cx):
    label = m['theme'].get('meter_label', 'progress')
    walls = [w for ws in st['walls'].values() for w in ws]
    per = f'~{fmt_s(sum(walls) / len(walls))}/unit' if walls else 'no unit finished yet'
    return [f'  {label:<13} {bar(cx["frac"], 32)} {cx["k"]}/{cx["n"]} units  '
            f'{cx["frac"] * 100:3.0f}%   {per}']


def blk_stage(m, st, cx):
    return [f'  {"stage":<13} {cx["stage"]}'] if st['started'] else []


def blk_groups(m, st, cx):
    gs = group_stats(st)
    if not gs:
        return []
    cells = []
    for g in gs:
        col = G if g['done'] == g['n'] else (C if g['done'] else D)
        cells.append(f'{g["group"]:>8} {bar(g["done"] / g["n"] if g["n"] else 0, 10, col)} '
                     f'{g["done"]}/{g["n"]}')
    return ['  ' + '    '.join(cells[i:i + 2]) for i in range(0, len(cells), 2)]


def blk_now(m, st, cx):
    if not st['current']:
        return []
    if len(st['live']) > 2:
        names = ', '.join(st['live'])
        return [f'  {"now":<13} {B}{len(st["live"])} running{X}  {D}{names[:cx["cols"] - 30]}{X}']
    g = unit_group(st['current'])
    typ = st['walls'].get(g) or [w for ws in st['walls'].values() for w in ws]
    typ_s = sum(typ) / len(typ) if typ else None
    since = cx['now'] - st['current_since'] if st['current_since'] else 0
    return [f'  {"now":<13} {B}{st["current"]:<18}{X} started {fmt_s(since)} ago'
            + (f'  (typ. {typ_s:.0f} s) {bar(since / typ_s, 12, C)}' if typ_s else '')]


def blk_last(m, st, cx):
    if not st['last']:
        return []
    mt = st['last'].get('metrics') or {}
    keys = m.get('metrics') or list(mt)[:6]
    shown = ' · '.join(f'{k} {mt[k]:.3g}' for k in keys if k in mt)
    return [f'  {"last":<13} {st["last"].get("unit", ""):<18} {shown}']


def blk_gpu(m, st, cx):
    """One line per device; a swarm's local agents (`gpu: "GPU<idx>"`) are named on their card."""
    on = {}
    for n, a in sorted(st['agents'].items()):
        if a.get('gpu') is not None:
            ico = _agent_icon(m, st, n)
            on.setdefault(str(a['gpu']), []).append(f'{ico} {n}' if ico else n)
    L = []
    for line in gpu_lines(m.get('_gpus')):
        label = line.split(' ', 1)[0]                     # 'GPU0'
        L.append('  ' + line + (f'  {D}← {", ".join(on[label])}{X}' if label in on else ''))
    return L


def blk_log(m, st, cx):
    scan = cx['scan']
    if scan['hits']:
        return [f'  {"log":<13} {R}{B}{scan["hits"]} failure signature(s){X}  last: '
                f'{R}{scan["last_hit"][:70]}{X}']
    tail = scan['lines'][-1][:70] if scan['lines'] else '(empty)'
    return [f'  {"log":<13} {G}✓{X} no failure signatures   {D}{tail}{X}']


def blk_stale(m, st, cx):
    if st['last_ts'] and not st['ended'] and \
            cx['now'] - st['last_ts'] > (m.get('stale_after_s') or 900):
        return [f'  {R}{B}STALE{X}  no heartbeat for {fmt_s(cx["now"] - st["last_ts"])}']
    return []


def blk_thought(m, st, cx):
    return [f'  {M}❝ {cx["thought"]} ❞{X}'] if cx['thought'] else []


def blk_footer(m, st, cx):
    return [f'{D}  {datetime.now().strftime("%H:%M:%S")} · repaint {m.get("every", 5)}s · '
            f'{Path(m.get("_path", "spec")).name}{X}']


def blk_group_table(m, st, cx):
    L = [f'{B}per-unit breakdown{X}  {D}{m.get("title", "")}{X}']
    if not st['started']:
        return L + [f'{D}no plan yet{X}']
    for g in group_stats(st, cx):
        eta = fmt_s(g['eta_s']) if g['done'] < g['n'] else 'done'
        mw = f'{fmt_s(g["mean_wall"])}/unit' if g['mean_wall'] else '—'
        L.append(f'  {g["group"]:>8}  {g["done"]:>2}/{g["n"]:<2}  {mw:>13}   ETA {eta}')
    return L + [f'{D}{"─" * 60}{X}']


def blk_units(m, st, cx):
    if not st['started']:
        return []
    L = []
    keys = m.get('metrics') or []
    units = st['units']
    live = set(st['current'].split(' + ')) if st['current'] else set()
    cur = next((i for i, u in enumerate(units) if u in live), None)
    if cur is None:
        cur = sum(1 for u in units if u in st['done'])
    body = max(5, (cx.get('rows') or 30) - len(group_stats(st)) - 4)
    lo, hi = max(0, cur - body // 3), 0
    hi = min(len(units), lo + body)
    lo = max(0, hi - body)
    if lo > 0:
        L.append(f'{D}  … {lo} earlier{X}')
    for u in units[lo:hi]:
        if u in st['done']:
            e, mt = st['done'][u], (st['done'][u].get('metrics') or {})
            vals = '  '.join(f'{k[:7]} {mt[k]:.3g}' for k in keys if k in mt)
            ok = (e.get('status') or 'ok') == 'ok'
            who = f'{e["agent"]:<10.10} ' if e.get('agent') else ''
            L.append(f'  {G + "✓" if ok else R + "✗"}{X} {u:<18} {e.get("wall_s") or 0:5.0f}s  '
                     f'{D}{who}{vals if ok else e.get("status")}{X}')
        elif u in live:
            L.append(f'  {C}▶{X} {B}{u:<18}{X} {C}running{X}')
        else:
            L.append(f'  {D}· {u:<18}{X}')
    if hi < len(units):
        L.append(f'{D}  … {len(units) - hi} more{X}')
    return L


def blk_charts(m, st, cx):
    """Sparkline history per metric.  Text fallback of the peek-tui `charts` pane."""
    done = done_in_order(st)
    if not done:
        return [f'  {D}charts: no unit finished yet{X}']
    series = [('wall s', [e.get('wall_s') for e in done])]
    series += [(k, [(e.get('metrics') or {}).get(k) for e in done])
               for k in (m.get('metrics') or list((done[-1].get('metrics') or {}))[:4])]
    width = max(8, (cx.get('cols') or 100) - 40)   # the sparkline takes what the pane leaves
    L = []
    for name, vs in series:
        xs = [v for v in vs if v is not None]
        if xs:
            f = fmt_s if name == 'wall s' else (lambda v: f'{v:.3g}')
            L.append(f'  {name[:12]:<13} {C}{spark(xs, width)}{X}  {D}{f(min(xs))}…{f(max(xs))}{X}'
                     f'  last {B}{f(xs[-1])}{X}')
    return L


# --- swarm blocks (SPEC §3b) -----------------------------------------------------------
STATE_ORDER = {'waiting_human': 0, 'blocked': 1, 'failed': 2, 'working': 3, 'idle': 4, 'done': 5}
STATE_COL = {'waiting_human': R + B, 'blocked': Y, 'failed': R, 'working': C, 'idle': D,
             'done': G}


def _fit(text: str, width: int) -> str:
    text = ' '.join(str(text).split())
    return text if len(text) <= width else text[:max(0, width - 1)] + '…'


def _to_human(r: dict) -> bool:
    return bool(r.get('needs_human')) or r.get('to') == 'human'


STATE_GLYPH = {'working': '▶', 'waiting_human': '!', 'blocked': '■', 'failed': '✗', 'idle': '·',
               'done': '✓'}
LANE_PALETTE = ['\033[38;5;39m', '\033[38;5;141m', '\033[38;5;43m', '\033[38;5;170m',
                '\033[38;5;112m', '\033[38;5;75m']   # working cells, one colour per stage
LANE_CELL = {'working': (C, '█'), 'waiting_human': (R, '█'), 'blocked': (Y, '▒'),
             'failed': (R, '╳'), 'idle': (D, '─'), 'done': (G, '░')}


def _agent_icon(m: dict, st: dict, name: str) -> str:
    """theme.agent_icons: {name: icon} or a list dealt out in name order (no collisions)."""
    v = m['theme'].get('agent_icons')
    if isinstance(v, list) and v:
        names = sorted(st['agents'])
        return v[names.index(name) % len(v)] if name in names else ''
    return icon_for(m['theme'], 'agent_icons', name)


def _where(a: dict) -> str:
    """'local GPU0' / 'cloud' from an agent's declared `gpu` / `host`."""
    if a.get('gpu') is not None:
        return f'⌂ {a["gpu"]}'
    return a.get('host') or ''


def blk_roster(m, st, cx):
    ag = st['agents']
    if not ag:
        return []
    th = m['theme']
    stale = m.get('agent_stale_after_s') or m.get('stale_after_s') or 900
    rows = sorted(ag.items(), key=lambda kv: (STATE_ORDER.get(kv[1]['state'], 9), kv[0]))
    tally = {}
    for _, a in rows:
        tally[a['state']] = tally.get(a['state'], 0) + 1
    L = [f'  {B}agents{X}  ' + ' · '.join(
        f'{STATE_COL.get(s, "")}{icon_for(th, "state_icons", s)}{n} {s.replace("_", " ")}{X}'
        for s, n in sorted(tally.items(), key=lambda t: STATE_ORDER.get(t[0], 9)))]
    cap = max(4, cx['rows'] // 2)
    icons = any(_agent_icon(m, st, n) for n in ag)
    who_w = max(cells(n) for n in ag) + (3 if icons else 0)
    mh = {n: ' @ '.join(x for x in (a.get('model'), a.get('harness')) if x) for n, a in ag.items()}
    mh_w = min(36, max(map(cells, mh.values())))
    wh_w = max(cells(_where(a)) for a in ag.values())
    st_w = max(cells(icon_for(th, 'state_icons', s) + ' ' + s.replace('_', ' ')) for s in tally)
    taskw = max(12, min(36, cx['cols'] - who_w - mh_w - wh_w - st_w - 44))
    for name, a in rows[:cap]:
        u = st['usage'].get(name, {})
        quiet = (not st['ended'] and a['state'] not in ('done', 'failed') and a['last_ts']
                 and cx['now'] - a['last_ts'] > stale)
        ico = _agent_icon(m, st, name)
        who = f'{ico} {name}' if icons else name
        sic = icon_for(th, 'state_icons', a['state'])
        state = (sic + ' ' if sic else '') + a['state'].replace('_', ' ')
        what = a['task'] or a['note'] or ''
        usd = u.get('usd') or 0
        L.append(f'  {Y + "?" if quiet else " "}{X} {B}{pad(who, who_w)}{X}  '
                 f'{D}{pad(_fit(mh[name], mh_w), mh_w)}  {pad(_where(a), wh_w)}{X}  '
                 f'{STATE_COL.get(a["state"], "")}{pad(state, st_w)}{X} '
                 f'{fmt_s(cx["now"] - a["since"]) if a["since"] else "—":>8}  '
                 f'{pad(_fit(what, taskw), taskw)}  {D}{fmt_n(u.get("tokens")):>6} tok '
                 f'{("$" + format(usd, ",.2f")) if usd else "local":>8}  ✓{a["done"]}{X}')
    if len(rows) > cap:
        L.append(f'{D}  … {len(rows) - cap} more agents{X}')
    return L


def blk_lanes(m, st, cx):
    """One swimlane per agent across the rate window, coloured by state (SPEC §3b)."""
    ag = st['agents']
    if not ag:
        return []
    th = m['theme']
    w = float(m.get('lanes_window_s') or max(cx['elapsed'] or 0, 60))   # default: the run
    icons = any(_agent_icon(m, st, n) for n in ag)
    who_w = max(cells(n) for n in ag) + (3 if icons else 0)
    width = max(10, cx['cols'] - who_w - 8)
    t1 = st['last_ts'] if st['ended'] else cx['now']
    t0 = t1 - w
    step = w / width
    L = [f'  {B}lanes{X}  {D}last {fmt_s(w)}  ' + '─' * max(0, width - 22 - len(fmt_s(w)))
         + f' now{X}']
    for name in sorted(ag):
        h = ag[name].get('hist') or [[ag[name]['since'], ag[name]['state'], None]]
        row, j, cur, grp, last = [], 0, None, None, None
        for i in range(width):
            t = t0 + (i + 0.5) * step
            while j < len(h) and (h[j][0] or 0) <= t:
                cur, grp = h[j][1], (h[j][2] if len(h[j]) > 2 else None)
                j += 1
            if cur is None:
                row.append(' ')
                continue
            col, glyph = LANE_CELL.get(cur, (D, '·'))
            if cur == 'working' and grp in st['groups']:
                col = LANE_PALETTE[st['groups'].index(grp) % len(LANE_PALETTE)]
            row.append((col if col != last else '') + glyph)
            last = col
        ico = _agent_icon(m, st, name)
        L.append(f'  {pad(f"{ico} {name}" if icons else name, who_w)}  {"".join(row)}{X}')
    used = {x[2] for a in ag.values() for x in a.get('hist', []) if len(x) > 2 and x[2]}
    stages = [g for g in st['groups'] if g in used]
    legend = ' '.join(f'{LANE_PALETTE[st["groups"].index(g) % len(LANE_PALETTE)]}██{X}{D} {g}{X}'
                      for g in stages) or f'{C}██{X}{D} working{X}'
    legend += '   ' + '  '.join(f'{LANE_CELL[s][0]}{LANE_CELL[s][1] * 2}{X}{D} '
                                f'{icon_for(th, "state_icons", s)}{s.replace("_", " ")}{X}'
                                for s in ('waiting_human', 'blocked', 'idle', 'done'))
    return L + [f'  {"":<{who_w}}  {legend}']


def blk_feed(m, st, cx):
    if not st['feed'] and not st['open']:
        return []
    th = m['theme']
    opn = sorted(st['open'].values(), key=lambda r: r['ts'] or 0)
    human = sum(1 for r in opn if _to_human(r))
    head = f'  {B}feed{X}  '
    head += (f'{R}{B}{human} waiting on you{X} · ' if human else '') + \
        (f'{Y}{len(opn)} open{X}' if opn else f'{D}nothing open{X}')
    L = [head]
    textw = max(20, cx['cols'] - 44)

    def line(r, mark):
        age = fmt_s(cx['now'] - r['ts']) if r['ts'] else '—'
        frm = r.get('from') or '?'
        ico = _agent_icon(m, st, frm) if frm in st['agents'] else ''
        who = (f'{ico} ' if ico else '') + frm + (f'→{r["to"]}' if r.get('to') else '')
        kic = icon_for(th, 'kind_icons', r.get('kind') or '')
        kind = (kic + ' ' if kic else '') + (r.get('kind') or '')
        return (f'  {mark} {D}{age:>7}{X} {pad(_fit(who, 20), 20)} {D}{pad(kind, 13)}{X} '
                f'{_fit(r.get("text", ""), textw)}')
    for r in opn:
        L.append(line(r, f'{R}!{X}' if _to_human(r) else f'{Y}?{X}'))
    keep = int(m.get('feed_keep') or FEED_KEEP)
    budget = max(3, cx['rows'] // 3)
    open_ids = set(st['open'])
    recent = [r for r in st['feed'][-keep:] if str(r.get('id')) not in open_ids][-budget:]
    for r in recent:
        L.append(line(r, f'{D}·{X}'))
    return L


def blk_cost(m, st, cx):
    us = st['usage']
    if not us:
        return []
    usd = sum(u.get('usd') or 0 for u in us.values())
    tok = sum(u.get('tokens') or 0 for u in us.values())
    local = [n for n, a in st['agents'].items() if a.get('gpu') is not None and n in us]
    ltok = sum(us[n].get('tokens') or 0 for n in local)
    budget = m.get('budget_usd') or st['budget_usd']
    L = [f'  {"cost":<13} {B}${usd:,.2f}{X} · {fmt_n(tok)} tokens'
         + (f' {D}({fmt_n(ltok)} local, $0){X}' if local else '') + f' · {len(us)} agents']
    if budget:
        f = usd / budget
        L[0] += f'   {bar(min(f, 1.0), 20, R if f > 0.9 else Y if f > 0.7 else G)} of ${budget:,.0f}'
    bits = []
    sp, w = st['spend'], cx['window']
    base = next((pt for pt in sp if pt[0] >= cx['now'] - w), None)
    if base and sp[-1][0] > base[0]:
        prev = [pt for pt in sp if pt[0] < base[0]]
        t0, v0 = (prev[-1] if prev else base)
        bits.append(f'burn ${(sp[-1][1] - v0) / max(cx["now"] - t0, 1) * 3600:,.2f}/h '
                    f'{D}(last {fmt_s(min(w, cx["now"] - t0))}){X}')
    if st['k'] and st['n'] > st['k'] and not st['ended']:
        proj = usd + usd / st['k'] * (st['n'] - st['k'])
        warn = f' {R}{B}over budget{X}' if budget and proj > budget else ''
        bits.append(f'projected ${proj:,.2f} at finish{warn}')
    if bits:
        L.append(f'  {"":<13} ' + ' · '.join(bits))
    if len(sp) > 2 and st['t0']:
        # $ per slice of the run, as a sparkline: when did the money go?
        width = max(10, min(60, cx['cols'] - 40))
        t0, t1 = sp[0][0], sp[-1][0]
        if t1 > t0:
            edges = [t0 + (t1 - t0) * i / width for i in range(width + 1)]
            vals, j, prev = [], 0, sp[0][1]
            for e in edges[1:]:
                while j < len(sp) and sp[j][0] <= e:
                    j += 1
                cur = sp[j - 1][1] if j else sp[0][1]
                vals.append(max(0.0, cur - prev))
                prev = cur
            L.append(f'  {"":<13} {Y}{spark(vals, width, lo=0)}{X}  {D}$ over the run{X}')
    top = sorted(us.items(), key=lambda kv: -(kv[1].get('usd') or 0))[:3]
    if usd and len(us) > 1:
        L.append(f'  {"":<13} {D}top: ' + '  '.join(
            f'{n} {100 * (u.get("usd") or 0) / usd:.0f}%' for n, u in top if u.get('usd')) + X)
    return L


_CPU_PREV: dict = {}


def cpu_sample() -> dict | None:
    """Host CPU from /proc (Linux): per-core busy %, load, memory.  None elsewhere.
    $PEEK_CPU_SAMPLE names a JSON file of the same shape (screenshots, tests)."""
    fake = os.environ.get('PEEK_CPU_SAMPLE')
    if fake:
        return json.loads(Path(fake).read_text())
    try:
        lines = Path('/proc/stat').read_text().splitlines()
    except OSError:
        return None
    now = {}
    for ln in lines:
        if ln.startswith('cpu'):
            f = ln.split()
            v = list(map(int, f[1:8]))
            now[f[0]] = (sum(v), v[3] + v[4])            # total, idle+iowait
    if not _CPU_PREV:                                    # one-shot reader: take a short delta
        _CPU_PREV.update(now)
        time.sleep(0.15)
        return cpu_sample()
    busy = {}
    for k, (tot, idle) in now.items():
        pt, pi = _CPU_PREV.get(k, (tot, idle))
        busy[k] = 100.0 * (1 - (idle - pi) / (tot - pt)) if tot > pt else 0.0
    _CPU_PREV.clear()
    _CPU_PREV.update(now)
    mem = {}
    try:
        for ln in Path('/proc/meminfo').read_text().splitlines():
            k, v = ln.split(':', 1)
            mem[k] = int(v.split()[0]) / 1048576           # GiB
    except OSError:
        pass
    cores = [busy[k] for k in sorted((k for k in busy if k != 'cpu'), key=lambda k: int(k[3:]))]
    return {'all': busy.get('cpu', 0.0), 'cores': cores, 'load': os.getloadavg(),
            'mem_used': mem.get('MemTotal', 0) - mem.get('MemAvailable', 0),
            'mem_total': mem.get('MemTotal', 0)}


def blk_cpu(m, st, cx):
    c = cpu_sample()
    if c is None:
        la = os.getloadavg()
        return [f'  {"cpu":<13} load {la[0]:.1f} {la[1]:.1f} {la[2]:.1f}']
    n = len(c['cores'])
    col = R if c['all'] > 90 else Y if c['all'] > 70 else G
    L = [f'  {"cpu":<13} {bar(c["all"] / 100, 20, col)} {c["all"]:3.0f} % of {n} cores · '
         f'load {c["load"][0]:.1f} · mem {c["mem_used"]:.0f}/{c["mem_total"]:.0f} GB']
    # one cell per core, nine levels: a heat strip of the whole socket
    width = max(10, cx['cols'] - 17)
    strip = ''.join(V_LEVELS[max(1, min(8, int(round(v / 100 * 8))))] for v in c['cores'][:width])
    L.append(f'  {"":<13} {col}{strip}{X}')
    return L


def task_states(st: dict) -> dict:
    """unit -> done | failed | running | ready | blocked | stuck (behind a failed dep)."""
    out = {}
    for u in st['units']:
        e = st['done'].get(u)
        if e is not None:
            out[u] = 'done' if (e.get('status') or 'ok') == 'ok' else 'failed'
        elif u in st['live']:
            out[u] = 'running'
    for u in st['units']:
        if u in out:
            continue
        ds = st['deps'].get(u) or []
        if any(out.get(d) == 'failed' for d in ds):
            out[u] = 'stuck'
        elif all(out.get(d) == 'done' for d in ds):
            out[u] = 'ready'
        else:
            out[u] = 'blocked'
    return out


def blk_deps(m, st, cx):
    if not st['started'] or not st['units']:
        return []
    ts = task_states(st)
    n = {k: 0 for k in ('done', 'running', 'ready', 'blocked', 'stuck', 'failed')}
    for v in ts.values():
        n[v] += 1
    L = [f'  {B}tasks{X}  {G}✓ {n["done"]}{X}  {C}▶ {n["running"]}{X}  ◇ {n["ready"]} ready  '
         f'{D}⧗ {n["blocked"]} blocked{X}' + (f'  {R}✗ {n["failed"]} failed · '
                                             f'{n["stuck"]} stuck behind them{X}'
                                             if n['failed'] else '')]
    w = cx['cols'] - 16
    ready = [u for u, v in ts.items() if v == 'ready']
    if ready:
        L.append(f'  {"ready":<13} {_fit(", ".join(ready), w)}')
    running = {u for u, v in ts.items() if v == 'running'}
    nxt = [u for u, v in ts.items() if v == 'blocked' and all(
        ts.get(d) in ('done', 'running') for d in st['deps'].get(u) or [])]
    if running and nxt:
        L.append(f'  {"unlocks next":<13} {D}{_fit(", ".join(nxt), w)}{X}')
    for u in [u for u, v in ts.items() if v == 'stuck'][:3]:
        bad = [d for d in st['deps'].get(u) or [] if ts.get(d) == 'failed']
        L.append(f'  {R}{"stuck":<13}{X} {u}  {D}← {", ".join(bad)}{X}')
    return L


BLOCKS = {'title': blk_title, 'divider': blk_divider, 'meter': blk_meter, 'stage': blk_stage,
          'groups': blk_groups, 'now': blk_now, 'last': blk_last, 'gpu': blk_gpu,
          'log': blk_log, 'stale': blk_stale, 'thought': blk_thought, 'footer': blk_footer,
          'group_table': blk_group_table, 'units': blk_units, 'charts': blk_charts,
          'roster': blk_roster, 'feed': blk_feed, 'cost': blk_cost, 'deps': blk_deps,
          'lanes': blk_lanes, 'cpu': blk_cpu}


def render_pane(m: dict, name: str, rows: int = 30, cols: int = 100) -> str:
    pane = pane_by_name(m, name)
    blocks = pane.get('blocks') or []
    if 'raw' in blocks:
        raise SystemExit(f'pane {name!r} is a raw pane (the command\'s own output); read it '
                         f'with your mux (`{Herdr().read_hint()}` / `{Tmux().read_hint()}`, '
                         f'ids in {Path(m["_state"]).name})')
    st = read_progress(m['progress'], m.get('_fold'))
    cx = ctx_of(m, st, rows, cols)
    out = []
    for b in blocks:
        fn = BLOCKS.get(b)
        if fn is None:
            raise SystemExit(f'unknown block {b!r} in pane {name!r}; known blocks: '
                             f'{", ".join(sorted(BLOCKS))}')
        out.extend(fn(m, st, cx))
    return '\n'.join(out)


# --- v1 API: render_pane supersedes these, but the first consumer's tests import them --
def render_show(m: dict) -> str:
    """The default window's first renderer pane."""
    return render_pane(m, render_panes(m)[0]['name'])


def render_runs(m: dict, rows: int = 30) -> str:
    """The default window's second renderer pane (falls back to the first)."""
    rps = render_panes(m)
    return render_pane(m, (rps[1] if len(rps) > 1 else rps[0])['name'], rows)


def loop(fn, every: float, once: bool) -> int:
    while True:
        out = fn()
        if once:
            print(out)
            return 0
        sys.stdout.write(CLEAR + out + '\n')
        sys.stdout.flush()
        time.sleep(every)


# ------------------------------------------------------------------ multiplexers
# `launch` drives a terminal multiplexer through one small interface.  A "workspace" is
# a herdr workspace or a tmux session; pane ids are the mux's own stable ids (herdr
# `wXX:pN`, tmux `%N`), cached in M.state.json together with the mux name.  Neither
# backend ever starts a mux server: a server started from inside a script or an agent
# session inherits that process's environment for every pane it later opens.
SHELLS = ('bash', 'zsh', 'sh', 'fish', 'dash', 'ksh')


class Herdr:
    name = 'herdr'
    start_hint = 'start it with `herdr` in a terminal, then re-run'

    def _call(self, *args) -> dict:
        r = subprocess.run(['herdr', *args], capture_output=True, text=True, timeout=30)
        try:
            return json.loads(r.stdout)
        except ValueError:
            return {'error': {'message': (r.stdout + r.stderr).strip()}}

    def _ok(self, d: dict) -> dict:
        if 'error' in d:
            raise SystemExit(f'herdr: {d["error"].get("message", d["error"])}')
        return d['result']

    def available(self) -> bool:
        if not shutil.which('herdr'):
            return False
        r = subprocess.run(['herdr', 'status', 'server', '--json'], capture_output=True,
                           text=True, timeout=10)
        try:
            return bool(json.loads(r.stdout).get('running'))
        except ValueError:
            return False

    def find_workspace(self, label: str) -> str | None:
        for w in self._call('workspace', 'list').get('result', {}).get('workspaces', []):
            if w.get('label') == label:
                return w['workspace_id']
        return None

    def create_workspace(self, label: str, cwd: str) -> tuple[str, str]:
        r = self._ok(self._call('workspace', 'create', '--cwd', cwd, '--label', label,
                                '--no-focus'))
        return r['workspace']['workspace_id'], r['root_pane']['pane_id']

    def first_pane(self, ws: str) -> str | None:
        ps = [p['pane_id'] for p in self._call('pane', 'list').get('result', {}).get('panes', [])
              if p['workspace_id'] == ws]
        return sorted(ps)[0] if ps else None

    def exists(self, pid: str) -> bool:
        return 'error' not in self._call('pane', 'get', pid)

    def idle(self, pid: str) -> bool:
        """True when the pane's shell itself holds the foreground — i.e. no job is running.

        Not "all foreground processes are shells": an idle prompt still runs shell rc
        hooks (`lesspipe`, `which`) as extra foreground processes, which made `launch`
        report a fresh pane as busy and skip starting its readout.  The foreground
        process *group* leader is the shell only when nothing is running.
        """
        pi = self._call('pane', 'process-info', '--pane', pid).get('result', {}) \
            .get('process_info', {})
        fg, shell = pi.get('foreground_process_group_id'), pi.get('shell_pid')
        if fg is not None and shell is not None:
            return fg == shell
        return all(p.get('name') in SHELLS for p in pi.get('foreground_processes', []))

    def split(self, pid: str, direction: str, ratio: float, cwd: str) -> str:
        """`ratio` is the share the existing pane keeps (herdr's own meaning)."""
        return self._ok(self._call('pane', 'split', pid, '--direction', direction,
                                   '--ratio', str(ratio), '--cwd', cwd))['pane']['pane_id']

    def rename(self, pid: str, title: str) -> None:
        self._call('pane', 'rename', pid, title)

    def run(self, pid: str, cmd: str) -> None:
        self._call('pane', 'run', pid, cmd)

    def focus_hint(self, ws: str, label: str) -> str:
        return f'herdr workspace focus {ws}'

    def read_hint(self) -> str:
        return 'herdr pane read <id>'


class Tmux:
    """tmux ≥ 3.1 (percent `split-window -l`).  `PEEK_TMUX_SOCKET` selects `tmux -L <name>`."""
    name = 'tmux'
    start_hint = 'start one with `tmux new -s main` in a terminal (detach: C-b d), then re-run'

    def __init__(self):
        sock = os.environ.get('PEEK_TMUX_SOCKET')
        self.base = ['tmux', *(['-L', sock] if sock else [])]

    def _t(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run([*self.base, *args], capture_output=True, text=True, timeout=30)

    def _out(self, *args) -> str:
        r = self._t(*args)
        if r.returncode:
            raise SystemExit(f'tmux {args[0]}: {r.stderr.strip()}')
        return r.stdout.strip()

    @staticmethod
    def session_name(label: str) -> str:
        return re.sub(r'[:.]', '_', label)   # tmux rewrites these itself; match it

    def available(self) -> bool:
        # list-sessions talks to a running server and never starts one
        return bool(shutil.which('tmux')) and self._t('list-sessions').returncode == 0

    def find_workspace(self, label: str) -> str | None:
        for line in self._t('list-sessions', '-F', '#{session_id}\t#{session_name}') \
                .stdout.splitlines():
            sid, _, name = line.partition('\t')
            if name == self.session_name(label):
                return sid
        return None

    def create_workspace(self, label: str, cwd: str) -> tuple[str, str]:
        sid, pid = self._out('new-session', '-d', '-s', self.session_name(label), '-c', cwd,
                             '-P', '-F', '#{session_id} #{pane_id}').split()
        return sid, pid

    def first_pane(self, ws: str) -> str | None:
        ps = self._t('list-panes', '-s', '-t', ws, '-F', '#{pane_id}').stdout.split()
        return sorted(ps, key=lambda p: int(p.lstrip('%')))[0] if ps else None

    def exists(self, pid: str) -> bool:
        r = self._t('display-message', '-p', '-t', pid, '#{pane_id}')
        return r.returncode == 0 and r.stdout.strip() == pid

    def idle(self, pid: str) -> bool:
        """Same test as herdr: the shell leads the terminal's foreground process group."""
        r = self._t('display-message', '-p', '-t', pid, '#{pane_pid} #{pane_current_command}')
        shell, _, cmd = r.stdout.strip().partition(' ')
        tpgid = subprocess.run(['ps', '-o', 'tpgid=', '-p', shell], capture_output=True,
                               text=True).stdout.strip()
        if tpgid.lstrip('-').isdigit() and shell.isdigit():
            return int(tpgid) == int(shell)
        return cmd in SHELLS

    def split(self, pid: str, direction: str, ratio: float, cwd: str) -> str:
        new = round((1 - ratio) * 100)          # tmux sizes the NEW pane
        return self._out('split-window', '-d', '-t', pid, '-h' if direction == 'right' else '-v',
                         '-l', f'{max(1, min(99, new))}%', '-c', cwd, '-P', '-F', '#{pane_id}')

    def rename(self, pid: str, title: str) -> None:
        self._t('select-pane', '-t', pid, '-T', title)

    def run(self, pid: str, cmd: str) -> None:
        self._t('send-keys', '-t', pid, '-l', cmd)
        self._t('send-keys', '-t', pid, 'Enter')

    def focus_hint(self, ws: str, label: str) -> str:
        s = ' '.join(self.base[1:] + ['']).lstrip()
        name = shlex.quote(self.session_name(label))
        return f'tmux {s}attach -t {name}  (inside tmux: tmux {s}switch-client -t {name})'

    def read_hint(self) -> str:
        return 'tmux capture-pane -p -t <id>'


MUXES = {'herdr': Herdr, 'tmux': Tmux}


def pick_mux(m: dict, want: str | None = None):
    """--mux > $PEEK_MUX > spec `mux` > auto (inside tmux → tmux, else a running herdr,
    else a running tmux server)."""
    want = want or os.environ.get('PEEK_MUX') or m.get('mux') or 'auto'
    if want != 'auto':
        if want not in MUXES:
            raise SystemExit(f'unknown mux {want!r}; known: {", ".join(MUXES)}, auto')
        mx = MUXES[want]()
        if not mx.available():
            raise SystemExit(f'no running {want} server — {mx.start_hint}')
        return mx
    order = (['tmux', 'herdr'] if os.environ.get('TMUX') else ['herdr', 'tmux'])
    for name in order:
        mx = MUXES[name]()
        if mx.available():
            return mx
    raise SystemExit('no running terminal multiplexer found (herdr or tmux). '
                     'Start one in a terminal — `tmux new -s main` or `herdr` — then re-run; '
                     'compute-peek never starts a mux server itself.')


def _split_dir(i: int) -> str:
    return 'right' if i == 2 else 'down'


def _split_ratio(i: int) -> float:
    return 0.42 if i == 1 else 0.5


def cmd_launch(a) -> int:
    m = load_manifest(a.manifest)
    mx = pick_mux(m, getattr(a, 'mux', None))
    label, cwd = m.get('workspace', 'vitals'), m['cwd']
    ps = panes_of(m)
    names = [p.get('name') for p in ps]
    if any(n is None for n in names) or len(set(names)) != len(names):
        raise SystemExit('every pane needs a distinct "name"')
    state = json.loads(Path(m['_state']).read_text()) if Path(m['_state']).exists() else {}
    if state and state.get('mux', 'herdr') != mx.name:   # pre-mux state files were herdr's
        print(f'state was recorded under {state.get("mux", "herdr")}; now {mx.name} — '
              f'cached pane ids dropped')
        state = {}
    ws = mx.find_workspace(label)
    created = False
    if ws is None:
        ws, root = mx.create_workspace(label, cwd)
        state = {'workspace': ws, 'panes': {ps[0]['name']: root}}
        created = True
        print(f'created {mx.name} workspace {ws} ({label})')
    panes = state.get('panes', {}) if state.get('workspace') == ws else {}
    if not panes.get(ps[0]['name']) or not mx.exists(panes[ps[0]['name']]):
        root = mx.first_pane(ws)   # adopt the workspace's first pane as the root pane
        if root is None:
            raise SystemExit(f'workspace {ws} has no panes')
        panes[ps[0]['name']] = root
    for i, p in enumerate(ps[1:], start=1):  # pane i splits off pane i-1
        name = p['name']
        if panes.get(name) and mx.exists(panes[name]):
            continue
        panes[name] = mx.split(panes[ps[i - 1]['name']], p.get('dir', _split_dir(i)),
                               p.get('ratio', _split_ratio(i)), cwd)
    for name in names:
        mx.rename(panes[name], f'{label}:{name}')

    me, mf, py = (shlex.quote(str(Path(__file__).resolve())), shlex.quote(m['_path']),
                  shlex.quote(sys.executable))
    for p in render_panes(m):
        pid = panes[p['name']]
        if mx.idle(pid):
            run = (f'{shlex.quote(str(TUI_BIN))} --manifest {mf} --peek {me} --python {py}'
                   if is_tui_pane(p) else
                   f'{py} -u {me} pane --manifest {mf} --name {p["name"]}')
            mx.run(pid, f'cd {shlex.quote(cwd)} && {run}')
            print(f'{p["name"]:<8} {pid}  started{" (peek-tui)" if is_tui_pane(p) else ""}')
        else:
            print(f'{p["name"]:<8} {pid}  already running')

    # `command` may be a list — one process per shard — each in its own raw pane; the
    # declared raw panes are used in order and extra ones are split downward.
    cmds = m.get('command')
    cmds = cmds if isinstance(cmds, list) else ([cmds] if cmds else [])
    raw_names = [p['name'] for p in ps if 'raw' in (p.get('blocks') or [])]
    raw = [(n, panes[n]) for n in raw_names]
    if cmds and not raw:
        raise SystemExit('spec has a "command" but no pane with blocks ["raw"] to run it in')
    while len(raw) < len(cmds):
        name = f'raw{len(raw)}'
        if not panes.get(name) or not mx.exists(panes[name]):
            panes[name] = mx.split(raw[-1][1], 'down', 0.5, cwd)
            mx.rename(panes[name], f'{label}:{name}')
        raw.append((name, panes[name]))
    Path(m['_state']).write_text(json.dumps({'mux': mx.name, 'workspace': ws, 'panes': panes},
                                            indent=2) + '\n')

    if a.start:
        if not cmds:
            raise SystemExit('spec has no "command"')
        logs = [l for l in (m.get('log') or []) if not any(c in l for c in '*?[')]
        st = read_progress(m['progress'], m.get('_fold'))
        live = st['started'] and not st['ended'] and st['last_ts'] and \
            time.time() - st['last_ts'] < (m.get('stale_after_s') or 900)
        for i, (cmd, (name, pid)) in enumerate(zip(cmds, raw)):
            if not mx.idle(pid):
                print(f'{name:<8} {pid}  BUSY — not starting (a process is in the foreground)')
            elif live:
                print(f'{name:<8} {pid}  a run is live per progress — not starting')
            else:
                log = logs[i] if i < len(logs) else None
                if log:
                    Path(log).parent.mkdir(parents=True, exist_ok=True)
                full = cmd + (f' 2>&1 | tee -a {shlex.quote(log)}' if log else '')
                mx.run(pid, f'cd {shlex.quote(cwd)} && {full}')
                print(f'{name:<8} {pid}  STARTED: {cmd[:80]}')
    else:
        for name, pid in raw:
            print(f'{name:<8} {pid}  {"idle" if mx.idle(pid) else "busy"}')
    print(f'{mx.name} workspace {ws} ({label}){" [new]" if created else ""} — '
          f'`{mx.focus_hint(ws, label)}`')
    return 0


# ------------------------------------------------------------------ spec library
def cmd_init(a) -> int:
    p = Path(a.manifest)
    if p.exists():
        raise SystemExit(f'{p} exists; not overwriting')
    src = load_manifest(a.like) if a.like else {}
    swarm = a.kind == 'swarm'
    spec = {
        'title': a.title or ('my agent swarm' if swarm else 'my compute run'),
        'workspace': a.workspace or src.get('workspace') or 'peek',
        'cwd': os.getcwd(),
        'panes': src.get('panes') or [dict(x) for x in (SWARM_PANES if swarm else DEFAULT_PANES)],
        'progress': 'logs/<run>/progress.jsonl',   # never inherited: clobbering risk
        'log': 'logs/<run>/run.log',
        'command': ('tail -F logs/<run>/orchestrator.log' if swarm else
                    'python3 -u scripts/my_run.py --progress logs/<run>/progress.jsonl'),
        'gpu': None,                                # per-run provision; use a UUID from `gpus`
        'every': src.get('every', 5),
        'stale_after_s': src.get('stale_after_s', 4 * 3600 if swarm else 900),
        'metrics': src.get('metrics', []),
        'failure_patterns': src.get('failure_patterns', DEFAULT_FAILURES),
        'theme': src.get('theme') or (
            {'icon': '🐝', 'meter_label': 'hive', 'stages': ['scouting', 'building', 'buzzing',
                                                            'honey'],
             'thought_key': None, 'thoughts': []} if swarm else
            {'icon': '⚙', 'meter_label': 'progress', 'stages': ['◔', '◑', '◕', '●'],
             'thought_key': None, 'thoughts': []}),
    }
    if swarm:   # SPEC §2b knobs; null = default
        spec.update({k: src.get(k) for k in ('rate_window_s', 'budget_usd',
                                             'agent_stale_after_s', 'feed_keep')})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'wrote {p}'
          + (f' (shape from {a.like})' if a.like else '')
          + '\n  edit title/command/progress/log/metrics/theme, set "gpu" to a UUID '
            '(see: compute-peek.py gpus), then:\n'
            f'  compute-peek.py launch --manifest {p} --start')
    return 0


SPEC_GLOBS = ['.peek/*.json', 'scripts/vitals_*.json']


def _is_spec(f: Path) -> bool:
    return not f.name.endswith(('.state.json', '.fold.json'))   # launcher files, not specs


def cmd_specs(a) -> int:
    root = Path(a.project or os.getcwd())
    found = [f for g in SPEC_GLOBS for f in sorted(root.glob(g)) if _is_spec(f)]
    print(f'peek-specs under {root}  {D}(reuse a shape with `init --like <file>`){X}')
    if not found:
        print(f'  {D}none — this project has not created a peek-spec yet{X}')
        return 0
    for f in found:
        try:
            m = json.loads(f.read_text())
        except ValueError as e:
            print(f'  {f.relative_to(root)}   {R}unreadable: {e}{X}')
            continue
        print(f'  {f.relative_to(root)}')
        print(f'      {m.get("title", "")}')
        shape = '  '.join(f'{p.get("name")}[{"+".join(p.get("blocks") or [])}]'
                          for p in (m.get('panes') or DEFAULT_PANES))
        print(f'      {D}{shape}{X}')
    return 0


def cmd_gpus(a) -> int:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f'nvidia-smi unavailable: {e}')
    print(f'{B}device UUIDs — paste the UUID into a spec\'s "gpu"{X} '
          f'{D}(never the index: it means different cards per CUDA_DEVICE_ORDER){X}')
    for line in out.splitlines():
        idx, uuid, name, mt = [s.strip() for s in line.split(',')]
        print(f'  idx {idx}  {B}{uuid}{X}  {name}  ({float(mt) / 1024:.0f} GB)')
    return 0


# ------------------------------------------------------------------ agent readout
def cmd_status(a) -> int:
    m = load_manifest(a.manifest)
    st = read_progress(m['progress'], m.get('_fold'))
    cx = ctx_of(m, st)
    scan = cx['scan']
    print(f'{m.get("title")}: {"not started" if not st["started"] else "finished" if st["ended"] else "running"}; '
          f'{st["k"]}/{st["n"]} done; ETA {fmt_s(cx["eta"])} (≈ {cx["finish"]}); '
          f'current {st["current"] if len(st["live"]) <= 2 else str(len(st["live"])) + " units"}; '
          f'failures {scan["hits"]}')
    if st['open']:
        for r in sorted(st['open'].values(), key=lambda r: r['ts'] or 0):
            print(f'  OPEN{" (needs human)" if _to_human(r) else ""} {r.get("kind")} '
                  f'from {r.get("from")}: {_fit(r.get("text", ""), 120)}')
    if st['agents']:
        print('  agents: ' + ', '.join(f'{n} {a["state"]}' + (f' ({a["task"]})' if a['task'] else '')
                                       for n, a in sorted(st['agents'].items())))
    if st['usage']:
        print(f'  spend: ${sum(u.get("usd") or 0 for u in st["usage"].values()):,.2f}, '
              f'{fmt_n(sum(u.get("tokens") or 0 for u in st["usage"].values()))} tokens')
    for g in group_stats(st, cx):
        print(f'  {g["group"]:>8} {g["done"]}/{g["n"]}  '
              f'ETA {fmt_s(g["eta_s"]) if g["done"] < g["n"] else "done"}')
    if st['last']:
        print('  last:', st['last'].get('unit'), json.dumps(st['last'].get('metrics')))
    for l in gpu_lines(m.get('_gpus')):
        print('  ' + l)
    if scan['last_hit']:
        print('  LAST FAILURE:', scan['last_hit'])
    return 0


def state_json(m: dict) -> dict:
    """Everything peek-tui draws, folded here so there is one reader of the JSONL."""
    st = read_progress(m['progress'], m.get('_fold'))
    cx = ctx_of(m, st)
    th = m['theme']
    stages = th.get('stages') or []
    cur = (st['current'] or '').split(' + ')[0]
    typ = st['walls'].get(unit_group(cur)) if cur else None
    typ = typ or [w for ws in st['walls'].values() for w in ws]
    stale = bool(st['last_ts'] and not st['ended'] and
                 cx['now'] - st['last_ts'] > (m.get('stale_after_s') or 900))
    return {'title': m.get('title', 'compute run'), 'icon': cx['icon'],
            'meter_label': th.get('meter_label', 'progress'), 'thought': cx['thought'],
            'started': st['started'], 'ended': st['ended'], 'current': st['current'],
            'shards': st['shards'], 'k': cx['k'], 'n': cx['n'], 'frac': cx['frac'],
            'elapsed_s': cx['elapsed'], 'eta_s': cx['eta'], 'finish': cx['finish'],
            'stale': stale, 'every': m.get('every', 5),
            'metrics': m.get('metrics') or [], 'groups': group_stats(st, cx),
            'units': st['units'], 'stages': stages,
            'stage_i': min(len(stages) - 1, int(cx['frac'] * len(stages))) if stages else None,
            'now_s': cx['now'] - st['current_since'] if st['current'] and st['current_since'] else None,
            'typ_s': sum(typ) / len(typ) if cur and typ else None,
            'done': [{'unit': e.get('unit'), 'group': e.get('group'),
                      'wall_s': e.get('wall_s'), 'ts': e.get('ts'),
                      'status': e.get('status') or 'ok', 'agent': e.get('agent'),
                      'metrics': e.get('metrics') or {}} for e in done_in_order(st)],
            'gpus': [gpu_query(d) for d in m.get('_gpus') or []],
            'log': {'hits': cx['scan']['hits'], 'last_hit': cx['scan']['last_hit']},
            'agents': st['agents'], 'open': list(st['open'].values()),
            'usage': st['usage'], 'tasks': task_states(st) if st['deps'] else {}}


def cmd_state(a) -> int:
    print(json.dumps(state_json(load_manifest(a.manifest))))
    return 0


# ------------------------------------------------------------------ peek-tui (ratatui)
TUI_DIR = Path(__file__).resolve().parent / 'peek-tui'
TUI_BIN = TUI_DIR / 'target' / 'release' / 'peek-tui'


def is_tui_pane(p: dict) -> bool:
    """A pane whose only block is `charts` is drawn by peek-tui when it is built."""
    return (p.get('blocks') or []) == ['charts'] and TUI_BIN.exists()


def cmd_build_tui(a) -> int:
    return subprocess.run(['cargo', 'build', '--release', '--quiet'], cwd=TUI_DIR).returncode


# ------------------------------------------------------------------ demo
DEMO_GROUPS = {'amp': 1.0, 'freq': 1.4, 'ctl': 0.5, 'noise': 0.8}   # group -> relative cost


def cmd_demo_feed(a) -> int:
    """Emit a synthetic sweep to --progress: the SPEC §2 contract, at demo speed, forever."""
    import math
    import random
    rnd = random.Random(a.seed)
    prog = Path(a.progress)
    prog.parent.mkdir(parents=True, exist_ok=True)
    units = [f'{g}={v} s{sd}' for g in DEMO_GROUPS for v in range(4) for sd in range(3)]
    sweep = 0
    while True:
        sweep += 1
        t0 = time.time()
        walls = []
        with prog.open('w') as fh:
            def emit(**e):
                fh.write(json.dumps({'ts': time.time(), **e}) + '\n')
                fh.flush()
            emit(event='sweep_start', n=len(units), groups=list(DEMO_GROUPS), units=units,
                 meta={'sweep': sweep}, k=0, elapsed_s=0.0, eta_s=None)
            drift = rnd.random()
            for k, u in enumerate(units, 1):
                g = unit_group(u)
                emit(event='run_start', group=g, unit=u, k=k - 1, n=len(units),
                     elapsed_s=time.time() - t0, eta_s=None)
                wall = a.every * DEMO_GROUPS[g] * rnd.uniform(0.7, 1.3)
                time.sleep(wall)
                walls.append(wall)
                eta = sum(walls) / len(walls) * (len(units) - k)
                x = k / len(units)
                metrics = {'active_frac': 0.45 + 0.35 * math.sin(6 * x + drift * 3) + rnd.gauss(0, 0.04),
                           'mean_hz': 2 + 9 * x + rnd.gauss(0, 0.5),
                           'loss': 1.8 * math.exp(-3 * x) + rnd.uniform(0, 0.08),
                           'entropy': 2 + math.sin(11 * x) * 0.6 + rnd.gauss(0, 0.1)}
                emit(event='run_end', group=g, unit=u, wall_s=wall, k=k, n=len(units),
                     elapsed_s=time.time() - t0, eta_s=eta,
                     metrics={n: round(v, 4) for n, v in metrics.items()})
                print(f'[sweep {sweep}] {k}/{len(units)} done, ETA {fmt_s(eta)}  {u:<12} '
                      f'loss {metrics["loss"]:.3f}', flush=True)
            emit(event='sweep_end', out='(demo: nothing written)', k=len(units), n=len(units),
                 elapsed_s=time.time() - t0, eta_s=0.0)
        if not a.loop:
            return 0
        print(f'[sweep {sweep}] finished — next sweep in {a.pause:.0f}s', flush=True)
        time.sleep(a.pause)


# The swarm demo is a decompilation campaign: every function of a binary flows
# triage → lift → retype → name → match (recompiles byte-identical), worked by a mix of
# cloud and local models in different harnesses.  Local models sit on GPUs; `match`
# is CPU work (recompile + diff).  Names, prices and versions are illustrative.
DEMO_AGENTS = [   # name, model, harness, gpu label (local) or None (cloud), $ per M tokens
    ('opus', 'Claude Opus 5.5', 'claude-code', None, 15.0),
    ('sonnet', 'Claude Sonnet 5.5', 'claude-code', None, 5.0),
    ('gpt', 'GPT-6.1', 'codex', None, 6.0),
    ('deepseek', 'DeepSeek 4 Pro', 'hermes', None, 1.1),
    ('glm', 'GLM 5.3', 'opencode', None, 1.5),
    ('qwen', 'Qwen 3.8 27B', 'pi', 'GPU0', 0.0),
    ('gemma', 'Gemma 4 31B', 'omp', 'GPU1', 0.0),
]
DEMO_STEPS = {'triage': 0.4, 'lift': 1.4, 'retype': 1.0, 'name': 0.6, 'match': 1.2}
DEMO_QUESTIONS = ['is this the PAL or the NTSC build? the frame timer differs',
                  'symbol mangling looks like gcc 2.7 — can you confirm the toolchain?',
                  'struct at 0x8009c3a0: player state or camera rig?',
                  'jump table in {fn} needs the switch base — is there a map file?',
                  'ok to rename the whole audio_* family? it touches 40 call sites']
DEMO_ANSWERS = ['NTSC — the map file is in /ref', 'gcc 2.7.2 with -O2 -G0, per the build log',
                'camera rig; player state lives at 0x8009d100', 'yes, rename the family',
                'no map file — derive the switch base from the bounds check']
DEMO_BLOCKS = ['objdiff waiting for a free CPU slot', 'ghidra re-analysing after a retype',
               'GPU busy — queued behind another local model', 'rate-limited (429), backing off',
               'two agents renamed the same struct — merging']
DEMO_THEME = {
    'icon': '🧩', 'meter_label': 'reassembled',
    'stages': ['🥚 bytes', '🦴 skeleton', '🧠 meaning', '🏛  source'],
    'thought_key': 'match',
    'thoughts': [[0.0, 'mostly gibberish — the compiler is laughing at us'],
                 [0.35, 'shapes in the fog: a state machine, a suspicious memcpy'],
                 [0.6, 'it reads like a human wrote it'],
                 [0.85, 'nearly byte-perfect — the matching gods smile'],
                 [0.99, '🏆 100 % match — ship the source']],
    'agent_icons': {'opus': '🦉', 'sonnet': '🎻', 'gpt': '📜', 'deepseek': '🐋',
                    'glm': '🐼', 'qwen': '🐉', 'gemma': '💎', 'human': '🧑'},
    'state_icons': {'working': '🔧', 'waiting_human': '🙋', 'blocked': '🧱', 'idle': '💤',
                    'done': '🏁', 'failed': '💥'},
    'kind_icons': {'claim': '✋', 'answer': '💬', 'question': '❓', 'escalation': '🚨',
                   'stall': '🐌', 'note': '📝'},
}


def parse_dur(s: str) -> float:
    """'14d' / '6h' / '90m' / '45s' / '0' -> seconds."""
    s = str(s).strip()
    mult = {'d': 86400, 'h': 3600, 'm': 60, 's': 1}.get(s[-1:], None)
    return float(s[:-1]) * mult if mult else float(s)


def cmd_demo_swarm_feed(a) -> int:
    """A synthetic decompilation campaign (SPEC §2b).  `--history 14d` back-fills two
    weeks of events instantly, then runs live; `--until-now` stops after the back-fill."""
    import heapq
    import random
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from peek_progress import Progress
    rnd = random.Random(a.seed)
    units, deps, addrs = [], {}, set()

    def add_function():
        while True:
            fn = f'fn_{0x80010000 + rnd.randrange(0, 0x40000) * 4:08x}'
            if fn not in addrs:
                addrs.add(fn)
                break
        prev, new = None, []
        for step in DEMO_STEPS:
            u = f'{step}={fn}'
            units.append(u)
            new.append(u)
            if prev:
                deps[u] = [prev]
            prev = u
        return fn, new
    for _ in range(max(1, a.tasks // len(DEMO_STEPS))):
        add_function()
    agents = DEMO_AGENTS[:a.agents]
    info = {n: (mdl, h, g, price) for n, mdl, h, g, price in agents}
    hist = parse_dur(a.history)
    mean_cost = sum(DEMO_STEPS.values()) / len(DEMO_STEPS)
    unit_s = (hist * len(agents) / (len(units) * mean_cost * 0.7 * 1.6)) if hist else a.every
    now0 = time.time()
    vt = now0 - hist
    p = Progress(a.progress, eta=False)
    p.sweep_start(units, list(DEMO_STEPS), deps=deps, budget_usd=a.budget,
                  meta={'demo': 'decomp', 'binary': 'retro_game.elf'}, ts=vt)
    for n, (mdl, h, g, _) in info.items():
        p.agent(n, 'idle', model=mdl, harness=h, gpu=g, host=None if g else 'cloud', ts=vt)
    free, started, ok, matched = {n for n, *_ in agents}, set(), set(), set()
    tok = {n: 0 for n in info}
    heap, seq, live = [], itertools.count(), not hist

    def push(t, *ev):
        heapq.heappush(heap, (t, next(seq), *ev))

    def binary_match() -> float:   # share of functions that recompile byte-identical
        return round(len(matched) / max(1, len(addrs)), 4)
    while True:
        ready = [u for u in units if u not in started and all(d in ok for d in deps.get(u, []))]
        # local models take the volume steps; the big cloud models take retype/match
        for name in sorted(free, key=lambda n: (info[n][2] is None, n)):
            pick = next((u for u in ready if (info[name][2] is None) ==
                         (u.split('=')[0] in ('retype', 'match'))), ready[0] if ready else None)
            if pick is None:
                break
            ready.remove(pick)
            free.discard(name)
            started.add(pick)
            p.agent(name, 'working', task=pick, ts=vt)
            p.run_start(pick, agent=name, ts=vt)
            if rnd.random() < 0.25:
                p.message(name, 'claim', f'taking {pick}', ts=vt)
            cost = DEMO_STEPS[pick.split('=')[0]] * (1.4 if info[name][2] else 1.0)
            push(vt + unit_s * cost * rnd.uniform(0.6, 1.5), 'end', name, pick)
        if not heap:
            break
        t, _, kind, name, arg = heapq.heappop(heap)
        if not live and t >= now0:
            if a.until_now:
                break
            live, scale = True, a.every / unit_s
            heap = [(now0 + (x[0] - now0) * scale, *x[1:]) for x in heap] + \
                [(now0 + (t - now0) * scale, 0, kind, name, arg)]
            heapq.heapify(heap)
            unit_s = a.every
            continue
        if live and t > time.time():
            time.sleep(t - time.time())
        vt = t
        if kind == 'end':
            step, fn = arg.split('=')
            failed = rnd.random() < (0.12 if step == 'match' else 0.03)
            used = int(DEMO_STEPS[step] * rnd.uniform(30_000, 120_000))
            tok[name] += used
            p.usage(name, tokens=tok[name], usd=tok[name] * info[name][3] / 1e6, ts=vt)
            fm = None
            if step == 'match':
                fm = round(rnd.uniform(0.9, 0.995) if failed else 1.0, 4)
                if not failed:
                    matched.add(fn)
            metrics = {'match': binary_match(), 'tokens_k': round(used / 1000, 1)}
            if fm is not None:
                metrics['fn_match'] = fm
            p.run_end(arg, agent=name, status='failed' if failed else 'ok', metrics=metrics,
                      ts=vt)
            print(f'{p.k}/{p.n} done  {name:<9} {"FAILED " if failed else ""}{arg}'
                  f'  binary match {metrics["match"]:.1%}', flush=True)
            if failed:
                e = p.message(name, 'escalation',
                              f'{fn} stuck at {fm:.1%} — compiler flags?' if fm else
                              f'{arg} failed: lifter produced invalid C', ts=vt)
                if rnd.random() < 0.7:
                    push(vt + unit_s * rnd.uniform(2, 8), 'triage', name, e)
            else:
                ok.add(arg)
                if step == 'lift' and rnd.random() < 0.18:
                    callee, new = add_function()
                    p.plan(new, deps={u: deps[u] for u in new if u in deps}, ts=vt)
                    p.message(name, 'note', f'xref: {fn} calls unseen {callee}', ts=vt)
            r = rnd.random()
            if r < 0.08:
                q = p.message(name, 'question', rnd.choice(DEMO_QUESTIONS).format(fn=fn),
                              to='human', needs_human=True, ts=vt)
                p.agent(name, 'waiting_human', note='asked the human', ts=vt)
                push(vt + unit_s * rnd.uniform(1, 4), 'answer', name, q)
            elif r < 0.14 and info[name][2]:
                s = p.message(name, 'stall', f'context overflow on {fn}, retrying in chunks',
                              ts=vt)
                p.agent(name, 'blocked', note='context overflow', ts=vt)
                push(vt + unit_s * rnd.uniform(0.5, 2), 'unstall', name, s)
            elif r < 0.20:
                p.agent(name, 'blocked', note=rnd.choice(DEMO_BLOCKS), ts=vt)
                push(vt + unit_s * rnd.uniform(0.5, 2), 'unblock', name, None)
            else:
                p.agent(name, 'idle', ts=vt)
                push(vt + unit_s * rnd.uniform(0.05, 0.6), 'free', name, None)   # think time
        elif kind == 'free':
            free.add(name)
        elif kind == 'answer':
            p.message('human', 'answer', rnd.choice(DEMO_ANSWERS), to=name, re=arg, ts=vt)
            p.agent(name, 'idle', ts=vt)
            free.add(name)
        elif kind == 'triage':
            p.message('human', 'answer', 'parking it; revisit after the struct pass', re=arg,
                      ts=vt)
        elif kind == 'unstall':
            p.message(name, 'note', 'chunked retry worked', re=arg, ts=vt)
            p.agent(name, 'idle', ts=vt)
            free.add(name)
        elif kind == 'unblock':
            p.agent(name, 'idle', ts=vt)
            free.add(name)
    if not a.until_now:
        for name in info:
            p.agent(name, 'done', ts=vt)
        p.sweep_end(out='src/ (demo: nothing written)', ts=vt)
        print('campaign finished', flush=True)
    return 0


def demo_swarm_spec(cwd: str, command: str | None, gpus: list) -> dict:
    """The decompilation-campaign window: four panes, every swarm block, GPU + CPU."""
    return {
        'title': 'decomp campaign · retro_game.elf · cloud + local models, six harnesses',
        'workspace': 'peek-swarm', 'cwd': cwd, 'progress': 'swarm.jsonl', 'log': 'swarm.log',
        'command': command, 'gpu': gpus, 'every': 2, 'stale_after_s': 6 * 3600,
        'agent_stale_after_s': 12 * 3600, 'budget_usd': 150,
        'metrics': ['match', 'fn_match', 'tokens_k'],
        'failure_patterns': DEFAULT_FAILURES,
        'panes': [
            {'name': 'summary', 'blocks': ['title', 'divider', 'meter', 'stage', 'cost',
                                           'roster', 'lanes', 'stale', 'thought', 'footer']},
            {'name': 'board', 'dir': 'down', 'ratio': 0.5,
             'blocks': ['feed', 'deps', 'group_table']},
            {'name': 'compute', 'dir': 'right', 'ratio': 0.6,
             'blocks': ['gpu', 'cpu', 'charts', 'log']},
            {'name': 'log', 'dir': 'down', 'ratio': 0.5, 'blocks': ['raw']}],
        'theme': DEMO_THEME,
    }


def cmd_demo(a) -> int:
    """Write a demo spec (all GPUs, charts pane first) and launch it with the feeder."""
    d = Path(a.dir).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    try:
        uuids = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=5).stdout.split()
    except Exception:  # noqa: BLE001
        uuids = []
    me, py = shlex.quote(str(Path(__file__).resolve())), shlex.quote(sys.executable)
    if a.kind == 'swarm':
        spec = demo_swarm_spec(
            str(d), f'{py} -u {me} demo-feed --swarm --progress swarm.jsonl --every {a.every} '
                    f'--history {a.history} --agents {a.agents} --tasks {a.tasks}', uuids)
        spec['workspace'] = a.workspace if a.workspace != 'peek-demo' else 'peek-swarm'
        return _launch_demo(a, d, spec, 'swarm.json', tui=False)
    spec = {
        'title': 'showcase sweep · 4 groups × 4 values × 3 seeds', 'workspace': a.workspace,
        'cwd': str(d),
        'panes': [{'name': 'charts', 'blocks': ['charts']},
                  {'name': 'summary', 'dir': 'right', 'ratio': 0.36,
                   'blocks': ['title', 'divider', 'meter', 'stage', 'groups', 'now', 'last',
                              'charts', 'log', 'stale', 'thought', 'footer']},
                  {'name': 'log', 'dir': 'down', 'ratio': 0.4, 'blocks': ['raw']}],
        'progress': 'progress.jsonl', 'log': 'run.log',
        'command': f'{py} -u {me} demo-feed --progress progress.jsonl --every {a.every}',
        'gpu': uuids, 'every': 1, 'stale_after_s': 60,
        'metrics': ['active_frac', 'mean_hz', 'loss', 'entropy'],
        'theme': {'icon': '🪰', 'meter_label': 'wake-o-meter', 'stages': ['🥚', '🐛', '🫘', '🪰'],
                  'thought_key': 'active_frac',
                  'thoughts': [[0.0, 'deep sleep — not a twitch'], [0.3, 'stirring…'],
                               [0.55, 'buzzing about'], [0.75, 'full aerobatics']]},
    }
    return _launch_demo(a, d, spec, 'demo.json', tui=True)


def _launch_demo(a, d: Path, spec: dict, fname: str, tui: bool) -> int:
    mf = d / fname
    mf.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'wrote {mf}')
    if tui and not TUI_BIN.exists():
        print('peek-tui not built — building (compute-peek.py build-tui)')
        if cmd_build_tui(a):
            raise SystemExit('cargo build failed')
    return cmd_launch(argparse.Namespace(manifest=str(mf), start=True, mux=a.mux))


# ------------------------------------------------------------------ self-check
def cmd_selftest(a) -> int:
    """Smallest thing that fails if the folding, block dispatch or pane lookup breaks."""
    d = Path(tempfile.mkdtemp())
    prog = d / 'p.jsonl'
    t0 = time.time()
    units = ['a=1 s0', 'a=2 s0', 'b=1 s0', 'b=2 s0']
    recs = [{'ts': t0, 'event': 'sweep_start', 'n': 4, 'groups': ['a', 'b'], 'units': units},
            {'ts': t0, 'event': 'run_start', 'unit': units[0]}]
    for i, u in enumerate((units[0], units[2])):   # one finished unit per group
        recs.append({'ts': t0 + 10 * (i + 1), 'event': 'run_end', 'unit': u,
                     'group': unit_group(u), 'wall_s': 10.0, 'k': i + 1, 'n': 4,
                     'elapsed_s': 10.0 * (i + 1), 'eta_s': 10.0 * (4 - i - 1),
                     'metrics': {'x': 0.6 * (i + 1)}})
    recs.append({'ts': t0 + 20, 'event': 'run_start', 'unit': units[3]})
    prog.write_text(''.join(json.dumps(r) + '\n' for r in recs))
    (d / 'l.log').write_text('hello\n')
    m = {'_path': str(d / 'spec.json'), '_state': str(d / 'spec.state.json'), 'cwd': str(d),
         'title': 'selftest', 'workspace': 't', 'progress': [str(prog)], 'log': [str(d / 'l.log')],
         'panes': [dict(p) for p in DEFAULT_PANES], 'every': 1, 'stale_after_s': 900,
         'metrics': ['x'], 'failure_patterns': DEFAULT_FAILURES, '_gpus': [],
         'theme': {'icon': '⚙', 'meter_label': 'ticks', 'stages': ['a', 'b', 'c', 'd'],
                   'thought_key': 'x', 'thoughts': [[0, 'first'], [0.5, 'half']]}}
    st = read_progress(m['progress'], m.get('_fold'))
    assert st['k'] == 2 and st['n'] == 4, st['k']
    assert st['eta_s'] == 20.0 and st['current'] == units[3], (st['eta_s'], st['current'])
    assert [g['done'] for g in group_stats(st)] == [1, 1]
    for p in DEFAULT_PANES:
        if 'raw' in p['blocks']:
            continue
        out = render_pane(m, p['name'], 30)
        assert out.strip(), p['name']
    # bespoke window: renamed panes, reordered/absent blocks, theme label honoured
    m['panes'] = [{'name': 'only', 'blocks': ['thought', 'meter', 'stage', 'footer']}]
    out = render_pane(m, 'only', 10)
    assert 'ticks' in out and '❝' in out and 'half' in out, out
    assert 'summary' not in [p['name'] for p in panes_of(m)]
    try:
        render_pane({**m, 'panes': [{'name': 'x', 'blocks': ['nope']}]}, 'x')
    except SystemExit:
        pass
    else:
        raise AssertionError('unknown block accepted')
    assert m.get('panes') and render_panes(m)[0]['name'] == 'only'
    # ratatui glyph port: eighths resolve sub-cell progress; sparkline spans nine levels
    assert bar(0.25, 2).count('▌') == 1 and spark([0, 4, 8], 3) == '▁▄█', spark([0, 4, 8], 3)
    m['panes'] = [{'name': 'c', 'blocks': ['charts']}]
    assert '▁' in render_pane(m, 'c') or '▄' in render_pane(m, 'c')
    sj = state_json(m)
    assert sj['k'] == 2 and [d['unit'] for d in sj['done']] == [units[0], units[2]], sj['done']
    json.dumps(sj)
    _selftest_swarm(d)
    print(f'selftest ok — {len(BLOCKS)} blocks, {len(DEFAULT_PANES)} default panes, '
          f'fold/shards/pane-lookup/render, swarm fold/blocks, incremental fold + checkpoint '
          f'verified')
    return 0


def _selftest_swarm(d: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from peek_progress import Progress
    assert fmt_s(3 * 86400 + 4 * 3600) == '3d 04h' and fmt_n(4_560_000) == '4.56M'
    f = d / 'swarm.jsonl'
    pr = Progress(f, eta=False)
    t = time.time() - 3600
    pr.sweep_start(['a=1', 'b=1', 'c=1'], deps={'b=1': ['a=1'], 'c=1': ['b=1']},
                   budget_usd=10, ts=t)
    pr.run_start('a=1', agent='ada', ts=t + 1)
    q = pr.message('ada', 'question', 'which?', needs_human=True, ts=t + 2)
    pr.agent('ada', 'waiting_human', ts=t + 3)
    pr.usage('ada', tokens=1000, usd=1.0, ts=t + 4)
    pr.usage('ada', tokens=3000, usd=3.0, ts=t + 600)      # cumulative: latest wins
    m = {'_path': str(d / 'sw.json'), 'title': 'sw', 'progress': [str(f)], 'log': [],
         'panes': [dict(x) for x in SWARM_PANES], 'theme': {}, 'metrics': [], '_gpus': [],
         'every': 1, 'stale_after_s': 900, 'failure_patterns': []}
    st = read_progress(m['progress'])
    assert st['open'] and st['agents']['ada']['state'] == 'waiting_human', st['agents']
    assert sum(u['usd'] for u in st['usage'].values()) == 3.0
    assert task_states(st) == {'a=1': 'running', 'b=1': 'blocked', 'c=1': 'blocked'}
    pr.message('human', 'answer', 'this one', re=q, ts=t + 700)
    pr.run_end('a=1', agent='ada', status='failed', ts=t + 800)
    pr.plan(['d=1'], deps={'d=1': ['a=1']}, ts=t + 900)
    st = read_progress(m['progress'])              # incremental: only the new lines
    assert not st['open'] and st['n'] == 4 and st['k'] == 1, (st['open'], st['n'], st['k'])
    assert task_states(st)['b=1'] == 'stuck' and task_states(st)['d=1'] == 'stuck'
    _FOLDS.pop(str(f))
    full = read_progress(m['progress'])            # from scratch: same answer
    assert (full['n'], full['k'], full['agents'], full['usage']) == \
        (st['n'], st['k'], st['agents'], st['usage'])
    for name in ('summary', 'board'):
        assert render_pane(m, name, 30, 120).strip()
    m['panes'].append({'name': 'hw', 'blocks': ['lanes', 'cpu', 'gpu']})
    lanes = render_pane(m, 'hw', 20, 100)
    assert 'lanes' in lanes and 'cpu' in lanes, lanes
    assert cells('🐝a') == 3 and pad('🐝', 4) == '🐝  '
    assert 'over budget' in render_pane(m, 'summary', 30, 120)   # $3 per task × 4 > $10
    # truncate-and-regrow between reads restarts the fold
    pr.sweep_start(['z=1'], ts=t + 1000)
    pr.run_end('z=1', ts=t + 1001)
    st = read_progress(m['progress'])
    assert st['units'] == ['z=1'] and st['k'] == 1 and not st['agents'], st['units']
    # on-disk checkpoint for big files: a fresh reader resumes from it, then folds deltas
    big = d / 'big.jsonl'
    pb = Progress(big, eta=False)
    units = [f'u={i}' for i in range(4000)]
    pb.sweep_start(units, ts=t)
    for i, u in enumerate(units):
        pb.run_start(u, agent=f'a{i % 7}', ts=t + i)
        pb.message(f'a{i % 7}', 'note', 'x' * 200, ts=t + i)
        pb.run_end(u, agent=f'a{i % 7}', ts=t + i + 0.5)
    ck = str(d / 'big.fold.json')
    st1 = read_progress([str(big)], ck)
    assert Path(ck).exists() and st1['k'] == 4000
    _FOLDS.clear()
    pb.plan(['u=extra'], ts=t + 5000)
    st2 = read_progress([str(big)], ck)
    assert st2['n'] == 4001 and st2['k'] == 4000 and _FOLDS[str(big)]['off'] == big.stat().st_size


# ------------------------------------------------------------------ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def render_args(s):
        s.add_argument('--manifest', required=True)
        s.add_argument('--every', type=float, default=None)
        s.add_argument('--once', action='store_true')
        s.add_argument('--rows', type=int, default=None)
        s.add_argument('--cols', type=int, default=None)
        return s

    for name in ('show', 'runs'):
        render_args(sub.add_parser(name))
    s = render_args(sub.add_parser('pane'))
    s.add_argument('--name', required=True)

    s = sub.add_parser('launch')
    s.add_argument('--manifest', required=True)
    s.add_argument('--start', action='store_true', help='also start the manifest command')
    s.add_argument('--mux', choices=[*MUXES, 'auto'], default=None,
                   help='terminal multiplexer (default: $PEEK_MUX, spec "mux", else auto)')
    s.set_defaults(fn=cmd_launch)

    s = sub.add_parser('init', help='write a starter peek-spec')
    s.add_argument('--manifest', required=True)
    s.add_argument('--like', default=None, help='copy the SHAPE from an existing spec')
    s.add_argument('--kind', choices=['compute', 'swarm'], default='compute',
                   help='starter window when not cloning one (swarm: agent runs, SPEC §2b)')
    s.add_argument('--title', default=None)
    s.add_argument('--workspace', default=None)
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser('specs', help='list this project\'s peek-spec library')
    s.add_argument('--project', default=None)
    s.set_defaults(fn=cmd_specs)

    s = sub.add_parser('gpus', help='index / UUID / name, for pinning a spec to a device')
    s.set_defaults(fn=cmd_gpus)

    s = sub.add_parser('status')
    s.add_argument('--manifest', required=True)
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser('state', help='folded state as JSON (what peek-tui draws)')
    s.add_argument('--manifest', required=True)
    s.set_defaults(fn=cmd_state)

    sub.add_parser('build-tui', help='cargo build the ratatui charts pane (peek-tui/)'
                   ).set_defaults(fn=cmd_build_tui)

    s = sub.add_parser('demo-feed', help='emit a synthetic sweep (the demo command)')
    s.add_argument('--progress', required=True)
    s.add_argument('--every', type=float, default=2.5, help='mean seconds per unit')
    s.add_argument('--seed', type=int, default=None)
    s.add_argument('--pause', type=float, default=8.0, help='seconds between sweeps')
    s.add_argument('--no-loop', dest='loop', action='store_false')
    s.add_argument('--swarm', action='store_true', help='an agent swarm instead (SPEC §2b)')
    s.add_argument('--agents', type=int, default=7)
    s.add_argument('--tasks', type=int, default=40)
    s.add_argument('--history', default='0', help='swarm: back-fill this much past (14d, 6h)')
    s.add_argument('--until-now', action='store_true', help='swarm: stop after the back-fill')
    s.add_argument('--budget', type=float, default=60.0)
    s.set_defaults(fn=lambda a: (cmd_demo_swarm_feed if a.swarm else cmd_demo_feed)(a))

    s = sub.add_parser('demo', help='launch the design-showcase window on a synthetic run')
    s.add_argument('--dir', default='~/.cache/compute-peek/demo')
    s.add_argument('--workspace', default='peek-demo')
    s.add_argument('--every', type=float, default=2.5)
    s.add_argument('--mux', choices=[*MUXES, 'auto'], default=None)
    s.add_argument('--kind', choices=['compute', 'swarm'], default='compute')
    s.add_argument('--history', default='0', help='swarm: back-fill (e.g. 14d) then go live')
    s.add_argument('--agents', type=int, default=7)
    s.add_argument('--tasks', type=int, default=40)
    s.set_defaults(fn=cmd_demo)

    sub.add_parser('selftest').set_defaults(fn=cmd_selftest)
    a = ap.parse_args(argv)

    if a.cmd in ('show', 'runs', 'pane'):
        m = load_manifest(a.manifest)
        if a.cmd == 'pane':
            name = a.name
        else:
            rps = render_panes(m)
            i = 0 if a.cmd == 'show' else 1
            if i >= len(rps):
                raise SystemExit(f'{a.cmd}: this spec declares {len(rps)} renderer pane(s); '
                                 f'use `pane --name <name>`')
            name = rps[i]['name']
        every = a.every or m.get('every', 5)
        tty = sys.stdout.isatty()

        def draw():   # re-measured every repaint, so a resized pane reflows
            size = os.get_terminal_size() if tty else os.terminal_size((100, 30))
            return render_pane(m, name, a.rows or size.lines, a.cols or size.columns)
        return loop(draw, every, a.once)
    return a.fn(a)


if __name__ == '__main__':
    raise SystemExit(main())

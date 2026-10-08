#!/usr/bin/env python3
"""Regenerate the README screenshots from real renders (needs `rich`; peek-tui built).

    python3 docs/capture.py [OUT_DIR]        # default: docs/

Each window is laid out exactly as `launch` would split it (SPEC §1: pane i splits off
pane i-1 by dir/ratio), every pane is the real renderer's `--once` output at that pane's
size, and the frame is exported as SVG.  Runs are synthetic (`demo-feed`), time-stretched
so the numbers look like a real hour / week; a stub `nvidia-smi` with generic cards keeps
the capturing machine's hardware out of the pictures.
"""
from __future__ import annotations

import json
import os
import random
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
PEEK = ROOT / 'compute-peek.py'
TUI = ROOT / 'peek-tui' / 'target' / 'release' / 'peek-tui'
PY = sys.executable

FAKE_SMI = r'''#!/usr/bin/env python3
import random, sys
cards = [('0', 'GPU-00000000-demo-0000-0000-000000000000', 'NVIDIA RTX A6000', 49140, 300),
         ('1', 'GPU-11111111-demo-1111-1111-111111111111', 'NVIDIA RTX A6000', 49140, 300)]
args = ' '.join(sys.argv[1:])
pick = [c for c in cards if f'--id={c[0]}' in args or f'--id={c[1]}' in args] or cards
for i, uuid, name, mem, pl in pick:
    if 'uuid' in args:
        print(f'{i}, {uuid}, {name}, {mem}')
    else:
        r = random.Random(int(i) * 7 + 1)
        print(f'{i}, {name}, {r.randint(58, 71)}, {r.randint(83, 97)}, {r.randint(29000, 41000)}, '
              f'{mem}, {r.uniform(220, 285):.1f}, {pl}')
'''


def run(*cmd, **kw) -> str:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True,
                          **kw).stdout


def rects(panes: list[dict], w: int, h: int) -> tuple[dict, Layout]:
    """Pane name -> (cols, rows), and the rich Layout tree, following the launcher's splits."""
    def dir_(i, p):
        return p.get('dir', 'right' if i == 2 else 'down')

    def ratio(i, p):
        return p.get('ratio', 0.42 if i == 1 else 0.5)
    tree = {'pane': panes[0]['name']}
    leaf = {panes[0]['name']: tree}
    for i, p in enumerate(panes[1:], start=1):
        node = leaf[panes[i - 1]['name']]
        old = dict(node)
        new = {'pane': p['name']}
        node.clear()
        node.update(split=dir_(i, p), ratio=ratio(i, p), a=old, b=new)
        leaf[old['pane']], leaf[p['name']] = old, new
    sizes = {}

    def walk(n, cw, ch) -> Layout:
        if 'pane' in n:
            sizes[n['pane']] = (cw, ch)
            return Layout(name=n['pane'])
        if n['split'] == 'down':
            k = round(ch * n['ratio'])
            lay = Layout()
            first = walk(n['a'], cw, k)
            first.size = k
            lay.split_column(first, walk(n['b'], cw, ch - k))
        else:
            k = round(cw * n['ratio'])
            lay = Layout()
            first = walk(n['a'], k, ch)
            first.size = k
            lay.split_row(first, walk(n['b'], cw - k, ch))
        return lay
    return sizes, walk(tree, w, h)


def window_svg(spec_path: Path, raw_text: str, out: Path, title: str, w: int, h: int,
               env: dict) -> None:
    spec = json.loads(spec_path.read_text())
    sizes, lay = rects(spec['panes'], w, h)
    ws = spec.get('workspace', 'peek')
    for p in spec['panes']:
        cw, ch = sizes[p['name']]
        iw, ih = cw - 2, ch - 2
        if p['blocks'] == ['raw']:
            body = Text('\n'.join(raw_text.splitlines()[-ih:]), style='grey70')
        elif p['blocks'] == ['charts'] and TUI.exists():
            body = Text.from_ansi(run(TUI, '--manifest', spec_path, '--peek', PEEK, '--python',
                                      PY, '--once', '--ansi', '--tab', '1', '--width', iw,
                                      '--height', ih, env=env))
        else:
            body = Text.from_ansi(run(PY, PEEK, 'pane', '--manifest', spec_path, '--name',
                                      p['name'], '--once', '--rows', ih, '--cols', iw, env=env))
        body.no_wrap, body.overflow = True, 'crop'
        target = _find(lay, p['name'])
        target.update(Panel(body, title=f'{ws}:{p["name"]}', title_align='left',
                            border_style='grey39', padding=0))
    con = Console(width=w, height=h, record=True, force_terminal=True, color_system='truecolor',
                  file=open(os.devnull, 'w'))
    con.print(lay)
    out.write_text(con.export_svg(title=title))
    print(f'wrote {out}')


def _find(lay: Layout, name: str) -> Layout:
    if lay.name == name:
        return lay
    for c in lay.children:
        hit = _find(c, name)
        if hit is not None:
            return hit
    return None


def stretch(path: Path, factor: float, keep_run_ends: int | None = None) -> None:
    """Scale a fast synthetic run's clock by `factor`, optionally cut it mid-run, and end it
    'just now' — so a 2-second demo reads like an hour-long sweep."""
    recs, ends = [], 0
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if keep_run_ends is not None and e['event'] == 'run_end':
            ends += 1
            if ends > keep_run_ends:
                break
        if keep_run_ends is not None and e['event'] == 'sweep_end':
            break
        recs.append(e)
    t0 = recs[0]['ts']
    shift = time.time() - 25 - (t0 + (recs[-1]['ts'] - t0) * factor)
    for e in recs:
        e['ts'] = t0 + (e['ts'] - t0) * factor + shift
        for k in ('wall_s', 'elapsed_s', 'eta_s'):
            if e.get(k) is not None:
                e[k] *= factor
    path.write_text(''.join(json.dumps(e) + '\n' for e in recs))


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs'
    out.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix='peek-capture-'))
    smi = d / 'bin' / 'nvidia-smi'
    smi.parent.mkdir()
    smi.write_text(FAKE_SMI)
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, 'PATH': f'{smi.parent}:{os.environ["PATH"]}'}
    gpus = ['GPU-00000000-demo-0000-0000-000000000000', 'GPU-11111111-demo-1111-1111-111111111111']

    # 1. compute sweep, ~65 % through, unit walls ~1 min
    log = run(PY, PEEK, 'demo-feed', '--progress', d / 'sweep.jsonl', '--every', '0.02',
              '--no-loop', '--seed', '7', env=env)
    stretch(d / 'sweep.jsonl', 3000, keep_run_ends=31)
    raw = '\n'.join(   # the driver's stdout, re-derived from the stretched clock
        f'[sweep 1] {e["k"]}/{e["n"]} done, ETA {e["eta_s"] / 60:.0f} min  {e["unit"]:<10} '
        f'loss {e["metrics"]["loss"]:.3f}'
        for e in map(json.loads, (d / 'sweep.jsonl').read_text().splitlines())
        if e['event'] == 'run_end')
    sweep = {
        'title': 'lr sweep · 4 schedules × 4 rates × 3 seeds', 'workspace': 'lr-sweep',
        'cwd': str(d), 'progress': 'sweep.jsonl', 'gpu': gpus, 'every': 5,
        'stale_after_s': 900, 'metrics': ['active_frac', 'mean_hz', 'loss', 'entropy'],
        'panes': [{'name': 'charts', 'blocks': ['charts']},
                  {'name': 'summary', 'dir': 'right', 'ratio': 0.5,
                   'blocks': ['title', 'divider', 'meter', 'stage', 'groups', 'now', 'last',
                              'gpu', 'log', 'stale', 'thought', 'footer']},
                  {'name': 'log', 'dir': 'down', 'ratio': 0.62, 'blocks': ['raw']}],
        'theme': {'icon': '🔥', 'meter_label': 'burn-in', 'stages': ['🌱', '🌿', '🌳', '🍎'],
                  'thought_key': 'loss',
                  'thoughts': [[0.0, 'converging nicely'], [0.5, 'still finding its feet'],
                               [1.2, 'loss says: give me a minute']]},
    }
    (d / 'sweep.json').write_text(json.dumps(sweep))
    window_svg(d / 'sweep.json', raw, out / 'compute-window.svg',
               'compute-peek · compute sweep', 200, 50, env)

    # 2. agent swarm: a decompilation campaign, twelve days in (the `demo --kind swarm` spec)
    log = run(PY, PEEK, 'demo-feed', '--swarm', '--progress', d / 'swarm.jsonl', '--history',
              '12d', '--until-now', '--tasks', '150', '--seed', '9', env=env)
    import importlib.util
    spec_mod = importlib.util.spec_from_file_location('compute_peek', PEEK)
    cp = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(cp)
    swarm = cp.demo_swarm_spec(str(d), None, gpus)
    swarm['workspace'] = 'decomp'
    (d / 'swarm.json').write_text(json.dumps(swarm))
    # a stand-in host: 32 cores, the match stage's recompile+diff fanned out over half of them
    r = random.Random(5)
    cores = [r.uniform(70, 99) if i % 2 == 0 else r.uniform(3, 30) for i in range(32)]
    (d / 'cpu.json').write_text(json.dumps({
        'all': sum(cores) / len(cores), 'cores': cores, 'load': [17.2, 15.8, 14.1],
        'mem_used': 41.0, 'mem_total': 128.0}))
    env2 = {**env, 'PEEK_CPU_SAMPLE': str(d / 'cpu.json')}
    window_svg(d / 'swarm.json', log, out / 'swarm-window.svg',
               'compute-peek · decompilation campaign, day 12', 210, 66, env2)

    # 3. peek-tui on its own: the atlas and hardware tabs
    if TUI.exists():
        for tab, name in ((3, 'atlas'), (4, 'hardware')):
            ans = run(TUI, '--manifest', d / 'sweep.json', '--peek', PEEK, '--python', PY,
                      '--once', '--ansi', '--tab', tab, '--width', 140, '--height', 40, env=env)
            con = Console(width=140, record=True, force_terminal=True,
                          color_system='truecolor', file=open(os.devnull, 'w'))
            t = Text.from_ansi(ans)
            t.no_wrap = True
            con.print(t)
            (out / f'peek-tui-{name}.svg').write_text(con.export_svg(
                title=f'peek-tui · {name} tab'))
            print(f'wrote {out / f"peek-tui-{name}.svg"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

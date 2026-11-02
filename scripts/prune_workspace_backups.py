"""Prune backups that the terrain pipeline writes on every run.

Keeps what the pipeline can still read back: the checkpoint generations the
pointer files name (plus the newest KEEP_GENERATIONS of each kind), step inputs
referenced by a *-step.json, lake-leaf proofs named by the lake report, and in
each write-only backup directory the entries written within RECENT of its
newest entry (the latest run). Everything else is a stale copy.

  python prune_workspace_backups.py            # report only
  python prune_workspace_backups.py --apply
"""
import argparse, json, shutil
from pathlib import Path

import campaign_world as cw

A = cw.WORKSPACE
C = A / 'terrain-checkpoints'
KEEP_GENERATIONS = 2
RECENT = 6 * 3600
WRITE_ONLY = ['terrain-repair/local-coarse-seams-before', 'terrain-repair/river-roots-before',
              'terrain-repair/river-patches-before', 'terrain-repair/coarse-height-before',
              'terrain-repair/boundary-height-before', 'terrain-repair/camera-town-before',
              'terrain-repair/water-colour-before', 'terrain-repair/kit-map-before-sync',
              'terrain-repair/character-placeholders-before']


def size(p):
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def stale():
    out = []
    if C.exists():
        pointed = {Path(json.loads((C / f'{b}.json').read_text())['directory']).name
                   for b in ('polish', 'geometry') if (C / f'{b}.json').exists()}
        for kind in ('polish', 'geometry'):
            gens = sorted((d for d in C.glob(kind + '-*') if d.is_dir()), key=lambda d: d.stat().st_mtime)
            out += [d for d in gens[:-KEEP_GENERATIONS] if d.name not in pointed]
        refs = {Path(r['source']).name for f in C.glob('*-step.json')
                for r in json.loads(f.read_text()).get('files', {}).values() if r.get('source')}
        if (C / 'step-inputs').exists():
            out += [f for f in (C / 'step-inputs').iterdir() if f.name not in refs]
    lake = A / 'analysis/lake-instance-world-frame.json'
    if lake.exists() and (A / 'terrain-repair/lake-leaf-before').exists():
        proofs = {c['baseline_sha256'] + '.bin' for c in json.loads(lake.read_text())['changes']}
        out += [f for f in (A / 'terrain-repair/lake-leaf-before').iterdir() if f.name not in proofs]
    for rel in WRITE_ONLY:
        d = A / rel
        entries = list(d.iterdir()) if d.exists() else []
        if entries:
            newest = max(e.stat().st_mtime for e in entries)
            out += [e for e in entries if e.stat().st_mtime < newest - RECENT]
    out += sorted(A.glob('*.previous-*.pack'))
    return out


def checked_target(path):
    """Resolve each deletion target before use; never follow a link or leave this workspace."""
    workspace = A.resolve()
    if path.is_symlink() or path.is_junction():
        raise RuntimeError(f'Refusing to prune a linked path: {path}')
    target = path.resolve()
    if workspace not in target.parents:
        raise RuntimeError(f'Prune target is outside the workspace: {target}')
    return target


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--apply', action='store_true'); args = ap.parse_args()
    items = stale()
    total = sum(size(p) for p in items)
    print(f'{len(items)} stale backups, {total / 2**20:.0f} MB')
    if args.apply:
        for p in items:
            p = checked_target(p)
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        print('Pruned')


if __name__ == '__main__':
    main()

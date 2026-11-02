"""Landmark area-of-interest cells in map.hex (landmarks_spec.py).

Appends the landmark AOI keys to the map's areas_of_interest list (a rerun reuses
their indices) and tags their cells: index + 1 in byte 8 bits 3-7 (low five bits)
and byte 9 bit 0 (sixth bit), as in vanilla (5,935 cells of 35 AOIs; byte 8 bits
0-2 and byte 9 bits 1-7 are zero on every vanilla and current cell). The index is
six bits wide: at most 63 AOIs.

Cells, per kind: 'wall' hexes within WALL_HALF_WIDTH of the wall line, one key per
segment; 'pass', 'site', 'yardang' hexes within the landmark's radius; 'mountain'
the impassable hexes of a corridor round its points (all of the corridor when it
has few); 'lake' none (label only, like the vanilla rivers); 'land_cells' n takes
the n land hexes nearest the point (志贺岛 is too small for the canvas).
Only land hexes are tagged, never a settlement slot (byte 2 high nibble) or a hex
another AOI already holds. Earlier landmarks win (passes come before the walls).
Nothing else in the cell changes.

Reads:  <workspace>/work/final-map/map.hex (or --map)
Writes: with --apply the same map.hex (backup in <workspace>/backups/landmarks-<time>/,
        or next to --map), and analysis/landmarks-aoi-cells.json (next to --map when given).
Run after the map edits (roads, lakes, frontier) and before all_directions_build.py setup
(kit copy) / kit_map.sync_from_final().

  python landmarks_map.py                  # dry run
  python landmarks_map.py --apply
  python landmarks_map.py --map <main checkout>/work/landmarks-test/map.hex --apply
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import campaign_world as cw
import map_hex
import landmarks_spec as spec

A = cw.WORKSPACE
FINAL = A / 'work/final-map/map.hex'
WALL_HALF_WIDTH = 1.0
MOUNTAIN_IMPASSABLE_SHARE = .3
MAX_AOI = 63


def aoi_index(c):
    return (((c[..., 9].astype(np.int32) & 1) << 5) | (c[..., 8] >> 3)) - 1


def set_aoi(c, rr, qq, idx):
    v = idx + 1
    c[rr, qq, 8] = (c[rr, qq, 8] & 7) | ((v & 31) << 3)
    c[rr, qq, 9] = (c[rr, qq, 9] & 0xFE) | ((v >> 5) & 1)


def clear_aoi(c, mask):
    c[mask, 8] &= 7
    c[mask, 9] &= 0xFE


def candidate_cells(lm, land, free, impassable):
    """{aoi key: boolean mask over the window, (q0, r0)}."""
    kind = lm['kind']
    out = {}
    if kind == 'lake':
        return out
    keys = spec.aoi_keys(lm)
    if kind == 'wall':
        for key, seg in zip(keys, spec.wall_segments_xz(lm)):
            q0, r0, d = spec.hex_distance_window(spec.polyline_samples(seg, .25), 3)
            out[key] = (d <= WALL_HALF_WIDTH, q0, r0)
        return out
    pts = spec.landmark_xz(lm)
    if lm.get('land_cells'):
        q0, r0, d = spec.hex_distance_window(pts, 12)
        h, w = d.shape
        ok = land[r0:r0 + h, q0:q0 + w] & free[r0:r0 + h, q0:q0 + w]
        order = np.argsort(np.where(ok, d, np.inf), axis=None)[:lm['land_cells']]
        m = np.zeros_like(ok)
        m.flat[order] = True
        out[keys[0]] = (m & ok, q0, r0)
        return out
    radius = lm['radius']
    samples = spec.polyline_samples(pts, .25) if len(pts) > 1 else pts
    q0, r0, d = spec.hex_distance_window(samples, int(radius) + 2)
    m = d <= radius
    if kind == 'mountain':
        h, w = d.shape
        imp = m & impassable[r0:r0 + h, q0:q0 + w] & land[r0:r0 + h, q0:q0 + w]
        if imp.sum() >= MOUNTAIN_IMPASSABLE_SHARE * m.sum():
            m = imp
    out[keys[0]] = (m, q0, r0)
    return out


def aoi_list(old, keys):
    """(new AOI list, old index -> new index or -1): landmark keys (spec.PREFIX) no longer in
    keys (renamed or removed landmarks) leave the list, every other entry keeps its order, and
    new keys are appended in spec order. Deterministic: depends only on old and keys."""
    stale = [k for k in old if k.startswith(spec.PREFIX) and k not in keys]
    names = [k for k in old if k not in stale] + [k for k in keys if k not in old]
    remap = np.array([names.index(k) if k in names else -1 for k in old], np.int32)
    return names, remap, stale


def tag(m):
    c = m['cells'].copy()
    old = list(m['areas_of_interest'])
    keys = spec.all_keys()
    names, remap, stale = aoi_list(old, keys)
    if len(names) > MAX_AOI:
        raise ValueError(f'{len(names)} areas of interest; the cell index holds {MAX_AOI}')
    index = {k: names.index(k) for k in keys}
    idx = aoi_index(c)
    # Cells of stale landmark keys, and cells whose index points past the list, lose their tag;
    # the cells of every other entry carry its new index (entries after a stale key move up).
    dangling = (idx >= len(old))
    tagged = (idx >= 0) & ~dangling
    newidx = np.full(idx.shape, -1, np.int32)
    newidx[tagged] = remap[idx[tagged]]
    stale_cells = int((tagged & (newidx < 0)).sum())
    dangling_cells = int(dangling.sum())
    clear_aoi(c, idx >= 0)
    keep = newidx >= 0
    rr_, qq_ = np.nonzero(keep)
    set_aoi(c, rr_, qq_, newidx[keep])
    idx = aoi_index(c)
    ours = np.isin(idx, list(index.values()))
    rerun_cleared = int(ours.sum())
    clear_aoi(c, ours)
    idx = aoi_index(c)
    land = (c[..., 0] & 3) != 1
    slot = (c[..., 2] >> 4) > 0
    impassable = ((c[..., 2] >> 3) & 1) == 1
    ids = map_hex.region_ids(c)
    regions = m['land_regions'] + m['sea_regions']
    taken = idx >= 0
    report = []
    for lm in spec.LANDMARKS:
        for key, (mask, q0, r0) in candidate_cells(lm, land, land & ~slot & ~taken, impassable).items():
            free = land & ~slot & ~taken
            h, w = mask.shape
            sel = mask & free[r0:r0 + h, q0:q0 + w]
            rr, qq = np.nonzero(sel)
            rr, qq = rr + r0, qq + q0
            set_aoi(c, rr, qq, index[key])
            taken[rr, qq] = True
            owners = sorted({regions[i] for i in ids[rr, qq] if 0 <= i < len(regions)})
            report.append(dict(key=key, landmark=lm['key'], kind=lm['kind'], index=index[key], cells=int(len(qq)),
                               candidates=int(mask.sum()), regions=owners,
                               bbox=[int(qq.min()), int(rr.min()), int(qq.max()), int(rr.max())] if len(qq) else None))
        if lm['kind'] == 'lake':
            report.append(dict(key=spec.aoi_keys(lm)[0], landmark=lm['key'], kind='lake', index=index[spec.aoi_keys(lm)[0]], cells=0))
    # Checks: only bytes 8/9 changed, and only by our indices; vanilla AOI cells untouched.
    before, after = m['cells'], c
    diff = np.any(before != after, axis=2)
    assert not np.any(before[..., [i for i in range(16) if i not in (8, 9)]] != after[..., [i for i in range(16) if i not in (8, 9)]])
    vidx = aoi_index(before)
    new_idx = aoi_index(after)
    vanilla = (vidx >= 0) & (vidx < len(old))
    vanilla[vanilla] = ~np.char.startswith(np.array(old, dtype=str)[vidx[vanilla]], spec.PREFIX)
    assert np.all(new_idx[vanilla] >= 0) and np.array_equal(np.array(names, dtype=str)[new_idx[vanilla]], np.array(old, dtype=str)[vidx[vanilla]]), 'vanilla AOI cells changed'
    assert not np.any((new_idx >= 0) & ~vanilla & ~np.isin(new_idx, list(index.values()))), 'cells left on a stale AOI'
    for r in report:
        assert int((new_idx == r['index']).sum()) == r['cells'], r['key']
    return dict(m, cells=c, areas_of_interest=names), report, dict(changed_cells=int(diff.sum()), rerun_cleared=rerun_cleared,
                                                                   list_length=len(names), appended=[k for k in keys if k not in old],
                                                                   stale_removed=stale, stale_cells_cleared=stale_cells,
                                                                   dangling_cells_cleared=dangling_cells,
                                                                   moved_indices=int(np.sum(remap != np.arange(len(old)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--map', type=Path, default=FINAL)
    args = ap.parse_args()
    m = map_hex.read(args.map)
    out, report, summary = tag(m)
    labels = {k: list(p) for lm in spec.LANDMARKS for k, p in spec.label_positions(lm)}
    for r in report:
        r['label'] = labels.get(r['key'])
        print(f"{r['key']:34s} {r['kind']:8s} idx {r['index']:2d} cells {r['cells']:4d} label {r['label']} {','.join(r.get('regions', [])[:3])}", flush=True)
    print('list', summary['list_length'], 'appended', len(summary['appended']), 'changed cells', summary['changed_cells'],
          'rerun cleared', summary['rerun_cleared'], flush=True)
    print('stale AOIs removed', summary['stale_removed'], 'their cells cleared', summary['stale_cells_cleared'],
          'dangling cells cleared', summary['dangling_cells_cleared'], 'entries moved', summary['moved_indices'], flush=True)
    doc = dict(map=str(args.map), summary=summary, aoi=report, applied=args.apply, live_verified=False)
    if not args.apply:
        print('dry run: nothing written')
        return
    if args.map == FINAL:
        back = A / f'backups/landmarks-{time.strftime("%Y%m%d-%H%M%S")}'
        reportpath = A / 'analysis/landmarks-aoi-cells.json'
    else:
        back = args.map.parent / f'backups/landmarks-{time.strftime("%Y%m%d-%H%M%S")}'
        reportpath = args.map.parent / 'landmarks-aoi-cells.json'
    back.mkdir(parents=True)
    shutil.copy2(args.map, back / args.map.name)
    map_hex.write(out, args.map)
    check = map_hex.read(args.map)
    assert np.array_equal(check['cells'], out['cells']) and check['areas_of_interest'] == out['areas_of_interest']
    doc['backup'] = str(back)
    reportpath.parent.mkdir(parents=True, exist_ok=True)
    reportpath.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    print('map.hex written; backup at', back)


if __name__ == '__main__':
    main()

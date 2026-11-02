"""Beach / cliff marking (map.hex byte 0 bits 0-1) on the new sea shores of the canvas.

Vanilla marks every land hex beside open sea as beach (kind 2) or cliff (kind 3): 3832 sea-shore hexes,
2895 cliff (76 %), 937 beach, none plain (kind 0); river and lake shores are 95 % beach. The canvas shores
outside the native rectangle were left plain (91 % of 16749 hexes kind 0). Here each such kind-0 hex takes
the kind of its real shore, the same test coast_tiles uses for the beach pieces (coast_tiles.shore_types:
ETOPO 2022 in analysis/west-real-terrain.npz; LOW = real land and sea within 2 hexes and land within 1 hex
at or below 60 m): LOW -> beach (2), STEEP or UNKNOWN (no real coast within 2 hexes) -> cliff (3).

Kept as they are (CAIME validators and the SDK area rule):
  * town and port slots (byte 2 high nibble), sprawl and road hexes (byte 3 bits 0-6): RoadsValidator and
    the road tracer reject beach and cliff hexes on a road;
  * passable hexes within SPRAWL_RING hexes of a sprawl hex: SprawlValidator wants at most one hazard area
    (impassable, river, beach, cliff) touching a sprawl blob and none near it untouched; an impassable hex is
    a hazard whatever its kind, so marking it changes no hazard area, a passable one would add one;
  * passable hexes of a region whose SDK area count (hex_components.region_areas: impassable is class 7
    whatever the kind, plain 0, beach 1, cliff 6/8) would pass SDK_AREA_LIMIT: that region's passable hexes
    stay plain (reported);
  * hexes already beach or cliff, the native rectangle (the filler, revealed_filler.filler_hex: vanilla
    non-playable filler that is land now, playable or not, is canvas and marked: kind bits of non-playable
    hexes have no gameplay effect), water. Nothing but byte 0 bits 0-1 changes; the
    check compares every other byte of every cell.

  python apply_shore_kinds.py                         # dry run on work/final-map/map.hex: report only
  python apply_shore_kinds.py --map IN --out OUT      # isolated test: write the marked copy to OUT
  python apply_shore_kinds.py --apply                 # back up and write work/final-map/map.hex

Writes analysis/shore-kinds.json (and output/offline-view/shore-kinds.png, north up). Afterwards (with
--apply): kit_map.sync_from_final(), database build and Tw3kBuild pathfinding / validate
(build_combined_geometry.ps1).
"""
import argparse, collections, json, shutil, time
from pathlib import Path

import numpy as np

import campaign_world as cw
import coast_tiles as ct
import map_hex
from hex_components import hex_dilate, region_areas, SDK_AREA_LIMIT
from revealed_filler import filler_hex, revealed_hex, vanilla_playable_hex

A = cw.WORKSPACE
FINAL = A / 'work/final-map/map.hex'
BEACH, CLIFF = 2, 3
SPRAWL_RING = 2              # hexes: CAIME SprawlValidator outline (two rings round a sprawl blob)


def plan(m, elev):
    """(new cells, report). elev: (TH, TW) ETOPO metres on the canvas tile grid."""
    cells = m['cells'].copy()
    H, W = cells.shape[:2]
    kind = cells[..., 0] & 3
    water, sea, river, lake = ct.water_kinds(cells, m['land_ground_types'], m['sea_ground_types'])
    outside = np.ones((H, W), bool)
    outside[cw.NATIVE_HEX_SLICE] = False
    filler = filler_hex(cells, m['land_regions'] + m['sea_regions'])
    revealed = revealed_hex(cells, m['land_regions'] + m['sea_regions'])
    outside |= filler                    # vanilla non-playable filler that is land now (playable or not) is canvas
    shore = ~water & hex_dilate(sea) & outside
    imp = (cells[..., 2] & 8) > 0
    slot = (cells[..., 2] >> 4) > 0
    sprawl = (cells[..., 3] & 1) > 0
    road = (cells[..., 3] & 0x7e) > 0
    near_sprawl = sprawl.copy()
    for _ in range(SPRAWL_RING):
        near_sprawl = hex_dilate(near_sprawl)
    cand = shore & (kind == 0) & ~slot & ~sprawl & ~road & (imp | ~near_sprawl)
    types = ct.shore_types(cand, elev)
    new_kind = {t: (BEACH if t == 'low' else CLIFF) for t in ('low', 'steep', 'unknown')}
    for (q, r), t in types.items():
        cells[r, q, 0] = (cells[r, q, 0] & 0xFC) | new_kind[t]
    # SDK area rule: regions whose area count would pass the limit keep their passable shore plain
    ids = map_hex.region_ids(cells)
    names = m['land_regions'] + m['sea_regions']
    before = region_areas(map_hex.region_ids(m['cells']), m['cells'])
    reverted = {}
    for _ in range(3):
        after = region_areas(ids, cells)
        over = [k for k, v in after.items() if v > max(SDK_AREA_LIMIT, before.get(k, 0))]
        if not over:
            break
        for k in over:
            sel = (ids == k) & cand & ~imp
            cells[sel, 0] = m['cells'][sel, 0]
            reverted[names[k] if 0 <= k < len(names) else str(k)] = int(sel.sum())
    after = region_areas(ids, cells)
    changed = (cells[..., 0] & 3) != kind
    tcount = collections.Counter(types.values())
    rep = {
        'canvas_sea_shore_hexes': int(shore.sum()), 'revealed_sea_shore_hexes': int((shore & revealed).sum()),
        'marked_revealed': int((((cells[..., 0] & 3) != kind) & revealed).sum()),
        'filler_sea_shore_hexes': int((shore & filler).sum()), 'marked_filler': int((((cells[..., 0] & 3) != kind) & filler).sum()),
        'by_kind_before': {k: int((shore & (kind == k)).sum()) for k in (0, 2, 3)},
        'kept_plain': {'slot': int((shore & (kind == 0) & slot).sum()), 'sprawl': int((shore & (kind == 0) & sprawl).sum()),
                       'road': int((shore & (kind == 0) & road & ~sprawl).sum()),
                       'passable_near_sprawl': int((shore & (kind == 0) & ~slot & ~sprawl & ~road & ~imp & near_sprawl).sum())},
        'real_shore_types': dict(sorted(tcount.items())),
        'marked_beach': int((changed & ((cells[..., 0] & 3) == BEACH)).sum()),
        'marked_cliff': int((changed & ((cells[..., 0] & 3) == CLIFF)).sum()),
        'marked_passable': int((changed & ~imp).sum()),
        'sdk_area_reverted_regions': reverted,
        'sdk_area_max_after': max(after.values()), 'sdk_area_limit': SDK_AREA_LIMIT,
        'by_kind_after': {k: int((shore & ((cells[..., 0] & 3) == k)).sum()) for k in (0, 2, 3)},
    }
    return cells, changed, rep


def check(before, after, filler):
    """Only byte 0 bits 0-1 of canvas sea-shore land hexes change, to beach or cliff; no vanilla playable hex
    (revealed_filler.vanilla_playable_hex) and no other native-rectangle hex than the filler."""
    diff = before != after
    assert not diff[..., 1:].any(), 'a byte other than byte 0 changed'
    assert not ((before[..., 0] ^ after[..., 0]) & 0xFC).any(), 'byte 0 changed outside bits 0-1'
    ch = diff[..., 0]
    assert ((before[ch, 0] & 3) == 0).all() and np.isin(after[ch, 0] & 3, (BEACH, CLIFF)).all()
    assert not (ch & vanilla_playable_hex()).any(), 'a vanilla playable hex changed'
    nat = np.zeros(ch.shape, bool)
    nat[cw.NATIVE_HEX_SLICE] = True
    assert not (ch & nat & ~filler).any(), 'a native-rectangle hex other than the filler changed'


def preview(before, after, changed, path):
    from PIL import Image
    kb, ka = before[..., 0] & 3, after[..., 0] & 3
    img = np.full(kb.shape + (3,), 40, np.uint8)
    img[kb == 1] = (30, 50, 90)
    img[(kb != 1)] = (90, 90, 80)
    img[kb == 2] = (230, 210, 140); img[kb == 3] = (150, 60, 50)
    img[changed & (ka == BEACH)] = (255, 240, 0); img[changed & (ka == CLIFF)] = (255, 60, 200)
    Image.fromarray(img[::-1][::2, ::2]).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', type=Path, default=FINAL)
    ap.add_argument('--out', type=Path, help='write the marked map here (isolated test)')
    ap.add_argument('--apply', action='store_true', help='back up and write work/final-map/map.hex')
    args = ap.parse_args()
    m = map_hex.read(args.map)
    elev = np.load(A / 'analysis/west-real-terrain.npz')['elevation']
    cells, changed, rep = plan(m, elev)
    check(m['cells'], cells, filler_hex(m['cells'], m['land_regions'] + m['sea_regions']))
    rep['map'] = str(args.map)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    pv = cw._ROOT / 'output/offline-view/shore-kinds.png'
    pv.parent.mkdir(parents=True, exist_ok=True)
    preview(m['cells'], cells, changed, pv)
    target = args.out or (FINAL if args.apply else None)
    if target is None:
        print('dry run; preview', pv)
        return
    if target == FINAL:
        if args.map != FINAL:
            raise SystemExit('--apply writes work/final-map/map.hex from itself only')
        bak = FINAL.with_name(f'map.hex.before-shore-kinds-{time.strftime("%Y%m%d-%H%M%S")}')
        shutil.copy2(FINAL, bak)
        rep['backup'] = str(bak)
    m['cells'] = cells
    map_hex.write(m, target)
    import region_connectivity; region_connectivity.check_map(target)   # land regions stay whole (v17 crash); [to be absorbed by the map bake]
    (A / 'analysis/shore-kinds.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1) + '\n', encoding='utf-8') if target == FINAL else None
    print('wrote', target)


if __name__ == '__main__':
    main()

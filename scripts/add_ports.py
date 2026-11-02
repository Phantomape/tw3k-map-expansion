"""Port slots for new towns that historically had a harbour: town stamp redesigned as vanilla's port stamp.

How vanilla marks a port (3k_dlc07_main_map, checked here against all 32 vanilla ports at run time):
  * map.hex byte 2 high nibble is the settlement slot: 1 primary, 2 port. A town without a port has
    the 7-hex flower (resource town) or the 19-hex radius-2 hexagon (city) of primary hexes. A port
    town has the same footprint with three contiguous hexes of its outer ring taken out of the stamp:
    one land port hex L (slot 2, in the town's region) and the two water hexes W1, W2 beyond it
    (slot 2, in the water region; W1 touches L, W2 touches W1 and lies two hexes from L). So a port
    town is 4 + 1 (resource) or 16 + 1 (city) primary + land port hexes. 19 of the 32 vanilla ports
    have W1, W2 in river water, 13 in sea water.
  * every slot hex has byte 2 exactly 16 (primary) or 32 (port): passable, no other byte-2 bit; and
    byte 3 bit 0 (sprawl) set.
  * prepare_province.land_port_hexes then gives the region a port slot template (3k_minor_port or
    3k_province_capital_port), region_gameplay_db picks a port building donor, layout and battle map,
    and coast_props puts docks at the land port hex. Nothing in the database is written here.

What this step changes, per PORTS entry: the stamp centre C (the design town) and one of the vanilla
port patterns (primary offsets, L, W1, W2 as cube offsets from C, as found on the vanilla map; first
the orientations vanilla uses, then any rotation or mirror of them) are chosen so that every primary
hex and L is land of the town's own region, W1 and W2 are water of an allowed water region with no
slot and no ford, C stays as near the current town as possible (0 when the coast allows; a city's
19-hex stamp is all land, so a port city moves at least one hex towards the water) and as few
impassable hexes as possible are opened. Then:
  * new primary hexes: byte 2 = 16, byte 3 |= 1; L, W1, W2: byte 2 = 32, byte 3 |= 1;
  * old primary hexes outside the new footprint: byte 2 = 0 (passable, no slot), sprawl kept;
  * no other byte and no other hex changes: region ids, ground (bytes 5-6), rivers (byte 4), roads and
    fords (byte 3 bits 1-7) and byte 15 stay as they are (checked);
  * the design town follows C (relocation_previous_town, layout_revision_reason recorded).

Checks before writing (nothing is written if one fails): only the planned hexes changed and only in
bytes 2 and 3 bit 0; every port region gains exactly one land port hex and no other region's count
changes (move_town's rule); each new stamp has a vanilla port shape; no port on an inland lake
(prepare_province.landlocked_ports); SDK areas per region <= hex_components.SDK_AREA_LIMIT; towns
keep MIN_SPACING. Before / after hex diagrams go to output/offline-view/ports/.

  python add_ports.py            # dry run: report and diagrams, nothing written in the workspace
  python add_ports.py --apply    # back up, then write work/final-map/map.hex and the design (atomic)

Tamralipti is a river port (W1, W2 in tw3k_riv_sea_ganges_1 beside Tamluk, as 19 vanilla ports are):
apply_rivers --replace must keep slot-2 river hexes as they are (it restores river hexes to their
pre-river bytes otherwise, and keeps rivers TOWN_RING hexes off slots).

Afterwards: apply_roads.py --only-towns <the moved towns that end road legs> --apply,
all_directions_build.py database (port slot rows, building donors, layouts, battle maps),
sdk_memory_export.py (map-data and anchor audit), then the terrain pipeline (town plateaus, docks:
coast_props.py, which holds 10 ports at most).
"""
import argparse
import collections
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

import campaign_world as cw
import map_hex
from hex_components import region_areas, SDK_AREA_LIMIT
from layout_common import MIN_SPACING, hex_xy

A = cw.WORKSPACE
FINAL = A / 'work/final-map/map.hex'
OUT = cw._ROOT / 'output/offline-view/ports'
# Water regions a port may face: 'sea' = any non-river, non-lake water region (the Sea Beyond counts:
# all authored seas are DB copies of it, and the Korean ports on them have a working port slot);
# 'river' also allows river water regions, as 19 vanilla river ports do.
LAKES = ('3k_main_sea_lake',)
SEARCH_RADIUS = 6          # hexes the stamp centre may move towards the water

# region: (water kinds allowed, why). Only reviewed historical harbours.
PORTS = {
    'tw3k_hist_wa_nu_capital': ('sea', 'Nu (Hakata Bay): Hou Han shu, gold seal of 57; Wei zhi, sea traffic with Han and Daifang'),
    'tw3k_hist_dunxun_capital': ('sea', 'Dunxun (Mergui): Liang shu, "east and west meet, over ten thousand people a day"'),
    'tw3k_hist_rinan_lumber': ('sea', 'Lurong, Rinan: the Han gateway on the South Sea; the 166 Da Qin embassy came in beyond Rinan'),
    'tw3k_hist_xitu_fish': ('sea', 'Xitu (Panduranga): a fishing town (all 8 vanilla fishing towns have a port slot)'),
    'tw3k_hist_east_tianzhu_market': ('river', 'Tamralipti (Tamluk): the Ganges-mouth seaport, on the Rupnarayan / Hooghly estuary'),
}
# Stop-loss 2026-10-26 (v18): the Tamralipti river port moved the town onto ganges_1 and the Ganges, routed round
# the port, cut tw3k_hist_east_tianzhu_market in two (1,545 hexes at 727,538; whole in v16). Off unless
# TW3K_TAMRALIPTI_PORT=1 (the town keeps its v16 place; the trading-port resource waits for a port slot).
# Back with the town and river specs in the map bake. [step to be absorbed by the map bake]
if os.environ.get('TW3K_TAMRALIPTI_PORT', '0') != '1':
    PORTS.pop('tw3k_hist_east_tianzhu_market')


# ------------------------------------------------------------------ hex geometry (cube coordinates)
def to_cube(q, r):
    """Odd-q hex (q, r; odd columns half a row north) -> cube (x, y, z) with x + y + z = 0."""
    y2 = 2 * r + q % 2
    z = (y2 - q) // 2
    return (q, -q - z, z)


def to_hex(c):
    x, _, z = c
    y2 = 2 * z + x
    return (x, (y2 - x % 2) // 2)


def cadd(a, b):
    return tuple(i + j for i, j in zip(a, b))


def csub(a, b):
    return tuple(i - j for i, j in zip(a, b))


def cdist(a, b):
    return max(abs(i - j) for i, j in zip(a, b))


def rot60(c):
    x, y, z = c
    return (-z, -x, -y)


def mirror(c):
    x, y, z = c
    return (x, z, y)


FLOWER = [(0, 0, 0)] + [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]
HEXAGON = sorted((a, b, -a - b) for a in range(-2, 3) for b in range(-2, 3) if abs(a + b) <= 2)


def neighbours(q, r, W, H):
    for dq, dr in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1 if q % 2 else -1), (-1, 1 if q % 2 else -1)):
        if 0 <= q + dq < W and 0 <= r + dr < H:
            yield q + dq, r + dr


# ------------------------------------------------------------------ vanilla port patterns
def vanilla_patterns(V):
    """{'resource' | 'city': [pattern]} with pattern = (primary offsets, L, W1, W2) as cube offsets from
    the stamp centre, one per distinct vanilla port (sorted). Asserts the rule in the docstring."""
    c = V['cells']; ids = map_hex.region_ids(c); H, W = ids.shape
    water = (c[..., 0] & 3) == 1; slot = c[..., 2] >> 4
    out = {'resource': set(), 'city': set()}
    for i, reg in enumerate(V['land_regions']):
        pr, pq = np.nonzero((ids == i) & (slot == 2) & ~water)
        if not len(pr):
            continue
        assert len(pr) == 1, reg
        rr, qq = np.nonzero((ids == i) & (slot == 1))
        prim = {to_cube(int(q), int(r)) for q, r in zip(qq, rr)}
        L = to_cube(int(pq[0]), int(pr[0]))
        w1 = [(a, b) for a, b in neighbours(int(pq[0]), int(pr[0]), W, H) if water[b, a] and slot[b, a] == 2]
        assert len(w1) == 1, reg
        w2 = [(a, b) for a, b in neighbours(*w1[0], W, H) if water[b, a] and slot[b, a] == 2]
        assert len(w2) == 1, reg
        W1, W2 = to_cube(*w1[0]), to_cube(*w2[0])
        kind, full = ('resource', FLOWER) if len(prim) == 4 else ('city', HEXAGON)
        assert len(prim) in (4, 16), (reg, len(prim))
        centres = [cc for cc in {csub(p, f) for p in prim for f in full}
                   if {cadd(cc, f) for f in full} == prim | {L, W1, W2}]
        assert len(centres) == 1, (reg, centres)
        C = centres[0]
        assert cdist(L, W2) == 2, reg
        out[kind].add((tuple(sorted(csub(p, C) for p in prim)), csub(L, C), csub(W1, C), csub(W2, C)))
    return {k: sorted(v) for k, v in out.items()}


def symmetric(patterns):
    """Every rotation and mirror of the patterns not already among them (sorted)."""
    seen = set(patterns); extra = set()
    for prim, L, W1, W2 in patterns:
        for m in (False, True):
            f = (lambda c: mirror(c)) if m else (lambda c: c)
            cells = [tuple(sorted(f(p) for p in prim)), f(L), f(W1), f(W2)]
            for _ in range(6):
                pat = (tuple(sorted(cells[0])), cells[1], cells[2], cells[3])
                if pat not in seen:
                    extra.add(pat)
                cells = [[rot60(p) for p in cells[0]], rot60(cells[1]), rot60(cells[2]), rot60(cells[3])]
    return sorted(extra)


# ------------------------------------------------------------------ planning
def plan_port(F, design, region, kinds, patterns):
    """Best placement {centre, primaries, L, W1, W2, ...} for one port, or {'error': ...}."""
    c = F['cells']; ids = map_hex.region_ids(c); H, W = ids.shape
    names = F['land_regions'] + F['sea_regions']; n_land = len(F['land_regions'])
    water = (c[..., 0] & 3) == 1; slot = c[..., 2] >> 4; imp = (c[..., 2] & 8) > 0
    ford = (c[..., 3] & 128) > 0
    me = names.index(region)
    county = next(t for t in design['regions'] if t['region'] == region)
    tq, tr = county['town']
    rr, qq = np.nonzero((ids == me) & (slot == 1))
    old = sorted((int(q), int(r)) for q, r in zip(qq, rr))
    land_ports = int(((ids == me) & (slot == 2) & ~water).sum())
    if land_ports == 1 and len(old) in (4, 16):
        return {'present': True}       # a rerun: the port is already there
    kind = 'city' if len(old) == 19 else 'resource' if len(old) == 7 else None
    if kind is None or land_ports:
        return {'error': f'stamp of {len(old)} primary and {land_ports} land port hexes is not a vanilla no-port stamp (7 or 19)'}

    def water_ok(q, r):
        if not (0 <= q < W and 0 <= r < H) or not water[r, q] or slot[r, q] or ford[r, q]:
            return None
        name = names[ids[r, q]]
        if ids[r, q] < n_land or name in LAKES or 'lake' in name:
            return None
        river = '_riv_sea_' in name
        if river and kinds != 'river':
            return None
        return name

    # Land of another authored region of the same province may join the stamp (as move_town re-divides
    # a province), never a slot hex or land of another province or of vanilla.
    mates = {names.index(t['region']) for t in design['regions']
             if t['province_key'] == county['province_key'] and t['region'] != region and t['region'] in names}

    def land_ok(q, r):
        return 0 <= q < W and 0 <= r < H and not water[r, q] and (ids[r, q] == me or (ids[r, q] in mates and not slot[r, q]))

    towns = {t['region']: t['town'] for t in design['regions'] if t['region'] != region}

    def nearest(q, r):
        return min(math.hypot(*np.subtract(hex_xy(q, r), hex_xy(*t))) for t in towns.values())
    # A moved centre keeps MIN_SPACING from every other town, or (Kyushu: Nu and Yamatai's rice town
    # are already 6.1 apart) at least its current nearest-town distance.
    spacing = min(MIN_SPACING, nearest(tq, tr))
    best = None
    T = to_cube(tq, tr)
    sets = [('vanilla orientation', patterns[kind]), ('rotated / mirrored', symmetric(patterns[kind]))]
    for rank, (label, pats) in enumerate(sets):
        for dq in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            for dz in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
                dc = (dq, -dq - dz, dz)
                move = cdist(dc, (0, 0, 0))
                if move > SEARCH_RADIUS:
                    continue
                C = cadd(T, dc)
                for pi, (prim, L, W1, W2) in enumerate(pats):
                    cells = [to_hex(cadd(C, p)) for p in prim]
                    lh, w1, w2 = to_hex(cadd(C, L)), to_hex(cadd(C, W1)), to_hex(cadd(C, W2))
                    if not all(land_ok(*h) for h in cells + [lh]):
                        continue
                    s1, s2 = water_ok(*w1), water_ok(*w2)
                    if s1 is None or s2 is None or s1 != s2:
                        continue
                    cq, cr = to_hex(C)
                    if move and nearest(cq, cr) < spacing:
                        continue
                    opened = sum(bool(imp[r, q]) for q, r in cells + [lh])
                    new_cells = sum((q, r) not in set(old) for q, r in cells + [lh])
                    taken = sorted((q, r) for q, r in cells + [lh] if ids[r, q] != me)
                    key = (rank, len(taken), move, opened, new_cells, pi)
                    if best is None or key < best[0]:
                        best = (key, {'region': region, 'kind': kind, 'pattern_set': label, 'pattern': pi,
                                      'town_before': [tq, tr], 'centre': [cq, cr], 'moved_hexes': move,
                                      'primaries': sorted(cells), 'land_port': list(lh), 'water_port': [list(w1), list(w2)],
                                      'water_region': s1, 'opened_impassable': opened, 'new_stamp_hexes': new_cells,
                                      'nearest_town': [round(nearest(tq, tr), 2), round(nearest(cq, cr), 2)],
                                      'taken_from_province': [[q, r, names[ids[r, q]]] for q, r in taken],
                                      'old_primaries': old})
        if best is not None:
            break
    if best is None:
        return {'error': f'no vanilla port stamp fits within {SEARCH_RADIUS} hexes of the town (land of the region, '
                         f'{"sea or river" if kinds == "river" else "sea"} water with no slot or ford)'}
    return best[1]


def apply_plan(cells, p, names):
    """Write one port's stamp into cells (in place)."""
    if p['taken_from_province']:
        ids = map_hex.region_ids(cells)
        for q, r, _ in p['taken_from_province']:
            ids[r, q] = names.index(p['region'])
        map_hex.set_region_ids(cells, ids)
    new = {tuple(h) for h in p['primaries']}
    for q, r in p['old_primaries']:
        if (q, r) not in new:
            cells[r, q, 2] = 0
    for q, r in sorted(new):
        cells[r, q, 2] = 16
        cells[r, q, 3] |= 1
    for q, r in [p['land_port']] + p['water_port']:
        cells[r, q, 2] = 32
        cells[r, q, 3] |= 1


def stamp_kind(cells, ids, me):
    """(primaries, land ports) of a region and whether that is a vanilla port shape."""
    water = (cells[..., 0] & 3) == 1; slot = cells[..., 2] >> 4
    return int(((ids == me) & (slot == 1)).sum()), int(((ids == me) & (slot == 2) & ~water).sum())


# ------------------------------------------------------------------ diagrams
def diagram(before, after, names, p, path, radius=7, s=16):
    """Before / after hexes round the port: region colours, water, slot hexes, impassable, town."""
    from PIL import Image, ImageDraw
    me = names.index(p['region'])
    q0, r0 = p['town_before']
    panels = []
    for title, cells, town in (('before', before, p['town_before']), ('after', after, p['centre'])):
        ids = map_hex.region_ids(cells)
        water = (cells[..., 0] & 3) == 1; slot = cells[..., 2] >> 4; imp = (cells[..., 2] & 8) > 0
        cols = range(q0 - 2 * radius, q0 + 2 * radius + 1)
        w = int((len(cols) + 1) * s * 1.5 * .58) + 40
        h = int((2 * radius + 3) * s * 1.0) + 60
        im = Image.new('RGB', (w, h), (255, 255, 255)); d = ImageDraw.Draw(im)
        others = {}
        for q in cols:
            for r in range(r0 - radius - 1, r0 + radius + 2):
                if not (0 <= r < cells.shape[0] and 0 <= q < cells.shape[1]):
                    continue
                x = 20 + (q - cols[0]) * s * .87
                y = h - 30 - (r - (r0 - radius - 1) + (q % 2) * .5) * s
                name = names[ids[r, q]]
                if water[r, q]:
                    col = (60, 90, 170) if '_riv_sea_' in name else (90, 140, 200)
                elif ids[r, q] == me:
                    col = (235, 200, 120)
                else:
                    col = others.setdefault(name, tuple(int(v) for v in (150 + 40 * (len(others) % 3), 190 - 30 * (len(others) % 2), 150)))
                if imp[r, q] and not water[r, q]:
                    col = tuple(int(v * .6) for v in col)
                if slot[r, q] == 1:
                    col = (170, 40, 40)
                elif slot[r, q] == 2:
                    col = (255, 140, 0) if not water[r, q] else (255, 230, 0)
                pts = [(x + s * .58 * math.cos(math.radians(a)), y + s * .58 * math.sin(math.radians(a))) for a in range(0, 360, 60)]
                d.polygon(pts, fill=col, outline=(120, 120, 120))
                if [q, r] == list(town):
                    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 0, 0))
        d.text((6, 4), f'{p["region"]} {title}: town {list(town)}', fill=(0, 0, 0))
        d.text((6, h - 18), 'red primary, orange land port, yellow water port, dark = impassable', fill=(0, 0, 0))
        panels.append(im)
    out = Image.new('RGB', (panels[0].size[0] * 2 + 8, panels[0].size[1]), (255, 255, 255))
    out.paste(panels[0], (0, 0)); out.paste(panels[1], (panels[0].size[0] + 8, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.png')
    out.save(tmp); os.replace(tmp, path)


def write_text_atomic(path, text):
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--only', nargs='*', help='regions (default: every PORTS entry)')
    ap.add_argument('--write-to', type=Path, help='offline checks: write map.hex and the design into this folder only')
    ap.add_argument('--out', type=Path, default=OUT, help='diagram folder (default output/offline-view/ports)')
    args = ap.parse_args()
    import prepare_province
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    F, V = map_hex.read(FINAL), map_hex.read(cw.VANILLA_MAP)
    names = F['land_regions'] + F['sea_regions']
    patterns = vanilla_patterns(V)
    report = {'vanilla_patterns': {k: len(v) for k, v in patterns.items()}, 'ports': {}, 'skipped': {}}
    cells = F['cells'].copy()
    planned = []
    for region in sorted(args.only or PORTS):
        kinds, why = PORTS[region]
        p = plan_port(F, design, region, kinds, patterns)
        if p.get('present'):
            report['skipped'][region] = 'port already present'
            continue
        if 'error' in p:
            report['skipped'][region] = p['error']
            continue
        p['why'] = why
        planned.append(p)
        report['ports'][region] = {k: v for k, v in p.items() if k != 'old_primaries'}
        apply_plan(cells, p, names)

    # Byte 15 (region-border sides) where land changed hands (move_town.refresh_border_masks).
    from move_town import refresh_border_masks
    report['border_masks_refreshed'] = refresh_border_masks(cells, map_hex.region_ids(F['cells']))
    # Checks.
    before = F['cells']
    ids0 = map_hex.region_ids(before); ids1 = map_hex.region_ids(cells)
    taken = np.zeros(ids0.shape, bool)
    for p in planned:
        for q, r, _ in p['taken_from_province']:
            taken[r, q] = True
    touched = np.zeros(ids0.shape, bool)
    for p in planned:
        for q, r in p['old_primaries'] + [tuple(h) for h in p['primaries']] + [tuple(p['land_port'])] + [tuple(h) for h in p['water_port']]:
            touched[r, q] = True
    from hex_components import hex_dilate, region_components
    diff = (np.delete(before, 15, axis=2) != np.delete(cells, 15, axis=2)).any(axis=2)
    other_bytes = np.delete(np.arange(16), [0, 1, 2, 3, 15])
    sprawl_only = ((before[..., 3] ^ cells[..., 3]) & 0xFE) == 0
    pieces0, pieces1 = region_components(ids0), region_components(ids1)
    involved = sorted(set(ids0[taken].tolist()) | set(ids1[taken].tolist()))
    checks = {
        'changes_only_planned_hexes': bool(not (diff & ~touched).any()),
        'byte15_only_beside_ownership_changes': bool(not ((before[..., 15] != cells[..., 15]) & ~hex_dilate(taken)).any()),
        'only_bytes_0_3_15_and_byte3_bit0': bool((before[..., other_bytes] == cells[..., other_bytes]).all() and sprawl_only.all()
                                               and ((before[..., 0] & 7) == (cells[..., 0] & 7)).all()),
        'region_ids_change_only_on_taken_land': bool(not ((ids0 != ids1) & ~taken).any()),
        'region_pieces_not_increased': bool(all(pieces1[i] <= pieces0[i] for i in involved)),
    }
    lp0 = prepare_province.land_port_hexes(F); lp1 = prepare_province.land_port_hexes(dict(F, cells=cells))
    wanted = {p['region'] for p in planned}
    changed = {k: [lp0.get(k, 0), lp1.get(k, 0)] for k in sorted(set(lp0) | set(lp1)) if lp0.get(k, 0) != lp1.get(k, 0)}
    checks['port_counts_change_only_for_ports'] = bool(set(changed) == wanted and all(v == [0, 1] for v in changed.values()))
    checks['every_port_planned_or_present'] = not [k for k, v in report['skipped'].items() if v != 'port already present']
    shapes = {p['region']: stamp_kind(cells, ids1, names.index(p['region'])) for p in planned}
    checks['vanilla_port_shapes'] = bool(all(v in ((4, 1), (16, 1)) for v in shapes.values()))
    landlocked = prepare_province.landlocked_ports(dict(F, cells=cells))
    checks['no_inland_lake_ports'] = not landlocked
    areas = region_areas(ids1, cells)
    checks['sdk_areas_within_limit'] = bool(max(areas.values()) <= SDK_AREA_LIMIT)
    report.update({'checks': checks, 'land_port_hexes_changed': changed, 'stamps': shapes,
                   'max_sdk_areas': [names[max(areas, key=areas.get)], int(max(areas.values()))],
                   'land_port_hexes_total': [int(sum(lp0.values())), int(sum(lp1.values()))]})
    for p in planned:
        diagram(before, cells, names, p, args.out / f'{p["region"]}.png')
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not all(checks.values()):
        raise RuntimeError(f'port checks fail: {[k for k, v in checks.items() if not v]}; nothing written')
    if not (args.apply or args.write_to):
        print('Dry run only; add --apply to write map.hex and the design.')
        return
    if not planned:
        print('Nothing to write: every port is already on the map.')
        return
    if args.write_to:
        args.write_to.mkdir(parents=True, exist_ok=True)
        target_map, target_design = args.write_to / 'map.hex', args.write_to / cw.DESIGN_PATH.name
    else:
        back = A / f'backups/ports-{time.strftime("%Y%m%d-%H%M%S")}'
        back.mkdir(parents=True)
        for path in (FINAL, cw.DESIGN_PATH):
            shutil.copy2(path, back / path.name)
        report['backup'] = str(back)
        target_map, target_design = FINAL, cw.DESIGN_PATH
    map_hex.write(dict(F, cells=cells), target_map)
    map_hex.read(target_map)
    by_region = {t['region']: t for t in design['regions']}
    for p in planned:
        t = by_region[p['region']]
        if p['centre'] != list(t['town']):
            t['relocation_previous_town'] = list(t['town'])
            t['town'] = list(p['centre'])
            if 'climate_seed' in t:
                t['climate_seed'] = list(p['centre'])
            t['layout_revision_reason'] = 'port stamp: centre moved to the water (add_ports.py)'
    write_text_atomic(target_design, json.dumps(design, ensure_ascii=False, indent=2) + '\n')
    if args.write_to:
        print('Ports written to', args.write_to)
        return
    (A / 'analysis').mkdir(exist_ok=True)
    write_text_atomic(A / 'analysis/add-ports.json', json.dumps(report, ensure_ascii=False, indent=1))
    print('Ports written; backup at', report['backup'])


if __name__ == '__main__':
    main()

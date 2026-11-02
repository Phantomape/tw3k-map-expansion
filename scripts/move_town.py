"""Move one authored town and re-partition only its province's regions.

The town's settlement stamp moves to the new hex; the province's land (its
regions' current hexes, nothing else) is re-divided among its towns by a
multi-source Dijkstra (impassable hexes cost more, so borders follow the
mountains). Passability is kept except around the new town and along a
corridor from it to the province capital. The old stamp hexes take the bytes
of their nearest non-stamp neighbour. Vanilla playable land, sea and every
region outside the province are untouched (checked).

  python move_town.py tw3k_hist_dunhuang_lumber 660 846            # dry run
  python move_town.py tw3k_hist_dunhuang_lumber 660 846 --apply    # back up, write map.hex and design

Afterwards: all_directions_build.py database, then
build_polished_campaign_terrain.ps1 -RebuildGeometry.
"""
import argparse, collections, heapq, json, math, os, shutil, time
import xml.etree.ElementTree as ET

import numpy as np

import campaign_world as cw
import map_hex
from analyze_reference import ROOT
from layout_common import hex_xy, stamp_offsets, line_cells, MIN_SPACING, BORDER_NOISE

A = cw.WORKSPACE
FINAL = A / 'work/final-map/map.hex'
IMPASSABLE_COST = 4.0
OPEN_RADIUS = 4       # hexes opened around the moved town


def neighbours(q, r):
    odd = q % 2
    for dq, dr in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1 if odd else -1), (-1, 1 if odd else -1)):
        yield q + dq, r + dr


def one_sided_road_edges(cells):
    """Road edge bits (byte 3 bits 1-6, direction order of apply_roads.DIRS) without the partner bit."""
    m = ((cells[:, :, 3] >> 1) & 63).astype(np.uint8)
    H, W = m.shape
    dirs = (((0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)),
            ((0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)))
    bad = 0
    for r, q in zip(*np.nonzero(m)):
        for d in range(6):
            if m[r, q] >> d & 1:
                dq, dr = dirs[q & 1][d]; a, b = q + dq, r + dr
                if not (0 <= a < W and 0 <= b < H) or not m[b, a] >> ((d + 3) % 6) & 1:
                    bad += 1
    return bad


def refresh_border_masks(cells, ids_before):
    """Byte 15 (region-border edge mask, add_inland_lake.edge_mask) on every hex whose owner changed and
    its hex neighbours; set_region_ids does not maintain it. Returns the number of hexes rewritten."""
    from add_inland_lake import edge_mask
    from hex_components import hex_dilate
    ids = map_hex.region_ids(cells)
    rr, qq = np.nonzero(hex_dilate(ids != ids_before))
    for r, q in zip(rr.tolist(), qq.tolist()):
        cells[r, q, 15] = edge_mask(ids, q, r)
    return len(rr)


def write_text_atomic(path, text):
    """<name>.<pid>.tmp beside the target, then os.replace (as map_hex.write)."""
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('region'); ap.add_argument('q', type=int); ap.add_argument('r', type=int)
    ap.add_argument('--reason', default='moved to its historical site')
    ap.add_argument('--head-start', type=float, default=0., help='distance head start of the moved town (larger territory)')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    by_region = {c['region']: c for c in design['regions']}
    moved = by_region[args.region]
    group = [c['region'] for c in design['regions'] if c['province_key'] == moved['province_key']]
    capital = next((k for k in group if by_region[k]['is_capital']), None)   # None: the province capital is a vanilla region
    F, V = map_hex.read(FINAL), map_hex.read(cw.VANILLA_MAP)
    cells = F['cells'].copy(); H, W = cells.shape[:2]
    names = list(F['land_regions']) + list(F['sea_regions']); index = {n: i for i, n in enumerate(names)}
    ids = map_hex.region_ids(cells)
    owner = np.array(names + ['?'], dtype=object)[ids]
    zone = np.isin(owner, group)
    vnames = V['land_regions'] + V['sea_regions']
    vplay = np.zeros((H, W), bool)
    vplay[cw.NATIVE_HEX_SLICE] = ~np.vectorize(lambda s: 'non_playable' in s)(np.array(vnames + ['?'], dtype=object)[map_hex.region_ids(V['cells'])])
    assert not (zone & vplay).any(), 'province holds vanilla playable land'
    slot = cells[:, :, 2] >> 4

    towns = {k: list(by_region[k]['town']) for k in group}
    if capital is None:
        # Authored counties in a vanilla commandery: the corridor heads for the vanilla capital's
        # primary-slot hex nearest its stamp centre (clipped to the province land like any corridor).
        capital = next((j.findtext('region') for j in ET.parse(A / 'work/native-split-xml/region_to_province_junctions.xml').getroot()
                        if j.findtext('province') == moved['province_key'] and j.findtext('is_capital') == '1'), None)
        rr_c, qq_c = np.nonzero((owner == capital) & (slot == 1))
        if not len(qq_c):
            raise RuntimeError(f'{moved["province_key"]}: no capital among the authored regions and none with a primary slot on the map')
        cx, cy = hex_xy(qq_c, rr_c); i = int(np.argmin(np.hypot(cx - cx.mean(), cy - cy.mean())))
        cap_town = [int(qq_c[i]), int(rr_c[i])]
    else:
        cap_town = None
    old = towns[args.region]; towns[args.region] = [args.q, args.r]
    if not zone[args.r, args.q]:
        raise RuntimeError(f'{(args.q, args.r)} is outside the province land')
    for k in group:
        if k != args.region:
            d = math.hypot(*np.subtract(hex_xy(*towns[k]), hex_xy(args.q, args.r)))
            if d < MIN_SPACING:
                raise RuntimeError(f'too close to {k}: {d:.1f} < {MIN_SPACING}')

    # Old stamp hexes take the bytes of the nearest non-stamp hex of the zone.
    old_stamp = stamp_offsets(owner == args.region, slot, tuple(old))
    stamp_cells = {(q, r) for _, _, q, r in old_stamp}
    for q, r in stamp_cells:
        seen, frontier = {(q, r)}, [(q, r)]
        while frontier:
            nxt = []
            for a in frontier:
                for b in neighbours(*a):
                    if b in seen or not (0 <= b[0] < W and 0 <= b[1] < H):
                        continue
                    seen.add(b); nxt.append(b)
            pick = [b for b in nxt if zone[b[1], b[0]] and b not in stamp_cells and slot[b[1], b[0]] == 0]
            if pick:
                bq, br = pick[0]
                cells[r, q, 0] = (cells[r, q, 0] & 0xF8) | (F['cells'][br, bq, 0] & 7)
                cells[r, q, 2:5] = F['cells'][br, bq, 2:5]
                break
            frontier = nxt

    # Territories: Dijkstra over the province land.
    keys = list(group)
    # Value noise in the step cost: equal-cost bisectors are straight lines.
    from terrain_noise import fbm
    qq0, rr0 = np.meshgrid(np.arange(W), np.arange(H))
    cost = np.where((cells[:, :, 2] & 8) > 0, IMPASSABLE_COST, 1.0) * (1 + BORDER_NOISE * fbm(qq0, rr0, (0, 0), 9, 3, seed=211))
    dist = np.full((H, W), np.inf); near = np.full((H, W), -1, np.int32); heap = []
    for i, k in enumerate(keys):
        q, r = towns[k]; d0 = -args.head_start if k == args.region else 0.
        dist[r, q] = d0; near[r, q] = i; heap.append((d0, q, r, i))
    heapq.heapify(heap)
    while heap:
        d, q, r, i = heapq.heappop(heap)
        if d > dist[r, q] or near[r, q] != i:
            continue
        for nq, nr in neighbours(q, r):
            if 0 <= nq < W and 0 <= nr < H and zone[nr, nq]:
                nd = d + .5 * (cost[r, q] + cost[nr, nq])
                if nd < dist[nr, nq]:
                    dist[nr, nq] = nd; near[nr, nq] = i; heapq.heappush(heap, (nd, nq, nr, i))
    # Land pockets cut off by water (lake beaches between towns) keep their region.
    if (zone & (near < 0)).any():
        print(int((zone & (near < 0)).sum()), 'province hexes unreachable over land keep their current region')
        zone = zone & (near >= 0)
    for i, k in enumerate(keys):
        ids[zone & (near == i)] = index[k]

    # New stamp, opened surroundings and a corridor to the capital.
    q0, r0 = towns[args.region]
    for dq, d2r, oq, orr in old_stamp:
        q = q0 + dq; r = (2 * r0 + q0 % 2 + d2r - q % 2) // 2
        if not zone[r, q]:
            raise RuntimeError(f'stamp cell {(q, r)} leaves the province land')
        ids[r, q] = index[args.region]
        cells[r, q, 0] = (cells[r, q, 0] & 0xF8) | (F['cells'][orr, oq, 0] & 7)
        cells[r, q, 2:5] = F['cells'][orr, oq, 2:5]
        cells[r, q, 2] &= np.uint8(0xF7)
    qq, rr = np.meshgrid(np.arange(W), np.arange(H))
    x0, y0 = hex_xy(q0, r0); x, y = hex_xy(qq, rr)
    disc = zone & (np.hypot(x - x0, y - y0) <= OPEN_RADIUS)
    cells[disc, 2] &= np.uint8(0xF7)
    corridor = [(q, r) for q, r in line_cells(towns[args.region], cap_town or towns[capital], H, W) if zone[r, q]]
    for q, r in corridor:
        cells[r, q, 2] &= np.uint8(0xF7)
    # Bytes 2:5 are copied whole above, and byte 3 bits 1-6 are the road edge mask: a copied mask
    # points at neighbours that have no partner bit (one-sided road edges). Every hex keeps its own
    # road mask; only the sprawl (bit 0) and ford (bit 7) bits move with the stamp.
    cells[:, :, 3] = (cells[:, :, 3] & 0x81) | (F['cells'][:, :, 3] & 0x7E)
    map_hex.set_region_ids(cells, ids)

    # Checks.
    final_owner = np.array(names + ['?'], dtype=object)[map_hex.region_ids(cells)]
    outside_same = bool((final_owner[~zone] == owner[~zone]).all() and (cells[~zone] == F['cells'][~zone]).all())
    counts = {k: int((final_owner == k).sum()) for k in keys}
    no_town = [k for k in keys if not ((final_owner == k) & ((cells[:, :, 2] >> 4) == 1)).any()]
    # Port hexes (slot 2) do not move with the stamp: the re-partition must leave each region its own
    # port hex count, or its port slot template no longer matches the map.
    ports = {k: [int(((owner == k) & (slot == 2)).sum()), int(((final_owner == k) & ((cells[:, :, 2] >> 4) == 2)).sum())] for k in keys}
    port_changed = {k: v for k, v in ports.items() if v[0] != v[1]}
    report = {'region': args.region, 'from': old, 'to': [q0, r0], 'hexes': counts, 'corridor': len(corridor),
              'opened': int(disc.sum()), 'outside_province_identical': outside_same, 'regions_without_primary_slot': no_town,
              'port_hexes_changed': port_changed,
              'one_sided_road_edges': [one_sided_road_edges(F['cells']), one_sided_road_edges(cells)]}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    preview(owner, final_owner, cells, keys, towns, old, zone)
    if port_changed:
        raise RuntimeError(f'move changes which regions own port hexes ([before, after] per region: {port_changed}); '
                           'their port slot templates would not match the map; nothing written')
    if not outside_same or no_town or min(counts.values()) <= 100 or \
            report['one_sided_road_edges'][1] > report['one_sided_road_edges'][0]:
        raise RuntimeError('move fails its checks; nothing written')
    if not args.apply:
        print('Dry run only; add --apply to write map.hex and the design.')
        return
    report['border_masks_refreshed'] = refresh_border_masks(cells, map_hex.region_ids(F['cells']))
    back = A / f'migration-move-town-{time.strftime("%Y%m%d-%H%M%S")}'
    back.mkdir(parents=True)
    for p in (FINAL, cw.DESIGN_PATH):
        shutil.copy2(p, back / p.name)
    map_hex.write(dict(F, cells=cells), FINAL)
    map_hex.read(FINAL)
    moved['relocation_previous_town'] = old
    moved['town'] = [q0, r0]
    if 'climate_seed' in moved:
        moved['climate_seed'] = [q0, r0]
    moved['layout_revision_reason'] = args.reason
    write_text_atomic(cw.DESIGN_PATH, json.dumps(design, ensure_ascii=False, indent=2) + '\n')
    report['backup'] = str(back)
    write_text_atomic(A / f'analysis/move-town-{args.region}.json', json.dumps(report, ensure_ascii=False, indent=1))
    print('Moved; backup at', back)


def preview(before, after, cells, keys, towns, old, zone):
    from PIL import Image, ImageDraw
    rr, qq = np.nonzero(zone)
    q0, q1, r0, r1 = qq.min() - 10, qq.max() + 11, rr.min() - 10, rr.max() + 11
    pal = {k: np.array(c) for k, c in zip(keys, [(214, 96, 77), (90, 160, 90), (90, 120, 200), (220, 180, 60), (160, 90, 180), (60, 170, 170)])}
    panels = []
    for own in (before, after):
        sub = own[r0:r1, q0:q1]; rgb = np.full(sub.shape + (3,), 225, np.uint8)
        rgb[(cells[r0:r1, q0:q1, 0] & 3) == 1] = (90, 130, 180)
        for k, c in pal.items():
            rgb[sub == k] = c
        imp = (cells[r0:r1, q0:q1, 2] & 8) > 0
        rgb[imp] = (rgb[imp] * .7).astype(np.uint8)
        im = Image.fromarray(rgb[::-1]).resize(((q1 - q0) * 3, int((r1 - r0) * 3 * cw.HEX_DZ / cw.HEX_DX)), Image.Resampling.NEAREST)
        d = ImageDraw.Draw(im); Hh = im.size[1]
        for k in keys:
            tq, tr = towns[k]; x, y = (tq - q0) * 3, Hh - (tr - r0) * 3 * cw.HEX_DZ / cw.HEX_DX
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 0, 0))
        x, y = (old[0] - q0) * 3, Hh - (old[1] - r0) * 3 * cw.HEX_DZ / cw.HEX_DX
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 0, 0))
        x = (cw.PAD_LEFT - q0) * 3; d.line([(x, 0), (x, Hh)], fill=(255, 0, 0))
        panels.append(im)
    out = Image.new('RGB', (panels[0].size[0] * 2 + 6, panels[0].size[1]), (255, 255, 255))
    out.paste(panels[0], (0, 0)); out.paste(panels[1], (panels[0].size[0] + 6, 0))
    out.save(ROOT / 'output/move-town-preview.png')


if __name__ == '__main__':
    main()

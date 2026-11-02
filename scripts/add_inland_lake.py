"""Add an inland lake (sea_lake hexes) to the expanded 190 campaign map.

Dry run by default: reports the hexes, region budget impact, terrain/water
prediction and every file that would change. Nothing is written without --apply.

What a vanilla inland lake is (analysis/inland-lake-research.md):
  * map.hex: sea flag (byte0 & 3 == 1), owner 3k_main_sea_lake (ONE sea region
    shared by all six vanilla lakes, disjoint), ground sea_lake, byte2 = 8,
    bytes 3-4 and 8-14 zero, byte7 = the surrounding land climate,
    byte15 = region-border edge mask (exact rule, verified on vanilla).
  * terrain: an LF depression below the water surface;
  * water: the five big lakes are campaign_lake_0N.wsmodel props, the high
    lake (19 hexes, LF ~5) one scaled small_lake_1.wsmodel prop, each in a
    bmd_objects.3k_main_sea_lake.<n>.31.bin leaf of global_props.bin.

Tier 1 (what --apply does): only the map.hex part. The terrain pipeline then
treats the hexes as sea: polish_campaign_terrain.py rolls the relief down to
-2 under them (relief=-2+(relief+2)*shore) and the ocean background plane that
extend_campaign_ocean_background.py adds to sea_mesh_0 (y -1.049, whole world)
shows through, exactly like the canvas seas (Korea/Japan/Wa Nu coasts).
Afterwards: all_directions_build.py database, then
build_polished_campaign_terrain.ps1 -RebuildGeometry (pathfinding, borders,
lookup, terrain), then a live check.

Tier 2 (vanilla-style lake prop at its own altitude) is only reported: it
needs global_props.bin additions plus pipeline changes (see the notes).

  python add_inland_lake.py bosten --preset bosten
  python add_inland_lake.py mylake --latlon "42.1,86.8;42.1,87.3;41.9,87.3;41.9,86.8"
  python add_inland_lake.py mylake --hex "430,905;440,905;440,915;430,915"
  python add_inland_lake.py bosten --preset bosten --apply
"""
import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import campaign_world as cw
import map_hex

A = cw.WORKSPACE
FINAL = A / 'work/final-map/map.hex'
REGISTRY = A / 'work/inland-lakes.json'
SURFACE = A / f'analysis/polished-surface-{cw.TERRAIN_VARIANTS[-1]}.npy'
TIMING = A / 'analysis/pipeline-timing.log'
LAKE_REGION = '3k_main_sea_lake'
SDK_MAX_REGIONS = 511
OCEAN_PLANE_Y = -1.049016            # extend_campaign_ocean_background.py
SEA_BED = -2.0                       # polish: relief=-2+(relief+2)*shore
TOWN_INNER = {True: 8, False: 5}     # polish town clearance inner radius (hexes)
# small_lake_1.rigid_model_v2 LOD0 half extents (x, z), flat at y 0.
SMALL_LAKE_HALF = (3.08, 2.71)
VANILLA_LAKE_Y_SCALE = .42           # every vanilla lake leaf scales y by 0.42

# Approximate outlines (lat, lon). Coarse; check them against a real map.
PRESETS = {
    'bosten': [(42.12, 86.80), (42.15, 87.05), (42.08, 87.35), (41.95, 87.40), (41.85, 87.20), (41.83, 86.95), (41.90, 86.78)],
    'lopnur': [(40.75, 90.15), (40.80, 90.50), (40.55, 90.85), (40.20, 90.80), (40.05, 90.55), (40.20, 90.20), (40.50, 90.05)],
    'issykkul': [(42.65, 76.25), (42.80, 77.00), (42.70, 77.80), (42.55, 78.30), (42.25, 78.25), (42.15, 77.50), (42.25, 76.70), (42.40, 76.25)],
}

# map.hex byte15: one bit per neighbour whose region differs (vanilla: exact).
# Order N, S, NE, SE, NW, SW for odd-q columns with south-up rows.
EDGE_BITS = (4, 32, 8, 16, 128, 64)


def neighbours(q, r):
    o = q % 2
    return ((q, r + 1), (q, r - 1), (q + 1, r + o), (q + 1, r + o - 1), (q - 1, r + o), (q - 1, r + o - 1))


def edge_mask(ids, q, r):
    h, w = ids.shape
    m = 0
    for bit, (a, b) in zip(EDGE_BITS, neighbours(q, r)):
        if 0 <= a < w and 0 <= b < h and ids[b, a] != ids[r, q]:
            m |= bit
    return m


def parse_pairs(text):
    return [tuple(float(v) for v in p.split(',')) for p in text.replace(' ', '').split(';') if p]


def polygon_hexes(poly, w, h):
    """Hexes whose centre (q, r + q%2/2) lies inside the polygon (hex units)."""
    poly = np.asarray(poly, float)
    q0, q1 = int(np.floor(poly[:, 0].min())) - 1, int(np.ceil(poly[:, 0].max())) + 1
    r0, r1 = int(np.floor(poly[:, 1].min())) - 1, int(np.ceil(poly[:, 1].max())) + 1
    qq, rr = np.meshgrid(np.arange(max(q0, 0), min(q1, w - 1) + 1), np.arange(max(r0, 0), min(r1, h - 1) + 1))
    x, y = qq.astype(float), rr + (qq % 2) * .5
    inside = np.zeros(qq.shape, bool)
    for (xa, ya), (xb, yb) in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (ya > y) != (yb > y)
        xi = xa + (y - ya) * (xb - xa) / np.where(yb == ya, 1e-12, yb - ya)
        inside ^= crosses & (x < xi)
    return sorted(zip(qq[inside].tolist(), rr[inside].tolist()))


def components(cells_set):
    seen, comps = set(), []
    for start in cells_set:
        if start in seen:
            continue
        comp, dq = [], collections.deque([start])
        seen.add(start)
        while dq:
            c = dq.popleft()
            comp.append(c)
            for n in neighbours(*c):
                if n in cells_set and n not in seen:
                    seen.add(n)
                    dq.append(n)
        comps.append(comp)
    return comps


def predict_terrain(cells_after, lake, margin=24):
    """Replay polish's coastal roll-off locally on the current polished surface."""
    if not SURFACE.exists():
        return None
    surface = np.load(SURFACE, mmap_mode='r')
    qs, rs = [q for q, _ in lake], [r for _, r in lake]
    tx0, tx1 = max(0, 2 * min(qs) - margin), min(surface.shape[1], 2 * max(qs) + 2 + margin)
    ty0, ty1 = max(0, 2 * min(rs) - margin), min(surface.shape[0], 2 * max(rs) + 3 + margin)
    ty, tx = np.mgrid[ty0:ty1, tx0:tx1]
    hq, hr = cw.tile_hex(tx, ty)
    physical = (cells_after[hr, hq, 0] & 3) != 1
    coast = np.asarray(Image.fromarray((physical * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    shore = np.clip((coast - .25) / .5, 0, 1)
    shore = shore * shore * (3 - 2 * shore)
    before = np.asarray(surface[ty0:ty1, tx0:tx1], np.float32)
    after = SEA_BED + (before + 2) * shore
    lake_set = set(lake)
    wet = [(q, r) for q, r in lake
           if float(after[2 * r + q % 2 - ty0:2 * r + q % 2 + 2 - ty0, 2 * q - tx0:2 * q + 2 - tx0].mean()) < OCEAN_PLANE_Y]
    ring = {n for c in lake for n in neighbours(*c) if n not in lake_set}
    ring_h = [float(before[2 * r + q % 2 - ty0, 2 * q - tx0]) for q, r in ring
              if ty0 <= 2 * r + q % 2 < ty1 and tx0 <= 2 * q < tx1]
    return {'surface_file': str(SURFACE), 'surface_mtime': time.strftime('%Y-%m-%d %H:%M', time.localtime(SURFACE.stat().st_mtime)),
            'current_height_under_lake': [round(float(np.percentile(before[(~physical)], p)), 3) for p in (5, 50, 95)] if (~physical).any() else None,
            'shore_ring_height_pct5_50_95': [round(float(np.percentile(ring_h, p)), 3) for p in (5, 50, 95)] if ring_h else None,
            'hexes_predicted_under_ocean_plane': len(wet), 'hexes_in_lake': len(lake),
            'note': 'local replay of polish relief=-2+(relief+2)*smoothstep((blur(land,3)-.25)/.5); '
                    'water shows where the result is below the ocean plane y -1.049. Approximate (town clearance and approach grading not replayed).'}


def prop_proposal(lake, ring_heights):
    """Tier 2: a scaled small_lake_1 instance covering the lake (report only)."""
    pts = np.array([cw.block_centre(q, r) for q, r in lake], float)   # tile block centres, not SW corners
    centre = pts.mean(axis=0)
    if len(pts) > 2:
        evals, evecs = np.linalg.eigh(np.cov((pts - centre).T))
        major = evecs[:, np.argmax(evals)]
    else:
        major = np.array([1., 0.])
    minor = np.array([-major[1], major[0]])
    half_major = float(np.abs((pts - centre) @ major).max()) + cw.DX
    half_minor = float(np.abs((pts - centre) @ minor).max()) + cw.DX
    sx, sz = half_major / SMALL_LAKE_HALF[0], half_minor / SMALL_LAKE_HALF[1]
    row0 = [sx * major[0], 0., sx * major[1]]
    row2 = [sz * minor[0], 0., sz * minor[1]]
    surface_y = None if not ring_heights else round(ring_heights[0] - .25, 3)
    return {'model': 'rigidmodels/campaign/props/lakes/small_lake_1.wsmodel',
            'world_centre_xz': [round(float(v), 3) for v in centre],
            'matrix_3x3_rows_assumed': [[round(v, 4) for v in row0], [0., VANILLA_LAKE_Y_SCALE, 0.], [round(v, 4) for v in row2]],
            'suggested_surface_y': surface_y,
            'basis': 'vanilla high lake (19 hexes, LF ~4.9, shore ~6.2): one small_lake_1 at y 5.97 with matrix '
                     '[-0.8,0,0.75 / 0,0.42,0 / -0.75,0,-0.8] in bmd_objects.3k_main_sea_lake.475.31.bin'}


def pipeline_busy():
    if not TIMING.exists():
        return False
    lines = TIMING.read_text(encoding='utf-8', errors='replace').strip().splitlines()
    recent = time.time() - TIMING.stat().st_mtime < 3 * 3600
    return recent and bool(lines) and '\tTOTAL\t' not in lines[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', help='lake key, e.g. bosten')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--latlon', help='polygon "lat,lon;lat,lon;..." (projected with plan_western_regions.project)')
    g.add_argument('--hex', help='polygon "q,r;q,r;..." in canvas hexes')
    g.add_argument('--preset', choices=sorted(PRESETS))
    ap.add_argument('--region', default=LAKE_REGION, help='sea region to own the hexes (default: the vanilla lake region)')
    ap.add_argument('--map', type=Path, default=FINAL)
    ap.add_argument('--allow-town-proximity', action='store_true', help='do not fail on hexes inside a town clearance radius')
    ap.add_argument('--apply', action='store_true', help='write map.hex (Tier 1) with a backup and a registry entry')
    ap.add_argument('--force', action='store_true', help='apply even if the terrain pipeline timing log looks mid-run')
    args = ap.parse_args()

    m = map_hex.read(args.map)
    cells = m['cells']
    h, w = cells.shape[:2]
    names = m['land_regions'] + m['sea_regions']
    ids = map_hex.region_ids(cells)

    if args.hex:
        poly = parse_pairs(args.hex)
        source = {'hex_polygon': poly}
    else:
        import plan_western_regions as pw
        ll = parse_pairs(args.latlon) if args.latlon else PRESETS[args.preset]
        poly = [pw.project(lat, lon) for lat, lon in ll]
        source = {'latlon_polygon': ll, 'projected_hex_polygon': [[round(a, 2), round(b, 2)] for a, b in poly],
                  'projection': 'plan_western_regions.project (west scale %.2f about Dunhuang)' % pw.WEST_SCALE}
    lake = polygon_hexes(poly, w, h)
    problems, warnings = [], []
    if not lake:
        problems.append('polygon contains no hex centre')

    # ---- per-hex eligibility ----
    native = np.zeros((h, w), bool)
    native[cw.NATIVE_HEX_SLICE] = True
    import revealed_filler
    native &= ~revealed_filler.revealed_hex(cells, names)   # the revealed filler (vanilla non-playable land now playable) is canvas
    flag = cells[..., 0] & 3
    slot = cells[..., 2] >> 4
    rejected = collections.Counter()
    for q, r in lake:
        if native[r, q]:
            rejected['inside the native (vanilla) map rectangle, not revealed filler'] += 1
        if flag[r, q] == 1:
            rejected['already sea'] += 1
        elif flag[r, q] in (2, 3):
            rejected['river hex (land/sea flag 2/3)'] += 1
        if slot[r, q]:
            rejected['settlement or port slot'] += 1
        if cells[r, q, 3] & 1:
            rejected['town plot flag'] += 1
        if cells[r, q, 8] or cells[r, q, 9]:
            rejected['area-of-interest marker'] += 1
    already_lake = sum(1 for q, r in lake if flag[r, q] == 1 and names[ids[r, q]] == args.region)
    for why, n in rejected.items():
        if why == 'already sea' and n == already_lake == len(lake):
            continue
        problems.append(f'{n} hexes: {why}')

    # ---- region budget ----
    total = len(names)
    new_region = args.region not in m['sea_regions']
    if args.region in m['land_regions']:
        problems.append(f'{args.region} is a land region')
    budget = {'regions_now': total, 'sdk_maximum': SDK_MAX_REGIONS, 'left_now': SDK_MAX_REGIONS - total,
              'region': args.region, 'new_region': new_region, 'regions_after': total + new_region,
              'left_after': SDK_MAX_REGIONS - total - new_region}
    if new_region:
        problems.append(f'{args.region} is not a map.hex sea region: a new sea region needs DB rows (regions, campaign_map_regions; '
                        'clone 3k_main_sea_lake as prepare_province.py does for 3k_main_sea_non_playable), a design new_sea_regions '
                        'entry (audit_campaign_polish.py counts it) and a map.hex sea list entry. Not implemented; reuse 3k_main_sea_lake.')

    # ---- owners losing hexes, contiguity ----
    lake_set = set(lake)
    losing = collections.Counter(names[ids[r, q]] for q, r in lake if flag[r, q] != 1)
    split = []
    for region, n in losing.items():
        if 'non_playable' in region:
            continue
        mine = set(zip(*np.nonzero(ids == names.index(region))[::-1]))
        before = len(components(mine))
        after_set = mine - lake_set
        if not after_set:
            problems.append(f'{region} would lose all its hexes')
            continue
        after = len(components(after_set))
        if after > before:
            split.append(region)
        if n > .3 * len(mine):
            warnings.append(f'{region} loses {n} of {len(mine)} hexes ({n / len(mine):.0%})')
    if split:
        problems.append(f'land regions made non-contiguous: {split}')
    lake_parts = components(lake_set)
    if len(lake_parts) > 1:
        warnings.append(f'lake polygon gives {len(lake_parts)} disconnected hex groups')

    # ---- towns ----
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    near = []
    for t in design['regions']:
        tq, tr = t['town']
        d = min(np.hypot(q - tq, r + q % 2 * .5 - tr - tq % 2 * .5) for q, r in lake) if lake else 1e9
        if d < 20:
            near.append({'region': t['region'], 'town': t['town'], 'distance_hex': round(float(d), 1),
                         'clearance_inner': TOWN_INNER[bool(t.get('is_capital'))], 'battle_layout': t.get('battle_layout')})
            if d < TOWN_INNER[bool(t.get('is_capital'))]:
                (warnings if args.allow_town_proximity else problems).append(
                    f'{t["region"]} town {t["town"]} is {d:.1f} hexes from the lake (polish keeps a {TOWN_INNER[bool(t.get("is_capital"))]}-hex level apron around it)')
    near.sort(key=lambda v: v['distance_hex'])

    # ---- cell encoding (vanilla lake cell pattern) ----
    after = cells.copy()
    ground = m['land_ground_types'] + m['sea_ground_types']
    v = ground.index('sea_lake') + 1
    if args.region in names:
        rid = names.index(args.region)
        lake_rc = (np.array([r for _, r in lake], int), np.array([q for q, _ in lake], int))
        sub = after[lake_rc]
        sub[:, 0] = (sub[:, 0] & ~np.uint8(7)) | 1
        sub[:, 2] = 8
        sub[:, 3] = 0
        sub[:, 4] = 0
        sub[:, 5] = np.uint8((v & 15) << 4)
        sub[:, 6] = np.uint8((v >> 4) & 7)
        sub[:, 8:15] = 0
        after[lake_rc] = sub
        new_ids = map_hex.region_ids(after)
        new_ids[lake_rc] = rid
        map_hex.set_region_ids(after, new_ids)
        touched = set(lake)
        for c in lake:
            touched.update(n for n in neighbours(*c) if 0 <= n[0] < w and 0 <= n[1] < h)
        for q, r in touched:
            after[r, q, 15] = edge_mask(new_ids, q, r)
        changed_cells = int((after != cells).any(axis=2).sum())
    else:
        touched, changed_cells = set(), 0

    terrain = predict_terrain(after, lake) if lake and not new_region else None
    if terrain and terrain['hexes_predicted_under_ocean_plane'] == 0:
        problems.append('no hex is predicted to drop below the ocean plane: the lake is too narrow for the 3-tile coastal blur')
    elif terrain and terrain['hexes_predicted_under_ocean_plane'] < .5 * len(lake):
        warnings.append(f'only {terrain["hexes_predicted_under_ocean_plane"]}/{len(lake)} hexes predicted under water (narrow lake)')
    ring_pct = terrain and terrain.get('shore_ring_height_pct5_50_95')

    report = {
        'lake': args.name, 'source': source, 'map': str(args.map), 'map_size': [w, h],
        'hexes': len(lake), 'hex_bbox_qr': [min(q for q, _ in lake), max(q for q, _ in lake), min(r for _, r in lake), max(r for _, r in lake)] if lake else None,
        'hex_list': [list(c) for c in lake],
        'owners_losing_hexes': dict(losing), 'region_budget': budget, 'towns_within_20_hexes': near,
        'cell_encoding': {'byte0': 'sea flag 1 + region id', 'byte2': 8, 'bytes3_4': 0, 'byte5_6': f'ground sea_lake ({v - 1})',
                          'byte7': 'kept (climate)', 'bytes8_14': 0, 'byte15': 'region-border edge mask recomputed on lake + neighbours',
                          'cells_changed': changed_cells, 'cells_with_byte15_recomputed': len(touched)},
        'terrain_prediction': terrain,
        'files_written_by_apply': [str(args.map), str(A / 'backups/inland-lakes') + '/<stamp>/map.hex (backup)', str(REGISTRY)],
        'files_rewritten_downstream': {
            'all_directions_build.py database': 'kit map.hex copy, startpos/DB build (region list unchanged when reusing 3k_main_sea_lake)',
            'build_polished_campaign_terrain.ps1 -RebuildGeometry': 'SDK pathfinding/borders/trade-routes/lookup (Tw3kBuild), display/regions, '
                'lf_height_map.dds/.compressed_map, global_meshes/land_mesh_*.rigid_model_v2/.compressed_map, global_map blend, climate, '
                'camera_heightmap.png, tile caches, strategic/minimap art; sea_mesh_0 background plane is unchanged and supplies the water',
        },
        'tier2_vanilla_style_prop_not_applied': {
            'proposal': prop_proposal(lake, ring_pct) if lake else None,
            'needs': ['global_props.bin: clone root record bmd_objects.3k_main_sea_lake.475.bin (identity matrix, region 3k_main_sea_lake), '
                      'its parent prefab (child path renamed) and leaf .475.31.bin with the proposal matrix in absolute expanded world '
                      'coordinates, unused prefab number; also as loose files, after complete_campaign_resource_scenes.py (which deletes loose '
                      'bmd_objects not in its bundle)',
                      'translate_campaign_lake_instances.py: whitelist authored lake leaves (it asserts every lake leaf is vanilla and the '
                      'counts are exactly 79 files / 97 instances)',
                      'polish_campaign_terrain.py: per-lake bed/shore target around the prop surface y instead of the fixed -2 sea bed '
                      '(otherwise the prop floats ~2 units above a pit already filled by the ocean plane)',
                      'optional: lake colour in sea_colour.dds after expand_campaign_water_colour.py (canvas is filled with the open-ocean median)'],
        },
        'problems': problems, 'warnings': warnings, 'mode': 'apply' if args.apply else 'dry-run',
    }

    if args.apply:
        if problems:
            print(json.dumps(report, ensure_ascii=False, indent=1))
            sys.exit('Refusing --apply: ' + '; '.join(problems))
        if pipeline_busy() and not args.force:
            sys.exit(f'Refusing --apply: {TIMING} has no TOTAL line yet and changed recently (terrain pipeline running?). Use --force once it is idle.')
        stamp = time.strftime('%Y%m%d-%H%M%S')
        backup = A / 'backups/inland-lakes' / stamp / 'map.hex'
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.map, backup)
        m['cells'] = after
        map_hex.write(m, args.map)
        check = map_hex.read(args.map)
        cids = map_hex.region_ids(check['cells'])
        assert all((check['cells'][r, q, 0] & 3) == 1 and names[cids[r, q]] == args.region for q, r in lake)
        assert ((check['cells'] != cells).any(axis=2).sum()) == changed_cells
        reg = json.loads(REGISTRY.read_text(encoding='utf-8')) if REGISTRY.exists() else {'lakes': []}
        reg['lakes'] = [l for l in reg['lakes'] if l['name'] != args.name] + [{
            'name': args.name, 'region': args.region, 'source': source, 'hexes': [list(c) for c in lake],
            'applied': stamp, 'backup': str(backup), 'tier': 1, 'live_verified': False}]
        REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding='utf-8')
        report['backup'] = str(backup)
        report['next'] = 'python all_directions_build.py database; build_polished_campaign_terrain.ps1 -RebuildGeometry; live check'
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if problems and not args.apply:
        print('DRY RUN: problems listed above would block --apply', file=sys.stderr)


if __name__ == '__main__':
    main()

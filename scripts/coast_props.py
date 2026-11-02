"""Coast scenery for the new regions: docks at the new ports, coastal-wave sound emitters on the new coasts.

What vanilla does (3k_dlc07_main_map; staged global_props.bin, vanilla map.hex and DB):
  * a port is a land hex with settlement slot 2 (map.hex byte 2 high nibble) beside water whose
    own sea hexes carry slot 2 too (each vanilla port has its own '<sea>_to_<town>' sea region);
    the region then has a 'port' slot template (3k_minor_port, 3k_province_capital_port) whose
    harbour building (wa_harbour.bmd) the game draws itself, and a port capital uses a port
    settlement layout (campaign_settlement_display_settlement_layouts: settlement_port_layout_1/2).
    prepare_province.py already applies that rule to the new regions (port slot templates follow
    the map's port hexes, layouts follow the donor), so it is in the database;
  * the static harbour scenery around a port is global_props.bin prefabs: one in the town's region
    (piers: port_wood_walkway*, port_scaffold_bamboo_1, port_fence_rough*, barrels, crates, fishing
    nets) and one in the port's sea region (moored building_boat_*_anim, fisher_man.csc, fish_anim,
    a coastal-wave sound), every leaf authored at absolute coordinates (world_space_leaves.py);
  * no vanilla coast carries surf or wave models or particle emitters (3k_main_campaign_enviro_waves
    is an empty stub placed nowhere; the shoreline foam is the ocean shader's). The coasts' wave
    effect is ambient sound: 68 'Sound_Environment_Coastal_Waves_Rock/_Sand' emitters (53 points,
    15 line lists of 2-11 vertices) in global_props leaves, one point per ~28 vanilla sea-coast hexes, Rock on rocky and
    most beach shores ~1.3 hexes off the coast hex centre at y ~0.25, Sand only on beaches at the
    water's edge. Sound records keep vanilla-frame coordinates under their root's translation (the
    world-space bake moves models, emitters and scenes only), so new ones are written the same way.

What this step does, on the staged bundle after the prefab chain (after hehuang_lake_prop.py):
  * docks: every new region (tw3k_*) with a land port hex gets the scenery of a vanilla port of its
    kind (its primary slot in the design): trading port <- 3k_main_taishan_resource_1, fishing port
    <- 3k_main_yizhou_island_resource_1, port slot only (3k_minor_port) <- the pier of
    3k_main_youbeiping_resource_1; a design fishing port with no port hex but the sea within
    FISH_REACH hexes of its town gets the fishing scenery at its coast hex nearest the town. The
    vanilla leaves are cloned whole (byte-exact but for the transforms) under new prefab numbers in
    the new land / sea region with one rigid transform per port, a turn about the vertical and a
    shift, chosen so the pieces standing in water in vanilla stand in water here (final LF below
    WATER_Y), the land pieces on low land and off the town models; heights follow max(ground, water);
  * waves: new sea coasts (land hexes beside navigable sea that were not vanilla coast) within
    WAVE_NEAR hexes of a new playable region get single-point coastal-wave sounds cloned from vanilla
    single-emitter leaves, spaced evenly at the vanilla point rate per coast hex, Sand on beach hexes and
    Rock elsewhere, placed on the shore where the LF falls to the vanilla emitter height, none within
    WAVE_CLEAR hexes of a vanilla wave emitter;
  * the root records, parents and leaves are added to the bundle (a rerun replaces them) and written
    as loose files, as hehuang_lake_prop.py does; analysis/coast-props.json lists every placement and
    analysis/coast-props-markers.json the markers for render_campaign_view.py --markers.

  python coast_props.py [--apply]
"""
import argparse
import collections
import json
import math
import re
import struct

import numpy as np

import campaign_world as cw
import map_hex
import props_bundle as pb
from read_root_prefabs import decode
from final_terrain_surface import load_final_surface, sample_final_surface

A = cw.WORKSPACE
S = A / 'work/release-input'
WATER_Y = 0.0          # vanilla moored boats sit at y 0.02, fishermen 0.04: the sea plane
DOCK_BASE = 9800       # unused prefab numbers: DOCK_BASE + 10 * port + part
DOCK_PORTS = 20        # 9800-9999 (landmarks 9600-9799 below, nothing above): 13 new ports after the Korea rebuild
DOCK_MAX = 9999        # last dock number, a literal for river_water_props.OTHER_RANGES (read without importing)
assert DOCK_MAX == DOCK_BASE + 10 * DOCK_PORTS - 1
WAVE_BASE = 8000       # WAVE_BASE + emitter
FISH_REACH = 3         # hexes from a design fishing town to the sea
TOWN_CLEAR = 0.8       # hexes kept free round settlement and port slot hexes
SHIFT_MAX = 4.6        # hexes a dock may move off its port hex (new coasts reach the water ~2 hexes out)
WAVE_NEAR = 12         # hexes from new playable land within which new coasts get wave sounds
WAVE_CLEAR = 6         # hexes kept free round vanilla wave emitters
WAVE_HEIGHT = {'Rock': .25, 'Sand': 0.}   # vanilla medians: Rock y .245 on the shore, Sand -.09 at the edge
TEMPLATES = {
    'trade': ('3k_main_taishan_resource_1', [('3k_main_taishan_resource_1', 2302, ('31',)),
                                             ('3k_main_sea_huang_to_taishan_resource_1', 2302, ('16', '23'))]),
    'fish': ('3k_main_yizhou_island_resource_1', [('3k_main_yizhou_island_resource_1', 4546, ('31',)),
                                                  ('3k_main_sea_zhang_yizhou_island_resource_1', 4546, ('16', '23'))]),
    'pier': ('3k_main_youbeiping_resource_1', [('3k_main_sea_huang_to_youbeiping_resource_1', 1919, ('31',))]),
}
KIND_OF_SLOT = {'3k_resource_water_trading_port': 'trade', '3k_resource_wood_fish': 'fish'}
LANDLOCKED = {'3k_main_sea_lake', '3k_main_sea_non_playable'}
DIRS = (((0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)),
        ((0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)))
OURS = re.compile(r'bmd_objects\.[A-Za-z0-9_]+\.(\d+)(\.\d+)?\.bin$')


def ours(key):
    g = OURS.search(key)
    return bool(g) and (DOCK_BASE <= int(g.group(1)) < DOCK_BASE + 10 * DOCK_PORTS or WAVE_BASE <= int(g.group(1)) < WAVE_BASE + 1000)


def neighbours(q, r, W, H):
    for dq, dr in DIRS[q & 1]:
        a, b = q + dq, r + dr
        if 0 <= a < W and 0 <= b < H:
            yield a, b


def hex_dilate(mask):
    """Mask grown by one hex: out[r, q] |= mask at each neighbour (q, r) + DIRS[q & 1]."""
    out = mask.copy()
    H, W = mask.shape
    pad = np.zeros((H + 2, W + 2), bool)
    pad[1:-1, 1:-1] = mask
    for parity in (0, 1):
        cols = np.arange(parity, W, 2)
        for dq, dr in DIRS[parity]:
            out[:, cols] |= pad[1 + dr:H + 1 + dr, 1 + cols + dq]
    return out


def sounds(blob):
    """Sound emitters of a leaf: name, type, point count and the offset of the first point."""
    out = []
    for m in re.finditer(rb'Sound_[A-Za-z0-9_]+', blob):
        s = m.start()
        if struct.unpack_from('<H', blob, s - 2)[0] != len(m.group()):
            continue
        p = s + len(m.group())
        tl = struct.unpack_from('<H', blob, p)[0]
        typ = blob[p + 2:p + 2 + tl]
        if not typ.startswith(b'SST_'):
            continue
        p += 2 + tl
        n = struct.unpack_from('<I', blob, p)[0]
        out.append({'name': m.group().decode(), 'type': typ.decode(), 'count': n, 'points_offset': p + 4})
    return out


def leaf_rows(blob):
    """Placed records (models, emitters, composite scenes), sounds, and whether that is every record."""
    from translate_campaign_lake_instances import records as leaf_records
    from world_space_leaves import placed_records
    try:
        rows = leaf_records(blob)
    except Exception:
        rows = []
    rows += placed_records(blob)
    snd = sounds(blob)
    return rows, snd, len(rows) + len(snd) == len(re.findall(rb'BHM_[A-Z_]+', blob))


def clone_prefab(root, assets, pre, rec, leaves, owner, number, edit):
    """Root record, parent and leaves of one template prefab under a new owner and number.
    leaves: [(child record, suffix)] to keep; edit(suffix, leaf bytes) -> new leaf bytes."""
    stem = f'{pre}bmd_objects.{owner}.{number}'
    pbody = assets[rec['path']]
    cpos, children = decode(pbody)
    chunks, new = [], {}
    for ch, suf in leaves:
        key = f'{stem}.{suf}.bin'
        new[key] = edit(suf, assets[ch['path']])
        chunks.append(pb.rebind(pbody[ch['start']:ch['end']], ch, owner, key))
    parent, pc = pb.prefab_list(pbody, cpos, children, [], chunks)
    assert [x['path'] for x in pc] == [f'{stem}.{s}.bin' for _, s in leaves] and all(x['region'] == owner for x in pc)
    new[stem + '.bin'] = parent
    return pb.rebind(root[rec['start']:rec['end']], rec, owner, stem + '.bin'), new


class Ground:
    def __init__(self, variant):
        self.lf = load_final_surface(variant, S)

    def h(self, x, z):
        return sample_final_surface(self.lf, np.asarray(x, np.float64) / cw.DX * 2, np.asarray(z, np.float64) / cw.DZ * 2)


# ------------------------------------------------------------------ docks
def ports(m, design):
    """[(region, kind, anchor land hex, sea region, why)] of the new regions' ports."""
    c = m['cells']
    H, W = c.shape[:2]
    names = m['land_regions'] + m['sea_regions']
    ids = map_hex.region_ids(c)
    water = (c[..., 0] & 3) == 1
    slot = c[..., 2] >> 4
    sea_names = set(m['sea_regions']) - LANDLOCKED
    regions = {r['region']: r for r in design['regions']}
    out = []
    for r, q in zip(*np.nonzero((slot == 2) & ~water)):
        reg = names[ids[r, q]]
        if not reg.startswith('tw3k_'):
            continue
        prim = regions.get(reg, {}).get('gameplay_template', {}).get('primary_slot')
        nb = list(neighbours(int(q), int(r), W, H))
        seas = [names[ids[b, a]] for a, b in nb if water[b, a] and slot[b, a] == 2] or [names[ids[b, a]] for a, b in nb if water[b, a]]
        out.append((reg, KIND_OF_SLOT.get(prim, 'pier'), (int(q), int(r)), seas[0], f'port hex; primary slot {prim}'))
    have = {p[0] for p in out}
    for reg, d in regions.items():
        if reg in have or d.get('gameplay_template', {}).get('primary_slot') != '3k_resource_wood_fish' or reg not in names:
            continue
        tq, tr = d['town']
        best = None
        for b in range(max(0, tr - FISH_REACH - 1), min(H, tr + FISH_REACH + 2)):
            for a in range(max(0, tq - FISH_REACH - 1), min(W, tq + FISH_REACH + 2)):
                if water[b, a] or ids[b, a] != names.index(reg):
                    continue
                sea = [names[ids[y, x]] for x, y in neighbours(a, b, W, H) if water[y, x] and names[ids[y, x]] in sea_names]
                dd = math.dist(cw.display_position(a, b), cw.display_position(tq, tr)) / cw.HEX_DX
                if sea and dd <= FISH_REACH and (best is None or dd < best[0]):
                    best = (dd, (a, b), sea[0])
        if best:
            out.append((reg, 'fish', best[1], best[2], 'design fishing port, no port hex: its coast hex nearest the town'))
    return sorted(out)


def fit(inst, anchor_t, anchor, ground, avoid):
    """(score, turn deg, shift xz, water match, town overlap, high-land term) of the template pieces at the port."""
    P = np.array([(i['x'], i['z']) for i in inst]) - anchor_t
    ht = np.array([i['h'] for i in inst])
    wet_t = ht < WATER_Y
    w = np.clip(np.abs(ht - WATER_Y) / .2, .2, 1.)
    offs = np.array([(dx, dz) for dx in np.arange(-SHIFT_MAX, SHIFT_MAX + .01, .2) for dz in np.arange(-SHIFT_MAX, SHIFT_MAX + .01, .2)]) * cw.HEX_DX
    best = None
    for deg in range(0, 360, 5):
        t = math.radians(deg)
        cs, sn = math.cos(t), math.sin(t)
        R = np.stack((cs * P[:, 0] + sn * P[:, 1], -sn * P[:, 0] + cs * P[:, 1]), 1) + anchor
        X = R[None, :, 0] + offs[:, None, 0]
        Z = R[None, :, 1] + offs[:, None, 1]
        h = ground.h(X, Z)
        wet = h < WATER_Y
        match = ((wet == wet_t[None]) * w[None]).sum(1) / w.sum()
        high = (np.maximum(0, h - ht[None] - .3) * ~wet_t[None]).mean(1)
        near = np.zeros(len(offs))
        for ax, az in avoid:
            near += (np.hypot(X - ax, Z - az) < TOWN_CLEAR * cw.HEX_DX).mean(1)
        score = match - .5 * high - 2 * near - .01 * np.hypot(offs[:, 0], offs[:, 1]) / cw.HEX_DX
        i = int(score.argmax())
        if best is None or score[i] > best[0]:
            best = (float(score[i]), deg, offs[i], float(match[i]), float(near[i]), float(high[i]))
    return best


def moved(blob, rows, snd, root_t, rot, centre, target, ground):
    """Leaf bytes with every record turned by rot (deg) about centre and moved onto target, heights
    following max(ground, water); sound points keep the vanilla convention (point + root_t = world)."""
    t = math.radians(rot)
    cs, sn = math.cos(t), math.sin(t)
    Q = np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])        # world turn on column vectors

    def go(x, y, z):
        dx, dz = x - centre[0], z - centre[1]
        nx, nz = cs * dx + sn * dz + target[0], -sn * dx + cs * dz + target[1]
        return nx, y - max(float(ground.h(x, z)), WATER_Y) + max(float(ground.h(nx, nz)), WATER_Y), nz
    out = bytearray(blob)
    placed = []
    for r in rows:
        p = r['position_offset']
        m = np.array(struct.unpack_from('<12f', out, p - 36), np.float64)
        axes = m[:9].reshape(3, 3) @ Q.T                       # rows are the local axes in world space
        nx, ny, nz = go(*m[9:])
        struct.pack_into('<12f', out, p - 36, *axes.ravel(), nx, ny, nz)
        placed.append((r['model'].replace('\\', '/').split('/')[-1], round(nx, 3), round(ny, 3), round(nz, 3)))
    for s in snd:
        for i in range(s['count']):
            p = s['points_offset'] + 12 * i
            x, y, z = struct.unpack_from('<3f', out, p)
            nx, ny, nz = go(x + root_t[0], y + root_t[1], z + root_t[2])
            struct.pack_into('<3f', out, p, nx - root_t[0], ny - root_t[1], nz - root_t[2])
            placed.append((s['name'], round(nx, 3), round(ny, 3), round(nz, 3)))
    return bytes(out), placed


def family(name):
    return re.sub(r'(_\d+)?(_anim)?(_mid|_short|_halfsize|_quartersize|_support)?\.(rigid_model_v2|wsmodel|csc)$', '', name)


def place_docks(ctx, targets):
    m, assets, root, recs, pre, ground = ctx['map'], ctx['assets'], ctx['root'], ctx['recs'], ctx['pre'], ctx['ground']
    c = m['cells']
    names = m['land_regions'] + m['sea_regions']
    ids = map_hex.region_ids(c)
    slot, water = c[..., 2] >> 4, (c[..., 0] & 3) == 1
    avoid_all = [cw.display_position(int(q), int(r)) for r, q in zip(*np.nonzero((slot > 0) & ~water))]
    by_path = {r['path']: r for r in recs}
    new_recs, new_assets, report = [], {}, []
    for pi, (reg, kind, hexq, sea, why) in enumerate(targets):
        # Dock prefabs are numbered DOCK_BASE + 10 * port + part inside DOCK_BASE..DOCK_BASE + 10 * DOCK_PORTS - 1
        # (the range a rerun recognises as ours): one port more would take another range's numbers.
        assert pi < DOCK_PORTS, f'{len(targets)} new ports: dock numbering DOCK_BASE + 10 * port holds {DOCK_PORTS}'
        src_region, parts = TEMPLATES[kind]
        assert len(parts) <= 10, (kind, len(parts))
        sq, sr = next((int(q), int(r)) for r, q in zip(*np.nonzero((slot == 2) & ~water & (ids == names.index(src_region)))))
        anchor_t, anchor = np.array(cw.display_position(sq, sr)), np.array(cw.display_position(*hexq))
        plan, inst = [], []
        for path_region, number, suffixes in parts:
            rec = by_path[f'{pre}bmd_objects.{path_region}.{number}.bin']
            t = rec['matrix'][12:15]
            keep = []
            for ch in decode(assets[rec['path']])[1]:
                suf = ch['path'].rsplit('.', 2)[-2]
                if suf not in suffixes:
                    continue
                rows, snd, whole = leaf_rows(assets[ch['path']])
                assert whole, ('leaf with records this step cannot move', ch['path'])
                keep.append((ch, suf, rows, snd))
                for r in rows:
                    inst.append({'x': r['matrix'][9], 'z': r['matrix'][11]})
                for s in snd:
                    for i in range(s['count']):
                        x, _, z = struct.unpack_from('<3f', assets[ch['path']], s['points_offset'] + 12 * i)
                        inst.append({'x': x + t[0], 'z': z + t[2]})
            plan.append((path_region, rec, keep))
        for i in inst:
            i['h'] = float(ground.h(i['x'], i['z']))
        score, rot, off, match, near, high = fit(inst, anchor_t, anchor, ground, [a for a in avoid_all if math.dist(a, anchor) < 6 * cw.HEX_DX])
        target = anchor + off
        placed_all = []
        for part_i, (path_region, rec, keep) in enumerate(plan):
            owner = sea if '_sea_' in path_region else reg
            info = {k[1]: k for k in keep}

            def edit(suf, blob, info=info, rec=rec):
                body, placed = moved(blob, info[suf][2], info[suf][3], rec['matrix'][12:15], rot, anchor_t, target, ground)
                placed_all.extend(placed)
                return body
            r_, n_ = clone_prefab(root, assets, pre, rec, [(k[0], k[1]) for k in keep], owner, DOCK_BASE + 10 * pi + part_i, edit)
            new_recs.append(r_)
            new_assets.update(n_)
        report.append({'region': reg, 'kind': kind, 'why': why, 'anchor_hex': list(hexq), 'sea_region': sea, 'template': src_region,
                       'template_port_hex': [sq, sr], 'turn_deg': rot, 'shift_hexes': [round(float(v) / cw.HEX_DX, 2) for v in off],
                       'water_match': round(match, 3), 'town_overlap': round(near, 3), 'score': round(score, 3),
                       'pieces': len(placed_all), 'families': dict(collections.Counter(family(n) for n, *_ in placed_all)), 'placed': placed_all})
        print(f'dock {reg} ({kind}) <- {src_region}: {len(placed_all)} pieces, turn {rot} deg, shift {np.round(off / cw.HEX_DX, 2).tolist()} hex, water match {match:.2f}', flush=True)
    return new_recs, new_assets, report


# ------------------------------------------------------------------ waves
def vanilla_waves(assets, recs, pre):
    """Canvas positions of the existing coastal-wave emitters, and the single-point template leaves."""
    pos, templates = [], {}
    for rec in recs:
        if ours(rec['path']):
            continue
        t = rec['matrix'][12:15]
        for ch in decode(assets[rec['path']])[1]:
            b = assets.get(ch['path'])
            if not b or b'Coastal_Waves' not in b:
                continue
            rows, snd, whole = leaf_rows(b)
            for s in snd:
                if 'Coastal_Waves' not in s['name']:
                    continue
                for i in range(s['count']):
                    x, _, z = struct.unpack_from('<3f', b, s['points_offset'] + 12 * i)
                    pos.append((x + t[0], z + t[2]))
                kind = s['name'].rsplit('_', 1)[-1]
                if whole and not rows and len(snd) == 1 and s['type'] == 'SST_POINT' and len(decode(assets[rec['path']])[1]) == 1:
                    templates.setdefault(kind, []).append((rec, ch))
    return np.array(pos), {k: sorted(v, key=lambda e: e[0]['path'])[0] for k, v in templates.items()}


def coast_hexes(m):
    c = m['cells']
    kind = c[..., 0] & 3
    ids = map_hex.region_ids(c)
    names = m['land_regions'] + m['sea_regions']
    nl = len(m['land_regions'])
    nav = np.array([i >= nl and '_riv_' not in n and n not in ('3k_main_sea_lake',) for i, n in enumerate(names)] + [False])
    sea = (kind == 1) & nav[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
    coast = (kind != 1) & hex_dilate(sea)
    return coast, sea, kind, ids, names


def place_waves(ctx, rate):
    m, assets, root, recs, pre, ground = ctx['map'], ctx['assets'], ctx['root'], ctx['recs'], ctx['pre'], ctx['ground']
    coast, sea, kind, ids, names = coast_hexes(m)
    H, W = coast.shape
    v = map_hex.read(cw.VANILLA_MAP)
    vcoast_n, *_ = coast_hexes(v)
    vcoast = np.zeros_like(coast)
    vcoast[cw.NATIVE_HEX_SLICE] = vcoast_n
    newland = np.isin(ids, [i for i, n in enumerate(names[:len(m['land_regions'])]) if n.startswith('tw3k_')])
    near = newland.copy()
    for _ in range(WAVE_NEAR):
        near = hex_dilate(near)
    cand = coast & ~vcoast & near
    existing, templates = vanilla_waves(assets, recs, pre)
    rr, qq = np.nonzero(cand)
    P = np.array([cw.display_position(int(q), int(r)) for q, r in zip(qq, rr)])
    if len(existing):
        dmin = np.min(np.hypot(P[:, None, 0] - existing[None, :, 0], P[:, None, 1] - existing[None, :, 1]), 1) / cw.HEX_DX
        keep = dmin >= WAVE_CLEAR
        rr, qq, P = rr[keep], qq[keep], P[keep]
    order = np.random.default_rng(190).permutation(len(P))
    target = max(1, round(rate * len(P)))

    def spaced(D):
        cell = D * cw.HEX_DX
        grid, chosen = collections.defaultdict(list), []
        for i in order:
            x, z = P[i]
            gx, gz = int(x // cell), int(z // cell)
            if any(math.hypot(x - P[j][0], z - P[j][1]) < cell for a in (-1, 0, 1) for b in (-1, 0, 1) for j in grid[(gx + a, gz + b)]):
                continue
            grid[(gx, gz)].append(i)
            chosen.append(i)
        return chosen
    lo, hi = 1., 80.
    for _ in range(18):
        mid = (lo + hi) / 2
        if len(spaced(mid)) > target:
            lo = mid
        else:
            hi = mid
    spacing = hi
    chosen = spaced(spacing)
    new_recs, new_assets, report, skipped = [], {}, [], 0
    for k, i in enumerate(sorted(chosen, key=lambda j: (qq[j], rr[j]))):
        q, r = int(qq[i]), int(rr[i])
        seas = [(a, b) for a, b in neighbours(q, r, W, H) if sea[b, a]]
        if not seas:
            skipped += 1
            continue
        sort = 'Sand' if kind[r, q] == 2 else 'Rock'
        c0 = np.array(cw.display_position(q, r))
        u = np.mean([cw.display_position(a, b) for a, b in seas], 0) - c0
        u /= np.linalg.norm(u)
        ts = np.arange(0, 3.01, .05) * cw.HEX_DX
        hs = ground.h(c0[0] + ts * u[0], c0[1] + ts * u[1])
        hit = np.nonzero(hs <= WATER_Y + WAVE_HEIGHT[sort])[0]
        if not len(hit):
            skipped += 1
            continue
        x, z = c0 + ts[hit[0]] * u
        y = max(float(hs[hit[0]]), WATER_Y)
        rec, ch = templates[sort]
        t = rec['matrix'][12:15]
        owner = names[ids[seas[0][1], seas[0][0]]]

        def edit(suf, blob, x=x, y=y, z=z, t=t):
            s = sounds(blob)[0]
            out = bytearray(blob)
            struct.pack_into('<3f', out, s['points_offset'], x - t[0], y - t[1], z - t[2])
            return bytes(out)
        r_, n_ = clone_prefab(root, assets, pre, rec, [(ch, ch['path'].rsplit('.', 2)[-2])], owner, WAVE_BASE + k, edit)
        new_recs.append(r_)
        new_assets.update(n_)
        land = names[ids[r, q]]
        report.append({'sound': sort, 'coast_hex': [q, r], 'land_region': land, 'sea_region': owner, 'world': [round(x, 3), round(y, 3), round(z, 3)]})
    summary = {'candidate_coast_hexes': int(cand.sum()), 'after_vanilla_clearance': int(len(P)), 'vanilla_rate_per_coast_hex': rate,
               'target': target, 'spacing_hexes': round(spacing, 2), 'placed': len(report), 'skipped_no_shore': skipped,
               'by_sound': dict(collections.Counter(e['sound'] for e in report)),
               'templates': {k: v[1]['path'] for k, v in templates.items()},
               'by_land_region': dict(collections.Counter(e['land_region'] for e in report).most_common())}
    print('waves', {k: v for k, v in summary.items() if k != 'by_land_region'}, flush=True)
    return new_recs, new_assets, report, summary


def vanilla_wave_rate():
    """Coastal-wave sound points per vanilla sea-coast hex (every vertex of a line list counts: the
    vanilla lines run along the shore with ~3 hexes between vertices, the spacing of the points)."""
    from campaign_ground_textures import VanillaPacks
    pre = f'terrain/campaigns/{cw.TERRAIN_VARIANTS[0]}/'
    _, va = pb.unbundle(VanillaPacks().read(pre + 'global_props.bin'))
    n = sum(s['count'] for b in va.values() if b'Coastal_Waves' in b for s in sounds(b) if 'Coastal_Waves' in s['name'])
    coast, *_ = coast_hexes(map_hex.read(cw.VANILLA_MAP))
    return n / int(coast.sum()), n, int(coast.sum())


# ------------------------------------------------------------------ bundle
def run(apply):
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    m = map_hex.read(A / 'work/final-map/map.hex')
    targets = ports(m, design)
    rate, vn, vc = vanilla_wave_rate()
    report = {'water_y': WATER_Y, 'dock_templates': {k: v[0] for k, v in TEMPLATES.items()},
              'vanilla_wave_points': vn, 'vanilla_sea_coast_hexes': vc, 'variants': []}
    markers = []
    for variant in cw.TERRAIN_VARIANTS:
        pre = f'terrain/campaigns/{variant}/'
        data = (S / (pre + 'global_props.bin')).read_bytes()
        rootkey = pre + 'bmd_objects.bin'
        es, assets = pb.unbundle(data)
        root = assets[rootkey]
        cp, rs = decode(root)
        kept = [r for r in rs if not ours(r['path'])]
        assets = {k: v for k, v in assets.items() if not ours(k)}
        ctx = {'map': m, 'assets': assets, 'root': root, 'recs': kept, 'pre': pre, 'ground': Ground(variant)}
        d_recs, d_assets, docks = place_docks(ctx, targets)
        w_recs, w_assets, waves, wsum = place_waves(ctx, rate)
        new_recs, new_assets = d_recs + w_recs, {**d_assets, **w_assets}
        updated, chk = pb.prefab_list(root, cp, rs, kept, new_recs)
        assert len({r['path'] for r in chk}) == len(chk)
        bundled = pb.rebundle(es, assets, rootkey, updated, new_assets)
        report['variants'].append({'variant': variant, 'roots_before': len(rs), 'roots_after': len(chk), 'bundle_bytes': [len(data), len(bundled)],
                                   'docks': docks, 'dock_pieces': sum(d['pieces'] for d in docks), 'waves_summary': wsum, 'waves': waves})
        for d in docks:
            for n, x, y, z in d['placed']:
                col = [60, 120, 255] if ('boat' in n or 'fisher' in n or 'fish_anim' in n) else ([0, 255, 255] if n.startswith('Sound_') else [255, 60, 200])
                markers.append([x, z, col, 2])
        for w in waves:
            markers.append([w['world'][0], w['world'][2], [255, 255, 0] if w['sound'] == 'Sand' else [255, 140, 0], 4])
        if apply:
            pb.write_stage(S, pre, bundled, updated, new_assets)
            for loose in (S / pre).glob('bmd_objects.*.bin'):
                if ours(pre + loose.name) and pre + loose.name not in new_assets:
                    loose.unlink()
        print(variant, 'roots', len(rs), '->', len(chk), '; dock pieces', sum(d['pieces'] for d in docks), '; wave emitters', len(waves),
              'applied' if apply else 'dry run', flush=True)
    report['applied'] = apply
    report['live_verified'] = False
    tag = '' if apply else '-dry'
    (A / 'analysis').mkdir(exist_ok=True)
    (A / f'analysis/coast-props{tag}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    (A / f'analysis/coast-props-markers{tag}.json').write_text(json.dumps(markers), encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    run(ap.parse_args().apply)


if __name__ == '__main__':
    main()

"""Desert riparian belts (tugai): poplar and reed along the frontier rivers that cross real desert.

The desert rivers of the canvas (Shule and Dang at Dunhuang, Tarim, Konqi, Cherchen, Khotan, Yarkand,
Aksu, the lower Ili, Syr and Amu Darya) ran as bare water through bare sand: the foliage step plants
poplars only within 12 hexes of a town (foliage_biomes: desert .22) and frontier_forest leaves the
Western Regions band sparse on purpose. In reality each carries a narrow tugai belt: Euphrates poplar
(胡杨, Populus euphratica), tamarisk and reed (芦苇) a few km wide on both banks, bare desert beyond.

plant() adds, on canvas land only (the native map keeps vanilla's rows):
  * river hexes: apply_rivers' river channels (work/final-map/river-channels.json, every family) that are
    water in map.hex, outside the native map, whose surroundings are real desert (RESOLVE desert share over
    the (2 DESERT_R + 1)^2 hex box >= DESERT_SHARE; strategic_ink biome) below the cold climate;
  * belt hexes: land 1..len(BELT) hex steps (4-neighbour rings) from such a river hex, not a settlement
    slot, road hex or road tile, at least TOWN_CLEAR hexes from a town, and empty (one clump per hex, as
    vanilla);
  * per belt hex d steps out: a poplar with chance BELT[d-1][0] (sizes by POPLAR), else a reed clump
    (vanilla farm_foliage_small_1/2, the only low green campaign clump; no campaign reed or willow
    model exists: willow is battle-only) with chance BELT[d-1][1], drawn a quarter hex toward the water.
Draws are stateless hashes of the belt hex (tree_bake, props bake step 3) and every row asks
ctx['accept'] (dry footprint over the water plane and the river level, and the post-filter rules) before it
is written, so ring-1 rows under the drawn water are never planted (WET TREES used to drop them after).
Called after every other planting in polish_campaign_foliage (RIPARIAN BELTS block). numpy only.
"""
import json

import numpy as np

import campaign_world as cw
import tree_bake as tb

# (poplar, reed) chance per empty belt hex 1, 2 and 3 steps from the river hexes. Ring 1 lies under the drawn
# water: polish carves the channel and its river level (river-water-level.npy) stands over 83 % of ring-1
# ground (ring 2: 16 %, ring 3: 6 %; test copy), so WET TREES dropped three quarters of a ring-1 belt. The
# visible bank is ring 2 (reed at the water's edge, poplar), ring 3 the outer poplars.
BELT = ((0., 0.), (.50, .30), (.25, 0.))
POPLAR = {'arid_tree_poplar_large_1': .14, 'arid_tree_poplar_large_2': .14, 'arid_tree_poplar_medium_1': .22,
          'arid_tree_poplar_medium_2': .22, 'arid_tree_poplar_small_1': .14, 'arid_tree_poplar_small_2': .14}
REED = ('farm_foliage_small_1', 'farm_foliage_small_2')
DESERT_SHARE = .5
DESERT_R = 3
TOWN_CLEAR = 2.5          # hexes from a displayed town (the plot; the oasis belt reaches the town fields)


def _box_share(m, k):
    a = np.pad(m.astype(np.float64), k, mode='edge').cumsum(0).cumsum(1)
    a = np.pad(a, ((1, 0), (1, 0)))
    n = 2 * k + 1
    return (a[n:, n:] - a[:-n, n:] - a[n:, :-n] + a[:-n, :-n]) / (n * n)


def river_hexes(cells, channels_path, desert, cold):
    """Canvas river hexes in real desert (bool hex mask) and their channel family per hex (dict)."""
    H, W = cells.shape[:2]
    water = (cells[..., 0] & 3) == 1
    out = np.zeros((H, W), bool)
    fam = {}
    doc = json.loads(channels_path.read_text(encoding='utf-8'))
    for name, pts in doc['channels'].items():
        for q, r in pts:
            if 0 <= q < W and 0 <= r < H and water[r, q]:
                out[r, q] = True
                fam[(q, r)] = name
    # drawn-only hexes (river spec option B: land inside one region, carved and watered as links) carry belts too
    for name, pts in doc.get('drawn', {}).items():
        for q, r in pts:
            if 0 <= q < W and 0 <= r < H:
                out[r, q] = True
                fam[(q, r)] = name
    nat = np.zeros((H, W), bool)
    nat[cw.NATIVE_HEX_SLICE] = True
    out &= ~nat & (_box_share(desert, DESERT_R) >= DESERT_SHARE) & ~cold
    return out, fam


def plant(byname, vanilla_rows, ctx):
    """Add belt rows to byname[name]['rows'] (and 'added'). ctx: sample(field, tx, tz), surface, land,
    cells, towns_shown, road_cover (tile bool or None), cold (hex bool), channels (path),
    accept(tx, ty, species) -> bool array (tree_bake.TreeAccept.accept)."""
    import strategic_ink
    from fastops import grow_steps
    accept = ctx['accept']
    land, cells = ctx['land'], ctx['cells']
    H, W = land.shape
    biome, bnames, _, _ = strategic_ink.load()
    desert = biome == bnames.index('desert')
    riv, fam = river_hexes(cells, ctx['channels'], desert, ctx['cold'])
    occ = np.zeros((H, W), np.int32)
    for g in byname.values():
        if g['rows']:
            p = np.frombuffer(b''.join(bytes(r[:12]) for r in g['rows']), '<f4').reshape(-1, 3)
            q, r = cw.world_hex(p[:, 0].astype(np.float64), p[:, 2].astype(np.float64))
            np.add.at(occ, (np.clip(np.asarray(r), 0, H - 1), np.clip(np.asarray(q), 0, W - 1)), 1)
    dist = grow_steps(riv, len(BELT))
    slot = (cells[..., 2] >> 4) > 0
    road = ((cells[..., 3].astype(np.int32) >> 1) & 63) > 0
    nat = np.zeros((H, W), bool)
    nat[cw.NATIVE_HEX_SLICE] = True
    cand = land & ~nat & (dist >= 1) & (dist <= len(BELT)) & ~slot & ~road & (occ == 0)
    r0, q0 = np.nonzero(cand)
    towns = np.asarray(ctx['towns_shown'], np.float64)
    zr = r0 + (q0 % 2) * .5
    td = np.full(len(r0), np.inf)
    for s in range(0, len(r0), 4096):
        td[s:s + 4096] = np.hypot(towns[None, :, 0] - q0[s:s + 4096, None], towns[None, :, 1] - zr[s:s + 4096, None]).min(1)
    keep = td >= TOWN_CLEAR
    r0, q0 = r0[keep], q0[keep]
    d = dist[r0, q0]
    u = tb.u01('rip.kind', q0, r0)
    pop_p = np.array([BELT[k - 1][0] for k in d]); reed_p = np.array([BELT[k - 1][1] for k in d])
    kind = np.where(u < pop_p, 1, np.where(u < pop_p + reed_p, 2, 0))
    # nearest river hex (for the reed's lean toward the water): the river neighbour at one step
    names_p = list(POPLAR); w_p = np.array([POPLAR[n] for n in names_p]); w_p /= w_p.sum()
    pick_p = tb.categorical(w_p, 'rip.poplar', q0, r0)
    pick_r = tb.integers(len(REED), 'rip.reed', q0, r0)
    jit = np.column_stack((tb.uniform(-.35, .35, 'rip.jx', q0, r0), tb.uniform(-.35, .35, 'rip.jy', q0, r0)))
    u_pool = tb.u01('rip.pool', q0, r0)
    report = {'river_hexes': int(riv.sum()), 'belt_candidates': int(len(r0)), 'by_family': {}, 'poplars': 0, 'reeds': 0, 'rejected': 0}
    road_cover = ctx.get('road_cover')
    for i in np.flatnonzero(kind):
        q, r = int(q0[i]), int(r0[i])
        tx = 2 * q + 1 + 2 * jit[i, 0]
        ty = 2 * r + q % 2 + 1 + 2 * jit[i, 1]
        if kind[i] == 2:   # reed: a quarter hex toward the neighbour one ring nearer the river
            s = 1 if q % 2 else -1
            nb = [(q + a, r + b) for a, b in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, s), (-1, s))
                  if 0 <= q + a < W and 0 <= r + b < H and dist[r + b, q + a] < d[i]]
            if nb:
                nq, nr = nb[0]
                tx += (2 * nq + 1 - (2 * q + 1)) * .25
                ty += (2 * nr + nq % 2 + 1 - (2 * r + q % 2 + 1)) * .25
        hq, hr = (int(v) for v in cw.tile_hex(int(tx), int(ty)))
        if not (0 <= hq < W and 0 <= hr < H) or not land[hr, hq] or riv[hr, hq] or slot[hr, hq]:
            continue
        if road_cover is not None and road_cover[min(int(ty), road_cover.shape[0] - 1), min(int(tx), road_cover.shape[1] - 1)]:
            continue
        name = names_p[pick_p[i]] if kind[i] == 1 else REED[pick_r[i]]
        if not accept(np.array([tx]), np.array([ty]), name)[0]:
            report['rejected'] += 1
            continue
        y = float(ctx['sample'](ctx['surface'], np.array([tx]), np.array([ty]))[0])
        g = byname.get(name)
        pool = (vanilla_rows.get(name) or ([g['template']] if g and g['template'] else [])) if g else []
        if not pool:
            continue
        row = bytearray(pool[min(int(u_pool[i] * len(pool)), len(pool) - 1)])
        row[0:12] = np.array([tx * cw.DX / 2, y, ty * cw.DZ / 2], '<f4').tobytes()
        g['rows'].append(bytes(row))
        g['added'] += 1
        occ[r, q] += 1
        report['poplars' if kind[i] == 1 else 'reeds'] += 1
        # the river this belt hex follows: the family of a river hex within the belt reach
        f = next((fam[(q + a, r + b)] for a in range(-3, 4) for b in range(-3, 4) if (q + a, r + b) in fam and riv[r + b, q + a]), '?')
        report['by_family'][f] = report['by_family'].get(f, 0) + 1
    return report

"""Vanilla-density forests and rock clusters for the frontier (campaign tree list).

The procedural replanting of polish_campaign_foliage.py lays three jittered trees on a
quarter of the hexes at most, ~61 trees per 1000 frontier land hexes against vanilla's 464
(taiga 0.18 trees/hex, tropical 0.21). Vanilla (dlc07 tree list, 205,767 entries, offline
survey 2026-10-23, scratch farms-rocks/vtrees.py, vrocks.py) plants ONE clump model per hex
on a near-regular lattice (nearest neighbour 0.67/0.85/0.98 hexes at the 10/50/90th
percentiles, never two in a hex: 1.0 per occupied hex), and ties density to the ground:
  dense_forest 0.85 trees/hex, light_forest 0.82, hilly_light_forest 0.81,
  hilly_dense_forest 0.63, mountain 0.62, grassland 0.40, hilly_grassland 0.13,
  desert 0.12, wetland 0.04; cold dense/light forest 0.91/0.83, temperate 0.85/0.83,
  subtropical light/hilly light 0.80/0.81;
thinned at towns (0.125 within 2 hexes, 0.31 at 2-4, 0.42 at 4-7, 0.48 beyond 10) and
roads (0.12 on road hexes, 0.32 next to them, 0.50 beyond); species by climate: cold
cold_tree_fir_large_1/medium_1 (~90%), temperate katsura (seven groups) with ~20% firs,
subtropical castanopsis (five groups) with ~35% bamboo_1-3.
Rocks: general_rock_large/medium/small (1,558 entries, 83% on impassable mountain/desert)
in clusters of 3 (median; 90th percentile 15) of one rock per hex, denser on steep ground
(rock entries per 1000 land hexes by slope in world units per hex: 16 below 0.1, 12 at
0.2-0.4, 17 at 0.4-0.7, 43 above 0.7); rock_shrub_tiny/small (4,945) on passable open
ground (0.07 impassable), ~13 per 1000 flat hexes.

plant() adds, in the frontier only (canvas outside the native map, and native hexes the
foliage step may replant), on hexes the foliage step's own rule allows:
  * one tree per empty hex of real forest biome (RESOLVE taiga/conifer, temperate forest,
    tropical moist/mangrove, tropical dry) within FILL_REACH hexes of playable land, at
    vanilla's density for that forest, outside the Western Regions band (west of the
    native map at its latitudes: deserts and steppe, left sparse on purpose);
  * rock clusters on impassable steep frontier mountains and rock shrubs on open arid,
    steppe and alpine ground, Western Regions included.
Towns (TOWN_CLEAR) and roads (analysis/road-tile-cover.npy, road hexes) stay clear.
Props bake (step 3): every draw is a stateless hash of the hex and its purpose (tree_bake), and every row
asks ctx['accept'] (tree_bake.TreeAccept.accept: dry footprint and the post-filter rules) before it is
written. The one order-dependent part left is the rock-cluster growth: clusters claim hexes from a shared
mask, seeds in row-major hex order (a function of the seed set only, never of earlier draws), so a seed
added or removed changes only the clusters that compete with it for hexes.
Called by polish_campaign_foliage.py (FRONTIER FOREST block); numpy only.
"""
import math

import numpy as np

import campaign_world as cw
from terrain_noise import fbm
import biome_profiles as _bp
import tree_bake as tb

_K = _bp.consts('frontier_forest')   # values: biome_profiles.CONSTS['frontier_forest']

FILL_REACH = 40          # hexes from playable land that the forest fill reaches (nominal; see FILL_FADE)
# --- FILL FADE (ground materials batch) --- the fill (and apply_forest_ground's forest ground) stopped on
# a hard line FILL_REACH hexes from playable land: full vanilla-density forest on one side, bare
# ground on the other. fill_mask() is the one shared hex-level yes/no mask: the distance from
# playable land is meandered by FILL_MEANDER hexes of fbm (FILL_MEANDER_GRAIN hexes), turned into a
# weight that falls from 1 at FILL_FADE[0] to 0 at FILL_FADE[1] hexes (smoothstep), and dithered
# per hex against a quantile-uniform fbm of FILL_DITHER_GRAIN hexes (forest islands and gaps,
# deterministic: same seeds, same mask on every run and in both scripts).
FILL_FADE = (30, 50)
FILL_MEANDER, FILL_MEANDER_GRAIN = 6., 14.
FILL_DITHER_GRAIN = 2.5
# --- end FILL FADE
FOREST = _bp.biome_map('tree_fill', 'tree_fill')   # real biome: (trees per hex: the fill level, shaped by forest_density in fill_probability; species mix); no temperate
MIX = _K['MIX']   # vanilla tree-list shares by climate (dlc07)
STEEP_FACTOR = _K['STEEP_FACTOR']       # vanilla mountain 0.62 vs light forest 0.82
TOWN_RAMP = _K['TOWN_RAMP']   # vanilla thinning beyond the cleared ring
ROAD_HEX_FACTOR = _K['ROAD_HEX_FACTOR']
ROCKS = ('general_rock_large_1', 'general_rock_large_2', 'general_rock_medium_1', 'general_rock_medium_2',
         'general_rock_small_1', 'general_rock_small_2')
SHRUBS = ('rock_shrub_tiny_1', 'rock_shrub_tiny_2', 'rock_shrub_small_1')
# Rock entries per impassable hex by slope (world units per hex): vanilla general rocks (~6 per 1000
# impassable hexes, 3x on the steepest) at about 2/3, as the band rule already sets
# general_rock_medium_1 on bare heights above 4.8.
ROCK_DENSITY = _K['ROCK_DENSITY']
ROCK_CLUSTER_P = _K['ROCK_CLUSTER_P']      # geometric growth: median cluster of 3
SHRUB_DENSITY = _K['SHRUB_DENSITY']     # per open arid/steppe/alpine hex (vanilla ~13 per 1000 flat land hexes, mostly grassland)
DRY_GROUND = .25         # hex-centre height of the shore candidate mask (the per-tree test is tree_bake's dry footprint)
OPEN_BIOMES = _bp.biome_set('shrub')   # desert, steppe, alpine, tundra, grass, savanna


def fill_weight(play, floor=None):
    """FILL FADE: fade weight per hex (0..1) from the playable-land mask (hex grid, rows r, cols q). floor: per-hex
    minimum (biome_profiles.fill_floor: FOREST_FILL_FLOOR on the far-taiga classes with TW3K_FOREST_FILL_FLOOR) or None."""
    from fastops import grow_steps
    H, W = play.shape
    d = grow_steps(play, FILL_FADE[1] + int(FILL_MEANDER) + 2).astype(np.float32)
    rr, qq = np.mgrid[:H, :W].astype(np.float32)
    zr = rr + (qq % 2) * .5
    d = d + FILL_MEANDER * fbm(qq, zr, (0, 0), FILL_MEANDER_GRAIN, 3, seed=311)
    t = np.clip((FILL_FADE[1] - d) / (FILL_FADE[1] - FILL_FADE[0]), 0, 1)
    w = t * t * (3 - 2 * t)
    if floor is not None:   # [FILL FLOOR] far taiga keeps a floor (biome_profiles FOREST_FILL_FLOOR)
        w = np.maximum(w, floor)
    return w.astype(np.float32)


def fill_mask(play, floor=None):
    """FILL FADE: shared hex-level yes/no forest-fill mask (tree fill and forest ground); floor as fill_weight."""
    H, W = play.shape
    w = fill_weight(play, floor)
    rr, qq = np.mgrid[:H, :W].astype(np.float32)
    zr = rr + (qq % 2) * .5
    n = fbm(qq, zr, (0, 0), FILL_DITHER_GRAIN, 3, seed=312)
    g = np.mgrid[0:512, 0:512].astype(np.float32)
    qs = np.quantile(fbm(g[1], g[0], (0, 0), FILL_DITHER_GRAIN, 3, seed=312), np.linspace(0, 1, 257))
    u = np.interp(n, qs, np.linspace(0, 1, 257))
    return (u < w) & (w > 0)


# Open eco classes the fill also reaches, sparsely (props rule data, calibration 2026-10-30): with eco_bio as the one
# biome source, hexes strategic_ink called taiga / forest and eco_bio calls tundra, steppe, grass, paddy or wetland
# left the fill (tree_fill classes only): -20 to -30 % trees on those classes. (trees per hex, species mix)
FILL_EXTRA = {'tundra': (.40, 'cold'), 'steppe': (.03, 'cold_mixed'), 'grass': (.30, 'temperate'),
              'paddy': (.015, 'subtropical'), 'wetland': (.02, 'temperate')}
FILL_CLASSES = {**FILL_EXTRA, **FOREST}


def fill_probability(shape, fill, eco, bnames):
    """Fill tree probability per hex inside fill_mask: the class level (FOREST: tree_fill trees per hex, the
    pre-bake fill's mean, ~vanilla's .83-.91 cold forest per hex) x the shared forest density's shape
    (tree_bake.forest_shape: density / forest_fill over its class mean, so the fade counts once and the field
    decides where, the level how many; calibration 2026-10-30); 0 where the class has no fill level. Where the
    field is faded out but fill_mask still reaches (the native rectangle's repurposed filler, whose ground the
    caller's replant_ok lets the fill take: forest_fill is 0 there for the woodland ground's sake) the shape is 1,
    the level alone, as the pre-bake fill planted there (v4: Dunxun, Yilumo, Lin Yi -40 to -90 % without it).
    Beyond the fade fill_mask is empty and the west band is excluded by the caller, so nothing is planted there;
    outside fill_mask nothing is planted. fill: kept for the signature (the shape already carries the fade)."""
    lut = np.array([FILL_CLASSES[n][0] if n in FILL_CLASSES else 0. for n in bnames] + [0.], np.float32)
    level = lut[np.clip(np.asarray(eco, np.int64), 0, len(bnames))]
    return np.clip(level * np.asarray(shape, np.float32), 0, 1).astype(np.float32)


def fill_pick(q, r, dens):
    """Forest fill draw per hex: stateless, so the picked set only grows with the density."""
    return tb.u01('ff.forest', q, r) < dens


def _choice(mix, purpose, q, r):
    """Species per hex from the mix weights (stateless draw keyed by hex)."""
    names = list(mix)
    p = np.array([mix[k] for k in names], np.float64)
    return [names[i] for i in tb.categorical(p, purpose, q, r)]


def plant(byname, vanilla_rows, ctx):
    """Add rows to byname[name]['rows'] (and 'added'). ctx: dict with
    sample(field, tx, tz) (sample_many), surface, alpha (tile grids), land (hex bool),
    cells (map.hex cells), towns_shown ((n, 2) x/DX, z/DZ), replant_ok(q, r) -> bool array,
    town_clear (hexes), road_cover (tile bool or None), plan, names (map regions),
    accept(tx, ty, species) -> bool array (tree_bake.TreeAccept.accept),
    forest (tree_bake.forest_field_hex: hex density, fill, eco classes and their names)."""
    accept = ctx['accept']
    H, W = ctx['land'].shape
    land = ctx['land']
    cells = ctx['cells']
    DX, DZ = cw.DX, cw.DZ
    # species and candidate classes: landform-tiles eco_bio (real_bio_eco), the classes the forest density
    # was made from (one biome source for the woodland ground and the trees)
    ff = ctx['forest']
    biome, bnames = ff['eco'], list(ff['names'])
    # Existing entries per hex (the fill only takes empty hexes: one clump per hex).
    occ = np.zeros((H, W), np.int32)
    for g in byname.values():
        if g['rows']:
            p = np.frombuffer(b''.join(bytes(r[:12]) for r in g['rows']), '<f4').reshape(-1, 3)
            q, r = cw.world_hex(p[:, 0].astype(np.float64), p[:, 2].astype(np.float64))
            np.add.at(occ, (np.asarray(r), np.asarray(q)), 1)
    # Playable land and its reach.
    import map_hex
    ids = map_hex.region_ids(cells)
    names = ctx['names']
    npl = np.array(['non_playable' in n for n in names] + [True])[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
    play = land & ~npl
    from fastops import grow_steps
    reach = fill_mask(play, _bp.fill_floor(biome, bnames))   # FILL FADE (was grow_steps(play, FILL_REACH) <= FILL_REACH); [FILL FLOOR]
    nat = np.zeros((H, W), bool)
    nat[cw.NATIVE_HEX_SLICE] = True
    import revealed_filler
    nat &= ~revealed_filler.filler_hex(cells, names)   # vanilla non-playable filler that is land now (playable or not) is canvas: visual
    rr_, qq_ = np.mgrid[:H, :W]
    west = _bp.west_band(qq_, rr_, cw)   # Western Regions band: q < PAD_LEFT at the native rows
    # Hex heights and slope from the polished surface (tile at the block centre).
    tyc = np.minimum(2 * rr_ + qq_ % 2 + 1, ctx['surface'].shape[0] - 1)
    txc = np.minimum(2 * qq_ + 1, ctx['surface'].shape[1] - 1)
    hh = ctx['surface'][tyc, txc].astype(np.float32)
    gy, gx = np.gradient(hh)
    slope = np.hypot(gx, gy)
    imp = (cells[..., 2] & 8) > 0
    road_hex = ((cells[..., 3].astype(np.int32) >> 1) & 63) > 0
    shore = ~land | (hh <= DRY_GROUND)
    # one step out along rows and columns, edge-safe (np.roll wrapped the far map edge onto this one)
    s0 = shore.copy()
    shore[1:] |= s0[:-1]; shore[:-1] |= s0[1:]; shore[:, 1:] |= s0[:, :-1]; shore[:, :-1] |= s0[:, 1:]
    road_cover = ctx.get('road_cover')
    towns = np.asarray(ctx['towns_shown'], np.float64)

    def town_dist(q, r):
        out = np.empty(len(q))
        for s in range(0, len(q), 4096):
            qs, rs = q[s:s + 4096, None], r[s:s + 4096, None]
            out[s:s + 4096] = np.hypot(towns[None, :, 0] - qs, towns[None, :, 1] - rs).min(1)
        return out

    def place(names_, q, r, jitter, purpose):
        added = 0
        tx = 2 * q + 1 + 2 * tb.uniform(-jitter, jitter, purpose + '.jx', q, r)
        ty = 2 * r + q % 2 + 1 + 2 * tb.uniform(-jitter, jitter, purpose + '.jy', q, r)
        hq, hr = cw.tile_hex(tx.astype(np.int64), ty.astype(np.int64))
        ok = land[hr, hq] & (occ[hr, hq] == 0) | (hq == q) & (hr == r)
        ok &= land[hr, hq]
        if road_cover is not None:
            ix = np.clip(tx.astype(np.int64), 0, road_cover.shape[1] - 1)
            iy = np.clip(ty.astype(np.int64), 0, road_cover.shape[0] - 1)
            ok &= ~road_cover[iy, ix]
        y = ctx['sample'](ctx['surface'], tx, ty)
        # dry footprint (real lakes on land hexes, Baikal, show where the ground is under the water plane) and the
        # post-filter rules, before the row exists
        if ok.any():
            ok[ok] = accept(tx[ok], ty[ok], [names_[i] for i in np.flatnonzero(ok)])
        u_pool = tb.u01(purpose + '.pool', q, r)
        for i in np.flatnonzero(ok):
            nm = names_[i]
            g = byname.get(nm)
            if g is None:
                continue
            pool = vanilla_rows.get(nm) or ([g['template']] if g['template'] else [])
            if not pool:
                continue
            row = bytearray(pool[min(int(u_pool[i] * len(pool)), len(pool) - 1)])
            row[0:12] = np.array([tx[i] * DX / 2, y[i], ty[i] * DZ / 2], '<f4').tobytes()
            g['rows'].append(bytes(row))
            g['added'] += 1
            occ[r[i], q[i]] += 1
            added += 1
        return added

    report = {}
    # ------------------------------------------------------------- forest fill
    fid = {bnames.index(k): v for k, v in FILL_CLASSES.items() if k in bnames}
    # Forest fill: inside fill_mask (the yes/no islands and clearings shared with apply_forest_ground, dithered
    # on the fill fade) with probability level x shape (fill_probability: the class's tree_fill level shaped by the
    # shared forest density; the fade counts once). Kept from props: the species tables (tree_fill classes),
    # towns, roads, shore, bare heights and slope.
    pfill = fill_probability(ff['shape'], ff['fill'], biome, bnames)
    cand = land & reach & ~west & np.isin(biome, list(fid)) & (occ == 0) & ~shore & (pfill > 0)
    r0, q0 = np.nonzero(cand)
    ok = ctx['replant_ok'](q0, r0)
    r0, q0 = r0[ok], q0[ok]
    zr = r0 + q0 % 2 * .5
    td = town_dist(q0.astype(np.float64), zr)
    keep = td >= ctx['town_clear']
    r0, q0, zr, td = r0[keep], q0[keep], zr[keep], td[keep]
    dens = pfill[r0, q0].astype(np.float64)
    dens = dens * np.interp(td - ctx['town_clear'], TOWN_RAMP[0], TOWN_RAMP[1])
    dens = np.where(road_hex[r0, q0], dens * ROAD_HEX_FACTOR, dens)
    dens = np.where(hh[r0, q0] > 4.8, 0, dens)          # bare heights keep the rock rule
    dens = np.where(imp[r0, q0] & (slope[r0, q0] > .4), dens * STEEP_FACTOR, dens)
    dens = np.clip(dens, 0, 1)
    pick = fill_pick(q0, r0, dens)
    q1, r1 = q0[pick], r0[pick]
    mixes = [fid[int(b)][1] for b in biome[r1, q1]]
    names_ = [None] * len(q1)
    for mk in sorted(set(mixes)):
        idx = np.array([i for i, m in enumerate(mixes) if m == mk], np.int64)
        for i, n in zip(idx.tolist(), _choice(MIX[mk], 'ff.forest.species', q1[idx], r1[idx])):
            names_[i] = n
    report['forest_trees'] = place(names_, q1, r1, .3, 'ff.forest')
    report['forest_candidate_hexes'] = int(len(r0))
    by = {}
    for n in names_:
        by[n] = by.get(n, 0) + 1
    report['forest_by_species'] = dict(sorted(by.items(), key=lambda kv: -kv[1]))
    # ------------------------------------------------------------- rock clusters
    frontier = land & ~nat & ~shore
    cand = frontier & reach & imp & (occ == 0)
    r0, q0 = np.nonzero(cand)
    ok = ctx['replant_ok'](q0, r0)
    r0, q0 = r0[ok], q0[ok]
    keep = town_dist(q0.astype(np.float64), r0 + q0 % 2 * .5) >= ctx['town_clear']
    keep &= ~road_hex[r0, q0]
    r0, q0 = r0[keep], q0[keep]
    allowed = np.zeros((H, W), bool)
    allowed[r0, q0] = True
    p_rock = np.interp(slope[r0, q0], ROCK_DENSITY[0], ROCK_DENSITY[1]) / 3.0   # seeds; clusters average ~3
    seeds = tb.u01('ff.rock.seed', q0, r0) < p_rock
    rq, rr2, clusters = [], [], 0
    # Order-dependent by nature: clusters claim hexes from the shared allowed mask, seeds in row-major hex
    # order (np.nonzero). Every growth draw is keyed by the seed hex and the step k, so a cluster's shape
    # depends only on its seed and on which hexes earlier (row-major) clusters already took.
    sq_, sr_ = q0[seeds], r0[seeds]
    steps = np.arange(15)[None, :]
    grow = tb.u01('ff.rock.grow', sq_[:, None], sr_[:, None], steps)
    u_mem = tb.u01('ff.rock.member', sq_[:, None], sr_[:, None], steps)
    u_nb = tb.u01('ff.rock.nb', sq_[:, None], sr_[:, None], steps)
    for s, (q, r) in enumerate(zip(sq_.tolist(), sr_.tolist())):
        clusters += 1
        members = [(q, r)]
        allowed[r, q] = False
        k = 0
        while grow[s, k] < ROCK_CLUSTER_P and len(members) < 15:
            bq, br = members[min(int(u_mem[s, k] * len(members)), len(members) - 1)]
            nb = [(bq + dq, br + dr) for dq, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1 - 2 * (bq % 2 == 0)), (-1, 1 - 2 * (bq % 2 == 0)))]
            nb = [(a, b) for a, b in nb if 0 <= a < W and 0 <= b < H and allowed[b, a]]
            if not nb:
                break
            a, b = nb[min(int(u_nb[s, k] * len(nb)), len(nb) - 1)]
            k += 1
            allowed[b, a] = False
            members.append((a, b))
        for a, b in members:
            rq.append(a)
            rr2.append(b)
    rq, rr2 = np.array(rq, np.int64), np.array(rr2, np.int64)
    report['rocks'] = place([ROCKS[int(i)] for i in tb.integers(len(ROCKS), 'ff.rock.species', rq, rr2)], rq, rr2, .3, 'ff.rock')
    report['rock_clusters'] = clusters
    # ------------------------------------------------------------- rock shrubs
    oid = [bnames.index(k) for k in OPEN_BIOMES if k in bnames]
    cand = frontier & reach & ~imp & np.isin(biome, oid) & (occ == 0) & ~road_hex
    r0, q0 = np.nonzero(cand)
    pick = tb.u01('ff.shrub', q0, r0) < SHRUB_DENSITY
    r0, q0 = r0[pick], q0[pick]
    if len(r0):
        ok = ctx['replant_ok'](q0, r0)
        r0, q0 = r0[ok], q0[ok]
        keep = town_dist(q0.astype(np.float64), r0 + q0 % 2 * .5) >= ctx['town_clear']
        r0, q0 = r0[keep], q0[keep]
    report['rock_shrubs'] = place([SHRUBS[int(i)] for i in tb.integers(len(SHRUBS), 'ff.shrub.species', q0, r0)], q0, r0, .35, 'ff.shrub')
    return report

"""Vanilla mountain props on the frontier mountains, placed by vanilla's own statistics.

Vanilla dlc07 global_props (offline survey 2026-10-23, scratch farms-rocks/vprops.py):
762 mountain-model instances (rigidmodels/campaign/mountains/cold, /arid, /temperate,
/subtropical; terraces aside) in 1-7-instance clusters, one root per cluster, leaves
'.31' (all seasons):
  * cold peaks/ridges (340; 94% on impassable hexes; scale ~0.11; nearest neighbour
    5.1/7.6/11.4 hexes at the 25/50/75th percentiles; 10th percentile 7.8 hexes from a
    town, 4.5 from a road, 12 from water), arid mountains (59; scale 1.0; 7.5 hexes from
    towns, 2.8 from roads, 7.5 from water), subtropical karst (324, scale ~3, large
    clusters reconcile_terrain_prefabs keeps off authored land), temperate (39, near towns);
  * density per 1000 impassable hexes rises with slope (2.1 below 0.2 world units per
    hex, 4.0 to 0.4, 4.6 to 0.7, 5.6 above) and on ridges (topographic position: 2.0 on
    level ground, 3.4 at +0.05..0.3, 7.2 above +0.3, 2.2 in hollows); by climate 2.6 cold,
    1.4 arid;
  * cold instances use BHM_CUSTOM_VERTEX_OFFSETTING (Y is an offset over the terrain:
    10/50/90th percentile -11.6/-7.6/-4.2 against the hex ground), the rest BHM_IGNORE
    (absolute Y, sunk -0.63 median).
Vegetation props (13,638) sit 100% on vanilla's mountain fine tiles (Y median +2.5 above
the LF), so none are added here: the frontier has no mountain tiles. Rocks are added to
the tree list instead (frontier_forest.py), like most vanilla rocks.

This step clones whole vanilla cold and arid clusters (their root record, parent and '.31'
leaf: instance records copied byte for byte, positions moved) onto frontier sites chosen by
those rules: impassable land outside the native map within REACH hexes of playable land,
climate family by real elevation/biome, seeds weighted by slope and ridge position, kept
apart by the vanilla spacing, clear of towns, roads, lakes and coasts, and with the whole
model footprint on impassable land. Cluster layouts turn by 0 or 180 degrees (sign-free for
any matrix convention), single instances by any angle about the vertical. One root per
cluster (region of the site hex), identity root matrix, world-space leaf
(world_space_leaves.py). Rebuilt from the staged bundle each run after the lake leaves:

  python frontier_mountain_props.py [--apply]
"""
import argparse
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np

import campaign_world as cw
import props_bundle as pb
from parse_global_props import parse
from read_root_prefabs import decode
from translate_campaign_lake_instances import records as leaf_records, MODEL

A = cw.WORKSPACE
S = A / 'work/release-input'
FIRST_PREFAB = 7100           # cluster numbers FIRST_PREFAB..FIRST_PREFAB+899 (vanilla roots stop at 5421; Hehuang uses 9475)
REACH = 60                    # hexes from playable land
TOWN_CLEAR, ROAD_CLEAR, WATER_CLEAR = 8, 4, 7
SPACING = 5.0                 # hexes between clusters (vanilla nearest neighbour 25th percentile)
FOOTPRINT_CORE = .8           # share of the model's bounding radius counted as its footprint
# Footprint rules (vanilla: cold footprints are 0.9 impassable at the median, 0.33 at the
# 10th percentile, always on land; arid 0.23 median; roads 1.4-1.8% of footprint hexes):
FOOTPRINT_IMPASSABLE = {'cold': .75, 'arid': .5}
FOOTPRINT_ROAD = .03
# Mountain relief at the site: 7x7-hex height range (vanilla cold 0.33/1.45/2.99 at the
# 10/50/90th percentiles, arid 0.5/1.14/2.24); flat impassable steppe and desert take none.
MIN_RELIEF = {'cold': .6, 'arid': .9}   # arid sites: the hilly half of vanilla's (rolling Mongolian steppe stays bare)
# Instances per 1000 impassable hexes (vanilla), by slope and by topographic position.
SLOPE_DENS = ((0, .2, .4, .7, 1.2), (2.1, 4.0, 4.6, 5.6, 5.6))
TPI_FACTOR = ((-.3, -.05, .05, .3, .6), (1.1, 1.05, 1.0, 1.7, 3.5))
FAMILY_SCALE = {'cold': 1.0, 'arid': .55}    # 2.6 vs 1.4 per 1000 in vanilla
MAX_INSTANCES = 1200                          # budget: vanilla holds 762 mountain instances
# --- GROUND MATERIALS (ground_materials.py) --- rock density toward vanilla's rate.
# The first batch placed 103 cold instances on 67,839 cold candidate hexes (1.5 per 1000,
# vanilla 2.6): seeds were drawn at exactly the vanilla rate and then ~half were refused
# (spacing, footprint on impassable land, water, roads). Now each family gets a target =
# the vanilla rate summed over its candidates (dens x FAMILY_SCALE, instances), seeds are
# drawn SEED_OVERSAMPLE times denser, and a family stops once it reaches its target, so the
# refusals no longer halve the density. Real ridges weigh more: REAL_RIDGE scales the seed
# rate by the real-DEM slope per hex (m per hex), on top of the polished-surface slope/TPI.
# Cold family by the same latitude-dependent line as the cold ground band
# (ground_materials.SNOWLINE), so the rock sets and the winter ground agree.
SEED_OVERSAMPLE = {'cold': 8.0, 'arid': 1.5}   # cold: 3x reached 2.0 per 1000, 8x 2.7 (vanilla 2.6), 20x packs to the spacing (3.4); arid (STATELESS, no cap; v21 copy): 1.4 -> 180, 1.5 -> the target ~199, 1.6 -> 207, 3.0 -> 332
REAL_RIDGE = ((150., 400., 900.), (1.0, 1.35, 1.8))
# --- end GROUND MATERIALS
# --- STATELESS DRAWS (coordinator 2026-10-31, v21b) --- every draw of a site comes from a hash of (SEED, variant,
# family, q, r, purpose) (site_u), and the spacing is resolved in a fixed hash order of (q, r): adding or removing a
# candidate changes no other site's draws (v20 -> v21 redrew 330 of 359 sites through the shared stream). The family
# target no longer caps placement (a cap makes a site depend on every site placed before it); it stays in the
# report as the vanilla-rate reference, and SEED_OVERSAMPLE sets the rate. Invariant: frontier_mountain_stable.
SEED = 4410
ACCEPT, ORDER, PICK, TURN = 0, 1, 2, 3          # draw purposes
FAMILY_ID = {'cold': 1, 'arid': 2, 'temperate': 3}
_M64 = (1 << 64) - 1


def _mix(x):
    """splitmix64 finaliser on uint64 arrays."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(_M64)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(_M64)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(_M64)
    return x ^ (x >> np.uint64(31))


def site_u(variant, family, q, r, purpose):
    """Uniform [0, 1) per site, a pure function of (SEED, variant, family, q, r, purpose); family None for draws
    shared by all families (the spacing order). Vectorised over q, r."""
    with np.errstate(over='ignore'):
        h = _mix(np.uint64(SEED) ^ np.uint64(zlib.crc32(variant.encode())))
        h = _mix(h ^ np.uint64(FAMILY_ID.get(family, 0) << 8 | purpose))
        h = _mix(h ^ np.asarray(q, np.int64).astype(np.uint64))
        h = _mix(h ^ np.asarray(r, np.int64).astype(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def model_radius(assets, model, cache={}):
    """Horizontal LOD0 bounding radius in model units (reconcile_terrain_prefabs.model_radius)."""
    key = model.lower()
    if key not in cache:
        r = 0.
        try:
            geo = key if key.endswith('.rigid_model_v2') else ET.fromstring(assets.read(key)).findtext('geometry').lower()
            g = assets.read(geo)
            lods = struct.unpack_from('<I', g, 8)[0]
            groups = struct.unpack_from('<I', g, 140)[0]
            p = 140 + lods * 28
            for _ in range(groups):
                lo = struct.unpack_from('<3f', g, p + 24)
                hi = struct.unpack_from('<3f', g, p + 36)
                r = max(r, abs(lo[0]), abs(lo[2]), abs(hi[0]), abs(hi[2]))
                p += struct.unpack_from('<I', g, p + 4)[0]
        except (KeyError, AttributeError, struct.error, ET.ParseError, TypeError):
            r = 0.
        cache[key] = r
    return cache[key]


def model_triangles(assets, model, cache={}):
    """LOD0 triangles of a model (.wsmodel -> its geometry): the LOD0 header's index bytes / 6 (u16 indices).
    None when the model cannot be read."""
    key = model.lower()
    if key not in cache:
        try:
            geo = key if key.endswith('.rigid_model_v2') else ET.fromstring(assets.read(key)).findtext('geometry').lower()
            g = assets.read(geo)
            cache[key] = struct.unpack_from('<I', g, 140 + 8)[0] // 6 if g[:4] == b'RMV2' else None
        except (KeyError, AttributeError, struct.error, ET.ParseError, TypeError):
            cache[key] = None
    return cache[key]


def family_deviation(counts, targets, tol):
    """{family: (count, target, |count - target| / target, over tol)} for every family with a target > 0."""
    out = {}
    for f, t in targets.items():
        if t:
            d = abs(counts.get(f, 0) - t) / t
            out[f] = (counts.get(f, 0), t, round(d, 4), d > tol)
    return out


def window_extreme(a, k, op, fill):
    """op (np.maximum / np.minimum) over the (2k+1)^2 window of every cell, clipped at the map
    edges (fill pads outside: -inf for a maximum, +inf for a minimum). np.roll wrapped: the
    west edge took the east edge's relief and the south edge the north's."""
    out = a
    for ax in (0, 1):
        n = out.shape[ax]
        p = np.pad(out, [(k, k) if i == ax else (0, 0) for i in (0, 1)], constant_values=fill)
        acc = out.copy()
        for d in range(2 * k + 1):
            acc = op(acc, p[d:d + n] if ax == 0 else p[:, d:d + n])
        out = acc
    return out


def vanilla_assets():
    from repair_campaign_river_roots import Assets, G
    from analyze_reference import inventory
    a = Assets(original_only=True)
    for e in inventory(G / 'models.pack', False)['entries']:
        a.index.setdefault(e['path'], (G / 'models.pack', e))
    return a


def leaf_parts(b):
    """(head, model names, record list [(start, end, model index, mode)], tail) of a leaf."""
    starts = [m.start() for m in MODEL.finditer(b) if m.start() >= 8 and struct.unpack_from('<H', b, m.start() - 8)[0] == 2
              and struct.unpack_from('<H', b, m.start() - 2)[0] == len(m.group()) and 0 < struct.unpack_from('<I', b, m.start() - 6)[0] < 10000]
    assert len(starts) == 1
    s = starts[0]
    n = struct.unpack_from('<I', b, s - 6)[0]
    pos = s - 2
    models = []
    for _ in range(n):
        le = struct.unpack_from('<H', b, pos)[0]
        models.append(b[pos + 2:pos + 2 + le].decode())
        pos += 2 + le
    count = struct.unpack_from('<I', b, pos)[0]
    p = pos + 4
    recs = []
    for _ in range(count):
        idx = struct.unpack_from('<I', b, p + 2)[0]
        le = struct.unpack_from('<H', b, p + 105)[0]
        recs.append((p, p + 131 + le, idx, b[p + 107:p + 107 + le].decode()))
        p += 131 + le
    return b[:s - 8], models, recs, b[p:]


def build_leaf(head, tail, items):
    """items: [(model, record bytes, matrix 12 floats)] -> leaf bytes."""
    models = []
    for m, _, _ in items:
        if m not in models:
            models.append(m)
    out = bytearray(head) + struct.pack('<HI', 2, len(models))
    for m in models:
        e = m.encode()
        out += struct.pack('<H', len(e)) + e
    out += struct.pack('<I', len(items))
    for m, rec, mat in items:
        r = bytearray(rec)
        struct.pack_into('<I', r, 2, models.index(m))
        struct.pack_into('<12f', r, 24, *mat)
        out += r
    return bytes(out + tail)


def family(model):
    m = model.lower()
    if '/mountains/cold/' in m:
        return 'cold'
    if '/mountains/arid/' in m:
        return 'arid'
    return None


def vanilla_clusters(va, variant):
    """Pure cold or arid vanilla clusters: one root, one '.31' leaf, mountain models only."""
    pre = f'terrain/campaigns/{variant}/'
    es, assets = pb.unbundle(va.read(pre + 'global_props.bin'))
    rootkey = pre + 'bmd_objects.bin'
    _, roots = decode(assets[rootkey])
    vb = va.read(pre + 'lf_height_map.dds')
    vh, vw = struct.unpack_from('<2I', vb, 12)
    vg = cw.LEVEL + (np.frombuffer(vb, '<u2', offset=128).reshape(vh, vw)[::-1].astype(np.float64) / 65535 - cw.NORM) * cw.GAIN

    def vground(x, z):
        u = min(max(x / cw.DX * 8, 0), vw - 1.001)
        v = min(max(z / cw.DZ * 8, 0), vh - 1.001)
        u0, v0 = int(u), int(v)
        fu, fv = u - u0, v - v0
        return (vg[v0, u0] * (1 - fu) + vg[v0, u0 + 1] * fu) * (1 - fv) + (vg[v0 + 1, u0] * (1 - fu) + vg[v0 + 1, u0 + 1] * fu) * fv

    out = []
    for rt in roots:
        if max(abs(v) for v in rt['matrix'][12:15]) > 1e-6:
            continue
        _, ch = decode(assets[rt['path']])
        if len(ch) != 1 or not ch[0]['path'].endswith('.31.bin'):
            continue
        leaf = assets[ch[0]['path']]
        try:
            head, models, recs, tail = leaf_parts(leaf)
        except AssertionError:
            continue
        fams = {family(models[i]) for _, _, i, _ in recs}
        if len(fams) != 1 or None in fams or not recs:
            continue
        items = []
        for s, e, i, mode in recs:
            m = struct.unpack_from('<12f', leaf, s + 24)
            if m[10] < -100:
                continue
            dy = m[10] - vground(m[9], m[11]) if mode == 'BHM_IGNORE' else None
            items.append(dict(model=models[i], rec=leaf[s:e], m=m, mode=mode, dy=dy))
        if not items:
            continue
        cxz = np.mean([[it['m'][9], it['m'][11]] for it in items], axis=0)
        out.append(dict(family=fams.pop(), root=rt, parent=assets[rt['path']], child=ch[0], head=head, tail=tail,
                        items=items, centre=cxz, rootrec=assets[rootkey][rt['start']:rt['end']]))
    return out, es, assets


def measure_vanilla(va, variant, eligible, nat):
    """Vanilla's own rate per family: instances of its cold / arid mountain models (every leaf record of the vanilla
    global_props.bin, moved by the native shift) standing on hexes of the native rectangle that pass this step's
    eligibility test for that family (eligible[f], the candidate test without the canvas-only ~nat), per such hex."""
    from translate_campaign_lake_instances import records
    es, assets = pb.unbundle(va.read(f'terrain/campaigns/{variant}/global_props.bin'))
    H, W = nat.shape
    hits = {f: {'instances': 0, 'leaves': set(), 'all_native': 0} for f in eligible}
    for k, b in assets.items():
        if b'/mountains/' not in b.lower():
            continue
        try:
            rs = records(b)
        except AssertionError:
            continue
        for r in rs:
            f = family(r['model'])
            if f not in eligible:
                continue
            q, rr = (int(v) for v in cw.world_hex(r['matrix'][9] + cw.NATIVE_SHIFT_WORLD[0], r['matrix'][11] + cw.NATIVE_SHIFT_WORLD[1]))
            if not (0 <= q < W and 0 <= rr < H and nat[rr, q]):
                continue
            hits[f]['all_native'] += 1
            if eligible[f][rr, q]:
                hits[f]['instances'] += 1
                hits[f]['leaves'].add(k)
    out = {}
    for f, h in hits.items():
        n = int((eligible[f] & nat).sum())
        out[f] = {'eligible_native_hexes': n, 'instances_on_eligible': h['instances'], 'clusters_on_eligible': len(h['leaves']),
                  'native_instances_all': h['all_native'], 'rate_per_eligible_hex': h['instances'] / n if n else None}
    return out


def run(apply, preview=None, measure=False):
    import biome_profiles as _bp
    import map_hex
    import strategic_ink
    from fastops import grow_steps
    va = vanilla_assets()
    plan = json.loads((A / 'work/current-region-design.json').read_text(encoding='utf-8'))
    m = map_hex.read(A / 'work/final-map/map.hex')
    c = m['cells']
    H, W = c.shape[:2]
    names = m['land_regions'] + m['sea_regions']
    land = (c[..., 0] & 3) != 1
    imp = (c[..., 2] & 8) > 0
    road = ((c[..., 3].astype(np.int32) >> 1) & 63) > 0
    ids = map_hex.region_ids(c)
    npl = np.array(['non_playable' in n for n in names] + [True])[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
    biome, bnames, _, elev = strategic_ink.load()
    nat = np.zeros((H, W), bool)
    nat[cw.NATIVE_HEX_SLICE] = True
    report = {'variants': []}
    for variant in cw.TERRAIN_VARIANTS:
        pre = f'terrain/campaigns/{variant}/'
        surface = np.load(A / f'analysis/polished-surface-{variant}.npy')
        rr_, qq_ = np.mgrid[:H, :W]
        hh = surface[np.minimum(2 * rr_ + qq_ % 2 + 1, surface.shape[0] - 1), np.minimum(2 * qq_ + 1, surface.shape[1] - 1)].astype(np.float32)
        gy, gx = np.gradient(hh)
        slope = np.hypot(gx, gy)
        # Topographic position: hex height minus the 7x7 hex mean (as the vanilla survey).
        # Summed in float64: float32 cumulative sums over the whole map erred by up to 0.048
        # against the +-0.05 TPI_FACTOR steps.
        k = 3
        pad = np.pad(hh.astype(np.float64), k, mode='edge')
        cs = pad.cumsum(0).cumsum(1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        box = (cs[2 * k + 1:, 2 * k + 1:] - cs[:-2 * k - 1, 2 * k + 1:] - cs[2 * k + 1:, :-2 * k - 1] + cs[:-2 * k - 1, :-2 * k - 1]) / (2 * k + 1) ** 2
        tpi = hh - box[:H, :W]
        play = land & ~npl
        reach = grow_steps(play, REACH) <= REACH
        wet = ~land | (hh <= .25)
        dwater = grow_steps(wet, WATER_CLEAR)
        droad = grow_steps(road, ROAD_CLEAR)
        towns = np.array([t['town'] for t in plan['regions']])
        town_m = np.zeros((H, W), bool)
        town_m[towns[:, 1], towns[:, 0]] = True
        town_m |= ((c[..., 2] >> 4) > 0)   # every settlement slot
        dtown = grow_steps(town_m, TOWN_CLEAR)
        relief = window_extreme(hh, 3, np.maximum, -np.inf) - window_extreme(hh, 3, np.minimum, np.inf)
        free_all = land & imp & reach & (dwater > WATER_CLEAR) & (droad > ROAD_CLEAR) & (dtown > TOWN_CLEAR) & (relief >= min(MIN_RELIEF.values()))
        free = free_all & ~nat
        slots = (c[..., 2] >> 4) > 0
        snowy = [bnames.index(n) for n in ('snow', 'alpine', 'tundra') if n in bnames]
        dry = [bnames.index(n) for n in ('desert', 'steppe', 'grass', 'savanna') if n in bnames]
        # --- GROUND MATERIALS --- cold line by latitude (ground_materials.SNOWLINE: 3000 m south of 47 N)
        import ground_materials as _gm
        _z = np.load(A / 'analysis/west-real-terrain.npz')
        _lat = _z['lat'][np.minimum(2 * rr_ + qq_ % 2, _z['lat'].shape[0] - 1), np.minimum(2 * qq_, _z['lat'].shape[1] - 1)]
        del _z
        cold = (elev >= _gm.snowline(_lat)) | (np.isin(biome, snowy) & (elev >= 1800))
        _egy, _egx = np.gradient(elev)
        real_ridge = np.interp(np.hypot(_egx, _egy), *REAL_RIDGE).astype(np.float32)
        del _egy, _egx, _lat
        # --- end GROUND MATERIALS
        arid = ~cold & np.isin(biome, dry) & (elev >= 600)
        if measure:   # --- MOUNTAIN TARGET --- vanilla's rate per eligible hex (biome_profiles props MOUNTAIN_TARGET)
            fams = (('cold', cold), ('arid', arid))
            out = {'candidate_test': measure_vanilla(va, variant, {f: free_all & msk & (relief >= MIN_RELIEF[f]) for f, msk in fams}, nat),
                   'site_test': measure_vanilla(va, variant, {f: land & imp & msk & (relief >= MIN_RELIEF[f]) for f, msk in fams}, nat),
                   'impassable_family': measure_vanilla(va, variant, {f: land & imp & msk for f, msk in fams}, nat)}
            for name, msk_of in (('candidate_test', lambda f, msk: free_all & msk & (relief >= MIN_RELIEF[f])),
                                 ('site_test', lambda f, msk: land & imp & msk & (relief >= MIN_RELIEF[f])),
                                 ('impassable_family', lambda f, msk: land & imp & msk)):
                for f, msk in fams:
                    out[name][f]['canvas_hexes'] = int((msk_of(f, msk) & ~nat).sum())
                    rt = out[name][f]['rate_per_eligible_hex']
                    out[name][f]['canvas_target'] = round(rt * out[name][f]['canvas_hexes']) if rt else None
            return out
        clusters, _, _ = vanilla_clusters(va, variant)
        es, assets = pb.unbundle((S / (pre + 'global_props.bin')).read_bytes())   # the staged bundle (lake leaves done)
        by_fam = {f: [x for x in clusters if x['family'] == f] for f in FAMILY_SCALE}
        dens = np.interp(slope, SLOPE_DENS[0], SLOPE_DENS[1]) * np.interp(tpi, TPI_FACTOR[0], TPI_FACTOR[1]) / 1000
        dens = dens * real_ridge   # --- GROUND MATERIALS --- real ridges/steep slopes
        target = {}                # --- GROUND MATERIALS --- instances per family at the vanilla rate
        placed, taken = [], np.zeros((H, W), bool)
        stats = {f: {'clusters': 0, 'instances': 0, 'candidates': 0, 'triangles': 0} for f in FAMILY_SCALE}
        order = []
        for f, msk in (('cold', cold), ('arid', arid)):
            cand = free & msk & (relief >= MIN_RELIEF[f])
            stats[f]['candidates'] = int(cand.sum())
            mean_n = np.mean([len(x['items']) for x in by_fam[f]]) if by_fam[f] else 1
            r0, q0 = np.nonzero(cand)
            p = dens[r0, q0] * FAMILY_SCALE[f] / mean_n
            target[f] = int(round(_bp.CONSTS['props']['MOUNTAIN_TARGET']['rate'][f] * _bp.CONSTS['props']['MOUNTAIN_TARGET']['scale'][f] * len(r0)))   # --- MOUNTAIN TARGET --- vanilla rate x candidates
            stats[f]['target'] = target[f]
            pick = site_u(variant, f, q0, r0, ACCEPT) < p * SEED_OVERSAMPLE[f]   # --- STATELESS --- oversampled seeds
            for q, r in zip(q0[pick], r0[pick]):
                order.append((f, int(q), int(r)))
        # --- STATELESS --- spacing resolved in the hash order of (q, r) (ties: q, r), not a shuffle of a shared stream
        if order:
            oq = np.array([o[1] for o in order]); orr = np.array([o[2] for o in order])
            order = [order[i] for i in np.lexsort((orr, oq, site_u(variant, None, oq, orr, ORDER)))]
        instances = 0
        num = FIRST_PREFAB
        rootkey = pre + 'bmd_objects.bin'
        root = assets[rootkey]
        cp, rs = decode(root)
        if preview:   # --- GROUND MATERIALS --- offline preview on an already-processed stage: drop the earlier run's roots
            _st = [x for x in rs if x['path'].split('.')[-2].isdigit() and FIRST_PREFAB <= int(x['path'].split('.')[-2]) < FIRST_PREFAB + 900]
            if _st:
                _drop = {x['path'] for x in _st} | {x['path'][:-4] + '.31.bin' for x in _st}
                root, _ = pb.prefab_list(root, cp, rs, [x for x in rs if x not in _st], [])
                assets = {k: v for k, v in assets.items() if k not in _drop}; assets[rootkey] = root
                es = [(k, o) for k, o in es if k not in _drop]
                cp, rs = decode(root)
                print('preview: dropped', len(_st), 'earlier frontier mountain roots', flush=True)
        have = {r['path'] for r in rs}
        new_assets, new_roots = {}, []
        for f, q, r in order:
            if instances >= MAX_INSTANCES or not by_fam[f]:
                break
            if taken[max(0, r - int(SPACING)):r + int(SPACING) + 1, max(0, q - int(SPACING)):q + int(SPACING) + 1].any():
                continue
            cl = by_fam[f][min(int(site_u(variant, f, q, r, PICK) * len(by_fam[f])), len(by_fam[f]) - 1)]   # --- STATELESS ---
            cx, cz = cw.block_centre(q, r)
            single = len(cl['items']) == 1
            u = float(site_u(variant, f, q, r, TURN))
            turn = 2 * math.pi * u if single else (math.pi if u < .5 else 0.)
            ca, sa = math.cos(turn), math.sin(turn)
            items, ok, cover = [], True, []
            for it in cl['items']:
                mt = list(it['m'])
                dx, dz = mt[9] - cl['centre'][0], mt[11] - cl['centre'][1]
                if single:
                    dx = dz = 0.
                x, z = cx + ca * dx - sa * dz, cz + sa * dx + ca * dz
                hq, hr = (int(v) for v in cw.world_hex(x, z))
                if not (0 <= hq < W and 0 <= hr < H):
                    ok = False
                    break
                sc = float(np.mean([math.sqrt(sum(v * v for v in mt[i:i + 3])) for i in (0, 3, 6)]))
                reach_h = max(1., sc * model_radius(va, it['model']) / cw.HEX_DX * FOOTPRINT_CORE)
                R = int(reach_h) + 1
                y0, y1, x0, x1 = max(0, hr - R), min(H, hr + R + 1), max(0, hq - R), min(W, hq + R + 1)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                disc = np.hypot(xx - hq, yy - hr) <= reach_h
                if not (land[y0:y1, x0:x1][disc].all() and not wet[y0:y1, x0:x1][disc].any() and not slots[y0:y1, x0:x1][disc].any()
                        and not taken[y0:y1, x0:x1][disc].any() and imp[y0:y1, x0:x1][disc].mean() >= FOOTPRINT_IMPASSABLE[f]
                        and road[y0:y1, x0:x1][disc].mean() <= FOOTPRINT_ROAD and not nat[y0:y1, x0:x1][disc].any()):
                    ok = False
                    break
                cover.append((slice(y0, y1), slice(x0, x1), disc))
                # Orientation: the vanilla yaw composed with the turn (same vertical axis).
                r3 = np.array(mt[:9]).reshape(3, 3)
                rot = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]]) if single else np.diag([ca, 1, ca])
                m3 = (r3 @ rot).ravel().tolist()
                if it['mode'] == 'BHM_IGNORE':
                    tx, tz = x / cw.DX * 2, z / cw.DZ * 2
                    gy_ = float(surface[min(int(tz), surface.shape[0] - 1), min(int(tx), surface.shape[1] - 1)])
                    y = gy_ + it['dy']
                else:
                    y = mt[10]
                items.append((it['model'], it['rec'], m3 + [x, y, z]))
            if not ok:
                continue
            region = names[ids[r, q]] if 0 <= ids[r, q] < len(m['land_regions']) else '3k_main_reg_non_playable'
            while f'{pre}bmd_objects.{region}.{num}.bin' in have:
                num += 1
            assert num < FIRST_PREFAB + 900
            npath = f'{pre}bmd_objects.{region}.{num}.bin'
            nleaf = f'{pre}bmd_objects.{region}.{num}.31.bin'
            num += 1
            have.add(npath)
            leaf = build_leaf(cl['head'], cl['tail'], items)
            chk = leaf_records(leaf)
            assert len(chk) == len(items) and all(abs(a['matrix'][9] - b[2][9]) < 1e-3 for a, b in zip(chk, items))
            pbody = cl['parent']
            cc = cl['child']
            # Child record: the new leaf and the site's region (it kept the vanilla template's region).
            parent = pbody[:cc['start']] + pb.rebind(pbody[cc['start']:cc['end']], cc, region, nleaf) + pbody[cc['end']:]
            assert [(x['path'], x['region']) for x in decode(parent)[1]] == [(nleaf, region)]
            new_assets[npath] = parent
            new_assets[nleaf] = leaf
            new_roots.append(pb.rebind(cl['rootrec'], cl['root'], region, npath))   # root record: the site's region
            for ys, xs, disc in cover:
                taken[ys, xs] |= disc
            taken[max(0, r - 1):r + 2, max(0, q - 1):q + 2] = True
            stats[f]['clusters'] += 1
            stats[f]['instances'] += len(items)
            stats[f]['triangles'] += sum(model_triangles(va, it[0]) or 0 for it in items)   # LOD0 x instances
            instances += len(items)
            placed.append({'family': f, 'hex': [q, r], 'region': region, 'root': npath, 'template': cl['root']['path'],
                           'models': [it[0].split('/')[-1] for it in items], 'xyz': [[round(v, 3) for v in it[2][9:12]] for it in items]})
        # Root list: vanilla-staged records plus the new ones (a rerun starts from the staged
        # bundle that reconcile rebuilt from terrain-before-polish, so nothing is doubled).
        stale = [x for x in rs if x['path'].split('.')[-2].isdigit() and FIRST_PREFAB <= int(x["path"].split(".")[-2]) < FIRST_PREFAB + 900]
        assert not stale, 'staged bundle already holds frontier mountain roots: run after reconcile_terrain_prefabs'
        updated, _ = pb.prefab_list(root, cp, rs, rs, new_roots)
        out = pb.rebundle(es, assets, rootkey, updated, new_assets)
        report['variants'].append({'variant': variant, 'roots_added': len(new_roots), 'instances': instances, 'by_family': stats,
                                   'bundle_bytes': [parse(out)[1] + sum(len(b) for k, b in assets.items()), len(out)], 'placed': placed})
        if preview:
            from pathlib import Path
            Path(preview).write_bytes(out)
        if apply:
            pb.write_stage(S, pre, out, updated, new_assets)
        print(variant, 'frontier mountain props:', len(new_roots), 'clusters,', instances, 'instances', json.dumps(stats), 'applied' if apply else 'dry run', flush=True)
    report['applied'] = apply
    report['live_verified'] = False
    # Only an applied run writes the report the later steps and audits read; a preview reports
    # next to its bundle, a dry run to its own file (both used to replace the applied report).
    if preview:
        from pathlib import Path
        rpath = Path(str(preview) + '.report.json')
    elif apply:
        rpath = A / 'analysis/frontier-mountain-props.json'
    else:
        rpath = A / 'analysis/frontier-mountain-props.dry-run.json'
    report['preview'] = str(preview) if preview else None
    rpath.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('report:', rpath, flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--preview', help='write the bundle to this file only (offline renders)')
    ap.add_argument('--measure-vanilla-rate', action='store_true', help='print vanilla instances per eligible hex (rule table MOUNTAIN_TARGET)')
    a = ap.parse_args()
    if a.apply and a.preview:
        ap.error('--preview drops the earlier frontier roots for an offline bundle; it does not combine with --apply')
    if a.measure_vanilla_rate:
        print(json.dumps(run(False, measure=True), indent=1))
        return
    run(a.apply, a.preview)


if __name__ == '__main__':
    main()

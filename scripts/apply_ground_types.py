"""Field-battle ground types (map.hex bytes 5/6) on the new regions from real relief, learned from vanilla.

The new regions were laid out filled with plains; later steps only overlay forest (apply_forest_ground),
wetland (apply_climate) and desert, so 58% of their passable playable hexes were plains and hills were
almost absent (analysis/region-diversity-2026-10-25.md section 2). This step learns, the way apply_climate
learns climate, "RESOLVE biome x ETOPO elevation x local relief -> ground family" from the vanilla map and
sets the family's ground on the new regions:

  * features per hex (apply_climate.hex_geo, analysis/west-real-terrain.npz, the whole canvas): majority
    RESOLVE biome, mean ETOPO metres, relief = std (float64) of the hex-mean metres over the hex and its six
    neighbours. Vanilla takes the same fields at the native slice (as apply_climate.learn_vanilla);
  * learned on vanilla passable playable land (no beach, wetland, river crossing, towns and their ring):
    family flat (grassland, light/dense forest), hilly (hilly_*), desert. Key (biome, elevation band, relief
    band), else (biome, relief), (elevation, relief), (relief), the first with >= LEARN_MIN vanilla hexes;
  * per hex the family is drawn from the key's shares with a smooth uniform dither field (rank of an fbm,
    seeded): hilly where u < p_hilly, desert where u > 1 - p_desert, else flat; a hex already hilly stays
    hilly (the step adds relief, it never flattens authored ridges);
  * supply (desert and tundra cost -10 military supplies, hills -5): desert only on the real desert core
    (RESOLVE desert, DESERT_WATER_CLEAR hexes from water and Natural Earth rivers, DESERT_OASIS_CLEAR from the
    oases, SUPPLY_TOWN_CLEAR from towns); elsewhere the desert share goes to flat / hilly in the learned
    ratio, and desert laid by earlier steps there becomes grassland / hilly_grassland. Tundra by rule
    (vanilla has none): RESOLVE tundra/snow or >= TUNDRA_ELEV m, not near towns; its hilly hexes stay hilly;
  * open ground (OPEN_FROM: plains, grassland, hilly_*, desert, tundra, passable mountain) takes OPEN_TO;
    forest ground (light_forest) only turns hilly_light_forest where the hex draws hilly (FOREST_TO), so trees
    and forest ground stay where apply_forest_ground put them. Vanilla never has plains, hilly_plains or
    passable mountain (54 of 105,066 mountain hexes); none is produced.
Applied on passable playable land of tw3k_ regions (inside the native slice only where their ground differs
from the vanilla hex: the authored ground of the Korea, Hexi, Juyan, Wuyuan, Nanzhong layouts, and on the revealed
filler, revealed_filler.revealed_hex: vanilla non-playable land now playable, treated as canvas; the vanilla
learning sample is vanilla's own playable land only, revealed_filler.vanilla_playable_hex), kinds 0 and 3 (no water, no
river bank beach), not next to a ford (SDK area class 8), not wetland or river crossing ground, more than
TOWN_RING hexes from settlement slots and authored towns. Road hexes take ground as any other hex (vanilla
road hexes carry ordinary ground: grassland 60, hilly_light_forest 21, light_forest 11, hilly_grassland 8 %).
Impassable hexes, passability, region ids, roads, rivers, ports and slots: untouched (only bytes 5/6 change).

Order: after apply_rivers, before apply_forest_ground and apply_climate. apply_forest_ground then turns
grassland -> light_forest and hilly_grassland -> hilly_light_forest in its forest zone (the forest / hill
combination); apply_climate's wetland takes flat hexes (grassland is one of its sources). On a map that is
already forest-grounded, the forest hexes this step switches are rewritten in apply_forest_ground's record
(analysis/forest-ground-hexes.npz: src -> FOREST_BASE of the new ground, dst -> new ground), so a later
forest fade reverts them to this step's open ground. This step records what it set in
analysis/ground-types-hexes.npz (q, r, ground before, ground set); a rerun is a no-op.

  python apply_ground_types.py                  # dry run: report analysis/ground-types.json
  python apply_ground_types.py --apply          # back up and write work/final-map/map.hex
  python apply_ground_types.py --write-to X.hex # write the map elsewhere only (offline checks)
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

import campaign_world as cw
import map_hex
from hex_components import area_class, hex_dilate, hex_neighbour_values, region_areas
from revealed_filler import revealed_hex, vanilla_playable_hex
from terrain_noise import fbm
import biome_profiles as _bp

_K = _bp.consts('apply_ground_types')   # values: biome_profiles.CONSTS['apply_ground_types']
ELEV_BINS, ROUGH_BINS, LEARN_MIN = _K['ELEV_BINS'], _K['ROUGH_BINS'], _K['LEARN_MIN']
FAMILY = _K['FAMILY']
FAMILIES = ('flat', 'hilly', 'desert')
OPEN_FROM, OPEN_TO = _K['OPEN_FROM'], _K['OPEN_TO']
FOREST_FROM, FOREST_TO, FOREST_BASE = _K['FOREST_FROM'], _K['FOREST_TO'], _K['FOREST_BASE']
TOWN_RING = _K['TOWN_RING']

A = cw.WORKSPACE
MAP = A / 'work/final-map/map.hex'
REPORT = A / 'analysis/ground-types.json'
OWNED = A / 'analysis/ground-types-hexes.npz'
FOREST_OWNED = A / 'analysis/forest-ground-hexes.npz'


def ground(cells):
    return (cells[..., 5].astype(np.int32) >> 4) + ((cells[..., 6].astype(np.int32) & 7) << 4) - 1


def set_ground(cells, mask, index):
    v = int(index) + 1
    cells[mask, 5] = (cells[mask, 5] & 0x0F) | ((v & 15) << 4)
    cells[mask, 6] = (cells[mask, 6] & 0xF8) | ((v >> 4) & 7)


def _bin(v, edges):
    return np.searchsorted(np.asarray(edges, np.float64), np.asarray(v, np.float64), side='right')


def relief(elev):
    """Std (m, float64) of the hex-mean metres over each hex and its six neighbours."""
    e = np.asarray(elev, np.float64)
    stack = np.concatenate([e[None], hex_neighbour_values(e)])
    return stack.std(0)


def town_seed(cells, H, W):
    """Settlement slots, sprawl and authored towns."""
    seed = ((cells[..., 2] >> 4) > 0) | ((cells[..., 3] & 1) > 0)
    plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    for t in sorted(plan['regions'], key=lambda t: t['region']):
        q, r = t['town']
        if 0 <= q < W and 0 <= r < H:
            seed[r, q] = True
    return seed


def grow(mask, steps):
    """Hexes within steps hex steps of mask."""
    out = np.asarray(mask, bool)
    for _ in range(steps):
        out = hex_dilate(out)
    return out


def town_mask(cells, H, W):
    """Settlement slots, sprawl and authored towns, grown TOWN_RING hexes."""
    return grow(town_seed(cells, H, W), TOWN_RING)


def _in_poly(lon, lat, poly):
    """Even-odd point-in-polygon on lon/lat arrays."""
    inside = np.zeros(lon.shape, bool)
    pts = [tuple(map(float, v)) for v in poly]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        if y0 == y1:
            continue
        cross = (y0 > lat) != (y1 > lat)
        inside ^= cross & (lon < x0 + (lat - y0) * (x1 - x0) / (y1 - y0))
    return inside


def oasis_hexes(geo):
    """Oasis hexes on the hex lat/lon: oasis_discs (radius x OASIS_DISC_REACH km), oasis_seeds (their full
    reach, oasis_ground SEED_EDGE x R x (1 + SEED_RIVER)) and the oasis_areas polygons."""
    ok = geo['ok']
    la_ = np.where(ok, geo['lat'], 0.).astype(np.float64); lo_ = np.where(ok, geo['lon'], 0.).astype(np.float64)
    og = _bp.consts('oasis_ground')
    reach = [(lo, la, rad * _K['OASIS_DISC_REACH']) for _, (lo, la, rad) in sorted(_bp.region('oasis_discs')['discs'].items())]
    reach += [(lo, la, og['SEED_EDGE'] * float(np.sqrt(km2 / np.pi)) * (1 + og['SEED_RIVER']))
              for _, (lo, la, km2) in sorted(_bp.oasis_seeds().items())]
    out = np.zeros(ok.shape, bool)
    for lo, la, km in reach:
        out |= np.hypot((lo_ - lo) * 111.32 * np.cos(np.radians(la)), (la_ - la) * 110.6) < km
    for _, poly in sorted(_bp.region('oasis_areas')['polygons'].items()):
        out |= _in_poly(lo_, la_, poly)
    return out & ok


def region_masks(m):
    c = m['cells']
    ids = map_hex.region_ids(c)
    names = m['land_regions'] + m['sea_regions']
    ix = np.where((ids >= 0) & (ids < len(names)), ids, len(names))
    play = np.array(['non_playable' not in n for n in names] + [False])[ix]
    new = np.array([n.startswith('tw3k_') for n in names] + [False])[ix]
    return ids, play, new


def keys(biome, eb, rb):
    """Learned-table keys, finest first: (biome, elev, relief), (biome, relief), (elev, relief), (relief)."""
    b = biome.astype(np.int64)
    return [(b * 16 + eb) * 16 + rb, b * 16 + rb + 100000, eb * 16 + rb + 200000, rb.astype(np.int64) + 300000]


def learn_vanilla(geo, rough):
    """Family counts per key on vanilla passable playable land at the native slice."""
    v = map_hex.read(cw.VANILLA_MAP)
    vc = v['cells']; vh, vw = vc.shape[:2]
    G = v['land_ground_types']
    sl = cw.NATIVE_HEX_SLICE
    g = ground(vc)
    _, play, _ = region_masks(v)
    kind = vc[..., 0] & 3
    fam = np.full(g.shape, -1, np.int32)
    for name, f in FAMILY.items():
        fam[g == G.index(name)] = FAMILIES.index(f)
    play &= vanilla_playable_hex()[sl]      # vanilla's own playable land only, never the (now revealed) filler
    sample = (fam >= 0) & play & ((vc[..., 2] & 8) == 0) & ((kind == 0) | (kind == 3)) & ~town_mask_vanilla(vc) & geo['ok'][sl]
    b = geo['biome'][sl]; e = geo['elev'][sl]; rg = rough[sl]
    eb, rb = _bin(e, ELEV_BINS), _bin(rg, ROUGH_BINS)
    tables = []
    for k in keys(b, eb, rb):
        ks, f = k[sample], fam[sample]
        uk, inv = np.unique(ks, return_inverse=True)
        cnt = np.zeros((len(uk), len(FAMILIES)), np.float64)
        np.add.at(cnt, (inv, f), 1.)
        tables.append((uk, cnt))
    # report: vanilla ground share per biome on the learning sample
    by_biome = {}
    for bi, bn in enumerate(geo['names']):
        s = sample & (b == bi)
        if s.sum() >= 50:
            cnt = np.bincount(g[s], minlength=len(G))
            by_biome[bn] = {'hexes': int(s.sum()), **{G[i]: round(float(cnt[i] / s.sum()), 3) for i in np.argsort(-cnt, kind='stable') if cnt[i]}}
    return tables, {'sample_hexes': int(sample.sum()), 'by_biome': by_biome}


def town_mask_vanilla(vc):
    seed = ((vc[..., 2] >> 4) > 0) | ((vc[..., 3] & 1) > 0)
    out = seed.copy()
    for _ in range(TOWN_RING):
        out = hex_dilate(out)
    return out


def shares(tables, biome, eb, rb):
    """(3, H, W) family shares from the finest key with >= LEARN_MIN vanilla hexes; level used per hex."""
    out = np.full((len(FAMILIES),) + biome.shape, np.nan, np.float64)
    level = np.full(biome.shape, -1, np.int8)
    for lv, (k, (uk, cnt)) in enumerate(zip(keys(biome, eb, rb), tables)):
        n = cnt.sum(1)
        good = n >= LEARN_MIN
        uk_g, p = uk[good], cnt[good] / n[good, None]
        if not len(uk_g):
            continue
        pos = np.clip(np.searchsorted(uk_g, k), 0, len(uk_g) - 1)
        hit = (level < 0) & (uk_g[pos] == k)
        for f in range(len(FAMILIES)):
            out[f][hit] = p[pos[hit], f]
        level[hit] = lv
    return out, level


def dither(H, W):
    """Uniform [0, 1) field with fbm's smooth structure: rank of the fbm values (stable sort)."""
    grain, octaves, seed = _K['DITHER']
    r, q = np.mgrid[0:H, 0:W].astype(np.float64)
    f = np.asarray(fbm(q, r, (0, 0), grain, octaves, seed=seed), np.float64).ravel()
    rank = np.empty(f.size, np.float64)
    rank[np.argsort(f, kind='stable')] = np.arange(f.size, dtype=np.float64)
    return ((rank + .5) / f.size).reshape(H, W)


# campaign_ground_types movement_cost and the ground-type effect bundles' military supplies (vanilla DB,
# work/standalone_kit/raw_data/db, 2026-10-25); report only.
MOVE = {'plains': 120, 'grassland': 120, 'light_forest': 120, 'tundra': 120, 'desert': 240, 'hilly_grassland': 240,
        'hilly_plains': 240, 'hilly_light_forest': 360, 'mountain': 360, 'dense_forest': 360, 'hilly_dense_forest': 360, 'wetland': 360}
SUPPLY = {'plains': 0, 'grassland': 0, 'desert': -10, 'tundra': -10, 'mountain': -10, 'light_forest': -5, 'dense_forest': -5,
          'hilly_grassland': -5, 'hilly_plains': -5, 'hilly_light_forest': -5, 'hilly_dense_forest': -5, 'wetland': -5}


def gameplay(g, mask, G):
    """Mean movement cost and military-supply effect over mask (river crossings left out)."""
    names = np.array(G)[g[mask]]
    sel = np.isin(names, list(MOVE))
    n = max(1, int(sel.sum()))
    return {'movement_cost': round(float(sum(MOVE[x] for x in names[sel].tolist())) / n, 1),
            'military_supplies': round(float(sum(SUPPLY[x] for x in names[sel].tolist())) / n, 2)}


def distribution(g, mask, G):
    n = max(1, int(mask.sum()))
    cnt = np.bincount(g[mask], minlength=len(G))
    return {G[i]: round(float(cnt[i] / n), 3) for i in np.argsort(-cnt, kind='stable') if cnt[i]}


def compute(m):
    """The ground types for map dict m, in memory: {'after': new cells, ...the fields main() reports}. Reads the
    vanilla map and analysis/west-real-terrain.npz (apply_climate.hex_geo) and the design towns (town_seed);
    writes nothing. main() adds the forest-record rewrite, the report and the writes; the map bake calls this."""
    import apply_climate as ac
    cells = m['cells']; H, W = cells.shape[:2]
    G = m['land_ground_types']
    g = ground(cells)
    land = (g >= 0) & (g < len(G))
    kind = cells[..., 0] & 3
    imp = (cells[..., 2] & 8) > 0
    ids, play, new = region_masks(m)
    inside = np.zeros((H, W), bool); inside[cw.NATIVE_HEX_SLICE] = True
    geo = ac.hex_geo(H, W)
    rough = relief(geo['elev'])
    tables, vinfo = learn_vanilla(geo, rough)
    bn = geo['names']
    eb, rb = _bin(geo['elev'], ELEV_BINS), _bin(rough, ROUGH_BINS)
    p, level = shares(tables, geo['biome'], eb, rb)
    isb = lambda names: np.isin(geo['biome'], [bn.index(n) for n in names if n in bn])
    # Supply: desert only on the real desert core; its learned share elsewhere goes to flat / hilly.
    town_far = ~grow(town_seed(cells, H, W), _K['SUPPLY_TOWN_CLEAR'])
    water = ((cells[..., 0] & 3) == 1) | np.isin(g, [G.index(n) for n in ('river', 'river_navigable_crossing')]) | geo['river']
    desert_bio = isb(_K['DESERT_BIOMES'])
    desert_ok = (desert_bio & ~grow(~desert_bio & geo['ok'], _K['DESERT_EDGE_CLEAR']) & ~grow(water, _K['DESERT_WATER_CLEAR'])
                 & ~grow(oasis_hexes(geo), _K['DESERT_OASIS_CLEAR']) & town_far)
    rest = p[0] + p[1]
    ph = np.where(desert_ok, p[1], p[1] / np.where(rest > 0, rest, 1.))
    pd = np.where(desert_ok, p[2], 0.)
    u = dither(H, W)
    fam = np.where(u < ph, 1, np.where(u > 1 - pd, 2, 0))
    fam = np.where((g == G.index('desert')) & (fam == 0) & desert_ok, 2, fam)   # earlier desert stays desert on the core
    # Wetland candidates (biome_profiles.wetland_candidate_hex, as apply_climate lays them; off the town clearance)
    # draw no hills: in the map bake ground types runs before the wetland, and a hill there lost the marsh
    # (Mekong delta -744 hexes, 2026-10-26). Authored hills (next line) still stay.
    wetc = ac.wetland_candidates(cells, m, geo)[0] & ~ac.town_clear_hex(cells) & geo['ok']
    fam = np.where(wetc & (fam == 1), 0, fam)
    # Hills already on the map (authored ridges, e.g. Korea's Sobaek) stay hills: this step adds relief, never flattens.
    fam = np.where(np.isin(g, [G.index(n) for n in G if n.startswith('hilly_')]), 1, fam)
    tundra = (isb(_K['TUNDRA_BIOMES']) | (geo['elev'] >= _K['TUNDRA_ELEV'])) & town_far
    fam = np.where(tundra & (fam != 1), 3, fam)      # 3 = tundra
    fam_names = FAMILIES + ('tundra',)
    # target hexes
    open_g = np.isin(g, [G.index(n) for n in OPEN_FROM])
    forest_g = np.isin(g, [G.index(n) for n in FOREST_FROM])
    # Inside the native slice only the authored ground of tw3k_ regions (hexes whose ground differs from the
    # vanilla hex: Korea, Hexi, Juyan, Wuyuan, Nanzhong relayouts) and the revealed filler (vanilla non-playable
    # land now playable, revealed_filler.revealed_hex, of any playable region): canvas. Vanilla's own playable
    # hexes with vanilla's ground keep it.
    vm = map_hex.read(cw.VANILLA_MAP); vg = ground(vm['cells'])
    revealed = revealed_hex(cells, m['land_regions'] + m['sea_regions'])
    authored = ~inside.copy()
    authored[cw.NATIVE_HEX_SLICE] = g[cw.NATIVE_HEX_SLICE] != vg
    authored |= revealed
    target = (land & (new | revealed) & play & authored & ~imp & ((kind == 0) | (kind == 3)) & (area_class(cells) != 8)
              & (open_g | forest_g) & ~town_mask(cells, H, W) & geo['ok'] & (level >= 0))
    lut_open = np.array([G.index(OPEN_TO[f]) for f in fam_names])
    lut_forest = np.array([G.index(FOREST_TO[f]) for f in fam_names])
    out = np.where(open_g, lut_open[fam], lut_forest[fam])
    out = np.where(target, out, g)
    changed = out != g
    after = cells.copy()
    for gi in sorted(set(out[changed].tolist())):
        set_ground(after, changed & (out == gi), gi)
    # ---- checks: only bytes 5/6, passability, roads/rivers/ports/slots, SDK areas
    diff_bytes = np.nonzero((after != cells).any((0, 1)))[0].tolist()
    assert set(diff_bytes) <= {5, 6}, diff_bytes
    assert ((after[..., 2] & 8) == (cells[..., 2] & 8)).all()
    assert (ground(after)[~target] == g[~target]).all()
    assert not (changed & vanilla_playable_hex() & ~authored).any(), 'a vanilla-ground vanilla-playable hex changed'
    assert not np.isin(ground(after)[changed], [G.index(n) for n in ('mountain', 'plains', 'hilly_plains', 'dense_forest', 'hilly_dense_forest')]).any()
    # invariant: no hill drawn on a wetland candidate (authored hills there were hills before and are unchanged)
    assert not (changed & wetc & np.isin(ground(after), [G.index(n) for n in G if n.startswith('hilly_')])).any(), 'hill drawn on a wetland candidate'
    areas_before, areas_after = region_areas(ids, cells), region_areas(ids, after)
    assert areas_before == areas_after, 'SDK area classes changed'
    return {'after': after, 'cells': cells, 'H': H, 'W': W, 'G': G, 'g': g, 'land': land, 'kind': kind, 'imp': imp, 'ids': ids, 'play': play, 'new': new, 'inside': inside, 'geo': geo, 'rough': rough, 'tables': tables, 'vinfo': vinfo, 'bn': bn, 'eb': eb, 'rb': rb, 'p': p, 'level': level, 'desert_ok': desert_ok, 'fam': fam, 'fam_names': fam_names, 'target': target, 'out': out, 'changed': changed, 'vm': vm, 'vg': vg, 'revealed': revealed, 'authored': authored, 'diff_bytes': diff_bytes, 'areas_before': areas_before, 'areas_after': areas_after}


def rewrite_forest_record(record, changed, g, after, G):
    """apply_forest_ground's record (q, r, src, dst arrays, rewritten in place): forest hexes this step switched
    flat <-> hilly get dst = their new ground and src = its FOREST_BASE, so a later forest fade reverts them to
    this step's open ground. Returns the entries rewritten (bool array)."""
    fq, fr, fsrc, fdst = record
    hit = changed[fr, fq] & (g[fr, fq] == fdst)
    if hit.any():
        nd = ground(after)[fr[hit], fq[hit]]
        fdst[hit] = nd
        fsrc[hit] = np.array([G.index(FOREST_BASE[G[i]]) for i in nd.tolist()], fsrc.dtype)
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='back up and write work/final-map/map.hex')
    ap.add_argument('--write-to', help='write the resulting map.hex here only')
    ap.add_argument('--diagrams', help='write before/after hex diagrams of a few windows to this directory')
    args = ap.parse_args()
    import apply_climate as ac
    t0 = time.time()
    m = map_hex.read(MAP)
    res = compute(m)
    (after, cells, H, W, G, g, land, kind, imp, ids, play, new, inside, geo, rough, tables, vinfo, bn, eb, rb, p, level, desert_ok, fam, fam_names, target, out, changed, vm, vg, revealed, authored, diff_bytes, areas_before, areas_after) = (res[k] for k in (
        'after', 'cells', 'H', 'W', 'G', 'g', 'land', 'kind', 'imp', 'ids', 'play', 'new', 'inside', 'geo', 'rough', 'tables', 'vinfo', 'bn', 'eb', 'rb', 'p', 'level', 'desert_ok', 'fam', 'fam_names', 'target', 'out', 'changed', 'vm', 'vg', 'revealed', 'authored', 'diff_bytes', 'areas_before', 'areas_after'))
    # ---- apply_forest_ground record: forest hexes switched flat <-> hilly keep a consistent revert
    forest_record = None
    if FOREST_OWNED.exists():
        fq, fr, fsrc, fdst = _load(FOREST_OWNED)
        hit = rewrite_forest_record((fq, fr, fsrc, fdst), changed, g, after, G)
        forest_record = {'path': str(FOREST_OWNED), 'entries': int(len(fq)), 'rewritten': int(hit.sum())}
    # ---- report
    zone = land & new & play & ~inside & ~imp
    zone_in = land & new & play & inside & ~revealed & ~imp
    zone_rv = land & play & revealed & ~imp
    ga = ground(after)
    _, vplay, _ = region_masks(vm)
    vzone = (vg >= 0) & (vg < len(G)) & vplay & ((vm['cells'][..., 2] & 8) == 0)
    sl = cw.NATIVE_HEX_SLICE
    by_biome = {}
    for bi, name in enumerate(bn):
        s = zone & (geo['biome'] == bi)
        vs = vzone & (geo['biome'][sl] == bi)
        if s.sum() >= 50 or vs.sum() >= 50:
            by_biome[name] = {'hexes': int(s.sum()), 'before': distribution(g, s, G), 'after': distribution(ga, s, G),
                              'vanilla_hexes': int(vs.sum()), 'vanilla': distribution(vg, vs, G)}
    report = {
        'map': str(MAP), 'applied': bool(args.apply), 'write_to': args.write_to,
        'scope': 'passable playable land of tw3k_ regions; inside the native slice only hexes whose ground differs from vanilla, plus the revealed filler (any playable region)',
        'distribution_scope': 'new_regions: outside the native slice (analysis/region-diversity-2026-10-25.md section 2 count); vanilla: vanilla map passable playable land',
        'distribution': {'new_regions_hexes': int(zone.sum()), 'before': distribution(g, zone, G), 'after': distribution(ga, zone, G),
                         'vanilla_hexes': int(vzone.sum()), 'vanilla': distribution(vg, vzone, G),
                         'new_regions_in_native_slice_hexes': int(zone_in.sum()), 'in_slice_before': distribution(g, zone_in, G),
                         'in_slice_after': distribution(ga, zone_in, G),
                         'revealed_hexes': int(zone_rv.sum()), 'revealed_before': distribution(g, zone_rv, G),
                         'revealed_after': distribution(ga, zone_rv, G)},
        'gameplay': {'before': gameplay(g, zone, G), 'after': gameplay(ga, zone, G), 'vanilla': gameplay(vg, vzone, G)},
        'desert_core_hexes': int((desert_ok & zone).sum()),
        'by_biome': by_biome,
        'target_hexes': int(target.sum()), 'changed_hexes': int(changed.sum()),
        'revealed_hexes': int(revealed.sum()), 'changed_revealed_hexes': int((changed & revealed).sum()),
        'changes': {f'{G[a]} -> {G[b]}': int(n) for (a, b), n in sorted(
            {(int(a), int(b)): int(((g == a) & (out == b) & changed).sum()) for a, b in set(zip(g[changed].tolist(), out[changed].tolist()))}.items())},
        'family_on_target': {f: int((target & (fam == i)).sum()) for i, f in enumerate(fam_names)},
        'table_level_on_target': {lv: int((target & (level == i)).sum()) for i, lv in enumerate(('biome_elev_relief', 'biome_relief', 'elev_relief', 'relief'))},
        'learned_vanilla': vinfo,
        'learned_keys': {lv: int((t[1].sum(1) >= LEARN_MIN).sum()) for lv, t in zip(('biome_elev_relief', 'biome_relief', 'elev_relief', 'relief'), tables)},
        'relief_quantiles_m': {'new_target': [round(float(x), 1) for x in np.percentile(rough[target], (10, 50, 90))] if target.any() else None,
                               'vanilla_sample': [round(float(x), 1) for x in np.percentile(rough[sl][vzone], (10, 50, 90))]},
        'forest_ground_record': forest_record,
        'sdk_area_counts_unchanged': True, 'changed_bytes': diff_bytes,
        'provenance': {'map_sha256': _sha(MAP), 'west_real_terrain_sha256': _sha(ac.REAL), 'vanilla_sha256': _sha(cw.VANILLA_MAP),
                       'design_sha256': _sha(cw.DESIGN_PATH)},
    }
    if args.diagrams:
        report['diagrams'] = diagrams(args.diagrams, g, ga, cells, geo, G)
    # ---- write
    if args.write_to:
        dst = Path(args.write_to)
        m2 = dict(m); m2['cells'] = after
        map_hex.write(m2, dst)
        import region_connectivity; region_connectivity.check_map(dst)   # land regions stay whole (v17 crash); [to be absorbed by the map bake]
        report['written'] = str(dst)
    if args.apply and changed.any():
        # Backups first, then the records, the map last (a failure before it leaves the old map and records
        # that still describe it only by entries the next run re-derives).
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bak = A / 'backups' / f'map-before-ground-types-{stamp}.hex'
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MAP, bak)
        report['backup'] = str(bak)
        if forest_record and forest_record['rewritten']:
            shutil.copy2(FOREST_OWNED, A / 'backups' / f'forest-ground-hexes-before-ground-types-{stamp}.npz')
            _savez_atomic(FOREST_OWNED, q=fq, r=fr, src=fsrc, dst=fdst)
        # own record: hexes set now (src: the ground before this step, also across reruns), plus earlier
        # entries that still hold what this step set
        parts = []
        lut = {}
        if OWNED.exists():
            oq, orr, osrc, odst = _load(OWNED)
            keep = (ga[orr, oq] == odst) & ~changed[orr, oq]
            parts.append((oq[keep], orr[keep], osrc[keep], odst[keep]))
            lut = {(int(a), int(b)): int(c) for a, b, c in zip(oq, orr, osrc)}
        rr_, qq_ = np.nonzero(changed)
        prev_src = np.array([lut.get((int(a), int(b)), int(c)) for a, b, c in zip(qq_, rr_, g[changed])], np.int32)
        parts.append((qq_, rr_, prev_src, ga[changed]))
        _savez_atomic(OWNED, **{k: np.concatenate([x[i] for x in parts]).astype(np.int32) for i, k in enumerate(('q', 'r', 'src', 'dst'))})
        m['cells'] = after
        map_hex.write(m, MAP)
        import region_connectivity; region_connectivity.check_map(MAP)   # land regions stay whole (v17 crash); [to be absorbed by the map bake]
    report['seconds'] = round(time.time() - t0, 1)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_name(REPORT.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, REPORT)
    brief = {k: report[k] for k in ('distribution', 'gameplay', 'desert_core_hexes', 'target_hexes', 'changed_hexes', 'revealed_hexes', 'changed_revealed_hexes', 'changes', 'family_on_target', 'table_level_on_target', 'forest_ground_record')}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print('dry run (no map written)' if not (args.apply or args.write_to) else 'written', REPORT, f'{report["seconds"]}s')


COLOURS = {'dense_forest': (20, 80, 30), 'desert': (226, 200, 120), 'grassland': (150, 200, 100), 'hilly_dense_forest': (10, 60, 20),
           'hilly_grassland': (170, 140, 70), 'hilly_light_forest': (70, 120, 50), 'hilly_plains': (200, 160, 110), 'light_forest': (60, 160, 70),
           'mountain': (120, 110, 105), 'plains': (225, 235, 160), 'river': (60, 110, 200), 'river_navigable_crossing': (90, 160, 230),
           'tundra': (205, 215, 225), 'wetland': (80, 160, 160)}
WINDOWS = {'tianshan_foothills': (84.5, 43.6), 'tianshan_south_kucha': (83.0, 41.9), 'mongolian_steppe': (104.5, 47.0),
           'korea_hills': (127.6, 36.4), 'indochina_annamite': (107.0, 15.0), 'khorat_hills': (101.8, 16.0)}


def diagrams(out_dir, before, after, cells, geo, G, radius=36, scale=7):
    """Before / after hex diagrams of the ground (colour), shaded by ETOPO relief, towns and impassable marked."""
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    H, W = before.shape
    imp = (cells[..., 2] & 8) > 0
    sea = (cells[..., 0] & 3) == 1
    slot = (cells[..., 2] >> 4) > 0
    road = ((cells[..., 3].astype(np.int32) >> 1) & 63) > 0
    lat, lon = np.nan_to_num(geo['lat'], nan=999.), np.nan_to_num(geo['lon'], nan=999.)
    e = geo['elev'].astype(np.float64)
    written = []
    for name, (lo, la) in sorted(WINDOWS.items()):
        d = (lon - lo) ** 2 + (la - lat) ** 2
        r0, q0 = np.unravel_index(int(np.argmin(d)), d.shape)
        q_lo, q_hi = max(0, q0 - radius), min(W, q0 + radius); r_lo, r_hi = max(0, r0 - radius), min(H, r0 + radius)
        nq, nr = q_hi - q_lo, r_hi - r_lo
        panels = []
        for grid in (before, after):
            im = Image.new('RGB', (nq * scale, int((nr + .5) * scale)), (30, 30, 40)); dr_ = ImageDraw.Draw(im)
            for q in range(q_lo, q_hi):
                for r in range(r_lo, r_hi):
                    x = (q - q_lo) * scale; y = int(((r_hi - 1 - r) + .5 * (1 - q % 2)) * scale)   # north up
                    if sea[r, q]:
                        col = (40, 70, 130)
                    else:
                        gi = int(grid[r, q]); col = COLOURS.get(G[gi], (255, 0, 255)) if 0 <= gi < len(G) else (0, 0, 0)
                        dz = (e[min(H - 1, r + 1), q] - e[max(0, r - 1), q]) - (e[r, min(W - 1, q + 1)] - e[r, max(0, q - 1)])
                        k = float(np.clip(1 + dz / 1500., .6, 1.35))
                        col = tuple(int(min(255, c * k)) for c in col)
                        if imp[r, q]:
                            col = tuple(int(c * .45) for c in col)
                    dr_.rectangle([x, y, x + scale - 1, y + scale - 1], fill=col)
                    if road[r, q] and not sea[r, q]:
                        dr_.point([(x + scale // 2, y + scale // 2)], fill=(90, 50, 20))
                    if slot[r, q]:
                        dr_.rectangle([x + 1, y + 1, x + scale - 2, y + scale - 2], outline=(200, 0, 0))
            panels.append(im)
        legend_h = 16 * ((len(COLOURS) + 3) // 4) + 22
        canvas = Image.new('RGB', (panels[0].width * 2 + 10, panels[0].height + legend_h), (255, 255, 255))
        canvas.paste(panels[0], (0, 20)); canvas.paste(panels[1], (panels[0].width + 10, 20))
        dr_ = ImageDraw.Draw(canvas)
        dr_.text((4, 4), f'{name} hex ({q0},{r0}) before', fill=(0, 0, 0)); dr_.text((panels[0].width + 14, 4), 'after (dark = impassable, red = slot, dots = road)', fill=(0, 0, 0))
        for i, (gn, col) in enumerate(COLOURS.items()):
            x = 4 + (i % 4) * 170; y = panels[0].height + 24 + (i // 4) * 16
            dr_.rectangle([x, y, x + 10, y + 10], fill=col); dr_.text((x + 14, y - 1), gn, fill=(0, 0, 0))
        path = out_dir / f'{name}.png'
        tmp = path.with_name(path.stem + f'.{os.getpid()}.tmp.png')
        canvas.save(tmp); os.replace(tmp, path)
        written.append(str(path))
    return written


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    """(q, r, src, dst) arrays of a hex record, the file closed afterwards (os.replace needs it closed)."""
    with np.load(path) as o:
        return tuple(np.array(o[k]) for k in ('q', 'r', 'src', 'dst'))


def _savez_atomic(path, **arrays):
    tmp = path.with_name(path.stem + f'.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    main()

"""Terrain realism audit: ground materials and logical ground against real geography.

  python audit_terrain_realism.py [PACK] [--out DIR]

Read-only. Samples the built pack's global_blend.dds at every tile centre (material
index in byte 0, cold band G in byte 1), the logical map (final-map/map.hex) and the
real-geography grid analysis/west-real-terrain.npz (RESOLVE biome, ETOPO metres,
Natural Earth lakes/rivers/playas, lon/lat; tile grid, south-up). Every rule is also
counted on the vanilla part of the map, so a rule only matters where the authored
land is clearly worse than vanilla's own rate.

Rules (tile level unless noted):
  desert_green      real desert, not oasis, material temperate/subtropical/cold/dark steppe
  desert_farm_wet   real desert, not oasis, farm or wetland material
  forest_sand       real forest/conifer/taiga/temperate/tropical, sand material
  steppe_sand       real steppe/grass outside known sand seas, sand material
  cold_tropic       real tundra/taiga/snow, subtropical material
(no tropical-vs-cold rule: slots 4-7 'cold_*' are vanilla's dark woodland ground, 65 %
of vanilla's own tropics, and G band 0 is not a cold signal there.)
  peak_green        above the snow line, temperate/subtropical/farm material
  lowland_cold      below 1200 m and south of 42 N, cold band (G = 0)
  ground_forest_desert / ground_desert_forest   (per hex) logical ground vs real biome

Oasis = within OASIS_WATER tiles of a real river/lake/playa or of a map water hex, or
within OASIS_TOWN tiles of a town plot. Writes ground-realism.json and
ground-realism.png (north up) into --out (default analysis/terrain-realism).

--profiles adds the per-landform regression metrics of the biome profile table (design
analysis/biome-profile-table-design.md section 4): every tile is classed with
biome_profiles.classify (hard argmax, weight > .5) on real canvas ground (outside the native map
beyond REAL_GROUND_FADE, real-terrain weight > .5, land) and, for reference, on the vanilla
ground of the same landform; per landform the family shares against AUDIT bounds, the banned
share, green specks (desert landforms, outside oases), material luminance, cold band, farm and
mud; then the fixed windows (Turpan, Taklamakan erg, Lop Nur, Hexi gobi, Qilian > 3000 m, East
Tianzhu, Funan/Mekong). Writes landform-metrics.json and, against --baseline (default
landform-metrics-baseline.json in --out, if present), prints every metric that moved beyond its
tolerance. --save-baseline writes the baseline instead. Not covered (polish internals): tonal
buffer, ecotone pairs, trees; the logical ground rules above stand in for 'logical ground'.

  python audit_terrain_realism.py [PACK] --profiles [--save-baseline]
"""
import argparse, json, struct, sys
from pathlib import Path

import numpy as np
from scipy import ndimage

import campaign_world as cw, map_hex
from analyze_reference import inventory
from ground_materials import sand_mask, SNOWLINE

OASIS_WATER, OASIS_TOWN = 12, 10        # tiles (2 tiles per hex)
SAND, COLD, FARM, ROAD, MUD = range(0, 4), range(4, 8), range(8, 13), 13, 14
STEPPE_LIGHT, STEPPE_DARK = (19, 20), (21, 22)
SUBTROP, TEMPERATE, WET = range(23, 27), range(27, 31), 31


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument('pack', nargs='?', type=Path, default=root / 'output/tw3k_regions_and_battles_terrain_repair_candidate.pack')
    ap.add_argument('--out', type=Path, default=root / 'analysis/terrain-realism')
    ap.add_argument('--profiles', action='store_true', help='per-landform regression metrics (biome profile table)')
    ap.add_argument('--baseline', type=Path, help='landform metrics to compare with (default: <out>/landform-metrics-baseline.json)')
    ap.add_argument('--save-baseline', action='store_true', help='write the landform metrics as the baseline')
    ap.add_argument('--gate', action='store_true', help='with --profiles: exit 3 when a desert landform keeps more banned '
                    'slots than GATE_BANNED (biome step 6: green specks on desert must stay gone)')
    ap.add_argument('--blend', type=Path, help='read this global_blend.dds instead of the pack entry (a scratch-workspace stage; never a live stage while a pipeline runs)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- game materials at tile centres
    key = 'terrain/campaigns/3k_dlc07_main_map/global_map/global_blend.dds'
    if args.blend:
        with args.blend.open('rb') as f:
            head = f.read(128)
        bh, bw = struct.unpack_from('<2I', head, 12)
        mm = np.fromfile(args.blend, np.uint8, bh * bw * 2, offset=128).reshape(bh, bw, 2)   # a copy, no file mapping
    else:
        e = next(x for x in inventory(args.pack, False)['entries'] if x['path'] == key)
        # read, never memory-map: packaging rewrites the candidate pack, and a mapped file cannot be replaced
        with args.pack.open('rb') as f:
            f.seek(e['offset']); head = f.read(128)
            bh, bw = struct.unpack_from('<2I', head, 12)
            mm = np.frombuffer(f.read(bh * bw * 2), np.uint8).reshape(bh, bw, 2)
    TH, TW = cw.TILE_H, cw.TILE_W
    per = bw // TW
    rows = (bh - 1) - (np.arange(TH) * per + per // 2)
    cols = np.arange(TW) * per + per // 2
    blk = np.asarray(mm[rows][:, cols])
    mat, band = blk[..., 0], blk[..., 1]
    del mm

    # ---- real geography
    z = np.load(cw.WORKSPACE / 'analysis/west-real-terrain.npz')
    names = list(z['biome_names'])
    bi = {n: i for i, n in enumerate(names)}
    biome, elev, lon, lat = z['biome'], z['elevation'], z['lon'], z['lat']
    water = z['lake'] | z['river'] | z['salt']
    assert biome.shape == (TH, TW), (biome.shape, TH, TW)

    # ---- logical map
    F = map_hex.read(cw.WORKSPACE / 'work/final-map/map.hex')
    cells = F['cells']; H, W = cells.shape[:2]
    rnames = F['land_regions'] + F['sea_regions']
    q, r = tile_hexes(TH, TW, H, W)
    hland = (cells[:, :, 0] & 3) != 1
    land = hland[r, q]
    town = ((cells[:, :, 2] >> 4) == 1)[r, q]
    native_hex = np.zeros((H, W), bool); native_hex[cw.NATIVE_HEX_SLICE] = True
    native = native_hex[r, q]
    owner = map_hex.region_ids(cells)

    oasis = oasis_tiles(water, land, town)

    def isin(a, s):
        return np.isin(a, list(s))
    desert = biome == bi['desert']
    forestish = isin(biome, [bi[n] for n in ('forest', 'conifer', 'taiga', 'temperate', 'tropical')])
    steppeish = isin(biome, [bi['steppe'], bi['grass']])
    coldbio = isin(biome, [bi[n] for n in ('tundra', 'taiga', 'snow')])
    snowline = np.interp(lat, SNOWLINE[0], SNOWLINE[1])
    green = isin(mat, list(TEMPERATE) + list(SUBTROP) + list(COLD) + list(STEPPE_DARK))
    sandmask = sand_mask(lon.astype(np.float64), lat.astype(np.float64))

    rules = {
        'desert_green': desert & ~oasis & green,
        'desert_farm_wet': desert & ~oasis & isin(mat, list(FARM) + [WET]),
        'forest_sand': forestish & isin(mat, SAND),
        'steppe_sand': steppeish & ~sandmask & isin(mat, SAND),
        'cold_tropic': coldbio & isin(mat, SUBTROP),
        'peak_green': (elev > snowline) & isin(mat, list(TEMPERATE) + list(SUBTROP) + list(FARM)),
        'lowland_cold': (elev < 1200) & (lat < 42) & (band == 0),
    }
    report = {'pack': str(args.blend or args.pack), 'tile_grid': [TW, TH], 'oasis_tiles': {'water': OASIS_WATER, 'town': OASIS_TOWN}, 'rules': {}}
    flagged_any = np.zeros((TH, TW), np.uint8)
    for k, (name, m) in enumerate(rules.items(), 1):
        m = m & land
        base = {'native': None, 'new': None}
        for side, sel in (('native', native), ('new', ~native)):
            # rate among tiles where the rule's condition could apply (its biome/elevation side)
            cond = {'desert_green': desert & ~oasis, 'desert_farm_wet': desert & ~oasis, 'forest_sand': forestish,
                    'steppe_sand': steppeish & ~sandmask, 'cold_tropic': coldbio,
                    'peak_green': elev > snowline, 'lowland_cold': (elev < 1200) & (lat < 42)}[name] & land & sel
            n = int(cond.sum()); hit = int((m & sel).sum())
            base[side] = {'eligible_tiles': n, 'flagged_tiles': hit, 'rate': round(hit / n, 4) if n else None}
        # clusters on the new side, hex level
        hexmask = np.zeros((H, W), bool)
        np.logical_or.at(hexmask, (r[m & ~native], q[m & ~native]), True)
        lab, nlab = ndimage.label(hexmask, structure=np.ones((3, 3)))
        clusters = []
        if nlab:
            sizes = np.bincount(lab.ravel())[1:]
            for i in np.argsort(-sizes)[:15]:
                rr, qq = np.nonzero(lab == i + 1)
                cr, cq = int(np.median(rr)), int(np.median(qq))
                t = (min(TH - 1, cr * 2 + cq % 2), min(TW - 1, cq * 2))
                regs = np.bincount(owner[rr, qq].clip(0)).argmax()
                mats = np.bincount(mat[np.isin(r * W + q, rr * W + qq) & m].ravel(), minlength=32) if sizes[i] < 4000 else None
                clusters.append({'hexes': int(sizes[i]), 'centre_hex': [cq, cr], 'lonlat': [round(float(lon[t]), 2), round(float(lat[t]), 2)],
                                 'region': rnames[regs] if regs < len(rnames) else None,
                                 'top_materials': [int(x) for x in np.argsort(-mats)[:3]] if mats is not None else None})
        report['rules'][name] = {'native': base['native'], 'new': base['new'], 'new_flagged_hexes': int(hexmask.sum()), 'clusters': clusters}
        flagged_any[m & ~native] = k

    # logical ground vs biome, per hex (biome at the hex's centre tile)
    gt = F['land_ground_types']
    gidx = (cells[:, :, 5].astype(int) >> 4) + ((cells[:, :, 6].astype(int) & 7) << 4) - 1
    hq, hr = np.meshgrid(np.arange(W), np.arange(H))
    hb = biome[np.minimum(TH - 1, hr * 2 + hq % 2), np.minimum(TW - 1, hq * 2)]
    gname = np.array(gt + ['?'] * 3, dtype=object)[np.clip(gidx, 0, len(gt) + 2)]
    is_forest_g = np.vectorize(lambda s: 'forest' in s)(gname)
    is_desert_g = np.vectorize(lambda s: s in ('desert',))(gname)
    for name, m in (('ground_forest_desert', hland & is_forest_g & (hb == bi['desert'])),
                    ('ground_desert_forest', hland & is_desert_g & np.isin(hb, [bi[n] for n in ('forest', 'conifer', 'taiga', 'temperate', 'tropical')]))):
        report['rules'][name] = {'native_hexes': int((m & native_hex).sum()), 'new_hexes': int((m & ~native_hex).sum())}

    report['legend'] = {i + 1: k for i, k in enumerate(rules)}
    (args.out / 'ground-realism.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    # overview image (north up), half tile resolution
    from PIL import Image
    colours = np.array([[40, 40, 40], [255, 60, 60], [255, 160, 0], [230, 220, 40], [200, 120, 255],
                        [60, 200, 255], [255, 255, 255], [80, 255, 120], [120, 120, 255]], np.uint8)
    img = np.where(land[..., None], np.uint8(90), np.uint8(25)).repeat(3, 2).astype(np.uint8)
    img[native & land] = (70, 70, 80)
    img[flagged_any > 0] = colours[flagged_any[flagged_any > 0]]
    Image.fromarray(img[::-1][::2, ::2]).save(args.out / 'ground-realism.png')
    for name, v in report['rules'].items():
        if 'new' in v and v['new']:
            print(f"{name:22s} native {v['native']['rate']} | new {v['new']['rate']} ({v['new']['flagged_tiles']} tiles, {v['new_flagged_hexes']} hexes)")
        else:
            print(f"{name:22s} native {v.get('native_hexes')} hexes | new {v.get('new_hexes')} hexes")
    print('report:', args.out / 'ground-realism.json')
    if args.profiles:
        rep = profiles(args, mat, band, land, native, oasis, z)
        if args.gate:
            # [STEP 2] the gate counts what the generator enforced, on the generator's own landform field
            # (analysis/landform-tiles.npz: dominant landform, ban reach, per-slot whitelist); the classify-based
            # 'banned_open' below is the fallback for a workspace without the field.
            import landform_tiles
            lt = landform_tiles.load(cw.WORKSPACE)
            if lt is not None:
                res = landform_tiles.banned_share(mat, lt, sel=land & ~native)
                rep['gate'] = {'source': str(cw.WORKSPACE / landform_tiles.NAME), 'limit': GATE_BANNED, 'landforms': res}
                (args.out / 'landform-metrics.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1) + chr(10), encoding='utf-8')
                bad = {k: v['share'] for k, v in res.items() if v['share'] > GATE_BANNED}
                if bad:
                    print(f'GATE FAIL: banned share above {GATE_BANNED} on enforced landforms (landform-tiles): {bad}')
                    sys.exit(3)
                print(f'GATE PASS: enforced landforms banned share <= {GATE_BANNED} (landform-tiles): '
                      + str({k: v['share'] for k, v in res.items()}))
            else:
                print(f'WARNING: no {landform_tiles.NAME} in the workspace: gate on the classify-based banned_open')
                bad = {k: v['new'].get('banned_open') for k, v in rep['landforms'].items()
                       if k in GATE_LANDFORMS and (v['new'].get('banned_open') or 0) > GATE_BANNED}
                if bad:
                    print(f'GATE FAIL: banned share away from oases above {GATE_BANNED} on desert floor landforms: {bad}')
                    sys.exit(3)
                print(f'GATE PASS: desert floor landforms banned share away from oases <= {GATE_BANNED}: '
                      + str({k: rep['landforms'][k]['new'].get('banned_open') for k in GATE_LANDFORMS}))


# ------------------------------------------------------------------ per-landform metrics (--profiles)
# Mean colour of each blend slot (harvest textures, mip 4; campaign_ground_textures.load_layers): luminance only.
SLOT_RGB = ((208, 178, 119), (196, 166, 104), (182, 158, 105), (187, 163, 106), (82, 80, 25), (61, 70, 32), (89, 100, 33),
            (121, 116, 37), (220, 183, 75), (170, 156, 57), (170, 171, 91), (152, 155, 62), (111, 129, 36), (165, 148, 105),
            (135, 114, 57), (85, 147, 125), (99, 120, 116), (72, 129, 107), (86, 120, 111), (189, 161, 97), (176, 148, 81),
            (156, 135, 60), (134, 126, 45), (170, 157, 62), (141, 139, 41), (110, 113, 38), (109, 105, 37), (177, 156, 79),
            (155, 148, 53), (141, 140, 40), (123, 124, 35), (124, 121, 54))
REAL_GROUND_FULL = 72          # tile cells from the native map: polish REAL_GROUND_FADE[1] (full real-ground weight)
DESERT_LANDFORMS = ('gobi', 'erg', 'drymtn', 'salt_crust', 'yardang')
GATE_BANNED = .002             # --gate: banned share allowed on an enforced landform (fixed by the user: never widen; r16 before step 6: gobi 3.1 %)
# --gate counts the desert floor only, away from the step-6 whitelist (the oasis mask here: water within
# OASIS_WATER tiles, towns within OASIS_TOWN; polish's oasis belts and river belts keep their greens). drymtn is
# reported, not gated: polish classifies its ranges on the ecotone-edited biome (treeline forests of the Zagros,
# Hindu Kush and Altai), this audit on the raw RESOLVE biome, so their woodland is not a speck signal.
GATE_LANDFORMS = ('gobi', 'erg', 'salt_crust', 'yardang')
FARM_WATER_BINS = (0, 1, 2, 3, 8, np.inf)   # hexes to water: vanilla 0.01 / 0.14 / 0.5 / 1.3-1.7 / 2-2.6 % farm
FARM_WINDOW = 120              # tile cells (60 hexes)
# lon0, lat0, lon1, lat1 boxes of the design's fixed windows
WINDOWS = {'turpan': (88.4, 42.55, 90.4, 43.1), 'lop_nur': (89.8, 39.8, 91.2, 41.0), 'hexi_gobi': (93.0, 39.3, 100.0, 41.5),
           'qilian_high': (97.0, 37.0, 103.0, 39.5), 'east_tianzhu': (87.0, 21.5, 92.5, 27.0), 'funan_mekong': (104.0, 8.5, 107.5, 12.5)}
# tolerance before a move is reported
TOL = {'tiles_rel': .02, 'share': .02, 'banned': .002, 'specks_rel': .05, 'lum': 6., 'cold': .02, 'farm': .005, 'mud': .005,
       'window': .01}


def _shares(mat, sel):
    n = int(sel.sum())
    h = np.bincount(mat[sel].ravel(), minlength=32)[:32] / max(1, n)
    return n, h


def _family(h):
    import biome_profiles as bp
    return {f: round(float(h[list(s)].sum()), 4) for f, s in bp.FAMILIES.items()}


def _lum(mat, sel):
    rgb = np.asarray(SLOT_RGB, np.float32)
    L = (rgb @ np.array([.299, .587, .114], np.float32))[mat[sel]]
    return [round(float(v), 1) for v in (L.mean(), np.percentile(L, 10), np.percentile(L, 90))] if L.size else None


def _specks(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return {'count': 0, 'area_median': 0}
    sz = np.sort(np.bincount(lab.ravel())[1:]); c = np.cumsum(sz)
    return {'count': int(n), 'area_median': int(sz[np.searchsorted(c, c[-1] / 2)])}


def _farm_windows(farm, land_sel):
    TH, TW = farm.shape; k = FARM_WINDOW; h, w = TH // k, TW // k
    f = farm[:h * k, :w * k].reshape(h, k, w, k).sum((1, 3)); l = land_sel[:h * k, :w * k].reshape(h, k, w, k).sum((1, 3))
    ok = l > k * k / 2
    v = f[ok] / l[ok]
    return {'median': round(float(np.median(v)), 4), 'p90': round(float(np.percentile(v, 90)), 4), 'windows': int(v.size)} if v.size else None


def landform_block(mat, band, sel, name, water_hex, oasis):
    import biome_profiles as bp
    n, h = _shares(mat, sel)
    if not n:
        return {'tiles': 0}
    fam = _family(h); aud = bp.AUDIT.get(name, {})
    out = {'tiles': n, 'families': fam, 'luminance': _lum(mat, sel), 'cold_band': round(float((band[sel] == 0).mean()), 4),
           'farm': fam['farm'], 'mud': fam['mud']}
    if aud.get('banned'):
        out['banned'] = round(float(h[list(aud['banned'])].sum()), 4)
        so = sel & ~oasis
        out['banned_open'] = round(float(np.isin(mat[so], list(aud['banned'])).mean()), 4) if so.any() else None
    if aud.get('bounds'):
        out['outside_bounds'] = {f: fam[f] for f, (lo, hi) in aud['bounds'].items() if not lo <= fam[f] <= hi}
    if name in DESERT_LANDFORMS:
        out['green_specks'] = _specks(sel & ~oasis & np.isin(mat, bp.FAMILIES['green']))
    mud = sel & (mat == 14)
    if mud.any():
        out['mud_blob_median'] = int(np.median(np.bincount(ndimage.label(mud)[0].ravel())[1:]))
    b = np.digitize(water_hex[sel], FARM_WATER_BINS[1:-1])
    farm = np.isin(mat[sel], list(range(8, 13)))
    out['farm_by_water_hex'] = {f'{FARM_WATER_BINS[i]}-{FARM_WATER_BINS[i + 1]}': round(float(farm[b == i].mean()), 4) if (b == i).any() else None
                                for i in range(len(FARM_WATER_BINS) - 1)}
    return out


def tile_hexes(TH, TW, H, W):
    """(q, r) hex of every tile cell as this audit rounds it (the land / native masks of the rules and --gate)."""
    ty, tx = np.mgrid[:TH, :TW]
    q = np.clip(np.rint(tx / 2).astype(int), 0, W - 1)
    r = np.clip(np.rint(ty / 2 - (q % 2) * .5).astype(int), 0, H - 1)
    return q, r


def real_landforms(land, z):
    """(landform names, hard class top, its weight topw, real canvas ground) per tile: biome_profiles.classify
    argmax, a class counts where topw > .5; real ground = land, >= REAL_GROUND_FULL tiles from the native map,
    real-terrain weight > .5. The selection of --profiles and --gate (also stage_invariants.desert_banned_share)."""
    import biome_profiles as bp
    TH, TW = land.shape
    bnames = [str(n) for n in z['biome_names']]
    y, x = np.mgrid[:TH, :TW].astype(np.float32)
    origin = tuple(cw.load_authored_coordinates()['noise_origin'])
    w, _ = bp.classify(z['elevation'].astype(np.float32), z['biome'], bnames, z['lon'].astype(np.float64),
                       z['lat'].astype(np.float64), x / 2, y / 2, origin)
    names = list(w)
    W = np.stack([w[n] for n in names]); top = W.argmax(0); topw = W.max(0); del W, w
    dn = np.hypot(np.maximum(np.maximum(cw.NATIVE_TILE_X0 - x, x - cw.NATIVE_TILE_X1 + 1), 0),
                  np.maximum(np.maximum(cw.NATIVE_TILE_Y0 - y, y - cw.NATIVE_TILE_Y1 + 1), 0))
    real = land & (dn >= REAL_GROUND_FULL) & (z['weight'] > .5)
    return names, top, topw, real


def oasis_tiles(water, land, town):
    """The oasis mask of the rules, 'banned_open' and --gate (this audit's stand-in for the step-6 whitelist):
    real water or the sea within OASIS_WATER tiles, towns within OASIS_TOWN."""
    return ((ndimage.distance_transform_edt(~(water | ~land)) <= OASIS_WATER)
            | (ndimage.distance_transform_edt(~town) <= OASIS_TOWN))


def desert_banned_shares(mat, names, top, topw, real, oasis):
    """--gate metric: per gated desert floor landform (GATE_LANDFORMS), the share of its real-ground tiles away
    from the oasis mask holding a slot banned for it (biome_profiles.AUDIT 'banned'): landform_block's 'banned_open'."""
    import biome_profiles as bp
    out = {}
    for i, name in enumerate(names):
        aud = bp.AUDIT.get(name, {})
        if name in GATE_LANDFORMS and aud.get('banned'):
            so = (top == i) & (topw > .5) & real & ~oasis
            out[name] = round(float(np.isin(mat[so], list(aud['banned'])).mean()), 4) if so.any() else None
    return out


def profiles(args, mat, band, land, native, oasis, z):
    import biome_profiles as bp, ground_materials as gm
    TH, TW = mat.shape
    biome, elev = z['biome'], z['elevation'].astype(np.float32)
    lon, lat = z['lon'].astype(np.float64), z['lat'].astype(np.float64)
    names, top, topw, real = real_landforms(land, z)
    van = land & native
    water = z['lake'] | z['river'] | z['salt'] | ~land
    water_hex = ndimage.distance_transform_edt(~water) / 2
    farm_all = np.isin(mat, list(range(8, 13)))
    rep = {'pack': str(args.blend or args.pack), 'real_ground': 'land, >= %d tiles from the native map, real weight > .5' % REAL_GROUND_FULL,
           'landforms': {}, 'windows': {}}
    for i, name in enumerate(names):
        cls = (top == i) & (topw > .5)
        rep['landforms'][name] = {'new': landform_block(mat, band, cls & real, name, water_hex, oasis),
                                  'vanilla': landform_block(mat, band, cls & van, name, water_hex, oasis)}
    rep['new_farm_windows'] = _farm_windows(farm_all & real, real)
    rep['vanilla_farm_windows'] = _farm_windows(farm_all & van, van)

    def box(b):
        return land & (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
    _, h = _shares(mat, box(WINDOWS['turpan']))
    rep['windows']['turpan'] = {'green': _family(h)['green']}
    erg = land & ~native & gm.sand_mask(lon, lat); _, h = _shares(mat, erg)
    rep['windows']['taklamakan_erg'] = {'arid_1': round(float(h[0]), 4), 'ref': .9}
    _, h = _shares(mat, box(WINDOWS['lop_nur'])); f = _family(h)
    rep['windows']['lop_nur'] = {'farm': f['farm'], 'wetland': f['wetland'], 'ref': 0}
    g = box(WINDOWS['hexi_gobi']) & (top == names.index('gobi')) & (topw > .5); _, h = _shares(mat, g)
    rep['windows']['hexi_gobi'] = {f'arid_{j + 1}': round(float(h[j]), 4) for j in range(4)}
    rep['windows']['hexi_gobi']['ref'] = {'arid_1': .13, 'arid_2': .47, 'arid_3': .16, 'arid_4': .19}
    _, h = _shares(mat, box(WINDOWS['qilian_high']) & (elev > 3000))
    rep['windows']['qilian_high'] = {'cold_1': round(float(h[4]), 4), 'ref': .76}
    _, h = _shares(mat, box(WINDOWS['east_tianzhu']))
    rep['windows']['east_tianzhu'] = {'slot_4': round(float(h[4]), 4), 'before_split': .40}
    fm = box(WINDOWS['funan_mekong'])
    rep['windows']['funan_mekong'] = {'farm': _farm_windows(farm_all & fm, fm), 'target_median': .025}

    out = args.out / 'landform-metrics.json'
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('landform metrics:', out)
    for name, v in rep['landforms'].items():
        n, o = v['new'], v['vanilla']
        if n.get('tiles'):
            print(f"  {name:8s} new {n['tiles']:8d} lum {n['luminance'][0]:5.1f} green {n['families']['green']:.3f} mud {n['mud']:.3f}"
                  f" farm {n['farm']:.3f} banned {n.get('banned', '-')} outside {n.get('outside_bounds', {})}"
                  + (f" specks {n['green_specks']['count']}" if 'green_specks' in n else '')
                  + (f" | vanilla {o['tiles']} lum {o['luminance'][0]:.1f} green {o['families']['green']:.3f}" if o.get('tiles') else ''))
    for k, v in rep['windows'].items():
        print(f'  window {k}: {v}')
    base = args.baseline or args.out / 'landform-metrics-baseline.json'
    if args.save_baseline:
        base.write_text(out.read_text(encoding='utf-8'), encoding='utf-8'); print('baseline written:', base)
    elif base.exists():
        moved = compare(json.loads(base.read_text(encoding='utf-8')), rep)
        print(f'against {base}: {len(moved)} moved metric(s)')
        for m in moved:
            print('  ', m)
    return rep


def _num_moves(a, b, path=''):
    """Numeric leaves of two window records whose difference exceeds TOL['window']."""
    if isinstance(b, dict):
        return [m for k in b for m in _num_moves((a or {}).get(k) if isinstance(a, dict) else None, b[k], f'{path}.{k}')]
    if isinstance(b, (int, float)) and isinstance(a, (int, float)) and abs(b - a) > TOL['window'] and not path.endswith('windows'):
        return [f'{path} {a} -> {b}']
    return []


def compare(old, new):
    """Metrics that moved beyond TOL between two landform-metrics reports."""
    moved = []
    for name, sides in new['landforms'].items():
        a, b = old['landforms'].get(name, {}).get('new', {}), sides['new']
        if not a.get('tiles') or not b.get('tiles'):
            if bool(a.get('tiles')) != bool(b.get('tiles')):
                moved.append(f'{name}: tiles {a.get("tiles")} -> {b.get("tiles")}')
            continue
        if abs(b['tiles'] - a['tiles']) > TOL['tiles_rel'] * a['tiles']:
            moved.append(f"{name}: tiles {a['tiles']} -> {b['tiles']}")
        for f, v in b['families'].items():
            if abs(v - a['families'][f]) > TOL['share']:
                moved.append(f"{name}: {f} {a['families'][f]} -> {v}")
        if b.get('banned', 0) - a.get('banned', 0) > TOL['banned']:
            moved.append(f"{name}: banned {a.get('banned')} -> {b.get('banned')}")
        if 'green_specks' in b and b['green_specks']['count'] > a['green_specks']['count'] * (1 + TOL['specks_rel']):
            moved.append(f"{name}: green specks {a['green_specks']} -> {b['green_specks']}")
        if abs(b['luminance'][0] - a['luminance'][0]) > TOL['lum']:
            moved.append(f"{name}: luminance {a['luminance'][0]} -> {b['luminance'][0]}")
        if abs(b['cold_band'] - a['cold_band']) > TOL['cold']:
            moved.append(f"{name}: cold band {a['cold_band']} -> {b['cold_band']}")
        for f in ('farm', 'mud'):
            if abs(b[f] - a[f]) > TOL[f]:
                moved.append(f'{name}: {f} {a[f]} -> {b[f]}')
    for wname, v in new['windows'].items():
        moved += [f'window {wname}{m}' for m in _num_moves(old['windows'].get(wname), v)]
    return moved


if __name__ == '__main__':
    main()

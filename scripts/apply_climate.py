"""Gameplay climate and wetlands for canvas land outside the native map (final-map/map.hex).

Run with a Python that has scipy (the tw3k env: env/python-path.txt). Dry run by default;
--apply backs up work/final-map/map.hex (map.hex.before-climate-<stamp>) and writes it.
Writes analysis/climate-wetland.json (report) and analysis/climate-wetland-hex.npz.

Cells outside the native slice copied their byte 7 from template cells (89% temperate).
Here every land hex outside the native slice (new tw3k_ regions and non-playable canvas
filler) gets a climate from real geography, mapped to the four game climates the way
vanilla does it:

  * per hex: majority RESOLVE biome, mean ETOPO metres, lat/lon from
    analysis/west-real-terrain.npz (build_west_real_terrain.py, tile grid, whole canvas);
  * the vanilla mapping is learned at run time from the vanilla map (cw.VANILLA_MAP)
    against the same fields at the same places: a table (biome, latitude band,
    elevation band) -> majority climate, used where vanilla has >= LEARN_MIN hexes and a
    majority >= LEARN_SHARE; elsewhere (north of vanilla, Tibet, taiga, tundra) rules
    that extrapolate vanilla's (see classify());
  * within NATIVE_REACH hexes of the native rectangle the native climate is carried
    outward along a meandered front, as polish (native_campaign_climate) carries it,
    so the climate never cuts on the native border;
  * Hehuang patch hexes take the patch's tile climate (hehuang_terrain.Patch.climate);
  * a majority filter removes single-hex speckle.
The revealed filler (revealed_filler.revealed_hex: vanilla non-playable land inside the native slice that is
now playable) is canvas here too: real-geography climate and wetlands, the native carry only on and next to
vanilla playable land (REVEALED_CARRY). Vanilla's own playable hexes keep their bytes; the learned table and
the carried native climate sample vanilla playable land only (revealed_filler.vanilla_playable_hex).

Map.hex climate byte: byte7 = (climate index + 1) << 1, index into m['climates'].

Wetlands: vanilla marks its marshes (Red River delta, Poyang, Dongting, Tai Hu, Yellow
River mouth) with ground 'wetland' (15.9 per 1000 native land hexes; 2.4% impassable,
never a settlement slot, median 8 m) and paints blend material 31 'wetlands' on them
(56% of a wetland hex on average, 84% of material 31 lies on wetland hexes). Here
wetland ground goes on flat low land hexes outside the native slice inside RESOLVE
wetland ecoregions (mangroves, swamp forests, flooded grasslands; the broad Amur and
Nenjiang flooded grasslands only next to water) and a few named marshes RESOLVE has no
polygon for (WETLANDS_EXTRA). Town hexes and their neighbours keep their ground; the
impassable bit is never touched. paint_wetland_blend() is the polish side (material 31
on those hexes), called from polish_campaign_terrain.py in the CLIMATE PATCH block.

Terrain attrition: vanilla map.hex has no per-hex attrition (bytes 10-14, byte 7 bit 0
and byte 0 bit 2 are zero on every vanilla hex; the 'attritions' list is names only),
and the terrain_land attritions (desert, mountain, snow, swamp, wasteland) do 0.005%
damage in campaign_map_attrition_damages. Nothing is written for attrition.

  <tw3k python> apply_climate.py            # dry run, report only
  <tw3k python> apply_climate.py --apply    # back up and write map.hex
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import campaign_world as cw
import map_hex
from hex_components import hex_dilate, hex_label
from revealed_filler import filler_hex, revealed_hex, revealed_weight, vanilla_playable_hex
from terrain_noise import fbm
import biome_profiles as _bp

_K = _bp.consts('apply_climate')   # values: biome_profiles.CONSTS['apply_climate']

A = cw.WORKSPACE
MAP = A / 'work/final-map/map.hex'
REAL = A / 'analysis/west-real-terrain.npz'
ECO = cw._ROOT / 'work/geodata/eco-grid-v2.npz'
ECO_BOX, ECO_STEP = (40., -12., 152., 78.), .02            # frontier_terrain.ECO_BOX / ECO_STEP
REPORT = A / 'analysis/climate-wetland.json'
HEX_OUT = A / 'analysis/climate-wetland-hex.npz'

# climate_map.cm / tile climate order (polish) and global blend G band, by map.hex climate name.
TILE_CLIMATE = _K['TILE_CLIMATE']
BLEND_BAND = _K['BLEND_BAND']   # measured on vanilla: 90-99% per climate

LAT_BINS = _K['LAT_BINS']
ELEV_BINS = _K['ELEV_BINS']
LEARN_MIN, LEARN_SHARE = _K['LEARN_MIN'], _K['LEARN_SHARE']
COLD_ELEV = _bp.cold_line('COLD_ELEV')        # m: vanilla land at or above it is 86% cold
NORTH_COLD = _bp.cold_line('NORTH_COLD')         # deg: north of vanilla's land (42.6 N); continental Dfb/Dfc and taiga
# classify() rule groups (biome_profiles.BIOMES flags gc_*)
GC_TROPICAL = _bp.biome_set('gc_tropical')   # tropical, tropical_dry, savanna, paddy, mangrove
GC_SUB31 = _bp.biome_set('gc_sub31')         # grass, other, temperate, wetland, forest: subtropical south of 31 N
GC_ARID = _bp.biome_set('gc_arid')           # desert, alpine
GC_COLD = _bp.biome_set('gc_cold')           # snow, tundra, taiga: always cold
NATIVE_REACH = (24, 30, 10)   # hexes from the native rectangle where the native climate is carried
                               # (polish: 48 tile cells + a 72-cell fbm wave): mean, wave, minimum
NATIVE_INSET = 2        # hexes: the outermost native cells alternate; seed a little inside
# Revealed filler (revealed_filler.revealed_hex: vanilla non-playable land now playable) is canvas. Where its
# polish ground weight (revealed_weight, polish NONPLAY_RAMP) is below REVEALED_CARRY it takes the climate of the
# nearest vanilla playable hex (the native carry), elsewhere the real-geography climate, as polish ramps it.
REVEALED_RAMP = _bp.consts('polish')['NONPLAY_RAMP']
REVEALED_CARRY = .5
SMOOTH = (5, 2)         # majority filter window (hexes) and passes
# Latitude/longitude thresholds meander by up to this many degrees (fbm, 20-hex features):
# a plain latitude rule drew straight east-west climate lines across the steppe.
GEO_MEANDER = 1.5
# Western Regions (west of Dunhuang, north of the Tibetan plateau: the plan_western_regions
# projection): polish sets its tile climate from the real biome with its own table
# (polish_campaign_terrain _clim: desert/steppe/grass/alpine/tundra/snow/wetland arid,
# conifer/taiga cold, forest temperate, tropics subtropical); map.hex follows it there,
# except high ground >= COLD_ELEV further than WR_TOWN_KEEP hexes from a town (Tian Shan,
# Pamir and Kunlun crests: cold, as vanilla's high ground). Town hexes there take the
# polish climate: the hex-mean DEM at a levelled oasis town reads the mountain foot.
WR_LON_MAX, WR_LAT_MIN, WR_LAT_MAX = (_bp.region('western_regions')[k] for k in ('lon_max', 'lat_min', 'lat_max'))   # DUNHUANG_REAL lon; Kunlun to the Irtysh/Balkhash
WR_TOWN_KEEP = _K['WR_TOWN_KEEP']
POLISH_CLIMATE = _bp.biome_map('gameplay_polish', 'gameplay')
POLISH_OTHER = _bp.OTHER_DEFAULT['gameplay_polish']   # 'other' (no POLISH_CLIMATE entry): arid here, while polish keeps its band

# Wetland ecoregions (Ecoregions2017 ECO_ID): full = every flat low hex, near_water = only
# hexes within WATER_REACH of water (the broad meadow plains are mostly dry grassland).
WETLAND_ECO_FULL = _K['WETLAND_ECO_FULL']
WETLAND_ECO_NEAR_WATER = _K['WETLAND_ECO_NEAR_WATER']
NEAR_WATER_ECO_MAX_ELEV = _K['NEAR_WATER_ECO_MAX_ELEV']   # m: the Zeya-Bureya and Songnen plains above it are dry meadow steppe
# Named marshes without a RESOLVE polygon: (name, lon0, lon1, lat0, lat1, max metres, near water only).
WETLANDS_EXTRA = _bp.region('wetlands_extra')['boxes']   # 8 boxes (biome_profiles.REGIONS)
FULL_MAX_ELEV = _K['FULL_MAX_ELEV']      # m: swamp/mangrove ecoregion hexes above it are not marsh
NEAR_WATER_MAX_ELEV = _K['NEAR_WATER_MAX_ELEV']
WATER_REACH = _K['WATER_REACH']         # hexes
FLAT_STD = _K['FLAT_STD']           # m: elevation std over the 3x3 hex neighbourhood
WETLAND_GROUND_FROM = ('grassland', 'plains', 'light_forest', 'dense_forest', 'tundra', 'wetland')
MIN_WETLAND_COMPONENT = 3
TOWN_CLEAR = 1          # hexes around a settlement slot that keep their ground

# Blend (polish): material 31, soft edge and holes like vanilla's marsh paint.
WETLAND_MATERIAL = _K['WETLAND_MATERIAL']
BLEND_EDGE = 6          # LF pixels: ramp from the wetland hex edge inward
WET_SOFT = _K['WET_SOFT']   # ground contrast fix §4: (box radius px, floor) of the smooth wetland hex weight
WET_PIXEL_MASK = __import__('os').environ.get('TW3K_WET_PIXEL_MASK', '0') == '1'   # 0: the hard hex mask as before
BLEND_NOISE = (3., .6)  # hexes, amplitude of the patch noise
BLEND_CUT = _K['BLEND_CUT']       # mean 0.58 of a wetland hex painted (vanilla 0.56)
# --- FIX2 WETLAND PATCHES --- vanilla paints material 31 in large contiguous patches that hug
# the water (50% of its wetland hexes lie within 2 hexes of sea/lake/river, median 77% of such
# a hex painted, the inland ends of its marshes mostly unpainted; 87% of material 31 on wetland
# hexes, G band temperate/subtropical). The Mekong/Tonle Sap wetland is one 3,800-hex cluster
# (vanilla's largest: 382) whose hexes are 79% more than 1 hex from water: an even 58% fill
# drew a grey-blue speckled sheet (offline 2048bq Funan). The paint share now follows the hex
# distance to water (WET_WATER: hexes -> share term) and a coarse noise (WET_NOISE: grain hexes,
# octaves, amplitude, cut), so marsh lies in patches along rivers and shores, and the rest of the
# wetland keeps the donor ground (subtropical/temperate mixes) underneath.
WET_WATER = _K['WET_WATER']
WET_NOISE = _K['WET_NOISE']
# --- end FIX2 WETLAND PATCHES


def hex_grow(mask, steps):
    """Hexes within `steps` hex steps of mask (hex_dilate repeated; scipy's binary_dilation with its
    default cross structure grew 4 square-grid neighbours a step and missed 2 of the 6)."""
    out = np.asarray(mask, bool)
    for _ in range(steps):
        out = hex_dilate(out)
    return out


def _ndimage():
    from scipy import ndimage    # the map.hex side only; polish (paint_wetland_blend) runs without scipy
    return ndimage


def _box_blur(a, radius):
    """Separable box blur, edges clamped (native_campaign_climate._box_blur_reference)."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]; pad[axis] = (radius + 1, radius)
        c = np.cumsum(np.pad(out, pad, mode='edge'), axis=axis, dtype=np.float64)
        hi = np.take(c, np.arange(2 * radius + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * radius - 1), axis=axis)
        out = ((hi - lo) / (2 * radius + 1)).astype(np.float32)
    return out


def ground(cells):
    return (cells[..., 5].astype(np.int32) >> 4) + ((cells[..., 6].astype(np.int32) & 7) << 4) - 1


def set_ground(cells, mask, index):
    v = index + 1
    cells[mask, 5] = (cells[mask, 5] & 0x0F) | ((v & 15) << 4)
    cells[mask, 6] = (cells[mask, 6] & 0xF8) | ((v >> 4) & 7)


def climate_of(cells):
    return (cells[..., 7].astype(np.int32) >> 1) - 1


def set_climate(cells, mask, index):
    cells[mask, 7] = (cells[mask, 7] & 0xF1) | (((index + 1) & 7) << 1)


def hex_tiles(H, W):
    """Tile block (2x2 cells) of every hex: rows 2r+odd(q), +1; columns 2q, 2q+1."""
    q = np.arange(W)[None, :].repeat(H, 0); r = np.arange(H)[:, None].repeat(W, 1)
    ty = 2 * r + (q % 2); tx = 2 * q
    return q, r, ty, tx


def hex_geo(H, W):
    """Per hex: majority real biome, mean metres, lat, lon, any real river tile (canvas hex grid)."""
    z = np.load(REAL)
    x0, x1, y0, y1 = (int(v) for v in z['box'])
    names = [str(n) for n in z['biome_names']]
    _, _, ty, tx = hex_tiles(H, W)
    ty = ty - y0; tx = tx - x0
    ok = (ty >= 0) & (tx >= 0) & (ty + 1 < y1 - y0) & (tx + 1 < x1 - x0)
    ty = np.clip(ty, 0, y1 - y0 - 2); tx = np.clip(tx, 0, x1 - x0 - 2)
    bio, el, la, lo, rv = z['biome'], z['elevation'], z['lat'], z['lon'], z['river']
    four = [(ty, tx), (ty + 1, tx), (ty, tx + 1), (ty + 1, tx + 1)]
    B = np.stack([bio[a, b] for a, b in four], -1)
    best = np.zeros((H, W), np.int8); biome = np.zeros((H, W), np.uint8)
    for k in range(len(names)):
        n = (B == k).sum(-1); up = n > best; biome[up] = k; best[up] = n[up]
    elev = sum(el[a, b].astype(np.float32) for a, b in four) / 4
    lat = sum(la[a, b] for a, b in four) / 4; lon = sum(lo[a, b] for a, b in four) / 4
    river = np.zeros((H, W), bool)
    for a, b in four:
        river |= rv[a, b]
    ok &= np.isfinite(lat) & np.isfinite(lon)
    r_, q_ = np.mgrid[0:H, 0:W].astype(np.float32)
    lat_n = lat + GEO_MEANDER * fbm(q_, r_, (0, 0), 20, 3, seed=187)
    lon_n = lon + GEO_MEANDER * fbm(q_, r_, (0, 0), 20, 3, seed=188)
    return {'lat_n': lat_n, 'lon_n': lon_n, 'biome': biome, 'elev': elev, 'lat': lat, 'lon': lon, 'river': river, 'ok': ok, 'names': names}


def eco_ids(lon, lat):
    z = np.load(ECO)
    eco = z['eco']
    lon0, lat0, lon1, lat1 = ECO_BOX
    gx = np.clip(((np.nan_to_num(lon) - lon0) / ECO_STEP).astype(np.int64), 0, eco.shape[1] - 1)
    gy = np.clip(((lat1 - np.nan_to_num(lat)) / ECO_STEP).astype(np.int64), 0, eco.shape[0] - 1)
    return eco[gy, gx]


def _bin(v, edges):
    return np.searchsorted(np.asarray(edges, np.float64), np.asarray(v, np.float64), side='right')


# ---------------------------------------------------------------- learned vanilla mapping
def learn_vanilla(geo):
    """(biome, lat band, elevation band) -> vanilla climate, from the vanilla map at the same places."""
    v = map_hex.read(cw.VANILLA_MAP)
    vc = v['cells']; vh, vw = vc.shape[:2]
    names = v['climates']
    sl = cw.NATIVE_HEX_SLICE
    g = ground(vc); land = (g >= 0) & (g < len(v['land_ground_types']))
    cl = climate_of(vc)
    # vanilla's own playable land only (revealed_filler.vanilla_playable_hex): never the non-playable filler
    b = geo['biome'][sl]; e = geo['elev'][sl]; la = geo['lat'][sl]; ok = geo['ok'][sl] & land & vanilla_playable_hex()[sl]
    lb, eb = _bin(la, LAT_BINS), _bin(e, ELEV_BINS)
    key = (b.astype(np.int64) * 16 + lb) * 16 + eb
    table, rows = {}, []
    for k in np.unique(key[ok]):
        s = ok & (key == k)
        cnt = np.bincount(cl[s], minlength=len(names)); n = int(s.sum()); top = int(cnt.argmax())
        bi, rest = divmod(int(k), 256); li, ei = divmod(rest, 16)
        rows.append({'biome': geo['names'][bi], 'lat_band': li, 'elev_band': ei, 'hexes': n,
                     'climate': names[top], 'share': round(float(cnt[top] / n), 3), 'used': bool(n >= LEARN_MIN and cnt[top] / n >= LEARN_SHARE)})
        if rows[-1]['used']:
            table[int(k)] = top
    summary = {}
    for bi, bn in enumerate(geo['names']):
        s = ok & (b == bi)
        if s.sum() >= 50:
            cnt = np.bincount(cl[s], minlength=len(names))
            summary[bn] = {'hexes': int(s.sum()), **{names[i]: round(float(cnt[i] / s.sum()), 3) for i in range(len(names)) if cnt[i]}}
    high = ok & (e >= COLD_ELEV)
    summary['_elev_ge_%d' % COLD_ELEV] = {'hexes': int(high.sum()), 'cold': round(float((cl[high] == names.index('cold')).mean()), 3) if high.any() else None}
    return table, rows, summary


def classify(geo, table, names):
    """Gameplay climate index (into names) per hex from real geography; -1 where no geography. The rules
    (vanilla's own tendencies carried beyond its coverage, lat 17-43 N, < 4000 m; then the learned table;
    snow/tundra/taiga and >= COLD_ELEV always cold) are biome_profiles.classify_gameplay (RULES['gameplay'])."""
    return _bp.classify_gameplay(geo, table, names)


def majority(values, mask, classes, size, passes):
    ndimage = _ndimage()
    v = values.copy()
    for _ in range(passes):
        score = np.stack([ndimage.uniform_filter(((v == c) & mask).astype(np.float32), size) for c in classes])
        best = np.array(classes)[score.argmax(0)]
        v = np.where(mask, best, v)
    return v


def native_carry(cells, names, land):
    """Native climate carried outward from inside the native slice: (value, distance in hexes
    from the native rectangle, meandered reach). The carried value is that of the nearest vanilla
    playable land hex (revealed_filler.vanilla_playable_hex) a little inside the rectangle, never the
    non-playable filler (cold sheets under shroud, now partly revealed canvas) or its seas."""
    ndimage = _ndimage()
    H, W = land.shape
    sl = cw.NATIVE_HEX_SLICE
    seed = np.zeros((H, W), bool)
    seed[sl[0].start + NATIVE_INSET:sl[0].stop - NATIVE_INSET,
         sl[1].start + NATIVE_INSET:sl[1].stop - NATIVE_INSET] = True
    seed &= vanilla_playable_hex()
    _, (iy, ix) = ndimage.distance_transform_edt(~seed, return_indices=True)
    carried = climate_of(cells)[iy, ix]
    r, q = np.mgrid[0:H, 0:W].astype(np.float32)
    dq = np.maximum(np.maximum(sl[1].start - q, q - (sl[1].stop - 1)), 0)
    dr = np.maximum(np.maximum(sl[0].start - r, r - (sl[0].stop - 1)), 0)
    dn = np.hypot(dq, dr)
    reach = np.maximum(NATIVE_REACH[0] + NATIVE_REACH[1] * fbm(q, r, (0, 0), 35, 4, seed=193), NATIVE_REACH[2])
    return carried, dn, reach


def hehuang_climate(H, W, names, cells=None, rnames=None):
    """Hehuang patch tile climate per hex (majority of the 4 tiles inside the patch mask), or None. cells / rnames: the
    map being built (the patch caches are checked against it, not against an earlier final map.hex)."""
    import hehuang_terrain
    hh = hehuang_terrain.load(map_cells=cells, map_names=rnames)
    if hh is None:
        return None, None
    ty0, tx0 = cw.TILE_H, cw.TILE_W
    y, x = np.mgrid[:ty0, :tx0].astype(np.float32)
    q, r = x / 2, y / 2
    mask = hh.climate_mask(q, r)
    tile = hh.climate(np.zeros((ty0, tx0), np.uint16), q, r)
    del y, x, q, r
    inv = {v: names.index(k) for k, v in TILE_CLIMATE.items()}
    _, _, ty, tx = hex_tiles(H, W)
    ty = np.clip(ty, 0, ty0 - 2); tx = np.clip(tx, 0, tx0 - 2)
    four = [(ty, tx), (ty + 1, tx), (ty, tx + 1), (ty + 1, tx + 1)]
    n = sum(mask[a, b].astype(np.int8) for a, b in four)
    out = np.full((H, W), -1, np.int32)
    votes = np.zeros((4, H, W), np.int8)
    for a, b in four:
        t = tile[a, b]; m = mask[a, b]
        for c in range(4):
            votes[c] += (m & (t == c)).astype(np.int8)
    best = votes.argmax(0)
    lut = np.array([inv[c] for c in range(4)])
    sel = n >= 2
    out[sel] = lut[best[sel]]
    return out, {'hexes': int(sel.sum()), 'version': getattr(hh, 'version', None), 'sha': getattr(hh, 'sha', None)}


# ---------------------------------------------------------------- wetlands
def water_hex(cells, m, geo):
    """Water for the wetland rule's near-water test: river / lake / coast ground or a real river tile."""
    names_all = m['land_ground_types'] + m['sea_ground_types']
    water_g = [names_all.index(n) for n in ('river_navigable', 'river_navigable_shallows', 'river_non_navigable', 'sea_lake',
                                           'river', 'river_navigable_crossing', 'sea_coast') if n in names_all]
    return np.isin(ground(cells), water_g) | geo['river']


def wetland_candidates(cells, m, geo):
    """biome_profiles.wetland_candidate_hex on this map: (flat wetland-rule hexes, {source: mask})."""
    return _bp.wetland_candidate_hex(eco_ids(geo['lon'], geo['lat']), geo['elev'], geo['lon'], geo['lat'], water_hex(cells, m, geo))


_LGT = []


def _land_ground_types():
    """map.hex land ground type names (the vanilla template order; read once per process)."""
    if not _LGT:
        _LGT.extend(map_hex.read(cw.VANILLA_MAP)['land_ground_types'])
        assert _LGT.index('wetland') == 13
    return _LGT


def town_clear_hex(cells):
    """Settlement slots and TOWN_CLEAR hexes round them: they keep their ground (no wetland)."""
    slot = ((cells[..., 2] >> 4) > 0) | ((cells[..., 3] & 1) > 0)
    return hex_grow(slot, TOWN_CLEAR)


def wetland_hexes(cells, m, geo, target, isnew):
    H, W = target.shape
    G = m['land_ground_types']
    g = ground(cells)
    cand, sources = wetland_candidates(cells, m, geo)
    allowed = np.isin(g, [G.index(n) for n in WETLAND_GROUND_FROM])
    wet = cand & allowed & target & ~town_clear_hex(cells) & geo['ok']
    lab = hex_label(wet); n = int(lab.max())        # hex pieces (a 3x3 label joined 2 non-neighbours)
    if n:
        sz = np.bincount(lab.ravel()); wet &= sz[lab] >= MIN_WETLAND_COMPONENT
        wet &= lab > 0
    by_source = {nm: int((wet & s).sum()) for nm, s in sources.items() if (wet & s).any()}
    by_source_new = {nm: int((wet & s & isnew).sum()) for nm, s in sources.items() if (wet & s & isnew).any()}
    return wet, {'all': by_source, 'new_regions': by_source_new}


def paint_wetland_blend(cur, cells, protected=None, names=None, filler=True):
    """Polish side: blend material 31 on wetland hexes outside the native slice and on the filler
    (revealed_filler.filler_hex: vanilla non-playable filler that is land now, playable or not; visual only;
    names: the map's land_regions + sea_regions, required; filler False: outside the native slice only, as before the filler).

    cur: global_blend (oh, ow, 2) uint8, file rows north-up; channel 0 material, 1 climate band.
    cells: final map.hex cells (canvas grid, rows south-up). protected: tile-grid bool
    (rows south-up, polish _protected) never painted. Returns the painted pixel count.
    Vanilla look: the marsh fills the wetland hex with a soft edge and a few holes; the
    band (G) is the hex's gameplay climate (vanilla never pairs material 31 with arid)."""
    oh, ow = cur.shape[:2]
    H, W = cells.shape[:2]
    names_cl = ['arid', 'cold', 'default', 'subtropical', 'temperate']
    g = ground(cells)
    wet = g == 13                                  # land_ground_types index of 'wetland'
    G_WET_FROM = _land_ground_types() if WET_PIXEL_MASK else []
    assert names is not None, 'paint_wetland_blend needs the map region names (filler)'
    inside = np.zeros((H, W), bool); inside[cw.NATIVE_HEX_SLICE] = True
    wet &= (~inside | filler_hex(cells, names)) if filler else ~inside
    if not wet.any():
        return 0
    # [WET PIXEL MASK] the land hexes the soft edge may reach: the same zone rule, land, not river or wetland-free water
    spill = ((cells[..., 0] & 3) != 1) & ~np.isin(g, (10, 11))
    spill &= (~inside | filler_hex(cells, names)) if filler else ~inside
    # never into the town clearance (town spec) or onto ground that may not be marsh (hills, mountains: details review);
    # the reach past the hex edge is bounded by the blur radius (WET_SOFT)
    spill &= ~town_clear_hex(cells) & np.isin(g, [G_WET_FROM.index(n) for n in WETLAND_GROUND_FROM if n in G_WET_FROM])
    cl = climate_of(cells)
    band = np.full((H, W), BLEND_BAND['temperate'], np.uint8)
    for i, n in enumerate(names_cl):
        if n in BLEND_BAND and n != 'arid':
            band[cl == i] = BLEND_BAND[n]
    # --- FIX2 WETLAND PATCHES --- hex ring distance to water (sea/lake hexes, river ground 10/11)
    water = ((cells[..., 0] & 3) == 1) | np.isin(g, (10, 11))
    dwat = np.where(water, 0, 99).astype(np.int16); grown = water.copy()
    for step in range(1, WET_WATER[0][-1] + 1):
        gm = hex_dilate(grown)                     # the six hex neighbours (an 8-neighbour step reached 2 non-neighbours)
        dwat[gm & ~grown] = step; grown = gm
    wshare = np.interp(dwat, *WET_WATER).astype(np.float32); del dwat, grown, water
    # --- end FIX2 WETLAND PATCHES
    # numpy only (polish runs without scipy): blocks of BLOCK hexes holding wetland, each
    # computed on a margin so the result does not depend on the blocking.
    BLOCK, MARGIN = 64, 2
    painted = 0
    for r0 in range(0, H, BLOCK):
        for q0 in range(0, W, BLOCK):
            # The soft mask can spill from a wet hex in the neighbouring block. Inspect its halo too;
            # skipping on the core alone clipped that spill to the arbitrary 64-hex processing grid.
            support = wet[max(0, r0 - MARGIN):min(H, r0 + BLOCK + MARGIN),
                          max(0, q0 - MARGIN):min(W, q0 + BLOCK + MARGIN)] if WET_PIXEL_MASK else wet[r0:r0 + BLOCK, q0:q0 + BLOCK]
            if not support.any():
                continue
            # Pixel window (south-up rows): hex q covers pixels 8q..8q+7; rows 8r+4*odd(q)..+7.
            cx0, cx1 = 8 * q0, min(ow, 8 * (q0 + BLOCK))
            cy0, cy1 = 8 * r0, min(oh, 8 * (r0 + BLOCK))
            px0, px1 = max(0, cx0 - 8 * MARGIN), min(ow, cx1 + 8 * MARGIN)
            py0, py1 = max(0, cy0 - 8 * MARGIN), min(oh, cy1 + 8 * MARGIN)
            yy, xx = np.mgrid[py0:py1, px0:px1]
            hq, hr = cw.tile_hex(xx // 4, yy // 4)
            m = wet[hr, hq]
            if not m.any():
                continue
            if WET_PIXEL_MASK:
                # [WET PIXEL MASK] smooth hex weight (.5 on the hex edge), smoothstep above the floor: the marsh edge
                # follows the noise and the water distance, not the hex outline; it may reach into a neighbouring land
                # hex (spill) by the soft margin
                sw = (_box_blur(m.astype(np.float32), WET_SOFT[0]) - WET_SOFT[1]) / (1 - WET_SOFT[1])
                sw = np.clip(sw, 0, 1); ramp = sw * sw * (3 - 2 * sw)
                field = ramp * _box_blur(wshare[hr, hq], 8) + WET_NOISE[2] * fbm(xx / 8., yy / 8., (0, 0), WET_NOISE[0], WET_NOISE[1], seed=311)
                paint = spill[hr, hq] & (field > WET_NOISE[3])
            else:
                # Box blur of the mask: .5 on the hex edge, 1 at BLEND_EDGE pixels inside (a linear ramp).
                ramp = np.clip((_box_blur(m.astype(np.float32), BLEND_EDGE) - .5) * 2, 0, 1)
                # --- FIX2 WETLAND PATCHES --- (was ramp + BLEND_NOISE fbm > BLEND_CUT)
                field = ramp * _box_blur(wshare[hr, hq], 8) + WET_NOISE[2] * fbm(xx / 8., yy / 8., (0, 0), WET_NOISE[0], WET_NOISE[1], seed=311)
                paint = m & (field > WET_NOISE[3])
            paint &= (yy >= cy0) & (yy < cy1) & (xx >= cx0) & (xx < cx1)
            if protected is not None:
                paint &= ~protected[np.clip(yy // 4, 0, protected.shape[0] - 1), np.clip(xx // 4, 0, protected.shape[1] - 1)]
            if not paint.any():
                continue
            fy = oh - 1 - yy[paint]; fx = xx[paint]
            cur[fy, fx, 0] = WETLAND_MATERIAL
            cur[fy, fx, 1] = band[hr[paint], hq[paint]]
            painted += int(paint.sum())
    return painted


class WetlandCandidates:
    """Read-only wetland coverage for the single blend selector.

    ``rows(y0, y1, width)`` returns (weight float32, band uint8, hard_valid
    bool), all north-up DDS pixels. No material is sampled or written here.
    cells/domain are south-up hex grids; protected is the existing south-up
    tile grid. ``domain`` overrides the native/filler zone, not the land,
    ground-type, town-clearance or protected exclusions.

    water_field is {'shape': (DDS_height, DDS_width), 'mask': reader_or_array}
    or the same with 'distance_px'. A reader accepts (y0,y1,x0,x1), north-up.
    The mask must be the caller's authoritative pixel river/lake/sea mask;
    distance_px is its nonnegative Euclidean distance in LF pixels. No hex
    water-ring approximation is used. Distances beyond the last WET_WATER
    knot are equivalent, so mask readers need only a bounded halo EDT.
    Optional 'source' metadata is retained in provenance for the caller.

    Only this new candidate path uses SciPy window EDTs. The legacy painter
    above is unchanged and remains the compatibility path.
    """

    slot = WETLAND_MATERIAL

    @classmethod
    def from_inputs(cls, cells, names, protected=None, filler=True, *,
                    water_field=None, domain=None, ground_types=None,
                    eligible_fade_px=None, threshold_width=None, noise_scale=1.):
        if water_field is None:
            raise ValueError('Wetland candidates require authoritative pixel water_field')
        if not isinstance(water_field, dict):
            raise TypeError('water_field needs shape and exactly one mask/distance_px source')
        modes = [k for k in ('mask', 'distance_px') if k in water_field]
        if len(modes) != 1 or 'shape' not in water_field:
            raise ValueError('water_field needs shape and exactly one mask/distance_px source')
        self = cls()
        self.shape = tuple(int(v) for v in water_field['shape'])
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError('water_field.shape must be positive (DDS_height, DDS_width)')
        H, W = cells.shape[:2]
        if self.shape[1] != W * 8 or self.shape[0] not in (H * 8, H * 8 + 4):
            raise ValueError('water_field frame does not match the hex/DDS frame')
        self._water_mode, self._water = modes[0], water_field[modes[0]]
        if not callable(self._water) and not hasattr(self._water, 'read'):
            if np.shape(self._water) != self.shape:
                raise ValueError('water array shape differs from water_field.shape')
            if self._water_mode == 'mask' and np.asarray(self._water).dtype != np.bool_:
                raise TypeError('water mask arrays must be boolean')
        self._protected = None if protected is None else np.asarray(protected, bool)
        if self._protected is not None and self._protected.shape != (self.shape[0] // 4, self.shape[1] // 4):
            raise ValueError('protected must match the south-up DDS tile grid')
        if domain is None:
            inside = np.zeros((H, W), bool); inside[cw.NATIVE_HEX_SLICE] = True
            if filler:
                if names is None:
                    raise ValueError('filler wetland candidates require region names')
                zone = ~inside | filler_hex(cells, names)
            else:
                zone = ~inside
        else:
            zone = np.asarray(domain, bool)
            if zone.shape != (H, W):
                raise ValueError('domain must be a south-up hex grid matching cells')
        g = ground(cells)
        types = list(_land_ground_types() if ground_types is None else ground_types)
        if 'wetland' not in types or types.index('wetland') != 13:
            raise ValueError('Expected authoritative land_ground_types order (wetland=13)')
        self._wet = (g == 13) & zone
        self._eligible = ((cells[..., 0] & 3) != 1) & ~np.isin(g, (10, 11)) & zone
        self._eligible &= ~town_clear_hex(cells) & np.isin(g, [types.index(n) for n in WETLAND_GROUND_FROM if n in types])
        cl = climate_of(cells)
        self._band = np.full((H, W), BLEND_BAND['temperate'], np.uint8)
        for i, name in enumerate(('arid', 'cold', 'default', 'subtropical', 'temperate')):
            if name in BLEND_BAND and name != 'arid':
                self._band[cl == i] = BLEND_BAND[name]
        self._radius = max(4, int(WET_SOFT[0]))
        self._fade = float(self._radius if eligible_fade_px is None else eligible_fade_px)
        self._threshold_width = float(max(1e-4, 1 - WET_NOISE[3]) if threshold_width is None else threshold_width)
        self._noise_scale = float(noise_scale)
        if not np.isfinite([self._fade, self._threshold_width, self._noise_scale]).all() or self._fade <= 0 or self._threshold_width <= 0 or self._noise_scale < 0:
            raise ValueError('Wetland fade/threshold must be positive and noise_scale nonnegative')
        self._water_cap = float(WET_WATER[0][-1]) * 8
        self.provenance = {'frame': 'DDS north-up pixels; hex/tile fields south-up',
                           'water_mode': self._water_mode, 'water_source': water_field.get('source'),
                           'water_distance_cap_px': self._water_cap,
                           'eligible_fade_px': self._fade, 'threshold_width': self._threshold_width,
                           'wet_support_radius_px': self._radius, 'noise_scale': self._noise_scale}
        return self

    def _hex_pixels(self, field, box):
        y0, y1, x0, x1 = box
        xx = np.arange(x0, x1)[None, :]
        yy = (self.shape[0] - 1 - np.arange(y0, y1))[:, None]
        q, r = cw.tile_hex(xx // 4, yy // 4)
        H, W = field.shape
        valid = (q >= 0) & (q < W) & (r >= 0) & (r < H)
        return field[np.clip(r, 0, H - 1), np.clip(q, 0, W - 1)], valid

    def _logical_valid_pixels(self, box):
        out, valid = self._hex_pixels(self._eligible, box)
        out = out & valid
        if self._protected is not None:
            y0, y1, x0, x1 = box
            ty = ((self.shape[0] - 1 - np.arange(y0, y1)) // 4)[:, None]
            tx = (np.arange(x0, x1) // 4)[None, :]
            out &= ~self._protected[ty, tx]
        return out

    def _valid_pixels(self, box):
        out = self._logical_valid_pixels(box)
        water = self._read_water(box)
        out &= ~water if self._water_mode == 'mask' else water > 0
        return out

    def _read_water(self, box):
        a, b, c, d = box
        source = self._water
        raw = source(box) if callable(source) else source.read(box) if hasattr(source, 'read') else source[a:b, c:d]
        raw = np.asarray(raw)
        if raw.shape != (b - a, d - c):
            raise ValueError('water reader returned a crop with the wrong shape')
        if self._water_mode == 'mask':
            if raw.dtype != np.bool_:
                raise TypeError('water mask reader must return boolean pixels')
        else:
            raw = np.asarray(raw, np.float32)
            if np.isnan(raw).any() or (raw < 0).any():
                raise ValueError('water distance must be nonnegative pixels, not NaN')
        return raw

    def _distance_read_box(self, box):
        h, w = self.shape
        y0, y1, x0, x1 = box
        halo = int(np.ceil(self._water_cap)) + 1 if self._water_mode == 'mask' else 0
        return (max(0, y0 - halo), min(h, y1 + halo), max(0, x0 - halo), min(w, x1 + halo))

    def _distance_from_read(self, box, union, water):
        from scipy.ndimage import distance_transform_edt
        y0, y1, x0, x1 = box
        read_box = self._distance_read_box(box)
        a, b, c, d = read_box
        # Keep the original distance halo, even when the eligibility fade needs
        # a wider union. This preserves the old EDT boundary/padding exactly.
        raw = water[a-union[0]:b-union[0], c-union[2]:d-union[2]]
        dist = (distance_transform_edt(~raw) if raw.any() else np.full(raw.shape, self._water_cap)) if self._water_mode == 'mask' else raw
        return np.minimum(dist[y0-a:y1-a, x0-c:x1-c], self._water_cap).astype(np.float32)

    def _water_distance(self, box):
        read_box = self._distance_read_box(box)
        return self._distance_from_read(box, read_box, self._read_water(read_box))

    def window(self, box):
        """(weight, band, hard_valid) for absolute (y0,y1,x0,x1), north-up."""
        from scipy.ndimage import distance_transform_edt
        y0, y1, x0, x1 = (int(v) for v in box)
        h, w = self.shape
        if not (0 <= y0 <= y1 <= h and 0 <= x0 <= x1 <= w):
            raise ValueError('Wetland candidate box is outside the DDS frame')
        box = (y0, y1, x0, x1)
        valid = self._logical_valid_pixels(box)
        bands, _ = self._hex_pixels(self._band, box)
        weights = np.zeros(valid.shape, np.float32)
        if not valid.size or not valid.any():
            return weights, bands.astype(np.uint8), valid
        halo = max(self._radius, int(np.ceil(self._fade)) + 1)
        outer = (max(0, y0-halo), min(h, y1+halo), max(0, x0-halo), min(w, x1+halo))
        a, b, c, d = outer
        crop = (slice(y0-a, y1-a), slice(x0-c, x1-c))
        wet, wet_valid = self._hex_pixels(self._wet, outer)
        if not (wet & wet_valid).any():
            # No wet coverage, but retain the established physical-water
            # exclusion in hard_valid: it still needs one core-only read.
            water = self._read_water(box)
            valid &= ~water if self._water_mode == 'mask' else water > 0
            return weights, bands.astype(np.uint8), valid
        support = (_box_blur((wet & wet_valid).astype(np.float32), self._radius)[crop] - WET_SOFT[1]) / (1 - WET_SOFT[1])
        support = np.clip(support, 0, 1); support = support * support * (3 - 2 * support)
        distance_box = self._distance_read_box(box)
        union = (min(outer[0], distance_box[0]), max(outer[1], distance_box[1]),
                 min(outer[2], distance_box[2]), max(outer[3], distance_box[3]))
        water = self._read_water(union)
        core_water = water[y0-union[0]:y1-union[0], x0-union[2]:x1-union[2]]
        valid &= ~core_water if self._water_mode == 'mask' else core_water > 0
        if not valid.any():
            return weights, bands.astype(np.uint8), valid
        outer_water = water[a-union[0]:b-union[0], c-union[2]:d-union[2]]
        eligible = self._logical_valid_pixels(outer)
        eligible &= ~outer_water if self._water_mode == 'mask' else outer_water > 0
        inside = distance_transform_edt(np.pad(eligible, 1, constant_values=False))[1:-1, 1:-1][crop]
        inside = np.clip((inside - .5) / self._fade, 0, 1).astype(np.float32)
        inside = inside * inside * (3 - 2 * inside)
        proximity = np.interp(self._distance_from_read(box, union, water) / 8., *WET_WATER).astype(np.float32)
        xx, yy = np.meshgrid(np.arange(x0, x1) / 8., (h - 1 - np.arange(y0, y1)) / 8.)
        noise = self._noise_scale * WET_NOISE[2] * fbm(xx, yy, (0, 0), WET_NOISE[0], WET_NOISE[1], seed=311)
        # Eligibility attenuates the whole continuous field, including its noise,
        # before thresholding. Hard exclusions remain hard in the final selector.
        field = inside * (support * proximity + noise)
        t = np.clip((field - WET_NOISE[3]) / self._threshold_width + .5, 0, 1)
        weights = (t * t * (3 - 2 * t)).astype(np.float32)
        weights[~valid | (support <= 0)] = 0
        return weights, bands.astype(np.uint8), valid

    def rows(self, y0, y1, width):
        if int(width) != self.shape[1]:
            raise ValueError('rows width differs from wetland candidate DDS frame')
        return self.window((y0, y1, 0, int(width)))


def wetland_candidate_rows(shared, box):
    """Pure selector adapter: box=(north_y0,north_y1,x0,x1), no blend access."""
    model = shared if isinstance(shared, WetlandCandidates) else shared['wetland_candidates']
    weight, band, valid = model.window(box)
    return {'slot': model.slot, 'weight': weight, 'hard_valid': valid, 'band': band}


# ---------------------------------------------------------------- main
def shares(cl, mask, names):
    n = max(1, int(mask.sum()))
    return {names[i]: round(float(((cl == i) & mask).sum() / n), 3) for i in range(len(names)) if ((cl == i) & mask).any()}


def compute(m, no_wetlands=False):
    """Climate and wetlands for map dict m, in memory: {'after_cells': new cells, ...the fields main() reports}.
    Reads the vanilla map, analysis/west-real-terrain.npz, eco-grid-v2 and the Hehuang / Korea patches; writes
    nothing. main() adds the report, analysis/climate-wetland-hex.npz and the writes; the map bake calls this."""
    ndimage = _ndimage()
    cells = m['cells']; H, W = cells.shape[:2]
    names = m['climates']
    assert names == ['arid', 'cold', 'default', 'subtropical', 'temperate'], names
    assert m['land_ground_types'].index('wetland') == 13
    g = ground(cells); land = (g >= 0) & (g < len(m['land_ground_types']))
    inside = np.zeros((H, W), bool); inside[cw.NATIVE_HEX_SLICE] = True
    ids = map_hex.region_ids(cells); rnames = m['land_regions'] + m['sea_regions']
    isnew = np.array([n.startswith('tw3k_') for n in rnames] + [False])[np.where((ids >= 0) & (ids < len(rnames)), ids, len(rnames))]
    revealed = revealed_hex(cells, rnames)                    # vanilla non-playable land now playable: canvas
    canvas = ~inside | revealed
    target = land & canvas
    zones = {'new_regions': land & ~inside & isnew, 'canvas_filler': land & ~inside & ~isnew, 'revealed': land & revealed,
             'new_regions_in_slice': land & inside & isnew & ~revealed, 'native': land & inside & ~isnew & ~revealed}
    before = climate_of(cells).copy()
    geo = hex_geo(H, W)
    table, rows, vsummary = learn_vanilla(geo)
    real, hit = classify(geo, table, names)
    carried, dn, reach = native_carry(cells, names, land)
    out = np.where(real >= 0, real, before)
    # canvas outside the rectangle: within the meandered reach of it; revealed filler: where polish's ground
    # weight (revealed_weight) is still below REVEALED_CARRY, i.e. on and next to vanilla playable land
    near = (target & ~inside & (dn <= reach)) | (revealed & land & (revealed_weight(cells, rnames, REVEALED_RAMP) < REVEALED_CARRY))
    out = np.where(near, carried, out)
    # High ground in carry range keeps its real cold (classify: >= COLD_ELEV real metres is always cold): the carry
    # painted the Hengduan high belt west of the rectangle subtropical.
    high_cold = near & geo['ok'] & (geo['elev'] >= COLD_ELEV) & (real == names.index('cold'))
    out = np.where(high_cold, real, out)
    classes = [names.index(n) for n in ('arid', 'cold', 'subtropical', 'temperate')]
    out = majority(out, target & ~near, classes, *SMOOTH)
    # Western Regions: the polish table, cold crests away from towns.
    slot = ((cells[..., 2] >> 4) == 1)
    town_near = hex_grow(slot, WR_TOWN_KEEP)
    wr = target & geo['ok'] & (geo['lon_n'] < WR_LON_MAX) & (geo['lat_n'] >= WR_LAT_MIN) & (geo['lat_n'] < WR_LAT_MAX) & ~near
    plut = np.array([names.index(POLISH_CLIMATE.get(n, POLISH_OTHER)) for n in geo['names']])
    wr_val = plut[geo['biome']]
    wr_val = np.where((geo['elev'] >= COLD_ELEV) & ~town_near, names.index('cold'), wr_val)
    out = np.where(wr, majority(np.where(wr, wr_val, out), wr, classes, 3, 1), out)
    out = np.where(wr & slot, wr_val, out)                    # the town hex itself: unsmoothed polish class
    hh, hh_info = hehuang_climate(H, W, names, cells, rnames)
    if hh is not None:
        hsel = target & (hh >= 0); out = np.where(hsel, hh, out)
    out = np.where(target, out, before)
    after_cells = cells.copy()
    set_climate(after_cells, target, 0)                       # clear, then write per class
    for c in classes:
        set_climate(after_cells, target & (out == c), c)
    assert (climate_of(after_cells)[~target] == before[~target]).all()
    wet_before = int((g == 13)[target].sum())
    if no_wetlands:
        wet, by_source = np.zeros((H, W), bool), {}
    else:
        wet, by_source = wetland_hexes(after_cells, m, geo, target, isnew)
        set_ground(after_cells, wet, 13)
    changed_bytes = np.nonzero((after_cells != cells).any((0, 1)))[0].tolist()
    assert set(changed_bytes) <= {5, 6, 7}, changed_bytes
    assert ((after_cells[..., 2] & 8) == (cells[..., 2] & 8)).all()            # passability untouched
    after = climate_of(after_cells)
    return {'cells': cells, 'H': H, 'W': W, 'names': names, 'g': g, 'land': land, 'inside': inside, 'ids': ids, 'rnames': rnames, 'isnew': isnew, 'revealed': revealed, 'canvas': canvas, 'target': target, 'zones': zones, 'before': before, 'geo': geo, 'table': table, 'rows': rows, 'vsummary': vsummary, 'real': real, 'hit': hit, 'carried': carried, 'dn': dn, 'reach': reach, 'out': out, 'near': near, 'high_cold': high_cold, 'classes': classes, 'slot': slot, 'town_near': town_near, 'wr': wr, 'plut': plut, 'wr_val': wr_val, 'hh': hh, 'hh_info': hh_info, 'after_cells': after_cells, 'wet_before': wet_before, 'wet': wet, 'by_source': by_source, 'changed_bytes': changed_bytes, 'after': after}


def main():
    ndimage = _ndimage()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--apply', action='store_true', help='back up and write work/final-map/map.hex')
    ap.add_argument('--no-wetlands', action='store_true')
    ap.add_argument('--write-to', type=Path, help='write the resulting map.hex here only (offline checks)')
    args = ap.parse_args()
    t0 = time.time()
    m = map_hex.read(MAP)
    res = compute(m, no_wetlands=args.no_wetlands)
    (cells, H, W, names, g, land, inside, ids, rnames, isnew, revealed, canvas, target, zones, before, geo, table, rows, vsummary, real, hit, carried, dn, reach, out, near, high_cold, classes, slot, town_near, wr, plut, wr_val, hh, hh_info, after_cells, wet_before, wet, by_source, changed_bytes, after) = (res[k] for k in (
        'cells', 'H', 'W', 'names', 'g', 'land', 'inside', 'ids', 'rnames', 'isnew', 'revealed', 'canvas', 'target', 'zones', 'before', 'geo', 'table', 'rows', 'vsummary', 'real', 'hit', 'carried', 'dn', 'reach', 'out', 'near', 'high_cold', 'classes', 'slot', 'town_near', 'wr', 'plut', 'wr_val', 'hh', 'hh_info', 'after_cells', 'wet_before', 'wet', 'by_source', 'changed_bytes', 'after'))
    # Consistency with the polish tile climate (previous polish run) and towns.
    pt_path = A / 'analysis/polished-tile-climate.npy'
    agree = {}
    towns = []
    tile_inv = {v: names.index(k) for k, v in TILE_CLIMATE.items()}
    if pt_path.exists():
        pt = np.load(pt_path)       # read whole and closed (never a memory map of a workspace file)
        _, _, ty, tx = hex_tiles(H, W)
        pth = np.asarray(pt[np.clip(ty, 0, pt.shape[0] - 1), np.clip(tx, 0, pt.shape[1] - 1)]).astype(np.int32)
        pth = np.array([tile_inv.get(i, -1) for i in range(256)])[np.clip(pth, 0, 255)]
        for zk, zm in zones.items():
            agree[zk] = {'before': round(float((before[zm] == pth[zm]).mean()), 3) if zm.any() else None,
                         'after': round(float((after[zm] == pth[zm]).mean()), 3) if zm.any() else None,
                         'polish_tile_climate_share': shares(pth, zm, names)}
        design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
        for t in design['regions']:
            q, r = t['town']
            if not (0 <= q < W and 0 <= r < H) or not canvas[r, q]:
                continue
            towns.append({'region': t['region'], 'hex': [q, r], 'before': names[before[r, q]], 'after': names[after[r, q]],
                          'polish_tile': names[pth[r, q]] if pth[r, q] >= 0 else None,
                          'biome': geo['names'][geo['biome'][r, q]], 'metres': round(float(geo['elev'][r, q])), 'lat': round(float(geo['lat'][r, q]), 2)})
    report = {
        'map': str(MAP), 'apply': args.apply, 'seconds': None,
        'learned_vanilla': {'by_biome': vsummary, 'table_cells_used': sum(r['used'] for r in rows), 'table': rows,
                            'lat_bins': LAT_BINS, 'elev_bins': ELEV_BINS},
        'target_hexes': int(target.sum()), 'revealed_hexes': int((target & revealed).sum()), 'from_learned_table': int((hit & target).sum()), 'native_carry_hexes': int(near.sum()), 'carry_high_cold_hexes': int(high_cold.sum()),
        'western_regions_hexes': int(wr.sum()), 'hehuang': hh_info, 'hehuang_hexes_outside_slice': int((target & (hh >= 0)).sum()) if hh is not None else 0,
        'climate_share': {zk: {'before': shares(before, zm, names), 'after': shares(after, zm, names)} for zk, zm in zones.items()},
        'climate_changed_hexes': {zk: int((before != after)[zm].sum()) for zk, zm in zones.items()},
        'wetland': {'before_outside_slice': wet_before, 'after_outside_slice': int((ground(after_cells) == 13)[target].sum()),
                    'new_regions': int((wet & zones['new_regions']).sum()), 'canvas_filler': int((wet & zones['canvas_filler']).sum()),
                    'revealed': int((wet & zones['revealed']).sum()),
                    'per_1000_land_new_regions': round(1000 * float((wet & zones['new_regions']).sum()) / max(1, int(zones['new_regions'].sum())), 2),
                    'per_1000_land_native': round(1000 * float(((g == 13) & zones['native']).sum()) / max(1, int(zones['native'].sum())), 2),
                    'impassable_share': round(float(((cells[..., 2] & 8) > 0)[wet].mean()), 3) if wet.any() else None,
                    'impassable_share_new_regions': round(float(((cells[..., 2] & 8) > 0)[wet & zones['new_regions']].mean()), 3) if (wet & zones['new_regions']).any() else None,
                    'native_impassable_share': round(float(((cells[..., 2] & 8) > 0)[(g == 13) & zones['native']].mean()), 3),
                    'by_source': by_source},
        'polish_agreement': agree,
        'towns_outside_slice': {'count': len(towns), 'after_vs_polish_tile_mismatch': [t for t in towns if t['after'] != t['polish_tile']],
                                'before_vs_polish_tile_mismatch_count': sum(t['before'] != t['polish_tile'] for t in towns),
                                'after_share': {k: sum(t['after'] == k for t in towns) for k in ('arid', 'cold', 'subtropical', 'temperate')}},
        'attrition': 'unchanged: vanilla map.hex carries no per-hex attrition and terrain attrition damage is 0.005%',
        'changed_bytes': changed_bytes,
    }
    np.savez_compressed(HEX_OUT, climate=after.astype(np.int8), climate_before=before.astype(np.int8), wetland=wet, target=target,
                        new_regions=zones['new_regions'], biome=geo['biome'], elev=geo['elev'], lat=geo['lat'], lon=geo['lon'],
                        biome_names=np.array(geo['names']))
    if args.write_to:
        m2 = dict(m); m2['cells'] = after_cells
        map_hex.write(m2, args.write_to)
        import region_connectivity; region_connectivity.check_map(args.write_to)   # land regions stay whole (v17 crash); [to be absorbed by the map bake]
        report['written'] = str(args.write_to)
    if args.apply:
        stamp = time.strftime('%Y%m%d-%H%M%S')
        backup = MAP.with_name(f'map.hex.before-climate-{stamp}')
        shutil.copy2(MAP, backup)
        m['cells'] = after_cells
        map_hex.write(m, MAP)
        import region_connectivity; region_connectivity.check_map(MAP)   # land regions stay whole (v17 crash); [to be absorbed by the map bake]
        assert (map_hex.read(MAP)['cells'] == after_cells).all()
        report['backup'] = str(backup)
    report['seconds'] = round(time.time() - t0, 1)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    brief = {k: report[k] for k in ('target_hexes', 'revealed_hexes', 'from_learned_table', 'native_carry_hexes', 'carry_high_cold_hexes', 'hehuang_hexes_outside_slice',
                                    'climate_share', 'climate_changed_hexes', 'wetland', 'polish_agreement')}
    brief['towns'] = {k: v for k, v in report['towns_outside_slice'].items() if k != 'after_vs_polish_tile_mismatch'}
    brief['towns']['after_vs_polish_tile_mismatch'] = len(report['towns_outside_slice']['after_vs_polish_tile_mismatch'])
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print('dry run (no map written)' if not args.apply else f'written {MAP} (backup {report["backup"]})', REPORT, f'{report["seconds"]}s')


if __name__ == '__main__':
    sys.exit(main())

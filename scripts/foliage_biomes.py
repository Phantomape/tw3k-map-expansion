"""Tree species and density from the real ecoregion of each hex (polish_campaign_foliage).

The procedural replanting chose arid/cold/tropical/temperate trees from straight canvas bands
(biome_bands), so a taiga coast, a steppe and a jungle could get the same trees and density. With
the per-hex ecoregion biome of strategic_ink's cache (numpy only), species and density follow the
real vegetation: dense firs in taiga, broadleaf in temperate forest, subtropical trees in the
tropics, sparse trees in steppe and savanna, poplars only at desert oases, rocks on bare heights.
Returns None where no cache covers the canvas, so the old band rule applies.
"""
import numpy as np

import biome_profiles as _bp
import strategic_ink

# RESOLVE class groups (biome_profiles.BIOMES flags fol_*)
ROCK_HIGH = _bp.biome_set('fol_rock_high')       # snow, tundra, alpine: rock above 3.5
FIR = _bp.biome_set('fol_fir')                   # taiga, conifer
SUBTROPICAL = _bp.biome_set('fol_subtropical')   # tropical, mangrove
DRY = _bp.biome_set('fol_dry')                   # tropical_dry, savanna
SPARSE = _bp.biome_set('fol_sparse')             # steppe, alpine, tundra

_cache = None

# --- ROCK RELIEF --- canvas rock entries (general_rock_*, the 'bare heights' rule below and the frontier_forest
# clusters) on flat arid ground are dropped. The bare-heights test is an absolute surface height (> 4.8, vanilla's
# high mountains), but the canvas's high basins stand above it everywhere (Dunhuang-Guazhou ~1100-1500 m is
# surface 4.9-6.5), so rocks lay evenly over flat gobi (live O27: Dunhuang 2026-10-25). Kept only where the real
# ground has relief: the ETOPO elevation range over the 7x7-hex window (frontier_mountain_props' relief window)
# at least ROCK_RELIEF m on the arid classes (the rule and its constants: biome_profiles.flat_arid_hex,
# CONSTS['props']); the cold rock rule (ROCK_HIGH) and native rows are untouched. The props bake's accept refuses
# these rows before planting; this filter stays behind it until it counts 0.
ROCK_PREFIX = 'general_rock'


def flat_rock_hexes():
    """Hex mask (rows r, cols q) of arid hexes without real relief (ROCK RELIEF: biome_profiles.flat_arid_hex on
    the strategic_ink biome and ETOPO grids). Raises RuntimeError when the strategic_ink cache is unusable: the
    rule never switches itself off silently."""
    try:
        biome, names, _, elev = strategic_ink.load()
    except SystemExit as e:
        raise RuntimeError(f'ROCK RELIEF needs the strategic_ink cache (biome + ETOPO elevation): {e}') from None
    return _bp.flat_arid_hex(biome, names, elev)


def drop_flat_rocks(groups, canvas_hex, flat=None):
    """Remove canvas rock rows on flat arid hexes from tree groups ({'name', 'rows', 'added'}).
    canvas_hex(x, z) -> (q, r, canvas bool) per row position; flat: flat_rock_hexes() when already built
    (the props bake's accept rule reads the same mask). Returns {name: dropped}."""
    flat = flat_rock_hexes() if flat is None else flat
    out = {}
    import struct
    for g in groups:
        if not g['name'].startswith(ROCK_PREFIX) or not g['rows']:
            continue
        p = np.array([struct.unpack_from('<3f', r, 0) for r in g['rows']], np.float64)
        q, r, canvas = canvas_hex(p[:, 0], p[:, 2])
        ok = (q >= 0) & (q < flat.shape[1]) & (r >= 0) & (r < flat.shape[0])
        drop = canvas & ok
        drop[drop] = flat[r[drop], q[drop]]
        if drop.any():
            g['rows'] = [row for row, d in zip(g['rows'], drop) if not d]
            g['added'] = max(0, g['added'] - int(drop.sum()))
            out[g['name']] = int(drop.sum())
    return out
# --- end ROCK RELIEF

# --- AUTUMN SHRUBS --- the harvest-season orange of rock_shrub_* and the katsura (deciduous, vanilla's
# temperate tree) on open desert and cold land: frontier_forest's rock shrubs (OPEN_BIOMES: desert, steppe,
# alpine, tundra) and the sparse katsura of steppe / grass (choose: fol_sparse .008) stood as orange specks
# on the Gobi and round Lake Baikal (live O27 Dingling). Canvas rows are dropped, after every draw (no random
# stream moves): rock shrubs on real desert or map.hex cold climate; katsura there too unless the hex is real
# forest (tree_fill biomes: the deciduous Manchurian / Ussuri / Siberian mixed forest keeps its broadleaves).
# The families and their bans: biome_profiles.SPECIES_RULES / species_bans.


def species_ban_masks(cold):
    """{family: hex mask where it may not stand} for every biome_profiles.SPECIES_RULES family (species_bans on
    the strategic_ink biome and map.hex cold climate). Raises RuntimeError without the strategic_ink cache (as
    ROCK RELIEF)."""
    try:
        biome, names, _, _ = strategic_ink.load()
    except SystemExit as e:
        raise RuntimeError(f'AUTUMN SHRUBS needs the strategic_ink cache (biome): {e}') from None
    return {k: _bp.species_bans(k, biome, names, cold) for k in _bp.SPECIES_RULES}


def drop_autumn_shrubs(groups, canvas_hex, cold, masks=None):
    """Remove canvas rows of a SPECIES_RULES family on its banned hexes (autumn-orange shrubs / katsura on open
    desert / cold land). cold: map.hex cold climate per hex; masks: species_ban_masks(cold) when already built
    (the props bake's accept rule reads the same masks). Returns {name: dropped}."""
    masks = species_ban_masks(cold) if masks is None else masks
    out = {}
    import struct
    for g in groups:
        key = next((k for k in masks if g['name'].startswith(k)), None)
        if key is None or not g['rows']:
            continue
        mask = masks[key]
        p = np.array([struct.unpack_from('<3f', r, 0) for r in g['rows']], np.float64)
        q, r, canvas = canvas_hex(p[:, 0], p[:, 2])
        drop = canvas & (q >= 0) & (q < mask.shape[1]) & (r >= 0) & (r < mask.shape[0])
        drop[drop] = mask[r[drop], q[drop]]
        if drop.any():
            g['rows'] = [row for row, d in zip(g['rows'], drop) if not d]
            g['added'] = max(0, g['added'] - int(drop.sum()))
            out[g['name']] = int(drop.sum())
    return out
# --- end AUTUMN SHRUBS


def _load():
    global _cache
    if _cache is None:
        try:
            biome, names, _, _ = strategic_ink.load()
            _cache = (biome, names)
        except SystemExit:
            _cache = (None, None)
    return _cache


# Shared forest density (user decision 2026-10-26: woodland ground and trees read ONE field,
# biome_profiles.forest_density, persisted in landform-tiles.npz as forest_density). The field SHAPES the band
# chance; the level per class is BAND_MEAN, the expected chance of the pre-bake rule (choose's high / low chances
# by the sin patch, which stood above -.1 / .05 / .1 on .587 / .459 / .416 of the hexes), so the props bake keeps
# the tree counts per biome (calibration 2026-10-30: .45 x density lost 54 % of the procedural trees, mostly the
# fade beyond the fill reach and the patch factor's mean .8). chance = BAND_MEAN[class] x shape, shape the hex's
# density over its class mean (tree_bake.forest_shape: mean 1 per class, 1 where the field is faded out).
BAND_MEAN = {'fir': .253,           # .36 / .10 at .587
             'subtropical': .305,   # .45 / .10 at .587
             'dry': .090,           # .16 / .03 at .459
             'forest': .221,        # .32 / .08 at .587
             'sparse': .008,
             'open': .050,          # grass, wetland, paddy and the rest: .10 / .015 at .416
             'desert': .004}        # beyond TOWN_OASIS
# Per eco class factor on the band level and the map forest level (calibration 2026-10-30, measured on the main
# workspace: the one biome source, eco_bio instead of strategic_ink, moved trees from open eco classes that
# strategic_ink called forest / taiga; these factors keep the per-biome counts of the pre-bake rule within 10 %)
CLASS_LEVEL = {'tundra': 1.3, 'steppe': 1.3, 'grass': 1.8, 'paddy': 1.3, 'savanna': 1.25, 'wetland': 1.15}
# --- TOWN OASIS --- poplar belts round the towns of the dry country (props rule, not forest density). The rule data
# is biome_profiles CONSTS['props']['TOWN_OASIS'] (diversity, d9ebb683; its comment holds the rationale): the chance
# 'chance' on band-walk hexes within 'ring_hex' of their nearest town whose eco_bio class is in 'classes', or in
# 'oasis_classes' on an oasis hex (landform-tiles 'oasis' >= 'oasis_min'); and the floor: a town's oasis hexes (walk
# hexes within 'floor_ring_hex' of their nearest town, class in classes or oasis_classes, oasis >= oasis_min) hold at
# least ceil('min_hit_share' x their count) poplar hits of 'trees_per_hit' trees. The pre-bake rule fired on desert
# only, so the oasis towns eco_bio calls steppe (Yanqi, Qiuci, Gumo: strategic_ink said forest) lost their belts.
SPECIES_OASIS = ('arid_tree_poplar_small_1', 'arid_tree_poplar_medium_1')


_oasis_rule = None


def town_oasis_rule(refresh=False):
    """The TOWN OASIS rule data, biome_profiles CONSTS['props']['TOWN_OASIS'], read once (refresh=True rereads it
    after a table edit). Raises KeyError without the entry: the rule never switches itself off silently."""
    global _oasis_rule
    if _oasis_rule is not None and not refresh:
        return _oasis_rule
    rule = dict(_bp.CONSTS['props']['TOWN_OASIS'])
    for k in ('classes', 'oasis_classes', 'ring_hex', 'floor_ring_hex'):
        rule[k] = tuple(rule[k])
    _oasis_rule = rule
    return rule


def oasis_chance_hex(b, oasis, rule=None):
    """True when a hex of eco_bio class b and oasis value takes the TOWN OASIS chance (inside the ring)."""
    rule = rule or town_oasis_rule()
    return b in rule['classes'] or (b in rule['oasis_classes'] and oasis >= rule['oasis_min'])


def oasis_floor_hits(n_oasis, rule=None):
    """Poplar hits a town's oasis hexes must hold: ceil(min_hit_share x their count)."""
    rule = rule or town_oasis_rule()
    return np.ceil(np.asarray(n_oasis, np.float64) * rule['min_hit_share'] - 1e-9).astype(np.int64)


def band_class(b):
    """BAND_MEAN key of a RESOLVE class name."""
    if b == 'desert':
        return 'desert'
    if b in FIR:
        return 'fir'
    if b in SUBTROPICAL:
        return 'subtropical'
    if b in DRY:
        return 'dry'
    if b == 'forest':
        return 'forest'
    if b in SPARSE:
        return 'sparse'
    return 'open'


# Where the shared field is faded out (forest_fill < tree_bake.SHAPE_FILL_MIN: the native rectangle's repurposed
# filler, the Western Regions band, beyond the fill reach) the band keeps the pre-bake sin patch as its shape: the
# same high / low chances over the same hexes, divided by the class mean (calibration 2026-10-30: shape 1 there
# moved trees between regions the long-wave patch had favoured, e.g. Shan -40 %, Lin Yi -80 %)
PATCH_LEVELS = {'fir': (.36, .10, -.1), 'subtropical': (.45, .10, -.1), 'dry': (.16, .03, .05),
                'forest': (.32, .08, -.1), 'open': (.10, .015, .1)}


def patch_shape(b, patch):
    """Band shape of class b from the pre-bake sin patch (polish's patch value of the hex): its high / low chance over
    the class mean BAND_MEAN; 1 for the classes without a patch rule (sparse, desert, rocks)."""
    k = band_class(b) if b is not None else 'open'
    if k not in PATCH_LEVELS:
        return 1.
    hi, lo, thr = PATCH_LEVELS[k]
    return (hi if patch > thr else lo) / BAND_MEAN[k]


STEPPE_CAP = .015   # tree probability cap on map grassland hexes (props rule)
# Map forest ground level: the pre-bake floor's expected chance (.32 / .08 by the sin patch, .587 above -.1), shaped
# like the band chance; the table's map_forest_floor stays the minimum (calibration 2026-10-30: the bare .08 floor
# lost a third of the trees on open classes under map forest: grass, paddy, steppe, tundra)
MAP_FOREST_MEAN = .221


def map_ground_chance(species, chance, ground_name, shape=1., biome=None):
    """The band chance after the map ground type: a forest-ground hex keeps at least MAP_FOREST_MEAN x shape
    (the hex's forest shape, tree_bake.forest_shape) and never less than biome_profiles CONSTS ground_materials
    FOREST_DENSITY['map_forest_floor'] (gameplay forest has trees; the woodland ground gets no such floor), a
    grassland hex at most STEPPE_CAP. Rocks are left as they are."""
    if species.startswith(ROCK_PREFIX):
        return chance
    if 'forest' in ground_name:
        chance = max(chance, _bp.CONSTS['ground_materials']['FOREST_DENSITY']['map_forest_floor'],
                     min(1., MAP_FOREST_MEAN * CLASS_LEVEL.get(biome, 1.) * float(shape)))
    if ground_name in ('grassland', 'hilly_grassland'):
        chance = min(chance, STEPPE_CAP)
    return chance


def choose(q, r, ground, shape, distance, rng, biome=None, oasis=0.):
    """(species, chance) for the hex (q, r), or None to keep the band rule. ground: surface relief; shape: the
    shared forest density of the hex over its class mean (tree_bake.forest_shape, mean 1 per class); distance:
    hexes to the nearest town; biome: the hex's RESOLVE class name, from landform-tiles eco_bio (the classes the
    density was made from; polish passes it), else the strategic_ink biome. The chance is BAND_MEAN[class] x
    shape (monotone in the density), except the bare-heights rocks and the desert town oases (props rules)."""
    if biome is None:
        grid, names = _load()
        if grid is None:
            return None
        biome = names[int(grid[r, q])]
    b = biome
    if ground > 4.8 or b in ROCK_HIGH and ground > 3.5:
        return 'general_rock_medium_1', .06
    chance = min(1., BAND_MEAN[band_class(b)] * CLASS_LEVEL.get(b, 1.) * float(shape))
    rule = town_oasis_rule()   # TOWN OASIS: oasis: landform-tiles 'oasis' at the hex (block-centre tile)
    if rule['ring_hex'][0] <= distance < rule['ring_hex'][1] and oasis_chance_hex(b, oasis, rule):
        sp = SPECIES_OASIS[0] if rng.random() < .5 else SPECIES_OASIS[1]
        return sp, max(rule['chance'], chance)
    if b == 'desert':
        sp = SPECIES_OASIS[0] if rng.random() < .5 else SPECIES_OASIS[1]
        return sp, chance
    if b in FIR:
        sp = 'cold_tree_fir_small_1' if rng.random() < .5 else 'cold_tree_fir_medium_1'
    elif b in SUBTROPICAL:
        sp = 'bamboo_small_1' if rng.random() < .18 else 'subtropical_castanopsis_medium_1'
    elif b in DRY:
        sp = 'subtropical_castanopsis_medium_1'
    elif b in SPARSE:
        sp = 'cold_tree_fir_small_1' if b != 'steppe' else 'temperate_tree_katsura_small_1'
    else:   # forest, grass, wetland, paddy
        sp = 'temperate_tree_katsura_small_1' if rng.random() < .5 else 'temperate_tree_katsura_medium_1'
    return sp, chance


# --- TREELINE ALPINE --- no trees above the treeline on the Tibetan alpine meadow / alpine steppe (beautify H).
# The sparse rule (choose: fol_sparse .008, cold_tree_fir_small_1 on alpine) and the other plantings set firs
# on the open plateau far above any forest. Decided by real geography only, never by surface height: a hex
# whose block-centre tile is biome_profiles.classify alpine_meadow + alpine_steppe >= TREELINE_CLASS_MIN (real
# terrain, analysis/west-real-terrain.npz) and whose ETOPO elevation there is above COLD_LINES['TREELINE'] at
# its latitude. Real conifer stays: hexes whose strategic_ink biome or block-centre RESOLVE class is a fol_fir
# biome (taiga, conifer: the south-east Tibet margin) are never in the mask. polish_campaign_foliage applies
# it after every placement (TREELINE ALPINE block there): canvas tree rows on the mask are dropped, rocks and
# shrubs stay, no random stream moves. Below the treeline the .008 sparse firs stay as they are.
TREELINE_BOX = (76., 105., 26.5, 40.)    # lon0, lon1, lat0, lat1 searched (the classify Tibet bands with margin)
TREELINE_MARGIN = 96                     # tile cells of context around the box (classify blurs, dryness 2 x DRY_R)
TREE_PREFIXES = ('cold_tree_', 'temperate_tree_', 'arid_tree_', 'subtropical_', 'bamboo_')


def treeline_hexes(real_terrain, origin):
    """Hex mask (rows r, cols q): above the treeline on the alpine meadow / steppe (TREELINE ALPINE).
    real_terrain: path of west-real-terrain.npz; origin: polish's authored noise origin (NQ, NR)."""
    biome, names, _, _ = strategic_ink.load()
    H, W = biome.shape
    with np.load(real_terrain) as z:
        x0, x1, y0, y1 = (int(v) for v in z['box'])
        lon, lat = z['lon'], z['lat']
        inb = (lon >= TREELINE_BOX[0]) & (lon <= TREELINE_BOX[1]) & (lat >= TREELINE_BOX[2]) & (lat <= TREELINE_BOX[3])
        if not inb.any():
            return np.zeros((H, W), bool)
        rows, cols = np.flatnonzero(inb.any(1)), np.flatnonzero(inb.any(0))
        r0, r1 = max(0, int(rows[0]) - TREELINE_MARGIN), min(inb.shape[0], int(rows[-1]) + 1 + TREELINE_MARGIN)
        c0, c1 = max(0, int(cols[0]) - TREELINE_MARGIN), min(inb.shape[1], int(cols[-1]) + 1 + TREELINE_MARGIN)
        del inb
        win = (slice(r0, r1), slice(c0, c1))
        lon, lat = lon[win].copy(), lat[win].copy()
        elev = z['elevation'][win].astype(np.float32)
        rbio = z['biome'][win].copy()
        bnames = [str(n) for n in z['biome_names']]
    ty, tx = np.mgrid[y0 + r0:y0 + r1, x0 + c0:x0 + c1].astype(np.float32)
    w, _ = _bp.classify(elev, rbio, bnames, lon, lat, tx / 2, ty / 2, origin)   # polish: q = x / 2, r = y / 2
    tib = w['alpine_meadow'] + w['alpine_steppe']
    del w, tx, ty
    # the rule: biome_profiles.alpine_treeline_tile (CONSTS['props'] TREELINE_CLASS_MIN, COLD_LINES TREELINE).
    # Inputs as before: classify rerun on the raw RESOLVE grid inside the box (not the landform field yet)
    fir_t = np.isin(rbio, [bnames.index(n) for n in FIR if n in bnames])
    tile = _bp.alpine_treeline_tile(tib, elev, lat, fir_t)
    # block-centre tile of every hex (tile 2q+1, 2r+q%2+1), where it lies in the cropped window
    rr, qq = np.mgrid[:H, :W]
    cy = 2 * rr + qq % 2 + 1 - y0 - r0
    cx = 2 * qq + 1 - x0 - c0
    ok = (cy >= 0) & (cy < tile.shape[0]) & (cx >= 0) & (cx < tile.shape[1])
    out = np.zeros((H, W), bool)
    out[ok] = tile[cy[ok], cx[ok]]
    return out & ~np.isin(biome, [names.index(n) for n in FIR if n in names])


def drop_treeline_trees(groups, canvas_hex, mask):
    """Remove canvas tree rows (TREE_PREFIXES) on mask hexes. canvas_hex(x, z) -> (q, r, canvas bool).
    Returns ({name: dropped}, sorted [(q, r)] of the dropped rows)."""
    import struct
    out, where = {}, []
    for g in groups:
        if not g['name'].startswith(TREE_PREFIXES) or not g['rows']:
            continue
        p = np.array([struct.unpack_from('<3f', r, 0) for r in g['rows']], np.float64)
        q, r, canvas = canvas_hex(p[:, 0], p[:, 2])
        drop = canvas & (q >= 0) & (q < mask.shape[1]) & (r >= 0) & (r < mask.shape[0])
        drop[drop] = mask[r[drop], q[drop]]
        if drop.any():
            g['rows'] = [row for row, d in zip(g['rows'], drop) if not d]
            g['added'] = max(0, g['added'] - int(drop.sum()))
            out[g['name']] = int(drop.sum())
            where += list(zip(q[drop].tolist(), r[drop].tolist()))
    return out, sorted(where)
# --- end TREELINE ALPINE

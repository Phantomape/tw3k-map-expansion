"""Props bake (bake design step 3, analysis/pipeline-bake-design.md 3.4): stateless random numbers and the one
accept(cell, species) predicate of the campaign tree list (polish_campaign_foliage and its planters).

Random numbers. Every planter used to draw from its own sequential stream (np.random.default_rng(190219),
8190, 8891, 4473, 9051 + patch): one tree more or less upstream moved every later draw, so each fix had to
become a post-plant filter. Here a draw is a pure function of (purpose, keys): splitmix64 over the purpose
id and integer keys (hex q, r or tile x, y, an index), vectorised in numpy, no Generator objects. The same
hex with the same purpose gets the same number whatever was planted before it, in any order, in any chunking.
Not keyed on the variant (every variant reseeded the same way before).

Accept. Every planter asks accept(tx, ty, species) before it writes a row; the rules are the ones the four
post-plant filters encode (WET TREES, TREELINE ALPINE, ROCK RELIEF, AUTUMN SHRUBS), evaluated on the row's
position exactly as the filter will read it back (float32 world coordinates), so a row accepted here is a
row the filter keeps. The filters keep running after every planting and must count 0 removals before they
are deleted (design 3.4: the retirement order). The rule masks come from the existing rule functions
(foliage_biomes, which call biome_profiles' predicates); nothing here redefines a rule.

Dry ground: one predicate, dry_enough(lo, local) = lo >= max(WET_LEVEL, local water level + BANK_TREE_MARGIN),
lo the footprint minimum of the ground (3 x 3 samples, +-FOOT tile), local the highest water level under the
footprint: the max of the river level, the lake level on lake tiles and the sea level on sea tiles
(local_water). It replaces the three per-tree tests (DRY_GROUND .25 at the tree's point in frontier_forest,
native_edge_trees, riparian_belts; WET_LEVEL plus the river level in WET TREES). Ownership (integrator,
2026-10-26): the water facts (per-tile level, footprint, lake and sea levels) come from the river spec / height
bake, the margin is a river-spec constant, the predicate is diversity's rule function. Until those land both
live here, in one place, marked [to read the river spec]. frontier_forest's hex-level shore mask (DRY_GROUND at
the hex centre, one ring out) is a candidate rule, not the per-tree test, and stays.

numpy only (campaign_world for the frame).
"""
import hashlib
import struct

import numpy as np

import campaign_world as cw

SEED = 190                      # one global salt: changing it reshuffles every procedural tree at once
LAKE_PLANE = -.015              # campaign water plane (polish LAKE_WATER)
WET_LEVEL = LAKE_PLANE + .065   # the plane + margin; the river level takes the same margin
# ---- [to read the river spec] water facts and bank margin (river spec / height bake; one place until then)
WATER_PLANE = LAKE_PLANE        # sea level and lake level: polish carves every lake bowl and the sea to LAKE_WATER
BANK_TREE_MARGIN = WET_LEVEL - LAKE_PLANE   # river-spec constant: footprint ground this far above the local water
LAKE_REGION = '3k_main_sea_lake'            # map.hex region of the inland lakes (polish's _lake); other water = sea
WATER_KINDS = ('none', 'river', 'lake', 'sea')
DRY_LEVEL = WET_LEVEL           # floor of dry_enough where no water fact covers the footprint
RIVER_MARGIN = BANK_TREE_MARGIN # (old name, the WET TREES filter's margin)
# ---- end [to read the river spec]
FOOT = .35                      # footprint half width (tile cells), 3 x 3 samples
# species classes of the filters' rules (one definition: foliage_biomes)
from foliage_biomes import TREE_PREFIXES, ROCK_PREFIX

# ------------------------------------------------------------------ stateless random numbers
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_PURPOSES = {}


def purpose_id(purpose):
    """64-bit id of a purpose string (blake2b of SEED and the name: stable across runs and machines,
    unlike hash())."""
    pid = _PURPOSES.get(purpose)
    if pid is None:
        pid = int.from_bytes(hashlib.blake2b(f'{SEED}:{purpose}'.encode(), digest_size=8).digest(), 'little')
        _PURPOSES[purpose] = pid
    return pid


def _mix(z):
    """splitmix64 finalizer (uint64 arrays, wrapping arithmetic)."""
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def hash64(purpose, *keys):
    """uint64 per element of the broadcast keys (integers; negative keys wrap as two's complement)."""
    ks = np.broadcast_arrays(*[np.asarray(k).astype(np.int64) for k in keys]) if keys else [np.zeros((), np.int64)]
    with np.errstate(over='ignore'):
        h = np.full(ks[0].shape, purpose_id(purpose), np.uint64)
        for k in ks:
            h = _mix(h ^ _mix(k.astype(np.uint64) + _GOLDEN))
    return h


def u01(purpose, *keys):
    """float64 in [0, 1) per element: the top 53 bits of hash64."""
    return (hash64(purpose, *keys) >> np.uint64(11)).astype(np.float64) * (1. / (1 << 53))


def uniform(lo, hi, purpose, *keys):
    return lo + (hi - lo) * u01(purpose, *keys)


def integers(n, purpose, *keys):
    """int64 in [0, n) (n scalar or array, n >= 1)."""
    return np.minimum((u01(purpose, *keys) * n).astype(np.int64), np.asarray(n, np.int64) - 1)


def normal(purpose, *keys):
    """Standard normal (Box-Muller over two independent purposes)."""
    a, b = u01(purpose + '/a', *keys), u01(purpose + '/b', *keys)
    return np.sqrt(-2. * np.log1p(-a)) * np.cos(2. * np.pi * b)


def categorical(p, purpose, *keys):
    """Index into p (weights, normalised here) per element."""
    c = np.cumsum(np.asarray(p, np.float64))
    return np.minimum(np.searchsorted(c / c[-1], u01(purpose, *keys), side='right'), len(c) - 1)


class Draw:
    """A fixed number behind the rng.random() interface (foliage_biomes.choose takes an rng and draws
    at most once): the caller passes the hashed draw of the hex."""
    __slots__ = ('u',)

    def __init__(self, u):
        self.u = float(u)

    def random(self):
        return self.u


def row_keys(rows):
    """(x bits, z bits) int64 per row: the stored float32 position as integer keys (position-keyed draws for
    rows, independent of their order in the list)."""
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    p = np.frombuffer(b''.join(bytes(r[:12]) for r in rows), '<u4').reshape(-1, 3).astype(np.int64)
    return p[:, 0], p[:, 2]


# ------------------------------------------------------------------ frame helpers
def native_many(x, z):
    """World points inside the native rectangle (polish_campaign_foliage's native test)."""
    q = x / cw.DX; zr = z / cw.DZ
    return (cw.PAD_LEFT <= q) & (q < cw.PAD_LEFT + cw.NATIVE_W) & (cw.PAD_BOTTOM <= zr) & (zr < cw.PAD_BOTTOM + cw.NATIVE_H)


NATIVE_Y0_TOL = .25   # stage_invariants native_y0_trees_grounded: |tree - final LF| for vanilla y == 0 rows


def native_row_heights(y, moved, ground):
    """Heights of kept native (vanilla) rows. A row follows the ground change under it: vanilla y + (final -
    vanilla ground). Vanilla stores y == 0 for some rows (the native south edge, e.g. bamboo at z ~289): those
    are not placed heights, so they go on the final ground, the final LF that procedural rows use
    (pack_row y = sample(surface, ...)) (coordinator 2026-10-30). Every other row keeps the delta rule (single
    vanilla baseline). Same float64 arithmetic as before for y != 0, so those rows stay byte-identical."""
    y = np.asarray(y, np.float64)
    return np.where(y == 0, np.asarray(ground, np.float64), y + np.asarray(moved, np.float64))


def stored(tx, ty):
    """World (x, z) float64 of tile positions as a row stores them (float32), and their tile coordinates read
    back the way the post-filters read rows (x / DX * 2)."""
    x = np.asarray(np.asarray(tx, np.float64) * cw.DX / 2, np.float32).astype(np.float64)
    z = np.asarray(np.asarray(ty, np.float64) * cw.DZ / 2, np.float32).astype(np.float64)
    return x, z, x / cw.DX * 2, z / cw.DZ * 2


def positions(rows):
    if not rows:
        return np.zeros((0, 3))
    return np.frombuffer(b''.join(bytes(r[:12]) for r in rows), '<f4').reshape(-1, 3).astype(np.float64)


def load_river_level(path, shape):
    """analysis/river-water-level.npy as the (TH, TW) float array (a stamped writer may wrap it in an npz-style
    container: the level is the array of the tile shape), or None when the file is missing."""
    if not path.exists():
        return None
    level = np.load(path, allow_pickle=False)
    if hasattr(level, 'files'):
        level = next(level[k] for k in sorted(level.files, key=lambda k: k != 'level') if level[k].shape == shape and level[k].dtype.kind == 'f')
    assert level.shape == shape, (level.shape, shape)
    return level


# ------------------------------------------------------------------ water facts and the dry predicate
def dry_enough(lo, local):
    """[to read the river spec] the one dry-ground predicate (diversity's rule function when it lands):
    footprint minimum lo >= max(WET_LEVEL, local water level + BANK_TREE_MARGIN); local NaN / -inf = no water."""
    return lo >= np.fmax(WET_LEVEL, np.asarray(local, np.float64) + BANK_TREE_MARGIN)


def local_water(cells, names, river_level, shape, sea_level=WATER_PLANE, lake_level=WATER_PLANE, chunk=256):
    """[to read the river spec] (level, kind) on the tile grid: level = the highest water level of the tile
    (river level where river-water-level.npy has one, lake level on tiles of lake hexes, sea level on tiles of
    the other water hexes; NaN = dry land), kind = WATER_KINDS index of the level that wins. The tile -> hex
    owner rule is campaign_world.tile_hex. sea_level / lake_level: scalars or tile arrays."""
    water = (cells[..., 0] & 3) == 1
    ids = (cells[..., 1].astype(np.int32) << 5) + (cells[..., 0].astype(np.int32) >> 3) - 1
    lake = water & (ids == names.index(LAKE_REGION)) if LAKE_REGION in names else np.zeros(water.shape, bool)
    sea = water & ~lake
    TH, TW = shape
    level = np.full(shape, np.nan, np.float32); kind = np.zeros(shape, np.uint8)
    tx = np.arange(TW)
    for y0 in range(0, TH, chunk):
        ys = np.arange(y0, min(TH, y0 + chunk))
        hq, hr = cw.tile_hex(tx[None, :], ys[:, None])
        hq, hr = np.broadcast_arrays(hq, hr)
        lv = level[y0:y0 + len(ys)]; kd = kind[y0:y0 + len(ys)]
        for k, m, v in ((3, sea[hr, hq], sea_level), (2, lake[hr, hq], lake_level)):
            v = np.broadcast_to(np.asarray(v, np.float32), shape)[y0:y0 + len(ys)] if np.ndim(v) else np.float32(v)
            vv = np.where(m, v, np.nan)
            win = m & ~(lv >= vv)
            lv[win] = np.broadcast_to(vv, lv.shape)[win]; kd[win] = k
        if river_level is not None:
            rv = river_level[y0:y0 + len(ys)].astype(np.float32)
            win = np.isfinite(rv) & ~(lv >= rv)
            lv[win] = rv[win]; kd[win] = 1
    return level, kind


def forest_field_hex(A, shape):
    """The shared forest field per hex (rows r, cols q), read from landform-tiles.npz (polish GROUND MATERIALS
    writes it) at each hex's block-centre tile (landform_tiles.hex_view):
      density  forest_density (biome_profiles: base x patch x forest_fill), float32
      fill     forest_fill (the fill fade, 0 on the west band and the native rectangle), float32
      eco      eco_bio (real_bio_eco, the classes the density was made from), uint8; names: meta biome_names
    Required: without it the trees would not follow the woodland ground (RuntimeError)."""
    import landform_tiles
    lt = landform_tiles.load(A)
    need = ('forest_density', 'forest_fill', 'eco_bio', 'oasis')
    if lt is None or any(k not in lt for k in need) or 'biome_names' not in lt['meta']:
        raise RuntimeError(f'{landform_tiles.NAME} without {need} / meta biome_names: run polish (GROUND MATERIALS) first')
    hv = lambda k: landform_tiles.hex_view(lt, k, shape)
    out = {'density': hv('forest_density').astype(np.float32), 'fill': hv('forest_fill').astype(np.float32),
           'eco': hv('eco_bio'), 'names': list(lt['meta']['biome_names'])}
    out['shape'] = forest_shape(out['density'], out['fill'], out['eco'])
    # oasis_ground's oasis field (polish persists it in landform-tiles): the TOWN OASIS floor counts its hexes
    out['oasis'] = hv('oasis').astype(np.float32)
    return out


SHAPE_FILL_MIN = .05   # forest_fill below this: the field is faded out, the shape is 1 (the class level)


def forest_shape(density, fill, eco):
    """Per hex the shared forest density as a SHAPE with mean 1 per eco class: density / forest_fill (base x patch,
    the fade taken out: the planters keep their own reach) over the mean of that ratio on the class's hexes where
    the field is defined (fill >= SHAPE_FILL_MIN); 1 where it is not (beyond the fade, the west band, the native
    rectangle) and for classes whose density is 0 (desert, snow). The planters multiply their per-class level by
    it, so the field decides where the trees of a class stand and the level how many (calibration 2026-10-30)."""
    density = np.asarray(density, np.float64); fill = np.asarray(fill, np.float64); eco = np.asarray(eco)
    on = fill >= SHAPE_FILL_MIN
    dn = np.where(on, density / np.maximum(fill, SHAPE_FILL_MIN), 0.)
    out = np.ones(density.shape, np.float32)
    n = int(eco.max()) + 1 if eco.size else 0
    cnt = np.bincount(eco[on].ravel().astype(np.int64), minlength=n).astype(np.float64)
    tot = np.bincount(eco[on].ravel().astype(np.int64), weights=dn[on].ravel(), minlength=n)
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.)
    m = mean[eco.astype(np.int64)]
    use = on & (m > 1e-6)
    out[use] = (dn[use] / m[use]).astype(np.float32)
    return out


# ------------------------------------------------------------------ the accept predicate
def _hex_lookup(mask, q, r):
    ok = (q >= 0) & (q < mask.shape[1]) & (r >= 0) & (r < mask.shape[0])
    out = np.zeros(len(q), bool)
    out[ok] = mask[r[ok], q[ok]]
    return out


class TreeAccept:
    """accept(tx, ty, species) -> bool per row about to be planted (tile coordinates, species name per row or one
    name). Rules (each one also a post-plant filter that must count 0 behind it):
      dry       every planted row: dry_enough(footprint minimum, highest local water level under the footprint:
                river, lake, sea); rejections also counted per water kind in self.dry_by_water
      treeline  canvas tree rows (TREE_PREFIXES) on treeline hexes (foliage_biomes.treeline_hexes)
      flat_rock canvas rock rows (general_rock_*) on flat arid hexes (foliage_biomes.flat_rock_hexes)
      species_ban  canvas rows of a biome_profiles.SPECIES_RULES family on its banned hexes (rock shrubs on
                desert / cold, katsura there unless real forest: foliage_biomes.species_ban_masks)
      habitat   per-prefix hex whitelists (region whitelists; none for tree-list species yet)
    canvas = outside the native rectangle at the stored position, as the filters test it; the dry rule also
    holds for rows planted inside the native rectangle (replanted hexes), whose WET filter never ran.
    Masks given as None switch their rule off (their filter is off then too). Counts rejections per rule and
    species in self.rejected."""

    def __init__(self, surface, sample, water=None, water_kind=None, treeline=None, flat=None, bans=None, habitat=None):
        """water, water_kind: local_water() (or a river level alone, kind river); None = the plane floor only."""
        self.surface, self.sample, self.water = surface, sample, water
        self.water_kind = water_kind if water_kind is not None or water is None else np.where(np.isfinite(water), 1, 0).astype(np.uint8)
        self.dry_by_water = {}
        self.treeline, self.flat, self.bans = treeline, flat, dict(bans or {})
        self.habitat = dict(habitat or {})
        self.rejected = {}

    def footprint(self, tx, ty):
        """(lowest ground, highest local water level, its WATER_KINDS index) over the 3 x 3 footprint of tile
        positions (NaN = no water never wins; -inf and kind 0 where no water fact covers the footprint)."""
        lo = np.full(len(tx), np.inf); lv = np.full(len(tx), -np.inf); kd = np.zeros(len(tx), np.uint8)
        wl = self.water
        for ox in (-FOOT, 0, FOOT):
            for oz in (-FOOT, 0, FOOT):
                lo = np.minimum(lo, self.sample(self.surface, tx + ox, ty + oz))
                if wl is not None:
                    iy = np.clip((ty + oz).astype(np.int64), 0, wl.shape[0] - 1); ix = np.clip((tx + ox).astype(np.int64), 0, wl.shape[1] - 1)
                    v = wl[iy, ix].astype(np.float64)
                    win = v > lv
                    lv = np.where(win, v, lv); kd = np.where(win, self.water_kind[iy, ix], kd)
        return lo, lv, kd

    def dry(self, tx, ty, count=False):
        """dry_enough at tile positions as stored; count: add rejections to dry_by_water (the water kind that
        sets the level, 'plane' where the WET_LEVEL floor does)."""
        lo, lv, kd = self.footprint(tx, ty)
        ok = dry_enough(lo, lv)
        if count and (~ok).any():
            by = np.where(lv + BANK_TREE_MARGIN >= WET_LEVEL, kd, 0)[~ok]
            for k, n in zip(*np.unique(by, return_counts=True)):
                key = WATER_KINDS[k] if k else 'plane'
                self.dry_by_water[key] = self.dry_by_water.get(key, 0) + int(n)
        return ok

    def rules(self, tx, ty, species, count=False):
        """{rule: bool reject per row} at the stored positions."""
        tx = np.atleast_1d(np.asarray(tx, np.float64)); ty = np.atleast_1d(np.asarray(ty, np.float64))
        n = len(tx)
        sp = np.full(n, species, object) if isinstance(species, str) else np.asarray(species, object)
        x, z, sx, sz = stored(tx, ty)
        canvas = ~native_many(x, z)
        q, r = cw.world_hex(x, z); q, r = np.asarray(q, np.int64).reshape(-1), np.asarray(r, np.int64).reshape(-1)
        out = {'dry': ~self.dry(sx, sz, count)}
        uniq, inv = np.unique(sp.astype(str), return_inverse=True) if n else (np.zeros(0, str), np.zeros(0, np.int64))
        pre = lambda p: np.array([u.startswith(p) for u in uniq.tolist()], bool)[inv] if n else np.zeros(0, bool)
        if self.treeline is not None:
            out['treeline'] = canvas & pre(TREE_PREFIXES) & _hex_lookup(self.treeline, q, r)
        if self.flat is not None:
            out['flat_rock'] = canvas & pre(ROCK_PREFIX) & _hex_lookup(self.flat, q, r)
        if self.bans:
            ban = np.zeros(n, bool); taken = np.zeros(n, bool)
            for family, m in self.bans.items():   # first matching family wins (as drop_autumn_shrubs)
                f = pre(family) & ~taken; taken |= f
                if m is not None:
                    ban |= f & _hex_lookup(m, q, r)
            out['species_ban'] = canvas & ban
        for prefix, allowed in self.habitat.items():
            out[f'habitat:{prefix}'] = canvas & pre(prefix) & ~_hex_lookup(allowed, q, r)
        return out, sp

    def accept(self, tx, ty, species, count=True):
        rules, sp = self.rules(tx, ty, species, count)
        bad = np.zeros(len(sp), bool)
        for name, m in rules.items():
            if count and m.any():
                c = self.rejected.setdefault(name, {})
                for s in sp[m & ~bad]:
                    c[s] = c.get(s, 0) + 1
            bad |= m
        return ~bad


# ------------------------------------------------------------------ the WET TREES post-filter (legacy)
def drop_wet_trees(groups, surface, river_level, sample):
    """--- L2048 WET TREES --- post-plant filter, kept running behind accept() until it counts 0 (design 3.4).
    Canvas rows whose footprint ground lies under the water plane + margin, or under the local river level +
    the same margin, are dropped; native rows stay. Returns (below plane, below river level)."""
    wet_dropped = river_dropped = 0
    for g in groups:
        if not g['rows']:
            continue
        p = positions(g['rows']); _tx, _tz = p[:, 0] / cw.DX * 2, p[:, 2] / cw.DZ * 2
        _lo = np.full(len(p), np.inf); _lv = np.full(len(p), -np.inf)
        for _ox in (-.35, 0, .35):
            for _oz in (-.35, 0, .35):
                _lo = np.minimum(_lo, sample(surface, _tx + _ox, _tz + _oz))
                if river_level is not None:
                    _rl = river_level[np.clip((_tz + _oz).astype(np.int64), 0, river_level.shape[0] - 1), np.clip((_tx + _ox).astype(np.int64), 0, river_level.shape[1] - 1)]
                    _lv = np.fmax(_lv, _rl.astype(np.float64))   # NaN (off the band) never wins
        _canvas = ~native_many(p[:, 0], p[:, 2])
        _plane = (_lo < WET_LEVEL) & _canvas
        _river = ~_plane & _canvas & (_lo < _lv + WET_LEVEL - LAKE_PLANE)   # same margin over the river level as over the plane
        _wet = _plane | _river; wet_dropped += int(_plane.sum()); river_dropped += int(_river.sum())
        if _wet.any():
            g['rows'] = [_r for _r, _w in zip(g['rows'], _wet) if not _w]; g['added'] = max(0, g['added'] - int(_wet.sum()))
    return wet_dropped, river_dropped


def pack_row(template, tx, ty, y):
    """A row from its template at tile position (tx, ty), ground height y (world x = tx DX / 2)."""
    row = bytearray(template)
    struct.pack_into('<3f', row, 0, tx * cw.DX / 2, y, ty * cw.DZ / 2)
    return bytes(row)

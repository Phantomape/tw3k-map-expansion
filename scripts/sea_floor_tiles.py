"""Vanilla sea floors (generic_sea) under the sea cells that polish re-floors, instead of land floors.

The new canvas sea and the edited native sea (the drowned vanilla coasts: Lin Yi / Rinan, Kimhae,
...) were floored with the land generic tiles (generic_16x16 / 2x2 / 1x1); the live candidates
v14-v16 showed vanilla's drowned shoreline as a long pale strip in the sea off Lin Yi (象林).
What vanilla does (3k_main_map cache, 3k_dlc07_main_map tile lists):
  * every sea cell is floored with generic_sea: 8x8 in open water (12,120 cache owners), 2x2
    (12,445) and 1x1 (2,032) near shores (4x4 too, not used here); no land generic tile sits on
    an all-sea footprint. The generic_sea meshes differ from the land ones in their mesh flags
    (rigid_model_v2 LOD/material bytes: 0x67 / 0x33 vs 0xffffff / 0xb3);
  * cache layer 0 owner: flags 0x700 | one-hot quarter turn (all four turns about equally often),
    extra 0, the owner cell at the turn's corner of the block (0x10: (0, s-1), 0x20: (s-1, s-1),
    0x40: (s-1, 0), 0x80: (0, 0); every 8x8 and 2x2 owner of 3k_main_map), children -owner-1
    with flags 0x710, the block's x / y and lo = hi = 0;
  * the list record has the owner's anchor, flags and climate (all 26,597 cache owners match a
    record); local tile_list.bin keeps 0x7?0, global_map/tile_list.bin the turn only (0x?0);
  * ranges: lo = hi = 0.14639506 (the sea plane) on 97% of the 8x8 owners over the -2.44 floor,
    i.e. the LF extrema over the footprint plus two cells, raised to that plane.
Here: the floors polish lays (and repair_native_placements lists) on open-sea cells (map.hex sea
hexes outside river and lake regions) become generic_sea floors of the same footprint: a 16x16 land
floor of all-sea cells four 8x8, one over sea and land cells 2x2 groups (sea or land each), an
all-sea 2x2 a sea 2x2, a sea cell's 1x1 a sea 1x1. The same rule (split) runs in both scripts.
The strip itself was holes: native generic_sea records touching the edited sea were kept by
repair_native_placements and culled by clear_obsolete_terrain_overlays, so the part of their
footprint outside the edit had no fine tile (38,458 sea cells in v16, all inside the native map;
vanilla 134). repair now replaces them like native land floors over their whole footprint (as
polish's cache does) and clear_obsolete_terrain_overlays treats generic_sea as a floor, not an overlay.
"""
import re

import numpy as np

B = chr(92)
SEA_TILE = {s: f'terrain{B}tiles{B}campaign{B}generic_sea{B}{s}x{s}{B}' for s in (8, 2, 1)}
SEA_RANGE_FLOOR = 0.14639505743980408    # vanilla generic_sea lo/hi over the open-sea floor (float32)
TURNS = (0x10, 0x20, 0x40, 0x80)
CHILD_FLAGS = 0x710
WATER_FILE = 'analysis/sea-floor-water.npy'                 # tile cells polish floors with generic_sea
CACHE_ANCHORS_FILE = 'analysis/sea-floor-cache-anchors.npy'  # uint8 size at each sea floor anchor (cache)
RIVER_OR_LAKE = ('_riv_sea_', '3k_main_sea_lake')
SEA_RECORD = re.compile(re.escape(B) + r'generic_sea' + re.escape(B) + r'(\d+)x\d+' + re.escape(B))   # N of a generic_sea NxN path
# Floor tiles (land generic and the generic_sea seabed), on '/'-separated paths: floors are not special
# overlays (clear_obsolete_terrain_overlays.py never culls them; audit_native_overlay_preservation.py takes
# a vanilla generic_sea floor re-floored inside polish's cache replacement as a floor, not a lost overlay).
FLOOR_FAMILY = re.compile(r'/(generic|generic_sea)/')


def open_sea_hexes(cells, names):
    """Sea hexes (map.hex kind 1) outside river channels and lakes (bool, hex grid)."""
    ids = (cells[:, :, 1].astype(int) << 5) + (cells[:, :, 0].astype(int) >> 3) - 1
    keep = np.array([not any(k in n for k in RIVER_OR_LAKE) for n in names] + [False])
    return ((cells[:, :, 0] & 3) == 1) & keep[np.where(ids >= 0, ids, len(names))]


def water_cells(cells, names, hy, hx):
    """Open-sea tile cells: the hex of each cell (hy, hx from campaign_world.tile_hex) is open sea."""
    return open_sea_hexes(cells, names)[hy, hx]


def turn(x, y):
    """Deterministic quarter turn (one-hot 0x10..0x80) per anchor: vanilla mixes all four."""
    h = (np.asarray(x, np.uint32) * np.uint32(0x9E3779B1)) ^ (np.asarray(y, np.uint32) * np.uint32(0x85EBCA77))
    return np.array(TURNS, np.int64)[((h >> np.uint32(13)) & np.uint32(3)).astype(np.int64)]


def owner_offset(size, t):
    """(dx, dy) of the owner cell in a size x size block for the turns t (vanilla pairing)."""
    t = np.asarray(t)
    s = size - 1
    dx = np.where((t == 0x20) | (t == 0x40), s, 0)
    dy = np.where((t == 0x10) | (t == 0x20), s, 0)
    return dx, dy


def climate_at(size, x, y, biome):
    """Climate of a floor as the land floors take it: the cell at the footprint centre."""
    TH, TW = biome.shape
    half = np.asarray(size) // 2
    return biome[np.minimum(np.asarray(y) + half, TH - 1), np.minimum(np.asarray(x) + half, TW - 1)]


def _all(mask, size, x, y):
    """mask holds every cell of the size x size blocks at (x, y) (blocks reaching off the map: False)."""
    x = np.asarray(x, np.int64); y = np.asarray(y, np.int64)
    TH, TW = mask.shape
    ok = (x + size <= TW) & (y + size <= TH)
    out = ok.copy()
    for dy in range(size):
        for dx in range(size):
            out[ok] &= mask[y[ok] + dy, x[ok] + dx]
    return out


def _any(mask, size, x, y):
    x = np.asarray(x, np.int64); y = np.asarray(y, np.int64)
    TH, TW = mask.shape
    out = np.zeros(len(x), bool)
    for dy in range(size):
        for dx in range(size):
            yy, xx = y + dy, x + dx
            ok = (yy < TH) & (xx < TW)
            out[ok] |= mask[yy[ok], xx[ok]]
    return out


def split(size, x, y, water):
    """Land floors (size 16 / 2 / 1 at anchors x, y) -> (land size, x, y), (sea size, x, y).

    16: all cells sea -> four sea 8x8; some -> its 64 2x2 groups, each sea (all cells sea) or land 2x2;
    none -> kept. 2: all sea -> sea 2x2. 1: sea -> sea 1x1. Land floors keep their input order;
    the land 2x2 groups of a split 16 follow at the end (callers re-sort where order matters)."""
    size = np.asarray(size, np.int64); x = np.asarray(x, np.int64); y = np.asarray(y, np.int64)
    land_s, land_x, land_y, sea_s, sea_x, sea_y = [], [], [], [], [], []
    keep = np.ones(len(size), bool)
    g = np.flatnonzero(size == 16)
    full = _all(water, 16, x[g], y[g]); some = ~full & _any(water, 16, x[g], y[g])
    keep[g[full | some]] = False
    for dy in (0, 8):
        for dx in (0, 8):
            sea_s.append(np.full(int(full.sum()), 8)); sea_x.append(x[g[full]] + dx); sea_y.append(y[g[full]] + dy)
    gx = (x[g[some]][:, None] + np.tile(np.arange(0, 16, 2), 8)[None]).ravel()
    gy = (y[g[some]][:, None] + np.repeat(np.arange(0, 16, 2), 8)[None]).ravel()
    gs = _all(water, 2, gx, gy)
    sea_s.append(np.full(int(gs.sum()), 2)); sea_x.append(gx[gs]); sea_y.append(gy[gs])
    t = np.flatnonzero(size == 2); ts = _all(water, 2, x[t], y[t]); keep[t[ts]] = False
    sea_s.append(np.full(int(ts.sum()), 2)); sea_x.append(x[t[ts]]); sea_y.append(y[t[ts]])
    # A 2x2 cut by the map's top row (TH is odd) is listed as 1x1 cells (floor_relay): all-sea ones
    # become sea 1x1 cells, so cache and lists agree.
    TH, TW = water.shape
    cut = t[~ts & ((y[t] + 2 > TH) | (x[t] + 2 > TW))]
    for dy in (0, 1):
        for dx in (0, 1):
            ok = (y[cut] + dy < TH) & (x[cut] + dx < TW)
            cut = cut[~ok | water[np.minimum(y[cut] + dy, TH - 1), np.minimum(x[cut] + dx, TW - 1)]]
    keep[cut] = False
    for dy in (0, 1):
        for dx in (0, 1):
            ok = (y[cut] + dy < TH) & (x[cut] + dx < TW)
            sea_s.append(np.ones(int(ok.sum()), np.int64)); sea_x.append(x[cut][ok] + dx); sea_y.append(y[cut][ok] + dy)
    o = np.flatnonzero(size == 1); os_ = water[y[o], x[o]]; keep[o[os_]] = False
    sea_s.append(np.full(int(os_.sum()), 1)); sea_x.append(x[o[os_]]); sea_y.append(y[o[os_]])
    land = (np.concatenate((size[keep], np.full(int((~gs).sum()), 2))), np.concatenate((x[keep], gx[~gs])),
            np.concatenate((y[keep], gy[~gs])))
    sea = tuple(np.concatenate(a).astype(np.int64) for a in (sea_s, sea_x, sea_y))
    return land, sea


def anchors(shape, size, x, y):
    """uint8 grid with each sea floor's size at its anchor cell (0 elsewhere)."""
    a = np.zeros(shape, np.uint8)
    a[np.asarray(y, np.int64), np.asarray(x, np.int64)] = np.asarray(size, np.uint8)
    return a


def lf_extremes(lf):
    """Per tile cell LF min / max (uint16) of the 4x4 texels of a south-up LF (4*TH x 4*TW)."""
    h, w = lf.shape
    b = np.ascontiguousarray(lf).reshape(h // 4, 4, w // 4, 4)
    return b.min(axis=(1, 3)), b.max(axis=(1, 3))


def _window(a, lo, hi, fn, fill):
    """fn over cells [i+lo, i+hi) on both axes (clipped to the map), as polish's _win."""
    out = a
    for axis in (0, 1):
        n = a.shape[axis]; acc = None
        for k in range(lo, hi):
            s = np.full(out.shape, fill, a.dtype)
            if k >= 0:
                if k < n:
                    (s[:n - k] if axis == 0 else s[:, :n - k])[...] = (out[k:] if axis == 0 else out[:, k:])
            elif -k < n:
                (s[-k:] if axis == 0 else s[:, -k:])[...] = (out[:n + k] if axis == 0 else out[:, :n + k])
            acc = s if acc is None else fn(acc, s)
        out = acc
    return out


def ranges(tmin, tmax, size, x, y):
    """lo, hi (LF units 0..1) of sea floors: LF extrema over the footprint plus two cells, raised to
    the vanilla sea plane SEA_RANGE_FLOOR."""
    size = np.asarray(size, np.int64); x = np.asarray(x, np.int64); y = np.asarray(y, np.int64)
    lo = np.zeros(len(size)); hi = np.zeros(len(size))
    for s in np.unique(size).tolist():
        i = np.flatnonzero(size == s)
        lo[i] = _window(tmin, -2, s + 2, np.minimum, 65535)[y[i], x[i]] / 65535
        hi[i] = _window(tmax, -2, s + 2, np.maximum, 0)[y[i], x[i]] / 65535
    return np.maximum(lo, SEA_RANGE_FLOOR), np.maximum(hi, SEA_RANGE_FLOOR)


def stamp(layer0, size, x, y, climate, value, lo=None, hi=None):
    """Write sea floor owners into cache layer 0 (canvas frame). value: {size: one-based cache path
    value of SEA_TILE[size]}. lo/hi default 0 (set per variant later). Returns the turns."""
    TH, TW = layer0.shape
    size = np.asarray(size, np.int64); x = np.asarray(x, np.int64); y = np.asarray(y, np.int64)
    t = turn(x, y)
    lo = np.zeros(len(size)) if lo is None else lo; hi = np.zeros(len(size)) if hi is None else hi
    for s in np.unique(size).tolist():
        i = np.flatnonzero(size == s)
        ox, oy = owner_offset(s, t[i]); px, py = x[i] + ox, y[i] + oy
        parent = py * TW + px
        for dy in range(s):
            for dx in range(s):
                yy, xx = y[i] + dy, x[i] + dx
                c = layer0[yy, xx]
                c['path'] = -parent - 1; c['flags'] = CHILD_FLAGS; c['x'] = x[i]; c['y'] = y[i]
                c['climate'] = climate[i]; c['extra'] = 0; c['lo'] = 0; c['hi'] = 0
                layer0[yy, xx] = c
        c = layer0[py, px]
        c['path'] = value[s]; c['flags'] = 0x700 | t[i]; c['lo'] = lo[i]; c['hi'] = hi[i]
        layer0[py, px] = c
    return t


def owner_cells(size, x, y):
    """(py, px) of the owner cells of the sea floors (turn corner)."""
    size = np.asarray(size, np.int64); x = np.asarray(x, np.int64); y = np.asarray(y, np.int64)
    t = turn(x, y); px = x.copy(); py = y.copy()
    for s in np.unique(size).tolist():
        i = np.flatnonzero(size == s); ox, oy = owner_offset(s, t[i]); px[i] += ox; py[i] += oy
    return py, px

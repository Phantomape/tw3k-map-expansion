"""Vanilla mountain fine tiles on the high ground of the real-terrain canvas, for the tile cache
section of polish_campaign_terrain.py (same mechanism as salt_pools.py).

New land took generic floor tiles everywhere, so its mountains had none of vanilla's rock faces,
scree and cliff detail (offline views: "everything yellow and green", 2048bq). Vanilla paints its
mountains with families of fine tiles (mountains_cold, mountains_subtropical, mountains_temperate,
mountains_yellow; 14% of 3k_main_map land cells). This copies whole vanilla donor windows of one
family onto WIN x WIN windows of replaced canvas ground whose real terrain is mountain:
  * mountain cells: real-terrain weight > .5 and (real elevation >= HIGH_M metres, or slope of the
    real relief >= STEEP world units per tile cell over a SLOPE_RADIUS-cell box);
  * family per window from its real biome mix and elevation (FAMILY_RULES);
  * a window is used when >= MIN_SHARE of it is mountain and all of it is replaced physical land
    outside town plots, salt pans, lakes and the cells other stamps already took;
  * owners are copied only when their whole footprint lies in the donor window and they are a
    mountain-family or generic floor tile (roads, rivers, coast and cliffs of the donor stay home);
    lo/hi are recomputed from the target LF downstream, exactly as for the salt pans.
Cache path values are one-based (tail[p-1]); see salt_pools.py for the record layout.
"""
import json

import numpy as np

import salt_pools
from native_campaign_climate import box_blur

WIN = 48                     # tile cells (24 hexes): donor and target window side. Vanilla mountain
                             # ranges are mostly huge tiles (large_square 43x43, large_diamond 968 cells,
                             # large_triangle 484); a 32 window held only their 1x1/2x1 fillers
MIN_SHARE = .5               # share of a target window that must be mountain
HIGH_M = 4800.               # metres: only the high peaks count by elevation alone (the plateau is flat)
STEEP = .15                  # world units of relief per tile cell (box mean of |gradient|; ~top 13% of real land, vanilla paints 14%)
SLOPE_RADIUS = 3
STRIDE = 8
# Donor windows (r0, c0) in 3k_main_map tile cells, WIN square: the windows whose owners lying wholly
# inside cover the most cells (scratch survey 2026-10-23, stride 4: family share .49-.90, with generic
# floor .78-.92).
DONORS = {
    'mountains_cold': [(608, 220), (672, 164), (544, 48), (1232, 1548)],
    'mountains_subtropical': [(492, 588), (360, 460), (440, 1116), (488, 876)],
    'mountains_temperate': [(1000, 632), (672, 980), (1040, 688), (1056, 796)],
    'mountains_yellow': [(568, 572), (672, 700), (688, 596), (728, 648)],
}
COLD = ('alpine', 'tundra', 'snow', 'conifer', 'taiga')
TROPICAL = ('tropical', 'tropical_dry', 'savanna', 'mangrove', 'paddy')
TEMPERATE = ('forest', 'temperate')
# --- TILE PROPS --- Some vanilla mountain tiles carry prop models in their bmd_data.bin: the Guizhou stone
# forest (mountains_yellow: yellow_mt_rock_1-5 with unique_tree_1-4), South China karst towers
# (mountains_subtropical: subtropic_mountain_*, unique_tree_*, castanopsis), central China crags
# (mountains_temperate: temperate_mt_1-5) and snow peaks (mountains_cold). Stamped by family alone, the
# karst and the stone forest stood in the Transbaikal steppe, the Tian Shan foothills above Turpan, Iran and
# the Himalayan foothills (live O27 Dingling). PROP_PIECES: the pieces whose bmd_data references a model
# (vanilla packs, survey 2026-10-25). PROP_HABITAT: the Ecoregions2017 ids under vanilla's own prop owners
# of the family (3k_main_map tile_list at the native position, >= 5 owners; Yunnan Plateau 643 holds the
# real Shilin; Northern Vietnam lowland rain forests 260 added: the karst of north Vietnam). Elsewhere a
# window keeps the family's bare pieces (ground and cliffs, no models). Cold peaks: everywhere, as before.
PROP_PIECES = {
    'mountains_subtropical': ('11x11_diamond', '11x11_square', '11x6_triangle_a', '11x6_triangle_b', '11x6_triangle_c',
                              '11x6_triangle_e', '11x6_triangle_f', '11x6_triangle_g', '16x6_skew', '16x6_skew_mirror',
                              '6x4_a1', '6x4_a2', '6x4_a3', '6x4_a4', '6x4_b1', '6x4_b2', '6x4_b3', 'large_diamond',
                              'large_skew', 'large_skew_mirror', 'large_square', 'large_triangle', 'small_diamond',
                              'small_rectangle', 'small_skew', 'small_skew_mirror', 'small_square', 'small_triangle'),
    'mountains_temperate': ('11x11_diamond', '11x11_square', '11x6_triangle', '16x6_skew', '16x6_skew_mirror', '6x4_a',
                            '6x4_b', 'large_diamond', 'large_skew', 'large_skew_mirror', 'large_square', 'large_triangle',
                            'small_diamond', 'small_rectangle', 'small_skew', 'small_skew_mirror', 'small_square',
                            'small_triangle'),
    'mountains_yellow': ('11x11_diamond', '11x6_triangle_a', '11x6_triangle_b', '16x6_skew', '16x6_skew_mirror', '4x3',
                         '4x3_mirror', '4x4a', '4x4b', '6x4_a1', '6x4_a2', '6x4_b1', '6x4_b2', 'small_diamond',
                         'small_triangle'),
}
PROP_HABITAT = {
    # Jian Nan 236 (444 owners), South China-Vietnam 268 (99), Changjiang Plain 657 (23), Guizhou Plateau 642 (11),
    # Yunnan Plateau 643 (1: Shilin), Northern Vietnam lowland 260 (added)
    'mountains_subtropical': frozenset((236, 268, 657, 642, 643, 260)),
    # Guizhou Plateau 642 (181), Jian Nan 236 (30), Daba Mountains 659 (8)
    'mountains_yellow': frozenset((642, 236, 659)),
    # mountains_temperate (temperate_mt_1-5 crags) has no habitat: owner decision 2026-10-25, its prop tiles stay
    # everywhere as before (vanilla habitat would be Loess 653, Huang He 667, Changjiang 657, Qin Ling 677, Daba 659,
    # Jian Nan 236, Ordos 761). Only the stone forest and the karst towers are restricted.
}
LAST_STATS = {}          # stamp_all: prop owners kept / dropped per family (polish report)
COVER_FILE = 'analysis/mountain-tile-cover.npy'
REPORT_FILE = 'analysis/mountain-tiles.json'
BS = chr(92)


def family_of(name):
    parts = name.rstrip(BS).split(BS)
    return parts[3] if len(parts) > 3 else ''


def owner_index(native0):
    """(owner cell index per native cell, active cells (flags != 16), active cells per owner): built once
    per native grid for donor_owners (one whole-grid scan per owner took ~15 ms, seconds per stamp_all)."""
    path = native0['path'].astype(np.int64)
    owner_all = np.where(path >= 0, np.arange(path.size).reshape(path.shape), -path - 1)
    active = native0['flags'] != 16
    return owner_all, active, np.bincount(owner_all[active], minlength=path.size)


def donor_owners(native0, tail_paths, origin, family, index=None):
    """Owners fully inside the donor window that are `family` or generic floor tiles (salt_pools layout).
    `index` is owner_index(native0) (built here when not given)."""
    r0, c0 = origin
    r1, c1 = r0 + WIN, c0 + WIN
    oh, ow = native0.shape
    owner_all, active, count = owner_index(native0) if index is None else index
    owner = owner_all[r0:r1, c0:c1]
    act = active[r0:r1, c0:c1]
    owners = []
    flat = native0.ravel()
    for o, n_in in zip(*np.unique(owner[act], return_counts=True)):
        rec = flat[o]
        v = int(rec['path'])
        if not 0 < v <= len(tail_paths):
            continue
        name = tail_paths[v - 1]
        fam = family_of(name)
        if not (fam == family or ('\\generic\\' in name and 'generic_sea' not in name)):
            continue
        if count[o] != n_in:            # an active cell of this owner lies outside the window
            continue
        pr, pc = divmod(int(o), ow)
        owners.append(dict(path=v, flags=int(rec['flags']), extra=int(rec['extra']), parent=(pr - r0, pc - c0),
                           cells=[(int(a), int(b)) for a, b in np.argwhere((owner == o) & act)], x=int(rec['x']) - c0, y=int(rec['y']) - r0,
                           family=fam))
    return owners


def mountain_mask(real, shape):
    """(mountain cells, family index per cell) on the tile grid from analysis/west-real-terrain.npz."""
    TH, TW = shape
    x0, x1, y0, y1 = (int(v) for v in real['box'])
    w = np.zeros(shape, np.float32); rel = np.zeros(shape, np.float32); elev = np.zeros(shape, np.float32)
    bio = np.full(shape, 255, np.uint8)
    w[y0:y1, x0:x1] = real['weight']; rel[y0:y1, x0:x1] = real['relief']; elev[y0:y1, x0:x1] = real['elevation']
    bio[y0:y1, x0:x1] = real['biome']
    gy, gx = np.gradient(rel)
    # Edge-clamped (2 SLOPE_RADIUS + 1)^2 box mean, accumulated in float64: the float32 2-D cumsum this
    # replaced ran to ~1.3e6 and was off by up to .012 against STEEP (10.7k cells on the wrong side).
    box = box_blur(np.hypot(gx, gy), SLOPE_RADIUS)
    mountain = (w > .5) & ((elev >= HIGH_M) | (box >= STEEP))
    names = [str(n) for n in real['biome_names']]
    idx = {n: i for i, n in enumerate(names)}
    def isin(group):
        return np.isin(bio, [idx[n] for n in group if n in idx])
    fam = np.full(shape, 3, np.uint8)                       # 3 yellow: dry mountains (desert/steppe/grass)
    fam[isin(TEMPERATE)] = 2
    fam[isin(TROPICAL)] = 1
    fam[isin(COLD) | (elev >= HIGH_M)] = 0
    return mountain, fam


FAMILIES = ('mountains_cold', 'mountains_subtropical', 'mountains_temperate', 'mountains_yellow')


def plan(mountain, fam, allowed, taken=None):
    """Greedy non-overlapping WIN windows: [(R, C, family)]."""
    TH, TW = mountain.shape
    ok = allowed & ~(taken if taken is not None else np.zeros_like(allowed))
    sm = np.pad(mountain.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sa = np.pad((~ok).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    def boxsum(s, r, c):
        return s[r + WIN, c + WIN] - s[r, c + WIN] - s[r + WIN, c] + s[r, c]
    cand = []
    for r in range(0, TH - WIN, STRIDE):
        for c in range(0, TW - WIN, STRIDE):
            if boxsum(sa, r, c):
                continue
            share = boxsum(sm, r, c) / (WIN * WIN)
            if share >= MIN_SHARE:
                f = np.bincount(fam[r:r + WIN, c:c + WIN][mountain[r:r + WIN, c:c + WIN]], minlength=4)
                cand.append((share, r, c, int(f.argmax())))
    cand.sort(reverse=True)
    used = np.zeros((TH, TW), bool)
    picked = []
    for share, r, c, f in cand:
        if used[r:r + WIN, c:c + WIN].any():
            continue
        used[r:r + WIN, c:c + WIN] = True
        picked.append((r, c, FAMILIES[f]))
    return picked


def tile_eco(real, shape):
    """--- TILE PROPS --- Ecoregions2017 id per tile cell (apply_climate.eco_ids on the real lon/lat)."""
    import apply_climate
    TH, TW = shape
    x0, x1, y0, y1 = (int(v) for v in real['box'])
    eco = np.zeros(shape, np.int32)
    eco[y0:y1, x0:x1] = apply_climate.eco_ids(real['lon'], real['lat'])
    return eco


def owner_props(o, tail_paths):
    """--- TILE PROPS --- True when the owner's tile carries vanilla prop models (PROP_PIECES)."""
    parts = tail_paths[o['path'] - 1].rstrip(BS).split(BS)
    return len(parts) > 4 and parts[4].lower() in PROP_PIECES.get(parts[3].lower(), ())


def stamp_all(layer0, native0, tail_paths, picked, biome, TW, seed=190, eco=None):
    """Stamp each window with a donor of its family (cycled); returns (cover mask list, placements).
    With `eco` (tile_eco), TILE PROPS: owners whose tile carries vanilla props are stamped only where the
    ecoregion at their parent cell is in PROP_HABITAT for their family (the rng draws are unchanged)."""
    rng = np.random.default_rng(seed)
    index = owner_index(native0)
    donors = {f: [donor_owners(native0, tail_paths, o, f, index) for o in DONORS[f]] for f in FAMILIES}
    covered, placed = [], []
    LAST_STATS.clear()
    for k, (R, C, f) in enumerate(picked):
        owners = donors[f][int(rng.integers(len(donors[f])))]
        if eco is not None and f in PROP_HABITAT:   # --- TILE PROPS ---
            keep = [o for o in owners if not owner_props(o, tail_paths)
                    or int(eco[R + o['parent'][0], C + o['parent'][1]]) in PROP_HABITAT[f]]
            st = LAST_STATS.setdefault(f, {'prop_owners_kept': 0, 'prop_owners_dropped': 0})
            n_props = sum(owner_props(o, tail_paths) for o in owners)
            st['prop_owners_dropped'] += len(owners) - len(keep)
            st['prop_owners_kept'] += n_props - (len(owners) - len(keep))
            owners = keep
        # Climate per owner from the tile climate at its parent cell. One climate for the whole
        # 24-hex window (its centre) drew a hard climate line where a window met the native border
        # (audit_native_edges climate_cut at hex 887,674 and 887,722, r14 pipeline).
        for o in owners:
            climate = int(biome[R + o['parent'][0], C + o['parent'][1]])
            cov, pl = salt_pools.stamp(layer0, [o], (R, C), climate, TW)
            covered += cov
            placed += pl
    return covered, placed

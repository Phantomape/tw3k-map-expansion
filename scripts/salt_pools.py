"""Vanilla salt-pan tiles on real salt flats (Natural Earth playas), for the tile cache section of
polish_campaign_terrain.py.

3k_main_map has six salt pans: mosaics of salt_pools tiles (8x8, 9x9, 4x6, 2x2) interlocked with
generic floor tiles, all on cache layer 0. 3k_dlc07_main_map has none. This copies one vanilla pan
(CLUSTER: few owners cross its window) onto every salt-flat window of replaced (rebuilt) ground:
  * the cache of the donor variant (3k_main_map, whose tail path table the pipeline already uses)
    stores path value p for tail[p-1] (one-based);
  * an owner is a parent cell (path >= 0) and its children (-parent-1); the parent carries rotation
    flags and lo/hi, children copy x/y/climate;
  * the tile_list record of an owner is (1, list path index, climate, x, y, flags | extra << 2, lo, hi)
    with x/y the footprint's minimum column/row, exactly the cache parent's x/y.
Only owners whose whole footprint lies in the window and that are salt or generic floor are copied;
roads, coast and river tiles of the donor pan are dropped (their cells get floors as usual).
"""
import numpy as np

SALT_NAMES = ('8x8_salt_pool_1', '9x9_salt_pool_2', '4x6_salt_pool_3', '2x2_salt_pool_4')
CLUSTER = (1102, 1115, 1092, 1112)      # donor window rows r0:r1, cols c0:c1 (3k_main_map tile cells)
MIN_SALT_SHARE = .5                     # share of a window that must be salt flat
# The pans are rigid flat basin meshes (rim at 0, pond floors to about -90 units) authored on level
# ground: every vanilla pan footprint spans <= .035 of normalised LF. On the plateau the flat mesh cut
# through the slope (Yangtong .18, Xihai .18 per footprint): black underside sheets and torn white rim
# slivers (live 2048k). polish_campaign_terrain.py plans the windows before its height projection
# and levels the ground under them (level_field); its cache stage stamps the same windows.
LEVEL_PAD = 3            # tile cells around a window held at the pan level (the projection's
                         # near-town blur and the LF frame offset stay off the footprint)
LEVEL_APRON = 24         # tile cells of smoothstep from the pan level back to the ground (its steepest
                         # step stays under the projection grade, 1/3 world per cell, on a .12 slope)
LEVEL_MARGIN = .05       # world units above the footprint median
PLAYA_MAX_RELIEF = 1.2   # world units: a playa group (not a gameplay site) on rougher ground is skipped
EDIT_ALPHA = .1          # edit alpha a window needs (inside the native map only Xihai's town edit,
                         # whose >= .5 disc cannot hold a window beside the town plot); the levelling
                         # weight scales by alpha / this, so nothing changes where alpha is 0
SITE_RING_COST = .05     # world units of relief range per tile cell of ring gap (site window choice)
SALT_RGB = (205, 206, 172)   # vanilla sea_colour.dds under every pan: the pond floors take the water colour
WINDOWS_FILE = 'analysis/salt-pan-windows.json'


def donor_cluster(native0, tail_paths):
    """Owners of the donor window: list of dict(path, flags, extra, parent (dr, dc), cells [(dr, dc)],
    x, y relative to the window) for salt and generic floor tiles fully inside it."""
    r0, r1, c0, c1 = CLUSTER
    oh, ow = native0.shape
    win = native0[r0:r1, c0:c1]
    owner = np.where(win['path'] >= 0, (np.arange(r0, r1)[:, None] * ow + np.arange(c0, c1)[None, :]), -win['path'].astype(np.int64) - 1)
    owners = []
    flat = native0.ravel()
    # children (path -o-1) per owner on the whole grid, counted once instead of one grid scan per owner
    path = native0['path']
    kids = np.bincount(-path[path < 0].astype(np.int64) - 1, minlength=path.size)
    for o in np.unique(owner[win['flags'] != 16]):
        pr, pc = divmod(int(o), ow)
        rec = flat[o]
        name = tail_paths[int(rec['path']) - 1]
        if not ('salt_pools' in name or '\\generic\\' in name) or 'generic_sea' in name:
            continue
        rows, cols = np.nonzero(owner == o)
        # whole footprint inside the window: no cell of this owner outside it
        if kids[o] != np.count_nonzero((owner == o) & (win['path'] < 0)):
            continue
        if not (r0 <= pr < r1 and c0 <= pc < c1):
            continue
        owners.append(dict(path=int(rec['path']), flags=int(rec['flags']), extra=int(rec['extra']), name=name,
                           parent=(pr - r0, pc - c0), cells=list(zip((rows).tolist(), (cols).tolist())),
                           x=int(rec['x']) - c0, y=int(rec['y']) - r0))
    return owners, (r1 - r0, c1 - c0)


def windows(salt, replaced, land, shape):
    """Top-left (row, col) of non-overlapping window placements on the tile grid."""
    wh, ww = shape
    TH, TW = salt.shape
    out = []
    for r in range(0, TH - wh, wh):
        for c in range(0, TW - ww, ww):
            s = salt[r:r + wh, c:c + ww]
            if not s.any():
                continue
            if s.mean() >= MIN_SALT_SHARE and replaced[r:r + wh, c:c + ww].all() and land[r:r + wh, c:c + ww].all():
                out.append((r, c))
    return out


def site_windows(sites, replaced, land, shape, taken=(), relief=None):
    """One window per gameplay salt site (tile (x, y) of a resource town), on a ring offset
    around the town that is wholly replaced land and clear of the town footprint and of
    windows already taken. Playas rarely sit under the resource towns (yangtong 229 hexes,
    shanshan 44 hexes from the nearest Natural Earth salt flat). Without `relief` the first
    free position (nearest ring) wins; with it, the free position of least relief range plus
    SITE_RING_COST per cell of ring gap, so levelling cuts the least ground."""
    wh, ww = shape
    TH, TW = replaced.shape
    out, boxes = [], [(r, c, r + wh, c + ww) for r, c in taken]
    for tx, ty in sites:
        best = None
        for d in range(4, 41, 4):                     # gap between town cell and window edge
            cands = [(ty - wh // 2, tx + d), (ty - wh // 2, tx - d - ww),
                     (ty + d, tx - ww // 2), (ty - d - wh, tx - ww // 2),
                     (ty + d, tx + d), (ty - d - wh, tx - d - ww), (ty + d, tx - d - ww), (ty - d - wh, tx + d)]
            ok = [(r, c) for r, c in cands if 0 <= r <= TH - wh and 0 <= c <= TW - ww
                  and replaced[r:r + wh, c:c + ww].all() and land[r:r + wh, c:c + ww].all()
                  and not any(r < b2 and r + wh > b0 and c < b3 and c + ww > b1 for b0, b1, b2, b3 in boxes)]
            for r, c in ok:
                cost = 0. if relief is None else float(np.ptp(relief[r:r + wh, c:c + ww])) + SITE_RING_COST * d
                if best is None or cost < best[0]:
                    best = (cost, r, c)
            if best is not None and relief is None:
                break
        if best is not None:
            out.append(best[1:]); boxes.append((best[1], best[2], best[1] + wh, best[2] + ww))
    return out


def groups(wins, shape):
    """Windows that touch or overlap (edge or corner) share one level: lists of indices into wins."""
    wh, ww = shape
    parent = list(range(len(wins)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, (r, c) in enumerate(wins):
        for j in range(i):
            r2, c2 = wins[j]
            if r <= r2 + wh and r2 <= r + wh and c <= c2 + ww and c2 <= c + ww:
                parent[find(i)] = find(j)
    out = {}
    for i in range(len(wins)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


# --- SALT CRUST (beautify G) --- salt-pan tiles on the classify salt_crust range too: the
# biome_profiles REGIONS['salt_crusts'] outlines (Lop Nur, Halaqi, Qarhan) below RULES salt_crust
# max_elev, plus the Natural Earth 'salt' raster of west-real-terrain.npz (tile grid), join the
# strategic_ink playa hexes as salt flat. Their windows (and, from now on, every playa window; the
# gameplay site windows are unchanged) stay CLEAR_HEX hexes off water hexes (rivers, lakes, sea), road
# hexes and town plots, and CLEAR_CELLS tile cells off the real rivers and lakes of the npz.
CLEAR_HEX = 2            # hex steps (hex_components.hex_dilate) around water / road / town-plot hexes
CLEAR_CELLS = 2          # tile cells around the npz real river / lake cells


def crust_mask(lon, lat, real_elev):
    """Tile mask of the classify salt_crust range: inside REGIONS['salt_crusts'] (biome_profiles.poly_mask,
    unblurred) and below RULES['salt_crust']['max_elev'] (real metres, box 1 as classify's e)."""
    import biome_profiles as _bp
    import ground_materials as gm
    e = gm._box(real_elev, 1)
    return (_bp.poly_mask(lon, lat, _bp.REGIONS['salt_crusts']['polygons']) > .5) & (e < _bp.RULES['salt_crust']['max_elev'])


def crust_by_polygon(lon, lat):
    """{polygon name: tile mask} of REGIONS['salt_crusts'] (for the per-polygon report)."""
    import biome_profiles as _bp
    return {n: _bp.poly_mask(lon, lat, {n: p}) > .5 for n, p in sorted(_bp.REGIONS['salt_crusts']['polygons'].items())}


def clear_mask(cells, hy, hx, real_river=None, real_lake=None):
    """Tile mask where salt windows may lie: CLEAR_HEX hexes off water hexes (byte0 & 3 == 1: rivers, lakes,
    sea), road hexes (byte3 bits 1-6) and town plots (byte2 >> 4, byte3 bit 0); CLEAR_CELLS tile cells off
    real_river > 0 and real_lake (tile grids, optional)."""
    from hex_components import hex_dilate
    water = (cells[:, :, 0] & 3) == 1
    road = ((cells[:, :, 3].astype(np.int32) >> 1) & 63) > 0
    plot = ((cells[:, :, 2] >> 4) > 0) | ((cells[:, :, 3] & 1) > 0)
    bad = water | road | plot
    for _ in range(CLEAR_HEX):
        bad = hex_dilate(bad)
    out = ~bad[hy, hx]
    real = None
    if real_river is not None:
        real = np.asarray(real_river) > 0
    if real_lake is not None:
        real = (np.asarray(real_lake) > 0) if real is None else (real | (np.asarray(real_lake) > 0))
    if real is not None:
        for _ in range(CLEAR_CELLS):
            g = real.copy(); g[1:] |= real[:-1]; g[:-1] |= real[1:]; g[:, 1:] |= real[:, :-1]; g[:, :-1] |= real[:, 1:]
            real = g
        out &= ~real
    return out


CRUST_STEP = 2           # tile cells: placement scan step of the crust windows (the playa grid steps a window)


def crust_windows(crust, ok, shape, taken):
    """Extra windows on the crust: row-major scan in CRUST_STEP steps, first fit wins; a window needs
    MIN_SALT_SHARE crust, `ok` (replaced & land & clear) over its whole footprint and no overlap (edge
    contact allowed) with `taken` or an earlier pick. The playa grid (windows) steps a whole window and
    left the crust between lake, rivers and towns empty."""
    wh, ww = shape
    TH, TW = crust.shape
    c = np.pad(np.cumsum(np.cumsum(crust, 0, dtype=np.int64), 1, dtype=np.int64), ((1, 0), (1, 0)))
    o = np.pad(np.cumsum(np.cumsum(~ok, 0, dtype=np.int64), 1, dtype=np.int64), ((1, 0), (1, 0)))
    def box(s, r, cc):
        return s[r + wh, cc + ww] - s[r, cc + ww] - s[r + wh, cc] + s[r, cc]
    occ = np.zeros((TH, TW), bool)
    for r, cc in taken:
        occ[r:r + wh, cc:cc + ww] = True
    out = []
    rows, cols = np.nonzero(crust)
    if not len(rows):
        return out
    for r in range(max(0, int(rows.min()) - wh), min(TH - wh, int(rows.max())) + 1, CRUST_STEP):
        for cc in range(max(0, int(cols.min()) - ww), min(TW - ww, int(cols.max())) + 1, CRUST_STEP):
            if box(c, r, cc) < MIN_SALT_SHARE * wh * ww or box(o, r, cc) or occ[r:r + wh, cc:cc + ww].any():
                continue
            out.append((r, cc)); occ[r:r + wh, cc:cc + ww] = True
    return out
# --- end SALT CRUST


def plan(salt, replaced, land, shape, sites, relief, clear=None, crust=None):
    """Windows for both pan kinds, before the height projection. Natural Earth playa windows
    (decorative) are kept only where their group's ground spans <= PLAYA_MAX_RELIEF world units:
    a flat sheet over a rougher DEM slope would cut a terrace into real relief. Gameplay salt
    sites get a window wherever a ring position is free, levelled whatever the slope. Returns
    (windows, site windows placed, playa windows skipped). SALT CRUST: clear (tile bool) keeps playa /
    crust windows where wholly clear; crust (tile bool) adds crust_windows around the playa and site
    windows (placed after both, so those stay where they were; same relief test per group of crust
    windows). Order: playa and crust windows sorted, then the site windows (last, as before)."""
    wh, ww = shape
    ok = replaced if clear is None else (replaced & clear)
    playa = windows(salt, ok, land, shape)

    def flat(wins):
        out = []
        for g in groups(wins, shape):
            v = np.concatenate([relief[wins[i][0]:wins[i][0] + wh, wins[i][1]:wins[i][1] + ww].ravel() for i in g])
            if float(np.ptp(v)) <= PLAYA_MAX_RELIEF:
                out += [wins[i] for i in g]
        return out
    keep = flat(playa)
    keep.sort()
    placed = site_windows(sites, replaced, land, shape, keep, relief)
    skipped = len(playa) - len(keep)
    if crust is not None:
        extra = crust_windows(crust, ok & land, shape, keep + placed)
        kept = flat(extra)
        skipped += len(extra) - len(kept)
        keep = sorted(keep + kept)
    return keep + placed, len(placed), skipped


def level_field(wins, shape, height, scale=None):
    """(weight, target) on the tile grid: each window group is held at the median of `height` over
    its windows plus LEVEL_MARGIN, over the windows grown by LEVEL_PAD, with a smoothstep apron of
    LEVEL_APRON cells (Euclidean distance to the grown boxes). `scale` (0..1, e.g. edit alpha /
    EDIT_ALPHA) multiplies the weight so nothing changes where it is 0. Where aprons of two groups
    meet, the nearer group (larger weight) wins."""
    wh, ww = shape
    TH, TW = height.shape
    weight = np.zeros((TH, TW), np.float32); target = np.zeros((TH, TW), np.float32); levels = []
    reach = LEVEL_PAD + LEVEL_APRON
    for g in groups(wins, shape):
        boxes = [(wins[i][0], wins[i][0] + wh, wins[i][1], wins[i][1] + ww) for i in g]
        level = float(np.median(np.concatenate([height[r0:r1, c0:c1].ravel() for r0, r1, c0, c1 in boxes]))) + LEVEL_MARGIN
        R0 = max(0, min(b[0] for b in boxes) - reach); R1 = min(TH, max(b[1] for b in boxes) + reach)
        C0 = max(0, min(b[2] for b in boxes) - reach); C1 = min(TW, max(b[3] for b in boxes) + reach)
        yy = np.arange(R0, R1, dtype=np.float32)[:, None] + .5; xx = np.arange(C0, C1, dtype=np.float32)[None, :] + .5
        d = np.full((R1 - R0, C1 - C0), np.inf, np.float32)
        for r0, r1, c0, c1 in boxes:
            dy = np.maximum(np.maximum(r0 - LEVEL_PAD - yy, yy - r1 - LEVEL_PAD), 0)
            dx = np.maximum(np.maximum(c0 - LEVEL_PAD - xx, xx - c1 - LEVEL_PAD), 0)
            d = np.minimum(d, np.hypot(dy, dx))
        t = np.clip(1 - d / LEVEL_APRON, 0, 1); t = t * t * (3 - 2 * t)
        if scale is not None:
            t = t * scale[R0:R1, C0:C1]
        win = (slice(R0, R1), slice(C0, C1)); better = t > weight[win]
        weight[win] = np.where(better, t, weight[win]); target[win] = np.where(better, level, target[win])
        levels.append({'windows': [list(map(int, wins[i])) for i in g], 'level': round(level, 4)})
    return weight, target, levels


def footprint_spans(wins, shape, norm_lf):
    """Normalised LF range (max - min) over each window's footprint, LF pixels 4r..4(r+wh)
    (rows south-first, the frame tile t = LF pixel 4t); vanilla pans span <= .035."""
    wh, ww = shape
    return [round(float(np.ptp(norm_lf[4 * r:4 * (r + wh), 4 * c:4 * (c + ww)])), 4) for r, c in wins]


def stamp(layer0, owners, at, climate, TW):
    """Write the donor owners at window origin `at` into cache layer 0. Returns (covered mask cells,
    [(path value, x, y, flags_for_list, climate, footprint rows, footprint cols)] for list records)."""
    R, C = at
    placed, covered = [], []
    for o in owners:
        pr, pc = o['parent'][0] + R, o['parent'][1] + C
        parent_index = pr * TW + pc
        for dr, dc in o['cells']:
            r, c = dr + R, dc + C
            cell = layer0[r, c]
            cell['flags'] = o['flags'] if (dr, dc) == o['parent'] else 0x710
            cell['x'] = o['x'] + C; cell['y'] = o['y'] + R
            cell['climate'] = climate; cell['extra'] = o['extra'] if (dr, dc) == o['parent'] else 0
            cell['lo'] = 0; cell['hi'] = 0
            cell['path'] = o['path'] if (dr, dc) == o['parent'] else -parent_index - 1
            layer0[r, c] = cell
            covered.append((r, c))
        rows = [d[0] + R for d in o['cells']]; cols = [d[1] + C for d in o['cells']]
        placed.append((o['path'], o['x'] + C, o['y'] + R, o['flags'] | (o['extra'] << 2), climate,
                       (min(rows), max(rows)), (min(cols), max(cols)), (pr, pc)))
    return covered, placed


def add_list_paths(src, names):
    """dlc07 tile_list.bin with `names` appended to its path table (indices of existing records stay)."""
    import struct
    count = struct.unpack_from('<I', src, 12)[0]
    pos = 16
    for _ in range(count):
        pos += 2 + struct.unpack_from('<H', src, pos)[0]
    extra = b''.join(struct.pack('<H', len(n.encode())) + n.encode() for n in names)
    out = bytearray(src[:pos]) + extra + src[pos:]
    struct.pack_into('<I', out, 12, count + len(names))
    return bytes(out)

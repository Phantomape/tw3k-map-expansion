"""Vanilla sea_coast tiles on the new sea shores of the canvas, for the tile cache section of
polish_campaign_terrain.py (stamped like the salt pans and mountain tiles; records ride salt_placed).

What vanilla does (3k_dlc07_main_map global_map/tile_list.bin + map.hex, 3k_main_map cache, 3553 pieces):
  * every land hex beside water is beach (map.hex byte 0 kind 2) or cliff (kind 3); beach hexes carry
    sea_coast pieces (17124 of their 17412 tile cells), cliff hexes blockout_cliff (out of scope here).
    Only ~740 pieces line the sea; ~2500 line the navigable rivers and ~230 the lakes;
  * a piece covers whole hex blocks (the 2x2 tile block at (2q, 2r + odd(q))): 2641 one-hex 2x2
    pieces (straight/turn/end), 518 two-hex 2x4 pieces (mid_curve, a column pair) and 394 two-hex
    4x3 pieces (mid_straight, scurve; neighbouring columns, 8 of their 12 cells owned);
  * the piece and quarter turn follow the hex's water-neighbour mask (6 bits, CAIME directions):
    water mask alone picks vanilla's most used option for 90% of the pieces, water mask + beach-run
    mask (neighbours that are beach) for 93%; end pieces sit where the beach run meets a cliff;
  * cache layer 0 holds the piece (parent flags 0x7?0 = one-hot quarter turn, extra 0, children
    -parent-1 with flags 0x710), layer 2 holds generic_sea under every coast cell: one generic_sea
    2x2 at the piece's x/y with the piece's flags under a 2x2 piece, one generic_sea 1x1 (0x710) per
    owned cell under a two-hex piece. The list has a record for each (same climate as the piece);
  * record lo/hi of the pieces = LF extrema over the footprint plus two cells (80% exact, the road
    formula); vanilla's generic_sea ranges follow no LF formula (57% have lo = 0.1464), the
    underlays here take the same footprint + 2 formula;
  * local tile_list.bin keeps the 0x7?0 flags, global_map/tile_list.bin rotation only (0x?0).
How the new shores are tiled (target: canvas land hex of kind 0/2 beside open sea, i.e. map.hex sea
ground types sea_coast/sea_ocean, whose block is free replaced ground; not beside a river or lake hex):
  1. vanilla two-hex patterns (block offsets + water masks) seen >= MIN_PATTERN times, most hexes and
     most used first, where both hexes are targets with those masks and the owned cells are free;
  2. one 2x2 piece per remaining target: options of the (water, run) key, else of the water key
     (each seen >= MIN_PATTERN times); a mask vanilla never tiles is left to the floors (reported);
  3. a piece whose footprint LF span is above the vanilla SPAN_QUANTILE of its size, or whose range
     does not reach the vanilla sea-shore water line band (lo <= LO_MAX, hi >= HI_MIN), is skipped.
Run mask on the canvas: neighbours that are land beside water and not cliff (vanilla: beach).
Where (region_cells): outside the native rectangle and on the filler (revealed_filler.filler_hex: vanilla
non-playable filler that is land now, playable or not; canvas, visual only), and inside it only on really edited
native ground
(LF changed from vanilla at halo 0, or a real-terrain patch such as korea, with edit alpha > EDIT_ALPHA
so repair_native_placements relays its floors), never within NATIVE_GUARD_CELLS of unedited native land.
Only cells that are relaid floor ground are ever taken: a vanilla sea_coast, blockout_cliff or any other
non-floor owner (and any active layer-2 owner) is never overwritten.
"""
import collections

import numpy as np

from hex_components import hex_dilate, neighbour_bits
from road_tiles import ROT, FLAGS, CHILD_FLAGS, mesh_dims, origin, tile_hex, _pick

SET = 'sea_coast'
SEA = 'generic_sea'
MIN_PATTERN = 2
SPAN_QUANTILE = 99.5
SHORE_QUANTILE = .5          # percent: vanilla sea-shore piece lo/hi band (lo <= p100-q, hi >= p q)
RIVER_CLEAR = 3              # hexes kept clear of river water hexes (mouths: river water models sit there)
RIVER_CLEAR_CELLS = 8        # tile cells kept clear of the river water band (analysis/river-water-level.npy)
DOCK_CLEAR = 6               # hexes kept clear of coast_props dock anchors
PLOT_CLEAR = 9               # tile cells (max filter) kept clear of town/port plots, as mountain_tiles
NATIVE_GUARD_CELLS = 8       # tile cells kept clear of unedited native land (region_cells)
EDIT_ALPHA = .02             # polish edit alpha above which repair_native_placements relays floors
COVER_FILE = 'analysis/coast-tile-cover.npy'
REPORT_FILE = 'analysis/coast-tiles.json'
B = chr(92)


def tile_path(piece, tset=SET):
    return f'terrain{B}tiles{B}campaign{B}{tset}{B}{piece}{B}'


SEA_1 = tile_path('1x1', SEA)
SEA_2 = tile_path('2x2', SEA)


def ground_types(cells):
    """map.hex ground type id per hex: land ids 1..len(land types), water len(land)+1+sea index."""
    return ((cells[..., 5] >> 4) | ((cells[..., 6] & 7) << 4)).astype(np.int64)


def water_kinds(cells, land_types, sea_types):
    """(water, open sea, river, lake) hex masks. Open sea = sea_coast / sea_ocean ground."""
    kind = cells[..., 0] & 3
    water = kind == 1
    gt = ground_types(cells)
    gid = {n: len(land_types) + 1 + i for i, n in enumerate(sea_types)}
    sea = water & np.isin(gt, [gid[n] for n in ('sea_coast', 'sea_ocean') if n in gid])
    river = water & np.isin(gt, [gid[n] for n in gid if n.startswith('river')])
    lake = water & np.isin(gt, [gid[n] for n in ('sea_lake',) if n in gid])
    return water, sea, river, lake


def masks(cells):
    """(water-neighbour bits, run-neighbour bits) per hex. Run: land beside water that is not cliff
    (in vanilla exactly the beach hexes, since every land hex beside water is beach or cliff)."""
    kind = cells[..., 0] & 3
    water = kind == 1
    shore = ~water & hex_dilate(water) & (kind != 3)
    return neighbour_bits(water), neighbour_bits(shore)


def piece_dims(read, names):
    out = {}
    for pc in sorted(names):
        out[pc] = mesh_dims(read(f'terrain/tiles/campaign/{SET}/{pc}/mesh.rigid_model_v2'))
    return out


def footprint(dims, pc, rt):
    w, h = dims[pc]
    return (h, w) if rt % 2 else (w, h)


def covered_hexes(x, y, fw, fh, H, W):
    """Hexes whose whole 2x2 block lies in the footprint, sorted."""
    out = set()
    for cy in range(y, y + fh):
        for cx in range(x, x + fw):
            q, r = tile_hex(cx, cy)
            if 0 <= q < W and 0 <= r < H:
                ox, oy = origin(q, r)
                if x <= ox and ox + 2 <= x + fw and y <= oy and oy + 2 <= y + fh:
                    out.add((q, r))
    return sorted(out)


class Learned:
    """Vanilla coast layout statistics (see module doc)."""
    def __init__(self, list_bytes, vmap, read):
        from read_tile_list import parse
        paths, _, _, _, recs, _ = parse(list_bytes)
        cells = vmap['cells']
        H, W = cells.shape[:2]
        wm, run = masks(cells)
        _, sea, _, _ = water_kinds(cells, vmap['land_ground_types'], vmap['sea_ground_types'])
        seaside = hex_dilate(sea)
        coast = [(r, paths[r[1]].split(B)[4]) for r in recs if B + SET + B in paths[r[1]]]
        self.dims = piece_dims(read, {pc for _, pc in coast})
        self.multi = collections.defaultdict(collections.Counter)
        self.single = collections.defaultdict(collections.Counter)    # (wm, run) -> options
        self.single_w = collections.defaultdict(collections.Counter)  # wm -> options
        spans = collections.defaultdict(list)
        shore_lo, shore_hi = [], []
        self.pieces = 0
        for r, pc in coast:
            if (r[5] & 0xf0) not in ROT:
                continue
            rt = ROT[r[5] & 0xf0]
            x, y = r[3], r[4]
            fw, fh = footprint(self.dims, pc, rt)
            hexes = covered_hexes(x, y, fw, fh, H, W)
            if not hexes:
                continue
            self.pieces += 1
            spans[len(hexes)].append(r[7] - r[6])
            if any(seaside[h[1], h[0]] for h in hexes):
                shore_lo.append(r[6]); shore_hi.append(r[7])
            if len(hexes) == 1:
                q, rr = hexes[0]
                if (fw, fh) == (2, 2) and origin(q, rr) == (x, y):
                    self.single[int(wm[rr, q]), int(run[rr, q])][pc, rt] += 1
                    self.single_w[int(wm[rr, q])][pc, rt] += 1
            else:
                key = (fw, fh, tuple(sorted((origin(*h)[0] - x, origin(*h)[1] - y, int(wm[h[1], h[0]])) for h in hexes)))
                self.multi[key][pc, rt] += 1
        self.span_max = {n: float(np.percentile(v, SPAN_QUANTILE)) for n, v in spans.items()}
        self.lo_max = float(np.percentile(shore_lo, 100 - SHORE_QUANTILE))
        self.hi_min = float(np.percentile(shore_hi, SHORE_QUANTILE))
        self.sea_shore_pieces = len(shore_lo)

    def options(self, w, r):
        """(options [(piece, turn), count], key level) for a single hex."""
        for table, key, level in ((self.single, (w, r), 'water+run'), (self.single_w, w, 'water')):
            opts = [(o, c) for o, c in sorted(table.get(key, {}).items()) if c >= MIN_PATTERN]
            if opts:
                return opts, level
        return [], None


def cache_shapes(layer, tail_paths, tset):
    """{(piece, turn): (owned cells [(dx, dy)] relative to the record x/y, parent (dx, dy))}: the most
    common ownership of each `tset` tile in a donor cache layer (path values one-based)."""
    flat = layer.ravel()
    ow = layer.shape[1]
    active = flat['flags'] != 16
    own = np.where(flat['path'] >= 0, np.arange(flat.size), -flat['path'].astype(np.int64) - 1)
    values = np.array([i + 1 for i, p in enumerate(tail_paths) if B + tset + B in p])
    parents = np.flatnonzero(active & (flat['path'] > 0) & np.isin(flat['path'], values))
    cells = np.flatnonzero(active & np.isin(own, parents))
    cells = cells[np.argsort(own[cells], kind='stable')]
    o = own[cells]
    starts, ends = np.searchsorted(o, parents), np.searchsorted(o, parents, 'right')
    seen = collections.defaultdict(collections.Counter)
    for p, a, b in zip(parents.tolist(), starts.tolist(), ends.tolist()):
        rec = flat[p]
        if (int(rec['flags']) & 0xf0) not in ROT:
            continue
        pc = tail_paths[int(rec['path']) - 1].split(B)[4]
        x, y = int(rec['x']), int(rec['y'])
        cy, cx = np.divmod(cells[a:b], ow)
        py, px = divmod(p, ow)
        seen[pc, ROT[int(rec['flags']) & 0xf0]][(tuple(sorted(zip((cx - x).tolist(), (cy - y).tolist()))), (px - x, py - y))] += 1
    return {k: max(sorted(c.items()), key=lambda kv: kv[1])[0] for k, c in sorted(seen.items())}


def targets(cells, land_types, sea_types, allowed_hex):
    """Shore hexes to tile: land of kind 0/2 beside open sea, not within RIVER_CLEAR hexes of river
    water, not beside a lake, inside allowed_hex."""
    kind = cells[..., 0] & 3
    water, sea, river, lake = water_kinds(cells, land_types, sea_types)
    near_river = river.copy()
    for _ in range(RIVER_CLEAR):
        near_river = hex_dilate(near_river)
    return ~water & np.isin(kind, (0, 2)) & hex_dilate(sea) & ~near_river & ~hex_dilate(lake) & allowed_hex


def layout(target, wm, run, learned, free, shapes, span):
    """Pieces for the target hexes. free (TH, TW) cells a piece may own; shapes cache_shapes of the
    coast pieces; span(x, y, fw, fh) -> (lo, hi) LF range of a footprint plus two cells.
    Returns (placements [(piece, turn, x, y, fw, fh, hexes)], stats)."""
    H, W = target.shape
    TH, TW = free.shape
    tr, tq = np.nonzero(target)            # row-major
    tid = np.full((H, W), -1, np.int64)
    tid[tr, tq] = np.arange(len(tq))
    done = np.zeros(len(tq), bool)
    avail = free.copy()
    stats = collections.Counter()
    placements = []

    def owned(pc, rt, x, y):
        cells_, _ = shapes[pc, rt]
        return [(x + dx, y + dy) for dx, dy in cells_]

    def fits(pc, rt, x, y, n):
        fw, fh = footprint(learned.dims, pc, rt)
        if x < 0 or y < 0 or x + fw > TW or y + fh > TH:
            return False, 'edge'
        if not all(avail[cy, cx] for cx, cy in owned(pc, rt, x, y)):
            return False, 'blocked'
        lo, hi = span(x, y, fw, fh)
        if hi - lo > learned.span_max.get(n, np.inf):
            return False, 'span'
        if lo > learned.lo_max or hi < learned.hi_min:
            return False, 'waterline'
        return True, None

    def accept(pc, rt, x, y, hexes):
        for cx, cy in owned(pc, rt, x, y):
            avail[cy, cx] = False
        for h in hexes:
            done[tid[h[1], h[0]]] = True
        fw, fh = footprint(learned.dims, pc, rt)
        placements.append((pc, rt, x, y, fw, fh, tuple(hexes)))

    # 1. two-hex patterns
    ox, oy = origin(tq, tr)
    cands = []
    for key in sorted(learned.multi):
        fw, fh, hx = key
        opts = [(o, c) for o, c in sorted(learned.multi[key].items()) if o in shapes]
        total = sum(learned.multi[key].values())
        if total < MIN_PATTERN or not opts:
            continue
        dx0, dy0, m0 = hx[0]
        sel = np.flatnonzero(wm[tr, tq] == m0)
        X, Y = ox[sel] - dx0, oy[sel] - dy0
        ok = (X >= 0) & (Y >= 0) & (X + fw <= TW) & (Y + fh <= TH)
        hexes = []
        for dx, dy, m in hx:
            cx, cy = X + dx, Y + dy
            q, r = tile_hex(cx, cy)
            inside = (q >= 0) & (q < W) & (r >= 0) & (r < H)
            qq, rr = np.clip(q, 0, W - 1), np.clip(r, 0, H - 1)
            ok &= inside & (cx % 2 == 0) & (2 * rr + qq % 2 == cy) & (tid[rr, qq] >= 0) & (wm[rr, qq] == m)
            hexes.append((qq, rr))
        for i in np.flatnonzero(ok).tolist():
            cands.append((-len(hx), -total, int(Y[i]), int(X[i]), key, [(int(q[i]), int(r[i])) for q, r in hexes], opts))
    cands.sort(key=lambda c: c[:4])
    for _, _, y, x, key, hexes, opts in cands:
        if any(done[tid[r, q]] for q, r in hexes):
            continue
        good, why = [], None
        for o, c in opts:
            f, why_ = fits(*o, x, y, len(hexes))
            if f:
                good.append((o, c))
            else:
                why = why_
        if not good:
            stats[f'multi_skipped_{why}'] += 1
            continue
        pc, rt = _pick(good, *hexes[0])
        accept(pc, rt, x, y, hexes)
        stats['multi_pieces'] += 1
        stats['multi_hexes'] += len(hexes)

    # 2. one 2x2 piece per remaining hex
    unmatched = collections.Counter()
    for i in np.flatnonzero(~done).tolist():
        q, r = int(tq[i]), int(tr[i])
        w, rn = int(wm[r, q]), int(run[r, q])
        opts, level = learned.options(w, rn)
        if not opts:
            stats['unmatched_mask_hexes'] += 1
            unmatched[w] += 1
            continue
        x, y = origin(q, r)
        good, why = [], None
        for o, c in opts:
            if o not in shapes:
                continue
            f, why_ = fits(*o, x, y, 1)
            if f:
                good.append((o, c))
            else:
                why = why_
        if not good:
            stats[f'single_skipped_{why}'] += 1
            continue
        pc, rt = _pick(good, q, r)
        accept(pc, rt, x, y, [(q, r)])
        stats['single_pieces'] += 1
        stats[f'single_by_{level.replace("+", "_")}'] += 1
    stats = dict(sorted(stats.items()))
    stats['target_hexes'] = int(len(tq))
    stats['tiled_hexes'] = int(sum(len(p[6]) for p in placements))
    stats['unmatched_masks'] = {str(k): v for k, v in sorted(unmatched.items(), key=lambda kv: (-kv[1], kv[0]))}
    return placements, stats


def lf_range(lf, x, y, fw, fh):
    """LF extrema (/65535) over the footprint plus two tile cells (south-up LF, 4 texels per cell)."""
    b = lf[max(0, (y - 2) * 4):(y + fh + 2) * 4, max(0, (x - 2) * 4):(x + fw + 2) * 4]
    return float(b.min()) / 65535, float(b.max()) / 65535


def stamp(layer0, layer2, placements, shapes, sea_shapes, value, climate, lf):
    """Write the pieces into cache layer 0 and their generic_sea underlays into layer 2.
    value(tile path) -> one-based cache value. Returns (layer-0 cover, layer-2 cover, records
    [(cache value, x, y, flags, climate, (row0, row1), (col0, col1), (pr, pc))] in salt_placed form).
    Underlay records come before their piece and name the piece's parent cell as (pr, pc): polish's
    salt_placed range loop writes each record's range into layer 0 at (pr, pc), so the piece's own
    record, written last, leaves the piece's range there (layer 2 keeps the ranges set here)."""
    TH, TW = layer0.shape
    cover0 = np.zeros((TH, TW), bool)
    cover2 = np.zeros((TH, TW), bool)
    out = []

    def put(layer, cells_, parent, val, flags, x, y, cl, lo, hi):
        px, py = parent
        pidx = py * TW + px
        for cx, cy in cells_:
            c = layer[cy, cx]
            is_parent = (cx, cy) == (px, py)
            c['path'] = val if is_parent else -pidx - 1
            c['flags'] = flags if is_parent else CHILD_FLAGS
            c['x'], c['y'], c['climate'], c['extra'] = x, y, cl, 0
            c['lo'], c['hi'] = (lo, hi) if is_parent else (0, 0)
            layer[cy, cx] = c

    for pc, rt, x, y, fw, fh, _ in placements:
        cells_, (pdx, pdy) = shapes[pc, rt]
        cells_ = [(x + dx, y + dy) for dx, dy in cells_]
        parent = (x + pdx, y + pdy)
        cl = int(climate[y + fh // 2, x + fw // 2])
        lo, hi = lf_range(lf, x, y, fw, fh)
        under = []
        if (fw, fh) == (2, 2) and ('2x2', rt) in sea_shapes:
            sc, (sdx, sdy) = sea_shapes['2x2', rt]
            ucells = [(x + dx, y + dy) for dx, dy in sc]
            assert sorted(ucells) == sorted(cells_), (pc, rt, x, y)
            ulo, uhi = lf_range(lf, x, y, 2, 2)
            put(layer2, ucells, (x + sdx, y + sdy), value(SEA_2), FLAGS[rt], x, y, cl, ulo, uhi)
            under.append((value(SEA_2), x, y, FLAGS[rt], cl, (y, y + 1), (x, x + 1), (parent[1], parent[0])))
        else:
            for cx, cy in sorted(cells_, key=lambda t: (t[1], t[0])):
                ulo, uhi = lf_range(lf, cx, cy, 1, 1)
                put(layer2, [(cx, cy)], (cx, cy), value(SEA_1), FLAGS[0], cx, cy, cl, ulo, uhi)
                under.append((value(SEA_1), cx, cy, FLAGS[0], cl, (cy, cy), (cx, cx), (parent[1], parent[0])))
        put(layer0, cells_, parent, value(tile_path(pc)), FLAGS[rt], x, y, cl, lo, hi)
        for cx, cy in cells_:
            cover0[cy, cx] = True
            cover2[cy, cx] = True
        out += under
        out.append((value(tile_path(pc)), x, y, FLAGS[rt], cl, (y, y + fh - 1), (x, x + fw - 1), (parent[1], parent[0])))
    return cover0, cover2, out


def allowed_cells(replacement, region, taken, plot_cells, river_band, TH, TW):
    """Tile cells a coast piece may own: replaced ground in `region` (region_cells), away from
    town/port plots and the river water band, not taken by other stamps."""
    import fastops
    ok = replacement & region & ~taken
    if plot_cells is not None:
        ok &= ~fastops.max_filter_mask(plot_cells, PLOT_CLEAR)
    if river_band is not None:
        ok &= ~fastops.max_filter_mask(river_band, 2 * RIVER_CLEAR_CELLS + 1)
    return ok


def edited_native(edited_lf, alpha, patch=None):
    """Native ground that was really edited: LF changed from vanilla (native_edit_mask.edited_tiles,
    halo 0) or a real-terrain patch (hehuang_terrain height weight > 0: korea, hehuang), and relaid by
    polish (edit alpha > EDIT_ALPHA, repair_native_placements' own floor rule). Tile grid."""
    e = edited_lf if patch is None else (edited_lf | patch)
    return e & (alpha > EDIT_ALPHA)


def region_cells(TH, TW, x0, y0, x1, y1, edited, land, filler):
    """Cells a coast piece may own by place: outside the native tile rectangle [x0, x1) x [y0, y1),
    on the filler (filler: tile grid of revealed_filler.filler_hex, canvas), or inside it on
    edited ground (edited_native); never within NATIVE_GUARD_CELLS of unedited native land (vanilla land,
    beaches and cliffs keep their own tiles; vanilla sea beside a new coast is fine)."""
    import fastops
    yy = np.arange(TH)[:, None]
    xx = np.arange(TW)[None, :]
    rect = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1) & ~filler
    unedited_land = rect & ~edited & land
    return (~rect | edited) & ~fastops.max_filter_mask(unedited_land, 2 * NATIVE_GUARD_CELLS + 1)


def patch_tiles(hh, hy, hx):
    """Real-terrain patch hexes (height weight > 0, no growth) on the tile grid, or None."""
    if hh is None:
        return None
    parts = hh.parts if hasattr(hh, 'part_list') else [hh]
    return np.logical_or.reduce([p.hex_mask(0) for p in parts])[hy, hx]


def dock_hexes(coast_props, H, W):
    """Hexes within DOCK_CLEAR of the coast_props dock anchors (analysis/coast-props.json)."""
    m = np.zeros((H, W), bool)
    for v in coast_props.get('variants', []):
        for d in v.get('docks', []):
            q, r = d['anchor_hex']
            if 0 <= q < W and 0 <= r < H:
                m[r, q] = True
    for _ in range(DOCK_CLEAR):
        m = hex_dilate(m)
    return m


def blocks_ok(cell_mask, H, W):
    """(H, W) hexes whose whole 2x2 tile block lies in cell_mask."""
    from road_tiles import blocks_in
    return blocks_in(cell_mask, H, W)


def plan(cells, land_types, sea_types, learned, free, shapes, lf, extra_block_hex=None):
    """Targets + layout on the canvas (shared by polish and the offline driver).
    free: (TH, TW) cells a piece may own. Returns (placements, stats, target hex mask)."""
    H, W = cells.shape[:2]
    allowed = blocks_ok(free, H, W)
    if extra_block_hex is not None:
        allowed &= ~extra_block_hex
    target = targets(cells, land_types, sea_types, allowed)
    wm, run = masks(cells)
    placements, stats = layout(target, wm, run, learned, free, shapes, lambda x, y, fw, fh: lf_range(lf, x, y, fw, fh))
    return placements, stats, target


def report(placements, stats, learned, cover0):
    by_piece = collections.Counter(p[0] for p in placements)
    return {**stats, 'pieces': len(placements), 'cells': int(cover0.sum()),
            'by_piece': dict(sorted(by_piece.items())),
            'vanilla': {'pieces': learned.pieces, 'sea_shore_pieces': learned.sea_shore_pieces,
                        'span_max': {str(k): round(v, 5) for k, v in sorted(learned.span_max.items())},
                        'lo_max': round(learned.lo_max, 5), 'hi_min': round(learned.hi_min, 5)}}


# ---------------------------------------------------------------- real shore type
# Beach pieces only where the real shore is a beach or a low coast. The basis is ETOPO 2022 (60") in
# analysis/west-real-terrain.npz (`elevation`, metres, per canvas tile cell). Inside the Korea patch
# window the DEM is first aligned to the map's coasts row by row exactly as korea_patch.build does
# (korea_patch.coast_warp: the projection puts the real Korean coasts up to 30 hexes off the map's).
# A shore hex is LOW when the aligned DEM has both land and sea within ALIGN_R hexes (the real coast
# is there) and its land within LOW_R hexes stays at or below LOW_COAST_M metres; else STEEP (rocky
# or high: vanilla would draw cliff) or UNKNOWN (no real coast within ALIGN_R: the map's shore is not
# a real one, e.g. Jeju's stylised position, so no beach is claimed).
ALIGN_R = 2
LOW_R = 1
LOW_COAST_M = 60.          # metres: tidal flats and low shore (Korea west); rocky/high coasts rise above it


def korea_aligned_elevation(A, cells_final, vanilla_map):
    """(e, (x0, x1, y0, y1)): ETOPO elevation sampled through korea_patch's coast warp on its window
    (the same land masks and warp as korea_patch.build), or None without the patch module/data."""
    import korea_patch as kp
    from scipy import ndimage
    import map_hex
    import campaign_world as cw
    q0, q1, r0, r1 = kp.window(A)['hex_win']
    x0, x1, y0, y1 = 2 * q0, 2 * q1, 2 * r0, 2 * r1
    ty, tx = np.mgrid[y0:y1, x0:x1]
    hq, hr = cw.tile_hex(tx, ty)
    z = np.load(A / 'analysis/west-real-terrain.npz')
    bx0, bx1, by0, by1 = (int(v) for v in z['box'])
    V = map_hex.read(vanilla_map)
    vq, vr = hq - cw.PAD_LEFT, hr - cw.PAD_BOTTOM
    inside = (vq >= 0) & (vq < cw.NATIVE_W) & (vr >= 0) & (vr < cw.NATIVE_H)
    vqc, vrc = np.clip(vq, 0, cw.NATIVE_W - 1), np.clip(vr, 0, cw.NATIVE_H - 1)
    vcells = V['cells'][vrc, vqc]
    land = ((cells_final[hr, hq, 0] & 3) != 1) & ((vcells[..., 0] & 3) != 1) & inside
    R = kp.WARP_REACH
    xa0, xa1 = max(bx0, x0 - R), min(bx1, x1 + R)
    dem_land = np.zeros((y1 - y0, x1 - x0 + 2 * R), bool)
    dem_land[:, xa0 - (x0 - R):xa1 - (x0 - R)] = z['elevation'][y0 - by0:y1 - by0, xa0 - bx0:xa1 - bx0] > 0
    xd, _, _ = kp.coast_warp(land, dem_land, x0, ndimage, r0)
    xs_ = np.clip(np.rint(xd).astype(np.int64) + x0 - bx0, 0, bx1 - bx0 - 1)
    return z['elevation'][ty - by0, xs_].astype(np.float32), (x0, x1, y0, y1)


def shore_types(target, elev, windows=()):
    """{(q, r): 'low' | 'steep' | 'unknown'} for the target hexes. elev: (TH, TW) ETOPO metres on the
    canvas tile grid; windows: [(e, (x0, x1, y0, y1))] aligned DEM pieces pasted over it first."""
    e = elev.copy()
    for w, (x0, x1, y0, y1) in windows:
        e[y0:y1, x0:x1] = w
    out = {}
    r_, q_ = np.nonzero(target)
    TH, TW = e.shape
    for q, r in zip(q_.tolist(), r_.tolist()):
        x, y = origin(q, r)
        a = e[max(0, y - 2 * ALIGN_R):y + 2 + 2 * ALIGN_R, max(0, x - 2 * ALIGN_R):x + 2 + 2 * ALIGN_R]
        if not ((a > 0).any() and (a <= 0).any()):
            out[q, r] = 'unknown'
            continue
        b = e[max(0, y - 2 * LOW_R):y + 2 + 2 * LOW_R, max(0, x - 2 * LOW_R):x + 2 + 2 * LOW_R]
        out[q, r] = 'low' if float(b.max()) <= LOW_COAST_M else 'steep'
    return out


def beach_block(cells, land_types, sea_types, elev, windows, names):
    """(hex mask, stats): shore hexes that are not a LOW real shore (shore_types), blocked from beach
    pieces, on the whole canvas: the native rectangle (its edited shores) and the new shores outside it
    (vanilla lines about 75 % of its sea shore with cliffs; a beach on a steep real coast reads as sand
    under a cliff). stats counts each type per side ('native', 'canvas'); the filler
    (revealed_filler.filler_hex; names: the map's land_regions + sea_regions) is canvas. elev None (no
    real terrain): the native shores are all blocked and the canvas shores left open, as before real terrain."""
    import campaign_world as cw
    from revealed_filler import filler_hex
    H, W = cells.shape[:2]
    rect = np.zeros((H, W), bool)
    rect[cw.NATIVE_HEX_SLICE] = True
    rect &= ~filler_hex(cells, names)
    shore = targets(cells, land_types, sea_types, np.ones((H, W), bool))
    if elev is None:
        return shore & rect, {'no_real_terrain': int((shore & rect).sum())}
    types = shore_types(shore, elev, windows)
    block = np.zeros((H, W), bool)
    stats = {'native': collections.Counter(), 'canvas': collections.Counter()}
    for (q, r), t in sorted(types.items()):
        stats['native' if rect[r, q] else 'canvas'][t] += 1
        if t != 'low':
            block[r, q] = True
    return block, {k: dict(sorted(v.items())) for k, v in stats.items()}


def save_npy(path, a):
    """np.save through a temporary file in the same directory (never a torn file)."""
    import os
    path = str(path)
    tmp = f'{path}.{os.getpid()}.tmp.npy'
    np.save(tmp, a)
    os.replace(tmp, path)


def save_text(path, text):
    import os
    path = str(path)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)

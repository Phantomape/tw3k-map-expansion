"""Vanilla road tiles on the gameplay roads of map.hex, for the tile cache section of
polish_campaign_terrain.py (the visual layer of apply_roads.py's road edge masks).

What vanilla does (3k_dlc07_main_map global_map/tile_list.bin + its map.hex, 7551 road pieces):
  * a road hex is the 2x2 tile block at (2q, 2r + odd(q)); every road piece covers whole road
    hexes (one 2x2 piece per hex, or 4x2 / 6x2 / 4x3 / 8x3 / 4x4 pieces over 2-4 hexes, whose spare
    cells hold no road), x/y is the footprint's minimum column/row, size is the mesh bbox / 128;
  * a piece joins its neighbours where its road crosses the footprint edge (blend0: the channels
    other than the one keeping the ground below). In the block frame the neighbour across a side
    edge is fixed (right edge: upper cell up-right, lower cell down-right; left: up-left /
    down-left); across the top or bottom edge the road runs in the left or right cell column
    (its "lane"). Vanilla's pieces meet with equal lanes on every vertical edge seen from both
    sides (965 of 965) and their side ports match the hex masks (98%);
  * pattern = (rotated footprint, [(block offset, road edge mask) of the hexes it covers]) is what
    the vanilla road layout shows: the same pattern, translated by whole hex columns, takes the
    same piece and rotation.
How the new roads are tiled (every road hex that is not a vanilla road hex and whose block is
free cache ground, i.e. replaced cells about to get floors):
  1. multi-hex vanilla patterns seen at least MIN_PATTERN times, largest first, where every covered
     hex is a target with the same mask, the footprint holds no other road and is free, the
     piece's ports equal the hexes' external road edges, its lanes agree with lanes already fixed
     (vanilla tiles beside the new road, earlier pieces) and its height span stays within the
     vanilla spans of pieces that size;
  2. one 2x2 piece per remaining hex whose ports are exactly the hex mask, lanes fixed outward from
     the constrained hexes (breadth-first, so a lane change falls on a straight vertical hex where
     vanilla has a switching piece), weighted by vanilla use of that piece for that mask;
  3. hexes with masks no 2x2 piece has (sharp 60 degree turns, rare junctions) take the closest
     piece and are reported.
All new roads are roads_tracks (level-1 tracks: frontier and desert routes; campaign_map_roads
upgrades the gameplay level, not these tiles).

Cache (layer 0, as the 3k_main_map donor cache stores road owners): parent cell flags 0x7?0 =
one-hot quarter turn, extra 0, lo/hi = LF extrema over the footprint plus two cells (the vanilla
list formula); children path -parent-1, flags 0x710, lo/hi 0; the owned cells and parent offset of
each piece and turn are read from the donor cache (some footprints leave corner cells to floors).
Cells of the old 2x2 floor grid split by an odd-column road block take 1-cell generic_1x1 owners
(vanilla's 3k_main_map cache has 26531 such owners).
"""
import collections
import struct

import numpy as np

SET = 'roads_tracks'
ROT = {0x10: 0, 0x20: 1, 0x40: 2, 0x80: 3}
FLAGS = (0x710, 0x720, 0x740, 0x780)       # list record / cache parent flags per quarter turn (cw)
CHILD_FLAGS = 0x710
SINGLE_FLAGS = 0x710                      # 1-cell generic_1x1 cache owners in split floor cells
MIN_PATTERN = 2                           # multi-hex patterns seen at least this often in vanilla
VARIETY = .25                             # options within this share of the most used one are mixed
SPAN_QUANTILE = 99.5
LANE_INSET = .25                          # cells inside a block edge where a vertical crossing's lane is read                      # multi-hex piece height span limit (vanilla percentile per size)
# CAIME Hex.Directions_FlatTop (dq, dr) for even / odd columns: 0 up, then clockwise.
DIRS = (((0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)),
        ((0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)))


def piece_path(piece, tset=SET):
    return f'terrain\\tiles\\campaign\\{tset}\\{piece}\\'


def road_masks(cells):
    """Road edge mask per hex: map.hex cell byte 3 bits 1-6."""
    return ((cells[..., 3] >> 1) & 63).astype(np.uint8)


def mutual_masks(mask):
    """Road edge masks keeping only bits whose neighbour carries the reverse bit. Vanilla has no
    other kind; the canvas map has some (earlier authored roads pointing at plain land), which
    would otherwise draw a road running off into nothing. Returns (masks, dangling bit count)."""
    H, W = mask.shape
    q, r = np.meshgrid(np.arange(W), np.arange(H))
    out = np.zeros_like(mask)
    for d in range(6):
        dq = np.where(q % 2 == 0, DIRS[0][d][0], DIRS[1][d][0])
        dr = np.where(q % 2 == 0, DIRS[0][d][1], DIRS[1][d][1])
        nq, nr = q + dq, r + dr
        inside = (nq >= 0) & (nq < W) & (nr >= 0) & (nr < H)
        back = mask[np.clip(nr, 0, H - 1), np.clip(nq, 0, W - 1)] >> ((d + 3) % 6) & 1
        out |= ((mask >> d & 1) & back & inside).astype(np.uint8) << d
    return out, int(sum(bin(int(v)).count('1') for v in (mask ^ out)[mask != out]))


def origin(q, r):
    return 2 * q, 2 * r + q % 2


def tile_hex(cx, cy):
    q = cx // 2
    return q, (cy - q % 2) // 2


def neighbour(q, r, d):
    dq, dr = DIRS[q % 2][d]
    return q + dq, r + dr


def direction(a, b):
    for d in range(6):
        if neighbour(*a, d) == tuple(b):
            return d
    return None


def bits(m):
    return [d for d in range(6) if m >> d & 1]


# ------------------------------------------------------------------ pieces
def mesh_dims(blob):
    g = 140 + 28 * struct.unpack_from('<I', blob, 8)[0]
    b = struct.unpack_from('<6f', blob, g + 24)
    return round((b[3] - b[0]) / 128), round((b[5] - b[2]) / 128)


def bc3_alpha_free(data, mip):
    from campaign_fine_tiles import bc3_rgba
    return bc3_rgba(data, mip).astype(np.float32) / 255


class Pieces:
    """Footprints and road images (blend0 road weight) of the roads_tracks pieces vanilla uses."""
    def __init__(self, read, names):
        self.dims, self.img = {}, {}
        for pc in sorted(names):
            pre = f'terrain/tiles/campaign/{SET}/{pc}/'
            try:
                dims = mesh_dims(read(pre + 'mesh.rigid_model_v2'))
                t = bc3_alpha_free(read(pre + 'blend0.dds'), 2)
            except KeyError:
                continue
            h, w = t.shape[:2]
            mh, mw = max(1, h // 8), max(1, w // 8)
            ring = np.concatenate([t[:mh].reshape(-1, 4), t[-mh:].reshape(-1, 4), t[:, :mw].reshape(-1, 4), t[:, -mw:].reshape(-1, 4)])
            self.img[pc] = 1 - t[..., int(ring.mean(0).argmax())]   # the channel keeping the ground below is not road
            self.dims[pc] = dims
        self.ports = {(pc, rt): None for pc in self.dims for rt in range(4)}   # pieces/turns available

    def footprint(self, pc, rt):
        w, h = self.dims[pc]
        return (h, w) if rt % 2 else (w, h)

    def weight(self, pc, rt, fx, fn):
        """Road weight at rotated-footprint points (fx cells from the west edge, fn from the north
        edge); same mapping as campaign_fine_tiles.FineTiles.composite."""
        img = self.img[pc]
        S = img.shape
        tw, th = self.dims[pc]
        fw, fh = self.footprint(pc, rt)
        u, v = ((fx, fn), (fn, fw - fx), (fw - fx, fh - fn), (fh - fn, fx))[rt]
        tu = np.clip(((u + 1) / (tw + 2) * S[1]).astype(int), 0, S[1] - 1)
        tv = np.clip(((th + 1 - v) / (th + 2) * S[0]).astype(int), 0, S[0] - 1)
        return img[tv, tu]

    def lane(self, pc, rt, X, Y, h, d):
        """Cell column (0 west, 1 east of hex h's block) where the piece's road crosses the top (d 0)
        or bottom (d 3) edge of that block: the stronger road weight just inside the edge."""
        fw, fh = self.footprint(pc, rt)
        bx, by = origin(*h)
        fn = (Y + fh) - (by + 2) + LANE_INSET if d == 0 else (Y + fh) - by - LANE_INSET
        s = np.linspace(-.35, .35, 8)
        w = [self.weight(pc, rt, bx - X + .5 + k + s, np.full(8, fn)).max() for k in (0, 1)]
        return int(w[1] > w[0])

    def placed_ports(self, pc, rt, X, Y, need):
        """[(hex, direction, lane or None)] for the external road edges `need` of a piece at (X, Y)."""
        return [(h, d, self.lane(pc, rt, X, Y, h, d) if d in (0, 3) else None) for h, d in sorted(need)]


def vertical_edge(a, d):
    """Key of the edge from hex a in direction 0 (up) or 3 (down): (lower hex, upper hex)."""
    b = neighbour(*a, d)
    return (tuple(a), b) if d == 0 else (b, tuple(a))


# ------------------------------------------------------------------ learning from vanilla
def learn(list_bytes, vcells, read):
    """Patterns, pieces and height spans of the vanilla dlc07 road layout."""
    from read_tile_list import parse
    paths, _, _, _, recs, _ = parse(list_bytes)
    mask = road_masks(vcells)
    H, W = mask.shape
    road = [(r, paths[r[1]].split('\\')[4]) for r in recs if '\\roads_' in paths[r[1]]]
    pieces = Pieces(read, {pc for _, pc in road})
    patterns = collections.defaultdict(collections.Counter)
    spans = collections.defaultdict(list)
    for r, pc in road:
        if pc not in pieces.dims or (r[5] & 0xf0) not in ROT:
            continue
        rt = ROT[r[5] & 0xf0]
        x, y = r[3], r[4]
        fw, fh = pieces.footprint(pc, rt)
        hexes = set()
        for cy in range(y, y + fh):
            for cx in range(x, x + fw):
                q, rr = tile_hex(cx, cy)
                if 0 <= q < W and 0 <= rr < H and mask[rr, q]:
                    hexes.add((q, rr))
        if not hexes or not all(x <= origin(*h)[0] and origin(*h)[0] + 2 <= x + fw and y <= origin(*h)[1] and origin(*h)[1] + 2 <= y + fh for h in hexes):
            continue
        key = (fw, fh, tuple(sorted((origin(*h)[0] - x, origin(*h)[1] - y, int(mask[h[1], h[0]])) for h in hexes)))
        patterns[key][pc, rt] += 1
        spans[len(hexes)].append(r[7] - r[6])
    span_max = {n: float(np.percentile(v, SPAN_QUANTILE)) for n, v in spans.items()}
    return patterns, pieces, span_max


def cache_shapes(native0, tail_paths):
    """{(piece, turn): (owned cells [(dx, dy)] relative to the record x/y, parent (dx, dy))}: the most
    common ownership of each road piece in the donor display cache (path values one-based)."""
    flat = native0.ravel()
    ow = native0.shape[1]
    active = flat['flags'] != 16
    own = np.where(flat['path'] >= 0, np.arange(flat.size), -flat['path'].astype(np.int64) - 1)
    road_values = np.array([i + 1 for i, p in enumerate(tail_paths) if '\\roads_' in p])
    parents = np.flatnonzero(active & (flat['path'] > 0) & np.isin(flat['path'], road_values))
    cells = np.flatnonzero(active & np.isin(own, parents))
    order = np.argsort(own[cells], kind='stable')
    cells = cells[order]
    o = own[cells]
    starts = np.searchsorted(o, parents)
    ends = np.searchsorted(o, parents, 'right')
    seen = collections.defaultdict(collections.Counter)
    for p, a, b in zip(parents, starts, ends):
        rec = flat[p]
        if (int(rec['flags']) & 0xf0) not in ROT:
            continue
        pc = tail_paths[int(rec['path']) - 1].split('\\')[4]
        x, y = int(rec['x']), int(rec['y'])
        cy, cx = np.divmod(cells[a:b], ow)
        py, px = divmod(int(p), ow)
        seen[pc, ROT[int(rec['flags']) & 0xf0]][(tuple(sorted(zip((cx - x).tolist(), (cy - y).tolist()))), (px - x, py - y))] += 1
    return {k: c.most_common(1)[0][0] for k, c in seen.items()}


def default_shape(fw, fh, rt):
    cells = tuple((dx, dy) for dx in range(fw) for dy in range(fh))
    parent = ((0, fh - 1), (fw - 1, fh - 1), (fw - 1, 0), (0, 0))[rt]   # NW / NE / SE / SW
    return cells, parent


# ------------------------------------------------------------------ layout
def _pick(options, q, r):
    """Deterministic mix of the options used about as often as the most used one."""
    top = max(c for _, c in options)
    viable = [o for o, c in options if c >= VARIETY * top]
    return viable[((q * 73856093) ^ (r * 19349663)) % len(viable)]


def layout(mask, target, free, road_cell, patterns, pieces, existing=(), span=None, span_max=None, existing_mask=None):
    """Pieces for the target hexes.

    mask (H, W) road edge masks, target (H, W) hexes to tile, free (TH, TW) cells a piece may take,
    road_cell (TH, TW) cells of any road hex, existing [(piece, turn, x, y)] road tiles kept on the
    map (their lanes bind the new pieces) and existing_mask the masks they were laid for (vanilla:
    a join bit added later has no port on the kept piece), span(x, y, fw, fh) -> LF height span of a
    footprint.
    Returns (placements [(piece, turn, x, y, fw, fh, hexes)], stats)."""
    H, W = mask.shape
    TH, TW = free.shape
    tr, tq = np.nonzero(target)
    tid = np.full((H, W), -1, np.int64)
    tid[tr, tq] = np.arange(len(tq))
    done = np.zeros(len(tq), bool)
    avail = free.copy()
    lanes = {}                      # vertical edge -> lane
    ports_at = {}                   # (hex, direction) -> lane of the piece covering hex
    stats = collections.Counter()
    placements = []

    def external(hexes, m=mask):
        hs = set(hexes)
        return {(h, d) for h in hexes for d in bits(m[h[1], h[0]]) if neighbour(*h, d) not in hs}

    em = mask if existing_mask is None else existing_mask

    kept_hexes = set()
    for pc, rt, x, y in existing:
        if (pc, rt) not in pieces.ports:
            continue
        fw, fh = pieces.footprint(pc, rt)
        hexes = {tile_hex(cx, cy) for cx in range(x, x + fw, 1) for cy in range(y, y + fh)}
        hexes = [h for h in hexes if 0 <= h[0] < W and 0 <= h[1] < H and em[h[1], h[0]]
                 and x <= origin(*h)[0] <= x + fw - 2 and y <= origin(*h)[1] <= y + fh - 2]
        kept_hexes.update(hexes)
        for a, d, lane in pieces.placed_ports(pc, rt, x, y, external(hexes, em)):
            ports_at[a, d] = lane
            if lane is not None:
                lanes[vertical_edge(a, d)] = lane

    def lane_ok(ports, strict=True):
        bad = 0
        for a, d, lane in ports:
            if lane is not None and d in (0, 3):
                k = vertical_edge(a, d)
                if k in lanes and lanes[k] != lane:
                    bad += 1
        return bad

    def accept(pc, rt, x, y, hexes, ports):
        fw, fh = pieces.footprint(pc, rt)
        for h in hexes:
            done[tid[h[1], h[0]]] = True
        avail[y:y + fh, x:x + fw] = False
        for a, d, lane in ports:
            ports_at[a, d] = lane
            if lane is not None and d in (0, 3):
                lanes[vertical_edge(a, d)] = lane
        placements.append((pc, rt, x, y, fw, fh, tuple(hexes)))

    # summed-area tables (static: free cells and road cells per footprint)
    def sat(a):
        s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
        s[1:, 1:] = a.astype(np.int64).cumsum(0).cumsum(1)
        return s
    S_free, S_road = sat(free), sat(road_cell)

    def box(s, x, y, w, h):
        return s[y + h, x + w] - s[y, x + w] - s[y + h, x] + s[y, x]

    # 1. multi-hex patterns
    ox, oy = origin(tq, tr)
    cands = []
    for key, opts in patterns.items():
        fw, fh, hx = key
        n = len(hx)
        total = sum(opts.values())
        opts = [(o, c) for o, c in opts.items() if o in pieces.ports]
        if n < 2 or total < MIN_PATTERN or not opts:
            continue
        dx0, dy0, m0 = hx[0]
        sel = np.flatnonzero(mask[tr, tq] == m0)
        X, Y = ox[sel] - dx0, oy[sel] - dy0
        ok = (X >= 0) & (Y >= 0) & (X + fw <= TW) & (Y + fh <= TH)
        hexes = []
        for dx, dy, m in hx:
            cx, cy = X + dx, Y + dy
            q, r = tile_hex(cx, cy)
            inside = (q >= 0) & (q < W) & (r >= 0) & (r < H)
            qq, rr = np.clip(q, 0, W - 1), np.clip(r, 0, H - 1)
            ok &= inside & (cx % 2 == 0) & (2 * rr + qq % 2 == cy) & (tid[rr, qq] >= 0) & (mask[rr, qq] == m)
            hexes.append((qq, rr))
        for i in np.flatnonzero(ok):
            x, y = int(X[i]), int(Y[i])
            if box(S_free, x, y, fw, fh) != fw * fh or box(S_road, x, y, fw, fh) != 4 * n:
                continue
            cands.append((-n, -total, y, x, key, [(int(q[i]), int(r[i])) for q, r in hexes], opts))
    cands.sort(key=lambda c: c[:4])
    for _, _, y, x, key, hexes, opts in cands:
        fw, fh, hx = key
        if any(done[tid[r, q]] for q, r in hexes) or not avail[y:y + fh, x:x + fw].all():
            continue
        if span and span_max and span(x, y, fw, fh) > span_max.get(len(hexes), np.inf):
            stats['multi_skipped_for_slope'] += 1
            continue
        need = external(hexes)
        good = []
        for (pc, rt), c in opts:
            ports = pieces.placed_ports(pc, rt, x, y, need)
            if lane_ok(ports):
                continue
            good.append(((pc, rt, ports), c))
        if not good:
            continue
        pc, rt, ports = _pick(good, *hexes[0])
        accept(pc, rt, x, y, hexes, ports)
        stats['multi_pieces'] += 1
        stats['multi_hexes'] += len(hexes)

    # 2./3. one 2x2 piece per remaining hex
    single = collections.defaultdict(list)           # mask -> [((piece, turn), lanes {0|3: lane}, weight)]
    used = collections.Counter()
    for (fw, fh, hx), opts in patterns.items():
        if len(hx) == 1 and (fw, fh) == (2, 2):
            for o, c in opts.items():
                used[hx[0][2], o] += c
    for (m, o), c in used.items():
        if c >= MIN_PATTERN and o in pieces.ports:
            need = {((0, 0), d) for d in bits(m)}
            single[m].append((o, {d: lane for _, d, lane in pieces.placed_ports(*o, 0, 0, need) if lane is not None}, c, m))

    inexact = collections.Counter()

    def constraints(q, r):
        out = {}
        for d in (0, 3):
            if mask[r, q] >> d & 1:
                k = vertical_edge((q, r), d)
                if k in lanes:
                    out[d] = lanes[k]
        return out

    def place_single(i):
        q, r = int(tq[i]), int(tr[i])
        x, y = origin(q, r)
        if not avail[y:y + 2, x:x + 2].all():
            stats['blocked_hexes'] += 1
            done[i] = True
            return
        m = int(mask[r, q])
        want = constraints(q, r)
        opts = single.get(m)
        if not opts:
            # closest vanilla mask: most shared directions, fewest extra or missing ones
            def score(mm):
                return 2 * bin(mm & m).count('1') - bin(mm ^ m).count('1')
            best = max(single, key=lambda mm: (score(mm), sum(c for _, _, c, _ in single[mm])))
            opts = single[best]
            stats['inexact_hexes'] += 1
            inexact[''.join(map(str, bits(m))) + '->' + ''.join(map(str, bits(best)))] += 1
        scored = [(sum(want.get(d) != ln.get(d) for d in want), o, w) for o, ln, w, _ in opts]
        least = min(s for s, _, _ in scored)
        if least:
            stats['lane_mismatch_edges'] += least
        pc, rt = _pick([(o, w) for s, o, w in scored if s == least], q, r)
        accept(pc, rt, x, y, [(q, r)], pieces.placed_ports(pc, rt, x, y, external([(q, r)])))
        stats['single_pieces'] += 1

    # breadth-first from hexes whose lanes are already bound
    pending = collections.deque(i for i in np.flatnonzero(~done) if constraints(int(tq[i]), int(tr[i])))
    queued = np.zeros(len(tq), bool)
    queued[list(pending)] = True
    rest = iter(np.flatnonzero(~done))
    while True:
        if not pending:
            i = next((j for j in rest if not done[j]), None)
            if i is None:
                break
            pending.append(i)
            queued[i] = True
        i = pending.popleft()
        if done[i]:
            continue
        place_single(i)
        q, r = int(tq[i]), int(tr[i])
        for d in bits(mask[r, q]):
            b = neighbour(q, r, d)
            if 0 <= b[0] < W and 0 <= b[1] < H:
                j = tid[b[1], b[0]]
                if j >= 0 and not done[j] and not queued[j]:
                    pending.append(j)
                    queued[j] = True
    # seams: road edges whose two sides do not both carry a port (or carry different lanes)
    covered = {h: i for i, p in enumerate(placements) for h in p[6]}
    for h in covered:
        for d in bits(mask[h[1], h[0]]):
            b = neighbour(*h, d)
            if b in covered and h > b:
                continue                                  # count new-new edges once
            if covered.get(b) == covered[h]:
                stats['edges_inside_pieces'] += 1
                continue
            here, there = (h, d) in ports_at, (b, (d + 3) % 6) in ports_at
            if not there and b not in covered:
                tiled = b in kept_hexes
                stats['edges_into_kept_tile_without_port' if tiled else 'edges_to_untiled_hexes'] += 1
                if tiled:
                    stats.setdefault('kept_tile_joins', []).append((h, d)) if len(stats.get('kept_tile_joins', [])) < 200 else None
            elif not (here and there):
                stats['seam_edges_unmatched_port'] += 1
            elif ports_at[h, d] != ports_at[b, (d + 3) % 6]:
                stats['seam_edges_lane'] += 1
            else:
                stats['edges_joined'] += 1
    stats = dict(stats)
    stats['inexact_by_mask'] = dict(inexact.most_common())
    stats['target_hexes'] = int(len(tq))
    stats['tiled_hexes'] = len(covered)
    return placements, stats


# ------------------------------------------------------------------ cache + list
def stamp(layer0, placements, shapes, tail_value, climate, lf):
    """Write the pieces into cache layer 0. lf: south-up LF (4 texels per tile cell, values /65535).
    Returns (owned cell mask, records [(path, x, y, flags, climate, (row0, row1), (col0, col1), (pr, pc))])."""
    TH, TW = layer0.shape
    cover = np.zeros((TH, TW), bool)
    out = []
    for pc, rt, x, y, fw, fh, _ in placements:
        cells, (px, py) = shapes.get((pc, rt)) or default_shape(fw, fh, rt)
        parent = (y + py) * TW + x + px
        cl = int(climate[y + fh // 2, x + fw // 2])
        block = lf[max(0, (y - 2) * 4):(y + fh + 2) * 4, max(0, (x - 2) * 4):(x + fw + 2) * 4]
        lo, hi = float(block.min()) / 65535, float(block.max()) / 65535
        for dx, dy in cells:
            c = layer0[y + dy, x + dx]
            is_parent = (dx, dy) == (px, py)
            c['path'] = tail_value(pc) if is_parent else -parent - 1
            c['flags'] = FLAGS[rt] if is_parent else CHILD_FLAGS
            c['x'], c['y'], c['climate'], c['extra'] = x, y, cl, 0
            c['lo'], c['hi'] = (lo, hi) if is_parent else (0, 0)
            layer0[y + dy, x + dx] = c
            cover[y + dy, x + dx] = True
        out.append((piece_path(pc), x, y, FLAGS[rt], cl, (y, y + fh - 1), (x, x + fw - 1), (y + py, x + px)))
    return cover, out


def split_floor_cells(before, after):
    """Cells of `after` (floor mask) whose aligned 2x2 floor group was whole in `before` but not in
    `after`: the 2x2 floor layout would leave them bare."""
    def whole(m):
        TH, TW = m.shape
        p = np.zeros((TH + TH % 2, TW + TW % 2), bool)
        p[:TH, :TW] = m
        g = p.reshape(p.shape[0] // 2, 2, p.shape[1] // 2, 2).all(axis=(1, 3))
        return np.repeat(np.repeat(g, 2, 0), 2, 1)[:TH, :TW]
    return after & whole(before) & ~whole(after)


def stamp_singles(layer0, cells, one_value, climate, lf):
    """1-cell generic_1x1 owners on the given cells (vanilla cache style)."""
    ys, xs = np.nonzero(cells)
    for y, x in zip(ys.tolist(), xs.tolist()):
        block = lf[max(0, (y - 2) * 4):(y + 3) * 4, max(0, (x - 2) * 4):(x + 3) * 4]
        c = layer0[y, x]
        c['path'], c['flags'], c['x'], c['y'] = one_value, SINGLE_FLAGS, x, y
        c['climate'], c['extra'] = int(climate[y, x]), 0
        c['lo'], c['hi'] = float(block.min()) / 65535, float(block.max()) / 65535
        layer0[y, x] = c
    return len(ys)


def existing_road_tiles(layer0, tail_paths):
    """[(piece, turn, x, y)] of road owners in a cache layer (canvas frame, path values one-based)."""
    flat = layer0.ravel()
    act = (flat['flags'] != 16) & (flat['path'] > 0)
    road_values = np.array([i + 1 for i, p in enumerate(tail_paths) if '\\roads_' in p])
    out = []
    for p in np.flatnonzero(act & np.isin(flat['path'], road_values)):
        rec = flat[p]
        f = int(rec['flags']) & 0xf0
        if f in ROT:
            out.append((tail_paths[int(rec['path']) - 1].split('\\')[4], ROT[f], int(rec['x']), int(rec['y'])))
    return out


def native_join_blocks(target_hex, layers, tail_paths):
    """Tile cells of target road hexes on kept native ground whose owners are all generic land floors
    (layer 0) with nothing on layers 1-2: added to the intended cache edit so the floors there are
    re-laid around the road pieces. Returns (cells mask, hexes added, hexes refused)."""
    TH, TW = layers[0].shape
    generic = np.zeros(len(tail_paths) + 1, bool)
    for i, p in enumerate(tail_paths):
        generic[i + 1] = '\\generic\\' in p and 'generic_sea' not in p
    flat = layers[0].ravel()
    own = np.where(flat['path'] >= 0, np.arange(flat.size), -flat['path'].astype(np.int64) - 1)
    pv = flat['path'][own]
    gen_cell = ((flat['flags'] == 16) | (generic[np.clip(pv, 0, len(tail_paths))] & (pv > 0))).reshape(TH, TW)
    for L in layers[1:]:
        gen_cell &= L['flags'] == 16
    r, q = np.nonzero(target_hex)
    x, y = origin(q, r)
    ok = (y + 2 <= TH) & (x + 2 <= TW)
    x, y, q, r = x[ok], y[ok], q[ok], r[ok]
    good = gen_cell[y, x] & gen_cell[y + 1, x] & gen_cell[y, x + 1] & gen_cell[y + 1, x + 1]
    out = np.zeros((TH, TW), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            out[y[good] + dy, x[good] + dx] = True
    return out, int(good.sum()), int((~good).sum())


def road_cells(mask_hex, TH, TW):
    """(TH, TW) cells belonging to a road hex block."""
    H, W = mask_hex.shape
    yy, xx = np.mgrid[:TH, :TW]
    q, r = tile_hex(xx, yy)
    inside = (q < W) & (r >= 0) & (r < H)
    return inside & mask_hex[np.clip(r, 0, H - 1), np.clip(q, 0, W - 1)].astype(bool)


def list_records(placed, list_paths, lf):
    """tile_list rows (1, path index, climate, x, y, flags, lo, hi) for stamped pieces, lo/hi the LF
    extrema over the footprint plus two cells (the vanilla road record formula)."""
    out = []
    for path, x, y, fl, cl, rows, cols, _ in placed:
        b = lf[max(0, (rows[0] - 2) * 4):(rows[1] + 3) * 4, max(0, (cols[0] - 2) * 4):(cols[1] + 3) * 4]
        out.append((1, list_paths.index(path), cl, x, y, fl, float(b.min()) / 65535, float(b.max()) / 65535))
    return out


def blocks_in(cell_mask, H, W):
    """(H, W) hexes whose whole 2x2 tile block lies in cell_mask."""
    TH, TW = cell_mask.shape
    q, r = np.meshgrid(np.arange(W), np.arange(H))
    x, y = origin(q, r)
    ok = (y + 1 < TH) & (x + 1 < TW)
    x, y = np.clip(x, 0, TW - 2), np.clip(y, 0, TH - 2)
    return ok & cell_mask[y, x] & cell_mask[y + 1, x] & cell_mask[y, x + 1] & cell_mask[y + 1, x + 1]


def south_up_lf(dds):
    h, w = struct.unpack_from('<2I', dds, 12)
    return np.frombuffer(dds, '<u2', offset=128).reshape(h, w)[::-1]


def add_records(src, placed, lf):
    """tile_list.bin bytes with the road records appended (missing piece paths added to its table)."""
    from read_tile_list import parse
    import salt_pools
    need = sorted({p[0] for p in placed} - set(parse(src)[0]))
    if need:
        src = salt_pools.add_list_paths(src, need)
    pp, _, cp, start, rs, _ = parse(src)
    add = list_records(placed, pp, lf)
    hd = bytearray(src[:start])
    struct.pack_into('<I', hd, cp, len(rs) + len(add))
    return bytes(hd) + src[start:start + 21 * len(rs)] + b''.join(struct.pack('<HIBhhHff', *r) for r in add)

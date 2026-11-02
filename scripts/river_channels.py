"""Water outline of the frontier rivers for polish_campaign_terrain.py RIVER CHANNELS.

The channel used to be carved by a 4-neighbour step distance from the river hexes on the tile grid.
A hex there is a 2x2 cell brick and odd columns sit one cell higher, so the water edge followed
the brick outline: a regular half-hex sawtooth on both banks and the same width everywhere
(offline r14: Amur, Mekong, Funan, Northeast). Here the edge is a smooth curve:

  * the river hex centres are spread with a Gaussian (SIGMA hexes); a chain of centres gives a
    ridge whose profile is a Gaussian of the distance to the chain's smoothed centre line, so
    that distance is read back from the density (the half-hex zigzag of a hex chain averages out);
  * the half-width along the river grows with the river hexes upstream (flow accumulation over the
    hex graph, from the mouths) and varies with low-frequency noise;
  * every river hex centre stays under water (WET_R), and a land hex the outline would flood keeps
    a dry disc (LAND_R), so the map's water and land hexes read as such;
  * pieces of one river cut apart (apply_rivers.py routes the rivers round the towns, as vanilla does;
    any other cut) are joined by a channel across the land between them (GAP_MAX), and a piece that ends
    on land near the coast
    (Natural Earth centre lines end at the real coast, inside the canvas coast) runs on to the
    open water (MOUTH_MAX). These links are visual only: their hexes stay land in map.hex.

All loops run in a fixed order (sorted hexes, fixed neighbour order); no randomness but the seeded
terrain noise, so the output is deterministic.
"""
import collections
import math

import numpy as np
from scipy import ndimage

import campaign_world as cw
from hex_components import hex_dilate
from revealed_filler import river_canvas_hex
from native_campaign_climate import box_blur
from gpu_noise import fbm   # terrain_noise.fbm, same bytes (GPU for grid inputs, CPU fallback)

CELL = (cw.DX / 2, cw.DZ / 2)   # world units per tile cell (x, z)
HEX = cw.DZ                      # world units between neighbouring hex centres
SIGMA = .8          # hexes: Gaussian over the river hex centres
HW_MIN = .65        # hexes: water half-width at a source
HW_MAX = 1.2        # hexes: half-width of a large river (bank hex centres lie 1 hex out)
ACC_FULL = 1500     # river hexes upstream at which the half-width reaches HW_MAX (log scale)
HW_NOISE = .35      # relative half-width change by the noise
NOISE_SCALE = 12    # hexes: largest feature of that noise
NOISE_SEED = 131
HW_BLUR = 6         # tile cells: the half-width is averaged along the river over this box
WET_R = .4          # hexes: every river hex centre has water within this radius
LAND_R = .45        # hexes: dry disc kept round a land hex centre the outline would flood ...
LAND_GEO = 6        # ... when two of its wet neighbours are more than this many river steps apart
GAP_MAX = 8         # hexes of land between two pieces of one river that are joined
MOUTH_MAX = 28      # hexes of land a piece ending near open water runs on to reach it
MOUTH_COS = .5      # the way to the water must continue the river's direction (cosine)
MOUTH_HEIGHT = 1.5  # world units: a mouth link starts from an end at most this far above the water
TAPER = 6           # hexes along the river over which an end on land narrows ...
TAPER_MIN = .6      # ... to this share of the half-width
TANGENT_STEPS = 4   # river hexes back from an end that give its direction
# Endorheic rivers (apply_rivers WR_BASIN: the Tarim, Bosten / Lop Nur, Shule, Balkhash and Aral basins;
# region names tw3k_riv_sea_<key>_<n>). One that reaches its lake has a mouth there (the level eases to the
# lake as at a sea mouth); one without any mouth dies out in the sand: its lowest end narrows to SAND_MIN
# of the half-width over SAND_TAPER hexes (at the end the WET_R disc is all that is left).
ENDORHEIC = frozenset({'tarim', 'konqi', 'kaidu', 'yarkand', 'aksu', 'khotan', 'cherchen', 'shule', 'dang', 'ili',
                       'syr_darya', 'amu_darya', 'balkhash', 'aral'})
SAND_TAPER = 30     # hexes along the river over which an end in the sand narrows ...
SAND_MIN = .05      # ... to this share of the half-width; the wet disc round each river hex centre (WET_R)
                    # shrinks with it, so the water thins out to a thread (offline r20: the Taitema tail
                    # stayed a straight full-width channel to its last hex, SAND_TAPER 12 / SAND_MIN .15 with
                    # the full WET_R disc left at the end)
# Arid (endorheic) rivers are narrow: vanilla-like water a hex wide or less, widening only with a large flow
# (offline r20: the Shule north of Dunhuang one straight 2-hex channel, half-width up to HW_MAX like a big river).
HW_MIN_ARID = .35   # hexes: half-width of an arid river at a source ...
HW_MAX_ARID = .7    # ... and of the largest (Amu / Syr Darya, lower Tarim)
ACC_FULL_ARID = 3000  # river hexes upstream at which an arid river reaches HW_MAX_ARID (log scale)
ARID_WET_R = .6     # share of an arid half-width the wet disc round its hex centres may take (the noise narrows
                    # the channel by up to HW_NOISE)


def neighbours(q, r):
    """Six hex neighbours, as hex_components.hex_dilate (odd columns: (q±1, r+1)), in circular
    order N, NE, SE, S, SW, NW (rows run south-up; odd columns sit half a hex higher)."""
    o1, o0 = (1, 0) if q % 2 else (0, -1)
    return ((q, r + 1), (q + 1, r + o1), (q + 1, r + o0), (q, r - 1), (q - 1, r + o0), (q - 1, r + o1))


def _is_end(h, inpiece):
    """A hex is a river end when it has at most one neighbour in its piece, or two touching ones
    (X, Y) whose further neighbours in the piece lie on one side only (a tip, not a bend of a
    zigzag chain). inpiece(q, r): membership test."""
    nb = [n for n in neighbours(*h) if inpiece(*n)]
    if len(nb) <= 1:
        return True
    if len(nb) > 2 or nb[1] not in neighbours(*nb[0]):
        return False
    skip = {h, nb[0], nb[1]}
    rest = sorted({m for x in nb for m in neighbours(*x) if m not in skip and inpiece(*m)})
    if not rest:
        return True
    seen = {rest[0]}; todo = [rest[0]]
    while todo:
        a = todo.pop()
        for m in neighbours(*a):
            if m in rest and m not in seen:
                seen.add(m); todo.append(m)
    return len(seen) == len(rest)


def _neighbour_shifts(mask):
    """Six (H, W) bool arrays: mask at each hex neighbour (False off the map, no wraparound)."""
    m = np.asarray(mask, bool)
    W = m.shape[1]
    odd = (np.arange(W) % 2 == 1)[None, :]
    up = np.zeros_like(m); up[:-1] = m[1:]       # m[r+1, q]
    dn = np.zeros_like(m); dn[1:] = m[:-1]       # m[r-1, q]
    out = [up, dn]
    for dq in (1, -1):
        same = np.zeros_like(m); other = np.zeros_like(m)
        if dq == 1:
            same[:, :-1] = m[:, 1:]; other[:, :-1] = np.where(odd[:, :-1], up[:, 1:], dn[:, 1:])
        else:
            same[:, 1:] = m[:, :-1]; other[:, 1:] = np.where(odd[:, 1:], up[:, :-1], dn[:, :-1])
        out += [same, other]
    return out


def hex_label(mask):
    """Labels of the hex-connected pieces of mask (doubled-row grid; as apply_rivers.hex_label)."""
    H, W = mask.shape
    q = np.arange(W)
    rows = 2 * np.arange(H)[:, None] + q[None, :] % 2
    grid = np.zeros((2 * H + 1, W), bool)
    grid[rows, q[None, :]] = mask
    grid[rows + 1, q[None, :]] = mask
    return ndimage.label(grid)[0][rows, q[None, :]] * mask


def centre(q, r):
    """World (x, z) of a hex centre on the tile grid: the middle of its 2x2 cell brick."""
    return (2 * q + 1) * CELL[0], (2 * r + q % 2 + 1) * CELL[1]


def _bfs(starts, passable, goal, limit, H, W):
    """Shortest hex path (list of hexes, starts excluded) from any start through passable hexes to
    the first hex meeting goal(q, r), at most limit steps; None if there is none."""
    prev = {s: None for s in starts}
    frontier = sorted(starts)
    for _ in range(limit):
        nxt = []
        for h in frontier:
            for n in neighbours(*h):
                q, r = n
                if n in prev or not (0 <= q < W and 0 <= r < H) or not passable[r, q]:
                    continue
                prev[n] = h
                if goal(q, r):
                    path = [n]
                    while prev[path[-1]] is not None and prev[prev[path[-1]]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                nxt.append(n)
        frontier = nxt
        if not frontier:
            break
    return None


def hex_distance(a, b):
    """Hex steps between (q, r) a and b (odd columns half a hex higher)."""
    def cube(q, r):
        z = r - (q - (q & 1)) // 2
        return q, z
    (x1, z1), (x2, z2) = cube(*a), cube(*b)
    dx, dz = x2 - x1, z2 - z1
    return max(abs(dx), abs(dz), abs(dx + dz))


def plan_links(river, family, open_water, allowed, height, water_level, water_level_hex=None):
    """Visual channels over land: (H, W) bool mask of link hexes and a report.

    river: river hexes; family: (H, W) int river key per river hex (-1 elsewhere); open_water: sea,
    lakes; allowed: land hexes a link may cross; height: ground height per hex (world units). A
    mouth link starts only from an end at most MOUTH_HEIGHT above water_level: a lowland river end,
    not a mountain source that points at a lake (Yarlung Tsangpo and Manasarovar, prototype r15).
    (Ends within 7 hexes of a town were skipped as town keep-out cuts; apply_rivers now routes the rivers
    round the towns, so a river no longer ends at a town it should run past.)
    water_level_hex: (H, W) water level per hex in place of water_level (polish LAKE LEVEL: a lake with a level of
    its own); None: water_level everywhere, as before."""
    H, W = river.shape
    lab = hex_label(river)
    n_pieces = int(lab.max())
    link = np.zeros((H, W), bool)
    report = {'gaps': [], 'mouths': []}
    if not n_pieces:
        return link, report
    has_mouth = np.zeros(n_pieces + 1, bool)
    has_mouth[np.unique(lab[hex_dilate(open_water) & river])] = True
    ends = collections.defaultdict(list)   # piece -> its end hexes
    for r, q in zip(*np.nonzero(river)):
        k = lab[r, q]
        if _is_end((int(q), int(r)), lambda a, b, k=k: 0 <= a < W and 0 <= b < H and lab[b, a] == k):
            ends[int(k)].append((int(q), int(r)))
    # Gaps: every end's shortest way over land to another piece of its river; per pair of pieces
    # only the shortest is laid (two distributaries are joined where they split, not at their mouths).
    passable = allowed & ~river
    cand = []
    for k in sorted(ends):
        for e in sorted(ends[k]):
            fam = family[e[1], e[0]]

            def goal(q, r, k=k, fam=fam):
                return any(0 <= a < W and 0 <= b < H and lab[b, a] not in (0, k) and family[b, a] == fam
                           for a, b in neighbours(q, r))
            path = _bfs([e], passable, goal, GAP_MAX, H, W)
            if path is None:
                continue
            tq, tr = path[-1]
            other = min(int(lab[b, a]) for a, b in neighbours(tq, tr)
                        if 0 <= a < W and 0 <= b < H and lab[b, a] not in (0, k) and family[b, a] == fam)
            cand.append((len(path), e, (min(k, other), max(k, other)), path))
    used = set()
    pairs = set()
    for n, e, pair, path in sorted(cand):
        if pair in pairs:
            continue
        pairs.add(pair)
        used.add(e)
        for q, r in path:
            link[r, q] = True
        report['gaps'].append({'from': list(e), 'to_piece_near': list(path[-1]), 'hexes': n})
    # Mouths: a piece without open water at either end runs on from a free end whose direction
    # leads to the water.
    wet_goal = hex_dilate(open_water)
    for k in sorted(ends):
        if has_mouth[k]:
            continue
        best = None
        for e in sorted(ends[k]):
            if any(hex_distance(e, u) <= 2 for u in used) or height[e[1], e[0]] > (water_level if water_level_hex is None else water_level_hex[e[1], e[0]]) + MOUTH_HEIGHT:
                continue
            path = _bfs([e], passable & ~link, lambda q, r: wet_goal[r, q], MOUTH_MAX, H, W)
            if path is None:
                continue
            # direction of the river at this end: from the piece hexes TANGENT_STEPS back
            back = _piece_back(lab, k, e, TANGENT_STEPS, H, W)
            ex, ez = centre(*e)
            bx, bz = centre(*back)
            tx, tz = centre(*path[-1])
            u = np.array([ex - bx, ez - bz]); v = np.array([tx - ex, tz - ez])
            if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
                continue
            cos = float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
            if cos < MOUTH_COS:
                continue
            if best is None or len(path) < len(best[1]):
                best = (e, path, cos)
        if best is not None:
            e, path, cos = best
            for q, r in path:
                link[r, q] = True
            report['mouths'].append({'from': list(e), 'to': list(path[-1]), 'hexes': len(path), 'cos': round(cos, 3),
                                     'end_height': round(float(height[e[1], e[0]]), 3)})
    report['link_hexes'] = int(link.sum())
    return link, report


def _piece_back(lab, k, e, steps, H, W):
    """The piece hex `steps` hex steps from end e inside piece k (the farthest reached if shorter)."""
    seen = {e}
    frontier = [e]
    last = e
    for _ in range(steps):
        nxt = []
        for h in frontier:
            for n in neighbours(*h):
                q, r = n
                if n not in seen and 0 <= q < W and 0 <= r < H and lab[r, q] == k:
                    seen.add(n); nxt.append(n)
        if not nxt:
            break
        frontier = sorted(nxt)
        last = frontier[0]
    return last


def flow_accumulation(wet, mouths, height):
    """River hexes upstream of each wet hex (itself included), over the hex graph of wet. Roots are
    the mouth hexes; a piece without one drains to its lowest hex (height, then position)."""
    H, W = wet.shape
    lab = hex_label(wet)
    dist = {}
    roots = []
    for k in range(1, int(lab.max()) + 1):
        ys, xs = np.nonzero((lab == k) & mouths)
        if not len(ys):
            ys, xs = np.nonzero(lab == k)
            i = min(range(len(ys)), key=lambda j: (float(height[ys[j], xs[j]]), int(ys[j]), int(xs[j])))
            ys, xs = ys[i:i + 1], xs[i:i + 1]
        roots += [(int(q), int(r)) for r, q in zip(ys, xs)]
    frontier = sorted(roots)
    for h in frontier:
        dist[h] = 0
    d = 0
    while frontier:
        d += 1
        nxt = []
        for h in frontier:
            for n in neighbours(*h):
                q, r = n
                if 0 <= q < W and 0 <= r < H and wet[r, q] and n not in dist:
                    dist[n] = d; nxt.append(n)
        frontier = sorted(nxt)
    acc = {h: 1.0 for h in dist}
    for h in sorted(dist, key=lambda h: (-dist[h], h[1], h[0])):
        if dist[h] == 0:
            continue
        down = min((n for n in neighbours(*h) if dist.get(n, -1) == dist[h] - 1), key=lambda n: (n[1], n[0]))
        acc[down] += acc[h]
    out = np.zeros((H, W), np.float64)
    for (q, r), v in acc.items():
        out[r, q] = v
    return out


def drawn_hexes(shape, channels_path=None):
    """(H, W) bool: the drawn-only river hexes of apply_rivers' river-channels.json ('drawn', river spec option B):
    land hexes a river crosses inside one region, carved and watered as visual links. None/absent file: none."""
    import json
    from pathlib import Path
    out = np.zeros(shape, bool)
    if channels_path is None or not Path(channels_path).exists():
        return out
    doc = json.loads(Path(channels_path).read_text(encoding='utf-8'))
    H, W = shape
    for k in sorted(doc.get('drawn', {})):
        qr = np.asarray(doc['drawn'][k], np.int64).reshape(-1, 2)
        ok = (qr[:, 0] >= 0) & (qr[:, 0] < W) & (qr[:, 1] >= 0) & (qr[:, 1] < H)
        out[qr[ok, 1], qr[ok, 0]] = True
    return out


def river_families(names, ids, channels_path=None, prefix='tw3k_riv_sea_'):
    """(fam, famh): channel key -> int (keys in name order) and (H, W) int32 key per frontier river hex
    (-1 elsewhere), as polish plans the river channels by. With apply_rivers' river-channels.json beside
    map.hex (channels_path; river regions merge several channels) the keys come from it; without it (maps
    from before the merge) from the region names tw3k_riv_sea_<key>_<n>. Every hex of a river region must
    have a channel in the file (a stale file raises); file hexes no longer in a river region are ignored."""
    import json
    from pathlib import Path
    rk = [i for i, k in enumerate(names) if k.startswith(prefix)]
    famh = np.full(ids.shape, -1, np.int32)
    river = np.isin(ids, rk)
    if channels_path is not None and Path(channels_path).exists():
        doc = json.loads(Path(channels_path).read_text(encoding='utf-8'))
        fam = {k: i for i, k in enumerate(sorted(doc['channels']))}
        H, W = ids.shape
        for k in sorted(doc['channels']):
            qr = np.asarray(doc['channels'][k], np.int64).reshape(-1, 2)
            ok = (qr[:, 0] >= 0) & (qr[:, 0] < W) & (qr[:, 1] >= 0) & (qr[:, 1] < H)
            q, r = qr[ok, 0], qr[ok, 1]
            on = river[r, q]
            famh[r[on], q[on]] = fam[k]
        # drawn-only hexes (river spec option B: land a river crosses inside one region) take their channel key
        for k in sorted(doc.get('drawn', {})):
            qr = np.asarray(doc['drawn'][k], np.int64).reshape(-1, 2)
            ok = (qr[:, 0] >= 0) & (qr[:, 0] < W) & (qr[:, 1] >= 0) & (qr[:, 1] < H)
            if k in fam:
                famh[qr[ok, 1], qr[ok, 0]] = fam[k]
        missing = river & (famh < 0)
        if missing.any():
            raise RuntimeError(f'{channels_path}: {int(missing.sum())} river-region hexes have no channel '
                               '(stale river-channels.json: rerun apply_rivers)')
        return fam, famh
    fam = {k: i for i, k in enumerate(sorted({names[i][len(prefix):].rsplit('_', 1)[0] for i in rk}))}
    for i in rk:
        famh[ids == i] = fam[names[i][len(prefix):].rsplit('_', 1)[0]]
    return fam, famh


def sand_ends(wet, mouths, height, family, endorheic):
    """(H, W) bool: the end hex where an endorheic river dies out in the sand. Per piece of wet (river and
    link hexes) without a mouth hex that holds a river hex of an endorheic family (family: int key per
    river hex, -1 elsewhere; endorheic: the keys), its lowest end hex (height, then row, column). Pieces
    of other rivers (cut at the native guard, say) have no sand end."""
    H, W = wet.shape
    out = np.zeros((H, W), bool)
    if not endorheic:
        return out
    lab = hex_label(wet)
    keys = np.array(sorted(endorheic), np.int64)
    has_mouth = np.zeros(int(lab.max()) + 1, bool); has_mouth[np.unique(lab[mouths & wet])] = True
    endo = np.zeros(int(lab.max()) + 1, bool); endo[np.unique(lab[wet & np.isin(family, keys)])] = True
    best = {}
    for r, q in zip(*np.nonzero(wet)):
        k = int(lab[r, q])
        if has_mouth[k] or not endo[k]:
            continue
        if _is_end((int(q), int(r)), lambda a, b, k=k: 0 <= a < W and 0 <= b < H and lab[b, a] == k):
            key = (float(height[r, q]), int(r), int(q))
            if k not in best or key < best[k]:
                best[k] = key
    for k in sorted(best):
        out[best[k][1], best[k][2]] = True
    return out


def sand_taper(wet, sand):
    """Half-width factor per wet hex towards the ends where a river dies out in the sand (sand_ends):
    SAND_MIN at the end hex, rising smoothly to 1 over SAND_TAPER hexes along the river; 1 elsewhere."""
    H, W = wet.shape
    out = np.ones((H, W), np.float64)
    if sand is None or not sand.any():
        return out
    dist = {(int(q), int(r)): 0 for r, q in zip(*np.nonzero(sand))}
    frontier = sorted(dist)
    for d in range(1, SAND_TAPER + 1):
        nxt = []
        for h in frontier:
            for n in neighbours(*h):
                if n not in dist and 0 <= n[0] < W and 0 <= n[1] < H and wet[n[1], n[0]]:
                    dist[n] = d; nxt.append(n)
        frontier = sorted(nxt)
    for (q, r), d in dist.items():
        t = d / SAND_TAPER
        out[r, q] = SAND_MIN + (1 - SAND_MIN) * t * t * (3 - 2 * t)
    return out


def end_taper(wet, open_water, sand=None):
    """Half-width factor per wet hex: TAPER_MIN at a river end on land (a source, or an end the data
    cuts), rising to 1 over TAPER hexes along the river; 1 elsewhere. Without it a piece drained to
    the wrong end ended at full width, as a canal (offline r14 Amur window: the Hailar source).
    sand: end hexes where a river dies out in the sand (sand_ends): SAND_MIN there, rising to 1 over
    SAND_TAPER hexes, smoothly (sand_taper; the lower factor wins)."""
    H, W = wet.shape
    lab = hex_label(wet)
    mouth = hex_dilate(open_water)
    dist = {}
    for r, q in zip(*np.nonzero(wet & ~mouth)):
        k = lab[r, q]
        if _is_end((int(q), int(r)), lambda a, b, k=k: 0 <= a < W and 0 <= b < H and lab[b, a] == k):
            dist[(int(q), int(r))] = 0
    frontier = sorted(dist)
    for d in range(1, TAPER + 1):
        nxt = []
        for h in frontier:
            for n in neighbours(*h):
                if n not in dist and 0 <= n[0] < W and 0 <= n[1] < H and wet[n[1], n[0]]:
                    dist[n] = d; nxt.append(n)
        frontier = sorted(nxt)
    out = np.ones((H, W), np.float64)
    for (q, r), d in dist.items():
        out[r, q] = TAPER_MIN + (1 - TAPER_MIN) * d / TAPER
    return np.minimum(out, sand_taper(wet, sand))


def half_width(acc, arid=None):
    """Water half-width (hexes) from the river hexes upstream; arid (endorheic) river hexes (arid: (H, W)
    bool) narrower, by their own scale."""
    t = np.clip(np.log1p(acc) / np.log1p(ACC_FULL), 0, 1)
    hw = HW_MIN + (HW_MAX - HW_MIN) * t
    if arid is not None:
        ta = np.clip(np.log1p(acc) / np.log1p(ACC_FULL_ARID), 0, 1)
        hw = np.where(arid, HW_MIN_ARID + (HW_MAX_ARID - HW_MIN_ARID) * ta, hw)
    return hw


def _bricks(mask):
    """Tile cells (rows, cols) of the 2x2 bricks of the hexes in mask, and the hex (r, q) of each."""
    r, q = np.nonzero(mask)
    y0 = 2 * r + q % 2; x0 = 2 * q
    ys = np.concatenate([y0, y0, y0 + 1, y0 + 1]); xs = np.concatenate([x0, x0 + 1, x0, x0 + 1])
    return ys, xs, np.tile(r, 4), np.tile(q, 4)


def _disc_distance(mask, shape, reach=2):
    """World distance from tile cells to the nearest centre of a hex in mask, within the
    (2*reach) x (2*reach) cells round each centre (inf elsewhere)."""
    out = np.full(shape, np.inf)
    r, q = np.nonzero(mask)
    if not len(r):
        return out
    cx, cz = (2 * q + 1) * CELL[0], (2 * r + q % 2 + 1) * CELL[1]
    for dy in range(-reach, reach):
        for dx in range(-reach, reach):
            ys = 2 * r + q % 2 + 1 + dy; xs = 2 * q + 1 + dx
            ok = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
            d = np.hypot((xs[ok] + .5) * CELL[0] - cx[ok], (ys[ok] + .5) * CELL[1] - cz[ok])
            np.minimum.at(out, (ys[ok], xs[ok]), d)
    return out


def _points_distance(px, pz, shape, reach=2):
    """World distance from tile cells to the nearest of the world points (px, pz), within the
    (2*reach+1) x (2*reach+1) cells round each point's cell (inf elsewhere)."""
    out = np.full(shape, np.inf)
    px = np.asarray(px, np.float64); pz = np.asarray(pz, np.float64)
    if not len(px):
        return out
    ix = np.floor(px / CELL[0]).astype(np.int64); iz = np.floor(pz / CELL[1]).astype(np.int64)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            ys = iz + dy; xs = ix + dx
            ok = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
            d = np.hypot((xs[ok] + .5) * CELL[0] - px[ok], (ys[ok] + .5) * CELL[1] - pz[ok])
            np.minimum.at(out, (ys[ok], xs[ok]), d)
    return out


def channel_distance(wet, ring, hw_hex, land, q, r, noise_origin, wet_r=None):
    """Signed distance (tile cells, + onto the bank) from the river water edge.

    wet: river and link hexes; ring: open water hexes beside a mouth (they keep the channel full
    width into the sea); hw_hex: half-width (hexes) per wet hex; land: land hexes that must stay dry;
    q, r: tile-grid hex coordinates (x/2, y/2) for the noise; wet_r: (H, W) wet disc radius (hexes) per wet
    hex, WET_R where None (rounded to .02 hexes)."""
    shape = q.shape
    rho = np.zeros(shape, np.float64)
    ys, xs, _, _ = _bricks(wet | ring)
    rho[ys, xs] = .25                           # each hex centre spread over its brick
    sig = SIGMA * HEX
    rho = ndimage.gaussian_filter(rho, (sig / CELL[1], sig / CELL[0]), mode='constant', truncate=4.)
    rho0 = CELL[0] * CELL[1] / (HEX * math.sqrt(2 * math.pi) * sig)   # on a straight chain's centre line
    with np.errstate(divide='ignore'):
        d = sig * np.sqrt(2 * np.maximum(np.log(rho0 / np.maximum(rho, 1e-300)), 0))
    # half-width: averaged along the river from its hexes, times low-frequency noise
    ys, xs, hr, hq = _bricks(wet)
    num = np.zeros(shape, np.float32); den = np.zeros(shape, np.float32)
    num[ys, xs] = hw_hex[hr, hq]; den[ys, xs] = 1
    hw = box_blur(num, HW_BLUR).astype(np.float64) / np.maximum(box_blur(den, HW_BLUR), 1e-3)
    hw *= 1 + HW_NOISE * fbm(q, r, noise_origin, NOISE_SCALE, 3, seed=NOISE_SEED).astype(np.float64)
    s = d - hw * HEX
    del d, hw
    if wet_r is None:
        s = np.minimum(s, _disc_distance(wet & ~ring, shape) - WET_R * HEX)
    else:
        rad = np.round(np.asarray(wet_r, np.float64) / .02) * .02
        for v in np.unique(rad[wet & ~ring]):
            s = np.minimum(s, _disc_distance(wet & ~ring & (rad == v), shape) - float(v) * HEX)
    # A land hex between two parts of the river (two of its wet neighbours more than LAND_GEO river
    # steps apart, not counting the way through the hex itself) whose centre the outline floods keeps
    # a dry disc. A bank hex in a bend or step of the chain may lie half under water, as with the old
    # brick outline: dry discs there notched the bank at every step (prototype r15).
    ys, xs, hr, hq = _bricks(land)
    flooded = np.zeros(land.shape, np.float64)
    np.add.at(flooded, (hr, hq), (s[ys, xs] < 0).astype(np.float64))
    near = np.zeros(land.shape, bool)
    for sh in _neighbour_shifts(wet):
        near |= sh
    drown = np.zeros(land.shape, bool)
    H, W = land.shape
    for r, q in zip(*np.nonzero(land & (flooded >= 2) & near)):
        x = (int(q), int(r))
        nb = [n for n in neighbours(*x) if 0 <= n[0] < W and 0 <= n[1] < H and wet[n[1], n[0]]]
        if len(nb) < 2:
            continue
        seen = {nb[0]: 0}; frontier = [nb[0]]
        for d in range(1, LAND_GEO + 1):
            nxt = []
            for h in frontier:
                for n in neighbours(*h):
                    if n not in seen and n != x and 0 <= n[0] < W and 0 <= n[1] < H and wet[n[1], n[0]]:
                        seen[n] = d; nxt.append(n)
            frontier = sorted(nxt)
        if any(n not in seen for n in nb[1:]):
            drown[r, q] = True
    if drown.any():
        s = np.maximum(s, LAND_R * HEX - _disc_distance(drown, shape))
    return (s / (HEX / 2)).astype(np.float32), int(drown.sum())


MOUTH_BLOCK = 512   # tile cells: mouth_ease works in blocks of this side holding river cells


def mouth_ease(open_water, near, reach):
    """Share of the river level kept above the water plane near open water: smoothstep of the
    Euclidean world distance (tile cells CELL apart) to the nearest open-water cell over reach tile
    cells (x pitch), 1 beyond. The 4-neighbour step distance it replaces eased the level along
    diamond contours, so the bed (RIVER_DEPTH under the level) dropped under the ocean plane in
    diamonds at the mouths (live 2048n: Ganges, Funan). float32 (TH, TW); computed in
    MOUTH_BLOCK blocks holding a near cell (1 elsewhere), each with a halo wider than reach, so it
    equals the full-grid transform there."""
    open_water = np.asarray(open_water, bool)
    near = np.asarray(near, bool)
    out = np.ones(open_water.shape, np.float32)
    span = reach * CELL[0]
    halo = int(math.ceil(span / min(CELL))) + 2
    H, W = open_water.shape
    for by in range(0, H, MOUTH_BLOCK):
        for bx in range(0, W, MOUTH_BLOCK):
            by1, bx1 = min(H, by + MOUTH_BLOCK), min(W, bx + MOUTH_BLOCK)
            if not near[by:by1, bx:bx1].any():
                continue
            y0, y1, x0, x1 = max(0, by - halo), min(H, by1 + halo), max(0, bx - halo), min(W, bx1 + halo)
            win = open_water[y0:y1, x0:x1]
            if not win.any():
                continue
            d = ndimage.distance_transform_edt(~win, sampling=(CELL[1], CELL[0]))[by - y0:by1 - y0, bx - x0:bx1 - x0]
            t = np.clip(d / span, 0, 1)
            out[by:by1, bx:bx1] = (t * t * (3 - 2 * t)).astype(np.float32)
    return out


# --- polish_campaign_terrain RIVER CHANNELS: the carve itself (polish passes its arrays in) --------------
RIVER_SLOPE = .3       # world units per tile cell up from the water edge (under the projection's 1/3 grade)
RIVER_DEPTH = .6       # relief stage: bed at most this far under the level (polish's slope projection,
                       # TOWN_GRADE 1.0 per world unit = .33 per cell, bounds the channel anyway; carve_beds
                       # deepens it after the projection)
RIVER_FREEBOARD = .12  # water level under the local bank height; band land is held this far above it
RIVER_BAND = 4         # tile cells past the water edge that carry the raised level in the sea height map
RIVER_BANK_R = 6       # tile cells: bank height = soft minimum of the land within this box of the river
RIVER_SOFTMIN = .25    # world units: temperature of that soft minimum
RIVER_MOUTH = 16       # tile cells (x pitch, Euclidean): the level eases to the water plane this close to other water
# RIVER BEDS (carve_beds): after polish's slope projection the bed under the water is cut to BED_DEPTH under
# the (eased) level, BED_SLOPE per tile cell from the water edge. The projection's grade held the channel to
# .33 x its half-width in cells: centre depth over the drawn ground p5/p50 .18/.48 against vanilla's .82/1.38
# (2048p pack), and the water shader, transparent and foaming by depth, showed the bed's coarse triangles as
# pale diagonal lines and read the shallowest stretches as land bars across the river (live 2048p).
# BED_SLOPE stays under audit_ground_seams' lf_cliff floor (800 u16 = .175 per LF pixel = .70 per cell).
BED_SLOPE = .65
BED_DEPTH = 1.3
BED_EDGE = -.05        # cells: only cells this far inside the water edge are cut (the waterline never moves)
BED_CUT = .35          # world units per cell: the cut deepens at most this fast inward from the water edge, on top of
                       # the ground's own slope (ground the projection left above the level inside the outline would
                       # otherwise drop to the bed in one pixel: lf_cliff steps up to 1.9 in the prototype)
BED_PAD = 4.           # cells: bed distance where nothing is cut (bilinear taps beside the river stay positive)
CENTRE_R = .5          # hexes: round every river hex centre the bed is pulled at least ...
CENTRE_DEPTH = .15     # ... this far under the level (smoothstep from the disc rim): the projection held some centres
                       # .1-.28 above the stamped level (WR rivers: Kaidu 640-643,1062-1065, Tarim 504-508,913-916),
                       # gaps in the river line. Not on the end hex in the sand itself (Shule 798,997-999 is meant
                       # dry) nor where sand_taper < CENTRE_SAND (0: the thread stays wet to the end hex).
CENTRE_SAND = 0.
CENTRE_R_MIN = .15     # hexes: the smallest such disc (a thin thread still has a wet centre)


def carve(relief, names, ids, cells, land, physical, outer, hy, hx, q, r, noise_origin, families_path, lake_water,
          grow_steps, lake_levels=None):
    """RIVER CHANNELS on polish's tile-cell relief (TH, TW): (relief, river_level, bed, report).

    relief: float32 world units; names / ids / cells / land: map.hex; physical / outer: tile masks; hy, hx:
    tile -> hex index; q, r: tile hex coordinates (noise); bed = (s, wl), float32 (TH, TW): distance from the
    water edge (tile cells, < 0 under the water, BED_PAD where nothing is cut) and the eased river level, for
    carve_beds after the projection. Without frontier rivers: (relief, all-NaN level, None, None).
    lake_levels: None (every open water at lake_water, as before) or (field, zone) from polish LAKE LEVEL: float64
    (TH, TW) water level the river eases to near a lake with a level of its own, and the bool zone where it applies
    (the level eases from the lake's level back to lake_water over RIVER_MOUTH); outside the zone every value is the
    lake_water one."""
    TH, TW = relief.shape
    H, W = land.shape
    rk = [i for i, k in enumerate(names) if k.startswith('tw3k_riv_sea_')]
    river_level = np.full((TH, TW), np.nan, np.float32)
    if not rk:
        return relief, river_level, None, None
    # Canvas for the channel: outside the native map plus the revealed filler (revealed_filler.river_canvas_hex),
    # where apply_rivers lays rivers as on the canvas. Those pieces run on from canvas rivers across the native
    # border; outside this mask they were open water, and the canvas level eased down to the water plane right at
    # the border: a pit and a 1-3 unit wall on the west edge (audit_native_edges, integ 458ea872). The water
    # (river level, beds) stays outside the native map, as the water stack downstream expects; inside it the river
    # hexes keep land relief (polish: native_river).
    outer_hex = river_canvas_hex(cells, names)
    rvh = np.isin(ids, rk) & ~land & outer_hex                                  # river hexes
    # channel key per river hex: apply_rivers' river-channels.json (river regions merge several channels),
    # else the region names
    fam, famh = river_families(names, ids, families_path)
    openh = ~land & ~np.isin(ids, rk)                                           # sea, lakes, native water
    slot = (cells[:, :, 2] >> 4) > 0
    sprawl = (cells[:, :, 3] & 1) > 0
    nguard = ~outer_hex
    for _ in range(2):
        nguard = hex_dilate(nguard)
    hcy = np.clip(2 * np.arange(H)[:, None] + np.arange(W)[None, :] % 2, 0, TH - 1)
    hcx = np.clip(2 * np.arange(W)[None, :], 0, TW - 1)
    links, link_report = plan_links(rvh, famh, openh, land & ~hex_dilate(slot | sprawl) & ~nguard, relief[hcy, hcx], lake_water,
                                    None if lake_levels is None else np.where(lake_levels[1][hcy, hcx], lake_levels[0][hcy, hcx], lake_water))
    # river spec option B: drawn-only hexes (land inside one region) are carved and watered as links
    drawn = drawn_hexes(land.shape, families_path) & land & outer_hex
    link_report['drawn_hexes'] = int(drawn.sum())
    links = links | drawn
    wet = rvh | links
    mouthh = wet & hex_dilate(openh)
    ring = openh & hex_dilate(mouthh) & outer_hex
    water_outer = outer                                                         # tiles that carry water
    outer = outer | outer_hex[hy, hx]                                           # tiles the channel is carved on
    rv = rvh[hy, hx] & ~physical & outer
    rvx = rv | (links[hy, hx] & outer)                                          # river and link cells
    acc = flow_accumulation(wet, mouthh, relief[hcy, hcx])
    # endorheic rivers without a lake mouth die out in the sand (sand_ends; WR rivers only)
    sand = sand_ends(wet, mouthh, relief[hcy, hcx], famh, {fam[f] for f in fam if f in ENDORHEIC})
    arid = np.isin(famh, [fam[f] for f in sorted(fam) if f in ENDORHEIC]) & wet
    hw = half_width(acc, arid)
    # wet discs inside the narrow arid channel (ARID_WET_R of its half-width), both shrinking towards a sand end:
    # discs wider than the channel read as beads on a thread (offline Shule west of Dunhuang)
    wet_r = np.where(arid, np.minimum(WET_R, ARID_WET_R * hw), WET_R) * sand_taper(wet, sand)
    s, drowned = channel_distance(wet, ring, hw * end_taper(wet, openh, sand), land & ~wet, q, r, noise_origin, wet_r=wet_r)
    s[~outer] = np.inf
    reach = int(RIVER_BAND + np.ceil(RIVER_DEPTH / RIVER_SLOPE) + 2 * HW_MAX) + 3
    dout = grow_steps(rvx, max(reach, RIVER_BANK_R)).astype(np.float32)       # 0 on the river, ring index outside
    bank = physical & ~rvx & (dout <= RIVER_BANK_R)
    # Bank height: a soft minimum (temperature RIVER_SOFTMIN) of the land within RIVER_BANK_R, so the level
    # follows the LOWER bank; a plain mean sat above the low bank of a valley side and the dry-bank hold
    # below had to raise that bank by up to 2.8 (offline simulation on the 2048m surface).
    bden = np.maximum(box_blur(bank.astype(np.float32), RIVER_BANK_R), 1e-3)
    bm = box_blur(np.where(bank, relief, 0).astype(np.float32), RIVER_BANK_R) / bden
    bh = bm - RIVER_SOFTMIN * np.log(np.maximum(box_blur(np.where(bank, np.exp(-np.clip(relief - bm, -8, 8) / RIVER_SOFTMIN), 0).astype(np.float32), RIVER_BANK_R) / bden, 1e-30))
    del bden, bm
    # One smooth level along the river: the bank height averaged over the river cells nearby.
    rden = box_blur(rvx.astype(np.float32), 8)
    wl = box_blur(np.where(rvx, bh - RIVER_FREEBOARD, 0).astype(np.float32), 8) / np.maximum(rden, 1e-3)
    wl = np.minimum(wl, bh - RIVER_FREEBOARD)   # never above the local bank (the average along the river may be)
    open_ = (~physical) & ~rv   # sea, lakes, native water
    # Smooth easing (mouth_ease): Euclidean distance, smoothstep. The 4-neighbour step distance eased along
    # diamond contours and the bed dropped under the ocean plane in diamonds (live 2048n Ganges, Funan).
    dm = mouth_ease(open_, dout <= reach, RIVER_MOUTH)
    wl_raw = wl
    wl = np.maximum(lake_water + (np.maximum(wl, lake_water) - lake_water) * dm, lake_water).astype(np.float32)
    if lake_levels is not None:
        lwf, lz = lake_levels
        wl = np.where(lz, np.maximum(lwf + (np.maximum(wl_raw, lwf) - lwf) * dm, lwf), wl).astype(np.float32)
    del wl_raw
    near = (dout <= reach) & (rden > .02)
    near &= water_outer   # the relief changes on water tiles only: native river hexes keep land relief (polish)
    cut = np.maximum(wl + RIVER_SLOPE * s, wl - RIVER_DEPTH)                     # channel profile (s=0 on the water edge)
    # Ground past the water edge stays dry (the river level must not flood a low bank), river hexes past
    # a narrower edge included; open water is never raised.
    hold = np.where(~open_ & (s > 0), wl + np.clip(RIVER_SLOPE * s, 0, RIVER_FREEBOARD), -np.inf)
    new = np.where(near, np.maximum(np.minimum(relief, cut), hold), relief).astype(np.float32)
    raised = float((new - relief).max())
    band = near & (s <= RIVER_BAND) & ~open_ & water_outer
    river_level[band] = wl[band]
    # bed inputs for carve_beds: mouth_ease eased the level only, the bed follows that level down; open water
    # (sea, lakes) is never cut
    bed_s = np.where(near & ~open_ & np.isfinite(s) & water_outer, np.minimum(s, BED_PAD), BED_PAD).astype(np.float32)
    centres = rvh & ~sand & (sand_taper(wet, sand) >= CENTRE_SAND)
    # the disc stays inside the channel: no wider than the hex's wet disc (narrow arid rivers, sand tapers)
    crad = np.round(np.maximum(np.minimum(CENTRE_R, wet_r), CENTRE_R_MIN) / .02) * .02
    # along the line through neighbouring river hex centres, not in discs round them: on a thin thread (sand
    # tapers, narrow arid rivers) discs wider than the channel read as beads
    seg = []
    for r_, q_ in zip(*np.nonzero(centres)):
        for a, b_ in neighbours(int(q_), int(r_)):
            if 0 <= a < W and 0 <= b_ < H and centres[b_, a] and (b_, a) > (int(r_), int(q_)):
                seg.append((int(q_), int(r_), a, b_))
    seg.sort()
    t = np.zeros((TH, TW))
    if seg:
        S_ = np.array(seg, np.int64)
        x1, z1 = centre(S_[:, 0], S_[:, 1]); x2, z2 = centre(S_[:, 2], S_[:, 3])
        rad = np.minimum(crad[S_[:, 1], S_[:, 0]], crad[S_[:, 3], S_[:, 2]])
        f = np.linspace(0, 1, 5)[:, None]
        px, pz = (x1 + f * (x2 - x1)).ravel(), (z1 + f * (z2 - z1)).ravel()
        pr = np.tile(rad, len(f))
        # the cells round the line (their centres within half a cell diagonal) get the full weight, so the
        # centre pixels themselves are pulled down; it falls to 0 over the disc radius beyond them
        d0 = .5 * math.hypot(*CELL)
        for v in np.unique(pr):
            k = pr == v
            t = np.maximum(t, np.clip(1 - np.maximum(_points_distance(px[k], pz[k], (TH, TW)) - d0, 0) / (float(v) * HEX), 0, 1))
    lone = centres.copy()
    if seg:
        lone[S_[:, 1], S_[:, 0]] = False; lone[S_[:, 3], S_[:, 2]] = False
    if lone.any():
        lr, lq = np.nonzero(lone)
        lx, lz = centre(lq, lr)
        t = np.maximum(t, np.clip(1 - np.maximum(_points_distance(lx, lz, (TH, TW)) - .5 * math.hypot(*CELL), 0) / (CENTRE_R_MIN * HEX), 0, 1))
    bed_w = (t * t * (3 - 2 * t) * water_outer).astype(np.float32)   # canvas tiles only (the water stops at the native border)
    report = {'river_cells': int(rv.sum()), 'link_cells': int((rvx & ~rv).sum()), 'band_cells': int(band.sum()),
              'level_range': [float(np.nanmin(river_level)), float(np.nanmax(river_level))], 'max_bank_raise': raised,
              'river_hex_cells_wet': float(((s < 0) & rv).sum() / max(int(rv.sum()), 1)), 'land_hexes_given_dry_disc': drowned,
              'links': link_report, 'sand_ends': [[int(q_), int(r_)] for r_, q_ in zip(*np.nonzero(sand))]}
    return new, river_level, (bed_s, wl.astype(np.float32), bed_w), report


def carve_beds(encoded, s, wl, allowed, level, norm, gain, centre=None, slope=BED_SLOPE, depth=BED_DEPTH, rate=BED_CUT, rows=256):
    """RIVER BEDS on the final quantized LF (encoded: (oh, ow) uint16, south-up rows, changed in place):
    every allowed pixel with s < BED_EDGE (s, wl: carve's bed arrays resized to the LF) is lowered towards
    wl + max(slope * s, -depth), by at most rate * |s| (so the cut starts at 0 on the waterline). Only under
    the water: the waterline (s = 0), the banks and the river level are unchanged. centre (weights 0..1 round the
    river hex centres, carve's third bed array resized): the ground there blends towards CENTRE_DEPTH under the
    level by that weight, so every centre is wet. Returns a report (float64 sums)."""
    oh = encoded.shape[0]
    cut = 0
    lowered_sum = 0.
    lowered_max = 0.
    for r0 in range(0, oh, rows):
        sl = slice(r0, min(r0 + rows, oh))
        ss = s[sl].astype(np.float64)
        cw_ = centre[sl].astype(np.float64) if centre is not None else np.zeros(ss.shape)
        m = allowed[sl] & ((ss < BED_EDGE) | (cw_ > 0))
        if not m.any():
            continue
        cur = encoded[sl].astype(np.float64)
        ground = level + (cur / 65535 - norm) * gain
        w_ = wl[sl].astype(np.float64)
        bed = np.where(ss < BED_EDGE, np.maximum(w_ + np.maximum(slope * ss, -depth), ground + rate * ss), ground)
        bed = np.minimum(bed, ground * (1 - cw_) + (w_ - CENTRE_DEPTH) * cw_)
        enc = np.rint(np.clip(norm + (bed - level) / gain, 0, 1) * 65535)
        low = m & (enc < cur)
        if not low.any():
            continue
        d = (cur - enc)[low] / 65535 * gain
        cut += int(low.sum())
        lowered_sum += float(d.sum(dtype=np.float64))
        lowered_max = max(lowered_max, float(d.max()))
        block = encoded[sl]
        block[low] = enc[low].astype(encoded.dtype)
    return {'pixels_lowered': cut, 'mean_lowered': round(lowered_sum / max(cut, 1), 4), 'max_lowered': round(lowered_max, 4),
            'slope_per_cell': slope, 'depth': depth, 'cut_rate_per_cell': rate}

"""Connected pieces per region on the hex map (numpy only; same count as sdk_memory_export's BFS, ~150x faster)."""
import collections
import numpy as np


def region_components(ids):
    """Connected pieces per region id with the SDK preflight's hex adjacency (column x: (x, y±1),
    (x±1, y+x%2), (x±1, y+x%2-1)), by union-find over vertical runs. ids: (h, w) int array."""
    h, w = ids.shape
    run_lo, run_hi, run_id = [], [], []
    starts, total = [], 0
    for x in range(w):
        col = ids[:, x]
        cut = np.flatnonzero(np.diff(col)) + 1
        lo = np.concatenate(([0], cut)); hi = np.concatenate((cut, [h])) - 1
        starts.append(total); total += len(lo)
        run_lo.append(lo); run_hi.append(hi); run_id.append(col[lo])
    starts.append(total)
    lo = np.concatenate(run_lo); hi = np.concatenate(run_hi); rid = np.concatenate(run_id)
    parent = list(range(len(lo)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for x in range(w - 1):
        s = x % 2
        a0, a1, b0, b1 = starts[x], starts[x + 1], starts[x + 1], starts[x + 2]
        i, j = a0, b0
        while i < a1 and j < b1:
            # run i of column x reaches rows lo-1+s .. hi+s of column x+1
            l, u = lo[i] - 1 + s, hi[i] + s
            if hi[j] < l:
                j += 1; continue
            if lo[j] > u:
                i += 1; continue
            if rid[i] == rid[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
            if hi[j] < u:
                j += 1
            else:
                i += 1
    roots = collections.Counter()
    seen = set()
    for k in range(len(lo)):
        r = find(k)
        if r not in seen:
            seen.add(r); roots[int(rid[k])] += 1
    return roots


# ---- SDK region areas (what MapDataBuilder actually numbers)
# tooldatabuilder campaignregionsprocess (+0x718d0 / +0x71710) labels each hex with a 16-bit word:
# low 9 bits region index, high 7 bits area index inside the region, 0x7f = 'not yet assigned'.
# An area is a hex-connected run of one region AND one traversal class (HEX_MAP_MASK_DATA::apply_mask),
# found in row-major order. The 128th area of a region gets index 127 = 'unassigned', so a flood fill
# over two or more such hexes never ends: its stack doubles until the 2400 MB tracked-memory limit.
# Classes, matched against map_data.esf REGION_AREA_DATA of an exported map (byte0 bits 0-1 = t):
#   7 impassable (byte2 bit 3); else t=0 -> 0, t=2 -> 1, sea t=1 -> 3 if ford (byte3 bit 7) else 4,
#   t=3 -> 8 next to a ford, else 6. (The ford-bank split is approximate: allow a few areas of margin.)
SDK_AREA_LIMIT = 120        # hard limit is 127 per region; margin for the approximate ford-bank class


def hex_dilate(mask):
    """mask or any of its six hex neighbours (odd columns: (x±1, y+1); even: (x±1, y-1))."""
    m = np.asarray(mask, bool)
    o = m.copy()
    o[1:] |= m[:-1]; o[:-1] |= m[1:]
    o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
    odd = (np.arange(m.shape[1]) % 2 == 1)[None, :]
    up = np.zeros_like(m); up[:-1] = m[1:]
    dn = np.zeros_like(m); dn[1:] = m[:-1]
    o[:, :-1] |= np.where(odd[:, :-1], up[:, 1:], dn[:, 1:])
    o[:, 1:] |= np.where(odd[:, 1:], up[:, :-1], dn[:, :-1])
    return o


# CAIME Hex.Directions_FlatTop (dq, dr) for even / odd columns: 0 up, then clockwise (the same
# six neighbours as hex_dilate; road_tiles.DIRS holds the same table).
HEX_DIRS = (((0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)),
            ((0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)))


def hex_min_filter(a, steps):
    """Minimum of a float array over each hex and every hex within `steps` hex steps (HEX_DIRS
    neighbours, repeated). Off-grid neighbours are ignored (no wraparound); NaN counts as +inf."""
    cur = np.where(np.isnan(a), np.inf, np.asarray(a, np.float64))
    h, w = cur.shape
    odd = (np.arange(w) % 2 == 1)[None, :]
    for _ in range(steps):
        out = cur.copy()
        for d in range(6):
            for parity, sel in ((0, ~odd), (1, odd)):
                dq, dr = HEX_DIRS[parity][d]
                shifted = np.full((h, w), np.inf)
                r0, r1 = max(0, -dr), min(h, h - dr)
                c0, c1 = max(0, -dq), min(w, w - dq)
                shifted[r0:r1, c0:c1] = cur[r0 + dr:r1 + dr, c0 + dq:c1 + dq]
                out = np.where(sel, np.minimum(out, shifted), out)
        cur = out
    return cur


def hex_mean_filter(a, steps):
    """`steps` passes of the mean over each hex and its on-grid HEX_DIRS neighbours (a hex-shaped
    blur; off-grid neighbours are left out of the mean, no wraparound). float64."""
    cur = np.asarray(a, np.float64)
    h, w = cur.shape
    odd = (np.arange(w) % 2 == 1)[None, :]
    for _ in range(steps):
        tot = cur.copy(); cnt = np.ones((h, w))
        for d in range(6):
            for parity, sel in ((0, ~odd), (1, odd)):
                dq, dr = HEX_DIRS[parity][d]
                shifted = np.zeros((h, w)); ok = np.zeros((h, w), bool)
                r0, r1 = max(0, -dr), min(h, h - dr)
                c0, c1 = max(0, -dq), min(w, w - dq)
                shifted[r0:r1, c0:c1] = cur[r0 + dr:r1 + dr, c0 + dq:c1 + dq]; ok[r0:r1, c0:c1] = True
                use = sel & ok
                tot = tot + np.where(use, shifted, 0.); cnt = cnt + use
        cur = tot / cnt
    return cur


def neighbour_bits(mask):
    """uint8 per hex: bit d set where the neighbour in CAIME direction d (HEX_DIRS) is in mask.
    Off-grid neighbours count as not in mask (no wraparound). Any bit set == hex_dilate(mask) & ~mask
    on hexes outside mask."""
    m = np.asarray(mask, bool)
    h, w = m.shape
    out = np.zeros((h, w), np.uint8)
    odd = (np.arange(w) % 2 == 1)[None, :]
    for d in range(6):
        nb = np.zeros((h, w), bool)
        for parity, sel in ((0, ~odd), (1, odd)):
            dq, dr = HEX_DIRS[parity][d]
            shifted = np.zeros((h, w), bool)
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dq), min(w, w - dq)
            shifted[r0:r1, c0:c1] = m[r0 + dr:r1 + dr, c0 + dq:c1 + dq]
            nb |= shifted & sel
        out |= nb.astype(np.uint8) << d
    return out


def hex_neighbour_values(a):
    """(6, h, w) values of the six hex neighbours (CAIME direction order, HEX_DIRS) of every hex;
    an off-grid neighbour takes the hex's own value (edge clamp, no wraparound)."""
    a = np.asarray(a)
    h, w = a.shape
    odd = (np.arange(w) % 2 == 1)[None, :]
    out = np.empty((6, h, w), a.dtype)
    for d in range(6):
        v = a.copy()
        for parity, sel in ((0, ~odd), (1, odd)):
            dq, dr = HEX_DIRS[parity][d]
            shifted = a.copy()
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dq), min(w, w - dq)
            shifted[r0:r1, c0:c1] = a[r0 + dr:r1 + dr, c0 + dq:c1 + dq]
            v = np.where(sel, shifted, v)
        out[d] = v
    return out


def hex_label(mask):
    """Labels of the hex-connected pieces of mask (same adjacency as hex_dilate), by a doubled-row
    grid: hex (q, r) covers rows 2r+q%2 and 2r+q%2+1 of column q, so 4-connectivity there is hex
    adjacency. scipy is imported here only (the rest of this module stays numpy only)."""
    from scipy import ndimage
    mask = np.asarray(mask, bool)
    H, W = mask.shape
    q = np.arange(W)
    rows = 2 * np.arange(H)[:, None] + q[None, :] % 2
    grid = np.zeros((2 * H + 1, W), bool)
    grid[rows, q[None, :]] = mask
    grid[rows + 1, q[None, :]] = mask
    return ndimage.label(grid)[0][rows, q[None, :]] * mask


def area_class(cells):
    """Traversal class per hex as the SDK regions process splits areas (see above)."""
    t = cells[:, :, 0] & 3
    imp = (cells[:, :, 2] & 8) > 0
    ford = (t == 1) & ((cells[:, :, 3] & 0x80) > 0)
    cls = np.select([imp, t == 0, t == 2, ford, t == 1], [7, 0, 1, 3, 4], 6)
    cls[(cls == 6) & hex_dilate(ford)] = 8
    return cls.astype(np.int64)


def region_areas(ids, cells):
    """SDK areas per region id: connected pieces of equal (region, traversal class)."""
    key = np.asarray(ids, np.int64) * 16 + area_class(cells)
    out = collections.Counter()
    for k, n in region_components(key).items():
        out[k // 16] += n
    return out

"""Forest/open edge cleanup of the global blend (polish_campaign_terrain.py, EDGE CLEANUP block).
Pure numpy (no scipy): runs in the pipeline Python. Deterministic (no random numbers).

Vanilla forest edges (exemplar strips, offline 2026-10-23, output/offline-view/exemplar/metrics_*.json):
about 0.25 woodland/open edges per hex in the edge core (+-5 hexes) and 0.02 at 5-15 hexes out;
3.8 detached islands per 1000 hex^2, 90th-percentile island 3.9 hex^2. The canvas had .44 and
.13-.19 edges per hex and 14 islands per 1000 hex^2 (donor patches dithered per pixel, ecotone tongues).

On real-terrain canvas ground only (weight > .5), with woodland = materials 4/5 (WOODLAND):
  * the large-scale boundary is the 0.5 level of the woodland share box-blurred twice over
    BOUNDARY_R tile cells; d = signed distance to it in hexes (forest side negative);
  * within |d| <= BAND hexes the woodland mask takes a majority over a (2*MAJORITY_PX+1)^2 pixel box:
    pixel dither and sub-hex tongues go, the meander of the boundary stays;
  * beyond |d| > CORE hexes, islands of the minority side (woodland on the open side, clearings on
    the forest side) that do not contain a (2*ISLAND_ERODE+1)^2 pixel square are removed (opening by
    reconstruction: erosion, then ISLAND_REBUILD geodesic dilation steps inside the class), about
    6 hex^2 and smaller; larger patches keep their exact outline;
  * a pixel that changes class takes the material (both channels) of the nearest pixel that keeps the
    target class (4-neighbour BFS from the class border, _fill); unreached pixels keep theirs.
Native ground is untouched (weight 0 there). Row x column blocks with a halo, on a thread pool; each
step runs on the bounding box of the pixels it may change (plus its reach), blocks return their
writes and the blend is written once all have run (no copy of it). Offline on the 2048 stage blend
(12 threads): ~18-20 s, 6.28 M pixels changed, byte-identical across runs and block sizes.
"""
import numpy as np

WOODLAND = (4, 5)
BOUNDARY_R = 6         # tile cells (box radius, applied twice): large-scale boundary
BAND = 15.             # hexes: majority smoothing band
CORE = 5.              # hexes: islands are removed beyond it
MAJORITY_PX = 8        # pixels (8 per hex): majority box radius
ISLAND_ERODE = 8       # pixels: islands that cannot hold a 17x17 pixel square go (~4.5-6 hex^2)
ISLAND_REBUILD = 12    # pixels: geodesic dilation steps that rebuild the surviving patches
FILL_STEPS = 16        # pixels: BFS reach of the material fill
BLOCK, COLS, HALO = 256, 1024, 48  # block rows x columns, halo (>= ISLAND_ERODE + ISLAND_REBUILD + MAJORITY_PX)
MAX_D = 40             # tile cells of distance measured from the boundary


def _box_sum(a, r):
    """Box sum over (2r+1)^2 (edges: window clipped), int32, via cumulative sums."""
    a = a.astype(np.int32)
    for ax in (0, 1):
        c = np.cumsum(a, axis=ax, dtype=np.int32)
        n = a.shape[ax]
        hi = np.take(c, np.minimum(np.arange(n) + r, n - 1), axis=ax)
        lo_idx = np.arange(n) - r - 1
        lo = np.where((lo_idx >= 0)[:, None] if ax == 0 else (lo_idx >= 0)[None, :], np.take(c, np.maximum(lo_idx, 0), axis=ax), 0)
        a = hi - lo
    return a


def _dilate(m, r=1):
    out = m.copy()
    for k in range(1, r + 1):
        out[:-k] |= m[k:]; out[k:] |= m[:-k]
    m2 = out.copy()
    for k in range(1, r + 1):
        out[:, :-k] |= m2[:, k:]; out[:, k:] |= m2[:, :-k]
    return out


def _erode(m, r):
    return ~_dilate(~m, r)


def _open_rec(cls, erode, steps):
    """Opening by reconstruction (limited): pixels of cls connected within `steps` to a core
    that survives an erosion of radius `erode`."""
    rec = _erode(cls, erode)
    for _ in range(steps):
        nxt = _dilate(rec, 1) & cls
        if (nxt == rec).all():
            break
        rec = nxt
    return rec


def _fill(packed, seed, need):
    """Each `need` pixel takes the value of the nearest `seed` pixel within FILL_STEPS (4-neighbour
    BFS, fastops.propagate_nearest; deterministic); -1 where none is reached. Only seed pixels on the
    seed's own border within reach of a need pixel start the search (same reach, small front)."""
    from fastops import propagate_nearest
    start = seed & _dilate(~seed, 1) & _dilate(need, FILL_STEPS)
    near, dist = propagate_nearest(packed, start, FILL_STEPS)
    return np.where(dist <= FILL_STEPS, near, -1)


def tile_fields(cur, oh, ow, ground_w, TH, TW):
    """Signed distance (hexes) to the large-scale woodland boundary and the ground weight, tile grid
    (south-up), from the current blend."""
    from native_campaign_climate import box_blur
    from fastops import grow_steps
    f = (_LUT[cur[..., 0]].view(np.uint8).reshape(oh // 4, 4, ow // 4, 4).sum((1, 3), dtype=np.uint8) / np.float32(16))[::-1]
    ft = np.zeros((TH, TW), np.float32); ft[:f.shape[0], :f.shape[1]] = f[:TH, :TW]
    wl = box_blur(box_blur(ft, BOUNDARY_R), BOUNDARY_R) > .5
    # cells with a 4-neighbour of the other side, on both sides (no wraparound at the grid edge)
    edge = np.zeros_like(wl)
    edge[1:] |= wl[1:] != wl[:-1]; edge[:-1] |= wl[:-1] != wl[1:]
    edge[:, 1:] |= wl[:, 1:] != wl[:, :-1]; edge[:, :-1] |= wl[:, :-1] != wl[:, 1:]
    dist = np.minimum(grow_steps(edge, MAX_D), MAX_D + 1).astype(np.float32) / 2
    return {'d': np.where(wl, -dist, dist).astype(np.float32), 'w': np.asarray(ground_w, np.float32)}


def _px_sep(T, txv, tyv):
    """Bilinear samples of tile field T on the grid tyv (rows) x txv (columns), separably:
    each distinct tile row is interpolated along x once."""
    H, W = T.shape
    x0 = np.clip(np.floor(txv).astype(np.int32), 0, W - 2); fx = np.clip(txv - x0, 0, 1).astype(np.float32)
    y0 = np.clip(np.floor(tyv).astype(np.int32), 0, H - 2); fy = np.clip(tyv - y0, 0, 1).astype(np.float32)
    u = np.unique(np.concatenate((y0, y0 + 1)))
    Tu = T[u]; hr = Tu[:, x0] * (1 - fx) + Tu[:, x0 + 1] * fx
    i0 = np.searchsorted(u, y0); i1 = np.searchsorted(u, y0 + 1)
    return hr[i0] * (1 - fy)[:, None] + hr[i1] * fy[:, None]


def _px(T, tx, ty):
    H, W = T.shape
    x0 = np.clip(np.floor(tx).astype(np.int32), 0, W - 2); y0 = np.clip(np.floor(ty).astype(np.int32), 0, H - 2)
    fx = np.clip(tx - x0, 0, 1).astype(np.float32); fy = np.clip(ty - y0, 0, 1).astype(np.float32)
    return (T[y0, x0] * (1 - fx) + T[y0, x0 + 1] * fx) * (1 - fy) + (T[y0 + 1, x0] * (1 - fx) + T[y0 + 1, x0 + 1] * fx) * fy


def _span(m, margin, n):
    """(lo, hi) of the True entries of the 1-D mask m widened by margin and clipped to [0, n)."""
    k = np.flatnonzero(m)
    return max(0, int(k[0]) - margin), min(n, int(k[-1]) + margin + 1)


def _bbox(mask, margin):
    """Slices of the bounding box of mask widened by margin (clipped to mask's shape)."""
    r = _span(mask.any(1), margin, mask.shape[0]); c = _span(mask.any(0), margin, mask.shape[1])
    return slice(*r), slice(*c)


# Reach (pixels, Chebyshev) of each step: it runs on the bounding box of the pixels it may change
# widened by its reach, which gives the bytes of running it on the whole window (within reach of
# those pixels the window edge is the only edge it sees).
_REC_REACH = ISLAND_ERODE + ISLAND_REBUILD + 1
_FILL_REACH = FILL_STEPS + 2
_LUT = np.zeros(256, bool); _LUT[list(WOODLAND)] = True


def block(cur, row0, row1, oh, F, col0=0, src=None, cols=None, defer=False):
    """Clean rows row0..row1 (and local columns cols=(a, b), default all) of cur (north-first; cur's
    columns start at absolute column col0). src: the blend to read (default cur). Returns pixels
    changed; defer=True writes nothing and returns (changed, rows, cols, packed R | G << 8)."""
    src = cur if src is None else src
    a0, a1 = cols if cols is not None else (0, src.shape[1])
    e0, e1 = max(0, row0 - HALO), min(src.shape[0], row1 + HALO)
    f0, f1 = max(0, a0 - HALO), min(src.shape[1], a1 + HALO)
    none = (0, None, None, None) if defer else 0
    blk = src[e0:e1, f0:f1]
    m0 = _LUT[blk[..., 0]]
    if m0.all() or not m0.any():   # uniform window: nothing to clean
        return none
    n, wpx = blk.shape[:2]
    tyv = (oh - 1 - np.arange(e0, e1) + .5) / 4 - .5; txv = (np.arange(col0 + f0, col0 + f0 + wpx) + .5) / 4 - .5
    Hf, Wf = F['w'].shape
    y0 = np.clip(np.floor(tyv).astype(np.int32), 0, Hf - 2); x0 = np.clip(np.floor(txv).astype(np.int32), 0, Wf - 2)
    tr0, tr1, tc0, tc1 = int(y0.min()), int(y0.max()) + 2, int(x0.min()), int(x0.max()) + 2
    wt = F['w'][tr0:tr1, tc0:tc1] > .5
    if not wt.any():
        return none
    # rows / columns whose bilinear corners include a tile with w > .5 (ok is False elsewhere)
    rk = wt.any(1); ck = wt.any(0)
    rr = rk[y0 - tr0] | rk[y0 + 1 - tr0]; cc = ck[x0 - tc0] | ck[x0 + 1 - tc0]
    if not (rr.any() and cc.any()):
        return none
    rs, cs = slice(*_span(rr, 0, n)), slice(*_span(cc, 0, wpx))
    ok = np.zeros((n, wpx), bool); d = np.zeros((n, wpx), np.float32)
    ok[rs, cs] = _px_sep(F['w'], txv[cs], tyv[rs]) > .5
    if not ok.any():
        return none
    d[rs, cs] = _px_sep(F['d'], txv[cs], tyv[rs])
    W = m0
    band = ok & (np.abs(d) <= BAND)
    T = W.copy()
    if band.any():
        k = (2 * MAJORITY_PX + 1) ** 2
        bs = _bbox(band, 0); ws = _bbox(band, MAJORITY_PX)
        maj = _box_sum(W[ws], MAJORITY_PX) * 2 > k
        oy, ox = bs[0].start - ws[0].start, bs[1].start - ws[1].start
        maj = maj[oy:oy + bs[0].stop - bs[0].start, ox:ox + bs[1].stop - bs[1].start]
        T[bs] = np.where(band[bs], maj, W[bs])
    outer = ok & (np.abs(d) > CORE)
    if outer.any():
        isl = outer & (d > 0) & T          # woodland islands on the open side
        if isl.any():
            ws = _bbox(isl, _REC_REACH)
            T[ws] = np.where(isl[ws] & ~_open_rec(T[ws], ISLAND_ERODE, ISLAND_REBUILD), False, T[ws])
        isl = outer & (d < 0) & ~T         # clearings on the forest side
        if isl.any():
            ws = _bbox(isl, _REC_REACH)
            T[ws] = np.where(isl[ws] & ~_open_rec(~T[ws], ISLAND_ERODE, ISLAND_REBUILD), True, T[ws])
    a, b = row0 - e0, row1 - e0; c_, d_ = a0 - f0, a1 - f0
    ch = np.zeros((n, wpx), bool)   # only the block's own pixels change
    ch[a:b, c_:d_] = ok[a:b, c_:d_] & (T[a:b, c_:d_] != W[a:b, c_:d_])
    if not ch.any():
        return none
    gy, gx, gv = [], [], []
    for target in (True, False):
        need = ch & (T == target)
        if not need.any():
            continue
        ws = _bbox(need, _FILL_REACH)
        sub = blk[ws]
        packed = sub[..., 0].astype(np.int32) | (sub[..., 1].astype(np.int32) << 8)
        val = _fill(packed, (W[ws] == target) & (T[ws] == target), need[ws])
        iy, ix = np.nonzero(need[ws] & (val >= 0))
        gy.append(iy + ws[0].start + e0); gx.append(ix + ws[1].start + f0); gv.append(val[iy, ix])
    gy, gx, gv = np.concatenate(gy), np.concatenate(gx), np.concatenate(gv)
    nch = int(ch.sum())
    if defer:
        return nch, gy, gx, gv
    cur[gy, gx, 0] = (gv & 255).astype(np.uint8); cur[gy, gx, 1] = (gv >> 8).astype(np.uint8)
    return nch


def apply(cur, oh, ow, ground_w, TH, TW, pool=None):
    """Whole blend. Blocks read cur and return their writes, which are applied once all blocks have
    run: no copy of the blend, and the result does not depend on the order blocks run in."""
    F = tile_fields(cur, oh, ow, ground_w, TH, TW)
    jobs = [(r, c) for r in range(0, oh, BLOCK) for c in range(0, ow, COLS)]
    run = (lambda rc: block(cur, rc[0], min(rc[0] + BLOCK, oh), oh, F, cols=(rc[1], min(rc[1] + COLS, ow)), defer=True))
    res = list(pool.map(run, jobs)) if pool is not None else [run(j) for j in jobs]
    total = 0
    for nch, gy, gx, gv in res:
        if nch:
            cur[gy, gx, 0] = (gv & 255).astype(np.uint8); cur[gy, gx, 1] = (gv >> 8).astype(np.uint8); total += nch
    return total

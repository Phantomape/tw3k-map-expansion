"""Scan a built campaign pack for ground seams and attribute each one to a cause.

Read-only. Reads one built pack (--pack, built from --map) plus the
vanilla data packs and the logical map, then writes a JSON report and a PNG
overview into --out, which must lie outside the game folder and the stage.

Layers and defect classes (all compared with vanilla at the same place):
  coarse  global land meshes rasterised at quarter-hex spacing
    coarse_hole_native_removed   land had a vanilla triangle, now none (water shows)
    coarse_hole_new_land         interior land outside vanilla with no triangle
    coarse_hole_prefab_removed   vanilla cutout whose mountain model was removed
    coarse_stack                 overlapping faces disagree in height
    coarse_step                  height jump far above vanilla's at that place
  fine    runtime tile_list placements on the half-hex tile grid
    fine_uncovered               land without any fine tile where vanilla had one
    fine_range_stale             stored height range disagrees with the height map
    fine_special_removed         vanilla special tile removed though ground unchanged
    fine_climate_patch           replacement tile climate differs from vanilla
  lf      low-frequency height map
    lf_cliff                     single-pixel jump far above vanilla's
"""
import tw3k_paths
import argparse, json, math, os, re, struct, sys, time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_reference import ROOT, inventory, read_hex
import campaign_world as _frame
from campaign_world import DX, W, H
from native_tile_dimensions import QUARTER_TURN_FLAGS, decode_dimensions, placement_footprint
from read_tile_list import parse as parse_tiles

GAME = tw3k_paths.GAME_DATA
SHIFT_HEX = _frame.NATIVE_OFFSET_HEX
SHIFT_TILE = _frame.NATIVE_OFFSET_TILE
SHIFT_LF = _frame.NATIVE_OFFSET_LF
B = chr(92)


class Packs:
    def __init__(self, packs):
        self.index = {}
        self.packs = list(packs)
        for pack in self.packs:
            for e in inventory(pack, False)['entries']:
                self.index.setdefault(e['path'], (pack, e))

    def read(self, key):
        pack, e = self.index[key]
        assert not e['compressed'], key
        with pack.open('rb') as f:
            f.seek(e['offset'])
            return f.read(e['size'])

    def keys(self, prefix, suffix=''):
        return sorted(k for k in self.index if k.startswith(prefix) and k.endswith(suffix))

class StageAssets:
    """Read staged replacements with the same native fallback as the game."""
    def __init__(self, stage, originals):
        self.stage=stage;self.originals=originals
    def read(self,key):
        p=self.stage/key
        return p.read_bytes() if p.is_file() else self.originals.read(key)
    def keys(self,prefix,suffix=''):
        local={p.relative_to(self.stage).as_posix() for p in self.stage.rglob('*')
               if p.is_file() and p.relative_to(self.stage).as_posix().startswith(prefix) and p.name.endswith(suffix)}
        return sorted(local | set(self.originals.keys(prefix,suffix)))


def lf_pixels(data):
    h, w = struct.unpack_from('<2I', data, 12)
    return np.frombuffer(data, '<u2', offset=128, count=h * w).reshape(h, w)[::-1]


def mesh_triangles(data, shift=(0., 0., 0.)):
    mp = struct.unpack_from('<I', data, 152)[0]
    _, _, _, vo, vc, io, ic = struct.unpack_from('<HH5I', data, mp)
    v = np.frombuffer(data, '<f4', offset=mp + vo, count=vc * 4).reshape(vc, 4)[:, :3].astype(np.float64) + shift
    i = np.frombuffer(data, '<u2', offset=mp + io, count=ic).reshape(-1, 3)
    return v[i]


# Coarse coverage samples sit off the step lattice. In the tile-aligned frame
# native coarse vertices lie on that lattice, so unshifted samples land on
# vertices and grid edges, where co-located vertices
# (cliffs, cutouts) read as stacked faces and float32 rounding decides cover.
SAMPLE_OFFSET = (.37, .29)


def raster(tris, nx, nz, step, offset=(0., 0.)):
    """Per-sample triangle count and min/max height on a regular grid.

    Sample (i, j) is at ((i + offset[0]) * step, (j + offset[1]) * step)."""
    count = np.zeros(nx * nz, np.uint16)
    low = np.full(nx * nz, np.inf, np.float32)
    high = np.full(nx * nz, -np.inf, np.float32)
    x = tris[:, :, 0] / step - offset[0]
    z = tris[:, :, 2] / step - offset[1]
    # In the tile-aligned frame native vertices sit exactly on sample points;
    # float32 positions (~2e-4 samples of error near x=500) must not drop a
    # sample from one mesh's bounds and keep it in the other's.
    i0 = np.clip(np.ceil(x.min(1) - 1e-3).astype(int), 0, nx - 1)
    i1 = np.clip(np.floor(x.max(1) + 1e-3).astype(int), 0, nx - 1)
    j0 = np.clip(np.ceil(z.min(1) - 1e-3).astype(int), 0, nz - 1)
    j1 = np.clip(np.floor(z.max(1) + 1e-3).astype(int), 0, nz - 1)
    wb, hb = i1 - i0 + 1, j1 - j0 + 1
    ok = (wb > 0) & (hb > 0)
    a = np.stack((x[:, 0], z[:, 0]), 1)
    d = np.stack((x[:, 1] - x[:, 0], z[:, 1] - z[:, 0]), 1)
    e = np.stack((x[:, 2] - x[:, 0], z[:, 2] - z[:, 0]), 1)
    det = d[:, 0] * e[:, 1] - e[:, 0] * d[:, 1]
    # Vertical skirt faces project to zero area; rounding must not make one
    # mesh's copy count as ground cover and the other's not.
    ok &= np.abs(det) > 1e-6
    y = tris[:, :, 1]
    keys = np.where(ok, wb * 4096 + hb, -1)
    for key in np.unique(keys[keys >= 0]):
        sel = np.flatnonzero(keys == key)
        gw, gh = int(key // 4096), int(key % 4096)
        ox, oz = np.meshgrid(np.arange(gw), np.arange(gh))
        ox, oz = ox.ravel(), oz.ravel()
        chunk = max(1, 4_000_000 // (gw * gh))
        for start in range(0, len(sel), chunk):
            t = sel[start:start + chunk]
            px = i0[t, None] + ox[None]
            pz = j0[t, None] + oz[None]
            rx = px - a[t, 0:1]
            rz = pz - a[t, 1:2]
            u = (rx * e[t, 1:2] - rz * e[t, 0:1]) / det[t, None]
            w = (d[t, 0:1] * rz - d[t, 1:2] * rx) / det[t, None]
            inside = (u >= -1e-3) & (w >= -1e-3) & (u + w <= 1 + 1e-3)
            if not inside.any():
                continue
            h = y[t, 0:1] + u * (y[t, 1:2] - y[t, 0:1]) + w * (y[t, 2:3] - y[t, 0:1])
            idx = (pz * nx + px)[inside]
            hv = h[inside].astype(np.float32)
            np.add.at(count, idx, 1)
            np.minimum.at(low, idx, hv)
            np.maximum.at(high, idx, hv)
    return count.reshape(nz, nx), low.reshape(nz, nx), high.reshape(nz, nx)


def _raster_setup(tris, nx, nz, step, offset):
    """raster()'s per-triangle sample bounds, edge vectors and size keys (-1: no cover)."""
    x = tris[:, :, 0] / step - offset[0]
    z = tris[:, :, 2] / step - offset[1]
    i0 = np.clip(np.ceil(x.min(1) - 1e-3).astype(int), 0, nx - 1)
    i1 = np.clip(np.floor(x.max(1) + 1e-3).astype(int), 0, nx - 1)
    j0 = np.clip(np.ceil(z.min(1) - 1e-3).astype(int), 0, nz - 1)
    j1 = np.clip(np.floor(z.max(1) + 1e-3).astype(int), 0, nz - 1)
    wb, hb = i1 - i0 + 1, j1 - j0 + 1
    ok = (wb > 0) & (hb > 0)
    a = np.stack((x[:, 0], z[:, 0]), 1)
    d = np.stack((x[:, 1] - x[:, 0], z[:, 1] - z[:, 0]), 1)
    e = np.stack((x[:, 2] - x[:, 0], z[:, 2] - z[:, 0]), 1)
    det = d[:, 0] * e[:, 1] - e[:, 0] * d[:, 1]
    ok &= np.abs(det) > 1e-6
    return i0, i1, j0, j1, a, d, e, det, tris[:, :, 1], np.where(ok, wb * 4096 + hb, -1)


def _raster_into(s, mine, count, low, high, r0, c0, width):
    """Accumulate triangles `mine` (setup s) into flat local grids of `width` columns whose
    first sample is (r0, c0), per size key in index order, as raster()."""
    i0, i1, j0, j1, a, d, e, det, y, keys = s
    # Stable sort: per key the triangles keep raster()'s index order.
    mine = mine[np.argsort(keys[mine], kind='stable')]
    cut = np.flatnonzero(np.diff(keys[mine])) + 1
    for sel in np.split(mine, cut) if len(mine) else ():
        key = keys[sel[0]]
        gw, gh = int(key // 4096), int(key % 4096)
        ox, oz = np.meshgrid(np.arange(gw), np.arange(gh))
        ox, oz = ox.ravel(), oz.ravel()
        chunk = max(1, 1_000_000 // (gw * gh))
        for start in range(0, len(sel), chunk):
            t = sel[start:start + chunk]
            px = i0[t, None] + ox[None]
            pz = j0[t, None] + oz[None]
            rx = px - a[t, 0:1]
            rz = pz - a[t, 1:2]
            u = (rx * e[t, 1:2] - rz * e[t, 0:1]) / det[t, None]
            w = (d[t, 0:1] * rz - d[t, 1:2] * rx) / det[t, None]
            inside = (u >= -1e-3) & (w >= -1e-3) & (u + w <= 1 + 1e-3)
            if not inside.any():
                continue
            h = y[t, 0:1] + u * (y[t, 1:2] - y[t, 0:1]) + w * (y[t, 2:3] - y[t, 0:1])
            idx = ((pz - r0) * width + px - c0)[inside]
            hv = h[inside].astype(np.float32)
            np.add.at(count, idx, 1)
            np.minimum.at(low, idx, hv)
            np.maximum.at(high, idx, hv)


def raster_bands(tris, nx, nz, step, offset=(0., 0.), bands=48, workers=8):
    """raster() on a thread pool: identical arrays (the same arithmetic per triangle, and the
    count / min / max accumulations commute). Each band of grid rows takes the triangles whose
    first row j0 lies in it, in raster()'s key-then-index order, into local rows [r0, max j1];
    the bands are summed into the full grid. raster() stays the reference: its source keys the
    vanilla raster cache, so it is left untouched."""
    from concurrent.futures import ThreadPoolExecutor
    s = _raster_setup(tris, nx, nz, step, offset)
    j0, j1, keys = s[2], s[3], s[-1]
    edges = np.linspace(0, nz, bands + 1).astype(int)

    def band(r0, r1):
        mine = np.flatnonzero((keys >= 0) & (j0 >= r0) & (j0 < r1))
        rows = int(j1[mine].max()) + 1 - r0 if len(mine) else 0
        count = np.zeros(nx * rows, np.uint16)
        low = np.full(nx * rows, np.inf, np.float32)
        high = np.full(nx * rows, -np.inf, np.float32)
        _raster_into(s, mine, count, low, high, r0, 0, nx)
        return r0, rows, count.reshape(rows, nx), low.reshape(rows, nx), high.reshape(rows, nx)

    count = np.zeros((nz, nx), np.uint16)
    low = np.full((nz, nx), np.inf, np.float32)
    high = np.full((nz, nx), -np.inf, np.float32)
    with ThreadPoolExecutor(workers) as pool:
        for r0, rows, c, l, h in pool.map(band, edges[:-1], edges[1:]):
            s = slice(r0, r0 + rows)
            count[s] += c
            np.minimum(low[s], l, out=low[s])
            np.maximum(high[s], h, out=high[s])
    return count, low, high


def raster_box(tris, nx, nz, step, offset=(0., 0.)):
    """raster() of one triangle set over the box of samples its triangles reach: (row0, col0,
    count, low, high). Summed into a full grid (counts add, low / high take min / max, which
    commute) a set of boxes gives raster()'s arrays for all their triangles together."""
    s = _raster_setup(tris, nx, nz, step, offset)
    i0, i1, j0, j1, keys = s[0], s[1], s[2], s[3], s[-1]
    mine = np.flatnonzero(keys >= 0)
    if not len(mine):
        return 0, 0, np.zeros((0, 0), np.uint16), np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
    r0, c0 = int(j0[mine].min()), int(i0[mine].min())
    rows, cols = int(j1[mine].max()) + 1 - r0, int(i1[mine].max()) + 1 - c0
    count = np.zeros(rows * cols, np.uint16)
    low = np.full(rows * cols, np.inf, np.float32)
    high = np.full(rows * cols, -np.inf, np.float32)
    _raster_into(s, mine, count, low, high, r0, c0, cols)
    return r0, c0, count.reshape(rows, cols), low.reshape(rows, cols), high.reshape(rows, cols)


def mesh_rasters(blobs, nx, nz, step, offset, cache, workers=8):
    """raster_bands() of the meshes' triangles (mesh_triangles of each blob, concatenated), and
    their number, assembled from per-mesh raster_box()es cached in `cache` by mesh content and
    raster code: a seam repair pass rewrites a few meshes, so the next scan rasterises only those.
    The cache keeps the current meshes' entries only; meshes may appear or vanish between scans."""
    import hashlib, inspect
    from concurrent.futures import ThreadPoolExecutor
    code = hashlib.sha256(repr((nx, nz, step, tuple(offset))).encode() + ''.join(
        inspect.getsource(f) for f in (mesh_triangles, _raster_setup, _raster_into, raster_box)).encode()).digest()
    names = [hashlib.sha256(code + hashlib.sha256(b).digest()).hexdigest()[:32] + '.npz' for b in blobs]

    def one(job):
        blob, name = job
        path = cache / name
        if path.exists():
            try:
                with np.load(path) as z:
                    return int(z['n']), tuple(int(v) for v in z['at']), z['c'], z['l'], z['h']
            except Exception as error:
                print('Mesh raster cache unreadable, recomputing:', name, error, flush=True)
        tris = mesh_triangles(blob)
        r0, c0, c, l, h = raster_box(tris, nx, nz, step, offset)
        partial = path.with_name(path.stem + f'.{os.getpid()}.partial.npz')
        np.savez(partial, n=len(tris), at=np.array([r0, c0]), c=c, l=l, h=h)
        try:
            os.replace(partial, path)
        except OSError:
            try:
                partial.unlink(missing_ok=True)   # a concurrent scan holds or replaced it
            except OSError:
                pass
        return len(tris), (r0, c0), c, l, h

    cache.mkdir(parents=True, exist_ok=True)
    count = np.zeros((nz, nx), np.uint16)
    low = np.full((nz, nx), np.inf, np.float32)
    high = np.full((nz, nx), -np.inf, np.float32)
    total = 0
    # Meshes with the same content (one name) are rasterised once and merged once each.
    jobs = dict(zip(names, blobs))
    with ThreadPoolExecutor(workers) as pool:
        done = dict(zip(jobs, pool.map(one, zip(jobs.values(), jobs))))
        for n, (r0, c0), c, l, h in (done[name] for name in names):
            total += n
            s = np.s_[r0:r0 + c.shape[0], c0:c0 + c.shape[1]]
            count[s] += c
            np.minimum(low[s], l, out=low[s])
            np.maximum(high[s], h, out=high[s])
    keep = set(names); now = time.time()
    for old in cache.glob('*.npz'):
        # A .partial lives seconds while a scan writes it; one an hour old was left by a killed scan.
        if old.name not in keep and ('.partial' not in old.name or now - old.stat().st_mtime > 3600):
            try:
                old.unlink()
            except OSError:
                pass   # another scan has it open; it is only a stale cache
    return count, low, high, total


def hex_of(x, z_raw):
    # Raw points only (ties resolved as campaign_world, not round-half-even); tile
    # grids use the owner rule tile_hex: a raw tile origin sits on the tie.
    return _frame.raw_hex(x, z_raw)


def components(mask):
    """4-connected components of a hex-grid boolean mask (rows r, cols q)."""
    seen = np.zeros(mask.shape, bool)
    out = []
    for r0, q0 in zip(*np.nonzero(mask)):
        if seen[r0, q0]:
            continue
        cells = []
        dq = deque([(r0, q0)])
        seen[r0, q0] = True
        while dq:
            r, q = dq.popleft()
            cells.append((r, q))
            for rr, qq in ((r - 1, q), (r + 1, q), (r, q - 1), (r, q + 1)):
                if 0 <= rr < mask.shape[0] and 0 <= qq < mask.shape[1] and mask[rr, qq] and not seen[rr, qq]:
                    seen[rr, qq] = True
                    dq.append((rr, qq))
        out.append(cells)
    return out


def pool_to_hex(mask_xz, step=None, weight=None, offset=(0., 0.)):
    """Map a raster on raw-world coordinates (rows z, cols x) to hex counts;
    step None: a tile-cell grid, pooled by the owner rule (tile_hex)."""
    jj, ii = np.nonzero(mask_xz)
    q, r = _frame.tile_hex(ii, jj) if step is None else hex_of((ii + offset[0]) * step, (jj + offset[1]) * step)
    grid = np.zeros((H, W), np.int64)
    np.add.at(grid, (r, q), 1 if weight is None else weight[jj, ii])
    return grid


# tile_list.bin records ('<HIBhhHff', read_tile_list.parse) as one structured array.
TILE_RECORD = np.dtype([('kind', '<u2'), ('path', '<u4'), ('climate', 'u1'), ('x', '<i2'), ('y', '<i2'),
                        ('flags', '<u2'), ('lo', '<f4'), ('hi', '<f4')])


def _bound(v, n):
    """Python slice bound (step 1): negative counts from the end, then clamped to [0, n]."""
    v = np.asarray(v, np.int64)
    return np.clip(np.where(v < 0, v + n, v), 0, n)


def paint(shape, y0, y1, x0, x1):
    """Per-cell count of the slices grid[y0:y1, x0:x1] (Python slice bounds), one per row;
    a difference image, so the cost does not grow with the rectangle area."""
    y0, y1, x0, x1 = _bound(y0, shape[0]), _bound(y1, shape[0]), _bound(x0, shape[1]), _bound(x1, shape[1])
    ok = (y0 < y1) & (x0 < x1)
    y0, y1, x0, x1 = y0[ok], y1[ok], x0[ok], x1[ok]
    d = np.zeros((shape[0] + 1, shape[1] + 1), np.int32)
    for yy, xx, s in ((y0, x0, 1), (y0, x1, -1), (y1, x0, -1), (y1, x1, 1)):
        np.add.at(d, (yy, xx), s)
    return d.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)[:-1, :-1]


def footprint_cells(shape, x, y, w, h):
    """Flat indices of the cells of footprints [y, y+h) x [x, x+w) inside shape, and the
    record (row of x) each belongs to."""
    flat, rows = [], []
    key = w.astype(np.int64) * 65536 + h
    for k in np.unique(key):
        gw, gh = int(k // 65536), int(k % 65536)
        if gw <= 0 or gh <= 0:
            continue
        sel = np.flatnonzero(key == k)
        oz, ox = np.divmod(np.arange(gw * gh), gw)
        cy = y[sel, None] + oz
        cx = x[sel, None] + ox
        ok = (cy >= 0) & (cy < shape[0]) & (cx >= 0) & (cx < shape[1])
        flat.append((cy * shape[1] + cx)[ok])
        rows.append(np.broadcast_to(sel[:, None], ok.shape)[ok])
    return (np.concatenate(flat), np.concatenate(rows)) if flat else (np.zeros(0, np.int64), np.zeros(0, np.int64))


def _pmax(a, b):
    """Python max(a, b) elementwise (a unless b > a; NaN kept where Python keeps it)."""
    return np.where(b > a, b, a)


def _pmin(a, b):
    return np.where(b < a, b, a)


class LFCells:
    """Per 4x4 LF cell min / max (a partial edge cell over its own pixels), and sample_range()
    for many tile placements at once: a placement's block covers whole cells, so its extremes
    are those of a (h+4) x (w+4) cell window, h and w at most `most`. The cell grids are framed
    by neutral cells (2 before, most + 4 after) so clipped windows need no special case, and
    running extremes over 2^k rows (built as needed) reduce a window to two rows of cells."""
    def __init__(self, lf, most):
        self.lf, self.shape, self.most = lf, lf.shape, most
        pad = most + 4
        th, tw = -(-lf.shape[0] // 4), -(-lf.shape[1] // 4)
        self.cells, self.levels = [], []
        for fill, f in ((np.iinfo(lf.dtype).max, np.minimum), (0, np.maximum)):
            a = lf
            if lf.shape != (th * 4, tw * 4):
                a = np.full((th * 4, tw * 4), fill, lf.dtype)
                a[:lf.shape[0], :lf.shape[1]] = lf
            c = f.reduce(f.reduce(a.reshape(th, 4, tw * 4), axis=1).reshape(th, tw, 4), axis=2)
            self.cells.append(c)
            g = np.full((th + 2 + pad, tw + 2 + pad), fill, lf.dtype)
            g[2:2 + th, 2:2 + tw] = c
            self.levels.append([g])

    def sample(self, x, y, w, h):
        """sample_range(lf, x, y, w, h) per placement as (min, max, has): min / max are u16 (only
        where has), has False where the block is empty. Placements with a negative x or y are
        sampled by sample_range() itself (a slice bound there may count from the end)."""
        hl, wl = self.shape
        lo = np.zeros(len(x), self.cells[0].dtype)
        hi = np.zeros(len(x), self.cells[0].dtype)
        has = np.zeros(len(x), bool)
        own = (x < 0) | (y < 0)
        for k in np.flatnonzero(own):
            block = range_block(self.lf, int(x[k]), int(y[k]), int(w[k]), int(h[k]))
            if block.size:
                lo[k], hi[k], has[k] = block.min(), block.max(), True
        empty = (np.minimum(hl, (y + h + 2) * 4) <= np.maximum(0, (y - 2) * 4)) | (np.minimum(wl, (x + w + 2) * 4) <= np.maximum(0, (x - 2) * 4))
        use = ~own & ~empty
        assert (h[use] <= self.most).all() and (w[use] <= self.most).all()
        has |= use
        rh, rw = h + 4, w + 4
        key = rh.astype(np.int64) * 65536 + rw
        for k in np.unique(key[use]):
            gh, gw = int(k // 65536), int(k % 65536)
            sel = np.flatnonzero(use & (key == k))
            lv = gh.bit_length() - 1
            for levels, f in ((self.levels[0], np.minimum), (self.levels[1], np.maximum)):
                while len(levels) <= lv:
                    s = 2 ** (len(levels) - 1)
                    levels.append(f(levels[-1][:-s], levels[-1][s:]))
            cols = x[sel, None] + np.arange(gw)
            for out, levels, f in ((lo, self.levels[0], np.minimum), (hi, self.levels[1], np.maximum)):
                m = levels[lv]
                out[sel] = f.reduce(f(m[y[sel, None], cols], m[(y[sel] + gh - 2 ** lv)[:, None], cols]), axis=1)
        return lo, hi, has


def range_block(lf, x, y, w, h):
    """The LF pixels a fine tile's stored height range is sampled over (2 tiles of margin)."""
    return lf[max(0, (y - 2) * 4):min(lf.shape[0], (y + h + 2) * 4), max(0, (x - 2) * 4):min(lf.shape[1], (x + w + 2) * 4)]


def sample_range(lf, x, y, w, h):
    block = range_block(lf, x, y, w, h)
    return (float(block.min()) / 65535, float(block.max()) / 65535) if block.size else None


def lf_jumps(lf_c, dv, ry, rx, threshold, axis, rows=512):
    """Rows / cols of candidate LF steps along axis above threshold and, where the vanilla
    LF lies (at row ry, col rx; dv its |steps|), above 4x vanilla's own step there + 200
    (the per-pixel rule of the whole-array form); in row chunks on threads."""
    from concurrent.futures import ThreadPoolExecutor
    n = lf_c.shape[0] - (1 if axis == 0 else 0)

    def chunk(s):
        e = min(n, s + rows)
        dc = np.abs(np.diff(lf_c[s:e + (1 if axis == 0 else 0)].astype(np.int32), axis=axis))
        flag = dc > threshold
        a, b = max(s, ry), min(e, ry + dv.shape[0])
        if a < b:
            win = flag[a - s:b - s, rx:rx + dv.shape[1]]
            win &= dc[a - s:b - s, rx:rx + dv.shape[1]] > 4 * dv[a - ry:b - ry] + 200
        jj, ii = np.nonzero(flag)
        return jj + s, ii

    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(chunk, range(0, n, rows)))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def static_sections(args, van, cand, pre, land, vanilla_hex, ids, names, lf_c, lf_v):
    """Height-map, mountain-model, fine-tile and steep-slope checks. None of them
    reads a coarse mesh, and the seam repair loop rewrites only coarse meshes."""
    report = {}
    found = {}

    def add(cls, grid, cause):
        found[cls] = (grid, cause)

    # ---------------- low-frequency height map ----------------
    ry, rx = SHIFT_LF[1], SHIFT_LF[0]
    lf_px = DX / 8
    cliff = np.zeros((H, W), np.int64)
    for axis in (0, 1):
        dv = np.abs(np.diff(lf_v.astype(np.int32), axis=axis))
        # The vanilla step percentile over the same multiset (a sorted int32 rebuild from
        # the step histogram), so the same np.percentile result, without the selection cost.
        threshold = max(800, int(np.percentile(np.repeat(np.arange(65536, dtype=np.int32), np.bincount(dv.ravel(), minlength=65536)), 99.995)))
        jj, ii = lf_jumps(lf_c, dv, ry, rx, threshold, axis)
        q, r = hex_of(ii * lf_px, jj * lf_px)
        np.add.at(cliff, (r, q), 1)
        report.setdefault('lf_thresholds_u16', []).append(threshold)
    add('lf_cliff', cliff, 'Height-map jump far above vanilla at that place: an edited/blended surface meets retained data with a step.')

    # Vanilla cutouts are normally covered by mountain models; flag only those whose model was removed.
    prefab_hex = np.zeros((H, W), bool)
    if not args.skip_prefabs:
        sys.path.insert(0, str(Path(__file__).parent))
        from parse_global_props import parse as parse_props
        from read_root_prefabs import decode as decode_roots

        def roots(data):
            entries, base = parse_props(data)
            offs = sorted({o for _, o in entries}) + [len(data) - base]
            nxt = {o: offs[i + 1] for i, o in enumerate(offs[:-1])}
            look = {p: data[base + o:base + nxt[o]] for p, o in entries}
            return look, decode_roots(look[pre + 'bmd_objects.bin'])[1]
        vlook, vroots = roots(van.read(pre + 'global_props.bin'))
        _, croots = roots(cand.read(pre + 'global_props.bin'))
        kept = {r['path'] for r in croots}
        dz = DX * 2 / math.sqrt(3)
        removed = 0
        for rec in vroots:
            if rec['path'] in kept:
                continue
            stem = rec['path'][:-4]
            leaves = [p for p in vlook if p.startswith(stem + '.') and p.endswith('.31.bin')]
            models = [m.group().lower() for p in leaves for m in re.finditer(rb'(?i)rigidmodels/[a-zA-Z0-9_./]+', vlook[p])]
            if not any(b'/campaign/mountains/' in m or b'/campaign/rocks/' in m for m in models):
                continue
            removed += 1
            for leaf in leaves:
                body = vlook[leaf]
                for pos in range(36, len(body) - 12):
                    x, y, z = struct.unpack_from('<3f', body, pos)
                    if not (0 < x < 596 and 0 < z < 543 and -10 < y < 100):
                        continue
                    rot = struct.unpack_from('<9f', body, pos - 36)
                    if not all(math.isfinite(v) and abs(v) < 100 for v in rot) or sum(abs(v) for v in rot) < .1:
                        continue
                    q = round((x + rec['matrix'][12]) / DX)
                    r = round((z + rec['matrix'][14]) / dz - (q % 2) * .5)
                    q, r = q + SHIFT_HEX[0], r + SHIFT_HEX[1]
                    if 0 <= q < W and 0 <= r < H:
                        prefab_hex[max(0, r - 3):r + 4, max(0, q - 3):q + 4] = True
        report['removed_vanilla_mountain_models'] = removed

    # ---------------- fine tiles ----------------
    TW, TH = 2 * W, 2 * H + 1
    dims_cache = {}

    def dims(path):
        if path not in dims_cache:
            m = re.search(r'generic_(\d+)x(\d+)', path)
            if m:
                dims_cache[path] = tuple(map(int, m.groups()))
            else:
                parts = path.strip(B).split(B)
                key = 'terrain/tiles/campaign/_tile_database/tiles/' + parts[-2] + '_' + parts[-1] + '.bin'
                dims_cache[path] = decode_dimensions(van.read(key)) if key in van.index else (1, 1)
        return dims_cache[path]

    tc_data, tv_data = cand.read(pre + 'tile_list.bin'), van.read(pre + 'tile_list.bin')
    tc = parse_tiles(tc_data)
    tv = parse_tiles(tv_data)
    climates = tc[1]
    report['fine_records'] = {'candidate': len(tc[4]), 'vanilla': len(tv[4]), 'climates': climates}

    def columns(tiles, data):
        """The records as arrays (the same rows parse() decoded), with each footprint
        (placement_footprint of dims) and whether its path is a generic tile."""
        rec = np.frombuffer(data, TILE_RECORD, count=len(tiles[4]), offset=tiles[3])
        c = {k: rec[k].astype(np.int64) for k in ('path', 'climate', 'x', 'y', 'flags')}
        c['lo'], c['hi'] = rec['lo'].astype(np.float64), rec['hi'].astype(np.float64)
        size = np.zeros((len(tiles[0]), 2), np.int64)
        for p in np.unique(c['path']):
            size[p] = dims(tiles[0][p])
        turn = (c['flags'] & QUARTER_TURN_FLAGS) != 0
        s = size[c['path']]
        c['w'], c['h'] = np.where(turn, s[:, 1], s[:, 0]), np.where(turn, s[:, 0], s[:, 1])
        c['generic'] = np.array(['/generic/' in p.replace(B, '/') for p in tiles[0]], bool)[c['path']]
        return c
    cc, cv = columns(tc, tc_data), columns(tv, tv_data)

    def coverage(c, shift):
        # Fine tiles per cell, and the climate of the last record (in list order) covering it.
        flat, rows = footprint_cells((TH, TW), c['x'] + shift[0], c['y'] + shift[1], c['w'], c['h'])
        cov = np.bincount(flat, minlength=TH * TW).astype(np.uint16).reshape(TH, TW)
        last = np.full(TH * TW, -1, np.int64)
        np.maximum.at(last, flat, rows)
        clim = np.where(last >= 0, c['climate'][last], 255).astype(np.uint8).reshape(TH, TW)
        return cov, clim

    cov_c, clim_c = coverage(cc, (0, 0))
    cov_v, clim_v = coverage(cv, SHIFT_TILE)
    tq, tr = _frame.tile_hex(np.arange(TW)[None], np.arange(TH)[:, None])
    t_land, t_van = land[tr, tq], vanilla_hex[tr, tq]
    t_land_full = t_land.copy()
    vland = t_land & t_van & (cov_v > 0)
    report['fine_model_check'] = {
        'vanilla_land_cells_covered_fraction': float((cov_v[t_land & t_van] > 0).mean()),
        'note': 'Footprints are database width x height, swapped for quarter-turn placements (flags 0x20/0x80).'}
    add('fine_uncovered', pool_to_hex(t_land & (cov_c == 0) & (cov_v > 0)),
        'Land cell has no runtime fine tile although vanilla had one: a tile was removed and nothing replaced it.')
    # Any current land cell without a land fine tile, inside or outside the vanilla canvas.
    k = ~np.array(['generic_sea' in p for p in tc[0]], bool)[cc['path']]
    x, y, w, h = cc['x'][k], cc['y'][k], cc['w'][k], cc['h'][k]
    land_cov = paint((TH, TW), np.maximum(0, y), np.maximum(0, y + h), np.maximum(0, x), np.maximum(0, x + w)) > 0
    add('fine_uncovered_land', pool_to_hex(t_land & ~land_cov),
        'Land cell has no land fine tile at all (sea tiles ignored); the coarse surface shows its plain texture in tile-sized squares.')
    patch = (cov_c > 0) & vland & (clim_c != clim_v) & (clim_v != 255)
    add('fine_climate_patch', pool_to_hex(patch),
        'Replacement fine tiles use a different climate texture set from vanilla at the same place '
        '(hand-made latitude biome in polish_campaign_terrain.py / repair_native_placements.py).')
    del tq, tr

    # Placement keys (path text, climate, x, y, flags), vanilla's in the canvas frame; equal keys
    # share a group. Per vanilla group its last record (as a dict filled in list order) and first.
    text = {}
    pid_c = np.array([text.setdefault(p, len(text)) for p in tc[0]], np.int64)
    pid_v = np.array([text.setdefault(p, len(text)) for p in tv[0]], np.int64)
    nc, nv = len(cc['x']), len(cv['x'])
    ka = np.concatenate([(pid[c['path']] * 256 + c['climate']) * 65536 + c['flags'] for pid, c in ((pid_c, cc), (pid_v, cv))])
    kb = np.concatenate([(c['x'] + s[0] + 2 ** 20) * 2 ** 21 + c['y'] + s[1] + 2 ** 20 for c, s in ((cc, (0, 0)), (cv, SHIFT_TILE))])
    order = np.lexsort((kb, ka))
    sa, sb = ka[order], kb[order]
    new = np.ones(len(order), bool)
    new[1:] = (sa[1:] != sa[:-1]) | (sb[1:] != sb[:-1])
    group = np.empty(len(order), np.int64)
    group[order] = np.cumsum(new) - 1
    gc, gv = group[:nc], group[nc:]
    last = np.full(int(new.sum()), -1, np.int64)
    np.maximum.at(last, gv, np.arange(nv))
    first = np.full(len(last), nv, np.int64)
    np.minimum.at(first, gv, np.arange(nv))
    present = np.zeros(len(last), bool)
    present[gc] = True
    donor = last[gc]
    del ka, kb, order, sa, sb, new, group

    most = int(max(cc['w'].max(initial=1), cc['h'].max(initial=1)))
    cells_c, cells_v = LFCells(lf_c, most), LFCells(lf_v, most)
    n_lo, n_hi, has = cells_c.sample(cc['x'], cc['y'], cc['w'], cc['h'])
    n0, n1 = n_lo.astype(np.float64) / 65535, n_hi.astype(np.float64) / 65535
    lo, hi = cc['lo'], cc['hi']
    dev = _pmax(np.abs(n0 - lo), np.abs(n1 - hi))
    keep = has & ((donor >= 0) | cc['generic'])
    d = np.flatnonzero(has & (donor >= 0))
    di = donor[d]
    # Vanilla range = sample widened by a per-placement padding (never narrowed).
    o_lo, o_hi, o_has = cells_v.sample(cv['x'][di], cv['y'][di], cc['w'][d], cc['h'][d])
    o0, o1 = o_lo.astype(np.float64) / 65535, o_hi.astype(np.float64) / 65535
    d6, d7 = cv['lo'][di], cv['hi'][di]
    keep[d[~o_has | (d6 - o0 > 2e-4) | (d7 - o1 < -2e-4)]] = False
    moved = _pmax(np.abs(n0[d] - o0), np.abs(n1[d] - o1)) > 5e-5
    e0 = np.where(moved, n0[d] + _pmin(0., d6 - o0), d6)
    e1 = np.where(moved, n1[d] + _pmax(0., d7 - o1), d7)
    dev[d] = _pmax(np.abs(e0 - lo[d]), np.abs(e1 - hi[d]))
    s = keep & (dev > 1e-3)
    del n_lo, n_hi, has, n0, n1, dev, keep, o_lo, o_hi, o_has, o0, o1, d6, d7, moved, e0, e1
    stale = paint((TH, TW), np.maximum(0, cc['y'][s]), cc['y'][s] + cc['h'][s], np.maximum(0, cc['x'][s]), cc['x'][s] + cc['w'][s]) > 0
    stale_count = Counter()
    kinds = [('retained', s & (donor >= 0)), ('generated', s & (donor < 0))]
    for name, m in sorted(kinds, key=lambda kv: int(np.argmax(kv[1])) if kv[1].any() else nc):
        if m.any():
            stale_count[name] += int(m.sum())
    removed_special = np.zeros((TH, TW), bool)
    groups = np.flatnonzero((last >= 0) & ~present)
    for g in groups[np.argsort(first[groups])]:
        rec = tv[4][last[g]]
        path = tv[0][rec[1]]
        if '/generic/' in path.replace(B, '/'):
            continue
        w, h = placement_footprint(dims(path), rec[5])
        x, y = rec[3], rec[4]
        kx, ky = rec[3] + SHIFT_TILE[0], rec[4] + SHIFT_TILE[1]
        a = lf_v[max(0, y * 4):(y + h) * 4, max(0, x * 4):(x + w) * 4]
        b = lf_c[max(0, (y + SHIFT_TILE[1]) * 4):(y + SHIFT_TILE[1] + h) * 4, max(0, (x + SHIFT_TILE[0]) * 4):(x + SHIFT_TILE[0] + w) * 4]
        if a.size and a.shape == b.shape and np.array_equal(a, b):
            removed_special[ky:ky + h, kx:kx + w] = True
            stale_count['special_removed_ground_unchanged'] += 1
    removed_special &= t_land_full
    report['fine_counts'] = dict(stale_count)
    add('fine_range_stale', pool_to_hex(stale),
        'Fine tile height range disagrees with the current height map by more than 0.001 (about 0.014 height); '
        'the whole tile renders shifted, giving a square step against its neighbours.')
    add('fine_special_removed', pool_to_hex(removed_special),
        'A vanilla river/road/mountain/coast tile on land was removed although the height map under it is unchanged.')

    # ---------------- steep ground under flat generic tiles ----------------
    # Planar generic textures stretch on steep LF. Vanilla covers its steep ground
    # with mountain/cliff tiles, so compare authored land against vanilla land.
    def generic_mask(c, shape):
        k = c['generic']
        x, y, w, h = c['x'][k], c['y'][k], c['w'][k], c['h'][k]
        return paint(shape, np.maximum(0, y), np.clip(y + h, 0, shape[0]), np.maximum(0, x), np.clip(x + w, 0, shape[1])) > 0

    def cell_slope(cells):
        th, tw = cells.shape[0] // 4, cells.shape[1] // 4
        return (cells.cells[1][:th, :tw].astype(np.float32) - cells.cells[0][:th, :tw]) / 65535 * _frame.GAIN / (DX / 2)

    design = json.loads(_frame.DESIGN_PATH.read_text(encoding='utf-8'))
    authored_ids = [names.index(r['region']) for r in design['regions'] if r['region'] in names]
    sc = cell_slope(cells_c)[:TH, :TW]
    gc = generic_mask(cc, sc.shape)
    hq, hr = _frame.tile_hex(np.arange(sc.shape[1])[None], np.arange(sc.shape[0])[:, None])
    authored_cells = np.isin(ids[hr, hq], authored_ids) & land[hr, hq]
    steep_new = authored_cells & gc & (sc > 1.5)
    # Cells wholly under the frontier river water (every LF pixel of the cell below polish's stamped
    # river level, analysis/river-water-level.npy) are bed, not visible ground: the deep beds under
    # the water (river_channels RIVER BEDS) are cut into the land hexes the water outline covers.
    # They are reported apart (steep_generic_submerged); a cell partly above the water still counts.
    submerged, sub_note = submerged_cells(cells_c, sc.shape, args.map)
    steep_sub = steep_new & submerged
    steep_new &= ~submerged
    sv = cell_slope(cells_v)
    gv = generic_mask(cv, sv.shape)
    vq, vr = _frame.tile_hex(np.arange(sv.shape[1])[None], np.arange(sv.shape[0])[:, None])
    native_land = land[_frame.NATIVE_HEX_SLICE]
    vq, vr = np.clip(vq, 0, _frame.NATIVE_W - 1), np.clip(vr, 0, _frame.NATIVE_H - 1)
    vland_cells = native_land[vr, vq]
    steep_van = vland_cells & gv & (sv > 1.5)
    ratio = float(steep_new.sum() / max(1, authored_cells.sum()))
    vratio = float(steep_van.sum() / max(1, vland_cells.sum()))
    report['steep_generic'] = {'slope_threshold': 1.5, 'authored_land_cells': int(authored_cells.sum()),
                               'authored_steep_generic_cells': int(steep_new.sum()), 'authored_ratio': ratio,
                               'vanilla_land_cells': int(vland_cells.sum()), 'vanilla_steep_generic_cells': int(steep_van.sum()),
                               'vanilla_ratio': vratio, 'steep_generic_submerged': int(steep_sub.sum()),
                               'submerged_rule': sub_note}
    add('steep_generic_authored', pool_to_hex(steep_new),
        'Authored land steeper than 1.5 (height per horizontal unit) covered only by planar generic tiles; '
        'textures stretch into vertical walls. Vanilla uses mountain/cliff tiles on such slopes.')
    del hq, hr, vq, vr
    return found, report, prefab_hex


SUBMERGED_EPS = 1e-3   # world units: a cell counts as under the river water when its highest LF pixel is this far below


def submerged_cells(cells, shape, map_path):
    """(TH, TW) bool: tile cells whose highest LF pixel lies under the frontier river level by SUBMERGED_EPS,
    and a note. Empty (with the reason) when the level file is missing or stamped for another map."""
    import hashlib
    lp = _frame.WORKSPACE / 'analysis/river-water-level.npy'
    sp = lp.with_suffix('.json')
    empty = np.zeros(shape, bool)
    if not (lp.exists() and sp.exists()):
        return empty, 'no river level file: nothing excluded'
    stamp = json.loads(sp.read_text(encoding='utf-8'))
    if stamp.get('map_sha256') != hashlib.sha256(Path(map_path).read_bytes()).hexdigest():
        return empty, 'river level stamped for another map.hex: nothing excluded'
    lvl = np.load(lp)
    hi = _frame.LEVEL + (cells.cells[1].astype(np.float64) / 65535 - _frame.NORM) * _frame.GAIN
    th, tw = min(shape[0], lvl.shape[0], hi.shape[0]), min(shape[1], lvl.shape[1], hi.shape[1])
    out = empty.copy()
    L = lvl[:th, :tw].astype(np.float64)
    out[:th, :tw] = np.isfinite(L) & (hi[:th, :tw] < np.nan_to_num(L, nan=-1e9) - SUBMERGED_EPS)
    return out, f'every LF pixel of the cell under the stamped river level by {SUBMERGED_EPS}'


def _static_worker(args, van, cand, pre, land, vanilla_hex, ids, names):
    """static_sections() in a worker process; it decodes the height maps itself
    rather than receive them pickled (the candidate's is half a gigabyte)."""
    return static_sections(args, van, cand, pre, land, vanilla_hex, ids, names,
                           lf_pixels(cand.read(pre + 'lf_height_map.dds')), lf_pixels(van.read(pre + 'lf_height_map.dds')))



# v21e packaging: the final gate died writing ground-seam-masks.npz over a file another process held open (OSError
# 22 on Windows) after a 23-minute pipeline. Outputs go to a temporary file in the same folder and replace the old
# one (os.replace, retried while another reader holds it); the preflight below fails at the start, not at the end.
def _replace_atomic(path, write, tries=8, wait=2.):
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as f:
        write(f)
    for k in range(tries):
        try:
            os.replace(tmp, path); return
        except OSError as e:
            if k == tries - 1:
                raise OSError(f'{path} is held open by another process (left the new output at {tmp})') from e
            time.sleep(wait)


def _check_writable(out):
    """Preflight: every output this audit replaces can be replaced now (not held open by another process)."""
    for name in ('ground-seam-masks.npz', 'ground-seam-scan.json'):
        p = out / name
        if p.exists():
            try:
                with open(p, 'r+b'):
                    pass
            except OSError as e:
                raise SystemExit(f'preflight: {p} cannot be replaced ({e}); close the reader holding it and rerun')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--pack', type=Path, help='built pack to scan; must match --map')
    source.add_argument('--stage', type=Path, help='read-only staged source scan; final built-pack scan remains required')
    ap.add_argument('--variant', default='3k_dlc07_main_map')
    ap.add_argument('--map', type=Path, default=_frame.WORKSPACE/'work/final-map/map.hex')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--top', type=int, default=40, help='clusters listed per class')
    ap.add_argument('--skip-prefabs', action='store_true', help='skip the slow mountain-model scan')
    ap.add_argument('--strict-first-five', action='store_true', help='fail if any of the first five audit rows has defects')
    ap.add_argument('--max-steep-ratio', type=float, help='fail if authored steep-generic ratio exceeds this value')
    args = ap.parse_args()
    out = args.out.resolve()
    stage = (_frame.WORKSPACE/'work/release-input').resolve()
    for protected in (GAME.resolve(), stage):
        assert not (out == protected or out.is_relative_to(protected)), f'--out must not be inside {protected}'
    out.mkdir(parents=True, exist_ok=True)
    _check_writable(out)

    pre = f'terrain/campaigns/{args.variant}/'
    van = Packs([GAME / 'terrain.pack', GAME / 'fast.pack', GAME / 'data.pack'])
    cand = StageAssets(args.stage,van) if args.stage else Packs([args.pack])
    report = {'pack': str(args.pack) if args.pack else None, 'stage':str(args.stage) if args.stage else None, 'variant': args.variant, 'map': str(args.map), 'classes': {}, 'notes': []}

    # Logical map: land, regions, vanilla footprint.
    meta = read_hex(args.map)
    assert (meta['width'], meta['height']) == (W, H)
    cells = np.frombuffer(args.map.read_bytes()[-4 - W * H * 16:-4], np.uint8).reshape(H, W, 16)
    land = (cells[:, :, 0] & 3) != 1
    ids = (cells[:, :, 1].astype(int) << 5) + (cells[:, :, 0].astype(int) >> 3) - 1
    names = meta['land_regions'] + meta['sea_regions']
    vanilla_hex = np.zeros((H, W), bool)
    vanilla_hex[_frame.NATIVE_HEX_SLICE] = True
    interior = land.copy()
    for dr, dq in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)):
        interior &= np.roll(np.roll(land, dr, 0), dq, 1)
    interior[:3] = interior[-3:] = False   # the mesh legitimately ends at the canvas rim
    interior[:, :3] = interior[:, -3:] = False
    hex_defects = {}

    def add(cls, grid, cause):
        hex_defects[cls] = (grid > 0, grid, cause)

    # Everything but the coarse meshes is unchanged between seam repair passes:
    # reuse the static sections when their inputs (and this code) are the same.
    import hashlib, inspect
    lf_dds = cand.read(pre + 'lf_height_map.dds')
    lf_c = lf_pixels(lf_dds)
    lf_v = lf_pixels(van.read(pre + 'lf_height_map.dds'))
    static_id = hashlib.sha256(repr((args.variant, args.skip_prefabs, W, H, [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in van.packs])).encode())
    here = Path(__file__).resolve().parent
    for name in ('audit_ground_seams.py', 'campaign_world.py', 'native_tile_dimensions.py', 'read_tile_list.py', 'parse_global_props.py', 'read_root_prefabs.py', 'analyze_reference.py', 'fast_hex.py'):
        static_id.update(hashlib.sha256((here / name).read_bytes()).digest())
    # The DDS bytes themselves (header and the pixels lf_c decodes), not a copy of the pixels.
    for blob in (lf_dds, cand.read(pre + 'tile_list.bin'), cand.read(pre + 'global_props.bin'), args.map.read_bytes(), _frame.DESIGN_PATH.read_bytes()):
        static_id.update(hashlib.sha256(blob).digest())
    for lp in (_frame.WORKSPACE / 'analysis/river-water-level.npy', _frame.WORKSPACE / 'analysis/river-water-level.json'):   # steep_generic_submerged
        static_id.update(hashlib.sha256(lp.read_bytes() if lp.exists() else b'').digest())
    static_cache = _frame.WORKSPACE / 'analysis/cache' / f'seam-static-{args.variant}-{static_id.hexdigest()[:24]}.npz'
    found = None
    # One file (grids plus the report as JSON bytes), replaced atomically; anything
    # unreadable (an interrupted older run) is simply recomputed.
    if static_cache.exists():
        try:
            with np.load(static_cache) as z:
                grids = {k: z[k] for k in z.files}
            saved = json.loads(grids.pop('_meta').tobytes().decode('utf-8'))
            found = {cls: (grids[cls], cause) for cls, cause in saved['causes']}
            static_report, prefab_hex = saved['report'], grids['_prefab_hex']
            print('Static seam sections reused:', static_cache.name, flush=True)
        except Exception as error:
            print('Static seam cache unreadable, recomputing:', error, flush=True)
            found = None
    # The candidate coarse raster does not depend on the static sections: on a cache miss
    # they run in a worker process beside it (a thread loses most of the overlap to the GIL).
    step = DX / 4
    nx = int(_frame.WORLD_WIDTH / step) + 3
    nz = int((H + 1) * DX / step) + 3
    # The meshes present now (a step may add or drop some), each rasterised once per content.
    blobs = [cand.read(k) for k in cand.keys(pre + 'global_meshes/land_mesh_', '.rigid_model_v2')]
    mesh_cache = _frame.WORKSPACE / 'analysis/cache' / f'seam-mesh-raster-{args.variant}'
    fresh = found is None
    if fresh:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(1) as pool:
            job = pool.submit(_static_worker, args, van, cand, pre, land, vanilla_hex, ids, names)
            cc, cl, ch, n_tri_c = mesh_rasters(blobs, nx, nz, step, SAMPLE_OFFSET, mesh_cache)
            found, static_report, prefab_hex = job.result()
    else:
        cc, cl, ch, n_tri_c = mesh_rasters(blobs, nx, nz, step, SAMPLE_OFFSET, mesh_cache)
    del blobs
    if fresh:
        static_cache.parent.mkdir(parents=True, exist_ok=True)
        for old in static_cache.parent.glob(f'seam-static-{args.variant}-*'):
            try:
                old.unlink()
            except OSError:
                pass   # another scan has it open; it is only a stale cache
        meta = json.dumps({'causes': [[cls, cause] for cls, (_, cause) in found.items()], 'report': static_report}, ensure_ascii=False).encode('utf-8')
        partial = static_cache.with_name(static_cache.stem + f'.{os.getpid()}.partial.npz')
        np.savez_compressed(partial, _meta=np.frombuffer(meta, np.uint8), _prefab_hex=prefab_hex, **{cls: grid for cls, (grid, _) in found.items()})
        try:
            os.replace(partial, static_cache)
        except OSError:
            pass   # a concurrent scan cleared it; the next scan recomputes
    del lf_c, lf_v, lf_dds
    report['lf_thresholds_u16'] = static_report['lf_thresholds_u16']
    add('lf_cliff', *found['lf_cliff'])

    # ---------------- coarse land meshes ----------------
    tri_v = np.concatenate([mesh_triangles(van.read(k), (_frame.NATIVE_SHIFT_RAW[0], 0, _frame.NATIVE_SHIFT_RAW[1]))
                            for k in van.keys(pre + 'global_meshes/land_mesh_', '.rigid_model_v2')])
    report['coarse_triangles'] = {'candidate': n_tri_c, 'vanilla': len(tri_v)}
    # Vanilla geometry never changes: reuse its raster when the triangles, the grid and raster() are the same.
    # Replaced atomically like the static cache; anything unreadable (an interrupted older run) is recomputed.
    key = hashlib.sha256(np.ascontiguousarray(tri_v).tobytes() + repr((nx, nz, step, SAMPLE_OFFSET)).encode() + inspect.getsource(raster).encode()).hexdigest()[:24]
    vcache = _frame.WORKSPACE / 'analysis/cache' / f'seam-vanilla-raster-{key}.npz'
    vc = None
    if vcache.exists():
        try:
            with np.load(vcache) as z:
                vc, vl, vhh = z['c'], z['l'], z['h']
        except Exception as error:
            print('Vanilla raster cache unreadable, recomputing:', error, flush=True)
            vc = None
    if vc is None:
        vc, vl, vhh = raster(tri_v, nx, nz, step, SAMPLE_OFFSET)
        vcache.parent.mkdir(parents=True, exist_ok=True)
        partial = vcache.with_name(vcache.stem + f'.{os.getpid()}.partial.npz')
        np.savez(partial, c=vc, l=vl, h=vhh)
        try:
            os.replace(partial, vcache)
        except OSError:
            partial.unlink(missing_ok=True)   # another scan holds or replaced it; the next scan reuses or recomputes
    # Sample columns and rows broadcast (hex_of is elementwise; q depends on x alone).
    sq, sr = hex_of((np.arange(nx)[None] + SAMPLE_OFFSET[0]) * step, (np.arange(nz)[:, None] + SAMPLE_OFFSET[1]) * step)
    s_land, s_inner, s_van = land[sr, sq], interior[sr, sq], vanilla_hex[sr, sq]
    hole = s_land & (cc == 0)
    add('coarse_hole_native_removed', pool_to_hex(hole & (vc > 0), step, offset=SAMPLE_OFFSET),
        'A vanilla coarse triangle here was deleted and the area was not refilled; the ocean background shows. '
        'fill_polished_mesh_gaps.py removes triangles by centroid but only refills grid faces whose own centroid is edited.')
    add('coarse_hole_new_land', pool_to_hex(hole & s_inner & ~s_van, step, offset=SAMPLE_OFFSET),
        'Interior land outside the vanilla canvas has no coarse triangle at all.')
    vanilla_cutout = hole & (vc == 0) & s_van
    stack = (cc >= 2) & ((ch - cl) > .15)
    add('coarse_stack', pool_to_hex(stack, step, offset=SAMPLE_OFFSET),
        'Overlapping coarse faces disagree by more than 0.15 in height (retained native faces against regenerated ones).')
    top_c = np.where(cc > 0, ch, np.nan)
    top_v = np.where(vc > 0, vhh, np.nan)
    stepmask = np.zeros(cc.shape, bool)
    for axis in (0, 1):
        dc = np.abs(np.diff(top_c, axis=axis))
        dv = np.abs(np.diff(top_v, axis=axis))
        finite_v = dv[np.isfinite(dv)]
        threshold = max(.4, float(np.percentile(finite_v, 99.99)))
        flag = (dc > threshold) & (~np.isfinite(dv) | (dc > 4 * dv + .1))
        flag = np.nan_to_num(flag, nan=False).astype(bool)
        if axis == 0:
            stepmask[:-1] |= flag
        else:
            stepmask[:, :-1] |= flag
    add('coarse_step', pool_to_hex(stepmask & s_land, step, offset=SAMPLE_OFFSET),
        'Coarse surface jumps far above the vanilla jump at the same place (seam between retained and regenerated geometry).')
    del cl, ch, vl, vhh, top_c, top_v

    if 'removed_vanilla_mountain_models' in static_report:
        report['removed_vanilla_mountain_models'] = static_report['removed_vanilla_mountain_models']
    exposed = vanilla_cutout & prefab_hex[sr, sq]
    add('coarse_hole_prefab_removed', pool_to_hex(exposed, step, offset=SAMPLE_OFFSET),
        'A vanilla coarse cutout that used to sit under a mountain/rock model; the model was removed, so the cutout shows water.')
    report['coarse_vanilla_cutouts_still_covered_samples'] = int((vanilla_cutout & ~prefab_hex[sr, sq]).sum())
    del cc, vc, hole, stack, stepmask, sq, sr, s_land, s_inner, s_van, vanilla_cutout, exposed
    for name, value in static_report.items():
        if name not in ('lf_thresholds_u16', 'removed_vanilla_mountain_models'):
            report[name] = value
    for cls, (grid, cause) in found.items():
        if cls != 'lf_cliff':
            add(cls, grid, cause)

    # ---------------- clusters ----------------
    def region_name(r, q):
        i = ids[r, q]
        return names[i] if 0 <= i < len(names) else None

    for cls, (mask, grid, cause) in hex_defects.items():
        comps = components(mask)
        rows = []
        for comp in comps:
            rr = np.array([c[0] for c in comp])
            qq = np.array([c[1] for c in comp])
            regions = Counter(region_name(r, q) for r, q in comp)
            rows.append({'hexes': len(comp), 'samples': int(grid[rr, qq].sum()),
                         'q': [int(qq.min()), int(qq.max())], 'r': [int(rr.min()), int(rr.max())],
                         'centre': [int(round(qq.mean())), int(round(rr.mean()))],
                         'inside_vanilla_canvas': bool(vanilla_hex[int(round(rr.mean())), int(round(qq.mean()))]),
                         'regions': [k for k, _ in regions.most_common(3)]})
        rows.sort(key=lambda x: -x['samples'])
        report['classes'][cls] = {'cause': cause, 'hexes': int(mask.sum()), 'clusters': len(rows), 'top': rows[:args.top]}

    # ---------------- overview image ----------------
    colours = {'coarse_hole_native_removed': (255, 0, 0), 'coarse_hole_prefab_removed': (255, 120, 0),
               'coarse_hole_new_land': (255, 0, 200), 'coarse_stack': (255, 255, 0), 'coarse_step': (200, 200, 0),
               'fine_uncovered': (0, 255, 255), 'fine_uncovered_land': (255, 0, 255), 'fine_range_stale': (0, 120, 255), 'fine_special_removed': (160, 0, 255),
               'fine_climate_patch': (0, 200, 0), 'lf_cliff': (255, 255, 255)}
    img = np.zeros((H, W, 3), np.uint8)
    img[land] = (70, 70, 70)
    img[~land] = (20, 30, 60)
    img[land & ~vanilla_hex] = (90, 80, 60)
    for cls in colours:
        if cls in hex_defects:
            img[hex_defects[cls][0]] = colours[cls]
    images = [(img, 'ground-seam-overview.png')]
    for cls in colours:
        if cls in hex_defects:
            layer = np.zeros((H, W, 3), np.uint8)
            layer[land] = (60, 60, 60)
            layer[hex_defects[cls][0]] = colours[cls]
            images.append((layer, f'{cls}.png'))
    # PNG encoding (zlib) releases the GIL: the files are written on threads.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda job: Image.fromarray(job[0][::-1]).resize((W * 2, H * 2), Image.NEAREST).save(out / job[1]), images))
    _replace_atomic(out / 'ground-seam-masks.npz', lambda f: np.savez_compressed(f, **{k: v[1] for k, v in hex_defects.items()}))
    report['legend'] = {k: list(v) for k, v in colours.items()}
    first_five=('coarse_hole_native_removed','lf_cliff','coarse_step','coarse_stack','coarse_hole_prefab_removed','fine_range_stale')
    report['first_five_passed']=all(report['classes'][k]['hexes']==0 for k in first_five)
    report['image_orientation'] = 'north up (row 0 = highest r); x = q'
    _replace_atomic(out / 'ground-seam-scan.json', lambda f: f.write((json.dumps(report, ensure_ascii=False, indent=1) + '\n').encode('utf-8')))
    for cls, v in report['classes'].items():
        print(f"{cls:28s} hexes {v['hexes']:7d} clusters {v['clusters']:5d}", flush=True)
    print('Report:', out / 'ground-seam-scan.json')
    if args.max_steep_ratio is not None and report['steep_generic']['authored_ratio'] > args.max_steep_ratio:
        print('Authored steep generic ratio', report['steep_generic']['authored_ratio'], 'exceeds', args.max_steep_ratio)
        sys.exit(1)
    if args.strict_first_five and not report['first_five_passed']:
        raise SystemExit('First five audit rows still contain defects; terrain acceptance blocked')


if __name__ == '__main__':
    main()

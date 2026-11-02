"""Where two packs differ for the live renderer (render_live_view), as world grids, so that art_review can reuse the
base render of every camera whose view misses all of them.

Two grids of 1 x 1 world-unit cells (rows south-up, cell (x // 1, z // 1)):
  height   inputs that move the ground or water surface: lf_height_map, lf_sea_height_map, land / sea / lake mesh
           triangles. The renderer frames its mesh raster and its march bounds on the whole x +- 6 dist window, so a
           camera is affected by a height change anywhere in that window.
  surface  inputs drawn only where a ray lands: global_blend texels, tile-list holes, object rows (trees, props,
           tile props, town boxes: render_live_objects.Objects, the renderer's own rows). A camera is affected by
           a surface change near its ground hits.
Pack keys the renderer reads that none of these structures read (read tracking, see Reads) are 'global': any
change to one of them affects every camera.

Each structure is compared in the renderer's own parsed form (arrays, triangles, object rows), not by pack key:
the packaging renumbers bmd_objects / water model files between builds.
"""
import hashlib, zlib

import numpy as np

CELL = 1.0            # world units per change-grid cell


# ------------------------------------------------------------------ grids
class Grid:
    def __init__(self, width, height, cell=CELL):
        self.cell = cell
        self.a = np.zeros((int(np.ceil(height / cell)) + 1, int(np.ceil(width / cell)) + 1), bool)

    def mark(self, x0, z0, x1, z1):
        """Mark the cells of world boxes (scalars or arrays)."""
        H, W = self.a.shape
        c0 = np.clip(np.floor(np.asarray(x0, float) / self.cell), 0, W - 1).astype(np.int64)
        c1 = np.clip(np.floor(np.asarray(x1, float) / self.cell), 0, W - 1).astype(np.int64)
        r0 = np.clip(np.floor(np.asarray(z0, float) / self.cell), 0, H - 1).astype(np.int64)
        r1 = np.clip(np.floor(np.asarray(z1, float) / self.cell), 0, H - 1).astype(np.int64)
        for a, b, c, d in zip(np.atleast_1d(c0), np.atleast_1d(c1), np.atleast_1d(r0), np.atleast_1d(r1)):
            self.a[c:d + 1, a:b + 1] = True

    def any(self, x0, z0, x1, z1):
        H, W = self.a.shape
        c0, c1 = int(np.clip(np.floor(x0 / self.cell), 0, W - 1)), int(np.clip(np.floor(x1 / self.cell), 0, W - 1))
        r0, r1 = int(np.clip(np.floor(z0 / self.cell), 0, H - 1)), int(np.clip(np.floor(z1 / self.cell), 0, H - 1))
        if x1 < 0 or z1 < 0 or x0 > W * self.cell or z0 > H * self.cell:
            return False
        return bool(self.a[r0:r1 + 1, c0:c1 + 1].any())

    def boxes(self):
        """Bounding boxes of the marked cells' connected blobs (for the report), largest first."""
        from scipy import ndimage
        lab, n = ndimage.label(self.a)
        out = []
        for sl in ndimage.find_objects(lab):
            r, c = sl
            out.append([c.start * self.cell, r.start * self.cell, c.stop * self.cell, r.stop * self.cell])
        return sorted(out, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


# ------------------------------------------------------------------ array inputs
def array_blocks(a, b, block=32):
    """(row0, row1, col0, col1) pixel blocks where two rasters differ (north-first rows), or None if their shapes
    differ."""
    if a.shape != b.shape:
        return None
    H, W = a.shape[:2]
    out = []
    for r0 in range(0, H, block):
        da, db = np.asarray(a[r0:r0 + block]), np.asarray(b[r0:r0 + block])
        d = da != db
        if d.ndim == 3:
            d = d.any(2)
        if not d.any():
            continue
        cols = np.flatnonzero(d.any(0))
        for c in np.unique(cols // block):
            out.append((r0, min(r0 + block, H), int(c) * block, min((int(c) + 1) * block, W)))
    return out


def lf_box(blk, shape, per, DX, DZ, pad=2):
    """World box of an LF-frame raster block (render_campaign_view.Ground._bilinear: x = col / per * DX, south-up
    v = H - 1 - row, z = v / per * DZ), padded by `pad` pixels for the bilinear reads."""
    r0, r1, c0, c1 = blk
    H = shape[0]
    return ((c0 - pad) / per * DX, (H - 1 - (r1 - 1) - pad) / per * DZ, (c1 + pad) / per * DX, (H - 1 - r0 + pad) / per * DZ)


def blend_box(blk, shape, world, pad=2):
    """World box of a global_blend block (Ground.texel: col = x / Ww * W, row-from-top = (1 - z / Wh) * H)."""
    r0, r1, c0, c1 = blk
    H, W = shape[:2]
    Ww, Wh = world
    return ((c0 - pad) / W * Ww, (1 - (r1 + pad) / H) * Wh, (c1 + pad) / W * Ww, (1 - (r0 - pad) / H) * Wh)


# ------------------------------------------------------------------ row inputs
_P = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93,
               0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x2545F4914F6CDD1D, 0x9FB21C651E98DF25], np.uint64)


def row_keys(M, extra=None):
    """64-bit key per row of a numeric matrix (exact bytes), plus optional per-row uint64 extras."""
    u = np.ascontiguousarray(M).view(np.uint8).reshape(len(M), -1)
    u = np.pad(u, ((0, 0), (0, (-u.shape[1]) % 8))).view(np.uint64)
    k = np.zeros(len(M), np.uint64)
    with np.errstate(over='ignore'):
        for j in range(u.shape[1]):
            k = k * np.uint64(0x100000001B3) + (u[:, j] ^ (u[:, j] >> np.uint64(29))) * _P[j % len(_P)]
        if extra is not None:
            k = k * np.uint64(0x100000001B3) + extra.astype(np.uint64) * _P[0]
    return k


def changed_rows(ka, kb):
    """Rows of a whose key is not among b's, and of b not among a's (multiset by key: duplicates count)."""
    def multi(k):
        o = np.argsort(k, kind='stable')
        s = k[o]
        rank = np.zeros(len(k), np.int64)
        if len(k):
            first = np.r_[True, s[1:] != s[:-1]]
            idx = np.arange(len(k))
            start = np.maximum.accumulate(np.where(first, idx, 0))
            rank[o] = idx - start
        return k, rank
    (a, ra), (b, rb) = multi(ka), multi(kb)
    ca = np.empty(len(a), [('k', np.uint64), ('r', np.int64)]); ca['k'], ca['r'] = a, ra
    cb = np.empty(len(b), [('k', np.uint64), ('r', np.int64)]); cb['k'], cb['r'] = b, rb
    return ~np.isin(ca, cb), ~np.isin(cb, ca)


def object_matrix(o):
    """Numeric columns of render_live_objects rows (everything Overlay draws) and the model name crc."""
    M = np.column_stack([o.x, o.y, o.z, o.A.reshape(len(o.x), 4), o.lo, o.hi, o.ys, o.shape.astype(np.float64), o.col,
                         o.listed.astype(np.float64), o.canopy.astype(np.float64), o.seed]).astype(np.float64)
    crc = np.array([zlib.crc32(str(m).encode()) for m in o.model], np.uint64)
    return M, crc


def triangle_keys(t):
    return row_keys(np.ascontiguousarray(t, np.float32).reshape(len(t), 9))


# ------------------------------------------------------------------ the change set
class Reads:
    """Pack keys read through PackSource.read / Ground._map while active (class-level patch, one process)."""
    def __init__(self):
        self.keys = set()

    def __enter__(self):
        import render_live_view as L, render_campaign_view as rv
        self._orig = L.PackSource.read, rv.Ground._map
        keys = self.keys
        read, gmap = self._orig

        def tracked_read(s, key):
            keys.add(key)
            return read(s, key)

        def tracked_map(s, key, dtype, channels):
            keys.add(key)
            return gmap(s, key, dtype, channels)
        L.PackSource.read, rv.Ground._map = tracked_read, tracked_map
        return self

    def __exit__(self, *exc):
        import render_live_view as L, render_campaign_view as rv
        L.PackSource.read, rv.Ground._map = self._orig


def structures(L, pack, P):
    """The renderer's parsed inputs of one pack (fresh state, the renderer's own constructors) and the keys read."""
    from render_live_objects import Objects
    L._STATE.clear()
    with Reads() as r:
        live = L.live_state(pack, P)
        g, sh = live['ground'], live['shader']
        holes = L.tile_holes(g.packs)
        objs = Objects(g, towns=P['towns'])
        s = dict(lf=g.lf, sea=g.seamap, blend=g.blend, water=sh.meshes.water, land=sh.meshes.land, holes=holes, objects=objs,
                 index=g.packs.index, mod_keys=g.packs.mod_keys)
    L._STATE.clear()
    # water/land triangles come from these keys even when the triangle cache spared reading them; the triangles
    # are compared below, so the keys are covered
    return s, set(r.keys) | set(sh.meshes.source_keys())


def entry_digest(index, key):
    src, off, size, comp = index[key]
    with open(src, 'rb') as f:
        f.seek(off)
        return hashlib.sha1(f.read(size)).hexdigest()


def compare(L, base, pack, P, render_reads=None):
    """Change grids of `pack` against `base`. render_reads: keys the renderer read in earlier runs of this renderer
    (None: unknown, so every camera counts as changed). Returns a dict with Grid 'height', 'surface', list 'global'
    and counts."""
    import campaign_world as cw
    from campaign_fine_tiles import CELL_X, CELL_Z
    sa, ra = structures(L, base, P)
    sb, rb = structures(L, pack, P)
    world = (cw.WORLD_WIDTH, cw.WORLD_HEIGHT)
    height, surface = Grid(*world), Grid(*world)
    glob, counts = [], {}
    for name, per in (('lf', cw.LF_PX_PER_HEX), ('sea', cw.SEA_PX_PER_HEX)):
        blks = array_blocks(sa[name], sb[name])
        if blks is None:
            glob.append(f'{name} raster size changed')
            continue
        counts[name + '_blocks'] = len(blks)
        for blk in blks:
            height.mark(*lf_box(blk, sa[name].shape, per, cw.DX, cw.DZ))
    blks = array_blocks(sa['blend'], sb['blend'])
    if blks is None:
        glob.append('global_blend size changed')
    else:
        counts['blend_blocks'] = len(blks)
        for blk in blks:
            surface.mark(*blend_box(blk, sa['blend'].shape, world))
    for name in ('water', 'land'):
        ta, tb = sa[name], sb[name]
        da, db = changed_rows(triangle_keys(ta), triangle_keys(tb))
        counts[name + '_triangles'] = int(da.sum() + db.sum())
        for t in (ta[da], tb[db]):
            if len(t):
                height.mark(t[:, :, 0].min(1), t[:, :, 2].min(1), t[:, :, 0].max(1), t[:, :, 2].max(1))
    if sa['holes'].shape != sb['holes'].shape:
        glob.append('tile grid size changed')
    else:
        jj, ii = np.nonzero(sa['holes'] != sb['holes'])
        counts['hole_cells'] = int(len(jj))
        if len(jj):
            surface.mark(ii * CELL_X, jj * CELL_Z, (ii + 1) * CELL_X, (jj + 1) * CELL_Z)
    oa, ob = sa['objects'], sb['objects']
    (Ma, ca), (Mb, cb) = object_matrix(oa), object_matrix(ob)
    da, db = changed_rows(row_keys(Ma, ca), row_keys(Mb, cb))
    counts['object_rows'] = int(da.sum() + db.sum())
    for o, d in ((oa, da), (ob, db)):
        if d.any():
            surface.mark(o.x[d] - o.r[d], o.z[d] - o.r[d], o.x[d] + o.r[d], o.z[d] + o.r[d])
    # keys the renderer reads beyond the compared structures
    covered = ra | rb
    if render_reads is None:
        glob.append('no read record for this renderer yet: every camera renders')
    else:
        for k in sorted(set(render_reads) - covered):
            ia, ib = sa['index'].get(k), sb['index'].get(k)
            if (ia is None) != (ib is None):
                glob.append(f'{k} added/removed')
            elif ia is not None and (ia[2] != ib[2] or entry_digest(sa['index'], k) != entry_digest(sb['index'], k)):
                glob.append(f'{k} changed')
    return dict(height=height, surface=surface, glob=glob, counts=counts, covered=sorted(covered))


# ------------------------------------------------------------------ cameras
def camera_region(cam, footprint, pad=15.):
    """(height window, surface box) of a camera: the renderer's x +- 6 dist mesh window (render_live_view.gbuffer and
    LiveShader.frame pads) united with the ground hits; the surface box is the bounding box of the hits (the 80 x 45
    footprint of the base render), of target +- dist (the eye) and, for hits farther than 3 dist (grazing rays,
    where full-resolution rays between the footprint's reach much farther), of those hits pushed out to 1.5 x their
    distance from the target. Both padded by `pad` world units (sun shadow rays 12, object overlay 3, AO radii)."""
    x, z, dist = cam[:3]
    reach = dist * 6
    pts = np.asarray(footprint, float).reshape(-1, 2)
    c = np.array([x, z])
    far = pts[np.hypot(*(pts - c).T) > 3 * dist] if len(pts) else pts
    pts = np.vstack([pts, c - dist, c + dist, c + (far - c) * 1.5])
    sx0, sz0 = pts.min(0) - pad
    sx1, sz1 = pts.max(0) + pad
    span = 2 * reach
    fpad = .02 * span + 1.0 + pad          # LiveShader.frame pad (.02 span + DX) and the raster cell
    hx0, hz0 = min(x - reach - fpad, sx0), min(z - reach - fpad, sz0)
    hx1, hz1 = max(x + reach + fpad, sx1), max(z + reach + fpad, sz1)
    return (hx0, hz0, hx1, hz1), (sx0, sz0, sx1, sz1)


def affected(ch, cam, footprint):
    """Why a camera must render ([] = it sees no change)."""
    if ch['glob']:
        return ['global: ' + ch['glob'][0]]
    hwin, sbox = camera_region(cam, footprint)
    why = []
    if ch['height'].any(*hwin):
        why.append('height change in the view window')
    if ch['surface'].any(*sbox):
        why.append('surface change in the view')
    return why

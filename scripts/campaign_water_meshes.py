"""Water the campaign actually draws, from its meshes (render_campaign_view --water meshes).

Live test 2048n: the game draws water only from meshes; lf_sea_height_map draws nothing.
Water triangles, in the source's own world frame (x, z world units, y = LF world height):
  * global_meshes/sea_mesh_*.rigid_model_v2 (type 101, 4 f32 per vertex, raw frame: z * ZS);
    sea_mesh_0 carries the ocean background quad at y -1.029 under the whole canvas
    (extend_campaign_ocean_background.py);
  * models/river_N.wsmodel and every models/*.wsmodel a global_props leaf instances (the frontier
    river/lake surfaces of river_water_props.py) -> geometry (type 68, 48-byte vertex, f32 xyz +
    model pivot, world frame; repair_campaign_river_roots: root translations are not applied);
  * lake props (*/props/lakes/*.wsmodel instances in global_props.bin leaves; 32-byte vertex,
    f16 xyz + pivot, row vector: v @ M3 + t; root translations not applied, as for rivers).
The ground they are tested against is the coarse land mesh (global_meshes/land_mesh_*, type
101, raw frame): its triangle fans cut by the flat ocean plane are the square/diamond pools
seen live at Irtysh/Ganges/Funan (the LF contour of the same plane is a smooth blob). Skirt
vertices (LF - 1) lose to the max z-buffer. Its holes are filled by fine tiles (river beds,
borders, coasts), so the caller compares there with the LF (render_campaign_view.mesh_level).

Triangles are cached per source signature (output/offline-view/cache/water-meshes-*.npz);
fields() rasterises a window: highest water and highest land triangle per cell.
"""
import hashlib, os, re, struct, sys
import xml.etree.ElementTree as ET

import numpy as np

import campaign_world as cw

VERSION = 3   # 3: models/ instanced by global_props leaves (river_water_props) drawn as rivers
LAKE_MODEL = re.compile(r'/props/lakes/', re.I)
# Sea meshes show 1.014 above their vertices: native open sea mesh y = lf_sea_height - 1.014 everywhere,
# and live 2048n pools match land mesh < -0.015 (ocean quad -1.029 + 1.014), e.g. the Funan river pool
# over a land mesh that bottoms out at -0.81. River models / lake props: no constant offset known, none.
SEA_LIFT = 1.014


def _mesh(d, stride_pos):
    """(vertices xyz f64, triangles int) of a rigid_model_v2's first mesh; stride_pos: 'f4' or 'f2'."""
    mp = struct.unpack_from('<I', d, 152)[0]
    typ, _, _, vo, vc, io, ic = struct.unpack_from('<HH5I', d, mp)
    stride = (io - vo) // vc
    dt = np.dtype('<' + stride_pos)
    v = np.frombuffer(d, dt, offset=mp + vo, count=vc * stride // dt.itemsize).reshape(vc, -1)[:, :3].astype(np.float64)
    return typ, v, np.frombuffer(d, '<u2', offset=mp + io, count=ic).reshape(-1, 3).astype(np.int64), mp


class WaterMeshes:
    def __init__(self, ground, variant, cache_dir):
        self.g = ground
        self.pre = f'terrain/campaigns/{variant}/'
        sig = self._signature()
        path = cache_dir / f'water-meshes-{"vanilla" if ground.vanilla else "stage"}-{sig}.npz'
        if path.is_file():
            with np.load(path) as z:   # closed again: other processes may replace or prune it
                self.water, self.land = z['water'], z['land']
        else:
            self.water, self.land = self._load()
            cache_dir.mkdir(parents=True, exist_ok=True)
            olds = []
            for p in cache_dir.glob(f'water-meshes-{"vanilla" if ground.vanilla else "stage"}-*.npz'):
                try:
                    if '.tmp.' not in p.name:
                        olds.append((p.stat().st_mtime, p))
                except OSError:    # pruned meanwhile by another process
                    pass
            olds = [p for _, p in sorted(olds)]
            for old in olds[:-2]:  # the two newest per source kind stay (art_review compares two packs)
                try:
                    old.unlink()
                except OSError:    # open in another process (art_review render workers): keep it
                    pass
            tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
            np.savez(tmp, water=self.water, land=self.land)
            try:
                os.replace(tmp, path)   # whole file or nothing: parallel loaders never read a half-written cache
            except OSError:
                tmp.unlink()
        # per-triangle xz bounds for window culling
        self.bounds = {k: (t[:, :, 0].min(1), t[:, :, 0].max(1), t[:, :, 2].min(1), t[:, :, 2].max(1))
                       for k, t in (('water', self.water), ('land', self.land))}
        self.win = None

    # --- sources
    def _keys(self, sub, suffix):
        if self.g.vanilla:
            return self.g.packs.keys(self.pre + sub, suffix)
        d = self.g.stage / self.pre / sub.rsplit('/', 1)[0]
        stem = sub.rsplit('/', 1)[1]
        return sorted(f'{self.pre}{sub.rsplit("/", 1)[0]}/{p.name}' for p in d.glob(stem + '*' + suffix))

    def source_keys(self):
        """Pack keys the triangles come from (the cache signature's keys; on a cache hit none of them is read)."""
        return (self._keys('global_meshes/', '.rigid_model_v2') + self._keys('models/', '.rigid_model_v2')
                + [self.pre + 'global_props.bin'])

    def _signature(self):
        h = hashlib.sha1(str(VERSION).encode())
        for k in self.source_keys():
            if self.g.vanilla or not (self.g.stage / k).is_file():
                h.update(repr((k, self.g.packs.index[k][1:])).encode())
            else:
                st = (self.g.stage / k).stat()
                h.update(repr((k, st.st_size, st.st_mtime_ns)).encode())
        return h.hexdigest()[:16]

    def _load(self):
        g, water, land = self.g, [], []
        for k in self._keys('global_meshes/', '.rigid_model_v2'):
            typ, v, t, _ = _mesh(g.read(k), 'f4')
            if typ != 101:
                continue
            v[:, 2] *= cw.ZS   # raw frame z -> world z
            if '/sea_mesh_' in k:
                v[:, 1] += SEA_LIFT
            (water if '/sea_mesh_' in k else land if '/land_mesh_' in k else []).append(v[t])
        geoms = {}

        def model(key):   # wsmodel -> (local vertices + pivot, triangles)
            if key not in geoms:
                geom = ET.fromstring(g.read(key).decode('utf-8', 'replace')).findtext('geometry')
                try:
                    d = g.read(geom)
                except KeyError:   # pack index keys are lower case
                    d = g.read(geom.lower())
                mp = struct.unpack_from('<I', d, 152)[0]
                _, _, _, vo, vc, io, _ = struct.unpack_from('<HH5I', d, mp)
                typ, v, t, mp = _mesh(d, 'f4' if (io - vo) // vc == 48 else 'f2')
                geoms[key] = (v + struct.unpack_from('<3f', d, mp + 628), t)
            return geoms[key]
        from props_bundle import embedded
        from translate_campaign_lake_instances import records
        lakes = 0
        # River-water models: vanilla's 24 models/river_N plus every model under this map's models/
        # a global_props leaf instances (river_water_props.py: models/tw3k_water_N, vanilla river
        # scenes, placed by pivot like river_N; their leaf matrices are identity).
        rivers = set(g.packs.keys(self.pre + 'models/river_', '.wsmodel'))
        props = embedded(g.read(self.pre + 'global_props.bin'))[0]
        for k, b in props.items():
            if not k.endswith('.bin') or k.endswith('/bmd_objects.bin') or self.pre.encode() + b'models/' not in b:
                continue
            for r in records(b):
                if r['model'].startswith(self.pre + 'models/'):
                    rivers.add(r['model'])
        for k in sorted(rivers):
            v, t = model(k)
            water.append(v[t])
        for k, b in props.items():
            if not k.endswith('.bin') or k.endswith('/bmd_objects.bin') or b'/lakes/' not in b:
                continue
            for r in records(b):
                if LAKE_MODEL.search(r['model']):
                    v, t = model(r['model'].lower())
                    M = np.array(r['matrix'], np.float64)
                    water.append((v @ M[:9].reshape(3, 3) + M[9:])[t]); lakes += 1
        print(f'water meshes: {sum(map(len, water))} water triangles ({lakes} lake props), '
              f'{sum(map(len, land))} land triangles', file=sys.stderr, flush=True)
        return np.concatenate(water).astype(np.float32), np.concatenate(land).astype(np.float32)

    # --- raster
    def fields(self, x0, z0, step, nx, nz):
        """(water, land) highest triangle heights on cell centres x0 + (i + .5) * step (rows south-up),
        -inf where no triangle; cached for the last window."""
        key = (x0, z0, step, nx, nz)
        if self.win and self.win[0] == key:
            return self.win[1]
        x1, z1 = x0 + nx * step, z0 + nz * step
        out = []
        for name, tris in (('water', self.water), ('land', self.land)):
            a, b, c, d = self.bounds[name]
            sel = (b >= x0) & (a <= x1) & (d >= z0) & (c <= z1)
            out.append(raster(tris[sel], x0, z0, step, nx, nz))
        self.win = (key, tuple(out))
        return self.win[1]


# render_gpu.ZFill while a caller holding pipeline_lock.gpu wants raster's fill on the GPU (render_live_view.gbuffer)
GPU_FILL = None


def raster(tris, x0, z0, step, nx, nz, chunk=4_000_000):
    """Max z-buffer of triangles (N, 3, xyz) on an (nz, nx) grid of cell centres; edges inclusive.
    Row bands on a thread pool (numpy releases the GIL); each band owns its rows. With GPU_FILL set the same band
    items are filled on the GPU (tests/test_render_gpu.py: equal grids)."""
    grid = np.full((nz, nx), -np.inf, np.float32)
    if not len(tris):
        return grid
    T = tris.astype(np.float64)
    px = (T[:, :, 0] - x0) / step - .5; pz = (T[:, :, 2] - z0) / step - .5; py = T[:, :, 1]
    ix0 = np.clip(np.ceil(px.min(1) - 1e-6), 0, nx).astype(np.int64); ix1 = np.clip(np.floor(px.max(1) + 1e-6), -1, nx - 1).astype(np.int64)
    iz0 = np.clip(np.ceil(pz.min(1) - 1e-6), 0, nz).astype(np.int64); iz1 = np.clip(np.floor(pz.max(1) + 1e-6), -1, nz - 1).astype(np.int64)
    # affine set-up per triangle: barycentric l1, l2 and height y = K . (x, z, 1)
    ax, az, bx, bz, cx, cz = px[:, 0], pz[:, 0], px[:, 1], pz[:, 1], px[:, 2], pz[:, 2]
    det = (bx - ax) * (cz - az) - (cx - ax) * (bz - az)
    ok = (ix1 >= ix0) & (iz1 >= iz0) & (np.abs(det) > 1e-12)
    det = np.where(ok, det, 1)
    a1, b1 = (cz - az) / det, -(cx - ax) / det; c1 = -a1 * ax - b1 * az
    a2, b2 = -(bz - az) / det, (bx - ax) / det; c2 = -a2 * ax - b2 * az
    d1, d2 = py[:, 1] - py[:, 0], py[:, 2] - py[:, 0]
    K = np.stack([a1, b1, c1, a2, b2, c2, a1 * d1 + a2 * d2, b1 * d1 + b2 * d2, py[:, 0] + c1 * d1 + c2 * d2], 1)
    keep = np.flatnonzero(ok)
    K, ix0, ix1, iz0, iz1 = K[keep], ix0[keep], ix1[keep], iz0[keep], iz1[keep]

    def items(r0, r1):
        sel = np.flatnonzero((iz1 >= r0) & (iz0 < r1))
        x0t, z0b, z1b = ix0[sel], np.maximum(iz0[sel], r0), np.minimum(iz1[sel], r1 - 1)
        w = ix1[sel] - x0t + 1; cnt = w * (z1b - z0b + 1)
        k = K[sel]   # rebased to each triangle's first candidate cell: small f32 offsets
        k[:, 2::3] += k[:, 0::3] * x0t[:, None] + k[:, 1::3] * z0b[:, None]
        return x0t, z0b, w, cnt, k.astype(np.float32)

    def band(r0, r1):
        x0t, z0b, w, cnt, k = items(r0, r1)
        cs = np.cumsum(cnt); start = 0
        while start < len(cnt):   # chunks of ~chunk candidate cells
            stop = max(int(np.searchsorted(cs, (cs[start - 1] if start else 0) + chunk, 'right')), start + 1)
            n = cnt[start:stop]
            loc = np.repeat(np.arange(start, stop), n)
            off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            dx, dz = off % w[loc], off // w[loc]
            _fill(grid, x0t[loc] + dx, z0b[loc] + dz, dx.astype(np.float32), dz.astype(np.float32), k[loc])
            start = stop
    import concurrent.futures as cf, os
    rows = np.linspace(0, nz, min(nz, 4 * (os.cpu_count() or 4)) + 1).astype(int)
    if GPU_FILL is not None:   # the same items (the rebased constants depend on the band rows), one launch
        parts = [items(a, b) for a, b in zip(rows[:-1], rows[1:])]
        GPU_FILL(grid, *(np.concatenate([p[i] for p in parts]) for i in range(5)), eps=_fill.__defaults__[0])
        return grid
    with cf.ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        list(ex.map(lambda r: band(*r), zip(rows[:-1], rows[1:])))
    return grid


def _fill(grid, xx, zz, dx, dz, k, eps=1e-5):
    l1 = k[:, 0] * dx + k[:, 1] * dz + k[:, 2]
    l2 = k[:, 3] * dx + k[:, 4] * dz + k[:, 5]
    inside = (l1 >= -eps) & (l2 >= -eps) & (l1 + l2 <= 1 + eps)
    k, dx, dz = k[inside], dx[inside], dz[inside]
    y = k[:, 6] * dx + k[:, 7] * dz + k[:, 8]
    flat = grid.reshape(-1); cell = zz[inside] * grid.shape[1] + xx[inside]
    while len(cell):   # max z-buffer: write, keep only the losers' rivals (shared edges, skirts: few passes)
        up = y > flat[cell]
        cell, y = cell[up], y[up]
        flat[cell] = y

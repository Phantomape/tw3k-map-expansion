"""Fine campaign terrain tiles for the offline view (render_campaign_view.py --tiles).

What the files say (vanilla packs, read-only):
  * global_map/tile_list.bin records '<HIBhhHff' = (1, path index, climate, x, y, flags, lo, hi);
    x/y is the footprint's minimum tile column / row (rows south-up, as in tile_map.tiles), flags
    bits 4-7 are a one-hot rotation: 0x10 = 0, 0x20 = 90 cw, 0x40 = 180, 0x80 = 270 (width and
    height swap for 0x20/0x80). Checked against every road owner of the 3k_main_map display cache:
    the parent cell (local cell 0,0) sits NW / NE / SE / SW of the footprint for 0x10/0x20/0x40/0x80.
  * terrain/tiles/campaign/<set>/<piece>/mesh.rigid_model_v2 spans 128 local units per tile cell
    (group bounding box), so footprint = bbox / 128.
  * blend0.dds (BC3, roads_*, river*, canal*, sea_coast) is a 4-weight splat map, R+G+B+A = 255:
    one channel keeps the ground below (global blend): R on most pieces, G or B on some mirrored
    pieces and junctions (taken as the channel dominating the margin ring); the others paint extra
    texture-array layers (roads: the running surface, plus verges on paved/imperial roads). It
    covers the footprint plus one cell on every side (128 px per cell, sea_coast 32):
    (w + 2) x (h + 2) cells; road centre lines meet the cell centres only with that margin.
    Texture rows run south-up in the tile's local frame (the shader's t = local z). With this, the
    vanilla dlc07 road pieces join end to end and fall on the 3k_main_map display-cache road cells.
  * The layer index of G/B/A is a per-draw shader constant (g_layerN_texture_index, filled by the
    engine) and is not in any data file; LAYERS below is a guess checked by eye against live
    captures (pale imperial_road tracks at Tai).
  * salt_pools (and other custom-mesh sets) have no blend0: custom_mesh.rigid_model_v2 with its own
    material 'terrain\\textures\\campaign\\default\\salt_pools\\salt_pools'. Their top-down footprint
    is drawn from that mesh (triangles textured with salt_pools_base_colour.dds). The mesh is a
    basin (rim at 0, pond floors down to -90 units); the ponds are water coloured by
    3k_main_map/sea_colour.dds, which vanilla paints pale salt (205, 206, 172) under every pan, so
    pond floors are drawn in the sea_colour.dds colour at that point.
"""
import struct

import numpy as np

import campaign_world as cw

CELL_X = cw.VANILLA_TILE_CELL
CELL_Z = cw.VANILLA_TILE_CELL * cw.ENGINE_Z_RATIO
REC = np.dtype([('a', '<u2'), ('p', '<u4'), ('c', 'u1'), ('x', '<i2'), ('y', '<i2'), ('f', '<u2'), ('lo', '<f4'), ('hi', '<f4')])
POND_DEPTH = -8.        # custom-mesh units below the rim that count as pond (salt pans: floor to -90)
ROT = {0x10: 0, 0x20: 1, 0x40: 2, 0x80: 3}
# texture-array layer for the G, B, A weights per tile set prefix (first match wins); guessed.
LAYERS = (('roads_', (13, 13, 14)), ('river_crossing', (13, 14, 14)), ('river', (14, 14, 14)),
          ('canal', (14, 14, 14)), ('sea_coast', (0, 0, 0)))


def bc3_rgba(data, mip=0):
    """RGBA uint8 array of one mip of a BC3 (DXT5) DDS."""
    from campaign_ground_textures import _bc_colour
    h, w = struct.unpack_from('<2I', data, 12)
    assert data[84:88] == b'DXT5', data[84:88]
    off = 128
    for _ in range(mip):
        off += max(1, w // 4) * max(1, h // 4) * 16
        w, h = max(1, w // 2), max(1, h // 2)
    bw, bh = max(1, w // 4), max(1, h // 4)
    blk = np.frombuffer(data, np.uint8, bw * bh * 16, off).reshape(-1, 16)
    a0, a1 = blk[:, 0].astype(np.int32)[:, None], blk[:, 1].astype(np.int32)[:, None]
    bits = np.zeros(len(blk), np.uint64)
    for i in range(6):
        bits |= blk[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    idx = ((bits[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64))[None]) & np.uint64(7)).astype(np.int64)
    k = np.arange(8)[None]
    pal8 = np.where(k == 0, a0, np.where(k == 1, a1, ((8 - k) * a0 + (k - 1) * a1) // 7))
    pal6 = np.where(k == 0, a0, np.where(k == 1, a1, np.where(k < 6, ((6 - k) * a0 + (k - 1) * a1) // 5, np.where(k == 6, 0, 255))))
    alpha = np.take_along_axis(np.where(a0 > a1, pal8, pal6), idx, 1).astype(np.uint8)
    px = np.concatenate([_bc_colour(blk[:, 8:]), alpha[..., None]], -1)
    return px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)[:h, :w]


def mesh_group(blob):
    """(bbox min xyz, max xyz, first group offset) of a rigid_model_v2 (LOD 0, first group)."""
    g = 140 + 28 * struct.unpack_from('<I', blob, 8)[0]
    b = struct.unpack_from('<6f', blob, g + 24)
    return b[:3], b[3:], g


def custom_mesh(blob):
    """Triangles of a campaign custom tile mesh (LOD 0): xyz (n, 3) float, uv (n, 2), tris (m, 3)."""
    lo, hi, g = mesh_group(blob)
    head, nvert, ioff, nidx = struct.unpack_from('<I', blob, g + 8)[0], *struct.unpack_from('<3I', blob, g + 12)
    stride = (ioff - head) // nvert
    v = np.frombuffer(blob, np.uint8, nvert * stride, g + head).reshape(nvert, stride)
    xyz = v[:, :12].copy().view('<f4').reshape(nvert, 3)
    uv = v[:, 28:32].copy().view('<f2').reshape(nvert, 2).astype(np.float32)
    tris = np.frombuffer(blob, '<u2', nidx, g + ioff).reshape(-1, 3).astype(np.int64)
    return xyz, uv, tris, lo, hi


class FineTiles:
    """Tile records of one source and the blend0 compositing for Shader.albedo."""
    def __init__(self, ground, sets=None, variant='3k_dlc07_main_map'):
        self.g = ground
        pre = f'terrain/campaigns/{variant}/'
        blob = None
        for key in (pre + 'tile_list.bin', pre + 'global_map/tile_list.bin'):
            try:
                blob = ground.read(key)
                break
            except (KeyError, FileNotFoundError):
                continue
        paths, start = self._paths(blob)
        recs = np.frombuffer(blob, REC, offset=start, count=(len(blob) - start) // REC.itemsize)
        self.paths = [p.replace('\\', '/').lower() for p in paths]
        self.kind = {}          # path index -> 'blend' | 'custom'
        self.layers = {}
        packs = ground.packs
        for i, p in enumerate(self.paths):
            tset = p.split('/')[3] if p.count('/') >= 4 else ''
            if sets and not any(tset.startswith(s) for s in sets):
                continue
            if p + 'blend0.dds' in packs.index:
                self.kind[i] = 'blend'
                self.layers[i] = next((l for s, l in LAYERS if tset.startswith(s)), (14, 14, 14))
            elif p + 'custom_mesh.rigid_model_v2' in packs.index:
                self.kind[i] = 'custom'
        keep = np.isin(recs['p'], np.fromiter(self.kind, np.int64)) if self.kind else np.zeros(len(recs), bool)
        self.recs = recs[keep].copy()
        self.dims = {}
        for i in np.unique(self.recs['p']):
            name = 'mesh.rigid_model_v2' if self.kind[int(i)] == 'blend' else 'custom_mesh.rigid_model_v2'
            lo, hi, _ = mesh_group(packs.read(self.paths[i] + name))
            self.dims[int(i)] = (round((hi[0] - lo[0]) / 128), round((hi[2] - lo[2]) / 128))
        self.cache = {}
        self.stats = {}

    @staticmethod
    def _paths(b):
        count = struct.unpack_from('<I', b, 12)[0]
        pos, out = 16, []
        for _ in range(count):
            n = struct.unpack_from('<H', b, pos)[0]; pos += 2
            out.append(b[pos:pos + n].decode()); pos += n
        n, pos = struct.unpack_from('<I', b, pos)[0], pos + 4
        for _ in range(n):                                  # climate names
            pos += 2 + struct.unpack_from('<H', b, pos)[0]
        return out, pos + 24 + 44 + 1 + 4                   # info block, record count

    def blend0(self, i, mip):
        key = (i, mip)
        if key not in self.cache:
            data = self.g.packs.read(self.paths[i] + 'blend0.dds')
            n = struct.unpack_from('<I', data, 28)[0] or 1
            self.cache[key] = bc3_rgba(data, min(mip, n - 1)).astype(np.float32) / 255
        return self.cache[key]

    def sea_colour(self, lx, lz):
        """sRGB float of terrain/campaigns/3k_main_map/sea_colour.dds (the water colour map dlc07 reads,
        UV = world / world bounds) at local world points. Vanilla paints pale salt (205,206,172) under
        every salt pan; elsewhere it is ocean blue."""
        if 'sea' not in self.cache:
            key = 'terrain/campaigns/3k_main_map/sea_colour.dds'
            p = self.g.stage / key
            if not self.g.vanilla and p.is_file():
                path, base = p, 0
            else:
                path, base, _, _ = self.g.packs.index[key]
            with open(path, 'rb') as f:
                f.seek(base); head = f.read(152)
            h, w = struct.unpack_from('<2I', head, 12)
            dx10 = head[84:88] == b'DX10'
            assert (struct.unpack_from('<I', head, 128)[0] in (70, 71, 72)) if dx10 else head[84:88] == b'DXT1'
            blocks = np.memmap(path, np.uint8, 'r', base + (148 if dx10 else 128), (h // 4) * (w // 4) * 8).reshape(h // 4, w // 4, 8)
            self.cache['sea'] = (blocks, h, w)
        from campaign_ground_textures import _bc_colour
        blocks, h, w = self.cache['sea']
        col = np.clip((lx / self.g.world[0] * w).astype(int), 0, w - 1)
        row = np.clip(((1 - lz / self.g.world[1]) * h).astype(int), 0, h - 1)
        px = _bc_colour(np.asarray(blocks[row // 4, col // 4]))
        return px[np.arange(len(px)), (row % 4) * 4 + col % 4].astype(np.float32) / 255

    def channels(self, i):
        """(channel that keeps the ground below, [(channel, texture-array layer)]) of a blend0 piece.
        The ground channel is R on most pieces but G or B on some mirrored pieces and junctions, so it
        is taken as the channel that dominates the one-cell margin ring; the remaining channels get the
        set's layers in order of coverage (sea_coast: always R)."""
        key = ('ch', i)
        if key not in self.cache:
            t = self.blend0(i, 3)
            h, w = t.shape[:2]; mh, mw = max(1, h // 8), max(1, w // 8)
            ring = np.concatenate([t[:mh].reshape(-1, 4), t[-mh:].reshape(-1, 4), t[:, :mw].reshape(-1, 4), t[:, -mw:].reshape(-1, 4)])
            base = 0 if self.paths[i].split('/')[3] == 'sea_coast' else int(ring.mean(0).argmax())
            rest = [c for c in np.argsort(-(t.reshape(-1, 4) > .5).mean(0), kind='stable') if c != base and t[..., c].max() > 0]
            self.cache[key] = (base, [(int(c), l) for c, l in zip(rest, self.layers[i])])
        return self.cache[key]

    def custom_raster(self, i, res=16):
        """Top-down raster (local frame, rows from the north edge at rotation 0) of a custom mesh:
        (rgb float, coverage bool, top height in mesh units) at `res` px per tile cell."""
        key = ('custom', i, res)
        if key in self.cache:
            return self.cache[key]
        from campaign_ground_textures import decode_dds
        blob = self.g.packs.read(self.paths[i] + 'custom_mesh.rigid_model_v2')
        xyz, uv, tris, lo, hi = custom_mesh(blob)
        w, h = self.dims[i]
        W, H = w * res, h * res
        try:
            stem = blob[blob.index(b'terrain\\textures'):].split(b'\0', 1)[0].decode().replace('\\', '/').lower()
            tex = decode_dds(self.g.packs.read(stem + '_base_colour.dds'), 2).astype(np.float32) / 255
        except (ValueError, KeyError):
            tex = None
        rgb = np.zeros((H, W, 3), np.float32); cov = np.zeros((H, W), bool); top = np.full((H, W), -np.inf, np.float32)
        px = (xyz[:, 0] - lo[0]) / 128 * res              # column from the west edge
        py = (hi[2] - xyz[:, 2]) / 128 * res              # row from the local north edge (mesh z: -1024..0)
        for t in tris:
            ax, ay = px[t], py[t]
            x0, x1 = int(max(0, np.floor(ax.min()))), int(min(W - 1, np.ceil(ax.max())))
            y0, y1 = int(max(0, np.floor(ay.min()))), int(min(H - 1, np.ceil(ay.max())))
            if x1 < x0 or y1 < y0:
                continue
            gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1] + .5
            d = (ay[1] - ay[2]) * (ax[0] - ax[2]) + (ax[2] - ax[1]) * (ay[0] - ay[2])
            if abs(d) < 1e-9:
                continue
            l0 = ((ay[1] - ay[2]) * (gx - ax[2]) + (ax[2] - ax[1]) * (gy - ay[2])) / d
            l1 = ((ay[2] - ay[0]) * (gx - ax[2]) + (ax[0] - ax[2]) * (gy - ay[2])) / d
            l2 = 1 - l0 - l1
            m = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not m.any():
                continue
            y = l0 * xyz[t[0], 1] + l1 * xyz[t[1], 1] + l2 * xyz[t[2], 1]
            sub_top = top[y0:y1 + 1, x0:x1 + 1]
            m &= y > sub_top
            if not m.any():
                continue
            sub_top[m] = y[m]
            cov[y0:y1 + 1, x0:x1 + 1] |= m
            if tex is not None:
                u = l0 * uv[t[0], 0] + l1 * uv[t[1], 0] + l2 * uv[t[2], 0]
                v = l0 * uv[t[0], 1] + l1 * uv[t[1], 1] + l2 * uv[t[2], 1]
                S = tex.shape
                rgb[y0:y1 + 1, x0:x1 + 1][m] = tex[(np.mod(v[m], 1) * S[0]).astype(int) % S[0], (np.mod(u[m], 1) * S[1]).astype(int) % S[1]]
            else:
                rgb[y0:y1 + 1, x0:x1 + 1][m] = 0
        self.cache[key] = (rgb, cov, top)
        return self.cache[key]

    def composite(self, rgb, wx, wz, footprint, layer_rgb):
        """Paint the tiles over `rgb` (float sRGB, shape (..., 3)) at world points wx, wz.
        layer_rgb(layer, mask) -> (k, 3) colours of texture-array `layer` at the masked points."""
        lx, lz = self.g.local(np.asarray(wx, np.float64), np.asarray(wz, np.float64))
        col = np.floor(lx / CELL_X).astype(np.int64); row = np.floor(lz / CELL_Z).astype(np.int64)
        c0, c1, r0, r1 = col.min(), col.max(), row.min(), row.max()
        r = self.recs
        dims = np.array([self.dims[int(p)] for p in r['p']], np.int64).reshape(-1, 2)
        rot = np.vectorize(lambda f: ROT.get(f, 0), otypes=[np.int64])((r['f'] >> 0) & 0xf0) if len(r) else np.zeros(0, np.int64)
        fw = np.where(rot % 2 == 1, dims[:, 1], dims[:, 0]); fh = np.where(rot % 2 == 1, dims[:, 0], dims[:, 1])
        sel = np.flatnonzero((r['x'] <= c1) & (r['x'] + fw > c0) & (r['y'] <= r1) & (r['y'] + fh > r0))
        flat = rgb.reshape(-1, 3)
        lxf, lzf = lx.ravel(), lz.ravel()
        cid = (row.ravel() - r0) * (c1 - c0 + 1) + (col.ravel() - c0)
        order = np.argsort(cid, kind='stable'); scid = cid[order]
        mip = int(np.clip(np.floor(np.log2(max(footprint, 1e-6) / CELL_X * 128)), 0, 6))
        drawn = 0
        for k in sel:
            i, x0, y0, rt, w, h = int(r['p'][k]), int(r['x'][k]), int(r['y'][k]), int(rot[k]), int(fw[k]), int(fh[k])
            cells = [(rr - r0) * (c1 - c0 + 1) + (cc - c0) for rr in range(max(y0, r0), min(y0 + h, r1 + 1))
                     for cc in range(max(x0, c0), min(x0 + w, c1 + 1))]
            a = np.searchsorted(scid, cells); b = np.searchsorted(scid, cells, 'right')
            idx = np.concatenate([order[s:e] for s, e in zip(a, b)]) if len(cells) else np.zeros(0, np.int64)
            if not len(idx):
                continue
            fx = lxf[idx] / CELL_X - x0                    # cells from the footprint's west edge
            fn = (y0 + h) - lzf[idx] / CELL_Z              # cells from its north edge
            u, v = ((fx, fn), (fn, w - fx), (w - fx, h - fn), (h - fn, fx))[rt]
            tw, th = (w, h) if rt % 2 == 0 else (h, w)     # unrotated tile size
            if self.kind[i] == 'blend':
                t = self.blend0(i, mip)
                S = t.shape
                tu = np.clip(((u + 1) / (tw + 2) * S[1]).astype(int), 0, S[1] - 1)
                tv = np.clip(((th + 1 - v) / (th + 2) * S[0]).astype(int), 0, S[0] - 1)   # texture rows run south-up
                wts = t[tv, tu]
                base, layers = self.channels(i)
                out = flat[idx] * wts[:, base:base + 1]
                for ch, layer in layers:
                    m = wts[:, ch] > 0
                    if m.any():
                        out[m] += wts[m, ch:ch + 1] * layer_rgb(layer, idx[m])
                flat[idx] = out
            else:
                crgb, cov, top = self.custom_raster(i)
                res = crgb.shape[1] // tw
                tu = np.clip((u * res).astype(int), 0, crgb.shape[1] - 1)
                tv = np.clip((v * res).astype(int), 0, crgb.shape[0] - 1)
                m = cov[tv, tu]
                col = crgb[tv[m], tu[m]]
                pond = top[tv[m], tu[m]] < POND_DEPTH            # basin floor: the engine fills it with water
                if pond.any():
                    col[pond] = self.sea_colour(lxf[idx[m]][pond], lzf[idx[m]][pond])
                flat[idx[m]] = col
            drawn += 1
            self.stats[self.paths[i].split('/')[3]] = self.stats.get(self.paths[i].split('/')[3], 0) + 1
        return rgb, drawn

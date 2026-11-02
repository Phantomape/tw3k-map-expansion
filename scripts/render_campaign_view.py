"""Offline approximate render of the 190 campaign ground (3k_dlc07_main_map).

Replays the campaign far-terrain pixel shader (rigidterrain.hlsl ps_global_blend,
disassembled from shaders.pack) on the CPU:
  * global_map/global_blend.dds: R = material slot (0..31), G = climate band
    (0 cold, 1 arid, 2 temperate, 3 sub_tropical); the four texels around a point
    are fetched (no filtering of indices), each picks texture-array layer
    g_global_blend_indices[R + round(G*k) + season_offset] (k and the offset are
    engine cbuffer constants, not in any data file); this tool assumes the table is the
    identity on R (texture_arrays.xml has empty <climate/> entries and the 32
    groups line up with the observed R values), i.e. layer = R;
  * each corner's layer is sampled at world.xz / tile_size and the four colours
    are bilinearly blended with the texel fractions, exactly as the shader does;
  * hill shading from lf_height_map.dds, water as the game draws it (--water meshes, live
    2048n: only meshes draw water): the highest sea mesh / ocean background quad / river
    model / lake prop triangle, where it is above the coarse land mesh (the LF in its holes;
    campaign_water_meshes.py, sea meshes +1.014;
    --water seamap: where lf_sea_height_map is above the LF, which the game does NOT draw;
    --water threshold: a flat --sea-level), campaign trees as dots, region towns from
    current-region-design.json.
Textures default to the 190 start season ('harvest': groups without a harvest
texture use summer). --tiles also composites the fine campaign tiles that carry a
blend0 splat map (roads, river beds, canals, coasts) and custom-mesh tiles (salt pans)
from tile_list.bin (campaign_fine_tiles.py). Mountain alpha layers, prefabs/props,
river/lake models, shroud and the real lighting/fog are not rendered
(see analysis/offline-view-research.md).

Read-only: stage files and vanilla packs are memory-mapped, nothing is written
outside output/offline-view (and its cache).

  python render_campaign_view.py --world 376.941 659.814 [--radius 40] [--out x.png]
  python render_campaign_view.py --hex 565 856 --oblique 40,0,30
  python render_campaign_view.py --cameras analysis/live-check-canvas-2048h.md
  python render_campaign_view.py --world 788.87 565.10 --vanilla   (vanilla packs, native map)
  python render_campaign_view.py --hex 575 773 --radius 6 --tiles   (with fine tiles: roads, salt pans)
"""
import argparse, json, math, re, struct, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import campaign_world as cw
from campaign_ground_textures import VanillaPacks, load_layers as load_texture_layers, ROOT

VARIANT = '3k_dlc07_main_map'
OUT = ROOT / 'output/offline-view'
WATER_SHALLOW = np.array([74, 112, 118], np.float32)
WATER_DEEP = np.array([22, 48, 70], np.float32)
TREE_COLOURS = [  # (prefix, rgb, canopy radius in world units; guessed from live captures)
    ('cold_tree_fir', (34, 52, 30), .11), ('temperate_tree_metasequoia', (48, 72, 34), .11),
    ('temperate_tree_katsura', (78, 92, 38), .12), ('subtropical_castanopsis', (52, 78, 30), .12),
    ('bamboo', (88, 118, 46), .08), ('arid_tree_poplar', (176, 128, 44), .09),
    ('general_rock', (150, 138, 120), .08), ('rock_shrub', (120, 112, 90), .04),
    ('farm_foliage', (110, 120, 50), .04), ('generic_flower', (170, 120, 140), .03), ('', (70, 90, 40), .08)]


# --------------------------------------------------------------------- sources
class Ground:
    """Memory-mapped global blend + LF heights for the modded stage or vanilla."""
    def __init__(self, stage=None, vanilla=False, allow_missing_seamap=False):
        self.vanilla = vanilla
        self.packs = VanillaPacks()
        if stage and (Path(stage) / 'global_map').is_dir():   # given the variant folder itself
            stage = Path(stage).parents[2]
        self.stage = Path(stage) if stage else cw.WORKSPACE / 'work/release-input'
        pre = f'terrain/campaigns/{VARIANT}/'
        self.blend = self._map(pre + 'global_map/global_blend.dds', np.uint8, 2)
        self.lf = self._map(pre + 'lf_height_map.dds', '<u2', 0)
        # Water-surface height, same u16 encoding as the LF at half its resolution. The stage's
        # own map only: _map falls back to the vanilla packs, whose native-frame map silently drew
        # vanilla water over the canvas when the stage copy was missing or unreadable.
        key = pre + 'lf_sea_height_map.dds'
        if vanilla:
            try:
                self.seamap = self._map(key, '<u2', 0)
            except KeyError:
                self.seamap = None
        else:
            try:
                if not (self.stage / key).is_file():
                    raise FileNotFoundError(self.stage / key)
                self.seamap = self._map(key, '<u2', 0)
            except (OSError, ValueError, struct.error) as e:
                if not allow_missing_seamap:
                    raise SystemExit(f'stage lf_sea_height_map unreadable ({e!r}); fix the stage, or pass '
                                     '--allow-missing-seamap to draw flat threshold water instead')
                print(f'WARNING: stage lf_sea_height_map unreadable ({e!r}): drawing flat threshold water '
                      '(--allow-missing-seamap), NOT the stage water surface', file=sys.stderr, flush=True)
                self.seamap = None
        if vanilla:
            self.world = cw.VANILLA_WORLD
            self.shift = tuple(cw.NATIVE_SHIFT_WORLD)   # canvas world -> native world: subtract
        else:
            self.world = (cw.WORLD_WIDTH, cw.WORLD_HEIGHT)
            self.shift = (0., 0.)
        self.trees_key = f'campaign_maps/{VARIANT}/display/trees/trees.campaign_tree_list'

    def _map(self, key, dtype, channels):
        p = self.stage / key
        if not self.vanilla and p.is_file():
            path, base = p, 0
        else:
            path, base, size, comp = self.packs.index[key]
            assert not comp
        with open(path, 'rb') as f:
            f.seek(base)
            head = f.read(128)
        h, w = struct.unpack_from('<2I', head, 12)
        shape = (h, w, channels) if channels else (h, w)
        return np.memmap(path, dtype=dtype, mode='r', offset=base + 128, shape=shape)

    def read(self, key):
        p = self.stage / key
        return p.read_bytes() if (not self.vanilla and p.is_file()) else self.packs.read(key)

    # --- local frame: x right, z north, both in the source's own world units
    def local(self, wx, wz):
        return wx - self.shift[0], wz - self.shift[1]

    def texel(self, arr, wx, wz):
        """Continuous (col, row-from-top) of world points in a north-first raster."""
        lx, lz = self.local(wx, wz)
        h, w = arr.shape[:2]
        return lx / self.world[0] * w, (1 - lz / self.world[1]) * h

    def window(self, arr, cols, rows, pad=2):
        c0 = int(max(0, np.floor(np.nanmin(cols)) - pad)); c1 = int(min(arr.shape[1], np.ceil(np.nanmax(cols)) + pad + 1))
        r0 = int(max(0, np.floor(np.nanmin(rows)) - pad)); r1 = int(min(arr.shape[0], np.ceil(np.nanmax(rows)) + pad + 1))
        return np.asarray(arr[r0:r1, c0:c1]), c0, r0

    def heights(self, wx, wz):
        """Bilinear LF world heights in final_terrain_surface.sample_final_surface's frame:
        tile t (x / DX * 2, z / DZ * 2) is LF pixel 4t, rows south-up; the vanilla LF is
        the native window of it (NATIVE_OFFSET_LF). The old world-size texel basis with
        pixel centres at +0.5 read up to a pixel off."""
        return self._bilinear(self.lf, cw.LF_PX_PER_HEX, cw.NATIVE_OFFSET_LF, wx, wz)

    def sea_heights(self, wx, wz):
        """Bilinear water-surface world heights from lf_sea_height_map (sea_px: tile t is
        pixel 2t, the LF frame at half resolution; vanilla window at NATIVE_OFFSET_SEA)."""
        return self._bilinear(self.seamap, cw.SEA_PX_PER_HEX, cw.NATIVE_OFFSET_SEA, wx, wz)

    def _bilinear(self, arr, per, native_off, wx, wz):
        h, w = arr.shape
        ox, oy = native_off if self.vanilla else (0, 0)
        u = np.clip(np.asarray(wx, np.float64) / cw.DX * per - ox, 0, w - 1)
        v = np.clip(np.asarray(wz, np.float64) / cw.DZ * per - oy, 0, h - 1)
        win, c0, r0 = self.window(arr, u, h - 1 - v)   # the DDS raster is north-first
        win = win[::-1]; s0 = h - r0 - win.shape[0]          # south-up window and its first LF row
        u0, v0 = u.astype(np.intp), v.astype(np.intp); fu, fv = u - u0, v - v0
        u1, v1 = np.minimum(u0 + 1, w - 1) - c0, np.minimum(v0 + 1, h - 1) - s0; u0, v0 = u0 - c0, v0 - s0
        a, b, c, d = (win[y, x].astype(np.float64) for y, x in ((v0, u0), (v0, u1), (v1, u0), (v1, u1)))
        raw = (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv
        return cw.LEVEL + (raw / 65535 - cw.NORM) * cw.GAIN

    def blend_corners(self, wx, wz):
        """Shader replay: 4 corner (R, G) texels and the lerp fractions."""
        cols, rows = self.texel(self.blend, wx, wz)
        win, c0, r0 = self.window(self.blend, cols, rows)
        ix, iy = np.floor(cols).astype(int), np.floor(rows).astype(int)
        fx, fy = cols - ix, rows - iy
        out = []
        for dy in (0, 1):
            for dx in (0, 1):
                x = np.clip(ix + dx - c0, 0, win.shape[1] - 1); y = np.clip(iy + dy - r0, 0, win.shape[0] - 1)
                out.append(win[y, x])
        return out, fx, fy


# --------------------------------------------------------------------- shading
class Shader:
    def __init__(self, ground, tile=2.0, relief=1.0, sea_level=0.2, sun=(315, 40), detail=True, season='harvest', tiles=None,
                 water='meshes'):
        self.g = ground
        self.seamap = water == 'seamap' and ground.seamap is not None   # else flat sea_level
        self.meshes = None          # --water meshes: campaign_water_meshes.WaterMeshes
        if water == 'meshes':
            from campaign_water_meshes import WaterMeshes
            self.meshes = WaterMeshes(ground, VARIANT, OUT / 'cache')
        self.mwin = None            # (x0, z0, step, nx, nz, water grid, land grid) in the source frame
        self.tiles = tiles          # campaign_fine_tiles.FineTiles or None
        self.season = season
        self.tile = tile            # world units per texture repeat (g_blend_pixel_scale_array)
        self.relief = relief
        self.sea = sea_level
        self.planes = []            # HEHUANG PATCH: (x0, z0, x1, z1, y) lake-prop water planes
        az, alt = map(math.radians, sun)
        self.sun = np.array([math.sin(az) * math.cos(alt), math.sin(alt), math.cos(az) * math.cos(alt)], np.float32)
        self.mips = {}
        self.detail = None
        if detail:
            from campaign_ground_textures import decode_dds
            try:
                d = decode_dds(ground.packs.read('terrain/textures/campaign/default/medium_detail_map_colour.dds'), 0)
                k = max(1, d.shape[0] // 512)
                d = d[:d.shape[0] // k * k, :d.shape[1] // k * k].reshape(d.shape[0] // k, k, d.shape[1] // k, k, 3)
                self.detail = d.mean((1, 3)).astype(np.float32) / 255
            except KeyError:
                pass

    def layers(self, mip):
        if mip not in self.mips:
            self.mips[mip] = load_texture_layers(mip, self.g.packs, season=self.season)[2]
        return self.mips[mip]

    def albedo(self, wx, wz, footprint):
        """sRGB float colour (0..1) of world points; footprint = world size of one output pixel."""
        corners, fx, fy = self.g.blend_corners(wx, wz)
        mip = int(np.clip(np.round(np.log2(max(footprint, 1e-6) / self.tile * 1024)), 1, 9))
        tex = self.layers(mip)
        S = tex.shape[1]
        u = (np.mod(wx / self.tile, 1) * S).astype(int) % S
        v = (np.mod(-wz / self.tile, 1) * S).astype(int) % S
        cols = [tex[np.clip(c[..., 0], 0, tex.shape[0] - 1), v, u].astype(np.float32) for c in corners]
        fx, fy = fx[..., None], fy[..., None]
        top = cols[0] * (1 - fx) + cols[1] * fx
        bot = cols[2] * (1 - fx) + cols[3] * fx
        rgb = (top * (1 - fy) + bot * fy) / 255
        if self.tiles is not None:
            ui, vi = u.ravel(), v.ravel()
            rgb, _ = self.tiles.composite(rgb, wx, wz, footprint,
                                          lambda layer, idx: tex[layer, vi[idx], ui[idx]].astype(np.float32) / 255)
        if self.detail is not None:
            # mid-distance detail map, photoshop overlay as in the shader, at a guessed strength
            D = self.detail.shape[0]
            du = (np.mod(wx / (self.tile * 8), 1) * D).astype(int) % D
            dv = (np.mod(-wz / (self.tile * 8), 1) * D).astype(int) % D
            d = self.detail[dv, du]
            ov = np.where(rgb < .5, 2 * rgb * d, 1 - 2 * (1 - rgb) * (1 - d))
            rgb = rgb + .35 * (ov - rgb)
        return rgb

    def water_level(self, wx, wz):
        """Water-surface height the game draws at world points (meshes: -inf where none; seamap or flat sea level)."""
        if self.meshes:
            return self.mesh_level(wx, wz)
        return self.g.sea_heights(wx, wz) if self.seamap else np.full(np.shape(wx), self.sea)

    def mesh_level(self, wx, wz, cells=4096):
        """Highest water triangle where it is above the land mesh (nearest cell), else -inf. Land mesh
        holes (fine tiles: river beds, borders, coasts) count as the LF ground. The window is rasterised
        on demand to cover the points (<= cells^2, >= 16 cells per hex) and reused."""
        lx, lz = (np.asarray(a, np.float64) for a in self.g.local(wx, wz))
        x0, x1, z0, z1 = lx.min(), lx.max(), lz.min(), lz.max()
        w = self.mwin
        if not w or x0 < w[0] or z0 < w[1] or x1 > w[0] + w[3] * w[2] or z1 > w[1] + w[4] * w[2]:
            pad = .02 * max(x1 - x0, z1 - z0) + cw.DX
            x0, z0, x1, z1 = x0 - pad, z0 - pad, x1 + pad, z1 + pad
            step = max((x1 - x0) / cells, (z1 - z0) / cells, cw.DX / 16)
            nx, nz = int(np.ceil((x1 - x0) / step)), int(np.ceil((z1 - z0) / step))
            w = self.mwin = (x0, z0, step, nx, nz) + self.meshes.fields(x0, z0, step, nx, nz)
        i = np.clip(((lx - w[0]) / w[2]).astype(np.intp), 0, w[3] - 1)
        j = np.clip(((lz - w[1]) / w[2]).astype(np.intp), 0, w[4] - 1)
        water, land = w[5][j, i].astype(np.float64), w[6][j, i].astype(np.float64)
        hole = np.isfinite(water) & ~np.isfinite(land)
        if hole.any():
            land[hole] = self.g.heights(np.asarray(wx, np.float64)[hole], np.asarray(wz, np.float64)[hole])
        return np.where(water > land, water, -np.inf)

    def surface(self, wx, wz):
        """Visible top: max(ground, water) - what the oblique ray march hits. Mesh water replaces the
        LF ground wherever it shows (the game's ground there is the land mesh below it)."""
        if self.meshes:
            level = self.mesh_level(wx, wz)
            return np.where(np.isfinite(level), level, self.g.heights(wx, wz))
        return np.maximum(self.g.heights(wx, wz), self.water_level(wx, wz))

    def normals(self, wx, wz, eps):
        hx = self.g.heights(wx + eps, wz) - self.g.heights(wx - eps, wz)
        hz = self.g.heights(wx, wz + eps) - self.g.heights(wx, wz - eps)
        n = np.stack([-hx * self.relief, np.full_like(hx, 2 * eps), -hz * self.relief], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def shade(self, wx, wz, footprint):
        h = self.g.heights(wx, wz)
        rgb = self.albedo(wx, wz, footprint)
        n = self.normals(wx, wz, max(footprint, .04))
        lam = np.clip((n * self.sun).sum(-1), 0, 1)
        light = .45 + .75 * lam / self.sun[1] * .75
        rgb = rgb * np.clip(light, .25, 1.35)[..., None]
        level = self.water_level(wx, wz)
        wet = np.isfinite(level) if self.meshes else h < level   # mesh water shows even above the LF
        for x0, z0, x1, z1, y in self.planes:   # a lake prop shows wherever the ground is under it
            inside = (wx >= x0) & (wx <= x1) & (wz >= z0) & (wz <= z1)
            level = np.where(inside, np.maximum(level, y), level); wet |= inside & (h < y)
        depth = np.clip((level - h) / 1.5, 0, 1)[..., None]
        water = wet.astype(np.float64)[..., None]
        wcol = (WATER_SHALLOW * (1 - depth) + WATER_DEEP * depth) / 255
        rgb = rgb * (1 - water) + wcol * water
        return np.clip(rgb, 0, 1), h


# ----------------------------------------------------------------- overlays
def load_towns(design=None):
    """[(region, x, z, is_capital)] of every region town of a region design (default cw.DESIGN_PATH): the towns the
    renderers draw as house boxes. art_review passes each side's own design (art_review.cmd_run --design/--base-design)."""
    d = json.loads(Path(design or cw.DESIGN_PATH).read_text(encoding='utf-8'))
    out = []
    for r in d['regions']:
        if r.get('town'):
            x, z = cw.display_position(*r['town'])
            out.append((r['region'], x, z, bool(r.get('is_capital'))))
    return out


def load_trees(ground):
    try:
        blob = ground.read(ground.trees_key)
    except (KeyError, FileNotFoundError):
        return []
    pos = 24
    groups = []
    for _ in range(struct.unpack_from('<I', blob, 20)[0]):
        n = struct.unpack_from('<H', blob, pos)[0]; pos += 2
        name = blob[pos:pos + n].decode(); pos += n
        count = struct.unpack_from('<I', blob, pos)[0]; pos += 4
        xyz = np.empty((count, 3), np.float32)
        for i in range(count):
            xyz[i] = struct.unpack_from('<3f', blob, pos)
            pos += 18 + struct.unpack_from('<I', blob, pos + 14)[0] * 4
        if ground.vanilla:
            xyz[:, 0] += ground.shift[0]; xyz[:, 2] += ground.shift[1]
        colour, radius = next((c, r) for p, c, r in TREE_COLOURS if name.startswith(p))
        groups.append((name, xyz, colour, radius))
    return groups


# ----------------------------------------------------------------- top-down
def render_topdown(shader, cx, cz, half, size, trees=(), towns=(), label=None, markers=()):
    p = 2 * half / size
    jj, ii = np.mgrid[:size, :size].astype(np.float64)
    wx = cx - half + (ii + .5) * p
    wz = cz + half - (jj + .5) * p
    rgb, _ = shader.shade(wx, wz, p)
    img = (rgb * 255).astype(np.uint8)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for name, xyz, colour, radius in trees:
        m = (np.abs(xyz[:, 0] - cx) < half) & (np.abs(xyz[:, 2] - cz) < half)
        if not m.any():
            continue
        r = max(.6, radius / p)
        dark = tuple(int(c * .6) for c in colour)
        for x, _, z in xyz[m]:
            px, py = (x - cx + half) / p, (cz + half - z) / p
            dr.ellipse([px - r, py - r, px + r, py + r], fill=colour, outline=dark if r > 2 else None)
    for name, x, z, cap in towns:
        if abs(x - cx) < half and abs(z - cz) < half:
            px, py = (x - cx + half) / p, (cz + half - z) / p
            s = 6 if cap else 4
            dr.rectangle([px - s, py - s, px + s, py + s], outline=(255, 255, 255), fill=(200, 30, 30) if cap else (230, 160, 30))
            dr.text((px + s + 2, py - 6), name.replace('tw3k_hist_', '').replace('tw3k_exp_', ''), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    for x, z, colour, rad in markers:   # --markers: prop positions (render draws no props)
        if abs(x - cx) < half and abs(z - cz) < half:
            px, py = (x - cx + half) / p, (cz + half - z) / p
            dr.ellipse([px - rad, py - rad, px + rad, py + rad], fill=tuple(colour), outline=(0, 0, 0))
    # scale bar: 10 hexes
    bar = 10 * cw.HEX_DX / p
    dr.rectangle([10, size - 18, 10 + bar, size - 12], fill=(255, 255, 255), outline=(0, 0, 0))
    dr.text((14 + bar, size - 22), '10 hex', fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    if label:
        dr.text((8, 6), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return im


# ----------------------------------------------------------------- oblique
def camera_basis(cx, cz, dist, heading, height, ground, model):
    hy = float(ground.heights(np.array([cx]), np.array([cz]))[0])
    th = math.radians(heading)
    fwd_h = np.array([math.sin(th), 0, math.cos(th)])
    horiz = math.sqrt(max(dist ** 2 - height ** 2, 1e-6)) if model == 'slant' else dist
    eye = np.array([cx, hy, cz]) - fwd_h * horiz + np.array([0, height, 0])
    f = np.array([cx, hy, cz]) - eye; f /= np.linalg.norm(f)
    right = np.cross([0, 1, 0], f); right /= np.linalg.norm(right)   # x east, y up, z north
    up = np.cross(f, right)
    return eye, f, right, up


def render_oblique(shader, cam, size=(1280, 720), fov=35., model='horizontal', trees=(), towns=(), label=None):
    cx, cz, dist, heading, height = cam
    g = shader.g
    W, H = size
    eye, f, right, up = camera_basis(cx, cz, dist, heading, height, g, model)
    t = math.tan(math.radians(fov) / 2)
    jj, ii = np.mgrid[:H, :W].astype(np.float64)
    sx = (2 * (ii + .5) / W - 1) * t * W / H
    sy = (1 - 2 * (jj + .5) / H) * t
    d = f[None, None] + sx[..., None] * right + sy[..., None] * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    # Heights of the region the frustum can reach (coarse bound for marching).
    reach = dist * 6
    gx = np.linspace(cx - reach, cx + reach, 256); gz = np.linspace(cz - reach, cz + reach, 256)
    GX, GZ = np.meshgrid(gx, gz)
    hh = shader.surface(GX, GZ)
    hmax, hmin = float(hh.max()) + .2, float(hh.min()) - .2
    hit = np.full((H, W), np.nan)
    dy = d[..., 1]
    down = dy < -1e-4
    t0 = np.where(down, (eye[1] - hmax) / -np.where(down, dy, -1), np.nan)
    t1 = np.where(down, (eye[1] - hmin) / -np.where(down, dy, -1), np.nan)
    t0 = np.maximum(t0, 0)
    steps = 160
    lo = t0.copy(); found = np.zeros((H, W), bool)
    prev = t0.copy()
    for k in range(1, steps + 1):
        tk = t0 + (t1 - t0) * k / steps
        act = down & ~found
        if not act.any():
            break
        px = eye[0] + d[..., 0][act] * tk[act]; pz = eye[2] + d[..., 2][act] * tk[act]
        py = eye[1] + dy[act] * tk[act]
        gh = shader.surface(px, pz)
        below = py <= gh
        idx = np.flatnonzero(act)[below]
        found.flat[idx] = True
        lo.flat[idx] = prev.flat[idx]
        hit.flat[idx] = tk.flat[idx]
        prev[act] = tk[act]
    # refine by bisection
    a, b = lo[found], hit[found]
    dd = d[found]
    for _ in range(8):
        m = (a + b) / 2
        px = eye[0] + dd[:, 0] * m; pz = eye[2] + dd[:, 2] * m; py = eye[1] + dd[:, 1] * m
        below = py <= shader.surface(px, pz)
        b = np.where(below, m, b); a = np.where(below, a, m)
    px = eye[0] + dd[:, 0] * b; pz = eye[2] + dd[:, 2] * b
    foot = b * 2 * t / H / np.clip(-dd[:, 1], .2, 1)
    rgb = np.zeros((H, W, 3), np.float32) + np.array([.62, .68, .74], np.float32)  # sky/haze
    col, _ = shader.shade(px, pz, float(np.median(foot)))
    # distance haze
    haze = np.clip((b - dist) / (dist * 6), 0, .6)[:, None]
    col = col * (1 - haze) + np.array([.62, .68, .74]) * haze
    rgb[found] = col
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)

    def project(p):
        v = p - eye
        z = v @ f
        if z <= .1:
            return None
        return (W / 2 + (v @ right) / z / t * H / 2, H / 2 - (v @ up) / z / t * H / 2, z)
    pts = []
    for name, xyz, colour, radius in trees:
        m = (np.abs(xyz[:, 0] - cx) < reach) & (np.abs(xyz[:, 2] - cz) < reach)
        for x, y, z in xyz[m]:
            pr = project(np.array([x, y + radius, z]))
            if pr and 0 <= pr[0] < W and 0 <= pr[1] < H:
                pts.append((pr[2], pr[0], pr[1], colour, radius))
    pts.sort(key=lambda q: -q[0])
    for z, x, y, colour, radius in pts:
        r = max(.7, radius / z / t * H / 2)
        dr.ellipse([x - r, y - 1.4 * r, x + r, y + r], fill=colour, outline=tuple(int(c * .6) for c in colour) if r > 2.5 else None)
    for name, x, z, cap in towns:
        y = float(g.heights(np.array([x]), np.array([z]))[0])
        pr = project(np.array([x, y, z]))
        if pr and 0 <= pr[0] < W and 0 <= pr[1] < H:
            s = 6 if cap else 4
            dr.rectangle([pr[0] - s, pr[1] - s, pr[0] + s, pr[1] + s], outline=(255, 255, 255), fill=(200, 30, 30) if cap else (230, 160, 30))
            dr.text((pr[0] + s + 2, pr[1] - 6), name.replace('tw3k_hist_', '').replace('tw3k_exp_', ''), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    if label:
        dr.text((8, 6), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return im


# ----------------------------------------------------------------- CLI
CAM_ROW = re.compile(r'^\|\s*([^|`]+?)\s*\|\s*`?\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)`?')


def parse_cameras(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        m = CAM_ROW.match(line.strip())
        if m:
            rows.append((m.group(1).strip(), tuple(float(v) for v in m.groups()[1:])))
    return rows


def safe(name):
    return re.sub(r'[^\w.-]+', '_', name).strip('_') or 'view'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world', nargs=2, type=float, metavar=('X', 'Z'))
    ap.add_argument('--hex', nargs=2, type=int, metavar=('Q', 'R'))
    ap.add_argument('--cameras', type=Path, help='markdown test doc with | name | (x,z,dist,heading,pitch) | rows')
    ap.add_argument('--radius', type=float, default=40, help='half window in hexes (top-down)')
    ap.add_argument('--size', type=int, default=1024, help='top-down output pixels')
    ap.add_argument('--out', type=Path, help='output png (single view) or directory (--cameras)')
    ap.add_argument('--stage', type=Path, help='release-input root or the 3k_dlc07_main_map folder')
    ap.add_argument('--vanilla', action='store_true', help='render the vanilla packs instead (native map)')
    ap.add_argument('--oblique', help='also render a perspective view: dist,heading,pitch (pitch = camera height)')
    ap.add_argument('--fov', type=float, default=35., help='vertical field of view (deg) for oblique views')
    ap.add_argument('--camera-model', choices=('slant', 'horizontal'), default='horizontal',
                    help='dist is the eye-target distance (slant) or the horizontal back-off')
    ap.add_argument('--tile', type=float, default=2.0, help='world units per ground-texture repeat')
    ap.add_argument('--relief', type=float, default=1.0, help='hill-shading exaggeration')
    ap.add_argument('--water', choices=('meshes', 'seamap', 'threshold'), default='meshes',
                    help='meshes: sea/ocean-background/river/lake water triangles above the coarse land mesh (as the '
                         'game); seamap: where lf_sea_height_map is above the LF (the game does not draw it; a stage '
                         'without a readable map fails unless --allow-missing-seamap); threshold: flat --sea-level')
    ap.add_argument('--allow-missing-seamap', action='store_true',
                    help='stage renders: draw threshold water (with a warning) when the stage lf_sea_height_map is '
                         'missing or unreadable, instead of failing')
    ap.add_argument('--sea-level', type=float, default=0.2, help='--water threshold: world height drawn as water (land LF heights start ~0.26)')
    ap.add_argument('--season', default='harvest', choices=('xml', 'spring', 'summer', 'harvest', 'autumn', 'winter'),
                    help='ground texture season (190 starts in harvest; xml = texture_arrays.xml as listed)')
    ap.add_argument('--no-trees', action='store_true')
    ap.add_argument('--water-plane', action='append', default=[], help='x0,z0,x1,z1,y: also draw water under a lake prop rectangle at height y; '
                    '"hehuang" reads analysis/hehuang-patch.npz')
    ap.add_argument('--no-detail', action='store_true')
    ap.add_argument('--markers', type=Path, help='JSON [[x, z, [r, g, b], radius px], ...] drawn on the top-down view '
                    '(e.g. analysis/coast-props-markers.json from coast_props.py)')
    ap.add_argument('--tiles', nargs='?', const='all', metavar='SETS',
                    help='composite fine tiles (blend0 splat maps, custom meshes); optional comma list of tile-set '
                         'prefixes, e.g. roads_,salt_pools')
    # FARMS-ROCKS PATCH: prop marker overlay (prop_markers.py); props are not rendered otherwise
    ap.add_argument('--props', nargs='?', const='stage', metavar='GLOBAL_PROPS',
                    help='mark mountain/rock prop instances (top-down): the stage global_props.bin or a given file')
    args = ap.parse_args()

    ground = Ground(args.stage, args.vanilla, args.allow_missing_seamap or args.water != 'seamap')
    tiles = None
    if args.tiles:
        from campaign_fine_tiles import FineTiles
        tiles = FineTiles(ground, None if args.tiles == 'all' else args.tiles.split(','), VARIANT)
    shader = Shader(ground, tile=args.tile, relief=args.relief, sea_level=args.sea_level, detail=not args.no_detail,
                    season=args.season, tiles=tiles, water=args.water)
    for wp in args.water_plane:   # HEHUANG PATCH
        if wp == 'hehuang':
            import hehuang_terrain
            pr = hehuang_terrain.load().prop
            shader.planes.append((*pr['world_rect'], pr['water']))
        else:
            shader.planes.append(tuple(float(v) for v in wp.split(',')))
    trees = [] if args.no_trees else load_trees(ground)
    towns = [] if args.vanilla else load_towns()
    tag = ('vanilla' if args.vanilla else 'stage') + ('-tiles' if tiles else '') + \
          ('' if shader.meshes else '-seamap' if shader.seamap else '-flatsea')
    views = []
    if args.cameras:
        for name, (x, z, dist, heading, pitch) in parse_cameras(args.cameras):
            views.append((name, x, z, (x, z, dist, heading, pitch)))
        outdir = args.out or OUT / safe(args.cameras.stem)
    else:
        if args.hex:
            x, z = cw.display_position(*args.hex)
        elif args.world:
            x, z = args.world
        else:
            ap.error('give --world, --hex or --cameras')
        cam = None
        if args.oblique:
            dist, heading, pitch = (float(v) for v in args.oblique.split(','))
            cam = (x, z, dist, heading, pitch)
        views.append((None, x, z, cam))
        outdir = None
    for name, x, z, cam in views:
        q, r = cw.world_hex(x, z)
        lab = f'{name or ""} {tag} world ({x:.2f},{z:.2f}) hex ({int(q)},{int(r)}) radius {args.radius:g} hex'.strip()
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            top_path = outdir / f'{safe(name)}-top.png'
        else:
            top_path = args.out or OUT / f'view-{tag}-{x:.1f}-{z:.1f}.png'
            top_path.parent.mkdir(parents=True, exist_ok=True)
        markers = json.loads(args.markers.read_text(encoding='utf-8')) if args.markers else ()
        im = render_topdown(shader, x, z, args.radius * cw.HEX_DX, args.size, trees, towns, lab, markers)
        if args.props:   # FARMS-ROCKS PATCH
            import prop_markers
            blob = ground.read(f'terrain/campaigns/{VARIANT}/global_props.bin') if args.props == 'stage' else Path(args.props).read_bytes()
            print('prop markers', prop_markers.draw(im, blob, x, z, args.radius * cw.HEX_DX, args.size))
        im.save(top_path)
        print('wrote', top_path)
        if cam:
            ob = render_oblique(shader, cam, fov=args.fov, model=args.camera_model, trees=trees, towns=towns,
                                label=f'{name or ""} {tag} camera {cam}')
            ob_path = top_path.with_name(top_path.stem.replace('-top', '') + '-oblique.png')
            ob.save(ob_path)
            print('wrote', ob_path)


if __name__ == '__main__':
    main()

"""Offline campaign view tuned toward the live game look (renderer only, never used for packaging).

Builds on render_campaign_view.py (whose default behaviour is left untouched) and reads its
inputs straight from a built mod pack overlaid on the vanilla packs, so the render shows exactly
what the game showed for that pack (e.g. the 2048p live check: candidate v16, sha256 e89b4179...).
No stage or live-workspace file is opened.

  PackSource   vanilla pack index with the mod pack's entries on top (mod pack wins, as in game)
  PackGround   render_campaign_view.Ground on a PackSource (global blend, LF, sea map memory-mapped
               read-only from the pack file)
  PackWater    campaign_water_meshes.WaterMeshes listing its meshes from the PackSource

Caches go to output/offline-view/render-calib/cache under the main checkout (tw3k_paths.MAIN_ROOT), never to the
shared offline-view cache files of other kinds.
"""
import tw3k_paths
import math
import os
import re
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# All output of this tool lives under the main checkout's output/offline-view (worktrees have no
# output folder of their own). The vanilla texture/index caches there are shared read-mostly.
MAIN_ROOT = tw3k_paths.MAIN_ROOT
OUT = MAIN_ROOT / 'output/offline-view/render-calib'
CACHE = OUT / 'cache'
CANDIDATE_PACK = MAIN_ROOT / 'output/tw3k_regions_and_battles_terrain_repair_candidate.pack'
V16_SHA256 = 'e89b417948334d86d791304a9cfefd9e6161de87a51ce0effba38dbd08dc8c5c'
V16_STAT = (4360035419, 1792913156896542000)   # size, mtime_ns of the v16 build (10-25 00:25)


def find_v16():
    """The v16 candidate pack the 2048p screenshots show. Later builds replace the candidate path and keep
    the older one as candidate.previous-*.pack (a rename: size and mtime stay), so it is found by those."""
    for p in sorted((MAIN_ROOT / 'output').glob('tw3k_regions_and_battles_terrain_repair_candidate*.pack')):
        st = p.stat()
        if (st.st_size, st.st_mtime_ns) == V16_STAT:
            return p
    return None


# TW3K_RENDER_CALIB_PACK overrides; else v16 wherever it now lives; else the current candidate.
DEFAULT_PACK = Path(os.environ.get('TW3K_RENDER_CALIB_PACK') or find_v16() or CANDIDATE_PACK)
# campaign_world reads the canvas (grid, padding) from a region design at import, and the design gives the towns
# drawn as house boxes (render_campaign_view.load_towns). There is no default (2026-11-01: the old default, a
# private copy under <main>-envcheck, was a 10-25 design: 44 old and 28 missing Korea towns against the c14r design).
# TW3K_DESIGN must name the design of the pack being rendered: art_review sets each side's (--design /
# --base-design); a manual render sets it by hand. The workspace defaults to an empty private folder, so the live
# workspace (work/all-directions-30) is never opened by this tool.
if not os.environ.get('TW3K_DESIGN'):
    raise RuntimeError('render_live_view needs TW3K_DESIGN: the region design of the pack it renders (its towns are '
                       'drawn); art_review sets it per side, a manual render sets it by hand')
os.environ.setdefault('TW3K_WORKSPACE', str(tw3k_paths.SCRATCH / 'render-ws'))

import campaign_ground_textures as cgt  # noqa: E402
cgt.CACHE = MAIN_ROOT / 'output/offline-view/cache'   # vanilla pack index + decoded texture layers

import campaign_world as cw  # noqa: E402
import render_campaign_view as rv  # noqa: E402
from campaign_water_meshes import WaterMeshes  # noqa: E402

VARIANT = rv.VARIANT
PRE = f'terrain/campaigns/{VARIANT}/'


class PackSource(cgt.VanillaPacks):
    """Vanilla index with a mod pack's (uncompressed) entries on top; same read()/keys() API."""
    def __init__(self, pack=DEFAULT_PACK):
        super().__init__()
        self.pack = Path(pack)
        st = self.pack.stat()
        CACHE.mkdir(parents=True, exist_ok=True)
        cache = CACHE / f'pack-index-{self.pack.stem}-{st.st_size}-{st.st_mtime_ns}.pkl'
        if cache.is_file():
            mod = pickle.loads(cache.read_bytes())
        else:
            from analyze_reference import inventory
            mod = {e['path']: (str(self.pack), e['offset'], e['size'], e['compressed'])
                   for e in inventory(self.pack, False)['entries']}
            cache.write_bytes(pickle.dumps(mod))
        self.mod_keys = set(mod)
        self.index = {**self.index, **mod}


class PackGround(rv.Ground):
    """rv.Ground whose 'stage' is a built pack (canvas frame, not the vanilla native frame)."""
    def __init__(self, pack=DEFAULT_PACK):
        self.vanilla = False
        self.packs = PackSource(pack)
        self.stage = OUT / 'no-stage'          # never holds files: every _map/read goes to the pack index
        assert not self.stage.exists()
        self.blend = self._map(PRE + 'global_map/global_blend.dds', np.uint8, 2)
        self.lf = self._map(PRE + 'lf_height_map.dds', '<u2', 0)
        self.seamap = self._map(PRE + 'lf_sea_height_map.dds', '<u2', 0)
        self.world = (cw.WORLD_WIDTH, cw.WORLD_HEIGHT)
        self.shift = (0., 0.)
        self.trees_key = f'campaign_maps/{VARIANT}/display/trees/trees.campaign_tree_list'

    def read(self, key):
        return self.packs.read(key)

    # (x0, z0, step, nx, nz, land grid) of the coarse land mesh over the view (P['ground'] == 'mesh'), else None
    mesh = None

    def heights(self, wx, wz):
        """Ground heights: the LF, or with self.mesh the land mesh the game draws (bilinear over its cell centres);
        land-mesh holes (fine tiles: coasts, rivers, borders) and points outside the window keep the LF."""
        lf = super().heights(wx, wz)
        if self.mesh is None:
            return lf
        x0, z0, step, nx, nz, land = self.mesh
        fi = (np.asarray(wx, np.float64) - x0) / step - .5
        fj = (np.asarray(wz, np.float64) - z0) / step - .5
        inside = (fi >= 0) & (fi <= nx - 1) & (fj >= 0) & (fj <= nz - 1)
        i0 = np.clip(np.floor(fi).astype(np.intp), 0, nx - 2)
        j0 = np.clip(np.floor(fj).astype(np.intp), 0, nz - 2)
        u, v = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
        with np.errstate(invalid='ignore'):        # inf (hole) x 0 weight -> nan: the LF takes those points
            h = ((land[j0, i0] * (1 - u) + land[j0, i0 + 1] * u) * (1 - v)
                 + (land[j0 + 1, i0] * (1 - u) + land[j0 + 1, i0 + 1] * u) * v)
        return np.where(inside & np.isfinite(h), h, lf)


class PackWater(WaterMeshes):
    """Water meshes listed from the pack index (the stage version globs the stage folder)."""
    def _keys(self, sub, suffix):
        return self.g.packs.keys(self.pre + sub, suffix)

    def _signature(self):
        import hashlib
        h = hashlib.sha1(b'pack-water-1')
        for k in (self._keys('global_meshes/', '.rigid_model_v2') + self._keys('models/', '.rigid_model_v2')
                  + [self.pre + 'global_props.bin']):
            h.update(repr((k, self.g.packs.index[k][1:])).encode())
        return h.hexdigest()[:16]


def pack_shader(ground, cls=None, **kw):
    """A shader on the pack ground with mesh water read from the pack (cache under render-calib)."""
    cls = cls or rv.Shader
    sh = cls(ground, water='threshold', **kw)
    sh.meshes = PackWater(ground, VARIANT, CACHE / 'water')
    return sh


# ------------------------------------------------------------------ camera
def eye_height(ground, cam, abs_height):
    """Camera tuple for rv.render_oblique: the live camera's 5th value is the eye height above the
    target ground (rel) or the absolute eye y (abs, converted to a height above the target)."""
    x, z, dist, heading, h = cam
    if abs_height:
        h = h - float(ground.heights(np.array([x]), np.array([z]))[0])
    return (x, z, dist, heading, h)


_STATE = {}
# 'cpu' or 'gpu': gbuffer's ray marches, terrain AO and sun shadows on the GPU (render_gpu, bit for bit the NumPy
# result); the caller holds pipeline_lock.gpu around each render. art_review puts the backend in its cache key.
BACKEND = os.environ.get('TW3K_RENDER_BACKEND', 'cpu')


def state(pack=DEFAULT_PACK):
    """Shared ground, shader and tree list (loaded once per process)."""
    if pack not in _STATE:
        g = PackGround(pack)
        _STATE[pack] = dict(ground=g, shader=pack_shader(g), trees=rv.load_trees(g))
    return _STATE[pack]


def make_renderer(variant, pack=DEFAULT_PACK, **opts):
    """cam, (w, h) -> PIL image. 'baseline' = render_campaign_view.render_oblique (default look; options
    fov, abs_height, model, trees); 'live' = render_live with LIVE_DEFAULTS updated by opts."""
    st = state(pack)
    if variant == 'baseline':
        fov, abs_height = opts.pop('fov', 35.), opts.pop('abs_height', False)
        model, trees = opts.pop('model', 'horizontal'), opts.pop('trees', True)
        if opts:
            raise ValueError(f'unknown baseline options {sorted(opts)}')

        def render(cam, size):
            return rv.render_oblique(st['shader'], eye_height(st['ground'], cam, abs_height), size=size, fov=fov,
                                     model=model, trees=st['trees'] if trees else ())
        return render
    if variant == 'live':
        P = dict(LIVE_DEFAULTS)
        unknown = set(opts) - set(P)
        if unknown:
            raise ValueError(f'unknown live options {sorted(unknown)}')
        P.update(opts)
        live = live_state(pack, P)

        def render(cam, size):
            return render_live(live, cam, size, P)
        return render
    raise ValueError(f'unknown variant {variant}')


# ------------------------------------------------------------------ live look
NRO_SUFFIX = '_normal_roughness_occlusion_map.dds'
LIVE_DEFAULTS = dict(
    # Light and grade fit to 12 live shots (render_calib; distribution loss: per-shot Lab mean/std and
    # sharpness, which unlike plain dE/SSIM under misregistration does not reward flat or blurred
    # renders); fog colour, depth of field and occlusion strengths are set, not fit.
    fov=45., height_scale=1.0,             # camera (render_calib camera fit)
    lighting=True,                         # False: render_campaign_view's hill shading and haze
    normal_eps=.04,                        # LF normal difference step (world units)
    nro=1.0,                               # tangent normal strength of the per-layer normal_roughness_occlusion maps
    tex_occlusion=1.0,                     # weight of their occlusion channel
    sun_az=315., sun_alt=40., sun=1.0, ambient=.65, sky_tint=(.95, 1., 1.08), sun_tint=(1.06, 1., .9),
    shadows=True, shadow_soft=.3,          # height-field sun shadows (shadow_soft: light left in shadow)
    ao=1.0, ao_radii=(.6, 2.5),            # terrain horizon ambient occlusion
    exposure=.75, contrast=1.05, saturation=.8, gamma=.95, gain=(.94, 1., 1.03),
    fog_density=.21, fog_start=.5, haze=(.78, .82, .86),   # aerial perspective (start: x eye-target distance)
    dof=2., dof_focus=1.0,                 # depth-of-field blur (px at 1280 wide per unit CoC), focus at target distance x
    water_shallow=(60, 90, 110), water_deep=(30, 32, 70),
    sky=(.78, .82, .86),
    trees=True, props=True,                # height-field impostors (render_live_objects)
    # tree-list scale and canopy tint / foot occlusion fit like the light (scale .3 vs .42: loss 16.3 vs 16.8)
    tree_scale=.3, tree_height=1., tree_lump=.6, crown_size=.3, canopy_flat=1.6, tree_gap=.45, tree_tint=(1.7, 1.55, 1.4),
    object_cell=.03, object_ao=.35,
    # Tile cells no tile_list.bin record covers are holes in game (the land mesh leaves them to a fine tile;
    # v16 Lin Yi: open sea without its generic_sea floor). Drawn flat in this colour, after fog, never
    # filled with the LF ground or water. None: off.
    missing_tiles=(1., 0., 0.),
    # Height fog and sky (hfog=True): the live haze is a blue-grey layer aloft (negative falloff: fog grows with
    # height), so the high strategic and border cameras look through it and ordinary ones stay clear; it is
    # composited after the grade in display colours, and rays that hit nothing get a horizon-to-zenith
    # gradient. Fit on cached gbuffers: 8 horizon shots + 6 ordinary guard shots (loss dist - 20 ssim6), fog
    # colour seeded with the screenshots' top-band mean (x1.05); 2 flat shots held out. A first fit on the
    # horizon shots alone chose plain distance fog and washed out the ordinary views (41-shot dist 21.6 -> 26.6).
    # hfog=False keeps the step-2 aerial perspective above (fog_density / fog_start / haze, before the grade).
    hfog=True, hfog_colour=(.583, .623, .721), hfog_density=.0028, hfog_start=.8, hfog_falloff=-.09, hfog_max=.95,
    sky_zenith=(.62, .72, .86), sky_span=.25,
    # Towns (render_live_objects): 'layout' = the vanilla settlement layout prefabs at each town (pieces, heights
    # and colours read from the packs; scale and building level fit), 'boxes' = the older house-box cluster.
    # town_min: pieces smaller than this (world units) are dropped; town_decal: strength of the ground decals.
    towns='boxes', town_scale=.07, town_level=(2, 1), town_min=.01, town_decal=1.,
    # Ground heights: 'mesh' = the coarse land mesh (global_meshes/land_mesh_*), which is what the game draws outside
    # the fine tiles (holes keep the LF); 'lf' = lf_height_map. v16: the two agree in the native map (0.9% of points
    # > 0.25 wu) but not on much of the canvas (15.5%, banded), where 'mesh' removes plateau rims and gullies the game
    # does not show (41 shots: dist 15.33 -> 14.75, ssim6 .328 -> .343).
    ground='mesh',
    # Canopies: tree_colour 'species' = render_campaign_view's per-family guess lit with tree_tint; 'texture' = each clump
    # model's own leaf atlas (season-swapped, alpha-weighted; read from the packs) lit with tree_tint_texture (fit on
    # 6 forest shots, dist - 20 ssim6; not the default: the 41-shot gate failed). The atlas mean includes bark and
    # branches (olive-brown): a leaf-only mask from the atlas alpha or material channel is the next step. A
    # distant-canopy fade was tried and rejected.
    tree_colour='species', tree_tint_texture=(1.262, 1.743, .539),
)


def decode_bc3_rgba(data, mip):
    """RGBA of one mip of a BC3 DDS with a legacy or DX10 header."""
    from campaign_fine_tiles import bc3_rgba
    if data[84:88] == b'DX10':
        data = data[:84] + b'DXT5' + data[88:128] + data[148:]
    return bc3_rgba(data, mip)


def nro_layers(packs, mip, season):
    """uint8 [L, S, S, 4] normal_roughness_occlusion maps per texture-array layer (R roughness,
    G normal y, B occlusion, A normal x), the season's variant where shipped (else summer, else flat)."""
    cache = CACHE / f'nro_layers_{season}_mip{mip}.npy'
    if cache.is_file():
        return np.load(cache)
    _, paths = cgt.texture_array(packs, VARIANT)
    paths = cgt.seasonal_paths(paths, season, packs)
    out = []
    for p in paths:
        cands = [p.replace('_base_colour.dds', NRO_SUFFIX)]
        cands.append(re.sub(r'_(spring|harvest|autumn|winter)_', '_summer_', cands[0]))
        arr = None
        for c in cands:
            if c in packs.index:
                arr = decode_bc3_rgba(packs.read(c), mip)
                break
        out.append(arr)
    S = next(a.shape[0] for a in out if a is not None)
    flat = np.zeros((S, S, 4), np.uint8) + np.array([200, 128, 255, 128], np.uint8)
    arr = np.stack([a if a is not None and a.shape[0] == S else flat for a in out])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def tile_holes(packs, key=PRE + 'global_map/tile_list.bin'):
    """bool [TILE_H, TILE_W] (rows south-up, tile cell (x / CELL_X, z / CELL_Z)): cells that no record of the
    tile list covers. Footprint = the piece's mesh (or custom mesh) group box / 128 per cell, width and height
    swapped by the 0x20 / 0x80 quarter turns, from the record's minimum column / row (campaign_fine_tiles).
    Vanilla dlc07: 1,810 of 2.5 M cells; v16: 45,660, all inside the native map (Lin Yi sea, 38,244 of them
    open sea: stage_invariants.sea_floor_coverage)."""
    from campaign_fine_tiles import FineTiles, REC, mesh_group
    import hashlib
    src = packs.index[key]
    st = os.stat(src[0])
    sig = hashlib.sha1(repr((key, str(src[0]), src[1], src[2], st.st_size, st.st_mtime_ns)).encode()).hexdigest()[:16]
    cache = CACHE / f'tile-holes-{sig}.npy'
    if cache.is_file():
        return np.load(cache)
    blob = packs.read(key)
    paths, start = FineTiles._paths(blob)
    recs = np.frombuffer(blob, REC, offset=start, count=(len(blob) - start) // REC.itemsize)
    paths = [p.replace('\\', '/').lower() for p in paths]
    dims = np.zeros((len(paths), 2), np.int64)
    for i in np.unique(recs['p']):
        for name in ('mesh.rigid_model_v2', 'custom_mesh.rigid_model_v2'):
            if paths[int(i)] + name in packs.index:
                lo, hi, _ = mesh_group(packs.read(paths[int(i)] + name))
                dims[i] = round((hi[0] - lo[0]) / 128), round((hi[2] - lo[2]) / 128)
                break
    TH, TW = cw.TILE_H, cw.TILE_W
    pi = recs['p'].astype(np.int64)
    turn = (recs['f'] & 0xa0) > 0
    w, h = np.where(turn, dims[pi, 1], dims[pi, 0]), np.where(turn, dims[pi, 0], dims[pi, 1])
    x, y = recs['x'].astype(np.int64), recs['y'].astype(np.int64)
    x0, x1, y0, y1 = np.clip(x, 0, TW), np.clip(x + w, 0, TW), np.clip(y, 0, TH), np.clip(y + h, 0, TH)
    # footprint counts by 2D prefix sums of the corner marks (stage_invariants.sea_floor_coverage)
    n = (TH + 1) * (TW + 1)
    mark = np.bincount(np.concatenate((y0 * (TW + 1) + x0, y1 * (TW + 1) + x1)), minlength=n).astype(np.int32)
    mark -= np.bincount(np.concatenate((y0 * (TW + 1) + x1, y1 * (TW + 1) + x0)), minlength=n).astype(np.int32)
    holes = mark.reshape(TH + 1, TW + 1).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)[:TH, :TW] == 0
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, holes)
    return holes


def missing_tile_at(live, wx, wz):
    """True where world points fall in a tile cell without a fine tile (tile_holes of the live pack)."""
    from campaign_fine_tiles import CELL_X, CELL_Z
    if 'holes' not in live:
        live['holes'] = tile_holes(live['ground'].packs)
    holes = live['holes']
    tx, ty = np.floor(wx / CELL_X).astype(np.int64), np.floor(wz / CELL_Z).astype(np.int64)
    inside = (tx >= 0) & (tx < holes.shape[1]) & (ty >= 0) & (ty < holes.shape[0])
    out = np.zeros(np.shape(wx), bool)
    out[inside] = holes[ty[inside], tx[inside]]
    return out


class LiveShader(rv.Shader):
    """rv.Shader plus the per-layer normal/roughness/occlusion maps, blended like the base colour."""
    def __init__(self, ground, **kw):
        super().__init__(ground, **kw)
        self.nros = {}

    def frame(self, x0, z0, x1, z1, cells=4096):
        """Fix the water / land-mesh raster window for one view (rv.Shader.mesh_level re-rasterises whenever a
        point falls outside its window, which far horizon rays do at every step). The window is the view's own:
        reusing an earlier, larger window (coarser cells) made a shot depend on the shots rendered before it in
        the same process (the land-mesh ground and the water edges), so it is reused only when identical."""
        pad = .02 * max(x1 - x0, z1 - z0) + cw.DX
        x0, z0, x1, z1 = x0 - pad, z0 - pad, x1 + pad, z1 + pad
        step = max((x1 - x0) / cells, (z1 - z0) / cells, cw.DX / 16)
        nx, nz = int(np.ceil((x1 - x0) / step)), int(np.ceil((z1 - z0) / step))
        if self.mwin and self.mwin[:5] == (x0, z0, step, nx, nz):
            return
        self.mwin = (x0, z0, step, nx, nz) + self.meshes.fields(x0, z0, step, nx, nz)

    def mesh_level(self, wx, wz, cells=4096):
        """rv.Shader.mesh_level inside the framed window; no water outside it (no re-raster)."""
        w = self.mwin
        if w is None:
            return super().mesh_level(wx, wz, cells)
        lx, lz = (np.asarray(a, np.float64) for a in self.g.local(wx, wz))
        fi, fj = (lx - w[0]) / w[2], (lz - w[1]) / w[2]
        inside = (fi >= 0) & (fi < w[3]) & (fj >= 0) & (fj < w[4])
        i = np.clip(fi.astype(np.intp), 0, w[3] - 1)
        j = np.clip(fj.astype(np.intp), 0, w[4] - 1)
        water, land = w[5][j, i].astype(np.float64), w[6][j, i].astype(np.float64)
        water = np.where(inside, water, -np.inf)
        hole = np.isfinite(water) & ~np.isfinite(land)
        if hole.any():
            land[hole] = self.g.heights(np.asarray(wx, np.float64)[hole], np.asarray(wz, np.float64)[hole])
        return np.where(water > land, water, -np.inf)

    def nro(self, mip):
        if mip not in self.nros:
            self.nros[mip] = nro_layers(self.g.packs, mip, self.season)
        return self.nros[mip]

    def nro_at(self, wx, wz, footprint):
        """(tangent normal x, y in -1..1, occlusion 0..1) at world points (shader corner blend)."""
        corners, fx, fy = self.g.blend_corners(wx, wz)
        mip = int(np.clip(np.round(np.log2(max(footprint, 1e-6) / self.tile * 1024)), 1, 9))
        tex = self.nro(mip)
        S = tex.shape[1]
        u = (np.mod(wx / self.tile, 1) * S).astype(int) % S
        v = (np.mod(-wz / self.tile, 1) * S).astype(int) % S
        w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
        acc = 0
        for c, wi in zip(corners, w):
            acc = acc + tex[np.clip(c[..., 0], 0, tex.shape[0] - 1), v, u].astype(np.float32) * wi[..., None]
        acc /= 255
        return acc[..., 3] * 2 - 1, acc[..., 1] * 2 - 1, acc[..., 2]


def live_state(pack, P):
    key = (pack, 'live')
    if key not in _STATE:
        g = state(pack)['ground']
        _STATE[key] = dict(ground=g, shader=pack_shader(g, LiveShader))
    return _STATE[key]


def march(surface, eye, d, hmin, hmax, steps=160, refine=8):
    """First hit of rays eye + t d with a height field surface(x, z): (found, t). Uniform steps between the
    hmax and hmin planes, then bisection (render_campaign_view.render_oblique's scheme)."""
    if getattr(surface, 'gpu', None) is not None:     # render_gpu.Surface
        return surface.gpu.march(eye, d, hmin, hmax, steps, refine)
    H, W = d.shape[:2]
    dy = d[..., 1]
    down = dy < -1e-4
    sdy = -np.where(down, dy, -1)
    t0 = np.maximum(np.where(down, (eye[1] - hmax) / sdy, np.nan), 0)
    t1 = np.where(down, (eye[1] - hmin) / sdy, np.nan)
    found = np.zeros((H, W), bool)
    lo, hit, prev = t0.copy(), np.full((H, W), np.nan), t0.copy()
    for k in range(1, steps + 1):
        tk = t0 + (t1 - t0) * k / steps
        act = down & ~found
        if not act.any():
            break
        ta = tk[act]
        da = d[act]
        below = eye[1] + da[:, 1] * ta <= surface(eye[0] + da[:, 0] * ta, eye[2] + da[:, 2] * ta)
        idx = np.flatnonzero(act)[below]
        found.flat[idx] = True
        lo.flat[idx] = prev.flat[idx]
        hit.flat[idx] = tk.flat[idx]
        prev[act] = ta
    a, b, dd = lo[found], hit[found], d[found]
    for _ in range(refine):
        m = (a + b) / 2
        below = eye[1] + dd[:, 1] * m <= surface(eye[0] + dd[:, 0] * m, eye[2] + dd[:, 2] * m)
        b = np.where(below, m, b)
        a = np.where(below, a, m)
    t = np.full((H, W), np.nan)
    t[found] = b
    return found, t


def sun_vector(P):
    az, alt = map(math.radians, (P['sun_az'], P['sun_alt']))
    return np.array([math.sin(az) * math.cos(alt), math.sin(alt), math.cos(az) * math.cos(alt)])


def terrain_ao(g, wx, wz, h, radii):
    """Horizon occlusion: mean rise of the ground over 8 directions at each radius (0 open .. 1 buried)."""
    if getattr(g, 'gpu', None) is not None:
        return g.gpu.ao(wx, wz, h, radii)
    occ = 0
    n = 0
    for r in radii:
        for a in np.arange(8) * np.pi / 4:
            hi = g.heights(wx + r * np.cos(a), wz + r * np.sin(a))
            occ = occ + np.clip((hi - h) / r, 0, 1)
            n += 1
    return occ / n


def sun_shadow(surface, wx, wz, h, s, length=12., steps=20):
    """1 lit .. 0 shadowed: does the ground rise above the ray toward the sun within `length` world units."""
    if getattr(surface, 'gpu', None) is not None:
        return surface.gpu.shadow(wx, wz, h, s, length, steps, objects=surface.objects)
    lit = np.ones(np.shape(wx))
    for k in range(1, steps + 1):
        t = length * (k / steps) ** 1.5
        y = h + .02 + s[1] * t
        rise = surface(wx + s[0] * t, wz + s[2] * t) - y
        lit = np.minimum(lit, np.clip(1 - rise / (.05 + .02 * t), 0, 1))
    return lit


def ground_terms(sh, wx, wz, footprint, P, surface):
    """Light-independent terms of ground/water hit points: base colour, LF + normal-map normal, texture
    occlusion, terrain horizon occlusion, sun shadow (needs the sun direction), mesh water and its depth."""
    g = sh.g
    h = g.heights(wx, wz)
    n = sh.normals(wx, wz, max(footprint, P['normal_eps']))
    occ = np.ones_like(h)
    if P['nro']:
        tx, ty, to = sh.nro_at(wx, wz, footprint)
        n = n + P['nro'] * np.stack([tx, np.zeros_like(tx), -ty], -1) * n[..., 1:2]
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        occ = 1 - P['tex_occlusion'] * (1 - to)
    level = sh.water_level(wx, wz)
    return dict(albedo=sh.albedo(wx, wz, footprint), n=n, occ=occ, h=h,
                horizon=terrain_ao(g, wx, wz, h, P['ao_radii']) if P['ao'] else np.zeros_like(h),
                shadow=sun_shadow(surface, wx, wz, h, sun_vector(P)) if P['shadows'] else np.ones_like(h),
                wet=np.isfinite(level) if sh.meshes else h < level, depth=np.clip((level - h) / 1.5, 0, 1),
                rise=np.ones_like(h), canopy=np.zeros(h.shape, bool))


def gbuffer(live, cam, size, P):
    """Geometry pass of a live view: ray hits (terrain, mesh water, object impostors) and the per-hit terms
    finish() lights. Everything that depends only on the grade/light strengths is left to finish()."""
    sh, g = live['shader'], live['ground']
    x, z, dist = cam[:3]
    reach = dist * 6
    if BACKEND == 'gpu':      # the water / land mesh raster of the window on the GPU (render_gpu.ZFill, equal grids)
        import campaign_water_meshes as cwm, render_gpu
        cwm.GPU_FILL = render_gpu.ZFill()
        try:
            sh.frame(x - reach, z - reach, x + reach, z + reach)
        finally:
            cwm.GPU_FILL = None
    else:
        sh.frame(x - reach, z - reach, x + reach, z + reach)
    # ground heights from the land mesh (what the game draws) or the LF; set before the camera, whose eye height is
    # taken over the target ground
    g.mesh = sh.mwin[:5] + (sh.mwin[6],) if P['ground'] == 'mesh' else None
    if BACKEND == 'gpu':
        import render_gpu
        dev = g.gpu = render_gpu.Device(g)
        try:
            dev.sync(sh)
            return _gbuffer(live, cam, size, P, sh, g, render_gpu.Surface(sh.surface, dev), dev)
        finally:
            g.gpu = None
            dev.free()
    return _gbuffer(live, cam, size, P, sh, g, sh.surface, None)


def _gbuffer(live, cam, size, P, sh, g, ground_surface, dev):
    x, z, dist = cam[:3]
    W, H = size
    reach = dist * 6
    eye, d, tf, target = camera_rays(g, cam, size, P)
    GX, GZ = np.meshgrid(np.linspace(x - reach, x + reach, 256), np.linspace(z - reach, z + reach, 256))
    hh = sh.surface(GX, GZ)
    found, t = march(ground_surface, eye, d, float(hh.min()) - .2, float(hh.max()) + .2)
    surface, ov = ground_surface, None
    if (P['trees'] or P['props']) and found.any():
        ov = object_overlay(live, eye, d, found, t, P)
        if np.isfinite(ov.hmax):
            from render_live_objects import march_objects
            surface = lambda wx, wz: np.maximum(sh.surface(wx, wz), ov.height(wx, wz))
            if dev is not None:
                import render_gpu
                dev.set_overlay(ov)
                surface = render_gpu.Surface(surface, dev, objects=True)
            t = march_objects(surface, eye, d, found, t, ov.hrel)
        else:
            ov = None
    tt = t[found]
    dd = d[found]
    px, pz = eye[0] + dd[:, 0] * tt, eye[2] + dd[:, 2] * tt
    foot = float(np.median(tt * 2 * tf / H / np.clip(-dd[:, 1], .2, 1))) if found.any() else 1.
    G = dict(size=size, found=found, t=t, tt=tt, dist=dist, target=target,
             dy=d[..., 1].astype(np.float32), eye_y=float(eye[1]), base_y=float(eye[1] - cam[4] * P['height_scale']))
    # ground (and water) hits in a cell without a fine tile, not object tops; painted if missing_tiles
    miss = missing_tile_at(live, px, pz)
    if ov is not None and miss.any():
        top = ov.height(px[miss], pz[miss])
        miss[np.flatnonzero(miss)[np.isfinite(top) & (top >= sh.surface(px[miss], pz[miss]) + 1e-3)]] = False
    G['missing'] = miss
    if not P['lighting']:   # render_campaign_view's hill shading (step-1 look), objects in its light
        col, _ = sh.shade(px, pz, foot)
        G['flat'] = object_flat(sh, ov, px, pz, col, P) if ov is not None else col
        return G
    T = ground_terms(sh, px, pz, foot, P, surface)
    if ov is not None and P['town_decal'] and getattr(live['objects'], 'decals', None):   # town ground decals
        w, c = live['objects'].decal_tint(live['objects'].decals, px, pz, P['town_decal'])
        T['albedo'] = T['albedo'] * (1 - w[:, None]) + c * w[:, None]
    if ov is not None:
        object_terms(sh, ov, px, pz, T, P, surface)
    G.update(T)
    return G


def finish(G, P):
    """Light, water, aerial perspective, grade and depth of field of a gbuffer -> PIL image."""
    from PIL import Image
    W, H = G['size']
    found, tt, target = G['found'], G['tt'], G['target']
    sky = np.array(P['sky'], np.float64)
    img = np.zeros((H, W, 3)) + sky
    if 'flat' in G:
        col = G['flat']
    else:
        s = sun_vector(P)
        n = G['n']
        lam = np.clip((n * s).sum(-1), 0, 1)
        lit = P['shadow_soft'] + (1 - P['shadow_soft']) * G['shadow']
        ao = np.clip(1 - P['ao'] * 1.5 * G['horizon'], 0, 1) * (1 - P['object_ao'] * (1 - G['rise']))
        amb = P['ambient'] * (.75 + .25 * n[..., 1]) * ao * G['occ']
        light = amb[..., None] * np.array(P['sky_tint']) + (P['sun'] * lam * lit)[..., None] * np.array(P['sun_tint'])
        tint = P['tree_tint_texture'] if P['tree_colour'] == 'texture' else P['tree_tint']
        col = G['albedo'] * np.where(G['canopy'][:, None], np.array(tint), 1.) * light
        depth = G['depth'][..., None]
        ws, wd = np.array(P['water_shallow'], np.float64) / 255, np.array(P['water_deep'], np.float64) / 255
        wcol = (ws * (1 - depth) + wd * depth) * (P['ambient'] + P['sun'] * s[1])
        col = np.where(G['wet'][..., None], wcol, col)
    if P['hfog']:
        pass                # height fog after the grade (height_fog)
    elif P['fog_density']:
        fd = 1 - np.exp(-np.maximum(tt - P['fog_start'] * target, 0) * P['fog_density'] / target)
        col = col * (1 - fd[:, None]) + np.array(P['haze']) * fd[:, None]
    elif 'flat' in G:   # baseline haze (render_campaign_view)
        hz = np.clip((tt - G['dist']) / (G['dist'] * 6), 0, .6)[:, None]
        col = col * (1 - hz) + sky * hz
    img[found] = col
    if 'flat' not in G or P['fog_density']:
        img[~found] = np.array(P['haze'])
    img = grade(img, P)
    if P['hfog']:
        img = height_fog(img, G, P)
    m = np.zeros(found.shape, bool)
    m[found] = G['missing']
    if P['missing_tiles'] is not None and m.any():   # defect colour: no light, fog or grade
        img[m] = np.array(P['missing_tiles'], np.float64)
    if P['dof']:
        img = depth_of_field(img, np.where(found, G['t'], 1e9), P['dof_focus'] * target, P['dof'])
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    out.info['missing_share'] = float(m.mean())   # share of the view on cells without a fine tile
    return out


def height_fog(img, G, P):
    """Exponential height fog along each ray (display colours, after the grade) and the sky gradient.
    Optical depth = density * integral over the ray, from hfog_start x the eye-target distance to the hit, of
    exp(-falloff (y - base)), base = the ground under the camera target; world units throughout."""
    found, dy, tt = G['found'], G['dy'], G['tt']
    fogc = np.array(P['hfog_colour'], np.float64)
    a = dy[found].astype(np.float64)
    k = P['hfog_falloff']
    s0 = np.minimum(P['hfog_start'] * G['target'], tt)
    e0 = math.exp(-k * (G['eye_y'] - G['base_y']))
    kd = k * a
    small = np.abs(kd) < 1e-6
    integ = np.where(small, tt - s0, (np.exp(-kd * s0) - np.exp(-kd * tt)) / np.where(small, 1., kd)) * e0
    fd = np.minimum(1 - np.exp(-P['hfog_density'] * np.maximum(integ, 0)), P['hfog_max'])
    img = img.copy()
    img[found] = img[found] * (1 - fd[:, None]) + fogc * fd[:, None]
    up = np.clip(dy[~found] / P['sky_span'], 0, 1)[:, None]
    img[~found] = fogc * (1 - up) + np.array(P['sky_zenith'], np.float64) * up
    return img


def render_live(live, cam, size, P):
    return finish(gbuffer(live, cam, size, P), P)


def grade(rgb, P):
    """Exposure, per-channel gain, saturation, contrast about mid grey, gamma (display sRGB 0..1)."""
    rgb = np.clip(rgb * P['exposure'] * np.array(P['gain']), 0, None)
    lum = (rgb * np.array([.2126, .7152, .0722])).sum(-1, keepdims=True)
    rgb = lum + (rgb - lum) * P['saturation']
    rgb = .5 + (rgb - .5) * P['contrast']
    return np.clip(rgb, 0, 1) ** (1 / P['gamma'])


def depth_of_field(img, t, focus, strength):
    """Blur by circle of confusion |1 - focus / t| (px = strength * coc at this width / 1280)."""
    from scipy.ndimage import gaussian_filter
    H, W = t.shape
    coc = np.clip(np.abs(1 - focus / np.where(np.isfinite(t), t, 1e9)), 0, 1.5) * strength * W / 1280
    levels = [0, .75, 1.5, 3, 6]
    S = np.stack([img] + [np.stack([gaussian_filter(img[..., c], s) for c in range(3)], -1) for s in levels[1:]])
    idx = np.interp(coc, levels, np.arange(len(levels)))
    i0 = np.clip(np.floor(idx).astype(int), 0, len(levels) - 2)
    f = (idx - i0)[..., None]
    jj, ii = np.mgrid[:H, :W]
    return S[i0, jj, ii] * (1 - f) + S[i0 + 1, jj, ii] * f


def camera_rays(g, cam, size, P):
    """eye, ray directions (H, W, 3), tan(fov / 2) and the eye-target distance of a live camera."""
    x, z, dist, heading, height = cam
    height *= P['height_scale']
    W, H = size
    eye, f, right, up = rv.camera_basis(x, z, dist, heading, height, g, 'horizontal')
    tf = math.tan(math.radians(P['fov']) / 2)
    jj, ii = np.mgrid[:H, :W].astype(np.float64)
    sx = (2 * (ii + .5) / W - 1) * tf * W / H
    sy = (1 - 2 * (jj + .5) / H) * tf
    d = f[None, None] + sx[..., None] * right + sy[..., None] * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return eye, d, tf, math.hypot(dist, height)


# ------------------------------------------------------------------ objects
def object_overlay(live, eye, d, found, t, P):
    """Object height field over the ground the first march reached (trees/props of render_live_objects)."""
    from render_live_objects import Objects, Overlay
    if 'objects' not in live or live['objects'].town_mode != P['towns']:
        live['objects'] = Objects(live['ground'], towns=P['towns'])
    hx = eye[0] + d[..., 0][found] * t[found]
    hz = eye[2] + d[..., 2][found] * t[found]
    pad = 3.
    ov = Overlay(live['objects'], hx.min() - pad, hz.min() - pad, hx.max() + pad, hz.max() + pad, P)
    if np.isfinite(ov.hmax):
        jj, ii = np.nonzero(np.isfinite(ov.top))
        k = np.linspace(0, len(jj) - 1, min(len(jj), 200000)).astype(int)
        gx, gz = ov.x0 + (ii[k] + .5) * ov.cell, ov.z0 + (jj[k] + .5) * ov.cell
        ov.hrel = float(np.max(ov.top[jj[k], ii[k]] - live['shader'].surface(gx, gz))) + .05
    return ov


def _on_objects(sh, ov, wx, wz):
    """(mask of hit points on an object top, cell i, j, object top, ground) at those points."""
    ground = sh.surface(wx, wz)
    top = ov.height(wx, wz)
    on = np.isfinite(top) & (top >= ground + 1e-3)
    i, j, _ = ov.idx(wx[on], wz[on])
    if not hasattr(ov, 'n'):
        ov.n = ov.normals()
    return on, i, j, top, ground


def object_terms(sh, ov, wx, wz, T, P, surface):
    """Overwrite the ground terms T on object hits: object colour and top-field normal, sun shadow from
    terrain and objects, ambient darkened toward the object's foot (rise), no water."""
    on, i, j, top, ground = _on_objects(sh, ov, wx, wz)
    if not on.any():
        return
    T['albedo'][on] = ov.col[j, i]
    T['n'][on] = ov.n[j, i]
    T['occ'][on] = 1
    T['horizon'][on] = 0
    T['wet'][on] = False
    T['rise'][on] = np.clip((top[on] - ground[on]) / .5, 0, 1)
    T['canopy'][on] = ov.tree[j, i]
    if P['shadows']:
        T['shadow'][on] = sun_shadow(surface, wx[on], wz[on], top[on], sun_vector(P))


def object_flat(sh, ov, wx, wz, col, P):
    """Objects in render_campaign_view's hill-shading light (lighting off)."""
    on, i, j, top, ground = _on_objects(sh, ov, wx, wz)
    if not on.any():
        return col
    lam = np.clip((ov.n[j, i] * sh.sun).sum(-1), 0, 1)
    rise = np.clip((top[on] - ground[on]) / .5, 0, 1)
    light = np.clip(.45 + .75 * lam / sh.sun[1] * .75, .25, 1.35) * (1 - P['object_ao'] * (1 - rise))
    col = col.copy()
    col[on] = ov.col[j, i] * light[:, None]
    return col

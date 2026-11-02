"""Snow bake (analysis/snow-bake-design.md): the campaign snow mask, year-round peaks.

The game's snow is a deferred post-process reading one RGBA mask over the whole map's world bounds; the season's
snowing intensity picks the channel (R < .1, G .1-.5 summer / harvest, B .5-.7 spring / autumn, A > .7 winter).
The mask this bake draws:
  native rectangle  vanilla 3k_snowmask's A (winter) channel only, reprojected, max the peaks (vanilla hung the
                    snow post-process in its winter groups alone, so its R / G / B were never shown and are dropped);
  peaks             biome_profiles.perennial_snow >= its weight, all four channels (white every season: this
                    project's design, user 2026-10-23, not vanilla behaviour);
  canvas land       A = winter_snow; R / G / B carry the peaks alone (shoulder_snow is not drawn); no snow on water.
The shader is a switch, not a shade (analysis/snow-v22b-findings.md §3, snow_emulate): outside the native core every
byte, on every mip level, is 0 or 255; winter_snow is a coverage share, dithered by the rank of the fbm field below
(invariant snow_mask_binary).
Rule data and predicates are biome_profiles' (CONSTS snow_bake, COLD_LINES PERENNIAL_SNOW); the fuzz is the ground's
fbm field (classify: scale 9, 3 octaves, seed 211), so snow edges wobble as the ground does.

  python snow_bake.py                  the products into the stage: the mask (MASK_KEY), the material copy
                                       (MATERIAL_KEY), the 25 lighting groups (campaign_world.SNOW_LIGHTING_DIR), and
                                       analysis/snow-bake.json; pipeline graph node 'snow', after the LF's last
                                       writer (sync_final); never in the sync_final side lane (side_lane_lf)
  python snow_bake.py --preview DIR    mask + channel previews + stats into DIR (writes nothing in the workspace)
"""
import argparse, json, re, struct, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_world as cw
import biome_profiles as bp

VANILLA_MASK = 'terrain/textures/campaign/default/snow/3k_snowmask.dds'
MASK_KEY = 'terrain/textures/campaign/default/snow/tw3k_exp_snowmask.dds'
PX_PER_UNIT = 1.0                       # vanilla 512 px over 595 world units: 0.86
OLD = (595.0999755859375, 541.2066650390625)   # vanilla g_world_bounds.zw
FBM = (9, 3, 211)                        # biome_profiles.classify's fuzz field
PEAK_SET = .5                            # perennial_snow weight of the peak set (design 3.2)
RING = 24                                # px (world units) each side of the native edge: vanilla A <-> canvas A
RING_WOBBLE = 12                         # px: fbm shift of the ring distance
TILE_KM = 2.2                            # km per tile cell (.0195 deg of latitude a row): winter_snow_coverage's reference blurs


def world_size():
    """Canvas g_world_bounds.zw (engine_bounds of expand_campaign_water_colour)."""
    return (float(np.float32(cw.WORLD_WIDTH)),
            float(np.float32(cw.TILE_H * cw.WORLD_WIDTH / cw.TILE_W * cw.ENGINE_Z_RATIO)))


def mask_shape():
    w, h = world_size()
    return int(round(h * PX_PER_UNIT)), int(round(w * PX_PER_UNIT))


def vanilla_mask():
    """Vanilla 3k_snowmask as RGBA uint8 (512, 512, 4) and its 148-byte DX10 header."""
    from repair_campaign_river_roots import Assets
    d = Assets(original_only=True).read(VANILLA_MASK)
    assert d[84:88] == b'DX10' and struct.unpack_from('<I', d, 128)[0] == 91, 'vanilla mask is not B8G8R8A8_UNORM_SRGB'
    bgra = np.frombuffer(d, np.uint8, 512 * 512 * 4, 148).reshape(512, 512, 4)
    return bgra[..., [2, 1, 0, 3]].copy(), d[:148]


def reproject(rgba, shape):
    """Vanilla uv (x / OLD.x, 1 - z / OLD.y) onto the canvas mask ((x + sx) / NEW.x, 1 - (z + sz) / NEW.y)."""
    nh, nw = shape; oh, ow = rgba.shape[:2]; new = world_size(); sx, sz = cw.NATIVE_SHIFT_WORLD
    ax, ay = new[0] / nw * ow / OLD[0], new[1] / nh * oh / OLD[1]
    bx, by = -sx * ow / OLD[0], (OLD[1] - new[1] + sz) * oh / OLD[1]
    return np.stack([np.asarray(Image.fromarray(rgba[..., i]).transform((nw, nh), Image.Transform.AFFINE,
                                                                       (ax, 0, bx, 0, ay, by), Image.Resampling.BILINEAR,
                                                                       fillcolor=0)) for i in range(4)], -1)


def native_box(shape):
    """bool mask pixels inside the native rectangle's world box."""
    nh, nw = shape; new = world_size(); sx, sz = cw.NATIVE_SHIFT_WORLD
    xs = (np.arange(nw) + .5) / nw * new[0]; zs = (1 - (np.arange(nh) + .5) / nh) * new[1]
    return ((zs >= sz) & (zs < sz + OLD[1]))[:, None] & ((xs >= sx) & (xs < sx + OLD[0]))[None, :]


def water_tiles(ws, lake=None):
    """bool tile grid (south-up): sea hexes of map.hex and the real-terrain lakes."""
    import map_hex, stage_invariants as si
    if lake is None:
        lake = np.load(ws / 'analysis/west-real-terrain.npz')['lake']
    cells = map_hex.read(ws / 'work/final-map/map.hex')['cells']; H, W = cells.shape[:2]
    r, q = si.tile_hex_grid(cw.TILE_H, cw.TILE_W, H, W)
    return ((cells[:, :, 0] & 3) == 1)[r, q] | lake


def read_mask(data):
    """RGBA uint8 top level of a snow mask DDS (B8G8R8A8), (h, w, mips)."""
    h, w = struct.unpack_from('<2I', data, 12)
    assert data[84:88] == b'DX10' and struct.unpack_from('<I', data, 128)[0] == 91, 'snow mask is not B8G8R8A8_UNORM_SRGB'
    return np.frombuffer(data, np.uint8, h * w * 4, 148).reshape(h, w, 4)[..., [2, 1, 0, 3]], struct.unpack_from('<I', data, 28)[0]


def read_levels(data):
    """Every mip level of a snow mask DDS as RGBA uint8, top first."""
    h, w = struct.unpack_from('<2I', data, 12); n = struct.unpack_from('<I', data, 28)[0]
    out, off = [], 148
    for _ in range(n):
        out.append(np.frombuffer(data, np.uint8, h * w * 4, off).reshape(h, w, 4)[..., [2, 1, 0, 3]])
        off += h * w * 4; h, w = max(1, h // 2), max(1, w // 2)
    return out


def tile_fields(ws):
    """Peak, B and A weights (0..1) and water on the tile grid (south-up), from the real terrain (ETOPO metres,
    lon/lat per tile) and map.hex."""
    import gpu_noise
    z = np.load(ws / 'analysis/west-real-terrain.npz')
    x0, x1, y0, y1 = (int(v) for v in z['box']); TH, TW = cw.TILE_H, cw.TILE_W
    assert (x0, x1, y0, y1) == (0, TW, 0, TH), f'real terrain box {z["box"]} is not the whole tile grid'
    e, lon, lat, bio, names = z['elevation'], z['lon'], z['lat'], z['biome'], list(z['biome_names'])
    y, x = np.mgrid[:TH, :TW].astype(np.float32)
    fz = gpu_noise.fbm(x / 2, y / 2, tuple(cw.load_authored_coordinates()['noise_origin']), *FBM[:2], seed=FBM[2])
    dry = bp.dryness(bio, names, real_elev=e)
    # the regional mean elevation (snow_reference_elev, 100 km): the perennial line's dry-plateau raise, for the
    # winter coverage and the visible peaks alike
    el, er = bp.snow_reference_elev(e, cell_km=TILE_KM)
    peak = bp.perennial_snow(e, lat, lon, fz, e_region=er)
    b = bp.shoulder_snow(e, lat, lon, fz)
    if bp.snow_coverage_on():   # TW3K_SNOW_COVERAGE: the winter share of the ground white, for the binary mask below
        a = bp.winter_snow_coverage(e, lat, lon, fz, dry=dry, e_local=el, e_region=er)
    else:
        a = bp.winter_snow(e, lat, lon, fz, dry=dry)
    water = water_tiles(ws, z['lake'])
    return {'peak': peak, 'b': b, 'a': a, 'water': water, 'lon': lon, 'lat': lat, 'elev': e, 'fuzz': fz, 'e_region': er}


def lf_path(ws):
    return ws / 'work/release-input' / f'terrain/campaigns/{cw.TERRAIN_VARIANTS[0]}/lf_height_map.dds'


def lf_mask(ws, shape):
    """Stage LF world heights at the mask pixel centres (north-up): the frame of render_campaign_view.Ground.heights
    (LF column x / DX * LF_PX_PER_HEX, row z / DZ * LF_PX_PER_HEX south-up), bilinear, read row pair by row pair."""
    # Rows are read with seeks inside one with-block, never memory-mapped: no handle outlives the call (a mapped file
    # blocks a rewrite on Windows), and the bake runs after the LF's last writer (graph node 'snow', see the pipeline).
    p = lf_path(ws)
    with open(p, 'rb') as fh:
        h, w = struct.unpack_from('<2I', fh.read(128), 12)
        nh, nw = shape; W, H = world_size(); per = cw.LF_PX_PER_HEX
        u = np.clip((np.arange(nw) + .5) / nw * W / cw.DX * per, 0, w - 1)
        v = np.clip((1 - (np.arange(nh) + .5) / nh) * H / cw.DZ * per, 0, h - 1)
        rows = (h - 1) - v                               # file rows are north-first
        cols = np.arange(w, dtype=np.float64); out = np.empty(shape, np.float64)

        def row(r):
            fh.seek(128 + r * w * 2)
            return np.frombuffer(fh.read(w * 2), '<u2').astype(np.float64)
        for i, r in enumerate(rows):
            r0 = int(np.floor(r)); fr = r - r0; r1 = min(r0 + 1, h - 1)
            out[i] = np.interp(u, cols, row(r0) * (1 - fr) + row(r1) * fr)
    return cw.LEVEL + (out / 65535 - cw.NORM) * cw.GAIN


def summit_pixels(lat, lon, points, within=None):
    """{name: (row, col)} nearest mask pixel of each (name, lon, lat, ...) point; with `within` (degrees), only the
    points that close to a pixel (the grid covers them)."""
    out = {}
    for name, lo, la, *_ in points:
        d = (lon - lo) ** 2 * np.cos(np.radians(la)) ** 2 + (lat - la) ** 2
        i = np.argmin(d)
        if within is None or d.flat[i] <= within * within:
            out[name] = tuple(int(v) for v in np.unravel_index(i, d.shape))
    return out


# Visible-mountain peaks (user 2026-11-01; snow-v22b-findings §6): place follows the mountains the game draws, area
# follows the real ice. A mountain is an LF high standing PEAK_PROM world units over the low ground PEAK_R2 px around
# (max over PEAK_RB px - min over PEAK_R2 px). Around each pixel (a PEAK_RB px box) the share of real ground above the
# perennial line (ETOPO raised to the named-summit cones, each cone's apex stamped at its nearest pixel: sampled on the
# mask grid a cone loses its tip) is the share of that box that is white, taken from the top of the LF down (a pixel is
# white when the LF share above it, from the box's mean and spread, is under that share). So a range's white area is
# its real area above the line, drawn on its visible tops (a drop below the LF top whitened whole high plateaus: round
# 1, Pamir 5x its glaciers). Every named summit keeps its PEAK_NAMED_PX highest LF pixels within PEAK_WIN px white
# (the user's visibility floor; snow_peaks_visible).
PEAK_RB, PEAK_R2, PEAK_PROM = 12, 16, 1.5
PEAK_NAMED_PX = 20
# white share = K x the box's real share above the line: on the coarse mask the LF-top estimate undershoots (K 1:
# ranges .2-.6 of their real glaciers; K 2: 1.3-1.5x), 1.4 brings them near 1x (diversity's rule level is ~1:1)
PEAK_AREA_K = 1.4
# snow_peaks_visible: every named summit (perennial_summits + these) holds >= PEAK_MIN_PX white peak pixels within
# PEAK_WIN px; the exceptions sit in an LF lake basin in game (terrain item, not snow; coordinator 2026-11-01)
EXTRA_SUMMITS = (('Bogda', 88.31, 43.80, 5445), ('Muztagh Ata', 75.12, 38.28, 7509))
PEAK_EXCEPTIONS = {'Nyainqentanglha': 'LF lake basin (Namtso) at the summit', 'Kangrinboqe': 'LF lake basin (Manasarovar) at the summit'}
PEAK_MIN_PX, PEAK_WIN = 15, 15


def named_summits():
    return tuple((n, lo, la, h) for n, lo, la, h, _ in bp.REGIONS['perennial_summits']['points']) + EXTRA_SUMMITS


def visible_peaks(lf, e, lat, lon, e_region):
    """bool mask pixels: the white tops of the visible mountains (rule above). lf: world heights, e: real elevation
    (m), lat / lon: degrees, e_region: snow_reference_elev's regional mean (m; the dry-plateau raise of the perennial
    line), all on the mask grid."""
    from scipy import ndimage, special
    ec = np.maximum(e, bp.summit_elevation(lat, lon))
    for name, (r, c) in summit_pixels(lat, lon, bp.REGIONS['perennial_summits']['points'], within=.1).items():
        h = next(p[3] for p in bp.REGIONS['perennial_summits']['points'] if p[0] == name)
        ec[r, c] = max(ec[r, c], h)
    n = 2 * PEAK_RB + 1
    if e_region is None:   # without it the dry-plateau raise is silently off (Qiangtang ridges perennial: round 1)
        raise ValueError('visible_peaks needs e_region (snow_reference_elev regional mean)')
    line = bp.perennial_snowline_m(lat, lon, e_region=e_region)
    above = ndimage.uniform_filter(((ec >= line) & (e > 0)).astype(np.float64), n)
    prom = ndimage.maximum_filter(lf, n) - ndimage.minimum_filter(lf, 2 * PEAK_R2 + 1)
    mu = ndimage.uniform_filter(lf, n)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(lf * lf, n) - mu * mu, 1e-6))
    share_higher = special.ndtr(-(lf - mu) / sd)          # LF share of the box standing above this pixel
    cap = (e > 0) & (prom >= PEAK_PROM) & (above > 0) & (share_higher < PEAK_AREA_K * above)
    # The mean / spread estimate cannot rank under ~1 % (a box's top pixel reads ~.013): where a box holds real ground
    # above the line but the share rule whitened nothing in it, its single LF top is white (Altai: 40 px above the
    # line, .8-1.3 % per box, else no white at all). Ties broken by position (the LF saturates at its ceiling: a flat
    # top of equal pixels would all read as tops, Hengduan 7.6x its limit), and only where the rule left the box bare.
    tie = lf + np.arange(lf.size, dtype=np.float64).reshape(lf.shape) * 1e-12
    top = (tie >= ndimage.maximum_filter(tie, n)) & (ndimage.maximum_filter(cap, n) == 0)
    cap |= top & (e > 0) & (prom >= PEAK_PROM) & (above > 0)
    for name, (r, c) in summit_pixels(lat, lon, named_summits(), within=.1).items():
        r0, c0 = max(r - PEAK_WIN, 0), max(c - PEAK_WIN, 0)
        win = np.where(e[r0:r + PEAK_WIN + 1, c0:c + PEAK_WIN + 1] > 0, lf[r0:r + PEAK_WIN + 1, c0:c + PEAK_WIN + 1], -np.inf)
        top = np.argsort(win, axis=None)[::-1][:PEAK_NAMED_PX]
        cap[r0 + top // win.shape[1], c0 + top % win.shape[1]] = True
    return cap


def to_mask(field, shape):
    """Tile-grid field (south-up) -> mask pixels (north-up), area average."""
    nh, nw = shape
    return np.asarray(Image.fromarray(np.ascontiguousarray(field[::-1], np.float32), 'F').resize((nw, nh), Image.Resampling.BOX))


def bake(ws):
    """(rgba uint8 mask, vanilla header, stats)."""
    shape = mask_shape(); van, hdr = vanilla_mask(); nat = native_box(shape)
    f = tile_fields(ws)
    peak, b, a = (to_mask(f[k], shape) for k in ('peak', 'b', 'a'))
    dry_land = 1 - to_mask(f['water'].astype(np.float32), shape)
    # the peak set: the tops of the mountains the game draws (user 2026-11-01), on land (a switch, see below)
    pk = visible_peaks(lf_mask(ws, shape), to_mask(f['elev'], shape), to_mask(f['lat'], shape), to_mask(f['lon'], shape),
                       to_mask(f['e_region'], shape)) \
        & (dry_land > 0)
    # Only winter snow besides the peaks (user 2026-10-23): R, G, B carry the peaks alone; shoulder_snow (B) is not
    # drawn. Vanilla only ever showed A (the post-process sat in the winter groups), so its R / G / B are dropped
    # too: with snow in every season they would put snow on vanilla ground in summer.
    a = a * dry_land
    # native rectangle: vanilla winter snow, blended with the canvas rule over a ring RING px each side of its edge.
    # Outside, vanilla is its value at the nearest box pixel; the ring's distance wobbles by the fbm field (RING_WOBBLE
    # px) so the blend never reads as a line along the box edge (Liaodong: no vanilla snow, full canvas snow north of it)
    va = reproject(van, shape)[..., 3] / 255.
    from scipy import ndimage
    din = ndimage.distance_transform_edt(nat)
    dout, (iy, ix) = ndimage.distance_transform_edt(~nat, return_indices=True)
    # Inside, the wobble fades to 0 at RING px, so the blend never reaches deeper than RING (24 px: the native edge's
    # non-playable filler and the authored regions on it; vanilla playable land 0 px, 32 px of Liaodong at RING + wobble)
    wob = to_mask(f['fuzz'], shape) * RING_WOBBLE * np.where(nat, np.clip(1 - din / RING, 0, 1), 1)
    w = np.clip(.5 + (np.where(nat, din, -dout) + wob) / (2 * RING), 0, 1)   # 1 from RING px inside, 0 beyond RING px outside
    # extended edge values smoothed (no streaks), and off water: vanilla's own edge rows over the sea must not spread
    ve = np.where(nat, va, ndimage.gaussian_filter(va[iy, ix], RING / 3) * dry_land)
    a = w * ve + (1 - w) * a
    # The game's snow is a switch, not a shade (analysis/snow-v22b-findings.md §3: sRGB mask, smoothstep(.80, .85)):
    # a byte under 232 shows only where the shader's noise helps, under 170 never. So outside the native core every
    # byte is 0 or 255: the peaks are the peak set, and the winter weight is a coverage share, dithered by the
    # ground's fbm field (its rank, uniform 0..1: a share p snows on about p of the area, in fbm-shaped patches, no
    # grid). The core keeps vanilla's own bytes (snow_native_parity).
    core = native_core(shape)
    fz = to_mask(f['fuzz'], shape)
    a = np.where(core, np.maximum(a, pk), (a > dither_field(fz)) | pk)
    out = np.stack([pk, pk, pk, a], -1).astype(np.float64)
    rgba = np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)
    stats = {'shape': list(shape), 'px_per_unit': PX_PER_UNIT,
             'channels_nonzero': {c: int((rgba[..., i] > 0).sum()) for i, c in enumerate('RGBA')},
             'peak_px': int(pk.sum()), 'peak_px_native': int((pk & nat).sum()),
             'binary': 'outside the native core, every mip level (snow_mask_binary)',
             'winter_rule': 'winter_snow_coverage' if bp.snow_coverage_on() else 'winter_snow'}
    return rgba, hdr, stats, f, nat, (~core, fz)


def native_core(shape):
    """bool: the native rectangle deeper than RING px, which the canvas blend never reaches: vanilla's own bytes."""
    from scipy import ndimage
    return ndimage.distance_transform_edt(native_box(shape)) > RING


def dither_field(fz):
    """Rank of a field as 0..1 uniform thresholds (ties by position): a pixel snows when its share exceeds it."""
    r = np.empty(fz.size, np.float64)
    r[np.argsort(fz, axis=None, kind='stable')] = (np.arange(fz.size) + .5) / fz.size
    return r.reshape(fz.shape)


def _half(x, size):
    return np.asarray(Image.fromarray(np.ascontiguousarray(x, np.float32), 'F').resize(size, Image.Resampling.BOX))


def mip_levels(rgba, binary=None):
    """RGBA uint8 levels, top first, box filtered. binary = (bool region, fbm field) at the top level: inside the
    region every level is dithered back to 0 / 255 (the box mean of the top level as the share, the box-filtered field's
    rank as the threshold), so far views keep the snow cover instead of a grey the shader would drop."""
    levels = [rgba]; mean = rgba.astype(np.float32) / 255
    reg, fz = (binary[0].astype(np.float32), binary[1]) if binary is not None else (None, None)
    while levels[-1].shape[:2] != (1, 1):
        h, w = levels[-1].shape[:2]; size = (max(1, w // 2), max(1, h // 2))
        mean = np.stack([_half(mean[..., i], size) for i in range(4)], -1)
        lv = mean
        if reg is not None:
            reg, fz = _half(reg, size), _half(fz, size)
            lv = np.where((reg >= .5)[..., None], mean > dither_field(fz)[..., None], mean)
        levels.append(np.clip(np.rint(lv * 255), 0, 255).astype(np.uint8))
    return levels


def write_dds(rgba, header148, path, binary=None):
    """B8G8R8A8_UNORM_SRGB with a full mip chain (mip_levels), the vanilla header with this size."""
    h, w = rgba.shape[:2]
    levels = [lv[..., [2, 1, 0, 3]].tobytes() for lv in mip_levels(rgba, binary)]
    hdr = bytearray(header148)
    struct.pack_into('<5I', hdr, 8, 0x2100F, h, w, w * 4, 0)
    struct.pack_into('<I', hdr, 28, len(levels))
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(bytes(hdr) + b''.join(levels))


REGIONS = {   # lon0, lon1, lat0, lat1: peak-area check against the real glaciers (rule data review only)
    'pamir_karakoram': (70, 78, 34.5, 40), 'kunlun': (78, 97, 34, 37.5), 'tianshan': (70, 96, 40, 45.5),
    'himalaya': (78, 97, 26, 31), 'hengduan': (97, 104, 26, 33.5), 'qilian': (94, 104, 37, 40.5),
    'altai': (85, 92, 46, 51), 'inner_tibet': (80, 97, 31, 34)}


def preview(rgba, f, nat, out):
    out.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate('RGBA'):
        Image.fromarray(rgba[..., i]).save(out / f'snowmask-{c}.png')
    shape = rgba.shape[:2]
    e = to_mask(np.clip(f['elev'], 0, 7000) / 7000, shape)
    shade = np.dstack([e * 160 + 40] * 3)
    shade[to_mask(f['water'].astype(np.float32), shape) > .5] = (30, 50, 90)
    col = shade.copy()
    a, pk = rgba[..., 3] / 255., rgba[..., 1] / 255.     # A winter, G summer (the peaks: every season)
    col = col * (1 - a[..., None] * .6) + np.array([120, 170, 255]) * a[..., None] * .6      # winter only: blue
    col = col * (1 - pk[..., None]) + 255 * pk[..., None]                                     # all year: white
    edge = nat ^ np.roll(nat, 1, 0) | nat ^ np.roll(nat, 1, 1)
    col[edge] = (255, 60, 60)
    Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)).save(out / 'snowmask-overview.png')
    # peak km2 by region (tile cells: lon/lat area)
    lat, lon, peak = f['lat'], f['lon'], f['peak'] >= PEAK_SET
    km2 = (111.32 * np.cos(np.radians(lat))) * 111.32 * np.abs(np.gradient(lon, axis=1)) * np.abs(np.gradient(lat, axis=0))
    return {k: round(float(km2[peak & (lon >= a0) & (lon < a1) & (lat >= b0) & (lat < b1)].sum()))
            for k, (a0, a1, b0, b1) in REGIONS.items()}


VANILLA_MATERIAL = 'materials/environment/campaign/3k_main_campaign_snow.xml.material'
MATERIAL_KEY = 'materials/environment/campaign/tw3k_exp_campaign_snow.xml.material'
POST = re.compile(r"post_processes='([^']*)'")
RAIN = 'campagin_rain'   # vanilla's spelling


def material():
    """The vanilla snow material with only its mask and name changed."""
    from repair_campaign_river_roots import Assets
    m = Assets(original_only=True).read(VANILLA_MATERIAL).decode('utf-8')
    assert m.count(VANILLA_MASK) == 1 and m.count('<name>3k_main_campaign_snow.xml</name>') == 1, 'vanilla snow material changed'
    return m.replace(VANILLA_MASK, MASK_KEY).replace('<name>3k_main_campaign_snow.xml</name>', '<name>tw3k_exp_campaign_snow.xml</name>')


def lighting_groups():
    """{relative path: text} of the vanilla 3k_main_map lighting groups with the snow material in every entry.
    An entry holds one post-process (vanilla never lists two, and no separator is known): the spring rain entries keep
    their rain (peaks are not white in spring rain), the winter entries swap vanilla's snow for this one, every other
    entry takes it. Returns (groups, counts)."""
    from repair_campaign_river_roots import Assets
    a = Assets(original_only=True); pre = cw.VANILLA_LIGHTING_DIR + '/'
    keys = sorted(k for k in a.index if k.startswith(pre) and k.endswith('.environment_group'))
    assert len(keys) == 25, f'{len(keys)} vanilla lighting groups (25 expected)'
    out = {}; n = {'snow': 0, 'rain_kept': 0}
    def sub(m):
        v = m.group(1)
        if RAIN in v:
            n['rain_kept'] += 1; return m.group(0)
        assert v in ('', VANILLA_MATERIAL), f'unknown post-process {v!r}'
        n['snow'] += 1; return f"post_processes='{MATERIAL_KEY}'"
    for k in keys:
        out[k[len(pre):]] = POST.sub(sub, a.read(k).decode('utf-8'))
    return out, n


def fingerprint(ws):
    import hashlib
    h = {}
    for rel in ('work/final-map/map.hex', 'analysis/west-real-terrain.npz', str(lf_path(ws).relative_to(ws)).replace('\\', '/')):
        with open(ws / rel, 'rb') as fh:
            h[rel] = hashlib.file_digest(fh, 'sha256').hexdigest()
    here = Path(__file__).resolve().parent
    h['rule_code'] = hashlib.sha256(b''.join((here / n).read_bytes() for n in ('snow_bake.py', 'biome_profiles.py'))).hexdigest()
    return h


def clear_stage(ws):
    """Switch off (TW3K_SNOW_BAKE unset): remove this bake's own products from an earlier on-run, so the stage is as
    before the snow bake (translate_campaign_environment_spheres keeps the vanilla lighting references)."""
    import shutil
    S = ws / 'work/release-input'; gone = []
    for p in (S / MASK_KEY, S / MATERIAL_KEY, ws / 'analysis/snow-bake.json'):
        if p.is_file():
            p.unlink(); gone.append(str(p))
    if (S / cw.SNOW_LIGHTING_DIR).is_dir():
        shutil.rmtree(S / cw.SNOW_LIGHTING_DIR); gone.append(str(S / cw.SNOW_LIGHTING_DIR))
    return {'snow_bake': 'off (TW3K_SNOW_BAKE=1 turns it on)', 'removed': gone}


def write_stage(ws):
    """The bake's products into the release stage, plus analysis/snow-bake.json (inputs, counts)."""
    S = ws / 'work/release-input'
    if not bp.snow_coverage_on():   # stage_invariants.snow_switches stops the pipeline at its start already
        raise RuntimeError('TW3K_SNOW_BAKE=1 requires TW3K_SNOW_COVERAGE=1 (the binary mask reads winter snow as a coverage share)')
    rgba, hdr, stats, f, nat, binary = bake(ws)
    write_dds(rgba, hdr, S / MASK_KEY, binary)
    (S / MATERIAL_KEY).parent.mkdir(parents=True, exist_ok=True); (S / MATERIAL_KEY).write_text(material(), encoding='utf-8', newline='')
    groups, n = lighting_groups()
    for rel, t in groups.items():
        p = S / cw.SNOW_LIGHTING_DIR / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(t, encoding='utf-8', newline='')
    stats.update({'world_size': list(world_size()), 'ring_px': RING, 'peak_set': PEAK_SET, 'lighting_groups': len(groups),
                  'post_processes': n, 'inputs': fingerprint(ws)})
    (ws / 'analysis/snow-bake.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preview', type=Path, help='mask + previews + stats into this folder; the stage is not touched')
    o = ap.parse_args()
    if o.preview is None:
        print(json.dumps(write_stage(cw.WORKSPACE) if cw.snow_bake_on() else clear_stage(cw.WORKSPACE)))
        return
    rgba, hdr, stats, f, nat, binary = bake(cw.WORKSPACE)
    write_dds(rgba, hdr, o.preview / MASK_KEY.rsplit('/', 1)[-1], binary)
    stats['peak_km2_by_region'] = preview(rgba, f, nat, o.preview)
    (o.preview / 'snow-bake.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()

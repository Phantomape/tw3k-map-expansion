"""Ink-wash base art for the campaign strategic map (the far zoom-out parchment map).

rebuild_campaign_strategic_maps.py used to paint flat parchment shaded by the LF relief, so
land without relief showed nothing but borders. This paints what vanilla's hand-drawn map
shows, from data that covers every hex alike:
  * watercolour washes per biome (RESOLVE ecoregions at each hex's lon/lat): green for forest,
    yellow for desert and steppe, pale for alpine; forest washes are strong on hills and faint
    on cultivated plains, as on the vanilla map;
  * sepia brush strokes on shadowed slopes, stronger with height, light on lit slopes;
  * snow above the latitude snowline (real ETOPO metres), white salt flats (Natural Earth playas);
  * mottled water.

Two halves (the slow geodata build needs scipy and h5py; the pipeline reads its cache):
  * build the per-hex layers once (set OPENBLAS_NUM_THREADS=1, or scipy runs slower):
        <tw3k python> scripts/strategic_ink.py --build
    -> work/geodata/strategic-ink-<W>x<H>-<pad left>-<pad bottom>.npz (biome per hex, salt per hex);
  * base_art() loads that cache with numpy only and paints; a missing cache for the current
    canvas stops the rebuild with the command to run.
Pixel arrays are north-up (row 0 = north), as in the rebuild script; q, r give each pixel's hex.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import campaign_world as cw

GEODATA = Path(__file__).resolve().parents[1] / 'work/geodata'
PAPER = np.array([220., 208., 173.])
OCEAN = np.array([173., 193., 188.])
WASH = {   # biome -> (colour, strength on high ground, strength on plains)
    'forest': ((122, 148, 106), .62, .18), 'conifer': ((108, 136, 106), .66, .22), 'taiga': ((104, 130, 104), .66, .30),
    'tropical': ((106, 140, 98), .68, .30), 'mangrove': ((104, 132, 104), .60, .45), 'tropical_dry': ((150, 160, 112), .50, .20),
    'savanna': ((196, 190, 140), .40, .30), 'steppe': ((214, 204, 158), .45, .40), 'grass': ((188, 192, 142), .35, .20),
    'desert': ((226, 204, 152), .55, .55), 'alpine': ((192, 186, 152), .45, .40), 'tundra': ((198, 192, 168), .40, .40),
    'wetland': ((162, 182, 158), .45, .45), 'paddy': ((190, 196, 140), .25, .25), 'snow': ((244, 242, 236), .0, .0),
}
SALT = (236, 232, 220)
WASH_SOFTEN = 2.5     # hexes: biome edges blend instead of switching
STROKE_INK = np.array([92., 72., 52.])
THREADS = min(8, os.cpu_count() or 1)


def rows(fn, h, step=16):
    """Run fn(row slice) over bands of rows on threads (numpy releases the GIL); fn writes its band.
    Only for per-pixel work: every element sees the same operations as one whole-array pass."""
    with ThreadPoolExecutor(THREADS) as pool:
        return list(pool.map(fn, (slice(y, min(y + step, h)) for y in range(0, h, step))))


def separable(a, k):
    """Two-pass edge-padded filter of a 2-D float32 array with taps k (rows first, then columns),
    summed tap by tap from zero like the whole-array loop, in bands of rows on threads."""
    rad = len(k) // 2; h, w = a.shape; out = np.empty_like(a)
    def band(s):
        n = s.stop - s.start; v = a[np.clip(np.arange(s.start - rad, s.stop + rad), 0, h - 1)]; acc = np.zeros((n, w), a.dtype)
        for i, wt in enumerate(k): acc += wt * v[i:i + n]
        v = np.pad(acc, ((0, 0), (rad, rad)), mode='edge'); acc = np.zeros_like(acc)
        for i, wt in enumerate(k): acc += wt * v[:, i:i + w]
        out[s] = acc
    rows(band, h)
    return out


def cache_path():
    return GEODATA / f'strategic-ink-{cw.W}x{cw.H}-{cw.PAD_LEFT}-{cw.PAD_BOTTOM}.npz'


# The cache's provenance beyond the canvas: the code build() runs (these functions and every
# module-level name they use, transitively within their module, read as source text so load()
# stays numpy-only) and the size and write time of the geodata it reads (eco-grid-v2.npz is
# frontier_terrain's own raster cache of the ecoregions).
SOURCE_CODE = {'strategic_ink.py': ('build',),
               'frontier_terrain.py': ('pixel_lonlat', 'etopo', 'biomes', 'playas', 'fill_rings'),
               'plan_frontier_regions.py': ('place_array',),
               # place_array's fit and Western Regions projection live there (single-source audit, 2026-11-01)
               'campaign_world.py': ('lonlat_fit',), 'plan_western_regions.py': ('project',)}
SOURCE_DATA = ('ETOPO_2022_v1_60s_N90W180_surface.nc', 'ne_10m_playas.zip', 'Ecoregions2017.zip', 'eco-grid-v2.npz')


KEY_NAMES_ONLY = {'WASH'}   # the build only reads these dicts' keys; their values tune the parchment


def _code_closure(path, names):
    """Normalised AST of the module-level definitions names reach in path (functions, classes,
    assignments). ast.dump drops comments and layout and docstrings are removed, so only code
    that can change the cache counts; KEY_NAMES_ONLY dicts contribute their sorted keys."""
    import ast
    text = path.read_text(encoding='utf-8'); top = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for n in ast.walk(t):
                    if isinstance(n, ast.Name): top.setdefault(n.id, []).append(node)
    seen, todo = set(), list(names)
    while todo:
        n = todo.pop()
        if n in seen or n not in top: continue
        seen.add(n); todo += [m.id for node in top[n] for m in ast.walk(node) if isinstance(m, ast.Name)]
    def norm(n, node):
        if n in KEY_NAMES_ONLY and isinstance(getattr(node, 'value', None), ast.Dict):
            return n + '=' + repr(sorted(ast.literal_eval(k) for k in node.value.keys))
        for f in ast.walk(node):
            b = getattr(f, 'body', None)
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and b and isinstance(b[0], ast.Expr) \
                    and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                f.body = b[1:] or [ast.Pass()]
        return ast.dump(node)
    return '\n'.join(norm(n, node) for n in sorted(seen) for node in top[n])


def source_key():
    import hashlib, json
    here = Path(__file__).resolve().parent; h = hashlib.sha256()
    for name, funcs in SOURCE_CODE.items():
        h.update(name.encode() + b'|' + _code_closure(here / name, funcs).encode('utf-8') + b'\0')
    stats = {}
    for name in SOURCE_DATA:
        p = GEODATA / name
        stats[name] = [p.stat().st_size, p.stat().st_mtime_ns] if p.exists() else None
    h.update(json.dumps(stats, sort_keys=True).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- cache build (geodata stack)
def build():
    """Biome and salt per canvas hex from lon/lat (inverse projection), ETOPO and the ecoregions."""
    biome, names, salt, elev = compute()
    GEODATA.mkdir(parents=True, exist_ok=True)
    # Keyed after the build: eco_grid() may just have written eco-grid-v2.npz.
    np.savez_compressed(cache_path(), biome=biome, names=np.array(names), salt=salt, elev=elev,
                        canvas=np.array([cw.W, cw.H, cw.PAD_LEFT, cw.PAD_BOTTOM]), source_key=np.array(source_key()))
    print('wrote', cache_path(), {n: int((biome == i).sum()) for i, n in enumerate(names)}, 'salt hexes', int(salt.sum()))


def key_path():
    return cache_path().with_name(cache_path().name + '.key.json')


def stamp():
    """Compute the layers afresh; when they equal the cache's, record the source key beside it (the cache file is
    not rewritten, so its bytes and the geodata.json pin stay as they are)."""
    import json
    biome, names, salt, elev = compute()
    z = np.load(cache_path())
    same = (np.array_equal(z['biome'], biome) and [str(n) for n in z['names']] == list(names)
            and np.array_equal(z['salt'], salt) and np.array_equal(z['elev'], elev)
            and tuple(z['canvas']) == (cw.W, cw.H, cw.PAD_LEFT, cw.PAD_BOTTOM))
    if not same:
        raise SystemExit(f'{cache_path().name} differs from a fresh build: not stamped (rebuild it with --build)')
    key_path().write_text(json.dumps({'source_key': source_key(), 'by': 'stamp (layers equal)'}), encoding='utf-8')
    print('stamped', cache_path())


def compute():
    """(biome, names, salt, elev): the layers build() writes."""
    import frontier_terrain as ft
    import plan_frontier_regions as pf
    rr, qq = np.mgrid[0:cw.H, 0:cw.W]
    lon = np.full((cw.H, cw.W), np.nan); lat = np.full((cw.H, cw.W), np.nan)
    for r0 in range(0, cw.H, 256):     # row bands keep the triangulation small
        sl = slice(r0, min(r0 + 256, cw.H))
        lon[sl], lat[sl] = ft.pixel_lonlat(qq[sl].astype(np.float64), rr[sl] + (qq[sl] % 2) * .5)
    ok = np.isfinite(lon)
    e = np.zeros(lon.shape, np.float32)
    e[ok] = ft.etopo(lon[ok], lat[ok])
    names = sorted(WASH)
    biome = np.full(lon.shape, names.index('grass'), np.uint8)
    b = ft.biomes(lon[ok], lat[ok], e[ok], np.zeros(ok.sum()))
    biome[ok] = np.array([names.index(x) if x in names else names.index('grass') for x in b], np.uint8)
    salt = np.zeros((cw.H, cw.W), bool)
    hex_rings = []
    for part in (part for rec, parts in ft.playas() for part in parts):
        q, r = pf.place_array(part[:, 0], part[:, 1])
        if np.nanmax(q) >= 0 and np.nanmin(q) <= cw.W and np.nanmax(r) >= 0 and np.nanmin(r) <= cw.H:
            hex_rings.append(np.c_[q, r])
    if hex_rings:
        salt = ft.fill_rings(hex_rings, (cw.H, cw.W), lambda y: y, lambda x: x)
    from scipy import ndimage
    elev = ndimage.gaussian_filter(np.maximum(e, 0), 2.0).astype(np.float16)   # metres, softened over ~2 hexes
    return biome, names, salt, elev


def load():
    path = cache_path()
    if not path.exists():
        raise SystemExit(f'Strategic ink cache {path.name} is missing for canvas {cw.W}x{cw.H} padding '
                         f'{cw.PAD_LEFT}/{cw.PAD_BOTTOM}. Build it with: <tw3k python> scripts/strategic_ink.py --build')
    z = np.load(path)
    if tuple(z['canvas']) != (cw.W, cw.H, cw.PAD_LEFT, cw.PAD_BOTTOM):
        raise SystemExit(f'Strategic ink cache {path.name} does not match the canvas; rebuild it with --build.')
    if 'elev' not in z:
        raise SystemExit(f'Strategic ink cache {path.name} predates the elevation layer; rebuild it with --build.')
    import json
    key = str(z['source_key']) if 'source_key' in z else (
        json.loads(key_path().read_text(encoding='utf-8')).get('source_key') if key_path().exists() else None)
    if key is None:
        raise SystemExit(f'Strategic ink cache {path.name} has no source key, so it cannot be checked against the code '
                         'and geodata it was built from: rebuild it with --build, or verify it and record the key with '
                         '<tw3k python> scripts/strategic_ink.py --stamp')
    if key != source_key():
        raise SystemExit(f'Strategic ink cache {path.name} was built from other code or geodata (strategic_ink.build, '
                         f'frontier_terrain biomes/playas/projection, plan_frontier_regions.place_array or {", ".join(SOURCE_DATA)}). '
                         'Build it with: <tw3k python> scripts/strategic_ink.py --build')
    return z['biome'], [str(n) for n in z['names']], z['salt'], z['elev'].astype(np.float32)


# ---------------------------------------------------------------- painting (numpy + PIL only)
def blur(a, sigma):
    """Separable gaussian blur of a 2-D array in numpy (edge padding)."""
    rad = max(1, int(3 * sigma))
    k = np.exp(-np.arange(-rad, rad + 1, dtype=np.float32) ** 2 / (2 * sigma * sigma)); k /= k.sum()
    return separable(a.astype(np.float32), k)


def upsample(a, shape):
    return np.array(Image.fromarray(a.astype(np.float32), 'F').resize((shape[1], shape[0]), Image.Resampling.BILINEAR))


def base_art(physical, smooth, q, r):
    """Land/sea base art (float RGB, north-up) before boundaries, rivers and towns.
    physical: land mask per pixel; smooth: smoothed LF relief per pixel; q, r: hex per pixel."""
    h, w = physical.shape
    biome, names, salt_hex, elev = load()
    rng = np.random.default_rng(11)
    noise = rng.standard_normal((h // 4 + 1, w // 4 + 1)); fibre = rng.random((h // 2 + 1, w // 2 + 1))
    with ThreadPoolExecutor(2) as pool:   # PIL resizes release the GIL
        blot, texture = pool.map(lambda a: upsample(blur(*a), (h, w)), ((noise, 3), (fibre, .7)))
    blot /= blot.std() + 1e-9

    # washes: colour and strength per hex, softened over a hex or so, then read per pixel
    colour_hex = np.zeros(biome.shape + (3,), np.float32)
    hi_hex = np.zeros(biome.shape, np.float32); lo_hex = np.zeros(biome.shape, np.float32)
    for i, name in enumerate(names):
        c, hi, lo = WASH[name]
        m = biome == i
        colour_hex[m] = c; hi_hex[m] = hi; lo_hex[m] = lo
    colour_hex = np.stack([blur(colour_hex[..., c], WASH_SOFTEN) for c in range(3)], -1)
    hi_hex, lo_hex = blur(hi_hex, WASH_SOFTEN), blur(lo_hex, WASH_SOFTEN)
    # Hill weight from real elevation, not the LF relief: flat filler land and vanilla's own height
    # scale would otherwise switch the forest wash on and off in straight lines at their edges.
    hill = np.clip((elev - 150) / 1400, 0, 1)
    strength_hex = lo_hex + (hi_hex - lo_hex) * hill
    snow_id = names.index('snow'); xx = np.arange(w, dtype=np.float32)
    out = np.empty((h, w, 3))

    # Everything below is per pixel (the relief gradient reads one row either side), so it runs in
    # bands of rows on threads with the same operations, element for element, as whole arrays.
    def band(s):
        ph, rq = physical[s], (r[s], q[s]); bl = blot[s]; yy = np.arange(s.start, s.stop, dtype=np.float32)[:, None]
        grain = .9 * np.sin(xx * .071 + yy * .037) + .7 * np.sin(xx * .017 - yy * .043)
        img = np.where(ph[..., None], PAPER, OCEAN).astype(np.float32) + grain[..., None]
        strength = strength_hex[rq]
        strength = np.clip(strength * np.clip(1 + .35 * bl, .3, 1.6), 0, .85)[..., None] * ph[..., None]
        img = img * (1 - strength) + colour_hex[rq] * strength
        del strength

        a, b = max(s.start - 1, 0), min(s.stop + 1, h)
        gy, gx = (g[s.start - a:s.stop - a] for g in np.gradient(smooth[a:b]))
        lit = np.clip(gx * 2.5 + gy * 2.5, -1.5, 1.5)                     # light from the north-west (row 0 = north): NW faces rise to the SE
        mountain = np.clip((smooth[s] - cw.LEVEL - 2.0) / 5.0, 0, 1)
        stroke = np.clip(-lit, 0, None) * (.18 + .82 * mountain) * (.7 + .6 * texture[s])
        stroke = np.clip(stroke, 0, .8)[..., None] * ph[..., None]
        img = img * (1 - stroke) + STROKE_INK * stroke
        img += (np.clip(lit, 0, None) * 16 * mountain * ph)[..., None]

        pb = biome[rq]
        snow = ph & (pb == snow_id)
        img[snow] = img[snow] * .3 + np.array([244., 242., 236.]) * .7
        img[ph & salt_hex[rq]] = SALT
        water = ~ph
        img[water] += (3 * bl[water])[:, None]
        out[s] = img
    rows(band, h)
    return out


if __name__ == '__main__':
    if sys.argv[1:] == ['--build']:
        build()
    elif sys.argv[1:] == ['--stamp']:
        stamp()
    else:
        raise SystemExit('usage: strategic_ink.py --build | --stamp   (needs scipy and h5py: the tw3k Python, env/python-path.txt)')

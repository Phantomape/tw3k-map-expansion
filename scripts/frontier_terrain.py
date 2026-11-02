"""Real-geography terrain for the frontier ring (previews now, the terrain run later).

Elevation comes from ETOPO 2022 (60 arc-seconds), lakes and rivers from Natural Earth 10m,
all in work/geodata. Canvas pixels are mapped back to lon/lat by inverting
plan_frontier_regions.place_array numerically, so the frontier keeps exactly the projection the
towns use. Metres become relief units through a curve calibrated on vanilla's own height map
(binned medians of vanilla relief against ETOPO elevation at the same places), so new
mountains have vanilla's vertical exaggeration.
"""
import io, struct, zipfile
from functools import lru_cache

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import griddata

import campaign_world as cw
import plan_frontier_regions as pf
from analyze_reference import ROOT

GEODATA = ROOT / 'work/geodata'
ETOPO = GEODATA / 'ETOPO_2022_v1_60s_N90W180_surface.nc'


# ---------------------------------------------------------------- shapefiles
NUMERIC = {'scalerank', 'BIOME_NUM', 'ECO_ID'}


def read_shapes(zip_path, fields=('name', 'scalerank', 'featurecla')):
    """[(attrs, [part arrays of (lon, lat)])] for polygon or polyline shapefiles inside a zip."""
    with zipfile.ZipFile(zip_path) as z:
        shp = z.read(next(n for n in z.namelist() if n.endswith('.shp')))
        dbf = z.read(next(n for n in z.namelist() if n.endswith('.dbf')))
    count, header, reclen = struct.unpack_from('<IHH', dbf, 4)
    cols, pos, at = [], 32, 1
    while dbf[pos] != 0x0D:
        cols.append((dbf[pos:pos + 11].split(b'\0')[0].decode(), at, dbf[pos + 16]))
        at += dbf[pos + 16]; pos += 32
    want = [c for c in cols if c[0] in fields]
    records = []
    for i in range(count):
        row = dbf[header + i * reclen: header + (i + 1) * reclen]
        rec = {}
        for name, off, ln in want:
            v = row[off:off + ln].decode('utf-8', 'replace').strip()
            if name in NUMERIC:
                try:
                    v = float(v)
                except ValueError:
                    v = 99.
            rec[name] = v
        records.append(rec)
    out, pos, i = [], 100, 0
    while pos < len(shp):
        _, length = struct.unpack_from('>2I', shp, pos); pos += 8
        body = shp[pos:pos + length * 2]; pos += length * 2
        kind = struct.unpack_from('<I', body)[0]
        if kind in (3, 5):
            nparts, npoints = struct.unpack_from('<2I', body, 36)
            starts = list(struct.unpack_from('<' + 'I' * nparts, body, 44)) + [npoints]
            pts = np.frombuffer(body, '<f8', count=npoints * 2, offset=44 + 4 * nparts).reshape(npoints, 2)
            out.append((records[i], [pts[a:b] for a, b in zip(starts, starts[1:])]))
        i += 1
    return out


@lru_cache(None)
def lakes():
    return read_shapes(GEODATA / 'ne_10m_lakes.zip')


@lru_cache(None)
def playas():
    return read_shapes(GEODATA / 'ne_10m_playas.zip')


@lru_cache(None)
def rivers():
    return read_shapes(GEODATA / 'ne_10m_rivers_lake_centerlines.zip')


def fill_rings(rings, shape, to_row, to_col):
    """Even-odd scanline fill of lon/lat rings on a (rows, cols) grid. Exact for rings that reach far
    outside the grid (Eurasia), where PIL's polygon fill goes wrong. to_row/to_col map lon/lat arrays."""
    h, w = shape
    grid = np.zeros(shape, bool)
    rows, xs = [], []
    for ring in rings:
        y = to_row(ring[:, 1]); x = to_col(ring[:, 0])
        y0, y1, x0, x1 = y[:-1], y[1:], x[:-1], x[1:]
        lo = np.ceil(np.minimum(y0, y1) - .5).astype(np.int64)
        hi = np.ceil(np.maximum(y0, y1) - .5).astype(np.int64)      # rows whose centre lies in [min, max)
        lo, hi = np.clip(lo, 0, h), np.clip(hi, 0, h)
        n = hi - lo
        keep = n > 0
        if not keep.any():
            continue
        e = np.repeat(np.flatnonzero(keep), n[keep])
        row = np.repeat(lo[keep], n[keep]) + (np.arange(e.size) - np.repeat(np.cumsum(n[keep]) - n[keep], n[keep]))
        t = (row + .5 - y0[e]) / (y1[e] - y0[e])
        rows.append(row); xs.append(x0[e] + t * (x1[e] - x0[e]))
    if not rows:
        return grid
    row, x = np.concatenate(rows), np.concatenate(xs)
    order = np.lexsort((x, row)); row, x = row[order], x[order]
    a, b = np.arange(0, row.size - 1, 2), np.arange(1, row.size, 2)
    ok = row[a] == row[b]
    for r_, xa, xb in zip(row[a][ok], x[a][ok], x[b][ok]):
        c0, c1 = int(np.clip(np.ceil(xa - .5), 0, w)), int(np.clip(np.ceil(xb - .5), 0, w))
        grid[r_, c0:c1] = True
    return grid


# ---------------------------------------------------------------- inverse projection
def pixel_lonlat(q, r):
    """lon, lat of canvas hex positions (float arrays) by inverting place_array on a 0.05 degree grid."""
    qmin, qmax, rmin, rmax = q.min(), q.max(), r.min(), r.max()
    lon, lat = np.meshgrid(np.arange(30, 160, .25), np.arange(-12, 84, .25))
    cq, cr = pf.place_array(lon, lat)
    hit = (cq > qmin - 40) & (cq < qmax + 40) & (cr > rmin - 40) & (cr < rmax + 40)
    lon0, lon1, lat0, lat1 = lon[hit].min() - .5, lon[hit].max() + .5, lat[hit].min() - .5, lat[hit].max() + .5
    lon, lat = np.meshgrid(np.arange(lon0, lon1, .05), np.arange(lat0, lat1, .05))
    cq, cr = pf.place_array(lon, lat)
    pts = np.c_[cq.ravel(), cr.ravel()]
    out_lon = griddata(pts, lon.ravel(), (q, r), 'linear')
    out_lat = griddata(pts, lat.ravel(), (q, r), 'linear')
    return out_lon, out_lat


def etopo(lon, lat):
    """Bilinear ETOPO elevation (m) at lon/lat arrays."""
    with h5py.File(ETOPO, 'r') as f:
        la, lo = f['lat'], f['lon']
        step = 1 / 60
        i0 = max(int((np.nanmin(lat) + 90) / step) - 2, 0); i1 = min(int((np.nanmax(lat) + 90) / step) + 3, la.shape[0])
        j0 = max(int((np.nanmin(lon) + 180) / step) - 2, 0); j1 = min(int((np.nanmax(lon) + 180) / step) + 3, lo.shape[0])
        z = f['z'][i0:i1, j0:j1].astype(np.float32)
    y = (lat + 90) / step - .5 - i0
    x = (lon + 180) / step - .5 - j0
    return ndimage.map_coordinates(z, [np.nan_to_num(y), np.nan_to_num(x)], order=1, mode='nearest')


# ---------------------------------------------------------------- calibration
@lru_cache(None)
def metres_to_relief():
    """(metres, relief above LEVEL) table from vanilla land pixels inside the native footprint."""
    from repair_campaign_river_roots import Assets
    b = Assets(original_only=True).read('terrain/campaigns/3k_dlc07_main_map/lf_height_map.dds')
    vh, vw = struct.unpack_from('<2I', b, 12)
    van = np.frombuffer(b, '<u2', offset=128, count=vh * vw).reshape(vh, vw)[::-1]
    ys, xs = np.mgrid[0:vh:6, 0:vw:6]
    rel = (van[ys, xs].astype(np.float32) / 65535 - cw.NORM) * cw.GAIN
    tx = cw.NATIVE_TILE_X0 + xs / 4; ty = cw.NATIVE_TILE_Y0 + ys / 4
    q, r = tx / 2, ty / 2
    lon, lat = pixel_lonlat(q, r)
    ok = np.isfinite(lon) & (rel > .05)
    e = etopo(lon[ok], lat[ok]); rel = rel[ok]
    land = e > 0
    e, rel = e[land], rel[land]
    edges = np.array([0, 50, 100, 200, 300, 500, 700, 1000, 1300, 1600, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
    mids, meds = [0.], [0.]
    for a, b_ in zip(edges, edges[1:]):
        m = (e >= a) & (e < b_)
        if m.sum() > 200:
            mids.append(float(np.median(e[m]))); meds.append(float(np.median(rel[m])))
    meds = np.maximum.accumulate(meds)
    return np.array(mids), np.array(meds)


KNEE = 2200          # metres; above it vanilla has too few samples, so heights compress towards CEILING
CEILING = 9.6        # relief above LEVEL; the height map holds about 10


def to_relief(e):
    mids, meds = metres_to_relief()
    keep = mids <= KNEE
    mids, meds = mids[keep], meds[keep]
    low = np.interp(e, mids, meds)
    top = meds[-1] + (CEILING - meds[-1]) * (1 - np.exp(-(e - mids[-1]) / 2500))
    return np.where(e > mids[-1], top, low)


# ---------------------------------------------------------------- biomes
BIOME_COLOUR = {
    'snow': (238, 240, 244), 'barren': (150, 136, 118), 'alpine': (142, 146, 104), 'desert': (206, 186, 142),
    'steppe': (170, 166, 112), 'taiga': (58, 86, 66), 'conifer': (62, 92, 70), 'forest': (66, 98, 58),
    'tropical': (48, 102, 56), 'tropical_dry': (104, 120, 64), 'savanna': (160, 158, 96), 'paddy': (146, 170, 92),
    'wetland': (104, 130, 96), 'mangrove': (60, 96, 74), 'grass': (128, 150, 88), 'tundra': (132, 132, 110),
    'salt': (226, 222, 210),
}
# RESOLVE Ecoregions 2017 BIOME_NUM -> preview biome; the game material it would use in brackets
ECO_BIOME = {1: 'tropical',        # tropical moist broadleaf     [subtropical, dense trees]
             2: 'tropical_dry',    # tropical dry broadleaf       [subtropical, sparse trees]
             3: 'conifer',         # tropical/subtropical conifer [subtropical, trees]
             4: 'forest',          # temperate broadleaf & mixed  [temperate, trees]
             5: 'conifer',         # temperate conifer            [temperate/cold, trees]
             6: 'taiga',           # boreal forest / taiga        [cold, trees]
             7: 'savanna',         # tropical grassland/savanna   [subtropical/arid]
             8: 'steppe',          # temperate grassland          [steppe]
             9: 'wetland',         # flooded grassland            [wetlands]
             10: 'alpine',         # montane grassland            [cold]
             11: 'tundra',         # tundra                       [cold]
             12: 'grass',          # mediterranean                [arid/temperate]
             13: 'desert',         # desert & xeric shrub         [arid]
             14: 'mangrove',       # mangroves                    [wetlands]
             98: 'wetland', 99: 'snow'}
ECO_STEP = .02       # degrees per ecoregion grid cell
ECO_BOX = (40., -12., 152., 78.)


ECO_CACHE = 'eco-grid-v2.npz'


def eco_grid_key():
    """What the eco-grid cache is made from: eco_grid's code (with ECO_BIOME, ECO_STEP, ECO_BOX, read_shapes,
    fill_rings: strategic_ink's normalised code closure) and the content of Ecoregions2017.zip."""
    import hashlib
    from pathlib import Path
    import strategic_ink
    h = hashlib.sha256(strategic_ink._code_closure(Path(__file__), ('_eco_rasterise',)).encode('utf-8'))
    with open(GEODATA / 'Ecoregions2017.zip', 'rb') as f:
        h.update(hashlib.file_digest(f, 'sha256').digest())
    return h.hexdigest()


def _key_path(cache):
    return cache.with_name(cache.name + '.key.json')


@lru_cache(None)
def eco_grid():
    """(biome index grid over ECO_BOX, biome names, ecoregion id grid) rasterised from Ecoregions2017,
    cached as work/geodata/eco-grid-v2.npz with its key beside it (eco-grid-v2.npz.key.json). A cache whose key
    is missing or differs stops the run: rebuild it (delete it) or verify and record its key with
    `<tw3k python> scripts/frontier_terrain.py --stamp-eco-grid`."""
    import json
    cache = GEODATA / ECO_CACHE
    names = sorted(set(ECO_BIOME.values()))
    if cache.exists():
        kp = _key_path(cache)
        key = json.loads(kp.read_text(encoding='utf-8')).get('key') if kp.exists() else None
        if key != eco_grid_key():
            raise SystemExit(f'{cache} has {"no key" if key is None else "a key of other code or source data"}: '
                             'delete it to rebuild, or run frontier_terrain.py --stamp-eco-grid to verify it against '
                             'the source and record the key')
        z = np.load(cache)
        return z['biome'], names, z['eco']
    biome, eco = _eco_rasterise(names)
    np.savez_compressed(cache, biome=biome, eco=eco)
    _key_path(cache).write_text(json.dumps({'key': eco_grid_key(), 'by': 'eco_grid build'}), encoding='utf-8')
    return biome, names, eco


def stamp_eco_grid():
    """Rasterise the ecoregions afresh; when the arrays equal the cache's, record the cache's key (the cache file
    itself is not rewritten)."""
    import json
    cache = GEODATA / ECO_CACHE
    names = sorted(set(ECO_BIOME.values()))
    biome, eco = _eco_rasterise(names)
    z = np.load(cache)
    if not (np.array_equal(z['biome'], biome) and np.array_equal(z['eco'], eco)):
        raise SystemExit(f'{cache} differs from a fresh rasterisation: not stamped (delete it to rebuild)')
    _key_path(cache).write_text(json.dumps({'key': eco_grid_key(), 'by': 'stamp_eco_grid (arrays equal)'}),
                                encoding='utf-8')
    print('stamped', cache)


def _eco_rasterise(names):
    """Larger polygons first so enclaves overwrite them (even-odd fill, so holes stay open for the enclave drawn
    later)."""
    lon0, lat0, lon1, lat1 = ECO_BOX
    w, h = int((lon1 - lon0) / ECO_STEP), int((lat1 - lat0) / ECO_STEP)
    keep = []
    for rec, parts in read_shapes(GEODATA / 'Ecoregions2017.zip', ('ECO_NAME', 'BIOME_NUM', 'ECO_ID')):
        parts = [p for p in parts if p[:, 0].max() > lon0 and p[:, 0].min() < lon1 and p[:, 1].max() > lat0 and p[:, 1].min() < lat1]
        if parts:
            area = sum(abs(np.dot(p[:-1, 0], p[1:, 1]) - np.dot(p[1:, 0], p[:-1, 1])) for p in parts)
            keep.append((area, rec, parts))
    biome = np.full((h, w), 255, np.uint8)
    eco = np.zeros((h, w), np.int32)
    for _, rec, parts in sorted(keep, key=lambda t: -t[0]):
        m = fill_rings(parts, (h, w), lambda la: (lat1 - la) / ECO_STEP, lambda lo: (lo - lon0) / ECO_STEP)
        biome[m] = names.index(ECO_BIOME.get(int(rec['BIOME_NUM']), 'grass'))
        eco[m] = int(rec['ECO_ID'])
    return biome, eco


def biomes(lon, lat, e, slope):
    """Biome name per pixel: the RESOLVE ecoregion biome, with snow above the snowline and paddy on
    low tropical deltas (ecoregions describe natural vegetation, not Han-era farmland)."""
    grid, names, _ = eco_grid()
    lon0, lat0, lon1, lat1 = ECO_BOX
    gx = np.clip(((np.nan_to_num(lon) - lon0) / ECO_STEP).astype(np.int64), 0, grid.shape[1] - 1)
    gy = np.clip(((lat1 - np.nan_to_num(lat)) / ECO_STEP).astype(np.int64), 0, grid.shape[0] - 1)
    idx = grid[gy, gx].astype(np.int64)
    table = np.array(names + ['grass'], dtype=object)
    b = table[np.where(idx == 255, len(names), idx)]
    low_delta = (lat < 24) & (e < 25) & (slope < 2) & np.isin(b, ['tropical', 'tropical_dry', 'savanna', 'wetland'])
    b[low_delta] = 'paddy'
    snowline = np.interp(lat, [25, 30, 36, 42, 50, 56], [5600, 5700, 5600, 4300, 3200, 2600])   # dry Tibet high, Altai/Sayan low
    b[e > snowline] = 'snow'
    return b


# ---------------------------------------------------------------- water
def rasterise(shapes, q0, r0, w, h, scale, project, width=None, fill=True):
    """Draw lon/lat shapes onto a (h, w) pixel mask; pixel = (q - q0) * scale."""
    img = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(img)
    for item in shapes:
        parts, lw = item if width is None else (item, width)
        for part in parts:
            if len(part) < 2:
                continue
            q, r = project(part[:, 0], part[:, 1])
            xy = list(zip((q - q0) * scale, (r - r0) * scale))
            if fill:
                d.polygon(xy, fill=255)
            else:
                d.line(xy, fill=255, width=max(1, int(round(lw))))
    return np.asarray(img) > 127


def water(q0, r0, w, h, scale, lon_box):
    """Lake, river and salt-flat (playa) masks on the pixel grid, from Natural Earth."""
    lon0, lat0, lon1, lat1 = lon_box
    def inside(parts):
        return any(p[:, 0].max() > lon0 and p[:, 0].min() < lon1 and p[:, 1].max() > lat0 and p[:, 1].min() < lat1 for p in parts)
    # natural lakes only: reservoirs behind modern dams did not exist in 190
    lk = [parts for rec, parts in lakes() if rec.get('scalerank', 99) <= 8 and 'Reservoir' not in rec.get('featurecla', '') and inside(parts)]
    lake = rasterise([(p, 0) for p in lk], q0, r0, w, h, scale, pf.place_array)
    rv = [(parts, max(1.0, (9 - rec.get('scalerank', 9)) * .45 * scale)) for rec, parts in rivers()
          if rec.get('scalerank', 99) <= 9 and rec.get('featurecla', '').startswith('River') and inside(parts)]
    river = rasterise(rv, q0, r0, w, h, scale, pf.place_array, fill=False)
    salt = rasterise([(parts, 0) for rec, parts in playas() if inside(parts)], q0, r0, w, h, scale, pf.place_array)
    return lake, river, salt


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--stamp-eco-grid']:
        stamp_eco_grid()
    else:
        raise SystemExit('usage: frontier_terrain.py --stamp-eco-grid   (verify the eco-grid cache and record its key)')

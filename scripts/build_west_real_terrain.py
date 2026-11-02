"""Real-geography terrain for the Western Regions (needs scipy and h5py:
the tw3k Python (env/python-path.txt), with OPENBLAS_NUM_THREADS=1). Writes analysis/west-real-terrain.npz for polish_campaign_terrain.py
and output/west-real-terrain.png (preview).

Method (frontier_terrain.py, built by the frontier-planning session):
  * every tile cell of the zone is mapped back to lon/lat with frontier_terrain.pixel_lonlat
    (west of Dunhuang that is plan_western_regions.project, so the DEM lands under the towns);
  * ETOPO 2022 metres -> relief with frontier_terrain.to_relief (vanilla-calibrated curve);
  * near the native map the offset (vanilla - DEM) at the nearest native land cell is spread,
    smoothed (sigma 8 cells) and faded out over EDGE_FADE cells, so the DEM meets vanilla
    heights without a step;
  * Natural Earth lakes, rivers and playas, and the RESOLVE ecoregion biome per cell.
The zone is a lon/lat box blended out over BOX_FADE degrees at its sides.

  <tw3k python> build_west_real_terrain.py
"""
import tw3k_paths
import json, struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import campaign_world as cw
import frontier_terrain as ft
from analyze_reference import ROOT

A = cw.WORKSPACE
LON0, LON1, LAT0, LAT1 = 40.0, 141.0, -3.0, 85.0     # the whole canvas: Western Regions, frontier ring, old filler
BOX_FADE = 1.5                                        # degrees
EDGE_FADE = 80                                        # tile cells (40 hexes)
BIOME_CODES = ['desert', 'steppe', 'grass', 'alpine', 'tundra', 'snow', 'forest', 'conifer', 'taiga',
               'temperate', 'tropical', 'tropical_dry', 'savanna', 'wetland', 'paddy', 'mangrove', 'other']


def main():
    TH, TW = cw.TILE_H, cw.TILE_W
    # Zone bounding box in tile cells from the projected lon/lat box corners.
    import plan_frontier_regions as pf
    lon, lat = np.meshgrid(np.linspace(LON0, LON1, 60), np.linspace(LAT0, LAT1, 30))
    cq, cr = pf.place_array(lon, lat)
    x0, x1 = max(0, int(np.floor(cq.min() * 2)) - 4), min(TW, int(np.ceil(cq.max() * 2)) + 4)
    y0, y1 = max(0, int(np.floor(cr.min() * 2)) - 4), min(TH, int(np.ceil(cr.max() * 2)) + 4)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    q, r = xs / 2, ys / 2
    print('zone tiles', (x0, x1, y0, y1), flush=True)
    plon, plat = ft.pixel_lonlat(q.ravel(), r.ravel())
    plon, plat = plon.reshape(q.shape), plat.reshape(q.shape)
    ok = np.isfinite(plon)
    e = np.where(ok, ft.etopo(np.nan_to_num(plon), np.nan_to_num(plat)), 0).astype(np.float32)
    rel = (cw.LEVEL + ft.to_relief(np.maximum(e, 0))).astype(np.float32)
    # Box weight: 1 inside, smooth out over BOX_FADE degrees.
    def ramp(v, a, b):
        t = np.clip(np.minimum(v - a, b - v) / BOX_FADE, 0, 1)
        return t * t * (3 - 2 * t)
    weight = np.where(ok, ramp(plon, LON0, LON1) * ramp(plat, LAT0, LAT1), 0).astype(np.float32)
    # Native map: keep vanilla heights there; spread the vanilla-DEM offset outward.
    native = np.zeros(q.shape, bool)
    tx, ty = xs.astype(int), ys.astype(int)
    native = (tx >= cw.NATIVE_TILE_X0) & (tx < cw.NATIVE_TILE_X1) & (ty >= cw.NATIVE_TILE_Y0) & (ty < cw.NATIVE_TILE_Y1)
    if native.any():
        from repair_campaign_river_roots import Assets
        b = Assets(original_only=True).read('terrain/campaigns/3k_dlc07_main_map/lf_height_map.dds')
        vh, vw = struct.unpack_from('<2I', b, 12)
        van = np.frombuffer(b, '<u2', offset=128, count=vh * vw).reshape(vh, vw)[::-1]
        vy = np.clip((ty - cw.NATIVE_TILE_Y0) * 4 + 2, 0, vh - 1); vx = np.clip((tx - cw.NATIVE_TILE_X0) * 4 + 2, 0, vw - 1)
        vrel = cw.LEVEL + (van[vy, vx].astype(np.float32) / 65535 - cw.NORM) * cw.GAIN
        vland = native & (vrel > cw.LEVEL - 1.5)
        off = np.where(vland, vrel - rel, 0).astype(np.float32)
        dist, (iy, ix) = ndimage.distance_transform_edt(~vland, return_indices=True)
        off = ndimage.gaussian_filter(off[iy, ix], 8)
        t = np.clip(1 - dist / EDGE_FADE, 0, 1); t = t * t * (3 - 2 * t)
        rel = rel + off * t
        weight = np.where(native, 0, weight)
    rel = np.where(e <= 0, np.minimum(rel, cw.LEVEL), rel)          # depressions (Turpan) clamp at the plain
    # Water, salt and biomes.
    lake, river, salt = ft.water(x0 / 2, y0 / 2, x1 - x0, y1 - y0, 2, (LON0 - 2, LAT0 - 2, LON1 + 2, LAT1 + 2))
    gy, gx = np.gradient(rel)
    slope = np.hypot(gx, gy) * 2
    bio = ft.biomes(plon, plat, e, slope)
    codes = np.array([BIOME_CODES.index(b) if b in BIOME_CODES else len(BIOME_CODES) - 1 for b in bio.ravel()], np.uint8).reshape(bio.shape)
    np.savez_compressed(A / 'analysis/west-real-terrain.npz', box=np.array([x0, x1, y0, y1]), relief=rel, weight=weight,
                        lake=lake, river=river, salt=salt, biome=codes, elevation=e, lon=plon.astype(np.float32), lat=plat.astype(np.float32),
                        biome_names=np.array(BIOME_CODES))
    print('saved; relief range', float(rel[weight > .5].min()), float(rel[weight > .5].max()), 'lake cells', int(lake.sum()), flush=True)
    preview(rel, weight, lake, river, salt, bio, (x0, x1, y0, y1))


def preview(rel, weight, lake, river, salt, bio, box):
    x0, x1, y0, y1 = box
    col = {k: np.array(v, np.float32) for k, v in ft.BIOME_COLOUR.items()} if hasattr(ft, 'BIOME_COLOUR') else {}
    base = np.full(rel.shape + (3,), 200, np.float32)
    for name, c in col.items():
        base[bio == name] = c
    gy, gx = np.gradient(rel)
    shade = np.clip(1 + (gx - gy) * 1.6, .45, 1.5)[..., None]
    img = np.clip(base * shade, 0, 255)
    img[salt] = (235, 232, 220); img[river] = (70, 110, 170); img[lake] = (50, 95, 160)
    img = (img * (.35 + .65 * weight[..., None]) + 120 * (1 - weight[..., None]) * .65).astype(np.uint8)
    im = Image.fromarray(img[::-1]).resize(((x1 - x0) // 2, (y1 - y0) // 2))
    d = ImageDraw.Draw(im); f = ImageFont.truetype(tw3k_paths.font('msyh.ttc'), 12); Hh = im.size[1]
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    for t in design['regions']:
        tq, tr = t['town']
        x, y = (tq * 2 - x0) / 2, Hh - (tr * 2 - y0) / 2
        if 0 <= x < im.size[0] and 0 <= y < Hh:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(180, 0, 0) if t['is_capital'] else (30, 30, 30))
            if t['is_capital']:
                d.text((x + 4, y - 8), t['name'].split('·')[0], font=f, fill=(0, 0, 0))
    im.save(ROOT / 'output/west-real-terrain.png')


if __name__ == '__main__':
    main()

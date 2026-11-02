"""Hehuang (河湟 / 西海 / Qinghai Lake) real-geography terrain patch: data preparation.

Run with Anaconda (scipy, h5py):  <tw3k python> scripts/hehuang_patch.py build [--preview]
Writes analysis/hehuang-patch.npz, which every later stage and audit reads
(polish_campaign_terrain via hehuang_terrain, polish_campaign_foliage,
hehuang_lake_prop, native_edit_mask.hehuang_mask and the native audits).

The area is vanilla 3k_main_reg_non_playable land inside the native map (given to
the xihai / xiping regions). It keeps vanilla's datum but takes the real relief:

  weight    jittered ellipse around hex (950, 905), semi-axes 52 (east) / 45 (west)
            x 42 hexes, full weight inside 0.8 of the radius, smoothstep to 0 at 1.0
            (an 8-10 hex blend band), zero within 8 hexes of vanilla playable land and
            on vanilla town plots;
  relief    raw DEM relief (frontier_terrain.to_relief of the npz `elevation`, NOT
            the offset-matched npz `relief`) plus a smoothed (sigma 24 cells) offset
            of vanilla minus DEM, so the band does not step;
  lake      Qinghai Lake = the 3k_main_sea_lake hexes of map.hex in the window. Its
            water is a scaled vanilla lake prop at its own height (`water`, the DEM
            lake level plus offset), the add_inland_lake Tier 2 / vanilla high-lake
            precedent: a bed bowl 0.1-0.5 below the water inside the lake hexes, a
            shore ring >= water + SHORE_RISE, and every land cell under the prop's
            rectangle above the water (the plane shows wherever terrain is lower);
  towns     plots levelled to the real relief (median over 3 hexes, Gaussian-mixed
            over TOWN_CLUSTER hexes); lake-shore towns stand >= water + 0.3 and
            reach the shore by a slope;
  ground    class per tile cell: 0 steppe/meadow, 1 alpine rock, 2 snow/scree,
            3 conifer, 4 Chaka desert, 5 lake.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

import campaign_world as cw

A = cw.WORKSPACE
OUT = A / 'analysis/hehuang-patch.npz'
VERSION = 2   # bump when the build logic changes (build_rules_sha keys constants only): 2 = town weight on the town hex (11-01)
CENTRE = (950., 905.)
SEMI_EAST, SEMI_WEST, SEMI_R = 52., 45., 42.
FULL, JITTER = .8, .08
PROTECT_HEXES = 8
OFFSET_SIGMA = 24          # tile cells
HEX_WIN = (860, 1040, 825, 990)   # q0, q1, r0, r1 (exclusive ends)
BED_DEPTH, BED_EDGE, SHORE_RISE, SHORE_HEXES = .5, .1, .2, 3
BASIN = (6, 20)            # hexes from the lake: constant offset, then blend
RIM_MARGIN = .15           # land under the prop rectangle stays this far above the water
LAKE_TOWN_CLEAR = .3
TOWN_CLUSTER = 8           # hexes
TOWN_FLAT = {True: (3, 9), False: (2, 6)}   # capital?: (full, zero) radius in hexes
# Lake towns (level >= water + LAKE_TOWN_CLEAR): patch towns within LAKE_TOWN_REACH hexes of the lake hexes (the build's
# own distance field dl at the town cell), from the design, not by key (P6: the regions are redesigned). 2026-11-01
# design: 破羌 4.6 and 龙夷 6.9 hexes, the next town 14.2 (xihai_salt): the same two as the old key list.
LAKE_TOWN_REACH = 10.
SMALL_LAKE = 'rigidmodels/campaign/props/lakes/small_lake_1.wsmodel'
SMALL_LAKE_X, SMALL_LAKE_Z = (-3.1207449, 3.0364077), (-2.7091434, 2.7091434)   # LOD0 bounds (probe of models.pack)
VANILLA_LAKE_Y_SCALE = .42
CLASS_NAMES = ['steppe', 'alpine_rock', 'snow', 'conifer', 'desert', 'lake']


def rules_sha():
    """Fingerprint of this build's rule constants (npz 'build_rules_sha'; hehuang_terrain.check_sources)."""
    import hehuang_terrain
    return hehuang_terrain.build_rules_sha(sys.modules[__name__])


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def vanilla_tile_height(tx, ty):
    """Vanilla dlc07 LF at tile cells (LF texel 4t of the native frame), relief units."""
    from repair_campaign_river_roots import Assets
    b = Assets(original_only=True).read('terrain/campaigns/3k_dlc07_main_map/lf_height_map.dds')
    vh, vw = struct.unpack_from('<2I', b, 12)
    van = np.frombuffer(b, '<u2', offset=128, count=vh * vw).reshape(vh, vw)[::-1]
    vy = np.clip((ty - cw.NATIVE_TILE_Y0) * 4, 0, vh - 1)
    vx = np.clip((tx - cw.NATIVE_TILE_X0) * 4, 0, vw - 1)
    return (cw.LEVEL + (van[vy, vx].astype(np.float32) / 65535 - cw.NORM) * cw.GAIN).astype(np.float32)


def build(preview=False, out=None, preview_dir=None, map_cells=None, map_names=None):
    """Writes the patch cache (default OUT). map_cells / map_names: the map to build from (bake_map step 10 passes the
    map it is building and never the final map.hex); default (CLI) the workspace's final map.hex."""
    from scipy import ndimage
    import frontier_terrain as ft
    import map_hex
    from terrain_noise import fbm
    q0, q1, r0, r1 = HEX_WIN
    x0, x1, y0, y1 = 2 * q0, 2 * q1, 2 * r0, 2 * r1
    ty, tx = np.mgrid[y0:y1, x0:x1]
    hq, hr = cw.tile_hex(tx, ty)
    q, r = tx / 2., ty / 2.
    src = A / 'analysis/west-real-terrain.npz'
    z = np.load(src)
    bx0, bx1, by0, by1 = (int(v) for v in z['box'])
    assert bx0 <= x0 and x1 <= bx1 and by0 <= y0 and y1 <= by1, 'west-real-terrain box does not cover the Hehuang window'
    sl = np.s_[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    e = z['elevation'][sl].astype(np.float32)
    bio = z['biome'][sl]
    names = [str(n) for n in z['biome_names']]
    raw = (cw.LEVEL + ft.to_relief(np.maximum(e, 0))).astype(np.float32)
    van = vanilla_tile_height(tx, ty)

    # Maps: final (lake hexes, regions) and vanilla (playable regions, town plots).
    map_given = map_cells is not None
    F = {'cells': map_cells, 'names': list(map_names)} if map_given else map_hex.read(A / 'work/final-map/map.hex')
    fn = F['names'] if map_given else F['land_regions'] + F['sea_regions']
    fid = map_hex.region_ids(F['cells'])
    V = map_hex.read(cw.VANILLA_MAP)
    vn = V['land_regions'] + V['sea_regions']
    vid = map_hex.region_ids(V['cells'])
    vq, vr = hq - cw.PAD_LEFT, hr - cw.PAD_BOTTOM
    inside = (vq >= 0) & (vq < cw.NATIVE_W) & (vr >= 0) & (vr < cw.NATIVE_H)
    vqc, vrc = np.clip(vq, 0, cw.NATIVE_W - 1), np.clip(vr, 0, cw.NATIVE_H - 1)
    vname = np.array(vn + ['?'])[np.where(vid >= 0, vid, len(vn))][vrc, vqc]
    fname = np.array(fn + ['?'])[np.where(fid >= 0, fid, len(fn))][hr, hq]
    protected = inside & (vname == fname) & (vname != '3k_main_reg_non_playable')
    vcells = V['cells'][vrc, vqc]
    plots = inside & (((vcells[..., 2] >> 4) > 0) | ((vcells[..., 3] & 1) > 0))
    lake = (fname == '3k_main_sea_lake') & inside
    land = (F['cells'][hr, hq, 0] & 3) != 1

    # Weight: jittered ellipse, protected and plot clips.
    semi_q = np.where(q < CENTRE[0], SEMI_WEST, SEMI_EAST)
    d = np.hypot((q - CENTRE[0]) / semi_q, (r - CENTRE[1]) / SEMI_R)
    d = d * (1 + JITTER * fbm(q, r, (0, 0), 14, 3, seed=211))
    w = smoothstep((1 - d) / (1 - FULL))
    pd = ndimage.distance_transform_edt(~protected) / 2.
    w *= np.clip(pd / PROTECT_HEXES, 0, 1)
    w[plots] = 0
    w = w.astype(np.float32)

    # Relief: raw DEM + smoothed offset (datum kept).
    offset = ndimage.gaussian_filter((van - raw).astype(np.float64), OFFSET_SIGMA, mode='nearest').astype(np.float32)
    # A lake is level: around it the offset is one constant (its median over the lake),
    # blended back to the smoothed offset between BASIN hexes from the shore. The
    # varying vanilla datum otherwise tilted the shore up to 1 unit under the water.
    dl = ndimage.distance_transform_edt(~lake) / 2.
    basin = 1 - smoothstep((dl - BASIN[0]) / (BASIN[1] - BASIN[0]))
    offset = (offset * (1 - basin) + float(np.median(offset[lake])) * basin).astype(np.float32)
    hh = raw + offset

    # Water: DEM lake level + offset, measured over the lake hexes.
    lk = lake & (w > .99)
    assert lk.sum() > 500, ('Qinghai lake hexes missing or outside full weight', int(lake.sum()), int(lk.sum()))
    water = float(np.median(hh[lake]))

    # Towns: plots levelled to the real relief, clustered, lake towns above the water.
    import hehuang_terrain
    plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    towns = []
    for t in plan['regions']:
        tq, tr = t['town']
        cx, cy = cw.display_position(tq, tr)
        cx, cy = cx / cw.DX * 2 - x0, cy / cw.DZ * 2 - y0
        tw = hehuang_terrain.town_weight(w, tq, tr, x0, y0)   # the town hex's own tiles (not round(display): its east neighbour)
        if tw <= .5:
            continue
        disc = (np.hypot(tx - x0 - cx, ty - y0 - cy) <= 6) & land & ~lake
        towns.append(dict(region=t['region'], name=t['name'], town=[tq, tr], capital=bool(t['is_capital']), cx=float(cx), cy=float(cy),
                          w=tw, median=float(np.median(hh[disc])), vanilla=float(np.median(van[disc]))))
    if towns:
        p = np.array([(t['cx'], t['cy']) for t in towns])
        h0 = np.array([t['median'] for t in towns])
        g = np.exp(-((p[:, None] - p[None]) ** 2).sum(-1) / (2 * (2 * TOWN_CLUSTER) ** 2))
        lev = (g @ h0) / g.sum(1)
        for t, v in zip(towns, lev):
            t['level'] = float(v)
            if float(dl[int(round(t['cy'])), int(round(t['cx']))]) <= LAKE_TOWN_REACH:
                t['level'] = max(t['level'], water + LAKE_TOWN_CLEAR)
                t['lake_town'] = True
        num = np.zeros(hh.shape, np.float64)
        den = np.zeros(hh.shape, np.float64)
        flat = np.zeros(hh.shape, np.float32)
        for t in towns:
            dd = np.hypot(tx - x0 - t['cx'], ty - y0 - t['cy']) / 2.   # hexes
            full, zero = TOWN_FLAT[t['capital']]
            c = 1 - smoothstep((dd - full) / (zero - full))
            gk = np.exp(-dd ** 2 / (2 * 5. ** 2))
            num += gk * t['level']
            den += gk
            flat = np.maximum(flat, c)
        mix = np.where(den > 1e-6, num / np.maximum(den, 1e-6), hh)
        hh = np.where(land & ~lake, hh * (1 - flat) + mix * flat, hh).astype(np.float32)
    # the Qinghai Lake shore has lake towns: none found means the design moved them (or the reach rule is stale)
    assert any(t.get('lake_town') for t in towns), ('no patch town within LAKE_TOWN_REACH hexes of the lake', LAKE_TOWN_REACH)

    # Lake bowl, shore ring and the prop rectangle rim.
    # --- L2048 HEHUANG SHORE --- one smoothed contour both ways: `lake | blur>=.5` kept every
    # convex hex tooth of the lake hexes (right-angle shore steps, live 2048l).
    lsoft = ndimage.gaussian_filter(lake.astype(np.float32), 3.0)
    lakec = lsoft >= .5
    din = ndimage.distance_transform_edt(lakec)            # cells to the shore, inside
    dout = ndimage.distance_transform_edt(~lakec)          # cells to the lake, outside
    bed = water - BED_EDGE - (BED_DEPTH - BED_EDGE) * smoothstep(din / 4.)
    centres = np.array([cw.block_centre(a, b) for a, b in zip(*[v.ravel() for v in (hq[lake], hr[lake])])])
    ux, uz = centres[:, 0], centres[:, 1]
    pad = cw.DX * 1.0
    xmin, xmax, zmin, zmax = ux.min() - pad, ux.max() + pad, uz.min() - pad, uz.max() + pad
    wx, wz = (tx + .5) * cw.DX / 2, (ty + .5) * cw.DZ / 2
    rect = (wx >= xmin - cw.DX) & (wx <= xmax + cw.DX) & (wz >= zmin - cw.DZ) & (wz <= zmax + cw.DZ)
    ring = (dout > 0) & (dout <= 2 * SHORE_HEXES)
    held = ring | (rect & ~lakec)
    # Floor: water + SHORE_RISE on the shore ring, water + RIM_MARGIN under the rest of
    # the prop rectangle (+1 hex), falling at .25 per cell beyond them (no step at its edge).
    fd = ndimage.distance_transform_edt(~held)
    floor = (water + RIM_MARGIN - .25 * fd).astype(np.float32)
    # The shore starts just above the water and reaches SHORE_RISE 3 cells out: a full
    # step at the hex-toothed outline shaded as a dotted rim (offline preview).
    floor[ring] = np.maximum(floor[ring], water + .08 + (SHORE_RISE - .08) * smoothstep((dout[ring] - 1) / 3.))
    lift = np.where(lakec, 0, np.clip(floor - hh, 0, None))
    hh = np.where(lakec, bed, np.maximum(hh, floor)).astype(np.float32)
    # --- L2048 HEHUANG SHORE --- the water line follows the continuous field, not the cell
    # outline: a bed/floor step between neighbour cells drew half-hex stairs once upsampled.
    band = np.abs(lsoft - .5) < .25
    hh = np.where(band, water + .6 * (.5 - lsoft), hh).astype(np.float32)
    shore = np.clip(1.2 - dout / 10., 0, 1).astype(np.float32) * ~lakec

    # Ground classes (mock-up rules).
    gy, gx = np.gradient(hh)
    slope = np.hypot(gx, gy) * 2
    cls = np.zeros(hh.shape, np.uint8)
    cls[(e > 3900) | (slope > 1.2)] = 1
    cls[e > 4500] = 2
    con = (bio == names.index('conifer')) | ((e > 2500) & (e < 3400) & (gy < -.15) & (slope > .4))
    cls[con & (e < 3900)] = 3
    cls[bio == names.index('desert')] = 4
    cls[lakec] = 5

    # Prop: one axis-aligned scaled small_lake_1 (diagonal matrix, so row/column
    # conventions agree) at the water height, covering the lake hexes + 1 hex.
    sx = (xmax - xmin) / (SMALL_LAKE_X[1] - SMALL_LAKE_X[0])
    sz = (zmax - zmin) / (SMALL_LAKE_Z[1] - SMALL_LAKE_Z[0])
    px = xmin - SMALL_LAKE_X[0] * sx
    pz = zmin - SMALL_LAKE_Z[0] * sz
    prop = dict(model=SMALL_LAKE, matrix=[sx, 0., 0., 0., VANILLA_LAKE_Y_SCALE, 0., 0., 0., sz, px, water, pz],
                world_rect=[xmin, zmin, xmax, zmax], water=water,
                basis='vanilla high lake: small_lake_1 at y 5.969, matrix [-0.801,0,0.747 / 0,0.42,0 / -0.747,0,-0.801], '
                      'bmd_objects.3k_main_sea_lake.475.31.bin; LF bed ~4.9, shore ~6.2')

    # Hex mask (weight > 0 on any of a hex's tile cells).
    hexmask = np.zeros((r1 - r0, q1 - q0), bool)
    hexmask[(hr - r0)[w > 0], (hq - q0)[w > 0]] = True
    sha = hashlib.sha256()
    for p_ in (src, None if map_given else A / 'work/final-map/map.hex', cw.DESIGN_PATH, Path(__file__)):
        sha.update(np.ascontiguousarray(F['cells']).tobytes() if p_ is None else p_.read_bytes())   # the map built, when given
    stats = dict(version=VERSION, hexes_weight_gt0=int(hexmask.sum()), tiles_full=int((w >= .999).sum()),
                 hexes_full_approx=int((w >= .999).sum() / 4), lake_hexes=int(len(set(zip(hq[lake].tolist(), hr[lake].tolist())))),
                 water=water, bed_min=float(hh[lakec].min()), shore_lift={'max': float(lift.max()), 'mean_where_lifted': float(lift[lift > 0].mean()) if (lift > 0).any() else 0., 'cells_over_0.05': int((lift > .05).sum()), 'cells_over_0.3': int((lift > .3).sum())}, relief_minus_vanilla_in_patch={'mean_abs': float(np.abs(hh - van)[w > 0].mean()), 'max_abs': float(np.abs(hh - van)[w > 0].max())},
                 class_share_core={n: float(((cls == i) & (w >= .999)).sum() / max(1, (w >= .999).sum())) for i, n in enumerate(CLASS_NAMES)},
                 towns=towns, prop=prop)
    out = Path(out) if out else OUT
    import hehuang_terrain
    np.savez_compressed(out, version=VERSION, sha=sha.hexdigest(), design_sha=hehuang_terrain.design_sha(HEX_WIN, plan),
                        inputs_sha=json.dumps(hehuang_terrain.input_sha(HEX_WIN, None, F['cells'], fn)), build_rules_sha=rules_sha(), box=np.array([x0, x1, y0, y1]), hex_box=np.array([q0, q1, r0, r1]),
                        weight=w.astype(np.float16), hex_mask=hexmask, relief=hh.astype(np.float32), vanilla=van, ground=cls, lake=lakec,
                        lake_hex=lake, shore=shore, water=np.float32(water), class_names=np.array(CLASS_NAMES),
                        towns=json.dumps(towns, ensure_ascii=False), prop=json.dumps(prop))
    out.with_suffix('.json').write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k not in ('towns',)}, ensure_ascii=False))
    for t in towns:
        print(' town', t['region'], t['name'], 'w', round(t['w'], 2), 'vanilla', round(t['vanilla'], 2), 'median', round(t['median'], 2), 'level', round(t['level'], 2))
    if preview:
        _preview(hh, van, w, cls, lakec, rect, x0, y0, preview_dir)
    print('->', out)


def _preview(hh, van, w, cls, lake, rect, x0, y0, outdir=None):
    from PIL import Image
    outdir = Path(outdir) if outdir else Path(__file__).resolve().parents[1] / 'output/offline-view/hehuang-after'
    outdir.mkdir(parents=True, exist_ok=True)
    gy, gx = np.gradient(hh)
    sh = np.clip(1 + (gx - gy) * 1.2, .45, 1.5)[..., None]
    pal = np.array([(176, 170, 96), (150, 140, 120), (235, 235, 235), (40, 80, 45), (214, 190, 140), (50, 95, 160)], np.float32)
    img = pal[cls] * sh
    vcol = np.array([60, 85, 45], np.float32) * sh
    img = img * w[..., None] + vcol * (1 - w[..., None])
    edge = (w > 0) & ~np.roll(w > 0, 1, 0) | (w > 0) & ~np.roll(w > 0, 1, 1)
    img[edge] = (255, 0, 0)
    re = rect & ~np.roll(rect, 1, 0) | rect & ~np.roll(rect, 1, 1)
    img[re] = (0, 200, 255)
    Image.fromarray(np.clip(img[::-1], 0, 255).astype(np.uint8)).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(outdir / 'patch-ground-classes.png')
    t = np.clip((hh - 4) / 8, 0, 1)[..., None]
    Image.fromarray(np.clip((np.array([70, 110, 60]) * (1 - t) + np.array([235, 225, 200]) * t) * sh, 0, 255)[::-1].astype(np.uint8)).resize((hh.shape[1] * 2, hh.shape[0] * 2), Image.NEAREST).save(outdir / 'patch-relief.png')


def load():
    """The patch, or None (numpy only; used by the pipeline stages)."""
    if not OUT.exists():
        return None
    z = np.load(OUT)
    out = {k: z[k] for k in z.files}
    out['towns'] = json.loads(str(out['towns']))
    out['prop'] = json.loads(str(out['prop']))
    out['water'] = float(out['water'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['build'])
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--out', help='npz path (default analysis/hehuang-patch.npz; a dry run elsewhere)')
    ap.add_argument('--preview-dir')
    a = ap.parse_args()
    if a.action == 'build':
        build(a.preview, a.out, a.preview_dir)


if __name__ == '__main__':
    main()

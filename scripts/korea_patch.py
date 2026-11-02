"""Korean peninsula real-geography terrain patch: data preparation (second patch of hehuang_terrain).

Run with Anaconda (scipy, h5py):  <tw3k python> scripts/korea_patch.py build [--preview]
Writes analysis/korea-patch.npz/.json, which hehuang_terrain.load() combines with the
Hehuang patch for polish_campaign_terrain, polish_campaign_foliage, native_edit_mask and
the native audits (identity / overlay / cache exempt the or-ed masks grown by 2 hexes).

The peninsula lies inside the native rectangle, on vanilla 3k_main_reg_non_playable land
(the mod's tw3k_hist_* Korean regions); vanilla had weak relief and one material there.
No vanilla playable region is in it (Liaodong and Donglai are clipped as in Hehuang).

  height    `height_weight`: land of the peninsula and the northern highlands, edges from
            window() (the design's peninsula-class regions; 2026-11-01 west q 1578, north r 1048,
            east q 1767: 2 hexes inside the 10-hex native rim),
            jittered, smoothstep to full over BAND hexes inward (the blend into untouched
            vanilla); zero within PROTECT_HEXES of vanilla playable land, on vanilla plots
            and at sea.
  ground    `weight` = height weight faded to zero within COAST_KEEP hexes of the sea, so
            the coast keeps vanilla ground, coast tiles and trees (the coastline is right).
  relief    raw DEM relief (frontier_terrain.to_relief of west-real-terrain `elevation`)
            plus a land-only smoothed (sigma OFFSET_SIGMA cells) vanilla-minus-DEM offset:
            vanilla's datum at large scale, the real ranges (Taebaek, Rangrim, Kaema,
            Sobaek) at full amplitude; RANGE_GAIN lifts the relief above the local datum so
            the ranges read at campaign scale; within COAST_BLEND hexes of the sea it returns
            to vanilla (the coastline and its heights are kept).
  towns     plots levelled to the real relief (median over 3 hexes, Gaussian-mixed over
            TOWN_CLUSTER hexes), flat to TOWN_FLAT[0] and eased back over to TOWN_FLAT[1]
            hexes, so plots meet their surroundings with gentle slopes (no rims).
  ground    class per tile cell (codes 10+, global across patches):
            10 grass (lowland open country: steppe donor, arid climate), 11 broadleaf
            (temperate forest), 12 mixed (temperate climate, fir and katsura), 13 conifer
            (cold forest), 14 rock (high or steep, cold); with class_climate,
            steppe_classes and species (foliage) stored in the npz.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import campaign_world as cw
from hehuang_patch import smoothstep, vanilla_tile_height

import biome_profiles as _bp

A = cw.WORKSPACE
OUT = A / ('analysis/korea-patch-forests.npz' if _bp.korea_forests_on() else 'analysis/korea-patch.npz')   # TW3K_KOREA_FORESTS: own cache
VERSION = 2   # bump when the build logic changes (build_rules_sha keys constants only): 2 = town weight on the town hex (11-01)
FORESTS = _bp.korea_forests()   # forest zones and species: biome_profiles REGIONS korea_forests (rules_sha in the npz)
# window and edges (q0, q1, r0, r1 / weight 0 on the west, north, east lines): window(), from the design (P6)
BAND = 9.                  # hexes: blend into untouched vanilla
EDGE_JITTER = 3.           # hexes of fbm on the edge distance (no straight blend line)
PROTECT_HEXES = 8
OFFSET_SIGMA = 40          # tile cells (20 hexes): the datum scale, broader than the ranges
RANGE_SIGMA = 4            # tile cells: ranges (above it) vs detail (below it)
RANGE_GAIN, DETAIL_GAIN = 1.8, .7   # ranges stand out at campaign scale; the DEM's fine crumple is calmed
LAND_MIN_M = 40            # metres: map land where the DEM is sea (coast offsets) reads as lowland, not a pit
LAND_FLOOR = .6            # relief: lowland floor; polish holds plots >= 1.0, so lower plains made town mounds
COAST_RISE, COAST_FOOT = .9, .5   # relief per hex above vanilla inland from the shore (mountain coasts, DEM/map coast offsets)
COAST_BLEND = (.25, 1.25)  # hexes from the sea: vanilla relief -> patch relief
WARP_REACH, WARP_MAX = 120, 70     # tile cells: DEM search beyond the window, largest coast shift
WARP_NORTH = (1030, 1060)          # hex rows: the coast warp fades out (no coasts to align north of it)
COAST_KEEP = (1., 3.)      # hexes from the sea: vanilla ground -> patch ground
TOWN_CLUSTER = 4           # hexes
TOWN_FLAT = {True: (4.5, 16.), False: (3.5, 14.)}   # capital?: (full, zero) radius in hexes; full >= polish's plot + apron
                           # FIX2 KOREA GROUND: zero radius 11/10 -> 16/14: lowland towns lifted to MIN_TOWN_HEIGHT
                           # (Lelang, Jinbeongun: .36 above the .64 plain) read as plateau rims where the ease ended
                           # (~3.75 hexes at Kimhae: a smaller flat left a 0.12 step at the plateau edge, run 1)
MIN_TOWN_HEIGHT = 1.0      # polish_campaign_terrain.MIN_TOWN_HEIGHT (the patch target replaces its clamp)
CLASS_NAMES = {10: 'grass', 11: 'broadleaf', 12: 'mixed', 13: 'conifer', 14: 'rock'}
CLASS_CLIMATE = {10: 3, 11: 2, 12: 2, 13: 1, 14: 1}   # 0 subtropical, 1 cold, 2 temperate, 3 arid
STEPPE_CLASSES = [10]
DONOR_CLASSES = [10, 14]   # open ground takes donor ground; forest keeps vanilla's woodland floor (materials 4/5)
FOREST_CLASSES = [11, 12, 13]
FOREST_TOWN = (3, 8, .3)   # polish_campaign_terrain.FOREST_TOWN: ring (hexes) and forest share
FOREST_FLOOR = (4., 6.)    # hexes: woodland floor painted under forest towns (full, zero)
# --- FIX2 KOREA GROUND --- (offline 2048bq: hard tile staircases on every forest/open edge, plain
# towns on dark woodland discs). DONOR_SOFT: Gaussian sigma (tile cells) of the open-class donor
# mask (1.5 made a 0->1 step within ~2 cells, read through polish's per-tile share as half-hex
# stairs) plus DONOR_FRAY x fbm (hexes grain) so the forest edge meanders. PATCH_FOREST_TOWN: ring
# forest share needed for a forest town (polish's .3 made all 30 Korean towns forest towns: the
# Pyongyang plain under Lelang/Jinbeongun turned to woodland discs); FOREST_DISC_JITTER: hexes
# of fbm on the disc radius (no round discs).
DONOR_SOFT, DONOR_FRAY = 3.5, (.22, 2.5)
PATCH_FOREST_TOWN = .6
FOREST_DISC_JITTER = 1.5
# Coastal towns: the town flat (MIN_TOWN_HEIGHT 1.0, TOWN_FLAT up to 11 hexes) ran to the shore and
# ended in ~2-unit walls at the water (Jinbeongun, Lelang resource 1). After levelling the
# coastal cap applies again, except within TOWN_COAST_PLOT hexes of the town (the plot itself).
TOWN_COAST_PLOT = 1.6
TOWN_COAST_RISE = .25      # relief per hex above vanilla inland from the shore, inside town flats
# --- end FIX2 KOREA GROUND
# class: (species, dense, sparse): chance per hex (patch noise > -.1 / else), 3 trees (1 rock)
# (foliage samples every other hex in q and r: a forest hit plants 8 trees over +-1.1 hexes;
#  3 per hit at .3 left the forest at a tenth of vanilla's density, run 1). Table: REGIONS korea_forests species.
SPECIES = FORESTS['species']


def rules_sha():
    """Fingerprint of this build's rule constants and its REGIONS korea_patch window rule (npz 'build_rules_sha';
    hehuang_terrain.check_sources). The forest rule keeps its own 'rules_sha' (korea_forests_sha)."""
    import hehuang_terrain
    return hehuang_terrain.build_rules_sha(sys.modules[__name__], _bp.region('korea_patch'))


def on_peninsula(lon, lat, poly=None):
    """True where (lon, lat) lies inside REGIONS korea_patch 'peninsula' (south of the Yalu / Tumen line, closed over the
    sea): even-odd ray test, arrays or scalars."""
    poly = np.asarray(poly if poly is not None else _bp.region('korea_patch')['peninsula'], np.float64)
    x, y = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
    inside = np.zeros(np.broadcast(x, y).shape, bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        cross = (y1 > y) != (y2 > y)
        xc = x1 + (y - y1) * (x2 - x1) / np.where(y2 != y1, y2 - y1, 1)
        inside ^= cross & (x < xc)
    return inside


def window(A=None, map_cells=None, map_names=None):
    """{'hex_win': (q0, q1, r0, r1) exclusive ends, 'edges': (west, north, east) hexes (weight 0 on these lines),
    'regions': the class regions} from the region design (P6, 2026-11-01; was HEX_WIN / EDGE_* constants): the land
    hexes (final map.hex, inside the native rectangle) of the design regions whose province stem is in region-traits
    province_classes[REGIONS korea_patch province_class] give the west edge and the window's west / south ends
    (REGIONS korea_patch offsets); the north / east edge and the window's north / east ends are the native frame's.
    map_cells / map_names: the map (bake_map step 10 passes the map it is building); default the final map.hex.
    Stops on an empty class, a class region missing from the map, or a class town off the real peninsula
    (on_peninsula: south of the Yalu / Tumen). Reads the final map.hex on disk: call it only after the map bake
    (polish, korea_patch build), never inside bake_map, which builds the map from scratch (the file is last run's)."""
    import map_hex
    A = Path(A) if A else cw.WORKSPACE
    k = _bp.region('korea_patch')
    cls = k['province_class']
    stems = set(cw.region_traits()['province_classes'].get(cls, ()))
    if not stems:
        raise ValueError(f'region-traits province_classes[{cls!r}] is empty: no peninsula to patch')
    plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    regs = [t for t in plan['regions'] if t.get('province_key', '').split('_province_', 1)[-1] in stems]
    if not regs:
        raise ValueError(f'no design region in a {cls!r} province ({sorted(stems)})')
    with np.load(A / 'analysis/west-real-terrain.npz') as z:
        bx0, by0 = int(z['box'][0]), int(z['box'][2])
        off = []
        for t in regs:
            x, zz = cw.display_position(*t['town'])
            tx, ty = int(x / (cw.DX / 2)) - bx0, int(zz / (cw.DZ / 2)) - by0
            lat, lon = float(z['lat'][ty, tx]), float(z['lon'][ty, tx])
            if not on_peninsula(lon, lat - k['peninsula_margin'], k['peninsula']):
                off.append((t['region'], round(lon, 2), round(lat, 2)))
    if off:
        raise ValueError(f'{cls!r} class towns off the peninsula (lon, lat): {off}')
    if map_cells is None:
        F = map_hex.read(A / 'work/final-map/map.hex')
        map_cells, map_names = F['cells'], F['land_regions'] + F['sea_regions']
    F = {'cells': map_cells}
    names = list(map_names)
    ids = map_hex.region_ids(F['cells'])
    want = sorted(t['region'] for t in regs)
    missing = [n for n in want if n not in names]
    if missing:
        raise ValueError(f'class regions not in the final map: {missing}')
    m = np.isin(ids, [names.index(n) for n in want]) & ((F['cells'][..., 0] & 3) != 1)
    nq0, nr0 = cw.PAD_LEFT, cw.PAD_BOTTOM
    nq1, nr1 = nq0 + cw.NATIVE_W, nr0 + cw.NATIVE_H
    inside = np.zeros(m.shape, bool)
    inside[nr0:nr1, nq0:nq1] = True
    rr, qq = np.nonzero(m & inside)
    if not len(qq):
        raise ValueError('class regions have no land inside the native rectangle')
    qmin, rmin = int(qq.min()), int(rr.min())
    return {'hex_win': (qmin + k['win_west'], nq1, rmin + k['win_south'], nr1),
            'edges': (float(qmin + k['west_edge']), float(nr1 - k['north_edge_below_top']), float(nq1 - k['east_edge_inside'])),
            'regions': want}


def coast_warp(land, dem_land, x0, ndimage, r0):
    """Per tile row, the DEM column for each window column: the projection puts the real
    coasts up to 30 hexes off the map's (vanilla) coasts (DEM east coast 5-26 hexes east of
    the map in the south, Hamgyong 15-30 east in the north), which cut the Taebaek and the
    Kaema ranges off. Each row maps the map's west/east coast onto the DEM's (linear
    between, shifted outside); offsets are median-filtered and smoothed over rows and fade
    to zero north of the peninsula. dem_land: DEM land (e > 0) on canvas columns x0-WARP_REACH..; r0: the window's
    first hex row.
    """
    rows, cols = land.shape
    reach = WARP_REACH
    oW = np.zeros(rows)
    oE = np.zeros(rows)
    Wm = np.zeros(rows)
    Em = np.full(rows, cols - 1.)
    ok = np.zeros(rows, bool)
    for y in range(rows):
        xs = np.flatnonzero(land[y])
        if len(xs) < 20:
            continue
        # main landmass of the row: the longest run of map land (islands ignored)
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 6) + 1)
        run = max(runs, key=len)
        wm, em = run[0], run[-1]
        d = np.flatnonzero(dem_land[y])                       # DEM land, in dem_land columns
        if not len(d):
            continue
        druns = [r_ for r_ in np.split(d, np.flatnonzero(np.diff(d) > 6) + 1) if len(r_) >= 16]
        if not druns:
            continue
        # the DEM landmass overlapping the map's most: its coasts are the anchors
        best = max(druns, key=lambda r_: min(r_[-1], em + reach) - max(r_[0], wm + reach))
        Wm[y], Em[y] = wm, em
        oE[y] = (best[-1] - reach - em) if em < cols - 3 else 0.   # land runs on past the window: no east coast
        oW[y] = (best[0] - reach - wm) if wm > 2 else 0.            # land runs on past the window: no west coast
        ok[y] = True
    for o in (oW, oE):
        o[~ok] = 0
        o[:] = ndimage.median_filter(o, 31, mode='nearest')
        o[:] = np.clip(o, -WARP_MAX, WARP_MAX)
        o[:] = ndimage.gaussian_filter1d(o, 12, mode='nearest')
    fade = 1 - np.clip((np.arange(rows) + 2 * r0 - 2 * WARP_NORTH[0]) / (2 * (WARP_NORTH[1] - WARP_NORTH[0])), 0, 1)
    oW *= fade
    oE *= fade
    Wm = ndimage.gaussian_filter1d(np.where(ok, Wm, np.nan_to_num(Wm)), 3, mode='nearest')
    Em = ndimage.gaussian_filter1d(np.where(ok, Em, cols - 1), 3, mode='nearest')
    xx = np.arange(cols)[None, :].astype(np.float64)
    t = np.clip((xx - Wm[:, None]) / np.maximum(Em - Wm, 1)[:, None], 0, 1)
    return xx + oW[:, None] * (1 - t) + oE[:, None] * t, oW, oE


def cold_line(lat):
    """Metres above which the forest is conifer (biome_profiles REGIONS korea_forests cold_line: off ~400 m on the
    Yalu; on ~1130 m at 41 N, the real Korean pine / broadleaf to spruce-fir line)."""
    base, per_lat, lat0, lo, hi = FORESTS['cold_line']
    return np.clip(base - per_lat * (lat - lat0), lo, hi)


def build(preview=False, out=None, preview_dir=None, map_cells=None, map_names=None):
    """Writes the patch cache (default OUT). map_cells / map_names: the map to build from (bake_map step 10 passes the
    map it is building and never the final map.hex); default (CLI) the workspace's final map.hex."""
    from scipy import ndimage
    import frontier_terrain as ft
    import map_hex
    from terrain_noise import fbm
    win = window(None, map_cells, map_names)
    q0, q1, r0, r1 = win['hex_win']
    EDGE_WEST, EDGE_NORTH, EDGE_EAST = win['edges']
    x0, x1, y0, y1 = 2 * q0, 2 * q1, 2 * r0, 2 * r1
    ty, tx = np.mgrid[y0:y1, x0:x1]
    hq, hr = cw.tile_hex(tx, ty)
    q, r = tx / 2., ty / 2.
    src = A / 'analysis/west-real-terrain.npz'
    z = np.load(src)
    bx0, bx1, by0, by1 = (int(v) for v in z['box'])
    assert bx0 <= x0 and x1 <= bx1 and by0 <= y0 and y1 <= by1, 'west-real-terrain box does not cover the Korea window'
    sl = np.s_[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    e = z['elevation'][sl].astype(np.float32)
    lat = z['lat'][sl].astype(np.float32)
    bio = z['biome'][sl]
    names = [str(n) for n in z['biome_names']]
    van = vanilla_tile_height(tx, ty)

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
    # Land in the final map AND in vanilla: former-sea land (north_okjeo's new coast) has a
    # vanilla sea-floor datum and its own polish relief; it stays out like the sea.
    land = ((F['cells'][hr, hq, 0] & 3) != 1) & ((vcells[..., 0] & 3) != 1) & inside

    # DEM aligned to the map's coasts (coast_warp), sampled per row.
    R = WARP_REACH
    xa0, xa1 = max(bx0, x0 - R), min(bx1, x1 + R)
    dem_land = np.zeros((y1 - y0, x1 - x0 + 2 * R), bool)
    dem_land[:, xa0 - (x0 - R):xa1 - (x0 - R)] = z['elevation'][y0 - by0:y1 - by0, xa0 - bx0:xa1 - bx0] > 0
    xd, warp_w, warp_e = coast_warp(land, dem_land, x0, ndimage, r0)
    xs_ = np.clip(np.rint(xd).astype(np.int64) + x0 - bx0, 0, bx1 - bx0 - 1)
    ys_ = (ty - by0)
    e = z['elevation'][ys_, xs_].astype(np.float32)
    lat = z['lat'][ys_, xs_].astype(np.float32)
    bio = z['biome'][ys_, xs_]
    raw = (cw.LEVEL + ft.to_relief(np.maximum(e, LAND_MIN_M))).astype(np.float32)

    # Height weight: jittered edges, protected / plot / sea clips.
    d = np.minimum(q - EDGE_WEST, EDGE_NORTH - r) + EDGE_JITTER * fbm(q, r, (0, 0), 12, 3, seed=311)
    d = np.minimum(d, EDGE_EAST - q)                 # unjittered: never inside the native rim
    hw = smoothstep(d / BAND)
    pd = ndimage.distance_transform_edt(~protected) / 2.
    hw *= np.clip(pd / PROTECT_HEXES, 0, 1)
    hw[plots | ~land] = 0
    hw = hw.astype(np.float32)
    dsea = (ndimage.distance_transform_edt(land) / 2.).astype(np.float32)   # hexes to the nearest sea cell (0 at sea)
    w = (hw * smoothstep((dsea - COAST_KEEP[0]) / (COAST_KEEP[1] - COAST_KEEP[0]))).astype(np.float32)

    # Relief: vanilla's land datum + the DEM above its own datum (raw + smoothed vanilla-minus-DEM
    # offset), ranges lifted, detail calmed; capped by a coastal rise and vanilla at the shore.
    m = land.astype(np.float64)

    def gland(a, sigma):   # land-only (normalised) Gaussian
        den_ = ndimage.gaussian_filter(m, sigma, mode='nearest')
        return ndimage.gaussian_filter(m * a, sigma, mode='nearest') / np.maximum(den_, 1e-6)
    datum_raw, datum_van = gland(raw, OFFSET_SIGMA), gland(van, OFFSET_SIGMA)
    raw_m = gland(raw, RANGE_SIGMA)
    hh = datum_van + RANGE_GAIN * (raw_m - datum_raw) + DETAIL_GAIN * (raw - raw_m)
    cap = van + COAST_RISE * np.maximum(dsea - COAST_FOOT, 0)
    hh = (hh + cap - np.sqrt((hh - cap) ** 2 + .3 ** 2)) / 2           # smooth min
    hh = (hh + LAND_FLOOR + np.sqrt((hh - LAND_FLOOR) ** 2 + .3 ** 2)) / 2   # smooth max (no crease)
    c = 1 - smoothstep((dsea - COAST_BLEND[0]) / (COAST_BLEND[1] - COAST_BLEND[0]))
    hh = (hh * (1 - c) + van * c).astype(np.float32)

    # Towns: plots levelled to the real relief, eased into it.
    import hehuang_terrain
    plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    towns = []
    for t in plan['regions']:
        tq, tr = t['town']
        cx, cy = cw.display_position(tq, tr)
        cx, cy = cx / cw.DX * 2 - x0, cy / cw.DZ * 2 - y0
        tw = hehuang_terrain.town_weight(hw, tq, tr, x0, y0)   # the town hex's own tiles (not round(display): its east neighbour)
        if tw <= .5:
            continue
        disc = (np.hypot(tx - x0 - cx, ty - y0 - cy) <= 6) & land
        towns.append(dict(region=t['region'], name=t['name'], town=[tq, tr], capital=bool(t['is_capital']), cx=float(cx), cy=float(cy),
                          w=tw, median=float(np.median(hh[disc])), vanilla=float(np.median(van[disc])),
                          elevation_m=float(np.median(e[disc]))))
    if towns:
        p = np.array([(t['cx'], t['cy']) for t in towns])
        h0 = np.array([t['median'] for t in towns])
        g = np.exp(-((p[:, None] - p[None]) ** 2).sum(-1) / (2 * (2 * TOWN_CLUSTER) ** 2))
        lev = (g @ h0) / g.sum(1)
        num = np.zeros(hh.shape, np.float64)
        den = np.zeros(hh.shape, np.float64)
        flat = np.zeros(hh.shape, np.float32)
        for t, v in zip(towns, lev):
            # A shore town stays under the coastal rise (no plateau standing over the sea).
            ccap = float(cap[int(round(t['cy'])), int(round(t['cx']))])
            t['coast_cap'] = ccap
            t['level'] = float(max(min(v, ccap), MIN_TOWN_HEIGHT))
            dd = np.hypot(tx - x0 - t['cx'], ty - y0 - t['cy']) / 2.   # hexes
            full, zero = TOWN_FLAT[t['capital']]
            cc = 1 - smoothstep((dd - full) / (zero - full))
            gk = np.exp(-dd ** 2 / (2 * 5. ** 2))
            num += gk * t['level']
            den += gk
            flat = np.maximum(flat, cc)
        mix = np.where(den > 1e-6, num / np.maximum(den, 1e-6), hh)
        before = hh.copy()
        hh = np.where(land, hh * (1 - flat) + mix * flat, hh).astype(np.float32)
        # --- FIX2 KOREA GROUND --- coastal cap again outside the plots (smooth min, as above)
        plot_keep = np.zeros(hh.shape, np.float32)
        for t in towns:
            dd = np.hypot(tx - x0 - t['cx'], ty - y0 - t['cy']) / 2.
            plot_keep = np.maximum(plot_keep, 1 - smoothstep((dd - TOWN_COAST_PLOT) / 1.5))
        cap2 = van + TOWN_COAST_RISE * np.maximum(dsea - COAST_FOOT, 0) + plot_keep * 10.
        hh = np.where(land, hh * (1 - flat) + flat * (hh + cap2 - np.sqrt((hh - cap2) ** 2 + .15 ** 2)) / 2, hh).astype(np.float32)
        hh = np.where(land, hh * (1 - c) + van * c, hh).astype(np.float32)
        # --- end FIX2 KOREA GROUND
        for t in towns:
            dd = np.hypot(tx - x0 - t['cx'], ty - y0 - t['cy']) / 2.
            ring = land & (dd <= TOWN_FLAT[t['capital']][1])
            t['cut_fill_max'] = float(np.abs(hh - before)[ring].max())
            gy_, gx_ = np.gradient(hh)
            t['max_slope_in_ease'] = float((np.hypot(gx_, gy_) * 2)[ring].max())   # relief units per hex

    # Ground classes.
    gy, gx = np.gradient(hh)
    slope = np.hypot(gx, gy) * 2                     # relief units per hex
    n = fbm(q, r, (0, 0), 6, 3, seed=313)
    ed = e + 160 * n                                 # dithered elevation: class lines meander
    cl = cold_line(lat)
    cls = np.full(hh.shape, 11, np.uint8)
    cls[ed > cl - FORESTS['mixed_band']] = 12
    cls[ed > cl] = 13
    low = np.isin(bio, [names.index(k) for k in ('steppe', 'grass', 'wetland', 'paddy') if k in names]) & (ed < 250)
    cls[((ed < 90) & (slope < .35)) | low] = 10
    cls[((ed > 1850) | ((slope > 1.5) & (e > 700)))] = 14
    cls[~land] = 255
    # Forest towns (as polish FOREST_TOWN): a town whose 3-8 hex ring is mostly forest
    # stands on the forest floor like vanilla's forest towns: no open (grass) disc under it,
    # which the levelled flat lowland would otherwise draw, and no donor ground there.
    forest_floor = np.zeros(hh.shape, np.float32)
    for t in towns:
        dd = np.hypot(tx - x0 - t['cx'], ty - y0 - t['cy']) / 2.
        ring = land & (dd >= FOREST_TOWN[0]) & (dd < FOREST_TOWN[1])
        fshare = float(np.isin(cls[ring], FOREST_CLASSES).mean()) if ring.any() else 0.
        t['forest_ring_share'] = fshare
        t['forest_town'] = fshare >= PATCH_FOREST_TOWN   # FIX2 KOREA GROUND (was FOREST_TOWN[2])
        if t['forest_town']:
            dd = dd + FOREST_DISC_JITTER * fbm(q, r, (0, 0), 3, 3, seed=317)   # FIX2 KOREA GROUND: frayed disc
            vals, cnt = np.unique(cls[ring & np.isin(cls, FOREST_CLASSES)], return_counts=True)
            disc = land & (dd < FOREST_FLOOR[1]) & (cls == 10)
            cls[disc] = vals[np.argmax(cnt)]
            forest_floor = np.maximum(forest_floor, land * (1 - smoothstep((dd - FOREST_FLOOR[0]) / (FOREST_FLOOR[1] - FOREST_FLOOR[0]))))
    # --- FIX2 KOREA GROUND --- soft, frayed donor mask (was gaussian_filter(..., 1.5))
    donor_mask = ndimage.gaussian_filter(np.isin(cls, DONOR_CLASSES).astype(np.float32), DONOR_SOFT)
    donor_mask = np.clip((donor_mask - .5) * 1.6 + .5 + DONOR_FRAY[0] * fbm(q, r, (0, 0), DONOR_FRAY[1], 4, seed=318), 0, 1)
    donor_mask[~land] = ndimage.gaussian_filter(np.isin(cls, DONOR_CLASSES).astype(np.float32), DONOR_SOFT)[~land]
    # --- end FIX2 KOREA GROUND

    hexmask = np.zeros((r1 - r0, q1 - q0), bool)
    hexmask[(hr - r0)[hw > 0], (hq - q0)[hw > 0]] = True
    sha = hashlib.sha256()
    for p_ in (src, None if map_given else A / 'work/final-map/map.hex', cw.DESIGN_PATH, Path(__file__)):
        sha.update(np.ascontiguousarray(F['cells']).tobytes() if p_ is None else p_.read_bytes())   # the map built, when given
    core = w >= .999
    gy0, gx0 = np.gradient(van)
    stats = dict(version=VERSION, coast_warp_hexes={'west': [float(v) for v in np.percentile(warp_w / 2, [0, 50, 100])], 'east': [float(v) for v in np.percentile(warp_e / 2, [0, 50, 100])],
                                                    'east_by_row': {int(r0 + i / 2): round(float(warp_e[i] / 2), 1) for i in range(0, len(warp_e), 20)}}, hexes_height_weight_gt0=int(hexmask.sum()), hexes_ground_weight_gt0=int(len(set(zip(hq[w > 0].tolist(), hr[w > 0].tolist())))),
                 tiles_full=int((hw >= .999).sum()), protected_cells_in_window=int(protected.sum()), plot_cells_in_window=int(plots.sum()),
                 relief_minus_vanilla_on_land={'mean_abs': float(np.abs(hh - van)[hw > 0].mean()), 'max_abs': float(np.abs(hh - van)[hw > 0].max())},
                 relief_p5_p50_p95={'vanilla': [float(v) for v in np.percentile(van[hw > .99], [5, 50, 95])], 'patch': [float(v) for v in np.percentile(hh[hw > .99], [5, 50, 95])]},
                 mean_slope_per_hex={'vanilla': float((np.hypot(gx0, gy0) * 2)[hw > .99].mean()), 'patch': float(slope[hw > .99].mean())},
                 class_share_core={CLASS_NAMES[k]: float(((cls == k) & core).sum() / max(1, core.sum())) for k in CLASS_NAMES},
                 towns=towns)
    out = Path(out) if out else OUT
    import hehuang_terrain
    np.savez_compressed(out, version=VERSION, sha=sha.hexdigest(), design_sha=hehuang_terrain.design_sha((q0, q1, r0, r1), plan),
                        inputs_sha=json.dumps(hehuang_terrain.input_sha((q0, q1, r0, r1), None, F['cells'], fn)), build_rules_sha=rules_sha(), rules_sha=_bp.korea_forests_sha(), box=np.array([x0, x1, y0, y1]), hex_box=np.array([q0, q1, r0, r1]),
                        weight=w.astype(np.float16), height_weight=hw.astype(np.float16), hex_mask=hexmask, relief=hh.astype(np.float32),
                        vanilla=van, ground=cls, class_names=np.array([CLASS_NAMES[k] for k in sorted(CLASS_NAMES)]),
                        class_climate=np.array(sorted(CLASS_CLIMATE.items()), np.int32), steppe_classes=np.array(STEPPE_CLASSES, np.int32),
                        donor_mask=donor_mask.astype(np.float16), forest_floor=forest_floor.astype(np.float16), species=json.dumps({str(k): v for k, v in SPECIES.items()}), towns=json.dumps(towns, ensure_ascii=False))
    out.with_suffix('.json').write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k != 'towns'}, ensure_ascii=False))
    for t in towns:
        print(' town', t['region'], t['name'], 'w %.2f vanilla %.2f median %.2f level %.2f elev %d m cut/fill %.2f slope %.2f'
              % (t['w'], t['vanilla'], t['median'], t['level'], t['elevation_m'], t['cut_fill_max'], t['max_slope_in_ease']))
    print('towns levelled', len(towns), 'forest towns (no open disc)', [t['region'] for t in towns if t['forest_town']])
    if preview:
        _preview(hh, van, hw, w, cls, land, x0, y0, plan, preview_dir)
    print('->', out)


def _preview(hh, van, hw, w, cls, land, x0, y0, plan, outdir=None):
    from PIL import Image, ImageDraw
    outdir = Path(outdir) if outdir else Path(__file__).resolve().parents[1] / 'output/offline-view/korea-after'
    outdir.mkdir(parents=True, exist_ok=True)
    pal = {10: (176, 170, 96), 11: (70, 110, 50), 12: (55, 92, 55), 13: (35, 70, 45), 14: (150, 140, 120)}

    def shade(h):
        gy, gx = np.gradient(h)
        return np.clip(1 + (gx - gy) * 1.6, .4, 1.6)[..., None]

    def save(img, name):
        im = Image.fromarray(np.clip(img[::-1], 0, 255).astype(np.uint8)).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST)
        d = ImageDraw.Draw(im)
        H = img.shape[0]
        for t in plan['regions']:
            cx, cy = cw.display_position(*t['town'])
            ix, iy = cx / cw.DX * 2 - x0, cy / cw.DZ * 2 - y0
            if 0 <= ix < img.shape[1] and 0 <= iy < H:
                X, Y = ix * 2, (H - 1 - iy) * 2
                d.rectangle([X - 3, Y - 3, X + 3, Y + 3], outline=(255, 0, 0))
                d.text((X + 5, Y - 6), t['region'].replace('tw3k_hist_', '').replace('tw3k_exp_', ''), fill=(255, 255, 0))
        im.save(outdir / name)

    lut = np.zeros((256, 3), np.float32)
    lut[255] = (30, 50, 90)
    for k, v in pal.items():
        lut[k] = v
    img = lut[cls] * shade(hh)
    vcol = np.array([95, 105, 50], np.float32) * shade(van)
    img = np.where(land[..., None], img * w[..., None] + vcol * (1 - w[..., None]), (30, 50, 90))
    edge = ((hw > 0) & ~np.roll(hw > 0, 1, 0)) | ((hw > 0) & ~np.roll(hw > 0, 1, 1))
    img[edge & land] = (255, 0, 0)
    save(img, 'patch-ground-classes.png')
    for name, h in (('patch-relief.png', hh * hw + van * (1 - hw)), ('patch-relief-before.png', van)):
        t = np.clip((h - .5) / 7, 0, 1)[..., None]
        im = (np.array([70, 110, 60]) * (1 - t) + np.array([235, 225, 200]) * t) * shade(h)
        save(np.where(land[..., None], im, (30, 50, 90)), name)


def load():
    """The patch npz as a dict, or None."""
    if not OUT.exists():
        return None
    z = np.load(OUT)
    out = {k: z[k] for k in z.files}
    out['towns'] = json.loads(str(out['towns']))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['build'])
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--out', help='npz path (default analysis/korea-patch.npz; a dry run elsewhere)')
    ap.add_argument('--preview-dir')
    a = ap.parse_args()
    if a.action == 'build':
        build(a.preview, a.out, a.preview_dir)


if __name__ == '__main__':
    main()

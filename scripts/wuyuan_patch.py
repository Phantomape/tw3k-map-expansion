"""Wuyuan north bank real-geography terrain patch: data preparation (third patch of hehuang_terrain).

Run with Anaconda (scipy, h5py):  <tw3k python> scripts/wuyuan_patch.py build [--out NPZ]
Writes analysis/wuyuan-patch.npz/.json, which hehuang_terrain.load() combines with the Hehuang and Korea patches for
polish_campaign_terrain (the one writer of this ground's height) and the native audits.

Why (release run C2, coordinator / user 2026-11-01): vanilla draws the Yin Shan foot on its 3k_main_reg_non_playable
filler north of the Shuofang Yellow River 2 to 3.5 relief units above the real ground (ETOPO through
frontier_terrain.to_relief; 麻池古城 1031 m reads 3.70, the vanilla slope there 5.7 to 7.5). Polish levelled Jiuyuan
(tw3k_rb_wuyuan_capital, hex 1218,1000) near its real height and so cut a three-sided pit into that slope (up to 4.49).
c21 laid the real plain between the vanilla bank and the Daqing Shan foot (40.70 N) only; its 4-hex north band climbed back to the vanilla Yin Shan (still 2-3.7 too high) and read as a wall (user 2026-11-01). c22: the whole Wuyuan stretch of the Yin Shan to 41.8 N, where vanilla and the real ground already agree (north edge row -0.14 on average), so the patch ends where nothing is left to blend. The Tumed / Hohhot plain east of q 1251 (vanilla also 1-2.3 too high, its east end not yet found) is a follow-up.

  window    BOX q 1176-1251, r 995-1031 (hexes, ends exclusive below); core CORE q 1180-1247, r 995-1027. Height weight 1
            from BAND hexes inside the box edge, smoothstep to 0 at it (distance to the filler land outside the box,
            so the south edge also eases where the filler runs on south of r 995); 0 on vanilla playable land and
            within PROTECT_HEXES of it, on vanilla town plots, on water and on every hex vanilla does not give to the
            filler. Ground weight 0: the ground classes, climate, donor and cache stay polish's own, and the box keeps
            the vanilla filler's materials (diversity, c22 review 2026-11-02: the landform tiles give this filler
            ground_w 0, so the rule table does not repaint it; global_blend identical to C2 in the box + 2.5 hexes).
            Materials following the real landform: SINGLE / ground step 2, after the release.
  relief    LEVEL + to_relief(ETOPO) (west-real-terrain `elevation`); within BANK_SEAM hexes of the vanilla bank
            (vanilla playable land or water) the vanilla-minus-DEM offset along the bank is added and fades out, so the
            filler meets the unchanged bank at its height and reaches the real plain BANK_SEAM hexes inland.
  towns     design towns at weight > .5 levelled to the median patch relief round them (at least MIN_TOWN_HEIGHT),
            flat to TOWN_FLAT[0] and eased back over to TOWN_FLAT[1] hexes.
Frozen criteria (coordinator 2026-11-01, checked by stage_invariants wuyuan_*, fixed before the c22 dry build): CORE_MAX_DIFF; the band no steeper than the vanilla filler round the box (wuyuan_band_slope, no exemption); town audit 2; town cut <= 120 cells, <= 3.5.
Appendix (user 2026-11-01, after the c22 dry build; limit .5 unchanged): the core rule skips the town flats, the cells\nwhere the build levels the ground to a town's height (npz town_flat > 0; core_checked); the build reports the cells\nexcluded and their largest difference.
Appendix 2 (user 2026-11-01, after the c22 trial; threshold 0 unchanged): wuyuan_band_slope exempt for release C3 up to
.0366 (stage_invariants.EXEMPT): the east band on the river bank (q 1246-1248, r 996-997) meets the Tumed plain, not
in this repaint.
TW3K_WUYUAN=0 switches the patch off for diagnostics only, never as a fallback: the wuyuan_* invariants and town_plateau_check fail then (integrator 2026-11-01).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import campaign_world as cw
from hehuang_patch import smoothstep, vanilla_tile_height

A = cw.WORKSPACE
OUT = A / 'analysis/wuyuan-patch.npz'
VERSION = 3                        # 2: c22 box (Yin Shan to 41.8 N); 3: npz town_flat (the core rule's exclusion)
BOX = (1176, 1252, 995, 1032)      # q0, q1, r0, r1 (ends exclusive): the declared scope, core plus the transition band
CORE = (1180, 1248, 995, 1028)     # the core: vanilla bank (r 995) to 41.8 N (r 1027), Jiuyuan and its pasture inside
BAND = 4.                          # hexes: weight 0 -> 1 from the box edge inward
PROTECT_HEXES = 1.                 # hexes from vanilla playable land (Korea and Hehuang keep 8; coordinator 2026-11-01)
BANK_SEAM = 4.                     # hexes: vanilla bank height -> real plain
BANK_RING = 1.                     # hexes: filler next to the bank whose vanilla-minus-DEM offset anchors the seam
BANK_SIGMA = 8                     # tile cells: smoothing of that offset along the bank
TOWN_FLAT = {True: (3.5, 8.), False: (3.5, 8.)}   # capital?: (full, zero) radius in hexes (>= plot + apron; 4.5 held
                                                  # Jiuyuan's level 0.51 under the rising Daqing Shan foot, dry run 1)
MIN_TOWN_HEIGHT = 1.0              # polish_campaign_terrain.MIN_TOWN_HEIGHT
CORE_MAX_DIFF = .5                 # frozen: |patch relief - (LEVEL + to_relief(ETOPO))| in the core outside the bank seam
FILLER = '3k_main_reg_non_playable'


def rules_sha():
    """Fingerprint of this build's rule constants (npz 'build_rules_sha'; hehuang_terrain.check_sources)."""
    import hehuang_terrain
    return hehuang_terrain.build_rules_sha(sys.modules[__name__])


def _fields(map_cells=None, map_names=None, grow=0):
    """Tile-grid fields over BOX grown by `grow` hexes: (tx, ty, hq, hr, land, filler, protected, plots, water)."""
    import map_hex
    q0, q1, r0, r1 = BOX[0] - grow, BOX[1] + grow, BOX[2] - grow, BOX[3] + grow
    ty, tx = np.mgrid[2 * r0:2 * r1, 2 * q0:2 * q1]
    hq, hr = cw.tile_hex(tx, ty)
    if map_cells is None:
        F = map_hex.read(A / 'work/final-map/map.hex')
        map_cells, map_names = F['cells'], F['land_regions'] + F['sea_regions']
    fn = list(map_names)
    fid = map_hex.region_ids(map_cells)
    V = map_hex.read(cw.VANILLA_MAP)
    vn = V['land_regions'] + V['sea_regions']
    vid = map_hex.region_ids(V['cells'])
    vq, vr = hq - cw.PAD_LEFT, hr - cw.PAD_BOTTOM
    inside = (vq >= 0) & (vq < cw.NATIVE_W) & (vr >= 0) & (vr < cw.NATIVE_H)
    vqc, vrc = np.clip(vq, 0, cw.NATIVE_W - 1), np.clip(vr, 0, cw.NATIVE_H - 1)
    vname = np.array(vn + ['?'])[np.where(vid >= 0, vid, len(vn))][vrc, vqc]
    fname = np.array(fn + ['?'])[np.where(fid >= 0, fid, len(fn))][hr, hq]
    vcells = V['cells'][vrc, vqc]
    water = ((map_cells[hr, hq, 0] & 3) == 1) | ((vcells[..., 0] & 3) == 1) | ~inside
    land = ~water
    filler = inside & (vname == FILLER)
    protected = inside & ~filler & land           # vanilla playable land (the bank), whatever the mod map calls it now
    plots = inside & (((vcells[..., 2] >> 4) > 0) | ((vcells[..., 3] & 1) > 0))
    return tx, ty, hq, hr, land, filler, protected, plots, water, fname


def build(out=None, map_cells=None, map_names=None, plan=None, save=True):
    """Writes the patch cache (default OUT). map_cells / map_names: the map to build from (bake_map step 10 passes the
    map it is building); default (CLI) the workspace's final map.hex. plan: the region design (default the workspace's).
    save=False writes nothing and returns {box, height_weight, relief, towns} (town_plateau_check's estimate)."""
    from scipy import ndimage
    import frontier_terrain as ft
    import map_hex
    import hehuang_terrain
    G = int(BAND) + 3                              # margin: the filler and the bank outside the box
    tx, ty, hq, hr, land, filler, protected, plots, water, fname = _fields(map_cells, map_names, G)
    q0, q1, r0, r1 = BOX
    x0, x1, y0, y1 = 2 * q0, 2 * q1, 2 * r0, 2 * r1
    gx0, gy0 = 2 * (q0 - G), 2 * (r0 - G)
    crop = np.s_[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]
    in_box = np.zeros(land.shape, bool)
    in_box[crop] = True

    src = A / 'analysis/west-real-terrain.npz'
    z = np.load(src)
    bx0, bx1, by0, by1 = (int(v) for v in z['box'])
    assert bx0 <= tx.min() and tx.max() < bx1 and by0 <= ty.min() and ty.max() < by1, 'west-real-terrain does not cover the box'
    e = z['elevation'][ty - by0, tx - bx0].astype(np.float32)
    raw = (cw.LEVEL + ft.to_relief(np.maximum(e, 0))).astype(np.float32)
    van = vanilla_tile_height(tx, ty)

    # Height weight: distance to the filler land left outside the box, the bank protection, plots / water / non-filler.
    outside = land & filler & ~in_box
    d = ndimage.distance_transform_edt(~outside) / 2.
    hw = smoothstep(d / BAND)
    pd = ndimage.distance_transform_edt(~protected) / 2.
    hw *= np.clip(pd / PROTECT_HEXES, 0, 1)
    hw[plots | ~land | ~filler | ~in_box] = 0
    hw = hw.astype(np.float32)

    # Relief: the real plain, eased onto the vanilla bank height over BANK_SEAM hexes.
    bank = protected | water
    dbank = (ndimage.distance_transform_edt(~bank) / 2.).astype(np.float32)
    ring = (land & filler & (dbank <= BANK_RING)).astype(np.float64)
    den = ndimage.gaussian_filter(ring, BANK_SIGMA, mode='nearest')
    off = ndimage.gaussian_filter(ring * (van - raw), BANK_SIGMA, mode='nearest') / np.maximum(den, 1e-6)
    off = np.where(den > 1e-4, off, 0.)
    fade = 1 - smoothstep((dbank - .5) / BANK_SEAM)
    hh = (raw + off * fade).astype(np.float32)

    # Towns: plots levelled to the patch relief, eased into it.
    if plan is None:
        plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    towns = []
    for t in plan['regions']:
        tq, tr = t['town']
        tw = hehuang_terrain.town_weight(hw, tq, tr, gx0, gy0)
        if tw <= .5:
            continue
        cx, cy = cw.display_position(tq, tr)
        cx, cy = cx / cw.DX * 2 - gx0, cy / cw.DZ * 2 - gy0
        dd = np.hypot(tx - gx0 - cx, ty - gy0 - cy) / 2.
        disc = (dd <= 3) & land
        towns.append(dict(region=t['region'], name=t['name'], town=[tq, tr], capital=bool(t['is_capital']), cx=float(cx), cy=float(cy),
                          w=tw, median=float(np.median(hh[disc])), vanilla=float(np.median(van[disc])), elevation_m=float(np.median(e[disc]))))
    flat = np.zeros(hh.shape, np.float32)
    if towns:
        num = np.zeros(hh.shape); den_t = np.zeros(hh.shape)
        for t in towns:
            t['level'] = float(max(t['median'], MIN_TOWN_HEIGHT))
            dd = np.hypot(tx - gx0 - t['cx'], ty - gy0 - t['cy']) / 2.
            full, zero = TOWN_FLAT[t['capital']]
            flat = np.maximum(flat, 1 - smoothstep((dd - full) / (zero - full)))
            gk = np.exp(-dd ** 2 / (2 * 5. ** 2))
            num += gk * t['level']; den_t += gk
        mix = np.where(den_t > 1e-6, num / np.maximum(den_t, 1e-6), hh)
        before = hh.copy()
        hh = np.where(land, hh * (1 - flat) + mix * flat, hh).astype(np.float32)
        for t in towns:
            dd = np.hypot(tx - gx0 - t['cx'], ty - gy0 - t['cy']) / 2.
            ring_t = land & (dd <= TOWN_FLAT[t['capital']][1])
            t['cut_fill_max'] = float(np.abs(hh - before)[ring_t].max())

    hw, hh, raw, van, dbank, land_c, filler_c, flat = (a[crop] for a in (hw, hh, raw, van, dbank, land, filler, flat))
    flat = flat.astype(np.float16).astype(np.float32)   # as the cache stores it: the build and the invariant see one mask
    if not save:   # the arrays exactly as the cache stores them (polish reads the cache)
        return {'box': (x0, x1, y0, y1), 'height_weight': hw.astype(np.float16).astype(np.float32), 'relief': hh.astype(np.float32),
                'towns': {t['region']: t for t in towns}}
    hq_c, hr_c = hq[crop], hr[crop]
    hexmask = np.zeros((r1 - r0, q1 - q0), bool)
    hexmask[(hr_c - r0)[hw > 0], (hq_c - q0)[hw > 0]] = True
    core = core_checked(hq_c, hr_c, hw, dbank, flat)
    flat_core = core_checked(hq_c, hr_c, hw, dbank, None) & (flat > 0)
    diff = np.abs(hh - raw)
    sha = hashlib.sha256()
    for p_ in (src, None if map_cells is not None else A / 'work/final-map/map.hex', cw.DESIGN_PATH, Path(__file__)):
        if p_ is None:
            sha.update(np.ascontiguousarray(map_cells).tobytes())
        else:
            sha.update(p_.read_bytes())
    if map_cells is None:
        F = map_hex.read(A / 'work/final-map/map.hex')
        map_cells, map_names = F['cells'], F['land_regions'] + F['sea_regions']
    stats = dict(version=VERSION, box=BOX, core=CORE, hexes_height_weight_gt0=int(hexmask.sum()), tiles_full=int((hw >= .999).sum()),
                 core_tiles_checked=int(core.sum()), core_max_abs_minus_real=float(diff[core].max()) if core.any() else None,
                 core_limit=CORE_MAX_DIFF, core_town_flat_tiles_excluded=int(flat_core.sum()),
                 core_town_flat_max_abs_minus_real=float(diff[flat_core].max()) if flat_core.any() else None,
                 relief_minus_vanilla={'mean_abs': float(np.abs(hh - van)[hw > 0].mean()), 'max_abs': float(np.abs(hh - van)[hw > 0].max())},
                 relief_p5_p50_p95={'vanilla': [float(v) for v in np.percentile(van[hw > .99], [5, 50, 95])],
                                    'patch': [float(v) for v in np.percentile(hh[hw > .99], [5, 50, 95])]},
                 towns=towns)
    out = Path(out) if out else OUT
    hb = (q0, q1, r0, r1)
    np.savez_compressed(out, version=VERSION, sha=sha.hexdigest(), design_sha=hehuang_terrain.design_sha(hb, plan),
                        inputs_sha=json.dumps(hehuang_terrain.input_sha(hb, None, map_cells, list(map_names))), build_rules_sha=rules_sha(),
                        box=np.array([x0, x1, y0, y1]), hex_box=np.array(hb),
                        weight=np.zeros(hw.shape, np.float16), height_weight=hw.astype(np.float16), hex_mask=hexmask,
                        relief=hh.astype(np.float32), real=raw.astype(np.float32), vanilla=van, bank_distance=dbank.astype(np.float16),
                        town_flat=flat.astype(np.float16),
                        ground=np.full(hw.shape, 255, np.uint8), class_names=np.array([], '<U1'),
                        class_climate=np.zeros((0, 2), np.int32), steppe_classes=np.zeros(0, np.int32),
                        towns=json.dumps(towns, ensure_ascii=False))
    out.with_suffix('.json').write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k != 'towns'}, ensure_ascii=False))
    for t in towns:
        print(' town', t['region'], t['name'], 'w %.2f vanilla %.2f median %.2f level %.2f elev %d m cut/fill %.2f'
              % (t['w'], t['vanilla'], t['median'], t['level'], t['elevation_m'], t['cut_fill_max']))
    print('->', out)


def core_checked(hq, hr, hw, dbank, flat):
    """Tile cells the CORE_MAX_DIFF rule checks: in CORE, at full height weight, more than BANK_SEAM + .5 hexes from the
    vanilla bank, and outside the town flats (flat > 0: where the build levels the ground to a town's height; user
    2026-11-01, after the c22 dry build: 6 of 7,618 core cells, all in the Jiuyuan and pasture flats, were .5-.62 off the
    rising real ground; the limit stays .5, the flats are held by the town audit and the cut check). flat None: no
    exclusion (reporting)."""
    m = core_mask(hq, hr) & (np.asarray(hw, np.float32) >= .999) & (np.asarray(dbank, np.float32) > BANK_SEAM + .5)
    return m if flat is None else m & ~(np.asarray(flat, np.float32) > 0)


def core_mask(hq, hr):
    """Tile cells whose hex lies in CORE."""
    return (hq >= CORE[0]) & (hq < CORE[1]) & (hr >= CORE[2]) & (hr < CORE[3])


def same_arrays(a, b):
    """Keys whose arrays differ between two patch npz files (np.savez_compressed stamps zip times, so the bytes of two
    builds differ; the contents must not)."""
    with np.load(a) as x, np.load(b) as y:
        keys = sorted(set(x.files) | set(y.files))
        return [k for k in keys if k not in x.files or k not in y.files or not np.array_equal(x[k], y[k])]


def check(path=None):
    """Rebuild into a temporary file and compare every array with the cache (integrator 2026-11-01: deterministic
    from the whitelisted inputs). Returns the differing keys ([] = reproducible)."""
    import tempfile
    path = Path(path) if path else OUT
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / 'wuyuan-patch.npz'
        build(tmp)
        return same_arrays(path, tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['build', 'check'])
    ap.add_argument('--out', help='npz path (default analysis/wuyuan-patch.npz; a dry run elsewhere; check: the cache to check)')
    a = ap.parse_args()
    if a.action == 'build':
        build(a.out)
    else:
        bad = check(a.out)
        print('wuyuan patch reproducible' if not bad else f'wuyuan patch differs from a rebuild: {bad}')
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

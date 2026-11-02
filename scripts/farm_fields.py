"""Farmland (global_blend materials 8-12) around the authored towns, as vanilla lays it.

Vanilla 190 (dlc07) paints farm materials (farms_grain_1/2 = 8/9, farms_rice_1/2 = 10/11,
farms_tea_1 = 12) as a resource of the region, not around every town (offline survey of
the 201 vanilla towns, 2026-10-23, scratch farms-rocks/vfarm.py):
  * farm-resource regions (3k_resource_wood_farms_grain / _rice, _tea) carry ~14-19 farm
    hexes within 3 hexes of the town, ~52-58 within 6, ~110-125 within 10, ~180-200 within
    15 (the field-weighted mean distance is 8.4 hexes); cities 3.3 within 10 (84% none),
    livestock 9.6, lumber 5.0, mines/ports/salt ~0;
  * materials follow the climate: arid/temperate/cold fields are grain (grain_2 the outer
    rims of grain_1 dabs), subtropical fields rice (rice_1 + rice_2) and, in tea regions,
    tea alone; tea sits a little above the town (10-16% of the fields 0.1-1.0 above it);
  * fields are round brush dabs (1-2 hexes) merged into blobs (median 16 hexes, 90th
    percentile 271) on flat ground: farm share 0.085 below 0.05 world units per hex of
    slope, 0.076 to 0.1, 0.035 to 0.2, 0.017 to 0.3, 0.004 to 0.5, none above; within
    -0.5..+1.0 of the town height; thin next to river/lake hexes (0.007 within 1 hex vs
    0.066 beyond 8), denser along roads (0.073 within 4 hexes vs 0.018 beyond 8);
  * arid towns: almost none (Wuwei resource_2 0, Anding resource_1 46 hexes within 15).
So each authored town receives the field complex of a vanilla farm town of its climate
(its farm pixels in connected dabs reaching within FOOTPRINT_CORE hexes of the town),
turned by a random angle (and mirrored), scaled by the town's kind and climate and cut by
the vanilla slope/height/water rules with a smooth noise, so the cut edges stay blobby.

Called once per variant by polish_campaign_terrain.py (FARM FIELDS block) on the material
blend before it is written. Pure numpy. The offline CLI applies it to a copy of a staged
blend for previews:
  python farm_fields.py --preview OUT_DIR   (copies the stage blend there, paints it, reports)
"""
import json
import math
import os
import struct
import zlib

import numpy as np

import campaign_world as cw
from terrain_noise import fbm
import biome_profiles as _bp

_K = _bp.consts('farm_fields')   # values: biome_profiles.CONSTS['farm_fields']

FARM = np.arange(8, 13, dtype=np.uint8)
GRAIN, RICE, TEA = (8, 9), (10, 11), 12
FOOTPRINT_R = 16      # hexes of donor window read around the vanilla town
FOOTPRINT_CORE = 6    # a donor dab complex must reach this close to its town
# Vanilla farm towns (region key: dlc07 map) used as donors, by field kind. Picked from the
# survey: the farm regions whose fields are their own (not a neighbour's) and of one climate.
DONORS = _K['DONORS']
# Share of a donor's fields a town keeps, by town kind (vanilla means: farm towns ~180
# hexes within 15, cities ~8, the other resources about none) ...
KIND_SCALE = _K['KIND_SCALE']
# ... and by climate (tile climate 0 subtropical, 1 cold, 2 temperate, 3 arid). Arid
# fields also need irrigated ground (oasis belts, river valleys, the town's own oasis).
CLIMATE_SCALE = _K['CLIMATE_SCALE']
CITY_CLIMATE = _K['CITY_CLIMATE']     # cities: no cold fields
STEPPE_SCALE = _K['STEPPE_SCALE']   # pastoral steppe towns
ARID_OWN_OASIS = _K['ARID_OWN_OASIS']    # hexes around an arid town that count as irrigated
STEPPE_BIOMES = _bp.biome_set('ff_steppe')   # pastoral steppe towns (steppe, grass)
DRY_BIOMES = _bp.biome_set('ff_dry')         # no tea donors (tropical_dry, savanna, desert, steppe)
# Vanilla rules, as keep factors (relative to flat ground at the town's height).
SLOPE_KEEP = _K['SLOPE_KEEP']          # world units per hex
HEIGHT_KEEP = _K['HEIGHT_KEEP']        # above the town
TEA_SLOPE, TEA_HEIGHT = _K['TEA_SLOPE'], _K['TEA_HEIGHT']
BORDER_CLEAR, BORDER_FADE = 12, 20   # tile cells kept free of new fields on either side of the native border, then a fade
NOISE_GRAIN = 2.5     # hexes: noise cutting the fields
# Farm pixel mask (ground contrast fix §4): the tile masks (allowed ground, arid irrigation) are read bilinearly per
# pixel and cut at .5 with a jitter of this amplitude (second noise) inside the allowed tiles (the tile gate stays
# hard): a per-tile cut drew the field edges as tile squares. TW3K_FARM_PIXEL_MASK=0 restores the per-tile masks.
FARM_PIXEL_MASK = os.environ.get('TW3K_FARM_PIXEL_MASK', '0') == '1'
PIXEL_JITTER = .3


def kind_of(layout):
    layout = str(layout or '')
    if 'farm' in layout or 'tea' in layout:
        return 'farm'
    if layout.startswith('settlement_'):
        return 'city'
    return 'other'


def vanilla_towns():
    """{region: (q, r)} of vanilla towns (centroid of the primary slot hexes), native hexes."""
    import map_hex
    v = map_hex.read(cw.VANILLA_MAP)
    c = v['cells']
    slot = (c[..., 2] >> 4) == 1
    ids = map_hex.region_ids(c)
    names = v['land_regions'] + v['sea_regions']
    out = {}
    rr, qq = np.nonzero(slot)
    for i in np.unique(ids[rr, qq]):
        if 0 <= i < len(names):
            k = ids[rr, qq] == i
            out[names[i]] = (int(round(qq[k].mean())), int(round(rr[k].mean())))
    return out


def _components(mask):
    """8-connected labels of a small boolean array (numpy only: min-label propagation)."""
    h, w = mask.shape
    lab = np.where(mask, np.arange(h * w).reshape(h, w) + 1, 0).astype(np.int64)
    big = np.iinfo(np.int64).max
    while True:
        p = np.where(mask, lab, big)
        m = p.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    s = np.full_like(p, big)
                    s[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = p[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
                    m = np.minimum(m, s)
        m = np.where(mask, m, 0)
        if np.array_equal(m, lab):
            return lab
        lab = m


def donor_footprints(donor_blend, towns):
    """{donor region: (window R/G uint8 (2R+1, 2R+1, 2), field mask)} from the vanilla blend
    (north-first rows, 8 px per hex): its farm dabs connected to within FOOTPRINT_CORE hexes."""
    hh, ww = donor_blend.shape[:2]
    R = FOOTPRINT_R * 8
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.hypot(xx, yy) / 8
    out = {}
    for keys in DONORS.values():
        for k in keys:
            if k not in towns:
                continue
            q, r = towns[k]
            cx, cy = 8 * q, hh - 1 - int(round((r + q % 2 * .5) * 8))
            if not (R <= cx < ww - R and R <= cy < hh - R):
                continue
            win = np.ascontiguousarray(donor_blend[cy - R:cy + R + 1, cx - R:cx + R + 1])
            farm = np.isin(win[..., 0], FARM) & (d <= FOOTPRINT_R)
            lab = _components(farm)
            keep = np.unique(lab[farm & (d <= FOOTPRINT_CORE)])
            out[k] = (win, farm & np.isin(lab, keep[keep > 0]))
    return out


def _interp(v, table):
    return np.interp(v, table[0], table[1]).astype(np.float32)


def shore_band(water):
    """water grown one tile cell along rows and columns, edges clamped (np.roll wrapped the far map edge onto
    this one)."""
    w0 = np.asarray(water, bool)
    out = w0.copy()
    out[1:] |= w0[:-1]; out[:-1] |= w0[1:]; out[:, 1:] |= w0[:, :-1]; out[:, :-1] |= w0[:, 1:]
    return out


def filler_keep(cells, names, hy, hx):
    """(TH, TW) float32 field keep weight on the filler (revealed_filler.filler_hex: vanilla non-playable
    filler that is land now, playable or not; canvas), 0 elsewhere: 0 within BORDER_CLEAR tile cells (BORDER_CLEAR / 2
    hex steps, hex_components.hex_dilate rings) of vanilla playable land (vanilla_playable_hex), then a linear
    fade over BORDER_FADE cells, as the native border band. hy, hx: hex row / column per tile cell."""
    from hex_components import hex_dilate
    import revealed_filler
    rv = revealed_filler.filler_hex(cells, names)
    steps = (BORDER_CLEAR + BORDER_FADE + 1) // 2
    d = np.full(rv.shape, steps + 1, np.float64)
    grown = revealed_filler.vanilla_playable_hex()
    d[grown] = 0
    for k in range(1, steps + 1):
        nxt = hex_dilate(grown)
        d[nxt & ~grown] = k
        grown = nxt
    w = np.clip((2 * d - BORDER_CLEAR) / BORDER_FADE, 0, 1) * rv
    return w.astype(np.float32)[hy, hx]


def paint(cur, donor_blend, plan, biome, surface, alpha, physical, protected, plots, irrigated, real_biome=None,
          real_names=None, seed=0, edited=None, ignore=None, filler=None, candidate_sink=None, domains=None):
    """Paint fields into cur (north-first (oh, ow, 2) uint8 blend, 8 px per hex) in place.

    biome, surface, alpha, physical, protected, plots, irrigated: south-up tile grids
    (TH, TW): tile climate, final surface height, this variant's edit alpha, land, vanilla
    protected ground, town building plots, irrigated weight (oasis belts) in [0, 1].
    real_biome/real_names: per-hex ecoregion (strategic_ink) or None. edited: tile mask of native
    ground whose LF really changed (native_edit_mask.edited_tiles, no halo) or None: inside the
    native map fields go only there (audit_native_identity counts blend edits elsewhere).
    ignore: (oh, ow) bool mask of farm pixels the 'already farmed' rule does not count (the
    GROUND MATERIALS valley/oasis fields laid before this; they stay where no complex covers them).
    filler: filler_keep() (required): the filler takes fields as canvas, cleared and faded from vanilla
    playable land as the native border band.
    candidate_sink(box, slots, weights): prepare read-only proposals instead of
    writing cur. In that mode cur is the immutable inherited reference used only
    to avoid duplicating existing authored farms. Returns a report."""
    assert filler is not None, 'farm_fields.paint needs filler_keep() (filler)'

    oh, ow = cur.shape[:2]
    TH, TW = biome.shape
    towns = vanilla_towns()
    shift = (cw.PAD_LEFT, cw.PAD_BOTTOM)
    feet = donor_footprints(donor_blend, towns)
    # Slope (world units per hex) of the final surface, at tile resolution.
    gz, gx = np.gradient(surface.astype(np.float32))
    slope = np.hypot(gx, gz) * 2
    tx_, ty_ = np.meshgrid(np.arange(TW), np.arange(TH))
    border = np.maximum.reduce([cw.NATIVE_TILE_X0 - tx_, tx_ - (cw.NATIVE_TILE_X1 - 1),
                                cw.NATIVE_TILE_Y0 - ty_, ty_ - (cw.NATIVE_TILE_Y1 - 1)])
    inside = np.minimum.reduce([tx_ - cw.NATIVE_TILE_X0, cw.NATIVE_TILE_X1 - 1 - tx_,
                                ty_ - cw.NATIVE_TILE_Y0, cw.NATIVE_TILE_Y1 - 1 - ty_])
    # New fields go on canvas land and, inside the native map, on edited ground only
    # (audit_native_identity), never within BORDER_CLEAR of the native border (audit_native_edges).
    # The filler (filler > 0) is canvas, cleared from vanilla playable land (filler_keep).
    allowed = physical & ~protected & ~plots & ((border > 0) | ((alpha > .02) & (True if edited is None else edited)) | (filler > 0))
    # Fields fade in over BORDER_FADE cells past the cleared band, as a keep factor: the noise
    # cut then frays them; a hard limit drew a straight field edge (Lin Yi, offline view).
    border_w = np.clip((np.where(border > 0, border, inside) - BORDER_CLEAR) / BORDER_FADE, 0, 1).astype(np.float32)
    border_w *= np.where(border > 0, 1, np.clip(alpha / .25, 0, 1)).astype(np.float32)   # inside: with the edit's own ramp
    border_w = np.where(filler > 0, filler, border_w).astype(np.float32)                   # filler: its own band
    if domains is not None:
        allowed = physical & ~protected & ~plots & ~domains['native_keep']
        border_w = domains['material_takeover']
    del tx_, ty_, border, inside
    # Coast/lake shore: no fields within one tile cell of water (vanilla: rivers and lakes thin them).
    shore = ~physical
    shore = shore_band(shore)
    allowed &= ~shore
    rbn = list(real_names) if real_names is not None else []
    steppe_ids = [rbn.index(n) for n in STEPPE_BIOMES if n in rbn]
    dry_ids = [rbn.index(n) for n in DRY_BIOMES if n in rbn]
    R = FOOTPRINT_R * 8
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    report = {'towns': [], 'pixels': 0}
    order = sorted(plan['regions'], key=lambda t: kind_of(t.get('battle_layout')) == 'farm')
    for t in order:
        kind = kind_of(t.get('battle_layout'))
        tq, tr = t['town']
        ty, tx = min(TH - 1, int((tr + tq % 2 * .5) * 2)), min(TW - 1, tq * 2)
        if not allowed[max(0, ty - 30):ty + 31, max(0, tx - 30):tx + 31].any():
            report['towns'].append({'region': t['region'], 'kind': kind, 'skipped': 'no editable land'})
            continue
        cl = int(biome[ty, tx])
        hb = int(real_biome[tr, tq]) if real_biome is not None else -1
        steppe = hb in steppe_ids and cl in (2, 3)
        scale = KIND_SCALE[kind] * (CLIMATE_SCALE if kind == 'farm' else CITY_CLIMATE).get(cl, 1.)
        if steppe:
            scale = min(scale, STEPPE_SCALE[kind])
        layout = str(t.get('battle_layout', ''))
        if scale <= 0:
            report['towns'].append({'region': t['region'], 'kind': kind, 'climate': cl, 'steppe': bool(steppe), 'skipped': 'none for this kind/climate'})
            continue
        cls = 'rice' if cl == 0 else 'grain_cold' if cl == 1 else 'grain_arid' if cl == 3 else 'grain_temperate'
        if 'tea' in layout and cl == 0:
            cls = 'tea'
        pool = [k for k in DONORS[cls] if k in feet]
        if not pool:
            continue
        h32 = zlib.crc32(t['region'].encode()) ^ seed
        dk = pool[h32 % len(pool)]
        win, fmask = feet[dk]
        teaw = None
        if cls == 'rice' and hb not in dry_ids:
            tp = [k for k in DONORS['tea'] if k in feet]
            teaw = feet[tp[(h32 >> 8) % len(tp)]] if tp else None
        cx, cy = 8 * tq, oh - 1 - int(round((tr + tq % 2 * .5) * 8))
        y0, y1, x0, x1 = max(0, cy - R), min(oh, cy + R + 1), max(0, cx - R), min(ow, cx + R + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        sy, sx = oy[y0 - cy + R:y1 - cy + R, x0 - cx + R:x1 - cx + R], ox[y0 - cy + R:y1 - cy + R, x0 - cx + R:x1 - cx + R]
        # Existing fields (vanilla ground kept around imported towns): skip towns already farmed.
        have = np.isin(cur[y0:y1, x0:x1, 0], FARM) & (np.hypot(sx, sy) <= 80)
        if candidate_sink is not None and domains is not None:
            # Imported donor farm pixels are only candidates now; they cannot
            # suppress the authored farm before final source selection.
            py = ((oh - 1 - np.arange(y0, y1)) // 4).clip(0, TH-1)[:, None]
            px = (np.arange(x0, x1) // 4).clip(0, TW-1)[None, :]
            have &= domains['native_keep'][py, px]
        if ignore is not None:
            have &= ~ignore[y0:y1, x0:x1]
        target = fmask.sum() * scale
        if have.sum() > .3 * max(target, 1):
            report['towns'].append({'region': t['region'], 'kind': kind, 'skipped': 'already farmed', 'existing_hexes': round(float(have.sum()) / 64, 1)})
            continue
        ang = (h32 % 3600) / 3600 * 2 * math.pi
        mir = -1 if (h32 >> 12) & 1 else 1
        ca, sa = math.cos(ang), math.sin(ang)
        ux = np.rint(ca * sx + sa * sy).astype(np.int64) * mir + R
        uy = np.rint(-sa * sx + ca * sy).astype(np.int64) + R
        ok = (ux >= 0) & (ux <= 2 * R) & (uy >= 0) & (uy <= 2 * R)
        ux, uy = ux.clip(0, 2 * R), uy.clip(0, 2 * R)
        field = fmask[uy, ux] & ok
        if not field.any():
            continue
        # Tile fields under each pixel (pixel rows north-first, tiles south-up).
        tyy = ((oh - 1 - np.arange(y0, y1)) // 4).clip(0, TH - 1)[:, None]
        txx = (np.arange(x0, x1) // 4).clip(0, TW - 1)[None, :]
        hgt0 = float(surface[ty, tx])
        dh = surface[tyy, txx] - hgt0
        sl = slope[tyy, txx]
        keep = _interp(sl, SLOPE_KEEP) * _interp(dh, HEIGHT_KEEP) * min(scale, 1.) * border_w[tyy, txx]
        if FARM_PIXEL_MASK:
            # [FARM PIXEL MASK] bilinear tile masks per pixel, cut at .5 / .2 with a jittered threshold
            _fy = ((oh - 1 - np.arange(y0, y1) + .5) / 4 - .5).clip(0, TH - 1)[:, None]
            _fx = ((np.arange(x0, x1) + .5) / 4 - .5).clip(0, TW - 1)[None, :]
            _y0 = np.minimum(_fy.astype(np.int64), TH - 2); _x0 = np.minimum(_fx.astype(np.int64), TW - 2)
            _wy, _wx = (_fy - _y0).astype(np.float32), (_fx - _x0).astype(np.float32)
            def _bl(a):
                a = np.asarray(a, np.float32)
                return (a[_y0, _x0] * (1 - _wx) + a[_y0, _x0 + 1] * _wx) * (1 - _wy) + (a[_y0 + 1, _x0] * (1 - _wx) + a[_y0 + 1, _x0 + 1] * _wx) * _wy
            _gx, _gy = np.meshgrid(np.arange(x0, x1) / 8., (oh - 1 - np.arange(y0, y1)) / 8.)
            _j = PIXEL_JITTER * fbm(_gx, _gy, (0, 0), 1.5, 2, seed=212)
            # the tile gate stays hard (allowed: river, steep, banned ground); the bilinear value only softens the edge
            # inside the allowed tiles (details review)
            hard = allowed[tyy, txx] & (_bl(allowed) > .5 + _j)
            if cl == 3:
                own = np.hypot(sx, sy) <= ARID_OWN_OASIS * 8
                hard = hard & (((irrigated[tyy, txx] > .2) & (_bl(irrigated) > .2 + .4 * _j)) | own)
        else:
            hard = allowed[tyy, txx]
            if cl == 3:
                own = np.hypot(sx, sy) <= ARID_OWN_OASIS * 8
                hard = hard & ((irrigated[tyy, txx] > .2) | own)
        # Smooth noise cuts (blobby edges), anchored in world pixels so neighbours agree.
        gxx, gyy = np.meshgrid(np.arange(x0, x1) / 8., (oh - 1 - np.arange(y0, y1)) / 8.)
        u = .5 + .5 * fbm(gxx, gyy, (0, 0), NOISE_GRAIN, 3, seed=211)
        u = np.clip((u - .5) * 1.6 + .5, 0, 1)
        take = field & hard & (u < keep)
        val = win[uy, ux]
        if teaw is not None:
            tw_, tm_ = teaw
            tea = (sl >= TEA_SLOPE[0]) & (sl <= TEA_SLOPE[1]) & (dh >= TEA_HEIGHT[0]) & (dh <= TEA_HEIGHT[1])
            # Tea takes its donor's dabs where they exist, else the rice dab turns tea.
            val = np.where((tea & tm_[uy, ux])[..., None], tw_[uy, ux], val)
            val[..., 0] = np.where(tea & np.isin(val[..., 0], RICE), TEA, val[..., 0])
            take |= field & hard & tea & tm_[uy, ux] & (u < keep * 1.2)
        if candidate_sink is not None:
            # A continuous coverage proposal, not an independently selected
            # final farm pixel. The only material draw is in ground_bake.
            support = field.astype(np.float32)
            if FARM_PIXEL_MASK:
                from ground_bake import smooth
                support = smooth(support, 4)
            proposal = support * hard * np.clip((keep - u + .2) / .4, 0, 1)
            if teaw is not None:
                proposal = np.maximum(proposal, field * hard * tea * tm_[uy, ux] *
                                      np.clip((keep * 1.2 - u + .2) / .4, 0, 1))
            candidate_sink((y0, y1, x0, x1), val[..., 0], proposal)
            take = proposal > 0
        n = int(take.sum())
        if not n:
            report['towns'].append({'region': t['region'], 'kind': kind, 'climate': cl, 'skipped': 'no suitable ground'})
            continue
        blk = cur[y0:y1, x0:x1]
        report['g_kept_px'] = report.get('g_kept_px', 0) + int((blk[take, 1] != val[take, 1]).sum())   # [STEP 4] G the old two-channel copy would have changed
        if candidate_sink is None:
            blk[take, 0] = val[take, 0]   # [STEP 4] material channel only: the G cold band stays the climate's
        mats = np.bincount(val[take][:, 0], minlength=13)[8:13]
        report['pixels'] += n
        report['towns'].append({'region': t['region'], 'kind': kind, 'climate': cl, 'steppe': bool(steppe), 'donor': dk,
                                'scale': round(scale, 2), 'hexes': round(n / 64, 1), 'donor_hexes': round(float(fmask.sum()) / 64, 1),
                                'materials_hexes': [round(float(v) / 64, 1) for v in mats]})
    painted = [x for x in report['towns'] if 'hexes' in x]
    report['painted_towns'] = len(painted)
    report['hexes'] = round(report['pixels'] / 64, 1)
    report['by_kind'] = {k: round(sum(x['hexes'] for x in painted if x['kind'] == k), 1) for k in KIND_SCALE}
    report['materials_hexes'] = [round(sum(x['materials_hexes'][i] for x in painted), 1) for i in range(5)]
    return report


# ----------------------------------------------------------------- offline preview
def _preview(out_dir):
    """Paint a copy of the workspace's staged blend (approximate inputs: no protection
    beyond the native edit rule, oasis belts from the real-river layer)."""
    from pathlib import Path
    import map_hex
    import strategic_ink
    from repair_campaign_river_roots import Assets
    A = cw.WORKSPACE
    S = A / 'work/release-input'
    V = cw.TERRAIN_VARIANTS[-1]
    pre = f'terrain/campaigns/{V}/'
    out = Path(out_dir) / pre / 'global_map'
    out.mkdir(parents=True, exist_ok=True)
    src = (S / (pre + 'global_map/global_blend.dds')).read_bytes()
    h, w = struct.unpack_from('<2I', src, 12)
    cur = np.frombuffer(src, np.uint8, offset=128).reshape(h, w, 2).copy()
    van = Assets(original_only=True).read(pre + 'global_map/global_blend.dds')
    vh, vw = struct.unpack_from('<2I', van, 12)
    donor = np.frombuffer(van, np.uint8, offset=128).reshape(vh, vw, 2)
    plan = json.loads((A / 'work/current-region-design.json').read_text(encoding='utf-8'))
    biome = np.load(A / 'analysis/polished-tile-climate.npy')
    surface = np.load(A / f'analysis/polished-surface-{V}.npy')
    alpha = np.load(A / f'analysis/polished-edit-alpha-{V}.npy')
    TH, TW = biome.shape
    m = map_hex.read(A / 'work/final-map/map.hex')
    c = m['cells']
    land = (c[..., 0] & 3) != 1
    ty, tx = np.mgrid[:TH, :TW]
    hq, hr = cw.tile_hex(tx, ty)
    physical = land[hr, hq]
    ids = map_hex.region_ids(c)
    names = m['land_regions'] + m['sea_regions']
    authored = np.isin(ids, [names.index(t['region']) for t in plan['regions'] if t['region'] in names])
    plots = ((((c[..., 2] >> 4) > 0) | ((c[..., 3] & 1) > 0)) & authored)[hr, hq]
    protected = np.zeros_like(physical)
    irrigated = np.zeros((TH, TW), np.float32)
    rt = A / 'analysis/west-real-terrain.npz'
    if rt.exists():
        z = np.load(rt)
        x0, x1, y0, y1 = (int(v) for v in z['box'])
        irrigated[y0:y1, x0:x1] = np.clip(z['river'] * 5 - .15, 0, 1)
    rb, rn, _, _ = strategic_ink.load()
    from native_edit_mask import edited_tiles
    rep = paint(cur, donor, plan, biome, surface, alpha, physical, protected, plots, irrigated, rb, rn,
                edited=edited_tiles(V, halo=0)[:TH, :TW], filler=filler_keep(c, names, hr, hq))
    (out / 'global_blend.dds').write_bytes(src[:128] + cur.tobytes())
    print(json.dumps({k: v for k, v in rep.items() if k != 'towns'}, ensure_ascii=False))
    (Path(out_dir) / 'farm-fields-preview.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preview', required=True, help='output root for the painted blend copy (release-input layout)')
    _preview(ap.parse_args().preview)

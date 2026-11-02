"""Audit the four native-map borders for visible seams before any live test.

The native map's outer rows were the edge of the vanilla world; on the expanded
canvas they sit inland. Along every land segment of each border this measures:

  wall     largest one-cell height step near the border, against the segment's
           own interior (native terrain 20-60 cells in) - walls, notches
  slope    steepest two-hex rise in the canvas band outside the border - long
           smooth inclines like the Xiaowan side of the Tibetan plateau
  stripes  blend-map banding parallel to the border: total variation of the
           along-border mean, band vs interior (stripes survive the averaging,
           natural texture does not)
  cut      material or climate changing exactly on the border line, against the
           same measure on interior lines (straight cuts, dashed rows); an interior line
           crossing the repainted vanilla filler is measured on the pre-polish blend
  tone_cut the same lines and filler handling on the slot colour (ground_tone MATERIAL_TONE,
           CIE76), every season; thresholds measured on vanilla (CONSTS ground_materials
           TONE_CUT, build_material_tones). Ground contrast fix design §10

  python audit_native_edges.py                    # current stage
  python audit_native_edges.py --pack X.pack      # a packed candidate
  --strict exits non-zero when any segment fails.
"""
import argparse, json
from pathlib import Path

import numpy as np

import biome_profiles as bp
import campaign_world as cw
import ground_tone as gt
from terrain_layers import open_source, load_layers, across, tangent_range, edge_point_hex

SEGMENT = 16            # tile cells along the border per segment (8 hexes)
GATING = ('wall', 'cut', 'climate_cut', 'tone_cut')
STRIPE_SWING = 1.5      # blend units a band must swing to count as a reversal
STRAIGHT_WINDOW = 48    # tile cells of border used to judge a slope's straightness
LIMITS = {
    'wall_step': .6,        # one tile cell, world height units
    'wall_ratio': 2.0,      # against the segment's interior steepest step
    'slope_rise': 1.6,      # per two hexes (4 cells) in the canvas band outside
    'stripe_reversals': 4,  # extra light/dark band reversals in the border band over the interior
    'straight_rise': .8,    # along-border mean height rise per two hexes
    'straightness': .7,     # that mean rise / typical single-column rise
    'cut_ratio': 3.0,       # border-line discontinuity / interior lines
    'cut_min': 3.0,         # blend units (the dashed rows and straight cuts measured 9-14)
    'climate_cut': .5,      # share of the segment where climate flips exactly at the border
}
REF_LINES = (-30, -36, -42, -48)   # interior reference lines (tile cells inside the border)
# Registered tone CUT failures on main (design 10.1, shrink only): 'tone_cut_known' instead of the gating 'tone_cut'.
TONE_CUT_KNOWN = Path(__file__).with_name('native_edge_tone_cut_known.json')
_KNOWN = {(s['edge'], tuple(s['hex'])) for s in json.loads(TONE_CUT_KNOWN.read_text(encoding='utf-8'))['segments']}
_LUTS = {}


def tone_cut_limits():
    """CONSTS ground_materials TONE_CUT (vanilla-measured min / floor per season, ratio), or None while not measured."""
    return bp.CONSTS['ground_materials'].get('TONE_CUT')


def _lut(season):
    if season not in _LUTS:
        _LUTS[season] = gt.tone_lut(season)
    return _LUTS[season]


def blend_metrics(L, edge, t0, t1):
    """cut and tone_cut of one border segment from L['blend'] (L['filler'] and L['blend_ref'] optional); no heights."""
    m = {}
    raw = across(L['blend'], edge, t0, t1, -100, 60, 4)
    b = raw.astype(np.float32).sum(axis=2)
    has_ref = 'filler' in L and 'blend_ref' in L
    raw0 = across(L['blend_ref'], edge, t0, t1, -100, 60, 4) if has_ref else None
    fill = {k: np.repeat(across(L['filler'], edge, t0, t1, k - 1, k + 1).any(axis=0), 4) if has_ref else None for k in REF_LINES}

    def jump(img, k):   # per tangent column, the 4 pixel rows either side of line k
        return np.abs(img[(100 + k) * 4:(100 + k) * 4 + 4].mean(axis=0) - img[(100 + k) * 4 - 4:(100 + k) * 4].mean(axis=0))

    def tone_jump(img, k):
        return gt.line_tone_jump(img, (100 + k) * 4)

    def interior(img, img0, fn):
        # per tangent column: filler columns from the pre-polish blend, the others from the stage as before
        out, used = [], False
        for k in REF_LINES:
            if fill[k] is not None and fill[k].any():
                used = True
                out.append(float(np.where(fill[k], fn(img0, k), fn(img, k)).mean()))
            else:
                out.append(float(fn(img, k).mean()))
        return float(np.mean(out)), used
    # Discontinuity exactly on the border line versus interior lines.
    m['cut'] = float(jump(b, 0).mean())
    # Reference lines inside the native map. Where a line crosses the vanilla non-playable filler
    # (revealed_filler.filler_hex) on vanilla land, which polish repaints with canvas ground (NONPLAY GROUND), its jump is
    # taken on the pre-polish blend (terrain-before-polish: the vanilla texture) in the filler columns, so the border is still
    # compared with vanilla's own texture and not with the smoother repaint. Thresholds unchanged.
    m['cut_interior_old'] = float(np.mean([float(jump(b, k).mean()) for k in REF_LINES]))
    b0 = raw0.astype(np.float32).sum(axis=2) if has_ref else None
    m['cut_interior'], m['cut_interior_vanilla_ref'] = interior(b, b0, jump)
    # The same lines on the slot colour, every season (§10): fails when the border line's colour jump exceeds vanilla's
    # own interior lines (min = their p99) and 3x this segment's interior (not below floor = their p50).
    lim = tone_cut_limits()
    m['tone_cut'], m['tone_cut_interior'], m['tone_cut_fail_seasons'] = {}, {}, []
    if lim:
        for season, sl in lim['by_season'].items():
            lab = _lut(season)[raw[..., 0]]
            lab0 = _lut(season)[raw0[..., 0]] if has_ref else None
            tc = float(tone_jump(lab, 0).mean())
            ti = interior(lab, lab0, tone_jump)[0]
            m['tone_cut'][season], m['tone_cut_interior'][season] = tc, ti
            if tc > sl['min'] and tc > lim['ratio'] * max(ti, sl['floor']):
                m['tone_cut_fail_seasons'].append(season)
        m['tone_cut_max'] = max(m['tone_cut'].values())
    else:   # not measured: None, never a passing 0
        m['tone_cut_max'] = None
    return m


def segment_metrics(L, edge, t0, t1):
    land = across(L['land'], edge, t0, t1, -24, 24)
    if land.mean() < .6:
        return None
    m = {}
    # Height across the border at tile resolution (every 4th LF pixel).
    h = across(L["heights"], edge, t0, t1, -80, 80, 4)[::4, ::4].astype(np.float32)
    steps = np.abs(np.diff(h, axis=0))                       # n = -80..79
    near = steps[80 - 12:80 + 12]                             # 12 cells either side
    interior = steps[80 - 60:80 - 20]
    lnd = across(L['land'], edge, t0, t1, -80, 80)
    land_steps = (lnd[1:] & lnd[:-1])
    near_land = land_steps[80 - 12:80 + 12]
    m['wall_step'] = float(np.percentile(np.where(near_land, near, 0).max(axis=0), 90))
    m['interior_step'] = float(np.percentile(np.where(land_steps[80 - 60:80 - 20], interior, 0).max(axis=0), 90))
    # A steep band running straight along the border survives averaging along it;
    # a meandering mountain front is smeared out by the averaging.
    # Judge straightness at screen scale: 24 hexes of border centred on the segment.
    lo, hi = tangent_range(edge)
    w0, w1 = max(lo, t0 - STRAIGHT_WINDOW // 2 + SEGMENT // 2), min(hi, t0 + STRAIGHT_WINDOW // 2 + SEGMENT // 2)
    hw = across(L['heights'], edge, w0, w1, -80, 80, 4)[::4, ::4].astype(np.float32)
    hm = hw.mean(axis=1)[80 - 10:80 + 70]
    mean_rise = np.abs(hm[4:] - hm[:-4]).max()
    col_rise = np.percentile(np.abs(hw[4:] - hw[:-4])[80 - 10:80 + 66].max(axis=0), 50)
    m['straight_rise'] = float(mean_rise)
    m['straightness'] = float(mean_rise / max(col_rise, .05))
    rise = np.abs(h[4:] - h[:-4])[80:80 + 70]                # outside band, two hexes
    rise_land = (lnd[4:] & lnd[:-4])[80:80 + 70]
    m['slope_rise'] = float(np.percentile(np.where(rise_land, rise, 0).max(axis=0), 75))
    # Blend banding parallel to the border (pixels; both channels summed).
    b = across(L['blend'], edge, t0, t1, -100, 60, 4).astype(np.float32).sum(axis=2)
    profile = b.mean(axis=1)                                  # along-border mean per pixel row
    # Stripes are repeated light/dark bands, not one smooth fade: count reversals of
    # the band profile (smoothed over one tile cell) whose swing exceeds the floor.
    def reversals(seg):
        seg = np.convolve(seg, np.ones(4) / 4, mode='valid')
        count, last, direction = 0, seg[0], 0
        for v in seg[1:]:
            d = v - last
            if abs(d) < STRIPE_SWING:
                continue
            if direction and np.sign(d) != direction:
                count += 1
            direction, last = np.sign(d), v
        return count
    m['stripe_reversals'] = reversals(profile[(100 - 12) * 4:(100 + 50) * 4])
    # Same-length interior band for the native texture's own banding.
    m['stripe_reversals_interior'] = reversals(profile[0:62 * 4])
    tv = np.abs(np.diff(profile))
    m['stripe_tv'] = float(tv[(100 - 12) * 4:(100 + 50) * 4].mean())
    m['stripe_ratio'] = float(m['stripe_tv'] / max(tv[0:(100 - 40) * 4].mean(), .05))
    m.update(blend_metrics(L, edge, t0, t1))
    if 'climate' in L:   # absent with --no-tiles (tile_list.bin not rebuilt yet)
        c = across(L['climate'], edge, t0, t1, -2, 2)
        valid = (c[1] != 255) & (c[2] != 255)
        m['climate_cut'] = float(((c[1] != c[2]) & valid).mean())
        m['climate_cut_interior'] = float(np.mean([((a != bb) & (a != 255) & (bb != 255)).mean()
                                                   for a, bb in zip(across(L['climate'], edge, t0, t1, -40, -30)[:-1],
                                                                    across(L['climate'], edge, t0, t1, -40, -30)[1:])]))
    else:   # not measured (--no-tiles): None, never a passing 0
        m['climate_cut'] = m['climate_cut_interior'] = None
    fails = []
    if m['wall_step'] > LIMITS['wall_step'] and m['wall_step'] > LIMITS['wall_ratio'] * max(m['interior_step'], .05):
        fails.append('wall')
    if m['slope_rise'] > LIMITS['slope_rise']:
        fails.append('slope')
    if m['stripe_reversals'] >= LIMITS['stripe_reversals'] + m['stripe_reversals_interior']:
        fails.append('stripes')
    if m['straight_rise'] > LIMITS['straight_rise'] and m['straightness'] > LIMITS['straightness']:
        fails.append('straight_slope')
    if m['cut'] > LIMITS['cut_min'] and m['cut'] > LIMITS['cut_ratio'] * max(m['cut_interior'], .2):
        fails.append('cut')
    if m['tone_cut_fail_seasons']:
        fails.append('tone_cut_known' if (edge, tuple(edge_point_hex(edge, t0 + SEGMENT // 2))) in _KNOWN else 'tone_cut')
    if m['climate_cut'] is not None and m['climate_cut'] > LIMITS['climate_cut'] and m['climate_cut'] > 3 * m['climate_cut_interior']:
        fails.append('climate_cut')
    m['fails'] = fails
    return m


def _fmt(v):
    return 'n/a' if v is None else f'{v:.2f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', type=Path)
    ap.add_argument('--pack', type=Path)
    ap.add_argument('--variant', default='3k_dlc07_main_map')
    ap.add_argument('--out', type=Path, default=cw.WORKSPACE / 'analysis/native-edge-audit.json')
    ap.add_argument('--strict', action='store_true')
    # Right after polish the tile cache and tile_list.bin are still the previous run's
    # (sync_final_terrain_surface.py --cache-only rebuilds them later): skip the tile climate.
    ap.add_argument('--no-tiles', action='store_true', help='skip the fine-tile climate check (stale tile_list.bin)')
    args = ap.parse_args()
    source, vanilla = open_source(args.stage, args.pack)
    L = load_layers(source, vanilla, args.variant,
                    want=('heights', 'blend', 'land') if args.no_tiles else ('heights', 'blend', 'land', 'climate'))
    # Vanilla filler on the tile grid and the pre-polish blend (the cut reference lines that cross the filler).
    import revealed_filler
    from fast_hex import read_hex
    _mp = cw.WORKSPACE / 'work/final-map/map.hex'
    _meta = read_hex(_mp)
    _cells = np.frombuffer(_mp.read_bytes()[-4 - cw.W * cw.H * 16:-4], np.uint8).reshape(cw.H, cw.W, 16)
    _ty, _tx = np.mgrid[:cw.TILE_H, :cw.TILE_W]
    _q, _r = cw.tile_hex(_tx, _ty)
    # only filler that was vanilla land: on former vanilla sea the pre-polish blend is the sea bed
    import map_hex
    _vland = np.zeros((cw.H, cw.W), bool)
    _vland[cw.NATIVE_HEX_SLICE] = (map_hex.read(cw.VANILLA_MAP)['cells'][:, :, 0] & 3) != 1
    L['filler'] = (revealed_filler.filler_hex(_cells, _meta['land_regions'] + _meta['sea_regions']) & _vland)[_r, _q]
    _ref = cw.WORKSPACE / f'terrain-before-polish/terrain/campaigns/{args.variant}/global_map/global_blend.dds'
    if _ref.exists():
        from terrain_layers import _dds
        L['blend_ref'] = _dds(_ref.read_bytes(), 2)
    else:
        print('WARNING: no pre-polish blend', _ref, '- filler reference lines use the stage blend')
    del _cells, _ty, _tx, _q, _r
    report = {'source': str(args.pack or args.stage or 'stage'), 'variant': args.variant, 'limits': LIMITS, 'tone_cut_limits': tone_cut_limits(),
              'segment_cells': SEGMENT, 'edges': {}}
    total = {}
    # Checks not run (--no-tiles: climate_cut) report None and are listed, so a reader or a
    # gate never takes them for a pass.
    unmeasured = (('climate_cut',) if args.no_tiles else ()) + (() if tone_cut_limits() else ('tone_cut_max',))
    report['unmeasured'] = list(unmeasured)
    for edge in ('north', 'south', 'west', 'east'):
        lo, hi = tangent_range(edge)
        segs = []
        for t0 in range(lo, hi - SEGMENT + 1, SEGMENT):
            m = segment_metrics(L, edge, t0, t0 + SEGMENT)
            if m is None:
                continue
            m['hex'] = edge_point_hex(edge, t0 + SEGMENT // 2)
            segs.append(m)
            for f in m['fails']:
                total[f] = total.get(f, 0) + 1
        report['edges'][edge] = {'all': segs, 'land_segments': len(segs), 'failing': [s for s in segs if s['fails']],
                                 'worst': {k: max((s[k] for s in segs if s[k] is not None), default=None if k in unmeasured else 0)
                                           for k in ('wall_step', 'slope_rise', 'straight_rise', 'stripe_reversals', 'cut', 'tone_cut_max', 'climate_cut')}}
    report['failures'] = total
    # Walls and cuts are unambiguous defects and gate the build. Stripes and straight
    # slopes are heuristics (mirrored texture and native relief also trigger them):
    # they are reported and previewed for review, not gating.
    report['gating'] = {k: v for k, v in total.items() if k in GATING}
    report['warnings'] = {k: v for k, v in total.items() if k not in GATING}
    report['passed'] = not report['gating']
    # segments whose cut reference lines crossed the filler (taken on the pre-polish blend)
    report['cut_reference_pre_polish'] = [{'edge': n, 'hex': x['hex'], 'cut': x['cut'], 'ref_old': x['cut_interior_old'], 'ref': x['cut_interior'],
                                          'cut_fails': 'cut' in x['fails']}
                                         for n, e in report['edges'].items() for x in e['all'] if x.get('cut_interior_vanilla_ref')]
    # every listed tone CUT segment, failing or not (a passing one leaves the list)
    report['tone_cut_known'] = [{'edge': n, 'hex': x['hex'], 'tone_cut': x['tone_cut'], 'tone_cut_interior': x['tone_cut_interior'],
                                 'fail_seasons': x['tone_cut_fail_seasons']}
                                for n, e in report['edges'].items() for x in e['all'] if (n, tuple(x['hex'])) in _KNOWN]
    for e in report['edges'].values():
        del e['all']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    for edge, e in report['edges'].items():
        print(f"{edge:5s} land segments {e['land_segments']:3d} failing {len(e['failing']):3d} worst "
              + ' '.join(f'{k}={_fmt(v)}' for k, v in e['worst'].items()))
        for s in e['failing'][:6]:
            print('   ', s['hex'], s['fails'], ' '.join(f'{k}={_fmt(s[k])}' for k in ('wall_step', 'interior_step', 'slope_rise', 'straight_rise', 'straightness', 'stripe_reversals', 'cut', 'tone_cut_max', 'climate_cut')))
    if unmeasured:
        print('not measured (None):', ', '.join(unmeasured))
    print('gating failures', report['gating'] or 'none', '| warnings', report['warnings'] or 'none', '->', args.out)
    raise SystemExit(1 if args.strict and report['gating'] else 0)


if __name__ == '__main__':
    main()

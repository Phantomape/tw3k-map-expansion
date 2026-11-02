"""Offline frontier plan: compute canvas hexes for proposed towns and relief features.

Positions = vanilla-capital fit (2048 frame) + local correction from nearby vanilla
capitals only (several authored towns are known to be misplaced), so new places line up with their neighbours.
Writes JSON and a preview PNG; never touches the design or the map.
"""
import tw3k_paths
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import campaign_world as cw, map_hex

OUT = tw3k_paths.MAIN_ROOT / 'analysis/frontier-plan-2026-10-18'


fit = cw.lonlat_fit   # the one copy of the lon/lat fit (campaign_world)


# Reference anchors with known coordinates: vanilla capitals only. The projection never read the authored towns
# (tw3k_hist_* entries were listed here but skipped below; dropped 2026-11-01, P5 prep: no design key is an anchor,
# so a rebuilt design leaves the projection unchanged).
ANCHORS = {
    '3k_dlc06_liaodong_capital': (123.17, 41.27), '3k_main_youzhou_capital': (116.35, 39.88),
    '3k_main_chengdu_capital': (104.06, 30.66), '3k_main_jianning_capital': (103.8, 25.5),
    '3k_dlc06_yongchang_capital': (99.17, 25.11), '3k_main_jiaozhi_capital': (106.05, 21.18),
    '3k_dlc06_jiuzhen_capital': (105.78, 19.8), '3k_main_hepu_capital': (109.2, 21.66),
    '3k_main_wuwei_capital': (102.64, 37.93), '3k_main_jincheng_capital': (103.2, 36.2),
}

design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
town = {r['region']: tuple(r['town']) for r in design['regions']}
# Anchor towns not in the design (the vanilla capitals): their primary slot hexes on the vanilla map, moved by the
# native padding. The canvas map keeps every vanilla slot (vanilla playable land is byte-identical but for the
# registered road / AOI hexes), so this is the position the final map gives, without reading the workspace's
# current map.hex at import (the map bake builds a map from scratch and reads no earlier map).
_V = map_hex.read(cw.VANILLA_MAP)
_vnames = _V['land_regions'] + _V['sea_regions']
_vids = map_hex.region_ids(_V['cells'])
_vprimary = _V['cells'][:, :, 2] >> 4 == 1
for key in ANCHORS:
    if key not in town and key in _vnames:
        rr, qq = np.nonzero((_vids == _vnames.index(key)) & _vprimary)
        if len(qq):
            town[key] = (int(round((qq + cw.PAD_LEFT).mean())), int(round((rr + cw.PAD_BOTTOM).mean())))
del _V, _vids, _vprimary
anchors = []
for key, (lon, lat) in ANCHORS.items():
    if key in town and key.startswith('3k_'):   # vanilla only: several authored towns await relocation
        fq, fr = fit(lon, lat)
        anchors.append((lon, lat, town[key][0] - fq, town[key][1] - fr, key))


import plan_western_regions as _west


ANCHOR_REACH = 14.0   # degrees; anchor weights fade to zero over the last 4
PRIOR_WEIGHT = 1 / 14.0 ** 2
SEAM_BAND = 3.0   # degrees east of Dunhuang over which the local correction fades in from the Western Regions offset


def place_array(lon, lat):
    """Vectorised projection to canvas hexes (q, r), unrounded.
    West of Dunhuang: the Western Regions projection (same east-west compression about Dunhuang),
    so Tibet, Bengal and the Altai stay consistent with the Western Regions. East of it: fit plus an
    inverse-distance-weighted local correction from vanilla anchors within ~12 degrees, faded in over
    SEAM_BAND from the Western Regions offset so the two projections meet without a step."""
    lon = np.asarray(lon, np.float64); lat = np.asarray(lat, np.float64)
    fq, fr = fit(lon, lat)
    wq, wr = _west.project(lat, np.minimum(lon, _west.DUNHUANG_REAL[1]))
    wsum = np.zeros_like(lon); dq = np.zeros_like(lon); dr = np.zeros_like(lon)
    for alon, alat, cq, cr, _ in anchors:
        d = np.hypot((lon - alon) * np.cos(np.radians(lat)), lat - alat)
        t = np.clip((ANCHOR_REACH - d) / 4, 0, 1)
        w = t * t * (3 - 2 * t) / (d + 1.5) ** 2
        wsum += w; dq += w * cq; dr += w * cr
    # a zero-correction prior keeps the field continuous where the anchors fade out
    cq = dq / (wsum + PRIOR_WEIGHT)
    cr = dr / (wsum + PRIOR_WEIGHT)
    sq, sr = fit(np.full_like(lon, _west.DUNHUANG_REAL[1]), lat)
    s = np.clip((lon - _west.DUNHUANG_REAL[1]) / SEAM_BAND, 0, 1)
    s = s * s * (3 - 2 * s)
    eq = fq + s * cq + (1 - s) * (wq - sq)
    er = fr + s * cr + (1 - s) * (wr - sr)
    west = lon < _west.DUNHUANG_REAL[1]
    return np.where(west, wq, eq), np.where(west, wr, er)


def place_float(lon, lat):
    q, r = place_array(lon, lat)
    return float(q), float(r)


def place(lon, lat):
    q, r = place_float(lon, lat)
    return int(round(q)), int(round(r))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = map_hex.read(cw.WORKSPACE / 'work/final-map/map.hex')
    names = F['land_regions'] + F['sea_regions']
    ids = map_hex.region_ids(F['cells'])
    H, W = ids.shape
    land = (F['cells'][:, :, 0] & 3) != 1
    plan = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    report = {'anchors': [{'key': k, 'correction': [round(cq, 1), round(cr, 1)]} for _, _, cq, cr, k in anchors], 'provinces': []}
    for p in plan['provinces']:
        rows = []
        for t in p['towns']:
            q, r = place(t['lon'], t['lat'])
            inside = 0 <= q < W and 0 <= r < H
            status = 'outside canvas'
            owner = None
            if inside:
                owner = names[ids[r, q]] if 0 <= ids[r, q] < len(names) else None
                if not land[r, q]:
                    status = 'sea - needs land'
                elif owner and (owner.startswith('3k_main_reg_non_playable') or owner.startswith('tw3k_internal_non_playable')):
                    status = 'free'
                elif owner and owner.startswith('3k_'):
                    status = 'vanilla playable - blocked'
                else:
                    status = 'authored region - conflict'
            rows.append(dict(t, q=q, r=r, status=status, owner=owner))
        feats = []
        for f in p.get('relief', []):
            feats.append(dict(f, hexes=[list(place(lon, lat)) for lon, lat in f['points']]))
        report['provinces'].append(dict(p, towns=rows, relief=feats))
    (OUT / 'frontier-plan.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    # Preview: north up, 1 px per hex, crop to the area of interest.
    img = np.zeros((H, W, 3), np.uint8)
    img[~land] = (28, 40, 70)
    img[land] = (95, 95, 95)
    authored = np.isin(ids, [names.index(k) for k in town if k in names and k.startswith('tw3k_')])
    img[land & authored] = (120, 105, 80)
    vanilla_play = land & np.vectorize(lambda i: 0 <= i < len(names) and names[i].startswith('3k_') and 'non_playable' not in names[i])(ids)
    img[vanilla_play] = (70, 110, 70)
    im = Image.fromarray(img[::-1])
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(tw3k_paths.font('msyh.ttc'), 22)
    colour = {'free': (255, 230, 0), 'sea - needs land': (255, 80, 80), 'authored region - conflict': (255, 150, 0),
              'vanilla playable - blocked': (200, 0, 200), 'outside canvas': (255, 255, 255)}
    for p in report['provinces']:
        for f in p['relief']:
            pts = [(q, H - 1 - r) for q, r in f['hexes']]
            draw.line(pts, fill=(170, 120, 255) if f['kind'] == 'ridge' else (90, 200, 255), width=5)
        for t in p['towns']:
            x, y = t['q'], H - 1 - t['r']
            rad = 9 if t.get('capital') else 6
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=colour[t['status']], outline=(0, 0, 0), width=2)
            if t.get('capital'):
                draw.text((x + 12, y - 14), p['name'], font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    im.save(OUT / 'frontier-plan-preview.png')
    for region, box in plan.get('crops', {}).items():
        dq, dr = cw.PAD_LEFT - 658, cw.PAD_BOTTOM - 230   # crops are given on the padding (658, 230) canvas
        q0, r0, q1, r1 = box[0] + dq, box[1] + dr, box[2] + dq, box[3] + dr
        im.crop((q0, H - 1 - r1, q1, H - 1 - r0)).save(OUT / f'frontier-plan-{region}.png')
    print(json.dumps({p['name']: [(t['name'], t['q'], t['r'], t['status']) for t in p['towns']] for p in report['provinces']}, ensure_ascii=False, indent=0))
    print('anchor corrections:', [(a['key'].replace('tw3k_hist_', '').replace('3k_main_', '').replace('3k_dlc06_', ''), a['correction']) for a in report['anchors']])


if __name__ == '__main__':
    main()

"""Draft layout for the full Western Regions (西域) on the current canvas.

Real coordinates (lat, lon) are projected with the fit of vanilla capitals to
their real seats (vanilla scale), then compressed east-west by WEST_SCALE about
Dunhuang, the anchor that stays where it is next to vanilla Hexi. North-south
keeps vanilla scale. Writes analysis/western-regions-draft.json and a preview
PNG (north up, world aspect) for review before anything enters the design.

  python plan_western_regions.py
"""
import tw3k_paths
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import campaign_world as cw
from analyze_reference import ROOT

A = cw.WORKSPACE
WEST_SCALE = 0.6
# Least-squares fit of 16 vanilla capitals (2048x2047 frame): hex = f(lon, lat).
# Fitted with the native map at padding (658, 230); the constants follow the
# padding so a canvas translation (migrate_canvas.py) keeps every projection.
_DQ, _DR = cw.PAD_LEFT - 658, cw.PAD_BOTTOM - 230
FIT_Q = (28.54, -0.85, -2109.9 + _DQ)
FIT_R = (-0.89, 25.72, -85.5 + _DR)
DUNHUANG_REAL = (40.14, 94.66)
DUNHUANG_HEX = (565 + _DQ, 856 + _DR)


def real_hex(lat, lon):
    return FIT_Q[0] * lon + FIT_Q[1] * lat + FIT_Q[2], FIT_R[0] * lon + FIT_R[1] * lat + FIT_R[2]


def project(lat, lon):
    q, r = real_hex(lat, lon)
    q0, r0 = real_hex(*DUNHUANG_REAL)
    return (DUNHUANG_HEX[0] + WEST_SCALE * (q - q0), r + DUNHUANG_HEX[1] - r0)


# Capital seats (approximate identifications; reviewed by the user before use).
EXISTING = {
    '疏勒': (39.47, 75.99), '莎车': (38.42, 77.24), '姑墨': (41.17, 80.26), '温宿': (41.27, 79.23),
    '于阗': (37.11, 79.93), '乌孙': (42.40, 78.60), '龟兹': (41.72, 82.96), '精绝': (37.95, 82.70),
    '焉耆': (42.06, 86.57), '小宛': (38.40, 84.50), '车师前部': (42.95, 89.06), '且末': (38.13, 85.53),
    '车师后部': (44.00, 89.10), '鄯善': (39.03, 88.17), '敦煌': (40.14, 94.66), '伊吾': (42.82, 93.51),
    '蒲类': (43.60, 93.00),
}
NEW = {
    '皮山': (37.62, 78.28), '渠勒': (36.90, 81.60), '戎卢': (37.00, 83.00), '尉犁': (41.73, 86.15),
    '危须': (42.20, 87.10), '尉头': (40.90, 78.40), '依耐': (37.90, 76.40), '蒲犁': (37.77, 75.23),
    '西夜': (37.88, 77.41), '子合': (37.40, 77.00), '捐毒': (39.70, 74.50), '休循': (39.60, 73.50),
    '山国': (41.50, 88.20), '卑陆': (43.90, 88.00), '劫国': (43.60, 86.70), '单桓': (43.90, 86.40),
    '大宛': (41.25, 71.53), '康居': (42.50, 69.00), '大月氏': (36.76, 66.90), '罽宾': (33.75, 72.82),
}
# Relief skeleton: (name, polyline of (lat, lon), height, width) / valleys without height.
RELIEF = [
    ('天山', [(43.3, 93.5), (43.8, 88.3), (43.0, 86.0), (42.2, 83.0), (42.2, 80.2), (41.5, 77.5), (40.5, 75.0)], 6.8, 20),
    ('昆仑山', [(36.0, 77.0), (36.2, 80.0), (36.5, 83.0), (37.3, 86.0)], 7.5, 26),
    ('阿尔金山', [(37.3, 86.0), (38.3, 88.5), (39.2, 91.0), (39.5, 94.0)], 5.5, 16),
    ('喀喇昆仑', [(36.3, 74.5), (35.8, 76.5), (35.2, 78.5)], 8.0, 22),
    ('帕米尔', [(39.6, 74.0), (38.5, 73.6), (37.5, 74.0), (36.8, 73.5)], 8.0, 30),
    ('阿赖山', [(39.6, 69.5), (39.7, 71.5), (39.8, 73.4)], 6.0, 16),
    ('突厥斯坦山', [(39.3, 67.5), (39.5, 69.5)], 5.0, 12),
    ('兴都库什', [(36.6, 71.8), (35.8, 70.0), (35.2, 68.3), (34.8, 67.0)], 7.5, 24),
    ('天山北支(博罗科努)', [(44.3, 85.0), (44.2, 82.5), (43.7, 80.5)], 5.2, 14),
]
VALLEYS = [
    ('塔里木河', [(40.8, 79.5), (41.0, 82.0), (40.8, 84.5), (40.3, 87.5), (40.3, 90.0)], 5),
    ('罗布泊', [(40.4, 90.0), (40.2, 90.6)], 6),
    ('费尔干纳盆地', [(40.9, 70.8), (40.7, 71.8), (40.6, 72.6)], 9),
    ('伊犁河谷', [(43.9, 82.5), (43.9, 81.0), (43.7, 79.0)], 6),
    ('锡尔河', [(40.3, 69.6), (41.5, 68.5), (42.8, 68.0)], 4),
    ('阿姆河', [(37.2, 68.5), (37.2, 66.8), (37.9, 65.2)], 4),
]


def current_positions():
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    names = {'shule': '疏勒', 'shache': '莎车', 'gumo': '姑墨', 'wensu': '温宿', 'yutian': '于阗', 'wusun': '乌孙',
             'qiuci': '龟兹', 'jingjue': '精绝', 'yanqi': '焉耆', 'xiaowan': '小宛', 'jushi_front': '车师前部',
             'qiemo': '且末', 'jushi_rear': '车师后部', 'shanshan': '鄯善', 'dunhuang': '敦煌', 'yiwu': '伊吾', 'pulei': '蒲类'}
    out = {}
    for c in design['regions']:
        stem, _, kind = c['region'].removeprefix('tw3k_hist_').rpartition('_')
        if stem in names and kind == 'capital':
            out[names[stem]] = tuple(c['town'])
    return out


def main():
    cur = current_positions()
    draft = {'west_scale': WEST_SCALE, 'anchor': {'敦煌': DUNHUANG_HEX}, 'canvas': [cw.W, cw.H],
             'existing': {k: {'current': cur.get(k), 'proposed': [round(v) for v in project(*p)]} for k, p in EXISTING.items()},
             'new': {k: [round(v) for v in project(*p)] for k, p in NEW.items()},
             'relief': [{'name': n, 'points': [[round(v) for v in project(*p)] for p in pts], 'height': h, 'width': w} for n, pts, h, w in RELIEF],
             'valleys': [{'name': n, 'points': [[round(v) for v in project(*p)] for p in pts], 'width': w} for n, pts, w in VALLEYS]}
    (A / 'analysis/western-regions-draft.json').write_text(json.dumps(draft, ensure_ascii=False, indent=1), encoding='utf-8')

    # Preview in world aspect (hex rows are HEX_DZ/HEX_DX taller than columns).
    q0, q1, r0, r1 = 140, 760, 680, 1020
    s = 2.0
    ys = cw.HEX_DZ / cw.HEX_DX
    W_, H_ = int((q1 - q0) * s), int((r1 - r0) * s * ys)
    im = Image.new('RGB', (W_, H_), (232, 220, 190))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(tw3k_paths.font('msyh.ttc'), 15)
    small = ImageFont.truetype(tw3k_paths.font('msyh.ttc'), 12)
    px = lambda q, r: ((q - q0) * s, H_ - (r - r0) * s * ys)
    # Native map edge and the old (1400) western canvas edge.
    nx = px(cw.PAD_LEFT, 0)[0]
    d.line([(nx, 0), (nx, H_)], fill=(150, 150, 150), width=1)
    d.text((nx + 4, 4), '原版地图西界', font=small, fill=(110, 110, 110))
    for n, pts, w in VALLEYS:
        d.line([px(*project(*p)) for p in pts], fill=(90, 140, 200), width=int(w * s))
        d.text(px(*project(*pts[0])), n, font=small, fill=(40, 80, 150))
    for n, pts, h, w in RELIEF:
        d.line([px(*project(*p)) for p in pts], fill=(120, 95, 70), width=int(w * s))
        d.text(px(*project(*pts[len(pts) // 2])), n, font=small, fill=(70, 50, 30))
    for k, p in EXISTING.items():
        a = px(*project(*p))
        if k in cur:
            b = px(*cur[k])
            d.line([b, a], fill=(160, 160, 160), width=1)
            d.ellipse([b[0] - 3, b[1] - 3, b[0] + 3, b[1] + 3], outline=(130, 130, 130))
        d.ellipse([a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5], fill=(30, 60, 160))
        d.text((a[0] + 6, a[1] - 9), k, font=font, fill=(20, 40, 120))
    for k, p in NEW.items():
        a = px(*project(*p))
        d.ellipse([a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5], fill=(200, 40, 40))
        d.text((a[0] + 6, a[1] - 9), k, font=font, fill=(160, 20, 20))
    d.text((8, H_ - 40), f'蓝=现有国家新位置（灰圈=当前位置）  红=新增  东西比例 {WEST_SCALE}，以敦煌为锚', font=font, fill=(0, 0, 0))
    out = ROOT / 'output/western-regions-draft.png'
    im.save(out)
    print(out, im.size)


if __name__ == '__main__':
    main()

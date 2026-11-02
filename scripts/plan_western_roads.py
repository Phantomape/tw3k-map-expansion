"""Han-era roads of the Western Regions as hex paths: the Southern Road (南道), the Northern Road
(北道) and the Eastern Han New Northern Road (北新道 / 伊吾路), with their crossings of the Pamirs.

Waypoints are the towns the roads are recorded through (汉书·西域传, 魏略): the current design's
town hexes where the state is a region, the Western Regions projection (plan_western_regions) for
passes, gates and relay posts. Between waypoints the road takes the cheapest hex path over the real
terrain (ETOPO slope; lakes and sea closed), so it follows valleys and mountain feet.
Read-only: writes analysis/roads/western-roads.json and a preview PNG.

  <tw3k python> scripts/plan_western_roads.py      (needs the geodata stack)
"""
import tw3k_paths
import heapq, json, math, os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import campaign_world as cw
import map_hex
import plan_western_regions as west
from analyze_reference import ROOT

OUT = Path(os.environ.get('TW3K_ROADS_DIR', ROOT / 'analysis/roads'))   # TW3K_ROADS_DIR: a test copy's plans
STEMS = {'疏勒': 'shule', '莎车': 'shache', '姑墨': 'gumo', '温宿': 'wensu', '于阗': 'yutian', '乌孙': 'wusun',
         '龟兹': 'qiuci', '精绝': 'jingjue', '焉耆': 'yanqi', '小宛': 'xiaowan', '车师前部': 'jushi_front',
         '且末': 'qiemo', '车师后部': 'jushi_rear', '鄯善': 'shanshan', '敦煌': 'dunhuang', '伊吾': 'yiwu',
         '皮山': 'pishan', '渠勒': 'qule', '戎卢': 'ronglu', '尉犁': 'weili', '尉头': 'weitou', '蒲犁': 'puli',
         '捐毒': 'juandu', '休循': 'xiuxun', '卑陆': 'beilu', '罽宾': 'jibin', '大宛': 'dayuan', '康居': 'kangju',
         '大月氏': 'dayuezhi'}
# Hexi corridor towns (design regions, by key) and the vanilla commandery the Hexi road joins
TOWN_KEYS = {'删丹': 'tw3k_hist_zhangye_iron', '觻得': 'tw3k_hist_zhangye_capital', '氐池': 'tw3k_hist_zhangye_lumber',
             '昭武': 'tw3k_hist_zhangye_pasture', '表氏': 'tw3k_hist_jiuquan_lumber', '福禄': 'tw3k_hist_jiuquan_capital',
             '会水': 'tw3k_hist_jiuquan_iron', '玉门': 'tw3k_hist_jiuquan_pasture',
             # Bosten / Lop Nur towns off the trunks (spurs below)
             '山国': 'tw3k_hist_shanguo_capital', '焉耆·屯田': 'tw3k_hist_yanqi_farm', '焉耆·马场': 'tw3k_hist_yanqi_pasture',
             '龟兹·马场': 'tw3k_hist_qiuci_pasture', '鄯善·盐场': 'tw3k_hist_shanshan_lumber', '伊循·屯田': 'tw3k_hist_shanshan_farm'}
VANILLA = {'武威': '3k_main_wuwei_capital'}     # site: centre of the capital's main settlement slot
# Places that are not regions: (lat, lon)
POINTS = {'阳关': (39.93, 94.06), '玉门关': (40.35, 93.86), '白龙堆': (40.35, 91.6), '楼兰': (40.52, 89.92),
          '居卢仓': (40.45, 90.9), '乌垒': (41.95, 84.35), '轮台': (41.78, 84.25), '高昌壁': (42.85, 89.53),
          '悬度': (36.3, 74.6), '瓦罕': (37.0, 73.2), '阿尔金山北麓': (39.55, 91.2)}
ROUTES = {
    '南道': ['阳关', '阿尔金山北麓', '鄯善', '且末', '小宛', '精绝', '戎卢', '渠勒', '于阗', '皮山', '莎车'],
    '南道·西出葱岭': ['莎车', '蒲犁', '瓦罕', '大月氏'],
    '南道·罽宾道（悬度）': ['皮山', '悬度', '罽宾'],
    '北道': ['玉门关', '白龙堆', '居卢仓', '楼兰', '尉犁', '轮台', '乌垒', '龟兹', '姑墨', '温宿', '尉头', '疏勒'],
    '北道·车师支线': ['车师前部', '焉耆', '尉犁'],
    '北道·西出葱岭': ['疏勒', '捐毒', '休循', '大宛', '康居'],
    '北新道（伊吾路）': ['敦煌', '伊吾', '高昌壁', '车师前部'],
    '北新道·车师后部': ['伊吾', '车师后部', '卑陆'],
    '敦煌—玉门、阳关': ['玉门关', '敦煌', '阳关'],
    # the Han Hexi road: joins the Western Regions roads to the vanilla Liang roads at Wuwei
    '河西道': ['武威', '删丹', '觻得', '表氏', '福禄', '敦煌'],
}
# Spurs to towns off a trunk: from the trunk hex nearest the town (a junction stop '<town>岔口')
SPURS = {'河西道·氐池': ('河西道', '氐池'), '河西道·昭武': ('河西道', '昭武'),
         '河西道·会水': ('河西道', '会水'), '河西道·玉门': ('河西道', '玉门'),
         # trunk None: the trunk route passing nearest the town
         '支线·山国': (None, '山国'), '支线·焉耆·屯田': (None, '焉耆·屯田'), '支线·焉耆·马场': (None, '焉耆·马场'),
         '支线·龟兹·马场': (None, '龟兹·马场'), '支线·鄯善·盐场': (None, '鄯善·盐场'), '支线·伊循·屯田': (None, '伊循·屯田')}
SLOPE_COST = 25.0       # per relief unit of slope between neighbours
MARGIN = 40             # hexes around the waypoints' box searched for paths


def neighbours(q, r, q0=0):
    """Hex neighbours of (q, r); q0 is the column a box-relative q is offset by, since the column
    parity (odd columns half a row up) is that of the absolute column q + q0."""
    odd = (q + q0) % 2
    return ((q, r + 1), (q, r - 1), (q + 1, r), (q - 1, r), (q + 1, r + (1 if odd else -1)), (q - 1, r + (1 if odd else -1)))


def main():
    import frontier_terrain as ft
    design = {c['region']: c['town'] for c in json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))['regions']}
    site = {}
    for name, stem in STEMS.items():
        t = design.get(f'tw3k_hist_{stem}_capital')
        if t:
            site[name] = tuple(t)
    for name, key in TOWN_KEYS.items():
        site[name] = tuple(design[key])
    for name, (lat, lon) in POINTS.items():
        q, r = west.project(lat, lon)
        site[name] = (int(round(q)), int(round(r)))
    F = map_hex.read(cw.WORKSPACE / 'work/final-map/map.hex')
    ids, names = map_hex.region_ids(F['cells']), F['land_regions'] + F['sea_regions']
    for name, key in VANILLA.items():           # centre of the capital's main settlement slot
        rr_, qq_ = np.nonzero((ids == names.index(key)) & (F['cells'][:, :, 2] >> 4 == 1))
        site[name] = (int(round(qq_.mean())), int(round(rr_.mean())))
    missing = sorted(({p for path in ROUTES.values() for p in path} | {t for _, t in SPURS.values()}) - set(site))
    if missing:
        raise SystemExit(f'No site for {missing}')
    qs = [q for q, r in site.values()]; rs = [r for q, r in site.values()]
    q0, q1, r0, r1 = min(qs) - MARGIN, max(qs) + MARGIN, min(rs) - MARGIN, max(rs) + MARGIN
    water = (F['cells'][:, :, 0] & 3) == 1
    rr, qq = np.mgrid[r0:r1, q0:q1]
    lon, lat = ft.pixel_lonlat(qq.astype(np.float64), rr + (qq % 2) * .5)
    rel = ft.to_relief(np.maximum(ft.etopo(lon, lat), 0)).astype(np.float64)
    blocked = water[r0:r1, q0:q1]

    def path(a, b):
        start, goal = (a[0] - q0, a[1] - r0), (b[0] - q0, b[1] - r0)
        dist = {start: 0.0}; prev = {}; heap = [(0.0, 0.0, start)]
        gx, gy = goal[0], goal[1] + ((goal[0] + q0) % 2) * .5
        while heap:
            _, d, (q, r) = heapq.heappop(heap)
            if (q, r) == goal:
                break
            if d > dist[(q, r)]:
                continue
            for nq, nr in neighbours(q, r, q0):
                if not (0 <= nq < q1 - q0 and 0 <= nr < r1 - r0) or blocked[nr, nq]:
                    continue
                nd = d + 1 + SLOPE_COST * abs(rel[nr, nq] - rel[r, q])
                if nd < dist.get((nq, nr), math.inf):
                    dist[(nq, nr)] = nd; prev[(nq, nr)] = (q, r)
                    h = math.hypot(nq - gx, nr + ((nq + q0) % 2) * .5 - gy) * .9      # A* heuristic (below the true step cost)
                    heapq.heappush(heap, (nd + h, nd, (nq, nr)))
        if goal not in prev and goal != start:
            raise SystemExit(f'No path {a} -> {b}')
        out = [goal]
        while out[-1] != start:
            out.append(prev[out[-1]])
        return [(q + q0, r + r0) for q, r in reversed(out)]

    routes = {}
    for name, stops in ROUTES.items():
        hexes = []
        for a, b in zip(stops, stops[1:]):
            seg = path(site[a], site[b])
            hexes += seg if not hexes else seg[1:]
        routes[name] = {'stops': stops, 'hexes': [list(h) for h in hexes]}
        print(name, len(hexes), 'hexes', flush=True)
    sprawl = (F['cells'][:, :, 3] & 1) != 0
    for name, (trunk, town) in SPURS.items():
        tq, tr = site[town]
        d2 = lambda h: (h[0] - tq) ** 2 + ((h[1] + (h[0] % 2) * .5 - tr - (tq % 2) * .5) * cw.HEX_DZ / cw.HEX_DX) ** 2
        trunks = [trunk] if trunk else list(ROUTES)
        cand = [(tuple(h), t) for t in trunks for h in routes[t]['hexes'] if not sprawl[h[1], h[0]]]
        junction, trunk = min(cand, key=lambda c: d2(c[0]))
        site[town + '岔口'] = junction
        hexes = path(junction, site[town])
        routes[name] = {'stops': [town + '岔口', town], 'hexes': [list(h) for h in hexes], 'trunk': trunk}
        print(name, 'from', trunk, junction, len(hexes), 'hexes', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'western-roads.json').write_text(json.dumps({'canvas': [cw.W, cw.H, cw.PAD_LEFT, cw.PAD_BOTTOM],
        'sites': {k: list(v) for k, v in site.items()}, 'routes': routes}, ensure_ascii=False) + '\n', encoding='utf-8')

    # preview: relief, water, roads, stops
    s = 3; ys = cw.HEX_DZ / cw.HEX_DX
    t = np.clip(rel / 9, 0, 1)
    img = np.stack([200 - 90 * t, 190 - 100 * t, 150 - 90 * t], -1)
    img[blocked] = (90, 130, 170)
    im = Image.fromarray(img.astype(np.uint8)[::-1]).resize(((q1 - q0) * s, int((r1 - r0) * s * ys)), Image.Resampling.BILINEAR)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(tw3k_paths.font('msyh.ttc'), 14)
    px = lambda q, r: ((q - q0 + .5) * s, ((r1 - 1 - r) + .5 - (q % 2) * .5) * s * ys)
    colours = {'南道': (170, 40, 40), '北道': (40, 60, 170), '北新道': (30, 130, 60), '敦煌': (90, 90, 90), '河西': (150, 90, 20), '支线': (200, 0, 200)}
    for name, rt in routes.items():
        c = next(v for k, v in colours.items() if name.startswith(k))
        d.line([px(*h) for h in rt['hexes']], fill=c, width=3)
    for name, (q, r) in site.items():
        x, y = px(q, r)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((x + 6, y - 9), name, font=font, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    im.save(OUT / 'western-roads-preview.png')
    print('wrote', OUT / 'western-roads.json')


if __name__ == '__main__':
    main()

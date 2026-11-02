"""Layout helpers and Western Regions tables shared by the live scripts (add_ports, apply_rivers, check_town_positions,
fix_canvas_strips, move_town, western_passability). Moved verbatim out of relayout_western_regions.py, a one-shot layout
tool now in scripts/archive (one design writer, 2026-11-01)."""
import colorsys, math

import numpy as np

import campaign_world as cw
import plan_western_regions as plan

FILLER = '3k_main_reg_non_playable'
MIN_SPACING = 11      # towns closer than this are pushed apart
BORDER_NOISE = .6     # relative step-cost noise: organic borders
ZONE = dict(q_max=760 + plan._DQ, r_min=600 + plan._DR, r_max=1080 + plan._DR)

EXISTING_STEMS = {'疏勒': 'shule', '莎车': 'shache', '姑墨': 'gumo', '温宿': 'wensu', '于阗': 'yutian', '乌孙': 'wusun',
                  '龟兹': 'qiuci', '精绝': 'jingjue', '焉耆': 'yanqi', '小宛': 'xiaowan', '车师前部': 'jushi_front',
                  '且末': 'qiemo', '车师后部': 'jushi_rear', '鄯善': 'shanshan', '敦煌': 'dunhuang', '伊吾': 'yiwu', '蒲类': 'pulei'}
# New states: key stem, capital town name (治所, 汉书/后汉书西域传; 邑落 where none is recorded), extra regions.
NEW_STATES = {
    '皮山': ('pishan', '皮山城', []), '渠勒': ('qule', '鞬都城', []), '戎卢': ('ronglu', '卑品城', []),
    '尉犁': ('weili', '尉犁城', []), '危须': ('weixu', '危须城', []), '尉头': ('weitou', '尉头谷', []),
    '依耐': ('yinai', '邑落', []), '蒲犁': ('puli', '蒲犁谷', []), '西夜': ('xiye', '呼犍谷', []),
    '子合': ('zihe', '邑落', []), '捐毒': ('juandu', '衍敦谷', []), '休循': ('xiuxun', '鸟飞谷', []),
    '山国': ('shanguo', '邑落', []), '卑陆': ('beilu', '乾当城', []), '劫国': ('jieguo', '丹渠谷', []),
    '单桓': ('danhuan', '单桓城', []),
    '罽宾': ('jibin', '循鲜城', ['farm']),
    '大宛': ('dayuan', '贵山城', ['horses', 'farm', 'iron']),
    '康居': ('kangju', '卑阗城', ['pasture', 'farm']),
    '大月氏': ('dayuezhi', '监氏城', ['farm', 'pasture', 'iron']),
}


def hex_xy(q, r):
    """Column-unit plane position of hex centres (rows are HEX_DZ/HEX_DX apart)."""
    q = np.asarray(q, np.float64)
    return q, (np.asarray(r, np.float64) + (q.astype(np.int64) % 2) * .5) * cw.HEX_DZ / cw.HEX_DX


def stamp_offsets(own, slot, town):
    tq, tr = town
    return [(int(q) - tq, (2 * int(r) + int(q) % 2) - (2 * tr + tq % 2), int(q), int(r))
            for r, q in zip(*np.nonzero(own & (slot == 1)))]


def unique_colour(used, seed):
    rng = np.random.default_rng(seed)
    while True:
        h, s, v = rng.random(), .45 + .4 * rng.random(), .45 + .45 * rng.random()
        c = [int(round(x * 255)) for x in colorsys.hsv_to_rgb(h, s, v)]
        if tuple(c) not in used:
            used.add(tuple(c))
            return c


def line_cells(a, b, H, W):
    """Hexes along the segment a-b (hex coords), dilated by one neighbour ring."""
    ax, ay = hex_xy(*a); bx, by = hex_xy(*b)
    n = max(2, int(math.hypot(bx - ax, by - ay) * 3))
    out = set()
    for t in np.linspace(0, 1, n):
        x, y = ax + (bx - ax) * t, ay + (by - ay) * t
        q = int(round(x)); r = int(round(y * cw.HEX_DX / cw.HEX_DZ - (q % 2) * .5))
        for dq, dr in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1 if q % 2 else -1), (-1, 1 if q % 2 else -1)):
            if 0 <= q + dq < W and 0 <= r + dr < H:
                out.add((q + dq, r + dr))
    return out

"""Landmarks (areas of interest) for the new regions: one shared definition.

Read by landmarks_map.py (map.hex AOI cells), landmarks_db.py (DB rows, labels, loc)
and landmarks_props.py (vanilla AOI prop models in world-space leaves).

How a vanilla dlc07 area of interest is built (3k_dlc07_main_map, 35 of them):
  campaign_areas_of_interest        key, title/description (loc: campaign_areas_of_interest_
                                    {title,description}_<key>), icon ui/aoi/*.png, group 3k_main_aoi_empty
  campaign_map_areas_of_interest    key -> 3k_dlc07_main_map
  label positions                   logical hex (q, r) rows, several per key allowed (rivers)
  map.hex                           areas_of_interest name list; per cell index+1 in
                                    byte 8 bits 3-7 (low five bits) and byte 9 bit 0 (sixth bit)
  props                             vanilla models under rigidmodels/campaign/area_of_interest in
                                    region-keyed bmd_objects leaves (world-space, BHM_IGNORE y)
  gameplay                          none on dlc07 (campaign_area_of_interest_effect_bundles has one
                                    hidden test row, 3k_main map only)
Kinds here mirror them: 'wall' = Great Wall segments (_01.. keys, cells along the line, great_wall
straight pieces + towers), 'pass' = a tower with a short wall, 'mountain' = cells + label (no
shrine .wsmodel), 'lake' = label only (0 cells, like the vanilla rivers), 'site' = a small cell
patch, 'yardang' = cells + stone_forest pillars.

Positions are canvas hexes (2048 x 2047, padding 888/375), projected once from lon/lat with
plan_frontier_regions.place_float (the projection the new towns use) and frozen here, so no
step depends on the projection code or the map it reads. Lon/lat are kept for reference.
Every landmark existed or was named by AD 190; dropped: 莫高窟 (366), 丸都 (c. 209),
冈仁波齐 (no pre-190 attestation), 扶南 (first named c. 245), 乐浪 / 龙城 (already towns),
交河 / 楼兰 (left for a possible town rename).
"""
import math

import campaign_world as cw

GROUP = '3k_main_aoi_empty'
CAMPAIGN_MAP = '3k_dlc07_main_map'
PREFIX = 'tw3k_aoi_'
LABEL_KEY_BASE = 1900900001      # campaign_map_areas_of_interest_label_positions.key (int)
ICON = 'ui/aoi/3k_main_aoi_{}.png'

LANDMARKS = [
    dict(key='yumen_pass', kind='pass', title='玉门关', icon='great_wall', radius=1.5,
         lonlat=[(93.87, 40.35)], hexes=[(781.36, 1007.10)], along='hexi_wall',
         description='汉武帝时置，为出敦煌、通西域之门户（《史记·大宛列传》）。'),
    dict(key='yang_pass', kind='pass', title='阳关', icon='great_wall', radius=1.5,
         lonlat=[(94.06, 39.93)], hexes=[(784.83, 996.13)], heading=0.0,
         description='与玉门关并为西域门户，敦煌郡龙勒县有阳关（《汉书·地理志》）。'),
    dict(key='hexi_wall', kind='wall', title='河西汉塞', icon='great_wall', segments=4, labelled=(1, 3), label_at={1: .75},   # segment 2's middle is 冥安
         lonlat=[(93.40, 40.38), (93.87, 40.36), (94.40, 40.42), (95.00, 40.52), (95.80, 40.55),
                 (96.60, 40.40), (97.40, 40.15), (98.20, 39.95), (98.90, 40.05)],
         hexes=[(773.30, 1008.29), (781.36, 1007.36), (790.40, 1008.43), (804.49, 1010.50), (827.11, 1010.73),
                (849.77, 1006.35), (872.64, 999.28), (895.70, 993.38), (915.62, 995.33)],
         # --- WALL LINE (props only) --- the line the wall pieces follow (landmarks_props; props_xz). The
         # Shule (apply_rivers, 7d10483a: real valley floor) runs within 0.2-2 hexes of `hexes` from Yumen to
         # Guazhou and crossed it six times. Real line: in the Dunhuang section the wall, Yumen Pass and Hecang
         # stand on the south side of the Shule, which flows north of them (its marshes the moat); in the Guazhou
         # section the wall runs north of the river. So: south of the river, 1.6-3 hexes off its hexes, from the
         # west end to q 838, one crossing (q 839-840, beside the road causeway at 839,1010, a beacon tower on
         # each bank), north/east of the river to q 853, then the old line on to Jiuquan. Yumen Pass keeps its
         # point (a vertex here); Dunhuang's slots stay 1.9+ hexes off. AOI cells and labels follow this
         # line too (wall_segments_xz); `hexes` stays as the line projected from lonlat.
         props_hexes=[(773.30, 1007.30), (781.36, 1007.10), (788.00, 1007.60), (793.50, 1006.30), (797.50, 1005.10),
                      (801.50, 1005.10), (806.00, 1006.40), (812.00, 1009.00), (828.00, 1009.00), (833.00, 1007.90),
                      (837.50, 1008.30), (842.00, 1010.30), (846.00, 1008.50), (850.00, 1007.30), (853.50, 1006.00),
                      (872.64, 999.28), (895.70, 993.38), (915.62, 995.33)],
         description='汉武帝时自酒泉列亭障至玉门，以护河西、通西域之路（《汉书·西域传》）。'),
    dict(key='juyan_wall', kind='wall', title='居延塞', icon='great_wall', segments=2, labelled=(1,),
         lonlat=[(98.90, 40.05), (99.40, 40.40), (99.90, 40.95), (100.40, 41.40), (100.90, 41.80), (101.05, 41.95)],
         hexes=[(915.62, 995.33), (929.54, 1003.95), (943.25, 1017.76), (957.04, 1029.01), (970.87, 1038.99),
                (974.99, 1042.76)],
         description='太初三年（前102年）路博德筑遮虏障于居延，以断匈奴南下之道（《汉书·武帝纪》）。'),
    dict(key='mount_yanran', kind='mountain', title='燕然山', icon='mount_tai', radius=6,
         lonlat=[(104.9, 45.4)], hexes=[(1080.26, 1129.49)],
         description='永元元年（89年）窦宪大破北匈奴，登燕然山，班固作铭刻石纪功（《后汉书·窦宪传》）。'),
    dict(key='mount_langjuxu', kind='mountain', title='狼居胥山', icon='mount_heng_north', radius=6,
         lonlat=[(108.9, 48.5)], hexes=[(1180.70, 1206.24)],
         description='元狩四年（前119年）霍去病击匈奴，封狼居胥山，禅于姑衍（《史记·卫将军骠骑列传》）。'),
    dict(key='lake_beihai', kind='lake', title='北海', icon='river_yangtze',
         lonlat=[(105.5, 52.0)], hexes=[(1086.77, 1302.55)],
         description='苏武使匈奴被留，徙北海上无人处牧羝，十九年乃还（《汉书·苏武传》）。'),
    dict(key='mount_congling', kind='mountain', title='葱岭', icon='mount_emei', radius=3,
         lonlat=[(74.6, 39.3), (74.8, 38.3), (75.1, 37.3)], hexes=[(451.92, 997.25), (455.86, 971.35), (461.50, 945.36)],
         description='西域南北二道皆西逾葱岭，北道出大宛、康居，南道出大月氏（《汉书·西域传》）。'),
    dict(key='lake_puchang', kind='lake', title='蒲昌海', icon='river_xi',
         lonlat=[(90.3, 40.3)], hexes=[(720.26, 1009.00)],
         description='一名盐泽，去玉门、阳关三百余里，冬夏不增减（《汉书·西域传》）。'),
    dict(key='mount_kunlun', kind='mountain', title='昆仑', icon='mount_wutai', radius=3,
         lonlat=[(79.0, 36.3), (80.5, 36.0), (82.0, 36.0)], hexes=[(528.80, 916.17), (554.64, 907.12), (580.32, 905.79)],
         description='汉使穷河源出于阗，其山多玉石，天子案古图书，名之曰昆仑（《史记·大宛列传》）。'),
    dict(key='mount_tianshan', kind='mountain', title='天山', icon='mount_song', radius=3,
         lonlat=[(85.0, 43.4), (86.5, 43.3), (88.0, 43.55), (88.6, 43.85)],
         hexes=[(627.92, 1093.44), (653.66, 1089.54), (679.22, 1094.63), (689.34, 1101.81)],
         description='天汉二年（前99年）贰师将军李广利击右贤王于天山（《汉书·武帝纪》）。'),
    dict(key='bailongdui', kind='yardang', title='白龙堆', icon='stone_forest', radius=3,
         lonlat=[(91.3, 40.6)], hexes=[(737.23, 1015.82)],
         description='楼兰东当白龙堆，乏水草，汉使往来常苦之（《汉书·西域传》）。'),
    dict(key='lake_qinghai', kind='lake', title='青海', icon='river_yellow',
         lonlat=[(100.2, 36.9)], hexes=[(956.51, 913.46)],
         description='古称仙海、西海；元始四年（4年）于此置西海郡（《汉书·王莽传》）。'),
    dict(key='heshenggu', kind='site', title='纥升骨城', icon='mount_hua', radius=2,
         lonlat=[(125.4, 41.3)], hexes=[(1618.29, 997.29)],
         description='相传朱蒙于前37年建国卒本，都纥升骨城（《三国史记》，后世追记）。'),
    dict(key='shika_seal', kind='site', title='志贺岛金印', icon='river_xi', radius=0, land_cells=6,
         lonlat=[(130.30, 33.68)], hexes=[(1778.19, 800.57)],
         description='建武中元二年（57年）倭奴国奉贡朝贺，光武赐以印绶（《后汉书·东夷传》）；金印出土于志贺岛。'),
    dict(key='ma_yuan_pillars', kind='site', title='马援铜柱', icon='stone_forest', radius=2,
         lonlat=[(108.3, 15.9)], hexes=[(1201.0, 383.0)],   # projected (1194.70, 392.42) is the 林邑 town; moved to the coast south of it, the frontier the pillars marked
         description='建武十九年（43年）马援平交趾，相传立铜柱为汉南界（事见《后汉书·马援传》，铜柱之说出晋人所记）。'),
    dict(key='bonan_lanjin', kind='site', title='博南山·兰津', icon='mount_emei', radius=2,
         lonlat=[(99.5, 25.45), (99.35, 25.33)], hexes=[(957.67, 634.72), (954.68, 631.61)],
         description='永平十二年（69年）置永昌郡，通博南山、渡兰仓水，行者歌曰“度博南，越兰津”（《华阳国志》）。'),
]
BY_KEY = {lm['key']: lm for lm in LANDMARKS}
# --- DUNHUANG YARDANG --- props-only scenes (landmarks_props; no AOI cells, label or DB row: landmarks_map
# and landmarks_db read LANDMARKS only). 敦煌雅丹 (Dunhuang Yardang geopark, 三垄沙 its eastern belt): centre
# 40.53 N 93.06 E, field 92.92-93.22 E x 40.38-40.57 N (UNESCO tentative list 5990), north-west of Yumen
# Pass past the Shule's end; ridges trend roughly WNW-ESE with the prevailing east wind (heading_deg from
# world +x, i.e. east, anticlockwise). stretch: field length / width.
PROP_SCENES = [
    dict(key='dunhuang_yardang', kind='yardang', title='敦煌雅丹', radius=2.2, stretch=1.6, count=24, heading_deg=-12,
         lonlat=[(93.06, 40.50)], hexes=[(767.42, 1011.68)]),
]
BY_KEY.update({lm['key']: lm for lm in PROP_SCENES})


def aoi_keys(lm):
    """The AOI keys of a landmark: walls are split into segments like the vanilla Great Wall."""
    if lm['kind'] == 'wall':
        return [f"{PREFIX}{lm['key']}_{i + 1:02d}" for i in range(lm['segments'])]
    return [PREFIX + lm['key']]


def all_keys():
    return [k for lm in LANDMARKS for k in aoi_keys(lm)]


# --- geometry -------------------------------------------------------------
# Hex (q, r) frame: odd-q, r north. In units of DX a hex centre sits at
# (q, (r + odd(q)/2) * ZS); neighbours are ZS (= 2/sqrt3) apart.

def hex_centre_xz(q, r):
    """Hex centre in DX units (integer hex)."""
    return q + .5, (r + (int(q) % 2) * .5 + .5) * cw.ZS


def float_hex_xz(qf, rf):
    """A frozen float hex (projection output, not tied to a column parity) in DX units:
    the parity offset is averaged (+1/4 row) so polylines stay smooth."""
    return qf + .5, (rf + .75) * cw.ZS


def to_world(xz):
    """DX units -> world (prefab) coordinates."""
    return xz[0] * cw.DX, xz[1] / cw.ZS * cw.DZ


def to_dx(x, z):
    return x / cw.DX, z / cw.DZ * cw.ZS


def hex_distance_window(points_xz, margin):
    """(q0, r0, d): d[r - r0, q - q0] = distance in hexes from hex (q, r)'s centre to the
    nearest of points_xz (DX units), over the points' bounding box plus `margin` hexes."""
    import numpy as np
    xs = [p[0] for p in points_xz]
    zs = [p[1] / cw.ZS for p in points_xz]
    q0 = max(0, int(min(xs)) - margin - 1)
    q1 = min(cw.W, int(max(xs)) + margin + 2)
    r0 = max(0, int(min(zs)) - margin - 1)
    r1 = min(cw.H, int(max(zs)) + margin + 2)
    qq = np.arange(q0, q1)[None, :]
    rr = np.arange(r0, r1)[:, None]
    cx = np.broadcast_to(qq + .5, (r1 - r0, q1 - q0)).astype(np.float64)
    cz = (rr + (qq % 2) * .5 + .5) * cw.ZS
    best = np.full((r1 - r0, q1 - q0), np.inf)
    for px, pz in points_xz:
        best = np.minimum(best, np.hypot(cx - px, cz - pz) / cw.ZS)
    return q0, r0, best


def polyline_samples(points, step):
    """Points every `step` along a polyline (same units as points), endpoints included."""
    out = [points[0]]
    for (x0, z0), (x1, z1) in zip(points, points[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, z1 - z0) / step)))
        out += [(x0 + (x1 - x0) * i / n, z0 + (z1 - z0) * i / n) for i in range(1, n + 1)]
    return out


def split_polyline(points, parts):
    """Split a polyline into `parts` pieces of equal arc length (each a polyline)."""
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    total = sum(seg)
    cuts = [total * i / parts for i in range(parts + 1)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        out.append(sub_polyline(points, seg, a, b))
    return out


def sub_polyline(points, seg, a, b):
    def at(s):
        acc = 0.0
        for i, L in enumerate(seg):
            if s <= acc + L or i == len(seg) - 1:
                t = 0 if L == 0 else min(max((s - acc) / L, 0), 1)
                return i, (points[i][0] + (points[i + 1][0] - points[i][0]) * t, points[i][1] + (points[i + 1][1] - points[i][1]) * t)
            acc += L
    ia, pa = at(a)
    ib, pb = at(b)
    return [pa] + [points[i] for i in range(ia + 1, ib + 1)] + [pb]


def landmark_xz(lm):
    """The landmark's polyline/points in DX units."""
    return [float_hex_xz(q, r) for q, r in lm['hexes']]


def props_xz(lm):
    """The line the landmark's props follow (DX units): 'props_hexes' when given (WALL LINE), else its hexes."""
    return [float_hex_xz(q, r) for q, r in lm.get('props_hexes', lm['hexes'])]


def wall_segments_xz(lm):
    # --- WALL LINE --- AOI cells and labels (landmarks_map) follow the props line, so the hover area is the wall
    return split_polyline(props_xz(lm), lm['segments'])


def label_positions(lm):
    """[(aoi key, (q, r))]: logical hex label positions in the canvas frame."""
    def nearest_hex(px, pz):
        q = int(round(px - .5))
        r = int(round(pz / cw.ZS - .5 - (q % 2) * .5))
        return q, r
    keys = aoi_keys(lm)
    if lm['kind'] == 'wall':
        out = []
        for i, seg in enumerate(wall_segments_xz(lm)):
            if i not in lm.get('labelled', ()):
                continue
            s = polyline_samples(seg, .25)
            f = lm.get('label_at', {}).get(i, .5)
            out.append((keys[i], nearest_hex(*s[min(len(s) - 1, int(round(f * (len(s) - 1))))])))
        return out
    xz = landmark_xz(lm)
    if lm['kind'] == 'mountain' and len(xz) > 1:
        pick = xz[(len(xz) - 1) // 2]
    elif len(xz) > 1:
        pick = ((xz[0][0] + xz[1][0]) / 2, (xz[0][1] + xz[1][1]) / 2)
    else:
        pick = xz[0]
    return [(keys[0], nearest_hex(*pick))]

"""Major frontier rivers (Natural Earth scalerank <= RIVER_RANK) as navigable river water in map.hex.

Runs on the current work/final-map/map.hex, after apply_roads.py (fords go where roads cross) and before
the landmark / climate steps. Adds sea regions tw3k_riv_sea_frontier_1.._k to map.hex and to the design's
imported_sea_regions (prepare_province adds their regions / campaign_map_regions rows).

Why the river is cut into several regions: the SDK regions process (MapDataBuilder) numbers each region's
areas (hex-connected runs of one region and one traversal class) with 7 bits, and its flood fill never ends
on area 127 (stack grows to the 2400 MB tracked-memory limit). See hex_components.region_areas. Every ford
is one area of the river region and one stretch of river between fords another, so a region holds at most
RIVER_AREAS of them. Wilderness slivers a river cuts off become river water (each would be an area).

Cells, as vanilla's Yellow River and Yangtze (checked on the vanilla map):
  * river hex: vanilla river_navigable sea cell; ford: vanilla river_navigable_shallows cell with the bridge
    bit (byte 3 bit 7; CAIME IsBridge). Vanilla fords carry no road bits; here a ford where a road crosses
    keeps that hex's road edges, so the road network is unchanged (TownRouteTracer walks road edges only,
    RoadsValidator allows roads on bridges);
  * land river edges (byte 4) pointing into the new water are cleared (RiversValidator);
  * bank: land beside river water becomes beach (byte 0 kind 2), as vanilla's river banks, so armies can
    walk the bank and embark (CAIME makes every other land hex beside water a cliff, which blocks movement);
    land beside a ford becomes cliff (kind 3, a bridge cliff: the crossing) with vanilla's
    river_navigable_crossing ground where it is passable playable land.
Fords: every road crossing (the road hex keeps its edges), then every FORD_SPACING hexes (widened until the
CAIME pathfinding limit of MAX_BRIDGES bridges is kept) where both banks are passable playable land and the
ford lies FORD_TOWN_CLEAR hexes or more from every slot and sprawl hex (vanilla: 5-21, median ~13). Road
hexes beside the river that are not a crossing become beach (walkable; RoadsValidator warns). Banks keep
their impassable flag (an opened hex inside impassable land is one more SDK area); road hexes are
opened if a crossing ever finds one closed.
Avoided: the native map (plus NATIVE_GUARD hexes; the revealed filler, revealed_filler.revealed_hex, is not native:
a river piece there is kept only when connected and long enough, revealed_segments), town slots and sprawl (plus TOWN_RING hexes round them),
road hexes closer than FORD_TOWN_CLEAR to a town (a ford there), lakes and all other water, landmark AOI
cells (analysis/landmarks-aoi-cells.json, when present).
Towns, as vanilla (no slot on a river hex, rivers never break at a town, on-bank towns beside the water):
where the town keep-out cuts a river line, its pieces are joined round the town by the cheapest hex path
outside the keep-out (reroute_towns: near the raw centre line, on lower ETOPO ground, off impassable land),
and a line cut where it runs into a lake or the sea runs on to that water. CAIME's SprawlValidator rejects
beach/river hazards within 2 hexes of a sprawl blob that do not touch it (LogLevel.Error, Validate menu
only, not run by the export); vanilla breaks that rule at its river towns and so do these banks.
Channels and regions: rivers are first packed into CHANNELS as before (one river where the budget allows,
merged channels inside one basin, RIVER_BASIN; named after the basin unless one river holds BASIN_SHARE).
Regions then merge the channels of one drainage system (drainage(): basin, folded by DRAINAGE) up to
REGION_AREAS SDK areas each, named tw3k_riv_sea_<system>_<n> (vanilla keeps all its lakes in one region).
The channel key of every river hex goes to river-channels.json beside map.hex: polish plans the river
channels (gap links, sand ends) by it, so merging regions leaves the terrain as it was.

Western Regions and Central Asia (WR_*): the Tarim system (Yarkand, Aksu, Khotan, Tarim, Konqi), Kaidu,
Cherchen, Shule and Dang, Ili, Syr Darya and Amu Darya, kept from Natural Earth by name (scalerank 5-9)
or authored where Natural Earth 10m lacks them (WR_LINES: per segment a cheapest path on the real valley
floor, ETOPO relief with the regional slope taken out, plus a fixed meander field where the floor is flat;
trace_line). A WR river ends at its first contact with its lake (a line end running on along the shore is
peeled back: shore water merged the channel into the lake bowl at Bosten). Endorheic:
they end in Bosten, Lop Nur, Balkhash, the Aral lakes, or in the sand (Cherchen at Taitema, Shule at
Halachi, Dang in the Dunhuang fan; river_channels.sand_ends tapers those out in polish). Laid after the
other rivers, which stay byte-identical. A WR line on a road hex is rerouted beside the road (the oasis
roads follow the rivers), so it crosses a road only where it must; road crossings take fords while the
bridge budget lasts (MAX_BRIDGES - BRIDGE_MARGIN; vanilla-shaped, grown along the river to WR_FORD_HEXES,
crossings between two land regions first) and stay land beyond it (causeways: the road crosses on land,
polish's gap link draws the channel over it). Heihe (Zhangye -> Juyan) lies inside the native map
rectangle: NATIVE_GUARD keeps it dry, as the upper Shule above Changma.

River ports (add_ports: slot nibble on a river hex, Tamralipti on the Ganges): --replace keeps them
river water with their slot and sprawl bytes, the town keep-out leaves them and the land beside them open
(the river passes the port town on its port side, as vanilla's river ports), no ford is put on them, and
a port within PORT_SEA_JOIN hexes of the sea is joined to it.

Checks (nothing is written if one fails): SDK areas per region <= hex_components.SDK_AREA_LIMIT (river
regions <= REGION_AREAS), land neighbours per river region <= MAX_ADJ_REGION, region
count <= 511 - RESERVED_IDS, road edges unchanged and the traced town routes identical (apply_roads'
TownRouteTracer replica), native map / slots / sprawl untouched, every ford with land on two sides,
river ports kept (bytes and river region) and beside other river water.

  python apply_rivers.py            # dry run: report and preview in output/offline-view/rivers
  python apply_rivers.py --apply    # back up and write work/final-map/map.hex and the design
  python apply_rivers.py --write-to X.hex          # write the map elsewhere only (offline checks)
  python apply_rivers.py --replace --pre-river backups/rivers-<t>/map.hex --post-river <map that step wrote> [--apply]
        [--pre-river-of PRE0 POST0]   # when that earlier step was itself a --replace of PRE0 / POST0
                                      # (an --apply --replace now also saves backups/rivers-<t>/map-stripped.hex)
  python apply_rivers.py ... --no-wr --write-to X.hex   # without the WR rivers: the old map's hexes (regions merged)
                                    # undo the rivers already on the map (strip_rivers), then lay them again;
                                    # afterwards rerun apply_forest_ground, landmarks_map, apply_climate,
                                    # refresh_border_masks (each a no-op away from the changed river hexes)

Afterwards: all_directions_build.py database (new region rows), sdk_memory_export.py (map-data) and
Tw3kBuild pathfinding / borders / trade-routes, then the terrain pipeline (river water props name their
river region; polish reads work/final-map/river-channels.json, which goes with map.hex).
"""
import argparse, collections, json, math, os, re, shutil, time, unicodedata
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

import campaign_world as cw
import map_hex
from add_inland_lake import edge_mask, neighbours
from hex_components import region_areas, hex_dilate, hex_label, hex_mean_filter, hex_min_filter, neighbour_bits, SDK_AREA_LIMIT
from revealed_filler import NON_PLAYABLE, revealed_hex
from layout_common import unique_colour

A = cw.WORKSPACE
ROOT = cw._ROOT
FINAL = A / 'work/final-map/map.hex'
OUT = ROOT / 'output/offline-view/rivers'
AOI_FILES = (ROOT / 'analysis/landmarks-aoi-cells.json', A / 'analysis/landmarks-aoi-cells.json')
RIVER_RANK = 4              # Natural Earth scalerank kept as river water
RIVER_PREFIX = 'tw3k_riv_sea_'   # regions tw3k_riv_sea_<river>_<n> ('_riv_sea_': fords join the land network)
MAX_ADJ_LAND = 9            # land regions per river CHANNEL (RegionsValidator: more slows the campaign AI)
# River regions merge the channels of one drainage system (as vanilla's one lake region, 3k_main_sea_lake,
# 18 land neighbours on the vanilla map): REGION_AREAS SDK areas at most (MapDataBuilder blew its 2400 MB
# tracked limit on regions near the 127-area index, analysis/frontier-rivers-mapdata-2026-10-22.md: kept
# well under hex_components.SDK_AREA_LIMIT), MAX_ADJ_REGION land neighbours at most (a RegionsValidator
# warning past MAX_ADJ_LAND, not a limit).
REGION_AREAS = 80
MAX_ADJ_REGION = 40
DRAINAGE = {'mesopotamia': 'indus',         # Shatt al-Arab (25 hexes): with the Indus, both Arabian Sea
            'shule': 'tarim'}               # Shule and Dang: endorheic, east of Lop Nur, with the Tarim system
FREE_IDS = 20               # region ids left free after the rivers
# Natural Earth local / section names folded into one river key
RIVER_FAMILY = {'heilong_jiang': 'amur', 'lancang': 'mekong', 'nu': 'salween', 'ayeyarwady': 'irrawaddy',
                'irrawaddy_delta': 'irrawaddy', 'n_mai': 'irrawaddy', 'yarlung': 'brahmaputra', 'dihang': 'brahmaputra',
                'ertis': 'irtysh', 'ertix': 'irtysh', 'verkhniy_yenisey': 'yenisey', 'malyy_yenisey': 'yenisey',
                'kyzyl_khem': 'yenisey', 'bykovskaya_protoka': 'lena', 'olenekskaya_protoka': 'lena',
                'za': 'mekong', 'maquan': 'brahmaputra', 'shiquan': 'indus', 'hailar': 'argun', 'shishhid_gol': 'yenisey', 'chang_jiang': 'yangtze', 'jinsha': 'yangtze', 'tongtian': 'yangtze', 'tuotuo': 'yangtze'}
RIVER_AREAS = 100           # SDK areas per river region (hard limit 127)
MIN_RIVER = 12              # hexes: shorter river pieces stay valleys
RIVER_SLIVER = 12           # hexes: wilderness cut off by a river below this becomes river water
FORD_SPACING = 12           # hexes along the river between fords (where both banks are passable)
TOWN_RING = 1               # hexes kept dry around settlement slots and sprawl (the slot and sprawl hexes too)
FORD_TOWN_CLEAR = 5         # fords / bridges at least this many hexes from any slot or sprawl hex (vanilla 5-21)
REROUTE_PAD = 30            # hexes round a town cut searched for the way round the town
REROUTE_RAW = .5            # path cost per hex step away from the raw centre line
REROUTE_METRES = 100.       # path cost 1 per this many metres (ETOPO) above the lower cut end
REROUTE_IMPASSABLE = 3.     # path cost of an impassable hex
REROUTE_ROAD = 1000.        # path cost of a road hex closer than FORD_TOWN_CLEAR to a town (a ford there)
NATIVE_GUARD = 5            # hexes kept dry outside the native map
REVEALED_GAP = 2            # hex steps beyond NATIVE_GUARD: a revealed river piece ending this near vanilla river water is dropped
# Region names: a river region holds one river where the region budget allows; merged regions take the
# basin of their rivers (RIVER_BASIN) unless one river holds BASIN_SHARE of the hexes.
RIVER_BASIN = {'amur': 'amur', 'argun': 'amur',
               'lena': 'siberia', 'aldan': 'siberia', 'angara': 'siberia', 'yenisey': 'siberia', 'selenga': 'siberia',
               'ideriyn': 'siberia', 'ob': 'siberia', 'irtysh': 'siberia', 'pechora': 'siberia',
               'ural': 'caspian', 'volga': 'caspian', 'huang': 'qinghai', 'yangtze': 'qinghai',
               'mekong': 'indochina', 'salween': 'indochina', 'irrawaddy': 'indochina',
               'brahmaputra': 'ganges', 'ganges': 'ganges', 'indus': 'indus', 'shatt_al_arab': 'mesopotamia'}
BASIN_SHARE = .9
RESERVED_IDS = 4            # region ids kept free (as apply_frontier_regions)
MAX_BRIDGES = 127           # CAIME PathfindingExporter: bridges (connected bridge-hex groups) it can address
BRIDGE_MARGIN = 7           # bridges kept free for later edits
FILLER = '3k_main_reg_non_playable'
BRIDGE = 0x80               # byte 3 bit 7
ROAD_BITS = 0x7E            # byte 3 bits 1-6

# ---- Western Regions and Central Asia (the endorheic Tarim, Bosten / Lop Nur, Shule, Balkhash and Aral
# basins). Natural Earth draws these rivers at scalerank 5-9, under RIVER_RANK, so they are kept by name
# (WR_NE_RIVERS: Natural Earth 'name' slug -> river key, inside WR_BOX); stretches Natural Earth 10m lacks are authored
# (WR_LINES, below). They are laid after the other rivers and leave those byte-identical (--no-wr gives the
# old map): their fords use the bridges the others leave (MAX_BRIDGES - BRIDGE_MARGIN), their regions are
# packed and named on their own (WR_REGIONS ids out of the FREE_IDS margin) and appended after the others'.
WR_BOX = (55., 34., 100., 48.)  # lon0, lat0, lon1, lat1
WR_NE_RIVERS = {'konqi': 'konqi', 'tarim': 'tarim', 'yarkant': 'yarkand', 'aksu': 'aksu', 'toxkan': 'aksu',
                'yurungkax': 'khotan', 'qarqan': 'cherchen', 'ile': 'ili', 'syr_darya': 'syr_darya',
                'naryn': 'syr_darya', 'amu_darya': 'amu_darya', 'panj': 'amu_darya'}
# Natural Earth names the Bosten outflow from the lake past Korla 'Tarim'; from its vertex nearest this
# point (the head of NE's 'Konqi He', 86.372E 41.243N) to the lake it is the Konqi.
WR_SPLIT = {'tarim': ((86.372, 41.243), 'konqi')}
# Authored centre lines (lon, lat), each segment snapped to the valley floor on ETOPO (trace_line).
# 'join': the line's last point runs on to the nearest hex of that river (or 'water': open water).
# Sources: Natural Earth end points where a line continues one; county seats and gauges on the rivers
# (Bayanbulak, Dashankou, Hejing, Yanqi, Qiemo, Subei, Dunhuang, Guazhou, Changma), Tianditu/OSM courses.
WR_LINES = {
    # 开都河: Yulduz basin (Bayanbulak) -> Dashankou gorge -> Hejing -> Yanqi -> Bosten. On the canvas Nanhe
    # (南河城, the Yanqi capital) stands where Hejing and Yanqi lie; Yanqi is on the Kaidu's south bank (the
    # river bends round its north and east into the lake), so the line keeps that side of the town stamp:
    # over the Hejing road north-west of it, past its north, then due south between it and Weixu over the
    # Weixu road straight into the lake's north shore (canvas points one hex outside the town keep-out ring;
    # through the town, the keep-out sent it round the south; nearer the town's roads it ran down the shore).
    'kaidu': {'points': [(83.70, 43.10), (84.15, 43.00), (84.55, 42.92), (84.90, 42.72), (85.30, 42.50), (85.73, 42.24),
                         (85.967, 42.366), (86.317, 42.379), (86.609, 42.35), (86.78, 42.239), (86.80, 41.96)], 'join': 'water'},
    # 孔雀河 below NE's end: past Loulan (on its delta in the Han) into Lop Nur
    'konqi': {'points': [(89.387, 40.633), (89.70, 40.58), (89.95, 40.50), (90.15, 40.45)], 'join': 'water'},
    # 和田河 below NE's Yurungkax: north across the Taklamakan past Mazar Tagh to the Tarim (summer flow)
    'khotan': {'points': [(80.436, 37.833), (80.62, 38.08), (80.80, 38.45), (80.86, 38.90), (80.92, 39.40),
                          (80.98, 39.90), (81.05, 40.30), (81.10, 40.47)], 'join': 'tarim'},
    # 车尔臣河 below NE's Qarqan: Qiemo, then north-east along the desert edge to Taitema (ends in the sand)
    'cherchen': {'points': [(85.679, 37.773), (85.60, 38.02), (85.66, 38.25), (86.00, 38.50), (86.50, 38.72),
                            (87.00, 38.92), (87.50, 39.10), (87.90, 39.30), (88.20, 39.48)]},
    # 疏勒河: Shule Nanshan -> Changma gorge -> Guazhou -> north of Dunhuang -> Yumen Pass -> Halachi (sand)
    'shule': {'points': [(98.30, 38.25), (97.95, 38.55), (97.60, 38.95), (97.25, 39.35), (96.95, 39.70), (96.85, 39.88),
                         (96.75, 40.10), (96.50, 40.30), (96.30, 40.45), (95.80, 40.60), (95.30, 40.58), (95.00, 40.52),
                         (94.60, 40.45), (94.20, 40.42), (93.86, 40.40), (93.40, 40.45)]},
    # 党河: Danghe Nanshan -> Subei -> Danghe gorge -> Dunhuang, where it ends in the oasis fan (Hanshu: 氐置水
    # 东北入泽, into a marsh; a line on to the Shule crossed the Dunhuang road within FORD_TOWN_CLEAR of the towns)
    'dang': {'points': [(95.85, 38.40), (95.45, 38.80), (95.10, 39.20), (94.88, 39.50), (94.75, 39.80), (94.68, 40.10)]},
}
WR_LINE_RANK = 9.5          # authored lines lose a shared hex to any Natural Earth river
TRACE_PAD = 12              # hexes round an authored line searched for its valley floor
TRACE_OFF = .15             # path cost per hex step away from the authored segment
TRACE_SAMPLES = 7           # ETOPO samples per hex (its floor is the lowest)
VALLEY_SMOOTH = 8           # hex mean passes for the regional slope (relief: floor minus that ...
VALLEY_R = 4                # ... minus its lowest value within this many hex steps: 0 on the low line)
VALLEY_METRES = 10.         # path cost 1 per this many metres of relief above the local valley floor
MEANDER_COST = 1.           # path cost of the meander field (0..1 per step): bends on a plain that ETOPO leaves flat
MEANDER_WAVE = 6            # hexes between the meander field's lattice points
MEANDER_SEED = 4243         # fixed seed: the field is one map-wide array (deterministic, position-stable)
JOIN_MAX = 15               # hexes an authored end runs on to its 'join' target
WR_BASIN = {'konqi': 'tarim', 'tarim': 'tarim', 'yarkand': 'tarim', 'aksu': 'tarim', 'khotan': 'tarim',
            'cherchen': 'tarim', 'kaidu': 'tarim', 'ili': 'balkhash', 'syr_darya': 'aral', 'amu_darya': 'aral',
            'shule': 'shule', 'dang': 'shule'}
WR_FAMILIES = frozenset(WR_BASIN)
RIVER_BASIN.update(WR_BASIN)
WR_REGIONS = 10             # region ids the Western Regions rivers may take out of the FREE_IDS margin
WR_FORD_SPACING = 16        # hexes along a Western Regions river between spaced fords (widened to fit)
WR_FORD_HEXES = 4           # vanilla ford groups: 4 hexes (30 of vanilla's 31 bridges)
WR_ROAD_COST = 20.          # reroute path cost of a road hex near a WR line (a crossing only where needed)
PORT_SEA_JOIN = 4           # hexes: a river port this close to the sea is joined to it
PORT_RIVER_JOIN = 3         # hexes: a river port this close to a laid river is joined to it (river spec 6.7)
LAKES = ('3k_main_sea_lake',)   # inland lake regions (not the sea a river port is joined to)


def channels_path(map_path):
    """The river-channel sidecar of a map: work/final-map/river-channels.json beside final-map/map.hex,
    <X>.river-channels.json beside any other map (--write-to)."""
    map_path = Path(map_path)
    return map_path.with_name('river-channels.json') if map_path.name == 'map.hex' else map_path.with_name(map_path.name + '.river-channels.json')


def drainage(channel):
    """Drainage system (region group) of a channel key: its basin (RIVER_BASIN; basin and 'frontier' keys
    stand for themselves), folded by DRAINAGE."""
    b = RIVER_BASIN.get(channel, channel)
    return DRAINAGE.get(b, b)


def write_text_atomic(path, text):
    """UTF-8 text through <name>.<pid>.tmp and os.replace (an interrupted write leaves the old file)."""
    import os
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def hex_grow(mask, steps):
    """Hexes within `steps` hex steps of mask (hex_dilate repeated). scipy's binary_dilation with its
    default cross structure grew 4 square-grid neighbours a step and missed 2 of the 6 hex neighbours."""
    out = np.asarray(mask, bool)
    for _ in range(steps):
        out = hex_dilate(out)
    return out


def river_family(rec):
    """Readable river key for region names: Natural Earth name_en (else name), local names folded."""
    n = str(rec.get('name_en') or rec.get('name') or 'frontier').replace('\x00', '').strip()
    slug = re.sub(r'[^a-z0-9]+', '_', unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode().lower()).strip('_')
    return RIVER_FAMILY.get(slug, slug or 'frontier')


def _nb(q, r):
    odd = q % 2
    return ((q, r + 1), (q, r - 1), (q + 1, r), (q - 1, r), (q + 1, r + (1 if odd else -1)), (q - 1, r + (1 if odd else -1)))


def chain(pq, pr):
    """Hex-connected chain through the projected points (float q, r), as apply_frontier_regions did."""
    path = []
    for (aq, ar), (bq, br) in zip(zip(pq, pr), list(zip(pq, pr))[1:]):
        n_ = max(1, int(math.hypot(bq - aq, br - ar) * 4))
        for t in np.linspace(0, 1, n_ + 1):
            c = (int(round(aq + (bq - aq) * t)), int(round(ar + (br - ar) * t)))
            if not path or c != path[-1]:
                while path and c not in _nb(*path[-1]) and c != path[-1]:
                    path.append(min(_nb(*path[-1]), key=lambda v: (v[0] - c[0]) ** 2 + (v[1] - c[1]) ** 2))
                if not path or c != path[-1]:
                    path.append(c)
    return path


def wr_family(rec, parts):
    """River key of a Western Regions river kept by name (WR_NE_RIVERS inside WR_BOX), else None."""
    n = str(rec.get('name') or '').replace('\x00', '').strip()      # frontier_terrain reads 'name', not name_en
    slug = re.sub(r'[^a-z0-9]+', '_', unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode().lower()).strip('_')
    if slug not in WR_NE_RIVERS or not str(rec.get('featurecla', '')).startswith(('River', 'Lake Centerline')):
        return None
    lo0, la0, lo1, la1 = WR_BOX
    pts = np.concatenate(parts)
    if not ((pts[:, 0] > lo0) & (pts[:, 0] < lo1) & (pts[:, 1] > la0) & (pts[:, 1] < la1)).all():
        return None
    return WR_NE_RIVERS[slug]


def trace_line(points, H, W):
    """Hex path along authored (lon, lat) points on the real valley floor.

    Per hex the floor is the lowest of TRACE_SAMPLES ETOPO samples inside it (a gorge narrower than a
    hex still counts). Its relief: the floor less its VALLEY_SMOOTH-pass hex mean (the regional slope
    taken out), less the lowest such value within VALLEY_R hex steps, so the low line of a trough, a
    gorge or a fan foot reads 0 whatever the valley's slope along it. Per segment between waypoints
    (authored from towns, gauges and gorges: the line passes each) the cheapest hex path (Dijkstra): 1 a step, 1 per VALLEY_METRES of relief, MEANDER_COST times
    the meander field (bends where the floor is flat at hex scale), TRACE_OFF per hex step off the
    straight segment. So a line follows the real valley floor between its waypoints (the Shule north of
    Dunhuang and the Cherchen below Qiemo ran as straight segments: the old cost, height above the
    segment's lower end, is flat across a plain). Deterministic: heap ties by (row, col).
    """
    import heapq
    import plan_frontier_regions as pf
    lon = np.array([p[0] for p in points], np.float64); lat = np.array([p[1] for p in points], np.float64)
    fq, fr = pf.place_array(lon, lat)
    pts = [(int(round(a)), int(round(b))) for a, b in zip(fq, fr)]
    qs, rs = [p[0] for p in pts], [p[1] for p in pts]
    pad = TRACE_PAD + VALLEY_R
    y0, y1 = max(0, min(rs) - pad), min(H, max(rs) + pad + 1)
    x0, x1 = max(0, min(qs) - pad), min(W, max(qs) + pad + 1)
    x0 -= x0 % 2
    floor = window_floor(y0, y1, x0, x1)
    floor = np.where(np.isfinite(floor), floor, np.nanmax(floor))
    relief = floor - hex_mean_filter(floor, VALLEY_SMOOTH)     # the valley's slope taken out
    relief = relief - hex_min_filter(relief, VALLEY_R)
    hh, ww = floor.shape
    cost_v = 1 + relief / VALLEY_METRES + MEANDER_COST * meander_field(H, W)[y0:y0 + hh, x0:x0 + ww]
    path = [pts[0]]
    for (aq, ar), (bq, br) in zip(pts, pts[1:]):
        if (aq, ar) == (bq, br):
            continue
        seg = np.zeros((hh, ww), bool)
        for q, r in chain([aq, bq], [ar, br]):
            if 0 <= r - y0 < hh and 0 <= q - x0 < ww:
                seg[r - y0, q - x0] = True
        cost = cost_v + TRACE_OFF * hex_steps(seg, 2 * TRACE_PAD)
        src, dst = (ar - y0, aq - x0), (br - y0, bq - x0)
        dist = {src: 0.}; prev = {}; heap = [(0., src)]
        while heap:
            d, h = heapq.heappop(heap)
            if h == dst:
                break
            if d > dist[h]:
                continue
            for a, b in neighbours(h[1] + x0, h[0] + y0):
                n = (b - y0, a - x0)
                if not (0 <= n[0] < hh and 0 <= n[1] < ww):
                    continue
                nd = d + float(cost[n])
                if nd < dist.get(n, np.inf):
                    dist[n] = nd; prev[n] = h; heapq.heappush(heap, (nd, n))
        rev = [dst]
        while rev[-1] != src:
            rev.append(prev[rev[-1]])
        path += [(q + x0, r + y0) for r, q in reversed(rev[:-1])]
    return path


_MEANDER = {}


def meander_field(H, W):
    """Smooth value noise in [0, 1] over the whole map (hex rows x columns): a MEANDER_WAVE lattice of
    np.random.default_rng(MEANDER_SEED) values, smoothstep-bilinear between them. On a plain where the
    valley floor is flat at hex scale (relief ~0: the lower Shule trough, the Cherchen fan foot) it bends
    the cheapest path by a hex or two as a meandering river would; in a real valley the relief term
    (1 per VALLEY_METRES) outweighs it."""
    key = (H, W, MEANDER_WAVE, MEANDER_SEED)
    if key not in _MEANDER:
        gy, gx = H // MEANDER_WAVE + 2, W // MEANDER_WAVE + 2
        lat = np.random.default_rng(MEANDER_SEED).random((gy, gx))
        y = np.arange(H, dtype=np.float64) / MEANDER_WAVE; x = np.arange(W, dtype=np.float64) / MEANDER_WAVE
        iy, ix = y.astype(int), x.astype(int)
        ty, tx = y - iy, x - ix
        ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
        a_ = lat[iy][:, ix]; b_ = lat[iy][:, ix + 1]; c_ = lat[iy + 1][:, ix]; d_ = lat[iy + 1][:, ix + 1]
        top = a_ + (b_ - a_) * tx[None, :]; bot = c_ + (d_ - c_) * tx[None, :]
        _MEANDER[key] = top + (bot - top) * ty[:, None]
    return _MEANDER[key]


def window_floor(y0, y1, x0, x1):
    """Valley floor per hex of rows y0:y1, columns x0:x1: the lowest ETOPO metres of TRACE_SAMPLES
    samples (the centre and a ring at .35 hex) inside each hex (frontier_terrain inverse projection,
    one inversion for all samples)."""
    import frontier_terrain as ft
    rr, qq = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    offs = [(0., 0.)] + [(.35 * math.cos(k * math.pi / 3), .35 * math.sin(k * math.pi / 3)) for k in range(TRACE_SAMPLES - 1)]
    qa = np.stack([qq + dq for dq, _ in offs]); ra = np.stack([rr + dr for _, dr in offs])
    lon, lat = ft.pixel_lonlat(qa, ra)
    e = ft.etopo(np.nan_to_num(lon), np.nan_to_num(lat)).astype(np.float64)
    e = np.where(np.isfinite(lon) & np.isfinite(lat), e, np.inf)
    return np.where(np.isfinite(e).any(0), e.min(0), np.nan)


def join_line(path, target, H, W):
    """Hexes from the line's last hex on to the nearest hex of target (breadth-first, at most JOIN_MAX
    steps, ties by (row, col)); [] when the end already touches target, None when target is farther."""
    end = path[-1]
    if any(0 <= a < W and 0 <= b < H and target[b, a] for a, b in _nb(*end)):
        return []
    seen = {end: None}; frontier = [end]
    for _ in range(JOIN_MAX):
        nxt = []
        for h in frontier:
            for n in sorted(_nb(*h), key=lambda v: (v[1], v[0])):
                if n in seen or not (0 <= n[0] < W and 0 <= n[1] < H):
                    continue
                seen[n] = h
                if target[n[1], n[0]]:
                    out = []
                    while seen[n] is not None:
                        out.append(n); n = seen[n]
                    return [h_ for h_ in reversed(out) if not target[h_[1], h_[0]]]
                nxt.append(n)
        frontier = sorted(nxt, key=lambda v: (v[1], v[0]))
    return None


def raster_rivers(H, W, open_water=None, wr=True):
    """Hex-connected chains of the Natural Earth river centrelines (as apply_frontier_regions did): rivers of
    scalerank <= RIVER_RANK, the Western Regions rivers kept by name (WR_NE_RIVERS) and the authored lines
    (WR_LINES). Returns the river mask, the river key per hex (lowest scalerank, then file order) and a
    report row per authored line. open_water: sea and lake hexes (an authored 'join': 'water' target)."""
    import frontier_terrain as ft
    import plan_frontier_regions as pf
    riv = np.zeros((H, W), bool)
    name = np.full((H, W), '', dtype=object)          # river key per hex: lowest scalerank, then file order
    rank = np.full((H, W), 99.)

    def lay(path, fam, sr):
        for q, r in path:
            if 0 <= q < W and 0 <= r < H:
                riv[r, q] = True
                if sr < rank[r, q]:
                    rank[r, q] = sr; name[r, q] = fam
    for rec, parts in ft.rivers():
        sr = rec.get('scalerank', 99)
        wfam = wr_family(rec, parts) if wr else None
        if wfam is None and (sr > RIVER_RANK or not str(rec.get('featurecla', '')).startswith('River')):
            continue
        fam = wfam or river_family(rec)
        for part in parts:
            if fam in WR_SPLIT:
                (slo, sla), other = WR_SPLIT[fam]
                k = int(np.argmin(np.hypot(part[:, 0] - slo, part[:, 1] - sla)))
                if np.hypot(part[k, 0] - slo, part[k, 1] - sla) < .05:   # the split vertex is on this part
                    first_is_other = part[0, 0] > part[-1, 0]          # the Bosten end lies east
                    segs = ((part[:k + 1], other), (part[k:], fam)) if first_is_other else ((part[:k + 1], fam), (part[k:], other))
                    for sp, f in segs:
                        pq, pr = pf.place_array(sp[:, 0], sp[:, 1]); lay(chain(pq, pr), f, sr)
                    continue
            pq, pr = pf.place_array(part[:, 0], part[:, 1])
            lay(chain(pq, pr), fam, sr)
    lines = {}
    for fam in sorted(WR_LINES, key=lambda f: (WR_LINES[f].get('join') in WR_LINES, f)) if wr else []:   # join targets first
        spec = WR_LINES[fam]
        path = trace_line(spec['points'], H, W)
        joined = []
        if spec.get('join') == 'water' and open_water is not None:
            # into the lake head-on: the line stops at the first hex of its final run beside (or in) the
            # water, not running on along the shore to an end point inside the lake
            wet = [0 <= q < W and 0 <= r < H and (open_water[r, q] or any(
                0 <= a < W and 0 <= b < H and open_water[b, a] for a, b in _nb(q, r))) for q, r in path]
            k = len(path)
            while k > 1 and wet[k - 1]:
                k -= 1
            if k < len(path):
                path = path[:k + 1]
            joined = join_line(path, open_water, H, W)
        elif spec.get('join'):
            joined = join_line(path, riv & (name == spec['join']), H, W)
        lay(path + (joined or []), fam, WR_LINE_RANK)
        lines[fam] = {'hexes': len(path) + len(joined or []), 'join': spec.get('join'), 'join_hexes': len(joined or []),
                      'joined': joined is not None, 'start': list(path[0]), 'end': list((path + (joined or []))[-1])}
    return riv, name, lines


def hex_steps(mask, limit):
    """Hex steps to the nearest hex of mask (hex_dilate rings), limit + 1 beyond limit."""
    d = np.full(mask.shape, limit + 1, np.int32)
    cur = np.asarray(mask, bool).copy(); d[cur] = 0
    for i in range(1, limit + 1):
        nxt = hex_dilate(cur); d[nxt & ~cur] = i; cur = nxt
    return d


def window_elevation(y0, y1, x0, x1):
    """ETOPO metres per hex of rows y0:y1, columns x0:x1 (frontier_terrain inverse projection)."""
    import frontier_terrain as ft
    rr, qq = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    lon, lat = ft.pixel_lonlat(qq, rr)
    e = ft.etopo(np.nan_to_num(lon), np.nan_to_num(lat)).astype(np.float64)
    return np.where(np.isfinite(lon) & np.isfinite(lat), e, np.nan)


def reroute_towns(raw, riv0, cut, passable, open_water, imp, road_cost, H, W):
    """Ways round the towns for a river line the town keep-out cuts.

    raw: raster centre lines; riv0: raw hexes outside every keep-out (all pieces, before MIN_RIVER);
    cut: raw hexes inside the town keep-out only (not the native guard or AOIs); passable: hexes a river
    may take (land outside every keep-out); open_water: sea and lake hexes. Per cut (a hex-connected run of cut hexes) the river pieces
    beside it are joined, largest first, by the cheapest hex path through passable hexes (Dijkstra:
    1 a step, REROUTE_RAW per hex step off the raw line, 1 per REROUTE_METRES above the lower end,
    REROUTE_IMPASSABLE on impassable ground, road_cost per hex: REROUTE_ROAD on a road hex near a town,
    WR_ROAD_COST on other road hexes near a Western Regions river line); a cut beside one
    piece and open water (the line runs on into a lake or the sea) runs on to that water. Returns the
    path mask and a report row per cut. Deterministic: cuts by their first hex, heap ties by (row, col)."""
    import heapq
    plab = hex_label(riv0)
    psize = np.bincount(plab.ravel()); psize[0] = 0
    clab = hex_label(cut)
    paths = np.zeros((H, W), bool)
    rows = []
    for k in range(1, int(clab.max()) + 1):
        ck = clab == k
        ys, xs = np.nonzero(ck)
        ring = hex_dilate(ck) & ~ck
        touch = sorted({int(v) for v in np.unique(plab[ring & riv0]) if v})
        row = {'cut_hex': [int(xs[0]), int(ys[0])], 'cut_hexes': int(ck.sum()), 'pieces': [[int(t), int(psize[t])] for t in touch]}
        wet_end = ring & open_water
        if not touch:
            row['result'] = 'no river beside the cut'; rows.append(row); continue
        if len(touch) == 1 and not wet_end.any():
            row['result'] = 'line ends inside the keep-out'; rows.append(row); continue
        sel = np.isin(plab, touch) & hex_dilate(ring)
        sy, sx = np.nonzero(sel | ck)
        y0, y1 = max(0, sy.min() - REROUTE_PAD), min(H, sy.max() + REROUTE_PAD + 1)
        x0, x1 = max(0, sx.min() - REROUTE_PAD), min(W, sx.max() + REROUTE_PAD + 1)
        x0 -= x0 % 2                                   # even first column keeps the hex parity in the window
        win = (slice(y0, y1), slice(x0, x1))
        draw = hex_steps(raw[win], 2 * REROUTE_PAD).astype(np.float64)
        elev = window_elevation(y0, y1, x0, x1)
        lp = plab[win]
        ends = {t: sorted(zip(*np.nonzero((lp == t) & (hex_dilate(ck[win]) & ~ck[win])))) for t in touch}
        ref = np.nanmin([elev[e] for t in touch for e in ends[t]] or [0.])
        step_cost = 1 + REROUTE_RAW * draw + np.where(np.isfinite(elev), np.maximum(elev - ref, 0) / REROUTE_METRES, 0) \
            + REROUTE_IMPASSABLE * imp[win] + road_cost[win]
        free = passable[win] & (lp == 0)          # never through a piece (the ends are the sources and goals)
        hh, ww = free.shape
        tree = [max(touch, key=lambda t: (psize[t], -t))]
        goals = [t for t in touch if t not in tree]
        if wet_end.any() and len(touch) == 1:
            goals = ['water']
        joined = []
        while goals:
            src = [e for t in tree for e in ends[t]]
            if 'water' in goals:
                tgt = hex_dilate(wet_end[win]) & free
            else:
                tgt = np.isin(lp, goals) & hex_dilate(ck[win])
            dist = {s: 0. for s in src}; prev = {}
            heap = [(0., s) for s in sorted(src)]
            found = None
            while heap:
                d, h = heapq.heappop(heap)
                if d > dist.get(h, np.inf):
                    continue
                if tgt[h] and h not in src:
                    found = h; break
                r, q = h
                for a, b in neighbours(q + x0, r + y0):
                    rr_, qq_ = b - y0, a - x0
                    if not (0 <= rr_ < hh and 0 <= qq_ < ww):
                        continue
                    n = (rr_, qq_)
                    if not (free[n] or tgt[n]):
                        continue
                    nd = d + (float(step_cost[n]) if free[n] else 0.)
                    if nd < dist.get(n, np.inf):
                        dist[n] = nd; prev[n] = h; heapq.heappush(heap, (nd, n))
            if found is None:
                row.setdefault('unjoined', []).extend([g for g in goals])
                break
            path = []
            h = found
            while h not in src:
                if lp[h] == 0:
                    path.append(h)
                h = prev[h]
            for r, q in path:
                paths[r + y0, q + x0] = True
            got = 'water' if 'water' in goals else int(lp[found])
            joined.append({'to': got, 'path_hexes': len(path), 'cost': round(float(dist[found]), 3)})
            if got == 'water':
                goals = []
            else:
                tree.append(got); goals.remove(got)
        row['joined'] = joined
        row['result'] = 'rerouted' if joined and 'unjoined' not in row else ('partly rerouted' if joined else 'not rerouted')
        rows.append(row)
    return paths, rows


def draw_only_in_regions(riv, land, ids, n_land, port, scope):
    """River spec 6.2 / 6.4 option B (user 2026-10-30): a river never cuts a land region. River hexes laid over one
    region's land that would split it (a piece of that region left without a land path to its largest piece, which
    it had before the river) stay land of that region and are drawn only: their channel is carved and watered like
    a visual link (river_channels), map.hex keeps the region's land. The non-playable filler is a region like any
    other. River pieces the rule leaves shorter than MIN_RIVER (no port) are drawn only too.
    ids / land: the map before the rivers; n_land: land region count (ids below it are land regions); scope: the river
    hexes the rule may draw (the Western Regions rivers and the revealed-filler segments the user's option B is for;
    the older rivers' registered cuts, region_connectivity_known.json, stay as they are).
    Returns (riv, drawn, report)."""
    drawn = np.zeros(riv.shape, bool)
    regions = sorted(int(v) for v in np.unique(ids[riv & land]) if 0 <= v < n_land)
    base = {}
    for k in regions:
        m = land & (ids == k)
        base[k] = int(hex_label(m).max())
    rounds = 0
    while True:
        rounds += 1
        add = np.zeros(riv.shape, bool)
        for k in regions:
            m = land & (ids == k) & ~riv
            lab = hex_label(m)
            n = int(lab.max())
            if n <= base[k]:
                continue
            sz = np.bincount(lab.ravel()); sz[0] = 0
            keep = int(sz.argmax())
            stray = (lab > 0) & (lab != keep)
            add |= riv & scope & (ids == k) & land & hex_dilate(stray) & ~port
        if not add.any() or rounds > 200:
            break
        riv = riv & ~add; drawn |= add
    lab = hex_label(riv); sz = np.bincount(lab.ravel()); sz[0] = 0
    short = riv & scope & (sz[lab] < MIN_RIVER) & ~np.isin(lab, np.unique(lab[port & riv]))
    riv = riv & ~short; drawn |= short
    rep = {'drawn_hexes': int(drawn.sum()), 'short_pieces_drawn_hexes': int(short.sum()), 'rounds': rounds,
           'regions_drawn': len({int(v) for v in np.unique(ids[drawn])})}
    return riv, drawn, rep


def revealed_segments(riv, revealed, land, cells, F, rname):
    """River pieces (hex-connected) that touch the revealed filler: kept only when they hold >= MIN_RIVER hexes,
    are hex-connected to water (sea, lake, a vanilla river: any kind 1 hex) or run on into a canvas river outside
    the native rectangle, and do not stop in the NATIVE_GUARD gap beside a vanilla river (within
    NATIVE_GUARD + REVEALED_GAP hex steps of vanilla river water without touching it: e.g. the Wuyuan north
    branch, Ulaan Gol, parallel to the vanilla Yellow River, dropped until it can join it cleanly).
    Returns (riv without the dropped pieces, report rows sorted by first end hex)."""
    H, W = riv.shape
    rect = np.zeros((H, W), bool); rect[cw.NATIVE_HEX_SLICE] = True
    gnames = F['land_ground_types'] + F['sea_ground_types']
    g = (cells[:, :, 5].astype(np.int32) >> 4) + ((cells[:, :, 6].astype(np.int32) & 7) << 4) - 1
    vriv = ~land & rect & ~revealed & np.isin(g, [gnames.index(n) for n in ('river_navigable', 'river_navigable_shallows') if n in gnames])
    dv = hex_steps(vriv, NATIVE_GUARD + REVEALED_GAP)
    lab = hex_label(riv)
    keys = np.unique(lab[riv & revealed])
    out = riv.copy()
    rows = []
    for k in sorted(int(x) for x in keys if x > 0):
        comp = lab == k
        ys, xs = np.nonzero(comp)
        n = int(comp.sum())
        ring = hex_dilate(comp) & ~comp
        water = bool((ring & ~land).any())
        canvas = bool((comp & ~rect).any())
        touches_vriv = bool((ring & vriv).any())
        gap = bool((dv[comp] <= NATIVE_GUARD + REVEALED_GAP).any()) and not touches_vriv
        nb = np.zeros((H, W), np.int8)
        for d in range(6):
            nb += hex_neighbour_shift(comp, d)
        ends = sorted((int(q), int(r)) for r, q in zip(*np.nonzero(comp & (nb <= 1))))
        if not ends:
            i0, i1 = int(np.argmin(xs)), int(np.argmax(xs))
            ends = [(int(xs[i0]), int(ys[i0])), (int(xs[i1]), int(ys[i1]))]
        names = collections.Counter(str(v) for v in rname[comp] if v)
        keep = n >= MIN_RIVER and (water or canvas) and not gap
        reason = 'kept' if keep else ('short' if n < MIN_RIVER else 'vanilla river guard gap' if gap else 'not connected')
        if not keep:
            out &= ~comp
        rows.append({'river': names.most_common(1)[0][0] if names else '', 'hexes': n, 'revealed_hexes': int((comp & revealed).sum()),
                     'ends': [list(e) for e in (ends if len(ends) <= 2 else [ends[0], ends[-1]])], 'end_count': len(ends),
                     'to_water': water, 'to_canvas_river': canvas, 'vanilla_river_steps': int(dv[comp].min()),
                     'result': reason})
    rows.sort(key=lambda r: r['ends'][0])
    return out, rows


def hex_neighbour_shift(mask, d):
    """mask moved onto each hex from its neighbour in direction d (apply_roads.DIRS; edges clamped, no wrap)."""
    import apply_roads as ar
    H, W = mask.shape
    sh = np.zeros((H, W), np.int8)
    for par in (0, 1):
        dq, dr = ar.DIRS[par][d]
        cols = np.arange(par, W, 2); nc = cols + dq; ok = (nc >= 0) & (nc < W); cols, nc = cols[ok], nc[ok]
        if dr >= 0:
            sh[:H - dr, cols] = mask[dr:, nc]
        else:
            sh[-dr:, cols] = mask[:H + dr, nc]
    return sh


def strip_rivers(M, P, R):
    """M (the current map) with an earlier river step undone, for --replace.

    P: the map that river step read (its backups/rivers-<time>/map.hex); R: the map it wrote (the next
    step's backup; check: this script's old version on P gives R byte for byte). Every bit the river step
    changed (P ^ R, bytes 0-14; byte 15 is recomputed) goes back to P where M still holds R's value there;
    a bit a later step changed again keeps M's value and is reported as a conflict. Region ids: M's river
    hexes go back to P's region, the river regions leave the region list (later ids move down), their
    sea palette entries go too. River port hexes (a settlement slot nibble on a river hex of M: add_ports'
    river ports) go back to P too and are returned with M's bytes, for main() to keep them river water
    with their slot and sprawl bytes. Returns (map dict like map_hex.read, report, ports {'mask', 'cells'})."""
    names_p = P['land_regions'] + P['sea_regions']
    names_r = R['land_regions'] + R['sea_regions']
    names_m = M['land_regions'] + M['sea_regions']
    old = [n for n in names_r if n.startswith(RIVER_PREFIX)]
    if names_r[:len(names_p)] != names_p or names_r[len(names_p):] != old:
        raise SystemExit('--post-river is not --pre-river plus appended river regions')
    if [n for n in names_m if n.startswith(RIVER_PREFIX)] != old:
        raise SystemExit('the current map holds other river regions than --post-river')
    cm, cp, cr = M['cells'], P['cells'], R['cells']
    if not (cm.shape == cp.shape == cr.shape):
        raise SystemExit('map sizes differ')
    keep = [n for n in names_m if not n.startswith(RIVER_PREFIX)]
    index = {n: i for i, n in enumerate(keep)}
    to_new_m = np.array([index.get(n, -1) for n in names_m] + [-1], np.int32)
    to_new_p = np.array([index.get(n, -1) for n in names_p] + [-1], np.int32)
    if (to_new_p[:-1] < 0).any():
        raise SystemExit('a region of --pre-river is gone from the current map')
    idm, idp, idr = map_hex.region_ids(cm), map_hex.region_ids(cp), map_hex.region_ids(cr)
    river_r = np.isin(idr, [names_r.index(n) for n in old])
    river_m = np.isin(idm, [names_m.index(n) for n in old])
    report = {'old_river_regions': old, 'old_river_hexes': int(river_r.sum())}
    # river hexes a later step re-assigned (a lake, a relayout) keep that region
    region_conflict = river_r & ~(river_m & (np.array(names_m + [''], object)[np.where(idm >= 0, idm, len(names_m))]
                                             == np.array(names_r + [''], object)[np.where(idr >= 0, idr, len(names_r))]))
    stray = river_m & ~river_r
    if stray.any():
        raise SystemExit(f'{int(stray.sum())} river-region hexes of the current map were not river in --post-river')
    ids = to_new_m[np.where(idm >= 0, idm, len(names_m))]
    back = river_r & ~region_conflict
    ids[back] = to_new_p[np.where(idp >= 0, idp, len(names_p))][back]
    mask = (cp ^ cr).copy()
    mask[..., 0] &= 0x07; mask[..., 1] = 0; mask[..., 15] = 0     # region bits handled above, byte 15 recomputed
    # river port hexes (add_ports: a settlement slot on river water, slot nibble 2): saved with M's bytes
    # for main() to lay again as they are; here they go back to P like every other river hex
    port = river_m & ((cm[:, :, 2] >> 4) > 0)
    conflict = (((cm ^ cr) & mask) != 0) & ~port[..., None]
    out = np.where(conflict, cm, (cm & ~mask) | (cp & mask)).astype(np.uint8)
    out[port] = cp[port]
    map_hex.set_region_ids(out, ids)
    out[..., 15] = map_hex.border_masks(out)
    hexes = conflict.any(2)
    report.update({'river_changed_hexes': int(mask.any(2).sum()), 'restored_hexes': int((mask.any(2) & ~hexes).sum()),
                   'conflict_hexes': int(hexes.sum()), 'conflict_bytes': {int(b): int(conflict[..., b].sum()) for b in range(16) if conflict[..., b].any()},
                   'conflict_sample': [[int(q), int(r)] for r, q in zip(*np.nonzero(hexes))][:20],
                   'region_conflict_hexes': int(region_conflict.sum()),
                   'river_port_hexes': [[int(q), int(r)] for r, q in zip(*np.nonzero(port))]})
    pal = list(M['palettes'])
    if len(pal) > 1 and len(pal[1]) == 4 * len(M['sea_regions']):
        drop = {M['sea_regions'].index(n) for n in old}
        pal[1] = b''.join(pal[1][4 * i:4 * i + 4] for i in range(len(M['sea_regions'])) if i not in drop)
    S = dict(M, sea_regions=[n for n in M['sea_regions'] if not n.startswith(RIVER_PREFIX)], cells=out, palettes=pal)
    return S, report, {'mask': port, 'cells': cm[port].copy()}


def aoi_cells(H, W):
    m = np.zeros((H, W), bool)
    for p in AOI_FILES:
        if not p.exists():
            continue
        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                if len(x) == 2 and all(isinstance(v, int) for v in x):
                    q, r = x
                    if 0 <= q < W and 0 <= r < H:
                        m[r, q] = True
                else:
                    for v in x:
                        walk(v)
        walk(json.loads(p.read_text(encoding='utf-8')))
    return m


def lay_rivers(F, V, ports=None, wr=False, revealed_rivers=False, aoi=None, strip_info=None):
    """Lay the rivers on F (a map dict without river regions: the strip_rivers output for --replace, or the map
    bake's pre-river state) against the vanilla map V. In memory only: nothing is read from or written to the
    workspace except, when aoi is None, the landmark AOI files (aoi_cells). ports: strip_rivers' river ports
    ({'mask', 'cells'}) or None. wr: the Western Regions rivers (TW3K_WR_RIVERS); revealed_rivers: rivers on the
    revealed filler (TW3K_REVEALED_RIVERS). Returns {'cells', 'ids', 'water', 'ford', 'road', 'allnames', 'sea',
    'pal', 'river_regions', 'report', 'channels_doc', 'failed'} (main() writes, the map bake builds on it)."""
    import apply_roads as ar
    cells = F['cells'].copy(); H, W = cells.shape[:2]
    names = F['land_regions'] + F['sea_regions']
    if any(n.startswith(RIVER_PREFIX) for n in names):
        raise SystemExit('Rivers are already on this map (--replace undoes them first)')
    ids = map_hex.region_ids(cells).copy()
    kind = cells[:, :, 0] & 3
    land = kind != 1
    imp = (cells[:, :, 2] & 8) > 0
    slot = cells[:, :, 2] >> 4
    sprawl = (cells[:, :, 3] & 1) > 0
    road = (cells[:, :, 3] & ROAD_BITS) > 0
    filler = ids == names.index(FILLER)
    # Native map: the rectangle less the revealed filler (vanilla non-playable land now playable,
    # revealed_filler.revealed_hex), which takes rivers, banks and the guard as canvas does, and less vanilla's
    # non-playable seas (water there now: a river on revealed land may end in them, as canvas rivers end in
    # canvas seas; rivers are laid on land only, so no sea hex changes).
    native = np.zeros((H, W), bool); native[cw.NATIVE_HEX_SLICE] = True
    revealed = revealed_hex(cells, names)
    vnames = V['land_regions'] + V['sea_regions']
    vnp = np.zeros((H, W), bool)
    vnp[cw.NATIVE_HEX_SLICE] = np.isin(map_hex.region_ids(V['cells']), [i for i, n in enumerate(vnames) if n in NON_PLAYABLE])
    native &= ~(revealed | (vnp & ~land))
    # Stop-loss 2026-10-26 (user, v18): rivers into the revealed filler cut regions there (8 fragments beyond
    # v16 in daner / zhuyai / east_tianzhu; those river hexes carry no water anyway). Off unless
    # TW3K_REVEALED_RIVERS=1: the whole native rectangle is native again (guard included), as in v16. Back with the
    # river spec in the map bake (river lines and region borders together). [step to be absorbed by the map bake]
    if not revealed_rivers:
        native = np.zeros((H, W), bool); native[cw.NATIVE_HEX_SLICE] = True
        revealed = np.zeros((H, W), bool)
    if aoi is None:
        aoi = aoi_cells(H, W)
    report = {'workspace': str(A), 'warnings': [], 'aoi_cells': int(aoi.sum()), 'revealed_hexes': int(revealed.sum())}
    if strip_info is not None:
        report['strip_rivers'] = strip_info

    # ---- river hexes
    raw, rname, report['wr_lines'] = raster_rivers(H, W, ~land, wr=wr)
    town = (slot > 0) | sprawl
    # River ports (add_ports: W1, W2 river hexes with slot nibble 2, kept by strip_rivers): they stay river
    # water as they are, and the keep-out leaves them and the land beside them open, so the river runs
    # past the port town on its port side (as vanilla's 19 river ports).
    port = ports['mask'] if ports is not None else np.zeros((H, W), bool)
    town_ko = hex_grow(town, TOWN_RING) & ~(hex_dilate(port) & ~town)
    report['river_ports'] = [[int(q), int(r)] for r, q in zip(*np.nonzero(port))]
    other_ko = hex_grow(native, NATIVE_GUARD) | aoi
    keep_out = town_ko | other_ko
    dtown = hex_steps(town, FORD_TOWN_CLEAR)
    road_near = road & (dtown < FORD_TOWN_CLEAR) & ~keep_out     # a river there would be a ford by a town
    # A Western Regions line on a road hex is cut there too and rerouted beside the road (the oasis roads
    # follow these rivers; laid on the road hexes they made 37 crossings, groups of up to 24 ford hexes):
    # the way round crosses a road only where it must (WR_ROAD_COST a road hex near a WR line).
    wr_raw = raw & np.isin(rname, sorted(WR_FAMILIES))
    road_wr = road & wr_raw & land & ~keep_out & ~road_near
    wr_zone = hex_grow(wr_raw, 2 * REROUTE_PAD) if wr_raw.any() else wr_raw
    road_cost = REROUTE_ROAD * road_near + WR_ROAD_COST * (road & ~road_near & wr_zone)
    riv0 = raw & land & ~keep_out & ~road_near & ~road_wr
    # Vanilla's rivers run on past its towns (no slot on a river hex, on-bank towns beside it): where the
    # town keep-out cuts the line, the pieces are joined round the town (reroute_towns).
    cut = raw & land & (town_ko | road_near | road_wr) & ~other_ko
    reroute, report['town_cuts'] = reroute_towns(raw, riv0, cut, land & ~keep_out, ~land, imp, road_cost, H, W)
    report['wr_road_hexes_cut'] = int(road_wr.sum())
    riv = riv0 | reroute | port
    report['reroute_hexes'] = int(reroute.sum())
    lab = hex_label(riv); sz = np.bincount(lab.ravel()); sz[0] = 0
    riv = (sz[lab] >= MIN_RIVER) | port
    # River spec 6.7: a river port planned at its town (the town does not move, a resource stamp's W1 / W2 touch the
    # centre) is joined to the laid river by the shortest path of land hexes that are no slot hex and in no other
    # keep-out (PORT_RIVER_JOIN at most): a short inlet off the river, not a reroute of it.
    report['river_port_river_joins'] = []
    if port.any():
        plb = hex_label(port)
        for k in range(1, int(plb.max()) + 1):
            g = plb == k
            if (hex_dilate(g) & riv & ~port).any():
                continue
            start = sorted((int(q), int(r)) for r, q in np.argwhere(g))
            j = None
            for h in start:                         # the shortest inlet from any hex of the port
                jj = join_line([h], riv & ~port, H, W)
                if jj is None or len(jj) > PORT_RIVER_JOIN or any(not land[b, a] or slot[b, a] or other_ko[b, a] for a, b in jj):
                    continue
                if j is None or len(jj) < len(j):
                    j = jj
            if j is None:
                report['river_port_river_joins'].append({'port': start, 'joined': None}); continue
            for a, b in j:
                riv[b, a] = True
            report['river_port_river_joins'].append({'port': start, 'joined': [[int(a), int(b)] for a, b in j]})
    # Failing that, a river port a few hexes short of the sea (Tamralipti: the Ganges ended two hexes above its estuary)
    # is joined to it by the shortest hex path over land (not a town hex, outside the native guard and
    # AOIs), at most PORT_SEA_JOIN hexes, so the port reaches the sea along its river.
    report['river_port_sea_joins'] = []
    if port.any():
        lake_ids = [names.index(n) for n in LAKES if n in names]
        sea = ~land & ~np.isin(ids, lake_ids)
        plb = hex_label(port)
        for k in range(1, int(plb.max()) + 1):
            if (hex_dilate(plb == k) & riv & ~port).any():
                continue                            # joined to the river above (river spec 6.7)
            best = None
            for r, q in sorted(map(tuple, np.argwhere(plb == k))):
                j = join_line([(int(q), int(r))], sea, H, W)
                if j is None or len(j) > PORT_SEA_JOIN or any(not land[b, a] or town[b, a] or other_ko[b, a] for a, b in j):
                    continue
                if best is None or len(j) < len(best):
                    best = j
            if best:
                for a, b in best:
                    riv[b, a] = True
                report['river_port_sea_joins'].append([[int(a), int(b)] for a, b in best])
    wild = filler & land & ~keep_out & ~road_near
    l0 = hex_label(filler & land); s0 = np.bincount(l0.ravel())
    l1 = hex_label(wild & ~riv); s1 = np.bincount(l1.ravel()); s1[0] = 0
    touch = np.unique(l1[hex_dilate(riv) & (l1 > 0)])
    sliver = np.isin(l1, touch) & (s1[l1] <= RIVER_SLIVER) & (s1[l1] < s0[l0])
    riv |= sliver
    # lone land hexes the river encloses become water (GroundTypesValidator)
    for _ in range(2):
        wet = riv | ~land
        dry_nb = np.zeros((H, W), np.int8)
        for d in range(6):
            sh = np.zeros((H, W), bool)
            for par in (0, 1):
                dq, dr = ar.DIRS[par][d]
                cols = np.arange(par, W, 2); nc = cols + dq; ok = (nc >= 0) & (nc < W); cols, nc = cols[ok], nc[ok]
                if dr >= 0:
                    sh[:H - dr, cols] = ~wet[dr:, nc]
                else:
                    sh[-dr:, cols] = ~wet[:H + dr, nc]
            dry_nb += sh
        lone = land & ~riv & (dry_nb == 0) & hex_dilate(riv) & ~keep_out & ~road_near
        riv |= lone
    report['lone_land_to_water'] = int(lone.sum())
    # river key per hex (nearest rasterised hex); Western Regions river hexes
    _, (ny, nx) = ndimage.distance_transform_edt(rname == '', return_indices=True)
    fam_all = rname[ny, nx]
    # A Western Regions river ends where it first meets its lake: a line end that runs on along the
    # shore (each hex beside the water, the hex before it too) is peeled back to that first contact.
    # River water hugging the shore merged the channel into the lake bowl (Bosten: below-plane flats,
    # straight shores, the lake drawn smaller). Leaves only (one river neighbour): a river crossing past
    # the shore is left as it is. Port hexes stay.
    shore = hex_dilate(~land) & land
    wr_any = np.isin(fam_all, sorted(WR_FAMILIES))
    peeled = 0
    while True:
        nbits = neighbour_bits(riv)
        cnt = sum(((nbits >> d) & 1).astype(np.int8) for d in range(6))
        leaf = riv & shore & wr_any & ~port & (cnt == 1) & (neighbour_bits(riv & shore) > 0)
        if not leaf.any():
            break
        riv &= ~leaf; peeled += int(leaf.sum())
    report['wr_shore_hexes_peeled'] = peeled
    report['river_hexes_rasterised'] = int(raw.sum())
    report['river_slivers_to_water'] = int(sliver.sum())
    riv, report['revealed_segments'] = revealed_segments(riv, revealed, land, cells, F, rname)
    # River spec option B: river hexes that would split a land region are drawn only (draw_only_in_regions)
    riv, drawn, report['draw_only'] = draw_only_in_regions(riv, land, ids, len(F['land_regions']), port, wr_any | revealed)

    def nbs(q, r):
        return [(a, b) for a, b in neighbours(q, r) if 0 <= a < W and 0 <= b < H]

    def bank_ok(a, b):
        return land[b, a] and not riv[b, a] and not imp[b, a] and slot[b, a] == 0 and not sprawl[b, a] \
            and not filler[b, a] and not keep_out[b, a]

    plab = hex_label(riv)
    dist = np.full((H, W), -1, np.int32)
    pieces = collections.defaultdict(list)
    for k in range(1, int(plab.max()) + 1):
        ys, xs = np.nonzero(plab == k)
        s = (ys[xs.argmin()], xs[xs.argmin()]); dist[s] = 0; fr = [s]
        while fr:
            nf = []
            for r, q in fr:
                pieces[k].append((r, q))
                for a, b in nbs(q, r):
                    if plab[b, a] == k and dist[b, a] < 0:
                        dist[b, a] = dist[r, q] + 1; nf.append((b, a))
            fr = nf

    # ---- fords: every road crossing (river hex on a road), then spaced where both banks are passable
    # playable land, as many as the pathfinding format allows (MAX_BRIDGES bridges, a bridge being a
    # connected group of bridge hexes)
    road_ford = riv & road & ~port
    base_bridges = len(np.unique(hex_label((cells[:, :, 3] & BRIDGE) > 0))) - 1
    budget = MAX_BRIDGES - BRIDGE_MARGIN - base_bridges

    def sides(group):
        """Separate land groups next to a bridge-hex group (CAIME BridgesValidator). PathfindingExporter needs exactly
        two: BridgesGenerator.SplitEdges keeps two sides, and a bridge cliff on a third crashes it (index -1)."""
        gs = set(map(tuple, group))
        landn = {p for r, q in gs for p in nbs(q, r) if land[p[1], p[0]] and not riv[p[1], p[0]]}
        seen, n = set(), 0
        for p in landn:
            if p in seen:
                continue
            n += 1; todo = [p]; seen.add(p)
            while todo:
                x = todo.pop()
                for y in nbs(*x):
                    if y in landn and y not in seen:
                        seen.add(y); todo.append(y)
        return n

    wr_hex = riv & wr_any
    # a road crossing whose land does not fall into exactly two sides (a bend, a branch) takes the next river hexes too
    road_ford0 = road_ford.copy()
    grown = 0
    for _ in range(6):
        gl = hex_label(road_ford); bad = [k for k in range(1, int(gl.max()) + 1) if sides([(r, q) for r, q in np.argwhere(gl == k)]) != 2]
        if not bad:
            break
        add = hex_dilate(np.isin(gl, bad)) & riv & ~road_ford
        grown += int(add.sum()); road_ford |= add
    report['road_fords_grown_for_two_sides'] = grown
    # still one-sided (the road runs along the river's end): those river hexes stay land, the river stops;
    # on a Western Regions river only the road hexes stay land (causeways) and the river runs on
    gl = hex_label(road_ford); bad = [k for k in range(1, int(gl.max()) + 1) if sides([(r, q) for r, q in np.argwhere(gl == k)]) != 2]
    dry = np.isin(gl, bad) & (gl > 0)
    wr_bad = np.isin(gl, np.unique(gl[wr_hex & dry])) & dry   # the WR groups
    road_ford &= ~dry
    dry = (dry & ~wr_bad) | (wr_bad & road_ford0)
    riv &= ~dry
    report['road_crossings_left_dry'] = {'bridges': len(bad), 'hexes': int(dry.sum()),
                                         'at': [[int(v) for v in np.argwhere(gl == k)[0][::-1]] for k in bad]}
    # Two bridges must not share a bank hex: BridgesGenerator walks a bridge's edges from its first bank
    # hex's first bridge neighbour, so a shared bank sends one bridge's walk into the other and leaves its
    # bridge cliffs without edge data (HLCIGenerator then indexes bridge -1). Close road crossings on one
    # river are joined into one bridge through the river hexes between them; otherwise the later one stays dry.
    old_bridge = (cells[:, :, 3] & BRIDGE) > 0
    joined = dropped = 0
    for _ in range(40):                             # (10 sufficed before the Western Regions rivers)
        gl = hex_label(road_ford | old_bridge)
        clash = None
        for r, q in zip(*np.nonzero(hex_dilate(road_ford) & land & ~riv)):
            ks = {int(gl[b, a]) for a, b in nbs(q, r) if gl[b, a]}
            if len(ks) > 1:
                clash = ks; break
        if clash is None:
            break
        ka, kb = sorted(clash)[:2]
        ga, gb = gl == ka, gl == kb
        link = hex_dilate(ga) & hex_dilate(gb) & riv & ~road_ford
        if link.any() and not (old_bridge & (ga | gb)).any() and sides([(r, q) for r, q in np.argwhere(ga | gb | link)]) == 2:
            road_ford |= link; joined += 1
        else:
            drop = (gb if not (old_bridge & gb).any() else ga) & road_ford
            if not drop.any():
                break
            riv &= ~drop; road_ford &= ~drop; dropped += 1
    report['road_crossings_joined'] = joined; report['road_crossings_dropped_near_other_bridge'] = dropped


    near_bridge = hex_dilate(hex_dilate(road_ford | old_bridge))   # spaced fords keep clear of every other bridge's banks
    # the Western Regions pieces (most hexes of a WR river)
    wr_piece = {k: collections.Counter(fam_all[r, q] in WR_FAMILIES for r, q in hexes).most_common(1)[0][0]
                for k, hexes in pieces.items()}
    wr_keys = [k for k in sorted(pieces) if wr_piece[k]]
    base_keys = [k for k in sorted(pieces) if not wr_piece[k]]
    wr_mask = np.isin(plab, wr_keys) & (plab > 0)

    def spaced(spacing, keys, near):
        out = np.zeros((H, W), bool)
        for k in keys:
            hexes = pieces[k]
            last = -10 ** 9; placed = [tuple(p) for p in np.argwhere(road_ford & (plab == k))]
            for r, q in hexes:
                if road_ford[r, q]:
                    last = dist[r, q]; continue
                if not riv[r, q] or dist[r, q] - last < spacing:
                    continue
                if near[r, q] or dtown[r, q] < FORD_TOWN_CLEAR or sides([(r, q)]) != 2:
                    continue
                ok = [p for p in nbs(q, r) if bank_ok(*p)]
                if not any(b2 not in nbs(*b1) for i, b1 in enumerate(ok) for b2 in ok[i + 1:]):
                    continue
                if any((q - fq) ** 2 + (r - fr_) ** 2 < (spacing // 2) ** 2 for fr_, fq in placed):
                    continue
                placed.append((r, q)); last = dist[r, q]; out[r, q] = True
        return out
    spacing = FORD_SPACING
    while True:                                     # the other rivers first, exactly as before the WR rivers
        extra = spaced(spacing, base_keys, near_bridge)
        groups = len(np.unique(hex_label((road_ford & ~wr_mask) | extra))) - 1
        if groups <= budget or spacing > 400:
            break
        spacing += 4

    def border_ford(r, q):
        """The river hex's passable banks hold playable land of two regions (vanilla's fords join two)."""
        return len({int(ids[b, a]) for a, b in nbs(q, r) if bank_ok(a, b)}) >= 2

    def grow_ford(group, others):
        """Vanilla ford shape: a ford group grown along its river to WR_FORD_HEXES hexes, by the river hexes
        of its piece nearest its first hex along the river, each FORD_TOWN_CLEAR or more from towns and two
        hexes clear of every other bridge, while the group keeps exactly two land sides."""
        k = int(plab[group[0]]); d0 = int(dist[group[0]])
        clear = ~hex_dilate(hex_dilate(others))
        g = list(group)
        while len(g) < WR_FORD_HEXES:
            gm = set(g)
            cand = sorted({(b, a) for r, q in g for a, b in nbs(q, r)
                           if (b, a) not in gm and plab[b, a] == k and riv[b, a] and not road[b, a]
                           and dtown[b, a] >= FORD_TOWN_CLEAR and clear[b, a]},
                          key=lambda h: (abs(int(dist[h]) - d0), h))
            for h in cand:
                if sides(g + [h]) == 2:
                    g.append(h); break
            else:
                break
        return g

    # Western Regions rivers: their own spaced fords in the bridges the others leave, each moved to the
    # nearest border hex (banks in two land regions, as vanilla) within half the spacing where there is
    # one; the spacing widens until they fit. Then every WR ford group is grown to the vanilla shape.
    budget_wr = budget - groups
    # WR road crossings beyond that budget stay land (a causeway: the river is cut for the crossing's hexes,
    # polish's gap link draws the channel across; the road crosses on land). Fords go first to crossings
    # whose banks hold two land regions (vanilla), then to the largest rivers, then by position.
    glw = hex_label(road_ford & wr_mask)
    rank_wr = sorted(range(1, int(glw.max()) + 1), key=lambda kk: (
        not any(border_ford(r, q) for r, q in np.argwhere(glw == kk)),
        -len(pieces[int(plab[tuple(np.argwhere(glw == kk)[0])])]), tuple(np.argwhere(glw == kk)[0])))
    causeway = np.isin(glw, rank_wr[max(budget_wr, 0):]) & (glw > 0)
    riv &= ~causeway; road_ford &= ~causeway
    report['wr_causeways'] = [[int(q), int(r)] for r, q in (np.argwhere(glw == kk)[0] for kk in sorted(rank_wr[max(budget_wr, 0):]))]
    near_wr = hex_dilate(hex_dilate(road_ford | old_bridge | extra))
    wr_spacing = WR_FORD_SPACING
    while True:
        cand = spaced(wr_spacing, wr_keys, near_wr)
        extra_wr = np.zeros((H, W), bool)
        for r, q in zip(*np.nonzero(cand)):
            if border_ford(r, q):
                extra_wr[r, q] = True; continue
            k = plab[r, q]; d1 = int(dist[r, q])
            busy = hex_dilate(hex_dilate(extra_wr | (cand & ~(plab == k))))
            alt = [(abs(int(dist[b, a]) - d1), b, a) for b, a in pieces[k]
                   if abs(int(dist[b, a]) - d1) <= wr_spacing // 2 and riv[b, a] and not road_ford[b, a]
                   and not near_wr[b, a] and not busy[b, a] and dtown[b, a] >= FORD_TOWN_CLEAR
                   and border_ford(b, a) and sides([(b, a)]) == 2]
            if alt:
                _, b, a = min(alt); extra_wr[b, a] = True
            elif not busy[r, q]:
                extra_wr[r, q] = True
        ford0 = (road_ford & wr_mask) | extra_wr
        groups_wr = len(np.unique(hex_label(ford0))) - 1
        if groups_wr <= budget_wr:
            break
        if wr_spacing > 400:                        # every piece still takes one: no spaced WR fords at all
            ford0 = road_ford & wr_mask; wr_spacing = None
            break
        wr_spacing += 8
    fl0 = hex_label(ford0)
    shapes = collections.Counter(int((fl0 == kk).sum()) for kk in range(1, int(fl0.max()) + 1))
    grown_wr = np.zeros((H, W), bool)
    allf = road_ford | old_bridge | extra | ford0
    for kk in range(1, int(fl0.max()) + 1):
        gmask = fl0 == kk
        for h in grow_ford(sorted(map(tuple, np.argwhere(gmask))), (allf & ~gmask) | grown_wr):
            grown_wr[h] = True
    extra_wr = grown_wr & ~road_ford
    ford = (road_ford | extra | extra_wr) & ~port
    groups = len(np.unique(hex_label(ford))) - 1
    gl_wr = hex_label(ford & wr_mask)
    report['wr_fords'] = {'road_ford_hexes': int((road_ford & wr_mask).sum()),
                          'road_ford_bridges': len(np.unique(hex_label(road_ford & wr_mask))) - 1,
                          'bridges': len(np.unique(gl_wr)) - 1, 'budget': int(budget_wr), 'spacing_used': wr_spacing,
                          'causeways': len(report['wr_causeways']), 'one_sided_crossings_causeway_hexes': int((wr_bad & road_ford0).sum()),
                          'border_bridges': int(sum(any(border_ford(r, q) for r, q in np.argwhere(gl_wr == kk))
                                                    for kk in range(1, int(gl_wr.max()) + 1))),
                          'group_sizes': dict(sorted(collections.Counter(int((gl_wr == kk).sum()) for kk in range(1, int(gl_wr.max()) + 1)).items())),
                          'group_sizes_before_growth': dict(sorted(shapes.items()))}
    report.update({'road_fords': int(road_ford.sum()), 'road_ford_bridges': len(np.unique(hex_label(road_ford))) - 1,
                   'spaced_fords': int(extra.sum()), 'ford_spacing_used': spacing, 'bridges_before': base_bridges,
                   'bridges_after': base_bridges + groups, 'bridge_limit': MAX_BRIDGES})

    # ---- paint (vanilla templates; a ford keeps its road edges)
    vc = V['cells']
    gnames = V['land_ground_types'] + V['sea_ground_types']
    assert gnames == F['land_ground_types'] + F['sea_ground_types']
    vg = (vc[:, :, 5].astype(int) >> 4) + ((vc[:, :, 6].astype(int) & 7) << 4) - 1

    def template(name, bridge):
        m = (vg == gnames.index(name)) & ((vc[:, :, 0] & 3) == 1) & ((vc[:, :, 2] >> 4) == 0) \
            & (((vc[:, :, 3] & BRIDGE) > 0) == bridge) & ((vc[:, :, 3] & ROAD_BITS) == 0)
        return vc[tuple(np.argwhere(m)[0])].copy()
    t_river, t_ford = template('river_navigable', False), template('river_navigable_shallows', True)
    roadbits = cells[:, :, 3] & ROAD_BITS
    cells[riv & ~ford] = t_river
    cells[ford] = t_ford
    cells[ford, 3] |= roadbits[ford]
    if port.any():                                  # river ports: their slot and sprawl bytes as add_ports wrote them
        cells[port, 2] = ports['cells'][:, 2]; cells[port, 3] = ports['cells'][:, 3]
    opened = int((ford & road & imp).sum())
    import apply_roads as _ar
    rmask = cells[:, :, 4] & 63
    cut = 0
    for r, q in zip(*np.nonzero((rmask > 0) & hex_dilate(riv) & ~riv)):
        for d, (dq, dr) in enumerate(_ar.DIRS[q & 1]):
            a, b = q + dq, r + dr
            if 0 <= a < W and 0 <= b < H and riv[b, a] and cells[r, q, 4] >> d & 1:
                cells[r, q, 4] &= ~np.uint8(1 << d); cut += 1
    report['land_river_edges_into_water_cleared'] = cut
    water = riv
    bank = hex_dilate(water) & land & ~water & ~native & (slot == 0) & ~sprawl & ~aoi
    fbank = hex_dilate(ford) & bank
    gcross = gnames.index('river_navigable_crossing') + 1
    beach = bank & ~fbank & (kind == 0)
    cells[beach, 0] = (cells[beach, 0] & 0xFC) | 2
    cliff = fbank & (kind != 1)
    cells[cliff, 0] = (cells[cliff, 0] & 0xFC) | 3
    cross = np.zeros((H, W), bool)
    for r, q in zip(*np.nonzero(fbank)):
        if bank_ok(q, r) or road[r, q]:
            cross[r, q] = True
    cells[cross, 5] = (cells[cross, 5] & 0x0F) | ((gcross & 15) << 4)
    cells[cross, 6] = (cells[cross, 6] & 0xF8) | ((gcross >> 4) & 7)
    open_road = fbank & road & imp
    cells[open_road, 2] &= 0xF7
    report['road_hexes_opened'] = opened + int(open_road.sum())

    # ---- river regions, vanilla style: each river is cut into reaches (at a confluence with another river,
    # or where the next stretch would border more than MAX_ADJ_LAND land regions or hold more than
    # RIVER_AREAS SDK areas), and a river's reaches are packed into as few regions as those limits allow.
    # Deterministic: pieces by their westernmost hex, BFS layers from it, rivers and regions in name order.
    fam = np.where(riv, fam_all, '')
    landreg = land & ~riv & (ids < len(F['land_regions']))

    def adj_of(hexes):
        return {int(ids[b, a]) for r, q in hexes for a, b in nbs(q, r) if landreg[b, a]}

    def areas_of(mask):
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min() - xs.min() % 2, xs.max() + 1   # even start keeps column parity
        mm, ff = mask[y0:y1, x0:x1], ford[y0:y1, x0:x1]
        return int(hex_label(mm & ~ff).max() and len(np.unique(hex_label(mm & ~ff))) - 1) +             (len(np.unique(hex_label(mm & ff))) - 1 if (mm & ff).any() else 0)
    reaches = []                                    # [river, hexes, adjacent land regions]
    for k in sorted(pieces, key=lambda k: pieces[k][0]):
        layers = collections.defaultdict(list)
        for r, q in pieces[k]:
            if riv[r, q]:
                layers[int(dist[r, q])].append((r, q))
        cur, cadj, cfam = [], set(), None
        for d in sorted(layers):
            L = layers[d]
            la = adj_of(L)
            if cur and len(cadj | la) > MAX_ADJ_LAND:
                reaches.append([None, cur, cadj]); cur, cadj = [], set()
            cur += L; cadj |= la
        if cur:
            reaches.append([None, cur, cadj])
    for rc in reaches:                              # a reach is named after the river with most of its hexes
        rc[0] = collections.Counter(fam[r, q] for r, q in rc[1]).most_common(1)[0][0]
    out = []                                        # a reach above RIVER_AREAS is halved along its layers
    while reaches:
        f_, hx, ad = reaches.pop(0)
        m = np.zeros((H, W), bool); m[tuple(np.array(hx).T)] = True
        n_ = areas_of(m)
        if n_ > RIVER_AREAS and len(hx) > 1:
            hx = sorted(hx, key=lambda h: dist[h]); h2 = len(hx) // 2
            reaches[:0] = [[f_, hx[:h2], adj_of(hx[:h2])], [f_, hx[h2:], adj_of(hx[h2:])]]
            continue
        out.append((f_, hx, ad, n_, m))
    bins = []                                       # per river, first fit
    for f_, hx, ad, n_, m in sorted(out, key=lambda x: (x[0], -len(x[2]), -x[3], min(x[1]))):
        for b in bins:
            if b['fam'] == f_ and len(b['adj'] | ad) <= MAX_ADJ_LAND:
                n2 = areas_of(b['mask'] | m)
                if n2 <= RIVER_AREAS:
                    b['adj'] |= ad; b['areas'] = n2; b['mask'] |= m; break
        else:
            bins.append({'fam': f_, 'adj': set(ad), 'areas': n_, 'mask': m.copy()})
    limit = 511 - len(names) - FREE_IDS
    filler_id = names.index(FILLER)

    def basin(f):
        return RIVER_BASIN.get(f, 'frontier')

    def centroid(m):
        ys, xs = np.nonzero(m)
        return float(np.mean(xs, dtype=np.float64)), float(np.mean(ys, dtype=np.float64))
    for b in bins:
        b['cen'] = centroid(b['mask'])

    def pack(bins, limit):
        while len(bins) > limit:                      # over budget: merge two compatible regions
            # within one basin (RIVER_BASIN) before across basins; then rivers bordering only the wilderness
            # (the filler region: they carry no gameplay), the same river, the fewest bordering land regions,
            # the nearest (centroids) and the fewest SDK areas.
            # (Merging by adjacency and areas alone put the Salween into an Amur region and the upper
            # Yellow River and Yangtze into a Lena region.)
            pairs = []
            for i in range(len(bins)):
                for j in range(i + 1, len(bins)):
                    bi, bj = bins[i], bins[j]
                    if len(bi['adj'] | bj['adj']) > MAX_ADJ_LAND or bi['areas'] + bj['areas'] > RIVER_AREAS:
                        continue
                    kin = 0 if bi['fam'] == bj['fam'] else 1 if basin(bi['fam']) == basin(bj['fam']) else 2
                    gap = math.hypot(bi['cen'][0] - bj['cen'][0], bi['cen'][1] - bj['cen'][1])
                    play = len((bi['adj'] | bj['adj']) - {filler_id})
                    pairs.append((kin == 2, play > 0, kin, play, round(gap, 6), bi['areas'] + bj['areas'], i, j))
            if not pairs:
                break
            i, j = min(pairs)[-2:]
            big = i if bins[i]['mask'].sum() >= bins[j]['mask'].sum() else j
            bins[i]['fam'] = bins[big]['fam']; bins[i]['adj'] |= bins[j]['adj']; bins[i]['mask'] |= bins[j]['mask']
            bins[i]['areas'] = areas_of(bins[i]['mask']); bins[i]['cen'] = centroid(bins[i]['mask']); del bins[j]
        return bins
    # the Western Regions rivers are packed on their own (WR_REGIONS ids), the others exactly as before
    wr_bins = pack([b for b in bins if b['fam'] in WR_FAMILIES], WR_REGIONS)
    bins = pack([b for b in bins if b['fam'] not in WR_FAMILIES], limit)

    def region_name(m):
        """The river holding BASIN_SHARE of the hexes, else the basin holding it, else 'frontier'."""
        fc = collections.Counter(fam[m].tolist()); total = sum(fc.values())
        top, n = min(fc.items(), key=lambda kv: (-kv[1], kv[0]))
        if n >= BASIN_SHARE * total:
            return top
        bc = collections.Counter()
        for f, v in fc.items():
            bc[basin(f)] += v
        top, n = min(bc.items(), key=lambda kv: (-kv[1], kv[0]))
        return top if n >= BASIN_SHARE * total else 'frontier'
    for b in bins + wr_bins:
        b['fam'] = region_name(b['mask'])
    # ---- channels -> regions. The packed bins above are the river CHANNELS (key = b['fam'], the family
    # polish's river channels plan their gap links and sand ends by; river-channels.json keeps them per
    # hex). Regions merge the channels of one drainage system (drainage(); vanilla keeps all its inland
    # lakes in one non-contiguous region, 3k_main_sea_lake): first fit in channel order while the merged
    # region stays at or below REGION_AREAS SDK areas, else the system takes a second region.
    order = lambda b: (b['fam'], tuple(np.argwhere(b['mask'])[0]))
    systems = collections.defaultdict(list)
    for b in sorted(bins, key=order) + sorted(wr_bins, key=order):
        systems[drainage(b['fam'])].append(b)
    regions = []
    for g in sorted(systems):
        mine = []
        for b in systems[g]:
            for rg in mine:
                n2 = areas_of(rg['mask'] | b['mask'])
                if n2 <= REGION_AREAS:
                    rg['mask'] = rg['mask'] | b['mask']; rg['adj'] = rg['adj'] | b['adj']; rg['areas'] = n2
                    rg['channels'].append(b); break
            else:
                mine.append({'group': g, 'mask': b['mask'].copy(), 'adj': set(b['adj']), 'areas': b['areas'], 'channels': [b]})
        regions += mine
    sea = list(F['sea_regions']); river_regions, loads, adjacency, contents, channel_of = [], [], [], [], []
    count = collections.Counter()
    ccount = collections.Counter()
    channel_hexes = collections.defaultdict(list)
    for rg in regions:                              # drainage systems in name order, then first-fit order
        count[rg['group']] += 1
        key = f"{RIVER_PREFIX}{rg['group']}_{count[rg['group']]}"
        river_regions.append(key); sea.append(key); loads.append(rg['areas']); adjacency.append(len(rg['adj']))
        ids[rg['mask']] = len(F['land_regions']) + sea.index(key)
        fc = collections.Counter(fam[rg['mask']].tolist())
        contents.append({f: fc[f] for f in sorted(fc)})
        chans = []
        for b in rg['channels']:
            ccount[b['fam']] += 1
            chans.append(f"{b['fam']}_{ccount[b['fam']]}")
            channel_hexes[b['fam']] += [[int(q), int(r)] for r, q in np.argwhere(b['mask'])]
        channel_of.append(chans)
    # drawn-only hexes (river spec option B): land in map.hex, carved and watered as a visual link
    drawn_hexes = collections.defaultdict(list)
    for r, q in np.argwhere(drawn):
        drawn_hexes[str(fam_all[r, q])].append([int(q), int(r)])
    channels_doc = {'format': 1, 'by': 'apply_rivers.py',
                    'note': 'river channel key per frontier river hex [q, r]; polish_campaign_terrain plans the '
                            'river channels by it (river_channels.river_families), not by the region names. '
                            'drawn: land hexes a river crosses inside one region (river spec option B), drawn only',
                    'channels': {k: sorted(channel_hexes[k]) for k in sorted(channel_hexes)},
                    'drawn': {k: sorted(drawn_hexes[k]) for k in sorted(drawn_hexes)}}
    report['river_region_limit'] = limit
    report['wr_region_limit'] = WR_REGIONS
    report['region_areas_limit'] = REGION_AREAS
    map_hex.set_region_ids(cells, ids)
    allnames = F['land_regions'] + sea
    changed = (cells != F['cells']).any(2)
    for r, q in zip(*np.nonzero(hex_dilate(changed))):
        cells[r, q, 15] = edge_mask(ids, q, r)
    pal = list(F['palettes'])
    if len(pal) > 1 and len(pal[1]) == 4 * len(F['sea_regions']):
        src = F['sea_regions'].index(next(n for n in F['sea_regions'] if '_riv_sea_' in n))
        pal[1] = pal[1] + pal[1][4 * src:4 * src + 4] * len(river_regions)

    # ---- checks
    areas = region_areas(ids, cells)
    top = [(allnames[k], v) for k, v in areas.most_common(8)]
    blocked = {allnames[k]: v for k, v in areas.items() if v > SDK_AREA_LIMIT}
    river_sdk = {k: int(areas[allnames.index(k)]) for k in river_regions}   # class-aware (hex_components)
    river_over = {k: v for k, v in river_sdk.items() if v > REGION_AREAS}
    g0 = ar.Grid(F['cells'], {'cells': vc, 'names': V['land_regions'] + V['sea_regions']}, names)
    mask0 = g0.mask.copy()
    g1 = ar.Grid(cells, {'cells': vc, 'names': V['land_regions'] + V['sea_regions']}, allnames)
    mask1 = g1.mask.copy()
    before, after = ar.trace_towns(g0, mask0), ar.trace_towns(g1, mask1)
    pairs = lambda p: {k for k in p if k != 'off-map' and k[0] != k[1]}
    fb, fa = ar.components(before), ar.components(after)
    split = [p for p in pairs(before) if fa(p[0]) != fa(p[1])]
    fl = hex_label(ford)
    ford_sides = [tuple(int(v) for v in np.argwhere(fl == k)[0][::-1]) for k in range(1, int(fl.max()) + 1)
                  if sides([(r, q) for r, q in np.argwhere(fl == k)]) != 2]
    protected = (native | (slot > 0) | sprawl | aoi | (kind == 1)) & ~port   # river ports change as planned (port_bad checks them)
    dtown30 = hex_steps(town, 30)
    ford_town = sorted(int(dtown30[fl == k].min()) for k in range(1, int(fl.max()) + 1))
    near_fords = [tuple(int(v) for v in np.argwhere(fl == k)[0][::-1]) for k in range(1, int(fl.max()) + 1)
                  if dtown30[fl == k].min() < FORD_TOWN_CLEAR]
    report['ford_town_distance'] = dict(sorted(collections.Counter(ford_town).items()))
    report['bridges_near_towns'] = near_fords
    report['river_beside_town_slots'] = int((hex_dilate(water) & (slot > 0)).sum())
    report.update({
        'river_hexes': int(water.sum()), 'fords': int(ford.sum()), 'pieces': len(pieces),
        'revealed_river_hexes': int((water & revealed).sum()), 'revealed_bank_hexes': int((bank & revealed).sum()),
        'river_regions': {k: {'areas': a, 'adjacent_land_regions': j, 'channels': ch, 'river_hexes': f}
                          for k, a, j, ch, f in zip(river_regions, loads, adjacency, channel_of, contents)},
        'max_adjacent_land_regions': max(adjacency), 'total_regions': len(allnames),
        'free_ids_after_reserve': 511 - len(allnames) - RESERVED_IDS,
        'banks_to_beach': int(beach.sum()), 'banks_to_bridge_cliff': int(cliff.sum()), 'crossing_ground': int(cross.sum()), 'road_hexes_to_beach': int((beach & road).sum()),
        'sdk_areas_top': top, 'sdk_area_limit': SDK_AREA_LIMIT, 'sdk_area_overflow': blocked,
        'river_region_sdk_areas': river_sdk, 'river_region_sdk_areas_over_limit': river_over,
        'checks': {'road_edges_unchanged': bool((mask0 == mask1).all()),
                   'traced_pairs_before': len(pairs(before)), 'traced_pairs_after': len(pairs(after)),
                   'traced_pairs_identical': pairs(before) == pairs(after), 'town_links_split': split,
                   'protected_cells_unchanged': bool((cells[protected] == F['cells'][protected])[:, :15].all()),
                   'bridges_without_two_sides': ford_sides[:20], 'bridges_without_two_sides_count': len(ford_sides)}})
    rivreg = np.isin(ids, [allnames.index(k) for k in river_regions])
    pq = [[int(q), int(r)] for r, q in zip(*np.nonzero(port))]
    port_bad = [h for h, c in zip(pq, ports['cells'] if port.any() else []) if not rivreg[h[1], h[0]]
                or cells[h[1], h[0], 2] != c[2] or cells[h[1], h[0], 3] != c[3]]
    plb = hex_label(port)
    port_alone = []                                 # port groups with no other river hex beside them
    for kp in range(1, int(plb.max()) + 1):
        gp = plb == kp
        if not (hex_dilate(gp) & rivreg & ~port).any():
            r_, q_ = np.argwhere(gp)[0]
            port_alone.append([int(q_), int(r_)])
    report['checks'].update({'river_ports_kept': not port_bad, 'river_ports_bad': port_bad,
                             'river_ports_without_river': port_alone})
    report['failed_by'] = {k: bool(v) for k, v in {
        'blocked': blocked, 'ids_reserved': len(allnames) > 511 - RESERVED_IDS, 'road_edges': not report['checks']['road_edges_unchanged'],
        'split': split, 'ids_free': len(allnames) > 511 - FREE_IDS + WR_REGIONS, 'adjacency': max(adjacency) > MAX_ADJ_REGION,
        'protected': not report['checks']['protected_cells_unchanged'], 'ford_sides': ford_sides, 'bridges': report['bridges_after'] > MAX_BRIDGES,
        'near_fords': near_fords, 'water_on_town': (water & town & ~port).any(), 'loads': max(loads) > REGION_AREAS,
        'river_over': river_over, 'port_bad': port_bad, 'port_alone': port_alone}.items() if v}
    failed = blocked or len(allnames) > 511 - RESERVED_IDS or not report['checks']['road_edges_unchanged'] \
        or split or len(allnames) > 511 - FREE_IDS + WR_REGIONS or max(adjacency) > MAX_ADJ_REGION or not report['checks']['protected_cells_unchanged'] or ford_sides or report['bridges_after'] > MAX_BRIDGES \
        or near_fords or (water & town & ~port).any() or max(loads) > REGION_AREAS or river_over or port_bad or port_alone
    return {'cells': cells, 'ids': ids, 'water': water, 'ford': ford, 'road': road, 'allnames': allnames, 'sea': sea,
            'pal': pal, 'river_regions': river_regions, 'report': report, 'channels_doc': channels_doc, 'failed': failed}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--replace', action='store_true', help='undo the rivers already on the map first (strip_rivers)')
    ap.add_argument('--pre-river', type=Path, help='--replace: the map the earlier river step read (backups/rivers-<time>/map.hex)')
    ap.add_argument('--post-river', type=Path, help='--replace: the map the earlier river step wrote')
    ap.add_argument('--pre-river-of', type=Path, nargs=2, metavar=('PRE0', 'POST0'),
                    help='--replace when the earlier river step was itself a --replace of PRE0 / POST0: the map it '
                         'read is strip_rivers(--pre-river, PRE0, POST0) (newer steps save it as map-stripped.hex)')
    ap.add_argument('--debug-map', type=Path, help='also write the candidate map here even when a check fails (offline inspection)')
    ap.add_argument('--no-wr', action='store_true', help='without the Western Regions rivers (checks that the others are unchanged)')
    ap.add_argument('--write-to', type=Path, help='write the map here instead (no backup, no design change): offline checks')
    args = ap.parse_args()
    # Stop-loss 2026-10-26 (user, v17 start crash): the Western Regions rivers cut ~40 land regions into pieces
    # (1-4 hex islands; the campaign's border tracing crashed at 617,1035 beside tw3k_riv_sea_tarim_1). Off unless
    # TW3K_WR_RIVERS=1; they come back with the river spec in the map bake, river lines and region borders together.
    if os.environ.get('TW3K_WR_RIVERS', '0') != '1':
        args.no_wr = True
    F, V = map_hex.read(FINAL), map_hex.read(cw.VANILLA_MAP)
    strip_report = None
    ports = None
    if args.replace:
        if not (args.pre_river and args.post_river):
            raise SystemExit('--replace needs --pre-river and --post-river')
        P = map_hex.read(args.pre_river)
        if args.pre_river_of:
            P, chain_report, _ = strip_rivers(P, map_hex.read(args.pre_river_of[0]), map_hex.read(args.pre_river_of[1]))
            print('strip_rivers (pre-river)', json.dumps(chain_report), flush=True)
            if chain_report['conflict_hexes'] or chain_report['region_conflict_hexes']:
                raise SystemExit('--pre-river-of: the earlier strip has conflicts')
        F, strip_report, ports = strip_rivers(F, P, map_hex.read(args.post_river))
        print('strip_rivers', json.dumps(strip_report), flush=True)
        stripped = F
    strip_info = None
    if strip_report is not None:
        strip_info = dict(strip_report, pre_river=str(args.pre_river), post_river=str(args.post_river),
                          pre_river_of=[str(v) for v in args.pre_river_of] if args.pre_river_of else None)
    res = lay_rivers(F, V, ports=ports, wr=not args.no_wr, revealed_rivers=os.environ.get('TW3K_REVEALED_RIVERS', '0') == '1',
                     strip_info=strip_info)
    cells, ids, water, ford, road, allnames, sea, pal, river_regions, report, channels_doc, failed = (res[k] for k in (
        'cells', 'ids', 'water', 'ford', 'road', 'allnames', 'sea', 'pal', 'river_regions', 'report', 'channels_doc', 'failed'))
    OUT.mkdir(parents=True, exist_ok=True)
    preview(cells, ids, water, ford, road, allnames, river_regions, report, OUT / 'rivers-preview.png')
    write_text_atomic(OUT / 'rivers-apply-report.json', json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ('warnings',)}, ensure_ascii=False, indent=1))
    if args.debug_map:                              # offline inspection, written even when a check fails
        map_hex.write(dict(F, sea_regions=sea, cells=cells, palettes=pal), args.debug_map)
    if failed:
        raise SystemExit(f'River apply fails its checks; nothing written (see {OUT / "rivers-apply-report.json"})')
    channels_text = json.dumps(channels_doc, separators=(',', ':')) + chr(10)
    if args.write_to:
        map_hex.write(dict(F, sea_regions=sea, cells=cells, palettes=pal), args.write_to)
        import region_connectivity; region_connectivity.check_map(args.write_to)   # land regions stay whole (v17 crash); [to be absorbed by the map bake]
        write_text_atomic(channels_path(args.write_to), channels_text)
        print('Map written to', args.write_to, '(no backup, design unchanged); channels', channels_path(args.write_to))
        return
    if not args.apply:
        print('Dry run only; add --apply to write map.hex and the design.')
        return

    back = A / f'backups/rivers-{time.strftime("%Y%m%d-%H%M%S")}'
    back.mkdir(parents=True)
    for p in (FINAL, cw.DESIGN_PATH, channels_path(FINAL)):
        if p.exists():
            shutil.copy2(p, back / p.name)
    if args.replace:                                # the map this step read: the next --replace's --pre-river
        map_hex.write(stripped, back / 'map-stripped.hex')
    map_hex.write(dict(F, sea_regions=sea, cells=cells, palettes=pal), FINAL)
    import region_connectivity; region_connectivity.check_map(FINAL)   # land regions stay whole (v17 crash); [to be absorbed by the map bake]
    map_hex.read(FINAL)
    write_text_atomic(channels_path(FINAL), channels_text)
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    used = {tuple(c['colour']) for c in design['regions']} | {tuple(v['colour']) for v in design.get('imported_sea_regions', [])}
    imported = design.setdefault('imported_sea_regions', [])
    if args.replace:                                # the replaced rivers' entries go, those that persist stay
        imported[:] = [v for v in imported if not str(v.get('region', '')).startswith(RIVER_PREFIX) or v.get('region') in river_regions]
        used = {tuple(c['colour']) for c in design['regions']} | {tuple(v['colour']) for v in imported}
    for k, rr in enumerate(river_regions):
        if not any(v['region'] == rr for v in imported):
            imported.append({'region': rr, 'source_region': None, 'is_sea': True, 'colour': unique_colour(used, 4242 + k),
                             'purpose': 'frontier rivers (Natural Earth, apply_rivers.py)'})
    budget = design.setdefault('sdk_region_budget', {})
    budget['total_regions'] = len(allnames)
    write_text_atomic(cw.DESIGN_PATH, json.dumps(design, ensure_ascii=False, indent=2) + '\n')
    report['backup'] = str(back)
    (A / 'analysis').mkdir(exist_ok=True)
    write_text_atomic(A / 'analysis/rivers-apply-report.json', json.dumps(report, ensure_ascii=False, indent=1))
    print('Rivers written; backup at', back)


def preview(cells, ids, water, ford, road, names, river_regions, report, path):
    kind = cells[:, :, 0] & 3; imp = (cells[:, :, 2] & 8) > 0
    img = np.zeros(cells.shape[:2] + (3,), np.uint8)
    img[:] = (222, 214, 190)
    img[imp & (kind != 1)] = (170, 160, 140)
    img[np.isin(ids, [i for i, n in enumerate(names) if 'non_playable' in n]) & (kind != 1)] = (150, 150, 150)
    img[kind == 1] = (120, 160, 200)
    img[road & ~water] = (120, 70, 30)
    pal = [(200, 30, 30), (20, 110, 40), (120, 40, 160), (220, 120, 0), (0, 120, 160), (160, 0, 90)]
    for i, rr in enumerate(river_regions):
        img[water & (ids == names.index(rr))] = pal[i % len(pal)]
    img = img[::-1]
    im = Image.fromarray(img); d = ImageDraw.Draw(im); H = cells.shape[0]
    for r, q in zip(*np.nonzero(ford)):
        y = H - 1 - r
        d.ellipse([q - 3, y - 3, q + 3, y + 3], outline=(0, 0, 0), fill=(255, 0, 255) if road[r, q] else (255, 255, 0))
    d.rectangle([8, 8, 760, 34 + 22 * len(river_regions)], fill=(255, 255, 255))
    d.text((14, 12), f"rivers {report['river_hexes']} hexes, fords {report['fords']} (road {report['road_fords']}, spaced "
           f"{report['spaced_fords']}), {len(river_regions)} river regions; roads brown", fill=(0, 0, 0))
    for i, rr in enumerate(river_regions):
        d.rectangle([14, 34 + 22 * i, 30, 50 + 22 * i], fill=pal[i % len(pal)])
        d.text((38, 36 + 22 * i), f"{rr}: {int((water & (ids == names.index(rr))).sum())} hexes, {report['river_regions'][rr]['areas']} areas, {report['river_regions'][rr]['adjacent_land_regions']} land neighbours", fill=(0, 0, 0))
    im.save(path)


if __name__ == '__main__':
    main()

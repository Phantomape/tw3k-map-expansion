"""Ground patchiness audit of a campaign global blend (hard straight edges, dark islands, farm squares).

  python audit_ground_patchiness.py vanilla              (alarm levels from vanilla -> analysis/ground-patchiness-vanilla.json)
  python audit_ground_patchiness.py run [PACK] [--label NAME] [--season harvest] [--windows-only]
  python audit_ground_patchiness.py run --blend global_blend.dds [--lf lf_height_map.dds] --label NAME
  python audit_ground_patchiness.py baseline [PACK]      (v18 window numbers -> analysis/ground-patchiness-baseline.json)
  python audit_ground_patchiness.py validate [PACK]      (six v19 live close-ups beside the heat crops)

Metric definitions: ground_patchiness.py; validation and tuning notes: analysis/ground-patchiness.md.
Slot tones and the hard-pair threshold come from the ground tone table (ground_tone: biome_profiles
CONSTS MATERIAL_TONE / CONTRAST, measured by build_material_tones.py), the one table the ground bake reads.

vanilla: every metric over the vanilla dlc07 blend (vanilla packs; land = LF above the water plane), per
vanilla land region (map.hex regions with >= MIN_REGION_HEXES land hexes, leaving out the non-playable filler
and EDGE_OF_WORLD_HEX hexes beside it); the alarm level of each
ALARM_METRIC is its ALARM_PCT[metric] percentile over those regions. A window above an alarm level is 'lit'.
run: numbers for the whole map and every window (terrain_regression.WINDOWS, which include the six
live-check camera boxes) -> <out>/patchiness.json; heatmaps (north up; grey = ground L*, orange = hard
edges, red = straight hard segments, blue = dark islands, magenta = farm / wetland squares) ->
<out>/map.png (1 px per hex) and <out>/windows/<name>.png. Default out: output/offline-view/patchiness/<label>.
The whole-map passes read ~1 GB and peak at a few GB: take a pipeline lock slot (3 or 4) first;
--windows-only reads row bands only.

Read-only: packs / DDS files are read with plain reads (never memory-mapped, nothing kept open); nothing
but the outputs above is written. Never point --blend at a live workspace stage while a pipeline runs.
"""
import tw3k_paths
import argparse, json, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import campaign_world as cw
import ground_patchiness as gp
import terrain_regression
from terrain_regression import WINDOWS, part_box

terrain_regression.cw = cw     # part_box reads the module global terrain_regression sets in Snapshot

ROOT = gp.ROOT
CANDIDATE = tw3k_paths.MAIN_ROOT / 'output/tw3k_regions_and_battles_terrain_repair_candidate.pack'
V18 = tw3k_paths.MAIN_ROOT / 'output/tw3k_regions_and_battles_terrain_repair_candidate.previous-1793078765845440200.pack'
OUT = tw3k_paths.MAIN_ROOT / 'output/offline-view/patchiness'
BASELINE = ROOT / 'analysis/ground-patchiness-baseline.json'
MIN_REGION_HEXES = 50
EDGE_OF_WORLD_HEX = 8
PCTS = (50, 75, 90, 95, 99)
# v19 live close-ups (output/live-canvas-2048s, analysis/live-check-canvas-2048s-result.md): place, window,
# shot, camera (x, z), and whether the shot shows the fault (True) or is a control
LIVE = (('ring-west', 'ring-west-ground', 'S19-ring-west-close.png', (588.72, 539.52), True),
        ('funan', 'funan-ground', 'S19-funan-close.png', (730.89, 206.94), True),
        ('indus', 'indus-ground', 'S19-indus-close.png', (313.71, 462.44), True),
        ('taklamakan', 'taklamakan-ground', 'S19-taklamakan-close.png', (397.82, 756.86), False),
        ('mongolia', 'mongolia-ground', 'S19-mongolia-close.png', (694.18, 917.18), False),
        ('siberia', 'siberia-ground', 'S19-siberia-close.png', (1121.36, 1348.79), False))
LIVE_DIR = tw3k_paths.MAIN_ROOT / 'output/live-canvas-2048s'


def source(args):
    if args.blend:
        return gp.BlendSource(blend=args.blend, lf=args.lf), tw3k_paths.portable(args.blend)
    return gp.BlendSource(pack=args.pack), tw3k_paths.portable(args.pack)   # no machine path in tracked reports


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n',
                    encoding='utf-8')


def window_numbers(band, parts, T, acc=None, islands=None):
    """Numbers of a window over all its parts (from the whole-map accumulators when given, else per part box
    with context) and the heat layers of each part."""
    if acc is not None:
        sel = np.zeros(acc.shape, bool)
        for q0, q1, r0, r1 in parts:
            sel[r0:r1 + 1, q0:q1 + 1] = True
        heat = gp.heat_layers(acc)
        return (gp.summarise(acc, [i for i in islands if sel[i['centre_hex'][1], i['centre_hex'][0]]], sel),
                [{k: v[r0:r1 + 1, q0:q1 + 1] for k, v in heat.items()} for q0, q1, r0, r1 in parts])
    tot, isl, maps = None, [], []
    for q0, q1, r0, r1 in parts:
        _, m, inbox, sums = gp.analyse_box(band, q0, q1, r0, r1, T)
        maps.append(m); isl += inbox
        tot = gp.merge_sums(tot, sums)
    return gp.numbers(tot, isl), maps


def save_heat(maps, path, scale=4):
    ims = [Image.fromarray(gp.heat_rgb(m, scale)) for m in maps]
    out = Image.new('RGB', (sum(i.width for i in ims) + 6 * (len(ims) - 1), max(i.height for i in ims)), (255, 255, 255)); x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width + 6
    out.save(path)


def thresholds(T):
    return {'hard_pair_dE': T.max_de, 'tones': T.source, 'season': T.season, 'alarm': gp.alarms(),
            **{k: getattr(gp, k) for k in ('HARD_SIDE_FRAC', 'SIDE_PX', 'STRAIGHT_R', 'STRAIGHT_COH', 'STRAIGHT_MIN', 'SEG_MIN_PX',
                                           'ISLAND_DL', 'ISLAND_LEVELS', 'ISLAND_MIN_TILES', 'ISLAND_MAX_TILES', 'ISLAND_RING',
                                           'ISLAND_RING_SHARE', 'SQUARE_SLOTS', 'SQUARE_MIN_PX', 'SQUARE_MAX_PX')}}


def run(args):
    t0 = time.time()
    src, name = source(args)
    T = gp.Tones(args.season, args.tones)
    out = (args.out or OUT / args.label).resolve()
    (out / 'windows').mkdir(parents=True, exist_ok=True)
    rep = {'kind': 'ground-patchiness', 'version': 1, 'source': name, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'thresholds': thresholds(T), 'windows': {}}
    acc = isl = None
    if not args.windows_only:
        acc, isl = gp.analyse_map(src, T, workers=args.workers, cache=args.cache)
        rep['global'] = gp.summarise(acc, isl)
        rep['global_islands_top'] = sorted(isl, key=lambda i: -i['tiles'])[:40]
        Image.fromarray(gp.heat_rgb(gp.heat_layers(acc), 1)).save(out / 'map.png')
        print('global', json.dumps(rep['global']), flush=True)
    for wname, parts, watch, _ in WINDOWS:
        pb = [part_box(p) for p in parts]
        nums, maps = window_numbers(src.band, pb, T, acc, isl)
        rep['windows'][wname] = {'parts': [list(p) for p in pb], 'watch': watch, 'metrics': nums}
        save_heat(maps, out / 'windows' / f'{wname}.png')
        print(f"{wname:26s} seg/100 {nums['straight_segment_per100']:6.2f}  island {nums['dark_island_share']:6.3f}  "
              f"square/100 {nums['square_per100']:6.2f}  hard/100 {nums['hard_edge_per100']:6.2f}  lit {','.join(nums['lit']) or '-'}", flush=True)
    rep['runtime_s'], rep['peak_mb'] = round(time.time() - t0, 1), terrain_regression.peak_mb()
    dump(out / 'patchiness.json', rep)
    print(f"-> {out}  ({rep['runtime_s']} s, peak {rep['peak_mb']} MB)")
    return rep


def vanilla(args):
    """Alarm levels from vanilla's own distribution, region by region."""
    import map_hex
    t0 = time.time()
    T = gp.Tones(args.season, args.tones)
    src = gp.BlendSource.vanilla()
    acc, isl = gp.analyse_map(src, T)
    cells = src.cells
    ids = np.where((cells[:, :, 0] & 3) != 1, map_hex.region_ids(cells), -1)   # land regions (pixels: LF land)
    names = map_hex.read(gp.vanilla_map())['land_regions']
    # the non-playable filler and the playable ground within EDGE_OF_WORLD_HEX of it are left out: vanilla cuts
    # its ground off there with long straight lines nobody sees under the shroud (the longest straight segments
    # of the first pass were all on that border: yizhou island, cangwu, nanhai, shoufang)
    from scipy import ndimage
    npl = np.isin(ids, [i for i, n in enumerate(names) if 'non_playable' in n])
    ids[ndimage.binary_dilation(npl, iterations=EDGE_OF_WORLD_HEX)] = -1
    groups = gp.summarise_groups(acc, ids[:acc.shape[0], :acc.shape[1]], isl, alarm={})
    regions = {names[g] if 0 <= g < len(names) else str(g): v for g, v in groups.items() if v['land_hexes'] >= MIN_REGION_HEXES}
    dist, alarm = {}, {}
    for k in gp.ALARM_METRICS + ('contrast_de_mean', 'hard_tile_share', 'straight_tile_share', 'dark_island_max_hexes'):
        v = np.array([r[k] for r in regions.values() if r[k] is not None], np.float64)
        dist[k] = {str(p): round(float(np.percentile(v, p)), 4) for p in PCTS} if v.size else None
        if k in gp.ALARM_METRICS and v.size:
            alarm[k] = round(float(np.percentile(v, gp.ALARM_PCT[k])), 4)
    rep = {'kind': 'ground-patchiness-vanilla', 'source': 'vanilla dlc07 global_blend + lf_height_map (game packs), land = LF >= PLANE, regions = vanilla map.hex',
           'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'thresholds': {**thresholds(T), 'alarm': None},
           'min_region_hexes': MIN_REGION_HEXES, 'edge_of_world_hex': EDGE_OF_WORLD_HEX, 'regions_used': len(regions), 'alarm_percentile': dict(gp.ALARM_PCT), 'alarm': alarm,
           'distribution': dist, 'global': gp.summarise(acc, isl, alarm={}), 'regions': regions,
           'runtime_s': round(time.time() - t0, 1), 'peak_mb': terrain_regression.peak_mb()}
    dump(gp.VANILLA_JSON, rep)
    out = (args.out or OUT / 'vanilla').resolve(); out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gp.heat_rgb(gp.heat_layers(acc), 1)).save(out / 'map.png')
    print(json.dumps({'alarm': alarm, 'distribution': dist}, indent=1))
    print('->', gp.VANILLA_JSON, '|', out / 'map.png', f"({rep['runtime_s']} s, peak {rep['peak_mb']} MB)")


def baseline(args):
    """Per-window numbers of the baseline pack (v18 by default) for terrain_regression."""
    args.windows_only, args.label = True, args.label or 'v18-baseline'
    rep = run(args)
    dump(BASELINE, {'kind': 'ground-patchiness-baseline', 'source': rep['source'], 'created': rep['created'],
                    'note': 'v18 = output/tw3k_regions_and_battles_terrain_repair_candidate.previous-1793078765845440200.pack '
                            '(sha256 b14d6f7b...), per window box with context margins, as terrain_regression measures them',
                    'thresholds': rep['thresholds'], 'windows': {k: v['metrics'] for k, v in rep['windows'].items()}})
    print('->', BASELINE)


def validate(args):
    """Six live close-ups: shot | blend tones | heatmap of the camera box, with the numbers."""
    src, name = source(args)
    T = gp.Tones(args.season, args.tones)
    out = (args.out or OUT / 'validation').resolve(); out.mkdir(parents=True, exist_ok=True)
    win = {n: [part_box(p) for p in parts] for n, parts, _, _ in WINDOWS}
    rows, panels = [], []
    for place, wname, shot, (x, z), expect in LIVE:
        (q0, q1, r0, r1), = win[wname]
        nums, maps, _, _ = gp.analyse_box(src.band, q0, q1, r0, r1, T)
        lit = bool(nums['lit'])
        rows.append({'place': place, 'window': wname, 'camera': [x, z], 'camera_hex': [int(v) for v in cw.world_hex(x, z)],
                     'expected_lit': expect, 'lit': lit, 'ok': lit == expect, 'metrics': nums})
        R, land = src.band(r0 * 8, (r1 + 1) * 8, q0 * 8, (q1 + 1) * 8)
        col = T.srgb[R]; col[~land] = (28, 38, 52)
        th = 300
        heat = Image.fromarray(gp.heat_rgb(maps, 8)); heat = heat.resize((round(heat.width * th / heat.height), th), Image.NEAREST)
        blend = Image.fromarray(col[::-1]).resize(heat.size, Image.NEAREST)
        live = Image.open(LIVE_DIR / shot).convert('RGB'); live = live.crop((0, 45, live.width, live.height))
        live = live.resize((round(live.width * th / live.height), th))
        row = Image.new('RGB', (live.width + blend.width + heat.width + 16, th + 34), (255, 255, 255))
        row.paste(live, (0, 34)); row.paste(blend, (live.width + 8, 34)); row.paste(heat, (live.width + blend.width + 16, 34))
        d = ImageDraw.Draw(row)
        d.text((4, 2), f"{place} ({'fault' if expect else 'control'}) -> {'LIT ' + ','.join(nums['lit']) if lit else 'not lit'}   "
                       f"seg/100 {nums['straight_segment_per100']}  island {nums['dark_island_share']}  square/100 {nums['square_per100']}  "
                       f"hard/100 {nums['hard_edge_per100']}", fill=(200, 0, 0) if lit else (0, 0, 0))
        d.text((4, 18), f'live close-up (camera {x}, {z}; looks north)   |   blend tones, hex box q {q0}-{q1}, r {r0}-{r1} (north up)   |   heatmap',
               fill=(60, 60, 60))
        panels.append(row)
        print(f"{place:11s} expected {'lit' if expect else 'not lit':8s} got {'lit ' + ','.join(nums['lit']) if lit else 'not lit'}  "
              + json.dumps({k: nums[k] for k in gp.ALARM_METRICS}), flush=True)
    sheet = Image.new('RGB', (max(p.width for p in panels), sum(p.height + 8 for p in panels)), (255, 255, 255)); y = 0
    for p in panels:
        sheet.paste(p, (0, y)); y += p.height + 8
    sheet.save(out / 'validation.png')
    dump(out / 'validation.json', {'source': name, 'thresholds': thresholds(T), 'rows': rows})
    print('->', out / 'validation.png', '| separated' if all(r['ok'] for r in rows) else '| NOT separated')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for n, fn, default in (('run', run, CANDIDATE), ('baseline', baseline, V18), ('validate', validate, CANDIDATE), ('vanilla', vanilla, None)):
        s = sub.add_parser(n); s.set_defaults(fn=fn)
        if n != 'vanilla':
            s.add_argument('pack', nargs='?', type=Path, default=default)
            s.add_argument('--blend', type=Path, help='a loose global_blend.dds instead of the pack entry (scratch copy, never a live stage)')
            s.add_argument('--lf', type=Path, help='the matching lf_height_map.dds (land mask); without it every pixel counts as land')
            s.add_argument('--label')
            s.add_argument('--windows-only', action='store_true', help='windows only (row-band reads, no whole-map pass)')
            s.add_argument('--workers', type=int, default=1, help='whole-map strips on this many processes (about 5 GB each)')
            s.add_argument('--cache', type=Path, help='folder of strip results kept by their input bytes (same numbers, bit for bit)')
        s.add_argument('--season', default='harvest', help='slot tones: harvest (the table, 190 start) or another season (same extractor)')
        s.add_argument('--tones', type=Path, help='material-tones.json to read while CONSTS MATERIAL_TONE is empty (read-only)')
        s.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.cmd in ('run', 'validate') and not args.label:
        args.label = 'candidate' if args.pack == CANDIDATE and not args.blend else Path(args.blend or args.pack).stem
    args.fn(args)


if __name__ == '__main__':
    main()

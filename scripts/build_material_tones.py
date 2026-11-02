"""Measure the ground material tone table and vanilla's own neighbour contrast (ground contrast fix design §1).

  python build_material_tones.py [--season harvest] [--out analysis/material-tones.json]

1. MATERIAL_TONE: Lab(D65) of each texture's mean gamma-encoded sRGB, a proxy for the slot's tone, not the mean
   of its pixels' Lab values or the game's lit colour. Vanilla texture_arrays.xml supplies the 32 layer names;
   mip 1 is decoded directly from the packs without using the layer cache. The engine's climate/season lookup
   (g_global_blend_indices[R + G*k + season_offset]) has not been captured. Like render_campaign_view, this
   measurement ASSUMES layer = R and the seasonal_paths fallback, rather than claiming those are verified
   engine mappings. The JSON records those assumptions and the actual texture paths for every season.
2. Vanilla contrast: on vanilla's native land (the vanilla 3k_dlc07 global_blend), the CIE76 distance between
   the tones (start season) of horizontally and vertically adjacent pixels' slots, overall and per climate band (G
   of the first pixel): share of differing pairs, percentiles of the differing pairs, share above max_dE. User
   decisions 2026-10-27: the canvas no harsher than vanilla, by vanilla's own distribution per region (band); the
   threshold is the percentile CONTRAST_PCT (99) of the differing pairs, per band ('by_band') and overall.
   Tones are rounded to two decimals then converted to float32 before measuring, exactly as the consuming
   ground_tone helper reads the JSON table. Thresholds keep the actual percentile, not a display-rounded value.
Writes the JSON (data for biome_profiles CONSTS ground_materials MATERIAL_TONE / CONTRAST, pasted after review)
and prints both tables. Read-only on the vanilla packs; nothing else is written.
"""
import argparse, hashlib, json, struct, sys, time
from pathlib import Path

import numpy as np

import campaign_world as cw
from campaign_ground_textures import (GAME, PACKS, SEASONS, SEASON_FALLBACK, VARIANT, VanillaPacks,
                                      decode_dds, seasonal_paths, texture_array)
from ground_tone import slot_distance

CONTRAST_PCT = 99
PCTS = (50, 75, 90, 95, 99, 99.9)
MIP = 1
BLEND_KEY = f'terrain/campaigns/{VARIANT}/global_map/global_blend.dds'
ARRAY_KEY = f'terrain/campaigns/{VARIANT}/global_map/texture_arrays.xml'


def fresh_packs():
    """Rebuild the small pack index in memory; no persistent index/layer cache is read or written."""
    from analyze_reference import inventory
    packs = VanillaPacks.__new__(VanillaPacks)
    packs.index = {}
    for name in PACKS:
        path = GAME / name
        for entry in inventory(path, include_hash=False)['entries']:
            packs.index.setdefault(entry['path'], (str(path), entry['offset'], entry['size'], entry['compressed']))
    return packs


def read_asset(packs, key, assets):
    """Hash only the measured asset bytes, and identify the pack/entry actually read."""
    data = packs.read(key)
    if assets is not None:
        path, offset, size, compressed = packs.index[key]
        stat = Path(path).stat()
        assets[key] = {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data),
                       'pack': str(Path(path).resolve()), 'pack_size_bytes': stat.st_size,
                       'pack_mtime_ns': stat.st_mtime_ns, 'entry_offset': offset,
                       'entry_size_bytes': size, 'compressed': bool(compressed)}
    return data


def srgb_to_lab(rgb):
    """CIELAB (D65) of sRGB 0-255 colours, shape (..., 3)."""
    c = np.asarray(rgb, np.float64) / 255.
    c = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    M = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
    xyz = c @ M.T / np.array([.95047, 1., 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def tones(season, packs, assets=None, measured=None):
    names, paths = texture_array(packs)
    if len(names) != 32:
        raise ValueError(f'Expected 32 blend slots, found {len(names)}')
    paths = seasonal_paths(paths, season, packs)
    measured = {} if measured is None else measured
    for path in paths:
        if path not in measured:
            arr = decode_dds(read_asset(packs, path, assets), MIP)
            measured[path] = srgb_to_lab(arr.reshape(-1, 3).mean(0))
    # JSON holds two-decimal numbers; ground_tone.tone_table reads them as float32.
    lab = np.asarray(np.round([measured[path] for path in paths], 2), np.float32)
    return names, paths, lab


def _percentile(values, counts, percentile):
    """np.percentile's linear order-statistic interpolation, without expanding repeated slot pairs."""
    order = np.argsort(values)
    values, cumulative = np.asarray(values[order], np.float64), np.cumsum(counts[order], dtype=np.int64)
    position = (int(cumulative[-1]) - 1) * percentile / 100.
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    a, b = values[np.searchsorted(cumulative, (lo, hi), side='right')]
    return float(a + (b - a) * (position - lo))


def _stats(counts, distances):
    total = int(counts.sum())
    mask = (distances > 0) & (counts > 0)
    diff = int(counts[mask].sum())
    out = {'pairs': total, 'differing_share': round(diff / max(total, 1), 5),
           'percentiles_of_differing': {str(p): _percentile(distances[mask], counts[mask], p) for p in PCTS}
           if diff else {}}
    out['max_dE'] = out['percentiles_of_differing'].get(str(CONTRAST_PCT))
    out['share_above_max'] = round(float(counts[distances > out['max_dE']].sum()) / max(total, 1), 6) \
        if out['max_dE'] is not None else None
    return out


def _pair_histogram(rg, cells, row_block=128):
    """Count (first pixel G, slot A, slot B) on land, with at most row_block+1 rows of land/index grids."""
    h, w = rg.shape[:2]
    H, W = cells.shape[:2]
    q = np.clip(np.arange(w) // 8, 0, W - 1)
    counts = np.zeros((4, 32, 32), np.int64)
    for start in range(0, h, row_block):
        end = min(h, start + row_block)
        ys = np.arange(start, min(h, end + 1))
        rr = np.clip((((h - 1 - ys) // 4)[:, None] - q[None, :] % 2) // 2, 0, H - 1)
        land = (cells[rr, q[None, :], 0] & 3) != 1
        z = rg[start:end]
        if (z[..., 0] >= 32).any() or (z[..., 1] >= 4).any():
            raise ValueError('Vanilla global_blend has an unsupported material slot or climate band')
        m = land[:end - start, :-1] & land[:end - start, 1:]
        code = z[:, :-1, 1].astype(np.int32) * 1024 + z[:, :-1, 0].astype(np.int32) * 32 + z[:, 1:, 0]
        counts += np.bincount(code[m], minlength=4096).reshape(4, 32, 32)
        n = min(end, h - 1) - start
        m = land[:n] & land[1:n + 1]
        a, b = rg[start:start + n], rg[start + 1:start + n + 1]
        code = a[..., 1].astype(np.int32) * 1024 + a[..., 0].astype(np.int32) * 32 + b[..., 0]
        counts += np.bincount(code[m], minlength=4096).reshape(4, 32, 32)
    return counts


def vanilla_contrast(lab, packs, assets=None):
    d = read_asset(packs, BLEND_KEY, assets)
    h, w = struct.unpack_from('<2I', d, 12)
    if d[:4] != b'DDS ' or d[84:88] != bytes(4) or struct.unpack_from('<I', d, 88)[0] != 16:
        raise ValueError('Expected an uncompressed 16-bit RG global_blend DDS')
    rg = np.frombuffer(d, np.uint8, offset=128, count=h * w * 2).reshape(h, w, 2)
    import map_hex
    cells = map_hex.read(cw.VANILLA_MAP)['cells']
    H, W = cells.shape[:2]
    if (h, w) != ((2 * H + 1) * 4, W * 8):
        raise ValueError('Vanilla blend/map dimensions disagree with the 4-pixel-per-tile frame')
    counts = _pair_histogram(rg, cells)
    distances = slot_distance(lab)  # same float32 implementation as ground bake/invariants
    total = counts.sum(0)
    out = _stats(total, distances)
    out['share_above'] = {str(t): round(float(total[distances > t].sum()) / max(int(total.sum()), 1), 6)
                          for t in (5, 10, 15, 20, 25, 30, 40)}
    out['by_band'] = {str(k): _stats(counts[k], distances) for k in range(4) if counts[k].any()}
    return out


def vanilla_tone_cut(lab_by, packs, assets=None):
    """Native-edge tone CUT thresholds (ground contrast fix design §10) from vanilla alone: the vanilla blend placed on
    the canvas grid at the native rectangle, every native-border segment (audit_native_edges.SEGMENT) whose native side
    (the 24 tile cells inside the border) is >= 60 % vanilla land, the tone jump (ground_tone.line_tone_jump, mean over
    the segment) of each interior reference line (audit_native_edges.REF_LINES), per season. min = p99 of those line
    values, floor = min / ratio (design 10.1: p50 is 0), ratio = the index CUT's cut_ratio."""
    import map_hex
    from audit_native_edges import LIMITS, REF_LINES, SEGMENT
    from terrain_layers import _dds, across, tangent_range
    from ground_tone import line_tone_jump
    van = _dds(read_asset(packs, BLEND_KEY, assets), 2)[..., 0]   # south-up slots
    R = np.full((cw.TILE_H * 4, cw.TILE_W * 4), 255, np.uint8)
    y0, x0 = cw.NATIVE_TILE_Y0 * 4, cw.NATIVE_TILE_X0 * 4
    R[y0:y0 + van.shape[0], x0:x0 + van.shape[1]] = van[:R.shape[0] - y0, :R.shape[1] - x0]
    land_hex = np.zeros((cw.H, cw.W), bool)
    land_hex[cw.NATIVE_HEX_SLICE] = (map_hex.read(cw.VANILLA_MAP)['cells'][:, :, 0] & 3) != 1
    ty, tx = np.mgrid[:cw.TILE_H, :cw.TILE_W]
    q, r = cw.tile_hex(tx, ty)
    land = land_hex[r, q]
    del ty, tx, q, r
    luts = {}
    for season, lab in lab_by.items():
        luts[season] = np.zeros((256, 3), np.float32)
        luts[season][:32] = lab
    lines = {s: [] for s in lab_by}
    segments = 0
    for edge in ('north', 'south', 'west', 'east'):
        lo, hi = tangent_range(edge)
        for t0 in range(lo, hi - SEGMENT + 1, SEGMENT):
            if across(land, edge, t0, t0 + SEGMENT, -24, 0).mean() < .6:
                continue
            segments += 1
            raw = across(R, edge, t0, t0 + SEGMENT, -100, 60, 4)
            if (raw[:100 * 4] >= 32).any():
                raise ValueError(f'{edge} {t0}: vanilla slot outside 0-31 inside the native rectangle')
            for season, lut in luts.items():
                lab = lut[raw]
                lines[season] += [float(line_tone_jump(lab, (100 + k) * 4).mean()) for k in REF_LINES]
    out = {'ratio': LIMITS['cut_ratio'], 'percentile_min': 99, 'floor_rule': 'min / ratio (design 10.1)', 'by_season': {},
           'segments': segments, 'reference_lines': list(REF_LINES)}
    for season, v in lines.items():
        v = np.asarray(v)
        mn = float(np.percentile(v, 99))
        out['by_season'][season] = {'min': mn, 'floor': mn / out['ratio'],
                                    'percentiles': {str(p): float(np.percentile(v, p)) for p in PCTS}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', default='harvest', choices=SEASONS, help='the start season: the contrast tones')
    ap.add_argument('--out', type=Path, default=cw._ROOT / 'analysis/material-tones.json')
    a = ap.parse_args()
    packs = fresh_packs()
    assets, measured, resolved = {}, {}, {}
    read_asset(packs, ARRAY_KEY, assets)
    lab_by = {}
    for season in SEASONS:
        names, resolved[season], lab_by[season] = tones(season, packs, assets, measured)
    lab = lab_by[a.season]
    con = vanilla_contrast(lab, packs, assets)
    tone_cut = vanilla_tone_cut(lab_by, packs, assets)
    tone_cut['provenance'] = {'blend_sha256': assets[BLEND_KEY]['sha256'], 'pack': assets[BLEND_KEY]['pack'],
                              'date': time.strftime('%Y-%m-%d'), 'segments': tone_cut['segments'],
                              'selection': 'native-border segments with >= 60 % vanilla land in the 24 tile cells inside',
                              'measurement_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    data = {'start_season': a.season, 'contrast_percentile': CONTRAST_PCT, 'TONE_CUT': tone_cut,
            'MATERIAL_TONE': {se: [[round(float(x), 2) for x in l] for l in lab_by[se]] for se in SEASONS},
            'slots': {i: names[i] for i in range(len(names))}, 'CONTRAST': con,
            'provenance': {'method': 'Lab(D65) of mean gamma-encoded sRGB; not mean per-pixel Lab or lit game colour',
                           'mip': MIP, 'tone_quantization': 'round to 2 decimal places, then float32',
                           'distance': 'ground_tone.slot_distance: Euclidean CIELAB (CIE76), float32',
                           'percentile': 'linear interpolation over differing land-neighbour pairs; unrounded threshold',
                           'engine_mapping_assumption': 'texture array layer = R; climate/season cbuffer not captured',
                           'season_fallback_assumption': SEASON_FALLBACK, 'resolved_texture_paths': resolved,
                           'asset_sha256': assets,
                           'vanilla_map': {'path': str(cw.VANILLA_MAP.resolve()),
                                           'sha256': hashlib.sha256(cw.VANILLA_MAP.read_bytes()).hexdigest()},
                           'pack_index': 'fresh in-memory inventory, first pack wins; no index or layer cache',
                           'measurement_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    a.out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    for i in range(len(lab)):
        print(f'{i:2d} {names[i]:28s} L {lab[i, 0]:6.1f} a {lab[i, 1]:6.1f} b {lab[i, 2]:6.1f}')
    print(json.dumps(con, indent=1))
    print(json.dumps({s: {k: v[k] for k in ('min', 'floor')} for s, v in tone_cut['by_season'].items()}), tone_cut['segments'], 'segments')
    print('wrote', a.out)


if __name__ == '__main__':
    sys.exit(main())

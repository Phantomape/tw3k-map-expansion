"""Load the rendered campaign terrain layers from a stage directory or a pack.

All arrays are south-up (row 0 = south) in the canvas frame:
  heights   LF world heights, 4 pixels per tile cell
  blend     global_map/global_blend.dds channels (uint8, same pixel grid as heights)
  land      final map land mask per tile cell
  climate   fine-tile climate per tile cell (0 sub_tropical, 1 cold, 2 temperate, 3 arid, 255 none)
"""
import tw3k_paths
import re, struct
from pathlib import Path

import numpy as np

import campaign_world as cw
from audit_ground_seams import Packs, StageAssets, B
from native_tile_dimensions import decode_dimensions, placement_footprint
from read_tile_list import parse as parse_tiles

GAME = tw3k_paths.GAME_DATA


def open_source(stage=None, pack=None):
    van = Packs([GAME / 'terrain.pack', GAME / 'fast.pack', GAME / 'data.pack'])
    if pack:
        return Packs([Path(pack)]), van
    return StageAssets(Path(stage) if stage else cw.WORKSPACE / 'work/release-input', van), van


def _dds(data, channels):
    h, w = struct.unpack_from('<2I', data, 12)
    dtype = '<u2' if channels == 0 else np.uint8
    a = np.frombuffer(data, dtype, offset=128)
    return (a.reshape(h, w) if channels == 0 else a.reshape(h, w, channels))[::-1]


def load_layers(source, vanilla, variant='3k_dlc07_main_map', want=('heights', 'blend', 'land', 'climate')):
    pre = f'terrain/campaigns/{variant}/'
    out = {}
    if 'heights' in want:
        raw = _dds(source.read(pre + 'lf_height_map.dds'), 0)
        out['heights'] = cw.LEVEL + (raw.astype(np.float32) / 65535 - cw.NORM) * cw.GAIN
    if 'blend' in want:
        out['blend'] = _dds(source.read(pre + 'global_map/global_blend.dds'), 2)
    if 'land' in want:
        cells = np.frombuffer((cw.WORKSPACE / 'work/final-map/map.hex').read_bytes()[-4 - cw.W * cw.H * 16:-4],
                              np.uint8).reshape(cw.H, cw.W, 16)
        land_hex = (cells[:, :, 0] & 3) != 1
        ty, tx = np.mgrid[:cw.TILE_H, :cw.TILE_W]
        # Owner rule, not rint: round-half-even gave even columns three tile columns.
        q, r = cw.tile_hex(tx, ty)
        out['land'] = land_hex[r, q]
    if 'climate' in want:
        paths, _, _, _, recs, _ = parse_tiles(source.read(pre + 'tile_list.bin'))
        dims = {}
        clim = np.full((cw.TILE_H, cw.TILE_W), 255, np.uint8)
        for rec in recs:
            p = paths[rec[1]]
            if 'generic_sea' in p:
                continue
            if rec[1] not in dims:
                m = re.search(r'generic_(\d+)x(\d+)', p)
                if m:
                    dims[rec[1]] = tuple(map(int, m.groups()))
                else:
                    bits = p.strip(B).split(B)
                    key = 'terrain/tiles/campaign/_tile_database/tiles/' + bits[-2] + '_' + bits[-1] + '.bin'
                    dims[rec[1]] = decode_dimensions(vanilla.read(key)) if key in vanilla.index else (1, 1)
            w, h = placement_footprint(dims[rec[1]], rec[5])
            clim[max(0, rec[4]):max(0, rec[4] + h), max(0, rec[3]):max(0, rec[3] + w)] = rec[2]
        out['climate'] = clim
    return out


# Each native border as (name, outward unit normal in (dx, dy) tile cells, line position).
def borders():
    X0, X1, Y0, Y1 = cw.NATIVE_TILE_X0, cw.NATIVE_TILE_X1, cw.NATIVE_TILE_Y0, cw.NATIVE_TILE_Y1
    return {'north': (Y1, (0, 1)), 'south': (Y0 - 1, (0, -1)), 'west': (X0 - 1, (-1, 0)), 'east': (X1, (1, 0))}


def across(array, edge, t0, t1, n0, n1, scale=1):
    """Samples [n, t] across a native border; n>=0 is outside (n=0 first outside cell).

    scale is pixels per tile cell for the array (1 for tile grids, 4 for LF/blend).
    """
    line, (dx, dy) = borders()[edge]
    n = np.arange(n0 * scale, n1 * scale)[:, None]
    t = np.arange(t0 * scale, t1 * scale)[None, :]
    if dy:
        start = line * scale if dy > 0 else line * scale + (scale - 1)
        rows = start + dy * n
        rows = np.clip(rows, 0, array.shape[0] - 1)
        return array[rows, t]
    start = line * scale if dx > 0 else line * scale + (scale - 1)
    cols = start + dx * n
    cols = np.clip(cols, 0, array.shape[1] - 1)
    return array[t, cols]


def tangent_range(edge):
    X0, X1, Y0, Y1 = cw.NATIVE_TILE_X0, cw.NATIVE_TILE_X1, cw.NATIVE_TILE_Y0, cw.NATIVE_TILE_Y1
    return (X0, X1) if edge in ('north', 'south') else (Y0, Y1)


def edge_point_hex(edge, t):
    """Canvas hex [q, r] of the border cell at tangent position t (tile cells)."""
    line, (dx, dy) = borders()[edge]
    x, y = (t, line) if dy else (line, t)
    q, r = cw.tile_hex(x, y)
    return [int(q), int(r)]

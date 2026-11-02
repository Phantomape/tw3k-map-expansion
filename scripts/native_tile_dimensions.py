"""Decode native FASTBIN tile dimensions after all three string fields."""
import struct

def decode_dimensions(data):
 if data[:8] != b'FASTBIN0':raise ValueError('Not a native tile database record')
 pos=10;strings=[]
 for _ in range(3):
  n=struct.unpack_from('<H',data,pos)[0];pos+=2
  if pos+n>len(data):raise ValueError('Truncated native tile string')
  strings.append(data[pos:pos+n]);pos+=n
 _,w,h=struct.unpack_from('<H2I',data,pos)
 if not (0<w<=256 and 0<h<=256) or len(strings[2]) not in (0,w*h):
  raise ValueError(f'Invalid native tile dimensions/mask: {w}x{h}, mask={len(strings[2])}')
 return w,h

# tile_list flags bits 4-7 hold the placement rotation one-hot (0x10/0x20/0x40/0x80).
# Swapping width and height for 0x20 and 0x80 (quarter turns) makes vanilla
# 3k_main_map land placements cover every cell; the nominal footprint leaves 41255
# cells uncovered. Record byte 7 is climate, not rotation.
QUARTER_TURN_FLAGS = 0x20 | 0x80


def placement_footprint(size, flags):
 w, h = size
 return (h, w) if flags & QUARTER_TURN_FLAGS else (w, h)


# --- canvas land floors: one definition for the cache (polish_campaign_terrain.py) and
# the tile lists (polish_campaign_terrain.py, repair_native_placements.py) ---
# tile_map.tiles path values are ONE-BASED into the cache's path table: value v names
# paths[v - 1]. Vanilla 3k_main_map: every cache owner of value index(generic_8x8)
# (1,291) covers 16x16 cells and sits on the anchor of one of the 1,291 generic_16x16
# list records; value index(generic_1x1) owns 2x2 cells under the 63,304 generic_2x2
# records; index(generic_1x1)+1 owns 1 cell under the 26,531 generic_1x1 records. All
# five generic sizes match list name == paths[v - 1] at the same anchor, 100%, and the
# cache footprint equals the name's size (no "twice the nominal size").
GENERIC_TILE = 'terrain\\tiles\\campaign\\generic\\generic_{0}x{0}\\'
FLOOR_TILE = {16: GENERIC_TILE.format(16), 2: GENERIC_TILE.format(2), 1: GENERIC_TILE.format(1)}
# Polish's floor templates are found by these cache values (index of the name, i.e. the
# tile one entry before it): FLOOR_TEMPLATE_KEY[size] names the tile FLOOR_TILE[size].
FLOOR_TEMPLATE_KEY = {16: GENERIC_TILE.format(8), 2: GENERIC_TILE.format(1)}


def cache_tile_path(paths, value):
    """The tile a tile_map.tiles owner value names (values are one-based)."""
    assert 1 <= value <= len(paths), value
    return paths[value - 1]


def floor_list_tiles(paths):
    """{size: list tile name} of the canvas floors, checked against the cache path table:
    the template key's value names the floor tile."""
    out = {}
    for size, key in FLOOR_TEMPLATE_KEY.items():
        name = cache_tile_path(paths, paths.index(key))
        assert name == FLOOR_TILE[size], (size, name)
        out[size] = name
    return out


def floor_relay(edit, biome):
    """List placements [(size, x, y, climate)] that floor every cell of edit (bool, tile grid)
    the way polish lays the cache floors: a whole 16x16 block (16-aligned) of one climate takes
    a generic_16x16 (climate at the block centre), every other whole 2x2 group (2-aligned)
    a generic_2x2 (climate at its centre), each remaining cell a generic_1x1. Row-major blocks."""
    import numpy as np
    H, W = edit.shape
    out = []
    for y in range(0, H, 16):
        for x in range(0, W, 16):
            bh, bw = min(16, H - y), min(16, W - x)
            part = edit[y:y + bh, x:x + bw]
            if not part.any():
                continue
            if bh == 16 and bw == 16 and part.all() and np.all(biome[y:y + 16, x:x + 16] == biome[y, x]):
                out.append((16, x, y, int(biome[y + 8, x + 8])))
                continue
            for dy in range(0, bh, 2):
                for dx in range(0, bw, 2):
                    sub = part[dy:dy + 2, dx:dx + 2]
                    if not sub.any():
                        continue
                    if sub.shape == (2, 2) and sub.all():
                        out.append((2, x + dx, y + dy, int(biome[y + dy + 1, x + dx + 1])))
                    else:
                        for sy, sx in zip(*np.nonzero(sub)):
                            yy, xx = y + dy + int(sy), x + dx + int(sx)
                            out.append((1, xx, yy, int(biome[yy, xx])))
    return out

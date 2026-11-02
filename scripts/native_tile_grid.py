"""The two vanilla fine-tile bases on the canvas tile grid, in one place (pipeline-bake-design 3.3: the tile bake).

The game fixes which base each output starts from: the display cache (tile_map.tiles) from 3k_main_map
(campaign_world.CACHE_DONOR_VARIANT), the tile lists (tile_list.bin) from 3k_dlc07_main_map. Both sit at the
native rectangle's tile offset (NATIVE_TILE_X0, NATIVE_TILE_Y0) on the canvas; these functions are the only
place that offset is applied to vanilla records (polish's cache segment, tile_bake, and the cache audits)."""
import numpy as np
import campaign_world as _frame

CACHE_DT = np.dtype([('path', '<i4'), ('flags', '<u2'), ('x', '<u2'), ('y', '<u2'), ('climate', 'u1'), ('extra', 'u1'), ('lo', '<f4'), ('hi', '<f4')])


def cache_layer(native, i):
    """Layer i of a vanilla tile_map.tiles (bytes) on the native grid with canvas coordinates: child cells point
    at their owner's canvas cell index (-index-1), active owners carry canvas x/y. Shape NATIVE_TILE_H x _W."""
    ow, oh = _frame.NATIVE_TILE_W, _frame.NATIVE_TILE_H; on = ow * oh
    layer = np.frombuffer(native, dtype=CACHE_DT, offset=i * on * 20, count=on).reshape(oh, ow).copy()
    child = layer['path'] < 0; parents = -layer['path'][child] - 1
    layer['path'][child] = _frame.native_cache_owner(parents)
    active = layer['flags'] != 16; layer['x'][active] += _frame.NATIVE_TILE_X0; layer['y'][active] += _frame.NATIVE_TILE_Y0
    return layer


def list_xy(x, y):
    """A vanilla tile_list record's x, y on the canvas tile grid."""
    return x + _frame.NATIVE_TILE_X0, y + _frame.NATIVE_TILE_Y0

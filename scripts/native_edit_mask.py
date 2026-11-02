"""The one definition of "terrain edited here" for every step that may delete,
replace or re-seat vanilla content.

Edited = the whole canvas outside the native rectangle, plus native hexes where
the staged LF height differs from vanilla, grown by HALO hexes. Earlier steps
each used their own proxy (polish alpha > 0 / > 1e-7 / > .02, every hex of every
design region, dilated replacement masks); those covered 16-46% of the native
map although the LF differs on far less, and vanilla coarse geometry, trees,
special tiles and cache owners were deleted or regenerated across that area.

Every lookup uses tile-block ownership (cw.tile_hex: hex q owns tiles 2q, 2q+1
and rows 2r+odd(q), +1), so LF texels, tiles and coarse vertices agree:

  edited_hexes(variant)            bool (H, W) on the expanded hex grid
  edited_tiles(variant)            bool (TILE_H, TILE_W) tile grid, hexes[tile_hex(tx, ty)]
  at_world(variant, x, z)          edited_tiles at the tile holding world points
"""
import struct
from functools import lru_cache

import numpy as np

import campaign_world as cw

HALO = 2          # hexes around a changed LF hex that still count as edited
TOLERANCE = 2     # LF u16 units; projection rounding is below this


def _lf(blob):
    h, w = struct.unpack_from('<2I', blob, 12)
    return np.frombuffer(blob, '<u2', offset=128).reshape(h, w)[::-1].astype(np.int32)


def _grow(m, n):
    """n steps of 4-neighbour dilation in (r, q) index space. This is not a true
    odd-q hex ring (it misses one diagonal neighbour per side; a hex ring would
    add ~1.6k native hexes at HALO 2); kept so the edit area stays as it was."""
    for _ in range(n):
        g = m.copy()
        g[1:] |= m[:-1]
        g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    return m


@lru_cache(maxsize=4)
def edited_hexes(variant, stage=None, halo=HALO):
    from repair_campaign_river_roots import Assets
    stage = stage or (cw.WORKSPACE / 'work/release-input')
    key = f'terrain/campaigns/{variant}/lf_height_map.dds'
    cur = _lf((stage / key).read_bytes())
    van = _lf(Assets(original_only=True).read(key))
    vh, vw = van.shape
    ox, oy = cw.NATIVE_OFFSET_LF
    diff = np.abs(cur[oy:oy + vh, ox:ox + vw] - van) > TOLERANCE
    # Changed texels -> owning hex in canvas coordinates (LF texel 4t is tile t).
    # Texels above the last native row of even columns belong to canvas hexes
    # outside the native rectangle, which are edited anyway.
    r, c = np.nonzero(diff)
    q, rr = cw.tile_hex((c + ox) // 4, (r + oy) // 4)
    changed = np.zeros((cw.H, cw.W), bool)
    changed[rr, q] = True
    out = np.ones((cw.H, cw.W), bool)
    out[cw.NATIVE_HEX_SLICE] = _grow(changed[cw.NATIVE_HEX_SLICE], halo)
    return out


def edited_tiles(variant, stage=None, halo=HALO):
    hexes = edited_hexes(variant, stage, halo)
    q, r = cw.tile_hex(np.arange(cw.TILE_W, dtype=np.int32)[None, :], np.arange(cw.TILE_H, dtype=np.int32)[:, None])
    return hexes[r, q]


def at_world(variant, x, z, stage=None):
    """Edited flag for world-space points (x, z arrays): edited_tiles at the tile
    (floor of the snapped tile coordinate) holding each point, via cw.world_hex."""
    hexes = edited_hexes(variant, stage)
    q, r = cw.world_hex(x, z)
    return hexes[r, q]


# --- HEHUANG PATCH (hehuang_terrain.py) ---
HEHUANG_GROW = 2   # hexes around the patch weight > 0 that audits exempt


def hehuang_mask(grow=HEHUANG_GROW):
    """bool (H, W) hex grid: the real-geography patches (hehuang_terrain.PATCHES, each
    analysis/<name>-patch.npz hex_mask, height weight > 0, or-ed) grown by `grow` hexes,
    or None without a patch. Audits count differences inside it as intended
    ('exempt_hehuang'); outside it nothing changes."""
    import hehuang_terrain
    p = hehuang_terrain.load()
    return None if p is None else p.hex_mask(grow)


def hehuang_tiles(grow=HEHUANG_GROW):
    """hehuang_mask on the tile grid (tile_hex ownership), or None."""
    m = hehuang_mask(grow)
    if m is None:
        return None
    q, r = cw.tile_hex(np.arange(cw.TILE_W, dtype=np.int32)[None, :], np.arange(cw.TILE_H, dtype=np.int32)[:, None])
    return m[r, q]

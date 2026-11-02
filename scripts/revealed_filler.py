"""Revealed filler: vanilla non-playable land that is now playable mod land.

Vanilla hid large tracts of the native map under shroud as region 3k_main_reg_non_playable and
painted them with filler (cold_1 sheets, round brush stamps, flat filler relief). The expansion
made ~156k of those land hexes playable (Indochina, Songliao, the Gobi rim, Wuyuan, the q=888
west edge). Steps that keep the native rectangle as vanilla treat these hexes as canvas instead
(user approval 2026-10-25); vanilla's originally playable land stays byte-identical.

One mask for every step (polish, apply_ground_types, apply_climate, apply_forest_ground,
apply_rivers, coast_tiles, apply_shore_kinds, farm_fields, the native identity/edge audits,
terrain_regression):

  revealed_hex(cells, names)          bool (H, W) on the canvas hex grid
  river_canvas_hex(cells, names)      bool (H, W): outside the native rectangle, revealed filler, and vanilla
                                      non-playable hexes that are water now (polish RIVER CHANNELS)
  revealed_weight(cells, names, ramp) float (H, W): 0 on and next to vanilla playable land,
                                      1 from about `ramp` hexes away (polish ground weight)
  vanilla_playable_hex()              bool (H, W): vanilla's own playable land (never revealed)
  filler_hex(cells, names)            bool (H, W): all vanilla non-playable filler that is land now, playable
                                      or still non-playable (user 2026-10-25: repaint all of it; ground
                                      materials only; gameplay steps keep revealed_hex)
  filler_weight(cells, names, ramp)   float: filler_hex ramped like revealed_weight (polish ground)

cells: the canvas map.hex cells (H, W, 16); names: land_regions + sea_regions of that map.
"""
import numpy as np

import campaign_world as cw

NON_PLAYABLE = ('3k_main_reg_non_playable', '3k_main_sea_non_playable')


_VANILLA = None


def _vanilla():
    global _VANILLA
    if _VANILLA is not None:
        return _VANILLA
    from fast_hex import read_hex
    meta = read_hex(cw.VANILLA_MAP)
    b = cw.VANILLA_MAP.read_bytes()
    vc = np.frombuffer(b[-4 - cw.NATIVE_W * cw.NATIVE_H * 16:-4], np.uint8).reshape(cw.NATIVE_H, cw.NATIVE_W, 16)
    vnames = meta['land_regions'] + meta['sea_regions']
    vids = (vc[:, :, 1].astype(int) << 5) + (vc[:, :, 0].astype(int) >> 3) - 1
    _VANILLA = (vc, vnames, vids)   # read once per process
    return _VANILLA


def vanilla_playable_hex():
    """Vanilla land hexes outside the non-playable regions, on the canvas grid."""
    vc, vnames, vids = _vanilla()
    np_ids = [i for i, n in enumerate(vnames) if n in NON_PLAYABLE]
    out = np.zeros((cw.H, cw.W), bool)
    out[cw.NATIVE_HEX_SLICE] = ((vc[:, :, 0] & 3) != 1) & ~np.isin(vids, np_ids)
    return out


def revealed_hex(cells, names):
    """Vanilla non-playable hexes (land or sea region) that are now land of a playable region."""
    vc, vnames, vids = _vanilla()
    vnp = np.zeros((cw.H, cw.W), bool)
    vnp[cw.NATIVE_HEX_SLICE] = np.isin(vids, [i for i, n in enumerate(vnames) if n in NON_PLAYABLE])
    ids = (cells[:, :, 1].astype(int) << 5) + (cells[:, :, 0].astype(int) >> 3) - 1
    land = (cells[:, :, 0] & 3) != 1
    cur_np = np.isin(ids, [i for i, n in enumerate(names) if n in NON_PLAYABLE])
    return vnp & land & ~cur_np


def river_canvas_hex(cells, names):
    """Hexes the river steps treat as canvas: outside the native rectangle, the revealed filler and vanilla's
    non-playable hexes that are water now (the rivers apply_rivers laid on revealed land, vanilla's non-playable
    seas). The complement of apply_rivers' native mask, read back on the map it wrote (its rivers are water)."""
    vc, vnames, vids = _vanilla()
    vnp = np.zeros((cw.H, cw.W), bool)
    vnp[cw.NATIVE_HEX_SLICE] = np.isin(vids, [i for i, n in enumerate(vnames) if n in NON_PLAYABLE])
    rect = np.zeros((cw.H, cw.W), bool)
    rect[cw.NATIVE_HEX_SLICE] = True
    return ~rect | revealed_hex(cells, names) | (vnp & ((cells[:, :, 0] & 3) == 1))


def revealed_weight(cells, names, ramp):
    """revealed_hex times 1 - 3 x the box-blurred (radius `ramp` hexes) share of vanilla playable land."""
    from native_campaign_climate import box_blur
    play = vanilla_playable_hex().astype(np.float32)
    return (np.clip(1 - 3 * box_blur(play, ramp), 0, 1) * revealed_hex(cells, names)).astype(np.float32)


def filler_hex(cells, names):
    """Vanilla non-playable hexes (land or sea region) that are land now, whatever their region."""
    vc, vnames, vids = _vanilla()
    vnp = np.zeros((cw.H, cw.W), bool)
    vnp[cw.NATIVE_HEX_SLICE] = np.isin(vids, [i for i, n in enumerate(vnames) if n in NON_PLAYABLE])
    return vnp & ((cells[:, :, 0] & 3) != 1)


def filler_weight(cells, names, ramp):
    """filler_hex times 1 - 3 x the box-blurred (radius `ramp` hexes) share of vanilla playable land."""
    from native_campaign_climate import box_blur
    play = vanilla_playable_hex().astype(np.float32)
    return (np.clip(1 - 3 * box_blur(play, ramp), 0, 1) * filler_hex(cells, names)).astype(np.float32)

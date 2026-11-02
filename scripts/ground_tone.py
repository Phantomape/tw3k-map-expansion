"""Ground tone helpers (ground contrast fix design §1): the slot tone table as an array and the neighbour contrast
measure, one implementation for the ground bake (tone-continuous sampling), the stage invariant tone_contrast_edges
and the patchiness metric. Data: biome_profiles CONSTS ground_materials MATERIAL_TONE / CONTRAST. numpy only."""
import numpy as np

import biome_profiles as _bp


START_SEASON = 'harvest'


def tone_table(season=START_SEASON):
    """(32, 3) float32 CIELAB per slot for a season ({season: [[L, a, b] x 32]}, build_material_tones), or None
    while MATERIAL_TONE is not measured."""
    t = _bp.CONSTS['ground_materials'].get('MATERIAL_TONE')
    if not t:
        return None
    return np.asarray(t[season], np.float32)


def max_dE(band=None):
    """Vanilla's neighbour contrast threshold: CONTRAST max_dE (p99 of differing neighbour pairs on vanilla's native
    land), or that of one climate band ('by_band', the region's own distribution); None while not measured."""
    c = _bp.CONSTS['ground_materials'].get('CONTRAST')
    if not c:
        return None
    if band is not None and str(band) in c.get('by_band', {}) and c['by_band'][str(band)].get('max_dE') is not None:
        return float(c['by_band'][str(band)]['max_dE'])
    return float(c['max_dE'])


def slot_distance(tones):
    """(32, 32) CIE76 distance between slot tones."""
    return np.sqrt(((tones[:, None, :] - tones[None, :, :]) ** 2).sum(-1)).astype(np.float32)


def _edge_blocks(slots, mask, rows):
    """Horizontal/vertical pairs once each, bounded temporaries on a full canvas."""
    h = slots.shape[0]
    for y in range(0, h, rows):
        end = min(y + rows, h)
        yield (slots[y:end, :-1], slots[y:end, 1:],
               None if mask is None else mask[y:end, :-1] & mask[y:end, 1:], (y, end, False))
        end = min(y + rows, h - 1)
        if end > y:
            yield (slots[y:end, :], slots[y + 1:end + 1, :],
                   None if mask is None else mask[y:end, :] & mask[y + 1:end + 1, :], (y, end, True))


def contrast_edges(slots, tones, threshold, mask=None, rows=256):
    """Adjacent pixel pairs (horizontal + vertical) of a slot image whose tone distance exceeds threshold, and the
    number of pairs counted (both pixels in mask when given): (above, pairs)."""
    D = slot_distance(tones)
    above = pairs = 0
    for a, b, ma, _ in _edge_blocks(slots, mask, rows):
        d = D[a, b]
        if ma is not None:
            d = d[ma]
        above += int((d > threshold).sum()); pairs += int(d.size)
    return above, pairs


def contrast_by_band(slots, bands, tones, mask=None, rows=256):
    """Neighbour counts using each source G band's vanilla threshold.

    A horizontal pair belongs to its left pixel's band, a vertical pair to its
    first row's band, exactly as build_material_tones measures vanilla. The caller
    supplies both arrays in the same row orientation. Cross-band pairs are counted
    once; the first pixel selects the reference distribution.
    """
    if slots.shape != bands.shape:
        raise ValueError('Material slots and climate bands must have the same shape')
    D = slot_distance(tones)
    result = {}
    for a, b, ma, (y, end, vertical) in _edge_blocks(slots, mask, rows):
        band = bands[y:end, :] if vertical else bands[y:end, :-1]
        d = D[a, b]
        if ma is not None:
            band, d = band[ma], d[ma]
        for k in np.unique(band):
            threshold = max_dE(int(k))
            if threshold is None:
                raise ValueError('CONTRAST has not been measured')
            sel = band == k
            stats = result.setdefault(str(int(k)), {'pairs': 0, 'above': 0, 'max_dE': threshold})
            stats['pairs'] += int(sel.sum())
            stats['above'] += int((d[sel] > threshold).sum())
    return result


def seasons():
    """The seasons MATERIAL_TONE was measured for, in table order (build_material_tones: campaign_ground_textures.SEASONS)."""
    return tuple(_bp.CONSTS['ground_materials'].get('MATERIAL_TONE') or ())


def tone_lut(season=START_SEASON):
    """(256, 3) float32 slot -> Lab lookup for uint8 R images: rows 0-31 the slot tones, rows 32-255 zero (no slot)."""
    lut = np.zeros((256, 3), np.float32)
    lut[:32] = tone_table(season)
    return lut


def line_tone_jump(lab, row):
    """Per column CIE76 distance between the Lab means of the 4 rows below row and the 4 rows from row on, for a
    (rows, columns, 3) image across a line (the native-edge tone CUT, ground contrast fix design §10)."""
    a, c = lab[row:row + 4].mean(axis=0), lab[row - 4:row].mean(axis=0)
    return np.sqrt(((a - c) ** 2).sum(-1))

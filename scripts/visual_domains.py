"""Shared visual ownership, independent of height-edit alpha or gameplay changes.

build() is called once before ground source selection. Its south-up tile fields are
persisted by landform_tiles and read by the ground and props bakes. Explicit tile
edit exceptions may release native playable land; an explicit protect mask wins.
No input map, mask or gameplay field is modified.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 1
MASKS = ('native_keep', 'native_filler', 'canvas_land')
WEIGHTS = ('material_takeover', 'vegetation_takeover')
FIELDS = MASKS + WEIGHTS


def material_ecology(real_bio, edited_bio, names, takeover, legacy_steppe, blur,
                     *, authored_steppe=None, ecotone_enabled=True):
    """Material-only ecology from the shared ownership field.

    Height preparation keeps its original fields. Once material ownership is
    established, steppe donors and ecotone detail must not restart at the old
    native rectangle or its independent real-ground fade. No material is chosen
    here. The final selector still applies ownership to source probabilities.
    """
    import biome_profiles as bp
    weight = np.asarray(takeover, np.float32)
    if any(np.shape(a) != weight.shape for a in (real_bio, edited_bio, legacy_steppe)):
        raise ValueError('Material ecology inputs must share the tile frame')
    if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
        raise ValueError('Material ecology requires a valid ownership weight')
    slots = [names.index(n) for n in bp.biome_set('polish_steppe_donor') if n in names]
    geographical = np.isin(edited_bio, slots)
    if authored_steppe is not None:
        if np.shape(authored_steppe) != weight.shape:
            raise ValueError('Authored steppe must share the tile frame')
        geographical |= authored_steppe
    steppe = ((1-weight)*np.asarray(legacy_steppe, np.float32) +
              weight*geographical).astype(np.float32)
    zone = None
    if ecotone_enabled:
        # Weight the geographic source before smoothing. A thresholded legacy
        # real_gw mask would reintroduce a second, unrelated ownership boundary.
        changed = (edited_bio != real_bio).astype(np.float32) * weight
        zone = (np.clip(blur(changed, 4) * 4, 0, 1) * weight).astype(np.float32)
    return steppe, zone


def material_climate_shares(inherited, geographical, takeover):
    """Continuous donor climate support; unknown geography retains inheritance."""
    weight = np.asarray(takeover, np.float32) * (geographical < 4)
    return [((1-weight)*(inherited == c) + weight*(geographical == c)).astype(np.float32)
            for c in range(4)]


def steppe_donor(picked, weight, draw):
    """Resolve only the arid/steppe source-family split, never final material.

    draw(mask) is an independent world-coordinate uniform purpose. Its input is
    only the partial-weight arid pixels. This preserves exact T=0/1 endpoints
    without a new hard switch at an arbitrarily small positive takeover.
    """
    arid = picked == 3
    take = arid & (weight >= 1)
    partial = arid & (weight > 0) & (weight < 1)
    if partial.any():
        take[partial] = draw(partial) < weight[partial]
    return np.where(take, 9, picked)


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _array_sha(a):
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(json.dumps([a.dtype.str, list(a.shape)]).encode('ascii'))
    # Row chunks avoid copying a full strided map or raster just to hash it.
    for row in a:
        h.update(np.ascontiguousarray(row).tobytes())
    return h.hexdigest()


def producer_sha256():
    # Git archive writes LF; Windows checkouts may use CRLF. Hash code semantics
    # in UTF-8 with universal-newline normalization, not checkout line endings.
    return hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def domains_sha256(domains):
    return _json_sha({key: _array_sha(domains[key]) for key in FIELDS})


def _rules(rules=None):
    if rules is None:
        import biome_profiles as bp
        rules = bp.CONSTS['ground_materials']['VISUAL_DOMAINS']
    out = {}
    for key in ('material_ramp_hex', 'vegetation_ramp_hex'):
        pair = tuple(float(v) for v in rules[key])
        if len(pair) != 2 or not np.isfinite(pair).all() or not 0 <= pair[0] < pair[1]:
            raise ValueError(f'visual domains: invalid {key}: {pair}')
        out[key] = list(pair)
    if set(rules) != set(out):
        raise ValueError(f'visual domains: unknown rules {set(rules) - set(out)}')
    return out


def _tile_mask(mask, shape, key):
    if mask is None:
        return np.zeros(shape, bool)
    a = np.asarray(mask)
    if a.shape != shape or a.dtype != np.bool_:
        raise ValueError(f'visual domains: {key} must be a bool tile mask of shape {shape}')
    return a


def _ramp(distance, pair, eligible):
    t = np.clip((distance - pair[0]) / (pair[1] - pair[0]), 0, 1)
    t *= t * (3 - 2 * t)
    t[~eligible] = 0
    return t.astype(np.float32, copy=False)


def build(cells, names, shape, *, edit_exceptions=None, protect=None, rules=None):
    """Return three exact bool masks, two float32 weights, and provenance in meta.

    shape is a south-up tile shape. edit_exceptions/protect are explicit bool tile
    masks, never height alpha. Default native_keep is the original playable land;
    filler and outside-canvas land are eligible without changing their map regions.
    Distances are Euclidean in the half-hex tile lattice (0.5 hex per tile), from
    the protected set, with no reset at the native rectangle. With no protected
    tile all eligible land has full takeover. Forest target density does not
    include these ownership weights: consumers apply takeover exactly once.
    """
    import campaign_world as cw
    import revealed_filler as rf
    from scipy.ndimage import distance_transform_edt

    shape = tuple(int(v) for v in shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f'visual domains: invalid tile shape {shape}')
    cells = np.asarray(cells)
    if cells.ndim != 3 or cells.shape[:2] != (cw.H, cw.W) or cells.shape[2] != 16:
        raise ValueError('visual domains: cells must be the canvas map.hex cell array')
    rules = _rules(rules)
    exceptions = _tile_mask(edit_exceptions, shape, 'edit_exceptions')
    protect = _tile_mask(protect, shape, 'protect')
    playable = rf.vanilla_playable_hex()
    filler = rf.filler_hex(cells, names)
    if playable.shape != cells.shape[:2] or filler.shape != playable.shape:
        raise ValueError('visual domains: original masks disagree with the canvas')
    land = (cells[..., 0] & 3) != 1
    out = {key: np.zeros(shape, bool) for key in MASKS}
    eligible = np.zeros(shape, bool)
    tx = np.arange(shape[1], dtype=np.int32)[None, :]
    for y0 in range(0, shape[0], 128):
        y1 = min(y0 + 128, shape[0])
        ty = np.arange(y0, y1, dtype=np.int32)[:, None]
        hq, hr = cw.tile_hex(tx, ty)
        rect = ((tx >= cw.NATIVE_TILE_X0) & (tx < cw.NATIVE_TILE_X1) &
                (ty >= cw.NATIVE_TILE_Y0) & (ty < cw.NATIVE_TILE_Y1))
        physical = land[hr, hq]
        keep = (playable[hr, hq] & rect & ~exceptions[y0:y1]) | protect[y0:y1]
        out['native_keep'][y0:y1] = keep
        out['native_filler'][y0:y1] = filler[hr, hq] & rect
        out['canvas_land'][y0:y1] = physical & ~rect
        eligible[y0:y1] = physical & ~keep & (
            out['native_filler'][y0:y1] | out['canvas_land'][y0:y1] | exceptions[y0:y1])

    if out['native_keep'].any():
        distance = distance_transform_edt(~out['native_keep'], sampling=(.5, .5)).astype(np.float32)
        for key, rule in zip(WEIGHTS, ('material_ramp_hex', 'vegetation_ramp_hex')):
            out[key] = _ramp(distance, rules[rule], eligible)
    else:
        out.update({key: eligible.astype(np.float32) for key in WEIGHTS})
    out['meta'] = {
        'schema': SCHEMA, 'shape': list(shape), 'frame': 'tile-south-up',
        'method': 'smoothstep(distance_to_native_keep_hex); no height alpha; no native rectangle reset',
        'rules': rules, 'rules_sha256': _json_sha(rules),
        'producer_sha256': producer_sha256(), 'cells_sha256': _array_sha(cells),
        'names_sha256': _json_sha(list(names)), 'vanilla_playable_sha256': _array_sha(playable),
        'filler_hex_sha256': _array_sha(filler),
        'edit_exceptions_sha256': _array_sha(exceptions), 'protect_sha256': _array_sha(protect),
        'native_tile_rect': [cw.NATIVE_TILE_X0, cw.NATIVE_TILE_Y0, cw.NATIVE_TILE_X1, cw.NATIVE_TILE_Y1],
    }
    out['meta']['domains_sha256'] = domains_sha256(out)
    validate(out, shape)
    return out


def validate(domains, shape=None, check_hash=True, check_producer=False):
    """Check the exact shared fields. A loaded NPZ needs meta['visual_domains']
    passed as this dictionary's meta. Raises ValueError rather than rebuilding a
    missing mask from rounded weights. Returns the input dictionary.
    """
    m = domains['meta']
    if m.get('schema') != SCHEMA:
        raise ValueError(f'visual domains: unsupported schema {m.get("schema")}')
    shape = tuple(m['shape'] if shape is None else shape)
    if list(shape) != m['shape']:
        raise ValueError('visual domains: metadata shape mismatch')
    for key in FIELDS:
        a = np.asarray(domains[key])
        dtype = np.bool_ if key in MASKS else np.float32
        if a.shape != shape or a.dtype != dtype:
            raise ValueError(f'visual domains: {key} has wrong shape/dtype')
        if key in WEIGHTS:
            if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
                raise ValueError(f'visual domains: invalid {key} values')
            if (a[domains['native_keep']] != 0).any():
                raise ValueError(f'visual domains: {key} writes protected land')
    if np.any(domains['native_filler'] & domains['canvas_land']):
        raise ValueError('visual domains: native filler overlaps canvas land')
    if check_hash:
        if m.get('domains_sha256') != domains_sha256(domains):
            raise ValueError('visual domains: exact field hash mismatch')
        if m.get('rules_sha256') != _json_sha(_rules(m['rules'])):
            raise ValueError('visual domains: rules hash mismatch')
    if check_producer:
        if m.get('producer_sha256') != producer_sha256():
            raise ValueError('visual domains: stale producer; rerun ground bake')
        if m.get('rules_sha256') != _json_sha(_rules()):
            raise ValueError('visual domains: stale rules; rerun ground bake')
    return domains

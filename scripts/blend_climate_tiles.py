"""The blend's climate per tile: analysis/blend-climate-tiles.npz (bake step 4 product for the tile bake).

polish writes it right after global_blend.dds of 3k_dlc07_main_map (never in --cache-only, which does not
rewrite the blend), so the tile bake chooses every canvas tile's climate from the G band the ground actually
carries (biome_profiles.band_to_tile_climate) instead of the climate map (1.59 M tiles disagreed: cold tile
over subtropical blend, the old square sand patches).

Tile climate rule (user decision 2026-10-26, native ring): a canvas tile in 'ring' takes 'native', every other
canvas tile band_to_tile_climate(band); native tiles keep their inherited climate. polish writes band with the
blend (save); the tile bake (polish cache segment) adds ring and native from its own tile climate grid (add_ring),
once, before it applies the rule; tile_climate_matches_rule reads the same fields.

Fields (all rows south-up):
  band          uint8 (TILE_H, TILE_W), row 0 southmost, as tile_list y and polished-tile-climate.npy:
                the mode of the tile's 4x4 blend pixels' G band (0 cold, 1 arid, 2 temperate, 3 subtropical:
                CONSTS apply_climate BLEND_BAND). Ties take the smallest band.
  ring          bool (TILE_H, TILE_W): the native ring (ring_mask, CONSTS ground_materials NATIVE_RING).
  native        uint8 (TILE_H, TILE_W): polish's tile climate (tile-list order: sub_tropical 0, cold 1,
                temperate 2, arid 3; the native climate inherited and carried outward) on ring tiles, 255 elsewhere.
  meta          json: blend_sha256 (the global_blend.dds bytes as written), map_sha256, design_sha256,
                canvas, schema, native_band (the ring parameters, CONSTS ground_materials NATIVE_RING). A reader whose stage blend has another sha treats the file as stale.
numpy only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 2   # 2: ring, native
NAME = 'analysis/blend-climate-tiles.npz'


def band_tiles(cur):
    """(TILE_H, TILE_W) uint8 south-up from a north-first (H*4, W*4, 2) blend: mode of each 4x4 G block,
    ties to the smallest band."""
    h, w = cur.shape[0] // 4, cur.shape[1] // 4
    g = np.ascontiguousarray(cur[:h * 4, :w * 4, 1][::-1]).reshape(h, 4, w, 4)
    best = np.zeros((h, w), np.uint8); top = np.full((h, w), -1, np.int8)
    for b in range(4):                          # ascending: a later band wins only with a strictly larger count
        n = (g == b).sum((1, 3), dtype=np.int8)
        up = n > top; best[up] = b; top[up] = n[up]
    return best


def ring_mask(shape, q, r, origin):
    """bool (TILE_H, TILE_W): canvas tiles outside the native rectangle at less than R = reach + meander x m tile
    cells from it (biome_profiles.native_edge: the one native edge geometry; NATIVE_RING reach / meander).
    q, r: polish's tile-grid hex coordinates, origin its noise origin."""
    import biome_profiles as bp
    c = bp.CONSTS['ground_materials']['NATIVE_RING']
    dn, m = bp.native_edge(shape, q, r, origin)
    return (dn > 0) & (dn < c['reach'] + c['meander'] * m)


def save(A, dds_bytes, cur):
    """Write <A>/analysis/blend-climate-tiles.npz (band + meta); dds_bytes: global_blend.dds as written, cur: its
    pixels. The ring fields come later from the tile bake (add_ring)."""
    import landform_tiles
    path = Path(A) / NAME
    m = dict(landform_tiles.meta(A), schema=SCHEMA, blend_sha256=hashlib.sha256(dds_bytes).hexdigest())
    _write(path, {'band': band_tiles(cur)}, m)
    return path


def add_ring(A, tile_climate, q, r, origin):
    """Add ring and native to <A>/analysis/blend-climate-tiles.npz, from the tile bake's own tile climate grid
    (polish cache segment: the native climate inherited and carried outward, before the rule), once per tile bake:
    the tile bake and tile_climate_matches_rule then read the same ring and native. Keeps band and blend_sha256."""
    import biome_profiles as bp
    path = Path(A) / NAME
    with np.load(path) as z:
        band, m = z['band'], json.loads(str(z['meta'].reshape(-1)[0]))
    ring = ring_mask(tile_climate.shape, q, r, origin)
    native = np.where(ring, np.asarray(tile_climate), 255).astype(np.uint8)
    m['native_band'] = dict(bp.CONSTS['ground_materials']['NATIVE_RING'], **bp.CONSTS['ground_materials']['NATIVE_EDGE'])
    _write(path, {'band': band, 'ring': ring, 'native': native}, m)
    return path


def _write(path, arrays, m):
    from map_layers import savez_atomic   # byte-deterministic (np.savez_compressed stamps the time)
    savez_atomic(path, meta=np.array(json.dumps(m)), **arrays)


def load(A, blend_bytes=None, fields=False):
    """(band, meta), or with fields=True {'band', 'ring', 'native', 'meta'}; None when missing. blend_bytes:
    the stage's global_blend.dds, checked against blend_sha256 (RuntimeError when it differs: stale)."""
    path = Path(A) / NAME
    if not path.is_file():
        return None
    with np.load(path) as z:
        f = {k: z[k] for k in ('band', 'ring', 'native') if k in z.files}
        m = json.loads(str(z['meta'].reshape(-1)[0]))
    if blend_bytes is not None and hashlib.sha256(blend_bytes).hexdigest() != m['blend_sha256']:
        raise RuntimeError(f'{path} was written for another global_blend.dds (stale)')
    if fields:
        if 'ring' not in f:
            raise RuntimeError(f'{path} predates the native ring (schema {m.get("schema")}); rerun polish')
        return dict(f, meta=m)
    return f['band'], m


def tile_rule(f):
    """The canvas tile climate the rule asks for (tile-list order): native on the ring, band_to_tile_climate(band)
    elsewhere. Native tiles are not covered by the rule (they keep the inherited climate)."""
    import biome_profiles as bp
    return np.where(f['ring'], f['native'], bp.band_to_tile_climate(f['band'])).astype(np.uint8)

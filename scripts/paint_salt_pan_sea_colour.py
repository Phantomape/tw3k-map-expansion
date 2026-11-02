"""Paint vanilla's pale salt into sea_colour.dds under every salt pan (polish_campaign_terrain.py).

The salt-pan meshes are flat basins whose pond floors the engine fills with water, coloured
from terrain/campaigns/3k_main_map/sea_colour.dds (dlc07 has none; the dlc06 copy shares the
material). Vanilla paints pale salt (205, 206, 172) there under every pan; our canvas holds the
open-ocean fill (16, 52, 65) of expand_campaign_water_colour.py, so the ponds drew near-black.

Runs after sync_final_terrain_surface.py (whose polish --cache-only writes the pan cover) and
after the expand_campaign_water_colour.py side lane (which writes the textures fresh).
Reads  analysis/salt-pan-cover.npy and both sea_colour.dds in work/release-input;
writes those two sea_colour.dds in place and analysis/salt-pan-sea-colour.json.
Every BC1 block (every mip) whose area is at least half pan cover grown by GROW tile cells is
set to one solid salt block, so a second run changes nothing; other blocks stay byte-identical.

SALT LAKES (beautify G/H): before the pans, the saline lakes of biome_profiles REGIONS['salt_lakes']
(3k_main_sea_lake hex pieces of work/final-map/map.hex found by a lon/lat point through
analysis/west-real-terrain.npz) are painted turquoise the same way, the mechanism vanilla uses for its
own coloured water (the lake material samples this texture; vanilla's high lake, Xihai, is pale
green-grey there, its other lakes blue): blocks at least half lake cover (grown by grow_tiles onto the
shore) take one solid colour from rgb_shore to rgb_deep by the distance to the shore. Every other lake
keeps its colour. The pans are painted after the lakes, so they win any overlap.
"""
import json, struct, hashlib
import numpy as np
import campaign_world as _frame
import salt_pools
from expand_campaign_water_colour import KEYS, NEW_BOUNDS

A = _frame.WORKSPACE; S = A / 'work/release-input'
CELL_X = _frame.VANILLA_TILE_CELL
CELL_Z = _frame.VANILLA_TILE_CELL * _frame.ENGINE_Z_RATIO
GROW = 1          # tile cells: the water shader's filtered lookup at the pan rim still reads salt
SUB = 4           # sample points per block side for its cover share


def rgb565(rgb):
    r, g, b = (int(round(v * m / 255)) for v, m in zip(rgb, (31, 63, 31)))
    return (r << 11) | (g << 5) | b


def solid_block(rgb):
    c = rgb565(rgb)
    return np.frombuffer(struct.pack('<HHI', c, c, 0), np.uint8)   # c0 == c1: every index is colour 0


def grow(mask, n):
    out = mask.copy()
    for _ in range(n):
        g = out.copy(); g[1:] |= out[:-1]; g[:-1] |= out[1:]; g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
        g[1:, 1:] |= out[:-1, :-1]; g[:-1, :-1] |= out[1:, 1:]; g[1:, :-1] |= out[:-1, 1:]; g[:-1, 1:] |= out[1:, :-1]
        out = g
    return out


def paint(path, cover, boxes, level=None, palette=None):
    """Solid salt blocks where a block is at least half cover. level/palette (SALT LAKES): per-tile palette
    index (uint8) and the colours it indexes; a hit block takes the entry of the mean index over its covered
    sample points instead of salt."""
    data = path.read_bytes()
    assert data[84:88] == b'DX10' and struct.unpack_from('<I', data, 128)[0] in (71, 72)
    h, w = struct.unpack_from('<2I', data, 12); mips = struct.unpack_from('<I', data, 28)[0] or 1
    body = np.frombuffer(data, np.uint8).copy(); salt = solid_block(salt_pools.SALT_RGB)
    pal = None if palette is None else np.stack([solid_block(c) for c in palette])
    off, lw, lh, painted = 148, w, h, []
    TH, TW = cover.shape
    for lev in range(mips):
        bw, bh = max(1, (lw + 3) // 4), max(1, (lh + 3) // 4)
        blocks = body[off:off + bw * bh * 8].reshape(bh, bw, 8)
        n = 0
        for r0, r1, c0, c1 in boxes:
            # block rows/cols over the box's world extent (rows run north-first: UV v = 1 - z / H)
            x0, x1 = c0 * CELL_X, c1 * CELL_X; z0, z1 = r0 * CELL_Z, r1 * CELL_Z
            j0 = max(0, int(x0 / NEW_BOUNDS[0] * lw) // 4); j1 = min(bw, int(np.ceil(x1 / NEW_BOUNDS[0] * lw / 4)) + 1)
            i0 = max(0, int((1 - z1 / NEW_BOUNDS[1]) * lh) // 4); i1 = min(bh, int(np.ceil((1 - z0 / NEW_BOUNDS[1]) * lh / 4)) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            s = (np.arange(SUB) + .5) * 4 / SUB
            px = (np.arange(j0, j1)[:, None] * 4 + s[None]).ravel(); py = (np.arange(i0, i1)[:, None] * 4 + s[None]).ravel()
            col = np.clip(np.floor(px / lw * NEW_BOUNDS[0] / CELL_X).astype(int), 0, TW - 1)
            row = np.clip(np.floor((1 - py / lh) * NEW_BOUNDS[1] / CELL_Z).astype(int), 0, TH - 1)
            cv = cover[row[:, None], col[None, :]]
            share = cv.reshape(i1 - i0, SUB, j1 - j0, SUB).mean((1, 3))
            hit = share >= .5
            sub = blocks[i0:i1, j0:j1]
            if pal is None:
                n += int((hit & ~(sub == salt).all(-1)).sum())
                sub[hit] = salt
            else:
                lv = (level[row[:, None], col[None, :]].astype(np.float64) * cv).reshape(i1 - i0, SUB, j1 - j0, SUB).sum((1, 3))
                lv = lv / np.maximum(cv.reshape(i1 - i0, SUB, j1 - j0, SUB).sum((1, 3)), 1)
                want = pal[np.clip(np.rint(lv), 0, len(pal) - 1).astype(np.int64)]
                n += int((hit & ~(sub == want).all(-1)).sum())
                sub[hit] = want[hit]
        painted.append(n)
        off += bw * bh * 8; lw, lh = max(1, lw // 2), max(1, lh // 2)
    assert off == len(data)
    out = bytes(data[:148]) + body[148:].tobytes()
    if out != data:
        path.write_bytes(out)
    return {'path': path.relative_to(S).as_posix(), 'size': [w, h], 'mips': mips, 'blocks_painted_per_mip': painted,
            'input_sha256': hashlib.sha256(data).hexdigest(), 'output_sha256': hashlib.sha256(out).hexdigest()}


def cover_boxes(cover):
    """One box per cover group: the cover's 8-connected bounding boxes (r0, r1, c0, c1), row-major order."""
    boxes = []
    if cover.any():
        seen = np.zeros_like(cover)
        for r, c in zip(*np.nonzero(cover)):
            if seen[r, c]:
                continue
            r0, r1, c0, c1 = r, r + 1, c, c + 1
            while True:   # grow the box until no cover touches its border
                e = cover[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1]
                nr, nc = np.nonzero(e)
                R0, R1 = max(0, r0 - 1) + nr.min(), max(0, r0 - 1) + nr.max() + 1
                C0, C1 = max(0, c0 - 1) + nc.min(), max(0, c0 - 1) + nc.max() + 1
                if (R0, R1, C0, C1) == (r0, r1, c0, c1):
                    break
                r0, r1, c0, c1 = R0, R1, C0, C1
            seen[r0:r1, c0:c1] = True; boxes.append((int(r0), int(r1), int(c0), int(c1)))
    return boxes


# --- SALT LAKES --- (biome_profiles REGIONS['salt_lakes'])
LAKE_LEVELS = 16          # palette steps from rgb_shore to rgb_deep
LAKE_REGION = '3k_main_sea_lake'


def lake_cover(spec, cells, names, lon, lat):
    """(cover, level, report) on the tile grid for the salt lakes: the hex piece (hex_components.hex_label) of
    the lake region holding each point (nearest lake hex within snap_deg, by block-centre lon/lat) as its tile
    blocks, grown by grow_tiles cells; level: palette index by the Euclidean distance to the shore (cap deep_hex)."""
    import map_hex
    from hex_components import hex_label
    TH, TW = _frame.TILE_H, _frame.TILE_W
    ids = map_hex.region_ids(cells)
    lake = ((cells[:, :, 0] & 3) == 1) & (ids == names.index(LAKE_REGION)) if LAKE_REGION in names else np.zeros(ids.shape, bool)
    lab = hex_label(lake)
    rr, qq = np.nonzero(lake)
    ty = np.clip(2 * rr + qq % 2 + 1, 0, TH - 1); tx = np.clip(2 * qq + 1, 0, TW - 1)
    hlon, hlat = lon[ty, tx].astype(np.float64), lat[ty, tx].astype(np.float64)
    hexes = np.zeros(lake.shape, bool); report = {}
    for name, (plon, plat) in sorted(spec['lakes'].items()):
        d = np.hypot((hlon - plon) * np.cos(np.radians(plat)), hlat - plat) if len(rr) else np.zeros(0)
        d = np.where(np.isfinite(d), d, np.inf)
        i = int(np.argmin(d)) if len(d) else -1
        if i < 0 or d[i] > spec['snap_deg']:
            report[name] = {'hexes': 0, 'note': f'no lake hex within {spec["snap_deg"]} deg'}; continue
        piece = lab == lab[rr[i], qq[i]]
        pr, pq = np.nonzero(piece)
        hexes |= piece
        report[name] = {'hexes': int(piece.sum()), 'hex_q': [int(pq.min()), int(pq.max())], 'hex_r': [int(pr.min()), int(pr.max())],
                        'snap_deg': round(float(d[i]), 4)}
    cover = np.zeros((TH, TW), bool)
    hr, hq = np.nonzero(hexes)
    for dy in (0, 1):
        for dx in (0, 1):
            y = 2 * hr + hq % 2 + dy; x = 2 * hq + dx
            ok = (y < TH) & (x < TW); cover[y[ok], x[ok]] = True
    # Euclidean distance to the shore (tile cells, scipy EDT over the lake's padded bounding box), capped at deep_hex
    from scipy import ndimage
    cap = spec['deep_hex'] * _frame.TILE_PER_HEX
    level = np.zeros((TH, TW), np.uint8)
    if cover.any():
        ys, xs = np.nonzero(cover)
        y0, y1, x0, x1 = max(0, ys.min() - 1), min(TH, ys.max() + 2), max(0, xs.min() - 1), min(TW, xs.max() + 2)
        sub = np.pad(cover[y0:y1, x0:x1], 1)   # off-grid counts as shore (no wraparound)
        depth = ndimage.distance_transform_edt(sub)[1:-1, 1:-1] - .5
        level[y0:y1, x0:x1] = np.rint(np.clip(depth / max(cap, 1e-6), 0, 1) * (LAKE_LEVELS - 1)).astype(np.uint8) * cover[y0:y1, x0:x1]
    return grow(cover, spec['grow_tiles']), level, report


def lake_palette(spec):
    a, b = np.asarray(spec['rgb_shore'], np.float64), np.asarray(spec['rgb_deep'], np.float64)
    return [tuple(int(v) for v in np.rint(a + (b - a) * t)) for t in np.linspace(0, 1, LAKE_LEVELS)]


def salt_lakes():
    """Paint the salt lakes into both KEYS; returns the report section."""
    import biome_profiles as _bp, map_hex
    spec = _bp.region('salt_lakes')
    with np.load(A / 'analysis/west-real-terrain.npz') as z:
        x0, x1, y0, y1 = (int(v) for v in z['box'])
        lon = np.full((_frame.TILE_H, _frame.TILE_W), np.nan, np.float32); lat = lon.copy()
        lon[y0:y1, x0:x1] = z['lon']; lat[y0:y1, x0:x1] = z['lat']
    mh = map_hex.read(A / 'work/final-map/map.hex')
    cover, level, rep = lake_cover(spec, mh['cells'], mh['land_regions'] + mh['sea_regions'], lon, lat)
    del lon, lat, mh
    boxes = cover_boxes(cover)
    reports = [paint(S / key, cover, boxes, level, lake_palette(spec)) for key in KEYS]
    for t in reports:
        print(t['path'], 'salt-lake blocks painted per mip', t['blocks_painted_per_mip'], flush=True)
    return {'palette': lake_palette(spec), 'lakes': rep, 'cover_cells_grown': int(cover.sum()), 'boxes': len(boxes), 'textures': reports}
# --- end SALT LAKES


def main():
    lakes = salt_lakes()   # SALT LAKES first: the pans below win any overlap
    cover = np.load(A / 'analysis/salt-pan-cover.npy')
    assert cover.shape == (_frame.TILE_H, _frame.TILE_W)
    cover = grow(cover, GROW)
    # one box per pan group: the cover's 8-connected bounding boxes (grown cover, a few dozen)
    boxes = cover_boxes(cover)
    reports = [paint(S / key, cover, boxes) for key in KEYS]
    result = {'salt_rgb': list(salt_pools.SALT_RGB), 'grow_tile_cells': GROW, 'pan_boxes_tile_rows_cols': boxes,
              'cover_cells_grown': int(cover.sum()), 'textures': reports, 'salt_lakes': lakes, 'live_verified': False}
    (A / 'analysis/salt-pan-sea-colour.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    for t in reports:
        print(t['path'], 'salt blocks painted per mip', t['blocks_painted_per_mip'], flush=True)


if __name__ == '__main__':
    main()

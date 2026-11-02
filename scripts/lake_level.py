"""Lake water levels: one spec, solved once by the height bake (polish), read by everyone.

height-bake-design 2.1: a water level is not authored per lake, it is solved by the height bake from the
ground and written as a product that every later step reads. Until now every frontier lake showed its water
at one fixed level, LAKE_WATER; the real Bosten sits on a ~5-unit plateau, so polish dug a ~5-unit bowl round
it to reach that level (diagnosis bosten-level-20261031).

TW3K_LAKE_LEVEL=1 (default off) solves a level of its own for each lake of data/lake-level-spec.json (only
Bosten so far) and shapes the ground round it from one smooth signed distance to the lake outline (v2; v1 carved
by lake hexes and drew hex-sawtooth walls, lakelevel-runs/FINDINGS-v1.md). Off, nothing here changes any output.
v3 (plan B, user 2026-10-31: no town moves): the town level solve reads the product (town_envelope: the town ground on
the water side follows the bank envelope; v2 solved the town plateaus before the lake and left a terrace), the waterline
stops at the lake-hex edge at building plots instead of cutting back into the lake, and the reach is derived by polish
from the cluster mix's actual input and written to the product (v2 listed tied towns by hand and missed some).
v4 (user 2026-10-31): the bank and bed reach the LF evaluated per LF pixel from the LF signed distance (lf_shape_delta),
not as the bilinear resize of the tile-grid shape (v3: tile-grid gradients at the level crossing, axis share .73 against
the designed outline's .52). The tile shape stays for the tile-grid readers; same spec, same product.
v5 (user 2026-10-31, option a): the bed under the waterline (s < 0) is cut after the slope projection to the designed
bed profile (cut_bed, polish LAKE LEVEL BED, like RIVER BEDS): v4's projection held 100 % of the bed within .5 of the
waterline at its grade (TOWN_GRADE on per-axis edges: axis-aligned crossing gradients, lakelevel-runs/v4). The bed has
one writer after the projection; the land side (s >= 0) keeps the projection's ground.

  spec     data/lake-level-spec.json: per lake the seed point, the level policy, the outline (offset and
           smoothing of the lake hexes), the bank and bed profiles (the bank envelope), the freeboard and the
           mesh pad.
  product  <workspace>/analysis/lake-water-levels.json + lake-water-levels.npz (polish writes both): per lake
           the solved level, its cells, stamps, the reach towns and each window town's nearest plot distance
           and envelope cap (json), the signed distance to its waterline on the LF grid in a window round it
           and its reach (tile cells the switch may change; npz). The ground shaping, the town level solve and
           the projection bounds (polish), the water mesh (river_water_mesh) and the checks (stage_invariants)
           read the level, the freeboard, that distance and the reach from the product, nowhere else.

Waterline: the zero contour of the signed distance s. The shaped ground crosses the level there (h(0) = level).
  land (s >= 0)  level + clip(relief - level, floor(s), cap(s)), blended back to the relief beyond ring_width:
                 cap(s)   = cap_rise (1 - exp(-s / cap_length)) + cap_slope s   (vanilla low banks)
                 floor(s) = freeboard min(1, s / floor_length)                  (land under the mesh pad)
  bed  (s < 0)   level - depth (1 - (1 - u)^2), u = min(1, -s / floor_at)       (steep first, flat floor)
Bank envelope (town_envelope, v3): E(s) = max(freeboard, cap(s)) + (far_slope - cap_slope) ramp(s), ramp 0 out to
  ring_width, then (s - rw)^2 / (2 bw) over blend_width, then linear: the ground of the bank and of every town on the
  water side stays under level + E(s); further out E rises at far_slope (the projection grade) and never binds.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

LAKE_WATER = -.015       # the campaign water plane (world height): every lake without a level of its own
SURFACE_MARGIN = .03     # water meshes stand this far above their water level (river_water_mesh)
FINE = 4                 # LF pixels per tile cell (the LF and the tile grid share the frame: tile t = pixels 4t..4t+3)
ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / 'data/lake-level-spec.json'
PRODUCT = 'analysis/lake-water-levels.json'
FIELD = 'analysis/lake-water-levels.npz'
KIND = 'tw3k-lake-water-levels'
SPEC_VERSION = 3


def enabled():
    """TW3K_LAKE_LEVEL=1: per-lake water levels (default off: the one water plane, v20 lakes)."""
    return os.environ.get('TW3K_LAKE_LEVEL', '0') == '1'


def load_spec(path=SPEC_PATH):
    raw = Path(path).read_bytes()
    spec = json.loads(raw.decode('utf-8'))
    if spec.get('version') != SPEC_VERSION or not isinstance(spec.get('lakes'), dict):
        raise ValueError(f'{path}: not a version-{SPEC_VERSION} lake level spec')
    return spec, hashlib.sha256(raw).hexdigest()


def lf_pitch():
    """(z, x) world units per LF pixel."""
    import campaign_world as cw
    return cw.DZ / 2 / FINE, cw.DX / 2 / FINE


def find_component(lake_tiles, seed_tile_yx, search_cells):
    """The 8-connected component of lake_tiles (TH, TW bool) holding the seed tile cell, or the one nearest to
    it within search_cells (Chebyshev); ValueError when there is none. Returns (mask, label count)."""
    from scipy import ndimage
    sy, sx = (int(v) for v in seed_tile_yx)
    th, tw = lake_tiles.shape
    y0, y1, x0, x1 = max(0, sy - search_cells), min(th, sy + search_cells + 1), max(0, sx - search_cells), min(tw, sx + search_cells + 1)
    win = lake_tiles[y0:y1, x0:x1]
    if not win.any():
        raise ValueError(f'no lake cell within {search_cells} cells of tile {sy, sx}')
    wy, wx = np.nonzero(win)
    k = int(np.argmin(np.hypot(wy + y0 - sy, wx + x0 - sx) * 1e6 + (wy + y0) * 1e3 + (wx + x0) * 1e-3))   # nearest, ties by position
    lab, n = ndimage.label(lake_tiles, np.ones((3, 3), bool))
    return lab == lab[wy[k] + y0, wx[k] + x0], n


def shore_ring(comp, land, ring_cells):
    """Land cells within ring_cells (Euclidean, tile cells) of the component, in a window round it."""
    from scipy import ndimage
    ys, xs = np.nonzero(comp)
    pad = int(np.ceil(ring_cells)) + 2
    sl = (slice(max(0, ys.min() - pad), ys.max() + pad + 1), slice(max(0, xs.min() - pad), xs.max() + pad + 1))
    d = ndimage.distance_transform_edt(~comp[sl])
    ring = np.zeros(comp.shape, bool)
    ring[sl] = (d > 0) & (d <= ring_cells) & land[sl] & ~comp[sl]
    return ring


def solve_level(relief_orig, comp, land, entry):
    """The lake's water level from its shore: the ring_percentile of the original relief (before the coastal
    falloff and the lake bowl) on the land within ring_cells of the lake, minus level_below_ring (the level sits
    just under the surrounding plain, as vanilla's lakes do). Returns a dict (level and its ring statistics)."""
    ring = shore_ring(comp, land, entry['ring_cells'])
    if not ring.any():
        raise ValueError('lake without shore land in its ring')
    vals = np.asarray(relief_orig, np.float64)[ring]
    p = float(np.percentile(vals, entry['ring_percentile']))
    inside = np.asarray(relief_orig, np.float64)[comp]
    return {'level': round(p - float(entry['level_below_ring']), 4), 'ring_percentile_value': round(p, 4),
            'ring_cells': int(ring.sum()), 'ring_p5_p50_p95': [round(float(v), 4) for v in np.percentile(vals, [5, 50, 95])],
            'lake_relief_p5_p50_p95': [round(float(v), 4) for v in np.percentile(inside, [5, 50, 95])],
            'relief_ring_sha256': hashlib.sha256(np.ascontiguousarray(vals, '<f8').tobytes()).hexdigest()}


def window(comp, cells, shape):
    """Tile window (y0, y1, x0, x1) of the component's bounding box grown by cells, clipped to shape."""
    ys, xs = np.nonzero(comp)
    return (max(0, int(ys.min()) - cells), min(shape[0], int(ys.max()) + cells + 1),
            max(0, int(xs.min()) - cells), min(shape[1], int(xs.max()) + cells + 1))


def _edt(m, sampling):
    from scipy import ndimage
    return ndimage.distance_transform_edt(m, sampling=sampling) if m.any() else np.zeros(m.shape)


def outline_sdf(lake_t, plot_t, entry, pitch):
    """Signed distance to the waterline (world units, + land) on the LF grid of a tile window.

    lake_t, plot_t: (h, w) bool tile cells of the lake and of authored building plots in the window; pitch: (z, x)
    world units per LF pixel. The water region is the lake cells grown by outline_offset (vanilla's waterline lies
    0.25-0.45 outside the lake hexes) and smoothed by a Gaussian of outline_smooth (world); the growth never enters a
    plot cell (there the waterline stops at the lake-hex edge, plan B: the plot stays where it is) and the region
    always holds every lake cell (it never cuts back into the lake: vanilla's lake hexes are >= 98.6% wet). s is the
    Euclidean distance to its edge, smoothed by one pixel, then held <= -half a pixel on the lake cells and >= +half
    on the plot cells (the smoothing would otherwise move the line across a hex edge at a plot corner). Returns
    (s float32 (FINE h, FINE w), info dict)."""
    from scipy import ndimage
    pz, px = pitch
    lk = np.repeat(np.repeat(np.asarray(lake_t, bool), FINE, 0), FINE, 1)
    pl = (np.repeat(np.repeat(np.asarray(plot_t, bool), FINE, 0), FINE, 1) if plot_t is not None
          else np.zeros(lk.shape, bool)) & ~lk
    grown = ((_edt(~lk, (pz, px)) < entry['outline_offset']) & ~pl).astype(np.float32)
    sig = float(entry['outline_smooth'])
    water = ((ndimage.gaussian_filter(grown, (sig / pz, sig / px), mode='nearest', truncate=3.) >= .5) & ~pl) | lk
    del grown
    half = .25 * (pz + px)   # the edge lies half a pixel from the pixel centres on either side
    s = np.where(water, -(_edt(water, (pz, px)) - half), _edt(~water, (pz, px)) - half)
    s = ndimage.gaussian_filter(s, 1., mode='nearest', truncate=3.)
    s = np.where(lk, np.minimum(s, -half), np.where(pl, np.maximum(s, half), s)).astype(np.float32)
    return s, {'water_px': int(water.sum()), 'lake_px': int(lk.sum()), 'plot_px': int(pl.sum()),
               'plot_px_beside_water': int((pl & (s < 2 * half)).sum()), 'lake_px_dry': int((lk & (s >= 0)).sum())}


def tile_samples(s_fine):
    """The tile-cell values of a fine field as the bilinear resize back to the LF grid reads them: tile centre
    t + 1/2 lies at pixel 4t + 1.5 (the mean of pixels 4t+1 and 4t+2 on both axes)."""
    a = .5 * (s_fine[1::FINE].astype(np.float64) + s_fine[2::FINE])
    return (.5 * (a[:, 1::FINE] + a[:, 2::FINE])).astype(np.float32)


def vertex_samples(s_fine):
    """The values at the water mesh vertices (tile cell centres, LF pixel 4t+2: river_water_mesh.lf_cells)."""
    return s_fine[2::FINE, 2::FINE]


def bank_cap(s, b):
    sp = np.maximum(s, 0)
    return b['cap_rise'] * (1 - np.exp(-sp / b['cap_length'])) + b['cap_slope'] * sp


def bank_floor(s, b, freeboard):
    return freeboard * np.minimum(np.maximum(s, 0) / b['floor_length'], 1)


def bed_depth(s, d):
    u = np.clip(-np.asarray(s, np.float64) / d['floor_at'], 0, 1)
    return d['depth'] * (1 - (1 - u) ** 2)


def town_envelope(s, entry):
    """E(s): the bank envelope over the level (world units) at signed distance s (array): max(freeboard, bank_cap)
    out to ring_width, then rising to far_slope over blend_width (quadratic ramp), linear beyond; -inf under the
    waterline (s < 0). The bank and every town's ground on the water side stay under level + E(s) (v3)."""
    b = entry['bank']
    s = np.asarray(s, np.float64)
    rw, bw = float(b['ring_width']), float(b['blend_width'])
    u = s - rw
    ramp = np.where(u <= 0, 0., np.where(u < bw, u * u / (2 * bw), bw / 2 + u - bw))
    e = np.maximum(float(entry['freeboard']), bank_cap(s, b)) + (float(b['far_slope']) - float(b['cap_slope'])) * ramp
    return np.where(s >= 0, e, -np.inf)


def plot_distances(s, plot_label):
    """label -> smallest signed distance over the LF pixels of each label (plot_label: int LF grid like s, 0 =
    none): a town's nearest plot pixel to the waterline. Returns {label: float}."""
    from scipy import ndimage
    labs = np.unique(plot_label[plot_label > 0])
    if not len(labs):
        return {}
    v = ndimage.minimum(np.asarray(s, np.float64), plot_label, labs)
    return {int(k): float(x) for k, x in zip(labs, np.atleast_1d(v))}


def town_cap(level, s_min, entry):
    """The highest plateau level of a town whose nearest plot pixel lies s_min from the waterline: level + E(s_min);
    +inf without a plot in the window (s_min None or not finite)."""
    if s_min is None or not np.isfinite(s_min):
        return float('inf')
    return float(level) + float(town_envelope(np.array([s_min]), entry)[0])


def shaped_ground(relief, s, level, entry):
    """The ground round a level lake: (height, weight) for relief at signed distance s (same shapes). weight 0
    (beyond ring_width + blend_width): the relief itself, untouched."""
    b, fb = entry['bank'], float(entry['freeboard'])
    r = np.asarray(relief, np.float64); s = np.asarray(s, np.float64)
    land = level + np.clip(r - level, bank_floor(s, b, fb), bank_cap(s, b))
    t = np.clip((s - b['ring_width']) / b['blend_width'], 0, 1); t = t * t * (3 - 2 * t)
    land = land * (1 - t) + r * t
    h = np.where(s >= 0, land, level - bed_depth(s, entry['bed']))
    return h, 1 - t


def lf_shape_delta(pre_tile, shaped_tile, keep_tile, s, level, entry, resize):
    """v4: the shaped ground on the LF grid. The tile-grid shape (shaped_ground on the tile samples of s, written into
    the tile relief for the tile-grid readers: river carve, town solve) reaches the LF only through the bilinear resize
    of the relief, which draws the bank on the tile grid (4 LF px per tile: axis-biased gradients at the level
    crossing, lakelevel-runs/v3b). Returns the float32 LF window delta that replaces that resize by shaped_ground
    evaluated per LF pixel from the LF signed distance s, on the resized pre-shape relief:
        where(weight(s) > 0 & keep, shaped_ground(R(pre), s), R(pre)) - R(shaped_tile)
    pre_tile / shaped_tile: the window's tile relief before / after the tile shape (shaped_tile = pre where the tile
    shape did not write); keep_tile: tile cells the shape may write (polish: outer); resize: tile window -> the LF
    window grid (s.shape), polish's resize_high. Zero where neither shape writes (R(pre) = R(shaped_tile) there)."""
    rp = np.asarray(resize(np.asarray(pre_tile, np.float32)), np.float64)
    rt = np.asarray(resize(np.asarray(shaped_tile, np.float32)), np.float64)
    h, w = shaped_ground(rp, s, level, entry)
    keep = np.repeat(np.repeat(np.asarray(keep_tile, bool), FINE, 0), FINE, 1)[:s.shape[0], :s.shape[1]]
    return (np.where((w > 0) & keep, h, rp) - rt).astype(np.float32)


def cut_bed(encoded, s, level, entry, allowed, lf_level, lf_norm, lf_gain):
    """v5 LAKE LEVEL BED: the bed under the waterline on the final quantized LF, after the slope projection (whose grade
    held it ~.2 over the design at the waterline, lakelevel-runs/v4), the way RIVER BEDS cuts the river beds.
    encoded: the LF window (uint16, the same grid as s; changed in place); allowed: bool LF mask of pixels the cut may
    write (polish: not fixed, not a plot pixel). Every allowed pixel with s < 0 takes the quantized designed bed
    level - bed_depth(s); nothing at s >= 0 changes (the waterline, the bank and the towns keep the projection's ground).
    Returns a report (pixels written, raised / lowered counts, mean / max change in world units)."""
    s = np.asarray(s, np.float64)
    m = np.asarray(allowed, bool) & (s < 0)
    bed = level - bed_depth(s[m], entry['bed'])
    enc = np.rint(np.clip(lf_norm + (bed - lf_level) / lf_gain, 0, 1) * 65535)
    cur = encoded[m].astype(np.float64)
    d = (enc - cur) / 65535 * lf_gain
    encoded[m] = enc.astype(encoded.dtype)
    return {'pixels': int(m.sum()), 'raised': int((d > 0).sum()), 'lowered': int((d < 0).sum()),
            'mean_change': round(float(d.mean()), 4) if d.size else 0., 'max_abs_change': round(float(np.abs(d).max()), 4) if d.size else 0.}


def projection_bounds(s, level, entry, lsb, town=None):
    """(floor, ceil) on the LF grid for the slope projection: inside the waterline (s < 0) the bed stays under
    level - ceil_fraction x its depth (and at least one LSB under the level); on the land side within ring_width
    the ground stays over level + bank_floor(s) (+ one LSB) and under level + town_envelope(s) (+ one LSB), and so
    does town ground (town: bool LF mask, the pixels freed from a plateau, v3) at any s >= 0. -inf / +inf
    elsewhere."""
    s = np.asarray(s, np.float64)
    rw = entry['bank']['ring_width']
    bank = (s >= 0) & (s <= rw)
    fl = np.where(bank, level + bank_floor(s, entry['bank'], entry['freeboard']) + lsb, -np.inf)
    up = bank if town is None else bank | (np.asarray(town, bool) & (s >= 0))
    cl = np.where(s < 0, np.minimum(level - entry['ceil_fraction'] * bed_depth(s, entry['bed']), level) - lsb,
                  np.where(up, level + town_envelope(np.maximum(s, 0), entry) + lsb, np.inf))
    return fl.astype(np.float32), cl.astype(np.float32)


def cells_from_list(shape, yx):
    m = np.zeros(shape, bool)
    a = np.asarray(yx, np.int64).reshape(-1, 2)
    if len(a):
        m[a[:, 0], a[:, 1]] = True
    return m


def write_product(ws, lakes, stamp, fields=None, reach=None):
    """Atomic write of the product (lakes: name -> dict with level, cells_yx, window_tile, reach_window_tile ...;
    stamp: map / spec / canvas; fields: name -> float32 signed distance on the LF grid of its window; reach: name ->
    bool tile mask over its reach_window_tile; both written to FIELD first)."""
    p = Path(ws) / PRODUCT
    p.parent.mkdir(parents=True, exist_ok=True)
    if fields:
        f = Path(ws) / FIELD
        tmpf = f.with_name(f.name + '.tmp.npz')
        arrs = {f'sdf_{k}': np.asarray(v, np.float32) for k, v in sorted(fields.items())}
        arrs.update({f'reach_{k}': np.asarray(v, bool) for k, v in sorted((reach or {}).items())})
        np.savez(tmpf, **arrs)
        os.replace(tmpf, f)
        stamp = {**stamp, 'field_sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps({'kind': KIND, 'version': SPEC_VERSION, **stamp, 'lakes': lakes}, indent=1), encoding='utf-8')
    os.replace(tmp, p)
    return p


def read_product(ws, map_sha256=None):
    """The product, refused when it was solved on another map.hex (map_sha256 given) or by another version."""
    p = Path(ws) / PRODUCT
    if not p.is_file():
        raise FileNotFoundError(f'{p}: no lake water levels (run polish_campaign_terrain.py with TW3K_LAKE_LEVEL=1)')
    d = json.loads(p.read_text(encoding='utf-8'))
    if d.get('kind') != KIND or d.get('version') != SPEC_VERSION:
        raise ValueError(f'{p}: not a version-{SPEC_VERSION} lake water level product')
    if map_sha256 is not None and d.get('map_sha256') != map_sha256:
        raise ValueError(f'{p} was solved on map {d.get("map_sha256")}, current is {map_sha256}: rerun polish')
    return d


def read_fields(ws, product):
    """name -> (s float32 on the LF grid of its window, (ty0, ty1, tx0, tx1) tile window); refused when the npz is
    not the one the product was written with."""
    if not (product or {}).get('lakes'):
        return {}
    f = Path(ws) / FIELD
    raw = f.read_bytes()
    if hashlib.sha256(raw).hexdigest() != product.get('field_sha256'):
        raise ValueError(f'{f} is not the field of {PRODUCT} (sha mismatch): rerun polish')
    z = np.load(f)
    return {n: (z[f'sdf_{n}'], tuple(e['window_tile'])) for n, e in sorted(product['lakes'].items())}


def vertex_field(shape, fields):
    """(TH, TW) float64 signed distance at the water mesh vertices (tile cells) of every product lake (the
    smallest where windows overlap), +inf outside their windows; None without lakes."""
    if not fields:
        return None
    out = np.full(shape, np.inf)
    for n, (s, (y0, y1, x0, x1)) in sorted(fields.items()):
        v = vertex_samples(s)
        out[y0:y1, x0:x1] = np.minimum(out[y0:y1, x0:x1], v[:y1 - y0, :x1 - x0])
    return out


def water_field(shape, product, fields, pad_extra, base=LAKE_WATER):
    """(water level field, mesh-pad vertex mask) on the tile grid: each product lake's level at the vertices
    within its mesh_pad + pad_extra of the waterline (s < mesh_pad + pad_extra), base elsewhere; the mask holds
    the vertices with s < mesh_pad (the lake surface reaches past them). No lakes: (base, None)."""
    lakes = (product or {}).get('lakes') or {}
    if not lakes:
        return base, None
    field = np.full(shape, base, np.float64)
    pad = np.zeros(shape, bool)
    for name in sorted(lakes):
        e = lakes[name]; s, (y0, y1, x0, x1) = fields[name]
        v = vertex_samples(s)[:y1 - y0, :x1 - x0]
        sp = float(e['spec']['mesh_pad'])
        fw = field[y0:y1, x0:x1]; fw[v < sp + pad_extra] = float(e['level'])
        pw = pad[y0:y1, x0:x1]; pw |= v < sp
    return field, pad


def town_centres(design_regions):
    """region -> (tile y, tile x) of each design town (polish's town centre cell)."""
    return {t['region']: (int(round((t['town'][1] + t['town'][0] % 2 * .5) * 2)), int(t['town'][0] * 2)) for t in design_regions}


def reach_window(mask):
    """(y0, y1, x0, x1) bounding box of a bool tile mask (the stored part of a reach)."""
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def reach_mask(ws, product, shape):
    """(TH, TW) bool: the reach of the product lakes (the identity check's 'inside'), as polish derived it from the
    cluster mix's input (npz reach_<lake> over each lake's reach_window_tile; the field sha binds it to the json).
    Returns (mask, {lake: reach towns})."""
    out = np.zeros(shape, bool)
    lakes = (product or {}).get('lakes') or {}
    if not lakes:
        return out, {}
    f = Path(ws) / FIELD
    if hashlib.sha256(f.read_bytes()).hexdigest() != product.get('field_sha256'):
        raise ValueError(f'{f} is not the field of {PRODUCT} (sha mismatch): rerun polish')
    z = np.load(f)
    towns = {}
    for n, e in sorted(lakes.items()):
        y0, y1, x0, x1 = e['reach_window_tile']
        out[y0:y1, x0:x1] |= z[f'reach_{n}']
        towns[n] = e.get('reach_towns', [])
    return out, towns

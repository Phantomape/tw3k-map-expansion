"""Water surface meshes for the frontier rivers and lakes, in vanilla's river-model convention.

Vanilla draws its navigable rivers with props, not with lf_sea_height_map: 24 type-68 models
(river_water_codec) whose surface is a ribbon over a carved channel. Decoded conventions (river-
water-codec.json, all 24 models):
  * vertex y is the world/LF height itself: ribbon edge vertices lie ~0.15 under the final LF, the
    centre line 0.5-2.3 above the bed (the terrain clips the ribbon to the channel). Vanilla lake
    props sit at y 0.01-0.08 over shores at LF 0.0-0.4. (Only the type-101 sea meshes carry the
    -1.014 offset to lf_sea_height_map.)
  * uv0 = 0.1 per world unit both ways: v along the river, growing upstream (the shader pans the
    wave normals along +v, so they move downstream), u across, 0 on the centre line and growing to
    the right of the upstream direction; uv1 = position / world size.
  * triangles wind up-facing (cross(e1, e2).y > 0; the shader culls CCW), normal ~up (p50 0.7 deg),
    tangent = dP/dv (upstream), bitangent = dP/du, N.(T x B) = +1; normal, tangent and bitangent
    are packed z first (river_water_codec.pack_dir);
  * colour bytes are 0 in every vanilla river vertex; foam is not a vertex field: campaign_water.xml
    samples riverfoam_01 at world-position uvs and weights it by the coastal-wave terms of the water
    depth (eye-space water y minus the scene under it, scaled by up to 4 on steep ground), so foam
    follows the shallow channel edges, as along the Yellow River.

Here the surface is one grid over the tile-cell centres (LF pixel 4t+2, the cell of polish's river
level), shared by every model, so the surface is continuous across chunks and between river and lake:
  * height H = FLOOR off the river band; on it the river level (analysis/river-water-level.npy)
    made constant across the river (river_surface: the lowest level of each chain hex's section,
    blended along the river, never above the level itself), sinking into the channel near river ends
    on land, never below FLOOR. FLOOR is SURFACE_MARGIN above polish's LAKE_WATER, so no water
    surface is ever coplanar with the ocean background plane under it;
  * river quads: all corners in the level band, one within WET_REACH cells of a wet cell (final LF
    under H anywhere in the cell). The quad edge lies just past the water edge; the terrain clips it;
  * lake quads: frontier 3k_main_sea_lake hexes and the low ground (LF under FLOOR) connected to them,
    grown by LAKE_REACH cells: vanilla lakes are props at the lake's own height (campaign_lake_0*,
    small_lake_1), not the ocean plane, which showed through as flat plates;
  * mouth quads: open sea inside the blurred wet river cells, so the river surface ends in a rounded
    outline over the sea, not along the sea hexes' outline; river and lake surfaces part along the
    smooth half contour of the blurred lake ground;
  * every river hex centre lies on the surface; surface edges end under the final LF (bank cap);
    normals follow the surface before bank cap and end taper (H_shade), tangents the lattice
    differences of the global uv0 field, so both agree at every vertex models share;
  * each quad is split along its flatter diagonal;
  * flow uv: v is 0.1 x the distance from the river's mouths along its hex chain (Dijkstra, edges
    up to DIST_REACH hexes), u the offset from the chain; each vertex blends the linear charts of
    the chain hexes near it (weight falling along the river only; charts disagreeing with the
    nearest hex's by more than V_GATE are left out, so two branches or a loop never average);
  * models: the quads of each CHUNK x CHUNK cell tile and class, vertices sorted, all deterministic.
"""
import collections
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import campaign_world as cw
import lake_level
import river_water_codec as C
from hex_components import hex_dilate, hex_label
from terrain_noise import fbm

CELL = (cw.DX / 2, cw.DZ / 2)      # world units per tile cell (x, z)
HEX = cw.DZ                        # world units between neighbouring hex centres
LAKE_WATER = lake_level.LAKE_WATER         # the campaign water plane (one constant, shared with polish)
SURFACE_MARGIN = lake_level.SURFACE_MARGIN # water meshes stand at least this far above it
FLOOR = LAKE_WATER + SURFACE_MARGIN
UV_PER_WORLD = .1                  # vanilla uv0 scale (river-water-codec.json)
WET_REACH = 1                      # cells: river quads reach this far past a wet cell
LAKE_REACH = 2                     # cells: lake quads reach this far past the lake ground
FLOOR_PAST_PAD = 1.                 # world units: a level lake's floor (lake_surface) reaches this far past its mesh pad
PLUME_SIGMA, PLUME_LEVEL = 2., .2  # cells, share: mouth surface over the open sea (blurred wet cells)
LAKE_SIGMA = 2.                    # cells: river/lake surfaces part on the half contour of the blurred lake ground ...
SEAM_NOISE, SEAM_SCALE, SEAM_SEED = .45, 3., 137   # ... moved by this much low-frequency noise (features in hexes)
CHUNK = 160                        # cells per model tile side (~53 x 62 world units; vanilla models span 30-50)
BANK_CLEAR = .02                   # surface edges end at least this far under the final LF ...
CAP_SLOPE, CAP_STEPS = .3, 8       # ... rising inward at most CAP_SLOPE per cell (polish RIVER_SLOPE) over CAP_STEPS cells
V_GATE = .25                       # uv: charts further than this from the nearest hex's are dropped
GRAD_R = 3.                        # hexes: least-squares flow direction radius
DIST_REACH = 2.                    # hexes: upstream distance graph edges reach this far
EDGE_HEXES = 2                     # hexes: a river piece this close to the canvas edge leaves the map there
ALONG_R = 2.5                      # hexes: a chart's weight falls to 0 this far along its river ...
ACROSS_FLAT, ACROSS_R = 3., 5.     # ... and is flat this far across it, falling to 0 at ACROSS_R
END_TAPER_R = 1.5                  # hexes: at a river end on land the surface sinks into the channel over this distance ...
END_TAPER_DROP = .6                # ... by up to this much at the end hex (polish RIVER_DEPTH): the water narrows to a tip
END_CLEAR = 2                      # hexes: no end this close to open water or the canvas edge
FLAT_DROP = .1                     # the level made flat across the river stays at most this far under polish's level at
                                   # the vertex: on steep stretches a section's lowest level lay .2-.4 under it and under
                                   # the bed at the river centre (WR Kaidu 640-641,1063-1065 dry; polish RIVER BEDS keeps
                                   # every centre river_channels.CENTRE_DEPTH = .15 under the level)
RIVER_MATERIAL = 'materials/campaign_navigable_river_plane_default.xml.material'
LAKE_MATERIAL = 'materials/environment/campaign/3k_main_campaign_lakes.xml.material'
LAKE_SOUTH_MATERIAL = 'materials/environment/campaign/3k_main_campaign_lakes_south.xml.material'
OPEN_SQUARE = np.ones((3, 3), bool)


def lf_cells(field):
    """(centre, minimum) LF height per tile cell (TILE_H, TILE_W), float64: the centre texel 4t+2 and
    the lowest texel of the cell's 5 x 5 block (4t .. 4t+4, edges shared). field: final_terrain_surface
    .load_final_surface (rows south-up, u16)."""
    th, tw = cw.TILE_H, cw.TILE_W
    centre = np.empty((th, tw), np.float64)
    low = np.empty((th, tw), np.float64)
    h, w = field.shape
    for y0 in range(0, th, 256):
        y1 = min(th, y0 + 256)
        block = np.asarray(field[4 * y0:min(h, 4 * y1 + 1)], np.uint16)
        centre[y0:y1] = block[2::4][:y1 - y0, 2::4][:, :tw]
        pad = np.full((4 * (y1 - y0) + 1, 4 * tw + 1), 65535, np.uint16)
        pad[:block.shape[0], :block.shape[1]] = block[:4 * (y1 - y0) + 1, :4 * tw + 1]
        a = pad[:-1].reshape(y1 - y0, 4, 4 * tw + 1).min(1)
        a = np.minimum(a, pad[4::4][:y1 - y0])                       # the shared top edge row
        b = a[:, :-1].reshape(y1 - y0, tw, 4).min(2)
        low[y0:y1] = np.minimum(b, a[:, 4::4][:, :tw])               # the shared right edge column
    for a in (centre, low):
        a *= cw.GAIN / 65535
        a += cw.LEVEL - cw.NORM * cw.GAIN
    return centre, low


def cell_hex(shape):
    """(q, r) arrays of the hex owning each tile cell."""
    ty, tx = np.indices(shape)
    q = tx // 2
    return q, (ty - q % 2) // 2


def hex_centre(q, r):
    """World (x, z) of a hex centre: the shared corner of its 2 x 2 cell brick."""
    return (2 * np.asarray(q) + 1) * CELL[0], (2 * np.asarray(r) + np.asarray(q) % 2 + 1) * CELL[1]


def cell_world(ty, tx):
    """World (x, z) of tile-cell centres."""
    return (np.asarray(tx, np.float64) + .5) * CELL[0], (np.asarray(ty, np.float64) + .5) * CELL[1]


def outer_cells():
    out = np.ones((cw.TILE_H, cw.TILE_W), bool)
    out[cw.NATIVE_TILE_SLICE] = False
    return out


def lake_surface(product, fields, shape=None):
    """(floor, pad) of the water mesh: (FLOOR, None) (a scalar, as before) without a lake level product (lake_level,
    polish TW3K_LAKE_LEVEL=1); with one, floor is a (TILE_H, TILE_W) float64 field, each level lake's level +
    SURFACE_MARGIN at the vertices within its mesh_pad + FLOOR_PAST_PAD of its waterline and FLOOR elsewhere, and
    pad the vertices within mesh_pad of the waterline (plan: every quad touching one is a lake quad, so the mesh's
    outer vertices lie at least mesh_pad past the waterline, under the bank polish shaped from the same distance).
    fields: lake_level.read_fields of the product."""
    if not product or not product.get('lakes'):
        return FLOOR, None
    f, pad = lake_level.water_field(shape or (cw.TILE_H, cw.TILE_W), product, fields, FLOOR_PAST_PAD)
    return f + SURFACE_MARGIN, pad


def _at(F, *idx):
    """F (scalar or field) at idx."""
    return F if np.ndim(F) == 0 else F[idx]


def plan(level, lf_centre, lf_low, river_hex, lake_hex, sea_hex, cap=True, floor=None, lake_pad=None):
    """Cell/quad classes of the water surface.

    level: (TH, TW) river level (NaN off the band); lf_centre, lf_low: lf_cells; river_hex, lake_hex,
    sea_hex: (H, W) bool frontier river hexes (map.hex tw3k_riv_sea_*), frontier lake hexes, other open
    water. Returns dict with H (vertex heights), band, wet, river_quad, lake_quad, mouth_quad (quad (y, x)
    has corners (y..y+1, x..x+1)), lake_ground and counts. cap=False leaves the bank cap (bank_cap) to
    the caller, after it has reshaped the river surface (river_surface). floor, lake_pad: lake_surface (None:
    FLOOR, no level lake)."""
    shape = level.shape
    F = FLOOR if floor is None else floor
    outer = outer_cells()
    q, r = cell_hex(shape)
    q = np.clip(q, 0, river_hex.shape[1] - 1)
    r = np.clip(r, 0, river_hex.shape[0] - 1)
    band = np.isfinite(level) & outer
    if np.ndim(F) == 0:
        H = np.where(band, np.maximum(np.nan_to_num(level, nan=F), F), F).astype(np.float64)
    else:
        F32 = F.astype(np.result_type(level.dtype, np.float32))   # the scalar path's dtype (level's: float32), value for value
        H = np.where(band, np.maximum(np.where(np.isnan(level), F32, level), F32), F32).astype(np.float64)
    wet = band & (lf_low < H)
    lake_c = lake_hex[r, q] & outer
    sea_c = sea_hex[r, q] & outer & ~lake_c
    del q, r

    def corners(m):
        """Quads (TH-1, TW-1) with all four corners in m, and with any corner in m."""
        a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
        return a & b & c & d, a | b | c | d

    near_wet = ndimage.binary_dilation(wet, OPEN_SQUARE, iterations=WET_REACH) if WET_REACH else wet
    all_band, _ = corners(band)
    _, any_near = corners(near_wet)
    river_quad = all_band & any_near
    # Every river hex centre (the middle of the quad at its brick's lower-left cell) lies on the
    # surface, wet or not.
    hr, hq = np.nonzero(river_hex)
    hy, hx = 2 * hr + hq % 2, 2 * hq
    ok = (hy < shape[0] - 1) & (hx < shape[1] - 1)
    centre_q = np.zeros_like(river_quad)
    centre_q[hy[ok], hx[ok]] = True
    river_quad |= centre_q & all_band
    # Lake ground: lake cells and the low ground (LF under FLOOR) joined to them, never the open sea
    # or the river band; then LAKE_REACH cells of margin for the terrain to clip.
    low = (lf_low < F) & outer & ~sea_c & ~band
    lab, n = ndimage.label(low | lake_c, OPEN_SQUARE)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[lake_c])] = True
    keep[0] = False
    lake_ground = keep[lab]
    del lab, low
    lake_reach = ndimage.binary_dilation(lake_ground, OPEN_SQUARE, iterations=LAKE_REACH) & outer & ~sea_c
    all_lake, _ = corners(lake_reach)
    if lake_pad is not None:
        # level lakes: every quad touching a vertex within mesh_pad of the waterline (lake_level product)
        _, any_pad = corners(lake_pad & outer & ~sea_c)
        all_lake |= any_pad
        lake_ground = lake_ground | (lake_pad & outer & ~sea_c)
        del any_pad
    lake_quad = all_lake & ~river_quad
    # River and lake surfaces part along a smooth line (the half contour of the lake ground blurred by
    # LAKE_SIGMA cells), not along the band's end at the lake hexes' outline: the materials differ.
    field = ndimage.gaussian_filter(lake_ground.astype(np.float32), LAKE_SIGMA, mode='constant', truncate=3.)
    a, b, c, d = field[:-1, :-1], field[:-1, 1:], field[1:, :-1], field[1:, 1:]
    favg = (a.astype(np.float64) + b + c + d) / 4
    del field, a, b, c, d
    cand = river_quad | lake_quad
    # Where the two meet, the half contour wanders with low-frequency noise (SEAM_NOISE, features of
    # SEAM_SCALE hexes): the materials' shades part along an irregular shore-like line, not a straight
    # one across the river mouth.
    lakeish = favg >= .5
    sy, sx = np.nonzero(cand & (favg > .05) & (favg < .95))
    if len(sy):
        nz = fbm(sx / 2., sy / 2., (0., 0.), SEAM_SCALE, 2, seed=SEAM_SEED).astype(np.float64)
        lakeish[sy, sx] = favg[sy, sx] + SEAM_NOISE * nz >= .5
    del favg
    lake_quad = cand & (lakeish | (lake_quad & ~all_band))
    river_quad = cand & ~lake_quad
    del cand, lakeish
    # Mouths: the river surface runs on at FLOOR over the open sea past a wet river cell, to the
    # PLUME_LEVEL contour of the wet cells blurred by PLUME_SIGMA cells: a rounded end, not the
    # sea hexes' outline.
    blur = ndimage.gaussian_filter(wet.astype(np.float32), PLUME_SIGMA, mode='constant', truncate=3.)
    plume = sea_c & (blur >= PLUME_LEVEL)
    del blur
    all_pb, _ = corners(plume | band)
    _, any_plume = corners(plume)
    mouth_quad = all_pb & any_plume & ~river_quad & ~lake_quad
    out = {'H': H, 'band': band, 'wet': wet, 'river_quad': river_quad, 'lake_quad': lake_quad,
           'mouth_quad': mouth_quad, 'lake_ground': lake_ground, 'plume': plume, 'sea_cells': sea_c,
           'lake_cells': lake_c, 'capped_vertices': 0, 'bank_edge_vertices_over_lf': 0, 'floor': F}
    if cap:
        bank_cap(out, lf_centre)
    return out


def bank_cap(p, lf_centre):
    """Bank cap, in place on p['H']: a surface edge must end under the terrain. Where the final LF at
    an edge vertex lies under the surface (ground lowered after polish drew the level: the native-rim
    smoothing, town aprons), the surface is pulled down to BANK_CLEAR under it there, rising inward
    at most CAP_SLOPE per cell; never under FLOOR. Edges over ground under the water plane meet the
    ocean background plane instead and keep their height."""
    H = p['H']
    shape = H.shape
    F = p.get('floor', FLOOR)
    water = np.zeros(shape, bool)
    quads = p['river_quad'] | p['lake_quad'] | p['mouth_quad']
    inner = np.ones(shape, bool)
    for dy in (0, 1):
        for dx in (0, 1):
            water[dy:dy + shape[0] - 1, dx:dx + shape[1] - 1] |= quads
    full = np.zeros((shape[0] + 1, shape[1] + 1), bool)
    full[1:-1, 1:-1] = quads
    for dy in (0, 1):
        for dx in (0, 1):
            inner &= full[dy:dy + shape[0], dx:dx + shape[1]]
    edge = water & ~inner & (lf_centre >= (LAKE_WATER if np.ndim(F) == 0 else np.where(F == FLOOR, LAKE_WATER, F - SURFACE_MARGIN)))
    capped, nbad = 0, 0
    if edge.any():
        cap = np.where(edge, lf_centre - BANK_CLEAR, np.inf)
        bad = edge & (H > cap)
        nbad = int(bad.sum())
        if bad.any():
            # one window per cluster of offending vertices (their reach merged), in label order
            reach, _ = ndimage.label(ndimage.binary_dilation(bad, OPEN_SQUARE, iterations=CAP_STEPS + 1), OPEN_SQUARE)
            for sl in ndimage.find_objects(reach):
                c = np.where(bad[sl], cap[sl], np.inf)
                for _ in range(CAP_STEPS):
                    c = np.minimum(c, ndimage.grey_erosion(c, size=(3, 3), mode='nearest') + CAP_SLOPE)
                win = H[sl]
                new = np.where(water[sl], np.maximum(np.minimum(win, c), _at(F, *sl)), win)
                capped += int((new < win).sum())
                H[sl] = new
    p['capped_vertices'] = capped
    p['bank_edge_vertices_over_lf'] = nbad
    return p


def chain(river_hex, plan_, lake_hex, sea_hex):
    """River hex chain for the flow uv: the map's river hexes and the hexes whose cell brick is mostly
    wet band (visual links), with centres, labels and upstream distance (world units) from the mouths.
    Roots: chain hexes beside open water; a piece without one drains from its lowest hex."""
    Hh, Wh = river_hex.shape
    band, wet, H = plan_['band'], plan_['wet'], plan_['H']
    qq, rr = np.meshgrid(np.arange(Wh), np.arange(Hh))
    y0 = 2 * rr + qq % 2
    x0 = 2 * qq
    ok = (y0 + 1 < band.shape[0]) & (x0 + 1 < band.shape[1])
    y0c, x0c = np.where(ok, y0, 0), np.where(ok, x0, 0)
    wcount = sum(wet[y0c + dy, x0c + dx].astype(np.int32) for dy in (0, 1) for dx in (0, 1)) * ok
    bcount = sum(band[y0c + dy, x0c + dx].astype(np.int32) for dy in (0, 1) for dx in (0, 1)) * ok
    hmean = sum(H[y0c + dy, x0c + dx] for dy in (0, 1) for dx in (0, 1)) / 4
    outer_h = np.ones((Hh, Wh), bool)
    outer_h[cw.NATIVE_HEX_SLICE] = False
    ch = ((river_hex & (bcount > 0)) | (wcount >= 2)) & outer_h
    lab = hex_label(ch)
    # Roots (v = 0): a piece drains to the sea or off the canvas (its hexes beside the sea, or within
    # EDGE_HEXES of the canvas edge); failing both, into a lake; failing that, from its lowest hex. A
    # river leaving a lake for the sea or the canvas edge flows away from the lake.
    sea_m = ch & hex_dilate(sea_hex & ~river_hex)
    lake_m = ch & hex_dilate(lake_hex & ~river_hex)
    edge_m = np.zeros_like(ch)
    edge_m[:EDGE_HEXES] = True
    edge_m[-EDGE_HEXES:] = True
    edge_m[:, :EDGE_HEXES] = True
    edge_m[:, -EDGE_HEXES:] = True
    edge_m &= ch
    mouth = sea_m | lake_m
    roots, sinks = set(), set()
    for k in range(1, int(lab.max()) + 1):
        ys, xs = np.nonzero(lab == k)
        for m in (sea_m | edge_m, lake_m):
            hit = m[ys, xs]
            if hit.any():
                roots.update((int(y), int(x)) for y, x in zip(ys[hit], xs[hit]))
                break
        else:
            i = min(range(len(ys)), key=lambda j: (float(hmean[ys[j], xs[j]]), int(ys[j]), int(xs[j])))
            roots.add((int(ys[i]), int(xs[i])))
            sinks.add((int(ys[i]), int(xs[i])))
    ys, xs = np.nonzero(ch)
    cx, cz = hex_centre(xs, ys)
    xz = np.stack([cx, cz], 1).astype(np.float64)
    labels = lab[ys, xs]
    # Upstream distance: Dijkstra over the chain hexes of one piece, edges to every hex within
    # DIST_REACH hex steps at their straight length. Six-neighbour hex steps overstate a chain's
    # length by up to 15% (zigzags); 18 directions keep it within ~4%.
    n = len(xz)
    pairs = cKDTree(xz).query_pairs(DIST_REACH * HEX * 1.0001, output_type='ndarray') if n else np.zeros((0, 2), np.int64)
    pairs = pairs[labels[pairs[:, 0]] == labels[pairs[:, 1]]] if len(pairs) else pairs
    length = np.hypot(*(xz[pairs[:, 0]] - xz[pairs[:, 1]]).T) if len(pairs) else np.zeros(0)
    graph = coo_matrix((np.r_[length, length], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                       shape=(n, n)).tocsr()
    index = np.full((Hh, Wh), -1, np.int64)
    index[ys, xs] = np.arange(n)
    src = np.array(sorted(int(index[y, x]) for y, x in roots), np.int64)
    dist = dijkstra(graph, directed=False, indices=src, min_only=True) if n else np.zeros(0)
    assert np.isfinite(dist).all(), 'chain hex without a root'
    dist = dist.astype(np.float64)
    # River ends on land (END_TAPER_*): sources (no neighbouring chain hex further upstream) and the
    # sinks of pieces that reach no open water; never within END_CLEAR hexes of open water or the
    # canvas edge, where the surface runs on into the sea, a lake or off the map.
    up = np.full(n, -np.inf)
    nb = cKDTree(xz).query_pairs(HEX * 1.0001, output_type='ndarray') if n else np.zeros((0, 2), np.int64)
    nb = nb[labels[nb[:, 0]] == labels[nb[:, 1]]] if len(nb) else nb
    if len(nb):
        np.maximum.at(up, nb[:, 0], dist[nb[:, 1]])
        np.maximum.at(up, nb[:, 1], dist[nb[:, 0]])
    clear = (lake_hex | sea_hex) & ~river_hex
    for _ in range(END_CLEAR):
        clear = hex_dilate(clear)
    clear |= edge_m
    sink = np.zeros(n, bool)
    if sinks:
        sink[[int(index[y, x]) for y, x in sorted(sinks)]] = True
    end = (((up <= dist + 1e-9) & (dist > 0)) | sink) & ~clear[ys, xs]
    return {'r': ys, 'q': xs, 'xz': xz, 'label': labels, 'dist': dist, 'mask': ch,
            'mouths': int(mouth.sum()), 'roots': len(roots), 'ends': np.flatnonzero(end)}


def _kernel(d, radius):
    """Wendland C2 weight, 0 at radius."""
    t = np.clip(d / radius, 0, 1)
    return (1 - t) ** 4 * (4 * t + 1)


def flow_directions(ch):
    """Unit upstream direction per chain hex: weighted least-squares gradient of the upstream
    distance over the chain hexes of the same piece within GRAD_R."""
    P, dist, lab = ch['xz'], ch['dist'], ch['label']
    n = len(P)
    U = np.zeros((n, 2), np.float64)
    if not n:
        return U
    tree = cKDTree(P)
    pairs = tree.query_pairs(GRAD_R * HEX * 1.0001, output_type='ndarray')
    i = np.concatenate([pairs[:, 0], pairs[:, 1]]) if len(pairs) else np.zeros(0, np.int64)
    j = np.concatenate([pairs[:, 1], pairs[:, 0]]) if len(pairs) else np.zeros(0, np.int64)
    same = lab[i] == lab[j]
    i, j = i[same], j[same]
    dp = P[j] - P[i]
    dv = dist[j] - dist[i]
    w = _kernel(np.hypot(dp[:, 0], dp[:, 1]), GRAD_R * HEX * 1.0001)
    M = np.zeros((n, 2, 2), np.float64)
    b = np.zeros((n, 2), np.float64)
    for a in range(2):
        np.add.at(b[:, a], i, w * dp[:, a] * dv)
        for c in range(2):
            np.add.at(M[:, a, c], i, w * dp[:, a] * dp[:, c])
    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    good = det > 1e-9 * np.maximum(M[:, 0, 0] + M[:, 1, 1], 1e-12) ** 2
    g = np.zeros((n, 2), np.float64)
    g[good, 0] = (M[good, 1, 1] * b[good, 0] - M[good, 0, 1] * b[good, 1]) / det[good]
    g[good, 1] = (M[good, 0, 0] * b[good, 1] - M[good, 1, 0] * b[good, 0]) / det[good]
    # A straight chain makes M rank one: the gradient along it is b projected on its direction.
    tr = M[:, 0, 0] + M[:, 1, 1]
    line = ~good & (tr > 0)
    if line.any():
        ev = np.stack([M[line, 0, 0], M[line, 1, 0]], 1)
        ev2 = np.stack([M[line, 0, 1], M[line, 1, 1]], 1)
        e = np.where((np.hypot(ev[:, 0], ev[:, 1]) >= np.hypot(ev2[:, 0], ev2[:, 1]))[:, None], ev, ev2)
        e /= np.maximum(np.hypot(e[:, 0], e[:, 1]), 1e-12)[:, None]
        s = np.sum(b[line] * e, 1)
        g[line] = e * np.sign(s)[:, None]
    norm = np.hypot(g[:, 0], g[:, 1])
    U = np.where(norm[:, None] > 1e-12, g / np.maximum(norm, 1e-12)[:, None], np.array([1., 0.]))
    return U


def flow_uv(points, ch, U, values=None):
    """(u, v) uv0 of world (x, z) points: blended linear charts of the chain hexes (see module doc).
    A chart's weight falls with the distance along its river only (ALONG_R) and stays flat across
    it up to ACROSS_FLAT: weights falling across the river moved the blend's centre with the point
    and halved du/dx. Returns uv (n, 2) float64 and the count of points with no chart in reach;
    with values (one per chain hex, NaN: none), also their blend under the same weights (NaN where
    no chart with a value is in reach) and each point's nearest chain hex."""
    P, dist = ch['xz'], ch['dist']
    R = np.stack([U[:, 1], -U[:, 0]], 1)                     # right of the upstream direction
    tree = cKDTree(P)
    _, nearest = tree.query(points)
    lists = tree.query_ball_point(points, math.hypot(ALONG_R, ACROSS_R) * HEX, return_sorted=True)
    counts = np.fromiter((len(x) for x in lists), np.int64, len(points))
    idx = np.fromiter((k for x in lists for k in x), np.int64, int(counts.sum()))
    own = np.repeat(np.arange(len(points)), counts)
    dp = points[own] - P[idx]
    along = np.sum(dp * U[idx], 1)
    across = np.sum(dp * R[idx], 1)
    pv = UV_PER_WORLD * (dist[idx] + along)
    pu = UV_PER_WORLD * across
    dn = points - P[nearest]
    ref_v = UV_PER_WORLD * (dist[nearest] + np.sum(dn * U[nearest], 1))
    ref_u = UV_PER_WORLD * np.sum(dn * R[nearest], 1)
    w = (_kernel(np.abs(along), ALONG_R * HEX)
         * _kernel(np.maximum(np.abs(across) - ACROSS_FLAT * HEX, 0), (ACROSS_R - ACROSS_FLAT) * HEX)
         * (np.abs(pv - ref_v[own]) <= V_GATE))
    ws = np.bincount(own, w, len(points))
    su = np.bincount(own, w * pu, len(points))
    sv = np.bincount(own, w * pv, len(points))
    ok = ws > 1e-12
    uv = np.stack([np.where(ok, su / np.where(ok, ws, 1), ref_u), np.where(ok, sv / np.where(ok, ws, 1), ref_v)], 1)
    if values is None:
        return uv, int((~ok).sum())
    val = np.asarray(values, np.float64)[idx]
    has = np.isfinite(val)
    wv = np.where(has, w, 0.)
    wsv = np.bincount(own, wv, len(points))
    sval = np.bincount(own, wv * np.where(has, val, 0.), len(points))
    blend = np.where(wsv > 1e-12, sval / np.where(wsv > 1e-12, wsv, 1), np.nan)
    return uv, int((~ok).sum()), blend, nearest


def section_levels(level, band, wet, ch):
    """Per chain hex: the lowest river level of the band cells nearest it (its wet cells where it has
    any), float64; NaN where no band cell is nearest. polish's level is the soft-minimum bank height
    of each cell's own box, so it differs across one river by up to ~0.3 (the low bank's side lower):
    a surface drawn on it tilted 5-10 degrees in strips that turned as the low bank changed side."""
    n = len(ch['xz'])
    out = np.full(n, np.inf)
    out_wet = np.full(n, np.inf)
    ys, xs = np.nonzero(band)
    if not n or not len(ys):
        return np.full(n, np.nan)
    wx, wz = cell_world(ys, xs)
    _, k = cKDTree(ch['xz']).query(np.stack([wx, wz], 1))
    lv = level[ys, xs].astype(np.float64)
    np.minimum.at(out, k, lv)
    w = wet[ys, xs]
    np.minimum.at(out_wet, k[w], lv[w])
    res = np.where(np.isfinite(out_wet), out_wet, out)
    return np.where(np.isfinite(res), res, np.nan)


def lattice_tangents(keys, uv, W1):
    """Unit (x, 0, z) directions of dP/dv (the vanilla tangent: upstream) at lattice vertices keys
    (vy * W1 + vx, sorted), from the
    uv0 field's own lattice differences (central, one-sided at its border): the same at a vertex
    whichever model holds it (per-model triangle sums differed along model borders). Rows with no
    usable uv gradient get (0, 0, 1); returns (T (n, 3), count of those)."""
    n = len(keys)

    def nb(off):
        i = np.minimum(np.searchsorted(keys, keys + off), n - 1)
        return i, keys[i] == keys + off

    def diff(off, step):
        ip, hp = nb(off)
        im, hm = nb(-off)
        d = np.full((n, 2), np.nan)
        both = hp & hm
        d[both] = (uv[ip[both]] - uv[im[both]]) / (2 * step)
        only = hp & ~hm
        d[only] = (uv[ip[only]] - uv[only]) / step
        only = hm & ~hp
        d[only] = (uv[only] - uv[im[only]]) / step
        return d
    gx = diff(1, CELL[0])           # (du/dx, dv/dx)
    gz = diff(W1, CELL[1])          # (du/dz, dv/dz)
    det = gx[:, 0] * gz[:, 1] - gz[:, 0] * gx[:, 1]
    ok = np.isfinite(det) & (np.abs(det) > 1e-9)
    T = np.tile([0., 0., 1.], (n, 1))
    # (dx/dv, dz/dv) = (-du/dz, du/dx) / det
    T[ok, 0] = -gz[ok, 0] / det[ok]
    T[ok, 2] = gx[ok, 0] / det[ok]
    nrm = np.hypot(T[:, 0], T[:, 2])
    bad = ~ok | (nrm < 1e-12)
    T[bad] = [0., 0., 1.]
    T[~bad] /= nrm[~bad, None]
    return T, int(bad.sum())


def river_surface(p, level, ch, U):
    """The river surface, in place on p, then the bank cap: flow uv of every river/mouth vertex
    (uv_keys, uv), a level constant across the river (each chain hex's section_levels blended along
    the river with the flow_uv weights, never above polish's level at the vertex, never under FLOOR),
    sinking by up to END_TAPER_DROP within END_TAPER_R of a river end on land (the waterline closes
    to a tip on the channel's floor instead of stopping at the band's end), and H_shade: the surface
    before taper and cap, which the normals follow (the cap's 0.3-per-cell rim and the taper are under
    the banks; shading them tilted the visible edge strip)."""
    H = p['H']
    F = p.get('floor', FLOOR)
    W1 = H.shape[1] + 1
    rq = p['river_quad'] | p['mouth_quad']
    ys, xs = np.nonzero(rq)
    keys = np.unique(np.concatenate([ys, ys, ys + 1, ys + 1]).astype(np.int64) * W1 + np.concatenate([xs, xs + 1, xs, xs + 1]))
    del ys, xs, rq
    py, px = keys // W1, keys % W1
    wx, wz = cell_world(py, px)
    pts = np.stack([wx, wz], 1)
    sec = section_levels(level, p['band'], p['wet'], ch)
    uv, far, blend, nearest = flow_uv(pts, ch, U, sec)
    lv = level[py, px].astype(np.float64)
    flat_ok = p['band'][py, px] & np.isfinite(blend) & np.isfinite(lv)
    Fp = _at(F, py, px)
    flat = np.where(flat_ok, np.maximum(np.maximum(np.minimum(blend, lv), lv - FLAT_DROP), Fp), H[py, px])
    drop = np.zeros(len(keys))
    ends = ch['ends']
    if len(ends):
        d, j = cKDTree(ch['xz'][ends]).query(pts)
        same = ch['label'][ends[j]] == ch['label'][nearest]
        t = np.clip(d / (END_TAPER_R * HEX), 0, 1)
        drop = np.where(same & flat_ok, END_TAPER_DROP * (1 - t * t * (3 - 2 * t)), 0.)
    Hs = H.copy()
    Hs[py, px] = flat
    on = np.zeros(H.shape, bool)       # lattice points of any water quad
    quads = p['river_quad'] | p['lake_quad'] | p['mouth_quad']
    for dy in (0, 1):
        for dx in (0, 1):
            on[dy:dy + quads.shape[0], dx:dx + quads.shape[1]] |= quads
    Hs[~on] = np.nan
    del on, quads
    H[py, px] = np.maximum(flat - drop, Fp)
    bank_cap(p, p['lf_centre'])
    T, t_bad = lattice_tangents(keys, uv, W1)
    lowered = flat_ok & (flat < np.maximum(lv, Fp) - 1e-9)
    p.update(H_shade=Hs, uv_keys=keys, uv=uv, uv_far=far, tangent=T,
             surface_stats={'vertices': int(len(keys)), 'flattened': int(flat_ok.sum()),
                            'lowered_under_level': int(lowered.sum()),
                            'lowered_p50_p99_max': [round(float(x), 4) for x in np.percentile((np.maximum(lv, Fp) - flat)[lowered], [50, 99, 100])] if lowered.any() else [0, 0, 0],
                            'ends': int(len(ends)), 'tapered_vertices': int((drop > 0).sum()),
                            'tangent_fallback': t_bad})
    return p


def normals(H, vy, vx):
    """Unit normals of the height lattice at lattice points (vy, vx): central differences, one-sided
    at the grid border and beside NaN points (off the water surface: their FLOOR height under a
    river at level 4 tilted the edge normals up to 87 degrees); the same value whichever model holds
    the point."""
    th, tw = H.shape
    y0, y1 = np.maximum(vy - 1, 0), np.minimum(vy + 1, th - 1)
    x0, x1 = np.maximum(vx - 1, 0), np.minimum(vx + 1, tw - 1)
    x0 = np.where(np.isfinite(H[vy, x0]), x0, vx)
    x1 = np.where(np.isfinite(H[vy, x1]), x1, vx)
    y0 = np.where(np.isfinite(H[y0, vx]), y0, vy)
    y1 = np.where(np.isfinite(H[y1, vx]), y1, vy)
    gx = np.where(x1 > x0, (H[vy, x1] - H[vy, x0]) / (np.maximum(x1 - x0, 1) * CELL[0]), 0.)
    gy = np.where(y1 > y0, (H[y1, vx] - H[y0, vx]) / (np.maximum(y1 - y0, 1) * CELL[1]), 0.)
    n = np.stack([-gx, np.ones_like(gx), -gy], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def chunks(quad, kind):
    """[(tile (ty, tx), kind, quad rows, quad cols)] of quad mask per CHUNK tile, sorted."""
    ys, xs = np.nonzero(quad)
    if not len(ys):
        return []
    key = (ys // CHUNK) * (quad.shape[1] // CHUNK + 1) + xs // CHUNK
    order = np.lexsort((xs, ys, key))
    ys, xs, key = ys[order], xs[order], key[order]
    cut = np.flatnonzero(np.diff(key)) + 1
    out = []
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(ys)]):
        out.append(((int(ys[a] // CHUNK), int(xs[a] // CHUNK)), kind, ys[a:b], xs[a:b]))
    return out


def build_model(template, qy, qx, H, uv_of, kind, H_shade=None, tangent_of=None):
    """decode()-shaped model of quads (qy, qx) on the lattice: vertices (sorted cell centres),
    two up-facing triangles per quad split along its flatter diagonal ((a, c, b), (b, c, d) on b-c,
    else (a, c, d), (a, d, b) on a-d: a non-planar quad split across its fold showed as a triangle),
    uv0 from uv_of(ty, tx) -> (n, 2). Normals follow H_shade (default H); river tangents come from
    tangent_of(ty, tx) -> (n, 3) when given (the same at a vertex in every model), else from this
    model's triangles."""
    cy = np.concatenate([qy, qy, qy + 1, qy + 1])
    cx = np.concatenate([qx, qx + 1, qx, qx + 1])
    key = cy.astype(np.int64) * (H.shape[1] + 1) + cx
    uniq, inv = np.unique(key, return_inverse=True)
    assert len(uniq) <= C.U16_LIMIT + 1, len(uniq)
    vy, vx = uniq // (H.shape[1] + 1), uniq % (H.shape[1] + 1)
    n = len(qy)
    a, b, c, d = inv[:n], inv[n:2 * n], inv[2 * n:3 * n], inv[3 * n:]
    hv = H[vy, vx]
    ad = np.abs(hv[a] - hv[d]) < np.abs(hv[b] - hv[c]) - 1e-9
    t1 = np.where(ad[:, None], np.stack([a, c, d], 1), np.stack([a, c, b], 1))
    t2 = np.where(ad[:, None], np.stack([a, d, b], 1), np.stack([b, c, d], 1))
    tris = np.stack([t1, t2], 1).reshape(-1, 3)
    wx, wz = cell_world(vy, vx)
    wy = H[vy, vx]
    world = np.stack([wx, wy, wz], 1)
    lo32 = world.min(0).astype(np.float32)
    hi32 = world.max(0).astype(np.float32)
    pivot = ((lo32.astype(np.float64) + hi32.astype(np.float64)) / 2).astype(np.float32)
    v = np.zeros(len(uniq), C.VERTEX)
    v['pos'][:, :3] = (world - pivot.astype(np.float64)).astype(np.float32)
    v['pos'][:, 3] = 1
    uv0 = uv_of(vy, vx)
    v['uv'][:, 0:2] = uv0.astype(np.float32)
    v['uv'][:, 2] = (wx / cw.WORLD_WIDTH).astype(np.float32)
    v['uv'][:, 3] = (wz / cw.WORLD_HEIGHT).astype(np.float32)
    nrm = normals(H if H_shade is None else H_shade, vy, vx)
    v['normal'] = C.pack_dir(nrm)
    # Vanilla's frame (river_water_codec): tangent = dP/dv (upstream), bitangent = dP/du (right of
    # upstream), N.(T x B) = +1, all three packed z first.
    if kind == 'river' and tangent_of is not None:
        T = tangent_of(vy, vx)
    elif kind == 'river':
        _, T = _uv_frame(world, tris, uv0, nrm)
    else:
        T = np.tile([0., 0., 1.], (len(uniq), 1))      # lake uv0 = 0.1 (x, z): dP/dv = +z
    T = T - nrm * np.sum(T * nrm, 1, keepdims=True)
    v['tangent'] = C.pack_dir(T)
    v['bitangent'] = C.pack_dir(np.cross(nrm, T))
    m = dict(template)
    m['vertices'] = v
    m['triangles'] = tris.astype(np.uint16)
    m['bounds'] = tuple(float(x) for x in np.concatenate([lo32, hi32]))
    m['pivot'] = tuple(float(x) for x in pivot)
    return m, world, tris


def _uv_frame(world, tris, uv, nrm):
    """Per-vertex dP/du and dP/dv (area-weighted over its triangles), float64."""
    p0, p1, p2 = world[tris[:, 0]], world[tris[:, 1]], world[tris[:, 2]]
    t0, t1, t2 = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(r) > 1e-12
    r = np.where(ok, r, 1)
    du = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None] * ok[:, None]
    dv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None] * ok[:, None]
    area = np.linalg.norm(np.cross(e1, e2), axis=1)[:, None]
    T = np.zeros_like(world)
    B = np.zeros_like(world)
    for k in range(3):
        np.add.at(T, tris[:, k], du * area)
        np.add.at(B, tris[:, k], dv * area)
    bad = np.linalg.norm(T, axis=1) < 1e-12
    T[bad] = [1., 0., 0.]
    return T, B

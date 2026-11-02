"""Coarse land mesh down into the frontier river beds and lakes, so their water surfaces show.

The game draws the ground from the coarse land meshes (global_meshes/land_mesh_*, type 101, raw
frame: z * ZS is world z); where they have holes, fine tiles draw it from the LF. Water shows only
where a water surface stands above that ground (render_campaign_view --water meshes reproduces the
live 2048n result this way). Measured on the 2048 stage (audit in river_water_props' report):
  * vanilla river hexes (Yellow River, Yangtze, Xiang): 90% are land-mesh holes (generic_sea/2x2
    fine tiles), and where a triangle exists it follows the LF (land - LF p50 0.001, p5/p95 -0.01/0.03);
    vanilla lake hexes: 60% holes, the rest on the LF;
  * frontier river hexes: no holes, and polish's extension grids (step DX/2 x 1.15^k up to ~4 hexes,
    the vertex budget of each native mesh) stand 0.42 (p50) to 1.2 (p95) above the carved channel:
    43% of the river hex centres stay dry under them; frontier lakes 0.34 above (the diamond plates:
    the triangle fans of a 2-3 hex grid cut by the water plane).
Holes the vanilla way would need generic_sea tile records and their tile_map.tiles cache owners,
which polish's cache stage writes; this step does what vanilla's remaining triangles do instead:
it makes the coarse surface follow the LF over the water.

Every coarse triangle (outside the native map and its guard, not a vertical skirt) that holds a
lattice point (tile-cell corner, LF texel 4t) of the water footprint (the corners of the water
surface cells of river_water_mesh, grown by FOOT_REACH cells) is replaced by a triangulation of
  * its corners, kept; a corner all of whose triangles are replaced is seated on the LF;
  * points every EDGE_STEP along its edges where every face of the edge is replaced, on the LF,
    from the edge's lower-keyed end, so both triangles of an edge get the same points; an edge a
    kept face shares (or none) gets no points: no T-junction in the kept face, nothing for the seam
    loop's height normalisation (every edited vertex onto the LF) to pull off its straight edge;
  * the footprint lattice points inside it, on the LF;
(2-D Delaunay of that point set, checked to tile the triangle exactly; winding as the original).
New faces wholly at or below coarse_mesh_skirts.OCEAN_DROP_Y are left out, as the open-sea rule.
Faces go back into their own mesh while it has room (MAX_VERTICES), else, in 40-unit blocks, into
the nearest land mesh with room: no new mesh files (polish resets every land mesh it knows on each
run; a file only this step wrote would reach the next run's gap-fill, islands and seam steps). Each
changed mesh's compressed_map takes the LF over the footprint.

Runs after the islands step (last coarse-mesh edit of the graph), before the seam scan/repair.
Idempotent: a mesh whose bytes and compressed_map are exactly this step's last output (sha256 in
analysis/river-bed-mesh.json) is restored from before-river-bed-mesh/, whose bytes must hash to
the input that run recorded (else it stops); any other mesh is the pipeline's fresh output and is
used as it is (and backed up as this run's input). Land meshes added by an older version of this
step (report 'added_meshes') move to terrain-repair/river-bed-mesh-stale/<run>/.

  python river_bed_mesh.py [--apply]
"""
import argparse
import hashlib
import json
import math
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay

import campaign_world as cw
import map_hex
import memory_budget as MB
import river_water_mesh as M
import river_water_props as P
from coarse_mesh_skirts import OCEAN_DROP_Y
from fastops import cm_read, cm_write
from final_terrain_surface import load_final_surface, sample_final_surface

A = cw.WORKSPACE
S = A / 'work/release-input'
BACK = A / 'before-river-bed-mesh'
REPORT = A / 'analysis/river-bed-mesh.json'
STEP = cw.DX / 2            # raw units per tile cell (x and raw z)
SUB = 2                     # footprint points per tile cell side: the bed is seated every STEP / SUB (two LF texels)
FSTEP = STEP / SUB
EDGE_STEP = cw.DX / 4       # edge points every half tile cell
BED_TOL = .03               # a face under the water further than this from the LF at a footprint point is replaced
                            # (live 2048p: kept faces stood up to .5 above the bed, p95 .27, vanilla .005, and showed
                            # through the water as pale triangles)
FOOT_REACH = 1              # cells grown round the water surface cells
GUARD = 4                   # tile cells kept clear of the native map
MAX_VERTICES = 64000
SHOW, HIDE = .005, .02         # water this far over/under the LF shows/does not; a face within HIDE of it decides
KEY = 1e4                   # vertex position key: 1e-4 raw units
CM_WORKER_GB = 1.           # one compressed_map update's temporaries (bounds the thread pool by free commit)


def mesh_read(data):
    mp = struct.unpack_from('<I', data, 152)[0]
    typ, flags, size, vo, vc, io, ic = struct.unpack_from('<HH5I', data, mp)
    assert typ == 101 and io - vo == vc * 16, (typ, vo, vc, io)
    v = np.frombuffer(data, '<f4', vc * 4, mp + vo).reshape(vc, 4).copy()
    t = np.frombuffer(data, '<u2', ic, mp + io).reshape(-1, 3).astype(np.int64)
    return mp, v, t


def mesh_write(src, v, t):
    """Mesh bytes of template src (header up to its vertices) with vertices v (n, 4 f32) and faces t."""
    mp = struct.unpack_from('<I', src, 152)[0]
    typ, flags, size, vo, vc, io, ic = struct.unpack_from('<HH5I', src, mp)
    assert len(v) <= 65535 and (not len(t) or int(t.max()) < len(v))
    vb = np.ascontiguousarray(v, '<f4').tobytes()
    ib = np.ascontiguousarray(t, '<u2').tobytes()
    head = bytearray(src[:mp + vo])
    box = (float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 2].min()), float(v[:, 0].max()), float(v[:, 1].max()), float(v[:, 2].max()))
    struct.pack_into('<II', head, 144, len(vb), len(ib))
    struct.pack_into('<HH5I', head, mp, typ, flags, vo + len(vb) + len(ib), vo, len(v), vo + len(vb), t.size)
    struct.pack_into('<6f', head, mp + 24, *box)
    return bytes(head) + vb + ib, box


def keys_of(xz):
    k = np.rint(np.asarray(xz, np.float64) * KEY).astype(np.int64)
    assert (k >= 0).all() and (k < 1 << 30).all()
    return (k[..., 0] << 31) | k[..., 1]


def footprint(field, lvl, m):
    """Lattice corners (TH+1, TW+1) of the water footprint; the footprint points on the SUB-times finer
    lattice ((TH*SUB+1, TW*SUB+1), pitch FSTEP): every fine point round river and mouth water, only the tile-cell
    corners over lakes (a lake is deep and broad: seating its floor every FSTEP would not fit the vertex budget),
    and the fine points round river and mouth water alone; and the river_water_mesh plan."""
    p = P.surface(field, lvl, m)
    quads = p['river_quad'] | p['lake_quad'] | p['mouth_quad']
    cells = np.zeros(p['H'].shape, bool)
    for dy in (0, 1):
        for dx in (0, 1):
            cells[dy:dy + quads.shape[0], dx:dx + quads.shape[1]] |= quads
    # water surface height at each lattice corner: the highest water cell around it (-inf: none)
    wc = np.where(cells, p['H'], -np.inf)
    water = np.full((cells.shape[0] + 1, cells.shape[1] + 1), -np.inf)
    for dy in (0, 1):
        for dx in (0, 1):
            np.maximum(water[dy:dy + cells.shape[0], dx:dx + cells.shape[1]], wc, out=water[dy:dy + cells.shape[0], dx:dx + cells.shape[1]])
    del wc
    p['water_corner'] = water
    rcells = p['river_quad'] | p['mouth_quad']
    if FOOT_REACH:
        cells = ndimage.binary_dilation(cells, M.OPEN_SQUARE, iterations=FOOT_REACH)
        rcells = ndimage.binary_dilation(rcells, M.OPEN_SQUARE, iterations=FOOT_REACH + 1)
    corner = np.zeros((cells.shape[0] + 1, cells.shape[1] + 1), bool)
    rcorner = np.zeros_like(corner)
    for dy in (0, 1):
        for dx in (0, 1):
            corner[dy:dy + cells.shape[0], dx:dx + cells.shape[1]] |= cells
            rcorner[dy:dy + rcells.shape[0], dx:dx + rcells.shape[1]] |= rcells
    rcorner &= corner
    # keep clear of the native map and its guard
    x0, y0 = cw.NATIVE_TILE_X0 - GUARD, cw.NATIVE_TILE_Y0 - GUARD
    corner[max(0, y0):cw.NATIVE_TILE_Y1 + GUARD + 1, max(0, x0):cw.NATIVE_TILE_X1 + GUARD + 1] = False
    rcorner[max(0, y0):cw.NATIVE_TILE_Y1 + GUARD + 1, max(0, x0):cw.NATIVE_TILE_X1 + GUARD + 1] = False
    # the SUB-times finer lattice: a fine point lies in the footprint where a corner of its tile-cell edge / cell does
    fy = np.arange((corner.shape[0] - 1) * SUB + 1)
    fx = np.arange((corner.shape[1] - 1) * SUB + 1)
    ya, yb, xa, xb = fy // SUB, -(-fy // SUB), fx // SUB, -(-fx // SUB)
    rfine = rcorner[ya][:, xa] & rcorner[ya][:, xb] & rcorner[yb][:, xa] & rcorner[yb][:, xb]
    fine = rfine.copy()
    fine[::SUB, ::SUB] |= corner
    return corner, fine, rfine, p


def planar(c, pts):
    """Heights at (x, z) points pts of the plane through face corners c (3, 3 raw x, y, z)."""
    a, b, d = c
    m_ = np.array([[b[0] - a[0], d[0] - a[0]], [b[2] - a[2], d[2] - a[2]]])
    st = np.linalg.solve(m_, (pts - a[[0, 2]]).T)
    return a[1] + st[0] * (b[1] - a[1]) + st[1] * (d[1] - a[1])


def lf(field, x, z):
    return sample_final_surface(field, np.asarray(x, np.float64) / STEP, np.asarray(z, np.float64) / STEP)


def inside_points(tri, corner, margin=.2, pitch=FSTEP):
    """Footprint lattice points (corner: a lattice of this pitch) strictly inside triangle tri ((3, 2) raw x,
    z), at least margin lattice steps from its edges."""
    lo = np.floor(tri.min(0) / pitch).astype(int)
    hi = np.ceil(tri.max(0) / pitch).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [corner.shape[1] - 1, corner.shape[0] - 1])
    if (hi < lo).any():
        return np.zeros((0, 2))
    gx, gz = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    sel = corner[gz, gx]
    if not sel.any():
        return np.zeros((0, 2))
    pts = np.stack([gx[sel], gz[sel]], 1).astype(np.float64) * pitch
    a, b, c = tri
    area2 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    s = np.sign(area2)
    ok = np.ones(len(pts), bool)
    for p0, p1 in ((a, b), (b, c), (c, a)):
        e = p1 - p0
        cross = (e[0] * (pts[:, 1] - p0[1]) - e[1] * (pts[:, 0] - p0[0])) * s
        ok &= cross / max(np.hypot(*e), 1e-12) > max(margin * pitch, 1e-9)
    return pts[ok]


def inside_points_many(tris, corner, margin=.2, pitch=FSTEP, budget=1 << 24):
    """inside_points of every triangle of tris ((n, 3, 2) raw x, z) at once: yields (face index, points) per
    block of faces, the points of each face in its own order and the faces in order (the same arithmetic per
    point, so the same points). budget bounds the bounding-box lattice points of a block."""
    lo = np.maximum(np.floor(tris.min(1) / pitch).astype(int), 0)
    hi = np.minimum(np.ceil(tris.max(1) / pitch).astype(int), [corner.shape[1] - 1, corner.shape[0] - 1])
    nx, nz = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    cnt = np.where((nx > 0) & (nz > 0), nx * nz, 0)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    s = np.sign(area2)
    edges = [(p0, p1 - p0) for p0, p1 in ((a, b), (b, c), (c, a))]
    lens = [np.maximum(np.hypot(e[:, 0], e[:, 1]), 1e-12) for _, e in edges]
    tol = max(margin * pitch, 1e-9)
    ends = np.cumsum(cnt)
    i0 = 0
    while i0 < len(tris):
        i1 = max(i0 + 1, int(np.searchsorted(ends, ends[i0 - 1] + budget if i0 else budget, 'right')))
        n = cnt[i0:i1]
        f = np.repeat(np.arange(i0, i1), n)
        off = np.arange(len(f)) - np.repeat(np.cumsum(n) - n, n)
        gx = lo[f, 0] + off % nx[f]
        gz = lo[f, 1] + off // nx[f]
        sel = corner[gz, gx]
        f, gx, gz = f[sel], gx[sel], gz[sel]
        pts = np.stack([gx, gz], 1).astype(np.float64) * pitch
        ok = np.ones(len(pts), bool)
        for (p0, e), L in zip(edges, lens):
            cross = (e[f, 0] * (pts[:, 1] - p0[f, 1]) - e[f, 1] * (pts[:, 0] - p0[f, 0])) * s[f]
            ok &= cross / L[f] > tol
        yield f[ok], pts[ok]
        i0 = i1


def pick_faces(corners, tris, fine, rfine, water, field, ids):
    """The faces (ids, in order) that disagree with the LF about the water: at one of their footprint lattice
    points (fine) under the water a face hides water the LF shows, lets water through over dry LF (the
    straight wedges of a coarse fan beside a lake), or stands more than BED_TOL off the LF on a river bed
    point (rfine; the water shader is transparent by depth: a face above the bed shows through).
    corners (n, 3, 3) raw x, y, z of the faces, tris their (n, 3, 2) plan."""
    bad = np.zeros(len(ids), bool)
    for f, pts in inside_points_many(tris, fine, margin=0.):
        w = water[np.rint(pts[:, 1] / STEP).astype(int), np.rint(pts[:, 0] / STEP).astype(int)]
        live = np.isfinite(w)
        f, pts, w = f[live], pts[live], w[live]
        if not len(f):
            continue
        g = lf(field, pts[:, 0], pts[:, 1])
        # the plane of each face at its points: one solve per face, as planar (the same LAPACK call, so the
        # same heights; a closed-form 2x2 rounds differently)
        coarse = np.empty(len(f))
        cut = np.r_[0, np.flatnonzero(np.diff(f)) + 1, len(f)]
        for a, b in zip(cut[:-1], cut[1:]):
            coarse[a:b] = planar(corners[f[a]], pts[a:b])
        hidden = (w > g + SHOW) & (w < coarse + HIDE)     # water the LF shows, the face hides
        spilled = (w < g - SHOW) & (w > coarse - HIDE)    # water over dry LF the face lets through
        off = (rfine[np.rint(pts[:, 1] / FSTEP).astype(int), np.rint(pts[:, 0] / FSTEP).astype(int)]
               & (w > g + SHOW) & (np.abs(coarse - g) > BED_TOL))   # river bed the face misdraws
        bad[f[hidden | spilled | off]] = True
    return ids[bad].astype(np.int64)


def run(apply):
    lvl, stamp = P.level_checked()
    m = map_hex.read(A / 'work/final-map/map.hex')
    report = {'kind': 'river-bed-mesh', 'level_stamp': stamp, 'variants': [], 'applied': apply, 'live_verified': False}
    previous = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {}
    for variant in cw.TERRAIN_VARIANTS:
        pre = f'terrain/campaigns/{variant}/'
        gdir = S / pre / 'global_meshes'
        prev = next((v for v in previous.get('variants', []) if v['variant'] == variant), {})
        made = set(prev.get('added_meshes', []))
        written = prev.get('outputs', {})
        inputs = {}
        field = load_final_surface(variant, S)
        corner, fine, rfine, plan_ = footprint(field, lvl, m)
        water = plan_['water_corner']
        # --- inputs: the pipeline's meshes (our own earlier output restored from the backup)
        meshes = {}
        restored = []
        for path in sorted(gdir.glob('land_mesh_*.rigid_model_v2'), key=lambda q: int(q.stem.split('_')[-1])):
            if path.name in made:
                continue
            data = path.read_bytes()
            cmp_ = path.with_suffix('.compressed_map')
            cmd = cmp_.read_bytes()
            # Provenance: a mesh is our own earlier output only if its bytes and its compressed_map's are
            # exactly what the last run wrote (sha256 in the report); then its input comes back from the
            # backup, which must hash to the input that run recorded. Anything else (a fresh polish /
            # gap-fill / islands output, a seam repair after us) is the pipeline's and is used as it is.
            mine = written.get(path.name)
            if mine and mine[0] == hashlib.sha256(data).hexdigest() and mine[1] == hashlib.sha256(cmd).hexdigest():
                bdata = (BACK / pre / 'global_meshes' / path.name).read_bytes()
                bcm = (BACK / pre / 'global_meshes' / cmp_.name).read_bytes()
                want = prev.get('inputs', {}).get(path.name)
                if want != [hashlib.sha256(bdata).hexdigest(), hashlib.sha256(bcm).hexdigest()]:
                    raise SystemExit(f'{path.name} is an earlier output of this step but its backup is not the input that run '
                                     f'recorded: rerun the coarse steps from polish')
                data, cmd = bdata, bcm
                restored.append(path.name)
            inputs[path.name] = [hashlib.sha256(data).hexdigest(), hashlib.sha256(cmd).hexdigest()]
            meshes[path.name] = [data, cmd]
        names = list(meshes)
        decoded = {n: mesh_read(meshes[n][0]) for n in names}
        # --- candidate faces
        cs = np.zeros((corner.shape[0] + 1, corner.shape[1] + 1), np.int64)
        cs[1:, 1:] = np.cumsum(np.cumsum(corner, 0), 1)
        refine, allkeys = {}, []
        for n in names:
            _, v, t = decoded[n]
            tri = v[t][:, :, [0, 2]].astype(np.float64)
            area = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
            lo = np.clip(np.floor(tri.min(1) / STEP).astype(np.int64), 0, [corner.shape[1] - 1, corner.shape[0] - 1])
            hi = np.clip(np.ceil(tri.max(1) / STEP).astype(np.int64), 0, [corner.shape[1] - 1, corner.shape[0] - 1])
            cnt = cs[hi[:, 1] + 1, hi[:, 0] + 1] - cs[lo[:, 1], hi[:, 0] + 1] - cs[hi[:, 1] + 1, lo[:, 0]] + cs[lo[:, 1], lo[:, 0]]
            tx, tz = tri[:, :, 0] / STEP, tri[:, :, 1] / STEP
            near_native = ((tx > cw.NATIVE_TILE_X0 - GUARD) & (tx < cw.NATIVE_TILE_X1 + GUARD)
                           & (tz > cw.NATIVE_TILE_Y0 - GUARD) & (tz < cw.NATIVE_TILE_Y1 + GUARD)).any(1)
            cand = np.flatnonzero((cnt > 0) & (np.abs(area) > 1e-6) & ~near_native)
            # a face is replaced where it disagrees with the LF about the water; others stay as they are
            refine[n] = pick_faces(v[t[cand], :3].astype(np.float64), tri[cand], fine, rfine, water, field, cand)
            k = keys_of(v[t][:, :, [0, 2]])
            allkeys.append(k)
        # --- edge / vertex classification over all meshes
        K = np.concatenate(allkeys)                                   # (F, 3) vertex keys per face
        flag = np.concatenate([np.isin(np.arange(len(decoded[n][2])), refine[n]) for n in names])
        e = np.concatenate([K[:, [0, 1]], K[:, [1, 2]], K[:, [2, 0]]])
        ef = np.concatenate([flag, flag, flag])
        e.sort(1)
        order = np.lexsort((e[:, 1], e[:, 0]))
        es, efs = e[order], ef[order]
        start = np.r_[0, np.flatnonzero((np.diff(es[:, 0]) != 0) | (np.diff(es[:, 1]) != 0)) + 1]
        total = np.diff(np.r_[start, len(es)])
        refd = np.add.reduceat(efs.astype(np.int64), start)
        ukeys = es[start]
        inner_edge = (refd == total) & (total >= 2)
        edge_inner = {(int(a), int(b)): bool(f) for (a, b), f, r_ in zip(ukeys, inner_edge, refd) if r_ > 0}
        vk = K.ravel()
        vf = np.repeat(flag, 3)
        vo_ = np.argsort(vk, kind='stable')
        vks, vfs = vk[vo_], vf[vo_]
        vst = np.r_[0, np.flatnonzero(np.diff(vks)) + 1]
        vtot = np.diff(np.r_[vst, len(vks)])
        vref = np.add.reduceat(vfs.astype(np.int64), vst)
        vertex_inner = set(int(x) for x in vks[vst][(vref == vtot)])
        del K, e, es, efs, order, vk, vf, vks, vfs
        # --- refill
        edge_cache = {}

        def edge_points(ka, pa, kb, pb):
            """Inner points of edge (a, b), from the lower key, with heights (n, 3) raw x, y, z: on the
            LF every EDGE_STEP where every face of the edge is replaced, none where a kept face (or
            nothing) is on the other side."""
            rev = ka > kb
            if rev:
                ka, pa, kb, pb = kb, pb, ka, pa
            key = (ka, kb)
            if key not in edge_cache:
                # An edge a kept face shares stays the bare edge (its two corners). Points on the
                # straight edge would be T-junctions of the kept face, and the seam loop's
                # normalize_coarse_partition_heights seats every vertex of the edited area on the LF:
                # they would leave the kept face's straight edge by its whole LF error (up to ~1.4
                # beside the canvas mesh's 8-hex triangles), a coarse_step that each seam repair
                # pass moves one kept triangle further instead of closing.
                if edge_inner.get(key, False):
                    L = math.hypot(pb[0] - pa[0], pb[2] - pa[2])
                    n_ = max(1, int(math.ceil(L / EDGE_STEP - 1e-9)))
                    s = np.arange(1, n_) / n_
                    xyz = pa[None, :] + (pb - pa)[None, :] * s[:, None]
                    if len(s):
                        xyz[:, 1] = lf(field, xyz[:, 0], xyz[:, 2])
                else:
                    xyz = np.zeros((0, 3))
                edge_cache[key] = xyz
            out = edge_cache[key]
            return out[::-1] if rev else out

        def area2(p, f):
            return ((p[f[:, 1], 0] - p[f[:, 0], 0]) * (p[f[:, 2], 1] - p[f[:, 0], 1])
                    - (p[f[:, 2], 0] - p[f[:, 0], 0]) * (p[f[:, 1], 1] - p[f[:, 0], 1]))

        added = {n: [] for n in names}                                 # new faces (k, 3, 3) raw xyz per owner
        stats = {'faces_replaced': 0, 'faces_added': 0, 'faces_dropped_open_sea': 0, 'fan_fills': 0, 'corners_seated': 0,
                 'interior_points': 0}
        seat = {}
        for n in names:
            _, v, t = decoded[n]
            for i in refine[n]:
                c = v[t[i], :3].astype(np.float64)
                ck = keys_of(c[:, [0, 2]])
                corners = c.copy()
                for j in range(3):
                    if int(ck[j]) in vertex_inner:
                        corners[j, 1] = seat.setdefault(int(ck[j]), float(lf(field, c[j:j + 1, 0], c[j:j + 1, 2])[0]))
                ring = []
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    ring.append(corners[a:a + 1])
                    ring.append(edge_points(int(ck[a]), corners[a], int(ck[b]), corners[b]))
                ring = np.concatenate(ring)                            # boundary polygon, in order
                ip = inside_points(c[:, [0, 2]], fine)
                stats['interior_points'] += len(ip)
                pts = ring
                if len(ip):
                    pts = np.concatenate([ring, np.column_stack([ip[:, 0], lf(field, ip[:, 0], ip[:, 1]), ip[:, 1]])])
                p2 = pts[:, [0, 2]]
                orig = float(area2(c[:, [0, 2]], np.array([[0, 1, 2]]))[0])
                faces = None
                op = p2                                           # plan points the winding is read from
                if len(pts) == 3:
                    faces = np.array([[0, 1, 2]])
                else:
                    try:
                        # Boundary points (corners, edge points) pulled a hair towards the centre: every
                        # point is then strictly inside the hull and a Delaunay vertex, and consecutive
                        # edge points are joined by slivers that are flat in plan (vertical in 3-D where an
                        # edge point left the straight edge: they close the edge, not a crack).
                        cen2 = p2[:3].mean(0)
                        q2 = p2.copy()
                        q2[:len(ring)] += (cen2 - p2[:len(ring)]) * 1e-6
                        f = Delaunay(q2).simplices
                        ar = area2(p2, f)
                        sgn = np.sign(orig)
                        if (len(np.unique(f)) == len(pts) and abs((ar * sgn).sum() - abs(orig)) <= 1e-6 * abs(orig)
                                and (ar * sgn > -1e-9 * abs(orig)).all()):
                            faces, op = f, q2
                    except Exception:   # qhull refuses (degenerate input): the fan below
                        faces = None
                if faces is None:
                    # fan from the centre over the boundary polygon (always exact on a triangle's ring)
                    stats['fan_fills'] += 1
                    cen = c.mean(0)
                    cen[1] = float(lf(field, cen[:1], cen[2:3])[0]) if ip.size else float(corners[:, 1].mean())
                    pts = np.concatenate([ring, cen[None]])
                    p2 = pts[:, [0, 2]]
                    k = len(ring)
                    faces = np.stack([np.arange(k), (np.arange(k) + 1) % k, np.full(k, k)], 1)
                    op = p2
                ar = area2(op, faces)
                flip = np.sign(ar) != np.sign(orig)
                faces[flip] = faces[flip][:, [0, 2, 1]]
                xyz = pts[faces]
                drop = (xyz[:, :, 1] <= OCEAN_DROP_Y).all(1)
                stats['faces_dropped_open_sea'] += int(drop.sum())
                added[n].append(xyz[~drop])
                stats['faces_replaced'] += 1
                stats['faces_added'] += int((~drop).sum())
        stats['corners_seated'] = len(seat)

        # --- assemble: new faces join their own mesh while it has room; blocks that do not fit join the
        # nearest mesh with room (no new mesh files: polish resets every land mesh it knows each run, and
        # a file only this step wrote would survive into the next run's coarse steps)
        def pack(xyz):
            """(vertices (n, 4) f32, faces) of face corners (k, 3, 3), vertices deduplicated by position."""
            flat = xyz.reshape(-1, 3)
            k = keys_of(flat[:, [0, 2]])
            uk, first, inv = np.unique(k, return_index=True, return_inverse=True)
            vv = np.column_stack([flat[first], np.ones(len(first))]).astype('<f4')
            return vv, inv.reshape(-1, 3)

        def merged(base_v, base_t, new):
            """base mesh plus faces new (k, 3, 3); new vertices meet base vertices at the same position
            and height. (vertices, faces)"""
            if not len(new):
                return base_v, base_t
            nv, nt = pack(new)
            bk = keys_of(base_v[:, [0, 2]]) if len(base_v) else np.zeros(0, np.int64)
            nk = keys_of(nv[:, [0, 2]])
            order = np.argsort(bk, kind='stable')
            pos = np.minimum(np.searchsorted(bk[order], nk), max(len(bk) - 1, 0))
            same = np.zeros(len(nv), bool)
            remap = np.zeros(len(nv), np.int64)
            if len(bk):
                bi = order[pos]
                same = (bk[bi] == nk) & (np.abs(base_v[bi, 1] - nv[:, 1]) < 1e-5)
                remap[same] = bi[same]
            fresh = np.flatnonzero(~same)
            remap[fresh] = len(base_v) + np.arange(len(fresh))
            return np.vstack([base_v, nv[fresh]]), np.vstack([base_t, remap[nt]])

        def fits(base_v, new):
            if not len(new):
                return True
            nk = np.unique(keys_of(new.reshape(-1, 3)[:, [0, 2]]))
            bk = keys_of(base_v[:, [0, 2]]) if len(base_v) else np.zeros(0, np.int64)
            return len(base_v) + len(np.setdiff1d(nk, bk)) <= MAX_VERTICES

        cur, changed, spill, mesh_report = {}, set(), [], {}
        for n in names:
            mp, v, t = decoded[n]
            vk = keys_of(v[:, [0, 2]])
            moved = np.isin(vk, np.fromiter(seat, np.int64, len(seat)))
            keep = np.ones(len(t), bool)
            keep[refine[n]] = False
            if keep.all() and not moved.any():
                cur[n] = None                       # untouched, loaded on demand
                continue
            v = v.copy()
            for j in np.flatnonzero(moved):
                v[j, 1] = seat[int(vk[j])]
            used = np.unique(t[keep])
            base_v, base_t = v[used], np.searchsorted(used, t[keep])
            new = np.concatenate(added[n]) if added[n] else np.zeros((0, 3, 3))
            if fits(base_v, new):
                base_v, base_t = merged(base_v, base_t, new)
            else:
                c = new.mean(1)
                o = np.lexsort((c[:, 0], np.floor(c[:, 2] / 40.), np.floor(c[:, 0] / 40.)))
                new = new[o]
                blocks = np.floor(new.mean(1)[:, [0, 2]] / 40.).astype(np.int64)
                cut = np.r_[0, np.flatnonzero((np.diff(blocks[:, 0]) != 0) | (np.diff(blocks[:, 1]) != 0)) + 1, len(new)]
                spill += [(n, new[a:b]) for a, b in zip(cut[:-1], cut[1:])]
            cur[n] = [base_v, base_t]
            changed.add(n)
            mesh_report[n] = {'mesh': n, 'faces_replaced': int((~keep).sum()), 'new_faces': int(len(new))}

        def state(n):
            if cur[n] is None:
                _, v, t = decoded[n]
                used = np.unique(t)
                cur[n] = [v[used], np.searchsorted(used, t)]
            return cur[n]

        centre = {n: decoded[n][1][:, [0, 2]].astype(np.float64).mean(0) for n in names}
        hosted = {}
        for owner, block in spill:
            c = block.reshape(-1, 3)[:, [0, 2]].mean(0)
            for n in sorted(names, key=lambda k: (float(np.hypot(*(centre[k] - c))), k)):
                bv, bt = state(n)
                if fits(bv, block):
                    cur[n] = list(merged(bv, bt, block))
                    changed.add(n)
                    hosted.setdefault(n, []).append(owner)
                    break
            else:
                raise RuntimeError('no land mesh has room for refined faces near ' + owner)
        outputs = {}
        written_ = {n: mesh_write(meshes[n][0], *cur[n]) for n in sorted(changed)}
        # each compressed_map on its own (NumPy / ndimage release the GIL); the same calls, in order
        with ThreadPoolExecutor(MB.pool_size(CM_WORKER_GB, cap=8, env='TW3K_RBM_WORKERS')) as pool:
            cms = list(pool.map(lambda n: cm_update(meshes[n][1], written_[n][1], field, corner), sorted(changed)))
        for n, cm in zip(sorted(changed), cms):
            src, cmd = meshes[n]
            outputs[n] = (written_[n][0], cm, src, cmd)
            r = mesh_report.setdefault(n, {'mesh': n, 'faces_replaced': 0})
            r.update(vertices=int(len(cur[n][0])), faces=int(len(cur[n][1])), hosts_spill_of=sorted(set(hosted.get(n, []))))
        mesh_report = [mesh_report[n] for n in sorted(mesh_report)]
        added_meshes = []
        totals = {'land_meshes': len(names),
                  'vertices_before': int(sum(len(np.unique(decoded[n][2])) for n in names)),
                  'faces_before': int(sum(len(decoded[n][2]) for n in names)),
                  'vertices_after': int(sum(len(state(n)[0]) for n in names)),
                  'faces_after': int(sum(len(state(n)[1]) for n in names))}
        stats.update(totals)
        vr = {'variant': variant, 'restored_inputs': restored, 'footprint_corners': int(corner.sum()),
              'meshes_changed': sorted(k for k in outputs if k not in added_meshes), 'added_meshes': added_meshes,
              **stats, 'meshes': mesh_report}
        if apply:
            stale = [gdir / x for x in sorted(made)]
            stale += [gdir / x.replace('.rigid_model_v2', '.compressed_map') for x in sorted(made)]
            stale = [x for x in stale if x.exists()]
            if stale:
                dest = A / 'terrain-repair/river-bed-mesh-stale' / str(time.time_ns()) / pre / 'global_meshes'
                dest.mkdir(parents=True, exist_ok=True)
                for x in stale:
                    shutil.move(str(x), str(dest / x.name))
            for name, (data, cmd, src, srccm) in sorted(outputs.items()):
                if src is not None and name not in restored:
                    b = BACK / pre / 'global_meshes'
                    b.mkdir(parents=True, exist_ok=True)
                    P.pb.write_atomic(b / name, src)
                    P.pb.write_atomic(b / name.replace('.rigid_model_v2', '.compressed_map'), srccm)
                P.pb.write_atomic(gdir / name, data)
                P.pb.write_atomic(gdir / name.replace('.rigid_model_v2', '.compressed_map'), cmd)
            # inputs restored from the backup but not changed this time go back as they were
            for name in restored:
                if name not in outputs:
                    P.pb.write_atomic(gdir / name, meshes[name][0])
                    P.pb.write_atomic(gdir / name.replace('.rigid_model_v2', '.compressed_map'), meshes[name][1])
        vr['outputs'] = {k: [hashlib.sha256(d[0]).hexdigest(), hashlib.sha256(d[1]).hexdigest()] for k, d in sorted(outputs.items())}
        vr['inputs'] = {k: inputs[k] for k in sorted(outputs)}
        report['variants'].append(vr)
        print(variant, {k: v for k, v in vr.items() if k not in ('meshes', 'outputs', 'inputs', 'meshes_changed', 'restored_inputs')},
              'restored', len(restored), flush=True)
    if apply:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        tmp = REPORT.with_name(REPORT.name + '.tmp')
        tmp.write_text(json.dumps(report, indent=1), encoding='utf-8')
        tmp.replace(REPORT)
    return report


def cm_update(cmd, box, field, corner):
    """compressed_map of a changed mesh: the template's heights over the new box, the LF where the
    footprint (corner; None: everywhere) lies; pitch DX/3 at least, as add_island_ground_patches."""
    oldbox = struct.unpack_from('<6f', cmd, 26)
    a = cm_read(cmd).astype(np.float64)
    w_ = max(a.shape[1] if corner is not None else 16, math.ceil((box[3] - box[0]) / (cw.DX / 3) / 16) * 16)
    h_ = max(a.shape[0] if corner is not None else 16, math.ceil((box[5] - box[2]) / (cw.DX / 3) / 16) * 16)
    xx, zz = np.meshgrid(np.linspace(box[0], box[3], w_), np.linspace(box[2], box[5], h_))
    value = lf(field, xx, zz)
    if corner is None:
        height = value
    else:
        ax = np.clip((xx - oldbox[0]) / max(oldbox[3] - oldbox[0], 1e-9) * (a.shape[1] - 1), 0, a.shape[1] - 1)
        az = np.clip((zz - oldbox[2]) / max(oldbox[5] - oldbox[2], 1e-9) * (a.shape[0] - 1), 0, a.shape[0] - 1)
        height = oldbox[1] + ndimage.map_coordinates(a, [az, ax], order=1, mode='nearest') / 65535 * (oldbox[4] - oldbox[1])
        gx = np.clip(np.rint(xx / STEP).astype(int), 0, corner.shape[1] - 1)
        gz = np.clip(np.rint(zz / STEP).astype(int), 0, corner.shape[0] - 1)
        mask = corner[gz, gx]
        height[mask] = value[mask]
    lo = min(float(height.min()), box[1])
    hi = max(float(height.max()), box[4], lo + .001)
    values = np.rint((height - lo) / (hi - lo) * 65535).clip(0, 65535).astype('<u2')
    return bytes(cm_write(cmd, values, (box[0], lo, box[2], box[3], hi, box[5])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    run(ap.parse_args().apply)


if __name__ == '__main__':
    main()

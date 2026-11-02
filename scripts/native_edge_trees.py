"""Feather canvas tree density into vanilla's along the native rectangle edge (campaign tree list).

Vanilla trees are kept as they are inside the native map; the canvas is replanted by
polish_campaign_foliage's band rule and frontier_forest. Where vanilla forest meets the native
edge the two densities met in a step: at q=888, rows 520-630 (Ailao) ~15 canvas trees per hex
column against ~33-40 vanilla (stage tree list 2026-10-24, offline column count). feather() moves
the canvas density towards the vanilla density measured just inside the edge, over FEATHER hexes:
full vanilla density beside the edge, the canvas's own density FEATHER hexes out (smoothstep).

Per edge (west, east, south, north) and per position along it (row r or column q):
  native   trees per land hex in the strip 1..FEATHER hexes inside the edge, ALONG hexes either side
  far      canvas trees per land hex FEATHER+1..2*FEATHER hexes out (the canvas's own density)
  current  canvas trees per land hex at the same depth (+-1) and ALONG hexes either side
  target   far + (native - far) * smoothstep(1 - (depth - .5) / FEATHER)
Only where vanilla forest meets the edge (native >= FOREST_MIN). A canvas hex below its target
takes one tree with probability target - current (at most one per hex per pass); above it, its
canvas trees are thinned with probability 1 - target / current. Rocks and shrubs are not trees:
not counted, never added or dropped. Added trees take the species mix of the vanilla trees in the
strip opposite (their own rows as templates, ground height from the caller's sampler).
Deterministic and order-free (props bake, step 3): every draw is a stateless hash (tree_bake) keyed by the
hex and edge (adds), or by the row's stored float32 position (thinning), never by a draw count; added rows
ask accept(tx, ty, species) (dry footprint and the post-filter rules) before they are written.
Native rows (inside the rectangle) are never touched.
"""
import numpy as np

import campaign_world as cw
import tree_bake as tb

FEATHER = 10        # hexes from the native edge over which the canvas density reaches vanilla's
ALONG = 8           # hexes either side along the edge over which densities are averaged
FOREST_MIN = .15    # vanilla trees per land hex in the inside strip: below it no forest meets the edge
JITTER = .4         # hexes: an added tree's offset from its hex block centre
NOT_TREES = ('general_rock', 'rock_shrub')


def is_tree(name):
    return not name.startswith(NOT_TREES)


def _box1(a, r):
    """Moving sum over +-r along axis 0 (float64, edge-clipped, no wrap)."""
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0, dtype=np.float64)])
    n = a.shape[0]; i = np.arange(n)
    return c[np.minimum(i + r + 1, n)] - c[np.maximum(i - r, 0)]


def _edges():
    """(name, along-axis, inside depth -> index, outside depth -> index) of the four edges.
    Index functions map (depth, position) to (r, q) hex arrays; depth 1 is the hex next to the edge."""
    q0, q1 = cw.PAD_LEFT, cw.PAD_LEFT + cw.NATIVE_W      # native columns q0..q1-1
    r0, r1 = cw.PAD_BOTTOM, cw.PAD_BOTTOM + cw.NATIVE_H   # native rows r0..r1-1
    rows, cols = np.arange(r0, r1), np.arange(q0, q1)
    return (('west', rows, lambda d, p: (p, q0 - 1 + d), lambda d, p: (p, q0 - d)),
            ('east', rows, lambda d, p: (p, q1 - d), lambda d, p: (p, q1 - 1 + d)),
            ('south', cols, lambda d, p: (r0 - 1 + d, p), lambda d, p: (r0 - d, p)),
            ('north', cols, lambda d, p: (r1 - d, p), lambda d, p: (r1 - 1 + d, p)))


def feather(groups, land, sample_surface, ok_hex, towns_shown, town_clear, road_cover=None, accept=None):
    """Edit groups in place: list of {'name', 'rows' (bytes rows), 'added'}.

    land: hex bool (H, W); sample_surface(tx, tz) -> ground height (tile coords, arrays);
    ok_hex(q, r) -> bool array: the hexes the foliage step may replant (alpha, patch rules);
    towns_shown: (n, 2) towns as drawn, x/DX, z/DZ units; town_clear: hexes kept clear round them;
    road_cover: tile bool grid or None (road tiles stay clear); accept(tx, ty, species) -> bool array
    (tree_bake.TreeAccept.accept, required). Returns a report dict."""
    assert accept is not None, 'feather needs the props bake accept predicate'
    H, W = land.shape
    DX, DZ = cw.DX, cw.DZ
    # Tree entries per hex, canvas and native, and native rows per species and hex (templates).
    canvas_n = np.zeros((H, W), np.int32); native_n = np.zeros((H, W), np.int32)
    where = []   # per group: (q, r, native) of its rows
    for g in groups:
        if not g['rows']:
            where.append(None); continue
        p = np.frombuffer(b''.join(bytes(r_[:12]) for r_ in g['rows']), '<f4').reshape(-1, 3).astype(np.float64)
        q, r = cw.world_hex(p[:, 0], p[:, 2]); q, r = np.asarray(q), np.asarray(r)
        nat = (cw.PAD_LEFT <= q) & (q < cw.PAD_LEFT + cw.NATIVE_W) & (cw.PAD_BOTTOM <= r) & (r < cw.PAD_BOTTOM + cw.NATIVE_H)
        where.append((q, r, nat))
        if is_tree(g['name']):
            np.add.at(native_n, (r[nat], q[nat]), 1); np.add.at(canvas_n, (r[~nat], q[~nat]), 1)
    landf = land.astype(np.float64)
    towns = np.asarray(towns_shown, np.float64)
    report = {'added': 0, 'dropped': 0, 'rejected': 0, 'edges': {}}
    drop_hex = {}   # (r, q) -> drop probability for canvas trees
    add_q, add_r, add_species = [], [], []
    tree_groups = [i for i, g in enumerate(groups) if is_tree(g['name']) and g['rows']]
    # Vanilla tree rows per hex of the inside strips (templates for added trees), taken before thinning.
    strip = np.zeros((H, W), bool)
    for _, pos, inside, _o in _edges():
        dd, pp = np.meshgrid(np.arange(1, FEATHER + 1), pos, indexing='ij'); sr, sq = inside(dd, pp); strip[sr, sq] = True
    native_rows = {}
    for gi in tree_groups:
        q, r, nat = where[gi]
        for j in np.flatnonzero(nat & strip[r, q]):
            native_rows.setdefault((int(r[j]), int(q[j])), []).append((gi, groups[gi]['rows'][j]))
    for ei, (name, pos, inside, outside) in enumerate(_edges()):
        P = len(pos); D = np.arange(1, 2 * FEATHER + 1)
        dd, pp = np.meshgrid(D, pos, indexing='ij')   # (depth, position)
        ir, iq = inside(np.minimum(dd, FEATHER), pp); orr, oq = outside(dd, pp)
        okin = (orr >= 0) & (orr < H) & (oq >= 0) & (oq < W)
        orr, oq = np.clip(orr, 0, H - 1), np.clip(oq, 0, W - 1)
        # Inside strip (depth 1..FEATHER): vanilla density per position.
        nat_t = _box1(native_n[ir[:FEATHER], iq[:FEATHER]].sum(0).astype(np.float64), ALONG)
        nat_l = _box1(landf[ir[:FEATHER], iq[:FEATHER]].sum(0), ALONG)
        native = np.where(nat_l > 0, nat_t / np.maximum(nat_l, 1), 0)
        # Outside: canvas density per (depth, position).
        c_t = np.where(okin, canvas_n[orr, oq], 0).astype(np.float64); c_l = np.where(okin, landf[orr, oq], 0)
        far = _box1(c_t[FEATHER:].sum(0), ALONG) / np.maximum(_box1(c_l[FEATHER:].sum(0), ALONG), 1)
        # Current density at each depth: +-1 depth, +-ALONG positions.
        ct = _box1(_box1(c_t, 1).T, ALONG).T; cl = _box1(_box1(c_l, 1).T, ALONG).T
        current = ct / np.maximum(cl, 1)
        t = np.clip(1 - (D[:FEATHER, None] - .5) / FEATHER, 0, 1); w = t * t * (3 - 2 * t)
        target = far + (native - far) * w
        forest = native >= FOREST_MIN
        live = forest[None, :] & okin[:FEATHER] & land[orr[:FEATHER], oq[:FEATHER]]
        cur = current[:FEATHER]
        e_add = e_drop = 0
        di_, pi_ = np.nonzero(live)
        u_add = tb.u01('edge.add', oq[di_, pi_], orr[di_, pi_], ei)   # keyed by the hex and edge
        # Species mix of the vanilla trees opposite (per position, the strip +-ALONG): built lazily.
        for k, (di, pi) in enumerate(zip(di_.tolist(), pi_.tolist())):
            rr_, qq_ = int(orr[di, pi]), int(oq[di, pi])
            tg, cu = float(target[di, pi]), float(cur[di, pi])
            if tg > cu:
                if u_add[k] < min(1., tg - cu):
                    add_q.append(qq_); add_r.append(rr_); add_species.append((name, int(pi))); e_add += 1
            elif cu > tg and canvas_n[rr_, qq_] > 0:
                drop_hex[(rr_, qq_)] = max(drop_hex.get((rr_, qq_), 0.), 1 - tg / cu); e_drop += 1
        report['edges'][name] = {'forest_positions': int(forest.sum()), 'add_draws': e_add, 'thin_hexes': e_drop}
    # Thin: canvas tree rows on hexes with a drop probability (draw keyed by the row's stored position, so the
    # outcome of a row does not depend on its place in the list or on the rows before it).
    if drop_hex:
        for gi in tree_groups:
            q, r, nat = where[gi]; g = groups[gi]
            keep = np.ones(len(g['rows']), bool)
            kx, kz = tb.row_keys(g['rows'])
            u_thin = tb.u01('edge.thin', kx, kz)
            for i in np.flatnonzero(~nat):
                pr = drop_hex.get((int(r[i]), int(q[i])))
                if pr is not None and u_thin[i] < pr:
                    keep[i] = False
            n = int((~keep).sum())
            if n:
                g['rows'] = [row for row, k in zip(g['rows'], keep) if k]; g['added'] = max(0, g['added'] - n); report['dropped'] += n
    # Add: one tree per drawn hex, species from the vanilla strip opposite, a vanilla row of it as template.
    if add_q:
        q = np.array(add_q, np.int64); r = np.array(add_r, np.int64)
        ok = ok_hex(q, r) & land[r, q]
        zr = r + q % 2 * .5
        td = np.hypot(towns[None, :, 0] - q[:, None], towns[None, :, 1] - zr[:, None]).min(1) if len(towns) else np.full(len(q), np.inf)
        ok &= td >= town_clear
        tx = 2 * q + 1 + 2 * tb.uniform(-JITTER, JITTER, 'edge.jx', q, r); ty = 2 * r + q % 2 + 1 + 2 * tb.uniform(-JITTER, JITTER, 'edge.jy', q, r)
        hq, hr = cw.tile_hex(tx.astype(np.int64), ty.astype(np.int64)); hq, hr = np.asarray(hq), np.asarray(hr)
        ok &= land[hr, hq] & (hq == q) & (hr == r)
        if road_cover is not None:
            ok &= ~road_cover[np.clip(ty.astype(np.int64), 0, road_cover.shape[0] - 1), np.clip(tx.astype(np.int64), 0, road_cover.shape[1] - 1)]
        y = np.asarray(sample_surface(tx, ty), np.float64)
        # Species pools: per (edge, position) the native tree rows of the strip opposite.
        edges = {e[0]: e for e in _edges()}
        pools = {}
        u_pool = tb.u01('edge.pool', q, r)
        picked = []   # (i, group index, template row): species first, then one accept call for all
        for i in np.flatnonzero(ok):
            key = add_species[i]
            if key not in pools:
                ename, pos, inside, _ = edges[key[0]]
                lo, hi = max(0, key[1] - ALONG), min(len(pos), key[1] + ALONG + 1)
                dd, pp = np.meshgrid(np.arange(1, FEATHER + 1), pos[lo:hi], indexing='ij')
                sr, sq = inside(dd, pp)
                pools[key] = [e for cell in sorted(set(zip(np.ravel(sr).tolist(), np.ravel(sq).tolist()))) for e in native_rows.get(cell, ())]
            pool = pools[key]
            if not pool:
                continue
            gi, src = pool[min(int(u_pool[i] * len(pool)), len(pool) - 1)]
            picked.append((i, gi, src))
        if picked:
            ii = np.array([p_[0] for p_ in picked], np.int64)
            fine = accept(tx[ii], ty[ii], [groups[p_[1]]['name'] for p_ in picked])
            report['rejected'] = int((~fine).sum())
            for (i, gi, src), f in zip(picked, fine):
                if not f:
                    continue
                g = groups[gi]; row = bytearray(src)
                row[0:12] = np.array([tx[i] * DX / 2, y[i], ty[i] * DZ / 2], '<f4').tobytes()
                g['rows'].append(bytes(row)); g['added'] += 1; report['added'] += 1
    return report

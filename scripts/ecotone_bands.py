"""Ecotones the USR3 ECOTONE block does not cover (polish_campaign_terrain.py, MATERIALS
ECOTONE block). Pure numpy (no scipy): runs in the pipeline Python.

RESOLVE ecoregions cut alpine meadow against conifer/taiga, steppe, snow and desert along
straight vector outlines; the classes pick different donors (alpine/snow/desert: arid sand,
conifer/taiga: cold forest, steppe: steppe), so the outlines showed as straight-edged
polygons of dark conifer against pale alpine (Altai, hex 652,1272; live/offline 2026-10-23).
The USR3 block handles open ground (steppe, grass, savanna and now tundra) against forest
with its 'forest on ridges' relief rule; that rule is backwards on a mountain treeline.

  treeline()      alpine <-> forest (taiga, conifer, forest, temperate): real elevation decides,
                  alpine above and forest below a treeline that is half the latitude line
                  TREELINE and half the local boundary elevation (so the band stays where
                  RESOLVE drew it, but follows the contours), +- TREE_FUZZ m of fbm, faded
                  into the original classes at the band edge (band ~ ECOTONE cells).
  alpine_dither() alpine <-> steppe / snow / desert / grass: argmax of each class's blurred
                  share plus fbm (as polish's own climate score), only near alpine.
  tundra_forest() tundra <-> forest (taiga, conifer, forest, temperate): the RESOLVE outline domain-warped on
                  flat ground, then a mosaic of the blurred tundra share + fbm, uplands to tundra and river
                  corridors to forest (larch tongues): [9.3] the straight arctic treeline.
Noise and box blurs run on gpu_noise (same bytes as terrain_noise / fastops, CPU fallback).
"""
import numpy as np

from gpu_noise import fbm, box_blur as _gpu_box
import biome_profiles as _bp

_K = _bp.consts('ecotone_bands')   # values: biome_profiles.CONSTS['ecotone_bands']

# Treeline (m) by latitude: Himalaya ~3900, Pamir/Kunlun 3600-3400, Tian Shan ~2900,
# Altai 2300-2100, Sayan ~1900, Stanovoy/Verkhoyansk 1300-1000.
TREELINE = _bp.cold_line('TREELINE')
TREE_LOCAL = _K['TREE_LOCAL']        # share of the local boundary elevation in the treeline
TREE_FUZZ = _K['TREE_FUZZ']       # m of fbm on the treeline
TREE_SIGMA = _K['TREE_SIGMA']       # m: soft step around the treeline
FOREST = _bp.biome_set('treeline_forest')    # taiga, conifer, forest, temperate (no tropics: polish _efor has them)
ALPINE_PEERS = _bp.biome_set('alpine_peer')   # steppe, snow, desert, grass (order: noise seeds 150 + k)
DITHER_AMP = _K['DITHER_AMP']       # fbm of ECOTONE_GRAIN hexes: meander of the boundary
DITHER_FINE = _K['DITHER_FINE']      # fbm of a third of that: interleaved tongues and islands (blur + coarse noise alone smoothed
                       # the RESOLVE outline into a cleaner but still single line: mixed band 2098 -> 1105 cells)


def _box(a, radius):
    return _gpu_box(np.asarray(a, np.float32), radius)   # native_campaign_climate.box_blur, same bytes


def treeline_m(lat):
    return _bp.treeline_m(lat)   # rule predicate (COLD_LINES TREELINE)


def _dominant(eb, ids, radius):
    best = np.full(eb.shape, -1., np.float32); cls = np.zeros(eb.shape, eb.dtype)
    for i in ids:
        sh = _box(eb == i, radius); up = sh > best; best[up] = sh[up]; cls[up] = i
    return cls


def treeline(eb, bnames, elev, lat, q, r, origin, band, grain):
    """Redraw alpine/forest cells near their boundary by elevation. Returns (eb, flipped)."""
    ix = lambda *ns: [bnames.index(n) for n in ns if n in bnames]
    al = ix('alpine'); fo = ix(*FOREST)
    if not al or not fo:
        return eb, 0
    A = np.isin(eb, al); F = np.isin(eb, fo)
    if not A.any() or not F.any():
        return eb, 0
    shA = _box(_box(A, band), band); shF = _box(_box(F, band), band)
    near = (A | F) & (shA > .02) & (shF > .02)
    if not near.any():
        return eb, 0
    e = _box(elev, 2)
    wgt = (shA * shF * (A | F)).astype(np.float32)
    wb = _box(_box(wgt, 2 * band), 2 * band)
    tloc = _box(_box(e * wgt, 2 * band), 2 * band) / np.maximum(wb, 1e-6)
    tl = treeline_m(lat)
    T = np.where(wb > 1e-4, TREE_LOCAL * tloc + (1 - TREE_LOCAL) * tl, tl)
    z = (e - T + TREE_FUZZ * fbm(q, r, origin, grain, 3, seed=141)) / TREE_SIGMA
    above = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    t = np.clip(np.minimum(shA, shF) / .25, 0, 1); t = t * t * (3 - 2 * t)
    p_alp = t * above + (1 - t) * A
    u = .5 + .5 * fbm(q, r, origin, grain / 2, 3, seed=142)
    new_alp = near & (u < p_alp); new_for = near & ~new_alp
    out = eb.copy()
    to_alp = new_alp & F; to_for = new_for & A
    if to_alp.any():
        out[to_alp] = _dominant(eb, al, 2 * band)[to_alp]
    if to_for.any():
        out[to_for] = _dominant(eb, fo, 2 * band)[to_for]
    return out, int(to_alp.sum() + to_for.sum())


def alpine_dither(eb, bnames, q, r, origin, band, grain):
    """alpine <-> steppe/snow/desert/grass: blurred share + fbm argmax near alpine."""
    ix = lambda *ns: [bnames.index(n) for n in ns if n in bnames]
    al = ix('alpine'); ids = al + ix(*ALPINE_PEERS)
    if not al:
        return eb, 0
    inset = np.isin(eb, ids)
    shA = _box(np.isin(eb, al), band)
    zone = inset & (shA > .02) & (shA < .98)
    if not zone.any():
        return eb, 0
    # Only zone cells change, and the twice-blurred share there reads cells within 2*(band//2):
    # shares and noise are computed in the zone's bounding box grown by 2*(band//2)+2 (same values:
    # fbm is per pixel, and the blurs of a 0/1 mask are exact -- every float32 the four passes add
    # into their float64 prefix sums is 0 or in [1/k**3, 1] (k = 2*(band//2)+1), a multiple of
    # 2**-24/k**3, so no prefix sum of a line rounds while (line + k) * k**3 <= 2**29; else the grid).
    R = band // 2; kw = 2 * R + 1; m = 2 * R + 2
    if (max(eb.shape) + kw) * kw ** 3 <= 2 ** 29:
        rr = np.flatnonzero(zone.any(axis=1)); cc = np.flatnonzero(zone.any(axis=0))
        win = (slice(max(0, rr[0] - m), min(eb.shape[0], rr[-1] + m + 1)), slice(max(0, cc[0] - m), min(eb.shape[1], cc[-1] + m + 1)))
    else:
        win = (slice(None), slice(None))
    ew, zw, qw, rw = eb[win], zone[win], q[win], r[win]
    best = np.full(ew.shape, -9., np.float32); cls = ew.copy()
    for k, i in enumerate(ids):
        sh = _box(_box(ew == i, R), R)
        if not sh[zw].any():
            continue
        s = sh + (DITHER_AMP * fbm(qw, rw, origin, grain, 3, seed=150 + k) + DITHER_FINE * fbm(qw, rw, origin, grain / 3, 2, seed=160 + k)) * (sh > 0)
        up = s > best; best[up] = s[up]; cls[up] = i
    out = eb.copy(); ch = zw & (cls != ew)
    out[win][ch] = cls[ch]
    return out, int(ch.sum())



TUNDRA_FOREST = _K['TUNDRA_FOREST']


def tundra_forest(eb, bnames, elev, river, q, r, origin):
    """Redraw tundra/forest cells near their boundary (CONSTS ecotone_bands TUNDRA_FOREST). elev: real metres,
    river: real river mask (tile grid). Returns (eb, flipped)."""
    k = TUNDRA_FOREST
    ix = lambda *ns: [bnames.index(n) for n in ns if n in bnames]
    tu = ix('tundra'); fo = ix(*FOREST)
    if not tu or not fo:
        return eb, 0
    TF = np.isin(eb, tu + fo)
    if not (np.isin(eb, tu).any() and np.isin(eb, fo).any()):
        return eb, 0
    rr = np.flatnonzero(TF.any(axis=1)); cc = np.flatnonzero(TF.any(axis=0)); m = 4 * k['BAND'] + 8
    win = (slice(max(0, rr[0] - m), min(eb.shape[0], rr[-1] + m + 1)), slice(max(0, cc[0] - m), min(eb.shape[1], cc[-1] + m + 1)))
    ew, el, rv, qw, rw = eb[win], np.asarray(elev, np.float32)[win], np.asarray(river, np.float32)[win], q[win], r[win]
    tfw = TF[win]; isT = np.isin(ew, tu)
    # 1. domain warp of the outline on flat ground (only tundra <-> forest swaps)
    e1 = _box(el, 8); rel = np.sqrt(np.maximum(_box(el * el, 8) - e1 * e1, 0))
    flat = _box(rel, 8) < .5 * (k['REL'][0] + k['REL'][1]); del e1, rel
    dx = k['WARP'] * fbm(qw, rw, origin, k['WGRAIN'], 3, seed=163)
    dy = k['WARP'] * fbm(qw, rw, origin, k['WGRAIN'], 3, seed=164)
    yy, xx = np.mgrid[:ew.shape[0], :ew.shape[1]]
    wv = ew[np.clip(np.rint(yy + dy), 0, ew.shape[0] - 1).astype(np.int32), np.clip(np.rint(xx + dx), 0, ew.shape[1] - 1).astype(np.int32)]
    del yy, xx, dx, dy
    sw = tfw & np.isin(wv, tu + fo) & (isT != np.isin(wv, tu)) & flat
    w1 = np.where(sw, wv, ew); del wv, sw, flat
    # 2. mosaic near the warped boundary
    T = np.isin(w1, tu); F = np.isin(w1, fo); B = k['BAND']
    sh = _box(_box(T, B), B); shF = _box(_box(F, B), B)
    near = (T | F) & (sh > .02) & (shF > .02)
    fr = sh / np.maximum(sh + shF, 1e-6); del sh, shF
    up = np.clip((_box(el, 2) - _box(_box(el, 24), 24)) / 150, -1, 1)
    cor = np.clip(_box(_box(rv, k['RIVER_R']), k['RIVER_R']) * 6, 0, 1)
    s = (fr + k['NOISE'] * fbm(qw, rw, origin, k['GRAIN'], 3, seed=161) + k['FINE'] * fbm(qw, rw, origin, k['GRAIN'] / 4, 3, seed=162)
         + k['RELIEF'] * up - k['RIVER'] * cor * np.clip((.85 - fr) / .25, 0, 1))
    out_w = w1.copy()
    out_w[near & F & (s > .5)] = tu[0]
    toF = near & T & (s <= .5)
    if toF.any():
        out_w[toF] = _dominant(w1, fo, 2 * B)[toF]
    out = eb.copy(); out[win] = out_w
    return out, int((out_w != ew).sum())


def straight_share(eb, bnames):
    """Share of half-overlapping windows on the tundra/forest boundary whose large-scale contour (the tundra share among
    tundra + forest, box-smoothed twice, > .5) lies on one straight line (PCA minor std below tol cells); TUNDRA_FOREST
    STRAIGHT = (window, smooth, tol, min boundary cells). The straight RESOLVE arctic treeline: canvas main 2026-10-30
    .251 of 275 windows. Returns (share, windows, window centres that are straight)."""
    blk, smooth, tol, min_px = TUNDRA_FOREST['STRAIGHT']
    ix = lambda *ns: [bnames.index(n) for n in ns if n in bnames]
    T = np.isin(eb, ix('tundra')); TF = T | np.isin(eb, ix(*FOREST))
    s = _box(_box(T, smooth), smooth) / np.maximum(_box(_box(TF, smooth), smooth), 1e-6)
    T = TF & (s > .5); F = TF & ~T
    bd = T & (np.roll(F, 1, 0) | np.roll(F, -1, 0) | np.roll(F, 1, 1) | np.roll(F, -1, 1))
    n, hits = 0, []
    for by in range(0, bd.shape[0] - blk + 1, blk // 2):
        for bx in range(0, bd.shape[1] - blk + 1, blk // 2):
            yy, xx = np.nonzero(bd[by:by + blk, bx:bx + blk])
            if len(yy) < min_px:
                continue
            n += 1
            if np.sqrt(max(np.linalg.eigvalsh(np.cov(np.stack([xx, yy]).astype(np.float64)))[0], 0.)) < tol:
                hits.append((bx + blk // 2, by + blk // 2))
    return len(hits) / max(n, 1), n, hits


def pair_counts(eb, bnames, pairs):
    """Adjacent-cell pairs (4-neighbour) per class pair in a window."""
    res = {}
    for a, b in pairs:
        ia, ib = bnames.index(a), bnames.index(b); n = 0
        for x, y in ((eb[:, :-1], eb[:, 1:]), (eb[:-1, :], eb[1:, :])):
            n += int(((x == ia) & (y == ib)).sum() + ((x == ib) & (y == ia)).sum())
        res[f'{a}-{b}'] = n
    return res

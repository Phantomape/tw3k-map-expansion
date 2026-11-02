"""Desert-ground rules of polish_campaign_terrain.py (real-terrain canvas ground): aridity of high
ground, the tonal-buffer gate, real oases as seeds of the polish oasis belts, and the subtropical donor
split. Pure numpy (no scipy): runs in the pipeline Python. Noise on gpu_noise (CPU fallback, same bytes).
"""
import numpy as np

import biome_profiles as _bp
import gpu_noise

_K = _bp.consts('oasis_ground')   # values and survey notes: biome_profiles.CONSTS['oasis_ground']


def _box(a, radius):
    from native_campaign_climate import box_blur
    return box_blur(np.asarray(a, np.float32), radius)


def _smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


def _ix(bnames, *ns):
    return [bnames.index(n) for n in ns if n in bnames]


# ------------------------------------------------------------------ aridity
# Dry high ground: RESOLVE has one 'alpine' class for every montane grassland, the wet Ili/Yulduz
# meadows and the bare south slopes of the Tian Shan (Bogda above Turpan, Kucha, Aksu), the Kunlun
# and the western Qilian alike. The desert share box-blurred twice over DRY_R tile cells (~35 km
# reach) separates them (offline survey 2026-10-24 on alpine cells): Ili/Tekes/Nalati, Yulduz,
# Altai .00, Qilian above Zhangye .09, Shihezi/Urumqi south .11-.14 | Bogda north (Tianchi) .22 |
# Aksu north, Kunlun above Khotan .28, Kucha north .38, Bogda south .45, Kunlun above Niya .49,
# western Qilian above Dunhuang .75. DRY maps that share to 0 -> 1 (smoothstep).
DRY_R = _K['DRY_R']
DRY = _K['DRY']
ARID_R = _K['ARID_R']   # tile cells: box radius (twice) of the arid share (polish's tonal buffer)
ARID_ALPINE = _bp.biome_set('arid_alpine')   # counted arid times dryness (with desert)
DRY_ASPECT = _K['DRY_ASPECT']   # (gain, m per tile cell, box radius): sunny flanks drier (biome_profiles.dryness)
CONIFER_FLOOR, CONIFER_FLOOR_DRY = _K['CONIFER_FLOOR'], _K['CONIFER_FLOOR_DRY']   # m, dryness ramps
CONIFER_FLOOR_FUZZ, CONIFER_FLOOR_OPEN = _K['CONIFER_FLOOR_FUZZ'], _K['CONIFER_FLOOR_OPEN']
FLOOR_FOREST = _bp.biome_set('treeline_forest')   # taiga, conifer, forest, temperate


# dryness(real_bio, bnames, real_elev=None): 0 (humid) .. 1 (dry) per tile, smoothstep of the twice box-blurred RESOLVE
# desert share. arid_share(real_bio, bnames, dry=None): blurred share of arid ground per tile, desert and
# alpine/snow times dryness, box radius ARID_R twice. polish's tonal buffer (forest -> open ground
# shading) is multiplied by 1 - arid_share: it was made for humid steppe and forest edges, and on arid
# ground swapped the sand for green steppe_3/4 (Jushi/Turpan window: 48 % of its green; real biome
# desert/alpine). Both are classifier variants of biome_profiles (the same code, these constants).
dryness = _bp.dryness
arid_share = _bp.arid_share


# ------------------------------------------------------------------ lower forest line of dry ranges
def conifer_floor(eb, bnames, real_bio, real_elev, q, r, origin):
    """(classes, cells changed): forest classes (FLOOR_FOREST) of eb below the dry ranges' lower forest line
    (CONIFER_FLOOR m, fbm fuzz) where dryness (of real_bio, with aspect) is above CONIFER_FLOOR_DRY take the
    dominant CONIFER_FLOOR_OPEN class around them (box radius DRY_R; ties to the first in that order)."""
    forest = np.isin(eb, _ix(bnames, *FLOOR_FOREST))
    if not forest.any():
        return eb, 0
    dry = dryness(real_bio, bnames, real_elev)
    e = _box(real_elev, 1) + CONIFER_FLOOR_FUZZ * gpu_noise.fbm(q, r, origin, 6, 3, seed=251)
    p = _smooth((CONIFER_FLOOR[1] - e) / (CONIFER_FLOOR[1] - CONIFER_FLOOR[0])) * _smooth((dry - CONIFER_FLOOR_DRY[0]) / (CONIFER_FLOOR_DRY[1] - CONIFER_FLOOR_DRY[0]))
    flip = forest & (p > .5)
    if not flip.any():
        return eb, 0
    best = np.full(eb.shape, -1., np.float32); cls = np.zeros(eb.shape, eb.dtype)
    for i in _ix(bnames, *CONIFER_FLOOR_OPEN):
        sh = _box(eb == i, DRY_R); up = sh > best; best[up] = sh[up]; cls[up] = i
    out = eb.copy(); out[flip] = cls[flip]
    return out, int(flip.sum())


# ------------------------------------------------------------------ real oases as seeds
# polish's oasis belts follow real rivers only (Tarim etc.), so the oases of the Turpan basin, the
# west end of the Hexi corridor and the Tarim rim stood as bare sand. Each real oasis seeds the polish
# oasis array (temperate donor + OASIS_REMAP; ground_materials and its DESERT_EXTRA pass exclude it
# through gw * (1 - 4 oasis); farm_mud_fields lays oasis fields in it): a noisy falloff from the seed,
# held to the low, gentle ground the irrigation water reaches (no climb above the seed, no steep
# slopes), stretched along Natural Earth rivers.
# (lon, lat, km^2). Coordinates: the town or ruin; centres shared with ground_materials.OASES are
# taken from it. Areas: Hanshu 96 (Xiyu zhuan) population x 2 ha per person (irrigated fields,
# orchards, fallow and settlement), floor 50 km^2 -- Kucha (Qiuci) 81,317, Yanqi 32,100, Aksu (Gumo)
# 24,500, Khotan (Yutian) 19,300, Kashgar (Shule) 18,647, Yarkand (Shache) 16,373, Charklik/Loulan
# (Shanshan) 14,100, Korla (Weili) 9,600, Niya (Jingjue) 3,360, Cherchen (Qiemo) 1,610; the Turpan
# basin (Jushi front 6,050 plus the Han garrison fields of Gaochang and Liuzhong), Yiwu (Hami) and the
# Dunhuang commandery are estimated at garrison-field scale. Wuwei, Zhangye and Jiuquan lie in the
# native map (real ground weight 0) and take no seed.
OASIS_SEEDS = _bp.oasis_seeds()   # biome_profiles.REGIONS['oasis_seeds'] (sources and areas there)
SEED_CORE, SEED_EDGE = _K['SEED_CORE'], _K['SEED_EDGE']   # x R = sqrt(km^2 / pi): weight 1 inside SEED_CORE R, 0 from SEED_EDGE R (.5 at R)
SEED_WARP, SEED_GRAIN = _K['SEED_WARP'], _K['SEED_GRAIN']    # fbm warp of the distance (no circles); hexes
SEED_RIVER = _K['SEED_RIVER']                    # reach x (1 + this x river share): oases run along their rivers
SEED_RISE = _K['SEED_RISE']           # m above the seed ground: full -> none (the water runs downhill)
SEED_SLOPE = _K['SEED_SLOPE']           # m per tile cell of real elevation (box 1): full -> none


def _window(clo, cla, S, shape, lo0, lo1, la0, la1):
    """Tile slice covering the lon/lat box (coarse grid clo/cla every S cells), or None."""
    ci, cj = np.nonzero((cla > la0) & (cla < la1) & (clo > lo0) & (clo < lo1))
    if not len(ci):
        return None
    r0, r1 = max(0, (ci.min() - 1) * S), min(shape[0], (ci.max() + 2) * S)
    c0, c1 = max(0, (cj.min() - 1) * S), min(shape[1], (cj.max() + 2) * S)
    return (slice(r0, r1), slice(c0, c1))


# ------------------------------------------------------------------ Turpan basin: Han oases
# [TURPAN3 2026-10-25] The karez are mostly post-Han (their wide spread is Qing), so the Turpan basin takes
# no karez band: Han farmland was ~120-200 km^2 (Jushi front 6,050 people, the tuntian of Gaochang bi and
# Liuzhong). biome_profiles.REGIONS['turpan_han_oasis']: solid patches on the Han water sources (gorge
# outflows of the Flaming Mountains and the Yarnaz) and the spring line along the mountains' south foot, a
# chain of small overlapping discs. Weight 1 within core x R, smoothstep to 0 at edge x R (.5 at R); the
# fbm warp moves only the outline (never inside the core). Rise/slope limits as the seeds. Stretches of
# the spring line narrower than 1 hex after the limits are left out (opening with a 2x2-tile square), and
# clean_islands removes small separate oasis islands in the basin box.
HAN = _bp.region('turpan_han_oasis')


def _hex_km(lon, lat, k):
    """km per hex (2 tile cells along x) at tile index k."""
    kk = min(k[1] + 1, lon.shape[1] - 1)
    kx = 111.32 * np.cos(np.radians(float(lat[k])))
    return 2 * float(np.hypot((lon[k[0], kk] - lon[k]) * kx, (lat[k[0], kk] - lat[k]) * 110.6)) / max(kk - k[1], 1)


def _disc(lon, lat, e, lim_s, wn, lo, la, R, core, edge):
    """Solid disc weight around (lo, la), radius R km (.5 at R); None when the centre is off this grid."""
    d = np.hypot((lon - lo) * 111.32 * np.cos(np.radians(la)), (lat - la) * 110.6)
    k = np.unravel_index(np.argmin(d), d.shape)
    if d[k] > 3.:
        return None
    t = (d - core * R) / ((edge - core) * R) + HAN['warp'] * wn   # the warp moves the outline only
    w = np.where(d <= core * R, 1., _smooth(1 - t))
    w = w * _smooth((SEED_RISE[1] - (e - float(e[k]))) / (SEED_RISE[1] - SEED_RISE[0])) * lim_s
    return w.astype(np.float32)


def _open2(m):
    """Morphological opening of a boolean tile mask with a 2x2 square (1 hex); no wrap-around."""
    er = np.zeros_like(m); er[:-1, :-1] = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
    op = er.copy(); op[1:, :] |= er[:-1, :]; op[:, 1:] |= er[:, :-1]; op[1:, 1:] |= er[:-1, :-1]
    return op


def _grow1(m):
    g = m.copy(); g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
    return g


def han_oases(e, lim_s, lon, lat, q, r, origin, clo, cla, S):
    """(oasis weight of the Turpan Han oases: patches + spring line, {source: tiles at weight >= .5})."""
    out = np.zeros(lon.shape, np.float32); rep = {}
    lo0, lo1, la0, la1 = HAN['box']
    sl = _window(clo, cla, S, lon.shape, lo0, lo1, la0, la1)
    if sl is None:
        return out, rep
    L, A, E, LS = lon[sl], lat[sl], e[sl], lim_s[sl]
    wn = gpu_noise.fbm(q[sl], r[sl], origin, HAN['grain_hex'], 3, seed=253)
    core, edge = HAN['core'], HAN['edge']
    for name in sorted(HAN['patches']):
        p = HAN['patches'][name]; lo, la = p['lonlat']
        w = _disc(L, A, E, LS, wn, lo, la, float(np.sqrt(p['km2'] / np.pi)), core, edge)
        if w is not None:
            out[sl] = np.maximum(out[sl], w); rep[name] = int((w >= .5).sum())
    for name in sorted(HAN['spring_lines']):
        s = HAN['spring_lines'][name]; pts = [tuple(p) for p in s['points']]
        k0 = np.unravel_index(np.argmin(np.hypot(L - pts[0][0], A - pts[0][1])), L.shape); hk = _hex_km(L, A, k0)
        R, step = s['radius_hex'] * hk, s['step_hex'] * hk
        centres = []
        for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
            n = max(int(np.ceil(np.hypot((xb - xa) * 111.32 * np.cos(np.radians(ya)), (yb - ya) * 110.6) / step)), 1)
            centres += [(xa + i / n * (xb - xa), ya + i / n * (yb - ya)) for i in range(n)]
        centres.append(pts[-1])
        line = np.zeros(L.shape, np.float32)
        for lo, la in centres:
            w = _disc(L, A, E, LS, wn, lo, la, R, core, edge)
            if w is not None:
                line = np.maximum(line, w)
        keep = _grow1(_open2(line >= .5))   # stretches narrower than 1 hex left out (their fringe with them)
        line = np.where(keep, line, 0).astype(np.float32)
        out[sl] = np.maximum(out[sl], line); rep[name] = int((line >= .5).sum())
    return out, rep


def _components(m):
    """8-connected components of a boolean mask, labelled in row-major scan order (deterministic)."""
    lab = np.zeros(m.shape, np.int32); n = 0; H, W = m.shape
    for i0, j0 in zip(*np.nonzero(m)):
        if lab[i0, j0]:
            continue
        n += 1; lab[i0, j0] = n; stack = [(int(i0), int(j0))]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < H and 0 <= b < W and m[a, b] and not lab[a, b]:
                        lab[a, b] = n; stack.append((a, b))
    return lab, n


def clean_islands(oasis, lon, lat):
    """Turpan basin box: oasis support (> HAN['support']) components holding no core (weight >= .5) of at
    least island_min_hex hexes (4 tile cells a hex) or with a source tile are set to 0 (stray specks).
    Returns (oasis, components removed)."""
    lo0, lo1, la0, la1 = HAN['box']
    S = 16; sl = _window(lon[::S, ::S], lat[::S, ::S], S, lon.shape, lo0, lo1, la0, la1)
    if sl is None:
        return oasis, 0
    L, A = lon[sl], lat[sl]; inb = (L >= lo0) & (L < lo1) & (A >= la0) & (A < la1)
    o = oasis[sl]; src = np.zeros(o.shape, bool)
    pts = [tuple(HAN['patches'][n]['lonlat']) for n in sorted(HAN['patches'])]
    pts += [tuple(p) for n in sorted(HAN['spring_lines']) for p in HAN['spring_lines'][n]['points']]
    for lo, la in pts:
        src[np.unravel_index(np.argmin(np.hypot(L - lo, A - la)), L.shape)] = True
    cl, nc = _components((o >= .5) & inb)
    big = np.zeros(nc + 1, bool)
    if nc:
        big[1:] = np.bincount(cl.ravel(), minlength=nc + 1)[1:] >= 4 * HAN['island_min_hex']
        big[np.unique(cl[src])] = True; big[0] = False
    sp, ns = _components((o > HAN['support']) & inb)
    keep = np.zeros(ns + 1, bool)
    if ns:
        keep[np.unique(sp[big[cl]])] = True; keep[0] = False
    out = oasis.copy(); out[sl] = np.where((sp > 0) & ~keep[sp], 0, o).astype(oasis.dtype)
    return out, int(ns - keep.sum())


def oasis_seeds(real_elev, lon, lat, real_river, q, r, origin):
    """Oasis weight in [0, 1] per tile from OASIS_SEEDS and the Turpan Han oases (polish multiplies it by
    real_gw x physical, then runs clean_islands)."""
    out = np.zeros(lon.shape, np.float32)
    e = _box(real_elev, 1); gy, gx = np.gradient(e); slope = np.hypot(gx, gy); del gy, gx
    lim_s = _smooth((SEED_SLOPE[1] - slope) / (SEED_SLOPE[1] - SEED_SLOPE[0]))
    rv = np.clip(_box(real_river, 2) * 3, 0, 1)
    warp = 1 + SEED_WARP * gpu_noise.fbm(q, r, origin, SEED_GRAIN, 3, seed=249)
    S = 16   # coarse lon/lat grid that finds each seed's tile window (as ground_materials.oasis_mask)
    clo, cla = lon[::S, ::S], lat[::S, ::S]
    out = np.maximum(out, han_oases(e, lim_s, lon, lat, q, r, origin, clo, cla, S)[0])
    for name in sorted(OASIS_SEEDS):
        lo, la, km2 = OASIS_SEEDS[name]
        R = float(np.sqrt(km2 / np.pi)); reach = SEED_EDGE * R * (1 + SEED_RIVER)
        dl = reach / 110.6; dg = dl / max(np.cos(np.radians(la)), .2)
        ci, cj = np.nonzero((np.abs(cla - la) < dl + .5) & (np.abs(clo - lo) < dg + .5))
        if not len(ci):
            continue
        r0, r1 = max(0, (ci.min() - 1) * S), min(lon.shape[0], (ci.max() + 2) * S)
        c0, c1 = max(0, (cj.min() - 1) * S), min(lon.shape[1], (cj.max() + 2) * S)
        sl = (slice(r0, r1), slice(c0, c1))
        d = np.hypot((lon[sl] - lo) * 111.32 * np.cos(np.radians(la)), (lat[sl] - la) * 110.6)
        k = np.unravel_index(np.argmin(d), d.shape)
        if d[k] > 3.:   # the seed is not on this grid
            continue
        e0 = float(e[sl][k])
        de = d * warp[sl] / (1 + SEED_RIVER * rv[sl])
        w = _smooth((SEED_EDGE * R - de) / ((SEED_EDGE - SEED_CORE) * R))
        w = w * _smooth((SEED_RISE[1] - (e[sl] - e0)) / (SEED_RISE[1] - SEED_RISE[0])) * lim_s[sl]
        out[sl] = np.maximum(out[sl], w.astype(np.float32))
    return out


# ------------------------------------------------------------------ subtropical donor split
# The subtropical donor patches run from the flattest (64,202: 99 % cold_1, dense forest) to the most
# rugged, and flat ground took the flattest: the Bengal paddy plains and the Sundarbans drew dense
# forest blobs (East Tianzhu window: material 4 on 40 % of the ground). Real ground now picks its
# list by RESOLVE biome: paddy and mangrove take vanilla's paddy lowland (DONOR_PADDY in polish),
# open subtropical ground (savanna, grass, wetland, other) the mixed patches without the dense forest
# (DONOR_SUB_OPEN), and only forest biomes keep the full list. Off real ground nothing changes.
SUB_PADDY = _bp.biome_set('sub_paddy')     # ('paddy', 'mangrove')
SUB_FOREST = _bp.biome_set('sub_forest')   # tropical, tropical_dry, forest, temperate, conifer, taiga
SUB_R = _K['SUB_R']            # tile cells: box radius of the shares
SUB_KEYS = _K['SUB_KEYS']  # polish pseudo-climates: paddy lowland, open subtropical


def subtropical_split(real_bio, bnames, real_gw):
    """(paddy share, forest share) per tile; off real ground (real_gw <= .5) 0 and 1 (the old list)."""
    on = real_gw > .5
    P = _box(np.isin(real_bio, _ix(bnames, *SUB_PADDY)) & on, SUB_R)
    F = _box(np.where(on, np.isin(real_bio, _ix(bnames, *SUB_FOREST)), True), SUB_R)
    return P.astype(np.float32), F.astype(np.float32)


def subtropical_pick(picked, P, F, u):
    """Per pixel: subtropical (0) pixels take SUB_KEYS[0] where u < P, SUB_KEYS[1] where u is above
    P + (1 - P) F (non-forest share), else stay 0. P, F bilinear shares, u dither noise in [0, 1]."""
    s = picked == 0
    paddy = s & (u < P)
    opn = s & ~paddy & (u >= P + (1 - P) * F)
    return np.where(paddy, SUB_KEYS[0], np.where(opn, SUB_KEYS[1], picked)).astype(picked.dtype)

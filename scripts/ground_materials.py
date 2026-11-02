"""Ground-material diversity for real-terrain canvas ground (polish_campaign_terrain.py,
GROUND MATERIALS block). Pure numpy (no scipy): runs in the pipeline Python.

The canvas ground took the arid donor (bright arid_1/arid_2 sand) on every desert,
alpine, tundra and snow cell, so the Gobi, the Tibetan plateau, the Pamir/Tian Shan
crests and the Siberian tundra all read as yellow sand. Vanilla's own ground, measured
against the real geography under the native map (west-real-terrain.npz biome/ETOPO
under the vanilla blend; offline survey 2026-10-23, scratchpad materials/mv1.py, mv2.py):

  * Hexi gobi (flat): arid_2 47, arid_4 19, arid_3 16, arid_1 13 %; sand seas (Badain
    Jaran, Alxa): arid_1 90-95 %; desert steppe (Tengger, Ordos): arid_1 20-30, steppe_1-4.
  * Hexi piedmont / Qilian foothills: temperate_1, cold_4, cold_1-3, G cold.
  * High ground: >= 2500 m 89-100 % G = 0 (cold band); Qilian > 3000 m cold_1 76,
    cold_2 9, cold_3 8, cold_4 6 %.

Classes (tile grid, real ground only: times real_gw, donor_share, physical; never oasis):
  cold   real elevation above a latitude-dependent line (SNOWLINE), and tundra: G -> 0
         (cold band), climate_map.cm -> cold (polish block); map.hex is cold there too
         (apply_climate: >= 3000 m, snow, tundra, north of 47 N).
  high   cold and steep: vanilla's high-mountain mix (cold_1-dominated).
  plateau cold and flat (Qiangtang, Pamir plateau): grey-brown cold_4/steppe_3-4/mud/arid_3.
  tundra RESOLVE tundra (and 'grass' north of 62 N): grey-brown olive.
  gobi   desert outside the sand seas: vanilla Hexi gobi gravel (arid_2/arid_4/arid_3/arid_1).
  drymtn steep desert below the cold line (Beishan, Kuruktag, Gobi Altai): darker brown.
  alpine montane meadow (RESOLVE alpine) below the cold line where humid (oasis_ground.dryness):
         vanilla's dark alpine mix; dry alpine takes drymtn.
  sgravel steppe at 700-2500 m south of 50 N (Mongolian plateau, Kazakh uplands): part
         of its sand flecks -> gravel.
Sand seas (SAND_SEAS, lon/lat outlines) keep the donor dune sand.

Pixel pass: only bright source materials (REPLACEABLE) change, with probability W (sum
of class weights), dithered by fbm quantiles; the new material is drawn from the
class-weighted mix with one independent fbm field per material (weighted sampling
argmax log(u_m)/p_m), so each material forms its own organic blobs, mixes cross-fade
between classes, and nothing repeats periodically. Woodland, farms, wetlands, oasis and
meadow greens keep their material. The pixel noise runs on gpu_noise.fbm_at (the bytes of
terrain_noise on the gathered points; CPU fallback).
"""
import numpy as np

import gpu_noise
from terrain_noise import fbm

# ------------------------------------------------------------------ constants
# The values live in biome_profiles (CONSTS['ground_materials'], LANDFORMS mixes and gains, with their
# survey notes); the names here read them back.
import biome_profiles as _bp
_K = _bp.consts('ground_materials')
# Cold band line (m): 3000 m south of 47 N, lower to the north (biome_profiles.COLD_LINES). Edge fuzz +-SNOW_FUZZ m (fbm).
SNOWLINE = _bp.cold_line('SNOWLINE')
SNOW_FUZZ = _K['SNOW_FUZZ']
SNOW_RAMP = _K['SNOW_RAMP']      # m: 0 -> 1 above the (fuzzed) line
STEEP = _K['STEEP']  # m per tile cell (~2 km) of real elevation: ridge weight 0 -> 1
TUNDRA_GRASS_LAT = _K['TUNDRA_GRASS_LAT']

# Mixes (material slot -> share) and take gains per class (biome_profiles.LANDFORMS).
MIX = _bp.mixes()
CLASS_GAIN = _bp.gains()
RELIEF = _K['RELIEF']   # [RELIEF GROUND] open ground by real relief (biome_profiles CONSTS ground_materials)
OPEN_EXTRA = np.zeros(256, bool); OPEN_EXTRA[_K['OPEN_EXTRA']] = True   # donor greens/browns redrawn on open ground
RELIEF_SPECS = _bp.relief_specs()   # landform -> own relief grading (blend, mix or None = global parts, noise)
MATERIALS = sorted({m for v in MIX.values() for m in v} | {m for v in RELIEF['mix'].values() for m in v}
                   | {m for sp in RELIEF_SPECS.values() if sp['mix'] for v in sp['mix'].values() for m in v})
# Only these source materials change: bright sand and bright steppe (the arid and steppe donors' light members).
REPLACEABLE = np.zeros(256, bool); REPLACEABLE[_K['REPLACEABLE']] = True
# [REALISM] On real desert below the cold line and outside oases (fields['Wd']) the donors' greens also change.
DESERT_EXTRA = np.zeros(256, bool); DESERT_EXTRA[_K['DESERT_EXTRA']] = True
DESERT_RAMP = _K['DESERT_RAMP']
# [STEP 6] banned slots per landform (biome_profiles.AUDIT 'banned'): row k+1 = landform k of the classify
# order, row 0 = no dominant landform. TW3K_ENFORCE_BANNED=0 leaves the table empty (pre-step-6 output).
# [STOP-LOSS 2026-10-26, video package] enforced on the desert landforms only (biome_profiles.BAN_ENFORCED):
# on alpine_meadow the redraw turned the mirrored cold_1/2 band along the native west border (97 E 32 N) into
# arid_1 sand. The table is the rule predicate biome_profiles.ban_table (step 2).
BAN_ENFORCED = _bp.BAN_ENFORCED
BAN = _bp.ban_table()
GRAIN = _K['GRAIN']   # hexes: take-dither, material blobs, G-band dither
# [GRAIN PER LANDFORM] (ground mottle, 2026-11-01): a landform's own material-blob grain (biome_profiles LANDFORMS 'grain';
# the rest, open ground included, GRAIN[1]). tile_fields gives each tile the share of each grain in its mix (fields 'GS':
# row 0 the default grain, row k GRAIN_LEVELS[k-1]); apply_rows draws every material's noise at each grain present and
# joins the uniforms with a Gaussian copula, u = Phi(sum_k s_k Phi^-1(u_k) / sqrt(sum_k s_k^2)): uniform for any shares
# (the mix shares hold) and continuous with them (no seam where landforms meet). No landform grain: the old path.
GRAINS = _bp.grains()
# [DARK REDRAW] (ground mottle r8, 2026-11-01): landforms with LANDFORMS 'redraw' slots (the dark cold donors cold_1-3 on
# plateau, alpine_meadow, high) take those donor pixels with their class weight (fields 'Wr', like the desert-green pass
# DESERT_EXTRA): bilinear weight and the take dither, so the edge is graded (the step-6 ban, tile-rint, drew tile blocks).
REDRAW = _bp.redraws()
REDRAW_EXTRA = np.zeros(256, bool)
for _sl in REDRAW.values():
    REDRAW_EXTRA[list(_sl)] = True
GRAIN_LEVELS = tuple(sorted({float(g) for g in GRAINS.values()} - {float(GRAIN[1])}))
# RESOLVE class groups (biome_profiles.BIOMES flags; members and order as before, inconsistencies kept)
DRY_HIGH = _bp.biome_set('gm_dry_high')          # tile_fields: high / plateau (cold band) ground
FARM_DRY = _bp.biome_set('farm_dry')             # farm_mud_fields: farming only where watered
FARM_ADMIT = _bp.biome_set('farm_admit')         # farm_mud_fields: admit 1
FARM_ADMIT_HALF = _bp.biome_set('farm_admit_half')   # admit .5
FARM_HUM_DRY = _bp.biome_set('farm_hum_dry')     # tea humidity: 1 - blurred share of these

# Sand seas (lon, lat outlines; real dune fields). Edges blurred and fbm-warped.
SAND_SEAS = _bp.region('sand_seas')['polygons']   # Taklamakan ... Lut (biome_profiles.REGIONS)


def _box(a, radius):
    from native_campaign_climate import box_blur
    return box_blur(np.asarray(a, np.float32), radius)


def _smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


def _in_poly(lon, lat, poly):
    p = np.asarray(poly, np.float64); inside = np.zeros(lon.shape, bool)
    x0, y0 = p[-1]
    for x1, y1 in p:
        c = ((y1 > lat) != (y0 > lat)) & (lon < (x0 - x1) * (lat - y1) / np.where(y0 == y1, 1e-9, y0 - y1) + x1)
        inside ^= c; x0, y0 = x1, y1
    return inside


def sand_mask(lon, lat):
    m = np.zeros(lon.shape, bool)
    for poly in SAND_SEAS.values():
        p = np.asarray(poly); b = (lon >= p[:, 0].min()) & (lon <= p[:, 0].max()) & (lat >= p[:, 1].min()) & (lat <= p[:, 1].max())
        ii, jj = np.nonzero(b)
        if len(ii):
            m[ii, jj] |= _in_poly(lon[ii, jj], lat[ii, jj], poly)
    return m


def snowline(lat):
    return _bp.snowline_m(lat)   # rule predicate (COLD_LINES SNOWLINE)


def tile_fields(real_elev, real_bio, bnames, lon, lat, ground_w, q, r, origin, oasis=None, desert_w=None, river=None, keep=None):
    """Class weights per tile cell. ground_w: real-ground weight (real_gw x donor_share x
    physical in polish). Returns dict: W (take weight), P (materials x tiles mix, normalised
    where W > 0), cold (G band weight), and the class weights for reports. river: real river share per
    tile (relief valleys); keep: whitelist weight beyond the oases (polish alpine belts: high-valley
    meadows) where the banned slots are never redrawn."""
    # the landform classes (biome_profiles.classify: cold line, ridge, tundra, desert in/out of the ergs,
    # steppe gravel, alpine humid/dry by oasis_ground.dryness)
    w, k = _bp.classify(real_elev, real_bio, bnames, lon, lat, q, r, origin)
    e, des, nc, cold, dry_high, tundra_c = k['e'], k['des'], k['nc'], k['cold'], k['dry_high'], k['tundra_c']; del k
    gw = np.clip(ground_w, 0, 1).astype(np.float32)
    if oasis is not None:
        gw = gw * np.clip(1 - 4 * oasis, 0, 1)
    raw = {c: wc for c, wc in w.items()}   # class weights before the real-ground weight (banned_tiles)
    P = np.zeros((len(MATERIALS),) + e.shape, np.float32); W = np.zeros(e.shape, np.float32); Wp = np.zeros_like(W)
    # [RELIEF GROUND] valley / slope / crest parts by real relief: the open ground outside the landform
    # classes takes them whole, the RELIEF 'classes' landforms a classes_blend share of their mix
    wo, parts = relief_fields(e, real_bio, bnames, river, raw)
    Pr = np.zeros_like(P)
    for part, pw in parts.items():
        for m, s in RELIEF['mix'][part].items():
            Pr[MATERIALS.index(m)] += pw * s
    Wx = np.zeros_like(W)
    for c, wc in w.items():
        wc = (wc * CLASS_GAIN[c] * gw).astype(np.float32); w[c] = wc; Wp += wc
        if c != 'erg':          # the dune sand keeps its donor sand (the redrawn erg mix showed a blocky cell
            W += wc             # pattern in the replica, 2026-10-25); only DESERT_EXTRA and the ban draw from erg
        sp = RELIEF_SPECS.get(c)
        own = 1 - sp['blend'] if sp else 1.
        for m, s in MIX[c].items():
            P[MATERIALS.index(m)] += wc * s * own
        if sp:
            P += wc * sp['blend'] * (Pr if sp['mix'] is None else own_relief(sp, parts, q, r, origin))
            Wx += wc * sp['blend']   # the donors' greens/browns there are redrawn (OPEN_EXTRA) with the graded share
    Wo = (wo * gw).astype(np.float32); Wp += Wo; W += Wo
    P += Wo * Pr; del Pr, parts
    Wo = np.clip(Wo + Wx, 0, 1).astype(np.float32); del Wx   # redraw weight of the donors' greens/browns
    if GRAIN_LEVELS:   # [GRAIN PER LANDFORM] share of each grain in the tile's mix (open ground: the default)
        GS = np.zeros((len(GRAIN_LEVELS) + 1,) + e.shape, np.float32)
        for c, wc in w.items():
            if c in GRAINS and float(GRAINS[c]) != float(GRAIN[1]):
                GS[1 + GRAIN_LEVELS.index(float(GRAINS[c]))] += wc
        GS[1:] /= np.maximum(Wp, 1e-6)
        GS[0] = np.clip(1 - GS[1:].sum(0), 0, 1)
    Wr = None
    if REDRAW:   # [DARK REDRAW] the redraw classes' weight (one slot set: REDRAW_EXTRA)
        Wr = np.clip(sum(w[c] for c in REDRAW if c in w), 0, 1).astype(np.float32)
    P /= np.maximum(Wp, 1e-6)
    W = np.clip(W, 0, 1)
    # desert-green pass (DESERT_EXTRA): desert_w (real ground ramped from the native border, polish) when
    # given, else the class weight gw; never in oases
    dw = gw if desert_w is None else np.clip(desert_w, 0, 1) * (np.clip(1 - 4 * oasis, 0, 1) if oasis is not None else 1)
    Wd = np.clip(des * nc * dw, 0, 1).astype(np.float32)
    coldw = (np.maximum(cold * dry_high, _box(tundra_c, 3)) * gw).astype(np.float32)
    # forests above the line (conifer/taiga) also take the cold band
    coldw = np.maximum(coldw, (cold * gw).astype(np.float32))
    # step 6 reaches the real ground (gw, not oasis). bc1061ad (reach also the desert pass weight dw, for the
    # NONPLAY filler) reverted for the video package (stop-loss 2026-10-26: NONPLAY is off, and the wider reach
    # cleared the soft mirror band to a straight sand/green line at q 884, Qilian).
    reach = gw.astype(np.float32)   # the ban reach: real ground, not oasis
    if keep is not None:
        reach = reach * np.clip(1 - 4 * keep, 0, 1)
    dom, domw = _bp.dominant({c: (raw[c] * CLASS_GAIN[c] * reach).astype(np.float32) for c in raw})
    # 'dom'/'domw'/'reach': the persisted landform field (landform_tiles.npz); 'L' the ban rows (0 where no table)
    out = {'W': W, 'Wd': Wd, 'Wo': Wo, 'P': P.astype(np.float16), 'cold': coldw, 'classes': w,
            'L': np.where((domw > _bp.BAN_MIN) & BAN[1:].any(), dom, 0).astype(np.uint8), 'dom': dom, 'domw': domw,
            'reach': reach, 'allowed': _bp.allowed_slots(dom.shape),
            'w_alpine': (raw['alpine_meadow'] + raw['alpine_steppe']).astype(np.float32)}   # raw classify weight (props treeline threshold)
    if GRAIN_LEVELS:
        out['GS'] = GS.astype(np.float16)
    if Wr is not None:
        out['Wr'] = Wr
    return out


def own_relief(sp, parts, q, r, origin):
    """(materials, tiles) mix of a landform's own relief grading: valley (humid + dry parts), slope, crest
    shares of the tile times its parts' mixes; sp['noise'] (scale hexes, amplitude, seed) shifts share from
    slope/crest to the valley by a low-frequency fbm (large light/dark sheets at one height)."""
    v = parts['valley_humid'] + parts['valley_dry']; sl = parts['slope']; cr = parts['crest']
    if sp.get('noise'):
        sc, amp, seed = sp['noise']
        n = amp * fbm(q, r, origin, sc, 3, seed=int(seed)).astype(np.float32)
        v2 = np.clip(v + n, 0, 1); k = np.where(sl + cr > 1e-6, (1 - v2) / np.maximum(sl + cr, 1e-6), 0)
        v, sl, cr = v2, sl * k, cr * k
    out = np.zeros((len(MATERIALS),) + v.shape, np.float32)
    for part, share in (('valley', v), ('slope', sl), ('crest', cr)):
        for m, s in sp['mix'][part].items():
            out[MATERIALS.index(m)] += share * s
    return out


def relief_fields(e, real_bio, bnames, river, raw):
    """The relief rule predicate (biome_profiles.relief_parts)."""
    return _bp.relief_parts(e, real_bio, bnames, river, raw)


# ------------------------------------------------------------------ farms and mud
# Vanilla dlc07 (offline survey 2026-10-24, the whole blend against vanilla map.hex): farms_grain_1/2,
# farms_rice_1/2, farms_tea_1 (8-12) are 2.26 % of the land pixels, mud (14) 0.35 %. Farm share by
# distance to water hexes: .01 % within .5 hex, .14 % at 1-2, .5 % at 2-3, 1.3-1.7 % at 3-8, 2-2.6 %
# beyond (fields stand back from the bank, on the valley floor and the plain); fields are dab blobs
# (area-weighted median 264 hex^2), grain_2 the rims of grain_1 dabs. Mud: blobs (area-weighted median
# 52 hex^2) on flat low ground. Here, on real-terrain canvas ground only (the GROUND MATERIALS weight):
#   farms  river valleys (map.hex river hexes: rising from 1 to 3 hexes, full to 9, gone by 16; Natural
#          Earth rivers narrower), lake shores, wide flat lowland plains (Ganges, Mekong delta, Manchuria
#          south of FARM_LAT), the ring around authored towns beyond FARM_TOWN_CLEAR hexes (farm_fields.py
#          paints the town's own field complex inside it, later), towns on arid ground as oases (their own
#          disc, ARID_TOWN; polish hands these fields to farm_fields as 'ignore' so its 'already farmed'
#          rule does not skip the town), desert oases (polish oasis belts, OASES, OASIS_AREAS) - on flat ground
#          and, for valleys, the valley floor (below the local mean elevation). Arid and steppe ground
#          only where watered (valley or oasis). Kinds by climate: rice on subtropical lowland, tea on
#          subtropical humid hills (never dry ground), grain elsewhere (temperate, arid oases, cold south
#          of FARM_LAT).
#   mud    flat low ground next to river hexes (to MUD_RIVER hexes) and lake hexes (MUD_LAKE).
# Pixel pass: a field is where a 4-octave fbm (FARM_GRAIN hexes, uniformised) falls under the bilinear
# weight, its edge frayed by a fine fbm (soft irregular blocks, no tile steps: every field is bilinear
# per pixel and no hard tile mask cuts it); the inner part of a field is the _1 material where a dab
# fbm allows, the rim the _2 material (vanilla's grain_2 rims); one kind per blob (weighted sampling on
# coarse fbm). Mud the same way, first (nearer the water). Only FARM_SOURCE materials change (no roads,
# wetlands, farms, mud, shroud) and no pixel of the cold G band takes a field.
FARM_KINDS = _K['FARM_KINDS']      # grain, rice, tea: (inner, rim) material
FARM_GRAIN, FARM_FRAY, FARM_DAB, FARM_KIND_GRAIN = (_K[k] for k in ('FARM_GRAIN', 'FARM_FRAY', 'FARM_DAB', 'FARM_KIND_GRAIN'))   # hexes (FARM_GRAIN: 4 octaves, field complexes)
FARM_FRAY_AMP = _K['FARM_FRAY_AMP']        # weight units of the fine edge noise
FARM_RIM = _K['FARM_RIM']             # weight margin under which a field pixel is rim (_2) material
FARM_INNER = _K['FARM_INNER']   # share of the inner part that takes the _1 material (grain, rice, tea)
FARM_RIVER = _K['FARM_RIVER']     # hexes from river hexes: 0 -> 1 -> 1 -> 0
FARM_REAL_RIVER = _K['FARM_REAL_RIVER']   # hexes from Natural Earth rivers (no river hexes)
FARM_LAKE = _K['FARM_LAKE']
FARM_TOWN_CLEAR, FARM_TOWN = _K['FARM_TOWN_CLEAR'], _K['FARM_TOWN']   # hexes from authored towns
FARM_TOWN_KIND = _K['FARM_TOWN_KIND']
# Valley, lake and plain weights halved after the r16 review (Funan/Mekong plains densely tiled: 60-hex
# window median 5.1%, p90 10% on farmable land against vanilla's p75 2.5%, p90 8%); towns and oases kept.
FARM_WEIGHT = _K['FARM_WEIGHT']
FARM_LAT = _bp.region('han_farming_frontier')['lat']      # degrees N: grain fades out (Han-era farming frontier)
FARM_ELEV = _K['FARM_ELEV']   # m: no fields above
FARM_SLOPE = _K['FARM_SLOPE']   # world units per hex (vanilla: farm_fields.SLOPE_KEEP)
TEA_SLOPE = _K['TEA_SLOPE']
TEA_ELEV = _K['TEA_ELEV']
RICE_ELEV = _K['RICE_ELEV']
RICE_LAT = _K['RICE_LAT']
PLAIN_ELEV = _K['PLAIN_ELEV']   # m: rice / grain plains fade out
VALLEY_REL = _K['VALLEY_REL']   # m above the 10-cell local mean: valley floor 1 -> 0
MUD_RIVER = _K['MUD_RIVER']
MUD_LAKE = _K['MUD_LAKE']
MUD_WEIGHT = _K['MUD_WEIGHT']
MUD_ELEV = _K['MUD_ELEV']
MUD_GRAIN = _K['MUD_GRAIN']
# Irrigated oases without a Natural Earth river in the real-terrain layer (lon, lat, radius km;
# Ferghana an outline): the oasis setting, edges fbm-warped. Towns on arid ground are oases too
# (ARID_TOWN, hexes from the town: 0 -> 1 -> 1 -> 0, times FARM_TOWN_KIND; its width the farm_fields ARID_OWN_OASIS).
OASES = _bp.region('oasis_discs')['discs']   # biome_profiles.REGIONS (22 discs)
OASIS_AREAS = _bp.region('oasis_areas')['polygons']   # Ferghana
OASIS_FARM = _K['OASIS_FARM']            # oasis setting weight before FARM_WEIGHT['oasis']
FARM_OASES_SKIP = tuple(_K.get('FARM_OASES_SKIP', ()))   # [ARID FARM] OASES discs without fields (Turpan: the Han oases)
ARID_TOWN = _K['ARID_TOWN']
if ARID_TOWN[2] is None:   # [ARID FARM] full to farm_fields ARID_OWN_OASIS (one constant for both farm writers)
    _aoo = float(_bp.CONSTS['farm_fields']['ARID_OWN_OASIS'])
    ARID_TOWN = (ARID_TOWN[0], ARID_TOWN[1], _aoo, _aoo + ARID_TOWN[3])
TOWN_WARP = _K['TOWN_WARP']          # fbm warp of the town distance (no circular rings)
ARID_TOWN_REACH = ARID_TOWN[3] / (1 - TOWN_WARP)   # hexes: the farthest an arid town's own fields reach (fbm in [-1, 1])
FARM_SOURCE = np.zeros(256, bool); FARM_SOURCE[_K['FARM_SOURCE']] = True


def _ramp(d, t):
    """0 below t[0], 1 from t[1] to t[2], 0 beyond t[3] (smoothstep ramps)."""
    return (_smooth((d - t[0]) / max(t[1] - t[0], 1e-6)) * _smooth((t[3] - d) / max(t[3] - t[2], 1e-6))).astype(np.float32)


def _dist(seed, cap):
    """Euclidean distance (hexes) from the nearest seed tile reached within cap tile cells
    (4-neighbour BFS nearest seed, then hypot, as polish's tonal buffer); 1e6 beyond."""
    near, dist = _nearest(seed, cap)
    if near is None:
        return np.full(seed.shape, 1e6, np.float32)
    return np.where(dist <= cap, _seed_distance(near), np.float32(1e6))


def _nearest(seed, cap):
    from fastops import propagate_nearest
    H, W = seed.shape
    if not seed.any():
        return None, None
    return propagate_nearest(np.arange(H * W, dtype=np.int32).reshape(H, W), seed, int(cap))


def _seed_distance(near):
    """Euclidean distance (hexes, float32) from each tile to the tile index near holds."""
    H, W = near.shape
    sy, sx = np.divmod(near, np.int32(W))
    dy = (np.arange(H, dtype=np.int32)[:, None] - sy).astype(np.float32); dx = (np.arange(W, dtype=np.int32)[None, :] - sx).astype(np.float32)
    del sy, sx
    return np.hypot(dx, dy) * np.float32(.5)


def oasis_mask(lon, lat, q, r, origin, skip=()):
    """OASES / OASIS_AREAS weight in [0, 1] per tile (fbm-warped edges); skip: OASES names left out."""
    out = np.zeros(lon.shape, np.float32)
    warp = 1 + .35 * gpu_noise.fbm(q, r, origin, 6, 3, seed=246)
    S = 16   # coarse lon/lat grid that finds each oasis' tile window
    clo, cla = lon[::S, ::S], lat[::S, ::S]
    for name in sorted(OASES):
        if name in skip:
            continue
        lo, la, rad = OASES[name]
        dl = 1.6 * rad / 110.6; dg = dl / max(np.cos(np.radians(la)), .2)
        ci, cj = np.nonzero((np.abs(cla - la) < dl + .5) & (np.abs(clo - lo) < dg + .5))
        if not len(ci):
            continue
        r0, r1 = max(0, (ci.min() - 1) * S), min(lon.shape[0], (ci.max() + 2) * S)
        c0, c1 = max(0, (cj.min() - 1) * S), min(lon.shape[1], (cj.max() + 2) * S)
        ii, jj = np.nonzero((np.abs(lat[r0:r1, c0:c1] - la) < dl) & (np.abs(lon[r0:r1, c0:c1] - lo) < dg))
        ii += r0; jj += c0
        if len(ii):
            d = np.hypot((lon[ii, jj] - lo) * 111.32 * np.cos(np.radians(la)), (lat[ii, jj] - la) * 110.6) * warp[ii, jj]
            out[ii, jj] = np.maximum(out[ii, jj], _smooth((rad - d) / (.45 * rad)))
    for name in sorted(OASIS_AREAS):
        poly = OASIS_AREAS[name]; p = np.asarray(poly)
        ci, cj = np.nonzero((clo >= p[:, 0].min() - .5) & (clo <= p[:, 0].max() + .5) & (cla >= p[:, 1].min() - .5) & (cla <= p[:, 1].max() + .5))
        if not len(ci):
            continue
        r0, r1 = max(0, (ci.min() - 1) * S), min(lon.shape[0], (ci.max() + 2) * S)
        c0, c1 = max(0, (cj.min() - 1) * S), min(lon.shape[1], (cj.max() + 2) * S)
        m = _in_poly(lon[r0:r1, c0:c1], lat[r0:r1, c0:c1], poly)
        if m.any():
            v = _smooth((_box(m, 4) - .5 + .3 * gpu_noise.fbm(q[r0:r1, c0:c1], r[r0:r1, c0:c1], origin, 8, 3, seed=247)) * 2.2 + .5)
            out[r0:r1, c0:c1] = np.maximum(out[r0:r1, c0:c1], v)
    return out


def farm_mud_fields(real_elev, real_bio, bnames, lon, lat, ground_w, climate, surface, river, lake, real_river,
                    towns, q, r, origin, oasis=None, cold=None):
    """Farm and mud weights per tile cell (south-up grids). ground_w: the GROUND MATERIALS weight;
    climate: tile climate (0 subtropical, 1 cold, 2 temperate, 3 arid); surface: world height;
    river / lake: tiles of map.hex river / lake hexes (water); real_river: Natural Earth river tiles;
    towns: [(tile x, tile y, kind)] of authored towns (farm_fields.kind_of); q, r, origin: polish's fbm
    grid; oasis: polish oasis belts; cold: G cold-band weight (no fields where it is set).
    Returns {'F', 'K' (3, H, W) kind shares (grain, rice, tea), 'M', 'parts' (report)}."""
    ix = lambda *ns: [bnames.index(n) for n in ns if n in bnames]
    gw = np.clip(ground_w, 0, 1).astype(np.float32)
    H, W = gw.shape
    e = _box(real_elev, 1)
    gz, gx = np.gradient(surface.astype(np.float32)); slope = _box(np.hypot(gx, gz) * 2, 1); del gz, gx
    flat = np.interp(slope, *FARM_SLOPE).astype(np.float32)
    rel = e - _box(e, 10)
    floor = _smooth((VALLEY_REL[1] - rel) / (VALLEY_REL[1] - VALLEY_REL[0]))
    d_r = _dist(river, 2 * FARM_RIVER[3] + 2)
    d_rr = _dist(real_river & ~river, 2 * FARM_REAL_RIVER[3] + 2)
    d_l = _dist(lake, 2 * FARM_LAKE[3] + 2)
    tw = np.zeros((H, W), np.float32); aw = np.zeros((H, W), np.float32); clear = np.ones((H, W), np.float32)
    if towns:
        # nearest authored town: its kind weight and whether it stands on arid ground (an oasis town)
        ts = np.zeros((H, W), bool); kw = np.zeros((H, W), np.float32); ar = np.zeros((H, W), bool)
        for tx, ty, kind in sorted(towns):
            if 0 <= ty < H and 0 <= tx < W:
                ts[ty, tx] = True; kw[ty, tx] = max(kw[ty, tx], FARM_TOWN_KIND.get(kind, 0.)); ar[ty, tx] = climate[ty, tx] == 3
        near, dist = _nearest(ts, 2 * FARM_TOWN[3] + 2)
        d_t = np.where(dist <= 2 * FARM_TOWN[3] + 2, _seed_distance(near), np.float32(1e6))
        kn = kw.ravel()[near.ravel()].reshape(H, W); an = ar.ravel()[near.ravel()].reshape(H, W); del near, dist
        dw = d_t * (1 + TOWN_WARP * gpu_noise.fbm(q, r, origin, 5, 3, seed=248)).astype(np.float32)
        tw = np.where(an, 0, _ramp(dw, FARM_TOWN) * kn).astype(np.float32)
        aw = np.where(an, _ramp(dw, ARID_TOWN) * kn, 0).astype(np.float32); del dw
        # 0 inside FARM_TOWN_CLEAR of a non-arid town: farm_fields paints the town's own complex there
        # (arid towns: their oasis; polish passes farm_fields these fields to ignore in its skip rule)
        clear = np.where(an, 1, _smooth((d_t - FARM_TOWN_CLEAR) / 1.)).astype(np.float32); del kn, an, d_t
    # moisture / setting terms
    v_r = _ramp(d_r, FARM_RIVER); v_rr = _ramp(d_rr, FARM_REAL_RIVER); v_l = _ramp(d_l, FARM_LAKE)
    sub = climate == 0
    dry = np.isin(real_bio, ix(*FARM_DRY)) | (climate == 3)
    plain_ok = _smooth((_box(flat, 6) - .55) / .2)
    pe = np.where(sub, _smooth((PLAIN_ELEV[0][1] - e) / (PLAIN_ELEV[0][1] - PLAIN_ELEV[0][0])),
                  _smooth((PLAIN_ELEV[1][1] - e) / (PLAIN_ELEV[1][1] - PLAIN_ELEV[1][0])))
    oas = np.maximum.reduce([np.zeros((H, W), np.float32) if oasis is None else np.clip(oasis, 0, 1).astype(np.float32),
                             OASIS_FARM * oasis_mask(lon, lat, q, r, origin, skip=FARM_OASES_SKIP), aw])
    wf = FARM_WEIGHT
    valley = np.maximum.reduce([wf['river'] * v_r, wf['real_river'] * v_rr, wf['lake'] * v_l]) * floor
    watered = np.maximum(np.clip(np.maximum.reduce([v_r, v_rr, v_l]) * floor * 1.5, 0, 1), np.clip(oas * 2, 0, 1))
    setting = np.maximum.reduce([valley, wf['plain'] * plain_ok * pe * ~dry, wf['town'] * tw, wf['oasis'] * oas])
    setting = np.where(dry, setting * watered, setting)   # arid / steppe ground: watered only
    # where farming is possible at all
    admit = np.isin(real_bio, ix(*FARM_ADMIT)).astype(np.float32)
    admit = np.maximum(admit, np.isin(real_bio, ix(*FARM_ADMIT_HALF)) * .5)
    admit = _box(admit, 2) * _smooth((FARM_ELEV[1] - e) / (FARM_ELEV[1] - FARM_ELEV[0]))
    admit *= np.where(sub, 1., _smooth((FARM_LAT[1] - lat) / (FARM_LAT[1] - FARM_LAT[0])))
    if cold is not None:
        admit *= np.clip(1 - 2 * cold, 0, 1)
    # kinds: rice on subtropical lowland, tea on subtropical humid hills, grain elsewhere
    hum = 1 - _box(np.isin(real_bio, ix(*FARM_HUM_DRY)), 3)
    rice = sub * _smooth((RICE_ELEV[1] - e) / (RICE_ELEV[1] - RICE_ELEV[0])) * _smooth((RICE_LAT[1] - lat) / (RICE_LAT[1] - RICE_LAT[0])) * (1 - .5 * dry)
    hill = np.interp(slope, *TEA_SLOPE).astype(np.float32) * _ramp(e, TEA_ELEV)
    tea = sub * hill * hum * (1 - dry)
    F_flat = gw * admit * flat * setting * clear
    # tea fields stand on hills beside the valleys and towns (the setting without the floor rule)
    tea_set = np.maximum.reduce([.6 * wf['river'] * v_r, .6 * wf['real_river'] * v_rr, wf['town'] * tw])
    F_tea = gw * admit * tea * tea_set * clear * .8
    F = np.maximum(F_flat, F_tea).astype(np.float32)
    k_tea = np.where(F > 0, F_tea / np.maximum(F_flat + F_tea, 1e-6), 0)
    k_rice = (1 - k_tea) * np.clip(rice, 0, 1)
    K = np.stack([1 - k_tea - k_rice, k_rice, k_tea]).astype(np.float16)
    # mud
    m = np.maximum(MUD_WEIGHT['river'] * _ramp(d_r, MUD_RIVER), MUD_WEIGHT['lake'] * _ramp(d_l, MUD_LAKE))
    M = (gw * m * flat * floor * (e < MUD_ELEV)).astype(np.float32)
    parts = {k: int((v > .05).sum()) for k, v in (('valley', valley), ('plain', plain_ok * pe), ('town', tw), ('oasis', oas),
                                                     ('farm', F), ('tea', F_tea), ('mud', M))}
    return {'F': F, 'K': K, 'M': M, 'parts': parts}


class Quantiles:
    """fbm -> uniform [0,1]; the default calibration remains the legacy sample."""
    def __init__(self, origin, specs, *, calibration_size=1024, calibration_step=.25):
        if not isinstance(calibration_size, int) or calibration_size < 2:
            raise ValueError('Quantile calibration size must be an integer >=2')
        if not np.isfinite(calibration_step) or calibration_step <= 0:
            raise ValueError('Quantile calibration step must be finite and positive')
        if calibration_size == 1024 and calibration_step == .25:
            # Preserve the expression and sample bytes used by every old field.
            g = np.mgrid[0:1024, 0:1024][::-1] / 4.
        else:
            g = np.mgrid[0:calibration_size, 0:calibration_size][::-1] * calibration_step
        self.q = {k: np.quantile(fbm(g[0], g[1], origin, sc, oc, seed=sd), np.linspace(0, 1, 257)) for k, (sc, oc, sd) in specs.items()}
        self.specs = specs; self.origin = origin

    def u(self, k, hx, hy):
        sc, oc, sd = self.specs[k]
        return self.uniform(k, fbm(hx, hy, self.origin, sc, oc, seed=sd))

    def uniform(self, k, v):
        return np.interp(v, self.q[k], np.linspace(0, 1, 257)).astype(np.float32)

    def noise_at(self, keys, xa, ya, mask):
        """fbm of keys (one scale and octave count) at the pixels of mask on the grid xa
        (columns) by ya (rows): (len(keys), mask.sum()), the bytes of u's fbm on those points."""
        sc, oc, _ = self.specs[keys[0]]; assert all(self.specs[k][:2] == (sc, oc) for k in keys)
        return gpu_noise.fbm_at(xa, ya, mask, self.origin, sc, oc, seeds=[self.specs[k][2] for k in keys])


def make_quantiles(origin, farm=False):
    sp = {'take': (GRAIN[0], 3, 221), 'g': (GRAIN[2], 3, 222)}   # 3 octaves: blobs of GRAIN down to GRAIN/4 hexes
    for m in MATERIALS:
        sp[m] = (GRAIN[1], 3, 230 + m)
        for k, g in enumerate(GRAIN_LEVELS, 1):   # [GRAIN PER LANDFORM] own seeds (1000 k + 230 + m)
            sp[f'{m}@{g}'] = (g, 3, 1000 * k + 230 + m)
    if farm:   # farms and mud (seeds 240-251)
        sp.update({'farm': (FARM_GRAIN, 4, 240), 'fray': (FARM_FRAY, 2, 241), 'mfray': (FARM_FRAY, 2, 251),
                   'dab': (FARM_DAB, 2, 242), 'mud': (MUD_GRAIN, 3, 250)})
        for k in range(len(FARM_KINDS)):
            sp[f'kind{k}'] = (FARM_KIND_GRAIN, 2, 243 + k)
    return Quantiles(origin, sp)


def _bilinear(T, tx, ty):
    """Sample a tile-grid field at fractional tile coords (tx, ty arrays, same shape)."""
    H, W = T.shape[-2:]
    x0 = np.clip(np.floor(tx).astype(np.int32), 0, W - 2); y0 = np.clip(np.floor(ty).astype(np.int32), 0, H - 2)
    fx = np.clip(tx - x0, 0, 1).astype(np.float32); fy = np.clip(ty - y0, 0, 1).astype(np.float32)
    a = T[..., y0, x0].astype(np.float32); b = T[..., y0, x0 + 1].astype(np.float32)
    c = T[..., y0 + 1, x0].astype(np.float32); d = T[..., y0 + 1, x0 + 1].astype(np.float32)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _bilinear_rows(T, tx, ty):
    """_bilinear on the grid of tx (columns) by ty (rows), its float32 operations with the
    horizontal terms once per distinct tile row: (h, i0, i1, fy), the value at row i, column j
    being h[..., i0[i], j] * (1 - fy[i]) + h[..., i1[i], j] * fy[i]."""
    H, W = T.shape[-2:]
    x0 = np.clip(np.floor(tx).astype(np.int32), 0, W - 2); y0 = np.clip(np.floor(ty).astype(np.int32), 0, H - 2)
    fx = np.clip(tx - x0, 0, 1).astype(np.float32); fy = np.clip(ty - y0, 0, 1).astype(np.float32)
    u = np.unique(np.concatenate((y0, y0 + 1))); Tu = T[..., u, :]
    h = Tu[..., x0].astype(np.float32); h *= 1 - fx; h += Tu[..., x0 + 1].astype(np.float32) * fx
    return h, np.searchsorted(u, y0), np.searchsorted(u, y0 + 1), fy


def apply_rows(cur, row0, row1, oh, fields, Q, col0=0):
    """Apply to cur[row0:row1] (north-first blend rows, absolute rows row0..row1 of a blend
    of height oh; cur's columns start at absolute column col0). Returns (materials changed,
    G pixels set), and (..., farm pixels, mud pixels) when fields carry the farm/mud weights
    (farm_mud_fields merged in; Q from make_quantiles(origin, farm=True))."""
    blk = cur[row0:row1]
    n, wpx = blk.shape[:2]
    yy = np.arange(row0, row1)[:, None]; xx = np.arange(col0, col0 + wpx)[None, :]
    ty = np.broadcast_to((oh - 1 - yy + .5) / 4 - .5, (n, wpx)); tx = np.broadcast_to((xx + .5) / 4 - .5, (n, wpx))
    Wt, Ct = fields['W'], fields['cold']
    # quick reject: tile rows/cols covered
    tr0 = int(max(0, np.floor(ty.min()) - 1)); tr1 = int(min(Wt.shape[0], np.ceil(ty.max()) + 2))
    tc0 = int(max(0, np.floor(tx.min()) - 1)); tc1 = int(min(Wt.shape[1], np.ceil(tx.max()) + 2))
    fm = 'F' in fields and (fields['F'][tr0:tr1, tc0:tc1].any() or fields['M'][tr0:tr1, tc0:tc1].any())
    wdx = 'Wd' in fields and fields['Wd'][tr0:tr1, tc0:tc1].any()
    wrx = 'Wr' in fields and fields['Wr'][tr0:tr1, tc0:tc1].any()
    bnx = 'L' in fields and fields['L'][tr0:tr1, tc0:tc1].any()
    wox = 'Wo' in fields and fields['Wo'][tr0:tr1, tc0:tc1].any()
    if not (Wt[tr0:tr1, tc0:tc1].any() or Ct[tr0:tr1, tc0:tc1].any() or fm or wdx or bnx or wox or wrx):
        return (0, 0, 0, 0) if 'F' in fields else (0, 0)
    hx = xx / 8.; hy = (oh - 1 - yy) / 8.
    changed = gset = 0
    tx, ty = tx[0], ty[:, 0]   # the fields on the block grid: _bilinear_rows
    h, i0, i1, fy = _bilinear_rows(Ct, tx, ty); f1 = fy[:, None]
    # G band
    cw_ = h[i0] * (1 - f1) + h[i1] * f1; m = cw_ > 0
    si, sj = np.nonzero(m)
    if len(si):
        ug = Q.uniform('g', Q.noise_at(('g',), hx[0], hy[:, 0], m)[0])
        hit = ug < cw_[si, sj]
        blk[si[hit], sj[hit], 1] = 0; gset = int(hit.sum())
    # materials
    h = _bilinear_rows(Wt, tx, ty)[0]; wv = h[i0] * (1 - f1) + h[i1] * f1; del h
    m = (wv > 0) & REPLACEABLE[blk[..., 0]]
    if wdx:
        # desert-green pass: donor greens on real desert take the desert mix with the desert weight
        h = _bilinear_rows(fields['Wd'], tx, ty)[0]; wd = h[i0] * (1 - f1) + h[i1] * f1; del h
        xm = (wd > 0) & DESERT_EXTRA[blk[..., 0]]
        wv = np.where(xm, wd, wv); m |= xm
    if wrx:
        # [DARK REDRAW] the dark cold donors on the redraw landforms take their mix with the class weight
        h = _bilinear_rows(fields['Wr'], tx, ty)[0]; wr = h[i0] * (1 - f1) + h[i1] * f1; del h
        xr = (wr > 0) & REDRAW_EXTRA[blk[..., 0]]
        wv = np.where(xr, np.maximum(wv, wr), wv); m |= xr
    if wox:
        # [RELIEF GROUND] the donors' greens and browns on open ground take the relief mix with the open weight
        h = _bilinear_rows(fields['Wo'], tx, ty)[0]; wo = h[i0] * (1 - f1) + h[i1] * f1; del h
        xo = (wo > 0) & OPEN_EXTRA[blk[..., 0]]
        wv = np.where(xo, np.maximum(wv, wo), wv); m |= xo
    if bnx:
        # [STEP 6] a material banned for the landform dominating the pixel's tile is always redrawn (no take
        # dither: the dither is what left the donor greens as specks)
        lb, al = _ban_rows_fields(fields, tx, ty)
        bm = _banned(lb, al, blk[..., 0])
        # [STEP 4] every banned pixel joins the take mask with weight 2 (> any u0 in [0, 1)), so it is always
        # sampled from the cleared mix below: no banned donor pixel is kept, which _assert_no_banned relies on
        wv = np.where(bm, np.float32(2), wv); m |= bm
    si, sj = np.nonzero(m)
    if len(si):
        u0 = Q.uniform('take', Q.noise_at(('take',), hx[0], hy[:, 0], m)[0])
        hit = u0 < wv[si, sj]; si, sj = si[hit], sj[hit]
        if len(si):
            c0 = sj.min(); h = _bilinear_rows(fields['P'], tx[c0:sj.max() + 1], ty)[0]
            P = h[:, i0[si], sj - c0] * (1 - fy[si]) + h[:, i1[si], sj - c0] * fy[si]; del h   # (M, k)
            if bnx:
                # [STEP 4] banned slots leave the mix before sampling (never drawn, never redrawn afterwards)
                P = _clear_banned(P.astype(np.float32), lb[si, sj], al[si, sj])
            best = np.full(len(si), -np.inf, np.float32); pick = np.zeros(len(si), np.uint8)
            # GPU: the noise of every material of the mix at the taken pixels in one call; the CPU
            # evaluates each material on its own pixels only (the same bytes per pixel).
            mats = [mm for i, mm in enumerate(MATERIALS) if (P[i] > 1e-3).any()]
            if mats and gpu_noise.active():
                m[...] = False; m[si, sj] = True; nz = Q.noise_at(mats, hx[0], hy[:, 0], m)
            else:
                nz = None
            gs = None
            if 'GS' in fields:   # [GRAIN PER LANDFORM] grain shares at the taken pixels; levels present here
                h = _bilinear_rows(fields['GS'], tx[c0:sj.max() + 1], ty)[0]
                gs = (h[:, i0[si], sj - c0] * (1 - fy[si]) + h[:, i1[si], sj - c0] * fy[si]).astype(np.float32); del h
                lv = [k for k in range(1, gs.shape[0]) if (gs[k] > 1e-4).any()]
                if not lv:
                    gs = None
                elif gpu_noise.active():
                    nzg = {k: Q.noise_at(tuple(f'{mm}@{GRAIN_LEVELS[k - 1]}' for mm in mats), hx[0], hy[:, 0], m) for k in lv}
            for i, mm in enumerate(MATERIALS):
                pm = P[i]; ok = pm > 1e-3
                if not ok.any():
                    continue
                u = Q.uniform(mm, nz[mats.index(mm)][ok]) if nz is not None else Q.u(mm, hx[0, sj[ok]], hy[si[ok], 0])
                if gs is not None:
                    u = _grain_join(u, gs[:, ok], lv, [Q.uniform(f'{mm}@{GRAIN_LEVELS[k - 1]}', nzg[k][mats.index(mm)][ok]) if nz is not None
                                                       else Q.u(f'{mm}@{GRAIN_LEVELS[k - 1]}', hx[0, sj[ok]], hy[si[ok], 0]) for k in lv])
                u = np.clip(u, 1e-4, 1 - 1e-6)
                key = np.full(len(si), -np.inf, np.float32); key[ok] = np.log(u) / pm[ok]
                up = key > best; best[up] = key[up]; pick[up] = mm
            blk[si, sj, 0] = pick; changed = len(si)
    if 'F' not in fields:
        if bnx:
            _assert_no_banned(blk, lb, al, row0, col0, 'materials')
        return changed, gset
    fpx = mpx = 0
    if fm:
        fpx, mpx = _farm_mud_rows(blk, fields, Q, tx, ty, i0, i1, fy, hx[0], hy[:, 0])
    if bnx:
        _assert_no_banned(blk, lb, al, row0, col0, 'materials + farms/mud')
    return changed, gset, fpx, mpx


def _grain_join(u0, gs, lv, us):
    """[GRAIN PER LANDFORM] the uniform of one material at pixels with grain shares gs (rows: default, levels): the
    default-grain uniform u0 where its share is 1 (the old bytes), else the Gaussian copula of u0 and the level uniforms us
    (levels lv) weighted by their shares (uniform for any shares; continuous across landform boundaries)."""
    from scipy.special import ndtr, ndtri
    mix = gs[0] < 1 - 1e-6
    if not mix.any():
        return u0
    eps = 1e-7
    zf = lambda v: ndtri(np.clip(v[mix].astype(np.float64), eps, 1 - eps))
    s0 = gs[0][mix].astype(np.float64)
    num = s0 * zf(u0); den = s0 ** 2
    for k, uk in zip(lv, us):
        sk = gs[k][mix].astype(np.float64)
        num += sk * zf(uk); den += sk ** 2
    out = u0.copy()
    out[mix] = ndtr(num / np.sqrt(np.maximum(den, 1e-12))).astype(u0.dtype)
    return out


# ---------------------------------------------------------------- [STEP 4] ban before sampling, end assertion
def _ban_rows_fields(fields, tx, ty):
    """(ban row per pixel, whitelist bits per pixel) on the block grid: the tile under each pixel (rint, as
    the ban has always been read)."""
    Lt = fields['L']
    li = np.clip(np.rint(ty), 0, Lt.shape[0] - 1).astype(np.int64); lj = np.clip(np.rint(tx), 0, Lt.shape[1] - 1).astype(np.int64)
    al = fields.get('allowed')
    return Lt[li[:, None], lj[None, :]], (al[li[:, None], lj[None, :]] if al is not None else None)


def _banned(lb, al, mat):
    """bool: the slot is banned for the pixel's ban row and not whitelisted there (allowed_slots bits)."""
    bm = BAN[lb, mat]
    if al is not None:
        bm &= ~((al >> mat.astype(np.uint32)) & 1).astype(bool)
    return bm


def _clear_banned(P, rows, al):
    """P (materials, pixels) with every slot banned for the pixel's row (and not whitelisted) set to 0; a
    pixel left with no slot above the pick threshold takes its landform's own mix without the banned slots,
    never the slot-0 default."""
    mats = np.asarray(MATERIALS)
    ban = BAN[rows][:, mats].T                         # (materials, pixels)
    if al is not None:
        ban &= ~((al[None, :] >> mats[:, None].astype(np.uint32)) & 1).astype(bool)
    P[ban] = 0
    # INTENDED OUTPUT CHANGE vs main (the 814bb10d behaviour, out of the video pack by the stop-loss): main tests
    # P.sum(0) <= 1e-3, so a pixel whose every slot sits at or below the pick threshold but whose sum is above it
    # falls to the slot-0 default; here it takes its landform's own mix. Counted in FALLBACK_LOG.
    empty = (P.max(0) <= 1e-3) & (rows > 0)
    if empty.any():
        FALLBACK_LOG.append((int(empty.sum()), int(((P.sum(0) > 1e-3) & empty).sum()),
                             np.bincount(rows[empty], minlength=len(_bp.LANDFORMS) + 1)))
        names = list(_bp.LANDFORMS)
        for k in np.unique(rows[empty]).tolist():
            sel = empty & (rows == k)
            for mm, sh in MIX[names[k - 1]].items():
                if not BAN[k, mm]:
                    P[MATERIALS.index(mm), sel] = sh
    return P


class BannedSlotError(RuntimeError):
    pass


FALLBACK_LOG = []   # (pixels on the own-mix fallback, of which main would draw slot 0, per ban row) per block


def fallback_report(clear=True):
    """The own-mix fallback over the run: pixels, pixels main would have left on slot 0, pixels per landform."""
    names = list(_bp.LANDFORMS)
    tot = sum(e[0] for e in FALLBACK_LOG); slot0 = sum(e[1] for e in FALLBACK_LOG)
    per = sum((e[2] for e in FALLBACK_LOG), np.zeros(len(names) + 1, np.int64))
    if clear:
        FALLBACK_LOG.clear()
    return {'pixels': int(tot), 'main_slot0_pixels': int(slot0), 'by_landform': {names[k - 1]: int(v) for k, v in enumerate(per) if k and v}}


def _assert_no_banned(blk, lb, al, row0, col0, stage):
    """The ground-materials invariant: no pixel holds a slot banned (and not whitelisted) for its tile's
    landform. Raises BannedSlotError with the first offending pixel; nothing is redrawn."""
    bm = _banned(lb, al, blk[..., 0])
    if bm.any():
        r, c = np.argwhere(bm)[0]
        raise BannedSlotError(f'{int(bm.sum())} banned pixel(s) after {stage}; first at blend row {row0 + int(r)}, '
                              f'col {col0 + int(c)}, slot {int(blk[r, c, 0])}, ban row {int(lb[r, c])}')


def count_banned(cur, L, allowed, oh, rows=1024):
    """Pixels of a whole north-first blend holding a slot banned (and not whitelisted) for their tile's ban row."""
    ow = cur.shape[1]; n = 0
    tx = (np.arange(ow) + .5) / 4 - .5
    for r0 in range(0, oh, rows):
        r1 = min(oh, r0 + rows)
        ty = (oh - 1 - np.arange(r0, r1) + .5) / 4 - .5
        lb, al = _ban_rows_fields({'L': L, 'allowed': allowed}, tx, ty)
        n += int(_banned(lb, al, cur[r0:r1, :, 0]).sum())
    return n


class BanLedger:
    """The polish-level check: after each later blend writer, the banned pixels it added. strict (the
    TW3K_BAN_ASSERT=strict switch) raises BannedSlotError on the first writer that adds any."""

    def __init__(self, cur, fields, oh, strict=False):
        self.L, self.allowed, self.oh, self.strict = fields['L'], fields.get('allowed'), oh, strict
        self.last = count_banned(cur, self.L, self.allowed, oh); self.report = {'ground materials': self.last}

    def banned(self, mat, r0, c0):
        """bool of a north-first blend window (top-left row r0, col c0) whose material slots `mat` would be banned
        (and not whitelisted) for the window's tiles: what a later writer checks before it writes (town spec)."""
        h, w = mat.shape
        tx = (np.arange(c0, c0 + w) + .5) / 4 - .5
        ty = (self.oh - 1 - np.arange(r0, r0 + h) + .5) / 4 - .5
        lb, al = _ban_rows_fields({'L': self.L, 'allowed': self.allowed}, tx, ty)
        return _banned(lb, al, mat)

    def check(self, cur, writer):
        n = count_banned(cur, self.L, self.allowed, self.oh); self.report[writer] = n - self.last; self.last = n
        if self.strict and self.report[writer] > 0:
            raise BannedSlotError(f'{writer} added {self.report[writer]} banned pixel(s) (ground-materials invariant)')
        return self.report[writer]


class WriterLedger:
    """blend_single_writer baseline (ground contrast fix §1 end state): after the mirror/donor blend, each later
    blend writer's changed material pixels, and per pixel how many writers changed it (uint8 counter). Report:
    {writer: changed px}, plus 'pixels_rewritten' (changed by >= 1 later writer) and 'pixels_rewritten_2plus'."""

    def __init__(self, cur):
        self.prev = cur[..., 0].copy(); self.count = np.zeros(self.prev.shape, np.uint8); self.report = {}

    def check(self, cur, writer):
        ch = cur[..., 0] != self.prev
        self.report[writer] = int(ch.sum()); self.count += ch; self.prev[ch] = cur[..., 0][ch]
        return self.report[writer]

    def finish(self):
        self.report['pixels_rewritten'] = int((self.count >= 1).sum())
        self.report['pixels_rewritten_2plus'] = int((self.count >= 2).sum())
        self.prev = self.count = None
        return self.report


def assert_blend(cur, fields_or_L, oh, allowed=None, stage='later passes', rows=1024):
    """The same invariant over a whole north-first blend after the later passes (town ground, farm fields,
    wetlands, edge cleanup): fields_or_L is tile_fields' result or its 'L' (tile grid). Raises on the first
    banned pixel."""
    L = fields_or_L['L'] if isinstance(fields_or_L, dict) else fields_or_L
    if allowed is None and isinstance(fields_or_L, dict):
        allowed = fields_or_L.get('allowed')
    ow = cur.shape[1]
    tx = (np.arange(ow) + .5) / 4 - .5
    for r0 in range(0, oh, rows):
        r1 = min(oh, r0 + rows)
        ty = (oh - 1 - np.arange(r0, r1) + .5) / 4 - .5
        lb, al = _ban_rows_fields({'L': L, 'allowed': allowed}, tx, ty)
        _assert_no_banned(cur[r0:r1], lb, al, r0, 0, stage)


def _farm_mud_rows(blk, fields, Q, tx, ty, i0, i1, fy, hxv, hyv):
    """Mud, then farms, on the block (see farm_mud_fields). Returns (farm pixels, mud pixels)."""
    f1 = fy[:, None]
    h = _bilinear_rows(fields['M'], tx, ty)[0]; mv = h[i0] * (1 - f1) + h[i1] * f1
    m = (mv > 0) & FARM_SOURCE[blk[..., 0]]
    si, sj = np.nonzero(m); mpx = 0
    if len(si):
        u = Q.uniform('mud', Q.noise_at(('mud',), hxv, hyv, m)[0]) + FARM_FRAY_AMP * (Q.uniform('mfray', Q.noise_at(('mfray',), hxv, hyv, m)[0]) - .5)
        hit = u < mv[si, sj]
        blk[si[hit], sj[hit], 0] = 14; mpx = int(hit.sum())
    h = _bilinear_rows(fields['F'], tx, ty)[0]; fv = h[i0] * (1 - f1) + h[i1] * f1; del h
    m = (fv > 0) & FARM_SOURCE[blk[..., 0]] & (blk[..., 1] != 0)   # no fields in the cold G band
    si, sj = np.nonzero(m)
    if not len(si):
        return 0, mpx
    w = fv[si, sj]
    u = Q.uniform('farm', Q.noise_at(('farm',), hxv, hyv, m)[0]) + FARM_FRAY_AMP * (Q.uniform('fray', Q.noise_at(('fray',), hxv, hyv, m)[0]) - .5)
    hit = u < w
    if not hit.any():
        return 0, mpx
    keep = np.zeros(len(si), bool); keep[hit] = True
    m[si[~keep], sj[~keep]] = False
    si, sj, margin = si[hit], sj[hit], (w - u)[hit]
    udab = Q.uniform('dab', Q.noise_at(('dab',), hxv, hyv, m)[0])
    # kind per pixel: bilinear shares, one kind per coarse-noise blob where they mix
    c0 = sj.min(); h = _bilinear_rows(fields['K'], tx[c0:sj.max() + 1], ty)[0]
    K = h[:, i0[si], sj - c0] * (1 - fy[si]) + h[:, i1[si], sj - c0] * fy[si]; del h
    kind = np.argmax(K, axis=0).astype(np.int8)
    mix = K.max(axis=0) < .999
    if mix.any():
        mm = np.zeros(m.shape, bool); mm[si[mix], sj[mix]] = True
        nz = Q.noise_at(tuple(f'kind{k}' for k in range(len(FARM_KINDS))), hxv, hyv, mm)
        best = np.full(int(mix.sum()), -np.inf, np.float32); pick = np.zeros(int(mix.sum()), np.int8)
        for k in range(len(FARM_KINDS)):
            pk = K[k][mix]; ok = pk > 1e-3
            uk = np.clip(Q.uniform(f'kind{k}', nz[k]), 1e-4, 1 - 1e-6)
            key = np.full(len(pk), -np.inf, np.float32); key[ok] = np.log(uk[ok]) / pk[ok]
            up = key > best; best[up] = key[up]; pick[up] = k
        kind[mix] = pick
    inner = (margin > FARM_RIM) & (udab < np.asarray(FARM_INNER, np.float32)[kind])
    mats = np.asarray(FARM_KINDS, np.uint8)
    blk[si, sj, 0] = mats[kind, np.where(inner, 0, 1)]
    return len(si), mpx

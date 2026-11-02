"""Ambient particle emitters and fauna / living-campaign scenes for the new regions.

Vanilla (190, dlc07) authors ambience as world-space leaves under its town
roots: particle emitters (leaves, blossoms, mist, snow drift, dust ...) and
composite scenes (deer, birds, cows, horses, people working ...). The new
regions had almost none (inventory 2026-10-23: emitters 10.9 vs 0.3 per 1000
land hexes, fauna 6.4 vs 0.6, living 4.9 vs 0.2). This step learns vanilla's
rules from the native leaves of the staged bundle every run and applies them to
the new regions with real geography:

  * habitat: the game climate and ground type (final-map/map.hex), the RESOLVE
    ecoregion biome, ETOPO elevation and latitude (analysis/west-real-terrain.npz,
    biome channel from frontier_terrain.biomes), water (map.hex sea/lake/river
    grounds and the Natural Earth lake mask) and forest (forest ground types; campaign tree density keeps herds and field work out of woods);
  * density: vanilla groups per eligible native hex (same rule, same 2..24 hex
    town ring vanilla uses: every vanilla instance sits 2-24 hexes from a town),
    applied to eligible new-region hexes, capped by BUDGET;
  * spacing: vanilla's 10th-percentile nearest-neighbour distance between groups;
  * groups: a random vanilla group of the family is copied (members, offsets
    rotated about the vertical, 3x3 transforms, height above ground), so herds,
    flocks and emitter pairs keep vanilla's formation and scale.

Assets are vanilla only. Each family writes its own leaf kind, cloned from a
vanilla leaf: emitters keep their family's most common vanilla leaf header (its
season / time-of-day conditions: blossoms spring, leaves_orange autumn, snow
drift winter, fireflies night ...). Scenes use an ungated vanilla scene leaf
header (no building_level / settlement_level conditions; vanilla horses, cows
and workers are gated on their region's resource building, which the new
regions do not share): all five seasons for wild fauna, ha/su/au/sp (no winter,
as vanilla cows, pigs and farm work) for livestock and workers. Records are
vanilla records (template name or path, 3x3 and position replaced, the
template's condition masks kept).

Structure: per (owning region, 16x16-hex cell) one root (identity matrix, so
world_space_leaves.bake skips it; BHM_IGNORE as every vanilla root) with one
parent listing up to 8 leaves, one leaf per family; prefab numbers from
PREFAB0 up. All positions are absolute world coordinates on the polished
surface (analysis/polished-surface-<variant>.npy), as vanilla's BHM_IGNORE
leaves are.

Pipeline: last step of the prefab chain in build_polished_campaign_terrain.ps1,
  @('ambient',@('place_ambient.py','--apply'),@('hehuang-lake','foliage'))
on the staged global_props.bin that reconcile_terrain_prefabs.py rebuilds from
the pre-polish base every run (so it never sees its own earlier output; a
rerun on the same stage replaces its earlier roots).
  reads   work/release-input/terrain/campaigns/<v>/global_props.bin
          work/final-map/map.hex, analysis/west-real-terrain.npz,
          analysis/polished-surface-<v>.npy, work/current-region-design.json (canvas),
          work/release-input/campaign_maps/<v>/display/trees/trees.campaign_tree_list (foliage step)
  writes  work/release-input/terrain/campaigns/<v>/global_props.bin and bmd_objects.bin,
          loose bmd_objects.<region>.<n>.bin parents and bmd_objects.<region>.<n>.<layer>.bin leaves,
          analysis/ambient-placements.json (rules learned, counts, every placed instance)

  python place_ambient.py [--apply] [--seed 190]
"""
import argparse
import collections
import json
import math
import re
import struct
import sys

import numpy as np
try:
    from scipy import ndimage
except ImportError:
    # The pipeline interpreter (Get-Tw3kPython) has no scipy until the switch to the
    # tw3k env is verified; rerun this step in that env (same arguments, same cwd).
    import os
    import subprocess
    TW3K_ENV = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env', 'python-path.txt'), encoding='utf-8').readline().strip()
    if os.path.exists(TW3K_ENV) and os.path.abspath(sys.executable).lower() != TW3K_ENV.lower():
        sys.exit(subprocess.run([TW3K_ENV, '-B', os.path.abspath(__file__), *sys.argv[1:]]).returncode)
    raise

import campaign_world as cw
import map_hex
import props_bundle as pb
from read_root_prefabs import decode
from world_space_leaves import placed_records, _emitter, _scene, _string

A = cw.WORKSPACE
S = A / 'work/release-input'
PREFAB0 = 5500          # vanilla prefab numbers end near 5421. Own range, disjoint from the other prefab
PREFAB1 = 7099          # steps: mountain props 7100-7999, coast waves 8000-8199, Hehuang 9475, landmarks 9600-9799, docks 9800-9999
CELL = 16               # hexes per root cell side
LAYERS = (16, 17, 18, 19, 20, 23, 24, 27, 28, 31)   # vanilla leaf suffixes, unique within a parent
MIN_SPACING = 2.0       # hexes between groups of one family (vanilla herds' p10 is ~1)
MAX_CHILDREN = 8        # vanilla parents hold at most 8 leaves
OPEN_TREES = 10         # campaign trees per 5x5 hexes at most under herds and field work (vanilla medians: farmers 1, cows 4, horses 0)
TOWN_RING = (2, 24)     # every vanilla ambient instance sits 2..24 hexes from a town (p2..p95)
BUDGET_FAMILY = 2.0     # added groups per family <= this x vanilla native groups of that family
BUDGET_TOTAL = 1.0      # added instances in all <= this x vanilla native ambient instances
NEW_ZONES = (1, 2)      # 1 new region outside the native slice, 2 new region inside it
IDENTITY16 = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)

FOREST_G = {'dense_forest', 'light_forest', 'hilly_dense_forest', 'hilly_light_forest'}
OPEN_G = {'grassland', 'plains', 'hilly_grassland', 'hilly_plains'}
HILL_G = {'hilly_grassland', 'hilly_plains', 'hilly_light_forest', 'hilly_dense_forest', 'mountain'}
WATER_G = {'sea_coast', 'sea_ocean', 'sea_lake', 'river', 'river_navigable', 'river_non_navigable',
           'river_navigable_shallows', 'river_navigable_crossing'}
WOOD_B = {'forest', 'temperate', 'conifer', 'taiga', 'tropical', 'tropical_dry', 'mangrove'}
BROADLEAF_B = {'forest', 'temperate', 'conifer'}
TROPIC_B = {'tropical', 'tropical_dry', 'savanna', 'paddy', 'mangrove'}
DRY_B = {'desert', 'steppe', 'alpine', 'savanna', 'grass'}
COLD_B = {'alpine', 'snow', 'tundra'}


def isin(a, s):
    return np.isin(a, sorted(s))


# ------------------------------------------------------------------ families
# learn: vanilla family keys (vanilla_family()); header: 'own' (the family's own
# most common vanilla leaf header), 'all5' or 'warm4' (ungated scene headers).
# rule(C) -> bool hex mask. weight(C) -> sampling weight (optional).
FAMILIES = {
    # particle emitters ---------------------------------------------------
    'leaves': dict(learn=['emit:leaves'], header='own',
                   rule=lambda C: isin(C.bio, WOOD_B) & isin(C.g, FOREST_G | OPEN_G | {'mountain'}) & isin(C.clim, {'temperate', 'subtropical', 'cold'})),
    'leaves_orange': dict(learn=['emit:leaves_orange'], header='own',
                          rule=lambda C: isin(C.bio, BROADLEAF_B | {'taiga'}) & isin(C.g, FOREST_G | OPEN_G | {'mountain'}) & isin(C.clim, {'temperate', 'cold'})),
    'leaves_yellow': dict(learn=['emit:leaves_yellow'], header='own',
                          rule=lambda C: isin(C.bio, BROADLEAF_B | {'taiga'}) & isin(C.g, FOREST_G | OPEN_G | {'mountain'}) & isin(C.clim, {'temperate', 'cold', 'arid'})),
    'blossoms': dict(learn=['emit:blossoms'], header='own',
                     rule=lambda C: isin(C.bio, BROADLEAF_B) & isin(C.g, FOREST_G | OPEN_G | {'mountain'}) & isin(C.clim, {'temperate', 'cold'}) & (C.elev < 1800)),
    'pollen': dict(learn=['emit:pollen'], header='own',
                   rule=lambda C: isin(C.bio, BROADLEAF_B) & isin(C.g, FOREST_G | OPEN_G) & (C.clim == 'temperate') & (C.elev < 800)),
    'mist': dict(learn=['emit:mist'], header='own',
                 rule=lambda C: isin(C.g, {'mountain', 'hilly_light_forest', 'hilly_dense_forest'}) & isin(C.bio, WOOD_B) & isin(C.clim, {'subtropical', 'cold', 'temperate'})),
    'fireflies': dict(learn=['emit:fireflies'], header='own',
                      rule=lambda C: ((C.dwater <= 6) | C.wet) & isin(C.clim, {'temperate', 'subtropical'}) & ~isin(C.bio, DRY_B | COLD_B | {'desert'}) & (C.elev < 1200)),
    'butterflies': dict(learn=['emit:butterflies'], header='own',
                        rule=lambda C: (C.clim == 'subtropical') & isin(C.bio, TROPIC_B) & isin(C.g, FOREST_G | {'mountain'})),
    'flies': dict(learn=['emit:flies'], header='own',
                  rule=lambda C: (C.clim == 'subtropical') & isin(C.bio, TROPIC_B) & isin(C.g, FOREST_G | OPEN_G | {'mountain'})),
    'snow_drift': dict(learn=['emit:snow_drift'], header='own',
                       rule=lambda C: isin(C.g, HILL_G) & (((C.elev >= 1100) & isin(C.clim, {'arid', 'cold'})) | isin(C.bio, COLD_B))),
    'snow_drift_single': dict(learn=['emit:snow_drift_single'], header='own',
                              rule=lambda C: isin(C.g, OPEN_G) & isin(C.clim, {'temperate', 'cold', 'arid'}) & (C.lat >= 33) & (C.elev < 1600) & (C.bio != 'desert')),
    'snow_drift_light': dict(learn=['emit:snow_drift_light'], header='own',
                             rule=lambda C: isin(C.g, OPEN_G | FOREST_G) & isin(C.clim, {'temperate', 'cold'}) & (C.lat >= 33) & (C.elev < 600)),
    'wind': dict(learn=['emit:wind'], header='own',
                 rule=lambda C: isin(C.g, HILL_G) & (C.elev >= 900) & isin(C.clim, {'arid', 'temperate', 'cold'}) & ~isin(C.bio, TROPIC_B)),
    'dust_single': dict(learn=['emit:dust_single'], header='own',
                        rule=lambda C: (C.clim == 'arid') & isin(C.g, OPEN_G | {'desert'}) & isin(C.bio, DRY_B | {'desert'})),
    'dust_cloud': dict(learn=['emit:dust_cloud'], header='own',
                       rule=lambda C: (C.clim == 'arid') & isin(C.g, OPEN_G | {'desert'}) & isin(C.bio, DRY_B | {'desert'})),
    'heat_haze': dict(learn=['emit:heat_haze'], header='own',
                      rule=lambda C: (C.clim == 'arid') & isin(C.g, OPEN_G | {'desert'}) & (C.elev < 2500)),
    # fauna ---------------------------------------------------------------
    'deer': dict(learn=['fauna:deer'], header='all5',
                 rule=lambda C: isin(C.clim, {'temperate', 'cold'}) & isin(C.bio, BROADLEAF_B | {'taiga'}) & isin(C.g, FOREST_G | OPEN_G)),
    'pigs': dict(learn=['fauna:pig'], header='warm4',
                 rule=lambda C: (C.clim == 'subtropical') & isin(C.bio, TROPIC_B | {'forest'}) & isin(C.g, FOREST_G | OPEN_G)),
    'cows': dict(learn=['fauna:cow'], header='warm4',
                 rule=lambda C: isin(C.g, OPEN_G) & isin(C.clim, {'temperate', 'subtropical'}) & ~isin(C.bio, COLD_B | {'desert'}) & (C.dtown <= 19) & (C.tdens <= OPEN_TREES)),
    'horses': dict(learn=['fauna:horse'], header='all5',
                   rule=lambda C: isin(C.g, OPEN_G | {'desert'}) & isin(C.bio, {'steppe', 'alpine', 'grass', 'savanna'}) & isin(C.clim, {'arid', 'temperate', 'cold'}) & (C.tdens <= OPEN_TREES)),
    'cranes': dict(learn=['fauna:crane'], header='all5',
                   rule=lambda C: (C.wet | (C.dwater <= 4)) & isin(C.clim, {'temperate', 'subtropical', 'cold'}) & ~isin(C.g, WATER_G)),
    'birds': dict(learn=['fauna:single_circle_1', 'fauna:single_circle_5', 'fauna:single_circle_8', 'fauna:single_circle_10',
                         'fauna:single_large_circle_1', 'fauna:single_large_circle_2'], header='all5',
                  rule=lambda C: C.land.copy(),
                  # birds over lakes and wetlands: 3x the draw weight within 3 hexes of water or on wetland
                  weight=lambda C: np.where(C.wet | C.lake3, 3., 1.)),
    # living campaign -----------------------------------------------------
    'farmers': dict(learn=['living:planting_man', 'living:shovel_man', 'living:rake_man', 'living:sieve_rice'], header='warm4',
                    rule=lambda C: isin(C.g, OPEN_G) & (C.dtown <= 18) & ~isin(C.bio, COLD_B) & ((C.clim != 'arid') | (C.dwater <= 6)) & (C.tdens <= OPEN_TREES)),
    'woodcutters': dict(learn=['living:chopping_man', 'living:chopping_man_2', 'living:axe_man', 'living:saw_man', 'living:double_sawing_men'], header='warm4',
                        rule=lambda C: isin(C.g, FOREST_G | {'mountain'}) & isin(C.bio, WOOD_B) & (C.dtown <= 20)),
    'fishers': dict(learn=['living:fisher_man'], header='warm4', on_water=True,
                    rule=lambda C: (C.g == 'sea_coast') & (C.dland <= 3) & (C.dtown <= 16)),
    'lookouts': dict(learn=['living:lookout_man'], header='all5',
                     rule=lambda C: isin(C.g, OPEN_G | {'mountain'}) & (C.dtown <= 20)),
}
# Vanilla families not placed: lookout_military (stands on town walls/towers: town
# dressing), panda (Sichuan only), waterfall(_splashes) (needs a river cliff),
# clouds / cloud_peaks / clouds_static / snow_drift_wide (one-off landmarks),
# and the rare workers (mining, hammer, organizing, rope_pulling, sieve_plants: <15 each).


def vanilla_family(kind, model):
    p = model.replace(chr(92), '/').lower()
    if kind == 'emitter':
        return 'emit:' + p.replace('3k_main_campaign_enviro_', '')
    s = p.split('/')
    f = s[-1].split('.')[0]
    if len(s) > 3 and s[2] == 'fauna':
        for a in ('cow', 'horse', 'deer', 'doe', 'pig', 'panda', 'crane'):
            if a in f:
                return 'fauna:' + ('deer' if a == 'doe' else a)
        return 'fauna:' + f
    if len(s) > 3 and s[2] == 'living_campaign':
        return 'living:' + f
    return None


def conditions(head):
    keys = {m.group(1).decode() for m in re.finditer(rb'[\x02-\x30]\x00([a-z_0-9]{2,})', head)}
    return keys


# ------------------------------------------------------------------ context
class Context:
    """Per-hex habitat fields on the canvas (rows south-up, as map.hex)."""

    def __init__(self):
        m = map_hex.read(A / 'work/final-map/map.hex')
        c = m['cells'].astype(np.int32)
        H, W = c.shape[:2]
        names = m['land_ground_types'] + m['sea_ground_types']
        g = (c[..., 5] >> 4) + ((c[..., 6] & 7) << 4) - 1
        self.g = np.array(names + ['?'], object)[np.where((g >= 0) & (g < len(names)), g, len(names))]
        self.land = (g >= 0) & (g < 14)
        cl = (c[..., 7] >> 1) - 1
        self.clim = np.array(m['climates'] + ['-'], object)[np.where(cl >= 0, cl, len(m['climates']))]
        ids = map_hex.region_ids(c)
        regions = m['land_regions'] + m['sea_regions']
        idc = np.where((ids >= 0) & (ids < len(regions)), ids, len(regions))
        self.region = np.array(regions + [''], object)[idc]
        new = np.array([n.startswith('tw3k_') for n in regions] + [False])[idc]
        npl = np.array(['non_playable' in n for n in regions] + [False])[idc]
        inside = np.zeros((H, W), bool)
        inside[cw.NATIVE_HEX_SLICE] = True
        self.zone = np.full((H, W), 3, np.int8)
        self.zone[inside & ~new & ~npl] = 0
        self.zone[~inside & new] = 1
        self.zone[inside & new] = 2
        town = (c[..., 2] >> 4) == 1
        self.dtown = ndimage.distance_transform_edt(~town)
        R = np.load(A / 'analysis/west-real-terrain.npz')
        bn = list(R['biome_names']) + ['other']
        qq, rr = np.meshgrid(np.arange(W), np.arange(H))
        ty = np.clip(2 * rr + (qq % 2), 0, R['biome'].shape[0] - 1)
        tx = np.clip(2 * qq, 0, R['biome'].shape[1] - 1)
        b = R['biome'][ty, tx]
        self.bio = np.array(bn, object)[np.minimum(b, len(bn) - 1)]
        self.elev = R['elevation'][ty, tx]
        self.lat = np.nan_to_num(R['lat'][ty, tx], nan=35.)
        lake = R['lake'][ty, tx] | (self.g == 'sea_lake')
        water = isin(self.g, WATER_G) | lake
        self.wet = (self.g == 'wetland') | (self.bio == 'wetland')
        self.dwater = ndimage.distance_transform_edt(~water)
        self.lake3 = ndimage.distance_transform_edt(~(lake | isin(self.g, {'river_navigable', 'river_non_navigable', 'river'}))) <= 3
        self.dland = ndimage.distance_transform_edt(~self.land)
        self.ring = (self.dtown >= TOWN_RING[0]) & (self.dtown <= TOWN_RING[1])
        # campaign trees (the foliage step's tree list) per 5x5 hexes: open-country scenes avoid woods
        trees = np.zeros((H, W), np.float32)
        tl = S / f'campaign_maps/{cw.TERRAIN_VARIANTS[0]}/display/trees/trees.campaign_tree_list'
        if tl.exists():
            blob, pos, xs, zs = tl.read_bytes(), 24, [], []
            for _ in range(struct.unpack_from('<I', blob, 20)[0]):
                n = struct.unpack_from('<H', blob, pos)[0]
                pos += 2 + n
                count = struct.unpack_from('<I', blob, pos)[0]
                pos += 4
                for _ in range(count):
                    x, _, z = struct.unpack_from('<3f', blob, pos)
                    pos += 18 + struct.unpack_from('<I', blob, pos + 14)[0] * 4
                    xs.append(x)
                    zs.append(z)
            tq, tr = cw.world_hex(np.array(xs), np.array(zs))
            ok = (tq >= 0) & (tq < W) & (tr >= 0) & (tr < H)
            np.add.at(trees, (tr[ok], tq[ok]), 1)
        self.tdens = ndimage.uniform_filter(trees, 5) * 25
        # owner of a water hex: the region of the nearest land hex (vanilla fishers belong to their town's region)
        _, (iy, ix) = ndimage.distance_transform_edt(~self.land, return_indices=True)
        self.land_owner = self.region[iy, ix]
        self.land_zone = self.zone[iy, ix]
        self.shape = (H, W)


# ------------------------------------------------------------------ bundle
def load_bundle(variant):
    pre = f'terrain/campaigns/{variant}/'
    rootkey = pre + 'bmd_objects.bin'
    es, assets = pb.unbundle((S / (pre + 'global_props.bin')).read_bytes(), rootkey)
    return pre, es, assets, rootkey


def record_start(r):
    return r['position_offset'] - (40 + len(r['model'])) if r['kind'] == 'emitter' else r['position_offset'] - 38


def leaf_parts(b, kind):
    """(head incl. list version, [(start, end)] of the kind's records, tail) of a single-kind leaf."""
    rows = [r for r in placed_records(b) if r['kind'] == kind]
    parse_one = _emitter if kind == 'emitter' else _scene
    spans = []
    for r in sorted(rows, key=record_start):
        s = record_start(r)
        rec, end = parse_one(b, s)
        spans.append((s, end))
    return b[:spans[0][0] - 4], spans, b[spans[-1][1]:]


def emitter_record(name, matrix, post):
    n = name.encode()
    return struct.pack('<HH', 6, len(n)) + n + struct.pack('<12f', *matrix) + post


def scene_record(path, matrix, trailer):
    p = path.encode()
    mode = b'BHM_IGNORE'
    return struct.pack('<H', 6) + struct.pack('<12f', *matrix) + struct.pack('<H', len(p)) + p + struct.pack('<H', len(mode)) + mode + trailer


def make_leaf(head, records, tail):
    return head + struct.pack('<I', len(records)) + b''.join(records) + tail


# ------------------------------------------------------------------ learning
def learn(assets, rootkey, C, surface):
    """Vanilla instances, groups (one leaf = one group) and leaf headers per vanilla family, native zone only."""
    inst = collections.defaultdict(list)
    heads = collections.defaultdict(collections.Counter)
    sample = {}
    H, W = C.shape
    for k, b in assets.items():
        if k == rootkey:
            continue
        try:
            rows = placed_records(b)
        except Exception:
            continue
        if not rows:
            continue
        kinds = {r['kind'] for r in rows}
        for r in rows:
            f = vanilla_family(r['kind'], r['model'])
            if f is None:
                continue
            m = r['matrix']
            q, rr = cw.world_hex(m[9], m[11])
            q, rr = int(q), int(rr)
            if not (0 <= q < W and 0 <= rr < H) or C.zone[rr, q] != 0:
                continue
            dy = m[10] - surf_at(surface, m[9], m[11])
            inst[f].append(dict(leaf=k, model=r['model'], m3=m[:9], x=m[9], y=m[10], z=m[11], dy=float(dy), q=q, r=rr))
        if len(kinds) == 1:
            kind = kinds.pop()
            fams = {vanilla_family(r['kind'], r['model']) for r in rows}
            head, spans, tail = leaf_parts(b, kind)
            key = (kind, head, tail, b[spans[0][0]:spans[0][1]])
            if len(fams) == 1:
                f = fams.pop()
                heads[f][(head, tail)] += 1
                sample.setdefault((f, head, tail), (k, spans))
            heads['*' + kind][(head, tail)] += 1
            sample.setdefault(('*' + kind, head, tail), (k, spans))
    return inst, heads, sample


def surf_at(surface, x, z):
    tx, ty = x / cw.DX * 2, z / cw.DZ * 2
    h, w = surface.shape
    x_, y_ = min(max(tx, 0), w - 1), min(max(ty, 0), h - 1)
    x0, y0 = int(x_), int(y_)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x_ - x0, y_ - y0
    return (float(surface[y0, x0]) * (1 - fx) + float(surface[y0, x1]) * fx) * (1 - fy) + \
           (float(surface[y1, x0]) * (1 - fx) + float(surface[y1, x1]) * fx) * fy


def templates(assets, heads, sample):
    """Leaf header/tail and record bytes per family."""
    out = {}
    seasons5 = {'ha', 'au', 'wi', 'sp', 'su'}
    for name, fam in FAMILIES.items():
        if fam['header'] == 'own':
            vf = fam['learn'][0]
            # the most common ungated condition class of the family's leaves, then its most common bytes
            classes = collections.Counter()
            for (head, tail), n in heads[vf].items():
                cd = frozenset(conditions(head))
                if not any(c.startswith(('building', 'settlement', 'campaign_map_object')) for c in cd):
                    classes[cd] += n
            best = classes.most_common(1)[0][0]
            (head, tail), n = max(((ht, n) for ht, n in heads[vf].items() if frozenset(conditions(ht[0])) == best), key=lambda t: t[1])
            k, spans = sample[(vf, head, tail)]
            b = assets[k]
            s, e = spans[0]
            nlen = struct.unpack_from('<H', b, s + 2)[0]
            post = b[s + 4 + nlen + 48:e]
            out[name] = dict(kind='emitter', head=head, tail=tail, post=post, source=k, header_leaves=n,
                             conditions=sorted(conditions(head) & (seasons5 | {'day', 'night', 'time_of_day'})))
        else:
            want = seasons5 if fam['header'] == 'all5' else {'ha', 'su', 'au', 'sp'}
            best = None
            for (head, tail), n in heads['*scene'].most_common():
                cd = conditions(head)
                if any(c.startswith(('building', 'settlement', 'campaign_map_object', 'time_of_day')) for c in cd):
                    continue
                if (cd & seasons5) != want:
                    continue
                best = (head, tail, n)
                break
            head, tail, n = best
            k, spans = sample[('*scene', head, tail)]
            b = assets[k]
            s, e = spans[0]
            trailer = b[e - 35:e]
            out[name] = dict(kind='scene', head=head, tail=tail, trailer=trailer, source=k, header_leaves=n,
                             conditions=sorted(conditions(head) & seasons5))
    return out


# ------------------------------------------------------------------ placement
def poisson(rng, cand, weight, n, spacing, occupied=()):
    """Up to n hexes from cand ((k, 2) rows of (r, q)) drawn by weight, no two closer than spacing."""
    if n <= 0 or not len(cand):
        return []
    # weighted random order without replacement (Efraimidis-Spirakis keys)
    order = np.argsort(np.log(rng.random(len(cand))) / np.asarray(weight, float))[::-1]
    cell = max(spacing, 1.)
    grid = collections.defaultdict(list)
    for (r, q) in occupied:
        grid[(int(r // cell), int(q // cell))].append((r, q))
    out = []
    s2 = spacing * spacing
    for i in order:
        r, q = cand[i]
        gr, gq = int(r // cell), int(q // cell)
        ok = True
        for dr in (-1, 0, 1):
            for dq in (-1, 0, 1):
                for (r2, q2) in grid.get((gr + dr, gq + dq), ()):
                    if (r - r2) ** 2 + (q - q2) ** 2 < s2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            out.append((int(r), int(q)))
            grid[(gr, gq)].append((r, q))
            if len(out) >= n:
                break
    return out


def groups_of(instances):
    g = collections.defaultdict(list)
    for i in instances:
        g[i['leaf']].append(i)
    return list(g.values())


def nn_p10(points):
    from scipy.spatial import cKDTree
    if len(points) < 3:
        return 6.
    P = np.array(points, float)
    d = cKDTree(P).query(P, 2)[0][:, 1]
    return float(np.percentile(d, 10))


def run(apply, seed):
    C = Context()
    report = {'variants': [], 'apply': apply, 'seed': seed, 'live_verified': False}
    for variant in cw.TERRAIN_VARIANTS:
        surface = np.load(A / f'analysis/polished-surface-{variant}.npy', mmap_mode='r')
        pre, es, assets, rootkey = load_bundle(variant)
        root = assets[rootkey]
        cp, rs = decode(root)
        # A rerun on the same stage replaces this step's earlier roots.
        mine = re.compile(re.escape(pre) + r'bmd_objects\.[^/.]+\.(\d+)\.')
        def ours(path):
            m = mine.match(path)
            return bool(m) and PREFAB0 <= int(m.group(1)) <= PREFAB1 and path.count('.') in (3, 4)
        stale = [k for k in assets if ours(k)]
        for k in stale:
            assets.pop(k)
        rs_kept = [r for r in rs if not ours(r['path'])]
        inst, heads, sample = learn(assets, rootkey, C, surface)
        T = templates(assets, heads, sample)
        # Root / parent templates: a vanilla root whose parent has exactly one leaf.
        tpl = next(r for r in rs_kept if r['path'] in assets and len(decode(assets[r['path']])[1]) == 1
                   and r['height_mode'] == 'BHM_IGNORE' and not r['region'].startswith('tw3k_'))
        pbody = assets[tpl['path']]
        pcp, pch = decode(pbody)
        child_tpl = pbody[pch[0]['start']:pch[0]['end']]
        rng = np.random.default_rng(seed)
        occupied_all = collections.defaultdict(list)
        # existing new-zone instances of each vanilla family keep their place
        for k, b in assets.items():
            try:
                rows = placed_records(b)
            except Exception:
                continue
            for r in rows:
                f = vanilla_family(r['kind'], r['model'])
                q, rr = cw.world_hex(r['matrix'][9], r['matrix'][11])
                if f and 0 <= rr < C.shape[0] and 0 <= q < C.shape[1] and C.zone[int(rr), int(q)] in NEW_ZONES:
                    occupied_all[f].append((int(rr), int(q)))
        newmask = isin(C.zone, NEW_ZONES) & C.land
        natmask = (C.zone == 0) & C.land
        plan, rules = [], {}
        vanilla_total = sum(len(inst[v]) for fam in FAMILIES.values() for v in fam['learn'])
        for name, fam in FAMILIES.items():
            I = [i for v in fam['learn'] for i in inst[v]]
            G = groups_of(I)
            rule = fam['rule'](C) & C.ring
            # groups copied from vanilla: those inside the rule (fishers on the sea, not on rivers)
            Gin = [g for g in G if rule[g[0]['r'], g[0]['q']]] or G
            if fam.get('on_water'):
                elig_nat, elig_new = rule & (C.land_zone == 0), rule & isin(C.land_zone, NEW_ZONES)
            else:
                elig_nat, elig_new = rule & natmask, rule & newmask
            in_rule = sum(1 for i in I if rule[i['r'], i['q']])
            rate = len(G) / max(1, int(elig_nat.sum()))
            spacing = max(MIN_SPACING, nn_p10([(np.mean([i['r'] for i in g]), np.mean([i['q'] for i in g])) for g in G]))
            target = rate * int(elig_new.sum())
            cap = BUDGET_FAMILY * len(G)
            n = int(round(min(target, cap)))
            cand = np.argwhere(elig_new)
            w = fam['weight'](C)[elig_new] if 'weight' in fam else np.ones(len(cand))
            occ = [p for v in fam['learn'] for p in occupied_all[v]]
            centres = poisson(np.random.default_rng([seed, len(plan), sum(map(ord, name))]), cand, w, n, spacing, occ)
            rules[name] = dict(vanilla_families=fam['learn'], vanilla_instances=len(I), vanilla_groups=len(G),
                               vanilla_group_size_mean=round(len(I) / max(1, len(G)), 2),
                               vanilla_in_rule_share=round(in_rule / max(1, len(I)), 3),
                               eligible_native_hexes=int(elig_nat.sum()), groups_per_1000_eligible=round(1000 * rate, 2),
                               spacing_hexes=round(spacing, 1), eligible_new_hexes=int(elig_new.sum()),
                               target_groups=round(target, 1), budget_groups=cap, placed_groups=len(centres),
                               header=fam['header'], template_leaf=T[name]['source'], leaf_conditions=T[name]['conditions'])
            frng = np.random.default_rng([seed, sum(map(ord, name)) * 7 + 1])
            for (r, q) in centres:
                g = Gin[frng.integers(len(Gin))]
                cx = np.mean([i['x'] for i in g]); cz = np.mean([i['z'] for i in g])
                x0, z0 = cw.world_position(q, r)
                bx, bz = x0 + frng.uniform(.15, .85) * cw.DX, z0 + frng.uniform(.15, .85) * cw.DZ
                a = frng.uniform(0, 2 * math.pi); ca, sa = math.cos(a), math.sin(a)
                members = []
                for i in g:
                    ox, oz = i['x'] - cx, i['z'] - cz
                    x, z = bx + ca * ox - sa * oz, bz + sa * ox + ca * oz
                    if fam.get('on_water'):
                        y = i['y']          # vanilla fishers stand on the water plane (absolute height)
                    else:
                        # emitters never below the new ground (vanilla snow drift sits ~1 under its mountain models)
                        y = surf_at(surface, x, z) + (max(i['dy'], 0.) if T[name]['kind'] == 'emitter' else i['dy'])
                    # members stay in the group's own region and on its kind of ground (a group can span a few hexes)
                    mq, mr = (int(v) for v in cw.world_hex(x, z))
                    if not (0 <= mq < C.shape[1] and 0 <= mr < C.shape[0]):
                        continue
                    if fam.get('on_water'):
                        if C.g[mr, mq] != 'sea_coast' or C.land_owner[mr, mq] != C.land_owner[r, q]:
                            continue
                    elif not C.land[mr, mq] or C.region[mr, mq] != C.region[r, q] or C.zone[mr, mq] not in NEW_ZONES:
                        continue
                    members.append(dict(model=i['model'], m3=list(i['m3']), x=x, y=y, z=z))
                if not members:
                    continue
                owner = C.land_owner[r, q] if fam.get('on_water') else C.region[r, q]
                plan.append(dict(family=name, kind=T[name]['kind'], q=q, r=r, region=str(owner), members=members))
        # global budget: drop whole groups at random, largest families first, until within BUDGET_TOTAL
        total = sum(len(p['members']) for p in plan)
        limit = BUDGET_TOTAL * vanilla_total
        if total > limit:
            order = rng.permutation(len(plan))
            keep = np.ones(len(plan), bool)
            for i in order:
                if total <= limit:
                    break
                keep[i] = False
                total -= len(plan[i]['members'])
            plan = [p for p, k in zip(plan, keep) if k]
        # ---------------------------------------------------------- build roots / parents / leaves
        bycell = collections.defaultdict(lambda: collections.defaultdict(list))
        for p in plan:
            bycell[(p['region'], p['q'] // CELL, p['r'] // CELL)][p['family']].append(p)
        used = {r['path'] for r in rs_kept} | set(assets)
        number = PREFAB0
        new_assets, new_roots, placed = {}, [], []
        for (region, cq, cr), fams in sorted(bycell.items()):
            fam_names = sorted(fams)
            for chunk in range(0, len(fam_names), MAX_CHILDREN):
                names = fam_names[chunk:chunk + MAX_CHILDREN]
                while True:
                    parent = f'{pre}bmd_objects.{region}.{number}.bin'
                    number += 1
                    if parent not in used:
                        break
                assert number <= PREFAB1, 'ambient prefab numbers exhausted'
                children = []
                for layer, fname in zip(LAYERS, names):
                    t = T[fname]
                    recs = []
                    for p in fams[fname]:
                        for mbr in p['members']:
                            mat = tuple(mbr['m3']) + (mbr['x'], mbr['y'], mbr['z'])
                            if t['kind'] == 'emitter':
                                recs.append(emitter_record(mbr['model'], mat, t['post']))
                            else:
                                recs.append(scene_record(mbr['model'], mat, t['trailer']))
                            placed.append(dict(family=fname, model=mbr['model'].split('/')[-1], x=float(np.float32(mbr['x'])), y=float(np.float32(mbr['y'])),
                                               z=float(np.float32(mbr['z'])), q=p['q'], r=p['r'], region=region))
                    leaf = make_leaf(t['head'], recs, t['tail'])
                    path = parent[:-4] + f'.{layer}.bin'
                    rows = placed_records(leaf)
                    assert len(rows) == len(recs) and all(r['kind'] == t['kind'] for r in rows), ('leaf round trip', path)
                    new_assets[path] = leaf
                    children.append(pb.rebind(child_tpl, pch[0], region, path))   # the template child, region and path replaced
                body, chk = pb.prefab_list(pbody, pcp, pch, [], children)
                assert [c['path'] for c in chk] == [parent[:-4] + f'.{l}.bin' for l in LAYERS[:len(names)]]
                assert all(c['region'] == region for c in chk)
                new_assets[parent] = body
                rec = bytearray(pb.rebind(root[tpl['start']:tpl['end']], tpl, region, parent))
                struct.pack_into('<16f', rec, 4 + len(parent), *IDENTITY16)   # u16 kind, u16 length, path, 16f
                new_roots.append(bytes(rec))
        updated, chk = pb.prefab_list(root, cp, rs, rs_kept, new_roots)
        for r in chk[len(rs_kept):]:
            assert r['matrix'] == IDENTITY16 and r['region'].startswith('tw3k_') and r['path'] in new_assets, r['path']
        bundled = pb.rebundle(es, assets, rootkey, updated, new_assets)
        counts = collections.Counter(p['family'] for p in placed)
        stale_loose = [str(p) for p in (S / pre).glob('bmd_objects.*.bin')
                       if ours(pre + p.name) and (pre + p.name) not in new_assets]
        report['variants'].append(dict(variant=variant, roots_added=len(new_roots), leaves_added=sum(1 for k in new_assets if k.count('.') == 4),
                                       instances_added=len(placed), instances_by_family=dict(sorted(counts.items())),
                                       groups_by_family=dict(sorted(collections.Counter(p['family'] for p in plan).items())),
                                       vanilla_native_ambient_instances=vanilla_total, replaced_earlier_roots=len(stale),
                                       stale_loose_files=len(stale_loose), stale_loose_sample=stale_loose[:5], rules=rules,
                                       new_land_hexes=int(newmask.sum()), native_land_hexes=int(natmask.sum()),
                                       bundle_assets=struct.unpack_from('<I', bundled)[0], placed=placed))
        print(variant, 'ambient: roots', len(new_roots), 'instances', len(placed), dict(sorted(counts.items())),
              'applied' if apply else 'dry run', flush=True)
        if apply:
            pb.write_stage(S, pre, bundled, updated, new_assets)
    out = A / 'analysis/ambient-placements.json'
    if apply:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--seed', type=int, default=190)
    a = ap.parse_args()
    run(a.apply, a.seed)


if __name__ == '__main__':
    main()

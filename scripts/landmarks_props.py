"""Landmark props: vanilla area-of-interest models in new world-space leaves (landmarks_spec.py).

Vanilla AOI props are ordinary prefab scenes: a root record in the root prefab list
(terrain/campaigns/<map>/bmd_objects.bin, region-keyed), a parent
bmd_objects.<region>.<n>.bin with one child, and the leaf bmd_objects.<region>.<n>.31.bin
(FASTBIN0 v35) whose model table lists rigidmodels/campaign/area_of_interest/... and whose
instances carry absolute world matrices (3x3 rows = the model's x/y/z axes, then the
position; y absolute, BHM_IGNORE; leaf flag byte 1 = world space). Example: the tower
bmd_objects.3k_main_anding_resource_1.2281(.31).bin, the TEMPLATE here.

This step builds one such scene per (landmark, owning region), prefab numbers from
PREFAB_BASE, on the staged global_props.bin, like hehuang_lake_prop.py:
  * walls (河西汉塞, 居延塞): great_wall_straight pieces (4 x 2 model units, x along the
    wall, centre line at local z -1, scale 0.21 as vanilla: 0.84 world units each) chained
    along the wall line, a great_wall_tower_1 at every bend and every TOWER_EVERY pieces
    (烽燧); no piece within SETTLEMENT_CLEAR hexes of a settlement slot or ending inside a pass;
  * passes (玉门关, 阳关): a tower on the site (玉门关 snapped onto the 河西 line) and
    PASS_PIECES straight pieces each side;
  * 白龙堆: YARDANG_COUNT stone_forest pillars YARDANG_SPACING apart, flattened (scale 0.42/0.2/0.3) and turned
    north-east/south-west like the Lop Nur yardangs, seeded RNG;
  * mountains, lakes and sites get no props (no shrine .wsmodel).
Every instance sits on the polished surface (analysis/polished-surface-<map>.npy, the
heights resource-scenes seats towns with): y = lowest surface under the piece minus a
small sink, as vanilla pieces sit within ~0.3 of it. Each instance record is a byte copy
of a vanilla record of the same model (world-space flag, BHM_IGNORE) with its model
index and matrix rewritten; header and tail of the leaf are the template's.

Prefab-chain node 'landmark-props' in build_polished_campaign_terrain.ps1, after
'hehuang-lake' (both rewrite global_props.bin; it must follow world-space-leaves and
lake-leaves, which would otherwise shift or re-translate leaves).
Reads:  <stage>/terrain/campaigns/<map>/global_props.bin (staged by reconcile_terrain_prefabs,
        world-space leaves already baked), <workspace>/analysis/polished-surface-<map>.npy,
        <workspace>/work/final-map/map.hex (owning regions, settlement slots, water)
Writes: with --apply the staged global_props.bin, the loose root list, parents and leaves
        under <stage>, and analysis/landmark-props.json (next to --stage when given).
A rerun replaces its own scenes (prefab numbers PREFAB_BASE..PREFAB_MAX).

  python landmarks_props.py [--apply] [--stage DIR] [--map MAP] [--surface NPY]
"""
import argparse
import json
import math
import random
import re
import struct
from pathlib import Path

import numpy as np

import campaign_world as cw
import map_hex
import landmarks_spec as spec
import props_bundle as pb
from read_root_prefabs import decode
from translate_campaign_lake_instances import records as leaf_records, MODEL
from world_space_leaves import is_world_space

A = cw.WORKSPACE
STAGE = A / 'work/release-input'
TEMPLATE_REGION = '3k_main_anding_resource_1'
TEMPLATE = 2281
PREFAB_BASE, PREFAB_MAX = 9600, 9799
AOI = 'rigidmodels/campaign/area_of_interest/'
STRAIGHT = AOI + 'great_wall/great_wall_straight.rigid_model_v2'
TOWER = AOI + 'great_wall/great_wall_tower_1.rigid_model_v2'
PILLARS = [AOI + f'stone_forest/stone_forest_pillar_{i}.rigid_model_v2' for i in range(1, 6)]
WALL_SCALE = .21
PIECE_UNITS = 4.0                 # great_wall_straight: x 0..4, z -2..0 (model bbox)
TOWER_EVERY = 12                  # straight pieces between beacon towers on a straight run
SAMPLE = .05                      # world units between blocking tests along the wall line
SETTLEMENT_CLEAR = 1.6            # hexes kept free of wall round a settlement slot
PASS_PIECES = 2
# world distance from a pass site that its own tower and pieces cover (no wall piece ends inside)
PASS_EXTENT = (1.32 + PASS_PIECES * PIECE_UNITS) * WALL_SCALE + .02
WALL_SINK, TOWER_SINK = .03, .05
YARDANG_COUNT = 18
YARDANG_SPACING = .6               # world units between pillar origins
YARDANG_SCALE = (.42, .2, .3)
YARDANG_SINK = .15
YARDANG_HEADING = math.radians(45)  # long axis NE-SW
YARDANG_WALL_CLEAR = 1.5            # hexes kept between pillars and any wall line (DUNHUANG YARDANG)


# --- leaf format -------------------------------------------------------------

def split_leaf(b):
    """(head, model list, [record bytes], tail) of a FASTBIN0 leaf (see leaf_records)."""
    for m in MODEL.finditer(b):
        s = m.start()
        if s >= 8 and struct.unpack_from('<H', b, s - 8)[0] == 2 and struct.unpack_from('<H', b, s - 2)[0] == len(m.group()) \
                and 0 < struct.unpack_from('<I', b, s - 6)[0] < 10000:
            break
    else:
        raise ValueError('no model table')
    n = struct.unpack_from('<I', b, s - 6)[0]
    pos, models = s - 2, []
    for _ in range(n):
        le = struct.unpack_from('<H', b, pos)[0]
        models.append(b[pos + 2:pos + 2 + le].decode())
        pos += 2 + le
    count = struct.unpack_from('<I', b, pos)[0]
    p, recs = pos + 4, []
    for _ in range(count):
        le = struct.unpack_from('<H', b, p + 105)[0]
        recs.append(b[p:p + 131 + le])
        p += 131 + le
    return b[:s - 6], models, recs, b[p:]


def build_leaf(head, tail, templates, instances):
    """instances: [(model, 12 floats)]; templates: {model: record bytes}."""
    models = sorted({m for m, _ in instances})
    out = bytearray(head) + struct.pack('<I', len(models))
    for m in models:
        e = m.encode()
        out += struct.pack('<H', len(e)) + e
    out += struct.pack('<I', len(instances))
    for m, matrix in instances:
        rec = bytearray(templates[m])
        struct.pack_into('<I', rec, 2, models.index(m))
        struct.pack_into('<12f', rec, 24, *matrix)
        out += rec
    out += tail
    return bytes(out)


# --- placement ---------------------------------------------------------------

class Ground:
    def __init__(self, surface):
        self.s = surface

    def __call__(self, x, z):
        """Bilinear surface height at world (x, z); the surface is tile-resolution, rows south-up."""
        h, w = self.s.shape
        tx = min(max(x / cw.DX * 2, 0), w - 1)
        ty = min(max(z / cw.DZ * 2, 0), h - 1)
        x0, y0 = int(tx), int(ty)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        fx, fy = tx - x0, ty - y0
        s = self.s
        return float((s[y0, x0] * (1 - fx) + s[y0, x1] * fx) * (1 - fy) + (s[y1, x0] * (1 - fx) + s[y1, x1] * fx) * fy)


def yaw_matrix(dx, dz, sx, sy, sz, pos):
    """Rows: local x -> (dx, 0, dz), local y -> up, local z -> (-dz, 0, dx); scaled; then position."""
    return [sx * dx, 0., sx * dz, 0., sy, 0., -sz * dz, 0., sz * dx, *pos]


def piece_at(ground, a, b):
    """great_wall_straight whose centre line runs from world point a to b on the ground:
    stretched along its x axis to |ab| (pieces are 4 model units at WALL_SCALE), pitched
    to the slope between the ends (vanilla tilts straight pieces the same way), origin at
    a moved half the wall thickness across (local z -1 is the centre line)."""
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    dx, dz = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    h0, h1 = ground(*a), ground(*b)
    hm = ground((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    y0 = min(h0, hm - (h1 - h0) / 2) - WALL_SINK          # the line a->b never above the ground's midpoint
    ex = np.array([dx * L, h1 - h0, dz * L]); ex /= np.linalg.norm(ex)
    ez = np.array([-dz, 0., dx])
    ey = np.cross(ez, ex)
    ox, oz = a[0] + WALL_SCALE * (-dz), a[1] + WALL_SCALE * dx
    row_x = ex * math.hypot(L, h1 - h0) / PIECE_UNITS          # 4 model units span the run
    return STRAIGHT, [float(v) for v in (*row_x, *(ey * WALL_SCALE), *(ez * WALL_SCALE), ox, y0, oz)]


def tower_at(ground, p, dx=1., dz=0.):
    L = math.hypot(dx, dz) or 1
    return TOWER, yaw_matrix(dx / L, dz / L, WALL_SCALE, WALL_SCALE, WALL_SCALE, (p[0], ground(*p) - TOWER_SINK, p[1]))


def causeway_cells(c, report_path):
    """--- WALL LINE --- Western Regions road crossings that apply_rivers leaves land (causeways: the road
    crosses on land, polish's gap link draws the channel across). apply_rivers lists each one by its first
    hex (rivers-apply-report.json 'wr_causeways'); the rest of the group is the road hexes reached from it
    that bridge the channel (water on two sides that do not touch each other: a hex in the river's line,
    not a road along the bank), at most WR_FORD_HEXES (4) per crossing. Empty without the report."""
    H, W = c.shape[:2]
    out = np.zeros((H, W), bool)
    try:
        seeds = json.loads(Path(report_path).read_text(encoding='utf-8')).get('wr_causeways') or []
    except (OSError, ValueError):
        return out
    water = (c[..., 0] & 3) == 1
    road = ((c[..., 3].astype(np.int32) >> 1) & 63) > 0

    def nbrs(q, r):
        s = 1 if q % 2 else -1          # odd-q: odd columns sit half a hex north
        for dq, dr in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, s), (-1, s)):
            if 0 <= q + dq < W and 0 <= r + dr < H:
                yield q + dq, r + dr
    for q, r in seeds:
        group, todo = [], [(q, r)]
        while todo and len(group) < 4:
            a, b = todo.pop(0)
            wn = [(x, y) for x, y in nbrs(a, b) if water[y, x]]
            bridges = any((x2, y2) not in set(nbrs(x1, y1)) for i, (x1, y1) in enumerate(wn) for x2, y2 in wn[i + 1:])
            if out[b, a] or water[b, a] or not road[b, a] or not bridges:
                continue
            out[b, a] = True
            group.append((a, b))
            todo += list(nbrs(a, b))
    return out


class Blockers:
    """Settlement slots, water (river causeways included) and pass sites, by hex distance from world points."""
    def __init__(self, m, rivers_report=None):
        c = m['cells']
        self.slot = (c[..., 2] >> 4) > 0
        self.water = (c[..., 0] & 3) == 1
        if rivers_report is not None:   # --- WALL LINE --- the river runs over its causeways too
            self.water = self.water | causeway_cells(c, rivers_report)
        sr, sq = np.nonzero(self.slot)
        self.slots = np.column_stack([sq + .5, (sr + (sq % 2) * .5 + .5) * cw.ZS])
        self.passes = [pass_site(lm)[0] for lm in spec.LANDMARKS if lm['kind'] == 'pass']   # world

    def near_slot(self, xz, d):
        x, z = xz
        near = self.slots[(np.abs(self.slots[:, 0] - x) < d * 2 + 2) & (np.abs(self.slots[:, 1] - z) < d * 2 + 2)]
        return len(near) and np.min(np.hypot(near[:, 0] - x, near[:, 1] - z)) / cw.ZS < d

    def in_pass(self, *world_points):
        """Any of the world points within a pass's own tower and pieces (PASS_EXTENT)."""
        return any(math.hypot(w[0] - p[0], w[1] - p[1]) < PASS_EXTENT for w in world_points for p in self.passes)

    def on_water(self, xz):
        q, r = cw.world_hex(*spec.to_world(xz))
        return bool(self.water[int(r), int(q)])


def wall_instances(lm, ground, block):
    """[(model, matrix)] for a wall landmark plus a summary. Along each run of the line the
    free stretches (not within SETTLEMENT_CLEAR hexes of a settlement slot, not inside a
    pass, not on water; sampled every SAMPLE world units) are filled with whole pieces,
    each stretched or squeezed to stretch length / n (n = round(length / piece)), so pieces
    meet end to end and stop flush at a pass or a gap."""
    pts = [spec.to_world(p) for p in spec.props_xz(lm)]   # --- WALL LINE --- props line (landmarks_spec)
    piece = PIECE_UNITS * WALL_SCALE
    out, gaps, since_tower = [], 0, 0
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        dx, dz = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        at = lambda s: (a[0] + dx * s, a[1] + dz * s)
        n_s = max(2, int(math.ceil(L / SAMPLE)) + 1)
        ss = np.linspace(0, L, n_s)
        wet = [block.on_water(spec.to_dx(*at(t))) for t in ss]
        free = [not (w or block.near_slot(spec.to_dx(*at(t)), SETTLEMENT_CLEAR) or block.in_pass(at(t))) for t, w in zip(ss, wet)]
        runs, start = [], None
        for j, (t, f) in enumerate(zip(ss, free)):
            if f and start is None:
                start, start_j = t, j
            if not f and start is not None:
                runs.append((start, prev, start_j > 0 and wet[start_j - 1], wet[j]))
                start = None
            prev = t
        if start is not None:
            runs.append((start, L, start_j > 0 and wet[start_j - 1], False))
        gaps += max(0, len(runs) - 1) + (0 if free[0] else 1) * (i == 0)
        for s0, s1, wet0, wet1 in runs:
            length = s1 - s0
            if length < .5 * piece:
                continue
            n = max(1, round(length / piece))
            step = length / n
            if wet0:                           # --- WALL LINE --- beacon tower on the bank where the wall leaves the water
                out.append(tower_at(ground, at(s0), dx, dz))
                since_tower = 0
            for k in range(n):
                p0, p1 = at(s0 + step * k), at(s0 + step * (k + 1))
                out.append(piece_at(ground, p0, p1))
                since_tower += 1
                if since_tower >= TOWER_EVERY and k < n - 1:
                    out.append(tower_at(ground, p1, dx, dz))
                    since_tower = 0
            if wet1:                           # --- WALL LINE --- and where it meets the water
                out.append(tower_at(ground, at(s1), dx, dz))
                since_tower = 0
        if i + 1 < len(pts) - 1 and free[-1]:      # beacon tower on the bend hides the joint
            out.append(tower_at(ground, b, dx, dz))
            since_tower = 0
    return out, dict(pieces=sum(m == STRAIGHT for m, _ in out), towers=sum(m == TOWER for m, _ in out), gaps=gaps)


def pass_site(lm):
    """World position and unit direction of a pass: on its wall's line ('along'), facing along
    it, or at its own point facing `heading` (radians from world +x)."""
    p = spec.to_world(spec.float_hex_xz(*lm['hexes'][0]))
    if 'along' in lm:
        line = [spec.to_world(q) for q in spec.props_xz(spec.BY_KEY[lm['along']])]   # --- WALL LINE ---
        a, b = min(zip(line, line[1:]), key=lambda s: _dist_seg(p, *s))
        dx, dz = b[0] - a[0], b[1] - a[1]
        t = min(max(((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / (dx * dx + dz * dz), 0), 1)
        p = (a[0] + t * dx, a[1] + t * dz)
    else:
        dx, dz = math.cos(lm['heading']), math.sin(lm['heading'])
    L = math.hypot(dx, dz)
    return p, dx / L, dz / L


def pass_instances(lm, ground):
    p, dx, dz = pass_site(lm)
    piece = PIECE_UNITS * WALL_SCALE
    half = 1.32 * WALL_SCALE                # tower half width (model bbox +-1.32)
    out = [tower_at(ground, p, dx, dz)]
    for side in (1, -1):
        for k in range(PASS_PIECES):
            s0 = half + k * piece
            a = (p[0] + side * dx * s0, p[1] + side * dz * s0)
            b = (p[0] + side * dx * (s0 + piece), p[1] + side * dz * (s0 + piece))
            out.append(piece_at(ground, a, b) if side == 1 else piece_at(ground, b, a))
    return out, dict(pieces=2 * PASS_PIECES, towers=1)


def _dist_seg(p, a, b):
    ax, az, bx, bz = a[0], a[1], b[0], b[1]
    t = ((p[0] - ax) * (bx - ax) + (p[1] - az) * (bz - az)) / max((bx - ax) ** 2 + (bz - az) ** 2, 1e-9)
    t = min(max(t, 0), 1)
    return math.hypot(p[0] - ax - t * (bx - ax), p[1] - az - t * (bz - az))


def yardang_instances(lm, ground, block):
    rng = random.Random('tw3k-landmarks-' + lm['key'])
    cx, cz = spec.to_world(spec.float_hex_xz(*lm['hexes'][0]))
    radius = lm['radius'] * cw.DZ * .9
    # --- DUNHUANG YARDANG --- per-landmark field length (n x radius along the heading), count and heading;
    # 白龙堆 keeps the defaults (a disc, YARDANG_COUNT, YARDANG_HEADING): its draws are unchanged.
    stretch = lm.get('stretch', 1.)
    heading = math.radians(lm['heading_deg']) if 'heading_deg' in lm else YARDANG_HEADING
    ux, uz = math.cos(heading), math.sin(heading)
    walls = [[spec.to_world(p) for p in spec.props_xz(w)] for w in spec.LANDMARKS if w['kind'] == 'wall']
    out = []
    tries = 0
    while len(out) < lm.get('count', YARDANG_COUNT) and tries < 2000:
        tries += 1
        a, d = rng.uniform(0, 2 * math.pi), radius * math.sqrt(rng.random())
        if 'stretch' in lm:                    # field frame: u along the heading
            u, v = d * math.cos(a) * stretch, d * math.sin(a)
            x, z = cx + u * ux - v * uz, cz + u * uz + v * ux
        else:
            x, z = cx + d * math.cos(a), cz + d * math.sin(a)
        if block.on_water(spec.to_dx(x, z)) or block.near_slot(spec.to_dx(x, z), SETTLEMENT_CLEAR)                 or any(math.hypot(x - m[9], z - m[11]) < YARDANG_SPACING for _, m in out) \
                or any(_dist_seg((x, z), p, q) < YARDANG_WALL_CLEAR * cw.HEX_DX for w in walls for p, q in zip(w, w[1:])):
            continue
        h = heading + rng.uniform(-.25, .25) + (math.pi if rng.random() < .5 else 0)
        s = rng.uniform(.8, 1.15)
        sx, sy, sz = (v * s for v in YARDANG_SCALE)
        y = min(ground(x + dx, z + dz) for dx, dz in ((0, 0), (.3, 0), (-.3, 0), (0, .3), (0, -.3))) - YARDANG_SINK
        out.append((rng.choice(PILLARS), yaw_matrix(math.cos(h), math.sin(h), sx, sy, sz, (x, y, z))))
    return out, dict(pillars=len(out))


def plan(m, ground, rivers_report=None):
    """{landmark key: [(model, matrix)]} and summaries."""
    block = Blockers(m, rivers_report)
    inst, summary = {}, {}
    for lm in spec.LANDMARKS + spec.PROP_SCENES:   # PROP_SCENES: props only, no AOI (DUNHUANG YARDANG)
        if lm['kind'] == 'wall':
            inst[lm['key']], summary[lm['key']] = wall_instances(lm, ground, block)
        elif lm['kind'] == 'pass':
            inst[lm['key']], summary[lm['key']] = pass_instances(lm, ground)
        elif lm['kind'] == 'yardang':
            inst[lm['key']], summary[lm['key']] = yardang_instances(lm, ground, block)
    return inst, summary


# --- bundle -------------------------------------------------------------------

def templates_from(assets, pre):
    """Template leaf head/tail and one vanilla record per model used, from simple
    (template-header) world-space AOI leaves."""
    t_leaf = assets[f'{pre}bmd_objects.{TEMPLATE_REGION}.{TEMPLATE}.31.bin']
    head, models, recs, tail = split_leaf(t_leaf)
    assert models == [TOWER] and len(recs) == 1
    want = {STRAIGHT, TOWER, *PILLARS}
    found = {TOWER: recs[0]}
    for k, v in assets.items():
        if not want - set(found):
            break
        if not k.endswith('.31.bin') or b'area_of_interest' not in v:
            continue
        h, ms, rs, t = split_leaf(v)
        if h != head or t != tail:
            continue
        for r in rs:
            mdl = ms[struct.unpack_from('<I', r, 2)[0]].lower()
            if mdl in want and mdl not in found and r[6:24] == recs[0][6:24]:
                found[mdl] = r
    missing = want - set(found)
    assert not missing, ('no vanilla template record for', missing)
    # every template record is world-space flagged and BHM_IGNORE, like the tower's
    for mdl, r in found.items():
        probe = build_leaf(head, tail, {mdl: r}, [(mdl, [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])])
        rr = leaf_records(probe)
        assert len(rr) == 1 and is_world_space(probe, rr[0]) and b'BHM_IGNORE' in r, mdl
    return head, tail, found


def run(stage, mapfile, surface_path, apply, reportpath):
    m = map_hex.read(mapfile)
    names = m['land_regions'] + m['sea_regions']
    ids = map_hex.region_ids(m['cells'])
    ground = Ground(np.load(surface_path, mmap_mode='r'))
    inst, summary = plan(m, ground, A / 'analysis/rivers-apply-report.json')
    report = {'template': f'{TEMPLATE_REGION}.{TEMPLATE}', 'prefab_range': [PREFAB_BASE, PREFAB_MAX], 'landmarks': summary, 'variants': []}
    for variant in cw.TERRAIN_VARIANTS:
        pre = f'terrain/campaigns/{variant}/'
        rootkey = pre + 'bmd_objects.bin'
        es, assets = pb.unbundle((stage / (pre + 'global_props.bin')).read_bytes(), rootkey)
        ours = re.compile(re.escape(pre) + r'bmd_objects\.[A-Za-z0-9_]+\.(\d+)(?:\.31)?\.bin$')

        def is_ours(k):
            mm = ours.match(k)
            return bool(mm) and PREFAB_BASE <= int(mm.group(1)) <= PREFAB_MAX
        root = assets[rootkey]
        cp, rs = decode(root)
        head, tail, tmpl = templates_from(assets, pre)
        t_parent = f'{pre}bmd_objects.{TEMPLATE_REGION}.{TEMPLATE}.bin'
        t_root = [r for r in rs if r['path'] == t_parent]
        assert len(t_root) == 1 and t_root[0]['region'] == TEMPLATE_REGION and t_root[0]['height_mode'] == 'BHM_IGNORE'
        t_root = t_root[0]
        pbody = assets[t_parent]
        _, children = decode(pbody)
        assert len(children) == 1 and children[0]['path'] == t_parent[:-4] + '.31.bin'
        child = children[0]
        # scenes: one per (landmark, owning region of each instance)
        groups = {}
        for key in sorted(inst):
            for model, mtx in inst[key]:
                q, r = cw.world_hex(mtx[9], mtx[11])
                region = names[ids[int(r), int(q)]]
                groups.setdefault((key, region), []).append((model, mtx))
        new, recs, scenes = {}, [], []
        for n, ((key, region), items) in enumerate(sorted(groups.items())):
            number = PREFAB_BASE + n
            assert number <= PREFAB_MAX
            parent_path = f'{pre}bmd_objects.{region}.{number}.bin'
            leaf_path = f'{pre}bmd_objects.{region}.{number}.31.bin'
            assert parent_path not in assets or is_ours(parent_path)
            leaf = build_leaf(head, tail, tmpl, items)
            check = leaf_records(leaf)
            assert len(check) == len(items) and all(is_world_space(leaf, c) for c in check)
            assert all(c['model'] == mdl and max(abs(a - b) for a, b in zip(c['matrix'], mtx)) < 1e-3 for c, (mdl, mtx) in zip(check, items))
            parent = pbody[:child['start']] + pb.rebind(pbody[child['start']:child['end']], child, region, leaf_path) + pbody[child['end']:]
            _, pc = decode(parent)
            assert [(x['path'], x['region']) for x in pc] == [(leaf_path, region)]
            new[parent_path] = parent
            new[leaf_path] = leaf
            recs.append(pb.rebind(root[t_root['start']:t_root['end']], t_root, region, parent_path))
            scenes.append(dict(landmark=key, region=region, parent=parent_path, leaf=leaf_path, instances=len(items),
                               models={mm: sum(1 for x, _ in items if x == mm) for mm in sorted({x for x, _ in items})}))
        kept = [r for r in rs if not is_ours(r['path'])]
        removed_roots = len(rs) - len(kept)
        updated, chk = pb.prefab_list(root, cp, rs, kept, recs)
        assert [c['path'] for c in chk[len(kept):]] == [s['parent'] for s in scenes]
        assert all(c['region'] == s['region'] for c, s in zip(chk[len(kept):], scenes))
        bundled = pb.rebundle(es, {k: v for k, v in assets.items() if not is_ours(k)}, rootkey, updated, new)
        report['variants'].append(dict(variant=variant, scenes=scenes, roots_added=len(recs), rerun_removed_roots=removed_roots,
                                       bundle_assets=struct.unpack_from('<I', bundled)[0], bundle_bytes=len(bundled)))
        if apply:
            pb.write_stage(stage, pre, bundled, updated, new)
        print(variant, len(scenes), 'landmark scenes,', sum(s['instances'] for s in scenes), 'instances',
              'applied' if apply else 'dry run', flush=True)
    for k, s in summary.items():
        print(' ', k, s, flush=True)
    report['applied'] = apply
    report['live_verified'] = False
    if apply:
        reportpath.parent.mkdir(parents=True, exist_ok=True)
        reportpath.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    return report, inst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--stage', type=Path, default=STAGE)
    ap.add_argument('--map', type=Path, default=A / 'work/final-map/map.hex')
    ap.add_argument('--surface', type=Path, default=None)
    args = ap.parse_args()
    surface = args.surface or A / f'analysis/polished-surface-{cw.TERRAIN_VARIANTS[0]}.npy'
    reportpath = A / 'analysis/landmark-props.json' if args.stage == STAGE else args.stage / 'landmark-props.json'
    run(args.stage, args.map, surface, args.apply, reportpath)


if __name__ == '__main__':
    main()

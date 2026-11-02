"""Trees and large props for the live-look offline view (render_live_view.py; renderer only).

Every object becomes a height-field impostor built from its model's LOD0 bounding box, rasterised per
render on a grid over the visible ground and ray-marched together with the terrain, so objects have
volume, occlude, cast sun shadows and show their real silhouette height:

  trees   campaign_maps/<map>/display/trees/trees.campaign_tree_list: every entry is one clump model
          (rigidmodels/campaign/vegetation/<family>/<name>, the summer wsmodel where only seasonal ones
          ship; rocks under rigidmodels/campaign/rocks/) at its listed position. The file has no
          per-entry scale; the engine draws them at about 0.3 of the model box (Dunhuang rock spots,
          and the 1/24-cell tile bmd frame below puts the same clumps at ~0.4), tree_scale. Canopies are
          flat-topped mats broken into crown_size crowns with gaps (a clump model holds several trees).
  props   terrain/campaigns/<map>/global_props.bin leaf records (model + 3x4 world matrix, row vectors:
          world = local @ M[:9] + M[9:]): mountains (peaks), rocks (domes), vegetation (canopy),
          settlement pieces and the Great Wall (oriented boxes). River/lake models are water
          (campaign_water_meshes), ground decals are flat, and the autumn/winter/spring copies that
          sit on every seasonal summer model are skipped (the 190 start looks like summer).
  tiles   the models of the fine tiles' bmd_data.bin (mountain tiles: rock pillars, their trees) at
          every global_map/tile_list.bin placement.
  towns   settlements are engine-placed (not in any of the above). P['towns'] = 'boxes': a house-box cluster
          at each region town of the region design (capital: larger, with a wall ring). 'layout': the vanilla
          prefabs/campaign/settlement_layouts/layouts_land/land_layout_*.bmd pieces (buildings, roofs, walls,
          platforms, trees; read with their LOD0 boxes and texture colours) at each town, one layout per town
          picked by name, the pieces of one building level (town_level: capitals, others; *_destr / *_damaged
          skipped), town_scale world units per layout unit; built per view for the towns in it, and their
          ground decals tint the ground under the town (decal_tint).

Colours are the mean of each model's base-colour texture (smallest mip), else a per-family guess.
Model bounds: RMV2 LOD0 group boxes (as reconcile_terrain_prefabs.model_radius reads them).
"""
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

import render_campaign_view as rv

FAMILY_COLOUR = {   # fallback base colours (sRGB 0..1)
    'cold': (.20, .27, .18), 'temperate': (.27, .36, .16), 'subtropical': (.24, .34, .14), 'arid': (.52, .42, .22),
    'bamboo': (.36, .46, .18), 'rock': (.52, .48, .42), 'mountain': (.55, .45, .33), 'settlement': (.55, .44, .34),
    'wall': (.62, .56, .46), 'other': (.40, .40, .32)}
DOME, PEAK, BOX = 0, 1, 2
VEGETATION = ('cold', 'temperate', 'subtropical', 'arid', 'bamboo')
TOWN_ROOF, TOWN_WALL = (.38, .30, .25), (.62, .56, .48)
EXTRA_PACKS = ('models.pack', 'models2.pack')


def add_model_packs(packs):
    """Index the vanilla model packs too (mountain/rock/settlement geometry lives there); never overrides."""
    if getattr(packs, '_models_added', False):
        return
    import pickle
    from campaign_ground_textures import GAME
    from render_live_view import CACHE
    cache = CACHE / 'pack-index-models.pkl'
    if cache.is_file():
        extra = pickle.loads(cache.read_bytes())
    else:
        from analyze_reference import inventory
        extra = {}
        for n in EXTRA_PACKS:
            for e in inventory(GAME / n, False)['entries']:
                extra.setdefault(e['path'], (str(GAME / n), e['offset'], e['size'], e['compressed']))
        cache.write_bytes(pickle.dumps(extra))
    for k, v in extra.items():
        packs.index.setdefault(k, v)
    packs._models_added = True


class Models:
    """LOD0 bounding boxes and mean base colours of rigid models / wsmodels from a pack source."""
    def __init__(self, packs):
        self.packs = packs
        self.bounds = {}
        self.colours = {}
        self.lower = {k.lower(): k for k in packs.index if k.startswith(('rigidmodels/', 'terrain/'))}

    def _key(self, key):
        return self.lower.get(key.lower())

    def geometry(self, key):
        k = self._key(key)
        if k is None:
            return None
        if k.endswith('.wsmodel'):
            try:
                geo = ET.fromstring(self.packs.read(k).decode('utf-8', 'replace')).findtext('geometry')
            except ET.ParseError:
                return None
            return self._key(geo) if geo else None
        return k

    def bbox(self, key):
        """((lo xyz), (hi xyz)) of LOD0, or None."""
        if key not in self.bounds:
            out = None
            geo = self.geometry(key)
            if geo:
                try:
                    g = self.packs.read(geo)
                    if g[:4] == b'RMV2':
                        lods = struct.unpack_from('<I', g, 8)[0]
                        groups = struct.unpack_from('<I', g, 140)[0]
                        p = 140 + lods * 28
                        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
                        for _ in range(groups):
                            lo = np.minimum(lo, struct.unpack_from('<3f', g, p + 24))
                            hi = np.maximum(hi, struct.unpack_from('<3f', g, p + 36))
                            p += struct.unpack_from('<I', g, p + 4)[0]
                        if np.isfinite(lo).all() and np.isfinite(hi).all():
                            out = (lo, hi)
                except (KeyError, struct.error, AssertionError):
                    out = None
            self.bounds[key] = out
        return self.bounds[key]

    def colour(self, key, fallback):
        """Mean sRGB colour of the model's (summer) base-colour texture, else fallback."""
        if key not in self.colours:
            col = None
            geo = self.geometry(key)
            if geo:
                try:
                    g = self.packs.read(geo)
                    tex = [t.decode('ascii', 'replace').replace('\\', '/').lower() for t in re.findall(rb'[\w/\\.-]+\.dds', g)]
                    tex = [t for t in tex if 'base_colour' in t or t.endswith('_d.dds')]
                    tex.sort(key=lambda t: ('winter' in t, 'autumn' in t, 'spring' in t))
                    for t in tex:
                        if t in self.packs.index:
                            col = mean_colour(self.packs.read(t))
                            break
                except (KeyError, struct.error, ValueError, AssertionError):
                    col = None
            self.colours[key] = np.array(col if col is not None else fallback, np.float64)
        return self.colours[key]


def leaf_colour(models, key, season='harvest'):
    """Mean sRGB of a vegetation model's base-colour atlas in the season the campaign starts in (the model
    references one season's file, e.g. cold_trees_winter_base_colour; the engine swaps the season token), alpha-
    weighted (BC3 cutout: leaves and needles, not the transparent gaps), or None."""
    geo = models.geometry(key) if key else None
    if not geo:
        return None
    tex = [t.decode('ascii', 'replace').replace('\\', '/').lower() for t in re.findall(rb'[\w/\\.-]+\.dds', models.packs.read(geo))]
    tex = [t for t in tex if 'base_colour' in t]
    for t in tex:
        cands = [re.sub(r'_(spring|summer|harvest|autumn|winter)_', f'_{s_}_', t) for s_ in (season, 'summer', 'autumn')] + [t]
        for c in cands:
            if c in models.packs.index:
                try:
                    return np.array(mean_colour(models.packs.read(c)), np.float64)
                except (ValueError, struct.error, AssertionError):
                    break
    return None


def mean_colour(dds):
    """Mean sRGB (0..1) of a BC1/BC3 texture's 4x4-or-larger mip, alpha-weighted where BC3 alpha exists."""
    from render_live_view import decode_bc3_rgba
    from campaign_ground_textures import decode_dds
    h, w = struct.unpack_from('<2I', dds, 12)
    mip = max(0, int(np.log2(max(1, min(h, w) // 16))))
    fourcc = dds[84:88]
    fmt = struct.unpack_from('<I', dds, 128)[0] if fourcc == b'DX10' else None
    if fourcc == b'DXT5' or fmt in (77, 78):
        a = decode_bc3_rgba(dds, mip).astype(np.float64)
        wgt = a[..., 3:4] / 255
        if wgt.sum() < 1:
            wgt = np.ones_like(wgt)
        return ((a[..., :3] * wgt).sum((0, 1)) / wgt.sum() / 255).tolist()
    if fourcc == b'DXT1' or fmt in (70, 71, 72):
        return (decode_dds(dds, mip).reshape(-1, 3).mean(0) / 255).tolist()
    raise ValueError('unsupported texture format')


SEASON_VARIANT = re.compile(r'_(autumn|winter|spring)(_|\.)')


def family(model):
    m = model.lower()
    if '/mountains/' in m or re.search(r'/area_of_interest/mount_', m):
        return 'mountain', PEAK
    if '/rocks/' in m or 'rock' in m.rsplit('/', 1)[-1]:
        return 'rock', DOME
    if 'great_wall' in m or '/wall' in m.rsplit('/', 1)[-1][:5]:
        return 'wall', BOX
    if '/settlements/' in m or '/area_of_interest/' in m or '/landmarks/' in m:
        return 'settlement', BOX
    if '/vegetation/' in m:
        for f in ('cold', 'temperate', 'subtropical', 'arid', 'bamboo'):
            if f in m:
                return f, DOME
        return 'temperate', DOME
    return 'other', BOX


def arrays(rows):
    """Flat arrays of object rows (see Objects): position, local xz -> world xz map and its inverse, box, y scale,
    shape, colour, flags, xz radius, top and a per-object seed."""
    import types
    o = types.SimpleNamespace()
    n = len(rows)
    o.x = np.array([r[0] for r in rows]); o.y = np.array([r[1] for r in rows]); o.z = np.array([r[2] for r in rows])
    o.A = np.array([r[3] for r in rows]).reshape(n, 2, 2)
    o.Ainv = np.linalg.inv(o.A)
    o.lo = np.array([r[4] for r in rows]).reshape(n, 3); o.hi = np.array([r[5] for r in rows]).reshape(n, 3)
    o.ys = np.array([r[6] for r in rows]); o.shape = np.array([r[7] for r in rows], np.int8)
    o.col = np.array([r[8] for r in rows]).reshape(n, 3)
    o.listed = np.array([r[9] for r in rows], bool)    # tree list entries (trees toggle, tree_scale)
    o.model = np.array([r[10] for r in rows], object)
    o.canopy = np.array([r[11] for r in rows], bool)   # crown-broken canopy shape and colour jitter
    # world AABB radius in xz (corners of the local box through A)
    corners = np.stack([np.stack([o.lo[:, 0], o.lo[:, 2]], -1), np.stack([o.hi[:, 0], o.lo[:, 2]], -1),
                        np.stack([o.lo[:, 0], o.hi[:, 2]], -1), np.stack([o.hi[:, 0], o.hi[:, 2]], -1)], 1)
    w = np.einsum('nkc,ncd->nkd', corners, o.A)
    o.r = np.abs(w).max((1, 2))
    o.top = o.y + o.hi[:, 1] * o.ys
    # keyed by position, not row index: adding or removing one object must not re-jitter every later one
    # (art_review reuses base renders of views no change reaches)
    q = np.round(np.c_[o.x, o.z] * 1000).astype(np.int64) if n else np.zeros((0, 2), np.int64)
    o.seed = ((q[:, 0] * 73856093 ^ q[:, 1] * 19349663) % 1000) / 1000.
    return o


class Objects:
    """All tree clumps and props of a pack ground as flat arrays (built once per process)."""
    def __init__(self, ground, props=True, trees=True, towns='boxes'):
        self.g = ground
        self.town_mode = towns
        self.layouts, self.town_sets, self.town_key = {}, {}, None
        self.tree_keys = {}     # tree-list group -> (model key, species colour); leaf atlases read lazily
        add_model_packs(ground.packs)
        self.models = Models(ground.packs)
        # (x, y, z, A 2x2 local xz -> world xz, local lo, hi, y scale, shape, colour, from the tree list,
        #  model name, canopy shape)
        rows = []
        if trees:
            self._trees(rows)
        if props:
            self._props(rows)
            self._tile_props(rows)
            if towns == 'boxes':
                self._towns(rows)
        self.__dict__.update(vars(arrays(rows)))
        print(f'objects: {self.listed.sum()} tree-list entries, {(~self.listed).sum()} props and town pieces', flush=True)

    def texture_colours(self):
        """Row colours with tree-list rows in their model's own leaf-atlas colour (P['tree_colour'] == 'texture').
        Built on first use: decoding every tree model's atlas is init cost the default 'species' never needs."""
        if getattr(self, '_col_tex', None) is None:
            self._col_tex = self.col.copy()
            for name, (key, col) in self.tree_keys.items():
                tex = leaf_colour(self.models, key)
                self._col_tex[self.listed & (self.model == name)] = col if tex is None else tex
        return self._col_tex

    def tree_model(self, name):
        """Model key of a tree-list group: the plain model, else its summer/harvest wsmodel (subtropical
        and bamboo groups ship only seasonal ones), else any model whose file name starts with it."""
        ends = [f'/{name}{s}' for s in ('.rigid_model_v2', '.wsmodel', '_summer.wsmodel', '_harvest.wsmodel')]
        keys = [k for k in self.models.lower if k.rsplit('/', 1)[-1].startswith(name)]
        for e in ends:
            for k in keys:
                if k.endswith(e):
                    return k
        return next(iter(sorted(keys)), None)

    def _trees(self, rows):
        for name, xyz, colour, _ in rv.load_trees(self.g):
            key = self.tree_model(name)
            bb = self.models.bbox(key) if key else None
            if bb is None:
                lo, hi = np.array([-.4, 0, -.4]), np.array([.4, .8, .4])
            else:
                lo, hi = bb
            col = np.array(colour, np.float64) / 255   # species colour (render_campaign_view.TREE_COLOURS guess)
            self.tree_keys[name] = (key, col)
            rock = name.startswith(('general_rock', 'rock_shrub'))
            for x, y, z in xyz:
                rows.append((x, y, z, (1., 0., 0., 1.), lo, hi, 1., DOME, col, True, name, not rock))

    def _props(self, rows):
        """Leaf model records of the bundle. Leaves are world-space (translate_world_space_leaves): the
        root/parent prefab matrices that shift native content are not applied again (composed, the
        native leaves would float ~4 units off the ground; raw they sit on it)."""
        from props_bundle import embedded
        from translate_campaign_lake_instances import records
        pre = f'terrain/campaigns/{rv.VARIANT}/'
        assets = embedded(self.g.read(pre + 'global_props.bin'))[0]
        for k, b in assets.items():
            if not k.endswith('.bin') or k.endswith('/bmd_objects.bin') or b'rigidmodels/' not in b:
                continue
            try:
                recs = records(b)
            except (AssertionError, struct.error, UnicodeDecodeError):
                continue      # parents (child prefab lists) and model-less variant switches
            for r in recs:
                m = r['model']
                ml = m.lower()
                if ml.startswith(pre + 'models/') or '/lakes/' in ml or 'water_lily' in ml or '/river' in ml:
                    continue      # water (campaign_water_meshes)
                if '/decals/' in ml or SEASON_VARIANT.search(ml.rsplit('/', 1)[-1]):
                    continue      # flat ground decals; other-season copies of a summer model at the same spot
                M = np.array(r['matrix'], np.float64)
                if M[10] < -100:
                    continue
                R = M[:9].reshape(3, 3)
                if abs(R[0, 0] * R[2, 2] - R[0, 2] * R[2, 0]) < 1e-8:   # no footprint in xz (degenerate)
                    continue
                bb = self.models.bbox(m)
                if bb is None:
                    continue
                fam, shape = family(m)
                rows.append((M[9], M[10], M[11], (R[0, 0], R[0, 2], R[2, 0], R[2, 2]), bb[0], bb[1], R[1, 1], shape,
                             self.models.colour(m, FAMILY_COLOUR[fam]), False, ml, fam in VEGETATION))

    def _tile_props(self, rows):
        """Models of the fine campaign tiles' bmd_data.bin (mountain tiles: the rock pillars and their trees)
        at every tile_list.bin placement. bmd units are 1/24 tile cell (a 22-cell piece spans 0..528,
        its mesh 0..2816 at 128 per cell); x runs east, z from the piece's north edge; the placement's
        one-hot rotation turns the piece as in campaign_fine_tiles.composite. Bases sit on the LF."""
        from campaign_fine_tiles import FineTiles, REC, ROT, CELL_X, CELL_Z, mesh_group
        from translate_campaign_lake_instances import records
        try:
            blob = self.g.read(f'terrain/campaigns/{rv.VARIANT}/global_map/tile_list.bin')
        except KeyError:
            return
        paths, start = FineTiles._paths(blob)
        paths = [q.replace('\\', '/').lower() for q in paths]
        recs = np.frombuffer(blob, REC, offset=start, count=(len(blob) - start) // REC.itemsize)
        pieces = {}
        for i in np.unique(recs['p']):
            q = paths[int(i)]
            if q + 'bmd_data.bin' not in self.g.packs.index:
                continue
            try:
                rs = records(self.g.packs.read(q + 'bmd_data.bin'))
                lo, hi, _ = mesh_group(self.g.packs.read(q + 'mesh.rigid_model_v2'))
            except (AssertionError, KeyError, struct.error):
                continue
            keep = []
            for r in rs:
                m = r['model'].lower()
                if '/decals/' in m or SEASON_VARIANT.search(m.rsplit('/', 1)[-1]) or 'water' in m:
                    continue
                bb = self.models.bbox(m)
                if bb is not None:
                    fam, shape = family(m)
                    keep.append((np.array(r['matrix'], np.float64), bb, shape, self.models.colour(m, FAMILY_COLOUR[fam]),
                                 m, fam in VEGETATION))
            if keep:
                pieces[int(i)] = ((round((hi[0] - lo[0]) / 128), round((hi[2] - lo[2]) / 128)), keep)
        sel = recs[np.isin(recs['p'], list(pieces))]
        k = 1 / 24
        turn = {0: ((1, 0), (0, 1)), 1: ((0, 1), (-1, 0)), 2: ((-1, 0), (0, -1)), 3: ((0, -1), (1, 0))}
        n0 = len(rows)
        pos = []
        for r in sel:
            (tw, th), keep = pieces[int(r['p'])]
            rt = ROT.get(int(r['f']) & 0xf0, 0)
            fw, fh = (tw, th) if rt % 2 == 0 else (th, tw)
            Mt = np.array(turn[rt], np.float64)       # rows: d(fx, fn) per du, dv
            for M, (lo, hi), shape, col, m, veg in keep:
                u, v = M[9] * k, M[11] * k
                fx, fn = ((u, v), (fw - v, u), (fw - u, fh - v), (v, fh - u))[rt]
                x, z = (r['x'] + fx) * CELL_X, (r['y'] + fh - fn) * CELL_Z
                R = M[:9].reshape(3, 3)
                A = np.array([[R[0, 0], R[0, 2]], [R[2, 0], R[2, 2]]]) * k @ Mt @ np.diag([CELL_X, -CELL_Z])
                if abs(np.linalg.det(A)) < 1e-12:
                    continue
                rows.append((x, 0., z, tuple(A.ravel()), lo, hi, R[1, 1] * k * CELL_X, shape, col, False, m, veg))
                pos.append((x, z))
        if pos:
            pos = np.array(pos)
            y = self.g.heights(pos[:, 0], pos[:, 1])
            for j, yy in enumerate(y):
                rows[n0 + j] = (rows[n0 + j][0], float(yy)) + rows[n0 + j][2:]
        print(f'tile props: {len(rows) - n0} models on {len(sel)} placements of {len(pieces)} pieces', flush=True)

    def _towns(self, rows):
        """Settlements are engine-placed models (not in global_props): a deterministic cluster of house boxes
        around each region town of the region design (capitals larger, with a wall ring)."""
        try:
            towns = rv.load_towns()
        except (OSError, KeyError, ValueError):
            return
        rng = np.random.default_rng(190)
        for name, x, z, cap in towns:
            n, rad = (34, 1.0) if cap else (16, .6)
            a = rng.uniform(0, 2 * np.pi, n)
            r = rad * np.sqrt(rng.uniform(0, 1, n))
            hx, hz = x + r * np.cos(a), z + r * np.sin(a)
            hy = self.g.heights(hx, hz)
            for i in range(n):
                c, s_ = np.cos(a[i]), np.sin(a[i])
                w, d, h = rng.uniform(.07, .14), rng.uniform(.05, .09), rng.uniform(.06, .12) * (1.4 if cap else 1)
                rows.append((hx[i], hy[i], hz[i], (c, s_, -s_, c), np.array([-w, 0, -d]), np.array([w, h, d]), 1., BOX,
                             np.array(TOWN_ROOF if i % 3 else TOWN_WALL), False, 'town:' + name, False))
            if cap:   # wall ring of short segments
                k = 24
                for i in range(k):
                    t = 2 * np.pi * i / k
                    wx, wz = x + rad * 1.15 * np.cos(t), z + rad * 1.15 * np.sin(t)
                    c, s_ = np.cos(t + np.pi / 2), np.sin(t + np.pi / 2)
                    L = np.pi * rad * 1.15 / k * 1.05
                    rows.append((wx, float(self.g.heights(np.array([wx]), np.array([wz]))[0]), wz, (c, s_, -s_, c),
                                 np.array([-L, 0, -.03]), np.array([L, .1, .03]), 1., BOX, np.array(TOWN_WALL), False,
                                 'town-wall:' + name, False))

    def layout(self, name):
        """Pieces of one settlement layout: [(model, 3x4 matrix, level or 0, is a ground decal)]."""
        if name not in self.layouts:
            from translate_campaign_lake_instances import records
            out = []
            for r in records(self.g.packs.read(f'prefabs/campaign/settlement_layouts/layouts_land/{name}.bmd')):
                m = r['model'].lower()
                base = m.rsplit('/', 1)[-1]
                if re.search(r'destr|damaged|burned|siege', base) or SEASON_VARIANT.search(base):
                    continue
                lv = re.search(r'_level(\d)', base)
                out.append((m, np.array(r['matrix'], np.float64), int(lv.group(1)) if lv else 0, '/decals/' in m))
            self.layouts[name] = out
        return self.layouts[name]

    def town_pieces(self, x0, z0, x1, z1, P):
        """Arrays of the layout pieces of the towns within the window (built per town, cached per town options),
        and the town decals [(x, z, A, lo, hi, colour)] there. Layout x east, z south (as the tile bmd frames),
        y up; a piece's base sits at the town ground + its layout height."""
        import zlib
        key = (P['town_scale'], tuple(P['town_level']), P['town_min'])
        if key != self.town_key:
            self.town_key, self.town_sets = key, {}
            self.towns = rv.load_towns()
            self.layout_names = sorted(k.rsplit('/', 1)[-1][:-4] for k in self.g.packs.index
                                       if k.startswith('prefabs/campaign/settlement_layouts/layouts_land/land_layout_'))
        k = P['town_scale']
        rows, decals = [], []
        reach = 12 * k
        for name, x, z, cap in self.towns:
            if x + reach < x0 or x - reach > x1 or z + reach < z0 or z - reach > z1:
                continue
            if name not in self.town_sets:
                lay = self.layout(self.layout_names[zlib.crc32(name.encode()) % len(self.layout_names)])
                level = P['town_level'][0 if cap else 1]
                y0 = float(self.g.heights(np.array([x]), np.array([z]))[0])
                tr, td = [], []
                for m, M, lv, decal in lay:
                    if lv and lv != level:
                        continue
                    bb = self.models.bbox(m)
                    if bb is None:
                        continue
                    R = M[:9].reshape(3, 3)
                    A = np.array([[R[0, 0], -R[0, 2]], [R[2, 0], -R[2, 2]]]) * k   # rows: d(world x, z) per local x, z
                    if abs(np.linalg.det(A)) < 1e-14:
                        continue
                    wx, wz = x + M[9] * k, z - M[11] * k
                    if decal:
                        td.append((wx, wz, A, bb[0], bb[1], self.models.colour(m, (.5, .45, .35))))
                        continue
                    size = max((bb[1][0] - bb[0][0]) * np.abs(A[0]).max(), (bb[1][2] - bb[0][2]) * np.abs(A[1]).max())
                    if size < P['town_min'] or bb[1][1] * R[1, 1] * k < P['town_min'] / 4:
                        continue       # sub-pixel pieces and flat markers
                    fam, shape = family(m)
                    tr.append((wx, y0 + M[10] * k, wz, tuple(A.ravel()), bb[0], bb[1], R[1, 1] * k, shape,
                               self.models.colour(m, FAMILY_COLOUR[fam]), False, m, fam in VEGETATION))
                self.town_sets[name] = (tr, td)
            rows += self.town_sets[name][0]
            decals += self.town_sets[name][1]
        return (arrays(rows) if rows else None), decals

    def decal_tint(self, decals, wx, wz, strength):
        """(weight 0..1, colour) of town ground decals at world points (box footprint, soft outer rim)."""
        w = np.zeros(np.shape(wx))
        col = np.zeros(np.shape(wx) + (3,))
        for dx, dz, A, lo, hi, c in decals:
            Ai = np.linalg.inv(A)
            ux = (wx - dx) * Ai[0, 0] + (wz - dz) * Ai[1, 0]
            uz = (wx - dx) * Ai[0, 1] + (wz - dz) * Ai[1, 1]
            inside = (ux >= lo[0]) & (ux <= hi[0]) & (uz >= lo[2]) & (uz <= hi[2])
            if inside.any():
                fx = np.minimum(ux - lo[0], hi[0] - ux) / max(hi[0] - lo[0], 1e-6)
                fz = np.minimum(uz - lo[2], hi[2] - uz) / max(hi[2] - lo[2], 1e-6)
                a = np.clip(np.minimum(fx, fz) / .15, 0, 1) * strength
                take = inside & (a >= w)
                w = np.where(take, a, w)
                col[take] = c
        return w, col

    def raster(self, x0, z0, cell, nx, nz, P):
        """(top height grid, colour grid, is-tree grid) of the objects over cells x0 + (i + .5) cell (rows
        south-up); -inf where none. With P['towns'] == 'layout' the window's town pieces are added."""
        top, col, tree = self._raster(self, x0, z0, cell, nx, nz, P)   # the arrays live on self
        self.decals = []
        if self.town_mode == 'layout' and P['props']:
            T, self.decals = self.town_pieces(x0, z0, x0 + nx * cell, z0 + nz * cell, P)
            if T is not None:
                t2, c2, r2 = self._raster(T, x0, z0, cell, nx, nz, P)
                up = t2 > top
                top = np.where(up, t2, top)
                col[up] = c2[up]
                tree[up] = r2[up]
        return top, col, tree

    def _raster(self, ob, x0, z0, cell, nx, nz, P):
        """(top height grid, colour grid, is-tree grid) of the objects over cells x0 + (i + .5) cell
        (rows south-up); -inf where none. Tree-list entries carry no scale of their own: they are drawn at
        P['tree_scale'] (the engine's, fit to the live shots) about their base, heights further times
        P['tree_height']."""
        top = np.full((nz, nx), -np.inf, np.float32)
        col = np.zeros((nz, nx, 3), np.float32)
        tree = np.zeros((nz, nx), bool)
        ts = P['tree_scale']
        rad = np.where(ob.listed, ob.r * ts, ob.r)
        sel = np.flatnonzero((ob.x + rad >= x0) & (ob.x - rad <= x0 + nx * cell) &
                             (ob.z + rad >= z0) & (ob.z - rad <= z0 + nz * cell))
        if not P['trees']:
            sel = sel[~ob.listed[sel]]
        if not P['props']:
            sel = sel[ob.listed[sel]]
        if not len(sel):
            return top, col, tree
        i0 = np.clip(np.floor((ob.x[sel] - rad[sel] - x0) / cell), 0, nx - 1).astype(np.int64)
        i1 = np.clip(np.ceil((ob.x[sel] + rad[sel] - x0) / cell), 0, nx - 1).astype(np.int64)
        j0 = np.clip(np.floor((ob.z[sel] - rad[sel] - z0) / cell), 0, nz - 1).astype(np.int64)
        j1 = np.clip(np.ceil((ob.z[sel] + rad[sel] - z0) / cell), 0, nz - 1).astype(np.int64)
        w, h = i1 - i0 + 1, j1 - j0 + 1
        cnt = w * h
        cs = np.cumsum(cnt)
        start = 0
        chunk = 6_000_000
        while start < len(sel):
            stop = max(int(np.searchsorted(cs, (cs[start - 1] if start else 0) + chunk, 'right')), start + 1)
            n = cnt[start:stop]
            loc = np.repeat(np.arange(start, stop), n)
            off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            ci = i0[loc] + off % w[loc]
            cj = j0[loc] + off // w[loc]
            o = sel[loc]
            wx = x0 + (ci + .5) * cell - ob.x[o]
            wz = z0 + (cj + .5) * cell - ob.z[o]
            s = np.where(ob.listed[o], ts, 1.)
            lx = (wx * ob.Ainv[o, 0, 0] + wz * ob.Ainv[o, 1, 0]) / s
            lz = (wx * ob.Ainv[o, 0, 1] + wz * ob.Ainv[o, 1, 1]) / s
            lo, hi = ob.lo[o], ob.hi[o]
            cx, cz = (lo[:, 0] + hi[:, 0]) / 2, (lo[:, 2] + hi[:, 2]) / 2
            hx, hz = np.maximum((hi[:, 0] - lo[:, 0]) / 2, 1e-3), np.maximum((hi[:, 2] - lo[:, 2]) / 2, 1e-3)
            u, v = (lx - cx) / hx, (lz - cz) / hz
            r2 = u * u + v * v
            shp = ob.shape[o]
            # only cells inside the footprint (box or ellipse) can be drawn: the profile is computed for those
            # alone, each shape with its own terms (elementwise, so the values are the full arrays' values)
            fp = np.flatnonzero(np.where(shp == BOX, (np.abs(u) <= 1) & (np.abs(v) <= 1), r2 <= 1))
            o, ci, cj, wx, wz, u, v, r2, shp = o[fp], ci[fp], cj[fp], wx[fp], wz[fp], u[fp], v[fp], r2[fp], shp[fp]
            hi = hi[fp]
            H = np.maximum(hi[:, 1], .05) * ob.ys[o] * np.where(ob.listed[o], ts * P['tree_height'], 1.)
            # lumps: a clump model holds several trees, so its dome breaks into crowns with gaps between
            # them (tree_freq crowns across, tree_gap of the dome height left open); mountains get crags
            ph = ob.seed[o] * 6.283
            istree = ob.canopy[o] & (shp == DOME)
            peaked, boxed = shp == PEAK, shp == BOX
            domed = ~(peaked | boxed | istree)
            prof = np.ones(len(o))
            sphere = np.zeros(len(o))
            sel_t, sel_d, sel_p = np.flatnonzero(istree), np.flatnonzero(domed), np.flatnonzero(peaked)
            sphere[sel_t] = np.sqrt(np.clip(1 - r2[sel_t], 0, 1))
            sphere[sel_d] = np.sqrt(np.clip(1 - r2[sel_d], 0, 1))

            def lump(q):
                uq, vq, pq = u[q], v[q], ph[q]
                return np.sin(uq * 5.1 + pq) * np.sin(vq * 4.7 - pq) + .5 * np.sin(uq * 9.3 - 2 * pq) * np.sin(vq * 8.9 + pq)
            k = 2 * np.pi / P['crown_size']   # crowns in world units, so large prop clumps break up as finely
            pt = ph[sel_t]
            crown = .5 + .5 * np.sin(wx[sel_t] * k + pt) * np.sin(wz[sel_t] * k * .93 - 1.3 * pt)
            # canopy: flat-topped (a clump or forest-patch model is a mat of crowns, not one dome)
            prof[sel_t] = np.minimum(sphere[sel_t] * P['canopy_flat'], 1) * (1 - P['tree_lump'] + P['tree_lump'] * crown)
            prof[sel_d] = sphere[sel_d] * (1 + .25 * lump(sel_d))
            prof[sel_p] = np.clip(1 - np.sqrt(r2[sel_p]), 0, 1) ** .85 * (1 + .18 * lump(sel_p))
            inside = prof > 0
            inside &= ~istree | (prof > P['tree_gap'] * sphere)
            ty = ob.y[o] + H * prof
            keep = np.flatnonzero(inside)
            cell_id = cj[keep] * nx + ci[keep]
            order = np.argsort(ty[keep], kind='stable')
            flat = top.reshape(-1)
            k2 = keep[order]
            cid = cell_id[order]
            higher = ty[k2] > flat[cid]
            cid, k2 = cid[higher], k2[higher]
            flat[cid] = ty[k2]          # ascending order: the highest write per cell lands last
            cano = ob.canopy[o[k2]]
            shade = np.where(cano, .82 + .36 * ob.seed[o[k2]], 1.)[:, None]
            cols = ob.texture_colours() if P.get('tree_colour') == 'texture' else ob.col
            col.reshape(-1, 3)[cid] = cols[o[k2]] * shade
            tree.reshape(-1)[cid] = ob.canopy[o[k2]]
            start = stop
        return top, col, tree


class Overlay:
    """Object height field over a window, sampled by nearest cell (the ray-march surface adds it)."""
    def __init__(self, objects, x0, z0, x1, z1, P, max_cells=4096):
        span = max(x1 - x0, z1 - z0)
        self.cell = max(span / max_cells, P['object_cell'])
        self.x0, self.z0 = x0, z0
        self.nx, self.nz = int(np.ceil((x1 - x0) / self.cell)), int(np.ceil((z1 - z0) / self.cell))
        self.top, self.col, self.tree = objects.raster(x0, z0, self.cell, self.nx, self.nz, P)
        self.hmax = float(self.top.max()) if np.isfinite(self.top).any() else -np.inf
        self.hrel = 0.       # tallest object above the ground (set by the caller, which knows the ground)

    def idx(self, wx, wz):
        i = ((np.asarray(wx) - self.x0) / self.cell).astype(np.intp)
        j = ((np.asarray(wz) - self.z0) / self.cell).astype(np.intp)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.nz)
        return np.clip(i, 0, self.nx - 1), np.clip(j, 0, self.nz - 1), ok

    def height(self, wx, wz):
        i, j, ok = self.idx(wx, wz)
        return np.where(ok, self.top[j, i], -np.inf)

    def normals(self):
        """Per-cell normals of the object tops (finite cells), y up."""
        t = np.where(np.isfinite(self.top), self.top, np.nan).astype(np.float64)
        gz, gx = np.gradient(t, self.cell)
        n = np.stack([-gx, np.ones_like(gx), -gz], -1)
        n = np.where(np.isfinite(n), n, np.array([0., 1., 0.]))
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


def march_objects(surface, eye, d, found, t, hrel, steps=96, refine=8):
    """Refine ground hits t for object tops: objects stand on the ground, so a ray meets one only in
    the last stretch before its ground hit where it is less than hrel above the ground; that stretch
    is stepped finely (uniform steps over the whole ray miss canopies thinner than a step)."""
    if getattr(surface, 'gpu', None) is not None:     # render_gpu.Surface over the overlay
        return surface.gpu.march_objects(eye, d, found, t, hrel, steps, refine)
    dd = d[found]
    tg = t[found]
    span = hrel / np.clip(-dd[:, 1], .05, 1) + .05
    ta = np.maximum(tg - span, 0)
    hit = tg.copy()
    lo = tg.copy()
    done = np.zeros(len(tg), bool)
    prev = ta.copy()
    for k in range(1, steps + 1):
        tk = ta + (tg - ta) * k / steps
        act = np.flatnonzero(~done)
        if not len(act):
            break
        x = eye[0] + dd[act, 0] * tk[act]
        z = eye[2] + dd[act, 2] * tk[act]
        below = eye[1] + dd[act, 1] * tk[act] <= surface(x, z)
        i = act[below]
        done[i] = True
        hit[i] = tk[i]
        lo[i] = prev[i]
        prev[act] = tk[act]
    a, b = lo, hit
    for _ in range(refine):
        m = (a + b) / 2
        below = eye[1] + dd[:, 1] * m <= surface(eye[0] + dd[:, 0] * m, eye[2] + dd[:, 2] * m)
        b = np.where(below, m, b)
        a = np.where(below, a, m)
    out = t.copy()
    out[found] = np.where(done, b, tg)
    return out

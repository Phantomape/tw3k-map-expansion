"""Real-geography patches inside the native map: the polish, foliage and audit side (numpy only).

Named patches (PATCHES), each built by its own Anaconda script into
analysis/<name>-patch.npz: hehuang (hehuang_patch.py: 河湟 / Qinghai Lake) and korea
(korea_patch.py: the Korean peninsula) and wuyuan (wuyuan_patch.py: the Hetao north bank under Jiuyuan, height only). This module expands them to the canvas tile
grid and applies them. load() returns the one patch present (a Patch: exactly the
single-patch behaviour) or a PatchSet that combines several with the same API (their
windows must not overlap). Every polish call site is marked "HEHUANG PATCH" so the
edits merge independently of other polish work.

Optional npz keys (absent in the Hehuang npz, which keeps the defaults below):
  height_weight   weight of the relief blend and the alpha raise (default: weight);
                  `weight` then drives only the ground (climate, donor, cache cut, trees)
  class_climate   (n, 2) ground class -> tile climate (default CLASS_CLIMATE)
  steppe_classes  ground classes that take the steppe donor (default [0])
  species         JSON {class: [[species...], dense, sparse(, trees per hit, jitter hexes)]} replanting (foliage)
  forest_floor    0..1: woodland floor painted under forest towns (polish paint_forest_floor)
  donor_mask      0..1 factor on the donor share (default 1: every class takes donor ground)
Ground class codes are global across patches (hehuang 0-5, korea 10+).

  Patch.load(name)       one patch or None (no npz, or TW3K_<NAME>=0, e.g. TW3K_HEHUANG=0)
  load(names)            the patches present: None, a Patch, or a PatchSet
  p.parts                the single patches (per-patch random streams in foliage)
  p.w / p.relief / ...   canvas tile-grid fields (zero / NaN outside the window)
  p.raise_alpha(alpha)   alpha = max(alpha, w): the projection and cache treat it as edited
  p.blend(local_relief)  local_relief*(1-w) + hh_relief*w
  p.town_target(region, default)   plot level for a patch town
  p.climate(biome, q, r) tile climate from the ground class inside the dithered w > .5
  p.steppe(...)          steppe donor (class steppe) inside the dithered w > .5
  p.donor_share(...)     donor share of the blend inside the patch
  p.cache_cut(q, r)      cache owners replaced inside the meandered w > .3 contour
  p.hex_mask(grow)       hex grid mask (weight > 0), optionally grown (audits)
"""
import json
import os

import numpy as np

import campaign_world as cw

PATCHES = ('hehuang', 'korea', 'wuyuan')


def patch_path(name):
    if name == 'korea':
        import biome_profiles as bp
        if bp.korea_forests_on():   # TW3K_KOREA_FORESTS: the forest-zone build has its own cache
            return cw.WORKSPACE / 'analysis/korea-patch-forests.npz'
    return cw.WORKSPACE / f'analysis/{name}-patch.npz'


def check_rules(name, z, path):
    """Hard stop on a korea cache built from other forest rules (npz 'rules_sha' vs biome_profiles.korea_forests_sha):
    a stale cache is never mixed in silently. Rebuild: korea_patch.py build (same TW3K_KOREA_FORESTS)."""
    if name != 'korea':
        return
    import biome_profiles as bp
    want = bp.korea_forests_sha()
    got = str(z['rules_sha']) if 'rules_sha' in z.files else None
    if got != want:
        raise RuntimeError(f'{path}: built with other korea forest rules (rules_sha {str(got)[:12]} != {want[:12]}); '
                           f'rebuild: python scripts/korea_patch.py build (TW3K_KOREA_FORESTS={int(bp.korea_forests_on())})')


def hex_tiles(q, r):
    """(ys, xs) canvas tile indices of hex (q, r): the 2 x 2 block cw.tile_hex maps to it (x 2q..2q+1, y from
    2r + q % 2)."""
    x, y = 2 * int(q), 2 * int(r) + int(q) % 2
    return np.array([y, y + 1, y, y + 1]), np.array([x, x, x + 1, x + 1])


def town_weight(w, q, r, x0, y0):
    """A patch's weight at a town: the largest of w (a window array, tile (x0, y0) at its corner) over the town hex's own
    tiles; 0 when none is in the window. The patch builds level a town above .5. Was w at round(display position),
    which lies on the hex's east edge and so sampled the east neighbour: a shore town whose east neighbour is water
    was skipped (c7 Lelang south salt, Yemaek fish; coordinator 11-01)."""
    ys, xs = hex_tiles(q, r)
    ys, xs = ys - y0, xs - x0
    ok = (ys >= 0) & (ys < w.shape[0]) & (xs >= 0) & (xs < w.shape[1])
    return float(np.max(np.asarray(w, np.float32)[ys[ok], xs[ok]])) if ok.any() else 0.


def design_towns(hex_box, plan=None):
    """[(region, name, [q, r], is_capital)] of the design regions whose town hex lies in hex_box (q0, q1, r0, r1,
    exclusive ends), sorted: the part of the region design a patch build reads (its town levelling)."""
    q0, q1, r0, r1 = (int(v) for v in hex_box)
    if plan is None:
        plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    return sorted((t['region'], t['name'], [int(v) for v in t['town']], bool(t['is_capital'])) for t in plan['regions']
                  if q0 <= t['town'][0] < q1 and r0 <= t['town'][1] < r1)


def design_sha(hex_box, plan=None):
    """Design key of a patch cache (npz 'design_sha'): sha256 of design_towns. A relayout of the window's regions
    (key, name, town hex, capital) changes it; region edits elsewhere do not."""
    import hashlib
    return hashlib.sha256(json.dumps(design_towns(hex_box, plan), ensure_ascii=False).encode('utf-8')).hexdigest()


def design_mismatch(z, plan=None):
    """None when the patch cache z matches the current region design, else the reason. With 'design_sha' (built
    since P6): equal to design_sha(hex_box). Older caches: their towns (region, name, town, capital) must be design
    towns in the window, and every design town in the window standing at patch weight > .5 (the build's own rule:
    korea height_weight, hehuang weight) must be among them."""
    if plan is None:
        plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    hb = [int(v) for v in z['hex_box']]
    if 'design_sha' in z.files:
        got, want = str(z['design_sha']), design_sha(hb, plan)
        return None if got == want else f'design_sha {got[:12]} != {want[:12]}'
    have = {t['region']: (t['region'], t['name'], [int(v) for v in t['town']], bool(t['capital'])) for t in json.loads(str(z['towns']))}
    want = {t[0]: t for t in design_towns(hb, plan)}
    bad = sorted(k for k in have if want.get(k) != have[k])
    x0, x1, y0, y1 = (int(v) for v in z['box'])
    w = z['height_weight'] if 'height_weight' in z.files else z['weight']
    for k, t in want.items():
        if k not in have and town_weight(w, *t[2], x0, y0) > .5:
            bad.append(k)
    return f'towns differ from the design: {bad[:6]}' if bad else None


def check_design(name, z, path):
    """Hard stop on a patch cache built from another region design (design_mismatch): the towns it levelled are
    baked in (P6, 2026-11-01). Rebuild: python scripts/<name>_patch.py build."""
    why = design_mismatch(z)
    if why:
        raise RuntimeError(f'{path}: built from another region design ({why}); rebuild: python scripts/{name}_patch.py build')


_FILE_SHA = {}


def _file_sha(path):
    """sha256 of a file, cached per process by (path, size, mtime)."""
    import hashlib
    st = path.stat(); k = (str(path), st.st_size, st.st_mtime_ns)
    if k not in _FILE_SHA:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''):
                h.update(b)
        _FILE_SHA[k] = h.hexdigest()
    return _FILE_SHA[k]


def input_sha(hex_box, A=None, map_cells=None, map_names=None):
    """{'map_window': sha256 of the map inside hex_box as the patch builds read it (water: kind == 1, and each hex's
    region by name), 'west_real_terrain': sha256 of analysis/west-real-terrain.npz}: the workspace inputs a patch build
    reads (single-source audit item 8, 2026-11-01). The map is keyed on the window only: map edits elsewhere do not stale
    a patch. Beach / cliff kinds (bake step 12, after the climate step) are land as for the builds, so the bake's climate
    step and the final map give one key. map_cells / map_names: the map to key (the bake passes the map it is building:
    it never reads an earlier map.hex); default the workspace's final map.hex. The vanilla map and LF (game install) are
    fixed and not keyed; the design is design_sha."""
    import hashlib
    import map_hex
    A = cw.WORKSPACE if A is None else A
    q0, q1, r0, r1 = (int(v) for v in hex_box)
    if map_cells is None:
        F = map_hex.read(A / 'work/final-map/map.hex')
        map_cells, map_names = F['cells'], F['land_regions'] + F['sea_regions']
    crop = map_cells[r0:r1, q0:q1]
    ids = map_hex.region_ids(crop)
    present = sorted({map_names[i] for i in set(ids.ravel().tolist()) if 0 <= i < len(map_names)})
    rank = {map_names.index(n): k for k, n in enumerate(present)}
    reg = np.vectorize(lambda i: rank.get(i, -1), otypes=[np.int16])(ids) if ids.size else ids.astype(np.int16)
    h = hashlib.sha256(((crop[..., 0] & 3) == 1).tobytes())
    h.update(reg.astype('<i2').tobytes())
    h.update(json.dumps(present, ensure_ascii=False).encode('utf-8'))
    return {'map_window': h.hexdigest(), 'west_real_terrain': _file_sha(A / 'analysis/west-real-terrain.npz')}


def build_rules_sha(module, extra=None):
    """sha256 of a patch build module's rule constants (every module-level UPPER_CASE int / float / str / bool / tuple /
    list / dict; paths are not rules) plus `extra` (e.g. the REGIONS entry it reads): npz 'build_rules_sha'.
    The build's code is not keyed: a change to the build logic must bump the module's VERSION (a rule constant), or the
    old caches stay 'current' and are not rebuilt (11-01: the town-hex sampling fix, VERSION 1 -> 2)."""
    import hashlib
    rules = {k: v for k, v in sorted(vars(module).items()) if k.isupper() and isinstance(v, (bool, int, float, str, tuple, list, dict))}
    return hashlib.sha256(repr((rules, extra)).encode('utf-8')).hexdigest()


def _builder(name):
    import importlib
    return importlib.import_module(f'{name}_patch')


def source_mismatch(name, z, A=None, map_cells=None, map_names=None):
    """None when the cache's recorded inputs and build rules (npz 'inputs_sha' JSON, 'build_rules_sha', written since
    2026-11-01) equal the current sources, else the reason. A cache without them (built before the keys) is stale:
    nothing could tell its inputs, so it is never reused silently (coordinator 11-01). map_cells / map_names: the map to
    check against (input_sha; the bake passes its own), default the workspace's final map.hex."""
    missing = [k for k in ('inputs_sha', 'build_rules_sha') if k not in z.files]
    if missing:
        return f'cache predates the input / rule keys (no {", ".join(missing)}): rebuild it'
    got, want = json.loads(str(z['inputs_sha'])), input_sha(z['hex_box'], A, map_cells, map_names)
    why = [f'{k} {str(got.get(k))[:12]} != {v[:12]}' for k, v in want.items() if got.get(k) != v]
    rules = _builder(name).rules_sha()
    if str(z['build_rules_sha']) != rules:
        why.append(f'build rules {str(z["build_rules_sha"])[:12]} != {rules[:12]}')
    return '; '.join(why) or None


def check_sources(name, z, path, map_cells=None, map_names=None):
    """Hard stop on a patch cache built from other inputs or rules (source_mismatch, against map_cells / map_names when
    given). Rebuild: python scripts/<name>_patch.py build."""
    why = source_mismatch(name, z, None, map_cells, map_names)
    if why:
        raise RuntimeError(f'{path}: built from other inputs or rules ({why}); rebuild: python scripts/{name}_patch.py build')


PATH = patch_path('hehuang')
# Ground class -> tile climate (0 subtropical, 1 cold, 2 temperate, 3 arid).
# --- L2048 HEHUANG CLIMATE --- steppe and alpine rock stay cold like the vanilla land around
# the patch: arid (3) on 16k tiles drew a sand-white blob whose tile-climate steps read as
# hard white/green right-angle blocks (live 2048l). The steppe look comes from the donor.
CLASS_CLIMATE = {0: 1, 1: 1, 2: 1, 3: 1, 4: 3}
GROUND_W, CACHE_W = .75, .55        # contours of the ground field: ground classes, cache replacement
# Ground field = w + broad meander + fine fray. A small dither left the ground
# switch on the jittered ellipse, read as a sand disc with a crisp rim in vanilla
# forest (offline render, first Hehuang run): the broad term (unit-variance fbm x .45, ~10 hexes, fading in the core)
# pushes the steppe edge in and out by several hexes like a real vegetation line.
MEANDER, MEANDER_SCALE, FRAY = .55, 12, .1


def blend_fields(native, hw, relief):
    """The pre-polish target ground: native * (1 - hw) + relief * hw (canvas tile grids; hw a patch height weight).
    The one formula behind Patch.blend (polish's local relief), target_ground (audit_local_town_elevations' cut
    check) and town_plateau_check's estimate (c21, coordinator / integrator 2026-11-01)."""
    return native * (1 - hw) + relief * hw


def target_ground(native, patch=None):
    """native with the loaded patches' relief blended in (Patch.blend), or native without a patch."""
    return native if patch is None else patch.blend(native)


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


class Patch:
    def __init__(self, z, name='hehuang', path=None):
        self.z = z
        self.name = name
        self.path = path or patch_path(name)
        self.x0, self.x1, self.y0, self.y1 = (int(v) for v in z['box'])
        self.q0, self.q1, self.r0, self.r1 = (int(v) for v in z['hex_box'])
        self.win = np.s_[self.y0:self.y1, self.x0:self.x1]
        self.water = float(z['water']) if 'water' in z.files else None
        self.towns = {t['region']: t for t in json.loads(str(z['towns']))}
        self.prop = json.loads(str(z['prop'])) if 'prop' in z.files else None
        self.version = int(z['version'])
        self.sha = str(z['sha'])
        self.class_climate = {int(c): int(v) for c, v in z['class_climate']} if 'class_climate' in z.files else dict(CLASS_CLIMATE)
        self.steppe_classes = [int(c) for c in z['steppe_classes']] if 'steppe_classes' in z.files else [0]
        self.species = {int(c): (tuple(v[0]),) + tuple(v[1:]) for c, v in json.loads(str(z['species'])).items()} if 'species' in z.files else {}
        self._full = {}

    @staticmethod
    def load(name='hehuang', map_cells=None, map_names=None):
        """map_cells / map_names: the map the cache is checked against (check_sources); the bake passes the map it is
        building, every later stage leaves them None (the workspace's final map.hex)."""
        path = patch_path(name)
        if os.environ.get('TW3K_' + name.upper(), '1') == '0':
            return None
        if not path.exists():
            if name == 'korea' and os.environ.get('TW3K_KOREA_FORESTS', '0') == '1':
                raise RuntimeError(f'{path} missing (TW3K_KOREA_FORESTS=1): python scripts/korea_patch.py build')
            return None
        z = np.load(path)
        check_rules(name, z, path)
        check_design(name, z, path)
        check_sources(name, z, path, map_cells, map_names)
        return Patch(z, name, path)

    @property
    def parts(self):
        return [self]

    def in_window(self, tx, ty):
        """Tile cells (arrays) inside this patch's window."""
        return (tx >= self.x0) & (tx < self.x1) & (ty >= self.y0) & (ty < self.y1)

    def _canvas(self, key, fill, dtype):
        if key not in self._full:
            a = np.full((cw.TILE_H, cw.TILE_W), fill, dtype)
            src = 'weight' if key == 'height_weight' and key not in self.z.files else key
            if src in self.z.files:
                a[self.win] = self.z[src].astype(dtype)
            self._full[key] = a
        return self._full[key]

    @property
    def w(self):
        return self._canvas('weight', 0, np.float32)

    @property
    def hw(self):
        """Height weight (relief blend, alpha raise); the ground weight w when the npz has none."""
        return self._canvas('height_weight', 0, np.float32)

    @property
    def relief(self):
        return self._canvas('relief', cw.LEVEL, np.float32)

    @property
    def ground(self):
        return self._canvas('ground', 255, np.uint8)

    @property
    def lake(self):
        return self._canvas('lake', False, bool)

    @property
    def shore(self):
        return self._canvas('shore', 0, np.float32)

    def _dithered(self, q, r, seed=None):
        """The ground field (one field for climate, donor, cache cut and tree thinning)."""
        if '_gf' not in self._full:
            from terrain_noise import fbm
            w = self.w
            out = np.zeros_like(w)
            win = self.win
            qq, rr = q[win], r[win]
            nf = fbm(qq, rr, (0, 0), MEANDER_SCALE, 3, seed=221)
            nf = (nf - nf.mean()) / max(float(nf.std()), 1e-6)          # unit-variance meander
            ww = w[win]
            # The meander spans the whole blend band and fades in the core (no forest islands there).
            g = ww + MEANDER * nf * (1 - .6 * ww * ww) + FRAY * fbm(qq, rr, (0, 0), 1.5, 2, seed=222)
            out[win] = np.where(w[win] > 0, g, 0)
            self._full['_gf'] = out.astype(np.float32)
        return self._full['_gf']

    def ground_field(self):
        """The ground field on the canvas tile grid (built on first use)."""
        y, x = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        q = np.zeros((cw.TILE_H, cw.TILE_W), np.float32); r = np.zeros_like(q)
        q[self.win], r[self.win] = x / 2, y / 2
        return self._dithered(q, r)

    # -- height ---------------------------------------------------------------
    def raise_alpha(self, alpha, protected=None):
        w = self.hw
        if protected is not None:
            # hehuang_patch fades the weight out 8 hexes before vanilla playable land; the
            # polish protection rule is the authority, so any overlap is zeroed and counted.
            self.protected_overlap = int(((w > 0) & protected).sum())
            w = np.where(protected, 0, w)
        return np.maximum(alpha, w)

    def blend(self, local_relief):
        return blend_fields(local_relief, self.hw, self.relief)

    def town_target(self, region, default):
        t = self.towns.get(region)
        return float(t['level']) if t else default

    # -- ground ----------------------------------------------------------------
    def climate_mask(self, q, r):
        """Cells whose tile climate the patch sets (dithered w > GROUND_W, land classes)."""
        if '_cm' not in self._full:
            self._full['_cm'] = (self._dithered(q, r, 221) > GROUND_W) & np.isin(self.ground, list(self.class_climate))
        return self._full['_cm']

    def climate(self, biome, q, r):
        m = self.climate_mask(q, r)
        lut = np.zeros(256, biome.dtype)
        for c, v in self.class_climate.items():
            lut[c] = v
        out = biome.copy()
        out[m] = lut[self.ground[m]]
        return out

    def steppe(self, q, r):
        return (self._dithered(q, r, 221) > GROUND_W) & np.isin(self.ground, self.steppe_classes)

    @property
    def donor_mask(self):
        """Optional npz `donor_mask` (korea: open classes only; forest keeps vanilla's woodland floor), else 1."""
        return self._canvas('donor_mask', 1, np.float32)

    def donor_share(self):
        g = self._full.get('_gf', self.w)
        return (_smooth((g - .2) / .5) * (self.w > 0) * self.donor_mask).astype(np.float32)

    def paint_forest_floor(self, cur, woodland=(4, 5), material=4):
        """Blend pixels (cur: global_blend rows north-up, 4 per tile cell) of npz `forest_floor`
        (dithered by its value) that hold an open material take the woodland floor: forest
        towns sit on it like vanilla's (no light disc). Returns the pixels changed."""
        m = self._canvas('forest_floor', 0, np.float32)
        ys, xs = np.nonzero(m > 0)
        if not len(ys):
            return 0
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        # --- FIX2 KOREA GROUND --- bilinear (not per-tile 4x4 blocks) and dithered by fbm grain
        # (not 2x2-pixel white noise): the disc edge read as tile stairs and speckle (offline 2048bq).
        from PIL import Image
        from terrain_noise import fbm
        tile = m[y0:y1, x0:x1][::-1].astype(np.float32)
        sub = np.asarray(Image.fromarray(tile, 'F').resize((tile.shape[1] * 4, tile.shape[0] * 4), Image.BILINEAR), np.float32)
        oh = cur.shape[0]
        r0 = oh - 4 * y1
        win = cur[r0:r0 + sub.shape[0], 4 * x0:4 * x0 + sub.shape[1], 0]
        yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
        u = np.clip(.5 + 1.6 * fbm((4 * x0 + xx) / 8., (oh - 1 - r0 - yy) / 8., (0, 0), 1.5, 3, seed=4401), 0, 1)
        # --- end FIX2 KOREA GROUND
        sel = (u < sub) & ~np.isin(win, list(woodland))
        win[sel] = material
        return int(sel.sum())

    def oasis(self):
        return (self.shore * .7 * self.w).astype(np.float32)

    def cache_cut(self, q, r):
        return self._dithered(q, r, 223) > CACHE_W

    # -- masks -----------------------------------------------------------------
    def _hex0(self):
        m = np.zeros((cw.H, cw.W), bool)
        m[self.r0:self.r1, self.q0:self.q1] = self.z['hex_mask']
        return m

    def hex_mask(self, grow=0):
        m = self._hex0()
        for _ in range(grow):
            g = m.copy()
            g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
            m = g
        return m

    def tile_mask(self, grow=0):
        hm = self.hex_mask(grow)
        q, r = cw.tile_hex(np.arange(cw.TILE_W, dtype=np.int32)[None, :], np.arange(cw.TILE_H, dtype=np.int32)[:, None])
        return hm[r, q]

    def report(self):
        return {'npz': str(self.path), 'version': self.version, 'sha': self.sha, 'water': self.water,
                'hexes_weight_gt0': int(self.z['hex_mask'].sum()), 'protected_overlap_cells': getattr(self, 'protected_overlap', None), 'towns': {k: round(v['level'], 3) for k, v in self.towns.items()}}


class PatchSet(Patch):
    """Several patches behind the Patch API: canvas fields pasted per window, class maps
    merged (codes are global), masks or-ed, towns merged; `parts` gives foliage its
    per-patch random streams. Each part's own fields are computed exactly as when alone."""
    def __init__(self, parts):
        self.part_list = list(parts)
        for i, a in enumerate(self.part_list):
            for b in self.part_list[i + 1:]:
                assert a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0, ('patch windows overlap', a.name, b.name)
        self.name = '+'.join(p.name for p in self.part_list)
        self.towns, self.class_climate, self.species, self.steppe_classes = {}, {}, {}, []
        for p in self.part_list:
            assert not set(p.towns) & set(self.towns), ('town in two patches', p.name)
            assert not set(p.class_climate) & set(self.class_climate), ('ground class codes collide', p.name)
            self.towns.update(p.towns)
            self.class_climate.update(p.class_climate)
            self.species.update(p.species)
            self.steppe_classes += p.steppe_classes
        own = [p for p in self.part_list if p.prop is not None]
        self.prop = own[0].prop if own else None          # hehuang_lake_prop: the Hehuang lake
        self.water = own[0].water if own else None
        self._full = {}

    @property
    def parts(self):
        return self.part_list

    def _canvas(self, key, fill, dtype):
        if key not in self._full:
            a = np.full((cw.TILE_H, cw.TILE_W), fill, dtype)
            for p in self.part_list:
                a[p.win] = p._canvas(key, fill, dtype)[p.win]
            self._full[key] = a
        return self._full[key]

    def _dithered(self, q, r, seed=None):
        if '_gf' not in self._full:
            out = np.zeros((cw.TILE_H, cw.TILE_W), np.float32)
            for p in self.part_list:
                out[p.win] = p._dithered(q, r, seed)[p.win]
            self._full['_gf'] = out
        return self._full['_gf']

    def ground_field(self):
        if '_gf' not in self._full:
            out = np.zeros((cw.TILE_H, cw.TILE_W), np.float32)
            for p in self.part_list:
                out[p.win] = p.ground_field()[p.win]
            self._full['_gf'] = out
        return self._full['_gf']

    def _hex0(self):
        return np.logical_or.reduce([p._hex0() for p in self.part_list])

    def raise_alpha(self, alpha, protected=None):
        out = Patch.raise_alpha(self, alpha, protected)
        if protected is not None:
            for p in self.part_list:
                p.protected_overlap = int(((p.hw > 0) & protected).sum())
        return out

    def report(self):
        return {p.name: p.report() for p in self.part_list}


def load(names=PATCHES, map_cells=None, map_names=None):
    """The named patches present: None, the one Patch, or a PatchSet of several. map_cells / map_names: the map the
    caches are checked against (the bake's own map; default the workspace's final map.hex)."""
    parts = [p for p in (Patch.load(n, map_cells, map_names) for n in names) if p is not None]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else PatchSet(parts)

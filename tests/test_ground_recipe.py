"""Ground mottle step 1 (user decision 2026-11-01): the v24 ground recipe is the default, the v23 recipe stays as a named
version behind TW3K_GROUND_RECIPE=v23 (snapshot byte-identical to its frozen copy), the recipe name keys polish's setup
memo, the v24 mixes are close in tone and carry no mud, and ground_patchiness.fragmentation measures what the invariant
ground_fragmentation compares."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import biome_profiles as bp  # noqa: E402

TUNDRA_OWNED = ('tundra',)   # tundra keeps its own recipe and switches (TW3K_TUNDRA_SCREE / TW3K_TUNDRA_BAN)


def _snapshot(recipe):
    env = dict(os.environ, TW3K_GROUND_RECIPE=recipe)
    code = 'import json, biome_profiles as bp; print(json.dumps(bp.snapshot()))'
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT / 'scripts', env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _parts(spec):
    """The material dicts of a landform under the current table: its mix and its own relief parts."""
    out = [('mix', spec['mix'])]
    rel = spec.get('relief')
    if rel and rel.get('mix'):
        out += [(p, m) for p, m in rel['mix'].items()]
    return out


class RecipeSwitch(unittest.TestCase):
    def test_v23_is_the_frozen_old_table(self):
        frozen = json.loads((ROOT / 'analysis/biome-profiles-frozen-v23.json').read_text(encoding='utf-8'))['snapshot']
        self.assertEqual(_snapshot('v23'), frozen)

    def test_default_is_v24_and_frozen(self):
        frozen = json.loads((ROOT / 'analysis/biome-profiles-frozen.json').read_text(encoding='utf-8'))['snapshot']
        self.assertEqual(_snapshot('v24'), frozen)
        self.assertEqual(bp.GROUND_RECIPE_DEFAULT, 'v24')

    def test_unknown_recipe_refused(self):
        env = dict(os.environ, TW3K_GROUND_RECIPE='v99')
        r = subprocess.run([sys.executable, '-c', 'import biome_profiles'], cwd=ROOT / 'scripts', env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('TW3K_GROUND_RECIPE', r.stderr)

    def test_recipe_keys_the_polish_setup_memo(self):
        src = (ROOT / 'scripts/polish_campaign_terrain.py').read_text(encoding='utf-8')
        i = src.index('def _setup_memo_key():')
        body = src[i:src.index('return h.hexdigest()', i)]
        self.assertIn('_bp.ground_recipe()', body)

    def test_v23_rollback_names_every_changed_value(self):
        """Every landform whose mix / relief differs between v23 and v24 is listed in GROUND_RECIPE_V23 (the switch
        restores exactly the listed ones; the snapshot test catches a wrong value, this one a forgotten landform)."""
        v23, v24 = _snapshot('v23'), _snapshot('v24')
        changed = {k for k in set(v24) | set(v23) if v24.get(k) != v23.get(k)}
        self.assertEqual(changed, {'landforms.mixes', 'landforms.relief', 'consts.farm_fields.ARID_OWN_OASIS',
                                   'landforms.grains', 'landforms.redraw', 'consts.ground_materials.ARID_TOWN',
                                   'consts.ground_materials.FARM_OASES_SKIP'})


@unittest.skipUnless(bp.ground_recipe() == 'v24', 'the v24 recipe is not loaded (TW3K_GROUND_RECIPE)')
class V24Mixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ground_patchiness as gp
        cls.lab = gp.Tones('harvest').lab

    def de(self, a, b):
        return float(np.linalg.norm(self.lab[a] - self.lab[b]))

    def test_no_mud_in_landform_mixes(self):
        for c, spec in bp.LANDFORMS.items():
            if c in TUNDRA_OWNED:
                continue
            for part, m in _parts(spec):
                self.assertNotIn(14, m, f'{c} {part}')

    def test_close_tones(self):
        """At most 4 materials per mix / relief part, pairwise dE76 <= 20 (harvest tones)."""
        for c, spec in bp.LANDFORMS.items():
            if c in TUNDRA_OWNED:
                continue
            for part, m in _parts(spec):
                self.assertLessEqual(len(m), 4, f'{c} {part}')
                worst = max((self.de(a, b) for a in m for b in m), default=0.)
                self.assertLessEqual(worst, 20., f'{c} {part}: {m}')

    def test_expected_contrast_not_above_v23(self):
        """Expected dE of two random pixels of the base mix, sum p_i p_j dE_ij: v24 <= v23 for every landform."""
        E = lambda m: sum(pa * pb * self.de(a, b) for a, pa in m.items() for b, pb in m.items()) / sum(m.values()) ** 2
        for c, old in bp.GROUND_RECIPE_V23['landforms'].items():
            self.assertLessEqual(E(bp.LANDFORMS[c]['mix']), E(old['mix']) + 1e-9, c)

    def test_shares_sum_to_one(self):
        for c, spec in bp.LANDFORMS.items():
            for part, m in _parts(spec):
                self.assertAlmostEqual(sum(m.values()), 1., places=6, msg=f'{c} {part}')


class Fragmentation(unittest.TestCase):
    def test_synthetic(self):
        """Two landforms on 8 x 4 tiles (32 x 16 px): the west half one slot (no edges, 1 patch), the east half a
        2 px checkerboard of two slots (edges and 64 patches), counted only inside each class."""
        import ground_patchiness as gp

        class Src:
            h, w = 16, 32

            def __init__(self, R):
                self.R = R

            def rows(self, name, y0, y1):
                return np.stack([self.R[y0:y1], np.zeros_like(self.R[y0:y1])], -1)

        R = np.zeros((16, 32), np.uint8)
        yy, xx = np.mgrid[:16, :16]
        R[:, 16:] = np.where(((yy // 2) + (xx // 2)) % 2 == 0, 2, 21)
        dom = np.zeros((4, 8), np.uint8); dom[:, :4] = 1; dom[:, 4:] = 2
        domw = np.ones((4, 8), np.float32)
        lab = np.zeros((256, 3), np.float32); lab[2] = (60, 0, 30); lab[21] = (50, 0, 30)
        T = type('T', (), {'lab': lab})()
        gp_min = gp.FRAG_MIN_HEX
        try:
            gp.FRAG_MIN_HEX = 0
            r = gp.fragmentation(Src(R), dom, domw, T, ['a', 'b'], strip=16)
        finally:
            gp.FRAG_MIN_HEX = gp_min
        self.assertEqual(r['a']['E'], 0.)
        self.assertEqual(r['a']['n100'], round(1 / 4. * 100, 2))   # 1 patch on 4 hexes
        # east: 8 checker cells per row band of 2 px -> 7 vertical edges of 2 px per band x 8 bands, likewise horizontal
        pairs = 2 * 7 * 2 * 8
        self.assertAlmostEqual(r['b']['E'], round(pairs * 10. / 4., 3))
        self.assertEqual(r['b']['n100'], round(64 / 4. * 100, 2))


    def test_groups_split(self):
        """groups splits pairs and patches: a uniform image cut into two groups gives one patch and no edge in each."""
        import ground_patchiness as gp

        class Src:
            h, w = 16, 32

            def rows(self, name, y0, y1):
                R = np.full((y1 - y0, 32), 2, np.uint8)
                return np.stack([R, np.zeros_like(R)], -1)

        dom = np.ones((4, 8), np.uint8); domw = np.ones((4, 8), np.float32)
        groups = np.zeros((4, 8), np.uint8); groups[:, 4:] = 1
        T = type('T', (), {'lab': np.zeros((256, 3), np.float32)})()
        gp_min = gp.FRAG_MIN_HEX
        try:
            gp.FRAG_MIN_HEX = 0
            r = gp.fragmentation(Src(), dom, domw, T, ['a'], strip=16, groups=groups)
        finally:
            gp.FRAG_MIN_HEX = gp_min
        self.assertEqual(set(r), {0, 1})
        for g in (0, 1):
            self.assertEqual(r[g]['a']['E'], 0.)
            self.assertEqual(r[g]['a']['n100'], 25.)

    def test_hex_dilate_and_tiles(self):
        import ground_patchiness as gp
        m = np.zeros((9, 9), bool); m[4, 4] = True
        self.assertEqual(int(gp.hex_dilate(m, 1).sum()), 7)
        t = gp.hex_to_tiles(m, (18, 18))
        self.assertEqual(int(t.sum()), 4)                      # one hex = 2 x 2 tiles
        self.assertTrue(t[8:10, 8:10].all())                    # even column q 4: tiles 8-9, rows 2r .. 2r+1
        m2 = np.zeros((9, 9), bool); m2[4, 3] = True
        self.assertTrue(gp.hex_to_tiles(m2, (18, 18))[9:11, 6:8].all())   # odd column: rows 2r+1 .. 2r+2



class GrainJoin(unittest.TestCase):
    """[GRAIN PER LANDFORM] ground_materials._grain_join: the Gaussian copula of the per-grain uniforms."""

    def setUp(self):
        import ground_materials as gm
        self.join = gm._grain_join
        self.rng = np.random.default_rng(7)

    def test_default_share_one_keeps_bytes(self):
        u0 = self.rng.random(1000).astype(np.float32)
        gs = np.zeros((2, 1000), np.float32); gs[0] = 1
        out = self.join(u0, gs, [1], [self.rng.random(1000).astype(np.float32)])
        self.assertTrue(np.array_equal(out, u0))

    def test_uniform_for_any_shares(self):
        """The joined value stays uniform (a material's share in the mix holds) at every share split."""
        n = 200000
        for s1 in (.25, .5, .9, 1.):
            gs = np.zeros((2, n), np.float32); gs[0] = 1 - s1; gs[1] = s1
            out = self.join(self.rng.random(n).astype(np.float32), gs, [1], [self.rng.random(n).astype(np.float32)])
            hist = np.histogram(out, bins=10, range=(0, 1))[0] / n
            self.assertLess(np.abs(hist - .1).max(), .005, s1)

    def test_continuous_in_shares(self):
        """A small change of the shares moves the value a little (no seam where landforms meet)."""
        u0, u1 = np.full(1, .3, np.float32), np.full(1, .8, np.float32)
        vals = [float(self.join(u0, np.array([[1 - t], [t]], np.float32), [1], [u1])[0]) for t in np.linspace(0, 1, 101)]
        self.assertLess(max(abs(a - b) for a, b in zip(vals, vals[1:])), .05)
        self.assertAlmostEqual(vals[-1], .8, places=4)

    def test_no_step_at_boundary(self):
        """A strip across a landform boundary (grain 3 west, 7 east, shares ramping over 8 hexes): the pixel-to-pixel
        change of the joined uniform has no peak at the boundary (gradient statistic: the change rate in the ramp
        stays within the range of the two sides)."""
        from terrain_noise import fbm
        W, Hh = 480, 800                                   # pixels (8 per hex): 60 x 100 hexes
        y, x = np.mgrid[:Hh, :W].astype(np.float64) / 8.
        u3, u7 = (fbm(x, y, (0, 0), g, 3, seed=s) for g, s in ((3., 230), (7., 1230)))
        u3, u7 = (np.argsort(np.argsort(v, axis=None)).reshape(v.shape) / v.size for v in (u3, u7))   # rank -> uniform
        t = np.clip((x[0] - 26) / 8, 0, 1)                 # east share 0 -> 1 over hexes 26..34
        gs = np.stack([np.broadcast_to(1 - t, (Hh, W)), np.broadcast_to(t, (Hh, W))]).reshape(2, -1).astype(np.float32)
        u = self.join(u3.ravel().astype(np.float32), gs, [1], [u7.ravel().astype(np.float32)]).reshape(Hh, W)
        rate = np.abs(np.diff(u, axis=1)).mean(0)        # mean horizontal change per column
        cols = np.arange(W - 1) / 8.
        west, east, ramp = rate[cols < 22], rate[cols > 38], rate[(cols >= 26) & (cols <= 34)]
        self.assertGreater(west.mean(), east.mean())       # the finer grain changes faster
        # column rates in the ramp stay inside the range the two sides span (same statistic on both sides)
        self.assertLessEqual(ramp.max(), west.max())       # no spike at the boundary
        self.assertGreaterEqual(ramp.min(), east.min())    # no dead band either


class GrainSnapshot(unittest.TestCase):
    def test_no_grain_key_without_grains(self):
        snap = _snapshot('v23')
        self.assertNotIn('landforms.grains', snap)
        self.assertNotIn('landforms.redraw', snap)

    def test_redraw_is_graded_not_banned(self):
        """[DARK REDRAW] the dark donors are redrawn through the take weight (graded edge), not the tile-rint ban: the
        ban table is the v23 one."""
        code = 'import json, biome_profiles as bp; print(json.dumps([list(bp.BAN_ENFORCED), sorted(bp.redraws())]))'
        out = lambda r: json.loads(subprocess.run([sys.executable, '-c', code], cwd=ROOT / 'scripts', capture_output=True, text=True,
                                                  check=True, env=dict(os.environ, TW3K_GROUND_RECIPE=r)).stdout)
        (b24, r24), (b23, r23) = out('v24'), out('v23')
        self.assertEqual(b24, b23)
        self.assertEqual(r24, ['alpine_meadow', 'high', 'plateau'])
        self.assertEqual(r23, [])


if __name__ == '__main__':
    unittest.main()


def _arid_town_ramp(recipe):
    env = dict(os.environ, TW3K_GROUND_RECIPE=recipe, TW3K_GPU='0')
    code = 'import json, ground_materials as gm, biome_profiles as bp; print(json.dumps([gm.ARID_TOWN, bp.CONSTS["farm_fields"]["ARID_OWN_OASIS"]]))'
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT / 'scripts', env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


_FARM_CODE = """
import json, numpy as np, ground_materials as gm, biome_profiles as bp
H = W = 64
bio = np.zeros((H, W), np.int32)
yy, xx = np.mgrid[:H, :W].astype(np.float32)
z = lambda: np.zeros((H, W), bool)
out = gm.farm_mud_fields(np.full((H, W), 500., np.float32), bio, ['desert'], np.full((H, W), 50., np.float32),
                         np.full((H, W), 30., np.float32), np.ones((H, W), np.float32), np.full((H, W), 3, np.uint8),
                         np.zeros((H, W), np.float32), z(), z(), z(), [(32, 32, 'farm')], xx / 2, yy / 2, (0., 0.))
d = np.hypot(xx - 32, yy - 32) * .5
F = out['F']
own = bp.CONSTS['farm_fields']['ARID_OWN_OASIS']
print(json.dumps({'own': float(F[(d > 1.6) & (d < own)].max()), 'beyond': float(F[d > gm.ARID_TOWN_REACH].max()),
                  'at5': float(F[(d > 4.5) & (d < 5.5)].max())}))
"""


class AridTownFarms(unittest.TestCase):
    """[ARID FARM] (C2 review 2026-11-01): an arid town's own fields reach farm_fields ARID_OWN_OASIS (one constant for both
    farm writers); beyond it arid fields need real water. TW3K_GROUND_RECIPE=v23 keeps the 9-hex disc."""

    def test_farm_rule_single_source(self):
        src = (ROOT / 'scripts/biome_profiles.py').read_text(encoding='utf-8')
        self.assertEqual(src.count("'ARID_OWN_OASIS':"), 1)
        ramp, own = _arid_town_ramp('v24')
        self.assertEqual(ramp[2:], [own, own + .5])
        self.assertEqual(_arid_town_ramp('v23')[0], [1.2, 2.5, 4.5, 9.0])

    def test_arid_town_fields_need_water_beyond_own_disc(self):
        """A farm town on flat desert with no river, lake or oasis: v24 fields only within its own reach (ARID_TOWN_REACH:
        ARID_OWN_OASIS + .5, over 1 - TOWN_WARP)."""
        res = {}
        for recipe in ('v24', 'v23'):
            o = subprocess.run([sys.executable, '-c', _FARM_CODE], cwd=ROOT / 'scripts',
                               env=dict(os.environ, TW3K_GPU='0', TW3K_GROUND_RECIPE=recipe), capture_output=True, text=True, check=True)
            res[recipe] = json.loads(o.stdout.strip().splitlines()[-1])
        self.assertGreater(res['v24']['own'], 0)
        self.assertEqual(res['v24']['beyond'], 0)
        self.assertGreater(res['v23']['at5'], 0)   # the old disc: fields 5 hexes out on dry ground


class TurpanDisc(unittest.TestCase):
    def test_turpan_disc_waters_no_fields(self):
        """[ARID FARM] the Turpan OASES disc is out of the farm oasis term (v24); its other readers keep every disc."""
        code = ('import json, numpy as np, ground_materials as gm; lo, la, _ = gm.OASES["Turpan"]; '
                'a = np.full((8, 8), lo, np.float32); b = np.full((8, 8), la, np.float32); y, x = np.mgrid[:8, :8].astype(np.float32); '
                'print(json.dumps([float(gm.oasis_mask(a, b, x, y, (0., 0.)).max()), '
                'float(gm.oasis_mask(a, b, x, y, (0., 0.), skip=gm.FARM_OASES_SKIP).max()), list(gm.FARM_OASES_SKIP)]))')
        got = {}
        for recipe in ('v24', 'v23'):
            o = subprocess.run([sys.executable, '-c', code], cwd=ROOT / 'scripts', env=dict(os.environ, TW3K_GPU='0', TW3K_GROUND_RECIPE=recipe),
                               capture_output=True, text=True, check=True)
            got[recipe] = json.loads(o.stdout.strip().splitlines()[-1])
        self.assertEqual(got['v24'][2], ['Turpan']); self.assertGreater(got['v24'][0], .9); self.assertEqual(got['v24'][1], 0)
        self.assertEqual(got['v23'][2], []); self.assertGreater(got['v23'][1], .9)
        src = (ROOT / 'scripts/ground_materials.py').read_text(encoding='utf-8')
        self.assertEqual(src.count('oasis_mask('), 2)   # the definition and farm_mud_fields' one call


class FarmOnDryGround(unittest.TestCase):
    def test_registered_hard_after_polish_and_sync_final(self):
        import stage_invariants as si
        spec = si.INVARIANTS['farm_on_dry_ground']
        self.assertEqual(spec[2], ('polish', 'sync_final'))
        self.assertIn(si.WEST_REAL_TERRAIN, spec[1])
        self.assertFalse(spec[6])   # not warn-only

    def test_threshold_per_recipe(self):
        """Each recipe carries its own threshold (the v23 rollback must not be locked by the v24 value)."""
        code = 'import stage_invariants as si; print(si.INVARIANTS["farm_on_dry_ground"][4])'
        got = {}
        for recipe in ('v24', 'v23'):
            o = subprocess.run([sys.executable, '-c', code], cwd=ROOT / 'scripts', env=dict(os.environ, TW3K_GROUND_RECIPE=recipe),
                               capture_output=True, text=True, check=True)
            got[recipe] = int(o.stdout.strip().splitlines()[-1])
        self.assertEqual(got, {'v24': 315, 'v23': 38415})

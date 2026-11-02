"""P6 (2026-11-01): patch caches follow the region design (hehuang_terrain.design_towns / design_sha /
design_mismatch, invariant patch_design) and the Hehuang lake towns come from the design, not a key list.
Synthetic inputs only."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_ENV = {}
_T = _TMP = None
np = cw = ht = si = None


def setUpModule():
    """Environment set here and restored in tearDownModule, not at import: the full suite shares one process."""
    global _T, _TMP, np, cw, ht, si
    _TMP = tempfile.TemporaryDirectory(prefix='tw3k-patch-design-tests-')
    _T = Path(_TMP.name)
    (_T / 'design.json').write_text(json.dumps({'grid': [892 + 80, 702 + 20], 'padding': [20, 10, 60, 10]}), encoding='utf-8')
    for k, v in (('TW3K_DESIGN', str(_T / 'design.json')), ('TW3K_WORKSPACE', str(_T))):
        _ENV[k] = os.environ.get(k); os.environ[k] = v
    if str(ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT / 'scripts'))
    import numpy as np
    import campaign_world as cw
    import hehuang_terrain as ht
    import stage_invariants as si


def tearDownModule():
    for k, v in _ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _TMP.cleanup()

BOX = (100, 140, 50, 80)   # hex window


def plan():
    return {'regions': [
        {'region': 'a_capital', 'name': 'A', 'town': [110, 60], 'is_capital': True},
        {'region': 'a_res', 'name': 'B', 'town': [120, 70], 'is_capital': False},
        {'region': 'far', 'name': 'F', 'town': [10, 10], 'is_capital': False}]}


def npz(towns, weight=1., key=None):
    q0, q1, r0, r1 = BOX
    x0, x1, y0, y1 = 2 * q0, 2 * q1, 2 * r0, 2 * r1
    d = dict(hex_box=np.array(BOX), box=np.array([x0, x1, y0, y1]), weight=np.full((y1 - y0, x1 - x0), weight, np.float16),
             towns=json.dumps(towns))
    if key:
        d['design_sha'] = key
    p = Path(tempfile.mkdtemp(dir=_T)) / 'p.npz'
    np.savez(p, **d)
    return np.load(p)


def towns_of(pl):
    return [dict(region=t['region'], name=t['name'], town=t['town'], capital=t['is_capital']) for t in pl['regions'] if t['region'] != 'far']


class PatchDesign(unittest.TestCase):
    def test_design_towns_window_only(self):
        self.assertEqual([t[0] for t in ht.design_towns(BOX, plan())], ['a_capital', 'a_res'])
        p2 = plan(); p2['regions'][2]['town'] = [11, 10]
        self.assertEqual(ht.design_sha(BOX, plan()), ht.design_sha(BOX, p2))   # edits outside the window: same key
        p3 = plan(); p3['regions'][1]['town'] = [121, 70]
        self.assertNotEqual(ht.design_sha(BOX, plan()), ht.design_sha(BOX, p3))

    def test_keyed_cache(self):
        z = npz(towns_of(plan()), key=ht.design_sha(BOX, plan()))
        self.assertIsNone(ht.design_mismatch(z, plan()))
        p3 = plan(); p3['regions'][0]['name'] = 'A2'
        self.assertIn('design_sha', ht.design_mismatch(z, p3))

    def test_legacy_cache(self):
        z = npz(towns_of(plan()))
        self.assertIsNone(ht.design_mismatch(z, plan()))
        p3 = plan(); p3['regions'][1]['town'] = [121, 70]               # a levelled town moved
        self.assertIn('a_res', ht.design_mismatch(z, p3))
        p4 = plan(); p4['regions'].append({'region': 'new', 'name': 'N', 'town': [130, 75], 'is_capital': False})
        self.assertIn('new', ht.design_mismatch(z, p4))                  # a new town at weight > .5 not levelled
        self.assertIsNone(ht.design_mismatch(npz(towns_of(plan()), weight=.3), p4))   # below .5: the build skips it too

    def test_load_stops_on_stale(self):
        z = npz(towns_of(plan()), key='0' * 64)
        orig = cw.DESIGN_PATH
        try:
            cw.DESIGN_PATH = _T / 'plan.json'; cw.DESIGN_PATH.write_text(json.dumps(plan()), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                ht.check_design('hehuang', z, 'p.npz')
        finally:
            cw.DESIGN_PATH = orig

    def keyed(self, inputs, rules):
        p = Path(tempfile.mkdtemp(dir=_T)) / 'k.npz'
        np.savez(p, hex_box=np.array(BOX), inputs_sha=json.dumps(inputs), build_rules_sha=rules)
        return np.load(p)

    def test_source_mismatch(self):
        cur = {'map_window': 'a' * 64, 'west_real_terrain': 'b' * 64}
        rules = types.SimpleNamespace(rules_sha=lambda: 'r' * 64)
        with mock.patch.object(ht, 'input_sha', lambda hb, A=None, mc=None, mn=None: dict(cur)), mock.patch.object(ht, '_builder', lambda n: rules):
            self.assertIsNone(ht.source_mismatch('hehuang', self.keyed(cur, 'r' * 64)))
            self.assertIn('map_window', ht.source_mismatch('hehuang', self.keyed(dict(cur, map_window='c' * 64), 'r' * 64)))
            self.assertIn('west_real_terrain', ht.source_mismatch('hehuang', self.keyed(dict(cur, west_real_terrain='c' * 64), 'r' * 64)))
            self.assertIn('build rules', ht.source_mismatch('hehuang', self.keyed(cur, 's' * 64)))
            with self.assertRaises(RuntimeError):
                ht.check_sources('hehuang', self.keyed(cur, 's' * 64), 'p.npz')
            self.assertIn('predates', ht.source_mismatch('hehuang', npz(towns_of(plan()))))   # no keys: stale, rebuild
            with self.assertRaises(RuntimeError):
                ht.check_sources('hehuang', npz(towns_of(plan())), 'p.npz')

    def test_build_rules_sha(self):
        m = types.ModuleType('m'); m.BAND, m.A, m.helper = 9., Path('x'), 1
        s0 = ht.build_rules_sha(m)
        m.A, m.helper = Path('y'), 2                       # paths and lower-case names are not rules
        self.assertEqual(ht.build_rules_sha(m), s0)
        m.BAND = 9.5
        self.assertNotEqual(ht.build_rules_sha(m), s0)
        self.assertNotEqual(ht.build_rules_sha(m, {'k': 1}), ht.build_rules_sha(m, {'k': 2}))

    def test_version_bump_stales_old_caches(self):
        # the build's code is not keyed: VERSION carries a logic change (2 = town weight on the town hex). A cache keyed
        # with the previous VERSION (its inputs current) must be stale, so the first bake rebuilds it.
        import hehuang_patch, korea_patch
        cur = {'map_window': 'a' * 64, 'west_real_terrain': 'b' * 64}
        for name, mod in (('hehuang', hehuang_patch), ('korea', korea_patch)):
            self.assertGreaterEqual(mod.VERSION, 2, name)
            new = mod.rules_sha()
            with mock.patch.object(mod, 'VERSION', mod.VERSION - 1):
                old = mod.rules_sha()
            self.assertNotEqual(old, new, name)
            with mock.patch.object(ht, 'input_sha', lambda hb, A=None, mc=None, mn=None: dict(cur)):
                self.assertIn('build rules', ht.source_mismatch(name, self.keyed(cur, old)), name)
                self.assertIsNone(ht.source_mismatch(name, self.keyed(cur, new)), name)

    def test_builds_write_the_keys(self):
        for f in ('hehuang_patch.py', 'korea_patch.py', 'wuyuan_patch.py'):
            src = (ROOT / 'scripts' / f).read_text(encoding='utf-8')
            for k in ('design_sha=', 'inputs_sha=', 'build_rules_sha=rules_sha()'):
                self.assertIn(k, src, (f, k))

    def test_hex_tiles_match_tile_hex(self):
        q0, r0 = cw.PAD_LEFT + 40, cw.PAD_BOTTOM + 40   # inside whatever canvas the suite loaded
        for q, r in ((q0, r0), (q0 + 1, r0), (q0 + 2, r0 + 3), (q0 + 3, r0 + 3)):
            ys, xs = ht.hex_tiles(q, r)
            hq, hr = cw.tile_hex(xs, ys)
            self.assertTrue((np.asarray(hq) == q).all() and (np.asarray(hr) == r).all(), (q, r))

    def test_town_weight_on_the_town_hex(self):
        # the old sample, round(display position), falls on a neighbour hex for some hexes (c7: (1628, 924), its east
        # neighbour water); take such a hex of the loaded canvas: its own tiles at full weight, everything else 0
        found = None
        for q in range(cw.PAD_LEFT + 10, cw.PAD_LEFT + cw.NATIVE_W - 10, 3):
            for r in range(cw.PAD_BOTTOM + 10, cw.PAD_BOTTOM + cw.NATIVE_H - 10, 37):
                ex, ey = (int(round(v / (d / 2))) for v, d in zip(cw.display_position(q, r), (cw.DX, cw.DZ)))
                ys, xs = ht.hex_tiles(q, r)
                if not ((xs == ex) & (ys == ey)).any():
                    found = (q, r, ex, ey); break
            if found:
                break
        if found is None:
            self.skipTest('no hex of this canvas has its display sample off its own tiles')
        q, r, ex, ey = found
        x0, y0 = 2 * q - 6, 2 * r - 6
        w = np.zeros((16, 16), np.float32)
        ys, xs = ht.hex_tiles(q, r); w[ys - y0, xs - x0] = 1.
        self.assertEqual(float(w[ey - y0, ex - x0]), 0.)            # the old sample: a neighbour, weight 0
        self.assertEqual(ht.town_weight(w, q, r, x0, y0), 1.)       # the town hex itself
        self.assertEqual(ht.town_weight(w, q + 40, r, x0, y0), 0.)  # outside the window

    def test_builds_sample_the_town_hex(self):
        for f in ('hehuang_patch.py', 'korea_patch.py', 'wuyuan_patch.py'):
            src = (ROOT / 'scripts' / f).read_text(encoding='utf-8')
            self.assertIn('hehuang_terrain.town_weight(', src, f)
            self.assertNotIn('[iy, ix] <= .5', src, f)

    def test_towns_flattened_registered(self):
        fn, reads, steps, kind, thr, tol, warn_only, _ = si.INVARIANTS['patch_towns_flattened']
        self.assertIs(fn, si.patch_towns_flattened)
        self.assertEqual((steps, kind, thr, warn_only), (('map',), 'exact', 0, False))

    def test_invariant_registered(self):
        fn, reads, steps, kind, thr, tol, warn_only, _ = si.INVARIANTS['patch_design']
        self.assertIs(fn, si.patch_design)
        self.assertEqual((steps, kind, thr, warn_only), (('map',), 'exact', 0, False))

    def test_lake_towns_from_design(self):
        src = (ROOT / 'scripts' / 'hehuang_patch.py').read_text(encoding='utf-8')
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertNotIn('LAKE_TOWNS', names)                             # no key list
        self.assertIn('LAKE_TOWN_REACH', names)
        self.assertIn("assert any(t.get('lake_town') for t in towns)", src)


if __name__ == '__main__':
    unittest.main()

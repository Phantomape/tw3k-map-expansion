"""Frontier mountain props: stateless per-site draws (frontier_mountain_props STATELESS DRAWS).

Needs TW3K_WORKSPACE (campaign_world reads the design at import); reads no other workspace file.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import frontier_mountain_props as fm

V = '3k_dlc07_main_map'


class StatelessDrawTest(unittest.TestCase):
    def test_pure_function_of_site(self):
        q, r = np.arange(1000) % 97, np.arange(1000) // 97 + 900
        a = fm.site_u(V, 'arid', q, r, fm.ACCEPT)
        self.assertTrue(np.array_equal(a, fm.site_u(V, 'arid', q.copy(), r.copy(), fm.ACCEPT)))
        # a site's draw does not depend on which other sites are drawn with it
        self.assertTrue(np.array_equal(a[500:], fm.site_u(V, 'arid', q[500:], r[500:], fm.ACCEPT)))
        self.assertEqual(float(fm.site_u(V, 'arid', int(q[7]), int(r[7]), fm.ACCEPT)), a[7])

    def test_keys_separate(self):
        q, r = np.arange(5000) % 211, np.arange(5000) // 211
        base = fm.site_u(V, 'arid', q, r, fm.ACCEPT)
        for other in (fm.site_u(V, 'cold', q, r, fm.ACCEPT), fm.site_u(V, 'arid', q, r, fm.PICK),
                      fm.site_u('other_variant', 'arid', q, r, fm.ACCEPT), fm.site_u(V, 'arid', r, q, fm.ACCEPT)):
            self.assertLess(abs(np.corrcoef(base, other)[0, 1]), .05)
            self.assertFalse(np.array_equal(base, other))

    def test_uniform(self):
        q, r = np.arange(200000) % 2048, np.arange(200000) // 2048
        u = fm.site_u(V, 'cold', q, r, fm.ACCEPT)
        self.assertTrue(((u >= 0) & (u < 1)).all())
        hist = np.histogram(u, 10, (0, 1))[0] / len(u)
        self.assertLess(np.abs(hist - .1).max(), .005)

    def test_order_independent_of_candidates(self):
        """The spacing order of two sites is the same whatever else is a candidate."""
        q, r = np.arange(300) % 30, np.arange(300) // 30
        key = fm.site_u(V, None, q, r, fm.ORDER)
        full = np.lexsort((r, q, key))
        keep = np.arange(300) % 3 != 0
        sub = np.lexsort((r[keep], q[keep], key[keep]))
        self.assertEqual([int(i) for i in np.arange(300)[keep][sub]], [int(i) for i in full if keep[i]])

    def test_no_shared_stream(self):
        src = (ROOT / 'scripts/frontier_mountain_props.py').read_text(encoding='utf-8')
        self.assertNotIn('default_rng(', src)
        self.assertNotIn('.shuffle(', src)

    def test_family_deviation(self):
        import biome_profiles as bp
        tol = bp.CONSTS['props']['MOUNTAIN_FAMILY_COUNT_TOL']
        self.assertEqual(tol, .10)
        d = fm.family_deviation({'cold': 157, 'arid': 195}, {'cold': 346, 'arid': 199}, tol)
        self.assertTrue(d['cold'][3])            # 157 vs 346: -55 %
        self.assertFalse(d['arid'][3])           # 195 vs 199: -2 %
        self.assertEqual(d['arid'][:3], (195, 199, round(4 / 199, 4)))
        d = fm.family_deviation({'arid': 220}, {'arid': 199, 'cold': 0}, tol)
        self.assertTrue(d['arid'][3])            # +10.6 %
        self.assertNotIn('cold', d)              # no target, no check
        self.assertEqual(fm.family_deviation({}, {'arid': 10}, tol)['arid'], (0, 10, 1.0, True))
        for f in ('cold', 'arid'):   # a family forced 20 % low warns, the other (on target) does not
            t = {'cold': 150, 'arid': 200}
            c = dict(t); c[f] = int(t[f] * .8)
            d = fm.family_deviation(c, t, tol)
            self.assertTrue(d[f][3], f)
            self.assertFalse(d['arid' if f == 'cold' else 'cold'][3], f)

    def test_model_triangles(self):
        import struct

        class Assets:
            def __init__(self, files):
                self.files = files

            def read(self, k):
                return self.files[k]
        g = bytearray(400)
        g[:4] = b'RMV2'
        struct.pack_into('<I', g, 8, 1)
        struct.pack_into('<3I', g, 140, 1, 1000, 6612)   # LOD0: groups, vertex bytes, index bytes
        a = Assets({'m/a.rigid_model_v2': bytes(g),
                    'm/b.wsmodel': b'<material><geometry>m/a.rigid_model_v2</geometry></material>'})
        self.assertEqual(fm.model_triangles(a, 'M/A.rigid_model_v2', {}), 1102)
        self.assertEqual(fm.model_triangles(a, 'm/b.wsmodel', {}), 1102)
        self.assertIsNone(fm.model_triangles(a, 'm/missing.rigid_model_v2', {}))

    def test_targets_from_rule_table(self):
        """v21 baseline: the rule-table targets equal the v21 placement (cold 157, arid 195) on its candidates."""
        import biome_profiles as bp
        t = bp.CONSTS['props']['MOUNTAIN_TARGET']
        self.assertEqual(round(t['rate']['cold'] * t['scale']['cold'] * 57327), 157)
        self.assertEqual(round(t['rate']['arid'] * t['scale']['arid'] * 66204), 195)
        src = (ROOT / 'scripts/frontier_mountain_props.py').read_text(encoding='utf-8')
        self.assertIn("CONSTS['props']['MOUNTAIN_TARGET']", src)


if __name__ == '__main__':
    unittest.main()

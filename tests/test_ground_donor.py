"""ground_donor.compress_choice: identity off the DONOR_COMPRESS biomes, [0, cap) on them, soft edge (tiny arrays)."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ground_donor as gd


def box(a, r):
    """Separable box mean of radius r with edge clamp (stands in for native_campaign_climate.box_blur)."""
    a = np.asarray(a, np.float32)
    for ax in (0, 1):
        p = np.pad(a, [(r, r) if i == ax else (0, 0) for i in range(2)], mode='edge')
        c = np.cumsum(np.insert(p, 0, 0, axis=ax), axis=ax, dtype=np.float64)
        n = a.shape[ax]
        a = ((np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, range(n), axis=ax)) / (2 * r + 1)).astype(np.float32)
    return a


class CompressChoice(unittest.TestCase):
    names = ['tundra', 'taiga', 'forest']

    def setUp(self):
        rng = np.random.default_rng(1)
        self.choice = rng.random((40, 60)).astype(np.float32)
        self.eco = np.zeros((40, 60), np.uint8); self.eco[:, 30:] = 1   # taiga on the east half

    def test_no_compress_biome_is_identity(self):
        out = gd.compress_choice(self.choice, np.full((40, 60), 2, np.uint8), self.names, 3, {'taiga': .3}, box)
        np.testing.assert_array_equal(out, self.choice)
        out = gd.compress_choice(self.choice, self.eco, self.names, 3, {'steppe': .3}, box)   # absent from names
        np.testing.assert_array_equal(out, self.choice)

    def test_full_share_scales_into_cap(self):
        out = gd.compress_choice(self.choice, self.eco, self.names, 3, {'taiga': .3}, box)
        core = out[:, 45:]
        np.testing.assert_allclose(core, self.choice[:, 45:] * np.float32(.3), rtol=1e-6)
        self.assertTrue((core < .3).all())
        np.testing.assert_array_equal(out[:, :15], self.choice[:, :15])
        self.assertEqual(out.dtype, np.float32)

    def test_edge_is_soft_and_monotone(self):
        ones = np.full((40, 60), .99, np.float32)
        out = gd.compress_choice(ones, self.eco, self.names, 3, {'taiga': .3}, box)[20]
        self.assertTrue((np.diff(out) <= 1e-7).all())
        self.assertGreater(len(np.unique(np.round(out[24:37], 4))), 5)   # a ramp across the edge, not a step



class Treeline(unittest.TestCase):
    names = ['tundra', 'taiga', 'forest']
    rule = {'from': ('tundra',), 'forest': ('taiga', 'conifer'), 'band': 6., 'climates': (3, 1)}

    def setUp(self):
        self.eco = np.zeros((20, 40), np.uint8); self.eco[:, 20:] = 1   # tundra west, taiga east

    def test_weight_ramps_to_the_line(self):
        p, zone = gd.treeline_weight(self.eco, self.names, self.rule)
        row = p[10]
        self.assertGreater(row[19], .9); self.assertTrue((row[:14] == 0).all())
        self.assertTrue((np.diff(row[:20]) >= 0).all())
        self.assertFalse(zone[:, 20:].any()); self.assertTrue(zone[:, 15:20].all())

    def test_absent_forest_is_none(self):
        self.assertIsNone(gd.treeline_weight(np.zeros((20, 40), np.uint8), self.names, self.rule))

    def test_pick_dithers_only_arid_cold_in_zone(self):
        picked = np.array([[3, 3, 1, 2, 3]]); p = np.array([[.9, .1, .1, .9, .9]], np.float32)
        u = np.full((1, 5), .5, np.float32); zone = np.array([[True, True, True, True, False]])
        np.testing.assert_array_equal(gd.treeline_pick(picked, p, u, zone, (3, 1)), [[1, 3, 3, 2, 3]])


if __name__ == '__main__':
    unittest.main()

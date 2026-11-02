import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import biome_profiles as bp  # noqa: E402


class SnowCoverage(unittest.TestCase):
    def test_switch_default_off(self):
        old = os.environ.pop('TW3K_SNOW_COVERAGE', None)
        try:
            self.assertFalse(bp.snow_coverage_on())
        finally:
            if old is not None:
                os.environ['TW3K_SNOW_COVERAGE'] = old

    def test_flat_dry_plateau_is_thin(self):
        e = np.full((40, 40), 4800., np.float32); lat = np.full_like(e, 33.); lon = np.full_like(e, 86.)
        el, er = bp.snow_reference_elev(e, 2.2)
        a = bp.winter_snow_coverage(e, lat, lon, 0., None, el, er)
        self.assertAlmostEqual(float(a.mean()), 1 - bp.CONSTS['snow_bake']['COVERAGE']['winter_dry'], places=3)

    def test_grows_upward_on_a_slope(self):
        # one slope: its local / regional means are the same for every point on it (the dryness of the region does not
        # change along it), so coverage may only grow with elevation
        e = np.linspace(2000., 6500., 200, dtype=np.float32)[None].repeat(3, 0)
        lat = np.full_like(e, 36.); lon = np.full_like(e, 75.)
        for ref in (2500., 3500., 4500.):
            el = er = np.full_like(e, ref)
            a = bp.winter_snow_coverage(e, lat, lon, 0., None, el, er)
            self.assertTrue((np.diff(a[1]) >= -1e-6).all(), ref)
        self.assertTrue((np.diff(a[1]) >= -1e-6).all())
        self.assertEqual(float(a[1, -1]), 1.)

    def test_dry_interior_raises_the_perennial_line_only(self):
        lat = np.array([34., 28., 38.3], np.float32); lon = np.array([86., 86.9, 75.1], np.float32)
        base = bp.perennial_snowline_m(lat, lon)
        er = np.array([5080., 3550., 3590.], np.float32)   # Qiangtang interior, Himalaya crest region, Muztagh Ata
        raised = bp.perennial_snowline_m(lat, lon, er)
        self.assertGreater(float(raised[0] - base[0]), 200.)
        self.assertLess(float(raised[0]), 6250.)
        self.assertTrue(np.allclose(raised[1:], base[1:]))


if __name__ == '__main__':
    unittest.main()

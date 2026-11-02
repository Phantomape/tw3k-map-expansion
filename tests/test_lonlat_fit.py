"""One lon/lat fit (campaign_world.lonlat_fit): plan_frontier_regions uses it, bit for bit what its own copy gave.
check_town_positions uses it too. Still pending, after the release (it changes bits of west-real-terrain, an input of
the patch caches): plan_western_regions FIT_Q/FIT_R (constant -2109.9/-85.5; cancels in project() but not in every
float bit). No new copy may appear."""
from pathlib import Path
import os
import re
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault('TW3K_WORKSPACE', str(SCRIPTS.parent / 'work/all-directions-30'))
PENDING = {'plan_western_regions.py'}   # after the release, with west-real-terrain and the patch caches


class LonLatFit(unittest.TestCase):
    def test_same_bits_as_the_replaced_copy(self):
        import campaign_world as cw
        lat, lon = np.meshgrid(np.arange(-10, 75, .37), np.arange(40, 160, .41), indexing='ij')
        old = (28.54 * lon - 0.85 * lat - 2110 + cw.PAD_LEFT - 658, -0.89 * lon + 25.72 * lat - 86 + cw.PAD_BOTTOM - 230)
        new = cw.lonlat_fit(lon, lat)
        self.assertTrue((old[0] == new[0]).all() and (old[1] == new[1]).all())

    def test_no_other_copy(self):
        hits = [p.name for p in sorted(SCRIPTS.glob('*.py')) if p.name != 'campaign_world.py'
                and re.search(r'28\.54\s*\*', p.read_text(encoding='utf-8', errors='replace'))]
        self.assertEqual(sorted(set(hits) - PENDING), [])


if __name__ == '__main__':
    unittest.main()

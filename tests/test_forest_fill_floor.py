import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import biome_profiles as bp  # noqa: E402
import frontier_forest as ff  # noqa: E402


class FillFloor(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get('TW3K_FOREST_FILL_FLOOR')

    def tearDown(self):
        if self._old is None:
            os.environ.pop('TW3K_FOREST_FILL_FLOOR', None)
        else:
            os.environ['TW3K_FOREST_FILL_FLOOR'] = self._old

    def test_off_no_floor(self):
        os.environ['TW3K_FOREST_FILL_FLOOR'] = '0'
        self.assertIsNone(bp.fill_floor(np.zeros((2, 2), np.uint8), ['taiga']))

    def test_floor_only_on_taiga_classes_far_out(self):
        os.environ['TW3K_FOREST_FILL_FLOOR'] = '1'
        bn = ['taiga', 'grass', 'conifer']
        eco = np.zeros((120, 120), np.uint8); eco[:, 60:] = 1; eco[:, 100:] = 2
        play = np.zeros((120, 120), bool); play[:10, :10] = True
        off = ff.fill_weight(play)
        on = ff.fill_weight(play, bp.fill_floor(eco, bn))
        f = bp.FOREST_FILL_FLOOR
        self.assertEqual(float(off[-1, -1]), 0.)
        self.assertAlmostEqual(float(on[-1, 30]), f, places=6)    # taiga far out
        self.assertEqual(float(on[-1, 80]), 0.)                    # grass keeps the fade
        self.assertAlmostEqual(float(on[-1, -1]), f, places=6)    # conifer
        self.assertTrue((on >= off).all())


if __name__ == '__main__':
    unittest.main()

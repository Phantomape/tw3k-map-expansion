import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import biome_profiles as bp  # noqa: E402


class NoOasisMasks(unittest.TestCase):
    def test_classes_and_arctic_grass(self):
        bn = ['desert', 'steppe', 'grass', 'alpine', 'tundra', 'snow', 'forest', 'wetland']
        eco = np.array([[0, 1, 2, 2, 4, 5, 6, 7]], np.uint8)
        lat = np.array([[40, 40, 40, 70, 70, 60, 50, 47]], np.float32)
        q, r = np.mgrid[:1, :8][::-1].astype(np.float32) * 40
        no_o, no_a = bp.no_oasis_masks(eco, bn, lat, q, r, (0., 0.))
        # desert, steppe, southern grass, forest and wetland keep their belts; arctic grass, tundra lose both; snow the oasis only
        self.assertEqual(no_o.tolist(), [[False, False, False, True, True, True, False, False]])
        self.assertEqual(no_a.tolist(), [[False, False, False, True, True, False, False, False]])

    def test_south_of_the_cold_line_keeps_oases(self):
        bn = ['tundra', 'snow']
        eco = np.zeros((1, 50), np.uint8); eco[0, 25:] = 1
        lat = np.full((1, 50), 32., np.float32)   # Ladakh latitude: far south of NORTH_COLD - fuzz
        q, r = np.mgrid[:1, :50][::-1].astype(np.float32) * 7
        no_o, no_a = bp.no_oasis_masks(eco, bn, lat, q, r, (0., 0.))
        self.assertFalse(no_o.any() or no_a.any())

    def test_switch_default_off(self):
        import os
        old = os.environ.pop('TW3K_TUNDRA_OASIS_OFF', None)
        try:
            self.assertFalse(bp.tundra_oasis_off_on())
        finally:
            if old is not None:
                os.environ['TW3K_TUNDRA_OASIS_OFF'] = old


if __name__ == '__main__':
    unittest.main()

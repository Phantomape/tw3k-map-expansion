"""tile_bake.seabed_replaced on hand-made 4x4 footprints. Release run 20261101-134607: vanilla generic_sea 4x4 records
on the rebuilt Korean coast (vanilla's coast is narrower: 8-15 of their 16 cells are map.hex land) were only partly
inside polish's cache replacement, so they stayed as vanilla and their land cells got no land floor (132 bare land
cells, stage_invariants.tile_land_coverage 1127 > 1051). Replayed on a copy of that stage: 1127 -> 929, Korea 0."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tile_bake as tb

N = 4


def fp(cells):
    m = np.zeros((N, N), bool)
    for y, x in cells:
        m[y, x] = True
    return m


class SeabedReplacedTest(unittest.TestCase):
    def test_wholly_replaced(self):
        everything = np.ones((N, N), bool)
        self.assertTrue(tb.seabed_replaced(fp([]), everything, fp([])))

    def test_partly_replaced_rest_land(self):
        # the Korean coast case: the unreplaced cells are land, so the record goes and they take land floors
        replaced = fp([(y, x) for y in range(N) for x in range(N) if x < 2])
        land = fp([(y, x) for y in range(N) for x in range(N) if x >= 2])
        self.assertTrue(tb.seabed_replaced(fp([]), replaced, land))

    def test_reaching_unreplaced_sea_stays(self):
        replaced = fp([(y, x) for y in range(N) for x in range(N) if x < 2])
        land = fp([(0, 2), (0, 3)])                     # cells (1..3, 2..3) are sea outside the replacement
        self.assertFalse(tb.seabed_replaced(fp([]), replaced, land))

    def test_unedited_all_land_stays(self):
        # vanilla's own seabed on land (Wuyuan, Liaodong) away from any edit: kept as vanilla lists it
        self.assertFalse(tb.seabed_replaced(fp([]), fp([]), np.ones((N, N), bool)))

    def test_edit_without_replacement_on_land(self):
        self.assertTrue(tb.seabed_replaced(fp([(0, 0)]), fp([]), np.ones((N, N), bool)))


if __name__ == '__main__':
    unittest.main()

"""terrain_regression.edge_counts: registered ('*_known') native-edge failures are not counted as failing."""
import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import terrain_regression as tr

BOX = (1280, 1310, 1070, 1080)


class EdgeCounts(unittest.TestCase):
    def test_known_only_is_not_failing(self):
        fail = [((1292, 1077), ['tone_cut_known']), ((1300, 1077), ['tone_cut_known'])]
        self.assertEqual(tr.edge_counts(fail, BOX), (0, 2))

    def test_known_plus_real_counts_as_failing(self):
        fail = [((1292, 1077), ['tone_cut_known', 'stripes']), ((1284, 1077), ['stripes']), ((1300, 1077), ['tone_cut'])]
        self.assertEqual(tr.edge_counts(fail, BOX), (3, 0))

    def test_outside_box_ignored(self):
        fail = [((1000, 1077), ['stripes']), ((1292, 1200), ['tone_cut_known'])]
        self.assertEqual(tr.edge_counts(fail, BOX), (0, 0))

    def test_empty_fails_is_failing(self):
        self.assertEqual(tr.edge_counts([((1292, 1077), [])], BOX), (1, 0))


if __name__ == '__main__':
    unittest.main()

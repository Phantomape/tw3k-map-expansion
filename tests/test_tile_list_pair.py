"""stage_invariants.tile_list_pair_diff on hand-made records ('<HIBhhHff' tuples as read_tile_list.parse gives them).
On the real lists (2026-10-31): vanilla 3k_dlc07_main_map 0 of 195,490; candidates v22 / v22b 0 of 1.35M / 1.34M."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stage_invariants as si

B = chr(92)
PATHS = [f'terrain{B}tiles{B}campaign{B}generic{B}2x2{B}', f'terrain{B}tiles{B}campaign{B}roads_tracks{B}straight{B}']


def rec(p, x, flags, lo=.2, hi=.3, climate=1):
    return (1, p, climate, x, 7, flags, lo, hi)


def pair(local, glob):
    return si.tile_list_pair_diff((PATHS, local), (PATHS, glob))


class TileListPair(unittest.TestCase):
    def test_vanilla_convention_passes(self):
        local = [rec(0, 0, 0x710), rec(1, 2, 0x720)]
        glob = [rec(1, 2, 0x720), rec(0, 0, 0x10)]          # order may differ; generic drops 0x700, roads keep it
        self.assertEqual(pair(local, glob)[0], 0)

    def test_height_or_climate_mismatch_counts_both_sides(self):
        local = [rec(0, 0, 0x710, hi=.3)]
        v, d = pair(local, [rec(0, 0, 0x10, hi=.31)])
        self.assertEqual((v, d['only_local'], d['only_global']), (2, 1, 1))
        self.assertEqual(pair(local, [rec(0, 0, 0x10, climate=2)])[0], 2)

    def test_flag_convention(self):
        v, d = pair([rec(0, 0, 0x10)], [rec(0, 0, 0x10)])
        self.assertEqual((v, d['local_without_700']), (1, 1))
        v, d = pair([rec(0, 0, 0x710)], [rec(0, 0, 0x710)])   # generic must drop 0x700 in the global list
        self.assertEqual((v, d['global_off_convention']), (1, 1))
        v, d = pair([rec(1, 0, 0x710)], [rec(1, 0, 0x10)])    # roads must keep it
        self.assertEqual((v, d['global_off_convention']), (1, 1))

    def test_missing_record(self):
        v, d = pair([rec(0, 0, 0x710), rec(0, 2, 0x710)], [rec(0, 0, 0x10)])
        self.assertEqual((v, d['only_local']), (1, 1))


if __name__ == '__main__':
    unittest.main()

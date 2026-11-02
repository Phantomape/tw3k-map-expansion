"""layout_diff / bake_map --diff (release L0, l0_roundtrip): a small vanilla map and a small canvas map built from it;
extract stores only the hexes that differ, build gives the canvas map back byte for byte, a different vanilla stops."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import layout_diff as ld  # noqa: E402
import map_hex  # noqa: E402


def small_map(h, w, seed, regions):
    rng = np.random.default_rng(seed)
    return {'major': 0, 'minor': 19, 'opaque': bytes(range(8)), 'game': 'three_kingdoms', 'campaign_map': 'test_map',
            'land_regions': regions, 'sea_regions': ['sea_a'], 'land_ground_types': ['plains'], 'sea_ground_types': ['sea'],
            'climates': ['temperate'], 'attritions': ['none'], 'areas_of_interest': [], 'palettes': [bytes(8), bytes(4)],
            'cells': rng.integers(0, 256, (h, w, 16), dtype=np.uint8)}


class LayoutDiffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.native = (slice(2, 6), slice(3, 8))                    # vanilla 4 x 5 at offset (r 2, q 3) on a 9 x 12 canvas
        self.van = small_map(4, 5, 1, ['v_a', 'v_b'])
        map_hex.write(self.van, self.tmp / 'vanilla.hex')
        canvas = small_map(9, 12, 2, ['v_a', 'v_b', 'new_c'])
        base = ld.base_cells(self.van['cells'], (9, 12), self.native)
        keep = np.zeros((9, 12), bool); keep[self.native] = True; keep[3, 4] = False; keep[4, 6] = False
        canvas['cells'] = np.where(keep[..., None], base, canvas['cells'])   # two vanilla hexes edited, the rest kept
        self.canvas = canvas
        map_hex.write(canvas, self.tmp / 'map.hex')

    def test_roundtrip_and_counts(self):
        man = ld.extract(self.tmp / 'map.hex', self.tmp / 'vanilla.hex', self.tmp / 'layout', self.native)
        self.assertEqual(man['counts']['stored_in_vanilla_rectangle'], 2)            # only the edited vanilla hexes
        self.assertEqual(man['counts']['stored_outside_rectangle'], 9 * 12 - 4 * 5)
        m, _ = ld.build(self.tmp / 'layout', self.tmp / 'vanilla.hex')
        map_hex.write(m, self.tmp / 'baked.hex')
        self.assertEqual((self.tmp / 'baked.hex').read_bytes(), (self.tmp / 'map.hex').read_bytes())

    def test_other_vanilla_stops(self):
        ld.extract(self.tmp / 'map.hex', self.tmp / 'vanilla.hex', self.tmp / 'layout', self.native)
        other = dict(self.van); other['cells'] = self.van['cells'].copy(); other['cells'][0, 0, 0] ^= 1
        map_hex.write(other, self.tmp / 'vanilla2.hex')
        with self.assertRaises(SystemExit):
            ld.build(self.tmp / 'layout', self.tmp / 'vanilla2.hex')

    def test_unchanged_bytes_not_stored(self):
        ld.extract(self.tmp / 'map.hex', self.tmp / 'vanilla.hex', self.tmp / 'layout', self.native)
        z = np.load(self.tmp / 'layout' / ld.NPZ)
        bits = ((z['mask'][:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(bool)
        self.assertTrue((z['bytes'][~bits] == 0).all())

    def test_vanilla_hexes_not_stored(self):
        ld.extract(self.tmp / 'map.hex', self.tmp / 'vanilla.hex', self.tmp / 'layout', self.native)
        z = np.load(self.tmp / 'layout' / ld.NPZ)
        stored = set(zip(z['q'].tolist(), z['r'].tolist()))
        for r in range(2, 6):
            for q in range(3, 8):
                self.assertEqual((q, r) in stored, (q, r) in {(4, 3), (6, 4)})


if __name__ == '__main__':
    unittest.main()

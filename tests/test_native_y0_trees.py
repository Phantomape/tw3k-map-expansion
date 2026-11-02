"""Native vanilla y = 0 tree rows: the height rule (tree_bake.native_row_heights) and the stage invariant
native_y0_trees_grounded. Synthetic inputs only (design and workspace in a temporary folder, as check_tree_bake)."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

_TMP = tempfile.TemporaryDirectory(prefix='tw3k-native-y0-tests-')
_T = Path(_TMP.name)
(_T / 'design.json').write_text(json.dumps({'grid': [892 + 80, 702 + 20], 'padding': [20, 10, 60, 10]}), encoding='utf-8')
os.environ['TW3K_DESIGN'] = str(_T / 'design.json')
os.environ['TW3K_WORKSPACE'] = str(_T)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np  # noqa: E402

import campaign_world as cw  # noqa: E402
import tree_bake as tb  # noqa: E402
import stage_invariants as si  # noqa: E402


def tree_list(groups):
    """A campaign tree list: 24-byte header (group count at 20), per group name + rows (seasons 0)."""
    out = bytearray(24); struct.pack_into('<I', out, 20, len(groups))
    for name, rows in groups:
        out += struct.pack('<H', len(name)) + name.encode() + struct.pack('<I', len(rows))
        for x, y, z in rows:
            out += struct.pack('<3f', x, y, z) + b'\0\0' + struct.pack('<I', 0)
    return bytes(out)


def lf_dds(raw, n=8):
    head = bytearray(128); head[:4] = b'DDS '; struct.pack_into('<2I', head, 12, n, n)
    return bytes(head) + np.full((n, n), raw, '<u2').tobytes()


class NativeRowHeights(unittest.TestCase):
    def test_y0_on_ground_others_by_delta(self):
        y = np.array([0., 1.5, -.3, 0.], np.float32).astype(np.float64)
        moved = np.array([.2, .3, .4, -.5]); ground = np.array([1.3, 9., 9., 2.])
        out = tb.native_row_heights(y, moved, ground)
        self.assertEqual(out[0], 1.3); self.assertEqual(out[3], 2.)
        # y != 0: the delta rule, bit for bit as the old y[i] + m_ (byte-identical rows)
        for i in (1, 2):
            self.assertEqual(struct.pack('<f', out[i]), struct.pack('<f', y[i] + moved[i]))

    def test_deterministic(self):
        a = tb.native_row_heights([0., 2.], [.1, .1], [3., 0.])
        self.assertTrue(np.array_equal(a, tb.native_row_heights([0., 2.], [.1, .1], [3., 0.])))


class NativeY0Invariant(unittest.TestCase):
    RAW = 30000

    def setUp(self):
        self.ws = Path(tempfile.mkdtemp(dir=_T))
        self.ground = cw.LEVEL + (self.RAW / 65535 - cw.NORM) * cw.GAIN
        SX, SZ = cw.NATIVE_SHIFT_WORLD
        self.inside = (10., 10.); self.outside = (-SX - 5., 10.)
        self.assertTrue(tb.native_many(np.float64(10.) + SX, np.float64(10.) + SZ))
        self.assertFalse(tb.native_many(np.float64(-SX - 5.) + SX, np.float64(10.) + SZ))
        self.vanilla = tree_list([('bamboo_1', [(10., 0., 10.), (12., 4., 10.)]), ('rock', [(-SX - 5., 0., 10.)])])

        def stage(y0):
            s = lambda x, z: (float(np.float32(np.float64(np.float32(x)) + SX)), float(np.float32(np.float64(np.float32(z)) + SZ)))
            (ax, az), (bx, bz), (cx, cz) = s(10., 10.), s(12., 10.), s(-SX - 5., 10.)
            # the y != 0 row and the canvas y = 0 row stand far off the ground: neither counts
            return tree_list([('bamboo_1', [(ax, y0, az), (bx, self.ground - 3, bz), (ax + 1, self.ground - 3, az)]),
                              ('rock', [(cx, self.ground - 3, cz)])])
        self.stage = stage

    def run_check(self, y0):
        (self.ws / si.TREES).parent.mkdir(parents=True, exist_ok=True); (self.ws / si.TREES).write_bytes(self.stage(y0))
        (self.ws / si.LF).parent.mkdir(parents=True, exist_ok=True); (self.ws / si.LF).write_bytes(lf_dds(self.RAW))
        ctx = si.Ctx(self.ws); ctx._c['vanilla_trees'] = self.vanilla
        return si.native_y0_trees_grounded(ctx)

    def test_buried_row_fails(self):
        r = self.run_check(self.ground - .505)   # the delta rule: 0 + (final - vanilla ground)
        self.assertEqual(r['value'], 1); self.assertEqual(r['detail']['stage_rows'], 1)
        self.assertAlmostEqual(r['detail']['off'][0]['tree_minus_lf'], -.505, places=2)

    def test_grounded_row_passes(self):
        self.assertEqual(self.run_check(self.ground)['value'], 0)
        self.assertEqual(self.run_check(self.ground + .2)['value'], 0)

    def test_registered_after_graph_hard(self):
        fn, reads, steps, kind, thr, tol, warn_only, _ = si.INVARIANTS['native_y0_trees_grounded']
        self.assertIs(fn, si.native_y0_trees_grounded)
        self.assertEqual((steps, kind, thr, warn_only), (('graph',), 'exact', 0, False))
        self.assertEqual(set(reads), {si.TREES, si.LF})


if __name__ == '__main__':
    unittest.main()

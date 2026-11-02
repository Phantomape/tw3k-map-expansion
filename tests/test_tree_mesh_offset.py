"""tree_mesh_offset helpers: the tree-list reader and the mesh-raster sampler (stage_invariants)."""
import struct, sys, tempfile, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stage_invariants as si


def tree_list(groups):
    """A campaign_tree_list blob: 20 header bytes, group count, then per group name, count and rows
    (x, y, z, 2 bytes, extra count n, n * 4 bytes)."""
    b = bytearray(20) + struct.pack('<I', len(groups))
    for name, rows in groups:
        b += struct.pack('<H', len(name)) + name.encode() + struct.pack('<I', len(rows))
        for (x, y, z), extra in rows:
            b += struct.pack('<3f', x, y, z) + b'\0\0' + struct.pack('<I', extra) + bytes(4 * extra)
    return bytes(b)


class TreeXyz(unittest.TestCase):
    def ctx(self, blob):
        d = Path(tempfile.mkdtemp())
        p = d / si.TREES
        p.parent.mkdir(parents=True)
        p.write_bytes(blob)
        return si.Ctx(d)

    def test_rows_with_extras(self):
        blob = tree_list([('pine_a', [((1, 2, 3), 0), ((4, 5, 6), 2)]), ('general_rock_x', [((7, 8, 9), 1)])])
        xyz = si._tree_xyz(self.ctx(blob))
        np.testing.assert_allclose(xyz, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_trailing_bytes_fail(self):
        with self.assertRaises(ValueError):
            si._tree_xyz(self.ctx(tree_list([('pine_a', [((1, 2, 3), 0)])]) + b'x'))

    def test_no_tree_list(self):
        self.assertIsNone(si._tree_xyz(si.Ctx(Path(tempfile.mkdtemp()))))

    def test_truncated_list_is_reported_not_raised(self):
        """A truncated list (struct.error) or a layout change (ValueError) must not stop the mesh gates in _lf_mesh."""
        blob = tree_list([('pine_a', [((1, 2, 3), 0), ((4, 5, 6), 2)])])
        for bad in (blob[:-7], blob + b'xx'):
            xyz, reason = si._tree_xyz_safe(self.ctx(bad))
            self.assertIsNone(xyz)
            self.assertTrue(reason.startswith(('error:', 'ValueError:')), reason)

    def test_safe_passes_good_list(self):
        xyz, reason = si._tree_xyz_safe(self.ctx(tree_list([('pine_a', [((1, 2, 3), 0)])])))
        self.assertIsNone(reason)
        self.assertEqual(xyz.shape, (1, 3))


class Bilinear(unittest.TestCase):
    def test_plane_exact(self):
        a = np.add.outer(np.arange(4) * 2., np.arange(5) * 1.)   # h = 2 y + x over cell indices
        fx, fy = np.array([.5, 2.25, 3.9]), np.array([.5, 1.75, 2.9])
        np.testing.assert_allclose(si._bilinear(a, fx, fy), 2 * fy + fx)

    def test_hole_neighbour_uses_nearest(self):
        a = np.array([[1., np.nan], [3., 4.]])
        self.assertEqual(si._bilinear(a, np.array([.2]), np.array([.9]))[0], 3.)

    def test_hole_under_tree_is_nan(self):
        a = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self.assertTrue(np.isnan(si._bilinear(a, np.array([.5]), np.array([.5]))[0]))


class Registered(unittest.TestCase):
    def test_graph_and_mesh_final_warn_metric(self):
        fn, reads, steps, kind, thr, tol, warn_only, _ = si.INVARIANTS['tree_mesh_offset']
        self.assertEqual((steps, kind, thr, tol, warn_only), (('graph', 'mesh_final'), 'metric', .01, 0, True))
        self.assertIn(si.TREES, reads)


if __name__ == '__main__':
    unittest.main()

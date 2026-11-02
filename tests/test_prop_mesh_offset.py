"""prop_mesh_offset / prop_scene_offset counting (stage_invariants) and the shared leaf reader (prop_instances)."""
import struct, sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prop_instances as pi
import stage_invariants as si

MT, RK = 'rigidmodels/campaign/mountains/arid/arid_mountain_01.rigid_model_v2', 'rigidmodels/campaign/rocks/general/general_rock_large_1.wsmodel'
ST, VG = 'rigidmodels/campaign/settlements/fence_1_level1.wsmodel', 'rigidmodels/campaign/vegetation/arid/arid_tree_1.wsmodel'


def leaf(models, recs):
    """A leaf asset: u16 2, u32 model count, the model table (u16 length + name), u32 record count, then records of
    131 + mode length bytes (u16 20, u32 model index, 12 floats at +24, u16 mode length at +105, mode at +107)."""
    b = bytearray(b'HEAD') + struct.pack('<HI', 2, len(models))
    for m in models:
        b += struct.pack('<H', len(m)) + m.encode()
    b += struct.pack('<I', len(recs))
    for idx, mat, mode in recs:
        r = bytearray(131 + len(mode))
        struct.pack_into('<HI', r, 0, 20, idx)
        struct.pack_into('<12f', r, 24, *mat)
        struct.pack_into('<H', r, 105, len(mode)); r[107:107 + len(mode)] = mode.encode()
        b += r
    return bytes(b)


def mat(x, y, z, sy=1.):
    return (1, 0, 0, 0, sy, 0, 0, 0, 1, x, y, z)


class LeafReader(unittest.TestCase):
    def test_records_with_mode(self):
        b = leaf([MT, ST], [(0, mat(1, 2, 3), 'BHM_IGNORE'), (1, mat(4, 5, 6, 2.), 'BHM_CUSTOM_VERTEX_OFFSETTING')])
        r = pi.leaf_records(b)
        self.assertEqual([(m, mode) for m, _, mode in r], [(MT, 'BHM_IGNORE'), (ST, 'BHM_CUSTOM_VERTEX_OFFSETTING')])
        self.assertEqual(r[1][1][9:], (4., 5., 6.)); self.assertEqual(r[1][1][4], 2.)

    def test_not_a_leaf(self):
        with self.assertRaises(AssertionError):
            pi.leaf_records(b'no model table here')

    def test_scene_key(self):
        self.assertEqual(pi.scene_of('terrain/campaigns/v/bmd_objects.tw3k_hist_gumo_farm.2412.31.bin'), ('tw3k_hist_gumo_farm', 2412))
        self.assertEqual(pi.scene_of('terrain/campaigns/v/bmd_objects.bin'), ('?', -1))
        self.assertEqual(pi.family(RK), 'rocks'); self.assertEqual(pi.family('x'), '?')

    def test_scene_piece_mask(self):
        m = pi.scene_piece_mask(['settlements', 'rocks', 'vegetation', 'settlements', 'settlements'],
                                ['BHM_IGNORE'] * 4 + ['BHM_CUSTOM_VERTEX_OFFSETTING'], [1, 1, 1, -200, 1], [2412, 2412, 2412, 2412, 2412])
        self.assertEqual(list(m), [True, True, False, False, False])
        self.assertFalse(pi.scene_piece_mask(['settlements'], ['BHM_IGNORE'], [1], [9999])[0])


class PropOffsetStats(unittest.TestCase):
    def test_counts(self):
        fam = ['mountains', 'rocks', 'rocks', 'mountains', 'mountains', 'settlements', 'rocks', 'rocks']
        mode = ['BHM_IGNORE'] * 3 + ['BHM_CUSTOM_VERTEX_OFFSETTING'] + ['BHM_IGNORE'] * 4
        y = [0, 0, 0, 0, -200, 0, 0, 0]
        base = [1., .2, -2., 5., 5., 5., 1., .5]      # floats, within, buried, offset mode, parked, family, no ground, other
        ground = [0, 0, 0, 0, 0, 0, np.nan, 0]
        playable = [1, 1, 1, 1, 1, 1, 1, 0]
        native = [0, 1, 0, 0, 0, 0, 0, 0]
        value, det, d = si.prop_offset_stats(fam, mode, y, base, ground, playable, native)
        self.assertEqual(det['playable']['props'], 3)              # only the first three count
        self.assertEqual(det['playable']['floating'], 1)
        self.assertAlmostEqual(value, round(1 / 3, 4))
        self.assertEqual(det['playable']['buried_below'], 1)       # info only
        self.assertEqual(det['other'], {'props': 1, 'floating': 1, 'share': 1.0, 'buried_below': 0, 'float_p95': .5})
        self.assertEqual(det['no_ground_under'], 1)
        self.assertEqual(det['native_interior']['props'], 1)
        self.assertEqual(det['canvas_playable']['props'], 2)
        self.assertTrue(np.isnan(d[3]) and np.isnan(d[4]) and np.isnan(d[6]))

    def test_threshold_is_strict(self):
        value, det, _ = si.prop_offset_stats(['rocks'], ['BHM_IGNORE'], [0], [.25], [0], [1], [0])
        self.assertEqual(det['playable']['floating'], 0)


class SceneOffsets(unittest.TestCase):
    def scene(self, region, num, offsets, fam='settlements', native=False):
        n = len(offsets)
        return dict(fam=[fam] * n, mode=['BHM_IGNORE'] * n, y=list(offsets), ground=[0.] * n, region=[region] * n,
                    prefab=[num] * n, native=[native] * n)

    def run_(self, *scenes):
        a = {k: sum((s[k] for s in scenes), []) for k in scenes[0]}
        return si.scene_offsets(a['fam'], a['mode'], a['y'], a['ground'], a['region'], a['prefab'], a['native'])

    def test_median_and_one_entry_per_scene(self):
        n, det = self.run_(self.scene('a', 2412, [3, 3, 3, -9]),      # median 3: off, despite one buried piece
                           self.scene('a', 2476, [.5, .4, .6]),       # on its ground
                           self.scene('b', 2412, [-1.5, -1.4, -1.6]),  # buried, another region: another scene
                           self.scene('c', 5825, [7, 7, 7]))          # not a resource template: not counted
        self.assertEqual(n, 2)
        self.assertEqual(det['scenes'], 3)
        self.assertEqual(set(det['off']), {'a.2412', 'b.2412'})

    def test_spread_info(self):
        n, det = self.run_(self.scene('a', 1913, list(np.linspace(-3, 3, 21))))   # median 0, a slope
        self.assertEqual(n, 0)
        self.assertEqual(det['spread_over'], 1)
        self.assertGreater(det['spread']['a.1913'], 1)

    def test_vegetation_and_parked_left_out(self):
        s = self.scene('a', 2412, [0, 0, 0])
        v = self.scene('a', 2412, [5, 5, 5, 5, 5], fam='vegetation')
        p = self.scene('a', 2412, [-200, -200, -200, -200])
        n, det = self.run_(s, v, p)
        self.assertEqual(n, 0)

    def test_reconciliation_count(self):
        n, det = self.run_(self.scene('a', 9001, [2.] * 10), self.scene('b', 9002, [2.] * 9), self.scene('c', 9003, [2.] * 10, native=True))
        self.assertEqual(n, 0)   # no resource template
        self.assertEqual(det['canvas_settlement_groups'], {'min_pieces': 10, 'scenes': 1, 'off': 1})


class Registered(unittest.TestCase):
    def test_rows(self):
        for name, step, thr, wo in (('prop_mesh_offset', 'mesh_final', .059, True), ('prop_scene_offset', 'mesh_final', 2, False),
                                    ('prop_lf_offset', 'resource_scenes', .094, True), ('prop_scene_lf_offset', 'resource_scenes', 1, False)):
            fn, reads, steps, k, t, tl, w, _ = si.INVARIANTS[name]
            self.assertEqual((steps, k, t, tl, w), ((step,), 'metric', thr, 0, wo))
            self.assertIn(si.PROPS, reads)
            if step == 'resource_scenes':
                self.assertNotIn(si.MESHES, reads)   # the graph's gap-fill rebuilds the meshes meanwhile

    def test_step_order(self):
        self.assertLess(si.STEPS.index('sync_final'), si.STEPS.index('resource_scenes'))
        self.assertLess(si.STEPS.index('resource_scenes'), si.STEPS.index('graph'))
        self.assertIn('resource_scenes', si.STEP_SCRIPTS)


if __name__ == '__main__':
    unittest.main()

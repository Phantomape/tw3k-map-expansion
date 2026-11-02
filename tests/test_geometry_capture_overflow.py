"""terrain_checkpoint.overflow_rels / files('geometry') on a hand-made workspace. Release rehearsal 2026-11-01 (run
20261101-153856): a run that stopped after fill_polished_mesh_gaps left 18 overflow land meshes (land_mesh_184..201,
both files) in the stage; the next geometry capture took them into terrain-before-polish and base_land_mesh_set failed
after polish (36)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campaign_world as cw
import terrain_checkpoint as tc

V = cw.TERRAIN_VARIANTS[0]


class OverflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        A = Path(self.tmp.name)
        self.saved = tc.A, tc.S
        tc.A, tc.S = A, A / 'work/release-input'
        gm = tc.S / f'terrain/campaigns/{V}/global_meshes'
        gm.mkdir(parents=True)
        for n in (0, 183, 184, 185):
            for ext in ('rigid_model_v2', 'compressed_map'):
                (gm / f'land_mesh_{n}.{ext}').write_bytes(b'x')
        bm = A / f'terrain-before-polish/terrain/campaigns/{V}/global_meshes'
        bm.mkdir(parents=True)
        (bm / 'land_mesh_190.rigid_model_v2').write_bytes(b'x')   # an earlier capture's leftover
        (A / 'analysis').mkdir()
        (A / f'analysis/coarse-mesh-overflow-{V}.json').write_text(json.dumps({'first': 184, 'files': {}}), encoding='utf-8')

    def tearDown(self):
        tc.A, tc.S = self.saved
        self.tmp.cleanup()

    def test_overflow_rels(self):
        got = {r.rsplit('/', 1)[1] for r in tc.overflow_rels()}
        self.assertEqual(got, {'land_mesh_184.rigid_model_v2', 'land_mesh_184.compressed_map', 'land_mesh_185.rigid_model_v2',
                               'land_mesh_185.compressed_map', 'land_mesh_190.rigid_model_v2'})

    def test_geometry_files_skip_overflow(self):
        names = {p.name for p in tc.files('geometry')}
        self.assertIn('land_mesh_183.rigid_model_v2', names)
        self.assertNotIn('land_mesh_184.rigid_model_v2', names)
        self.assertNotIn('land_mesh_185.compressed_map', names)

    def test_no_manifest_same(self):
        # the base set comes from stage_invariants.BASE_LAND_MESHES alone: a copied analysis without the manifest
        before = tc.overflow_rels()
        (tc.A / f'analysis/coarse-mesh-overflow-{V}.json').unlink()
        self.assertEqual(tc.overflow_rels(), before)
        self.assertNotIn('land_mesh_184.rigid_model_v2', {p.name for p in tc.files('geometry')})


if __name__ == '__main__':
    unittest.main()

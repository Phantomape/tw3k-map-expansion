"""ensure_patch_caches: the patch caches' one writer, bake_map step 10 (coordinator 11-01). Rebuild missing / stale
caches from the map the bake is building, keep current ones, stop when a rebuild still does not match; nothing reads an
earlier final map.hex on that path. Synthetic: the builder and the vanilla / design reads are mocked."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_ENV = {}
_T = _TMP = None
np = cw = epc = ht = map_hex = None


def setUpModule():
    """Environment set here and restored in tearDownModule, not at import: the full suite shares one process."""
    global _T, _TMP, np, cw, epc, ht, map_hex
    _TMP = tempfile.TemporaryDirectory(prefix='tw3k-ensure-patch-tests-')
    _T = Path(_TMP.name)
    (_T / 'design.json').write_text(json.dumps({'grid': [892 + 80, 702 + 20], 'padding': [20, 10, 60, 10]}), encoding='utf-8')
    for k, v in (('TW3K_DESIGN', str(_T / 'design.json')), ('TW3K_WORKSPACE', str(_T))):
        _ENV[k] = os.environ.get(k); os.environ[k] = v
    if str(ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT / 'scripts'))
    import numpy as np
    import campaign_world as cw
    import ensure_patch_caches as epc
    import hehuang_terrain as ht
    import map_hex


def tearDownModule():
    for k, v in _ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _TMP.cleanup()


class Ensure(unittest.TestCase):
    def setUp(self):
        self.dp = _T / 'plan.json'; self.dp.write_text(json.dumps({'regions': []}), encoding='utf-8')
        self.state = {}      # name -> reason or None
        self.built = []

    def run_(self, after_build=None, env=None, cells='bake'):
        def build(n):
            self.built.append(n)
            self.state[n] = after_build
        seen = []
        with mock.patch.object(cw, 'DESIGN_PATH', self.dp), \
                mock.patch.object(epc, 'stale', lambda n, p, plan, mc=None, mn=None: (seen.append(mc), self.state.get(n))[1]), \
                mock.patch.dict(os.environ, env or {}):
            out = epc.ensure(('hehuang', 'korea'), build, map_cells=cells, map_names=['r'])
        self.assertTrue(all(c == cells for c in seen))   # every check against the map passed in
        return out

    def test_current_kept(self):
        self.assertEqual(self.run_(), {'hehuang': 'current', 'korea': 'current'})
        self.assertEqual(self.built, [])

    def test_old_layout_cache_rebuilt(self):
        self.state = {'hehuang': 'design_sha a != b', 'korea': 'missing'}
        out = self.run_()
        self.assertEqual(self.built, ['hehuang', 'korea'])
        self.assertEqual(out['hehuang'], 'rebuilt (design_sha a != b)')

    def test_rebuild_that_still_mismatches_stops(self):
        self.state = {'korea': 'cache predates the input / rule keys'}
        with self.assertRaises(RuntimeError):
            self.run_(after_build='map_window x != y')

    def test_switched_off_skipped(self):
        self.state = {'hehuang': 'missing'}
        self.assertEqual(self.run_(env={'TW3K_HEHUANG': '0'})['hehuang'], 'off')
        self.assertEqual(self.built, [])

    def test_empty_workspace_reads_no_final_map(self):
        # a fresh workspace (no work/final-map/map.hex, no caches): step 10 builds from the bake's cells and step 11
        # loads against them; any read of the final map.hex fails the test
        ws = Path(tempfile.mkdtemp(dir=_T))
        real_read = map_hex.read

        def read(p):
            if 'final-map' in str(p):
                raise AssertionError(f'final map.hex read on the bake path: {p}')
            return real_read(p)
        cells = np.zeros((cw.H, cw.W, 16), np.uint8); names = ['r']
        built = {}

        def build(n):   # a cache keyed on the cells passed in, as <name>_patch.build(map_cells=...) writes it
            p = ht.patch_path(n); p.parent.mkdir(parents=True, exist_ok=True)
            built[n] = p
            np.savez(p, hex_box=np.array([0, 4, 0, 4]), towns='[]', inputs_sha=json.dumps(ht.input_sha([0, 4, 0, 4], None, cells, names)),
                     build_rules_sha='r', design_sha=ht.design_sha([0, 4, 0, 4], {'regions': []}))
        (ws / 'analysis').mkdir()
        np.savez(ws / 'analysis/west-real-terrain.npz', box=np.array([0, 1, 0, 1]))
        rules = mock.Mock(); rules.rules_sha.return_value = 'r'
        with mock.patch.object(cw, 'WORKSPACE', ws), mock.patch.object(cw, 'DESIGN_PATH', self.dp), \
                mock.patch.object(map_hex, 'read', read), mock.patch.object(ht, '_builder', lambda n: rules), \
                mock.patch.dict(os.environ, {'TW3K_KOREA': '0'}):
            self.assertFalse((ws / 'work/final-map/map.hex').exists())
            out = epc.ensure(('hehuang', 'korea'), build, map_cells=cells, map_names=names)
            self.assertEqual(out, {'hehuang': 'rebuilt (missing)', 'korea': 'off'})
            self.assertIsNone(ht.source_mismatch('hehuang', np.load(built['hehuang']), None, cells, names))   # step 11's check

    def test_bake_step10_is_the_writer(self):
        bake = (ROOT / 'scripts' / 'bake_map.py').read_text(encoding='utf-8')
        self.assertIn("ensure_patch_caches.ensure(map_cells=cells, map_names=m['land_regions'] + m['sea_regions'])", bake)
        self.assertLess(bake.index('ensure_patch_caches.ensure('), bake.index('apply_climate.compute(m)'))
        ps = (ROOT / 'scripts' / 'build_polished_campaign_terrain.ps1').read_text(encoding='utf-8')
        self.assertNotIn('ensure_patch_caches', ps)            # the pipeline only checks (patch_design)
        clim = (ROOT / 'scripts' / 'apply_climate.py').read_text(encoding='utf-8')
        self.assertIn('hehuang_terrain.load(map_cells=cells, map_names=rnames)', clim)


if __name__ == '__main__':
    unittest.main()

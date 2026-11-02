"""Derived geodata caches carry a key of what they were made from (single-source audit item 8): the eco-grid cache
(frontier_terrain) and the strategic ink cache stop the run without a matching key; a stamp records the key only
when a fresh computation equals the cache, and never rewrites the cache. Temporary folders, stubbed inputs."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault('TW3K_WORKSPACE', str(SCRIPTS.parent / 'work/all-directions-30'))
sys.modules.setdefault('plan_frontier_regions', types.ModuleType('plan_frontier_regions'))   # reads the vanilla map
import frontier_terrain as ft  # noqa: E402
import strategic_ink as si  # noqa: E402


class EcoGrid(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-ecokey-tests-')
        self.geo = Path(self._tmp.name)
        (self.geo / 'Ecoregions2017.zip').write_bytes(b'shapes v1')
        self.arrays = (np.arange(6, dtype=np.uint8).reshape(2, 3), np.arange(6, dtype=np.int32).reshape(2, 3))
        self.p = [mock.patch.object(ft, 'GEODATA', self.geo),
                  mock.patch.object(ft, '_eco_rasterise', lambda names: tuple(a.copy() for a in self.arrays))]
        for x in self.p:
            x.start()
        ft.eco_grid.cache_clear()

    def tearDown(self):
        for x in self.p:
            x.stop()
        ft.eco_grid.cache_clear()
        self._tmp.cleanup()

    def test_build_writes_key_then_loads(self):
        b, names, e = ft.eco_grid()
        self.assertTrue((self.geo / 'eco-grid-v2.npz.key.json').exists())
        ft.eco_grid.cache_clear()
        b2, _, e2 = ft.eco_grid()
        self.assertTrue(np.array_equal(b, b2) and np.array_equal(e, e2))

    def test_missing_or_other_key_stops(self):
        ft.eco_grid()
        (self.geo / 'eco-grid-v2.npz.key.json').unlink()
        ft.eco_grid.cache_clear()
        with self.assertRaisesRegex(SystemExit, 'no key'):
            ft.eco_grid()
        (self.geo / 'eco-grid-v2.npz.key.json').write_text(json.dumps({'key': 'other'}), encoding='utf-8')
        ft.eco_grid.cache_clear()
        with self.assertRaisesRegex(SystemExit, 'other code or source data'):
            ft.eco_grid()

    def test_source_change_stops(self):
        ft.eco_grid()
        (self.geo / 'Ecoregions2017.zip').write_bytes(b'shapes v2')
        ft.eco_grid.cache_clear()
        with self.assertRaises(SystemExit):
            ft.eco_grid()

    def test_stamp_only_when_equal_and_cache_untouched(self):
        ft.eco_grid()
        cache = self.geo / 'eco-grid-v2.npz'
        before = cache.read_bytes()
        (self.geo / 'eco-grid-v2.npz.key.json').unlink()
        ft.stamp_eco_grid()
        self.assertEqual(cache.read_bytes(), before)
        self.assertTrue((self.geo / 'eco-grid-v2.npz.key.json').exists())
        (self.geo / 'eco-grid-v2.npz.key.json').unlink()
        self.arrays = (self.arrays[0] + 1, self.arrays[1])
        with self.assertRaisesRegex(SystemExit, 'not stamped'):
            ft.stamp_eco_grid()
        self.assertFalse((self.geo / 'eco-grid-v2.npz.key.json').exists())


class StrategicInk(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-inkkey-tests-')
        self.cache = Path(self._tmp.name) / 'strategic-ink-test.npz'
        self.layers = (np.zeros((2, 2), np.uint8), ['grass'], np.zeros((2, 2), bool), np.ones((2, 2), np.float16))
        self.p = [mock.patch.object(si, 'cache_path', lambda: self.cache),
                  mock.patch.object(si, 'source_key', lambda: 'K1'),
                  mock.patch.object(si, 'compute', lambda: self.layers)]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        self._tmp.cleanup()

    def write(self, key=None):
        b, n, s, e = self.layers
        extra = {} if key is None else {'source_key': np.array(key)}
        np.savez_compressed(self.cache, biome=b, names=np.array(n), salt=s, elev=e,
                            canvas=np.array([si.cw.W, si.cw.H, si.cw.PAD_LEFT, si.cw.PAD_BOTTOM]), **extra)

    def test_keyed_cache_loads(self):
        self.write('K1')
        self.assertEqual(si.load()[1], ['grass'])

    def test_no_key_stops(self):
        self.write()
        with self.assertRaisesRegex(SystemExit, 'has no source key'):
            si.load()

    def test_stamp_then_loads_cache_untouched(self):
        self.write()
        before = self.cache.read_bytes()
        si.stamp()
        self.assertEqual(self.cache.read_bytes(), before)
        self.assertEqual(si.load()[1], ['grass'])

    def test_stamp_refuses_a_different_cache(self):
        self.write()
        self.layers = (self.layers[0] + 1,) + self.layers[1:]
        with self.assertRaisesRegex(SystemExit, 'not stamped'):
            si.stamp()
        self.assertFalse(si.key_path().exists())

    def test_other_key_stops(self):
        self.write('K0')
        with self.assertRaisesRegex(SystemExit, 'other code or geodata'):
            si.load()


if __name__ == '__main__':
    unittest.main()

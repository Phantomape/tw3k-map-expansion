"""bake_map --diff (release path) makes the patch caches current for the map it writes: bake_diff calls
ensure_patch_caches.ensure with that map's cells and region names after the l0_roundtrip check, and reports it
(integrator 11-01: no step 10 runs on this path, so the map checkpoint's patch_design would stop otherwise). Synthetic:
layout_diff.build, the map write and ensure are mocked. A real c5q bake --diff in a scratch workspace without caches gave
patch_design 0 (P6 notes)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
# The environment is set in setUpModule and restored in tearDownModule (it leaked into later modules of a full
# discover run: test_all_directions_setup, test_region_traits, test_sdk_empty_tables), as the P6 tests do.
_ENV, _TMP, _T = {}, None, None
np = bake_map = cw = ensure_patch_caches = layout_diff = ml = None


def setUpModule():
    global _TMP, _T, np, bake_map, cw, ensure_patch_caches, layout_diff, ml
    _TMP = tempfile.TemporaryDirectory(prefix='tw3k-bake-diff-tests-')
    _T = Path(_TMP.name)
    (_T / 'design.json').write_text(json.dumps({'grid': [892 + 80, 702 + 20], 'padding': [20, 10, 60, 10]}), encoding='utf-8')
    for k, v in (('TW3K_DESIGN', str(_T / 'design.json')), ('TW3K_WORKSPACE', str(_T))):
        _ENV[k] = os.environ.get(k); os.environ[k] = v
    if str(ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT / 'scripts'))
    import numpy as np
    import bake_map
    import campaign_world as cw
    import ensure_patch_caches
    import layout_diff
    import map_layers as ml


def tearDownModule():
    for k, v in _ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _TMP.cleanup()


class BakeDiffPatches(unittest.TestCase):
    def run_(self, recorded_sha):
        cells = np.zeros((4, 4, 16), np.uint8)
        m = {'cells': cells, 'land_regions': ['a'], 'sea_regions': ['s']}
        man = {'map_sha256': 'x' * 64, 'vanilla_sha256': 'v', 'counts': {}}
        out = _T / 'o' / 'map.hex'; out.parent.mkdir(exist_ok=True)
        calls, order = [], []
        mh = types.SimpleNamespace(write=lambda mm, p: (order.append('write'), Path(p).write_bytes(b'map')))

        def ensure(**k):
            order.append('ensure'); calls.append(k); return {'hehuang': 'current', 'korea': 'rebuilt (missing)'}
        args = types.SimpleNamespace(layout=_T / 'layout')
        with mock.patch.object(layout_diff, 'build', lambda layout, van: (m, man)), \
                mock.patch.object(ml, 'sha256', lambda p: recorded_sha), \
                mock.patch.object(ensure_patch_caches, 'ensure', ensure):
            bake_map.bake_diff(args, cw, mh, ml, out)
        return m, calls, order, out

    def test_ensure_after_the_write_with_the_maps_cells(self):
        m, calls, order, out = self.run_('x' * 64)
        self.assertEqual(order, ['write', 'ensure'])
        self.assertIs(calls[0]['map_cells'], m['cells'])
        self.assertEqual(calls[0]['map_names'], ['a', 's'])
        rep = json.loads(out.with_name(out.name + '.bake-report.json').read_text(encoding='utf-8'))
        self.assertEqual(rep['patch_caches'], {'hehuang': 'current', 'korea': 'rebuilt (missing)'})

    def test_no_ensure_when_the_roundtrip_fails(self):
        with self.assertRaises(SystemExit):
            self.run_('y' * 64)


if __name__ == '__main__':
    unittest.main()

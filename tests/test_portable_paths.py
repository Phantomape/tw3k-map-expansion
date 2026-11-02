"""Tracked reports record paths portably (no drive letter, no user folder: coordinator 2026-11-01).
tw3k_paths.portable gives a path relative to the main checkout, 'scratch:<rel>' under the scratch folder, else the
file name; audit_ground_patchiness writes its source through it; the tracked data files the release takes hold no
machine path, so regenerating them cannot bring one back unnoticed."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('TW3K_WORKSPACE', str(ROOT / 'work/all-directions-30'))
import tw3k_paths as T  # noqa: E402

MACHINE = re.compile(r'(?<![A-Za-z0-9\\])[A-Za-z]:[\\/]{1,2}(?=[A-Za-z0-9_.$%{(-])')
TRACKED = ['analysis/art-review-cameras.json', 'analysis/ground-patchiness-baseline.json', 'data/lake-level-spec.json',
           'data/oss-rebuild/params.json', 'analysis/ground-fragmentation-baseline.json']


class Portable(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-portable-')
        t = Path(self._tmp.name)
        self.main, self.scratch = t / 'TW3K', t / 'TW3K-scratch'
        self.p = mock.patch.object(T, 'resolve', lambda key, *a, **k: {'main_root': self.main, 'scratch': self.scratch}[key])
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self._tmp.cleanup()

    def test_forms(self):
        self.assertEqual(T.portable(self.main / 'output/x.pack'), 'output/x.pack')
        self.assertEqual(T.portable(self.scratch / 'run/a.png'), 'scratch:run/a.png')
        self.assertEqual(T.portable(Path(self._tmp.name) / 'elsewhere/b.dds'), 'b.dds')
        for p in (self.main / 'output/x.pack', self.scratch / 'a', Path(self._tmp.name) / 'b'):
            self.assertIsNone(MACHINE.search(T.portable(p)))

    def test_patchiness_source_is_portable(self):
        import audit_ground_patchiness as agp
        with mock.patch.object(agp.gp, 'BlendSource', lambda **k: k, create=True):
            _, name = agp.source(types.SimpleNamespace(blend=self.main / 'output/blend.dds', lf=None, pack=None))
            self.assertEqual(name, 'output/blend.dds')
            _, name = agp.source(types.SimpleNamespace(blend=None, lf=None, pack=self.scratch / 'p/c.pack'))
            self.assertEqual(name, 'scratch:p/c.pack')


class TrackedFiles(unittest.TestCase):
    def test_no_machine_path(self):
        for rel in TRACKED:
            p = ROOT / rel
            if not p.is_file():
                continue                                   # sparse checkout
            text = p.read_text(encoding='utf-8')
            self.assertIsNone(MACHINE.search(text), rel)
            json.loads(text)


if __name__ == '__main__':
    unittest.main()

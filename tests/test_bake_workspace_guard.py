"""bake_map refuses a bake that writes its map outside the live workspace while TW3K_WORKSPACE is the live one
(2026-11-01 12:04: a verification bake with out in scratch and the live workspace rewrote the live patch caches).
A scratch workspace may bake anywhere; a live bake writes inside the live workspace (BUILD.md map.baked.hex)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import bake_map


class BakeWorkspaceGuardTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); t = Path(self.d.name)
        self.live, self.scratch = t / 'live', t / 'scratch'
        for p in (self.live, self.scratch):
            p.mkdir()

    def tearDown(self):
        self.d.cleanup()

    def test_live_workspace_out_elsewhere_refused(self):
        self.assertIn('outside the live workspace', bake_map.workspace_mix(self.scratch / 'map.hex', self.live, self.live))

    def test_live_workspace_out_inside_allowed(self):
        self.assertIsNone(bake_map.workspace_mix(self.live / 'work/final-map/map.baked.hex', self.live, self.live))

    def test_scratch_workspace_out_anywhere_allowed(self):
        self.assertIsNone(bake_map.workspace_mix(self.scratch.parent / 'cand/map.hex', self.scratch, self.live))

    def test_cli_stops_before_any_work(self):
        """The real entry point exits at once (nothing baked, no cache touched) for the 12:04 combination."""
        live = ROOT / 'work/all-directions-30'
        if not (live / 'work/current-region-design.json').exists():
            self.skipTest('no live workspace in this checkout')
        env = {k: v for k, v in os.environ.items() if not k.startswith('TW3K_')}
        env['TW3K_WORKSPACE'] = str(live)
        out = self.scratch / 'map.hex'
        r = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/bake_map.py'), '--diff', '--out', str(out)],
                           env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('outside the live workspace', r.stdout + r.stderr)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()

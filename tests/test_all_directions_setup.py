"""all_directions_build.setup reads and copies no file of the open-source input blocklist (the root
analysis/standalone-prototype.json, the reference mod's folders, ...) and writes no build report: the workspace
report analysis/standalone-prototype.json has one writer, all_directions_build.database. Runs setup on a minimal fake root and workspace in a temporary folder, recording every path
it opens or copies."""
import builtins
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
BLOCKED = ('standalone-prototype', '190' + '-expanded', '190' + '_expanded', 'iron' + 'ic_')

_RUN = r'''
import builtins, json, shutil, sys
from pathlib import Path
seen = []
_open = builtins.open
def rec_open(f, *a, **k):
    seen.append(str(f)); return _open(f, *a, **k)
builtins.open = rec_open
for name in ('copy2', 'copyfile', 'copytree'):
    f = getattr(shutil, name)
    def wrap(src, dst, *a, _f=f, **k):
        seen.append(str(src)); return _f(src, dst, *a, **k)
    setattr(shutil, name, wrap)
_rb, _rt = Path.read_bytes, Path.read_text
Path.read_bytes = lambda self: (seen.append(str(self)), _rb(self))[1]
Path.read_text = lambda self, *a, **k: (seen.append(str(self)), _rt(self, *a, **k))[1]
import all_directions_build as B
B.ROOT = Path(sys.argv[1]); B.setup()
print(json.dumps(seen))
'''


class Setup(unittest.TestCase):
    def test_no_blocklisted_input_and_no_report(self):
        with tempfile.TemporaryDirectory(prefix='tw3k-setup-tests-') as t:
            t = Path(t)
            root, ws = t / 'root', t / 'ws'
            for rel in ('raw_data/db', 'binaries', 'raw_data/EmpireDesignData/campaigns/3k_main_campaign_map',
                        'raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map'):
                d = root / 'work/standalone_kit' / rel
                d.mkdir(parents=True)
                (d / 'x.txt').write_text('kit', encoding='utf-8')
            (root / 'analysis').mkdir()
            (root / 'analysis/standalone-prototype.json').write_text('{"changed_rows": {"' + 'iron' + 'ic_x": []}}',
                                                                    encoding='utf-8')
            (ws / 'work/final-map').mkdir(parents=True)
            (ws / 'analysis').mkdir()
            (ws / 'work/final-map/map.hex').write_bytes(b'map')
            for src in ('work/current-region-design.json', 'work/canvas-authored-coordinates.json'):
                real = Path(os.environ.get('TW3K_WORKSPACE', SCRIPTS.parent / 'work/all-directions-30')) / src
                if not real.is_file():
                    real = SCRIPTS.parent / 'work/all-directions-30' / src
                if real.is_file():
                    shutil.copy2(real, ws / src)
            if not (ws / 'work/current-region-design.json').is_file():
                self.skipTest('no region design to import campaign_world with')
            env = dict(os.environ, TW3K_WORKSPACE=str(ws))
            r = subprocess.run([sys.executable, '-c', _RUN, str(root)], cwd=SCRIPTS, env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            seen = json.loads(r.stdout.strip().splitlines()[-1])
            hits = [p for p in seen if any(b in p.replace('\\', '/') for b in BLOCKED)
                    and not p.replace('\\', '/').startswith(ws.as_posix())]
            self.assertEqual(hits, [])
            self.assertFalse((ws / 'analysis/standalone-prototype.json').exists())



class OneWriter(unittest.TestCase):
    """Only all_directions_build.database writes the workspace report; prepare_province returns it and restores the
    vanilla tables RESTORE_TABLES plus the previous report's tables (read only)."""
    def test_only_database_writes_the_report(self):
        import re
        writers = []
        for p in sorted(SCRIPTS.glob('*.py')):
            text = p.read_text(encoding='utf-8', errors='replace')
            for line in text.splitlines():
                if 'standalone-prototype.json' in line and re.search(r'write_text|dump\(|json\.dump|open\(.*[\'"]w', line):
                    writers.append(p.name)
            if 'reportpath' in text and re.search(r'dump\(reportpath', text):
                writers.append(p.name)
        self.assertEqual(sorted(set(writers)), ['all_directions_build.py'])

    def test_restore_tables(self):
        sys.path.insert(0, str(SCRIPTS))
        try:
            import prepare_province as pp
        finally:
            sys.path.remove(str(SCRIPTS))
        with tempfile.TemporaryDirectory(prefix='tw3k-restore-tests-') as t:
            prev = Path(t) / 'report.json'
            self.assertEqual(pp.restore_tables(prev), list(pp.RESTORE_TABLES))
            prev.write_text(json.dumps({'changed_rows': {'regions': [], 'building_levels': []}}), encoding='utf-8')
            self.assertEqual(pp.restore_tables(prev), list(pp.RESTORE_TABLES) + ['building_levels'])
        self.assertEqual(len(set(pp.RESTORE_TABLES)), 15)


if __name__ == '__main__':
    unittest.main()

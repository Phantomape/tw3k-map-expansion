"""art_review renders each side with its own region design (coordinator 2026-11-01: the renderer's envcheck default was
a 10-25 design, so a review of the c14r pack would have drawn the old Korea towns): --base-design is required,
--design defaults to the live design (never envcheck), the town list keys the cache per side, the render processes get
the side's TW3K_DESIGN, and every render must have drawn that side's towns. Temporary folders only; nothing renders."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import art_review as A  # noqa: E402
import render_campaign_view as rv  # noqa: E402


def design(path, towns):
    path.write_text(json.dumps({'regions': [dict(region=n, town=[q, r], is_capital=cap) for n, q, r, cap in towns]
                                + [dict(region='sea_x')]}), encoding='utf-8')
    return path


class Design(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-art-design-')
        t = Path(self._tmp.name)
        self.old = design(t / 'old.json', [('a', 10, 20, True), ('b', 30, 40, False)])
        self.new = design(t / 'new.json', [('a', 10, 20, True), ('c', 50, 60, False)])

    def tearDown(self):
        A._W.clear()
        self._tmp.cleanup()

    def test_load_towns_reads_the_given_design(self):
        self.assertEqual([t[0] for t in rv.load_towns(self.old)], ['a', 'b'])
        self.assertEqual([t[0] for t in rv.load_towns(self.new)], ['a', 'c'])

    def test_render_key_per_side(self):
        to, so = A.side_towns(self.old)
        tn, sn = A.side_towns(self.new)
        self.assertNotEqual(so, sn)
        self.assertNotEqual(A.render_key(to), A.render_key(tn))
        self.assertEqual(A.render_key(to), A.render_key(A.side_towns(self.old)[0]))

    def test_missing_design_exits(self):
        with self.assertRaises(SystemExit):
            A.side_towns(Path(self._tmp.name) / 'none.json')
        with self.assertRaises(SystemExit):
            A.side_towns(None)

    def test_base_design_required(self):
        a = argparse.Namespace(base_design=None, design=str(self.new))
        with self.assertRaisesRegex(SystemExit, 'base-design'):
            A.cmd_run(a)

    def test_design_default_is_live_not_envcheck(self):
        got = {}
        with mock.patch.object(A, 'cmd_run', lambda a: got.update(vars(a))), \
                mock.patch.object(sys, 'argv', ['art_review.py', 'run', '--pack', 'p', '--map', 'm']):
            A.main()
        self.assertEqual(Path(got['design']), A.LIVE_DESIGN)
        self.assertNotIn('envcheck', str(got['design']).lower())
        self.assertIsNone(got['base_design'])

    def test_town_changes_and_affected(self):
        ch = A.town_changes(rv.load_towns(self.old), rv.load_towns(self.new))
        self.assertEqual(sorted(ch), sorted([rv.load_towns(self.old)[1][1:3], rv.load_towns(self.new)[1][1:3]]))
        b = rv.load_towns(self.old)[1]
        self.assertTrue(A.town_affected(ch, [[b[1] - .1, b[2] - .1], [b[1] + .1, b[2] + .1]]))
        self.assertEqual(A.town_affected(ch, [[-50, -50], [-49, -49]]), [])
        self.assertEqual(A.town_changes(rv.load_towns(self.old), rv.load_towns(self.old)), [])

    def test_check_towns(self):
        _, s = A.side_towns(self.old)
        good = {('x', 'v1'): ('p', {'towns': {'towns_sha256': s}})}
        A.check_towns(good, s)
        for meta in ({'towns': {'towns_sha256': 'f' * 64}}, {}):
            with self.assertRaises(SystemExit):
                A.check_towns({('x', 'v1'): ('p', meta)}, s)

    def test_recorded_towns_must_be_the_sides(self):
        import campaign_world as cw
        A.record_towns()
        _, s = A.side_towns(self.old)
        with mock.patch.object(cw, 'DESIGN_PATH', self.old):
            rv.load_towns()                                   # what a renderer does (no argument)
            m = A.drawn_towns(self.old, s)
            self.assertEqual((m['towns_sha256'], m['count']), (s, 2))
            rv.load_towns(self.new)                           # a renderer that drew another design
            with self.assertRaisesRegex(RuntimeError, 'drew town list'):
                A.drawn_towns(self.old, s)
        with mock.patch.object(cw, 'DESIGN_PATH', self.new):
            with self.assertRaisesRegex(RuntimeError, 'reads design'):
                A.drawn_towns(self.old, s)

    def test_renderer_has_no_default_design(self):
        import subprocess
        env = {k: v for k, v in os.environ.items() if k not in ('TW3K_DESIGN', 'TW3K_WORKSPACE')}
        r = subprocess.run([sys.executable, '-B', '-c', 'import render_live_view'], cwd=Path(A.__file__).parent, env=env,
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('needs TW3K_DESIGN', r.stderr)
        self.assertNotIn('envcheck', Path(A.__file__).with_name('render_live_view.py').read_text(encoding='utf-8').lower()
                         .split('raise runtimeerror')[1])

    def test_workers_get_the_sides_design(self):
        seen = []

        class Pool:
            def __init__(self, *a, **k): seen.append(os.environ.get('TW3K_DESIGN'))
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def submit(self, *a): raise AssertionError('no job expected')
        before = os.environ.get('TW3K_DESIGN')
        with mock.patch('concurrent.futures.ProcessPoolExecutor', Pool):
            A._render_todo([], 'k', 1, self.old, 's')
        self.assertEqual(seen, [str(self.old.resolve())])
        self.assertEqual(os.environ.get('TW3K_DESIGN'), before)


if __name__ == '__main__':
    unittest.main()

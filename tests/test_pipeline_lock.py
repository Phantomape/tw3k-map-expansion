"""Pipeline lock helper: each slot checked on its own path, one mkdir per slot, no comma or space in names.

Runs in a temporary folder; the real <main checkout>\\work slots are never touched.
"""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pipeline_lock as pl


class PipelineLockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-lock-tests-')
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_slot_names_are_single_components(self):
        self.assertEqual(sorted(pl.SLOTS), [1, 2, 3, 4])
        for n, name in pl.SLOTS.items():
            self.assertTrue(pl.valid_name(name), name)
            for bad in (',', ' ', '/', '\\'):
                self.assertNotIn(bad, name)
            self.assertEqual(pl.path(n, self.root).parent, self.root)
        for bad in ('.pipeline-lock,.pipeline-lock-2', '.pipeline-lock .pipeline-lock-2', '.pipeline-lock-1', '.pipeline-lock-5'):
            self.assertFalse(pl.valid_name(bad), bad)

    def test_each_slot_checked_separately(self):
        self.assertEqual(pl.held(self.root), [])
        for n in pl.SLOTS:
            pl.path(n, self.root).mkdir()
            self.assertEqual(pl.held(self.root), [n], n)
            self.assertEqual(pl.busy(self.root), [pl.SLOTS[n]])
            pl.path(n, self.root).rmdir()
        pl.path(1, self.root).mkdir()
        self.assertEqual(pl.held(self.root, pl.PIPELINE), [1])
        self.assertEqual(pl.held(self.root, pl.HEAVY), [])

    def test_comma_list_directory_is_stray_not_free(self):
        (self.root / '.pipeline-lock,.pipeline-lock-2').mkdir()
        self.assertEqual(pl.held(self.root), [])
        self.assertEqual(pl.stray(self.root), ['.pipeline-lock,.pipeline-lock-2'])
        self.assertEqual(pl.busy(self.root, pl.PIPELINE), ['.pipeline-lock,.pipeline-lock-2'])

    def test_acquire_makes_exactly_one_directory(self):
        token = pl.acquire(3, 'test', self.root, settle=0)
        self.assertIsNotNone(token)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['.pipeline-lock-3'])
        self.assertEqual(pl.owner(3, self.root), token)
        self.assertIsNone(pl.acquire(3, 'other', self.root, settle=0))
        self.assertEqual(pl.owner(3, self.root), token)

    def test_release_only_by_owner(self):
        token = pl.acquire(4, 'test', self.root, settle=0)
        self.assertFalse(pl.release(4, 'someone else', self.root))
        self.assertFalse(pl.release(4, None, self.root))
        self.assertTrue(pl.path(4, self.root).exists())
        self.assertTrue(pl.release(4, token, self.root))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_lost_slot_returns_none(self):
        real_sleep = pl.time.sleep
        def steal(_):
            (pl.path(2, self.root) / pl.OWNER).write_text('thief', encoding='utf-8')
        pl.time.sleep = steal
        try:
            self.assertIsNone(pl.acquire(2, 'test', self.root, settle=0))
        finally:
            pl.time.sleep = real_sleep
        self.assertEqual(pl.owner(2, self.root), 'thief')

    def test_stale_slot_reported_not_removed(self):
        import contextlib, io, os, subprocess
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        dead = f'speedup cam fit {p.pid} b3fc79b3'
        pl.path(4, self.root).mkdir()
        (pl.path(4, self.root) / pl.OWNER).write_text(dead, encoding='utf-8')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertIsNone(pl.acquire(4, 'test', self.root, settle=0))
            self.assertIsNone(pl.acquire(4, 'test', self.root, settle=0))
        self.assertEqual(err.getvalue().count('残锁'), 1)                 # once per process
        self.assertIn(dead, err.getvalue())                               # owner.txt printed
        self.assertEqual(pl.stale(4, self.root), dead)
        self.assertEqual(pl.owner(4, self.root), dead)                    # left in place
        for live in (f'test {os.getpid()} 0123abcd', 'written by hand, no pid', f'by hand run {p.pid} 2026-11-01'):
            (pl.path(4, self.root) / pl.OWNER).write_text(live, encoding='utf-8')
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertIsNone(pl.acquire(4, 'test', self.root, settle=0))
            self.assertIsNone(pl.stale(4, self.root), live)
            self.assertEqual(err.getvalue(), '')
        self.assertIsNone(pl.stale(3, self.root))                         # free slot


class GpuLockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-gpu-lock-tests-')
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def dead_pid(self):
        import subprocess
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        return p.pid

    def test_gpu_lock_is_not_a_pipeline_slot(self):
        self.assertIsNotNone(pl.gpu_acquire('render', self.root, settle=0))
        self.assertEqual(pl.busy(self.root), [])
        self.assertEqual(pl.stray(self.root), [])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['.gpu-lock'])

    def test_exclusive_and_owner_release(self):
        tok = pl.gpu_acquire('render', self.root, settle=0)
        self.assertIsNone(pl.gpu_acquire('polish', self.root, settle=0))
        self.assertFalse(pl.gpu_release('polish 1 x', self.root))
        self.assertTrue(pl.gpu_release(tok, self.root))
        self.assertFalse(pl.gpu_path(self.root).exists())

    def test_stale_lock_of_a_dead_process_is_taken_over(self):
        pl.gpu_path(self.root).mkdir()
        (pl.gpu_path(self.root) / pl.OWNER).write_text(f'polish {self.dead_pid()} deadbeef', encoding='utf-8')
        self.assertTrue(pl.gpu_stale(self.root))
        tok = pl.gpu_acquire('render', self.root, settle=0)
        self.assertIsNotNone(tok)
        self.assertEqual(pl.gpu_owner(self.root), tok)

    def test_live_owner_is_not_stale(self):
        tok = pl.gpu_acquire('render', self.root, settle=0)
        self.assertFalse(pl.gpu_stale(self.root))
        self.assertIsNone(pl.gpu_acquire('polish', self.root, settle=0))
        pl.gpu_release(tok, self.root)

    def test_context_waits_then_gives_up(self):
        import time
        tok = pl.gpu_acquire('polish', self.root, settle=0)
        t = time.time()
        with pl.gpu('render', wait=.3, root=self.root, poll=.1, settle=0) as got:
            self.assertIsNone(got)
        self.assertGreaterEqual(time.time() - t, .3)
        self.assertEqual(pl.gpu_owner(self.root), tok)          # the waiter never touched the holder's lock
        pl.gpu_release(tok, self.root)
        with pl.gpu('render', wait=1, root=self.root, poll=.1, settle=0) as got:
            self.assertIsNotNone(got)
            self.assertTrue(pl.gpu_path(self.root).exists())
        self.assertFalse(pl.gpu_path(self.root).exists())

    def test_pid_alive(self):
        import os
        self.assertTrue(pl.pid_alive(os.getpid()))
        self.assertFalse(pl.pid_alive(self.dead_pid()))


if __name__ == '__main__':
    unittest.main()

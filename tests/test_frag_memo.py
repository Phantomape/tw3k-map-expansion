"""stage_invariants.ground_fragmentation recomputes fragmentation only when its inputs change (blend bytes, the
landform tiles, the rebuilt-area groups, the tones, the code) and otherwise returns the recorded numbers, equal to a
fresh computation. Tiny DDS / npz files in a temporary workspace; the real fragmentation runs (counted)."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault('TW3K_WORKSPACE', str(SCRIPTS.parent / 'work/all-directions-30'))
import ground_patchiness as gp  # noqa: E402
import stage_invariants as si  # noqa: E402

H = W = 160            # blend pixels: 40 x 40 tiles, 20 x 20 hexes


def dds(arr):
    head = bytearray(128)
    struct.pack_into('<2I', head, 12, H, W)
    return bytes(head) + np.ascontiguousarray(arr[::-1]).tobytes()


class Memo(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-frag-memo-')
        t = Path(self._tmp.name)
        self.ws = t / 'ws'
        (self.ws / 'analysis').mkdir(parents=True)
        rng = np.random.default_rng(7)
        self.blend = np.zeros((H, W, 2), np.uint8)
        self.blend[..., 0] = rng.integers(0, 4, (H, W))
        self.hexmask = np.zeros((20, 20), bool)
        self.hexmask[:, 10:] = True
        self.base, self.rebuilt = t / 'baseline.json', t / 'rebuilt.npz'
        self.base.write_text(json.dumps({'tolerance': .1, 'landforms': {'a': {'E': 1e9, 'n100': 1e9}}}), encoding='utf-8')
        np.savez(self.ws / si.LANDFORM_TILES, meta=np.array([json.dumps({'landforms': ['a']})]),
                 dom=np.ones((40, 40), np.uint8), domw=np.ones((40, 40), np.float32))
        (self.ws / si.MAP).parent.mkdir(parents=True, exist_ok=True)
        (self.ws / si.MAP).write_bytes(b'map')
        self.write()
        self.calls = 0
        real = gp.fragmentation

        def fragmentation(*a, **k):
            self.calls += 1
            return real(*a, **k)
        self.p = [mock.patch.object(gp, 'fragmentation', fragmentation), mock.patch.object(si, 'FRAG_BASELINE', self.base),
                  mock.patch.object(si, 'FRAG_REBUILT', self.rebuilt)]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        self._tmp.cleanup()

    def write(self):
        p = self.ws / si.BLEND
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(dds(self.blend))
        np.savez(self.rebuilt, mask=np.packbits(self.hexmask.ravel()), shape=np.array(self.hexmask.shape),
                 meta=np.array(json.dumps({'map_b_sha256': 'x'})))

    def run_(self):
        return si.ground_fragmentation(si.Ctx(self.ws))

    def test_memo_hit_equals_fresh(self):
        first = self.run_()
        self.assertEqual(self.calls, 1)
        self.assertEqual(set(first['detail']['outside']), {'a'})
        self.assertEqual(set(first['detail']['rebuilt']), {'a'})
        self.assertEqual(self.run_(), first)                # unchanged inputs: recorded numbers, same result
        self.assertEqual(self.calls, 1)
        (self.ws / si.FRAG_MEMO).unlink()
        self.assertEqual(self.run_(), first)                # a fresh computation gives the same
        self.assertEqual(self.calls, 2)

    def test_input_changes_recompute(self):
        self.run_()
        self.blend[0, 0, 0] = 3 - self.blend[0, 0, 0]       # the blend changes
        self.write()
        self.run_()
        self.assertEqual(self.calls, 2)
        self.hexmask[:, 8:10] = True                        # the rebuilt area (groups) changes
        self.write()
        self.run_()
        self.assertEqual(self.calls, 3)
        np.savez(self.ws / si.LANDFORM_TILES, meta=np.array([json.dumps({'landforms': ['a']})]),
                 dom=np.ones((40, 40), np.uint8), domw=np.full((40, 40), .9, np.float32))   # landform weights
        self.run_()
        self.assertEqual(self.calls, 4)
        self.run_()
        self.assertEqual(self.calls, 4)

    def test_unreadable_memo_recomputes(self):
        self.run_()
        (self.ws / si.FRAG_MEMO).write_text('not json', encoding='utf-8')
        self.run_()
        self.assertEqual(self.calls, 2)
        self.assertEqual(len(json.loads((self.ws / si.FRAG_MEMO).read_text(encoding='utf-8'))['groups']), 2)


if __name__ == '__main__':
    unittest.main()

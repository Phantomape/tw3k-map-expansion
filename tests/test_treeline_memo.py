"""stage_invariants.treeline_straight recomputes line_numbers only when its inputs change (blend bytes, the LF land
mask, the class-line hexes, the rule, the tones, the code) and otherwise returns the recorded numbers. Tiny DDS
files in a temporary workspace; line_numbers and class_line_hexes are stubbed (their own tests cover them)."""
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
import campaign_world as cw  # noqa: E402
import ground_patchiness as gp  # noqa: E402
import stage_invariants as si  # noqa: E402

H, W = 12, 16


def dds(arr):
    head = bytearray(128)
    struct.pack_into('<2I', head, 12, H, W)
    return bytes(head) + np.ascontiguousarray(arr[::-1]).tobytes()


def lf_raw(height):
    """uint16 LF raw value for a height (inverse of band()'s decode)."""
    return np.uint16(np.clip(round((cw.NORM + (height - cw.LEVEL) / cw.GAIN) * 65535), 0, 65535))


class Memo(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-treeline-memo-')
        self.ws = Path(self._tmp.name)
        (self.ws / 'analysis').mkdir()
        self.blend = np.zeros((H, W, 2), np.uint8)
        self.lf = np.full((H, W), lf_raw(1.0), np.uint16)
        self.lf[:, :4] = lf_raw(-1.0)                       # some water
        self.write()
        np.savez(self.ws / si.LANDFORM_TILES, meta=np.array([json.dumps({'biome_names': ['a']})]),
                 eco_bio=np.zeros((2, 2), np.uint8))
        self.calls = 0

        def line_numbers(band, sel, tones):
            self.calls += 1
            return {'straight_segment_per100': .25, 'land_hexes': 10.5, 'hard_edge_per100': 1., 'hard_tile_share': .5,
                    'straight_segments': 3, 'straight_segment_max_hex': 2.5, 'blocks': 1}
        self.p = [mock.patch.object(gp, 'line_numbers', line_numbers),
                  mock.patch.object(gp, 'class_line_hexes', lambda *a, **k: np.ones((2, 2), bool))]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        self._tmp.cleanup()

    def write(self):
        for rel, arr in ((si.BLEND, self.blend), (si.LF, self.lf)):
            p = self.ws / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(dds(arr))

    def run_(self):
        return si.treeline_straight(si.Ctx(self.ws))

    def test_memo_hit_and_misses(self):
        first = self.run_()
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.run_(), first)
        self.assertEqual(self.calls, 1)                     # unchanged inputs: recorded numbers
        self.lf[:, 8:] = lf_raw(2.0)                        # heights change, the land mask does not
        self.write()
        self.assertEqual(self.run_(), first)
        self.assertEqual(self.calls, 1)
        self.lf[:, 4] = lf_raw(-1.0)                        # the land mask changes
        self.write()
        self.run_()
        self.assertEqual(self.calls, 2)
        self.blend[0, 0, 0] = 3                             # the blend changes
        self.write()
        self.run_()
        self.assertEqual(self.calls, 3)

    def test_unreadable_memo_recomputes(self):
        self.run_()
        (self.ws / si.TREELINE_MEMO).write_text('not json', encoding='utf-8')
        self.run_()
        self.assertEqual(self.calls, 2)
        self.assertEqual(json.loads((self.ws / si.TREELINE_MEMO).read_text(encoding='utf-8'))['numbers']['blocks'], 1)


if __name__ == '__main__':
    unittest.main()

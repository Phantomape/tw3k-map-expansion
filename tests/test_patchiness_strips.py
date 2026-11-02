"""ground_patchiness.analyse_map with strip workers and the strip cache equals the one-process pass bit for bit
(every accumulator array, the tile maps and the islands) on a synthetic blend + LF."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ground_patchiness as gp


def dds(path, a):
    head = bytearray(128)
    struct.pack_into('<2I', head, 12, a.shape[0], a.shape[1])
    path.write_bytes(bytes(head) + np.ascontiguousarray(a).tobytes())


def reference(src, T):
    """The one-process pass as it was before strips were split out (Acc updated in place, strip by strip)."""
    W, Hh = src.hexes
    acc = gp.Acc(0, W - 1, 0, Hh - 1)
    H = src.h - src.h % 4
    Lt = np.zeros((H // 4, src.w // 4), np.float32); lt = np.zeros_like(Lt)
    for y in range(0, H, gp.STRIP_PX):
        ya, yb = max(0, y - gp.STRIP_MARGIN), min(H, y + gp.STRIP_PX + gp.STRIP_MARGIN)
        R, land = src.band(ya, yb)
        c0, c1 = y - ya, min(H, y + gp.STRIP_PX) - ya
        L = gp.pixel_pass(R, land, 0, ya, T, acc, core=(c0, c1))
        a, b = gp.tile_lightness(L[c0:c1], land[c0:c1])
        Lt[y // 4:y // 4 + len(a)], lt[y // 4:y // 4 + len(a)] = a, b
    ilab, isl = gp.dark_islands(Lt, lt)
    ty, tx = np.nonzero(ilab)
    acc.add('island_tiles', *gp.px_hex(tx * 4, ty * 4))
    for i in isl:
        i['centre_hex'] = [int(v) for v in gp.px_hex(i['centre_tile'][0] * 4, i['centre_tile'][1] * 4)]
    return acc, isl


class Strips(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp(prefix='tw3k-patchiness-'))
        rng = np.random.default_rng(3)
        h, w = 2 * gp.STRIP_PX + 260, 640
        yy, xx = np.mgrid[:h, :w]
        R = ((xx // 37 + yy // 29) % 7 + 3).astype(np.uint8)
        R[(xx // 64 + yy // 64) % 5 == 0] = 12                     # straight-edged patches
        R[rng.random((h, w)) < .02] = 5                            # speckle
        for k, x0 in enumerate(range(80, w - 60, 90)):             # long straight edges between unlike slots
            R[100:h - 100, x0:x0 + 45] = (0, 2, 4, 6, 14, 20)[k % 6]
        for _ in range(300):                                       # small square blobs of the square slots
            y, x = rng.integers(0, h - 12), rng.integers(40, w - 12)
            R[y:y + rng.integers(4, 12), x:x + rng.integers(4, 12)] = (11, 31)[_ % 2]
        blend = np.stack([R, np.zeros_like(R)], -1)
        import campaign_world as cw
        lf = np.full((h, w), int((gp.PLANE - cw.LEVEL) / cw.GAIN * 65535 + cw.NORM * 65535) + 2000, np.uint16)
        lf[:, :40] = 0                                             # sea strip
        dds(cls.d / 'blend.dds', blend[::-1]); dds(cls.d / 'lf.dds', lf[::-1])
        cls.src = gp.BlendSource(blend=cls.d / 'blend.dds', lf=cls.d / 'lf.dds')
        cls.T = gp.Tones('harvest')
        cls.ref = reference(cls.src, cls.T)

    def same(self, got):
        acc, isl = got
        racc, risl = self.ref
        for k in gp.Acc.FIELDS:
            self.assertTrue(np.array_equal(acc.a[k].view(np.uint32), racc.a[k].view(np.uint32)), k)
        self.assertEqual(isl, risl)

    def test_one_process(self):
        self.same(gp.analyse_map(self.src, self.T, log=lambda *_: None))

    def test_workers_and_cache(self):
        cache = self.d / 'cache'
        self.same(gp.analyse_map(self.src, self.T, log=lambda *_: None, workers=2, cache=cache))
        self.assertEqual(len(list(cache.glob('strip-*.pkl'))), 3)
        self.same(gp.analyse_map(self.src, self.T, log=lambda *_: None, cache=cache))   # all strips from the cache

    def test_cache_key_follows_input_bytes(self):
        H = self.src.h - self.src.h % 4
        ya, yb = gp._strip_rows(self.src, 0, H)
        k = gp._strip_key(self.src, self.T, ya, yb)
        self.assertEqual(k, gp._strip_key(self.src, self.T, ya, yb))
        self.assertNotEqual(k, gp._strip_key(self.src, self.T, ya, yb - 4))


if __name__ == '__main__':
    unittest.main()

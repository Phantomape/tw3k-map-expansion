"""layout_cleanup.mask (post-release todo #49, option c): with data/map-layout/layout-cleanup.npz the mask is the one
polish computed inline before (the old code is kept here verbatim as the reference, run on the real npz); without it
the mask is all 0, a WARN is printed and nothing raises (the public source build)."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFilter

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import layout_cleanup as lc  # noqa: E402
import map_layers as ml  # noqa: E402


def blur(a, radius):   # polish_campaign_terrain.blur
    return np.asarray(Image.fromarray(np.clip(a * 255, 0, 255).astype('uint8')).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255


def old_inline(TH, TW, H, W, land, hy, hx, plan, q, r, _town_window, _town_centre, npz):
    """polish_campaign_terrain.py before #49 (main dcee1aac, lines 586-601), unchanged but for the npz path."""
    layout_cleanup = np.zeros((TH, TW), np.float32)
    _lcz = ml.load_npz(npz)
    if _lcz['old_land'].shape != (H, W) or _lcz['changed_coast'].shape != (H, W): raise RuntimeError('canvas')
    oldland = _lcz['old_land'].astype(bool); changed_land = (oldland != land)[hy, hx]
    layout_cleanup = np.maximum(layout_cleanup, blur(changed_land.astype(float), 3)); layout_cleanup[changed_land] = 1
    for town in plan['regions']:
        for prior_town in town.get('cleanup_towns', [town.get('previous_town')]):
            if prior_town is None or prior_town == town['town']: continue
            tq, tr = prior_town; win = _town_window(tq, tr, 16); sq, sr = _town_centre(tq, tr); distance_old = np.hypot(q[win] - sq, r[win] - sr)
            t = np.clip((distance_old - 7) / 9, 0, 1); layout_cleanup[win] = np.maximum(layout_cleanup[win], 1 - t * t * (3 - 2 * t))
    if plan.get('coast_coverage_revision') == 4:
        changed = _lcz['changed_coast'].astype(bool)[hy, hx]
        layout_cleanup = np.maximum(layout_cleanup, blur(changed.astype(float), 3)); layout_cleanup[changed] = 1
    return layout_cleanup


@unittest.skipUnless(lc.NPZ.exists(), 'private data: layout-cleanup.npz is not in the public snapshot')
class Present(unittest.TestCase):
    def test_byte_identical_to_old_inline(self):
        z = ml.load_npz(lc.NPZ)
        H, W = z['old_land'].shape
        rng = np.random.default_rng(49)
        land = z['old_land'].copy()
        land[rng.random((H, W)) < .01] ^= True          # a current map that differs from the old layout
        # hex resolution tile grid (hy, hx identity) on a crop keeps the test small; the npz itself is whole
        TH, TW = 512, 640
        y0, x0 = 700, 900
        hy, hx = np.meshgrid(np.arange(y0, y0 + TH), np.arange(x0, x0 + TW), indexing='ij')
        q, r = hx.astype(np.float64), hy.astype(np.float64)
        tw = lambda tq, tr, reach: (np.abs(q - tq) <= reach) & (np.abs(r - tr) <= reach)
        tc = lambda tq, tr: (tq + .5, tr + .5)
        towns = [[x0 + 100, y0 + 100], [x0 + 300, y0 + 400]]
        for rev in (4, 3):
            plan = {'coast_coverage_revision': rev, 'regions': [
                {'town': towns[0], 'previous_town': [x0 + 120, y0 + 90]},
                {'town': towns[1], 'cleanup_towns': [[x0 + 310, y0 + 380], towns[1]]},
                {'town': [x0 + 5, y0 + 5]}]}
            want = old_inline(TH, TW, H, W, land, hy, hx, plan, q, r, tw, tc, lc.NPZ)
            got = lc.mask((TH, TW), land, hy, hx, plan, q, r, blur, tw, tc)
            self.assertEqual(got.dtype, want.dtype)
            self.assertEqual(got.tobytes(), want.tobytes())
            self.assertGreater(int((got > 0).sum()), 0)


class Missing(unittest.TestCase):
    def call(self, t, marker):
        land = np.ones((8, 10), bool)
        hy, hx = np.meshgrid(np.arange(8), np.arange(10), indexing='ij')
        plan = {'coast_coverage_revision': 4, 'regions': [{'town': [1, 1], 'previous_town': [5, 5]}]}
        return lc.mask((8, 10), land, hy, hx, plan, hx * 1., hy * 1., blur,
                       lambda *a: self.fail('no window without data'), lambda *a: self.fail('no centre'),
                       path=Path(t) / 'layout-cleanup.npz', marker=marker)

    def test_private_tree_missing_is_an_error(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                self.call(t, Path(t) / 'PUBLIC_SOURCE')                  # no marker: the private repository

    def test_marker_is_the_snapshot_one(self):
        import make_release_snapshot as S
        self.assertEqual(lc.MARKER.name, S.PUBLIC_SOURCE)
        self.assertEqual(lc.MARKER.parent, SCRIPTS.parent)
        self.assertFalse(lc.MARKER.exists())                             # never tracked in the private repository

    def test_public_tree_all_zero_with_warn(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / 'PUBLIC_SOURCE').write_text('public\n', encoding='utf-8')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                got = self.call(t, Path(t) / 'PUBLIC_SOURCE')
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(got.shape, (8, 10))
        self.assertFalse(got.any())
        self.assertIn('WARN', buf.getvalue())
        self.assertIn('layout cleanup skipped', buf.getvalue())


if __name__ == '__main__':
    unittest.main()

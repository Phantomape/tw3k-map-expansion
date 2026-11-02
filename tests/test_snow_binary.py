"""snow_bake's binary mask (dither_field, mip_levels) and stage_invariants.snow_mask_binary_diff on hand-made arrays.
The shader is a switch (analysis/snow-v22b-findings.md §3): outside the native core every byte is 0 or >= 232."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import snow_bake as sb
import stage_invariants as si


def smooth_field(h, w, seed=3):
    """A smooth stand-in for the fbm field (blurred noise)."""
    from scipy import ndimage
    return ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal((h, w)), 6)


class DitherTest(unittest.TestCase):
    def test_share_becomes_coverage(self):
        t = sb.dither_field(smooth_field(256, 256))
        self.assertAlmostEqual(float(t.min()), .5 / t.size, places=9)
        for p in (.1, .4, .8):
            self.assertAlmostEqual(float((p > t).mean()), p, delta=.005)
        self.assertEqual(float((0. > t).mean()), 0.)
        self.assertEqual(float((1. > t).mean()), 1.)

    def test_mips_stay_binary_in_region(self):
        fz = smooth_field(64, 96)
        top = np.zeros((64, 96, 4), np.uint8)
        top[..., 3] = np.where(.35 > sb.dither_field(fz), 255, 0)
        top[20:24, 30:34, :3] = 255
        region = np.ones((64, 96), bool)
        levels = sb.mip_levels(top, (region, fz))
        self.assertEqual(levels[-1].shape[:2], (1, 1))
        for lv in levels:
            self.assertTrue(np.isin(lv, (0, 255)).all())
        self.assertAlmostEqual(float((levels[2][..., 3] > 0).mean()), .35, delta=.06)   # cover kept at distance
        bad, _ = si.snow_mask_binary_diff(levels, ~region)
        self.assertEqual(bad, 0)

    def test_box_mips_are_flagged(self):
        top = np.zeros((32, 32, 4), np.uint8); top[::2, ::2, 3] = 255   # 25 % cover: box mips give 64
        levels = sb.mip_levels(top)
        bad, per = si.snow_mask_binary_diff(levels, np.zeros((32, 32), bool))
        self.assertEqual(per[0], 0)
        self.assertGreater(per[1], 0)
        core = np.ones((32, 32), bool)
        self.assertEqual(si.snow_mask_binary_diff(levels, core)[0], 0)        # vanilla's core is exempt

    def test_shade_outside_core(self):
        lv = np.zeros((8, 8, 4), np.uint8); lv[1, 1, 3] = 102; lv[6, 6, 3] = 231; lv[7, 7, 3] = 232
        core = np.zeros((8, 8), bool); core[6:, 6:] = True
        bad, per = si.snow_mask_binary_diff([lv], core)
        self.assertEqual((bad, per), (1, [1]))


class PeakTest(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:80, :80].astype(np.float64)
        self.r = np.hypot(y - 40, x - 40)
        self.lf = np.clip(9 - self.r * .25, 2, None)                  # an LF cone, 7 world units over its plain
        self.lat = np.full((80, 80), 43.); self.lon = np.full((80, 80), 88.)
        line = float(sb.bp.perennial_snowline_m(np.float32(43.), np.float32(88.)))
        self.line = line
        self.er = np.full((80, 80), 1000.)                             # regional mean: no dry-plateau raise

    def test_cone_above_line_has_white_top(self):
        e = np.full((80, 80), 1000.); e[self.r < 4] = self.line + 600
        cap = sb.visible_peaks(self.lf, e, self.lat, self.lon, self.er)
        self.assertTrue(cap[40, 40])
        self.assertFalse(cap[40, 75])                                   # the plain stays bare
        self.assertTrue((self.r[cap] < 6).all())                         # the top only

    def test_white_area_follows_real_area(self):
        """More real ground above the line, more white, always from the LF top down."""
        sizes = []
        for rad in (3, 6, 9):
            e = np.full((80, 80), 1000.); e[self.r < rad] = self.line + 600
            cap = sb.visible_peaks(self.lf, e, self.lat, self.lon, self.er)
            sizes.append(int(cap.sum()))
            self.assertLessEqual(float(self.r[cap].max()), float(self.r[~cap & (self.r < 30)].min()) + 1.5)
        self.assertTrue(sizes[0] < sizes[1] < sizes[2], sizes)

    def test_cone_below_line_bare(self):
        e = np.full((80, 80), 1000.); e[self.r < 4] = self.line - 300
        self.assertEqual(int(sb.visible_peaks(self.lf, e, self.lat, self.lon, self.er).sum()), 0)

    def test_small_body_keeps_its_top(self):
        """Altai: a few real pixels above the line (~1 % of the box) still give the visible top a white pixel."""
        e = np.full((80, 80), 1000.); e[40, 40] = self.line + 300; e[41, 40] = self.line + 100
        cap = sb.visible_peaks(self.lf, e, self.lat, self.lon, self.er)
        self.assertTrue(cap[40, 40])
        self.assertLessEqual(int(cap.sum()), 9)

    def test_saturated_flat_top_not_flooded(self):
        """LF at its ceiling: a flat top of equal pixels must not all count as box tops (Hengduan 7.6x, K 1.4 trial)."""
        lf = np.minimum(self.lf, 7.)                                     # flat top, r < 8
        e = np.full((80, 80), 1000.); e[40, 40] = self.line + 300
        cap = sb.visible_peaks(lf, e, self.lat, self.lon, self.er)
        self.assertGreater(int(cap.sum()), 0)
        self.assertLessEqual(int(cap.sum()), 9)

    def test_e_region_required(self):
        e = np.full((80, 80), self.line + 800)
        with self.assertRaises(TypeError):
            sb.visible_peaks(self.lf, e, self.lat, self.lon)
        with self.assertRaises(ValueError):
            sb.visible_peaks(self.lf, e, self.lat, self.lon, None)

    def test_flat_plateau_bare(self):
        e = np.full((80, 80), self.line + 800)
        self.assertEqual(int(sb.visible_peaks(np.full((80, 80), 9.), e, self.lat, self.lon, self.er).sum()), 0)

    def test_visible_count(self):
        g = np.zeros((80, 80), np.uint8); g[38:42, 38:42] = 255
        lat = 43 + (40 - np.mgrid[:80, :80][0]) * .01; lon = 88 + (np.mgrid[:80, :80][1] - 40) * .01
        pts = (('A', 88., 43.), ('B', 88.3, 43.))
        bad, counts = si.snow_peaks_visible_count(g, lat, lon, pts, {}, 15, 15)
        self.assertEqual(counts, {'A': 16, 'B': 0}); self.assertEqual(bad, ['B'])
        self.assertEqual(si.snow_peaks_visible_count(g, lat, lon, pts, {'B': 'basin'}, 15, 15)[0], [])


class LfOrderTest(unittest.TestCase):
    SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'

    def test_pipeline_side_lane_has_no_lf_reader(self):
        self.assertEqual(si.side_lane_lf(None)['value'], 0)

    def test_snow_bake_in_side_lane_is_caught(self):
        text = "$tw3kSideLaneSteps=@('expand_campaign_water_colour.py','snow_bake.py');$tw3kSideLane=@()"
        self.assertEqual(si.side_lane_lf_readers(text, self.SCRIPTS), ['snow_bake.py'])
        self.assertIsNone(si.side_lane_lf_readers('no list here', self.SCRIPTS))

    def test_lf_mask_releases_the_file(self):
        """No handle outlives lf_mask: the LF can be replaced right after (sync_final rewrites it in place)."""
        import struct, tempfile, os
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d); p = sb.lf_path(ws); p.parent.mkdir(parents=True)
            h = w = 64
            head = bytearray(128); head[:4] = b'DDS '; struct.pack_into('<2I', head, 12, h, w)
            p.write_bytes(bytes(head) + np.full(h * w, 30000, '<u2').tobytes())
            out = sb.lf_mask(ws, (8, 8))
            self.assertTrue(np.allclose(out, sb.cw.LEVEL + (30000 / 65535 - sb.cw.NORM) * sb.cw.GAIN))
            os.replace(p, p.with_suffix('.old')); p.with_suffix('.old').unlink()


class SwitchTest(unittest.TestCase):
    def run_with(self, bake, cov):
        import os
        old = {k: os.environ.get(k) for k in ('TW3K_SNOW_BAKE', 'TW3K_SNOW_COVERAGE')}
        os.environ['TW3K_SNOW_BAKE'], os.environ['TW3K_SNOW_COVERAGE'] = bake, cov
        try:
            return si.snow_switches(None)
        finally:
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v

    def test_bake_needs_coverage(self):
        self.assertEqual(self.run_with('1', '0')['value'], 1)
        self.assertEqual(self.run_with('1', '1')['value'], 0)
        with self.assertRaises(si.Missing):
            self.run_with('0', '0')


if __name__ == '__main__':
    unittest.main()

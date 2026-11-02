"""lake_level (v3): the level solver, the waterline signed distance (stops at plot hexes, never cuts into the lake),
the bank / bed profile, the bank envelope (town level solve), the projection bounds, the product round trip and reach, the water mesh pad and the off-path identity of the helpers that read it.

No project stage or game assets are read (campaign_world gets a minimal design in a temporary folder).
"""
import atexit
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
_tmp = tempfile.TemporaryDirectory(prefix='tw3k-lake-level-tests-')
atexit.register(_tmp.cleanup)
_design = Path(_tmp.name) / 'design.json'
_design.write_text(json.dumps({'grid': [2048, 2047], 'padding': [888, 375, 268, 970],
                               'old_grid_offset': [888, 375], 'tile_aligned_frame': True}), encoding='utf-8')
os.environ['TW3K_WORKSPACE'] = _tmp.name
os.environ['TW3K_DESIGN'] = str(_design)
import lake_level as LL
import river_water_mesh as M

SPEC, _ = LL.load_spec()
ENTRY = SPEC['lakes']['bosten']
PITCH = LL.lf_pitch()


def plateau_lake(shape=(40, 40)):
    """A 6 x 8 lake on a plateau at 5, ground rising eastwards by .01 per cell, one low notch on the shore."""
    yy, xx = np.indices(shape)
    relief = (5. + .01 * xx).astype(np.float32)
    lake = (yy >= 15) & (yy < 21) & (xx >= 14) & (xx < 22)
    relief[lake] = 5.4                      # the DEM lake surface stands over its shore, as at Bosten
    relief[14, 16] = 4.0                    # one low shore cell (the percentile ignores it)
    return relief, lake, ~lake


def hex_lake(shape=(48, 56)):
    """A lake of whole hexes (2 x 2 cell bricks, odd columns shifted one cell), as polish's tile masks draw them."""
    ty, tx = np.indices(shape)
    q = tx // 2
    r = (ty - q % 2) // 2
    return (q - 14) ** 2 / 36. + (r - 12) ** 2 / 16. <= 1


class SolverTest(unittest.TestCase):
    def test_constants_shared(self):
        self.assertEqual(M.LAKE_WATER, -.015)
        self.assertEqual(M.SURFACE_MARGIN, .03)
        self.assertEqual(M.FLOOR, LL.LAKE_WATER + LL.SURFACE_MARGIN)

    def test_switch_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('TW3K_LAKE_LEVEL', None)
            self.assertFalse(LL.enabled())
            os.environ['TW3K_LAKE_LEVEL'] = '1'
            self.assertTrue(LL.enabled())
            os.environ.pop('TW3K_LAKE_LEVEL')

    def test_level_is_ring_percentile_minus_level_below_ring(self):
        relief, lake, land = plateau_lake()
        r = LL.solve_level(relief, lake, land, ENTRY)
        ring = LL.shore_ring(lake, land, ENTRY['ring_cells'])
        self.assertFalse((ring & lake).any())
        want = float(np.percentile(relief[ring].astype(np.float64), ENTRY['ring_percentile'])) - ENTRY['level_below_ring']
        self.assertAlmostEqual(r['level'], round(want, 4), places=6)
        self.assertGreater(r['level'], 4.8)       # the single low notch does not set the level
        self.assertLess(r['level'], 5.4 - .12)    # nor does the DEM lake surface

    def test_level_needs_shore(self):
        _, lake, _ = plateau_lake()
        with self.assertRaises(ValueError):
            LL.solve_level(np.zeros(lake.shape, np.float32), lake, np.zeros(lake.shape, bool), ENTRY)

    def test_find_component_from_seed_and_nearby(self):
        _, lake, _ = plateau_lake()
        other = np.zeros_like(lake); other[2:4, 2:4] = True
        m, n = LL.find_component(lake | other, (17, 17), 6)
        self.assertTrue(np.array_equal(m, lake)); self.assertEqual(n, 2)
        m, _ = LL.find_component(lake | other, (12, 18), 6)
        self.assertTrue(np.array_equal(m, lake))
        with self.assertRaises(ValueError):
            LL.find_component(lake, (35, 35), 3)

    def test_spec_numbers_within_vanilla_ranges(self):
        """The frozen parameters sit in the vanilla study's measured ranges (lessons.md)."""
        e, b, d = ENTRY, ENTRY['bank'], ENTRY['bed']
        self.assertTrue(.25 <= e['outline_offset'] <= .45)
        self.assertTrue(.5 <= e['mesh_pad'] <= 1.)
        self.assertGreaterEqual(e['freeboard'] - LL.SURFACE_MARGIN, .05 - 1e-9)   # land under the pad >= .05 over the water surface
        cap = lambda s: float(LL.bank_cap(np.float64(s), b))
        self.assertTrue(.1 <= cap(.3) <= .3 and .1 <= cap(.6) <= .3 and cap(3.) <= .6)
        slope0 = b['cap_rise'] / b['cap_length'] + b['cap_slope']
        self.assertTrue(.3 <= slope0 <= .8)
        bed_slope0 = 2 * d['depth'] / d['floor_at']
        self.assertTrue(1.5 <= bed_slope0 <= 2.5)
        self.assertTrue(.6 <= d['floor_at'] <= 1.2 and 1. <= d['depth'] <= 1.5)


class OutlineTest(unittest.TestCase):
    def setUp(self):
        self.lake = hex_lake()
        self.s, self.info = LL.outline_sdf(self.lake, None, ENTRY, PITCH)

    def test_every_lake_cell_wet_and_waterline_outside_the_hexes(self):
        st = LL.tile_samples(self.s)
        self.assertTrue((st[self.lake] < 0).all())
        self.assertEqual(self.info['lake_px_dry'], 0)
        from scipy import ndimage
        rim = ndimage.binary_dilation(self.lake) & ~self.lake
        self.assertGreater(float((st[rim] < 0).mean()), .5)
        far = ~ndimage.binary_dilation(self.lake, iterations=4)
        self.assertTrue((st[far] > 0).all())

    def test_waterline_smooth_not_hex_stepped(self):
        """Share of waterline gradient directions within 10 degrees of the axes (vanilla .08-.32, hex blocks .49-.67)."""
        gz, gx = np.gradient(self.s.astype(np.float64), *PITCH)
        wl = np.abs(self.s) < .06
        ang = np.degrees(np.arctan2(np.abs(gz[wl]), np.abs(gx[wl])))
        self.assertLess(float(((ang <= 10) | (ang >= 80)).mean()), .35)

    def test_plot_stops_the_growth_at_the_lake_hex_edge(self):
        """v3 (plan B): the waterline stops at the lake-hex edge beside a plot hex (the plot stays dry, right at the
        water) and never cuts back into the lake: every lake pixel stays inside it, every plot pixel outside."""
        from scipy import ndimage
        ys, xs = np.nonzero(self.lake)
        plot = np.zeros_like(self.lake)
        r0 = ys.max() - 1
        plot[r0:r0 + 5, 26:30] = True                       # a plot hex block in a notch: lake cells beside it on 2 sides
        plot &= ~self.lake
        self.assertTrue(plot.any() and (ndimage.binary_dilation(plot) & self.lake).any())
        s, info = LL.outline_sdf(self.lake, plot, ENTRY, PITCH)
        lk = np.repeat(np.repeat(self.lake, 4, 0), 4, 1); pl = np.repeat(np.repeat(plot, 4, 0), 4, 1)
        half = .25 * sum(PITCH)
        self.assertTrue((s[lk] <= -half + 1e-6).all())
        self.assertTrue((s[pl] >= half - 1e-6).all())
        self.assertEqual(info['lake_px_dry'], 0)
        self.assertGreater(info['plot_px_beside_water'], 0)   # the waterline runs along the plot's lake-hex edge
        # elsewhere the line is the v2 one: same field away from the plot
        s0, _ = LL.outline_sdf(self.lake, None, ENTRY, PITCH)
        far = ~ndimage.binary_dilation(pl, iterations=40) & (np.abs(s0) < 1.)   # nearest edge beyond the plot's reach
        np.testing.assert_allclose(s[far], s0[far], atol=1e-6)

    def test_tile_samples_match_the_bilinear_frame(self):
        """tile t is read at LF pixel 4t + 1.5 (PIL bilinear: pixel centres)."""
        f = np.add.outer(np.arange(40.), 10 * np.arange(48.)).astype(np.float32)
        t = LL.tile_samples(f)
        ty, tx = np.indices(t.shape)
        np.testing.assert_allclose(t, (4 * ty + 1.5) + 10 * (4 * tx + 1.5), rtol=1e-6)
        np.testing.assert_array_equal(LL.vertex_samples(f), f[2::4, 2::4])


class ProfileTest(unittest.TestCase):
    L = 4.8

    def test_profile_continuous_and_on_the_right_side(self):
        b = ENTRY['bank']
        s = np.linspace(-3, 8, 2201)
        for R in (4.7, 4.85, 5.3, 6.5, 9.):
            h, w = LL.shaped_ground(np.full_like(s, R), s, self.L, ENTRY)
            self.assertAlmostEqual(float(h[np.argmin(np.abs(s))]), self.L, delta=.01)
            self.assertTrue((h[s < 0] < self.L).all())
            self.assertTrue((h[(s >= 0) & (s <= b["ring_width"])] >= self.L).all())
            dry = (s >= b['floor_length']) & (s <= b['ring_width'])
            self.assertTrue((h[dry] >= self.L + ENTRY['freeboard'] - 1e-9).all())
            far = s >= b['ring_width'] + b['blend_width']
            self.assertTrue((w[far] == 0).all())
            self.assertTrue(np.allclose(h[far], R))
            self.assertLess(float(np.abs(np.diff(h)).max() / (s[1] - s[0])), 3.)   # no vertical step
        h, _ = LL.shaped_ground(np.full_like(s, 9.), s, self.L, ENTRY)
        self.assertAlmostEqual(float(h[s <= -ENTRY['bed']['floor_at']].max()), self.L - ENTRY['bed']['depth'], places=6)

    def test_projection_bounds_hold_the_profile(self):
        s = np.linspace(-3, 6, 901)
        rw = ENTRY['bank']['ring_width']
        h, _ = LL.shaped_ground(np.full_like(s, 6.), s, self.L, ENTRY)
        fl, cl = LL.projection_bounds(s, self.L, ENTRY, 2e-4)
        self.assertTrue((cl[s < 0] < self.L).all())
        self.assertTrue((h[s < 0] <= cl[s < 0] + 1e-6).all())
        on = np.isfinite(fl)
        self.assertTrue((fl[on] <= h[on] + 2e-4 + 1e-6).all())
        self.assertTrue((fl[on] > self.L).all())
        bank = (s >= 0) & (s <= rw)
        self.assertTrue((h[bank] <= cl[bank] + 1e-6).all())            # the shaped bank lies under the envelope ceiling
        self.assertTrue((fl[bank] <= cl[bank]).all())
        self.assertTrue(np.isinf(cl[s > rw]).all() and np.isinf(fl[s < 0]).all())
        town = s > rw                                                   # freed town ground: the ceiling at any s >= 0
        _, ct = LL.projection_bounds(s, self.L, ENTRY, 2e-4, town=town)
        np.testing.assert_allclose(ct[town], self.L + LL.town_envelope(s[town], ENTRY) + 2e-4, rtol=1e-6)


def _bilinear(a, shape):
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(a, np.float32)).resize((shape[1], shape[0]), Image.Resampling.BILINEAR), np.float64)


class LfShapeTest(unittest.TestCase):
    """v4: the bank and bed evaluated per LF pixel (lf_shape_delta) instead of the bilinear resize of the tile shape."""
    L = 4.8

    def setUp(self):
        self.lake = hex_lake()
        self.s, _ = LL.outline_sdf(self.lake, None, ENTRY, PITCH)
        ty, tx = np.indices(self.lake.shape)
        self.pre = (6.5 + .02 * tx + .01 * ty).astype(np.float32)          # high ground: the bank cap binds at the water
        self.pre[self.lake] = 2.                                            # the coastal falloff's deep basin
        h, w = LL.shaped_ground(self.pre, LL.tile_samples(self.s), self.L, ENTRY)
        self.keep = np.ones(self.lake.shape, bool); self.keep[:, :6] = False  # cells the shape may not write
        self.tsh = np.where((w > 0) & self.keep, h, self.pre).astype(np.float32)
        self.R = lambda a: _bilinear(a, self.s.shape)
        self.d = LL.lf_shape_delta(self.pre, self.tsh, self.keep, self.s, self.L, ENTRY, self.R)
        self.keep_px = np.repeat(np.repeat(self.keep, 4, 0), 4, 1)

    def test_resize_plus_delta_is_the_lf_evaluation(self):
        out = self.R(self.tsh) + self.d
        h, w = LL.shaped_ground(self.R(self.pre), self.s, self.L, ENTRY)
        m = (w > 0) & self.keep_px
        np.testing.assert_allclose(out[m], h[m], atol=1e-5)
        b = ENTRY['bank']
        far = (w == 0) & (self.s > b['ring_width'] + b['blend_width'] + 2)
        self.assertTrue(far.any())
        np.testing.assert_allclose(self.d[far], 0, atol=1e-5)               # nothing changes beyond the shape
        self.assertEqual(self.d.dtype, np.float32)

    def test_bank_is_a_function_of_the_lf_distance(self):
        """Where the bank cap binds (high ground) and in the bed, the LF ground is level + cap(s) / level - depth(s) per
        LF pixel: no tile-grid residue; the bilinear resize of the tile shape leaves one (v3b's tile-grid gradients)."""
        b = ENTRY['bank']
        m = self.keep_px & (np.abs(self.s) <= 1.)
        ideal = np.where(self.s >= 0, self.L + LL.bank_cap(self.s, b), self.L - LL.bed_depth(self.s, ENTRY['bed']))
        lf = self.R(self.tsh) + self.d
        self.assertLess(float(np.abs(lf - ideal)[m].max()), 1e-5)
        self.assertGreater(float(np.abs(self.R(self.tsh) - ideal)[m].max()), .02)


class BedCutTest(unittest.TestCase):
    """v5: the bed under the waterline cut to the designed profile after the projection (cut_bed), land untouched."""
    L, LEVEL, NORM, GAIN = 4.8, 0., .5, 14.3

    def setUp(self):
        self.s, _ = LL.outline_sdf(hex_lake(), None, ENTRY, PITCH)
        rng = np.random.default_rng(5)
        h = self.L + np.where(self.s >= 0, .02 + .2 * self.s, -.3 * np.minimum(-self.s, 1)) + rng.normal(0, .01, self.s.shape)
        self.enc0 = np.rint(np.clip(self.NORM + (h - self.LEVEL) / self.GAIN, 0, 1) * 65535).astype('<u2')
        self.allowed = np.ones(self.s.shape, bool); self.allowed[:, :20] = False
        self.enc = self.enc0.copy()
        self.rep = LL.cut_bed(self.enc, self.s, self.L, ENTRY, self.allowed, self.LEVEL, self.NORM, self.GAIN)
        self.h = self.LEVEL + (self.enc.astype(np.float64) / 65535 - self.NORM) * self.GAIN

    def test_bed_on_the_design(self):
        m = self.allowed & (self.s < 0)
        self.assertTrue(m.any())
        dev = np.abs(self.h - (self.L - LL.bed_depth(self.s, ENTRY['bed'])))[m]
        self.assertLessEqual(float(dev.max()), .5 * self.GAIN / 65535 + 1e-9)
        self.assertEqual(self.rep['pixels'], int(m.sum()))
        self.assertGreater(self.rep['lowered'], 0)                           # the held-up bed comes down to the design

    def test_land_and_disallowed_untouched(self):
        keep = (self.s >= 0) | ~self.allowed
        np.testing.assert_array_equal(self.enc[keep], self.enc0[keep])
        self.assertEqual(self.enc.dtype, self.enc0.dtype)


class EnvelopeTest(unittest.TestCase):
    """The bank envelope (v3): vanilla's low banks near the water, the projection grade far away."""

    def test_vanilla_band(self):
        E = lambda v: float(LL.town_envelope(np.array([v]), ENTRY)[0])
        self.assertTrue(.1 <= E(.3) <= .3 and .1 <= E(.6) <= .3)        # +.1..+.3 within .3-.6 world
        self.assertLessEqual(E(3.), .6)                                   # <= +.6 within 3 world
        self.assertEqual(E(0.), ENTRY['freeboard'])                       # a plot at the water's edge: level + freeboard
        self.assertEqual(E(-.1), -np.inf)

    def test_monotone_and_under_the_grade(self):
        s = np.linspace(0, 20, 20001)
        e = LL.town_envelope(s, ENTRY)
        d = np.diff(e) / np.diff(s)
        self.assertTrue((d >= -1e-9).all())
        self.assertLessEqual(float(d.max()), ENTRY['bank']['far_slope'] + 1e-6)
        far = s > ENTRY['bank']['ring_width'] + ENTRY['bank']['blend_width']
        np.testing.assert_allclose(d[far[1:]], ENTRY['bank']['far_slope'], rtol=1e-6)

    def test_bank_cap_under_envelope(self):
        s = np.linspace(0, ENTRY['bank']['ring_width'], 301)
        self.assertTrue((LL.bank_cap(s, ENTRY['bank']) <= LL.town_envelope(s, ENTRY) + 1e-12).all())

    def test_town_cap_and_plot_distances(self):
        self.assertEqual(LL.town_cap(4.8, None, ENTRY), float('inf'))
        self.assertEqual(LL.town_cap(4.8, float('inf'), ENTRY), float('inf'))
        self.assertAlmostEqual(LL.town_cap(4.8, 2.68, ENTRY), 4.8 + float(LL.town_envelope(np.array([2.68]), ENTRY)[0]))
        s = np.arange(12.).reshape(3, 4); lab = np.array([[0, 1, 1, 0], [2, 2, 0, 0], [0, 0, 0, 1]])
        self.assertEqual(LL.plot_distances(s, lab), {1: 1., 2: 4.})


class ProductTest(unittest.TestCase):
    def test_round_trip_field_and_stamp(self):
        lake = hex_lake()
        s, _ = LL.outline_sdf(lake, None, ENTRY, PITCH)
        ys, xs = np.nonzero(lake)
        with tempfile.TemporaryDirectory() as ws:
            reach = np.zeros((10, 20), bool); reach[2:8, 3:17] = True
            LL.write_product(ws, {'x': {'level': 4.9, 'cells_yx': np.stack([ys, xs], 1).tolist(), 'spec': ENTRY,
                                        'window_tile': [0, lake.shape[0], 0, lake.shape[1]], 'reach_window_tile': [30, 40, 50, 70],
                                        'reach_towns': ['a', 'b']}}, {'map_sha256': 'abc'}, {'x': s}, {'x': reach})
            d = LL.read_product(ws, 'abc')
            f = LL.read_fields(ws, d)
            np.testing.assert_array_equal(f['x'][0], s)
            m, towns = LL.reach_mask(ws, d, (100, 120))
            self.assertEqual(towns, {'x': ['a', 'b']})
            self.assertEqual(int(m.sum()), int(reach.sum()))
            self.assertTrue(m[32, 53] and m[37, 66] and not m[31, 53] and not m[32, 52])
            self.assertEqual(LL.reach_window(m), (32, 38, 53, 67))
            with self.assertRaises(ValueError):
                LL.read_product(ws, 'other map')
            (Path(ws) / LL.FIELD).write_bytes(b'stale')
            with self.assertRaises(ValueError):
                LL.read_fields(ws, d)


class MeshTest(unittest.TestCase):
    """river_water_mesh without a product: the floor is the old scalar and every output is unchanged; with one, the
    level lake's surface stands at its level and its rim lies past mesh_pad."""

    def scene(self, lake_level=None):
        TH, TW = 40, 40
        rng = np.random.default_rng(3)
        lf_c = rng.uniform(.2, 1.5, (TH, TW))
        lake_hex = np.zeros((TH // 2, TW // 2), bool); lake_hex[7:11, 7:11] = True
        river_hex = np.zeros_like(lake_hex); river_hex[9, 2:7] = True
        sea_hex = np.zeros_like(lake_hex)
        q, r = M.cell_hex((TH, TW))
        lk = lake_hex[np.clip(r, 0, 19), np.clip(q, 0, 19)]
        lf_c[lk] = (lake_level if lake_level is not None else M.LAKE_WATER) - .5
        lf_low = lf_c - .05
        level = np.full((TH, TW), np.nan, np.float32)
        rv = river_hex[np.clip(r, 0, 19), np.clip(q, 0, 19)]
        level[rv] = .3
        return level, lf_c, lf_low, river_hex, lake_hex, sea_hex, lk

    def plan(self, floor, pad=None):
        level, lf_c, lf_low, river_hex, lake_hex, sea_hex, _ = self.scene()
        with patch.object(M, 'outer_cells', lambda: np.ones((40, 40), bool)):
            return M.plan(level, lf_c, lf_low, river_hex, lake_hex, sea_hex, cap=True, floor=floor, lake_pad=pad)

    def test_lake_surface_without_product_is_the_constant(self):
        self.assertEqual(M.lake_surface(None, {}), (M.FLOOR, None))
        self.assertEqual(M.lake_surface({'lakes': {}}, {}), (M.FLOOR, None))

    def test_plan_identical_off(self):
        a = self.plan(None)
        b = self.plan(M.FLOOR, None)
        c = self.plan(np.full((40, 40), M.FLOOR), None)
        for k in ('H', 'band', 'wet', 'river_quad', 'lake_quad', 'mouth_quad', 'lake_ground'):
            self.assertTrue(np.array_equal(a[k], b[k]), k)
            self.assertTrue(np.array_equal(a[k], c[k]), k)
        self.assertEqual(a['capped_vertices'], b['capped_vertices'])

    def test_level_lake_mesh_at_its_level_rim_past_pad(self):
        lv = 4.8
        _, _, _, river_hex, lake_hex, sea_hex, lk = self.scene(lv)
        s, _ = LL.outline_sdf(lk, None, ENTRY, PITCH)
        ys, xs = np.nonzero(lk)
        prod = {'lakes': {'x': {'level': lv, 'cells_yx': np.stack([ys, xs], 1).tolist(), 'spec': ENTRY, 'window_tile': [0, 40, 0, 40]}}}
        fields = {'x': (s, (0, 40, 0, 40))}
        F, pad = M.lake_surface(prod, fields, (40, 40))
        sv = LL.vertex_samples(s)
        # ground shaped from the same distance (as polish), read at the vertices
        h, _ = LL.shaped_ground(np.full((160, 160), lv + 1.), s, lv, ENTRY)
        lf_c = h[2::4, 2::4]; lf_low = lf_c - .02
        with patch.object(M, 'outer_cells', lambda: np.ones((40, 40), bool)):
            p = M.plan(np.full((40, 40), np.nan, np.float32), lf_c, lf_low, river_hex & False, lake_hex, sea_hex, cap=True, floor=F, lake_pad=pad)
        v = np.zeros((40, 40), bool)
        for dy in (0, 1):
            for dx in (0, 1):
                v[dy:dy + 39, dx:dx + 39] |= p['lake_quad']
        np.testing.assert_allclose(p['H'][v], lv + M.SURFACE_MARGIN)
        self.assertTrue(v[sv < 0].all())                                   # the surface covers the whole waterline area
        rim = v & (sv >= ENTRY['mesh_pad'])
        self.assertTrue(rim.any())
        self.assertTrue((lf_c[rim] >= lv + ENTRY['freeboard'] - 1e-9).all())
        self.assertTrue((sv[v & ~pad] >= ENTRY['mesh_pad']).all())        # every vertex past the pad is a rim vertex


if __name__ == '__main__':
    unittest.main()

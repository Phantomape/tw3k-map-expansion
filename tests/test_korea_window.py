"""korea_patch.window (P6, 2026-11-01): the patch window and edges from the design's peninsula-class regions
(region-traits province_classes, REGIONS korea_patch offsets), with its guards. Synthetic inputs only.

The environment is set in setUpModule and restored in tearDownModule, and every position is relative to the canvas
campaign_world loaded (a suite may have imported it with another design first), so the module passes alone and in the
full suite."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_ENV, _TMP = {}, None
np = bp = cw = kp = map_hex = K = None


def setUpModule():
    global _TMP, np, bp, cw, kp, map_hex, K
    _TMP = tempfile.TemporaryDirectory(prefix='tw3k-korea-window-tests-')
    t = Path(_TMP.name)
    (t / 'design.json').write_text(json.dumps({'grid': [892 + 80, 702 + 20], 'padding': [20, 10, 60, 10]}), encoding='utf-8')
    for k, v in (('TW3K_DESIGN', str(t / 'design.json')), ('TW3K_WORKSPACE', str(t))):
        _ENV[k] = os.environ.get(k); os.environ[k] = v
    if str(ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT / 'scripts'))
    import numpy as np
    import biome_profiles as bp
    import campaign_world as cw
    import korea_patch as kp
    import map_hex
    K = bp.region('korea_patch')


def tearDownModule():
    for k, v in _ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _TMP.cleanup()


class Window(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp(dir=_TMP.name))
        (self.ws / 'analysis').mkdir()
        # real terrain over the whole tile grid: lat 38 N, lon 127 E (on the peninsula) unless a test moves a town
        self.lat = np.full((cw.TILE_H, cw.TILE_W), 38., np.float32); self.lon = np.full_like(self.lat, 127.)
        # map, relative to the native rectangle: region 'pen_a' (province pen) on 21 x 21 hexes from (Q, R) = native corner +
        # (100, 100); 'xt' (province xt) 30 hexes west / 30 north of it
        self.Q, self.R = cw.PAD_LEFT + 100, cw.PAD_BOTTOM + 100
        Q, R = self.Q, self.R
        ids = np.full((cw.H, cw.W), -1, np.int64)
        ids[R:R + 21, Q:Q + 21] = 0; ids[R + 30:R + 41, Q - 30:Q - 19] = 1
        c = np.zeros((cw.H, cw.W, 16), np.uint8); v = ids + 1
        c[..., 0] = ((v & 31) << 3).astype(np.uint8); c[..., 1] = (v >> 5).astype(np.uint8)
        self.map = {'cells': c, 'land_regions': ['pen_a', 'xt'], 'sea_regions': []}
        self.town, self.xt_town = [Q + 10, R + 10], [Q - 25, R + 35]
        self.plan = {'regions': [{'region': 'pen_a', 'name': 'A', 'town': self.town, 'is_capital': True, 'province_key': 'tw3k_hist_province_pen'},
                                 {'region': 'xt', 'name': 'X', 'town': self.xt_town, 'is_capital': True, 'province_key': 'tw3k_hist_province_xt'}]}
        self.traits = {'province_classes': {'peninsula': ['pen'], 'northeast': ['xt']}}

    def run_(self):
        np.savez(self.ws / 'analysis/west-real-terrain.npz', box=np.array([0, cw.TILE_W, 0, cw.TILE_H]), lat=self.lat, lon=self.lon)
        dp = self.ws / 'plan.json'; dp.write_text(json.dumps(self.plan), encoding='utf-8')
        with mock.patch.object(cw, 'DESIGN_PATH', dp), mock.patch.object(cw, 'region_traits', lambda: self.traits), \
                mock.patch.object(map_hex, 'read', lambda p: self.map):
            return kp.window(self.ws)

    def tile_of(self, town):
        return tuple(int(v / (d / 2)) for v, d in zip(cw.display_position(*town), (cw.DX, cw.DZ)))

    def test_window_from_class_mask(self):
        w = self.run_()
        nq1, nr1 = cw.PAD_LEFT + cw.NATIVE_W, cw.PAD_BOTTOM + cw.NATIVE_H
        self.assertEqual(w['hex_win'], (self.Q + K['win_west'], nq1, self.R + K['win_south'], nr1))
        self.assertEqual(w['edges'], (float(self.Q + K['west_edge']), float(nr1 - K['north_edge_below_top']), float(nq1 - K['east_edge_inside'])))
        self.assertEqual(w['regions'], ['pen_a'])

    def test_empty_class_stops(self):
        self.traits['province_classes']['peninsula'] = []
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.run_()

    def test_xuantu_marked_peninsula_stops(self):
        # Xuantu (Liaodong, towns 42.0-42.9 N) put into the class: its town is off the peninsula
        self.traits['province_classes']['peninsula'] = ['pen', 'xt']
        tx, ty = self.tile_of(self.xt_town)
        self.lat[ty, tx] = 42.4; self.lon[ty, tx] = 123.3
        with self.assertRaisesRegex(ValueError, 'off the peninsula'):
            self.run_()

    def move_town(self, lon, lat):
        tx, ty = self.tile_of(self.town)
        self.lon[ty, tx] = lon; self.lat[ty, tx] = lat

    def test_tumen_south_bank_passes(self):
        # the north-east of the peninsula reaches 42.3-42.9 N along the Tumen: no fixed latitude cap
        self.move_town(129.7, 42.35)   # Hoeryong, south bank
        self.assertEqual(self.run_()['regions'], ['pen_a'])

    def test_north_of_the_rivers_stops(self):
        for lon, lat in ((129.51, 42.89), (130.36, 42.87), (124.39, 40.13)):   # Yanji, Hunchun (Tumen); Dandong city (Yalu)
            self.move_town(lon, lat)
            with self.assertRaisesRegex(ValueError, 'off the peninsula'):
                self.run_()

    def test_lower_yalu_banks(self):
        m = K['peninsula_margin']
        self.assertFalse(kp.on_peninsula(124.39, 40.13 - m))   # Dandong (Liaodong bank)
        for lon, lat in ((124.40, 40.10), (124.53, 40.20), (124.85, 40.35)):   # Sinuiju, Uiju, Lelang resource 1
            self.assertTrue(kp.on_peninsula(lon, lat - m))

    def test_on_peninsula(self):
        self.assertTrue(kp.on_peninsula(126.53, 33.38))     # Jeju
        self.assertTrue(kp.on_peninsula(125.75, 39.02))     # Pyongyang
        self.assertFalse(kp.on_peninsula(123.33, 42.71))    # Xuantu capital (Liaodong)
        self.assertFalse(kp.on_peninsula(131.98, 43.83))    # old North Okjeo pasture (beyond the Tumen)
        np.testing.assert_array_equal(kp.on_peninsula(np.array([126.5, 123.0]), np.array([36., 41.])), [True, False])

    def test_class_region_missing_from_map_stops(self):
        self.map['land_regions'] = ['other', 'xt']
        with self.assertRaisesRegex(ValueError, 'not in the final map'):
            self.run_()

    def test_old_design_calibration(self):
        # 2026-11-01 design: peninsula mask q 1571.., r 825.. -> the old constants (byte-identical patch)
        self.assertEqual((1571 + K['west_edge'], 1571 + K['win_west'], 825 + K['win_south']), (1578, 1560, 812))


if __name__ == '__main__':
    unittest.main()

"""render_gpu: every ground kernel equals the NumPy renderer byte for byte on a synthetic ground (LF raster, coarse
land mesh with holes, framed water window, object overlay). Needs CuPy and the GPU (skipped otherwise); holds the
machine's GPU lock while it runs."""
from pathlib import Path
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
# render_live_view needs the design of the pack it renders: the tracked release design (towns unused here)
os.environ.setdefault('TW3K_DESIGN', str(Path(__file__).resolve().parents[1] / 'work/all-directions-30/work/current-region-design.json'))

try:
    import cupy  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, 'no CuPy')
class GroundKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pipeline_lock as PL
        import render_live_view as L
        import render_live_objects as lo
        import render_gpu
        import campaign_world as cw
        cls.lock = PL.gpu('test_render_gpu', wait=600)
        if cls.lock.__enter__() is None:
            raise unittest.SkipTest('GPU busy')
        rng = np.random.default_rng(7)
        H, W = 400, 520
        yy, xx = np.mgrid[:H, :W]
        lf = (30000 + 9000 * np.sin(xx / 37.) * np.cos(yy / 23.) + rng.integers(0, 400, (H, W))).astype(np.uint16)
        g = L.PackGround.__new__(L.PackGround)
        g.vanilla, g.shift = False, (0., 0.)
        g.lf = lf
        g.world = (W / cw.LF_PX_PER_HEX * cw.DX, H / cw.LF_PX_PER_HEX * cw.DZ)
        x0, z0, step, nx, nz = 2.3, 3.1, .21, 150, 160
        land = (g.heights(x0 + (np.arange(nx) + .5)[None] * step, z0 + (np.arange(nz) + .5)[:, None] * step)
                + rng.normal(0, .05, (nz, nx))).astype(np.float32)
        land[40:60, 30:90] = -np.inf                     # holes: fine tiles
        water = np.full((nz, nx), -np.inf, np.float32)
        water[70:120, 20:110] = land[70:120, 20:110].max() * .6 + 1.2
        water[45:55, 40:50] = 2.0                        # water over a hole
        g.mesh = (x0, z0, step, nx, nz, land)
        sh = L.LiveShader.__new__(L.LiveShader)
        sh.g, sh.meshes = g, True
        sh.mwin = (x0 - 1, z0 - 1, step, nx, nz, water, land)
        ov = lo.Overlay.__new__(lo.Overlay)
        ov.x0, ov.z0, ov.cell, ov.nx, ov.nz = 5., 6., .05, 300, 280
        top = np.full((ov.nz, ov.nx), -np.inf, np.float32)
        for _ in range(60):
            i, j = rng.integers(0, ov.nx - 10), rng.integers(0, ov.nz - 10)
            top[j:j + 8, i:i + 8] = float(g.heights(np.array([ov.x0 + i * ov.cell]), np.array([ov.z0 + j * ov.cell]))[0]) + .4
        ov.top = top
        cls.L, cls.lo, cls.g, cls.sh, cls.ov = L, lo, g, sh, ov
        cls.dev = render_gpu.Device(g)
        cls.dev.sync(sh)
        cls.dev.set_overlay(ov)
        cls.rng = rng
        cls.W, cls.H = g.world

    @classmethod
    def tearDownClass(cls):
        cls.dev.free()
        cls.lock.__exit__(None, None, None)

    def pts(self, n=40000):
        x = self.rng.uniform(-3, self.W + 3, n)
        z = self.rng.uniform(-3, self.H + 3, n)
        return x, z

    def same(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.array_equal(a.view(np.uint64) if a.dtype == np.float64 else a,
                                       b.view(np.uint64) if b.dtype == np.float64 else b), f'{np.sum(a != b)} differ')

    def test_heights_surface_lf(self):
        x, z = self.pts()
        self.same(self.dev.heights(x, z), self.g.heights(x, z))
        self.same(self.dev.surface(x, z), self.sh.surface(x, z))
        self.same(self.dev.lf_heights(x, z), self.L.rv.Ground.heights(self.g, x, z))

    def test_surface_with_objects(self):
        x, z = self.pts()
        cpu = np.maximum(self.sh.surface(x, z), self.ov.height(x, z))
        self.same(self.dev.surface_objects(x, z), cpu)

    def rays(self):
        cam = (self.W / 2, self.H / 2, 8., 30., 6.)
        eye = np.array([cam[0] - 5, 9.5, cam[1] - 6])
        jj, ii = np.mgrid[:90, :160].astype(np.float64)
        f = np.array([cam[0], 2., cam[1]]) - eye
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 1, 0]); right /= np.linalg.norm(right)
        up = np.cross(right, f)
        d = f + ((2 * (ii + .5) / 160 - 1) * .7)[..., None] * right + ((1 - 2 * (jj + .5) / 90) * .4)[..., None] * up
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return eye, d

    def test_march(self):
        eye, d = self.rays()
        GX, GZ = np.meshgrid(np.linspace(0, self.W, 256), np.linspace(0, self.H, 256))
        hh = self.sh.surface(GX, GZ)
        a = self.L.march(self.sh.surface, eye, d, float(hh.min()) - .2, float(hh.max()) + .2)
        b = self.dev.march(eye, d, float(hh.min()) - .2, float(hh.max()) + .2)
        self.assertGreater(a[0].sum(), 1000)
        self.same(b[0], a[0])
        self.same(b[1], a[1])

    def test_march_objects(self):
        eye, d = self.rays()
        GX, GZ = np.meshgrid(np.linspace(0, self.W, 256), np.linspace(0, self.H, 256))
        hh = self.sh.surface(GX, GZ)
        found, t = self.L.march(self.sh.surface, eye, d, float(hh.min()) - .2, float(hh.max()) + .2)
        surface = lambda wx, wz: np.maximum(self.sh.surface(wx, wz), self.ov.height(wx, wz))
        a = self.lo.march_objects(surface, eye, d, found, t, .45)
        b = self.dev.march_objects(eye, d, found, t, .45)
        self.assertGreater(np.sum(a[found] != t[found]), 10)       # some rays meet an object
        self.same(b, a)

    def test_ao_and_shadow(self):
        x, z = self.pts(20000)
        h = self.g.heights(x, z)
        self.same(self.dev.ao(x, z, h, (.6, 2.5)), self.L.terrain_ao(self.g, x, z, h, (.6, 2.5)))
        s = self.L.sun_vector(dict(sun_az=315., sun_alt=40.))
        self.same(self.dev.shadow(x, z, h, s), self.L.sun_shadow(self.sh.surface, x, z, h, s))
        surface = lambda wx, wz: np.maximum(self.sh.surface(wx, wz), self.ov.height(wx, wz))
        self.same(self.dev.shadow(x, z, h, s, objects=True), self.L.sun_shadow(surface, x, z, h, s))
        # float32 heights (object tops: gbuffer passes the overlay's top): h + .02 adds in float32 on both
        h32 = (h + self.rng.uniform(0, .3, h.shape)).astype(np.float32)
        self.same(self.dev.shadow(x, z, h32, s, objects=True), self.L.sun_shadow(surface, x, z, h32, s))

    def test_mesh_raster(self):
        """campaign_water_meshes.raster with GPU_FILL equals the CPU fill (max per cell; ties, shared edges, slivers,
        triangles larger than a band, cells outside the grid)."""
        import campaign_water_meshes as cwm, render_gpu
        rng = np.random.default_rng(11)
        for trial in range(5):
            n = int(rng.integers(200, 6000))
            c = rng.uniform(-20, 320, (n, 1, 3))
            tris = c + rng.normal(0, (.4, 3., 25., 1., 60.)[trial], (n, 3, 3))
            tris[:, :, 1] = rng.uniform(-5, 20, (n, 3)).round(int(rng.integers(0, 3)))
            args = (tris, -7.3, 4.1, .37 + trial * .11, 700 + trial, 650 - trial)
            a = cwm.raster(*args)
            cwm.GPU_FILL = render_gpu.ZFill()
            try:
                b = cwm.raster(*args)
            finally:
                cwm.GPU_FILL = None
            self.assertGreater(np.isfinite(a).sum(), 100)
            self.assertTrue(np.array_equal(a, b), trial)     # == : a 0.0 / -0.0 tie may keep either sign


if __name__ == '__main__':
    unittest.main()

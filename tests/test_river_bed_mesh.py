"""river_bed_mesh: the batched candidate stage (inside_points_many, pick_faces) gives exactly what the per-face
loop gave (the same points in the same order, the same replaced faces) on synthetic lattices and triangles:
small, large, sliver, clockwise, partly or wholly off the lattice.

Baseline (real data, a few minutes, CPU): with TW3K_RBM_BASELINE naming a baseline JSON (the replaced-face sets
and the dry-run report of a known-good version on a frozen workspace copy: keys refine, report, and workspace,
default rbm-ws beside the JSON), the dry run of this version on that copy must give the same replaced faces and
the same output mesh and compressed_map sha256. A change to river_bed_mesh not meant to change its output
passes it first. The 2026-11 baseline: <scratch>/speedup/rbm-old.json on rbm-ws (the v22b inputs).

Importing river_bed_mesh reads the region design through campaign_world, so TW3K_WORKSPACE must name a workspace
root (one holding work/current-region-design.json, e.g. <main checkout>/work/all-directions-30 or rbm-ws), not the
repository root: there campaign_world fails with KeyError 'grid'. In a sparse worktree it must be set."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import river_bed_mesh as R


def per_face(corners, tris, fine, rfine, water, field, ids):
    """The candidate loop as it was before pick_faces."""
    pick = []
    for k, i in enumerate(ids):
        pts = R.inside_points(tris[k], fine, margin=0.)
        if not len(pts):
            continue
        gx = np.rint(pts[:, 0] / R.STEP).astype(int)
        gz = np.rint(pts[:, 1] / R.STEP).astype(int)
        w = water[gz, gx]
        live = np.isfinite(w)
        if not live.any():
            continue
        g = R.lf(field, pts[live, 0], pts[live, 1])
        coarse = R.planar(corners[k], pts[live])
        hidden = (w[live] > g + R.SHOW) & (w[live] < coarse + R.HIDE)
        spilled = (w[live] < g - R.SHOW) & (w[live] > coarse - R.HIDE)
        fx_ = np.rint(pts[live, 0] / R.FSTEP).astype(int)
        fz_ = np.rint(pts[live, 1] / R.FSTEP).astype(int)
        off = rfine[fz_, fx_] & (w[live] > g + R.SHOW) & (np.abs(coarse - g) > R.BED_TOL)
        if (hidden | spilled | off).any():
            pick.append(i)
    return np.array(pick, np.int64)


class Candidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        TH, TW = 90, 120                                   # tile cells
        fine = rng.random((TH * R.SUB + 1, TW * R.SUB + 1)) < .55
        rfine = fine & (rng.random(fine.shape) < .5)
        yy, xx = np.mgrid[:TH * 4, :TW * 4]
        field = (32768 + 600 * np.sin(xx / 13.) * np.cos(yy / 9.) + rng.integers(0, 50, xx.shape)).astype(np.uint16)
        gz, gx = np.mgrid[:TH + 1, :TW + 1] * R.STEP
        water = np.where(rng.random(gx.shape) < .6, R.lf(field, gx, gz) + rng.normal(0, .06, gx.shape), -np.inf)
        n = 3000
        c = rng.uniform(-3 * R.STEP, (TW + 3) * R.STEP, (n, 1, 2))
        size = rng.choice([.05, .4, 1.5, 6.], n)[:, None, None] * R.STEP
        tris = c + rng.normal(0, 1, (n, 3, 2)) * size
        tris[::7, 2] = tris[::7, 0] + (tris[::7, 1] - tris[::7, 0]) * rng.uniform(0, 1, (len(tris[::7]), 1))  # slivers
        y = R.lf(field, tris[..., 0].clip(0, TW * R.STEP), tris[..., 1].clip(0, TH * R.STEP)) + rng.normal(0, .05, (n, 3))
        corners = np.stack([tris[..., 0], y, tris[..., 1]], -1)
        cls.args = (corners, tris, fine, rfine, water, field, np.arange(n, dtype=np.int64) * 3 + 5)

    def test_inside_points_many(self):
        corners, tris, fine = self.args[:3]
        for margin in (0., .2):
            want = [R.inside_points(t, fine, margin=margin) for t in tris]
            for budget in (1, 500, 1 << 24):               # blocks of one face, of a few, of all
                got = [np.zeros((0, 2)) for _ in tris]
                seen = []
                for f, pts in R.inside_points_many(tris, fine, margin=margin, budget=budget):
                    seen.append(f)
                    for k in np.unique(f):
                        got[k] = pts[f == k]
                order = np.concatenate(seen)
                self.assertTrue((np.diff(order) >= 0).all())
                for k in range(len(tris)):
                    self.assertEqual(want[k].reshape(-1, 2).tobytes(), got[k].tobytes(), (margin, budget, k))

    def test_pick_faces(self):
        want = per_face(*self.args)
        got = R.pick_faces(*self.args)
        self.assertGreater(len(want), 100)
        self.assertLess(len(want), len(self.args[-1]))
        self.assertEqual(want.dtype, got.dtype)
        self.assertEqual(want.tobytes(), got.tobytes())


_RUN = '''
import hashlib, json
import river_bed_mesh as R
picked = []
pick = R.pick_faces
def rec(*a):
    r = pick(*a)
    picked.append(hashlib.sha256(r.tobytes()).hexdigest()[:16] + f':{len(r)}')
    return r
R.pick_faces = rec
rep = R.run(False)
for v in rep['variants']:
    v.pop('meshes', None)
print(json.dumps({'refine': picked, 'report': rep}))
'''


@unittest.skipUnless(os.environ.get('TW3K_RBM_BASELINE'), 'TW3K_RBM_BASELINE not set (real-data baseline)')
class Baseline(unittest.TestCase):
    def test_same_as_baseline(self):
        path = Path(os.environ['TW3K_RBM_BASELINE'])
        base = json.loads(path.read_text(encoding='utf-8'))
        env = dict(os.environ, TW3K_WORKSPACE=base.get('workspace', str(path.with_name('rbm-ws'))))
        out = subprocess.run([sys.executable, '-c', _RUN], cwd=Path(R.__file__).parent, env=env,
                             capture_output=True, text=True, check=True).stdout
        got = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(got['refine'], base['refine'])                  # replaced faces, mesh by mesh
        for a, b in zip(got['report']['variants'], base['report']['variants'], strict=True):
            self.assertEqual(a['outputs'], b['outputs'])                 # mesh + compressed_map sha256
        self.assertEqual(got['report'], base['report'])


if __name__ == '__main__':
    unittest.main()

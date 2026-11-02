"""art_review gate logic: check (whole-file sha256, newest finalized wins, only 'pass' exits 0), finalize merge rules,
camera loading."""
import hashlib, json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np

import art_review as ar
import art_review_changes as AC


def write(p, d):
    p.write_text(json.dumps(d), encoding='utf-8')


class Check(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.pack = self.d / 'cand.pack'
        self.pack.write_bytes(b'PFH5' + b'x' * 100000 + b'tail')
        self.sha = hashlib.sha256(self.pack.read_bytes()).hexdigest()
        self.rev = self.d / 'reviews'; self.rev.mkdir()

    def test_whole_file_hash(self):
        self.assertEqual(ar.sha256_file(self.pack, chunk=4096), self.sha)

    def test_no_review_fails(self):
        self.assertEqual(ar.check(self.pack, self.rev)[0], 1)

    def test_only_pass_passes(self):
        for v, code in (('pending-review', 1), ('auto-fail', 1), ('fail', 1), ('pass', 0), ('bogus', 1), (None, 1)):
            write(self.rev / 'a.json', dict(pack_sha256=self.sha, verdict=v, finalized='2026-10-30T20:00:00'))
            self.assertEqual(ar.check(self.pack, self.rev)[0], code, v)

    def test_other_pack_review_does_not_count(self):
        write(self.rev / 'a.json', dict(pack_sha256='0' * 64, verdict='pass', finalized='2026-10-30T20:00:00'))
        self.assertEqual(ar.check(self.pack, self.rev)[0], 1)

    def test_changed_pack_fails(self):
        write(self.rev / 'a.json', dict(pack_sha256=self.sha, verdict='pass', finalized='2026-10-30T20:00:00'))
        with open(self.pack, 'ab') as f:
            f.write(b'!')
        self.assertEqual(ar.check(self.pack, self.rev)[0], 1)

    def test_newest_finalized_wins(self):
        write(self.rev / 'a.json', dict(pack_sha256=self.sha, verdict='pass', finalized='2026-10-30T20:00:00'))
        write(self.rev / 'b.json', dict(pack_sha256=self.sha, verdict='fail', finalized='2026-10-30T21:00:00'))
        self.assertEqual(ar.check(self.pack, self.rev)[0], 1)
        write(self.rev / 'b.json', dict(pack_sha256=self.sha, verdict='fail', finalized=None))
        self.assertEqual(ar.check(self.pack, self.rev)[0], 0)

    def test_snapshot_survives_replacement(self):
        old = ar.SCRATCH
        ar.SCRATCH = self.d / 'scratch'
        try:
            snap, sha = ar.snapshot(self.pack)
            self.assertEqual(sha, self.sha)
            self.pack.rename(self.d / 'renamed.pack'); self.pack.write_bytes(b'new candidate')
            self.assertEqual(ar.sha256_file(snap), self.sha)
            self.assertEqual(ar.snapshot(self.d / 'renamed.pack'), (snap, sha))
        finally:
            ar.SCRATCH = old

    def test_review_files_ignored(self):
        write(self.rev / 'x.review.json', dict(pack_sha256=self.sha, verdict='pass', finalized='2026-10-30T20:00:00'))
        self.assertEqual(ar.check(self.pack, self.rev)[0], 1)

    def test_check_is_read_only(self):
        write(self.rev / 'a.json', dict(pack_sha256=self.sha, verdict='pass', finalized='2026-10-30T20:00:00'))
        before = {p: p.read_bytes() for p in self.d.rglob('*') if p.is_file()}
        ar.check(self.pack, self.rev)
        self.assertEqual(before, {p: p.read_bytes() for p in self.d.rglob('*') if p.is_file()})

    def test_paths_in_lock_refused(self):
        ar.refuse_paths_in_locks([self.pack, None, self.d / 'art-review' / 'map.hex'])
        with self.assertRaises(SystemExit):
            ar.refuse_paths_in_locks([self.pack, self.d / '.pipeline-lock-2' / 'art-review' / 'map.hex'])


class Merge(unittest.TestCase):
    D = dict(verdict='pending-review', cameras=[dict(name='a'), dict(name='b')])

    def test_all_pass(self):
        r = dict(overall='pass', cameras=dict(a=dict(verdict='pass'), b=dict(verdict='pass')))
        self.assertEqual(ar.merge_review(self.D, r)[0], 'pass')

    def test_auto_fail_never_passes(self):
        r = dict(overall='pass', cameras=dict(a=dict(verdict='pass'), b=dict(verdict='pass')))
        self.assertEqual(ar.merge_review(dict(self.D, verdict='auto-fail'), r)[0], 'auto-fail')

    def test_missing_camera_fails(self):
        r = dict(overall='pass', cameras=dict(a=dict(verdict='pass')))
        v, _, missing, _ = ar.merge_review(self.D, r)
        self.assertEqual((v, missing), ('fail', ['b']))

    def test_unchanged_views_need_no_verdict(self):
        d = dict(self.D, cameras=[dict(name='a'), dict(name='b', unchanged=True)])
        r = dict(overall='pass', cameras=dict(a=dict(verdict='pass')))
        self.assertEqual(ar.merge_review(d, r)[0], 'pass')

    def test_failed_camera_or_overall_fails(self):
        self.assertEqual(ar.merge_review(self.D, dict(overall='pass', cameras=dict(a=dict(verdict='pass'), b=dict(verdict='fail'))))[0], 'fail')
        self.assertEqual(ar.merge_review(self.D, dict(overall='fail', cameras=dict(a=dict(verdict='pass'), b=dict(verdict='pass'))))[0], 'fail')


class LastPassed(unittest.TestCase):
    """The base is the newest pack whose reviewer said overall 'pass', found on disk by its sha256."""
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.packs = {}
        for i, (name, created, overall) in enumerate((('p20', '2026-10-30T10:00:00', 'pass'),
                                                       ('p21', '2026-10-31T10:00:00', 'pass'),
                                                       ('p22', '2026-10-31T17:00:00', 'fail'),
                                                       ('p23', '2026-10-31T18:00:00', None))):
            p = self.d / f'{name}.pack'
            p.write_bytes(name.encode() * 1000)
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            self.packs[name] = (p, sha)
            write(self.d / f'{sha[:12]}.json', dict(pack_sha256=sha, created=created, version=name, pack_path=str(p)))
            if overall:
                write(self.d / f'{sha[:12]}.review.json', dict(overall=overall))

    def test_newest_pass_not_newest_pack(self):
        self.assertEqual(ar.last_passed(self.d)['version'], 'p21')

    def test_exclude_candidate(self):
        self.assertEqual(ar.last_passed(self.d, exclude=self.packs['p21'][1])['version'], 'p20')

    def test_locate_renamed_pack(self):
        d = ar.last_passed(self.d)
        p, sha = self.packs['p21']
        moved = p.with_name('cand.previous-123.pack')
        p.rename(moved)
        self.assertEqual(ar.locate_pack(d), moved)
        moved.write_bytes(b'other')
        self.assertIsNone(ar.locate_pack(d))


class RenderDeps(unittest.TestCase):
    """A cached render is reused only while the scripts it ran are unchanged; pipeline scripts it never loaded do not
    void it (v22b: a biome_profiles change voided every base render through a static import closure)."""
    def setUp(self):
        self.out, ar.OUT = ar.OUT, Path(tempfile.mkdtemp())

    def tearDown(self):
        ar.OUT = self.out

    def test_module_deps_are_loaded_scripts_only(self):
        d = ar.module_deps()
        self.assertIn('art_review', d)
        self.assertIn('art_review_changes', d)
        self.assertNotIn('json', d)
        self.assertTrue(ar.deps_ok(d))

    def test_deps_ok(self):
        d = ar.module_deps()
        self.assertFalse(ar.deps_ok({}))
        self.assertFalse(ar.deps_ok(None))
        self.assertFalse(ar.deps_ok(dict(d, art_review='0' * 40)))       # its source changed
        self.assertFalse(ar.deps_ok(dict(d, no_such_module_here='1' * 40)))   # a script it ran is gone

    def cache(self, meta):
        meta = dict(meta)
        meta.setdefault('backend', ar.backend())
        c = dict(name='v', cam=[1, 2, 3, 0, 20])
        png, js = ar.cache_paths('a' * 64, 'k' * 64, c)
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b'png'); write(js, meta)
        return ar.cached('a' * 64, 'k' * 64, c)

    def test_cached_needs_unchanged_deps(self):
        d = ar.module_deps()
        self.assertIsNotNone(self.cache(dict(deps=d)))
        self.assertIsNone(self.cache(dict(deps=dict(d, art_review='0' * 40))))
        self.assertIsNone(self.cache(dict()))                               # renders from before deps were recorded

    def test_changed_render_module_forces_rerender(self):
        """Edit a module a render ran: the cached render is no longer reused (coordinator 2026-10-31)."""
        import importlib
        here, d = ar.HERE, Path(tempfile.mkdtemp())
        mod = d / 'tw3k_probe_render_mod.py'
        mod.write_text('K = 1\n', encoding='utf-8')
        sys.path.insert(0, str(d))
        ar.HERE = d
        try:
            importlib.import_module('tw3k_probe_render_mod')
            deps = ar.module_deps()
            self.assertIn('tw3k_probe_render_mod', deps)
            self.assertIsNotNone(self.cache(dict(deps=deps)))
            mod.write_text('K = 22\n', encoding='utf-8')     # another size: the source memo keys on size and mtime
            self.assertIsNone(self.cache(dict(deps=deps)))
        finally:
            ar.HERE = here
            sys.path.remove(str(d))
            sys.modules.pop('tw3k_probe_render_mod', None)

    def test_changed_data_file_forces_rerender(self):
        """A data file opened during a render (a rule json, say) is a dep: editing it voids the cached render; derived
        caches and packs are not deps."""
        ar.track_opens()
        d = Path(tempfile.mkdtemp())
        rules, cache, pack = d / 'rules.json', d / 'cache' / 'x.npy', d / 'x.pack'
        cache.parent.mkdir()
        for p in (rules, cache, pack):
            p.write_text('1', encoding='utf-8')
            p.read_text(encoding='utf-8')
        deps = ar.module_deps()
        self.assertIn('file:' + str(rules.resolve()), deps)
        self.assertFalse(any(k.endswith(('x.npy', 'x.pack')) for k in deps))
        self.assertIsNotNone(self.cache(dict(deps=deps)))
        rules.write_text('22', encoding='utf-8')
        self.assertIsNone(self.cache(dict(deps=deps)))

    def test_cached_needs_same_backend(self):
        """A render from the other backend (or from before backends were recorded) is not reused: the GPU and the CPU
        differ on real views (v22b dunhuang, one pixel)."""
        d = ar.module_deps()
        other = 'cpu' if ar.backend() == 'gpu' else 'gpu'
        self.assertIsNotNone(self.cache(dict(deps=d)))
        self.assertIsNone(self.cache(dict(deps=d, backend=other)))
        self.assertIsNone(self.cache(dict(deps=d, backend=None)))

    def test_one_backend(self):
        g = lambda b: {'v': ('p', dict(backend=b))}
        self.assertEqual(ar.one_backend(g('gpu'), g('gpu')), 'gpu')
        with self.assertRaises(SystemExit):
            ar.one_backend(g('cpu'), g('gpu'))        # base on the CPU, candidate on the GPU
        with self.assertRaises(SystemExit):
            ar.one_backend(g('cpu'), {'v': ('p', {})})   # one without a recorded backend

    def test_render_reads(self):
        self.assertEqual(ar.render_reads([(None, dict(reads=['a', 'b'])), (None, dict(reads=['b', 'c']))]), {'a', 'b', 'c'})
        self.assertIsNone(ar.render_reads([(None, dict(reads=['a'])), (None, dict())]))


class Changes(unittest.TestCase):
    """art_review_changes: raster block -> world box in the renderer's own frame, multiset row diff, camera test."""
    def ground(self, world=None):
        import render_campaign_view as rv
        g = rv.Ground.__new__(rv.Ground)
        g.vanilla, g.shift, g.world = False, (0., 0.), world
        return g

    def test_lf_box_covers_the_changed_pixel(self):
        import campaign_world as cw
        H, W, per = 40, 50, cw.LF_PX_PER_HEX
        a = np.full((H, W), 30000, np.uint16); b = a.copy(); b[7, 31] = 40000
        (blk,) = AC.array_blocks(a, b, block=8)
        x0, z0, x1, z1 = AC.lf_box(blk, a.shape, per, cw.DX, cw.DZ)
        xs, zs = np.meshgrid(np.linspace(0, W / per * cw.DX, 400), np.linspace(0, H / per * cw.DZ, 400))
        g = self.ground()
        diff = g._bilinear(a, per, (0, 0), xs, zs) != g._bilinear(b, per, (0, 0), xs, zs)
        self.assertTrue(diff.any())
        self.assertTrue(((xs[diff] >= x0) & (xs[diff] <= x1) & (zs[diff] >= z0) & (zs[diff] <= z1)).all())

    def test_blend_box_covers_the_changed_texel(self):
        H, W, world = 40, 50, (100., 80.)
        a = np.zeros((H, W, 2), np.uint8); b = a.copy(); b[12, 20] = 255
        (blk,) = AC.array_blocks(a, b, block=8)
        x0, z0, x1, z1 = AC.blend_box(blk, a.shape, world)
        xs, zs = np.meshgrid(np.linspace(0, world[0], 400), np.linspace(0, world[1], 400))
        g = self.ground(world)
        g.blend = a; ca, _, _ = g.blend_corners(xs.ravel(), zs.ravel())
        g.blend = b; cb, _, _ = g.blend_corners(xs.ravel(), zs.ravel())
        diff = np.any([np.any(p != q, -1) for p, q in zip(ca, cb)], 0)
        self.assertTrue(diff.any())
        x, z = xs.ravel()[diff], zs.ravel()[diff]
        self.assertTrue(((x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)).all())

    def test_same_rasters_no_blocks(self):
        a = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        self.assertEqual(AC.array_blocks(a, a.copy()), [])
        self.assertIsNone(AC.array_blocks(a, a[:32]))

    def test_changed_rows_is_order_free_and_counts_duplicates(self):
        M = np.array([[1., 2.], [3., 4.], [1., 2.]])
        ka = AC.row_keys(M)
        da, db = AC.changed_rows(ka, AC.row_keys(M[[2, 1, 0]]))
        self.assertFalse(da.any() or db.any())
        da, db = AC.changed_rows(ka, AC.row_keys(M[:2]))
        self.assertEqual((int(da.sum()), int(db.sum())), (1, 0))
        da, db = AC.changed_rows(ka, AC.row_keys(np.array([[1., 2.], [3., 4.5], [1., 2.]])))
        self.assertEqual((da.tolist(), db.tolist()), ([False, True, False], [False, True, False]))

    def test_affected(self):
        ch = dict(height=AC.Grid(500, 500), surface=AC.Grid(500, 500), glob=[])
        cam, fp = [105, 110, 5, 0, 4], [[100, 100], [110, 120]]
        self.assertEqual(AC.affected(ch, cam, fp), [])
        ch['surface'].mark(300, 300, 301, 301)
        self.assertEqual(AC.affected(ch, cam, fp), [])
        ch['surface'].mark(118, 118, 119, 119)
        self.assertEqual(AC.affected(ch, cam, fp), ['surface change in the view'])
        ch['surface'] = AC.Grid(500, 500)
        ch['height'].mark(140, 100, 141, 101)          # outside the hits, inside the x +- 6 dist window
        self.assertEqual(AC.affected(ch, cam, fp), ['height change in the view window'])
        ch['glob'] = ['x changed']
        self.assertTrue(AC.affected(ch, cam, fp)[0].startswith('global'))


class Cameras(unittest.TestCase):
    def test_split_snow_extra_only(self):
        d = Path(tempfile.mkdtemp())
        write(d / 'c.json', dict(fixed=[dict(name='a', cam=[1, 2, 3, 0, 2]), dict(name='p', cam=None)],
                                 changes=[dict(name='s', cam=[1, 2, 40, 0, 35], review='snow-mask-crop')]))
        write(d / 'x.json', [dict(name='e', cam=[5, 6, 7, 0, 5])])
        cams, snow = ar.load_cameras(d / 'c.json', d / 'x.json')
        self.assertEqual([c['name'] for c in cams], ['a', 'e'])
        self.assertEqual([c['name'] for c in snow], ['s'])
        self.assertEqual([c['name'] for c in ar.load_cameras(d / 'c.json', d / 'x.json', 'e')[0]], ['e'])

    def test_later_version_change_left_out(self):
        d = Path(tempfile.mkdtemp())
        write(d / 'c.json', dict(fixed=[dict(name='a', cam=[1, 2, 3, 0, 2])],
                                 changes=[dict(name='n', cam=[1, 2, 3, 0, 2], from_version='v22'), dict(name='o', cam=[1, 2, 3, 0, 2], from_version='v21')]))
        self.assertEqual([c['name'] for c in ar.load_cameras(d / 'c.json', version='v21')[0]], ['a', 'o'])
        self.assertEqual([c['name'] for c in ar.load_cameras(d / 'c.json', version='v22')[0]], ['a', 'n', 'o'])
        self.assertEqual([c['name'] for c in ar.load_cameras(d / 'c.json')[0]], ['a', 'n', 'o'])
        self.assertEqual([c['name'] for c in ar.load_cameras(d / 'c.json', version='v21b')[0]], ['a', 'o'])
        self.assertTrue(ar.vnum('v21b') > ar.vnum('v21') and ar.vnum('v22') > ar.vnum('v21b'))

    def test_duplicate_names_refused(self):
        d = Path(tempfile.mkdtemp())
        write(d / 'c.json', dict(fixed=[dict(name='a', cam=[1, 2, 3, 0, 2])], changes=[]))
        write(d / 'x.json', [dict(name='a', cam=[5, 6, 7, 0, 5])])
        with self.assertRaises(SystemExit):
            ar.load_cameras(d / 'c.json', d / 'x.json')

    def test_repo_camera_file_loads(self):
        cams, snow = ar.load_cameras(ar.CAMERAS)
        self.assertGreater(len(cams), 40)
        self.assertTrue(all(len(c['cam']) == 5 for c in cams + snow))


if __name__ == '__main__':
    unittest.main()

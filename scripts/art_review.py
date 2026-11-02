"""Offline art review of a built candidate pack before any live test (analysis/art-review-gate.md; user 2026-10-30).

  python art_review.py run --pack PACK [--base BASE_PACK: default the last passed] --map MAP_HEX [--base-map MAP_HEX]
                           --base-design DESIGN [--design DESIGN: default the live workspace's] [--extra SHOTS.json]
                           [--only a,b] [--cameras analysis/art-review-cameras.json] [--snow-script snow_review.py]
  python art_review.py finalize PACK|SHA12        (merges <sha12>.review.json, the reviewer's per-camera verdicts)
  python art_review.py check PACK                 (exit 0 only if a review of this pack's sha256 says pass; read-only)
  python art_review.py status PACK

run renders every camera of the camera set (fixed + changes, the snow-mask-crop ones go to snow_review instead, plus
--extra) from both packs with the default live renderer (render_live_view: packs only, never a workspace), and writes
output/offline-view/art-review/<sha12>/:
  sbs/<cam>.png       base | this pack (640x360 each)
  squint/<cam>.png    the same blurred and shrunk (squint test, checklist 1)
  heat/<cam>.png      trees over the ground the camera sees: |tree - land mesh| > .25 red (base | this)
  montage.png, montage-squint.png, patchiness/, snow/
and analysis/art-review/<sha12>.json (main checkout): pack_sha256, pack_path, base, cameras, auto metrics, verdict
'auto-fail' (a hard metric failed) or 'pending-review'. The reviewer writes <sha12>.review.json; finalize sets
verdict 'pass' / 'fail' (auto-fail stays) and 'finalized'. art_review is the only writer of <sha12>.json.

Auto metrics (gate section 3) come from the stage invariants on a stage-shaped extract of each pack (LF, land meshes,
tree list) plus a map.hex copy (--map: the playable regions; a copy, never the live file): lf_mesh_diff,
lf_mesh_diff_nonplayable, land_mesh_budget (hard, their INVARIANTS thresholds), tree_mesh_offset (warn: its line, and
worse than the base pack), triangles and files vs the base (warn: > 10 % more triangles needs a live performance look,
gate section 5), patchiness windows lit beyond the stored baseline (warn).

Each side renders with its own region design (--design for the pack, --base-design for the base; the towns are
drawn from it as house boxes), and every render records the town list it drew, checked against that side's design.
Renders are cached per (pack sha256, render_key: town list + backend, camera) and reused while the scripts each
render ran are unchanged (module_deps): a base pack reviewed once renders only the new pack. Of the new pack only the views a change reaches are rendered (art_review_changes: change
grids of the renderer's parsed inputs; coordinator 2026-10-31), plus the SENTINELS; every other view reuses its base
render, is marked 'unchanged' and is left out of the montage and of the reviewer's verdicts. Unchanged sentinels and
AUDIT_EXTRA other unchanged views are rendered anyway and must equal the base render pixel for pixel, or the run
renders every view (warn). --full (release packs) renders and reviews every view. Renders run on 1-4 processes by
free commit (--workers); the patchiness audit runs beside them.
Locks: the whole run holds pipeline slot 3, taken only while slots 1-4 are all free and free commit >= 17 GB; before
every shot it pauses while a pipeline holds lock 1/2, the game runs or free commit < 12 GB.
"""
import tw3k_paths
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN_ROOT = tw3k_paths.MAIN_ROOT
REVIEWS = MAIN_ROOT / 'analysis/art-review'
OUT = MAIN_ROOT / 'output/offline-view/art-review'
SCRATCH = Path(os.environ.get('TW3K_ART_REVIEW_SCRATCH', tw3k_paths.SCRATCH / 'art-review'))
CAMERAS = HERE.parent / 'analysis/art-review-cameras.json'
LIVE_DESIGN = MAIN_ROOT / 'work/all-directions-30/work/current-region-design.json'
SIZE = (640, 360)
THR = .25
TREE_WARN = .01            # tree_mesh_offset line (coordinator 2026-10-30)
PERF_WARN = 1.10           # gate section 5: > 10 % more triangles than the base -> live performance look
MIN_FREE_START, MIN_FREE_SHOT = 17, 12
VERDICTS = ('pending-review', 'auto-fail', 'pass', 'fail')


# ------------------------------------------------------------------ identity
def sha256_file(path, chunk=1 << 24):
    """sha256 of the whole file (not the pack index)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def review_files(root=REVIEWS):
    return sorted(p for p in Path(root).glob('*.json') if not p.name.endswith('.review.json'))


def find_review(sha, root=REVIEWS):
    """The newest review of this sha256 by 'finalized' (unfinalized ones sort first), or None."""
    rows = []
    for p in review_files(root):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if d.get('pack_sha256') == sha:
            rows.append((d.get('finalized') or '', p.name, d))
    return max(rows, key=lambda r: (r[0], r[1]))[2] if rows else None


def last_passed(root=REVIEWS, exclude=None):
    """The auto review (<sha12>.json) of the newest pack whose reviewer file (<sha12>.review.json) says overall 'pass',
    newest by the auto review's 'created'; exclude: a sha256 to skip (the candidate itself). None if there is none.
    The base of a review is the last pack that passed, not the last pack built (coordinator 2026-10-31: v22b was
    compared with v22, which had failed)."""
    rows = []
    for p in Path(root).glob('*.review.json'):
        try:
            r = json.loads(p.read_text(encoding='utf-8'))
            d = json.loads(p.with_name(p.name[:-len('.review.json')] + '.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if r.get('overall') == 'pass' and d.get('pack_sha256') and d['pack_sha256'] != exclude:
            rows.append((d.get('created') or '', p.name, d))
    return max(rows, key=lambda r: (r[0], r[1]))[2] if rows else None


def locate_pack(d):
    """A file holding the pack of auto review d, checked by sha256: its snapshot, its recorded path, or another .pack
    in that path's folder (packaging renames the previous candidate to *.previous-*.pack). None if none matches."""
    sha, seen = d['pack_sha256'], set()
    cands = [d.get('snapshot'), d.get('pack_path')]
    if d.get('pack_path'):
        cands += sorted(Path(d['pack_path']).parent.glob('*.pack'), key=lambda p: -p.stat().st_mtime)
    for c in cands:
        if not c:
            continue
        c = Path(c)
        if c in seen or not c.is_file():
            continue
        seen.add(c)
        if sha256_file(c) == sha:
            return c
    return None


def pick_base(a, sha):
    """--base, else the last pack that passed review (last_passed + locate_pack)."""
    if a.base:
        return Path(a.base).resolve()
    d = last_passed(exclude=sha)
    if d is None:
        raise SystemExit('no passed art review to compare with: give --base')
    p = locate_pack(d)
    if p is None:
        raise SystemExit(f'the last passed pack ({d["pack_sha256"][:12]}, {d.get("version")}) is not on disk: give --base')
    print(f'base: last passed review {d["pack_sha256"][:12]} ({d.get("version")}, created {d.get("created")}) -> {p}',
          flush=True)
    return p


def check(pack, root=REVIEWS):
    """(exit code, message): 0 only for a review of this exact file whose verdict is 'pass'. Reads only."""
    sha = sha256_file(pack)
    d = find_review(sha, root)
    if d is None:
        return 1, f'no art review for {Path(pack).name} (sha256 {sha})'
    v = d.get('verdict')
    if v not in VERDICTS:
        return 1, f'unknown verdict {v!r} for sha256 {sha}'
    if v != 'pass':
        return 1, f'art review verdict {v} for sha256 {sha}'
    return 0, f'art review pass for sha256 {sha} (finalized {d.get("finalized")})'


def merge_review(d, review):
    """Verdict from the auto result d and the reviewer's file: auto-fail stays; pass needs overall 'pass' and a 'pass'
    for every reviewed camera of the run (a camera not reviewed cannot pass: gate section 4). A view the run marked
    'unchanged' (its base render reused: no input change reaches it) needs no verdict."""
    cams = [c['name'] for c in d.get('cameras', []) if not c.get('unchanged')]
    got = review.get('cameras', {})
    missing = [c for c in cams if c not in got]
    failed = [c for c in cams if c in got and got[c].get('verdict') != 'pass']
    if d.get('verdict') == 'auto-fail':
        verdict, why = 'auto-fail', 'a hard automatic metric failed'
    elif review.get('overall') != 'pass':
        verdict, why = 'fail', f'reviewer overall {review.get("overall")!r}'
    elif missing or failed:
        verdict, why = 'fail', f'cameras not reviewed {missing}, not passed {failed}'
    else:
        verdict, why = 'pass', 'auto metrics and every camera passed'
    return verdict, why, missing, failed


# ------------------------------------------------------------------ locks
def free_commit_gb():
    out = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB'],
                         capture_output=True, text=True).stdout
    return float(out.strip().replace(',', '.'))


def game_running():
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Three_Kingdoms.exe'], capture_output=True, text=True).stdout
    return 'Three_Kingdoms.exe' in out


def refuse_paths_in_locks(paths):
    """A review whose own input or output sits inside a lock folder waits on itself (C2 2026-11-01: the driver
    copied the map into .pipeline-lock-2, re-creating it, and the review waited 35 min on that slot)."""
    bad = [str(p) for p in paths if p and any(x.lower().startswith(('.pipeline-lock', '.gpu-lock'))
                                              for x in Path(p).resolve().parts)]
    if bad:
        raise SystemExit(f'paths inside a lock folder (this run would wait on its own lock): {bad}')


def take_slot3(PL, who):
    while True:
        busy = PL.busy(PL.WORK)
        why = 'game running' if game_running() else f'slots busy {busy}' if busy else None
        if why is None:
            f = free_commit_gb()
            if f >= MIN_FREE_START:
                tok = PL.acquire(3, who, PL.WORK)
                if tok:
                    print(time.strftime('%H:%M:%S'), 'took', PL.path(3, PL.WORK), flush=True)
                    return tok
                why = 'slot 3 lost'
            else:
                why = f'free commit {f:.1f} GB'
        print(time.strftime('%H:%M:%S'), why, '; waiting', flush=True)
        time.sleep(120)


def pause(PL):
    while True:
        busy = PL.busy(PL.WORK, PL.PIPELINE)
        why = 'game running' if game_running() else f'pipeline locks {busy}' if busy else None
        if why is None:
            f = free_commit_gb()
            if f >= MIN_FREE_SHOT:
                return
            why = f'free commit {f:.1f} GB'
        print(time.strftime('%H:%M:%S'), why, '; paused', flush=True)
        time.sleep(120)


# ------------------------------------------------------------------ cameras
def vnum(v):
    """'v21' -> (21, ''), 'v21b' -> (21, 'b'): a lettered rebuild sorts after its number."""
    import re
    m = re.fullmatch(r'[vV]?(\d+)([a-z]*)', str(v))
    if not m:
        raise SystemExit(f'bad version {v!r}')
    return int(m.group(1)), m.group(2)


def load_cameras(path, extra=None, only=None, version=None):
    """(cameras, snow-crop cameras); a change camera with 'from_version' later than version (e.g. v22 for a v21
    review) is left out."""
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    cams, snow = [], []
    for c in d.get('fixed', []) + d.get('changes', []):
        if c.get('cam') is None:
            continue
        if version and c.get('from_version') and vnum(c['from_version']) > vnum(version):
            continue
        (snow if c.get('review') == 'snow-mask-crop' else cams).append(c)
    for c in (json.loads(Path(extra).read_text(encoding='utf-8')) if extra else []):
        cams.append(dict(c, sets=c.get('sets', []) + ['extra']))
    if only:
        keep = set(only.split(','))
        cams = [c for c in cams if c['name'] in keep]
    names = [c['name'] for c in cams]
    if len(names) != len(set(names)):
        raise SystemExit(f'duplicate camera names: {sorted({n for n in names if names.count(n) > 1})}')
    return cams, snow


def cam_key(cam):
    return hashlib.sha256(json.dumps([round(float(v), 4) if not isinstance(v, str) else v for v in cam]).encode()).hexdigest()[:8]


_SRC = {}


def source_sha(path):
    """sha1 of a scripts-folder module's source ('' when the file is gone), memoised by size and mtime."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return ''
    k = (str(path), st.st_size, st.st_mtime_ns)
    if k not in _SRC:
        _SRC[k] = hashlib.sha1(path.read_bytes()).hexdigest()
    return _SRC[k]


_OPENED = set()     # data files opened in this process since track_opens (audit hook)
_SKIP_SUFFIX = {'.py', '.pyc', '.pyd', '.dll', '.pack'}   # code (module_deps), packs (sha256 + reads)


def track_opens():
    """Record every file this process opens from now on (once per process; render workers call it first thing)."""
    if not _OPENED:
        _OPENED.add(None)
        sys.addaudithook(lambda ev, args: _OPENED.add(args[0]) if ev == 'open' and args and isinstance(args[0], str)
                         else None)


def data_file(p):
    """An opened file that is a render input outside the packs: not code or a pack, not a derived cache (their names
    carry their own keys), not a render output, lock or the Python install."""
    q = Path(p)
    if q.suffix.lower() in _SKIP_SUFFIX:
        return False
    parts = {x.lower() for x in q.parts}
    if 'cache' in parts or '__pycache__' in parts or any(x.startswith(('.pipeline-lock', '.gpu-lock')) for x in parts):
        return False
    try:
        r = q.resolve()
        if not r.is_file():
            return False
    except OSError:
        return False
    for root in (OUT, SCRATCH, Path(sys.prefix), Path(sys.base_prefix)):
        if r.is_relative_to(root.resolve()):
            return False
    return True


def module_deps():
    """{module: source sha1} of every scripts-folder module loaded in this process, plus {'file:<path>': sha1} of
    every data file it opened since track_opens (data_file): the code and the inputs outside the packs a render ran
    with. A module a render executes is imported and a file it reads is opened, so the set may be larger than
    needed, never smaller (integrator 2026-10-31: a rule json read by the renderer must void its renders too)."""
    here = HERE.resolve()
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and f.endswith('.py') and Path(f).resolve().parent == here:
            out[Path(f).stem] = source_sha(f)
    for p in sorted(x for x in _OPENED if x):
        if data_file(p):
            out['file:' + str(Path(p).resolve())] = source_sha(p)
    return out


def deps_ok(deps):
    """True when every module and data file of a recorded deps dict still has the same content."""
    return bool(deps) and all(source_sha(n[5:] if n.startswith('file:') else HERE / f'{n}.py') == h
                              for n, h in deps.items())


def render_reads(renders):
    """Pack keys the renderer read for these (png, meta) renders (each records its own; a worker's first render
    carries the pack loading), or None when one has no record."""
    out = set()
    for _, meta in renders:
        if 'reads' not in meta:
            return None
        out.update(meta['reads'])
    return out


def towns_sha(towns):
    return hashlib.sha256(json.dumps(towns).encode()).hexdigest()


def side_towns(design):
    """(towns, towns_sha) of a side's region design; SystemExit when it is missing."""
    import render_campaign_view as rv
    if not design or not Path(design).is_file():
        raise SystemExit(f'region design {design} not found: every side renders with its own design')
    t = rv.load_towns(design)
    return t, towns_sha(t)


def render_key(towns):
    """Cache key of a render beyond the pack: the renderer code and its one input outside the packs, the town list
    of this side's region design (render_live_objects draws town boxes from it). A cached base render must equal what the
    base pack would render now, or reusing it for an unchanged view would be wrong. The renderer code is not in the
    key: each cached render records the scripts it ran (module_deps) and is reused only while they are unchanged
    (2026-10-31: a static import closure pulled in 57 modules, pipeline scripts among them, and a biome_profiles
    change voided the v22 base renders for v22b)."""
    h = hashlib.sha256(b'render key 3')
    h.update(json.dumps(towns).encode())
    # the backend and the NumPy that ran it: the GPU kernels equal NumPy bit for bit on the synthetic ground
    # (tests/test_render_gpu.py) but not on every real view (2026-11-01: v22b dunhuang, one pixel off by 1 on main),
    # so base and candidate must come from one backend (one_backend)
    h.update(f'{backend()} numpy {np.__version__}'.encode())
    return h.hexdigest()


def backend():
    """'gpu' when TW3K_RENDER_BACKEND says so or (unset) CuPy imports; else 'cpu'."""
    b = os.environ.get('TW3K_RENDER_BACKEND')
    if b:
        return b
    try:
        import cupy
        return 'gpu' if cupy else 'cpu'
    except ImportError:
        return 'cpu'


# ------------------------------------------------------------------ render
def footprint(L, live, cam, P, size=(80, 45)):
    """Ground points the camera sees (an 80x45 ray grid): the heatmap window."""
    sh, g = live['shader'], live['ground']
    eye, d, _, _ = L.camera_rays(g, cam, size, P)
    x, z, dist = cam[:3]
    reach = dist * 6
    GX, GZ = np.meshgrid(np.linspace(x - reach, x + reach, 128), np.linspace(z - reach, z + reach, 128))
    hh = sh.surface(GX, GZ)
    found, t = L.march(sh.surface, eye, d, float(hh.min()) - .2, float(hh.max()) + .2)
    px, pz = eye[0] + d[..., 0] * t, eye[2] + d[..., 2] * t
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(px[found], pz[found])]


def cache_paths(sha, rkey, c):
    d = OUT / 'cache' / f'{sha[:12]}-{rkey[:12]}'
    opts = c.get('opts') or {}
    cam = [float(v) for v in c['cam']]
    key = cam_key(cam + [json.dumps(opts, sort_keys=True)]) if opts else cam_key(cam)
    return d / f'{c["name"]}-{key}.png', d / f'{c["name"]}-{key}.json'


def cached(sha, rkey, c):
    """(png, meta) of a cached render whose recorded scripts are unchanged, else None."""
    png, js = cache_paths(sha, rkey, c)
    if not (png.is_file() and js.is_file()):
        return None
    meta = json.loads(js.read_text(encoding='utf-8'))
    return (png, meta) if deps_ok(meta.get('deps')) and meta.get('backend') == backend() else None


def one_backend(*groups):
    """The one backend every render of the groups ({name: (png, meta)}) ran on; SystemExit when they differ (a
    base/candidate pair from two backends would show the backends' difference as a change)."""
    seen = {m.get('backend') for g in groups for _, m in g.values()}
    if len(seen) != 1 or None in seen:
        raise SystemExit(f'renders from different backends {sorted(map(str, seen))}: base and candidate must use one')
    return seen.pop()


_W = {}   # worker state: pack, renderers by options


def record_towns():
    """Wrap render_campaign_view.load_towns in this process so every town list a renderer loads is recorded in
    _W['towns'] ({towns_sha: count})."""
    import render_campaign_view as rv
    if getattr(rv.load_towns, 'recording', False):
        return
    inner = rv.load_towns

    def load_towns(design=None):
        t = inner(design)
        _W.setdefault('towns', {})[towns_sha(t)] = len(t)
        return t
    load_towns.recording = True
    rv.load_towns = load_towns


def drawn_towns(design, expect):
    """meta['towns'] of a render: the side's design and the town lists the renderers of this worker loaded. A list
    other than this side's (expect: towns_sha of its design) is an error: the render drew another design's towns."""
    import campaign_world as cw
    if Path(cw.DESIGN_PATH).resolve() != Path(design).resolve():
        raise RuntimeError(f'render process reads design {cw.DESIGN_PATH}, not this side\'s {design}')
    got = dict(_W.get('towns', {}))
    bad = sorted(set(got) - {expect})
    if bad:
        raise RuntimeError(f'renderer drew town list {bad[0][:12]} ({got[bad[0]]} towns), not {expect[:12]} of {design}')
    return dict(design=str(design), design_sha256=sha256_file(design), towns_sha256=expect, count=got.get(expect))


def render_one(pack, sha, rkey, c, design, expect):
    """Render one camera into the cache (worker process: the pack is loaded once per worker and reused while the next
    camera is from the same pack). design: this side's region design (the worker's TW3K_DESIGN); expect: towns_sha of
    it. Returns (name, png, meta, pack keys read)."""
    track_opens()
    record_towns()
    import render_live_view as L
    import pipeline_lock as PL
    import art_review_changes as AC
    pause(PL)
    if _W.get('pack') != pack:
        L._STATE.clear()
        _W.clear(); _W.update(pack=pack, renderers={})
    cam = tuple(float(v) for v in c['cam'])
    opts = c.get('opts') or {}
    ok = json.dumps(opts, sort_keys=True)
    with AC.Reads() as r:
        if ok not in _W['renderers']:
            _W['renderers'][ok] = L.make_renderer('live', pack, **opts)
        P = dict(L.LIVE_DEFAULTS, **opts)
        live = L.live_state(pack, P)
        L.BACKEND = backend()
        t = time.time()
        if L.BACKEND == 'gpu':
            with PL.gpu(f'art_review render {sha[:12]} {c["name"]}') as tok:   # one shot at a time on the card
                assert tok
                im = _W['renderers'][ok](cam, SIZE)
        else:
            im = _W['renderers'][ok](cam, SIZE)
        secs = round(time.time() - t, 1)
    fp = footprint(L, live, cam, P)
    meta = dict(name=c['name'], cam=list(cam), opts=opts, seconds=secs, pack_sha256=sha, renderer=rkey, backend=L.BACKEND,
                footprint=fp, deps=module_deps(), reads=sorted(r.keys), towns=drawn_towns(design, expect))
    png, js = cache_paths(sha, rkey, c)
    png.parent.mkdir(parents=True, exist_ok=True)
    im.convert('RGB').save(png)
    js.write_text(json.dumps(meta), encoding='utf-8')
    return c['name'], str(png), meta, sorted(r.keys)


def workers_for(n_jobs, want=None):
    """Render processes: --workers, else 1 per 5 GB of free commit above the per-shot floor (a live renderer peaks at
    about 4.8 GB), at most 4 and at most the number of shots."""
    if want:
        return max(1, min(want, n_jobs))
    return max(1, min(4, n_jobs, int((free_commit_gb() - MIN_FREE_SHOT) // 5)))


def render_jobs(jobs, rkey, workers, design, expect):
    """Render (pack, sha, camera) jobs not yet cached, on `workers` processes with TW3K_DESIGN = design (this side's
    region design; expect: towns_sha of it); returns {(sha, name): (png, meta)} (each meta records the scripts the
    render ran, the pack keys it read and the towns it drew). Every render, cached or new, must have drawn expect."""
    out, todo = {}, []
    for pack, sha, c in jobs:
        hit = cached(sha, rkey, c)
        if hit:
            out[(sha, c['name'])] = hit
        else:
            todo.append((pack, sha, c))
    if todo:
        out.update(_render_todo(todo, rkey, workers, design, expect))
    check_towns(out, expect)
    return out


def check_towns(renders, expect):
    """SystemExit unless every render ({key: (png, meta)}) drew the town list expect."""
    bad = sorted(k[1] for k, (_, m) in renders.items() if (m.get('towns') or {}).get('towns_sha256') != expect)
    if bad:
        raise SystemExit(f'renders {bad[:5]} drew another design\'s towns (or no record): expected {expect[:12]}')


def _render_todo(todo, rkey, workers, design, expect):
    out = {}
    n = workers_for(len(todo), workers)
    print(time.strftime('%H:%M:%S'), f'rendering {len(todo)} shots on {n} process(es)', flush=True)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    # spawned workers even for one process: a render's module_deps are then the scripts it ran, not everything this
    # process loaded for the metrics; one pack at a time, so each worker loads each pack once
    # the workers inherit TW3K_DESIGN (campaign_world reads it at import; render_live_view only sets a default)
    old = os.environ.get('TW3K_DESIGN')
    os.environ['TW3K_DESIGN'] = str(Path(design).resolve())
    try:
        with ProcessPoolExecutor(n, mp_context=mp.get_context('spawn')) as ex:
            for sha in dict.fromkeys(s for _, s, _ in todo):
                futs = [(s, ex.submit(render_one, str(p), s, rkey, c, str(Path(design).resolve()), expect))
                        for p, s, c in todo if s == sha]
                for s, f in futs:
                    name, png, meta, keys = f.result()
                    out[(s, name)] = (Path(png), meta)
                    print(f'{s[:12]} {name:28s} {meta["seconds"]:6.1f} s  {meta["towns"]["count"]} towns', flush=True)
    finally:
        if old is None:
            os.environ.pop('TW3K_DESIGN', None)
        else:
            os.environ['TW3K_DESIGN'] = old
    return out


# ------------------------------------------------------------------ which views changed
# always rendered (coordinator 2026-10-31: changed views + a small sentinel set; the full set before a release pack);
# a sentinel the change grids call unchanged is also compared with its base render pixel for pixel (audit)
SENTINELS = ('full-map-far', 'border-north', 'border-south', 'border-west', 'town-dunhuang', 'town-wusun', 'town-funan',
             'korea-overview')
AUDIT_EXTRA = 2         # unchanged views rendered anyway and compared with the base, besides unchanged sentinels


def town_changes(tb, tc):
    """[(x, z)] of the towns the two sides draw differently: added, removed, moved or capital changed."""
    b, c = {t[0]: t[1:] for t in tb}, {t[0]: t[1:] for t in tc}
    return [(v[0], v[1]) for side, other in ((b, c), (c, b)) for k, v in side.items() if other.get(k) != v]


def town_affected(changes, footprint, margin=1.5):
    """Why a view must render because of town changes ([] = none in its footprint box, grown by the town radius)."""
    if not changes or not footprint:
        return []
    f = np.asarray(footprint, float)
    x0, z0 = f.min(0) - margin
    x1, z1 = f.max(0) + margin
    n = sum(1 for x, z in changes if x0 <= x <= x1 and z0 <= z <= z1)
    return [f'{n} town(s) drawn differently in the view'] if n else []


def plan_renders(L, base, pack, cams, rb, a, towns_changed=()):
    """{'render': names to render from the pack, 'unchanged': names whose base render is reused, 'audit': unchanged
    names rendered anyway to check, 'why': {name: reasons}, 'changes': change summary}. --full renders every camera."""
    import art_review_changes as AC
    names = [c['name'] for c in cams]
    if a.full:
        return dict(render=names, unchanged=[], audit=[], why={n: ['--full'] for n in names}, changes=None, tier='full')
    P = dict(L.LIVE_DEFAULTS)
    ch = AC.compare(L, base, pack, P, render_reads(rb.values()))
    why, unchanged = {}, []
    for c in cams:
        w = AC.affected(ch, [float(v) for v in c['cam']], rb[c['name']][1]['footprint']) if not c.get('opts') else ['own options']
        w = w + town_affected(towns_changed, rb[c['name']][1]['footprint'])
        if w:
            why[c['name']] = w
        else:
            unchanged.append(c['name'])
    sent = [n for n in SENTINELS if n in names]
    picks = sorted((n for n in unchanged if n not in sent), key=lambda n: hashlib.sha256((n + str(pack)).encode()).hexdigest())[:AUDIT_EXTRA]
    aud = [n for n in unchanged if n in sent or n in picks]
    render = [n for n in names if n in why or n in sent or n in aud]
    summary = dict(glob=ch['glob'], counts=ch['counts'], height_boxes=ch['height'].boxes()[:20], surface_boxes=ch['surface'].boxes()[:20])
    print(f'changes: {summary["counts"]}; global {ch["glob"] or "none"}', flush=True)
    print(f'views: {len(why)} changed, {len(unchanged)} unchanged ({len(aud)} audited), {len(sent)} sentinels', flush=True)
    return dict(render=render, unchanged=[n for n in unchanged if n not in render], audit=aud, why=why, changes=summary, tier='changed')


def audit(rb, rc, names):
    """Views judged unchanged whose new render differs from the base render (any pixel)."""
    from PIL import Image
    bad = []
    for n in names:
        x = np.asarray(Image.open(rb[n][0]).convert('RGB'), np.int16)
        y = np.asarray(Image.open(rc[n][0]).convert('RGB'), np.int16)
        if x.shape != y.shape or np.any(x != y):
            bad.append(dict(name=n, pixels=int((np.abs(x - y).max(2) > 0).sum()) if x.shape == y.shape else -1))
    return bad


# ------------------------------------------------------------------ metrics
def extract(L, pack, sha, map_hex):
    """Stage-shaped extract of the pack (LF, land meshes, tree list) + the map copy, kept per sha for reuse."""
    import stage_invariants as si
    ws = SCRATCH / 'packs' / sha[:12]
    done = ws / 'extracted.json'
    if not done.is_file():
        g = L.PackGround(pack)
        pre = L.PRE
        keys = [pre + 'lf_height_map.dds', g.trees_key] + list(g.packs.keys(pre + 'global_meshes/land_mesh_', '.rigid_model_v2'))
        for k in keys:
            p = ws / 'work/release-input' / k
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(g.read(k))
        done.write_text(json.dumps(dict(pack=str(pack), sha256=sha, files=len(keys))), encoding='utf-8')
    m = ws / si.MAP
    m.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(map_hex, m)
    return ws


def metrics(L, pack, sha, map_hex):
    import stage_invariants as si
    ws = extract(L, pack, sha, map_hex)
    ctx = si.Ctx(ws)
    res = {}
    for name in ('lf_mesh_diff', 'lf_mesh_diff_nonplayable', 'land_mesh_budget', 'tree_mesh_offset'):
        fn, _, _, _, thr, _, warn_only, _ = si.INVARIANTS[name]
        r = fn(ctx)
        res[name] = dict(value=r['value'], threshold=thr, hard=not warn_only)
    lm = si._lf_mesh(ctx)
    res['meshes'] = lm['meshes']
    res['trees'] = lm.get('trees')
    tm = ctx._c.get('tree_mesh')
    trees = None
    if tm is not None:
        trees = dict(x=tm['xyz'][:, 0], z=tm['xyz'][:, 2], off=tm['xyz'][:, 1] - tm['mesh'])
    return res, trees


def patchiness_start(pack, sha, out):
    """audit_ground_patchiness on the pack, in the background (it runs beside the renders): 2 strip processes when
    free commit allows (about 5 GB each), strips cached by their input bytes (ground_patchiness.analyse_map)."""
    w = 2 if free_commit_gb() >= MIN_FREE_START + 15 else 1
    return subprocess.Popen([sys.executable, '-B', str(HERE / 'audit_ground_patchiness.py'), 'run', str(pack), '--label', f'art-{sha[:12]}',
                             '--out', str(out / 'patchiness'), '--workers', str(w), '--cache', str(SCRATCH / 'patchiness-strips')])


def patchiness(proc, out):
    """Windows lit (metric > its alarm) that the stored baseline does not list."""
    d = out / 'patchiness'
    code = proc.wait()
    strips = sorted((SCRATCH / 'patchiness-strips').glob('strip-*.pkl'), key=lambda p: p.stat().st_mtime)
    for p in strips[:-48]:       # about three packs' strips stay (some 50 MB each)
        p.unlink(missing_ok=True)
    if code != 0:
        return dict(error=f'audit_ground_patchiness exit {code}')
    run = json.loads((d / 'patchiness.json').read_text(encoding='utf-8'))
    base = json.loads((HERE.parent / 'analysis/ground-patchiness-baseline.json').read_text(encoding='utf-8'))
    alarm = run['thresholds']['alarm']
    lit, new = {}, {}
    for w, v in run['windows'].items():
        on = sorted(k for k, a in alarm.items() if (v['metrics'].get(k) or 0) > a)
        if on:
            lit[w] = on
        extra = sorted(set(on) - set(base.get('windows', {}).get(w, {}).get('lit') or []))
        if extra:
            new[w] = extra
    return dict(lit=lit, new_vs_baseline=new)


# ------------------------------------------------------------------ images
def label(im, text):
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    for i, line in enumerate(text.split('\n')):
        d.text((6, 4 + 12 * i), line, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return im


def squint(im):
    """Blurred and shrunk to a quarter, shown back at half size (squint test)."""
    from PIL import Image, ImageFilter
    w, h = im.size
    return im.filter(ImageFilter.GaussianBlur(4)).resize((w // 4, h // 4), Image.BOX).resize((w // 2, h // 2), Image.NEAREST)


def heat(fp, trees, cam, text, size=420):
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull, Delaunay
    im = Image.new('RGB', (size, size), (255, 255, 255))
    pts = np.asarray(fp, float)
    if trees is None or len(pts) < 3:
        return label(im, text + ': no trees / ground hits'), None
    poly = pts[ConvexHull(pts).vertices]
    (x0, z0), (x1, z1) = poly.min(0), poly.max(0)
    span = max(x1 - x0, z1 - z0) * 1.04
    bx0, bz0 = (x0 + x1 - span) / 2, (z0 + z1 - span) / 2
    tx, tz, off = trees['x'], trees['z'], trees['off']
    idx = np.flatnonzero((tx >= x0) & (tx <= x1) & (tz >= z0) & (tz <= z1))
    idx = idx[Delaunay(poly).find_simplex(np.c_[tx[idx], tz[idx]]) >= 0]
    o = off[idx]
    ok = np.isfinite(o)
    a = np.abs(o[ok])
    st = dict(trees=int(len(idx)), over=int((a > THR).sum()), floating=int((o[ok] > THR).sum()), buried=int((o[ok] < -THR).sum()))
    px = lambda x, z: ((x - bx0) / span * size, size - (z - bz0) / span * size)
    d = ImageDraw.Draw(im)
    d.polygon([px(*p) for p in poly], outline=(120, 120, 120), fill=(242, 242, 242))
    r = 1 if len(idx) > 20000 else 2
    for k in np.argsort(np.where(ok, np.abs(o), -1)):
        x, y = px(tx[idx[k]], tz[idx[k]])
        v = o[k]
        c = (190, 190, 190) if not np.isfinite(v) else (220, 0, 0) if abs(v) > THR else (int(40 + 180 * abs(v) / THR), 140, 40)
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)
    x, y = px(cam[0], cam[1])
    d.line((x - 6, y, x + 6, y), fill=(0, 0, 255), width=2); d.line((x, y - 6, x, y + 6), fill=(0, 0, 255), width=2)
    return label(im, f'{text}  trees {st["trees"]}  >0.25: {st["over"]} (float {st["floating"]}, bury {st["buried"]})'), st


def pair(a, b, gap=4):
    from PIL import Image
    im = Image.new('RGB', (a.width + b.width + gap, max(a.height, b.height)), (0, 0, 0))
    im.paste(a, (0, 0)); im.paste(b, (a.width + gap, 0))
    return im


def montage(tiles, title, cols=3):
    from PIL import Image, ImageDraw
    w, h = tiles[0].size
    m = Image.new('RGB', (cols * (w + 4), ((len(tiles) + cols - 1) // cols) * (h + 4) + 24), (30, 30, 30))
    ImageDraw.Draw(m).text((6, 6), title, fill=(255, 255, 255))
    for i, t in enumerate(tiles):
        m.paste(t, ((i % cols) * (w + 4), 24 + (i // cols) * (h + 4)))
    return m


# ------------------------------------------------------------------ commands
def judge(cur, base, patch):
    hard, warn = [], []
    for name in ('lf_mesh_diff', 'lf_mesh_diff_nonplayable', 'land_mesh_budget'):
        m = cur[name]
        if m['threshold'] is not None and m['value'] is not None and m['value'] > m['threshold']:
            (hard if m['hard'] else warn).append(f'{name} {m["value"]} > {m["threshold"]}')
    t, tb = cur['tree_mesh_offset']['value'], base['tree_mesh_offset']['value']
    if t is not None and t > TREE_WARN:
        warn.append(f'tree_mesh_offset {t} > {TREE_WARN}')
    if t is not None and tb is not None and t > tb:
        warn.append(f'tree_mesh_offset {t} worse than the base {tb}')
    tri, trib = cur['meshes']['triangles'], base['meshes']['triangles']
    if tri > trib * PERF_WARN:
        warn.append(f'land-mesh triangles {tri} > {PERF_WARN:.2f} x base {trib}: look at frame rate / load time live (gate 5)')
    if cur['meshes']['files'] > base['meshes']['files']:
        warn.append(f'land-mesh files {cur["meshes"]["files"]} > base {base["meshes"]["files"]} (gate 5: engine-layer change, live test)')
    for w, ks in (patch.get('new_vs_baseline') or {}).items():
        warn.append(f'patchiness {w}: lit {ks} beyond the baseline')
    if patch.get('error'):
        warn.append(patch['error'])
    return hard, warn


def snapshot(path):
    """(hard link of the pack under SCRATCH/packs-in named by its sha256, sha256). Renders and metrics read the
    link: packaging renames the candidate and writes a new one at the same path mid-review (2026-10-31, v21e over
    v21d), which a link does not follow. Same volume only (a 4.5 GB copy is not made silently)."""
    d = SCRATCH / 'packs-in'
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f'incoming-{os.getpid()}.pack'
    if tmp.exists():
        tmp.unlink()
    os.link(path, tmp)
    sha = sha256_file(tmp)
    dst = d / f'{sha[:12]}.pack'
    if dst.exists():
        tmp.unlink()
    else:
        os.replace(tmp, dst)
    return dst, sha


def cmd_run(a):
    if not a.base_design:
        raise SystemExit('--base-design is required: the region design the base pack was built from (its towns are drawn)')
    if a.backend:
        os.environ['TW3K_RENDER_BACKEND'] = a.backend     # spawn workers inherit it
    refuse_paths_in_locks([a.pack, a.base, a.map, a.base_map, a.design, a.base_design, a.extra, OUT, REVIEWS])
    # this process (metrics, change grids) reads the canvas through the pack's design; each side's render
    # processes get their own design in _render_todo
    os.environ['TW3K_DESIGN'] = str(Path(a.design).resolve())
    import render_live_view as L
    import pipeline_lock as PL
    tc, tcs = side_towns(a.design)
    tb, tbs = side_towns(a.base_design)
    print(f'design {a.design}: {len(tc)} towns {tcs[:12]}; base design {a.base_design}: {len(tb)} towns {tbs[:12]}', flush=True)
    pack, sha = snapshot(Path(a.pack).resolve())
    a.base = str(pick_base(a, sha))
    base, bsha = snapshot(Path(a.base))
    pack_path = str(Path(a.pack).resolve())
    cams, snow = load_cameras(a.cameras, a.extra, a.only, a.version)
    print(f'{len(cams)} cameras ({len(snow)} snow crops)', flush=True)
    if sha == bsha:
        raise SystemExit('pack and base are the same file')
    tag = sha[:12]
    rp = REVIEWS / f'{tag}.json'
    if rp.is_file() and json.loads(rp.read_text(encoding='utf-8')).get('finalized') and not a.force:
        raise SystemExit(f'{rp} is finalized; --force to review again')
    out = OUT / tag
    for s in ('sbs', 'squint', 'heat'):
        (out / s).mkdir(parents=True, exist_ok=True)
    tok = take_slot3(PL, f'speedup art_review {tag}')
    try:
        proc = patchiness_start(pack, sha, out)
        cur, tcur = metrics(L, pack, sha, a.map)
        old, told = metrics(L, base, bsha, a.base_map or a.map)
        rkey, bkey = render_key(tc), render_key(tb)
        R = render_jobs([(base, bsha, c) for c in cams], bkey, a.workers, a.base_design, tbs)
        rb = {n: R[(bsha, n)] for n in (c['name'] for c in cams)}
        plan = plan_renders(L, base, pack, cams, rb, a, town_changes(tb, tc))
        R = render_jobs([(pack, sha, c) for c in cams if c['name'] in plan['render']], rkey, a.workers, a.design, tcs)
        rc = {n: R[(sha, n)] for n in plan['render']}
        bad = audit(rb, rc, plan['audit'])
        if bad:
            # a view judged unchanged rendered differently: the change grids missed an input; render everything
            plan['audit_failed'] = bad
            R = render_jobs([(pack, sha, c) for c in cams if c['name'] not in rc], rkey, a.workers, a.design, tcs)
            rc.update({n: R[(sha, n)] for n in (c['name'] for c in cams) if (sha, n) in R})
            plan['unchanged'] = []
        for n in plan['unchanged']:
            rc[n] = rb[n]
        used = one_backend(rb, rc)
        patch = patchiness(proc, out)
        snow_res = None
        if snow and a.snow_script and Path(a.snow_script).is_file():
            r = subprocess.run([sys.executable, '-B', str(a.snow_script), '--run', tag, '--pack', str(pack), '--cameras', str(a.cameras),
                                '--out', str(out / 'snow')], env=dict(os.environ, TW3K_DESIGN=str(Path(a.design).resolve())))
            sj = out / 'snow' / 'snow-crops.json'
            snow_res = json.loads(sj.read_text(encoding='utf-8')) if r.returncode == 0 and sj.is_file() else dict(error=f'exit {r.returncode}')
        elif snow:
            snow_res = dict(skipped=f'snow_review script not found ({a.snow_script})')
    finally:
        PL.release(3, tok, PL.WORK)
        print(time.strftime('%H:%M:%S'), 'released', PL.path(3, PL.WORK), flush=True)
    from PIL import Image
    tiles, sq, rows = [], [], []
    same = set(plan['unchanged'])
    for c in cams:
        n, cam = c['name'], c['cam']
        ib = label(Image.open(rb[n][0]).convert('RGB'), f'base {bsha[:8]}  {n}')
        ic = label(Image.open(rc[n][0]).convert('RGB'), f'this {tag[:8]}  {n}  ({", ".join(f"{v:g}" for v in cam)})'
                   + ('\nunchanged: base render reused' if n in same else ''))
        s = pair(ib, ic); s.save(out / 'sbs' / f'{n}.png')
        q = pair(squint(ib), squint(ic)); q.save(out / 'squint' / f'{n}.png')
        hb, sb = heat(rb[n][1]['footprint'], told, cam, 'base')
        hc, sc = heat(rc[n][1]['footprint'], tcur, cam, 'this')
        pair(hb, hc).save(out / 'heat' / f'{n}.png')
        if n not in same:      # the montage shows what the reviewer judges: changed views and sentinels
            tiles.append(s.resize((s.width // 2, s.height // 2), Image.BOX)); sq.append(q)
        rows.append(dict(name=n, cam=cam, group=c.get('group'), sets=c.get('sets'), impact=c.get('impact'), why=c.get('why'),
                         seconds=rc[n][1]['seconds'], trees_base=sb, trees=sc, unchanged=n in same,
                         changed=plan['why'].get(n, []), sentinel=n in SENTINELS))
    note = f'; {len(same)} unchanged views not shown (base render reused)' if same else ''
    montage(tiles, f'art review {tag}: base {bsha[:12]} (left) | this (right){note}').save(out / 'montage.png')
    montage(sq, f'art review {tag}: squint, base (left) | this (right){note}').save(out / 'montage-squint.png')
    hard, warn = judge(cur, old, patch)
    if plan.get('audit_failed'):
        warn.append(f'unchanged-view audit failed {plan["audit_failed"]}: every view rendered; the change grids miss an input')
    res = dict(kind='art-review', version=1, candidate=a.version, pack_sha256=sha, pack_path=pack_path, snapshot=str(pack),
               base=dict(pack_sha256=bsha, pack_path=str(Path(a.base).resolve()), snapshot=str(base)), run=a.run or f'manual-{datetime.now():%Y%m%d-%H%M%S}',
               created=datetime.now().isoformat(timespec='seconds'), renderer=rkey, base_renderer=bkey, backend=used,
               design=dict(path=str(Path(a.design).resolve()), sha256=sha256_file(a.design), towns=len(tc), towns_sha256=tcs),
               base_design=dict(path=str(Path(a.base_design).resolve()), sha256=sha256_file(a.base_design), towns=len(tb),
                                towns_sha256=tbs),
               tier=plan['tier'], changes=plan['changes'], audit=dict(views=plan['audit'], failed=plan.get('audit_failed', [])),
               cameras_file=str(a.cameras), cameras_sha256=sha256_file(a.cameras), cameras=rows,
               auto=dict(metrics=cur, base_metrics=old, patchiness=patch, snow=snow_res, hard=hard, warn=warn),
               out=str(out), montage=str(out / 'montage.png'),
               verdict='auto-fail' if hard else 'pending-review', finalized=None)
    REVIEWS.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('hard:', hard or 'none'); print('warn:', warn or 'none')
    print('verdict', res['verdict'], '->', rp); print('montage', out / 'montage.png')


def tag_of(arg):
    p = Path(arg)
    return sha256_file(p)[:12] if p.is_file() else arg[:12]


def cmd_finalize(a):
    tag = tag_of(a.target)
    rp, vp = REVIEWS / f'{tag}.json', REVIEWS / f'{tag}.review.json'
    d = json.loads(rp.read_text(encoding='utf-8'))
    review = json.loads(vp.read_text(encoding='utf-8'))
    verdict, why, missing, failed = merge_review(d, review)
    d.update(verdict=verdict, verdict_reason=why, finalized=datetime.now().isoformat(timespec='seconds'),
             review=dict(file=vp.name, sha256=sha256_file(vp), reviewer=review.get('reviewer'), missing=missing, failed=failed))
    rp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print(verdict, '-', why)


def cmd_check(a):
    code, msg = check(a.pack)
    print(msg)
    sys.exit(code)


def cmd_status(a):
    d = find_review(sha256_file(a.pack))
    if d is None:
        print('no review'); return
    print(json.dumps({k: d.get(k) for k in ('pack_sha256', 'run', 'created', 'verdict', 'verdict_reason', 'finalized', 'montage')},
                     ensure_ascii=False, indent=1))
    print('hard:', d['auto']['hard']); print('warn:', d['auto']['warn'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--pack', required=True)
    r.add_argument('--base', help='pack to compare with (default: the last pack whose review passed, see last_passed)')
    r.add_argument('--map', required=True, help='a COPY of work/final-map/map.hex for this pack (playable regions)')
    r.add_argument('--base-map', help='map.hex copy for the base pack (default: --map)')
    r.add_argument('--design', default=str(LIVE_DESIGN), help='region design this pack was built from (its towns are '
                   'drawn; default: the live workspace design)')
    r.add_argument('--base-design', help='region design the base pack was built from (required; e.g. a git show of the '
                   'design at the base build commit, copied to scratch)')
    r.add_argument('--cameras', type=Path, default=CAMERAS)
    r.add_argument('--extra', help='json list of {name, cam, why} cameras for this review only')
    r.add_argument('--only', help='comma-separated camera names')
    r.add_argument('--snow-script', default=str(HERE / 'snow_review.py'))
    r.add_argument('--run', help='pipeline run id (TW3K_INVARIANT_RUN) of the pack, if known')
    r.add_argument('--force', action='store_true')
    r.add_argument('--version', help='candidate version (e.g. v21): change cameras from a later version are left out')
    r.add_argument('--full', action='store_true', help='render and review every camera (release packs); default: views '
                   'no change reaches reuse the base render and are not reviewed')
    r.add_argument('--workers', type=int, help='render processes (default: by free commit, 1-4)')
    r.add_argument('--backend', choices=('cpu', 'gpu'), help='renderer backend (default: gpu when CuPy imports); '
                   'part of the render cache key')
    f = sub.add_parser('finalize'); f.add_argument('target')
    c = sub.add_parser('check'); c.add_argument('pack')
    s = sub.add_parser('status'); s.add_argument('pack')
    a = ap.parse_args()
    {'run': cmd_run, 'finalize': cmd_finalize, 'check': cmd_check, 'status': cmd_status}[a.cmd](a)


if __name__ == '__main__':
    main()

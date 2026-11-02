"""Commit budget of the terrain pipeline's heavy steps: pool sizes and the launch gate.

The machine's limit is system commit (RAM + page file), not RAM: other applications
pre-commit memory they never touch. So pools size themselves from free commit when they
start, and the launch gate is computed from what a run will need instead of a fixed number.

Changes no artifact (pipeline-bake-design: this is how steps execute, not what they bake):
a pool's size only decides how many processes or threads run the same independent jobs
(envelope strips, blend row blocks, gap-fill meshes), whose results are combined in job
order. Outputs are byte-identical for any size.

COSTS are measured commit (private bytes, PagefileUsage; shared memory counted once) on
all-directions-30, 2048 canvas, 16384x16380 LF (mem-budget, 2026-10-26).

CLI for launch scripts and the pipeline:
  python memory_budget.py need polish
      prints the estimate as JSON (GB): need with 1 worker / the full pool, free commit,
      page-file growth room
  python memory_budget.py gate polish [--wait-min N]
      exit 0 once free commit plus the page file's growth room covers the 1-worker need
      (printing whether the run will page or shrink its pools); otherwise wait up to N
      minutes (default 0) and exit 3 with the reason.
"""
import ctypes, json, os, sys, time

GB = 2 ** 30
# Free commit a pool leaves for the rest of the machine and the pipeline's concurrent
# steps (the post-sync graph runs other nodes beside gap-fill).
RESERVE_GB = 10.
# Reserve of a pool inside a step that runs alone (polish: the pipeline starts nothing beside it).
STEP_RESERVE_GB = 3.
# The gate's margin on top of a step's need.
GATE_MARGIN_GB = 2.
# Disk space on the page file's drive the gate leaves unused when it counts on page-file
# growth (the pipeline writes its outputs there too).
DISK_KEEP_GB = 12.

# main: main process peak; shm: shared memory a pool creates; per_worker: one worker
# process (or, for a thread pool, one thread's temporaries, inside main's peak when the
# pool runs at its cap); cap: largest pool (None: one per CPU; 0: no pool).
COSTS = {
    # polish --skip-cache with the GPU envelope (the normal path; no envelope processes). Peak in
    # the projection solver (20.2; was 28.2); the material blend peaks at 18.5 with 12 threads
    # (0.55 GB each) and runs fewer threads when commit is short, so it never sets the need.
    'polish': {'main': 20.2, 'shm': 0., 'per_worker': 0., 'cap': 0},
    # polish on the CPU envelope fallback (no usable GPU or VRAM): spawn workers + shared blocks
    # (cx, cz float64 and out float32 over the 16384x16380 LF).
    'polish-cpu-envelope': {'main': 20.2, 'shm': 5.4, 'per_worker': .4, 'cap': None},
    # sync_final_terrain_surface (4.7) with its polish --cache-only child (6.5-8.6) and the side
    # lane beside it (expand_campaign_water_colour 6.6, was 11.7).
    'sync-final': {'main': 12.0, 'shm': 0., 'per_worker': 0., 'cap': 0},
    # fill_polished_mesh_gaps: main 1.9, six workers 1.8 together (its own workers() keeps 1 GB each).
    'gap-fill': {'main': 1.9, 'shm': 0., 'per_worker': .35, 'cap': 6},
}
# The pipeline runs these one after another; its need is the largest.
PIPELINE = ('polish', 'sync-final', 'gap-fill')


def _status():
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + \
                   [(f, ctypes.c_ulonglong) for f in ('phys', 'phys_free', 'commit', 'commit_free', 'virtual', 'virtual_free', 'extended')]
    try:
        s = Status(); s.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s
    except (AttributeError, OSError):
        pass
    return None


def free_commit_gb():
    """Free system commit (commit limit - commit charge) in GB, or None off Windows."""
    s = _status()
    return None if s is None else s.commit_free / GB


def pagefile_growth_gb():
    """How far a system-managed page file can still grow (GB): up to 3x RAM, keeping
    DISK_KEEP_GB free on its drive. 0 for a fixed-size page file or off Windows."""
    try:
        import winreg, shutil
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management') as k:
            files = winreg.QueryValueEx(k, 'PagingFiles')[0]
        s = _status(); room = 0.
        for entry in files:
            parts = entry.split()
            if len(parts) != 3 or parts[1:] != ['0', '0'] or parts[0].startswith('?'):
                continue   # fixed size (or one automatic file for all drives: not counted)
            path = parts[0]
            size = os.path.getsize(path) / GB if os.path.exists(path) else 0.
            room += max(0., min(3 * s.phys / GB - size, shutil.disk_usage(path[:3]).free / GB - DISK_KEEP_GB))
        return room
    except Exception:
        return 0.


def pool_size(per_worker_gb, cap=None, reserve_gb=RESERVE_GB, extra_gb=0., env=None):
    """Workers for a pool started now: at most `cap` (default one per CPU), and one per
    per_worker_gb of free commit above reserve_gb + extra_gb (memory the caller will still
    allocate while the pool runs). Never below 1. `env` names an environment variable whose
    value, when set, is used instead (a fixed count, for tests and measurements)."""
    if env and os.environ.get(env):
        return max(1, int(os.environ[env]))
    n = cap or os.cpu_count() or 1
    free = free_commit_gb()
    if free is not None and per_worker_gb > 0:
        n = min(n, int((free - reserve_gb - extra_gb) / per_worker_gb))
    return max(1, n)


def need(step, workers=None):
    """Commit (GB) a step needs: main peak + shared memory + workers x per-worker cost. With
    workers None: (need with 1 worker, need with the full pool). 'pipeline': its largest step."""
    if step == 'pipeline':
        both = [need(s) for s in PIPELINE]
        return max(b[0] for b in both), max(b[1] for b in both)
    c = COSTS[step]
    cap = c['cap'] if c['cap'] is not None else (os.cpu_count() or 1)
    at = lambda n: c['main'] + c['shm'] + max(0, min(n, cap)) * c['per_worker']
    return at(workers) if workers is not None else (at(1), at(cap))


def _main(argv):
    steps = tuple(COSTS) + ('pipeline',)
    if len(argv) < 2 or argv[0] not in ('need', 'gate') or argv[1] not in steps:
        print(__doc__); print('steps:', ', '.join(steps)); return 2
    step = argv[1]; lo, full = need(step)
    lo += GATE_MARGIN_GB; full += GATE_MARGIN_GB
    def info():
        free = free_commit_gb(); grow = pagefile_growth_gb()
        return {'step': step, 'need_min_gb': round(lo, 1), 'need_full_gb': round(full, 1),
                'free_gb': None if free is None else round(free, 1), 'pagefile_growth_gb': round(grow, 1)}, free, grow
    if argv[0] == 'need':
        print(json.dumps(info()[0])); return 0
    wait = float(argv[argv.index('--wait-min') + 1]) if '--wait-min' in argv else 0.
    end = time.time() + wait * 60
    while True:
        d, free, grow = info()
        if free is None or free + grow >= lo:
            how = 'full speed' if free is None or free >= full else ('pools shrink to fit' if free >= lo else
                  f'the page file grows by up to {lo - free:.1f} GB (slower: paging)')
            print(f'memory gate open for {step}: {how}', json.dumps(d), flush=True); return 0
        if time.time() >= end:
            print(f'memory gate closed: {step} needs {lo:.1f} GB of commit (main peak + shared memory + 1 worker + '
                  f'{GATE_MARGIN_GB:.0f} GB margin); free {free:.1f} GB plus {grow:.1f} GB of page-file growth room. '
                  f'Close applications or free disk space on the page-file drive; {full:.1f} GB free runs at full speed.', flush=True)
            return 3
        time.sleep(30)


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))

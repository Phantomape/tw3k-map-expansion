"""Multi-process edge-cost envelope (polish_campaign_terrain.edge_cost_envelope).

Strips of rows (x pass) or columns (z pass) are independent, so worker
processes update them in place in shared memory. The cumulative edge-cost
coordinates depend only on the original heights: they are computed once and
shared. Results are identical to the single-process envelope.

Speed version (same strips, same dirty-strip and reach logic, same float64
element arithmetic, so every value, sweep count and skip is identical):
  * each strip is processed in cache-sized blocks with preallocated buffers
    (no 8 MB temporaries per numpy call, no page-fault churn); the z pass
    carries the running minimum from block to block (min is exact);
  * reach spans come from per-strip "any finite" bands instead of a full
    chessboard dilation of the whole array per call;
  * the cumulative coordinates are built in parallel strips.

Envelope runs on the GPU instead (gpu_envelope.py, same bytes) when it can; CpuEnvelope is
this module's worker-pool path.

Windows starts workers by re-importing the parent's main script; the pool is
created with the main module's file hidden so a worker never re-runs the
polish script, only this module.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context, shared_memory

import numpy as np

import memory_budget

_W = {}
_BUF = {}
ROW_BLOCK = 1         # x pass: one 16384-sample row per block (128 KB float64), tuned
COL_BLOCK = 128       # z pass: rows per block of a 64-column strip (64 KB float64), tuned
# Commit per worker process (interpreter, numpy, block buffers: 0.38 GB measured on the 16384x16380
# LF; the shared blocks, 5.4 GB there, are charged once, at creation) and what polish's main process
# still allocates after the pool starts (the projection's bounds and temporaries; memory_budget.COSTS).
WORKER_GB = .4
MAIN_GROWTH_GB = 6.


def _attach(names, shape):
    for key, (name, dtype) in names.items():
        shm = shared_memory.SharedMemory(name=name)
        _W[key] = (shm, np.ndarray(shape, dtype=dtype, buffer=shm.buf))


def _buffers(kind, rows, cols):
    """Per-process scratch (V, A, B float64, U float32) of at least rows x cols,
    returned as rows x cols views; grown, never shrunk."""
    b = _BUF.get(kind)
    if b is None or b[0].shape[0] < rows or b[0].shape[1] < cols:
        shape = (max(rows, b[0].shape[0] if b else 0), max(cols, b[0].shape[1] if b else 0))
        b = _BUF[kind] = tuple(np.empty(shape, np.float64) for _ in range(3)) + (np.empty(shape, np.float32),)
    return tuple(x[:rows, :cols] for x in b)


def _stats(t, u, acc):
    """Accumulate the strip's improvement terms exactly as the reference does
    on the whole strip: any(~finite(target) & finite(updated)), all(finite(target)),
    max(target - updated)."""
    ft = np.isfinite(t)
    if not acc[0] and np.any(~ft & np.isfinite(u)):
        acc[0] = True
    if acc[1]:
        if ft.all():
            m = float(np.max(t - u)) if t.size else -np.inf
            if m > acc[2]:
                acc[2] = m
        else:
            acc[1] = False


def _strip(axis, start, stop, lo=None, hi=None):
    out = _W['out'][1]
    coordinate_all = _W['cx' if axis == 1 else 'cz'][1]
    span = slice(lo, hi)
    selection = (slice(start, stop), span) if axis == 1 else (span, slice(start, stop))
    target = out[selection]
    coordinate = coordinate_all[selection]
    acc = [False, True, -np.inf]  # any inf->finite, all target finite, max(target-updated)
    if axis == 1:
        rows, cols = target.shape
        changed = np.zeros(cols, bool)
        for r0 in range(0, rows, ROW_BLOCK):
            r1 = min(r0 + ROW_BLOCK, rows)
            t = target[r0:r1]; c = coordinate[r0:r1]
            V, A, B, U = _buffers('x', r1 - r0, cols)
            np.copyto(V, t)
            np.subtract(V, c, out=A); np.minimum.accumulate(A, axis=1, out=B); np.add(B, c, out=B)
            np.add(V, c, out=A); np.minimum.accumulate(A[:, ::-1], axis=1, out=V[:, ::-1]); np.subtract(V, c, out=V)
            np.minimum(B, V, out=B); np.copyto(U, B, casting='same_kind')
            _stats(t, U, acc)
            changed |= (U != t).any(axis=0)
            t[...] = U
    else:
        rows, cols = target.shape
        changed_rows = np.zeros(rows, bool)
        V, A, B, U = _buffers('z', COL_BLOCK, cols)
        # Backward minimum (bottom -> top) for the whole strip first, kept in a
        # strip-sized buffer; then forward blocks (top -> bottom) finish each block.
        bwd = _BUF.get('bwd')
        if bwd is None or bwd.shape[0] < rows or bwd.shape[1] < cols:
            bwd = _BUF['bwd'] = np.empty((max(rows, bwd.shape[0] if bwd is not None else 0), max(cols, bwd.shape[1] if bwd is not None else 0)), np.float64)
        bwd = bwd[:rows, :cols]
        carry = np.full(cols, np.inf)
        for r1 in range(rows, 0, -COL_BLOCK):
            r0 = max(0, r1 - COL_BLOCK); n = r1 - r0
            t = target[r0:r1]; c = coordinate[r0:r1]; a = A[:n]; v = V[:n]; o = bwd[r0:r1]
            np.copyto(v, t); np.add(v, c, out=a)
            np.minimum.accumulate(a[::-1], axis=0, out=o[::-1])
            np.minimum(o, carry, out=o); carry[:] = o[0]
        carry[:] = np.inf
        for r0 in range(0, rows, COL_BLOCK):
            r1 = min(r0 + COL_BLOCK, rows); n = r1 - r0
            t = target[r0:r1]; c = coordinate[r0:r1]; a = A[:n]; b = B[:n]; v = V[:n]; u = U[:n]
            np.copyto(v, t); np.subtract(v, c, out=a)
            np.minimum.accumulate(a, axis=0, out=b); np.minimum(b, carry, out=b); carry[:] = b[-1]
            np.add(b, c, out=b)
            np.subtract(bwd[r0:r1], c, out=a)
            np.minimum(b, a, out=b); np.copyto(u, b, casting='same_kind')
            _stats(t, u, acc)
            changed_rows[r0:r1] = (u != t).any(axis=1)
            t[...] = u
        changed = changed_rows
    if acc[0]:
        improvement = float('inf')
    elif acc[1]:
        improvement = acc[2] if target.size else float(np.max(target - target))
    else:
        improvement = 0.
    changed = np.flatnonzero(changed) + (lo or 0)
    return improvement, changed


_TPOOL = None


def _tpool():
    global _TPOOL
    if _TPOOL is None:
        _TPOOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    return _TPOOL


def _cumulative_into(dst, oldheight, step, axis, strip=None, plain=None):
    """dst = concat(0, cumsum(max(step, |diff(oldheight)|))) along axis, built in
    independent strips across the other axis (same sequential float64 sums).
    plain (optional, shape of the diff): edges that cost exactly step, i.e. the
    original height jump there is artificial and grants no slope allowance."""
    n_other = oldheight.shape[1 - axis]; strip = strip or (8 if axis == 0 else 4)

    def work(s0):
        s1 = min(s0 + strip, n_other)
        sel = (slice(None), slice(s0, s1)) if axis == 0 else (slice(s0, s1), slice(None))
        o = oldheight[sel]
        costs = np.maximum(step, np.abs(np.diff(o, axis=axis)))
        if plain is not None:
            costs[plain[sel]] = step
        d = dst[sel]
        if axis == 0:
            d[0] = 0; np.cumsum(costs, dtype=np.float64, axis=0, out=d[1:])
        else:
            d[:, 0] = 0; np.cumsum(costs, dtype=np.float64, axis=1, out=d[:, 1:])
    list(_tpool().map(work, range(0, n_other, strip)))


def _cumulative(oldheight, step, axis):
    shape = list(oldheight.shape)
    out = np.empty(shape, np.float64)
    _cumulative_into(out, oldheight, step, axis)
    return out


class Envelope:
    """Reusable envelope over one original height field: on the GPU (gpu_envelope.py, the same
    bytes and sweep counts) when it is usable and has the VRAM for the problem, else on
    CpuEnvelope's worker pool. A GPU error mid-run recomputes that envelope (and every later
    one) on the CPU, whose pool is built then from the same original heights."""

    def __init__(self, oldheight, step_x, step_z, strip=64, workers=None, plain_x=None, plain_z=None):
        import gpu_envelope
        self.shape = oldheight.shape
        self.strip = strip
        self._args = (oldheight, step_x, step_z, strip, workers, plain_x, plain_z)
        self._gpu = gpu_envelope.create(oldheight, step_x, step_z, strip, plain_x, plain_z)
        self._cpu = None if self._gpu is not None else CpuEnvelope(*self._args)

    def strips(self, axis):
        count = self.shape[0] if axis == 1 else self.shape[1]
        return [(start, min(start + self.strip, count)) for start in range(0, count, self.strip)]

    def run(self, values, max_sweeps=12, reach=None):
        if self._gpu is not None:
            result = self._gpu.run(values, max_sweeps, reach)
            if result is not None:
                return result
            self._gpu = None
        if self._cpu is None:
            self._cpu = CpuEnvelope(*self._args)
        return self._cpu.run(values, max_sweeps, reach)

    def close(self):
        if self._gpu is not None:
            self._gpu.close(); self._gpu = None
        if self._cpu is not None:
            self._cpu.close(); self._cpu = None
        self._args = None   # polish keeps the Envelope (via its future) alive past `del high_old`


class CpuEnvelope:
    """Reusable envelope over one original height field (worker processes, shared memory)."""

    def __init__(self, oldheight, step_x, step_z, strip=64, workers=None, plain_x=None, plain_z=None):
        self.shape = oldheight.shape
        self.strip = strip
        self.key = (id(oldheight), oldheight.shape, float(step_x), float(step_z), strip)
        self.shm = {}
        names = {}
        for key, dtype, fill in (('cx', np.float64, (step_x, 1, plain_x)),
                                 ('cz', np.float64, (step_z, 0, plain_z)),
                                 ('out', np.float32, None)):
            size = int(np.prod(self.shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=size)
            arr = np.ndarray(self.shape, dtype=dtype, buffer=shm.buf)
            if fill is not None:
                _cumulative_into(arr, oldheight, fill[0], fill[1], plain=fill[2])
            self.shm[key] = (shm, arr)
            names[key] = (shm.name, np.dtype(dtype).str)
        # Pool size from free commit now that the shared blocks are charged (memory_budget.pool_size):
        # every strip is an independent job and run() combines their results in job order
        # (max improvement, union of dirty strips), so any worker count gives the same bytes.
        workers = workers or memory_budget.pool_size(WORKER_GB, os.cpu_count() or 4, reserve_gb=memory_budget.STEP_RESERVE_GB,
                                                     extra_gb=MAIN_GROWTH_GB, env='TW3K_ENVELOPE_WORKERS')
        self.workers = workers
        main = sys.modules['__main__']
        saved = getattr(main, '__file__', None), getattr(main, '__spec__', None)
        try:
            if saved[0] is not None:
                del main.__file__
            main.__spec__ = None
            self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                                            initializer=_attach, initargs=(names, self.shape))
            # Start every worker now, while the main file is hidden.
            list(self.pool.map(_noop, range(workers * 2)))
        finally:
            if saved[0] is not None:
                main.__file__ = saved[0]
            main.__spec__ = saved[1]

    def strips(self, axis):
        count = self.shape[0] if axis == 1 else self.shape[1]
        return [(start, min(start + self.strip, count)) for start in range(0, count, self.strip)]

    def run(self, values, max_sweeps=12, reach=None):
        """Envelope of `values`. With `reach` (pixels), only cells within that
        chessboard distance of a finite value are computed; the rest stay
        infinite. Callers pass a reach beyond which no result can be within the
        height range in use, so every result that can bind is identical."""
        out = self.shm['out'][1]
        np.copyto(out, np.asarray(values, dtype=np.float32))
        strips = {axis: self.strips(axis) for axis in (1, 0)}
        spans = {axis: [(None, None)] * len(strips[axis]) for axis in (1, 0)}
        if reach is not None:
            spans = _reach_spans(out, int(reach), strips)
        dirty = {axis: np.array([sp[1] > sp[0] if sp[0] is not None else True for sp in spans[axis]]) for axis in (1, 0)}
        for sweep in range(max_sweeps):
            improvement = 0.
            for axis in (1, 0):
                todo = np.flatnonzero(dirty[axis]); dirty[axis][:] = False
                jobs = [(axis,) + strips[axis][i] + spans[axis][i] for i in todo if spans[axis][i][0] is None or spans[axis][i][1] > spans[axis][i][0]]
                for imp, changed in self.pool.map(_strip_job, jobs, chunksize=4):
                    improvement = max(improvement, imp)
                    dirty[1 - axis][np.unique(changed // self.strip)] = True
            if improvement <= 1e-7:
                break
        return out.copy(), sweep + 1

    def close(self):
        self.pool.shutdown()
        for shm, _ in self.shm.values():
            shm.close(); shm.unlink()


def _reach_spans(out, radius, strips):
    """Per strip (start, stop) span of envelope_mp's
    region=_dilate(isfinite(out), radius); band=region[strip].any(other axis)."""
    finite = np.empty(out.shape, bool)

    def fin(s0):
        np.isfinite(out[s0:s0 + 512], out=finite[s0:s0 + 512])
    list(_tpool().map(fin, range(0, out.shape[0], 512)))
    spans = {}
    for axis in (1, 0):
        # axis 1: strips of rows; band over columns. axis 0: strips of columns; band over rows.
        a = finite if axis == 1 else finite.T
        n_lines, n_band = a.shape
        g = 64
        groups = [(s, min(s + g, n_lines)) for s in range(0, n_lines, g)]

        def gany(se):
            return a[se[0]:se[1]].any(axis=0)
        G = np.array(list(_tpool().map(gany, groups)))  # (groups, n_band)
        res = []
        for start, stop in strips[axis]:
            w0, w1 = max(0, start - radius), min(n_lines, stop + radius)
            g0, g1 = -(-w0 // g), w1 // g          # full groups inside [w0, w1)
            if g0 < g1:
                band = G[g0:g1].any(axis=0)
                if w0 < g0 * g:
                    band |= a[w0:g0 * g].any(axis=0)
                if g1 * g < w1:
                    band |= a[g1 * g:w1].any(axis=0)
            else:
                band = a[w0:w1].any(axis=0)
            idx = np.flatnonzero(band)
            if len(idx):
                res.append((max(0, int(idx[0]) - radius), min(n_band, int(idx[-1]) + radius + 1)))
            else:
                res.append((0, 0))
        spans[axis] = res
    return spans


def _dilate(mask, radius):
    """Chessboard dilation by `radius`: separable window sums of prefix counts."""
    out = mask
    for axis in (0, 1):
        n = out.shape[axis]
        c = np.cumsum(out, axis=axis, dtype=np.int32)
        c = np.concatenate((np.zeros_like(c.take([0], axis=axis)), c), axis=axis)
        hi = np.minimum(np.arange(n) + radius + 1, n); lo = np.maximum(np.arange(n) - radius, 0)
        out = (c.take(hi, axis=axis) - c.take(lo, axis=axis)) > 0
    return out


def _noop(_):
    return 0


def _strip_job(job):
    return _strip(*job)

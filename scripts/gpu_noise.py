"""terrain_noise.fbm on the GPU (CuPy), bit-identical, with an automatic CPU fallback.

fbm      drop-in for terrain_noise.fbm: grid inputs (q varying only along columns and r
         only along rows: broadcast rows/columns, mgrid planes) run on the GPU, anything
         else on terrain_noise.fbm.
fbm_at   the grid noise of several seeds at the pixels of a mask only (the polish blend's
         'near' climate noise).
ridged   drop-in for terrain_noise.ridged (its NumPy expression over fbm above).
box_blur drop-in for fastops.box_blur (native_campaign_climate.box_blur) on 2-D arrays.
active   True while the GPU path is in use.

Why the bytes match
- The per-axis part (minus origin, / scale, floor, float32 fraction, smoothstep, lattice
  index) runs on the host with the NumPy operations of terrain_noise._value_noise_grid.
  The kernel does the lattice hash in uint32 (terrain_noise masks its int64 hash mod 2**32
  at every step; the int64 products never overflow for these coordinates) and the float32
  interpolation and octave sum one for one in NumPy's order:
  (a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy, v*2-1, amp*(..), total+.., total/norm
  (NEP 50: NumPy runs the Python scalars amp, norm, 1, 2 as float32 too).
- The kernels are compiled by our own NVRTC call with --fmad=false (no FMA contraction)
  and -ftz=false (NumPy keeps float32 subnormals); CuPy's RawKernel/RawModule always add
  -ftz=true, so the cubin is loaded with cupy.cuda.function.Module.
- box_blur: one thread per line runs fastops' float64 prefix sum of the edge-padded line
  (radius+1 before, radius after) in NumPy's sequential cumsum order, c[i+k] and c[i] as two
  running sums, then (c[i+k]-c[i])/k rounded to float32; axis 0 first, then axis 1 (on the
  transposed array, so both passes read coalesced columns), as fastops.
- Start-up self-test: blend patterns compared with terrain_noise, box blurs (masks, signed
  zeros, subnormals, radii past the line length) with fastops, byte for byte.
Fallback: TW3K_GPU=0, no CuPy or device, a compile error or self-test mismatch, and any
error of a GPU call (out of memory, CUDA errors) print one line and switch the process to
terrain_noise for good; a failed call is recomputed on the CPU. Threads share one lock
around the device work (the GPU runs one call at a time anyway; the host axis work and the
copy out of the thread's pinned staging buffer run outside it).
"""
import hashlib
import os
import sys
import threading
import time
import warnings

import numpy as np

import terrain_noise

_SRC = r'''
__device__ __forceinline__ float lat(unsigned int ix, unsigned int iy, unsigned int st) {
    unsigned int h = ix * 374761393u + iy * 668265263u + st;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float)((h ^ (h >> 16)) & 0xFFFFFFu) / 16777216.0f;
}

// Column c, row r of the grid: FX/IX[o*W+c] and FY/IY[o*H+r] are octave o's smoothstep
// weight and lattice index of that column/row, st[o] its seed term.
__device__ __forceinline__ float fbm_px(const float* FX, const unsigned int* IX, int W, int c,
                                        const float* FY, const unsigned int* IY, int H, int r,
                                        const unsigned int* st, const float* amps, int octaves, float norm) {
    float total = 0.0f;
    for (int o = 0; o < octaves; ++o) {
        float fx = FX[o * W + c], fy = FY[o * H + r];
        unsigned int ix = IX[o * W + c], iy = IY[o * H + r], s = st[o];
        float a = lat(ix, iy, s), b = lat(ix + 1u, iy, s), cc = lat(ix, iy + 1u, s), d = lat(ix + 1u, iy + 1u, s);
        float v = (a * (1.0f - fx) + b * fx) * (1.0f - fy) + (cc * (1.0f - fx) + d * fx) * fy;
        total = total + amps[o] * (v * 2.0f - 1.0f);
    }
    return total / norm;
}

// out[r*W+c] for rows row0+blockIdx.y.
extern "C" __global__ void fbm_grid(const float* FX, const unsigned int* IX, int W,
                                    const float* FY, const unsigned int* IY, int H, int row0,
                                    const unsigned int* st, const float* amps, int octaves, float norm, float* out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x, r = row0 + blockIdx.y;
    if (c < W) out[(long long)r * W + c] = fbm_px(FX, IX, W, c, FY, IY, H, r, st, amps, octaves, norm);
}

// out[s*n+k] = seed s at flat grid index idx[k] (st holds octaves terms per seed).
extern "C" __global__ void fbm_at(const float* FX, const unsigned int* IX, int W,
                                  const float* FY, const unsigned int* IY, int H,
                                  const long long* idx, long long n,
                                  const unsigned int* st, const float* amps, int octaves, float norm, float* out) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    long long p = idx[k]; int r = (int)(p / W), c = (int)(p - (long long)r * W);
    out[blockIdx.y * n + k] = fbm_px(FX, IX, W, c, FY, IY, H, r, st + blockIdx.y * octaves, amps, octaves, norm);
}

// fastops.box_blur along axis 0 of an (n, lines) array, column j per thread: the line padded
// p[t] = a[clamp(t-radius-1)] (t < n+2*radius+1), c its running float64 sum (c[0] = p[0], then
// c[m] = c[m-1] + p[m] as numpy's cumsum), out[i] = (c[i+k] - c[i]) / k; lead = c[i+k], lag = c[i].
extern "C" __global__ void box_cols(const float* a, float* out, int n, int lines, int radius) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= lines) return;
    int k = 2 * radius + 1;
    double lag = (double)a[j], lead = lag;
    for (int t = 1; t < k; ++t) lead += (double)a[(long long)min(max(t - radius - 1, 0), n - 1) * lines + j];
    for (int i = 0; i < n; ++i) {
        lead += (double)a[(long long)min(i + radius, n - 1) * lines + j];
        if (i > 0) lag += (double)a[(long long)max(i - radius - 1, 0) * lines + j];
        out[(long long)i * lines + j] = (float)((lead - lag) / (double)k);
    }
}
'''
_OPTIONS = ('--fmad=false', '-ftz=false', '-prec-div=true', '-prec-sqrt=true')
# Kernel cache (cubin per source, options, NVRTC version and GPU architecture) inside the
# interpreter's environment (<env>\tw3k-gpu-cache): on the env's drive (C: was nearly
# full), shared by every checkout and worktree, never inside a workspace, and tied to the
# environment that holds the NVRTC wheel (recreating it drops stale cubins). CuPy's own
# kernel cache goes there too (this module launches no CuPy-compiled kernels).
# TW3K_GPU_CACHE overrides; on the system drive nothing is cached (a compile takes ~0.6 s).
CACHE_DIR = os.environ.get('TW3K_GPU_CACHE') or os.path.join(sys.prefix, 'tw3k-gpu-cache')
_CACHE = None if os.path.splitdrive(os.path.abspath(CACHE_DIR))[0].upper() == os.environ.get('SystemDrive', 'C:').upper() else CACHE_DIR
STAGE = 256 * 16384   # float32 elements of a thread's pinned download buffer (one blend row block)
_LOCK = threading.RLock()
_GPU = None   # None: not started; False: CPU for good; dict: CuPy and the kernels
_LOCAL = threading.local()


_CLAIM = None   # the process's GPU lock token (pipeline_lock.gpu), held until exit


def claim():
    """Hold the machine's GPU lock (pipeline_lock '.gpu-lock') for this process: polish and the offline renderer
    share one 8 GB card (coordinator 2026-10-31). Waits up to TW3K_GPU_WAIT seconds (default 600) for another
    holder; returns the reason it could not be had, or None. Released at exit."""
    global _CLAIM
    if _CLAIM:
        return None
    import atexit
    import pipeline_lock as PL
    wait = float(os.environ.get('TW3K_GPU_WAIT', '600'))
    t0 = time.time()
    while True:
        tok = PL.gpu_acquire(f'{os.path.basename(sys.argv[0]) or "python"} gpu', settle=.5)
        if tok:
            _CLAIM = tok
            atexit.register(PL.gpu_release, tok)
            return None
        if time.time() - t0 >= wait:
            return f'GPU lock held by {PL.gpu_owner()!r}'
        time.sleep(2)


def _off(reason):
    global _GPU
    with _LOCK:
        if _GPU is not False:
            print(f'[gpu-noise] CPU noise from here on: {reason}', flush=True)
        _GPU = False


def _module(nvrtc, function, cc, src=None, name='gpu_noise.cu'):
    """Load the cubin of src (default: this module's kernels) for these OPTIONS from the cache,
    or compile it with NVRTC."""
    src = src or _SRC; opts = list(_OPTIONS) + [f'-arch=sm_{cc}']
    key = hashlib.sha256((src + repr(opts) + repr(nvrtc.getVersion())).encode()).hexdigest()[:24]
    path = _CACHE and os.path.join(_CACHE, key + '.cubin')
    if path and os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                mod = function.Module(); mod.load(f.read())
            return mod, 'cached cubin'
        except Exception:
            pass   # unreadable or stale file: compile again below
    prog = nvrtc.createProgram(src, name, [], [])
    try:
        try:
            nvrtc.compileProgram(prog, opts)
        except Exception as e:
            raise RuntimeError(f'NVRTC {e}: {nvrtc.getProgramLog(prog)}') from None
        cubin = nvrtc.getCUBIN(prog)
    finally:
        nvrtc.destroyProgram(prog)
    mod = function.Module(); mod.load(cubin)
    if path:
        try:
            os.makedirs(_CACHE, exist_ok=True)
            with open(f'{path}.{os.getpid()}.tmp', 'wb') as f:
                f.write(cubin)
            os.replace(f'{path}.{os.getpid()}.tmp', path)
        except OSError:
            pass   # read-only or full: compile again next time
    return mod, 'NVRTC compile'


def cupy():
    """(cupy, cupyx, nvrtc, function) imported for kernels loaded by _module (raises without CuPy)."""
    if _CACHE:
        os.environ.setdefault('CUPY_CACHE_DIR', os.path.join(_CACHE, 'cupy'))
    with warnings.catch_warnings():
        # CuPy warns when it finds no CUDA root to add to the DLL path; these modules need
        # only NVRTC (and no CUDA headers), and a CuPy that cannot load fails at first use.
        warnings.filterwarnings('ignore', 'CUDA path could not be detected', UserWarning)
        import cupy as cp
        import cupyx
        from cupy.cuda import nvrtc, function
    return cp, cupyx, nvrtc, function


def _start():
    """Import CuPy, load the kernels and self-test them (once; sets _GPU)."""
    global _GPU
    if os.environ.get('TW3K_GPU', '1') == '0':
        return _off('TW3K_GPU=0')
    busy = claim()
    if busy:
        return _off(busy)
    t = time.perf_counter()
    try:
        cp, cupyx, nvrtc, function = cupy()
        cc = cp.cuda.Device(0).compute_capability
        mod, source = _module(nvrtc, function, cc)
        g = dict(cp=cp, cupyx=cupyx, grid=mod.get_function('fbm_grid'), at=mod.get_function('fbm_at'), box=mod.get_function('box_cols'), module=mod)
        _selftest(g)
        name = cp.cuda.runtime.getDeviceProperties(0)['name']
    except Exception as e:
        return _off(f'{type(e).__name__}: {" ".join(str(e).split())[:300]}')
    _GPU = g
    print(f'[gpu-noise] GPU noise on {name.decode() if isinstance(name, bytes) else name} '
          f'(sm_{cc}, {source}, {time.perf_counter() - t:.1f}s)', flush=True)


def _gpu():
    if _GPU is None:
        with _LOCK:
            if _GPU is None:
                _start()
    return _GPU or None


def active():
    """True when fbm/fbm_at run on the GPU (starts it on the first call)."""
    return _gpu() is not None


def _octaves(scale, octaves, seeds):
    """Octave scales, seed terms (uint32, per seed then octave), float32 amplitudes and norm,
    as terrain_noise.fbm computes them."""
    scales, amps, amp, norm, s = [], [], 1.0, 0.0, float(scale)
    for o in range(octaves):
        scales.append(s); amps.append(amp)
        norm += amp; amp *= .5; s /= 2.03
    st = np.array([((seed + o * 101) * 2147483647) & 0xFFFFFFFF for seed in seeds for o in range(octaves)], np.uint32)
    return scales, st, np.array(amps, np.float32), np.float32(norm)


def _axis(c, origin, scales):
    """Smoothstep weight (float32) and lattice index (uint32) per octave along one grid axis:
    the NumPy operations of terrain_noise.fbm/_value_noise_grid on that axis."""
    d = np.asarray(c, np.float64) - origin
    F = np.empty((len(scales), d.size), np.float32); I = np.empty((len(scales), d.size), np.uint32)
    for o, s in enumerate(scales):
        x1 = d / s; x0 = np.floor(x1); f = (x1 - x0).astype(np.float32)
        F[o] = f * f * (3 - 2 * f); I[o] = x0.astype(np.int64).astype(np.uint32)   # int64 hash mod 2**32
    return F, I


def _download(g, a):
    """Device array -> host. Small results go through the thread's pinned buffer (returned
    as a view, copied by the caller outside the lock); larger ones straight to a new array."""
    if a.size > STAGE:
        return a.get(), False
    stage = getattr(_LOCAL, 'stage', None)
    if stage is None:
        stage = _LOCAL.stage = g['cupyx'].empty_pinned(STAGE, np.float32)
    v = stage[:a.size].reshape(a.shape); a.get(out=v)
    return v, True


def _run(work, *args):
    """work(g, *args) under the lock; None (and the CPU from here on) on any error."""
    try:
        with _LOCK:
            g = _GPU
            if not g:
                return None
            out, staged = work(g, *args)
        return np.array(out) if staged else out
    except Exception as e:
        _off(f'{type(e).__name__}: {" ".join(str(e).split())[:300]}')
        return None


def _grid(g, FX, IX, FY, IY, st, amps, norm):
    cp = g['cp']; W, H = FX.shape[1], FY.shape[1]
    d = [cp.asarray(a) for a in (FX, IX, FY, IY, st, amps)]
    out = cp.empty((H, W), cp.float32)
    for r0 in range(0, H, 65535):   # gridDim.y <= 65535
        g['grid'](((W + 255) // 256, min(65535, H - r0)), (256,),
                  (d[0], d[1], np.int32(W), d[2], d[3], np.int32(H), np.int32(r0), d[4], d[5], np.int32(FX.shape[0]), norm, out))
    return _download(g, out)


def _at(g, FX, IX, FY, IY, idx, st, amps, norm):
    cp = g['cp']; W, H = FX.shape[1], FY.shape[1]; n = idx.size; k = st.size // FX.shape[0]
    d = [cp.asarray(a) for a in (FX, IX, FY, IY, idx, st, amps)]
    out = cp.empty((k, n), cp.float32)
    g['at'](((n + 255) // 256, k), (256,),
            (d[0], d[1], np.int32(W), d[2], d[3], np.int32(H), d[4], np.int64(n), d[5], d[6], np.int32(FX.shape[0]), norm, out))
    return _download(g, out)


def _box(g, a, radius):
    cp = g['cp']; H, W = a.shape
    d = cp.asarray(a); t = cp.empty_like(d)
    g['box'](((W + 127) // 128,), (128,), (d, t, np.int32(H), np.int32(W), np.int32(radius)))   # axis 0
    d = cp.ascontiguousarray(t.T); t = cp.empty_like(d)
    g['box'](((H + 127) // 128,), (128,), (d, t, np.int32(W), np.int32(H), np.int32(radius)))   # axis 1, transposed
    return _download(g, cp.ascontiguousarray(t.T))


def _grid_axes(q, r):
    """(columns, rows) of grid inputs (exactly constant along the other axis, finite), else None."""
    q, r = np.asarray(q), np.asarray(r)
    if q.ndim != 2 or r.shape != q.shape or not q.size or q.dtype.kind not in 'fiu' or r.dtype.kind not in 'fiu':
        return None
    cols, rows = q[0], r[:, 0]
    if not (q.strides[0] == 0 or q.shape[0] == 1 or np.array_equal(q, np.broadcast_to(cols, q.shape))):
        return None
    if not (r.strides[1] == 0 or r.shape[1] == 1 or np.array_equal(r, np.broadcast_to(rows[:, None], r.shape))):
        return None
    return (cols, rows) if np.isfinite(cols).all() and np.isfinite(rows).all() else None


def fbm(q, r, origin, scale, octaves=4, seed=0):
    """terrain_noise.fbm, same bytes; on the GPU for grid inputs."""
    if _gpu():
        axes = _grid_axes(q, r)
        if axes is not None and octaves > 0:
            scales, st, amps, norm = _octaves(scale, octaves, (seed,))
            FX, IX = _axis(axes[0], origin[0], scales); FY, IY = _axis(axes[1], origin[1], scales)
            out = _run(_grid, FX, IX, FY, IY, st, amps, norm)
            if out is not None:
                return out
    return terrain_noise.fbm(q, r, origin, scale, octaves, seed)


def ridged(q, r, origin, scale, octaves=4, seed=0):
    """terrain_noise.ridged, same bytes (fbm above)."""
    return np.clip(1 - np.abs(fbm(q, r, origin, scale, octaves, seed)) * 1.6, 0, 1) ** 2


def box_blur(a, radius):
    """fastops.box_blur (= native_campaign_climate.box_blur), same bytes; on the GPU for 2-D arrays."""
    a = np.asarray(a)
    if a.ndim == 2 and a.size and isinstance(radius, (int, np.integer)) and radius >= 0 and _gpu():
        out = _run(_box, np.ascontiguousarray(a, np.float32), int(radius))
        if out is not None:
            return out
    import fastops
    return fastops.box_blur(a, radius)


def fbm_at(xa, ya, mask, origin, scale, octaves=4, seeds=(0,)):
    """np.stack([terrain_noise.fbm(X, Y, origin, scale, octaves, s)[mask] for s in seeds]),
    X, Y the grid of xa (columns) by ya (rows), mask of shape (len(ya), len(xa)). Same bytes
    as terrain_noise on the whole grid or on the gathered points (its value_noise path)."""
    xa, ya, mask = np.asarray(xa), np.asarray(ya), np.asarray(mask, bool); seeds = tuple(seeds)
    assert xa.ndim == ya.ndim == 1 and mask.shape == (ya.size, xa.size)
    if _gpu() and octaves > 0 and seeds and np.isfinite(xa).all() and np.isfinite(ya).all():
        idx = np.flatnonzero(mask)
        if idx.size:
            scales, st, amps, norm = _octaves(scale, octaves, seeds)
            FX, IX = _axis(xa, origin[0], scales); FY, IY = _axis(ya, origin[1], scales)
            out = _run(_at, FX, IX, FY, IY, idx, st, amps, norm)
            if out is not None:
                return out
    px, py = np.broadcast_to(xa[None, :], mask.shape)[mask], np.broadcast_to(ya[:, None], mask.shape)[mask]
    return np.stack([terrain_noise.fbm(px, py, origin, scale, octaves, s) for s in seeds]) if seeds else np.empty((0, int(mask.sum())), np.float32)


def _selftest(g):
    """GPU vs terrain_noise on blend patterns (negative and positive canvas offsets)."""
    xa = np.arange(0, 2048) / 8; ya = np.arange(16316, 16380) / 8; sh = (ya.size, xa.size); o = (696, 343)
    X, Y = np.broadcast_to(xa[None, :], sh), np.broadcast_to(ya[:, None], sh)
    for scale, octaves, seed in ((3, 4, 71), (1, 2, 105), (2.5, 4, 73), (5, 3, 999)):
        scales, st, amps, norm = _octaves(scale, octaves, (seed,))
        FX, IX = _axis(xa, o[0], scales); FY, IY = _axis(ya, o[1], scales)
        out = np.array(_grid(g, FX, IX, FY, IY, st, amps, norm)[0])
        if not np.array_equal(out.view(np.uint32), terrain_noise.fbm(X, Y, o, scale, octaves, seed).view(np.uint32)):
            raise RuntimeError(f'self-test mismatch (grid {scale},{octaves},{seed})')
    m = (np.arange(X.size).reshape(sh) % 7 == 0) | (X > 100); scales, st, amps, norm = _octaves(1.5, 2, (90, 91))
    FX, IX = _axis(xa / 2, o[0], scales); FY, IY = _axis(ya / 2, o[1], scales)
    out = np.array(_at(g, FX, IX, FY, IY, np.flatnonzero(m), st, amps, norm)[0])
    ref = np.stack([terrain_noise.fbm(X[m] / 2, Y[m] / 2, o, 1.5, 2, s) for s in (90, 91)])
    if not np.array_equal(out.view(np.uint32), ref.view(np.uint32)):
        raise RuntimeError('self-test mismatch (points)')
    import fastops
    rng = np.random.default_rng(7)
    for shape, radius in (((37, 300), 0), ((37, 300), 5), ((300, 41), 40), ((64, 129), 2), ((5, 9), 20)):
        a = (rng.standard_normal(shape) * 50).astype(np.float32); a[::7] = 0; a[1, :5] = -0.; a[2, :3] = 1e-41; a[3, 1::2] = -1e-42
        for v in (a, rng.random(shape) > .6, a > 0):
            ref = fastops.box_blur(v, radius); out = np.array(_box(g, np.ascontiguousarray(v, np.float32), radius)[0])
            if not np.array_equal(out.view(np.uint32), ref.view(np.uint32)):
                raise RuntimeError(f'self-test mismatch (box blur {shape}, {radius})')

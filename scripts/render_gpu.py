"""The live renderer's ground queries on the GPU (CuPy): ray march, surface and height samples, terrain AO and sun
shadows (render_live_view), bit for bit the NumPy results.

Why the bytes match: every step is a correctly rounded IEEE float64 operation (+ - * /, comparisons, floor,
truncation, min/max) done in NumPy's order, with --fmad=false (no fused multiply-add, gpu_noise._OPTIONS);
values NumPy derives with transcendental functions (cos/sin of the AO directions, the shadow step lengths, the
sun vector) are computed on the host by NumPy and passed in. tests/test_render_gpu.py compares against the CPU
code on synthetic grounds; art_review compares a CPU and a GPU render of real views once per renderer change.

Use: Device(ground, shader) per pack ground (uploads the LF raster once); set_frame() after every
LiveShader.frame / PackGround.mesh change; then march(), surface(), heights(), ao(), shadow() on host arrays.
The GPU is the machine's one card: callers hold pipeline_lock.gpu around device work.
"""
import numpy as np

import campaign_world as cw

_SRC = r'''
struct Ground {
    const unsigned short* lf; int lw, lh; double per, dx, dz, level, norm, gain;     // LF raster (north-first rows)
    const float* land; int has_mesh; double mx0, mz0, mstep; int mnx, mnz;          // PackGround.mesh (land grid)
    const float* water; int has_win; double wx0, wz0, wstep; int wnx, wnz;           // LiveShader.mwin water grid
    const float* wland;                                                              // LiveShader.mwin land grid
    const float* top; int has_ov; double ox0, oz0, ocell; int onx, onz;               // render_live_objects.Overlay
};

__device__ double lf_h(const Ground& g, double x, double z) {
    // render_campaign_view.Ground._bilinear (canvas frame: no native offset)
    double u = x / g.dx * g.per, v = z / g.dz * g.per;
    u = fmin(fmax(u, 0.0), (double)(g.lw - 1)); v = fmin(fmax(v, 0.0), (double)(g.lh - 1));
    long long u0 = (long long)u, v0 = (long long)v;
    double fu = u - (double)u0, fv = v - (double)v0;
    long long u1 = u0 + 1 < g.lw - 1 ? u0 + 1 : g.lw - 1, v1 = v0 + 1 < g.lh - 1 ? v0 + 1 : g.lh - 1;
    const unsigned short* r0 = g.lf + (long long)(g.lh - 1 - v0) * g.lw;   // south-up row v -> raster row h-1-v
    const unsigned short* r1 = g.lf + (long long)(g.lh - 1 - v1) * g.lw;
    double a = r0[u0], b = r0[u1], c = r1[u0], d = r1[u1];
    double raw = (a * (1.0 - fu) + b * fu) * (1.0 - fv) + (c * (1.0 - fu) + d * fu) * fv;
    return g.level + (raw / 65535.0 - g.norm) * g.gain;
}

__device__ double heights(const Ground& g, double x, double z) {
    // render_live_view.PackGround.heights: the coarse land mesh (bilinear over cell centres), else the LF
    double lf = lf_h(g, x, z);
    if (!g.has_mesh) return lf;
    double fi = (x - g.mx0) / g.mstep - 0.5, fj = (z - g.mz0) / g.mstep - 0.5;
    bool inside = fi >= 0.0 && fi <= (double)(g.mnx - 1) && fj >= 0.0 && fj <= (double)(g.mnz - 1);
    double ffi = floor(fi), ffj = floor(fj);
    long long i0 = (long long)ffi, j0 = (long long)ffj;
    i0 = i0 < 0 ? 0 : (i0 > g.mnx - 2 ? g.mnx - 2 : i0);
    j0 = j0 < 0 ? 0 : (j0 > g.mnz - 2 ? g.mnz - 2 : j0);
    double u = fmin(fmax(fi - (double)i0, 0.0), 1.0), v = fmin(fmax(fj - (double)j0, 0.0), 1.0);
    const float* L = g.land;
    double l00 = L[j0 * g.mnx + i0], l01 = L[j0 * g.mnx + i0 + 1], l10 = L[(j0 + 1) * g.mnx + i0], l11 = L[(j0 + 1) * g.mnx + i0 + 1];
    double h = (l00 * (1.0 - u) + l01 * u) * (1.0 - v) + (l10 * (1.0 - u) + l11 * u) * v;
    return (inside && isfinite(h)) ? h : lf;
}

__device__ double surface(const Ground& g, double x, double z) {
    // rv.Shader.surface with LiveShader.mesh_level (framed window, nearest cell): water where it is above the land
    if (g.has_win) {
        double fi = (x - g.wx0) / g.wstep, fj = (z - g.wz0) / g.wstep;
        bool inside = fi >= 0.0 && fi < (double)g.wnx && fj >= 0.0 && fj < (double)g.wnz;
        long long i = (long long)fi, j = (long long)fj;
        i = i < 0 ? 0 : (i > g.wnx - 1 ? g.wnx - 1 : i);
        j = j < 0 ? 0 : (j > g.wnz - 1 ? g.wnz - 1 : j);
        double water = g.water[j * g.wnx + i], land = g.wland[j * g.wnx + i];
        if (!inside) water = (-__longlong_as_double(0x7ff0000000000000LL));
        if (isfinite(water) && !isfinite(land)) land = heights(g, x, z);
        if (water > land) return water;
    }
    return heights(g, x, z);
}

__device__ double objects(const Ground& g, double x, double z) {
    // Overlay.height: nearest cell (truncated index), -inf outside the window
    long long i = (long long)((x - g.ox0) / g.ocell), j = (long long)((z - g.oz0) / g.ocell);
    bool ok = i >= 0 && i < g.onx && j >= 0 && j < g.onz;
    if (!ok) return (-__longlong_as_double(0x7ff0000000000000LL));
    return g.top[j * g.onx + i];
}

__device__ double npmax(double a, double b) {   // np.maximum: NaN if either is NaN
    if (isnan(a) || isnan(b)) return __longlong_as_double(0x7ff8000000000000LL);
    return a >= b ? a : b;
}

__device__ double query(const Ground& g, int kind, double x, double z) {
    // 0 shader surface, 1 ground heights, 2 LF, 3 surface with the object tops (gbuffer's lambda)
    return kind == 0 ? surface(g, x, z) : kind == 1 ? heights(g, x, z) : kind == 2 ? lf_h(g, x, z)
         : npmax(surface(g, x, z), objects(g, x, z));
}

extern "C" __global__ void k_surface(Ground g, const double* x, const double* z, long long n, int kind, double* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = query(g, kind, x[i], z[i]);
}

extern "C" __global__ void k_march_objects(Ground g, double ex, double ey, double ez, const double* d, const double* tg,
                                           const double* ta, long long n, int steps, int refine, double* tout) {
    // render_live_objects.march_objects for one found ray: fine steps over the last stretch before the ground hit
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double dx = d[3 * r], dy = d[3 * r + 1], dz = d[3 * r + 2], a = ta[r], b = tg[r];
    double prev = a, lo = b, hit = b;
    bool done = false;
    for (int k = 1; k <= steps; ++k) {
        double tk = a + (b - a) * (double)k / (double)steps;
        if (ey + dy * tk <= query(g, 3, ex + dx * tk, ez + dz * tk)) { done = true; hit = tk; lo = prev; break; }
        prev = tk;
    }
    if (!done) { tout[r] = b; return; }
    double p = lo, q = hit;
    for (int k = 0; k < refine; ++k) {
        double m = (p + q) / 2.0;
        if (ey + dy * m <= query(g, 3, ex + dx * m, ez + dz * m)) q = m; else p = m;
    }
    tout[r] = q;
}

extern "C" __global__ void k_march(Ground g, double ex, double ey, double ez, const double* d, const double* t0,
                                   const double* t1, const unsigned char* down, long long n, int steps, int refine,
                                   unsigned char* found, double* tout) {
    // render_live_view.march for one ray: uniform steps between the hmax and hmin planes, then bisection
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    found[r] = 0; tout[r] = __longlong_as_double(0x7ff8000000000000LL);
    if (!down[r]) return;
    double dx = d[3 * r], dy = d[3 * r + 1], dz = d[3 * r + 2], a = t0[r], b = t1[r];
    double prev = a, lo = 0.0, hit = 0.0;
    bool f = false;
    for (int k = 1; k <= steps; ++k) {
        double tk = a + (b - a) * (double)k / (double)steps;
        if (ey + dy * tk <= surface(g, ex + dx * tk, ez + dz * tk)) { f = true; lo = prev; hit = tk; break; }
        prev = tk;
    }
    if (!f) return;
    double p = lo, q = hit;
    for (int k = 0; k < refine; ++k) {
        double m = (p + q) / 2.0;
        if (ey + dy * m <= surface(g, ex + dx * m, ez + dz * m)) q = m; else p = m;
    }
    found[r] = 1; tout[r] = q;
}

extern "C" __global__ void k_ao(Ground g, const double* x, const double* z, const double* h, long long n,
                                const double* ox, const double* oz, const double* rr, int m, double* out) {
    // render_live_view.terrain_ao: mean of clip((height at (x + r cos a, z + r sin a) - h) / r, 0, 1), radii outer
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double occ = 0.0;
    for (int k = 0; k < m; ++k) {
        double hi = heights(g, x[i] + ox[k], z[i] + oz[k]);
        occ = occ + fmin(fmax((hi - h[i]) / rr[k], 0.0), 1.0);
    }
    out[i] = occ / (double)m;
}

extern "C" __global__ void k_shadow(Ground g, const double* x, const double* z, const double* h02, long long n,
                                    double s0, double s1, double s2, const double* ts, int steps, int kind, double* out) {
    // render_live_view.sun_shadow; h02 = h + .02 by NumPy on the host (float32 object tops add in float32)
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double lit = 1.0;
    for (int k = 0; k < steps; ++k) {
        double t = ts[k];
        double y = h02[i] + s1 * t;
        double rise = query(g, kind, x[i] + s0 * t, z[i] + s2 * t) - y;
        lit = fmin(lit, fmin(fmax(1.0 - rise / (0.05 + 0.02 * t), 0.0), 1.0));
    }
    out[i] = lit;
}

__device__ void max_f32(float* a, float v) {
    // a = max(a, v); the stored value never decreases, so the result is the max whatever the order (NaN never wins)
    int* ai = (int*)a;
    int old = *ai;
    while (__int_as_float(old) < v) {
        int seen = atomicCAS(ai, old, __float_as_int(v));
        if (seen == old) break;
        old = seen;
    }
}

extern "C" __global__ void k_zfill(float* grid, long long gw, const long long* cs, long long items, const int* x0t,
                                   const int* z0b, const int* w, const float* k, long long n, float neps, float onep) {
    // campaign_water_meshes._fill over the band items of raster(): candidate cell n of item it (cs = cumsum of the
    // items' cell counts), float32 operations in NumPy's order
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    long long lo = 0, hi = items - 1;          // first item with cs > c
    while (lo < hi) { long long mid = (lo + hi) / 2; if (cs[mid] > c) hi = mid; else lo = mid + 1; }
    long long off = c - (lo ? cs[lo - 1] : 0);
    long long dx = off % w[lo], dz = off / w[lo];
    float fx = (float)dx, fz = (float)dz;
    const float* q = k + lo * 9;
    float l1 = __fadd_rn(__fadd_rn(__fmul_rn(q[0], fx), __fmul_rn(q[1], fz)), q[2]);
    float l2 = __fadd_rn(__fadd_rn(__fmul_rn(q[3], fx), __fmul_rn(q[4], fz)), q[5]);
    if (!(l1 >= neps && l2 >= neps && __fadd_rn(l1, l2) <= onep)) return;
    float y = __fadd_rn(__fadd_rn(__fmul_rn(q[6], fx), __fmul_rn(q[7], fz)), q[8]);
    max_f32(grid + (z0b[lo] + dz) * gw + (x0t[lo] + dx), y);
}
'''

BLOCK = 128


class ZFill:
    """campaign_water_meshes.raster's max z-buffer fill on the GPU (set as campaign_water_meshes.GPU_FILL by a caller
    holding pipeline_lock.gpu). Each candidate cell is tested and written with the CPU's float32 operations; the
    grid holds the max per cell whatever the write order, as the CPU's loop does (a 0.0 / -0.0 tie may keep the
    other sign)."""
    def __init__(self):
        import gpu_noise
        cp, _, nvrtc, function = gpu_noise.cupy()
        self.cp = cp
        cc = cp.cuda.Device(0).compute_capability
        self.k = gpu_noise._module(nvrtc, function, cc, _SRC, 'render_gpu.cu')[0].get_function('k_zfill')

    def __call__(self, grid, x0t, z0b, w, cnt, k, eps):
        """Fill grid (nz, nx float32, in place) from the items' rebased float32 constants k (n, 9)."""
        cp = self.cp
        cs = np.cumsum(cnt)
        n = int(cs[-1]) if len(cs) else 0
        if not n:
            return grid
        g = cp.asarray(grid)
        i32 = lambda a: cp.asarray(np.ascontiguousarray(a, np.int32))
        self.k(((n + BLOCK - 1) // BLOCK,), (BLOCK,),
               (g, np.int64(grid.shape[1]), cp.asarray(cs.astype(np.int64)), np.int64(len(cnt)), i32(x0t), i32(z0b), i32(w),
                cp.asarray(np.ascontiguousarray(k, np.float32)), np.int64(n), np.float32(-eps), np.float32(1 + eps)))
        grid[...] = cp.asnumpy(g)
        del g
        cp.get_default_memory_pool().free_all_blocks()
        return grid


class Surface:
    """A surface callable (the CPU function, for every other caller) that tells render_live_view.march /
    sun_shadow and render_live_objects.march_objects to run on `gpu` instead: the shader surface, or with
    objects=True the surface with the overlay's object tops (Device.set_overlay)."""
    def __init__(self, fn, gpu, objects=False):
        self.fn, self.gpu, self.objects = fn, gpu, objects

    def __call__(self, wx, wz):
        return self.fn(wx, wz)


class Device:
    """GPU copies of one pack ground's height inputs and the ground kernels."""
    def __init__(self, ground):
        import gpu_noise
        cp, _, nvrtc, function = gpu_noise.cupy()
        self.cp = cp
        cc = cp.cuda.Device(0).compute_capability
        self.mod, _ = gpu_noise._module(nvrtc, function, cc, _SRC, 'render_gpu.cu')
        self.k = {n: self.mod.get_function(n) for n in ('k_surface', 'k_march', 'k_march_objects', 'k_ao', 'k_shadow')}
        self.g = ground
        assert not ground.vanilla and tuple(ground.shift) == (0., 0.)
        self.lf = cp.asarray(np.ascontiguousarray(ground.lf))
        self.mesh_src = self.win_src = None
        self.land = self.water = self.wland = None
        self.ov_src, self.top = None, None
        self._empty = cp.zeros(1, cp.float32)

    # ---------------------------------------------------------------- state
    def sync(self, shader):
        """Upload PackGround.mesh and the shader's framed window when they changed (identity of the host arrays)."""
        cp = self.cp
        m = self.g.mesh
        if m is None:
            self.mesh_src, self.land = None, None
        elif self.mesh_src is None or self.mesh_src[5] is not m[5] or self.mesh_src[:5] != m[:5]:
            self.mesh_src, self.land = m, cp.asarray(np.ascontiguousarray(m[5], np.float32))
        w = getattr(shader, 'mwin', None)
        if w is None:
            self.win_src, self.water, self.wland = None, None, None
        elif self.win_src is None or self.win_src[5] is not w[5] or self.win_src[:5] != w[:5]:
            self.win_src = w
            self.water = cp.asarray(np.ascontiguousarray(w[5], np.float32))
            self.wland = self.land if (m is not None and w[6] is m[5]) else cp.asarray(np.ascontiguousarray(w[6], np.float32))

    def set_overlay(self, ov):
        """Upload an Overlay's top grid (None: no objects)."""
        if ov is None:
            self.ov_src, self.top = None, None
        elif self.ov_src is not ov:
            self.ov_src, self.top = ov, self.cp.asarray(np.ascontiguousarray(ov.top, np.float32))

    def _ground(self):
        """The Ground struct as a NumPy record (CuPy passes it by value)."""
        g, m, w = self.g, self.mesh_src, self.win_src
        lh, lw = g.lf.shape
        rec = np.zeros(1, _GROUND)[0]
        rec['lf'] = self.lf.data.ptr; rec['lw'] = lw; rec['lh'] = lh
        rec['per'] = cw.LF_PX_PER_HEX; rec['dx'] = cw.DX; rec['dz'] = cw.DZ
        rec['level'] = cw.LEVEL; rec['norm'] = cw.NORM; rec['gain'] = cw.GAIN
        rec['land'] = (self.land if m is not None else self._empty).data.ptr
        rec['has_mesh'] = m is not None
        if m is not None:
            rec['mx0'], rec['mz0'], rec['mstep'], rec['mnx'], rec['mnz'] = m[0], m[1], m[2], m[3], m[4]
        rec['water'] = (self.water if w is not None else self._empty).data.ptr
        rec['wland'] = (self.wland if w is not None else self._empty).data.ptr
        rec['has_win'] = w is not None
        if w is not None:
            rec['wx0'], rec['wz0'], rec['wstep'], rec['wnx'], rec['wnz'] = w[0], w[1], w[2], w[3], w[4]
        o = self.ov_src
        rec['top'] = (self.top if o is not None else self._empty).data.ptr
        rec['has_ov'] = o is not None
        if o is not None:
            rec['ox0'], rec['oz0'], rec['ocell'], rec['onx'], rec['onz'] = o.x0, o.z0, o.cell, o.nx, o.nz
        return rec

    def _grid(self, n):
        return ((n + BLOCK - 1) // BLOCK,), (BLOCK,)

    def _d(self, a):
        return self.cp.asarray(np.ascontiguousarray(a, np.float64))

    # ---------------------------------------------------------------- queries
    def _points(self, kind, wx, wz):
        cp = self.cp
        x, z = np.broadcast_arrays(np.asarray(wx, np.float64), np.asarray(wz, np.float64))
        n = x.size
        out = cp.empty(n, cp.float64)
        if n:
            self.k['k_surface'](*self._grid(n), (self._ground(), self._d(x.ravel()), self._d(z.ravel()), np.int64(n), np.int32(kind), out))
        return cp.asnumpy(out).reshape(x.shape)

    def surface(self, wx, wz):
        return self._points(0, wx, wz)

    def heights(self, wx, wz):
        return self._points(1, wx, wz)

    def lf_heights(self, wx, wz):
        return self._points(2, wx, wz)

    def surface_objects(self, wx, wz):
        return self._points(3, wx, wz)

    def march(self, eye, d, hmin, hmax, steps=160, refine=8):
        """render_live_view.march(shader.surface, ...): (found, t)."""
        cp = self.cp
        H, W = d.shape[:2]
        dy = d[..., 1]
        down = dy < -1e-4
        sdy = -np.where(down, dy, -1)
        t0 = np.maximum(np.where(down, (eye[1] - hmax) / sdy, np.nan), 0)
        t1 = np.where(down, (eye[1] - hmin) / sdy, np.nan)
        n = H * W
        found = cp.empty(n, cp.uint8); t = cp.empty(n, cp.float64)
        self.k['k_march'](*self._grid(n), (self._ground(), np.float64(eye[0]), np.float64(eye[1]), np.float64(eye[2]),
                                           self._d(d.reshape(-1, 3)), self._d(t0.ravel()), self._d(t1.ravel()),
                                           cp.asarray(down.ravel().astype(np.uint8)), np.int64(n), np.int32(steps),
                                           np.int32(refine), found, t))
        return cp.asnumpy(found).reshape(H, W).astype(bool), cp.asnumpy(t).reshape(H, W)

    def march_objects(self, eye, d, found, t, hrel, steps=96, refine=8):
        """render_live_objects.march_objects(surface with the overlay of set_overlay, ...)."""
        cp = self.cp
        dd = d[found]
        tg = t[found]
        span = hrel / np.clip(-dd[:, 1], .05, 1) + .05
        ta = np.maximum(tg - span, 0)
        n = len(tg)
        out = t.copy()
        if n:
            r = cp.empty(n, cp.float64)
            self.k['k_march_objects'](*self._grid(n), (self._ground(), np.float64(eye[0]), np.float64(eye[1]), np.float64(eye[2]),
                                                       self._d(dd), self._d(tg), self._d(ta), np.int64(n), np.int32(steps),
                                                       np.int32(refine), r))
            out[found] = cp.asnumpy(r)
        return out

    def ao(self, wx, wz, h, radii):
        """render_live_view.terrain_ao (offsets r cos a, r sin a by NumPy, in its loop order)."""
        cp = self.cp
        ox, oz, rr = [], [], []
        for r in radii:
            for a in np.arange(8) * np.pi / 4:
                ox.append(r * np.cos(a)); oz.append(r * np.sin(a)); rr.append(r)
        x, z, hh = (np.ascontiguousarray(np.asarray(v, np.float64)).ravel() for v in np.broadcast_arrays(wx, wz, h))
        n = x.size
        out = cp.empty(n, cp.float64)
        if n:
            self.k['k_ao'](*self._grid(n), (self._ground(), self._d(x), self._d(z), self._d(hh), np.int64(n), self._d(ox),
                                            self._d(oz), self._d(np.array(rr, np.float64)), np.int32(len(rr)), out))
        return cp.asnumpy(out).reshape(np.shape(wx))

    def shadow(self, wx, wz, h, s, length=12., steps=20, objects=False):
        """render_live_view.sun_shadow over the shader surface, or (objects=True) over the surface with the overlay's
        object tops (gbuffer's lambda); step lengths by NumPy."""
        cp = self.cp
        ts = np.array([length * (k / steps) ** 1.5 for k in range(1, steps + 1)], np.float64)
        # h + .02 as sun_shadow adds it, in h's own dtype: object tops are float32 (2026-11-01: adding in float64 here put
        # 793 shadow values of v22b dunhuang 3.6e-7 off and one pixel off by 1)
        x, z, hh = (np.ascontiguousarray(np.asarray(v, np.float64)).ravel() for v in np.broadcast_arrays(wx, wz, np.asarray(h) + .02))
        n = x.size
        out = cp.empty(n, cp.float64)
        if n:
            self.k['k_shadow'](*self._grid(n), (self._ground(), self._d(x), self._d(z), self._d(hh), np.int64(n),
                                                np.float64(s[0]), np.float64(s[1]), np.float64(s[2]), self._d(ts),
                                                np.int32(steps), np.int32(3 if objects else 0), out))
        return cp.asnumpy(out).reshape(np.shape(wx))

    def free(self):
        """Release this ground's device memory (before the GPU lock is released)."""
        self.lf = self.land = self.water = self.wland = self.top = None
        self.mesh_src = self.win_src = self.ov_src = None
        self.cp.get_default_memory_pool().free_all_blocks()


_GROUND = np.dtype([('lf', np.uint64), ('lw', np.int32), ('lh', np.int32), ('per', np.float64), ('dx', np.float64),
                    ('dz', np.float64), ('level', np.float64), ('norm', np.float64), ('gain', np.float64),
                    ('land', np.uint64), ('has_mesh', np.int32), ('mx0', np.float64), ('mz0', np.float64),
                    ('mstep', np.float64), ('mnx', np.int32), ('mnz', np.int32),
                    ('water', np.uint64), ('has_win', np.int32), ('wx0', np.float64), ('wz0', np.float64),
                    ('wstep', np.float64), ('wnx', np.int32), ('wnz', np.int32), ('wland', np.uint64),
                    ('top', np.uint64), ('has_ov', np.int32), ('ox0', np.float64), ('oz0', np.float64), ('ocell', np.float64),
                    ('onx', np.int32), ('onz', np.int32)], align=True)

"""envelope_mp's edge-cost envelope on the GPU (CuPy), bit-identical, with an automatic CPU fallback.

create(oldheight, step_x, step_z, strip, plain_x, plain_z) -> GpuEnvelope, or None (CPU: envelope_mp)
GpuEnvelope.run(values, max_sweeps, reach) -> (out, sweeps) as envelope_mp, or None after any error
(the caller then runs that envelope, and every later one, on the CPU: same bytes).

The same sweeps over the same strips: per axis pass the dirty strips with a non-empty reach span,
the same per-strip improvement (any inf -> finite, all finite, max(target - updated) in float32)
and dirty marks of the other axis, the same 1e-7 stop. The reach spans are _reach_spans' (bits of
finite values per 32 lines, then each strip's first and last finite band sample on the device).
Each line of a strip is solved over its span in chunks of CK samples, a warp's 32 lines staged
through shared memory (coalesced loads and stores along rows and columns alike):
  coords   once per axis: the edge-cost coordinate (envelope_mp._cumulative_into: float32 costs
           max(step, |diff(old)|), step on plain edges, summed one by one in float64) at every
           CK-th sample. A chunk recomputes its own coordinates from that checkpoint with the
           same sequential sums, so they are the CPU's coordinates bit for bit (no 4 GB of
           float64 coordinates on the device).
  minima   per chunk min(t - c) and min(t + c) (float64, as numpy's V - c and V + c);
  scan     their exclusive running minima across the chunks of the line (min is exact, so the
           chunked order gives numpy's minimum.accumulate values);
  apply    forward B = min + c, backward V = min - c inside the chunk from those carries,
           updated = float32(min(B, V)), stats and dirty marks, written where the bits change.
float32(min(B, V)) is taken as the minimum of float32(B) and float32(V) (rounding is monotone);
a float32 tie of -0 and +0 takes -0, the rounding of the smaller double (B and V are never -0
themselves: c >= step > 0 past the first sample, c = +0 at it). Ties of equal doubles are equal
bits, and min is taken without NaN (NaN values go to the CPU).
Compiled by gpu_noise's NVRTC call (--fmad=false -ftz=false, its cubin cache); device work holds
gpu_noise's lock. Device memory: need() (2.37 GB for the 16380 x 16384 DDS), all allocated at
create(); the host keeps no copy beyond the result (no shared memory, no worker processes).
Not on the GPU (envelope_mp's CPU path instead): TW3K_GPU=0, no CuPy or device, compile error or
self-test mismatch (envelope_mp's own strip code run in process on small problems), float costs
other than float32, step <= 0, non-finite coordinates, NaN values, and less free VRAM than the
problem needs (need()). A GPU error ends the GPU path for the process (one log line).
"""
import os
import time

import numpy as np

import gpu_noise

CK = 32              # samples per chunk: one warp's width (the staged tiles)
BLOCK = 128          # lines per block (a multiple of 32: warps hold 32 consecutive lines)
_SRC = r'''
#define INF __longlong_as_double(0x7ff0000000000000LL)
#define CK %d
#define WARPS (%d / 32)
#define FULL 0xffffffffu

// Line j of an axis pass: sample i is element j*sj + i*si of the field, its edge (i-1 -> i) bit
// j*pj + (i-1)*pi of the plain mask (little-endian packed bits, padded by 8 bytes; none: null).
// Chunk b holds samples [b*CK, b*CK + CK); per-chunk arrays are [b][line]. A warp works on 32
// consecutive lines j0 + lane of one chunk.

// float32 cost max(step, |o - prev|) as np.maximum(step, |diff|); step on a plain edge.
__device__ __forceinline__ float edge_cost(float o, float prev, float step, bool plain) {
    float d = fabsf(o - prev);
    d = (step >= d) ? step : d;
    return plain ? step : d;
}

__device__ __forceinline__ bool plain_bit(const unsigned int* pw, long long e) {
    return pw && ((((const unsigned char*)pw)[e >> 3] >> (e & 7)) & 1);
}

// Bit r-1 (r = 1 .. nr-1): plain flag of the chunk's edge (r-1 -> r); e1 = bit of edge r = 1.
__device__ __forceinline__ unsigned int plain_bits(const unsigned int* __restrict__ pw, long long e1, long long pi, int nr) {
    if (!pw) return 0u;
    if (pi == 1) { long long w = e1 >> 5; return __funnelshift_r(pw[w], pw[w + 1], (unsigned int)(e1 & 31)); }
    unsigned int m = 0;
    for (int r = 1; r < nr; r++) m |= (unsigned int)plain_bit(pw, e1 + (long long)(r - 1) * pi) << (r - 1);
    return m;
}

// Stage the chunk of the warp's lines into T[line][r] (targets) and O[line][r] (original heights)
// with coalesced loads: along the lines when their samples are contiguous (si == 1, rows), else
// across them (columns). Without the tiles a row pass reads 32 rows per load instruction and
// thrashes L1 (a 5x slower pass).
__device__ __forceinline__ void stage(float (*T)[CK + 1], float (*O)[CK + 1], const float* __restrict__ out, const float* __restrict__ old,
                                      int n, int L, long long sj, long long si, int j0, int i0, int lane) {
    if (si == 1) {
#pragma unroll 8
        for (int q = 0; q < 32; q++) {
            int jj = j0 + q, i = i0 + lane;
            if (jj < n && i < L) { long long k = (long long)jj * sj + i; T[q][lane] = out[k]; O[q][lane] = old[k]; }
        }
    } else {
        int jj = j0 + lane;
#pragma unroll 8
        for (int r = 0; r < CK; r++) {
            int i = i0 + r;
            if (jj < n && i < L) { long long k = (long long)jj * sj + (long long)i * si; T[lane][r] = out[k]; O[lane][r] = old[k]; }
        }
    }
    __syncwarp();
}

// ck[b][j] = coordinate of sample b*CK (sequential float64 sum of float32 costs, from 0).
extern "C" __global__ void coords(const float* __restrict__ old, int n, int L, long long sj, long long si,
                                  const unsigned int* __restrict__ pw, long long pj, long long pi, float step,
                                  double* __restrict__ ck, unsigned int* bad) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    long long base = (long long)j * sj; double c = 0.0; float prev = old[base];
    ck[j] = 0.0;
    for (int i = 1; i < L; i++) {
        float o = old[base + (long long)i * si];
        c += (double)edge_cost(o, prev, step, plain_bit(pw, (long long)j * pj + (long long)(i - 1) * pi)); prev = o;
        if (i %% CK == 0) ck[(long long)(i / CK) * n + j] = c;
    }
    if (!isfinite(c)) atomicOr(bad, 1u);
}

// fmin[b][j] = min(t - c), bmin[b][j] = min(t + c) over the chunk's samples inside span [lo, hi).
extern "C" __global__ void __launch_bounds__(WARPS * 32) minima(const float* __restrict__ out, const float* __restrict__ old, int n, int L, long long sj, long long si,
                                  const unsigned int* __restrict__ pw, long long pj, long long pi, float step,
                                  const double* __restrict__ ck, const int* __restrict__ span,
                                  double* __restrict__ fmin, double* __restrict__ bmin, unsigned int* bad) {
    __shared__ float T[WARPS][32][CK + 1], O[WARPS][32][CK + 1];
    int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, j0 = blockIdx.x * blockDim.x + wp * 32, j = j0 + lane, b = blockIdx.y, i0 = b * CK;
    int lo = 0, hi = 0;
    if (j < n) { lo = span[j]; hi = span[n + j]; }
    bool act = j < n && lo < hi && i0 < hi && i0 + CK > lo;
    if (!__ballot_sync(FULL, act)) return;
    stage(T[wp], O[wp], out, old, n, L, sj, si, j0, i0, lane);
    if (!act) return;
    int i1 = min(i0 + CK, hi);
    unsigned int pm = plain_bits(pw, (long long)j * pj + (long long)i0 * pi, pi, min(CK, L - i0));
    long long cb = (long long)b * n + j;
    double c = ck[cb], f = INF, g = INF; float prev = O[wp][lane][0]; bool nan = false;
#pragma unroll
    for (int r = 0; r < CK; r++) {
        int i = i0 + r;
        if (i < i1) {
            if (r > 0) { float o = O[wp][lane][r]; c += (double)edge_cost(o, prev, step, (pm >> (r - 1)) & 1); prev = o; }
            if (i >= lo) {
                float t = T[wp][lane][r]; nan |= isnan(t);
                double a = (double)t - c, e = (double)t + c;
                f = (a < f) ? a : f; g = (e < g) ? e : g;
            }
        }
    }
    if (nan) atomicOr(bad, 2u);
    fmin[cb] = f; bmin[cb] = g;
}

// Exclusive running minima across the chunks of each line: fmin forward, bmin backward.
extern "C" __global__ void scan(int n, const int* __restrict__ span, double* __restrict__ fmin, double* __restrict__ bmin) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int lo = span[j], hi = span[n + j];
    if (lo >= hi) return;
    int b0 = lo / CK, b1 = (hi - 1) / CK; double r = INF;
    for (int b = b0; b <= b1; b++) { long long k = (long long)b * n + j; double v = fmin[k]; fmin[k] = r; r = (v < r) ? v : r; }
    r = INF;
    for (int b = b1; b >= b0; b--) { long long k = (long long)b * n + j; double v = bmin[k]; bmin[k] = r; r = (v < r) ? v : r; }
}

__device__ __forceinline__ unsigned int fkey(float f) {   // order-preserving float -> uint
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The chunk's update from its carries, written back (where the bits change) from the tile.
// stats: flag[strip] (1: inf -> finite, 2: target not finite), key[strip] (fkey of
// max(target - updated)), other[strip of the other axis] (a value changed).
// lanes: a warp's lines share one strip (strip %% 32 == 0), reduced before the atomics.
extern "C" __global__ void __launch_bounds__(WARPS * 32) apply(float* __restrict__ out, const float* __restrict__ old, int n, int L, long long sj, long long si,
                                 const unsigned int* __restrict__ pw, long long pj, long long pi, float step,
                                 const double* __restrict__ ck, const int* __restrict__ span,
                                 const double* __restrict__ fmin, const double* __restrict__ bmin,
                                 int strip, int lanes, unsigned int* __restrict__ flag, unsigned int* __restrict__ key,
                                 unsigned int* __restrict__ other) {
    __shared__ float T[WARPS][32][CK + 1], O[WARPS][32][CK + 1];
    int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, j0 = blockIdx.x * blockDim.x + wp * 32, j = j0 + lane, b = blockIdx.y, i0 = b * CK;
    int lo = 0, hi = 0;
    if (j < n) { lo = span[j]; hi = span[n + j]; }
    bool act = j < n && lo < hi && i0 < hi && i0 + CK > lo;
    if (!__ballot_sync(FULL, act)) return;
    stage(T[wp], O[wp], out, old, n, L, sj, si, j0, i0, lane);
    unsigned int fl = 0, wm = 0; float m = __int_as_float(0xff800000);
    if (act) {
        int i1 = min(i0 + CK, hi), ia = max(lo, i0);
        unsigned int pm = plain_bits(pw, (long long)j * pj + (long long)i0 * pi, pi, min(CK, L - i0));
        long long cb = (long long)b * n + j;
        double cs[CK]; float bf[CK];
        double c = ck[cb], run = fmin[cb]; float prev = O[wp][lane][0];
#pragma unroll
        for (int r = 0; r < CK; r++) {
            int i = i0 + r;
            if (i < i1) {
                if (r > 0) { float o = O[wp][lane][r]; c += (double)edge_cost(o, prev, step, (pm >> (r - 1)) & 1); prev = o; }
                cs[r] = c;
                if (i >= ia) { double a = (double)T[wp][lane][r] - c; run = (a < run) ? a : run; bf[r] = (float)(run + c); }
            }
        }
        run = bmin[cb]; int last = -1;
#pragma unroll
        for (int r = CK - 1; r >= 0; r--) {
            int i = i0 + r;
            if (i < i1 && i >= ia) {
                float t = T[wp][lane][r]; double e = (double)t + cs[r]; run = (e < run) ? e : run;
                float w = (float)(run - cs[r]), v = bf[r];
                float u = (v < w) ? v : (w < v) ? w : __uint_as_float(__float_as_uint(v) | __float_as_uint(w));
                if (!isfinite(t)) fl |= isfinite(u) ? 3u : 2u;
                float d = t - u; m = (d > m) ? d : m;
                if (u != t) { int s = i / strip; if (s != last) { other[s] = 1u; last = s; } }
                if (__float_as_uint(u) != __float_as_uint(t)) { T[wp][lane][r] = u; wm |= 1u << r; }
            }
        }
    }
    __syncwarp();
    if (si == 1) {
        for (int q = 0; q < 32; q++) {
            unsigned int mq = __shfl_sync(FULL, wm, q);
            if ((mq >> lane) & 1) out[(long long)(j0 + q) * sj + i0 + lane] = T[wp][q][lane];
        }
    } else if (wm) {
        for (int r = 0; r < CK; r++)
            if ((wm >> r) & 1) out[(long long)j * sj + (long long)(i0 + r) * si] = T[wp][lane][r];
    }
    unsigned int mk = act ? fkey(m) : 0u;
    if (lanes) {
        for (int o = 16; o; o >>= 1) { fl |= __shfl_xor_sync(FULL, fl, o); unsigned int x = __shfl_xor_sync(FULL, mk, o); mk = (x > mk) ? x : mk; }
        if (lane) return;
        j = j0;   // lane 0 reports for the warp (its lines share one strip)
    } else if (!act) return;
    int s = j / strip;
    if (fl) atomicOr(&flag[s], fl);
    atomicMax(&key[s], mk);
}

// Reach spans (envelope_mp._reach_spans on the device). B[g][i]: bit q = finite(line 32g + q,
// sample i). Lines across lanes (sj == 1: a ballot per warp, grid (lines, samples)) or along a
// loop (rows: grid (samples, groups)).
extern "C" __global__ void finite_bits(const float* __restrict__ out, int n, int L, long long sj, long long si, unsigned int* __restrict__ B) {
    if (sj == 1) {
        int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
        unsigned int m = __ballot_sync(FULL, j < n && isfinite(out[(long long)(j < n ? j : 0) + (long long)i * si]));
        if (!(threadIdx.x & 31) && j < n) B[(long long)(j >> 5) * L + i] = m;
    } else {
        int i = blockIdx.x * blockDim.x + threadIdx.x, g = blockIdx.y;
        if (i >= L) return;
        unsigned int m = 0;
        for (int q = 0; q < 32 && g * 32 + q < n; q++) m |= (unsigned int)(isfinite(out[(long long)(g * 32 + q) * sj + i]) ? 1 : 0) << q;
        B[(long long)g * L + i] = m;
    }
}

// first[s], last[s]: the first and last sample with a finite value in lines [start - R, stop + R)
// of strip s (clipped to the lines), from the bits B; untouched (0x7f7f7f7f, -1) without one.
extern "C" __global__ void band(const unsigned int* __restrict__ B, int n, int L, int strip, long long R, int* __restrict__ first, int* __restrict__ last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, s = blockIdx.y;
    long long start = (long long)s * strip, w0 = start - R, w1 = (start + strip < n ? start + strip : n) + R;
    w0 = w0 > 0 ? w0 : 0; w1 = w1 < n ? w1 : n;
    bool any = false;
    if (i < L && w0 < w1) {
        long long g0 = w0 >> 5, g1 = (w1 - 1) >> 5;
        for (long long g = g0; g <= g1 && !any; g++) {
            unsigned int w = B[g * L + i];
            if (g == g0) w &= FULL << (w0 & 31);
            if (g == g1) w &= FULL >> (31 - ((w1 - 1) & 31));
            any = w != 0;
        }
    }
    unsigned int m = __ballot_sync(FULL, any);
    if (!(threadIdx.x & 31) && m) {
        int base = i;   // lane 0's sample
        atomicMin(&first[s], base + __ffs(m) - 1); atomicMax(&last[s], base + 31 - __clz(m));
    }
}
''' % (CK, BLOCK)
_STATE = None        # None: not started; False: CPU for good; dict: CuPy and the kernels


def _log(msg):
    print(f'[gpu-envelope] {msg}', flush=True)


def _off(reason):
    global _STATE
    with gpu_noise._LOCK:
        if _STATE is not False:
            _log(f'CPU envelope from here on: {reason}')
        _STATE = False


def _why(e):
    return f'{type(e).__name__}: {" ".join(str(e).split())[:300]}'


def _start():
    global _STATE
    if os.environ.get('TW3K_GPU', '1') == '0':
        return _off('TW3K_GPU=0')
    busy = gpu_noise.claim()
    if busy:
        return _off(busy)
    t = time.perf_counter()
    try:
        cp, _, nvrtc, function = gpu_noise.cupy()
        cc = cp.cuda.Device(0).compute_capability
        mod, source = gpu_noise._module(nvrtc, function, cc, _SRC, 'gpu_envelope.cu')
        g = dict(cp=cp, module=mod, **{k: mod.get_function(k) for k in ('coords', 'minima', 'scan', 'apply', 'finite_bits', 'band')})
        _selftest(g)
        name = cp.cuda.runtime.getDeviceProperties(0)['name']
    except Exception as e:
        return _off(_why(e))
    _STATE = g
    _log(f'GPU envelope on {name.decode() if isinstance(name, bytes) else name} (sm_{cc}, {source}, self-test {time.perf_counter() - t:.1f}s)')


def _gpu():
    if _STATE is None:
        with gpu_noise._LOCK:
            if _STATE is None:
                _start()
    return _STATE or None


def need(shape, plain_x=None, plain_z=None):
    """Device bytes of a GpuEnvelope: old and out (float32), the packed plain masks, the
    checkpoints of both axes and one axis's chunk minima (float64), small buffers, a margin."""
    H, W = shape; nx, nz = H * -(-W // CK), W * -(-H // CK)
    plain = sum(-(-p.size // 32) * 4 + 8 for p in (plain_x, plain_z) if p is not None)
    return 8 * H * W + plain + 8 * (nx + nz) + 16 * max(nx, nz) + 16 * (H + W) + (64 << 20)


def _unsupported(oldheight, step_x, step_z, strip, plain_x, plain_z):
    """Why this problem stays on the CPU path (None: the GPU computes the same bytes)."""
    if oldheight.ndim != 2 or oldheight.dtype != np.float32 or min(oldheight.shape) < 2 or max(oldheight.shape) >= 1 << 30:
        return f'heights {oldheight.dtype} {oldheight.shape}'
    for step in (step_x, step_z):
        # the CPU's cost dtype (NEP 50: a Python float step stays float32)
        if np.maximum(step, np.abs(np.diff(oldheight[:1, :2], axis=1))).dtype != np.float32 or not np.float32(step) > 0 or not np.isfinite(np.float32(step)):
            return f'step {step!r}'
    H, W = oldheight.shape
    for p, shape in ((plain_x, (H, W - 1)), (plain_z, (H - 1, W))):
        if p is not None and (p.shape != shape or p.dtype != bool):
            return f'plain mask {p.dtype} {p.shape}'
    return None if int(strip) > 0 else f'strip {strip}'


def create(oldheight, step_x, step_z, strip=64, plain_x=None, plain_z=None):
    """A GpuEnvelope for this problem, or None (TW3K_GPU=0, no usable GPU, unsupported inputs,
    too little free VRAM: the caller uses the CPU envelope)."""
    g = _gpu()
    if g is None:
        return None
    why = _unsupported(oldheight, step_x, step_z, strip, plain_x, plain_z)
    if why:
        _log(f'CPU envelope for this problem: {why}'); return None
    cp = g['cp']; n = need(oldheight.shape, plain_x, plain_z)
    try:
        with gpu_noise._LOCK:
            cp.get_default_memory_pool().free_all_blocks()   # cached blocks of earlier GPU calls
            free = cp.cuda.runtime.memGetInfo()[0]
            if free < n:
                _log(f'CPU envelope for this problem: needs {n / 2**30:.2f} GB of VRAM, {free / 2**30:.2f} GB free'); return None
            return GpuEnvelope(g, oldheight, step_x, step_z, strip, plain_x, plain_z)
    except (_Unsupported, cp.cuda.memory.OutOfMemoryError) as e:   # this problem only (VRAM taken meanwhile)
        _log(f'CPU envelope for this problem: {_why(e)}')
    except Exception as e:
        _off(_why(e))
    cp.get_default_memory_pool().free_all_blocks()
    return None


class _Unsupported(Exception):
    pass


def _packed(p):
    """Little-endian packed bits in whole uint32 words plus one (the kernels' funnel-shift reads)."""
    b = np.packbits(p, axis=None, bitorder='little')
    return np.concatenate((b, np.zeros(-(-b.size // 4) * 4 + 8 - b.size, np.uint8)))


class GpuEnvelope:
    def __init__(self, g, oldheight, step_x, step_z, strip=64, plain_x=None, plain_z=None):
        cp = self.cp = g['cp']; self.g = g
        self.shape = H, W = oldheight.shape; self.strip = s = int(strip)
        self.step = {1: np.float32(step_x), 0: np.float32(step_z)}
        self.strips = {1: [(a, min(a + s, H)) for a in range(0, H, s)], 0: [(a, min(a + s, W)) for a in range(0, W, s)]}
        self.nck = {1: -(-W // CK), 0: -(-H // CK)}
        self.old = cp.asarray(np.ascontiguousarray(oldheight, np.float32)); self.out = cp.empty(self.shape, cp.float32)
        # line j, sample i: field j*sj + i*si; plain edge bit j*pj + (i-1)*pi
        # (plain_x (H, W-1): edge (j, i-1) -> (j, i); plain_z (H-1, W): edge (i-1, j) -> (i, j))
        self.addr = {1: (W, 1, W - 1, 1), 0: (1, W, 1, W)}
        self.plain = {axis: None if p is None else cp.asarray(_packed(p)) for axis, p in ((1, plain_x), (0, plain_z))}
        self.ck = {1: cp.empty(H * self.nck[1], cp.float64), 0: cp.empty(W * self.nck[0], cp.float64)}
        m = max(H * self.nck[1], W * self.nck[0]); self.fmin = cp.empty(m, cp.float64); self.bmin = cp.empty(m, cp.float64)
        self.span = cp.empty(2 * max(H, W), cp.int32); self.bands = cp.empty(2 * max(len(self.strips[1]), len(self.strips[0])), cp.int32)
        # stats: flag, key per strip of the pass axis, other-axis marks, bad
        self.ns = max(len(self.strips[1]), len(self.strips[0])); self.stats = cp.empty(3 * self.ns + 1, cp.uint32)
        self._zero()
        for axis in (1, 0):
            n, L = (H, W) if axis == 1 else (W, H); sj, si, pj, pi = self.addr[axis]
            g['coords'](((n + BLOCK - 1) // BLOCK,), (BLOCK,), (self.old, np.int32(n), np.int32(L), np.int64(sj), np.int64(si), *self._plain(axis),
                        np.int64(pj), np.int64(pi), self.step[axis], self.ck[axis], self.stats[-1:]))
        if int(self.stats[-1:].get()[0]):
            raise _Unsupported('non-finite edge-cost coordinates')

    def _plain(self, axis):
        p = self.plain[axis]
        return (p if p is not None else np.uint64(0),)   # a null pointer: no plain edges

    def _zero(self, a=None, value=0, size=None):
        a = self.stats if a is None else a
        self.cp.cuda.runtime.memsetAsync(a.data.ptr, value, a.nbytes if size is None else size, self.cp.cuda.get_current_stream().ptr)

    def _spans(self, radius):
        """envelope_mp._reach_spans(out, radius, strips) from the device copy of out (the bits of
        finite values per 32 lines in the chunk-minima buffer, then each strip's band ends)."""
        g = self.g; H, W = self.shape; spans = {}; B = self.fmin
        for axis in (1, 0):
            n, L = (H, W) if axis == 1 else (W, H); sj, si = self.addr[axis][:2]; ns = len(self.strips[axis])
            grid = ((n + 127) // 128, L) if sj == 1 else ((L + 127) // 128, -(-n // 32))
            g['finite_bits'](grid, (128,), (self.out, np.int32(n), np.int32(L), np.int64(sj), np.int64(si), B))
            first, last = self.bands[:ns], self.bands[ns:2 * ns]; self._zero(first, 0x7f); self._zero(last, 0xff)
            g['band'](((L + 127) // 128, ns), (128,), (B, np.int32(n), np.int32(L), np.int32(self.strip), np.int64(radius), first, last))
            h = self.bands[:2 * ns].get()
            spans[axis] = [(max(0, int(f) - radius), min(L, int(l) + radius + 1)) if l >= 0 else (0, 0) for f, l in zip(h[:ns], h[ns:])]
        return spans

    def _pass(self, axis, jobs, spans):
        """One axis pass over the strips `jobs`: (improvement per job, dirty strips of the other axis)."""
        g = self.g; H, W = self.shape; n, L = (H, W) if axis == 1 else (W, H); s = self.strip
        span = np.zeros(2 * n, np.int32)
        for i in jobs:
            a, b = self.strips[axis][i]; sl, sh = spans[i]
            span[a:b] = 0 if sl is None else sl; span[n + a:n + b] = L if sh is None else sh
        self.span[:2 * n].set(span); self._zero()
        sj, si, pj, pi = self.addr[axis]; ns, st = self.ns, self.stats
        line = (self.out, self.old, np.int32(n), np.int32(L), np.int64(sj), np.int64(si), *self._plain(axis), np.int64(pj), np.int64(pi), self.step[axis], self.ck[axis], self.span)
        grid = ((n + BLOCK - 1) // BLOCK, self.nck[axis])
        g['minima'](grid, (BLOCK,), line + (self.fmin, self.bmin, st[3 * ns:]))
        g['scan'](((n + 127) // 128,), (128,), (np.int32(n), self.span, self.fmin, self.bmin))
        g['apply'](grid, (BLOCK,), line + (self.fmin, self.bmin, np.int32(s), np.int32(s % 32 == 0), st[:ns], st[ns:2 * ns], st[2 * ns:3 * ns]))
        h = st.get()
        if h[-1]:
            raise _Unsupported('NaN values')
        flag, key = h[:ns], h[ns:2 * ns]
        k = np.where(key & 0x80000000, key & 0x7FFFFFFF, ~key).astype(np.uint32).view(np.float32)
        imps = [float('inf') if flag[i] & 1 else 0. if flag[i] & 2 else float(k[i]) for i in jobs]
        return imps, np.flatnonzero(h[2 * ns:2 * ns + len(self.strips[1 - axis])])

    def _run(self, values, max_sweeps, reach):
        strips = self.strips
        self.out.set(np.ascontiguousarray(values, dtype=np.float32))
        spans = {axis: [(None, None)] * len(strips[axis]) for axis in (1, 0)}
        if reach is not None:
            spans = self._spans(int(reach))
        dirty = {axis: np.array([sp[1] > sp[0] if sp[0] is not None else True for sp in spans[axis]]) for axis in (1, 0)}
        for sweep in range(max_sweeps):
            improvement = 0.
            for axis in (1, 0):
                todo = np.flatnonzero(dirty[axis]); dirty[axis][:] = False
                jobs = [i for i in todo if spans[axis][i][0] is None or spans[axis][i][1] > spans[axis][i][0]]
                if jobs:
                    imps, changed = self._pass(axis, jobs, spans[axis])
                    for imp in imps:
                        improvement = max(improvement, imp)
                    dirty[1 - axis][changed] = True
            if improvement <= 1e-7:
                break
        return self.out.get(), sweep + 1

    def run(self, values, max_sweeps=12, reach=None):
        """envelope_mp.Envelope.run (same bytes and sweep count), or None after an error (the GPU
        memory is released; a GPU error also ends the GPU path for the process)."""
        try:
            with gpu_noise._LOCK:
                return self._run(values, max_sweeps, reach)
        except _Unsupported as e:
            _log(f'CPU envelope for this problem: {e}')
        except Exception as e:
            _off(_why(e))
        self.close()
        return None

    def close(self):
        for k in ('old', 'out', 'plain', 'ck', 'fmin', 'bmin', 'span', 'bands', 'stats'):
            self.__dict__.pop(k, None)
        try:
            self.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


class _Serial:
    """envelope_mp's worker pool, in process (the self-test reference)."""
    def map(self, fn, jobs, chunksize=1):
        return map(fn, jobs)


def _reference(oldheight, step_x, step_z, strip, plain_x, plain_z, values, max_sweeps, reach):
    """envelope_mp.CpuEnvelope.run, its strip code run in this process."""
    import envelope_mp as E
    H, W = oldheight.shape; cx = np.empty((H, W)); cz = np.empty((H, W)); out = np.empty((H, W), np.float32)
    E._cumulative_into(cx, oldheight, step_x, 1, plain=plain_x); E._cumulative_into(cz, oldheight, step_z, 0, plain=plain_z)
    ref = E.CpuEnvelope.__new__(E.CpuEnvelope); ref.shape = (H, W); ref.strip = strip; ref.shm = {'out': (None, out)}; ref.pool = _Serial()
    saved = dict(E._W)
    try:
        E._W.update(out=(None, out), cx=(None, cx), cz=(None, cz))
        return E.CpuEnvelope.run(ref, values, max_sweeps, reach)
    finally:
        E._W.clear(); E._W.update(saved); E._BUF.clear()


def _selftest(g):
    """GPU vs envelope_mp's strip code on small problems: partial strips and chunks, flat
    ground (cost = step), plain edges, infinite seeds, -0, all-finite targets, reach spans,
    strips of 64 (warp-reduced stats) and 40."""
    import envelope_mp as E
    rng = np.random.default_rng(3); H, W = 150, 203
    old = np.cumsum(rng.normal(0, .05, (H, W)), 1, dtype=np.float32) + np.cumsum(rng.normal(0, .05, (H, 1)), 0, dtype=np.float32)
    old[40:70, 50:120] = old[40, 50]; old[90] = 0
    px = rng.random((H, W - 1)) < .03; pz = rng.random((H - 1, W)) < .03
    seeds = rng.random((H, W)) < .01
    pins = np.where(seeds, -old, np.inf).astype(np.float32); pins[90, ::7] = -0.
    proposed = (old + rng.normal(0, 2, (H, W))).astype(np.float32)
    sparse = np.where(rng.random((H, W)) < .002, old, np.inf).astype(np.float32); sparse[:, 150:] = np.inf
    for strip, values, reach, plain in ((64, pins, None, True), (64, pins, 30, True), (64, proposed, 25, False),
                                        (64, sparse, 12, True), (40, proposed, None, True), (40, sparse, 9, False)):
        args = (old, .031, .037, strip, px if plain else None, pz if plain else None)
        env = GpuEnvelope(g, *args)
        try:
            if reach is not None:
                env.out.set(values)
                if env._spans(reach) != E._reach_spans(values, reach, env.strips):
                    raise RuntimeError(f'self-test mismatch (spans, strip {strip}, reach {reach})')
            got = env._run(values, 12, reach)
        finally:
            env.close()
        ref = _reference(*args, values, 12, reach)
        if got[1] != ref[1] or not np.array_equal(got[0].view(np.uint32), ref[0].view(np.uint32)):
            raise RuntimeError(f'self-test mismatch (strip {strip}, reach {reach}, plain {plain})')

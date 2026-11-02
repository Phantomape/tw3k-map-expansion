"""Deterministic fractal value noise for authored campaign relief.

Periodic sine sums repeat in straight, evenly spaced bands that read as artificial
stripes on the campaign map. Value noise hashes integer lattice points, so it has
no period; smooth interpolation and a few octaves give natural-looking variation.

Coordinates are canvas hex coordinates measured from an origin (the authored
noise origin), so the pattern moves with the land when the canvas grows.
"""
import os
import threading

import numpy as np


def _lattice(ix, iy, seed):
    # Integer hash -> [0, 1); wrapping uint32 arithmetic is intended.
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF).astype(np.float32) / float(0x1000000)


def value_noise(x, y, seed=0):
    """Single octave in [0, 1) with smoothstep interpolation between lattice points."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _lattice(x0, y0, seed); b = _lattice(x0 + 1, y0, seed)
    c = _lattice(x0, y0 + 1, seed); d = _lattice(x0 + 1, y0 + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _value_noise_grid(x1, y1, seed):
    """value_noise on the grid x1 (columns) by y1 (rows): the lattice is hashed once per distinct
    lattice row/column instead of per pixel. Same float operations, bit-identical result."""
    x0, y0 = np.floor(x1), np.floor(y1)
    fx, fy = (x1 - x0).astype(np.float32), (y1 - y0).astype(np.float32)
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    ux, ix = np.unique(x0, return_inverse=True)
    uy, iy = np.unique(y0, return_inverse=True)
    a = _lattice(ux[None, :], uy[:, None], seed); b = _lattice(ux[None, :] + 1, uy[:, None], seed)
    c = _lattice(ux[None, :], uy[:, None] + 1, seed); d = _lattice(ux[None, :] + 1, uy[:, None] + 1, seed)
    if 2 * len(uy) > len(y1):
        # Lattice nearly per pixel row: the direct per-pixel gather is cheaper.
        at = (iy[:, None], ix[None, :])
        fx, fy = fx[None, :], fy[:, None]
        return (a[at] * (1 - fx) + b[at] * fx) * (1 - fy) + (c[at] * (1 - fx) + d[at] * fx) * fy
    fx, fy = fx[None, :], fy[:, None]
    # Separable: interpolate along x once per lattice row, then pick rows per pixel
    # row. Same float32 operations per pixel as
    # (a[at]*(1-fx)+b[at]*fx)*(1-fy)+(c[at]*(1-fx)+d[at]*fx)*fy with at=(iy[:,None],ix[None,:]).
    ab = a[:, ix] * (1 - fx) + b[:, ix] * fx
    cd = c[:, ix] * (1 - fx) + d[:, ix] * fx
    return ab[iy] * (1 - fy) + cd[iy] * fy


def _grid_axes(dq, dr):
    """(columns, rows) when dq varies only along columns and dr only along rows (mgrid/meshgrid
    inputs, the common case on the tile grid); otherwise None."""
    if dq.ndim != 2 or dr.shape != dq.shape or dq.size < 4096:
        return None
    if not (np.array_equal(dq, np.broadcast_to(dq[:1], dq.shape)) and np.array_equal(dr, np.broadcast_to(dr[:, :1], dr.shape))):
        return None
    return dq[0], dr[:, 0]


def fbm(q, r, origin, scale, octaves=4, seed=0):
    """Fractal sum of value noise in [-1, 1]; scale is the largest feature size in hexes."""
    dq = np.asarray(q, np.float64) - origin[0]
    dr = np.asarray(r, np.float64) - origin[1]
    axes = _grid_axes(dq, dr)
    if axes is not None and dq.shape[0] >= 128 and threading.current_thread() is threading.main_thread():
        # Rows are independent: blocks of rows on a thread pool (numpy releases the
        # GIL); every pixel gets the same float operations as the serial sum.
        total = np.empty(dq.shape, np.float32)
        n = dq.shape[0]; step = max(32, -(-n // (2 * _THREADS)))

        def work(r0):
            total[r0:r0 + step] = _fbm_grid(axes[0], axes[1][r0:r0 + step], scale, octaves, seed)
        list(_pool().map(work, range(0, n, step)))
        return total
    total = np.zeros(np.shape(dq), np.float32)
    amp, norm, s = 1.0, 0.0, float(scale)
    for o in range(octaves):
        if axes is not None:
            total += amp * (_value_noise_grid(axes[0] / s, axes[1] / s, seed + o * 101) * 2 - 1)
        else:
            total += amp * (value_noise(dq / s, dr / s, seed + o * 101) * 2 - 1)
        norm += amp
        amp *= .5
        s /= 2.03
    return total / norm


def _fbm_grid(xa, ya, scale, octaves, seed):
    total = np.zeros((len(ya), len(xa)), np.float32)
    amp, norm, s = 1.0, 0.0, float(scale)
    for o in range(octaves):
        total += amp * (_value_noise_grid(xa / s, ya / s, seed + o * 101) * 2 - 1)
        norm += amp
        amp *= .5
        s /= 2.03
    return total / norm


_THREADS = os.cpu_count() or 4
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=_THREADS)
    return _POOL


def ridged(q, r, origin, scale, octaves=4, seed=0):
    """Ridged fractal noise in [0, 1]: sharp crests, broad valleys, no regular spacing."""
    return np.clip(1 - np.abs(fbm(q, r, origin, scale, octaves, seed)) * 1.6, 0, 1) ** 2

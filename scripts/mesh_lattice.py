"""Error-bounded lattice steps for the coarse land meshes (mesh bake stage 1, pipeline-bake-design 3.2.1).

The replacement grid of fill_polished_mesh_gaps used one step per model, widened until the model fit 65k
vertices: canvas faces then stood up to 2.7 hexes apart and missed the LF between their vertices (canvas
|LF - mesh| p95 .49 after the graph, vanilla interior .12; speedup 2026-10-30, polish-split.json). Here every
cell of one absolute lattice (K half-hex units, the gap-fill lattice unit DX/2 = one tile) takes the coarsest
step r in K, K/2, .., 1 whose bilinear interpolation of the final LF stays within TOL, measured on the LF's own
pixels (4 per unit). The step is a function of the absolute cell and the LF only, so every model that covers a
cell triangulates it identically, and models split into several files stay seamless.

numpy only."""
import numpy as np

K = 16            # coarse cell, lattice units (8 hexes); steps K, K/2, .., 1 are nested on one absolute lattice
TOL = .10         # world units: p99 of |bilinear - LF| per cell (vanilla interior |LF - mesh| p95 is .12)
# Geometry budget (user decision pending, option C proposed 2026-10-30): cells touching a playable region keep TOL;
# cells wholly in non-playable regions or sea take TOL_NONPLAYABLE (79 % of the canvas geometry lies there; run
# 20261030-090643 at TOL everywhere: 15.5M land-mesh triangles, main 6.6M, vanilla 2.7M)
TOL_PLAYABLE = TOL
TOL_NONPLAYABLE = .30
TOL_MAX = 3.      # ... and the cell's maximum within TOL_MAX x TOL
PX = 4            # LF pixels per lattice unit (lf_height_map is 4 x the tile grid)
STEPS = tuple(K >> i for i in range(K.bit_length()))   # 16, 8, 4, 2, 1


def _to_height(raw, level, norm, gain):
    return level + (raw.astype(np.float32) / 65535 - norm) * gain


def cell_tolerance(playable_cell):
    """Per-cell tolerance from a bool array of cells touching a playable region."""
    return np.where(playable_cell, TOL_PLAYABLE, TOL_NONPLAYABLE)


def cell_steps(field, rect, level, norm, gain, k=K, tol=TOL, rows=8):
    """Per k-cell step (int64, shape (cells z, cells x)) for rect = (ux0, ux1, uz0, uz1) in lattice units, aligned to
    k. field: the LF, raw uint16, south-up (final_terrain_surface.load_final_surface), 4 px per unit. tol: a scalar or
    a per-cell array of the rect's shape (cell_tolerance). Cells beyond the LF take step 1 (never reached by land
    faces). Processed in bands of `rows` cell rows."""
    tol = np.broadcast_to(np.asarray(tol, np.float64), ((rect[3] - rect[2]) // k, (rect[1] - rect[0]) // k))
    ux0, ux1, uz0, uz1 = rect
    assert ux0 % k == 0 and ux1 % k == 0 and uz0 % k == 0 and uz1 % k == 0
    ncx, ncz = (ux1 - ux0) // k, (uz1 - uz0) // k
    h, w = field.shape
    out = np.ones((ncz, ncx), np.int64)
    n = k * PX                                      # px per cell side
    for c0 in range(0, ncz, rows):
        c1 = min(ncz, c0 + rows)
        z0, z1 = (uz0 + c0 * k) * PX, (uz0 + c1 * k) * PX
        x0, x1 = ux0 * PX, ux1 * PX
        # window with the far edge row / column (the last nodes); cells beyond the LF stay at step 1
        zz = np.clip(np.arange(z0, z1 + 1), 0, h - 1); xx = np.clip(np.arange(x0, x1 + 1), 0, w - 1)
        win = _to_height(np.asarray(field[zz][:, xx]), level, norm, gain)
        inside_z = (np.arange(z0, z1 + 1) < h); inside_x = (np.arange(x0, x1 + 1) < w)
        best = np.ones((c1 - c0, ncx), np.int64)
        done = np.zeros_like(best, bool)
        for r in STEPS:
            s = r * PX                              # px between nodes
            nodes = win[::s, ::s]                   # (nz+1, nx+1)
            f = np.arange(s, dtype=np.float32) / s
            # bilinear prediction of every px but the far edge, block by block: (bz, s, bx, s)
            a, b, c, d = nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, :-1], nodes[1:, 1:]
            fz, fx = f[None, :, None, None], f[None, None, None, :]
            pred = ((a[:, None, :, None] * (1 - fx) + b[:, None, :, None] * fx) * (1 - fz)
                    + (c[:, None, :, None] * (1 - fx) + d[:, None, :, None] * fx) * fz)
            bz, bx = nodes.shape[0] - 1, nodes.shape[1] - 1
            err = np.abs(pred.reshape(bz * s, bx * s) - win[:-1, :-1])
            # per k-cell statistics
            e = err.reshape(c1 - c0, n, ncx, n).transpose(0, 2, 1, 3).reshape(c1 - c0, ncx, n * n)
            t = tol[c0:c1]
            ok = (np.percentile(e, 99, axis=2) <= t) & (e.max(axis=2) <= TOL_MAX * t)
            take = ok & ~done
            best[take] = r; done |= ok
            del pred, err, e
        best[~done] = 1
        # cells wholly beyond the LF
        cz_in = inside_z[:-1].reshape(c1 - c0, n).any(axis=1); cx_in = inside_x[:-1].reshape(ncx, n).any(axis=1)
        best[~(cz_in[:, None] & cx_in[None, :])] = 1
        out[c0:c1] = best
    return out

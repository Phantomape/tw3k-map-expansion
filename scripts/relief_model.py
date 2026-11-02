"""Authored relief model shared by polish_campaign_terrain.py and preview_relief.py.

Coordinates are canvas hexes (q, r) as float arrays (the tile grid halves).
"""
import numpy as np

from gpu_noise import fbm, ridged   # terrain_noise's, same bytes (GPU for grid inputs, CPU fallback)

FOOTHILL_SPREAD = 1.6    # foothills reach this many range widths
FOOTHILL_SHARE = .12     # foothill height share of the range amplitude
VALLEY_FLOOR = .1        # mountain height share left at a town (it rises over TOWN_VALLEY hexes)
TOWN_VALLEY = (10, 40)   # hexes: full damping inside, none beyond


def line_distance(points, q, r):
    dist = np.full(q.shape, 1e6, np.float32)
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        t = np.clip(((q - ax) * (bx - ax) + (r - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2), 0, 1)
        dist = np.minimum(dist, np.hypot(q - ax - t * (bx - ax), r - ay - t * (by - ay)))
    return dist


def line_distance_near(points, q, r, margin):
    """line_distance, evaluated per segment only within `margin` (q and r) of the
    segment's bounding box on the regular q/r grid. Cells nearer than `margin` to
    the polyline get exactly line_distance's value; farther cells get a value of at
    least the true distance (>= margin). Falls back to line_distance off-grid."""
    win = _grid_windows(q, r)
    if win is None:
        return line_distance(points, q, r)
    q1, r1 = win
    dist = np.full(q.shape, 1e6, np.float32)
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        c0, c1 = np.searchsorted(q1, min(ax, bx) - margin), np.searchsorted(q1, max(ax, bx) + margin, 'right')
        y0, y1 = np.searchsorted(r1, min(ay, by) - margin), np.searchsorted(r1, max(ay, by) + margin, 'right')
        if c0 >= c1 or y0 >= y1:
            continue
        sl = (slice(y0, y1), slice(c0, c1)); qq, rr = q[sl], r[sl]
        t = np.clip(((qq - ax) * (bx - ax) + (rr - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2), 0, 1)
        dist[sl] = np.minimum(dist[sl], np.hypot(qq - ax - t * (bx - ax), rr - ay - t * (by - ay)))
    return dist


def town_clearance(q, r, towns):
    """Level-ground weight around town plots (1 at the plot, 0 beyond)."""
    clearance = np.zeros(q.shape, np.float32)
    for town in towns:
        tq, tr = town['town']
        d = np.hypot(q - tq, r - tr - (tq % 2) * .5)
        inner, outer = (8, 20) if town['is_capital'] else (5, 13)
        t = np.clip((d - inner) / (outer - inner), 0, 1)
        clearance = np.maximum(clearance, 1 - t * t * (3 - 2 * t))
    return clearance


def town_valleys(q, r, towns):
    """Mountain height factor: VALLEY_FLOOR near towns rising to 1 over TOWN_VALLEY,
    so towns sit in broad valleys instead of round clearings.

    The factor is exactly 1 wherever the nearest town is TOWN_VALLEY[1] hexes or
    more away, so on the regular q/r grid each town's distance is evaluated in
    a window of that radius only (identical result to _town_valleys_reference)."""
    win = _grid_windows(q, r)
    if win is None:
        return _town_valleys_reference(q, r, towns)
    q1, r1 = win
    nearest = np.full(q.shape, np.inf, np.float32)
    R = TOWN_VALLEY[1] + 1
    for town in towns:
        tq, tr = town['town']
        cy = tr + (tq % 2) * .5
        c0, c1 = np.searchsorted(q1, tq - R), np.searchsorted(q1, tq + R, 'right')
        y0, y1 = np.searchsorted(r1, cy - R), np.searchsorted(r1, cy + R, 'right')
        if c0 >= c1 or y0 >= y1:
            continue
        sl = (slice(y0, y1), slice(c0, c1))
        nearest[sl] = np.minimum(nearest[sl], np.hypot(q[sl] - tq, r[sl] - tr - (tq % 2) * .5))
    t = np.clip((nearest - TOWN_VALLEY[0]) / (TOWN_VALLEY[1] - TOWN_VALLEY[0]), 0, 1)
    return VALLEY_FLOOR + (1 - VALLEY_FLOOR) * t * t * (3 - 2 * t)


def _grid_windows(q, r):
    """(q per column, r per row) when q varies only along columns (increasing) and
    r only along rows (increasing); otherwise None."""
    if q.ndim != 2 or q.shape != r.shape:
        return None
    q1, r1 = q[0], r[:, 0]
    if not (np.all(np.diff(q1) > 0) and np.all(np.diff(r1) > 0)):
        return None
    if not (np.array_equal(q, np.broadcast_to(q1, q.shape)) and np.array_equal(r, np.broadcast_to(r1[:, None], r.shape))):
        return None
    return q1, r1


def _town_valleys_reference(q, r, towns):
    nearest = np.full(q.shape, np.inf, np.float32)
    for town in towns:
        tq, tr = town['town']
        nearest = np.minimum(nearest, np.hypot(q - tq, r - tr - (tq % 2) * .5))
    t = np.clip((nearest - TOWN_VALLEY[0]) / (TOWN_VALLEY[1] - TOWN_VALLEY[0]), 0, 1)
    return VALLEY_FLOOR + (1 - VALLEY_FLOOR) * t * t * (3 - 2 * t)


def add_massifs(relief, q, r, origin, features, valley_factor=None):
    """Add every relief feature as a massif, evaluated near its line only.

    A massif, not a wall: the crest wanders (domain-warped distance), the height
    rises and falls along the range, ridged noise carves the core into peaks and
    valleys, and lower ridged foothills extend FOOTHILL_SPREAD widths."""
    H, W = q.shape
    for i, (name, points, amplitude, width) in enumerate(features):
        pq = [p[0] for p in points]; pr = [p[1] for p in points]
        reach = (FOOTHILL_SPREAD * 3 + .5) * width
        cols = np.flatnonzero((q[0] >= min(pq) - reach) & (q[0] <= max(pq) + reach))
        rows = np.flatnonzero((r[:, 0] >= min(pr) - reach) & (r[:, 0] <= max(pr) + reach))
        if not len(cols) or not len(rows):
            continue
        sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        bq, br = q[sl], r[sl]
        warp = width * .35 * fbm(bq, br, origin, max(8., width * 1.5), 3, seed=61 + i)
        d = np.maximum(line_distance(points, bq, br) + warp, 0)
        core = np.exp(-(d / width) ** 2)
        foot = np.exp(-(d / (FOOTHILL_SPREAD * width)) ** 2)
        along = .65 + .35 * fbm(bq, br, origin, 40, 3, seed=67 + i)
        lift = amplitude * along * (core * (.45 + .55 * ridged(bq, br, origin, 11, 4, seed=23))
                                    + FOOTHILL_SHARE * foot * ridged(bq, br, origin, 4, 3, seed=71))
        if valley_factor is not None:
            lift = lift * valley_factor[sl]
        relief[sl] += lift
    return relief


def relief(q, r, origin, features, valleys, clearance, level, valley_factor=None):
    """Inland relief as polish builds it (before its coastal falloff)."""
    out = level + .16 * fbm(q, r, origin, 18, 4, seed=11)
    add_massifs(out, q, r, origin, features, valley_factor)
    out = out * (1 - clearance) + level * clearance
    for name, points, width in valleys:
        v = np.exp(-(line_distance(points, q, r) / width) ** 2) * .55 * (1 - clearance)
        out = out * (1 - v) + level * v
    return out

"""Polish's LAYOUT CLEANUP mask: ground earlier layouts changed gets geographic relief again.

Its input data/map-layout/layout-cleanup.npz (extract_layout_cleanup.py) holds the land of the pre-layout-v2 map and
the coast-revision-4 change mask. Both are the project's own historical layouts, so the public source snapshot leaves
the file out (user decision 2026-11-02, option c, source compliance). In the public tree (the PUBLIC_SOURCE marker
that make_release_snapshot writes at the root) a missing file skips the whole cleanup (all 0) with a WARN: a public
build then differs from the release outside the native map. In the private repository a missing file stays an error
(a silent skip once gave another LF, 2026-10-26). With the file the mask is the one polish always computed (byte for
byte; tests/test_layout_cleanup.py). After the release a rule on public inputs only replaces it (post-release #49).
"""
from pathlib import Path

import numpy as np

import map_layers as ml

NPZ = ml.LAYOUT_DIR / 'layout-cleanup.npz'
MARKER = Path(__file__).resolve().parents[1] / 'PUBLIC_SOURCE'     # make_release_snapshot.PUBLIC_SOURCE
WARN = ('WARN: {path} not found (the public source build has no layout cleanup data): layout cleanup skipped, '
        'all 0; outside the native map the terrain differs from the release')


def mask(shape, land, hy, hx, plan, q, r, blur, town_window, town_centre, path=NPZ, marker=MARKER):
    """(TH, TW) float32 layout cleanup weight. A missing `path` is all 0 with a WARN in the public tree (`marker`
    exists), else an error. land is the (H, W) land of the current map; hy, hx the tile -> hex index; blur,
    town_window, town_centre are polish's own."""
    out = np.zeros(shape, np.float32)
    path = Path(path)
    if not path.exists():
        if not Path(marker).exists():
            raise FileNotFoundError(f'{path} is missing (private build: the layout cleanup data is required; only the '
                                    f'public source tree, marked by {Path(marker).name}, skips it)')
        print(WARN.format(path=path), flush=True)
        return out
    z = ml.load_npz(path)
    H, W = land.shape
    if z['old_land'].shape != (H, W) or z['changed_coast'].shape != (H, W):
        raise RuntimeError(f'{path} is not on the {W}x{H} canvas')
    oldland = z['old_land'].astype(bool); changed_land = (oldland != land)[hy, hx]
    out = np.maximum(out, blur(changed_land.astype(float), 3)); out[changed_land] = 1
    for town in plan['regions']:
        for prior_town in town.get('cleanup_towns', [town.get('previous_town')]):
            if prior_town is None or prior_town == town['town']:
                continue
            tq, tr = prior_town; win = town_window(tq, tr, 16); sq, sr = town_centre(tq, tr)
            distance_old = np.hypot(q[win] - sq, r[win] - sr)
            t = np.clip((distance_old - 7) / 9, 0, 1); out[win] = np.maximum(out[win], 1 - t * t * (3 - 2 * t))
    if plan.get('coast_coverage_revision') == 4:
        changed = z['changed_coast'].astype(bool)[hy, hx]
        out = np.maximum(out, blur(changed.astype(float), 3)); out[changed] = 1
    return out

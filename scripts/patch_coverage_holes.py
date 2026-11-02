"""Close coarse-ground holes left by the seam repair, without re-tessellating.

audit_edited_ground_coverage.py marks edited half-hex samples that no coarse
top face covers. Re-running fill_polished_mesh_gaps.py after the seam repair
would re-tessellate whole edited areas and undo it; instead the hexes holding
those samples go through the seam repair's own exact partition, which also
keeps each mesh's compressed height map in step. They are filed as holes left
by removed mountain props: native coarse meshes are cut out under mountain
models, and hidden decorative mountains expose those cutouts, so restoring the
native triangles would restore the hole.

  python patch_coverage_holes.py [--variant 3k_dlc07_main_map]
"""
import argparse, functools, json, runpy, sys
from pathlib import Path

import numpy as np
from PIL import Image

import campaign_world as cw
import repair_local_coarse_seams
from repair_local_coarse_seams import CLASSES, repair

A = cw.WORKSPACE
S = A / 'work/release-input'


def run(script):
    # In this process rather than a subprocess: the audit, repair() and the
    # normalize pass share one set of imports and the lru-cached edit mask
    # (native_edit_mask.edited_hexes; LF pixels do not change in this step).
    argv, sys.argv = sys.argv, [str(script)]
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='3k_dlc07_main_map')
    args = ap.parse_args()
    here = Path(__file__).parent
    run(here / 'audit_edited_ground_coverage.py')
    missing = np.asarray(Image.open(A / f'terrain-repair/ground-coverage-missing-{args.variant}.png')) > 0
    if not missing.any():
        print(args.variant, 'no coverage holes')
        return
    ty, tx = np.nonzero(missing)
    hexes = np.zeros((cw.H, cw.W), np.uint8)
    q, r = cw.tile_hex(tx, ty)   # tile owner, as audit_ground_seams pools hexes
    hexes[r, q] = 1
    scan = A / f'analysis/coverage-hole-scan-{args.variant}'
    scan.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scan / 'ground-seam-masks.npz', **{k: (hexes if k == 'coarse_hole_prefab_removed' else np.zeros_like(hexes)) for k in CLASSES})
    (scan / 'ground-seam-scan.json').write_text(json.dumps({'variant': args.variant, 'stage': str(S),
                                                            'map': str(A / 'work/final-map/map.hex'),
                                                            'source': 'audit_edited_ground_coverage.py holes'}), encoding='utf-8')
    print(args.variant, 'coverage holes:', int(missing.sum()), 'samples in', int(hexes.sum()), 'hexes; repairing')
    # repair() and normalize both look up the native coarse reference raster
    # (every native mesh read from the packs and hashed for its cache key); it
    # depends on the vanilla packs only, so normalize reuses repair's (read-only).
    repair_local_coarse_seams.native_reference = functools.cache(repair_local_coarse_seams.native_reference)
    repair(args.variant, scan)
    # repair() clips the surrounding triangles at the cell boundaries, so the
    # cut vertices keep the old triangle's interpolated height. The seam loop
    # reseats them with normalize_coarse_partition_heights.py after every
    # repair pass; this pass needs the same, or the continuity audit fails.
    run(here / 'normalize_coarse_partition_heights.py')


if __name__ == '__main__':
    main()

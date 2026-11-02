"""Gate: native coarse land meshes must keep their crack-sealing skirts.

For every native land_mesh_*.rigid_model_v2 whose vertex count still matches
vanilla, count skirt vertices (a partner 1.0 below the surface in vanilla) whose
drop has shrunk in the stage. Collapsed skirts open cracks between coarse
terrain blocks that show the water plane as thin blue lines.

  python audit_coarse_skirts.py            # report; exit 1 when skirts collapsed
"""
import json, struct, sys
from pathlib import Path

import numpy as np

import campaign_world as cw
from coarse_mesh_skirts import skirt_offsets
from repair_campaign_river_roots import Assets
from campaign_world import TERRAIN_VARIANTS

S = cw.WORKSPACE / 'work/release-input'


def vertices(b):
    mp = struct.unpack_from('<I', b, 152)[0]
    _, _, _, vo, vc, io, ic = struct.unpack_from('<HH5I', b, mp)
    return np.frombuffer(b, '<f4', offset=mp + vo, count=vc * 4).reshape(vc, 4)


def faces(b):
    mp = struct.unpack_from('<I', b, 152)[0]
    _, _, _, vo, vc, io, ic = struct.unpack_from('<HH5I', b, mp)
    return np.frombuffer(b, '<u2', offset=mp + io, count=ic)


def collapsed_by_position(native_v, stage_v, native_f=None, stage_f=None):
    """Vanilla skirt positions present in the stage whose lowest vertex no longer drops below the top.

    With faces, skirts are found by topology: a regenerated surface point at the
    same (x, z) slightly above the native top must not hide an intact skirt."""
    off = skirt_offsets(native_v, native_f)
    want = {cw.lattice_key(x, z): float(o) for x, z, o in zip(native_v[:, 0], native_v[:, 2], off) if o < -1e-3}
    if not want:
        return 0
    vertical = np.ones(len(stage_v), bool)
    if stage_f is not None:
        # Points used only by faces with plan area are rebuilt surface.
        f = np.asarray(stage_f).reshape(-1, 3)
        t = np.asarray(stage_v[:, :3], np.float64)[f]
        area = np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 2] - t[:, 0, 2]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 2] - t[:, 0, 2]))
        vertical[:] = False
        vertical[f[area <= 1e-6].ravel()] = True
    drop, count, skirted = {}, {}, {}
    for i, (x, z, o) in enumerate(zip(stage_v[:, 0], stage_v[:, 2], skirt_offsets(stage_v, stage_f))):
        k = cw.lattice_key(x, z)
        if k in want:
            drop[k] = min(drop.get(k, 0.), float(o))
            count[k] = count.get(k, 0) + 1
            skirted[k] = skirted.get(k, False) or bool(vertical[i])
    # A single vertex, or (with faces) points carrying no vertical face, means
    # the skirt edge was rebuilt as continuous surface (canvas border,
    # regenerated grid, repair cells): no crack remains to seal. Flattened
    # skirt faces still count.
    return sum(1 for k, d in drop.items() if count[k] > 1 and skirted[k] and d > want[k] + 1e-3)


def main():
    native = Assets(original_only=True)
    report = {'variants': []}
    total = 0
    for variant in TERRAIN_VARIANTS:
        prefix = f'terrain/campaigns/{variant}/global_meshes/'
        rows, rebuilt = [], 0
        for key in sorted(k for k in native.index if k.startswith(prefix + 'land_mesh_') and k.endswith('.rigid_model_v2')):
            path = S / key
            if not path.exists():
                rows.append({'mesh': Path(key).name, 'missing_from_stage': True})
                total += 1
                continue
            nb, sb = native.read(key), path.read_bytes()
            nv, sv = vertices(nb).copy(), vertices(sb)
            if len(nv) != len(sv):
                rebuilt += 1          # rebuilt meshes: compared by position below
            nv[:, 0] += cw.NATIVE_SHIFT_RAW[0]
            nv[:, 2] += cw.NATIVE_SHIFT_RAW[1]
            bad = collapsed_by_position(nv, sv, faces(nb), faces(sb))
            if bad:
                rows.append({'mesh': path.name, 'collapsed_skirt_positions': bad, 'rebuilt': len(nv) != len(sv)})
            total += bad
        report['variants'].append({'variant': variant, 'meshes_with_collapsed_skirts': rows, 'rebuilt_meshes_skipped': rebuilt})
        print(variant, 'meshes with collapsed skirts:', len(rows), 'vertices:', sum(r.get('collapsed_skirt_positions', 0) for r in rows), 'rebuilt meshes:', rebuilt)
    (cw.WORKSPACE / 'analysis/coarse-skirt-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if total:
        print('FAIL: collapsed coarse-mesh skirts open cracks to the water plane')
        sys.exit(1)
    print('PASS: native coarse-mesh skirts intact')


if __name__ == '__main__':
    main()

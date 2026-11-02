"""Qinghai Lake water: one scaled vanilla lake prop at the lake's own height.

Vanilla precedent (the dlc07 "high lake", 19 hexes, LF bed ~4.9, shore ~6.2): the
water is not the ocean plane but a small_lake_1.wsmodel instance in its own scene,
  root    bmd_objects.3k_main_sea_lake.475.bin   (identity matrix, region 3k_main_sea_lake, BHM_IGNORE)
  parent  bmd_objects.3k_main_sea_lake.475.bin   (one child, the leaf below)
  leaf    bmd_objects.3k_main_sea_lake.475.31.bin  small_lake_1 at world (128.575, 5.969, 466.362),
          3x3 [-0.801,0,0.747 / 0,0.42,0 / -0.747,0,-0.801] (absolute world coordinates in the leaf).
This step clones those three records under an unused prefab number (LAKE_PREFAB) and
writes the leaf's instance matrix from analysis/hehuang-patch.npz `prop`: an
axis-aligned (diagonal) scale, so the row/column matrix conventions agree, centred
on the lake hexes + 1 hex at y = water. The terrain under the rectangle is held above
the water by hehuang_patch.py; the plane shows only over the lake bed.

Runs in the prefab chain after translate_campaign_lake_instances.py (which treats
this leaf as authored: hehuang_lake_prop.AUTHORED_LEAVES) on the staged
global_props.bin, rebuilt from terrain-before-polish by reconcile_terrain_prefabs
each run. Also writes the loose root, parent and leaf files.

  python hehuang_lake_prop.py [--apply]
"""
import argparse
import json
import struct
import sys

import campaign_world as cw
import props_bundle as pb
from read_root_prefabs import decode

A = cw.WORKSPACE
S = A / 'work/release-input'
TEMPLATE = 475
LAKE_PREFAB = 9475
LAKE_REGION = '3k_main_sea_lake'


def keys(variant, number):
    pre = f'terrain/campaigns/{variant}/bmd_objects.{LAKE_REGION}.{number}'
    return pre + '.bin', pre + '.31.bin'


AUTHORED_LEAVES = {keys(v, LAKE_PREFAB)[1] for v in cw.TERRAIN_VARIANTS}


def run(apply):
    from hehuang_terrain import load
    from translate_campaign_lake_instances import records as leaf_records   # not at import: it imports AUTHORED_LEAVES from here
    patch = load()
    if patch is None or patch.prop is None:   # korea (or another patch) alone has no lake prop
        print('No Hehuang patch (analysis/hehuang-patch.npz): nothing to place', flush=True)
        return None
    prop = patch.prop
    report = {'prefab': LAKE_PREFAB, 'template': TEMPLATE, 'prop': prop, 'variants': []}
    for variant in cw.TERRAIN_VARIANTS:
        pre = f'terrain/campaigns/{variant}/'
        rootkey = pre + 'bmd_objects.bin'
        es, assets = pb.unbundle((S / (pre + 'global_props.bin')).read_bytes(), rootkey)
        root = assets[rootkey]
        cp, rs = decode(root)
        t_parent, t_leaf = keys(variant, TEMPLATE)
        n_parent, n_leaf = keys(variant, LAKE_PREFAB)
        template = [r for r in rs if r['path'] == t_parent]
        assert len(template) == 1 and template[0]['region'] == LAKE_REGION and template[0]['height_mode'] == 'BHM_IGNORE', 'vanilla high-lake root missing'
        t = template[0]
        # Staged roots carry the native shift as translation; leaves are placed in world
        # space (the engine ignores root translations), so the clone keeps the template's.
        assert max(abs(a - b) for a, b in zip(t['matrix'][:12], (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0))) < 1e-6, 'template root 3x3 is not identity'
        rows = leaf_records(assets[t_leaf])
        assert len(rows) == 1 and rows[0]['model'] == prop['model'], ('template leaf', [r['model'] for r in rows])
        # Leaf: the template leaf with the Hehuang matrix (12 floats: 3x3 then position).
        leaf = bytearray(assets[t_leaf])
        moff = rows[0]['position_offset'] - 36
        struct.pack_into('<12f', leaf, moff, *prop['matrix'])
        check = leaf_records(bytes(leaf))
        assert len(check) == 1 and max(abs(a - b) for a, b in zip(check[0]['matrix'], prop['matrix'])) < 1e-3
        # Parent: the template parent, its one child renamed.
        pbody = assets[t_parent]
        _, children = decode(pbody)
        assert len(children) == 1 and children[0]['path'] == t_leaf
        c = children[0]
        parent = pbody[:c['start']] + pb.rename(pbody[c['start']:c['end']], n_leaf) + pbody[c['end']:]
        _, pc = decode(parent)
        assert [x['path'] for x in pc] == [n_leaf]
        new = {n_parent: parent, n_leaf: bytes(leaf)}
        # Root list: the template record renamed, appended (or replaced on a rerun).
        rec = pb.rename(root[t['start']:t['end']], n_parent)
        existing = [r for r in rs if r['path'] == n_parent]
        updated, chk = pb.prefab_list(root, cp, rs, [r for r in rs if r['path'] != n_parent], [rec])
        assert len(chk) == len(rs) + (0 if existing else 1) and chk[-1]['path'] == n_parent
        bundled = pb.rebundle(es, assets, rootkey, updated, new, replace=True)
        report['variants'].append({'variant': variant, 'root': n_parent, 'leaf': n_leaf, 'rerun_replaced': bool(existing),
                                   'bundle_assets': struct.unpack_from('<I', bundled)[0], 'world_rect_xz': prop['world_rect'], 'water_y': prop['water']})
        if apply:
            pb.write_stage(S, pre, bundled, updated, new)
        print(variant, 'Hehuang lake prop', n_leaf, 'y', round(prop['water'], 3), 'rect', [round(v, 2) for v in prop['world_rect']], 'applied' if apply else 'dry run', flush=True)
    report['applied'] = apply
    report['live_verified'] = False
    if apply:
        (A / 'analysis/hehuang-lake-prop.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    run(ap.parse_args().apply)


if __name__ == '__main__':
    main()

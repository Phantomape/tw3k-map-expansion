"""Water surfaces of the frontier rivers and lakes as vanilla river-model props.

In game only meshes draw water: lf_sea_height_map alone draws nothing, so the frontier river channels
polish carves (RIVER CHANNELS, analysis/river-water-level.npy) showed as green troughs, and frontier
lakes and river beds under the campaign water plane showed the ocean background plane as flat plates.
Vanilla's Yellow River, Yangtze and Xiang are type-68 river models (river_water_codec) under 23 roots
in global_props.bin; vanilla lakes are lake models at the lake's height. This step builds the same for
the frontier (river_water_mesh: one continuous surface on the tile-cell grid, river level on the band,
lakes and mouths at FLOOR) and adds it to the staged bundle:

  models   terrain/campaigns/<v>/models/tw3k_water_<n>.wsmodel (+ .rigid_model_v2): vanilla river
           header (river_11), material campaign_navigable_river_plane_default (rivers, mouths) or
           3k_main_campaign_lakes[_south] (lakes)
  props    one scene per model as vanilla's river scenes (template bmd_objects.
           3k_main_riv_sea_yangtze_to_shangyong_resource_2.0: root BHM_IGNORE identity, parent with one
           child .16, leaf with one identity instance; the engine places river models by their pivot,
           repair_campaign_river_roots) under prefab numbers PREFAB0..PREFAB1, owned by the river's
           tw3k_riv_sea_* region (lakes: 3k_main_sea_lake)

Runs in the prefab chain after hehuang-lake (the final LF and polish's river level exist). A rerun
replaces its own scenes and models; models and loose scenes of an earlier run that this run does not
write again move to terrain-repair/river-water-stale/<run>/ (never deleted). Reads the river level
only with polish's stamp (map.hex sha256 + canvas), as sync_campaign_sea_height_texture does.

  python river_water_props.py [--apply] [--report PATH] [--keep-stale]
"""
import tw3k_paths
import argparse
import ast
import hashlib
import json
import re
import shutil
import struct
import time
from pathlib import Path

import numpy as np

import campaign_world as cw
import map_hex
import props_bundle as pb
import river_water_codec as C
import river_water_mesh as M
from analyze_reference import digest, inventory
from hex_components import hex_label
from final_terrain_surface import load_final_surface
from read_root_prefabs import decode
from translate_campaign_lake_instances import records as leaf_records

A = cw.WORKSPACE
S = A / 'work/release-input'
LEVEL_FILE = A / 'analysis/river-water-level.npy'
GAME = tw3k_paths.GAME_DATA / 'terrain.pack'
PREFAB0, PREFAB1 = 9000, 9399
TEMPLATE_REGION = '3k_main_riv_sea_yangtze_to_shangyong_resource_2'
TEMPLATE_NUMBER = 0
TEMPLATE_CHILD = 16
TEMPLATE_MODEL = 'terrain/campaigns/3k_dlc07_main_map/models/river_11.wsmodel.rigid_model_v2'   # campaign_navigable_river_plane_default
LAKE_REGION = '3k_main_sea_lake'
RIVER_PREFIX = 'tw3k_riv_sea_'
MODEL_STEM = 'models/tw3k_water_'
SOUTH_LAKE_Z = cw.NATIVE_SHIFT_WORLD[1] + 261.   # vanilla world z ~261 parts _south lakes (Dongting 233, Poyang 247) from the others (Tai 284+)
# Other prefab chain steps' number ranges (removal ranges as their own reruns match them), read from
# their sources without importing them; ours must be disjoint from every one.
OTHER_RANGES = (('place_ambient.py', 'PREFAB0', 'PREFAB1', 0), ('frontier_mountain_props.py', 'FIRST_PREFAB', None, 899),
                ('coast_props.py', 'WAVE_BASE', None, 999), ('coast_props.py', 'DOCK_BASE', 'DOCK_MAX', 0),
                ('hehuang_lake_prop.py', 'LAKE_PREFAB', None, 0), ('landmarks_props.py', 'PREFAB_BASE', 'PREFAB_MAX', 0))


def constant(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
                if isinstance(t, ast.Tuple) and name in [e.id for e in t.elts if isinstance(e, ast.Name)]:
                    vals = ast.literal_eval(node.value)
                    return vals[[e.id for e in t.elts].index(name)]
    raise KeyError(f'{name} not in {path.name}')


def other_ranges():
    here = Path(__file__).resolve().parent
    out = []
    for f, lo, hi, span in OTHER_RANGES:
        a = int(constant(here / f, lo))
        b = int(constant(here / f, hi)) if hi else a + span
        out.append((f'{f}:{lo}', a, b))
    return out


def check_ranges(rs):
    """Our range is disjoint from every other step's and from vanilla's roots."""
    ranges = other_ranges()
    for name, a, b in ranges:
        assert b < PREFAB0 or a > PREFAB1, f'prefab range {PREFAB0}-{PREFAB1} overlaps {name} {a}-{b}'
    vanilla = [int(m.group(1)) for r in rs if not r['region'].startswith('tw3k_')
               for m in [re.search(r'\.(\d+)\.bin$', r['path'])] if m and not any(a <= int(m.group(1)) <= b for _, a, b in ranges)
               and not PREFAB0 <= int(m.group(1)) <= PREFAB1]
    assert max(vanilla) < PREFAB0, ('vanilla prefab numbers reach', max(vanilla))
    return {'ours': [PREFAB0, PREFAB1], 'others': [[n, a, b] for n, a, b in ranges], 'vanilla_max': max(vanilla)}


def level_checked():
    """polish's river level, refused unless its stamp names this map.hex and canvas."""
    stamp = LEVEL_FILE.with_suffix('.json')
    want = {'map_sha256': digest(A / 'work/final-map/map.hex'), 'canvas': cw.canvas_stamp()}
    found = json.loads(stamp.read_text(encoding='utf-8')) if stamp.exists() else None
    if found != want:
        raise SystemExit(f'{LEVEL_FILE} was drawn on {found or "an unstamped map"}, current is {want}: rerun polish_campaign_terrain.py')
    lvl = np.load(LEVEL_FILE)
    assert lvl.shape == (cw.TILE_H, cw.TILE_W), (lvl.shape, 'river level from another canvas')
    return lvl, want


def hex_masks(m):
    c = m['cells']
    names = m['land_regions'] + m['sea_regions']
    rid = map_hex.region_ids(c)
    sea = (c[..., 0] & 3) == 1
    outer = np.ones(sea.shape, bool)
    outer[cw.NATIVE_HEX_SLICE] = False
    rk = [i for i, n in enumerate(names) if n.startswith(RIVER_PREFIX)]
    river = np.isin(rid, rk) & sea & outer
    # Lakes: frontier open water (hex-connected, rivers apart) holding 3k_main_sea_lake hexes and
    # reaching neither the ocean (tw3k_bulk_sea_*) nor the canvas edge. Real lakes mix lake hexes with
    # 3k_main_sea_non_playable ones (Irtysh lakes: islands of the latter inside the lake).
    open_ = sea & outer & ~river
    lab = hex_label(open_)
    n = int(lab.max())
    has_lake = np.zeros(n + 1, bool)
    if LAKE_REGION in names:
        has_lake[np.unique(lab[open_ & (rid == names.index(LAKE_REGION))])] = True
    ocean = np.zeros(n + 1, bool)
    ocean[np.unique(lab[open_ & np.isin(rid, [i for i, x in enumerate(names) if x.startswith('tw3k_bulk_sea_')])])] = True
    edge = np.zeros_like(open_)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    ocean[np.unique(lab[open_ & edge])] = True
    keep = has_lake & ~ocean
    keep[0] = False
    lake = keep[lab]
    other = sea & ~river & ~lake
    return names, rid, river, lake, other


def lake_floor():
    """river_water_mesh.lake_surface (floor, pad) from polish's lake water levels (lake_level product and field,
    TW3K_LAKE_LEVEL=1; refused unless solved on this map.hex); (FLOOR, None) (every lake on the one water plane, as
    before) with it off."""
    import lake_level
    if not lake_level.enabled():
        return M.FLOOR, None
    d = lake_level.read_product(A, digest(A / 'work/final-map/map.hex'))
    return M.lake_surface(d, lake_level.read_fields(A, d))


def surface(field, lvl, m, floor=None):
    """river_water_mesh plan + chain + per-vertex river uv (a dict of everything the models need).
    floor: (floor, pad) of river_water_mesh.lake_surface (None: lake_floor())."""
    names, rid, river, lake, other = hex_masks(m)
    lf_c, lf_lo = M.lf_cells(field)
    F, pad = lake_floor() if floor is None else floor
    p = M.plan(lvl, lf_c, lf_lo, river, lake, other, cap=False, floor=F, lake_pad=pad)
    p.update(names=names, rid=rid, river_hex=river, lake_hex=lake, sea_hex=other, lf_centre=lf_c, lf_low=lf_lo)
    ch = M.chain(river, p, lake, other)
    U = M.flow_directions(ch)
    p.update(chain=ch, U=U)
    return M.river_surface(p, lvl, ch, U)


def uv_lookup(p):
    """(river uv0, lake uv0, river tangent) lookups by lattice vertex (ty, tx)."""
    keys, uv, T = p['uv_keys'], p['uv'], p['tangent']
    W1 = cw.TILE_W + 1

    def at(vy, vx):
        k = vy.astype(np.int64) * W1 + vx
        i = np.searchsorted(keys, k)
        assert (keys[np.minimum(i, len(keys) - 1)] == k).all()
        return i

    def river(vy, vx):
        return uv[at(vy, vx)]

    def tangent(vy, vx):
        return T[at(vy, vx)]

    def lake(vy, vx):
        wx, wz = M.cell_world(vy, vx)
        return np.stack([wx, wz], 1) * M.UV_PER_WORLD
    return river, lake, tangent


def owner_of(p, qy, qx, kind):
    """Owning region of a chunk: lakes 3k_main_sea_lake; rivers the tw3k_riv_sea_* region with most
    of the chunk's quads, else the region of the nearest river hex."""
    if kind == 'lake':
        return LAKE_REGION
    q = qx // 2
    r = (qy - q % 2) // 2
    names, rid, river = p['names'], p['rid'], p['river_hex']
    ok = river[np.clip(r, 0, river.shape[0] - 1), np.clip(q, 0, river.shape[1] - 1)]
    if ok.any():
        ids, counts = np.unique(rid[r[ok], q[ok]], return_counts=True)
        best = ids[np.lexsort((ids, -counts))[0]]
        return names[int(best)]
    rr, qq = np.nonzero(river)
    cx, cz = M.hex_centre(qq, rr)
    wx, wz = M.cell_world(qy.mean(), qx.mean())
    i = int(np.argmin((cx - wx) ** 2 + (cz - wz) ** 2))
    return names[int(rid[rr[i], qq[i]])]


def models(p, template):
    """[(name, kind, owner, model dict, world vertices, triangles, material)] in prefab order."""
    river_uv, lake_uv, river_t = uv_lookup(p)
    groups = M.chunks(p['river_quad'] | p['mouth_quad'], 'river') + M.chunks(p['lake_quad'], 'lake')
    groups.sort(key=lambda g: (g[1] != 'river', g[0]))
    out = []
    for i, (tile, kind, qy, qx) in enumerate(groups):
        m, world, tris = M.build_model(template, qy, qx, p['H'], river_uv if kind == 'river' else lake_uv, kind,
                                       H_shade=p['H_shade'], tangent_of=river_t if kind == 'river' else None)
        if kind == 'river':
            material = M.RIVER_MATERIAL
        else:
            material = M.LAKE_SOUTH_MATERIAL if world[:, 2].mean() < SOUTH_LAKE_Z else M.LAKE_MATERIAL
        out.append((f'{i:03d}', kind, owner_of(p, qy, qx, kind), m, world, tris, material, tile))
    assert len(out) <= PREFAB1 - PREFAB0 + 1, f'{len(out)} water models exceed prefab range {PREFAB0}-{PREFAB1}'
    return out


def template_model():
    entries = {e['path']: e for e in inventory(GAME, False)['entries']}
    e = entries[TEMPLATE_MODEL]
    with GAME.open('rb') as f:
        f.seek(e['offset'])
        data = f.read(e['size'])
    m = C.decode(data)
    assert C.encode(m) == data
    return {k: v for k, v in m.items() if k not in ('vertices', 'triangles')}


def ours_re(pre):
    return re.compile(re.escape(pre) + r'bmd_objects\.[A-Za-z0-9_]+\.(\d+)(?:\.\d+)?\.bin$')


def stale_files(pre, keep):
    """Loose files of an earlier run (our models and scenes) that this run does not write."""
    base = S / pre
    pat = ours_re(pre)
    found = []
    if (base / 'models').exists():
        found += [p for p in (base / 'models').glob('tw3k_water_*') if p.is_file()]
    for p in base.glob('bmd_objects.*.bin'):
        mm = pat.match(pre + p.name)
        if mm and PREFAB0 <= int(mm.group(1)) <= PREFAB1:
            found.append(p)
    return sorted(p for p in found if (pre + p.relative_to(base).as_posix()) not in keep)


def run(apply, report_path, keep_stale=False):
    lvl, stamp = level_checked()
    m = map_hex.read(A / 'work/final-map/map.hex')
    template = template_model()
    report = {'kind': 'river-water-props', 'prefab_range': [PREFAB0, PREFAB1], 'level_stamp': stamp,
              'inputs': {'river_level_sha256': digest(LEVEL_FILE), 'map_sha256': stamp['map_sha256']},
              'frame': 'vertex y = LF height (vanilla river models: ribbon edges ~0.15 under the LF, centre 0.5-2.3 above the bed)',
              'floor': M.FLOOR, 'variants': [], 'applied': apply, 'live_verified': False}
    for variant in cw.TERRAIN_VARIANTS:
        pre = f'terrain/campaigns/{variant}/'
        rootkey = pre + 'bmd_objects.bin'
        lf_path = S / pre / 'lf_height_map.dds'
        st = lf_path.stat()
        field = load_final_surface(variant, S)
        p = surface(field, lvl, m)
        built = models(p, template)
        es, assets = pb.unbundle((S / (pre + 'global_props.bin')).read_bytes(), rootkey)
        root = assets[rootkey]
        cp, rs = decode(root)
        ranges = check_ranges(rs)
        pat = ours_re(pre)

        def is_ours(k):
            mm = pat.match(k)
            return bool(mm) and PREFAB0 <= int(mm.group(1)) <= PREFAB1
        old = [r for r in rs if is_ours(r['path'])]
        for r in old:   # only our own scenes live in our range
            for kid in decode(assets[r['path']])[1]:
                rows = leaf_records(assets[kid['path']])
                assert rows and all(x['model'].startswith(pre + MODEL_STEM) for x in rows), (kid['path'], [x['model'] for x in rows])
        t_parent = f'{pre}bmd_objects.{TEMPLATE_REGION}.{TEMPLATE_NUMBER}.bin'
        t_leaf = t_parent[:-4] + f'.{TEMPLATE_CHILD}.bin'
        t_root = [r for r in rs if r['path'] == t_parent]
        assert len(t_root) == 1 and t_root[0]['height_mode'] == 'BHM_IGNORE', 'vanilla river scene template missing'
        t_root = t_root[0]
        assert max(abs(a - b) for a, b in zip(t_root['matrix'], (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))) < 1e-6, 'template root is not identity'
        pbody = assets[t_parent]
        _, children = decode(pbody)
        assert [c['path'] for c in children] == [t_leaf]
        child = children[0]
        lbody = assets[t_leaf]
        lrows = leaf_records(lbody)
        assert len(lrows) == 1 and lrows[0]['model'].endswith('/models/river_13.wsmodel')
        assert max(abs(a - b) for a, b in zip(lrows[0]['matrix'], (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0))) < 1e-6
        mold = lrows[0]['model'].encode()
        mpos = lbody.index(struct.pack('<H', len(mold)) + mold)
        assert lbody.count(mold) == 1
        new, recs, scenes, files = {}, [], [], {}
        for i, (name, kind, owner, model, world, tris, material, tile) in enumerate(built):
            number = PREFAB0 + i
            parent_path = f'{pre}bmd_objects.{owner}.{number}.bin'
            leaf_path = parent_path[:-4] + f'.{TEMPLATE_CHILD}.bin'
            assert parent_path not in assets or is_ours(parent_path)
            wsm = f'{pre}{MODEL_STEM}{name}.wsmodel'
            geo = wsm + '.rigid_model_v2'
            mb = wsm.encode()
            leaf = lbody[:mpos] + struct.pack('<H', len(mb)) + mb + lbody[mpos + 2 + len(mold):]
            chk = leaf_records(leaf)
            assert len(chk) == 1 and chk[0]['model'] == wsm and chk[0]['matrix'] == lrows[0]['matrix']
            parent = pbody[:child['start']] + pb.rebind(pbody[child['start']:child['end']], child, owner, leaf_path) + pbody[child['end']:]
            _, pc = decode(parent)
            assert [(x['path'], x['region']) for x in pc] == [(leaf_path, owner)]
            new[parent_path] = parent
            new[leaf_path] = leaf
            recs.append(pb.rebind(root[t_root['start']:t_root['end']], t_root, owner, parent_path))
            data = C.encode(model)
            again = C.decode(data)
            assert np.array_equal(again['vertices'], model['vertices']) and np.array_equal(again['triangles'], model['triangles'])
            files[geo] = data
            files[wsm] = C.wsmodel(geo, material)
            v = model['vertices']
            scenes.append({'model': wsm, 'kind': kind, 'owner': owner, 'prefab': number, 'tile': list(tile), 'material': material,
                           'vertices': int(len(v)), 'triangles': int(len(tris)), 'bytes': len(data),
                           'bounds': [round(x, 3) for x in model['bounds']],
                           'y_range': [round(float(world[:, 1].min()), 3), round(float(world[:, 1].max()), 3)],
                           'sha256': hashlib.sha256(data).hexdigest()})
        kept = [r for r in rs if not is_ours(r['path'])]
        updated, chk = pb.prefab_list(root, cp, rs, kept, recs)
        assert [c['path'] for c in chk[len(kept):]] == [k for k in new if not k.endswith(f'.{TEMPLATE_CHILD}.bin')]
        bundled = pb.rebundle(es, {k: v for k, v in assets.items() if not is_ours(k)}, rootkey, updated, new)
        writes = set(new) | set(files)
        stale = stale_files(pre, writes)
        vr = {'variant': variant, 'lf': {'path': str(lf_path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
              'ranges': ranges, 'models': len(built), 'rerun_removed_roots': len(old), 'roots_added': len(recs),
              'river_models': sum(s['kind'] == 'river' for s in scenes), 'lake_models': sum(s['kind'] == 'lake' for s in scenes),
              'vertices': sum(s['vertices'] for s in scenes), 'triangles': sum(s['triangles'] for s in scenes),
              'max_model_vertices': max(s['vertices'] for s in scenes), 'quads': {k: int(p[k].sum()) for k in ('river_quad', 'mouth_quad', 'lake_quad')},
              'band_cells': int(p['band'].sum()), 'wet_cells': int(p['wet'].sum()), 'lake_ground_cells': int(p['lake_ground'].sum()),
              'chain_hexes': int(len(p['chain']['q'])), 'chain_mouths': p['chain']['mouths'], 'chain_roots': p['chain']['roots'],
              'uv_points_without_chart': p['uv_far'], 'river_surface': p['surface_stats'],
              'bank_capped_vertices': p['capped_vertices'], 'bank_edge_vertices_over_lf': p['bank_edge_vertices_over_lf'],
              'chain_ends': int(len(p['chain']['ends'])),'stale_moved': [s.relative_to(S).as_posix() for s in stale] if not keep_stale else [],
              'bundle_assets': struct.unpack_from('<I', bundled)[0], 'bundle_bytes': len(bundled), 'scenes': scenes}
        report['variants'].append(vr)
        if apply:
            pb.write_stage(S, pre, bundled, updated, new)
            for k in sorted(files):
                pb.write_atomic(S / k, files[k])
            if stale and not keep_stale:
                dest = A / 'terrain-repair/river-water-stale' / str(time.time_ns())
                for s in stale:
                    t = dest / s.relative_to(S)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(s), str(t))
        print(variant, f"{vr['models']} water models ({vr['river_models']} river, {vr['lake_models']} lake),",
              f"{vr['vertices']} vertices, {vr['triangles']} triangles, max {vr['max_model_vertices']} per model,",
              f"rerun removed {len(old)} roots, stale moved {len(vr['stale_moved'])},", 'applied' if apply else 'dry run', flush=True)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_name(report_path.name + '.tmp')
        tmp.write_text(json.dumps(report, indent=1), encoding='utf-8')
        tmp.replace(report_path)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--report', type=Path, default=A / 'analysis/river-water-props.json')
    ap.add_argument('--keep-stale', action='store_true', help='leave loose files of an earlier run in place')
    a = ap.parse_args()
    run(a.apply, a.report, a.keep_stale)


if __name__ == '__main__':
    main()

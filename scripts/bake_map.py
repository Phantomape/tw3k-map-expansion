"""Map bake: build map.hex in one pass from vanilla + the layout + the design + the rule steps.

The release path (BUILD.md step 4) is --diff: the layout is data/map-layout-diff (what the expanded map changes
relative to vanilla), and the bake checks that it reproduces the released map. Without --diff the bake reads the
older layout base L0:

    map.hex = bake(vanilla 3k_dlc07 map.hex, data/map-layout (L0, roads, stamps, lakes), design, rule steps,
                   geodata caches checked against data/map-layout/geodata.json)

analysis/map-bake-design.md section 4. Every step writes only its own fields (checked after each step); there is
no in-place edit and no reverse record: the bake reads no earlier map.hex and no step record, and a rerun is the
undo. The order:

   1 vanilla slice         vanilla cells at the native offset, region ids renumbered by name into the L0 list
   2 L0 layout             region, kind, impassable, slot, sprawl, bridge, river edges, ground, climate, AOI
                           on the override hexes (every canvas hex, the native hexes the layout changed)
   3 town / port stamps    slot nibble (byte 2 = 16 / 32, passable) and sprawl (stamps.json; design towns)
   4 lakes                 lakes.json (Lop Nur, Zaysan): add_inland_lake's lake hex
   5 roads                 roads.npz: road edge masks (frozen, user decision 2026-10-26), opened vanilla hexes
   6 landmark candidates   (--river-aoi landmarks) landmarks_map candidate hexes the rivers keep off
   7 rivers                apply_rivers.lay_rivers -> river regions, river hexes, banks, fords, river-channels.json
   8 ground types          apply_ground_types.compute
   9 forest ground         apply_forest_ground.compute (no record: nothing to fade back)
  10 Hehuang / Korea patch analysis/{hehuang,korea}-patch.npz rebuilt from this bake's map when missing or stale (their one writer)
  11 climate and wetland   apply_climate.compute
  12 shore kinds           apply_shore_kinds.plan
  13 landmark AOIs         landmarks_map.tag
  14 region border masks   map_hex.border_masks (refresh_border_masks)
  15 header                L0 header + the river regions (and palette entries) + the landmark AOIs

Then the checks: no unused bit, every region id valid, no rule-owned value left (climate 0), region
connectivity (region_connectivity.check_map on the written file), SDK areas.

  python bake_map.py --out X.hex [--design D] [--layout data/map-layout] [--river-aoi landmarks|none]
      (TW3K_WORKSPACE: the workspace whose analysis/ caches the rules read; TW3K_WR_RIVERS / TW3K_REVEALED_RIVERS /
      TW3K_TAMRALIPTI_PORT as for the rule scripts, all off by default)
Writes X.hex, X.hex.river-channels.json and X.hex.bake-report.json (atomic). In the workspace only the patch caches
(step 10, and after a --diff bake: TW3K_WORKSPACE/analysis/{hehuang,korea}-patch*.npz, rebuilt from this bake's map when
missing or stale) are written. So a design candidate is baked with TW3K_WORKSPACE set to its scratch workspace
(a copy of the workspace outside it), never the live one: otherwise the candidate's patch caches
replace the live workspace's.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True, help='map.hex to write (never the workspace map)')
    ap.add_argument('--design', type=Path, help='design file (default: TW3K_DESIGN or the workspace design)')
    ap.add_argument('--layout', type=Path, help='L0 folder (default data/map-layout)')
    ap.add_argument('--river-aoi', choices=('none', 'landmarks'), default='landmarks',
                    help="hexes the rivers keep off: 'landmarks' (default, design step 6) = the landmarks_map candidate "
                         "hexes on the pre-river map; 'none' = no keep-out (v18 was laid without an AOI file; on the v18 "
                         "inputs both give the same map)")
    ap.add_argument('--skip-geodata-check', action='store_true', help='do not compare the geodata caches with geodata.json')
    ap.add_argument('--diff', action='store_true',
                    help='release build: the map = vanilla + the layout diff (layout_diff.py; --layout default '
                         'data/map-layout-diff); no rule step runs; stops unless the result is the recorded map (l0_roundtrip); then '
                         'the patch caches are made current for it (ensure_patch_caches, as step 10)')
    # Verification controls (verify_map_bake.py): they put back what the bake takes out of the old in-place chain, to
    # show which differences against the old map each decision causes. Not for a release bake.
    ap.add_argument('--control', type=Path, help='control.npz of extract_map_layout.py --control-out')
    ap.add_argument('--control-ground', action='store_true', help='control: L0 ground as it was (forest ground and wetland of earlier rounds kept)')
    ap.add_argument('--control-keep-orphans', action='store_true', help='control: keep the 146 orphan sprawl hexes and the 6 orphan water port hexes')
    ap.add_argument('--control-forest-record', type=Path, help='control: give apply_forest_ground this record (its fade-back), as the old chain did')
    return ap.parse_args()


def workspace_mix(out, workspace, live):
    """The reason to refuse a bake whose output and workspace do not belong together, else None (2026-11-01). The bake
    writes the patch caches into TW3K_WORKSPACE/analysis (step 10 and after --diff), so a bake writing its map outside
    the live workspace while TW3K_WORKSPACE is the live one is a candidate or verification bake that would replace the
    live caches (it happened at 12:04 that day). A scratch workspace (a copy outside the live one) may bake anywhere; a bake
    for the live workspace writes inside it (BUILD.md: work/final-map/map.baked.hex)."""
    out, workspace, live = (Path(p).resolve() for p in (out, workspace, live))
    if workspace == live and not out.is_relative_to(live):
        return (f'--out {out} lies outside the live workspace {live}, but TW3K_WORKSPACE is the live one: this bake would '
                'rewrite the live patch caches. Bake a candidate or a check in a scratch workspace (a copy of the '
                'workspace, TW3K_WORKSPACE=<scratch>), or write a live bake inside the workspace.')
    return None


def main():
    args = parse()
    if args.design:
        os.environ['TW3K_DESIGN'] = str(args.design.resolve())
    import numpy as np
    import campaign_world as cw
    import map_hex
    import map_layers as ml
    out = args.out.resolve()
    if out == (cw.WORKSPACE / 'work/final-map/map.hex').resolve():
        raise SystemExit('--out is the workspace map: the bake writes elsewhere')
    why = workspace_mix(out, cw.WORKSPACE, cw._ROOT / 'work/all-directions-30')
    if why:
        raise SystemExit(why)
    t0 = time.time()
    if args.diff:
        bake_diff(args, cw, map_hex, ml, out)
        return
    res = bake(args, np, cw, map_hex, ml)
    m, channels_doc, report = res
    report['seconds'] = round(time.time() - t0, 1)
    report['peak_working_set_mb'] = peak_mb()
    # Written beside --out first and checked there: a map that fails the region connectivity check never takes the name.
    staging = out.with_name(out.name + '.unchecked')
    map_hex.write(m, staging)
    import region_connectivity
    region_connectivity.check_map(staging)
    os.replace(staging, out)
    import apply_rivers
    apply_rivers.write_text_atomic(apply_rivers.channels_path(out), json.dumps(channels_doc, separators=(',', ':')) + chr(10))
    report['out'] = {'map': str(out), 'sha256': ml.sha256(out), 'river_channels': str(apply_rivers.channels_path(out)),
                     'river_channels_sha256': ml.sha256(apply_rivers.channels_path(out))}
    ml.write_json_atomic(out.with_name(out.name + '.bake-report.json'), report)
    print(json.dumps({k: report[k] for k in ('out', 'steps', 'seconds', 'peak_working_set_mb')}, ensure_ascii=False, indent=1))


def bake_diff(args, cw, map_hex, ml, out):
    """Release path (layout_diff.py): vanilla + the stored hexes + the stored header; the written map must be the
    recorded one byte for byte (l0_roundtrip), else it never takes the name."""
    import shutil
    import layout_diff
    layout = (args.layout or cw._ROOT / 'data/map-layout-diff').resolve()
    m, man = layout_diff.build(layout, cw.VANILLA_MAP)
    staging = out.with_name(out.name + '.unchecked')
    map_hex.write(m, staging)
    got = ml.sha256(staging)
    if got != man['map_sha256']:
        staging.unlink(missing_ok=True)
        raise SystemExit(f'l0_roundtrip: baked map sha256 {got[:16]} is not the recorded {man["map_sha256"][:16]}')
    os.replace(staging, out)
    # The patch caches' writer on this path too (no step 10 runs here): made current for the map just written, from its
    # cells (the map checkpoint's patch_design then passes), in TW3K_WORKSPACE/analysis (a candidate: its scratch ws).
    import ensure_patch_caches
    patches = ensure_patch_caches.ensure(map_cells=m['cells'], map_names=m['land_regions'] + m['sea_regions'])
    import apply_rivers
    if (layout / layout_diff.CHANNELS).exists():
        shutil.copyfile(layout / layout_diff.CHANNELS, apply_rivers.channels_path(out))
    report = {'mode': 'diff', 'layout': str(layout), 'out': {'map': str(out), 'sha256': got}, 'vanilla_sha256': man['vanilla_sha256'],
              'counts': man['counts'], 'l0_roundtrip': 'pass', 'patch_caches': patches}
    ml.write_json_atomic(out.with_name(out.name + '.bake-report.json'), report)
    print(json.dumps({k: report[k] for k in ('out', 'l0_roundtrip', 'patch_caches')}, ensure_ascii=False, indent=1))


def peak_mb():
    """Peak working set of this process in MB (Windows GetProcessMemoryInfo), None elsewhere."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return round(pmc.PeakWorkingSetSize / 2 ** 20)
    except Exception:
        return None


class StepGuard:
    """Each step may change only the fields it declares (map_layers.FIELDS, plus the unused bits never)."""

    def __init__(self, ml, np):
        self.ml, self.np, self.rows = ml, np, []

    def check(self, name, before, after, allowed, extra=None):
        ml, np = self.ml, self.np
        changed = {}
        for k in ml.FIELDS:
            n = int((ml.get(before, k) != ml.get(after, k)).sum())
            if n:
                changed[k] = n
        bad = sorted(set(changed) - set(allowed))
        if bad:
            raise SystemExit(f'bake step {name} changed fields it does not own: {bad} ({changed})')
        if ml.unused_bits(after):
            raise SystemExit(f'bake step {name} set unused bits')
        row = {'step': name, 'changed_fields': changed}
        if extra:
            row.update(extra)
        self.rows.append(row)
        print(f'[bake] {name}: {changed}', flush=True)


def plan_river_ports(m, laid, V, design, np, map_hex):
    """River spec 6.7: add_ports' river ports (TW3K_TAMRALIPTI_PORT=1), planned on a first river pass. The stamp centre
    stays the design town; a vanilla port pattern (any rotation or mirror) is taken whose primaries and land port L
    are land of the region (pre-river map) and stay land in the first pass, whose W2 is a laid frontier river hex and
    whose W1 is a laid river hex or pre-river land with no slot (the town keep-out holds the river ~3 hexes off a
    town, a resource stamp's W1 lies 2 out: W1 is the one water hex the port adds). Preference: W1 already river,
    then the vanilla orientations, then the pattern order. m: the map before the rivers; laid: the first pass result.
    Returns ([plan], report rows); a plan holds the add_ports.apply_plan fields."""
    import add_ports as ap
    todo = sorted(r for r, (kinds, _) in ap.PORTS.items() if kinds == 'river')
    plans, rows = [], []
    if not todo:
        return plans, rows
    pats = ap.vanilla_patterns(V)
    c0 = m['cells']; names0 = m['land_regions'] + m['sea_regions']; ids0 = map_hex.region_ids(c0)
    c1 = laid['cells']; names1 = m['land_regions'] + laid['sea']; ids1 = map_hex.region_ids(c1)
    H, W = ids0.shape
    land0 = (c0[..., 0] & 3) != 1; slot0 = c0[..., 2] >> 4
    water1 = (c1[..., 0] & 3) == 1
    river1 = water1 & np.isin(ids1, [i for i, n in enumerate(names1) if n.startswith('tw3k_riv_sea_')])
    ford1 = (c1[..., 3] & 128) > 0
    for reg in todo:
        if reg not in names0:
            rows.append({'region': reg, 'port': None, 'reason': 'region not on the map'}); continue
        me = names0.index(reg)
        town = next(t for t in design['regions'] if t['region'] == reg)['town']
        rr, qq = np.nonzero((ids0 == me) & (slot0 == 1))
        old = sorted((int(q), int(r)) for q, r in zip(qq, rr))
        kind = 'city' if len(old) == 19 else 'resource' if len(old) == 7 else None
        if kind is None:
            rows.append({'region': reg, 'port': None, 'reason': f'stamp of {len(old)} primaries is not a no-port stamp'}); continue
        C = ap.to_cube(*town)
        inb = lambda h: 0 <= h[0] < W and 0 <= h[1] < H
        y0, x0 = max(0, town[1] - 8), max(0, town[0] - 8)
        riv_xy = [(int(y) + y0, int(x) + x0) for y, x in np.argwhere(river1[y0:town[1] + 9, x0:town[0] + 9] & ~ford1[y0:town[1] + 9, x0:town[0] + 9])]
        if not riv_xy:
            rows.append({'region': reg, 'port': None, 'reason': 'no laid river within 8 hexes of the town'}); continue
        best = None
        for rank, pset in enumerate((pats[kind], ap.symmetric(pats[kind]))):
            for pi, (prim, L, W1, W2) in enumerate(pset):
                cells_ = [ap.to_hex(ap.cadd(C, d)) for d in prim]
                lh, w1, w2 = (ap.to_hex(ap.cadd(C, d)) for d in (L, W1, W2))
                if not all(inb(h) for h in cells_ + [lh, w1, w2]):
                    continue
                if not all(land0[h[1], h[0]] and ids0[h[1], h[0]] == me and not water1[h[1], h[0]] for h in cells_ + [lh]):
                    continue
                ok_w = lambda h: (river1[h[1], h[0]] and not ford1[h[1], h[0]]) or (land0[h[1], h[0]] and ids0[h[1], h[0]] == me)
                if not (ok_w(w1) and ok_w(w2)):
                    continue
                # hexes to the laid river from the nearer water port (the inlet apply_rivers adds), cube distance
                gap = min(ap.cdist(ap.to_cube(*w), ap.to_cube(int(x), int(y))) for w in (w1, w2) for y, x in riv_xy) - 1
                if gap > 3:
                    continue
                w1_river = bool(river1[w1[1], w1[0]])
                key = (gap, rank, pi)
                if best is None or key < best[0]:
                    best = (key, {'region': reg, 'kind': kind, 'pattern_set': ('vanilla orientation', 'rotated / mirrored')[rank],
                                  'pattern': pi, 'primaries': sorted(cells_), 'land_port': list(lh), 'water_port': [list(w1), list(w2)],
                                  'w1_added': not w1_river, 'join_hexes': gap, 'old_primaries': old, 'taken_from_province': []})
        if best is None:
            rows.append({'region': reg, 'port': None, 'reason': 'no vanilla port pattern with W2 on the laid river at the town'}); continue
        plans.append(best[1])
        rows.append({'region': reg, 'port': {k: best[1][k] for k in ('land_port', 'water_port', 'join_hexes', 'pattern_set', 'pattern')}})
    return plans, rows


def bake(args, np, cw, map_hex, ml):
    layout = (args.layout or ml.LAYOUT_DIR).resolve()
    man = ml.read_json(layout / ml.MANIFEST)
    hdr = man['header']
    if tuple(man['grid']) != (cw.W, cw.H) or tuple(man['native_offset']) != (cw.PAD_LEFT, cw.PAD_BOTTOM):
        raise SystemExit(f'L0 grid {man["grid"]} / offset {man["native_offset"]} is not the design canvas')
    report = {'layout': str(layout), 'design': {'path': str(cw.DESIGN_PATH), 'sha256': ml.sha256(cw.DESIGN_PATH)},
              'workspace': str(cw.WORKSPACE), 'river_aoi': args.river_aoi,
              'switches': {k: os.environ.get(k, '0') for k in ('TW3K_WR_RIVERS', 'TW3K_REVEALED_RIVERS', 'TW3K_TAMRALIPTI_PORT')},
              'inputs': {f: ml.sha256(layout / f) for f in (ml.LAYOUT_NPZ, ml.MANIFEST, ml.ROADS_NPZ, ml.STAMPS, ml.LAKES, ml.GEODATA)}}
    # ---- geodata caches: present and as recorded
    geo = ml.read_json(layout / ml.GEODATA)['files']
    paths = ml.geodata_paths(cw.WORKSPACE, cw._ROOT)
    report['geodata'] = {}
    for k, rec in sorted(geo.items()):
        if k in ml.BAKE_WRITTEN or not ml.geodata_active(k):   # patch caches: written in step 10; the other korea setting
            continue
        p = paths[k]
        if not p.exists():
            raise SystemExit(f'geodata {k} missing: {p}')
        if not args.skip_geodata_check:
            sha = ml.sha256(p)
            if sha != rec['sha256']:
                raise SystemExit(f'geodata {k} ({p}) is not the recorded version: sha256 {sha[:16]} != {rec["sha256"][:16]}')
        report['geodata'][k] = str(p)
    for k in (k for k, *_ in ml.GEODATA_FILES if ml.geodata_active(k) and k not in geo and not args.skip_geodata_check):
        raise SystemExit(f'geodata {k} is not recorded in {ml.GEODATA} (the switch reads it; record it after review)')
    L = ml.load_npz(layout / ml.LAYOUT_NPZ)
    R = ml.load_npz(layout / ml.ROADS_NPZ)
    stamps = ml.read_json(layout / ml.STAMPS)['stamps']
    lakes = ml.read_json(layout / ml.LAKES)['lakes']
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    names = hdr['land_regions'] + hdr['sea_regions']
    H, W = cw.H, cw.W
    nat = np.zeros((H, W), bool); nat[cw.NATIVE_HEX_SLICE] = True
    guard = StepGuard(ml, np)

    # ---- 1 vanilla slice
    V = map_hex.read(cw.VANILLA_MAP)
    for k in ('land_ground_types', 'sea_ground_types', 'climates', 'attritions'):
        if V[k] != hdr[k]:
            raise SystemExit(f'vanilla {k} differs from the L0 header')
    if V['areas_of_interest'] != hdr['areas_of_interest']:
        raise SystemExit('the L0 AOI list is not vanilla\'s')
    cells = np.zeros((H, W, 16), np.uint8)
    cells[cw.NATIVE_HEX_SLICE] = V['cells']
    vn = ml.region_names(V); vid = map_hex.region_ids(V['cells'])
    lut = np.array([names.index(n) + 1 if n in names else 0 for n in vn] + [0], np.int32)
    vreg = np.zeros((H, W), np.int32)
    vreg[cw.NATIVE_HEX_SLICE] = lut[np.where((vid >= 0) & (vid < len(vn)), vid, len(vn))]
    ml.put(cells, 'region', vreg, nat)
    guard.rows.append({'step': '1 vanilla slice', 'native_hexes': int(nat.sum())})

    # ---- 2 L0 layout
    before = cells.copy()
    ov = L['override'] > 0
    control = []
    if args.control and (args.control_ground or args.control_keep_orphans):
        C = ml.load_npz(args.control)
        if args.control_ground:
            L['ground'] = C['ground_as_is']; control.append('ground as it was')
        if args.control_keep_orphans:
            L['sprawl'] = np.where((C['orphan'] > 0) | (C['orphan_port'] > 0), 1, L['sprawl']).astype(L['sprawl'].dtype)
            L['slot'] = np.where(C['orphan_port'] > 0, 2, L['slot']).astype(L['slot'].dtype); control.append('orphans kept')
    if args.control_forest_record:
        control.append(f'forest record {args.control_forest_record}')
    report['control'] = control
    for k in ('region', 'kind', 'impassable', 'slot', 'sprawl', 'bridge', 'river_edges', 'ground', 'climate', 'aoi'):
        ml.put(cells, k, L[k].astype(np.int64), ov)
    if (ml.get(cells, 'region')[~ov & nat] == 0).any():
        raise SystemExit('a vanilla hex outside L0 has a region the map no longer has')
    guard.check('2 L0 layout', before, cells, ('region', 'kind', 'impassable', 'slot', 'sprawl', 'bridge', 'river_edges', 'ground', 'climate', 'aoi'),
                {'override_hexes': int(ov.sum())})

    # ---- 3 town and port stamps
    before = cells.copy()
    by_region = {s['region']: s for s in stamps}
    towns = {t['region']: tuple(t['town']) for t in design['regions']}
    if sorted(by_region) != sorted(towns):
        raise SystemExit(f'stamps.json and the design hold other regions: {sorted(set(by_region) ^ set(towns))[:10]}')
    reg_now = ml.get(cells, 'region')
    kind_now = ml.get(cells, 'kind')
    stamp_mask = np.zeros((H, W), bool)
    for reg in sorted(by_region):
        s = by_region[reg]
        if tuple(s['town']) != towns[reg]:
            raise SystemExit(f'{reg}: design town {towns[reg]} is not the stamp town {s["town"]} (re-plan the stamp)')
        prim = [tuple(h) for h in s['primaries']] if 'primaries' in s else ml.shape_cells(tuple(s['town']), s['shape'])
        ports = ([tuple(s['land_port'])] if s.get('land_port') else []) + [tuple(h) for h in s.get('water_port', [])]
        me = names.index(reg) + 1
        for q, r in prim + ports[:1]:
            if reg_now[r, q] != me or kind_now[r, q] == 1:
                raise SystemExit(f'{reg}: stamp hex {(q, r)} is not land of the region')
        for q, r in ports[1:]:
            if kind_now[r, q] != 1:
                raise SystemExit(f'{reg}: water port hex {(q, r)} is not water at stamp time (river ports wait for the river spec)')
        for q, r in prim:
            cells[r, q, 2] = 16; cells[r, q, 3] |= 1; stamp_mask[r, q] = True
        for q, r in ports:
            cells[r, q, 2] = 32; cells[r, q, 3] |= 1; stamp_mask[r, q] = True
    guard.check('3 stamps', before, cells, ('slot', 'impassable', 'sprawl'), {'stamps': len(stamps), 'stamp_hexes': int(stamp_mask.sum())})

    # ---- 4 lakes
    before = cells.copy()
    lake_id = names.index('3k_main_sea_lake') + 1
    G = hdr['land_ground_types'] + hdr['sea_ground_types']
    n_lake = 0
    for lk in sorted(lakes, key=lambda x: x['name']):
        if lk['region'] != '3k_main_sea_lake' or lk['ground'] != 'sea_lake':
            raise SystemExit(f'lake {lk["name"]}: only sea_lake hexes of 3k_main_sea_lake')
        qs = np.array([h[0] for h in lk['hexes']]); rs = np.array([h[1] for h in lk['hexes']])
        m = np.zeros((H, W), bool); m[rs, qs] = True
        if (m & stamp_mask).any():
            raise SystemExit(f'lake {lk["name"]} on a town stamp')
        ml.put(cells, 'region', lake_id, m); ml.put(cells, 'kind', 1, m)
        cells[m, 2] = 8; cells[m, 3] = 0; cells[m, 4] = 0; cells[m, 5] &= 0xF0
        ml.put(cells, 'ground', G.index('sea_lake') + 1, m); ml.put(cells, 'aoi', 0, m)
        n_lake += int(m.sum())
    guard.check('4 lakes', before, cells, ('region', 'kind', 'impassable', 'slot', 'sprawl', 'road', 'bridge', 'river_edges', 'ground', 'aoi'),
                {'lake_hexes': n_lake})
    if (ml.get(cells, 'region') == 0).any():
        raise SystemExit(f'{int((ml.get(cells, "region") == 0).sum())} hexes without a region after the layout steps')

    # ---- 5 roads (frozen data)
    before = cells.copy()
    rq, rr = R['q'].astype(np.int64), R['r'].astype(np.int64)
    road = ml.get(cells, 'road'); road[rr, rq] = R['road']
    ml.put(cells, 'road', road)
    ml.put(cells, 'impassable', 0, _mask(np, H, W, R['opened_q'], R['opened_r']))
    guard.check('5 roads', before, cells, ('road', 'impassable'), {'road_hexes_set': int(len(rq)), 'opened': int(len(R['opened_q']))})

    m = {'major': hdr['major'], 'minor': hdr['minor'], 'opaque': bytes.fromhex(hdr['opaque']), 'game': hdr['game'],
         'campaign_map': hdr['campaign_map'], 'land_regions': list(hdr['land_regions']), 'sea_regions': list(hdr['sea_regions']),
         'land_ground_types': list(hdr['land_ground_types']), 'sea_ground_types': list(hdr['sea_ground_types']),
         'climates': list(hdr['climates']), 'attritions': list(hdr['attritions']), 'areas_of_interest': list(hdr['areas_of_interest']),
         'palettes': [bytes.fromhex(p) for p in hdr['palettes']], 'cells': cells}
    before = cells.copy()
    cells[..., 15] = map_hex.border_masks(cells)     # the rules read a consistent map (as every step's input had)
    guard.check('5b border masks (pre-river)', before, cells, ('border',))

    # ---- 6 landmark candidates (river keep-out)
    aoi = np.zeros((H, W), bool)
    if args.river_aoi == 'landmarks':
        aoi = landmark_candidates(m, np)
    guard.rows.append({'step': '6 landmark candidates', 'mode': args.river_aoi, 'hexes': int(aoi.sum())})

    # ---- 7 rivers
    import apply_rivers
    before = cells.copy()
    _wr, _rev = os.environ.get('TW3K_WR_RIVERS', '0') == '1', os.environ.get('TW3K_REVEALED_RIVERS', '0') == '1'
    rv = apply_rivers.lay_rivers(m, V, ports=None, wr=_wr, revealed_rivers=_rev, aoi=aoi)
    # river ports (river spec 6.7): planned on this pass, then the rivers are laid again with the port's two water
    # hexes kept as river water (apply_rivers' port mask: the town keep-out opens beside them, their slot bytes stay)
    port_plans, river_ports = plan_river_ports(m, rv, V, design, np, map_hex)
    port_mask = np.zeros((H, W), bool)
    if port_plans:
        for pl in port_plans:
            for q, r in pl['water_port']:
                port_mask[r, q] = True
        pc = cells[port_mask].copy(); pc[:, 2] = 32; pc[:, 3] |= 1
        rv = apply_rivers.lay_rivers(m, V, ports={'mask': port_mask, 'cells': pc}, wr=_wr, revealed_rivers=_rev, aoi=aoi)
    if rv['failed']:
        ml.write_json_atomic(args.out.with_name(args.out.name + '.rivers-report.json'), rv['report'])
        raise SystemExit('river step fails its checks (report beside --out)')
    m = dict(m, sea_regions=rv['sea'], cells=rv['cells'], palettes=rv['pal'])
    cells = m['cells']
    guard.check('7 rivers', before, cells, ('region', 'kind', 'impassable', 'road', 'bridge', 'river_edges', 'ground', 'climate', 'aoi', 'border')
                + (('slot', 'sprawl') if port_plans else ()),   # river ports: W1 / W2 slot bytes (checked below: port hexes only)
                {'river_regions': rv['river_regions'], 'river_hexes': rv['report']['river_hexes'], 'fords': rv['report']['fords']})
    if ((ml.get(before, 'slot') != ml.get(cells, 'slot')) & ~port_mask).any() or ((ml.get(before, 'sprawl') != ml.get(cells, 'sprawl')) & ~port_mask).any():
        raise SystemExit('the river step changed a slot or sprawl hex (other than a planned river port)')
    channels_doc = rv['channels_doc']

    # ---- 7.5 river port stamps (river spec 6.7): the port is picked on the river as laid; town and river stay
    import add_ports
    before = cells.copy()
    for pl in port_plans:
        q, r = pl['water_port'][0]
        if not ((cells[r, q, 0] & 3) == 1 and (cells[r, q, 2] >> 4) == 2):
            raise SystemExit(f"river port {pl['region']}: W1 {(q, r)} is not port water after the river step")
        add_ports.apply_plan(cells, pl, m['land_regions'] + m['sea_regions'])
    guard.check('7.5 river ports', before, cells, ('slot', 'sprawl'), {'ports': river_ports})

    # ---- 8 ground types
    import apply_ground_types
    before = cells.copy()
    gt = apply_ground_types.compute(m)
    cells = gt['after']; m['cells'] = cells
    record = None
    if args.control_forest_record:              # control: the old chain's forest record, rewritten as apply_ground_types did
        z = ml.load_npz(args.control_forest_record)
        record = (z['q'], z['r'], z['src'], z['dst'])
        apply_ground_types.rewrite_forest_record(record, gt['changed'], gt['g'], gt['after'], gt['G'])
    guard.check('8 ground types', before, cells, ('ground',), {'target_hexes': int(gt['target'].sum())})

    # ---- 9 forest ground
    import apply_forest_ground
    before = cells.copy()
    fo = apply_forest_ground.compute(m, record=record)
    cells = apply_forest_ground.set_ground(cells, fo['changed'], fo['new']); m['cells'] = cells
    guard.check('9 forest ground', before, cells, ('ground',), {'by_change': fo['counts']})

    # ---- 10 Hehuang / Korea patches: the caches' one writer (P6, coordinator 11-01). A cache missing or stale against this
    # bake's map (land / sea and regions are final here: step 12's beach / cliff stay land), the design, west-real-terrain
    # or its build rules is rebuilt from these cells (never from an earlier final map.hex); step 11 loads them against the
    # same cells. The pipeline's map checkpoint (patch_design) only checks.
    import ensure_patch_caches
    pc = ensure_patch_caches.ensure(map_cells=cells, map_names=m['land_regions'] + m['sea_regions'])
    guard.rows.append({'step': '10 hehuang / korea patch', 'mode': 'rebuilt from this bake when missing or stale', 'caches': pc})

    # ---- 11 climate and wetland
    import apply_climate
    before = cells.copy()
    cl = apply_climate.compute(m)
    cells = cl['after_cells']; m['cells'] = cells
    left = (cells[..., 7] == 0)
    if left.any():
        raise SystemExit(f'{int(left.sum())} hexes keep the rule-owned climate placeholder (0) after the climate step')
    guard.check('11 climate', before, cells, ('climate', 'ground'), {'target_hexes': int(cl['target'].sum()), 'wetland_hexes': int(cl['wet'].sum())})

    # ---- 12 shore kinds
    import apply_shore_kinds
    from revealed_filler import filler_hex
    before = cells.copy()
    with np.load(cw.WORKSPACE / 'analysis/west-real-terrain.npz') as z:
        elev = np.array(z['elevation'])
    cells, changed, srep = apply_shore_kinds.plan(m, elev)
    del elev
    apply_shore_kinds.check(m['cells'], cells, filler_hex(m['cells'], ml.region_names(m)))
    m['cells'] = cells
    guard.check('12 shore kinds', before, cells, ('kind',), {'marked_beach': srep['marked_beach'], 'marked_cliff': srep['marked_cliff']})

    # ---- 13 landmark AOIs
    import landmarks_map
    before = cells.copy()
    m, lrep, lsum = landmarks_map.tag(m)
    cells = m['cells']
    guard.check('13 landmarks', before, cells, ('aoi',), {'aoi_list': lsum['list_length'], 'changed_cells': lsum['changed_cells']})

    # ---- 14 region border masks
    before = cells.copy()
    cells[..., 15] = map_hex.border_masks(cells)
    guard.check('14 border masks', before, cells, ('border',))

    # ---- 15 header and final checks
    ids = map_hex.region_ids(cells)
    n = len(m['land_regions']) + len(m['sea_regions'])
    if (ids < 0).any() or (ids >= n).any():
        raise SystemExit('region ids out of range')
    if ml.unused_bits(cells):
        raise SystemExit('unused bits set')
    from hex_components import region_areas, SDK_AREA_LIMIT
    areas = region_areas(ids, cells)
    if max(areas.values()) > SDK_AREA_LIMIT:
        raise SystemExit(f'SDK areas over the limit: {max(areas.values())}')
    report['steps'] = guard.rows
    report['regions'] = {'land': len(m['land_regions']), 'sea': len(m['sea_regions']), 'max_sdk_areas': int(max(areas.values()))}
    return m, channels_doc, report


def _mask(np, H, W, q, r):
    m = np.zeros((H, W), bool)
    m[np.asarray(r, np.int64), np.asarray(q, np.int64)] = True
    return m


def landmark_candidates(m, np):
    """Hexes landmarks_map would tag on this (pre-river) map: every landmark's candidate window cells on land,
    off slots and off the vanilla AOIs (landmarks_map.candidate_cells, the same free mask as tag())."""
    import landmarks_map
    import landmarks_spec as spec
    c = m['cells']
    land = (c[..., 0] & 3) != 1
    slot = (c[..., 2] >> 4) > 0
    impassable = ((c[..., 2] >> 3) & 1) == 1
    taken = landmarks_map.aoi_index(c) >= 0
    out = np.zeros(land.shape, bool)
    for lm in spec.LANDMARKS:
        for key, (mask, q0, r0) in sorted(landmarks_map.candidate_cells(lm, land, land & ~slot & ~taken, impassable).items()):
            h, w = mask.shape
            sel = mask & land[r0:r0 + h, q0:q0 + w] & ~slot[r0:r0 + h, q0:q0 + w] & ~taken[r0:r0 + h, q0:q0 + w]
            rr, qq = np.nonzero(sel)
            out[rr + r0, qq + q0] = True
    return out


if __name__ == '__main__':
    sys.exit(main())
